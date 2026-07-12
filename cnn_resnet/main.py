import torch
import torch.nn as nn
import torch.optim as optim
import kagglehub

# Importação dos módulos locais que criamos
import data_setup
import model_builder
import engine
import evaluate

# Importação do novo módulo SSL
from simclr_core import TwoCropTransform, NTXentLoss, SimCLRModel
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys
import argparse

def main(model_type='baseline', mode='full', ssl=False, num_epochs=10, epochs_ssl=5, batch_size=1024, learning_rate=0.001):
    # ==========================================
    #  CONFIGURAÇÕES DO PROJETO
    # ==========================================
    
    DATA_DIR = kagglehub.competition_download('histopathologic-cancer-detection')
    print(f"Caminho do Dataset: {DATA_DIR}")
    
    MODEL_TYPE = model_type  # Escolha entre: 'baseline' ou 'resnet'
    MODE = mode           # Escolha entre: 'full' (100% dos dados) ou 'proto' (5% dos dados para teste)
    SSL = ssl              # Ativa ou desativa o pré-treinamento SimCLR
    
    NUM_EPOCHS = num_epochs
    EPOCHS_SSL = epochs_ssl          # Épocas dedicadas ao pré-treinamento SSL
    BATCH_SIZE = batch_size
    LEARNING_RATE = learning_rate

    data_setup.set_seed(42)  

    # ==========================================
    #  CONFIGURAÇÃO DO DISPOSITIVO
    # ==========================================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("="*50)
    print(f"INICIANDO PROJETO | Dispositivo: {device}")
    print(f"Modelo: {MODEL_TYPE.upper()} | Modo: {MODE.upper()} | Épocas: {NUM_EPOCHS} | SSL: {SSL}")
    print("="*50)

    # ==========================================
    #  PREPARAÇÃO DOS DADOS
    # ==========================================
    print("\nCarregando os dados...")
    train_loader, val_loader = data_setup.create_dataloaders(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
        num_workers=6,
        mode=MODE
    )

    # Visualizar os dados antes do treino
    # evaluate.visualizar_batch(train_loader)

    # ==========================================
    #  CONSTRUÇÃO DO MODELO
    # ==========================================
    print(f"\nConstruindo modelo {MODEL_TYPE}...")
    if MODEL_TYPE == 'baseline':
        model = model_builder.BaselineCNN().to(device)
    elif MODEL_TYPE == 'resnet':
        # Carrega a ResNet com pesos do ImageNet e permite treinar toda a rede
        model = model_builder.get_resnet18_model(pretrained=True, fine_tune=True).to(device)
    else:
        raise ValueError("Modelo inválido! Escolha 'baseline' ou 'resnet'.")

    # ==========================================
    #  FASE SSL (OPCIONAL)
    # ==========================================
    if SSL:
        print("\n>>> INICIANDO FASE SSL: Pré-treinamento SimCLR <<<")
        
        # Salva o transform original para restaurar depois
        dataset_treino = train_loader.dataset
        transform_original = dataset_treino.transform
        
        # Define e aplica o transform do SimCLR
        color_jitter = transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
        base_ssl_transform = transforms.Compose([
            transforms.RandomResizedCrop(64, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        dataset_treino.transform = TwoCropTransform(base_ssl_transform)
        
        # Loader exclusivo para SSL (sempre embaralhado e adaptado)
        ssl_loader = DataLoader(dataset_treino, 
                                batch_size=BATCH_SIZE, 
                                shuffle=True, 
                                num_workers=4, 
                                pin_memory=True, 
                                drop_last=True)
        
        # Substitui a última camada pela identidade temporariamente para virar apenas um extrator
        if MODEL_TYPE == 'baseline':
            camada_final_original = model.fc2
            model.fc2 = nn.Identity()
        else:
            camada_final_original = model.fc
            model.fc = nn.Identity()
            
        feature_dim = camada_final_original.in_features if hasattr(camada_final_original, 'in_features') else 512
        
        simclr_model = SimCLRModel(model, feature_dim=feature_dim).to(device)
        criterion_ssl = NTXentLoss(temperature=0.5).to(device)
        optimizer_ssl = optim.Adam(simclr_model.parameters(), lr=1e-3, weight_decay=1e-4)

        simclr_model.train()
        for epoch in range(EPOCHS_SSL):
            epoch_loss = 0.0
            loop = tqdm(ssl_loader, leave=False, desc=f"SSL Época {epoch+1}/{EPOCHS_SSL}")
            for images, _ in loop:
                view1, view2 = images[0].to(device), images[1].to(device)
                
                optimizer_ssl.zero_grad()
                _, proj1 = simclr_model(view1)
                _, proj2 = simclr_model(view2)
                
                loss = criterion_ssl(proj1, proj2)
                loss.backward()
                optimizer_ssl.step()
                
                epoch_loss += loss.item()
                loop.set_postfix(loss=loss.item())
            
            print(f"Época SSL {epoch+1:02d}/{EPOCHS_SSL} | Loss SimCLR: {epoch_loss/len(ssl_loader):.4f}")
        
        # Restaura a rede para o formato padrão e os transforms originais (Magia acontece aqui)
        if MODEL_TYPE == 'baseline':
            model.fc2 = camada_final_original
        else:
            model.fc = camada_final_original
        dataset_treino.transform = transform_original
        
        print(">>> FASE SSL CONCLUÍDA. Iniciando Fine-Tuning supervisionado... <<<\n")

    # Define a Função de Perda e o Otimizador
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # ==========================================
    #  TREINAMENTO
    # ==========================================
    print("\nIniciando motor de treinamento...")
    modelo_treinado, historico = engine.treinar_modelo(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=NUM_EPOCHS
    )

    # ==========================================
    #  AVALIAÇÃO E SALVAMENTO
    # ==========================================
    print("\nGerando gráficos e métricas...")
    
    # Configurando o destino dos gráficos
    PASTA_GRAFICOS = "cnn_resnet/plots"
    PREFIXO = f"{MODEL_TYPE}_{MODE}_ssl_" if SSL else f"{MODEL_TYPE}_{MODE}_"
    
    # Agora as funções criarão a pasta e salvarão as imagens em alta qualidade
    evaluate.plotar_historico(historico, save_dir=PASTA_GRAFICOS, prefix=PREFIXO)
    metricas = evaluate.avaliar_modelo(modelo_treinado, val_loader, criterion, device, save_dir=PASTA_GRAFICOS, prefix=PREFIXO)

    NOME_MODELO = f"modelo_{MODEL_TYPE}_{MODE}_ssl.pth" if SSL else f"modelo_{MODEL_TYPE}_{MODE}.pth"
    torch.save(modelo_treinado.state_dict(), NOME_MODELO)
    print(f"\nModelo salvo com sucesso: {NOME_MODELO}")

if __name__ == "__main__":
    # Pega os argumentos do terminal (ou usa os padrões se não forem passados)
    # sys.argv[0] é sempre o nome do arquivo ('main.py')
    
    arg_model_type = sys.argv[1] if len(sys.argv) > 1 else 'baseline'
    arg_mode = sys.argv[2] if len(sys.argv) > 2 else 'full'
    
    # Converte corretamente a string 'True' ou 'True/'true' do terminal para Booleano real
    arg_ssl = sys.argv[3].lower() == 'true' if len(sys.argv) > 3 else False
    
    # Executa a main passando o que veio do terminal
    main(model_type=arg_model_type, mode=arg_mode, ssl=arg_ssl)