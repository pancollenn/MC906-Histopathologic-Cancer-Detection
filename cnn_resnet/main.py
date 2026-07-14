import os

import torch
import torch.nn as nn
import torch.optim as optim
import kagglehub

# Importação dos módulos locais que criamos
import data_setup
import model_builder
import engine
import evaluate


def main():
    # ==========================================
    #  CONFIGURAÇÕES DO PROJETO
    # ==========================================
    
    DATA_DIR = kagglehub.competition_download('histopathologic-cancer-detection')
    print(f"Caminho do Dataset: {DATA_DIR}")
    
    MODEL_TYPE = 'resnet'  # Escolha entre: 'baseline' ou 'resnet'
    MODE = 'full'           # Escolha entre: 'full' (100% dos dados) ou 'proto' (5% dos dados para teste)
    
    NUM_EPOCHS = 10
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    NUM_WORKERS = os.cpu_count() or 1
    PLOT_EXAMPLES = False

    data_setup.set_seed(42)  

    # ==========================================
    #  CONFIGURAÇÃO DO DISPOSITIVO
    # ==========================================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("="*50)
    print(f"INICIANDO PROJETO | Dispositivo: {device}")
    print(f"Modelo: {MODEL_TYPE.upper()} | Modo: {MODE.upper()} | Épocas: {NUM_EPOCHS}")
    print("="*50)

    # ==========================================
    #  PREPARAÇÃO DOS DADOS
    # ==========================================
    print("\nCarregando os dados...")
    train_loader, val_loader = data_setup.create_dataloaders(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
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
    PREFIXO = f"{MODEL_TYPE}_{MODE}_"
    
    # Agora as funções criarão a pasta e salvarão as imagens em alta qualidade
    evaluate.plotar_historico(historico, save_dir=PASTA_GRAFICOS, prefix=PREFIXO)
    metricas = evaluate.avaliar_modelo(
        modelo_treinado,
        val_loader,
        criterion,
        device,
        save_dir=PASTA_GRAFICOS,
        prefix=PREFIXO,
        plot_examples=PLOT_EXAMPLES,
    )

    nome_arquivo = f"modelo_{MODEL_TYPE}_{MODE}.pth"
    torch.save(modelo_treinado.state_dict(), nome_arquivo)
    print(f"\nModelo salvo com sucesso: {nome_arquivo}")

if __name__ == '__main__':
    main()
