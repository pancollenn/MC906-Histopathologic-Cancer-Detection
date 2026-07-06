import os
import torch
import numpy as np
import torchvision
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve, accuracy_score
from tqdm import tqdm

def visualizar_batch(dataloader, num_imagens=16):
    """
    Extrai um batch do dataloader, desfaz a normalização e plota as imagens.
    """
    images, labels = next(iter(dataloader))
    
    images = images[:num_imagens]
    labels = labels[:num_imagens]
    
    grid = torchvision.utils.make_grid(images, nrow=4, padding=2)
    
    np_grid = grid.numpy()
    np_grid = np.transpose(np_grid, (1, 2, 0))
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_grid = std * np_grid + mean
    np_grid = np.clip(np_grid, 0, 1)
    
    plt.figure(figsize=(8, 8))
    plt.imshow(np_grid)
    plt.axis('off')
    plt.title('Amostra do DataLoader', fontsize=16)
    plt.show()
    
    nomes_classes = {0: 'Normal (0)', 1: 'Tumor (1)'}
    print("Rótulos correspondentes ao Grid (Esquerda para Direita):")
    
    labels_matriz = labels.view(-1, 4).numpy() 
    for linha in labels_matriz:
        nomes = [nomes_classes[lbl] for lbl in linha]
        print(f"{nomes[0]:<15} | {nomes[1]:<15} | {nomes[2]:<15} | {nomes[3]:<15}")


def plotar_historico(historico, save_dir=None, prefix=""):
    """
    Plota os gráficos de Loss e Acurácia/AUC e salva em disco se save_dir for informado.
    """
    epocas = range(1, len(historico['train_loss']) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epocas, historico['train_loss'], 'b-', label='Treino Loss')
    ax1.plot(epocas, historico['val_loss'], 'r-', label='Validação Loss')
    ax1.set_title('Histórico de Perda (Loss)')
    ax1.set_xlabel('Épocas')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epocas, historico['train_acc'], 'b--', label='Treino Acc')
    ax2.plot(epocas, historico['val_acc'], 'r--', label='Validação Acc')
    ax2.plot(epocas, historico['val_auc'], 'g-', linewidth=2, label='Validação AUC')
    ax2.set_title('Histórico de Acurácia e AUC-ROC')
    ax2.set_xlabel('Épocas')
    ax2.set_ylabel('Pontuação')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    
    # Lógica de Salvamento
    if save_dir:
        os.makedirs(save_dir, exist_ok=True) # Cria a pasta se não existir
        caminho_arquivo = os.path.join(save_dir, f"{prefix}historico.png")
        plt.savefig(caminho_arquivo, dpi=300, bbox_inches='tight') # dpi=300 para alta resolução
        print(f"Gráfico de histórico salvo em: {caminho_arquivo}")
        
    plt.show()


def avaliar_modelo(model, dataloader, criterion, device, save_dir=None, prefix=""):
    """
    Avalia o modelo, plota a Matriz de Confusão/Curva ROC e salva em disco.
    """
    print("Iniciando avaliação...")
    model.eval()
    
    loss_acumulada = 0.0
    todas_probabilidades = []
    todas_previsoes = []
    todos_labels = []
    
    with torch.no_grad():
        loop_aval = tqdm(dataloader, desc='Avaliando', leave=False)
        for images, labels in loop_aval:
            images = images.to(device)
            labels = labels.to(device).float().unsqueeze(1)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            loss_acumulada += loss.item() * images.size(0)
            
            probs = torch.sigmoid(outputs).cpu().numpy()
            preds = (probs >= 0.5).astype(float)
            
            todas_probabilidades.extend(probs)
            todas_previsoes.extend(preds)
            todos_labels.extend(labels.cpu().numpy())

    todos_labels = np.array(todos_labels).flatten()
    todas_probabilidades = np.array(todas_probabilidades).flatten()
    todas_previsoes = np.array(todas_previsoes).flatten()
    
    loss_media = loss_acumulada / len(todos_labels)
    acuracia = accuracy_score(todos_labels, todas_previsoes)
    roc_auc = roc_auc_score(todos_labels, todas_probabilidades)
    
    print("\n" + "="*40)
    print("🩺 RESULTADOS DA AVALIAÇÃO")
    print("="*40)
    print(f"Loss Média: {loss_media:.4f}")
    print(f"Acurácia:   {acuracia:.4f} ({acuracia*100:.2f}%)")
    print(f"ROC-AUC:    {roc_auc:.4f}")
    print("="*40)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    cm = confusion_matrix(todos_labels, todas_previsoes)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0], 
                xticklabels=['Normal (0)', 'Tumor (1)'], 
                yticklabels=['Normal (0)', 'Tumor (1)'])
    axes[0].set_title('Matriz de Confusão', fontsize=14)
    axes[0].set_xlabel('Previsão do Modelo', fontsize=12)
    axes[0].set_ylabel('Rótulo Verdadeiro', fontsize=12)
    
    fpr, tpr, thresholds = roc_curve(todos_labels, todas_probabilidades)
    axes[1].plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (área = {roc_auc:.4f})')
    axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('Taxa de Falsos Positivos (FPR)', fontsize=12)
    axes[1].set_ylabel('Taxa de Verdadeiros Positivos (TPR)', fontsize=12)
    axes[1].set_title('Característica de Operação do Receptor (ROC)', fontsize=14)
    axes[1].legend(loc="lower right")
    
    plt.tight_layout()
    
    # Lógica de Salvamento
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        caminho_arquivo = os.path.join(save_dir, f"{prefix}avaliacao.png")
        plt.savefig(caminho_arquivo, dpi=300, bbox_inches='tight')
        print(f"Gráfico de avaliação salvo em: {caminho_arquivo}")
        
    plt.show()

    return {'loss': loss_media, 'accuracy': acuracia, 'roc_auc': roc_auc}