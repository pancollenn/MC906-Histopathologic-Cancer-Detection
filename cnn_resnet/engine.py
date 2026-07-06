import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

def treinar_modelo(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=10):
    """
    Função genérica para treinar e validar modelos PyTorch.
    
    Args:
        model: O modelo de rede neural a ser treinado.
        train_loader: DataLoader contendo os dados de treinamento.
        val_loader: DataLoader contendo os dados de validação.
        criterion: Função de perda (ex: BCEWithLogitsLoss).
        optimizer: Otimizador (ex: Adam).
        device: 'cuda' ou 'cpu'.
        num_epochs (int): Quantidade de épocas de treinamento.
        
    Returns:
        model: O modelo treinado.
        historico (dict): Dicionário contendo o histórico de loss, acurácia e AUC-ROC.
    """
    
    historico = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'val_auc': []
    }

    print(f"Iniciando o treinamento no dispositivo: {device}")

    for epoch in range(num_epochs):
        # ==========================================
        #               FASE DE TREINO
        # ==========================================
        model.train()
        
        train_loss_acumulada = 0.0
        train_corretos = 0
        train_total = 0
        
        loop_treino = tqdm(train_loader, desc=f'Época {epoch+1}/{num_epochs} [Treino]', leave=False)
        
        for images, labels in loop_treino:
            images = images.to(device)
            # Formata os labels para bater com a saída de 1 neurônio e converte para float
            labels = labels.to(device).float().unsqueeze(1)
            
            # Zera os gradientes
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass e otimização
            loss.backward()
            optimizer.step()
            
            # Acúmulo de métricas
            train_loss_acumulada += loss.item() * images.size(0)
            
            # Transforma os logits em probabilidades (0 a 1) e extrai previsões binárias
            probabilidades = torch.sigmoid(outputs)
            previsoes = (probabilidades >= 0.5).float()
            
            train_corretos += (previsoes == labels).sum().item()
            train_total += labels.size(0)
            
            # Atualiza barra de progresso visual
            loop_treino.set_postfix(loss=loss.item())

        epoca_train_loss = train_loss_acumulada / train_total
        epoca_train_acc = train_corretos / train_total
        
        historico['train_loss'].append(epoca_train_loss)
        historico['train_acc'].append(epoca_train_acc)

        # ==========================================
        #              FASE DE VALIDAÇÃO
        # ==========================================
        model.eval()
        
        val_loss_acumulada = 0.0
        val_corretos = 0
        val_total = 0
        
        todas_probabilidades_val = []
        todos_labels_val = []
        
        with torch.no_grad():
            loop_val = tqdm(val_loader, desc=f'Época {epoch+1}/{num_epochs} [Validação]', leave=False)
            for images, labels in loop_val:
                images = images.to(device)
                labels = labels.to(device).float().unsqueeze(1)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss_acumulada += loss.item() * images.size(0)
                
                probabilidades = torch.sigmoid(outputs)
                previsoes = (probabilidades >= 0.5).float()
                
                val_corretos += (previsoes == labels).sum().item()
                val_total += labels.size(0)
                
                # Salva para cálculo posterior da curva ROC
                todas_probabilidades_val.extend(probabilidades.cpu().numpy())
                todos_labels_val.extend(labels.cpu().numpy())

        epoca_val_loss = val_loss_acumulada / val_total
        epoca_val_acc = val_corretos / val_total
        epoca_val_auc = roc_auc_score(todos_labels_val, todas_probabilidades_val)
        
        historico['val_loss'].append(epoca_val_loss)
        historico['val_acc'].append(epoca_val_acc)
        historico['val_auc'].append(epoca_val_auc)

        # ==========================================
        #               RELATÓRIO DA ÉPOCA
        # ==========================================
        print(f"Época {epoch+1:02d}/{num_epochs} | "
              f"Train Loss: {epoca_train_loss:.4f} | Train Acc: {epoca_train_acc:.4f} || "
              f"Val Loss: {epoca_val_loss:.4f} | Val Acc: {epoca_val_acc:.4f} | Val AUC: {epoca_val_auc:.4f}")

    print("Treinamento concluído!")
    
    return model, historico