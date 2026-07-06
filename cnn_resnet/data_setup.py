import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split

def set_seed(seed=42):
    """
    Congela a aleatoriedade de todas as bibliotecas para garantir a reprodutibilidade.
    """
    # 1. Congela o Python nativo
    random.seed(seed)
    
    # 2. Congela o NumPy 
    np.random.seed(seed)
    
    # 3. Congela o PyTorch (CPU)
    torch.manual_seed(seed)
    
    # 4. Congela o PyTorch (GPU/CUDA) 
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 
        
        # Força o cuDNN a ser determinístico (pode deixar o treino levemente mais lento, mas é 100% reproduzível)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class HistopathologicDataset(Dataset):
    """
    Dataset customizado para carregar as imagens do Kaggle Histopathologic Cancer Detection.
    """
    def __init__(self, dataframe, img_dir, transform=None):
        self.dataframe = dataframe
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # O Kaggle fornece os IDs sem a extensão, então adicionamos '.tif'
        img_name = self.dataframe.iloc[idx, 0] + '.tif'
        img_path = os.path.join(self.img_dir, img_name)
        
        # Abre a imagem e converte para RGB
        image = Image.open(img_path).convert('RGB')
        
        # Pega o rótulo da imagem (0 ou 1)
        label = int(self.dataframe.iloc[idx, 1])
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def create_dataloaders(data_dir, batch_size=64, num_workers=2, mode="full"):
    """
    Cria e retorna os DataLoaders de treino e validação.
    
    Args:
        data_dir (str): Caminho raiz onde estão a pasta 'train' e o 'train_labels.csv'.
        batch_size (int): Tamanho do lote.
        num_workers (int): Número de subprocessos para carregamento de dados.
        mode (str): "full" para usar todo o dataset, "proto" para usar apenas 5% (modo prototipagem).
        
    Returns:
        train_loader, val_loader: Dataloaders do PyTorch.
    """
    
    # 1. Caminhos
    train_dir = os.path.join(data_dir, 'train')
    csv_path = os.path.join(data_dir, 'train_labels.csv')

    # 2. Leitura do CSV
    df = pd.read_csv(csv_path)

    # 3. Tratamento de Prototipagem (se não tiver GPU e quiser testar rápido)
    if mode == "proto":
        print("Modo Prototipagem ativado: Reduzindo dataset para 5%...")
        df, _ = train_test_split(df, train_size=0.05, random_state=42, stratify=df['label'])

    # 4. Divisão Treino e Validação (90% / 10%)
    df_train, df_val = train_test_split(df, test_size=0.10, random_state=42, stratify=df['label'])
    
    # Reseta os índices para evitar problemas no DataLoader
    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    
    print(f"Total de imagens processadas -> Treino: {len(df_train)} | Validação: {len(df_val)}")

    # 5. Definição das Transformações
    CROP_SIZE = 64
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transforms = transforms.Compose([
        transforms.CenterCrop(CROP_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=90),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    val_transforms = transforms.Compose([
        transforms.CenterCrop(CROP_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # 6. Instanciação dos Datasets
    train_dataset = HistopathologicDataset(dataframe=df_train, img_dir=train_dir, transform=train_transforms)
    val_dataset = HistopathologicDataset(dataframe=df_val, img_dir=train_dir, transform=val_transforms)

    # 7. Criação dos DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader