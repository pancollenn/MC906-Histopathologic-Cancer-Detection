import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class BaselineCNN(nn.Module):
    """
    Uma Rede Neural Convolucional construída do zero.
    Projetada para receber imagens de entrada de tamanho 64x64.
    """
    def __init__(self):
        super(BaselineCNN, self).__init__()
        
        # Extrator de Características (Feature Extractor)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        
        # Camada de redução espacial
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Classificador (Fully Connected)
        # Após 4 reduções, a imagem 64x64 vira 4x4
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 1) # Saída de 1 neurônio (Câncer ou Normal)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        
        x = torch.flatten(x, 1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


def get_resnet18_model(pretrained=True, fine_tune=True):
    """
    Importa o modelo ResNet18 do torchvision para Transfer Learning.
    
    Args:
        pretrained (bool): Se True, baixa os pesos treinados no ImageNet.
        fine_tune (bool): Se True, treina toda a rede. Se False, congela
                          as camadas iniciais e treina apenas o classificador final.
    """
    # Carrega a rede com ou sem os pesos pré-treinados
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    
    # Congela os pesos se não formos fazer fine-tuning completo
    if not fine_tune:
        for param in model.parameters():
            param.requires_grad = False
            
    # O PyTorch já inclui um AdaptiveAvgPool2d na ResNet, então não
    # precisamos nos preocupar com o tamanho 64x64 da imagem de entrada.
    
    # Substitui a última camada (que originalmente classifica 1000 classes do ImageNet)
    # por uma camada de saída com apenas 1 neurônio
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1)
    
    return model