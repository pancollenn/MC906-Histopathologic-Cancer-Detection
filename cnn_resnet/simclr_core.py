import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. TRANSFORMAÇÃO DE DUPLA VISÃO (TWO CROP)
# ==========================================
class TwoCropTransform:
    """
    Pega uma imagem e aplica o pipeline de transformações (Data Augmentation) duas vezes.
    Retorna duas versões diferentes da mesma imagem (x_i e x_j).
    """
    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return q, k

# ==========================================
# 2. FUNÇÃO DE PERDA (NT-Xent / InfoNCE)
# ==========================================
class NTXentLoss(nn.Module):
    """
    Loss Contrastiva do SimCLR.
    Aproxima as representações das duas versões da mesma imagem (positivos)
    e as afasta das representações de todas as outras imagens do batch (negativos).
    """
    def __init__(self, temperature=0.5):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        batch_size = z_i.size(0)
        
        # Junta z_i e z_j em um único tensor [2*batch_size, dim]
        z = torch.cat([z_i, z_j], dim=0)
        z = F.normalize(z, dim=1) # Normaliza os vetores (L2)

        # Calcula a similaridade do cosseno entre todos os vetores do batch
        # Resultado é uma matriz [2*batch_size, 2*batch_size]
        sim_matrix = torch.matmul(z, z.T) / self.temperature

        # Cria os rótulos: o par positivo de z_i[k] é z_j[k] (que está no índice k + batch_size)
        labels = torch.cat([
            torch.arange(batch_size) + batch_size, 
            torch.arange(batch_size)
        ], dim=0).to(z.device)

        # Mascara a diagonal principal (não queremos comparar a imagem com ela mesma)
        mask = torch.eye(2 * batch_size, dtype=torch.bool).to(z.device)
        sim_matrix.masked_fill_(mask, -9e15)

        # A loss InfoNCE é essencialmente uma Cross Entropy sobre a matriz de similaridade
        loss = F.cross_entropy(sim_matrix, labels)
        return loss

# ==========================================
# 3. WRAPPER DO MODELO (Encoder + Projection Head)
# ==========================================
class SimCLRModel(nn.Module):
    """
    Envolve o codificador base (ResNet ou CNN) com um Projection Head.
    No SimCLR, a loss é calculada no Projection Head, mas para a tarefa final,
    usamos as representações do Encoder.
    """
    def __init__(self, base_encoder, feature_dim=512, projection_dim=128):
        super(SimCLRModel, self).__init__()
        self.encoder = base_encoder
        
        # Projection Head (Um pequeno MLP de 2 camadas)
        # O SimCLR provou que adicionar isso melhora a representação do encoder
        self.projector = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, projection_dim)
        )

    def forward(self, x):
        # 1. Extrai as características da imagem (Representação h)
        features = self.encoder(x)
        
        # 2. Mapeia para o espaço de projeção (Representação z para a loss)
        projections = self.projector(features)
        
        return features, projections