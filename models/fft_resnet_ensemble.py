"""Frozen FFT + ResNet late-fusion classifier."""

import torch
import torch.nn as nn


class FFTResNetEnsemble(nn.Module):
    """Combine separately trained FFT and ResNet classifiers.

    The two input models must output one logit per sample.  Their parameters are
    frozen when the ensemble is created, so only ``fusion_head`` is optimized.
    """

    def __init__(self, fft_model, resnet_model, hidden_dim=16):
        super().__init__()
        self.fft_model = fft_model
        self.resnet_model = resnet_model
        self.fusion_head = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
        )
        self.freeze_backbones()

    def freeze_backbones(self):
        """Freeze both base models and keep BatchNorm/dropout deterministic."""
        for model in (self.fft_model, self.resnet_model):
            for parameter in model.parameters():
                parameter.requires_grad = False
            model.eval()

    def train(self, mode=True):
        """Keep frozen backbones in evaluation mode during head training."""
        super().train(mode)
        self.fft_model.eval()
        self.resnet_model.eval()
        return self

    def forward(self, fft_features, images):
        with torch.no_grad():
            fft_logits = self.fft_model(fft_features)
            resnet_logits = self.resnet_model(images)
        return self.fusion_head(torch.cat((fft_logits, resnet_logits), dim=1))
