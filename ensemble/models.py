import torch
import torch.nn as nn


def _default_device(device=None):
    """Use CUDA when available unless the caller explicitly chooses a device."""
    return torch.device(device) if device is not None else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


class ImageFeatureDenseModel(nn.Module):
    """Simple dense network for 16 statistics from each of the three RGB channels."""

    def __init__(self, input_dim=16 * 3, device=None):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.to(_default_device(device))

    def forward(self, x):
        return self.network(x)


class FFTFeatureDenseModel(nn.Module):
    """Simple dense network for 26 frequency descriptors from each RGB channel."""

    def __init__(self, input_dim=26 * 3, device=None):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.to(_default_device(device))

    def forward(self, x):
        return self.network(x)
