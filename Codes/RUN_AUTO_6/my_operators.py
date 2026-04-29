"""
my_operators.py — Opérateurs custom pour problèmes inverses.

Chaque opérateur hérite de LinearOperator (operators_torch.py) et implémente :
    - forward(x)          : A x
    - adjoint(y)          : A^T y
    - input_dim()         : dimension de x
    - output_dim()        : dimension de y
    - compute_AtA_diag()  : diag(A^T A)  (optionnel, surcharge pour efficacité)

Usage dans le JSON :
    "operator": {
        "type": "custom",
        "file": "my_operators.py",
        "class": "InpaintingOperator",
        "params": {"mask_ratio": 0.5}
    }
"""

import torch
from operators_torch import LinearOperator, _default_device


class InpaintingOperator(LinearOperator):
    """Masque aléatoire : supprime un pourcentage de pixels.

    forward(x) = mask * x
    adjoint(y) = mask * y
    AtA_diag   = mask  (0 ou 1 par pixel)
    """

    def __init__(self, img_size=256, n_channels=3, mask_ratio=0.5,
                 device=None, dtype=torch.float32, seed=42):
        self.device = device if device is not None else _default_device(dtype)
        self.dtype = dtype
        self.img_size = img_size
        self.n_channels = n_channels
        self.mask_ratio = mask_ratio

        # Masque reproductible
        gen = torch.Generator().manual_seed(seed)
        n = n_channels * img_size * img_size
        self.mask = (torch.rand(n, generator=gen) > mask_ratio).to(
            device=self.device, dtype=self.dtype
        )

    def forward(self, x):
        x = self._as_tensor(x)
        return self.mask * x

    def adjoint(self, y):
        y = self._as_tensor(y)
        return self.mask * y

    def input_dim(self):
        return self.n_channels * self.img_size ** 2

    def output_dim(self):
        return self.n_channels * self.img_size ** 2

    def compute_AtA_diag(self):
        return self.mask.clone()


class SuperResolutionOperator(LinearOperator):
    """Sous-échantillonnage par average pooling (facteur k).

    forward(x) : image H×W → image (H/k)×(W/k) par moyenne de blocs k×k
    adjoint(y) : image (H/k)×(W/k) → image H×W par réplication / k²

    Note : output_dim < input_dim (m < n).
    """

    def __init__(self, img_size=256, n_channels=3, factor=4,
                 device=None, dtype=torch.float32):
        self.device = device if device is not None else _default_device(dtype)
        self.dtype = dtype
        self.img_size = img_size
        self.n_channels = n_channels
        self.factor = factor
        self.small_size = img_size // factor

        assert img_size % factor == 0, f"img_size ({img_size}) doit être divisible par factor ({factor})"

    def forward(self, x):
        x = self._as_tensor(x)
        k = self.factor
        C, H, W = self.n_channels, self.img_size, self.img_size
        x = x.reshape(C, H, W)
        # Average pooling : (C, H, W) → (C, H/k, W/k)
        x_small = x.reshape(C, self.small_size, k, self.small_size, k).mean(dim=(2, 4))
        return x_small.reshape(-1)

    def adjoint(self, y):
        y = self._as_tensor(y)
        k = self.factor
        C = self.n_channels
        y = y.reshape(C, self.small_size, self.small_size)
        # Réplication : (C, H/k, W/k) → (C, H, W), divisé par k²
        y_big = y.unsqueeze(3).unsqueeze(5)
        y_big = y_big.expand(C, self.small_size, 1, k, self.small_size, 1)
        # Reshape proprement
        y_big = y.reshape(C, self.small_size, 1, self.small_size, 1).expand(
            C, self.small_size, k, self.small_size, k
        ).reshape(C, self.img_size, self.img_size)
        return (y_big / (k * k)).reshape(-1)

    def input_dim(self):
        return self.n_channels * self.img_size ** 2

    def output_dim(self):
        return self.n_channels * self.small_size ** 2

    def compute_AtA_diag(self):
        # Chaque pixel contribue 1/k² à sa cellule, et l'adjoint réplique avec 1/k²
        # Donc (A^T A)_ii = 1/k^4 pour tout i (uniforme)
        val = 1.0 / (self.factor ** 4)
        return torch.full((self.input_dim(),), val,
                          device=self.device, dtype=self.dtype)
