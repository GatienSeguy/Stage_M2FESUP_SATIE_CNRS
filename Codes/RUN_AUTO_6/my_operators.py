"""
my_operators.py — Opérateurs custom pour problèmes inverses.

Chaque opérateur hérite de LinearOperator (operators_torch.py) et implémente :
    - forward(x)          : A x       (entrée aplatie n, sortie aplatie m)
    - adjoint(y)          : A^T y     (entrée aplatie m, sortie aplatie n)
    - input_dim()         : n
    - output_dim()        : m
    - compute_AtA_diag()  : diag(A^T A), vecteur de taille n

IMPORTANT : forward et adjoint travaillent toujours sur des vecteurs 1D aplatis.
            Le reshape (C, H, W) se fait en interne.

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

    forward(x) = mask * x       (n → n, mêmes dimensions)
    adjoint(y) = mask * y       (n → n)
    AtA_diag   = mask           (0 ou 1 par composante)
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
        x = self._as_tensor(x).reshape(-1)
        return self.mask * x

    def adjoint(self, y):
        y = self._as_tensor(y).reshape(-1)
        return self.mask * y

    def input_dim(self):
        return self.n_channels * self.img_size ** 2

    def output_dim(self):
        return self.n_channels * self.img_size ** 2

    def compute_AtA_diag(self):
        return self.mask.clone()


class SuperResolutionOperator(LinearOperator):
    """Sous-échantillonnage par average pooling (facteur k).

    forward(x) : vecteur n → vecteur m   (average pooling k×k par canal)
    adjoint(y) : vecteur m → vecteur n   (réplication uniforme / k²)

    Avec n = C*H*W et m = C*(H/k)*(W/k).

    Vérification adjoint : <Ax, y> = <x, A^T y> pour tout x, y.
    forward : x_small[c, i, j] = mean(x[c, i*k:(i+1)*k, j*k:(j+1)*k])
            = (1/k²) * sum(x[c, i*k:(i+1)*k, j*k:(j+1)*k])
    adjoint : (A^T y)[c, i*k+di, j*k+dj] = y[c, i, j] / k²  pour di, dj in [0, k)

    Preuve : <Ax, y> = sum_c sum_ij x_small[c,i,j] * y[c,i,j]
           = sum_c sum_ij (1/k²) sum_{di,dj} x[c, i*k+di, j*k+dj] * y[c,i,j]
           = sum_c sum_{i',j'} x[c,i',j'] * y[c, i'//k, j'//k] / k²
           = <x, A^T y>    ✓
    """

    def __init__(self, img_size=256, n_channels=3, factor=4,
                 device=None, dtype=torch.float32):
        self.device = device if device is not None else _default_device(dtype)
        self.dtype = dtype
        self.img_size = img_size
        self.n_channels = n_channels
        self.factor = factor
        self.small_size = img_size // factor

        assert img_size % factor == 0, \
            f"img_size ({img_size}) doit être divisible par factor ({factor})"

    def forward(self, x):
        """Average pooling : (C, H, W) → (C, H/k, W/k), aplati."""
        x = self._as_tensor(x).reshape(-1)
        k = self.factor
        C, H, W = self.n_channels, self.img_size, self.img_size
        S = self.small_size
        # (C, H, W) → (C, S, k, S, k) → mean sur axes (2, 4) → (C, S, S)
        x_img = x.reshape(C, S, k, S, k)
        x_small = x_img.mean(dim=(2, 4))  # (C, S, S)
        return x_small.reshape(-1)

    def adjoint(self, y):
        """Réplication : (C, H/k, W/k) → (C, H, W), chaque valeur / k²."""
        y = self._as_tensor(y).reshape(-1)
        k = self.factor
        C = self.n_channels
        S = self.small_size
        H = self.img_size
        # (C, S, S) → (C, S, 1, S, 1) → expand → (C, S, k, S, k) → (C, H, W)
        y_img = y.reshape(C, S, S)
        y_big = y_img.unsqueeze(2).unsqueeze(4)        # (C, S, 1, S, 1)
        y_big = y_big.expand(C, S, k, S, k)            # (C, S, k, S, k)
        y_big = y_big.reshape(C, H, H)                 # (C, H, W)
        return (y_big / (k * k)).reshape(-1)

    def input_dim(self):
        return self.n_channels * self.img_size ** 2

    def output_dim(self):
        return self.n_channels * self.small_size ** 2

    def compute_AtA_diag(self):
        # (A^T A)_ii = 1/k^4 pour tout i (uniforme)
        # Car A = (1/k²) * sum sur le bloc, A^T = (1/k²) * réplication
        # Donc A^T A = (1/k^4) * I_bloc (chaque pixel a le même poids)
        val = 1.0 / (self.factor ** 4)
        return torch.full((self.input_dim(),), val,
                          device=self.device, dtype=self.dtype)
