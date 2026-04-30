import torch
from operators_torch import LinearOperator, _default_device


class InpaintingOperator(LinearOperator):
    def __init__(self, img_size=256, n_channels=3, mask_type='random',
                 mask_ratio=0.5, device=None, dtype=torch.float32, seed=42):
        self.device = device if device is not None else _default_device(dtype)
        self.dtype = dtype
        self.img_size = img_size
        self.n_channels = n_channels

        H = W = img_size
        mask_2d = torch.ones(H, W, dtype=dtype)

        if mask_type == 'random':
            gen = torch.Generator().manual_seed(seed)
            mask_2d = (torch.rand(H, W, generator=gen) > mask_ratio).to(dtype)

        elif mask_type == 'box50':
            s = H // 2
            start = (H - s) // 2
            mask_2d[start:start+s, start:start+s] = 0

        elif mask_type == 'box25':
            s = H // 4
            start = (H - s) // 2
            mask_2d[start:start+s, start:start+s] = 0

        elif mask_type == 'half':
            mask_2d[:, W//2:] = 0

        elif mask_type == 'expand':
            # Inverse de box25 : tout masqué sauf le centre
            s = H // 4
            start = (H - s) // 2
            mask_2d[:, :] = 0
            mask_2d[start:start+s, start:start+s] = 1

        elif mask_type == 'sr2x':
            mask_2d[::2, ::2] = 1    # on garde 1 pixel sur 4
            mask_2d[::2, 1::2] = 0
            mask_2d[1::2, :] = 0

        elif mask_type == 'altlines':
            mask_2d[1::2, :] = 0

        # Même masque pour tous les canaux
        self.mask = mask_2d.unsqueeze(0).expand(n_channels, -1, -1).reshape(-1).to(self.device)

    def forward(self, x):
        return self._as_tensor(x).reshape(-1) * self.mask

    def adjoint(self, y):
        return self._as_tensor(y).reshape(-1) * self.mask

    def input_dim(self):
        return self.n_channels * self.img_size ** 2

    def output_dim(self):
        return self.n_channels * self.img_size ** 2

    def compute_AtA_diag(self):
        return self.mask.clone()


class SuperResolutionOperator(LinearOperator):
    """Sous-échantillonnage par average pooling (facteur k).

    forward(x) : vecteur n → vecteur m   (average pooling k×k par canal)
    adjoint(y) : vecteur m → vecteur n   (réplication uniforme / k^2)

    Avec n = C*H*W et m = C*(H/k)*(W/k).

    Vérification adjoint : <Ax, y> = <x, A^T y> pour tout x, y.
    forward : x_small[c, i, j] = mean(x[c, i*k:(i+1)*k, j*k:(j+1)*k])
            = (1/k^2) * sum(x[c, i*k:(i+1)*k, j*k:(j+1)*k])
    adjoint : (A^T y)[c, i*k+di, j*k+dj] = y[c, i, j] / k^2  pour di, dj in [0, k)

    Preuve : <Ax, y> = sum_c sum_ij x_small[c,i,j] * y[c,i,j]
           = sum_c sum_ij (1/k^2) sum_{di,dj} x[c, i*k+di, j*k+dj] * y[c,i,j]
           = sum_c sum_{i',j'} x[c,i',j'] * y[c, i'//k, j'//k] / k^2
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
    def input_dim(self):
        return self.n_channels * self.img_size ** 2

    def output_dim(self):
        return self.n_channels * self.small_size ** 2
    
    def forward(self, x):
        # Average pooling (divise par k^2)
        x = self._as_tensor(x).reshape(-1)
        k = self.factor
        C, S = self.n_channels, self.small_size
        x_img = x.reshape(C, S, k, S, k)
        x_small = x_img.mean(dim=(2, 4))
        return x_small.reshape(-1)

    def adjoint(self, y):
        # Réplication / k^2
        y = self._as_tensor(y).reshape(-1)
        k = self.factor
        C, S, H = self.n_channels, self.small_size, self.img_size
        y_img = y.reshape(C, S, S)
        y_big = y_img.unsqueeze(2).unsqueeze(4).expand(C, S, k, S, k).reshape(C, H, H)
        return (y_big / (k * k)).reshape(-1)

    def compute_AtA_diag(self):
        val = 1.0 / (self.factor ** 2)
        return torch.full((self.input_dim(),), val,
                        device=self.device, dtype=self.dtype)
    


class StructuredSROperator(LinearOperator):
    """SR comme inpainting structuré : m = n, masque régulier."""
    
    def __init__(self, img_size=256, n_channels=3, factor=4,
                 device=None, dtype=torch.float32):
        self.device = device if device is not None else _default_device(dtype)
        self.dtype = dtype
        self.img_size = img_size
        self.n_channels = n_channels
        self.factor = factor
        
        # Masque : 1 pixel sur k^2 est observé
        mask = torch.zeros(n_channels, img_size, img_size, dtype=dtype)
        mask[:, ::factor, ::factor] = 1.0
        self.mask = mask.reshape(-1).to(self.device)
    
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