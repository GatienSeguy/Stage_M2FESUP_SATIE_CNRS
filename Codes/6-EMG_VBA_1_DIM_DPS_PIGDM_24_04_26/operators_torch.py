from abc import ABC, abstractmethod
import torch


def _default_device(dtype=torch.float32):
    # MPS ne supporte pas float64 → fallback CPU
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available() and dtype != torch.float64:
        return torch.device("mps")
    return torch.device("cpu")


class LinearOperator(ABC):
    """Opérateur linéaire A (version PyTorch / GPU).

    L'utilisateur fournit deux opérations :
        forward(x)  ->  A x
        adjoint(y)  ->  A^T y
    Aucune matrice n'est jamais construite ni stockée.
    """

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def adjoint(self, y):
        pass

    @abstractmethod
    def input_dim(self):
        pass

    @abstractmethod
    def output_dim(self):
        pass

    def compute_AtA_diag(self):
        """Diagonale de A^T A, calculée génériquement.
        AtA_diag[i] = e_i^T A^T A e_i = (A^T(A e_i))[i]
        Coût : n appels forward + n appels adjoint  (précalcul unique).
        """
        n = self.input_dim()
        diag = torch.zeros(n, device=self.device, dtype=self.dtype)
        e_i = torch.zeros(n, device=self.device, dtype=self.dtype)
        for i in range(n):
            e_i[i] = 1.0
            diag[i] = self.adjoint(self.forward(e_i))[i]
            e_i[i] = 0.0
        return diag

    def create_observation(self, x0, sigma_b):
        """y = A x_0 + sigma_b * epsilon,  epsilon ~ N(0, I)."""
        x0 = self._as_tensor(x0)
        Ax0 = self.forward(x0)
        noise = torch.randn(Ax0.shape, device=Ax0.device, dtype=Ax0.dtype)
        return Ax0 + sigma_b * noise

    def _as_tensor(self, x):
        if isinstance(x, torch.Tensor):
            # MPS ne supporte pas float64 : déplacer d'abord, caster ensuite
            return x.to(device=self.device).to(dtype=self.dtype)
        return torch.as_tensor(x, device=self.device, dtype=self.dtype)


class GaussianBlurOperator(LinearOperator):
    def __init__(self, kernel_size=9, sigma=1.5, img_size=64, n_channels=1,
                 device=None, dtype=torch.float32):
        self.device = device if device is not None else _default_device(dtype)
        self.dtype = dtype
        # dtype complexe cohérent avec le dtype réel
        self._cdtype = torch.complex64 if dtype == torch.float32 else torch.complex128

        self.kernel_size = kernel_size
        self.sigma_blur = sigma
        self.img_size = img_size
        self.n_channels = n_channels

        # Noyau gaussien 2D normalisé
        k = kernel_size
        ax = torch.arange(k, dtype=dtype) - (k - 1) / 2.0
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        self.kernel = kernel.to(device=self.device)

        # Noyau plongé dans une image H×W (centre en (0,0)) pour FFT
        H = W = img_size
        kernel_padded = torch.zeros((H, W), dtype=dtype)
        half = k // 2
        for i in range(k):
            for j in range(k):
                ii = (i - half) % H
                jj = (j - half) % W
                kernel_padded[ii, jj] = kernel[i, j]

        # FFT effectuée sur le device cible
        kernel_padded = kernel_padded.to(device=self.device)
        self._kernel_fft = torch.fft.fft2(kernel_padded.to(self._cdtype))

    def input_dim(self):
        return self.n_channels * self.img_size ** 2

    def output_dim(self):
        return self.n_channels * self.img_size ** 2

    def _conv_fft(self, x, kernel_fft):
        """Applique la convolution FFT sur un tenseur 2D ou 3D (C,H,W)."""
        x_c = x.to(self._cdtype)
        out = torch.fft.ifft2(torch.fft.fft2(x_c) * kernel_fft).real
        return out.to(self.dtype)

    def forward(self, x):
        x = self._as_tensor(x)
        flat = x.ndim == 1
        if flat:
            x = x.reshape(self.n_channels, self.img_size, self.img_size)
        if x.ndim == 3:
            # broadcasting du noyau sur le premier axe (canaux)
            y = self._conv_fft(x, self._kernel_fft.unsqueeze(0))
        else:
            y = self._conv_fft(x, self._kernel_fft)
        return y.reshape(-1) if flat else y

    def adjoint(self, y):
        y = self._as_tensor(y)
        flat = y.ndim == 1
        if flat:
            y = y.reshape(self.n_channels, self.img_size, self.img_size)
        kernel_fft_conj = torch.conj(self._kernel_fft)
        if y.ndim == 3:
            x = self._conv_fft(y, kernel_fft_conj.unsqueeze(0))
        else:
            x = self._conv_fft(y, kernel_fft_conj)
        return x.reshape(-1) if flat else x

    def compute_AtA_diag(self):
        n_spatial = self.img_size ** 2
        val = (torch.abs(self._kernel_fft) ** 2).sum().real / n_spatial
        return torch.full((self.input_dim(),), val.item(),
                          device=self.device, dtype=self.dtype)
