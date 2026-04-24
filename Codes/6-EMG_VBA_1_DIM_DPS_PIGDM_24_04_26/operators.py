from abc import ABC, abstractmethod
import numpy as np


class LinearOperator(ABC):
    """Opérateur linéaire A.

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
        AtA_diag[i] = e_i^T A^T A e_I  =  (A-T(A e_i))[i]
        Coût : n appels forward + n appels adjoint  (précalcul unique).
        """
        n = self.input_dim()
        diag = np.zeros(n)
        e_i = np.zeros(n)
        for i in range(n):
            e_i[i] = 1.0
            diag[i] = self.adjoint(self.forward(e_i))[i]
            e_i[i] = 0.0
        return(diag)

    def create_observation(self, x0, sigma_b):
        """y = A x_0 + sigma_b espilon,   espilon approx N(0, I)."""
        Ax0 = self.forward(x0)
        return Ax0 + sigma_b * np.random.randn(*Ax0.shape)


class GaussianBlurOperator(LinearOperator):
    """Flou gaussien 2D circulaire, implémenté en FFT (pas de matrice)."""

    def __init__(self, kernel_size=9, sigma=1.5, img_size=64, n_channels=1):
        self.kernel_size = kernel_size
        self.sigma_blur = sigma
        self.img_size = img_size

        # Noyau gaussien 2D normalisé
        k = kernel_size
        ax = np.arange(k) - (k - 1) / 2.0
        xx, yy = np.meshgrid(ax, ax, indexing='ij')
        kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        kernel /= kernel.sum()
        self.kernel = kernel

        # Noyau plongé dans une image H×W (centre en (0,0)) pour FFT
        H = W = img_size
        kernel_padded = np.zeros((H, W))
        half = k // 2
        for i in range(k):
            for j in range(k):
                ii = (i - half) % H
                jj = (j - half) % W
                kernel_padded[ii, jj] = kernel[i, j]

        self._kernel_fft = np.fft.fft2(kernel_padded)

        self.n_channels = n_channels

    # --- Interface LinearOperator ---

    def input_dim(self):
        return self.n_channels * self.img_size ** 2

    def output_dim(self):
        return self.n_channels * self.img_size ** 2

    def forward(self, x):
        flat = x.ndim == 1
        if flat:
            x = x.reshape(self.n_channels, self.img_size, self.img_size)
        if x.ndim == 3:
            y = np.stack([np.fft.ifft2(np.fft.fft2(x[c]) * self._kernel_fft).real
                          for c in range(x.shape[0])])
        else:
            y = np.fft.ifft2(np.fft.fft2(x) * self._kernel_fft).real
        return y.ravel() if flat else y

    def adjoint(self, y):
        flat = y.ndim == 1
        if flat:
            y = y.reshape(self.n_channels, self.img_size, self.img_size)
        if y.ndim == 3:
            x = np.stack([np.fft.ifft2(np.fft.fft2(y[c]) * np.conj(self._kernel_fft)).real
                          for c in range(y.shape[0])])
        else:
            x = np.fft.ifft2(np.fft.fft2(y) * np.conj(self._kernel_fft)).real
        return x.ravel() if flat else x

    def compute_AtA_diag(self):
        n_spatial = self.img_size ** 2
        val = np.sum(np.abs(self._kernel_fft) ** 2).real / n_spatial
        return np.full(self.input_dim(), val)  # même valeur pour chaque canal