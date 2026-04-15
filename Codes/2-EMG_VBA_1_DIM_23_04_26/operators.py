from abc import ABC, abstractmethod
import numpy as np


class LinearOperator(ABC):
    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def adjoint(self, y):
        pass

    @abstractmethod
    def to_matrix(self):
        pass

    def create_observation(self, x0, sigma_b):
        Ax0 = self.forward(x0)
        return(Ax0 + sigma_b * np.random.randn(*Ax0.shape))


class GaussianBlurOperator(LinearOperator):
    def __init__(self, kernel_size=9, sigma=3.0, img_size=64):
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

        # Noyau plongé dans une image H×W pour FFT (centre en (0,0))
        H = W = img_size
        kernel_padded = np.zeros((H, W))
        half = k // 2
        for i in range(k):
            for j in range(k):
                ii = (i - half) % H
                jj = (j - half) % W
                kernel_padded[ii, jj] = kernel[i, j]
        self._kernel_fft = np.fft.fft2(kernel_padded)

        # Cache pour to_matrix()
        self._matrix = None

    def forward(self, x):
        flat = x.ndim == 1
        if flat:
            x = x.reshape(self.img_size, self.img_size)
        y = np.fft.ifft2(np.fft.fft2(x) * self._kernel_fft).real
        return( y.ravel() if flat else y)

    def adjoint(self, y):
        flat = y.ndim == 1
        if flat:
            y = y.reshape(self.img_size, self.img_size)
        x = np.fft.ifft2(np.fft.fft2(y) * np.conj(self._kernel_fft)).real
        return( x.ravel() if flat else x)

    def to_matrix(self):
        if self._matrix is not None:
            return self._matrix
        H = W = self.img_size
        n = H * W
        A = np.zeros((n, n))
        half = self.kernel_size // 2
        for i in range(H):
            for j in range(W):
                row = i * W + j
                for di in range(self.kernel_size):
                    for dj in range(self.kernel_size):
                        src_i = (i - di + half) % H
                        src_j = (j - dj + half) % W
                        col = src_i * W + src_j
                        A[row, col] += self.kernel[di, dj]
        self._matrix = A
        return(A)

