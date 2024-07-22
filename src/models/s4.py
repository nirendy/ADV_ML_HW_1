from typing import Tuple, Dict
import torch
import torch.nn as nn
import numpy as np
from numpy.linalg import matrix_power, inv
from src.datasets.base_dataset import BaseDataset
from src.models.architecture import Architecture
from src.utils.config_types import S4Config


class S4Layer(nn.Module):
    def __init__(self, A, B, C, L, kernel_method: bool):
        """
        A: (N, N)
        B: (N, 1)
        C: (1, N)
        L: int, sequence length
        kernel_method: bool, whether to use kernel method
        """
        super(S4Layer, self).__init__()
        self.kernel_method = kernel_method
        self.L = L
        self.A, self.B, self.C = self.discretize(A, B, C, step=1.0 / L)
        self.A = torch.tensor(self.A, dtype=torch.float32)
        self.B = torch.tensor(self.B, dtype=torch.float32)
        self.C = torch.tensor(self.C, dtype=torch.float32)
        if kernel_method:
            self.K = self.compute_kernel(self.A.numpy(), self.B.numpy(), self.C.numpy(), self.L)
            self.K = torch.tensor(self.K, dtype=torch.float32)

    def discretize(self, A: np.ndarray, B: np.ndarray, C: np.ndarray, step: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Discretize the state-space matrices using bilinear (Tustin) transform.
        """
        I = np.eye(A.shape[0])  # (N, N)
        BL = inv(I - (step / 2.0) * A)  # (N, N) = (N, N) - (N, N)
        Ab = BL @ (I + (step / 2.0) * A)  # (N, N) = (N, N) @ ((N, N) + (N, N))
        Bb = (BL * step) @ B  # (N, 1) = (N, N) @ (N, 1)
        return Ab, Bb, C

    def compute_kernel(self, Ab: np.ndarray, Bb: np.ndarray, Cb: np.ndarray, L: int) -> np.ndarray:
        """
        Compute the convolution kernel for the S4 layer.

        Ab: (N, N)
        Bb: (N, 1)
        Cb: (1, N)
        Returns K: (L,)
        """
        return np.array([
            (Cb @ matrix_power(Ab, l) @ Bb).item() # (1, 1) = (1, N) @ (N, N)^l @ (N, 1)
            for l in range(L)
        ])  # => (L,)

    def forward_kernel(self, u: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using the kernel method.
        u: (batch_size, L, d_model)
        Returns y: (batch_size, L, d_model)
        """
        batch_size, L, d_model = u.shape
        u = u.view(batch_size * d_model, L)  # (batch_size * d_model, L)
        u_padded = np.pad(u.cpu().detach().numpy(), ((0, 0), (0, self.K.shape[0])), 'constant')  # (batch_size * d_model, L + K)
        u_fft = np.fft.rfft(u_padded)  # (batch_size * d_model, L + K)
        K_padded = np.pad(self.K.cpu().detach().numpy(), (0, u_fft.shape[1] - self.K.shape[0]), 'constant')  # (L + K,)
        K_fft = np.fft.rfft(K_padded)  # (L + K,)
        y = np.fft.irfft(u_fft * K_fft)[:, :L]  # (batch_size * d_model, L)
        y = torch.tensor(y, dtype=u.dtype, device=u.device).view(batch_size, d_model, L).permute(0, 2, 1)  # (batch_size * d_model, L) => (batch_size, L, d_model)
        return y

    def forward_regressive(self, u: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using the regressive method.
        u: (batch_size, L, d_model)
        Returns y: (batch_size, L, d_model)
        """
        batch_size, L, d_model = u.shape
        u = u.permute(0, 2, 1).reshape(batch_size * d_model, L)  # (batch_size, L, d_model) => (batch_size * d_model, L)

        def step(x_k_1, u_k):
            """
            Perform one step of the recurrent computation.
            x_k_1: (batch_size * d_model, N)
            u_k: (batch_size * d_model,)
            Returns x_k: (batch_size * d_model, N)
            y_k: (batch_size * d_model,)
            """
            x_k = self.A @ x_k_1.T + self.B @ u_k.unsqueeze(-1).T  # (N, batch_size * d_model) = (N, N) @ (batch_size * d_model, N).T + (N, 1) @ (batch_size * d_model, 1).T
            y_k = self.C @ x_k  # (1, batch_size * d_model) = (1, N) @ (N, batch_size * d_model)
            return x_k.T, y_k.T  # (batch_size * d_model, N), (batch_size * d_model,)

        x_k_1 = torch.zeros((batch_size * d_model, self.A.shape[0]), dtype=u.dtype, device=u.device)  # (batch_size * d_model, N)
        ys = []
        for u_k in u.T:  # Iterate over time steps
            x_k_1, y_k = step(x_k_1, u_k)  # x_k_1: (batch_size * d_model, N), u_k: (batch_size * d_model,) => x_k: (batch_size * d_model, N), y_k: (batch_size * d_model,)
            ys.append(y_k)

        ys = torch.stack(ys, dim=1)  # (batch_size * d_model, L)
        ys = ys.view(batch_size, d_model, L).permute(0, 2, 1)  # (batch_size * d_model, L) => (batch_size, L, d_model)
        return ys

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the S4 layer.
        u: (batch_size, L, d_model)
        Returns y: (batch_size, L, d_model)
        """
        if self.kernel_method:
            return self.forward_kernel(u)
        else:
            return self.forward_regressive(u)


class S4Architecture(Architecture):
    model_config: S4Config

    def initialize_model(self, dataset: BaseDataset) -> None:
        """
        Initializes the S4 model.
        """
        self.model = nn.Module()
        self.model.embedding = nn.Embedding(dataset.vocab_size, self.model_config['d_model'])  # (vocab_size, d_model)
        self.model.s4_layers = nn.ModuleList(
            [S4Layer(self.make_A(self.model_config['state_size']),
                     np.random.randn(self.model_config['state_size'], 1),  # (N, 1)
                     np.random.randn(1, self.model_config['state_size']),  # (1, N)
                     L=100,  # Sequence length
                     kernel_method=False)
             for _ in range(self.model_config['num_layers'])]
        )
        self.model.fc = nn.Linear(self.model_config['d_model'], dataset.num_classes)  # (d_model, num_classes)

    def make_A(self, N: int) -> np.ndarray:
        """
        Creates the A matrix used in the S4 layer.
        A: (N, N)
        """
        def v(n, k):
            if n > k:
                return np.sqrt(2 * n + 1) * np.sqrt(2 * k + 1)
            elif n == k:
                return n + 1
            else:
                return 0
        return -np.array([[v(n, k) for k in range(1, N + 1)] for n in range(1, N + 1)])  # (N, N)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, L)
        Returns output: (batch_size, num_classes)
        """
        embedded = self.model.embedding(x)  # (batch_size, L, d_model)

        for layer in self.model.s4_layers:
            embedded = layer(embedded)  # (batch_size, L, d_model)

        output = self.model.fc(embedded[:, 0, :])  # (batch_size, d_model) => (batch_size, num_classes)
        return output
