from typing import Tuple, Dict
import torch
import torch.nn as nn
import numpy as np
from numpy.linalg import matrix_power, inv
from src.datasets.base_dataset import DatasetFactory
from src.datasets.text_dataset import TextDatasetFactory
from src.models.architecture import AbstractSequenceModel
from src.models.architecture import Architecture
from src.types import PHASE
from src.utils.config_types import S4Config


class S4Layer(nn.Module):
    def __init__(
            self,
            d_model: int,
            d_state: int,
            kernel_method: bool
    ):
        """
        A: (N, N)
        B: (N, 1)
        C: (1, N)
        L: int, sequence length
        kernel_method: bool, whether to use kernel method
        """
        super(S4Layer, self).__init__()
        self.kernel_method = kernel_method
        self.h = d_state

        A = self.make_A(self.h)
        B = np.random.randn(self.h, 1)  # (N, 1)
        C = np.random.randn(1, self.h)

        self.A_, self.B_, self.C_ = self.discretize(A, B, C, step=1.0 / L)
        self.A_ = torch.tensor(self.A_, dtype=torch.float32)
        self.B_ = torch.tensor(self.B_, dtype=torch.float32)
        self.C_ = torch.tensor(self.C_, dtype=torch.float32)
        if kernel_method:
            self.K = self.compute_kernel(self.A_.numpy(), self.B_.numpy(), self.C_.numpy(), self.L)
            self.K = torch.tensor(self.K, dtype=torch.float32)

    @staticmethod
    def make_A(N: int) -> np.ndarray:
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

    def discretize(self, A: np.ndarray, B: np.ndarray, C: np.ndarray, step: float) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray]:
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
            (Cb @ matrix_power(Ab, l) @ Bb).item()  # (1, 1) = (1, N) @ (N, N)^l @ (N, 1)
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
        u_padded = np.pad(u.cpu().detach().numpy(), ((0, 0), (0, self.K.shape[0])),
                          'constant')  # (batch_size * d_model, L + K)
        u_fft = np.fft.rfft(u_padded)  # (batch_size * d_model, L + K)
        K_padded = np.pad(self.K.cpu().detach().numpy(), (0, u_fft.shape[1] - self.K.shape[0]), 'constant')  # (L + K,)
        K_fft = np.fft.rfft(K_padded)  # (L + K,)
        y = np.fft.irfft(u_fft * K_fft)[:, :L]  # (batch_size * d_model, L)
        y = torch.tensor(y, dtype=u.dtype, device=u.device).view(batch_size, d_model, L).permute(0, 2,
                                                                                                 1)  # (batch_size * d_model, L) => (batch_size, L, d_model)
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
            x_k = self.A_ @ x_k_1.T + self.B_ @ u_k.unsqueeze(
                -1).T  # (N, batch_size * d_model) = (N, N) @ (batch_size * d_model, N).T + (N, 1) @ (batch_size * d_model, 1).T
            y_k = self.C_ @ x_k  # (1, batch_size * d_model) = (1, N) @ (N, batch_size * d_model)
            return x_k.T, y_k.T  # (batch_size * d_model, N), (batch_size * d_model,)

        x_k_1 = torch.zeros((batch_size * d_model, self.A_.shape[0]), dtype=u.dtype,
                            device=u.device)  # (batch_size * d_model, N)
        ys = []
        for u_k in u.T:  # Iterate over time steps
            x_k_1, y_k = step(x_k_1,
                              u_k)  # x_k_1: (batch_size * d_model, N), u_k: (batch_size * d_model,) => x_k: (batch_size * d_model, N), y_k: (batch_size * d_model,)
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


class S4Model(AbstractSequenceModel):
    def __init__(
            self,
            d_model: int,
            state_size: int,
            num_layers: int,
            vocab_size: int,
            phase_name: PHASE
    ):
        super(S4Model, self).__init__(vocab_size, d_model, phase_name)
        self.state_size = state_size
        self.num_layers = num_layers

        self.s4_layers = nn.ModuleList([
            S4Layer(
                d_model=d_model,
                d_state=state_size,
                kernel_method=True
            )
            for _ in range(num_layers)
        ])

    def forward_sequence_model(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters:
            x: (batch_size, seq_len, d_model)
        Returns:
            x: (batch_size, seq_len, d_model)
        """

        for layer in self.s4_layers:
            x, _ = layer(x)

        output = x
        return output


class S4Architecture(Architecture):
    model_config: S4Config

    def initialize_model(self, dataset: TextDatasetFactory) -> None:
        self.model = S4Model(
            d_model=self.model_config['d_model'],
            state_size=self.model_config['state_size'],
            num_layers=self.model_config['num_layers'],
            vocab_size=dataset.vocab_size,
            phase_name=dataset.phase_name,
        )
