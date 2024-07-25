from src.datasets.text_dataset import TextDatasetFactory
from src.models.architecture import AbstractSequenceModel
from src.models.architecture import Architecture
from src.types import PHASE
from src.utils.config_types import S4Config

"""Minimal version of S4D with extra options and features stripped out, for pedagogical purposes."""

import math
import torch
import torch.nn as nn
from einops import repeat


class S4DKernel(nn.Module):
    """Generate convolution kernel from diagonal SSM parameters."""

    def __init__(self, d_model, N, dt_min=0.001, dt_max=0.1, lr=None):
        super().__init__()
        # Generate dt
        H = d_model
        log_dt = torch.rand(H) * (
                math.log(dt_max) - math.log(dt_min)
        ) + math.log(dt_min)

        C = torch.randn(H, N // 2, dtype=torch.cfloat)
        self.C = nn.Parameter(torch.view_as_real(C))
        self.register("log_dt", log_dt, lr)

        log_A_real = torch.log(0.5 * torch.ones(H, N // 2))
        A_imag = math.pi * repeat(torch.arange(N // 2), 'n -> h n', h=H)
        self.register("log_A_real", log_A_real, lr)
        self.register("A_imag", A_imag, lr)

    def forward(self, L):
        """
        returns: (..., c, L) where c is number of channels (default 1)
        """

        # Materialize parameters
        dt = torch.exp(self.log_dt)  # (H)
        C = torch.view_as_complex(self.C)  # (H N)
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag  # (H N)

        # Vandermonde multiplication
        dtA = A * dt.unsqueeze(-1)  # (H N)
        K = dtA.unsqueeze(-1) * torch.arange(L, device=A.device)  # (H N L)
        C = C * (torch.exp(dtA) - 1.) / A
        K = 2 * torch.einsum('hn, hnl -> hl', C, torch.exp(K)).real

        return K

    def register(self, name, tensor, lr=None):
        """Register a tensor with a configurable learning rate and 0 weight decay"""

        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))

            optim = {"weight_decay": 0.0}
            if lr is not None: optim["lr"] = lr
            setattr(getattr(self, name), "_optim", optim)


class S4D(nn.Module):
    def __init__(self, d_model, d_state, transposed=True, **kernel_args):
        super().__init__()

        self.h = d_model
        self.n = d_state
        self.transposed = transposed

        self.D = nn.Parameter(torch.randn(self.h))

        # SSM Kernel
        self.kernel = S4DKernel(self.h, N=self.n, **kernel_args)

        # Pointwise
        self.activation = nn.GELU()

        # position-wise output transform to mix features
        self.output_linear = nn.Sequential(
            nn.Conv1d(self.h, 2 * self.h, kernel_size=1),
            nn.GLU(dim=-2),
        )

    def forward(self, u):  # absorbs return_output and transformer src mask
        """

        Args:
            u: (B, H, L) if transposed else (B, L, H)

        Returns:

        """

        if not self.transposed: u = u.transpose(-1, -2)
        L = u.size(-1)

        # Compute SSM Kernel
        k = self.kernel(L=L)  # (H L)

        # Convolution
        k_f = torch.fft.rfft(k, n=2 * L)  # (H L)
        u_f = torch.fft.rfft(u, n=2 * L)  # (B H L)
        y = torch.fft.irfft(u_f * k_f, n=2 * L)[..., :L]  # (B H L)

        # Compute D term in state space equation - essentially a skip connection
        y = y + u * self.D.unsqueeze(-1)

        y = self.activation(y)
        y = self.output_linear(y)
        if not self.transposed: y = y.transpose(-1, -2)
        return y, None  # Return a dummy state to satisfy this repo's interface, but this can be modified


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

        self.s4_layers = nn.ModuleList(
            [
                S4D(d_model, state_size)
                for _ in range(num_layers)
            ]
        )

    def forward_sequence_model(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters:
            x: (batch_size, seq_len, d_model)
        Returns:
            x: (batch_size, seq_len, d_model)
        """
        # Change dims to (batch_size, d_model, seq_len)
        x = x.permute(0, 2, 1)

        for layer in self.s4_layers:
            x, _ = layer(x)

        # Change dims back
        output = x.permute(0, 2, 1)
        return output


class S4CopyArchitecture(Architecture):
    model_config: S4Config

    def initialize_model(self, dataset: TextDatasetFactory) -> None:
        self.model = S4Model(
            d_model=self.model_config['d_model'],
            state_size=self.model_config['state_size'],
            num_layers=self.model_config['num_layers'],
            vocab_size=dataset.vocab_size,
            phase_name=dataset.phase_name,
        )
