import torch
import torch.nn as nn
import math
from src.datasets.base_dataset import DatasetFactory
from src.datasets.text_dataset import TextDatasetFactory
from src.models.architecture import AbstractSequenceModel
from src.models.architecture import Architecture
from src.types import PHASE
from src.utils.config_types import TransformerConfig


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        attn = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn, value)
        return output, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.Q = nn.Linear(d_model, d_model)
        self.K = nn.Linear(d_model, d_model)
        self.V = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention()

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        batch_size = query.size(0)

        # Linear projections
        query = self.Q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        key = self.K(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        value = self.V(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Apply attention
        attn_output, _ = self.attention(query, key, value)

        # Concatenate heads and put through final linear layer
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.out_linear(attn_output)

        return output


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(torch.relu(self.linear1(x)))


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dim_feedforward: int):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, dim_feedforward)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_output = self.attention(x, x, x)
        x = self.norm1(x + attn_output)
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()

        pe = self.initialize_positional_encoding(d_model, max_len)
        # Registers pe as a buffer so it is part of the model's state (e.g. saved to disk, moved to GPU)
        # but not a learnable parameter.
        self.register_buffer('pe', pe)

    @staticmethod
    def initialize_positional_encoding(d_model: int, max_len: int) -> torch.Tensor:
        pe = torch.zeros(max_len, d_model)

        # Create a tensor of positions (0, 1, 2, ..., max_len-1).unsqueeze(1) -> (max_len, 1)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # Calculate the division term
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Apply sine to even indices in the array
        pe[:, 0::2] = torch.sin(position * div_term)

        # Apply cosine to odd indices in the array
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add an extra dimension for batch size and transpose to (max_len, 1, d_model)
        pe = pe.unsqueeze(0).transpose(0, 1)

        return pe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0), :]


class TransformerModel(AbstractSequenceModel):
    def __init__(self, d_model, num_heads, num_layers, dim_feedforward, vocab_size, phase_name: PHASE):
        super(TransformerModel, self).__init__(vocab_size, d_model, phase_name)
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dim_feedforward = dim_feedforward

        self.positional_encoding = PositionalEncoding(d_model)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, dim_feedforward)
            for _ in range(num_layers)
        ])

    def forward_sequence_model(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters:
            x: (batch_size, seq_len, d_model)
        Returns:
            x: (batch_size, seq_len, d_model)
        """
        # Change dims to (seq_len, batch_size, d_model)
        x = x.permute(1, 0, 2)
        x = self.positional_encoding(x)

        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)

        # Change dims back
        output = x.permute(1, 0, 2)

        return output


class TransformerArchitecture(Architecture):
    model_config: TransformerConfig

    def initialize_model(self, dataset: TextDatasetFactory) -> None:
        self.model = TransformerModel(
            d_model=self.model_config['d_model'],
            num_heads=self.model_config['num_heads'],
            num_layers=self.model_config['num_layers'],
            dim_feedforward=self.model_config['dim_feedforward'],
            vocab_size=dataset.vocab_size,
            phase_name=dataset.phase_name
        )
