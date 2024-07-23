import torch
import torch.nn as nn
import math
from src.datasets.base_dataset import BaseDataset
from src.models.architecture import Architecture
from src.types import PHASE
from src.utils.config_types import TransformerConfig


class TransformerModel(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, dim_feedforward, vocab_size, phase_name: PHASE):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=dim_feedforward)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.phase_name = phase_name
        if phase_name == PHASE.CLASSIFICATION:
            self.fc = nn.Linear(d_model, 2)
        elif phase_name == PHASE.AUTOREGRESSIVE:
            self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids):
        x = self.embedding(input_ids)  # * attention_mask.unsqueeze(-1)  # Apply attention mask
        x = x.permute(1, 0, 2)  # Transformer expects (seq_len, batch, d_model)
        x = self.transformer(x)
        if self.phase_name == PHASE.CLASSIFICATION:
            x = x.mean(dim=0)
        x = self.fc(x)
        return x


class TransformerCopyArchitecture(Architecture):
    model_config: TransformerConfig

    def initialize_model(self, dataset: BaseDataset) -> None:
        self.model = TransformerModel(
            d_model=self.model_config['d_model'],
            num_heads=self.model_config['num_heads'],
            num_layers=self.model_config['num_layers'],
            dim_feedforward=self.model_config['dim_feedforward'],
            vocab_size=dataset.vocab_size,
            phase_name=dataset.phase_name
        )
