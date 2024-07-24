import torch
import torch.nn as nn
import math
from src.datasets.base_dataset import DatasetFactory
from src.models.architecture import AbstractSequenceModel
from src.models.architecture import Architecture
from src.types import PHASE
from src.utils.config_types import TransformerConfig


class TransformerModel(AbstractSequenceModel):
    def __init__(self, d_model, num_heads, num_layers, dim_feedforward, vocab_size, phase_name: PHASE):
        super(TransformerModel, self).__init__(vocab_size, d_model, phase_name)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=dim_feedforward)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward_sequence_model(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        return x


class TransformerCopyArchitecture(Architecture):
    model_config: TransformerConfig

    def initialize_model(self, dataset: DatasetFactory) -> None:
        self.model = TransformerModel(
            d_model=self.model_config['d_model'],
            num_heads=self.model_config['num_heads'],
            num_layers=self.model_config['num_layers'],
            dim_feedforward=self.model_config['dim_feedforward'],
            vocab_size=dataset.vocab_size,
            phase_name=dataset.phase_name
        )
