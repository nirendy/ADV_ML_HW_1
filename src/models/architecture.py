from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import Dict

import torch
from torch import nn
from typing_extensions import assert_never

from src.datasets.base_dataset import DatasetFactory
from src.types import PHASE


class AbstractSequenceModel(nn.Module, ABC):
    phase_name: PHASE
    fc: nn.Linear

    def __init__(
            self,
            vocab_size: int,
            d_model: int,
            phase_name: PHASE
    ):
        super(AbstractSequenceModel, self).__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.set_phase(phase_name)

    def set_phase(self, phase_name: PHASE):
        self.phase_name = phase_name
        if self.phase_name == PHASE.CLASSIFICATION:
            self.fc = nn.Linear(self.d_model, 2)
        elif self.phase_name == PHASE.AUTOREGRESSIVE:
            self.fc = nn.Linear(self.d_model, self.vocab_size)

    @abstractmethod
    def forward_sequence_model(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(x)
        x = self.forward_sequence_model(embedded)
        if self.phase_name == PHASE.CLASSIFICATION:
            # Average over the sequence dimension
            x = x.mean(dim=1)
        elif self.phase_name == PHASE.AUTOREGRESSIVE:
            x = x.contiguous().view(-1, x.size(-1))
        else:
            assert_never(self.phase_name)
        x = self.fc(x)
        return x


class Architecture(ABC):
    model_config: Dict[str, Any]
    model: AbstractSequenceModel

    def __init__(self, config: Dict[str, Any]):
        super(Architecture, self).__init__()
        self.model_config = config

    @abstractmethod
    def initialize_model(self, dataset: DatasetFactory) -> None:
        """Initialize the model architecture."""
        pass

    @property
    def param_count(self) -> int:
        """Return the number of parameters in the model."""
        return sum(p.numel() for p in self.model.parameters())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model."""
        return self.model(x)
