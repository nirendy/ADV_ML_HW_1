from abc import ABC, abstractmethod
from typing import Dict, Any
import logging
from typing import Iterator

import torch
import torch.nn as nn

from src.datasets.base_dataset import BaseDataset
from src.types import ARCH
from src.types import CONFIG_KEYS


class Architecture(ABC):
    model_config: Dict[str, Any]
    model: torch.nn.Module

    def __init__(self, config: Dict[str, Any]):
        super(Architecture, self).__init__()
        self.model_config = config

    @abstractmethod
    def initialize_model(self, dataset: BaseDataset) -> None:
        """Initialize the model architecture."""
        pass

    @property
    def param_count(self) -> int:
        """Return the number of parameters in the model."""
        return sum(p.numel() for p in self.model.parameters())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model."""
        return self.model(x)
