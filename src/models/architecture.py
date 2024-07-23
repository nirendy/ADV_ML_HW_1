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
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)

    @staticmethod
    def arch_to_config_override(arch: ARCH) -> CONFIG_KEYS:
        return arch

    @abstractmethod
    def initialize_model(self, dataset: BaseDataset) -> None:
        """Initialize the model architecture."""
        pass

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model."""
        pass

    def save_model(self, path: str):
        """General implementation to save model if model is a PyTorch model."""
        if hasattr(self, 'model') and isinstance(self.model, torch.nn.Module):
            try:
                torch.save(self.model.state_dict(), path)
                self.logger.info(f"Model saved to {path}.")
            except Exception as e:
                self.logger.error(f"Error saving model: {e}")
        else:
            raise NotImplementedError("Subclasses should implement save_model if not using PyTorch.")

    def load_model(self, path: str):
        """General implementation to load model if model is a PyTorch model."""
        if hasattr(self, 'model') and isinstance(self.model, torch.nn.Module):
            try:
                self.model.load_state_dict(torch.load(path))
                self.model.eval()
                self.logger.info(f"Model loaded from {path}.")
            except Exception as e:
                self.logger.error(f"Error loading model: {e}")
        else:
            raise NotImplementedError("Subclasses should implement load_model if not using PyTorch.")

    def predict(self, data: Any) -> Any:
        """General implementation to predict if model is a PyTorch model."""
        if hasattr(self, 'model') and isinstance(self.model, torch.nn.Module):
            self.model.eval()
            with torch.no_grad():
                return self.model(data)
        raise NotImplementedError("Subclasses should implement predict if not using PyTorch.")

    def configure_logging(self, log_level: int = logging.INFO):
        """Configure logging for the class."""
        logging.basicConfig(level=log_level)
        self.logger.setLevel(log_level)
        self.logger.info("Logging configured.")
