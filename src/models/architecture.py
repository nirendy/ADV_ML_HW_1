from abc import ABC, abstractmethod
from typing import Dict, Any
import logging
from typing import Iterator

import torch
from torch.utils.data import DataLoader
from src.utils.config_types import TrainingConfig


class Architecture(ABC):
    model_config: Dict[str, Any]
    training_config: TrainingConfig
    model: torch.nn.Module
    criterion: torch.nn.Module
    optimizer: torch.optim.Optimizer

    def __init__(self, config: Dict[str, Any], training_config: TrainingConfig):
        self.model_config = config
        self.training_config = training_config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)
        self.initialize_model()

    @abstractmethod
    def initialize_model(self) -> None:
        """Initialize the model architecture."""
        pass

    @abstractmethod
    def train_model(self, train_loader: DataLoader) -> None:
        """Train the model using the provided training data loader."""
        pass

    @abstractmethod
    def evaluate_model(self, test_loader: DataLoader) -> Dict[str, float]:
        """Evaluate the model using the provided test data loader.

        Returns:
            A dictionary containing evaluation metrics.
        """
        pass

    def update_config(self, new_config: Dict[str, Any]):
        """Update the model configuration and reinitialize the model.

        Args:
            new_config: A dictionary containing new configuration parameters.
        """
        self.model_config.update(new_config)
        self.initialize_model()
        self.logger.info("Configuration updated and model reinitialized.")

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

    def parameters(self) -> Iterator[torch.nn.Parameter]:
        """Return the model parameters."""
        if hasattr(self, 'model') and isinstance(self.model, torch.nn.Module):
            return self.model.parameters()

        raise NotImplementedError("Subclasses should implement parameters if not using PyTorch.")

    def configure_logging(self, log_level: int = logging.INFO):
        """Configure logging for the class."""
        logging.basicConfig(level=log_level)
        self.logger.setLevel(log_level)
        self.logger.info("Logging configured.")
