from abc import ABC, abstractmethod
from typing import Dict


class Architecture(ABC):
    config: Dict

    def __init__(self, config: Dict):
        self.config = config
        self.initialize_model()

    @abstractmethod
    def initialize_model(self):
        pass

    @abstractmethod
    def train_model(self, train_loader):
        pass

    @abstractmethod
    def evaluate_model(self, test_loader) -> Dict[str, float]:
        pass
