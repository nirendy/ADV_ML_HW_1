from abc import ABC, abstractmethod

class Architecture(ABC):
    @abstractmethod
    def initialize_model(self):
        pass

    @abstractmethod
    def train_model(self, train_loader):
        pass

    @abstractmethod
    def evaluate_model(self, test_loader):
        pass

    @abstractmethod
    def get_metrics(self):
        pass
