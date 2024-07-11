from abc import ABC, abstractmethod
from torch.utils.data import DataLoader

class Dataset(ABC):
    @abstractmethod
    def load_data(self):
        pass

    @abstractmethod
    def get_train_loader(self) -> DataLoader:
        pass

    @abstractmethod
    def get_test_loader(self) -> DataLoader:
        pass
