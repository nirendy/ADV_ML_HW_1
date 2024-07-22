from abc import ABC, abstractmethod
from pathlib import Path

from torch.utils.data import DataLoader
from torch.utils.data import Dataset, DataLoader, random_split

from src.types import PHASE
from src.types import SPLIT


# from datasets import load_dataset


class BaseDataset(ABC):
    base_data_dir = Path('./data')

    def __init__(self, phase_name: PHASE):
        super().__init__()
        self.phase_name = phase_name

    @abstractmethod
    def get_dataset(self, split: SPLIT) -> Dataset:
        """Return the DataLoader for the dataset.

        Returns:
            DataLoader: The DataLoader instance for the data.
        """
        pass

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        pass

    @property
    @abstractmethod
    def num_classes(self) -> int:
        pass

    def get_train_dataset(self) -> Dataset:
        return self.get_dataset(SPLIT.TRAIN)

    def get_test_dataset(self) -> Dataset:
        return self.get_dataset(SPLIT.TEST)
