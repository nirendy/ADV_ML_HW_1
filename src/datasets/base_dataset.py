from abc import ABC
from abc import abstractmethod
from pathlib import Path

from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from src.types import SPLIT


# from datasets import load_dataset


class DatasetFactory(ABC):
    base_data_dir = Path('./data')

    @abstractmethod
    def get_dataset(self, split: SPLIT) -> Dataset:
        """Return the DataLoader for the dataset.

        Returns:
            DataLoader: The DataLoader instance for the data.
        """
        pass

    def get_train_dataset(self) -> Dataset:
        return self.get_dataset(SPLIT.TRAIN)

    def get_test_dataset(self) -> Dataset:
        return self.get_dataset(SPLIT.TEST)
