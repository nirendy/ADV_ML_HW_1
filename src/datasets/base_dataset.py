from abc import ABC, abstractmethod
from pathlib import Path

from torch.utils.data import DataLoader
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import BertTokenizer
from typing_extensions import assert_never

from src.types import PHASE
from src.types import SPLIT


# from datasets import load_dataset


class BaseDataset(ABC):
    base_data_dir = Path('./data')

    def __init__(self, phase_name: PHASE):
        super().__init__()
        self.phase_name = phase_name
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    @abstractmethod
    def get_dataset(self, split: SPLIT) -> Dataset:
        """Return the DataLoader for the dataset.

        Returns:
            DataLoader: The DataLoader instance for the data.
        """
        pass

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size

    @property
    def num_classes(self) -> int:
        if self.phase_name == PHASE.CLASSIFICATION:
            return 2
        elif self.phase_name == PHASE.AUTOREGRESSIVE:
            return self.vocab_size
        assert_never(self.phase_name)

    def get_train_dataset(self) -> Dataset:
        return self.get_dataset(SPLIT.TRAIN)

    def get_test_dataset(self) -> Dataset:
        return self.get_dataset(SPLIT.TEST)
