import torch
from torch.utils.data import Dataset

from src.datasets.base_dataset import BaseDataset
from src.types import PHASE
from src.types import SPLIT


class ListOpsDataset(BaseDataset):
    @property
    def data_dir(self):
        return self.base_data_dir / 'preprocessed' / 'listops-1000'

    @property
    def vocab_size(self) -> int:
        vocab_size_file = self.data_dir / 'vocab_size.pt'
        return torch.load(vocab_size_file)

    @property
    def num_classes(self) -> int:
        return 10

    def get_dataset(self, split: SPLIT) -> Dataset:
        data_file = self.data_dir / f'{split}_clean.pt'
        target_file = self.data_dir / f'target_{split}_clean.pt'

        data = torch.load(data_file)
        targets = torch.load(target_file)

        return ListOpsTorchDataset(data, targets)


class ListOpsTorchDataset(Dataset):
    def __init__(self, data: torch.Tensor, targets: torch.Tensor):
        self.data = data
        self.targets = targets

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> tuple:
        return self.data[idx], self.targets[idx]
