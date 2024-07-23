from datasets import DatasetDict
from transformers import BertTokenizer
from torch.utils.data import Dataset
from typing_extensions import assert_never

from src.datasets.base_dataset import BaseDataset
from src.types import PHASE


class IMDBlraDataset(BaseDataset):

    @property
    def data_dir(self):
        return self.base_data_dir / 'preprocessed' / 'imdb_lra'

    def get_dataset(self, split: str) -> Dataset:
        imdb_lra_dataset = DatasetDict.load_from_disk(self.data_dir)
        return IMDBlraDatasetTorchDataset(imdb_lra_dataset[split], self.tokenizer, self.phase_name)


class IMDBlraDatasetTorchDataset(Dataset):
    def __init__(self, imdb_lra_dataset, tokenizer, phase_name: PHASE):
        self.imdb_lra_dataset = imdb_lra_dataset
        self.tokenizer = tokenizer
        self.phase_name = phase_name

    def __len__(self):
        return len(self.imdb_lra_dataset)

    def __getitem__(self, idx):
        item = self.imdb_lra_dataset[idx]
        encoding = (
            self.tokenizer(
                item['text'].decode(),
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )['input_ids'].squeeze(0)
        )
        if self.phase_name == PHASE.CLASSIFICATION:
            return encoding, item['label']
        elif self.phase_name == PHASE.AUTOREGRESSIVE:
            labels = encoding.clone()
            labels[:-1] = encoding[1:]
            labels[-1] = self.tokenizer.pad_token_id
            return (
                encoding,
                labels
            )
        assert_never(self.phase_name)
