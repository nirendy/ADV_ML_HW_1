from transformers import BertTokenizer
from torch.utils.data import Dataset

from src.datasets.base_dataset import BaseDataset
import tensorflow_datasets as tfds


class IMDBlraDataset(BaseDataset):
    def __init__(self, phase_name: str):
        super().__init__()
        self.phase_name = phase_name
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    @property
    def data_dir(self):
        return self.base_data_dir / 'raw' / 'imdb_lra'

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size

    @property
    def num_classes(self) -> int:
        if self.phase_name == 'classification':
            return 2
        elif self.phase_name == 'autoregressive':
            return self.vocab_size
        else:
            raise ValueError(f'Invalid phase name: {self.phase_name}')

    def get_dataset(self, split: str) -> Dataset:
        imdb_lra_dataset = tfds.load('imdb_reviews', data_dir=self.data_dir)

        return IMDBlraDatasetTorchDataset(imdb_lra_dataset[split])


class IMDBlraDatasetTorchDataset(Dataset):
    def __init__(self, imdb_lra_dataset, tokenizer):
        self.imdb_lra_dataset = imdb_lra_dataset
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.imdb_lra_dataset)

    def __getitem__(self, idx):
        item = self.imdb_lra_dataset[idx]

        return {
            'text': self.tokenizer(
                item['text'],
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )['input_ids'].squeeze(),
            'label': item['label']
        }
