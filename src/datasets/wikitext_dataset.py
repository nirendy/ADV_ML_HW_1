import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import BertTokenizer

from src.datasets.base_dataset import BaseDataset


class WikiTextDataset(BaseDataset):
    def __init__(self, phase_name: str):
        super().__init__()
        self.phase_name = phase_name
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    @property
    def data_dir(self):
        return self.base_data_dir / 'raw' / 'wikitext_cache'

    @property
    def vocab_size(self) -> int:
        vocab_size_file = self.data_dir / 'vocab_size.pt'
        return torch.load(vocab_size_file)

    @property
    def num_classes(self) -> int:
        return 10

    def get_dataset(self, split: str) -> Dataset:
        wikitext_dataset = load_dataset("Salesforce/wikitext", "wikitext-103-v1", cache_dir=str(self.data_dir))

        return WikiTextDatasetTorchDataset(wikitext_dataset[split], self.tokenizer)


class WikiTextDatasetTorchDataset(Dataset):
    def __init__(self, wikitext_dataset, tokenizer):
        self.wikitext_dataset = wikitext_dataset
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.wikitext_dataset)

    def __getitem__(self, idx):
        item = self.wikitext_dataset[idx]

        return {
            'text': self.tokenizer(
                item['text'],
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )['input_ids'].squeeze(),
            'label': item['label']
        }
