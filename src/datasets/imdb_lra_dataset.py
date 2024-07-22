from datasets import DatasetDict
from transformers import BertTokenizer
from torch.utils.data import Dataset
from src.datasets.base_dataset import BaseDataset


class IMDBlraDataset(BaseDataset):
    def __init__(self, phase_name: str):
        super().__init__()
        self.phase_name = phase_name
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    @property
    def data_dir(self):
        return self.base_data_dir / 'preprocessed' / 'imdb_lra'

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
        imdb_lra_dataset = DatasetDict.load_from_disk(self.data_dir)

        return IMDBlraDatasetTorchDataset(imdb_lra_dataset[split], self.tokenizer)


class IMDBlraDatasetTorchDataset(Dataset):
    def __init__(self, imdb_lra_dataset, tokenizer):
        self.imdb_lra_dataset = imdb_lra_dataset
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.imdb_lra_dataset)

    def __getitem__(self, idx):
        item = self.imdb_lra_dataset[idx]

        return (
            self.tokenizer(
                item['text'].decode(),
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )['input_ids'].squeeze(),
            item['label']
        )
