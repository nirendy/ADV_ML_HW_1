from abc import abstractmethod
from pathlib import Path

from datasets import Dataset
from torch.utils.data import Dataset as TorchDataset
from transformers import BertTokenizer
from typing_extensions import assert_never

from src.datasets.base_dataset import DatasetFactory
from src.types import PHASE
from src.types import SPLIT


class TextDataset(TorchDataset):
    def __init__(
            self,
            dataset: Dataset,
            tokenizer: BertTokenizer,
            phase_name: PHASE,
            with_decode: bool
    ):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.phase_name = phase_name
        self.with_decode = with_decode

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        item_text = item['text']
        if self.with_decode:
            item_text = item_text.decode()
        encoding = (
            self.tokenizer(
                item_text,
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


class TextDatasetFactory(DatasetFactory):
    with_decode = False

    def __init__(self, phase_name: PHASE):
        super().__init__()
        self.phase_name = phase_name
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    @property
    @abstractmethod
    def data_cache_dir(self) -> Path:
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

    def get_dataset(self, split: SPLIT) -> TorchDataset:
        return TextDataset(
            dataset=self.load_dataset()[split],
            tokenizer=self.tokenizer,
            phase_name=self.phase_name,
            with_decode=self.with_decode
        )

    @abstractmethod
    def load_dataset(self) -> Dataset:
        pass
