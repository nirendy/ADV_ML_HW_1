from pathlib import Path
from datasets import load_dataset

from src.consts import DATASETS_CONSTANTS
from src.datasets.text_dataset import TextDatasetFactory


class WikiTextDatasetFactory(TextDatasetFactory):

    @property
    def data_cache_dir(self) -> Path:
        return self.base_data_dir / 'raw' / 'wikitext_cache'

    def load_dataset(self):
        full_dataset = load_dataset("Salesforce/wikitext", "wikitext-103-v1", cache_dir=str(self.data_cache_dir))

        # reduce the dataset size for testing to WIKITEXT_TRAIN_SPLIT_SIZE
        full_dataset['train'] = full_dataset['train'].select(list(range(DATASETS_CONSTANTS.WIKITEXT_TRAIN_SPLIT_SIZE)))
        return full_dataset
