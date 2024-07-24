from pathlib import Path
from datasets import load_dataset
from src.datasets.text_dataset import TextDatasetFactory


class WikiTextDatasetFactory(TextDatasetFactory):

    @property
    def data_cache_dir(self) -> Path:
        return self.base_data_dir / 'raw' / 'wikitext_cache'

    def load_dataset(self):
        return load_dataset("Salesforce/wikitext", "wikitext-103-v1", cache_dir=str(self.data_cache_dir))
