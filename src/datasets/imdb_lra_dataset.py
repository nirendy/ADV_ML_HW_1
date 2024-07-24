from datasets import DatasetDict

from src.datasets.text_dataset import TextDatasetFactory


class IMDBlraDatasetFactory(TextDatasetFactory):
    with_decode = True

    @property
    def data_cache_dir(self):
        return self.base_data_dir / 'preprocessed' / 'imdb_lra'

    def load_dataset(self):
        return DatasetDict.load_from_disk(self.data_cache_dir)
