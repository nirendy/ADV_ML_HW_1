import os
from pathlib import Path

import tensorflow_datasets as tfds
from datasets import load_dataset, Dataset, DatasetDict

data_dir = (Path('..') / 'data' / 'raw').resolve()
out_dir = (Path('..') / 'data' / 'preprocessed').resolve()
# Load the LRA IMDb reviews dataset
imdb_lra = tfds.load('imdb_reviews', data_dir=data_dir / 'imdb_lra')

# Load the Wikitext-103 dataset
wikitext_dataset = load_dataset("Salesforce/wikitext", "wikitext-103-v1", cache_dir=str(data_dir / 'wikitext_cache'))

imdb_lra_dataset_dict = DatasetDict({
    'train': Dataset.from_list(list(tfds.as_numpy(imdb_lra['train']))),
    'test': Dataset.from_list(list(tfds.as_numpy(imdb_lra['test']))),
    'unsupervised': Dataset.from_list(list(tfds.as_numpy(imdb_lra['unsupervised']))),
})
imdb_lra_dataset_dict.save_to_disk(out_dir / 'imdb_lra')

# loaded_dataset = DatasetDict.load_from_disk(out_dir / 'imdb_lra')
