import os
from pathlib import Path

import tensorflow_datasets as tfds
from datasets import load_dataset

data_dir = (Path('..') / 'data' / 'raw').resolve()

# Load the LRA IMDb reviews dataset
imdb_lra = tfds.load('imdb_reviews', data_dir=data_dir / 'imdb_lra')

os.environ['HF_HOME'] = str(data_dir / 'huggingface_cache')

# Load the Wikitext-103 dataset
wikitext_dataset = load_dataset("Salesforce/wikitext", "wikitext-103-v1", cache_dir=str(data_dir / 'wikitext_cache'))
