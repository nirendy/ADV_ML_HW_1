import os
from pathlib import Path

import tensorflow_datasets as tfds
from datasets import load_dataset, Dataset, DatasetDict

from src.consts import PATHS, DATASETS_CONSTANTS

# Load the LRA IMDb reviews dataset
imdb_lra = tfds.load(DATASETS_CONSTANTS.IMDB_LRA_NAME, data_dir=PATHS.RAW_IMDB_LRA_DIR)

# Load the Wikitext-103 dataset
wikitext_dataset = load_dataset(
    DATASETS_CONSTANTS.WIKITEXT_DATASET_PATH,
    DATASETS_CONSTANTS.WIKITEXT_NAME,
    cache_dir=str(PATHS.RAW_WIKITEXT_DIR)
)

imdb_lra_dataset_dict = DatasetDict({
    split_name: Dataset.from_list(list(tfds.as_numpy(imdb_lra[split_name])))
    for split_name in DATASETS_CONSTANTS.IMDB_LRA_SPLIT_NAMES
})
imdb_lra_dataset_dict.save_to_disk(PATHS.PREPROCESSED_IMDB_LRA_DIR)
