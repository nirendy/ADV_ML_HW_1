import os
from pathlib import Path
from typing import NamedTuple

from src.types import SPLIT


class PATHS:
    PROJECT_DIR = Path(__file__).parent.parent.resolve()
    DATA_DIR = PROJECT_DIR / 'data'
    RAW_DATA_DIR = DATA_DIR / 'raw'
    PREPROCESSED_DATA_DIR = DATA_DIR / 'preprocessed'

    RAW_IMDB_LRA_DIR = RAW_DATA_DIR / 'imdb_lra'
    RAW_WIKITEXT_DIR = RAW_DATA_DIR / 'wikitext_cache'
    PREPROCESSED_IMDB_LRA_DIR = PREPROCESSED_DATA_DIR / 'imdb_lra'

    CHECKPOINTS_DIR = PROJECT_DIR / 'checkpoints'

    TENSORBOARD_DIR = PROJECT_DIR / 'tensorboard'


class DATASETS_CONSTANTS:
    IMDB_LRA_NAME = 'imdb_reviews'
    WIKITEXT_DATASET_PATH = 'Salesforce/wikitext'
    WIKITEXT_NAME = 'wikitext-103-v1'

    WIKITEXT_TRAIN_SPLIT_SIZE = 25_000

    IMDB_LRA_SPLIT_NAMES = [SPLIT.TRAIN, SPLIT.TRAIN]


# enum for architecture names


class STEPS:
    WARMUP_STEPS = 20
    SAVE_STEP = 500
    LOG_STEP = 50
    EVAL_STEP = 1000
    PRINT_GRAPH = False


class FORMATS:
    TIME = "%Y%m%d_%H-%M-%S"
    LOGGER_FORMAT = '%(asctime)s - %(message)s'


class DDP:
    MASTER_PORT = os.environ.get('MASTER_PORT', '12355')
    MASTER_ADDR = 'localhost'
    BACKEND = 'nccl'
    SHUFFLE = True
    DROP_LAST = True
    NUM_WORKERS = 0


class IAddArgs(NamedTuple):
    with_parallel: bool
    partition: str = 'gpu-a100-killable'
    time: int = 1200
    singal: str = 'USR1@120'
    nodes: int = 1
    ntasks: int = 1
    mem: int = int(5e4)
    cpus_per_task: int = 1
    gpus: int = 2
    account: str = 'gpu-research'
    workspace = PATHS.PROJECT_DIR
    outputs_relative_path = PATHS.TENSORBOARD_DIR.relative_to(PATHS.PROJECT_DIR)
