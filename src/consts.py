from pathlib import Path
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
    LOGS_DIR = PROJECT_DIR / 'logs'

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


class DDP:
    MASTER_PORT = '12355'
    MASTER_ADDR = 'localhost'
    BACKEND = 'nccl'
    SHUFFLE = True
    DROP_LAST = True
    NUM_WORKERS = 0
