from typing import TypeVar
from enum import Enum


class STREnum(str, Enum):
    def __str__(self):
        return str(self.value)


class SPLIT(STREnum):
    TRAIN = 'train'
    TEST = 'test'
    VALIDATION = 'validation'
    UNSUPERVISED = 'unsupervised'


class ARCH(STREnum):
    LSTM = 'lstm'
    TRANSFORMER = 'transformer'
    S4 = 's4'
    S4_COPY = 's4_copy'


class DATASET(STREnum):
    IMDB = 'imdb'
    LISTOPS = 'listops'
    WIKITEXT = 'wikitext'


class PHASE(STREnum):
    CLASSIFICATION = 'classification'
    AUTOREGRESSIVE = 'autoregressive'


class CONFIG_KEYS(STREnum):
    LSTM = 'lstm'
    TRANSFORMER = 'transformer'
    S4 = 's4'
    TRAINING = 'training'


IConfigName: TypeVar = TypeVar('IConfigName', str, str)
