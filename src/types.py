from typing import NamedTuple
from typing import NewType
from typing import Optional
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
    LSTM_COPY = 'lstm_copy'
    TRANSFORMER = 'transformer'
    TRANSFORMER_COPY = 'transformer_copy'
    S4 = 's4'
    S4_COPY = 's4_copy'


class DATASET(STREnum):
    IMDB = 'imdb'
    # LISTOPS = 'listops'
    WIKITEXT = 'wikitext'


class PHASE(STREnum):
    CLASSIFICATION = 'classification'
    AUTOREGRESSIVE = 'autoregressive'


class CONFIG_KEYS(STREnum):
    LSTM = 'lstm'
    TRANSFORMER = 'transformer'
    S4 = 's4'
    TRAINING = 'training'


class LR_SCHEDULER(STREnum):
    STEP = 'step'


class OPTIMIZER(STREnum):
    ADAM = 'adam'


class METRICS(STREnum):
    ACCURACY = 'accuracy'
    LOSS = 'loss'


IConfigName = NewType('IConfigName', str)


class IArgs(NamedTuple):
    config_name: IConfigName
    architecture: ARCH
    finetune_dataset: DATASET
    pretrain_dataset: Optional[DATASET]
    run_id: Optional[str]
