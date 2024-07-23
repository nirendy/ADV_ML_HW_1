import importlib
from typing import Type
from typing_extensions import assert_never

from src.datasets.base_dataset import BaseDataset
from src.datasets.imdb_lra_dataset import IMDBlraDataset
from src.datasets.listops_dataset import ListOpsDataset
from src.datasets.wikitext_dataset import WikiTextDataset
from src.models.architecture import Architecture
from src.models.lstm import LSTMArchitecture
from src.models.s4 import S4Architecture
from src.models.s4_copy import S4CopyArchitecture
from src.models.transformer import TransformerArchitecture
from src.models.transformer_copy import TransformerCopyArchitecture
from src.types import ARCH
from src.types import CONFIG_KEYS
from src.types import DATASET
from src.types import IConfigName
from src.utils.config_types import Config


def load_config(config_name: IConfigName) -> Config:
    # Dynamically import the config
    config_module = importlib.import_module(f'src.configs.{config_name}')
    return config_module.config


def get_arch_by_name(arch_name: ARCH) -> Type[Architecture]:
    if arch_name == ARCH.LSTM:
        return LSTMArchitecture
    elif arch_name == ARCH.TRANSFORMER:
        return TransformerArchitecture
    elif arch_name == ARCH.TRANSFORMER_COPY:
        return TransformerCopyArchitecture
    elif arch_name == ARCH.S4:
        return S4Architecture
    elif arch_name == ARCH.S4_COPY:
        return S4CopyArchitecture
    else:
        raise ValueError(f'Invalid architecture name: {arch_name}')


def get_config_name_by_arch(arch_name: ARCH) -> CONFIG_KEYS:
    if arch_name == ARCH.LSTM:
        return CONFIG_KEYS.LSTM
    elif arch_name == ARCH.TRANSFORMER or arch_name == ARCH.TRANSFORMER_COPY:
        return CONFIG_KEYS.TRANSFORMER
    elif arch_name == ARCH.S4 or arch_name == ARCH.S4_COPY:
        return CONFIG_KEYS.S4
    assert_never(arch_name)


def get_dataset_by_name(dataset_name: DATASET) -> Type[BaseDataset]:
    if dataset_name == DATASET.IMDB:
        return IMDBlraDataset
    elif dataset_name == DATASET.LISTOPS:
        return ListOpsDataset
    elif dataset_name == DATASET.WIKITEXT:
        return WikiTextDataset
    assert_never(dataset_name)
