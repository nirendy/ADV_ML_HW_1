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
from src.types import ARCH
from src.types import CONFIG_KEYS
from src.types import DATASET
from src.types import IConfigName
from src.utils.config_types import Config


# def run_experiment(
#         architectures: List[Architecture],
#         pretrain_datasets: List[Optional[BaseDataset]],
#         finetune_datasets: List[BaseDataset],
#         trainer: Trainer,
#         writer: SummaryWriter
# ) -> pd.DataFrame:
#     results = []
#
#     for architecture in architectures:
#         for pretrain_dataset in pretrain_datasets:
#             for finetune_dataset in finetune_datasets:
#                 pretrain_name = pretrain_dataset.__class__.__name__ if pretrain_dataset else "None"
#                 run_id = '.'.join([
#                     writer.get_logdir(),
#                     architecture.__class__.__name__,
#                     finetune_dataset.__class__.__name__,
#                     pretrain_name,
#                 ])
#                 metrics = trainer.train_and_evaluate_model(
#                     architecture, pretrain_dataset, finetune_dataset, writer, run_id
#                 )
#
#                 # Record results
#                 result = {
#                     'Architecture': architecture.__class__.__name__,
#                     'Pretrain Dataset': pretrain_name,
#                     'Finetune Dataset': finetune_dataset.__class__.__name__,
#                 }
#                 result.update(metrics)
#                 results.append(result)
#
#     writer.close()
#
#     # Convert results to DataFrame and display
#     df = pd.DataFrame(results)
#     return df


def load_config(config_name: IConfigName) -> Config:
    # Dynamically import the config
    config_module = importlib.import_module(f'src.configs.{config_name}')
    return config_module.config


def get_arch_by_name(arch_name: ARCH) -> Type[Architecture]:
    if arch_name == ARCH.LSTM:
        return LSTMArchitecture
    elif arch_name == ARCH.TRANSFORMER:
        return TransformerArchitecture
    elif arch_name == ARCH.S4:
        return S4Architecture
    elif arch_name == ARCH.S4_COPY:
        return S4CopyArchitecture
    else:
        raise ValueError(f'Invalid architecture name: {arch_name}')


def get_config_name_by_arch(arch_name: ARCH) -> CONFIG_KEYS:
    if arch_name == ARCH.LSTM:
        return CONFIG_KEYS.LSTM
    elif arch_name == ARCH.TRANSFORMER:
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
