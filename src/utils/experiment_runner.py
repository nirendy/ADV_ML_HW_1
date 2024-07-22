import importlib
from typing import List
from typing import Optional
from typing import Tuple

import pandas as pd
from torch.utils.tensorboard import SummaryWriter
import time

from src.datasets.base_dataset import BaseDataset
from src.datasets.listops_dataset import ListOpsDataset
from src.datasets.mathqa_dataset import MathQADataset
from src.datasets.retrieval_dataset import RetrievalDataset
from src.models.architecture import Architecture
from src.models.lstm import LSTMArchitecture
from src.models.s4 import S4Architecture
from src.models.transformer import TransformerArchitecture
from src.trainer import Trainer
from src.utils.config_types import Config


def run_experiment(
        architectures: List[Architecture],
        pretrain_datasets: List[Optional[BaseDataset]],
        finetune_datasets: List[BaseDataset],
        trainer: Trainer,
        writer: SummaryWriter
) -> pd.DataFrame:
    results = []

    for architecture in architectures:
        for pretrain_dataset in pretrain_datasets:
            for finetune_dataset in finetune_datasets:
                pretrain_name = pretrain_dataset.__class__.__name__ if pretrain_dataset else "None"
                run_id = '.'.join([
                    writer.get_logdir(),
                    architecture.__class__.__name__,
                    finetune_dataset.__class__.__name__,
                    pretrain_name,
                ])
                metrics = trainer.train_and_evaluate_model(
                    architecture, pretrain_dataset, finetune_dataset, writer, run_id
                )

                # Record results
                result = {
                    'Architecture': architecture.__class__.__name__,
                    'Pretrain Dataset': pretrain_name,
                    'Finetune Dataset': finetune_dataset.__class__.__name__,
                }
                result.update(metrics)
                results.append(result)

    writer.close()

    # Convert results to DataFrame and display
    df = pd.DataFrame(results)
    return df


def load_config(config_name: str) -> Config:
    # Dynamically import the config
    config_module = importlib.import_module(f'src.configs.{config_name}')
    return config_module.config


def init_experiment(config_name: str) -> Tuple[
    List[Architecture], List[Optional[BaseDataset]], List[BaseDataset], Trainer, SummaryWriter]:
    config = load_config(config_name)

    writer = SummaryWriter(log_dir=f"tensorboard/{config_name}/{time.strftime('%Y%m%d-%H%M%S')}")

    # Initialize architectures
    architectures = [
        LSTMArchitecture(config['lstm']),
        TransformerArchitecture(config['transformer']),
        S4Architecture(config['s4'])
    ]

    # Initialize datasets
    # pretrain_datasets = [None, MathQABaseDataset(), RetrievalBaseDataset()]
    pretrain_datasets = [None]
    finetune_datasets = [ListOpsDataset()]

    trainer = Trainer(config['training'])

    return architectures, pretrain_datasets, finetune_datasets, trainer, writer
