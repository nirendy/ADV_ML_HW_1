import importlib
import random
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter

from src.datasets.base_dataset import BaseDataset
from src.datasets.listops_dataset import ListOpsBaseDataset
from src.datasets.mathqa_dataset import MathQABaseDataset
from src.datasets.retrieval_dataset import RetrievalBaseDataset
from src.models.architecture import Architecture
from src.models.lstm import LSTMArchitecture
from src.models.s4 import S4Architecture
from src.models.transformer import TransformerArchitecture
from src.utils.config_types import Config


# Training Function
def train_and_evaluate_model(
        architecture: Architecture,
        pretrain_dataset: Optional[BaseDataset],
        finetune_dataset: BaseDataset,
        writer: SummaryWriter,
        run_id: str
) -> Dict[str, Any]:
    try:
        # Load data
        if pretrain_dataset:
            pretrain_dataset.load_data()
            pretrain_loader = pretrain_dataset.get_train_loader()
        else:
            pretrain_loader = None

        finetune_dataset.load_data()
        finetune_loader = finetune_dataset.get_train_loader()
        test_loader = finetune_dataset.get_test_loader()

        # Initialize and train model
        architecture.initialize_model()
        if pretrain_loader:
            architecture.train_model(pretrain_loader)
        architecture.train_model(finetune_loader)

        # Evaluate model
        metrics = architecture.evaluate_model(test_loader)
        # metrics = architecture.get_metrics()

        # Log metrics to TensorBoard
        for key, value in metrics.items():
            writer.add_scalar(f"{run_id}/{key}", value)

        return metrics

    except Exception as e:
        architecture.logger.error(f"Error during train and evaluate: {e}")
        return {}


# Reporting Function
def run_experiment(
        architectures: List[Architecture],
        pretrain_datasets: List[Optional[BaseDataset]],
        finetune_datasets: List[BaseDataset],
        config_name: str
) -> pd.DataFrame:
    results = []
    writer = SummaryWriter(log_dir=f"runs/{config_name}")

    for architecture in architectures:
        for pretrain_dataset in pretrain_datasets:
            for finetune_dataset in finetune_datasets:
                pretrain_name = pretrain_dataset.__class__.__name__ if pretrain_dataset else "None"
                run_id = '.'.join([
                    config_name,
                    architecture.__class__.__name__,
                    pretrain_name,
                    finetune_dataset.__class__.__name__
                ])
                metrics = train_and_evaluate_model(architecture, pretrain_dataset, finetune_dataset, writer, run_id)

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


# Set random seeds for reproducibility
def set_seed(seed: int):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def init_experiment(config_name: str) -> Tuple[List[Architecture], List[Optional[BaseDataset]], List[BaseDataset], str]:
    # Dynamically import the config
    config_module = importlib.import_module(f'configs.{config_name}')
    config: Config = config_module.config

    # Set random seeds for reproducibility
    set_seed(42)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize architectures
    architectures = [
        LSTMArchitecture(config['lstm'], config['training']),
        # TransformerArchitecture(config['transformer'], config['training']),
        # S4Architecture(config['s4'], config['training'])
    ]

    # Initialize datasets
    pretrain_datasets = [None, MathQABaseDataset(), RetrievalBaseDataset()]
    finetune_datasets = [ListOpsBaseDataset()]

    return architectures, pretrain_datasets, finetune_datasets, config_name
