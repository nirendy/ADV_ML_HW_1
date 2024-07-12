import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from typing import List, Optional, Dict, Any

from datasets.base_dataset import Dataset
from models.architecture import Architecture


# Training Function

def train_and_evaluate_model(
        architecture: Architecture,
        pretrain_dataset: Optional[Dataset],
        finetune_dataset: Dataset,
        writer: SummaryWriter,
        run_id: str
) -> Dict[str, Any]:
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


# Reporting Function

def run_experiment(
        architectures: List[Architecture],
        pretrain_datasets: List[Optional[Dataset]],
        finetune_datasets: List[Dataset]
) -> pd.DataFrame:
    results = []
    writer = SummaryWriter(log_dir="runs")

    for architecture in architectures:
        for pretrain_dataset in pretrain_datasets:
            for finetune_dataset in finetune_datasets:
                pretrain_name = pretrain_dataset.__class__.__name__ if pretrain_dataset else "None"
                run_id = f"{architecture.__class__.__name__}_{pretrain_name}_{finetune_dataset.__class__.__name__}"
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
