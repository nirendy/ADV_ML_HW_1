import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from abc import ABC, abstractmethod
import torch.nn.functional as F
import pandas as pd

from datasets.base_dataset import Dataset
from models.architecture import Architecture




# Training Function

def train_and_evaluate_model(architecture_cls, pretrain_dataset_cls, finetune_dataset_cls):
    # Initialize objects
    architecture = architecture_cls()
    pretrain_dataset = pretrain_dataset_cls()
    finetune_dataset = finetune_dataset_cls()

    # Load data
    pretrain_dataset.load_data()
    pretrain_loader = pretrain_dataset.get_train_loader()
    finetune_dataset.load_data()
    finetune_loader = finetune_dataset.get_train_loader()
    test_loader = finetune_dataset.get_test_loader()

    # Initialize and train model
    architecture.initialize_model()
    if pretrain_loader:
        architecture.train_model(pretrain_loader)
    architecture.train_model(finetune_loader)

    # Evaluate model
    architecture.evaluate_model(test_loader)
    return architecture.get_metrics()


# Reporting Function

def run_experiment(architectures, pretrain_datasets, finetune_datasets):
    results = []

    for architecture_cls in architectures:
        for pretrain_dataset_cls in pretrain_datasets:
            for finetune_dataset_cls in finetune_datasets:
                metrics = train_and_evaluate_model(architecture_cls, pretrain_dataset_cls, finetune_dataset_cls)

                # Record results
                result = {
                    'Architecture': architecture_cls.__name__,
                    'Pretrain Dataset': pretrain_dataset_cls.__name__,
                    'Finetune Dataset': finetune_dataset_cls.__name__,
                }
                result.update(metrics)
                results.append(result)

    # Convert results to DataFrame and display
    df = pd.DataFrame(results)
    return df
