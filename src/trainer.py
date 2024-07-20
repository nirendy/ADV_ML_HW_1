import random
from typing import Any
from typing import Dict
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.datasets.base_dataset import BaseDataset
from src.datasets.base_dataset import BaseDataset
from src.models.architecture import Architecture
from src.utils.config_types import TrainingConfig


class Trainer:
    def __init__(self, training_config: TrainingConfig):
        self.training_config = training_config

    def get_loss_fn(self) -> nn.Module:
        return nn.CrossEntropyLoss()

    def train_and_evaluate_model(
            self,
            architecture: Architecture,
            pretrain_dataset: Optional[BaseDataset],
            finetune_dataset: BaseDataset,
            writer: SummaryWriter,
            run_id: str
    ) -> Dict[str, Any]:
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Set random seeds for reproducibility
        set_seed(self.training_config['seed'])
        architecture.initialize_model(dataset=finetune_dataset)
        architecture.model.to(device)

        if pretrain_dataset:
            self._train_model(architecture, pretrain_dataset, writer, 'Pretrain', device, run_id)
        self._train_model(architecture, finetune_dataset, writer, 'Finetune', device, run_id)

        metrics = self._evaluate_model(architecture, finetune_dataset)
        return metrics

    def _train_model(
            self, architecture: Architecture, dataset: BaseDataset, writer: SummaryWriter, phase_name: str,
            device: torch.device, run_id: str
    ) -> None:
        data_loader = DataLoader(
            dataset.get_dataset('train'),
            batch_size=self.training_config['batch_size'],
            shuffle=True
        )
        optimizer = optim.Adam(architecture.model.parameters(), lr=self.training_config['learning_rate'])
        loss_fn = self.get_loss_fn()
        architecture.model.train()
        for epoch in range(self.training_config['epochs']):
            for i, data in enumerate(data_loader):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = architecture.forward(inputs)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()

                # Log the training loss
                if i % 10 == 0:
                    step = epoch * len(data_loader) + i
                    writer.add_scalar(f'{run_id}/{phase_name}/Loss', loss.item(), step)
                    print(
                        f'{phase_name} Epoch [{epoch + 1}/{self.training_config["epochs"]}], '
                        f'Step [{i + 1}/{len(data_loader)}], Loss: {loss.item():.4f}'
                    )

    def _evaluate_model(self, architecture: Architecture, dataset: BaseDataset) -> Dict[str, float]:
        data_loader = DataLoader(dataset.get_dataset('test'), batch_size=self.training_config['batch_size'])
        test_loss = 0
        correct = 0
        total = 0
        loss_fn = self.get_loss_fn()
        with torch.no_grad():
            for data in data_loader:
                inputs, labels = data
                outputs = architecture.forward(inputs)
                loss = loss_fn(outputs, labels)
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        return {'Test Loss': test_loss / len(data_loader), 'Test Accuracy': accuracy}


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
