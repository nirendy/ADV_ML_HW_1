import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.consts import DDP
from src.consts import FORMATS
from src.consts import PATHS
from src.types import ARCH
from src.types import DATASET
from src.consts import DATASETS_CONSTANTS
from src.types import PHASE
from src.types import SPLIT
from src.consts import STEPS
from src.datasets.base_dataset import BaseDataset
from src.models.architecture import Architecture
from src.utils.experiment_runner import get_arch_by_name
from src.utils.experiment_runner import get_config_name_by_arch
from src.utils.experiment_runner import get_dataset_by_name
from src.utils.experiment_runner import load_config
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = DDP.MASTER_ADDR
    os.environ['MASTER_PORT'] = DDP.MASTER_PORT
    dist.init_process_group(DDP.BACKEND, rank=rank, world_size=world_size)


class Trainer:
    def __init__(
            self,
            config_name: str,
            architecture: ARCH,
            finetune_dataset: DATASET,
            pretrain_dataset: Optional[DATASET],
            run_id: Optional[str] = None
    ):
        self._run_id = run_id or time.strftime(FORMATS.TIME)
        self._config_name = config_name
        self._config = load_config(config_name)
        self._arch_name = architecture
        self._finetune_dataset = finetune_dataset
        self._pretrain_dataset = pretrain_dataset

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)

        self.best_loss = float('inf')

    def configure_logging(self):
        (PATHS.LOGS_DIR / self.relative_path).mkdir(parents=True, exist_ok=True)
        if self.is_master_process:
            self.logger.addHandler(logging.StreamHandler())
            self.logger.addHandler(logging.FileHandler(PATHS.LOGS_DIR / self.relative_path / f'{self._run_id}.log'))
            self.dump_config()
        else:
            self.logger.addHandler(logging.NullHandler())

    @property
    def arch_config(self) -> Dict[str, Any]:
        return self._config.get(get_config_name_by_arch(self._arch_name).value)

    @property
    def relative_path(self) -> str:
        return '/'.join([
            self._config_name,
            self._arch_name,
            *(['pre_' + self._pretrain_dataset] if self._pretrain_dataset else []),
            self._finetune_dataset,
            self._run_id
        ])

    @property
    def training_config(self) -> Dict[str, Any]:
        return self._config['training']

    def dump_config(self):
        path = PATHS.LOGS_DIR / self.relative_path / f'config_{time.strftime(FORMATS.TIME)}.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self._config, f, indent=4)

    def log(self, msg: str):
        path = PATHS.LOGS_DIR / self.relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'a') as f:
            f.write(msg + '\n')

    def get_loss_fn(self, phase_name: PHASE) -> nn.Module:
        if phase_name == PHASE.CLASSIFICATION:
            return nn.CrossEntropyLoss()
        elif phase_name == PHASE.AUTOREGRESSIVE:
            return nn.CrossEntropyLoss(ignore_index=-1)  # Use ignore_index to ignore padding tokens
        else:
            raise ValueError(f'Invalid phase name: {phase_name}')

    def save_checkpoint(
            self,
            architecture: Architecture,
            optimizer: optim.Optimizer,
            epoch: int,
            step: int,
            during_pretraining: bool
    ):
        path = PATHS.CHECKPOINTS_DIR / self.relative_path / f'{step}.pth'
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            'epoch': epoch,
            'step': step,
            'model_state_dict': architecture.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_loss': self.best_loss,
            'during_pretraining': during_pretraining
        }
        torch.save(checkpoint, path)

        self.logger.info(f"Checkpoint saved at {path}")

    def load_checkpoint(self, path: Path, architecture: Architecture, optimizer: optim.Optimizer):
        if path.exists():
            checkpoint = torch.load(path)
            architecture.model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.best_loss = checkpoint['best_loss']
            self.logger.info(f"Checkpoint loaded from {path}")
            return checkpoint['epoch'], checkpoint['during_pretraining']
        else:
            self.logger.info(f"No checkpoint found at {path}")
            return 0

    @property
    def is_pretraining(self) -> bool:
        return self._pretrain_dataset is not None

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def world_size(self) -> int:
        return self._world_size

    @property
    def is_master_process(self) -> bool:
        return self._rank == 0

    @property
    def is_distributed(self) -> bool:
        return self._world_size > 1

    def get_optimizer(self, model: torch.nn.Module) -> optim.Optimizer:
        return optim.Adam(model.parameters(), lr=self.training_config['learning_rate'])

    def train_and_evaluate_model(self, rank=0, world_size=1) -> Dict[str, Any]:
        self._rank = rank or 0
        self._world_size = world_size or 1

        if self.is_distributed:
            setup(rank, world_size)
            device = torch.device(rank)
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.configure_logging()

        architecture = get_arch_by_name(self._arch_name)(self.arch_config)
        pretrain_dataset = (
            get_dataset_by_name(self._pretrain_dataset)(PHASE.AUTOREGRESSIVE)
            if self.is_pretraining
            else None
        )
        finetune_dataset = get_dataset_by_name(self._finetune_dataset)(PHASE.CLASSIFICATION)
        writer = SummaryWriter(log_dir=str(PATHS.TENSORBOARD_DIR / self.relative_path))

        architecture.initialize_model(dataset=finetune_dataset)
        self.logger.info(f"params count: {sum(p.numel() for p in architecture.model.parameters())}")
        if STEPS.PRINT_GRAPH:
            sample_input_tensor = next(iter(finetune_dataset.get_train_dataset()))[0].unsqueeze(
                0)  # Adjust based on your dataset's sample shape
            writer.add_graph(architecture.model, sample_input_tensor)

        optimizer = None
        if (checkpoint_path := self.get_latest_chkpt()) is not None:
            optimizer = self.get_optimizer(architecture.model)
            epoch, during_pretraining = (
                self.load_checkpoint(checkpoint_path, architecture, optimizer)
            )
            step = int(checkpoint_path.stem)
        else:
            epoch = 0
            step = 0
            during_pretraining = self.is_pretraining

        architecture.model.to(device)
        # Set random seeds for reproducibility
        set_seed(self.training_config['seed'])

        if during_pretraining:
            self._train_model(
                architecture=architecture,
                dataset_wrapper=pretrain_dataset,
                writer=writer,
                device=device,
                start_epoch=epoch,
                step=step
            )
            step = 0
            optimizer = None
        self._train_model(
            architecture=architecture,
            dataset_wrapper=finetune_dataset,
            writer=writer,
            optimizer=optimizer,
            device=device,
            start_epoch=epoch,
            step=step
        )

        metrics = self._evaluate_model(architecture, finetune_dataset, SPLIT.TEST, device)
        return metrics

    def _train_model(
            self, architecture: Architecture,
            dataset_wrapper: BaseDataset,
            writer: SummaryWriter,
            optimizer: Optional[optim.Optimizer],
            device: torch.device,
            start_epoch: int,
            step: int
    ) -> None:
        dataset = dataset_wrapper.get_dataset(SPLIT.TRAIN)
        if self.is_distributed:
            sampler = DistributedSampler(
                dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=DDP.SHUFFLE,
                drop_last=DDP.DROP_LAST
            )
            data_loader = DataLoader(
                dataset,
                batch_size=self.training_config['batch_size'],
                sampler=sampler,
                num_workers=DDP.NUM_WORKERS,
                # collate_fn=dataset.collate_fn
            )
        else:
            data_loader = DataLoader(
                dataset,
                batch_size=self.training_config['batch_size'],
                shuffle=True
            )

        if optimizer is None:
            optimizer = self.get_optimizer(architecture.model)
        loss_fn = self.get_loss_fn(dataset_wrapper.phase_name)
        architecture.model.train()
        for epoch in range(start_epoch, self.training_config['epochs']):
            for i, data in enumerate(data_loader):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = architecture.forward(inputs)

                if dataset_wrapper.phase_name == PHASE.CLASSIFICATION:
                    loss = loss_fn(outputs, labels)
                elif dataset_wrapper.phase_name == PHASE.AUTOREGRESSIVE:
                    # Shift inputs to create target for next token prediction
                    targets = inputs[:, 1:].contiguous().view(-1)
                    outputs = outputs[:, :-1].contiguous().view(-1, outputs.size(-1))
                    loss = loss_fn(outputs, targets)
                else:
                    raise ValueError(f'Invalid phase name: {dataset_wrapper.phase_name}')

                loss.backward()
                optimizer.step()
                step += 1

                if self.is_master_process:
                    # Log the training loss
                    if step % STEPS.LOG_STEP == 0:
                        writer.add_scalar(
                            '/'.join([
                                self.relative_path,
                                self._run_id,
                                dataset_wrapper.phase_name,
                                'Loss'
                            ]),
                            loss.item(),
                            step
                        )
                        self.best_loss = min(self.best_loss, loss.item())
                        self.logger.info(
                            f'{dataset_wrapper.phase_name} Epoch [{epoch + 1}/{self.training_config["epochs"]}], '
                            f'Step [{step}/{len(data_loader)}], Loss: {loss.item():.4f}'
                        )

                    if step >= STEPS.WARMUP_STEPS and step % STEPS.SAVE_STEP == 0:
                        self.save_checkpoint(
                            architecture, optimizer, epoch,
                            step=step,
                            during_pretraining=dataset_wrapper.phase_name == PHASE.AUTOREGRESSIVE
                        )
                    if step >= STEPS.WARMUP_STEPS and step % STEPS.EVAL_STEP == 0:
                        metrics = self._evaluate_model(architecture, dataset_wrapper, SPLIT.TEST, device)
                        for key, value in metrics.items():
                            writer.add_scalar(
                                '/'.join([
                                    self.relative_path,
                                    self._run_id,
                                    dataset_wrapper.phase_name,
                                    key
                                ]),
                                value,
                                step,
                            )
                        self.logger.info(f'Test Metrics: {metrics}')

    def _evaluate_model(
            self,
            architecture: Architecture,
            dataset: BaseDataset,
            split: SPLIT,
            device: torch.device,
    ) -> Dict[str, float]:
        data_loader = DataLoader(
            dataset.get_dataset(split),
            batch_size=self.training_config['batch_size']
        )

        test_loss = 0
        correct = 0
        total = 0
        loss_fn = self.get_loss_fn(PHASE.CLASSIFICATION)
        with torch.no_grad():
            for data in data_loader:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = architecture.forward(inputs)
                loss = loss_fn(outputs, labels)
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        return {'Test Loss': test_loss / len(data_loader), 'Test Accuracy': accuracy}

    def get_latest_chkpt(self) -> Optional[Path]:
        dir_path = PATHS.CHECKPOINTS_DIR / self.relative_path
        if dir_path.exists():
            return max(dir_path.iterdir(), key=lambda x: int(x.stem))
        return None


# Set random seeds for reproducibility
def set_seed(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
