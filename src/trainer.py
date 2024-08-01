import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Optional
from typing import TypedDict
from typing_extensions import assert_never

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from src.consts import DDP
from src.consts import FORMATS
from src.consts import PATHS
from src.consts import STEPS
from src.datasets.text_dataset import TextDatasetFactory
from src.models.architecture import Architecture
from src.types import ARCH
from src.types import CONFIG_KEYS
from src.types import DATASET
from src.types import IConfigName
from src.types import LR_SCHEDULER
from src.types import METRICS
from src.types import OPTIMIZER
from src.types import PHASE
from src.types import SPLIT
from src.utils.config_types import TrainingConfig
from src.utils.experiment_runner import construct_experiment_name
from src.utils.experiment_runner import create_run_id
from src.utils.experiment_runner import get_arch_by_name
from src.utils.experiment_runner import get_config_key_by_arch
from src.utils.experiment_runner import get_text_dataset_factory_by_name
from src.utils.experiment_runner import load_config
from src.utils.experiment_runner import params_count_report


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = DDP.MASTER_ADDR
    os.environ['MASTER_PORT'] = DDP.MASTER_PORT
    dist.init_process_group(DDP.BACKEND, rank=rank, world_size=world_size)


class Checkpoint(TypedDict):
    epoch: int
    total_steps: int
    model_state_dict: Dict[str, Any]
    optimizer_state_dict: Dict[str, Any]
    best_loss: float
    during_pretraining: bool


class Trainer:
    def __init__(
            self,
            config_name: IConfigName,
            architecture: ARCH,
            finetune_dataset: DATASET,
            pretrain_dataset: Optional[DATASET],
            run_id: Optional[str] = None
    ):
        self._run_id = create_run_id(run_id)
        self._config_name = config_name
        self._config = load_config(config_name)
        self._arch_name = architecture
        self._finetune_dataset = finetune_dataset
        self._pretrain_dataset = pretrain_dataset

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)
        # add time prefix to log
        formatter = logging.Formatter(FORMATS.LOGGER_FORMAT)
        for handler in self.logger.handlers:
            handler.setFormatter(formatter)

        self.best_loss = float('inf')
        self.total_steps = 0
        self.early_stopping_counter = 0

    def configure_logging(self):
        (PATHS.TENSORBOARD_DIR / self.relative_path).mkdir(parents=True, exist_ok=True)
        if self.is_master_process:
            self.logger.addHandler(logging.StreamHandler())
            self.logger.addHandler(
                logging.FileHandler(PATHS.TENSORBOARD_DIR / self.relative_path / f'{self._run_id}.log'))
            self.dump_config()
        else:
            self.logger.addHandler(logging.NullHandler())

    @property
    def config_key(self) -> CONFIG_KEYS:
        return get_config_key_by_arch(self._arch_name)

    @property
    def arch_config(self) -> Dict[str, Any]:
        return self._config[self.config_key.value]  # type: ignore

    @property
    def relative_path(self) -> str:
        prefix = construct_experiment_name(
            self._config_name,
            self._arch_name,
            self._finetune_dataset,
            self._pretrain_dataset
        )
        return f"{prefix}/{self._run_id}"

    @property
    def training_config(self) -> TrainingConfig:
        return self._config['training']

    def dump_config(self):
        path = PATHS.TENSORBOARD_DIR / self.relative_path / f'config_{time.strftime(FORMATS.TIME)}.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        combined_config = {
            'training': self.training_config,
            self.config_key: self.arch_config,
            'finetune_dataset': self._finetune_dataset,
            'pretrain_dataset': self._pretrain_dataset,
            'architecture': self._arch_name,
        }

        with open(path, 'w') as f:
            json.dump(combined_config, f, indent=4)

    def get_loss_fn(self, phase_name: PHASE, pad_token_id: int) -> nn.Module:
        if phase_name == PHASE.CLASSIFICATION:
            return nn.CrossEntropyLoss()
        elif phase_name == PHASE.AUTOREGRESSIVE:
            return nn.CrossEntropyLoss(ignore_index=pad_token_id)  # Use ignore_index to ignore padding tokens
        else:
            raise ValueError(f'Invalid phase name: {phase_name}')

    def save_checkpoint(
            self,
            architecture: Architecture,
            optimizer: optim.Optimizer,
            epoch: int,
            during_pretraining: bool
    ):
        path = PATHS.CHECKPOINTS_DIR / self.relative_path / f'{self.total_steps}.pth'
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = Checkpoint(
            epoch=epoch,
            total_steps=self.total_steps,
            model_state_dict=architecture.model.state_dict(),
            optimizer_state_dict=optimizer.state_dict(),
            best_loss=self.best_loss,
            during_pretraining=during_pretraining
        )
        torch.save(checkpoint, path)

        self.logger.info(f"Checkpoint saved at {path}")

    def load_checkpoint(self, device: torch.device) -> Optional[Checkpoint]:
        if (checkpoint_path := self.get_latest_chkpt()) is not None:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            return checkpoint
        return None

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

    @property
    def is_with_pretraining(self) -> bool:
        return self._pretrain_dataset is not None

    def get_optimizer(self, model: torch.nn.Module) -> optim.Optimizer:
        optimizer_type = self.training_config['optimizer_type']
        optimizer_params = self.training_config['optimizer_params']
        weight_decay = self.training_config['weight_decay']

        if optimizer_type == OPTIMIZER.ADAM:
            return optim.Adam(
                model.parameters(),
                lr=self.training_config['learning_rate'],
                weight_decay=weight_decay,
                **optimizer_params
            )
        assert_never(optimizer_type)

    def get_lr_scheduler(self, optimizer: optim.Optimizer) -> Optional[optim.lr_scheduler.StepLR]:
        scheduler_type = self.training_config['lr_scheduler']
        scheduler_params = self.training_config['lr_scheduler_params']

        if scheduler_type is None:
            return None
        if scheduler_type == LR_SCHEDULER.STEP:
            return optim.lr_scheduler.StepLR(optimizer, **scheduler_params)
        assert_never(scheduler_type)

    def train_and_evaluate_model(self, rank=0, world_size=1) -> Dict[str, Any]:
        set_seed(self.training_config['seed'])
        self._rank = rank or 0
        self._world_size = world_size or 1

        if self.is_distributed:
            setup(rank, world_size)
            device = torch.device(rank)
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.configure_logging()

        writer = SummaryWriter(log_dir=str(PATHS.TENSORBOARD_DIR / self.relative_path))
        architecture = get_arch_by_name(self._arch_name)(self.arch_config)

        checkpoint = self.load_checkpoint(device)

        skip_model_init = False
        if (
                (checkpoint is not None and checkpoint['during_pretraining'])
                or self.is_with_pretraining
        ):
            self._train_model(
                architecture=architecture,
                phase_name=PHASE.AUTOREGRESSIVE,
                checkpoint=checkpoint,
                writer=writer,
                device=device,
                skip_model_init=False,
            )
            checkpoint = None
            architecture.model.set_phase(PHASE.CLASSIFICATION)
            skip_model_init = True

        metrics = self._train_model(
            architecture=architecture,
            phase_name=PHASE.CLASSIFICATION,
            checkpoint=checkpoint,
            writer=writer,
            device=device,
            skip_model_init=skip_model_init,
        )

        return metrics

    def _train_model(
            self,
            architecture: Architecture,
            phase_name: PHASE,
            checkpoint: Optional[Checkpoint],
            writer: SummaryWriter,
            device: torch.device,
            skip_model_init: bool,  # we want to skip it if we are in the fine-tuning phase
    ) -> Dict[str, float]:
        if phase_name == PHASE.CLASSIFICATION:
            dataset_wrapper = get_text_dataset_factory_by_name(self._finetune_dataset)(phase_name)
        elif phase_name == PHASE.AUTOREGRESSIVE:
            if self._pretrain_dataset is None:
                raise ValueError(f'Phase name is {phase_name} but pretrain dataset is not provided')
            dataset_wrapper = get_text_dataset_factory_by_name(self._pretrain_dataset)(phase_name)
        else:
            raise ValueError(f'Invalid phase name: {phase_name}')
        dataset = dataset_wrapper.get_dataset(SPLIT.TRAIN, self.training_config['debug_data_size'])

        # We don't want to re-initialize the model if we moved to the fine-tuning phase
        if not skip_model_init:
            architecture.initialize_model(dataset_wrapper)
        optimizer = self.get_optimizer(architecture.model)  # TOOD: do we want to keep the pre-trained optimizer?

        lr_scheduler = self.get_lr_scheduler(optimizer)

        start_epoch = 0
        self.logger.info(f"Training {phase_name} phase")

        if checkpoint is not None:
            architecture.model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # Move the optimizer state to the correct device # TODO: understand why we must need it
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
            start_epoch = checkpoint['epoch']
            self.total_steps = checkpoint['total_steps']
            self.best_loss = checkpoint['best_loss']
            self.logger.info(f"Checkpoint loaded")

        self.logger.info(params_count_report(architecture))
        if STEPS.PRINT_GRAPH and self.is_master_process:
            sample_input_tensor = next(iter(dataset))[0].unsqueeze(0)
            writer.add_graph(architecture.model, sample_input_tensor)

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
                worker_init_fn=lambda x: set_seed(self.training_config['seed']),
                # collate_fn=dataset.collate_fn
            )
        else:
            data_loader = DataLoader(
                dataset,
                batch_size=self.training_config['batch_size'],
                shuffle=True
            )

        loss_fn = self.get_loss_fn(dataset_wrapper.phase_name, dataset_wrapper.tokenizer.pad_token_id)
        architecture.model.to(device)
        architecture.model.train()
        metrics = {}
        for epoch in range(start_epoch, self.training_config['epochs']):
            for i_epoch_step, data in enumerate(data_loader):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = architecture.forward(inputs)

                if dataset_wrapper.phase_name == PHASE.CLASSIFICATION:
                    loss = loss_fn(outputs, labels)
                elif dataset_wrapper.phase_name == PHASE.AUTOREGRESSIVE:
                    loss = loss_fn(outputs.view(-1, outputs.size(-1)), labels.view(-1))
                else:
                    raise ValueError(f'Invalid phase name: {dataset_wrapper.phase_name}')

                loss.backward()

                # Gradient clipping
                gradient_clip_value = self.training_config['gradient_clip_value']
                if gradient_clip_value is not None:
                    torch.nn.utils.clip_grad_norm_(architecture.model.parameters(), gradient_clip_value)

                optimizer.step()
                if lr_scheduler is not None:
                    if isinstance(lr_scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        lr_scheduler.step(loss)
                    else:
                        lr_scheduler.step()
                self.total_steps += 1

                if self.is_master_process:
                    # Log the training loss
                    if self.total_steps % STEPS.LOG_STEP == 0:
                        writer.add_scalar(
                            '/'.join([
                                self.relative_path,
                                dataset_wrapper.phase_name,
                                'Loss'
                            ]),
                            loss.item(),
                            self.total_steps
                        )

                        self.logger.info(
                            f'{dataset_wrapper.phase_name} Epoch [{epoch + 1}/{self.training_config["epochs"]}], '
                            f'Step [{i_epoch_step + 1}/{len(data_loader)}], Total Steps: {self.total_steps}, Loss: {loss.item():.4f}'
                        )

                    if self.total_steps >= STEPS.WARMUP_STEPS and self.total_steps % STEPS.SAVE_STEP == 0:
                        self.save_checkpoint(
                            architecture, optimizer, epoch,
                            during_pretraining=dataset_wrapper.phase_name == PHASE.AUTOREGRESSIVE
                        )

                if (
                        dataset_wrapper.phase_name == PHASE.CLASSIFICATION
                        and self.total_steps >= STEPS.WARMUP_STEPS
                        and self.total_steps % STEPS.EVAL_STEP == 0
                ):
                    metrics = self._evaluate_model(architecture, dataset_wrapper, SPLIT.TEST, device)
                    self.best_loss = min(self.best_loss, metrics[METRICS.LOSS])
                    for key, value in metrics.items():
                        writer.add_scalar(
                            '/'.join([
                                self.relative_path,
                                dataset_wrapper.phase_name,
                                'test_' + key
                            ]),
                            value,
                            self.total_steps,
                        )
                    self.logger.info(f'Test Metrics: {metrics}')

                    if self.training_config['early_stopping']:
                        # Early stopping
                        early_stopping_patience = self.training_config['early_stopping_patience']
                        if metrics[METRICS.LOSS] > self.best_loss:
                            self.early_stopping_counter += 1
                            if self.early_stopping_counter >= early_stopping_patience:
                                self.logger.info("Early stopping triggered")
                                return metrics
                        else:
                            self.early_stopping_counter = 0

        return metrics

    def _evaluate_model(
            self,
            architecture: Architecture,
            dataset_wrapper: TextDatasetFactory,
            split: SPLIT,
            device: torch.device,
    ) -> Dict[str, float]:
        # Debug data size
        dataset = dataset_wrapper.get_dataset(split, self.training_config['debug_data_size'])
        if self.is_distributed:
            sampler = DistributedSampler(
                dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=False,
                drop_last=DDP.DROP_LAST
            )
            data_loader = DataLoader(
                dataset,
                batch_size=self.training_config['batch_size'],
                sampler=sampler,
                num_workers=DDP.NUM_WORKERS,
                worker_init_fn=lambda x: set_seed(self.training_config['seed']),
                # collate_fn=dataset.collate_fn
            )
        else:
            data_loader = DataLoader(
                dataset,
                batch_size=self.training_config['batch_size']
            )

        test_loss = 0
        correct = 0
        total = 0
        loss_fn = self.get_loss_fn(PHASE.CLASSIFICATION, dataset_wrapper.tokenizer.pad_token_id)
        with torch.no_grad():
            for data in data_loader:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = architecture.forward(inputs)
                loss = loss_fn(outputs, labels)
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                # noinspection PyUnresolvedReferences
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        return {METRICS.LOSS: test_loss / len(data_loader), METRICS.ACCURACY: accuracy}

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
