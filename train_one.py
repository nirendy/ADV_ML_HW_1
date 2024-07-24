import argparse
from typing import NamedTuple
from typing import Optional

import torch
import torch.multiprocessing as mp

from src.trainer import set_seed
from src.types import ARCH
from src.trainer import Trainer
from src.types import DATASET
from src.types import IConfigName


class IArgs(NamedTuple):
    config_name: IConfigName
    architecture: ARCH
    finetune_dataset: DATASET
    pretrain_dataset: Optional[DATASET]
    run_id: Optional[str]


def train_one(
        rank: Optional[int],
        world_size: Optional[int],
        config_name: IConfigName,
        architecture: ARCH,
        finetune_dataset: DATASET,
        pretrain_dataset: Optional[DATASET],
        run_id: Optional[str]
):
    Trainer(
        config_name=config_name,
        architecture=architecture,
        finetune_dataset=finetune_dataset,
        pretrain_dataset=pretrain_dataset,
        run_id=run_id,
    ).train_and_evaluate_model(rank, world_size)


def main_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name', type=str, required=True)
    parser.add_argument('--architecture', type=str, required=True)
    parser.add_argument('--finetune_dataset', type=str, required=True)
    parser.add_argument('--pretrain_dataset', type=str, default=None)
    parser.add_argument('--run_id', type=str, default=None)
    parser.add_argument('--with_parallel', type=bool, default=False)
    main_args = parser.parse_args()
    main(
        args=IArgs(
            main_args.config_name,
            main_args.architecture,
            main_args.finetune_dataset,
            main_args.pretrain_dataset,
            main_args.run_id,
        ),
        is_parallel=main_args.with_parallel,
    )


def main(args: IArgs, is_parallel: bool):
    if is_parallel:
        set_seed(42)  # TODO: Need here? (we do it again after the spawn)
        world_size = torch.cuda.device_count()
        mp.spawn(
            fn=train_one,
            args=(world_size, *args),
            nprocs=world_size,
            join=True,
        )
    else:
        train_one(
            None,
            None,
            *args
        )


if __name__ == '__main__':
    main_parser()
