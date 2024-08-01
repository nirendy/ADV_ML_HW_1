import argparse
from typing import Optional

import torch
import torch.multiprocessing as mp

from src.consts import IAddArgs
from scripts.create_slurm_file import run_slurm
from src.trainer import set_seed
from src.types import ARCH
from src.trainer import Trainer
from src.types import DATASET
from src.types import IArgs
from src.types import IConfigName
from src.utils.argparse_utils import create_dict_from_argparse_remainder


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


def main(main_args: IArgs, with_slurm: bool, add_args: IAddArgs):
    func = main_with_slurm if with_slurm else main_local
    func(
        IArgs(
            main_args.config_name,
            main_args.architecture,
            main_args.finetune_dataset,
            main_args.pretrain_dataset,
            main_args.run_id,
        ),
        **(add_args._asdict())
    )


def main_with_slurm(args: IArgs, **kwargs):
    # python train_one.py  --config_name tiny1 --architecture transformer_copy --finetune_dataset imdb --with_parallel True --with_slurm True
    run_slurm(
        args,
        IAddArgs(
            **kwargs
        )
    )


def main_local(args: IArgs, with_parallel: bool):
    if with_parallel:
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


def main_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name', type=str, required=True)
    parser.add_argument('--architecture', type=str, required=True)
    parser.add_argument('--finetune_dataset', type=str, required=True)
    parser.add_argument('--pretrain_dataset', type=str, default=None)
    parser.add_argument('--run_id', type=str, default=None)
    parser.add_argument('--with_parallel', type=bool, default=False)
    parser.add_argument('--with_slurm', type=bool, default=False)
    parser.add_argument('--extra_args', nargs=argparse.REMAINDER)
    main_args = parser.parse_args()

    main(
        main_args=IArgs(
            **{
                k: v
                for k, v in vars(main_args).items()
                if k not in ['extra_args', 'with_parallel', 'with_slurm']
            }
        ),
        with_slurm=main_args.with_slurm,
        **{
            'with_parallel': main_args.with_parallel,
            **create_dict_from_argparse_remainder(main_args.extra_args)
        }
    )


if __name__ == '__main__':
    main_parser()
