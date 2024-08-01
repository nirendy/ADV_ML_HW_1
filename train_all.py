import argparse
from typing import List
from typing import Optional

from datasets import Dataset

from src.consts import DDP
from src.consts import IAddArgs
from src.types import ARCH
from src.types import DATASET
from src.types import IArgs
from src.types import IConfigName
import train_one
from src.utils.argparse_utils import create_dict_from_argparse_remainder
from src.utils.experiment_runner import construct_experiment_name


def all_configs(
        config_name: IConfigName,
        run_id: Optional[str] = None,
        architectures: List[ARCH] = (
                ARCH.LSTM,
                ARCH.LSTM_COPY,
                ARCH.TRANSFORMER,
                ARCH.TRANSFORMER_COPY,
                ARCH.S4,
                ARCH.S4_COPY,
        ),
        finetune_datasets: List[DATASET] = (
                DATASET.IMDB,
        ),
        pretrain_datasets: List[Optional[Dataset]] = (
                None,
                DATASET.IMDB,
                DATASET.WIKITEXT,
        )
) -> List[IArgs]:
    return [
        IArgs(
            config_name=config_name,
            architecture=architecture,
            finetune_dataset=finetune_dataset,
            pretrain_dataset=pretrain_dataset,
            run_id=run_id,
        )
        for architecture in architectures
        for pretrain_dataset in pretrain_datasets
        for finetune_dataset in finetune_datasets
    ]


def main_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name', type=str, required=True)
    parser.add_argument('--run_id', type=str, default=None)
    parser.add_argument('--with_parallel', type=bool, default=False)
    parser.add_argument('--with_slurm', type=bool, default=False)
    parser.add_argument('--extra_args', nargs=argparse.REMAINDER)
    args = parser.parse_args()

    for i, main_args in enumerate(all_configs(args.config_name, run_id=args.run_id)):
        print(f"{'-' * 30} {construct_experiment_name(*main_args[:-1])} {'-' * 30}")
        train_one.main(
            main_args=main_args,
            with_slurm=args.with_slurm,
            add_args=IAddArgs(
                with_parallel=args.with_parallel,
                master_port=str(int(DDP.MASTER_PORT) + i),
                **{
                    **create_dict_from_argparse_remainder(args.extra_args)
                }
            )
        )


if __name__ == '__main__':
    main_parser()
