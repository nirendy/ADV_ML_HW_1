import argparse
from typing import Optional

from src.types import ARCH
from src.trainer import Trainer
from src.types import DATASET
from src.types import IConfigName


def train_one(
        config_name: IConfigName,
        architecture: ARCH,
        finetune_dataset: DATASET,
        pretrain_dataset: Optional[DATASET],
        run_id: Optional[str] = None
):
    Trainer(
        config_name=config_name,
        architecture=architecture,
        finetune_dataset=finetune_dataset,
        pretrain_dataset=pretrain_dataset,
        run_id=run_id
    ).train_and_evaluate_model()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name', type=str, required=True)
    parser.add_argument('--architecture', type=str, required=True)
    parser.add_argument('--finetune_dataset', type=str, required=True)
    parser.add_argument('--pretrain_dataset', type=str, default=None)
    parser.add_argument('--run_id', type=str, default=None)

    args = parser.parse_args()
    train_one(
        architecture=args.architecture,
        finetune_dataset=args.finetune_dataset,
        pretrain_dataset=args.pretrain_dataset,
        config_name=args.config_name,
        run_id=args.run_id,
    )


if __name__ == '__main__':
    # main()
    train_one(
        'small', ARCH.LSTM, DATASET.LISTOPS, None,
        run_id='20240722_14-47-44'
    )
