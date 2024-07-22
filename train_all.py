import argparse
from typing import List
from typing import Optional

from datasets import Dataset

from src.types import ARCH
from src.types import CONFIG_KEYS
from src.trainer import Trainer
from src.types import DATASET
from src.types import IConfigName


def train_all(
        config_name: IConfigName,
        architectures: List[ARCH] = (
                ARCH.LSTM,
                ARCH.TRANSFORMER,
                ARCH.S4,
                ARCH.S4_COPY,
        ),
        finetune_datasets: List[Dataset] = (
                DATASET.IMDB,
        ),
        pretrain_datasets: List[Optional[Dataset]] = (
                None,
        )
):
    for architecture in architectures:
        for pretrain_dataset in pretrain_datasets:
            for finetune_dataset in finetune_datasets:
                pretrain_name = pretrain_dataset.__class__.__name__ if pretrain_dataset else "None"
                run_id = '.'.join([
                    writer.get_logdir(),
                    architecture.__class__.__name__,
                    finetune_dataset.__class__.__name__,
                    pretrain_name,
                ])
                metrics = trainer.train_and_evaluate_model(
                    architecture, pretrain_dataset, finetune_dataset, writer, run_id
                )

                # Record results
                result = {
                    'Architecture': architecture.__class__.__name__,
                    'Pretrain Dataset': pretrain_name,
                    'Finetune Dataset': finetune_dataset.__class__.__name__,
                }
                result.update(metrics)
                results.append(result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name', type=str, default='medium')

    args = parser.parse_args()
    train_one(
        architecture=args.architecture,
        finetune_dataset=args.finetune_dataset,
        pretrain_dataset=args.pretrain_dataset,
        config_name=args.config_name
    )


if __name__ == '__main__':
    main()
