from src.types import ARCH
from src.types import DATASET
from train_one import IArgs
from train_one import main

if __name__ == '__main__':
    main(
        args=IArgs(
            config_name='small',
            architecture=ARCH.TRANSFORMER,
            finetune_dataset=DATASET.IMDB,
            pretrain_dataset=None,
            run_id=None
        ),
        is_parallel=False,
    )
