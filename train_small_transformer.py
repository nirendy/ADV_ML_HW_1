from src.types import ARCH
from src.types import DATASET
from train_one import IArgs
from train_one import main

if __name__ == '__main__':
    main(
        args=IArgs(
            config_name='small',
            architecture=ARCH.TRANSFORMER_COPY,
            finetune_dataset=DATASET.WIKITEXT,
            pretrain_dataset=DATASET.WIKITEXT,
            run_id=None
        ),
        is_parallel=False,
    )
