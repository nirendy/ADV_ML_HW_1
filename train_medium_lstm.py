from src.types import ARCH
from src.types import DATASET
from train_one import IArgs
from train_one import main
import os

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    main(
        args=IArgs(
            config_name='medium',
            architecture=ARCH.LSTM,
            finetune_dataset=DATASET.IMDB,
            pretrain_dataset=None,
            # pretrain_dataset=DATASET.WIKITEXT,
            # pretrain_dataset=DATASET.IMDB,
            run_id=None
        ),
        is_parallel=False,
    )
