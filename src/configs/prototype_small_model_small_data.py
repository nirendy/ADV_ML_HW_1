from src.types import OPTIMIZER
from src.utils.config_types import Config
from src.utils.config_types import LSTMConfig
from src.utils.config_types import S4Config
from src.utils.config_types import TrainingConfig
from src.utils.config_types import TransformerConfig

# Purpose of this stage: Validate that the basic pipeline works.
# Expectation: Rapid overfitting on training data due to small size.

config = Config(
    lstm=LSTMConfig(
        d_model=32,
        num_layers=1
    ),
    transformer=TransformerConfig(
        d_model=32,
        num_heads=2,
        num_layers=1,
        dim_feedforward=64,
    ),
    s4=S4Config(
        d_model=32,
        state_size=32,
        num_layers=1
    ),
    training=TrainingConfig(
        batch_size=16,
        learning_rate=0.001,
        epochs=5,
        seed=42,
        debug_data_size=1000,
        # dropout_rate=0.1,
        weight_decay=0.01,
        lr_scheduler=None,
        lr_scheduler_params={'step_size': 5, 'gamma': 0.1},
        gradient_clip_value=1.0,
        # weight_init_method='xavier_uniform',
        optimizer_type=OPTIMIZER.ADAM,
        optimizer_params={'betas': (0.9, 0.999), 'eps': 1e-08},
        # use_batch_norm=False,
        # sequence_length=500,
        # embedding_dim=128,
        early_stopping=True,
        early_stopping_patience=3,
    ),
)
