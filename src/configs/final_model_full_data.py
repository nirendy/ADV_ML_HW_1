from src.types import OPTIMIZER
from src.utils.config_types import Config
from src.utils.config_types import LSTMConfig
from src.utils.config_types import S4Config
from src.utils.config_types import TrainingConfig
from src.utils.config_types import TransformerConfig

# Purpose of this stage: Train final model on full dataset for production-level performance.
# Expectation: Minimal overfitting with regularization techniques.

config = Config(
    lstm=LSTMConfig(
        d_model=128,
        num_layers=3
    ),
    transformer=TransformerConfig(
        d_model=128,
        num_heads=8,
        num_layers=3,
        dim_feedforward=256,
    ),
    s4=S4Config(
        d_model=128,
        state_size=128,
        num_layers=3
    ),
    training=TrainingConfig(
        batch_size=64,
        learning_rate=0.0005,
        epochs=20,
        seed=42,
        debug_data_size=50000,
        # dropout_rate=0.3,
        weight_decay=0.01,
        lr_scheduler='step',
        lr_scheduler_params={'step_size': 10, 'gamma': 0.1},
        gradient_clip_value=1.0,
        # weight_init_method='xavier_uniform',
        optimizer_type=OPTIMIZER.ADAM,
        optimizer_params={'betas': (0.9, 0.999), 'eps': 1e-08},
        # use_batch_norm=True,
        # sequence_length=500,
        # embedding_dim=128,
        early_stopping=True,
        early_stopping_patience=7,
    ),
)
