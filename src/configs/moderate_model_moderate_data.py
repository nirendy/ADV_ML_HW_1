from src.utils.config_types import Config
from src.utils.config_types import LSTMConfig
from src.utils.config_types import S4Config
from src.utils.config_types import TrainingConfig
from src.utils.config_types import TransformerConfig

# Purpose of this stage: Improve model performance and detect bugs with a moderate dataset.
# Expectation: Some overfitting but better generalization than Stage 1.

config = Config(
    lstm=LSTMConfig(
        d_model=64,
        num_layers=2
    ),
    transformer=TransformerConfig(
        d_model=64,
        num_heads=4,
        num_layers=2,
        dim_feedforward=128,
    ),
    s4=S4Config(
        d_model=64,
        state_size=64,
        num_layers=2
    ),
    training=TrainingConfig(
        batch_size=32,
        learning_rate=0.001,
        epochs=10,
        # debug_data_size=10000,
        # dropout_rate=0.2,
        # weight_decay=0.01,
        # lr_scheduler='step',
        # lr_scheduler_params={'step_size': 10, 'gamma': 0.1},
        # gradient_clip_value=1.0,
        # weight_init_method='xavier_uniform',
        # optimizer_type=OPTIMIZER.ADAM,
        # optimizer_params={'betas': (0.9, 0.999), 'eps': 1e-08},
        # use_batch_norm=True,
        # sequence_length=500,
        # embedding_dim=128,
        # early_stopping=True,
        # early_stopping_patience=5,
    ),
)
