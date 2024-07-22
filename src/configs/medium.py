from src.utils.config_types import Config
from src.utils.config_types import LSTMConfig
from src.utils.config_types import S4Config
from src.utils.config_types import TrainingConfig
from src.utils.config_types import TransformerConfig

config = Config(
    lstm=LSTMConfig(
        input_size=64,
        hidden_size=128,
        num_layers=4
    ),
    transformer=TransformerConfig(
        d_model=16,
        num_heads=2,
        num_layers=3,
        dim_feedforward=128,
    ),
    s4=S4Config(
        d_model=16,
        state_size=8,
        num_layers=3
    ),
    training=TrainingConfig(
        batch_size=128,
        learning_rate=0.01,
        epochs=1000,
        seed=42
    ),
)
