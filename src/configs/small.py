from src.utils.config_types import Config
from src.utils.config_types import LSTMConfig
from src.utils.config_types import S4Config
from src.utils.config_types import TrainingConfig
from src.utils.config_types import TransformerConfig

config = Config(
    lstm=LSTMConfig(
        d_model=16,
        # hidden_size=8,
        num_layers=2
    ),
    transformer=TransformerConfig(
        d_model=16,
        num_heads=2,
        num_layers=3,
        dim_feedforward=64,
    ),
    s4=S4Config(
        d_model=16,
        state_size=8,
        num_layers=3
    ),
    training=TrainingConfig(
        batch_size=32,
        learning_rate=0.01,
        epochs=1,
        seed=42
    ),
)
