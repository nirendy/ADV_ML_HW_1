from src.utils.config_types import Config
from src.utils.config_types import LSTMConfig
from src.utils.config_types import S4Config
from src.utils.config_types import TrainingConfig
from src.utils.config_types import TransformerConfig

config = Config(
    lstm=LSTMConfig(
        d_model=8,
        # hidden_size=20,
        num_layers=2
    ),
    transformer=TransformerConfig(
        d_model=64,
        num_heads=4,
        num_layers=2,
        dim_feedforward=64,
    ),
    s4=S4Config(
        d_model=64,
        state_size=32,
        num_layers=2
    ),
    training=TrainingConfig(
        batch_size=32,
        learning_rate=0.0001,
        epochs=10,
        seed=42
    ),
)
