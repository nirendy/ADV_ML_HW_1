from src.utils.config_types import Config
from src.utils.config_types import LSTMConfig
from src.utils.config_types import S4Config
from src.utils.config_types import TrainingConfig
from src.utils.config_types import TransformerConfig

config = Config(
    lstm=LSTMConfig(
        d_model=32,
        # hidden_size=128,
        num_layers=2,
    ),
    transformer=TransformerConfig(
        d_model=256,
        num_heads=4,
        num_layers=2,
        dim_feedforward=512,
    ),
    s4=S4Config(
        d_model=256,
        state_size=64,
        num_layers=8
    ),
    training=TrainingConfig(
        batch_size=32,
        learning_rate=0.001,
        epochs=10,
        seed=42
    ),
)
