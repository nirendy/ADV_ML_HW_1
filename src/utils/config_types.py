from typing import Any
from typing import Dict
from typing import Optional
from typing import TypedDict

from src.types import LR_SCHEDULER
from src.types import OPTIMIZER


class LSTMConfig(TypedDict):
    d_model: int
    # hidden_size: int # We enforce d_model == hidden_size, TODO: Fix that
    num_layers: int


class TransformerConfig(TypedDict):
    d_model: int
    num_heads: int
    num_layers: int
    dim_feedforward: int


class S4Config(TypedDict):
    d_model: int
    state_size: int
    num_layers: int


class TrainingConfig(TypedDict):
    batch_size: int
    learning_rate: float
    epochs: int
    seed: int
    debug_data_size: Optional[int]
    weight_decay: float
    lr_scheduler: Optional[LR_SCHEDULER]
    lr_scheduler_params: Dict[str, Any]
    gradient_clip_value: float
    optimizer_type: OPTIMIZER
    optimizer_params: Dict[str, Any]
    early_stopping: bool
    early_stopping_patience: int


class Config(TypedDict):
    lstm: LSTMConfig
    transformer: TransformerConfig
    s4: S4Config
    training: TrainingConfig
