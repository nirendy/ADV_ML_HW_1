from typing import TypedDict


class LSTMConfig(TypedDict):
    input_size: int
    hidden_size: int
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


class Config(TypedDict):
    lstm: LSTMConfig
    transformer: TransformerConfig
    s4: S4Config
    training: TrainingConfig
