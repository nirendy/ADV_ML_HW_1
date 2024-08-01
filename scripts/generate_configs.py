import os

from src.consts import PATHS

config_template = """\
from src.utils.config_types import Config
from src.utils.config_types import LSTMConfig
from src.utils.config_types import S4Config
from src.utils.config_types import TrainingConfig
from src.utils.config_types import TransformerConfig

# Purpose of this stage: {purpose}
# Expectation: {expectation}

config = Config(
    lstm=LSTMConfig(
        d_model={lstm_d_model},
        num_layers={lstm_num_layers}
    ),
    transformer=TransformerConfig(
        d_model={transformer_d_model},
        num_heads={transformer_num_heads},
        num_layers={transformer_num_layers},
        dim_feedforward={transformer_dim_feedforward},
    ),
    s4=S4Config(
        d_model={s4_d_model},
        state_size={s4_state_size},
        num_layers={s4_num_layers}
    ),
    training=TrainingConfig(
        batch_size={training_batch_size},
        learning_rate={training_learning_rate},
        epochs={training_epochs},
        # debug_data_size={training_debug_data_size},
        # dropout_rate={dropout_rate},
        # weight_decay={weight_decay},
        # lr_scheduler='{lr_scheduler}',
        # lr_scheduler_params={lr_scheduler_params},
        # gradient_clip_value={gradient_clip_value},
        # weight_init_method='{weight_init_method}',
        # optimizer_type='{optimizer_type}',
        # optimizer_params={optimizer_params},
        # use_batch_norm={use_batch_norm},
        # sequence_length={sequence_length},
        # embedding_dim={embedding_dim},
        # early_stopping={early_stopping},
        # early_stopping_patience={early_stopping_patience},
    ),
)

"""

configs_to_create = (
    {
        "config_name": "prototype_small_model_small_data",
        "purpose": "Validate that the basic pipeline works.",
        "expectation": "Rapid overfitting on training data due to small size.",
        "lstm_d_model": 32,
        "lstm_num_layers": 1,
        "transformer_d_model": 32,
        "transformer_num_heads": 2,
        "transformer_num_layers": 1,
        "transformer_dim_feedforward": 64,
        "s4_d_model": 32,
        "s4_state_size": 32,
        "s4_num_layers": 1,
        "training_batch_size": 16,
        "training_learning_rate": 0.001,
        "training_epochs": 5,
        "training_debug_data_size": 1000,
        "dropout_rate": 0.1,
        "weight_decay": 0.01,
        "lr_scheduler": 'step',
        "lr_scheduler_params": {"step_size": 5, "gamma": 0.1},
        "gradient_clip_value": 1.0,
        "weight_init_method": 'xavier_uniform',
        "optimizer_type": 'adam',
        "optimizer_params": {"betas": (0.9, 0.999), "eps": 1e-08},
        "use_batch_norm": False,
        "sequence_length": 500,
        "embedding_dim": 128,
        "early_stopping": True,
        "early_stopping_patience": 3
    },
    {
        "config_name": "moderate_model_moderate_data",
        "purpose": "Improve model performance and detect bugs with a moderate dataset.",
        "expectation": "Some overfitting but better generalization than Stage 1.",
        "lstm_d_model": 64,
        "lstm_num_layers": 2,
        "transformer_d_model": 64,
        "transformer_num_heads": 4,
        "transformer_num_layers": 2,
        "transformer_dim_feedforward": 128,
        "s4_d_model": 64,
        "s4_state_size": 64,
        "s4_num_layers": 2,
        "training_batch_size": 32,
        "training_learning_rate": 0.001,
        "training_epochs": 10,
        "training_debug_data_size": 10000,
        "dropout_rate": 0.2,
        "weight_decay": 0.01,
        "lr_scheduler": 'step',
        "lr_scheduler_params": {"step_size": 10, "gamma": 0.1},
        "gradient_clip_value": 1.0,
        "weight_init_method": 'xavier_uniform',
        "optimizer_type": 'adam',
        "optimizer_params": {"betas": (0.9, 0.999), "eps": 1e-08},
        "use_batch_norm": True,
        "sequence_length": 500,
        "embedding_dim": 128,
        "early_stopping": True,
        "early_stopping_patience": 5
    },
    {
        "config_name": "final_model_full_data",
        "purpose": "Train final model on full dataset for production-level performance.",
        "expectation": "Minimal overfitting with regularization techniques.",
        "lstm_d_model": 128,
        "lstm_num_layers": 3,
        "transformer_d_model": 128,
        "transformer_num_heads": 8,
        "transformer_num_layers": 3,
        "transformer_dim_feedforward": 256,
        "s4_d_model": 128,
        "s4_state_size": 128,
        "s4_num_layers": 3,
        "training_batch_size": 64,
        "training_learning_rate": 0.0005,
        "training_epochs": 20,
        "training_debug_data_size": 50000,
        "dropout_rate": 0.3,
        "weight_decay": 0.01,
        "lr_scheduler": 'step',
        "lr_scheduler_params": {"step_size": 10, "gamma": 0.1},
        "gradient_clip_value": 1.0,
        "weight_init_method": 'xavier_uniform',
        "optimizer_type": 'adam',
        "optimizer_params": {"betas": (0.9, 0.999), "eps": 1e-08},
        "use_batch_norm": True,
        "sequence_length": 500,
        "embedding_dim": 128,
        "early_stopping": True,
        "early_stopping_patience": 7
    }
)


def main(configs=configs_to_create):
    config_paths = PATHS.PROJECT_DIR / 'src' / 'configs'
    for config in configs:
        target = config_paths / (config["config_name"] + '.py')
        target.write_text(config_template.format(**config))

    print("Configuration files created successfully.")


if __name__ == "__main__":
    main()
