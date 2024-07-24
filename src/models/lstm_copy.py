from torch import nn

from src.datasets.base_dataset import DatasetFactory
from src.models.architecture import AbstractSequenceModel
from src.models.architecture import Architecture
from src.types import PHASE
from src.utils.config_types import LSTMConfig


class LSTMModel(AbstractSequenceModel):

    def __init__(self, d_model: int, hidden_size: int, num_layers: int, vocab_size: int, phase_name: PHASE):
        super(LSTMModel, self).__init__(vocab_size, d_model, phase_name)
        self.lstm = nn.LSTM(d_model, hidden_size, num_layers)

    def forward_sequence_model(self, x):
        """
        :param x: (batch_size, seq_len, d_model)
        :return: (batch_size, seq_len, hidden_size)
        """
        x, _ = self.lstm(x)
        return x


class LSTMCopyArchitecture(Architecture):
    model_config: LSTMConfig

    def initialize_model(self, dataset: DatasetFactory) -> None:
        self.model = LSTMModel(
            d_model=self.model_config['d_model'],
            hidden_size=self.model_config['d_model'],
            num_layers=self.model_config['num_layers'],
            vocab_size=dataset.vocab_size,
            phase_name=dataset.phase_name
        )
