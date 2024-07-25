from typing import Tuple, Dict

import torch
import torch.nn as nn
import torch.optim as optim

from src.datasets.base_dataset import DatasetFactory
from src.datasets.text_dataset import TextDatasetFactory
from src.models.architecture import AbstractSequenceModel
from src.models.architecture import Architecture
from src.types import PHASE
from src.utils.config_types import LSTMConfig


class LSTMCell(nn.Module):
    def __init__(self, d_model: int, hidden_size: int):
        super(LSTMCell, self).__init__()
        self.d_model = d_model
        self.hidden_size = hidden_size

        # Gates: input, forget, cell, output
        self.i2h = nn.Linear(d_model, 4 * hidden_size)
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size)

    def forward(self, x: torch.Tensor, hidden: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[
        torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        h, c = hidden
        gates = self.i2h(x) + self.h2h(h)
        i_gate, f_gate, o_gate, c_gate = gates.chunk(4, 1)

        i_gate = torch.sigmoid(i_gate)
        f_gate = torch.sigmoid(f_gate)
        o_gate = torch.sigmoid(o_gate)
        c_gate = torch.tanh(c_gate)

        c_next = f_gate * c + i_gate * c_gate
        h_next = o_gate * torch.tanh(c_next)

        return h_next, (h_next, c_next)


class LSTMLayer(nn.Module):
    def __init__(self, d_model: int, hidden_size: int, num_layers: int):
        super(LSTMLayer, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.cells = nn.ModuleList(
            [LSTMCell(d_model if i == 0 else hidden_size, hidden_size) for i in range(num_layers)])

    def forward(self, x: torch.Tensor, hidden: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[
        torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        h, c = hidden
        h_next, c_next = [], []
        seq_len, batch_size, _ = x.size()

        outputs = []
        for t in range(seq_len):
            input_t = x[t]
            for i, cell in enumerate(self.cells):
                h_t, (h_i, c_i) = cell(input_t, (h[i], c[i]))
                h_next.append(h_i)
                c_next.append(c_i)
                input_t = h_t
            outputs.append(h_t)

        h_next = torch.stack(h_next[-self.num_layers:])
        c_next = torch.stack(c_next[-self.num_layers:])

        return torch.stack(outputs), (h_next, c_next)


class LSTMModel(AbstractSequenceModel):

    def __init__(self, d_model: int, hidden_size: int, num_layers: int, vocab_size: int, phase_name: PHASE):
        super(LSTMModel, self).__init__(vocab_size, d_model, phase_name)
        self.lstm = LSTMLayer(d_model, hidden_size, num_layers)

    def forward_sequence_model(self, x):
        """
        Parameters:
            x: (batch_size, seq_len, d_model)
        Returns:
            x: (batch_size, seq_len, d_model)
        """
        batch_size = x.size(0)
        # Initialize hidden state
        h_0 = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(x.device)
        # Initialize cell state
        c_0 = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(x.device)

        # Change dims to (seq_len, batch_size, d_model)
        x = x.permute(1, 0, 2)
        lstm_out, _ = self.lstm(x, (h_0, c_0))

        # Change dims back
        output = lstm_out.permute(1, 0, 2)
        return output


class LSTMArchitecture(Architecture):
    model_config: LSTMConfig

    def initialize_model(self, dataset: TextDatasetFactory) -> None:
        self.model = LSTMModel(
            d_model=self.model_config['d_model'],
            hidden_size=self.model_config['d_model'],
            # hidden_size=self.model_config['hidden_size'], # disabled for now, we enforce d_model == state_size # TODO: enable this
            num_layers=self.model_config['num_layers'],
            vocab_size=dataset.vocab_size,
            phase_name=dataset.phase_name
        )
