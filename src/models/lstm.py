from typing import Dict

import torch
import torch.nn as nn
import torch.optim as optim

from src.datasets.base_dataset import BaseDataset
from src.models.architecture import Architecture
from src.utils.config_types import LSTMConfig


class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Gates: input, forget, cell, output
        self.i2h = nn.Linear(input_size, 4 * hidden_size)
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size)

    def forward(self, x, hidden):
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
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMLayer, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.cells = nn.ModuleList()
        for i in range(num_layers):
            self.cells.append(LSTMCell(input_size if i == 0 else hidden_size, hidden_size))

    def forward(self, x, hidden):
        h, c = hidden
        h_next, c_next = [], []
        seq_len, batch_size, _ = x.size()

        outputs = []
        for t in range(seq_len):
            input_t = x[t]
            for i, cell in enumerate(self.cells):
                h_t, c_t = cell(input_t, (h[i], c[i]))
                h_next.append(h_t)
                c_next.append(c_t)
                input_t = h_t
            outputs.append(h_t)

        h_next = torch.stack(h_next[-self.num_layers:])
        c_next = torch.stack(c_next[-self.num_layers:])

        return torch.stack(outputs), (h_next, c_next)


# Simple LSTM Architecture Implementation
class LSTMArchitecture(Architecture):
    model_config: LSTMConfig

    def initialize_model(self, dataset: BaseDataset) -> None:
        self.model = nn.Module()
        self.model.embedding = nn.Embedding(dataset.vocab_size, self.model_config['input_size'])
        self.model.lstm = LSTMLayer(
            self.model_config['input_size'], self.model_config['hidden_size'], self.model_config['num_layers']
        )
        self.model.fc = nn.Linear(self.model_config['hidden_size'], 1)

    def forward(self, x):
        batch_size = x.size(0)
        h_0 = torch.zeros(self.model_config['num_layers'], batch_size, self.model_config['hidden_size']).to(x.device)
        c_0 = torch.zeros(self.model_config['num_layers'], batch_size, self.model_config['hidden_size']).to(x.device)

        embedded = self.model.embedding(x)
        lstm_out, _ = self.model.lstm(embedded.permute(1, 0, 2), (h_0, c_0))

        # We only need the output of the last time step
        output = lstm_out[-1]
        output = self.model.fc(output)
        return output
