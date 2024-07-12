from typing import Dict

import torch
import torch.nn as nn
import torch.optim as optim

from models.architecture import Architecture
from utils.config_types import LSTMConfig


class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, 4 * hidden_size)

    def forward(self, x, hidden):
        hx, cx = hidden
        combined = torch.cat((x, hx), 1)
        gates = self.i2h(combined)
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)

        return hy, cy


# Simple LSTM Architecture Implementation
class LSTMArchitecture(Architecture):
    config: LSTMConfig

    def initialize_model(self):
        self.layers = nn.ModuleList([LSTMCell(self.config['input_size'], self.config['hidden_size'])])
        for _ in range(1, self.config['num_layers']):
            self.layers.append(LSTMCell(self.config['hidden_size'], self.config['hidden_size']))
        self.fc = nn.Linear(self.config['hidden_size'], 1)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters())

    def parameters(self):
        params = list(self.fc.parameters())
        for layer in self.layers:
            params.extend(layer.parameters())
        return params

    def train_model(self, train_loader):
        for data, target in train_loader:
            self.optimizer.zero_grad()
            output = self.forward(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

    def evaluate_model(self, test_loader) -> Dict[str, float]:
        test_loss = 0
        with torch.no_grad():
            for data, target in test_loader:
                output = self.forward(data)
                test_loss += self.criterion(output, target).item()

        return {'Test Loss': test_loss / len(test_loader)}

    def forward(self, x):
        h = [torch.zeros(x.size(0), self.config['hidden_size']).to(x.device) for _ in range(self.config['num_layers'])]
        c = [torch.zeros(x.size(0), self.config['hidden_size']).to(x.device) for _ in range(self.config['num_layers'])]

        for t in range(x.size(1)):
            inp = x[:, t]
            for l in range(self.config['num_layers']):
                h[l], c[l] = self.layers[l](inp, (h[l], c[l]))
                inp = h[l]

        output = self.fc(inp)
        return output
