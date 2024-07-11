import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from abc import ABC, abstractmethod
import torch.nn.functional as F
import pandas as pd

from datasets.base_dataset import Dataset
from models.architecture import Architecture
from strategies.base_strategy import Strategy
from utils.metrics import Metrics


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
    def __init__(self):
        self.hidden_size = 20
        self.input_size = 10
        self.num_layers = 2

    def initialize_model(self):
        self.layers = nn.ModuleList([LSTMCell(self.input_size, self.hidden_size)])
        for _ in range(1, self.num_layers):
            self.layers.append(LSTMCell(self.hidden_size, self.hidden_size))
        self.fc = nn.Linear(self.hidden_size, 1)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters())

    def parameters(self):
        params = list(self.fc.parameters())
        for layer in self.layers:
            params.extend(layer.parameters())
        return params

    def train_model(self, train_loader):
        self.train()
        for data, target in train_loader:
            self.optimizer.zero_grad()
            output = self.forward(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

    def evaluate_model(self, test_loader):
        self.eval()
        test_loss = 0
        with torch.no_grad():
            for data, target in test_loader:
                output = self.forward(data)
                test_loss += self.criterion(output, target).item()
        self.test_loss = test_loss / len(test_loader)

    def forward(self, x):
        h = [torch.zeros(x.size(0), self.hidden_size).to(x.device) for _ in range(self.num_layers)]
        c = [torch.zeros(x.size(0), self.hidden_size).to(x.device) for _ in range(self.num_layers)]

        for t in range(x.size(1)):
            inp = x[:, t]
            for l in range(self.num_layers):
                h[l], c[l] = self.layers[l](inp, (h[l], c[l]))
                inp = h[l]

        output = self.fc(inp)
        return output

    def get_metrics(self):
        return {'Test Loss': self.test_loss}
