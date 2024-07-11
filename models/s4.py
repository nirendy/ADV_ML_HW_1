import torch
import torch.nn as nn
import torch.optim as optim

from models.architecture import Architecture


class S4Layer(nn.Module):
    def __init__(self, d_model, state_size):
        super(S4Layer, self).__init__()
        self.d_model = d_model
        self.state_size = state_size
        self.W = nn.Parameter(torch.randn(d_model, state_size))
        self.U = nn.Parameter(torch.randn(d_model, state_size))
        self.V = nn.Parameter(torch.randn(state_size, d_model))

    def forward(self, x):
        # Apply state-space transformation
        h = torch.tanh(torch.matmul(x, self.W))
        y = torch.tanh(torch.matmul(h, self.U))
        out = torch.tanh(torch.matmul(y, self.V))
        return out


# Simplified S4 Architecture Implementation
class S4Architecture(Architecture):
    def __init__(self):
        self.d_model = 512
        self.state_size = 256
        self.num_layers = 3

    def initialize_model(self):
        self.layers = nn.ModuleList([S4Layer(self.d_model, self.state_size) for _ in range(self.num_layers)])
        self.fc = nn.Linear(self.d_model, 1)
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
        for layer in self.layers:
            x = layer(x)
        output = self.fc(x)
        return output

    def get_metrics(self):
        return {'Test Loss': self.test_loss}
