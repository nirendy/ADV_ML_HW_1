from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from models.architecture import Architecture


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src):
        src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


# Simplified Transformer Architecture Implementation
class TransformerArchitecture(Architecture):
    def initialize_model(self):
        encoder_layer = TransformerEncoderLayer(
            self.config['d_model'],
            self.config['nhead'],
            self.config['dim_feedforward'],
            self.config['dropout']
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, self.config['num_layers'])
        self.fc = nn.Linear(self.config['d_model'], 1)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters())

    def parameters(self):
        params = list(self.fc.parameters())
        params.extend(self.transformer_encoder.parameters())
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
        # self.test_loss = test_loss / len(test_loader)
        return {'Test Loss': test_loss / len(test_loader)}

    def forward(self, x):
        x = x.permute(1, 0, 2)  # (batch_size, seq_len, d_model) -> (seq_len, batch_size, d_model)
        output = self.transformer_encoder(x)
        output = self.fc(output[-1])  # Use the output of the last token
        return output
