import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

from src.datasets.base_dataset import BaseDataset


class MathQABaseDataset(BaseDataset):
    def load_data(self):
        # Generate synthetic arithmetic data for demonstration purposes
        def generate_arithmetic_sequence(length):
            ops = ['+', '-', '*', '/']
            seq = []
            for _ in range(length):
                num1, num2 = np.random.randint(1, 10, size=2)
                op = np.random.choice(ops)
                seq.append(f"{num1} {op} {num2}")
            return seq

        train_sequences = generate_arithmetic_sequence(100)
        test_sequences = generate_arithmetic_sequence(20)

        # Here we would parse these sequences and create data tensors
        # For simplicity, we just create dummy tensors
        self.train_data = torch.randn(100, 10, 10)  # 100 samples, 10 timesteps, 10 features
        self.train_targets = torch.randn(100, 1)
        self.test_data = torch.randn(20, 10, 10)
        self.test_targets = torch.randn(20, 1)

    def get_train_loader(self):
        train_dataset = TensorDataset(self.train_data, self.train_targets)
        return DataLoader(train_dataset, batch_size=16, shuffle=True)

    def get_test_loader(self):
        test_dataset = TensorDataset(self.test_data, self.test_targets)
        return DataLoader(test_dataset, batch_size=16, shuffle=False)
