import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

from src.datasets.base_dataset import BaseDataset


class RetrievalBaseDataset(BaseDataset):
    def load_data(self):
        # Generate synthetic retrieval data for demonstration purposes
        def generate_retrieval_sequence(length):
            seq = []
            for _ in range(length):
                data = np.random.randint(0, 100, size=(10, 10))  # Random sequences
                target = np.random.randint(0, 100)  # Target is a random integer
                seq.append((data, target))
            return seq

        train_sequences = generate_retrieval_sequence(100)
        test_sequences = generate_retrieval_sequence(20)

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
