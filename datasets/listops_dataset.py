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




class ListOpsDataset(Dataset):
    def load_data(self):
        # Dummy data for example purposes
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
