import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from abc import ABC, abstractmethod
import pandas as pd


# Basic setup

# abstract class for all models
class Architecture(ABC):
    @abstractmethod
    def initialize_model(self):
        pass

    @abstractmethod
    def train_model(self, train_loader):
        pass

    @abstractmethod
    def evaluate_model(self, test_loader):
        pass

    @abstractmethod
    def get_metrics(self):
        pass


# Abstract Dataset class
class Dataset(ABC):
    @abstractmethod
    def load_data(self):
        pass

    @abstractmethod
    def get_train_loader(self):
        pass

    @abstractmethod
    def get_test_loader(self):
        pass


# Abstract Training Strategy class
class TrainingStrategy(ABC):
    @abstractmethod
    def pretrain_model(self, model, pretrain_loader):
        pass

    @abstractmethod
    def fine_tune_model(self, model, train_loader):
        pass

    @abstractmethod
    def evaluate_model(self, model, test_loader):
        pass


# Task 1: Review LRA Dataset Subsets
# Action: Look at the descriptions of the six subsets in the LRA dataset.
# Time Estimate: 30 minutes
# Outcome: Choose one subset for the assignment.
# Chosen Dataset: ListOps

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


# Simple Training Strategy Implementation
class DirectTrainingStrategy(TrainingStrategy):
    def pretrain_model(self, model, pretrain_loader):
        # No pretraining for direct training strategy
        pass

    def fine_tune_model(self, model, train_loader):
        model.train_model(train_loader)

    def evaluate_model(self, model, test_loader):
        model.evaluate_model(test_loader)


# Reporting Function
import pandas as pd


def run_experiment(architectures, datasets, strategies):
    results = []

    for architecture_cls in architectures:
        for dataset_cls in datasets:
            for strategy_cls in strategies:
                # Initialize objects
                architecture = architecture_cls()
                dataset = dataset_cls()
                strategy = strategy_cls()

                # Load data
                dataset.load_data()
                train_loader = dataset.get_train_loader()
                test_loader = dataset.get_test_loader()

                # Initialize and train model
                architecture.initialize_model()
                strategy.pretrain_model(architecture, train_loader)
                strategy.fine_tune_model(architecture, train_loader)

                # Evaluate model
                strategy.evaluate_model(architecture, test_loader)
                metrics = architecture.get_metrics()

                # Record results
                result = {
                    'Architecture': architecture_cls.__name__,
                    'Dataset': dataset_cls.__name__,
                    'Strategy': strategy_cls.__name__,
                }
                result.update(metrics)
                results.append(result)

    # Convert results to DataFrame and display
    df = pd.DataFrame(results)
    return df


# %%
# Task 2: Set Up Google Colab Environment
# Action: Open Google Colab and set up a new notebook.
# Time Estimate: 30 minutes
# Outcome: Colab notebook ready for coding.
# !pip install torch torchvision transformers
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# %%


# Task 3: Implement LSTM Class
# Action: Write the class for an LSTM model from scratch.
# Define input size, hidden layers, and output size.
# Time Estimate: 2 hours
# Outcome: Basic LSTM class implemented.


# Simple LSTM Cell Implementation
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
class LSTM(Architecture):
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


# %%
# Task 5: Test LSTM with Dummy Data
# Action: Run the LSTM model with some dummy data to ensure it's working.
# Time Estimate: 1 hour
# Outcome: Verified that the LSTM model runs without errors.


# %%
# Task 6: Implement Transformer Class
# Action: Write the class for a Transformer model from scratch.
# Define model parameters like the number of heads, layers, and embedding size.
# Time Estimate: 3 hours
# Outcome: Basic Transformer class implemented.


# %%
# Task 8: Test Transformer with Dummy Data
# Action: Run the Transformer model with dummy data to ensure it's working.
# Time Estimate: 1 hour
# Outcome: Verified that the Transformer model runs without errors.


# %%
# Task 9: Implement S4 Class
# Action: Follow the setup guidelines from the S4 GitHub repository and write the class for S4 from scratch.
# Time Estimate: 3 hours
# Outcome: Basic S4 class implemented.


# %%
# Task 11: Test S4 with Dummy Data
# Action: Run the S4 model with dummy data to ensure it's working.
# Time Estimate: 1 hour
# Outcome: Verified that the S4 model runs without errors.


# %%
# Task 12: Prepare LRA Dataset Subset
# Action: Load the chosen LRA subset into the Colab notebook.
# Time Estimate: 1 hour
# Outcome: LRA subset data ready for training.


# %%
# Task 13: Write Training Loop for LSTM
# Action: Implement a training loop to train the LSTM model.
# Time Estimate: 2 hours
# Outcome: Training loop for LSTM ready.

# %%
# Task 14: Train LSTM Model Directly on Task
# Action: Train the LSTM model on the chosen LRA subset.
# Time Estimate: 4 hours (monitor progress)
# Outcome: Trained LSTM model and recorded metrics.

# %%
# Task 15: Download WikiText Dataset
# Action: Load the WikiText dataset into the Colab notebook.
# Time Estimate: 1 hour
# Outcome: WikiText dataset ready for pretraining.

# %%
# Task 16: Pretrain LSTM on WikiText
# Action: Pretrain the LSTM model on the WikiText dataset.
# Time Estimate: 4 hours (monitor progress)
# Outcome: Pretrained LSTM model.

# %%
# Task 17: Fine-tune LSTM on LRA Subset
# Action: Fine-tune the pretrained LSTM model on the chosen LRA subset.
# Time Estimate: 2 hours
# Outcome: Fine-tuned LSTM model and recorded metrics.

# %%
# Task 18: Write Training Loop for Transformer
# Action: Implement a training loop to train the Transformer model.
# Time Estimate: 2 hours
# Outcome: Training loop for Transformer ready.

# %%
# Task 19: Train Transformer Model Directly on Task
# Action: Train the Transformer model on the chosen LRA subset.
# Time Estimate: 4 hours (monitor progress)
# Outcome: Trained Transformer model and recorded metrics.

# %%
# Task 20: Pretrain Transformer on WikiText
# Action: Pretrain the Transformer model on the WikiText dataset.
# Time Estimate: 4 hours (monitor progress)
# Outcome: Pretrained Transformer model.

# %%
# Task 21: Fine-tune Transformer on LRA Subset
# Action: Fine-tune the pretrained Transformer model on the chosen LRA subset.
# Time Estimate: 2 hours
# Outcome: Fine-tuned Transformer model and recorded metrics.

# %%
# Task 22: Write Training Loop for S4
# Action: Implement a training loop to train the S4 model.
# Time Estimate: 2 hours
# Outcome: Training loop for S4 ready.

# %%
# Task 23: Train S4 Model Directly on Task
# Action: Train the S4 model on the chosen LRA subset.
# Time Estimate: 4 hours (monitor progress)
# Outcome: Trained S4 model and recorded metrics.

# %%
# Task 24: Pretrain S4 on WikiText
# Action: Pretrain the S4 model on the WikiText dataset.
# Time Estimate: 4 hours (monitor progress)
# Outcome: Pretrained S4 model.

# %%
# Task 25: Fine-tune S4 on LRA Subset
# Action: Fine-tune the pretrained S4 model on the chosen LRA subset.
# Time Estimate: 2 hours
# Outcome: Fine-tuned S4 model and recorded metrics.

# %%
# Task 26: Evaluate All Models
# Action: Run evaluations on all trained models and collect metrics.
# Time Estimate: 3 hours
# Outcome: Comprehensive metrics for all models and training strategies.

# Define the experiment setup
architectures = [LSTM, Transformer, S4]
datasets = [ListOpsDataset]
strategies = [DirectTrainingStrategy]

# Run the experiment and display the results
results_df = run_experiment(architectures, datasets, strategies)
print(results_df)

# %%
# Task 27: Create Comparison Table
# Action: Summarize the metrics in a comparison table.
# Time Estimate: 1 hour
# Outcome: Comparison table ready.

# %%
# Task 28: Write Conclusion and Report
# Action: Write the conclusions based on the comparison.
# Time Estimate: 3 hours
# Outcome: Completed report.

# %%
# Task 29: Finalize Colab Notebook
# Action: Ensure the Colab notebook is well-documented and shareable.
# Time Estimate: 1 hour
# Outcome: Ready-to-submit Colab notebook.

# %%
# Task 30: Submit Assignment
# Action: Share the Colab notebook with the specified email.
# Time Estimate: 30 minutes
