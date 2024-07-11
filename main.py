import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from abc import ABC, abstractmethod
import torch.nn.functional as F
import pandas as pd


# Basic setup

# Task 1: Review LRA Dataset Subsets
# Action: Look at the descriptions of the six subsets in the LRA dataset.
# Time Estimate: 30 minutes
# Outcome: Choose one subset for the assignment.
# Chosen Dataset: ListOps



# %%
# Task 2: Set Up Google Colab Environment
# Action: Open Google Colab and set up a new notebook.
# Time Estimate: 30 minutes
# Outcome: Colab notebook ready for coding.
# !pip install torch torchvision transformers

# %%


# Task 3: Implement LSTM Class
# Action: Write the class for an LSTM model from scratch.
# Define input size, hidden layers, and output size.
# Time Estimate: 2 hours
# Outcome: Basic LSTM class implemented.


# Simple LSTM Cell Implementation

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

# Transformer Encoder Layer Implementation
# Transformer Encoder Layer Implementation


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


# Simplified S4 Architecture Implementation


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
