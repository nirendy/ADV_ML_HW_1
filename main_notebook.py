#%% md
# !pip install torch tensorboard
#%%
import torch

from datasets.mathqa_dataset import MathQADataset
from datasets.retrieval_dataset import RetrievalDataset


#%%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#%%
from configs.try1 import config
from utils.experiment_runner import run_experiment
from datasets.listops_dataset import ListOpsDataset
from models.lstm import LSTMArchitecture
from models.transformer import TransformerArchitecture

from models.s4 import S4Architecture

architectures = [
    LSTMArchitecture(config['lstm']),
    TransformerArchitecture(config['transformer']),
    S4Architecture(config['s4'])
]
pretrain_datasets = [None, MathQADataset(), RetrievalDataset()]
finetune_datasets = [ListOpsDataset()]

# Run the experiment and display the results
results_df = run_experiment(architectures, pretrain_datasets, finetune_datasets)
#%%
print(results_df)
