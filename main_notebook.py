# %% md
# !pip install torch tensorboard
# %%
from utils.experiment_runner import init_experiment
from utils.experiment_runner import run_experiment

# Example usage
config_name = 'try1'

# (
#     architectures,
#     pretrain_datasets,
#     finetune_datasets,
#     config_name
# ) = init_experiment(config_name)
experiment_parts = init_experiment(config_name)

# Run the experiment and display the results
results_df = run_experiment(*experiment_parts)
# %%
print(results_df)
