# Advanced Deep Learning Project

This project aims to analyze the performance of state-of-the-art architectures for long context tasks using the Long
Range Arena (LRA) dataset. The focus is on three network architectures: LSTM, Transformer, and State Space Models (S4),
and three training strategies: direct training, pretraining on an external dataset, and pretraining on an LRA subset.

## Table of Contents

- [Installation](#installation)
- [Dataset Download](#dataset-download)
- [Directory Structure](#directory-structure)
- [Running the Experiment](#running-the-experiment)

## Installation

To set up the project, follow these steps:

```bash
# 1. Unzip the project folder
tar -xvzf ADV_ML_HW_1.tar.gz
cd ADV_ML_HW_1

# 2. Create a virtual environment and activate it.

python3 -m venv venv
source venv/bin/activate
  
# 3. Install the required packages.
pip install -r requirements.txt
```

## Dataset Download

To download the LRA dataset, follow these steps:

```bash
# 1. Download the dataset.
wget https://storage.googleapis.com/long-range-arena/lra_release.gz
mkdir -p ./data/raw
tar -xvzf lra_release.gz --strip-components=2 -C ./data/raw lra_release/lra_release/listops-1000
tar -xvzf lra_release.gz --strip-components=2 -C ./data/raw lra_release/lra_release/retrieval

```

## Directory Structure

```plaintext
.
├── src/
│   ├── configs/ # Contains the configuration files for the experiments.
│   │   ├── try1.py
│   │   └── ...
│   ├── datasets/ # Contains the dataset classes.
│   │   ├── base_dataset.py # Base class for the datasets.
│   │   ├── listops_dataset.py # Main dataset
│   │   ├── mathqa_dataset.py # External dataset
│   │   └── retrieval_dataset.py # LRA secondary dataset
│   ├── models/ # Contains the model classes.
│   │   ├── architecture.py # Base class for the models.
│   │   ├── lstm.py
│   │   ├── s4.py
│   │   └── transformer.py
│   └── utils/
│       ├── config_types.py
│       ├── experiment_runner.py
│       └── metrics.py
├── requirements.txt
└── main_notebook.py # Main notebook to run the experiments.

```

## Running the Experiment

1. Follow installation and download instructions.
2. Modify the configuration file in the `configs` directory.
3. Run the main notebook, choose the configuration file, and run the experiment.
4. View the results using tensorboard to visualize the metrics.

```bash
tensorboard --logdir=experiments
```
