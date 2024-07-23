#! /bin/bash

#SBATCH --job-name=imdb_transformer # Job name
#SBATCH --output=imdb_transformer.out # redirect stdout
#SBATCH --error=imdb_transformer.err # redirect stderr
#SBATCH --partition=gpu-a100-killable # (see resources section)
#SBATCH --time=1200 # max time (minutes)
#SBATCH --signal=USR1@120 # how to end job when time’s up
#SBATCH --nodes=1 # number of machines
#SBATCH --ntasks=1 # number of processes
#SBATCH --mem=50000 # CPU memory (MB)
#SBATCH --cpus-per-task=1 # CPU cores per process
#SBATCH --gpus=4 # GPUs in total
#SBATCH --account=gpu-research # billing account

echo $CUDA_VISIBLE_DEVICES
nvidia-smi

# Activate the conda environment
source /home/yandex/DL20232024a/nirendy/repos/ADV_ML_HW_1/venv/bin/activate

export PYTHONUNBUFFERED=1

# export CUDA_VISIBLE_DEVICES=2,3

#export TORCH_DISTRIBUTED_DEBUG=DETAIL

BASE=/home/yandex/DL20232024a/nirendy/repos/ADV_ML_HW_1/

#export PYTHONPATH=$PYTHONPATH:$BASE/mamba_py:$BASE/mamba_cuda


# Print diagnostic information
echo "Python version: $(python --version)"
echo "PYTHONPATH: $PYTHONPATH"
echo $(pwd)

# Change to the working directory
cd /home/yandex/DL20232024a/nirendy/repos/ADV_ML_HW_1/

# Export environment variables


# Change to the working directory
python train_one.py  --config_name large --architecture transformer --finetune_dataset imdb --with_parallel True

# Trap keyboard interrupt (Ctrl+C) to end the job
trap 'echo "Keyboard interrupt received. Ending job."; exit' INT
