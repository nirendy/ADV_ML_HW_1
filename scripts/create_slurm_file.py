import subprocess
from pathlib import Path
from src.consts import IAddArgs

from src.utils.experiment_runner import construct_experiment_name
from src.utils.experiment_runner import create_run_id
from src.types import IArgs


def slurm_template(
        experiment_name: str,
        slurm_output_dir: Path,
        args: IArgs,
        add_args: IAddArgs
) -> str:
    command_str = ' '.join([
        'python',
        'train_one.py',
        *[
            f'--{k} "{v}"'
            for k, v in
            list(args._asdict().items()) + [
                ('with_parallel', add_args.with_parallel),
            ]
            if v is not None
        ]
    ])

    return (
        f"""\
#! /bin/bash

#SBATCH --job-name={experiment_name} # Job name
#SBATCH --output={slurm_output_dir / 'out.log'} # redirect stdout
#SBATCH --error={slurm_output_dir / 'err.log'} # redirect stderr
#SBATCH --partition={add_args.partition} # (see resources section)
#SBATCH --time={add_args.time} # max time (minutes)
#SBATCH --signal={add_args.singal} # how to end job when time’s up
#SBATCH --nodes={add_args.nodes} # number of machines
#SBATCH --ntasks={add_args.ntasks} # number of processes
#SBATCH --mem={add_args.mem} # CPU memory (MB)
#SBATCH --cpus-per-task={add_args.cpus_per_task} # CPU cores per process
#SBATCH --gpus={add_args.gpus} # GPUs in total
#SBATCH --account={add_args.account} # billing account

nvidia-smi
gpustat --no-color -pfu

# Activate the conda environment
source {add_args.workspace}/venv/bin/activate

export PYTHONUNBUFFERED=1

# Print diagnostic information
echo $CUDA_VISIBLE_DEVICES
echo "Python version: $(python --version)"
echo "PYTHONPATH: $PYTHONPATH"
echo $(pwd)

# Change to the working directory
cd {add_args.workspace}

# Export environment variables

{command_str}

# Trap keyboard interrupt (Ctrl+C) to end the job
trap 'echo "Keyboard interrupt received. Ending job."; exit' INT

""")


def create_template(
        args: IArgs,
        add_args: IAddArgs
):
    """

    Returns:
        Path: Path to the slurm file

    """
    experiment_name = construct_experiment_name(
        args.config_name,
        args.architecture,
        args.finetune_dataset,
        args.pretrain_dataset
    )
    args = args._replace(run_id=create_run_id(args.run_id))

    slurm_output_dir = (
            add_args.workspace
            / add_args.outputs_relative_path
            / experiment_name
            / args.run_id
    )

    slurm_output_dir.mkdir(parents=True, exist_ok=True)
    template = slurm_template(experiment_name, slurm_output_dir, args, add_args)

    slurm_path = slurm_output_dir / f"{args.run_id}.slurm"

    slurm_path.write_text(template)

    return slurm_path


def run_slurm(args: IArgs, add_args: IAddArgs):
    slurm_path = create_template(args, add_args)
    print(f"Submitting job, to see the logs run: \nless +F {slurm_path.parent / 'err.log'}")
    subprocess.run(['sbatch', str(slurm_path)])
