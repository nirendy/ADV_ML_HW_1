import itertools
from pathlib import Path
import shutil
import sys

from src.consts import PATHS
from src.types import ARCH
from src.types import DATASET


def copy_files(source_dir: Path, target_dir: Path, dry_run: bool):
    # Check if the destination directory exists
    if not target_dir.exists():
        raise FileNotFoundError(f"Destination directory {target_dir} does not exist.")

    for file_path in source_dir.glob('**/*'):
        if file_path.is_file():
            # Determine the destination path
            relative_path = file_path.relative_to(source_dir)
            destination_path = target_dir / relative_path

            # Check if destination directory exists without creating it
            if not destination_path.parent.exists():
                if dry_run:
                    print(f"Error: Destination subdirectory {destination_path.parent} does not exist.")
                    continue
                else:
                    raise FileNotFoundError(f"Destination subdirectory {destination_path.parent} does not exist.")

            if dry_run:
                print(f"Would copy {file_path} to {destination_path}")
                pass
            else:
                # Copy the file
                shutil.copy(file_path, destination_path)


def move_latest_checkpoint(source_dir: Path, target_dir: Path, dry_run: bool):
    for experiment_dir in source_dir.iterdir():
        for run_dir in experiment_dir.iterdir():
            files = list(run_dir.glob('*.pth'))
            if not files:
                if dry_run:
                    print(f"Remove source directory: No checkpoint files found in {run_dir}")
                else:
                    run_dir.rmdir()
                continue

            latest_checkpoint = max(
                run_dir.glob('*.pth'),
                key=lambda file: int(file.stem)
            )

            relative_path = run_dir.relative_to(source_dir)
            destination_path = target_dir / relative_path
            if not destination_path.exists():
                if dry_run:
                    print(
                        f"Remove source directory {run_dir}: Destination directory {destination_path} does not exist."
                    )
                    continue
                else:
                    shutil.rmtree(run_dir)

            source_checkpoint = run_dir / latest_checkpoint.name
            if dry_run:
                print(f"Copy {source_checkpoint} to {destination_path}")
            else:
                shutil.copy(source_checkpoint, destination_path)


def clean_partial_experiment(target_dir: Path, dry_run: bool):
    expected_exts = {'.log', '.json', '.pth'}
    if not target_dir.exists():
        raise FileNotFoundError(f"Destination directory {target_dir} does not exist.")

    for experiment_dir in target_dir.iterdir():
        for run_dir in experiment_dir.iterdir():
            files_exts = set([
                file.suffix
                for file in run_dir.iterdir()
                if file.is_file()
            ])
            missing_exts = expected_exts - files_exts
            if missing_exts:
                if dry_run:
                    print(f"Delete: Missing files with extensions {missing_exts} in {run_dir}")
                    continue
                else:
                    # delete the directory
                    shutil.rmtree(run_dir)
            else:
                print(f"Found all expected files in {run_dir}")


def create_all_dirs_needed(dry_run):
    # use itertools.product to get all combinations of sizes and archs
    for (
            size,
            arch,
            pretrain_dataset,
            dataset,
    ) in itertools.product(*[
        ['medium'],
        ARCH.__members__.values(),
        [None, *DATASET.__members__.values()],
        [DATASET.IMDB],
    ]):
        # create the directory
        experiment_name = '_'.join([
            size,
            arch,
            *([f'pre_{pretrain_dataset}'] if (pretrain_dataset is not None) else []),
            dataset
        ])
        p = PATHS.TENSORBOARD_DIR / experiment_name
        if not p.exists():
            if dry_run:
                print(f"Would create directory {p}")
                continue
            else:
                p.mkdir(parents=True, exist_ok=True)


if __name__ == '__main__':
    dry_run = True
    # copy_files(PATHS.LOGS_DIR, PATHS.TENSORBOARD_DIR, dry_run=dry_run)
    # move_latest_checkpoint(PATHS.CHECKPOINTS_DIR, PATHS.TENSORBOARD_DIR, dry_run=dry_run)
    # clean_partial_experiment(PATHS.TENSORBOARD_DIR, dry_run=dry_run)
    create_all_dirs_needed(dry_run=dry_run)
