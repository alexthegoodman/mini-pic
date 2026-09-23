"""Resume training from latest checkpoint"""
import json
import torch
from pathlib import Path
import re

from train import Trainer, TrainingConfig


def load_resumed_config(checkpoint_dir: str, **overrides) -> TrainingConfig:
    """Rebuilds the config exactly as train.py saved it for this run, so a
    resume can't silently retrain with different architecture/optimizer
    hyperparameters than the checkpoint was actually produced with (that drift
    is what previously made this file's channels/text_embed_dim/learning_rate
    disagree with train.py's). Only `overrides` - training-duration knobs like
    num_epochs, not architecture - are allowed to differ from the saved run.
    """
    config_path = Path(checkpoint_dir) / "config.json"
    with open(config_path) as f:
        saved = json.load(f)
    saved.update(overrides)
    return TrainingConfig(**saved)


def find_latest_checkpoint(checkpoint_dir: str) -> str:
    """Find the latest checkpoint file by epoch number"""
    checkpoint_path = Path(checkpoint_dir)

    if not checkpoint_path.exists():
        raise ValueError(f"Checkpoint directory {checkpoint_dir} does not exist")

    # Find all checkpoint files
    checkpoint_files = list(checkpoint_path.glob("checkpoint_epoch_*.pt"))

    if not checkpoint_files:
        raise ValueError(f"No checkpoint files found in {checkpoint_dir}")

    # Extract epoch numbers and find the latest
    def get_epoch_num(filepath):
        match = re.search(r'checkpoint_epoch_(\d+)\.pt', filepath.name)
        return int(match.group(1)) if match else -1

    latest_checkpoint = max(checkpoint_files, key=get_epoch_num)
    return str(latest_checkpoint)


def main():
    """Resume training from latest checkpoint"""
    # Reload the exact config the checkpoint was trained with; only
    # duration/logging knobs are safe to override here.
    config = load_resumed_config(
        "./checkpoints",
        num_epochs=50,
        sample_interval=5,
    )

    # Create trainer
    trainer = Trainer(config)

    # Find and load latest checkpoint
    try:
        latest_checkpoint = find_latest_checkpoint(config.checkpoint_dir)
        print(f"\nFound latest checkpoint: {latest_checkpoint}")
        trainer.load_checkpoint(latest_checkpoint)
    except ValueError as e:
        print(f"Error: {e}")
        print("Starting training from scratch instead...")

    # Continue training
    trainer.train()


if __name__ == "__main__":
    main()
