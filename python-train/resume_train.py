"""Resume training from latest checkpoint"""
import torch
from pathlib import Path
import re

from train import Trainer, TrainingConfig


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
    # Create config with same parameters as original training
    config = TrainingConfig(
        # For quick testing - remove max_samples for full training
        max_samples=80000,
        num_epochs=50,
        batch_size=16,
        learning_rate=1e-4,
        warmup_steps=500,

        # Sample generation
        generate_samples=True,
        sample_interval=5,
        num_samples_per_prompt=4,
        use_ddim=True,
        ddim_steps=50,

        # Quality hyperparameters
        channels=[64, 128, 256],
        text_embed_dim=32,
        use_mid_attn=True,
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
