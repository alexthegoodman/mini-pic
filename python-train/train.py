"""Training script for text-conditioned diffusion model"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, SequentialLR
import json
import time
from pathlib import Path
from tqdm import tqdm
import numpy as np

from model import UNet
from dataloader import create_dataloader, DiffusionDataset


class TrainingConfig:
    """Training configuration"""
    def __init__(
        self,
        # Paths
        json_dir: str = "../../diffusiondb/unzipped-json/",
        image_dir: str = "../../diffusiondb/unzipped-64/",
        tokenizer_path: str = "../tokenizer.json",
        checkpoint_dir: str = "./checkpoints",

        # Dataset
        max_samples: int = None,  # None = load all
        train_ratio: float = 0.8,

        # Training
        num_epochs: int = 200,
        batch_size: int = 32,
        num_workers: int = 4,

        # Optimizer
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-2,
        betas: tuple = (0.9, 0.999),

        # Scheduler
        warmup_steps: int = 1000,

        # Model
        vocab_size: int = 8192,
        text_embed_dim: int = 32,
        time_embed_dim: int = 32,
        channels: list = None,
        use_mid_attn: bool = False,
        resnet_blocks_per_level: int = 1,

        # Logging
        log_interval: int = 100,
        save_interval: int = 5,  # Save every N epochs

        # Sampling during training
        generate_samples: bool = True,
        sample_interval: int = 5,  # Generate samples every N epochs
        sample_prompts: list = None,  # Custom prompts (default: uses some from dataset)
        num_samples_per_prompt: int = 4,
        use_ddim: bool = True,
        ddim_steps: int = 50,

        # Device
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        seed: int = 1337,
    ):
        self.json_dir = json_dir
        self.image_dir = image_dir
        self.tokenizer_path = tokenizer_path
        self.checkpoint_dir = checkpoint_dir

        self.max_samples = max_samples
        self.train_ratio = train_ratio

        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.betas = betas

        self.warmup_steps = warmup_steps

        self.vocab_size = vocab_size
        self.text_embed_dim = text_embed_dim
        self.time_embed_dim = time_embed_dim
        self.channels = channels or [16, 32, 64]
        self.use_mid_attn = use_mid_attn
        self.resnet_blocks_per_level = resnet_blocks_per_level

        self.log_interval = log_interval
        self.save_interval = save_interval

        self.generate_samples = generate_samples
        self.sample_interval = sample_interval
        self.sample_prompts = sample_prompts
        self.num_samples_per_prompt = num_samples_per_prompt
        self.use_ddim = use_ddim
        self.ddim_steps = ddim_steps

        self.device = device
        self.seed = seed

    def to_dict(self):
        """Convert config to dictionary for saving"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}


class Trainer:
    """Training loop for diffusion model"""

    def __init__(self, config: TrainingConfig):
        self.config = config

        # Set seed
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

        # Create checkpoint directory
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        # Save config
        with open(Path(config.checkpoint_dir) / "config.json", "w") as f:
            json.dump(config.to_dict(), f, indent=2)

        print("=== Diffusion Model Training Configuration ===")
        print(f"Device: {config.device}")
        print(f"Batch size: {config.batch_size}")
        print(f"Learning rate: {config.learning_rate}")
        print(f"Warmup steps: {config.warmup_steps}")
        print(f"Epochs: {config.num_epochs}")
        print(f"Weight decay: {config.weight_decay}")
        print(f"Model channels: {config.channels}")
        print(f"Text embed dim: {config.text_embed_dim}")
        print("=" * 45 + "\n")

        # Load dataset
        print("Loading dataset...")
        full_dataset = DiffusionDataset(
            config.json_dir,
            config.image_dir,
            config.tokenizer_path,
            config.max_samples,
        )

        # Update vocab size from tokenizer
        self.vocab_size = full_dataset.tokenizer.get_vocab_size()
        self.tokenizer = full_dataset.tokenizer
        print(f"Tokenizer vocab size: {self.vocab_size}")

        # Split dataset
        total_size = len(full_dataset)
        train_size = int(total_size * config.train_ratio)
        val_size = total_size - train_size

        indices = list(range(total_size))
        np.random.shuffle(indices)

        train_indices = indices[:train_size]
        val_indices = indices[train_size:]

        train_dataset = Subset(full_dataset, train_indices)
        val_dataset = Subset(full_dataset, val_indices)

        # Store dataset for getting sample prompts
        self.full_dataset = full_dataset

        print(f"Train size: {train_size}")
        print(f"Val size: {val_size}\n")

        # Create dataloaders
        from dataloader import collate_fn
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )

        # Create model
        print("Creating model...")
        self.model = UNet(
            vocab_size=self.vocab_size,
            text_embed_dim=config.text_embed_dim,
            time_embed_dim=config.time_embed_dim,
            use_mid_attn=config.use_mid_attn,
            resnet_blocks_per_level=config.resnet_blocks_per_level,
            channels=config.channels,
        ).to(config.device)

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}\n")

        # Create optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=config.betas,
        )

        # Create learning rate scheduler with warmup
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=1e-6 / config.learning_rate,
            end_factor=1.0,
            total_iters=config.warmup_steps,
        )

        # Constant LR after warmup (could add cosine decay here)
        from torch.optim.lr_scheduler import ConstantLR
        constant_scheduler = ConstantLR(
            self.optimizer,
            factor=1.0,
            total_iters=len(self.train_loader) * config.num_epochs - config.warmup_steps,
        )

        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, constant_scheduler],
            milestones=[config.warmup_steps],
        )

        print(f"Optimizer: AdamW (lr={config.learning_rate}, weight_decay={config.weight_decay})")
        print(f"Scheduler: Linear warmup for {config.warmup_steps} steps\n")

        # Training state
        self.start_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')

    def train_epoch(self, epoch: int):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.num_epochs}")

        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            batch = {k: v.to(self.config.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}

            # Forward pass
            loss, _ = self.model.compute_loss(batch)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping (optional but helpful for stability)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()
            self.scheduler.step()

            # Update metrics
            total_loss += loss.item()
            self.global_step += 1

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}',
            })

            # Log periodically
            if self.global_step % self.config.log_interval == 0:
                avg_loss = total_loss / (batch_idx + 1)
                print(f"\nStep {self.global_step}: loss={avg_loss:.4f}, lr={self.optimizer.param_groups[0]['lr']:.2e}")

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def validate(self):
        """Run validation"""
        self.model.eval()
        total_loss = 0

        for batch in tqdm(self.val_loader, desc="Validating"):
            # Move batch to device
            batch = {k: v.to(self.config.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}

            # Forward pass
            loss, _ = self.model.compute_loss(batch)
            total_loss += loss.item()

        return total_loss / len(self.val_loader)

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
        }

        # Save latest checkpoint
        checkpoint_path = Path(self.config.checkpoint_dir) / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")

        # Save best model
        if is_best:
            best_path = Path(self.config.checkpoint_dir) / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"Saved best model to {best_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.config.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']

        print(f"Loaded checkpoint from {checkpoint_path}")
        print(f"Resuming from epoch {self.start_epoch}, step {self.global_step}")

    @torch.no_grad()
    def generate_sample_images(self, epoch: int):
        """Generate sample images during training"""
        from inference import generate_samples, save_image_grid

        # Create samples directory
        samples_dir = Path(self.config.checkpoint_dir) / "samples"
        samples_dir.mkdir(exist_ok=True)

        # Get prompts
        if self.config.sample_prompts:
            prompts = self.config.sample_prompts
        else:
            # Use random prompts from dataset
            prompts = []
            for i in range(min(4, len(self.full_dataset.items))):
                prompts.append(self.full_dataset.items[i]['prompt'])

        print(f"\nGenerating samples with prompts:")
        for i, prompt in enumerate(prompts):
            print(f"  {i+1}. {prompt[:80]}...")

        # Generate images
        images = generate_samples(
            self.model,
            self.tokenizer,
            prompts,
            device=self.config.device,
            num_samples=self.config.num_samples_per_prompt,
            use_ddim=self.config.use_ddim,
            ddim_steps=self.config.ddim_steps,
        )

        # Save grid
        output_path = samples_dir / f"epoch_{epoch:03d}.png"
        save_image_grid(images, str(output_path), nrow=self.config.num_samples_per_prompt)
        print(f"Saved samples to {output_path}\n")

    def train(self):
        """Main training loop"""
        print("Starting training...\n")

        for epoch in range(self.start_epoch, self.config.num_epochs):
            start_time = time.time()

            # Train
            train_loss = self.train_epoch(epoch)

            # Validate
            val_loss = self.validate()

            epoch_time = time.time() - start_time

            # Print epoch summary
            print(f"\nEpoch {epoch+1}/{self.config.num_epochs} - "
                  f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                  f"Time: {epoch_time:.1f}s")

            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss

            if (epoch + 1) % self.config.save_interval == 0 or is_best:
                self.save_checkpoint(epoch, is_best=is_best)

            # Generate sample images
            if self.config.generate_samples and (epoch + 1) % self.config.sample_interval == 0:
                self.generate_sample_images(epoch)

            print("-" * 80 + "\n")

        print("Training complete!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")


def main():
    """Main training function"""
    config = TrainingConfig(
        # For quick testing - remove max_samples for full training
        max_samples=80000,  # Use 1000 samples for testing
        num_epochs=50,
        batch_size=16,
        learning_rate=1e-4,
        warmup_steps=500,

        # Sample generation
        generate_samples=True,
        sample_interval=5,  # Generate every 5 epochs
        num_samples_per_prompt=4,
        use_ddim=True,
        ddim_steps=50,

        # Quality hyperparameters
        # channels=[32, 64, 128],
        channels=[64, 128, 256],
        # channels=[128, 256, 512],
        text_embed_dim=32,
        use_mid_attn=True,
    )

    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
