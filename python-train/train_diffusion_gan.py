"""Training script for text-conditioned diffusion model with GAN adversarial training"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, SequentialLR
import json
import os
import time
from pathlib import Path
from tqdm import tqdm
import numpy as np

from model import UNet
from dataloader import create_dataloader, DiffusionDataset, NoiseSchedule


# ============================================================================
# Discriminator Architecture
# ============================================================================

class DiscriminatorBlock(nn.Module):
    """Discriminator block with spectral normalization for stability"""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 2):
        super().__init__()
        self.conv = nn.utils.spectral_norm(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=stride, padding=1)
        )
        self.norm = nn.InstanceNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.norm(self.conv(x)))


class PatchGANDiscriminator(nn.Module):
    """PatchGAN discriminator for adversarial training

    Evaluates images at multiple scales for better quality assessment.
    Uses spectral normalization for training stability.
    """

    def __init__(self, in_channels: int = 3, base_channels: int = 64):
        super().__init__()

        # Initial layer (no normalization)
        self.initial = nn.Sequential(
            nn.utils.spectral_norm(
                nn.Conv2d(in_channels, base_channels, kernel_size=4, stride=2, padding=1)
            ),
            nn.LeakyReLU(0.2)
        )

        # Downsampling blocks
        self.block1 = DiscriminatorBlock(base_channels, base_channels * 2, stride=2)  # 64->32
        self.block2 = DiscriminatorBlock(base_channels * 2, base_channels * 4, stride=2)  # 32->16
        self.block3 = DiscriminatorBlock(base_channels * 4, base_channels * 8, stride=2)  # 16->8

        # Final classification layer
        self.final = nn.utils.spectral_norm(
            nn.Conv2d(base_channels * 8, 1, kernel_size=4, stride=1, padding=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, 3, 64, 64] input images
        Returns:
            [batch, 1, H, W] patch predictions
        """
        h = self.initial(x)  # [batch, 64, 32, 32]
        h = self.block1(h)   # [batch, 128, 16, 16]
        h = self.block2(h)   # [batch, 256, 8, 8]
        h = self.block3(h)   # [batch, 512, 4, 4]
        h = self.final(h)    # [batch, 1, 3, 3]
        return h


# ============================================================================
# Training Configuration
# ============================================================================

class DiffusionGANConfig:
    """Training configuration for diffusion + GAN"""
    def __init__(
        self,
        # Paths
        json_dir: str = str(Path(os.environ.get("MINI_PIC_DATA_ROOT", "D:/DiffusionDB")) / "images-64-augmented"),
        image_dir: str = str(Path(os.environ.get("MINI_PIC_DATA_ROOT", "D:/DiffusionDB")) / "images-64-augmented"),
        tokenizer_path: str = "../tokenizer.json",
        checkpoint_dir: str = "./checkpoints_gan",

        # Dataset
        max_samples: int = None,
        train_ratio: float = 0.8,

        # Training
        num_epochs: int = 200,
        batch_size: int = 32,
        num_workers: int = 4,

        # Optimizer - Generator (UNet)
        gen_learning_rate: float = 1e-4,
        gen_weight_decay: float = 1e-2,
        gen_betas: tuple = (0.9, 0.999),

        # Optimizer - Discriminator
        disc_learning_rate: float = 4e-4,  # Usually higher than generator
        disc_weight_decay: float = 1e-2,
        disc_betas: tuple = (0.5, 0.999),  # Lower beta1 for discriminator

        # Loss weights
        diffusion_weight: float = 1.0,  # Weight for diffusion MSE loss
        adversarial_weight: float = 0.1,  # Weight for adversarial loss (start low)
        r1_gamma: float = 10.0,  # R1 gradient penalty weight

        # Training strategy
        disc_steps_per_gen: int = 1,  # Discriminator steps per generator step
        warmup_epochs: int = 5,  # Train diffusion only before adding GAN
        adversarial_start_epoch: int = 5,  # When to start adversarial training

        # Scheduler
        warmup_steps: int = 1000,

        # Model
        vocab_size: int = 8192,
        text_embed_dim: int = 32,
        time_embed_dim: int = 32,
        channels: list = None,
        use_mid_attn: bool = False,
        resnet_blocks_per_level: int = 1,

        # Discriminator
        disc_base_channels: int = 64,

        # Logging
        log_interval: int = 100,
        save_interval: int = 5,

        # Sampling during training
        generate_samples: bool = True,
        sample_interval: int = 5,
        sample_prompts: list = None,
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

        self.gen_learning_rate = gen_learning_rate
        self.gen_weight_decay = gen_weight_decay
        self.gen_betas = gen_betas

        self.disc_learning_rate = disc_learning_rate
        self.disc_weight_decay = disc_weight_decay
        self.disc_betas = disc_betas

        self.diffusion_weight = diffusion_weight
        self.adversarial_weight = adversarial_weight
        self.r1_gamma = r1_gamma

        self.disc_steps_per_gen = disc_steps_per_gen
        self.warmup_epochs = warmup_epochs
        self.adversarial_start_epoch = adversarial_start_epoch

        self.warmup_steps = warmup_steps

        self.vocab_size = vocab_size
        self.text_embed_dim = text_embed_dim
        self.time_embed_dim = time_embed_dim
        self.channels = channels or [16, 32, 64]
        self.use_mid_attn = use_mid_attn
        self.resnet_blocks_per_level = resnet_blocks_per_level

        self.disc_base_channels = disc_base_channels

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


# ============================================================================
# Trainer
# ============================================================================

class DiffusionGANTrainer:
    """Training loop for diffusion model with GAN adversarial training"""

    def __init__(self, config: DiffusionGANConfig):
        self.config = config

        # Set seed
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

        # Create checkpoint directory
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        # Save config
        with open(Path(config.checkpoint_dir) / "config.json", "w") as f:
            json.dump(config.to_dict(), f, indent=2)

        print("=== Diffusion + GAN Training Configuration ===")
        print(f"Device: {config.device}")
        print(f"Batch size: {config.batch_size}")
        print(f"Generator LR: {config.gen_learning_rate}")
        print(f"Discriminator LR: {config.disc_learning_rate}")
        print(f"Warmup steps: {config.warmup_steps}")
        print(f"Epochs: {config.num_epochs}")
        print(f"Adversarial start epoch: {config.adversarial_start_epoch}")
        print(f"Diffusion weight: {config.diffusion_weight}")
        print(f"Adversarial weight: {config.adversarial_weight}")
        print(f"Model channels: {config.channels}")
        print("=" * 45 + "\n")

        # Load dataset
        print("Loading dataset...")
        full_dataset = DiffusionDataset(
            config.json_dir,
            config.image_dir,
            config.tokenizer_path,
            config.max_samples,
        )

        self.vocab_size = full_dataset.tokenizer.get_vocab_size()
        self.tokenizer = full_dataset.tokenizer
        self.noise_schedule = full_dataset.noise_schedule
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

        # Create generator (UNet)
        print("Creating generator (UNet)...")
        self.generator = UNet(
            vocab_size=self.vocab_size,
            text_embed_dim=config.text_embed_dim,
            time_embed_dim=config.time_embed_dim,
            use_mid_attn=config.use_mid_attn,
            resnet_blocks_per_level=config.resnet_blocks_per_level,
            channels=config.channels,
        ).to(config.device)

        gen_params = sum(p.numel() for p in self.generator.parameters())
        print(f"Generator parameters: {gen_params:,}\n")

        # Create discriminator
        print("Creating discriminator...")
        self.discriminator = PatchGANDiscriminator(
            in_channels=3,
            base_channels=config.disc_base_channels,
        ).to(config.device)

        disc_params = sum(p.numel() for p in self.discriminator.parameters())
        print(f"Discriminator parameters: {disc_params:,}\n")

        # Create optimizers
        self.optimizer_G = AdamW(
            self.generator.parameters(),
            lr=config.gen_learning_rate,
            weight_decay=config.gen_weight_decay,
            betas=config.gen_betas,
        )

        self.optimizer_D = AdamW(
            self.discriminator.parameters(),
            lr=config.disc_learning_rate,
            weight_decay=config.disc_weight_decay,
            betas=config.disc_betas,
        )

        # Create learning rate scheduler with warmup for generator
        warmup_scheduler = LinearLR(
            self.optimizer_G,
            start_factor=1e-6 / config.gen_learning_rate,
            end_factor=1.0,
            total_iters=config.warmup_steps,
        )

        from torch.optim.lr_scheduler import ConstantLR
        constant_scheduler = ConstantLR(
            self.optimizer_G,
            factor=1.0,
            total_iters=len(self.train_loader) * config.num_epochs - config.warmup_steps,
        )

        self.scheduler_G = SequentialLR(
            self.optimizer_G,
            schedulers=[warmup_scheduler, constant_scheduler],
            milestones=[config.warmup_steps],
        )

        print(f"Generator optimizer: AdamW (lr={config.gen_learning_rate})")
        print(f"Discriminator optimizer: AdamW (lr={config.disc_learning_rate})")
        print(f"Generator scheduler: Linear warmup for {config.warmup_steps} steps\n")

        # Training state
        self.start_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')

    def compute_r1_penalty(self, real_images: torch.Tensor) -> torch.Tensor:
        """Compute R1 gradient penalty for discriminator regularization"""
        real_images.requires_grad_(True)
        real_pred = self.discriminator(real_images)

        # Compute gradients
        grads = torch.autograd.grad(
            outputs=real_pred.sum(),
            inputs=real_images,
            create_graph=True,
            only_inputs=True,
        )[0]

        # R1 penalty: ||grad||^2
        r1_penalty = grads.pow(2).reshape(grads.shape[0], -1).sum(1).mean()
        return r1_penalty

    def train_discriminator(self, real_images: torch.Tensor, fake_images: torch.Tensor) -> dict:
        """Train discriminator for one step"""
        self.optimizer_D.zero_grad()

        # Real images
        real_pred = self.discriminator(real_images)
        real_loss = F.softplus(-real_pred).mean()

        # Fake images
        fake_pred = self.discriminator(fake_images.detach())
        fake_loss = F.softplus(fake_pred).mean()

        # Total loss
        disc_loss = real_loss + fake_loss

        # R1 gradient penalty (apply every N steps for efficiency)
        if self.global_step % 16 == 0:
            r1_penalty = self.compute_r1_penalty(real_images)
            disc_loss = disc_loss + self.config.r1_gamma * r1_penalty * 0.5
        else:
            r1_penalty = torch.tensor(0.0)

        disc_loss.backward()
        self.optimizer_D.step()

        return {
            'disc_loss': disc_loss.item(),
            'disc_real': real_pred.mean().item(),
            'disc_fake': fake_pred.mean().item(),
            'r1_penalty': r1_penalty.item(),
        }

    def generate_denoised_samples(self, batch: dict) -> torch.Tensor:
        """Generate denoised images from noisy inputs (single-step for efficiency)"""
        with torch.no_grad():
            # Predict noise
            predicted_noise = self.generator(
                batch['noisy_images'],
                batch['timesteps'],
                batch['text_tokens'],
            )

            # Denoise (approximate, using noise prediction)
            # For efficiency, we do single-step denoising rather than full sampling
            timesteps = batch['timesteps']
            sqrt_alpha_bar = torch.tensor(
                [self.noise_schedule.sqrt_alpha_bars[t.item()] for t in timesteps],
                device=timesteps.device
            ).view(-1, 1, 1, 1)
            sqrt_one_minus_alpha_bar = torch.tensor(
                [self.noise_schedule.sqrt_one_minus_alpha_bars[t.item()] for t in timesteps],
                device=timesteps.device
            ).view(-1, 1, 1, 1)

            # Approximate denoised image
            denoised = (batch['noisy_images'] - sqrt_one_minus_alpha_bar * predicted_noise) / sqrt_alpha_bar
            denoised = torch.clamp(denoised, -1.0, 1.0)

        return denoised

    def train_epoch(self, epoch: int):
        """Train for one epoch"""
        self.generator.train()
        self.discriminator.train()

        total_gen_loss = 0
        total_diff_loss = 0
        total_adv_loss = 0
        total_disc_loss = 0

        use_adversarial = epoch >= self.config.adversarial_start_epoch

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.num_epochs}")

        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            batch = {k: v.to(self.config.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}

            # ===== Train Generator =====
            self.optimizer_G.zero_grad()

            # Diffusion loss (always computed)
            diff_loss, predicted_noise = self.generator.compute_loss(batch)
            gen_loss = self.config.diffusion_weight * diff_loss

            # Adversarial loss (only after warmup)
            if use_adversarial:
                # Generate denoised samples
                fake_images = self.generate_denoised_samples(batch)

                # Generator tries to fool discriminator
                fake_pred = self.discriminator(fake_images)
                adv_loss = F.softplus(-fake_pred).mean()
                gen_loss = gen_loss + self.config.adversarial_weight * adv_loss
            else:
                adv_loss = torch.tensor(0.0)

            gen_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)
            self.optimizer_G.step()
            self.scheduler_G.step()

            # ===== Train Discriminator =====
            disc_metrics = {'disc_loss': 0.0}
            if use_adversarial:
                for _ in range(self.config.disc_steps_per_gen):
                    fake_images = self.generate_denoised_samples(batch)
                    disc_metrics = self.train_discriminator(batch['images'], fake_images)

            # Update metrics
            total_gen_loss += gen_loss.item()
            total_diff_loss += diff_loss.item()
            total_adv_loss += adv_loss.item()
            total_disc_loss += disc_metrics['disc_loss']
            self.global_step += 1

            # Update progress bar
            pbar_dict = {
                'diff': f'{diff_loss.item():.4f}',
                'gen': f'{gen_loss.item():.4f}',
                'lr': f'{self.optimizer_G.param_groups[0]["lr"]:.2e}',
            }
            if use_adversarial:
                pbar_dict['adv'] = f'{adv_loss.item():.4f}'
                pbar_dict['disc'] = f'{disc_metrics["disc_loss"]:.4f}'
            pbar.set_postfix(pbar_dict)

            # Log periodically
            if self.global_step % self.config.log_interval == 0:
                avg_diff = total_diff_loss / (batch_idx + 1)
                log_str = f"\nStep {self.global_step}: diff_loss={avg_diff:.4f}"
                if use_adversarial:
                    avg_adv = total_adv_loss / (batch_idx + 1)
                    avg_disc = total_disc_loss / (batch_idx + 1)
                    log_str += f", adv_loss={avg_adv:.4f}, disc_loss={avg_disc:.4f}"
                print(log_str)

        metrics = {
            'gen_loss': total_gen_loss / len(self.train_loader),
            'diff_loss': total_diff_loss / len(self.train_loader),
            'adv_loss': total_adv_loss / len(self.train_loader) if use_adversarial else 0.0,
            'disc_loss': total_disc_loss / len(self.train_loader) if use_adversarial else 0.0,
        }
        return metrics

    @torch.no_grad()
    def validate(self):
        """Run validation"""
        self.generator.eval()
        total_loss = 0

        for batch in tqdm(self.val_loader, desc="Validating"):
            batch = {k: v.to(self.config.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}

            loss, _ = self.generator.compute_loss(batch)
            total_loss += loss.item()

        return total_loss / len(self.val_loader)

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'optimizer_D_state_dict': self.optimizer_D.state_dict(),
            'scheduler_G_state_dict': self.scheduler_G.state_dict(),
            'best_val_loss': self.best_val_loss,
        }

        checkpoint_path = Path(self.config.checkpoint_dir) / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")

        if is_best:
            best_path = Path(self.config.checkpoint_dir) / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"Saved best model to {best_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.config.device)

        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        self.optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
        self.scheduler_G.load_state_dict(checkpoint['scheduler_G_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']

        print(f"Loaded checkpoint from {checkpoint_path}")
        print(f"Resuming from epoch {self.start_epoch}, step {self.global_step}")

    @torch.no_grad()
    def generate_sample_images(self, epoch: int):
        """Generate sample images during training"""
        from inference import generate_samples, save_image_grid

        samples_dir = Path(self.config.checkpoint_dir) / "samples"
        samples_dir.mkdir(exist_ok=True)

        if self.config.sample_prompts:
            prompts = self.config.sample_prompts
        else:
            prompts = []
            for i in range(min(4, len(self.full_dataset.items))):
                prompts.append(self.full_dataset.items[i]['prompt'])

        print(f"\nGenerating samples with prompts:")
        for i, prompt in enumerate(prompts):
            print(f"  {i+1}. {prompt[:80]}...")

        images = generate_samples(
            self.generator,
            self.tokenizer,
            prompts,
            device=self.config.device,
            num_samples=self.config.num_samples_per_prompt,
            use_ddim=self.config.use_ddim,
            ddim_steps=self.config.ddim_steps,
        )

        output_path = samples_dir / f"epoch_{epoch:03d}.png"
        save_image_grid(images, str(output_path), nrow=self.config.num_samples_per_prompt)
        print(f"Saved samples to {output_path}\n")

    def train(self):
        """Main training loop"""
        print("Starting training...\n")

        for epoch in range(self.start_epoch, self.config.num_epochs):
            start_time = time.time()

            # Train
            metrics = self.train_epoch(epoch)

            # Validate
            val_loss = self.validate()

            epoch_time = time.time() - start_time

            # Print epoch summary
            use_adversarial = epoch >= self.config.adversarial_start_epoch
            summary = (f"\nEpoch {epoch+1}/{self.config.num_epochs} - "
                      f"Diff: {metrics['diff_loss']:.4f}, Val: {val_loss:.4f}")
            if use_adversarial:
                summary += f", Adv: {metrics['adv_loss']:.4f}, Disc: {metrics['disc_loss']:.4f}"
            summary += f", Time: {epoch_time:.1f}s"
            print(summary)

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


# ============================================================================
# Main
# ============================================================================

def main():
    """Main training function"""
    config = DiffusionGANConfig(
        # Dataset
        max_samples=80000,
        num_epochs=100,
        batch_size=16,

        # Optimizer
        gen_learning_rate=1e-4,
        disc_learning_rate=4e-4,
        warmup_steps=500,

        # Loss weights
        diffusion_weight=1.0,
        adversarial_weight=0.1,  # Start with low weight
        r1_gamma=10.0,

        # Training strategy
        disc_steps_per_gen=1,
        adversarial_start_epoch=1,  # Start GAN training after x epochs

        # Sample generation
        generate_samples=True,
        sample_interval=5,
        num_samples_per_prompt=4,
        use_ddim=True,
        ddim_steps=50,

        # Model architecture
        channels=[64, 128, 256],
        use_mid_attn=True,
        text_embed_dim=32,
        time_embed_dim=32,
        resnet_blocks_per_level=1,

        # Discriminator
        disc_base_channels=64,
    )

    trainer = DiffusionGANTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
