"""Inference utilities for diffusion model sampling"""
import torch
import torch.nn.functional as F
from typing import Optional
import numpy as np
from PIL import Image
from dataloader import NoiseSchedule, IMAGE_SIZE, IMAGE_CHANNELS


class DDPMSampler:
    """DDPM sampling for image generation"""

    def __init__(self, model, noise_schedule: NoiseSchedule, device: str = "cuda"):
        self.model = model
        self.noise_schedule = noise_schedule
        self.device = device
        self.num_timesteps = noise_schedule.num_timesteps

    @torch.no_grad()
    def sample(
        self,
        text_tokens: torch.Tensor,
        num_samples: int = 1,
        guidance_scale: float = 1.0,
        show_progress: bool = True,
    ) -> torch.Tensor:
        """
        Generate images using DDPM sampling

        Args:
            text_tokens: [batch_size, seq_len] text token IDs
            num_samples: Number of images to generate per prompt
            guidance_scale: CFG scale (1.0 = no guidance, >1.0 = stronger guidance)
            show_progress: Show sampling progress bar

        Returns:
            Generated images [batch_size * num_samples, 3, 64, 64] in range [-1, 1]
        """
        self.model.eval()

        batch_size = text_tokens.shape[0]
        total_samples = batch_size * num_samples

        # Repeat text tokens for num_samples
        if num_samples > 1:
            text_tokens = text_tokens.repeat_interleave(num_samples, dim=0)

        # Start from pure noise
        x = torch.randn(total_samples, IMAGE_CHANNELS, IMAGE_SIZE, IMAGE_SIZE, device=self.device)

        # Reverse diffusion process
        timesteps = list(range(self.num_timesteps - 1, -1, -1))

        if show_progress:
            from tqdm import tqdm
            timesteps = tqdm(timesteps, desc="Sampling")

        for t in timesteps:
            # Current timestep
            t_batch = torch.full((total_samples,), t, device=self.device, dtype=torch.long)

            # Predict noise
            predicted_noise = self.model(x, t_batch, text_tokens)

            # Get noise schedule parameters
            alpha = self.noise_schedule.alphas[t]
            alpha_bar = self.noise_schedule.alpha_bars[t]
            beta = self.noise_schedule.betas[t]

            # Compute mean
            if t > 0:
                alpha_bar_prev = self.noise_schedule.alpha_bars[t - 1]
            else:
                alpha_bar_prev = 1.0

            # DDPM sampling formula
            coef1 = 1.0 / np.sqrt(alpha)
            coef2 = beta / np.sqrt(1.0 - alpha_bar)

            x = coef1 * (x - coef2 * predicted_noise)

            # Add noise (except at last step)
            if t > 0:
                noise = torch.randn_like(x)
                sigma_t = np.sqrt(beta)
                x = x + sigma_t * noise

        return x

    @torch.no_grad()
    def sample_ddim(
        self,
        text_tokens: torch.Tensor,
        num_samples: int = 1,
        num_steps: int = 50,
        eta: float = 0.0,
        show_progress: bool = True,
    ) -> torch.Tensor:
        """
        Generate images using DDIM sampling (faster)

        Args:
            text_tokens: [batch_size, seq_len] text token IDs
            num_samples: Number of images to generate per prompt
            num_steps: Number of sampling steps (less than num_timesteps for speedup)
            eta: Stochasticity parameter (0 = deterministic, 1 = DDPM)
            show_progress: Show sampling progress bar

        Returns:
            Generated images [batch_size * num_samples, 3, 64, 64] in range [-1, 1]
        """
        self.model.eval()

        batch_size = text_tokens.shape[0]
        total_samples = batch_size * num_samples

        # Repeat text tokens for num_samples
        if num_samples > 1:
            text_tokens = text_tokens.repeat_interleave(num_samples, dim=0)

        # Start from pure noise
        x = torch.randn(total_samples, IMAGE_CHANNELS, IMAGE_SIZE, IMAGE_SIZE, device=self.device)

        # Create timestep schedule (uniformly spaced)
        timestep_schedule = np.linspace(self.num_timesteps - 1, 0, num_steps, dtype=int)

        if show_progress:
            from tqdm import tqdm
            timestep_schedule = tqdm(timestep_schedule, desc="DDIM Sampling")

        for i, t in enumerate(timestep_schedule):
            # Current timestep
            t_batch = torch.full((total_samples,), t, device=self.device, dtype=torch.long)

            # Predict noise
            predicted_noise = self.model(x, t_batch, text_tokens)

            # Get alpha values
            alpha_bar_t = self.noise_schedule.alpha_bars[t]

            # Get next alpha (for next timestep)
            if i < len(timestep_schedule) - 1:
                t_next = timestep_schedule[i + 1]
                alpha_bar_next = self.noise_schedule.alpha_bars[t_next]
            else:
                alpha_bar_next = 1.0

            # Predict x0
            sqrt_alpha_bar_t = np.sqrt(alpha_bar_t)
            sqrt_one_minus_alpha_bar_t = np.sqrt(1.0 - alpha_bar_t)
            pred_x0 = (x - sqrt_one_minus_alpha_bar_t * predicted_noise) / sqrt_alpha_bar_t

            # Clip predicted x0 to [-1, 1]
            pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)

            # Direction pointing to x_t
            dir_xt = np.sqrt(1.0 - alpha_bar_next - eta**2 * (1.0 - alpha_bar_next) / (1.0 - alpha_bar_t) * (1.0 - alpha_bar_t / alpha_bar_next)) * predicted_noise

            # Random noise
            noise = torch.randn_like(x) if eta > 0 else 0

            # DDIM update
            x = np.sqrt(alpha_bar_next) * pred_x0 + dir_xt + eta * np.sqrt((1.0 - alpha_bar_next) / (1.0 - alpha_bar_t)) * np.sqrt(1.0 - alpha_bar_t / alpha_bar_next) * noise

        return x


def tensor_to_pil(images: torch.Tensor) -> list[Image.Image]:
    """
    Convert tensor images to PIL Images

    Args:
        images: [batch, 3, 64, 64] in range [-1, 1]

    Returns:
        List of PIL Images
    """
    # Denormalize from [-1, 1] to [0, 255]
    images = ((images + 1.0) * 127.5).clamp(0, 255).to(torch.uint8)

    # Convert to numpy and transpose to HWC
    images = images.cpu().numpy().transpose(0, 2, 3, 1)

    # Convert to PIL
    pil_images = [Image.fromarray(img) for img in images]

    return pil_images


def save_image_grid(images: torch.Tensor, path: str, nrow: int = 8):
    """
    Save images as a grid

    Args:
        images: [batch, 3, 64, 64] in range [-1, 1]
        path: Output path
        nrow: Number of images per row
    """
    from torchvision.utils import make_grid

    # Denormalize from [-1, 1] to [0, 1]
    images = (images + 1.0) / 2.0

    # Create grid
    grid = make_grid(images, nrow=nrow, padding=2, normalize=False)

    # Convert to PIL and save
    grid = (grid * 255).clamp(0, 255).to(torch.uint8)
    grid = grid.cpu().numpy().transpose(1, 2, 0)
    Image.fromarray(grid).save(path)


@torch.no_grad()
def generate_samples(
    model,
    tokenizer,
    prompts: list[str],
    device: str = "cuda",
    num_samples: int = 1,
    use_ddim: bool = True,
    ddim_steps: int = 50,
) -> torch.Tensor:
    """
    High-level function to generate images from text prompts

    Args:
        model: Trained UNet model
        tokenizer: Tokenizer for encoding prompts
        prompts: List of text prompts
        device: Device to run on
        num_samples: Number of images per prompt
        use_ddim: Use DDIM (faster) vs DDPM (slower but higher quality)
        ddim_steps: Number of DDIM steps (only if use_ddim=True)

    Returns:
        Generated images [batch * num_samples, 3, 64, 64]
    """
    from dataloader import MAX_SEQ_LEN

    model.eval()

    # Tokenize prompts
    token_ids = []
    for prompt in prompts:
        encoding = tokenizer.encode(prompt)
        ids = encoding.ids

        # Pad/truncate
        if len(ids) > MAX_SEQ_LEN:
            ids = ids[:MAX_SEQ_LEN]
        while len(ids) < MAX_SEQ_LEN:
            pad_id = tokenizer.token_to_id("[PAD]")
            ids.append(pad_id)

        token_ids.append(ids)

    text_tokens = torch.tensor(token_ids, dtype=torch.long, device=device)

    # Create sampler
    noise_schedule = NoiseSchedule()
    sampler = DDPMSampler(model, noise_schedule, device=device)

    # Generate
    if use_ddim:
        images = sampler.sample_ddim(text_tokens, num_samples=num_samples, num_steps=ddim_steps)
    else:
        images = sampler.sample(text_tokens, num_samples=num_samples)

    return images
