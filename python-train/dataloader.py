import json
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional, Tuple, List
from PIL import Image
import numpy as np
from tokenizers import Tokenizer

# Constants
IMAGE_SIZE = 64
IMAGE_CHANNELS = 3
MAX_SEQ_LEN = 77  # CLIP standard
NUM_TIMESTEPS = 1000


class NoiseSchedule:
    """Linear noise schedule for diffusion process"""

    def __init__(self, num_timesteps: int = NUM_TIMESTEPS, beta_start: float = 0.0001, beta_end: float = 0.02):
        self.num_timesteps = num_timesteps

        # Linear schedule
        self.betas = np.linspace(beta_start, beta_end, num_timesteps, dtype=np.float32)
        self.alphas = 1.0 - self.betas

        # Cumulative product of alphas
        self.alpha_bars = np.cumprod(self.alphas)

        self.sqrt_alpha_bars = np.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = np.sqrt(1.0 - self.alpha_bars)

    def get_noise_params(self, timestep: int) -> Tuple[float, float]:
        """Get noise parameters for a given timestep"""
        return self.sqrt_alpha_bars[timestep], self.sqrt_one_minus_alpha_bars[timestep]


class DiffusionDataset(Dataset):
    """Dataset for DiffusionDB images and prompts"""

    def __init__(
        self,
        json_dir: str,
        image_dir: str,
        tokenizer_path: str,
        max_samples: Optional[int] = None,
    ):
        """
        Args:
            json_dir: Directory containing JSON metadata files
            image_dir: Directory containing images
            tokenizer_path: Path to tokenizer.json file
            max_samples: Maximum number of samples to load (None = load all)
        """
        self.image_dir = Path(image_dir)
        self.items = []

        # Load tokenizer
        print(f"Loading tokenizer from {tokenizer_path}...")
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.pad_token_id = self.tokenizer.token_to_id("[PAD]")
        print(f"Tokenizer loaded. Vocabulary size: {self.tokenizer.get_vocab_size()}")

        # Noise schedule for training
        self.noise_schedule = NoiseSchedule()

        # Load JSON files
        json_path = Path(json_dir)
        json_files = sorted(json_path.glob("*.json"))

        # Calculate how many files we need based on max_samples
        APPROX_ITEMS_PER_JSON = 1000
        if max_samples is not None:
            files_needed = min(
                (max_samples + APPROX_ITEMS_PER_JSON - 1) // APPROX_ITEMS_PER_JSON,
                len(json_files)
            )
            json_files = json_files[:files_needed]
            print(f"Loading up to {max_samples} samples from {files_needed} JSON files...")
        else:
            print(f"Loading all samples from {len(json_files)} JSON files...")

        # Parse JSON files
        for json_file in json_files:
            with open(json_file, 'r') as f:
                data = json.load(f)

            for image_filename, metadata in data.items():
                if max_samples is not None and len(self.items) >= max_samples:
                    break

                image_path = self.image_dir / image_filename

                # Only add if image exists
                if image_path.exists():
                    self.items.append({
                        'image_path': image_path,
                        'prompt': metadata['p'],
                        'seed': metadata['se'],
                        'cfg_scale': metadata['c'],
                        'steps': metadata['st'],
                        'sampler': metadata['sa'],
                    })

            if max_samples is not None and len(self.items) >= max_samples:
                break

        print(f"Loaded {len(self.items)} items")

    def __len__(self) -> int:
        return len(self.items)

    def tokenize(self, text: str) -> Tuple[List[int], List[bool]]:
        """Tokenize text and return token IDs and attention mask"""
        encoding = self.tokenizer.encode(text)
        ids = encoding.ids
        mask = [True] * len(ids)

        # Truncate if too long
        if len(ids) > MAX_SEQ_LEN:
            ids = ids[:MAX_SEQ_LEN]
            mask = mask[:MAX_SEQ_LEN]

        # Pad if too short
        while len(ids) < MAX_SEQ_LEN:
            ids.append(self.pad_token_id)
            mask.append(False)

        return ids, mask

    def load_image(self, path: Path) -> np.ndarray:
        """Load and preprocess image to CHW format normalized to [-1, 1]"""
        img = Image.open(path).convert('RGB')

        if img.size != (IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError(f"Image dimensions {img.size} don't match expected {IMAGE_SIZE}x{IMAGE_SIZE}")

        # Convert to numpy array and normalize to [-1, 1]
        img_array = np.array(img, dtype=np.float32)
        img_array = (img_array / 127.5) - 1.0

        # Convert from HWC to CHW
        img_array = img_array.transpose(2, 0, 1)

        return img_array

    def __getitem__(self, idx: int) -> dict:
        """Get a single item from the dataset"""
        item = self.items[idx]

        # Load image
        try:
            image = self.load_image(item['image_path'])
        except Exception as e:
            print(f"Failed to load image {item['image_path']}: {e}")
            image = np.zeros((IMAGE_CHANNELS, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)

        # Tokenize prompt
        try:
            tokens, mask = self.tokenize(item['prompt'])
        except Exception as e:
            print(f"Failed to tokenize prompt: {e}")
            tokens = [0] * MAX_SEQ_LEN
            mask = [False] * MAX_SEQ_LEN

        # Sample random timestep
        timestep = np.random.randint(0, NUM_TIMESTEPS)

        # Generate noise
        noise = np.random.randn(IMAGE_CHANNELS, IMAGE_SIZE, IMAGE_SIZE).astype(np.float32)

        # Apply noise schedule
        sqrt_alpha_bar, sqrt_one_minus_alpha_bar = self.noise_schedule.get_noise_params(timestep)
        noisy_image = sqrt_alpha_bar * image + sqrt_one_minus_alpha_bar * noise

        return {
            'image': torch.from_numpy(image),
            'noisy_image': torch.from_numpy(noisy_image),
            'noise': torch.from_numpy(noise),
            'timestep': torch.tensor(timestep, dtype=torch.long),
            'text_tokens': torch.tensor(tokens, dtype=torch.long),
            'text_mask': torch.tensor(mask, dtype=torch.float32),
            'prompt': item['prompt'],  # Keep for debugging
        }


def collate_fn(batch: List[dict]) -> dict:
    """Collate function for DataLoader"""
    return {
        'images': torch.stack([item['image'] for item in batch]),
        'noisy_images': torch.stack([item['noisy_image'] for item in batch]),
        'noise': torch.stack([item['noise'] for item in batch]),
        'timesteps': torch.stack([item['timestep'] for item in batch]),
        'text_tokens': torch.stack([item['text_tokens'] for item in batch]),
        'text_mask': torch.stack([item['text_mask'] for item in batch]),
        'prompts': [item['prompt'] for item in batch],
    }


def create_dataloader(
    json_dir: str,
    image_dir: str,
    tokenizer_path: str,
    batch_size: int = 32,
    max_samples: Optional[int] = None,
    num_workers: int = 4,
    shuffle: bool = True,
) -> DataLoader:
    """Create a DataLoader for the DiffusionDB dataset"""
    dataset = DiffusionDataset(json_dir, image_dir, tokenizer_path, max_samples)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
