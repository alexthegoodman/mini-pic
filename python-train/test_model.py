"""Test script to verify model forward pass"""
import torch
from model import UNet
from dataloader import IMAGE_SIZE, IMAGE_CHANNELS, MAX_SEQ_LEN

def test_model():
    # Create model
    print("Creating model...")
    model = UNet(
        vocab_size=8192,
        text_embed_dim=32,
        time_embed_dim=32,
        use_mid_attn=False,
        resnet_blocks_per_level=1,
        channels=[16, 32, 64],
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Create dummy batch
    batch_size = 2
    noisy_images = torch.randn(batch_size, IMAGE_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)
    timesteps = torch.randint(0, 1000, (batch_size,))
    text_tokens = torch.randint(0, 8192, (batch_size, MAX_SEQ_LEN))
    noise = torch.randn(batch_size, IMAGE_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)

    # Test forward pass
    print("\nTesting forward pass...")
    model.eval()
    with torch.no_grad():
        predicted_noise = model(noisy_images, timesteps, text_tokens)

    print(f"Input shape: {noisy_images.shape}")
    print(f"Output shape: {predicted_noise.shape}")
    print(f"Expected shape: {noise.shape}")
    assert predicted_noise.shape == noise.shape, "Output shape mismatch!"

    # Test loss computation
    print("\nTesting loss computation...")
    batch = {
        'noisy_images': noisy_images,
        'timesteps': timesteps,
        'text_tokens': text_tokens,
        'noise': noise,
    }

    model.train()
    loss, pred = model.compute_loss(batch)
    print(f"Loss: {loss.item():.4f}")

    # Test backward pass
    print("\nTesting backward pass...")
    loss.backward()
    print("Backward pass successful!")

    print("\n✓ All tests passed!")

if __name__ == "__main__":
    test_model()
