"""Debug script to trace dimensions through the model"""
import torch
from model import UNet
from dataloader import IMAGE_SIZE, IMAGE_CHANNELS, MAX_SEQ_LEN

def test_model_debug():
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

    # Create dummy batch
    batch_size = 2
    noisy_images = torch.randn(batch_size, IMAGE_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)
    timesteps = torch.randint(0, 1000, (batch_size,))
    text_tokens = torch.randint(0, 8192, (batch_size, MAX_SEQ_LEN))

    print(f"\nInput: {noisy_images.shape}")

    # Manually trace through
    model.eval()
    with torch.no_grad():
        # Text encoding
        text_emb = model.text_embedding(text_tokens)
        text_context = model.text_encoder(text_emb)
        print(f"Text context: {text_context.shape}")

        # Time embedding
        time_emb = model.time_embedding(timesteps)
        print(f"Time emb: {time_emb.shape}")

        # Initial conv
        h = model.conv_in(noisy_images)
        print(f"After conv_in: {h.shape}")

        # Down blocks
        h1, skip1 = model.down1(h, time_emb, text_context)
        print(f"After down1: h1={h1.shape}, skip1={skip1.shape}")

        h2, skip2 = model.down2(h1, time_emb, text_context)
        print(f"After down2: h2={h2.shape}, skip2={skip2.shape}")

        h3, skip3 = model.down3(h2, time_emb, text_context)
        print(f"After down3: h3={h3.shape}, skip3={skip3.shape}")

        # Bottleneck
        h = model.mid_block1(h3, time_emb)
        print(f"After mid_block1: {h.shape}")
        h = model.mid_block2(h, time_emb)
        print(f"After mid_block2: {h.shape}")

        # Up blocks
        print(f"\nUp1 expects: h={h.shape} + skip3={skip3.shape}")
        h = model.up1(h, skip3, time_emb, text_context)
        print(f"After up1: {h.shape}")

        print(f"\nUp2 expects: h={h.shape} + skip2={skip2.shape}")
        h = model.up2(h, skip2, time_emb, text_context)
        print(f"After up2: {h.shape}")

if __name__ == "__main__":
    test_model_debug()
