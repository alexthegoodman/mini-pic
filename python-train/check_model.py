"""Test script to verify model components are working correctly"""
import torch
import numpy as np
from model import UNet, TextEncoder, CrossAttention, AttentionBlock
from dataloader import DiffusionDataset
from pathlib import Path

def check_text_encoder():
    """Verify text encoder produces different embeddings for different prompts"""
    print("=" * 80)
    print("CHECKING TEXT ENCODER")
    print("=" * 80)

    # Load tokenizer
    tokenizer_path = "../tokenizer.json"
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()

    # Create text encoder
    text_encoder = TextEncoder(vocab_size=vocab_size, text_embed_dim=128, num_layers=4)
    text_encoder.eval()

    # Test prompts
    prompts = [
        "a red car on a highway",
        "a blue car on a highway",
        "a cat sitting on a table",
        "a dog sitting on a table",
    ]

    # Tokenize
    MAX_SEQ_LEN = 77
    token_ids = []
    for prompt in prompts:
        encoding = tokenizer.encode(prompt)
        ids = encoding.ids
        if len(ids) > MAX_SEQ_LEN:
            ids = ids[:MAX_SEQ_LEN]
        while len(ids) < MAX_SEQ_LEN:
            pad_id = tokenizer.token_to_id("[PAD]")
            ids.append(pad_id)
        token_ids.append(ids)

    text_tokens = torch.tensor(token_ids, dtype=torch.long)

    print(f"\nToken IDs shape: {text_tokens.shape}")
    print(f"First 10 tokens for each prompt:")
    for i, prompt in enumerate(prompts):
        print(f"  {i+1}. '{prompt}': {text_tokens[i, :10].tolist()}")

    # Forward pass
    with torch.no_grad():
        embeddings = text_encoder(text_tokens)

    print(f"\nText embeddings shape: {embeddings.shape}")
    print(f"Embeddings dtype: {embeddings.dtype}")
    print(f"Embeddings range: [{embeddings.min().item():.4f}, {embeddings.max().item():.4f}]")
    print(f"Embeddings mean: {embeddings.mean().item():.4f}")
    print(f"Embeddings std: {embeddings.std().item():.4f}")

    # Check if embeddings are different for different prompts
    print("\n--- Checking embedding differences ---")
    for i in range(len(prompts)):
        for j in range(i+1, len(prompts)):
            # Compute L2 distance between embeddings
            diff = torch.norm(embeddings[i] - embeddings[j]).item()
            print(f"  Distance between prompt {i+1} and {j+1}: {diff:.4f}")

    # Check if same semantic prompts are more similar
    print("\n--- Semantic similarity check ---")
    red_blue_dist = torch.norm(embeddings[0] - embeddings[1]).item()
    red_cat_dist = torch.norm(embeddings[0] - embeddings[2]).item()
    cat_dog_dist = torch.norm(embeddings[2] - embeddings[3]).item()

    print(f"  'red car' vs 'blue car': {red_blue_dist:.4f}")
    print(f"  'red car' vs 'cat on table': {red_cat_dist:.4f}")
    print(f"  'cat on table' vs 'dog on table': {cat_dog_dist:.4f}")

    if red_blue_dist < red_cat_dist:
        print("  [OK] Similar prompts (red/blue car) are closer than dissimilar ones")
    else:
        print("  [WARNING] Similar prompts should be closer!")

    # Check for NaN or Inf
    if torch.isnan(embeddings).any():
        print("\n  [ERROR] NaN values detected in embeddings!")
    elif torch.isinf(embeddings).any():
        print("\n  [ERROR] Inf values detected in embeddings!")
    else:
        print("\n  [OK] No NaN or Inf values")

    print()
    return embeddings


def check_cross_attention():
    """Verify cross-attention is actually using text context"""
    print("=" * 80)
    print("CHECKING CROSS ATTENTION")
    print("=" * 80)

    batch_size = 2
    channels = 64
    height, width = 16, 16
    text_embed_dim = 128
    seq_len = 77

    # Create cross attention module
    attn_block = AttentionBlock(channels=channels, context_dim=text_embed_dim, n_heads=4)
    attn_block.eval()

    # Create dummy inputs
    x = torch.randn(batch_size, channels, height, width)
    context1 = torch.randn(batch_size, seq_len, text_embed_dim)
    context2 = torch.randn(batch_size, seq_len, text_embed_dim)

    print(f"\nInput x shape: {x.shape}")
    print(f"Context shape: {context1.shape}")

    # Forward pass with different contexts
    with torch.no_grad():
        out1 = attn_block(x, context1)
        out2 = attn_block(x, context2)
        out_same = attn_block(x, context1)  # Should be identical to out1

    print(f"\nOutput shape: {out1.shape}")

    # Check if different contexts produce different outputs
    diff_contexts = torch.norm(out1 - out2).item()
    diff_same = torch.norm(out1 - out_same).item()

    print(f"\nDifference with different contexts: {diff_contexts:.6f}")
    print(f"Difference with same context: {diff_same:.6f}")

    if diff_contexts > 0.1:
        print("  [OK] Different text contexts produce different outputs")
    else:
        print("  [WARNING] Different contexts produce very similar outputs!")

    if diff_same < 1e-5:
        print("  [OK] Same context produces identical output (deterministic)")
    else:
        print("  [WARNING] Same context should produce identical output!")

    # Check output statistics
    print(f"\nOutput range: [{out1.min().item():.4f}, {out1.max().item():.4f}]")
    print(f"Output mean: {out1.mean().item():.4f}")
    print(f"Output std: {out1.std().item():.4f}")

    # Check for NaN or Inf
    if torch.isnan(out1).any():
        print("  [ERROR] NaN values detected in output!")
    elif torch.isinf(out1).any():
        print("  [ERROR] Inf values detected in output!")
    else:
        print("  [OK] No NaN or Inf values")

    print()


def check_full_model():
    """Verify full UNet model with different text prompts"""
    print("=" * 80)
    print("CHECKING FULL UNET MODEL")
    print("=" * 80)

    # Load tokenizer
    tokenizer_path = "../tokenizer.json"
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()

    # Create model
    model = UNet(
        vocab_size=vocab_size,
        text_embed_dim=128,
        time_embed_dim=32,
        channels=[32, 64, 128],
        use_mid_attn=True,
        text_encoder_layers=4,
    )
    model.eval()

    # Test prompts
    prompts = [
        "a beautiful sunset over the ocean",
        "a cute puppy playing in grass",
    ]

    # Tokenize
    MAX_SEQ_LEN = 77
    token_ids = []
    for prompt in prompts:
        encoding = tokenizer.encode(prompt)
        ids = encoding.ids
        if len(ids) > MAX_SEQ_LEN:
            ids = ids[:MAX_SEQ_LEN]
        while len(ids) < MAX_SEQ_LEN:
            pad_id = tokenizer.token_to_id("[PAD]")
            ids.append(pad_id)
        token_ids.append(ids)

    text_tokens = torch.tensor(token_ids, dtype=torch.long)

    # Create noisy images and timesteps
    batch_size = len(prompts)
    noisy_images = torch.randn(batch_size, 3, 64, 64)
    timesteps = torch.tensor([500, 500], dtype=torch.long)

    print(f"\nInput shapes:")
    print(f"  noisy_images: {noisy_images.shape}")
    print(f"  timesteps: {timesteps.shape}")
    print(f"  text_tokens: {text_tokens.shape}")

    # Forward pass
    with torch.no_grad():
        noise_pred1 = model(noisy_images, timesteps, text_tokens)

        # Try with swapped prompts
        text_tokens_swapped = text_tokens[[1, 0]]
        noise_pred2 = model(noisy_images, timesteps, text_tokens_swapped)

        # Try with same prompt
        noise_pred_same = model(noisy_images, timesteps, text_tokens)

    print(f"\nPredicted noise shape: {noise_pred1.shape}")
    print(f"Noise range: [{noise_pred1.min().item():.4f}, {noise_pred1.max().item():.4f}]")
    print(f"Noise mean: {noise_pred1.mean().item():.4f}")
    print(f"Noise std: {noise_pred1.std().item():.4f}")

    # Check if different prompts produce different noise predictions
    print("\n--- Checking text conditioning effect ---")
    diff_per_sample = []
    for i in range(batch_size):
        diff = torch.norm(noise_pred1[i] - noise_pred2[i]).item()
        diff_per_sample.append(diff)
        print(f"  Sample {i+1}: Difference when prompt swapped: {diff:.4f}")

    avg_diff = np.mean(diff_per_sample)
    print(f"  Average difference: {avg_diff:.4f}")

    if avg_diff > 0.1:
        print("  [OK] Different text prompts produce different noise predictions")
    else:
        print("  [WARNING] Text prompts don't seem to affect predictions much!")

    # Check determinism
    diff_same = torch.norm(noise_pred1 - noise_pred_same).item()
    print(f"\n  Difference with same inputs: {diff_same:.6f}")
    if diff_same < 1e-5:
        print("  [OK] Model is deterministic")
    else:
        print("  [WARNING] Model should be deterministic in eval mode!")

    # Check for NaN or Inf
    if torch.isnan(noise_pred1).any():
        print("\n  [ERROR] NaN values detected in predictions!")
    elif torch.isinf(noise_pred1).any():
        print("\n  [ERROR] Inf values detected in predictions!")
    else:
        print("  [OK] No NaN or Inf values")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")

    print()


def check_with_real_data():
    """Check model with real dataset samples"""
    print("=" * 80)
    print("CHECKING WITH REAL DATASET")
    print("=" * 80)

    # Load dataset
    json_dir = "../../diffusiondb/unzipped-json-augmented/"
    image_dir = "../../diffusiondb/unzipped-64-augmented/"
    tokenizer_path = "../tokenizer.json"

    print("\nLoading dataset...")
    dataset = DiffusionDataset(json_dir, image_dir, tokenizer_path, max_samples=100)

    # Get a few samples
    samples = [dataset[i] for i in range(3)]

    print(f"\nLoaded {len(samples)} samples")
    for i, sample in enumerate(samples):
        print(f"\nSample {i+1}:")
        print(f"  Prompt: {sample['prompt'][:80]}...")
        print(f"  Image shape: {sample['image'].shape}")
        print(f"  Noisy image shape: {sample['noisy_image'].shape}")
        print(f"  Text tokens shape: {sample['text_tokens'].shape}")
        print(f"  Text mask shape: {sample['text_mask'].shape}")
        print(f"  Timestep: {sample['timestep'].item()}")

        # Check how many non-padding tokens
        num_real_tokens = sample['text_mask'].sum().item()
        num_padding = len(sample['text_mask']) - num_real_tokens
        print(f"  Real tokens: {num_real_tokens}, Padding: {num_padding}")

    # Create model
    vocab_size = dataset.tokenizer.get_vocab_size()
    model = UNet(
        vocab_size=vocab_size,
        text_embed_dim=128,
        time_embed_dim=32,
        channels=[32, 64, 128],
        use_mid_attn=True,
        text_encoder_layers=4,
    )
    model.eval()

    # Test forward pass on real data
    print("\n--- Testing forward pass on real data ---")
    batch = {
        'noisy_images': torch.stack([s['noisy_image'] for s in samples]),
        'timesteps': torch.stack([s['timestep'] for s in samples]),
        'text_tokens': torch.stack([s['text_tokens'] for s in samples]),
        'noise': torch.stack([s['noise'] for s in samples]),
    }

    with torch.no_grad():
        loss, predicted_noise = model.compute_loss(batch)

    print(f"\nLoss: {loss.item():.4f}")
    print(f"Predicted noise shape: {predicted_noise.shape}")
    print(f"Predicted noise range: [{predicted_noise.min().item():.4f}, {predicted_noise.max().item():.4f}]")

    # Check if predictions are different for different samples
    print("\n--- Checking per-sample differences ---")
    for i in range(len(samples)):
        for j in range(i+1, len(samples)):
            diff = torch.norm(predicted_noise[i] - predicted_noise[j]).item()
            print(f"  Sample {i+1} vs {j+1}: {diff:.4f}")

    print()


def main():
    """Run all checks"""
    print("\n" + "=" * 80)
    print("MODEL VERIFICATION TESTS")
    print("=" * 80 + "\n")

    try:
        check_text_encoder()
    except Exception as e:
        print(f"[ERROR] Text encoder check failed: {e}\n")

    try:
        check_cross_attention()
    except Exception as e:
        print(f"[ERROR] Cross attention check failed: {e}\n")

    try:
        check_full_model()
    except Exception as e:
        print(f"[ERROR] Full model check failed: {e}\n")

    try:
        check_with_real_data()
    except Exception as e:
        print(f"[ERROR] Real data check failed: {e}\n")

    print("=" * 80)
    print("ALL CHECKS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
