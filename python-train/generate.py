"""Standalone script to generate images from trained model"""
import torch
import argparse
from pathlib import Path
from tokenizers import Tokenizer

from model import UNet
from inference import generate_samples, tensor_to_pil, save_image_grid


def load_model(checkpoint_path: str, device: str = "cuda") -> UNet:
    """Load trained model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Get model config from checkpoint dir
    config_path = Path(checkpoint_path).parent / "config.json"
    import json
    with open(config_path) as f:
        config = json.load(f)

    # Create model
    model = UNet(
        vocab_size=config['vocab_size'],
        text_embed_dim=config['text_embed_dim'],
        time_embed_dim=config['time_embed_dim'],
        use_mid_attn=config['use_mid_attn'],
        resnet_blocks_per_level=config['resnet_blocks_per_level'],
        channels=config['channels'],
    ).to(device)

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Loaded model from {checkpoint_path}")
    print(f"Epoch: {checkpoint['epoch']}, Best val loss: {checkpoint['best_val_loss']:.4f}")

    return model


def main():
    parser = argparse.ArgumentParser(description="Generate images from trained diffusion model")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt",
                       help="Path to model checkpoint")
    parser.add_argument("--tokenizer", type=str, default="../tokenizer.json",
                       help="Path to tokenizer")
    parser.add_argument("--prompts", type=str, nargs="+",
                       default=["a beautiful sunset over the ocean",
                               "a cyberpunk city at night",
                               "a cat wearing sunglasses",
                               "abstract colorful painting"],
                       help="Text prompts to generate")
    parser.add_argument("--num_samples", type=int, default=4,
                       help="Number of images per prompt")
    parser.add_argument("--output", type=str, default="generated.png",
                       help="Output path for generated images")
    parser.add_argument("--ddim_steps", type=int, default=50,
                       help="Number of DDIM steps (50 is good, 100 is better)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to use")

    args = parser.parse_args()

    # Load model
    print("Loading model...")
    model = load_model(args.checkpoint, args.device)

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = Tokenizer.from_file(args.tokenizer)

    # Generate images
    print(f"\nGenerating {args.num_samples} images for each prompt:")
    for i, prompt in enumerate(args.prompts):
        print(f"  {i+1}. {prompt}")

    images = generate_samples(
        model,
        tokenizer,
        args.prompts,
        device=args.device,
        num_samples=args.num_samples,
        use_ddim=True,
        ddim_steps=args.ddim_steps,
    )

    print(f"\nGenerated {images.shape[0]} images")

    # Save as grid
    save_image_grid(images, args.output, nrow=args.num_samples)
    print(f"Saved to {args.output}")

    # Optionally save individual images
    save_individual = input("\nSave individual images? (y/n): ").lower() == 'y'
    if save_individual:
        output_dir = Path(args.output).parent / "individual"
        output_dir.mkdir(exist_ok=True)

        pil_images = tensor_to_pil(images)
        for i, (prompt, img) in enumerate(zip(args.prompts * args.num_samples, pil_images)):
            prompt_slug = prompt[:30].replace(" ", "_")
            img.save(output_dir / f"{i:03d}_{prompt_slug}.png")

        print(f"Saved {len(pil_images)} individual images to {output_dir}")


if __name__ == "__main__":
    main()
