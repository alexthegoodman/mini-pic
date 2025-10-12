# PyTorch Diffusion Model Training

Text-conditioned diffusion model for 64x64 image generation using DiffusionDB dataset.

## Project Structure

```
python-train/
├── dataloader.py       # Dataset and dataloader for DiffusionDB
├── model.py           # U-Net diffusion model architecture
├── inference.py       # DDPM/DDIM sampling for generation
├── train.py           # Training script with sample generation
├── generate.py        # Standalone generation script
├── test_model.py      # Model tests
└── README.md
```

## Quick Start

### 1. Test the Model

```bash
uv run python test_model.py
```

### 2. Start Training (with automatic sample generation!)

```bash
uv run python train.py
```

or to resume

`uv run python resume_train.py`

The training script will:

- Train for 50 epochs on 1000 samples (quick test)
- Save checkpoints every 5 epochs
- **Generate sample images every 5 epochs** → saved to `checkpoints/samples/`
- Show training progress with loss and learning rate

### 3. Generate Images from Trained Model

```bash
uv run python generate.py --checkpoint checkpoints/best_model.pt --prompts "a sunset over mountains" "cyberpunk city" --num_samples 4 --output my_generation.png
```

## Sample Generation During Training

The training script automatically generates sample images to track progress!

Every 5 epochs (configurable), it will:

1. Pick prompts from the dataset (or use custom prompts)
2. Generate 4 images per prompt using DDIM sampling (50 steps)
3. Save a grid image to `checkpoints/samples/epoch_XXX.png`

Example output:

```
Generating samples with prompts:
  1. doom eternal, game concept art, veins and worms, muscular, crustacean exoske...
  2. a beautiful photorealistic painting of cemetery urbex unfinished building bu...
  3. ...
  4. ...
DDIM Sampling: 100%|████████| 50/50 [00:05<00:00]
Saved samples to checkpoints/samples/epoch_004.png
```

Watch the `checkpoints/samples/` folder to see your model improve over time!

## Model Architecture

**U-Net** with text conditioning (~635K parameters):

- **Encoder**: 3 downsampling blocks (64×64 → 32×32 → 16×16)
- **Bottleneck**: ResNet blocks + optional attention
- **Decoder**: 3 upsampling blocks with skip connections
- **Text Conditioning**: Embedding layer + cross-attention
- **Time Embedding**: Sinusoidal position embeddings

## Training Configuration

```python
TrainingConfig(
    # Dataset
    max_samples=1000,              # Use None for full dataset
    train_ratio=0.8,

    # Training
    num_epochs=50,
    batch_size=16,
    learning_rate=1e-4,
    warmup_steps=500,

    # Sample generation
    generate_samples=True,
    sample_interval=5,             # Generate every 5 epochs
    num_samples_per_prompt=4,
    use_ddim=True,
    ddim_steps=50,                 # Fast sampling (50 steps vs 1000)
)
```

## Full Training

For production training, edit `train.py`:

```python
config = TrainingConfig(
    max_samples=None,              # Load all ~1.4M samples
    num_epochs=200,
    batch_size=32,                 # Increase if GPU memory allows
    sample_interval=10,            # Generate less frequently
)
```

## Checkpoints & Outputs

Training creates:

```
checkpoints/
├── config.json                    # Training configuration
├── best_model.pt                  # Best validation loss model
├── checkpoint_epoch_N.pt          # Regular checkpoints
└── samples/                       # Generated samples during training
    ├── epoch_004.png
    ├── epoch_009.png
    └── ...
```

## Inference Options

### DDIM Sampling (Recommended)

- **Fast**: 50 steps (vs 1000 for DDPM)
- **Quality**: Nearly identical to DDPM
- **Usage**: `use_ddim=True, ddim_steps=50`

### DDPM Sampling

- **Slow**: 1000 steps
- **Quality**: Slightly better (marginal)
- **Usage**: `use_ddim=False`

## Custom Prompts for Training Samples

```python
config = TrainingConfig(
    ...
    sample_prompts=[
        "a beautiful sunset over the ocean",
        "cyberpunk city with neon lights",
        "portrait of a cat wearing a crown",
        "abstract colorful geometric shapes",
    ]
)
```

## Example Training Output

```
Epoch 10/50: 100%|████████| 50/50 [00:07<00:00, 7.12it/s, loss=0.2058, lr=1.00e-04]
Validating: 100%|████████| 13/13 [00:06<00:00, 2.04it/s]

Epoch 10/50 - Train Loss: 0.2552, Val Loss: 0.2441, Time: 13.4s
Saved checkpoint to checkpoints\checkpoint_epoch_9.pt
Saved best model to checkpoints\best_model.pt

Generating samples with prompts:
  1. doom eternal, game concept art...
  2. a beautiful photorealistic painting...
DDIM Sampling: 100%|████████| 50/50 [00:05<00:00]
Saved samples to checkpoints/samples/epoch_009.png
```

## Resume Training

```python
trainer = Trainer(config)
trainer.load_checkpoint("checkpoints/checkpoint_epoch_20.pt")
trainer.train()
```

## Performance Tips

1. **Watch the samples folder** - Best way to see if training is working
2. **Good loss**: Should drop to ~0.2-0.3 and stay stable
3. **Batch size**: Increase to 32+ if you have GPU memory
4. **Sample generation**: Disable if training is too slow (`generate_samples=False`)
5. **DDIM steps**: 50 is fast, 100 is better quality

## Troubleshooting

**No samples generated?** Check `generate_samples=True` and `sample_interval`

**Samples look like noise?** Train longer - it takes ~10-20 epochs to see structure

**Out of memory during sampling?** Reduce `num_samples_per_prompt` or `ddim_steps`

**Slow training?** Disable sample generation or increase `sample_interval`
