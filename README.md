# Mini-Pic

Mini-Pic trains a 64x64 text-conditioned image model on DiffusionDB. The data
preparation bins use `D:\DiffusionDB` by default. The 2M archives extracted by
`download-diffusiondb.ps1` place PNG images and `part-*.json` metadata files
directly in `D:\DiffusionDB\images`.

From the `mini-pic` directory, run:

```powershell
cargo run --release --bin train_tokenizer
cargo run --release --bin preprocess
cargo run --release --bin augment
```

The tokenizer is written to `mini-pic\tokenizer.json`. Preprocessing center-crops
and resizes source PNGs to `D:\DiffusionDB\images-64`. Augmentation writes the
resized original plus 11 variants per image, with matching JSON metadata, to
`D:\DiffusionDB\images-64-augmented`. For 10 archives this is up to 120,000
output PNGs, so check available disk space before running augmentation.

To use a different folder on D:, set `$env:MINI_PIC_DATA_ROOT = 'D:\your-folder'`
before running the commands. That folder must contain an `images` directory
with extracted PNG and `part-*.json` files. The Rust and Python training
defaults use the same augmented output folder.


`cargo run --release --bin mini-pic` to train

The native trainer defaults to 1,000 augmented images for a smoke run
(`total_samples = 1_000`). Set `total_samples = 0` in `src/training.rs` to use
all images. It keeps augmentations of one source image in the same train or
validation split. Every native U-Net preset has four downsampling stages
(64 -> 32 -> 16 -> 8 -> 4), matching upsampling stages, and eight text encoder
layers. Select a preset with `$env:MINI_PIC_UNET_PRESET = 'balanced'` before
training; `balanced` is the default.

To resume an interrupted native run, use the same preset and settings (the run
folder name is derived from them) and set `$env:MINI_PIC_RESUME = 'latest'` (or
an epoch number). Model, optimizer and scheduler state are restored from
`checkpoint/`. Without it, a run deletes any existing folder of the same name,
checkpoints included. `config.json` is now written before training starts.

| Preset | Channel widths | ResNet blocks per down/up level |
| --- | --- | ---: |
| `compact` | `[8, 16, 32, 32, 32]` | 2 |
| `balanced` (default) | `[16, 32, 64, 64, 64]` | 2 |
| `wide` | `[32, 64, 128, 128, 128]` | 2 |
| `extra-wide` | `[64, 128, 256, 256, 256]` | 2 |

Each preset gets a distinct checkpoint directory from its widths and ResNet
count. Earlier three-level checkpoints are incompatible; training quality needs
a fresh run to verify.

`cargo run --release --bin infer -- "<prompt>" <model_dir> [steps] [output_path]` to generate an
image. `model_dir` is the "Artifact dir: ..." path training printed at the start of that run
(steps and output_path are both optional - steps defaults to 50, output_path to `output.png`):

`cargo run --release --bin infer -- "a red apple on a table" "<artifact-dir-from-training>" 50 "test-output.png"`

Inference uses the Python pipeline's deterministic DDIM update, including
clean-image clipping at each step. Compare generated images across several
epochs and prompts; the next training run is needed to verify quality.

`cargo run --release --bin inspect_noise -- [--count N] [--out DIR] [--seed U64] [--progression-index I] [--progression-timesteps t1,t2,...]`
dumps exactly what the real `DiffusionBatcher` (dataset.rs) feeds the model, for
verifying training inputs independently of model output quality:
`<out>/batch/` holds `N` real dataset items as clean/noisy/noise PNGs plus a
per-item `log.json` entry (sampled timestep, noise schedule coefficients, tensor
min/max/mean, a tokenizer decode round-trip of the prompt, and a
reconstruction-error check that recovers the clean image from the noisy image
and the known noise - isolates whether the forward-noise math itself is
correct); `<out>/progression/` noises one fixed image at a fixed ascending
timestep list with one frozen noise sample, saved both as individual frames and
as a horizontal `strip.png`, so the noise level actually increasing with
timestep is visible directly.
