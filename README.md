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
validation split. The active native run now uses the Python run's U-Net widths
`[64, 128, 256]`, batch size 16, time embedding 32, text embedding 64, and two
text encoder layers. It retains standard epsilon MSE and uses the configured
learning rate directly. A low MSE alone does not establish image quality.

`cargo run --release --bin infer -- "<prompt>" <model_dir> [steps] [output_path]` to generate an
image. `model_dir` is the "Artifact dir: ..." path training printed at the start of that run
(steps and output_path are both optional - steps defaults to 50, output_path to `output.png`):

`cargo run --release --bin infer -- "a red apple on a table" "<artifact-dir-from-training>" 50 "test-output.png"`

Inference uses the Python pipeline's deterministic DDIM update, including
clean-image clipping at each step. Compare generated images across several
epochs and prompts; the next training run is needed to verify quality.
