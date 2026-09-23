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
`cargo run --release --bin infer -- "a red apple on a table" "D:/models/mini-pic_ch16-32-64_res8_temb64_tl4_th4_ep10_bs8_lr1e-4" "test-output.png"`
