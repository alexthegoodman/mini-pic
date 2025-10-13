# Mini-Pic

Download parts of diffusiondb to `../diffusiondb`

1. Run `cargo run --bin train_tokenizer --release` to create tokenizer.json
2. Run `cargo run --bin preprocess --release` to prepare 64×64 images and `cargo run --bin augment --release` to create 11 variations per image
3. Start training! `cargo run --bin mini-pic --release`
