#![recursion_limit = "256"] // wgpu/naga auto-trait (Sync) checks overflow at the default

//! Dumps exactly what the Rust/Burn training loop feeds the model, so noise
//! application and tokenization can be checked by eye/by number instead of
//! only inferred from a not-yet-converged model's output quality.
//!
//! Two things are written to `--out`:
//!
//! 1. `batch/` - `--count` real dataset items run through the *actual*
//!    `DiffusionBatcher` (same struct `training::run` builds its dataloader
//!    on top of - see dataset.rs), each saved as clean/noisy/noise PNGs plus
//!    a per-item entry in `log.json` with the sampled timestep, the noise
//!    schedule coefficients at that timestep, tensor min/max/mean for all
//!    three images, a round-tripped decode of the tokenized prompt, and a
//!    reconstruction check (recovering the clean image from noisy_image and
//!    the known noise via the inverse of the forward-diffusion formula -
//!    this isolates whether the noise math itself is correct, independent of
//!    the model).
//! 2. `progression/` - one fixed image noised at a fixed, ascending list of
//!    timesteps with one frozen noise sample, so the noise level actually
//!    increasing with timestep is something you can see directly rather than
//!    infer from randomly-sampled batch items.
//!
//! Usage: inspect_noise [--count N] [--out DIR] [--seed U64]
//!                       [--progression-index I] [--progression-timesteps t1,t2,...]
//!                       [--json-dir DIR] [--image-dir DIR] [--tokenizer PATH]

use burn::backend::wgpu::{Wgpu, WgpuDevice};
use burn::data::dataloader::{batcher::Batcher, Dataset};
use burn::tensor::{backend::Backend, ElementConversion, Tensor};
use mini_pic::data_paths;
use mini_pic::dataset::{DiffusionBatcher, DiffusionDataset, DiffusionItem, NoiseSchedule, NUM_TIMESTEPS};
use mini_pic::inference::DiffusionInference;
use serde::Serialize;
use std::path::{Path, PathBuf};

fn arg_value(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

#[derive(Serialize, Clone)]
struct TensorStats {
    min: f32,
    max: f32,
    mean: f32,
}

fn tensor_stats<B: Backend>(t: &Tensor<B, 4>) -> TensorStats {
    TensorStats {
        min: t.clone().min().into_scalar().elem(),
        max: t.clone().max().into_scalar().elem(),
        mean: t.clone().mean().into_scalar().elem(),
    }
}

#[derive(Serialize)]
struct ItemLog {
    index: usize,
    image_path: String,
    prompt: String,
    decoded_tokens: String,
    valid_token_count: usize,
    timestep: usize,
    sqrt_alpha_bar: f32,
    sqrt_one_minus_alpha_bar: f32,
    image: TensorStats,
    noisy_image: TensorStats,
    noise: TensorStats,
    // max|reconstructed_x0 - image| where reconstructed_x0 is solved back out
    // of noisy_image and the known noise. Should be ~1e-6 (fp32 rounding
    // only) - anything larger means the forward-noise formula the batcher
    // applies and the inverse this tool solves have drifted from each other.
    reconstruction_max_abs_error: f32,
    clean_png: String,
    noisy_png: String,
    noise_png: String,
}

#[derive(Serialize)]
struct ProgressionFrameLog {
    timestep: usize,
    sqrt_alpha_bar: f32,
    sqrt_one_minus_alpha_bar: f32,
    noisy: TensorStats,
    // Pearson correlation between the raw (pre-clip) noisy tensor and the
    // clean image, flattened. This is the real, unclipped signal-vs-noise
    // measure - save_image's display clamp to [-1,1] can make a frame look
    // like uniform static well before the underlying tensor actually is,
    // since three independent per-channel Gaussian noise draws saturate the
    // display range (and read as color speckle) long before the true signal
    // correlation drops to zero.
    signal_correlation: f32,
    // Fraction of pixels whose noisy value falls outside [-1, 1], i.e. the
    // fraction save_image's display clamp actually saturates to black/white.
    clipped_fraction: f32,
    png: String,
}

#[derive(Serialize)]
struct ProgressionLog {
    image_path: String,
    prompt: String,
    frames: Vec<ProgressionFrameLog>,
    strip_png: String,
    // The min/max every frame.png and strip.png were autoscaled against
    // (shared across all frames so they stay comparable to each other) -
    // read alongside each frame's own `noisy` stats to see how much of that
    // frame's real range the display actually spans.
    display_min: f32,
    display_max: f32,
}

fn pearson_correlation(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len() as f64;
    let mean_a = a.iter().map(|&x| x as f64).sum::<f64>() / n;
    let mean_b = b.iter().map(|&x| x as f64).sum::<f64>() / n;
    let mut cov = 0.0;
    let mut var_a = 0.0;
    let mut var_b = 0.0;
    for i in 0..a.len() {
        let da = a[i] as f64 - mean_a;
        let db = b[i] as f64 - mean_b;
        cov += da * db;
        var_a += da * da;
        var_b += db * db;
    }
    (cov / (var_a.sqrt() * var_b.sqrt())) as f32
}

fn clipped_fraction(values: &[f32]) -> f32 {
    values.iter().filter(|&&v| v < -1.0 || v > 1.0).count() as f32 / values.len() as f32
}

#[derive(Serialize)]
struct InspectionLog {
    json_dir: String,
    image_dir: String,
    tokenizer_path: String,
    seed: u64,
    sample_count: usize,
    items: Vec<ItemLog>,
    progression: Option<ProgressionLog>,
}

/// Concatenates same-shaped [1, C, H, W] tensors left-to-right into one
/// [1, C, H, W * len] strip.
fn tile_horizontal<B: Backend>(images: Vec<Tensor<B, 4>>) -> Tensor<B, 4> {
    Tensor::cat(images, 3)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let count: usize = arg_value(&args, "--count")
        .and_then(|v| v.parse().ok())
        .unwrap_or(8);
    let out_dir = arg_value(&args, "--out").unwrap_or_else(|| "noise-inspection".to_string());
    let seed: u64 = arg_value(&args, "--seed").and_then(|v| v.parse().ok()).unwrap_or(1337);
    let progression_index: usize = arg_value(&args, "--progression-index")
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let progression_timesteps: Vec<usize> = arg_value(&args, "--progression-timesteps")
        .map(|v| {
            v.split(',')
                .map(|s| s.trim().parse().expect("progression timestep must be a number"))
                .collect()
        })
        .unwrap_or_else(|| vec![0, 199, 399, 599, 799, 999]);
    for &t in &progression_timesteps {
        assert!(t < NUM_TIMESTEPS, "timestep {t} is out of range (0..{NUM_TIMESTEPS})");
    }

    let json_dir = arg_value(&args, "--json-dir")
        .unwrap_or_else(|| data_paths::augmented_dir().to_string_lossy().into_owned());
    let image_dir = arg_value(&args, "--image-dir")
        .unwrap_or_else(|| data_paths::augmented_dir().to_string_lossy().into_owned());
    let tokenizer_path = arg_value(&args, "--tokenizer").unwrap_or_else(|| "tokenizer.json".to_string());

    let batch_dir = Path::new(&out_dir).join("batch");
    let progression_dir = Path::new(&out_dir).join("progression");
    std::fs::remove_dir_all(&out_dir).ok();
    std::fs::create_dir_all(&batch_dir).expect("failed to create batch output dir");
    std::fs::create_dir_all(&progression_dir).expect("failed to create progression output dir");

    let device = WgpuDevice::default();
    // Same call training::run makes before building its dataloader - seeds
    // the backend's RNG so Tensor::random_like (the noise itself) is
    // reproducible across runs of this tool, matching how a real training
    // run seeds it. Per-item timestep sampling below still goes through
    // rand::thread_rng() inside DiffusionBatcher::batch, exactly as it does
    // in real training, so it is NOT reproduced run-to-run - that's a
    // property of the production code path, not a bug in this tool.
    <Wgpu as Backend>::seed(seed);

    println!("Loading dataset from {} / {} ...", json_dir, image_dir);
    let load_count = count.max(progression_index + 1);
    let dataset = DiffusionDataset::new(&json_dir, &image_dir, &tokenizer_path, Some(load_count))
        .expect("failed to load dataset");
    assert!(
        dataset.len() >= load_count,
        "dataset only has {} items, need at least {} for --count {} and --progression-index {}",
        dataset.len(), load_count, count, progression_index
    );

    let sample_count = count.min(dataset.len());
    let items: Vec<DiffusionItem> = dataset.items[..sample_count].to_vec();
    let tokenizer = dataset.tokenizer.clone();

    // The exact batcher training::run's dataloader is built on top of (see
    // DiffusionBatcher::batch in dataset.rs) - not a reimplementation of the
    // noise application, the real one.
    let batcher = DiffusionBatcher::<Wgpu>::new(device.clone(), tokenizer.clone());
    let batch = batcher.batch(items.clone(), &device);

    let noise_schedule = NoiseSchedule::linear(NUM_TIMESTEPS, 0.0001, 0.02);
    let timesteps_f32: Vec<f32> = batch
        .timesteps
        .clone()
        .into_data()
        .convert::<f32>()
        .to_vec()
        .expect("failed to read timesteps");

    let mut item_logs = Vec::with_capacity(sample_count);
    for i in 0..sample_count {
        let image_i = batch.images.clone().narrow(0, i, 1);
        let noisy_i = batch.noisy_images.clone().narrow(0, i, 1);
        let noise_i = batch.noise.clone().narrow(0, i, 1);

        let timestep = timesteps_f32[i].round() as usize;
        let (sqrt_alpha_bar, sqrt_one_minus_alpha_bar) = noise_schedule.get_noise_params(timestep);

        // Inverse of noisy = image * sqrt_alpha_bar + noise * sqrt_one_minus_alpha_bar.
        let reconstructed = (noisy_i.clone() - noise_i.clone() * (sqrt_one_minus_alpha_bar as f64))
            / (sqrt_alpha_bar as f64);
        let reconstruction_max_abs_error: f32 = (reconstructed - image_i.clone())
            .abs()
            .max()
            .into_scalar()
            .elem();

        let token_ids_i32: Vec<i32> = batch
            .text_tokens
            .clone()
            .narrow(0, i, 1)
            .into_data()
            .convert::<i32>()
            .to_vec()
            .expect("failed to read text_tokens");
        let mask_i: Vec<f32> = batch
            .text_mask
            .clone()
            .narrow(0, i, 1)
            .into_data()
            .convert::<f32>()
            .to_vec()
            .expect("failed to read text_mask");
        let valid_token_count = mask_i.iter().filter(|&&m| m > 0.5).count();
        let token_ids_u32: Vec<u32> = token_ids_i32.iter().map(|&x| x as u32).collect();
        let decoded_tokens = tokenizer
            .decode(&token_ids_u32[..valid_token_count.max(1).min(token_ids_u32.len())], true)
            .unwrap_or_else(|e| format!("<decode failed: {e}>"));

        let clean_png = batch_dir.join(format!("{i:03}_clean.png"));
        let noisy_png = batch_dir.join(format!("{i:03}_noisy.png"));
        let noise_png = batch_dir.join(format!("{i:03}_noise.png"));

        let noisy_stats = tensor_stats(&noisy_i);
        let noise_stats = tensor_stats(&noise_i);

        DiffusionInference::<Wgpu>::save_image(image_i.clone(), clean_png.to_str().unwrap())
            .expect("failed to save clean image");
        // noisy_image and noise are NOT bounded to [-1,1] like a clean image -
        // save_image's fixed (x+1)*127.5 clamp saturates a large, growing
        // fraction of pixels to solid black/white as soon as noise pushes a
        // value past that range, which makes even a moderately-noised frame
        // look like near-total static well before the real signal is gone
        // (see the progression frames below, where the true, unclamped
        // signal_correlation decays gradually while a fixed-range render
        // saturates almost immediately). Min-max autoscaling each image to
        // its own actual range (logged as noisy_image/noise below) keeps the
        // picture honest about how much structure is really left.
        save_autoscaled(noisy_i.clone(), noisy_stats.min, noisy_stats.max, &noisy_png)
            .expect("failed to save noisy image");
        save_autoscaled(noise_i.clone(), noise_stats.min, noise_stats.max, &noise_png)
            .expect("failed to save noise image");

        item_logs.push(ItemLog {
            index: i,
            image_path: items[i].image_path.to_string_lossy().into_owned(),
            prompt: items[i].metadata.prompt.clone(),
            decoded_tokens,
            valid_token_count,
            timestep,
            sqrt_alpha_bar,
            sqrt_one_minus_alpha_bar,
            image: tensor_stats(&image_i),
            noisy_image: noisy_stats,
            noise: noise_stats,
            reconstruction_max_abs_error,
            clean_png: clean_png.to_string_lossy().into_owned(),
            noisy_png: noisy_png.to_string_lossy().into_owned(),
            noise_png: noise_png.to_string_lossy().into_owned(),
        });

        println!(
            "[{i:>3}/{sample_count}] t={timestep:>4} recon_err={reconstruction_max_abs_error:.6} prompt={:?}",
            items[i].metadata.prompt
        );
        if reconstruction_max_abs_error > 1e-3 {
            println!("  WARNING: reconstruction error is higher than expected fp32 rounding noise");
        }
    }

    // --- Progression: one image, fixed ascending timesteps, frozen noise ---
    println!(
        "\nGenerating noise progression for item {progression_index} at timesteps {:?}",
        progression_timesteps
    );
    let progression_item = &items[progression_index];
    let progression_image_data: Vec<f32> = {
        // Reuse the batcher's own single-image loader so this exactly matches
        // what training feeds in, rather than a second image-loading path.
        let single = batcher.batch(vec![progression_item.clone()], &device);
        single
            .images
            .into_data()
            .convert::<f32>()
            .to_vec()
            .expect("failed to read progression image")
    };
    let progression_image = Tensor::<Wgpu, 1>::from_floats(progression_image_data.as_slice(), &device)
        .reshape([1, 3, mini_pic::dataset::IMAGE_SIZE, mini_pic::dataset::IMAGE_SIZE]);
    let frozen_noise = Tensor::<Wgpu, 4>::random_like(
        &progression_image,
        burn::tensor::Distribution::Normal(0.0, 1.0),
    );

    let clean_flat: Vec<f32> = progression_image
        .clone()
        .into_data()
        .convert::<f32>()
        .to_vec()
        .expect("failed to read clean progression image");

    // Pass 1: compute every frame and its raw stats before saving anything,
    // so all frames (and the strip) can be rendered against one shared
    // min/max range - see the comment on save_autoscaled for why a fixed
    // [-1,1] clamp is misleading here, and why per-frame autoscaling would
    // be equally misleading in the other direction (every frame, even
    // near-pure noise, would fill the full display range and look equally
    // "sharp").
    let mut frame_tensors = Vec::with_capacity(progression_timesteps.len());
    let mut frame_flats = Vec::with_capacity(progression_timesteps.len());
    let mut frame_meta = Vec::with_capacity(progression_timesteps.len());
    let mut global_min = f32::INFINITY;
    let mut global_max = f32::NEG_INFINITY;
    for &t in &progression_timesteps {
        let (sqrt_alpha_bar, sqrt_one_minus_alpha_bar) = noise_schedule.get_noise_params(t);
        let noisy = progression_image.clone() * (sqrt_alpha_bar as f64)
            + (frozen_noise.clone() * (sqrt_one_minus_alpha_bar as f64)) * 0.1;

        let noisy_flat: Vec<f32> = noisy
            .clone()
            .into_data()
            .convert::<f32>()
            .to_vec()
            .expect("failed to read noisy progression frame");
        for &v in &noisy_flat {
            global_min = global_min.min(v);
            global_max = global_max.max(v);
        }
        let signal_correlation = pearson_correlation(&clean_flat, &noisy_flat);
        let frame_clipped_fraction = clipped_fraction(&noisy_flat);
        let stats = tensor_stats(&noisy);

        println!(
            "  t={t:>4} sqrt_ab={sqrt_alpha_bar:.4} sqrt_1m={sqrt_one_minus_alpha_bar:.4} \
             signal_corr={signal_correlation:.4} clipped_frac(raw [-1,1])={frame_clipped_fraction:.4}"
        );

        frame_meta.push((t, sqrt_alpha_bar, sqrt_one_minus_alpha_bar, stats, signal_correlation, frame_clipped_fraction));
        frame_tensors.push(noisy);
        frame_flats.push(noisy_flat);
    }

    // Pass 2: save every frame (and the strip) against the shared range.
    let mut frame_logs = Vec::with_capacity(progression_timesteps.len());
    for (i, &t) in progression_timesteps.iter().enumerate() {
        let (_, sqrt_alpha_bar, sqrt_one_minus_alpha_bar, ref stats, signal_correlation, frame_clipped_fraction) =
            frame_meta[i];
        let frame_path = progression_dir.join(format!("t{t:04}.png"));
        save_autoscaled(frame_tensors[i].clone(), global_min, global_max, &frame_path)
            .expect("failed to save progression frame");

        frame_logs.push(ProgressionFrameLog {
            timestep: t,
            sqrt_alpha_bar,
            sqrt_one_minus_alpha_bar,
            noisy: stats.clone(),
            signal_correlation,
            clipped_fraction: frame_clipped_fraction,
            png: frame_path.to_string_lossy().into_owned(),
        });
    }
    let strip = tile_horizontal(frame_tensors);
    let strip_path = progression_dir.join("strip.png");
    save_strip_autoscaled(strip, progression_timesteps.len(), global_min, global_max, &strip_path)
        .expect("failed to save progression strip");

    let log = InspectionLog {
        json_dir: json_dir.clone(),
        image_dir: image_dir.clone(),
        tokenizer_path: tokenizer_path.clone(),
        seed,
        sample_count,
        items: item_logs,
        progression: Some(ProgressionLog {
            image_path: progression_item.image_path.to_string_lossy().into_owned(),
            prompt: progression_item.metadata.prompt.clone(),
            frames: frame_logs,
            strip_png: strip_path.to_string_lossy().into_owned(),
            display_min: global_min,
            display_max: global_max,
        }),
    };

    let log_path = Path::new(&out_dir).join("log.json");
    std::fs::write(&log_path, serde_json::to_string_pretty(&log).unwrap())
        .expect("failed to write log.json");

    println!("\nWrote {sample_count} batch items and {} progression frames.", log.progression.as_ref().unwrap().frames.len());
    println!("Log: {}", log_path.display());
    println!("Batch images: {}", batch_dir.display());
    println!("Progression images: {}", progression_dir.display());
}

/// Saves a [1, C, H, W] tensor by linearly mapping [lo, hi] to [0, 255],
/// rather than assuming the data already lives in [-1, 1] the way a clean
/// image or a final denoised output does. noisy_image and noise both range
/// well past [-1, 1] (see their logged min/max), so mapping them through
/// that fixed assumption instead of their real range saturates a large
/// fraction of pixels to solid black/white long before the signal is
/// actually gone - see the comment where this is called in the batch loop.
fn save_autoscaled(image_tensor: Tensor<Wgpu, 4>, lo: f32, hi: f32, path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let [_batch, channels, height, width] = image_tensor.dims();
    let data: Vec<f32> = image_tensor
        .into_data()
        .convert::<f32>()
        .to_vec()
        .expect("failed to convert tensor to vec");
    write_autoscaled_rgb(&data, channels, height, width, lo, hi, path)
}

/// save_image/save_autoscaled are both fixed to a square IMAGE_SIZE x
/// IMAGE_SIZE frame, so the horizontally-tiled strip (IMAGE_SIZE tall,
/// IMAGE_SIZE * n wide) is saved directly here instead, against the same
/// shared [lo, hi] every individual frame in the strip was saved against.
fn save_strip_autoscaled(
    image_tensor: Tensor<Wgpu, 4>,
    frame_count: usize,
    lo: f32,
    hi: f32,
    path: &PathBuf,
) -> Result<(), Box<dyn std::error::Error>> {
    use mini_pic::dataset::{IMAGE_CHANNELS, IMAGE_SIZE};

    let width = IMAGE_SIZE * frame_count;
    let data: Vec<f32> = image_tensor
        .into_data()
        .convert::<f32>()
        .to_vec()
        .expect("failed to convert strip tensor to vec");
    write_autoscaled_rgb(&data, IMAGE_CHANNELS, IMAGE_SIZE, width, lo, hi, path)
}

fn write_autoscaled_rgb(
    data: &[f32],
    channels: usize,
    height: usize,
    width: usize,
    lo: f32,
    hi: f32,
    path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let range = (hi - lo).max(1e-6);
    let mut rgb_data = vec![0u8; height * width * 3];
    for c in 0..channels {
        for h in 0..height {
            for w in 0..width {
                let chw_idx = c * (height * width) + h * width + w;
                let hwc_idx = (h * width + w) * 3 + c;
                let scaled = ((data[chw_idx] - lo) / range * 255.0).clamp(0.0, 255.0);
                rgb_data[hwc_idx] = scaled as u8;
            }
        }
    }

    image::save_buffer(path, &rgb_data, width as u32, height as u32, image::ColorType::Rgb8)?;
    Ok(())
}
