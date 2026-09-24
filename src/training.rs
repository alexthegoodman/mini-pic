use crate::dataset::{DiffusionBatcher, DiffusionDataset, DiffusionItem};
use crate::data_paths;
use crate::model::{UNet, UNetConfig};
use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
use burn::{
    data::{dataloader::DataLoaderBuilder, dataset::Dataset},
    module::AutodiffModule,
    prelude::*,
    record::{CompactRecorder, NoStdTrainingRecorder, Recorder},
    tensor::backend::AutodiffBackend,
};
use rand::{rngs::StdRng, seq::SliceRandom, SeedableRng};
use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};

#[derive(Config)]
pub struct TrainingConfig {
    // Training hyperparameters
    #[config(default = 50)]
    pub num_epochs: usize,

    // #[config(default = 16)]
    // #[config(default = 8)]
    #[config(default = 4)]
    pub batch_size: usize,

    #[config(default = 4)]
    pub num_workers: usize,

    #[config(default = 1337)]
    pub seed: u64,

    /// 0 loads every image found (a real training run); any other value caps
    /// the dataset for a quick smoke test. (burn's Config derive only
    /// accepts literal defaults, so this can't be an Option<usize> with a
    /// None default - see total_samples_limit() for the Option conversion.)
    /// This is the one place that decides which mode a run is in - run()
    /// used to hardcode Some(100) deep inside the function regardless of any
    /// config field, so changing modes meant editing code, not config.
    // #[config(default = 0)]
    // #[config(default = 40_000)]
    #[config(default = 1000)]
    pub total_samples: usize,

    // Learning rate schedule
    #[config(default = 1e-4)] // seems moderate
    // #[config(default = 1e-5)] // seems quite slow at the start
    // #[config(default = 1e-3)]
    pub learning_rate: f64,

    /// Every this many training batches (counted across epochs) the current
    /// model is saved to `<run dir>/model` and the `infer` binary is run on
    /// it, writing `<run dir>/samples/step-NNNNNN.png`. 0 disables.
    #[config(default = 500)]
    pub sample_every_batches: usize,

    /// Sampler steps passed to `infer` for those preview images.
    #[config(default = 20)]
    pub sample_steps: usize,

    // Optimizer
    pub optimizer: AdamWConfig,

    // Model config
    pub model: UNetConfig,

    // Dataset paths
    pub json_dir: String,
    pub image_dir: String,
    pub tokenizer_path: String,
}

impl TrainingConfig {
    /// total_samples as the Option<usize> DiffusionDataset::new expects: 0 -> None (load everything).
    pub fn total_samples_limit(&self) -> Option<usize> {
        (self.total_samples != 0).then_some(self.total_samples)
    }
}

/// All presets keep the five feature levels needed for a 4x4 bottleneck.
/// Only channel width varies; ResNet count, text, and time settings stay fixed
/// so runs can be compared more directly.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum UNetPreset {
    Compact,
    Balanced,
    Wide,
    ExtraWide,
    Test,
}

impl UNetPreset {
    fn parse(value: &str) -> Option<Self> {
        match value {
            "compact" => Some(Self::Compact),
            "balanced" => Some(Self::Balanced),
            "wide" => Some(Self::Wide),
            "extra-wide" => Some(Self::ExtraWide),
            "test" => Some(Self::ExtraWide),
            _ => None,
        }
    }

    fn from_env() -> Self {
        match std::env::var("MINI_PIC_UNET_PRESET") {
            Ok(value) => Self::parse(&value).unwrap_or_else(|| {
                panic!("unknown MINI_PIC_UNET_PRESET '{value}'; choose compact, balanced, wide, or extra-wide")
            }),
            Err(std::env::VarError::NotPresent) => Self::Test,
            Err(std::env::VarError::NotUnicode(_)) => {
                panic!("MINI_PIC_UNET_PRESET must be valid Unicode")
            }
        }
    }

    fn name(self) -> &'static str {
        match self {
             Self::Test => "test",
            Self::Compact => "compact",
            Self::Balanced => "balanced",
            Self::Wide => "wide",
            Self::ExtraWide => "extra-wide",
        }
    }

    fn model_config(self) -> UNetConfig {
        let channels = match self {
            // Self::Test => vec![8, 32, 64, 256, 512],
            // Self::Test => vec![256, 128, 64, 32, 16], // slower than turtle
            // Self::Test => vec![128, 64, 32, 16, 8],
            // Self::Test => vec![32, 32, 32, 32, 32], // save_file crash? overflow?
            Self::Test => vec![64, 32, 16, 8, 8],
            Self::Compact => vec![8, 16, 32, 32, 32],
            Self::Balanced => vec![16, 32, 64, 64, 64],
            Self::Wide => vec![32, 64, 128, 128, 128],
            Self::ExtraWide => vec![64, 128, 256, 256, 256],
        };

        UNetConfig::new(channels)
            .with_vocab_size(4096) // Replaced with the loaded tokenizer's size.
            .with_text_embed_dim(64)
            .with_text_encoder_layers(8)
            .with_time_embed_dim(128)
            .with_use_mid_attn(true)
            .with_resnet_blocks_per_level(2)
    }
}

const AUGMENTATION_SUFFIXES: [&str; 12] = [
    "_flip", "_vflip", "_rot90", "_rot180", "_rot270", "_blur",
    "_strongblur", "_bright", "_dark", "_gray", "_hcontrast", "_lcontrast",
];

fn source_image_id(path: &Path) -> String {
    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or_default();
    // These suffixes match the names emitted by augment.rs.
    for suffix in AUGMENTATION_SUFFIXES {
        if let Some(source) = stem.strip_suffix(suffix) {
            return source.to_owned();
        }
    }
    stem.to_owned()
}

fn split_by_source(
    items: Vec<DiffusionItem>,
    train_ratio: f32,
    seed: u64,
) -> (Vec<DiffusionItem>, Vec<DiffusionItem>) {
    let mut groups: BTreeMap<String, Vec<DiffusionItem>> = BTreeMap::new();
    for item in items {
        groups.entry(source_image_id(&item.image_path)).or_default().push(item);
    }
    let mut groups: Vec<_> = groups.into_values().collect();
    groups.shuffle(&mut StdRng::seed_from_u64(seed));
    let target = (groups.iter().map(Vec::len).sum::<usize>() as f32 * train_ratio) as usize;
    let group_count = groups.len();
    let mut train = Vec::new();
    let mut valid = Vec::new();
    for (index, group) in groups.into_iter().enumerate() {
        if train.len() < target && index + 1 < group_count {
            train.extend(group);
        } else {
            valid.extend(group);
        }
    }
    (train, valid)
}

/// Resolves MINI_PIC_RESUME ("latest" or an epoch number) to the epoch of a
/// checkpoint that exists in `artifact_dir/checkpoint`, or None if unset.
/// Panics rather than silently starting fresh: a fresh start wipes the folder.
fn resume_epoch(artifact_dir: &str) -> Option<usize> {
    let want = std::env::var("MINI_PIC_RESUME").ok()?;
    let checkpoint_dir = std::path::Path::new(artifact_dir).join("checkpoint");
    // Epochs that have both files (model, optim). Runs made by the old
    // LearnerBuilder also wrote a scheduler-N.mpk, which is constant and unused.
    let mut complete: Vec<usize> = std::fs::read_dir(&checkpoint_dir)
        .unwrap_or_else(|e| panic!("MINI_PIC_RESUME set but {} is unreadable: {e}", checkpoint_dir.display()))
        .filter_map(|entry| {
            let name = entry.ok()?.file_name().into_string().ok()?;
            let epoch: usize = name.strip_prefix("model-")?.strip_suffix(".mpk")?.parse().ok()?;
            checkpoint_dir.join(format!("optim-{epoch}.mpk")).exists().then_some(epoch)
        })
        .collect();
    complete.sort_unstable();

    let epoch = if want == "latest" {
        *complete.last().unwrap_or_else(|| panic!("no complete checkpoint in {}", checkpoint_dir.display()))
    } else {
        let epoch: usize = want.parse().expect("MINI_PIC_RESUME must be 'latest' or an epoch number");
        assert!(complete.contains(&epoch), "no complete checkpoint for epoch {epoch}; have {complete:?}");
        epoch
    };
    Some(epoch)
}

/// Saves the model as `<run dir>/model.bin` (the file `infer` loads) and runs
/// `cargo run --release --bin infer -- <run dir> <steps> <png>`, blocking until
/// it finishes so it never competes with training for the GPU. A failure is
/// reported and training carries on.
fn generate_samples<B: Backend>(model: &UNet<B>, artifact_dir: &str, step: usize, sample_steps: usize) {
    if let Err(e) = model
        .clone()
        .save_file(format!("{artifact_dir}/model"), &NoStdTrainingRecorder::new())
    {
        eprintln!("[step {step}] could not save model for sampling: {e}");
        return;
    }

    let samples_dir = Path::new(artifact_dir).join("samples");
    std::fs::create_dir_all(&samples_dir).ok();
    let out_path = samples_dir.join(format!("step-{step:06}.png"));

    println!("[step {step}] sampling -> {}", out_path.display());
    // Runs in the current working directory, where infer finds tokenizer.json.
    let status = std::process::Command::new("cargo")
        .args(["run", "--release", "--bin", "infer", "--"])
        .arg(artifact_dir)
        .arg(sample_steps.to_string())
        .arg(&out_path)
        .status();
    match status {
        Ok(s) if s.success() => {}
        Ok(s) => eprintln!("[step {step}] infer exited with {s}; continuing training"),
        Err(e) => eprintln!("[step {step}] could not launch infer: {e}; continuing training"),
    }
}

/// Writes `model-N.mpk` and `optim-N.mpk` (the same names and format the old
/// LearnerBuilder used) and deletes the pair from two epochs earlier.
fn save_checkpoint<B, O>(checkpoint_dir: &Path, epoch: usize, model: &UNet<B>, optim: &O)
where
    B: AutodiffBackend,
    O: Optimizer<UNet<B>, B>,
{
    std::fs::create_dir_all(checkpoint_dir).ok();
    let recorder = CompactRecorder::new();
    recorder
        .record(model.clone().into_record(), checkpoint_dir.join(format!("model-{epoch}")))
        .expect("Failed to save model checkpoint");
    recorder
        .record(optim.to_record(), checkpoint_dir.join(format!("optim-{epoch}")))
        .expect("Failed to save optimizer checkpoint");
    if let Some(old) = epoch.checked_sub(2) {
        for kind in ["model", "optim", "scheduler"] {
            std::fs::remove_file(checkpoint_dir.join(format!("{kind}-{old}.mpk"))).ok();
        }
    }
}

fn create_artifact_dir(artifact_dir: &str) {
    // Remove existing artifacts to get an accurate learner summary
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).ok();
}

/// Encodes the hyperparameters that actually change the model/run shape into
/// a directory name, so different sweeps land in different folders under
/// models_root instead of silently overwriting each other's checkpoint.
fn artifact_dir_name(config: &TrainingConfig) -> String {
    let channels = config
        .model
        .channels
        .iter()
        .map(|c| c.to_string())
        .collect::<Vec<_>>()
        .join("-");

    format!(
        "mini-pic_ch{channels}_res{res}_te-emb{temb}_ti-emd{tiemb}_tl{tl}_th{th}_ep{ep}_bs{bs}_lr{lr:.0e}_size{count}",
        channels = channels,
        res = config.model.resnet_blocks_per_level,
        temb = config.model.text_embed_dim,
        tiemb = config.model.time_embed_dim,
        tl = config.model.text_encoder_layers,
        th = config.model.text_encoder_heads,
        ep = config.num_epochs,
        bs = config.batch_size,
        lr = config.learning_rate,
        count = config.total_samples,
    )
}

pub fn run<B: AutodiffBackend>(models_root: &str, device: B::Device) {
    // Config - using recommended diffusion hyperparameters
    let optimizer = AdamWConfig::new()
        .with_weight_decay(1e-2)
        .with_beta_1(0.9)
        .with_beta_2(0.999)
        .with_epsilon(1e-8);

    let unet_preset = UNetPreset::from_env();
    let model_config = unet_preset.model_config();

    let mut config = TrainingConfig::new(
        optimizer,
        model_config,
        data_paths::augmented_dir().to_string_lossy().into_owned(),
        data_paths::augmented_dir().to_string_lossy().into_owned(),
        "tokenizer.json".to_string(),
    );
    // Keep the 1,000-image default for the requested smoke run.
    B::seed(config.seed);

    let artifact_dir = format!("{models_root}/{}", artifact_dir_name(&config));
    // Resuming must not go through create_artifact_dir, which deletes the
    // folder (checkpoints included).
    let resume_from = resume_epoch(&artifact_dir);
    match resume_from {
        Some(epoch) => println!("Resuming {artifact_dir} from the epoch {epoch} checkpoint"),
        None => create_artifact_dir(&artifact_dir),
    }

    println!("=== Diffusion Model Training Configuration ===");
    println!("Artifact dir: {}", artifact_dir);
    println!("Batch size: {}", config.batch_size);
    println!("Learning rate: {}", config.learning_rate);
    println!("Learning rate schedule: constant");
    println!("Epochs: {}", config.num_epochs);
    println!("Weight decay: 1e-2");
    println!("U-Net preset: {}", unet_preset.name());
    println!("Model channels: {:?}", config.model.channels);
    println!("ResNet blocks per level: {}", config.model.resnet_blocks_per_level);
    println!("Text embed dim: {}", config.model.text_embed_dim);
    println!("==============================================\n");

    // Determine dataset size - specify number of image samples you want
    // Each JSON file contains ~1000 images, so the loader will automatically
    // determine how many JSON files to load
    println!("Determining dataset split...");

    // config.total_samples is the single switch between a quick smoke test
    // and full training - see TrainingConfig::total_samples's doc comment.
    let total_samples = config.total_samples_limit();
    let train_ratio = 0.8;

    // Load the full dataset, then split it
    println!("Loading dataset ({} samples)...",
             total_samples.map(|x| x.to_string()).unwrap_or_else(|| "all".to_string()));

    let full_dataset = DiffusionDataset::new(
        &config.json_dir,
        &config.image_dir,
        &config.tokenizer_path,
        total_samples,
    )
    .expect("Failed to load dataset");

    let total_loaded = full_dataset.len();
    let (train_items, valid_items) = split_by_source(full_dataset.items, train_ratio, config.seed);
    let train_size = train_items.len();
    let valid_size = valid_items.len();
    assert!(train_size > 0 && valid_size > 0, "training needs at least two source images");
    println!("Splitting {} samples by source image: {} train, {} valid", total_loaded, train_size, valid_size);

    for item in &train_items[0..20] {
        println!("training prompt {:?}", item.metadata.prompt);
    }

    let train_dataset = DiffusionDataset {
        items: train_items,
        tokenizer: full_dataset.tokenizer.clone(),
    };

    let valid_dataset = DiffusionDataset {
        items: valid_items,
        tokenizer: full_dataset.tokenizer.clone(),
    };

    println!("Train size: {}", train_dataset.len());
    println!("Valid size: {}\n", valid_dataset.len());

    // Update model vocab size from the tokenizer, in config itself (not a
    // separate shadowed local) - config.save() below must persist the exact
    // shape the model was actually built with, or a later load (see
    // inference.rs) reconstructs the wrong architecture.
    let vocab_size = train_dataset.tokenizer.vocab_size();
    config.model = config.model.clone().with_vocab_size(vocab_size);

    println!("Tokenizer vocab size: {}", vocab_size);

    // Create model
    let model: crate::model::UNet<B> = config.model.init(&device);
    println!("Model initialized\n");

    // Clone tokenizers before moving datasets
    let train_tokenizer = train_dataset.tokenizer.clone();
    let valid_tokenizer = valid_dataset.tokenizer.clone();

    // Create batchers
    let batcher_train = DiffusionBatcher::<B>::new(
        device.clone(),
        train_tokenizer.clone(),
    );

    let batcher_valid = DiffusionBatcher::<B::InnerBackend>::new(
        device.clone(),
        valid_tokenizer.clone(),
    );

    // Create dataloaders
    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(train_dataset);

    let dataloader_valid = DataLoaderBuilder::new(batcher_valid)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(valid_dataset);

    // At 1,000 images, a 1,000-step warmup would consume roughly 20 epochs.
    // Use the configured learning rate directly for this smoke run.
    let batches_per_epoch = train_size.div_ceil(config.batch_size);
    let total_steps = batches_per_epoch * config.num_epochs;
    let lr = config.learning_rate;

    println!("Learning rate: constant at {}", lr);
    println!("Total training steps: {}\n", total_steps);

    // Saved before training so a crash mid-run still leaves the architecture
    // behind next to the checkpoints.
    config
        .save(format!("{artifact_dir}/config.json").as_str())
        .expect("Failed to save config");

    // Custom loop (LearnerBuilder has no hook for "every N batches").
    let mut model = model;
    let mut optim = config.optimizer.init::<B, UNet<B>>();
    let checkpoint_dir = PathBuf::from(&artifact_dir).join("checkpoint");
    let recorder = CompactRecorder::new();

    let mut start_epoch = 1;
    if let Some(epoch) = resume_from {
        let model_record = recorder
            .load(checkpoint_dir.join(format!("model-{epoch}")), &device)
            .expect("Failed to load model checkpoint");
        model = model.load_record(model_record);
        let optim_record = recorder
            .load(checkpoint_dir.join(format!("optim-{epoch}")), &device)
            .expect("Failed to load optimizer checkpoint");
        optim = optim.load_record(optim_record);
        start_epoch = epoch + 1;
    }

    // Batches are counted across epochs, so "every 500" lands wherever it
    // lands in an epoch; a resumed run continues the count.
    let mut global_step = (start_epoch - 1) * batches_per_epoch;
    let mut loss_log = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(Path::new(&artifact_dir).join("losses.csv"))
        .expect("Failed to open losses.csv");
    if global_step == 0 {
        writeln!(loss_log, "kind,epoch,step,loss").ok();
    }

    println!("Starting training at epoch {start_epoch}...\n");
    for epoch in start_epoch..=config.num_epochs {
        let mut epoch_loss = 0.0f64;
        let mut epoch_batches = 0usize;

        for batch in dataloader_train.iter() {
            let output = model.forward_step(batch);
            let loss: f32 = output.loss.clone().into_data().to_vec::<f32>().unwrap()[0];
            let grads = GradientsParams::from_grads(output.loss.backward(), &model);
            model = optim.step(lr, model, grads);

            global_step += 1;
            epoch_batches += 1;
            epoch_loss += loss as f64;
            writeln!(loss_log, "train,{epoch},{global_step},{loss}").ok();

            if global_step % 50 == 0 {
                println!(
                    "epoch {epoch}/{} batch {epoch_batches}/{batches_per_epoch} step {global_step} loss {loss:.5} (epoch avg {:.5})",
                    config.num_epochs,
                    epoch_loss / epoch_batches as f64,
                );
            }

            if config.sample_every_batches != 0 && global_step % config.sample_every_batches == 0 {
                generate_samples(&model, &artifact_dir, global_step, config.sample_steps);
            }
        }

        // Validation on the inner (no-autodiff) backend.
        let model_valid = model.valid();
        let mut valid_loss = 0.0f64;
        let mut valid_batches = 0usize;
        for batch in dataloader_valid.iter() {
            let loss: f32 = model_valid.forward_step(batch).loss.into_data().to_vec::<f32>().unwrap()[0];
            valid_loss += loss as f64;
            valid_batches += 1;
        }
        let train_avg = epoch_loss / epoch_batches.max(1) as f64;
        let valid_avg = valid_loss / valid_batches.max(1) as f64;
        println!("=== epoch {epoch} done: train loss {train_avg:.5}, valid loss {valid_avg:.5} ===");
        writeln!(loss_log, "train_epoch_avg,{epoch},{global_step},{train_avg}").ok();
        writeln!(loss_log, "valid_epoch_avg,{epoch},{global_step},{valid_avg}").ok();
        loss_log.flush().ok();

        save_checkpoint::<B, _>(&checkpoint_dir, epoch, &model, &optim);
    }

    println!("\nTraining complete!");
    println!("Saving model to {}", artifact_dir);

    // Save trained model
    model
        .save_file(
            format!("{artifact_dir}/model"),
            &NoStdTrainingRecorder::new(),
        )
        .expect("Failed to save trained model");

    println!("Model saved successfully!");
}

// ============================================================================
// Additional utility functions for training
// ============================================================================

/// Print training tips and recommendations
pub fn print_training_tips() {
    println!("\n=== Training Tips for Diffusion Models ===");
    println!("1. Start with batch_size=16, decrease if GPU memory is limited");
    println!("2. Learning rate 1e-4 is standard for diffusion models");
    println!("3. Use AdamW with weight_decay=1e-2");
    println!("4. This smoke run uses the selected learning rate directly");
    println!("5. Monitor loss - should decrease steadily");
    println!("6. Train for 100-500 epochs depending on dataset size");
    println!("7. Validation loss should track training loss");
    println!("8. If loss is NaN, reduce learning rate or batch size");
    println!("9. Generate samples every N epochs to check quality");
    println!("==========================================\n");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::DiffusionMetadata;

    #[test]
    fn unet_presets_vary_width_with_two_resnets_each() {
        for (name, channels) in [
            ("compact", vec![8, 16, 32, 32, 32]),
            ("balanced", vec![16, 32, 64, 64, 64]),
            ("wide", vec![32, 64, 128, 128, 128]),
            ("extra-wide", vec![64, 128, 256, 256, 256]),
        ] {
            let preset = UNetPreset::parse(name).unwrap();
            let config = preset.model_config();
            assert_eq!(preset.name(), name);
            assert_eq!(config.channels, channels);
            assert_eq!(config.resnet_blocks_per_level, 2);
            assert!(config.use_mid_attn);
        }
        assert_eq!(UNetPreset::parse("unknown"), None);
    }

    #[test]
    fn augmented_variants_stay_in_one_split() {
        let mut items = Vec::new();
        for source in ["a", "b", "c", "d", "e"] {
            for suffix in ["", "_flip", "_lcontrast"] {
                items.push(DiffusionItem {
                    image_path: format!("{source}{suffix}.png").into(),
                    metadata: DiffusionMetadata {
                        prompt: String::new(), seed: 0, cfg_scale: 0.0,
                        steps: 0, sampler: String::new(),
                    },
                });
            }
        }
        let (train, valid) = split_by_source(items, 0.8, 1337);
        assert!(!train.is_empty() && !valid.is_empty());
        for source in ["a", "b", "c", "d", "e"] {
            let in_train = train.iter().filter(|item| source_image_id(&item.image_path) == source).count();
            let in_valid = valid.iter().filter(|item| source_image_id(&item.image_path) == source).count();
            assert!((in_train == 3 && in_valid == 0) || (in_train == 0 && in_valid == 3));
        }
    }
}
