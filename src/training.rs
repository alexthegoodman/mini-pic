use crate::dataset::{DiffusionBatcher, DiffusionDataset, DiffusionItem};
use crate::data_paths;
use crate::model::{UNet, UNetConfig};
use burn::lr_scheduler::constant::ConstantLr;
use burn::optim::AdamWConfig;
use burn::train::metric::{CudaMetric, LearningRateMetric};
use burn::train::RegressionOutput;
use burn::{
    data::{dataloader::DataLoaderBuilder, dataset::Dataset},
    prelude::*,
    record::{CompactRecorder, NoStdTrainingRecorder},
    tensor::backend::AutodiffBackend,
    train::{metric::LossMetric, LearnerBuilder},
};
use rand::{rngs::StdRng, seq::SliceRandom, SeedableRng};
use std::collections::BTreeMap;
use std::path::Path;

#[derive(Config)]
pub struct TrainingConfig {
    // Training hyperparameters
    #[config(default = 50)]
    pub num_epochs: usize,

    // #[config(default = 16)]
    #[config(default = 8)]
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
    #[config(default = 16_000)]
    pub total_samples: usize,

    // Learning rate schedule
    #[config(default = 1e-4)]
    pub learning_rate: f64,

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
}

impl UNetPreset {
    fn parse(value: &str) -> Option<Self> {
        match value {
            "compact" => Some(Self::Compact),
            "balanced" => Some(Self::Balanced),
            "wide" => Some(Self::Wide),
            "extra-wide" => Some(Self::ExtraWide),
            _ => None,
        }
    }

    fn from_env() -> Self {
        match std::env::var("MINI_PIC_UNET_PRESET") {
            Ok(value) => Self::parse(&value).unwrap_or_else(|| {
                panic!("unknown MINI_PIC_UNET_PRESET '{value}'; choose compact, balanced, wide, or extra-wide")
            }),
            Err(std::env::VarError::NotPresent) => Self::Wide,
            Err(std::env::VarError::NotUnicode(_)) => {
                panic!("MINI_PIC_UNET_PRESET must be valid Unicode")
            }
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Compact => "compact",
            Self::Balanced => "balanced",
            Self::Wide => "wide",
            Self::ExtraWide => "extra-wide",
        }
    }

    fn model_config(self) -> UNetConfig {
        let channels = match self {
            Self::Compact => vec![8, 16, 32, 32, 32],
            Self::Balanced => vec![16, 32, 64, 64, 64],
            Self::Wide => vec![32, 64, 128, 128, 128],
            Self::ExtraWide => vec![64, 128, 256, 256, 256],
        };

        UNetConfig::new(channels)
            .with_vocab_size(4096) // Replaced with the loaded tokenizer's size.
            .with_text_embed_dim(64)
            .with_text_encoder_layers(8)
            .with_time_embed_dim(32)
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
        "mini-pic_ch{channels}_res{res}_temb{temb}_tl{tl}_th{th}_ep{ep}_bs{bs}_lr{lr:.0e}_size{count}",
        channels = channels,
        res = config.model.resnet_blocks_per_level,
        temb = config.model.text_embed_dim,
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
    create_artifact_dir(&artifact_dir);

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
    let total_steps = train_size.div_ceil(config.batch_size) * config.num_epochs;
    let lr_scheduler = ConstantLr::new(config.learning_rate);

    println!("Learning rate scheduler: constant at {}", config.learning_rate);
    println!("Total training steps: {}\n", total_steps);

    // Build learner with explicit type annotations
    // The RegressionOutput needs to sync to the same backend for metrics to work
    let learner = LearnerBuilder::<B, RegressionOutput<B>, RegressionOutput<B::InnerBackend>, UNet<B>, _, ConstantLr>::new(artifact_dir.as_str())
        .metric_train(CudaMetric::new())
        .metric_valid(CudaMetric::new())
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .metric_train_numeric(LearningRateMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .devices(vec![device.clone()])
        .num_epochs(config.num_epochs)
        .summary();

    println!("Starting Build...\n");

    let learner = learner.build(model, config.optimizer.init(), lr_scheduler);

    // OPEN BUG (unverified, needs a session with an actual CUDA device to
    // diagnose): this point was previously never reached - the process
    // exited with no panic message and no error. The two most likely
    // explanations, neither confirmed here: (1) the redundant dataset/batcher
    // self-test that used to sit just above this block (now removed - it
    // always reloaded 100 images regardless of config, ignoring
    // total_samples entirely) was stalling or erroring before training ever
    // started; (2) CudaDevice::default() aborting at the driver level on a
    // machine with no CUDA-capable GPU/driver can exit the process without a
    // Rust panic. Run with RUST_BACKTRACE=full and check the exit code if
    // this still reproduces.
    println!("Starting training...\n");
    
    println!("\nNow trying fit()...");
    // Train the model
    let model_trained = learner.fit(dataloader_train, dataloader_valid);

    println!("\nTraining complete!");
    println!("Saving model and config to {}", artifact_dir);

    // Save config
    config
        .save(format!("{artifact_dir}/config.json").as_str())
        .expect("Failed to save config");

    // Save trained model
    model_trained
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
