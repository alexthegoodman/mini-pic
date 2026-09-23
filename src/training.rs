use crate::dataset::{DiffusionBatcher, DiffusionDataset};
use crate::data_paths;
use crate::model::{UNet, UNetConfig};
use burn::lr_scheduler::linear::{LinearLrScheduler, LinearLrSchedulerConfig};
use burn::optim::{AdamW, AdamWConfig, Optimizer};
use burn::train::metric::{CudaMetric, LearningRateMetric};
use burn::train::RegressionOutput;
use burn::{
    data::{dataloader::DataLoaderBuilder, dataset::Dataset},
    prelude::*,
    record::{CompactRecorder, NoStdTrainingRecorder},
    tensor::backend::AutodiffBackend,
    train::{metric::LossMetric, LearnerBuilder},
    nn::{
        conv::{Conv2d, Conv2dConfig},
        loss::MseLoss,
        Embedding, EmbeddingConfig, Gelu, GroupNorm, GroupNormConfig, Linear, LinearConfig,
    },
};
use burn::data::dataloader::batcher::Batcher;
use burn::module::AutodiffModule;

#[derive(Config)]
pub struct TrainingConfig {
    // Training hyperparameters
    #[config(default = 50)]
    pub num_epochs: usize,

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
    // #[config(default = 10_000)]
    #[config(default = 5_000)]
    pub total_samples: usize,

    // Learning rate schedule
    #[config(default = 1e-4)]
    pub learning_rate: f64,

    #[config(default = 1000)]
    pub warmup_steps: usize,

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

// No Default impl here on purpose - it previously duplicated run()'s channel
// width (vec![16, 32, 64], with a second vec![64, 128, 256] commented out
// beside it) in a way nothing ever actually called. run()'s own
// TrainingConfig::new(...) below is the one active place that picks these
// numbers; keep it that way rather than adding a second copy back.

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

    // Model config - lightweight U-Net
    let model_config = UNetConfig::new(vec![16, 32, 64])
        .with_vocab_size(4096) // Will be updated after loading tokenizer
        .with_text_embed_dim(64);

    let mut config = TrainingConfig::new(
        optimizer,
        model_config,
        data_paths::augmented_dir().to_string_lossy().into_owned(),
        data_paths::augmented_dir().to_string_lossy().into_owned(),
        "tokenizer.json".to_string(),
    );
    // Quick-test size; pass 0 for a full production run over every image found.
    // .with_total_samples(2000); // do not set here, keep one source of truth for hyperparams in the config defaults
    B::seed(config.seed);

    let artifact_dir = format!("{models_root}/{}", artifact_dir_name(&config));
    create_artifact_dir(&artifact_dir);

    println!("=== Diffusion Model Training Configuration ===");
    println!("Artifact dir: {}", artifact_dir);
    println!("Batch size: {}", config.batch_size);
    println!("Learning rate: {}", config.learning_rate);
    println!("Warmup steps: {}", config.warmup_steps);
    println!("Epochs: {}", config.num_epochs);
    println!("Weight decay: 1e-2");
    println!("Model channels: {:?}", config.model.channels);
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
    let train_size = (total_loaded as f32 * train_ratio) as usize;
    let valid_size = total_loaded - train_size;

    println!("Splitting {} samples: {} train, {} valid", total_loaded, train_size, valid_size);

    // Split the dataset
    let train_items = full_dataset.items[..train_size].to_vec();
    let valid_items = full_dataset.items[train_size..].to_vec();

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

    // Learning rate scheduler
    // Option 1: Linear warmup + cosine decay (recommended for diffusion)
    let total_steps = (train_size / config.batch_size) * config.num_epochs;
    let lr_scheduler = LinearLrSchedulerConfig::new(
        0.0001,  // Start from 0
        config.learning_rate,
        config.warmup_steps,
    )
    .init()
    .expect("Couldn't ccreate learning rate");

    // Option 2: Cosine annealing (alternative)
    // let lr_scheduler = CosineAnnealingLrSchedulerConfig::new(
    //     config.learning_rate,
    //     config.num_epochs,
    // )
    // .with_min_lr(1e-6)
    // .init();

    println!("Learning rate scheduler: Linear warmup to {}", config.learning_rate);
    println!("Warmup steps: {}", config.warmup_steps);
    println!("Total training steps: {}\n", total_steps);

    // Build learner with explicit type annotations
    // The RegressionOutput needs to sync to the same backend for metrics to work
    let learner = LearnerBuilder::<B, RegressionOutput<B>, RegressionOutput<B::InnerBackend>, UNet<B>, _, LinearLrScheduler>::new(artifact_dir.as_str())
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
    println!("1. Start with batch_size=8, increase if GPU memory allows");
    println!("2. Learning rate 1e-4 is standard for diffusion models");
    println!("3. Use AdamW with weight_decay=1e-2");
    println!("4. Linear warmup helps with training stability");
    println!("5. Monitor loss - should decrease steadily");
    println!("6. Train for 100-500 epochs depending on dataset size");
    println!("7. Validation loss should track training loss");
    println!("8. If loss is NaN, reduce learning rate or batch size");
    println!("9. Generate samples every N epochs to check quality");
    println!("==========================================\n");
}

// create_large_model_config() (a fourth, never-called copy of the channel
// widths - vec![128, 256, 512, 512], four elements against UNetConfig::init's
// hard-coded 3 down/up levels, so channels[3] was silently ignored even if it
// had been used) was removed here. Scale run()'s own model_config up instead
// of reviving a second, independently-drifting config builder.
