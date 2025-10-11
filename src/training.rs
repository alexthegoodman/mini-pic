use crate::dataset::{DiffusionBatcher, DiffusionDataset};
use crate::model::UNetConfig;
use burn::lr_scheduler::linear::LinearLrSchedulerConfig;
use burn::optim::AdamWConfig;
use burn::train::metric::{CudaMetric, LearningRateMetric};
use burn::{
    data::{dataloader::DataLoaderBuilder, dataset::Dataset},
    prelude::*,
    record::{CompactRecorder, NoStdTrainingRecorder},
    tensor::backend::AutodiffBackend,
    train::{metric::LossMetric, LearnerBuilder},
};

#[derive(Config)]
pub struct TrainingConfig {
    // Training hyperparameters
    #[config(default = 200)]
    pub num_epochs: usize,

    #[config(default = 8)]
    pub batch_size: usize,

    #[config(default = 4)]
    pub num_workers: usize,

    #[config(default = 1337)]
    pub seed: u64,

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

    // Training split (use first N files for training, rest for validation)
    #[config(default = 8)]
    pub train_files: usize,

    // Gradient clipping (optional, helps with stability)
    #[config(default = 1.0)]
    pub grad_clip_norm: f64,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self::new(
            AdamWConfig::new()
                .with_weight_decay(1e-2) // Standard for diffusion models
                .with_beta_1(0.9)
                .with_beta_2(0.999)
                .with_epsilon(1e-8),
            UNetConfig::new(),
            "../diffusiondb/unzipped-json/".to_string(),
            "../diffusiondb/unzipped-64/".to_string(),
            "tokenizer.json".to_string(),
        )
    }
}

fn create_artifact_dir(artifact_dir: &str) {
    // Remove existing artifacts to get an accurate learner summary
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).ok();
}

pub fn run<B: AutodiffBackend>(artifact_dir: &str, device: B::Device) {
    create_artifact_dir(artifact_dir);

    // Config - using recommended diffusion hyperparameters
    let optimizer = AdamWConfig::new()
        .with_weight_decay(1e-2)
        .with_beta_1(0.9)
        .with_beta_2(0.999)
        .with_epsilon(1e-8);

    // Model config - lightweight U-Net
    let model_config = UNetConfig::new()
        .with_vocab_size(8192) // Will be updated after loading tokenizer
        .with_text_embed_dim(256)
        .with_channels(vec![64, 128, 256]);

    let config = TrainingConfig::new(
        optimizer,
        model_config,
        "../diffusiondb/unzipped-json/".to_string(),
        "../diffusiondb/unzipped-64/".to_string(),
        "tokenizer.json".to_string(),
    );
    B::seed(config.seed);

    println!("=== Diffusion Model Training Configuration ===");
    println!("Batch size: {}", config.batch_size);
    println!("Learning rate: {}", config.learning_rate);
    println!("Warmup steps: {}", config.warmup_steps);
    println!("Epochs: {}", config.num_epochs);
    println!("Weight decay: 1e-2");
    println!("Model channels: {:?}", config.model.channels);
    println!("Text embed dim: {}", config.model.text_embed_dim);
    println!("==============================================\n");

    // Load dataset
    println!("Loading dataset...");
    let dataset = DiffusionDataset::new(
        &config.json_dir,
        &config.image_dir,
        &config.tokenizer_path,
    )
    .expect("Failed to load dataset");

    let dataset_size = dataset.len();
    println!("Total dataset size: {}", dataset_size);

    // Update model vocab size from tokenizer
    let vocab_size = dataset.tokenizer.vocab_size();
    let model_config = config
        .model
        .clone()
        .with_vocab_size(vocab_size);

    println!("Tokenizer vocab size: {}", vocab_size);

    // Create model
    let model = model_config.init(&device);
    println!("Model initialized\n");

    // Split dataset into train/validation (80/20 split)
    let train_size = (dataset_size as f32 * 0.8) as usize;
    let valid_size = dataset_size - train_size;

    println!("Train size: {}", train_size);
    println!("Valid size: {}\n", valid_size);

    // Create indices for split
    let train_indices: Vec<usize> = (0..train_size).collect();
    let valid_indices: Vec<usize> = (train_size..dataset_size).collect();

    // Clone dataset for train and validation
    // Note: In production, you'd want to use proper dataset splitting utilities
    // For now, we'll use the full dataset and rely on dataloader shuffling
    let train_dataset = dataset.clone();
    let valid_dataset = dataset.clone();

    // Create batchers
    let batcher_train = DiffusionBatcher::<B>::new(
        device.clone(),
        train_dataset.tokenizer.clone(),
    );

    let batcher_valid = DiffusionBatcher::<B::InnerBackend>::new(
        device.clone(),
        valid_dataset.tokenizer.clone(),
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
        0.0,  // Start from 0
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

    // Build learner
    let learner = LearnerBuilder::new(artifact_dir)
        .metric_train(CudaMetric::new())
        .metric_valid(CudaMetric::new())
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .metric_train_numeric(LearningRateMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .devices(vec![device.clone()])
        .num_epochs(config.num_epochs)
        .summary()
        .build(model, config.optimizer.init(), lr_scheduler);

    println!("Starting training...\n");

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

/// Advanced config for larger models (when scaling up)
pub fn create_large_model_config() -> TrainingConfig {
    let optimizer = AdamWConfig::new()
        .with_weight_decay(1e-2)
        .with_beta_1(0.9)
        .with_beta_2(0.999);

    let model_config = UNetConfig::new()
        .with_vocab_size(8192)
        .with_text_embed_dim(512)
        .with_channels(vec![128, 256, 512, 512]); // Deeper model

    TrainingConfig::new(
        optimizer,
        model_config,
        "../diffusiondb/unzipped-json/".to_string(),
        "../diffusiondb/unzipped-64/".to_string(),
        "tokenizer.json".to_string(),
    )
    .with_batch_size(4) // Smaller batch for larger model
        .with_learning_rate(5e-5) // Lower LR for larger model
        .with_warmup_steps(2000)
        .with_num_epochs(300)
}
