use crate::dataset::{DiffusionBatcher, DiffusionDataset, DiffusionBatch};
use crate::model::UNetConfig;
use burn::lr_scheduler::linear::LinearLrSchedulerConfig;
use burn::optim::AdamWConfig;
use burn::train::metric::{CudaMetric, LearningRateMetric};
use burn::train::RegressionOutput;
use burn::{
    data::{dataloader::DataLoaderBuilder, dataset::Dataset},
    prelude::*,
    record::{CompactRecorder},
    tensor::backend::AutodiffBackend,
    train::{metric::LossMetric, LearnerBuilder},
};
use burn::data::dataloader::batcher::Batcher;

#[derive(Config)]
pub struct TrainingConfig {
    // Training hyperparameters
    #[config(default = 200)]
    pub num_epochs: usize,

    // #[config(default = 8)]
    #[config(default = 1)]
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
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self::new(
            AdamWConfig::new()
                .with_weight_decay(1e-2) // Standard for diffusion models
                .with_beta_1(0.9)
                .with_beta_2(0.999)
                .with_epsilon(1e-8),
            // UNetConfig::new(vec![64, 128, 256]),
            UNetConfig::new(vec![16, 32, 64]),
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
    let model_config = UNetConfig::new(vec![16, 32, 64])
        .with_vocab_size(8192) // Will be updated after loading tokenizer
        .with_text_embed_dim(32);

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

    // Determine dataset size - specify number of image samples you want
    // Each JSON file contains ~1000 images, so the loader will automatically
    // determine how many JSON files to load
    println!("Determining dataset split...");

    // For testing: use 2000 images total (will load ~2 JSON files)
    // For full training: use None to load all images
    let total_samples = Some(100); // Change to None for full dataset
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

    // Update model vocab size from tokenizer
    let vocab_size = train_dataset.tokenizer.vocab_size();
    let model_config = config
        .model
        .clone()
        .with_vocab_size(vocab_size);

    println!("Tokenizer vocab size: {}", vocab_size);

    // Create model
    let model: crate::model::UNet<B> = model_config.init(&device);
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

    // TEST: Try to create a test batch for validation
    println!("Testing batcher with single batch...");

    // Create a separate test dataset (load just 100 samples)
    let test_dataset = DiffusionDataset::new(
        &config.json_dir,
        &config.image_dir,
        &config.tokenizer_path,
        Some(100),
    )
    .expect("Failed to load test dataset");

    // Create a separate batcher for testing
    let test_batcher = DiffusionBatcher::<B>::new(
        device.clone(),
        test_dataset.tokenizer.clone(),
    );

    let test_items: Vec<_> = (0..config.batch_size)
        .filter_map(|i| test_dataset.get(i))
        .collect();
    println!("Got {} test items", test_items.len());

    println!("Creating batch...");
    let test_batch: DiffusionBatch<B> = test_batcher.batch(test_items, &device);
    println!("Batch created successfully!");
    println!("Batch shapes - images: {:?}, noisy: {:?}",
             test_batch.images.dims(),
             test_batch.noisy_images.dims());


    // Build learner with explicit type annotations
    // The RegressionOutput needs to sync to the same backend for metrics to work
    let learner = LearnerBuilder::<B, RegressionOutput<B>, RegressionOutput<B::InnerBackend>, _, _, _>::new(artifact_dir)
        .metric_train(CudaMetric::new())
        .metric_valid(CudaMetric::new())
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .metric_train_numeric(LearningRateMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .devices(vec![device.clone()])
        .num_epochs(config.num_epochs)
        .summary();
        // .build(model, config.optimizer.init(), lr_scheduler);

    println!("Starting Build...\n");

    let learner = learner.build(model, config.optimizer.init(), lr_scheduler);

    // TODO: never reaches here. just appears to fail silently and exits
    println!("Starting training...\n");
    
    // println!("\nNow trying fit()...");
    // // Train the model
    // let model_trained = learner.fit(dataloader_train, dataloader_valid);

    // println!("\nTraining complete!");
    // println!("Saving model and config to {}", artifact_dir);

    // // Save config
    // config
    //     .save(format!("{artifact_dir}/config.json").as_str())
    //     .expect("Failed to save config");

    // // Save trained model
    // model_trained
    //     .save_file(
    //         format!("{artifact_dir}/model"),
    //         &NoStdTrainingRecorder::new(),
    //     )
    //     .expect("Failed to save trained model");

    // println!("Model saved successfully!");
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

    let model_config = UNetConfig::new(vec![128, 256, 512, 512]) // Deeper model
        .with_vocab_size(8192)
        .with_text_embed_dim(512);

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
