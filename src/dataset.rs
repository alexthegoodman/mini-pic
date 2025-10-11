use burn::{
    data::{dataloader::batcher::Batcher, dataset::Dataset},
    tensor::{backend::Backend, Int, Tensor},
};
use image::GenericImageView;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use tokenizers::Tokenizer;

// Constants
pub const IMAGE_SIZE: usize = 64;
pub const IMAGE_CHANNELS: usize = 3;
pub const MAX_SEQ_LEN: usize = 77; // CLIP standard
pub const NUM_TIMESTEPS: usize = 1000;

// ============================================================================
// Core Data Structures
// ============================================================================

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DiffusionMetadata {
    pub prompt: String,
    pub seed: i64,
    pub cfg_scale: f64,
    pub steps: i64,
    pub sampler: String,
}

#[derive(Clone, Debug)]
pub struct DiffusionItem {
    pub image_path: PathBuf,
    pub metadata: DiffusionMetadata,
}

// JSON structure from diffusiondb
#[derive(Debug, Deserialize)]
struct ImageMetadataJson {
    p: String,  // prompt
    se: i64,    // seed
    c: f64,     // cfg scale
    st: i64,    // steps
    sa: String, // sampler
}

// ============================================================================
// Text Tokenizer (using tokenizers crate)
// ============================================================================

#[derive(Clone)]
pub struct TextTokenizer {
    tokenizer: Tokenizer,
    pad_token_id: u32,
}

impl TextTokenizer {
    pub fn from_file(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let tokenizer = Tokenizer::from_file(path)
            .map_err(|e| format!("Failed to load tokenizer: {:?}", e))?;
        let pad_token_id = tokenizer
            .token_to_id("[PAD]")
            .ok_or("PAD token not found in tokenizer")?;

        Ok(Self {
            tokenizer,
            pad_token_id,
        })
    }

    pub fn encode(&self, text: &str, max_len: usize) -> Result<(Vec<u32>, Vec<bool>), Box<dyn std::error::Error>> {
        let encoding = self.tokenizer.encode(text, false)
            .map_err(|e| format!("Failed to encode text: {:?}", e))?;
        let mut ids = encoding.get_ids().to_vec();
        let mut mask = vec![true; ids.len()];

        // Truncate if too long
        if ids.len() > max_len {
            ids.truncate(max_len);
            mask.truncate(max_len);
        }

        // Pad if too short
        while ids.len() < max_len {
            ids.push(self.pad_token_id);
            mask.push(false);
        }

        Ok((ids, mask))
    }

    pub fn vocab_size(&self) -> usize {
        self.tokenizer.get_vocab_size(true)
    }
}

// ============================================================================
// Noise Schedule
// ============================================================================

#[derive(Clone, Debug)]
pub struct NoiseSchedule {
    pub betas: Vec<f32>,
    pub alphas: Vec<f32>,
    pub alpha_bars: Vec<f32>,
    pub sqrt_alpha_bars: Vec<f32>,
    pub sqrt_one_minus_alpha_bars: Vec<f32>,
}

impl NoiseSchedule {
    pub fn linear(num_timesteps: usize, beta_start: f32, beta_end: f32) -> Self {
        let mut betas = Vec::with_capacity(num_timesteps);
        let mut alphas = Vec::with_capacity(num_timesteps);
        let mut alpha_bars = Vec::with_capacity(num_timesteps);

        // Linear schedule
        for t in 0..num_timesteps {
            let beta = beta_start + (beta_end - beta_start) * (t as f32) / (num_timesteps as f32);
            betas.push(beta);
            alphas.push(1.0 - beta);
        }

        // Compute cumulative product of alphas
        let mut alpha_bar = 1.0;
        for alpha in &alphas {
            alpha_bar *= alpha;
            alpha_bars.push(alpha_bar);
        }

        let sqrt_alpha_bars: Vec<f32> = alpha_bars.iter().map(|x| x.sqrt()).collect();
        let sqrt_one_minus_alpha_bars: Vec<f32> =
            alpha_bars.iter().map(|x| (1.0 - x).sqrt()).collect();

        Self {
            betas,
            alphas,
            alpha_bars,
            sqrt_alpha_bars,
            sqrt_one_minus_alpha_bars,
        }
    }

    pub fn get_noise_params(&self, timestep: usize) -> (f32, f32) {
        let sqrt_alpha_bar = self.sqrt_alpha_bars[timestep];
        let sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bars[timestep];
        (sqrt_alpha_bar, sqrt_one_minus_alpha_bar)
    }
}

// ============================================================================
// Dataset
// ============================================================================

#[derive(Clone)]
pub struct DiffusionDataset {
    pub items: Vec<DiffusionItem>,
    pub tokenizer: TextTokenizer,
}

impl DiffusionDataset {
    /// Create a new DiffusionDataset
    ///
    /// # Arguments
    /// * `json_dir` - Directory containing JSON metadata files
    /// * `image_dir` - Directory containing images
    /// * `tokenizer_path` - Path to tokenizer file
    /// * `max_samples` - Maximum number of image samples to load (None = load all)
    ///
    /// Note: Each JSON file contains ~1000 images. The loader will determine
    /// how many JSON files to load based on max_samples.
    pub fn new(
        json_dir: &str,
        image_dir: &str,
        tokenizer_path: &str,
        max_samples: Option<usize>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let mut items = Vec::new();
        let json_path = Path::new(json_dir);

        // Load tokenizer from file
        println!("Loading tokenizer from {}...", tokenizer_path);
        let tokenizer = TextTokenizer::from_file(tokenizer_path)?;
        println!("Tokenizer loaded. Vocabulary size: {}", tokenizer.vocab_size());

        // Read all JSON files
        let mut json_files: Vec<_> = fs::read_dir(json_path)?
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.path())
            .filter(|path| {
                path.extension()
                    .and_then(|ext| ext.to_str())
                    .map(|ext| ext == "json")
                    .unwrap_or(false)
            })
            .collect();

        // Sort to ensure consistent ordering
        json_files.sort();

        // Calculate how many files we need based on max_samples
        // Assume ~1000 items per JSON file
        const APPROX_ITEMS_PER_JSON: usize = 1000;
        let files_needed = if let Some(max) = max_samples {
            ((max + APPROX_ITEMS_PER_JSON - 1) / APPROX_ITEMS_PER_JSON).min(json_files.len())
        } else {
            json_files.len()
        };

        let json_files_to_load = &json_files[..files_needed];

        if let Some(max) = max_samples {
            println!("Loading up to {} samples from {} JSON files...", max, files_needed);
        } else {
            println!("Loading all samples from {} JSON files...", files_needed);
        }

        for json_file in json_files_to_load {
            let content = fs::read_to_string(&json_file)?;
            let data: HashMap<String, ImageMetadataJson> = serde_json::from_str(&content)?;

            for (image_filename, metadata_json) in data {
                // Check if we've reached the max_samples limit
                if let Some(max) = max_samples {
                    if items.len() >= max {
                        break;
                    }
                }

                let image_path = Path::new(image_dir).join(&image_filename);

                // Only add if image exists
                if image_path.exists() {
                    items.push(DiffusionItem {
                        image_path,
                        metadata: DiffusionMetadata {
                            prompt: metadata_json.p,
                            seed: metadata_json.se,
                            cfg_scale: metadata_json.c,
                            steps: metadata_json.st,
                            sampler: metadata_json.sa,
                        },
                    });
                }
            }

            // Early exit if we've loaded enough
            if let Some(max) = max_samples {
                if items.len() >= max {
                    break;
                }
            }
        }

        println!("Loaded {} items", items.len());

        Ok(Self { items, tokenizer })
    }

    pub fn get_item(&self, index: usize) -> Option<&DiffusionItem> {
        self.items.get(index)
    }
}

impl Dataset<DiffusionItem> for DiffusionDataset {
    fn get(&self, index: usize) -> Option<DiffusionItem> {
        self.items.get(index).cloned()
    }

    fn len(&self) -> usize {
        self.items.len()
    }
}


// ============================================================================
// Batch Structure
// ============================================================================

#[derive(Clone, Debug)]
pub struct DiffusionBatch<B: Backend> {
    pub images: Tensor<B, 4>,           // [batch_size, 3, 64, 64] - clean images
    pub noisy_images: Tensor<B, 4>,     // [batch_size, 3, 64, 64] - noised images
    pub noise: Tensor<B, 4>,            // [batch_size, 3, 64, 64] - the noise added
    pub timesteps: Tensor<B, 1>,        // [batch_size] - timesteps
    pub text_tokens: Tensor<B, 2, Int>, // [batch_size, max_seq_len]
    pub text_mask: Tensor<B, 2>,        // [batch_size, max_seq_len]
}

// ============================================================================
// Batcher
// ============================================================================

#[derive(Clone)]
pub struct DiffusionBatcher<B: Backend> {
    device: B::Device,
    tokenizer: TextTokenizer,
    noise_schedule: NoiseSchedule,
}

impl<B: Backend> DiffusionBatcher<B> {
    pub fn new(device: B::Device, tokenizer: TextTokenizer) -> Self {
        let noise_schedule = NoiseSchedule::linear(NUM_TIMESTEPS, 0.0001, 0.02);
        Self {
            device,
            tokenizer,
            noise_schedule,
        }
    }

    fn load_image(&self, path: &Path) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let img = image::open(path)?;
        let img = img.to_rgb8();
        let (width, height) = img.dimensions();

        if width != IMAGE_SIZE as u32 || height != IMAGE_SIZE as u32 {
            return Err(format!(
                "Image dimensions {}x{} don't match expected {}x{}",
                width, height, IMAGE_SIZE, IMAGE_SIZE
            )
            .into());
        }

        // Convert to float and normalize to [-1, 1]
        let mut data = Vec::with_capacity(IMAGE_CHANNELS * IMAGE_SIZE * IMAGE_SIZE);
        for pixel in img.pixels() {
            let channels = pixel.0;
            for &channel in &channels {
                // Normalize from [0, 255] to [-1, 1]
                data.push((channel as f32 / 127.5) - 1.0);
            }
        }

        // Reorder from HWC to CHW
        let mut chw_data = vec![0.0; data.len()];
        for c in 0..IMAGE_CHANNELS {
            for h in 0..IMAGE_SIZE {
                for w in 0..IMAGE_SIZE {
                    let hwc_idx = (h * IMAGE_SIZE + w) * IMAGE_CHANNELS + c;
                    let chw_idx = c * (IMAGE_SIZE * IMAGE_SIZE) + h * IMAGE_SIZE + w;
                    chw_data[chw_idx] = data[hwc_idx];
                }
            }
        }

        Ok(chw_data)
    }
}

impl<B: Backend> Batcher<B, DiffusionItem, DiffusionBatch<B>> for DiffusionBatcher<B> {
    fn batch(&self, items: Vec<DiffusionItem>, device: &B::Device) -> DiffusionBatch<B> {
        let batch_size = items.len();
        let mut rng = rand::thread_rng();

        // Load and process images
        let mut all_images = Vec::new();
        let mut text_tokens_vec = Vec::new();
        let mut text_mask_vec = Vec::new();
        let mut timesteps_vec = Vec::new();

        // println!("Items length: {:?}", items.len());

        for item in items {
            // Load image
            let image_data = self.load_image(&item.image_path).unwrap_or_else(|e| {
                eprintln!("Failed to load image {:?}: {}", item.image_path, e);
                vec![0.0; IMAGE_CHANNELS * IMAGE_SIZE * IMAGE_SIZE]
            });
            all_images.extend(image_data);

            // Tokenize text
            let (tokens, mask) = self
                .tokenizer
                .encode(&item.metadata.prompt, MAX_SEQ_LEN)
                .unwrap_or_else(|e| {
                    eprintln!("Failed to tokenize prompt: {}", e);
                    (vec![0; MAX_SEQ_LEN], vec![false; MAX_SEQ_LEN])
                });
            text_tokens_vec.extend(tokens);
            text_mask_vec.extend(mask.iter().map(|&b| if b { 1.0 } else { 0.0 }));

            // Sample random timestep
            let timestep = rng.gen_range(0..NUM_TIMESTEPS);
            timesteps_vec.push(timestep as i32);
        }

        // Create tensors
        let images = Tensor::<B, 1>::from_floats(all_images.as_slice(), device)
            .reshape([batch_size, IMAGE_CHANNELS, IMAGE_SIZE, IMAGE_SIZE]);

        let text_tokens = Tensor::<B, 1, Int>::from_ints(
            text_tokens_vec
                .iter()
                .map(|&x| x as i32)
                .collect::<Vec<_>>()
                .as_slice(),
            device,
        )
        .reshape([batch_size, MAX_SEQ_LEN]);

        let text_mask = Tensor::<B, 1>::from_floats(text_mask_vec.as_slice(), device)
            .reshape([batch_size, MAX_SEQ_LEN]);

        let timesteps =
            Tensor::<B, 1, Int>::from_ints(timesteps_vec.as_slice(), device).float();

        // Generate noise and apply to images
        let noise = Tensor::<B, 4>::random_like(&images, burn::tensor::Distribution::Normal(0.0, 1.0));

        // Apply noise schedule to each item in batch
        let mut noisy_images_data: Vec<f32> = Vec::new();
        let images_data: Vec<f32> = images.clone().into_data().convert::<f32>().to_vec().unwrap();
        let noise_data: Vec<f32> = noise.clone().into_data().convert::<f32>().to_vec().unwrap();
        let timesteps_data: Vec<i32> = timesteps.clone().into_data().convert::<i32>().to_vec().unwrap();

        let img_size = IMAGE_CHANNELS * IMAGE_SIZE * IMAGE_SIZE;
        for i in 0..batch_size {
            let t = timesteps_data[i] as usize;
            let (sqrt_alpha_bar, sqrt_one_minus_alpha_bar) = self.noise_schedule.get_noise_params(t);

            for j in 0..img_size {
                let idx = i * img_size + j;
                let noisy_pixel =
                    sqrt_alpha_bar * images_data[idx] + sqrt_one_minus_alpha_bar * noise_data[idx];
                noisy_images_data.push(noisy_pixel);
            }
        }

        let noisy_images = Tensor::<B, 1>::from_floats(noisy_images_data.as_slice(), device)
            .reshape([batch_size, IMAGE_CHANNELS, IMAGE_SIZE, IMAGE_SIZE]);

        DiffusionBatch {
            images,
            noisy_images,
            noise,
            timesteps,
            text_tokens,
            text_mask,
        }
    }
}
