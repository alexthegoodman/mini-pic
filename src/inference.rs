use burn::{
    config::Config,
    module::Module,
    record::{NoStdTrainingRecorder, Recorder},
    tensor::{backend::Backend, Distribution, Int, Tensor},
};
use std::path::Path;

use crate::{
    dataset::{NoiseSchedule, TextTokenizer, IMAGE_CHANNELS, IMAGE_SIZE, MAX_SEQ_LEN, NUM_TIMESTEPS},
    model::UNet,
    training::TrainingConfig,
};

/// Inference engine for diffusion-based image generation
pub struct DiffusionInference<B: Backend> {
    pub model: UNet<B>,
    pub tokenizer: TextTokenizer,
    pub noise_schedule: NoiseSchedule,
    pub device: B::Device,
}

impl<B: Backend> DiffusionInference<B> {
    /// Load a trained diffusion model from a checkpoint
    pub fn new(
        model_path: &str,
        tokenizer_path: &str,
        device: B::Device,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        // Load tokenizer
        println!("Loading tokenizer from {}...", tokenizer_path);
        let tokenizer = TextTokenizer::from_file(tokenizer_path)?;
        let vocab_size = tokenizer.vocab_size();

        // Load the exact architecture this checkpoint was trained with, from
        // the config.json training::run() saves next to the model file (same
        // directory). This used to reconstruct an independently-guessed
        // UNetConfig (channels [64,128,256], text_embed_dim 256) that had
        // already drifted from every real training config in this repo -
        // load_record below would either panic on a shape mismatch or,
        // worse, silently succeed against the wrong architecture.
        let config_path = Path::new(model_path)
            .parent()
            .map(|dir| dir.join("config.json"))
            .ok_or("model_path has no parent directory to find its config.json in")?;
        let training_config = TrainingConfig::load(&config_path)
            .map_err(|e| format!("Failed to load {}: {:?}", config_path.display(), e))?;
        let model_config = training_config.model.with_vocab_size(vocab_size);

        // Load trained model
        println!("Loading model from {}...", model_path);
        let record = NoStdTrainingRecorder::new()
            .load(model_path.into(), &device)
            .expect("Failed to load trained model");

        let model = model_config.init(&device).load_record(record);

        // Initialize noise schedule
        let noise_schedule = NoiseSchedule::linear(NUM_TIMESTEPS, 0.0001, 0.02);

        println!("Model loaded successfully!");

        Ok(Self {
            model,
            tokenizer,
            noise_schedule,
            device,
        })
    }

    /// Generate an image from a text prompt using Python parity DDIM sampling.
    pub fn generate(&self, prompt: &str, num_inference_steps: usize) -> Tensor<B, 4> {
        assert!((1..=NUM_TIMESTEPS).contains(&num_inference_steps));
        println!("Generating image for prompt: \"{}\"", prompt);

        // Tokenize prompt
        let (tokens, mask) = self
            .tokenizer
            .encode(prompt, MAX_SEQ_LEN)
            .expect("Failed to tokenize prompt");

        // Convert to tensor [1, MAX_SEQ_LEN]
        let text_tokens = Tensor::<B, 1, Int>::from_ints(
            tokens.iter().map(|&x| x as i32).collect::<Vec<_>>().as_slice(),
            &self.device,
        )
        .reshape([1, MAX_SEQ_LEN]);

        // Same polarity flip UNet::forward_step uses: encode()'s mask is
        // true at valid tokens, but the transformer's mask_pad wants
        // true = ignore.
        let text_mask = Tensor::<B, 1>::from_floats(
            mask.iter().map(|&valid| if valid { 1.0 } else { 0.0 }).collect::<Vec<_>>().as_slice(),
            &self.device,
        )
        .reshape([1, MAX_SEQ_LEN]);
        let mask_pad = text_mask.equal_elem(0.0);

        // Start from pure noise [1, 3, 64, 64]
        let mut x = Tensor::<B, 4>::random(
            [1, IMAGE_CHANNELS, IMAGE_SIZE, IMAGE_SIZE],
            Distribution::Normal(0.0, 1.0),
            &self.device,
        );

        // Python's np.linspace(999, 0, steps, dtype=int), traversed in reverse.
        let timesteps = ddim_timesteps(num_inference_steps);

        for (i, &t) in timesteps.iter().enumerate().rev() {
            let timestep = Tensor::<B, 1>::from_floats([t as f32], &self.device);

            // Predict noise
            let predicted_noise = self.model.forward(
                x.clone(),
                timestep,
                text_tokens.clone(),
                Some(mask_pad.clone()),
            );

            let alpha_bar_t = self.noise_schedule.alpha_bars[t];
            let alpha_bar_prev = if i > 0 {
                self.noise_schedule.alpha_bars[timesteps[i - 1]]
            } else {
                1.0
            };

            x = ddim_epsilon_step(x, predicted_noise, alpha_bar_t, alpha_bar_prev);

            if t % 100 == 0 {
                println!("Denoising step {}/{}", NUM_TIMESTEPS - t, NUM_TIMESTEPS);
            }
        }

        // Clamp to [-1, 1]
        x = x.clamp(-1.0, 1.0);

        println!("Generation complete!");
        x
    }

    /// Save a generated [1, 3, 64, 64] image tensor to a file.
    /// Converts from [-1, 1] range to [0, 255] RGB.
    pub fn save_image(image_tensor: Tensor<B, 4>, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let [_batch, channels, height, width] = image_tensor.dims();

        if channels != IMAGE_CHANNELS || height != IMAGE_SIZE || width != IMAGE_SIZE {
            return Err("Invalid image dimensions".into());
        }

        // Convert to [0, 255] range
        let image_data: Vec<f32> = ((image_tensor.clone() + 1.0) * 127.5)
            .clamp(0.0, 255.0)
            .into_data()
            .convert::<f32>()
            .to_vec()
            .expect("Failed to convert tensor to vec");

        // Convert from CHW to HWC format
        let mut rgb_data = vec![0u8; IMAGE_SIZE * IMAGE_SIZE * 3];
        for c in 0..IMAGE_CHANNELS {
            for h in 0..IMAGE_SIZE {
                for w in 0..IMAGE_SIZE {
                    let chw_idx = c * (IMAGE_SIZE * IMAGE_SIZE) + h * IMAGE_SIZE + w;
                    let hwc_idx = (h * IMAGE_SIZE + w) * 3 + c;
                    rgb_data[hwc_idx] = image_data[chw_idx] as u8;
                }
            }
        }

        image::save_buffer(
            path,
            &rgb_data,
            IMAGE_SIZE as u32,
            IMAGE_SIZE as u32,
            image::ColorType::Rgb8,
        )?;

        println!("Image saved to {}", path);
        Ok(())
    }
}

fn ddim_timesteps(steps: usize) -> Vec<usize> {
    if steps == 1 {
        return vec![NUM_TIMESTEPS - 1];
    }
    (0..steps)
        .map(|i| i * (NUM_TIMESTEPS - 1) / (steps - 1))
        .collect()
}

fn ddim_epsilon_step<B: Backend>(
    x: Tensor<B, 4>,
    predicted_noise: Tensor<B, 4>,
    alpha_bar_t: f32,
    alpha_bar_prev: f32,
) -> Tensor<B, 4> {
    // Same deterministic (eta=0) update and x0 clipping as python-train/inference.py.
    let pred_x0 = ((x - predicted_noise.clone() * ((1.0 - alpha_bar_t).sqrt() as f64))
        / (alpha_bar_t.sqrt() as f64))
        .clamp(-1.0, 1.0);
    pred_x0 * (alpha_bar_prev.sqrt() as f64)
        + predicted_noise * ((1.0 - alpha_bar_prev).sqrt() as f64)
}

/// Helper function to interpolate between two prompts for animation
pub fn interpolate_prompts(prompt1: &str, prompt2: &str, steps: usize) -> Vec<String> {
    // Simple implementation: just return the two prompts
    // In a real implementation, you'd interpolate in latent space
    let mut prompts = Vec::new();
    for i in 0..steps {
        if i < steps / 2 {
            prompts.push(prompt1.to_string());
        } else {
            prompts.push(prompt2.to_string());
        }
    }
    prompts
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::wgpu::{Wgpu, WgpuDevice};

    #[test]
    fn ddim_schedule_covers_training_endpoints() {
        let steps = ddim_timesteps(50);
        assert_eq!(steps.len(), 50);
        assert_eq!(steps[0], 0);
        assert_eq!(*steps.last().unwrap(), 999);
        assert!(steps.windows(2).all(|pair| pair[0] < pair[1]));
    }

    #[test]
    fn ddim_step_matches_python_epsilon_update() {
        let device = WgpuDevice::default();
        let x0 = Tensor::<Wgpu, 1>::from_floats([0.5, -0.25, 0.75, -0.5], &device)
            .reshape([1, 1, 2, 2]);
        let noise = Tensor::<Wgpu, 1>::from_floats([1.25, -0.75, 0.3, -1.1], &device)
            .reshape([1, 1, 2, 2]);
        let schedule = NoiseSchedule::linear(NUM_TIMESTEPS, 0.0001, 0.02);

        for (t, previous) in [(999, Some(900)), (900, Some(500)), (500, Some(0)), (0, None)] {
            let alpha_t = schedule.alpha_bars[t];
            let alpha_prev = previous.map(|p| schedule.alpha_bars[p]).unwrap_or(1.0);
            let noisy = x0.clone() * (alpha_t.sqrt() as f64)
                + noise.clone() * ((1.0 - alpha_t).sqrt() as f64);
            let expected = x0.clone() * (alpha_prev.sqrt() as f64)
                + noise.clone() * ((1.0 - alpha_prev).sqrt() as f64);
            let actual = ddim_epsilon_step(noisy, noise.clone(), alpha_t, alpha_prev);
            let error = (actual - expected).abs().max().into_scalar();
            assert!(error < 0.0001, "t={t}, error={error}");
        }
    }
}
