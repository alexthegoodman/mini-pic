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

    /// Generate an image from a text prompt using DDPM sampling.
    pub fn generate(&self, prompt: &str, num_inference_steps: usize) -> Tensor<B, 4> {
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

        // DDPM reverse diffusion process. When num_inference_steps < NUM_TIMESTEPS
        // we only visit a subsequence of timesteps (e.g. 0, 20, 40, ..., 980 for
        // 50 steps over 1000), so each iteration must denoise across the *gap*
        // between the current and next timestep in the subsequence, not a single
        // raw timestep. Using the single-step alpha_t/beta_t here (as if step_size
        // were always 1) undercorrects by a factor of step_size: on a subsequence
        // of every 20th timestep it removes about 1/20th of the noise a full
        // schedule would, so x barely denoises no matter how accurate the model's
        // predictions are - respacing alpha/beta to the actual gap (Nichol &
        // Dhariwal's "respaced" schedule: alpha_t' = alpha_bar_t / alpha_bar_prev)
        // fixes this for any step count, matching the num_inference_steps=NUM_TIMESTEPS
        // case exactly when step_size == 1.
        let step_size = (NUM_TIMESTEPS / num_inference_steps).max(1);
        let timesteps: Vec<usize> = (0..NUM_TIMESTEPS).step_by(step_size).collect();

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

            // Respaced single-jump alpha/beta covering exactly this subsequence
            // step (falls back to the model's own per-timestep alpha/beta when
            // step_size == 1, i.e. num_inference_steps == NUM_TIMESTEPS).
            let alpha_eff = alpha_bar_t / alpha_bar_prev;
            let beta_eff = 1.0 - alpha_eff;

            // x_{prev} = (1 / sqrt(alpha_eff)) * (x_t - (beta_eff / sqrt(1 - alpha_bar_t)) * noise_pred)
            let coef1 = 1.0 / alpha_eff.sqrt();
            let coef2 = beta_eff / (1.0 - alpha_bar_t).sqrt();

            x = (x - predicted_noise * coef2) * coef1;
            // Diffusion models are trained on x in [-1, 1]; without clamping every
            // step (not just once at the end), small per-step prediction error
            // compounds multiplicatively through coef1 (>1 every step) across the
            // whole trajectory and x explodes to tens of times its starting scale,
            // saturating to +/-1 per-pixel independently on the final clamp - i.e.
            // uncorrelated noise, regardless of how accurate the model is.
            x = x.clamp(-1.0, 1.0);

            // Add noise if not the final step
            if i > 0 {
                let noise = Tensor::<B, 4>::random_like(&x, Distribution::Normal(0.0, 1.0));
                let sigma = beta_eff.sqrt();
                x = x + noise * sigma;
            }

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
