use burn::{
    module::Module,
    record::{NoStdTrainingRecorder, Recorder},
    tensor::{backend::Backend, Distribution, Int, Tensor},
};

use crate::{
    dataset::{NoiseSchedule, TextTokenizer, IMAGE_CHANNELS, IMAGE_SIZE, MAX_SEQ_LEN, NUM_TIMESTEPS},
    model::{UNet, UNetConfig},
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

        // Initialize model config with correct vocab size
        let model_config = UNetConfig::new(vec![64, 128, 256])
            .with_vocab_size(vocab_size)
            .with_text_embed_dim(256);

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

    /// Generate an image from a text prompt using DDPM sampling
    pub fn generate(
        &self,
        prompt: &str,
        num_inference_steps: usize,
        guidance_scale: f32,
    ) -> Tensor<B, 4> {
        println!("Generating image for prompt: \"{}\"", prompt);

        // Tokenize prompt
        let (tokens, _mask) = self
            .tokenizer
            .encode(prompt, MAX_SEQ_LEN)
            .expect("Failed to tokenize prompt");

        // Convert to tensor [1, MAX_SEQ_LEN]
        let text_tokens = Tensor::<B, 1, Int>::from_ints(
            tokens.iter().map(|&x| x as i32).collect::<Vec<_>>().as_slice(),
            &self.device,
        )
        .reshape([1, MAX_SEQ_LEN]);

        // Start from pure noise [1, 3, 64, 64]
        let mut x = Tensor::<B, 4>::random(
            [1, IMAGE_CHANNELS, IMAGE_SIZE, IMAGE_SIZE],
            Distribution::Normal(0.0, 1.0),
            &self.device,
        );

        // DDPM reverse diffusion process
        let step_size = NUM_TIMESTEPS / num_inference_steps;

        for step in (0..NUM_TIMESTEPS).step_by(step_size).rev() {
            let t = step;
            let timestep = Tensor::<B, 1>::from_floats([t as f32], &self.device);

            // Predict noise
            let predicted_noise = self.model.forward(
                x.clone(),
                timestep,
                text_tokens.clone(),
            );

            // Compute denoising step
            let alpha = self.noise_schedule.alphas[t];
            let alpha_bar = self.noise_schedule.alpha_bars[t];
            let beta = self.noise_schedule.betas[t];

            // x_{t-1} = (1 / sqrt(alpha_t)) * (x_t - (beta_t / sqrt(1 - alpha_bar_t)) * noise_pred)
            let coef1 = 1.0 / alpha.sqrt();
            let coef2 = beta / (1.0 - alpha_bar).sqrt();

            x = (x - predicted_noise * coef2) * coef1;

            // Add noise if not the final step
            if t > 0 {
                let noise = Tensor::<B, 4>::random_like(&x, Distribution::Normal(0.0, 1.0));
                let sigma = beta.sqrt();
                x = x + noise * sigma;
            }

            if step % 100 == 0 {
                println!("Denoising step {}/{}", NUM_TIMESTEPS - step, NUM_TIMESTEPS);
            }
        }

        // Clamp to [-1, 1]
        x = x.clamp(-1.0, 1.0);

        println!("Generation complete!");
        x
    }

    /// Generate multiple images from a prompt
    pub fn generate_batch(
        &self,
        prompts: &[String],
        num_inference_steps: usize,
        guidance_scale: f32,
    ) -> Tensor<B, 4> {
        let batch_size = prompts.len();
        println!("Generating {} images...", batch_size);

        // Tokenize all prompts
        let mut all_tokens = Vec::new();
        for prompt in prompts {
            let (tokens, _mask) = self
                .tokenizer
                .encode(prompt, MAX_SEQ_LEN)
                .expect("Failed to tokenize prompt");
            all_tokens.extend(tokens.iter().map(|&x| x as i32));
        }

        // Convert to tensor [batch_size, MAX_SEQ_LEN]
        let text_tokens = Tensor::<B, 1, Int>::from_ints(all_tokens.as_slice(), &self.device)
            .reshape([batch_size, MAX_SEQ_LEN]);

        // Start from pure noise [batch_size, 3, 64, 64]
        let mut x = Tensor::<B, 4>::random(
            [batch_size, IMAGE_CHANNELS, IMAGE_SIZE, IMAGE_SIZE],
            Distribution::Normal(0.0, 1.0),
            &self.device,
        );

        // DDPM reverse diffusion process
        let step_size = NUM_TIMESTEPS / num_inference_steps;

        for step in (0..NUM_TIMESTEPS).step_by(step_size).rev() {
            let t = step;
            let timesteps = Tensor::<B, 1>::from_floats(
                vec![t as f32; batch_size].as_slice(),
                &self.device,
            );

            // Predict noise
            let predicted_noise = self.model.forward(
                x.clone(),
                timesteps,
                text_tokens.clone(),
            );

            // Compute denoising step
            let alpha = self.noise_schedule.alphas[t];
            let alpha_bar = self.noise_schedule.alpha_bars[t];
            let beta = self.noise_schedule.betas[t];

            let coef1 = 1.0 / alpha.sqrt();
            let coef2 = beta / (1.0 - alpha_bar).sqrt();

            x = (x - predicted_noise * coef2) * coef1;

            // Add noise if not the final step
            if t > 0 {
                let noise = Tensor::<B, 4>::random_like(&x, Distribution::Normal(0.0, 1.0));
                let sigma = beta.sqrt();
                x = x + noise * sigma;
            }

            if step % 100 == 0 {
                println!("Denoising step {}/{}", NUM_TIMESTEPS - step, NUM_TIMESTEPS);
            }
        }

        // Clamp to [-1, 1]
        x = x.clamp(-1.0, 1.0);

        println!("Batch generation complete!");
        x
    }

    /// Save generated image tensor to a file
    /// Converts from [-1, 1] range to [0, 255] RGB
    pub fn save_image(image_tensor: Tensor<B, 4>, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        // Assuming image_tensor is [1, 3, 64, 64] in CHW format
        let [_batch, channels, height, width] = image_tensor.dims();

        if channels != 3 || height != IMAGE_SIZE || width != IMAGE_SIZE {
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
        for c in 0..3 {
            for h in 0..IMAGE_SIZE {
                for w in 0..IMAGE_SIZE {
                    let chw_idx = c * (IMAGE_SIZE * IMAGE_SIZE) + h * IMAGE_SIZE + w;
                    let hwc_idx = (h * IMAGE_SIZE + w) * 3 + c;
                    rgb_data[hwc_idx] = image_data[chw_idx] as u8;
                }
            }
        }

        // Save using image crate
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
