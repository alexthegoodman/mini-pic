// use crate::{
//     inference::DiffusionInference,
//     training,
// };
// use burn::backend::wgpu::{Wgpu, WgpuDevice};
// use burn::tensor::backend::Backend;

// /// Load the diffusion model with WGPU backend
// pub fn load_diffusion_model() -> Result<DiffusionInference<Wgpu>, Box<dyn std::error::Error>> {
//     let device = WgpuDevice::default();

//     load_model_wgpu::<Wgpu>(
//         device,
//         "model",
//         "tokenizer.json",
//     )
// }

// /// Load the diffusion model with a generic backend
// pub fn load_model_wgpu<B: Backend>(
//     device: B::Device,
//     model_path: &str,
//     tokenizer_path: &str,
// ) -> Result<DiffusionInference<B>, Box<dyn std::error::Error>> {
//     println!("Loading diffusion model...");
//     let inference = DiffusionInference::new(model_path, tokenizer_path, device)?;
//     println!("Model loaded successfully!");
//     Ok(inference)
// }

// /// Quick helper to run training
// pub fn run_training() {
//     let device = WgpuDevice::default();
//     training::run::<burn::backend::Autodiff<Wgpu>>("artifacts", device);
// }
