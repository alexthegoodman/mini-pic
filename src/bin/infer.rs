#![recursion_limit = "256"] // wgpu/naga auto-trait (Sync) checks overflow at the default

use burn::backend::wgpu::{Wgpu, WgpuDevice};
use mini_pic::inference::DiffusionInference;

/// Usage: infer.exe "<prompt>" <model_dir> [steps] [output_path]
///
/// model_dir must contain the model/config.json pair training::run() saves -
/// its name is derived from that run's hyperparameters (see
/// training::artifact_dir_name), so there's no fixed default; copy the
/// "Artifact dir: ..." path training printed at the start of that run.
fn main() {
    let args: Vec<String> = std::env::args().collect();

    let usage = || {
        eprintln!("Usage: infer \"<prompt>\" <model_dir> [steps] [output_path]");
        std::process::exit(1);
    };

    let prompt = args.get(1).cloned().unwrap_or_else(usage);
    let model_dir = args.get(2).cloned().unwrap_or_else(usage);
    let steps: usize = args
        .get(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(50);
    let output_path = args.get(4).cloned().unwrap_or_else(|| "output.png".to_string());

    let model_path = format!("{model_dir}/model");
    let tokenizer_path = "tokenizer.json".to_string();

    let device = WgpuDevice::default();
    let inference = DiffusionInference::<Wgpu>::new(&model_path, &tokenizer_path, device)
        .expect("Failed to load model");

    let image = inference.generate(&prompt, steps);

    DiffusionInference::<Wgpu>::save_image(image, &output_path).expect("Failed to save image");

    println!("Done: {}", output_path);
}
