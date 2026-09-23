#![recursion_limit = "256"] // wgpu/naga auto-trait (Sync) checks overflow at the default

use burn::{backend::Autodiff, tensor::backend::Backend};
use mini_pic::{
    inference::{self, DiffusionInference},
    // interface::load_common_motion_2d,
    training,
};

// The actual per-run artifact dir is built dynamically from this run's
// hyperparameters (see training::artifact_dir_name) so different sweeps
// don't overwrite each other's checkpoint - this is just the parent folder.
static MODELS_ROOT: &str = "D:/models";

use burn::backend::wgpu::{Wgpu, WgpuDevice};

pub fn run_wgpu() {
    let device = WgpuDevice::default();
    run::<Wgpu>(device);
}

/// Train a regression model and predict results on a number of samples.
pub fn run<B: Backend>(device: B::Device) {
    training::run::<Autodiff<B>>(MODELS_ROOT, device.clone());
    // println!("Loading model...");
    // let inference: CommonMotionInference<B> = CommonMotionInference::new(device);
    // println!("Running inference...");
    // inference.infer("0, 5, 354, 154, 239, 91, \n1, 5, 544, 244, 106, 240, ".to_string());
}

fn main() {
    run_wgpu();
    // let inference = load_common_motion_2d();
    // println!("Running inference...");
    // inference.infer("0, 5, 354, 154, 239, 91, \n1, 5, 544, 244, 106, 240, ".to_string());
}