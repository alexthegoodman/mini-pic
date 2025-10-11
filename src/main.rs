use burn::{backend::Autodiff, tensor::backend::Backend};
use mini_pic::{
    inference::{self, DiffusionInference},
    // interface::load_common_motion_2d,
    training,
};

static ARTIFACT_DIR: &str = "/tmp/mini-pic-v001";

use burn::backend::cuda::{Cuda, CudaDevice};

pub fn run_cuda() {
    let device = CudaDevice::default();
    run::<Cuda>(device);
}

/// Train a regression model and predict results on a number of samples.
pub fn run<B: Backend>(device: B::Device) {
    training::run::<Autodiff<B>>(ARTIFACT_DIR, device.clone());
    // println!("Loading model...");
    // let inference: CommonMotionInference<B> = CommonMotionInference::new(device);
    // println!("Running inference...");
    // inference.infer("0, 5, 354, 154, 239, 91, \n1, 5, 544, 244, 106, 240, ".to_string());
}

fn main() {
    run_cuda();
    // let inference = load_common_motion_2d();
    // println!("Running inference...");
    // inference.infer("0, 5, 354, 154, 239, 91, \n1, 5, 544, 244, 106, 240, ".to_string());
}