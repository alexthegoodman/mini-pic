#![recursion_limit="256"] // for wgpu?

pub mod dataset;
pub mod inference;
pub mod interface;
pub mod model;
pub mod training;
pub use burn::backend::wgpu::Wgpu;