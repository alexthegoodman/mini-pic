use burn::backend::wgpu::{Wgpu, WgpuDevice};
use burn::tensor::{Bool, Tensor};
use mini_pic::model::CrossAttention;

#[test]
fn padded_text_positions_cannot_change_cross_attention_output() {
    let device = WgpuDevice::default();
    let attention = CrossAttention::<Wgpu>::new(8, 8, 2, &device);
    let query = Tensor::<Wgpu, 3>::from_floats([[[0.25; 8]]], &device);
    let context = Tensor::<Wgpu, 3>::from_floats([[[0.5; 8], [0.0; 8]]], &device);
    let changed_pad = Tensor::<Wgpu, 3>::from_floats([[[0.5; 8], [100.0; 8]]], &device);
    let mask = Tensor::<Wgpu, 2, Bool>::from_bool([[false, true]].into(), &device);

    let expected = attention.forward(query.clone(), context.clone(), Some(mask.clone()));
    let actual = attention.forward(query.clone(), changed_pad.clone(), Some(mask));
    let difference = (actual - expected).abs().max().into_scalar();
    assert!(difference < 1e-5, "padding changed output by {difference}");

    let unmasked_a = attention.forward(query.clone(), context, None);
    let unmasked_b = attention.forward(query, changed_pad, None);
    let unmasked_difference = (unmasked_a - unmasked_b).abs().max().into_scalar();
    assert!(unmasked_difference > 1e-4, "test contexts did not affect attention");
}
