use burn::backend::wgpu::{Wgpu, WgpuDevice};
use burn::tensor::{Bool, Int, Tensor};
use mini_pic::model::{CrossAttention, UNetConfig};

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

#[test]
fn unet_round_trips_64_pixels_through_four_downsampling_stages() {
    let device = WgpuDevice::default();
    let model = UNetConfig::new(vec![8, 8, 8, 8, 8])
        .with_vocab_size(16)
        .with_text_embed_dim(8)
        .with_text_encoder_layers(1)
        .with_text_encoder_heads(2)
        .with_text_encoder_dropout(0.0)
        .with_time_embed_dim(8)
        .with_use_mid_attn(true)
        .with_resnet_blocks_per_level(1)
        .init::<Wgpu>(&device);
    let images = Tensor::<Wgpu, 4>::zeros([1, 3, 64, 64], &device);
    let timesteps = Tensor::<Wgpu, 1>::zeros([1], &device);
    let tokens = Tensor::<Wgpu, 2, Int>::zeros([1, 4], &device);

    let output = model.forward(images, timesteps, tokens, None);
    assert_eq!(output.dims(), [1, 3, 64, 64]);
}
