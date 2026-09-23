use crate::dataset::{DiffusionBatch, IMAGE_CHANNELS};
use burn::{
    config::Config,
    module::Module,
    nn::{
        conv::{Conv2d, Conv2dConfig},
        loss::MseLoss,
        Embedding, EmbeddingConfig, Gelu, GroupNorm, GroupNormConfig, Linear, LinearConfig,
    },
    tensor::{
        activation::softmax,
        backend::{AutodiffBackend, Backend},
        Int, Tensor,
    },
    train::{
        RegressionOutput, TrainOutput, TrainStep, ValidStep,
    },
};

// ============================================================================
// Constants
// ============================================================================

// TIME_EMBED_DIM is now configurable via UNetConfig

// ============================================================================
// Time Embedding (Sinusoidal Position Embeddings)
// ============================================================================

#[derive(Module, Debug)]
pub struct TimeEmbedding<B: Backend> {
    mlp: Linear<B>,
    activation: Gelu,
    time_embed_dim: usize,
}

impl<B: Backend> TimeEmbedding<B> {
    pub fn new(time_embed_dim: usize, device: &B::Device) -> Self {
        let mlp = LinearConfig::new(time_embed_dim, time_embed_dim * 4)
            .with_bias(true)
            .init(device);

        Self {
            mlp,
            activation: Gelu::new(),
            time_embed_dim,
        }
    }

    pub fn forward(&self, timesteps: Tensor<B, 1>) -> Tensor<B, 2> {
        // Create sinusoidal position embeddings
        let batch_size = timesteps.dims()[0];
        let device = timesteps.device();

        // Generate frequency bands
        let half_dim = self.time_embed_dim / 2;
        let emb_scale = (10000.0_f32).ln() / (half_dim as f32 - 1.0);

        let mut frequencies = Vec::new();
        for i in 0..half_dim {
            frequencies.push((-emb_scale * i as f32).exp());
        }

        let freq_tensor = Tensor::<B, 1>::from_floats(frequencies.as_slice(), &device)
            .reshape([1, half_dim])
            .repeat(&[batch_size, 1]);

        let timesteps_expanded = timesteps.clone().reshape([batch_size, 1]);
        let args = timesteps_expanded * freq_tensor;

        // Compute sin and cos
        let sin_emb = args.clone().sin();
        let cos_emb = args.cos();

        // Concatenate [sin, cos]
        let emb = Tensor::cat(vec![sin_emb, cos_emb], 1); // [batch_size, time_embed_dim]

        // Pass through MLP
        let emb = self.mlp.forward(emb);
        self.activation.forward(emb)
    }
}

// ============================================================================
// Attention Modules
// ============================================================================

#[derive(Module, Debug)]
pub struct SelfAttention<B: Backend> {
    query: Linear<B>,
    key: Linear<B>,
    value: Linear<B>,
    proj_out: Linear<B>,
    n_heads: usize,
    d_head: usize,
}

impl<B: Backend> SelfAttention<B> {
    pub fn new(channels: usize, n_heads: usize, device: &B::Device) -> Self {
        assert_eq!(
            channels % n_heads,
            0,
            "channels must be divisible by n_heads"
        );
        let d_head = channels / n_heads;

        Self {
            query: LinearConfig::new(channels, channels).init(device),
            key: LinearConfig::new(channels, channels).init(device),
            value: LinearConfig::new(channels, channels).init(device),
            proj_out: LinearConfig::new(channels, channels).init(device),
            n_heads,
            d_head,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch_size, seq_len, channels] = x.dims();

        let q = self.query.forward(x.clone());
        let k = self.key.forward(x.clone());
        let v = self.value.forward(x);

        // Reshape to [batch, heads, seq_len, d_head]
        let q = q.reshape([batch_size, seq_len, self.n_heads, self.d_head]);
        let q = q.swap_dims(1, 2); // [batch, n_heads, seq_len, d_head]

        let k = k.reshape([batch_size, seq_len, self.n_heads, self.d_head]);
        let k = k.swap_dims(1, 2);

        let v = v.reshape([batch_size, seq_len, self.n_heads, self.d_head]);
        let v = v.swap_dims(1, 2);

        // Attention: softmax(Q @ K^T / sqrt(d_head)) @ V
        let scale = (self.d_head as f64).sqrt();
        let scores = q.matmul(k.swap_dims(2, 3)) / scale; // [batch, n_heads, seq_len, seq_len]
        let attn = softmax(scores, 3);

        let out = attn.matmul(v); // [batch, n_heads, seq_len, d_head]
        let out = out.swap_dims(1, 2); // [batch, seq_len, n_heads, d_head]
        let out = out.reshape([batch_size, seq_len, channels]);

        self.proj_out.forward(out)
    }
}

#[derive(Module, Debug)]
pub struct CrossAttention<B: Backend> {
    query: Linear<B>,
    key: Linear<B>,
    value: Linear<B>,
    proj_out: Linear<B>,
    n_heads: usize,
    d_head: usize,
}

impl<B: Backend> CrossAttention<B> {
    pub fn new(channels: usize, context_dim: usize, n_heads: usize, device: &B::Device) -> Self {
        assert_eq!(
            channels % n_heads,
            0,
            "channels must be divisible by n_heads"
        );
        let d_head = channels / n_heads;

        Self {
            query: LinearConfig::new(channels, channels).init(device),
            key: LinearConfig::new(context_dim, channels).init(device),
            value: LinearConfig::new(context_dim, channels).init(device),
            proj_out: LinearConfig::new(channels, channels).init(device),
            n_heads,
            d_head,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>, context: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch_size, seq_len, channels] = x.dims();
        let context_len = context.dims()[1];

        let q = self.query.forward(x);
        let k = self.key.forward(context.clone());
        let v = self.value.forward(context);

        // Reshape to multi-head format
        let q = q
            .reshape([batch_size, seq_len, self.n_heads, self.d_head])
            .swap_dims(1, 2);
        let k = k
            .reshape([batch_size, context_len, self.n_heads, self.d_head])
            .swap_dims(1, 2);
        let v = v
            .reshape([batch_size, context_len, self.n_heads, self.d_head])
            .swap_dims(1, 2);

        // Attention
        let scale = (self.d_head as f64).sqrt();
        let scores = q.matmul(k.swap_dims(2, 3)) / scale;
        let attn = softmax(scores, 3);

        let out = attn.matmul(v);
        let out = out.swap_dims(1, 2).reshape([batch_size, seq_len, channels]);

        self.proj_out.forward(out)
    }
}

// ============================================================================
// ResNet Block with Time Conditioning
// ============================================================================

#[derive(Module, Debug)]
pub struct ResNetBlock<B: Backend> {
    conv1: Conv2d<B>,
    conv2: Conv2d<B>,
    norm1: GroupNorm<B>,
    norm2: GroupNorm<B>,
    time_mlp: Linear<B>,
    activation: Gelu,
    residual_conv: Option<Conv2d<B>>,
}

impl<B: Backend> ResNetBlock<B> {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        time_emb_dim: usize,
        device: &B::Device,
    ) -> Self {
        let conv1 = Conv2dConfig::new([in_channels, out_channels], [3, 3])
            .with_padding(burn::nn::PaddingConfig2d::Explicit(1, 1))
            .init(device);

        let conv2 = Conv2dConfig::new([out_channels, out_channels], [3, 3])
            .with_padding(burn::nn::PaddingConfig2d::Explicit(1, 1))
            .init(device);

        let norm1 = GroupNormConfig::new(8, out_channels).init(device);
        let norm2 = GroupNormConfig::new(8, out_channels).init(device);

        let time_mlp = LinearConfig::new(time_emb_dim, out_channels).init(device);

        let residual_conv = if in_channels != out_channels {
            Some(Conv2dConfig::new([in_channels, out_channels], [1, 1]).init(device))
        } else {
            None
        };

        Self {
            conv1,
            conv2,
            norm1,
            norm2,
            time_mlp,
            activation: Gelu::new(),
            residual_conv,
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>, time_emb: Tensor<B, 2>) -> Tensor<B, 4> {
        let residual = match &self.residual_conv {
            Some(conv) => conv.forward(x.clone()),
            None => x.clone(),
        };

        // First conv block
        let mut h = self.norm1.forward(x);
        h = self.activation.forward(h);
        h = self.conv1.forward(h);

        // Add time embedding
        let time_out = self.activation.forward(self.time_mlp.forward(time_emb));
        let [batch, channels, height, width] = h.dims();
        let time_out = time_out.reshape([batch, channels, 1, 1]);
        let time_out = time_out.repeat(&[1, 1, height, width]);
        h = h + time_out;

        // Second conv block
        h = self.norm2.forward(h);
        h = self.activation.forward(h);
        h = self.conv2.forward(h);

        // Residual connection
        h + residual
    }
}

// ============================================================================
// Attention Block (combines spatial self-attention + cross-attention)
// ============================================================================

#[derive(Module, Debug)]
pub struct AttentionBlock<B: Backend> {
    norm1: GroupNorm<B>,
    self_attn: SelfAttention<B>,
    norm2: GroupNorm<B>,
    cross_attn: CrossAttention<B>,
    norm3: GroupNorm<B>,
    ffn: Linear<B>,
}

impl<B: Backend> AttentionBlock<B> {
    pub fn new(
        channels: usize,
        context_dim: usize,
        n_heads: usize,
        device: &B::Device,
    ) -> Self {
        Self {
            norm1: GroupNormConfig::new(8, channels).init(device),
            self_attn: SelfAttention::new(channels, n_heads, device),
            norm2: GroupNormConfig::new(8, channels).init(device),
            cross_attn: CrossAttention::new(channels, context_dim, n_heads, device),
            norm3: GroupNormConfig::new(8, channels).init(device),
            ffn: LinearConfig::new(channels, channels).init(device),
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>, context: Tensor<B, 3>) -> Tensor<B, 4> {
        let [batch, channels, height, width] = x.dims();
        let seq_len = height * width;

        // Reshape to sequence format for attention
        let x_seq = x
            .clone()
            .reshape([batch, channels, seq_len])
            .swap_dims(1, 2); // [batch, seq_len, channels]

        // Self-attention
        let attn_out = self.self_attn.forward(self.norm1_seq(x_seq.clone()));
        let x_seq = x_seq.clone() + attn_out;

        // Cross-attention with text context
        let cross_out = self.cross_attn.forward(self.norm2_seq(x_seq.clone()), context);
        let x_seq = x_seq.clone() + cross_out;

        // FFN
        let ffn_out = self.ffn.forward(self.norm3_seq(x_seq.clone()));
        let x_seq = x_seq + ffn_out;

        // Reshape back to image format
        x_seq.swap_dims(1, 2).reshape([batch, channels, height, width])
    }

    // Helper methods to apply group norm on sequence data
    fn norm1_seq(&self, x_seq: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, seq_len, channels] = x_seq.dims();
        let height = (seq_len as f32).sqrt() as usize;
        let x_img = x_seq.swap_dims(1, 2).reshape([batch, channels, height, height]);
        let normed = self.norm1.forward(x_img);
        normed.reshape([batch, channels, seq_len]).swap_dims(1, 2)
    }

    fn norm2_seq(&self, x_seq: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, seq_len, channels] = x_seq.dims();
        let height = (seq_len as f32).sqrt() as usize;
        let x_img = x_seq.swap_dims(1, 2).reshape([batch, channels, height, height]);
        let normed = self.norm2.forward(x_img);
        normed.reshape([batch, channels, seq_len]).swap_dims(1, 2)
    }

    fn norm3_seq(&self, x_seq: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, seq_len, channels] = x_seq.dims();
        let height = (seq_len as f32).sqrt() as usize;
        let x_img = x_seq.swap_dims(1, 2).reshape([batch, channels, height, height]);
        let normed = self.norm3.forward(x_img);
        normed.reshape([batch, channels, seq_len]).swap_dims(1, 2)
    }
}

// ============================================================================
// Down/Up Blocks
// ============================================================================

#[derive(Module, Debug)]
pub struct DownBlock<B: Backend> {
    resnet1: ResNetBlock<B>,
    resnet2: Option<ResNetBlock<B>>,
    attn: Option<AttentionBlock<B>>,
    downsample: Option<Conv2d<B>>,
}

impl<B: Backend> DownBlock<B> {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        time_emb_dim: usize,
        context_dim: usize,
        use_attn: bool,
        downsample: bool,
        num_resnet_blocks: usize,
        device: &B::Device,
    ) -> Self {
        let resnet1 = ResNetBlock::new(in_channels, out_channels, time_emb_dim, device);
        let resnet2 = if num_resnet_blocks > 1 {
            Some(ResNetBlock::new(out_channels, out_channels, time_emb_dim, device))
        } else {
            None
        };

        let attn = if use_attn {
            Some(AttentionBlock::new(out_channels, context_dim, 4, device))
        } else {
            None
        };

        let downsample = if downsample {
            Some(
                Conv2dConfig::new([out_channels, out_channels], [4, 4])
                    .with_stride([2, 2])
                    .with_padding(burn::nn::PaddingConfig2d::Explicit(1, 1))
                    .init(device),
            )
        } else {
            None
        };

        Self {
            resnet1,
            resnet2,
            attn,
            downsample,
        }
    }

    pub fn forward(
        &self,
        x: Tensor<B, 4>,
        time_emb: Tensor<B, 2>,
        context: Tensor<B, 3>,
    ) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let mut h = self.resnet1.forward(x, time_emb.clone());

        if let Some(ref resnet2) = self.resnet2 {
            h = resnet2.forward(h, time_emb);
        }

        if let Some(ref attn) = self.attn {
            h = attn.forward(h, context);
        }

        let h_skip = h.clone();

        let h_out = if let Some(ref down) = self.downsample {
            down.forward(h)
        } else {
            h
        };

        (h_out, h_skip)
    }
}

#[derive(Module, Debug)]
pub struct UpBlock<B: Backend> {
    resnet1: ResNetBlock<B>,
    resnet2: Option<ResNetBlock<B>>,
    resnet3: Option<ResNetBlock<B>>,
    attn: Option<AttentionBlock<B>>,
    upsample: Option<Conv2d<B>>,
}

impl<B: Backend> UpBlock<B> {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        time_emb_dim: usize,
        context_dim: usize,
        use_attn: bool,
        upsample: bool,
        num_resnet_blocks: usize,
        device: &B::Device,
    ) -> Self {
        let resnet1 = ResNetBlock::new(in_channels + out_channels, out_channels, time_emb_dim, device);
        let resnet2 = if num_resnet_blocks > 1 {
            Some(ResNetBlock::new(out_channels, out_channels, time_emb_dim, device))
        } else {
            None
        };
        let resnet3 = if num_resnet_blocks > 2 {
            Some(ResNetBlock::new(out_channels, out_channels, time_emb_dim, device))
        } else {
            None
        };

        let attn = if use_attn {
            Some(AttentionBlock::new(out_channels, context_dim, 4, device))
        } else {
            None
        };

        let upsample = if upsample {
            // Use ConvTranspose2d for upsampling
            Some(
                Conv2dConfig::new([out_channels, out_channels], [4, 4])
                    .with_stride([1, 1])
                    .with_padding(burn::nn::PaddingConfig2d::Explicit(1, 1))
                    .init(device),
            )
        } else {
            None
        };

        Self {
            resnet1,
            resnet2,
            resnet3,
            attn,
            upsample,
        }
    }

    pub fn forward(
        &self,
        x: Tensor<B, 4>,
        skip: Tensor<B, 4>,
        time_emb: Tensor<B, 2>,
        context: Tensor<B, 3>,
    ) -> Tensor<B, 4> {
        // Concatenate skip connection
        let h = Tensor::cat(vec![x, skip], 1);

        let mut h = self.resnet1.forward(h, time_emb.clone());

        if let Some(ref resnet2) = self.resnet2 {
            h = resnet2.forward(h, time_emb.clone());
        }

        if let Some(ref resnet3) = self.resnet3 {
            h = resnet3.forward(h, time_emb);
        }

        if let Some(ref attn) = self.attn {
            h = attn.forward(h, context);
        }

        if let Some(ref up) = self.upsample {
            // Manual bilinear upsample then conv
            let [batch, channels, height, width] = h.dims();
            let h_upsampled = h.clone().reshape([batch, channels, height, 1, width, 1]);
            let h_upsampled = h_upsampled.repeat(&[1, 1, 1, 2, 1, 2]);
            let h_upsampled = h_upsampled.reshape([batch, channels, height * 2, width * 2]);
            up.forward(h_upsampled)
        } else {
            h
        }
    }
}

// ============================================================================
// U-Net Model
// ============================================================================

#[derive(Module, Debug)]
pub struct UNet<B: Backend> {
    // Text encoder
    text_embedding: Embedding<B>,
    text_encoder: Linear<B>,

    // Time embedding
    time_embedding: TimeEmbedding<B>,

    // Initial convolution
    conv_in: Conv2d<B>,

    // Encoder
    down1: DownBlock<B>,
    down2: DownBlock<B>,
    down3: DownBlock<B>,

    // Bottleneck
    mid_block1: ResNetBlock<B>,
    mid_attn: Option<AttentionBlock<B>>,
    mid_block2: ResNetBlock<B>,

    // Decoder
    up1: UpBlock<B>,
    up2: UpBlock<B>,
    up3: UpBlock<B>,

    // Output
    norm_out: GroupNorm<B>,
    conv_out: Conv2d<B>,
    activation: Gelu,
}

#[derive(Config)]
pub struct UNetConfig {
    #[config(default = 8192)]
    pub vocab_size: usize,
    // Unswept here - python-train's UNet found 64 helped over its own 32
    // default (see its TextEncoder), but that encoder is a multi-layer
    // Transformer; this one is Embedding+Linear only, so the two aren't
    // directly comparable and the value hasn't been tuned on this side yet.
    #[config(default = 32)]
    pub text_embed_dim: usize,
    #[config(default = 32)]
    pub time_embed_dim: usize,
    #[config(default = false)]
    pub use_mid_attn: bool,
    #[config(default = 1)]
    pub resnet_blocks_per_level: usize,  // 1 or 2 ResNet blocks per down/up level
    pub channels: Vec<usize>,
}

impl Default for UNetConfig {
    fn default() -> Self {
        Self::new(vec![16, 32, 64])
    }
}

impl UNetConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> UNet<B> {
        let time_emb_dim = self.time_embed_dim * 4;

        // Text encoder
        let text_embedding = EmbeddingConfig::new(self.vocab_size, self.text_embed_dim).init(device);
        let text_encoder = LinearConfig::new(self.text_embed_dim, self.text_embed_dim).init(device);

        // Time embedding
        let time_embedding = TimeEmbedding::new(self.time_embed_dim, device);

        // Initial conv
        let conv_in = Conv2dConfig::new([IMAGE_CHANNELS, self.channels[0]], [3, 3])
            .with_padding(burn::nn::PaddingConfig2d::Explicit(1, 1))
            .init(device);

        // // Down blocks (64x64 -> 32x32 -> 16x16 -> 8x8)
        let down1 = DownBlock::new(
            self.channels[0],
            self.channels[0],
            time_emb_dim,
            self.text_embed_dim,
            false, // use_attn
            true,
            self.resnet_blocks_per_level,
            device,
        );
        let down2 = DownBlock::new(
            self.channels[0],
            self.channels[1],
            time_emb_dim,
            self.text_embed_dim,
            false, // use_attn
            true,
            self.resnet_blocks_per_level,
            device,
        );
        let down3 = DownBlock::new(
            self.channels[1],
            self.channels[2],
            time_emb_dim,
            self.text_embed_dim,
            false, // use_attn
            false,
            self.resnet_blocks_per_level,
            device,
        );

        // Bottleneck at 16x16
        let mid_block1 = ResNetBlock::new(self.channels[2], self.channels[2], time_emb_dim, device);
        let mid_attn = if self.use_mid_attn {
            Some(AttentionBlock::new(self.channels[2], self.text_embed_dim, 4, device))
        } else {
            None
        };
        let mid_block2 = ResNetBlock::new(self.channels[2], self.channels[2], time_emb_dim, device);

        // Up blocks
        let up1 = UpBlock::new(
            self.channels[2],
            self.channels[1],
            time_emb_dim,
            self.text_embed_dim,
            false, // use_attn
            true,
            self.resnet_blocks_per_level,
            device,
        );
        let up2 = UpBlock::new(
            self.channels[1],
            self.channels[0],
            time_emb_dim,
            self.text_embed_dim,
            false, // use_attn
            true,
            self.resnet_blocks_per_level,
            device,
        );
        let up3 = UpBlock::new(
            self.channels[0],
            self.channels[0],
            time_emb_dim,
            self.text_embed_dim,
            false, // use_attn
            false,
            self.resnet_blocks_per_level,
            device,
        );

        // Output
        let norm_out = GroupNormConfig::new(8, self.channels[0]).init(device);
        let conv_out = Conv2dConfig::new([self.channels[0], IMAGE_CHANNELS], [3, 3])
            .with_padding(burn::nn::PaddingConfig2d::Explicit(1, 1))
            .init(device);

        UNet {
            text_embedding,
            text_encoder,
            time_embedding,
            conv_in,
            down1,
            down2,
            down3,
            mid_block1,
            mid_attn,
            mid_block2,
            up1,
            up2,
            up3,
            norm_out,
            conv_out,
            activation: Gelu::new(),
        }
    }
}

impl<B: Backend> UNet<B> {
    pub fn forward(
        &self,
        noisy_images: Tensor<B, 4>,
        timesteps: Tensor<B, 1>,
        text_tokens: Tensor<B, 2, Int>,
    ) -> Tensor<B, 4> {
        // Encode text
        let text_emb = self.text_embedding.forward(text_tokens);
        let text_context = self.text_encoder.forward(text_emb); // [batch, seq_len, text_embed_dim]

        // Time embedding
        let time_emb = self.time_embedding.forward(timesteps);

        // Initial conv
        let mut h = self.conv_in.forward(noisy_images);

        // Encoder
        let (h1, skip1) = self.down1.forward(h, time_emb.clone(), text_context.clone());
        let (h2, skip2) = self.down2.forward(h1, time_emb.clone(), text_context.clone());
        let (h3, skip3) = self.down3.forward(h2, time_emb.clone(), text_context.clone());

        // Bottleneck
        let mut h = self.mid_block1.forward(h3, time_emb.clone());
        if let Some(ref attn) = self.mid_attn {
            h = attn.forward(h, text_context.clone());
        }
        h = self.mid_block2.forward(h, time_emb.clone());

        // Decoder
        h = self.up1.forward(h, skip3, time_emb.clone(), text_context.clone());
        h = self.up2.forward(h, skip2, time_emb.clone(), text_context.clone());
        h = self.up3.forward(h, skip1, time_emb, text_context);

        // Output
        h = self.norm_out.forward(h);
        h = self.activation.forward(h);
        self.conv_out.forward(h)
    }

    pub fn forward_step(&self, batch: DiffusionBatch<B>) -> RegressionOutput<B> {
        // Predict noise
        let predicted_noise = self.forward(
            batch.noisy_images.clone(),
            batch.timesteps.clone(),
            batch.text_tokens.clone(),
        );

        // Create dummy tensor with correct batch size and shape to bypass model
        // let [batch_size, channels, height, width] = batch.noisy_images.dims();
        // let device = batch.noisy_images.device();
        // let predicted_noise: Tensor<B, 4> = Tensor::zeros([batch_size, channels, height, width], &device);

        // MSE loss between predicted noise and actual noise
        let loss = MseLoss::new().forward(
            predicted_noise.clone(),
            batch.noise.clone(),
            burn::nn::loss::Reduction::Mean,
        );

        // Flatten for RegressionOutput (expects 2D tensors)
        let [batch_size, channels, height, width] = predicted_noise.dims();
        let output_flat = predicted_noise.clone().reshape([batch_size, channels * height * width]);
        let targets_flat = batch.noise.reshape([batch_size, channels * height * width]);

        RegressionOutput::new(loss, output_flat, targets_flat)
    }
}

impl<B: AutodiffBackend> TrainStep<DiffusionBatch<B>, RegressionOutput<B>> for UNet<B> {
    fn step(&self, batch: DiffusionBatch<B>) -> TrainOutput<RegressionOutput<B>> {
        let output = self.forward_step(batch);
        TrainOutput::new(self, output.loss.backward(), output)
    }
}

impl<B: Backend> ValidStep<DiffusionBatch<B>, RegressionOutput<B>> for UNet<B> {
    fn step(&self, batch: DiffusionBatch<B>) -> RegressionOutput<B> {
        self.forward_step(batch)
    }
}
