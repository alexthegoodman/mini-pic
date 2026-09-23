use crate::dataset::{DiffusionBatch, IMAGE_CHANNELS, MAX_SEQ_LEN};
use burn::{
    config::Config,
    module::Module,
    nn::{
        conv::{Conv2d, Conv2dConfig},
        loss::MseLoss,
        transformer::{TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput},
        Embedding, EmbeddingConfig, Gelu, GroupNorm, GroupNormConfig, Linear, LinearConfig,
        PositionalEncoding, PositionalEncodingConfig,
    },
    tensor::{
        activation::softmax,
        backend::{AutodiffBackend, Backend},
        Bool, Int, Tensor,
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

        // norm1 runs on x before conv1 changes the channel count, so it must
        // be sized to in_channels, not out_channels - norm2 runs after conv1
        // and is correctly sized to out_channels already.
        let norm1 = GroupNormConfig::new(8, in_channels).init(device);
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
    resnets: Vec<ResNetBlock<B>>,
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
        assert!(num_resnet_blocks >= 1, "num_resnet_blocks must be at least 1");
        let mut resnets = Vec::with_capacity(num_resnet_blocks);
        resnets.push(ResNetBlock::new(in_channels, out_channels, time_emb_dim, device));
        for _ in 1..num_resnet_blocks {
            resnets.push(ResNetBlock::new(out_channels, out_channels, time_emb_dim, device));
        }

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
            resnets,
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
        let mut h = x;
        for resnet in &self.resnets {
            h = resnet.forward(h, time_emb.clone());
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
    resnets: Vec<ResNetBlock<B>>,
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
        assert!(num_resnet_blocks >= 1, "num_resnet_blocks must be at least 1");
        // Only the first block absorbs the concatenated skip connection. x
        // always arrives at in_channels width (it's the previous stage's
        // output, chained through), and the matching encoder-level skip
        // tensor is also always in_channels wide in this UNet's symmetric
        // wiring (e.g. skip3 is channels[2] wide, matching up1's
        // in_channels) - so the concatenated width is 2 * in_channels, not
        // in_channels + out_channels.
        let mut resnets = Vec::with_capacity(num_resnet_blocks);
        resnets.push(ResNetBlock::new(in_channels * 2, out_channels, time_emb_dim, device));
        for _ in 1..num_resnet_blocks {
            resnets.push(ResNetBlock::new(out_channels, out_channels, time_emb_dim, device));
        }

        let attn = if use_attn {
            Some(AttentionBlock::new(out_channels, context_dim, 4, device))
        } else {
            None
        };

        let upsample = if upsample {
            // Refines features after forward()'s manual nearest-neighbor 2x
            // repeat (which does the actual, exact doubling). This conv must
            // be spatial-size-preserving: a 4x4 kernel at stride 1 with
            // padding 1 shrinks the output by 1px (input + 2*1 - 4 + 1 =
            // input - 1), which desyncs it from the skip connection it's
            // concatenated with one level up. 3x3/stride1/pad1 preserves
            // size exactly, same as every conv in ResNetBlock.
            Some(
                Conv2dConfig::new([out_channels, out_channels], [3, 3])
                    .with_stride([1, 1])
                    .with_padding(burn::nn::PaddingConfig2d::Explicit(1, 1))
                    .init(device),
            )
        } else {
            None
        };

        Self {
            resnets,
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
        let mut h = Tensor::cat(vec![x, skip], 1);

        for resnet in &self.resnets {
            h = resnet.forward(h, time_emb.clone());
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
// Text Encoder (Transformer)
// ============================================================================

/// Token embedding + sinusoidal positional encoding + an N-layer self-attention
/// transformer encoder + a final projection - the standard text-tower shape
/// (comparable to python-train's TextEncoder, which uses PyTorch's
/// nn.TransformerEncoder over a *learned* positional embedding; this uses
/// burn's own sinusoidal PositionalEncoding instead of adding a second learned
/// parameter tensor, which is the more common Transformer-original choice and
/// needs no separate init/shape bookkeeping).
#[derive(Module, Debug)]
pub struct TextEncoder<B: Backend> {
    embedding: Embedding<B>,
    pos_encoding: PositionalEncoding<B>,
    transformer: TransformerEncoder<B>,
    proj: Linear<B>,
}

impl<B: Backend> TextEncoder<B> {
    pub fn new(
        vocab_size: usize,
        text_embed_dim: usize,
        n_layers: usize,
        n_heads: usize,
        dropout: f64,
        device: &B::Device,
    ) -> Self {
        assert!(
            text_embed_dim % n_heads == 0,
            "text_embed_dim must be divisible by text_encoder_heads"
        );

        let embedding = EmbeddingConfig::new(vocab_size, text_embed_dim).init(device);
        let pos_encoding = PositionalEncodingConfig::new(text_embed_dim)
            .with_max_sequence_size(MAX_SEQ_LEN)
            .init(device);
        // dim_feedforward = 4x d_model is the "Attention Is All You Need" default,
        // matching python-train's TextEncoder (dim_feedforward=text_embed_dim*4).
        let transformer = TransformerEncoderConfig::new(text_embed_dim, text_embed_dim * 4, n_heads, n_layers)
            .with_dropout(dropout)
            .init(device);
        let proj = LinearConfig::new(text_embed_dim, text_embed_dim).init(device);

        Self {
            embedding,
            pos_encoding,
            transformer,
            proj,
        }
    }

    /// Args:
    /// - text_tokens: [batch, seq_len]
    /// - mask_pad: True at padding positions (to exclude from attention); None attends over every position.
    /// Returns: [batch, seq_len, text_embed_dim]
    pub fn forward(
        &self,
        text_tokens: Tensor<B, 2, Int>,
        mask_pad: Option<Tensor<B, 2, Bool>>,
    ) -> Tensor<B, 3> {
        let x = self.embedding.forward(text_tokens);
        let x = self.pos_encoding.forward(x);

        let mut input = TransformerEncoderInput::new(x);
        if let Some(mask_pad) = mask_pad {
            input = input.mask_pad(mask_pad);
        }
        let x = self.transformer.forward(input);

        self.proj.forward(x)
    }
}

// ============================================================================
// U-Net Model
// ============================================================================

#[derive(Module, Debug)]
pub struct UNet<B: Backend> {
    // Text encoder
    text_encoder: TextEncoder<B>,

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
    #[config(default = 4096)]
    pub vocab_size: usize,
    // Unswept here - python-train's TextEncoder found 64 helped over its own
    // 32 default. Both sides now use a real multi-layer Transformer encoder,
    // so that result should transfer better than it used to when this was
    // Embedding+Linear only - worth trying once training here is up and running.
    // #[config(default = 32)]
    #[config(default = 128)]
    pub text_embed_dim: usize,
    // Transformer depth/width for the text encoder. python-train's current
    // active config (see its train.py) uses 2 layers at text_embed_dim=64;
    // 4 is this side's unswept starting point, matching its own class default.
    #[config(default = 4)]
    pub text_encoder_layers: usize,
    #[config(default = 4)]
    pub text_encoder_heads: usize,
    #[config(default = 0.1)]
    pub text_encoder_dropout: f64,
    // #[config(default = 32)]
    #[config(default = 64)]
    pub time_embed_dim: usize,
    #[config(default = false)]
    pub use_mid_attn: bool,
    #[config(default = 8)]
    pub resnet_blocks_per_level: usize,  // ResNet blocks stacked per down/up level (>= 1, uncapped)
    pub channels: Vec<usize>,
}

// No Default impl here on purpose - it duplicated training.rs::run()'s
// channel width and nothing ever called UNetConfig::default(); every real
// call site constructs UNetConfig::new(...) explicitly. channels has no
// #[config(default = ...)] either, since burn's Config derive only accepts
// literal defaults and this is a Vec - it's a required constructor argument
// everywhere, which is the point: there is exactly one place (run(), in
// training.rs) where a new run's channel width should be decided.

impl UNetConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> UNet<B> {
        let time_emb_dim = self.time_embed_dim * 4;

        // Text encoder
        let text_encoder = TextEncoder::new(
            self.vocab_size,
            self.text_embed_dim,
            self.text_encoder_layers,
            self.text_encoder_heads,
            self.text_encoder_dropout,
            device,
        );

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
        text_mask_pad: Option<Tensor<B, 2, Bool>>,
    ) -> Tensor<B, 4> {
        // Encode text
        let text_context = self.text_encoder.forward(text_tokens, text_mask_pad); // [batch, seq_len, text_embed_dim]

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
        // batch.text_mask is 1.0 at valid tokens / 0.0 at padding (see dataset.rs);
        // the transformer's mask_pad wants the opposite polarity (true = ignore).
        let mask_pad = batch.text_mask.clone().equal_elem(0.0);

        // Predict noise
        let predicted_noise = self.forward(
            batch.noisy_images.clone(),
            batch.timesteps.clone(),
            batch.text_tokens.clone(),
            Some(mask_pad),
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
