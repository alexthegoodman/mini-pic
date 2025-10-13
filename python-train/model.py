import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List

IMAGE_CHANNELS = 3


# ============================================================================
# Time Embedding (Sinusoidal Position Embeddings)
# ============================================================================

class TimeEmbedding(nn.Module):
    """Sinusoidal time embedding with MLP"""

    def __init__(self, time_embed_dim: int):
        super().__init__()
        self.time_embed_dim = time_embed_dim
        self.mlp = nn.Linear(time_embed_dim, time_embed_dim * 4)
        self.activation = nn.GELU()

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timesteps: [batch_size] tensor of timestep values
        Returns:
            [batch_size, time_embed_dim * 4] embedded timesteps
        """
        batch_size = timesteps.shape[0]
        device = timesteps.device

        # Generate sinusoidal embeddings
        half_dim = self.time_embed_dim // 2
        emb_scale = math.log(10000.0) / (half_dim - 1)

        frequencies = torch.exp(torch.arange(half_dim, device=device, dtype=torch.float32) * -emb_scale)
        frequencies = frequencies.unsqueeze(0).repeat(batch_size, 1)  # [batch, half_dim]

        timesteps_expanded = timesteps.unsqueeze(1).float()  # [batch, 1]
        args = timesteps_expanded * frequencies  # [batch, half_dim]

        # Concatenate sin and cos embeddings
        sin_emb = torch.sin(args)
        cos_emb = torch.cos(args)
        emb = torch.cat([sin_emb, cos_emb], dim=1)  # [batch, time_embed_dim]

        # Pass through MLP
        emb = self.mlp(emb)
        emb = self.activation(emb)

        return emb


# ============================================================================
# Attention Modules
# ============================================================================

class SelfAttention(nn.Module):
    """Multi-head self-attention using PyTorch's nn.MultiheadAttention"""

    def __init__(self, channels: int, n_heads: int = 4):
        super().__init__()
        assert channels % n_heads == 0, "channels must be divisible by n_heads"

        self.channels = channels

        # PyTorch's MultiheadAttention (batch_first=True for easier usage)
        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=n_heads,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, channels]
        Returns:
            [batch, seq_len, channels]
        """
        # For self-attention: query=key=value=x
        out, _ = self.attn(x, x, x)
        return out


class CrossAttention(nn.Module):
    """Multi-head cross-attention using PyTorch's nn.MultiheadAttention"""

    def __init__(self, channels: int, context_dim: int, n_heads: int = 4):
        super().__init__()
        assert channels % n_heads == 0, "channels must be divisible by n_heads"

        self.channels = channels
        self.context_dim = context_dim

        # Project context to match channels dimension if different
        self.context_proj = nn.Linear(context_dim, channels) if context_dim != channels else None

        # PyTorch's MultiheadAttention (batch_first=True for easier usage)
        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=n_heads,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, channels] - query
            context: [batch, context_len, context_dim] - key and value
        Returns:
            [batch, seq_len, channels]
        """
        # Project context if needed
        if self.context_proj is not None:
            context = self.context_proj(context)

        # MultiheadAttention expects (query, key, value)
        # For cross-attention: query=x, key=context, value=context
        out, _ = self.attn(x, context, context)

        return out


# ============================================================================
# ResNet Block with Time Conditioning
# ============================================================================

class ResNetBlock(nn.Module):
    """ResNet block with time embedding conditioning"""

    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        self.norm1 = nn.GroupNorm(8, in_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)

        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        self.activation = nn.GELU()

        self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, in_channels, height, width]
            time_emb: [batch, time_emb_dim]
        Returns:
            [batch, out_channels, height, width]
        """
        residual = self.residual_conv(x) if self.residual_conv is not None else x

        # First conv block
        h = self.norm1(x)
        h = self.activation(h)
        h = self.conv1(h)

        # Add time embedding
        time_out = self.activation(self.time_mlp(time_emb))
        time_out = time_out[:, :, None, None]  # [batch, channels, 1, 1]
        h = h + time_out

        # Second conv block
        h = self.norm2(h)
        h = self.activation(h)
        h = self.conv2(h)

        # Residual connection
        return h + residual


# ============================================================================
# Attention Block (combines spatial self-attention + cross-attention)
# ============================================================================

class AttentionBlock(nn.Module):
    """Spatial attention with self-attention and cross-attention to text context"""

    def __init__(self, channels: int, context_dim: int, n_heads: int = 4):
        super().__init__()

        self.norm1 = nn.GroupNorm(8, channels)
        self.self_attn = SelfAttention(channels, n_heads)

        self.norm2 = nn.GroupNorm(8, channels)
        self.cross_attn = CrossAttention(channels, context_dim, n_heads)

        self.norm3 = nn.GroupNorm(8, channels)
        self.ffn = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, channels, height, width]
            context: [batch, seq_len, context_dim]
        Returns:
            [batch, channels, height, width]
        """
        batch, channels, height, width = x.shape

        # Reshape to sequence format [batch, seq_len, channels]
        x_seq = x.view(batch, channels, height * width).transpose(1, 2)

        # Self-attention
        attn_out = self.self_attn(self._apply_norm1(x_seq, height, width))
        x_seq = x_seq + attn_out

        # Cross-attention with text context
        cross_out = self.cross_attn(self._apply_norm2(x_seq, height, width), context)
        x_seq = x_seq + cross_out

        # FFN
        ffn_out = self.ffn(self._apply_norm3(x_seq, height, width))
        x_seq = x_seq + ffn_out

        # Reshape back to image format
        return x_seq.transpose(1, 2).view(batch, channels, height, width)

    def _apply_norm1(self, x_seq: torch.Tensor, height: int, width: int) -> torch.Tensor:
        batch, seq_len, channels = x_seq.shape
        x_img = x_seq.transpose(1, 2).view(batch, channels, height, width)
        normed = self.norm1(x_img)
        return normed.view(batch, channels, seq_len).transpose(1, 2)

    def _apply_norm2(self, x_seq: torch.Tensor, height: int, width: int) -> torch.Tensor:
        batch, seq_len, channels = x_seq.shape
        x_img = x_seq.transpose(1, 2).view(batch, channels, height, width)
        normed = self.norm2(x_img)
        return normed.view(batch, channels, seq_len).transpose(1, 2)

    def _apply_norm3(self, x_seq: torch.Tensor, height: int, width: int) -> torch.Tensor:
        batch, seq_len, channels = x_seq.shape
        x_img = x_seq.transpose(1, 2).view(batch, channels, height, width)
        normed = self.norm3(x_img)
        return normed.view(batch, channels, seq_len).transpose(1, 2)


# ============================================================================
# Down/Up Blocks
# ============================================================================

class DownBlock(nn.Module):
    """Downsampling block with ResNet blocks and optional attention"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        context_dim: int,
        use_attn: bool = False,
        downsample: bool = True,
        num_resnet_blocks: int = 1,
    ):
        super().__init__()

        self.resnet1 = ResNetBlock(in_channels, out_channels, time_emb_dim)
        self.resnet2 = ResNetBlock(out_channels, out_channels, time_emb_dim) if num_resnet_blocks > 1 else None

        self.attn = AttentionBlock(out_channels, context_dim, 4) if use_attn else None

        self.downsample = nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1) if downsample else None

    def forward(
        self,
        x: torch.Tensor,
        time_emb: torch.Tensor,
        context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch, in_channels, height, width]
            time_emb: [batch, time_emb_dim]
            context: [batch, seq_len, context_dim]
        Returns:
            (downsampled, skip_connection)
        """
        h = self.resnet1(x, time_emb)

        if self.resnet2 is not None:
            h = self.resnet2(h, time_emb)

        if self.attn is not None:
            h = self.attn(h, context)

        h_skip = h

        h_out = self.downsample(h) if self.downsample is not None else h

        return h_out, h_skip


class UpBlock(nn.Module):
    """Upsampling block with ResNet blocks and optional attention"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        context_dim: int,
        use_attn: bool = False,
        upsample: bool = True,
        num_resnet_blocks: int = 1,
        skip_channels: int = None,
    ):
        super().__init__()

        # Skip channels defaults to out_channels if not specified
        if skip_channels is None:
            skip_channels = out_channels

        self.resnet1 = ResNetBlock(in_channels + skip_channels, out_channels, time_emb_dim)
        self.resnet2 = ResNetBlock(out_channels, out_channels, time_emb_dim) if num_resnet_blocks > 1 else None
        self.resnet3 = ResNetBlock(out_channels, out_channels, time_emb_dim) if num_resnet_blocks > 2 else None

        self.attn = AttentionBlock(out_channels, context_dim, 4) if use_attn else None

        self.upsample = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1) if upsample else None

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
        time_emb: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: [batch, in_channels, height, width]
            skip: [batch, out_channels, height, width] - skip connection from encoder
            time_emb: [batch, time_emb_dim]
            context: [batch, seq_len, context_dim]
        Returns:
            [batch, out_channels, height*2, width*2] if upsample else [batch, out_channels, height, width]
        """
        # Concatenate skip connection
        h = torch.cat([x, skip], dim=1)

        h = self.resnet1(h, time_emb)

        if self.resnet2 is not None:
            h = self.resnet2(h, time_emb)

        if self.resnet3 is not None:
            h = self.resnet3(h, time_emb)

        if self.attn is not None:
            h = self.attn(h, context)

        if self.upsample is not None:
            # Bilinear upsample then conv
            h = F.interpolate(h, scale_factor=2, mode='bilinear', align_corners=False)
            h = self.upsample(h)

        return h


# ============================================================================
# U-Net Model
# ============================================================================

class TextEncoder(nn.Module):
    """Transformer-based text encoder for better semantic understanding"""

    def __init__(self, vocab_size: int, text_embed_dim: int, num_layers: int = 4, num_heads: int = 4):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, text_embed_dim)

        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, 77, text_embed_dim) * 0.02)  # MAX_SEQ_LEN = 77

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=text_embed_dim,
            nhead=num_heads,
            dim_feedforward=text_embed_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Final projection
        self.proj = nn.Linear(text_embed_dim, text_embed_dim)

    def forward(self, text_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            text_tokens: [batch, seq_len]
        Returns:
            [batch, seq_len, text_embed_dim]
        """
        # Embed tokens
        x = self.embedding(text_tokens)  # [batch, seq_len, embed_dim]

        # Add positional encoding
        x = x + self.pos_encoding[:, :x.shape[1], :]

        # Transformer encoding
        x = self.transformer(x)

        # Final projection
        x = self.proj(x)

        return x


class UNet(nn.Module):
    """U-Net for text-conditioned diffusion model"""

    def __init__(
        self,
        vocab_size: int = 8192,
        text_embed_dim: int = 128,  # Increased default
        time_embed_dim: int = 32,
        use_mid_attn: bool = False,
        resnet_blocks_per_level: int = 1,
        channels: Optional[List[int]] = None,
        loss_fn: str = "mse",
        text_encoder_layers: int = 4,
    ):
        super().__init__()

        if channels is None:
            channels = [16, 32, 64]

        self.channels = channels
        self.loss_fn = loss_fn
        time_emb_dim_expanded = time_embed_dim * 4

        # Text encoder - now much stronger
        self.text_encoder = TextEncoder(vocab_size, text_embed_dim, num_layers=text_encoder_layers)

        # Time embedding
        self.time_embedding = TimeEmbedding(time_embed_dim)

        # Initial conv
        self.conv_in = nn.Conv2d(IMAGE_CHANNELS, channels[0], kernel_size=3, padding=1)

        # Encoder (down blocks) - Enable attention at lower resolutions for better text conditioning
        self.down1 = DownBlock(channels[0], channels[0], time_emb_dim_expanded, text_embed_dim,
                               use_attn=False, downsample=True, num_resnet_blocks=resnet_blocks_per_level)
        self.down2 = DownBlock(channels[0], channels[1], time_emb_dim_expanded, text_embed_dim,
                               use_attn=True, downsample=True, num_resnet_blocks=resnet_blocks_per_level)  # Enable attn
        self.down3 = DownBlock(channels[1], channels[2], time_emb_dim_expanded, text_embed_dim,
                               use_attn=True, downsample=False, num_resnet_blocks=resnet_blocks_per_level)  # Enable attn

        # Bottleneck
        self.mid_block1 = ResNetBlock(channels[2], channels[2], time_emb_dim_expanded)
        self.mid_attn = AttentionBlock(channels[2], text_embed_dim, 4) if use_mid_attn else None
        self.mid_block2 = ResNetBlock(channels[2], channels[2], time_emb_dim_expanded)

        # Decoder (up blocks) - Enable attention to match encoder
        # Note: skip connections come from corresponding down blocks
        # up1 receives: bottleneck(64) + skip3(64), up2 receives: up1(32) + skip2(32), up3 receives: up2(16) + skip1(16)
        self.up1 = UpBlock(channels[2], channels[1], time_emb_dim_expanded, text_embed_dim,
                          use_attn=True, upsample=True, num_resnet_blocks=resnet_blocks_per_level,  # Enable attn
                          skip_channels=channels[2])
        self.up2 = UpBlock(channels[1], channels[0], time_emb_dim_expanded, text_embed_dim,
                          use_attn=True, upsample=True, num_resnet_blocks=resnet_blocks_per_level,  # Enable attn
                          skip_channels=channels[1])
        self.up3 = UpBlock(channels[0], channels[0], time_emb_dim_expanded, text_embed_dim,
                          use_attn=False, upsample=False, num_resnet_blocks=resnet_blocks_per_level,  # Keep off at highest res
                          skip_channels=channels[0])

        # Output
        self.norm_out = nn.GroupNorm(8, channels[0])
        self.activation = nn.GELU()
        self.conv_out = nn.Conv2d(channels[0], IMAGE_CHANNELS, kernel_size=3, padding=1)

    def forward(
        self,
        noisy_images: torch.Tensor,
        timesteps: torch.Tensor,
        text_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            noisy_images: [batch, 3, 64, 64]
            timesteps: [batch]
            text_tokens: [batch, seq_len]
        Returns:
            predicted_noise: [batch, 3, 64, 64]
        """
        # Encode text with improved transformer encoder
        text_context = self.text_encoder(text_tokens)  # [batch, seq_len, text_embed_dim]

        # Time embedding
        time_emb = self.time_embedding(timesteps)  # [batch, time_emb_dim * 4]

        # Initial conv
        h = self.conv_in(noisy_images)

        # Encoder
        h1, skip1 = self.down1(h, time_emb, text_context)
        h2, skip2 = self.down2(h1, time_emb, text_context)
        h3, skip3 = self.down3(h2, time_emb, text_context)

        # Bottleneck
        h = self.mid_block1(h3, time_emb)
        if self.mid_attn is not None:
            h = self.mid_attn(h, text_context)
        h = self.mid_block2(h, time_emb)

        # Decoder
        h = self.up1(h, skip3, time_emb, text_context)
        h = self.up2(h, skip2, time_emb, text_context)
        h = self.up3(h, skip1, time_emb, text_context)

        # Output
        h = self.norm_out(h)
        h = self.activation(h)
        predicted_noise = self.conv_out(h)

        return predicted_noise

    def compute_loss(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute loss between predicted and actual noise using the configured loss function

        Args:
            batch: Dictionary with keys 'noisy_images', 'timesteps', 'text_tokens', 'noise'
        Returns:
            (loss, predicted_noise)
        """
        predicted_noise = self.forward(
            batch['noisy_images'],
            batch['timesteps'],
            batch['text_tokens'],
        )

        # Select loss function
        if self.loss_fn == "mse":
            loss = F.mse_loss(predicted_noise, batch['noise'], reduction='sum') # testing sum to see if helps reveal actual loss (mean is too small)
            # loss = F.mse_loss(predicted_noise, batch['noise'])
        elif self.loss_fn == "l1":
            # loss = F.l1_loss(predicted_noise, batch['noise'])
            loss = F.l1_loss(predicted_noise, batch['noise'], reduction='sum')
        elif self.loss_fn == "smooth_l1":
            loss = F.smooth_l1_loss(predicted_noise, batch['noise'])
        elif self.loss_fn == "huber":
            loss = F.huber_loss(predicted_noise, batch['noise'])
        else:
            raise ValueError(f"Unknown loss function: {self.loss_fn}")

        return loss, predicted_noise
