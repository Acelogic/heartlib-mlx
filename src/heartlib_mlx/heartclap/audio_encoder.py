"""Audio encoder for HeartCLAP."""

from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from heartlib_mlx.nn.transformer import LlamaTransformerBlock, RMSNorm


class PatchEmbed(nn.Module):
    """Audio patch embedding.

    Converts audio spectrograms into patch embeddings.

    Args:
        patch_size: Size of each patch (time, freq).
        in_channels: Number of input channels.
        embed_dim: Embedding dimension.
    """

    def __init__(
        self,
        patch_size: tuple = (16, 16),
        in_channels: int = 1,
        embed_dim: int = 768,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # Convolution for patch embedding
        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def __call__(self, x: mx.array) -> mx.array:
        """Embed patches.

        Args:
            x: Input spectrogram of shape (batch, channels, time, freq).

        Returns:
            Patch embeddings of shape (batch, num_patches, embed_dim).
        """
        # Apply convolution
        x = self.proj(x)

        # Flatten spatial dimensions
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1, self.embed_dim)

        return x


class AudioTransformerEncoder(nn.Module):
    """Transformer encoder for audio.

    Based on MuQ-MuLan architecture for robust audio representation.

    Args:
        dim: Model dimension.
        n_heads: Number of attention heads.
        n_layers: Number of transformer layers.
        hidden_dim: MLP hidden dimension.
        max_seq_len: Maximum sequence length.
        norm_eps: Epsilon for normalization.
    """

    def __init__(
        self,
        dim: int = 768,
        n_heads: int = 12,
        n_layers: int = 12,
        hidden_dim: int = 3072,
        max_seq_len: int = 1024,
        norm_eps: float = 1e-6,
    ):
        super().__init__()

        self.dim = dim
        self.n_layers = n_layers

        # Learnable CLS token
        self.cls_token = mx.zeros((1, 1, dim))

        # Positional embedding
        self.pos_embed = mx.zeros((1, max_seq_len + 1, dim))

        # Transformer layers
        self.layers = [
            LlamaTransformerBlock(
                dim=dim,
                n_heads=n_heads,
                hidden_dim=hidden_dim,
                max_seq_len=max_seq_len,
                norm_eps=norm_eps,
            )
            for _ in range(n_layers)
        ]

        # Output norm
        self.norm = RMSNorm(dim, eps=norm_eps)

    def __call__(self, x: mx.array) -> mx.array:
        """Encode audio patches.

        Args:
            x: Patch embeddings of shape (batch, num_patches, dim).

        Returns:
            CLS token representation of shape (batch, dim).
        """
        batch_size = x.shape[0]

        # Add CLS token
        cls_tokens = mx.broadcast_to(self.cls_token, (batch_size, 1, self.dim))
        x = mx.concatenate([cls_tokens, x], axis=1)

        # Add positional embedding
        x = x + self.pos_embed[:, : x.shape[1], :]

        # Process through transformer
        for layer in self.layers:
            x, _ = layer(x)

        # Normalize
        x = self.norm(x)

        # Return CLS token
        return x[:, 0, :]


class AudioEncoder(nn.Module):
    """Complete audio encoder for HeartCLAP.

    Converts raw audio to embeddings in the shared space.

    Args:
        sample_rate: Audio sample rate.
        n_mels: Number of mel filterbanks.
        embed_dim: Transformer embedding dimension.
        output_dim: Output embedding dimension.
        n_heads: Number of attention heads.
        n_layers: Number of transformer layers.
    """

    def __init__(
        self,
        sample_rate: int = 48000,
        n_mels: int = 128,
        embed_dim: int = 768,
        output_dim: int = 512,
        n_heads: int = 12,
        n_layers: int = 12,
    ):
        super().__init__()

        self.sample_rate = sample_rate
        self.n_mels = n_mels

        # Mel spectrogram parameters (stored for conversion)
        self.hop_length = 480  # 10ms at 48kHz
        self.n_fft = 2048

        # Patch embedding
        self.patch_embed = PatchEmbed(
            patch_size=(16, 16),
            in_channels=1,
            embed_dim=embed_dim,
        )

        # Transformer encoder
        self.encoder = AudioTransformerEncoder(
            dim=embed_dim,
            n_heads=n_heads,
            n_layers=n_layers,
        )

        # Projection to output dimension
        self.proj = nn.Linear(embed_dim, output_dim)

    def __call__(self, audio: mx.array) -> mx.array:
        """Encode audio to embedding.

        Args:
            audio: Raw audio waveform of shape (batch, samples).

        Returns:
            Audio embedding of shape (batch, output_dim).
        """
        # Note: In practice, mel spectrogram computation would be done
        # in preprocessing. Here we assume input is already a spectrogram
        # or we'd need to add MLX-compatible spectrogram computation.

        # If input is raw audio (2D), assume it needs preprocessing
        if audio.ndim == 2:
            # Placeholder: in real implementation, compute mel spectrogram
            # For now, reshape assuming it's already processed
            batch_size = audio.shape[0]
            # Assume audio is (batch, time, freq) after preprocessing
            audio = audio[:, None, :, :]  # Add channel dim

        # Patch embedding
        x = self.patch_embed(audio)

        # Transformer encoding
        x = self.encoder(x)

        # Project to output dimension
        x = self.proj(x)

        # L2 normalize
        x = x / (mx.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)

        return x
