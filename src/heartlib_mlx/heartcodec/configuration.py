"""Configuration for HeartCodec model."""

from dataclasses import dataclass, field
from typing import List, Union
import json
from pathlib import Path


@dataclass
class HeartCodecConfig:
    """Configuration for HeartCodec neural audio codec.

    HeartCodec is a 12.5Hz neural audio codec that combines:
    1. A scalar quantization codec (SQ Codec) for audio encoding/decoding
    2. A flow matching decoder with LlamaTransformer for high-quality synthesis

    Attributes:
        # RVQ (Residual Vector Quantization) Parameters
        dim: Embedding dimension for codebooks (default: 512).
        codebook_size: Number of entries per codebook (default: 8192).
        decay: EMA decay rate for codebook updates (default: 0.9).
        commitment_weight: Weight for commitment loss (default: 1.0).
        threshold_ema_dead_code: Threshold for replacing dead codes (default: 2).
        use_cosine_sim: Use cosine similarity for codebook lookup (default: False).
        codebook_dim: Dimension of codebook entries (default: 32).
        num_quantizers: Number of RVQ codebooks (default: 8).

        # Diffusion Transformer Parameters
        attention_head_dim: Dimension per attention head (default: 64).
        in_channels: Input channels to flow matching (default: 1024).
        norm_type: Normalization type (default: "ada_norm_single").
        num_attention_heads: Number of attention heads (default: 24).
        num_layers: Number of transformer layers in main stack (default: 24).
        num_layers_2: Number of layers in second stack (default: 6).
        out_channels: Output channels from flow matching (default: 256).

        # SQ Codec Parameters
        num_bands: Number of frequency bands (default: 1).
        sample_rate: Audio sample rate in Hz (default: 48000).
        causal: Use causal convolutions (default: True).
        num_samples: Number of samples for codec (default: 2).
        downsample_factors: Downsampling factors per stage.
        downsample_kernel_sizes: Kernel sizes for downsampling.
        upsample_factors: Upsampling factors per stage.
        upsample_kernel_sizes: Kernel sizes for upsampling.
        latent_hidden_dim: Hidden dimension in latent space (default: 128).
        default_kernel_size: Default conv kernel size (default: 7).
        delay_kernel_size: Delay conv kernel size (default: 5).
        init_channel: Initial channel count (default: 64).
        res_kernel_size: Residual block kernel size (default: 7).
    """

    # Model type
    model_type: str = "heartcodec"

    # RVQ Parameters
    dim: int = 512
    codebook_size: int = 8192
    decay: float = 0.9
    commitment_weight: float = 1.0
    threshold_ema_dead_code: int = 2
    use_cosine_sim: bool = False
    codebook_dim: int = 32
    num_quantizers: int = 8

    # Diffusion Transformer Parameters
    attention_head_dim: int = 64
    in_channels: int = 1024
    norm_type: str = "ada_norm_single"
    num_attention_heads: int = 24
    num_layers: int = 24
    num_layers_2: int = 6
    out_channels: int = 128  # Must match latent_hidden_dim for scalar model

    # SQ Codec Parameters
    num_bands: int = 1
    sample_rate: int = 48000
    causal: bool = True
    num_samples: int = 2
    downsample_factors: List[int] = field(default_factory=lambda: [4, 4, 8, 8])
    downsample_kernel_sizes: List[int] = field(default_factory=lambda: [8, 8, 16, 16])
    upsample_factors: List[int] = field(default_factory=lambda: [8, 8, 4, 4])
    upsample_kernel_sizes: List[int] = field(default_factory=lambda: [16, 16, 8, 8])
    latent_hidden_dim: int = 128
    default_kernel_size: int = 7
    delay_kernel_size: int = 5
    init_channel: int = 64
    res_kernel_size: int = 7

    @classmethod
    def from_pretrained(cls, path: str) -> "HeartCodecConfig":
        """Load configuration from a pretrained model directory.

        Args:
            path: Path to the model directory containing config.json.

        Returns:
            HeartCodecConfig instance.
        """
        config_path = Path(path) / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r") as f:
            config_dict = json.load(f)

        return cls(**{k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__})

    def save_pretrained(self, path: Union[str, Path]) -> None:
        """Save configuration to a directory.

        Args:
            path: Path to save the configuration.
        """
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)

        config_dict = {
            k: getattr(self, k) for k in self.__dataclass_fields__
        }

        with open(save_path / "config.json", "w") as f:
            json.dump(config_dict, f, indent=2)

    def to_dict(self) -> dict:
        """Convert configuration to dictionary.

        Returns:
            Configuration as dictionary.
        """
        return {k: getattr(self, k) for k in self.__dataclass_fields__}

    @property
    def total_stride(self) -> int:
        """Total downsampling stride of the codec."""
        stride = 1
        for f in self.downsample_factors:
            stride *= f
        return stride

    @property
    def frame_rate(self) -> float:
        """Frame rate of the codec codes in Hz.

        HeartCodec operates at 12.5 Hz code rate:
        - Codes: 12.5 Hz
        - Flow matching latent: 25 Hz (2x codes)
        - Scalar model stride: 960
        - Scalar model num_samples: 2
        - Audio: 48000 Hz

        The formula is: sample_rate / (total_stride * num_samples * 2)
        where the extra *2 accounts for the flow matching 2x upsampling.
        """
        return self.sample_rate / (self.total_stride * self.num_samples * 2)

    @property
    def transformer_dim(self) -> int:
        """Dimension of the transformer in flow matching decoder."""
        return self.num_attention_heads * self.attention_head_dim
