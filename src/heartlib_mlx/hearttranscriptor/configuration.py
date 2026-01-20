"""Configuration for HeartTranscriptor model."""

from dataclasses import dataclass
from typing import Union
from pathlib import Path
import json


@dataclass
class HeartTranscriptorConfig:
    """Configuration for HeartTranscriptor lyrics recognition model.

    HeartTranscriptor is a Whisper-based model fine-tuned for
    music lyrics transcription.

    Attributes:
        model_type: Model type identifier.
        whisper_model: Base Whisper model name.
        d_model: Model dimension.
        encoder_layers: Number of encoder layers.
        decoder_layers: Number of decoder layers.
        encoder_attention_heads: Number of encoder attention heads.
        decoder_attention_heads: Number of decoder attention heads.
        encoder_ffn_dim: Encoder FFN dimension.
        decoder_ffn_dim: Decoder FFN dimension.
        vocab_size: Size of the vocabulary.
        max_source_positions: Maximum source sequence length.
        max_target_positions: Maximum target sequence length.
        sample_rate: Audio sample rate.
        n_mels: Number of mel filterbanks.
        n_fft: FFT size.
        hop_length: Hop length for spectrogram.
    """

    model_type: str = "hearttranscriptor"
    whisper_model: str = "large-v3"
    d_model: int = 1280
    encoder_layers: int = 32
    decoder_layers: int = 32
    encoder_attention_heads: int = 20
    decoder_attention_heads: int = 20
    encoder_ffn_dim: int = 5120
    decoder_ffn_dim: int = 5120
    vocab_size: int = 51866
    max_source_positions: int = 1500
    max_target_positions: int = 448
    sample_rate: int = 16000
    n_mels: int = 128
    n_fft: int = 400
    hop_length: int = 160

    @classmethod
    def from_pretrained(cls, path: str) -> "HeartTranscriptorConfig":
        """Load configuration from a pretrained model directory.

        Args:
            path: Path to the model directory.

        Returns:
            HeartTranscriptorConfig instance.
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

        config_dict = {k: getattr(self, k) for k in self.__dataclass_fields__}

        with open(save_path / "config.json", "w") as f:
            json.dump(config_dict, f, indent=2)

    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {k: getattr(self, k) for k in self.__dataclass_fields__}
