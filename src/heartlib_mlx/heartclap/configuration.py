"""Configuration for HeartCLAP model."""

from dataclasses import dataclass
from typing import Union
from pathlib import Path
import json


@dataclass
class HeartCLAPConfig:
    """Configuration for HeartCLAP audio-text alignment model.

    HeartCLAP learns a shared embedding space for audio and text,
    enabling:
    - Music tagging from audio
    - Audio retrieval from text descriptions
    - Similarity computation between audio and text

    Attributes:
        model_type: Model type identifier.
        audio_encoder_type: Type of audio encoder ("muq_mulan").
        text_encoder_type: Type of text encoder ("bert").
        embedding_dim: Dimension of the shared embedding space.
        audio_dim: Dimension of audio encoder output.
        text_dim: Dimension of text encoder output.
        sample_rate: Audio sample rate.
        max_audio_len: Maximum audio length in samples.
        max_text_len: Maximum text sequence length.
        projection_dim: Hidden dimension in projection heads.
        temperature: Temperature for contrastive loss.
    """

    model_type: str = "heartclap"
    audio_encoder_type: str = "muq_mulan"
    text_encoder_type: str = "bert"
    embedding_dim: int = 512
    audio_dim: int = 768
    text_dim: int = 768
    sample_rate: int = 48000
    max_audio_len: int = 480000  # 10 seconds at 48kHz
    max_text_len: int = 77
    projection_dim: int = 512
    temperature: float = 0.07

    @classmethod
    def from_pretrained(cls, path: str) -> "HeartCLAPConfig":
        """Load configuration from a pretrained model directory.

        Args:
            path: Path to the model directory.

        Returns:
            HeartCLAPConfig instance.
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
