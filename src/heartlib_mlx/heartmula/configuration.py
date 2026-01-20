"""Configuration for HeartMuLa model."""

from dataclasses import dataclass, field
from typing import Dict, Optional, Union
from pathlib import Path
import json


# Predefined model configurations
LLAMA_CONFIGS: Dict[str, Dict] = {
    "llama-3B": {
        "dim": 3072,
        "n_heads": 24,
        "n_kv_heads": 8,
        "n_layers": 28,
        "hidden_dim": 8192,
        "rope_base": 500000.0,
        "norm_eps": 1e-5,
    },
    "llama-7B": {
        "dim": 4096,
        "n_heads": 32,
        "n_kv_heads": 8,
        "n_layers": 32,
        "hidden_dim": 11008,
        "rope_base": 500000.0,
        "norm_eps": 1e-5,
    },
    "llama-300M": {
        "dim": 3072,
        "n_heads": 8,
        "n_kv_heads": 4,  # GQA with 4 kv heads
        "n_layers": 3,
        "hidden_dim": 8192,
        "rope_base": 500000.0,
        "norm_eps": 1e-5,
    },
    "llama-400M": {
        "dim": 4096,
        "n_heads": 8,
        "n_kv_heads": 8,
        "n_layers": 3,
        "hidden_dim": 11008,
        "rope_base": 500000.0,
        "norm_eps": 1e-5,
    },
}


@dataclass
class HeartMuLaConfig:
    """Configuration for HeartMuLa music language model.

    HeartMuLa is a hierarchical music generation model consisting of:
    1. A backbone transformer (LLaMA-3B by default) for sequence modeling
    2. A decoder transformer (LLaMA-300M by default) for multi-codebook generation

    Attributes:
        model_type: Model type identifier.
        backbone_flavor: Backbone model architecture ("llama-3B" or "llama-7B").
        decoder_flavor: Decoder model architecture ("llama-300M" or "llama-400M").
        text_vocab_size: Size of text vocabulary.
        audio_vocab_size: Size of audio vocabulary per codebook.
        audio_num_codebooks: Number of audio codebooks (RVQ levels).
        muq_dim: Multi-unit quantization embedding dimension.
        max_seq_len: Maximum sequence length.
        tie_word_embeddings: Whether to tie input/output embeddings.
    """

    model_type: str = "heartmula"
    backbone_flavor: str = "llama-3B"
    decoder_flavor: str = "llama-300M"
    text_vocab_size: int = 128256
    audio_vocab_size: int = 8197  # 8192 codes + 5 special tokens
    audio_num_codebooks: int = 8
    muq_dim: int = 512
    max_seq_len: int = 8192
    tie_word_embeddings: bool = False

    @property
    def backbone_config(self) -> Dict:
        """Get backbone transformer configuration."""
        return LLAMA_CONFIGS[self.backbone_flavor]

    @property
    def decoder_config(self) -> Dict:
        """Get decoder transformer configuration."""
        return LLAMA_CONFIGS[self.decoder_flavor]

    @property
    def backbone_dim(self) -> int:
        """Get backbone hidden dimension."""
        return self.backbone_config["dim"]

    @property
    def decoder_dim(self) -> int:
        """Get decoder hidden dimension."""
        return self.decoder_config["dim"]

    @classmethod
    def from_pretrained(cls, path: str) -> "HeartMuLaConfig":
        """Load configuration from a pretrained model directory.

        Args:
            path: Path to the model directory containing config.json.

        Returns:
            HeartMuLaConfig instance.
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
        """Convert configuration to dictionary.

        Returns:
            Configuration as dictionary.
        """
        return {k: getattr(self, k) for k in self.__dataclass_fields__}
