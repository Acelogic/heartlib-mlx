"""Music generation pipeline for HeartMuLa."""

from typing import Optional, Union, List
from pathlib import Path
from dataclasses import dataclass
import json

import mlx.core as mx

from heartlib_mlx.heartcodec import HeartCodec, HeartCodecConfig
from heartlib_mlx.heartmula import HeartMuLa, HeartMuLaConfig


@dataclass
class HeartMuLaGenConfig:
    """Configuration for music generation pipeline.

    Attributes:
        text_bos_id: Beginning of text token ID.
        text_eos_id: End of text token ID.
        audio_bos_id: Beginning of audio token ID.
        audio_eos_id: End of audio token ID.
        audio_empty_id: Empty audio token ID.
        sample_rate: Audio sample rate.
        frame_rate: Codec frame rate.
    """

    text_bos_id: int = 128000
    text_eos_id: int = 128001
    audio_bos_id: int = 8193
    audio_eos_id: int = 8194
    audio_empty_id: int = 8195
    sample_rate: int = 48000
    frame_rate: float = 12.5

    @classmethod
    def from_pretrained(cls, path: str) -> "HeartMuLaGenConfig":
        """Load configuration from file."""
        config_path = Path(path) / "generation_config.json"
        if config_path.exists():
            with open(config_path, "r") as f:
                config_dict = json.load(f)
            return cls(**{k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__})
        return cls()


class HeartMuLaGenPipeline:
    """Music generation pipeline using HeartMuLa and HeartCodec.

    This pipeline handles:
    1. Text preprocessing (tags and lyrics)
    2. Autoregressive audio generation
    3. Audio reconstruction via HeartCodec

    Example:
        >>> pipeline = HeartMuLaGenPipeline.from_pretrained("./ckpt-mlx")
        >>> audio = pipeline(
        ...     lyrics="[Verse]\\nHello world...",
        ...     tags="pop, acoustic, female vocal",
        ...     duration=30.0,
        ... )
        >>> pipeline.save_audio(audio, "output.mp3")
    """

    def __init__(
        self,
        heartmula: HeartMuLa,
        heartcodec: HeartCodec,
        tokenizer,
        config: HeartMuLaGenConfig,
    ):
        """Initialize the pipeline.

        Args:
            heartmula: HeartMuLa model.
            heartcodec: HeartCodec model.
            tokenizer: Text tokenizer.
            config: Generation configuration.
        """
        self.heartmula = heartmula
        self.heartcodec = heartcodec
        self.tokenizer = tokenizer
        self.config = config

    @classmethod
    def from_pretrained(
        cls,
        path: Union[str, Path],
        dtype: mx.Dtype = mx.bfloat16,
    ) -> "HeartMuLaGenPipeline":
        """Load pipeline from pretrained weights.

        Args:
            path: Path to the model directory.
            dtype: Data type for model weights.

        Returns:
            HeartMuLaGenPipeline instance.
        """
        from tokenizers import Tokenizer

        path = Path(path)

        # Load HeartMuLa
        heartmula_path = path / "heartmula"
        if not heartmula_path.exists():
            heartmula_path = path  # Try root path
        heartmula = HeartMuLa.from_pretrained(heartmula_path, dtype=dtype)

        # Load HeartCodec
        heartcodec_path = path / "heartcodec"
        if not heartcodec_path.exists():
            heartcodec_path = path
        heartcodec = HeartCodec.from_pretrained(heartcodec_path, dtype=dtype)

        # Load tokenizer
        tokenizer_path = path / "tokenizer.json"
        if tokenizer_path.exists():
            tokenizer = Tokenizer.from_file(str(tokenizer_path))
        else:
            tokenizer = None

        # Load config
        config = HeartMuLaGenConfig.from_pretrained(path)

        return cls(heartmula, heartcodec, tokenizer, config)

    def preprocess(
        self,
        lyrics: Optional[str] = None,
        tags: Optional[str] = None,
        use_cfg: bool = True,
    ) -> mx.array:
        """Preprocess text inputs.

        Args:
            lyrics: Lyrics text.
            tags: Comma-separated tags.
            use_cfg: Whether to prepare for classifier-free guidance.

        Returns:
            Token IDs.
        """
        # Normalize tags
        if tags:
            tags = tags.lower().strip()
            tags = f"<tag>{tags}</tag>"

        # Normalize lyrics
        if lyrics:
            lyrics = lyrics.strip()

        # Combine
        text = ""
        if tags:
            text += tags + " "
        if lyrics:
            text += lyrics

        # Tokenize
        if self.tokenizer is not None:
            encoding = self.tokenizer.encode(text)
            token_ids = mx.array([encoding.ids])
        else:
            # Fallback: simple character encoding
            token_ids = mx.array([[ord(c) for c in text]])

        # Add BOS/EOS
        bos = mx.array([[self.config.text_bos_id]])
        eos = mx.array([[self.config.text_eos_id]])
        token_ids = mx.concatenate([bos, token_ids, eos], axis=1)

        return token_ids

    def generate(
        self,
        text_ids: mx.array,
        duration: float = 30.0,
        temperature: float = 1.0,
        top_k: int = 50,
        cfg_scale: float = 1.5,
    ) -> mx.array:
        """Generate audio codes from text.

        Args:
            text_ids: Preprocessed text token IDs.
            duration: Target duration in seconds.
            temperature: Sampling temperature.
            top_k: Top-k sampling parameter.
            cfg_scale: Classifier-free guidance scale.

        Returns:
            Generated audio codes.
        """
        batch_size = text_ids.shape[0]

        # Calculate target frames
        target_frames = int(duration * self.config.frame_rate)

        # Initialize with BOS for audio
        audio_bos = mx.full(
            (batch_size, 1, self.heartmula.num_codebooks),
            self.config.audio_bos_id,
            dtype=mx.int32,
        )

        all_codes = [audio_bos]
        backbone_cache = None

        for frame_idx in range(target_frames):
            # Get current audio codes
            if len(all_codes) > 1:
                audio_codes = mx.concatenate(all_codes, axis=1)
            else:
                audio_codes = all_codes[0]

            # Generate next frame
            new_codes, backbone_cache = self.heartmula.generate_frame(
                text_ids=text_ids,
                audio_codes=audio_codes,
                backbone_cache=backbone_cache,
                temperature=temperature,
                top_k=top_k,
                cfg_scale=cfg_scale,
            )

            # Check for EOS
            if mx.any(new_codes[:, 0] == self.config.audio_eos_id):
                break

            all_codes.append(new_codes[:, None, :])

        # Concatenate all frames (skip BOS)
        codes = mx.concatenate(all_codes[1:], axis=1)

        return codes

    def postprocess(
        self,
        codes: mx.array,
        num_steps: int = 10,
        guidance_scale: float = 1.25,
    ) -> mx.array:
        """Convert codes to audio waveform.

        Args:
            codes: Audio codes.
            num_steps: ODE integration steps for HeartCodec.
            guidance_scale: CFG scale for HeartCodec.

        Returns:
            Audio waveform.
        """
        audio = self.heartcodec.detokenize(
            codes=codes,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
        )

        return audio

    def __call__(
        self,
        lyrics: Optional[str] = None,
        tags: Optional[str] = None,
        duration: float = 30.0,
        temperature: float = 1.0,
        top_k: int = 50,
        cfg_scale: float = 1.5,
        codec_steps: int = 10,
        codec_guidance: float = 1.25,
    ) -> mx.array:
        """Generate music from text.

        Args:
            lyrics: Lyrics text.
            tags: Comma-separated music tags.
            duration: Target duration in seconds.
            temperature: Sampling temperature.
            top_k: Top-k sampling parameter.
            cfg_scale: Classifier-free guidance scale for language model.
            codec_steps: ODE steps for audio codec.
            codec_guidance: CFG scale for audio codec.

        Returns:
            Generated audio waveform.
        """
        # Preprocess
        text_ids = self.preprocess(lyrics=lyrics, tags=tags)

        # Generate codes
        codes = self.generate(
            text_ids=text_ids,
            duration=duration,
            temperature=temperature,
            top_k=top_k,
            cfg_scale=cfg_scale,
        )

        # Postprocess to audio
        audio = self.postprocess(
            codes=codes,
            num_steps=codec_steps,
            guidance_scale=codec_guidance,
        )

        return audio

    def save_audio(
        self,
        audio: mx.array,
        path: Union[str, Path],
        sample_rate: Optional[int] = None,
    ) -> None:
        """Save audio to file.

        Args:
            audio: Audio waveform.
            path: Output file path.
            sample_rate: Sample rate (uses config default if None).
        """
        import numpy as np
        import soundfile as sf

        sample_rate = sample_rate or self.config.sample_rate

        # Convert to numpy
        audio_np = np.array(audio)

        # Handle shape
        if audio_np.ndim == 3:
            audio_np = audio_np[0]  # Remove batch dim
        if audio_np.ndim == 2 and audio_np.shape[-1] == 1:
            audio_np = audio_np[:, 0]  # Remove channel dim

        # Normalize
        audio_np = audio_np / max(np.abs(audio_np).max(), 1e-8)

        # Save
        sf.write(str(path), audio_np, sample_rate)
