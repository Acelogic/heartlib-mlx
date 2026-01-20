"""Music generation pipeline for HeartMuLa."""

from typing import Optional, Union, List, Dict, Any
from pathlib import Path
from dataclasses import dataclass
import json

import mlx.core as mx
import numpy as np

from heartlib_mlx.heartcodec import HeartCodec, HeartCodecConfig
from heartlib_mlx.heartmula import HeartMuLa, HeartMuLaConfig


@dataclass
class HeartMuLaGenConfig:
    """Configuration for music generation pipeline."""

    text_bos_id: int = 128000
    text_eos_id: int = 128001
    audio_eos_id: int = 8193
    empty_id: int = 0
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

    Matches the PyTorch implementation flow exactly:
    1. Text preprocessing (tags and lyrics) into combined token format
    2. Autoregressive audio generation with CFG
    3. Audio reconstruction via HeartCodec
    """

    def __init__(
        self,
        heartmula: HeartMuLa,
        heartcodec: HeartCodec,
        tokenizer,
        config: HeartMuLaGenConfig,
    ):
        self.heartmula = heartmula
        self.heartcodec = heartcodec
        self.tokenizer = tokenizer
        self.config = config
        self._parallel_number = heartmula.num_codebooks + 1  # 8 codebooks + 1 text

    @classmethod
    def from_pretrained(
        cls,
        path: Union[str, Path],
        dtype: mx.Dtype = mx.bfloat16,
    ) -> "HeartMuLaGenPipeline":
        """Load pipeline from pretrained weights."""
        from tokenizers import Tokenizer

        path = Path(path)

        # Load HeartMuLa
        heartmula_path = path / "heartmula"
        if not heartmula_path.exists():
            heartmula_path = path
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
        cfg_scale: float = 1.5,
    ) -> Dict[str, mx.array]:
        """Preprocess text inputs into combined token format.

        Args:
            lyrics: Lyrics text.
            tags: Comma-separated tags.
            cfg_scale: CFG scale (determines batch size).

        Returns:
            Dictionary with tokens, tokens_mask, muq_embed, muq_idx, pos.
        """
        # Process tags
        if tags:
            tags = tags.lower().strip()
            if not tags.startswith("<tag>"):
                tags = f"<tag>{tags}"
            if not tags.endswith("</tag>"):
                tags = f"{tags}</tag>"

        # Tokenize tags
        if self.tokenizer is not None and tags:
            tags_ids = self.tokenizer.encode(tags).ids
        else:
            tags_ids = []

        # Add BOS/EOS to tags
        if tags_ids and tags_ids[0] != self.config.text_bos_id:
            tags_ids = [self.config.text_bos_id] + tags_ids
        if tags_ids and tags_ids[-1] != self.config.text_eos_id:
            tags_ids = tags_ids + [self.config.text_eos_id]

        # MuQ placeholder position (after tags, before lyrics)
        muq_idx = len(tags_ids)

        # Process lyrics
        if lyrics:
            lyrics = lyrics.lower().strip()

        # Tokenize lyrics
        if self.tokenizer is not None and lyrics:
            lyrics_ids = self.tokenizer.encode(lyrics).ids
        else:
            lyrics_ids = []

        # Add BOS/EOS to lyrics
        if lyrics_ids and lyrics_ids[0] != self.config.text_bos_id:
            lyrics_ids = [self.config.text_bos_id] + lyrics_ids
        if lyrics_ids and lyrics_ids[-1] != self.config.text_eos_id:
            lyrics_ids = lyrics_ids + [self.config.text_eos_id]

        # Total prompt length: tags + 1 (MuQ placeholder) + lyrics
        prompt_len = len(tags_ids) + 1 + len(lyrics_ids)

        # Create combined tokens: (prompt_len, num_codebooks+1)
        # Last column is text, first num_codebooks columns are audio (zeros for prompt)
        tokens = np.zeros((prompt_len, self._parallel_number), dtype=np.int32)

        # Fill text tokens in last column
        if tags_ids:
            tokens[:len(tags_ids), -1] = tags_ids
        if lyrics_ids:
            start_idx = len(tags_ids) + 1
            tokens[start_idx:start_idx + len(lyrics_ids), -1] = lyrics_ids

        tokens = mx.array(tokens)

        # Create mask: only text column is valid for prompt
        tokens_mask = np.zeros((prompt_len, self._parallel_number), dtype=bool)
        tokens_mask[:, -1] = True
        tokens_mask = mx.array(tokens_mask)

        # MuQ embedding (zeros for now, no audio reference)
        muq_embed = mx.zeros((self.heartmula.config.muq_dim,), dtype=mx.float32)

        # Position indices
        pos = mx.arange(prompt_len, dtype=mx.int32)

        # Batch for CFG: duplicate if cfg_scale != 1.0
        bs_size = 2 if cfg_scale != 1.0 else 1

        def _cfg_cat(tensor: mx.array) -> mx.array:
            tensor = tensor[None, ...]  # Add batch dim
            if cfg_scale != 1.0:
                tensor = mx.concatenate([tensor, tensor], axis=0)
            return tensor

        return {
            "tokens": _cfg_cat(tokens),
            "tokens_mask": _cfg_cat(tokens_mask),
            "muq_embed": _cfg_cat(muq_embed),
            "muq_idx": [muq_idx] * bs_size,
            "pos": _cfg_cat(pos),
        }

    def _pad_audio_token(self, token: mx.array) -> tuple:
        """Pad generated audio token for next iteration.

        Args:
            token: Audio codes of shape (batch, num_codebooks).

        Returns:
            Tuple of (padded_token, padded_token_mask).
        """
        batch_size = token.shape[0]
        num_codebooks = token.shape[1]

        # Create padded token: (batch, 1, num_codebooks+1)
        # Audio codes go in first num_codebooks columns, text (empty_id) in last column
        text_col = mx.full((batch_size, 1, 1), self.config.empty_id, dtype=mx.int32)
        audio_cols = token[:, None, :]  # (batch, 1, num_codebooks)
        padded_token = mx.concatenate([audio_cols, text_col], axis=2)

        # Create mask: audio is valid (True), text is not (False)
        audio_mask = mx.ones((batch_size, 1, num_codebooks), dtype=mx.bool_)
        text_mask = mx.zeros((batch_size, 1, 1), dtype=mx.bool_)
        padded_token_mask = mx.concatenate([audio_mask, text_mask], axis=2)

        return padded_token, padded_token_mask

    def generate(
        self,
        inputs: Dict[str, Any],
        duration: float = 30.0,
        temperature: float = 1.0,
        top_k: int = 50,
        cfg_scale: float = 1.5,
    ) -> mx.array:
        """Generate audio codes from preprocessed inputs.

        Args:
            inputs: Preprocessed inputs from preprocess().
            duration: Target duration in seconds.
            temperature: Sampling temperature.
            top_k: Top-k sampling parameter.
            cfg_scale: Classifier-free guidance scale.

        Returns:
            Generated audio codes of shape (num_codebooks, num_frames).
        """
        prompt_tokens = inputs["tokens"]
        prompt_tokens_mask = inputs["tokens_mask"]
        continuous_segment = inputs["muq_embed"]
        starts = inputs["muq_idx"]
        prompt_pos = inputs["pos"]

        frames = []

        # Setup caches
        bs_size = 2 if cfg_scale != 1.0 else 1
        self.heartmula.setup_caches(bs_size)

        # Generate first frame with prompt
        curr_token = self.heartmula.generate_frame(
            tokens=prompt_tokens,
            tokens_mask=prompt_tokens_mask,
            input_pos=prompt_pos,
            temperature=temperature,
            topk=top_k,
            cfg_scale=cfg_scale,
            continuous_segments=continuous_segment,
            starts=starts,
        )

        # Take only first batch (conditional)
        frames.append(curr_token[0:1, :])

        # Calculate max frames
        max_audio_frames = int(duration * self.config.frame_rate)

        # Generate remaining frames
        for i in range(max_audio_frames):
            # Pad current token for next input
            curr_token_padded, curr_token_mask = self._pad_audio_token(curr_token)

            # Calculate next position
            next_pos = prompt_pos[:, -1:] + i + 1

            # Generate next frame
            curr_token = self.heartmula.generate_frame(
                tokens=curr_token_padded,
                tokens_mask=curr_token_mask,
                input_pos=next_pos,
                temperature=temperature,
                topk=top_k,
                cfg_scale=cfg_scale,
                continuous_segments=None,
                starts=None,
            )

            # Check for EOS
            if mx.any(curr_token[0:1, :] >= self.config.audio_eos_id):
                break

            frames.append(curr_token[0:1, :])

            # Print progress every 10 frames
            if (i + 1) % 10 == 0:
                print(f"Generated {i + 1}/{max_audio_frames} frames...")

        # Reset caches
        self.heartmula.reset_caches()

        # Stack frames: list of (1, num_codebooks) -> (num_frames, num_codebooks)
        codes = mx.concatenate(frames, axis=0)  # (num_frames, num_codebooks)

        # Transpose to (num_codebooks, num_frames) for HeartCodec
        codes = codes.transpose(1, 0)  # (num_codebooks, num_frames)

        return codes

    def postprocess(
        self,
        codes: mx.array,
        num_steps: int = 10,
        guidance_scale: float = 1.25,
    ) -> mx.array:
        """Convert codes to audio waveform.

        Args:
            codes: Audio codes of shape (num_codebooks, num_frames).
            num_steps: ODE integration steps for HeartCodec.
            guidance_scale: CFG scale for HeartCodec.

        Returns:
            Audio waveform.
        """
        # HeartCodec expects (batch, frames, num_quantizers)
        codes = codes.transpose(1, 0)[None, :, :]  # (1, num_frames, num_codebooks)

        # Calculate duration based on actual frames to avoid unnecessary padding
        # HeartCodec's frame_rate is 50.0 (48kHz / 960 hop size)
        num_frames = codes.shape[1]
        codec_frame_rate = self.heartcodec.config.frame_rate or 50.0
        duration = num_frames / codec_frame_rate

        audio = self.heartcodec.detokenize(
            codes=codes,
            duration=duration,
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
        print("Preprocessing...")
        inputs = self.preprocess(lyrics=lyrics, tags=tags, cfg_scale=cfg_scale)

        print(f"Generating {duration}s of audio...")
        codes = self.generate(
            inputs=inputs,
            duration=duration,
            temperature=temperature,
            top_k=top_k,
            cfg_scale=cfg_scale,
        )

        print("Running HeartCodec detokenize...")
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
        remove_dc: bool = True,
    ) -> None:
        """Save audio to file.

        Args:
            audio: Audio waveform.
            path: Output file path.
            sample_rate: Sample rate (uses config default if None).
            remove_dc: Whether to remove DC offset.
        """
        import soundfile as sf

        sample_rate = sample_rate or self.config.sample_rate

        # Convert to numpy
        audio_np = np.array(audio)

        # Handle shape
        if audio_np.ndim == 3:
            audio_np = audio_np[0]  # Remove batch dim
        if audio_np.ndim == 2 and audio_np.shape[-1] == 1:
            audio_np = audio_np[:, 0]  # Remove channel dim

        # Remove DC offset (the model can produce biased output)
        if remove_dc:
            audio_np = audio_np - audio_np.mean()

        # Normalize
        max_val = np.abs(audio_np).max()
        if max_val > 0:
            audio_np = audio_np / max_val * 0.95  # Leave headroom

        # Save
        sf.write(str(path), audio_np, sample_rate)
        print(f"Saved to {path}")
