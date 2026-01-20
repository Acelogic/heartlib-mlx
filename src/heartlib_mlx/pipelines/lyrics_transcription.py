"""Lyrics transcription pipeline for HeartTranscriptor."""

from typing import Optional, Union, List
from pathlib import Path

import mlx.core as mx
import numpy as np

from heartlib_mlx.hearttranscriptor import HeartTranscriptor, HeartTranscriptorConfig


class HeartTranscriptorPipeline:
    """Lyrics transcription pipeline using HeartTranscriptor.

    This pipeline handles:
    1. Audio loading and preprocessing
    2. Mel spectrogram computation
    3. Autoregressive text generation
    4. Text post-processing

    Example:
        >>> pipeline = HeartTranscriptorPipeline.from_pretrained("./ckpt-mlx")
        >>> lyrics = pipeline.transcribe("song.mp3")
        >>> print(lyrics)
        "[Verse]\\nHello world..."
    """

    def __init__(
        self,
        model: HeartTranscriptor,
        tokenizer,
        config: HeartTranscriptorConfig,
    ):
        """Initialize the pipeline.

        Args:
            model: HeartTranscriptor model.
            tokenizer: Whisper tokenizer.
            config: Model configuration.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

    @classmethod
    def from_pretrained(
        cls,
        path: Union[str, Path],
        dtype: mx.Dtype = mx.bfloat16,
    ) -> "HeartTranscriptorPipeline":
        """Load pipeline from pretrained weights.

        Args:
            path: Path to the model directory.
            dtype: Data type for model weights.

        Returns:
            HeartTranscriptorPipeline instance.
        """
        from tokenizers import Tokenizer

        path = Path(path)

        # Load model
        model = HeartTranscriptor.from_pretrained(path, dtype=dtype)

        # Load tokenizer
        tokenizer_path = path / "tokenizer.json"
        if tokenizer_path.exists():
            tokenizer = Tokenizer.from_file(str(tokenizer_path))
        else:
            tokenizer = None

        # Load config
        config = model.config

        return cls(model, tokenizer, config)

    def load_audio(
        self,
        path: Union[str, Path],
    ) -> np.ndarray:
        """Load audio from file.

        Args:
            path: Path to audio file.

        Returns:
            Audio waveform as numpy array.
        """
        import soundfile as sf

        # Load audio
        audio, sr = sf.read(str(path))

        # Convert to mono if stereo
        if audio.ndim > 1:
            audio = audio.mean(axis=-1)

        # Resample to 16kHz if needed
        target_sr = self.config.sample_rate
        if sr != target_sr:
            ratio = target_sr / sr
            new_len = int(len(audio) * ratio)
            indices = np.linspace(0, len(audio) - 1, new_len)
            audio = np.interp(indices, np.arange(len(audio)), audio)

        return audio.astype(np.float32)

    def compute_mel_spectrogram(
        self,
        audio: np.ndarray,
    ) -> mx.array:
        """Compute mel spectrogram from audio.

        Args:
            audio: Audio waveform.

        Returns:
            Mel spectrogram as MLX array.
        """
        n_fft = self.config.n_fft
        hop_length = self.config.hop_length
        n_mels = self.config.n_mels

        # Pad audio to ensure we get consistent length
        audio = np.pad(audio, (0, n_fft // 2))

        # Compute STFT
        # Simple implementation - in production, use a proper library
        num_frames = 1 + (len(audio) - n_fft) // hop_length
        frames = np.zeros((num_frames, n_fft))

        for i in range(num_frames):
            start = i * hop_length
            frames[i] = audio[start:start + n_fft] * np.hanning(n_fft)

        # FFT
        stft = np.fft.rfft(frames, axis=-1)
        magnitudes = np.abs(stft) ** 2

        # Mel filterbank
        mel_filters = self._create_mel_filterbank(n_fft, n_mels)
        mel_spec = np.dot(magnitudes, mel_filters.T)

        # Log scale
        mel_spec = np.log10(np.maximum(mel_spec, 1e-10))

        # Normalize
        mel_spec = (mel_spec + 4.0) / 4.0
        mel_spec = np.clip(mel_spec, -1.0, 1.0)

        # Transpose and add batch dim: (n_mels, time)
        mel_spec = mel_spec.T[None, :, :]

        return mx.array(mel_spec)

    def _create_mel_filterbank(
        self,
        n_fft: int,
        n_mels: int,
    ) -> np.ndarray:
        """Create mel filterbank matrix.

        Args:
            n_fft: FFT size.
            n_mels: Number of mel bands.

        Returns:
            Mel filterbank matrix.
        """
        sample_rate = self.config.sample_rate
        n_freqs = n_fft // 2 + 1

        # Mel scale conversion
        def hz_to_mel(hz):
            return 2595 * np.log10(1 + hz / 700)

        def mel_to_hz(mel):
            return 700 * (10 ** (mel / 2595) - 1)

        # Create mel points
        low_mel = hz_to_mel(0)
        high_mel = hz_to_mel(sample_rate / 2)
        mel_points = np.linspace(low_mel, high_mel, n_mels + 2)
        hz_points = mel_to_hz(mel_points)

        # Create filterbank
        bin_points = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)
        filterbank = np.zeros((n_mels, n_freqs))

        for i in range(n_mels):
            left = bin_points[i]
            center = bin_points[i + 1]
            right = bin_points[i + 2]

            for j in range(left, center):
                filterbank[i, j] = (j - left) / (center - left)
            for j in range(center, right):
                filterbank[i, j] = (right - j) / (right - center)

        return filterbank

    def transcribe(
        self,
        audio: Union[str, Path, np.ndarray, mx.array],
        language: str = "en",
        max_length: int = 448,
        temperature: float = 0.0,
    ) -> str:
        """Transcribe lyrics from audio.

        Args:
            audio: Audio file path or waveform.
            language: Language code (e.g., "en", "zh", "ja").
            max_length: Maximum output length.
            temperature: Sampling temperature (0 = greedy).

        Returns:
            Transcribed lyrics text.
        """
        # Load audio if path
        if isinstance(audio, (str, Path)):
            audio = self.load_audio(audio)

        # Convert to numpy if MLX array
        if isinstance(audio, mx.array):
            audio = np.array(audio)

        # Compute mel spectrogram
        mel = self.compute_mel_spectrogram(audio)

        # Language token mapping
        language_tokens = {
            "en": 50259, "zh": 50260, "de": 50261, "es": 50262,
            "ru": 50263, "ko": 50264, "fr": 50265, "ja": 50266,
            "pt": 50267, "tr": 50268, "pl": 50269, "ca": 50270,
        }
        language_token = language_tokens.get(language, 50259)

        # Generate
        token_ids = self.model.generate(
            mel,
            max_length=max_length,
            temperature=temperature,
            language_token=language_token,
        )

        # Decode tokens
        token_ids_np = np.array(token_ids[0])

        if self.tokenizer is not None:
            text = self.tokenizer.decode(token_ids_np.tolist())
        else:
            # Fallback: skip special tokens and decode
            # Filter out special tokens (>= 50257)
            regular_tokens = [t for t in token_ids_np if t < 50257]
            text = "".join(chr(t) for t in regular_tokens if 32 <= t < 127)

        # Clean up text
        text = self._postprocess_text(text)

        return text

    def _postprocess_text(self, text: str) -> str:
        """Clean up transcribed text.

        Args:
            text: Raw transcribed text.

        Returns:
            Cleaned text.
        """
        # Remove special tokens
        special_tokens = [
            "<|startoftranscript|>", "<|endoftext|>", "<|transcribe|>",
            "<|translate|>", "<|notimestamps|>",
        ]
        for token in special_tokens:
            text = text.replace(token, "")

        # Remove language tokens
        import re
        text = re.sub(r"<\|[a-z]{2}\|>", "", text)

        # Clean whitespace
        text = " ".join(text.split())

        return text.strip()

    def transcribe_with_timestamps(
        self,
        audio: Union[str, Path, np.ndarray],
        language: str = "en",
    ) -> List[dict]:
        """Transcribe with word-level timestamps.

        Args:
            audio: Audio input.
            language: Language code.

        Returns:
            List of dicts with 'text', 'start', 'end' keys.
        """
        # For now, return single segment
        # Full timestamp implementation requires more complex decoding
        text = self.transcribe(audio, language=language)
        return [{"text": text, "start": 0.0, "end": None}]

    def __call__(
        self,
        audio: Union[str, Path, np.ndarray, mx.array],
        language: str = "en",
    ) -> str:
        """Transcribe lyrics (convenience method).

        Args:
            audio: Audio input.
            language: Language code.

        Returns:
            Transcribed lyrics.
        """
        return self.transcribe(audio, language=language)
