"""Audio tagging pipeline for HeartCLAP."""

from typing import Optional, Union, List, Tuple
from pathlib import Path

import mlx.core as mx
import numpy as np

from heartlib_mlx.heartclap import HeartCLAP, HeartCLAPConfig


# Common music tags
DEFAULT_TAGS = [
    "pop", "rock", "hip hop", "jazz", "classical", "electronic",
    "r&b", "country", "folk", "blues", "metal", "punk", "indie",
    "reggae", "soul", "funk", "disco", "ambient", "house", "techno",
    "acoustic", "instrumental", "vocal", "male vocal", "female vocal",
    "upbeat", "slow", "fast", "energetic", "calm", "relaxing",
    "happy", "sad", "melancholic", "romantic", "aggressive", "peaceful",
    "guitar", "piano", "drums", "bass", "strings", "synth", "saxophone",
    "live", "studio", "lo-fi", "hi-fi", "vintage", "modern",
]


class HeartCLAPPipeline:
    """Audio tagging and retrieval pipeline using HeartCLAP.

    This pipeline enables:
    - Audio tagging: Given audio, predict relevant tags
    - Audio retrieval: Given text, find matching audio
    - Similarity: Compute audio-text similarity scores

    Example:
        >>> pipeline = HeartCLAPPipeline.from_pretrained("./ckpt-mlx")
        >>> tags = pipeline.tag_audio("song.mp3", top_k=5)
        >>> print(tags)  # [("pop", 0.85), ("upbeat", 0.72), ...]
    """

    def __init__(
        self,
        model: HeartCLAP,
        tokenizer,
        config: HeartCLAPConfig,
        tags: Optional[List[str]] = None,
    ):
        """Initialize the pipeline.

        Args:
            model: HeartCLAP model.
            tokenizer: Text tokenizer.
            config: Model configuration.
            tags: List of candidate tags for tagging.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.tags = tags or DEFAULT_TAGS

        # Pre-compute tag embeddings
        self._tag_embeddings = None

    def _get_tag_embeddings(self) -> mx.array:
        """Get or compute tag embeddings."""
        if self._tag_embeddings is None:
            # Tokenize all tags
            if self.tokenizer is not None:
                encodings = [self.tokenizer.encode(tag) for tag in self.tags]
                max_len = max(len(e.ids) for e in encodings)
                token_ids = mx.array([
                    e.ids + [0] * (max_len - len(e.ids)) for e in encodings
                ])
            else:
                # Fallback
                max_len = max(len(tag) for tag in self.tags)
                token_ids = mx.array([
                    [ord(c) for c in tag] + [0] * (max_len - len(tag))
                    for tag in self.tags
                ])

            self._tag_embeddings = self.model.embed_text(token_ids)

        return self._tag_embeddings

    @classmethod
    def from_pretrained(
        cls,
        path: Union[str, Path],
        dtype: mx.Dtype = mx.bfloat16,
        tags: Optional[List[str]] = None,
    ) -> "HeartCLAPPipeline":
        """Load pipeline from pretrained weights.

        Args:
            path: Path to the model directory.
            dtype: Data type for model weights.
            tags: Optional custom tag list.

        Returns:
            HeartCLAPPipeline instance.
        """
        from tokenizers import Tokenizer

        path = Path(path)

        # Load model
        model = HeartCLAP.from_pretrained(path, dtype=dtype)

        # Load tokenizer
        tokenizer_path = path / "tokenizer.json"
        if tokenizer_path.exists():
            tokenizer = Tokenizer.from_file(str(tokenizer_path))
        else:
            tokenizer = None

        # Load config
        config = model.config

        return cls(model, tokenizer, config, tags)

    def load_audio(
        self,
        path: Union[str, Path],
        target_sr: Optional[int] = None,
    ) -> mx.array:
        """Load audio from file.

        Args:
            path: Path to audio file.
            target_sr: Target sample rate.

        Returns:
            Audio waveform as MLX array.
        """
        import soundfile as sf

        target_sr = target_sr or self.config.sample_rate

        # Load audio
        audio, sr = sf.read(str(path))

        # Convert to mono if stereo
        if audio.ndim > 1:
            audio = audio.mean(axis=-1)

        # Resample if needed
        if sr != target_sr:
            # Simple resampling via interpolation
            ratio = target_sr / sr
            new_len = int(len(audio) * ratio)
            indices = np.linspace(0, len(audio) - 1, new_len)
            audio = np.interp(indices, np.arange(len(audio)), audio)

        # Truncate or pad to max length
        max_len = self.config.max_audio_len
        if len(audio) > max_len:
            audio = audio[:max_len]
        elif len(audio) < max_len:
            audio = np.pad(audio, (0, max_len - len(audio)))

        return mx.array(audio[None, :])  # Add batch dim

    def tag_audio(
        self,
        audio: Union[str, Path, mx.array],
        top_k: int = 5,
        threshold: float = 0.0,
    ) -> List[Tuple[str, float]]:
        """Get tags for audio.

        Args:
            audio: Audio file path or waveform.
            top_k: Number of top tags to return.
            threshold: Minimum similarity threshold.

        Returns:
            List of (tag, score) tuples.
        """
        # Load audio if path
        if isinstance(audio, (str, Path)):
            audio = self.load_audio(audio)

        # Get audio embedding
        audio_emb = self.model.embed_audio(audio)

        # Get tag embeddings
        tag_emb = self._get_tag_embeddings()

        # Compute similarities
        similarities = (audio_emb @ tag_emb.T)[0]

        # Convert to numpy for sorting
        similarities_np = np.array(similarities)

        # Get top-k
        top_k = min(top_k, len(self.tags))
        top_indices = np.argsort(similarities_np)[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(similarities_np[idx])
            if score >= threshold:
                results.append((self.tags[idx], score))

        return results

    def embed_audio(self, audio: Union[str, Path, mx.array]) -> mx.array:
        """Get embedding for audio.

        Args:
            audio: Audio file path or waveform.

        Returns:
            Audio embedding.
        """
        if isinstance(audio, (str, Path)):
            audio = self.load_audio(audio)
        return self.model.embed_audio(audio)

    def embed_text(self, text: Union[str, List[str]]) -> mx.array:
        """Get embedding for text.

        Args:
            text: Text string or list of strings.

        Returns:
            Text embedding(s).
        """
        if isinstance(text, str):
            text = [text]

        if self.tokenizer is not None:
            encodings = [self.tokenizer.encode(t) for t in text]
            max_len = max(len(e.ids) for e in encodings)
            token_ids = mx.array([
                e.ids + [0] * (max_len - len(e.ids)) for e in encodings
            ])
        else:
            max_len = max(len(t) for t in text)
            token_ids = mx.array([
                [ord(c) for c in t] + [0] * (max_len - len(t))
                for t in text
            ])

        return self.model.embed_text(token_ids)

    def similarity(
        self,
        audio: Union[str, Path, mx.array],
        texts: List[str],
    ) -> mx.array:
        """Compute similarity between audio and texts.

        Args:
            audio: Audio file path or waveform.
            texts: List of text descriptions.

        Returns:
            Similarity scores for each text.
        """
        audio_emb = self.embed_audio(audio)
        text_emb = self.embed_text(texts)
        return (audio_emb @ text_emb.T)[0]

    def search_by_text(
        self,
        query: str,
        audio_embeddings: mx.array,
        audio_paths: List[str],
        top_k: int = 5,
    ) -> List[Tuple[str, float]]:
        """Search for audio matching text query.

        Args:
            query: Text query.
            audio_embeddings: Pre-computed audio embeddings.
            audio_paths: Corresponding audio file paths.
            top_k: Number of results to return.

        Returns:
            List of (path, score) tuples.
        """
        # Get query embedding
        query_emb = self.embed_text(query)

        # Compute similarities
        similarities = (query_emb @ audio_embeddings.T)[0]
        similarities_np = np.array(similarities)

        # Get top-k
        top_k = min(top_k, len(audio_paths))
        top_indices = np.argsort(similarities_np)[::-1][:top_k]

        return [(audio_paths[idx], float(similarities_np[idx])) for idx in top_indices]

    def __call__(
        self,
        audio: Union[str, Path, mx.array],
        top_k: int = 5,
    ) -> List[Tuple[str, float]]:
        """Tag audio (convenience method).

        Args:
            audio: Audio input.
            top_k: Number of tags to return.

        Returns:
            List of (tag, score) tuples.
        """
        return self.tag_audio(audio, top_k=top_k)
