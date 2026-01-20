"""HeartCLAP - Audio-Text Alignment Model."""

from typing import Optional, Tuple, List, Union
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from heartlib_mlx.heartclap.configuration import HeartCLAPConfig
from heartlib_mlx.heartclap.audio_encoder import AudioEncoder
from heartlib_mlx.heartclap.text_encoder import TextEncoder


class HeartCLAP(nn.Module):
    """HeartCLAP: Audio-Text Alignment Model.

    Learns a shared embedding space for audio and text, enabling:
    - Music tagging from audio
    - Audio retrieval from text descriptions
    - Cross-modal similarity computation

    Based on contrastive learning with InfoNCE loss.

    Args:
        config: HeartCLAPConfig with model hyperparameters.
    """

    def __init__(self, config: HeartCLAPConfig):
        super().__init__()
        self.config = config

        # Audio encoder
        self.audio_encoder = AudioEncoder(
            sample_rate=config.sample_rate,
            embed_dim=config.audio_dim,
            output_dim=config.embedding_dim,
        )

        # Text encoder
        self.text_encoder = TextEncoder(
            embed_dim=config.text_dim,
            output_dim=config.embedding_dim,
        )

        # Learnable temperature parameter
        self.logit_scale = mx.array([mx.log(mx.array(1 / config.temperature))])

    def embed_audio(self, audio: mx.array) -> mx.array:
        """Encode audio to embedding.

        Args:
            audio: Audio waveform or spectrogram.

        Returns:
            Audio embedding of shape (batch, embedding_dim).
        """
        return self.audio_encoder(audio)

    def embed_text(
        self,
        token_ids: mx.array,
        attention_mask: Optional[mx.array] = None,
    ) -> mx.array:
        """Encode text to embedding.

        Args:
            token_ids: Text token IDs.
            attention_mask: Optional attention mask.

        Returns:
            Text embedding of shape (batch, embedding_dim).
        """
        return self.text_encoder(token_ids, attention_mask)

    def similarity(
        self,
        audio: mx.array,
        token_ids: mx.array,
        attention_mask: Optional[mx.array] = None,
    ) -> mx.array:
        """Compute similarity between audio and text.

        Args:
            audio: Audio waveform or spectrogram.
            token_ids: Text token IDs.
            attention_mask: Optional attention mask.

        Returns:
            Similarity matrix of shape (audio_batch, text_batch).
        """
        # Get embeddings
        audio_emb = self.embed_audio(audio)
        text_emb = self.embed_text(token_ids, attention_mask)

        # Compute scaled cosine similarity
        logit_scale = mx.exp(self.logit_scale)
        similarity = logit_scale * (audio_emb @ text_emb.T)

        return similarity

    def __call__(
        self,
        audio: Optional[mx.array] = None,
        token_ids: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None,
        return_loss: bool = False,
    ) -> Union[Tuple[mx.array, mx.array], mx.array]:
        """Forward pass.

        Args:
            audio: Audio input.
            token_ids: Text token IDs.
            attention_mask: Text attention mask.
            return_loss: Whether to compute and return contrastive loss.

        Returns:
            If return_loss: Tuple of (audio_emb, text_emb, loss).
            Otherwise: Tuple of (audio_emb, text_emb) or single embedding.
        """
        audio_emb = None
        text_emb = None

        if audio is not None:
            audio_emb = self.embed_audio(audio)

        if token_ids is not None:
            text_emb = self.embed_text(token_ids, attention_mask)

        if return_loss and audio_emb is not None and text_emb is not None:
            # Compute contrastive loss
            logit_scale = mx.exp(self.logit_scale)
            logits = logit_scale * (audio_emb @ text_emb.T)

            batch_size = audio_emb.shape[0]
            labels = mx.arange(batch_size)

            # Symmetric cross-entropy loss
            loss_audio = nn.losses.cross_entropy(logits, labels)
            loss_text = nn.losses.cross_entropy(logits.T, labels)
            loss = (loss_audio + loss_text) / 2

            return audio_emb, text_emb, loss

        if audio_emb is not None and text_emb is not None:
            return audio_emb, text_emb
        elif audio_emb is not None:
            return audio_emb
        else:
            return text_emb

    def get_tags(
        self,
        audio: mx.array,
        tag_texts: List[str],
        tokenizer,
        top_k: int = 5,
    ) -> List[Tuple[str, float]]:
        """Get top-k tags for audio.

        Args:
            audio: Audio waveform or spectrogram.
            tag_texts: List of candidate tag strings.
            tokenizer: Tokenizer for encoding tags.
            top_k: Number of top tags to return.

        Returns:
            List of (tag, score) tuples.
        """
        # Encode audio
        audio_emb = self.embed_audio(audio)

        # Encode tags
        tag_tokens = tokenizer(tag_texts, padding=True, return_tensors="mlx")
        tag_emb = self.embed_text(tag_tokens["input_ids"], tag_tokens.get("attention_mask"))

        # Compute similarities
        similarities = (audio_emb @ tag_emb.T)[0]

        # Get top-k using argsort
        top_k = min(top_k, len(tag_texts))
        sorted_indices = mx.argsort(similarities)[::-1]  # Descending order
        topk_indices = sorted_indices[:top_k]
        topk_scores = similarities[topk_indices]

        return [(tag_texts[int(idx)], float(score)) for idx, score in zip(topk_indices, topk_scores)]

    @classmethod
    def from_pretrained(
        cls,
        path: Union[str, Path],
        dtype: mx.Dtype = mx.bfloat16,
    ) -> "HeartCLAP":
        """Load a pretrained HeartCLAP model.

        Args:
            path: Path to the model directory.
            dtype: Data type for model weights.

        Returns:
            HeartCLAP instance with loaded weights.
        """
        from safetensors import safe_open

        path = Path(path)

        # Load config
        config = HeartCLAPConfig.from_pretrained(path)

        # Create model
        model = cls(config)

        # Load weights
        weights_path = path / "model.safetensors"
        if weights_path.exists():
            with safe_open(str(weights_path), framework="mlx") as f:
                weights = {k: f.get_tensor(k) for k in f.keys()}

            weights = {k: v.astype(dtype) for k, v in weights.items()}
            model.load_weights(list(weights.items()))

        return model

    def save_pretrained(self, path: Union[str, Path]) -> None:
        """Save the model to a directory.

        Args:
            path: Path to save the model.
        """
        from safetensors.numpy import save_file
        import numpy as np

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        self.config.save_pretrained(path)

        weights = dict(self.parameters())
        np_weights = {k: np.array(v) for k, v in weights.items()}
        save_file(np_weights, str(path / "model.safetensors"))
