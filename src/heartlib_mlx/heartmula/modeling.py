"""HeartMuLa - Music Language Model."""

from typing import Optional, Tuple, List, Union
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from heartlib_mlx.heartmula.configuration import HeartMuLaConfig
from heartlib_mlx.heartmula.embeddings import CombinedEmbedding
from heartlib_mlx.heartmula.backbone import HeartMuLaBackbone
from heartlib_mlx.heartmula.decoder import HeartMuLaDecoder


class HeartMuLa(nn.Module):
    """HeartMuLa: Music Language Model.

    A hierarchical music generation model that combines:
    1. Text embeddings for lyrics and tags
    2. Audio embeddings for multiple codebooks
    3. A backbone transformer (LLaMA-3B) for sequence modeling
    4. A local decoder (LLaMA-300M) for multi-codebook prediction

    The model generates audio in a frame-by-frame manner:
    - For each frame, the backbone predicts codebook 0
    - The decoder then predicts codebooks 1-7 autoregressively

    Args:
        config: HeartMuLaConfig with model hyperparameters.
    """

    def __init__(self, config: HeartMuLaConfig):
        super().__init__()
        self.config = config

        # Embeddings
        self.embeddings = CombinedEmbedding(
            text_vocab_size=config.text_vocab_size,
            audio_vocab_size=config.audio_vocab_size,
            audio_num_codebooks=config.audio_num_codebooks,
            dim=config.backbone_dim,
        )

        # Backbone transformer (LLaMA-3B)
        backbone_cfg = config.backbone_config
        self.backbone = HeartMuLaBackbone(
            dim=backbone_cfg["dim"],
            n_heads=backbone_cfg["n_heads"],
            n_kv_heads=backbone_cfg["n_kv_heads"],
            n_layers=backbone_cfg["n_layers"],
            hidden_dim=backbone_cfg["hidden_dim"],
            max_seq_len=config.max_seq_len,
            norm_eps=backbone_cfg["norm_eps"],
            rope_base=backbone_cfg["rope_base"],
        )

        # Projection from backbone to decoder (always create to match PyTorch)
        decoder_cfg = config.decoder_config
        self.projection = nn.Linear(backbone_cfg["dim"], decoder_cfg["dim"], bias=False)

        # Local decoder (LLaMA-300M)
        self.decoder = HeartMuLaDecoder(
            dim=decoder_cfg["dim"],
            n_heads=decoder_cfg["n_heads"],
            n_kv_heads=decoder_cfg["n_kv_heads"],
            n_layers=decoder_cfg["n_layers"],
            hidden_dim=decoder_cfg["hidden_dim"],
            max_seq_len=config.max_seq_len,
            norm_eps=decoder_cfg["norm_eps"],
            rope_base=decoder_cfg["rope_base"],
        )

        # Prediction heads
        # Codebook 0 is predicted by the backbone
        self.codebook0_head = nn.Linear(
            backbone_cfg["dim"],
            config.audio_vocab_size,
            bias=False,
        )

        # Codebooks 1-7 are predicted by the decoder
        # Shared head with different embeddings per codebook
        self.audio_head = nn.Linear(
            decoder_cfg["dim"],
            config.audio_vocab_size * (config.audio_num_codebooks - 1),
            bias=False,
        )

        # MuQ linear for audio conditioning (projects MuQ embeddings to model dim)
        self.muq_linear = nn.Linear(config.muq_dim, backbone_cfg["dim"], bias=True)

        # Store dimensions
        self.backbone_dim = backbone_cfg["dim"]
        self.decoder_dim = decoder_cfg["dim"]
        self.num_codebooks = config.audio_num_codebooks
        self.audio_vocab_size = config.audio_vocab_size

    def _embed_tokens(
        self,
        text_ids: Optional[mx.array] = None,
        audio_codes: Optional[mx.array] = None,
    ) -> mx.array:
        """Embed text and/or audio tokens.

        Args:
            text_ids: Text token IDs of shape (batch, text_len).
            audio_codes: Audio codes of shape (batch, audio_len, num_codebooks).

        Returns:
            Combined embeddings.
        """
        embeddings = []

        if text_ids is not None:
            text_emb = self.embeddings.embed_text(text_ids)
            embeddings.append(text_emb)

        if audio_codes is not None:
            audio_emb = self.embeddings.embed_audio(audio_codes)
            embeddings.append(audio_emb)

        if len(embeddings) == 0:
            raise ValueError("Must provide text_ids or audio_codes")

        return mx.concatenate(embeddings, axis=1) if len(embeddings) > 1 else embeddings[0]

    def generate_frame(
        self,
        text_ids: mx.array,
        audio_codes: Optional[mx.array] = None,
        backbone_cache: Optional[List] = None,
        temperature: float = 1.0,
        top_k: int = 50,
        cfg_scale: float = 1.5,
    ) -> Tuple[mx.array, List]:
        """Generate a single audio frame (all codebooks).

        Args:
            text_ids: Text conditioning tokens.
            audio_codes: Previously generated audio codes.
            backbone_cache: KV cache for backbone.
            temperature: Sampling temperature.
            top_k: Top-k sampling parameter.
            cfg_scale: Classifier-free guidance scale.

        Returns:
            Tuple of (new_codes, new_backbone_cache).
        """
        batch_size = text_ids.shape[0]

        # Get embeddings
        if audio_codes is not None:
            hidden = self._embed_tokens(text_ids, audio_codes)
        else:
            hidden = self._embed_tokens(text_ids)

        # Run backbone
        backbone_out, new_backbone_cache = self.backbone(hidden, cache=backbone_cache)

        # Get last position for prediction
        last_hidden = backbone_out[:, -1:, :]

        # Predict codebook 0
        logits_0 = self.codebook0_head(last_hidden)[:, 0, :]

        # Apply CFG if scale > 1
        if cfg_scale > 1.0:
            # Get unconditional prediction
            uncond_hidden = self.embeddings.get_unconditional(batch_size, 1)
            uncond_out, _ = self.backbone(uncond_hidden, cache=None)
            uncond_logits = self.codebook0_head(uncond_out[:, -1:, :])[:, 0, :]
            logits_0 = uncond_logits + cfg_scale * (logits_0 - uncond_logits)

        # Sample codebook 0
        code_0 = self._sample_topk(logits_0, temperature, top_k)

        # Generate codebooks 1-7 using decoder
        codes = [code_0]

        # Project backbone output for decoder
        decoder_input = self.projection(last_hidden)

        # Add codebook 0 embedding
        code_0_emb = self.embeddings.audio_embedding.embed_codebook(
            code_0[:, None], 0
        )
        decoder_hidden = decoder_input + code_0_emb

        decoder_cache = None
        for i in range(1, self.num_codebooks):
            # Run decoder
            decoder_out, decoder_cache = self.decoder(decoder_hidden, cache=decoder_cache)

            # Get logits for this codebook
            all_logits = self.audio_head(decoder_out[:, -1:, :])[:, 0, :]
            start_idx = (i - 1) * self.audio_vocab_size
            end_idx = i * self.audio_vocab_size
            logits_i = all_logits[:, start_idx:end_idx]

            # Sample
            code_i = self._sample_topk(logits_i, temperature, top_k)
            codes.append(code_i)

            # Update decoder input with new code embedding
            code_i_emb = self.embeddings.audio_embedding.embed_codebook(
                code_i[:, None], i
            )
            decoder_hidden = decoder_hidden + code_i_emb

        # Stack codes: (batch, num_codebooks)
        frame_codes = mx.stack(codes, axis=-1)

        return frame_codes, new_backbone_cache

    def _sample_topk(
        self,
        logits: mx.array,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> mx.array:
        """Sample from logits using top-k sampling.

        Args:
            logits: Logits of shape (batch, vocab_size).
            temperature: Sampling temperature.
            top_k: Number of top tokens to consider.

        Returns:
            Sampled token IDs of shape (batch,).
        """
        if temperature == 0:
            # Greedy decoding
            return mx.argmax(logits, axis=-1)

        # Apply temperature
        logits = logits / temperature

        # Get top-k using argsort
        top_k = min(top_k, logits.shape[-1])
        sorted_indices = mx.argsort(logits, axis=-1)[..., ::-1]  # Descending order
        topk_indices = sorted_indices[..., :top_k]
        topk_logits = mx.take_along_axis(logits, topk_indices, axis=-1)

        # Softmax and sample
        probs = mx.softmax(topk_logits, axis=-1)

        # Sample from multinomial
        samples = mx.random.categorical(mx.log(probs + 1e-10))

        # Map back to original indices
        return mx.take_along_axis(topk_indices, samples[:, None], axis=-1)[:, 0]

    def setup_caches(
        self,
        batch_size: int,
        max_seq_len: int,
    ) -> Tuple:
        """Setup KV caches for generation.

        Args:
            batch_size: Batch size.
            max_seq_len: Maximum sequence length.

        Returns:
            Tuple of (backbone_cache, decoder_cache).
        """
        backbone_cache = self.backbone.setup_cache(batch_size, max_seq_len)
        decoder_cache = self.decoder.setup_cache(batch_size, max_seq_len)
        return backbone_cache, decoder_cache

    def __call__(
        self,
        text_ids: Optional[mx.array] = None,
        audio_codes: Optional[mx.array] = None,
        labels: Optional[mx.array] = None,
    ) -> Tuple[mx.array, Optional[mx.array]]:
        """Forward pass for training or inference.

        Args:
            text_ids: Text token IDs.
            audio_codes: Audio codes.
            labels: Target labels for training.

        Returns:
            Tuple of (logits, loss).
        """
        # Get embeddings
        hidden = self._embed_tokens(text_ids, audio_codes)

        # Run backbone
        backbone_out, _ = self.backbone(hidden)

        # Codebook 0 logits
        logits_0 = self.codebook0_head(backbone_out)

        # For inference, just return codebook 0 logits
        if labels is None:
            return logits_0, None

        # For training, compute loss
        # Shift for next-token prediction
        shift_logits = logits_0[:, :-1, :].reshape(-1, self.audio_vocab_size)
        shift_labels = labels[:, 1:, 0].reshape(-1)

        # Cross-entropy loss
        loss = nn.losses.cross_entropy(shift_logits, shift_labels)

        return logits_0, loss

    @classmethod
    def from_pretrained(
        cls,
        path: Union[str, Path],
        dtype: mx.Dtype = mx.bfloat16,
    ) -> "HeartMuLa":
        """Load a pretrained HeartMuLa model.

        Args:
            path: Path to the model directory.
            dtype: Data type for model weights.

        Returns:
            HeartMuLa instance with loaded weights.
        """
        path = Path(path)

        # Load config
        config = HeartMuLaConfig.from_pretrained(path)

        # Create model
        model = cls(config)

        # Load weights using MLX's native loader (handles bfloat16 properly)
        weights_path = path / "model.safetensors"
        if weights_path.exists():
            weights = mx.load(str(weights_path))

            # Convert to target dtype if different
            weights = {k: v.astype(dtype) for k, v in weights.items()}

            # Load into model
            model.load_weights(list(weights.items()))
            mx.eval(model.parameters())

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

        # Save config
        self.config.save_pretrained(path)

        # Save weights
        weights = dict(self.parameters())
        np_weights = {k: np.array(v) for k, v in weights.items()}
        save_file(np_weights, str(path / "model.safetensors"))
