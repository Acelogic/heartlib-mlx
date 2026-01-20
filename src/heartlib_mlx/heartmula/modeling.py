"""HeartMuLa - Music Language Model."""

from typing import Optional, Tuple, List, Union
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from heartlib_mlx.heartmula.configuration import HeartMuLaConfig
from heartlib_mlx.heartmula.backbone import HeartMuLaBackbone
from heartlib_mlx.heartmula.decoder import HeartMuLaDecoder


def sample_topk(logits: mx.array, topk: int, temperature: float) -> mx.array:
    """Sample from logits using top-k sampling.

    Args:
        logits: Logits of shape (batch, vocab_size).
        topk: Number of top tokens to consider.
        temperature: Sampling temperature.

    Returns:
        Sampled token IDs of shape (batch, 1).
    """
    logits = logits / temperature

    # Get top-k
    topk = min(topk, logits.shape[-1])

    # Get top-k values and indices
    # argsort in descending order
    sorted_indices = mx.argsort(-logits, axis=-1)
    topk_indices = sorted_indices[:, :topk]
    topk_logits = mx.take_along_axis(logits, topk_indices, axis=-1)

    # Mask out non-top-k
    mask = mx.full(logits.shape, float("-inf"))
    mask = mx.put_along_axis(mask, topk_indices, topk_logits, axis=-1)

    # Log softmax and sample
    log_probs = mx.log(mx.softmax(mask, axis=-1) + 1e-10)
    samples = mx.random.categorical(log_probs)

    return samples[:, None]


class HeartMuLa(nn.Module):
    """HeartMuLa: Music Language Model.

    Matches the PyTorch implementation exactly:
    - Combined token format: (batch, seq_len, num_codebooks+1) where last dim is text
    - Embeddings are summed across codebook dimension, not concatenated
    - CFG via batch doubling (conditional + unconditional in same batch)
    - Per-codebook audio_head weights
    """

    def __init__(self, config: HeartMuLaConfig):
        super().__init__()
        self.config = config

        backbone_cfg = config.backbone_config
        decoder_cfg = config.decoder_config

        self.backbone_dim = backbone_cfg["dim"]
        self.decoder_dim = decoder_cfg["dim"]
        self.num_codebooks = config.audio_num_codebooks
        self.audio_vocab_size = config.audio_vocab_size

        # Text embeddings
        self.text_embeddings = nn.Embedding(config.text_vocab_size, self.backbone_dim)

        # Audio embeddings: single embedding table for all codebooks
        # Size: (audio_vocab_size * num_codebooks, dim)
        # Access: token + codebook_idx * audio_vocab_size
        self.audio_embeddings = nn.Embedding(
            config.audio_vocab_size * config.audio_num_codebooks,
            self.backbone_dim,
        )

        # Unconditional text embedding for CFG
        self.unconditional_text_embedding = nn.Embedding(1, self.backbone_dim)

        # Backbone transformer (LLaMA-3B)
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

        # Projection from backbone to decoder
        self.projection = nn.Linear(self.backbone_dim, self.decoder_dim, bias=False)

        # Local decoder (LLaMA-300M)
        self.decoder = HeartMuLaDecoder(
            dim=decoder_cfg["dim"],
            n_heads=decoder_cfg["n_heads"],
            n_kv_heads=decoder_cfg["n_kv_heads"],
            n_layers=decoder_cfg["n_layers"],
            hidden_dim=decoder_cfg["hidden_dim"],
            max_seq_len=config.audio_num_codebooks,  # Decoder only sees num_codebooks positions
            norm_eps=decoder_cfg["norm_eps"],
            rope_base=decoder_cfg["rope_base"],
        )

        # Prediction heads
        # Codebook 0 is predicted by the backbone
        self.codebook0_head = nn.Linear(self.backbone_dim, config.audio_vocab_size, bias=False)

        # Codebooks 1-7 are predicted by the decoder
        # Shape: (num_codebooks-1, decoder_dim, audio_vocab_size)
        # In MLX, we store this as a flattened Linear and index into output
        self.audio_head = nn.Linear(
            self.decoder_dim,
            config.audio_vocab_size * (config.audio_num_codebooks - 1),
            bias=False,
        )

        # MuQ linear for audio conditioning
        self.muq_linear = nn.Linear(config.muq_dim, self.backbone_dim, bias=True)

        # Cache state
        self._backbone_cache = None
        self._decoder_cache = None

    def setup_caches(self, batch_size: int):
        """Setup KV caches for generation."""
        self._backbone_cache = [None] * len(self.backbone.layers)
        self._decoder_cache = [None] * len(self.decoder.layers)

    def reset_caches(self):
        """Reset all caches."""
        self._backbone_cache = None
        self._decoder_cache = None

    def _embed_audio(self, codebook: int, tokens: mx.array) -> mx.array:
        """Embed audio tokens for a specific codebook.

        Args:
            codebook: Codebook index (0-7).
            tokens: Token IDs of shape (batch, seq_len) or (batch,).

        Returns:
            Embeddings of shape (batch, seq_len, dim) or (batch, dim).
        """
        return self.audio_embeddings(tokens + codebook * self.audio_vocab_size)

    def _embed_tokens(
        self,
        tokens: mx.array,
        uncond_mask: Optional[mx.array] = None,
    ) -> mx.array:
        """Embed combined tokens.

        Args:
            tokens: Combined tokens of shape (batch, seq_len, num_codebooks+1).
                    Last dimension is text, first num_codebooks are audio.
            uncond_mask: Boolean mask of shape (batch,) indicating unconditional samples.

        Returns:
            Embeddings of shape (batch, seq_len, num_codebooks+1, dim).
        """
        B, S, _ = tokens.shape

        # Text embeddings from last channel
        text_tokens = tokens[:, :, -1]  # (B, S)
        text_embeds = self.text_embeddings(text_tokens)  # (B, S, dim)

        # Apply unconditional embedding for CFG
        if uncond_mask is not None:
            uncond_embed = self.unconditional_text_embedding(
                mx.zeros((1,), dtype=mx.int32)
            )  # (1, dim)
            # Expand mask: (B,) -> (B, 1, 1)
            mask_expanded = uncond_mask[:, None, None]
            # Replace text embeddings with unconditional for masked samples
            text_embeds = mx.where(
                mx.broadcast_to(mask_expanded, text_embeds.shape),
                uncond_embed,
                text_embeds,
            )

        text_embeds = text_embeds[:, :, None, :]  # (B, S, 1, dim)

        # Audio embeddings from first num_codebooks channels
        # tokens[:, :, :-1] shape: (B, S, num_codebooks)
        audio_tokens = tokens[:, :, :-1]  # (B, S, num_codebooks)

        # Add codebook offsets: token + codebook_idx * vocab_size
        codebook_offsets = mx.arange(self.num_codebooks) * self.audio_vocab_size
        audio_tokens_offset = audio_tokens + codebook_offsets  # (B, S, num_codebooks)

        # Flatten for embedding lookup
        audio_tokens_flat = audio_tokens_offset.reshape(-1)  # (B * S * num_codebooks,)
        audio_embeds_flat = self.audio_embeddings(audio_tokens_flat)  # (B * S * num_codebooks, dim)
        audio_embeds = audio_embeds_flat.reshape(B, S, self.num_codebooks, -1)  # (B, S, num_codebooks, dim)

        # Concatenate: audio (num_codebooks) + text (1) -> (num_codebooks+1)
        embeds = mx.concatenate([audio_embeds, text_embeds], axis=2)  # (B, S, num_codebooks+1, dim)

        return embeds

    def generate_frame(
        self,
        tokens: mx.array,
        tokens_mask: mx.array,
        input_pos: mx.array,
        temperature: float,
        topk: int,
        cfg_scale: float,
        continuous_segments: Optional[mx.array] = None,
        starts: Optional[List[int]] = None,
    ) -> mx.array:
        """Generate a single audio frame (all codebooks).

        Args:
            tokens: Combined tokens of shape (batch, seq_len, num_codebooks+1).
            tokens_mask: Mask of shape (batch, seq_len, num_codebooks+1).
            input_pos: Position indices of shape (batch, seq_len).
            temperature: Sampling temperature.
            topk: Top-k sampling parameter.
            cfg_scale: Classifier-free guidance scale.
            continuous_segments: Optional MuQ embeddings.
            starts: Optional start positions for MuQ injection.

        Returns:
            Generated codes of shape (batch, num_codebooks).
            For CFG (batch=2), returns codes for conditional sample only (first half).
        """
        b, s, _ = tokens.shape

        # Determine unconditional mask for CFG
        uncond_mask = None
        if cfg_scale > 1.0 and b > 1:
            actual_B = b // 2
            # First half: conditional (False), Second half: unconditional (True)
            uncond_mask = mx.concatenate([
                mx.zeros((actual_B,), dtype=mx.bool_),
                mx.ones((actual_B,), dtype=mx.bool_),
            ])

        # Get embeddings: (B, S, num_codebooks+1, dim)
        embeds = self._embed_tokens(tokens, uncond_mask=uncond_mask)

        # Apply mask and sum across codebook dimension
        # tokens_mask: (B, S, num_codebooks+1) -> (B, S, num_codebooks+1, 1)
        masked_embeds = embeds * tokens_mask[:, :, :, None]
        h = mx.sum(masked_embeds, axis=2)  # (B, S, dim)

        # Inject MuQ embeddings if provided
        if continuous_segments is not None and starts is not None:
            # continuous_segments: (B, muq_dim) -> (B, backbone_dim)
            continuous_segments = self.muq_linear(continuous_segments)

            if uncond_mask is not None:
                # Get unconditional embedding: (1, backbone_dim)
                uncond_embed = self.unconditional_text_embedding(
                    mx.zeros((1,), dtype=mx.int32)
                )[0]  # (backbone_dim,)

                # Expand mask to (B, backbone_dim) for element-wise selection
                mask_expanded = uncond_mask[:, None]  # (B, 1)
                mask_expanded = mx.broadcast_to(mask_expanded, continuous_segments.shape)  # (B, backbone_dim)

                # Replace with uncond embedding where mask is True
                continuous_segments = mx.where(
                    mask_expanded,
                    mx.broadcast_to(uncond_embed, continuous_segments.shape),
                    continuous_segments,
                )

            # Inject at start positions using scatter-like operation
            # Note: We need to convert to float32 for numpy since bfloat16 isn't supported
            import numpy as np
            h_dtype = h.dtype
            h_np = np.array(h.astype(mx.float32))
            cs_np = np.array(continuous_segments.astype(mx.float32))
            for batch_idx, start in enumerate(starts):
                h_np[batch_idx, start] = cs_np[batch_idx]
            h = mx.array(h_np).astype(h_dtype)

        # Run backbone
        h, self._backbone_cache = self.backbone(h, cache=self._backbone_cache)

        # Get last position for prediction
        last_h = h[:, -1, :]  # (B, dim)

        # Predict codebook 0
        c0_logits = self.codebook0_head(last_h)  # (B, vocab_size)

        # Apply CFG for codebook 0
        if cfg_scale > 1.0 and b > 1 and (b % 2 == 0):
            actual_B = b // 2
            cond_logits = c0_logits[:actual_B, :]
            uncond_logits = c0_logits[actual_B:, :]
            guided_logits = uncond_logits + (cond_logits - uncond_logits) * cfg_scale
            c0_sample = sample_topk(guided_logits, topk, temperature)
            # Repeat for both branches to keep cache aligned
            c0_sample = mx.concatenate([c0_sample, c0_sample], axis=0)
        else:
            c0_sample = sample_topk(c0_logits, topk, temperature)

        c0_sample = c0_sample[:, 0]  # (B,)

        # Get codebook 0 embedding
        c0_embed = self._embed_audio(0, c0_sample)  # (B, dim)

        # Reset decoder cache for this frame
        self._decoder_cache = [None] * len(self.decoder.layers)

        # Initialize decoder with backbone output + c0 embedding
        curr_h = mx.stack([last_h, c0_embed], axis=1)  # (B, 2, dim)
        curr_sample = c0_sample[:, None]  # (B, 1)

        # Generate codebooks 1-7
        for i in range(1, self.num_codebooks):
            # Project and run decoder
            projected = self.projection(curr_h)
            decoder_h, self._decoder_cache = self.decoder(projected, cache=self._decoder_cache)

            # Get logits for codebook i from audio_head
            # audio_head output: (B, vocab_size * 7)
            all_logits = self.audio_head(decoder_h[:, -1, :])  # (B, vocab_size * 7)
            start_idx = (i - 1) * self.audio_vocab_size
            end_idx = i * self.audio_vocab_size
            ci_logits = all_logits[:, start_idx:end_idx]  # (B, vocab_size)

            # Apply CFG
            if cfg_scale > 1.0 and b > 1 and (b % 2 == 0):
                actual_B = b // 2
                cond_ci = ci_logits[:actual_B, :]
                uncond_ci = ci_logits[actual_B:, :]
                guided_ci = uncond_ci + (cond_ci - uncond_ci) * cfg_scale
                ci_sample = sample_topk(guided_ci, topk, temperature)
                ci_sample = mx.concatenate([ci_sample, ci_sample], axis=0)
            else:
                ci_sample = sample_topk(ci_logits, topk, temperature)

            ci_sample = ci_sample[:, 0]  # (B,)

            # Get embedding for next iteration
            ci_embed = self._embed_audio(i, ci_sample)  # (B, dim)
            curr_h = ci_embed[:, None, :]  # (B, 1, dim)
            curr_sample = mx.concatenate([curr_sample, ci_sample[:, None]], axis=1)

        # Force evaluation to prevent memory buildup
        mx.eval(curr_sample)

        return curr_sample  # (B, num_codebooks)

    @classmethod
    def from_pretrained(
        cls,
        path: Union[str, Path],
        dtype: mx.Dtype = mx.bfloat16,
    ) -> "HeartMuLa":
        """Load a pretrained HeartMuLa model."""
        path = Path(path)

        # Load config
        config = HeartMuLaConfig.from_pretrained(path)

        # Create model
        model = cls(config)

        # Load weights
        weights_path = path / "model.safetensors"
        if weights_path.exists():
            weights = mx.load(str(weights_path))
            weights = {k: v.astype(dtype) for k, v in weights.items()}
            model.load_weights(list(weights.items()))
            mx.eval(model.parameters())

        return model
