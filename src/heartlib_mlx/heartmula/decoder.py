"""Local decoder for HeartMuLa (LLaMA-300M)."""

from typing import Optional, Tuple, List

import mlx.core as mx
import mlx.nn as nn

from heartlib_mlx.nn.transformer import (
    RMSNorm,
    LlamaTransformerBlock,
)
from heartlib_mlx.nn.kv_cache import KVCache


class HeartMuLaDecoder(nn.Module):
    """Local decoder for HeartMuLa.

    This is a smaller transformer that predicts codebooks 1-7
    given the backbone output and previously predicted codebooks.

    Args:
        dim: Model dimension.
        n_heads: Number of attention heads.
        n_kv_heads: Number of key/value heads.
        n_layers: Number of transformer layers.
        hidden_dim: MLP hidden dimension.
        max_seq_len: Maximum sequence length.
        norm_eps: Epsilon for RMSNorm.
        rope_base: Base for RoPE.
    """

    def __init__(
        self,
        dim: int = 3072,
        n_heads: int = 8,
        n_kv_heads: int = 8,
        n_layers: int = 3,
        hidden_dim: int = 8192,
        max_seq_len: int = 8192,
        norm_eps: float = 1e-5,
        rope_base: float = 500000.0,
    ):
        super().__init__()

        self.dim = dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_layers = n_layers
        self.head_dim = dim // n_heads

        # Transformer layers
        self.layers = [
            LlamaTransformerBlock(
                dim=dim,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                hidden_dim=hidden_dim,
                max_seq_len=max_seq_len,
                norm_eps=norm_eps,
                rope_base=rope_base,
            )
            for _ in range(n_layers)
        ]

        # Output normalization
        self.norm = RMSNorm(dim, eps=norm_eps)

    def __call__(
        self,
        hidden_states: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[List[Tuple[mx.array, mx.array]]] = None,
    ) -> Tuple[mx.array, List[Tuple[mx.array, mx.array]]]:
        """Forward pass through decoder.

        Args:
            hidden_states: Input of shape (batch, seq_len, dim).
            mask: Optional attention mask.
            cache: Optional KV caches.

        Returns:
            Tuple of (output, new_caches).
        """
        # Compute offset from cache
        offset = 0
        if cache is not None and cache[0] is not None:
            offset = cache[0][0].shape[1]

        # Create causal mask if not provided
        if mask is None:
            seq_len = hidden_states.shape[1]
            mask = self._create_causal_mask(seq_len, offset)

        # Process through layers
        new_caches = []
        x = hidden_states

        for i, layer in enumerate(self.layers):
            layer_cache = cache[i] if cache is not None else None
            x, new_cache = layer(x, mask=mask, cache=layer_cache, offset=offset)
            new_caches.append(new_cache)

        # Final norm
        x = self.norm(x)

        return x, new_caches

    def _create_causal_mask(
        self,
        seq_len: int,
        offset: int = 0,
    ) -> mx.array:
        """Create causal attention mask."""
        total_len = seq_len + offset

        # Create causal mask using vectorized operations
        row_indices = mx.arange(seq_len)[:, None]
        col_indices = mx.arange(total_len)[None, :]
        causal_mask = col_indices <= row_indices + offset
        mask = mx.where(causal_mask, 0.0, float("-inf"))

        return mask[None, None, :, :]

    def setup_cache(
        self,
        batch_size: int,
        max_seq_len: int,
    ) -> KVCache:
        """Setup KV cache for generation.

        Args:
            batch_size: Batch size.
            max_seq_len: Maximum sequence length.

        Returns:
            KVCache instance.
        """
        return KVCache(
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            n_heads=self.n_kv_heads,
            head_dim=self.head_dim,
            n_layers=self.n_layers,
        )

    @classmethod
    def from_config(cls, config: dict) -> "HeartMuLaDecoder":
        """Create decoder from configuration dictionary.

        Args:
            config: Configuration dictionary.

        Returns:
            HeartMuLaDecoder instance.
        """
        return cls(
            dim=config["dim"],
            n_heads=config["n_heads"],
            n_kv_heads=config.get("n_kv_heads", config["n_heads"]),
            n_layers=config["n_layers"],
            hidden_dim=config.get("hidden_dim"),
            norm_eps=config.get("norm_eps", 1e-5),
            rope_base=config.get("rope_base", 10000.0),
        )
