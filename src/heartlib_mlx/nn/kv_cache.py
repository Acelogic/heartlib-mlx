"""KV Cache implementations for efficient autoregressive generation."""

from typing import Optional, Tuple

import mlx.core as mx


class KVCache:
    """Simple Key-Value cache for autoregressive generation.

    Stores key and value tensors for all attention layers to avoid
    recomputation during generation.

    Args:
        batch_size: Batch size.
        max_seq_len: Maximum sequence length.
        n_heads: Number of attention heads.
        head_dim: Dimension of each attention head.
        n_layers: Number of transformer layers.
    """

    def __init__(
        self,
        batch_size: int,
        max_seq_len: int,
        n_heads: int,
        head_dim: int,
        n_layers: int,
    ):
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.n_layers = n_layers

        # Initialize empty caches
        self.k_cache = [None] * n_layers
        self.v_cache = [None] * n_layers
        self.seq_len = 0

    def update(
        self,
        layer_idx: int,
        k: mx.array,
        v: mx.array,
    ) -> Tuple[mx.array, mx.array]:
        """Update cache for a specific layer and return full k, v.

        Args:
            layer_idx: Index of the transformer layer.
            k: New key tensor of shape (batch, seq_len, n_heads, head_dim).
            v: New value tensor with same shape.

        Returns:
            Tuple of full (k, v) tensors including cached values.
        """
        if self.k_cache[layer_idx] is None:
            # First call - initialize cache
            self.k_cache[layer_idx] = k
            self.v_cache[layer_idx] = v
        else:
            # Append to cache
            self.k_cache[layer_idx] = mx.concatenate(
                [self.k_cache[layer_idx], k], axis=1
            )
            self.v_cache[layer_idx] = mx.concatenate(
                [self.v_cache[layer_idx], v], axis=1
            )

        # Update sequence length (only on first layer to avoid double counting)
        if layer_idx == 0:
            self.seq_len += k.shape[1]

        return self.k_cache[layer_idx], self.v_cache[layer_idx]

    def get(self, layer_idx: int) -> Tuple[Optional[mx.array], Optional[mx.array]]:
        """Get cached k, v for a specific layer.

        Args:
            layer_idx: Index of the transformer layer.

        Returns:
            Tuple of (k, v) tensors or (None, None) if not cached.
        """
        return self.k_cache[layer_idx], self.v_cache[layer_idx]

    def reset(self) -> None:
        """Reset all caches."""
        self.k_cache = [None] * self.n_layers
        self.v_cache = [None] * self.n_layers
        self.seq_len = 0

    @property
    def current_seq_len(self) -> int:
        """Return the current sequence length in cache."""
        return self.seq_len


class RotatingKVCache:
    """Rotating Key-Value cache with fixed memory budget.

    When the cache is full, oldest entries are discarded to make room
    for new ones. This enables generation of arbitrarily long sequences
    with bounded memory.

    Args:
        batch_size: Batch size.
        max_seq_len: Maximum sequence length to cache.
        n_heads: Number of attention heads.
        head_dim: Dimension of each attention head.
        n_layers: Number of transformer layers.
    """

    def __init__(
        self,
        batch_size: int,
        max_seq_len: int,
        n_heads: int,
        head_dim: int,
        n_layers: int,
    ):
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.n_layers = n_layers

        # Pre-allocate cache buffers
        shape = (batch_size, max_seq_len, n_heads, head_dim)
        self.k_cache = [mx.zeros(shape) for _ in range(n_layers)]
        self.v_cache = [mx.zeros(shape) for _ in range(n_layers)]

        # Track valid positions
        self.seq_len = 0
        self.total_len = 0  # Total tokens seen (for position tracking)

    def update(
        self,
        layer_idx: int,
        k: mx.array,
        v: mx.array,
    ) -> Tuple[mx.array, mx.array]:
        """Update cache for a specific layer and return full k, v.

        Args:
            layer_idx: Index of the transformer layer.
            k: New key tensor of shape (batch, new_len, n_heads, head_dim).
            v: New value tensor with same shape.

        Returns:
            Tuple of full (k, v) tensors including cached values.
        """
        new_len = k.shape[1]

        if self.seq_len + new_len <= self.max_seq_len:
            # Cache has space - just append
            start = self.seq_len
            end = self.seq_len + new_len

            # Update cache in place
            self.k_cache[layer_idx] = mx.concatenate([
                self.k_cache[layer_idx][:, :start],
                k,
                self.k_cache[layer_idx][:, end:],
            ], axis=1)
            self.v_cache[layer_idx] = mx.concatenate([
                self.v_cache[layer_idx][:, :start],
                v,
                self.v_cache[layer_idx][:, end:],
            ], axis=1)

            if layer_idx == 0:
                self.seq_len += new_len
                self.total_len += new_len

            # Return valid portion of cache
            return (
                self.k_cache[layer_idx][:, :self.seq_len],
                self.v_cache[layer_idx][:, :self.seq_len],
            )
        else:
            # Cache is full - rotate (discard oldest)
            keep_len = self.max_seq_len - new_len

            # Shift old values and add new
            self.k_cache[layer_idx] = mx.concatenate([
                self.k_cache[layer_idx][:, -keep_len:] if keep_len > 0 else k[:, :0],
                k,
            ], axis=1)
            self.v_cache[layer_idx] = mx.concatenate([
                self.v_cache[layer_idx][:, -keep_len:] if keep_len > 0 else v[:, :0],
                v,
            ], axis=1)

            if layer_idx == 0:
                self.seq_len = self.max_seq_len
                self.total_len += new_len

            return self.k_cache[layer_idx], self.v_cache[layer_idx]

    def get(self, layer_idx: int) -> Tuple[mx.array, mx.array]:
        """Get cached k, v for a specific layer.

        Args:
            layer_idx: Index of the transformer layer.

        Returns:
            Tuple of (k, v) tensors (valid portion only).
        """
        return (
            self.k_cache[layer_idx][:, :self.seq_len],
            self.v_cache[layer_idx][:, :self.seq_len],
        )

    def reset(self) -> None:
        """Reset all caches."""
        shape = (self.batch_size, self.max_seq_len, self.n_heads, self.head_dim)
        self.k_cache = [mx.zeros(shape) for _ in range(self.n_layers)]
        self.v_cache = [mx.zeros(shape) for _ in range(self.n_layers)]
        self.seq_len = 0
        self.total_len = 0

    @property
    def current_seq_len(self) -> int:
        """Return the current sequence length in cache."""
        return self.seq_len

    @property
    def offset(self) -> int:
        """Return the position offset for RoPE."""
        return self.total_len - self.seq_len


class HierarchicalKVCache:
    """Hierarchical KV Cache for HeartMuLa's two-level architecture.

    Manages separate caches for the backbone (LLaMA-3B) and decoder
    (LLaMA-300M) transformers.

    Args:
        batch_size: Batch size.
        max_seq_len: Maximum sequence length.
        backbone_config: Dict with n_heads, head_dim, n_layers for backbone.
        decoder_config: Dict with n_heads, head_dim, n_layers for decoder.
    """

    def __init__(
        self,
        batch_size: int,
        max_seq_len: int,
        backbone_config: dict,
        decoder_config: dict,
    ):
        self.backbone_cache = RotatingKVCache(
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            n_heads=backbone_config["n_heads"],
            head_dim=backbone_config["head_dim"],
            n_layers=backbone_config["n_layers"],
        )

        self.decoder_cache = RotatingKVCache(
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            n_heads=decoder_config["n_heads"],
            head_dim=decoder_config["head_dim"],
            n_layers=decoder_config["n_layers"],
        )

    def reset(self) -> None:
        """Reset both caches."""
        self.backbone_cache.reset()
        self.decoder_cache.reset()

    def reset_decoder(self) -> None:
        """Reset only the decoder cache (for hierarchical generation)."""
        self.decoder_cache.reset()
