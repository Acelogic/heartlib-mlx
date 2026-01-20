"""Rotary Position Embeddings (RoPE) for MLX."""

from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn


class RotaryPositionEmbedding(nn.Module):
    """Rotary Position Embedding.

    Implements RoPE as described in "RoFormer: Enhanced Transformer with
    Rotary Position Embedding" (Su et al., 2021).

    Args:
        dim: Dimension of the embedding (must be even).
        max_seq_len: Maximum sequence length for caching.
        base: Base for the frequency computation.
        scaling_factor: Optional scaling factor for extended context.
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 8192,
        base: float = 10000.0,
        scaling_factor: float = 1.0,
    ):
        super().__init__()
        assert dim % 2 == 0, "Dimension must be even for RoPE"

        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        self.scaling_factor = scaling_factor

        # Compute inverse frequencies
        inv_freq = 1.0 / (base ** (mx.arange(0, dim, 2).astype(mx.float32) / dim))
        self._inv_freq = inv_freq

        # Cache for cos/sin values
        self._cos_cache: Optional[mx.array] = None
        self._sin_cache: Optional[mx.array] = None
        self._cached_seq_len = 0

    def _build_cache(self, seq_len: int) -> None:
        """Build the cos/sin cache for the given sequence length."""
        if seq_len <= self._cached_seq_len and self._cos_cache is not None:
            return

        # Create position indices
        positions = mx.arange(seq_len).astype(mx.float32) / self.scaling_factor

        # Compute frequencies: shape (seq_len, dim/2)
        freqs = mx.outer(positions, self._inv_freq)

        # Create full dimension by duplicating: shape (seq_len, dim)
        freqs = mx.concatenate([freqs, freqs], axis=-1)

        # Cache cos and sin
        self._cos_cache = mx.cos(freqs)
        self._sin_cache = mx.sin(freqs)
        self._cached_seq_len = seq_len

    def _rotate_half(self, x: mx.array) -> mx.array:
        """Rotate half the hidden dims of the input."""
        half_dim = x.shape[-1] // 2
        x1 = x[..., :half_dim]
        x2 = x[..., half_dim:]
        return mx.concatenate([-x2, x1], axis=-1)

    def __call__(
        self,
        q: mx.array,
        k: mx.array,
        offset: int = 0,
    ) -> Tuple[mx.array, mx.array]:
        """Apply rotary embeddings to query and key tensors.

        Args:
            q: Query tensor of shape (batch, seq_len, n_heads, head_dim) or
               (batch, n_heads, seq_len, head_dim).
            k: Key tensor with same shape as q.
            offset: Position offset for cached KV.

        Returns:
            Tuple of rotated (q, k) tensors.
        """
        # q, k shape: (batch, seq_len, n_heads, head_dim)
        seq_len = q.shape[1]

        # Build cache if needed
        self._build_cache(offset + seq_len)

        # Get the relevant slice of cos/sin
        assert self._cos_cache is not None and self._sin_cache is not None
        cos = self._cos_cache[offset : offset + seq_len]
        sin = self._sin_cache[offset : offset + seq_len]

        # Reshape for broadcasting with (batch, seq_len, n_heads, head_dim)
        # cos/sin are (seq_len, dim), need to become (1, seq_len, 1, dim)
        cos = cos[None, :, None, :]  # (1, seq_len, 1, dim)
        sin = sin[None, :, None, :]

        # Apply rotation
        q_rot = q * cos + self._rotate_half(q) * sin
        k_rot = k * cos + self._rotate_half(k) * sin

        return q_rot, k_rot

    def forward_one(
        self,
        x: mx.array,
        offset: int = 0,
    ) -> mx.array:
        """Apply rotary embeddings to a single tensor.

        Args:
            x: Input tensor of shape (batch, seq_len, n_heads, head_dim).
            offset: Position offset.

        Returns:
            Rotated tensor.
        """
        seq_len = x.shape[1]
        self._build_cache(offset + seq_len)

        assert self._cos_cache is not None and self._sin_cache is not None
        cos = self._cos_cache[offset : offset + seq_len]
        sin = self._sin_cache[offset : offset + seq_len]

        # Reshape for broadcasting with (batch, seq_len, n_heads, head_dim)
        cos = cos[None, :, None, :]  # (1, seq_len, 1, dim)
        sin = sin[None, :, None, :]

        return x * cos + self._rotate_half(x) * sin


def apply_rotary_pos_emb(
    q: mx.array,
    k: mx.array,
    cos: mx.array,
    sin: mx.array,
) -> Tuple[mx.array, mx.array]:
    """Apply rotary position embeddings to q and k.

    This is a functional version for cases where cos/sin are precomputed.

    Args:
        q: Query tensor.
        k: Key tensor.
        cos: Cosine values.
        sin: Sine values.

    Returns:
        Tuple of rotated (q, k).
    """
    def rotate_half(x):
        half_dim = x.shape[-1] // 2
        x1 = x[..., :half_dim]
        x2 = x[..., half_dim:]
        return mx.concatenate([-x2, x1], axis=-1)

    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin
    return q_rot, k_rot


def precompute_freqs_cis(
    dim: int,
    max_seq_len: int,
    base: float = 10000.0,
) -> Tuple[mx.array, mx.array]:
    """Precompute cosine and sine frequencies for RoPE.

    Args:
        dim: Head dimension.
        max_seq_len: Maximum sequence length.
        base: Base for frequency computation.

    Returns:
        Tuple of (cos, sin) arrays of shape (max_seq_len, dim).
    """
    inv_freq = 1.0 / (base ** (mx.arange(0, dim, 2).astype(mx.float32) / dim))
    positions = mx.arange(max_seq_len).astype(mx.float32)
    freqs = mx.outer(positions, inv_freq)
    freqs = mx.concatenate([freqs, freqs], axis=-1)
    return mx.cos(freqs), mx.sin(freqs)
