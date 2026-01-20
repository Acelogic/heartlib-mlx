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
        """Build the cos/sin cache for the given sequence length.

        Uses interleaved format to match torchtune:
        cos/sin are computed for dim/2 frequencies and then interleaved
        to match the interleaved rotation format.
        """
        if seq_len <= self._cached_seq_len and self._cos_cache is not None:
            return

        # Create position indices
        positions = mx.arange(seq_len).astype(mx.float32) / self.scaling_factor

        # Compute frequencies: shape (seq_len, dim/2)
        freqs = mx.outer(positions, self._inv_freq)

        # Compute cos and sin for each frequency: shape (seq_len, dim/2)
        cos_half = mx.cos(freqs)
        sin_half = mx.sin(freqs)

        # Interleave to match rotation format: [cos0, cos0, cos1, cos1, ...]
        # Stack and reshape: (seq_len, dim/2, 2) -> (seq_len, dim)
        self._cos_cache = mx.stack([cos_half, cos_half], axis=-1).reshape(seq_len, -1)
        self._sin_cache = mx.stack([sin_half, sin_half], axis=-1).reshape(seq_len, -1)
        self._cached_seq_len = seq_len

    def _rotate_half(self, x: mx.array) -> mx.array:
        """Rotate half the hidden dims of the input.

        Uses interleaved format to match torchtune:
        Given x = [x0, x1, x2, x3, ...], where pairs (x0,x1), (x2,x3) etc. are rotated together,
        returns [-x1, x0, -x3, x2, ...]
        """
        # x1 = even indices, x2 = odd indices
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        # Stack and flatten to interleave: [-x2[0], x1[0], -x2[1], x1[1], ...]
        rotated = mx.stack([-x2, x1], axis=-1)
        return rotated.reshape(x.shape)

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
    Uses interleaved format to match torchtune.

    Args:
        q: Query tensor.
        k: Key tensor.
        cos: Cosine values (interleaved format).
        sin: Sine values (interleaved format).

    Returns:
        Tuple of rotated (q, k).
    """
    def rotate_half(x):
        # Interleaved format: [-x2[0], x1[0], -x2[1], x1[1], ...]
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        rotated = mx.stack([-x2, x1], axis=-1)
        return rotated.reshape(x.shape)

    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin
    return q_rot, k_rot


def precompute_freqs_cis(
    dim: int,
    max_seq_len: int,
    base: float = 10000.0,
) -> Tuple[mx.array, mx.array]:
    """Precompute cosine and sine frequencies for RoPE.

    Uses interleaved format to match torchtune.

    Args:
        dim: Head dimension.
        max_seq_len: Maximum sequence length.
        base: Base for frequency computation.

    Returns:
        Tuple of (cos, sin) arrays of shape (max_seq_len, dim) in interleaved format.
    """
    inv_freq = 1.0 / (base ** (mx.arange(0, dim, 2).astype(mx.float32) / dim))
    positions = mx.arange(max_seq_len).astype(mx.float32)
    freqs = mx.outer(positions, inv_freq)

    # Compute cos/sin and interleave: [cos0, cos0, cos1, cos1, ...]
    cos_half = mx.cos(freqs)
    sin_half = mx.sin(freqs)
    cos = mx.stack([cos_half, cos_half], axis=-1).reshape(max_seq_len, -1)
    sin = mx.stack([sin_half, sin_half], axis=-1).reshape(max_seq_len, -1)
    return cos, sin
