"""Transformer layers for MLX (LLaMA-style architecture)."""

from typing import Optional, Tuple, Callable

import mlx.core as mx
import mlx.nn as nn

from heartlib_mlx.nn.rope import RotaryPositionEmbedding, apply_rotary_pos_emb


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Normalizes inputs using RMS rather than mean and variance,
    which is more efficient and works well for transformers.

    Args:
        dim: Dimension of the input.
        eps: Epsilon for numerical stability.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones((dim,))

    def __call__(self, x: mx.array) -> mx.array:
        """Apply RMS normalization.

        Args:
            x: Input tensor.

        Returns:
            Normalized tensor.
        """
        return mx.fast.rms_norm(x, self.weight, self.eps)


class LlamaAttention(nn.Module):
    """Multi-head attention with rotary position embeddings.

    Implements the attention mechanism used in LLaMA models with
    support for grouped-query attention (GQA).

    Args:
        dim: Model dimension.
        n_heads: Number of attention heads.
        n_kv_heads: Number of key/value heads (for GQA). If None, uses n_heads.
        head_dim: Dimension of each attention head. If None, computed as dim // n_heads.
        max_seq_len: Maximum sequence length for RoPE cache.
        rope_base: Base for RoPE frequency computation.
        bias: Whether to use bias in projections.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        max_seq_len: int = 8192,
        rope_base: float = 10000.0,
        bias: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads or n_heads
        self.head_dim = head_dim or (dim // n_heads)
        self.n_rep = self.n_heads // self.n_kv_heads  # Repeat factor for GQA

        self.scale = self.head_dim ** -0.5

        # Projections
        self.q_proj = nn.Linear(dim, n_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(n_heads * self.head_dim, dim, bias=bias)

        # Rotary embeddings
        self.rope = RotaryPositionEmbedding(
            dim=self.head_dim,
            max_seq_len=max_seq_len,
            base=rope_base,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
        offset: int = 0,
    ) -> Tuple[mx.array, Tuple[mx.array, mx.array]]:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, dim).
            mask: Attention mask of shape (batch, 1, seq_len, total_len).
            cache: Optional tuple of (k, v) cached tensors.
            offset: Position offset for RoPE (when using cache).

        Returns:
            Tuple of (output, (k, v)) where output has shape (batch, seq_len, dim).
        """
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape to (batch, seq_len, n_heads, head_dim)
        q = q.reshape(batch_size, seq_len, self.n_heads, self.head_dim)
        k = k.reshape(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        v = v.reshape(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        # Apply rotary embeddings
        q, k = self.rope(q, k, offset=offset)

        # Handle cache
        if cache is not None:
            k_cache, v_cache = cache
            k = mx.concatenate([k_cache, k], axis=1)
            v = mx.concatenate([v_cache, v], axis=1)
            # Evaluate to materialize and prevent computation graph accumulation
            mx.eval(k, v)

        # Transpose to (batch, n_heads, seq_len, head_dim) for SDPA
        # Note: SDPA handles GQA natively, no need to repeat K/V
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        # Use fused scaled dot-product attention (Flash Attention)
        # SDPA handles GQA automatically when n_kv_heads < n_heads
        output = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.scale, mask=mask
        )

        # Reshape back to (batch, seq_len, dim)
        output = output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        output = self.o_proj(output)

        # Transpose K/V back to (batch, seq_len, n_kv_heads, head_dim) for cache
        k_out = k.transpose(0, 2, 1, 3)
        v_out = v.transpose(0, 2, 1, 3)

        return output, (k_out, v_out)


class LlamaMLP(nn.Module):
    """SwiGLU MLP used in LLaMA models.

    Implements the gated linear unit with SiLU activation:
    output = down_proj(silu(gate_proj(x)) * up_proj(x))

    Args:
        dim: Model dimension.
        hidden_dim: Hidden dimension (typically 4 * dim or 8/3 * dim).
        bias: Whether to use bias in linear layers.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: Optional[int] = None,
        bias: bool = False,
    ):
        super().__init__()
        hidden_dim = hidden_dim or int(8 * dim / 3)
        # Round to multiple of 256 for efficiency
        hidden_dim = ((hidden_dim + 255) // 256) * 256

        self.gate_proj = nn.Linear(dim, hidden_dim, bias=bias)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=bias)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=bias)

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, dim).

        Returns:
            Output tensor of same shape.
        """
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class LlamaTransformerBlock(nn.Module):
    """Transformer block with pre-normalization (LLaMA style).

    Consists of:
    1. RMSNorm + Multi-head attention + residual
    2. RMSNorm + MLP + residual

    Args:
        dim: Model dimension.
        n_heads: Number of attention heads.
        n_kv_heads: Number of key/value heads (for GQA).
        hidden_dim: MLP hidden dimension.
        max_seq_len: Maximum sequence length.
        norm_eps: Epsilon for RMSNorm.
        rope_base: Base for RoPE.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        max_seq_len: int = 8192,
        norm_eps: float = 1e-6,
        rope_base: float = 10000.0,
    ):
        super().__init__()

        self.attention = LlamaAttention(
            dim=dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            max_seq_len=max_seq_len,
            rope_base=rope_base,
        )
        self.mlp = LlamaMLP(dim=dim, hidden_dim=hidden_dim)

        self.attention_norm = RMSNorm(dim, eps=norm_eps)
        self.mlp_norm = RMSNorm(dim, eps=norm_eps)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
        offset: int = 0,
    ) -> Tuple[mx.array, Tuple[mx.array, mx.array]]:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, dim).
            mask: Attention mask.
            cache: Optional KV cache.
            offset: Position offset for RoPE.

        Returns:
            Tuple of (output, new_cache).
        """
        # Attention with residual
        h = self.attention_norm(x)
        attn_out, new_cache = self.attention(h, mask=mask, cache=cache, offset=offset)
        x = x + attn_out

        # MLP with residual
        x = x + self.mlp(self.mlp_norm(x))

        return x, new_cache


class LlamaTransformer(nn.Module):
    """Full LLaMA-style transformer.

    Stack of transformer blocks with embedding and output projection.

    Args:
        vocab_size: Size of the vocabulary.
        dim: Model dimension.
        n_layers: Number of transformer layers.
        n_heads: Number of attention heads.
        n_kv_heads: Number of key/value heads (for GQA).
        hidden_dim: MLP hidden dimension.
        max_seq_len: Maximum sequence length.
        norm_eps: Epsilon for RMSNorm.
        rope_base: Base for RoPE.
    """

    def __init__(
        self,
        vocab_size: int,
        dim: int,
        n_layers: int,
        n_heads: int,
        n_kv_heads: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        max_seq_len: int = 8192,
        norm_eps: float = 1e-6,
        rope_base: float = 10000.0,
        tie_word_embeddings: bool = False,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads or n_heads
        self.head_dim = dim // n_heads

        # Token embedding
        self.embed_tokens = nn.Embedding(vocab_size, dim)

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

        # Output norm and projection
        self.norm = RMSNorm(dim, eps=norm_eps)

        if tie_word_embeddings:
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(dim, vocab_size, bias=False)

        self.tie_word_embeddings = tie_word_embeddings

    def __call__(
        self,
        input_ids: Optional[mx.array] = None,
        inputs_embeds: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache: Optional[list] = None,
    ) -> Tuple[mx.array, list]:
        """Forward pass.

        Args:
            input_ids: Token IDs of shape (batch, seq_len).
            inputs_embeds: Pre-computed embeddings of shape (batch, seq_len, dim).
            mask: Attention mask.
            cache: List of KV caches for each layer.

        Returns:
            Tuple of (logits, new_caches).
        """
        if inputs_embeds is None:
            assert input_ids is not None
            x = self.embed_tokens(input_ids)
        else:
            x = inputs_embeds

        batch_size, seq_len, _ = x.shape

        # Create causal mask if not provided
        if mask is None:
            mask = self._create_causal_mask(seq_len, cache)

        # Compute offset from cache
        offset = 0
        if cache is not None and cache[0] is not None:
            offset = cache[0][0].shape[1]

        # Process through layers
        new_caches = []
        for i, layer in enumerate(self.layers):
            layer_cache = cache[i] if cache is not None else None
            x, new_cache = layer(x, mask=mask, cache=layer_cache, offset=offset)
            new_caches.append(new_cache)

        # Final norm
        x = self.norm(x)

        # Output projection
        if self.lm_head is not None:
            logits = self.lm_head(x)
        else:
            # Tied embeddings
            logits = x @ self.embed_tokens.weight.T

        return logits, new_caches

    def _create_causal_mask(
        self,
        seq_len: int,
        cache: Optional[list] = None,
    ) -> mx.array:
        """Create causal attention mask.

        Args:
            seq_len: Current sequence length.
            cache: Optional cache to determine total length.

        Returns:
            Causal mask of shape (1, 1, seq_len, total_len).
        """
        total_len = seq_len
        if cache is not None and cache[0] is not None:
            total_len += cache[0][0].shape[1]

        # Create causal mask using vectorized operations
        cache_len = total_len - seq_len
        row_indices = mx.arange(seq_len)[:, None]
        col_indices = mx.arange(total_len)[None, :]
        causal_mask = col_indices <= row_indices + cache_len
        mask = mx.where(causal_mask, 0.0, float("-inf"))

        return mask[None, None, :, :]


class AdaLayerNormSingle(nn.Module):
    """Adaptive Layer Normalization for diffusion models.

    Modulates the normalization with scale and shift parameters
    predicted from a timestep embedding.

    Args:
        dim: Feature dimension.
        eps: Epsilon for normalization.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.norm = RMSNorm(dim, eps=eps)
        # Project timestep to scale and shift
        self.linear = nn.Linear(dim, 2 * dim)

    def __call__(self, x: mx.array, timestep_emb: mx.array) -> mx.array:
        """Apply adaptive normalization.

        Args:
            x: Input tensor of shape (batch, seq_len, dim).
            timestep_emb: Timestep embedding of shape (batch, dim).

        Returns:
            Normalized and modulated tensor.
        """
        # Get scale and shift from timestep
        scale_shift = self.linear(timestep_emb)
        scale, shift = mx.split(scale_shift, 2, axis=-1)

        # Normalize and modulate
        x = self.norm(x)
        x = x * (1 + scale[:, None, :]) + shift[:, None, :]

        return x


class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding for diffusion models.

    Args:
        dim: Embedding dimension.
        max_period: Maximum period for sinusoidal encoding.
    """

    def __init__(self, dim: int, max_period: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

        # MLP to project embedding
        self.linear1 = nn.Linear(dim, dim * 4)
        self.linear2 = nn.Linear(dim * 4, dim)

    def __call__(self, timesteps: mx.array) -> mx.array:
        """Compute timestep embeddings.

        Args:
            timesteps: Timestep values of shape (batch,).

        Returns:
            Embeddings of shape (batch, dim).
        """
        half_dim = self.dim // 2
        freqs = mx.exp(
            -mx.log(mx.array(self.max_period))
            * mx.arange(half_dim).astype(mx.float32)
            / half_dim
        )

        args = timesteps[:, None].astype(mx.float32) * freqs[None, :]
        embedding = mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)

        # MLP
        embedding = self.linear1(embedding)
        embedding = nn.silu(embedding)
        embedding = self.linear2(embedding)

        return embedding
