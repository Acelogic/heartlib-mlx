"""Text encoder for HeartCLAP."""

from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from heartlib_mlx.nn.transformer import LlamaTransformerBlock, RMSNorm


class TextEncoder(nn.Module):
    """Transformer-based text encoder for HeartCLAP.

    Encodes text descriptions into embeddings in the shared space.

    Args:
        vocab_size: Size of the vocabulary.
        embed_dim: Embedding dimension.
        output_dim: Output embedding dimension.
        n_heads: Number of attention heads.
        n_layers: Number of transformer layers.
        hidden_dim: MLP hidden dimension.
        max_seq_len: Maximum sequence length.
        norm_eps: Epsilon for normalization.
    """

    def __init__(
        self,
        vocab_size: int = 49408,
        embed_dim: int = 768,
        output_dim: int = 512,
        n_heads: int = 12,
        n_layers: int = 12,
        hidden_dim: int = 3072,
        max_seq_len: int = 77,
        norm_eps: float = 1e-6,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len

        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)

        # Positional embedding
        self.pos_embedding = nn.Embedding(max_seq_len, embed_dim)

        # Transformer layers
        self.layers = [
            LlamaTransformerBlock(
                dim=embed_dim,
                n_heads=n_heads,
                hidden_dim=hidden_dim,
                max_seq_len=max_seq_len,
                norm_eps=norm_eps,
            )
            for _ in range(n_layers)
        ]

        # Output norm
        self.norm = RMSNorm(embed_dim, eps=norm_eps)

        # Projection to output dimension
        self.proj = nn.Linear(embed_dim, output_dim)

    def __call__(
        self,
        token_ids: mx.array,
        attention_mask: Optional[mx.array] = None,
    ) -> mx.array:
        """Encode text to embedding.

        Args:
            token_ids: Token IDs of shape (batch, seq_len).
            attention_mask: Optional mask of shape (batch, seq_len).

        Returns:
            Text embedding of shape (batch, output_dim).
        """
        batch_size, seq_len = token_ids.shape

        # Token embeddings
        x = self.token_embedding(token_ids)

        # Add positional embeddings
        positions = mx.arange(seq_len)
        x = x + self.pos_embedding(positions)

        # Create causal mask
        mask = self._create_mask(seq_len, attention_mask)

        # Process through transformer
        for layer in self.layers:
            x, _ = layer(x, mask=mask)

        # Normalize
        x = self.norm(x)

        # Get embedding from EOS token position
        # Assuming last non-padded position is EOS
        if attention_mask is not None:
            # Find last valid position
            lengths = mx.sum(attention_mask, axis=-1).astype(mx.int32)
            # Use advanced indexing to get EOS positions
            indices = mx.clip(lengths - 1, 0, seq_len - 1)
            x = mx.take_along_axis(x, indices[:, None, None], axis=1)[:, 0, :]
        else:
            # Use last position
            x = x[:, -1, :]

        # Project to output dimension
        x = self.proj(x)

        # L2 normalize
        x = x / (mx.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)

        return x

    def _create_mask(
        self,
        seq_len: int,
        attention_mask: Optional[mx.array] = None,
    ) -> mx.array:
        """Create causal attention mask.

        Args:
            seq_len: Sequence length.
            attention_mask: Optional padding mask.

        Returns:
            Attention mask of shape (1, 1, seq_len, seq_len).
        """
        # Causal mask
        mask = mx.triu(mx.full((seq_len, seq_len), float("-inf")), k=1)

        # Add padding mask if provided
        if attention_mask is not None:
            # attention_mask: (batch, seq_len), 1 = valid, 0 = padding
            padding_mask = (1 - attention_mask[:, None, None, :]) * float("-inf")
            mask = mask[None, None, :, :] + padding_mask
        else:
            mask = mask[None, None, :, :]

        return mask
