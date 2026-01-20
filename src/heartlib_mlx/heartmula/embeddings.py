"""Embedding layers for HeartMuLa."""

from typing import Optional

import mlx.core as mx
import mlx.nn as nn


class TextEmbedding(nn.Module):
    """Text token embedding layer.

    Args:
        vocab_size: Size of the text vocabulary.
        dim: Embedding dimension.
    """

    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.embedding = nn.Embedding(vocab_size, dim)

    def __call__(self, token_ids: mx.array) -> mx.array:
        """Get embeddings for token IDs.

        Args:
            token_ids: Token IDs of shape (batch, seq_len).

        Returns:
            Embeddings of shape (batch, seq_len, dim).
        """
        return self.embedding(token_ids)


class AudioEmbedding(nn.Module):
    """Audio codebook embedding layer.

    Handles multiple codebooks by summing their embeddings.

    Args:
        vocab_size: Size of each codebook vocabulary.
        num_codebooks: Number of codebooks (RVQ levels).
        dim: Embedding dimension.
    """

    def __init__(
        self,
        vocab_size: int,
        num_codebooks: int,
        dim: int,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_codebooks = num_codebooks
        self.dim = dim

        # Separate embedding for each codebook
        self.embeddings = [
            nn.Embedding(vocab_size, dim) for _ in range(num_codebooks)
        ]

    def __call__(
        self,
        codes: mx.array,
        codebook_mask: Optional[mx.array] = None,
    ) -> mx.array:
        """Get embeddings for audio codes.

        Args:
            codes: Audio codes of shape (batch, seq_len, num_codebooks).
            codebook_mask: Optional mask of shape (num_codebooks,) indicating
                which codebooks to use.

        Returns:
            Summed embeddings of shape (batch, seq_len, dim).
        """
        batch_size, seq_len, n_codebooks = codes.shape

        # Sum embeddings from each codebook
        total_embedding = mx.zeros((batch_size, seq_len, self.dim))

        for i in range(min(n_codebooks, self.num_codebooks)):
            if codebook_mask is not None and not codebook_mask[i]:
                continue
            codebook_codes = codes[:, :, i]
            total_embedding = total_embedding + self.embeddings[i](codebook_codes)

        return total_embedding

    def embed_codebook(
        self,
        codes: mx.array,
        codebook_idx: int,
    ) -> mx.array:
        """Get embeddings for a single codebook.

        Args:
            codes: Audio codes of shape (batch, seq_len) for the given codebook.
            codebook_idx: Index of the codebook to use.

        Returns:
            Embeddings of shape (batch, seq_len, dim).
        """
        return self.embeddings[codebook_idx](codes)


class CombinedEmbedding(nn.Module):
    """Combined text and audio embedding layer.

    Handles mixed sequences of text and audio tokens.

    Args:
        text_vocab_size: Size of text vocabulary.
        audio_vocab_size: Size of audio vocabulary per codebook.
        audio_num_codebooks: Number of audio codebooks.
        dim: Embedding dimension.
    """

    def __init__(
        self,
        text_vocab_size: int,
        audio_vocab_size: int,
        audio_num_codebooks: int,
        dim: int,
    ):
        super().__init__()
        self.text_embedding = TextEmbedding(text_vocab_size, dim)
        self.audio_embedding = AudioEmbedding(
            audio_vocab_size, audio_num_codebooks, dim
        )

        # Unconditional embedding for classifier-free guidance
        self.unconditional_embedding = mx.zeros((dim,))

    def embed_text(self, token_ids: mx.array) -> mx.array:
        """Embed text tokens.

        Args:
            token_ids: Text token IDs.

        Returns:
            Text embeddings.
        """
        return self.text_embedding(token_ids)

    def embed_audio(
        self,
        codes: mx.array,
        codebook_mask: Optional[mx.array] = None,
    ) -> mx.array:
        """Embed audio codes.

        Args:
            codes: Audio codes.
            codebook_mask: Optional codebook mask.

        Returns:
            Audio embeddings.
        """
        return self.audio_embedding(codes, codebook_mask)

    def get_unconditional(self, batch_size: int, seq_len: int) -> mx.array:
        """Get unconditional embeddings for CFG.

        Args:
            batch_size: Batch size.
            seq_len: Sequence length.

        Returns:
            Unconditional embeddings of shape (batch, seq_len, dim).
        """
        return mx.broadcast_to(
            self.unconditional_embedding,
            (batch_size, seq_len, self.unconditional_embedding.shape[-1]),
        )


class PositionalEmbedding(nn.Module):
    """Learnable positional embeddings.

    Args:
        max_len: Maximum sequence length.
        dim: Embedding dimension.
    """

    def __init__(self, max_len: int, dim: int):
        super().__init__()
        self.max_len = max_len
        self.dim = dim
        self.embedding = nn.Embedding(max_len, dim)

    def __call__(self, seq_len: int, offset: int = 0) -> mx.array:
        """Get positional embeddings.

        Args:
            seq_len: Sequence length.
            offset: Position offset.

        Returns:
            Positional embeddings of shape (seq_len, dim).
        """
        positions = mx.arange(offset, offset + seq_len)
        return self.embedding(positions)
