"""Residual Vector Quantization for HeartCodec."""

from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn


class EMACodebook(nn.Module):
    """EMA-updated codebook matching PyTorch structure.

    Stores embeddings in _codebook.embed to match PyTorch weights.
    """

    def __init__(self, codebook_size: int, codebook_dim: int):
        super().__init__()
        # PyTorch stores as (1, codebook_size, codebook_dim)
        self.embed = mx.random.normal(shape=(1, codebook_size, codebook_dim)) * 0.02
        # EMA tracking (not used in inference, but needed for weight loading)
        self.cluster_size = mx.zeros((1, codebook_size))
        self.embed_avg = mx.zeros((1, codebook_size, codebook_dim))
        self.initted = mx.zeros((1,))


class VectorQuantizer(nn.Module):
    """Single-level Vector Quantizer.

    Maps continuous embeddings to discrete codes using a codebook.

    Args:
        codebook_size: Number of codes in the codebook.
        codebook_dim: Dimension of each code vector.
        use_cosine_sim: Use cosine similarity instead of L2 distance.
    """

    def __init__(
        self,
        codebook_size: int = 8192,
        codebook_dim: int = 512,
        use_cosine_sim: bool = False,
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.use_cosine_sim = use_cosine_sim

        # Use EMACodebook structure to match PyTorch weights
        # Note: Named 'codebook' (not '_codebook') so MLX tracks it as a parameter
        # PyTorch uses '_codebook' but we map in conversion
        self.codebook = EMACodebook(codebook_size, codebook_dim)

    def get_codebook_embeddings(self) -> mx.array:
        """Get the codebook embeddings (2D view)."""
        return self.codebook.embed[0]  # Remove leading batch dim

    def encode(self, x: mx.array) -> mx.array:
        """Encode continuous embeddings to discrete codes.

        Args:
            x: Input embeddings of shape (..., codebook_dim).

        Returns:
            Codes of shape (...).
        """
        # Flatten for distance computation
        original_shape = x.shape[:-1]
        x_flat = x.reshape(-1, self.codebook_dim)

        # Get codebook embeddings
        cb = self.get_codebook_embeddings()

        if self.use_cosine_sim:
            # Normalize both x and codebook
            x_norm = x_flat / (mx.linalg.norm(x_flat, axis=-1, keepdims=True) + 1e-8)
            cb_norm = cb / (mx.linalg.norm(cb, axis=-1, keepdims=True) + 1e-8)
            # Cosine similarity (higher is better, so negate for argmin)
            distances = -mx.matmul(x_norm, cb_norm.T)
        else:
            # L2 distance
            # ||x - c||^2 = ||x||^2 + ||c||^2 - 2 * x.c
            x_sq = mx.sum(x_flat ** 2, axis=-1, keepdims=True)
            cb_sq = mx.sum(cb ** 2, axis=-1)
            distances = x_sq + cb_sq - 2 * mx.matmul(x_flat, cb.T)

        # Find nearest code
        codes = mx.argmin(distances, axis=-1)
        return codes.reshape(original_shape)

    def decode(self, codes: mx.array) -> mx.array:
        """Decode discrete codes to continuous embeddings.

        Args:
            codes: Code indices of shape (...).

        Returns:
            Embeddings of shape (..., codebook_dim).
        """
        # Lookup embeddings from codebook
        cb = self.get_codebook_embeddings()
        return cb[codes]

    def __call__(
        self,
        x: mx.array,
    ) -> Tuple[mx.array, mx.array, mx.array]:
        """Forward pass with quantization.

        Args:
            x: Input embeddings of shape (..., codebook_dim).

        Returns:
            Tuple of (quantized, codes, commitment_loss).
        """
        codes = self.encode(x)
        quantized = self.decode(codes)

        # Commitment loss (for training)
        commitment_loss = mx.mean((x - mx.stop_gradient(quantized)) ** 2)

        # Straight-through estimator
        quantized = x + mx.stop_gradient(quantized - x)

        return quantized, codes, commitment_loss


class ResidualVQ(nn.Module):
    """Residual Vector Quantization.

    Applies multiple levels of vector quantization, where each level
    quantizes the residual from the previous level.

    Args:
        num_quantizers: Number of quantization levels (codebooks).
        codebook_size: Number of codes per codebook.
        codebook_dim: Dimension of code vectors (internal VQ dimension).
        dim: Input/output dimension (projected to/from codebook_dim).
        use_cosine_sim: Use cosine similarity for quantization.
    """

    def __init__(
        self,
        num_quantizers: int = 8,
        codebook_size: int = 8192,
        codebook_dim: int = 32,
        dim: int = 512,
        use_cosine_sim: bool = False,
    ):
        super().__init__()
        self.num_quantizers = num_quantizers
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.dim = dim

        # Input/output projections (always present to match PyTorch)
        # PyTorch: project_in maps dim -> codebook_dim
        # PyTorch: project_out maps codebook_dim -> dim
        self.project_in = nn.Linear(dim, codebook_dim, bias=True)
        self.project_out = nn.Linear(codebook_dim, dim, bias=True)

        # Create quantizer layers (named 'layers' to match PyTorch)
        self.layers = [
            VectorQuantizer(
                codebook_size=codebook_size,
                codebook_dim=codebook_dim,
                use_cosine_sim=use_cosine_sim,
            )
            for _ in range(num_quantizers)
        ]

    def encode(self, x: mx.array) -> mx.array:
        """Encode input to multi-level codes.

        Args:
            x: Input of shape (batch, seq_len, dim).

        Returns:
            Codes of shape (batch, seq_len, num_quantizers).
        """
        # Project to codebook dimension
        x = self.project_in(x)

        all_codes = []
        residual = x

        for quantizer in self.layers:
            codes = quantizer.encode(residual)
            all_codes.append(codes)

            # Compute residual for next level
            quantized = quantizer.decode(codes)
            residual = residual - quantized

        # Stack codes: (batch, seq_len, num_quantizers)
        return mx.stack(all_codes, axis=-1)

    def decode(self, codes: mx.array) -> mx.array:
        """Decode multi-level codes to embeddings.

        Args:
            codes: Codes of shape (batch, seq_len, num_quantizers).

        Returns:
            Embeddings of shape (batch, seq_len, dim).
        """
        # Sum up all quantized levels
        quantized = mx.zeros((*codes.shape[:-1], self.codebook_dim))

        for i, quantizer in enumerate(self.layers):
            level_codes = codes[..., i]
            quantized = quantized + quantizer.decode(level_codes)

        # Project back to output dimension
        quantized = self.project_out(quantized)

        return quantized

    def from_codes(self, codes: mx.array) -> mx.array:
        """Alias for decode() for compatibility.

        Args:
            codes: Codes of shape (batch, seq_len, num_quantizers).

        Returns:
            Embeddings of shape (batch, seq_len, dim).
        """
        return self.decode(codes)

    def __call__(
        self,
        x: mx.array,
        n_quantizers: Optional[int] = None,
    ) -> Tuple[mx.array, mx.array, mx.array]:
        """Forward pass with residual quantization.

        Args:
            x: Input of shape (batch, seq_len, dim).
            n_quantizers: Number of quantizers to use (default: all).

        Returns:
            Tuple of (quantized, codes, total_commitment_loss).
        """
        n_q = n_quantizers or self.num_quantizers

        # Project to codebook dimension
        x = self.project_in(x)

        all_codes = []
        all_quantized = []
        total_loss = mx.array(0.0)
        residual = x

        for quantizer in self.layers[:n_q]:
            quantized, codes, loss = quantizer(residual)
            all_codes.append(codes)
            all_quantized.append(quantized)
            total_loss = total_loss + loss

            # Update residual
            residual = residual - mx.stop_gradient(quantized)

        # Sum all quantized levels
        final_quantized = sum(all_quantized)

        # Project back to output dimension
        final_quantized = self.project_out(final_quantized)

        # Stack codes
        codes = mx.stack(all_codes, axis=-1)

        return final_quantized, codes, total_loss / n_q


class ScalarQuantizer(nn.Module):
    """Scalar Quantizer using rounding.

    Quantizes values by rounding to the nearest 1/9th.
    This is a simple fixed-point quantization without learned codebooks.
    """

    def __init__(self, num_levels: int = 9):
        super().__init__()
        self.num_levels = num_levels

    def encode(self, x: mx.array) -> mx.array:
        """Quantize by rounding.

        Args:
            x: Input tensor.

        Returns:
            Quantized tensor (still continuous for reconstruction).
        """
        return mx.round(self.num_levels * x) / self.num_levels

    def __call__(self, x: mx.array) -> Tuple[mx.array, mx.array]:
        """Forward pass with straight-through gradient.

        Args:
            x: Input tensor.

        Returns:
            Tuple of (quantized, quantized_detached).
        """
        quantized = self.encode(x)
        # Straight-through: gradient flows through as if no quantization
        quantized_st = x + mx.stop_gradient(quantized - x)
        return quantized_st, quantized
