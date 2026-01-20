"""Custom activation functions for HeartCodec."""

import mlx.core as mx
import mlx.nn as nn


class Snake(nn.Module):
    """Snake activation function.

    Defined as: x + (1/α) * sin²(αx)

    This activation is particularly effective for audio generation tasks
    as it introduces periodic inductive bias.

    Reference: "Neural Networks Fail to Learn Periodic Functions and
    How to Fix It" (Ziyin et al., 2020)

    Args:
        channels: Number of channels for learnable alpha parameter.
        alpha_init: Initial value for alpha. Higher values give more
            oscillations in the activation.
    """

    def __init__(self, channels: int, alpha_init: float = 1.0):
        super().__init__()
        self.channels = channels
        # Learnable frequency parameter per channel
        self.alpha = mx.ones((channels,)) * alpha_init

    def __call__(self, x: mx.array) -> mx.array:
        """Apply Snake activation.

        Args:
            x: Input tensor of shape (..., channels).

        Returns:
            Activated tensor with same shape.
        """
        # Ensure alpha broadcasts correctly
        alpha = self.alpha
        # Reshape for broadcasting if needed
        while alpha.ndim < x.ndim:
            alpha = alpha[None, ...]

        # Snake: x + (1/α) * sin²(αx)
        sin_term = mx.sin(alpha * x)
        return x + (1.0 / (alpha + 1e-8)) * (sin_term * sin_term)


class SnakeBeta(nn.Module):
    """Snake activation with separate beta parameter.

    Defined as: x + (1/β) * sin²(αx)

    Separates the frequency (α) and amplitude (β) parameters.

    Args:
        channels: Number of channels.
        alpha_init: Initial frequency parameter.
        beta_init: Initial amplitude parameter.
    """

    def __init__(
        self,
        channels: int,
        alpha_init: float = 1.0,
        beta_init: float = 1.0,
    ):
        super().__init__()
        self.channels = channels
        self.alpha = mx.ones((channels,)) * alpha_init
        self.beta = mx.ones((channels,)) * beta_init

    def __call__(self, x: mx.array) -> mx.array:
        """Apply SnakeBeta activation.

        Args:
            x: Input tensor.

        Returns:
            Activated tensor.
        """
        alpha = self.alpha
        beta = self.beta

        while alpha.ndim < x.ndim:
            alpha = alpha[None, ...]
            beta = beta[None, ...]

        sin_term = mx.sin(alpha * x)
        return x + (1.0 / (beta + 1e-8)) * (sin_term * sin_term)


def snake(x: mx.array, alpha: float = 1.0) -> mx.array:
    """Functional Snake activation.

    Args:
        x: Input tensor.
        alpha: Frequency parameter.

    Returns:
        Activated tensor.
    """
    sin_term = mx.sin(alpha * x)
    return x + (1.0 / alpha) * (sin_term * sin_term)


class PReLU(nn.Module):
    """Parametric ReLU activation.

    Defined as: max(0, x) + α * min(0, x)

    Args:
        num_parameters: Number of learnable parameters.
        init: Initial value for negative slope.
    """

    def __init__(self, num_parameters: int = 1, init: float = 0.25):
        super().__init__()
        self.weight = mx.ones((num_parameters,)) * init

    def __call__(self, x: mx.array) -> mx.array:
        """Apply PReLU.

        Args:
            x: Input tensor.

        Returns:
            Activated tensor.
        """
        weight = self.weight
        while weight.ndim < x.ndim:
            weight = weight[None, ...]

        return mx.maximum(0, x) + weight * mx.minimum(0, x)


class Swish(nn.Module):
    """Swish (SiLU) activation.

    Defined as: x * sigmoid(x)
    """

    def __call__(self, x: mx.array) -> mx.array:
        """Apply Swish activation.

        Args:
            x: Input tensor.

        Returns:
            Activated tensor.
        """
        return x * mx.sigmoid(x)
