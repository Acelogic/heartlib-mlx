"""Causal convolution and weight-normalized convolution layers for MLX."""

from typing import Optional

import mlx.core as mx
import mlx.nn as nn


class CausalConv1d(nn.Module):
    """1D Causal Convolution with left padding.

    This ensures the output at time t only depends on inputs at time <= t.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Size of the convolutional kernel.
        stride: Stride of the convolution.
        dilation: Dilation factor.
        groups: Number of groups for grouped convolution.
        bias: Whether to include a bias term.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups

        # Calculate padding for causal convolution
        self.padding = (kernel_size - 1) * dilation

        # MLX Conv1d: weight shape is (out_channels, kernel_size, in_channels // groups)
        scale = 1.0 / (in_channels * kernel_size) ** 0.5
        self.weight = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(out_channels, kernel_size, in_channels // groups),
        )

        if bias:
            self.bias = mx.zeros((out_channels,))
        else:
            self.bias = None

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, length, channels).

        Returns:
            Output tensor of shape (batch, length, out_channels).
        """
        # Apply left padding for causal convolution
        if self.padding > 0:
            # Pad on the left (time dimension)
            x = mx.pad(x, [(0, 0), (self.padding, 0), (0, 0)])

        # Perform convolution
        y = mx.conv1d(
            x,
            self.weight,
            stride=self.stride,
            padding=0,  # We already applied causal padding
            dilation=self.dilation,
            groups=self.groups,
        )

        if self.bias is not None:
            y = y + self.bias

        return y


class WeightNormConv1d(nn.Module):
    """1D Convolution with weight normalization.

    Implements weight normalization as: w = g * (v / ||v||)
    where g is the magnitude and v is the direction.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Size of the convolutional kernel.
        stride: Stride of the convolution.
        padding: Padding to apply.
        dilation: Dilation factor.
        groups: Number of groups for grouped convolution.
        bias: Whether to include a bias term.
        causal: Whether to use causal (left) padding.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        causal: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.causal = causal

        # For causal convolution, override padding
        if causal:
            self.causal_padding = (kernel_size - 1) * dilation
        else:
            self.causal_padding = 0

        # Weight normalization: weight = g * (v / ||v||)
        # v is the direction vector
        scale = 1.0 / (in_channels * kernel_size) ** 0.5
        self.weight_v = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(out_channels, kernel_size, in_channels // groups),
        )

        # g is the magnitude (per output channel)
        # Initialize to the norm of v
        v_norm = mx.sqrt(mx.sum(self.weight_v ** 2, axis=(1, 2), keepdims=True))
        self.weight_g = v_norm.squeeze((1, 2))

        if bias:
            self.bias = mx.zeros((out_channels,))
        else:
            self.bias = None

    def _get_normalized_weight(self) -> mx.array:
        """Compute the weight-normalized weight tensor."""
        # Normalize v
        v_norm = mx.sqrt(mx.sum(self.weight_v ** 2, axis=(1, 2), keepdims=True) + 1e-8)
        v_normalized = self.weight_v / v_norm

        # Scale by g
        weight = self.weight_g[:, None, None] * v_normalized
        return weight

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, length, channels).

        Returns:
            Output tensor of shape (batch, length, out_channels).
        """
        weight = self._get_normalized_weight()

        # Apply causal padding if needed
        if self.causal and self.causal_padding > 0:
            x = mx.pad(x, [(0, 0), (self.causal_padding, 0), (0, 0)])
            padding = 0
        else:
            padding = self.padding

        # Perform convolution
        y = mx.conv1d(
            x,
            weight,
            stride=self.stride,
            padding=padding,
            dilation=self.dilation,
            groups=self.groups,
        )

        if self.bias is not None:
            y = y + self.bias

        return y


class WeightNormConvTranspose1d(nn.Module):
    """1D Transposed Convolution with weight normalization.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Size of the convolutional kernel.
        stride: Stride of the convolution.
        padding: Padding to apply.
        output_padding: Additional padding on the output.
        dilation: Dilation factor.
        groups: Number of groups for grouped convolution.
        bias: Whether to include a bias term.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        output_padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation
        self.groups = groups

        # Weight normalization parameters
        # For conv_transpose: we use weight shape (out_channels, kernel_size, in_channels // groups)
        # because we'll transpose it for the backwards conv operation
        scale = 1.0 / (in_channels * kernel_size) ** 0.5
        self.weight_v = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(out_channels, kernel_size, in_channels // groups),
        )

        # g is the magnitude (per output channel)
        v_norm = mx.sqrt(mx.sum(self.weight_v ** 2, axis=(1, 2), keepdims=True))
        self.weight_g = v_norm.squeeze((1, 2))

        if bias:
            self.bias = mx.zeros((out_channels,))
        else:
            self.bias = None

    def _get_normalized_weight(self) -> mx.array:
        """Compute the weight-normalized weight tensor."""
        v_norm = mx.sqrt(mx.sum(self.weight_v ** 2, axis=(1, 2), keepdims=True) + 1e-8)
        v_normalized = self.weight_v / v_norm
        weight = self.weight_g[:, None, None] * v_normalized
        return weight

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, length, channels).

        Returns:
            Output tensor of shape (batch, output_length, out_channels).
        """
        batch_size, seq_len, channels = x.shape
        weight = self._get_normalized_weight()

        # Compute output length
        output_len = (seq_len - 1) * self.stride - 2 * self.padding + self.kernel_size + self.output_padding

        # For transposed convolution, we:
        # 1. Insert zeros between input elements (upsampling)
        # 2. Apply convolution with flipped, transposed weight

        # Insert zeros between input samples (upsample) BEFORE padding
        if self.stride > 1:
            # Upsample by interleaving with zeros
            # [x0, x1, x2] with stride=2 -> [x0, 0, x1, 0, x2]
            # Approach: reshape and concatenate
            x_expanded = x[:, :, None, :]  # (batch, seq_len, 1, channels)
            zeros = mx.zeros((batch_size, seq_len, self.stride - 1, channels))
            interleaved = mx.concatenate([x_expanded, zeros], axis=2)  # (batch, seq_len, stride, channels)
            x = interleaved.reshape(batch_size, seq_len * self.stride, channels)
            # Trim trailing zeros: we want seq_len + (seq_len-1)*(stride-1) = seq_len*stride - stride + 1
            upsampled_len = seq_len * self.stride - self.stride + 1
            x = x[:, :upsampled_len, :]

        # For transposed conv, flip the kernel along the spatial dimension
        # Weight shape: (out_channels, kernel_size, in_channels // groups)
        # Keep the same shape, just flip the kernel
        weight_flipped = weight[:, ::-1, :]

        # Pad for transposed convolution
        pad_amount = self.kernel_size - 1 - self.padding
        if pad_amount > 0:
            x = mx.pad(x, [(0, 0), (pad_amount, pad_amount), (0, 0)])

        # Regular convolution with flipped weight
        y = mx.conv1d(
            x,
            weight_flipped,
            stride=1,
            padding=0,
            dilation=self.dilation,
            groups=self.groups,
        )

        # Handle output padding
        if self.output_padding > 0:
            y = mx.pad(y, [(0, 0), (0, self.output_padding), (0, 0)])

        # Adjust output size if needed
        if y.shape[1] > output_len:
            y = y[:, :output_len, :]
        elif y.shape[1] < output_len:
            pad_size = output_len - y.shape[1]
            y = mx.pad(y, [(0, 0), (0, pad_size), (0, 0)])

        if self.bias is not None:
            y = y + self.bias

        return y


class Conv1d(nn.Module):
    """Standard 1D Convolution layer.

    Wrapper around MLX's conv1d with a familiar interface.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Size of the convolutional kernel.
        stride: Stride of the convolution.
        padding: Padding to apply.
        dilation: Dilation factor.
        groups: Number of groups for grouped convolution.
        bias: Whether to include a bias term.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        scale = 1.0 / (in_channels * kernel_size) ** 0.5
        self.weight = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(out_channels, kernel_size, in_channels // groups),
        )

        if bias:
            self.bias = mx.zeros((out_channels,))
        else:
            self.bias = None

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, length, channels).

        Returns:
            Output tensor of shape (batch, output_length, out_channels).
        """
        y = mx.conv1d(
            x,
            self.weight,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )

        if self.bias is not None:
            y = y + self.bias

        return y
