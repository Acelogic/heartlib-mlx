"""Scalar Codec (SQ) encoder/decoder for HeartCodec."""

from typing import List, Tuple, Optional

import mlx.core as mx
import mlx.nn as nn

from heartlib_mlx.nn.conv import CausalConv1d, WeightNormConv1d, WeightNormConvTranspose1d
from heartlib_mlx.heartcodec.activations import Snake, PReLU
from heartlib_mlx.heartcodec.quantizer import ScalarQuantizer


class ResidualUnit(nn.Module):
    """Residual unit with dilated convolutions.

    Uses weight-normalized convolutions with PReLU activation and
    dilated receptive fields for efficient context modeling.

    Matches PyTorch HeartCodec structure with two separate PReLU activations.

    Args:
        channels: Number of input/output channels.
        kernel_size: Convolution kernel size.
        dilation: Dilation factor.
        causal: Whether to use causal convolutions.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 7,
        dilation: int = 1,
        causal: bool = True,
    ):
        super().__init__()

        # Dilated conv
        self.conv1 = WeightNormConv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            dilation=dilation,
            causal=causal,
        )

        # Pointwise conv
        self.conv2 = WeightNormConv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=1,
            causal=False,
        )

        # Two separate PReLU with single parameter (matches PyTorch)
        self.activation = PReLU(1)
        self.activation2 = PReLU(1)

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass with residual connection.

        Args:
            x: Input of shape (batch, length, channels).

        Returns:
            Output with same shape.
        """
        residual = x
        x = self.activation(self.conv1(x))
        x = self.activation2(self.conv2(x))
        return x + residual


class ResEncoderBlock(nn.Module):
    """Encoder block with residual units and downsampling.

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
        kernel_size: Residual unit kernel size.
        stride: Downsampling stride.
        causal: Whether to use causal convolutions.
        num_residual: Number of residual units.
        dilations: Dilation factors for residual units.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 7,
        stride: int = 4,
        causal: bool = True,
        num_residual: int = 5,
        dilations: Optional[List[int]] = None,
    ):
        super().__init__()

        dilations = dilations or [1, 3, 5, 7, 9]

        # Residual units
        self.residual_units = [
            ResidualUnit(
                channels=in_channels,
                kernel_size=kernel_size,
                dilation=dilations[i % len(dilations)],
                causal=causal,
            )
            for i in range(num_residual)
        ]

        # Downsampling convolution
        self.downsample = WeightNormConv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=stride * 2,
            stride=stride,
            causal=causal,
        )

        # Single-parameter PReLU (matches PyTorch)
        self.activation = PReLU(1)

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass.

        Args:
            x: Input of shape (batch, length, in_channels).

        Returns:
            Downsampled output of shape (batch, length // stride, out_channels).
        """
        for unit in self.residual_units:
            x = unit(x)

        x = self.activation(self.downsample(x))
        return x


class ResDecoderBlock(nn.Module):
    """Decoder block with upsampling and residual units.

    Note: PyTorch version has activation=None in UpsampleLayer,
    so no activation is applied after upsampling.

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
        kernel_size: Residual unit kernel size.
        stride: Upsampling stride.
        causal: Whether to use causal convolutions.
        num_residual: Number of residual units.
        dilations: Dilation factors for residual units.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 7,
        stride: int = 4,
        causal: bool = True,
        num_residual: int = 5,
        dilations: Optional[List[int]] = None,
    ):
        super().__init__()

        dilations = dilations or [1, 3, 5, 7, 9]

        # Upsampling convolution (transposed)
        # Note: PyTorch uses activation=None here
        self.upsample = WeightNormConvTranspose1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=stride * 2,
            stride=stride,
            padding=stride // 2,
        )

        # No activation after upsample (matches PyTorch)

        # Residual units
        self.residual_units = [
            ResidualUnit(
                channels=out_channels,
                kernel_size=kernel_size,
                dilation=dilations[i % len(dilations)],
                causal=causal,
            )
            for i in range(num_residual)
        ]

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass.

        Args:
            x: Input of shape (batch, length, in_channels).

        Returns:
            Upsampled output of shape (batch, length * stride, out_channels).
        """
        # No activation after upsample (matches PyTorch)
        x = self.upsample(x)

        for unit in self.residual_units:
            x = unit(x)

        return x


class ScalarModel(nn.Module):
    """Scalar Codec model for audio encoding/decoding.

    Encodes audio waveforms to quantized latent representations
    and decodes them back to waveforms.

    Args:
        num_bands: Number of frequency bands (default: 1 for raw audio).
        sample_rate: Audio sample rate in Hz.
        causal: Whether to use causal convolutions.
        downsample_factors: Downsampling factors per encoder stage.
        downsample_kernel_sizes: Kernel sizes for downsampling.
        upsample_factors: Upsampling factors per decoder stage.
        upsample_kernel_sizes: Kernel sizes for upsampling.
        latent_hidden_dim: Hidden dimension in latent space.
        default_kernel_size: Default convolution kernel size.
        init_channel: Initial channel count.
        res_kernel_size: Residual block kernel size.
    """

    def __init__(
        self,
        num_bands: int = 1,
        sample_rate: int = 48000,
        causal: bool = True,
        downsample_factors: Optional[List[int]] = None,
        downsample_kernel_sizes: Optional[List[int]] = None,
        upsample_factors: Optional[List[int]] = None,
        upsample_kernel_sizes: Optional[List[int]] = None,
        latent_hidden_dim: int = 128,
        default_kernel_size: int = 7,
        delay_kernel_size: int = 5,
        init_channel: int = 64,
        res_kernel_size: int = 7,
    ):
        super().__init__()

        downsample_factors = downsample_factors or [4, 4, 8, 8]
        downsample_kernel_sizes = downsample_kernel_sizes or [8, 8, 16, 16]
        upsample_factors = upsample_factors or [8, 8, 4, 4]
        upsample_kernel_sizes = upsample_kernel_sizes or [16, 16, 8, 8]

        self.num_bands = num_bands
        self.sample_rate = sample_rate
        self.causal = causal

        # Calculate channel progression
        channels = [init_channel]
        for _ in downsample_factors:
            channels.append(channels[-1] * 2)

        # ========== Encoder ==========
        # Initial convolution
        self.encoder_in = WeightNormConv1d(
            in_channels=num_bands,
            out_channels=init_channel,
            kernel_size=default_kernel_size,
            causal=causal,
        )

        # Encoder blocks
        self.encoder_blocks = []
        for i, (stride, kernel_size) in enumerate(zip(downsample_factors, downsample_kernel_sizes)):
            self.encoder_blocks.append(
                ResEncoderBlock(
                    in_channels=channels[i],
                    out_channels=channels[i + 1],
                    kernel_size=res_kernel_size,
                    stride=stride,
                    causal=causal,
                )
            )

        # Final encoder conv to latent
        self.encoder_out = WeightNormConv1d(
            in_channels=channels[-1],
            out_channels=latent_hidden_dim,
            kernel_size=default_kernel_size,
            causal=causal,
        )

        # Scalar quantizer
        self.quantizer = ScalarQuantizer()

        # ========== Decoder ==========
        # Initial decoder conv from latent (uses delay_kernel_size per PyTorch)
        self.decoder_in = WeightNormConv1d(
            in_channels=latent_hidden_dim,
            out_channels=channels[-1],
            kernel_size=delay_kernel_size,
            causal=False,  # Note: PyTorch decoder uses non-causal here
        )

        # Decoder blocks (reverse order)
        self.decoder_blocks = []
        for i, (stride, kernel_size) in enumerate(zip(upsample_factors, upsample_kernel_sizes)):
            in_ch = channels[-(i + 1)]
            out_ch = channels[-(i + 2)]
            self.decoder_blocks.append(
                ResDecoderBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=res_kernel_size,
                    stride=stride,
                    causal=causal,
                )
            )

        # Final decoder conv
        self.decoder_out = WeightNormConv1d(
            in_channels=init_channel,
            out_channels=num_bands,
            kernel_size=default_kernel_size,
            causal=causal,
        )

        # Single-parameter PReLU (matches PyTorch)
        self.activation = PReLU(1)
        self.tanh = lambda x: mx.tanh(x)

    def encode(self, x: mx.array) -> mx.array:
        """Encode audio to quantized latent.

        Args:
            x: Audio waveform of shape (batch, samples, bands) or (batch, samples).

        Returns:
            Quantized latent of shape (batch, frames, latent_dim).
        """
        # Ensure 3D input
        if x.ndim == 2:
            x = x[:, :, None]

        # Encoder
        x = self.encoder_in(x)
        for block in self.encoder_blocks:
            x = block(x)
        x = self.encoder_out(x)

        # Apply tanh and quantize
        x = self.tanh(x)
        quantized, _ = self.quantizer(x)

        return quantized

    def decode(self, z: mx.array) -> mx.array:
        """Decode quantized latent to audio.

        Args:
            z: Quantized latent of shape (batch, frames, latent_dim).

        Returns:
            Audio waveform of shape (batch, samples, bands).
        """
        x = self.decoder_in(z)
        for block in self.decoder_blocks:
            x = block(x)
        x = self.activation(x)
        x = self.decoder_out(x)
        x = self.tanh(x)

        return x

    def __call__(
        self,
        x: mx.array,
    ) -> Tuple[mx.array, mx.array, mx.array]:
        """Forward pass: encode, quantize, decode.

        Args:
            x: Audio waveform of shape (batch, samples, bands).

        Returns:
            Tuple of (reconstruction, embedding, quantized_embedding).
        """
        # Ensure 3D
        if x.ndim == 2:
            x = x[:, :, None]

        # Encode
        h = self.encoder_in(x)
        for block in self.encoder_blocks:
            h = block(h)
        h = self.encoder_out(h)
        embedding = self.tanh(h)

        # Quantize
        quantized, _ = self.quantizer(embedding)

        # Decode
        out = self.decode(quantized)

        return out, embedding, quantized

    def inference(self, x: mx.array) -> Tuple[mx.array, mx.array, mx.array]:
        """Inference mode: encode, quantize, decode.

        Args:
            x: Audio waveform.

        Returns:
            Tuple of (embedding, quantized, reconstruction).
        """
        _, embedding, quantized = self(x)
        reconstruction = self.decode(quantized)
        return embedding, quantized, reconstruction
