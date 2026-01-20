"""HeartCodec - Neural Audio Codec with Flow Matching Decoder."""

from typing import Optional, Union
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from heartlib_mlx.heartcodec.configuration import HeartCodecConfig
from heartlib_mlx.heartcodec.scalar_codec import ScalarModel
from heartlib_mlx.heartcodec.flow_matching import FlowMatchingDecoder


class HeartCodec(nn.Module):
    """HeartCodec: Neural Audio Codec with Flow Matching Decoder.

    HeartCodec is a 12.5Hz neural audio codec that combines:
    1. ScalarModel: Convolutional encoder/decoder for audio
    2. FlowMatchingDecoder: Transformer-based generative decoder

    The codec operates at 48kHz sample rate with a frame rate of
    12.5Hz (3840 samples per frame).

    Args:
        config: HeartCodecConfig with model hyperparameters.
    """

    def __init__(self, config: HeartCodecConfig):
        super().__init__()
        self.config = config

        # Scalar codec for audio encoding/decoding
        self.scalar_model = ScalarModel(
            num_bands=config.num_bands,
            sample_rate=config.sample_rate,
            causal=config.causal,
            downsample_factors=config.downsample_factors,
            downsample_kernel_sizes=config.downsample_kernel_sizes,
            upsample_factors=config.upsample_factors,
            upsample_kernel_sizes=config.upsample_kernel_sizes,
            latent_hidden_dim=config.latent_hidden_dim,
            default_kernel_size=config.default_kernel_size,
            delay_kernel_size=config.delay_kernel_size,
            init_channel=config.init_channel,
            res_kernel_size=config.res_kernel_size,
        )

        # Flow matching decoder for high-quality synthesis
        self.flow_matching = FlowMatchingDecoder(
            dim=config.dim,
            codebook_size=config.codebook_size,
            codebook_dim=config.codebook_dim,
            num_quantizers=config.num_quantizers,
            attention_head_dim=config.attention_head_dim,
            in_channels=config.in_channels,
            num_attention_heads=config.num_attention_heads,
            num_layers=config.num_layers,
            num_layers_2=config.num_layers_2,
            out_channels=config.out_channels,
            use_cosine_sim=config.use_cosine_sim,
            decay=config.decay,
            commitment_weight=config.commitment_weight,
            threshold_ema_dead_code=config.threshold_ema_dead_code,
        )

    def encode(self, audio: mx.array) -> mx.array:
        """Encode audio to quantized latent representation.

        Args:
            audio: Audio waveform of shape (batch, samples) or (batch, samples, 1).

        Returns:
            Quantized latent of shape (batch, frames, latent_dim).
        """
        return self.scalar_model.encode(audio)

    def decode(self, latent: mx.array) -> mx.array:
        """Decode quantized latent to audio waveform.

        Args:
            latent: Quantized latent of shape (batch, frames, latent_dim).

        Returns:
            Audio waveform of shape (batch, samples, 1).
        """
        return self.scalar_model.decode(latent)

    def detokenize(
        self,
        codes: mx.array,
        duration: float = 29.76,
        num_steps: int = 10,
        guidance_scale: float = 1.25,
    ) -> mx.array:
        """Convert discrete codes back to audio waveform.

        This is the main inference method for audio generation.
        It uses the flow matching decoder to generate high-quality
        latents from the codes, then decodes to audio.

        Args:
            codes: Audio codes of shape (batch, frames, num_quantizers).
            duration: Target duration in seconds.
            num_steps: Number of ODE integration steps.
            guidance_scale: Classifier-free guidance scale (1.0 = no guidance).

        Returns:
            Audio waveform of shape (batch, samples, 1).
        """
        batch_size = codes.shape[0]
        frame_rate = self.config.frame_rate
        target_frames = int(duration * frame_rate)

        # Pad or truncate codes to target length
        current_frames = codes.shape[1]
        if current_frames < target_frames:
            # Pad with zeros
            pad_frames = target_frames - current_frames
            padding = mx.zeros((batch_size, pad_frames, codes.shape[2]), dtype=codes.dtype)
            codes = mx.concatenate([codes, padding], axis=1)
        elif current_frames > target_frames:
            # Truncate
            codes = codes[:, :target_frames, :]

        # Generate latents using flow matching
        # Output shape: (batch, seq_len * 2, out_channels) where out_channels=256
        latents = self.flow_matching(
            codes=codes,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
        )

        # Reshape latents for scalar_model decoder:
        # (batch, T, 256) -> (batch, T, 2, 128) -> (batch, 2, T, 128) -> (batch*2, T, 128)
        # This splits the 256-dim output into 2 parallel 128-dim streams
        bsz, t, f = latents.shape
        latents = latents.reshape(bsz, t, 2, f // 2)  # (batch, T, 2, 128)
        latents = latents.transpose(0, 2, 1, 3)  # (batch, 2, T, 128)
        latents = latents.reshape(bsz * 2, t, f // 2)  # (batch*2, T, 128)

        # Decode latents to audio
        # Input: (batch*2, T, 128) -> Output: (batch*2, samples, 1)
        audio = self.scalar_model.decode(latents)

        # Merge the 2 streams back
        # (batch*2, samples, 1) -> (batch, 2, samples, 1)
        samples = audio.shape[1]
        audio = audio.reshape(bsz, 2, samples, 1)

        # Average the two streams (simple merge strategy)
        audio = mx.mean(audio, axis=1)

        return audio

    def tokenize(self, audio: mx.array) -> mx.array:
        """Encode audio to discrete codes.

        Args:
            audio: Audio waveform of shape (batch, samples) or (batch, samples, 1).

        Returns:
            Audio codes of shape (batch, frames, num_quantizers).
        """
        # Get quantized latent
        latent = self.encode(audio)

        # Quantize to codes using flow matching's VQ
        codes = self.flow_matching.vq.encode(latent)

        return codes

    @classmethod
    def from_pretrained(
        cls,
        path: Union[str, Path],
        dtype: mx.Dtype = mx.bfloat16,
    ) -> "HeartCodec":
        """Load a pretrained HeartCodec model.

        Args:
            path: Path to the model directory.
            dtype: Data type for model weights.

        Returns:
            HeartCodec instance with loaded weights.
        """
        path = Path(path)

        # Load config
        config = HeartCodecConfig.from_pretrained(path)

        # Create model
        model = cls(config)

        # Load weights using MLX's native loader (handles bfloat16 properly)
        weights_path = path / "model.safetensors"
        if weights_path.exists():
            weights = mx.load(str(weights_path))

            # Convert to target dtype if different
            weights = {k: v.astype(dtype) for k, v in weights.items()}

            # Load into model (strict=False to ignore PreProcessor/PostProcessor weights)
            model.load_weights(list(weights.items()), strict=False)
            mx.eval(model.parameters())

        return model

    def save_pretrained(self, path: Union[str, Path]) -> None:
        """Save the model to a directory.

        Args:
            path: Path to save the model.
        """
        from safetensors.numpy import save_file
        import numpy as np

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save config
        self.config.save_pretrained(path)

        # Save weights
        weights = dict(self.parameters())
        # Convert to numpy for safetensors
        np_weights = {k: np.array(v) for k, v in weights.items()}
        save_file(np_weights, str(path / "model.safetensors"))

    def __call__(
        self,
        audio: Optional[mx.array] = None,
        codes: Optional[mx.array] = None,
        num_steps: int = 10,
        guidance_scale: float = 1.25,
    ) -> mx.array:
        """Forward pass for encoding or decoding.

        Args:
            audio: Input audio for encoding (optional).
            codes: Input codes for decoding (optional).
            num_steps: ODE integration steps for decoding.
            guidance_scale: CFG scale for decoding.

        Returns:
            Encoded codes (if audio provided) or decoded audio (if codes provided).
        """
        if audio is not None:
            return self.tokenize(audio)
        elif codes is not None:
            return self.detokenize(codes, num_steps=num_steps, guidance_scale=guidance_scale)
        else:
            raise ValueError("Either audio or codes must be provided")
