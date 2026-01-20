"""Flow Matching Decoder for HeartCodec - matches PyTorch architecture."""

from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from heartlib_mlx.nn.transformer import RMSNorm, LlamaAttention, LlamaMLP
from heartlib_mlx.heartcodec.quantizer import ResidualVQ
from heartlib_mlx.ode.solver import euler_solve


class FFNBlock(nn.Module):
    """FFN projection block with Conv1d + Linear.

    PyTorch uses this pattern for proj_in, proj_out, connection_proj.

    Args:
        in_features: Input features.
        out_features: Output features.
        hidden_features: Hidden dimension (for ffn_2).
        kernel_size: Conv1d kernel size.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: Optional[int] = None,
        kernel_size: int = 3,
    ):
        super().__init__()
        hidden_features = hidden_features or out_features
        self.kernel_size = kernel_size

        # Conv1d with kernel_size (no dilation)
        # PyTorch weight: (out, in, k)
        # MLX weight: (out, k, in) - we'll handle in conversion
        padding = kernel_size // 2
        self.ffn_1 = nn.Conv1d(
            in_channels=in_features,
            out_channels=hidden_features,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.ffn_2 = nn.Linear(hidden_features, out_features)

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass.

        Args:
            x: Input of shape (batch, seq_len, in_features).

        Returns:
            Output of shape (batch, seq_len, out_features).
        """
        # Conv1d expects (batch, seq, channels)
        x = self.ffn_1(x)
        # Apply scaling factor (matches PyTorch's ProjectLayer: x * kernel_size**-0.5)
        x = x * (self.kernel_size ** -0.5)
        # NOTE: PyTorch's ProjectLayer has NO activation between conv and linear
        x = self.ffn_2(x)
        return x


class TimestepEmbedder(nn.Module):
    """Timestep embedding with sinusoidal encoding + MLP.

    Args:
        hidden_size: Output dimension.
        frequency_embedding_size: Sinusoidal embedding dimension.
    """

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 512):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.linear_1 = nn.Linear(frequency_embedding_size, hidden_size, bias=True)
        self.linear_2 = nn.Linear(hidden_size, hidden_size, bias=True)

    def _timestep_embedding(self, t: mx.array, max_period: float = 10000.0, scale: float = 1000.0) -> mx.array:
        """Create sinusoidal timestep embeddings.

        Matches PyTorch's timestep_embedding with scale=1000 default.
        """
        half_dim = self.frequency_embedding_size // 2
        freqs = mx.exp(
            -mx.log(mx.array(max_period)) * mx.arange(half_dim) / half_dim
        )
        # Note: PyTorch multiplies by scale=1000, critical for correct embeddings!
        args = t[:, None] * freqs[None, :] * scale
        embedding = mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)
        return embedding

    def __call__(self, t: mx.array) -> mx.array:
        """Forward pass.

        Args:
            t: Timestep of shape (batch,).

        Returns:
            Embedding of shape (batch, hidden_size).
        """
        t_emb = self._timestep_embedding(t)
        t_emb = self.linear_1(t_emb)
        t_emb = nn.silu(t_emb)
        t_emb = self.linear_2(t_emb)
        return t_emb


class AdaLNSingle(nn.Module):
    """Adaptive LayerNorm Single for flow matching.

    Projects timestep embedding to scale/shift for all blocks.

    Args:
        dim: Model dimension.
        num_outputs: Number of output values (6 = shift1, scale1, gate1, shift2, scale2, gate2).
    """

    def __init__(self, dim: int, num_outputs: int = 6):
        super().__init__()
        self.emb = nn.Module()
        self.emb.timestep_embedder = TimestepEmbedder(dim)
        self.linear = nn.Linear(dim, num_outputs * dim, bias=True)

    def __call__(self, t: mx.array) -> tuple[mx.array, mx.array]:
        """Forward pass.

        Args:
            t: Timestep of shape (batch,).

        Returns:
            Tuple of:
            - conditioning: (batch, num_outputs, dim) for transformer blocks
            - embedded_timestep: (batch, dim) for scale_shift_table modulation
        """
        t_emb = self.emb.timestep_embedder(t)
        # Apply silu before linear (matches PyTorch: linear(silu(embedded_timestep)))
        conditioning = self.linear(nn.silu(t_emb))
        # Reshape to (batch, num_outputs, dim)
        batch_size = conditioning.shape[0]
        dim = t_emb.shape[-1]
        num_outputs = conditioning.shape[-1] // dim
        conditioning = conditioning.reshape(batch_size, num_outputs, dim)
        return conditioning, t_emb


class FlowMatchingTransformerBlock(nn.Module):
    """Transformer block with scale_shift_table for flow matching.

    Uses per-block learnable scale/shift that combines with timestep embedding.

    Args:
        dim: Model dimension.
        n_heads: Number of attention heads.
        head_dim: Dimension per head.
        mlp_hidden_dim: MLP hidden dimension.
        norm_eps: Epsilon for RMSNorm.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        head_dim: int = 64,
        mlp_hidden_dim: Optional[int] = None,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.dim = dim
        mlp_hidden_dim = mlp_hidden_dim or int(dim * 8 / 3)  # Default SwiGLU ratio

        # Norms
        self.attn_norm = RMSNorm(dim, eps=norm_eps)
        self.mlp_norm = RMSNorm(dim, eps=norm_eps)

        # Attention
        self.attn = LlamaAttention(
            dim=dim,
            n_heads=n_heads,
            head_dim=head_dim,
            bias=False,
        )

        # MLP
        self.mlp = LlamaMLP(dim=dim, hidden_dim=mlp_hidden_dim)

        # Per-block scale/shift table: (6, dim)
        # [shift1, scale1, gate1, shift2, scale2, gate2]
        self.scale_shift_table = mx.zeros((6, dim))

    def __call__(
        self,
        x: mx.array,
        adaln_cond: mx.array,
        mask: Optional[mx.array] = None,
    ) -> mx.array:
        """Forward pass.

        Args:
            x: Input of shape (batch, seq_len, dim).
            adaln_cond: AdaLN conditioning of shape (batch, 6, dim).
            mask: Optional attention mask.

        Returns:
            Output of shape (batch, seq_len, dim).
        """
        # Combine per-block table with shared conditioning
        # scale_shift_table: (6, dim), adaln_cond: (batch, 6, dim)
        cond = self.scale_shift_table[None, :, :] + adaln_cond  # (batch, 6, dim)

        # Split into components
        shift1 = cond[:, 0:1, :]  # (batch, 1, dim)
        scale1 = cond[:, 1:2, :]
        gate1 = cond[:, 2:3, :]
        shift2 = cond[:, 3:4, :]
        scale2 = cond[:, 4:5, :]
        gate2 = cond[:, 5:6, :]

        # Attention with adaptive norm
        h = self.attn_norm(x)
        h = h * (1 + scale1) + shift1
        attn_out, _ = self.attn(h, mask=mask)
        x = x + gate1 * attn_out

        # MLP with adaptive norm
        h = self.mlp_norm(x)
        h = h * (1 + scale2) + shift2
        x = x + gate2 * self.mlp(h)

        return x


class LlamaTransformerForFlowMatching(nn.Module):
    """Two-stage transformer for flow matching velocity prediction.

    Stage 1: 24 layers at dim=1536
    Stage 2: 6 layers at dim=3072 (doubled)

    Args:
        dim: Stage 1 dimension (1536).
        dim_2: Stage 2 dimension (3072).
        n_heads: Number of attention heads for stage 1.
        n_heads_2: Number of attention heads for stage 2.
        head_dim: Dimension per head.
        num_layers: Layers in stage 1.
        num_layers_2: Layers in stage 2.
        in_channels: Input conditioning dimension.
        out_channels: Output velocity dimension.
        mlp_hidden_dim: MLP hidden for stage 1.
        mlp_hidden_dim_2: MLP hidden for stage 2.
    """

    def __init__(
        self,
        dim: int = 1536,
        dim_2: int = 3072,
        n_heads: int = 24,
        n_heads_2: int = 48,
        head_dim: int = 64,
        num_layers: int = 24,
        num_layers_2: int = 6,
        in_channels: int = 1024,
        out_channels: int = 256,
        mlp_hidden_dim: int = 4096,
        mlp_hidden_dim_2: int = 8192,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.dim = dim
        self.dim_2 = dim_2

        # Input projection: FFN block with Conv1d
        self.proj_in = FFNBlock(in_channels, dim, hidden_features=dim)

        # Stage 1 time embedding
        self.adaln_single = AdaLNSingle(dim, num_outputs=6)

        # Stage 1 transformer blocks
        self.transformer_blocks = [
            FlowMatchingTransformerBlock(
                dim=dim,
                n_heads=n_heads,
                head_dim=head_dim,
                mlp_hidden_dim=mlp_hidden_dim,
                norm_eps=norm_eps,
            )
            for _ in range(num_layers)
        ]

        # Output norm for stage 1 (LayerNorm without affine, matches PyTorch)
        self.norm_out = nn.LayerNorm(dim, eps=norm_eps, affine=False)

        # Scale/shift for final stage 1 output
        self.scale_shift_table = mx.zeros((2, dim))

        # Connection projection: stage 1 output + latent -> stage 2 input
        # PyTorch has in_features=2560 which is 1536 + 1024
        # Actually looking at the shapes, it seems to be 1536 + out_channels*4
        # Let's use 1536 + 1024 = 2560
        self.connection_proj = FFNBlock(dim + in_channels, dim_2, hidden_features=dim_2)

        # Stage 2 time embedding
        self.adaln_single_2 = AdaLNSingle(dim_2, num_outputs=6)

        # Stage 2 transformer blocks
        self.transformer_blocks_2 = [
            FlowMatchingTransformerBlock(
                dim=dim_2,
                n_heads=n_heads_2,
                head_dim=head_dim,
                mlp_hidden_dim=mlp_hidden_dim_2,
                norm_eps=norm_eps,
            )
            for _ in range(num_layers_2)
        ]

        # Output norm for stage 2 (LayerNorm without affine, matches PyTorch)
        self.norm_out_2 = nn.LayerNorm(dim_2, eps=norm_eps, affine=False)

        # Scale/shift for final stage 2 output
        self.scale_shift_table_2 = mx.zeros((2, dim_2))

        # Output projection
        self.proj_out = FFNBlock(dim_2, out_channels, hidden_features=out_channels)

    def __call__(
        self,
        t: mx.array,
        hidden_states: mx.array,
    ) -> mx.array:
        """Forward pass to predict velocity.

        Args:
            t: Timestep of shape (batch,) or (1,).
            hidden_states: Concatenated input of shape (batch, seq_len, in_channels).
                           This is [x, incontext_x, mu] concatenated along channels:
                           - x: (batch, seq, 256) noisy latent
                           - incontext_x: (batch, seq, 256) context latent
                           - mu: (batch, seq, 512) VQ embeddings
                           Total: 256 + 256 + 512 = 1024

        Returns:
            Predicted velocity of shape (batch, seq_len, out_channels).
        """
        batch_size = hidden_states.shape[0]

        # Expand t if needed
        if t.shape[0] == 1 and batch_size > 1:
            t = mx.broadcast_to(t, (batch_size,))

        # Project concatenated hidden_states to stage 1 dim
        s = self.proj_in(hidden_states)

        # Stage 1 processing
        adaln_cond, embedded_timestep = self.adaln_single(t)  # (batch, 6, dim), (batch, dim)
        for block in self.transformer_blocks:
            s = block(s, adaln_cond)

        # Apply final stage 1 norm and scale/shift (matches PyTorch)
        # PyTorch: shift, scale = (scale_shift_table[None] + embedded_timestep[:, None]).chunk(2, dim=1)
        s = self.norm_out(s)
        # Combine scale_shift_table with embedded_timestep: (1, 2, dim) + (batch, 1, dim) -> (batch, 2, dim)
        combined = self.scale_shift_table[None, :, :] + embedded_timestep[:, None, :]
        shift = combined[:, 0:1, :]  # (batch, 1, dim)
        scale = combined[:, 1:2, :]  # (batch, 1, dim)
        s = s * (1 + scale) + shift

        # Concatenate original hidden_states with stage 1 output for connection
        # (matches PyTorch: x = torch.cat([hidden_states, s], dim=-1))
        h = mx.concatenate([hidden_states, s], axis=-1)
        h = self.connection_proj(h)

        # Stage 2 processing
        adaln_cond_2, embedded_timestep_2 = self.adaln_single_2(t)  # (batch, 6, dim_2), (batch, dim_2)
        for block in self.transformer_blocks_2:
            h = block(h, adaln_cond_2)

        # Apply final stage 2 norm and scale/shift (matches PyTorch)
        h = self.norm_out_2(h)
        # Combine scale_shift_table_2 with embedded_timestep_2
        combined_2 = self.scale_shift_table_2[None, :, :] + embedded_timestep_2[:, None, :]
        shift2 = combined_2[:, 0:1, :]  # (batch, 1, dim_2)
        scale2 = combined_2[:, 1:2, :]  # (batch, 1, dim_2)
        h = h * (1 + scale2) + shift2

        # Output projection
        velocity = self.proj_out(h)

        return velocity


class FlowMatchingDecoder(nn.Module):
    """Flow Matching Decoder for HeartCodec.

    Combines:
    1. ResidualVQ for encoding audio codes to embeddings
    2. LlamaTransformer for velocity estimation
    3. ODE solver for generating latents from codes

    Args:
        dim: RVQ embedding dimension.
        codebook_size: Number of codes per codebook.
        codebook_dim: Dimension of code vectors.
        num_quantizers: Number of RVQ levels.
        attention_head_dim: Dimension per attention head.
        in_channels: Conditioning input channels.
        num_attention_heads: Number of attention heads.
        num_layers: Transformer layers in first stage.
        num_layers_2: Transformer layers in second stage.
        out_channels: Output latent dimension.
        use_cosine_sim: Use cosine similarity in RVQ.
    """

    def __init__(
        self,
        dim: int = 512,
        codebook_size: int = 8192,
        codebook_dim: int = 32,
        num_quantizers: int = 8,
        attention_head_dim: int = 64,
        in_channels: int = 1024,
        num_attention_heads: int = 24,
        num_layers: int = 24,
        num_layers_2: int = 6,
        out_channels: int = 256,
        use_cosine_sim: bool = False,
        decay: float = 0.9,
        commitment_weight: float = 1.0,
        threshold_ema_dead_code: int = 2,
    ):
        super().__init__()

        self.dim = dim
        self.out_channels = out_channels
        self.in_channels = in_channels

        # VQ embedding for code lookup
        self.vq_embed = ResidualVQ(
            num_quantizers=num_quantizers,
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
            dim=dim,
            use_cosine_sim=use_cosine_sim,
        )

        # Projection from VQ embeddings (used for conditioning)
        self.cond_feature_emb = nn.Linear(dim, dim, bias=True)

        # Zero embedding for classifier-free guidance
        self.zero_cond_embedding1 = mx.zeros((dim,))

        # Velocity estimator
        transformer_dim = num_attention_heads * attention_head_dim  # 24 * 64 = 1536
        transformer_dim_2 = transformer_dim * 2  # 3072

        self.estimator = LlamaTransformerForFlowMatching(
            dim=transformer_dim,
            dim_2=transformer_dim_2,
            n_heads=num_attention_heads,
            n_heads_2=num_attention_heads * 2,  # 48
            head_dim=attention_head_dim,
            num_layers=num_layers,
            num_layers_2=num_layers_2,
            in_channels=in_channels,
            out_channels=out_channels,
            mlp_hidden_dim=int(transformer_dim * 8 / 3),  # ~4096
            mlp_hidden_dim_2=int(transformer_dim_2 * 8 / 3),  # ~8192
        )

    def solve_euler(
        self,
        x: mx.array,
        incontext_x: mx.array,
        incontext_length: int,
        t_span: mx.array,
        mu: mx.array,
        guidance_scale: float,
    ) -> mx.array:
        """Euler ODE solver matching PyTorch's implementation.

        Args:
            x: Initial noise (batch, seq, latent_dim).
            incontext_x: Context latent (batch, seq, latent_dim).
            incontext_length: Number of context frames.
            t_span: Time steps array.
            mu: Conditioning from VQ embeddings (batch, seq, 512).
            guidance_scale: CFG scale.

        Returns:
            Generated latent.
        """
        t = t_span[0]
        dt = t_span[1] - t_span[0]
        noise = x

        for step in range(1, len(t_span)):
            # Interpolate noise and context for incontext frames
            if incontext_length > 0:
                interp_factor = (1 - (1 - 1e-6) * t)
                x = mx.concatenate([
                    interp_factor * noise[:, :incontext_length, :] + t * incontext_x[:, :incontext_length, :],
                    x[:, incontext_length:, :]
                ], axis=1)

            if guidance_scale > 1.0:
                # Double batch for CFG
                x_doubled = mx.concatenate([x, x], axis=0)
                incontext_doubled = mx.concatenate([incontext_x, incontext_x], axis=0)
                # Unconditional has zeros for mu
                mu_uncond = mx.zeros_like(mu)
                mu_doubled = mx.concatenate([mu_uncond, mu], axis=0)

                # Concatenate [x, incontext_x, mu] along channel dim
                hidden_states = mx.concatenate([x_doubled, incontext_doubled, mu_doubled], axis=-1)
                t_tensor = mx.broadcast_to(mx.array([t]), (2,))

                # Run estimator
                dphi_dt = self.estimator(t_tensor, hidden_states)

                # Split and apply CFG
                dphi_dt_uncond, dphi_dt_cond = mx.split(dphi_dt, 2, axis=0)
                dphi_dt = dphi_dt_uncond + guidance_scale * (dphi_dt_cond - dphi_dt_uncond)
            else:
                # Concatenate [x, incontext_x, mu] along channel dim
                hidden_states = mx.concatenate([x, incontext_x, mu], axis=-1)
                t_tensor = mx.array([t])
                dphi_dt = self.estimator(t_tensor, hidden_states)

            x = x + dt * dphi_dt
            t = t + dt

            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t_span[step]

        # Restore context frames
        if incontext_length > 0:
            x = mx.concatenate([
                incontext_x[:, :incontext_length, :],
                x[:, incontext_length:, :]
            ], axis=1)

        return x

    def inference_codes(
        self,
        codes: mx.array,
        num_steps: int = 10,
        guidance_scale: float = 1.25,
    ) -> mx.array:
        """Generate latents from codes - matches PyTorch implementation.

        Args:
            codes: Audio codes of shape (batch, seq_len, num_quantizers).
            num_steps: Number of ODE integration steps.
            guidance_scale: Classifier-free guidance scale.

        Returns:
            Generated latent of shape (batch, seq_len * 2, out_channels).
        """
        batch_size, seq_len, _ = codes.shape

        # Get VQ embeddings from codes
        embeddings = self.vq_embed.from_codes(codes)

        # Project embeddings
        mu = self.cond_feature_emb(embeddings)  # (batch, seq, 512)

        # Upsample by 2x using nearest neighbor interpolation
        # PyTorch uses F.interpolate(x.permute(0,2,1), scale_factor=2, mode='nearest').permute(0,2,1)
        mu = mx.repeat(mu, 2, axis=1)  # (batch, seq*2, 512)

        num_frames = mu.shape[1]
        latent_length = num_frames  # For simple case, all frames are generated

        # Initialize random latent (matches PyTorch's latent_dim = 256)
        latents = mx.random.normal(shape=(batch_size, num_frames, self.out_channels))

        # Create masks (for simple generation, all frames are marked for generation)
        # latent_masks = 2 means "generate this frame"
        # For now, we use incontext_length = 0 (no context)
        incontext_length = 0
        incontext_latents = mx.zeros_like(latents)

        # Apply mask to mu: use mu where mask > 0.5, else use zero_cond_embedding
        # For simple generation with full mask, mu stays as is
        # (In PyTorch: quantized_feature_emb = (latent_masks > 0.5) * quantized_feature_emb +
        #              (latent_masks < 0.5) * self.zero_cond_embedding1)

        # Time span
        t_span = mx.linspace(0, 1, num_steps + 1)

        # Solve ODE
        latents = self.solve_euler(
            x=latents,
            incontext_x=incontext_latents,
            incontext_length=incontext_length,
            t_span=t_span,
            mu=mu,
            guidance_scale=guidance_scale,
        )

        return latents

    def __call__(
        self,
        codes: mx.array,
        num_steps: int = 10,
        guidance_scale: float = 1.25,
    ) -> mx.array:
        """Forward pass for inference.

        Args:
            codes: Audio codes.
            num_steps: ODE integration steps.
            guidance_scale: CFG scale.

        Returns:
            Generated latent.
        """
        return self.inference_codes(codes, num_steps, guidance_scale)
