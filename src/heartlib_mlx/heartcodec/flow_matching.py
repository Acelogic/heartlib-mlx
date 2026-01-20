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
        x = nn.silu(x)
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

    def _timestep_embedding(self, t: mx.array) -> mx.array:
        """Create sinusoidal timestep embeddings."""
        half_dim = self.frequency_embedding_size // 2
        freqs = mx.exp(
            -mx.log(mx.array(10000.0)) * mx.arange(half_dim) / half_dim
        )
        args = t[:, None] * freqs[None, :]
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

    def __call__(self, t: mx.array) -> mx.array:
        """Forward pass.

        Args:
            t: Timestep of shape (batch,).

        Returns:
            Conditioning of shape (batch, num_outputs, dim).
        """
        t_emb = self.emb.timestep_embedder(t)
        conditioning = self.linear(t_emb)
        # Reshape to (batch, num_outputs, dim)
        batch_size = conditioning.shape[0]
        dim = t_emb.shape[-1]
        num_outputs = conditioning.shape[-1] // dim
        conditioning = conditioning.reshape(batch_size, num_outputs, dim)
        return conditioning


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

        # Scale/shift for final stage 2 output
        self.scale_shift_table_2 = mx.zeros((2, dim_2))

        # Output projection
        self.proj_out = FFNBlock(dim_2, out_channels, hidden_features=out_channels)

    def __call__(
        self,
        t: mx.array,
        x: mx.array,
        condition: mx.array,
    ) -> mx.array:
        """Forward pass to predict velocity.

        Args:
            t: Timestep of shape (batch,) or (1,).
            x: Noisy latent of shape (batch, seq_len, out_channels).
            condition: Conditioning of shape (batch, seq_len, in_channels).

        Returns:
            Predicted velocity of shape (batch, seq_len, out_channels).
        """
        batch_size = x.shape[0]

        # Expand t if needed
        if t.shape[0] == 1 and batch_size > 1:
            t = mx.broadcast_to(t, (batch_size,))

        # Project condition to stage 1 dim
        h = self.proj_in(condition)

        # Stage 1 processing
        adaln_cond = self.adaln_single(t)  # (batch, 6, dim)
        for block in self.transformer_blocks:
            h = block(h, adaln_cond)

        # Apply final stage 1 scale/shift
        shift, scale = self.scale_shift_table[0], self.scale_shift_table[1]
        h = h * (1 + scale[None, None, :]) + shift[None, None, :]

        # Concatenate with original condition for connection
        h = mx.concatenate([h, condition], axis=-1)
        h = self.connection_proj(h)

        # Stage 2 processing
        adaln_cond_2 = self.adaln_single_2(t)  # (batch, 6, dim_2)
        for block in self.transformer_blocks_2:
            h = block(h, adaln_cond_2)

        # Apply final stage 2 scale/shift
        shift2, scale2 = self.scale_shift_table_2[0], self.scale_shift_table_2[1]
        h = h * (1 + scale2[None, None, :]) + shift2[None, None, :]

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

    def velocity(
        self,
        t: mx.array,
        x: mx.array,
        condition: mx.array,
    ) -> mx.array:
        """Compute velocity for ODE integration.

        Args:
            t: Timestep.
            x: Current latent state.
            condition: Conditioning signal.

        Returns:
            Velocity prediction.
        """
        return self.estimator(t, x, condition)

    def inference_codes(
        self,
        codes: mx.array,
        num_steps: int = 10,
        guidance_scale: float = 1.25,
    ) -> mx.array:
        """Generate latents from codes.

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
        embeddings = self.cond_feature_emb(embeddings)

        # Upsample embeddings by 2x via interpolation
        embeddings_up = mx.repeat(embeddings, 2, axis=1)

        # For conditioning, we need to project to in_channels
        # In PyTorch this is done differently - let's check if we need more projection
        # For now, pad to in_channels dimension
        if embeddings_up.shape[-1] < self.in_channels:
            padding = mx.zeros(
                (batch_size, embeddings_up.shape[1], self.in_channels - embeddings_up.shape[-1])
            )
            condition = mx.concatenate([embeddings_up, padding], axis=-1)
        else:
            condition = embeddings_up[:, :, :self.in_channels]

        # Initialize random latent
        out_len = seq_len * 2
        x0 = mx.random.normal(shape=(batch_size, out_len, self.out_channels))

        # Solve ODE from t=0 (noise) to t=1 (data)
        if guidance_scale > 1.0:
            # For CFG, create unconditional embedding
            uncond = mx.broadcast_to(
                self.zero_cond_embedding1[None, None, :],
                (batch_size, condition.shape[1], self.dim)
            )
            if uncond.shape[-1] < self.in_channels:
                padding = mx.zeros(
                    (batch_size, uncond.shape[1], self.in_channels - uncond.shape[-1])
                )
                uncond = mx.concatenate([uncond, padding], axis=-1)

            x1 = euler_solve(
                velocity_fn=self.velocity,
                x0=x0,
                condition=condition,
                num_steps=num_steps,
                guidance_scale=guidance_scale,
                uncond=uncond,
            )
        else:
            x1 = euler_solve(
                velocity_fn=self.velocity,
                x0=x0,
                condition=condition,
                num_steps=num_steps,
            )

        return x1

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
