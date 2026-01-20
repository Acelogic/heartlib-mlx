"""Test RoPE parity between PyTorch and MLX."""

import numpy as np
import torch
import mlx.core as mx
import sys

sys.path.insert(0, '/Users/mcruz/Developer/heartlib/src')
sys.path.insert(0, '/Users/mcruz/Developer/heartlib-mlx/src')


def main():
    print("=" * 70)
    print("ROPE PARITY TEST")
    print("=" * 70)

    # Test parameters
    dim = 128  # head_dim
    seq_len = 5
    batch_size = 1
    n_heads = 24
    rope_base = 500000.0

    # Create test input (Q-like tensor)
    np.random.seed(42)
    q_np = np.random.randn(batch_size, seq_len, n_heads, dim).astype(np.float32) * 0.1

    print("\n" + "=" * 60)
    print("Step 1: Check inverse frequencies")
    print("=" * 60)

    # PyTorch inverse freq (from torchtune)
    pt_inv_freq = 1.0 / (rope_base ** (torch.arange(0, dim, 2).float() / dim))

    # MLX inverse freq
    mlx_inv_freq = 1.0 / (rope_base ** (mx.arange(0, dim, 2).astype(mx.float32) / dim))
    mx.eval(mlx_inv_freq)

    pt_np = pt_inv_freq.numpy()
    mlx_np_inv = np.array(mlx_inv_freq.astype(mx.float32))

    corr = np.corrcoef(pt_np.flatten(), mlx_np_inv.flatten())[0, 1]
    print(f"Inverse freq correlation: {corr:.6f}")
    print(f"PT inv_freq[:5]: {pt_np[:5]}")
    print(f"MLX inv_freq[:5]: {mlx_np_inv[:5]}")

    print("\n" + "=" * 60)
    print("Step 2: Check cos/sin caches")
    print("=" * 60)

    # PyTorch cos/sin
    positions = torch.arange(seq_len).float()
    freqs = torch.outer(positions, pt_inv_freq)
    print(f"PT freqs shape: {freqs.shape}")  # (seq_len, dim/2)

    # MLX cos/sin
    mlx_positions = mx.arange(seq_len).astype(mx.float32)
    mlx_freqs = mx.outer(mlx_positions, mlx_inv_freq)
    mx.eval(mlx_freqs)
    print(f"MLX freqs shape: {mlx_freqs.shape}")  # (seq_len, dim/2)

    # Check freqs match
    pt_freqs_np = freqs.numpy()
    mlx_freqs_np = np.array(mlx_freqs.astype(mx.float32))
    corr = np.corrcoef(pt_freqs_np.flatten(), mlx_freqs_np.flatten())[0, 1]
    print(f"Freqs correlation: {corr:.6f}")

    # PT uses complex numbers approach vs MLX uses concat
    # Let's see what torchtune does exactly
    print("\n" + "=" * 60)
    print("Step 3: Check rotate_half implementations")
    print("=" * 60)

    q_pt = torch.tensor(q_np)
    q_mlx = mx.array(q_np)

    # PyTorch rotate_half (from torchtune - they use interleaved)
    # torchtune's rotate_half:
    #   x1 = x[..., ::2]
    #   x2 = x[..., 1::2]
    #   return torch.stack((-x2, x1), dim=-1).flatten(-2)

    def pt_rotate_half(x):
        """torchtune style"""
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        return torch.stack((-x2, x1), dim=-1).flatten(-2)

    # MLX rotate_half (our implementation)
    def mlx_rotate_half(x):
        """Our style"""
        half_dim = x.shape[-1] // 2
        x1 = x[..., :half_dim]
        x2 = x[..., half_dim:]
        return mx.concatenate([-x2, x1], axis=-1)

    pt_rot = pt_rotate_half(q_pt)
    mlx_rot = mlx_rotate_half(q_mlx)
    mx.eval(mlx_rot)

    pt_np = pt_rot.numpy()
    mlx_np_rot = np.array(mlx_rot.astype(mx.float32))

    corr = np.corrcoef(pt_np.flatten(), mlx_np_rot.flatten())[0, 1]
    print(f"rotate_half correlation: {corr:.6f}")

    # Show first few values to see the difference
    print(f"\nPT rotate_half first 8 values: {pt_np[0, 0, 0, :8]}")
    print(f"MLX rotate_half first 8 values: {mlx_np_rot[0, 0, 0, :8]}")

    # The difference:
    # PT (interleaved): [-x2[0], x1[0], -x2[1], x1[1], ...] (interleaved pairs)
    # MLX (split): [-x2[0], -x2[1], ..., -x2[63], x1[0], x1[1], ..., x1[63]] (split halves)

    print("\n" + "=" * 60)
    print("Step 4: Check cos/sin format")
    print("=" * 60)

    # PT uses freqs directly with interleaved format
    # freqs shape is (seq_len, dim/2)
    # Then it does: torch.stack([cos, sin], dim=-1).flatten(-2)
    # Which gives shape (seq_len, dim) in interleaved format

    pt_cos = torch.cos(freqs)
    pt_sin = torch.sin(freqs)
    print(f"PT cos/sin shapes (before expand): cos={pt_cos.shape}, sin={pt_sin.shape}")

    # For interleaved, torchtune does:
    # freqs_cis = torch.stack([cos, sin], dim=-1).flatten(-2)
    # This gives (seq_len, dim) where cos/sin values are interleaved

    # For split, MLX does:
    # freqs = concat([freqs, freqs], axis=-1)  # (seq_len, dim)
    # cos = cos(freqs), sin = sin(freqs)

    mlx_freqs_full = mx.concatenate([mlx_freqs, mlx_freqs], axis=-1)
    mlx_cos = mx.cos(mlx_freqs_full)
    mlx_sin = mx.sin(mlx_freqs_full)
    mx.eval(mlx_cos)
    mx.eval(mlx_sin)
    print(f"MLX cos/sin shapes: cos={mlx_cos.shape}, sin={mlx_sin.shape}")

    # Show first few values
    print(f"\nPT cos first 8 values: {pt_cos[0, :8].numpy()}")
    print(f"MLX cos first 8 values: {np.array(mlx_cos[0, :8].astype(mx.float32))}")

    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    print("""
The issue is the RoPE implementation style:

PyTorch (torchtune) uses INTERLEAVED format:
- rotate_half: [-x2[0], x1[0], -x2[1], x1[1], ...] (interleave pairs)
- cos/sin: [cos0, cos1, cos2, ...] each applied to pairs
- Formula: x' = x * cos + rotate_half(x) * sin

MLX uses SPLIT format:
- rotate_half: [-x64, ..., -x127, x0, ..., x63] (split in half, swap and negate)
- cos/sin: [cos0, ..., cos63, cos0, ..., cos63] (repeat)
- Formula: x' = x * cos + rotate_half(x) * sin

These are mathematically equivalent when done correctly, but the
cos/sin values need to be arranged differently!
""")


if __name__ == "__main__":
    main()
