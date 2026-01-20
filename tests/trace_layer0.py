"""Trace layer 0 step by step to find divergence."""

import numpy as np
import torch
import mlx.core as mx
import sys

sys.path.insert(0, '/Users/mcruz/Developer/heartlib/src')
sys.path.insert(0, '/Users/mcruz/Developer/heartlib-mlx/src')


def print_stats(name, arr):
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().float().numpy()
    elif isinstance(arr, mx.array):
        arr = np.array(arr.astype(mx.float32))
    print(f"{name}: shape={arr.shape}, range=[{arr.min():.4f}, {arr.max():.4f}], std={arr.std():.4f}")


def main():
    print("=" * 70)
    print("LAYER 0 TRACE")
    print("=" * 70)

    from heartlib.heartmula.modeling_heartmula import HeartMuLa as PTHeartMuLa
    from heartlib_mlx.heartmula import HeartMuLa as MLXHeartMuLa

    print("Loading models...")
    pt_model = PTHeartMuLa.from_pretrained('/Users/mcruz/Developer/heartlib/ckpt/HeartMuLa-oss-3B', dtype=torch.float32)
    pt_model.eval()
    mlx_model = MLXHeartMuLa.from_pretrained('/Users/mcruz/Developer/heartlib-mlx/ckpt-mlx/heartmula')

    # Create input hidden states (same as before)
    seq_len = 5
    num_codebooks = 8
    parallel = num_codebooks + 1

    tokens = np.zeros((1, seq_len, parallel), dtype=np.int64)
    tokens[0, :, -1] = [128000, 100, 200, 300, 128001]
    mask = np.zeros((1, seq_len, parallel), dtype=bool)
    mask[:, :, -1] = True

    pt_tokens = torch.tensor(tokens)
    pt_mask = torch.tensor(mask)

    mlx_tokens = mx.array(tokens)
    mlx_mask = mx.array(mask.astype(np.float32))

    # Get input hidden states (after embedding)
    with torch.no_grad():
        pt_embeds = pt_model._embed_tokens(pt_tokens, uncond_mask=None)
        pt_masked = pt_embeds * pt_mask.unsqueeze(-1)
        pt_h = pt_masked.sum(dim=2, dtype=pt_embeds.dtype)

    mlx_embeds = mlx_model._embed_tokens(mlx_tokens, uncond_mask=None)
    mlx_masked = mlx_embeds * mlx_mask[:, :, :, None]
    mlx_h = mx.sum(mlx_masked, axis=2)
    mx.eval(mlx_h)

    # Get layer 0
    pt_layer0 = pt_model.backbone.layers[0]
    mlx_layer0 = mlx_model.backbone.layers[0]

    print("\n" + "=" * 60)
    print("Input to Layer 0")
    print("=" * 60)
    pt_np = pt_h.detach().numpy()
    mlx_np = np.array(mlx_h.astype(mx.float32))
    corr = np.corrcoef(pt_np.flatten(), mlx_np.flatten())[0, 1]
    print(f"Correlation: {corr:.6f}")
    print_stats("  PT", pt_np)
    print_stats("  MLX", mlx_np)

    # Step 1: RMSNorm (attention)
    print("\n" + "=" * 60)
    print("Step 1: Attention RMSNorm")
    print("=" * 60)

    # Check norm weights
    pt_norm_w = pt_layer0.sa_norm.scale.detach().numpy()
    mlx_norm_w = np.array(mlx_layer0.attention_norm.weight.astype(mx.float32))
    print(f"PT norm weights: shape={pt_norm_w.shape}, all same? {np.allclose(pt_norm_w, pt_norm_w[0])}")
    print(f"MLX norm weights: shape={mlx_norm_w.shape}, all same? {np.allclose(mlx_norm_w, mlx_norm_w[0])}")

    # Run norm
    with torch.no_grad():
        pt_normed = pt_layer0.sa_norm(pt_h)

    mlx_normed = mlx_layer0.attention_norm(mlx_h)
    mx.eval(mlx_normed)

    pt_np = pt_normed.detach().numpy()
    mlx_np = np.array(mlx_normed.astype(mx.float32))
    corr = np.corrcoef(pt_np.flatten(), mlx_np.flatten())[0, 1]
    print(f"After attention norm correlation: {corr:.6f}")
    print_stats("  PT", pt_np)
    print_stats("  MLX", mlx_np)

    # Step 2: Q, K, V projections
    print("\n" + "=" * 60)
    print("Step 2: Q, K, V Projections")
    print("=" * 60)

    with torch.no_grad():
        pt_q = pt_layer0.attn.q_proj(pt_normed)
        pt_k = pt_layer0.attn.k_proj(pt_normed)
        pt_v = pt_layer0.attn.v_proj(pt_normed)

    mlx_q = mlx_layer0.attention.q_proj(mlx_normed)
    mlx_k = mlx_layer0.attention.k_proj(mlx_normed)
    mlx_v = mlx_layer0.attention.v_proj(mlx_normed)
    mx.eval(mlx_q)
    mx.eval(mlx_k)
    mx.eval(mlx_v)

    pt_np = pt_q.detach().numpy()
    mlx_np = np.array(mlx_q.astype(mx.float32))
    corr = np.corrcoef(pt_np.flatten(), mlx_np.flatten())[0, 1]
    print(f"Q projection correlation: {corr:.6f}")

    pt_np = pt_k.detach().numpy()
    mlx_np = np.array(mlx_k.astype(mx.float32))
    corr = np.corrcoef(pt_np.flatten(), mlx_np.flatten())[0, 1]
    print(f"K projection correlation: {corr:.6f}")

    pt_np = pt_v.detach().numpy()
    mlx_np = np.array(mlx_v.astype(mx.float32))
    corr = np.corrcoef(pt_np.flatten(), mlx_np.flatten())[0, 1]
    print(f"V projection correlation: {corr:.6f}")

    # Step 3: Reshape and RoPE
    print("\n" + "=" * 60)
    print("Step 3: Reshape and RoPE")
    print("=" * 60)

    # Get dimensions
    pt_n_heads = pt_layer0.attn.num_heads
    pt_n_kv_heads = pt_layer0.attn.num_kv_heads
    pt_head_dim = pt_layer0.attn.head_dim
    print(f"PT: n_heads={pt_n_heads}, n_kv_heads={pt_n_kv_heads}, head_dim={pt_head_dim}")

    mlx_n_heads = mlx_layer0.attention.n_heads
    mlx_n_kv_heads = mlx_layer0.attention.n_kv_heads
    mlx_head_dim = mlx_layer0.attention.head_dim
    print(f"MLX: n_heads={mlx_n_heads}, n_kv_heads={mlx_n_kv_heads}, head_dim={mlx_head_dim}")

    # Reshape Q
    batch = 1
    with torch.no_grad():
        pt_q_reshaped = pt_q.view(batch, seq_len, pt_n_heads, pt_head_dim)
        pt_k_reshaped = pt_k.view(batch, seq_len, pt_n_kv_heads, pt_head_dim)

    mlx_q_reshaped = mlx_q.reshape(batch, seq_len, mlx_n_heads, mlx_head_dim)
    mlx_k_reshaped = mlx_k.reshape(batch, seq_len, mlx_n_kv_heads, mlx_head_dim)
    mx.eval(mlx_q_reshaped)
    mx.eval(mlx_k_reshaped)

    print(f"PT Q reshaped: {pt_q_reshaped.shape}")
    print(f"MLX Q reshaped: {mlx_q_reshaped.shape}")

    # Check RoPE frequencies
    print("\nChecking RoPE...")

    # PyTorch uses apply_rotary_emb from torchtune
    # MLX uses our own RoPE implementation
    # Let's check if the cos/sin values are the same

    # Get position indices
    pt_pos = torch.arange(seq_len)

    # MLX RoPE precomputes freqs in attention
    mlx_rope = mlx_layer0.attention.rope
    print(f"MLX rope theta (base): {mlx_rope.base if hasattr(mlx_rope, 'base') else 'N/A'}")

    # Check if RoPE is applied correctly by comparing Q after RoPE
    # This requires running the full attention forward

    # Let's just run the full attention and compare
    print("\n" + "=" * 60)
    print("Step 4: Full Attention Output")
    print("=" * 60)

    # Need to set up caches first
    pt_model.setup_caches(1)
    mlx_model.setup_caches(1)

    # Run through full layer 0
    with torch.no_grad():
        # PyTorch uses causal mask
        pt_causal_mask = torch.tril(torch.ones(8192, 8192, dtype=torch.bool))
        pt_pos_tensor = torch.arange(seq_len).unsqueeze(0)
        pt_mask_indexed = pt_causal_mask[pt_pos_tensor, :]

        # Run layer
        pt_layer_out = pt_layer0(pt_h, input_pos=pt_pos_tensor, mask=pt_mask_indexed)

    # MLX layer
    mlx_cache = [None] * len(mlx_model.backbone.layers)
    mlx_layer_out, _ = mlx_layer0(mlx_h, mask=None, cache=None, offset=0)
    mx.eval(mlx_layer_out)

    pt_np = pt_layer_out.detach().numpy()
    mlx_np = np.array(mlx_layer_out.astype(mx.float32))
    corr = np.corrcoef(pt_np.flatten(), mlx_np.flatten())[0, 1]
    print(f"Layer 0 output correlation: {corr:.6f}")
    print_stats("  PT", pt_np)
    print_stats("  MLX", mlx_np)

    # Compare element by element for first few
    print("\nFirst 10 values comparison:")
    pt_flat = pt_np.flatten()[:10]
    mlx_flat = mlx_np.flatten()[:10]
    for i, (p, m) in enumerate(zip(pt_flat, mlx_flat)):
        diff = abs(p - m)
        print(f"  [{i}] PT: {p:.6f}, MLX: {m:.6f}, diff: {diff:.6f}")


if __name__ == "__main__":
    main()
