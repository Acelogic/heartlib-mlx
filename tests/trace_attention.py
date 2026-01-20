"""Trace attention computation step by step."""

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
    print("ATTENTION TRACE")
    print("=" * 70)

    from heartlib.heartmula.modeling_heartmula import HeartMuLa as PTHeartMuLa
    from heartlib_mlx.heartmula import HeartMuLa as MLXHeartMuLa

    print("Loading models...")
    pt_model = PTHeartMuLa.from_pretrained('/Users/mcruz/Developer/heartlib/ckpt/HeartMuLa-oss-3B', dtype=torch.float32)
    pt_model.eval()
    mlx_model = MLXHeartMuLa.from_pretrained('/Users/mcruz/Developer/heartlib-mlx/ckpt-mlx/heartmula')

    # Input
    np.random.seed(42)
    batch_size = 1
    seq_len = 5
    dim = 3072

    # Use random input for cleaner analysis
    hidden_np = np.random.randn(batch_size, seq_len, dim).astype(np.float32) * 0.1
    pt_h = torch.tensor(hidden_np)
    mlx_h = mx.array(hidden_np)

    # Get attention modules
    pt_attn = pt_model.backbone.layers[0].attn
    mlx_attn = mlx_model.backbone.layers[0].attention

    # Check scale factors
    print("\n" + "=" * 60)
    print("Scale Factors")
    print("=" * 60)
    pt_scale = 1.0 / (pt_attn.head_dim ** 0.5)
    mlx_scale = mlx_attn.scale
    print(f"PT scale: {pt_scale:.6f}")
    print(f"MLX scale: {mlx_scale:.6f}")
    print(f"PT head_dim: {pt_attn.head_dim}")
    print(f"MLX head_dim: {mlx_attn.head_dim}")

    # Get Q, K, V
    print("\n" + "=" * 60)
    print("Q, K, V Projections")
    print("=" * 60)

    with torch.no_grad():
        pt_q = pt_attn.q_proj(pt_h)
        pt_k = pt_attn.k_proj(pt_h)
        pt_v = pt_attn.v_proj(pt_h)

    mlx_q = mlx_attn.q_proj(mlx_h)
    mlx_k = mlx_attn.k_proj(mlx_h)
    mlx_v = mlx_attn.v_proj(mlx_h)
    mx.eval(mlx_q)
    mx.eval(mlx_k)
    mx.eval(mlx_v)

    print_stats("PT Q", pt_q)
    print_stats("MLX Q", mlx_q)

    # Reshape
    n_heads = 24
    n_kv_heads = 8
    head_dim = 128

    pt_q_r = pt_q.view(batch_size, seq_len, n_heads, head_dim)
    pt_k_r = pt_k.view(batch_size, seq_len, n_kv_heads, head_dim)

    mlx_q_r = mlx_q.reshape(batch_size, seq_len, n_heads, head_dim)
    mlx_k_r = mlx_k.reshape(batch_size, seq_len, n_kv_heads, head_dim)
    mx.eval(mlx_q_r)
    mx.eval(mlx_k_r)

    # Apply RoPE
    print("\n" + "=" * 60)
    print("After RoPE")
    print("=" * 60)

    # PyTorch uses torchtune's apply_rotary_emb
    from torchtune.modules import RotaryPositionalEmbeddings as PTRope

    # Create position indices
    pt_pos = torch.arange(seq_len)

    # Check if PT model has rope_k enabled
    print(f"PT RoPE applied to K: {getattr(pt_attn, 'rope_k', True)}")

    # MLX RoPE
    mlx_q_rot, mlx_k_rot = mlx_attn.rope(mlx_q_r, mlx_k_r, offset=0)
    mx.eval(mlx_q_rot)
    mx.eval(mlx_k_rot)

    # For PyTorch, we need to look at what rope does
    # Let's look at the internal rope implementation
    with torch.no_grad():
        # torchtune uses rope(q, input_pos) internally
        pt_q_rot = pt_attn.pos_embeddings(pt_q_r, input_pos=pt_pos)
        # Check if K also gets rope
        if hasattr(pt_attn, 'rope_k') and not pt_attn.rope_k:
            pt_k_rot = pt_k_r
        else:
            pt_k_rot = pt_attn.pos_embeddings(pt_k_r, input_pos=pt_pos)

    pt_np = pt_q_rot.detach().numpy()
    mlx_np = np.array(mlx_q_rot.astype(mx.float32))
    corr = np.corrcoef(pt_np.flatten(), mlx_np.flatten())[0, 1]
    print(f"Q after RoPE correlation: {corr:.6f}")
    print_stats("  PT Q rot", pt_np)
    print_stats("  MLX Q rot", mlx_np)

    pt_np = pt_k_rot.detach().numpy()
    mlx_np = np.array(mlx_k_rot.astype(mx.float32))
    corr = np.corrcoef(pt_np.flatten(), mlx_np.flatten())[0, 1]
    print(f"K after RoPE correlation: {corr:.6f}")
    print_stats("  PT K rot", pt_np)
    print_stats("  MLX K rot", mlx_np)

    # Attention scores
    print("\n" + "=" * 60)
    print("Attention Scores")
    print("=" * 60)

    # Transpose for attention: (batch, n_heads, seq_len, head_dim)
    pt_q_t = pt_q_rot.transpose(1, 2)
    pt_k_t = pt_k_rot.transpose(1, 2)

    mlx_q_t = mlx_q_rot.transpose(0, 2, 1, 3)
    mlx_k_t = mlx_k_rot.transpose(0, 2, 1, 3)
    mx.eval(mlx_q_t)
    mx.eval(mlx_k_t)

    # Repeat K for GQA
    n_rep = n_heads // n_kv_heads  # 3
    pt_k_rep = pt_k_t.repeat_interleave(n_rep, dim=1)
    mlx_k_rep = mx.repeat(mlx_k_t, n_rep, axis=1)
    mx.eval(mlx_k_rep)

    print(f"Q transposed shape: PT {pt_q_t.shape}, MLX {mlx_q_t.shape}")
    print(f"K repeated shape: PT {pt_k_rep.shape}, MLX {mlx_k_rep.shape}")

    # Compute attention scores
    with torch.no_grad():
        pt_scores = torch.matmul(pt_q_t, pt_k_rep.transpose(-2, -1)) * pt_scale

    mlx_scores = (mlx_q_t @ mlx_k_rep.transpose(0, 1, 3, 2)) * mlx_scale
    mx.eval(mlx_scores)

    pt_np = pt_scores.detach().numpy()
    mlx_np = np.array(mlx_scores.astype(mx.float32))
    corr = np.corrcoef(pt_np.flatten(), mlx_np.flatten())[0, 1]
    print(f"Attention scores correlation: {corr:.6f}")
    print_stats("  PT scores", pt_np)
    print_stats("  MLX scores", mlx_np)

    # Softmax
    print("\n" + "=" * 60)
    print("Attention Weights (after softmax)")
    print("=" * 60)

    # Apply causal mask to PT
    causal = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
    pt_scores_masked = pt_scores.masked_fill(~causal, float('-inf'))

    # Apply causal mask to MLX
    causal_mlx = mx.tril(mx.ones((seq_len, seq_len)))
    mask_mlx = mx.where(causal_mlx, mx.zeros_like(mlx_scores), mx.full(mlx_scores.shape, float('-inf')))
    mlx_scores_masked = mlx_scores + mask_mlx
    mx.eval(mlx_scores_masked)

    with torch.no_grad():
        pt_attn_weights = torch.softmax(pt_scores_masked, dim=-1)

    mlx_attn_weights = mx.softmax(mlx_scores_masked, axis=-1)
    mx.eval(mlx_attn_weights)

    pt_np = pt_attn_weights.detach().numpy()
    mlx_np = np.array(mlx_attn_weights.astype(mx.float32))
    corr = np.corrcoef(pt_np.flatten(), mlx_np.flatten())[0, 1]
    print(f"Attention weights correlation: {corr:.6f}")
    print_stats("  PT weights", pt_np)
    print_stats("  MLX weights", mlx_np)

    # Output
    print("\n" + "=" * 60)
    print("Attention Output")
    print("=" * 60)

    pt_v_r = pt_v.view(batch_size, seq_len, n_kv_heads, head_dim)
    pt_v_t = pt_v_r.transpose(1, 2)
    pt_v_rep = pt_v_t.repeat_interleave(n_rep, dim=1)

    mlx_v_r = mlx_v.reshape(batch_size, seq_len, n_kv_heads, head_dim)
    mlx_v_t = mlx_v_r.transpose(0, 2, 1, 3)
    mlx_v_rep = mx.repeat(mlx_v_t, n_rep, axis=1)
    mx.eval(mlx_v_rep)

    with torch.no_grad():
        pt_attn_out = torch.matmul(pt_attn_weights, pt_v_rep)

    mlx_attn_out = mlx_attn_weights @ mlx_v_rep
    mx.eval(mlx_attn_out)

    pt_np = pt_attn_out.detach().numpy()
    mlx_np = np.array(mlx_attn_out.astype(mx.float32))
    corr = np.corrcoef(pt_np.flatten(), mlx_np.flatten())[0, 1]
    print(f"Attention output correlation: {corr:.6f}")
    print_stats("  PT attn_out", pt_np)
    print_stats("  MLX attn_out", mlx_np)

    # Reshape and output projection
    print("\n" + "=" * 60)
    print("Final Output (after o_proj)")
    print("=" * 60)

    pt_attn_final = pt_attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
    mlx_attn_final = mlx_attn_out.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
    mx.eval(mlx_attn_final)

    with torch.no_grad():
        pt_output = pt_attn.o_proj(pt_attn_final)

    mlx_output = mlx_attn.o_proj(mlx_attn_final)
    mx.eval(mlx_output)

    pt_np = pt_output.detach().numpy()
    mlx_np = np.array(mlx_output.astype(mx.float32))
    corr = np.corrcoef(pt_np.flatten(), mlx_np.flatten())[0, 1]
    print(f"Final output correlation: {corr:.6f}")
    print_stats("  PT final", pt_np)
    print_stats("  MLX final", mlx_np)


if __name__ == "__main__":
    main()
