"""Trace backbone transformer to find divergence."""

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
    print(f"{name}: shape={arr.shape}, range=[{arr.min():.4f}, {arr.max():.4f}], std={arr.std():.4f}, mean={arr.mean():.4f}")


def main():
    print("=" * 70)
    print("BACKBONE TRACE TEST")
    print("=" * 70)

    from heartlib.heartmula.modeling_heartmula import HeartMuLa as PTHeartMuLa
    from heartlib_mlx.heartmula import HeartMuLa as MLXHeartMuLa

    print("Loading models...")
    pt_model = PTHeartMuLa.from_pretrained('/Users/mcruz/Developer/heartlib/ckpt/HeartMuLa-oss-3B', dtype=torch.float32)
    pt_model.eval()
    mlx_model = MLXHeartMuLa.from_pretrained('/Users/mcruz/Developer/heartlib-mlx/ckpt-mlx/heartmula')

    # Create simple input - same as generate_frame test
    seq_len = 5
    num_codebooks = 8
    parallel = num_codebooks + 1

    tokens = np.zeros((1, seq_len, parallel), dtype=np.int64)
    tokens[0, :, -1] = [128000, 100, 200, 300, 128001]

    mask = np.zeros((1, seq_len, parallel), dtype=bool)
    mask[:, :, -1] = True

    pt_tokens = torch.tensor(tokens)
    pt_mask = torch.tensor(mask)
    pt_pos = torch.arange(seq_len).unsqueeze(0)

    mlx_tokens = mx.array(tokens)
    mlx_mask = mx.array(mask.astype(np.float32))

    # Step 1: Embed tokens
    print("\n" + "=" * 60)
    print("Step 1: Embed Tokens")
    print("=" * 60)

    with torch.no_grad():
        pt_embeds = pt_model._embed_tokens(pt_tokens, uncond_mask=None)
    mlx_embeds = mlx_model._embed_tokens(mlx_tokens, uncond_mask=None)
    mx.eval(mlx_embeds)

    pt_np = pt_embeds.detach().numpy()
    mlx_np = np.array(mlx_embeds.astype(mx.float32))
    corr = np.corrcoef(pt_np.flatten(), mlx_np.flatten())[0, 1]
    print(f"Embeddings correlation: {corr:.6f}")
    print_stats("  PT", pt_np)
    print_stats("  MLX", mlx_np)

    # Step 2: Apply mask and sum
    print("\n" + "=" * 60)
    print("Step 2: Masked Sum")
    print("=" * 60)

    with torch.no_grad():
        pt_masked = pt_embeds * pt_mask.unsqueeze(-1)
        pt_h = pt_masked.sum(dim=2, dtype=pt_embeds.dtype)

    mlx_masked = mlx_embeds * mlx_mask[:, :, :, None]
    mlx_h = mx.sum(mlx_masked, axis=2)
    mx.eval(mlx_h)

    pt_np = pt_h.detach().numpy()
    mlx_np = np.array(mlx_h.astype(mx.float32))
    corr = np.corrcoef(pt_np.flatten(), mlx_np.flatten())[0, 1]
    print(f"After masked sum correlation: {corr:.6f}")
    print_stats("  PT", pt_np)
    print_stats("  MLX", mlx_np)

    # Step 3: Setup caches and run backbone
    print("\n" + "=" * 60)
    print("Step 3: Backbone (one layer)")
    print("=" * 60)

    # For fair comparison, let's check the first layer only
    # Get first layer weights
    pt_layer0 = pt_model.backbone.layers[0]
    mlx_layer0 = mlx_model.backbone.layers[0]

    # Check RMSNorm weights
    pt_norm_w = pt_layer0.sa_norm.scale.detach().numpy()
    mlx_norm_w = np.array(mlx_layer0.attention_norm.weight.astype(mx.float32))
    corr = np.corrcoef(pt_norm_w.flatten(), mlx_norm_w.flatten())[0, 1]
    print(f"Layer 0 input norm weights correlation: {corr:.6f}")

    # Check Q/K/V/O weights
    pt_q_w = pt_layer0.attn.q_proj.weight.detach().numpy()
    mlx_q_w = np.array(mlx_layer0.attention.q_proj.weight.astype(mx.float32))
    corr = np.corrcoef(pt_q_w.flatten(), mlx_q_w.flatten())[0, 1]
    print(f"Layer 0 Q projection weights correlation: {corr:.6f}")

    pt_k_w = pt_layer0.attn.k_proj.weight.detach().numpy()
    mlx_k_w = np.array(mlx_layer0.attention.k_proj.weight.astype(mx.float32))
    corr = np.corrcoef(pt_k_w.flatten(), mlx_k_w.flatten())[0, 1]
    print(f"Layer 0 K projection weights correlation: {corr:.6f}")

    # Step 4: Full backbone comparison
    print("\n" + "=" * 60)
    print("Step 4: Full Backbone Forward")
    print("=" * 60)

    pt_model.setup_caches(1)
    mlx_model.setup_caches(1)

    # Run PyTorch backbone
    with torch.no_grad():
        pt_causal_mask = torch.tril(torch.ones(8192, 8192, dtype=torch.bool))
        pt_mask_indexed = pt_causal_mask[pt_pos, :]
        pt_backbone_out = pt_model.backbone(pt_h, input_pos=pt_pos, mask=pt_mask_indexed)

    # Run MLX backbone
    mlx_backbone_out, _ = mlx_model.backbone(mlx_h, cache=mlx_model._backbone_cache)
    mx.eval(mlx_backbone_out)

    pt_np = pt_backbone_out.detach().numpy()
    mlx_np = np.array(mlx_backbone_out.astype(mx.float32))
    corr = np.corrcoef(pt_np.flatten(), mlx_np.flatten())[0, 1]
    print(f"Backbone output correlation: {corr:.6f}")
    print_stats("  PT", pt_np)
    print_stats("  MLX", mlx_np)

    # Check last position specifically (used for generation)
    pt_last = pt_backbone_out[:, -1, :]
    mlx_last = mlx_backbone_out[:, -1, :]
    pt_np = pt_last.detach().numpy()
    mlx_np = np.array(mlx_last.astype(mx.float32))
    corr = np.corrcoef(pt_np.flatten(), mlx_np.flatten())[0, 1]
    print(f"Last position correlation: {corr:.6f}")

    # Step 5: Codebook0 logits
    print("\n" + "=" * 60)
    print("Step 5: Codebook0 Logits")
    print("=" * 60)

    with torch.no_grad():
        pt_logits = pt_model.codebook0_head(pt_last)
    mlx_logits = mlx_model.codebook0_head(mlx_last)
    mx.eval(mlx_logits)

    pt_np = pt_logits.detach().numpy()
    mlx_np = np.array(mlx_logits.astype(mx.float32))
    corr = np.corrcoef(pt_np.flatten(), mlx_np.flatten())[0, 1]
    print(f"Logits correlation: {corr:.6f}")
    print_stats("  PT", pt_np)
    print_stats("  MLX", mlx_np)

    # Top-5 tokens
    pt_top5 = torch.topk(pt_logits[0], 5)
    mlx_top5_idx = mx.argsort(-mlx_logits[0])[:5]
    mlx_top5_val = mlx_logits[0, mlx_top5_idx]
    mx.eval(mlx_top5_idx)
    mx.eval(mlx_top5_val)

    print(f"PT top-5 tokens: {pt_top5.indices.numpy()}, values: {pt_top5.values.numpy()}")
    print(f"MLX top-5 tokens: {np.array(mlx_top5_idx)}, values: {np.array(mlx_top5_val.astype(mx.float32))}")


if __name__ == "__main__":
    main()
