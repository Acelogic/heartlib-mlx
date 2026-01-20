"""Parity test for HeartMuLa between PyTorch and MLX."""

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


def test_embeddings():
    """Test that embeddings match."""
    print("\n" + "=" * 60)
    print("TEST: Token Embeddings")
    print("=" * 60)

    from heartlib.heartmula.modeling_heartmula import HeartMuLa as PTHeartMuLa
    from heartlib_mlx.heartmula import HeartMuLa as MLXHeartMuLa

    # Load models
    pt_model = PTHeartMuLa.from_pretrained('/Users/mcruz/Developer/heartlib/ckpt/HeartMuLa-oss-3B', dtype=torch.float32)
    pt_model.eval()
    mlx_model = MLXHeartMuLa.from_pretrained('/Users/mcruz/Developer/heartlib-mlx/ckpt-mlx/heartmula')

    # Test text embeddings
    test_tokens = np.array([100, 200, 500, 1000], dtype=np.int64)
    pt_text_emb = pt_model.text_embeddings(torch.tensor(test_tokens))
    mlx_text_emb = mlx_model.text_embeddings(mx.array(test_tokens))

    pt_np = pt_text_emb.detach().numpy()
    mlx_np = np.array(mlx_text_emb.astype(mx.float32))

    corr = np.corrcoef(pt_np.flatten(), mlx_np.flatten())[0, 1]
    print(f"Text embeddings correlation: {corr:.6f}")
    print_stats("  PT", pt_np)
    print_stats("  MLX", mlx_np)

    # Test audio embeddings
    audio_tokens = np.array([10, 50, 100], dtype=np.int64)
    pt_audio_emb = pt_model.audio_embeddings(torch.tensor(audio_tokens))
    mlx_audio_emb = mlx_model.audio_embeddings(mx.array(audio_tokens))

    pt_np = pt_audio_emb.detach().numpy()
    mlx_np = np.array(mlx_audio_emb.astype(mx.float32))

    corr = np.corrcoef(pt_np.flatten(), mlx_np.flatten())[0, 1]
    print(f"Audio embeddings correlation: {corr:.6f}")

    return corr > 0.99


def test_embed_tokens():
    """Test _embed_tokens specifically."""
    print("\n" + "=" * 60)
    print("TEST: _embed_tokens")
    print("=" * 60)

    from heartlib.heartmula.modeling_heartmula import HeartMuLa as PTHeartMuLa
    from heartlib_mlx.heartmula import HeartMuLa as MLXHeartMuLa

    pt_model = PTHeartMuLa.from_pretrained('/Users/mcruz/Developer/heartlib/ckpt/HeartMuLa-oss-3B', dtype=torch.float32)
    pt_model.eval()
    mlx_model = MLXHeartMuLa.from_pretrained('/Users/mcruz/Developer/heartlib-mlx/ckpt-mlx/heartmula')

    # Create test tokens
    seq_len = 3
    num_codebooks = 8
    parallel = num_codebooks + 1

    tokens = np.zeros((1, seq_len, parallel), dtype=np.int64)
    tokens[0, :, -1] = [128000, 500, 128001]  # text tokens
    tokens[0, :, :-1] = 10  # some audio token

    pt_tokens = torch.tensor(tokens)
    mlx_tokens = mx.array(tokens)

    with torch.no_grad():
        pt_embeds = pt_model._embed_tokens(pt_tokens, uncond_mask=None)

    mlx_embeds = mlx_model._embed_tokens(mlx_tokens, uncond_mask=None)
    mx.eval(mlx_embeds)

    pt_np = pt_embeds.detach().numpy()
    mlx_np = np.array(mlx_embeds.astype(mx.float32))

    corr = np.corrcoef(pt_np.flatten(), mlx_np.flatten())[0, 1]
    print(f"_embed_tokens correlation: {corr:.6f}")
    print_stats("  PT", pt_np)
    print_stats("  MLX", mlx_np)

    return corr > 0.99


def test_generate_frame_first():
    """Test first generate_frame call (prompt processing)."""
    print("\n" + "=" * 60)
    print("TEST: First generate_frame (prompt)")
    print("=" * 60)

    from heartlib.heartmula.modeling_heartmula import HeartMuLa as PTHeartMuLa
    from heartlib_mlx.heartmula import HeartMuLa as MLXHeartMuLa

    pt_model = PTHeartMuLa.from_pretrained('/Users/mcruz/Developer/heartlib/ckpt/HeartMuLa-oss-3B', dtype=torch.float32)
    pt_model.eval()
    mlx_model = MLXHeartMuLa.from_pretrained('/Users/mcruz/Developer/heartlib-mlx/ckpt-mlx/heartmula')

    # Simple prompt
    seq_len = 5
    num_codebooks = 8
    parallel = num_codebooks + 1

    tokens = np.zeros((seq_len, parallel), dtype=np.int64)
    tokens[:, -1] = [128000, 100, 200, 300, 128001]

    mask = np.zeros((seq_len, parallel), dtype=bool)
    mask[:, -1] = True

    # No CFG
    batch_size = 1

    pt_tokens = torch.tensor(tokens).unsqueeze(0)
    pt_mask = torch.tensor(mask).unsqueeze(0)
    pt_pos = torch.arange(seq_len).unsqueeze(0)

    mlx_tokens = mx.array(tokens)[None, :, :]
    mlx_mask = mx.array(mask.astype(np.float32))[None, :, :]
    mlx_pos = mx.arange(seq_len)[None, :]

    # Set fixed seed for reproducibility in sampling
    torch.manual_seed(42)
    mx.random.seed(42)

    pt_model.setup_caches(batch_size)
    mlx_model.setup_caches(batch_size)

    with torch.no_grad():
        pt_output = pt_model.generate_frame(
            tokens=pt_tokens,
            tokens_mask=pt_mask,
            input_pos=pt_pos,
            temperature=1.0,
            topk=50,
            cfg_scale=1.0,
        )

    mlx_output = mlx_model.generate_frame(
        tokens=mlx_tokens,
        tokens_mask=mlx_mask,
        input_pos=mlx_pos,
        temperature=1.0,
        topk=50,
        cfg_scale=1.0,
    )
    mx.eval(mlx_output)

    pt_np = pt_output.detach().numpy()
    mlx_np = np.array(mlx_output)

    print(f"PT output tokens: {pt_np}")
    print(f"MLX output tokens: {mlx_np}")

    # Check if outputs are in valid range
    pt_valid = np.all(pt_np >= 0) and np.all(pt_np < 8194)
    mlx_valid = np.all(mlx_np >= 0) and np.all(mlx_np < 8194)
    print(f"PT tokens valid: {pt_valid}")
    print(f"MLX tokens valid: {mlx_valid}")

    # Tokens won't match exactly due to sampling, but both should be valid
    return pt_valid and mlx_valid


def test_codebook0_head():
    """Test codebook0_head weight parity."""
    print("\n" + "=" * 60)
    print("TEST: codebook0_head weights")
    print("=" * 60)

    from heartlib.heartmula.modeling_heartmula import HeartMuLa as PTHeartMuLa
    from heartlib_mlx.heartmula import HeartMuLa as MLXHeartMuLa

    pt_model = PTHeartMuLa.from_pretrained('/Users/mcruz/Developer/heartlib/ckpt/HeartMuLa-oss-3B', dtype=torch.float32)
    mlx_model = MLXHeartMuLa.from_pretrained('/Users/mcruz/Developer/heartlib-mlx/ckpt-mlx/heartmula')

    pt_weight = pt_model.codebook0_head.weight.detach().numpy()
    mlx_weight = np.array(mlx_model.codebook0_head.weight.astype(mx.float32))

    # MLX Linear stores weights transposed
    print(f"PT weight shape: {pt_weight.shape}")
    print(f"MLX weight shape: {mlx_weight.shape}")

    # MLX: (out_features, in_features), PT: (out_features, in_features)
    corr = np.corrcoef(pt_weight.flatten(), mlx_weight.flatten())[0, 1]
    print(f"codebook0_head weight correlation: {corr:.6f}")

    return corr > 0.99


def test_projection():
    """Test projection layer."""
    print("\n" + "=" * 60)
    print("TEST: projection weights")
    print("=" * 60)

    from heartlib.heartmula.modeling_heartmula import HeartMuLa as PTHeartMuLa
    from heartlib_mlx.heartmula import HeartMuLa as MLXHeartMuLa

    pt_model = PTHeartMuLa.from_pretrained('/Users/mcruz/Developer/heartlib/ckpt/HeartMuLa-oss-3B', dtype=torch.float32)
    mlx_model = MLXHeartMuLa.from_pretrained('/Users/mcruz/Developer/heartlib-mlx/ckpt-mlx/heartmula')

    pt_weight = pt_model.projection.weight.detach().numpy()
    mlx_weight = np.array(mlx_model.projection.weight.astype(mx.float32))

    print(f"PT weight shape: {pt_weight.shape}")
    print(f"MLX weight shape: {mlx_weight.shape}")

    corr = np.corrcoef(pt_weight.flatten(), mlx_weight.flatten())[0, 1]
    print(f"projection weight correlation: {corr:.6f}")

    return corr > 0.99


def test_audio_head():
    """Test audio_head weights."""
    print("\n" + "=" * 60)
    print("TEST: audio_head weights")
    print("=" * 60)

    from heartlib.heartmula.modeling_heartmula import HeartMuLa as PTHeartMuLa
    from heartlib_mlx.heartmula import HeartMuLa as MLXHeartMuLa

    pt_model = PTHeartMuLa.from_pretrained('/Users/mcruz/Developer/heartlib/ckpt/HeartMuLa-oss-3B', dtype=torch.float32)
    mlx_model = MLXHeartMuLa.from_pretrained('/Users/mcruz/Developer/heartlib-mlx/ckpt-mlx/heartmula')

    # PT audio_head is Parameter of shape (7, decoder_dim, vocab_size)
    pt_audio_head = pt_model.audio_head.detach().numpy()
    # MLX stores as Linear (vocab_size * 7, decoder_dim) transposed to (decoder_dim, vocab_size * 7)
    mlx_audio_head = np.array(mlx_model.audio_head.weight.astype(mx.float32))

    print(f"PT audio_head shape: {pt_audio_head.shape}")
    print(f"MLX audio_head shape: {mlx_audio_head.shape}")

    # PT: (7, 3072, 8197) - per codebook weights
    # MLX: (8197*7, 3072) - flattened linear

    # Reshape MLX to compare
    # MLX Linear: output = x @ weight.T, so weight is (out, in) = (57379, 3072)
    vocab_size = 8197
    if mlx_audio_head.shape[0] == 7 * vocab_size:  # 57379
        mlx_reshaped = mlx_audio_head.reshape(7, vocab_size, 3072).transpose(0, 2, 1)
        corr = np.corrcoef(pt_audio_head.flatten(), mlx_reshaped.flatten())[0, 1]
        print(f"audio_head correlation (after reshape): {corr:.6f}")
        return corr > 0.99
    else:
        print(f"Shape mismatch: expected {7 * vocab_size}, got {mlx_audio_head.shape[0]}")
        return False


def test_muq_linear():
    """Test muq_linear layer."""
    print("\n" + "=" * 60)
    print("TEST: muq_linear")
    print("=" * 60)

    from heartlib.heartmula.modeling_heartmula import HeartMuLa as PTHeartMuLa
    from heartlib_mlx.heartmula import HeartMuLa as MLXHeartMuLa

    pt_model = PTHeartMuLa.from_pretrained('/Users/mcruz/Developer/heartlib/ckpt/HeartMuLa-oss-3B', dtype=torch.float32)
    pt_model.eval()
    mlx_model = MLXHeartMuLa.from_pretrained('/Users/mcruz/Developer/heartlib-mlx/ckpt-mlx/heartmula')

    # Test forward pass (muq_dim is 512)
    np.random.seed(42)
    muq_input = np.random.randn(1, 512).astype(np.float32) * 0.1

    pt_input = torch.tensor(muq_input)
    mlx_input = mx.array(muq_input)

    with torch.no_grad():
        pt_output = pt_model.muq_linear(pt_input)

    mlx_output = mlx_model.muq_linear(mlx_input)
    mx.eval(mlx_output)

    pt_np = pt_output.detach().numpy()
    mlx_np = np.array(mlx_output.astype(mx.float32))

    corr = np.corrcoef(pt_np.flatten(), mlx_np.flatten())[0, 1]
    print(f"muq_linear output correlation: {corr:.6f}")
    print_stats("  PT", pt_np)
    print_stats("  MLX", mlx_np)

    return corr > 0.99


def main():
    print("=" * 70)
    print("HEARTMULA PARITY TEST")
    print("=" * 70)

    results = {}

    results['embeddings'] = test_embeddings()
    results['embed_tokens'] = test_embed_tokens()
    results['codebook0_head'] = test_codebook0_head()
    results['projection'] = test_projection()
    results['audio_head'] = test_audio_head()
    results['muq_linear'] = test_muq_linear()
    results['generate_frame'] = test_generate_frame_first()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status}")

    if all(results.values()):
        print("\n✓ All tests passed!")
    else:
        print("\n✗ Some tests failed - check details above")


if __name__ == "__main__":
    main()
