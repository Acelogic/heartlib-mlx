"""Parity test between PyTorch and MLX HeartCodec."""

import numpy as np
import torch
import mlx.core as mx
import sys

sys.path.insert(0, '/Users/mcruz/Developer/heartlib/src')
sys.path.insert(0, '/Users/mcruz/Developer/heartlib-mlx/src')


def print_stats(name, arr):
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()
    elif isinstance(arr, mx.array):
        arr = np.array(arr.astype(mx.float32))
    print(f"{name}: Range [{arr.min():.4f}, {arr.max():.4f}], Std: {arr.std():.4f}")


def test_scalar_decode():
    """Test scalar model decode with fixed latent."""
    print("\n" + "=" * 60)
    print("TEST: Scalar model decode")
    print("=" * 60)

    from heartlib.heartcodec.modeling_heartcodec import HeartCodec as PTHeartCodec
    from heartlib_mlx.heartcodec import HeartCodec as MLXHeartCodec

    pt_codec = PTHeartCodec.from_pretrained('/Users/mcruz/Developer/heartlib/ckpt/HeartCodec-oss')
    pt_codec.eval()
    mlx_codec = MLXHeartCodec.from_pretrained('/Users/mcruz/Developer/heartlib-mlx/ckpt-mlx/heartcodec')

    # Fixed latent (already in scalar model format: batch, time, 128)
    np.random.seed(42)
    fixed_latent = np.random.randn(2, 250, 128).astype(np.float32) * 0.35

    # PyTorch: expects (batch, channels, time)
    pt_latent = torch.tensor(fixed_latent).transpose(1, 2)
    with torch.no_grad():
        pt_audio = pt_codec.scalar_model.decode(pt_latent)
    pt_audio_np = pt_audio.squeeze().cpu().numpy().flatten()

    # MLX: expects (batch, time, channels)
    mlx_latent = mx.array(fixed_latent)
    mlx_audio = mlx_codec.scalar_model.decode(mlx_latent)
    mx.eval(mlx_audio)
    mlx_audio_np = np.array(mlx_audio.astype(mx.float32)).flatten()

    min_len = min(len(pt_audio_np), len(mlx_audio_np))
    corr = np.corrcoef(pt_audio_np[:min_len], mlx_audio_np[:min_len])[0, 1]

    print_stats("PT audio", pt_audio_np)
    print_stats("MLX audio", mlx_audio_np)
    print(f"Correlation: {corr:.6f}")
    return corr > 0.9


def main():
    print("=" * 70)
    print("HEARTCODEC PARITY TEST")
    print("=" * 70)

    # Test 1: Scalar decode
    scalar_pass = test_scalar_decode()

    # Test 2: Full estimator
    print("\n" + "=" * 60)
    print("TEST: Estimator with fixed input")
    print("=" * 60)

    from heartlib.heartcodec.modeling_heartcodec import HeartCodec as PTHeartCodec
    from heartlib_mlx.heartcodec import HeartCodec as MLXHeartCodec

    pt_codec = PTHeartCodec.from_pretrained('/Users/mcruz/Developer/heartlib/ckpt/HeartCodec-oss')
    pt_codec.eval()
    mlx_codec = MLXHeartCodec.from_pretrained('/Users/mcruz/Developer/heartlib-mlx/ckpt-mlx/heartcodec')

    np.random.seed(42)
    hidden_states = np.random.randn(1, 250, 1024).astype(np.float32) * 0.1
    t = np.array([0.5], dtype=np.float32)

    pt_hidden = torch.tensor(hidden_states)
    pt_t = torch.tensor(t)
    mlx_hidden = mx.array(hidden_states)
    mlx_t = mx.array(t)

    with torch.no_grad():
        pt_output = pt_codec.flow_matching.estimator(pt_hidden, timestep=pt_t)

    mlx_output = mlx_codec.flow_matching.estimator(mlx_t, mlx_hidden)
    mx.eval(mlx_output)

    pt_out_np = pt_output.cpu().numpy()
    mlx_out_np = np.array(mlx_output.astype(mx.float32))

    corr = np.corrcoef(pt_out_np.flatten(), mlx_out_np.flatten())[0, 1]
    print_stats("PT output", pt_out_np)
    print_stats("MLX output", mlx_out_np)
    print(f"Correlation: {corr:.6f}")
    estimator_pass = corr > 0.99

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Scalar decode: {'✓ PASS' if scalar_pass else '✗ FAIL'}")
    print(f"Estimator: {'✓ PASS' if estimator_pass else '✗ FAIL'}")

    if scalar_pass and estimator_pass:
        print("\n✓ All critical components pass parity test!")
    else:
        print("\n✗ Some components failed - check details above")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
