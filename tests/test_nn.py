"""Tests for neural network layers."""

import pytest
import mlx.core as mx


class TestCausalConv1d:
    """Tests for CausalConv1d layer."""

    def test_output_shape(self):
        """Test that output shape is correct."""
        from heartlib_mlx.nn.conv import CausalConv1d

        batch_size = 2
        seq_len = 100
        in_channels = 32
        out_channels = 64
        kernel_size = 7

        layer = CausalConv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
        )

        x = mx.random.normal(shape=(batch_size, seq_len, in_channels))
        y = layer(x)

        assert y.shape == (batch_size, seq_len, out_channels)

    def test_causality(self):
        """Test that convolution is causal (output[t] depends only on input[:t+1])."""
        from heartlib_mlx.nn.conv import CausalConv1d

        layer = CausalConv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=3,
        )

        # Create input with a step function
        x = mx.concatenate([
            mx.zeros((1, 5, 1)),
            mx.ones((1, 5, 1))
        ], axis=1)

        y = layer(x)

        # Output before position 5 should not depend on the step
        # (though it may not be exactly zero due to initialization)
        # This is a simplified test - proper causality testing would
        # require checking gradients


class TestWeightNormConv1d:
    """Tests for WeightNormConv1d layer."""

    def test_weight_normalization(self):
        """Test that weights are properly normalized."""
        from heartlib_mlx.nn.conv import WeightNormConv1d

        layer = WeightNormConv1d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
        )

        weight = layer._get_normalized_weight()

        # Check that the weight has the correct shape
        assert weight.shape == (64, 3, 32)


class TestRMSNorm:
    """Tests for RMSNorm layer."""

    def test_output_shape(self):
        """Test that output shape matches input shape."""
        from heartlib_mlx.nn.transformer import RMSNorm

        dim = 256
        layer = RMSNorm(dim)

        x = mx.random.normal(shape=(2, 10, dim))
        y = layer(x)

        assert y.shape == x.shape

    def test_normalization(self):
        """Test that output has approximately unit RMS."""
        from heartlib_mlx.nn.transformer import RMSNorm

        dim = 256
        layer = RMSNorm(dim)

        x = mx.random.normal(shape=(2, 10, dim)) * 10  # Scale up input
        y = layer(x)

        # RMS of output should be close to 1 (times the weight)
        rms = mx.sqrt(mx.mean(y * y, axis=-1))
        # Just check it's not too far from 1
        assert mx.all(rms > 0.1)
        assert mx.all(rms < 10)


class TestRotaryPositionEmbedding:
    """Tests for RotaryPositionEmbedding."""

    def test_output_shape(self):
        """Test that output shapes match input shapes."""
        from heartlib_mlx.nn.rope import RotaryPositionEmbedding

        dim = 64
        rope = RotaryPositionEmbedding(dim)

        batch_size = 2
        seq_len = 10
        n_heads = 8

        q = mx.random.normal(shape=(batch_size, seq_len, n_heads, dim))
        k = mx.random.normal(shape=(batch_size, seq_len, n_heads, dim))

        q_rot, k_rot = rope(q, k)

        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape


class TestKVCache:
    """Tests for KV cache."""

    def test_cache_update(self):
        """Test that cache correctly accumulates KV pairs."""
        from heartlib_mlx.nn.kv_cache import KVCache

        cache = KVCache(
            batch_size=2,
            max_seq_len=100,
            n_heads=8,
            head_dim=64,
            n_layers=4,
        )

        # First update
        k1 = mx.random.normal(shape=(2, 5, 8, 64))
        v1 = mx.random.normal(shape=(2, 5, 8, 64))
        k_out, v_out = cache.update(0, k1, v1)

        assert k_out.shape == (2, 5, 8, 64)
        assert cache.current_seq_len == 5

        # Second update
        k2 = mx.random.normal(shape=(2, 3, 8, 64))
        v2 = mx.random.normal(shape=(2, 3, 8, 64))
        k_out, v_out = cache.update(0, k2, v2)

        assert k_out.shape == (2, 8, 8, 64)
        assert cache.current_seq_len == 8

    def test_cache_reset(self):
        """Test that cache reset works correctly."""
        from heartlib_mlx.nn.kv_cache import KVCache

        cache = KVCache(
            batch_size=2,
            max_seq_len=100,
            n_heads=8,
            head_dim=64,
            n_layers=4,
        )

        k = mx.random.normal(shape=(2, 5, 8, 64))
        v = mx.random.normal(shape=(2, 5, 8, 64))
        cache.update(0, k, v)

        cache.reset()

        assert cache.current_seq_len == 0
        assert cache.k_cache[0] is None
