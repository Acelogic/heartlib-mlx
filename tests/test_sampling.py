"""Tests for sampling utilities."""

import pytest
import mlx.core as mx


class TestTopKSampling:
    """Tests for top-k sampling."""

    def test_greedy(self):
        """Test that temperature=0 gives greedy decoding."""
        from heartlib_mlx.utils.sampling import sample_topk

        logits = mx.array([[1.0, 2.0, 3.0, 0.5]])
        samples = sample_topk(logits, temperature=0.0, top_k=4)

        assert int(samples[0]) == 2  # Index of max value

    def test_respects_top_k(self):
        """Test that sampling respects top_k constraint."""
        from heartlib_mlx.utils.sampling import sample_topk

        mx.random.seed(42)

        # Create logits where one token is much more likely
        logits = mx.array([[10.0, 0.0, 0.0, 0.0, 0.0]])

        # With top_k=1, should always sample the most likely
        for _ in range(10):
            samples = sample_topk(logits, temperature=1.0, top_k=1)
            assert int(samples[0]) == 0


class TestTopPSampling:
    """Tests for nucleus (top-p) sampling."""

    def test_greedy(self):
        """Test that temperature=0 gives greedy decoding."""
        from heartlib_mlx.utils.sampling import sample_topp

        logits = mx.array([[1.0, 2.0, 3.0, 0.5]])
        samples = sample_topp(logits, temperature=0.0, top_p=0.9)

        assert int(samples[0]) == 2


class TestCFG:
    """Tests for classifier-free guidance."""

    def test_cfg_scale_1(self):
        """Test that cfg_scale=1.0 gives conditional logits."""
        from heartlib_mlx.utils.sampling import apply_cfg

        cond = mx.array([[1.0, 2.0, 3.0]])
        uncond = mx.array([[0.0, 0.0, 0.0]])

        result = apply_cfg(cond, uncond, cfg_scale=1.0)

        assert mx.allclose(result, cond, atol=1e-6)

    def test_cfg_scale_0(self):
        """Test that cfg_scale=0.0 gives unconditional logits."""
        from heartlib_mlx.utils.sampling import apply_cfg

        cond = mx.array([[1.0, 2.0, 3.0]])
        uncond = mx.array([[0.5, 0.5, 0.5]])

        result = apply_cfg(cond, uncond, cfg_scale=0.0)

        assert mx.allclose(result, uncond, atol=1e-6)

    def test_cfg_scale_2(self):
        """Test that cfg_scale=2.0 doubles the difference."""
        from heartlib_mlx.utils.sampling import apply_cfg

        cond = mx.array([[2.0, 2.0, 2.0]])
        uncond = mx.array([[1.0, 1.0, 1.0]])

        result = apply_cfg(cond, uncond, cfg_scale=2.0)

        # uncond + 2 * (cond - uncond) = 1 + 2 * 1 = 3
        expected = mx.array([[3.0, 3.0, 3.0]])
        assert mx.allclose(result, expected, atol=1e-6)


class TestRepetitionPenalty:
    """Tests for repetition penalty."""

    def test_penalty_reduces_likelihood(self):
        """Test that penalty reduces likelihood of repeated tokens."""
        from heartlib_mlx.utils.sampling import apply_repetition_penalty

        logits = mx.array([[1.0, 2.0, 3.0, 4.0]])
        generated = mx.array([[2]])  # Token 2 was generated

        penalized = apply_repetition_penalty(logits, generated, penalty=2.0)

        # Logit at position 2 should be reduced
        assert float(penalized[0, 2]) < float(logits[0, 2])
