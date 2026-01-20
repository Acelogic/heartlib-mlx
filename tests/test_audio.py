"""Tests for audio utilities."""

import pytest
import numpy as np
import mlx.core as mx


class TestResample:
    """Tests for audio resampling."""

    def test_same_rate(self):
        """Test that same rate returns unchanged audio."""
        from heartlib_mlx.utils.audio import resample

        audio = np.random.randn(1000).astype(np.float32)
        resampled = resample(audio, 44100, 44100)

        np.testing.assert_array_equal(audio, resampled)

    def test_upsample(self):
        """Test upsampling doubles the length."""
        from heartlib_mlx.utils.audio import resample

        audio = np.random.randn(1000).astype(np.float32)
        resampled = resample(audio, 22050, 44100)

        assert len(resampled) == 2000

    def test_downsample(self):
        """Test downsampling halves the length."""
        from heartlib_mlx.utils.audio import resample

        audio = np.random.randn(1000).astype(np.float32)
        resampled = resample(audio, 44100, 22050)

        assert len(resampled) == 500


class TestPadOrTruncate:
    """Tests for pad_or_truncate function."""

    def test_truncate(self):
        """Test truncation to target length."""
        from heartlib_mlx.utils.audio import pad_or_truncate

        audio = mx.random.normal(shape=(1000,))
        result = pad_or_truncate(audio, 500)

        assert result.shape == (500,)

    def test_pad(self):
        """Test padding to target length."""
        from heartlib_mlx.utils.audio import pad_or_truncate

        audio = mx.random.normal(shape=(500,))
        result = pad_or_truncate(audio, 1000)

        assert result.shape == (1000,)
        # Check that padded portion is zeros
        assert mx.allclose(result[500:], mx.zeros(500), atol=1e-6)

    def test_exact_length(self):
        """Test that exact length returns unchanged."""
        from heartlib_mlx.utils.audio import pad_or_truncate

        audio = mx.random.normal(shape=(1000,))
        result = pad_or_truncate(audio, 1000)

        assert mx.allclose(result, audio, atol=1e-6)


class TestSplitAudio:
    """Tests for audio splitting."""

    def test_no_overlap(self):
        """Test splitting without overlap."""
        from heartlib_mlx.utils.audio import split_audio

        audio = mx.arange(100).astype(mx.float32)
        chunks = split_audio(audio, chunk_length=25, hop_length=25, pad=False)

        assert chunks.shape == (4, 25)

    def test_with_overlap(self):
        """Test splitting with overlap."""
        from heartlib_mlx.utils.audio import split_audio

        audio = mx.arange(100).astype(mx.float32)
        chunks = split_audio(audio, chunk_length=30, hop_length=20, pad=False)

        assert chunks.shape[1] == 30
        # Number of chunks: (100 - 30) / 20 + 1 = 4
        assert chunks.shape[0] == 4


class TestMergeChunks:
    """Tests for chunk merging."""

    def test_round_trip(self):
        """Test that split and merge gives back original (approximately)."""
        from heartlib_mlx.utils.audio import split_audio, merge_chunks

        audio = mx.random.normal(shape=(1000,))
        chunk_length = 100
        hop_length = 50

        chunks = split_audio(audio, chunk_length, hop_length, pad=True)
        merged = merge_chunks(chunks, hop_length, total_length=1000)

        # Should be close to original
        assert mx.allclose(merged[:900], audio[:900], atol=0.1)
