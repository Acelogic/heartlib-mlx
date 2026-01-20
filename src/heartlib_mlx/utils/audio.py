"""Audio I/O utilities."""

from typing import Optional, Tuple, Union
from pathlib import Path

import numpy as np
import mlx.core as mx


def load_audio(
    path: Union[str, Path],
    sample_rate: Optional[int] = None,
    mono: bool = True,
    normalize: bool = True,
    return_sample_rate: bool = False,
) -> Union[mx.array, Tuple[mx.array, int]]:
    """Load audio from file.

    Args:
        path: Path to audio file.
        sample_rate: Target sample rate. If None, uses original.
        mono: Convert to mono if True.
        normalize: Normalize audio to [-1, 1] if True.
        return_sample_rate: Return sample rate along with audio.

    Returns:
        Audio waveform as MLX array, optionally with sample rate.
    """
    import soundfile as sf

    # Load audio
    audio, sr = sf.read(str(path), dtype="float32")

    # Convert to mono
    if mono and audio.ndim > 1:
        audio = audio.mean(axis=-1)

    # Resample if needed
    if sample_rate is not None and sr != sample_rate:
        audio = resample(audio, sr, sample_rate)
        sr = sample_rate

    # Normalize
    if normalize:
        max_val = np.abs(audio).max()
        if max_val > 0:
            audio = audio / max_val

    # Convert to MLX
    audio_mlx = mx.array(audio)

    if return_sample_rate:
        return audio_mlx, sr
    return audio_mlx


def save_audio(
    audio: Union[mx.array, np.ndarray],
    path: Union[str, Path],
    sample_rate: int = 48000,
    normalize: bool = True,
    format: Optional[str] = None,
) -> None:
    """Save audio to file.

    Args:
        audio: Audio waveform.
        path: Output file path.
        sample_rate: Sample rate.
        normalize: Normalize audio before saving.
        format: Audio format (inferred from extension if None).
    """
    import soundfile as sf

    # Convert to numpy
    if isinstance(audio, mx.array):
        audio = np.array(audio)

    # Handle shape
    if audio.ndim == 3:
        audio = audio[0]  # Remove batch dim
    if audio.ndim == 2 and audio.shape[-1] == 1:
        audio = audio[:, 0]  # Remove channel dim

    # Normalize
    if normalize:
        max_val = np.abs(audio).max()
        if max_val > 0:
            audio = audio / max_val

    # Determine format from extension
    path = Path(path)
    if format is None:
        format = path.suffix.lower().lstrip(".")

    # Save
    sf.write(str(path), audio, sample_rate, format=format)


def resample(
    audio: np.ndarray,
    orig_sr: int,
    target_sr: int,
) -> np.ndarray:
    """Resample audio to target sample rate.

    Simple linear interpolation resampling.
    For better quality, use librosa or torchaudio.

    Args:
        audio: Audio waveform.
        orig_sr: Original sample rate.
        target_sr: Target sample rate.

    Returns:
        Resampled audio.
    """
    if orig_sr == target_sr:
        return audio

    ratio = target_sr / orig_sr
    new_len = int(len(audio) * ratio)

    # Linear interpolation
    indices = np.linspace(0, len(audio) - 1, new_len)
    return np.interp(indices, np.arange(len(audio)), audio).astype(audio.dtype)


def pad_or_truncate(
    audio: Union[mx.array, np.ndarray],
    target_length: int,
    pad_value: float = 0.0,
) -> Union[mx.array, np.ndarray]:
    """Pad or truncate audio to target length.

    Args:
        audio: Audio waveform.
        target_length: Target length in samples.
        pad_value: Value to use for padding.

    Returns:
        Audio with target length.
    """
    is_mlx = isinstance(audio, mx.array)
    if is_mlx:
        audio = np.array(audio)

    current_length = len(audio)

    if current_length > target_length:
        audio = audio[:target_length]
    elif current_length < target_length:
        padding = np.full(target_length - current_length, pad_value, dtype=audio.dtype)
        audio = np.concatenate([audio, padding])

    if is_mlx:
        audio = mx.array(audio)

    return audio


def split_audio(
    audio: mx.array,
    chunk_length: int,
    hop_length: Optional[int] = None,
    pad: bool = True,
) -> mx.array:
    """Split audio into overlapping chunks.

    Args:
        audio: Audio waveform of shape (samples,) or (batch, samples).
        chunk_length: Length of each chunk.
        hop_length: Hop between chunks. If None, uses chunk_length (no overlap).
        pad: Pad the last chunk if True.

    Returns:
        Chunks of shape (num_chunks, chunk_length) or (batch, num_chunks, chunk_length).
    """
    hop_length = hop_length or chunk_length

    # Handle batch dimension
    has_batch = audio.ndim == 2
    if not has_batch:
        audio = audio[None, :]

    batch_size, total_length = audio.shape

    # Calculate number of chunks
    if pad:
        num_chunks = (total_length + hop_length - 1) // hop_length
        # Pad audio
        padded_length = (num_chunks - 1) * hop_length + chunk_length
        if padded_length > total_length:
            padding = mx.zeros((batch_size, padded_length - total_length))
            audio = mx.concatenate([audio, padding], axis=1)
    else:
        num_chunks = (total_length - chunk_length) // hop_length + 1

    # Extract chunks
    chunks = []
    for i in range(num_chunks):
        start = i * hop_length
        chunks.append(audio[:, start:start + chunk_length])

    chunks = mx.stack(chunks, axis=1)

    if not has_batch:
        chunks = chunks[0]

    return chunks


def merge_chunks(
    chunks: mx.array,
    hop_length: int,
    total_length: Optional[int] = None,
) -> mx.array:
    """Merge overlapping audio chunks.

    Uses linear crossfade for overlapping regions.

    Args:
        chunks: Audio chunks of shape (num_chunks, chunk_length) or
            (batch, num_chunks, chunk_length).
        hop_length: Hop between chunks.
        total_length: Target total length (truncates if specified).

    Returns:
        Merged audio.
    """
    # Handle batch dimension
    has_batch = chunks.ndim == 3
    if not has_batch:
        chunks = chunks[None, :, :]

    batch_size, num_chunks, chunk_length = chunks.shape

    # Calculate output length
    output_length = (num_chunks - 1) * hop_length + chunk_length

    # Initialize output
    output = mx.zeros((batch_size, output_length))
    weights = mx.zeros((batch_size, output_length))

    # Crossfade window
    overlap = chunk_length - hop_length
    if overlap > 0:
        fade_in = mx.linspace(0, 1, overlap)
        fade_out = 1 - fade_in
        window = mx.concatenate([
            fade_in,
            mx.ones(hop_length),
            fade_out if overlap < chunk_length else mx.array([]),
        ])[:chunk_length]
    else:
        window = mx.ones(chunk_length)

    # Merge chunks
    for i in range(num_chunks):
        start = i * hop_length
        end = start + chunk_length
        output = output.at[:, start:end].add(chunks[:, i, :] * window)
        weights = weights.at[:, start:end].add(window)

    # Normalize by weights
    output = output / mx.maximum(weights, 1e-8)

    if total_length is not None:
        output = output[:, :total_length]

    if not has_batch:
        output = output[0]

    return output


def compute_rms(audio: mx.array, frame_length: int = 2048, hop_length: int = 512) -> mx.array:
    """Compute RMS energy of audio.

    Args:
        audio: Audio waveform.
        frame_length: Analysis frame length.
        hop_length: Hop between frames.

    Returns:
        RMS values per frame.
    """
    # Square
    audio_sq = audio ** 2

    # Split into frames
    chunks = split_audio(audio_sq, frame_length, hop_length, pad=True)

    # Compute mean and sqrt
    rms = mx.sqrt(mx.mean(chunks, axis=-1))

    return rms
