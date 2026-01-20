"""High-level inference pipelines."""

from heartlib_mlx.pipelines.music_generation import HeartMuLaGenPipeline
from heartlib_mlx.pipelines.audio_tagging import HeartCLAPPipeline
from heartlib_mlx.pipelines.lyrics_transcription import HeartTranscriptorPipeline

__all__ = [
    "HeartMuLaGenPipeline",
    "HeartCLAPPipeline",
    "HeartTranscriptorPipeline",
]
