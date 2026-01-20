#!/usr/bin/env python3
"""Example: Music generation with HeartMuLa.

This example demonstrates how to generate music from text descriptions
using the HeartMuLa model with HeartCodec for audio synthesis.

Usage:
    python examples/generate.py --output output.mp3 --duration 30

Requirements:
    - HeartMuLa and HeartCodec weights in ./ckpt-mlx/
    - Run `heartlib-convert` first to convert PyTorch weights
"""

import argparse
from pathlib import Path

import mlx.core as mx


def main():
    parser = argparse.ArgumentParser(description="Generate music with HeartMuLa")
    parser.add_argument(
        "--model-path",
        type=str,
        default="./ckpt-mlx",
        help="Path to converted MLX model weights",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output.mp3",
        help="Output audio file path",
    )
    parser.add_argument(
        "--lyrics",
        type=str,
        default="[Verse]\nThe sun is shining bright today\nWalking down the empty street\n[Chorus]\nFeel the rhythm, feel the beat\nMusic makes my heart complete",
        help="Lyrics text",
    )
    parser.add_argument(
        "--tags",
        type=str,
        default="pop, acoustic, upbeat, female vocal",
        help="Comma-separated music tags",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=30.0,
        help="Duration in seconds",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k sampling parameter",
    )
    parser.add_argument(
        "--cfg-scale",
        type=float,
        default=1.5,
        help="Classifier-free guidance scale",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()

    # Set random seed if provided
    if args.seed is not None:
        mx.random.seed(args.seed)

    print("Loading models...")
    from heartlib_mlx import HeartMuLaGenPipeline

    pipeline = HeartMuLaGenPipeline.from_pretrained(args.model_path)

    print(f"Generating {args.duration}s of music...")
    print(f"  Tags: {args.tags}")
    print(f"  Lyrics: {args.lyrics[:50]}...")

    audio = pipeline(
        lyrics=args.lyrics,
        tags=args.tags,
        duration=args.duration,
        temperature=args.temperature,
        top_k=args.top_k,
        cfg_scale=args.cfg_scale,
    )

    print(f"Saving to {args.output}...")
    pipeline.save_audio(audio, args.output)

    print("Done!")


if __name__ == "__main__":
    main()
