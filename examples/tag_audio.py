#!/usr/bin/env python3
"""Example: Audio tagging with HeartCLAP.

This example demonstrates how to tag audio files with music descriptors
using the HeartCLAP audio-text alignment model.

Usage:
    python examples/tag_audio.py --input song.mp3 --top-k 10
"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Tag audio with HeartCLAP")
    parser.add_argument(
        "--model-path",
        type=str,
        default="./ckpt-mlx/heartclap",
        help="Path to HeartCLAP weights",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input audio file",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top tags to return",
    )
    parser.add_argument(
        "--custom-tags",
        type=str,
        default=None,
        help="Custom comma-separated tags to score against",
    )

    args = parser.parse_args()

    print("Loading HeartCLAP...")
    from heartlib_mlx import HeartCLAPPipeline

    # Parse custom tags if provided
    custom_tags = None
    if args.custom_tags:
        custom_tags = [t.strip() for t in args.custom_tags.split(",")]

    pipeline = HeartCLAPPipeline.from_pretrained(
        args.model_path,
        tags=custom_tags,
    )

    print(f"Analyzing {args.input}...")
    tags = pipeline.tag_audio(args.input, top_k=args.top_k)

    print("\nDetected tags:")
    print("-" * 40)
    for tag, score in tags:
        bar = "█" * int(score * 20)
        print(f"  {tag:20s} {score:.3f} {bar}")


if __name__ == "__main__":
    main()
