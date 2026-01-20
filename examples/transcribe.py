#!/usr/bin/env python3
"""Example: Lyrics transcription with HeartTranscriptor.

This example demonstrates how to transcribe lyrics from music audio
using the HeartTranscriptor model.

Usage:
    python examples/transcribe.py --input song.mp3
"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Transcribe lyrics with HeartTranscriptor")
    parser.add_argument(
        "--model-path",
        type=str,
        default="./ckpt-mlx/hearttranscriptor",
        help="Path to HeartTranscriptor weights",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input audio file",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        choices=["en", "zh", "ja", "ko", "es", "fr", "de", "ru", "pt"],
        help="Language code",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output text file (prints to stdout if not specified)",
    )

    args = parser.parse_args()

    print("Loading HeartTranscriptor...")
    from heartlib_mlx import HeartTranscriptorPipeline

    pipeline = HeartTranscriptorPipeline.from_pretrained(args.model_path)

    print(f"Transcribing {args.input}...")
    lyrics = pipeline.transcribe(args.input, language=args.language)

    print("\n" + "=" * 50)
    print("TRANSCRIBED LYRICS")
    print("=" * 50)
    print(lyrics)
    print("=" * 50)

    if args.output:
        with open(args.output, "w") as f:
            f.write(lyrics)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
