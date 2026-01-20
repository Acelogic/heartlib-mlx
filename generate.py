#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "mlx>=0.22.0",
#     "numpy>=1.24.0",
#     "safetensors>=0.4.0",
#     "huggingface-hub>=0.20",
#     "tokenizers>=0.15",
#     "soundfile>=0.12",
#     "tqdm>=4.60.0",
# ]
# ///
"""
Generate music with HeartMuLa MLX.

Usage with uv (no install needed):
    uv run generate.py --tags "pop, acoustic" --lyrics "[verse]\nHello world"

Usage with pip:
    pip install -e .
    python generate.py --tags "pop, acoustic" --lyrics "[verse]\nHello world"
"""

import argparse
import sys
from pathlib import Path

# Add src to path for development
src_path = Path(__file__).parent / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))


def main():
    parser = argparse.ArgumentParser(
        description="Generate music with HeartMuLa MLX",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Simple generation
  uv run generate.py --tags "electronic, ambient" --output ambient.wav

  # With lyrics
  uv run generate.py --tags "pop, acoustic" --lyrics "[verse]\\nHello world" --output song.wav

  # Longer generation with custom settings
  uv run generate.py --tags "jazz, piano" --duration 30 --cfg-scale 2.0 --output jazz.wav
""",
    )
    parser.add_argument(
        "--tags", "-t",
        type=str,
        default="electronic, ambient, instrumental",
        help="Music style tags (default: electronic, ambient, instrumental)",
    )
    parser.add_argument(
        "--lyrics", "-l",
        type=str,
        default="",
        help="Lyrics with structure tags like [verse], [chorus] (optional)",
    )
    parser.add_argument(
        "--duration", "-d",
        type=float,
        default=10.0,
        help="Duration in seconds (default: 10)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="output.wav",
        help="Output file path (default: output.wav)",
    )
    parser.add_argument(
        "--cfg-scale",
        type=float,
        default=1.5,
        help="Classifier-free guidance scale (default: 1.5)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (default: 1.0)",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=50,
        help="Top-k sampling (default: 50)",
    )
    parser.add_argument(
        "--checkpoint", "-c",
        type=str,
        default=None,
        help="Checkpoint directory (default: auto-detect)",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=48000,
        help="Output sample rate (default: 48000)",
    )
    args = parser.parse_args()

    # Find checkpoint
    ckpt_path = args.checkpoint
    if ckpt_path is None:
        candidates = [
            Path(__file__).parent / "ckpt-mlx",
            Path.home() / "Developer/heartlib-mlx/ckpt-mlx",
            Path("./ckpt-mlx"),
        ]
        for c in candidates:
            if c.exists() and (c / "heartmula").exists():
                ckpt_path = str(c)
                break

    if ckpt_path is None or not Path(ckpt_path).exists():
        print("Error: Checkpoint not found. Please specify --checkpoint or run:")
        print("  python -m heartlib_mlx.utils.convert --src ./ckpt --dst ./ckpt-mlx")
        sys.exit(1)

    print(f"Using checkpoint: {ckpt_path}")
    print(f"Tags: {args.tags}")
    print(f"Lyrics: {args.lyrics[:50] + '...' if len(args.lyrics) > 50 else args.lyrics or '(none)'}")
    print(f"Duration: {args.duration}s")
    print(f"Output: {args.output}")
    print()

    # Import after args parsing for faster --help
    import mlx.core as mx
    import numpy as np
    import soundfile as sf
    from tqdm import tqdm
    from tokenizers import Tokenizer

    from heartlib_mlx.heartmula import HeartMuLa
    from heartlib_mlx.heartcodec import HeartCodec

    # Load models
    print("Loading models...")
    model = HeartMuLa.from_pretrained(f"{ckpt_path}/heartmula")
    codec = HeartCodec.from_pretrained(f"{ckpt_path}/heartcodec")

    # Load tokenizer
    tokenizer_path = Path(ckpt_path).parent / "ckpt" / "tokenizer.json"
    if not tokenizer_path.exists():
        tokenizer_path = Path.home() / "Developer/heartlib/ckpt/tokenizer.json"
    if not tokenizer_path.exists():
        print(f"Error: Tokenizer not found at {tokenizer_path}")
        sys.exit(1)

    tokenizer = Tokenizer.from_file(str(tokenizer_path))

    # Config
    text_bos_id = 128000
    text_eos_id = 128001
    audio_eos_id = 8193
    num_codebooks = 8
    parallel = num_codebooks + 1
    frame_rate = 12.5

    # Tokenize
    tags_text = f"<tag>{args.tags}</tag>"
    tags_ids = tokenizer.encode(tags_text.lower()).ids
    if tags_ids[0] != text_bos_id:
        tags_ids = [text_bos_id] + tags_ids
    if tags_ids[-1] != text_eos_id:
        tags_ids = tags_ids + [text_eos_id]

    if args.lyrics:
        lyrics_ids = tokenizer.encode(args.lyrics.lower()).ids
        if lyrics_ids[0] != text_bos_id:
            lyrics_ids = [text_bos_id] + lyrics_ids
        if lyrics_ids[-1] != text_eos_id:
            lyrics_ids = lyrics_ids + [text_eos_id]
        prompt_len = len(tags_ids) + 1 + len(lyrics_ids)
        muq_idx = len(tags_ids)  # Position between tags and lyrics
    else:
        lyrics_ids = []
        # Add 1 for the muq embedding position even without lyrics
        prompt_len = len(tags_ids) + 1
        muq_idx = len(tags_ids)  # Position after tags

    # Build prompt
    prompt_tokens = np.zeros((prompt_len, parallel), dtype=np.int64)
    prompt_tokens[:len(tags_ids), -1] = tags_ids
    if lyrics_ids:
        prompt_tokens[len(tags_ids) + 1:, -1] = lyrics_ids

    prompt_mask = np.zeros((prompt_len, parallel), dtype=np.float32)
    prompt_mask[:, -1] = 1.0

    # Setup for CFG (batch=2)
    tokens = mx.array(prompt_tokens)[None, :, :]
    tokens = mx.concatenate([tokens, tokens], axis=0)
    mask = mx.array(prompt_mask)[None, :, :]
    mask = mx.concatenate([mask, mask], axis=0)

    muq_embed = mx.zeros((2, model.config.muq_dim))
    pos = mx.broadcast_to(mx.arange(prompt_len)[None, :], (2, prompt_len))

    max_frames = int(args.duration * frame_rate)
    model.setup_caches(2)

    # Generate
    print(f"Generating {max_frames} frames...")
    frames = []

    curr_token = model.generate_frame(
        tokens=tokens,
        tokens_mask=mask,
        input_pos=pos,
        temperature=args.temperature,
        topk=args.topk,
        cfg_scale=args.cfg_scale,
        continuous_segments=muq_embed,
        starts=[muq_idx, muq_idx],
    )
    mx.eval(curr_token)
    frames.append(curr_token[0:1])

    for i in tqdm(range(max_frames - 1), desc="Generating"):
        padded = mx.concatenate([
            curr_token[:, None, :],
            mx.zeros((2, 1, 1), dtype=mx.int32)
        ], axis=-1)
        padded_mask = mx.concatenate([
            mx.ones((2, 1, num_codebooks)),
            mx.zeros((2, 1, 1))
        ], axis=-1)

        curr_token = model.generate_frame(
            tokens=padded,
            tokens_mask=padded_mask,
            input_pos=pos[:, -1:] + i + 1,
            temperature=args.temperature,
            topk=args.topk,
            cfg_scale=args.cfg_scale,
        )
        mx.eval(curr_token)

        if mx.any(curr_token[0] >= audio_eos_id):
            break
        frames.append(curr_token[0:1])

    print(f"Generated {len(frames)} frames")

    # Decode
    print("Decoding audio...")
    frames_arr = mx.concatenate(frames, axis=0)[None, :, :]
    mx.eval(frames_arr)

    audio = codec.detokenize(frames_arr, duration=len(frames) / frame_rate)
    mx.eval(audio)
    audio_np = np.array(audio.astype(mx.float32)).flatten()

    # Save
    sf.write(args.output, audio_np, args.sample_rate)
    print(f"Saved: {args.output} ({len(audio_np) / args.sample_rate:.2f}s)")


if __name__ == "__main__":
    main()
