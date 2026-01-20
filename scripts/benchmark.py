#!/usr/bin/env python3
"""Benchmark PyTorch vs MLX HeartMuLa performance."""

import argparse
import time
import sys
import os

import numpy as np


def benchmark_pytorch(ckpt_path: str, num_frames: int = 50, warmup: int = 5):
    """Benchmark PyTorch HeartMuLa."""
    import torch

    sys.path.insert(0, os.path.expanduser('~/Developer/heartlib/src'))
    from heartlib.heartmula.modeling_heartmula import HeartMuLa
    from heartlib.heartcodec.modeling_heartcodec import HeartCodec

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"PyTorch device: {device}")

    # Load models
    print("Loading PyTorch models...")
    load_start = time.perf_counter()
    model = HeartMuLa.from_pretrained(f'{ckpt_path}/HeartMuLa-oss-3B', dtype=torch.float32)
    model = model.to(device)
    model.eval()
    codec = HeartCodec.from_pretrained(f'{ckpt_path}/HeartCodec-oss', device_map=device)
    load_time = time.perf_counter() - load_start
    print(f"Load time: {load_time:.2f}s")

    # Setup
    num_codebooks = 8
    parallel = num_codebooks + 1
    seq_len = 10

    tokens = torch.zeros((2, seq_len, parallel), dtype=torch.long, device=device)
    tokens[:, :, -1] = torch.tensor([128000, 100, 200, 300, 400, 500, 600, 700, 800, 128001])
    mask = torch.zeros((2, seq_len, parallel), dtype=torch.bool, device=device)
    mask[:, :, -1] = True

    muq_embed = torch.zeros((2, model.config.muq_dim), dtype=torch.float32, device=device)
    pos = torch.arange(seq_len, device=device).unsqueeze(0).repeat(2, 1)

    model.setup_caches(2)

    # Warmup
    print(f"Warming up ({warmup} frames)...")
    with torch.no_grad():
        curr_token = model.generate_frame(
            tokens=tokens, tokens_mask=mask, input_pos=pos,
            temperature=1.0, topk=50, cfg_scale=1.5,
            continuous_segments=muq_embed, starts=[1, 1]
        )
        for i in range(warmup - 1):
            padded = torch.zeros((2, 1, parallel), dtype=torch.long, device=device)
            padded[:, 0, :-1] = curr_token
            padded_mask = torch.ones_like(padded, dtype=torch.bool)
            padded_mask[..., -1] = False
            curr_token = model.generate_frame(
                tokens=padded, tokens_mask=padded_mask, input_pos=pos[:, -1:] + i + 1,
                temperature=1.0, topk=50, cfg_scale=1.5
            )

    # Reset caches
    model.setup_caches(2)

    # Benchmark generation
    print(f"Benchmarking generation ({num_frames} frames)...")
    frames = []
    gen_start = time.perf_counter()

    with torch.no_grad():
        curr_token = model.generate_frame(
            tokens=tokens, tokens_mask=mask, input_pos=pos,
            temperature=1.0, topk=50, cfg_scale=1.5,
            continuous_segments=muq_embed, starts=[1, 1]
        )
        frames.append(curr_token[0:1])

        for i in range(num_frames - 1):
            padded = torch.zeros((2, 1, parallel), dtype=torch.long, device=device)
            padded[:, 0, :-1] = curr_token
            padded_mask = torch.ones_like(padded, dtype=torch.bool)
            padded_mask[..., -1] = False
            curr_token = model.generate_frame(
                tokens=padded, tokens_mask=padded_mask, input_pos=pos[:, -1:] + i + 1,
                temperature=1.0, topk=50, cfg_scale=1.5
            )
            frames.append(curr_token[0:1])

    gen_time = time.perf_counter() - gen_start

    # Decode
    print("Benchmarking decode...")
    frames_tensor = torch.stack(frames).permute(1, 2, 0).squeeze(0)
    decode_start = time.perf_counter()
    audio = codec.detokenize(frames_tensor)
    decode_time = time.perf_counter() - decode_start

    return {
        "load_time": load_time,
        "gen_time": gen_time,
        "gen_fps": num_frames / gen_time,
        "decode_time": decode_time,
        "num_frames": num_frames,
        "audio_duration": num_frames / 12.5,
    }


def benchmark_mlx(ckpt_path: str, num_frames: int = 50, warmup: int = 5):
    """Benchmark MLX HeartMuLa."""
    import mlx.core as mx

    sys.path.insert(0, os.path.expanduser('~/Developer/heartlib-mlx/src'))
    from heartlib_mlx.heartmula import HeartMuLa
    from heartlib_mlx.heartcodec import HeartCodec

    print("MLX device: Apple Silicon GPU")

    # Load models
    print("Loading MLX models...")
    load_start = time.perf_counter()
    model = HeartMuLa.from_pretrained(f'{ckpt_path}/heartmula')
    codec = HeartCodec.from_pretrained(f'{ckpt_path}/heartcodec')
    load_time = time.perf_counter() - load_start
    print(f"Load time: {load_time:.2f}s")

    # Setup
    num_codebooks = 8
    parallel = num_codebooks + 1
    seq_len = 10

    tokens = mx.zeros((2, seq_len, parallel), dtype=mx.int32)
    tokens = tokens.at[:, :, -1].add(mx.array([128000, 100, 200, 300, 400, 500, 600, 700, 800, 128001]))
    mask = mx.zeros((2, seq_len, parallel))
    mask = mask.at[:, :, -1].add(1.0)

    muq_embed = mx.zeros((2, model.config.muq_dim))
    pos = mx.broadcast_to(mx.arange(seq_len)[None, :], (2, seq_len))

    model.setup_caches(2)

    # Warmup
    print(f"Warming up ({warmup} frames)...")
    curr_token = model.generate_frame(
        tokens=tokens, tokens_mask=mask, input_pos=pos,
        temperature=1.0, topk=50, cfg_scale=1.5,
        continuous_segments=muq_embed, starts=[1, 1]
    )
    mx.eval(curr_token)

    for i in range(warmup - 1):
        padded = mx.concatenate([curr_token[:, None, :], mx.zeros((2, 1, 1), dtype=mx.int32)], axis=-1)
        padded_mask = mx.concatenate([mx.ones((2, 1, num_codebooks)), mx.zeros((2, 1, 1))], axis=-1)
        curr_token = model.generate_frame(
            tokens=padded, tokens_mask=padded_mask, input_pos=pos[:, -1:] + i + 1,
            temperature=1.0, topk=50, cfg_scale=1.5
        )
        mx.eval(curr_token)

    # Reset caches
    model.setup_caches(2)

    # Benchmark generation
    print(f"Benchmarking generation ({num_frames} frames)...")
    frames = []
    gen_start = time.perf_counter()

    curr_token = model.generate_frame(
        tokens=tokens, tokens_mask=mask, input_pos=pos,
        temperature=1.0, topk=50, cfg_scale=1.5,
        continuous_segments=muq_embed, starts=[1, 1]
    )
    mx.eval(curr_token)
    frames.append(curr_token[0:1])

    for i in range(num_frames - 1):
        padded = mx.concatenate([curr_token[:, None, :], mx.zeros((2, 1, 1), dtype=mx.int32)], axis=-1)
        padded_mask = mx.concatenate([mx.ones((2, 1, num_codebooks)), mx.zeros((2, 1, 1))], axis=-1)
        curr_token = model.generate_frame(
            tokens=padded, tokens_mask=padded_mask, input_pos=pos[:, -1:] + i + 1,
            temperature=1.0, topk=50, cfg_scale=1.5
        )
        mx.eval(curr_token)
        frames.append(curr_token[0:1])

    gen_time = time.perf_counter() - gen_start

    # Decode
    print("Benchmarking decode...")
    frames_arr = mx.concatenate(frames, axis=0)[None, :, :]
    mx.eval(frames_arr)
    decode_start = time.perf_counter()
    audio = codec.detokenize(frames_arr, duration=num_frames / 12.5)
    mx.eval(audio)
    decode_time = time.perf_counter() - decode_start

    return {
        "load_time": load_time,
        "gen_time": gen_time,
        "gen_fps": num_frames / gen_time,
        "decode_time": decode_time,
        "num_frames": num_frames,
        "audio_duration": num_frames / 12.5,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark PyTorch vs MLX HeartMuLa")
    parser.add_argument("--frames", type=int, default=50, help="Number of frames to generate")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup frames")
    parser.add_argument("--pytorch-ckpt", default=os.path.expanduser("~/Developer/heartlib/ckpt"),
                        help="PyTorch checkpoint path")
    parser.add_argument("--mlx-ckpt", default=os.path.expanduser("~/Developer/heartlib-mlx/ckpt-mlx"),
                        help="MLX checkpoint path")
    parser.add_argument("--mlx-only", action="store_true", help="Only benchmark MLX")
    parser.add_argument("--pytorch-only", action="store_true", help="Only benchmark PyTorch")
    args = parser.parse_args()

    print("=" * 70)
    print("HEARTMULA BENCHMARK")
    print("=" * 70)
    print(f"Frames: {args.frames} (~{args.frames / 12.5:.1f}s of audio)")
    print(f"Warmup: {args.warmup} frames")
    print()

    results = {}

    if not args.mlx_only:
        print("-" * 70)
        print("PyTorch (MPS)")
        print("-" * 70)
        try:
            results["pytorch"] = benchmark_pytorch(args.pytorch_ckpt, args.frames, args.warmup)
        except Exception as e:
            print(f"PyTorch benchmark failed: {e}")
            results["pytorch"] = None
        print()

    if not args.pytorch_only:
        print("-" * 70)
        print("MLX")
        print("-" * 70)
        try:
            results["mlx"] = benchmark_mlx(args.mlx_ckpt, args.frames, args.warmup)
        except Exception as e:
            print(f"MLX benchmark failed: {e}")
            results["mlx"] = None
        print()

    # Summary
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)

    if results.get("pytorch"):
        pt = results["pytorch"]
        print(f"\nPyTorch (MPS):")
        print(f"  Load time:     {pt['load_time']:.2f}s")
        print(f"  Generation:    {pt['gen_time']:.2f}s ({pt['gen_fps']:.2f} frames/s)")
        print(f"  Decode:        {pt['decode_time']:.2f}s")
        print(f"  Total:         {pt['gen_time'] + pt['decode_time']:.2f}s")
        print(f"  Real-time:     {pt['audio_duration'] / (pt['gen_time'] + pt['decode_time']):.2f}x")

    if results.get("mlx"):
        mlx = results["mlx"]
        print(f"\nMLX:")
        print(f"  Load time:     {mlx['load_time']:.2f}s")
        print(f"  Generation:    {mlx['gen_time']:.2f}s ({mlx['gen_fps']:.2f} frames/s)")
        print(f"  Decode:        {mlx['decode_time']:.2f}s")
        print(f"  Total:         {mlx['gen_time'] + mlx['decode_time']:.2f}s")
        print(f"  Real-time:     {mlx['audio_duration'] / (mlx['gen_time'] + mlx['decode_time']):.2f}x")

    if results.get("pytorch") and results.get("mlx"):
        pt, mlx = results["pytorch"], results["mlx"]
        print(f"\nSpeedup (MLX vs PyTorch):")
        print(f"  Load:          {pt['load_time'] / mlx['load_time']:.2f}x faster")
        print(f"  Generation:    {pt['gen_time'] / mlx['gen_time']:.2f}x faster")
        print(f"  Decode:        {pt['decode_time'] / mlx['decode_time']:.2f}x {'faster' if mlx['decode_time'] < pt['decode_time'] else 'slower'}")
        total_pt = pt['gen_time'] + pt['decode_time']
        total_mlx = mlx['gen_time'] + mlx['decode_time']
        print(f"  Total:         {total_pt / total_mlx:.2f}x faster")

    print()


if __name__ == "__main__":
    main()
