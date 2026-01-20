"""Generate music with both PyTorch and MLX HeartMuLa and compare."""

import numpy as np
import torch
import mlx.core as mx
import sys
import soundfile as sf
import time
from tqdm import tqdm
from tokenizers import Tokenizer
from dataclasses import dataclass

sys.path.insert(0, '/Users/mcruz/Developer/heartlib/src')
sys.path.insert(0, '/Users/mcruz/Developer/heartlib-mlx/src')

# Config for generation
@dataclass
class GenConfig:
    text_bos_id: int = 128000
    text_eos_id: int = 128001
    audio_eos_id: int = 8193
    empty_id: int = 0


def main():
    print("=" * 70)
    print("MUSIC GENERATION COMPARISON")
    print("=" * 70)

    # Input parameters
    tags = "<tag>electronic, ambient, chill, atmospheric</tag>"
    lyrics = "[intro]\n\n[verse]\nFloating through the digital sky\nWhere the stars never die\n\n[outro]"
    max_audio_frames = 125  # ~10 seconds (125 frames * 80ms = 10s)

    print(f"\nTags: {tags}")
    print(f"Lyrics: {lyrics[:50]}...")
    print(f"Max frames: {max_audio_frames}")

    config = GenConfig()
    ckpt_path = '/Users/mcruz/Developer/heartlib/ckpt'
    tokenizer = Tokenizer.from_file(f'{ckpt_path}/tokenizer.json')

    # Tokenize inputs
    tags_ids = tokenizer.encode(tags.lower()).ids
    if tags_ids[0] != config.text_bos_id:
        tags_ids = [config.text_bos_id] + tags_ids
    if tags_ids[-1] != config.text_eos_id:
        tags_ids = tags_ids + [config.text_eos_id]

    lyrics_ids = tokenizer.encode(lyrics.lower()).ids
    if lyrics_ids[0] != config.text_bos_id:
        lyrics_ids = [config.text_bos_id] + lyrics_ids
    if lyrics_ids[-1] != config.text_eos_id:
        lyrics_ids = lyrics_ids + [config.text_eos_id]

    muq_idx = len(tags_ids)
    prompt_len = len(tags_ids) + 1 + len(lyrics_ids)
    num_codebooks = 8
    parallel_number = num_codebooks + 1  # audio codebooks + text

    # Build prompt tokens: (prompt_len, parallel_number)
    # Format: [audio_codebook_0, ..., audio_codebook_7, text_token]
    prompt_tokens_np = np.zeros((prompt_len, parallel_number), dtype=np.int64)
    prompt_tokens_np[:len(tags_ids), -1] = tags_ids
    prompt_tokens_np[len(tags_ids) + 1:, -1] = lyrics_ids

    # Build mask: text tokens are active
    prompt_mask_np = np.zeros((prompt_len, parallel_number), dtype=bool)
    prompt_mask_np[:, -1] = True

    print(f"\nPrompt length: {prompt_len} tokens")

    # ========================================
    # PyTorch Generation
    # ========================================
    print("\n" + "=" * 60)
    print("PyTorch HeartMuLa Generation")
    print("=" * 60)

    from heartlib.heartmula.modeling_heartmula import HeartMuLa as PTHeartMuLa
    from heartlib.heartcodec.modeling_heartcodec import HeartCodec as PTHeartCodec

    print("Loading PyTorch models...")
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    dtype = torch.float32  # MPS has limited bfloat16 support

    pt_codec = PTHeartCodec.from_pretrained(f'{ckpt_path}/HeartCodec-oss', device_map=device)
    pt_model = PTHeartMuLa.from_pretrained(f'{ckpt_path}/HeartMuLa-oss-3B', dtype=dtype)
    pt_model = pt_model.to(device)
    pt_model.eval()

    # Prepare inputs
    cfg_scale = 1.5
    temperature = 1.0
    topk = 50

    # For CFG, duplicate batch
    prompt_tokens_pt = torch.tensor(prompt_tokens_np).unsqueeze(0)  # (1, seq, parallel)
    prompt_tokens_pt = torch.cat([prompt_tokens_pt, prompt_tokens_pt], dim=0).to(device)  # (2, seq, parallel)

    prompt_mask_pt = torch.tensor(prompt_mask_np).unsqueeze(0)
    prompt_mask_pt = torch.cat([prompt_mask_pt, prompt_mask_pt], dim=0).to(device)

    muq_dim = pt_model.config.muq_dim
    muq_embed_pt = torch.zeros((2, muq_dim), dtype=dtype, device=device)
    prompt_pos_pt = torch.arange(prompt_len, device=device).unsqueeze(0).repeat(2, 1)

    print("Generating with PyTorch...")
    start_time = time.time()

    pt_frames = []
    pt_model.setup_caches(2)

    with torch.no_grad():
        # First frame with prompt
        curr_token = pt_model.generate_frame(
            tokens=prompt_tokens_pt,
            tokens_mask=prompt_mask_pt,
            input_pos=prompt_pos_pt,
            temperature=temperature,
            topk=topk,
            cfg_scale=cfg_scale,
            continuous_segments=muq_embed_pt,
            starts=[muq_idx, muq_idx],
        )
        pt_frames.append(curr_token[0:1, :])  # Take conditional sample

        # Generate remaining frames
        for i in tqdm(range(max_audio_frames), desc="PyTorch frames"):
            # Pad audio token
            padded = torch.ones((2, 1, parallel_number), device=device, dtype=torch.long) * config.empty_id
            padded[:, 0, :-1] = curr_token
            padded_mask = torch.ones_like(padded, dtype=torch.bool)
            padded_mask[..., -1] = False

            curr_token = pt_model.generate_frame(
                tokens=padded,
                tokens_mask=padded_mask,
                input_pos=prompt_pos_pt[:, -1:] + i + 1,
                temperature=temperature,
                topk=topk,
                cfg_scale=cfg_scale,
            )
            if torch.any(curr_token[0, :] >= config.audio_eos_id):
                break
            pt_frames.append(curr_token[0:1, :])

    pt_time = time.time() - start_time

    # Decode audio
    pt_frames_tensor = torch.stack(pt_frames).permute(1, 2, 0).squeeze(0)  # (num_codebooks, num_frames)
    print(f"Generated {pt_frames_tensor.shape[1]} frames")
    pt_audio = pt_codec.detokenize(pt_frames_tensor)
    # PyTorch returns stereo (2, samples), convert to mono by averaging channels
    pt_audio_np = pt_audio.cpu().numpy().mean(axis=0)  # (samples,)

    print(f"Generated in {pt_time:.2f}s")
    print(f"Audio shape: {pt_audio_np.shape}")
    print(f"Range: [{pt_audio_np.min():.4f}, {pt_audio_np.max():.4f}]")
    print(f"Std: {pt_audio_np.std():.4f}")

    sf.write('/Users/mcruz/Developer/heartlib-mlx/tests/pytorch_music.wav', pt_audio_np, 48000)
    print("Saved: tests/pytorch_music.wav")

    # Clean up PyTorch models
    del pt_model, pt_codec
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ========================================
    # MLX Generation
    # ========================================
    print("\n" + "=" * 60)
    print("MLX HeartMuLa Generation")
    print("=" * 60)

    from heartlib_mlx.heartmula import HeartMuLa as MLXHeartMuLa
    from heartlib_mlx.heartcodec import HeartCodec as MLXHeartCodec

    print("Loading MLX models...")
    mlx_codec = MLXHeartCodec.from_pretrained('/Users/mcruz/Developer/heartlib-mlx/ckpt-mlx/heartcodec')
    mlx_model = MLXHeartMuLa.from_pretrained('/Users/mcruz/Developer/heartlib-mlx/ckpt-mlx/heartmula')

    # Prepare inputs
    prompt_tokens_mlx = mx.array(prompt_tokens_np)[None, :, :]  # (1, seq, parallel)
    prompt_tokens_mlx = mx.concatenate([prompt_tokens_mlx, prompt_tokens_mlx], axis=0)  # (2, seq, parallel)

    prompt_mask_mlx = mx.array(prompt_mask_np.astype(np.float32))[None, :, :]
    prompt_mask_mlx = mx.concatenate([prompt_mask_mlx, prompt_mask_mlx], axis=0)

    muq_embed_mlx = mx.zeros((2, muq_dim))
    prompt_pos_mlx = mx.broadcast_to(mx.arange(prompt_len)[None, :], (2, prompt_len))

    print("Generating with MLX...")
    start_time = time.time()

    mlx_frames = []
    mlx_model.setup_caches(2)

    # First frame with prompt
    curr_token = mlx_model.generate_frame(
        tokens=prompt_tokens_mlx,
        tokens_mask=prompt_mask_mlx,
        input_pos=prompt_pos_mlx,
        temperature=temperature,
        topk=topk,
        cfg_scale=cfg_scale,
        continuous_segments=muq_embed_mlx,
        starts=[muq_idx, muq_idx],
    )
    mlx_frames.append(curr_token[0:1, :])

    # Generate remaining frames
    for i in tqdm(range(max_audio_frames), desc="MLX frames"):
        # Pad audio token
        padded = mx.ones((2, 1, parallel_number), dtype=mx.int32) * config.empty_id
        # Update audio codebooks
        padded = mx.concatenate([
            curr_token[:, None, :],
            mx.zeros((2, 1, 1), dtype=mx.int32)  # text position
        ], axis=-1)
        padded_mask = mx.ones((2, 1, parallel_number))
        padded_mask = mx.concatenate([
            mx.ones((2, 1, num_codebooks)),
            mx.zeros((2, 1, 1))
        ], axis=-1)

        curr_pos = prompt_pos_mlx[:, -1:] + i + 1

        curr_token = mlx_model.generate_frame(
            tokens=padded.astype(mx.int32),
            tokens_mask=padded_mask,
            input_pos=curr_pos,
            temperature=temperature,
            topk=topk,
            cfg_scale=cfg_scale,
        )
        mx.eval(curr_token)

        if mx.any(curr_token[0, :] >= config.audio_eos_id):
            break
        mlx_frames.append(curr_token[0:1, :])

    mlx_time = time.time() - start_time

    # Stack frames: list of (1, num_codebooks) -> (batch, num_frames, num_codebooks)
    mlx_frames_arr = mx.concatenate(mlx_frames, axis=0)  # (num_frames, num_codebooks)
    mlx_frames_arr = mlx_frames_arr[None, :, :]  # (1, num_frames, num_codebooks)
    mx.eval(mlx_frames_arr)
    print(f"Generated {mlx_frames_arr.shape[1]} frames")

    # Decode audio (expects batch, frames, num_quantizers)
    # Duration should match generated frames: frames / frame_rate
    gen_duration = mlx_frames_arr.shape[1] / 12.5
    mlx_audio = mlx_codec.detokenize(mlx_frames_arr, duration=gen_duration)
    mx.eval(mlx_audio)
    mlx_audio_np = np.array(mlx_audio.astype(mx.float32)).flatten()

    print(f"Generated in {mlx_time:.2f}s")
    print(f"Audio shape: {mlx_audio_np.shape}")
    print(f"Range: [{mlx_audio_np.min():.4f}, {mlx_audio_np.max():.4f}]")
    print(f"Std: {mlx_audio_np.std():.4f}")

    sf.write('/Users/mcruz/Developer/heartlib-mlx/tests/mlx_music.wav', mlx_audio_np, 48000)
    print("Saved: tests/mlx_music.wav")

    # ========================================
    # Comparison
    # ========================================
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)

    print(f"\nGeneration time:")
    print(f"  PyTorch: {pt_time:.2f}s")
    print(f"  MLX: {mlx_time:.2f}s")
    if mlx_time > 0:
        print(f"  Ratio: {pt_time/mlx_time:.2f}x")

    print(f"\nAudio statistics:")
    print(f"  PyTorch - Range: [{pt_audio_np.min():.4f}, {pt_audio_np.max():.4f}], Std: {pt_audio_np.std():.4f}")
    print(f"  MLX     - Range: [{mlx_audio_np.min():.4f}, {mlx_audio_np.max():.4f}], Std: {mlx_audio_np.std():.4f}")

    print(f"\nBoth outputs saved - listen to compare quality:")
    print(f"  - tests/pytorch_music.wav")
    print(f"  - tests/mlx_music.wav")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
