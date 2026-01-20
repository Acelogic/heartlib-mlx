# HeartLib MLX

Apple MLX port of [HeartMuLa](https://github.com/HeartMuLa/heartlib) - a family of open-source music foundation models.

## Overview

HeartLib MLX provides efficient inference on Apple Silicon for all four HeartMuLa components:

- **HeartMuLa**: 3B parameter music language model for text-to-music generation
- **HeartCodec**: 12.5Hz neural audio codec with flow matching decoder
- **HeartCLAP**: Audio-text alignment model for music tagging and retrieval
- **HeartTranscriptor**: Whisper-based lyrics recognition model

## Installation

```bash
pip install -e .
```

### Requirements

- Python >= 3.10
- Apple Silicon Mac (M1/M2/M3/M4)
- MLX >= 0.22.0
- 32GB+ unified memory recommended for full bfloat16 inference

## Quick Start

### Download Weights

```bash
python scripts/download_weights.py --all
```

### Convert Weights

```bash
python -m heartlib_mlx.utils.convert --src ./ckpt --dst ./ckpt-mlx
```

### Generate Music

```python
from heartlib_mlx import HeartMuLaGenPipeline

# Load the pipeline
pipeline = HeartMuLaGenPipeline.from_pretrained("./ckpt-mlx")

# Generate music from text
audio = pipeline(
    lyrics="[Verse]\nHello world, this is a test song\n[Chorus]\nMusic makes the heart sing along",
    tags="pop, acoustic, female vocal",
    duration=30.0,
    cfg_scale=1.5,
)

# Save the audio
pipeline.save_audio(audio, "output.mp3")
```

### Tag Audio

```python
from heartlib_mlx import HeartCLAPPipeline

# Load the pipeline
pipeline = HeartCLAPPipeline.from_pretrained("./ckpt-mlx/heartclap")

# Get tags for audio
tags = pipeline.tag_audio("input.mp3", top_k=5)
print(tags)  # [("pop", 0.85), ("acoustic", 0.72), ...]
```

### Transcribe Lyrics

```python
from heartlib_mlx import HeartTranscriptorPipeline

# Load the pipeline
pipeline = HeartTranscriptorPipeline.from_pretrained("./ckpt-mlx/hearttranscriptor")

# Transcribe lyrics
lyrics = pipeline.transcribe("input.mp3")
print(lyrics)
```

## Examples

See the `examples/` directory for complete examples:

- `generate.py`: Music generation with various parameters
- `tag_audio.py`: Audio tagging and retrieval
- `transcribe.py`: Lyrics transcription

## Architecture

### HeartCodec

Neural audio codec operating at 12.5Hz with:
- Convolutional encoder/decoder (ScalarModel)
- Residual Vector Quantization (8 codebooks × 8192 entries)
- Flow matching decoder for high-fidelity synthesis

### HeartMuLa

Hierarchical music language model:
- Backbone: LLaMA-3B (28 layers, 3072 dim)
- Decoder: LLaMA-300M (3 layers) for multi-codebook prediction
- Classifier-free guidance for controllable generation

### HeartCLAP

Audio-text alignment model:
- Audio encoder: MuQ-MuLan based transformer
- Text encoder: BERT-style transformer
- Contrastive learning for shared embedding space

### HeartTranscriptor

Whisper-based lyrics recognition:
- Encoder: Audio feature extraction
- Decoder: Autoregressive text generation
- Fine-tuned for music lyrics

## Project Structure

```
heartlib-mlx/
├── src/heartlib_mlx/
│   ├── heartcodec/        # Neural audio codec
│   ├── heartmula/         # Music language model
│   ├── heartclap/         # Audio-text alignment
│   ├── hearttranscriptor/ # Lyrics recognition
│   ├── nn/                # Custom MLX layers
│   ├── ode/               # Flow matching ODE solvers
│   ├── pipelines/         # High-level APIs
│   └── utils/             # Utilities
├── tests/                 # Unit tests
├── scripts/               # Download and conversion scripts
└── examples/              # Usage examples
```

## Performance

### Memory Usage (bfloat16)

| Model | Memory | Notes |
|-------|--------|-------|
| HeartMuLa-3B | ~8GB | Backbone + decoder |
| HeartCodec | ~2GB | Flow matching |
| HeartCLAP | ~1GB | Audio + text encoders |
| HeartTranscriptor | ~3GB | Whisper-large |

### Benchmarks: MLX vs PyTorch MPS

Benchmarks run on Apple M2 Max with 32GB unified memory.

#### Model Loading

| Framework | HeartMuLa Load Time | Speedup |
|-----------|---------------------|---------|
| MLX | 1.41s | **2.6x faster** |
| PyTorch MPS | 3.62s | baseline |

#### Audio Generation (HeartMuLa 3B + HeartCodec)

Generation benchmark: 1500 frames (120 seconds of audio)

| Framework | Total Time | Frame Rate | Real-time Factor |
|-----------|------------|------------|------------------|
| PyTorch MPS | 575.5s | 2.6 frames/s | 0.21x |

#### HeartCodec Detokenize (10 ODE steps)

Converting codes to 5 seconds of audio:

| Framework | Time | Throughput |
|-----------|------|------------|
| MLX | 19.75s | 0.25x real-time |
| PyTorch MPS | 7.5s | 0.67x real-time |

> **Note**: MLX detokenize is currently slower than PyTorch MPS. This is an area for optimization, particularly in the flow matching decoder and ODE solver. Contributions welcome!

## License

Apache 2.0

## Citation

```bibtex
@article{heartmula2025,
  title={HeartMuLa: A Family of Open Sourced Music Foundation Models},
  author={HeartMuLa Team},
  journal={arXiv:2601.10547},
  year={2025}
}
```

## Acknowledgments

- [HeartMuLa Team](https://github.com/HeartMuLa) for the original PyTorch implementation
- [Apple MLX Team](https://github.com/ml-explore/mlx) for the MLX framework
