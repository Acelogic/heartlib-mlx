#!/usr/bin/env python3
"""Download HeartMuLa pretrained weights from HuggingFace.

Usage:
    python scripts/download_weights.py --all
    python scripts/download_weights.py --model heartcodec heartmula
"""

import argparse
import subprocess
from pathlib import Path


MODELS = {
    "heartcodec": "HeartMuLa/HeartCodec-oss",
    "heartmula": "HeartMuLa/HeartMuLa-oss-3B",
    "heartclap": "HeartMuLa/HeartCLAP-oss",
    "hearttranscriptor": "HeartMuLa/HeartTranscriptor-oss",
}


def download_model(model_name: str, output_dir: Path) -> None:
    """Download a single model from HuggingFace.

    Args:
        model_name: Name of the model to download.
        output_dir: Base output directory.
    """
    if model_name not in MODELS:
        print(f"Unknown model: {model_name}")
        print(f"Available models: {list(MODELS.keys())}")
        return

    repo_id = MODELS[model_name]
    local_dir = output_dir / model_name.replace("-", "_")

    print(f"Downloading {model_name} from {repo_id}...")

    try:
        subprocess.run(
            [
                "huggingface-cli",
                "download",
                repo_id,
                "--local-dir",
                str(local_dir),
            ],
            check=True,
        )
        print(f"Downloaded to {local_dir}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to download {model_name}: {e}")
    except FileNotFoundError:
        print("huggingface-cli not found. Install with: pip install huggingface-hub")


def main():
    parser = argparse.ArgumentParser(description="Download HeartMuLa weights")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./ckpt",
        help="Output directory for weights",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=None,
        help="Models to download (heartcodec, heartmula, heartclap, hearttranscriptor)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all models",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.all:
        models = list(MODELS.keys())
    elif args.models:
        models = args.models
    else:
        print("Specify --models or --all")
        return

    for model in models:
        download_model(model, output_dir)

    print("\nDownload complete!")
    print(f"Next step: Run weight conversion:")
    print(f"  python -m heartlib_mlx.utils.convert --src {output_dir} --dst ./ckpt-mlx")


if __name__ == "__main__":
    main()
