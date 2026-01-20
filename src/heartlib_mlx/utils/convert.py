"""Weight conversion utilities for PyTorch to MLX."""

from typing import Dict, Optional, List, Union, Callable
from pathlib import Path
import json
import re

import numpy as np
import mlx.core as mx


def convert_pytorch_to_mlx(
    src_path: Union[str, Path],
    dst_path: Union[str, Path],
    model_type: str = "auto",
    dtype: str = "bfloat16",
) -> None:
    """Convert PyTorch weights to MLX format.

    Args:
        src_path: Source directory with PyTorch weights.
        dst_path: Destination directory for MLX weights.
        model_type: Model type ("heartcodec", "heartmula", "heartclap",
            "hearttranscriptor", or "auto").
        dtype: Target data type.
    """
    src_path = Path(src_path)
    dst_path = Path(dst_path)
    dst_path.mkdir(parents=True, exist_ok=True)

    # Detect model type if auto
    if model_type == "auto":
        config_path = src_path / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
            model_type = config.get("model_type", "unknown")

    print(f"Converting {model_type} model from {src_path} to {dst_path}")

    # Get conversion function
    converters = {
        "heartcodec": convert_heartcodec_weights,
        "heartmula": convert_heartmula_weights,
        "heartclap": convert_heartclap_weights,
        "hearttranscriptor": convert_hearttranscriptor_weights,
    }

    converter = converters.get(model_type.lower())
    if converter is None:
        print(f"Unknown model type: {model_type}, using generic conversion")
        converter = convert_generic_weights

    # Load PyTorch weights
    weights = load_pytorch_weights(src_path)

    # Convert weights
    mlx_weights = converter(weights)

    # Convert dtype
    dtype_map = {
        "float32": mx.float32,
        "float16": mx.float16,
        "bfloat16": mx.bfloat16,
    }
    target_dtype = dtype_map.get(dtype, mx.bfloat16)
    mlx_weights = {k: mx.array(v).astype(target_dtype) for k, v in mlx_weights.items()}

    # Save as safetensors
    save_mlx_weights(mlx_weights, dst_path / "model.safetensors")

    # Copy config
    config_src = src_path / "config.json"
    if config_src.exists():
        import shutil
        shutil.copy(config_src, dst_path / "config.json")

    # Copy tokenizer if present
    tokenizer_src = src_path / "tokenizer.json"
    if tokenizer_src.exists():
        import shutil
        shutil.copy(tokenizer_src, dst_path / "tokenizer.json")

    print(f"Conversion complete. Saved {len(mlx_weights)} tensors.")


def load_pytorch_weights(path: Path) -> Dict[str, np.ndarray]:
    """Load PyTorch weights from various formats.

    Args:
        path: Path to weights directory.

    Returns:
        Dictionary of numpy arrays.
    """
    weights = {}

    # Try safetensors first
    safetensors_path = path / "model.safetensors"
    if safetensors_path.exists():
        from safetensors import safe_open
        with safe_open(str(safetensors_path), framework="numpy") as f:
            for key in f.keys():
                weights[key] = f.get_tensor(key)
        return weights

    # Try multiple safetensors files
    safetensors_files = list(path.glob("*.safetensors"))
    if safetensors_files:
        from safetensors import safe_open
        for sf_path in safetensors_files:
            with safe_open(str(sf_path), framework="numpy") as f:
                for key in f.keys():
                    weights[key] = f.get_tensor(key)
        return weights

    # Try PyTorch bin files
    bin_path = path / "pytorch_model.bin"
    if bin_path.exists():
        import torch
        state_dict = torch.load(str(bin_path), map_location="cpu")
        for key, value in state_dict.items():
            weights[key] = value.numpy()
        return weights

    # Try multiple bin files
    bin_files = list(path.glob("pytorch_model-*.bin"))
    if bin_files:
        import torch
        for bf_path in bin_files:
            state_dict = torch.load(str(bf_path), map_location="cpu")
            for key, value in state_dict.items():
                weights[key] = value.numpy()
        return weights

    raise FileNotFoundError(f"No weights found in {path}")


def save_mlx_weights(weights: Dict[str, mx.array], path: Path) -> None:
    """Save MLX weights to safetensors.

    Args:
        weights: Dictionary of MLX arrays.
        path: Output path.
    """
    # MLX's save_safetensors handles bfloat16 properly
    mx.save_safetensors(str(path), weights)


def convert_conv_weights(weight: np.ndarray) -> np.ndarray:
    """Convert PyTorch Conv1d weights to MLX format.

    PyTorch: (out_channels, in_channels, kernel_size)
    MLX: (out_channels, kernel_size, in_channels)

    Args:
        weight: PyTorch weight tensor.

    Returns:
        MLX weight tensor.
    """
    if weight.ndim == 3:
        # (out, in, k) -> (out, k, in)
        return weight.transpose(0, 2, 1)
    return weight


def convert_attention_weights(weights: Dict[str, np.ndarray], prefix: str) -> Dict[str, np.ndarray]:
    """Convert attention layer weights.

    Handles QKV fusion and other attention weight transformations.

    Args:
        weights: Full weight dictionary.
        prefix: Prefix for attention layer keys.

    Returns:
        Converted weights for this layer.
    """
    converted = {}

    # Check for fused QKV
    qkv_key = f"{prefix}.qkv_proj.weight"
    if qkv_key in weights:
        qkv = weights[qkv_key]
        # Split into Q, K, V
        dim = qkv.shape[0] // 3
        q, k, v = np.split(qkv, 3, axis=0)
        converted[f"{prefix}.q_proj.weight"] = q
        converted[f"{prefix}.k_proj.weight"] = k
        converted[f"{prefix}.v_proj.weight"] = v
    else:
        # Copy individual projections
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            key = f"{prefix}.{proj}.weight"
            if key in weights:
                converted[key] = weights[key]
            bias_key = f"{prefix}.{proj}.bias"
            if bias_key in weights:
                converted[bias_key] = weights[bias_key]

    return converted


def convert_generic_weights(weights: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Generic weight conversion.

    Handles common transformations:
    - Conv weight transpose
    - Weight normalization

    Args:
        weights: PyTorch weights.

    Returns:
        Converted weights.
    """
    converted = {}

    for key, value in weights.items():
        new_key = key

        # Convert conv weights
        if "conv" in key.lower() and "weight" in key and value.ndim == 3:
            value = convert_conv_weights(value)

        converted[new_key] = value

    return converted


def convert_heartcodec_weights(weights: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Convert HeartCodec weights from PyTorch to MLX format.

    Handles:
    1. Weight normalization: parametrizations.weight.original0/original1 -> weight_g/weight_v
    2. Channel ordering: (out, in, k) -> (out, k, in) for conv weights
    3. Layer naming: encoder.X -> encoder_in/encoder_blocks.X/encoder_out

    Args:
        weights: PyTorch weights.

    Returns:
        Converted MLX weights.
    """
    converted = {}

    # First pass: collect all weight normalization pairs and regular weights
    weight_norm_pairs = {}  # base_key -> {'g': tensor, 'v': tensor}
    regular_weights = {}

    for key, value in weights.items():
        if 'parametrizations.weight.original0' in key:
            # This is the magnitude (g) - shape (out, 1, 1)
            base_key = key.replace('.parametrizations.weight.original0', '')
            if base_key not in weight_norm_pairs:
                weight_norm_pairs[base_key] = {}
            weight_norm_pairs[base_key]['g'] = value
        elif 'parametrizations.weight.original1' in key:
            # This is the direction (v) - shape (out, in, k)
            base_key = key.replace('.parametrizations.weight.original1', '')
            if base_key not in weight_norm_pairs:
                weight_norm_pairs[base_key] = {}
            weight_norm_pairs[base_key]['v'] = value
        else:
            regular_weights[key] = value

    # Process weight normalization pairs
    for base_key, pair in weight_norm_pairs.items():
        if 'g' in pair and 'v' in pair:
            g = pair['g']  # (dim0, 1, 1) - magnitude
            v = pair['v']  # Direction tensor

            # Check if this is a transposed conv (upsample layer)
            is_transpose = 'up_conv' in base_key or 'upsample' in base_key

            if is_transpose:
                # PyTorch ConvTranspose1d weight: (in_channels, out_channels, kernel_size)
                # MLX WeightNormConvTranspose1d expects: (out_channels, kernel_size, in_channels)
                # PyTorch weight_norm applies g per in_channel
                # We compute full weight and convert to MLX format
                g_original = g  # (in, 1, 1)
                v_original = v  # (in, out, k)
                v_norm = np.sqrt(np.sum(v_original ** 2, axis=(1, 2), keepdims=True) + 1e-8)
                weight_full = g_original * v_original / v_norm  # (in, out, k)
                # Convert to MLX layout: (out, k, in)
                weight_mlx = weight_full.transpose(1, 2, 0)
                # Store the pre-computed weight directly (no weight_g/weight_v)
                new_key = _map_heartcodec_key(base_key)
                converted[f"{new_key}.weight"] = weight_mlx
                continue  # Skip the weight_g/weight_v assignment below
            else:
                # Regular Conv1d: PyTorch (out, in, k) -> MLX (out, k, in)
                g_converted = g.squeeze()
                # Ensure g_converted is at least 1D (not scalar)
                if g_converted.ndim == 0:
                    g_converted = g_converted.reshape(1)
                v_converted = v.transpose(0, 2, 1)

            # Map the key name
            new_key = _map_heartcodec_key(base_key)
            converted[f"{new_key}.weight_g"] = g_converted
            converted[f"{new_key}.weight_v"] = v_converted

    # Process regular weights
    for key, value in regular_weights.items():
        new_key = _map_heartcodec_key(key)

        # Handle conv weights (non-weight-normalized)
        if value.ndim == 3:
            if 'ffn_1.weight' in key or 'ffn_2.weight' in key:
                # These are 1D convs that need transposing
                if value.shape[2] > 1:  # kernel_size > 1
                    value = value.transpose(0, 2, 1)
            elif 'weight' in key and ('conv' in key.lower() or 'proj' in key.lower()):
                # Standard conv weight transpose
                value = value.transpose(0, 2, 1)

        # Handle activation weights (Snake/PReLU alpha)
        if 'activation' in key and 'weight' in key:
            # PyTorch PReLU: (1, channels, 1) or (channels,) -> MLX: (channels,)
            if value.ndim == 3 and value.shape[0] == 1 and value.shape[2] == 1:
                value = value[0, :, 0]  # Extract middle dimension
            elif value.ndim == 1:
                pass  # Already correct shape
            else:
                # Try to get a 1D array
                value = value.flatten()
                if value.shape[0] == 1:
                    # Single channel PReLU
                    pass

        converted[new_key] = value

    return converted


def _map_heartcodec_key(key: str) -> str:
    """Map PyTorch HeartCodec key names to MLX key names.

    PyTorch structure (with num_samples=2):
    Encoder:
    - encoder.0 = weight_norm Conv1d (encoder_in)
    - encoder.1 = PreProcessor (skip - not in MLX model)
    - encoder.2-6 = ResEncoderBlocks (encoder_blocks.0-4)
    - encoder.7 = weight_norm Conv1d (encoder_out)

    Decoder:
    - decoder.0 = weight_norm Conv1d (decoder_in)
    - decoder.1-5 = ResDecoderBlocks (decoder_blocks.0-4)
    - decoder.6 = PostProcessor (skip - not in MLX model)
    - decoder.7 = weight_norm Conv1d (decoder_out)

    ResEncoderBlock:
    - convs.X = residual_units.X
    - down_conv.layer = downsample
    - down_conv.activation = activation

    ResDecoderBlock:
    - up_conv.layer = upsample
    - convs.X = residual_units.X

    ResidualUnit:
    - activation1 = activation (PReLU used after conv1)
    - activation2 = activation2 (PReLU used after conv2, but MLX uses single activation)
    """
    new_key = key

    # === Scalar Model Mappings ===

    # Skip PreProcessor (encoder.1) and PostProcessor (decoder.6) - not in MLX model
    # We'll handle these by not mapping them

    # Map encoder blocks
    # encoder.2-6 -> encoder_blocks.0-4
    for old_idx in range(2, 7):
        new_idx = old_idx - 2
        new_key = new_key.replace(f'scalar_model.encoder.{old_idx}.', f'scalar_model.encoder_blocks.{new_idx}.')

    # Map encoder.0 (input conv) -> encoder_in
    new_key = new_key.replace('scalar_model.encoder.0.', 'scalar_model.encoder_in.')
    new_key = new_key.replace('scalar_model.encoder.0', 'scalar_model.encoder_in')

    # Map encoder.7 (output conv) -> encoder_out
    new_key = new_key.replace('scalar_model.encoder.7.', 'scalar_model.encoder_out.')
    new_key = new_key.replace('scalar_model.encoder.7', 'scalar_model.encoder_out')

    # Map decoder blocks
    # decoder.1-5 -> decoder_blocks.0-4
    for old_idx in range(1, 6):
        new_idx = old_idx - 1
        new_key = new_key.replace(f'scalar_model.decoder.{old_idx}.', f'scalar_model.decoder_blocks.{new_idx}.')

    # Map decoder.0 (input conv) -> decoder_in
    new_key = new_key.replace('scalar_model.decoder.0.', 'scalar_model.decoder_in.')
    new_key = new_key.replace('scalar_model.decoder.0', 'scalar_model.decoder_in')

    # Map decoder.7 (output conv) -> decoder_out
    new_key = new_key.replace('scalar_model.decoder.7.', 'scalar_model.decoder_out.')
    new_key = new_key.replace('scalar_model.decoder.7', 'scalar_model.decoder_out')

    # Map PostProcessor (decoder.6)
    # activation -> scalar_model.activation
    # conv -> scalar_model.post_conv
    new_key = new_key.replace('scalar_model.decoder.6.activation.', 'scalar_model.activation.')
    new_key = new_key.replace('scalar_model.decoder.6.activation', 'scalar_model.activation')
    new_key = new_key.replace('scalar_model.decoder.6.conv.', 'scalar_model.post_conv.')
    new_key = new_key.replace('scalar_model.decoder.6.conv', 'scalar_model.post_conv')

    # Map residual unit structure
    new_key = new_key.replace('.convs.', '.residual_units.')

    # Map downsample/upsample layers
    # Handle both with and without trailing dot
    new_key = new_key.replace('.down_conv.layer.', '.downsample.')
    new_key = new_key.replace('.down_conv.layer', '.downsample')  # No trailing dot
    new_key = new_key.replace('.down_conv.activation.', '.activation.')  # Block activation
    new_key = new_key.replace('.down_conv.activation', '.activation')  # No trailing dot
    new_key = new_key.replace('.up_conv.layer.', '.upsample.')
    new_key = new_key.replace('.up_conv.layer', '.upsample')  # No trailing dot
    # Note: up_conv has no activation in PyTorch (activation=None in UpsampleLayer)

    # Map activation names
    # PyTorch ResidualUnit has activation1 and activation2 (both PReLU with single param)
    # MLX ResidualUnit has activation and activation2
    new_key = new_key.replace('.activation1.', '.activation.')
    new_key = new_key.replace('.activation1', '.activation')  # No trailing dot

    # === Flow Matching ===
    # MLX architecture matches PyTorch structure, but some naming differences remain

    # Map MLP layer names (PyTorch: gate/up/down -> MLX: gate_proj/up_proj/down_proj)
    new_key = new_key.replace('.mlp.gate.', '.mlp.gate_proj.')
    new_key = new_key.replace('.mlp.up.', '.mlp.up_proj.')
    new_key = new_key.replace('.mlp.down.', '.mlp.down_proj.')

    # Map VQ module naming
    new_key = new_key.replace('flow_matching.vq.', 'flow_matching.vq_embed.')

    # Map VQ codebook: PyTorch uses _codebook, MLX uses codebook (no underscore)
    new_key = new_key.replace('._codebook.', '.codebook.')

    return new_key


def convert_heartmula_weights(weights: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Convert HeartMuLa weights from PyTorch to MLX format.

    Handles:
    1. Audio embeddings: keep as single (65576, 3072) table
    2. Audio head: (7, 3072, 8197) reshape to (57379, 3072) for nn.Linear
    3. Attention naming: output_proj -> o_proj
    4. MLP naming: w1/w2/w3 -> gate_proj/up_proj/down_proj
    5. Norm naming: sa_norm/mlp_norm with .scale -> attention_norm/mlp_norm with .weight
    6. Unconditional embedding: keep as nn.Embedding weight

    Args:
        weights: PyTorch weights.

    Returns:
        Converted MLX weights.
    """
    converted = {}

    for key, value in weights.items():
        # === Special handling for specific keys ===

        # 1. Keep audio_embeddings as single table (matches new model structure)
        if key == "audio_embeddings.weight":
            # Shape: (65576, 3072) = (8 * 8197, 3072) - keep as is
            converted["audio_embeddings.weight"] = value
            continue

        # 2. Convert audio_head from (7, 3072, 8197) to (57379, 3072)
        if key == "audio_head":
            # PyTorch: (num_codebooks-1, dim, vocab_size) = (7, 3072, 8197)
            # MLX nn.Linear: (out_features, in_features) = (57379, 3072)
            # First transpose each codebook's weights, then concatenate
            # (7, 3072, 8197) -> (7, 8197, 3072) -> (57379, 3072)
            transposed = value.transpose(0, 2, 1)  # (7, 8197, 3072)
            flattened = transposed.reshape(-1, value.shape[1])  # (57379, 3072)
            converted["audio_head.weight"] = flattened
            continue

        # 3. Keep unconditional_text_embedding as nn.Embedding weight
        if key == "unconditional_text_embedding.weight":
            # Shape: (1, 3072) - keep as embedding weight
            converted["unconditional_text_embedding.weight"] = value
            continue

        # === General key mapping ===
        new_key = _map_heartmula_key(key)
        converted[new_key] = value

    return converted


def _map_heartmula_key(key: str) -> str:
    """Map PyTorch HeartMuLa key names to MLX key names.

    PyTorch structure:
    - backbone.layers.X.attn.{q_proj, k_proj, v_proj, output_proj}.weight
    - backbone.layers.X.mlp.{w1, w2, w3}.weight
    - backbone.layers.X.{sa_norm, mlp_norm}.scale
    - backbone.norm.scale
    - decoder.layers.X... (same structure)

    MLX structure:
    - backbone.layers.X.attention.{q_proj, k_proj, v_proj, o_proj}.weight
    - backbone.layers.X.mlp.{gate_proj, up_proj, down_proj}.weight
    - backbone.layers.X.{attention_norm, mlp_norm}.weight
    - backbone.norm.weight
    """
    new_key = key

    # === Embedding mappings ===
    # text_embeddings stays the same (both PyTorch and MLX use text_embeddings)

    # === Attention mappings ===
    # output_proj -> o_proj
    new_key = new_key.replace(".attn.output_proj.", ".attention.o_proj.")
    new_key = new_key.replace(".attn.q_proj.", ".attention.q_proj.")
    new_key = new_key.replace(".attn.k_proj.", ".attention.k_proj.")
    new_key = new_key.replace(".attn.v_proj.", ".attention.v_proj.")

    # === MLP mappings ===
    # w1 -> gate_proj (gate in SwiGLU)
    # w3 -> up_proj (up in SwiGLU)
    # w2 -> down_proj (down in SwiGLU)
    new_key = new_key.replace(".mlp.w1.", ".mlp.gate_proj.")
    new_key = new_key.replace(".mlp.w3.", ".mlp.up_proj.")
    new_key = new_key.replace(".mlp.w2.", ".mlp.down_proj.")

    # === Norm mappings ===
    # sa_norm -> attention_norm
    # .scale -> .weight
    new_key = new_key.replace(".sa_norm.scale", ".attention_norm.weight")
    new_key = new_key.replace(".mlp_norm.scale", ".mlp_norm.weight")
    new_key = new_key.replace(".norm.scale", ".norm.weight")

    return new_key


def convert_heartclap_weights(weights: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Convert HeartCLAP weights.

    Args:
        weights: PyTorch weights.

    Returns:
        Converted MLX weights.
    """
    return convert_generic_weights(weights)


def convert_hearttranscriptor_weights(weights: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Convert HeartTranscriptor weights.

    Args:
        weights: PyTorch weights.

    Returns:
        Converted MLX weights.
    """
    converted = {}

    for key, value in weights.items():
        new_key = key

        # Handle Whisper-specific weight names
        # Map from HuggingFace Whisper to our implementation
        new_key = new_key.replace("model.encoder", "encoder")
        new_key = new_key.replace("model.decoder", "decoder")

        # Handle conv weights
        if "conv" in key.lower() and "weight" in key and value.ndim == 3:
            value = convert_conv_weights(value)

        converted[new_key] = value

    return converted


def main():
    """Command-line interface for weight conversion."""
    import argparse

    parser = argparse.ArgumentParser(description="Convert PyTorch weights to MLX")
    parser.add_argument("--src", type=str, required=True, help="Source directory")
    parser.add_argument("--dst", type=str, required=True, help="Destination directory")
    parser.add_argument("--model-type", type=str, default="auto",
                        choices=["auto", "heartcodec", "heartmula", "heartclap", "hearttranscriptor"],
                        help="Model type")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["float32", "float16", "bfloat16"],
                        help="Target data type")

    args = parser.parse_args()

    convert_pytorch_to_mlx(
        src_path=args.src,
        dst_path=args.dst,
        model_type=args.model_type,
        dtype=args.dtype,
    )


if __name__ == "__main__":
    main()
