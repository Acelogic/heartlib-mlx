"""HeartTranscriptor - Whisper-based Lyrics Recognition Model."""

from typing import Optional, Union, List
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from heartlib_mlx.hearttranscriptor.configuration import HeartTranscriptorConfig
from heartlib_mlx.nn.transformer import RMSNorm, LlamaTransformerBlock


class WhisperEncoder(nn.Module):
    """Whisper-style audio encoder.

    Processes mel spectrograms into contextualized representations.

    Args:
        d_model: Model dimension.
        n_heads: Number of attention heads.
        n_layers: Number of transformer layers.
        n_mels: Number of mel filterbanks.
        max_len: Maximum sequence length.
    """

    def __init__(
        self,
        d_model: int = 1280,
        n_heads: int = 20,
        n_layers: int = 32,
        n_mels: int = 128,
        max_len: int = 1500,
    ):
        super().__init__()

        self.d_model = d_model

        # Convolutional frontend
        self.conv1 = nn.Conv1d(
            in_channels=n_mels,
            out_channels=d_model,
            kernel_size=3,
            padding=1,
        )
        self.conv2 = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=3,
            stride=2,
            padding=1,
        )

        # Positional embedding
        self.pos_embedding = nn.Embedding(max_len, d_model)

        # Transformer layers
        self.layers = [
            LlamaTransformerBlock(
                dim=d_model,
                n_heads=n_heads,
                max_seq_len=max_len,
            )
            for _ in range(n_layers)
        ]

        # Layer norm
        self.ln = RMSNorm(d_model)

    def __call__(self, mel: mx.array) -> mx.array:
        """Encode mel spectrogram.

        Args:
            mel: Mel spectrogram of shape (batch, n_mels, time).

        Returns:
            Encoded features of shape (batch, time // 2, d_model).
        """
        # Transpose for conv: (batch, time, n_mels) -> (batch, n_mels, time)
        if mel.ndim == 2:
            mel = mel[None, :]
        if mel.shape[1] != 128:  # If time is first dim
            mel = mel.transpose(0, 2, 1)

        # Convolutional frontend
        x = nn.gelu(self.conv1(mel.transpose(0, 2, 1)))
        x = nn.gelu(self.conv2(x))

        # Add positional embedding
        seq_len = x.shape[1]
        positions = mx.arange(seq_len)
        x = x + self.pos_embedding(positions)

        # Transformer layers
        for layer in self.layers:
            x, _ = layer(x)

        x = self.ln(x)
        return x


class WhisperDecoder(nn.Module):
    """Whisper-style text decoder.

    Generates text tokens autoregressively from encoder output.

    Args:
        vocab_size: Size of the vocabulary.
        d_model: Model dimension.
        n_heads: Number of attention heads.
        n_layers: Number of transformer layers.
        max_len: Maximum sequence length.
    """

    def __init__(
        self,
        vocab_size: int = 51866,
        d_model: int = 1280,
        n_heads: int = 20,
        n_layers: int = 32,
        max_len: int = 448,
    ):
        super().__init__()

        self.d_model = d_model

        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # Positional embedding
        self.pos_embedding = nn.Embedding(max_len, d_model)

        # Transformer layers with cross-attention
        self.layers = [
            WhisperDecoderBlock(
                dim=d_model,
                n_heads=n_heads,
                max_seq_len=max_len,
            )
            for _ in range(n_layers)
        ]

        # Layer norm
        self.ln = RMSNorm(d_model)

    def __call__(
        self,
        token_ids: mx.array,
        encoder_output: mx.array,
        cache: Optional[List] = None,
    ) -> mx.array:
        """Decode tokens.

        Args:
            token_ids: Token IDs of shape (batch, seq_len).
            encoder_output: Encoder output of shape (batch, enc_len, d_model).
            cache: Optional KV cache.

        Returns:
            Hidden states of shape (batch, seq_len, d_model).
        """
        seq_len = token_ids.shape[1]
        offset = 0 if cache is None else cache[0][0].shape[1] if cache[0] is not None else 0

        # Token and positional embeddings
        x = self.token_embedding(token_ids)
        positions = mx.arange(offset, offset + seq_len)
        x = x + self.pos_embedding(positions)

        # Create causal mask
        mask = mx.triu(mx.full((seq_len, seq_len + offset), float("-inf")), k=1 + offset)
        mask = mask[None, None, :, :]

        # Transformer layers
        new_cache = []
        for i, layer in enumerate(self.layers):
            layer_cache = cache[i] if cache is not None else None
            x, kv = layer(x, encoder_output, mask=mask, cache=layer_cache)
            new_cache.append(kv)

        x = self.ln(x)
        return x, new_cache


class WhisperDecoderBlock(nn.Module):
    """Whisper decoder block with self-attention and cross-attention.

    Args:
        dim: Model dimension.
        n_heads: Number of attention heads.
        max_seq_len: Maximum sequence length.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        max_seq_len: int = 448,
    ):
        super().__init__()

        # Self-attention
        self.self_attn = LlamaTransformerBlock(
            dim=dim,
            n_heads=n_heads,
            max_seq_len=max_seq_len,
        )

        # Cross-attention
        self.cross_attn_norm = RMSNorm(dim)
        self.cross_attn_q = nn.Linear(dim, dim)
        self.cross_attn_k = nn.Linear(dim, dim)
        self.cross_attn_v = nn.Linear(dim, dim)
        self.cross_attn_out = nn.Linear(dim, dim)

        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5

    def __call__(
        self,
        x: mx.array,
        encoder_output: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[tuple] = None,
    ):
        """Forward pass.

        Args:
            x: Decoder input of shape (batch, seq_len, dim).
            encoder_output: Encoder output.
            mask: Causal mask.
            cache: Optional KV cache.

        Returns:
            Output and new cache.
        """
        # Self-attention
        x, self_kv = self.self_attn(x, mask=mask, cache=cache)

        # Cross-attention
        batch_size, seq_len, dim = x.shape
        enc_len = encoder_output.shape[1]

        h = self.cross_attn_norm(x)

        q = self.cross_attn_q(h).reshape(batch_size, seq_len, self.n_heads, self.head_dim)
        k = self.cross_attn_k(encoder_output).reshape(batch_size, enc_len, self.n_heads, self.head_dim)
        v = self.cross_attn_v(encoder_output).reshape(batch_size, enc_len, self.n_heads, self.head_dim)

        # Attention
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        attn = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        attn = mx.softmax(attn, axis=-1)
        out = attn @ v

        out = out.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, dim)
        out = self.cross_attn_out(out)

        x = x + out

        return x, self_kv


class HeartTranscriptor(nn.Module):
    """HeartTranscriptor: Whisper-based Lyrics Recognition Model.

    A Whisper model fine-tuned for transcribing lyrics from music audio.

    Args:
        config: HeartTranscriptorConfig with model hyperparameters.
    """

    def __init__(self, config: HeartTranscriptorConfig):
        super().__init__()
        self.config = config

        # Encoder
        self.encoder = WhisperEncoder(
            d_model=config.d_model,
            n_heads=config.encoder_attention_heads,
            n_layers=config.encoder_layers,
            n_mels=config.n_mels,
            max_len=config.max_source_positions,
        )

        # Decoder
        self.decoder = WhisperDecoder(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            n_heads=config.decoder_attention_heads,
            n_layers=config.decoder_layers,
            max_len=config.max_target_positions,
        )

        # Output projection
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def encode(self, mel: mx.array) -> mx.array:
        """Encode mel spectrogram.

        Args:
            mel: Mel spectrogram.

        Returns:
            Encoder output.
        """
        return self.encoder(mel)

    def decode(
        self,
        token_ids: mx.array,
        encoder_output: mx.array,
        cache: Optional[List] = None,
    ) -> tuple:
        """Decode tokens.

        Args:
            token_ids: Token IDs.
            encoder_output: Encoder output.
            cache: Optional KV cache.

        Returns:
            Logits and new cache.
        """
        hidden, new_cache = self.decoder(token_ids, encoder_output, cache)
        logits = self.lm_head(hidden)
        return logits, new_cache

    def __call__(
        self,
        mel: mx.array,
        token_ids: Optional[mx.array] = None,
        labels: Optional[mx.array] = None,
    ) -> tuple:
        """Forward pass.

        Args:
            mel: Mel spectrogram.
            token_ids: Decoder input tokens.
            labels: Target labels for training.

        Returns:
            Logits and optional loss.
        """
        # Encode
        encoder_output = self.encode(mel)

        # Decode
        if token_ids is None:
            # Return encoder output for generation
            return encoder_output, None

        logits, _ = self.decode(token_ids, encoder_output)

        if labels is not None:
            # Compute loss
            shift_logits = logits[:, :-1, :].reshape(-1, self.config.vocab_size)
            shift_labels = labels[:, 1:].reshape(-1)
            loss = nn.losses.cross_entropy(shift_logits, shift_labels)
            return logits, loss

        return logits, None

    def generate(
        self,
        mel: mx.array,
        max_length: int = 448,
        temperature: float = 0.0,
        language_token: int = 50259,  # <|en|>
        task_token: int = 50358,  # <|transcribe|>
    ) -> mx.array:
        """Generate transcript from mel spectrogram.

        Args:
            mel: Mel spectrogram.
            max_length: Maximum output length.
            temperature: Sampling temperature (0 = greedy).
            language_token: Language token ID.
            task_token: Task token ID.

        Returns:
            Generated token IDs.
        """
        batch_size = mel.shape[0] if mel.ndim > 2 else 1

        # Encode
        encoder_output = self.encode(mel)

        # Initialize with special tokens
        # <|startoftranscript|> = 50258
        start_token = 50258
        tokens = mx.array([[start_token, language_token, task_token]] * batch_size)

        cache = None
        for _ in range(max_length - 3):
            # Get logits for last token
            logits, cache = self.decode(tokens[:, -1:], encoder_output, cache)
            logits = logits[:, -1, :]

            # Sample
            if temperature == 0:
                next_token = mx.argmax(logits, axis=-1)
            else:
                probs = mx.softmax(logits / temperature, axis=-1)
                next_token = mx.random.categorical(mx.log(probs + 1e-10))

            # Append
            tokens = mx.concatenate([tokens, next_token[:, None]], axis=1)

            # Check for end token (50257 = <|endoftext|>)
            if mx.all(next_token == 50257):
                break

        return tokens

    @classmethod
    def from_pretrained(
        cls,
        path: Union[str, Path],
        dtype: mx.Dtype = mx.bfloat16,
    ) -> "HeartTranscriptor":
        """Load a pretrained HeartTranscriptor model.

        Args:
            path: Path to the model directory.
            dtype: Data type for model weights.

        Returns:
            HeartTranscriptor instance with loaded weights.
        """
        from safetensors import safe_open

        path = Path(path)

        # Load config
        config = HeartTranscriptorConfig.from_pretrained(path)

        # Create model
        model = cls(config)

        # Load weights
        weights_path = path / "model.safetensors"
        if weights_path.exists():
            with safe_open(str(weights_path), framework="mlx") as f:
                weights = {k: f.get_tensor(k) for k in f.keys()}

            weights = {k: v.astype(dtype) for k, v in weights.items()}
            model.load_weights(list(weights.items()))

        return model

    def save_pretrained(self, path: Union[str, Path]) -> None:
        """Save the model to a directory.

        Args:
            path: Path to save the model.
        """
        from safetensors.numpy import save_file
        import numpy as np

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        self.config.save_pretrained(path)

        weights = dict(self.parameters())
        np_weights = {k: np.array(v) for k, v in weights.items()}
        save_file(np_weights, str(path / "model.safetensors"))
