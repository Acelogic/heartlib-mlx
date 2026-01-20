"""Sampling utilities for text and audio generation."""

from typing import Optional, Tuple

import mlx.core as mx


def sample_topk(
    logits: mx.array,
    temperature: float = 1.0,
    top_k: int = 50,
) -> mx.array:
    """Sample from logits using top-k sampling.

    Args:
        logits: Logits of shape (..., vocab_size).
        temperature: Sampling temperature. 0 = greedy.
        top_k: Number of top tokens to consider.

    Returns:
        Sampled token indices.
    """
    if temperature == 0:
        # Greedy decoding
        return mx.argmax(logits, axis=-1)

    # Apply temperature
    logits = logits / temperature

    # Get top-k
    vocab_size = logits.shape[-1]
    top_k = min(top_k, vocab_size)

    # Get top-k logits and indices using argsort
    sorted_indices = mx.argsort(logits, axis=-1)[..., ::-1]  # Descending order
    topk_indices = sorted_indices[..., :top_k]
    topk_logits = mx.take_along_axis(logits, topk_indices, axis=-1)

    # Softmax over top-k
    probs = mx.softmax(topk_logits, axis=-1)

    # Sample from categorical
    samples = mx.random.categorical(mx.log(probs + 1e-10))

    # Map back to original vocabulary
    # Handle arbitrary batch dimensions
    original_shape = samples.shape
    samples_flat = samples.reshape(-1)
    topk_indices_flat = topk_indices.reshape(-1, top_k)

    result = mx.take_along_axis(
        topk_indices_flat,
        samples_flat[:, None],
        axis=-1,
    )[:, 0]

    return result.reshape(original_shape)


def sample_topp(
    logits: mx.array,
    temperature: float = 1.0,
    top_p: float = 0.9,
) -> mx.array:
    """Sample from logits using nucleus (top-p) sampling.

    Args:
        logits: Logits of shape (..., vocab_size).
        temperature: Sampling temperature.
        top_p: Cumulative probability threshold.

    Returns:
        Sampled token indices.
    """
    if temperature == 0:
        return mx.argmax(logits, axis=-1)

    # Apply temperature
    logits = logits / temperature

    # Sort by probability
    probs = mx.softmax(logits, axis=-1)
    sorted_indices = mx.argsort(probs, axis=-1)[..., ::-1]
    sorted_probs = mx.take_along_axis(probs, sorted_indices, axis=-1)

    # Compute cumulative probabilities
    cumsum_probs = mx.cumsum(sorted_probs, axis=-1)

    # Find cutoff
    mask = cumsum_probs <= top_p
    # Always include at least one token
    first_element = mx.arange(mask.shape[-1]) == 0
    mask = mask | first_element

    # Zero out tokens beyond cutoff
    sorted_probs = mx.where(mask, sorted_probs, 0.0)

    # Renormalize
    sorted_probs = sorted_probs / mx.sum(sorted_probs, axis=-1, keepdims=True)

    # Sample
    samples = mx.random.categorical(mx.log(sorted_probs + 1e-10))

    # Map back to original indices
    original_shape = samples.shape
    samples_flat = samples.reshape(-1)
    sorted_indices_flat = sorted_indices.reshape(-1, logits.shape[-1])

    result = mx.take_along_axis(
        sorted_indices_flat,
        samples_flat[:, None],
        axis=-1,
    )[:, 0]

    return result.reshape(original_shape)


def sample_typical(
    logits: mx.array,
    temperature: float = 1.0,
    typical_p: float = 0.9,
) -> mx.array:
    """Sample from logits using typical sampling.

    Selects tokens with entropy close to the expected entropy.

    Args:
        logits: Logits of shape (..., vocab_size).
        temperature: Sampling temperature.
        typical_p: Probability mass to consider.

    Returns:
        Sampled token indices.
    """
    if temperature == 0:
        return mx.argmax(logits, axis=-1)

    # Apply temperature
    logits = logits / temperature

    # Compute probabilities
    probs = mx.softmax(logits, axis=-1)

    # Compute entropy
    log_probs = mx.log(probs + 1e-10)
    entropy = -mx.sum(probs * log_probs, axis=-1, keepdims=True)

    # Compute "typicality" as |log(p) - entropy|
    typicality = mx.abs(log_probs + entropy)

    # Sort by typicality (lower is more typical)
    sorted_indices = mx.argsort(typicality, axis=-1)
    sorted_probs = mx.take_along_axis(probs, sorted_indices, axis=-1)

    # Find cumulative threshold
    cumsum_probs = mx.cumsum(sorted_probs, axis=-1)
    mask = cumsum_probs <= typical_p
    first_element = mx.arange(mask.shape[-1]) == 0
    mask = mask | first_element

    # Zero out atypical tokens
    sorted_probs = mx.where(mask, sorted_probs, 0.0)
    sorted_probs = sorted_probs / mx.sum(sorted_probs, axis=-1, keepdims=True)

    # Sample
    samples = mx.random.categorical(mx.log(sorted_probs + 1e-10))

    # Map back
    original_shape = samples.shape
    samples_flat = samples.reshape(-1)
    sorted_indices_flat = sorted_indices.reshape(-1, logits.shape[-1])

    result = mx.take_along_axis(
        sorted_indices_flat,
        samples_flat[:, None],
        axis=-1,
    )[:, 0]

    return result.reshape(original_shape)


def apply_cfg(
    cond_logits: mx.array,
    uncond_logits: mx.array,
    cfg_scale: float = 1.5,
) -> mx.array:
    """Apply classifier-free guidance to logits.

    Args:
        cond_logits: Conditional logits.
        uncond_logits: Unconditional logits.
        cfg_scale: Guidance scale. 1.0 = no guidance.

    Returns:
        Guided logits.
    """
    return uncond_logits + cfg_scale * (cond_logits - uncond_logits)


def apply_repetition_penalty(
    logits: mx.array,
    generated_tokens: mx.array,
    penalty: float = 1.2,
) -> mx.array:
    """Apply repetition penalty to logits.

    Args:
        logits: Current logits of shape (batch, vocab_size).
        generated_tokens: Previously generated tokens of shape (batch, seq_len).
        penalty: Repetition penalty factor. 1.0 = no penalty.

    Returns:
        Penalized logits.
    """
    if penalty == 1.0:
        return logits

    batch_size, vocab_size = logits.shape

    # Create a mask of which tokens have been generated
    # Shape: (batch, vocab_size)
    token_mask = mx.zeros((batch_size, vocab_size), dtype=mx.bool_)

    # Build one-hot mask for each token position
    for b in range(batch_size):
        for token in generated_tokens[b]:
            token_idx = int(token)
            if 0 <= token_idx < vocab_size:
                # Create one-hot and OR with existing mask
                one_hot = mx.arange(vocab_size) == token_idx
                token_mask = mx.where(
                    (mx.arange(batch_size) == b)[:, None],
                    token_mask | one_hot,
                    token_mask
                )

    # Apply penalty: divide positive logits, multiply negative logits
    penalty_factor = mx.where(logits > 0, 1.0 / penalty, penalty)
    penalized = mx.where(token_mask, logits * penalty_factor, logits)

    return penalized


def apply_presence_penalty(
    logits: mx.array,
    generated_tokens: mx.array,
    penalty: float = 0.5,
) -> mx.array:
    """Apply presence penalty to logits.

    Unlike repetition penalty, this applies a flat penalty
    to all tokens that have appeared.

    Args:
        logits: Current logits of shape (batch, vocab_size).
        generated_tokens: Previously generated tokens.
        penalty: Presence penalty value.

    Returns:
        Penalized logits.
    """
    if penalty == 0.0:
        return logits

    batch_size, vocab_size = logits.shape

    # Track which tokens have appeared
    for b in range(batch_size):
        seen = set()
        for token in generated_tokens[b]:
            token_idx = int(token)
            if 0 <= token_idx < vocab_size and token_idx not in seen:
                seen.add(token_idx)
                logits = logits.at[b, token_idx].add(-penalty)

    return logits


def apply_frequency_penalty(
    logits: mx.array,
    generated_tokens: mx.array,
    penalty: float = 0.5,
) -> mx.array:
    """Apply frequency penalty to logits.

    Penalty scales with the number of times a token has appeared.

    Args:
        logits: Current logits of shape (batch, vocab_size).
        generated_tokens: Previously generated tokens.
        penalty: Frequency penalty factor.

    Returns:
        Penalized logits.
    """
    if penalty == 0.0:
        return logits

    batch_size, vocab_size = logits.shape

    # Count token frequencies
    for b in range(batch_size):
        counts = {}
        for token in generated_tokens[b]:
            token_idx = int(token)
            if 0 <= token_idx < vocab_size:
                counts[token_idx] = counts.get(token_idx, 0) + 1

        for token_idx, count in counts.items():
            logits = logits.at[b, token_idx].add(-penalty * count)

    return logits


def apply_temperature_decay(
    temperature: float,
    step: int,
    decay_rate: float = 0.99,
    min_temperature: float = 0.1,
) -> float:
    """Apply temperature decay over generation steps.

    Args:
        temperature: Current temperature.
        step: Current generation step.
        decay_rate: Decay rate per step.
        min_temperature: Minimum temperature.

    Returns:
        Decayed temperature.
    """
    decayed = temperature * (decay_rate ** step)
    return max(decayed, min_temperature)
