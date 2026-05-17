"""Binary quantization: float vectors → packed bit arrays.

Sign-bit quantization: each dimension becomes 1 if positive, 0 otherwise.
Packed into uint8 arrays for efficient XOR + popcount via numpy.

A 1536-dim float32 vector (6144 bytes) becomes 192 bytes (32x compression).
"""

import numpy as np


def quantize(vectors: np.ndarray) -> np.ndarray:
    """Quantize float vectors to packed binary representation.

    Args:
        vectors: Array of shape (n, dim) with float32 values.

    Returns:
        Packed binary array of shape (n, ceil(dim/8)) with uint8 values.
    """
    if vectors.ndim == 1:
        vectors = vectors[None, :]
    return np.packbits((vectors > 0).astype(np.uint8), axis=1)


def quantize_batch(vectors: np.ndarray) -> np.ndarray:
    """Alias for quantize. Accepts (n, dim) float array."""
    return quantize(vectors)


def quantize_sign_magnitude(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """2-bit Sign-Magnitude quantization.

    Bit 0 (sign): 1 if positive, 0 if negative
    Bit 1 (magnitude): 1 if |val| > median(|vector|), 0 otherwise

    Args:
        vectors: Array of shape (n, dim).

    Returns:
        Tuple of (sign_bits, magnitude_bits), each shape (n, ceil(dim/8)).
    """
    if vectors.ndim == 1:
        vectors = vectors[None, :]

    sign_bits = np.packbits((vectors > 0).astype(np.uint8), axis=1)

    abs_vals = np.abs(vectors)
    thresholds = np.median(abs_vals, axis=1, keepdims=True)
    mag_bits = np.packbits((abs_vals > thresholds).astype(np.uint8), axis=1)

    return sign_bits, mag_bits
