"""64-bit DCT perceptual hashes and exact Hamming distance.

Variant: 32x32 grayscale, unnormalised DCT-II, top-left 8x8 (including DC),
strictly above the median. Row-major bits run from most to least significant.
This matches ImageHash's default pHash on the same preprocessed pixels; it is
not a claim of compatibility with every algorithm called pHash.
"""

from collections.abc import Sequence

import numpy as np
import scipy
from scipy.fft import dctn


HASH_BITS = 64
PHASH_PARAMETERS = {
    "version": 1, "dct_type": 2, "norm": "backward", "low_frequency_size": 8,
    "dc": "included", "threshold": "median-strict", "bit_order": "row-major-msb",
    "scipy": scipy.__version__,
}


def phash(pixels: np.ndarray) -> np.uint64:
    """Hash already preprocessed 32x32 grayscale uint8 pixels into eight bytes."""
    pixels = np.asarray(pixels)
    if pixels.shape != (32, 32) or pixels.dtype != np.uint8:
        raise ValueError("pHash expects 32x32 grayscale uint8 pixels; use load_image with PHASH_PREPROCESS")
    low = dctn(pixels.astype(np.float64), type=2, norm="backward")[:8, :8]
    bits = (low > np.median(low)).ravel()
    return np.uint64(int.from_bytes(np.packbits(bits, bitorder="big").tobytes(), "big"))



def hash_value(value: int) -> int:
    """Reject negative, overflowing and noninteger hashes instead of wrapping."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError("hash must be an unsigned 64-bit integer")
    value = int(value)
    if not 0 <= value < (1 << HASH_BITS):
        raise ValueError("hash must be an unsigned 64-bit integer")
    return value


def hash_array(values: Sequence[int] | np.ndarray) -> np.ndarray:
    """Validate a one-dimensional catalogue without silently truncating floats."""
    # NumPy may coerce [0, 2**64-1] to float64 and lose bits before validation.
    array = values if isinstance(values, np.ndarray) else np.asarray(values, dtype=object)
    if array.ndim != 1:
        raise ValueError("hashes must be one-dimensional")
    if array.size == 0:
        return np.empty(0, dtype=np.uint64)
    if array.dtype.kind == "u":
        return array.astype(np.uint64, copy=False)
    return np.array([hash_value(v) for v in array], dtype=np.uint64)


def hamming_distance(left: int, right: int) -> int:
    """Number of differing bits, in [0, 64]; no normalisation or approximation."""
    return (hash_value(left) ^ hash_value(right)).bit_count()


def hamming_distances(query: int, hashes: Sequence[int] | np.ndarray) -> np.ndarray:
    """Vectorised XOR/popcount against a catalogue; never allocate an N x N matrix."""
    return np.bitwise_count(hash_array(hashes) ^ np.uint64(hash_value(query)))
