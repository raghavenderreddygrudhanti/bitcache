"""XOR + POPCOUNT similarity search on packed binary vectors.

Hamming distance = number of differing bits between two binary vectors.
Computed as: popcount(XOR(a, b)) summed across bytes.

Uses numpy unpackbits for fast bit counting on contiguous arrays.
"""

import numpy as np


def hamming_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute Hamming distance between query and database vectors.

    Args:
        a: Query vectors, shape (nq, n_bytes) uint8.
        b: Database vectors, shape (n, n_bytes) uint8.

    Returns:
        Distance matrix of shape (nq, n) with uint32 values.
    """
    # XOR all pairs, count bits per byte via lookup, sum across bytes
    xor = np.bitwise_xor(a[:, None, :], b[None, :, :])
    return _popcount_bytes(xor).sum(axis=2).astype(np.uint32)


def hamming_distance_single(query: np.ndarray, database: np.ndarray) -> np.ndarray:
    """Compute Hamming distance between one query and all database vectors.

    Args:
        query: Single query, shape (n_bytes,) uint8.
        database: Database vectors, shape (n, n_bytes) uint8.

    Returns:
        Distances array of shape (n,) uint32.
    """
    xor = np.bitwise_xor(query[None, :], database)
    return _popcount_bytes(xor).sum(axis=1).astype(np.uint32)


def search(
    query: np.ndarray,
    database: np.ndarray,
    k: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Find k nearest vectors by Hamming distance.

    Args:
        query: Single query, shape (n_bytes,) uint8.
        database: Database vectors, shape (n, n_bytes) uint8.
        k: Number of results to return.

    Returns:
        Tuple of (distances, indices) each shape (k,).
    """
    k = min(k, len(database))
    if k == 0:
        return np.array([], dtype=np.uint32), np.array([], dtype=np.int64)

    dists = hamming_distance_single(query, database)

    if k < len(dists):
        # Partial sort for top-k
        part_idx = np.argpartition(dists, k)[:k]
        part_dists = dists[part_idx]
        sorted_order = np.argsort(part_dists)
        indices = part_idx[sorted_order]
        distances = part_dists[sorted_order]
    else:
        sorted_order = np.argsort(dists)
        indices = sorted_order[:k]
        distances = dists[indices]

    return distances, indices


def search_batch(
    queries: np.ndarray,
    database: np.ndarray,
    k: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Batch search: find k nearest for each query.

    Args:
        queries: Query vectors, shape (nq, n_bytes) uint8.
        database: Database vectors, shape (n, n_bytes) uint8.
        k: Number of results per query.

    Returns:
        Tuple of (distances, indices) each shape (nq, k).
    """
    k = min(k, len(database))
    nq = len(queries)

    all_dists = hamming_distance(queries, database)

    if k < all_dists.shape[1]:
        part_idx = np.argpartition(all_dists, k, axis=1)[:, :k]
        part_dists = np.take_along_axis(all_dists, part_idx, axis=1)
        sorted_order = np.argsort(part_dists, axis=1)
        indices = np.take_along_axis(part_idx, sorted_order, axis=1)
        distances = np.take_along_axis(part_dists, sorted_order, axis=1)
    else:
        sorted_order = np.argsort(all_dists, axis=1)[:, :k]
        indices = sorted_order
        distances = np.take_along_axis(all_dists, sorted_order, axis=1)

    return distances, indices


# Popcount lookup table: number of set bits for each byte value 0-255
_POPCOUNT_TABLE = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint8)


def _popcount_bytes(data: np.ndarray) -> np.ndarray:
    """Count set bits per byte using lookup table."""
    return _POPCOUNT_TABLE[data]
