"""Partition-aware staged retrieval.

Clusters binary codes into P partitions at build time. At search time,
routes the query to the top-R most relevant partitions and scans only
those. Reduces O(n) to O(n * R/P) per query.

Architecture:
  Build:
    1. Binary quantize all vectors
    2. Cluster binary codes into P partitions (k-means on binary centroids)
    3. Store partition assignments and centroids

  Search:
    1. Binary quantize query
    2. Compute Hamming distance to all P partition centroids
    3. Select top-R partitions (routing)
    4. Scan only vectors in those R partitions (binary Hamming)
    5. Float rerank top rf*k candidates
    6. Return top-k
"""

import numpy as np
from typing import Optional

from bitcache.quantize import quantize
from bitcache.search import hamming_distance_single


class PartitionedIndex:
    """Partition-aware two-stage retrieval.

    Clusters vectors into partitions for sublinear search. Only scans
    vectors in the most relevant partitions, then reranks with float.

    Args:
        dim: Vector dimensionality.
        n_partitions: Number of partitions (P). More = faster search, risk of missing neighbors.
        n_probe: Number of partitions to scan per query (R). More = better recall, slower.
        rerank_factor: Candidates from scanned partitions to rerank with float.
    """

    def __init__(
        self,
        dim: int,
        n_partitions: int = 32,
        n_probe: int = 4,
        rerank_factor: int = 100,
    ):
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim
        self.n_partitions = n_partitions
        self.n_probe = n_probe
        self.rerank_factor = rerank_factor
        self.n_bytes = (dim + 7) // 8

        self._codes: Optional[np.ndarray] = None
        self._vectors: Optional[np.ndarray] = None
        self._n_vectors: int = 0

        # Partition structures
        self._centroids: Optional[np.ndarray] = None  # (P, n_bytes) binary centroids
        self._assignments: Optional[np.ndarray] = None  # (n,) partition ID per vector
        self._partition_indices: Optional[list] = None  # partition_id -> list of vector indices

    def build(self, vectors: np.ndarray) -> None:
        """Build the partitioned index.

        Args:
            vectors: Float array of shape (n, dim).
        """
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.shape[1] != self.dim:
            raise ValueError(f"expected dim={self.dim}, got {vectors.shape[1]}")

        n = len(vectors)
        self._n_vectors = n

        # Normalize and store
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms > 1e-10, norms, 1.0)
        self._vectors = vectors / norms

        # Binary quantize
        self._codes = quantize(self._vectors)

        # Cluster binary codes into partitions using binary k-means
        self._centroids, self._assignments = self._binary_kmeans(
            self._codes, self.n_partitions
        )

        # Build partition index (which vectors belong to each partition)
        self._partition_indices = [[] for _ in range(self.n_partitions)]
        for i, pid in enumerate(self._assignments):
            self._partition_indices[pid].append(i)

    def search(self, query: np.ndarray, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
        """Partition-routed search.

        Args:
            query: Float vector of shape (dim,).
            k: Number of results.

        Returns:
            Tuple of (scores, indices). Scores are float32 inner products.
        """
        if self._codes is None or self._n_vectors == 0:
            return np.array([], dtype=np.float32), np.array([], dtype=np.int64)

        query = np.asarray(query, dtype=np.float32).ravel()
        norm = np.linalg.norm(query)
        query_unit = query / norm if norm > 1e-10 else query

        # Binary quantize query
        query_code = quantize(query_unit[None, :])[0]

        # Route: find closest partitions
        centroid_dists = hamming_distance_single(query_code, self._centroids)
        probe_partitions = np.argsort(centroid_dists)[:self.n_probe]

        # Gather candidate indices from probed partitions
        candidate_indices = []
        for pid in probe_partitions:
            candidate_indices.extend(self._partition_indices[pid])
        candidate_indices = np.array(candidate_indices, dtype=np.int64)

        if len(candidate_indices) == 0:
            return np.array([], dtype=np.float32), np.array([], dtype=np.int64)

        # Binary scan within probed partitions
        candidate_codes = self._codes[candidate_indices]
        hamming_dists = hamming_distance_single(query_code, candidate_codes)

        # Select top rf*k from candidates
        n_rerank = min(k * self.rerank_factor, len(candidate_indices))
        if n_rerank < len(candidate_indices):
            top_local = np.argpartition(hamming_dists, n_rerank)[:n_rerank]
        else:
            top_local = np.arange(len(candidate_indices))

        rerank_indices = candidate_indices[top_local]

        # Float rerank
        rerank_vectors = self._vectors[rerank_indices]
        scores = rerank_vectors @ query_unit

        # Top-k
        top_k_local = np.argsort(scores)[::-1][:k]
        final_indices = rerank_indices[top_k_local]
        final_scores = scores[top_k_local]

        return final_scores, final_indices

    def search_batch(self, queries: np.ndarray, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
        """Batch search.

        Args:
            queries: Float array of shape (nq, dim).
            k: Number of results per query.

        Returns:
            Tuple of (scores, indices) each shape (nq, k).
        """
        queries = np.asarray(queries, dtype=np.float32)
        if queries.ndim == 1:
            queries = queries[None, :]

        nq = len(queries)
        all_scores = np.zeros((nq, k), dtype=np.float32)
        all_indices = np.zeros((nq, k), dtype=np.int64)

        for i in range(nq):
            scores, indices = self.search(queries[i], k=k)
            n = len(scores)
            all_scores[i, :n] = scores
            all_indices[i, :n] = indices

        return all_scores, all_indices

    def _binary_kmeans(
        self, codes: np.ndarray, k: int, max_iter: int = 20, seed: int = 42
    ) -> tuple[np.ndarray, np.ndarray]:
        """K-means clustering on binary codes using Hamming distance.

        Args:
            codes: Binary codes, shape (n, n_bytes).
            k: Number of clusters.
            max_iter: Maximum iterations.

        Returns:
            Tuple of (centroids shape (k, n_bytes), assignments shape (n,)).
        """
        n = len(codes)
        rng = np.random.default_rng(seed)

        # Initialize centroids by random selection
        init_indices = rng.choice(n, size=min(k, n), replace=False)
        centroids = codes[init_indices].copy()

        assignments = np.zeros(n, dtype=np.int32)

        for iteration in range(max_iter):
            # Assign each vector to nearest centroid
            new_assignments = np.zeros(n, dtype=np.int32)
            for i in range(n):
                dists = hamming_distance_single(codes[i], centroids)
                new_assignments[i] = int(np.argmin(dists))

            # Check convergence
            if np.array_equal(new_assignments, assignments):
                break
            assignments = new_assignments

            # Update centroids (majority vote per bit position)
            for c in range(k):
                members = codes[assignments == c]
                if len(members) == 0:
                    # Empty cluster: reinitialize randomly
                    centroids[c] = codes[rng.integers(n)]
                else:
                    # Majority vote: for each bit, take the most common value
                    # Unpack to bits, take mean, threshold at 0.5
                    unpacked = np.unpackbits(members, axis=1)
                    majority = (unpacked.mean(axis=0) > 0.5).astype(np.uint8)
                    centroids[c] = np.packbits(majority)

        return centroids, assignments

    def __len__(self) -> int:
        return self._n_vectors

    @property
    def partition_sizes(self) -> list:
        """Number of vectors in each partition."""
        if self._partition_indices is None:
            return []
        return [len(p) for p in self._partition_indices]

    @property
    def vectors_scanned_per_query(self) -> int:
        """Expected number of vectors scanned per query."""
        if self._n_vectors == 0:
            return 0
        avg_partition_size = self._n_vectors / self.n_partitions
        return int(avg_partition_size * self.n_probe)
