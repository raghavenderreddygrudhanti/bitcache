"""Float-space routed retrieval: semantic routing + binary filtering.

Gen3 architecture:
  1. Float k-means to build partition centroids (preserves semantic neighborhoods)
  2. Route query to nearest centroids via float inner product (cheap: only P centroids)
  3. Binary Hamming scan inside selected partitions (fast: only scans subset)
  4. Float rerank top candidates (precise: small candidate set)

This fixes Gen2's failure: binary k-means centroids don't preserve semantic
structure at scale. Float centroids do — because they operate in the same
metric space as the original embeddings.
"""

import numpy as np
from typing import Optional

from bitcache.quantize import quantize
from bitcache.search import hamming_distance_single


class FloatRoutedIndex:
    """Float-space routed retrieval with binary candidate filtering.

    Uses float k-means for partition routing (preserves semantic structure)
    and binary Hamming scan within partitions (fast candidate filtering).

    Args:
        dim: Vector dimensionality.
        n_partitions: Number of partitions (P).
        n_probe: Partitions to scan per query (R).
        rerank_factor: Candidates to rerank with float from scanned partitions.
        kmeans_iter: K-means iterations for partition construction.
    """

    def __init__(
        self,
        dim: int,
        n_partitions: int = 128,
        n_probe: int = 8,
        rerank_factor: int = 100,
        kmeans_iter: int = 10,
    ):
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim
        self.n_partitions = n_partitions
        self.n_probe = n_probe
        self.rerank_factor = rerank_factor
        self.kmeans_iter = kmeans_iter
        self.n_bytes = (dim + 7) // 8

        self._codes: Optional[np.ndarray] = None
        self._vectors: Optional[np.ndarray] = None
        self._n_vectors: int = 0

        # Float centroids for routing
        self._centroids: Optional[np.ndarray] = None  # (P, dim) float32
        self._assignments: Optional[np.ndarray] = None  # (n,) partition IDs
        self._partition_indices: Optional[list] = None  # partition_id -> vector indices
        # Binary codes per partition for fast scan
        self._partition_codes: Optional[list] = None  # partition_id -> binary codes array

    def build(self, vectors: np.ndarray) -> None:
        """Build the float-routed index.

        Args:
            vectors: Float array of shape (n, dim).
        """
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.shape[1] != self.dim:
            raise ValueError(f"expected dim={self.dim}, got {vectors.shape[1]}")

        n = len(vectors)
        self._n_vectors = n

        # Normalize
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms > 1e-10, norms, 1.0)
        self._vectors = vectors / norms

        # Binary quantize all vectors
        self._codes = quantize(self._vectors)

        # Float k-means for partition centroids
        self._centroids, self._assignments = self._float_kmeans(
            self._vectors, self.n_partitions, self.kmeans_iter
        )

        # Build partition index
        self._partition_indices = [[] for _ in range(self.n_partitions)]
        for i, pid in enumerate(self._assignments):
            self._partition_indices[pid].append(i)

        # Pre-group binary codes per partition for cache-friendly scan
        self._partition_codes = []
        for pid in range(self.n_partitions):
            indices = self._partition_indices[pid]
            if indices:
                self._partition_codes.append(self._codes[indices])
            else:
                self._partition_codes.append(np.empty((0, self.n_bytes), dtype=np.uint8))

    def search(self, query: np.ndarray, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
        """Float-routed search.

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

        # Route: float inner product with centroids (cheap — only P vectors)
        centroid_scores = self._centroids @ query_unit
        probe_partitions = np.argsort(centroid_scores)[::-1][:self.n_probe]

        # Gather candidates from probed partitions
        query_code = quantize(query_unit[None, :])[0]
        candidate_indices = []
        candidate_dists = []

        for pid in probe_partitions:
            p_indices = self._partition_indices[pid]
            if not p_indices:
                continue
            p_codes = self._partition_codes[pid]
            # Binary Hamming scan within partition
            dists = hamming_distance_single(query_code, p_codes)
            candidate_indices.extend(p_indices)
            candidate_dists.extend(dists.tolist())

        if not candidate_indices:
            return np.array([], dtype=np.float32), np.array([], dtype=np.int64)

        candidate_indices = np.array(candidate_indices, dtype=np.int64)
        candidate_dists = np.array(candidate_dists, dtype=np.uint32)

        # Select top rf*k by Hamming distance
        n_rerank = min(k * self.rerank_factor, len(candidate_indices))
        if n_rerank < len(candidate_indices):
            top_local = np.argpartition(candidate_dists, n_rerank)[:n_rerank]
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
        """Batch search."""
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

    def _float_kmeans(
        self, vectors: np.ndarray, k: int, max_iter: int, seed: int = 42
    ) -> tuple[np.ndarray, np.ndarray]:
        """K-means in float space using inner product.

        Args:
            vectors: Unit-normalized float vectors (n, dim).
            k: Number of clusters.
            max_iter: Maximum iterations.

        Returns:
            Tuple of (centroids (k, dim), assignments (n,)).
        """
        n = len(vectors)
        rng = np.random.default_rng(seed)

        # Initialize: k-means++ style
        centroids = np.zeros((k, self.dim), dtype=np.float32)
        centroids[0] = vectors[rng.integers(n)]

        for c in range(1, k):
            # Compute distance to nearest existing centroid
            sims = vectors @ centroids[:c].T  # (n, c)
            max_sims = sims.max(axis=1)  # closest centroid similarity
            # Probability proportional to (1 - max_sim) — farther points more likely
            probs = 1.0 - max_sims
            probs = np.maximum(probs, 0)
            probs /= probs.sum() + 1e-10
            centroids[c] = vectors[rng.choice(n, p=probs)]

        # Iterate
        assignments = np.zeros(n, dtype=np.int32)
        for iteration in range(max_iter):
            # Assign
            sims = vectors @ centroids.T  # (n, k)
            new_assignments = sims.argmax(axis=1).astype(np.int32)

            if np.array_equal(new_assignments, assignments):
                break
            assignments = new_assignments

            # Update centroids
            for c in range(k):
                members = vectors[assignments == c]
                if len(members) > 0:
                    centroid = members.mean(axis=0)
                    norm = np.linalg.norm(centroid)
                    centroids[c] = centroid / norm if norm > 1e-10 else centroid
                else:
                    centroids[c] = vectors[rng.integers(n)]

        return centroids, assignments

    def __len__(self) -> int:
        return self._n_vectors

    @property
    def partition_sizes(self) -> list:
        if self._partition_indices is None:
            return []
        return [len(p) for p in self._partition_indices]

    @property
    def vectors_scanned_per_query(self) -> int:
        if self._n_vectors == 0:
            return 0
        avg_size = self._n_vectors / self.n_partitions
        return int(avg_size * self.n_probe)
