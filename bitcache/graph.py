"""Vamana graph index in binary metric space.

Implements a DiskANN/Vamana-style graph where edge selection, pruning,
and navigation all operate on Hamming distances between binary-quantized
vectors. Float32 reranking is applied only to the final candidate set.

Based on: "QuIVer: Rethinking ANN Graph Topology via Training-Free
Binary Quantization" (Xiao et al., arXiv:2605.02171, 2026)

Graph construction:
  1. Initialize random graph (R edges per node)
  2. For each vector, greedy search to find candidates
  3. Prune candidates using alpha-diversity rule
  4. Add reverse edges to maintain connectivity

Search:
  1. Beam search from entry point using Hamming distance
  2. Rerank top candidates using float32 inner product
"""

import numpy as np
from typing import Optional

from bitcache.quantize import quantize
from bitcache.search import hamming_distance_single


class VamanaIndex:
    """Graph-based ANN index operating in binary Hamming space.

    Builds a Vamana graph where all distance computations during
    construction and search use binary quantized vectors. Final
    results are optionally reranked with float32 precision.

    Args:
        dim: Vector dimensionality.
        R: Max edges per node (graph degree).
        L_build: Search list size during construction.
        alpha: Diversity pruning parameter (>= 1.0).
    """

    def __init__(
        self,
        dim: int,
        R: int = 32,
        L_build: int = 50,
        alpha: float = 1.2,
    ):
        self.dim = dim
        self.R = R
        self.L_build = L_build
        self.alpha = alpha
        self.n_bytes = (dim + 7) // 8

        self._codes: Optional[np.ndarray] = None
        self._vectors: Optional[np.ndarray] = None
        self._graph: Optional[list] = None
        self._entry_point: int = 0
        self._n_vectors: int = 0

    def build(self, vectors: np.ndarray) -> None:
        """Build the graph index from vectors.

        Args:
            vectors: Float array of shape (n, dim).
        """
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.shape[1] != self.dim:
            raise ValueError(f"expected dim={self.dim}, got {vectors.shape[1]}")

        n = len(vectors)
        self._n_vectors = n

        # Normalize and store float vectors for reranking
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms > 1e-10, norms, 1.0)
        self._vectors = vectors / norms

        # Binary quantize
        self._codes = quantize(self._vectors)

        # Initialize random graph
        self._graph = [set() for _ in range(n)]
        rng = np.random.default_rng(42)
        for i in range(n):
            neighbors = rng.choice(n, size=min(self.R, n - 1), replace=False)
            neighbors = neighbors[neighbors != i][:self.R]
            self._graph[i] = set(neighbors.tolist())

        # Pick entry point as medoid (closest to centroid in binary space)
        centroid_code = quantize(self._vectors.mean(axis=0, keepdims=True))[0]
        dists = hamming_distance_single(centroid_code, self._codes)
        self._entry_point = int(np.argmin(dists))

        # Iterative graph improvement
        order = rng.permutation(n)
        for idx in order:
            # Greedy search to find candidates
            candidates = self._greedy_search(self._codes[idx], self.L_build)

            # Prune with alpha-diversity
            pruned = self._robust_prune(idx, candidates)

            # Update edges
            self._graph[idx] = set(pruned)

            # Add reverse edges
            for neighbor in pruned:
                self._graph[neighbor].add(idx)
                if len(self._graph[neighbor]) > self.R:
                    # Prune over-full neighbor
                    neighbor_pruned = self._robust_prune(
                        neighbor, list(self._graph[neighbor])
                    )
                    self._graph[neighbor] = set(neighbor_pruned)

    def search(
        self,
        query: np.ndarray,
        k: int = 10,
        ef: int = 50,
        rerank: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Search for k nearest vectors.

        Args:
            query: Float vector of shape (dim,).
            k: Number of results.
            ef: Search beam width (higher = better recall, slower).
            rerank: If True, rerank candidates with float32 inner product.

        Returns:
            Tuple of (scores, indices). Scores are Hamming distances
            (if rerank=False) or negative inner products (if rerank=True).
        """
        if self._codes is None or self._n_vectors == 0:
            return np.array([]), np.array([], dtype=np.int64)

        query = np.asarray(query, dtype=np.float32)
        if query.ndim == 2:
            query = query[0]

        # Normalize query
        norm = np.linalg.norm(query)
        if norm > 1e-10:
            query_unit = query / norm
        else:
            query_unit = query

        # Quantize query
        query_code = quantize(query_unit[None, :])[0]

        # Beam search on graph
        candidates = self._greedy_search(query_code, max(ef, k))

        if rerank and self._vectors is not None:
            # Rerank with float32 inner product
            cand_indices = np.array(candidates[:min(ef, len(candidates))])
            scores = self._vectors[cand_indices] @ query_unit
            top_k_local = np.argsort(scores)[::-1][:k]
            indices = cand_indices[top_k_local]
            final_scores = scores[top_k_local]
            return final_scores, indices
        else:
            # Return Hamming distances
            cand_indices = np.array(candidates[:k])
            dists = np.array([
                int(hamming_distance_single(query_code, self._codes[i:i+1])[0])
                for i in cand_indices
            ])
            sorted_order = np.argsort(dists)
            return dists[sorted_order], cand_indices[sorted_order]

    def _greedy_search(self, query_code: np.ndarray, L: int) -> list:
        """Greedy beam search on the graph using Hamming distance.

        Args:
            query_code: Binary query, shape (n_bytes,).
            L: Max candidates to track.

        Returns:
            List of candidate indices sorted by Hamming distance.
        """
        visited = set()
        # Start from entry point
        candidates = [(self._hamming(query_code, self._entry_point), self._entry_point)]
        visited.add(self._entry_point)

        result = list(candidates)

        while candidates:
            # Pop closest unvisited
            candidates.sort()
            current_dist, current = candidates.pop(0)

            # Expand neighbors
            for neighbor in self._graph[current]:
                if neighbor in visited:
                    continue
                visited.add(neighbor)

                dist = self._hamming(query_code, neighbor)
                result.append((dist, neighbor))

                # Keep beam bounded
                if len(result) > L * 2:
                    result.sort()
                    result = result[:L]

                # Add to expansion queue if promising
                if len(result) < L or dist < result[-1][0]:
                    candidates.append((dist, neighbor))

            if len(candidates) > L:
                candidates.sort()
                candidates = candidates[:L]

        result.sort()
        return [idx for _, idx in result[:L]]

    def _robust_prune(self, node: int, candidates: list) -> list:
        """Alpha-diversity pruning.

        Keeps neighbors that are diverse — not too close to each other.
        A candidate is kept only if it's not "dominated" by an already-
        selected neighbor (within alpha factor).

        Args:
            node: The node being pruned.
            candidates: List of candidate neighbor indices.

        Returns:
            Pruned list of neighbor indices (max R).
        """
        if not candidates:
            return []

        node_code = self._codes[node]

        # Score all candidates
        scored = []
        for c in candidates:
            if c == node:
                continue
            dist = int(hamming_distance_single(node_code, self._codes[c:c+1])[0])
            scored.append((dist, c))
        scored.sort()

        pruned = []
        for dist_to_node, candidate in scored:
            if len(pruned) >= self.R:
                break

            # Check if candidate is dominated by any already-selected neighbor
            dominated = False
            for selected in pruned:
                dist_cand_to_selected = int(
                    hamming_distance_single(
                        self._codes[candidate], self._codes[selected:selected+1]
                    )[0]
                )
                if dist_cand_to_selected * self.alpha <= dist_to_node:
                    dominated = True
                    break

            if not dominated:
                pruned.append(candidate)

        return pruned

    def _hamming(self, query_code: np.ndarray, idx: int) -> int:
        """Hamming distance between query and indexed vector."""
        return int(hamming_distance_single(query_code, self._codes[idx:idx+1])[0])

    def __len__(self) -> int:
        return self._n_vectors
