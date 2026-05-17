"""Graph memory: entity-relationship storage + vector retrieval.

Combines vector similarity search with knowledge graph traversal
for multi-hop reasoning. Stores (entity, relation, entity) triples
and links them to vector embeddings.

Based on: "HippoRAG: Neurobiologically Inspired Long-Term Memory
for Large Language Models" (Gutiérrez et al., NeurIPS 2024)

Architecture:
  - Entities stored as nodes with vector embeddings
  - Relations stored as directed edges between entities
  - Search: vector similarity finds entry points, graph traversal
    expands context via relationships
"""

from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from bitcache.streaming import StreamingIndex


class GraphMemory:
    """Knowledge graph + vector retrieval for multi-hop reasoning.

    Stores entities with embeddings and connects them via typed
    relations. Search combines vector similarity (find relevant
    entities) with graph traversal (expand via relationships).

    Args:
        dim: Vector dimensionality for entity embeddings.
        max_hops: Maximum graph traversal depth during search.
    """

    def __init__(self, dim: int, max_hops: int = 2):
        self.dim = dim
        self.max_hops = max_hops

        self._index = StreamingIndex(dim=dim, rerank_factor=10)

        # Graph structure
        self._edges: Dict[str, List[Tuple[str, str]]] = defaultdict(list)  # entity → [(relation, target)]
        self._reverse_edges: Dict[str, List[Tuple[str, str]]] = defaultdict(list)  # entity → [(relation, source)]
        self._entity_data: Dict[str, Dict[str, Any]] = {}

    def add_entity(
        self,
        entity_id: str,
        vector: np.ndarray,
        name: Optional[str] = None,
        entity_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Add an entity with its embedding.

        Args:
            entity_id: Unique identifier for the entity.
            vector: Embedding vector.
            name: Human-readable name.
            entity_type: Type category (person, org, concept, etc.).
            metadata: Additional metadata.

        Returns:
            The entity ID.
        """
        meta = metadata or {}
        meta["name"] = name or entity_id
        meta["entity_type"] = entity_type or "unknown"

        self._index.insert(vector, id=entity_id, metadata=meta)
        self._entity_data[entity_id] = {
            "name": name or entity_id,
            "entity_type": entity_type or "unknown",
            "metadata": metadata or {},
        }
        return entity_id

    def add_relation(self, source: str, relation: str, target: str) -> bool:
        """Add a directed relation between two entities.

        Args:
            source: Source entity ID.
            relation: Relation type (e.g., "works_at", "integrates_with").
            target: Target entity ID.

        Returns:
            True if both entities exist and relation was added.
        """
        if source not in self._entity_data or target not in self._entity_data:
            return False

        # Avoid duplicates
        edge = (relation, target)
        if edge not in self._edges[source]:
            self._edges[source].append(edge)
            self._reverse_edges[target].append((relation, source))
        return True

    def get_relations(self, entity_id: str) -> List[Dict[str, str]]:
        """Get all outgoing relations from an entity.

        Returns:
            List of {relation, target, target_name} dicts.
        """
        results = []
        for relation, target in self._edges.get(entity_id, []):
            target_data = self._entity_data.get(target, {})
            results.append({
                "relation": relation,
                "target": target,
                "target_name": target_data.get("name", target),
            })
        return results

    def get_incoming_relations(self, entity_id: str) -> List[Dict[str, str]]:
        """Get all incoming relations to an entity.

        Returns:
            List of {relation, source, source_name} dicts.
        """
        results = []
        for relation, source in self._reverse_edges.get(entity_id, []):
            source_data = self._entity_data.get(source, {})
            results.append({
                "relation": relation,
                "source": source,
                "source_name": source_data.get("name", source),
            })
        return results

    def search(
        self,
        query: np.ndarray,
        k: int = 5,
        expand: bool = True,
        max_hops: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Search by vector similarity + graph expansion.

        1. Find top entities by vector similarity
        2. Expand each via graph traversal (follow relations)
        3. Return entities with their relationship context

        Args:
            query: Query vector.
            k: Number of seed entities from vector search.
            expand: Whether to expand via graph traversal.
            max_hops: Override default max_hops.

        Returns:
            List of result dicts with entity info and related context.
        """
        scores, ids, metas = self._index.search(query, k=k)

        if not ids:
            return []

        hops = max_hops if max_hops is not None else self.max_hops
        results = []

        for score, entity_id, meta in zip(scores, ids, metas):
            result = {
                "id": entity_id,
                "name": meta.get("name", entity_id),
                "entity_type": meta.get("entity_type", "unknown"),
                "score": float(score),
                "relations": self.get_relations(entity_id),
                "incoming": self.get_incoming_relations(entity_id),
            }

            if expand and hops > 0:
                result["expanded"] = self._expand(entity_id, hops)

            results.append(result)

        return results

    def search_by_relation(
        self,
        source: str,
        relation: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Find entities connected by a specific relation.

        Args:
            source: Source entity ID.
            relation: Filter by relation type. None returns all.

        Returns:
            List of connected entity dicts.
        """
        results = []
        for rel, target in self._edges.get(source, []):
            if relation is not None and rel != relation:
                continue
            target_data = self._entity_data.get(target, {})
            results.append({
                "id": target,
                "name": target_data.get("name", target),
                "entity_type": target_data.get("entity_type", "unknown"),
                "relation": rel,
            })
        return results

    def get_path(self, source: str, target: str, max_depth: int = 3) -> Optional[List[Dict[str, str]]]:
        """Find shortest path between two entities via BFS.

        Args:
            source: Start entity ID.
            target: End entity ID.
            max_depth: Maximum path length.

        Returns:
            List of {entity, relation} steps, or None if no path.
        """
        if source not in self._entity_data or target not in self._entity_data:
            return None

        visited: Set[str] = {source}
        queue: List[Tuple[str, List[Dict[str, str]]]] = [(source, [])]

        while queue:
            current, path = queue.pop(0)

            if len(path) >= max_depth:
                continue

            for relation, neighbor in self._edges.get(current, []):
                if neighbor == target:
                    return path + [{"entity": neighbor, "relation": relation}]

                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((
                        neighbor,
                        path + [{"entity": current, "relation": relation}],
                    ))

        return None

    def _expand(self, entity_id: str, hops: int) -> List[Dict[str, Any]]:
        """Expand entity context via graph traversal."""
        visited: Set[str] = {entity_id}
        expanded = []
        frontier = [entity_id]

        for hop in range(hops):
            next_frontier = []
            for node in frontier:
                for relation, neighbor in self._edges.get(node, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_frontier.append(neighbor)
                        neighbor_data = self._entity_data.get(neighbor, {})
                        expanded.append({
                            "id": neighbor,
                            "name": neighbor_data.get("name", neighbor),
                            "relation_from": node,
                            "relation": relation,
                            "hop": hop + 1,
                        })
            frontier = next_frontier

        return expanded

    def remove_entity(self, entity_id: str) -> bool:
        """Remove an entity and all its relations."""
        if entity_id not in self._entity_data:
            return False

        # Remove outgoing edges
        for relation, target in self._edges.get(entity_id, []):
            self._reverse_edges[target] = [
                (r, s) for r, s in self._reverse_edges[target] if s != entity_id
            ]
        del self._edges[entity_id]

        # Remove incoming edges
        for relation, source in self._reverse_edges.get(entity_id, []):
            self._edges[source] = [
                (r, t) for r, t in self._edges[source] if t != entity_id
            ]
        del self._reverse_edges[entity_id]

        del self._entity_data[entity_id]
        self._index.delete(entity_id)
        return True

    @property
    def num_entities(self) -> int:
        return len(self._entity_data)

    @property
    def num_relations(self) -> int:
        return sum(len(edges) for edges in self._edges.values())
