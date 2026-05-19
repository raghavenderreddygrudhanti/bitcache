//! Graph memory: entity-relationship storage + vector retrieval.
//!
//! Combines vector similarity search with knowledge graph traversal
//! for multi-hop reasoning. Stores (entity, relation, entity) triples
//! and links them to vector embeddings.

use std::collections::{HashMap, HashSet, VecDeque};

use crate::streaming::StreamingIndex;

/// A relation between two entities.
#[derive(Clone, Debug)]
pub struct Relation {
    pub relation_type: String,
    pub target: String,
}

/// A search result with graph context.
#[derive(Clone, Debug)]
pub struct GraphSearchResult {
    pub id: String,
    pub name: String,
    pub entity_type: String,
    pub score: f32,
    pub relations: Vec<RelationInfo>,
    pub expanded: Vec<ExpandedEntity>,
}

#[derive(Clone, Debug)]
pub struct RelationInfo {
    pub relation: String,
    pub target: String,
    pub target_name: String,
}

#[derive(Clone, Debug)]
pub struct ExpandedEntity {
    pub id: String,
    pub name: String,
    pub relation_from: String,
    pub relation: String,
    pub hop: usize,
}

/// Knowledge graph + vector retrieval for multi-hop reasoning.
pub struct GraphMemory {
    dim: usize,
    max_hops: usize,
    index: StreamingIndex,
    edges: HashMap<String, Vec<(String, String)>>,         // entity → [(relation, target)]
    reverse_edges: HashMap<String, Vec<(String, String)>>, // entity → [(relation, source)]
    entity_data: HashMap<String, EntityData>,
}

#[derive(Clone, Debug)]
struct EntityData {
    name: String,
    entity_type: String,
}

impl GraphMemory {
    pub fn new(dim: usize, max_hops: usize) -> Self {
        Self {
            dim,
            max_hops,
            index: StreamingIndex::new(dim, 10),
            edges: HashMap::new(),
            reverse_edges: HashMap::new(),
            entity_data: HashMap::new(),
        }
    }

    /// Add an entity with its embedding.
    pub fn add_entity(
        &mut self,
        entity_id: &str,
        vector: &[f32],
        name: Option<&str>,
        entity_type: Option<&str>,
    ) -> String {
        let name = name.unwrap_or(entity_id).to_string();
        let etype = entity_type.unwrap_or("unknown").to_string();

        let mut meta = HashMap::new();
        meta.insert("name".to_string(), name.clone());
        meta.insert("entity_type".to_string(), etype.clone());

        self.index.insert(vector, Some(entity_id.to_string()), Some(meta));
        self.entity_data.insert(entity_id.to_string(), EntityData {
            name,
            entity_type: etype,
        });

        entity_id.to_string()
    }

    /// Add a directed relation between two entities.
    pub fn add_relation(&mut self, source: &str, relation: &str, target: &str) -> bool {
        if !self.entity_data.contains_key(source) || !self.entity_data.contains_key(target) {
            return false;
        }

        let edge = (relation.to_string(), target.to_string());
        let edges = self.edges.entry(source.to_string()).or_default();
        if !edges.contains(&edge) {
            edges.push(edge);
            self.reverse_edges
                .entry(target.to_string())
                .or_default()
                .push((relation.to_string(), source.to_string()));
        }
        true
    }

    /// Search by vector similarity + graph expansion.
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        expand: bool,
        max_hops: Option<usize>,
    ) -> Vec<GraphSearchResult> {
        let (scores, ids, metas) = self.index.search(query, k);

        if ids.is_empty() {
            return vec![];
        }

        let hops = max_hops.unwrap_or(self.max_hops);

        ids.into_iter()
            .zip(scores.into_iter())
            .zip(metas.into_iter())
            .map(|((id, score), meta)| {
                let name = meta.get("name").cloned().unwrap_or_else(|| id.clone());
                let entity_type = meta.get("entity_type").cloned().unwrap_or_else(|| "unknown".to_string());

                let relations = self.get_relations(&id);
                let expanded = if expand && hops > 0 {
                    self.expand(&id, hops)
                } else {
                    vec![]
                };

                GraphSearchResult {
                    id,
                    name,
                    entity_type,
                    score,
                    relations,
                    expanded,
                }
            })
            .collect()
    }

    /// Get all outgoing relations from an entity.
    pub fn get_relations(&self, entity_id: &str) -> Vec<RelationInfo> {
        self.edges.get(entity_id)
            .map(|edges| {
                edges.iter().map(|(rel, target)| {
                    let target_name = self.entity_data.get(target)
                        .map(|d| d.name.clone())
                        .unwrap_or_else(|| target.clone());
                    RelationInfo {
                        relation: rel.clone(),
                        target: target.clone(),
                        target_name,
                    }
                }).collect()
            })
            .unwrap_or_default()
    }

    /// Find shortest path between two entities via BFS.
    pub fn get_path(&self, source: &str, target: &str, max_depth: usize) -> Option<Vec<(String, String)>> {
        if !self.entity_data.contains_key(source) || !self.entity_data.contains_key(target) {
            return None;
        }

        let mut visited: HashSet<String> = HashSet::new();
        visited.insert(source.to_string());

        let mut queue: VecDeque<(String, Vec<(String, String)>)> = VecDeque::new();
        queue.push_back((source.to_string(), vec![]));

        while let Some((current, path)) = queue.pop_front() {
            if path.len() >= max_depth {
                continue;
            }

            if let Some(edges) = self.edges.get(&current) {
                for (relation, neighbor) in edges {
                    if neighbor == target {
                        let mut result = path.clone();
                        result.push((relation.clone(), neighbor.clone()));
                        return Some(result);
                    }

                    if !visited.contains(neighbor) {
                        visited.insert(neighbor.clone());
                        let mut new_path = path.clone();
                        new_path.push((relation.clone(), current.clone()));
                        queue.push_back((neighbor.clone(), new_path));
                    }
                }
            }
        }

        None
    }

    /// Remove an entity and all its relations.
    pub fn remove_entity(&mut self, entity_id: &str) -> bool {
        if !self.entity_data.contains_key(entity_id) {
            return false;
        }

        // Remove outgoing edges
        if let Some(edges) = self.edges.remove(entity_id) {
            for (_, target) in &edges {
                if let Some(rev) = self.reverse_edges.get_mut(target) {
                    rev.retain(|(_, src)| src != entity_id);
                }
            }
        }

        // Remove incoming edges
        if let Some(rev_edges) = self.reverse_edges.remove(entity_id) {
            for (_, source) in &rev_edges {
                if let Some(fwd) = self.edges.get_mut(source) {
                    fwd.retain(|(_, tgt)| tgt != entity_id);
                }
            }
        }

        self.entity_data.remove(entity_id);
        self.index.delete(entity_id);
        true
    }

    fn expand(&self, entity_id: &str, hops: usize) -> Vec<ExpandedEntity> {
        let mut visited: HashSet<String> = HashSet::new();
        visited.insert(entity_id.to_string());
        let mut expanded = Vec::new();
        let mut frontier = vec![entity_id.to_string()];

        for hop in 0..hops {
            let mut next_frontier = Vec::new();
            for node in &frontier {
                if let Some(edges) = self.edges.get(node) {
                    for (relation, neighbor) in edges {
                        if !visited.contains(neighbor) {
                            visited.insert(neighbor.clone());
                            next_frontier.push(neighbor.clone());
                            let name = self.entity_data.get(neighbor)
                                .map(|d| d.name.clone())
                                .unwrap_or_else(|| neighbor.clone());
                            expanded.push(ExpandedEntity {
                                id: neighbor.clone(),
                                name,
                                relation_from: node.clone(),
                                relation: relation.clone(),
                                hop: hop + 1,
                            });
                        }
                    }
                }
            }
            frontier = next_frontier;
        }

        expanded
    }

    pub fn num_entities(&self) -> usize {
        self.entity_data.len()
    }

    pub fn num_relations(&self) -> usize {
        self.edges.values().map(|e| e.len()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_entity_and_relation() {
        let dim = 8;
        let mut gm = GraphMemory::new(dim, 2);

        let v1 = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let v2 = vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

        gm.add_entity("db", &v1, Some("prod-db-01"), Some("database"));
        gm.add_entity("api", &v2, Some("api-gateway"), Some("service"));

        assert!(gm.add_relation("api", "depends_on", "db"));
        assert_eq!(gm.num_entities(), 2);
        assert_eq!(gm.num_relations(), 1);

        let relations = gm.get_relations("api");
        assert_eq!(relations.len(), 1);
        assert_eq!(relations[0].relation, "depends_on");
        assert_eq!(relations[0].target, "db");
    }

    #[test]
    fn test_search_with_expansion() {
        let dim = 8;
        let mut gm = GraphMemory::new(dim, 2);

        let v1 = vec![1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let v2 = vec![0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0];
        let v3 = vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0];

        gm.add_entity("a", &v1, Some("Entity A"), Some("type1"));
        gm.add_entity("b", &v2, Some("Entity B"), Some("type2"));
        gm.add_entity("c", &v3, Some("Entity C"), Some("type3"));

        gm.add_relation("a", "links_to", "b");
        gm.add_relation("b", "links_to", "c");

        let results = gm.search(&v1, 1, true, None);
        assert!(!results.is_empty());
        assert_eq!(results[0].id, "a");
    }
}
