//! Bitcache: Routed binary vector retrieval engine for AI agent memory.
//!
//! A layered memory architecture combining:
//! - Binary quantization (32x compression)
//! - Staged retrieval (binary filter + float rerank)
//! - Float-space semantic routing (sublinear scan)
//! - Streaming mutations (insert/update/delete)
//! - Agent memory (importance, decay, eviction)
//! - Graph memory (entity-relation + multi-hop traversal)

pub mod quantize;
pub mod search;
pub mod index;
pub mod two_stage;
pub mod three_stage;
pub mod partitioned;
pub mod float_routed;
pub mod streaming;
pub mod memory;
pub mod graph_memory;
pub mod parallel;
pub mod forgetting;
pub mod confidence;
pub mod emotional;
pub mod temporal;

#[cfg(feature = "python")]
mod python;

// Re-exports
pub use index::BinaryIndex;
pub use two_stage::TwoStageIndex;
pub use three_stage::ThreeStageIndex;
pub use partitioned::PartitionedIndex;
pub use float_routed::FloatRoutedIndex;
pub use streaming::StreamingIndex;
pub use memory::AgentMemory;
pub use graph_memory::GraphMemory;
pub use parallel::{ParallelIndex, ConcurrentIndex};
