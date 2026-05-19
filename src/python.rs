//! PyO3 Python bindings for bitcache.

use pyo3::prelude::*;
use pyo3::types::PyDict;
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use std::collections::HashMap;

use crate::{
    BinaryIndex as RustBinaryIndex,
    TwoStageIndex as RustTwoStageIndex,
    FloatRoutedIndex as RustFloatRoutedIndex,
    StreamingIndex as RustStreamingIndex,
    AgentMemory as RustAgentMemory,
    GraphMemory as RustGraphMemory,
};

/// Binary vector index with XOR + POPCOUNT search.
#[pyclass]
struct BinaryIndex {
    inner: RustBinaryIndex,
}

#[pymethods]
impl BinaryIndex {
    #[new]
    fn new(dim: usize) -> Self {
        Self { inner: RustBinaryIndex::new(dim) }
    }

    fn add(&mut self, vectors: PyReadonlyArray2<f32>) {
        let arr = vectors.as_array();
        let flat: Vec<f32> = arr.iter().copied().collect();
        self.inner.add(&flat);
    }

    fn search<'py>(&self, py: Python<'py>, query: PyReadonlyArray1<f32>, k: usize) -> (Bound<'py, PyArray1<u32>>, Bound<'py, PyArray1<i64>>) {
        let q: Vec<f32> = query.as_array().iter().copied().collect();
        let (dists, indices) = self.inner.search(&q, k);
        let indices_i64: Vec<i64> = indices.into_iter().map(|i| i as i64).collect();
        (
            PyArray1::from_vec_bound(py, dists),
            PyArray1::from_vec_bound(py, indices_i64),
        )
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    #[getter]
    fn memory_usage_bytes(&self) -> usize {
        self.inner.memory_usage_bytes()
    }

    #[getter]
    fn compression_ratio(&self) -> f64 {
        self.inner.compression_ratio()
    }
}

/// Two-stage retrieval: binary filter → float32 rerank.
#[pyclass]
struct TwoStageIndex {
    inner: RustTwoStageIndex,
}

#[pymethods]
impl TwoStageIndex {
    #[new]
    #[pyo3(signature = (dim, rerank_factor=10))]
    fn new(dim: usize, rerank_factor: usize) -> Self {
        Self { inner: RustTwoStageIndex::new(dim, rerank_factor) }
    }

    fn add(&mut self, vectors: PyReadonlyArray2<f32>) {
        let arr = vectors.as_array();
        let flat: Vec<f32> = arr.iter().copied().collect();
        self.inner.add(&flat);
    }

    fn search<'py>(&self, py: Python<'py>, query: PyReadonlyArray1<f32>, k: usize) -> (Bound<'py, PyArray1<f32>>, Bound<'py, PyArray1<i64>>) {
        let q: Vec<f32> = query.as_array().iter().copied().collect();
        let (scores, indices) = self.inner.search(&q, k);
        let indices_i64: Vec<i64> = indices.into_iter().map(|i| i as i64).collect();
        (
            PyArray1::from_vec_bound(py, scores),
            PyArray1::from_vec_bound(py, indices_i64),
        )
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }
}

/// Float-space routed retrieval with binary candidate filtering.
#[pyclass]
struct FloatRoutedIndex {
    inner: RustFloatRoutedIndex,
}

#[pymethods]
impl FloatRoutedIndex {
    #[new]
    #[pyo3(signature = (dim, n_partitions=128, n_probe=8, rerank_factor=100, kmeans_iter=10))]
    fn new(dim: usize, n_partitions: usize, n_probe: usize, rerank_factor: usize, kmeans_iter: usize) -> Self {
        Self { inner: RustFloatRoutedIndex::new(dim, n_partitions, n_probe, rerank_factor, kmeans_iter) }
    }

    fn build(&mut self, vectors: PyReadonlyArray2<f32>) {
        let arr = vectors.as_array();
        let flat: Vec<f32> = arr.iter().copied().collect();
        self.inner.build(&flat);
    }

    fn search<'py>(&self, py: Python<'py>, query: PyReadonlyArray1<f32>, k: usize) -> (Bound<'py, PyArray1<f32>>, Bound<'py, PyArray1<i64>>) {
        let q: Vec<f32> = query.as_array().iter().copied().collect();
        let (scores, indices) = self.inner.search(&q, k);
        let indices_i64: Vec<i64> = indices.into_iter().map(|i| i as i64).collect();
        (
            PyArray1::from_vec_bound(py, scores),
            PyArray1::from_vec_bound(py, indices_i64),
        )
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    #[getter]
    fn scan_percentage(&self) -> f64 {
        self.inner.scan_percentage()
    }
}

/// Streaming index with insert/update/delete.
#[pyclass]
struct StreamingIndex {
    inner: RustStreamingIndex,
}

#[pymethods]
impl StreamingIndex {
    #[new]
    #[pyo3(signature = (dim, rerank_factor=10))]
    fn new(dim: usize, rerank_factor: usize) -> Self {
        Self { inner: RustStreamingIndex::new(dim, rerank_factor) }
    }

    #[pyo3(signature = (vector, id=None, metadata=None))]
    fn insert(&mut self, vector: PyReadonlyArray1<f32>, id: Option<String>, metadata: Option<HashMap<String, String>>) -> String {
        let v: Vec<f32> = vector.as_array().iter().copied().collect();
        self.inner.insert(&v, id, metadata)
    }

    fn delete(&mut self, id: &str) -> bool {
        self.inner.delete(id)
    }

    fn search<'py>(&self, py: Python<'py>, query: PyReadonlyArray1<f32>, k: usize) -> PyResult<(Bound<'py, PyArray1<f32>>, Vec<String>, Vec<HashMap<String, String>>)> {
        let q: Vec<f32> = query.as_array().iter().copied().collect();
        let (scores, ids, metas) = self.inner.search(&q, k);
        Ok((PyArray1::from_vec_bound(py, scores), ids, metas))
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }
}

/// Agent memory with importance, decay, and eviction.
#[pyclass]
struct AgentMemory {
    inner: RustAgentMemory,
}

#[pymethods]
impl AgentMemory {
    #[new]
    #[pyo3(signature = (dim, capacity=10000, decay_rate=0.05, reinforce_amount=0.1, rerank_factor=10))]
    fn new(dim: usize, capacity: usize, decay_rate: f64, reinforce_amount: f64, rerank_factor: usize) -> Self {
        Self { inner: RustAgentMemory::new(dim, capacity, decay_rate, reinforce_amount, rerank_factor) }
    }

    #[pyo3(signature = (vector, content, importance=0.5, id=None, metadata=None))]
    fn save_memory(
        &mut self,
        vector: PyReadonlyArray1<f32>,
        content: &str,
        importance: f64,
        id: Option<String>,
        metadata: Option<HashMap<String, String>>,
    ) -> String {
        let v: Vec<f32> = vector.as_array().iter().copied().collect();
        self.inner.save_memory(&v, content, importance, id, metadata)
    }

    #[pyo3(signature = (query, k=5, min_importance=0.0))]
    fn retrieve_memory<'py>(&mut self, py: Python<'py>, query: PyReadonlyArray1<f32>, k: usize, min_importance: f64) -> PyResult<Vec<Bound<'py, PyDict>>> {
        let q: Vec<f32> = query.as_array().iter().copied().collect();
        let results = self.inner.retrieve_memory(&q, k, min_importance);

        let mut py_results = Vec::new();
        for r in results {
            let dict = PyDict::new_bound(py);
            dict.set_item("id", &r.id)?;
            dict.set_item("content", &r.content)?;
            dict.set_item("importance", r.importance)?;
            dict.set_item("score", r.score)?;
            dict.set_item("access_count", r.access_count)?;
            py_results.push(dict);
        }
        Ok(py_results)
    }

    fn forget_memory(&mut self, id: &str) -> bool {
        self.inner.forget_memory(id)
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }
}

/// Graph memory: entity-relation storage + vector retrieval.
#[pyclass]
struct GraphMemory {
    inner: RustGraphMemory,
}

#[pymethods]
impl GraphMemory {
    #[new]
    #[pyo3(signature = (dim, max_hops=2))]
    fn new(dim: usize, max_hops: usize) -> Self {
        Self { inner: RustGraphMemory::new(dim, max_hops) }
    }

    #[pyo3(signature = (entity_id, vector, name=None, entity_type=None))]
    fn add_entity(&mut self, entity_id: &str, vector: PyReadonlyArray1<f32>, name: Option<&str>, entity_type: Option<&str>) -> String {
        let v: Vec<f32> = vector.as_array().iter().copied().collect();
        self.inner.add_entity(entity_id, &v, name, entity_type)
    }

    fn add_relation(&mut self, source: &str, relation: &str, target: &str) -> bool {
        self.inner.add_relation(source, relation, target)
    }

    #[pyo3(signature = (query, k=5, expand=true, max_hops=None))]
    fn search<'py>(&self, py: Python<'py>, query: PyReadonlyArray1<f32>, k: usize, expand: bool, max_hops: Option<usize>) -> PyResult<Vec<Bound<'py, PyDict>>> {
        let q: Vec<f32> = query.as_array().iter().copied().collect();
        let results = self.inner.search(&q, k, expand, max_hops);

        let mut py_results = Vec::new();
        for r in results {
            let dict = PyDict::new_bound(py);
            dict.set_item("id", &r.id)?;
            dict.set_item("name", &r.name)?;
            dict.set_item("entity_type", &r.entity_type)?;
            dict.set_item("score", r.score)?;
            py_results.push(dict);
        }
        Ok(py_results)
    }

    fn remove_entity(&mut self, entity_id: &str) -> bool {
        self.inner.remove_entity(entity_id)
    }

    #[getter]
    fn num_entities(&self) -> usize {
        self.inner.num_entities()
    }

    #[getter]
    fn num_relations(&self) -> usize {
        self.inner.num_relations()
    }
}

/// Python module definition.
#[pymodule]
fn _bitcache_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<BinaryIndex>()?;
    m.add_class::<TwoStageIndex>()?;
    m.add_class::<FloatRoutedIndex>()?;
    m.add_class::<StreamingIndex>()?;
    m.add_class::<AgentMemory>()?;
    m.add_class::<GraphMemory>()?;
    Ok(())
}
