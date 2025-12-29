mod error;
mod filter;
mod hybrid;
mod index;
mod simd;
mod storage;
mod tantivy_index;
mod utils;
mod vectordb;

use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use pyo3::{create_exception, PyErr};
use std::sync::Arc;

use error::VectorDbError;
use filter::Filter;
use vectordb::{SearchResult, VectorDB};

// Define custom Python exceptions
create_exception!(
    vjson,
    VjsonError,
    PyException,
    "Base exception for vjson errors"
);
create_exception!(
    vjson,
    DimensionMismatchError,
    VjsonError,
    "Vector dimension does not match database dimension"
);
create_exception!(
    vjson,
    NotFoundError,
    VjsonError,
    "Vector with specified ID was not found"
);
create_exception!(
    vjson,
    StorageError,
    VjsonError,
    "Storage I/O error occurred"
);
create_exception!(
    vjson,
    InvalidParameterError,
    VjsonError,
    "Invalid parameter provided"
);

/// Convert VectorDbError to appropriate Python exception
fn to_py_err(e: VectorDbError) -> PyErr {
    match e {
        VectorDbError::DimensionMismatch { expected, actual } => DimensionMismatchError::new_err(
            format!("Dimension mismatch: expected {}, got {}", expected, actual),
        ),
        VectorDbError::NotFound(id) => NotFoundError::new_err(format!("Not found: {}", id)),
        VectorDbError::Io(e) => StorageError::new_err(format!("Storage error: {}", e)),
        VectorDbError::Json(e) => InvalidParameterError::new_err(format!("JSON error: {}", e)),
        VectorDbError::InvalidParameter(msg) => InvalidParameterError::new_err(msg),
    }
}

/// Python wrapper for VectorDB
#[pyclass]
struct PyVectorDB {
    db: Arc<VectorDB>,
}

#[pymethods]
impl PyVectorDB {
    /// Create a new vector database
    ///
    /// Args:
    ///     path: Path to store the database files
    ///     dimension: Dimension of vectors (e.g., 128, 384, 768, 1536)
    ///     max_elements: Maximum number of vectors (default: 1000000)
    ///     ef_construction: HNSW construction parameter - higher = better quality, slower build (default: 200)
    #[new]
    #[pyo3(signature = (path, dimension, max_elements=1000000, ef_construction=200))]
    fn new(
        path: String,
        dimension: usize,
        max_elements: usize,
        ef_construction: usize,
    ) -> PyResult<Self> {
        let db =
            VectorDB::new(&path, dimension, max_elements, ef_construction).map_err(to_py_err)?;

        Ok(Self { db: Arc::new(db) })
    }

    /// Insert a single vector
    ///
    /// Args:
    ///     id: Unique identifier for the vector
    ///     vector: List of floats representing the vector
    ///     metadata: Dictionary of metadata (will be stored as JSON)
    fn insert(&self, py: Python, id: String, vector: Vec<f32>, metadata: PyObject) -> PyResult<()> {
        let meta_json = python_to_json_value(py, &metadata)?;

        self.db.insert(id, vector, meta_json).map_err(to_py_err)
    }

    /// Insert multiple vectors in a batch (much faster than individual inserts)
    ///
    /// Args:
    ///     items: List of tuples (id, vector, metadata) or (id, vector, metadata, text)
    fn insert_batch(&self, py: Python, items: PyObject) -> PyResult<()> {
        // Try to parse as list
        let list = items.downcast_bound::<PyList>(py)?;

        let mut converted_items = Vec::with_capacity(list.len());
        for item in list.iter() {
            let tuple = item.downcast::<pyo3::types::PyTuple>()?;

            // Support both 3-tuple and 4-tuple formats
            if tuple.len() == 3 {
                let id: String = tuple.get_item(0)?.extract()?;
                let vector: Vec<f32> = tuple.get_item(1)?.extract()?;
                let metadata = tuple.get_item(2)?;
                let meta_json = python_to_json_value(py, &metadata.into())?;
                converted_items.push((id, vector, meta_json, None));
            } else if tuple.len() == 4 {
                let id: String = tuple.get_item(0)?.extract()?;
                let vector: Vec<f32> = tuple.get_item(1)?.extract()?;
                let metadata = tuple.get_item(2)?;
                let text: String = tuple.get_item(3)?.extract()?;
                let meta_json = python_to_json_value(py, &metadata.into())?;
                converted_items.push((id, vector, meta_json, Some(text)));
            } else {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Each item must be (id, vector, metadata) or (id, vector, metadata, text)",
                ));
            }
        }

        self.db.insert_batch(converted_items).map_err(to_py_err)
    }

    /// Insert a vector with text content for hybrid search
    ///
    /// Args:
    ///     id: Unique identifier
    ///     vector: List of floats
    ///     text: Text content for full-text search
    ///     metadata: Dictionary of metadata
    fn insert_with_text(
        &self,
        py: Python,
        id: String,
        vector: Vec<f32>,
        text: String,
        metadata: PyObject,
    ) -> PyResult<()> {
        let meta_json = python_to_json_value(py, &metadata)?;

        self.db
            .insert_with_text(id, vector, text, meta_json)
            .map_err(to_py_err)
    }

    /// Search for k nearest neighbors with optional metadata filter
    ///
    /// Args:
    ///     query: Query vector (list of floats)
    ///     k: Number of nearest neighbors to return
    ///     ef_search: Search quality parameter - higher = better recall, slower (default: 50)
    ///     filter: Optional metadata filter dict (e.g., {"category": "tech"})
    ///
    /// Returns:
    ///     List of dicts with keys: id, distance, metadata
    ///
    /// Examples:
    ///     # Simple equality filter
    ///     results = db.search(query, k=10, filter={"category": "tech"})
    ///
    ///     # Numeric comparison
    ///     results = db.search(query, k=10, filter={"score": {"$gt": 0.5}})
    ///
    ///     # Multiple conditions (AND)
    ///     results = db.search(query, k=10, filter={
    ///         "category": "tech",
    ///         "score": {"$gt": 0.5}
    ///     })
    ///
    ///     # Nested fields (dot notation)
    ///     results = db.search(query, k=10, filter={"user.age": {"$gte": 18}})
    #[pyo3(signature = (query, k, ef_search=None, filter=None))]
    fn search(
        &self,
        py: Python,
        query: Vec<f32>,
        k: usize,
        ef_search: Option<usize>,
        filter: Option<PyObject>,
    ) -> PyResult<PyObject> {
        let parsed_filter = if let Some(filter_obj) = filter {
            Some(parse_filter(py, &filter_obj)?)
        } else {
            None
        };

        let results = self
            .db
            .search_filtered(&query, k, ef_search, parsed_filter.as_ref())
            .map_err(to_py_err)?;

        search_results_to_python(py, results)
    }

    /// Batch search for multiple queries in parallel
    ///
    /// Args:
    ///     queries: List of query vectors
    ///     k: Number of nearest neighbors to return per query
    ///     ef_search: Search quality parameter (default: 50)
    ///
    /// Returns:
    ///     List of lists of search results
    #[pyo3(signature = (queries, k, ef_search=None))]
    fn batch_search(
        &self,
        py: Python,
        queries: Vec<Vec<f32>>,
        k: usize,
        ef_search: Option<usize>,
    ) -> PyResult<PyObject> {
        let results = self
            .db
            .batch_search(&queries, k, ef_search)
            .map_err(to_py_err)?;

        let py_list = PyList::empty_bound(py);
        for result_set in results {
            let py_result = search_results_to_python(py, result_set)?;
            py_list.append(py_result)?;
        }

        Ok(py_list.into())
    }

    /// Get metadata for a specific vector ID
    fn get_metadata(&self, py: Python, id: String) -> PyResult<PyObject> {
        let metadata = self.db.get_metadata(&id).map_err(to_py_err)?;

        json_value_to_python(py, &metadata.data)
    }

    /// Get the number of vectors in the database
    fn __len__(&self) -> usize {
        self.db.len()
    }

    /// Load existing data from storage and rebuild index
    fn load(&self) -> PyResult<()> {
        self.db.load().map_err(to_py_err)
    }

    /// Save all data to storage (vectors and metadata are auto-saved on insert)
    fn save(&self) -> PyResult<()> {
        self.db.save().map_err(to_py_err)
    }

    /// Clear all data from the database
    fn clear(&self) -> PyResult<()> {
        self.db.clear().map_err(to_py_err)
    }

    /// Check if database is empty
    fn is_empty(&self) -> bool {
        self.db.is_empty()
    }

    /// Update a vector and its metadata
    ///
    /// Args:
    ///     id: Vector ID to update
    ///     vector: New vector values
    ///     metadata: New metadata dictionary
    fn update(&self, py: Python, id: String, vector: Vec<f32>, metadata: PyObject) -> PyResult<()> {
        let meta_json = python_to_json_value(py, &metadata)?;

        self.db.update(id, vector, meta_json).map_err(to_py_err)
    }

    /// Update a vector with text content
    ///
    /// Args:
    ///     id: Vector ID to update
    ///     vector: New vector values
    ///     text: New text content
    ///     metadata: New metadata dictionary
    fn update_with_text(
        &self,
        py: Python,
        id: String,
        vector: Vec<f32>,
        text: String,
        metadata: PyObject,
    ) -> PyResult<()> {
        let meta_json = python_to_json_value(py, &metadata)?;

        self.db
            .update_with_text(id, vector, text, meta_json)
            .map_err(to_py_err)
    }

    /// Delete a vector by ID
    ///
    /// Args:
    ///     id: Vector ID to delete
    fn delete(&self, id: String) -> PyResult<()> {
        self.db.delete(&id).map_err(to_py_err)
    }

    /// Delete multiple vectors in a batch
    ///
    /// Args:
    ///     ids: List of vector IDs to delete
    fn delete_batch(&self, ids: Vec<String>) -> PyResult<()> {
        self.db.delete_batch(&ids).map_err(to_py_err)
    }

    /// Check if a vector ID exists
    ///
    /// Args:
    ///     id: Vector ID to check
    ///
    /// Returns:
    ///     True if the ID exists, False otherwise
    fn contains(&self, id: String) -> bool {
        self.db.contains(&id)
    }

    /// Get vector by ID
    ///
    /// Args:
    ///     id: Vector ID
    ///
    /// Returns:
    ///     Vector as a list of floats
    fn get_vector(&self, id: String) -> PyResult<Vec<f32>> {
        self.db.get_vector(&id).map_err(to_py_err)
    }

    /// Rebuild the HNSW index from current data
    /// Useful after many deletes/updates to reclaim space and optimize performance
    fn rebuild_index(&self) -> PyResult<()> {
        self.db.rebuild_index().map_err(to_py_err)
    }

    /// Get metadata for multiple IDs in batch
    ///
    /// Args:
    ///     ids: List of vector IDs
    ///
    /// Returns:
    ///     List of dicts with 'id' and 'metadata' keys (only for found IDs)
    fn get_metadata_batch(&self, py: Python, ids: Vec<String>) -> PyResult<PyObject> {
        let results = self.db.get_metadata_batch(&ids);

        let py_list = PyList::empty_bound(py);
        for (id, metadata) in results {
            let dict = PyDict::new_bound(py);
            dict.set_item("id", id)?;
            let meta_obj = json_value_to_python(py, &metadata.data)?;
            dict.set_item("metadata", meta_obj)?;
            py_list.append(dict)?;
        }

        Ok(py_list.into())
    }

    /// Get vectors for multiple IDs in batch
    ///
    /// Args:
    ///     ids: List of vector IDs
    ///
    /// Returns:
    ///     List of dicts with 'id' and 'vector' keys (only for found IDs)
    fn get_vectors_batch(&self, py: Python, ids: Vec<String>) -> PyResult<PyObject> {
        let results = self.db.get_vectors_batch(&ids).map_err(to_py_err)?;

        let py_list = PyList::empty_bound(py);
        for (id, vector) in results {
            let dict = PyDict::new_bound(py);
            dict.set_item("id", id)?;
            dict.set_item("vector", vector)?;
            py_list.append(dict)?;
        }

        Ok(py_list.into())
    }

    /// Range search - find all vectors within a distance threshold
    ///
    /// Args:
    ///     query: Query vector
    ///     max_distance: Maximum distance threshold
    ///     ef_search: Search quality parameter
    ///
    /// Returns:
    ///     List of search results within the distance threshold
    #[pyo3(signature = (query, max_distance, ef_search=None))]
    fn range_search(
        &self,
        py: Python,
        query: Vec<f32>,
        max_distance: f32,
        ef_search: Option<usize>,
    ) -> PyResult<PyObject> {
        let results = self
            .db
            .range_search(&query, max_distance, ef_search)
            .map_err(to_py_err)?;

        search_results_to_python(py, results)
    }

    /// Get database statistics
    ///
    /// Returns:
    ///     Dict with database statistics (total_vectors, dimension, metadata_keys, etc.)
    fn get_stats(&self, py: Python) -> PyResult<PyObject> {
        let stats = self.db.get_stats();

        let dict = PyDict::new_bound(py);
        dict.set_item("total_vectors", stats.total_vectors)?;
        dict.set_item("dimension", stats.dimension)?;
        dict.set_item("metadata_keys", stats.metadata_keys)?;
        dict.set_item("active_vectors", stats.active_vectors)?;
        dict.set_item("index_size", stats.index_size)?;

        Ok(dict.into())
    }

    /// Update only metadata (fast, doesn't touch vector)
    ///
    /// Args:
    ///     id: Vector ID to update
    ///     metadata: New metadata dictionary
    fn update_metadata(&self, py: Python, id: String, metadata: PyObject) -> PyResult<()> {
        let meta_json = python_to_json_value(py, &metadata)?;

        self.db.update_metadata(id, meta_json).map_err(to_py_err)
    }

    /// Full-text search using Tantivy
    ///
    /// Args:
    ///     query: Text query string
    ///     limit: Maximum number of results
    ///
    /// Returns:
    ///     List of tuples (id, score)
    fn text_search(&self, py: Python, query: String, limit: usize) -> PyResult<PyObject> {
        let results = self.db.text_search(&query, limit).map_err(to_py_err)?;

        let py_list = PyList::empty_bound(py);
        for (id, score) in results {
            let tuple = pyo3::types::PyTuple::new_bound(py, &[id.into_py(py), score.into_py(py)]);
            py_list.append(tuple)?;
        }

        Ok(py_list.into())
    }

    /// Hybrid search combining vector similarity and full-text search
    ///
    /// Args:
    ///     query_vector: Query vector
    ///     query_text: Text query string
    ///     k: Number of results to return
    ///     ef_search: HNSW search quality (default: 50)
    ///     strategy: Fusion strategy - "rrf", "weighted", "max", "min", "average" (default: "rrf")
    ///     vector_weight: Weight for vector scores (for "weighted" strategy, default: 0.5)
    ///     text_weight: Weight for text scores (for "weighted" strategy, default: 0.5)
    ///
    /// Returns:
    ///     List of dicts with keys: id, vector_score, text_score, combined_score
    #[pyo3(signature = (query_vector, query_text, k, ef_search=None, strategy="rrf", vector_weight=0.5, text_weight=0.5))]
    fn hybrid_search(
        &self,
        py: Python,
        query_vector: Vec<f32>,
        query_text: String,
        k: usize,
        ef_search: Option<usize>,
        strategy: &str,
        vector_weight: f32,
        text_weight: f32,
    ) -> PyResult<PyObject> {
        use crate::hybrid::FusionStrategy;

        // Parse strategy
        let fusion_strategy = match strategy {
            "rrf" => FusionStrategy::ReciprocalRankFusion { k: 60.0 },
            "weighted" => FusionStrategy::WeightedSum {
                vector_weight,
                text_weight,
            },
            "max" => FusionStrategy::Max,
            "min" => FusionStrategy::Min,
            "average" => FusionStrategy::Average,
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Unknown strategy: {}. Use 'rrf', 'weighted', 'max', 'min', or 'average'",
                    strategy
                )))
            },
        };

        let results = self
            .db
            .hybrid_search(&query_vector, &query_text, k, ef_search, fusion_strategy)
            .map_err(to_py_err)?;

        let py_list = PyList::empty_bound(py);
        for result in results {
            let dict = PyDict::new_bound(py);
            dict.set_item("id", result.id)?;
            dict.set_item("vector_score", result.vector_score)?;
            dict.set_item("text_score", result.text_score)?;
            dict.set_item("combined_score", result.combined_score)?;
            py_list.append(dict)?;
        }

        Ok(py_list.into())
    }

    /// String representation
    fn __repr__(&self) -> String {
        format!("VectorDB(size={})", self.db.len())
    }
}

// Helper functions for Python/Rust conversions

fn parse_filter(py: Python, filter_dict: &PyObject) -> PyResult<Filter> {
    // Convert Python dict to JSON
    let json_val = python_to_json_value(py, filter_dict)?;

    // Parse simple filter format: {"key": "value"} or {"key": {"$gt": 5}}
    if let serde_json::Value::Object(map) = json_val {
        let mut filters = Vec::new();

        for (key, value) in map {
            if key == "$and" {
                if let serde_json::Value::Array(arr) = value {
                    let sub_filters: Result<Vec<_>, _> = arr
                        .iter()
                        .map(|v| {
                            let py_obj = json_value_to_python(py, v)?;
                            parse_filter(py, &py_obj)
                        })
                        .collect();
                    return Ok(Filter::And(sub_filters?));
                }
            } else if key == "$or" {
                if let serde_json::Value::Array(arr) = value {
                    let sub_filters: Result<Vec<_>, _> = arr
                        .iter()
                        .map(|v| {
                            let py_obj = json_value_to_python(py, v)?;
                            parse_filter(py, &py_obj)
                        })
                        .collect();
                    return Ok(Filter::Or(sub_filters?));
                }
            } else if let serde_json::Value::Object(op_map) = &value {
                // Operator syntax: {"key": {"$gt": 5}}
                for (op, op_val) in op_map {
                    let filter = match op.as_str() {
                        "$gt" => Filter::GreaterThan {
                            key: key.clone(),
                            value: op_val.as_f64().ok_or_else(|| {
                                PyErr::new::<pyo3::exceptions::PyValueError, _>(
                                    "$gt requires number",
                                )
                            })?,
                        },
                        "$lt" => Filter::LessThan {
                            key: key.clone(),
                            value: op_val.as_f64().ok_or_else(|| {
                                PyErr::new::<pyo3::exceptions::PyValueError, _>(
                                    "$lt requires number",
                                )
                            })?,
                        },
                        "$gte" => Filter::GreaterOrEqual {
                            key: key.clone(),
                            value: op_val.as_f64().ok_or_else(|| {
                                PyErr::new::<pyo3::exceptions::PyValueError, _>(
                                    "$gte requires number",
                                )
                            })?,
                        },
                        "$lte" => Filter::LessOrEqual {
                            key: key.clone(),
                            value: op_val.as_f64().ok_or_else(|| {
                                PyErr::new::<pyo3::exceptions::PyValueError, _>(
                                    "$lte requires number",
                                )
                            })?,
                        },
                        "$in" => Filter::In {
                            key: key.clone(),
                            values: op_val
                                .as_array()
                                .ok_or_else(|| {
                                    PyErr::new::<pyo3::exceptions::PyValueError, _>(
                                        "$in requires array",
                                    )
                                })?
                                .clone(),
                        },
                        "$nin" => Filter::NotIn {
                            key: key.clone(),
                            values: op_val
                                .as_array()
                                .ok_or_else(|| {
                                    PyErr::new::<pyo3::exceptions::PyValueError, _>(
                                        "$nin requires array",
                                    )
                                })?
                                .clone(),
                        },
                        "$exists" => Filter::Exists {
                            key: key.clone(),
                            exists: op_val.as_bool().ok_or_else(|| {
                                PyErr::new::<pyo3::exceptions::PyValueError, _>(
                                    "$exists requires boolean",
                                )
                            })?,
                        },
                        "$ne" => Filter::NotEquals {
                            key: key.clone(),
                            value: op_val.clone(),
                        },
                        "$between" => {
                            let arr = op_val.as_array().ok_or_else(|| {
                                PyErr::new::<pyo3::exceptions::PyValueError, _>(
                                    "$between requires array of [min, max]",
                                )
                            })?;
                            if arr.len() != 2 {
                                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                                    "$between requires exactly 2 values",
                                ));
                            }
                            Filter::Between {
                                key: key.clone(),
                                min: arr[0].as_f64().ok_or_else(|| {
                                    PyErr::new::<pyo3::exceptions::PyValueError, _>(
                                        "$between min must be number",
                                    )
                                })?,
                                max: arr[1].as_f64().ok_or_else(|| {
                                    PyErr::new::<pyo3::exceptions::PyValueError, _>(
                                        "$between max must be number",
                                    )
                                })?,
                            }
                        },
                        "$startsWith" => Filter::StartsWith {
                            key: key.clone(),
                            prefix: op_val
                                .as_str()
                                .ok_or_else(|| {
                                    PyErr::new::<pyo3::exceptions::PyValueError, _>(
                                        "$startsWith requires string",
                                    )
                                })?
                                .to_string(),
                        },
                        "$endsWith" => Filter::EndsWith {
                            key: key.clone(),
                            suffix: op_val
                                .as_str()
                                .ok_or_else(|| {
                                    PyErr::new::<pyo3::exceptions::PyValueError, _>(
                                        "$endsWith requires string",
                                    )
                                })?
                                .to_string(),
                        },
                        "$contains" => Filter::Contains {
                            key: key.clone(),
                            substring: op_val
                                .as_str()
                                .ok_or_else(|| {
                                    PyErr::new::<pyo3::exceptions::PyValueError, _>(
                                        "$contains requires string",
                                    )
                                })?
                                .to_string(),
                        },
                        "$regex" => Filter::Regex {
                            key: key.clone(),
                            pattern: op_val
                                .as_str()
                                .ok_or_else(|| {
                                    PyErr::new::<pyo3::exceptions::PyValueError, _>(
                                        "$regex requires string pattern",
                                    )
                                })?
                                .to_string(),
                        },
                        _ => {
                            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                                "Unknown operator: {}",
                                op
                            )))
                        },
                    };
                    filters.push(filter);
                }
            } else {
                // Simple equality: {"key": "value"}
                filters.push(Filter::Equals {
                    key: key.clone(),
                    value: value.clone(),
                });
            }
        }

        if filters.len() == 1 {
            Ok(filters.into_iter().next().unwrap())
        } else {
            Ok(Filter::And(filters))
        }
    } else {
        Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Filter must be a dictionary",
        ))
    }
}

fn python_to_json_value(py: Python, obj: &PyObject) -> PyResult<serde_json::Value> {
    let json_module = py.import_bound("json")?;
    let dumps_fn = json_module.getattr("dumps")?;
    let json_string: String = dumps_fn.call1((obj,))?.extract()?;

    serde_json::from_str(&json_string).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Invalid JSON: {}", e))
    })
}

fn json_value_to_python(py: Python, json: &serde_json::Value) -> PyResult<PyObject> {
    let json_string = serde_json::to_string(json)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))?;

    let json_module = py.import_bound("json")?;
    let loads_fn = json_module.getattr("loads")?;
    let result = loads_fn.call1((json_string,))?;

    Ok(result.into())
}

fn search_results_to_python(py: Python, results: Vec<SearchResult>) -> PyResult<PyObject> {
    let py_list = PyList::empty_bound(py);

    for result in results {
        let dict = PyDict::new_bound(py);
        dict.set_item("id", result.id)?;
        dict.set_item("distance", result.distance)?;

        let metadata_obj = json_value_to_python(py, &result.metadata)?;
        dict.set_item("metadata", metadata_obj)?;

        py_list.append(dict)?;
    }

    Ok(py_list.into())
}

/// Normalize a vector to unit length (for cosine similarity)
#[pyfunction]
fn normalize_vector(vector: Vec<f32>) -> Vec<f32> {
    utils::normalize_vector(&vector)
}

/// Normalize multiple vectors in batch
#[pyfunction]
fn normalize_vectors(vectors: Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    utils::normalize_vectors(&vectors)
}

/// Compute cosine similarity between two vectors
#[pyfunction]
fn cosine_similarity(a: Vec<f32>, b: Vec<f32>) -> f32 {
    utils::cosine_similarity(&a, &b)
}

/// Compute dot product between two vectors
#[pyfunction]
fn dot_product(a: Vec<f32>, b: Vec<f32>) -> f32 {
    utils::dot_product(&a, &b)
}

/// A high-performance vector database built with Rust
///
/// Features:
/// - Standard JSON for metadata (simple and human-readable)
/// - HNSW algorithm for approximate nearest neighbor search
/// - Thread-safe with parallel reads and locked writes
/// - Memory-mapped file I/O for large datasets
#[pymodule]
fn vjson(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Add main class
    m.add_class::<PyVectorDB>()?;

    // Add utility functions
    m.add_function(wrap_pyfunction!(normalize_vector, m)?)?;
    m.add_function(wrap_pyfunction!(normalize_vectors, m)?)?;
    m.add_function(wrap_pyfunction!(cosine_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(dot_product, m)?)?;

    // Add custom exceptions
    m.add("VjsonError", m.py().get_type_bound::<VjsonError>())?;
    m.add(
        "DimensionMismatchError",
        m.py().get_type_bound::<DimensionMismatchError>(),
    )?;
    m.add("NotFoundError", m.py().get_type_bound::<NotFoundError>())?;
    m.add("StorageError", m.py().get_type_bound::<StorageError>())?;
    m.add(
        "InvalidParameterError",
        m.py().get_type_bound::<InvalidParameterError>(),
    )?;

    // Add version
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    Ok(())
}
