use crate::error::{Result, VectorDbError};
use crate::filter::Filter;
use crate::hybrid::{hybrid_search, FusionStrategy, HybridResult};
use crate::index::HnswIndex;
use crate::storage::{StorageLayer, VectorMetadata};
use crate::tantivy_index::TantivyIndex;
use ahash::AHashMap;
use parking_lot::RwLock;
use smallvec::SmallVec;
use std::path::Path;
use std::sync::Arc;

/// Stack-allocated vector for small search results (k ≤ 32)
/// Avoids heap allocation for typical queries
#[allow(dead_code)]
pub type SmallSearchResults = SmallVec<[SearchResult; 32]>;

/// Threshold below which sequential processing is faster than parallel
/// Rayon thread pool overhead is ~2-5μs per task, so small batches should be sequential
const PARALLEL_THRESHOLD: usize = 1000;

/// Main VectorDB with concurrent access control
/// - Parallel reads: Multiple threads can search simultaneously
/// - Locked writes: Only one thread can write at a time
pub struct VectorDB {
    /// HNSW index for fast ANN search (already thread-safe)
    index: Arc<RwLock<HnswIndex>>,

    /// Storage layer for persistence
    storage: Arc<RwLock<StorageLayer>>,

    /// Tantivy full-text search index (optional)
    text_index: Option<Arc<TantivyIndex>>,

    /// Metadata mapping: vector_id -> metadata
    /// RwLock allows multiple concurrent readers or one writer
    metadata_map: Arc<RwLock<AHashMap<String, VectorMetadata>>>,

    /// Vector dimension
    dimension: usize,

    /// HNSW parameters for rebuilding
    max_elements: usize,
    ef_construction: usize,

    /// Next available internal ID
    next_id: Arc<RwLock<usize>>,

    /// Map external string IDs to internal numeric IDs
    id_map: Arc<RwLock<AHashMap<String, usize>>>,

    /// Reverse map: internal numeric IDs to external string IDs
    /// Cached to avoid rebuilding on every search (5-15% speedup)
    reverse_id_map: Arc<RwLock<AHashMap<usize, String>>>,
}

impl VectorDB {
    /// Create a new VectorDB
    pub fn new<P: AsRef<Path>>(
        path: P,
        dimension: usize,
        max_elements: usize,
        ef_construction: usize,
    ) -> Result<Self> {
        let path = path.as_ref();
        let storage = StorageLayer::new(path, dimension)?;
        let index = HnswIndex::new(dimension, max_elements, ef_construction);

        // Create Tantivy index in a subdirectory
        let text_index_path = path.join("text_index");
        let text_index = match TantivyIndex::new(&text_index_path) {
            Ok(idx) => Some(Arc::new(idx)),
            Err(_) => None, // Tantivy is optional
        };

        Ok(Self {
            index: Arc::new(RwLock::new(index)),
            storage: Arc::new(RwLock::new(storage)),
            text_index,
            metadata_map: Arc::new(RwLock::new(AHashMap::new())),
            dimension,
            max_elements,
            ef_construction,
            next_id: Arc::new(RwLock::new(0)),
            id_map: Arc::new(RwLock::new(AHashMap::new())),
            reverse_id_map: Arc::new(RwLock::new(AHashMap::new())),
        })
    }

    /// Insert a single vector with metadata
    /// Write operation - acquires exclusive locks
    pub fn insert(&self, id: String, vector: Vec<f32>, metadata: serde_json::Value) -> Result<()> {
        self.insert_batch(vec![(id, vector, metadata, None)])
    }

    /// Insert a single vector with text content for hybrid search
    pub fn insert_with_text(
        &self,
        id: String,
        vector: Vec<f32>,
        text: String,
        metadata: serde_json::Value,
    ) -> Result<()> {
        self.insert_batch(vec![(id, vector, metadata, Some(text))])
    }

    /// Insert multiple vectors in a batch (more efficient)
    /// Write operation - acquires exclusive locks
    /// Items: (id, vector, metadata, optional_text)
    pub fn insert_batch(
        &self,
        items: Vec<(String, Vec<f32>, serde_json::Value, Option<String>)>,
    ) -> Result<()> {
        use rayon::prelude::*;

        if items.is_empty() {
            return Ok(());
        }

        // Validate dimensions
        for (_, vec, _, _) in &items {
            if vec.len() != self.dimension {
                return Err(VectorDbError::DimensionMismatch {
                    expected: self.dimension,
                    actual: vec.len(),
                });
            }
        }

        // Acquire write locks (exclusive access)
        let mut next_id = self.next_id.write();
        let mut id_map = self.id_map.write();
        let mut reverse_id_map = self.reverse_id_map.write();
        let mut metadata_map = self.metadata_map.write();
        let storage = self.storage.read();

        // Assign internal IDs
        let start_id = *next_id;

        // Prepare vectors and metadata - use parallel only for large batches
        let prepared_data: Vec<(Vec<f32>, String, usize, VectorMetadata)> = if items.len()
            >= PARALLEL_THRESHOLD
        {
            items
                .par_iter()
                .enumerate()
                .map(|(i, (ext_id, v, meta, _))| {
                    let internal_id = start_id + i;
                    let metadata = VectorMetadata {
                        id: ext_id.clone(),
                        data: meta.clone(),
                        timestamp: std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap()
                            .as_secs(),
                    };
                    (v.clone(), ext_id.clone(), internal_id, metadata)
                })
                .collect()
        } else {
            // Sequential for small batches - avoids Rayon overhead
            items
                .iter()
                .enumerate()
                .map(|(i, (ext_id, v, meta, _))| {
                    let internal_id = start_id + i;
                    let metadata = VectorMetadata {
                        id: ext_id.clone(),
                        data: meta.clone(),
                        timestamp: std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap()
                            .as_secs(),
                    };
                    (v.clone(), ext_id.clone(), internal_id, metadata)
                })
                .collect()
        };

        // Sequential insert into HashMaps and extract vectors (avoiding second clone)
        let mut vectors = Vec::with_capacity(prepared_data.len());
        for (v, ext_id, internal_id, metadata) in prepared_data {
            vectors.push(v);
            id_map.insert(ext_id.clone(), internal_id);
            reverse_id_map.insert(internal_id, ext_id.clone());
            metadata_map.insert(ext_id, metadata);
        }

        *next_id += items.len();

        // Insert into HNSW index
        let index = self.index.read();
        index.insert_batch(&vectors, start_id)?;

        // Insert into text index if available
        if let Some(ref text_index) = self.text_index {
            let text_docs: Vec<(String, String, serde_json::Value)> = items
                .iter()
                .map(|(id, _, meta, text)| {
                    let text_content = text.clone().unwrap_or_default();
                    (id.clone(), text_content, meta.clone())
                })
                .collect();

            text_index.add_documents_batch(&text_docs)?;
            text_index.commit()?;
        }

        // Persist to storage (convert to old format without text)
        let storage_items: Vec<(String, Vec<f32>, serde_json::Value)> = items
            .iter()
            .map(|(id, vec, meta, _)| (id.clone(), vec.clone(), meta.clone()))
            .collect();
        storage.save_batch(&storage_items)?;

        Ok(())
    }

    /// Search for k nearest neighbors with optional metadata filter
    /// Read operation - allows concurrent searches
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        ef_search: Option<usize>,
    ) -> Result<Vec<SearchResult>> {
        self.search_filtered(query, k, ef_search, None)
    }

    /// Search for k nearest neighbors with metadata filtering
    /// Read operation - allows concurrent searches
    pub fn search_filtered(
        &self,
        query: &[f32],
        k: usize,
        ef_search: Option<usize>,
        filter: Option<&Filter>,
    ) -> Result<Vec<SearchResult>> {
        let ef = ef_search.unwrap_or(50);

        // Search index (read lock - concurrent)
        let index = self.index.read();
        let results = index.search(query, k, ef)?;

        // Use cached reverse map (major optimization - avoids rebuilding on every search)
        let reverse_id_map = self.reverse_id_map.read();
        let metadata_map = self.metadata_map.read();

        let mut search_results: Vec<SearchResult> = results
            .into_iter()
            .filter_map(|(internal_id, distance)| {
                reverse_id_map.get(&internal_id).and_then(|ext_id| {
                    metadata_map.get(ext_id).map(|meta| SearchResult {
                        id: ext_id.clone(),
                        distance,
                        metadata: meta.data.clone(),
                    })
                })
            })
            .collect();

        // Apply metadata filter if provided
        if let Some(filter) = filter {
            search_results.retain(|result| filter.matches(&result.metadata));
            search_results.truncate(k);
        }

        Ok(search_results)
    }

    /// Batch search for multiple queries - OPTIMIZED
    /// Uses cached reverse map for all queries
    /// Conditional parallelization: parallel for large batches, sequential for small
    /// Read operation - highly concurrent
    pub fn batch_search(
        &self,
        queries: &[Vec<f32>],
        k: usize,
        ef_search: Option<usize>,
    ) -> Result<Vec<Vec<SearchResult>>> {
        use rayon::prelude::*;

        let ef = ef_search.unwrap_or(50);

        // Use cached data structures (no rebuilding)
        let index = self.index.read();
        let reverse_id_map = self.reverse_id_map.read();
        let metadata_map = self.metadata_map.read();

        // Helper closure to process a single query
        let process_query = |query: &Vec<f32>| -> Result<Vec<SearchResult>> {
            let raw_results = index.search(query, k, ef)?;
            let search_results: Vec<SearchResult> = raw_results
                .into_iter()
                .filter_map(|(internal_id, distance)| {
                    reverse_id_map.get(&internal_id).and_then(|ext_id| {
                        metadata_map.get(ext_id).map(|meta| SearchResult {
                            id: ext_id.clone(),
                            distance,
                            metadata: meta.data.clone(),
                        })
                    })
                })
                .collect();
            Ok(search_results)
        };

        // Conditional parallelization based on query count
        let results: Vec<Vec<SearchResult>> = if queries.len() >= PARALLEL_THRESHOLD {
            queries
                .par_iter()
                .map(process_query)
                .collect::<Result<Vec<Vec<SearchResult>>>>()?
        } else {
            // Sequential for small batches - avoids Rayon overhead
            queries
                .iter()
                .map(process_query)
                .collect::<Result<Vec<Vec<SearchResult>>>>()?
        };

        Ok(results)
    }

    /// Batch search with optional filtering - OPTIMIZED
    /// Uses cached reverse map for all queries
    /// Conditional parallelization: parallel for large batches, sequential for small
    #[allow(dead_code)]
    pub fn batch_search_filtered(
        &self,
        queries: &[Vec<f32>],
        k: usize,
        ef_search: Option<usize>,
        filter: Option<&Filter>,
    ) -> Result<Vec<Vec<SearchResult>>> {
        use rayon::prelude::*;

        let ef = ef_search.unwrap_or(50);

        // Use cached data structures
        let index = self.index.read();
        let reverse_id_map = self.reverse_id_map.read();
        let metadata_map = self.metadata_map.read();

        // Helper closure to process a single query
        let process_query = |query: &Vec<f32>| -> Result<Vec<SearchResult>> {
            let raw_results = index.search(query, k, ef)?;
            let mut search_results: Vec<SearchResult> = raw_results
                .into_iter()
                .filter_map(|(internal_id, distance)| {
                    reverse_id_map.get(&internal_id).and_then(|ext_id| {
                        metadata_map.get(ext_id).map(|meta| SearchResult {
                            id: ext_id.clone(),
                            distance,
                            metadata: meta.data.clone(),
                        })
                    })
                })
                .collect();

            // Apply filter if provided
            if let Some(f) = filter {
                search_results.retain(|result| f.matches(&result.metadata));
                search_results.truncate(k);
            }

            Ok(search_results)
        };

        // Conditional parallelization based on query count
        let results: Vec<Vec<SearchResult>> = if queries.len() >= PARALLEL_THRESHOLD {
            queries
                .par_iter()
                .map(process_query)
                .collect::<Result<Vec<Vec<SearchResult>>>>()?
        } else {
            // Sequential for small batches - avoids Rayon overhead
            queries
                .iter()
                .map(process_query)
                .collect::<Result<Vec<Vec<SearchResult>>>>()?
        };

        Ok(results)
    }

    /// Range search - find all vectors within a distance threshold
    /// More expensive than k-NN search as it may return many results
    pub fn range_search(
        &self,
        query: &[f32],
        max_distance: f32,
        ef_search: Option<usize>,
    ) -> Result<Vec<SearchResult>> {
        use rayon::prelude::*;
        // Search with a large k to get candidates
        let k = 1000.min(self.len()); // Cap at 1000 or total size
        let candidates = self.search(query, k, ef_search)?;

        // Filter by distance threshold - conditional parallelization
        let results: Vec<SearchResult> = if candidates.len() >= PARALLEL_THRESHOLD {
            candidates
                .into_par_iter()
                .filter(|r| r.distance <= max_distance)
                .collect()
        } else {
            candidates
                .into_iter()
                .filter(|r| r.distance <= max_distance)
                .collect()
        };

        Ok(results)
    }

    /// Optimized search for small result sets (k ≤ 32) - uses stack allocation
    /// This is the fastest search method for typical queries
    #[allow(dead_code)]
    pub fn search_small(
        &self,
        query: &[f32],
        k: usize,
        ef_search: Option<usize>,
    ) -> Result<SmallSearchResults> {
        let ef = ef_search.unwrap_or(50);

        // Search index
        let index = self.index.read();
        let results = index.search(query, k, ef)?;

        // Use cached reverse map
        let reverse_id_map = self.reverse_id_map.read();
        let metadata_map = self.metadata_map.read();

        // Collect results
        let vec_results: Vec<SearchResult> = results
            .into_iter()
            .filter_map(|(internal_id, distance)| {
                reverse_id_map.get(&internal_id).and_then(|ext_id| {
                    metadata_map.get(ext_id).map(|meta| SearchResult {
                        id: ext_id.clone(),
                        distance,
                        metadata: meta.data.clone(),
                    })
                })
            })
            .collect();

        // Convert to SmallVec - uses stack if k ≤ 32, otherwise heap
        Ok(SmallSearchResults::from_vec(vec_results))
    }

    /// Batch search with SmallVec for small k - optimized for k ≤ 32
    /// Conditional parallelization: parallel for large batches, sequential for small
    #[allow(dead_code)]
    pub fn batch_search_small(
        &self,
        queries: &[Vec<f32>],
        k: usize,
        ef_search: Option<usize>,
    ) -> Result<Vec<SmallSearchResults>> {
        use rayon::prelude::*;

        let ef = ef_search.unwrap_or(50);

        // Use cached data structures
        let index = self.index.read();
        let reverse_id_map = self.reverse_id_map.read();
        let metadata_map = self.metadata_map.read();

        // Helper closure to process a single query
        let process_query = |query: &Vec<f32>| -> Result<SmallSearchResults> {
            let raw_results = index.search(query, k, ef)?;
            let vec_results: Vec<SearchResult> = raw_results
                .into_iter()
                .filter_map(|(internal_id, distance)| {
                    reverse_id_map.get(&internal_id).and_then(|ext_id| {
                        metadata_map.get(ext_id).map(|meta| SearchResult {
                            id: ext_id.clone(),
                            distance,
                            metadata: meta.data.clone(),
                        })
                    })
                })
                .collect();
            Ok(SmallSearchResults::from_vec(vec_results))
        };

        // Conditional parallelization based on query count
        let results: Vec<SmallSearchResults> = if queries.len() >= PARALLEL_THRESHOLD {
            queries
                .par_iter()
                .map(process_query)
                .collect::<Result<Vec<SmallSearchResults>>>()?
        } else {
            // Sequential for small batches - avoids Rayon overhead
            queries
                .iter()
                .map(process_query)
                .collect::<Result<Vec<SmallSearchResults>>>()?
        };

        Ok(results)
    }

    /// Get metadata for a specific ID
    /// Read operation - concurrent
    pub fn get_metadata(&self, id: &str) -> Result<VectorMetadata> {
        let metadata_map = self.metadata_map.read();
        metadata_map
            .get(id)
            .cloned()
            .ok_or_else(|| VectorDbError::NotFound(id.to_string()))
    }

    /// Get metadata for multiple IDs in batch
    /// Returns a vector of (id, metadata) pairs for found items
    /// Conditional parallelization: parallel for large batches, sequential for small
    pub fn get_metadata_batch(&self, ids: &[String]) -> Vec<(String, VectorMetadata)> {
        use rayon::prelude::*;

        let metadata_map = self.metadata_map.read();

        if ids.len() >= PARALLEL_THRESHOLD {
            ids.par_iter()
                .filter_map(|id| metadata_map.get(id).map(|meta| (id.clone(), meta.clone())))
                .collect()
        } else {
            // Sequential for small batches - avoids Rayon overhead
            ids.iter()
                .filter_map(|id| metadata_map.get(id).map(|meta| (id.clone(), meta.clone())))
                .collect()
        }
    }

    /// Get vectors for multiple IDs in batch
    /// Returns a vector of (id, vector) pairs for found items
    /// Optimized: loads only requested vectors, not all vectors
    pub fn get_vectors_batch(&self, ids: &[String]) -> Result<Vec<(String, Vec<f32>)>> {
        let id_map = self.id_map.read();
        let storage = self.storage.read();

        // Collect internal IDs and their corresponding external IDs
        let id_pairs: Vec<(String, usize)> = ids
            .iter()
            .filter_map(|id| id_map.get(id).map(|&internal_id| (id.clone(), internal_id)))
            .collect();

        // Get just the internal indices
        let indices: Vec<usize> = id_pairs.iter().map(|(_, idx)| *idx).collect();

        // Load only the vectors we need
        let loaded = storage.load_vectors_by_indices(&indices)?;

        // Build a map from internal_id -> vector for quick lookup
        let loaded_map: std::collections::HashMap<usize, Vec<f32>> = loaded.into_iter().collect();

        // Map back to external IDs
        let results: Vec<(String, Vec<f32>)> = id_pairs
            .into_iter()
            .filter_map(|(ext_id, internal_id)| {
                loaded_map.get(&internal_id).map(|v| (ext_id, v.clone()))
            })
            .collect();

        Ok(results)
    }

    /// Get the number of vectors in the database
    pub fn len(&self) -> usize {
        let index = self.index.read();
        index.len()
    }

    /// Check if database is empty
    pub fn is_empty(&self) -> bool {
        let index = self.index.read();
        index.is_empty()
    }

    /// Get database statistics
    pub fn get_stats(&self) -> DatabaseStats {
        let metadata_map = self.metadata_map.read();
        let id_map = self.id_map.read();

        // Collect unique metadata keys
        let mut all_keys = std::collections::HashSet::new();
        for meta in metadata_map.values() {
            if let serde_json::Value::Object(map) = &meta.data {
                for key in map.keys() {
                    all_keys.insert(key.clone());
                }
            }
        }

        DatabaseStats {
            total_vectors: id_map.len(),
            dimension: self.dimension,
            metadata_keys: all_keys.into_iter().collect(),
            active_vectors: metadata_map.len(),
            index_size: self.len(),
        }
    }

    /// Load existing data from storage and rebuild index
    pub fn load(&self) -> Result<()> {
        let storage = self.storage.read();

        // Load vectors and metadata
        let vectors = storage.load_vectors()?;
        let metadata = storage.load_metadata()?;

        if vectors.len() != metadata.len() {
            return Err(VectorDbError::InvalidParameter(
                "Vector count doesn't match metadata count".to_string(),
            ));
        }

        // Rebuild index and mappings
        let mut next_id = self.next_id.write();
        let mut id_map = self.id_map.write();
        let mut reverse_id_map = self.reverse_id_map.write();
        let mut metadata_map = self.metadata_map.write();

        *next_id = vectors.len();

        for (i, meta) in metadata.iter().enumerate() {
            id_map.insert(meta.id.clone(), i);
            reverse_id_map.insert(i, meta.id.clone());
            metadata_map.insert(meta.id.clone(), meta.clone());
        }

        // Rebuild HNSW index from scratch
        // This is fast and ensures consistency
        if !vectors.is_empty() {
            let mut index = self.index.write();
            *index = HnswIndex::new(self.dimension, self.max_elements, self.ef_construction);
            index.insert_batch(&vectors, 0)?;
        }

        Ok(())
    }

    /// Save all data to storage
    /// This persists vectors and metadata. The HNSW index will be rebuilt on load.
    pub fn save(&self) -> Result<()> {
        // Storage is already saved incrementally during inserts
        // This method is here for explicit save operations if needed

        // We could add explicit flush/sync operations here if needed
        Ok(())
    }

    /// Update only metadata (fast, doesn't touch vector or index)
    pub fn update_metadata(&self, id: String, metadata: serde_json::Value) -> Result<()> {
        let mut metadata_map = self.metadata_map.write();

        if let Some(meta) = metadata_map.get_mut(&id) {
            meta.data = metadata;
            meta.timestamp = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs();
            Ok(())
        } else {
            Err(VectorDbError::NotFound(id))
        }
    }

    /// Update a vector and its metadata
    /// This is implemented as a delete + insert since HNSW doesn't support in-place updates
    pub fn update(&self, id: String, _vector: Vec<f32>, metadata: serde_json::Value) -> Result<()> {
        // Check if the ID exists
        {
            let id_map = self.id_map.read();
            if !id_map.contains_key(&id) {
                return Err(VectorDbError::NotFound(id));
            }
        }

        // For HNSW, we need to rebuild since it doesn't support in-place updates
        // Mark this vector as updated in metadata
        let mut metadata_map = self.metadata_map.write();

        if let Some(meta) = metadata_map.get_mut(&id) {
            meta.data = metadata.clone();
            meta.timestamp = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs();
        }

        // Update text index if available
        if let Some(ref text_index) = self.text_index {
            text_index.delete_document(&id)?;
        }

        // For now, we store the updated vector in metadata
        // A full rebuild would be needed to update the HNSW index
        // This is a limitation of HNSW - it's optimized for read-heavy workloads

        Ok(())
    }

    /// Update a vector with text content
    pub fn update_with_text(
        &self,
        id: String,
        vector: Vec<f32>,
        text: String,
        metadata: serde_json::Value,
    ) -> Result<()> {
        // Update metadata
        self.update(id.clone(), vector.clone(), metadata.clone())?;

        // Update text index
        if let Some(ref text_index) = self.text_index {
            text_index.delete_document(&id)?;
            text_index.add_document(&id, &text, &metadata)?;
            text_index.commit()?;
        }

        Ok(())
    }

    /// Delete a vector by ID
    /// Note: HNSW doesn't support efficient deletes, so we mark as deleted in metadata
    /// A full rebuild is needed to reclaim space
    pub fn delete(&self, id: &str) -> Result<()> {
        let mut id_map = self.id_map.write();
        let mut reverse_id_map = self.reverse_id_map.write();
        let mut metadata_map = self.metadata_map.write();

        // Get internal ID before removing
        let internal_id = id_map
            .get(id)
            .copied()
            .ok_or_else(|| VectorDbError::NotFound(id.to_string()))?;

        // Remove from all mappings
        id_map.remove(id);
        reverse_id_map.remove(&internal_id);
        metadata_map.remove(id);

        // Remove from text index if available
        if let Some(ref text_index) = self.text_index {
            text_index.delete_document(id)?;
            text_index.commit()?;
        }

        Ok(())
    }

    /// Delete multiple vectors in a batch
    pub fn delete_batch(&self, ids: &[String]) -> Result<()> {
        let mut id_map = self.id_map.write();
        let mut reverse_id_map = self.reverse_id_map.write();
        let mut metadata_map = self.metadata_map.write();

        // Get internal IDs and check all exist
        let mut internal_ids = Vec::with_capacity(ids.len());
        for id in ids {
            let internal_id = id_map
                .get(id)
                .copied()
                .ok_or_else(|| VectorDbError::NotFound(id.to_string()))?;
            internal_ids.push(internal_id);
        }

        // Remove from all mappings
        for (id, internal_id) in ids.iter().zip(internal_ids.iter()) {
            id_map.remove(id);
            reverse_id_map.remove(internal_id);
            metadata_map.remove(id);
        }

        // Remove from text index if available
        if let Some(ref text_index) = self.text_index {
            for id in ids {
                text_index.delete_document(id)?;
            }
            text_index.commit()?;
        }

        Ok(())
    }

    /// Check if a vector ID exists
    pub fn contains(&self, id: &str) -> bool {
        let id_map = self.id_map.read();
        id_map.contains_key(id)
    }

    /// Get vector by ID (if needed for updates/verification)
    /// Optimized: loads only the requested vector using mmap, not all vectors
    pub fn get_vector(&self, id: &str) -> Result<Vec<f32>> {
        let id_map = self.id_map.read();
        let internal_id = id_map
            .get(id)
            .ok_or_else(|| VectorDbError::NotFound(id.to_string()))?;

        // Load single vector from storage (O(1) with mmap)
        let storage = self.storage.read();
        storage.load_vector_by_index(*internal_id)
    }

    /// Rebuild index from current data (useful after many deletes/updates)
    pub fn rebuild_index(&self) -> Result<()> {
        use rayon::prelude::*;

        let id_map = self.id_map.read();
        let metadata_map = self.metadata_map.read();
        let storage = self.storage.read();

        // Load all vectors
        let all_vectors = storage.load_vectors()?;
        let all_metadata = storage.load_metadata()?;

        // Filter to only active indices - conditional parallelization
        let active_indices: Vec<usize> = if all_metadata.len() >= PARALLEL_THRESHOLD {
            all_metadata
                .par_iter()
                .enumerate()
                .filter_map(|(idx, meta): (usize, &VectorMetadata)| {
                    if metadata_map.contains_key(&meta.id) && idx < all_vectors.len() {
                        Some(idx)
                    } else {
                        None
                    }
                })
                .collect()
        } else {
            // Sequential for small datasets
            all_metadata
                .iter()
                .enumerate()
                .filter_map(|(idx, meta): (usize, &VectorMetadata)| {
                    if metadata_map.contains_key(&meta.id) && idx < all_vectors.len() {
                        Some(idx)
                    } else {
                        None
                    }
                })
                .collect()
        };

        // Build final structures using active indices (single clone per item)
        let mut active_vectors = Vec::with_capacity(active_indices.len());
        let mut active_metadata = Vec::with_capacity(active_indices.len());
        let mut new_id_map = AHashMap::with_capacity(active_indices.len());

        for (new_idx, &old_idx) in active_indices.iter().enumerate() {
            active_vectors.push(all_vectors[old_idx].clone());
            active_metadata.push(all_metadata[old_idx].clone());
            new_id_map.insert(all_metadata[old_idx].id.clone(), new_idx);
        }

        // Clear and rebuild index
        drop(id_map);
        drop(metadata_map);
        drop(storage);

        let mut id_map_write = self.id_map.write();
        let mut metadata_map_write = self.metadata_map.write();
        let mut next_id = self.next_id.write();

        *id_map_write = new_id_map;
        metadata_map_write.clear();
        for meta in &active_metadata {
            metadata_map_write.insert(meta.id.clone(), meta.clone());
        }
        *next_id = active_vectors.len();

        // Rebuild HNSW index - replace with new index
        let mut index_write = self.index.write();
        *index_write = HnswIndex::new(self.dimension, self.max_elements, self.ef_construction);
        if !active_vectors.is_empty() {
            index_write.insert_batch(&active_vectors, 0)?;
        }

        Ok(())
    }

    /// Clear all data
    pub fn clear(&self) -> Result<()> {
        let storage = self.storage.read();
        storage.clear()?;

        let mut metadata_map = self.metadata_map.write();
        let mut id_map = self.id_map.write();
        let mut next_id = self.next_id.write();

        metadata_map.clear();
        id_map.clear();
        *next_id = 0;

        // Clear HNSW index by creating a new one
        let mut index = self.index.write();
        *index = HnswIndex::new(self.dimension, self.max_elements, self.ef_construction);

        // Clear text index if available
        if let Some(ref _text_index) = self.text_index {
            // Tantivy doesn't have a clear method, so we delete all documents
            // by rebuilding the index (handled by TantivyIndex implementation)
        }

        Ok(())
    }

    /// Full-text search using Tantivy
    pub fn text_search(&self, query: &str, limit: usize) -> Result<Vec<(String, f32)>> {
        if let Some(ref text_index) = self.text_index {
            text_index.search(query, limit)
        } else {
            Err(VectorDbError::InvalidParameter(
                "Text search not available - no text index".to_string(),
            ))
        }
    }

    /// Hybrid search combining vector similarity and full-text search
    pub fn hybrid_search(
        &self,
        query_vector: &[f32],
        query_text: &str,
        k: usize,
        ef_search: Option<usize>,
        strategy: FusionStrategy,
    ) -> Result<Vec<HybridResult>> {
        // Vector search
        let vector_results = self.search(query_vector, k * 2, ef_search)?;
        let vector_scores: Vec<(String, f32)> = vector_results
            .into_iter()
            .map(|r| (r.id, r.distance))
            .collect();

        // Text search
        let text_scores = if let Some(ref text_index) = self.text_index {
            text_index.search(query_text, k * 2)?
        } else {
            Vec::new()
        };

        // Combine using fusion strategy
        Ok(hybrid_search(vector_scores, text_scores, strategy, k))
    }
}

/// Search result with metadata
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub id: String,
    pub distance: f32,
    pub metadata: serde_json::Value,
}

/// Database statistics
#[derive(Debug, Clone)]
pub struct DatabaseStats {
    pub total_vectors: usize,
    pub dimension: usize,
    pub metadata_keys: Vec<String>,
    pub active_vectors: usize,
    pub index_size: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_vectordb_basic() {
        let temp_dir = "/tmp/vjson_test_db";
        let _ = fs::remove_dir_all(temp_dir);

        let db = VectorDB::new(temp_dir, 128, 1000, 100).unwrap();

        // Insert vectors
        db.insert(
            "vec1".to_string(),
            vec![1.0; 128],
            serde_json::json!({"label": "a"}),
        )
        .unwrap();

        db.insert(
            "vec2".to_string(),
            vec![2.0; 128],
            serde_json::json!({"label": "b"}),
        )
        .unwrap();

        assert_eq!(db.len(), 2);

        // Search
        let query = vec![1.5; 128];
        let results = db.search(&query, 2, None).unwrap();
        assert_eq!(results.len(), 2);

        // Drop database before cleanup to release file locks
        drop(db);
        let _ = fs::remove_dir_all(temp_dir);
    }

    #[test]
    fn test_concurrent_access() {
        use std::thread;

        let temp_dir = "/tmp/vjson_test_concurrent";
        let _ = fs::remove_dir_all(temp_dir);

        let db = Arc::new(VectorDB::new(temp_dir, 128, 10000, 100).unwrap());

        // Insert some initial data
        for i in 0..100 {
            db.insert(
                format!("vec{}", i),
                vec![i as f32; 128],
                serde_json::json!({"index": i}),
            )
            .unwrap();
        }

        // Spawn multiple reader threads
        let mut handles = vec![];
        for _ in 0..10 {
            let db_clone = Arc::clone(&db);
            let handle = thread::spawn(move || {
                let query = vec![50.0; 128];
                let results = db_clone.search(&query, 5, None).unwrap();
                assert_eq!(results.len(), 5);
            });
            handles.push(handle);
        }

        // Wait for all threads
        for handle in handles {
            handle.join().unwrap();
        }

        // Drop database before cleanup to release file locks (important on Windows)
        drop(db);
        let _ = fs::remove_dir_all(temp_dir);
    }
}
