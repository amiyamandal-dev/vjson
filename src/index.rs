use crate::error::{Result, VectorDbError};
use hnsw_rs::prelude::*;
use parking_lot::RwLock;
use std::sync::Arc;

/// Thread-safe HNSW index wrapper
pub struct HnswIndex {
    index: Arc<RwLock<Hnsw<'static, f32, DistL2>>>,
    dimension: usize,
    max_elements: usize,
}

impl HnswIndex {
    /// Create a new HNSW index
    pub fn new(dimension: usize, max_elements: usize, ef_construction: usize) -> Self {
        let hnsw = Hnsw::<f32, DistL2>::new(
            ef_construction, // ef_construction parameter (search quality during building)
            max_elements,    // max number of elements
            16,              // max_nb_connection (M parameter) - connections per layer
            256,             // max_layer - maximum layer in the graph
            DistL2 {},       // Distance function (L2/Euclidean distance)
        );

        Self {
            index: Arc::new(RwLock::new(hnsw)),
            dimension,
            max_elements,
        }
    }

    /// Insert a batch of vectors (write lock)
    /// This is more efficient than inserting one by one
    pub fn insert_batch(&self, vectors: &[Vec<f32>], start_id: usize) -> Result<()> {
        // Validate dimensions
        for vec in vectors {
            if vec.len() != self.dimension {
                return Err(VectorDbError::DimensionMismatch {
                    expected: self.dimension,
                    actual: vec.len(),
                });
            }
        }

        // Get write lock (exclusive access)
        let index = self.index.write();

        // Batch insert
        for (i, vec) in vectors.iter().enumerate() {
            let id = start_id + i;
            index.insert((vec.as_slice(), id));
        }

        Ok(())
    }

    /// Search for k nearest neighbors (read lock - allows parallel searches)
    pub fn search(&self, query: &[f32], k: usize, ef_search: usize) -> Result<Vec<(usize, f32)>> {
        if query.len() != self.dimension {
            return Err(VectorDbError::DimensionMismatch {
                expected: self.dimension,
                actual: query.len(),
            });
        }

        // Get read lock (shared access - multiple concurrent reads allowed)
        let index = self.index.read();

        // Perform search
        let results = index.search(query, k, ef_search);

        // Convert to (id, distance) pairs
        let results: Vec<(usize, f32)> = results
            .into_iter()
            .map(|neighbor| (neighbor.d_id, neighbor.distance))
            .collect();

        Ok(results)
    }

    /// Get the number of elements in the index
    pub fn len(&self) -> usize {
        let index = self.index.read();
        index.get_nb_point()
    }

    /// Check if index is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

// Make HnswIndex cloneable for sharing across threads
impl Clone for HnswIndex {
    fn clone(&self) -> Self {
        Self {
            index: Arc::clone(&self.index),
            dimension: self.dimension,
            max_elements: self.max_elements,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rayon::prelude::*;

    #[test]
    fn test_hnsw_index() {
        let index = HnswIndex::new(128, 1000, 100);

        let vectors: Vec<Vec<f32>> = (0..10).map(|i| vec![i as f32; 128]).collect();
        index.insert_batch(&vectors, 0).unwrap();

        assert_eq!(index.len(), 10);

        let query = vec![5.0; 128];
        let results = index.search(&query, 3, 50).unwrap();
        assert_eq!(results.len(), 3);

        // Verify the closest result is id=5 (vector filled with 5.0)
        assert_eq!(results[0].0, 5);
    }

    #[test]
    fn test_dimension_mismatch() {
        let index = HnswIndex::new(128, 1000, 100);

        let wrong_vector = vec![1.0; 64]; // Wrong dimension
        let result = index.insert_batch(&vec![wrong_vector], 0);

        assert!(result.is_err());
        match result {
            Err(VectorDbError::DimensionMismatch { expected, actual }) => {
                assert_eq!(expected, 128);
                assert_eq!(actual, 64);
            },
            _ => panic!("Expected DimensionMismatch error"),
        }
    }

    #[test]
    fn test_parallel_search() {
        let index = HnswIndex::new(128, 1000, 100);

        let vectors: Vec<Vec<f32>> = (0..100).map(|i| vec![i as f32; 128]).collect();
        index.insert_batch(&vectors, 0).unwrap();

        // Parallel search using rayon
        let queries: Vec<Vec<f32>> = (0..10).map(|i| vec![i as f32 * 10.0; 128]).collect();
        let results: Vec<_> = queries
            .par_iter()
            .map(|query| index.search(query, 5, 50).unwrap())
            .collect();

        assert_eq!(results.len(), 10);
        for result in results {
            assert_eq!(result.len(), 5);
        }
    }

    #[test]
    fn test_empty_index() {
        let index = HnswIndex::new(128, 1000, 100);

        assert!(index.is_empty());
        assert_eq!(index.len(), 0);

        // Search on empty index should return empty results
        let query = vec![1.0; 128];
        let results = index.search(&query, 5, 50).unwrap();
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_clone_index() {
        let index = HnswIndex::new(128, 1000, 100);

        let vectors: Vec<Vec<f32>> = (0..5).map(|i| vec![i as f32; 128]).collect();
        index.insert_batch(&vectors, 0).unwrap();

        // Clone the index
        let cloned_index = index.clone();

        // Both should have the same data
        assert_eq!(index.len(), cloned_index.len());
        assert_eq!(index.len(), 5);

        // Search should work on both
        let query = vec![2.0; 128];
        let results1 = index.search(&query, 3, 50).unwrap();
        let results2 = cloned_index.search(&query, 3, 50).unwrap();

        assert_eq!(results1.len(), results2.len());
    }

    #[test]
    fn test_large_batch_insert() {
        let index = HnswIndex::new(64, 10000, 100);

        // Insert 1000 vectors in batches
        for batch in 0..10 {
            let vectors: Vec<Vec<f32>> = (0..100)
                .map(|i| vec![(batch * 100 + i) as f32; 64])
                .collect();
            index.insert_batch(&vectors, batch * 100).unwrap();
        }

        assert_eq!(index.len(), 1000);

        // Search should find nearest neighbors
        let query = vec![500.0; 64];
        let results = index.search(&query, 10, 100).unwrap();
        assert_eq!(results.len(), 10);

        // Verify the closest result is around id=500
        assert!(results[0].0 >= 495 && results[0].0 <= 505);
    }
}
