/// Utility functions for vector operations
use crate::simd;

/// Normalize a vector to unit length (L2 normalization) - SIMD optimized
/// Uses platform-specific SIMD (NEON on ARM, AVX2/SSE on x86)
#[inline]
pub fn normalize_vector(vector: &[f32]) -> Vec<f32> {
    // Delegate to SIMD module for maximum performance
    simd::normalize_vector_simd(vector)
}

/// Scalar implementation for small vectors or fallback
#[allow(dead_code)]
#[inline]
fn normalize_vector_scalar(vector: &[f32]) -> Vec<f32> {
    let magnitude: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();

    if magnitude == 0.0 {
        return vector.to_vec();
    }

    vector.iter().map(|x| x / magnitude).collect()
}

/// SIMD-optimized normalization for larger vectors
#[allow(dead_code)]
#[inline]
fn normalize_vector_simd(vector: &[f32]) -> Vec<f32> {
    // Compute magnitude using auto-vectorization hints
    let mut sum = 0.0f32;

    // Process in chunks to encourage auto-vectorization
    let (chunks, remainder) = vector.as_chunks::<8>();

    // Auto-vectorized chunk processing
    for chunk in chunks {
        sum += chunk[0] * chunk[0];
        sum += chunk[1] * chunk[1];
        sum += chunk[2] * chunk[2];
        sum += chunk[3] * chunk[3];
        sum += chunk[4] * chunk[4];
        sum += chunk[5] * chunk[5];
        sum += chunk[6] * chunk[6];
        sum += chunk[7] * chunk[7];
    }

    // Handle remainder
    for &val in remainder {
        sum += val * val;
    }

    let magnitude = sum.sqrt();

    if magnitude == 0.0 {
        return vector.to_vec();
    }

    // Normalize with pre-allocation
    let inv_magnitude = 1.0 / magnitude;
    let mut result = Vec::with_capacity(vector.len());

    // Auto-vectorized normalization
    for chunk in chunks {
        result.push(chunk[0] * inv_magnitude);
        result.push(chunk[1] * inv_magnitude);
        result.push(chunk[2] * inv_magnitude);
        result.push(chunk[3] * inv_magnitude);
        result.push(chunk[4] * inv_magnitude);
        result.push(chunk[5] * inv_magnitude);
        result.push(chunk[6] * inv_magnitude);
        result.push(chunk[7] * inv_magnitude);
    }

    for &val in remainder {
        result.push(val * inv_magnitude);
    }

    result
}

/// Normalize a batch of vectors
pub fn normalize_vectors(vectors: &[Vec<f32>]) -> Vec<Vec<f32>> {
    use rayon::prelude::*;
    vectors.par_iter().map(|v| normalize_vector(v)).collect()
}

/// Compute cosine similarity between two vectors - SIMD optimized
/// Returns value in range [-1, 1], where 1 is most similar
#[inline]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    // Delegate to cross-platform SIMD implementation
    simd::cosine_similarity_simd(a, b)
}

/// Convert L2 distance to cosine similarity for normalized vectors
/// For normalized vectors: cosine_similarity = 1 - (L2_distance^2 / 2)
#[allow(dead_code)]
pub fn l2_to_cosine_similarity(l2_distance: f32) -> f32 {
    1.0 - (l2_distance * l2_distance / 2.0)
}

/// Compute dot product between two vectors - SIMD optimized
#[inline]
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    // Delegate to cross-platform SIMD implementation
    simd::dot_product_simd(a, b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_vector() {
        let vec = vec![3.0, 4.0];
        let normalized = normalize_vector(&vec);

        // Should be [0.6, 0.8]
        assert!((normalized[0] - 0.6).abs() < 0.001);
        assert!((normalized[1] - 0.8).abs() < 0.001);

        // Magnitude should be 1
        let mag: f32 = normalized.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((mag - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0];
        let b = vec![1.0, 0.0];
        let c = vec![0.0, 1.0];

        // Identical vectors
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 0.001);

        // Orthogonal vectors
        assert!(cosine_similarity(&a, &c).abs() < 0.001);

        // Opposite vectors
        let d = vec![-1.0, 0.0];
        assert!((cosine_similarity(&a, &d) + 1.0).abs() < 0.001);
    }

    #[test]
    fn test_dot_product() {
        let a = vec![2.0, 3.0];
        let b = vec![4.0, 5.0];

        // 2*4 + 3*5 = 8 + 15 = 23
        assert!((dot_product(&a, &b) - 23.0).abs() < 0.001);
    }

    #[test]
    fn test_zero_vector() {
        let zero = vec![0.0, 0.0, 0.0];
        let normalized = normalize_vector(&zero);

        // Should return zero vector
        assert_eq!(normalized, zero);
    }
}
