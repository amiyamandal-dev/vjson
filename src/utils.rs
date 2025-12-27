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
    if a.len() != b.len() {
        return 0.0;
    }

    // Use platform-specific SIMD for large vectors
    #[cfg(target_arch = "aarch64")]
    {
        if a.len() >= 16 {
            return unsafe { simd::cosine_similarity_neon(a, b) };
        }
    }

    #[cfg(target_arch = "x86_64")]
    {
        if a.len() >= 16 {
            return cosine_similarity_simd(a, b);
        }
    }

    cosine_similarity_scalar(a, b)
}

/// Scalar cosine similarity for small vectors
#[inline]
fn cosine_similarity_scalar(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let magnitude_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let magnitude_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if magnitude_a == 0.0 || magnitude_b == 0.0 {
        return 0.0;
    }

    dot_product / (magnitude_a * magnitude_b)
}

/// SIMD-optimized cosine similarity
#[allow(dead_code)]
#[inline]
fn cosine_similarity_simd(a: &[f32], b: &[f32]) -> f32 {
    let (chunks_a, remainder_a) = a.as_chunks::<8>();
    let (chunks_b, remainder_b) = b.as_chunks::<8>();

    let mut dot = 0.0f32;
    let mut mag_a = 0.0f32;
    let mut mag_b = 0.0f32;

    // Process chunks
    for (chunk_a, chunk_b) in chunks_a.iter().zip(chunks_b.iter()) {
        for i in 0..8 {
            dot += chunk_a[i] * chunk_b[i];
            mag_a += chunk_a[i] * chunk_a[i];
            mag_b += chunk_b[i] * chunk_b[i];
        }
    }

    // Handle remainder
    for (x, y) in remainder_a.iter().zip(remainder_b.iter()) {
        dot += x * y;
        mag_a += x * x;
        mag_b += y * y;
    }

    let magnitude_a = mag_a.sqrt();
    let magnitude_b = mag_b.sqrt();

    if magnitude_a == 0.0 || magnitude_b == 0.0 {
        return 0.0;
    }

    dot / (magnitude_a * magnitude_b)
}

/// Convert L2 distance to cosine similarity for normalized vectors
/// For normalized vectors: cosine_similarity = 1 - (L2_distance^2 / 2)
#[allow(dead_code)]
pub fn l2_to_cosine_similarity(l2_distance: f32) -> f32 {
    1.0 - (l2_distance * l2_distance / 2.0)
}

/// Compute dot product between two vectors
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }

    // Use platform-specific SIMD for large vectors
    #[cfg(target_arch = "aarch64")]
    {
        if a.len() >= 16 {
            return unsafe { simd::dot_product_neon(a, b) };
        }
    }

    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") && a.len() >= 32 {
            return unsafe { simd::dot_product_avx2(a, b) };
        }
    }

    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
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
