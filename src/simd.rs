/// SIMD-optimized vector operations
/// Platform-specific implementations for maximum performance

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Normalize a vector using platform-specific SIMD
#[inline]
pub fn normalize_vector_simd(vector: &[f32]) -> Vec<f32> {
    #[cfg(target_arch = "aarch64")]
    {
        if vector.len() >= 16 {
            return unsafe { normalize_vector_neon(vector) };
        }
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && vector.len() >= 32 {
            return unsafe { normalize_vector_avx2(vector) };
        }
        if is_x86_feature_detected!("sse2") && vector.len() >= 16 {
            return unsafe { normalize_vector_sse(vector) };
        }
    }

    // Fallback to scalar
    normalize_vector_scalar(vector)
}

/// Scalar fallback for small vectors
#[inline]
fn normalize_vector_scalar(vector: &[f32]) -> Vec<f32> {
    let magnitude: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();

    if magnitude == 0.0 {
        return vector.to_vec();
    }

    let inv_magnitude = 1.0 / magnitude;
    vector.iter().map(|x| x * inv_magnitude).collect()
}

// ============================================================================
// ARM NEON Implementation (Apple Silicon, ARM devices)
// ============================================================================

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn normalize_vector_neon(vector: &[f32]) -> Vec<f32> {
    let len = vector.len();
    let mut result = Vec::with_capacity(len);

    // Calculate magnitude using NEON (process 4 floats at a time)
    let mut sum_vec = vdupq_n_f32(0.0);

    let chunks = len / 4;

    // Process 4 floats at a time
    for i in 0..chunks {
        let offset = i * 4;
        let v = vld1q_f32(vector.as_ptr().add(offset));
        sum_vec = vfmaq_f32(sum_vec, v, v); // sum_vec += v * v
    }

    // Horizontal sum of sum_vec
    let sum_pair = vpadd_f32(vget_low_f32(sum_vec), vget_high_f32(sum_vec));
    let sum_final = vpadd_f32(sum_pair, sum_pair);
    let mut magnitude_sq = vget_lane_f32(sum_final, 0);

    // Add remainder
    for i in (chunks * 4)..len {
        let val = *vector.get_unchecked(i);
        magnitude_sq += val * val;
    }

    let magnitude = magnitude_sq.sqrt();

    if magnitude == 0.0 {
        return vector.to_vec();
    }

    let inv_mag = 1.0 / magnitude;
    let inv_mag_vec = vdupq_n_f32(inv_mag);

    // Normalize using NEON
    for i in 0..chunks {
        let offset = i * 4;
        let v = vld1q_f32(vector.as_ptr().add(offset));
        let normalized = vmulq_f32(v, inv_mag_vec);

        // Store results
        let mut temp = [0.0f32; 4];
        vst1q_f32(temp.as_mut_ptr(), normalized);
        result.extend_from_slice(&temp);
    }

    // Handle remainder
    for i in (chunks * 4)..len {
        result.push(*vector.get_unchecked(i) * inv_mag);
    }

    result
}

#[cfg(target_arch = "aarch64")]
#[inline]
pub unsafe fn dot_product_neon(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    let chunks = len / 4;

    let mut sum_vec = vdupq_n_f32(0.0);

    for i in 0..chunks {
        let offset = i * 4;
        let a_vec = vld1q_f32(a.as_ptr().add(offset));
        let b_vec = vld1q_f32(b.as_ptr().add(offset));
        sum_vec = vfmaq_f32(sum_vec, a_vec, b_vec);
    }

    // Horizontal sum
    let sum_pair = vpadd_f32(vget_low_f32(sum_vec), vget_high_f32(sum_vec));
    let sum_final = vpadd_f32(sum_pair, sum_pair);
    let mut result = vget_lane_f32(sum_final, 0);

    // Add remainder
    for i in (chunks * 4)..len {
        result += a.get_unchecked(i) * b.get_unchecked(i);
    }

    result
}

#[cfg(target_arch = "aarch64")]
#[inline]
pub unsafe fn cosine_similarity_neon(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }

    let len = a.len();
    let chunks = len / 4;

    let mut dot_vec = vdupq_n_f32(0.0);
    let mut mag_a_vec = vdupq_n_f32(0.0);
    let mut mag_b_vec = vdupq_n_f32(0.0);

    // Process 4 floats at a time
    for i in 0..chunks {
        let offset = i * 4;
        let a_vec = vld1q_f32(a.as_ptr().add(offset));
        let b_vec = vld1q_f32(b.as_ptr().add(offset));

        dot_vec = vfmaq_f32(dot_vec, a_vec, b_vec);
        mag_a_vec = vfmaq_f32(mag_a_vec, a_vec, a_vec);
        mag_b_vec = vfmaq_f32(mag_b_vec, b_vec, b_vec);
    }

    // Horizontal sums
    let dot_pair = vpadd_f32(vget_low_f32(dot_vec), vget_high_f32(dot_vec));
    let dot_final = vpadd_f32(dot_pair, dot_pair);
    let mut dot = vget_lane_f32(dot_final, 0);

    let mag_a_pair = vpadd_f32(vget_low_f32(mag_a_vec), vget_high_f32(mag_a_vec));
    let mag_a_final = vpadd_f32(mag_a_pair, mag_a_pair);
    let mut mag_a_sq = vget_lane_f32(mag_a_final, 0);

    let mag_b_pair = vpadd_f32(vget_low_f32(mag_b_vec), vget_high_f32(mag_b_vec));
    let mag_b_final = vpadd_f32(mag_b_pair, mag_b_pair);
    let mut mag_b_sq = vget_lane_f32(mag_b_final, 0);

    // Handle remainder
    for i in (chunks * 4)..len {
        let a_val = *a.get_unchecked(i);
        let b_val = *b.get_unchecked(i);
        dot += a_val * b_val;
        mag_a_sq += a_val * a_val;
        mag_b_sq += b_val * b_val;
    }

    let magnitude_a = mag_a_sq.sqrt();
    let magnitude_b = mag_b_sq.sqrt();

    if magnitude_a == 0.0 || magnitude_b == 0.0 {
        return 0.0;
    }

    dot / (magnitude_a * magnitude_b)
}

// ============================================================================
// x86_64 AVX2 Implementation (Intel/AMD)
// ============================================================================

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn normalize_vector_avx2(vector: &[f32]) -> Vec<f32> {
    let len = vector.len();
    let mut result = Vec::with_capacity(len);

    // Calculate magnitude using AVX2 (8 floats at a time)
    let mut sum_vec = _mm256_setzero_ps();

    let chunks = len / 8;
    let remainder = len % 8;

    for i in 0..chunks {
        let offset = i * 8;
        let v = _mm256_loadu_ps(vector.as_ptr().add(offset));
        sum_vec = _mm256_fmadd_ps(v, v, sum_vec); // sum_vec += v * v
    }

    // Horizontal sum
    let sum = horizontal_sum_avx2(sum_vec);
    let mut magnitude_sq = sum;

    // Add remainder
    for i in (chunks * 8)..len {
        let val = *vector.get_unchecked(i);
        magnitude_sq += val * val;
    }

    let magnitude = magnitude_sq.sqrt();

    if magnitude == 0.0 {
        return vector.to_vec();
    }

    let inv_mag = 1.0 / magnitude;
    let inv_mag_vec = _mm256_set1_ps(inv_mag);

    // Normalize using AVX2
    for i in 0..chunks {
        let offset = i * 8;
        let v = _mm256_loadu_ps(vector.as_ptr().add(offset));
        let normalized = _mm256_mul_ps(v, inv_mag_vec);

        let mut temp = [0.0f32; 8];
        _mm256_storeu_ps(temp.as_mut_ptr(), normalized);
        result.extend_from_slice(&temp);
    }

    // Handle remainder
    for i in (chunks * 8)..len {
        result.push(*vector.get_unchecked(i) * inv_mag);
    }

    result
}

#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn horizontal_sum_avx2(v: __m256) -> f32 {
    // Extract high and low 128-bit lanes
    let hi = _mm256_extractf128_ps(v, 1);
    let lo = _mm256_castps256_ps128(v);

    // Add them together
    let sum128 = _mm_add_ps(hi, lo);

    // Horizontal add within 128 bits
    let shuf = _mm_movehdup_ps(sum128);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehl_ps(sums, sums);
    let result = _mm_add_ss(sums, shuf2);

    _mm_cvtss_f32(result)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
pub unsafe fn dot_product_avx2(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    let chunks = len / 8;

    let mut sum_vec = _mm256_setzero_ps();

    for i in 0..chunks {
        let offset = i * 8;
        let a_vec = _mm256_loadu_ps(a.as_ptr().add(offset));
        let b_vec = _mm256_loadu_ps(b.as_ptr().add(offset));
        sum_vec = _mm256_fmadd_ps(a_vec, b_vec, sum_vec);
    }

    let mut result = horizontal_sum_avx2(sum_vec);

    // Add remainder
    for i in (chunks * 8)..len {
        result += a.get_unchecked(i) * b.get_unchecked(i);
    }

    result
}

// ============================================================================
// x86_64 SSE Implementation (fallback for older CPUs)
// ============================================================================

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
#[inline]
unsafe fn normalize_vector_sse(vector: &[f32]) -> Vec<f32> {
    let len = vector.len();
    let mut result = Vec::with_capacity(len);

    let mut sum_vec = _mm_setzero_ps();
    let chunks = len / 4;

    for i in 0..chunks {
        let offset = i * 4;
        let v = _mm_loadu_ps(vector.as_ptr().add(offset));
        let sq = _mm_mul_ps(v, v);
        sum_vec = _mm_add_ps(sum_vec, sq);
    }

    // Horizontal sum for SSE
    let shuf = _mm_movehdup_ps(sum_vec);
    let sums = _mm_add_ps(sum_vec, shuf);
    let shuf2 = _mm_movehl_ps(sums, sums);
    let result_vec = _mm_add_ss(sums, shuf2);
    let mut magnitude_sq = _mm_cvtss_f32(result_vec);

    // Add remainder
    for i in (chunks * 4)..len {
        let val = *vector.get_unchecked(i);
        magnitude_sq += val * val;
    }

    let magnitude = magnitude_sq.sqrt();

    if magnitude == 0.0 {
        return vector.to_vec();
    }

    let inv_mag = 1.0 / magnitude;
    let inv_mag_vec = _mm_set1_ps(inv_mag);

    for i in 0..chunks {
        let offset = i * 4;
        let v = _mm_loadu_ps(vector.as_ptr().add(offset));
        let normalized = _mm_mul_ps(v, inv_mag_vec);

        let mut temp = [0.0f32; 4];
        _mm_storeu_ps(temp.as_mut_ptr(), normalized);
        result.extend_from_slice(&temp);
    }

    for i in (chunks * 4)..len {
        result.push(*vector.get_unchecked(i) * inv_mag);
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_simd() {
        let vec = vec![3.0, 4.0, 0.0, 0.0];
        let normalized = normalize_vector_simd(&vec);

        assert!((normalized[0] - 0.6).abs() < 0.001);
        assert!((normalized[1] - 0.8).abs() < 0.001);

        let mag: f32 = normalized.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((mag - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_normalize_large() {
        let vec: Vec<f32> = (0..128).map(|i| i as f32).collect();
        let normalized = normalize_vector_simd(&vec);

        let mag: f32 = normalized.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((mag - 1.0).abs() < 0.001);
    }
}
