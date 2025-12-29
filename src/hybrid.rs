use rayon::prelude::*;
use std::collections::HashMap;

/// Hybrid search result combining vector and text scores
#[derive(Debug, Clone)]
pub struct HybridResult {
    pub id: String,
    pub vector_score: f32,
    pub text_score: f32,
    pub combined_score: f32,
}

/// Score fusion strategies for combining vector and text search
#[derive(Debug, Clone, Copy)]
pub enum FusionStrategy {
    /// Reciprocal Rank Fusion - position-based fusion (best default)
    ReciprocalRankFusion { k: f32 },

    /// Weighted sum of normalized scores
    WeightedSum {
        vector_weight: f32,
        text_weight: f32,
    },

    /// Take maximum score
    Max,

    /// Take minimum score
    Min,

    /// Average of both scores
    Average,
}

impl Default for FusionStrategy {
    fn default() -> Self {
        FusionStrategy::ReciprocalRankFusion { k: 60.0 }
    }
}

/// Combine vector search and text search results
pub fn hybrid_search(
    vector_results: Vec<(String, f32)>, // (id, distance) - lower is better
    text_results: Vec<(String, f32)>,   // (id, score) - higher is better
    strategy: FusionStrategy,
    top_k: usize,
) -> Vec<HybridResult> {
    match strategy {
        FusionStrategy::ReciprocalRankFusion { k } => {
            reciprocal_rank_fusion(&vector_results, &text_results, k, top_k)
        },
        FusionStrategy::WeightedSum {
            vector_weight,
            text_weight,
        } => weighted_sum(
            &vector_results,
            &text_results,
            vector_weight,
            text_weight,
            top_k,
        ),
        FusionStrategy::Max => max_fusion(&vector_results, &text_results, top_k),
        FusionStrategy::Min => min_fusion(&vector_results, &text_results, top_k),
        FusionStrategy::Average => average_fusion(&vector_results, &text_results, top_k),
    }
}

/// Reciprocal Rank Fusion (RRF)
/// Score = 1 / (k + rank)
/// Works well without score normalization
fn reciprocal_rank_fusion(
    vector_results: &[(String, f32)],
    text_results: &[(String, f32)],
    k: f32,
    top_k: usize,
) -> Vec<HybridResult> {
    // Calculate rank-based scores in parallel
    let vector_ranks: HashMap<String, f32> = vector_results
        .par_iter()
        .enumerate()
        .map(|(rank, (id, _))| {
            let score = 1.0 / (k + rank as f32 + 1.0);
            (id.clone(), score)
        })
        .collect();

    let text_ranks: HashMap<String, f32> = text_results
        .par_iter()
        .enumerate()
        .map(|(rank, (id, _))| {
            let score = 1.0 / (k + rank as f32 + 1.0);
            (id.clone(), score)
        })
        .collect();

    // Get all unique IDs
    let all_ids: Vec<String> = vector_ranks
        .keys()
        .chain(text_ranks.keys())
        .cloned()
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();

    // Combine scores in parallel
    let mut results: Vec<HybridResult> = all_ids
        .par_iter()
        .map(|id| {
            let vec_score = vector_ranks.get(id).copied().unwrap_or(0.0);
            let txt_score = text_ranks.get(id).copied().unwrap_or(0.0);
            let combined = vec_score + txt_score;

            HybridResult {
                id: id.clone(),
                vector_score: vec_score,
                text_score: txt_score,
                combined_score: combined,
            }
        })
        .collect();

    // Sort by combined score descending
    results.par_sort_by(|a, b| b.combined_score.partial_cmp(&a.combined_score).unwrap());
    results.truncate(top_k);

    results
}

/// Weighted sum with normalization
fn weighted_sum(
    vector_results: &[(String, f32)],
    text_results: &[(String, f32)],
    vector_weight: f32,
    text_weight: f32,
    top_k: usize,
) -> Vec<HybridResult> {
    // Convert distance to similarity and collect
    // Note: HashMap::with_capacity doesn't help with parallel collect
    let vector_scores: HashMap<String, f32> = vector_results
        .par_iter()
        .map(|(id, distance)| {
            let similarity = 1.0 / (1.0 + distance);
            (id.clone(), similarity)
        })
        .collect();

    let text_scores: HashMap<String, f32> = text_results
        .par_iter()
        .map(|(id, score)| (id.clone(), *score))
        .collect();

    // Normalize scores to [0, 1]
    let max_vec = vector_scores.values().copied().fold(0.0_f32, f32::max);
    let max_txt = text_scores.values().copied().fold(0.0_f32, f32::max);

    // Get all unique IDs
    let all_ids: Vec<String> = vector_scores
        .keys()
        .chain(text_scores.keys())
        .cloned()
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();

    // Combine with weights
    let mut results: Vec<HybridResult> = all_ids
        .par_iter()
        .map(|id| {
            let vec_score = vector_scores.get(id).copied().unwrap_or(0.0) / max_vec.max(1e-10);
            let txt_score = text_scores.get(id).copied().unwrap_or(0.0) / max_txt.max(1e-10);
            let combined = vec_score * vector_weight + txt_score * text_weight;

            HybridResult {
                id: id.clone(),
                vector_score: vec_score,
                text_score: txt_score,
                combined_score: combined,
            }
        })
        .collect();

    results.par_sort_by(|a, b| b.combined_score.partial_cmp(&a.combined_score).unwrap());
    results.truncate(top_k);

    results
}

/// Max score fusion
fn max_fusion(
    vector_results: &[(String, f32)],
    text_results: &[(String, f32)],
    top_k: usize,
) -> Vec<HybridResult> {
    let vector_scores: HashMap<String, f32> = vector_results
        .iter()
        .map(|(id, d)| (id.clone(), 1.0 / (1.0 + d)))
        .collect();

    let text_scores: HashMap<String, f32> = text_results
        .iter()
        .map(|(id, s)| (id.clone(), *s))
        .collect();

    let all_ids: Vec<String> = vector_scores
        .keys()
        .chain(text_scores.keys())
        .cloned()
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();

    let mut results: Vec<HybridResult> = all_ids
        .par_iter()
        .map(|id| {
            let vec_score = vector_scores.get(id).copied().unwrap_or(0.0);
            let txt_score = text_scores.get(id).copied().unwrap_or(0.0);
            let combined = vec_score.max(txt_score);

            HybridResult {
                id: id.clone(),
                vector_score: vec_score,
                text_score: txt_score,
                combined_score: combined,
            }
        })
        .collect();

    results.par_sort_by(|a, b| b.combined_score.partial_cmp(&a.combined_score).unwrap());
    results.truncate(top_k);
    results
}

/// Min score fusion
fn min_fusion(
    vector_results: &[(String, f32)],
    text_results: &[(String, f32)],
    top_k: usize,
) -> Vec<HybridResult> {
    let vector_scores: HashMap<String, f32> = vector_results
        .iter()
        .map(|(id, d)| (id.clone(), 1.0 / (1.0 + d)))
        .collect();

    let text_scores: HashMap<String, f32> = text_results
        .iter()
        .map(|(id, s)| (id.clone(), *s))
        .collect();

    let all_ids: Vec<String> = vector_scores
        .keys()
        .chain(text_scores.keys())
        .cloned()
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();

    let mut results: Vec<HybridResult> = all_ids
        .par_iter()
        .map(|id| {
            let vec_score = vector_scores.get(id).copied().unwrap_or(0.0);
            let txt_score = text_scores.get(id).copied().unwrap_or(0.0);
            let combined = vec_score.min(txt_score);

            HybridResult {
                id: id.clone(),
                vector_score: vec_score,
                text_score: txt_score,
                combined_score: combined,
            }
        })
        .collect();

    results.par_sort_by(|a, b| b.combined_score.partial_cmp(&a.combined_score).unwrap());
    results.truncate(top_k);
    results
}

/// Average score fusion
fn average_fusion(
    vector_results: &[(String, f32)],
    text_results: &[(String, f32)],
    top_k: usize,
) -> Vec<HybridResult> {
    weighted_sum(vector_results, text_results, 0.5, 0.5, top_k)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reciprocal_rank_fusion() {
        let vector_results = vec![
            ("doc1".to_string(), 0.1),
            ("doc2".to_string(), 0.2),
            ("doc3".to_string(), 0.3),
        ];

        let text_results = vec![
            ("doc2".to_string(), 10.0),
            ("doc3".to_string(), 5.0),
            ("doc4".to_string(), 3.0),
        ];

        let results = hybrid_search(
            vector_results,
            text_results,
            FusionStrategy::ReciprocalRankFusion { k: 60.0 },
            10,
        );

        assert!(!results.is_empty());
        // doc2 appears in both, should rank high
        assert_eq!(results[0].id, "doc2");
    }

    #[test]
    fn test_weighted_sum() {
        let vector_results = vec![("doc1".to_string(), 0.1), ("doc2".to_string(), 0.5)];

        let text_results = vec![("doc1".to_string(), 5.0), ("doc3".to_string(), 10.0)];

        let results = hybrid_search(
            vector_results,
            text_results,
            FusionStrategy::WeightedSum {
                vector_weight: 0.7,
                text_weight: 0.3,
            },
            10,
        );

        assert!(!results.is_empty());
    }
}
