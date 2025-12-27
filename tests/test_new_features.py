#!/usr/bin/env python3
"""Test all new features: batch ops, range search, normalization, stats, incremental updates"""

import math
import random
import shutil
import tempfile

import vjson

def test_all_new_features():
    """Comprehensive test for all newly added features"""

    temp_dir = tempfile.mkdtemp()
    print(f"Testing new features in: {temp_dir}")

    try:
        dimension = 128
        db = vjson.PyVectorDB(temp_dir, dimension=dimension)

        # ========== TEST 1: Batch Get Operations ==========
        print("\n=== TEST 1: Batch Get Operations ===")

        # Insert test data
        test_data = []
        for i in range(20):
            vector = [float(i + j * 0.1) for j in range(dimension)]
            metadata = {"index": i, "category": f"cat_{i % 3}", "score": i * 0.05}
            db.insert(f"vec_{i}", vector, metadata)
            test_data.append((f"vec_{i}", vector, metadata))

        print(f"✓ Inserted {len(db)} vectors")

        # Test get_metadata_batch
        ids_to_fetch = ["vec_0", "vec_5", "vec_10", "vec_15", "nonexistent"]
        batch_metadata = db.get_metadata_batch(ids_to_fetch)
        print(f"✓ get_metadata_batch: fetched {len(batch_metadata)} items")
        assert len(batch_metadata) == 4  # 4 exist, 1 doesn't
        assert all("id" in item and "metadata" in item for item in batch_metadata)
        assert batch_metadata[0]["metadata"]["index"] == 0

        # Test get_vectors_batch
        batch_vectors = db.get_vectors_batch(["vec_1", "vec_2", "vec_3"])
        print(f"✓ get_vectors_batch: fetched {len(batch_vectors)} vectors")
        assert len(batch_vectors) == 3
        assert all("id" in item and "vector" in item for item in batch_vectors)
        assert len(batch_vectors[0]["vector"]) == dimension

        # ========== TEST 2: Range Search ==========
        print("\n=== TEST 2: Range Search ===")

        query = [float(5 + j * 0.1) for j in range(dimension)]  # Close to vec_5

        # Search within small distance threshold
        range_results = db.range_search(query, max_distance=10.0)
        print(f"✓ Range search (max_distance=10.0): found {len(range_results)} results")
        assert len(range_results) > 0
        assert all(r["distance"] <= 10.0 for r in range_results)

        # Larger threshold should find more
        range_results_large = db.range_search(query, max_distance=50.0)
        print(
            f"✓ Range search (max_distance=50.0): found {len(range_results_large)} results"
        )
        assert len(range_results_large) >= len(range_results)

        # ========== TEST 3: Vector Normalization ==========
        print("\n=== TEST 3: Vector Normalization ===")

        # Test single vector normalization
        test_vec = [3.0, 4.0] + [0.0] * (dimension - 2)
        normalized = vjson.normalize_vector(test_vec)

        # Check magnitude is 1
        magnitude = math.sqrt(sum(x**2 for x in normalized))
        print(f"✓ Normalized vector magnitude: {magnitude:.6f}")
        assert abs(magnitude - 1.0) < 0.001

        # Test batch normalization
        test_vecs = [
            [1.0, 0.0] + [0.0] * (dimension - 2),
            [0.0, 1.0] + [0.0] * (dimension - 2),
            [1.0, 1.0] + [0.0] * (dimension - 2),
        ]
        normalized_batch = vjson.normalize_vectors(test_vecs)
        print(f"✓ Normalized {len(normalized_batch)} vectors")
        assert len(normalized_batch) == 3
        for vec in normalized_batch:
            mag = math.sqrt(sum(x**2 for x in vec))
            assert abs(mag - 1.0) < 0.001

        # ========== TEST 4: Similarity Functions ==========
        print("\n=== TEST 4: Similarity Functions ===")

        # Cosine similarity
        vec_a = [1.0, 0.0] + [0.0] * (dimension - 2)
        vec_b = [1.0, 0.0] + [0.0] * (dimension - 2)
        vec_c = [0.0, 1.0] + [0.0] * (dimension - 2)

        sim_identical = vjson.cosine_similarity(vec_a, vec_b)
        sim_orthogonal = vjson.cosine_similarity(vec_a, vec_c)
        print(f"✓ Cosine similarity (identical): {sim_identical:.6f}")
        print(f"✓ Cosine similarity (orthogonal): {sim_orthogonal:.6f}")
        assert abs(sim_identical - 1.0) < 0.001
        assert abs(sim_orthogonal) < 0.001

        # Dot product
        vec_d = [2.0, 3.0] + [0.0] * (dimension - 2)
        vec_e = [4.0, 5.0] + [0.0] * (dimension - 2)
        dot = vjson.dot_product(vec_d, vec_e)
        print(f"✓ Dot product: {dot:.6f}")
        assert abs(dot - 23.0) < 0.001  # 2*4 + 3*5 = 23

        # ========== TEST 5: Database Statistics ==========
        print("\n=== TEST 5: Database Statistics ===")

        stats = db.get_stats()
        print(f"✓ Database stats:")
        print(f"  Total vectors: {stats['total_vectors']}")
        print(f"  Dimension: {stats['dimension']}")
        print(f"  Active vectors: {stats['active_vectors']}")
        print(f"  Index size: {stats['index_size']}")
        print(f"  Metadata keys: {stats['metadata_keys']}")

        assert stats["total_vectors"] == 20
        assert stats["dimension"] == dimension
        assert stats["active_vectors"] == 20
        assert "index" in stats["metadata_keys"]
        assert "category" in stats["metadata_keys"]
        assert "score" in stats["metadata_keys"]

        # ========== TEST 6: Incremental Metadata Update ==========
        print("\n=== TEST 6: Incremental Metadata Update ===")

        # Update only metadata (fast operation)
        db.update_metadata("vec_0", {"index": 0, "updated": True, "new_field": "test"})

        updated_meta = db.get_metadata("vec_0")
        print(f"✓ Updated metadata: {updated_meta}")
        assert updated_meta["updated"] == True
        assert updated_meta["new_field"] == "test"
        assert updated_meta["index"] == 0

        # Update should fail for non-existent ID
        try:
            db.update_metadata("nonexistent", {"test": "fail"})
            assert False, "Should have raised error"
        except RuntimeError as e:
            print(f"✓ Update non-existent ID failed as expected: {str(e)[:50]}")

        # ========== TEST 7: Cosine Similarity with Normalized Vectors ==========
        print("\n=== TEST 7: Cosine Similarity Search (Normalized Vectors) ===")

        # Create a new database with normalized vectors
        temp_dir_norm = tempfile.mkdtemp()
        db_norm = vjson.PyVectorDB(temp_dir_norm, dimension=64)

        # Insert normalized vectors (for cosine similarity)
        for i in range(10):
            vec = [random.random() for _ in range(64)]
            normalized_vec = vjson.normalize_vector(vec)
            db_norm.insert(f"norm_vec_{i}", normalized_vec, {"index": i})

        # Search with normalized query
        query_unnorm = [random.random() for _ in range(64)]
        query_norm = vjson.normalize_vector(query_unnorm)

        results = db_norm.search(query_norm, k=3)
        print(f"✓ Search with normalized vectors: {len(results)} results")
        print(f"  Top result distance: {results[0]['distance']:.6f}")

        # For normalized vectors, L2 distance relates to cosine similarity
        # cosine_similarity ≈ 1 - (L2_distance^2 / 2)

        shutil.rmtree(temp_dir_norm)

        # ========== TEST 8: Statistics After Deletes ==========
        print("\n=== TEST 8: Statistics After Deletes ===")

        # Delete some vectors
        db.delete_batch(["vec_0", "vec_1", "vec_2"])

        stats_after_delete = db.get_stats()
        print(f"✓ Stats after delete:")
        print(f"  Total vectors (ID map): {stats_after_delete['total_vectors']}")
        print(f"  Active vectors: {stats_after_delete['active_vectors']}")
        print(f"  Index size: {stats_after_delete['index_size']}")

        assert stats_after_delete["active_vectors"] == 17  # 20 - 3
        assert stats_after_delete["total_vectors"] == 17

        # ========== TEST 9: Large Range Search ==========
        print("\n=== TEST 9: Large Range Search ===")

        # Insert more diverse vectors
        for i in range(20, 50):
            vec = [random.random() * 100 for _ in range(dimension)]
            db.insert(f"diverse_{i}", vec, {"type": "diverse"})

        # Range search with different thresholds
        query_diverse = [random.random() * 50 for _ in range(dimension)]

        for threshold in [10.0, 50.0, 100.0]:
            results = db.range_search(query_diverse, max_distance=threshold)
            print(f"✓ Range search (threshold={threshold}): {len(results)} results")
            if results:
                print(
                    f"  Closest: {results[0]['distance']:.2f}, Farthest: {results[-1]['distance']:.2f}"
                )

        # ========== TEST 10: Batch Operations Performance ==========
        print("\n=== TEST 10: Batch Operations Performance ===")

        # Create many IDs
        all_ids = [f"vec_{i}" for i in range(3, 20)] + [
            f"diverse_{i}" for i in range(20, 50)
        ]

        # Batch get metadata (should be faster than individual gets)
        batch_meta = db.get_metadata_batch(all_ids)
        print(f"✓ Batch metadata fetch: {len(batch_meta)} items")
        assert len(batch_meta) > 30

        # Batch get vectors
        batch_vecs = db.get_vectors_batch(all_ids[:10])
        print(f"✓ Batch vector fetch: {len(batch_vecs)} vectors")
        assert len(batch_vecs) == 10

        print("\n" + "=" * 70)
        print("ALL NEW FEATURES TESTS PASSED! ✓")
        print("=" * 70)
        print("\nFeatures tested:")
        print("  ✓ Batch get operations (metadata & vectors)")
        print("  ✓ Range search with distance threshold")
        print("  ✓ Vector normalization (single & batch)")
        print("  ✓ Cosine similarity computation")
        print("  ✓ Dot product computation")
        print("  ✓ Database statistics API")
        print("  ✓ Incremental metadata updates")
        print("  ✓ Normalized vector search for cosine similarity")
        print("  ✓ Statistics with deletes")
        print("  ✓ Large-scale range search")

    finally:
        shutil.rmtree(temp_dir)
        print(f"\nCleaned up: {temp_dir}")

        shutil.rmtree(temp_dir)
        print(f"\nCleaned up: {temp_dir}")

if __name__ == "__main__":
    test_all_new_features()
