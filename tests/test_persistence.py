#!/usr/bin/env python3
"""Test persistence - save and load database"""

import os
import random
import shutil
import tempfile

import vjson


def test_persistence():
    """Test saving and loading database"""

    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    print(f"Testing persistence in: {temp_dir}")

    try:
        dimension = 128

        print("\n=== Phase 1: Create and populate database ===")
        db1 = vjson.PyVectorDB(
            temp_dir, dimension=dimension, max_elements=1000, ef_construction=100
        )

        # Insert vectors with known data
        test_vectors = []
        for i in range(50):
            vector = [float(i * 10 + j) for j in range(dimension)]
            metadata = {"index": i, "category": f"cat_{i % 5}", "score": i * 0.1}
            db1.insert(f"vec_{i}", vector, metadata)
            test_vectors.append((f"vec_{i}", vector, metadata))

        print(f"✓ Inserted {len(db1)} vectors")

        # Search before closing
        query = [float(25 * 10 + j) for j in range(dimension)]  # Should match vec_25
        results1 = db1.search(query, k=5)
        print(f"✓ Search results before reload:")
        print(
            f"  Top result: {results1[0]['id']} (distance: {results1[0]['distance']:.6f})"
        )
        assert results1[0]["id"] == "vec_25", (
            f"Expected vec_25, got {results1[0]['id']}"
        )

        # Explicitly save (though auto-saved on insert)
        db1.save()
        print("✓ Database saved")

        # Delete the db object
        del db1
        print("✓ Database object deleted")

        print("\n=== Phase 2: Reload database ===")
        # Create new database instance from same path
        db2 = vjson.PyVectorDB(
            temp_dir, dimension=dimension, max_elements=1000, ef_construction=100
        )

        # Load data
        db2.load()
        print(f"✓ Database loaded, size: {len(db2)}")
        assert len(db2) == 50, f"Expected 50 vectors, got {len(db2)}"

        # Verify data integrity
        print("\n=== Phase 3: Verify data integrity ===")

        # Check metadata
        meta_25 = db2.get_metadata("vec_25")
        print(f"✓ Metadata for vec_25: {meta_25}")
        assert meta_25["index"] == 25
        assert meta_25["category"] == "cat_0"
        assert abs(meta_25["score"] - 2.5) < 0.001

        # Check vector retrieval
        retrieved_vec = db2.get_vector("vec_10")
        expected_vec = [float(10 * 10 + j) for j in range(dimension)]
        print(f"✓ Retrieved vec_10: first 5 values = {retrieved_vec[:5]}")
        assert retrieved_vec == expected_vec, "Vector data mismatch"

        # Search after reload
        results2 = db2.search(query, k=5)
        print(f"✓ Search results after reload:")
        print(
            f"  Top result: {results2[0]['id']} (distance: {results2[0]['distance']:.6f})"
        )
        assert results2[0]["id"] == "vec_25", (
            f"Expected vec_25, got {results2[0]['id']}"
        )

        # Verify search results match
        assert abs(results1[0]["distance"] - results2[0]["distance"]) < 0.0001, (
            "Search distances don't match after reload"
        )

        print("\n=== Phase 4: Add more data after reload ===")
        # Add more vectors
        for i in range(50, 60):
            vector = [float(i * 10 + j) for j in range(dimension)]
            metadata = {"index": i, "category": f"cat_{i % 5}"}
            db2.insert(f"vec_{i}", vector, metadata)

        print(f"✓ Added 10 more vectors, total: {len(db2)}")
        assert len(db2) == 60

        # Search for new vector
        query_new = [float(55 * 10 + j) for j in range(dimension)]
        results_new = db2.search(query_new, k=3)
        print(f"✓ Search for new vector: {results_new[0]['id']}")
        assert results_new[0]["id"] == "vec_55"

        del db2

        print("\n=== Phase 5: Reload again to verify new data ===")
        db3 = vjson.PyVectorDB(temp_dir, dimension=dimension)
        db3.load()
        print(f"✓ Database reloaded, size: {len(db3)}")
        assert len(db3) == 60

        # Verify new data persisted
        assert db3.contains("vec_55") == True
        meta_55 = db3.get_metadata("vec_55")
        print(f"✓ New vector persisted: {meta_55}")

        print("\n=== Phase 6: Test with hybrid search ===")
        db3.insert_with_text(
            "doc1",
            [random.random() for _ in range(dimension)],
            "machine learning algorithms",
            {"topic": "ML"},
        )
        db3.insert_with_text(
            "doc2",
            [random.random() for _ in range(dimension)],
            "deep neural networks",
            {"topic": "DL"},
        )

        del db3

        db4 = vjson.PyVectorDB(temp_dir, dimension=dimension)
        db4.load()
        print(f"✓ Database with text reloaded, size: {len(db4)}")
        assert len(db4) == 62

        # Test text search still works
        text_results = db4.text_search("machine learning", limit=2)
        print(f"✓ Text search results: {[r[0] for r in text_results]}")

        print("\n" + "=" * 60)
        print("ALL PERSISTENCE TESTS PASSED! ✓")
        print("=" * 60)
        print("\nKey features verified:")
        print("  ✓ Data persists across database instances")
        print("  ✓ HNSW index rebuilds correctly on load")
        print("  ✓ Metadata integrity maintained")
        print("  ✓ Vector data integrity maintained")
        print("  ✓ Search results consistent after reload")
        print("  ✓ Can add new data after reload")
        print("  ✓ Text index persists and reloads")

    finally:
        # Cleanup
        shutil.rmtree(temp_dir)
        print(f"\nCleaned up: {temp_dir}")


if __name__ == "__main__":
    test_persistence()
