#!/usr/bin/env python3
"""Test CRUD operations (Create, Read, Update, Delete)"""

import random
import shutil
import tempfile

import vjson


def test_crud_operations():
    """Test complete CRUD lifecycle"""

    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    print(f"Testing in: {temp_dir}")

    try:
        # Create database
        dimension = 128
        db = vjson.PyVectorDB(temp_dir, dimension=dimension)

        print("\n=== CREATE Operations ===")
        # Create vectors
        vector1 = [random.random() for _ in range(dimension)]
        vector2 = [random.random() for _ in range(dimension)]
        vector3 = [random.random() for _ in range(dimension)]

        db.insert("vec1", vector1, {"name": "Vector 1", "score": 10})
        db.insert("vec2", vector2, {"name": "Vector 2", "score": 20})
        db.insert("vec3", vector3, {"name": "Vector 3", "score": 30})

        print(f"✓ Inserted 3 vectors, total: {len(db)}")

        print("\n=== READ Operations ===")
        # Check if vectors exist
        assert db.contains("vec1") == True
        assert db.contains("vec2") == True
        assert db.contains("vec3") == True
        assert db.contains("nonexistent") == False
        print("✓ Contains check passed")

        # Get metadata
        meta1 = db.get_metadata("vec1")
        print(f"✓ Metadata for vec1: {meta1}")
        assert meta1["name"] == "Vector 1"
        assert meta1["score"] == 10

        # Get vector
        retrieved_vec1 = db.get_vector("vec1")
        print(f"✓ Retrieved vec1: {len(retrieved_vec1)} dimensions")
        assert len(retrieved_vec1) == dimension

        # Search
        query = vector1  # Should find vec1 as closest
        results = db.search(query, k=3)
        print(f"✓ Search found {len(results)} results")
        print(
            f"  Closest: {results[0]['id']} with distance {results[0]['distance']:.6f}"
        )
        assert results[0]["id"] == "vec1"

        print("\n=== UPDATE Operations ===")
        # Update metadata only
        new_vector1 = [random.random() for _ in range(dimension)]
        db.update("vec1", new_vector1, {"name": "Updated Vector 1", "score": 100})

        updated_meta = db.get_metadata("vec1")
        print(f"✓ Updated metadata: {updated_meta}")
        assert updated_meta["name"] == "Updated Vector 1"
        assert updated_meta["score"] == 100

        # Update with text
        db.insert_with_text("vec4", vector1, "machine learning", {"category": "ML"})
        print(f"✓ Inserted vec4 with text, total: {len(db)}")

        db.update_with_text(
            "vec4", vector1, "deep learning neural networks", {"category": "DL"}
        )
        updated_meta4 = db.get_metadata("vec4")
        print(f"✓ Updated vec4 with text: {updated_meta4}")
        assert updated_meta4["category"] == "DL"

        # Try to update non-existent vector (should fail)
        try:
            db.update("nonexistent", vector1, {"test": "fail"})
            assert False, "Should have raised error"
        except vjson.NotFoundError as e:
            print(f"✓ Update non-existent vector failed as expected: {str(e)[:50]}")

        print("\n=== DELETE Operations ===")
        # Delete single vector
        db.delete("vec2")
        print(f"✓ Deleted vec2 (marked as deleted in metadata)")
        assert db.contains("vec2") == False
        # Note: HNSW index size doesn't change until rebuild
        print(f"  HNSW index size: {len(db)} (unchanged until rebuild)")

        # Try to delete non-existent (should fail)
        try:
            db.delete("nonexistent")
            assert False, "Should have raised error"
        except vjson.NotFoundError as e:
            print(f"✓ Delete non-existent vector failed as expected: {str(e)[:50]}")

        # Batch delete
        db.delete_batch(["vec1", "vec3"])
        print(f"✓ Batch deleted vec1 and vec3")
        assert db.contains("vec1") == False
        assert db.contains("vec3") == False
        assert db.contains("vec4") == True

        print("\n=== REBUILD INDEX ===")
        # Add more vectors
        for i in range(10):
            vec = [random.random() for _ in range(dimension)]
            db.insert(f"rebuild_vec{i}", vec, {"index": i})

        print(f"✓ Added 10 more vectors, total: {len(db)}")

        # Delete some
        db.delete_batch([f"rebuild_vec{i}" for i in range(0, 5)])
        print(f"✓ Deleted 5 vectors, remaining: {len(db)}")

        # Rebuild index to reclaim space
        db.rebuild_index()
        print("✓ Index rebuilt successfully")

        # Verify search still works
        query = [random.random() for _ in range(dimension)]
        results = db.search(query, k=3)
        print(f"✓ Search after rebuild: {len(results)} results")

        print("\n=== CLEAR Operation ===")
        db.clear()
        print(f"✓ Database cleared, size: {len(db)}")
        assert len(db) == 0
        assert db.is_empty() == True

        print("\n" + "=" * 50)
        print("ALL CRUD TESTS PASSED! ✓")
        print("=" * 50)

    finally:
        # Cleanup
        shutil.rmtree(temp_dir)
        print(f"\nCleaned up: {temp_dir}")


if __name__ == "__main__":
    test_crud_operations()
