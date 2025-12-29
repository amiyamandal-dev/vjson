#!/usr/bin/env python3
"""Test advanced metadata filtering"""

import os
import random
import shutil
import tempfile

import vjson

def test_advanced_filters():
    """Test all advanced filter operators"""

    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    print(f"Testing advanced filters in: {temp_dir}")

    try:
        dimension = 64
        db = vjson.PyVectorDB(temp_dir, dimension=dimension)

        print("\n=== Setup: Insert test data ===")
        # Insert diverse test data
        test_data = [
            (
                "user1",
                {
                    "name": "Alice",
                    "age": 25,
                    "city": "New York",
                    "score": 0.85,
                    "tags": ["python", "rust"],
                },
            ),
            (
                "user2",
                {
                    "name": "Bob",
                    "age": 30,
                    "city": "San Francisco",
                    "score": 0.92,
                    "tags": ["java", "go"],
                },
            ),
            (
                "user3",
                {
                    "name": "Charlie",
                    "age": 35,
                    "city": "New York",
                    "score": 0.78,
                    "tags": ["python", "javascript"],
                },
            ),
            (
                "user4",
                {
                    "name": "Diana",
                    "age": 28,
                    "city": "Boston",
                    "score": 0.88,
                    "tags": ["rust", "c++"],
                },
            ),
            (
                "user5",
                {
                    "name": "Eve",
                    "age": 40,
                    "city": "Seattle",
                    "score": 0.95,
                    "tags": ["python", "rust", "go"],
                },
            ),
            (
                "prod1",
                {
                    "name": "Product A",
                    "category": "electronics",
                    "price": 299.99,
                    "stock": 50,
                },
            ),
            (
                "prod2",
                {
                    "name": "Product B",
                    "category": "books",
                    "price": 19.99,
                    "stock": 100,
                },
            ),
            (
                "prod3",
                {
                    "name": "Product C",
                    "category": "electronics",
                    "price": 499.99,
                    "stock": 25,
                },
            ),
        ]

        for id, metadata in test_data:
            vector = [random.random() for _ in range(dimension)]
            db.insert(id, vector, metadata)

        print(f"✓ Inserted {len(db)} items")

        # Test query vector
        query = [random.random() for _ in range(dimension)]

        print("\n=== Test 1: Basic Equality ===")
        results = db.search(query, k=10, filter={"city": "New York"})
        print(f"✓ City = 'New York': {[r['id'] for r in results]}")
        assert len(results) == 2
        assert all(r["metadata"]["city"] == "New York" for r in results)

        print("\n=== Test 2: Not Equals ($ne) ===")
        results = db.search(query, k=10, filter={"city": {"$ne": "New York"}})
        print(f"✓ City != 'New York': {[r['id'] for r in results]}")
        assert all(r["metadata"].get("city") != "New York" for r in results)

        print("\n=== Test 3: Greater Than ($gt) ===")
        results = db.search(query, k=10, filter={"age": {"$gt": 30}})
        print(f"✓ Age > 30: {[r['id'] for r in results]}")
        assert len(results) == 2  # Charlie (35) and Eve (40)

        print("\n=== Test 4: Between ($between) ===")
        results = db.search(query, k=10, filter={"age": {"$between": [25, 30]}})
        print(f"✓ Age between 25-30: {[r['id'] for r in results]}")
        assert len(results) == 3  # Alice (25), Bob (30), Diana (28)

        print("\n=== Test 5: String StartsWith ($startsWith) ===")
        results = db.search(query, k=10, filter={"name": {"$startsWith": "Prod"}})
        print(f"✓ Name starts with 'Prod': {[r['id'] for r in results]}")
        assert len(results) == 3  # All products

        print("\n=== Test 6: String EndsWith ($endsWith) ===")
        results = db.search(query, k=10, filter={"city": {"$endsWith": "York"}})
        print(f"✓ City ends with 'York': {[r['id'] for r in results]}")
        assert all("York" in r["metadata"].get("city", "") for r in results)

        print("\n=== Test 7: String Contains ($contains) ===")
        results = db.search(query, k=10, filter={"city": {"$contains": "San"}})
        print(f"✓ City contains 'San': {[r['id'] for r in results]}")
        assert any("San" in r["metadata"].get("city", "") for r in results)

        print("\n=== Test 8: Regex Pattern ($regex) ===")
        results = db.search(query, k=10, filter={"name": {"$regex": "^[A-D]"}})
        print(f"✓ Name matches regex '^[A-D]': {[r['id'] for r in results]}")
        # Should match Alice, Bob, Charlie, Diana
        assert len(results) >= 4

        print("\n=== Test 9: In Array ($in) ===")
        results = db.search(
            query, k=10, filter={"category": {"$in": ["electronics", "books"]}}
        )
        print(f"✓ Category in ['electronics', 'books']: {[r['id'] for r in results]}")
        assert len(results) == 3  # All products

        print("\n=== Test 10: Combined Filters (AND) ===")
        results = db.search(
            query, k=10, filter={"city": "New York", "age": {"$gte": 30}}
        )
        print(f"✓ City='New York' AND Age>=30: {[r['id'] for r in results]}")
        assert len(results) == 1  # Only Charlie
        assert results[0]["id"] == "user3"

        print("\n=== Test 11: Price Range Filter ===")
        results = db.search(query, k=10, filter={"price": {"$between": [20, 300]}})
        print(f"✓ Price between 20-300: {[r['id'] for r in results]}")
        assert len(results) == 1  # Only Product A (299.99)

        print("\n=== Test 12: Score Greater or Equal ===")
        results = db.search(query, k=10, filter={"score": {"$gte": 0.90}})
        print(f"✓ Score >= 0.90: {[r['id'] for r in results]}")
        assert len(results) == 2  # Bob (0.92) and Eve (0.95)

        print("\n=== Test 13: Exists Filter ===")
        results = db.search(query, k=10, filter={"age": {"$exists": True}})
        print(f"✓ Has 'age' field: {[r['id'] for r in results]}")
        assert len(results) == 5  # All users

        results = db.search(query, k=10, filter={"age": {"$exists": False}})
        print(f"✓ No 'age' field: {[r['id'] for r in results]}")
        assert len(results) == 3  # All products

        print("\n=== Test 14: Complex Combined Filter ===")
        results = db.search(
            query,
            k=10,
            filter={
                "name": {"$startsWith": "Product"},
                "price": {"$gt": 100},
                "stock": {"$lte": 50},
            },
        )
        print(f"✓ Complex product filter: {[r['id'] for r in results]}")
        assert len(results) == 2  # Product A and C

        print("\n=== Test 15: Case-sensitive String Filters ===")
        results = db.search(query, k=10, filter={"name": {"$contains": "alice"}})
        print(f"✓ Name contains 'alice' (lowercase): {[r['id'] for r in results]}")
        assert len(results) == 0  # Case-sensitive, should not match "Alice"

        results = db.search(query, k=10, filter={"name": {"$contains": "Alice"}})
        print(f"✓ Name contains 'Alice' (correct case): {[r['id'] for r in results]}")
        assert len(results) == 1

        print("\n" + "=" * 60)
        print("ALL ADVANCED FILTER TESTS PASSED! ✓")
        print("=" * 60)
        print("\nSupported operators:")
        print("  ✓ $eq (equals - default)")
        print("  ✓ $ne (not equals)")
        print("  ✓ $gt (greater than)")
        print("  ✓ $gte (greater than or equal)")
        print("  ✓ $lt (less than)")
        print("  ✓ $lte (less than or equal)")
        print("  ✓ $between (range, inclusive)")
        print("  ✓ $in (value in array)")
        print("  ✓ $nin (value not in array)")
        print("  ✓ $exists (field exists)")
        print("  ✓ $startsWith (string prefix)")
        print("  ✓ $endsWith (string suffix)")
        print("  ✓ $contains (substring)")
        print("  ✓ $regex (regular expression)")
        print("  ✓ Multiple filters (implicit AND)")

    finally:
        # Cleanup - ignore errors from locked Tantivy index files
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
            print(f"\nCleaned up: {temp_dir}")

if __name__ == "__main__":
    test_advanced_filters()
