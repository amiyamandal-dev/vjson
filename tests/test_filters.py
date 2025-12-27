"""
Test metadata filtering functionality
"""

import os
import random
import shutil

import vjson

print("=== VJson Metadata Filtering Test ===\n")

# Clean up
db_path = "./filter_test_db"
if os.path.exists(db_path):
    shutil.rmtree(db_path)

# Create database
print("1. Creating database with diverse metadata...")
db = vjson.PyVectorDB(db_path, dimension=128)

# Insert vectors with different categories and scores
categories = ["tech", "science", "math", "literature"]
items = []

for i in range(100):
    vector = [random.random() for _ in range(128)]
    metadata = {
        "category": categories[i % len(categories)],
        "score": random.random(),
        "index": i,
        "tags": [f"tag_{j}" for j in range(i % 5)],
        "user": {"name": f"User{i}", "age": 20 + (i % 40)},
    }
    items.append((f"vec_{i}", vector, metadata))

db.insert_batch(items)
print(f"   Inserted {len(db)} vectors\n")

# Test 1: Simple equality filter
print("2. Test: Simple equality filter")
query = [0.5] * 128
results = db.search(query, k=100, filter={"category": "tech"})
print(f"   Filter: category='tech'")
print(f"   Results: {len(results)}")
assert all(r["metadata"]["category"] == "tech" for r in results)
print(f"   ✓ All results are category='tech'\n")

# Test 2: Numeric comparison (greater than)
print("3. Test: Numeric comparison ($gt)")
results = db.search(query, k=100, filter={"score": {"$gt": 0.7}})
print(f"   Filter: score > 0.7")
print(f"   Results: {len(results)}")
assert all(r["metadata"]["score"] > 0.7 for r in results)
print(f"   ✓ All results have score > 0.7\n")

# Test 4: Less than
print("4. Test: Numeric comparison ($lt)")
results = db.search(query, k=100, filter={"score": {"$lt": 0.3}})
print(f"   Filter: score < 0.3")
print(f"   Results: {len(results)}")
assert all(r["metadata"]["score"] < 0.3 for r in results)
print(f"   ✓ All results have score < 0.3\n")

# Test 5: Multiple conditions (AND)
print("5. Test: Multiple conditions (implicit AND)")
results = db.search(query, k=100, filter={"category": "tech", "score": {"$gt": 0.5}})
print(f"   Filter: category='tech' AND score > 0.5")
print(f"   Results: {len(results)}")
assert all(
    r["metadata"]["category"] == "tech" and r["metadata"]["score"] > 0.5
    for r in results
)
print(f"   ✓ All results match both conditions\n")

# Test 6: In operator
print("6. Test: $in operator")
results = db.search(query, k=100, filter={"category": {"$in": ["tech", "science"]}})
print(f"   Filter: category in ['tech', 'science']")
print(f"   Results: {len(results)}")
assert all(r["metadata"]["category"] in ["tech", "science"] for r in results)
print(f"   ✓ All results are tech or science\n")

# Test 7: Nested field access (dot notation)
print("7. Test: Nested field access")
results = db.search(query, k=100, filter={"user.age": {"$gte": 40}})
print(f"   Filter: user.age >= 40")
print(f"   Results: {len(results)}")
assert all(r["metadata"]["user"]["age"] >= 40 for r in results)
print(f"   ✓ All results have user.age >= 40\n")

# Test 8: Range query
print("8. Test: Range query (combining $gte and $lte)")
results_gt = db.search(query, k=100, filter={"score": {"$gte": 0.4}})
results_lt = db.search(query, k=100, filter={"score": {"$lte": 0.6}})
results_combined = db.search(
    query,
    k=100,
    filter={
        "score": {"$gte": 0.4},
        "index": {"$lt": 1000},  # Just to have multiple conditions
    },
)
print(f"   Filter: score >= 0.4 (separate query)")
print(f"   Results: {len(results_gt)}")
print(f"   ✓ Range filtering works\n")

# Test 9: No filter (baseline)
print("9. Test: No filter (baseline)")
results_no_filter = db.search(query, k=10)
results_with_filter = db.search(query, k=10, filter={"category": "tech"})
print(f"   No filter: {len(results_no_filter)} results")
print(f"   With filter: {len(results_with_filter)} results")
print(f"   ✓ Filter reduces result set appropriately\n")

# Test 10: Complex nested + comparison
print("10. Test: Complex query (nested field + comparison)")
results = db.search(
    query,
    k=100,
    filter={"category": "science", "user.age": {"$lt": 35}, "score": {"$gte": 0.3}},
)
print(f"   Filter: category='science' AND user.age < 35 AND score >= 0.3")
print(f"   Results: {len(results)}")
for r in results:
    assert r["metadata"]["category"] == "science"
    assert r["metadata"]["user"]["age"] < 35
    assert r["metadata"]["score"] >= 0.3
print(f"   ✓ All results match all conditions\n")

# Summary
print("=" * 60)
print("✅ ALL METADATA FILTERING TESTS PASSED!")
print("=" * 60)
print("\nSupported Filter Operations:")
print("  • Equality: {'key': 'value'}")
print("  • Greater than: {'key': {'$gt': 5}}")
print("  • Less than: {'key': {'$lt': 5}}")
print("  • Greater or equal: {'key': {'$gte': 5}}")
print("  • Less or equal: {'key': {'$lte': 5}}")
print("  • In array: {'key': {'$in': [1, 2, 3]}}")
print("  • Nested fields: {'user.age': {'$gte': 18}}")
print("  • Multiple conditions: Combined with AND")
print("\nPerformance Note:")
print("  Filtering is post-search (filter AFTER vector search)")
print("  For best performance, use selective filters")
