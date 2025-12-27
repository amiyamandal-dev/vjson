#!/usr/bin/env python3
"""Final deployment verification"""

import vjson
import sys

print("=" * 60)
print("VJSON FINAL DEPLOYMENT VERIFICATION")
print("=" * 60)

tests_passed = []
tests_failed = []

# Test 1: Import
try:
    import vjson
    print("✓ Import successful")
    tests_passed.append("Import")
except Exception as e:
    print(f"✗ Import failed: {e}")
    tests_failed.append("Import")
    sys.exit(1)

# Test 2: Create database
try:
    db = vjson.PyVectorDB("./final_test_db", dimension=128)
    print("✓ Database creation successful")
    tests_passed.append("Create DB")
except Exception as e:
    print(f"✗ Database creation failed: {e}")
    tests_failed.append("Create DB")

# Test 3: Insert
try:
    db.insert("vec1", [0.1] * 128, {"test": "data"})
    print("✓ Insert successful")
    tests_passed.append("Insert")
except Exception as e:
    print(f"✗ Insert failed: {e}")
    tests_failed.append("Insert")

# Test 4: Search
try:
    results = db.search([0.1] * 128, k=1)
    assert len(results) == 1
    print("✓ Search successful")
    tests_passed.append("Search")
except Exception as e:
    print(f"✗ Search failed: {e}")
    tests_failed.append("Search")

# Test 5: Batch insert
try:
    batch = [(f"vec{i}", [float(i)] * 128, {"i": i}) for i in range(10)]
    db.insert_batch(batch)
    print("✓ Batch insert successful")
    tests_passed.append("Batch Insert")
except Exception as e:
    print(f"✗ Batch insert failed: {e}")
    tests_failed.append("Batch Insert")

# Test 6: Filter
try:
    results = db.search([0.1] * 128, k=10, filter={"i": {"$gte": 5}})
    assert len(results) > 0
    print("✓ Filter search successful")
    tests_passed.append("Filter")
except Exception as e:
    print(f"✗ Filter failed: {e}")
    tests_failed.append("Filter")

# Test 7: Size
try:
    size = len(db)
    assert size == 11  # 1 + 10
    print(f"✓ Database size correct: {size}")
    tests_passed.append("Size")
except Exception as e:
    print(f"✗ Size check failed: {e}")
    tests_failed.append("Size")

# Cleanup
import shutil
shutil.rmtree("./final_test_db", ignore_errors=True)

print("\n" + "=" * 60)
print(f"RESULTS: {len(tests_passed)}/{len(tests_passed) + len(tests_failed)} tests passed")
print("=" * 60)

if tests_failed:
    print(f"\n✗ FAILED: {', '.join(tests_failed)}")
    sys.exit(1)
else:
    print("\n✅ ALL TESTS PASSED - DEPLOYMENT VERIFIED")
    print("\nOptimized storage layer features:")
    print("  • Memory-mapped I/O for vectors")
    print("  • Large buffers (1MB) for metadata")
    print("  • Parallel deserialization with Rayon")
    print("  • Atomic file operations (crash-safe)")
    print("  • Lock-free vector counting")
    print("  • Thread-safe with parking_lot::RwLock")
    sys.exit(0)
