"""
Simple test without external dependencies
"""

import random

import vjson

print("=== VJson Vector Database - Simple Test ===\n")

# Create a vector database
print("1. Creating vector database...")
db = vjson.PyVectorDB(
    path="./simple_test_db", dimension=128, max_elements=10000, ef_construction=200
)
print(f"   Created: {db}")

# Insert single vector
print("\n2. Inserting single vector...")
vector1 = [random.random() for _ in range(128)]
db.insert(id="vec_1", vector=vector1, metadata={"type": "example", "index": 1})
print(f"   Inserted 1 vector. Total: {len(db)}")

# Batch insert
print("\n3. Batch inserting 100 vectors...")
items = []
for i in range(2, 102):
    vector = [random.random() for _ in range(128)]
    metadata = {"type": "batch", "index": i}
    items.append((f"vec_{i}", vector, metadata))

db.insert_batch(items)
print(f"   Inserted 100 vectors. Total: {len(db)}")

# Search
print("\n4. Searching for nearest neighbors...")
query = [random.random() for _ in range(128)]
results = db.search(query, k=5, ef_search=50)

print(f"   Found {len(results)} results:")
for i, result in enumerate(results, 1):
    print(f"   {i}. ID: {result['id']}, Distance: {result['distance']:.4f}")
    print(f"      Metadata: {result['metadata']}")

# Batch search
print("\n5. Batch search with 3 queries...")
queries = [[random.random() for _ in range(128)] for _ in range(3)]
batch_results = db.batch_search(queries, k=3, ef_search=50)

print(f"   Completed {len(batch_results)} searches")
for i, results in enumerate(batch_results):
    print(f"   Query {i + 1}: {len(results)} results")

# Get metadata
print("\n6. Getting metadata for a specific vector...")
metadata = db.get_metadata("vec_1")
print(f"   Metadata: {metadata}")

# Database info
print(f"\n7. Database info:")
print(f"   Total vectors: {len(db)}")
print(f"   Is empty: {db.is_empty()}")

print("\n=== Test Passed! ===")
