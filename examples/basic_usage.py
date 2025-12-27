"""
Basic usage example for vjson vector database
"""

import vjson
import numpy as np
import time

def main():
    print("=== VJson Vector Database - Basic Usage ===\n")

    # Create a vector database
    print("1. Creating vector database...")
    db = vjson.PyVectorDB(
        path="./example_db",
        dimension=128,
        max_elements=10000,
        ef_construction=200
    )

    # Insert single vector
    print("2. Inserting single vector...")
    db.insert(
        id="vec_1",
        vector=np.random.rand(128).tolist(),
        metadata={"type": "example", "index": 1}
    )

    # Batch insert (much faster)
    print("3. Batch inserting 1000 vectors...")
    start = time.time()
    items = []
    for i in range(2, 1002):
        vector = np.random.rand(128).tolist()
        metadata = {"type": "batch", "index": i}
        items.append((f"vec_{i}", vector, metadata))

    db.insert_batch(items)
    elapsed = time.time() - start
    print(f"   Inserted 1000 vectors in {elapsed:.3f}s ({1000/elapsed:.0f} vectors/sec)")

    # Search
    print("\n4. Searching for nearest neighbors...")
    query = np.random.rand(128).tolist()
    results = db.search(query, k=5, ef_search=50)

    print(f"   Found {len(results)} results:")
    for i, result in enumerate(results, 1):
        print(f"   {i}. ID: {result['id']}, Distance: {result['distance']:.4f}")

    # Get metadata
    print("\n5. Getting metadata for a specific vector...")
    metadata = db.get_metadata("vec_1")
    print(f"   Metadata: {metadata}")

    # Database info
    print(f"\n6. Database info:")
    print(f"   Total vectors: {len(db)}")
    print(f"   Is empty: {db.is_empty()}")

    print("\n=== Done! ===")

if __name__ == "__main__":
    main()
