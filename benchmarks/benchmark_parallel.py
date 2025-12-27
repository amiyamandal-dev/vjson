#!/usr/bin/env python3
"""
Performance benchmark to verify parallelization improvements
"""

import random
import shutil
import tempfile
import time

import vjson


def benchmark():
    temp_dir = tempfile.mkdtemp()
    print(f"Benchmarking in: {temp_dir}\n")

    try:
        # Create database with large dataset
        db = vjson.PyVectorDB(temp_dir, dimension=128)

        # Benchmark 1: Insert batch with parallel metadata prep
        print("=" * 60)
        print("BENCHMARK 1: Insert Batch (parallel metadata prep)")
        print("=" * 60)

        n_vectors = 10000
        items = []
        for i in range(n_vectors):
            vec = [random.random() for _ in range(128)]
            meta = {
                "index": i,
                "category": f"cat_{i % 10}",
                "score": random.random() * 100,
                "name": f"vector_{i}",
            }
            items.append((f"vec_{i}", vec, meta))

        start = time.time()
        db.insert_batch(items)
        elapsed = time.time() - start

        print(f"✓ Inserted {n_vectors} vectors in {elapsed:.3f}s")
        print(f"  Rate: {n_vectors / elapsed:.0f} vectors/sec\n")

        # Benchmark 2: Batch get metadata (parallel)
        print("=" * 60)
        print("BENCHMARK 2: Get Metadata Batch (parallel)")
        print("=" * 60)

        # Get metadata for 5000 random IDs
        test_ids = [f"vec_{i}" for i in random.sample(range(n_vectors), 5000)]

        start = time.time()
        results = db.get_metadata_batch(test_ids)
        elapsed = time.time() - start

        print(f"✓ Fetched metadata for {len(results)} vectors in {elapsed:.3f}s")
        print(f"  Rate: {len(results) / elapsed:.0f} items/sec\n")

        # Benchmark 3: Batch get vectors (parallel)
        print("=" * 60)
        print("BENCHMARK 3: Get Vectors Batch (parallel)")
        print("=" * 60)

        # Get vectors for 1000 random IDs
        test_ids = [f"vec_{i}" for i in random.sample(range(n_vectors), 1000)]

        start = time.time()
        results = db.get_vectors_batch(test_ids)
        elapsed = time.time() - start

        print(f"✓ Fetched {len(results)} vectors in {elapsed:.3f}s")
        print(f"  Rate: {len(results) / elapsed:.0f} vectors/sec\n")

        # Benchmark 4: Search with parallel result mapping
        print("=" * 60)
        print("BENCHMARK 4: Search (parallel result mapping)")
        print("=" * 60)

        query = [random.random() for _ in range(128)]

        start = time.time()
        results = db.search(query, k=100)
        elapsed = time.time() - start

        print(f"✓ Search completed in {elapsed:.3f}s")
        print(f"  Found {len(results)} results\n")

        # Benchmark 5: Range search (parallel filtering)
        print("=" * 60)
        print("BENCHMARK 5: Range Search (parallel filtering)")
        print("=" * 60)

        start = time.time()
        results = db.range_search(query, max_distance=50.0)
        elapsed = time.time() - start

        print(f"✓ Range search completed in {elapsed:.3f}s")
        print(f"  Found {len(results)} results within distance threshold\n")

        # Benchmark 6: Vector normalization (parallel)
        print("=" * 60)
        print("BENCHMARK 6: Normalize Vectors (parallel)")
        print("=" * 60)

        # Create 5000 random vectors to normalize
        vectors = [[random.random() for _ in range(128)] for _ in range(5000)]

        start = time.time()
        normalized = vjson.normalize_vectors(vectors)
        elapsed = time.time() - start

        print(f"✓ Normalized {len(normalized)} vectors in {elapsed:.3f}s")
        print(f"  Rate: {len(normalized) / elapsed:.0f} vectors/sec\n")

        # Benchmark 7: Rebuild index (parallel filtering)
        print("=" * 60)
        print("BENCHMARK 7: Rebuild Index (parallel filtering)")
        print("=" * 60)

        # Delete some vectors to make rebuild meaningful
        delete_ids = [f"vec_{i}" for i in range(0, n_vectors, 10)]  # Every 10th
        db.delete_batch(delete_ids)

        start = time.time()
        db.rebuild_index()
        elapsed = time.time() - start

        print(f"✓ Rebuilt index in {elapsed:.3f}s")
        print(f"  Filtered out {len(delete_ids)} deleted vectors\n")

        print("=" * 60)
        print("BENCHMARK SUMMARY")
        print("=" * 60)
        print("\nAll parallelized operations completed successfully!")
        print("\nKey parallelizations tested:")
        print("  ✓ Insert batch metadata preparation")
        print("  ✓ Batch metadata retrieval")
        print("  ✓ Batch vector retrieval")
        print("  ✓ Search result mapping")
        print("  ✓ Range search filtering")
        print("  ✓ Vector normalization")
        print("  ✓ Index rebuild filtering")
        print("\nThese operations now use Rayon parallel iterators for")
        print("significant performance improvements on multi-core systems.")

    finally:
        try:
            shutil.rmtree(temp_dir)
            print(f"\nCleaned up: {temp_dir}")
        except:
            print(f"\nNote: Manual cleanup may be needed: {temp_dir}")


if __name__ == "__main__":
    benchmark()
