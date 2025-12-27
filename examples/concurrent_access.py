"""
Demonstration of concurrent read/write capabilities
Shows parallel reads and locked writes
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import vjson


def main():
    print("=== VJson Concurrent Access Demo ===\n")

    # Create and populate database
    print("1. Creating and populating database with 10,000 vectors...")
    db = vjson.PyVectorDB(
        path="./concurrent_db", dimension=128, max_elements=100000, ef_construction=200
    )

    # Initial batch insert
    start = time.time()
    items = []
    for i in range(10000):
        vector = np.random.rand(128).tolist()
        metadata = {"batch": 0, "index": i}
        items.append((f"vec_{i}", vector, metadata))

    db.insert_batch(items)
    elapsed = time.time() - start
    print(f"   Populated in {elapsed:.3f}s\n")

    # Concurrent reads (parallel searches)
    print("2. Testing concurrent reads (parallel searches)...")
    print("   Running 100 searches across 8 threads...")

    search_count = 0
    search_lock = threading.Lock()

    def search_task(task_id):
        nonlocal search_count
        query = np.random.rand(128).tolist()
        results = db.search(query, k=10, ef_search=50)

        with search_lock:
            search_count += 1

        return (task_id, len(results))

    start = time.time()
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(search_task, i) for i in range(100)]
        results = [f.result() for f in as_completed(futures)]

    elapsed = time.time() - start
    print(f"   Completed {len(results)} searches in {elapsed:.3f}s")
    print(f"   Throughput: {len(results) / elapsed:.0f} searches/sec")
    print(f"   Average latency: {elapsed / len(results) * 1000:.2f}ms\n")

    # Batch search (even more efficient)
    print("3. Testing batch search (parallel processing)...")
    queries = [np.random.rand(128).tolist() for _ in range(100)]

    start = time.time()
    batch_results = db.batch_search(queries, k=10, ef_search=50)
    elapsed = time.time() - start

    print(f"   Completed 100 queries in {elapsed:.3f}s")
    print(f"   Throughput: {len(queries) / elapsed:.0f} searches/sec")
    print(f"   Average latency: {elapsed / len(queries) * 1000:.2f}ms\n")

    # Concurrent writes (serialized automatically)
    print("4. Testing concurrent writes (automatically serialized)...")
    print("   Running 100 inserts across 4 threads...")

    write_count = 0
    write_lock = threading.Lock()

    def insert_task(task_id):
        nonlocal write_count
        vector = np.random.rand(128).tolist()
        metadata = {"batch": 1, "thread": task_id}
        db.insert(f"new_vec_{task_id}", vector, metadata)

        with write_lock:
            write_count += 1

        return task_id

    start = time.time()
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(insert_task, i) for i in range(100)]
        results = [f.result() for f in as_completed(futures)]

    elapsed = time.time() - start
    print(f"   Completed {len(results)} inserts in {elapsed:.3f}s")
    print(f"   Throughput: {len(results) / elapsed:.0f} inserts/sec")
    print(f"   Note: Individual inserts are slow. Use insert_batch() for production!\n")

    # Batch write (recommended approach)
    print("5. Testing batch write (recommended for high throughput)...")
    print("   Inserting 1,000 vectors in a single batch...")

    items = []
    for i in range(1000):
        vector = np.random.rand(128).tolist()
        metadata = {"batch": 2, "index": i}
        items.append((f"batch_vec_{i}", vector, metadata))

    start = time.time()
    db.insert_batch(items)
    elapsed = time.time() - start

    print(f"   Completed in {elapsed:.3f}s")
    print(f"   Throughput: {len(items) / elapsed:.0f} inserts/sec")
    print(
        f"   Speed improvement: {(len(items) / elapsed) / (len(results) / elapsed):.1f}x faster\n"
    )

    # Mixed workload
    print("6. Testing mixed workload (reads + writes)...")
    print("   10 threads: 8 reading, 2 writing...")

    def mixed_task(task_id, is_writer):
        if is_writer:
            # Writer tasks
            for i in range(10):
                vector = np.random.rand(128).tolist()
                db.insert(f"mixed_w{task_id}_v{i}", vector, {"mixed": True})
            return f"Writer {task_id}: 10 inserts"
        else:
            # Reader tasks
            count = 0
            for i in range(50):
                query = np.random.rand(128).tolist()
                db.search(query, k=5, ef_search=50)
                count += 1
            return f"Reader {task_id}: {count} searches"

    start = time.time()
    with ThreadPoolExecutor(max_workers=10) as executor:
        # 8 readers, 2 writers
        futures = []
        for i in range(8):
            futures.append(executor.submit(mixed_task, i, False))
        for i in range(2):
            futures.append(executor.submit(mixed_task, i, True))

        results = [f.result() for f in as_completed(futures)]

    elapsed = time.time() - start
    print(f"   Completed mixed workload in {elapsed:.3f}s")
    for result in results:
        print(f"   - {result}")

    print(f"\n7. Final database state:")
    print(f"   Total vectors: {len(db)}")

    print("\n=== Demo Complete! ===")
    print("\nKey Takeaways:")
    print("  ✓ Multiple readers can search concurrently (no blocking)")
    print("  ✓ Writes are automatically serialized (thread-safe)")
    print("  ✓ batch_search() is optimized for parallel execution")
    print("  ✓ insert_batch() is 10-50x faster than individual inserts")


if __name__ == "__main__":
    main()
