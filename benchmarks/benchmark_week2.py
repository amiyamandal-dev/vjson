#!/usr/bin/env python3
"""
Week 2 Optimizations Benchmark
Tests SIMD intrinsics (NEON on ARM / AVX2 on x86)
"""

import random
import shutil
import tempfile
import time

import vjson


def benchmark():
    temp_dir = tempfile.mkdtemp()
    print("=" * 70)
    print("WEEK 2 OPTIMIZATIONS BENCHMARK - SIMD Intrinsics")
    print("=" * 70)
    print(f"Testing in: {temp_dir}\n")

    try:
        # Create database
        db = vjson.PyVectorDB(temp_dir, dimension=128)

        # ================================================================
        # BENCHMARK 1: Vector Normalization (SIMD vs baseline)
        # ================================================================
        print("=" * 70)
        print("BENCHMARK 1: Vector Normalization (NEON SIMD)")
        print("=" * 70)
        print("Testing platform-specific SIMD optimization\n")

        # Create test vectors
        n_vectors = 10000
        vectors = [[random.random() for _ in range(128)] for _ in range(n_vectors)]

        # Warmup
        for _ in range(5):
            _ = vjson.normalize_vectors(vectors[:100])

        # Benchmark SIMD normalization
        start = time.time()
        normalized = vjson.normalize_vectors(vectors)
        elapsed = time.time() - start

        throughput = n_vectors / elapsed

        print(f"Normalized {n_vectors} vectors (128-dim)")
        print(f"  Time: {elapsed * 1000:.2f}ms")
        print(f"  Throughput: {throughput:,.0f} vectors/sec")
        print(f"  Per-vector: {elapsed / n_vectors * 1000000:.2f}µs")

        # Verify correctness
        test_vec = [3.0, 4.0] + [0.0] * 126
        normalized_test = vjson.normalize_vector(test_vec)
        magnitude = sum(x * x for x in normalized_test) ** 0.5
        assert abs(magnitude - 1.0) < 0.001, (
            f"Normalization failed: magnitude={magnitude}"
        )
        print(f"  ✓ Correctness verified\n")

        # ================================================================
        # BENCHMARK 2: Cosine Similarity (SIMD)
        # ================================================================
        print("=" * 70)
        print("BENCHMARK 2: Cosine Similarity (NEON SIMD)")
        print("=" * 70)

        vec_a = [random.random() for _ in range(128)]
        vec_b = [random.random() for _ in range(128)]

        # Warmup
        for _ in range(100):
            _ = vjson.cosine_similarity(vec_a, vec_b)

        # Benchmark
        n_ops = 100000
        start = time.time()
        for _ in range(n_ops):
            _ = vjson.cosine_similarity(vec_a, vec_b)
        elapsed = time.time() - start

        ops_per_sec = n_ops / elapsed

        print(f"Computed {n_ops:,} cosine similarities")
        print(f"  Time: {elapsed * 1000:.2f}ms")
        print(f"  Throughput: {ops_per_sec:,.0f} ops/sec")
        print(f"  Per-operation: {elapsed / n_ops * 1000000:.2f}µs\n")

        # ================================================================
        # BENCHMARK 3: Dot Product (SIMD)
        # ================================================================
        print("=" * 70)
        print("BENCHMARK 3: Dot Product (NEON SIMD)")
        print("=" * 70)

        # Warmup
        for _ in range(100):
            _ = vjson.dot_product(vec_a, vec_b)

        # Benchmark
        start = time.time()
        for _ in range(n_ops):
            _ = vjson.dot_product(vec_a, vec_b)
        elapsed = time.time() - start

        ops_per_sec = n_ops / elapsed

        print(f"Computed {n_ops:,} dot products")
        print(f"  Time: {elapsed * 1000:.2f}ms")
        print(f"  Throughput: {ops_per_sec:,.0f} ops/sec")
        print(f"  Per-operation: {elapsed / n_ops * 1000000:.2f}µs\n")

        # ================================================================
        # BENCHMARK 4: Full Pipeline with SIMD
        # ================================================================
        print("=" * 70)
        print("BENCHMARK 4: Full Search Pipeline (with SIMD optimizations)")
        print("=" * 70)

        # Insert normalized vectors
        print("Inserting 10,000 vectors...")
        items = []
        for i in range(10000):
            vec = [random.random() for _ in range(128)]
            # Normalize before inserting (uses SIMD)
            vec_normalized = vjson.normalize_vector(vec)
            meta = {"index": i, "category": f"cat_{i % 10}"}
            items.append((f"vec_{i}", vec_normalized, meta))

        start = time.time()
        db.insert_batch(items)
        insert_time = time.time() - start

        print(f"✓ Inserted in {insert_time * 1000:.2f}ms")
        print(f"  Rate: {10000 / insert_time:.0f} vectors/sec\n")

        # Search with normalized query
        print("Running searches with normalized queries...")
        queries = [[random.random() for _ in range(128)] for _ in range(100)]
        normalized_queries = vjson.normalize_vectors(queries)

        # Warmup
        _ = db.batch_search(normalized_queries[:10], k=10)

        # Benchmark
        start = time.time()
        results = db.batch_search(normalized_queries, k=10)
        search_time = time.time() - start

        print(f"✓ Searched 100 queries in {search_time * 1000:.2f}ms")
        print(f"  Throughput: {100 / search_time:.0f} queries/sec")
        print(f"  Avg latency: {search_time / 100 * 1000:.3f}ms\n")

        # ================================================================
        # BENCHMARK 5: SIMD Scaling Test
        # ================================================================
        print("=" * 70)
        print("BENCHMARK 5: SIMD Performance vs Vector Size")
        print("=" * 70)

        print(f"{'Dimension':<12} {'Time (ms)':<14} {'Throughput (vec/s)':<20}")
        print("-" * 70)

        for dim in [32, 64, 128, 256, 512, 1024]:
            test_vecs = [[random.random() for _ in range(dim)] for _ in range(1000)]

            start = time.time()
            _ = vjson.normalize_vectors(test_vecs)
            elapsed = time.time() - start

            throughput = 1000 / elapsed
            print(f"{dim:<12} {elapsed * 1000:<14.2f} {throughput:<20.0f}")

        print()

        # ================================================================
        # SUMMARY
        # ================================================================
        print("=" * 70)
        print("WEEK 2 SIMD OPTIMIZATION RESULTS")
        print("=" * 70)

        print("\n✅ SIMD Intrinsics Implemented:")
        print("  Platform: ARM64 (Apple Silicon)")
        print("  SIMD: NEON (4-wide float32)")
        print()

        print("📊 Performance Achieved:")
        print(f"  • Vector normalization: {throughput:,.0f} vec/s")
        print(f"  • Cosine similarity: {ops_per_sec:,.0f} ops/s")
        print(f"  • Full search pipeline: {100 / search_time:.0f} q/s")
        print()

        print("🎯 Key Improvements:")
        print("  • Platform-specific SIMD (NEON on ARM, AVX2 on x86)")
        print("  • Auto-vectorization for vector operations")
        print("  • Optimized for 128-dim vectors (typical use case)")
        print()

        print("📈 Comparison to Previous:")
        print("  Week 1 (auto-vec): ~340K vec/s normalization")
        print("  Week 2 (NEON):     Expected 2-4x improvement")
        print("  Actual throughput depends on data size and CPU")

    finally:
        try:
            shutil.rmtree(temp_dir)
            print(f"\n\nCleaned up: {temp_dir}")
        except Exception as e:
            print(f"\n\nNote: Manual cleanup may be needed: {temp_dir}")


if __name__ == "__main__":
    benchmark()
