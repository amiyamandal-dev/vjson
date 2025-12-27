# Benchmarks

This directory contains performance benchmarks for the vjson vector database.

## Optimization Week Benchmarks

### Week 1: CPU Optimizations
- **benchmark_week1.py** - Comprehensive Week 1 benchmarks (SIMD, SmallVec, parallel operations)
- **benchmark_week1_fast.py** - Fast version of Week 1 benchmarks (smaller dataset, quicker to run)

### Week 2: Memory & I/O Optimizations
- **benchmark_week2.py** - Week 2 benchmarks (memory-mapped I/O, large buffers, parallel deserialization)

### Week 3: Concurrency Optimizations
- **benchmark_week3_concurrency.py** - Comprehensive concurrency tests (thread scaling, lock contention, batch operations)

## Component Benchmarks

- **benchmark_json.py** - JSON serialization/deserialization performance
- **benchmark_io.py** - I/O layer performance tests
- **benchmark_parallel.py** - Parallel search performance tests
- **gpu_comparison_template.py** - Template for comparing GPU vs CPU implementations

## Running Benchmarks

```bash
# Run a specific benchmark
python benchmarks/benchmark_week3_concurrency.py

# Run Week 1 fast benchmark (recommended for quick testing)
python benchmarks/benchmark_week1_fast.py

# Run all optimization benchmarks
python benchmarks/benchmark_week1.py
python benchmarks/benchmark_week2.py
python benchmarks/benchmark_week3_concurrency.py
```

## Performance Results

Final optimized performance (after all 3 weeks):
- Single-thread search: **1,414 queries/sec** (2.8x improvement)
- Batch search (4 threads): **5,569 queries/sec** (5.6x improvement)
- Concurrent scaling: **95% efficiency** up to 8 threads
- Memory usage: **0.08 MB** (20% reduction)
- P50 latency: **0.712ms** (2.8x faster)

See `../FINAL_OPTIMIZATION_REPORT.md` for complete results.
