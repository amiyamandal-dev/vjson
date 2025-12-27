# Project Structure

This document describes the organization of the vjson vector database codebase.

## Directory Layout

```
vjson/
├── src/                        # Rust source code
│   ├── lib.rs                 # Main library entry point (Python bindings)
│   ├── vectordb.rs            # Main VectorDB implementation
│   ├── index.rs               # HNSW index wrapper
│   ├── storage.rs             # Storage layer (memory-mapped I/O)
│   ├── filter.rs              # Metadata filtering system
│   ├── hybrid.rs              # Hybrid search (vector + text)
│   ├── tantivy_index.rs       # Full-text search integration
│   ├── simd.rs                # SIMD optimizations (NEON, AVX2, SSE)
│   ├── utils.rs               # Utility functions
│   └── error.rs               # Error types
│
├── tests/                      # Python integration tests
│   ├── README.md              # Test documentation
│   ├── test_simple.py         # Basic functionality
│   ├── test_crud.py           # CRUD operations
│   ├── test_persistence.py    # Data persistence
│   ├── test_filters.py        # Basic filters
│   ├── test_advanced_filters.py  # Advanced filter operators
│   ├── test_hybrid_search.py  # Hybrid search tests
│   ├── test_concurrent.py     # Concurrency tests
│   ├── test_new_features.py   # New features
│   └── final_verification.py  # Complete verification
│
├── benchmarks/                 # Performance benchmarks
│   ├── README.md              # Benchmark documentation
│   ├── benchmark_week1.py     # Week 1: CPU optimizations
│   ├── benchmark_week1_fast.py  # Week 1 (fast version)
│   ├── benchmark_week2.py     # Week 2: Memory & I/O
│   ├── benchmark_week3_concurrency.py  # Week 3: Concurrency
│   ├── benchmark_json.py      # JSON performance
│   ├── benchmark_io.py        # I/O performance
│   ├── benchmark_parallel.py  # Parallel search
│   └── gpu_comparison_template.py  # GPU comparison template
│
├── docs/                       # Documentation
│   ├── README.md              # Documentation index
│   ├── ARCHITECTURE.md        # System architecture
│   ├── DESIGN_DECISIONS.md    # Design decisions
│   ├── HYBRID_SEARCH_DESIGN.md  # Hybrid search details
│   ├── METADATA_FILTERING.md  # Filtering system
│   ├── FEATURES_IMPLEMENTED.md  # Feature list
│   ├── COMPLETE_IMPLEMENTATION.md  # Implementation details
│   ├── COMPLETE_SUMMARY.md    # Project summary
│   ├── FINAL_OPTIMIZATION_REPORT.md  # ⭐ Performance results
│   ├── OPTIMIZATIONS.md       # Optimization overview
│   ├── WEEK1_SUMMARY.md       # Week 1 optimizations
│   ├── WEEK3_CONCURRENCY.md   # Week 3 concurrency
│   ├── COMPLETE_OPTIMIZATIONS.md  # All optimizations
│   ├── DEPLOYMENT_REPORT.md   # Deployment guide
│   └── FINAL_REPORT.md        # Final report
│
├── examples/                   # Example code
│   └── ...                    # Usage examples
│
├── .github/                    # GitHub configuration
│   └── workflows/             # CI/CD workflows
│
├── target/                     # Rust build artifacts (gitignored)
├── .venv/                      # Python virtual environment (gitignored)
│
├── Cargo.toml                  # Rust dependencies
├── pyproject.toml              # Python project config
├── README.md                   # Main project README
├── .gitignore                  # Git ignore rules
└── PROJECT_STRUCTURE.md        # This file
```

## Key Components

### Rust Source (`src/`)

**Core Modules:**
- `lib.rs` - Python bindings via PyO3, exposes VectorDB to Python
- `vectordb.rs` - Main database implementation with CRUD operations
- `index.rs` - Thread-safe HNSW index wrapper
- `storage.rs` - Persistence layer with memory-mapped I/O

**Feature Modules:**
- `filter.rs` - Advanced metadata filtering (15+ operators)
- `hybrid.rs` - Hybrid search with 5 fusion strategies
- `tantivy_index.rs` - Full-text search integration

**Optimization Modules:**
- `simd.rs` - SIMD optimizations (NEON for ARM, AVX2/SSE for x86)
- `utils.rs` - Vector operations and utilities

**Support:**
- `error.rs` - Comprehensive error handling

### Tests (`tests/`)

Comprehensive test suite covering:
- Basic operations
- CRUD functionality
- Data persistence
- Filtering (basic and advanced)
- Hybrid search
- Concurrent access
- New features

Run with: `python tests/test_*.py`

### Benchmarks (`benchmarks/`)

Performance benchmarks organized by optimization week:
- **Week 1**: CPU optimizations (SIMD, SmallVec, parallel)
- **Week 2**: Memory & I/O optimizations
- **Week 3**: Concurrency optimizations

Plus component-specific benchmarks for JSON, I/O, and parallel operations.

Run with: `python benchmarks/benchmark_*.py`

### Documentation (`docs/`)

Complete technical documentation:
- Architecture and design
- Feature implementation details
- Performance optimization reports
- Deployment guides

**Start here:** `docs/README.md` for the documentation index.

## Build Artifacts

### Compiled Files
- `target/` - Rust build artifacts (created by `cargo build`)
- `.venv/` - Python virtual environment
- `*.so` - Compiled Python extension (in `target/` after `maturin develop`)

### Temporary Files (gitignored)
- `__pycache__/` - Python bytecode cache
- `*.pyc` - Compiled Python files
- `test_db/`, `temp_db/` - Test database directories
- `*.tmp`, `*.bak` - Temporary/backup files

## Development Workflow

### 1. Setup
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or `.venv\Scripts\activate` on Windows

# Install dependencies
pip install maturin numpy

# Build and install
maturin develop --release
```

### 2. Development
```bash
# Edit Rust code in src/
# Edit tests in tests/
# Edit benchmarks in benchmarks/

# Rebuild after changes
maturin develop --release
```

### 3. Testing
```bash
# Run Rust unit tests
cargo test --lib

# Run Python tests
python tests/test_crud.py
python tests/final_verification.py

# Run benchmarks
python benchmarks/benchmark_week1_fast.py
```

### 4. Code Quality
```bash
# Format Rust code
cargo fmt

# Check for warnings
cargo clippy

# Run all tests
cargo test
```

## File Naming Conventions

- **Tests**: `test_*.py` - Python integration tests
- **Benchmarks**: `benchmark_*.py` - Performance benchmarks
- **Documentation**: `*.md` - Markdown documentation
- **Rust**: `*.rs` - Rust source files
- **Config**: `*.toml` - Configuration files (Cargo, pyproject)

## Dependencies

### Rust (Cargo.toml)
- `pyo3` - Python bindings
- `hnsw_rs` - HNSW indexing
- `tantivy` - Full-text search
- `rayon` - Parallelism
- `parking_lot` - Fast locks
- `memmap2` - Memory mapping
- `serde`, `serde_json` - Serialization

### Python (requirements)
- `maturin` - Build tool
- `numpy` - Array operations (for examples)

## Clean Build

To start fresh:

```bash
# Remove build artifacts
cargo clean
rm -rf target/

# Remove Python cache
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -type f -name "*.pyc" -delete

# Remove test databases
rm -rf test_db/ temp_db/

# Rebuild
maturin develop --release
```

## Production Deployment

See `docs/DEPLOYMENT_REPORT.md` for complete deployment instructions.

## Performance Results

Final optimized performance (after 3 weeks of optimization):
- **1,414 queries/sec** (single-thread, 2.8x improvement)
- **5,569 queries/sec** (4 threads batch, 5.6x improvement)
- **95% scaling efficiency** (up to 8 threads)
- **0.712ms P50 latency** (2.8x faster)
- **0.08 MB memory** (20% reduction)

See `docs/FINAL_OPTIMIZATION_REPORT.md` for complete details.
