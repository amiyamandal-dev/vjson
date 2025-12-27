# vjson - High-Performance Vector Database

A production-ready vector database built with Rust, featuring HNSW indexing, full-text search, advanced filtering, and complete CRUD operations.

## 🚀 Features

### Core Capabilities
- ✅ **Fast Vector Search** - HNSW algorithm for approximate nearest neighbor search
- ✅ **Full CRUD Operations** - Create, Read, Update, Delete vectors
- ✅ **Persistence** - Automatic saving and loading from disk
- ✅ **Hybrid Search** - Combine vector similarity with full-text search
- ✅ **Advanced Filtering** - 14+ metadata filter operators
- ✅ **Thread-Safe** - Concurrent reads, safe writes
- ✅ **Memory Efficient** - Memory-mapped I/O for large datasets

### Performance
- **344,643** queries/second (vector search)
- **20,488** vectors/second (batch insert)
- **6,407** queries/second (8 concurrent threads)
- **~21%** storage overhead

## 📦 Installation

```bash
# Install from source
pip install maturin
maturin develop --release
```

## 🎯 Quick Start

```python
import vjson
import numpy as np

# Create database
db = vjson.PyVectorDB(
    path="./my_vectors",
    dimension=768,
    max_elements=1000000,
    ef_construction=200  # Higher = better quality, slower build
)

# Insert vectors
embedding = np.random.rand(768).tolist()
db.insert("doc1", embedding, {"title": "My Document", "score": 0.95})

# Insert with text for hybrid search
db.insert_with_text(
    "doc2",
    embedding,
    "Machine learning and artificial intelligence research",
    {"category": "AI", "year": 2024}
)

# Vector search
results = db.search(query_vector, k=10, ef_search=50)
for result in results:
    print(f"{result['id']}: {result['distance']}")
    print(f"  Metadata: {result['metadata']}")

# Search with filtering
results = db.search(
    query_vector,
    k=10,
    filter={
        "category": "AI",
        "year": {"$gte": 2020},
        "score": {"$between": [0.8, 1.0]}
    }
)

# Hybrid search (vector + text)
results = db.hybrid_search(
    query_vector,
    "machine learning neural networks",
    k=10,
    strategy="rrf"  # Reciprocal Rank Fusion
)

# CRUD operations
db.update("doc1", new_vector, {"title": "Updated Document"})
db.delete("old_doc")
db.delete_batch(["doc1", "doc2", "doc3"])

# Check existence
if db.contains("doc1"):
    vec = db.get_vector("doc1")
    meta = db.get_metadata("doc1")

# Persistence
db.save()  # Explicit save (auto-saved on insert)

# Load existing database
db2 = vjson.PyVectorDB("./my_vectors", dimension=768)
db2.load()  # Rebuilds index from stored data
print(f"Loaded {len(db2)} vectors")
```

## 🔍 Advanced Filtering

### Comparison Operators
```python
# Equals (default)
filter = {"category": "tech"}

# Not equals
filter = {"status": {"$ne": "deleted"}}

# Greater than / Less than
filter = {"score": {"$gt": 0.5}}
filter = {"age": {"$lt": 30}}
filter = {"score": {"$gte": 0.8}}
filter = {"price": {"$lte": 100}}

# Range (inclusive)
filter = {"price": {"$between": [10, 50]}}
```

### Array Operators
```python
# In array
filter = {"category": {"$in": ["tech", "science", "health"]}}

# Not in array
filter = {"status": {"$nin": ["spam", "deleted"]}}
```

### String Operators
```python
# Starts with
filter = {"name": {"$startsWith": "John"}}

# Ends with
filter = {"email": {"$endsWith": "@gmail.com"}}

# Contains substring
filter = {"description": {"$contains": "urgent"}}

# Regular expression
filter = {"name": {"$regex": "^[A-Z][a-z]+"}}
```

### Field Existence
```python
# Field exists
filter = {"email": {"$exists": True}}

# Field doesn't exist
filter = {"deleted_at": {"$exists": False}}
```

### Combining Filters (AND)
```python
# Multiple conditions (implicit AND)
filter = {
    "category": "AI",
    "year": {"$gte": 2020},
    "score": {"$gt": 0.8},
    "author.name": {"$startsWith": "Dr."}  # Nested fields
}
```

## 🔄 Hybrid Search

Combine vector similarity with full-text search using multiple fusion strategies:

```python
# Reciprocal Rank Fusion (default, robust)
results = db.hybrid_search(
    query_vector,
    "machine learning deep neural networks",
    k=10,
    strategy="rrf"
)

# Weighted sum (custom weights)
results = db.hybrid_search(
    query_vector,
    "pytorch tensorflow",
    k=10,
    strategy="weighted",
    vector_weight=0.7,
    text_weight=0.3
)

# Other strategies: "max", "min", "average"
```

## 📊 CRUD Operations

### Create
```python
# Single insert
db.insert("id1", vector, metadata)

# Batch insert (much faster)
items = [
    ("id1", vector1, {"key": "value"}),
    ("id2", vector2, {"key": "value"}),
]
db.insert_batch(items)

# Insert with text
db.insert_with_text("id3", vector, "searchable text", metadata)
```

### Read
```python
# Search
results = db.search(query, k=10)

# Get by ID
vector = db.get_vector("id1")
metadata = db.get_metadata("id1")

# Check existence
exists = db.contains("id1")

# Get size
count = len(db)
```

### Update
```python
# Update vector and metadata
db.update("id1", new_vector, new_metadata)

# Update with text
db.update_with_text("id1", new_vector, "new text", new_metadata)
```

### Delete
```python
# Delete single
db.delete("id1")

# Delete multiple
db.delete_batch(["id1", "id2", "id3"])

# Rebuild index after many deletes (reclaim space)
db.rebuild_index()

# Clear all
db.clear()
```

## 🏗️ Architecture

### Components

**Storage Layer** (`storage.rs`)
- Memory-mapped I/O for vectors
- JSON metadata storage
- Atomic writes for crash safety
- Parallel deserialization

**HNSW Index** (`index.rs`)
- Fast approximate nearest neighbor search
- Thread-safe with RwLock
- Configurable quality parameters
- O(log N) search complexity

**Full-Text Search** (`tantivy_index.rs`)
- Tantivy 0.22 integration
- Inverted index for text
- Automatic document deletion

**Hybrid Search** (`hybrid.rs`)
- 5 fusion strategies
- Parallel score computation
- Customizable weights

**Advanced Filters** (`filter.rs`)
- 14+ filter operators
- Nested field support
- Regex pattern matching
- Post-search filtering

### Concurrency Model

- **Parallel Reads**: Multiple threads can search simultaneously
- **Exclusive Writes**: Single writer with RwLock protection
- **Lock-Free Counting**: Atomic counters for metrics
- **Thread-Safe**: All operations are thread-safe

## ⚙️ Configuration

```python
db = vjson.PyVectorDB(
    path="./vectors",
    dimension=768,              # Vector dimension
    max_elements=1000000,       # Max vectors (can rebuild with more)
    ef_construction=200         # Build quality: 100-400
)

# Search quality parameter
results = db.search(
    query,
    k=10,
    ef_search=50  # Search quality: 10-500 (higher = better recall)
)
```

### Parameter Tuning

**ef_construction** (build quality):
- 100: Fast build, lower quality
- 200: Balanced (default)
- 400: Slower build, higher quality

**ef_search** (search quality):
- 10-20: Fast, ~90% recall
- 50: Balanced (default), ~95% recall
- 100-500: Slower, ~99% recall

## 🧪 Testing

```bash
# Run Rust unit tests
cargo test --lib

# Run Python integration tests
python tests/test_crud.py
python tests/test_persistence.py
python tests/test_advanced_filters.py
python tests/test_concurrent.py

# Quick verification
python tests/final_verification.py
```

**Test Coverage:**
- ✅ 17/17 Rust unit tests passing
- ✅ All Python integration tests passing
- ✅ CRUD operations
- ✅ Persistence and data integrity
- ✅ 15 filter operator tests
- ✅ Hybrid search
- ✅ Concurrent access

See [tests/README.md](tests/README.md) for complete test documentation.

## 📈 Performance Tips

1. **Use Batch Inserts**: 10-100x faster than single inserts
2. **Tune ef_search**: Balance between speed and recall
3. **Filter Early**: Use metadata filters to reduce search space
4. **Rebuild Periodically**: After many deletes to reclaim space
5. **Memory-Map Large Datasets**: Vectors are automatically memory-mapped

## 🔧 Advanced Usage

### Batch Operations
```python
# Batch insert
items = [(f"id{i}", vectors[i], metadata[i]) for i in range(1000)]
db.insert_batch(items)

# Batch search
queries = [query1, query2, query3]
all_results = db.batch_search(queries, k=10)
```

### Persistence Workflow
```python
# Session 1: Create and populate
db1 = vjson.PyVectorDB("./db", dimension=768)
db1.insert_batch(items)
# Auto-saved during inserts

# Session 2: Load and continue
db2 = vjson.PyVectorDB("./db", dimension=768)
db2.load()  # Rebuilds HNSW index (fast)
db2.insert(new_id, new_vector, metadata)
```

### Error Handling
```python
try:
    db.insert("id1", wrong_dimension_vector, {})
except RuntimeError as e:
    print(f"Error: {e}")  # "Dimension mismatch: expected 768, got 512"
```

## 📚 API Reference

### VectorDB Methods

**Constructor:**
- `PyVectorDB(path, dimension, max_elements=1000000, ef_construction=200)`

**Insert:**
- `insert(id, vector, metadata)`
- `insert_batch(items)`
- `insert_with_text(id, vector, text, metadata)`

**Search:**
- `search(query, k, ef_search=None, filter=None)`
- `batch_search(queries, k, ef_search=None)`
- `text_search(query, limit)`
- `hybrid_search(query_vector, query_text, k, ef_search=None, strategy="rrf", ...)`

**Read:**
- `get_vector(id)`
- `get_metadata(id)`
- `contains(id)`
- `len()` / `__len__()`
- `is_empty()`

**Update:**
- `update(id, vector, metadata)`
- `update_with_text(id, vector, text, metadata)`

**Delete:**
- `delete(id)`
- `delete_batch(ids)`
- `clear()`
- `rebuild_index()`

**Persistence:**
- `save()`
- `load()`

## 🛠️ Dependencies

**Core:**
- Rust 1.70+
- Python 3.8+

**Rust Crates:**
- `pyo3` 0.22 - Python bindings
- `hnsw_rs` 0.3.3 - HNSW indexing
- `tantivy` 0.22 - Full-text search
- `rayon` 1.10 - Parallelism
- `parking_lot` 0.12 - Fast locks
- `memmap2` 0.9 - Memory mapping
- `regex` 1.10 - Pattern matching

## 📚 Documentation

Comprehensive documentation is available in the `docs/` directory:

- **Architecture & Design**: System design, component architecture, design decisions
- **Performance**: Optimization reports, benchmarks, concurrency analysis
- **Features**: Complete feature documentation and implementation details
- **Deployment**: Production deployment guides and best practices

See [docs/README.md](docs/README.md) for the complete documentation index.

### Benchmarks

Performance benchmarks are in the `benchmarks/` directory. See [benchmarks/README.md](benchmarks/README.md) for details.

**Quick benchmark:**
```bash
python benchmarks/benchmark_week1_fast.py
```

## 📄 License

MIT OR Apache-2.0

## 🤝 Contributing

Contributions welcome! Please ensure:
- All tests pass (`cargo test && python test_*.py`)
- Code follows Rust formatting (`cargo fmt`)
- No clippy warnings (`cargo clippy`)

## 📖 References

- [HNSW Paper](https://arxiv.org/abs/1603.09320) - Malkov & Yashunin
- [Tantivy](https://github.com/quickwit-oss/tantivy) - Full-text search
- [PyO3](https://pyo3.rs) - Rust-Python bindings

## 🎯 Production Checklist

- ✅ Thread-safe concurrent access
- ✅ Crash-safe atomic writes
- ✅ Comprehensive error handling
- ✅ Memory-efficient storage
- ✅ Fast index rebuilding
- ✅ Extensive test coverage
- ✅ Performance benchmarks
- ✅ Production-ready code quality

---

**Status:** Production Ready ✅

Built with ❤️ using Rust and Python
