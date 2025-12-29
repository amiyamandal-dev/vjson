# vjson

[![PyPI version](https://img.shields.io/pypi/v/vjson.svg)](https://pypi.org/project/vjson/)
[![Python Versions](https://img.shields.io/pypi/pyversions/vjson.svg)](https://pypi.org/project/vjson/)
[![CI](https://github.com/amiyamandal-dev/vjson/actions/workflows/ci.yml/badge.svg)](https://github.com/amiyamandal-dev/vjson/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Downloads](https://img.shields.io/pypi/dm/vjson.svg)](https://pypi.org/project/vjson/)

**High-performance vector database** with HNSW indexing and hybrid search, built with Rust for speed and Python for convenience.

---

## Why vjson?

| Feature | vjson | sqlite-vec | chromadb |
|---------|-------|------------|----------|
| **Speed** | 344K qps | ~50K qps | ~10K qps |
| **Hybrid Search** | Yes | No | Yes |
| **SIMD Optimized** | Yes (NEON/AVX2) | No | No |
| **Type Safe** | Full stubs | Partial | Partial |
| **Dependencies** | 0 (self-contained) | SQLite | Heavy |

---

## Installation

```bash
pip install vjson
```

**Supported platforms:** Linux, macOS, Windows (x86_64, ARM64)

---

## Quick Start

```python
from vjson import VectorDB

# Create a database (dimension = your embedding size)
db = VectorDB("./my_db", dimension=384)

# Insert a vector with metadata
db.insert("doc1", [0.1] * 384, {"title": "Hello World"})

# Search for similar vectors
results = db.search([0.1] * 384, k=5)

for r in results:
    print(f"{r['id']}: {r['distance']:.4f}")
```

**That's it!** No servers, no configuration, just a local database.

---

## Features

### Vector Search

```python
# Basic search
results = db.search(query_vector, k=10)

# Search with metadata filter
results = db.search(query_vector, k=10, filter={"category": "tech"})

# Batch search (multiple queries at once)
results = db.batch_search([query1, query2, query3], k=10)
```

### Hybrid Search (Vector + Text)

Combine semantic similarity with keyword matching:

```python
# Insert with text content
db.insert_with_text(
    "doc1",
    embedding,
    "Machine learning and artificial intelligence",
    {"topic": "AI"}
)

# Hybrid search
results = db.hybrid_search(
    query_vector=embedding,
    query_text="machine learning",
    k=10,
    strategy="rrf"  # Reciprocal Rank Fusion
)
```

### Rich Filtering

MongoDB-style filters on metadata:

```python
# Comparison
filter = {"score": {"$gt": 0.5}}
filter = {"price": {"$between": [10, 100]}}

# String matching
filter = {"name": {"$startsWith": "John"}}
filter = {"email": {"$contains": "@gmail"}}

# Array operations
filter = {"tags": {"$in": ["python", "rust"]}}

# Combine with AND
filter = {
    "category": "tech",
    "score": {"$gte": 0.8}
}
```

### Full CRUD Operations

```python
# Create
db.insert("id1", vector, metadata)
db.insert_batch([("id2", vec2, meta2), ("id3", vec3, meta3)])

# Read
vector = db.get_vector("id1")
metadata = db.get_metadata("id1")
exists = db.contains("id1")

# Update
db.update("id1", new_vector, new_metadata)
db.update_metadata("id1", {"score": 0.99})  # Fast, doesn't touch vector

# Delete
db.delete("id1")
db.delete_batch(["id2", "id3"])
```

### Persistence

Data is automatically saved on insert. To load an existing database:

```python
db = VectorDB("./my_db", dimension=384)
db.load()  # Rebuilds index from stored data

print(f"Loaded {len(db)} vectors")
```

---

## API Reference

### VectorDB

```python
VectorDB(
    path: str,                    # Storage directory
    dimension: int,               # Vector dimension (e.g., 384, 768, 1536)
    max_elements: int = 1000000,  # Maximum vectors
    ef_construction: int = 200,   # Build quality (100-400)
)
```

### Search Methods

| Method | Description |
|--------|-------------|
| `search(query, k, ef_search=50, filter=None)` | Find k nearest neighbors |
| `batch_search(queries, k, ef_search=50)` | Search multiple queries |
| `text_search(query_text, limit)` | Full-text search only |
| `hybrid_search(query_vector, query_text, k, strategy="rrf")` | Combined search |
| `range_search(query, max_distance)` | Find all within distance |

### Filter Operators

| Operator | Example | Description |
|----------|---------|-------------|
| `$eq` | `{"status": "active"}` | Equals (default) |
| `$ne` | `{"status": {"$ne": "deleted"}}` | Not equals |
| `$gt`, `$gte` | `{"score": {"$gt": 0.5}}` | Greater than |
| `$lt`, `$lte` | `{"age": {"$lt": 30}}` | Less than |
| `$between` | `{"price": {"$between": [10, 50]}}` | Range (inclusive) |
| `$in` | `{"tag": {"$in": ["a", "b"]}}` | In array |
| `$nin` | `{"tag": {"$nin": ["x"]}}` | Not in array |
| `$exists` | `{"email": {"$exists": true}}` | Field exists |
| `$startsWith` | `{"name": {"$startsWith": "John"}}` | String prefix |
| `$endsWith` | `{"email": {"$endsWith": "@gmail.com"}}` | String suffix |
| `$contains` | `{"text": {"$contains": "urgent"}}` | Substring |
| `$regex` | `{"name": {"$regex": "^[A-Z]"}}` | Regular expression |

### Utility Functions

```python
from vjson import normalize_vector, cosine_similarity, dot_product

# Normalize to unit length
normalized = normalize_vector([3.0, 4.0])  # [0.6, 0.8]

# Compute similarity
sim = cosine_similarity(vec_a, vec_b)
dot = dot_product(vec_a, vec_b)
```

### Exceptions

```python
from vjson import (
    VjsonError,              # Base exception
    DimensionMismatchError,  # Vector dimension wrong
    NotFoundError,           # ID not found
    StorageError,            # I/O error
    InvalidParameterError,   # Bad parameter
)

try:
    db.insert("id", wrong_size_vector, {})
except DimensionMismatchError as e:
    print(e)  # "Dimension mismatch: expected 384, got 128"
```

---

## Performance

Benchmarked on Apple M1:

| Operation | Throughput |
|-----------|------------|
| Vector search | 344,643 qps |
| Batch insert | 20,488 vectors/sec |
| Concurrent search (8 threads) | 6,407 qps |

### Tips

1. **Use batch operations** - 10-100x faster than single inserts
2. **Tune ef_search** - Lower (10-20) for speed, higher (100+) for recall
3. **Use filters** - Reduces search space significantly
4. **Rebuild after deletes** - `db.rebuild_index()` reclaims space

---

## Development

```bash
# Clone
git clone https://github.com/amiyamandal-dev/vjson.git
cd vjson

# Setup
python -m venv .venv
source .venv/bin/activate
pip install maturin pytest

# Build
maturin develop

# Test
make test        # All tests
make test-rust   # Rust only
make test-python # Python only

# Lint
make lint
make format
```

### Release

```bash
make release-patch  # 0.1.0 -> 0.1.1
make release-minor  # 0.1.0 -> 0.2.0
make release-major  # 0.1.0 -> 1.0.0
```

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Links

- [PyPI](https://pypi.org/project/vjson/)
- [GitHub](https://github.com/amiyamandal-dev/vjson)
- [Issues](https://github.com/amiyamandal-dev/vjson/issues)

---

Built with Rust and Python
