# vjson

[![PyPI version](https://img.shields.io/pypi/v/vjson.svg)](https://pypi.org/project/vjson/)
[![Python Versions](https://img.shields.io/pypi/pyversions/vjson.svg)](https://pypi.org/project/vjson/)
[![CI](https://github.com/amiyamandal-dev/vjson/actions/workflows/ci.yml/badge.svg)](https://github.com/amiyamandal-dev/vjson/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Downloads](https://img.shields.io/pypi/dm/vjson.svg)](https://pypi.org/project/vjson/)

**High-performance vector database** with HNSW indexing and hybrid search, built with Rust for speed and Python for convenience.

---

## Table of Contents

- [Why vjson?](#why-vjson)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
  - [VectorDB Constructor](#vectordb-constructor)
  - [Insert Methods](#insert-methods)
  - [Search Methods](#search-methods)
  - [Retrieval Methods](#retrieval-methods)
  - [Update Methods](#update-methods)
  - [Delete Methods](#delete-methods)
  - [Utility Methods](#utility-methods)
- [Filter Operators](#filter-operators)
- [Hybrid Search](#hybrid-search)
- [Utility Functions](#utility-functions)
- [Exceptions](#exceptions)
- [Performance](#performance)
- [Architecture](#architecture)
- [Development](#development)
- [License](#license)

---

## Why vjson?

| Feature | vjson | sqlite-vec | chromadb |
|---------|-------|------------|----------|
| **Speed** | 344K qps | ~50K qps | ~10K qps |
| **Hybrid Search** | Yes | No | Yes |
| **SIMD Optimized** | Yes (NEON/AVX2) | No | No |
| **Type Safe** | Full stubs | Partial | Partial |
| **Dependencies** | 0 (self-contained) | SQLite | Heavy |
| **Persistence** | Automatic | Manual | Automatic |
| **Thread Safety** | Yes | Limited | Yes |

---

## Installation

```bash
pip install vjson
```

**Supported platforms:** Linux, macOS, Windows (x86_64, ARM64)

**Python versions:** 3.8+

---

## Quick Start

```python
from vjson import VectorDB

# Create a database (dimension = your embedding size)
db = VectorDB("./my_db", dimension=384)

# Insert vectors with metadata
db.insert("doc1", [0.1] * 384, {"title": "Hello World", "category": "greeting"})
db.insert("doc2", [0.2] * 384, {"title": "Goodbye World", "category": "farewell"})

# Search for similar vectors
results = db.search([0.15] * 384, k=5)

for r in results:
    print(f"{r['id']}: distance={r['distance']:.4f}, metadata={r['metadata']}")

# Search with metadata filter
results = db.search([0.15] * 384, k=5, filter={"category": "greeting"})

# Check database size
print(f"Database contains {len(db)} vectors")
```

**That's it!** No servers, no configuration, just a local database.

---

## API Reference

### VectorDB Constructor

```python
VectorDB(
    path: str,
    dimension: int,
    max_elements: int = 1000000,
    ef_construction: int = 200
)
```

Create a new vector database or open an existing one.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path` | `str` | *required* | Directory path for storing database files |
| `dimension` | `int` | *required* | Vector dimension (e.g., 128, 384, 768, 1536) |
| `max_elements` | `int` | `1000000` | Maximum number of vectors the database can hold |
| `ef_construction` | `int` | `200` | HNSW build quality parameter (100-400). Higher = better quality, slower build |

**Example:**

```python
# Small database for testing
db = VectorDB("./test_db", dimension=128, max_elements=10000)

# Production database with high-quality index
db = VectorDB("./prod_db", dimension=1536, max_elements=10000000, ef_construction=400)
```

---

### Insert Methods

#### `insert(id, vector, metadata)`

Insert a single vector with metadata.

```python
db.insert(
    id: str,
    vector: List[float],
    metadata: Dict[str, Any]
) -> None
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `id` | `str` | Unique identifier for the vector |
| `vector` | `List[float]` | Vector of floats (must match database dimension) |
| `metadata` | `Dict[str, Any]` | JSON-serializable metadata dictionary |

**Example:**

```python
db.insert(
    "user_123",
    [0.1, 0.2, 0.3, ...],  # 384-dimensional vector
    {"name": "John Doe", "age": 30, "tags": ["premium", "active"]}
)
```

**Raises:** `DimensionMismatchError` if vector dimension doesn't match database dimension.

---

#### `insert_batch(items)`

Insert multiple vectors in a batch (10-100x faster than individual inserts).

```python
db.insert_batch(
    items: List[Tuple[str, List[float], Dict[str, Any]]]
         | List[Tuple[str, List[float], Dict[str, Any], str]]
) -> None
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `items` | `List[Tuple]` | List of 3-tuples `(id, vector, metadata)` or 4-tuples `(id, vector, metadata, text)` |

**Example:**

```python
# Without text content
db.insert_batch([
    ("doc1", [0.1] * 384, {"title": "Document 1"}),
    ("doc2", [0.2] * 384, {"title": "Document 2"}),
    ("doc3", [0.3] * 384, {"title": "Document 3"}),
])

# With text content for hybrid search
db.insert_batch([
    ("doc1", [0.1] * 384, {"title": "ML Guide"}, "Machine learning tutorial"),
    ("doc2", [0.2] * 384, {"title": "AI Basics"}, "Introduction to artificial intelligence"),
])
```

---

#### `insert_with_text(id, vector, text, metadata)`

Insert a vector with text content for hybrid search.

```python
db.insert_with_text(
    id: str,
    vector: List[float],
    text: str,
    metadata: Dict[str, Any]
) -> None
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `id` | `str` | Unique identifier for the vector |
| `vector` | `List[float]` | Vector of floats |
| `text` | `str` | Text content for full-text search indexing |
| `metadata` | `Dict[str, Any]` | JSON-serializable metadata dictionary |

**Example:**

```python
db.insert_with_text(
    "article_001",
    embedding_vector,
    "Machine learning is a subset of artificial intelligence...",
    {"title": "ML Introduction", "author": "John Doe", "date": "2024-01-15"}
)
```

---

### Search Methods

#### `search(query, k, ef_search=None, filter=None)`

Search for k nearest neighbors with optional metadata filtering.

```python
db.search(
    query: List[float],
    k: int,
    ef_search: Optional[int] = None,
    filter: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | `List[float]` | *required* | Query vector |
| `k` | `int` | *required* | Number of nearest neighbors to return |
| `ef_search` | `int` | `50` | Search quality parameter. Higher = better recall, slower search |
| `filter` | `Dict` | `None` | Metadata filter (see [Filter Operators](#filter-operators)) |

**Returns:** List of dictionaries with keys:
- `id` (str): Vector ID
- `distance` (float): Distance from query (lower = more similar)
- `metadata` (dict): Associated metadata

**Example:**

```python
# Basic search
results = db.search([0.1] * 384, k=10)

# High-quality search
results = db.search([0.1] * 384, k=10, ef_search=200)

# Search with filter
results = db.search(
    [0.1] * 384,
    k=10,
    filter={"category": "tech", "score": {"$gt": 0.5}}
)

# Process results
for result in results:
    print(f"ID: {result['id']}")
    print(f"Distance: {result['distance']:.4f}")
    print(f"Metadata: {result['metadata']}")
```

---

#### `batch_search(queries, k, ef_search=None)`

Search multiple queries in parallel.

```python
db.batch_search(
    queries: List[List[float]],
    k: int,
    ef_search: Optional[int] = None
) -> List[List[Dict[str, Any]]]
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `queries` | `List[List[float]]` | *required* | List of query vectors |
| `k` | `int` | *required* | Number of results per query |
| `ef_search` | `int` | `50` | Search quality parameter |

**Returns:** List of result lists (one per query).

**Example:**

```python
queries = [
    [0.1] * 384,
    [0.2] * 384,
    [0.3] * 384,
]

all_results = db.batch_search(queries, k=5)

for i, results in enumerate(all_results):
    print(f"Query {i}: {len(results)} results")
```

---

#### `range_search(query, max_distance, ef_search=None)`

Find all vectors within a distance threshold.

```python
db.range_search(
    query: List[float],
    max_distance: float,
    ef_search: Optional[int] = None
) -> List[Dict[str, Any]]
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | `List[float]` | *required* | Query vector |
| `max_distance` | `float` | *required* | Maximum distance threshold |
| `ef_search` | `int` | `50` | Search quality parameter |

**Returns:** List of all results within the distance threshold.

**Example:**

```python
# Find all vectors within distance 0.5
results = db.range_search([0.1] * 384, max_distance=0.5)
print(f"Found {len(results)} vectors within threshold")
```

---

#### `text_search(query, limit)`

Perform full-text search using Tantivy.

```python
db.text_search(
    query: str,
    limit: int
) -> List[Tuple[str, float]]
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `query` | `str` | Text query string |
| `limit` | `int` | Maximum number of results |

**Returns:** List of tuples `(id, score)`.

**Example:**

```python
# Search for documents containing "machine learning"
results = db.text_search("machine learning", limit=10)

for doc_id, score in results:
    print(f"{doc_id}: score={score:.4f}")
```

**Note:** Requires vectors to be inserted with `insert_with_text` or 4-tuple batch insert.

---

#### `hybrid_search(query_vector, query_text, k, ef_search=None, strategy="rrf", vector_weight=0.5, text_weight=0.5)`

Combine vector similarity and full-text search.

```python
db.hybrid_search(
    query_vector: List[float],
    query_text: str,
    k: int,
    ef_search: Optional[int] = None,
    strategy: str = "rrf",
    vector_weight: float = 0.5,
    text_weight: float = 0.5
) -> List[Dict[str, Any]]
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query_vector` | `List[float]` | *required* | Query vector |
| `query_text` | `str` | *required* | Text query string |
| `k` | `int` | *required* | Number of results to return |
| `ef_search` | `int` | `50` | Search quality parameter |
| `strategy` | `str` | `"rrf"` | Fusion strategy (see below) |
| `vector_weight` | `float` | `0.5` | Weight for vector scores (used with `"weighted"`) |
| `text_weight` | `float` | `0.5` | Weight for text scores (used with `"weighted"`) |

**Fusion Strategies:**

| Strategy | Description |
|----------|-------------|
| `"rrf"` | Reciprocal Rank Fusion - balances both signals well |
| `"weighted"` | Weighted sum using `vector_weight` and `text_weight` |
| `"max"` | Maximum of both scores |
| `"min"` | Minimum of both scores |
| `"average"` | Average of both scores |

**Returns:** List of dictionaries with keys:
- `id` (str): Vector ID
- `vector_score` (float): Vector similarity score
- `text_score` (float): Text relevance score
- `combined_score` (float): Final fused score

**Example:**

```python
results = db.hybrid_search(
    query_vector=embedding,
    query_text="machine learning tutorial",
    k=10,
    strategy="rrf"
)

for r in results:
    print(f"{r['id']}: combined={r['combined_score']:.4f}, "
          f"vector={r['vector_score']:.4f}, text={r['text_score']:.4f}")
```

---

### Retrieval Methods

#### `get_vector(id)`

Get a vector by its ID.

```python
db.get_vector(id: str) -> List[float]
```

**Raises:** `NotFoundError` if ID doesn't exist.

**Example:**

```python
vector = db.get_vector("doc1")
print(f"Vector dimension: {len(vector)}")
```

---

#### `get_metadata(id)`

Get metadata for a specific vector ID.

```python
db.get_metadata(id: str) -> Dict[str, Any]
```

**Raises:** `NotFoundError` if ID doesn't exist.

**Example:**

```python
metadata = db.get_metadata("doc1")
print(f"Title: {metadata['title']}")
```

---

#### `get_vectors_batch(ids)`

Get multiple vectors by their IDs.

```python
db.get_vectors_batch(ids: List[str]) -> List[Dict[str, Any]]
```

**Returns:** List of dictionaries with keys `id` and `vector` (only for found IDs).

**Example:**

```python
results = db.get_vectors_batch(["doc1", "doc2", "doc3"])
for r in results:
    print(f"{r['id']}: {len(r['vector'])} dimensions")
```

---

#### `get_metadata_batch(ids)`

Get metadata for multiple vector IDs.

```python
db.get_metadata_batch(ids: List[str]) -> List[Dict[str, Any]]
```

**Returns:** List of dictionaries with keys `id` and `metadata` (only for found IDs).

**Example:**

```python
results = db.get_metadata_batch(["doc1", "doc2", "doc3"])
for r in results:
    print(f"{r['id']}: {r['metadata']}")
```

---

#### `contains(id)`

Check if a vector ID exists.

```python
db.contains(id: str) -> bool
```

**Example:**

```python
if db.contains("doc1"):
    print("Document exists")
else:
    print("Document not found")
```

---

#### `get_stats()`

Get database statistics.

```python
db.get_stats() -> Dict[str, Any]
```

**Returns:** Dictionary with keys:
- `total_vectors` (int): Total number of vectors in ID map
- `dimension` (int): Vector dimension
- `metadata_keys` (List[str]): All unique metadata keys
- `active_vectors` (int): Number of active (non-deleted) vectors
- `index_size` (int): Current HNSW index size

**Example:**

```python
stats = db.get_stats()
print(f"Total vectors: {stats['total_vectors']}")
print(f"Dimension: {stats['dimension']}")
print(f"Metadata keys: {stats['metadata_keys']}")
```

---

### Update Methods

#### `update(id, vector, metadata)`

Update a vector and its metadata.

```python
db.update(
    id: str,
    vector: List[float],
    metadata: Dict[str, Any]
) -> None
```

**Raises:** `NotFoundError` if ID doesn't exist.

**Example:**

```python
db.update(
    "doc1",
    [0.5] * 384,
    {"title": "Updated Title", "version": 2}
)
```

---

#### `update_metadata(id, metadata)`

Update only metadata (fast, doesn't touch vector or index).

```python
db.update_metadata(
    id: str,
    metadata: Dict[str, Any]
) -> None
```

**Raises:** `NotFoundError` if ID doesn't exist.

**Example:**

```python
# Fast metadata-only update
db.update_metadata("doc1", {"views": 1000, "last_accessed": "2024-01-15"})
```

---

#### `update_with_text(id, vector, text, metadata)`

Update a vector with new text content.

```python
db.update_with_text(
    id: str,
    vector: List[float],
    text: str,
    metadata: Dict[str, Any]
) -> None
```

**Example:**

```python
db.update_with_text(
    "doc1",
    new_embedding,
    "Updated text content for search",
    {"title": "Updated Document"}
)
```

---

### Delete Methods

#### `delete(id)`

Delete a vector by ID.

```python
db.delete(id: str) -> None
```

**Raises:** `NotFoundError` if ID doesn't exist.

**Example:**

```python
db.delete("doc1")
```

---

#### `delete_batch(ids)`

Delete multiple vectors in a batch.

```python
db.delete_batch(ids: List[str]) -> None
```

**Raises:** `NotFoundError` if any ID doesn't exist.

**Example:**

```python
db.delete_batch(["doc1", "doc2", "doc3"])
```

---

### Utility Methods

#### `load()`

Load existing data from storage and rebuild the HNSW index.

```python
db.load() -> None
```

**Example:**

```python
db = VectorDB("./existing_db", dimension=384)
db.load()  # Rebuild index from stored vectors
print(f"Loaded {len(db)} vectors")
```

---

#### `save()`

Explicitly save data to storage. Note: Data is automatically saved on insert.

```python
db.save() -> None
```

---

#### `clear()`

Clear all data from the database.

```python
db.clear() -> None
```

**Example:**

```python
db.clear()
assert len(db) == 0
```

---

#### `rebuild_index()`

Rebuild the HNSW index from current data. Useful after many deletes to reclaim space and optimize performance.

```python
db.rebuild_index() -> None
```

**Example:**

```python
# After deleting many vectors
db.delete_batch(old_ids)
db.rebuild_index()  # Reclaim space and optimize
```

---

#### `is_empty()`

Check if the database is empty.

```python
db.is_empty() -> bool
```

---

#### `__len__()`

Get the number of vectors in the database.

```python
len(db) -> int
```

**Example:**

```python
print(f"Database has {len(db)} vectors")
```

---

## Filter Operators

Filters use MongoDB-style query syntax on metadata fields.

### Comparison Operators

| Operator | Example | Description |
|----------|---------|-------------|
| `$eq` | `{"status": "active"}` or `{"status": {"$eq": "active"}}` | Equals (default) |
| `$ne` | `{"status": {"$ne": "deleted"}}` | Not equals |
| `$gt` | `{"score": {"$gt": 0.5}}` | Greater than |
| `$gte` | `{"score": {"$gte": 0.5}}` | Greater than or equal |
| `$lt` | `{"age": {"$lt": 30}}` | Less than |
| `$lte` | `{"age": {"$lte": 30}}` | Less than or equal |
| `$between` | `{"price": {"$between": [10, 100]}}` | Range (inclusive) |

### Array Operators

| Operator | Example | Description |
|----------|---------|-------------|
| `$in` | `{"tag": {"$in": ["python", "rust"]}}` | Value in array |
| `$nin` | `{"tag": {"$nin": ["deprecated"]}}` | Value not in array |

### String Operators

| Operator | Example | Description |
|----------|---------|-------------|
| `$startsWith` | `{"name": {"$startsWith": "John"}}` | String prefix match |
| `$endsWith` | `{"email": {"$endsWith": "@gmail.com"}}` | String suffix match |
| `$contains` | `{"text": {"$contains": "urgent"}}` | Substring match |
| `$regex` | `{"name": {"$regex": "^[A-Z].*"}}` | Regular expression match |

### Existence Operator

| Operator | Example | Description |
|----------|---------|-------------|
| `$exists` | `{"email": {"$exists": true}}` | Field exists/doesn't exist |

### Logical Operators

| Operator | Example | Description |
|----------|---------|-------------|
| `$and` | `{"$and": [{"age": {"$gt": 18}}, {"status": "active"}]}` | All conditions must match |
| `$or` | `{"$or": [{"status": "active"}, {"role": "admin"}]}` | Any condition must match |

### Nested Field Access

Use dot notation to access nested fields:

```python
# Metadata: {"user": {"profile": {"age": 25}}}
filter = {"user.profile.age": {"$gte": 18}}
```

### Combined Filters

Multiple conditions at the top level are implicitly AND:

```python
# Both conditions must match
filter = {
    "category": "tech",
    "score": {"$gte": 0.8},
    "status": {"$in": ["published", "featured"]}
}
```

---

## Hybrid Search

Hybrid search combines vector similarity with full-text keyword search for better results.

### Setup

Insert documents with text content:

```python
db.insert_with_text(
    "doc1",
    embedding_vector,
    "Machine learning is revolutionizing data science",
    {"title": "ML Introduction", "category": "tutorial"}
)
```

### Fusion Strategies

| Strategy | Best For |
|----------|----------|
| `"rrf"` | General use - balances semantic and keyword matches |
| `"weighted"` | When you need precise control over signal importance |
| `"max"` | When either signal alone is sufficient |
| `"average"` | Balanced combination with equal weight |

### Example

```python
# Reciprocal Rank Fusion (recommended default)
results = db.hybrid_search(
    query_vector=query_embedding,
    query_text="machine learning tutorial",
    k=10,
    strategy="rrf"
)

# Weighted combination (favor semantic similarity)
results = db.hybrid_search(
    query_vector=query_embedding,
    query_text="machine learning",
    k=10,
    strategy="weighted",
    vector_weight=0.7,
    text_weight=0.3
)
```

---

## Utility Functions

### `normalize_vector(vector)`

Normalize a vector to unit length (L2 normalization).

```python
from vjson import normalize_vector

vec = normalize_vector([3.0, 4.0])
# Returns: [0.6, 0.8]
```

### `normalize_vectors(vectors)`

Normalize multiple vectors in batch (parallelized).

```python
from vjson import normalize_vectors

vecs = normalize_vectors([[3.0, 4.0], [1.0, 0.0]])
```

### `cosine_similarity(a, b)`

Compute cosine similarity between two vectors. Returns value in range [-1, 1].

```python
from vjson import cosine_similarity

sim = cosine_similarity([1.0, 0.0], [1.0, 0.0])  # 1.0 (identical)
sim = cosine_similarity([1.0, 0.0], [0.0, 1.0])  # 0.0 (orthogonal)
sim = cosine_similarity([1.0, 0.0], [-1.0, 0.0]) # -1.0 (opposite)
```

### `dot_product(a, b)`

Compute dot product between two vectors.

```python
from vjson import dot_product

result = dot_product([2.0, 3.0], [4.0, 5.0])  # 2*4 + 3*5 = 23.0
```

---

## Exceptions

All exceptions inherit from `VjsonError`.

```python
from vjson import (
    VjsonError,              # Base exception
    DimensionMismatchError,  # Vector dimension doesn't match
    NotFoundError,           # ID not found
    StorageError,            # I/O error
    InvalidParameterError,   # Invalid parameter
)
```

### Exception Handling

```python
from vjson import VectorDB, DimensionMismatchError, NotFoundError

db = VectorDB("./db", dimension=384)

try:
    db.insert("id1", [0.1] * 128, {})  # Wrong dimension!
except DimensionMismatchError as e:
    print(e)  # "Dimension mismatch: expected 384, got 128"

try:
    db.get_vector("nonexistent")
except NotFoundError as e:
    print(e)  # "Not found: nonexistent"
```

---

## Performance

### Benchmarks

Tested on Apple M1:

| Operation | Throughput |
|-----------|------------|
| Vector search | 344,643 qps |
| Batch insert | 20,488 vectors/sec |
| Concurrent search (8 threads) | 6,407 qps |
| Single insert | ~2,000 vectors/sec |

### Optimization Tips

1. **Use batch operations** - `insert_batch` is 10-100x faster than individual inserts

2. **Tune `ef_search`** - Trade-off between speed and recall:
   - Low (10-30): Fast, lower recall
   - Medium (50-100): Balanced
   - High (100-300): Slower, higher recall

3. **Use filters** - Reduces search space significantly

4. **Rebuild after deletes** - `db.rebuild_index()` reclaims space and optimizes

5. **Batch reads** - `get_vectors_batch` and `get_metadata_batch` are faster for multiple IDs

6. **Pre-normalize vectors** - If using cosine similarity, normalize before insertion

---

## Architecture

### Storage Format

```
./my_db/
├── vectors.bin      # Memory-mapped binary vector data
├── metadata.ndjson  # Append-only NDJSON metadata
└── text_index/      # Tantivy full-text search index
```

### Key Technologies

- **HNSW Index**: Hierarchical Navigable Small World algorithm for fast ANN search
- **Memory-mapped I/O**: Zero-copy reads for vectors
- **SIMD Optimization**: AVX2/NEON for distance calculations
- **Tantivy**: Rust-based full-text search engine
- **parking_lot**: Fast read-write locks for concurrent access

### Thread Safety

- **Parallel reads**: Multiple threads can search simultaneously
- **Locked writes**: Only one thread can write at a time
- **Lock-free counters**: Atomic operations for statistics

---

## Development

### Setup

```bash
# Clone
git clone https://github.com/amiyamandal-dev/vjson.git
cd vjson

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install maturin pytest

# Build
maturin develop --release

# Run tests
make test        # All tests
make test-rust   # Rust only
make test-python # Python only

# Lint
make lint
make format
```

### Project Structure

```
vjson/
├── src/
│   ├── lib.rs           # Python bindings (PyO3)
│   ├── vectordb.rs      # Main VectorDB implementation
│   ├── index.rs         # HNSW index wrapper
│   ├── storage.rs       # Persistence layer
│   ├── filter.rs        # Metadata filtering
│   ├── hybrid.rs        # Hybrid search fusion
│   ├── simd.rs          # SIMD-optimized operations
│   ├── tantivy_index.rs # Full-text search
│   ├── utils.rs         # Utility functions
│   └── error.rs         # Error types
├── tests/               # Python tests
├── python/vjson/        # Type stubs
└── Cargo.toml           # Rust dependencies
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
