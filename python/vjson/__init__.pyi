"""Type stubs for vjson - High-performance vector database with hybrid search."""

from typing import Any, Literal, TypedDict

__version__: str

# Exceptions
class VjsonError(Exception):
    """Base exception for all vjson errors."""

    ...

class DimensionMismatchError(VjsonError):
    """Raised when vector dimension doesn't match database dimension."""

    ...

class NotFoundError(VjsonError):
    """Raised when a vector ID is not found."""

    ...

class StorageError(VjsonError):
    """Raised on storage I/O errors."""

    ...

class InvalidParameterError(VjsonError):
    """Raised when an invalid parameter is provided."""

    ...

class SearchResult(TypedDict):
    """Search result containing vector ID, distance, and metadata."""

    id: str
    distance: float
    metadata: dict[str, Any]

class HybridSearchResult(TypedDict):
    """Hybrid search result with separate vector and text scores."""

    id: str
    vector_score: float
    text_score: float
    combined_score: float

class DatabaseStats(TypedDict):
    """Database statistics."""

    total_vectors: int
    dimension: int
    metadata_keys: list[str]
    active_vectors: int
    index_size: int

class PyVectorDB:
    """
    High-performance vector database with HNSW indexing and hybrid search.

    Example:
        >>> db = PyVectorDB("./my_db", dimension=384)
        >>> db.insert("doc1", [0.1] * 384, {"title": "Hello"})
        >>> results = db.search([0.1] * 384, k=10)
    """

    def __new__(
        cls,
        path: str,
        dimension: int,
        max_elements: int = 1000000,
        ef_construction: int = 200,
    ) -> "PyVectorDB":
        """
        Create a new vector database.

        Args:
            path: Directory path to store database files
            dimension: Vector dimension (e.g., 384, 768, 1536)
            max_elements: Maximum number of vectors (default: 1M)
            ef_construction: HNSW build quality parameter (default: 200)
        """
        ...

    def insert(self, id: str, vector: list[float], metadata: dict[str, Any]) -> None:
        """
        Insert a single vector with metadata.

        Args:
            id: Unique identifier for the vector
            vector: List of floats (must match dimension)
            metadata: Arbitrary JSON-serializable metadata
        """
        ...

    def insert_batch(
        self,
        items: list[
            tuple[str, list[float], dict[str, Any]]
            | tuple[str, list[float], dict[str, Any], str]
        ],
    ) -> None:
        """
        Insert multiple vectors in batch (much faster than individual inserts).

        Args:
            items: List of (id, vector, metadata) or (id, vector, metadata, text) tuples
        """
        ...

    def insert_with_text(
        self, id: str, vector: list[float], text: str, metadata: dict[str, Any]
    ) -> None:
        """
        Insert a vector with text content for hybrid search.

        Args:
            id: Unique identifier
            vector: Vector embeddings
            text: Text content for full-text search
            metadata: JSON metadata
        """
        ...

    def search(
        self,
        query: list[float],
        k: int,
        ef_search: int | None = None,
        filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """
        Search for k nearest neighbors with optional metadata filtering.

        Args:
            query: Query vector
            k: Number of results to return
            ef_search: Search quality (higher = better recall, default: 50)
            filter: Metadata filter (e.g., {"category": "tech"})

        Returns:
            List of search results with id, distance, and metadata

        Example:
            >>> results = db.search(query, k=10, filter={"score": {"$gt": 0.5}})
        """
        ...

    def batch_search(
        self,
        queries: list[list[float]],
        k: int,
        ef_search: int | None = None,
    ) -> list[list[SearchResult]]:
        """
        Search multiple queries in parallel.

        Args:
            queries: List of query vectors
            k: Number of results per query
            ef_search: Search quality parameter

        Returns:
            List of result lists, one per query
        """
        ...

    def hybrid_search(
        self,
        query_vector: list[float],
        query_text: str,
        k: int,
        ef_search: int | None = None,
        strategy: Literal["rrf", "weighted", "max", "min", "average"] = "rrf",
        vector_weight: float = 0.5,
        text_weight: float = 0.5,
    ) -> list[HybridSearchResult]:
        """
        Hybrid search combining vector similarity and full-text search.

        Args:
            query_vector: Query vector for semantic search
            query_text: Query text for full-text search
            k: Number of results
            ef_search: HNSW search quality
            strategy: Score fusion strategy
            vector_weight: Weight for vector scores (weighted strategy)
            text_weight: Weight for text scores (weighted strategy)

        Returns:
            List of hybrid results with separate and combined scores
        """
        ...

    def text_search(self, query: str, limit: int) -> list[tuple[str, float]]:
        """
        Full-text search using Tantivy.

        Args:
            query: Text query string
            limit: Maximum results

        Returns:
            List of (id, score) tuples
        """
        ...

    def get_metadata(self, id: str) -> dict[str, Any]:
        """Get metadata for a specific vector ID."""
        ...

    def get_metadata_batch(self, ids: list[str]) -> list[dict[str, Any]]:
        """Get metadata for multiple IDs in batch."""
        ...

    def get_vector(self, id: str) -> list[float]:
        """Get vector by ID."""
        ...

    def get_vectors_batch(self, ids: list[str]) -> list[dict[str, Any]]:
        """Get vectors for multiple IDs in batch."""
        ...

    def update(
        self, id: str, vector: list[float], metadata: dict[str, Any]
    ) -> None:
        """Update a vector and its metadata."""
        ...

    def update_with_text(
        self, id: str, vector: list[float], text: str, metadata: dict[str, Any]
    ) -> None:
        """Update a vector with text content."""
        ...

    def update_metadata(self, id: str, metadata: dict[str, Any]) -> None:
        """Update only metadata (fast, doesn't touch vector)."""
        ...

    def delete(self, id: str) -> None:
        """Delete a vector by ID."""
        ...

    def delete_batch(self, ids: list[str]) -> None:
        """Delete multiple vectors in batch."""
        ...

    def contains(self, id: str) -> bool:
        """Check if a vector ID exists."""
        ...

    def range_search(
        self,
        query: list[float],
        max_distance: float,
        ef_search: int | None = None,
    ) -> list[SearchResult]:
        """Find all vectors within a distance threshold."""
        ...

    def get_stats(self) -> DatabaseStats:
        """Get database statistics."""
        ...

    def rebuild_index(self) -> None:
        """Rebuild HNSW index (useful after many deletes)."""
        ...

    def load(self) -> None:
        """Load existing data from storage."""
        ...

    def save(self) -> None:
        """Save all data to storage."""
        ...

    def clear(self) -> None:
        """Clear all data from the database."""
        ...

    def is_empty(self) -> bool:
        """Check if database is empty."""
        ...

    def __len__(self) -> int:
        """Get number of vectors in database."""
        ...

    def __repr__(self) -> str:
        """String representation."""
        ...

# Utility functions
def normalize_vector(vector: list[float]) -> list[float]:
    """Normalize a vector to unit length (L2 normalization)."""
    ...

def normalize_vectors(vectors: list[list[float]]) -> list[list[float]]:
    """Normalize multiple vectors in batch."""
    ...

def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    ...

def dot_product(a: list[float], b: list[float]) -> float:
    """Compute dot product between two vectors."""
    ...

# Re-export for convenience
VectorDB = PyVectorDB

__all__ = [
    "PyVectorDB",
    "VectorDB",
    "normalize_vector",
    "normalize_vectors",
    "cosine_similarity",
    "dot_product",
    "VjsonError",
    "DimensionMismatchError",
    "NotFoundError",
    "StorageError",
    "InvalidParameterError",
    "__version__",
]
