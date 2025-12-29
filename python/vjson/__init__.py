"""
vjson - High-performance vector database with hybrid search.

Built with Rust for speed, designed for Python convenience.

Example:
    >>> from vjson import VectorDB
    >>> db = VectorDB("./my_db", dimension=384)
    >>> db.insert("doc1", [0.1] * 384, {"title": "Hello World"})
    >>> results = db.search([0.1] * 384, k=10)
    >>> print(results[0]["id"])
    'doc1'

Exceptions:
    VjsonError: Base exception for all vjson errors
    DimensionMismatchError: Vector dimension doesn't match database
    NotFoundError: Vector ID not found
    StorageError: Storage I/O error
    InvalidParameterError: Invalid parameter provided
"""

from vjson.vjson import (
    PyVectorDB,
    normalize_vector,
    normalize_vectors,
    cosine_similarity,
    dot_product,
    __version__,
    # Exceptions
    VjsonError,
    DimensionMismatchError,
    NotFoundError,
    StorageError,
    InvalidParameterError,
)

# Convenience alias
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
