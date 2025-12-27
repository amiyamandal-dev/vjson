# Tests

This directory contains test files for the vjson vector database.

## Test Files

### Core Functionality Tests
- **test_simple.py** - Basic functionality test
- **test_crud.py** - Create, Read, Update, Delete operations
- **test_persistence.py** - Data persistence and loading

### Advanced Features
- **test_filters.py** - Basic metadata filtering
- **test_advanced_filters.py** - Advanced filter operators (15 tests)
- **test_hybrid_search.py** - Hybrid search (vector + text)
- **test_new_features.py** - New feature tests

### Concurrency & Performance
- **test_concurrent.py** - Concurrent access and thread safety

### Verification
- **final_verification.py** - Final complete verification of all features

## Running Tests

```bash
# Run all tests
python tests/test_simple.py
python tests/test_crud.py
python tests/test_persistence.py
python tests/test_filters.py
python tests/test_advanced_filters.py
python tests/test_hybrid_search.py
python tests/test_concurrent.py

# Quick verification
python tests/final_verification.py
```

## Test Coverage

- ✅ Vector insertion and search
- ✅ Batch operations
- ✅ CRUD operations (Create, Read, Update, Delete)
- ✅ Data persistence and loading
- ✅ Metadata filtering (15+ operators)
- ✅ Hybrid search (5 fusion strategies)
- ✅ Concurrent access and thread safety
- ✅ Range search
- ✅ Index rebuilding

## Rust Unit Tests

Run Rust unit tests with:

```bash
cargo test --lib
```

All 17 Rust unit tests should pass.
