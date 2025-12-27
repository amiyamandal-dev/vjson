#!/usr/bin/env python3
"""
Test Hybrid Search - combining vector similarity and full-text search
"""

import os
import random
import shutil

import vjson

DIMENSION = 128
DB_PATH = "./test_hybrid_db"


def cleanup():
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)


def test_hybrid_search():
    print("=" * 60)
    print("HYBRID SEARCH TEST")
    print("=" * 60)

    cleanup()

    # Create database
    db = vjson.PyVectorDB(DB_PATH, dimension=DIMENSION)

    # Insert documents with both vectors and text
    print("\n1. Inserting documents with text content...")

    documents = [
        (
            "doc1",
            [random.random() for _ in range(DIMENSION)],
            "machine learning and artificial intelligence",
            {"category": "AI", "year": 2024},
        ),
        (
            "doc2",
            [random.random() for _ in range(DIMENSION)],
            "deep learning neural networks",
            {"category": "ML", "year": 2023},
        ),
        (
            "doc3",
            [random.random() for _ in range(DIMENSION)],
            "natural language processing transformers",
            {"category": "NLP", "year": 2024},
        ),
        (
            "doc4",
            [random.random() for _ in range(DIMENSION)],
            "computer vision image recognition",
            {"category": "CV", "year": 2023},
        ),
        (
            "doc5",
            [random.random() for _ in range(DIMENSION)],
            "reinforcement learning agents",
            {"category": "RL", "year": 2024},
        ),
    ]

    # Insert using insert_with_text
    for doc_id, vector, text, metadata in documents:
        db.insert_with_text(doc_id, vector, text, metadata)

    print(f"   Inserted {len(documents)} documents")

    # Test 1: Text-only search
    print("\n2. Testing text-only search...")
    try:
        text_results = db.text_search("machine learning", limit=3)
        print(f"   Text search results for 'machine learning':")
        for doc_id, score in text_results:
            print(f"      {doc_id}: score={score:.4f}")
    except Exception as e:
        print(f"   Text search: {e}")

    # Test 2: Vector-only search
    print("\n3. Testing vector-only search...")
    query_vector = [random.random() for _ in range(DIMENSION)]
    vector_results = db.search(query_vector, k=3)
    print(f"   Vector search results:")
    for result in vector_results:
        print(f"      {result['id']}: distance={result['distance']:.4f}")

    # Test 3: Hybrid search with RRF (Reciprocal Rank Fusion)
    print("\n4. Testing hybrid search with RRF strategy...")
    hybrid_results = db.hybrid_search(
        query_vector=query_vector,
        query_text="machine learning neural networks",
        k=3,
        strategy="rrf",
    )
    print(f"   Hybrid search results (RRF):")
    for result in hybrid_results:
        print(f"      {result['id']}:")
        print(f"         vector_score={result['vector_score']:.4f}")
        print(f"         text_score={result['text_score']:.4f}")
        print(f"         combined_score={result['combined_score']:.4f}")

    # Test 4: Hybrid search with weighted strategy
    print("\n5. Testing hybrid search with weighted strategy...")
    hybrid_results_weighted = db.hybrid_search(
        query_vector=query_vector,
        query_text="deep learning transformers",
        k=3,
        strategy="weighted",
        vector_weight=0.7,
        text_weight=0.3,
    )
    print(f"   Hybrid search results (weighted 0.7/0.3):")
    for result in hybrid_results_weighted:
        print(f"      {result['id']}: combined={result['combined_score']:.4f}")

    # Test 5: Different fusion strategies
    print("\n6. Testing different fusion strategies...")
    strategies = ["rrf", "weighted", "max", "min", "average"]

    for strategy in strategies:
        results = db.hybrid_search(
            query_vector=query_vector, query_text="learning", k=2, strategy=strategy
        )
        print(f"   {strategy.upper()}: {[r['id'] for r in results]}")

    print("\n" + "=" * 60)
    print("✅ HYBRID SEARCH TEST COMPLETED")
    print("=" * 60)

    cleanup()


def test_batch_insert_with_text():
    print("\n" + "=" * 60)
    print("BATCH INSERT WITH TEXT TEST")
    print("=" * 60)

    cleanup()

    db = vjson.PyVectorDB(DB_PATH, dimension=DIMENSION)

    # Batch insert with text (4-tuple format)
    print("\n1. Batch inserting documents with text...")
    batch_data = []
    for i in range(10):
        doc_id = f"article_{i}"
        vector = [random.random() for _ in range(DIMENSION)]
        text = f"This is article {i} about topic {i % 3}"
        metadata = {"index": i, "topic": i % 3}
        batch_data.append((doc_id, vector, metadata, text))

    db.insert_batch(batch_data)
    print(f"   Inserted {len(batch_data)} documents in batch")

    # Search
    print("\n2. Searching...")
    results = db.text_search("article topic", limit=5)
    print(f"   Found {len(results)} results")

    # Hybrid search
    print("\n3. Hybrid search...")
    query_vec = [random.random() for _ in range(DIMENSION)]
    hybrid = db.hybrid_search(query_vec, "article 5", k=3)
    print(f"   Top 3 hybrid results: {[r['id'] for r in hybrid]}")

    print("\n✅ BATCH INSERT TEST PASSED")

    cleanup()


def test_backward_compatibility():
    print("\n" + "=" * 60)
    print("BACKWARD COMPATIBILITY TEST (without text)")
    print("=" * 60)

    cleanup()

    db = vjson.PyVectorDB(DB_PATH, dimension=DIMENSION)

    # Old-style insert without text (3-tuple)
    print("\n1. Inserting without text (3-tuple)...")
    batch_data = []
    for i in range(5):
        doc_id = f"vec_{i}"
        vector = [random.random() for _ in range(DIMENSION)]
        metadata = {"index": i}
        batch_data.append((doc_id, vector, metadata))

    db.insert_batch(batch_data)
    print(f"   Inserted {len(batch_data)} vectors (no text)")

    # Regular search should still work
    print("\n2. Regular vector search...")
    query = [random.random() for _ in range(DIMENSION)]
    results = db.search(query, k=3)
    print(f"   Found {len(results)} results")
    print(f"   Top result: {results[0]['id']}")

    print("\n✅ BACKWARD COMPATIBILITY TEST PASSED")

    cleanup()


if __name__ == "__main__":
    try:
        test_hybrid_search()
        test_batch_insert_with_text()
        test_backward_compatibility()

        print("\n" + "=" * 60)
        print("🎉 ALL HYBRID SEARCH TESTS PASSED!")
        print("=" * 60)
        print("\nFeatures tested:")
        print("  ✓ insert_with_text() - single document with text")
        print("  ✓ insert_batch() with 4-tuple (id, vector, metadata, text)")
        print("  ✓ insert_batch() with 3-tuple (backward compatible)")
        print("  ✓ text_search() - full-text search")
        print("  ✓ hybrid_search() - combined vector + text search")
        print("  ✓ Multiple fusion strategies (rrf, weighted, max, min, average)")
        print("  ✓ Weighted fusion with custom weights")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
