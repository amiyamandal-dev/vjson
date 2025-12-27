# Metadata Filtering Guide

VJson supports powerful metadata filtering to combine vector similarity search with structured metadata queries.

## Overview

Metadata filtering allows you to:
1. **Find similar vectors** based on embeddings (HNSW search)
2. **Filter results** based on metadata conditions

This enables **hybrid search**: "Find similar documents that are also in the 'tech' category with a score > 0.5"

## Basic Usage

```python
import vjson

db = vjson.PyVectorDB("./db", dimension=128)

# Insert with metadata
db.insert("doc1", [0.1] * 128, {
    "category": "tech",
    "score": 0.8,
    "author": "Alice"
})

# Search with filter
results = db.search(
    query=[0.1] * 128,
    k=10,
    filter={"category": "tech"}  # Only return 'tech' results
)
```

## Filter Operators

### 1. Equality

```python
# Simple equality
filter={"category": "tech"}

# Multiple fields (AND)
filter={
    "category": "tech",
    "author": "Alice"
}
```

### 2. Numeric Comparisons

```python
# Greater than
filter={"score": {"$gt": 0.5}}

# Less than
filter={"score": {"$lt": 0.9}}

# Greater or equal
filter={"score": {"$gte": 0.5}}

# Less or equal
filter={"score": {"$lte": 0.9}}
```

### 3. Array Membership

```python
# In array
filter={"category": {"$in": ["tech", "science", "math"]}}

# Not in array
filter={"category": {"$nin": ["spam", "ads"]}}
```

### 4. Existence Check

```python
# Field exists
filter={"premium": {"$exists": True}}

# Field doesn't exist
filter={"deleted": {"$exists": False}}
```

### 5. Nested Fields (Dot Notation)

```python
# Access nested fields
filter={"user.age": {"$gte": 18}}

filter={"settings.notifications.email": True}
```

## Complex Queries

### Combining Multiple Conditions

All conditions at the top level are combined with AND:

```python
filter={
    "category": "tech",           # AND
    "score": {"$gt": 0.5},        # AND
    "user.age": {"$gte": 18}      # AND
}
```

### Range Queries

```python
# Score between 0.4 and 0.8
filter={
    "score": {"$gte": 0.4},
    "score": {"$lte": 0.8}
}

# Or use separate queries and combine results
```

### Real-World Examples

#### Example 1: E-commerce Product Search

```python
# Find similar products that are:
# - In stock
# - Price under $100
# - Rated 4+ stars
# - In electronics category

results = db.search(
    query=product_embedding,
    k=20,
    filter={
        "in_stock": True,
        "price": {"$lte": 100},
        "rating": {"$gte": 4.0},
        "category": "electronics"
    }
)
```

#### Example 2: Document Search

```python
# Find similar documents that are:
# - Published in 2024
# - Tagged as 'research'
# - Author is verified
# - Has citations

results = db.search(
    query=document_embedding,
    k=10,
    filter={
        "year": 2024,
        "tags": {"$in": ["research", "academic"]},
        "author.verified": True,
        "citations": {"$gt": 0}
    }
)
```

#### Example 3: User Recommendation

```python
# Find similar users who are:
# - Age 25-40
# - Premium subscribers
# - Active in last 30 days
# - In same city

results = db.search(
    query=user_embedding,
    k=50,
    filter={
        "age": {"$gte": 25},
        "age_max": {"$lte": 40},  # Note: need separate field
        "subscription": "premium",
        "last_active_days": {"$lte": 30},
        "city": "San Francisco"
    }
)
```

## Performance Considerations

### How Filtering Works

```
1. HNSW Search → Find top K×N similar vectors (fast, O(log N))
2. Filter → Apply metadata conditions (slower, O(K×N))
3. Truncate → Return top K after filtering
```

**Important**: Filtering happens AFTER vector search, not before.

### Optimization Tips

#### ✅ Good Practices

1. **Use selective filters**
   ```python
   # Good: Filters out 90% of results
   filter={"premium": True}  # Only 10% of users
   ```

2. **Search with higher K when filtering**
   ```python
   # If you expect 50% match rate, search 2×K
   results = db.search(query, k=20, filter={"category": "tech"})
   # Internally searches k=40, then filters, then returns 20
   ```

3. **Index frequently-filtered fields**
   - Store common filters in simple fields (not nested)
   - Use boolean flags for common categories

#### ❌ Avoid

1. **Don't filter on high-cardinality fields as primary filter**
   ```python
   # Bad: Every user has unique ID
   filter={"user_id": "12345"}  # Just use get_metadata() instead
   ```

2. **Don't use filters that match everything**
   ```python
   # Bad: No filtering benefit
   filter={"exists": True}  # Matches all documents
   ```

3. **Don't expect pre-filtered search**
   ```python
   # This does NOT search only within "tech" category
   # It searches all vectors, then filters
   filter={"category": "tech"}
   ```

### When to Use Filtering

| Use Case | Should Use Filtering? |
|----------|----------------------|
| Category restriction | ✅ Yes |
| Score threshold | ✅ Yes |
| Date range | ✅ Yes |
| User permissions | ✅ Yes |
| Exact ID lookup | ❌ No (use get_metadata) |
| Very rare conditions (<1%) | ⚠️ Maybe (search higher K) |

## Limitations

### Current Limitations

1. **No OR operator** (only AND)
   - Workaround: Run multiple queries

2. **No full-text search**
   - Use vector search for semantic similarity
   - Use exact match for keywords

3. **Post-search filtering**
   - Cannot pre-filter index
   - May return fewer than K results if filter is very selective

4. **No aggregations**
   - Cannot count, sum, or group by metadata
   - Results are individual vectors only

### Future Enhancements

Potential future features:
- OR operator: `{"$or": [{"a": 1}, {"b": 2}]}`
- Pre-filtered indexes: Build separate indexes per category
- Regex matching: `{"name": {"$regex": "^A.*"}}`
- Array contains: `{"tags": {"$contains": "important"}}`

## Best Practices

### 1. Design Metadata Schema

```python
# Good metadata schema
{
    "category": "tech",        # Simple, indexed
    "score": 0.8,              # Numeric, filterable
    "published": 1640995200,   # Unix timestamp (numeric)
    "tags": ["ai", "ml"],      # Array for $in queries
    "author": {                # Nested for organization
        "name": "Alice",
        "verified": True
    }
}
```

### 2. Test Filter Selectivity

```python
# Check how many results pass filter
all_results = db.search(query, k=1000)
filtered = db.search(query, k=1000, filter={"category": "tech"})

selectivity = len(filtered) / len(all_results)
print(f"Filter selectivity: {selectivity:.2%}")

# Aim for 10-50% selectivity for good performance
```

### 3. Combine with Application Logic

```python
# First: Vector search with broad filter
results = db.search(query, k=50, filter={
    "category": {"$in": ["tech", "science"]},
    "score": {"$gt": 0.3}
})

# Then: Application-level filtering for complex logic
final_results = [
    r for r in results
    if custom_business_logic(r['metadata'])
]
```

## Examples

See `test_filters.py` for comprehensive examples of all filter types.

---

**Last Updated**: 2025-12-27  
**Version**: 1.0
