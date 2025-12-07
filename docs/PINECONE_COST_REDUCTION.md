# Pinecone Cost Reduction Strategy

We have implemented a **Caching Layer** to reduce the number of read operations (queries) sent to Pinecone. This is the most effective way to reduce costs immediately.

## 1. Caching Layer (Implemented)
**Impact:** Reduces costs by 100% for repeated queries.
**Mechanism:** 
- We intercept every call to `hybrid_search_parallel` and `hybrid_search_document_aware`.
- We generate a unique "cache key" based on the query embedding, filters (Space ID, Document ID, ACL tags), and top_k parameters.
- We check a local SQLite database (`pinecone_cache.db`) for this key.
- **Hit:** If found and not expired (default TTL: 24 hours), we return the cached results immediately. **Zero Pinecone cost.**
- **Miss:** If not found, we query Pinecone, return the results, and save them to the cache for next time.

**Files Created/Modified:**
- `pinecone_cache.py`: Handles the SQLite database operations.
- `pinecone_client.py`: Integrated the cache into the search functions.

## 2. Potential Future Steps (for further reduction)

If you need to reduce costs further, consider these advanced strategies:

### A. Local Keyword Search (Estimated 5-15% reduction)
**Concept:** Use a local keyword search (BM25) on your data stored in Supabase or a local index.
**Logic:** 
1. Run a cheap local keyword search first.
2. If it returns high-confidence matches (exact phrase matches), use those and skip Pinecone.
3. Only query Pinecone if local search results are ambiguous or low confidence.
**Complexity:** High. Requires maintaining a local index or optimizing Supabase text search.

### B. Semantic Cache with Similarity (Estimated 10-20% reduction)
**Concept:** Instead of exact cache matches, allow "similar" queries to hit the cache.
**Logic:**
1. When a query comes in, check if a *semantically similar* query was asked recently (e.g., "How to bake a cake" vs "How do I bake a cake").
2. If close enough (cosine similarity > 0.95), return the cached result of the previous query.
**Complexity:** Medium. Requires a small local vector index for the cache keys themselves.

### C. Optimizing Top-K
**Concept:** Reduce the `top_k` parameter (number of results requested).
**Impact:** Minor. Pinecone charges per query, but retrieving fewer vectors saves bandwidth and processing time, though likely not direct "Read Unit" costs unless you are retrieving massive amounts.

## Configuration
You can adjust the Time-To-Live (TTL) of the cache in `pinecone_client.py` by modifying the initialization:
```python
pinecone_cache = PineconeCache(ttl_hours=24)
```
Set it to a higher value (e.g., 168 hours = 1 week) for static datasets to save even more.
