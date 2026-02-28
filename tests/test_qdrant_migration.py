"""
Comprehensive Qdrant Migration Tests
Tests all vector DB operations that were previously backed by Pinecone.

Run with:
    cd /Users/davak/Documents/nuton_rag
    source venv/bin/activate
    python3 -m pytest tests/test_qdrant_migration.py -v

Or to run a single section:
    python3 tests/test_qdrant_migration.py
"""

import sys
import os
import uuid
import logging
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ── Colour helpers ────────────────────────────────────────────────────────────
GREEN = "\033[92m"
RED   = "\033[91m"
CYAN  = "\033[96m"
RESET = "\033[0m"

_passed: list[str] = []
_failed: list[str] = []


def ok(name: str) -> None:
    _passed.append(name)
    print(f"  {GREEN}✓{RESET} {name}")


def fail(name: str, err: str) -> None:
    _failed.append(name)
    print(f"  {RED}✗{RESET} {name}: {err}")


def section(title: str) -> None:
    print(f"\n{CYAN}━━ {title} ━━{RESET}")


# ── Fixed test data ───────────────────────────────────────────────────────────
TEST_DOC_ID   = f"test-migration-doc-{uuid.uuid4().hex[:8]}"
TEST_SPACE_ID = f"test-migration-space-{uuid.uuid4().hex[:8]}"
VECTOR_DIM    = 1024   # Jina CLIP-v2


def _dummy_vector(seed: int = 0) -> list[float]:
    """Deterministic pseudo-random unit vector."""
    import math
    raw = [math.sin(seed * 13.37 + i * 7.91) for i in range(VECTOR_DIM)]
    mag = math.sqrt(sum(x * x for x in raw))
    return [x / mag for x in raw]


# ═══════════════════════════════════════════════════════════════════════════════
# 1. IMPORT & CONNECTION
# ═══════════════════════════════════════════════════════════════════════════════
section("1 · Import & Connection")

try:
    from clients.qdrant_client import (
        client, COLLECTION,
        upsert_vectors, upsert_image_vectors,
        hybrid_search, hybrid_search_parallel,
        rerank_results,
        hybrid_search_document_aware, rerank_results_document_aware,
        delete_vectors, create_index,
        fetch_all_document_chunks, fetch_chunks_for_gap,
        fetch_document_vector_ids, update_vector_metadata,
        fetch_chunks_by_space, calculate_coverage_from_chunks,
        _to_qdrant_id, _build_filter, _keyword_search, _rrf_merge,
    )
    ok("qdrant_client module imports successfully")
except Exception as e:
    fail("qdrant_client module imports", str(e))
    print(f"\n{RED}Cannot continue — module import failed.{RESET}")
    sys.exit(1)

try:
    info = client.get_collection(COLLECTION)
    ok(f"Connected to Qdrant — collection '{COLLECTION}' has {info.points_count} points")
except Exception as e:
    fail("Connect to Qdrant collection", str(e))
    print(f"\n{RED}Cannot continue — connection failed.{RESET}")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════
section("2 · Helper functions")

try:
    uid = _to_qdrant_id("test::chunk_0")
    assert len(uid) == 36, "UUID should be 36 chars"
    assert _to_qdrant_id("test::chunk_0") == uid, "UUID5 must be deterministic"
    ok("_to_qdrant_id produces deterministic UUID5")
except Exception as e:
    fail("_to_qdrant_id", str(e))

try:
    f = _build_filter(doc_id="d1", space_id="s1")
    assert f is not None
    f_none = _build_filter()
    assert f_none is None
    ok("_build_filter returns correct Filter or None")
except Exception as e:
    fail("_build_filter", str(e))

try:
    result = create_index(dimension=1024)
    ok("create_index is a no-op (compatibility stub)")
except Exception as e:
    fail("create_index no-op", str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# 3. UPSERT
# ═══════════════════════════════════════════════════════════════════════════════
section("3 · Upsert vectors")

test_chunks = [
    {
        "text": f"Test chunk {i} about artificial intelligence and machine learning.",
        "pages": [i + 1],
        "chapter_number": 1,
        "chapter_title": "Introduction",
        "start_index": i * 500,
        "end_index":   i * 500 + 450,
    }
    for i in range(5)
]
test_embeddings = [{"embedding": _dummy_vector(i)} for i in range(5)]

try:
    upsert_vectors(
        doc_id=TEST_DOC_ID,
        space_id=TEST_SPACE_ID,
        embeddings=test_embeddings,
        chunks=test_chunks,
        batch_size=10,
        source_file="test_document.pdf",
    )
    ok("upsert_vectors (5 text chunks)")
except Exception as e:
    fail("upsert_vectors (text)", str(e))

# Allow a short propagation delay
time.sleep(1)

try:
    test_images = [
        {
            "id": f"img_{i}",
            "page": i + 1,
            "position_in_doc": i,
            "image_base64": "dGVzdA==",  # base64("test") — small, should go inline
        }
        for i in range(2)
    ]
    image_embeddings = [_dummy_vector(100 + i) for i in range(2)]
    upsert_image_vectors(
        doc_id=TEST_DOC_ID,
        space_id=TEST_SPACE_ID,
        images=test_images,
        embeddings=image_embeddings,
        source_file="test_document.pdf",
    )
    ok("upsert_image_vectors (2 image vectors)")
except Exception as e:
    fail("upsert_image_vectors", str(e))

# Give Qdrant a moment to index
time.sleep(1)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. SEARCH
# ═══════════════════════════════════════════════════════════════════════════════
section("4 · Search")

query_vec = _dummy_vector(0)   # Same as first chunk — should rank high

try:
    hits = hybrid_search(query_emb=query_vec, doc_id=TEST_DOC_ID, top_k=5)
    assert isinstance(hits, list), "Should return list"
    assert len(hits) > 0, "Should find at least one result"
    assert "id" in hits[0], "Hit must have 'id'"
    assert "score" in hits[0], "Hit must have 'score'"
    assert "metadata" in hits[0], "Hit must have 'metadata'"
    ok(f"hybrid_search — found {len(hits)} hits")
except Exception as e:
    fail("hybrid_search", str(e))

try:
    hits = hybrid_search_parallel(
        query_emb=query_vec,
        doc_id=TEST_DOC_ID,
        space_id=TEST_SPACE_ID,
        top_k=5,
    )
    assert len(hits) > 0
    ok(f"hybrid_search_parallel — found {len(hits)} hits")
except Exception as e:
    fail("hybrid_search_parallel", str(e))

try:
    hits = hybrid_search_parallel(
        query_emb=query_vec,
        query_sparse={"some": "ignored"},   # Should be silently ignored
        doc_id=TEST_DOC_ID,
        top_k=5,
    )
    assert len(hits) > 0
    ok("hybrid_search_parallel with sparse param (silently ignored)")
except Exception as e:
    fail("hybrid_search_parallel with sparse param", str(e))

try:
    hits = hybrid_search(query_emb=query_vec, space_id=TEST_SPACE_ID, top_k=10)
    assert isinstance(hits, list)
    ok(f"hybrid_search with space_id filter — found {len(hits)} hits")
except Exception as e:
    fail("hybrid_search with space_id filter", str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# 5. RERANKING
# ═══════════════════════════════════════════════════════════════════════════════
section("5 · Reranking")

try:
    hits = hybrid_search(query_emb=query_vec, doc_id=TEST_DOC_ID, top_k=5)
    reranked = rerank_results("artificial intelligence machine learning", hits, top_n=5)
    assert isinstance(reranked, list)
    assert len(reranked) == len(hits)
    assert all("rerank_score" in h for h in reranked)
    ok("rerank_results — all hits have rerank_score")
except Exception as e:
    fail("rerank_results", str(e))

try:
    hits = hybrid_search(query_emb=query_vec, doc_id=TEST_DOC_ID, top_k=5)
    reranked = rerank_results_document_aware(
        "artificial intelligence",
        hits,
        top_n_per_doc=3,
        max_total_results=5,
    )
    assert isinstance(reranked, list)
    ok(f"rerank_results_document_aware — {len(reranked)} results")
except Exception as e:
    fail("rerank_results_document_aware", str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# 6. DOCUMENT-AWARE SEARCH
# ═══════════════════════════════════════════════════════════════════════════════
section("6 · Document-aware search")

# Upsert a second document so we can test cross-doc search
DOC_ID_2 = f"test-migration-doc2-{uuid.uuid4().hex[:8]}"
try:
    chunks_2 = [
        {"text": f"Second document chunk {i} about neural networks.", "pages": [i + 1]}
        for i in range(3)
    ]
    emb_2 = [{"embedding": _dummy_vector(200 + i)} for i in range(3)]
    upsert_vectors(
        doc_id=DOC_ID_2,
        space_id=TEST_SPACE_ID,
        embeddings=emb_2,
        chunks=chunks_2,
    )
    time.sleep(1)
    ok(f"Second test document upserted ({DOC_ID_2})")
except Exception as e:
    fail("Upsert second test document", str(e))

try:
    doc_aware_hits = hybrid_search_document_aware(
        query_emb=query_vec,
        document_ids=[TEST_DOC_ID, DOC_ID_2],
        space_id=TEST_SPACE_ID,
        top_k_per_doc=3,
    )
    assert isinstance(doc_aware_hits, list)
    assert len(doc_aware_hits) > 0
    ok(f"hybrid_search_document_aware — found {len(doc_aware_hits)} hits across 2 docs")
except Exception as e:
    fail("hybrid_search_document_aware", str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# 7. FETCH ALL DOCUMENT CHUNKS
# ═══════════════════════════════════════════════════════════════════════════════
section("7 · fetch_all_document_chunks (notes pipeline)")

try:
    chunks = fetch_all_document_chunks(
        document_id=TEST_DOC_ID,
        space_id=TEST_SPACE_ID,
        max_chunks=100,
        enable_gap_filling=False,
    )
    assert isinstance(chunks, list)
    assert len(chunks) > 0, "Should retrieve at least one chunk"
    assert all("metadata" in c for c in chunks)
    ok(f"fetch_all_document_chunks — retrieved {len(chunks)} chunks")
except Exception as e:
    fail("fetch_all_document_chunks", str(e))

try:
    # Test with gap filling enabled
    chunks_gf = fetch_all_document_chunks(
        document_id=TEST_DOC_ID,
        max_chunks=100,
        enable_gap_filling=True,
        target_coverage=0.5,   # Low threshold so test doesn't take long
    )
    assert isinstance(chunks_gf, list)
    ok(f"fetch_all_document_chunks with gap-filling — {len(chunks_gf)} chunks")
except Exception as e:
    fail("fetch_all_document_chunks with gap-filling", str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# 8. COVERAGE CALCULATION
# ═══════════════════════════════════════════════════════════════════════════════
section("8 · Coverage calculation")

try:
    sample_chunks = [
        {"metadata": {"start_index": 0,   "end_index": 100}},
        {"metadata": {"start_index": 100, "end_index": 200}},
        {"metadata": {"start_index": 300, "end_index": 400}},  # gap at 200-300
    ]
    cov = calculate_coverage_from_chunks(sample_chunks)
    assert 0 < cov["coverage_percentage"] < 1.0
    assert len(cov["gaps"]) == 1
    assert cov["gaps"][0]["start"] == 200
    assert cov["gaps"][0]["end"]   == 300
    ok(f"calculate_coverage_from_chunks — {cov['coverage_percentage']:.0%} coverage, 1 gap detected")
except Exception as e:
    fail("calculate_coverage_from_chunks", str(e))

try:
    empty_cov = calculate_coverage_from_chunks([])
    assert empty_cov["coverage_percentage"] == 0.0
    ok("calculate_coverage_from_chunks with empty list returns 0%")
except Exception as e:
    fail("calculate_coverage_from_chunks (empty)", str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# 9. FETCH DOCUMENT VECTOR IDs  (space membership update pipeline)
# ═══════════════════════════════════════════════════════════════════════════════
section("9 · fetch_document_vector_ids")

fetched_ids: list[str] = []
try:
    fetched_ids = fetch_document_vector_ids(TEST_DOC_ID, max_ids=100)
    assert isinstance(fetched_ids, list)
    assert len(fetched_ids) > 0, "Should find vectors for the test document"
    ok(f"fetch_document_vector_ids — found {len(fetched_ids)} IDs")
except Exception as e:
    fail("fetch_document_vector_ids", str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# 10. UPDATE VECTOR METADATA  (space membership update pipeline)
# ═══════════════════════════════════════════════════════════════════════════════
section("10 · update_vector_metadata")

try:
    if fetched_ids:
        updated = update_vector_metadata(
            fetched_ids,
            {"nuton_space_id": TEST_SPACE_ID, "_test_tag": "migration_test"},
        )
        assert updated == len(fetched_ids), f"Expected {len(fetched_ids)} updates, got {updated}"
        ok(f"update_vector_metadata — updated {updated} vectors")
    else:
        fail("update_vector_metadata", "skipped — no IDs from previous step")
except Exception as e:
    fail("update_vector_metadata", str(e))

try:
    zero = update_vector_metadata([], {"foo": "bar"})
    assert zero == 0
    ok("update_vector_metadata with empty list returns 0")
except Exception as e:
    fail("update_vector_metadata (empty list)", str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# 11. FETCH CHUNKS BY SPACE
# ═══════════════════════════════════════════════════════════════════════════════
section("11 · fetch_chunks_by_space")

try:
    space_chunks = fetch_chunks_by_space(
        space_id=TEST_SPACE_ID,
        document_ids=[TEST_DOC_ID, DOC_ID_2],
        max_chunks_per_doc=50,
        enable_gap_filling=False,
    )
    assert isinstance(space_chunks, list)
    assert len(space_chunks) > 0
    ok(f"fetch_chunks_by_space — retrieved {len(space_chunks)} chunks from 2 docs")
except Exception as e:
    fail("fetch_chunks_by_space", str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# 12. HYBRID SEARCH (dense + keyword)
# ═══════════════════════════════════════════════════════════════════════════════
section("12 · Hybrid search (dense + keyword)")

try:
    dense_hits = hybrid_search(query_emb=query_vec, doc_id=TEST_DOC_ID, top_k=5)
    hybrid_hits = hybrid_search(
        query_emb=query_vec,
        query_text="artificial intelligence machine learning",
        doc_id=TEST_DOC_ID,
        top_k=5,
    )
    assert isinstance(hybrid_hits, list), "hybrid_search should return a list"
    assert len(hybrid_hits) > 0, "Should return at least one result"
    assert all("id" in h and "metadata" in h for h in hybrid_hits), "Hits must have id and metadata"
    ok(f"hybrid_search with query_text — {len(hybrid_hits)} hits (dense-only had {len(dense_hits)})")
except Exception as e:
    fail("hybrid_search with query_text", str(e))

try:
    from clients.qdrant_client import _build_filter as _bf
    kw_hits = _keyword_search(
        query_text="artificial intelligence",
        query_filter=_bf(doc_id=TEST_DOC_ID),
        top_k=5,
    )
    assert isinstance(kw_hits, list), "_keyword_search should return a list"
    # All returned chunks should contain at least one keyword word
    for h in kw_hits:
        text = h["metadata"].get("text", "").lower()
        assert "artificial" in text or "intelligence" in text or "machine" in text or "learning" in text, \
            f"Keyword hit text doesn't contain expected keywords: {text[:100]}"
    ok(f"_keyword_search returns {len(kw_hits)} chunks containing query keywords")
except Exception as e:
    fail("_keyword_search returns keyword-matching chunks", str(e))

try:
    list_a = [{"id": "a", "score": 0.9, "metadata": {}}, {"id": "b", "score": 0.8, "metadata": {}}]
    list_b = [{"id": "b", "score": 0.95, "metadata": {}}, {"id": "c", "score": 0.7, "metadata": {}}]
    merged = _rrf_merge(list_a, list_b, top_k=3)
    ids = [h["id"] for h in merged]
    assert len(merged) == 3, f"Expected 3 results, got {len(merged)}"
    assert len(set(ids)) == 3, "No duplicates — a, b, c all present"
    assert "b" in ids, "b appears in both lists so must be present"
    assert all("rrf_score" in h for h in merged), "All hits must have rrf_score"
    # b appears in both lists so must have higher RRF score than a or c alone
    b_score = next(h["rrf_score"] for h in merged if h["id"] == "b")
    a_score = next(h["rrf_score"] for h in merged if h["id"] == "a")
    assert b_score > a_score, "b (in both lists) should outscore a (in one list)"
    ok("_rrf_merge correctly combines and deduplicates two result sets")
except Exception as e:
    fail("_rrf_merge combines and deduplicates", str(e))

try:
    hybrid_doc_hits = hybrid_search_document_aware(
        query_emb=query_vec,
        query_text="artificial intelligence machine learning",
        document_ids=[TEST_DOC_ID, DOC_ID_2],
        space_id=TEST_SPACE_ID,
        top_k_per_doc=3,
    )
    assert isinstance(hybrid_doc_hits, list), "Should return a list"
    assert len(hybrid_doc_hits) > 0, "Should return hits"
    doc_ids_in_results = {h["metadata"].get("source_document_id") for h in hybrid_doc_hits}
    assert TEST_DOC_ID in doc_ids_in_results, "TEST_DOC_ID should appear in results"
    ok(f"hybrid_search_document_aware with query_text — {len(hybrid_doc_hits)} hits from {len(doc_ids_in_results)} doc(s)")
except Exception as e:
    fail("hybrid_search_document_aware with query_text", str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# 13. DELETE VECTORS
# ═══════════════════════════════════════════════════════════════════════════════
section("13 · delete_vectors (cleanup)")

try:
    delete_vectors(doc_id=TEST_DOC_ID, space_id=TEST_SPACE_ID)
    time.sleep(0.5)
    remaining = hybrid_search(query_emb=query_vec, doc_id=TEST_DOC_ID, top_k=10)
    assert len(remaining) == 0, f"Expected 0 results after delete, got {len(remaining)}"
    ok("delete_vectors — doc+space filter deletes all matching vectors")
except Exception as e:
    fail("delete_vectors (doc+space)", str(e))

try:
    delete_vectors(doc_id=DOC_ID_2)
    time.sleep(0.5)
    remaining2 = hybrid_search(query_emb=query_vec, doc_id=DOC_ID_2, top_k=10)
    assert len(remaining2) == 0
    ok("delete_vectors — doc-only filter deletes all matching vectors")
except Exception as e:
    fail("delete_vectors (doc only)", str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# 13. PIPELINE INTEGRATION SMOKE TESTS
# ═══════════════════════════════════════════════════════════════════════════════
section("14 · Pipeline integration smoke tests")

# ── Notes pipeline ────────────────────────────────────────────────────────────
try:
    from clients.qdrant_client import fetch_all_document_chunks as _fac
    assert callable(_fac)
    ok("notes pipeline — fetch_all_document_chunks importable")
except Exception as e:
    fail("notes pipeline import", str(e))

# ── Course pipeline ───────────────────────────────────────────────────────────
try:
    from clients.qdrant_client import upsert_vectors as _uv, hybrid_search as _hs, rerank_results as _rr
    assert callable(_uv) and callable(_hs) and callable(_rr)
    ok("course pipeline — upsert_vectors, hybrid_search, rerank_results importable")
except Exception as e:
    fail("course pipeline imports", str(e))

# ── Quiz pipeline ─────────────────────────────────────────────────────────────
try:
    from clients.qdrant_client import hybrid_search, fetch_all_document_chunks, rerank_results, calculate_coverage_from_chunks
    assert callable(hybrid_search) and callable(fetch_all_document_chunks)
    assert callable(rerank_results) and callable(calculate_coverage_from_chunks)
    ok("quiz pipeline — all required functions importable")
except Exception as e:
    fail("quiz pipeline imports", str(e))

# ── Flashcard pipeline ────────────────────────────────────────────────────────
try:
    from clients.qdrant_client import hybrid_search, fetch_all_document_chunks, rerank_results, calculate_coverage_from_chunks
    assert callable(hybrid_search) and callable(fetch_all_document_chunks)
    assert callable(rerank_results) and callable(calculate_coverage_from_chunks)
    ok("flashcard pipeline — all required functions importable")
except Exception as e:
    fail("flashcard pipeline imports", str(e))

# ── Space pipeline ────────────────────────────────────────────────────────────
try:
    from clients.qdrant_client import fetch_document_vector_ids, update_vector_metadata, hybrid_search, rerank_results
    assert callable(fetch_document_vector_ids) and callable(update_vector_metadata)
    ok("space pipeline — fetch_document_vector_ids, update_vector_metadata importable")
except Exception as e:
    fail("space pipeline imports", str(e))

# ── Top-level pipeline imports ────────────────────────────────────────────────
try:
    import importlib
    spec = importlib.util.spec_from_file_location(
        "pipeline_imports",
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pipeline.py"),
    )
    # Just verify the qdrant_client import line is correct in the file
    with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pipeline.py")) as f:
        content = f.read()
    assert "from clients.qdrant_client import" in content
    assert "from clients.pinecone_client import" not in content
    ok("pipeline.py — uses qdrant_client (no pinecone_client references)")
except Exception as e:
    fail("pipeline.py import check", str(e))

try:
    with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "requirements.txt")) as f:
        req = f.read()
    assert "qdrant-client" in req
    assert "pinecone" not in req
    ok("requirements.txt — qdrant-client present, pinecone removed")
except Exception as e:
    fail("requirements.txt check", str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
total = len(_passed) + len(_failed)
print(f"\n{'━'*50}")
print(f"  Results: {GREEN}{len(_passed)} passed{RESET}  /  {RED}{len(_failed)} failed{RESET}  (total {total})")
if _failed:
    print(f"\n  {RED}Failed tests:{RESET}")
    for name in _failed:
        print(f"    • {name}")
print(f"{'━'*50}\n")

sys.exit(0 if not _failed else 1)
