import os
import uuid
import logging
import concurrent.futures
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (
    PointStruct, Filter, FieldCondition, MatchValue, MatchAny,
    Range, FilterSelector, PointIdsList, MatchText,
)

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "https://qdrant.nuton.app:443")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "c2e8d3b15a811c46120f7b7bfc34898abe15e13a5be1c7012963cb8cb0405dda")
COLLECTION = "nuton-index-multi-modal"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)
    logger.info("Initialized Qdrant client")
except Exception as e:
    logger.error(f"Failed to initialize Qdrant client: {str(e)}")
    raise

try:
    client.create_payload_index(
        collection_name=COLLECTION,
        field_name="text",
        field_schema="text",
    )
    logger.info("Text payload index ensured on 'text' field")
except Exception as e:
    logger.warning(f"Could not ensure text index (may already exist): {e}")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_qdrant_id(string_id: str) -> str:
    """Convert an arbitrary string ID to a UUID5 string for Qdrant compatibility."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, string_id))


def _build_filter(
    doc_id: Optional[str] = None,
    space_id: Optional[str] = None,
    acl_tags: Optional[Any] = None,
    extra_conditions: Optional[List] = None,
) -> Optional[Filter]:
    """Build a Qdrant Filter from standard query parameters."""
    conditions = []
    if doc_id:
        conditions.append(FieldCondition(key="document_id", match=MatchValue(value=doc_id)))
    if space_id:
        conditions.append(FieldCondition(key="space_id", match=MatchValue(value=space_id)))
    if acl_tags:
        tags = list(acl_tags) if not isinstance(acl_tags, list) else acl_tags
        conditions.append(FieldCondition(key="acl_tags", match=MatchAny(any=tags)))
    if extra_conditions:
        conditions.extend(extra_conditions)
    return Filter(must=conditions) if conditions else None


def _rerank_with_cohere(query: str, texts: List[str]) -> List[float]:
    """
    Rerank texts using Cohere's rerank API.
    Falls back to uniform scores (0.5) if COHERE_API_KEY is not set.

    Returns a list of relevance scores aligned with `texts`.
    """
    cohere_api_key = os.getenv("COHERE_API_KEY")
    if not cohere_api_key:
        logger.warning("COHERE_API_KEY not set — skipping rerank, using uniform scores")
        return [0.5] * len(texts)

    try:
        import cohere
        co = cohere.Client(api_key=cohere_api_key)
        response = co.rerank(
            model="rerank-english-v3.0",
            query=query,
            documents=texts,
            top_n=len(texts),
        )
        scores = [0.0] * len(texts)
        for r in response.results:
            scores[r.index] = r.relevance_score
        return scores
    except Exception as e:
        logger.warning(f"Cohere rerank failed ({e}) — using uniform scores")
        return [0.5] * len(texts)


def _keyword_search(
    query_text: str,
    query_filter: Optional[Filter],
    top_k: int,
) -> List[Dict[str, Any]]:
    """
    Retrieve keyword-matching chunks via MatchText, then rank them with BM25.
    Returns up to top_k hits ordered by BM25 relevance score.
    """
    text_cond = FieldCondition(key="text", match=MatchText(text=query_text))
    conditions = list(query_filter.must) if query_filter else []
    combined = Filter(must=conditions + [text_cond])

    results, _ = client.scroll(
        collection_name=COLLECTION,
        scroll_filter=combined,
        limit=min(top_k * 3, 200),
        with_payload=True,
        with_vectors=False,
    )
    if not results:
        return []

    from rank_bm25 import BM25Okapi
    texts = [p.payload.get("text", "") for p in results]
    bm25 = BM25Okapi([t.lower().split() for t in texts])
    scores = bm25.get_scores(query_text.lower().split())

    hits = []
    for point, score in zip(results, scores):
        original_id = point.payload.get("_original_id", str(point.id))
        hits.append({
            "id": original_id,
            "score": float(score),
            "metadata": {k: v for k, v in point.payload.items() if k != "_original_id"},
        })
    hits.sort(key=lambda x: x["score"], reverse=True)
    return hits[:top_k]


def _rrf_merge(
    dense_hits: List[Dict[str, Any]],
    keyword_hits: List[Dict[str, Any]],
    top_k: int,
    k: int = 60,
) -> List[Dict[str, Any]]:
    """
    Reciprocal Rank Fusion of dense and keyword result sets.
    Score formula: sum(1 / (k + rank)) for each list the item appears in.
    """
    all_hits: Dict[str, Dict[str, Any]] = {}
    rrf_scores: Dict[str, float] = {}

    for rank, hit in enumerate(dense_hits):
        hid = hit["id"]
        rrf_scores[hid] = rrf_scores.get(hid, 0.0) + 1.0 / (k + rank + 1)
        all_hits.setdefault(hid, hit)

    for rank, hit in enumerate(keyword_hits):
        hid = hit["id"]
        rrf_scores[hid] = rrf_scores.get(hid, 0.0) + 1.0 / (k + rank + 1)
        all_hits.setdefault(hid, hit)

    sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)[:top_k]
    result = []
    for hid in sorted_ids:
        hit = all_hits[hid].copy()
        hit["rrf_score"] = rrf_scores[hid]
        result.append(hit)
    return result


# ---------------------------------------------------------------------------
# Public API  (same signatures as pinecone_client.py)
# ---------------------------------------------------------------------------

def upsert_vectors(
    doc_id: str,
    space_id: str,
    embeddings: List[Dict[str, Any]],
    chunks: List[Any],
    batch_size: int = 100,
    source_file: Optional[str] = None,
    include_images: bool = False,
    pdf_metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Upsert dense vectors to Qdrant with metadata.

    Sparse vectors are not supported in this Qdrant implementation and are
    silently ignored.
    """
    logger.info(
        f"upsert_vectors called with doc_id={doc_id}, space_id={space_id}, source_file={source_file}"
    )

    # Guard against embedding service error responses
    if embeddings and isinstance(embeddings, list) and len(embeddings) > 0:
        first_emb = embeddings[0]
        if isinstance(first_emb, dict) and "message" in first_emb and "status" in first_emb:
            error_msg = f"Embedding service returned an error: {first_emb}"
            logger.error(error_msg)
            raise ValueError(error_msg)

    for i in range(0, len(embeddings), batch_size):
        batch_emb = embeddings[i : i + batch_size]
        batch_chunks = chunks[i : i + batch_size]

        points: List[PointStruct] = []
        for j, (emb, chunk) in enumerate(zip(batch_emb, batch_chunks)):
            chunk_text = ""
            if isinstance(chunk, dict) and "text" in chunk:
                chunk_text = chunk["text"]
            elif isinstance(chunk, str):
                chunk_text = chunk

            original_id = f"{doc_id}::chunk_{i + j}"

            payload: Dict[str, Any] = {
                "_original_id": original_id,
                "document_id": doc_id,
                "space_id": space_id,
                "text": chunk_text[:3000],
                "page_number": (
                    ",".join(map(str, chunk.get("pages", [])))
                    if isinstance(chunk, dict) and isinstance(chunk.get("pages"), list)
                    else (chunk.get("pages") if isinstance(chunk, dict) else None) or "false"
                ),
            }

            if source_file:
                payload["source_file"] = source_file

            if isinstance(chunk, dict):
                if chunk.get("chapter_number"):
                    payload["chapter_number"] = str(chunk.get("chapter_number"))
                if chunk.get("chapter_title"):
                    payload["chapter_title"] = str(chunk.get("chapter_title"))[:500]
                if chunk.get("extraction_method"):
                    payload["extraction_method"] = str(chunk.get("extraction_method"))
                if chunk.get("extraction_quality"):
                    payload["extraction_quality"] = int(chunk.get("extraction_quality"))
                if chunk.get("heading_path"):
                    payload["heading_path"] = ", ".join(
                        str(h) for h in chunk.get("heading_path", [])
                    )
                if chunk.get("was_llm_corrected") is not None:
                    payload["was_llm_corrected"] = bool(chunk.get("was_llm_corrected"))
                if chunk.get("original_length"):
                    payload["original_length"] = int(chunk.get("original_length"))
                if chunk.get("corrected_length"):
                    payload["corrected_length"] = int(chunk.get("corrected_length"))
                if chunk.get("correction_model"):
                    payload["correction_model"] = str(chunk.get("correction_model"))
                if chunk.get("start_index") is not None:
                    payload["start_index"] = int(chunk.get("start_index"))
                if chunk.get("end_index") is not None:
                    payload["end_index"] = int(chunk.get("end_index"))

            if isinstance(chunk, dict) and chunk.get("metadata"):
                payload.update(chunk.get("metadata", {}))

            points.append(
                PointStruct(
                    id=_to_qdrant_id(original_id),
                    vector=emb["embedding"],
                    payload=payload,
                )
            )

        logger.info(f"Dense vectors to upsert: {len(points)}")
        if points:
            client.upsert(collection_name=COLLECTION, points=points)

    # Upsert image vectors if requested
    if include_images and pdf_metadata and pdf_metadata.get("images"):
        logger.info(f"Upserting {len(pdf_metadata['images'])} image vectors...")
        upsert_image_vectors(
            doc_id=doc_id,
            space_id=space_id,
            images=pdf_metadata["images"],
            source_file=source_file,
            batch_size=batch_size,
        )


def upsert_image_vectors(
    doc_id: str,
    space_id: str,
    images: List[Dict[str, Any]],
    embeddings: Optional[List[List[float]]] = None,
    source_file: Optional[str] = None,
    batch_size: int = 50,
) -> None:
    """Upsert image vectors to Qdrant for multimodal search."""
    if not images:
        logger.info("No images to upsert")
        return

    if embeddings is None:
        logger.warning("Skipping image upsert — embeddings required")
        return

    if len(embeddings) != len(images):
        raise ValueError(
            f"Number of embeddings ({len(embeddings)}) must match number of images ({len(images)})"
        )

    points: List[PointStruct] = []

    for idx, (img, embedding) in enumerate(zip(images, embeddings)):
        img_base64 = img.get("image_base64", "")
        image_size_kb = len(img_base64) / 1024 if img_base64 else 0
        original_id = f"{doc_id}::image_{idx}"

        payload: Dict[str, Any] = {
            "_original_id": original_id,
            "type": "image",
            "document_id": doc_id,
            "space_id": space_id,
            "image_id": img.get("id", f"img_{idx}"),
            "page": img.get("page", 0),
            "position_in_doc": img.get("position_in_doc", 0),
        }

        if source_file:
            payload["source_file"] = source_file

        if image_size_kb < 30 and img_base64:
            payload["image_base64"] = img_base64
            payload["image_storage"] = "inline"
        else:
            payload["image_storage"] = "reference"
            payload["image_size_kb"] = int(image_size_kb)

        points.append(
            PointStruct(
                id=_to_qdrant_id(original_id),
                vector=embedding,
                payload=payload,
            )
        )

        if len(points) >= batch_size:
            logger.info(f"Upserting batch of {len(points)} image vectors")
            client.upsert(collection_name=COLLECTION, points=points)
            points = []

    if points:
        logger.info(f"Upserting final batch of {len(points)} image vectors")
        client.upsert(collection_name=COLLECTION, points=points)

    logger.info(f"✅ Successfully upserted {len(images)} image vectors")


def hybrid_search_parallel(
    query_emb: List[float],
    query_text: Optional[str] = None,
    query_sparse: Optional[Dict[str, Any]] = None,
    top_k: int = 20,
    doc_id: Optional[str] = None,
    space_id: Optional[str] = None,
    acl_tags: Optional[List[str]] = None,
    include_full_text: bool = True,
) -> List[Dict[str, Any]]:
    """
    Hybrid (dense + keyword) vector search in Qdrant with optional filters.

    When query_text is provided, runs dense and keyword legs in parallel then
    merges with Reciprocal Rank Fusion. Falls back to dense-only when absent.

    Note: query_sparse is accepted for API compatibility but ignored.
    """
    if query_sparse:
        logger.debug("hybrid_search_parallel: query_sparse ignored (use query_text for keyword leg)")

    query_filter = _build_filter(doc_id, space_id, acl_tags)

    if query_text:
        with concurrent.futures.ThreadPoolExecutor() as ex:
            dense_fut = ex.submit(
                client.query_points,
                collection_name=COLLECTION,
                query=query_emb,
                limit=top_k * 2,
                query_filter=query_filter,
                with_payload=True,
            )
            keyword_fut = ex.submit(_keyword_search, query_text, query_filter, top_k * 2)

            dense_response = dense_fut.result()
            keyword_hits = keyword_fut.result()

        dense_hits = [
            {
                "id": h.payload.get("_original_id", str(h.id)),
                "score": h.score,
                "metadata": {k: v for k, v in h.payload.items() if k != "_original_id"},
            }
            for h in dense_response.points
        ]
        merged = _rrf_merge(dense_hits, keyword_hits, top_k=top_k)
        logger.info(
            f"Hybrid search: {len(dense_hits)} dense + {len(keyword_hits)} keyword "
            f"→ {len(merged)} RRF-merged hits"
        )
        return merged

    # Dense-only (backward compatible)
    response = client.query_points(
        collection_name=COLLECTION,
        query=query_emb,
        limit=top_k,
        query_filter=query_filter,
        with_payload=True,
    )

    hits = []
    for hit in response.points:
        original_id = hit.payload.get("_original_id", str(hit.id))
        hits.append({
            "id": original_id,
            "score": hit.score,
            "metadata": {k: v for k, v in hit.payload.items() if k != "_original_id"},
        })

    logger.info(f"Qdrant dense search returning {len(hits)} hits")
    return hits


def hybrid_search(
    query_emb: List[float],
    query_text: Optional[str] = None,
    query_sparse: Optional[Dict[str, Any]] = None,
    top_k: int = 10,
    doc_id: Optional[str] = None,
    space_id: Optional[str] = None,
    acl_tags: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Wrapper around hybrid_search_parallel for backward compatibility."""
    return hybrid_search_parallel(
        query_emb=query_emb,
        query_text=query_text,
        query_sparse=query_sparse,
        top_k=top_k,
        doc_id=doc_id,
        space_id=space_id,
        acl_tags=acl_tags,
        include_full_text=True,
    )


def rerank_results(
    query: str,
    hits: List[Dict[str, Any]],
    model: str = "rerank-english-v3.0",
    top_n: int = 15,
) -> List[Dict[str, Any]]:
    """
    Rerank hits using Cohere's rerank API.
    Falls back to score-based ordering if COHERE_API_KEY is not set.
    """
    hits_to_rerank = hits[:top_n]
    texts = [hit["metadata"].get("text", "") for hit in hits_to_rerank]

    scores = _rerank_with_cohere(query, texts)

    for idx, hit in enumerate(hits_to_rerank):
        hit["rerank_score"] = scores[idx]

    return sorted(hits_to_rerank, key=lambda x: x.get("rerank_score", 0), reverse=True)


def create_index(
    dimension: int,
    metric: str = "cosine",
    name: Optional[str] = None,
    sparse: bool = False,
) -> None:
    """
    No-op: collection management is handled server-side on Qdrant.
    This function exists for API compatibility.
    """
    logger.info(
        f"create_index called (dimension={dimension}, metric={metric}) — "
        "collection management is server-side on Qdrant, skipping."
    )


def delete_vectors(
    doc_id: str,
    space_id: Optional[str] = None,
) -> None:
    """Delete vectors by document ID and optionally space ID."""
    query_filter = _build_filter(doc_id=doc_id, space_id=space_id)
    if query_filter is None:
        logger.warning("delete_vectors: no filter specified, skipping to avoid accidental full delete")
        return

    try:
        logger.info(f"Deleting vectors for doc_id={doc_id}, space_id={space_id}")
        client.delete(
            collection_name=COLLECTION,
            points_selector=FilterSelector(filter=query_filter),
        )
        logger.info(f"Successfully deleted vectors for doc_id={doc_id}")
    except Exception as e:
        logger.error(f"Error deleting vectors: {str(e)}")
        raise


def hybrid_search_document_aware(
    query_emb: List[float],
    query_text: Optional[str] = None,
    query_sparse: Optional[Dict[str, Any]] = None,
    document_ids: List[str] = None,
    space_id: Optional[str] = None,
    acl_tags: Optional[List[str]] = None,
    top_k_per_doc: int = 5,
    include_full_text: bool = True,
) -> List[Dict[str, Any]]:
    """
    Document-aware hybrid search ensuring balanced representation across documents.

    When query_text is provided, runs dense + keyword legs per document and
    RRF-merges the results. Falls back to dense-only when absent.
    query_sparse is accepted for API compatibility but ignored.
    """
    if not document_ids:
        logger.warning("No document IDs provided for document-aware search")
        return []

    all_hits: Dict[str, Dict[str, Any]] = {}
    document_hit_counts: Dict[str, int] = {}

    logger.info(f"Performing document-aware search across {len(document_ids)} documents")

    def _search_one_doc(doc_id: str) -> Tuple[str, Dict[str, Dict[str, Any]]]:
        query_filter = _build_filter(doc_id=doc_id, space_id=space_id, acl_tags=acl_tags)

        if query_text:
            with concurrent.futures.ThreadPoolExecutor() as ex:
                dense_fut = ex.submit(
                    client.query_points,
                    collection_name=COLLECTION,
                    query=query_emb,
                    limit=top_k_per_doc * 2,
                    query_filter=query_filter,
                    with_payload=True,
                )
                keyword_fut = ex.submit(_keyword_search, query_text, query_filter, top_k_per_doc * 2)
                dense_response = dense_fut.result()
                keyword_hits = keyword_fut.result()

            dense_hits = [
                {
                    "id": h.payload.get("_original_id", str(h.id)),
                    "score": h.score,
                    "metadata": {
                        **{k: v for k, v in h.payload.items() if k != "_original_id"},
                        "source_document_id": doc_id,
                    },
                }
                for h in dense_response.points
            ]
            # Add source_document_id to keyword hits before merging
            for hit in keyword_hits:
                hit["metadata"]["source_document_id"] = doc_id

            merged = _rrf_merge(dense_hits, keyword_hits, top_k=top_k_per_doc)
            doc_hits: Dict[str, Dict[str, Any]] = {h["id"]: h for h in merged}
        else:
            response = client.query_points(
                collection_name=COLLECTION,
                query=query_emb,
                limit=top_k_per_doc,
                query_filter=query_filter,
                with_payload=True,
            )
            doc_hits = {}
            for hit in response.points:
                original_id = hit.payload.get("_original_id", str(hit.id))
                doc_hits[original_id] = {
                    "id": original_id,
                    "score": hit.score,
                    "metadata": {
                        **{k: v for k, v in hit.payload.items() if k != "_original_id"},
                        "source_document_id": doc_id,
                    },
                }

        return doc_id, doc_hits

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(_search_one_doc, doc_id): doc_id for doc_id in document_ids}
        for future in concurrent.futures.as_completed(futures):
            try:
                doc_id, doc_hits = future.result()
                all_hits.update(doc_hits)
                document_hit_counts[doc_id] = len(doc_hits)
                logger.info(f"Document {doc_id}: found {len(doc_hits)} chunks")
            except Exception as e:
                doc_id = futures[future]
                logger.error(f"Error searching document {doc_id}: {e}")
                document_hit_counts[doc_id] = 0

    logger.info(
        f"Document-aware search completed: {len(all_hits)} total chunks "
        f"across {len(document_ids)} documents"
    )
    return list(all_hits.values())


def rerank_results_document_aware(
    query: str,
    hits: List[Dict[str, Any]],
    model: str = "rerank-english-v3.0",
    top_n_per_doc: int = 3,
    max_total_results: int = 15,
) -> List[Dict[str, Any]]:
    """
    Rerank results while maintaining document diversity.
    Uses Cohere reranking (falls back to score-based if key not set).
    """
    if not hits:
        return []

    # Group hits by source document
    hits_by_document: Dict[str, List[Dict[str, Any]]] = {}
    for hit in hits:
        doc_id = hit.get("metadata", {}).get("source_document_id", "unknown")
        hits_by_document.setdefault(doc_id, []).append(hit)

    logger.info(f"Reranking {len(hits)} hits from {len(hits_by_document)} documents")

    reranked_by_document: Dict[str, List[Dict[str, Any]]] = {}
    for doc_id, doc_hits in hits_by_document.items():
        if not doc_hits:
            continue
        try:
            texts = [h["metadata"].get("text", "") for h in doc_hits]
            scores = _rerank_with_cohere(query, texts)
            for idx, score in enumerate(scores):
                doc_hits[idx]["rerank_score"] = score
            doc_reranked = sorted(doc_hits, key=lambda x: x.get("rerank_score", 0), reverse=True)
            reranked_by_document[doc_id] = doc_reranked[:top_n_per_doc]
            logger.info(
                f"Document {doc_id}: reranked {len(doc_hits)} -> "
                f"{len(reranked_by_document[doc_id])} chunks"
            )
        except Exception as e:
            logger.error(f"Error reranking document {doc_id}: {e}")
            reranked_by_document[doc_id] = doc_hits[:top_n_per_doc]

    # Round-robin through documents to maintain diversity
    final_results: List[Dict[str, Any]] = []
    max_rounds = max(len(v) for v in reranked_by_document.values()) if reranked_by_document else 0

    for round_idx in range(max_rounds):
        for doc_id, doc_hits in reranked_by_document.items():
            if round_idx < len(doc_hits) and len(final_results) < max_total_results:
                final_results.append(doc_hits[round_idx])

    if len(final_results) < max_total_results:
        for doc_id, doc_hits in reranked_by_document.items():
            for hit in doc_hits[top_n_per_doc:]:
                if len(final_results) < max_total_results:
                    final_results.append(hit)
                else:
                    break

    logger.info(f"Document-aware reranking completed: {len(final_results)} total results")
    return final_results


def calculate_coverage_from_chunks(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate coverage percentage from chunks using start_index and end_index."""
    ranges = []
    for chunk in chunks:
        metadata = chunk.get("metadata", {})
        start_idx = metadata.get("start_index")
        end_idx = metadata.get("end_index")
        if start_idx is not None and end_idx is not None:
            try:
                start_idx = int(start_idx)
                end_idx = int(end_idx)
                if end_idx > start_idx:
                    ranges.append((start_idx, end_idx))
            except (ValueError, TypeError):
                continue

    if not ranges:
        return {"coverage_percentage": 0.0, "gaps": [], "covered_ranges": []}

    ranges.sort()
    merged_ranges = []
    current_start, current_end = ranges[0]
    for start, end in ranges[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            merged_ranges.append((current_start, current_end))
            current_start, current_end = start, end
    merged_ranges.append((current_start, current_end))

    total_covered = sum(end - start for start, end in merged_ranges)
    total_document_chars = (
        merged_ranges[-1][1] - merged_ranges[0][0] if merged_ranges else 0
    )
    coverage_percentage = (
        total_covered / total_document_chars if total_document_chars > 0 else 0.0
    )

    gaps = []
    for i in range(len(merged_ranges) - 1):
        gap_start = merged_ranges[i][1]
        gap_end = merged_ranges[i + 1][0]
        gap_size = gap_end - gap_start
        if gap_size > 0:
            gaps.append({"start": gap_start, "end": gap_end, "size": gap_size})

    return {
        "coverage_percentage": coverage_percentage,
        "gaps": gaps,
        "covered_ranges": merged_ranges,
        "total_covered": total_covered,
        "total_document_chars": total_document_chars,
    }


def fetch_chunks_for_gap(
    document_id: str,
    gap_start: int,
    gap_end: int,
    space_id: Optional[str] = None,
    acl_tags: Optional[List[str]] = None,
    max_attempts: int = 3,
) -> List[Dict[str, Any]]:
    """Fetch chunks that fall within a specific gap in coverage."""
    range_condition = FieldCondition(
        key="start_index", range=Range(gte=float(gap_start), lt=float(gap_end))
    )
    query_filter = _build_filter(
        doc_id=document_id,
        space_id=space_id,
        acl_tags=acl_tags,
        extra_conditions=[range_condition],
    )

    try:
        from clients.chonkie_client import embed_query_multimodal

        neutral_query = "content information text data"
        query_result = embed_query_multimodal(neutral_query)
        query_emb = query_result["embedding"]

        response = client.query_points(
            collection_name=COLLECTION,
            query=query_emb,
            limit=50,
            query_filter=query_filter,
            with_payload=True,
        )

        gap_chunks = []
        for hit in response.points:
            original_id = hit.payload.get("_original_id", str(hit.id))
            gap_chunks.append({
                "id": original_id,
                "score": hit.score,
                "metadata": {k: v for k, v in hit.payload.items() if k != "_original_id"},
            })

        logger.info(f"Found {len(gap_chunks)} chunks in gap [{gap_start}, {gap_end}]")
        return gap_chunks

    except Exception as e:
        logger.warning(f"Error fetching chunks for gap [{gap_start}, {gap_end}]: {e}")
        return []


def fetch_all_document_chunks(
    document_id: str,
    space_id: Optional[str] = None,
    max_chunks: int = 2000,
    acl_tags: Optional[List[str]] = None,
    target_coverage: float = 0.80,
    enable_gap_filling: bool = True,
) -> List[Dict[str, Any]]:
    """
    Fetch ALL chunks for a document using broad query + gap-filling for 80%+ coverage.
    Returns chunks sorted by: chapter_number → page_number → start_index.
    """
    logger.info(
        f"🔍 Fetching all chunks for document {document_id}, "
        f"max_chunks={max_chunks}, target_coverage={target_coverage:.0%}"
    )

    broad_query = (
        "Extract all content, concepts, details, examples, explanations, "
        "definitions, facts, and information from this complete document"
    )

    try:
        from clients.chonkie_client import embed_query_multimodal
        query_result = embed_query_multimodal(broad_query)
        query_emb = query_result["embedding"]
    except Exception as e:
        logger.error(f"Error embedding broad query: {e}")
        from clients.chonkie_client import embed_query_v2
        query_embedded = embed_query_v2(broad_query)
        query_emb = query_embedded["embedding"]

    query_filter = _build_filter(doc_id=document_id, space_id=space_id, acl_tags=acl_tags)

    try:
        response = client.query_points(
            collection_name=COLLECTION,
            query=query_emb,
            limit=max_chunks,
            query_filter=query_filter,
            with_payload=True,
        )

        chunks = []
        chunk_ids: set = set()
        for hit in response.points:
            original_id = hit.payload.get("_original_id", str(hit.id))
            chunks.append({
                "id": original_id,
                "score": hit.score,
                "metadata": {k: v for k, v in hit.payload.items() if k != "_original_id"},
            })
            chunk_ids.add(original_id)

        logger.info(f"📥 Phase 1: Retrieved {len(chunks)} chunks from initial broad search")

        # Phase 2: Gap filling
        if enable_gap_filling and len(chunks) > 0:
            coverage_result = calculate_coverage_from_chunks(chunks)
            current_coverage = coverage_result["coverage_percentage"]
            logger.info(f"📊 Initial coverage: {current_coverage:.2%}")

            if current_coverage < target_coverage:
                gaps = coverage_result["gaps"]
                logger.info(
                    f"🔧 Coverage below target ({current_coverage:.2%} < {target_coverage:.0%}), "
                    f"filling {len(gaps)} gaps..."
                )
                gaps_sorted = sorted(gaps, key=lambda g: g["size"], reverse=True)
                gap_fill_iterations = 0
                max_gap_fill_iterations = 5

                while (
                    current_coverage < target_coverage
                    and gap_fill_iterations < max_gap_fill_iterations
                    and gaps_sorted
                ):
                    gap_fill_iterations += 1
                    for gap in gaps_sorted[:3]:
                        gap_chunks = fetch_chunks_for_gap(
                            document_id=document_id,
                            gap_start=gap["start"],
                            gap_end=gap["end"],
                            space_id=space_id,
                            acl_tags=acl_tags,
                        )
                        new_added = 0
                        for chunk in gap_chunks:
                            if chunk["id"] not in chunk_ids:
                                chunks.append(chunk)
                                chunk_ids.add(chunk["id"])
                                new_added += 1
                        logger.info(
                            f"   Gap [{gap['start']}, {gap['end']}] "
                            f"({gap['size']:,} chars): +{new_added} chunks"
                        )

                    coverage_result = calculate_coverage_from_chunks(chunks)
                    new_coverage = coverage_result["coverage_percentage"]
                    logger.info(
                        f"   Iteration {gap_fill_iterations}: "
                        f"Coverage improved from {current_coverage:.2%} → {new_coverage:.2%}"
                    )

                    if new_coverage <= current_coverage:
                        logger.info("   Coverage not improving, stopping gap-filling")
                        break
                    current_coverage = new_coverage
                    gaps_sorted = sorted(
                        coverage_result["gaps"], key=lambda g: g["size"], reverse=True
                    )

                logger.info(f"✅ Gap-filling complete: Final coverage {current_coverage:.2%}")

        def get_sort_key(chunk: Dict[str, Any]) -> tuple:
            metadata = chunk.get("metadata", {})

            chapter_num = metadata.get("chapter_number", 0)
            if isinstance(chapter_num, str):
                try:
                    chapter_num = int(chapter_num)
                except (ValueError, TypeError):
                    chapter_num = 0

            page_num = metadata.get("page_number", "0")
            if isinstance(page_num, str):
                page_num = page_num.split(",")[0] if page_num else "0"
                try:
                    page_num = int(page_num)
                except (ValueError, TypeError):
                    page_num = 0

            start_idx = metadata.get("start_index", 0)
            if isinstance(start_idx, str):
                try:
                    start_idx = int(start_idx)
                except (ValueError, TypeError):
                    start_idx = 0

            chunk_id = chunk.get("id", "")
            position = 0
            if "::" in chunk_id and "_" in chunk_id:
                try:
                    position = int(chunk_id.split("_")[-1])
                except (ValueError, IndexError):
                    position = 0

            return (chapter_num, page_num, start_idx, position)

        sorted_chunks = sorted(chunks, key=get_sort_key)
        logger.info(f"✅ Fetched and sorted {len(sorted_chunks)} chunks for note generation")
        return sorted_chunks

    except Exception as e:
        logger.error(f"❌ Error fetching chunks: {e}")
        return []


def fetch_document_vector_ids(
    document_id: str,
    max_ids: int = 5000,
) -> List[str]:
    """
    Fetch all Qdrant point IDs for a given document_id using scroll pagination.

    Returns Qdrant UUID strings (suitable for use with update_vector_metadata).
    """
    query_filter = _build_filter(doc_id=document_id)
    all_ids: List[str] = []
    offset = None

    while len(all_ids) < max_ids:
        batch_limit = min(100, max_ids - len(all_ids))
        results, offset = client.scroll(
            collection_name=COLLECTION,
            scroll_filter=query_filter,
            limit=batch_limit,
            offset=offset,
            with_payload=False,
            with_vectors=False,
        )
        all_ids.extend(str(p.id) for p in results)
        if offset is None:
            break

    logger.info(f"fetch_document_vector_ids: found {len(all_ids)} vectors for document {document_id}")
    return all_ids[:max_ids]


def update_vector_metadata(
    vector_ids: List[str],
    metadata_update: Dict[str, Any],
    max_workers: int = 10,
) -> int:
    """
    Update payload on existing Qdrant vectors (best-effort, batched).

    Args:
        vector_ids: List of Qdrant UUID strings (from fetch_document_vector_ids).
        metadata_update: Dict of payload fields to set/overwrite.
        max_workers: Unused (kept for API compatibility — Qdrant batch is single call).

    Returns:
        Count of successfully updated vectors.
    """
    if not vector_ids:
        return 0

    try:
        client.set_payload(
            collection_name=COLLECTION,
            payload=metadata_update,
            points=vector_ids,
        )
        logger.info(f"Updated metadata on {len(vector_ids)} vectors")
        return len(vector_ids)
    except Exception as e:
        logger.error(f"Failed to update metadata: {e}")
        return 0


def fetch_chunks_by_space(
    space_id: str,
    document_ids: List[str],
    max_chunks_per_doc: int = 2000,
    target_coverage: float = 0.80,
    enable_gap_filling: bool = True,
) -> List[Dict[str, Any]]:
    """
    Fetch chunks for multiple documents in a space.
    Calls fetch_all_document_chunks() per document and merges results.
    """
    all_chunks: List[Dict[str, Any]] = []
    chunk_ids_seen: set = set()

    for doc_id in document_ids:
        logger.info(f"Fetching chunks for document {doc_id} in space {space_id}")
        doc_chunks = fetch_all_document_chunks(
            document_id=doc_id,
            space_id=space_id,
            max_chunks=max_chunks_per_doc,
            target_coverage=target_coverage,
            enable_gap_filling=enable_gap_filling,
        )
        for chunk in doc_chunks:
            cid = chunk.get("id")
            if cid not in chunk_ids_seen:
                chunk_ids_seen.add(cid)
                all_chunks.append(chunk)

    logger.info(
        f"Fetched {len(all_chunks)} total chunks across "
        f"{len(document_ids)} documents in space {space_id}"
    )
    return all_chunks
