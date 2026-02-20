import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import logging
from pinecone import Pinecone
import concurrent.futures
from functools import lru_cache

# Load environment variables
load_dotenv()

# Environment variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "nuton-index")

# Index names - MULTIMODAL (1024 dims, Jina CLIP-v2)
DENSE_INDEX = "nuton-index-multi-modal"  # NEW: 1024-dim multimodal index
SPARSE_INDEX = f"{PINECONE_INDEX_NAME}-sparse"  # Keep existing sparse index

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Pinecone client using new style
try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    logger.info("Initialized Pinecone client")
except Exception as e:
    logger.error(f"Failed to initialize Pinecone client: {str(e)}")
    raise

def upsert_vectors(
    doc_id: str,
    space_id: str,
    embeddings: List[Dict[str, Any]],
    chunks: List[Any],
    batch_size: int = 100,
    source_file: Optional[str] = None,
    include_images: bool = False,  # NEW: Whether chunks include image data
    pdf_metadata: Optional[Dict[str, Any]] = None  # NEW: PDF metadata with images
) -> None:
    """
    Upsert dense (and optionally sparse) vectors to Pinecone, with metadata.

    Now supports both text and image chunks for multimodal search.

    Args:
        doc_id: Document id to include in metadata.
        space_id: Space id to include in metadata.
        embeddings: List of dicts with 'embedding' (dense) and optionally 'sparse' fields.
        chunks: List of chunk dicts (must align with embeddings).
        batch_size: Max vectors per upsert call.
        source_file: Original document filename to include in metadata.
        include_images: Whether to also upsert image vectors from pdf_metadata.
        pdf_metadata: Full PDF metadata including images (if include_images=True).
    """
    logger.info(f"upsert_vectors called with doc_id={doc_id}, space_id={space_id}, source_file={source_file}")
    
    # Check if embeddings contains error response
    if embeddings and isinstance(embeddings, list) and len(embeddings) > 0:
        first_emb = embeddings[0]
        if isinstance(first_emb, dict) and 'message' in first_emb and 'status' in first_emb:
            error_msg = f"Embedding service returned an error: {first_emb}"
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    # logger.info(f"Type of chunks: {type(chunks)}; Example: {chunks[:1]}")
    # logger.info(f"Type of embeddings: {type(embeddings)}; Example: {embeddings[:1]}")
    
    # Initialize indexes with new SDK format
    dense_index = pc.Index(DENSE_INDEX)
    sparse_index = pc.Index(SPARSE_INDEX)
    
    for i in range(0, len(embeddings), batch_size):
        batch_emb = embeddings[i:i+batch_size]
        batch_chunks = chunks[i:i+batch_size]
        
        # Prepare dense vectors in new format
        dense_vectors = []
        for j, (emb, chunk) in enumerate(zip(batch_emb, batch_chunks)):
            # Get the chunk text content
            chunk_text = ""
            if isinstance(chunk, dict) and "text" in chunk:
                chunk_text = chunk["text"]
            elif isinstance(chunk, str):
                chunk_text = chunk
                
            # Create metadata with document_id, space_id, text, and source_file
            metadata = {
                "document_id": doc_id,
                "space_id": space_id,
                "text": chunk_text[:3000],  # Limit text length for Pinecone metadata size limits
                "page_number": ','.join(map(str, chunk.get("pages", []))) if isinstance(chunk.get("pages"), list) else (chunk.get("pages") or "false")
            }

            # Add source_file if provided
            if source_file:
                metadata["source_file"] = source_file

            # Add chapter information if available
            if isinstance(chunk, dict):
                if chunk.get("chapter_number"):
                    metadata["chapter_number"] = str(chunk.get("chapter_number"))
                if chunk.get("chapter_title"):
                    metadata["chapter_title"] = str(chunk.get("chapter_title"))[:500]  # Limit length

                # Add enhanced metadata from hybrid PDF processor
                if chunk.get("extraction_method"):
                    metadata["extraction_method"] = str(chunk.get("extraction_method"))
                if chunk.get("extraction_quality"):
                    metadata["extraction_quality"] = int(chunk.get("extraction_quality"))
                if chunk.get("heading_path"):
                    # Store as comma-separated string for Pinecone compatibility
                    metadata["heading_path"] = ", ".join(str(h) for h in chunk.get("heading_path", []))

                # Add LLM correction metadata (from parallel quality correction)
                if chunk.get("was_llm_corrected") is not None:
                    metadata["was_llm_corrected"] = bool(chunk.get("was_llm_corrected"))
                if chunk.get("original_length"):
                    metadata["original_length"] = int(chunk.get("original_length"))
                if chunk.get("corrected_length"):
                    metadata["corrected_length"] = int(chunk.get("corrected_length"))
                if chunk.get("correction_model"):
                    metadata["correction_model"] = str(chunk.get("correction_model"))

                # Add chunk position indices for better coverage tracking
                if chunk.get("start_index") is not None:
                    metadata["start_index"] = int(chunk.get("start_index"))
                if chunk.get("end_index") is not None:
                    metadata["end_index"] = int(chunk.get("end_index"))

            # Add any existing metadata from the chunk
            if isinstance(chunk, dict) and chunk.get("metadata"):
                metadata.update(chunk.get("metadata", {}))
                
            # Create the vector record for new SDK format
            vector = {
                "id": f"{doc_id}::chunk_{i+j}",
                "values": emb["embedding"],
                "metadata": metadata
            }
            dense_vectors.append(vector)
            
        logger.info(f"Dense vectors to upsert: {len(dense_vectors)}")
        if dense_vectors:
            dense_index.upsert(vectors=dense_vectors)
        
        # Prepare sparse vectors if present
        if batch_emb and "sparse" in batch_emb[0]:
            sparse_vectors = []
            for j, (emb, chunk) in enumerate(zip(batch_emb, batch_chunks)):
                # Get the chunk text content
                chunk_text = ""
                if isinstance(chunk, dict) and "text" in chunk:
                    chunk_text = chunk["text"]
                elif isinstance(chunk, str):
                    chunk_text = chunk
                    
                # Create metadata with document_id, space_id, text, and source_file
                metadata = {
                    "document_id": doc_id,
                    "space_id": space_id,
                    "text": chunk_text[:3000],  # Limit text length for Pinecone metadata size limits
                    "page_number": ','.join(map(str, chunk.get("pages", []))) if isinstance(chunk.get("pages"), list) else (chunk.get("pages") or "false")
                }

                # Add source_file if provided
                if source_file:
                    metadata["source_file"] = source_file

                # Add chapter information if available
                if isinstance(chunk, dict):
                    if chunk.get("chapter_number"):
                        metadata["chapter_number"] = str(chunk.get("chapter_number"))
                    if chunk.get("chapter_title"):
                        metadata["chapter_title"] = str(chunk.get("chapter_title"))[:500]  # Limit length

                    # Add enhanced metadata from hybrid PDF processor
                    if chunk.get("extraction_method"):
                        metadata["extraction_method"] = str(chunk.get("extraction_method"))
                    if chunk.get("extraction_quality"):
                        metadata["extraction_quality"] = int(chunk.get("extraction_quality"))
                    if chunk.get("heading_path"):
                        # Store as comma-separated string for Pinecone compatibility
                        metadata["heading_path"] = ", ".join(str(h) for h in chunk.get("heading_path", []))

                    # Add LLM correction metadata (from parallel quality correction)
                    if chunk.get("was_llm_corrected") is not None:
                        metadata["was_llm_corrected"] = bool(chunk.get("was_llm_corrected"))
                    if chunk.get("original_length"):
                        metadata["original_length"] = int(chunk.get("original_length"))
                    if chunk.get("corrected_length"):
                        metadata["corrected_length"] = int(chunk.get("corrected_length"))
                    if chunk.get("correction_model"):
                        metadata["correction_model"] = str(chunk.get("correction_model"))

                    # Add chunk position indices for better coverage tracking
                    if chunk.get("start_index") is not None:
                        metadata["start_index"] = int(chunk.get("start_index"))
                    if chunk.get("end_index") is not None:
                        metadata["end_index"] = int(chunk.get("end_index"))

                # Add any existing metadata from the chunk
                if isinstance(chunk, dict) and chunk.get("metadata"):
                    metadata.update(chunk.get("metadata", {}))
                
                # Create the vector record for new SDK format
                vector = {
                    "id": f"{doc_id}::chunk_{i+j}",
                    "sparse_values": emb["sparse"],
                    "metadata": metadata
                }
                sparse_vectors.append(vector)
                
            logger.info(f"Sparse vectors to upsert: {len(sparse_vectors)}")
            if sparse_vectors:
                sparse_index.upsert(vectors=sparse_vectors)

    # NEW: Upsert image vectors if requested
    if include_images and pdf_metadata and pdf_metadata.get('images'):
        logger.info(f"Upserting {len(pdf_metadata['images'])} image vectors...")
        upsert_image_vectors(
            doc_id=doc_id,
            space_id=space_id,
            images=pdf_metadata['images'],
            source_file=source_file,
            batch_size=batch_size
        )


def upsert_image_vectors(
    doc_id: str,
    space_id: str,
    images: List[Dict[str, Any]],
    embeddings: Optional[List[List[float]]] = None,
    source_file: Optional[str] = None,
    batch_size: int = 50
) -> None:
    """
    Upsert image vectors to Pinecone for multimodal search.

    Args:
        doc_id: Document ID
        space_id: Space ID
        images: List of image metadata dicts from extraction
        embeddings: Pre-computed image embeddings (optional, will compute if not provided)
        source_file: Source filename
        batch_size: Batch size for upserts
    """
    if not images:
        logger.info("No images to upsert")
        return

    # Initialize indexes
    dense_index = pc.Index(DENSE_INDEX)

    # If embeddings not provided, we need to embed images
    if embeddings is None:
        logger.info("Image embeddings not provided - images will need to be embedded separately")
        # For now, skip embedding here - it should be done before calling this function
        logger.warning("Skipping image upsert - embeddings required")
        return

    if len(embeddings) != len(images):
        raise ValueError(f"Number of embeddings ({len(embeddings)}) must match number of images ({len(images)})")

    # Upsert in batches
    image_vectors = []

    for idx, (img, embedding) in enumerate(zip(images, embeddings)):
        # Determine image storage strategy (hybrid: inline <30KB, reference >30KB)
        img_base64 = img.get('image_base64', '')
        image_size_kb = len(img_base64) / 1024 if img_base64 else 0

        # Build metadata for image
        metadata = {
            "type": "image",
            "document_id": doc_id,
            "space_id": space_id,
            "image_id": img.get('id', f"img_{idx}"),
            "page": img.get('page', 0),
            "position_in_doc": img.get('position_in_doc', 0),
        }

        # Add source file
        if source_file:
            metadata["source_file"] = source_file

        # Hybrid storage: small images inline, large images as reference
        if image_size_kb < 30 and img_base64:
            # Store inline (small image)
            metadata["image_base64"] = img_base64[:40000]  # Pinecone limit
            metadata["image_storage"] = "inline"
        else:
            # Store reference only (large image)
            metadata["image_storage"] = "reference"
            metadata["image_size_kb"] = int(image_size_kb)

        # Create vector
        vector = {
            "id": f"{doc_id}::image_{idx}",
            "values": embedding,
            "metadata": metadata
        }

        image_vectors.append(vector)

        # Upsert batch when full
        if len(image_vectors) >= batch_size:
            logger.info(f"Upserting batch of {len(image_vectors)} image vectors")
            dense_index.upsert(vectors=image_vectors)
            image_vectors = []

    # Upsert remaining
    if image_vectors:
        logger.info(f"Upserting final batch of {len(image_vectors)} image vectors")
        dense_index.upsert(vectors=image_vectors)

    logger.info(f"✅ Successfully upserted {len(images)} image vectors")


@lru_cache(maxsize=100)
def get_filter_dict(doc_id: Optional[str] = None, space_id: Optional[str] = None, acl_tags: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Create and cache filter dictionaries to avoid recreation for frequent query patterns.
    """
    filter_dict = {}
    if doc_id:
        filter_dict["document_id"] = {"$eq": doc_id}
    if space_id:
        filter_dict["space_id"] = {"$eq": space_id}
    if acl_tags:
        filter_dict["acl_tags"] = {"$in": acl_tags}
    return filter_dict

def hybrid_search_parallel(
    query_emb: List[float],
    query_sparse: Optional[Dict[str, Any]] = None,
    top_k: int = 20,  # Increase default to ensure enough for reranking
    doc_id: Optional[str] = None,
    space_id: Optional[str] = None,
    acl_tags: Optional[List[str]] = None,
    include_full_text: bool = True
) -> List[Dict[str, Any]]:
    """
    Perform parallel hybrid search in Pinecone (dense + sparse), filter by doc_id and ACL tags.
    Uses concurrent execution to reduce latency.
    
    Args:
        query_emb: Dense query embedding.
        query_sparse: Sparse query representation (BM25/ColBERT), or None.
        top_k: Number of results to retrieve from each index.
        doc_id: Filter by document id.
        space_id: Filter results to this space ID.
        acl_tags: List of ACL tags to filter by.
        include_full_text: Whether to include full text in metadata or just summaries.
        
    Returns:
        List of merged, deduped hits (dicts with id, score, metadata).
    """
    dense_index = pc.Index(DENSE_INDEX)
    sparse_index = pc.Index(SPARSE_INDEX)
    
    # Build and cache filter
    filter_dict = get_filter_dict(doc_id, space_id, acl_tags)
    
    # Determine which metadata fields to include
    metadata_fields = ["document_id", "space_id", "source_file"]
    if include_full_text:
        metadata_fields.append("text")
    
    dense_results = None
    sparse_results = None
    
    # Execute searches in parallel using ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submit dense search task
        dense_future = executor.submit(
            dense_index.query,
            vector=query_emb,
            top_k=top_k,
            filter=filter_dict,
            include_metadata=True
        )
        
        # Submit sparse search task if sparse embeddings provided
        sparse_future = None
        if query_sparse:
            sparse_future = executor.submit(
                sparse_index.query,
                vector=query_sparse,
                top_k=top_k,
                filter=filter_dict,
                include_metadata=True
            )
        
        # Retrieve results
        dense_results = dense_future.result()
        if sparse_future:
            sparse_results = sparse_future.result()
    
    # Merge and dedupe by id, keep highest score
    all_hits = {}
    
    # Add dense matches to results
    for match in dense_results.matches:
        all_hits[match.id] = {
            "id": match.id,
            "score": match.score,
            "metadata": match.metadata
        }
    
    # Add sparse matches if available
    if sparse_results:
        for match in sparse_results.matches:
            match_id = match.id
            if match_id not in all_hits or match.score > all_hits[match_id]["score"]:
                all_hits[match_id] = {
                    "id": match_id,
                    "score": match.score,
                    "metadata": match.metadata
                }
    
    logger.info(f"Parallel search returning {len(all_hits)} total deduplicated hits")
    return list(all_hits.values())

# Keep the original hybrid_search function for backward compatibility
def hybrid_search(
    query_emb: List[float],
    query_sparse: Optional[Dict[str, Any]] = None,
    top_k: int = 10,
    doc_id: Optional[str] = None,
    space_id: Optional[str] = None,
    acl_tags: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    Original sequential hybrid search implementation.
    Now a wrapper around hybrid_search_parallel for backward compatibility.
    """
    return hybrid_search_parallel(
        query_emb=query_emb,
        query_sparse=query_sparse,
        top_k=top_k,
        doc_id=doc_id,
        space_id=space_id,
        acl_tags=acl_tags,
        include_full_text=True
    )

def rerank_results(
    query: str,
    hits: List[Dict[str, Any]],
    model: str = "bge-reranker-v2-m3",
    top_n: int = 15
) -> List[Dict[str, Any]]:
    """
    Rerank a list of hits using Pinecone's hosted reranker model.
    Args:
        query: The user query string.
        hits: List of hit dicts (must include 'id' and 'metadata' with 'text').
        model: Reranker model name.
        top_n: Number of hits to rerank (cost/latency control).
    Returns:
        List of reranked hits (dicts with id, score, metadata).
    """
    # from pinecone.inference import rerank
    
    # Limit to top_n hits
    hits_to_rerank = hits[:top_n]
    texts = [hit["metadata"]["text"] for hit in hits_to_rerank]
    
    # Call rerank
    rerank_results = pc.inference.rerank(
        model=model,
        query=query,
        documents=texts
    )

    # print('rerank_results.data', rerank_results.data)
    
    # Attach rerank score to hits using rerank_results.data
    for rerank_result in rerank_results.data:
        idx = rerank_result['index']
        score = rerank_result['score']
        hits_to_rerank[idx]["rerank_score"] = score
    
    # Sort by rerank_score descending
    reranked = sorted(hits_to_rerank, key=lambda x: x["rerank_score"], reverse=True)
    return reranked

def create_index(
    dimension: int, 
    metric: str = "cosine", 
    name: Optional[str] = None, 
    sparse: bool = False
) -> None:
    """
    Create Pinecone index if it doesn't exist.
    Args:
        dimension: Vector dimension.
        metric: Distance metric ('cosine', 'euclidean', or 'dotproduct').
        name: Index name (defaults to PINECONE_INDEX_NAME env var).
        sparse: Whether to create a sparse index.
    """
    from pinecone import ServerlessSpec
    
    index_name = name or (SPARSE_INDEX if sparse else DENSE_INDEX)
    
    # Check if index already exists
    if index_name in [idx.name for idx in pc.list_indexes()]:
        logger.info(f"Index {index_name} already exists")
        return
    
    # Create the index
    logger.info(f"Creating index {index_name} with dimension {dimension}")
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric=metric,
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
    logger.info(f"Successfully created index {index_name}")

def delete_vectors(
    doc_id: str,
    space_id: Optional[str] = None
) -> None:
    """
    Delete vectors by document ID and optionally space ID.
    Args:
        doc_id: Document ID to delete.
        space_id: Optional space ID to filter by.
    """
    dense_index = pc.Index(DENSE_INDEX)
    sparse_index = pc.Index(SPARSE_INDEX)
    
    # Build filter
    filter_dict = {"document_id": {"$eq": doc_id}}
    if space_id:
        filter_dict["space_id"] = {"$eq": space_id}
    
    try:
        # Delete from dense index
        logger.info(f"Deleting vectors from {DENSE_INDEX} with filter: {filter_dict}")
        dense_index.delete(filter=filter_dict)
        
        # Delete from sparse index
        logger.info(f"Deleting vectors from {SPARSE_INDEX} with filter: {filter_dict}")
        sparse_index.delete(filter=filter_dict)
        
        logger.info(f"Successfully deleted vectors for doc_id={doc_id}")
    except Exception as e:
        logger.error(f"Error deleting vectors: {str(e)}")
        raise

def hybrid_search_document_aware(
    query_emb: List[float],
    query_sparse: Optional[Dict[str, Any]] = None,
    document_ids: List[str] = None,
    space_id: Optional[str] = None,
    acl_tags: Optional[List[str]] = None,
    top_k_per_doc: int = 5,  # Results per document
    include_full_text: bool = True
) -> List[Dict[str, Any]]:
    """
    Perform document-aware hybrid search that ensures balanced representation across documents.
    Instead of global search that may favor dominant documents, this searches each document
    individually and combines results with document metadata.
    
    Args:
        query_emb: Dense query embedding.
        query_sparse: Sparse query representation (BM25/ColBERT), or None.
        document_ids: List of document IDs to search within.
        space_id: Filter results to this space ID.
        acl_tags: List of ACL tags to filter by.
        top_k_per_doc: Number of results to retrieve per document.
        include_full_text: Whether to include full text in metadata.
        
    Returns:
        List of hits with document source information, balanced across documents.
    """
    if not document_ids:
        logging.warning("No document IDs provided for document-aware search")
        return []
    
    dense_index = pc.Index(DENSE_INDEX)
    sparse_index = pc.Index(SPARSE_INDEX)
    
    all_hits = {}  # Use dict to avoid duplicates
    document_hit_counts = {}  # Track hits per document for logging
    
    logging.info(f"Performing document-aware search across {len(document_ids)} documents")
    
    # Search each document individually
    for doc_id in document_ids:
        try:
            # Build filter for this specific document
            filter_dict = get_filter_dict(doc_id, space_id, acl_tags)
            
            # Execute searches for this document
            dense_results = None
            sparse_results = None
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                # Submit dense search task
                dense_future = executor.submit(
                    dense_index.query,
                    vector=query_emb,
                    top_k=top_k_per_doc,
                    filter=filter_dict,
                    include_metadata=True
                )
                
                # Submit sparse search task if sparse embeddings provided
                sparse_future = None
                if query_sparse:
                    sparse_future = executor.submit(
                        sparse_index.query,
                        vector=query_sparse,
                        top_k=top_k_per_doc,
                        filter=filter_dict,
                        include_metadata=True
                    )
                
                # Retrieve results
                dense_results = dense_future.result()
                if sparse_future:
                    sparse_results = sparse_future.result()
            
            # Process results for this document
            doc_hits = {}
            
            # Add dense matches
            for match in dense_results.matches:
                doc_hits[match.id] = {
                    "id": match.id,
                    "score": match.score,
                    "metadata": {**match.metadata, "source_document_id": doc_id}
                }
            
            # Add sparse matches if available
            if sparse_results:
                for match in sparse_results.matches:
                    match_id = match.id
                    if match_id not in doc_hits or match.score > doc_hits[match_id]["score"]:
                        doc_hits[match_id] = {
                            "id": match_id,
                            "score": match.score,
                            "metadata": {**match.metadata, "source_document_id": doc_id}
                        }
            
            # Add document hits to global results
            all_hits.update(doc_hits)
            document_hit_counts[doc_id] = len(doc_hits)
            
            logging.info(f"Document {doc_id}: found {len(doc_hits)} chunks")
            
        except Exception as e:
            logging.error(f"Error searching document {doc_id}: {e}")
            document_hit_counts[doc_id] = 0
    
    # Log summary
    total_hits = len(all_hits)
    logging.info(f"Document-aware search completed: {total_hits} total chunks across {len(document_ids)} documents")
    for doc_id, count in document_hit_counts.items():
        logging.info(f"  Document {doc_id}: {count} chunks")
    
    return list(all_hits.values())

def rerank_results_document_aware(
    query: str,
    hits: List[Dict[str, Any]],
    model: str = "bge-reranker-v2-m3",
    top_n_per_doc: int = 3,  # Top results per document after reranking
    max_total_results: int = 15  # Maximum total results to return
) -> List[Dict[str, Any]]:
    """
    Rerank results while maintaining document diversity.
    This ensures that the final results include representation from multiple documents
    rather than being dominated by a single document.
    
    Args:
        query: The user query string.
        hits: List of hit dicts with 'source_document_id' in metadata.
        model: Reranker model name.
        top_n_per_doc: Maximum results per document after reranking.
        max_total_results: Maximum total results to return.
        
    Returns:
        List of reranked hits with balanced document representation.
    """
    if not hits:
        return []
    
    # Group hits by source document
    hits_by_document = {}
    for hit in hits:
        doc_id = hit.get("metadata", {}).get("source_document_id", "unknown")
        if doc_id not in hits_by_document:
            hits_by_document[doc_id] = []
        hits_by_document[doc_id].append(hit)
    
    logging.info(f"Reranking {len(hits)} hits from {len(hits_by_document)} documents")
    
    # Rerank hits within each document group
    reranked_by_document = {}
    for doc_id, doc_hits in hits_by_document.items():
        if not doc_hits:
            continue
            
        try:
            # Rerank this document's hits
            texts = [hit["metadata"]["text"] for hit in doc_hits]
            
            rerank_results = pc.inference.rerank(
                model=model,
                query=query,
                documents=texts
            )
            
            # Attach rerank scores
            for rerank_result in rerank_results.data:
                idx = rerank_result['index']
                score = rerank_result['score']
                doc_hits[idx]["rerank_score"] = score
            
            # Sort by rerank score and take top_n_per_doc
            doc_reranked = sorted(doc_hits, key=lambda x: x.get("rerank_score", 0), reverse=True)
            reranked_by_document[doc_id] = doc_reranked[:top_n_per_doc]
            
            logging.info(f"Document {doc_id}: reranked {len(doc_hits)} -> {len(reranked_by_document[doc_id])} chunks")
            
        except Exception as e:
            logging.error(f"Error reranking document {doc_id}: {e}")
            # Fallback: take original hits without reranking
            reranked_by_document[doc_id] = doc_hits[:top_n_per_doc]
    
    # Combine results from all documents, interleaving to maintain diversity
    final_results = []
    max_rounds = max(len(hits) for hits in reranked_by_document.values()) if reranked_by_document else 0
    
    # Round-robin through documents to maintain diversity
    for round_idx in range(max_rounds):
        for doc_id, doc_hits in reranked_by_document.items():
            if round_idx < len(doc_hits) and len(final_results) < max_total_results:
                final_results.append(doc_hits[round_idx])
    
    # If we still have space and some documents have more results, add them
    if len(final_results) < max_total_results:
        for doc_id, doc_hits in reranked_by_document.items():
            for hit in doc_hits[top_n_per_doc:]:  # Remaining hits beyond top_n_per_doc
                if len(final_results) < max_total_results:
                    final_results.append(hit)
                else:
                    break
            if len(final_results) >= max_total_results:
                break
    
    logging.info(f"Document-aware reranking completed: {len(final_results)} total results")
    
    # Log final distribution
    final_distribution = {}
    for hit in final_results:
        doc_id = hit.get("metadata", {}).get("source_document_id", "unknown")
        final_distribution[doc_id] = final_distribution.get(doc_id, 0) + 1
    
    for doc_id, count in final_distribution.items():
        logging.info(f"  Final results from document {doc_id}: {count} chunks")
    
    return final_results


def calculate_coverage_from_chunks(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate coverage percentage from chunks using start_index and end_index.
    Helper function for gap-filling logic.
    
    Args:
        chunks: List of chunk dictionaries with metadata
        
    Returns:
        Dictionary with coverage metrics
    """
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
        return {
            "coverage_percentage": 0.0,
            "gaps": [],
            "covered_ranges": []
        }
    
    ranges.sort()
    
    # Merge overlapping ranges
    merged_ranges = []
    current_start, current_end = ranges[0]
    
    for start, end in ranges[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            merged_ranges.append((current_start, current_end))
            current_start, current_end = start, end
    
    merged_ranges.append((current_start, current_end))
    
    # Calculate coverage
    total_covered = sum(end - start for start, end in merged_ranges)
    total_document_chars = merged_ranges[-1][1] - merged_ranges[0][0] if merged_ranges else 0
    coverage_percentage = total_covered / total_document_chars if total_document_chars > 0 else 0.0
    
    # Find gaps
    gaps = []
    for i in range(len(merged_ranges) - 1):
        gap_start = merged_ranges[i][1]
        gap_end = merged_ranges[i + 1][0]
        gap_size = gap_end - gap_start
        if gap_size > 0:
            gaps.append({
                "start": gap_start,
                "end": gap_end,
                "size": gap_size
            })
    
    return {
        "coverage_percentage": coverage_percentage,
        "gaps": gaps,
        "covered_ranges": merged_ranges,
        "total_covered": total_covered,
        "total_document_chars": total_document_chars
    }


def fetch_chunks_for_gap(
    document_id: str,
    gap_start: int,
    gap_end: int,
    space_id: Optional[str] = None,
    acl_tags: Optional[List[str]] = None,
    max_attempts: int = 3
) -> List[Dict[str, Any]]:
    """
    Fetch chunks that fall within a specific gap in coverage.
    
    Args:
        document_id: Document ID
        gap_start: Start index of the gap
        gap_end: End index of the gap
        space_id: Optional space ID filter
        acl_tags: Optional ACL tags
        max_attempts: Number of query variations to try
        
    Returns:
        List of chunks that fall within the gap
    """
    dense_index = pc.Index(DENSE_INDEX)
    
    # Build base filter
    filter_dict = {"document_id": {"$eq": document_id}}
    if space_id:
        filter_dict["space_id"] = {"$eq": space_id}
    if acl_tags:
        filter_dict["acl_tags"] = {"$in": acl_tags}
    
    # Try to fetch by filtering on start_index range
    # Note: Pinecone supports numeric range filters
    filter_dict["start_index"] = {"$gte": gap_start, "$lt": gap_end}
    
    try:
        # Use a neutral query for gap-filling (we're relying on the filter)
        from clients.chonkie_client import embed_query_multimodal
        neutral_query = "content information text data"
        query_result = embed_query_multimodal(neutral_query)
        query_emb = query_result["embedding"]
        
        results = dense_index.query(
            vector=query_emb,
            top_k=50,  # Get multiple chunks from gap
            filter=filter_dict,
            include_metadata=True
        )
        
        gap_chunks = []
        for match in results.matches:
            gap_chunks.append({
                "id": match.id,
                "score": match.score,
                "metadata": match.metadata
            })
        
        logging.info(f"Found {len(gap_chunks)} chunks in gap [{gap_start}, {gap_end}]")
        return gap_chunks
        
    except Exception as e:
        logging.warning(f"Error fetching chunks for gap [{gap_start}, {gap_end}]: {e}")
        return []


def fetch_all_document_chunks(
    document_id: str,
    space_id: Optional[str] = None,
    max_chunks: int = 2000,
    acl_tags: Optional[List[str]] = None,
    target_coverage: float = 0.80,
    enable_gap_filling: bool = True
) -> List[Dict[str, Any]]:
    """
    Fetch ALL chunks for a document using broad query + gap-filling for 80%+ coverage.
    Returns chunks sorted by: chapter_number → page_number → start_index.

    This function is designed for note generation where we need complete document coverage.
    It uses an iterative approach:
    1. Initial broad semantic search
    2. Calculate coverage using start_index/end_index
    3. Fill gaps with targeted queries until target_coverage is reached

    Args:
        document_id: Document ID to fetch chunks for
        space_id: Optional space ID filter
        max_chunks: Maximum number of chunks to retrieve (default: 2000)
        acl_tags: Optional ACL tags to filter by
        target_coverage: Target coverage percentage (default: 0.80 = 80%)
        enable_gap_filling: Whether to enable gap-filling (default: True)

    Returns:
        List of chunk dictionaries sorted by document position
    """
    logging.info(f"🔍 Fetching all chunks for document {document_id}, max_chunks={max_chunks}, target_coverage={target_coverage:.0%}")

    # Phase 1: Broad semantic search
    broad_query = "Extract all content, concepts, details, examples, explanations, definitions, facts, and information from this complete document"

    # Embed the query using multimodal embeddings (Jina CLIP-v2, 1024 dims)
    try:
        from clients.chonkie_client import embed_query_multimodal
        query_result = embed_query_multimodal(broad_query)
        query_emb = query_result["embedding"]
    except Exception as e:
        logging.error(f"Error embedding broad query: {e}")
        # Fallback: try without multimodal
        from clients.chonkie_client import embed_query_v2
        query_embedded = embed_query_v2(broad_query)
        query_emb = query_embedded["embedding"]

    # Search with high top_k to get all chunks
    dense_index = pc.Index(DENSE_INDEX)

    # Build filter
    filter_dict = {"document_id": {"$eq": document_id}}
    if space_id:
        filter_dict["space_id"] = {"$eq": space_id}
    if acl_tags:
        filter_dict["acl_tags"] = {"$in": acl_tags}

    try:
        # Query Pinecone with high top_k
        results = dense_index.query(
            vector=query_emb,
            top_k=max_chunks,
            filter=filter_dict,
            include_metadata=True
        )

        # Convert matches to list of dicts
        chunks = []
        chunk_ids = set()  # Track IDs to avoid duplicates
        
        for match in results.matches:
            chunks.append({
                "id": match.id,
                "score": match.score,
                "metadata": match.metadata
            })
            chunk_ids.add(match.id)

        logging.info(f"📥 Phase 1: Retrieved {len(chunks)} chunks from initial broad search")

        # Phase 2: Calculate coverage and fill gaps if needed
        if enable_gap_filling and len(chunks) > 0:
            coverage_result = calculate_coverage_from_chunks(chunks)
            current_coverage = coverage_result["coverage_percentage"]
            
            logging.info(f"📊 Initial coverage: {current_coverage:.2%}")
            
            if current_coverage < target_coverage:
                gaps = coverage_result["gaps"]
                logging.info(f"🔧 Coverage below target ({current_coverage:.2%} < {target_coverage:.0%}), filling {len(gaps)} gaps...")
                
                # Sort gaps by size (largest first) to maximize coverage improvement
                gaps_sorted = sorted(gaps, key=lambda g: g["size"], reverse=True)
                
                gap_fill_iterations = 0
                max_gap_fill_iterations = 5
                
                while current_coverage < target_coverage and gap_fill_iterations < max_gap_fill_iterations and gaps_sorted:
                    gap_fill_iterations += 1
                    
                    # Fill top gaps (largest gaps first)
                    gaps_to_fill = gaps_sorted[:3]  # Fill top 3 gaps per iteration
                    
                    for gap in gaps_to_fill:
                        gap_chunks = fetch_chunks_for_gap(
                            document_id=document_id,
                            gap_start=gap["start"],
                            gap_end=gap["end"],
                            space_id=space_id,
                            acl_tags=acl_tags
                        )
                        
                        # Add new chunks (avoid duplicates)
                        new_chunks_added = 0
                        for chunk in gap_chunks:
                            if chunk["id"] not in chunk_ids:
                                chunks.append(chunk)
                                chunk_ids.add(chunk["id"])
                                new_chunks_added += 1
                        
                        logging.info(f"   Gap [{gap['start']}, {gap['end']}] ({gap['size']:,} chars): +{new_chunks_added} chunks")
                    
                    # Recalculate coverage
                    coverage_result = calculate_coverage_from_chunks(chunks)
                    new_coverage = coverage_result["coverage_percentage"]
                    
                    logging.info(f"   Iteration {gap_fill_iterations}: Coverage improved from {current_coverage:.2%} → {new_coverage:.2%}")
                    
                    if new_coverage <= current_coverage:
                        logging.info("   Coverage not improving, stopping gap-filling")
                        break
                    
                    current_coverage = new_coverage
                    gaps_sorted = sorted(coverage_result["gaps"], key=lambda g: g["size"], reverse=True)
                
                logging.info(f"✅ Gap-filling complete: Final coverage {current_coverage:.2%}")

        # Sort chunks by position in document
        def get_sort_key(chunk: Dict[str, Any]) -> tuple:
            metadata = chunk.get("metadata", {})

            # Get chapter number
            chapter_num = metadata.get("chapter_number", 0)
            if isinstance(chapter_num, str):
                try:
                    chapter_num = int(chapter_num)
                except (ValueError, TypeError):
                    chapter_num = 0

            # Get page number
            page_num = metadata.get("page_number", "0")
            if isinstance(page_num, str):
                page_num = page_num.split(',')[0] if page_num else "0"
                try:
                    page_num = int(page_num)
                except (ValueError, TypeError):
                    page_num = 0

            # Get start_index for precise position tracking
            start_idx = metadata.get("start_index", 0)
            if isinstance(start_idx, str):
                try:
                    start_idx = int(start_idx)
                except (ValueError, TypeError):
                    start_idx = 0

            # Fallback: Try to infer position from chunk ID
            chunk_id = chunk.get("id", "")
            position = 0
            if "::" in chunk_id and "_" in chunk_id:
                try:
                    position = int(chunk_id.split("_")[-1])
                except (ValueError, IndexError):
                    position = 0

            return (chapter_num, page_num, start_idx, position)

        sorted_chunks = sorted(chunks, key=get_sort_key)

        logging.info(f"✅ Fetched and sorted {len(sorted_chunks)} chunks for note generation")

        return sorted_chunks

    except Exception as e:
        logging.error(f"❌ Error fetching chunks: {e}")
        return []


def fetch_document_vector_ids(
    document_id: str,
    max_ids: int = 5000
) -> List[str]:
    """
    Fetch all Pinecone vector IDs for a given document_id.

    Uses multiple semantically diverse query embeddings to maximise HNSW graph coverage,
    then deduplicates. A single query against a zero vector is unreliable because Pinecone
    uses approximate nearest-neighbour traversal — remote graph nodes can be missed.

    Args:
        document_id: The document/course ID to fetch vectors for.
        max_ids: Upper bound on total unique IDs to retrieve (default 5000).

    Returns:
        List of deduplicated vector ID strings.
    """
    from clients.chonkie_client import embed_query_multimodal

    dense_index = pc.Index(DENSE_INDEX)
    filter_dict = {"document_id": {"$eq": document_id}}

    # Diverse seed queries to maximise HNSW coverage across different semantic neighbourhoods
    seed_queries = [
        "concepts definitions principles theory",
        "examples applications methods steps",
        "data results analysis summary",
        "introduction overview background context",
    ]

    seen_ids: set = set()
    top_k_per_query = min(max_ids, 10000)  # Pinecone hard cap is 10 000

    for seed in seed_queries:
        if len(seen_ids) >= max_ids:
            break
        try:
            emb = embed_query_multimodal(seed)["embedding"]
            results = dense_index.query(
                vector=emb,
                filter=filter_dict,
                top_k=top_k_per_query,
                include_metadata=False,
                include_values=False,
            )
            for match in results.matches:
                seen_ids.add(match.id)
                if len(seen_ids) >= max_ids:
                    break
        except Exception as e:
            logging.warning(f"fetch_document_vector_ids: seed query failed ({seed!r}): {e}")

    ids = list(seen_ids)
    logging.info(f"fetch_document_vector_ids: found {len(ids)} vectors for document {document_id}")
    return ids


def update_vector_metadata(
    vector_ids: List[str],
    metadata_update: Dict[str, Any],
    max_workers: int = 10
) -> int:
    """
    Update metadata on existing Pinecone vectors (best-effort, parallel).

    Args:
        vector_ids: List of vector IDs to update.
        metadata_update: Dict of metadata fields to set/overwrite.
        max_workers: Max parallel threads for updates.

    Returns:
        Count of successfully updated vectors.
    """
    if not vector_ids:
        return 0

    dense_index = pc.Index(DENSE_INDEX)
    success_count = 0

    def _update_single(vid: str) -> bool:
        try:
            dense_index.update(id=vid, set_metadata=metadata_update)
            return True
        except Exception as e:
            logging.warning(f"Failed to update metadata for vector {vid}: {e}")
            return False

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_update_single, vid): vid for vid in vector_ids}
        for future in concurrent.futures.as_completed(futures):
            if future.result():
                success_count += 1

    logging.info(f"Updated metadata on {success_count}/{len(vector_ids)} vectors")
    return success_count


def fetch_chunks_by_space(
    space_id: str,
    document_ids: List[str],
    max_chunks_per_doc: int = 2000,
    target_coverage: float = 0.80,
    enable_gap_filling: bool = True
) -> List[Dict[str, Any]]:
    """
    Fetch chunks for multiple documents in a space by calling
    fetch_all_document_chunks() per document and merging results.

    Args:
        space_id: The space ID that all documents belong to.
        document_ids: List of document IDs to fetch chunks for.
        max_chunks_per_doc: Max chunks per individual document fetch.
        target_coverage: Target coverage for gap-filling.
        enable_gap_filling: Whether to enable gap-filling.

    Returns:
        Merged list of chunks from all documents, sorted by position.
    """
    all_chunks = []
    chunk_ids_seen = set()

    for doc_id in document_ids:
        logging.info(f"Fetching chunks for document {doc_id} in space {space_id}")
        doc_chunks = fetch_all_document_chunks(
            document_id=doc_id,
            space_id=space_id,
            max_chunks=max_chunks_per_doc,
            target_coverage=target_coverage,
            enable_gap_filling=enable_gap_filling
        )
        for chunk in doc_chunks:
            cid = chunk.get("id")
            if cid not in chunk_ids_seen:
                chunk_ids_seen.add(cid)
                all_chunks.append(chunk)

    logging.info(f"Fetched {len(all_chunks)} total chunks across {len(document_ids)} documents in space {space_id}")
    return all_chunks 