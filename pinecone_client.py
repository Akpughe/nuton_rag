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

# Index names
DENSE_INDEX = f"{PINECONE_INDEX_NAME}-dense"
SPARSE_INDEX = f"{PINECONE_INDEX_NAME}-sparse"

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
    source_file: Optional[str] = None
) -> None:
    """
    Upsert dense (and optionally sparse) vectors to Pinecone, with metadata.
    Args:
        doc_id: Document id to include in metadata.
        space_id: Space id to include in metadata.
        embeddings: List of dicts with 'embedding' (dense) and optionally 'sparse' fields.
        chunks: List of chunk dicts (must align with embeddings).
        batch_size: Max vectors per upsert call.
        source_file: Original document filename to include in metadata.
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