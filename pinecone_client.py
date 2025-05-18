import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import logging
from pinecone import Pinecone

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
    
    logger.info(f"Type of chunks: {type(chunks)}; Example: {chunks[:1]}")
    logger.info(f"Type of embeddings: {type(embeddings)}; Example: {embeddings[:1]}")
    
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
                "text": chunk_text[:3000]  # Limit text length for Pinecone metadata size limits
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
                    "text": chunk_text[:3000]  # Limit text length for Pinecone metadata size limits
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

def hybrid_search(
    query_emb: List[float],
    query_sparse: Optional[Dict[str, Any]] = None,
    top_k: int = 10,
    doc_id: Optional[str] = None,
    space_id: Optional[str] = None,
    acl_tags: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    Perform hybrid search in Pinecone (dense + sparse), filter by doc_id and ACL tags.
    Args:
        query_emb: Dense query embedding.
        query_sparse: Sparse query representation (BM25/ColBERT), or None.
        top_k: Number of results to retrieve from each index.
        doc_id: Filter by document id.
        space_id: Filter results to this space ID.
        acl_tags: List of ACL tags to filter by.
    Returns:
        List of merged, deduped hits (dicts with id, score, metadata).
    """
    dense_index = pc.Index(DENSE_INDEX)
    sparse_index = pc.Index(SPARSE_INDEX)
    
    # Build filter
    filter_dict = {}
    if doc_id:
        filter_dict["document_id"] = {"$eq": doc_id}
    if space_id:
        filter_dict["space_id"] = {"$eq": space_id}
    if acl_tags:
        filter_dict["acl_tags"] = {"$in": acl_tags}
    
    # Query dense
    logger.info(f"Querying dense index with filter: {filter_dict}")
    dense_results = dense_index.query(
        vector=query_emb, 
        top_k=top_k, 
        filter=filter_dict,
        include_metadata=True
    )
    # print('dense_results', dense_results)
    
    # Query sparse if available
    sparse_results = None
    if query_sparse:
        logger.info("Querying sparse index")
        sparse_results = sparse_index.query(
            vector=query_sparse, 
            top_k=top_k, 
            filter=filter_dict,
            include_metadata=True
        )
    
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
    
    logger.info(f"Returning {len(all_hits)} total deduplicated hits")
    return list(all_hits.values())

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