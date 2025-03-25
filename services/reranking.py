import cohere
from pinecone import Pinecone
import os
import logging
import numpy as np
from typing import NamedTuple, List, Dict, Any, Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RetrievalResult(NamedTuple):
    document: str
    document_id: Optional[str] = None
    space_id: Optional[str] = None
    relevance_score: float = 0.0
    index: int = 0

class RetrievalResults(NamedTuple):
    results: List[RetrievalResult]
    query: str

class RetrievalEngine:
    def __init__(self, top_k=10):
        logger.info(f"Initializing RetrievalEngine with top_k={top_k}")
        
        cohere_api_key = os.getenv('COHERE_API_KEY')
        if not cohere_api_key:
            logger.error("COHERE_API_KEY not found in environment variables")
            raise ValueError("COHERE_API_KEY not found")
            
        self.cohere_client = cohere.Client(cohere_api_key)
        logger.info("Initialized Cohere client")
        
        # Initialize Pinecone correctly
        pinecone_api_key = os.getenv('PINECONE_API_KEY')
        pinecone_index_name = os.getenv('PINECONE_INDEX_NAME')
        
        if not pinecone_api_key or not pinecone_index_name:
            logger.error(f"Pinecone credentials missing: API_KEY={bool(pinecone_api_key)}, INDEX_NAME={bool(pinecone_index_name)}")
            raise ValueError("Pinecone credentials missing")
            
        logger.info(f"Connecting to Pinecone index: {pinecone_index_name}")
        pc = Pinecone(api_key=pinecone_api_key)
        
        try:
            self.pinecone_index = pc.Index(pinecone_index_name)
            # Log successful connection
            logger.info(f"Successfully connected to Pinecone index: {pinecone_index_name}")
            # Log index stats
            stats = self.pinecone_index.describe_index_stats()
            logger.info(f"Index stats: {stats}")
            
            # Store the index dimension for later use
            self.index_dimension = stats.get('dimension', 1536)
            logger.info(f"Index dimension: {self.index_dimension}")
            
            # Check if the index is empty
            if stats.get('total_vector_count', 0) == 0:
                logger.warning("The Pinecone index is empty! This will cause queries to return no results.")
                
            # Check what metadata fields are available in the index
            namespaces = stats.get('namespaces', {})
            for ns_name, ns_data in namespaces.items():
                logger.info(f"Namespace: {ns_name}, Vector count: {ns_data.get('vector_count', 0)}")

            # Let's also log some sample vectors to understand the structure
            try:
                # Get a sample of vectors (if any exist)
                if stats.get('total_vector_count', 0) > 0:
                    logger.info("Retrieving sample vectors to analyze metadata structure...")
                    sample_query_results = self.pinecone_index.query(
                        vector=[0.1] * self.index_dimension,  # Random vector
                        top_k=2,
                        include_metadata=True
                    )
                    
                    if sample_query_results.get('matches'):
                        for i, match in enumerate(sample_query_results['matches']):
                            logger.info(f"Sample vector {i+1}: ID={match.get('id')}")
                            metadata = match.get('metadata', {})
                            logger.info(f"  Metadata keys: {list(metadata.keys())}")
                            if 'document_id' in metadata:
                                logger.info(f"  document_id: {metadata.get('document_id')}")
                            if 'space_id' in metadata:
                                logger.info(f"  space_id: {metadata.get('space_id')}")
            except Exception as e:
                logger.warning(f"Error retrieving sample vectors: {str(e)}")
                
        except Exception as e:
            logger.error(f"Failed to connect to Pinecone index: {str(e)}")
            raise
            
        self.top_k = top_k
        logger.info("RetrievalEngine initialized successfully")
    
    def pad_embedding(self, embedding, target_dim):
        """Pad embedding to match the target dimension"""
        current_dim = len(embedding)
        logger.info(f"Padding embedding from {current_dim} to {target_dim} dimensions")
        
        if current_dim == target_dim:
            return embedding
        elif current_dim > target_dim:
            # Truncate
            logger.warning(f"Truncating embedding from {current_dim} to {target_dim}")
            return embedding[:target_dim]
        else:
            # Pad with zeros
            logger.warning(f"Zero-padding embedding from {current_dim} to {target_dim}")
            padding = [0.0] * (target_dim - current_dim)
            return embedding + padding
    
    def retrieve_and_rerank(self, query, embeddings, document_ids=None, space_id=None):
        """
        Retrieve and rerank documents, with optional filtering by document_ids and space_id
        """
        # Initial Retrieval
        query_embedding = embeddings[0]['embedding']
        logger.info(f"Query: '{query}', Embedding dimension: {len(query_embedding)}")
        
        # Check if embedding dimension matches index dimension
        if len(query_embedding) != self.index_dimension:
            logger.warning(f"Embedding dimension mismatch: query={len(query_embedding)}, index={self.index_dimension}")
            query_embedding = self.pad_embedding(query_embedding, self.index_dimension)
        
        # Simplified filter approach
        filter_dict = {}
        
        # Add document_id filter if provided
        if document_ids:
            if isinstance(document_ids, list) and len(document_ids) > 1:
                filter_dict["document_id"] = {"$in": document_ids}
            else:
                # Handle single document_id (either as string or single-item list)
                doc_id = document_ids[0] if isinstance(document_ids, list) else document_ids
                filter_dict["document_id"] = doc_id
        
        # Add space_id filter if provided
        if space_id:
            filter_dict["space_id"] = space_id
            
        # Log the constructed filter
        logger.info(f"Querying Pinecone with filter: {filter_dict}")
        
        try:
            # Simple query with retry logic
            max_retries = 2
            for attempt in range(max_retries):
                try:
                    # Try the query with both filters
                    retrieval_results = self.pinecone_index.query(
                        vector=query_embedding, 
                        top_k=self.top_k, 
                        include_metadata=True,
                        filter=filter_dict if filter_dict else None
                    )
                    
                    # If we got results, stop retrying
                    if retrieval_results.get('matches'):
                        logger.info(f"Query successful on attempt {attempt+1}: {len(retrieval_results.get('matches', []))} matches")
                        break
                        
                    # If no results with both filters and this isn't the last attempt, try relaxing filters
                    if not retrieval_results.get('matches') and attempt < max_retries-1 and document_ids and space_id:
                        # Try with only document_ids filter (highest priority)
                        logger.warning("No results with both filters, trying only document_id filter")
                        doc_filter = {}
                        if isinstance(document_ids, list) and len(document_ids) > 1:
                            doc_filter["document_id"] = {"$in": document_ids}
                        else:
                            doc_id = document_ids[0] if isinstance(document_ids, list) else document_ids
                            doc_filter["document_id"] = doc_id
                        
                        filter_dict = doc_filter
                    
                except Exception as e:
                    logger.error(f"Query attempt {attempt+1} failed: {str(e)}")
                    if attempt == max_retries-1:
                        raise
                    
                    # Short delay before retry
                    import time
                    time.sleep(0.5)
                        
            # Format the results into our return structure
            results = []
            for i, match in enumerate(retrieval_results.get('matches', [])):
                metadata = match.get('metadata', {})
                document = metadata.get('text', '')
                document_id = metadata.get('document_id')
                match_space_id = metadata.get('space_id')
                score = match.get('score', 0.0)
                
                # Only include results that match the requested space_id if it was provided
                # This additional check ensures we respect both filters even if we had to relax one
                if space_id and match_space_id != space_id:
                    continue
                    
                results.append(
                    RetrievalResult(
                        document=document,
                        document_id=document_id,
                        space_id=match_space_id,
                        relevance_score=score,
                        index=i
                    )
                )
                
            logger.info(f"Final result count after all filtering: {len(results)}")
            return RetrievalResults(results=results, query=query)
                
        except Exception as e:
            logger.error(f"Error querying Pinecone: {str(e)}")
            # Return empty results when an error occurs
            return RetrievalResults(results=[], query=query)