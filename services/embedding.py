import os
import logging
from typing import List, Dict, Any
from langchain.docstore.document import Document
from openai import OpenAI
from pinecone import Pinecone

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultiFileVectorIndexer:
    def __init__(self, dimension=1536, model="text-embedding-ada-002"):  # default to ada-002 to match query
        # Log initialization
        logger.info(f"Initializing MultiFileVectorIndexer with dimension={dimension}, model={model}")
        
        self.api_key = os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            logger.error("OPENAI_API_KEY not found in environment variables")
            raise ValueError("OPENAI_API_KEY not found")
            
        self.openai_client = OpenAI(api_key=self.api_key)
        
        pinecone_api_key = os.getenv('PINECONE_API_KEY')
        pinecone_index_name = os.getenv('PINECONE_INDEX_NAME')
        
        if not pinecone_api_key or not pinecone_index_name:
            logger.error(f"Pinecone credentials missing: API_KEY={bool(pinecone_api_key)}, INDEX_NAME={bool(pinecone_index_name)}")
            raise ValueError("Pinecone credentials missing")
            
        logger.info(f"Connecting to Pinecone index: {pinecone_index_name}")
        pc = Pinecone(api_key=pinecone_api_key)
        
        try:
            self.index = pc.Index(pinecone_index_name)
            # Log successful connection
            logger.info(f"Successfully connected to Pinecone index: {pinecone_index_name}")
            # Log index stats
            stats = self.index.describe_index_stats()
            logger.info(f"Index stats: {stats}")
            
            # Check if the index is empty
            if stats.get('total_vector_count', 0) == 0:
                logger.warning("The Pinecone index is empty. This might indicate initialization issues.")
            
            # Store the actual dimension from Pinecone
            pinecone_dim = stats.get('dimension')
            if pinecone_dim and pinecone_dim != dimension:
                logger.warning(f"Dimension mismatch: Indexer configured for {dimension} but Pinecone uses {pinecone_dim}")
                self.dimension = pinecone_dim
                logger.info(f"Updated dimension to match Pinecone: {self.dimension}")
            else:
                self.dimension = dimension
                
        except Exception as e:
            logger.error(f"Failed to connect to Pinecone index: {str(e)}")
            raise
            
        self.model = model
        logger.info(f"MultiFileVectorIndexer initialized successfully with dimension={self.dimension}")
    
    def pad_embedding(self, embedding, target_dim):
        """Pad embedding to match the target dimension"""
        current_dim = len(embedding)
        
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
    
    def embed_and_index_chunks(self, chunks: List[Document], batch_size: int = 100) -> Dict[str, Any]:
        """
        Batch process embeddings and index in Pinecone
        Uses document_id from metadata as vector ID when available
        """
        logger.info(f"Starting embedding and indexing of {len(chunks)} chunks with batch size {batch_size}")
        
        # Check the actual dimension of the Pinecone index
        try:
            index_stats = self.index.describe_index_stats()
            pinecone_dim = index_stats.get('dimension')
            logger.info(f"Pinecone index dimension: {pinecone_dim}")
            
            if pinecone_dim and pinecone_dim != self.dimension:
                logger.warning(f"Dimension mismatch: configured={self.dimension}, Pinecone index={pinecone_dim}")
                self.dimension = pinecone_dim
                logger.info(f"Updated dimension to match Pinecone index: {self.dimension}")
        except Exception as e:
            logger.error(f"Error checking Pinecone dimension: {str(e)}")
        
        # Track successful and failed chunks
        indexing_results = {
            'total_chunks': len(chunks),
            'indexed_chunks': 0,
            'failed_chunks': 0,
            'indexed_docs': set()  # Keep track of document IDs that were indexed
        }
        
        # Batch embedding generation
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i+batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(chunks) + batch_size - 1) // batch_size
            
            logger.info(f"Processing batch {batch_num}/{total_batches} with {len(batch_chunks)} chunks")
            
            try:
                # Log the first few characters of each chunk for debugging
                for j, chunk in enumerate(batch_chunks[:3]):
                    truncated_content = chunk.page_content[:100] + "..." if len(chunk.page_content) > 100 else chunk.page_content
                    logger.info(f"Chunk {i+j} preview: {truncated_content}")
                    logger.info(f"Chunk {i+j} metadata: {chunk.metadata}")
                
                # Generate embeddings
                logger.info(f"Generating embeddings with model {self.model}")
                embeddings_response = self.openai_client.embeddings.create(
                    model=self.model,
                    input=[chunk.page_content for chunk in batch_chunks]
                )
                
                logger.info(f"Successfully generated {len(embeddings_response.data)} embeddings")
                
                # Prepare vectors for Pinecone
                vectors = []
                for chunk, embedding_data in zip(batch_chunks, embeddings_response.data):
                    # Extract the embedding
                    embedding = embedding_data.embedding
                    
                    # Check if we need to adjust the embedding dimension
                    embedding_dim = len(embedding)
                    if embedding_dim != self.dimension:
                        # Use the padding helper method
                        embedding = self.pad_embedding(embedding, self.dimension)
                    
                    # Create a unique vector ID that includes document_id for filtering
                    document_id = chunk.metadata.get('document_id')
                    space_id = chunk.metadata.get('space_id')
                    
                    # If we have both IDs, use a prefix system for the vector ID
                    if document_id and space_id:
                        vector_id = f"doc_{document_id}_{len(vectors)}"
                        indexing_results['indexed_docs'].add(document_id)
                    else:
                        # Fallback to legacy ID approach
                        vector_id = f"chunk_{i}_{len(vectors)}"
                    
                    # Log vector details for debugging
                    logger.info(f"Creating vector with ID: {vector_id}, document_id: {document_id}, space_id: {space_id}")
                    
                    vectors.append({
                        'id': vector_id,
                        'values': embedding,
                        'metadata': {
                            'text': chunk.page_content,
                            'source_file': chunk.metadata.get('source_file', 'unknown'),
                            'document_id': document_id,
                            'space_id': space_id,
                            'original_source': chunk.metadata.get('source', 'unknown')
                        }
                    })
                
                # Upsert to Pinecone
                logger.info(f"Upserting {len(vectors)} vectors to Pinecone")
                upsert_response = self.index.upsert(vectors)
                logger.info(f"Upsert response: {upsert_response}")
                
                # Verify data was inserted
                index_stats_after = self.index.describe_index_stats()
                logger.info(f"Index stats after upsert: {index_stats_after}")
                
                indexing_results['indexed_chunks'] += len(vectors)
                logger.info(f"Successfully indexed {len(vectors)} vectors in batch {batch_num}")
            
            except Exception as e:
                logger.error(f"Error processing batch {batch_num}: {str(e)}")
                indexing_results['failed_chunks'] += len(batch_chunks)
        
        logger.info(f"Embedding and indexing complete. Results: {indexing_results}")
        return indexing_results