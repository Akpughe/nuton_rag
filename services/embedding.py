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
    def __init__(self, dimension=1536, model="text-embedding-ada-002"):
        logger.info(f"Initializing MultiFileVectorIndexer with dimension={dimension}, model={model}")
        
        # Validate environment variables
        self.api_key = os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            logger.error("OPENAI_API_KEY not found in environment variables")
            raise ValueError("OPENAI_API_KEY not found")
            
        self.openai_client = OpenAI(api_key=self.api_key)
        
        # Validate Pinecone credentials
        pinecone_api_key = os.getenv('PINECONE_API_KEY')
        pinecone_index_name = os.getenv('PINECONE_INDEX_NAME')
        
        if not pinecone_api_key or not pinecone_index_name:
            logger.error(f"Pinecone credentials missing: API_KEY={bool(pinecone_api_key)}, INDEX_NAME={bool(pinecone_index_name)}")
            raise ValueError("Pinecone credentials missing")
            
        logger.info(f"Connecting to Pinecone index: {pinecone_index_name}")
        pc = Pinecone(api_key=pinecone_api_key)
        
        try:
            self.index = pc.Index(pinecone_index_name)
            
            # Get index stats and validate dimensions
            stats = self.index.describe_index_stats()
            logger.info(f"Index stats: {stats}")
            
            # Validate index is not empty
            if stats.get('total_vector_count', 0) == 0:
                logger.warning("The Pinecone index is empty. This might indicate initialization issues.")
            
            # Ensure dimension matches Pinecone's expectation
            pinecone_dim = stats.get('dimension')
            if pinecone_dim and pinecone_dim != dimension:
                logger.warning(f"Dimension mismatch: Indexer configured for {dimension} but Pinecone uses {pinecone_dim}")
                self.dimension = pinecone_dim
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
            logger.warning(f"Truncating embedding from {current_dim} to {target_dim}")
            return embedding[:target_dim]
        else:
            logger.warning(f"Zero-padding embedding from {current_dim} to {target_dim}")
            return embedding + [0.0] * (target_dim - current_dim)
    
    def embed_and_index_chunks(self, chunks: List[Document], batch_size: int = 100) -> Dict[str, Any]:
        """
        Batch process embeddings and index in Pinecone
        Uses document_id from metadata as vector ID when available
        """
        if not chunks:
            logger.warning("No chunks provided for embedding and indexing")
            return {"total_chunks": 0, "indexed_chunks": 0, "failed_chunks": 0, "indexed_docs": set()}
            
        logger.info(f"Starting embedding and indexing of {len(chunks)} chunks with batch size {batch_size}")
        
        # Validate Pinecone dimension
        try:
            index_stats = self.index.describe_index_stats()
            pinecone_dim = index_stats.get('dimension')
            
            if pinecone_dim and pinecone_dim != self.dimension:
                logger.warning(f"Dimension mismatch: configured={self.dimension}, Pinecone index={pinecone_dim}")
                self.dimension = pinecone_dim
        except Exception as e:
            logger.error(f"Error checking Pinecone dimension: {str(e)}")
        
        # Initialize results tracking
        indexing_results = {
            'total_chunks': len(chunks),
            'indexed_chunks': 0,
            'failed_chunks': 0,
            'indexed_docs': set()
        }
        
        # Process chunks in batches
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i+batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(chunks) + batch_size - 1) // batch_size
            
            logger.info(f"Processing batch {batch_num}/{total_batches} with {len(batch_chunks)} chunks")
            
            try:
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
                    # Extract and adjust embedding if needed
                    embedding = embedding_data.embedding
                    if len(embedding) != self.dimension:
                        embedding = self.pad_embedding(embedding, self.dimension)
                    
                    # Get IDs from metadata
                    document_id = chunk.metadata.get('document_id')
                    video_id = chunk.metadata.get('video_id')
                    space_id = chunk.metadata.get('space_id')
                    
                    # Use video_id as document_id if needed
                    if not document_id and video_id:
                        document_id = video_id
                    
                    # Create vector ID
                    if document_id and space_id:
                        vector_id = f"doc_{document_id}_{len(vectors)}"
                        indexing_results['indexed_docs'].add(document_id)
                    else:
                        vector_id = f"chunk_{i}_{len(vectors)}"
                        logger.warning(f"Missing document_id or space_id in chunk metadata: {chunk.metadata}")
                    
                    # Get source type for filtering
                    source_type = chunk.metadata.get('source_type', 'unknown')
                    
                    # Log chunk metadata before building the payload for Pinecone
                    logger.info(f"Processing chunk for vector_id approx {i}_{len(vectors)}. Metadata keys: {list(chunk.metadata.keys())}, Page value: {chunk.metadata.get('page')}, Line_in_page value: {chunk.metadata.get('line_in_page')}")

                    # Build metadata
                    metadata = {
                        'text': chunk.page_content,
                        'source_file': chunk.metadata.get('source_file', 'unknown'),
                        'document_id': document_id,
                        'space_id': space_id,
                        'original_source': chunk.metadata.get('source', 'unknown'),
                        'source_type': source_type,
                        'page': chunk.metadata.get('page'),
                        'line_in_page': chunk.metadata.get('line_in_page')
                    }
                    
                    # Ensure 'page' is an int or a placeholder like -1
                    if chunk.metadata.get('page') is None and source_type == 'document':
                        metadata['page'] = -1 
                    elif isinstance(chunk.metadata.get('page'), str): # Ensure it's not a string by mistake
                        try:
                            metadata['page'] = int(chunk.metadata.get('page'))
                        except ValueError:
                            metadata['page'] = -1 # Fallback for non-integer string page

                    # Ensure 'line_in_page' is an int or a placeholder like -1, if it exists
                    if 'line_in_page' in chunk.metadata and chunk.metadata.get('line_in_page') is None:
                        metadata['line_in_page'] = -1
                    elif 'line_in_page' in chunk.metadata and isinstance(chunk.metadata.get('line_in_page'), str):
                        try:
                            metadata['line_in_page'] = int(chunk.metadata.get('line_in_page'))
                        except ValueError:
                            metadata['line_in_page'] = -1 # Fallback
                    elif 'line_in_page' not in chunk.metadata: # If not present at all, explicitly set to null/placeholder
                        metadata['line_in_page'] = -1 # Or None, depending on how Pinecone handles nulls and your query needs
                    
                    # Add YouTube-specific fields if present
                    if source_type == 'youtube_video':
                        metadata.update({
                            'video_id': video_id,
                            'youtube_id': chunk.metadata.get('youtube_id'),
                            'thumbnail': chunk.metadata.get('thumbnail'),
                            'title': chunk.metadata.get('title', 'YouTube Video'),
                            'source_url': chunk.metadata.get('source_url')
                        })
                    
                    vectors.append({
                        'id': vector_id,
                        'values': embedding,
                        'metadata': metadata
                    })
                
                # Upsert to Pinecone
                if vectors:
                    logger.info(f"Upserting {len(vectors)} vectors to Pinecone")
                    upsert_response = self.index.upsert(vectors)
                    logger.info(f"Upsert response: {upsert_response}")
                    
                    indexing_results['indexed_chunks'] += len(vectors)
                
            except Exception as e:
                logger.error(f"Error processing batch {batch_num}: {str(e)}")
                indexing_results['failed_chunks'] += len(batch_chunks)
        
        logger.info(f"Embedding and indexing complete. Results: {indexing_results}")
        return indexing_results