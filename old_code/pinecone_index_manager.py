import os
import pinecone
from openai import OpenAI
from dotenv import load_dotenv
import logging
import json
from typing import List, Dict, Optional, Union
import time

# Load environment variables and setup logging
load_dotenv()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class PineconeIndexManager:
    """
    Utility class to manage Pinecone index for document embeddings
    """
    def __init__(self):
        # Initialize Pinecone
        pinecone_api_key = os.getenv('PINECONE_API_KEY')
        pinecone_environment = os.getenv('PINECONE_ENVIRONMENT')
        if not pinecone_api_key or not pinecone_environment:
            raise ValueError("Pinecone credentials are missing")
        
        # Update to new Pinecone client initialization
        self.pc = pinecone.Pinecone(api_key=pinecone_api_key)
        self.index_name = os.getenv('PINECONE_INDEX_NAME', 'documents-index')
        
        # Check if index exists, create if it doesn't
        if self.index_name not in [index.name for index in self.pc.list_indexes()]:
            logger.info(f"Creating new Pinecone index: {self.index_name}")
            self.pc.create_index(
                name=self.index_name,
                dimension=1536,  # OpenAI embedding dimension
                metric="cosine"
            )
        
        self.index = self.pc.Index(self.index_name)
        self.openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    def get_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using OpenAI API"""
        response = self.openai_client.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        return response.data[0].embedding
    
    def index_document_chunks(self, document_id: str, chunks: List[Dict[str, Union[str, int]]]):
        """
        Index document chunks to Pinecone
        
        Args:
            document_id: Unique identifier for the document
            chunks: List of dictionaries with:
                - text: The chunk text content
                - page: Optional page number
                - chunk_id: Unique identifier for the chunk
        """
        vectors = []
        start_time = time.time()
        
        for i, chunk in enumerate(chunks):
            chunk_text = chunk['text']
            embedding = self.get_embedding(chunk_text)
            
            # Create a unique ID for this vector
            chunk_id = chunk.get('chunk_id', f"{document_id}_chunk_{i}")
            
            vectors.append({
                "id": chunk_id,
                "values": embedding,
                "metadata": {
                    "document_id": document_id,
                    "text": chunk_text,
                    "page": chunk.get('page', 'unknown')
                }
            })
            
            # Upsert in batches of 100 for better performance
            if len(vectors) >= 100:
                self.index.upsert(vectors=vectors)
                vectors = []
        
        # Upsert any remaining vectors
        if vectors:
            self.index.upsert(vectors=vectors)
        
        logger.info(f"Indexed {len(chunks)} chunks for document {document_id} in {time.time() - start_time:.2f}s")
    
    def delete_document(self, document_id: str):
        """Delete all vectors for a specific document"""
        self.index.delete(filter={"document_id": document_id})
        logger.info(f"Deleted all vectors for document {document_id}")
    
    def search_documents(self, query: str, document_ids: Optional[List[str]] = None, top_k: int = 10):
        """
        Search for relevant document chunks
        
        Args:
            query: Search query text
            document_ids: Optional list of document IDs to filter by
            top_k: Number of results to return
            
        Returns:
            List of matching document chunks with their metadata
        """
        query_embedding = self.get_embedding(query)
        
        # Create filter if document_ids is provided
        filter_dict = None
        if document_ids:
            filter_dict = {"document_id": {"$in": document_ids}}
        
        # Query the index
        results = self.index.query(
            vector=query_embedding,
            filter=filter_dict,
            top_k=top_k,
            include_metadata=True
        )
        
        return results.matches

    def get_document_stats(self, document_id: str):
        """Get statistics about a document in the index"""
        # Count vectors for this document
        stats = self.index.describe_index_stats(
            filter={"document_id": document_id}
        )
        return stats

if __name__ == "__main__":
    # Example usage
    manager = PineconeIndexManager()
    
    # Example document chunks
    example_chunks = [
        {
            "text": "Artificial intelligence is the simulation of human intelligence processes by machines.",
            "page": 1
        },
        {
            "text": "Machine learning is a subset of artificial intelligence focusing on training algorithms to make predictions.",
            "page": 2
        }
    ]
    
    # Index example document
    manager.index_document_chunks("example_doc_123", example_chunks)
    
    # Search
    results = manager.search_documents("What is artificial intelligence?")
    print(f"Found {len(results)} matching chunks") 