import os
import logging
import time
from dotenv import load_dotenv
from pinecone import Pinecone
from openai import OpenAI

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def main():
    # Test Pinecone and OpenAI connections
    try:
        # Initialize OpenAI
        openai_api_key = os.getenv('OPENAI_API_KEY')
        if not openai_api_key:
            logger.error("OpenAI API key not found")
            return
            
        logger.info("OpenAI API key found")
        client = OpenAI(api_key=openai_api_key)
        
        # Initialize Pinecone
        pinecone_api_key = os.getenv('PINECONE_API_KEY')
        pinecone_index_name = os.getenv('PINECONE_INDEX_NAME')
        
        if not pinecone_api_key or not pinecone_index_name:
            logger.error(f"Pinecone credentials missing: API_KEY={bool(pinecone_api_key)}, INDEX_NAME={bool(pinecone_index_name)}")
            return
            
        logger.info(f"Connecting to Pinecone index: {pinecone_index_name}")
        pc = Pinecone(api_key=pinecone_api_key)
        
        # Get the index
        index = pc.Index(pinecone_index_name)
        
        # Get index stats
        index_stats = index.describe_index_stats()
        logger.info(f"Index stats before: {index_stats}")
        
        # List all existing vectors
        logger.info("Querying for all existing vectors...")
        try:
            # Create a query vector (all zeros)
            zero_vector = [0.0] * index_stats.get('dimension', 1536)
            
            # Query with a high limit to get all vectors
            all_vectors = index.query(
                vector=zero_vector,
                top_k=100,  # Get up to 100 vectors
                include_metadata=True
            )
            
            if all_vectors.get('matches'):
                logger.info(f"Found {len(all_vectors['matches'])} existing vectors")
                for i, match in enumerate(all_vectors['matches']):
                    logger.info(f"Vector {i+1}: id={match.get('id')}, score={match.get('score')}")
                    logger.info(f"  metadata: {match.get('metadata')}")
                    
                    # Get document_id and space_id if available
                    metadata = match.get('metadata', {})
                    doc_id = metadata.get('document_id')
                    space_id = metadata.get('space_id')
                    
                    if doc_id:
                        logger.info(f"  document_id: {doc_id}")
                    if space_id:
                        logger.info(f"  space_id: {space_id}")
            else:
                logger.info("No existing vectors found")
        except Exception as e:
            logger.error(f"Error querying existing vectors: {str(e)}")
        
        # Clean up - delete all vectors
        try:
            logger.info("Deleting all existing vectors...")
            delete_response = index.delete(delete_all=True)
            logger.info(f"Delete response: {delete_response}")
            
            # Wait a bit for the delete to propagate
            logger.info("Waiting for delete operation to complete...")
            time.sleep(5)
            
            # Check if vectors were deleted
            index_stats_after_delete = index.describe_index_stats()
            logger.info(f"Index stats after delete: {index_stats_after_delete}")
        except Exception as e:
            logger.error(f"Error deleting vectors: {str(e)}")
        
        # Create test vectors with different metadata structures
        logger.info("Creating test vectors with different metadata structures")
        
        # Generate embedding
        embedding_response = client.embeddings.create(
            model="text-embedding-ada-002",
            input="This is a test document for the RAG system."
        )
        
        embedding = embedding_response.data[0].embedding
        logger.info(f"Generated embedding with {len(embedding)} dimensions")
        
        # Get index dimension
        index_dim = index_stats.get('dimension')
        logger.info(f"Index dimension is {index_dim}")
        
        # Adjust embedding dimension if needed
        if len(embedding) != index_dim:
            logger.warning(f"Dimension mismatch: embedding={len(embedding)}, index={index_dim}")
            
            if len(embedding) < index_dim:
                # Pad with zeros
                padding = [0.0] * (index_dim - len(embedding))
                embedding = embedding + padding
                logger.info(f"Padded embedding to {len(embedding)} dimensions")
            else:
                # Truncate
                embedding = embedding[:index_dim]
                logger.info(f"Truncated embedding to {len(embedding)} dimensions")
            
        # Create test vectors with different metadata formats
        vectors = [
            {
                'id': 'test_vector_basic',
                'values': embedding,
                'metadata': {
                    'text': 'This is a basic test vector with minimal metadata.'
                }
            },
            {
                'id': 'test_vector_doc_id',
                'values': embedding,
                'metadata': {
                    'text': 'This is a test vector with document_id only.',
                    'document_id': 'doc123'
                }
            },
            {
                'id': 'test_vector_space_id',
                'values': embedding,
                'metadata': {
                    'text': 'This is a test vector with space_id only.',
                    'space_id': 'space456'
                }
            },
            {
                'id': 'test_vector_both_ids',
                'values': embedding,
                'metadata': {
                    'text': 'This is a test vector with both document_id and space_id.',
                    'document_id': 'doc789',
                    'space_id': 'space789'
                }
            },
            # Add a vector with the target IDs
            {
                'id': 'test_vector_target',
                'values': embedding,
                'metadata': {
                    'text': 'This is a test vector with the target document_id and space_id.',
                    'document_id': '7b1024dd-5b31-445c-8c33-89b9e3c41fc9',
                    'space_id': '1034b6e2-29f5-4620-a0f2-27fe8e04b98d'
                }
            }
        ]
        
        # Insert vectors
        logger.info(f"Upserting {len(vectors)} test vectors")
        upsert_response = index.upsert(vectors)
        logger.info(f"Upsert response: {upsert_response}")
        
        # Wait a bit for the upsert to propagate
        logger.info("Waiting for upsert operation to complete...")
        time.sleep(5)
        
        # Check index stats
        index_stats_after = index.describe_index_stats()
        logger.info(f"Index stats after upsert: {index_stats_after}")
        
        # List all vectors again
        logger.info("Querying all vectors after upsert...")
        all_vectors_after = index.query(
            vector=embedding,  # Use the same embedding
            top_k=100,
            include_metadata=True
        )
        
        if all_vectors_after.get('matches'):
            logger.info(f"Found {len(all_vectors_after['matches'])} vectors after upsert")
            for i, match in enumerate(all_vectors_after['matches']):
                logger.info(f"Vector {i+1}: id={match.get('id')}, score={match.get('score')}")
                logger.info(f"  metadata: {match.get('metadata')}")
        else:
            logger.error("No vectors found after upsert!")
            
        # Test queries with different filters
        filter_types = [
            ("No filter", None),
            ("Basic document_id", {"document_id": "doc123"}),
            ("Basic space_id", {"space_id": "space456"}),
            ("Both IDs", {"document_id": "doc789", "space_id": "space789"}),
            ("$in operator doc_id", {"document_id": {"$in": ["doc123", "doc789"]}}),
            ("Target IDs", {"document_id": "7b1024dd-5b31-445c-8c33-89b9e3c41fc9", "space_id": "1034b6e2-29f5-4620-a0f2-27fe8e04b98d"}),
            ("Target doc_id only", {"document_id": "7b1024dd-5b31-445c-8c33-89b9e3c41fc9"}),
            ("Target space_id only", {"space_id": "1034b6e2-29f5-4620-a0f2-27fe8e04b98d"}),
            ("Target doc_id with $in", {"document_id": {"$in": ["7b1024dd-5b31-445c-8c33-89b9e3c41fc9"]}})
        ]
        
        for filter_name, filter_dict in filter_types:
            logger.info(f"Testing query with {filter_name}...")
            query_results = index.query(
                vector=embedding,
                top_k=10,
                include_metadata=True,
                filter=filter_dict
            )
            
            match_count = len(query_results.get('matches', []))
            logger.info(f"  Results: {match_count} matches")
            
            if match_count > 0:
                for i, match in enumerate(query_results['matches']):
                    logger.info(f"  Match {i+1}: id={match.get('id')}")
                    logger.info(f"    metadata: {match.get('metadata')}")
            
    except Exception as e:
        logger.error(f"Test failed with error: {str(e)}")
        
if __name__ == "__main__":
    main() 