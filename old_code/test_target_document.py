import os
import logging
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
        logger.info(f"Index stats: {index_stats}")
        
        # Create a test document with the target IDs
        logger.info("Creating a test document with the target IDs")
        
        # Target IDs
        target_doc_id = "7b1024dd-5b31-445c-8c33-89b9e3c41fc9"
        target_space_id = "1034b6e2-29f5-4620-a0f2-27fe8e04b98d"
        
        # Generate embedding
        embedding_response = client.embeddings.create(
            model="text-embedding-ada-002",
            input="This is a test document with the target document ID and space ID for the RAG system."
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
        
        # Create test vector
        test_vector = {
            'id': f"doc_{target_doc_id}_0",
            'values': embedding,
            'metadata': {
                'text': 'This is a test document with the target document ID and space ID for the RAG system.',
                'document_id': target_doc_id,
                'space_id': target_space_id
            }
        }
        
        # Insert vector
        upsert_response = index.upsert([test_vector])
        logger.info(f"Upsert response: {upsert_response}")
        
        # Check index stats
        index_stats = index.describe_index_stats()
        logger.info(f"Updated index stats: {index_stats}")
        
        # Test query
        logger.info("Testing query with target IDs...")
        
        # Generate query embedding
        query_text = "What is this document about?"
        query_embedding_response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=query_text
        )
        
        query_embedding = query_embedding_response.data[0].embedding
        logger.info(f"Generated query embedding with {len(query_embedding)} dimensions")
        
        # Adjust query embedding dimension if needed
        if len(query_embedding) != index_dim:
            logger.warning(f"Query dimension mismatch: embedding={len(query_embedding)}, index={index_dim}")
            
            if len(query_embedding) < index_dim:
                # Pad with zeros
                padding = [0.0] * (index_dim - len(query_embedding))
                query_embedding = query_embedding + padding
                logger.info(f"Padded query embedding to {len(query_embedding)} dimensions")
            else:
                # Truncate
                query_embedding = query_embedding[:index_dim]
                logger.info(f"Truncated query embedding to {len(query_embedding)} dimensions")
        
        # Test query with target IDs
        target_query_results = index.query(
            vector=query_embedding,
            top_k=5,
            include_metadata=True,
            filter={
                "document_id": {"$in": [target_doc_id]},
                "space_id": target_space_id
            }
        )
        
        logger.info(f"Target query results: {len(target_query_results.get('matches', []))} matches")
        
        if target_query_results.get('matches'):
            logger.info("Success! Query with target IDs returned results.")
            
            # Print match details
            for i, match in enumerate(target_query_results['matches']):
                logger.info(f"Match {i+1} ID: {match.get('id')}")
                logger.info(f"Match {i+1} score: {match.get('score')}")
                logger.info(f"Match {i+1} metadata: {match.get('metadata')}")
        else:
            logger.error("Failed! Query with target IDs returned no results.")
    
    except Exception as e:
        logger.error(f"Test failed with error: {str(e)}")
        
if __name__ == "__main__":
    main() 