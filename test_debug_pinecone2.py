import os
import logging
import time
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
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
        pinecone_environment = os.getenv('PINECONE_ENVIRONMENT', 'us-east-1')
        
        if not pinecone_api_key or not pinecone_index_name:
            logger.error(f"Pinecone credentials missing: API_KEY={bool(pinecone_api_key)}, INDEX_NAME={bool(pinecone_index_name)}")
            return
            
        logger.info(f"Connecting to Pinecone index: {pinecone_index_name} in environment {pinecone_environment}")
        pc = Pinecone(api_key=pinecone_api_key)
        
        # Check if index exists
        indices = pc.list_indexes()
        logger.info(f"Available indices: {indices}")
        
        # Get the index
        try:
            index = pc.Index(pinecone_index_name)
            logger.info("Successfully connected to index")
        except Exception as e:
            logger.error(f"Error connecting to index: {str(e)}")
            
            # Try to create the index if it doesn't exist
            if pinecone_index_name not in indices:
                logger.info(f"Index {pinecone_index_name} not found, trying to create it...")
                try:
                    pc.create_index(
                        name=pinecone_index_name,
                        dimension=1536,  # OpenAI ada-002 dimension
                        metric="cosine",
                        spec=ServerlessSpec(cloud="aws", region=pinecone_environment)
                    )
                    logger.info(f"Index {pinecone_index_name} created successfully")
                    
                    # Wait for the index to initialize
                    logger.info("Waiting for index to initialize...")
                    time.sleep(60)  # Give it a minute to initialize
                    
                    # Get the index
                    index = pc.Index(pinecone_index_name)
                except Exception as create_error:
                    logger.error(f"Error creating index: {str(create_error)}")
                    return
            else:
                return
        
        # Get index stats
        index_stats = index.describe_index_stats()
        logger.info(f"Index stats: {index_stats}")
        
        # Define a specific namespace for our test
        namespace = "test_namespace"
        
        # Delete any existing vectors in the namespace
        try:
            logger.info(f"Deleting all vectors in namespace '{namespace}'...")
            delete_response = index.delete(namespace=namespace, delete_all=True)
            logger.info(f"Delete response: {delete_response}")
            
            # Wait for the delete to propagate
            logger.info("Waiting for delete operation to complete...")
            time.sleep(10)
            
            # Check if vectors were deleted
            index_stats_after_delete = index.describe_index_stats()
            logger.info(f"Index stats after delete: {index_stats_after_delete}")
        except Exception as e:
            logger.error(f"Error deleting vectors: {str(e)}")
        
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
        
        # Use the target ID
        target_doc_id = "7b1024dd-5b31-445c-8c33-89b9e3c41fc9"
        target_space_id = "1034b6e2-29f5-4620-a0f2-27fe8e04b98d"
        
        # Create a vector with the target IDs
        test_vector = {
            'id': 'test_vector_target',
            'values': embedding,
            'metadata': {
                'text': 'This is a test vector with the target document_id and space_id.',
                'document_id': target_doc_id,
                'space_id': target_space_id
            }
        }
        
        # Insert vector with explicit namespace
        logger.info(f"Upserting vector to namespace '{namespace}'...")
        upsert_response = index.upsert(vectors=[test_vector], namespace=namespace)
        logger.info(f"Upsert response: {upsert_response}")
        
        # Wait for the upsert to propagate
        logger.info("Waiting for upsert operation to complete...")
        time.sleep(10)
        
        # Check index stats
        index_stats_after = index.describe_index_stats()
        logger.info(f"Index stats after upsert: {index_stats_after}")
        
        # Try a simple query without filters first
        logger.info(f"Querying namespace '{namespace}' without filters...")
        query_results = index.query(
            vector=embedding,
            top_k=10,
            namespace=namespace,
            include_metadata=True
        )
        
        match_count = len(query_results.get('matches', []))
        logger.info(f"Results: {match_count} matches")
        
        if match_count > 0:
            for i, match in enumerate(query_results['matches']):
                logger.info(f"Match {i+1}: id={match.get('id')}, score={match.get('score')}")
                logger.info(f"  metadata: {match.get('metadata')}")
        else:
            logger.error("No matches found!")
            
        # Now try with filters
        logger.info(f"Querying namespace '{namespace}' with target ID filters...")
        filter_query_results = index.query(
            vector=embedding,
            top_k=10,
            namespace=namespace,
            include_metadata=True,
            filter={
                "document_id": target_doc_id,
                "space_id": target_space_id
            }
        )
        
        filter_match_count = len(filter_query_results.get('matches', []))
        logger.info(f"Filter results: {filter_match_count} matches")
        
        if filter_match_count > 0:
            for i, match in enumerate(filter_query_results['matches']):
                logger.info(f"Filter match {i+1}: id={match.get('id')}, score={match.get('score')}")
                logger.info(f"  metadata: {match.get('metadata')}")
            logger.info("SUCCESS! Filters are working correctly.")
        else:
            logger.error("No matches found with filters!")
            
            # Try different filter variations
            logger.info("Testing with just document_id filter...")
            doc_filter_results = index.query(
                vector=embedding,
                top_k=10,
                namespace=namespace,
                include_metadata=True,
                filter={"document_id": target_doc_id}
            )
            
            if len(doc_filter_results.get('matches', [])) > 0:
                logger.info("Document ID filter works!")
            else:
                logger.error("Document ID filter failed.")
                
            logger.info("Testing with just space_id filter...")
            space_filter_results = index.query(
                vector=embedding,
                top_k=10,
                namespace=namespace,
                include_metadata=True,
                filter={"space_id": target_space_id}
            )
            
            if len(space_filter_results.get('matches', [])) > 0:
                logger.info("Space ID filter works!")
            else:
                logger.error("Space ID filter failed.")
    
    except Exception as e:
        logger.error(f"Test failed with error: {str(e)}")
        
if __name__ == "__main__":
    main() 