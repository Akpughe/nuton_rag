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
        
        # Check if the index has any vectors
        total_vector_count = index_stats.get('total_vector_count', 0)
        logger.info(f"Total vectors in index: {total_vector_count}")
        
        if total_vector_count == 0:
            logger.warning("The index is empty! Inserting a test vector...")
            
            # Generate a test embedding
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
            
            # Insert test vector
            test_vector = {
                'id': 'test_vector',
                'values': embedding,
                'metadata': {
                    'text': 'This is a test document for the RAG system.',
                    'document_id': 'test_doc',
                    'space_id': 'test_space'
                }
            }
            
            upsert_response = index.upsert([test_vector])
            logger.info(f"Upsert response: {upsert_response}")
            
            # Check index stats again
            index_stats = index.describe_index_stats()
            logger.info(f"Updated index stats: {index_stats}")
            
        # Test query
        logger.info("Testing query...")
        
        # Generate query embedding
        query_text = "What is this document about?"
        query_embedding_response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=query_text
        )
        
        query_embedding = query_embedding_response.data[0].embedding
        logger.info(f"Generated query embedding with {len(query_embedding)} dimensions")
        
        # Get index dimension
        index_dim = index_stats.get('dimension')
        
        # Adjust embedding dimension if needed
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
        
        # Perform query with space_id and document_id filters
        # Without filters first
        basic_query_results = index.query(
            vector=query_embedding,
            top_k=5,
            include_metadata=True
        )
        
        logger.info(f"Basic query results: {len(basic_query_results.get('matches', []))} matches")
        
        # Test query with filters
        filter_query_results = index.query(
            vector=query_embedding,
            top_k=5,
            include_metadata=True,
            filter={
                "document_id": {"$in": ["test_doc"]},
                "space_id": "test_space"
            }
        )
        
        logger.info(f"Filter query results: {len(filter_query_results.get('matches', []))} matches")
        
        if filter_query_results.get('matches'):
            logger.info("Success! Query with filters returned results.")
            
            # Print first match details
            match = filter_query_results['matches'][0]
            logger.info(f"Match ID: {match.get('id')}")
            logger.info(f"Match score: {match.get('score')}")
            logger.info(f"Match metadata: {match.get('metadata')}")
        else:
            logger.error("Failed! Query with filters returned no results.")
            logger.info("Testing filter_query query parameters...")
            
            # Test with just one filter to see if the issue is specific to one
            doc_id_filter_results = index.query(
                vector=query_embedding,
                top_k=5,
                include_metadata=True,
                filter={"document_id": {"$in": ["test_doc"]}}
            )
            
            logger.info(f"Document ID filter only: {len(doc_id_filter_results.get('matches', []))} matches")
            
            space_id_filter_results = index.query(
                vector=query_embedding,
                top_k=5,
                include_metadata=True,
                filter={"space_id": "test_space"}
            )
            
            logger.info(f"Space ID filter only: {len(space_id_filter_results.get('matches', []))} matches")
        
        # Log the document_ids and space_ids configured in the main query
        logger.info("Target document_ids: ['7b1024dd-5b31-445c-8c33-89b9e3c41fc9']")
        logger.info("Target space_id: '1034b6e2-29f5-4620-a0f2-27fe8e04b98d'")
        
        # Try the actual query that's failing
        target_query_results = index.query(
            vector=query_embedding,
            top_k=5,
            include_metadata=True,
            filter={
                "document_id": {"$in": ["7b1024dd-5b31-445c-8c33-89b9e3c41fc9"]},
                "space_id": "1034b6e2-29f5-4620-a0f2-27fe8e04b98d"
            }
        )
        
        logger.info(f"Target query results: {len(target_query_results.get('matches', []))} matches")
        
        # If no results, check if these IDs exist in the index at all
        if not target_query_results.get('matches'):
            logger.info("No matches for target query. Testing if these IDs exist in the index...")
            
            doc_only_results = index.query(
                vector=query_embedding,
                top_k=5,
                include_metadata=True,
                filter={"document_id": {"$in": ["7b1024dd-5b31-445c-8c33-89b9e3c41fc9"]}}
            )
            
            logger.info(f"Document ID only: {len(doc_only_results.get('matches', []))} matches")
            
            space_only_results = index.query(
                vector=query_embedding,
                top_k=5,
                include_metadata=True,
                filter={"space_id": "1034b6e2-29f5-4620-a0f2-27fe8e04b98d"}
            )
            
            logger.info(f"Space ID only: {len(space_only_results.get('matches', []))} matches")
            
            # Try a broader query to see what document_ids and space_ids are in the index
            all_results = index.query(
                vector=query_embedding,
                top_k=10,
                include_metadata=True
            )
            
            logger.info(f"All results (no filter): {len(all_results.get('matches', []))} matches")
            
            # Extract unique document_ids and space_ids
            doc_ids = set()
            space_ids = set()
            
            for match in all_results.get('matches', []):
                metadata = match.get('metadata', {})
                doc_id = metadata.get('document_id')
                space_id = metadata.get('space_id')
                
                if doc_id:
                    doc_ids.add(doc_id)
                if space_id:
                    space_ids.add(space_id)
            
            logger.info(f"Found document_ids in index: {doc_ids}")
            logger.info(f"Found space_ids in index: {space_ids}")
    
    except Exception as e:
        logger.error(f"Test failed with error: {str(e)}")
        
if __name__ == "__main__":
    main() 