import os
import logging
import time
from dotenv import load_dotenv
from pinecone import Pinecone
from openai import OpenAI
from services.reranking import RetrievalEngine
from services.embedding import MultiFileVectorIndexer
from langchain.docstore.document import Document

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def main():
    logger.info("=== Testing Integrated RAG System ===")
    
    try:
        # Initialize OpenAI client
        openai_api_key = os.getenv('OPENAI_API_KEY')
        if not openai_api_key:
            logger.error("OpenAI API key not found")
            return
            
        logger.info("OpenAI API key found")
        client = OpenAI(api_key=openai_api_key)
        
        # 1. Initialize Vector Indexer
        logger.info("Initializing MultiFileVectorIndexer...")
        vector_indexer = MultiFileVectorIndexer(dimension=1536, model="text-embedding-ada-002")
        
        # 2. Initialize Retrieval Engine
        logger.info("Initializing RetrievalEngine...")
        retrieval_engine = RetrievalEngine(top_k=10)
        
        # 3. First check the index state and clean it up
        logger.info("Checking index state...")
        index_stats = vector_indexer.index.describe_index_stats()
        logger.info(f"Index stats: {index_stats}")
        
        # 4. Delete all existing vectors
        logger.info("Deleting all existing vectors...")
        delete_response = vector_indexer.index.delete(delete_all=True)
        logger.info(f"Delete response: {delete_response}")
        
        # Wait for the delete to propagate
        logger.info("Waiting for delete operation to complete...")
        time.sleep(5)
        
        # 5. Confirm deletion
        index_stats_after_delete = vector_indexer.index.describe_index_stats()
        logger.info(f"Index stats after delete: {index_stats_after_delete}")
        
        # 6. Create test documents
        logger.info("Creating test documents...")
        test_docs = [
            Document(
                page_content="This is a test document about artificial intelligence and machine learning.",
                metadata={
                    "document_id": "doc123",
                    "space_id": "space456",
                    "source_file": "ai_intro.pdf"
                }
            ),
            Document(
                page_content="Neural networks are a fundamental part of deep learning algorithms.",
                metadata={
                    "document_id": "doc456",
                    "space_id": "space456",
                    "source_file": "neural_networks.pdf"
                }
            ),
            Document(
                page_content="Natural language processing helps computers understand human language.",
                metadata={
                    "document_id": "doc789",
                    "space_id": "space789",
                    "source_file": "nlp_intro.pdf"
                }
            ),
            Document(
                page_content="This is a document with the target IDs we need to test with.",
                metadata={
                    "document_id": "7b1024dd-5b31-445c-8c33-89b9e3c41fc9",
                    "space_id": "1034b6e2-29f5-4620-a0f2-27fe8e04b98d",
                    "source_file": "target_doc.pdf"
                }
            )
        ]
        
        # 7. Index the documents
        logger.info("Indexing documents...")
        indexing_result = vector_indexer.embed_and_index_chunks(test_docs)
        logger.info(f"Indexing result: {indexing_result}")
        
        # Wait for indexing to propagate
        logger.info("Waiting for indexing to propagate...")
        time.sleep(5)
        
        # 8. Check index stats again
        index_stats_after_index = vector_indexer.index.describe_index_stats()
        logger.info(f"Index stats after indexing: {index_stats_after_index}")
        
        # 9. Generate a query embedding
        logger.info("Generating query embedding...")
        query = "What is artificial intelligence?"
        embedding_response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=query
        )
        
        query_embedding = embedding_response.data[0].embedding
        logger.info(f"Generated query embedding with {len(query_embedding)} dimensions")
        
        # 10. Test different query scenarios
        test_scenarios = [
            {
                "name": "No filters",
                "document_ids": None,
                "space_id": None
            },
            {
                "name": "Filter by document_id doc123",
                "document_ids": ["doc123"],
                "space_id": None
            },
            {
                "name": "Filter by space_id space456",
                "document_ids": None,
                "space_id": "space456"
            },
            {
                "name": "Filter by both document_id and space_id",
                "document_ids": ["doc123"],
                "space_id": "space456"
            },
            {
                "name": "Multiple document_ids",
                "document_ids": ["doc123", "doc456"],
                "space_id": None
            },
            {
                "name": "Target IDs",
                "document_ids": ["7b1024dd-5b31-445c-8c33-89b9e3c41fc9"],
                "space_id": "1034b6e2-29f5-4620-a0f2-27fe8e04b98d"
            }
        ]
        
        for scenario in test_scenarios:
            logger.info(f"\nTesting scenario: {scenario['name']}...")
            
            results = retrieval_engine.retrieve_and_rerank(
                query,
                [{'embedding': query_embedding}],
                document_ids=scenario['document_ids'],
                space_id=scenario['space_id']
            )
            
            logger.info(f"Retrieved {len(results.results) if hasattr(results, 'results') else 0} results")
            
            if hasattr(results, 'results') and results.results:
                for i, result in enumerate(results.results):
                    logger.info(f"Result {i+1}: document_id={result.document_id}, space_id={result.space_id}")
                    logger.info(f"  Text: {result.document[:100]}...")
                    logger.info(f"  Score: {result.relevance_score}")
            else:
                logger.info("No results found for this scenario")
                
        logger.info("\n=== Test Summary ===")
        logger.info("All tests completed")
        
    except Exception as e:
        logger.error(f"Test failed with error: {str(e)}", exc_info=True)
        
if __name__ == "__main__":
    main() 