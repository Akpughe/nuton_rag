import os
import pinecone
from openai import AsyncOpenAI
from dotenv import load_dotenv
import logging
import asyncio
from typing import List, Dict, Any, Tuple
import time

# Load environment variables and setup logging
load_dotenv()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class OptimizedContentProcessor:
    """
    Utility for optimized content processing to speed up
    quiz and flashcard generation
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
        self.index = self.pc.Index(self.index_name)
        
        self.openai_client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.max_concurrent_requests = 3  # Max concurrent API calls
    
    async def _get_embedding(self, text: str):
        """Get embedding for text using OpenAI API asynchronously"""
        response = await self.openai_client.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        return response.data[0].embedding
    
    async def extract_relevant_content(self, document_ids: List[str], query: str = "main concepts, key points, important definitions") -> Tuple[str, Dict]:
        """
        Extract the most relevant content from documents in parallel
        
        Args:
            document_ids: List of document IDs to query
            query: Search query to find relevant content
            
        Returns:
            Tuple of (concatenated content, references dictionary)
        """
        start_time = time.time()
        
        # Get query embedding
        query_embedding = await self._get_embedding(query)
        
        # Query Pinecone
        try:
            results = self.index.query(
                vector=query_embedding,
                filter={"document_id": {"$in": document_ids}},
                top_k=15,  # Get more results for better coverage
                include_metadata=True
            )
            
            # Process results
            content_chunks = []
            references = {}
            
            # Keep track of chunks we've added to avoid duplication
            seen_chunk_ids = set()
            
            for match in results.matches:
                chunk_id = match.id
                
                # Skip if we've already seen this chunk
                if chunk_id in seen_chunk_ids:
                    continue
                
                seen_chunk_ids.add(chunk_id)
                
                # Extract metadata
                if match.metadata and 'text' in match.metadata:
                    content_chunks.append(match.metadata['text'])
                    
                    # Track reference information
                    doc_id = match.metadata.get('document_id')
                    if doc_id:
                        page = match.metadata.get('page', 'unknown')
                        if doc_id not in references:
                            references[doc_id] = []
                        if page not in references[doc_id]:
                            references[doc_id].append(page)
            
            # Join content chunks
            content = "\n\n".join(content_chunks)
            logger.info(f"Extracted {len(content_chunks)} relevant chunks in {time.time() - start_time:.2f}s")
            
            return content, references
            
        except Exception as e:
            logger.error(f"Error querying Pinecone: {e}")
            return "", {}
    
    async def summarize_content(self, content: str, max_tokens: int = 4000) -> str:
        """
        Summarize content to fit within token limits
        
        Args:
            content: The content to summarize
            max_tokens: Maximum tokens to keep
            
        Returns:
            Summarized content
        """
        # If content is already short enough, return it as is
        if len(content.split()) < max_tokens * 0.75:  # Rough estimate
            return content
        
        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert at summarizing academic content while preserving key information."},
                    {"role": "user", "content": f"Summarize the following content while preserving all key concepts, definitions, and important facts. Focus on maintaining factual accuracy and capturing the most important information:\n\n{content}"}
                ],
                max_tokens=max_tokens
            )
            
            summarized = response.choices[0].message.content
            logger.info(f"Summarized content from {len(content.split())} words to {len(summarized.split())} words")
            return summarized
            
        except Exception as e:
            logger.error(f"Error summarizing content: {e}")
            return content[:max_tokens * 4]  # Rough character limit as fallback
    
    async def batch_process_document_queries(self, document_ids: List[str], queries: List[str]) -> Dict[str, List[Dict]]:
        """
        Process multiple document queries in parallel for different search intentions
        
        Args:
            document_ids: List of document IDs to search in
            queries: List of search queries (e.g., "key concepts", "definitions", "examples")
            
        Returns:
            Dictionary of query->results mapping
        """
        start_time = time.time()
        
        # Log parameters
        logger.info(f"Processing batch queries for document IDs: {document_ids}")
        
        # Check if document IDs are valid
        try:
            # Get total document stats
            stats = self.index.describe_index_stats(
                filter={"document_id": {"$in": document_ids}}
            )
            vector_count = stats.namespaces.get('', {}).get('vector_count', 0)
            logger.info(f"Found {vector_count} total vectors for the requested document IDs")
            
            if vector_count == 0:
                logger.warning(f"No vectors found for document IDs: {document_ids}")
                # Check each document ID individually
                for doc_id in document_ids:
                    doc_stats = self.index.describe_index_stats(
                        filter={"document_id": doc_id}
                    )
                    doc_vector_count = doc_stats.namespaces.get('', {}).get('vector_count', 0)
                    logger.info(f"Document {doc_id} has {doc_vector_count} vectors")
        except Exception as e:
            logger.error(f"Error checking document stats: {e}")
        
        # Generate embeddings for all queries in parallel
        embed_tasks = [self._get_embedding(query) for query in queries]
        embeddings = await asyncio.gather(*embed_tasks)
        
        # Create query-embedding pairs
        query_embeddings = dict(zip(queries, embeddings))
        
        # Query Pinecone for each embedding with throttling
        results = {}
        sem = asyncio.Semaphore(self.max_concurrent_requests)
        
        async def query_with_throttling(query: str, embedding: List[float]):
            async with sem:
                try:
                    # Add retry logic for robustness
                    retries = 3
                    for attempt in range(retries):
                        try:
                            result = self.index.query(
                                vector=embedding,
                                filter={"document_id": {"$in": document_ids}},
                                top_k=5,
                                include_metadata=True
                            )
                            
                            # Check if we got results
                            if not result.matches:
                                logger.warning(f"No matches found for query '{query}' (attempt {attempt+1}/{retries})")
                                if attempt < retries - 1:
                                    await asyncio.sleep(1)  # Wait before retry
                                    continue
                            
                            return query, result.matches
                        except Exception as query_error:
                            logger.error(f"Error on attempt {attempt+1} for query '{query}': {query_error}")
                            if attempt < retries - 1:
                                await asyncio.sleep(1)  # Wait before retry
                            else:
                                raise
                    
                    # If we reach here with no result, return empty list
                    return query, []
                except Exception as e:
                    logger.error(f"All retries failed for query '{query}': {e}")
                    return query, []
        
        # Create tasks for all queries
        tasks = [query_with_throttling(query, embedding) 
                for query, embedding in query_embeddings.items()]
        
        # Run all queries in parallel
        query_results = await asyncio.gather(*tasks)
        
        # Organize results
        for query, matches in query_results:
            results[query] = [
                {
                    "text": match.metadata.get("text", "") if hasattr(match, 'metadata') and match.metadata else "",
                    "document_id": match.metadata.get("document_id", "") if hasattr(match, 'metadata') and match.metadata else "",
                    "page": match.metadata.get("page", "unknown") if hasattr(match, 'metadata') and match.metadata else "unknown",
                    "score": match.score if hasattr(match, 'score') else 0
                } for match in matches if hasattr(match, 'metadata') and match.metadata and "text" in match.metadata
            ]
            logger.info(f"Query '{query}': found {len(results[query])} results")
        
        # If no results were found for any query, try a more general approach
        if all(len(r) == 0 for r in results.values()):
            logger.warning("No results found for any query, trying a more general approach")
            try:
                # Try without document filter as a test
                test_result = self.index.query(
                    vector=list(query_embeddings.values())[0],
                    top_k=5,
                    include_metadata=True
                )
                if test_result.matches:
                    logger.info(f"General query found {len(test_result.matches)} results without document filter")
                    # Check what document IDs are actually in the index
                    found_docs = set()
                    for match in test_result.matches:
                        if hasattr(match, 'metadata') and match.metadata and 'document_id' in match.metadata:
                            found_docs.add(match.metadata['document_id'])
                    logger.info(f"Documents found in index: {found_docs}")
                    logger.warning(f"The requested document IDs {document_ids} don't match any in the index")
            except Exception as e:
                logger.error(f"Error in general query test: {e}")
        
        logger.info(f"Batch processed {len(queries)} queries for {len(document_ids)} documents in {time.time() - start_time:.2f}s")
        return results

if __name__ == "__main__":
    # Example usage
    async def test_processor():
        processor = OptimizedContentProcessor()
        content, refs = await processor.extract_relevant_content(
            ["example_doc_123"], 
            "key concepts and definitions"
        )
        print(f"Extracted content: {len(content)} chars")
        print(f"References: {refs}")
        
        # Test batch processing
        batch_results = await processor.batch_process_document_queries(
            ["example_doc_123"],
            ["key concepts", "definitions", "examples", "formulas"]
        )
        for query, results in batch_results.items():
            print(f"Query '{query}': {len(results)} results")
    
    # Run the test
    asyncio.run(test_processor()) 