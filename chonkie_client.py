import os
import requests
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import logging
import tiktoken
from openai import OpenAI

load_dotenv()

CHONKIE_API_KEY = os.getenv("CHONKIE_API_KEY")
CHONKIE_BASE_URL = os.getenv("CHONKIE_BASE_URL", "https://api.chonkie.ai")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

HEADERS = {
    "Authorization": f"Bearer {CHONKIE_API_KEY}",
    "Content-Type": "application/json"
}

logging.basicConfig(level=logging.INFO)


def chunk_document(
    file_path: str,
    chunk_size: int = 512,
    overlap_tokens: int = 80,
    tokenizer: str = "gpt2",
    recipe: str = "markdown",
    lang: str = "en",
    min_characters_per_chunk: int = 12,
    return_type: str = "chunks"
) -> List[Dict[str, Any]]:
    """
    Chunk a document using Chonkie's recursive chunker API (multipart/form-data, file upload).
    Args:
        file_path: Path to the file to chunk.
        chunk_size: Target chunk size in tokens.
        overlap_tokens: Number of tokens to overlap between chunks.
        tokenizer: Tokenizer to use (e.g., 'gpt2').
        recipe: Chunking recipe (default: 'default').
        lang: Language code (default: 'en').
        min_characters_per_chunk: Minimum characters per chunk (default: 12).
        return_type: Output type (default: 'chunks').
    Returns:
        List of chunk dicts with 'text', 'start', 'end', and 'metadata'.
    Raises:
        Exception if the API call fails.
    """
    logging.info(f"Chunking document: {file_path}")
    url = f"{CHONKIE_BASE_URL}/v1/chunk/recursive"
    with open(file_path, "rb") as f:
        files = {"file": (os.path.basename(file_path), f)}
        data = {
            "tokenizer_or_token_counter": tokenizer,
            "chunk_size": str(chunk_size),
            "overlap_tokens": str(overlap_tokens),
            "recipe": recipe,
            "lang": lang,
            "min_characters_per_chunk": str(min_characters_per_chunk),
            "return_type": return_type
        }
        headers = HEADERS.copy()
        headers.pop("Content-Type", None)  # Let requests set the boundary
        resp = requests.post(url, files=files, data=data, headers=headers, timeout=120)
    if resp.status_code != 200:
        logging.error(f"Chonkie chunking failed: {resp.status_code} {resp.text}")
        raise Exception(f"Chonkie chunking failed: {resp.status_code} {resp.text}")
    
    data = resp.json()
    logging.info(f"chunk_document data type: {type(data)}")
    
    # Safely handle different response formats
    if isinstance(data, dict) and "chunks" in data:
        chunks = data["chunks"]
    else:
        chunks = data
    
    # Ensure chunks is a list
    if not isinstance(chunks, list):
        logging.error(f"Expected chunks to be a list, got {type(chunks)}")
        if isinstance(chunks, dict):
            # If it's a single dict, wrap it in a list
            chunks = [chunks]
        else:
            # If it's something else, try to convert or use empty list
            logging.error(f"Unexpected chunks format: {chunks}")
            chunks = []
    
    # Safe logging
    if chunks and isinstance(chunks, list):
        logging.info(f"chunk_document returned type: {type(chunks)}, count: {len(chunks)}, example: {chunks[0] if chunks else 'empty'}")
    else:
        logging.info(f"chunk_document returned type: {type(chunks)}, but it's empty or not a list")
    
    return chunks


def embed_chunks(
    chunks: List[Dict[str, Any]],
    embedding_model: str = "text-embedding-ada-002",
    batch_size: int = 64
) -> List[Dict[str, Any]]:
    """
    Embed a list of chunks using Chonkie's embeddings refinery API.
    Args:
        chunks: List of chunk dicts (must include 'text').
        embedding_model: Embedding model to use.
        batch_size: Number of chunks per API call.
    Returns:
        List of chunk dicts, each with an added 'embedding' field.
    Raises:
        Exception if any API call fails.
    """
    # logging.info(f"Embedding chunks: {len(chunks)}")
    url = f"{CHONKIE_BASE_URL}/v1/refine/embeddings"
    results = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        payload = {
            "chunks": batch,
            "embedding_model": embedding_model
        }
        # print('payload', payload)
        resp = requests.post(url, json=payload, headers=HEADERS)
        logging.info(f"embed_chunks response: {resp.status_code}")
        print('resp', resp.text)
        if resp.status_code != 200:
            error_msg = f"Chonkie embedding failed: {resp.status_code} {resp.text}"
            logging.error(error_msg)
            # Instead of raising, return the error as a formatted object
            # This allows the pipeline to detect and handle the error
            return [{"message": error_msg, "status": resp.status_code}]
        batch_result = resp.json()
        print('batch_result', batch_result)
        # Check if the result itself contains an error message
        if isinstance(batch_result, dict) and ("error" in batch_result or "message" in batch_result):
            error_msg = batch_result.get("error", batch_result.get("message", "Unknown API error"))
            logging.error(f"API returned error: {error_msg}")
            return [{"message": error_msg, "status": resp.status_code or 500}]
        
        logging.info(f"embed_chunks batch result type: {type(batch_result)}")
        # Some versions return {"chunks": [...]}, others just [...]
        batch_chunks = batch_result.get("chunks", batch_result)
        logging.info(f"embed_chunks batch returned type: {type(batch_chunks)}, example: {batch_chunks[0] if isinstance(batch_chunks, list) and batch_chunks else 'empty'}")
        
        # Skip invalid results
        if not isinstance(batch_chunks, list):
            logging.error(f"Expected batch_chunks to be a list, got {type(batch_chunks)}")
            continue
            
        results.extend(batch_chunks)
    
    logging.info(f"embed_chunks final results type: {type(results)}, count: {len(results)}")
    
    # Final validation - make sure we have embeddings
    if not results:
        logging.error("No results after embedding")
        return [{"message": "No results after embedding", "status": 500}]
        
    # Make sure the first result has an embedding
    if isinstance(results[0], dict) and "message" in results[0] and "status" in results[0]:
        logging.error(f"Embedding results contain error: {results[0]}")
        return results
        
    return results


def embed_query(
    query: str,
    embedding_model: str = "text-embedding-ada-002"
) -> Dict[str, Any]:
    """
    Embed a single query string for retrieval (dense and/or sparse).
    Args:
        query: The query string.
        embedding_model: Embedding model to use.
    Returns:
        Dict with 'embedding' and possibly 'sparse' fields.
    """
    chunk = {"text": query}
    embedded = embed_chunks([chunk], embedding_model=embedding_model, batch_size=1)
    return embedded[0]


def embed_chunks_v2(
    chunks: List[Dict[str, Any]],
    model: str = "text-embedding-ada-002",
    batch_size: int = 64
) -> List[Dict[str, Any]]:
    """
    Embed a list of chunks using OpenAI API directly.
    
    Args:
        chunks: List of chunk dicts (must include 'text').
        model: OpenAI embedding model to use.
        batch_size: Number of chunks per API call.
        
    Returns:
        List of chunk dicts, each with an added 'embedding' field.
        
    Raises:
        Exception if API call fails or API key is missing.
    """
    if not OPENAI_API_KEY:
        error_msg = "OpenAI API key not found. Please set OPENAI_API_KEY environment variable."
        logging.error(error_msg)
        return [{"message": error_msg, "status": 500}]
    
    logging.info(f"Embedding {len(chunks)} chunks with OpenAI model {model}")
    
    # Initialize OpenAI client
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    # Initialize tokenizer for counting tokens if needed
    tokenizer = tiktoken.encoding_for_model(model)
    
    results = []
    
    # Process chunks in batches
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        
        # Extract texts from chunks
        batch_texts = []
        for chunk in batch:
            if isinstance(chunk, dict) and 'text' in chunk:
                batch_texts.append(chunk['text'])
            elif isinstance(chunk, str):
                batch_texts.append(chunk)
            else:
                logging.warning(f"Skipping invalid chunk: {chunk}")
                continue
        
        try:
            # Call OpenAI embeddings API
            logging.info(f"Calling OpenAI API for batch {i//batch_size + 1}/{(len(chunks) + batch_size - 1)//batch_size}")
            response = client.embeddings.create(
                model=model,
                input=batch_texts
            )
            
            # Process the response
            if not response.data:
                logging.error("OpenAI API returned empty response")
                continue
                
            # Combine original chunks with embeddings
            for j, (chunk, embedding_data) in enumerate(zip(batch, response.data)):
                # Create a well-structured result
                if isinstance(chunk, str):
                    # Convert string chunk to dict
                    text = chunk
                    formatted_chunk = {
                        "text": text,
                        "start_index": 0,
                        "end_index": len(text),
                        "token_count": len(tokenizer.encode(text)),
                        "embedding": embedding_data.embedding
                    }
                else:
                    # Use existing chunk dict and add the embedding
                    formatted_chunk = {
                        "text": chunk["text"],
                        "start_index": chunk.get("start_index", 0),
                        "end_index": chunk.get("end_index", len(chunk["text"])),
                        "token_count": chunk.get("token_count", len(tokenizer.encode(chunk["text"]))),
                        "embedding": embedding_data.embedding
                    }
                    
                    # Preserve any other fields from the original chunk
                    for key, value in chunk.items():
                        if key not in formatted_chunk and key != "embedding":
                            formatted_chunk[key] = value
                
                results.append(formatted_chunk)
                
            logging.info(f"Successfully embedded batch {i//batch_size + 1}")
            
        except Exception as e:
            error_msg = f"Error calling OpenAI API: {str(e)}"
            logging.error(error_msg)
            return [{"message": error_msg, "status": 500}]
    
    if not results:
        logging.error("No results after embedding")
        return [{"message": "No results after embedding", "status": 500}]
    
    logging.info(f"Successfully embedded {len(results)} chunks with OpenAI")
    return results


def embed_query_v2(
    query: str,
    model: str = "text-embedding-ada-002"
) -> Dict[str, Any]:
    """
    Embed a single query string for retrieval using OpenAI API directly.
    
    Args:
        query: The query string.
        model: OpenAI embedding model to use.
        
    Returns:
        Dict with 'embedding' field.
        
    Raises:
        Exception if API call fails or API key is missing.
    """
    if not OPENAI_API_KEY:
        error_msg = "OpenAI API key not found. Please set OPENAI_API_KEY environment variable."
        logging.error(error_msg)
        return {"message": error_msg, "status": 500}
    
    logging.info(f"Embedding query with OpenAI model {model}: {query[:50]}...")
    
    # Initialize OpenAI client
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    try:
        # Call OpenAI embeddings API
        response = client.embeddings.create(
            model=model,
            input=[query]
        )
        
        # Process the response
        if not response.data:
            error_msg = "OpenAI API returned empty response"
            logging.error(error_msg)
            return {"message": error_msg, "status": 500}
        
        # Return the embedding in the expected format
        embedding_data = response.data[0]
        result = {
            "embedding": embedding_data.embedding
        }
        
        logging.info(f"Successfully embedded query with OpenAI")
        return result
        
    except Exception as e:
        error_msg = f"Error calling OpenAI API: {str(e)}"
        logging.error(error_msg)
        return {"message": error_msg, "status": 500} 