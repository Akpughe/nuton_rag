"""
Chonkie OSS-based chunking and embedding client.
Drop-in replacement for the Chonkie API version - completely free and local.
"""

import os
import logging
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import tiktoken
from openai import OpenAI

# Chonkie OSS imports
from chonkie import RecursiveChunker, TokenChunker
from chonkie.tokenizer import AutoTokenizer

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

logging.basicConfig(level=logging.INFO)


def chunk_document(
    file_path: Optional[str] = None,
    text: Optional[str] = None,
    chunk_size: int = 512,
    overlap_tokens: int = 80,
    tokenizer: str = "gpt2",
    chunker_type: str = "recursive",  # 'recursive' or 'token'
    **kwargs
) -> List[Dict[str, Any]]:
    """
    Chunk a document using Chonkie OSS (free, local version).

    Args:
        file_path: Path to the file to chunk. Optional if text is provided.
        text: Text to chunk directly. Optional if file_path is provided.
        chunk_size: Target chunk size in tokens.
        overlap_tokens: Number of tokens to overlap between chunks.
        tokenizer: Tokenizer to use (e.g., 'gpt2', 'cl100k_base' for GPT-4).
        chunker_type: Type of chunker ('recursive' or 'token').
        **kwargs: Additional arguments for specific chunker types.

    Returns:
        List of chunk dicts with 'text', 'start_index', 'end_index', 'token_count'.

    Raises:
        ValueError if neither file_path nor text is provided.
    """
    if not file_path and not text:
        error_msg = "Either file_path or text must be provided"
        logging.error(error_msg)
        raise ValueError(error_msg)

    # Read file if file_path is provided
    if file_path:
        logging.info(f"Reading document from file: {file_path}")

        if file_path.lower().endswith('.pdf'):
            # For PDFs, you'd need a PDF extraction library
            # Consider using PyPDF2, pdfplumber, or pymupdf
            logging.warning("PDF support requires additional libraries (PyPDF2, pdfplumber, etc.)")
            try:
                import pdfplumber
                with pdfplumber.open(file_path) as pdf:
                    text = "\n\n".join(page.extract_text() or "" for page in pdf.pages)
            except ImportError:
                raise ImportError(
                    "PDF support requires pdfplumber. Install with: pip install pdfplumber"
                )
        else:
            # Read text files directly
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()

    logging.info(f"Chunking document with Chonkie OSS ({chunker_type} chunker)")

    # Initialize the appropriate tokenizer
    # Chonkie uses AutoTokenizer which handles tiktoken models automatically
    if tokenizer in ['cl100k_base', 'p50k_base', 'r50k_base']:
        # For tiktoken models, use the tokenizer name directly
        token_counter = AutoTokenizer(tokenizer)
    else:
        # Default tokenizer (gpt2 or character-based)
        token_counter = AutoTokenizer(tokenizer)

    # Initialize the chunker
    if chunker_type == "recursive":
        chunker = RecursiveChunker(
            tokenizer=token_counter,
            chunk_size=chunk_size,
            min_characters_per_chunk=12,
            **kwargs
        )
    elif chunker_type == "token":
        chunker = TokenChunker(
            tokenizer=token_counter,
            chunk_size=chunk_size,
            chunk_overlap=overlap_tokens,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown chunker_type: {chunker_type}. Use 'recursive' or 'token'")

    # Chunk the text
    chunk_objects = chunker.chunk(text)

    # Convert Chonkie chunk objects to dicts matching your existing format
    chunks = []
    for chunk_obj in chunk_objects:
        chunk_dict = {
            "text": chunk_obj.text,
            "start_index": chunk_obj.start_index,
            "end_index": chunk_obj.end_index,
            "token_count": chunk_obj.token_count,
        }
        chunks.append(chunk_dict)

    logging.info(f"Created {len(chunks)} chunks using Chonkie OSS")
    return chunks


def embed_chunks(
    chunks: List[Dict[str, Any]],
    embedding_model: str = "text-embedding-ada-002",
    batch_size: int = 64
) -> List[Dict[str, Any]]:
    """
    Embed a list of chunks using OpenAI API directly.

    Args:
        chunks: List of chunk dicts (must include 'text').
        embedding_model: OpenAI embedding model to use.
        batch_size: Number of chunks per API call.

    Returns:
        List of chunk dicts, each with an added 'embedding' field.
    """
    if not OPENAI_API_KEY:
        error_msg = "OpenAI API key not found. Please set OPENAI_API_KEY environment variable."
        logging.error(error_msg)
        return [{"message": error_msg, "status": 500}]

    logging.info(f"Embedding {len(chunks)} chunks with OpenAI model {embedding_model}")

    client = OpenAI(api_key=OPENAI_API_KEY)
    tokenizer = tiktoken.encoding_for_model(embedding_model)

    results = []

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
            logging.info(f"Calling OpenAI API for batch {i//batch_size + 1}/{(len(chunks) + batch_size - 1)//batch_size}")
            response = client.embeddings.create(
                model=embedding_model,
                input=batch_texts
            )

            if not response.data:
                logging.error("OpenAI API returned empty response")
                continue

            # Combine original chunks with embeddings
            for chunk, embedding_data in zip(batch, response.data):
                if isinstance(chunk, str):
                    formatted_chunk = {
                        "text": chunk,
                        "start_index": 0,
                        "end_index": len(chunk),
                        "token_count": len(tokenizer.encode(chunk)),
                        "embedding": embedding_data.embedding
                    }
                else:
                    formatted_chunk = {
                        "text": chunk["text"],
                        "start_index": chunk.get("start_index", 0),
                        "end_index": chunk.get("end_index", len(chunk["text"])),
                        "token_count": chunk.get("token_count", len(tokenizer.encode(chunk["text"]))),
                        "embedding": embedding_data.embedding
                    }

                    # Preserve any other fields
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

    logging.info(f"Successfully embedded {len(results)} chunks")
    return results


def embed_query(
    query: str,
    embedding_model: str = "text-embedding-ada-002"
) -> Dict[str, Any]:
    """
    Embed a single query string for retrieval.

    Args:
        query: The query string.
        embedding_model: OpenAI embedding model to use.

    Returns:
        Dict with 'embedding' field.
    """
    if not OPENAI_API_KEY:
        error_msg = "OpenAI API key not found. Please set OPENAI_API_KEY environment variable."
        logging.error(error_msg)
        return {"message": error_msg, "status": 500}

    logging.info(f"Embedding query with OpenAI model {embedding_model}: {query[:50]}...")

    client = OpenAI(api_key=OPENAI_API_KEY)

    try:
        response = client.embeddings.create(
            model=embedding_model,
            input=[query]
        )

        if not response.data:
            error_msg = "OpenAI API returned empty response"
            logging.error(error_msg)
            return {"message": error_msg, "status": 500}

        embedding_data = response.data[0]
        result = {"embedding": embedding_data.embedding}

        logging.info("Successfully embedded query")
        return result

    except Exception as e:
        error_msg = f"Error calling OpenAI API: {str(e)}"
        logging.error(error_msg)
        return {"message": error_msg, "status": 500}


# Example usage
if __name__ == "__main__":
    # Test with sample text
    sample_text = """
    Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to the natural
    intelligence displayed by humans and animals. Leading AI textbooks define the field as the study
    of "intelligent agents": any device that perceives its environment and takes actions that maximize
    its chance of successfully achieving its goals.
    """

    # Chunk the text
    chunks = chunk_document(
        text=sample_text,
        chunk_size=512,
        overlap_tokens=80,
        tokenizer="gpt2",
        chunker_type="recursive"
    )

    print(f"\nCreated {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i+1}:")
        print(f"  Text: {chunk['text'][:100]}...")
        print(f"  Tokens: {chunk['token_count']}")
        print(f"  Range: {chunk['start_index']}-{chunk['end_index']}")

    # Embed the chunks (if you have OPENAI_API_KEY set)
    if OPENAI_API_KEY:
        embedded = embed_chunks(chunks, embedding_model="text-embedding-3-small")
        if embedded and "embedding" in embedded[0]:
            print(f"\n✅ Successfully embedded {len(embedded)} chunks")
            print(f"   Embedding dimension: {len(embedded[0]['embedding'])}")
