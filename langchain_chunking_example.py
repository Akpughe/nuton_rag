#!/usr/bin/env python3
"""
Free chunking using LangChain Text Splitters (already installed!)
Drop-in replacement for Chonkie API - works immediately, no additional installation.
"""

import os
import time
import logging
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import tiktoken
from openai import OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

logging.basicConfig(level=logging.INFO)


def chunk_document(
    file_path: Optional[str] = None,
    text: Optional[str] = None,
    chunk_size: int = 512,
    overlap_tokens: int = 80,
    tokenizer: str = "cl100k_base",
    chunker_type: str = "recursive"
) -> List[Dict[str, Any]]:
    """
    Chunk a document using LangChain Text Splitters (FREE, already installed).

    Args:
        file_path: Path to the file to chunk. Optional if text is provided.
        text: Text to chunk directly. Optional if file_path is provided.
        chunk_size: Target chunk size in tokens.
        overlap_tokens: Number of tokens to overlap between chunks.
        tokenizer: Tokenizer to use ('cl100k_base' for GPT-4, 'p50k_base' for GPT-3).
        chunker_type: Type of chunker ('recursive' or 'character').

    Returns:
        List of chunk dicts with 'text', 'start_index', 'end_index', 'token_count'.
    """
    if not file_path and not text:
        error_msg = "Either file_path or text must be provided"
        logging.error(error_msg)
        raise ValueError(error_msg)

    # Read file if file_path is provided
    if file_path:
        logging.info(f"Reading document from file: {file_path}")

        if file_path.lower().endswith('.pdf'):
            # Use PyPDF2 (already in requirements)
            try:
                import PyPDF2
                with open(file_path, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    text = "\n\n".join(page.extract_text() for page in reader.pages)
            except ImportError:
                raise ImportError("PDF support requires PyPDF2 (already in requirements.txt)")
        else:
            # Read text files directly
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()

    logging.info(f"Chunking document with LangChain ({chunker_type} splitter)")

    # Initialize the text splitter
    if chunker_type == "recursive":
        # Recursive splitter (best for general text)
        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name=tokenizer,
            chunk_size=chunk_size,
            chunk_overlap=overlap_tokens,
            separators=["\n\n", "\n", ". ", " ", ""]  # Try to split on paragraphs, sentences, etc.
        )
    elif chunker_type == "character":
        # Simple character splitter
        splitter = CharacterTextSplitter.from_tiktoken_encoder(
            encoding_name=tokenizer,
            chunk_size=chunk_size,
            chunk_overlap=overlap_tokens,
            separator="\n\n"
        )
    else:
        raise ValueError(f"Unknown chunker_type: {chunker_type}. Use 'recursive' or 'character'")

    # Split the text
    text_chunks = splitter.split_text(text)

    # Initialize tokenizer for counting
    enc = tiktoken.get_encoding(tokenizer)

    # Convert to dict format matching Chonkie's output
    chunks = []
    current_index = 0

    for chunk_text in text_chunks:
        # Find the chunk in the original text
        chunk_start = text.find(chunk_text, current_index)
        if chunk_start == -1:
            chunk_start = current_index

        chunk_end = chunk_start + len(chunk_text)
        token_count = len(enc.encode(chunk_text))

        chunk_dict = {
            "text": chunk_text,
            "start_index": chunk_start,
            "end_index": chunk_end,
            "token_count": token_count,
        }
        chunks.append(chunk_dict)

        current_index = chunk_end

    logging.info(f"Created {len(chunks)} chunks using LangChain")
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
                formatted_chunk = chunk.copy() if isinstance(chunk, dict) else {"text": chunk}
                formatted_chunk["embedding"] = embedding_data.embedding
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

    logging.info(f"Embedding query: {query[:50]}...")

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


# Example usage and benchmark
if __name__ == "__main__":
    print("\n" + "="*70)
    print("LangChain Text Splitter - FREE Alternative to Chonkie API")
    print("="*70)

    # Sample text
    sample_text = """
    Artificial Intelligence (AI) is revolutionizing technology. Machine learning, a subset of AI,
    enables computers to learn from data without explicit programming. Deep learning, using neural
    networks with multiple layers, has achieved remarkable success in image recognition, natural
    language processing, and game playing.

    Natural Language Processing (NLP) allows computers to understand and generate human language.
    Recent advances in transformer architectures, like BERT and GPT, have dramatically improved
    language understanding capabilities. These models can perform tasks such as translation,
    summarization, and question answering with impressive accuracy.

    Computer Vision enables machines to interpret visual information from the world. Convolutional
    Neural Networks (CNNs) have been particularly successful in this domain, achieving human-level
    performance in tasks like object detection and image classification.
    """ * 5  # Repeat for better benchmark

    print(f"\n📄 Text length: {len(sample_text)} characters")
    print(f"📄 Text preview: {sample_text[:100]}...")

    # Benchmark chunking
    print("\n⚡ Benchmarking chunking performance...")
    iterations = 10
    times = []

    for i in range(iterations):
        start = time.time()
        chunks = chunk_document(
            text=sample_text,
            chunk_size=512,
            overlap_tokens=80,
            tokenizer="cl100k_base",
            chunker_type="recursive"
        )
        elapsed = time.time() - start
        times.append(elapsed)

    avg_time = sum(times) / len(times)

    print(f"\n✅ Created {len(chunks)} chunks")
    print(f"\n📊 Performance Results:")
    print(f"   Average time: {avg_time*1000:.2f}ms")
    print(f"   Min time: {min(times)*1000:.2f}ms")
    print(f"   Max time: {max(times)*1000:.2f}ms")
    print(f"   Throughput: {len(sample_text) / avg_time:,.0f} chars/sec")

    # Show first chunk
    if chunks:
        print(f"\n📝 First chunk:")
        print(f"   Text: {chunks[0]['text'][:100]}...")
        print(f"   Tokens: {chunks[0]['token_count']}")
        print(f"   Range: {chunks[0]['start_index']}-{chunks[0]['end_index']}")

    # Comparison with Chonkie API
    estimated_api_time = avg_time + 0.25  # Add 250ms for network
    print(f"\n🔄 Comparison vs Chonkie API:")
    print(f"   LangChain (local): {avg_time*1000:.2f}ms")
    print(f"   Chonkie API (estimated): {estimated_api_time*1000:.2f}ms")
    print(f"   Speedup: {estimated_api_time / avg_time:.1f}x faster")

    print("\n💡 Benefits:")
    print("   ✅ FREE (no API costs)")
    print("   ✅ Fast (local processing)")
    print("   ✅ Already installed (langchain-text-splitters)")
    print("   ✅ Works offline")
    print("   ✅ No additional dependencies")

    print("\n" + "="*70)
    print("✅ Ready to use in your RAG system!")
    print("="*70)
