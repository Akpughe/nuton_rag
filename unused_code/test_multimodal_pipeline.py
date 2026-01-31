"""
Test Script: Multimodal RAG Pipeline
Tests the complete flow: Mistral OCR → Chonkie RecursiveChunker → Jina CLIP-v2 → Pinecone

Usage:
    python test_multimodal_pipeline.py <file_path_or_url>

Example:
    python test_multimodal_pipeline.py document.pdf
    python test_multimodal_pipeline.py https://example.com/document.pdf
"""

import os
import sys
import json
import logging
import hashlib
from typing import Dict, Any, List
from pathlib import Path
from dotenv import load_dotenv

# Import required modules
from mistral_ocr_extractor import MistralOCRExtractor, MistralOCRConfig
from multimodal_embeddings import MultimodalEmbedder
from chonkie import RecursiveChunker
from chonkie.tokenizer import AutoTokenizer
from pinecone import Pinecone

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "nuton-index-multi-modal"  # NEW multimodal index
JINA_API_KEY = os.getenv("JINA_API_KEY")
SPACE_ID = "test-multimodal"  # Test space ID


def extract_document_with_mistral(file_path_or_url: str) -> Dict[str, Any]:
    """
    Extract document using Mistral OCR.

    Args:
        file_path_or_url: Path to file or URL

    Returns:
        Extraction result with full_text, metadata, etc.
    """
    logger.info(f"📄 Extracting document with Mistral OCR: {file_path_or_url}")

    try:
        # Configure Mistral OCR
        config = MistralOCRConfig(
            enhance_metadata_with_llm=True,
            fallback_method="legacy",
            include_images=True,
            include_image_base64=False,  # Don't include base64 for test
        )

        # Initialize extractor
        extractor = MistralOCRExtractor(config=config)

        # Extract
        result = extractor.process_document(file_path_or_url)

        logger.info(f"✅ Extraction successful!")
        logger.info(f"   Method: {result.get('extraction_method')}")
        logger.info(f"   Pages: {result.get('total_pages')}")
        logger.info(f"   Chapters: {len(result.get('chapters', []))}")
        logger.info(f"   Images: {len(result.get('images', []))}")
        logger.info(f"   Text length: {len(result.get('full_text', ''))} chars")

        return result

    except Exception as e:
        logger.error(f"❌ Mistral OCR extraction failed: {e}")
        raise


def chunk_with_chonkie(text: str, chunk_size: int = 512, overlap: int = 50) -> List[Dict[str, Any]]:
    """
    Chunk text using local Chonkie RecursiveChunker.

    Args:
        text: Text to chunk
        chunk_size: Target chunk size in tokens
        overlap: Overlap between chunks in tokens (not used by RecursiveChunker)

    Returns:
        List of chunk dicts with text, start_index, end_index, token_count
    """
    logger.info(f"✂️ Chunking with Chonkie RecursiveChunker (size={chunk_size})")

    try:
        # Initialize tokenizer
        tokenizer = AutoTokenizer("cl100k_base")  # OpenAI tokenizer

        # Initialize RecursiveChunker
        chunker = RecursiveChunker(
            tokenizer=tokenizer,
            chunk_size=chunk_size,
            min_characters_per_chunk=12
        )

        # Chunk the text
        chunk_objects = chunker.chunk(text)

        # Convert to dicts
        chunks = []
        for i, chunk_obj in enumerate(chunk_objects):
            chunk_dict = {
                "text": chunk_obj.text,
                "start_index": chunk_obj.start_index,
                "end_index": chunk_obj.end_index,
                "token_count": chunk_obj.token_count,
                "chunk_index": i
            }
            chunks.append(chunk_dict)

        logger.info(f"✅ Chunking complete: {len(chunks)} chunks")
        logger.info(f"   Total tokens: {sum(c['token_count'] for c in chunks)}")
        logger.info(f"   Avg tokens/chunk: {sum(c['token_count'] for c in chunks) / len(chunks):.1f}")

        return chunks

    except Exception as e:
        logger.error(f"❌ Chunking failed: {e}")
        raise


def embed_with_multimodal(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Embed chunks using Jina CLIP-v2 multimodal embeddings.

    Args:
        chunks: List of chunk dicts

    Returns:
        List of chunks with 'embedding' field added (1024 dimensions)
    """
    logger.info(f"🧠 Embedding {len(chunks)} chunks with Jina CLIP-v2")

    try:
        # Initialize embedder
        embedder = MultimodalEmbedder(
            model="jina-clip-v2",
            api_key=JINA_API_KEY,
            batch_size=32
        )

        # Extract texts
        texts = [chunk['text'] for chunk in chunks]

        # Embed
        embeddings = embedder.embed_texts(texts, normalize=True)

        # Add embeddings to chunks
        chunks_with_embeddings = []
        for chunk, embedding in zip(chunks, embeddings):
            chunk_copy = chunk.copy()
            chunk_copy['embedding'] = embedding
            chunks_with_embeddings.append(chunk_copy)

        logger.info(f"✅ Embedding complete!")
        logger.info(f"   Embedding dimension: {len(embeddings[0])}")
        logger.info(f"   Total embedded: {len(chunks_with_embeddings)}")

        return chunks_with_embeddings

    except Exception as e:
        logger.error(f"❌ Embedding failed: {e}")
        raise


def upsert_to_pinecone(
    chunks: List[Dict[str, Any]],
    document_id: str,
    space_id: str,
    source_file: str,
    metadata: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Upsert chunks to Pinecone multimodal index.

    Args:
        chunks: List of chunks with embeddings
        document_id: Unique document ID
        space_id: Space ID
        source_file: Source filename
        metadata: Document metadata

    Returns:
        Upsert result
    """
    logger.info(f"📤 Upserting {len(chunks)} vectors to Pinecone index: {PINECONE_INDEX_NAME}")

    try:
        # Initialize Pinecone
        pc = Pinecone(api_key=PINECONE_API_KEY)

        # Get or create index
        try:
            index = pc.Index(PINECONE_INDEX_NAME)
            logger.info(f"✅ Connected to existing index: {PINECONE_INDEX_NAME}")
        except Exception as e:
            logger.warning(f"Index not found, creating new index: {PINECONE_INDEX_NAME}")

            # Create index with 1024 dimensions (Jina CLIP-v2)
            from pinecone import ServerlessSpec

            pc.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=1024,  # Jina CLIP-v2 dimension
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                )
            )

            import time
            time.sleep(5)  # Wait for index to be ready

            index = pc.Index(PINECONE_INDEX_NAME)
            logger.info(f"✅ Created new index: {PINECONE_INDEX_NAME}")

        # Prepare vectors for upsert
        vectors = []
        for i, chunk in enumerate(chunks):
            vector_id = f"{document_id}::chunk_{i}"

            # Build metadata
            vector_metadata = {
                "text": chunk['text'][:40000],  # Pinecone limit
                "chunk_index": i,
                "token_count": chunk.get('token_count', 0),
                "source_document_id": document_id,
                "space_id": space_id,
                "source_file": source_file,
                "extraction_method": metadata.get('extraction_method', 'unknown'),
                "total_chunks": len(chunks),
            }

            # Add chapter info if available
            if 'chapters' in metadata and metadata['chapters']:
                vector_metadata['has_chapters'] = True
                vector_metadata['chapter_count'] = len(metadata['chapters'])

            vector = {
                "id": vector_id,
                "values": chunk['embedding'],
                "metadata": vector_metadata
            }
            vectors.append(vector)

        # Upsert in batches
        batch_size = 100
        total_upserted = 0

        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            index.upsert(vectors=batch)
            total_upserted += len(batch)
            logger.info(f"   Upserted batch {i//batch_size + 1}/{(len(vectors) + batch_size - 1)//batch_size}")

        logger.info(f"✅ Upsert complete: {total_upserted} vectors")

        # Get index stats
        stats = index.describe_index_stats()
        logger.info(f"   Index stats: {stats.total_vector_count} total vectors")

        return {
            "success": True,
            "document_id": document_id,
            "vectors_upserted": total_upserted,
            "index_name": PINECONE_INDEX_NAME,
            "index_stats": stats
        }

    except Exception as e:
        logger.error(f"❌ Pinecone upsert failed: {e}")
        raise


def generate_document_id(file_path_or_url: str) -> str:
    """Generate unique document ID from file path or URL."""
    return hashlib.md5(file_path_or_url.encode()).hexdigest()


def test_multimodal_pipeline(file_path_or_url: str):
    """
    Test the complete multimodal RAG pipeline.

    Args:
        file_path_or_url: Path to file or URL to process
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Testing Multimodal RAG Pipeline")
    logger.info(f"{'='*80}\n")
    logger.info(f"Source: {file_path_or_url}")
    logger.info(f"Target Index: {PINECONE_INDEX_NAME}")
    logger.info(f"Space ID: {SPACE_ID}\n")

    # Generate document ID
    document_id = generate_document_id(file_path_or_url)
    source_file = Path(file_path_or_url).name if not file_path_or_url.startswith('http') else file_path_or_url

    logger.info(f"Document ID: {document_id}\n")

    try:
        # Step 1: Extract with Mistral OCR
        extraction_result = extract_document_with_mistral(file_path_or_url)
        full_text = extraction_result.get('full_text', '')

        if not full_text:
            logger.error("❌ No text extracted from document")
            return

        logger.info(f"\n{'-'*80}\n")

        # Step 2: Chunk with Chonkie RecursiveChunker
        chunks = chunk_with_chonkie(full_text, chunk_size=512)

        if not chunks:
            logger.error("❌ No chunks generated")
            return

        logger.info(f"\n{'-'*80}\n")

        # Step 3: Embed with Jina CLIP-v2
        chunks_with_embeddings = embed_with_multimodal(chunks)

        logger.info(f"\n{'-'*80}\n")

        # Step 4: Upsert to Pinecone
        upsert_result = upsert_to_pinecone(
            chunks=chunks_with_embeddings,
            document_id=document_id,
            space_id=SPACE_ID,
            source_file=source_file,
            metadata=extraction_result
        )

        # Success summary
        logger.info(f"\n{'='*80}")
        logger.info(f"🎉 TEST COMPLETE - SUCCESS!")
        logger.info(f"{'='*80}\n")

        logger.info(f"Summary:")
        logger.info(f"  Document ID: {document_id}")
        logger.info(f"  Source: {source_file}")
        logger.info(f"  Extraction: {extraction_result.get('extraction_method')}")
        logger.info(f"  Pages: {extraction_result.get('total_pages')}")
        logger.info(f"  Chunks: {len(chunks)}")
        logger.info(f"  Embedding dimension: 1024 (Jina CLIP-v2)")
        logger.info(f"  Vectors upserted: {upsert_result['vectors_upserted']}")
        logger.info(f"  Pinecone index: {PINECONE_INDEX_NAME}")
        logger.info(f"  Total vectors in index: {upsert_result['index_stats'].total_vector_count}")

        logger.info(f"\n✅ Ready for querying with multimodal embeddings!\n")

        # Save results to file
        output_file = f"test_results_{document_id}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            result_data = {
                "document_id": document_id,
                "source": file_path_or_url,
                "extraction": {
                    "method": extraction_result.get('extraction_method'),
                    "pages": extraction_result.get('total_pages'),
                    "chapters": len(extraction_result.get('chapters', [])),
                    "images": len(extraction_result.get('images', [])),
                },
                "chunking": {
                    "total_chunks": len(chunks),
                    "avg_tokens": sum(c['token_count'] for c in chunks) / len(chunks),
                },
                "embedding": {
                    "model": "jina-clip-v2",
                    "dimension": 1024,
                },
                "pinecone": {
                    "index_name": PINECONE_INDEX_NAME,
                    "vectors_upserted": upsert_result['vectors_upserted'],
                }
            }
            json.dump(result_data, f, indent=2)

        logger.info(f"💾 Results saved to: {output_file}\n")

    except Exception as e:
        logger.error(f"\n{'='*80}")
        logger.error(f"❌ TEST FAILED")
        logger.error(f"{'='*80}\n")
        logger.exception(f"Error: {e}")
        raise


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("\n❌ Error: No file or URL provided")
        print("\nUsage: python test_multimodal_pipeline.py <file_path_or_url>")
        print("\nExamples:")
        print("  python test_multimodal_pipeline.py document.pdf")
        print("  python test_multimodal_pipeline.py https://example.com/document.pdf")
        sys.exit(1)

    file_path_or_url = sys.argv[1]

    # Validate file exists if local path
    if not file_path_or_url.startswith(('http://', 'https://')):
        if not os.path.exists(file_path_or_url):
            print(f"\n❌ Error: File not found: {file_path_or_url}")
            sys.exit(1)

    # Check required environment variables
    if not PINECONE_API_KEY:
        print("\n❌ Error: PINECONE_API_KEY not found in environment")
        sys.exit(1)

    if not JINA_API_KEY:
        print("\n❌ Error: JINA_API_KEY not found in environment")
        sys.exit(1)

    # Run test
    test_multimodal_pipeline(file_path_or_url)


if __name__ == "__main__":
    main()
