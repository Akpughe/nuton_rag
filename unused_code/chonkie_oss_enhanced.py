"""
Enhanced Chonkie OSS Client with Full Metadata
Production-ready for Pinecone vector DB integration.

Features:
- Full metadata extraction (pages, chapters, headings)
- Pinecone-compatible output format
- Smart chapter/section detection
- Font-based heading detection
- Structure-aware chunking
"""

import os
import time
import logging
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import tiktoken
from openai import OpenAI

# Chonkie OSS imports
from chonkie import RecursiveChunker, TokenChunker, SemanticChunker, SentenceChunker
from chonkie.tokenizer import AutoTokenizer

# Our metadata extractor
from pdf_metadata_extractor import PDFMetadataExtractor, extract_pdf_with_metadata

# Mistral OCR integration
try:
    from mistral_ocr_extractor import MistralOCRExtractor, MistralOCRConfig
    MISTRAL_OCR_AVAILABLE = True
except ImportError:
    MISTRAL_OCR_AVAILABLE = False
    logging.warning("Mistral OCR not available. Install with: pip install mistralai")

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

logging.basicConfig(level=logging.INFO)


def _convert_to_markdown(text: str, pdf_metadata: Optional[Dict[str, Any]] = None) -> str:
    """
    Convert plain text to markdown format, preserving document structure.

    This helps with RAG retrieval by:
    - Adding markdown headers for chapters/sections (REPLACING duplicates)
    - Preserving lists and structure
    - Making content more semantically organized

    Args:
        text: Plain text to convert
        pdf_metadata: Optional PDF metadata with chapters/headings

    Returns:
        Text in markdown format
    """
    if not pdf_metadata:
        return text

    # Get chapters and headings
    chapters = pdf_metadata.get('chapters', [])

    if not chapters:
        return text

    # Sort chapters by position (reverse for replacement without index issues)
    chapters_sorted = sorted(chapters, key=lambda x: x.get('position', 0), reverse=True)

    # Build markdown version with headers
    markdown_text = text

    # Replace chapter titles with markdown headers
    for chapter in chapters_sorted:
        title = chapter.get('title', '')
        position = chapter.get('position', 0)
        level = chapter.get('level', 1)

        if not title or position >= len(markdown_text):
            continue

        # Create markdown header (# for level 1, ## for level 2, etc.)
        # Clean the title
        title_clean = title.replace('\n', ' ').strip()

        if not title_clean:
            continue

        # Find and replace the title with markdown version
        # Look for the title in the text
        if title_clean in markdown_text:
            # Replace with markdown header (WITHOUT duplicating the title)
            markdown_header = '#' * level + ' ' + title_clean

            # Replace the plain title with markdown version
            markdown_text = markdown_text.replace(title_clean, markdown_header, 1)

            logging.debug(f"Replaced '{title_clean}' with markdown header at level {level}")

    logging.info(f"Converted to markdown with {len(chapters)} headers")
    return markdown_text


def chunk_document_with_metadata(
    file_path: Optional[str] = None,
    text: Optional[str] = None,
    chunk_size: int = 512,
    overlap_tokens: int = 80,
    tokenizer: str = "cl100k_base",
    chunker_type: str = "recursive",
    # Format options
    recipe: str = "markdown",  # markdown, plain, code
    preserve_formatting: bool = True,
    # Metadata options
    extract_metadata: bool = True,
    detect_chapters: bool = True,
    detect_fonts: bool = True,
    detect_structure: bool = True,
    # Mistral OCR options
    use_mistral_ocr: bool = True,  # Use Mistral OCR as primary extraction method
    mistral_enhance_metadata: bool = True,  # Use LLM to enhance metadata
    mistral_fallback_to_legacy: bool = True,  # Fall back to legacy on error
    # Pinecone options
    pinecone_format: bool = False,
    namespace: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Chunk a document with full metadata extraction.

    Args:
        file_path: Path to file to chunk (PDF, PPTX, DOCX, images) or URL
        text: Text to chunk directly
        chunk_size: Target chunk size in tokens
        overlap_tokens: Overlap between chunks
        tokenizer: Tokenizer to use
        chunker_type: Type of chunker ('recursive', 'token', 'semantic', 'sentence')
        recipe: Output format ('markdown', 'plain', 'code') - markdown preserves structure
        preserve_formatting: Keep formatting like headers, lists, code blocks
        extract_metadata: Extract full metadata (pages, chapters, etc.)
        detect_chapters: Detect chapters and sections
        detect_fonts: Use font info for heading detection
        detect_structure: Detect document structure
        use_mistral_ocr: Use Mistral OCR as primary extraction (supports PDF/PPTX/DOCX/images/URLs)
        mistral_enhance_metadata: Use Mistral LLM to enhance metadata extraction
        mistral_fallback_to_legacy: Fall back to legacy extraction if Mistral fails
        pinecone_format: Format output for Pinecone
        namespace: Pinecone namespace (optional)
        **kwargs: Additional chunker arguments

    Returns:
        Dict with chunks and metadata
    """
    start_time = time.time()

    if not file_path and not text:
        raise ValueError("Either file_path or text must be provided")

    # Initialize result
    result = {
        'source': file_path or 'direct_text',
        'chunks': [],
        'metadata': {},
        'stats': {}
    }

    # Step 1: Extract metadata from document
    pdf_metadata = None

    # Try Mistral OCR first if enabled and available
    if file_path and use_mistral_ocr and MISTRAL_OCR_AVAILABLE and extract_metadata:
        logging.info(f"Attempting Mistral OCR extraction: {file_path}")

        try:
            # Configure Mistral OCR
            mistral_config = MistralOCRConfig(
                enhance_metadata_with_llm=mistral_enhance_metadata,
                fallback_method="legacy" if mistral_fallback_to_legacy else None,
                include_images=True,
                include_image_base64=True,
            )

            # Initialize extractor
            mistral_extractor = MistralOCRExtractor(config=mistral_config)

            # Extract document
            mistral_result = mistral_extractor.process_document(file_path)

            # Convert Mistral result to pdf_metadata format
            pdf_metadata = mistral_result
            text = mistral_result['full_text']

            # Store metadata
            result['metadata'] = {
                'file_name': mistral_result['file_name'],
                'total_pages': mistral_result['total_pages'],
                'chapters': mistral_result.get('chapters', []),
                'structure': mistral_result.get('structure', {}),
                'quality_score': mistral_result.get('metadata_quality', {}),
                'has_chapters': len(mistral_result.get('chapters', [])) > 0,
                'has_headings': len(mistral_result.get('headings', [])) > 0,
                'has_images': len(mistral_result.get('images', [])) > 0,
                'image_count': len(mistral_result.get('images', [])),
                'extraction_method': mistral_result.get('extraction_method', 'mistral_ocr'),
                'extraction_time_ms': mistral_result.get('extraction_time_ms', 0),
            }

            logging.info(f"✅ Mistral OCR extraction successful: "
                        f"{len(mistral_result.get('chapters', []))} chapters, "
                        f"{len(mistral_result.get('headings', []))} headings, "
                        f"{len(mistral_result.get('images', []))} images")

        except Exception as e:
            logging.warning(f"Mistral OCR extraction failed: {e}")

            # Fall back to legacy extraction if configured
            if mistral_fallback_to_legacy:
                logging.info("Falling back to legacy PDF extraction...")
                pdf_metadata = None  # Reset to trigger legacy extraction below
            else:
                raise

    # Legacy extraction (fallback or if Mistral OCR disabled/unavailable)
    if pdf_metadata is None and file_path and file_path.lower().endswith('.pdf') and extract_metadata:
        logging.info(f"Using legacy PDF extraction: {file_path}")

        try:
            pdf_metadata = extract_pdf_with_metadata(
                file_path,
                detect_chapters=detect_chapters,
                detect_fonts=detect_fonts,
                detect_structure=detect_structure
            )

            # Use extracted text
            text = pdf_metadata['full_text']

            # Store metadata
            result['metadata'] = {
                'file_name': pdf_metadata['file_name'],
                'total_pages': pdf_metadata['total_pages'],
                'chapters': pdf_metadata.get('chapters', []),
                'structure': pdf_metadata.get('structure', {}),
                'quality_score': pdf_metadata.get('metadata_quality', {}),
                'has_chapters': len(pdf_metadata.get('chapters', [])) > 0,
                'has_headings': len(pdf_metadata.get('headings', [])) > 0,
                'extraction_method': 'legacy_pdf',
            }

            logging.info(f"Legacy metadata extracted: {len(pdf_metadata.get('chapters', []))} chapters, "
                        f"{len(pdf_metadata.get('headings', []))} headings")

        except Exception as e:
            logging.warning(f"Could not extract PDF metadata: {e}")
            # Fall back to simple text extraction
            import PyPDF2
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                text = "\n\n".join(page.extract_text() for page in reader.pages)

    elif file_path and not text:
        # Read text file
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

    if not text:
        raise ValueError("No text to chunk")

    # Step 2: Convert to markdown format if requested
    if recipe == "markdown" and preserve_formatting:
        logging.info("Converting text to markdown format...")
        # Convert plain text to markdown-like structure
        # This helps preserve document structure for better RAG retrieval
        text = _convert_to_markdown(text, pdf_metadata)

    # Step 3: Chunk the text
    logging.info(f"Chunking with {chunker_type} chunker, size={chunk_size}, overlap={overlap_tokens}, recipe={recipe}")

    token_counter = AutoTokenizer(tokenizer)

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
    elif chunker_type == "semantic":
        chunker = SemanticChunker(
            tokenizer=token_counter,
            chunk_size=chunk_size,
            **kwargs
        )
    elif chunker_type == "sentence":
        chunker = SentenceChunker(
            tokenizer=token_counter,
            chunk_size=chunk_size,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown chunker_type: {chunker_type}. Must be 'recursive', 'token', 'semantic', or 'sentence'")

    chunk_objects = chunker.chunk(text)

    # Convert to dicts
    chunks = []
    for chunk_obj in chunk_objects:
        chunk_dict = {
            "text": chunk_obj.text,
            "start_index": chunk_obj.start_index,
            "end_index": chunk_obj.end_index,
            "token_count": chunk_obj.token_count,
        }
        chunks.append(chunk_dict)

    logging.info(f"Created {len(chunks)} chunks")

    # Step 3: Map chunks to metadata
    if pdf_metadata:
        logging.info("Mapping chunks to metadata...")
        extractor = PDFMetadataExtractor()
        chunks = extractor.map_chunks_to_metadata(chunks, pdf_metadata)
        logging.info("Metadata mapping complete")

    # Step 4: Format for Pinecone if requested
    if pinecone_format:
        chunks = format_for_pinecone(chunks, result['metadata'], namespace)

    result['chunks'] = chunks

    # Calculate stats
    elapsed = time.time() - start_time
    result['stats'] = {
        'total_chunks': len(chunks),
        'total_tokens': sum(c.get('token_count', 0) for c in chunks),
        'avg_tokens_per_chunk': sum(c.get('token_count', 0) for c in chunks) / len(chunks) if chunks else 0,
        'processing_time_ms': elapsed * 1000,
        'processing_time_s': elapsed,
        'chunker_type': chunker_type,
        'chunk_size': chunk_size,
        'overlap': overlap_tokens,
        'recipe': recipe,  # markdown, plain, code
        'preserve_formatting': preserve_formatting,
    }

    logging.info(f"Chunking complete in {elapsed*1000:.2f}ms: "
                f"{len(chunks)} chunks, "
                f"{result['stats']['total_tokens']} tokens")

    return result


def format_for_pinecone(
    chunks: List[Dict[str, Any]],
    document_metadata: Dict[str, Any],
    namespace: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Format chunks for Pinecone vector database.

    Pinecone structure:
    {
        'id': 'unique_id',
        'values': [embedding_vector],  # Added later
        'metadata': {
            'text': 'chunk_text',
            'page': 1,
            'chapter': 'Introduction',
            ...
        }
    }

    Args:
        chunks: List of chunks with metadata
        document_metadata: Document-level metadata
        namespace: Pinecone namespace

    Returns:
        List of Pinecone-formatted chunks
    """
    pinecone_chunks = []

    file_name = document_metadata.get('file_name', 'unknown')

    for i, chunk in enumerate(chunks):
        # Generate unique ID
        chunk_id = f"{file_name}_chunk_{i}"

        # Build Pinecone metadata
        # Note: Pinecone metadata values must be strings, numbers, booleans, or lists of strings
        metadata = {
            # Required fields
            'text': chunk['text'][:40000],  # Pinecone limit
            'chunk_index': i,
            'token_count': chunk.get('token_count', 0),

            # Source info
            'source_file': file_name,
            'char_start': chunk.get('start_index', 0),
            'char_end': chunk.get('end_index', 0),

            # Page info
            'pages': chunk.get('pages', []),
            'page_start': chunk.get('pages', [1])[0] if chunk.get('pages') else 1,
            'page_end': chunk.get('pages', [1])[-1] if chunk.get('pages') else 1,

            # Chapter/section info
            'chapter': chunk.get('chapter', ''),
            'chapter_number': chunk.get('chapter_number', ''),
            'section_level': chunk.get('section_level', 0),

            # Heading info
            'heading': chunk.get('heading', ''),
            'heading_level': chunk.get('heading_level', 0),

            # Position
            'position_in_doc': round(chunk.get('position_in_doc', 0), 4),

            # Content flags
            'has_tables': chunk.get('has_tables', False),
            'has_images': chunk.get('has_images', False),
            'figure_refs': chunk.get('figure_refs', []),
            'table_refs': chunk.get('table_refs', []),
        }

        # Add namespace if provided
        pinecone_chunk = {
            'id': chunk_id,
            'metadata': metadata
        }

        if namespace:
            pinecone_chunk['namespace'] = namespace

        # Note: 'values' (embedding) will be added when you embed
        # pinecone_chunk['values'] = embedding

        pinecone_chunks.append(pinecone_chunk)

    return pinecone_chunks


def embed_chunks_with_metadata(
    chunks: List[Dict[str, Any]],
    embedding_model: str = "text-embedding-3-small",
    batch_size: int = 64,
    pinecone_format: bool = False
) -> List[Dict[str, Any]]:
    """
    Embed chunks with OpenAI, preserving metadata.

    Args:
        chunks: List of chunks (with or without Pinecone format)
        embedding_model: OpenAI embedding model
        batch_size: Batch size for embedding
        pinecone_format: Whether chunks are in Pinecone format

    Returns:
        Chunks with embeddings added
    """
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not found in environment")

    logging.info(f"Embedding {len(chunks)} chunks with {embedding_model}")

    client = OpenAI(api_key=OPENAI_API_KEY)
    results = []

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]

        # Extract texts
        if pinecone_format:
            batch_texts = [c['metadata']['text'] for c in batch]
        else:
            batch_texts = [c['text'] for c in batch]

        try:
            logging.info(f"Embedding batch {i//batch_size + 1}/{(len(chunks) + batch_size - 1)//batch_size}")
            response = client.embeddings.create(
                model=embedding_model,
                input=batch_texts
            )

            # Add embeddings to chunks
            for chunk, embedding_data in zip(batch, response.data):
                chunk_copy = chunk.copy()

                if pinecone_format:
                    # Add to 'values' field for Pinecone
                    chunk_copy['values'] = embedding_data.embedding
                else:
                    # Add to 'embedding' field
                    chunk_copy['embedding'] = embedding_data.embedding

                results.append(chunk_copy)

            logging.info(f"Successfully embedded batch {i//batch_size + 1}")

        except Exception as e:
            logging.error(f"Error embedding batch: {e}")
            raise

    logging.info(f"Successfully embedded {len(results)} chunks")
    return results


def embed_query_with_metadata(
    query: str,
    embedding_model: str = "text-embedding-3-small"
) -> List[float]:
    """
    Embed a query for retrieval.

    Args:
        query: Query string
        embedding_model: OpenAI embedding model

    Returns:
        Embedding vector
    """
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not found in environment")

    client = OpenAI(api_key=OPENAI_API_KEY)

    response = client.embeddings.create(
        model=embedding_model,
        input=[query]
    )

    return response.data[0].embedding


# Example usage
if __name__ == "__main__":
    import json

    # Test with a PDF
    pdf_path = "/Users/davak/Documents/_study/Artificial Intelligence_An Overview.pdf"  # Replace with your PDF

    if not os.path.exists(pdf_path):
        print(f"❌ PDF not found: {pdf_path}")
        print("Please provide a PDF path to test")
    else:
        print(f"\n{'='*80}")
        print("Testing Enhanced Chonkie OSS with Full Metadata")
        print(f"{'='*80}\n")

        # Chunk with full metadata
        result = chunk_document_with_metadata(
            file_path=pdf_path,
            chunk_size=512,
            overlap_tokens=80,
            chunker_type="recursive",
            extract_metadata=True,
            detect_chapters=True,
            detect_fonts=True,
            detect_structure=True,
            pinecone_format=True,  # Format for Pinecone
            namespace="test_docs"
        )

        print(f"\n✅ Results:")
        print(f"   Total chunks: {result['stats']['total_chunks']}")
        print(f"   Processing time: {result['stats']['processing_time_ms']:.2f}ms")
        print(f"   Chapters detected: {len(result['metadata'].get('chapters', []))}")
        print(f"   Quality score: {result['metadata'].get('quality_score', {}).get('overall_quality', 0)}/100")

        if result['chunks']:
            print(f"\n📄 First chunk metadata:")
            chunk = result['chunks'][0]
            print(json.dumps(chunk.get('metadata', {}), indent=2))

        # Save results
        output_file = "test_enhanced_chunks.json"
        with open(output_file, 'w') as f:
            # Don't save full text in JSON to keep it readable
            save_result = result.copy()
            for chunk in save_result['chunks']:
                if 'metadata' in chunk and 'text' in chunk['metadata']:
                    chunk['metadata']['text'] = chunk['metadata']['text'][:200] + "..."

            json.dump(save_result, f, indent=2)

        print(f"\n💾 Saved to: {output_file}")
