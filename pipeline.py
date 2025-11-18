from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any, Optional
import os
import shutil
import logging
import tempfile
import json
import time
from functools import lru_cache

from chonkie_client import chunk_document, embed_chunks, embed_chunks_v2, embed_query, embed_query_v2, embed_chunks_multimodal, embed_query_multimodal
from pinecone_client import upsert_vectors, upsert_image_vectors, hybrid_search, hybrid_search_parallel, rerank_results, hybrid_search_document_aware, rerank_results_document_aware
from supabase_client import insert_pdf_record, insert_yts_record, get_documents_in_space, check_document_type, upsert_generated_content_notes
from groq_client import generate_answer, generate_answer_document_aware
import openai_client
from services.wetrocloud_youtube import WetroCloudYouTubeService
from services.youtube_transcript_service import YouTubeTranscriptService
from services.ytdlp_transcript_service import YTDLPTranscriptService
from flashcard_process import generate_flashcards, regenerate_flashcards
from pydantic import BaseModel
from typing import Optional, List
from quiz_process import generate_quiz, regenerate_quiz
from hybrid_pdf_processor import extract_and_chunk_pdf_async
from diagram_explainer import explain_diagrams_batch
from mistral_ocr_extractor import MistralOCRExtractor, MistralOCRConfig
from chonkie import RecursiveChunker
from chonkie.tokenizer import AutoTokenizer

from prompts import main_prompt, general_knowledge_prompt, simple_general_knowledge_prompt, no_docs_in_space_prompt, no_relevant_in_scope_prompt, additional_space_only_prompt
from intelligent_enrichment import create_enriched_system_prompt
from enrichment_examples import create_few_shot_enhanced_prompt
from enhanced_prompts import get_domain_from_context
from websearch_client import analyze_and_generate_queries, perform_contextual_websearch_async, synthesize_rag_and_web_results
from services.google_drive_service import GoogleDriveService


app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # List of allowed origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

NOTES_DIR = "note"
os.makedirs(NOTES_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FlashcardRequest(BaseModel):
    document_id: str
    space_id: Optional[str] = None
    user_id: str
    num_questions: Optional[int] = None
    acl_tags: Optional[List[str]] = None


class QuizRequest(BaseModel):
    document_id: str
    space_id: Optional[str] = None
    user_id: str
    question_type: str = "both"
    num_questions: int = 30
    acl_tags: Optional[str] = None
    rerank_top_n: int = 50
    use_openai_embeddings: bool = False  # Deprecated: Always uses multimodal embeddings (1024 dims) to match Pinecone index
    set_id: int = 1
    title: Optional[str] = None
    description: Optional[str] = None


class DriveFilesRequest(BaseModel):
    access_token: str
    refresh_token: str
    folder_id: Optional[str] = None
    file_types: Optional[List[str]] = ["pdf", "doc", "docx"]
    max_results: int = 100


class DriveImportRequest(BaseModel):
    file_ids: List[str]
    space_id: str
    access_token: str
    refresh_token: str


def flatten_chunks(chunks):
    """Ensure chunks is a flat list of dicts."""
    if not chunks:
        return []
    # If the first element is a list, flatten one level
    if isinstance(chunks[0], list):
        flat = []
        for item in chunks:
            if isinstance(item, list):
                flat.extend(item)
            else:
                flat.append(item)
        return flat
    return chunks


def _validate_and_truncate_history(
    history: Optional[List[Dict[str, str]]],
    max_messages: int = 10,
    max_tokens: int = 2000
) -> Optional[List[Dict[str, str]]]:
    """
    Validate and truncate conversation history for speed.

    Args:
        history: Raw conversation history
        max_messages: Maximum number of messages to keep
        max_tokens: Maximum estimated tokens (using char/4 approximation)

    Returns:
        Validated history or None if invalid/empty
    """
    if not history:
        return None

    # Validate structure
    validated = []
    for msg in history:
        if isinstance(msg, dict) and "role" in msg and "content" in msg:
            if msg["role"] in ["user", "assistant"] and isinstance(msg["content"], str):
                validated.append(msg)

    if not validated:
        return None

    # Truncate to last N messages (keep conversation recent)
    validated = validated[-max_messages:]

    # Fast token estimation (4 chars ≈ 1 token)
    total_chars = sum(len(msg["content"]) for msg in validated)
    estimated_tokens = total_chars // 4

    # If over budget, remove oldest messages
    while estimated_tokens > max_tokens and len(validated) > 2:
        validated.pop(0)  # Remove oldest
        total_chars = sum(len(msg["content"]) for msg in validated)
        estimated_tokens = total_chars // 4

    logging.info(f"Conversation history: {len(validated)} messages, ~{estimated_tokens} tokens")
    return validated


def process_document(
    file_path: str,
    metadata: Dict[str, Any],
    chunk_size: int = 512,
    overlap_tokens: int = 80,
    tokenizer: str = "gpt2",
    recipe: str = "markdown",
    lang: str = "en",
    min_characters_per_chunk: int = 12,
    embedding_model: str = "text-embedding-ada-002",
    batch_size: int = 64
) -> str:
    """
    Ingests a document: chunk, embed, insert into Supabase, upsert to Pinecone.
    Returns the document_id (Supabase id).
    """
    # 1. Chunk document
    chunks = chunk_document(
        file_path=file_path,
        chunk_size=chunk_size,
        overlap_tokens=overlap_tokens,
        tokenizer=tokenizer,
        recipe=recipe,
        lang=lang,
        min_characters_per_chunk=min_characters_per_chunk
    )
    chunks = flatten_chunks(chunks)
    # print(chunks)
    logging.info(f"Chunks type: {type(chunks)}, example: {chunks[:1]}")
    
    # 2. Embed chunks
    embeddings = embed_chunks(chunks, embedding_model=embedding_model, batch_size=batch_size)
    logging.info(f"Embeddings type: {type(embeddings)}, example: {embeddings[:1]}")
    
    # Check if embedding service returned an error
    if embeddings and len(embeddings) > 0 and isinstance(embeddings[0], dict):
        if 'message' in embeddings[0] and 'status' in embeddings[0]:
            error_msg = f"Embedding service error: {embeddings[0]['message']}"
            logging.error(error_msg)
            raise ValueError(error_msg)
    
    # 3. Prepare fields for pdfs table
    file_name = os.path.basename(file_path)
    file_type = os.path.splitext(file_path)[1][1:] or "unknown"
    # print('metadata', metadata)
    space_id = metadata.get("space_id")
    extracted_text = "No text extracted yet"
    
    # Use the file_path from metadata if provided, otherwise use the local file path
    db_file_path = metadata.get("file_path", file_path)
    
    pdfs_row = {
        "space_id": space_id,
        "file_path": db_file_path,  # This can now be a URL
        "extracted_text": extracted_text,
        "file_type": file_type,
        "file_name": metadata.get("filename", file_name)
    }
    # 4. Insert into Supabase and get document_id
    document_id = insert_pdf_record(pdfs_row)
    # 5. Upsert to Pinecone
    try:
        print('upserting vectors')
        # Use filename from metadata if provided, otherwise use file_name from path
        source_filename = metadata.get("filename", file_name)
        logging.info(f"Upserting with source file: {source_filename}")
        upsert_vectors(
            doc_id=document_id, 
            space_id=space_id, 
            embeddings=embeddings, 
            chunks=chunks,
            source_file=source_filename
        )
    except Exception as e:
        logging.error(f"Error in upsert_vectors: {e}")
        # If it's an embedding error, provide a more specific message
        if "Embedding service returned an error" in str(e):
            raise ValueError(f"Failed to embed document: {str(e)}")
        raise
    return document_id


async def process_document_with_openai(
    file_path: str,
    metadata: Dict[str, Any],
    chunk_size: int = 512,
    overlap_tokens: int = 80,
    tokenizer: str = "gpt2",
    recipe: str = "markdown",
    lang: str = "en",
    min_characters_per_chunk: int = 12,
    embedding_model: str = "text-embedding-ada-002",
    batch_size: int = 64,
    enable_chapter_detection: bool = True,
    quality_threshold: float = 0.65
) -> str:
    """
    Ingests a document using OpenAI for embeddings: chunk, embed with OpenAI, insert into Supabase, upsert to Pinecone.

    Now includes:
    - Chapter detection for better context
    - SELECTIVE quality correction (only corrects chunks below threshold)

    OPTIMIZATION: quality_threshold parameter saves 60-80% on LLM costs!

    Args:
        file_path: Path to the document file
        metadata: Document metadata including space_id, filename, etc.
        chunk_size: Target chunk size in tokens
        overlap_tokens: Token overlap between chunks
        tokenizer: Tokenizer to use (gpt2, gpt-4, etc.)
        recipe: Chunking recipe (markdown, text, etc.)
        lang: Language code
        min_characters_per_chunk: Minimum characters per chunk
        embedding_model: OpenAI embedding model to use
        batch_size: Batch size for embedding
        enable_chapter_detection: Enable chapter detection
        quality_threshold: Only correct chunks with quality score < threshold (default: 0.65)
                          Lower = more selective, higher = more aggressive

    Returns:
        The document_id (Supabase id)
    """
    try:
        chunks = []
        extraction_method = "unknown"
        extraction_result = None  # Initialize to track Mistral OCR result

        # 1. PRIMARY: Try Mistral OCR extraction first (supports PDF, PPTX, DOCX, images, URLs)
        file_ext = os.path.splitext(file_path)[1].lower()
        supported_formats = ['.pdf', '.pptx', '.docx', '.png', '.jpg', '.jpeg', '.webp', '.avif']

        if file_ext in supported_formats or file_path.startswith(('http://', 'https://')):
            try:
                logging.info(f"📄 Extracting document with Mistral OCR: {os.path.basename(file_path)}")

                # Configure Mistral OCR (matching test_multimodal_pipeline.py)
                mistral_config = MistralOCRConfig(
                    enhance_metadata_with_llm=True,
                    fallback_method="legacy",
                    include_images=True,
                    include_image_base64=True,  # Enable base64 for image embedding
                )

                # Initialize extractor and process document
                extractor = MistralOCRExtractor(config=mistral_config)
                extraction_result = extractor.process_document(file_path)

                # Get full text from extraction
                full_text = extraction_result.get('full_text', '')

                if not full_text:
                    raise ValueError("Mistral OCR returned no text")

                logging.info(f"✅ Mistral OCR extraction successful!")
                logging.info(f"   Method: {extraction_result.get('extraction_method')}")
                logging.info(f"   Pages: {extraction_result.get('total_pages')}")
                logging.info(f"   Text length: {len(full_text)} chars")
                extraction_method = extraction_result.get('extraction_method', 'mistral_ocr')

                # 2. Chunk with simple Chonkie RecursiveChunker (matching test_multimodal_pipeline.py)
                logging.info(f"✂️ Chunking with Chonkie RecursiveChunker (size={chunk_size})")

                # Initialize tokenizer
                chonkie_tokenizer = AutoTokenizer("cl100k_base")  # OpenAI tokenizer

                # Initialize RecursiveChunker
                chunker = RecursiveChunker(
                    tokenizer=chonkie_tokenizer,
                    chunk_size=chunk_size,
                    min_characters_per_chunk=min_characters_per_chunk
                )

                # Chunk the text
                chunk_objects = chunker.chunk(full_text)

                # Convert to dicts (matching test format)
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

                logging.info(f"✅ Chunking complete: {len(chunks)} chunks")
                logging.info(f"   Total tokens: {sum(c['token_count'] for c in chunks)}")
                logging.info(f"   Avg tokens/chunk: {sum(c['token_count'] for c in chunks) / len(chunks):.1f}")

            except Exception as e:
                logging.warning(f"Mistral OCR extraction failed: {e}")
                chunks = []

        # 3. FALLBACK: Use DocChunker for PDFs if Mistral OCR failed
        if not chunks and file_ext == '.pdf':
            try:
                logging.info(f"📄 Fallback: Using DocChunker with quality correction (threshold={quality_threshold})")

                # Use async hybrid processor for PDFs (with SELECTIVE LLM correction!)
                chunks = await extract_and_chunk_pdf_async(
                    pdf_path=file_path,
                    chunk_size=chunk_size,
                    overlap_tokens=overlap_tokens,
                    tokenizer=tokenizer,
                    recipe=recipe,
                    lang=lang,
                    min_characters_per_chunk=min_characters_per_chunk,
                    enable_llm_correction=True,
                    quality_threshold=quality_threshold
                )

                chunks = flatten_chunks(chunks)
                extraction_method = "docchunker_fallback"
                logging.info(f"✅ DocChunker generated {len(chunks)} chunks")

            except Exception as e:
                logging.warning(f"DocChunker also failed: {e}, trying basic chunking")
                chunks = []

        # 4. FINAL FALLBACK: Basic chunking for any other format or if all else failed
        if not chunks:
            logging.info(f"📄 Final fallback: Basic chunking for {file_path}")
            chunks = chunk_document(
                file_path=file_path,
                chunk_size=chunk_size,
                overlap_tokens=overlap_tokens,
                tokenizer=tokenizer,
                recipe=recipe,
                lang=lang,
                min_characters_per_chunk=min_characters_per_chunk
            )
            chunks = flatten_chunks(chunks)
            extraction_method = "basic_chunking"
        # print('chunks', chunks)
        
        if not chunks:
            logging.error(f"No chunks generated from document: {file_path}")
            raise ValueError("Document chunking returned no chunks. The document may be empty or in an unsupported format.")
            
        logging.info(f"Generated {len(chunks)} chunks from document")
        
        # 2. Embed chunks with multimodal embeddings (Jina CLIP-v2, 1024 dims)
        logging.info(f"Embedding chunks with Jina CLIP-v2 multimodal model (1024 dims)")
        embeddings = embed_chunks_multimodal(chunks, batch_size=batch_size)
        
        # Check if embedding service returned an error
        if embeddings and isinstance(embeddings[0], dict) and "message" in embeddings[0] and "status" in embeddings[0]:
            error_msg = f"Multimodal embedding error: {embeddings[0]['message']}"
            logging.error(error_msg)
            raise ValueError(error_msg)

        logging.info(f"Successfully embedded {len(embeddings)} chunks with Jina CLIP-v2 (1024 dims)")

        # 3. Prepare fields for pdfs table
        file_name = os.path.basename(file_path)
        file_type = os.path.splitext(file_path)[1][1:] or "unknown"
        
        # Safely access space_id
        if not isinstance(metadata, dict):
            logging.warning(f"Metadata is not a dictionary: {type(metadata)}")
            metadata = {"space_id": "default"}
            
        space_id = metadata.get("space_id")
        if not space_id:
            logging.warning("No space_id found in metadata, using default")
            space_id = "default"
            
        logging.info(f"Using space_id: {space_id}")
        
        extracted_text = "No text extracted yet"
        
        # Use the file_path from metadata if provided, otherwise use the local file path
        db_file_path = metadata.get("file_path", file_path)
        
        pdfs_row = {
            "space_id": space_id,
            "file_path": db_file_path,  # This can now be a URL
            "extracted_text": extracted_text,
            "file_type": file_type,
            "file_name": metadata.get("filename", file_name)
        }
        
        # 4. Insert into Supabase and get document_id
        document_id = insert_pdf_record(pdfs_row)
        logging.info(f"Inserted document into Supabase with ID: {document_id}")

        # 4.5. Process images if extracted by Mistral OCR (AFTER document_id is created)
        images = []
        image_embeddings = []
        images_with_data = []
        if extraction_result is not None and extraction_result.get('images'):
            images = extraction_result.get('images', [])
            logging.info(f"📷 Found {len(images)} images in document")

            # Filter images that have base64 data
            images_with_data = [img for img in images if img.get('image_base64')]

            if images_with_data:
                logging.info(f"🧠 Embedding {len(images_with_data)} images with Jina CLIP-v2")

                try:
                    from multimodal_embeddings import MultimodalEmbedder
                    from s3_image_storage import S3ImageStorage

                    # Step 1: Upload images to S3 and get URLs
                    logging.info(f"📤 Uploading {len(images_with_data)} images to S3...")
                    s3_storage = S3ImageStorage()
                    image_urls = s3_storage.upload_images_batch(
                        images=images_with_data,
                        document_id=document_id,
                        space_id=space_id
                    )

                    if not image_urls:
                        logging.warning("No images were uploaded to S3")
                        image_embeddings = []
                    else:
                        logging.info(f"✅ Uploaded {len(image_urls)} images to S3")

                        # Step 2: Embed images using URLs
                        embedder = MultimodalEmbedder(model="jina-clip-v2", batch_size=batch_size)
                        image_embeddings = embedder.embed_images(image_urls, normalize=True)
                        logging.info(f"✅ Image embedding complete: {len(image_embeddings)} embeddings (1024 dims)")

                        # Step 3: Add URLs to image metadata for later retrieval
                        for img, url in zip(images_with_data, image_urls):
                            img['image_url'] = url

                except Exception as e:
                    logging.error(f"❌ Image embedding failed: {e}")
                    # Continue without images rather than failing the whole pipeline
                    image_embeddings = []
            else:
                logging.info("No images with base64 data found")

        # 5. Upsert to Pinecone
        try:
            # Use filename from metadata if provided, otherwise use file_name from path
            source_filename = metadata.get("filename", file_name)
            logging.info(f"Upserting {len(embeddings)} vectors to Pinecone with source file: {source_filename}")
            upsert_vectors(
                doc_id=document_id, 
                space_id=space_id, 
                embeddings=embeddings, 
                chunks=chunks,
                source_file=source_filename
            )
            logging.info(f"Successfully upserted vectors to Pinecone for document: {document_id}")

            # 5.5. Upsert image vectors if we have any
            if image_embeddings and images_with_data:
                try:
                    logging.info(f"📤 Upserting {len(image_embeddings)} image vectors to Pinecone")
                    upsert_image_vectors(
                        doc_id=document_id,
                        space_id=space_id,
                        images=images_with_data,
                        embeddings=image_embeddings,
                        source_file=source_filename
                    )
                    logging.info(f"✅ Successfully upserted {len(image_embeddings)} image vectors")
                except Exception as e:
                    logging.error(f"❌ Error upserting image vectors: {e}")
                    # Continue without failing the whole pipeline
        except ValueError as e:
            logging.error(f"Error upserting vectors: {e}")
            raise ValueError(f"Failed to process document embeddings: {str(e)}")
            
        return document_id
        
    except Exception as e:
        logging.exception(f"Error processing document with OpenAI: {e}")
        raise


async def process_drive_files(file_ids: List[str], space_id: str, access_token: str, refresh_token: str) -> Dict[str, Any]:
    """
    Process multiple Google Drive files through the pipeline.
    
    Args:
        file_ids: List of Google Drive file IDs to process
        space_id: Space ID to associate with the documents
        access_token: User's Google access token
        refresh_token: User's Google refresh token
        
    Returns:
        Dictionary with processing results and any errors
    """
    results = {
        "processed_files": [],
        "errors": [],
        "updated_tokens": None
    }
    
    try:
        # Initialize Google Drive service
        drive_service = GoogleDriveService(access_token, refresh_token)
        
        # Get updated tokens after potential refresh
        updated_tokens = drive_service.get_updated_tokens()
        results["updated_tokens"] = updated_tokens
        
        for file_id in file_ids:
            try:
                logging.info(f"Processing Google Drive file: {file_id}")
                
                # Download file content and metadata
                file_content, file_metadata = drive_service.download_file(file_id)
                
                # Save to temporary file
                temp_file_path = drive_service.save_temp_file(file_content, file_metadata['name'])
                
                try:
                    # Prepare metadata for pipeline
                    pipeline_metadata = {
                        "filename": file_metadata['name'],
                        "space_id": space_id,
                        "file_path": file_metadata.get('webViewLink', ''),
                        "source": "google_drive",
                        "drive_file_id": file_id,
                        "file_size": file_metadata.get('size', 0),
                        "mime_type": file_metadata.get('mimeType', ''),
                        "created_time": file_metadata.get('createdTime', ''),
                        "modified_time": file_metadata.get('modifiedTime', '')
                    }
                    
                    # Process through existing pipeline
                    document_id = await process_document_with_openai(
                        file_path=temp_file_path,
                        metadata=pipeline_metadata
                    )
                    
                    results["processed_files"].append({
                        "file_id": file_id,
                        "document_id": document_id,
                        "filename": file_metadata['name'],
                        "status": "success"
                    })
                    
                    logging.info(f"Successfully processed Drive file {file_id} -> document {document_id}")
                    
                finally:
                    # Clean up temporary file
                    if os.path.exists(temp_file_path):
                        os.remove(temp_file_path)
                        logging.info(f"Cleaned up temporary file: {temp_file_path}")
                        
            except Exception as e:
                error_msg = f"Failed to process file {file_id}: {str(e)}"
                logging.error(error_msg)
                results["errors"].append({
                    "file_id": file_id,
                    "error": error_msg
                })
                
    except Exception as e:
        error_msg = f"Failed to initialize Google Drive service: {str(e)}"
        logging.error(error_msg)
        results["errors"].append({
            "general_error": error_msg
        })
    
    return results


# Add caching for query embeddings
@lru_cache(maxsize=100)
def get_query_embedding(query: str, use_openai: bool = True):
    """Cache query embeddings to avoid recalculating for repeated queries. Now uses multimodal embeddings (Jina CLIP-v2, 1024 dims)."""
    start_time = time.time()
    # Use multimodal embeddings (Jina CLIP-v2) for all queries
    result = embed_query_multimodal(query)
    logging.info(f"Query embedding (multimodal, 1024 dims) took {time.time() - start_time:.2f}s")
    return result


async def answer_query(
    query: str,
    document_id: Optional[str] = None,
    space_id: Optional[str] = None,
    acl_tags: Optional[List[str]] = None,
    rerank_top_n: int = 10,  # Reduced from 15 to 10 for better performance
    system_prompt: str = main_prompt,
    openai_model: str = "gpt-4o-mini",  # Fallback model for OpenAI when Groq fails
    use_openai_embeddings: bool = True,
    search_by_space_only: bool = False,
    max_context_chunks: int = 5,  # Limit context size for better performance
    allow_general_knowledge: bool = False,  # New parameter for allowing general knowledge supplementation
    enable_websearch: bool = False,  # New parameter for enabling contextual web search
    model: str = "meta-llama/llama-4-scout-17b-16e-instruct",  # User-selectable model parameter
    enrichment_mode: str = "simple",  # "simple" (default) or "advanced" for intelligent enrichment
    learning_style: Optional[str] = None,  # Learning style for personalized educational responses
    educational_mode: bool = False,  # Enable tutoring/educational approach
    conversation_history: Optional[List[Dict[str, str]]] = None,  # Conversation history for context continuity
    max_history_messages: int = 10,  # Maximum number of history messages to keep
    return_context: bool = False,  # Return context chunks for streaming endpoints
    include_diagrams: bool = True,  # Whether to include diagrams in response
    max_diagrams: int = 3  # Maximum number of diagrams to return
) -> Dict[str, Any]:
    """
    Optimized function to answer a user query using hybrid search, rerank, and LLM generation.
    Returns dict with answer, citations, and optionally diagrams.

    Args:
        query: The user's question.
        document_id: Filter results to this document ID.
        space_id: Filter results to this space ID.
        acl_tags: Optional list of ACL tags to filter by.
        rerank_top_n: Number of results to rerank.
        system_prompt: System prompt for the LLM (will be overridden if allow_general_knowledge is True).
        openai_model: Fallback model to use with OpenAI when Groq fails.
        use_openai_embeddings: Whether to use OpenAI directly for embeddings.
        search_by_space_only: If True, search based on space_id only, ignoring document_id.
        max_context_chunks: Maximum number of chunks to include in context for the LLM.
        allow_general_knowledge: If True, allows LLM to enrich answers with general knowledge even when documents are sufficient, providing additional insights and context.
        enable_websearch: If True, performs contextual web search to supplement RAG results.
        model: The model to use for generation. Auto-switches to GPT-4o when websearch is enabled.
        enrichment_mode: "simple" (default, classic enrichment) or "advanced" (intelligent domain-aware enrichment).
        learning_style: Learning style for personalized educational responses ("academic_focus", "deep_dive", "quick_practical", "exploratory_curious", "narrative_reader", "default", or None).
        educational_mode: If True, enables tutoring/educational approach with context-rich responses.
        conversation_history: Optional conversation history for context continuity.
        max_history_messages: Maximum number of history messages to keep.
        return_context: If True, includes context_chunks in the return dict for streaming endpoints.
        include_diagrams: If True, processes and returns diagrams found in search results.
        max_diagrams: Maximum number of diagrams to return (default: 3).
    """
    start_time = time.time()
    logging.info(f"Answering query: '{query}' for document {document_id if not search_by_space_only else 'None'} in space {space_id}, allow_general_knowledge: {allow_general_knowledge}, enable_websearch: {enable_websearch}, enrichment_mode: {enrichment_mode}, learning_style: {learning_style}, educational_mode: {educational_mode}")

    # Initialize context_chunks for optional return (used by streaming endpoints)
    context_chunks = []

    # Validate and truncate conversation history for speed
    validated_history = _validate_and_truncate_history(
        conversation_history,
        max_messages=max_history_messages,
        max_tokens=2000  # Keep history compact
    )

    # Auto-enable educational mode if learning style is specified
    if learning_style and not educational_mode:
        educational_mode = True
        logging.info(f"Auto-enabled educational mode due to learning style: {learning_style}")
    
    # Determine which LLM to use based on model parameter
    # Auto-switch to GPT-4o if websearch is enabled
    if enable_websearch:
        effective_model = "gpt-4o"
        use_openai = True
        # Smart auto-enable: If websearch is enabled, also enable general knowledge for richer synthesis
        # This allows the LLM to enrich the synthesis with foundational knowledge and make better connections
        if not allow_general_knowledge:
            allow_general_knowledge = True
            logging.info("Auto-enabled general knowledge to enrich web search synthesis with foundational insights")
    else:
        effective_model = model
        # Determine if the model is OpenAI or Groq based on model name
        use_openai = effective_model.startswith("gpt-") or effective_model in ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"]
    
    # Set initial system prompt based on allow_general_knowledge setting and enrichment mode
    if allow_general_knowledge:
        if enrichment_mode == "advanced":
            # Will be enhanced later with intelligent enrichment
            system_prompt = general_knowledge_prompt
        else:
            # Simple mode: use simple general knowledge prompt (classic enrichment without complexity)
            system_prompt = simple_general_knowledge_prompt
    else:
        system_prompt = main_prompt
    
    # Use cached embeddings for the query
    query_embedded = get_query_embedding(query, use_openai_embeddings)
    
    # Check for embedding errors
    if "message" in query_embedded and "status" in query_embedded:
        error_msg = f"Query embedding failed: {query_embedded['message']}"
        logging.error(error_msg)
        return {"answer": f"Error: {error_msg}", "citations": []}
    
    query_emb = query_embedded["embedding"]
    query_sparse = query_embedded.get("sparse")
    
    # Enhanced logic for search_by_space_only
    if search_by_space_only and space_id:
        logging.info("Using enhanced document-aware search for space-wide query")
        
        # Use additional space-only prompt for space-wide searches regardless of allow_general_knowledge setting
        if allow_general_knowledge:
            # Combine general knowledge prompt with space-only instructions
            system_prompt = general_knowledge_prompt + "\n\n" + additional_space_only_prompt
        else:
            # Combine main prompt with space-only instructions
            system_prompt = main_prompt + "\n\n" + additional_space_only_prompt
        
        # Get all documents in the space
        space_documents = get_documents_in_space(space_id)
        
        if space_documents["total_count"] == 0:
            if allow_general_knowledge:
                # If no documents found but general knowledge is allowed, still try to answer
                logging.info("No documents in space, but allow_general_knowledge is True - generating enriched answer with domain expertise")
                fallback_prompt = no_docs_in_space_prompt
                if use_openai:
                    answer, citations = openai_client.generate_answer(query, [], fallback_prompt, model=effective_model, conversation_history=validated_history)
                else:
                    try:
                        answer, citations = generate_answer(query, [], fallback_prompt, model=effective_model, conversation_history=validated_history)
                    except Exception as e:
                        logging.warning(f"Groq generation failed for general knowledge fallback, trying OpenAI: {e}")
                        answer, citations = openai_client.generate_answer(query, [], fallback_prompt, model=openai_model, conversation_history=validated_history)
                return {"answer": answer, "citations": citations}
            else:
                return {"answer": "No documents found in this space.", "citations": []}
        
        # Extract all document IDs
        all_document_ids = []
        all_document_ids.extend([doc["id"] for doc in space_documents["pdfs"]])
        all_document_ids.extend([doc["id"] for doc in space_documents["yts"]])
        
        logging.info(f"Found {len(all_document_ids)} documents in space: {space_documents['total_count']} total ({len(space_documents['pdfs'])} PDFs, {len(space_documents['yts'])} videos)")
        
        # Use document-aware search
        search_start = time.time()
        hits = hybrid_search_document_aware(
            query_emb=query_emb,
            query_sparse=query_sparse,
            document_ids=all_document_ids,
            space_id=space_id,
            acl_tags=acl_tags,
            top_k_per_doc=3,  # Get fewer results per document but from all documents
            include_full_text=True
        )
        logging.info(f"Document-aware search took {time.time() - search_start:.2f}s")
        
        if not hits:
            if allow_general_knowledge:
                # If no relevant context found but general knowledge is allowed
                logging.info("No relevant context found, but allow_general_knowledge is True - generating enriched answer with domain expertise")
                fallback_prompt = no_relevant_in_scope_prompt.format(query=query, scope="the user's space")
                if use_openai:
                    answer, citations = openai_client.generate_answer(query, [], fallback_prompt, model=effective_model, conversation_history=validated_history)
                else:
                    try:
                        answer, citations = generate_answer(query, [], fallback_prompt, model=effective_model, conversation_history=validated_history)
                    except Exception as e:
                        logging.warning(f"Groq generation failed for general knowledge fallback, trying OpenAI: {e}")
                        answer, citations = openai_client.generate_answer(query, [], fallback_prompt, model=openai_model, conversation_history=validated_history)
                return {"answer": answer, "citations": citations}
            else:
                return {"answer": "No relevant context found in any documents.", "citations": []}
        
        # Use document-aware reranking
        rerank_start = time.time()
        reranked = rerank_results_document_aware(
            query=query,
            hits=hits,
            top_n_per_doc=2,  # Top 2 results per document
            max_total_results=min(max_context_chunks * 2, 12)  # Allow more results for comprehensive coverage
        )
        logging.info(f"Document-aware reranking took {time.time() - rerank_start:.2f}s")
        
        # Use enhanced context for multi-document answers
        limited_context = reranked[:max_context_chunks * 2]  # Allow more context for comprehensive answers
        context_chunks = limited_context  # Store for optional return
        
        # Apply intelligent enrichment if general knowledge is enabled and in advanced mode
        if allow_general_knowledge and enrichment_mode == "advanced":
            # Build context from available chunks for enrichment analysis
            rag_context_preview = "\n\n".join([
                chunk["text"] if "text" in chunk 
                else chunk["metadata"]["text"] if "metadata" in chunk and "text" in chunk["metadata"] 
                else "" 
                for chunk in limited_context[:3]  # Use first few chunks for analysis
            ])
            
            # Create intelligent enrichment prompt with learning style integration
            enhanced_prompt, enrichment_metadata = create_enriched_system_prompt(
                query, rag_context_preview, allow_general_knowledge, general_knowledge_prompt,
                learning_style, educational_mode
            )
            
            # Add few-shot examples for better guidance
            domain = enrichment_metadata.get("domain", "general")
            system_prompt = create_few_shot_enhanced_prompt(enhanced_prompt, domain)
            
            # Log enrichment strategy
            if enrichment_metadata.get("enrichment_applied"):
                learning_info = f" with {learning_style} learning style" if learning_style else ""
                educational_info = " (educational mode)" if enrichment_metadata.get("educational_mode") else ""
                logging.info(f"Applied intelligent enrichment for {domain} domain{learning_info}{educational_info}: {enrichment_metadata.get('reason', 'standard enrichment')}")
            else:
                learning_info = f" with {learning_style} learning style" if learning_style else ""
                educational_info = " (educational mode)" if educational_mode else ""
                logging.info(f"Using standard prompt{learning_info}{educational_info}: {enrichment_metadata.get('reason', 'no specific enrichment needed')}")
        elif allow_general_knowledge and enrichment_mode == "simple":
            logging.info("Using simple general knowledge enrichment mode (classic behavior)")
        
        # Generate document-aware answer
        llm_start = time.time()
        if use_openai:
            # Use OpenAI with enhanced context formatting
            enhanced_context = []
            for chunk in limited_context:
                doc_id = chunk.get("metadata", {}).get("source_document_id", "unknown")
                # Find document name
                doc_name = "Unknown Document"
                for pdf_doc in space_documents["pdfs"]:
                    if pdf_doc["id"] == doc_id:
                        doc_name = pdf_doc.get("file_name", "Unknown PDF")
                        break
                for yt_doc in space_documents["yts"]:
                    if yt_doc["id"] == doc_id:
                        doc_name = yt_doc.get("file_name", "Unknown Video")
                        break

                # Add document source to chunk metadata
                enhanced_chunk = chunk.copy()
                if "metadata" in enhanced_chunk:
                    enhanced_chunk["metadata"]["source_document_name"] = doc_name
                enhanced_context.append(enhanced_chunk)

            answer, citations = openai_client.generate_answer(query, enhanced_context, system_prompt, model=effective_model, conversation_history=validated_history)
        else:
            try:
                answer, citations = generate_answer_document_aware(
                    query=query,
                    context_chunks=limited_context,
                    space_documents=space_documents,
                    system_prompt=system_prompt,
                    model=effective_model,
                    conversation_history=validated_history
                )
            except Exception as e:
                logging.warning(f"Groq document-aware generation failed, falling back to OpenAI: {e}")
                # Fall back to OpenAI generation
                enhanced_context = []
                for chunk in limited_context:
                    doc_id = chunk.get("metadata", {}).get("source_document_id", "unknown")
                    # Find document name
                    doc_name = "Unknown Document"
                    for pdf_doc in space_documents["pdfs"]:
                        if pdf_doc["id"] == doc_id:
                            doc_name = pdf_doc.get("file_name", "Unknown PDF")
                            break
                    for yt_doc in space_documents["yts"]:
                        if yt_doc["id"] == doc_id:
                            doc_name = yt_doc.get("file_name", "Unknown Video")
                            break

                    # Add document source to chunk metadata
                    enhanced_chunk = chunk.copy()
                    if "metadata" in enhanced_chunk:
                        enhanced_chunk["metadata"]["source_document_name"] = doc_name
                    enhanced_context.append(enhanced_chunk)

                answer, citations = openai_client.generate_answer(query, enhanced_context, system_prompt, model=openai_model, conversation_history=validated_history)
        
        logging.info(f"LLM generation took {time.time() - llm_start:.2f}s")
        
    else:
        # Original logic for single document or global search
        doc_id_param = None if search_by_space_only else document_id
        
        # Use parallel hybrid search with optimized top_k
        search_start = time.time()
        hits = hybrid_search_parallel(
            query_emb=query_emb,
            query_sparse=query_sparse,
            top_k=max(20, rerank_top_n * 2),  # Ensure we have enough results for reranking
            doc_id=doc_id_param,
            space_id=space_id,
            acl_tags=acl_tags,
            include_full_text=True  # Need full text for reranking
        )
        logging.info(f"Search took {time.time() - search_start:.2f}s")
        
        if not hits:
            if allow_general_knowledge:
                # If no relevant context found but general knowledge is allowed
                logging.info("No relevant context found, but allow_general_knowledge is True - generating enriched answer with domain expertise")
                fallback_prompt = no_relevant_in_scope_prompt.format(query=query, scope="the specified document(s)")
                if use_openai:
                    answer, citations = openai_client.generate_answer(query, [], fallback_prompt, model=effective_model, conversation_history=validated_history)
                else:
                    try:
                        answer, citations = generate_answer(query, [], fallback_prompt, model=effective_model, conversation_history=validated_history)
                    except Exception as e:
                        logging.warning(f"Groq generation failed for general knowledge fallback, trying OpenAI: {e}")
                        answer, citations = openai_client.generate_answer(query, [], fallback_prompt, model=openai_model, conversation_history=validated_history)
                return {"answer": answer, "citations": citations}
            else:
                return {"answer": "No relevant context found.", "citations": []}
        
        # Rerank results (faster with optimized parameters)
        rerank_start = time.time()
        reranked = rerank_results(query, hits, top_n=rerank_top_n)
        logging.info(f"Reranking took {time.time() - rerank_start:.2f}s")
        
        # Limit context to the top max_context_chunks chunks to reduce LLM input size
        limited_context = reranked[:max_context_chunks]
        context_chunks = limited_context  # Store for optional return
        
        # Apply intelligent enrichment if general knowledge is enabled and in advanced mode
        if allow_general_knowledge and enrichment_mode == "advanced":
            # Build context from available chunks for enrichment analysis
            rag_context_preview = "\n\n".join([
                chunk["text"] if "text" in chunk 
                else chunk["metadata"]["text"] if "metadata" in chunk and "text" in chunk["metadata"] 
                else "" 
                for chunk in limited_context[:3]  # Use first few chunks for analysis
            ])
            
            # Create intelligent enrichment prompt with learning style integration
            enhanced_prompt, enrichment_metadata = create_enriched_system_prompt(
                query, rag_context_preview, allow_general_knowledge, general_knowledge_prompt,
                learning_style, educational_mode
            )
            
            # Add few-shot examples for better guidance
            domain = enrichment_metadata.get("domain", "general")
            system_prompt = create_few_shot_enhanced_prompt(enhanced_prompt, domain)
            
            # Log enrichment strategy
            if enrichment_metadata.get("enrichment_applied"):
                learning_info = f" with {learning_style} learning style" if learning_style else ""
                educational_info = " (educational mode)" if enrichment_metadata.get("educational_mode") else ""
                logging.info(f"Applied intelligent enrichment for {domain} domain{learning_info}{educational_info}: {enrichment_metadata.get('reason', 'standard enrichment')}")
            else:
                learning_info = f" with {learning_style} learning style" if learning_style else ""
                educational_info = " (educational mode)" if educational_mode else ""
                logging.info(f"Using standard prompt{learning_info}{educational_info}: {enrichment_metadata.get('reason', 'no specific enrichment needed')}")
        elif allow_general_knowledge and enrichment_mode == "simple":
            logging.info("Using simple general knowledge enrichment mode (classic behavior)")
        
        # Generate answer with limited context
        llm_start = time.time()
        if use_openai:
            answer, citations = openai_client.generate_answer(query, limited_context, system_prompt, model=effective_model, conversation_history=validated_history)
        else:
            try:
                answer, citations = generate_answer(query, limited_context, system_prompt, model=effective_model, conversation_history=validated_history)
            except Exception as e:
                logging.warning(f"Groq generation failed, falling back to OpenAI: {e}")
                answer, citations = openai_client.generate_answer(query, limited_context, system_prompt, model=openai_model, conversation_history=validated_history)

        logging.info(f"LLM generation took {time.time() - llm_start:.2f}s")
    
    # Handle websearch integration if enabled
    if enable_websearch:
        try:
            websearch_start = time.time()
            logging.info("Starting contextual web search integration")
            
            # Note: GPT-4o is automatically used for websearch synthesis in websearch_client
            
            # Build RAG context for analysis
            rag_context = "\n\n".join([
                chunk["text"] if "text" in chunk
                else chunk["metadata"]["text"] if "metadata" in chunk and "text" in chunk["metadata"]
                else ""
                for chunk in limited_context
            ])

            # Combined: Analyze context and generate queries in ONE Groq call (faster!)
            context_analysis, search_queries = analyze_and_generate_queries(rag_context, query)

            # Perform contextual web search with intent context (PARALLEL Exa searches!)
            web_results = await perform_contextual_websearch_async(search_queries, context_analysis)
            
            if web_results:
                # Synthesize RAG results with web search results using GPT-4o
                synthesized_answer, combined_sources = synthesize_rag_and_web_results(
                    query=query,
                    rag_context=rag_context,
                    web_results=web_results,
                    context_analysis=context_analysis,
                    system_prompt=system_prompt,
                    has_general_knowledge=allow_general_knowledge,
                    conversation_history=validated_history
                )
                
                # Update answer and citations with synthesized results
                answer = synthesized_answer
                citations.extend(combined_sources)
                
                logging.info(f"Websearch integration took {time.time() - websearch_start:.2f}s")
            else:
                logging.info("No web results found, using original RAG answer")
                
        except Exception as e:
            logging.error(f"Websearch integration failed: {e}")
            logging.info("Falling back to original RAG answer")
    
    logging.info(f"Total query processing took {time.time() - start_time:.2f}s")

    # Process diagrams if requested
    diagrams = []
    if include_diagrams and context_chunks:
        try:
            diagram_start = time.time()

            # Separate image chunks from text chunks
            image_chunks = []
            text_only_chunks = []

            for chunk in context_chunks:
                chunk_metadata = chunk.get("metadata", {})
                if chunk_metadata.get("type") == "image":
                    image_chunks.append(chunk)
                else:
                    text_only_chunks.append(chunk)

            # Process diagrams if found
            if image_chunks:
                logging.info(f"Found {len(image_chunks)} diagram(s) in search results, processing...")

                # Use diagram explainer to generate descriptions
                diagrams = explain_diagrams_batch(
                    diagrams=image_chunks,
                    query=query,
                    text_chunks=text_only_chunks,
                    max_diagrams=max_diagrams
                )

                logging.info(f"Diagram processing took {time.time() - diagram_start:.2f}s, returned {len(diagrams)} diagrams")
            else:
                logging.info("No diagrams found in search results")

        except Exception as e:
            logging.error(f"Error processing diagrams: {e}")
            # Continue without diagrams rather than failing the whole request
            diagrams = []

    # Return answer and citations, optionally include context chunks and diagrams
    result = {"answer": answer, "citations": citations}
    if return_context:
        result["context_chunks"] = context_chunks
    if include_diagrams:
        result["diagrams"] = diagrams
    return result


def process_youtube(
    youtube_url: str,
    space_id: str,
    embedding_model: str = "text-embedding-ada-002",
    chunk_size: int = 512,
    overlap_tokens: int = 80
) -> str:
    """
    Process a YouTube video: extract transcript, chunk, embed, and index.
    
    Args:
        youtube_url: URL of the YouTube video
        space_id: Space ID to associate with the video
        embedding_model: Model to use for embeddings
        chunk_size: Size of chunks in tokens
        overlap_tokens: Number of tokens to overlap between chunks
        
    Returns:
        The document_id of the processed transcript
    """
    logging.info(f"Processing YouTube video: {youtube_url} for space: {space_id}")
    
    # Initialize WetroCloud service
    wetro_service = WetroCloudYouTubeService()
    
    # Extract video ID
    video_id = wetro_service.extract_video_id(youtube_url)
    if not video_id:
        error_msg = f"Invalid YouTube URL: {youtube_url}"
        logging.error(error_msg)
        raise ValueError(error_msg)
    
    # Get transcript (with fallback enabled by default)
    transcript_result = wetro_service.get_transcript(youtube_url)
    if not transcript_result.get('success'):
        error_msg = f"Failed to extract transcript: {transcript_result.get('message')}"
        logging.error(error_msg)
        raise ValueError(error_msg)

    transcript_text = transcript_result['text']

    # Get video title - prefer title from yt-dlp if available
    if 'video_title' in transcript_result and transcript_result['video_title']:
        # yt-dlp provides the title (either direct or fallback)
        video_title = transcript_result['video_title']
        logging.info(f"Using video title from {transcript_result.get('method', 'transcript service')}: {video_title}")
    else:
        # Fall back to external API for title (WetroCloud doesn't return title)
        yt_api_url = os.getenv('YT_API_URL', 'https://pdf-ocr-staging-production.up.railway.app')
        video_title = wetro_service.get_video_title(youtube_url, yt_api_url)
        logging.info(f"Using video title from external API: {video_title}")

    # Use thumbnail from transcript result if available, otherwise construct it
    thumbnail = transcript_result.get('thumbnail', f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg")
    
    # Insert into yts table
    yts_record = {
        "space_id": space_id,
        "yt_url": youtube_url,
        "extracted_text": transcript_text,
        "thumbnail": thumbnail,
        "file_name": video_title
    }
    
    document_id = insert_yts_record(yts_record)
    logging.info(f"Inserted YouTube transcript into yts table with ID: {document_id}")
    
    # Create file in uploads directory for backup/reference
    filename = f"yt_{video_id}_{document_id}.txt"
    file_path = os.path.join(UPLOAD_DIR, filename)
    
    try:
        # Write transcript to file for backup
        with open(file_path, 'w') as file:
            file.write(transcript_text)
        logging.info(f"Saved transcript to file: {file_path}")
        
        # Direct chunking of transcript text without requiring file
        logging.info(f"Chunking transcript directly from text")
        chunks = chunk_document(
            text=transcript_text,  # Pass text directly to chunker
            chunk_size=chunk_size,
            overlap_tokens=overlap_tokens,
            recipe="markdown",  # Use text recipe for transcript format with timestamps
            lang="en",
            min_characters_per_chunk=12
        )
        chunks = flatten_chunks(chunks)
        
        if not chunks:
            logging.error(f"No chunks generated from transcript")
            raise ValueError("Transcript chunking returned no chunks. The transcript may be empty.")
            
        logging.info(f"Generated {len(chunks)} chunks from transcript")
        
        # Validate and clean text in chunks
        cleaned_chunks = []
        for chunk in chunks:
            if isinstance(chunk, dict) and 'text' in chunk:
                # Ensure text is properly formatted for OpenAI API
                text = chunk['text'].strip()
                if text:  # Skip empty chunks
                    chunk['text'] = text
                    cleaned_chunks.append(chunk)
        
        logging.info(f"Cleaned chunks: {len(cleaned_chunks)}")
        
        # Embed chunks with multimodal embeddings (Jina CLIP-v2, 1024 dims)
        logging.info(f"Embedding chunks with Jina CLIP-v2 multimodal model (1024 dims)")
        embeddings = embed_chunks_multimodal(cleaned_chunks, batch_size=32)
        
        # Check for embedding errors
        if embeddings and isinstance(embeddings[0], dict) and "message" in embeddings[0] and "status" in embeddings[0]:
            error_msg = f"Multimodal embedding error: {embeddings[0]['message']}"
            logging.error(error_msg)
            raise ValueError(error_msg)
        
        # Upsert to Pinecone
        logging.info(f"Upserting vectors to Pinecone")
        upsert_vectors(
            doc_id=document_id,
            space_id=space_id,
            embeddings=embeddings,
            chunks=cleaned_chunks,
            source_file=video_title or f"YouTube: {video_id}"
        )
        
        return document_id
    
    finally:
        # Clean up file
        if os.path.exists(file_path):
            os.remove(file_path)
            logging.info(f"Removed transcript file: {file_path}")

@app.get("/")
async def root():
    return {"greeting": "Hello!", "message": "Welcome to Nuton RAG Pipeline!"}


@app.post("/process_document")
async def process_document_endpoint(
    files: List[UploadFile] = File(...),
    file_urls: str = Form(...),  # JSON string of URLs corresponding to each file
    space_id: str = Form(...),
    use_openai: bool = Form(True)
) -> JSONResponse:
    """
    Endpoint to process multiple documents: chunk, embed, index, and store metadata.
    Each file can have a corresponding URL that will be stored in the file_path column.
    
    Args:
        files: List of files to process
        file_urls: JSON string containing URLs corresponding to each file
        space_id: Space ID to associate with all documents
        use_openai: Whether to use OpenAI for embeddings
        
    Returns:
        List of document_ids for processed files
    """
    logging.info(f"Process document endpoint called with {len(files)} files, space_id: {space_id}, use_openai: {use_openai}")
    
    # Parse the file_urls JSON string
    try:
        urls = json.loads(file_urls)
        if not isinstance(urls, list):
            return JSONResponse({"error": "file_urls must be a JSON array"}, status_code=400)
    except json.JSONDecodeError:
        return JSONResponse({"error": "Invalid JSON in file_urls"}, status_code=400)
    
    # Validate that we have the same number of files and URLs
    if len(files) != len(urls):
        return JSONResponse(
            {"error": f"Number of files ({len(files)}) does not match number of URLs ({len(urls)})"},
            status_code=400
        )
    
    document_ids = []
    errors = []
    
    # Process each file with its corresponding URL
    for i, (file, url) in enumerate(zip(files, urls)):
        try:
            # Save the uploaded file temporarily
            temp_file_path = os.path.join(UPLOAD_DIR, file.filename)
            with open(temp_file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            # Prepare metadata with the URL as file_path
            metadata = {
                "filename": file.filename,
                "space_id": space_id,
                "file_path": url  # Store the URL in the file_path field
            }
            
            # Process the document using the appropriate function
            if use_openai:
                document_id = await process_document_with_openai(temp_file_path, metadata)
            else:
                document_id = process_document(temp_file_path, metadata)
                
            document_ids.append({"file": file.filename, "document_id": document_id, "url": url})
            
            # Clean up the temporary file
            os.remove(temp_file_path)
            
        except Exception as e:
            logging.exception(f"Error processing file {file.filename}: {e}")
            errors.append({"file": file.filename, "error": str(e)})
    
    # Return results
    if document_ids:
        result = {"document_ids": document_ids}
        if errors:
            result["errors"] = errors
        return JSONResponse(result)
    else:
        return JSONResponse({"error": "All files failed to process", "details": errors}, status_code=500)


@app.post("/answer_query_stream")
async def answer_query_stream_endpoint(
    query: str = Form(...),
    document_id: str = Form(...),
    space_id: Optional[str] = Form(None),
    acl_tags: Optional[str] = Form(None),
    use_openai_embeddings: bool = Form(True),
    search_by_space_only: bool = Form(False),
    rerank_top_n: Optional[int] = Form(10),
    max_context_chunks: Optional[int] = Form(5),
    fast_mode: bool = Form(False),
    allow_general_knowledge: bool = Form(False),
    enable_websearch: bool = Form(False),
    model: str = Form("openai/gpt-oss-120b"),
    enrichment_mode: str = Form("simple"),
    learning_style: Optional[str] = Form(None),
    educational_mode: bool = Form(False),
    conversation_history: Optional[str] = Form(None),
    include_diagrams: bool = Form(True),
    max_diagrams: int = Form(3)
):
    """
    STREAMING endpoint: Returns real-time status updates and progressive answers.
    Uses Server-Sent Events (SSE) for streaming responses.
    """
    # Parse params outside the generator
    start_time = time.time()
    acl_list = [tag.strip() for tag in acl_tags.split(",")] if acl_tags else None

    # Parse conversation history
    history_list = None
    if conversation_history:
        try:
            history_list = json.loads(conversation_history)
            if not isinstance(history_list, list):
                history_list = None
        except json.JSONDecodeError:
            history_list = None

    # Fast mode adjustments
    effective_rerank = rerank_top_n
    effective_chunks = max_context_chunks
    if fast_mode:
        effective_rerank = min(rerank_top_n, 5)
        effective_chunks = min(max_context_chunks, 3)

    async def generate_stream():
        """Generator function for SSE streaming."""
        try:
            # Send initial status
            yield f"data: {json.dumps({'status': '🔍 Searching documents...', 'type': 'status'})}\n\n"

            # Call answer_query to get RAG results (without websearch first)
            # Request context_chunks so we can use them for websearch synthesis
            rag_result = await answer_query(
                query,
                document_id,
                space_id=space_id,
                acl_tags=acl_list,
                use_openai_embeddings=use_openai_embeddings,
                search_by_space_only=search_by_space_only,
                rerank_top_n=effective_rerank,
                max_context_chunks=effective_chunks,
                allow_general_knowledge=allow_general_knowledge,
                enable_websearch=False,  # Don't do websearch in the main function
                model=model,
                enrichment_mode=enrichment_mode,
                learning_style=learning_style,
                educational_mode=educational_mode,
                conversation_history=history_list,
                return_context=True,  # Get context chunks for websearch synthesis
                include_diagrams=include_diagrams,
                max_diagrams=max_diagrams
            )

            # Stream RAG answer
            rag_time = int((time.time() - start_time) * 1000)
            yield f"data: {json.dumps({'status': '✅ Document search complete', 'type': 'status'})}\n\n"
            yield f"data: {json.dumps({'answer': rag_result['answer'], 'citations': rag_result['citations'], 'time_ms': rag_time, 'type': 'rag_answer'})}\n\n"

            # Stream diagrams if found
            if include_diagrams and rag_result.get('diagrams'):
                diagrams = rag_result['diagrams']
                yield f"data: {json.dumps({'status': f'📊 Found {len(diagrams)} diagram(s)', 'type': 'status'})}\n\n"
                yield f"data: {json.dumps({'diagrams': diagrams, 'type': 'diagrams'})}\n\n"

            # If websearch is enabled, do it now with progressive updates
            if enable_websearch:
                try:
                    websearch_start = time.time()

                    yield f"data: {json.dumps({'status': '🧠 Analyzing context for web search...', 'type': 'status'})}\n\n"

                    # Build RAG context from the context chunks (same as in answer_query function)
                    context_chunks_list = rag_result.get('context_chunks', [])
                    if context_chunks_list:
                        # Build context from chunks like in answer_query (lines 889-894)
                        rag_context = "\n\n".join([
                            chunk["text"] if "text" in chunk
                            else chunk["metadata"]["text"] if "metadata" in chunk and "text" in chunk["metadata"]
                            else ""
                            for chunk in context_chunks_list
                        ])
                    else:
                        # Fallback to using answer if context_chunks not available
                        logging.warning("No context_chunks in rag_result, falling back to using answer as context")
                        rag_context = rag_result['answer']

                    # Build correct system prompt based on parameters (same logic as answer_query)
                    synthesis_system_prompt = main_prompt

                    # Auto-enable educational mode if learning style is specified
                    effective_educational_mode = educational_mode
                    if learning_style and not educational_mode:
                        effective_educational_mode = True
                        logging.info(f"Auto-enabled educational mode for web synthesis due to learning style: {learning_style}")

                    # Set initial system prompt based on allow_general_knowledge setting
                    if allow_general_knowledge:
                        if enrichment_mode == "advanced":
                            synthesis_system_prompt = general_knowledge_prompt
                        else:
                            synthesis_system_prompt = simple_general_knowledge_prompt
                    else:
                        synthesis_system_prompt = main_prompt

                    # Apply intelligent enrichment if enrichment_mode is advanced
                    if allow_general_knowledge and enrichment_mode == "advanced":
                        # Build context preview from RAG answer for enrichment analysis
                        rag_context_preview = rag_context[:2000]  # Use first 2000 chars as preview

                        # Create intelligent enrichment prompt with learning style integration
                        enhanced_prompt, enrichment_metadata = create_enriched_system_prompt(
                            query, rag_context_preview, allow_general_knowledge, general_knowledge_prompt,
                            learning_style, effective_educational_mode
                        )

                        # Add few-shot examples for better guidance
                        domain = enrichment_metadata.get("domain", "general")
                        synthesis_system_prompt = create_few_shot_enhanced_prompt(enhanced_prompt, domain)

                        # Log enrichment strategy
                        if enrichment_metadata.get("enrichment_applied"):
                            learning_info = f" with {learning_style} learning style" if learning_style else ""
                            educational_info = " (educational mode)" if enrichment_metadata.get("educational_mode") else ""
                            logging.info(f"Applied intelligent enrichment for web synthesis - {domain} domain{learning_info}{educational_info}: {enrichment_metadata.get('reason', 'standard enrichment')}")
                        else:
                            learning_info = f" with {learning_style} learning style" if learning_style else ""
                            educational_info = " (educational mode)" if effective_educational_mode else ""
                            logging.info(f"Using standard prompt for web synthesis{learning_info}{educational_info}: {enrichment_metadata.get('reason', 'no specific enrichment needed')}")
                    elif allow_general_knowledge and enrichment_mode == "simple":
                        logging.info("Using simple general knowledge enrichment mode for web synthesis (classic behavior)")

                    # Combined: Analyze context and generate queries in ONE Groq call
                    context_analysis, search_queries = analyze_and_generate_queries(rag_context, query)

                    yield f"data: {json.dumps({'status': f'🔍 Generated {len(search_queries)} targeted search queries', 'type': 'status'})}\n\n"

                    # Perform contextual web search with parallel Exa searches
                    yield f"data: {json.dumps({'status': '🌐 Searching web in parallel...', 'type': 'status'})}\n\n"
                    web_results = await perform_contextual_websearch_async(search_queries, context_analysis)

                    if web_results:
                        yield f"data: {json.dumps({'status': f'✅ Found {len(web_results)} quality web sources', 'type': 'status'})}\n\n"

                        # Synthesize RAG results with web search results
                        yield f"data: {json.dumps({'status': '💭 Synthesizing RAG + web insights...', 'type': 'status'})}\n\n"

                        synthesized_answer, combined_sources = synthesize_rag_and_web_results(
                            query=query,
                            rag_context=rag_context,
                            web_results=web_results,
                            context_analysis=context_analysis,
                            system_prompt=synthesis_system_prompt,  # Use computed prompt with all parameters
                            has_general_knowledge=allow_general_knowledge,
                            conversation_history=history_list
                        )

                        # Stream the web-enhanced answer as a new event
                        web_time = int((time.time() - websearch_start) * 1000)
                        total_citations = rag_result['citations'] + combined_sources

                        yield f"data: {json.dumps({'status': '✅ Web enrichment complete', 'type': 'status'})}\n\n"
                        yield f"data: {json.dumps({'answer': synthesized_answer, 'citations': total_citations, 'time_ms': web_time, 'type': 'web_enhanced_answer'})}\n\n"

                        logging.info(f"Websearch integration took {web_time}ms")
                    else:
                        yield f"data: {json.dumps({'status': '⚠️ No web results found, using RAG answer', 'type': 'status'})}\n\n"

                except Exception as e:
                    logging.error(f"Websearch integration failed: {e}")
                    yield f"data: {json.dumps({'status': '⚠️ Web search failed, using RAG answer', 'type': 'status'})}\n\n"

            # Send final completion
            total_time = int((time.time() - start_time) * 1000)
            yield f"data: {json.dumps({'status': '🎉 Complete', 'time_ms': total_time, 'type': 'complete'})}\n\n"

        except Exception as e:
            logging.exception(f"Error in streaming endpoint: {e}")
            yield f"data: {json.dumps({'error': str(e), 'type': 'error'})}\n\n"

    return StreamingResponse(generate_stream(), media_type="text/event-stream")


@app.post("/answer_query")
async def answer_query_endpoint(
    query: str = Form(...),
    document_id: str = Form(...),
    space_id: Optional[str] = Form(None),
    acl_tags: Optional[str] = Form(None),  # Comma-separated
    use_openai_embeddings: bool = Form(True),
    search_by_space_only: bool = Form(False),
    rerank_top_n: Optional[int] = Form(10),
    max_context_chunks: Optional[int] = Form(5),
    fast_mode: bool = Form(False),  # New parameter for faster but potentially lower quality results
    allow_general_knowledge: bool = Form(False),  # New parameter for allowing general knowledge supplementation
    enable_websearch: bool = Form(False),  # New parameter for enabling contextual web search
    model: str = Form("openai/gpt-oss-120b"),  # User-selectable model parameter (updated to Groq model)
    enrichment_mode: str = Form("simple"),  # "simple" (default) or "advanced" for intelligent enrichment
    learning_style: Optional[str] = Form(None),  # Learning style for personalized educational responses
    educational_mode: bool = Form(False),  # Enable tutoring/educational approach
    conversation_history: Optional[str] = Form(None),  # JSON string: "[{role, content}, ...]"
    include_diagrams: bool = Form(True),  # Whether to include diagrams in response
    max_diagrams: int = Form(3)  # Maximum number of diagrams to return
) -> JSONResponse:
    """
    Optimized endpoint to answer a query using the RAG pipeline.
    Returns the answer, citations, and optionally diagrams.

    Args:
        query: User's question
        document_id: Document ID to search within
        space_id: Optional space ID to filter by
        acl_tags: Optional comma-separated ACL tags to filter by
        use_openai_embeddings: Whether to use OpenAI directly for embeddings
        search_by_space_only: If True, search by space_id only, ignoring document_id
        rerank_top_n: Number of results to rerank (default: 10)
        max_context_chunks: Maximum number of chunks to include in context (default: 5)
        fast_mode: If True, uses optimized settings for faster response time
        allow_general_knowledge: If True, allows LLM to enrich answers with general knowledge, expanding beyond documents with additional insights and context
        enable_websearch: If True, performs contextual web search to supplement RAG results
        model: The model to use for generation (default: openai/gpt-oss-120b from Groq)
        enrichment_mode: "simple" (default, classic enrichment) or "advanced" (intelligent domain-aware enrichment)
        learning_style: Learning style for personalized educational responses ("academic_focus", "deep_dive", "quick_practical", "exploratory_curious", "narrative_reader", "default", or None)
        educational_mode: If True, enables tutoring/educational approach with context-rich responses
        conversation_history: JSON string of conversation history for context continuity
        include_diagrams: If True, processes and returns diagrams found in search results
        max_diagrams: Maximum number of diagrams to return (default: 3)
    """
    start_time = time.time()
    acl_list = [tag.strip() for tag in acl_tags.split(",")] if acl_tags else None

    # Parse conversation history from JSON string
    history_list = None
    if conversation_history:
        try:
            history_list = json.loads(conversation_history)
            if not isinstance(history_list, list):
                logging.warning("conversation_history is not a list, ignoring")
                history_list = None
        except json.JSONDecodeError as e:
            logging.warning(f"Failed to parse conversation_history JSON: {e}")
            history_list = None

    # If fast_mode is enabled, adjust parameters for faster response
    if fast_mode:
        rerank_top_n = min(rerank_top_n, 5)  # Limit reranking in fast mode
        max_context_chunks = min(max_context_chunks, 3)  # Use fewer chunks in context

    try:
        result = await answer_query(
            query,
            document_id,
            space_id=space_id,
            acl_tags=acl_list,
            use_openai_embeddings=use_openai_embeddings,
            search_by_space_only=search_by_space_only,
            rerank_top_n=rerank_top_n,
            max_context_chunks=max_context_chunks,
            allow_general_knowledge=allow_general_knowledge,
            enable_websearch=enable_websearch,
            model=model,
            enrichment_mode=enrichment_mode,
            learning_style=learning_style,
            educational_mode=educational_mode,
            conversation_history=history_list,
            include_diagrams=include_diagrams,
            max_diagrams=max_diagrams
        )
        result["time_ms"] = int((time.time() - start_time) * 1000)  # Add time taken in ms
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/process_youtube")
async def process_youtube_endpoint(
    youtube_urls: str = Form(...),  # JSON string array of YouTube URLs
    space_id: str = Form(...),
    embedding_model: str = Form("text-embedding-ada-002"),
    chunk_size: int = Form(512),
    overlap_tokens: int = Form(80)
) -> JSONResponse:
    """
    Endpoint to process multiple YouTube videos: extract transcripts, chunk, embed, and index.
    Returns the document_ids for all processed videos.

    Args:
        youtube_urls: JSON string array of YouTube URLs to process
        space_id: Space ID to associate with all videos
        embedding_model: Model to use for embeddings
        chunk_size: Size of chunks in tokens
        overlap_tokens: Number of tokens to overlap between chunks

    Returns:
        List of document_ids with their corresponding YouTube URLs
    """
    logging.info(f"Process YouTube endpoint called with multiple URLs, space_id: {space_id}")

    # Parse the youtube_urls JSON string
    try:
        urls = json.loads(youtube_urls)
        if not isinstance(urls, list):
            return JSONResponse({"error": "youtube_urls must be a JSON array"}, status_code=400)
    except json.JSONDecodeError:
        return JSONResponse({"error": "Invalid JSON in youtube_urls"}, status_code=400)

    document_ids = []
    errors = []

    # Process each YouTube URL
    for url in urls:
        try:
            document_id = process_youtube(
                youtube_url=url,
                space_id=space_id,
                embedding_model=embedding_model,
                chunk_size=chunk_size,
                overlap_tokens=overlap_tokens
            )

            document_ids.append({"youtube_url": url, "document_id": document_id})

        except Exception as e:
            logging.exception(f"Error processing YouTube URL {url}: {e}")
            errors.append({"youtube_url": url, "error": str(e)})

    # Return results
    if document_ids:
        result = {"document_ids": document_ids}
        if errors:
            result["errors"] = errors
        return JSONResponse(result)
    else:
        return JSONResponse({"error": "All YouTube URLs failed to process", "details": errors}, status_code=500)


@app.post("/extract_youtube_transcript")
async def extract_youtube_transcript_endpoint(
    video_url: str = Form(...),
    use_proxy: bool = Form(False),
    languages: str = Form("en")  # Comma-separated language codes
) -> JSONResponse:
    """
    Extract transcript from a YouTube video using youtube-transcript-api.

    This endpoint uses the youtube-transcript-api library which:
    - Supports WebShare proxy to bypass cloud provider IP blocking
    - Works reliably on cloud deployments (AWS, GCP, Azure, etc.)
    - Provides direct access to YouTube transcripts without external APIs

    Args:
        video_url: YouTube video URL or video ID
        use_proxy: Enable WebShare proxy (recommended for cloud deployments)
                  Requires WEBSHARE_PROXY_USERNAME and WEBSHARE_PROXY_PASSWORD env vars
        languages: Comma-separated list of preferred language codes (default: "en")

    Returns:
        JSON response with transcript text and metadata
    """
    logging.info(f"Extract YouTube transcript endpoint called for: {video_url}, use_proxy: {use_proxy}")

    # Parse languages
    lang_list = [lang.strip() for lang in languages.split(",")]

    try:
        # Initialize the service
        yt_service = YouTubeTranscriptService(use_proxy=use_proxy)

        # Get proxy status for diagnostics
        proxy_status = yt_service.get_proxy_status()

        # Get transcript
        result = yt_service.get_transcript(video_url, languages=lang_list)

        if result['success']:
            return JSONResponse({
                'success': True,
                'video_id': result['video_id'],
                'transcript': result['text'],
                'thumbnail': result['thumbnail'],
                'language': result['language'],
                'transcript_entries': result['transcript_entries'],
                'proxy_status': proxy_status
            })
        else:
            error_response = {
                'success': False,
                'error': result['message'],
                'proxy_status': proxy_status
            }
            if 'suggestions' in result:
                error_response['suggestions'] = result['suggestions']

            return JSONResponse(error_response, status_code=400)

    except Exception as e:
        logging.exception(f"Error in extract_youtube_transcript_endpoint: {e}")
        return JSONResponse({
            'success': False,
            'error': str(e)
        }, status_code=500)


@app.get("/youtube_proxy_status")
async def youtube_proxy_status_endpoint() -> JSONResponse:
    """
    Check WebShare proxy configuration status.

    Returns:
        JSON response with proxy configuration details
    """
    has_username = bool(os.getenv('WEBSHARE_PROXY_USERNAME'))
    has_password = bool(os.getenv('WEBSHARE_PROXY_PASSWORD'))

    return JSONResponse({
        'proxy_configured': has_username and has_password,
        'has_username': has_username,
        'has_password': has_password,
        'message': 'Proxy fully configured' if (has_username and has_password) else 'Missing WebShare credentials',
        'instructions': {
            'step_1': 'Sign up at https://www.webshare.io/',
            'step_2': 'Purchase RESIDENTIAL proxy package (not Static or Proxy Server)',
            'step_3': 'Get your proxy username and password from dashboard',
            'step_4': 'Set environment variables: WEBSHARE_PROXY_USERNAME and WEBSHARE_PROXY_PASSWORD',
            'step_5': 'Restart the server'
        }
    })


@app.post("/extract_transcript_ytdlp")
async def extract_transcript_ytdlp_endpoint(
    video_url: str = Form(...),
    languages: str = Form("en")  # Comma-separated language codes
) -> JSONResponse:
    """
    Extract transcript from YouTube using yt-dlp (RECOMMENDED - Most Reliable).

    This endpoint uses yt-dlp which:
    - Works WITHOUT proxies in 99% of cases
    - Uses YouTube's internal APIs directly
    - Actively maintained for YouTube changes
    - Doesn't get blocked by IP restrictions
    - Most robust solution available

    Args:
        video_url: YouTube video URL or video ID
        languages: Comma-separated list of preferred language codes (default: "en")

    Returns:
        JSON response with transcript text and metadata
    """
    logging.info(f"Extract YouTube transcript (yt-dlp) endpoint called for: {video_url}")

    # Parse languages
    lang_list = [lang.strip() for lang in languages.split(",")]

    try:
        # Initialize yt-dlp service
        ytdlp_service = YTDLPTranscriptService()

        # Get transcript
        result = ytdlp_service.get_transcript(video_url, languages=lang_list)

        if result['success']:
            return JSONResponse({
                'success': True,
                'video_id': result['video_id'],
                'video_title': result['video_title'],
                'transcript': result['text'],
                'thumbnail': result['thumbnail'],
                'language': result['language'],
                'is_automatic': result['is_automatic'],
                'transcript_entries': result['transcript_entries'],
                'method': result['method']
            })
        else:
            return JSONResponse({
                'success': False,
                'error': result['message']
            }, status_code=400)

    except Exception as e:
        logging.exception(f"Error in extract_transcript_ytdlp_endpoint: {e}")
        return JSONResponse({
            'success': False,
            'error': str(e)
        }, status_code=500)


@app.get("/youtube_info_ytdlp")
async def youtube_info_ytdlp_endpoint(
    video_url: str
) -> JSONResponse:
    """
    Get YouTube video information and available subtitles using yt-dlp.

    Args:
        video_url: YouTube video URL or video ID

    Returns:
        JSON response with video info and available subtitles
    """
    logging.info(f"YouTube info (yt-dlp) endpoint called for: {video_url}")

    try:
        ytdlp_service = YTDLPTranscriptService()

        # Get video info
        video_info = ytdlp_service.get_video_info(video_url)

        if not video_info['success']:
            return JSONResponse({
                'success': False,
                'error': video_info['message']
            }, status_code=400)

        # Get available subtitles
        subtitles_info = ytdlp_service.get_available_subtitles(video_url)

        return JSONResponse({
            'success': True,
            'video_id': video_info['video_id'],
            'title': video_info['title'],
            'description': video_info['description'],
            'duration': video_info['duration'],
            'channel': video_info['channel'],
            'upload_date': video_info['upload_date'],
            'view_count': video_info['view_count'],
            'thumbnail': video_info['thumbnail'],
            'video_url': video_info['video_url'],
            'available_subtitles': subtitles_info.get('available_subtitles', []),
            'method': 'yt-dlp'
        })

    except Exception as e:
        logging.exception(f"Error in youtube_info_ytdlp_endpoint: {e}")
        return JSONResponse({
            'success': False,
            'error': str(e)
        }, status_code=500)


@app.get("/youtube_transcript_info")
async def youtube_transcript_info_endpoint(
    video_url: str,
    use_proxy: bool = False
) -> JSONResponse:
    """
    Get available transcript languages for a YouTube video.

    Args:
        video_url: YouTube video URL or video ID
        use_proxy: Enable WebShare proxy (recommended for cloud deployments)

    Returns:
        JSON response with available transcripts and video info
    """
    logging.info(f"YouTube transcript info endpoint called for: {video_url}")

    try:
        yt_service = YouTubeTranscriptService(use_proxy=use_proxy)

        # Get available transcripts
        transcripts_result = yt_service.get_available_transcripts(video_url)

        # Get video info
        video_info = yt_service.get_video_info(video_url)

        if transcripts_result['success']:
            return JSONResponse({
                'success': True,
                'video_id': transcripts_result['video_id'],
                'video_url': video_info.get('video_url'),
                'thumbnail': video_info.get('thumbnail'),
                'available_transcripts': transcripts_result['available_transcripts']
            })
        else:
            return JSONResponse({
                'success': False,
                'error': transcripts_result['message']
            }, status_code=400)

    except Exception as e:
        logging.exception(f"Error in youtube_transcript_info_endpoint: {e}")
        return JSONResponse({
            'success': False,
            'error': str(e)
        }, status_code=500)


@app.post("/test_vcyon_transcript")
async def test_vcyon_transcript_endpoint(
    video_url: str = Form(...),
    languages: str = Form("en")  # Comma-separated language codes
) -> JSONResponse:
    """
    Test the Vcyon API for YouTube transcript extraction.

    This endpoint specifically tests the Vcyon API integration:
    - Extracts video ID from URL
    - Calls Vcyon's transcript API directly
    - Returns detailed information about video and transcript
    - Gets video metadata using Vcyon's video info endpoint

    Args:
        video_url: YouTube video URL
        languages: Comma-separated list of preferred language codes (default: "en")

    Returns:
        JSON response with transcript, video info, and API status
    """
    logging.info(f"Test Vcyon API endpoint called for: {video_url}")

    # Parse languages
    lang_list = [lang.strip() for lang in languages.split(",")]

    try:
        # Initialize WetroCloud service with Vcyon and yt-dlp fallback
        wetro_service = WetroCloudYouTubeService(
            enable_vcyon_fallback=True,   # Try Vcyon first when WetroCloud fails
            enable_ytdlp_fallback=True    # Use yt-dlp as final fallback
        )

        # Extract video ID
        video_id = wetro_service.extract_video_id(video_url)
        if not video_id:
            return JSONResponse({
                'success': False,
                'error': 'Invalid YouTube URL - could not extract video ID'
            }, status_code=400)

        # Get video information from Vcyon
        logging.info(f"Getting video info from Vcyon for video ID: {video_id}")
        video_info = wetro_service._get_video_info_from_vcyon(video_url)

        # Get transcript from Vcyon
        logging.info(f"Getting transcript from Vcyon for video ID: {video_id}")
        transcript_result = wetro_service._get_transcript_from_vcyon(video_url, lang_list)
        print('transcript_result', transcript_result)

        if transcript_result['success']:
            response_data = {
                'success': True,
                'method': 'vcyon',
                'video_id': video_id,
                'video_url': video_url,
                'transcript': transcript_result['text'],
                'transcript_length': len(transcript_result['text']),
                'transcript_entries_count': len(transcript_result.get('transcript_entries', [])),
                'language': transcript_result.get('language', 'unknown'),
                'thumbnail': transcript_result.get('thumbnail'),
            }

            # Add video metadata if available
            if video_info:
                response_data['video_info'] = {
                    'title': video_info.get('title', 'N/A'),
                    'author': video_info.get('author', 'N/A'),
                    'description': video_info.get('description', '')[:200] + '...' if video_info.get('description') else 'N/A',
                    'duration': video_info.get('duration', 0),
                    'view_count': video_info.get('view_count', 0),
                    'thumbnails': video_info.get('thumbnails', [])
                }
            else:
                response_data['video_info'] = None
                response_data['video_info_note'] = 'Video info not available from Vcyon API'

            # Add a preview of the transcript (first 500 characters)
            if transcript_result['text']:
                lines = transcript_result['text'].split('\n')
                response_data['transcript_preview'] = '\n'.join(lines[:10])  # First 10 lines

            return JSONResponse(response_data)
        else:
            return JSONResponse({
                'success': False,
                'error': transcript_result.get('message', 'Unknown error from Vcyon API'),
                'video_id': video_id,
                'video_url': video_url,
                'video_info': video_info if video_info else None
            }, status_code=400)

    except Exception as e:
        logging.exception(f"Error in test_vcyon_transcript_endpoint: {e}")
        return JSONResponse({
            'success': False,
            'error': str(e)
        }, status_code=500)


@app.post("/generate_flashcards")
async def generate_flashcards_endpoint(
    request: FlashcardRequest
) -> JSONResponse:
    """
    Endpoint to generate flashcards from a document.
    
    Args:
        request: FlashcardRequest with document_id, space_id, user_id, etc.
        
    Returns:
        JSON response with flashcards or error message.
    """
    logging.info(f"Generate flashcards endpoint called for document: {request.document_id}, user: {request.user_id}")
    
    try:
        result = generate_flashcards(
            document_id=request.document_id,
            space_id=request.space_id,
            user_id=request.user_id,
            num_questions=request.num_questions,
            acl_tags=request.acl_tags
        )
        
        return JSONResponse(result)
    except Exception as e:
        logging.exception(f"Error in generate_flashcards_endpoint: {e}")
        return JSONResponse({"error": str(e)}, status_code=500) 
    

@app.post("/regenerate_flashcards")
async def regenerate_flashcards_endpoint(
    request: FlashcardRequest
) -> JSONResponse:
    """
    Endpoint to regenerate flashcards from a document.
    
    Args:
        request: FlashcardRequest with document_id, space_id, user_id, etc.
        
    Returns:
        JSON response with flashcards or error message.
    """
    logging.info(f"Re-Generate flashcards endpoint called for document: {request.document_id}, user: {request.user_id}")
    
    try:
        result = regenerate_flashcards(
            document_id=request.document_id,
            space_id=request.space_id,
            user_id=request.user_id,
            num_questions=request.num_questions,
            acl_tags=request.acl_tags
        )
        
        return JSONResponse(result)
    except Exception as e:
        logging.exception(f"Error in regenerate_flashcards_endpoint: {e}")
        return JSONResponse({"error": str(e)}, status_code=500) 


@app.post("/generate_quiz")
async def generate_quiz_endpoint(
    request: QuizRequest
) -> JSONResponse:
    """
    Endpoint to generate a quiz from a document.
    Args:
        request: QuizRequest containing parameters:
            - document_id: The document to generate the quiz from.
            - space_id: Optional space ID.
            - user_id: UUID of the user creating the quiz.
            - question_type: Type of questions to generate ("mcq", "true_false", or "both").
            - num_questions: Total number of questions to generate.
            - acl_tags: Optional comma-separated ACL tags.
            - rerank_top_n: Number of results to rerank.
            - use_openai_embeddings: Whether to use OpenAI for embeddings.
            - set_id: Quiz set number.
            - title: Optional quiz title.
            - description: Optional quiz description.
    Returns:
        JSON response with quiz or error message.
    """
    # Validate question_type
    if request.question_type not in ["mcq", "true_false", "both"]:
        return JSONResponse({"error": "question_type must be one of 'mcq', 'true_false', or 'both'"}, status_code=400)
        
    acl_list = [tag.strip() for tag in request.acl_tags.split(",")] if request.acl_tags else None
    try:
        result = generate_quiz(
            document_id=request.document_id,
            space_id=request.space_id,
            user_id=request.user_id,
            question_type=request.question_type,
            num_questions=request.num_questions,
            acl_tags=acl_list,
            rerank_top_n=request.rerank_top_n,
            use_openai_embeddings=request.use_openai_embeddings,
            set_id=request.set_id,
            title=request.title,
            description=request.description
        )
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500) 

@app.post("/regenerate_quiz")
async def regenerate_quiz_endpoint(
    request: QuizRequest
) -> JSONResponse:
    """
    Endpoint to regenerate quiz questions from a document, avoiding duplicates from previous sets.
    Args:
        request: QuizRequest containing parameters:
            - document_id: The document to regenerate the quiz from.
            - space_id: Optional space ID.
            - user_id: UUID of the user creating the quiz.
            - question_type: Type of questions to generate ("mcq", "true_false", or "both").
            - num_questions: Total number of questions to generate.
            - acl_tags: Optional comma-separated ACL tags.
            - rerank_top_n: Number of results to rerank.
            - use_openai_embeddings: Whether to use OpenAI for embeddings.
            - set_id: Quiz set number (will be auto-incremented).
            - title: Optional quiz title.
            - description: Optional quiz description.
    Returns:
        JSON response with new quiz questions or error message.
    """
    # Validate question_type
    if request.question_type not in ["mcq", "true_false", "both"]:
        return JSONResponse({"error": "question_type must be one of 'mcq', 'true_false', or 'both'"}, status_code=400)
        
    acl_list = [tag.strip() for tag in request.acl_tags.split(",")] if request.acl_tags else None
    try:
        result = regenerate_quiz(
            document_id=request.document_id,
            space_id=request.space_id,
            user_id=request.user_id,
            question_type=request.question_type,
            num_questions=request.num_questions,
            acl_tags=acl_list,
            rerank_top_n=request.rerank_top_n,
            use_openai_embeddings=request.use_openai_embeddings,
            title=request.title,
            description=request.description
        )
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/generate_notes")
async def generate_notes_endpoint(
    document_id: str = Form(...),
    space_id: Optional[str] = Form(None),
    academic_level: str = Form("graduate"),
    include_diagrams: bool = Form(True),
    include_mermaid: bool = Form(True),
    max_chunks: int = Form(2000),
    target_coverage: float = Form(0.85),
    enable_gap_filling: bool = Form(True),
    acl_tags: Optional[str] = Form(None)
) -> JSONResponse:
    """
    Generate comprehensive study notes from a document.

    This endpoint creates extensive, well-formatted markdown study notes that cover
    every detail of the document, organized hierarchically with proper formatting,
    diagrams, and mermaid visualizations.

    Args:
        document_id: Document ID to generate notes for (required)
        space_id: Optional space ID filter
        academic_level: Target academic level - one of:
            - "undergraduate": Clear explanations with examples
            - "graduate": Advanced analysis with frameworks
            - "msc": Technical depth with methodologies
            - "phd": Critical analysis with research gaps
        include_diagrams: Whether to include diagrams from PDF (default: True)
        include_mermaid: Whether to generate mermaid diagrams (default: True)
        max_chunks: Maximum number of chunks to retrieve (default: 2000)
        target_coverage: Target coverage percentage for document (default: 0.85 = 85%)
            Higher values ensure more complete coverage but take longer
        enable_gap_filling: Whether to enable intelligent gap-filling to reach target coverage (default: True)
        acl_tags: Optional comma-separated ACL tags

    Returns:
        JSON response with:
        {
            "notes_markdown": "# Complete markdown notes...",
            "metadata": {
                "academic_level": "graduate",
                "total_pages": 120,
                "total_chapters": 8,
                "total_chunks_processed": 350,
                "diagrams_included": 12,
                "generation_time_seconds": 145.2,
                "coverage_score": 0.98,
                "text_coverage_percentage": 0.95,
                "notes_length_chars": 50000,
                "generated_at": "2025-11-04T..."
            },
            "status": "success"
        }

    Example:
        POST /generate_notes
        Form data:
            document_id=abc123
            academic_level=graduate
            include_diagrams=true
            include_mermaid=true
            target_coverage=0.90
    """
    from note_generation_process import generate_comprehensive_notes

    # Validate academic level
    valid_levels = ["undergraduate", "graduate", "msc", "phd"]
    if academic_level not in valid_levels:
        return JSONResponse({
            "error": f"Invalid academic_level. Must be one of: {', '.join(valid_levels)}",
            "status": "error"
        }, status_code=400)

    # Validate target_coverage
    if not 0.0 <= target_coverage <= 1.0:
        return JSONResponse({
            "error": f"Invalid target_coverage. Must be between 0.0 and 1.0 (got {target_coverage})",
            "status": "error"
        }, status_code=400)

    # Parse ACL tags
    acl_list = [tag.strip() for tag in acl_tags.split(",")] if acl_tags else None

    try:
        logger.info(f"🚀 Generating notes for document {document_id}, level={academic_level}, target_coverage={target_coverage:.0%}")

        # Generate notes
        result = await generate_comprehensive_notes(
            document_id=document_id,
            space_id=space_id,
            academic_level=academic_level,
            personalization_options=None,
            include_diagrams=include_diagrams,
            include_mermaid=include_mermaid,
            max_chunks=max_chunks,
            target_coverage=target_coverage,
            enable_gap_filling=enable_gap_filling,
            acl_tags=acl_list
        )

        if result.get("status") == "error":
            logger.error(f"Note generation failed: {result.get('message', 'Unknown error')}")
            return JSONResponse(result, status_code=500)

        logger.info(f"✅ Notes generated successfully: {len(result.get('notes_markdown', ''))} characters")

        # Save notes to database (generated_content table)
        try:
            # Check document type (PDF or YouTube)
            doc_type, _ = check_document_type(document_id)
            is_youtube = (doc_type == "youtube")
            
            # Ensure space_id is available
            if not space_id:
                logger.warning(f"No space_id provided for document {document_id}, cannot save to database")
            else:
                # Upsert notes to generated_content table
                content_id = upsert_generated_content_notes(
                    document_id=document_id,
                    notes_markdown=result.get("notes_markdown", ""),
                    space_id=space_id,
                    is_youtube=is_youtube
                )
                logger.info(f"💾 Notes saved to generated_content table with ID: {content_id}")
                result["saved_to_database"] = True
                result["content_id"] = content_id
        except Exception as db_error:
            logger.error(f"⚠️ Failed to save notes to database: {db_error}")
            result["database_save_error"] = str(db_error)
            # Don't fail the endpoint, just log the error

        # Save notes to file in note/ folder
        try:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{document_id}_{academic_level}_{timestamp}.md"
            filepath = os.path.join(NOTES_DIR, filename)

            with open(filepath, "w", encoding="utf-8") as f:
                f.write(result.get("notes_markdown", ""))

            logger.info(f"💾 Notes saved to: {filepath}")
            result["saved_to_file"] = filepath
        except Exception as file_error:
            logger.error(f"⚠️ Failed to save notes to file: {file_error}")
            result["file_save_error"] = str(file_error)

        return JSONResponse(result)

    except Exception as e:
        logger.error(f"❌ Error in note generation endpoint: {e}", exc_info=True)
        return JSONResponse({
            "notes_markdown": "",
            "metadata": {},
            "status": "error",
            "message": str(e)
        }, status_code=500)


@app.post("/api/google-drive/files")
async def get_drive_files(request: DriveFilesRequest) -> JSONResponse:
    """
    List files from user's Google Drive.
    
    Args:
        request: DriveFilesRequest with access tokens and filter options
        
    Returns:
        JSON response with file list and updated tokens
    """
    try:
        logging.info(f"Listing Google Drive files for user")
        
        # Initialize Google Drive service
        drive_service = GoogleDriveService(request.access_token, request.refresh_token)
        
        # List files with filters
        files = drive_service.list_files(
            folder_id=request.folder_id,
            file_types=request.file_types,
            max_results=request.max_results
        )
        
        # Get updated tokens
        updated_tokens = drive_service.get_updated_tokens()
        
        return JSONResponse({
            "files": files,
            "updated_tokens": updated_tokens
        })
        
    except Exception as e:
        logging.error(f"Error listing Drive files: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/google-drive/import")
async def import_drive_files(request: DriveImportRequest) -> JSONResponse:
    """
    Import selected Google Drive files into the RAG pipeline.
    
    Args:
        request: DriveImportRequest with file IDs and space information
        
    Returns:
        JSON response with processing results
    """
    try:
        logging.info(f"Importing {len(request.file_ids)} Google Drive files to space {request.space_id}")
        
        # Process the files
        results = await process_drive_files(
            file_ids=request.file_ids,
            space_id=request.space_id,
            access_token=request.access_token,
            refresh_token=request.refresh_token
        )
        
        # Determine overall status
        total_files = len(request.file_ids)
        successful_files = len(results["processed_files"])
        
        if successful_files == total_files:
            status = "completed"
            message = f"Successfully processed all {total_files} files"
        elif successful_files > 0:
            status = "partial_success"
            message = f"Processed {successful_files}/{total_files} files successfully"
        else:
            status = "failed"
            message = f"Failed to process any files"
        
        response_data = {
            "status": status,
            "message": message,
            "processed_files": results["processed_files"],
            "errors": results["errors"],
            "updated_tokens": results["updated_tokens"]
        }
        
        return JSONResponse(response_data)
        
    except Exception as e:
        logging.error(f"Error importing Drive files: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)