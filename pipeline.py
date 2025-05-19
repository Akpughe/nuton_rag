from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional
import os
import shutil
import logging
import tempfile
import json

from chonkie_client import chunk_document, embed_chunks, embed_chunks_v2, embed_query, embed_query_v2
from pinecone_client import upsert_vectors, hybrid_search, rerank_results
from supabase_client import insert_pdf_record, insert_yts_record
from groq_client import generate_answer
import openai_client
from services.wetrocloud_youtube import WetroCloudYouTubeService
from flashcard_process import generate_flashcards, regenerate_flashcards
from pydantic import BaseModel
from typing import Optional, List


app = FastAPI()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO)


class FlashcardRequest(BaseModel):
    document_id: str
    space_id: Optional[str] = None
    num_questions: Optional[int] = None
    acl_tags: Optional[List[str]] = None


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
    print(chunks)
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
    print('metadata', metadata)
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


def process_document_with_openai(
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
    Ingests a document using OpenAI for embeddings: chunk, embed with OpenAI, insert into Supabase, upsert to Pinecone.
    Returns the document_id (Supabase id).
    """
    try:
        # 1. Chunk document
        logging.info(f"Chunking document: {file_path}")
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
        print('chunks', chunks)
        
        if not chunks:
            logging.error(f"No chunks generated from document: {file_path}")
            raise ValueError("Document chunking returned no chunks. The document may be empty or in an unsupported format.")
            
        logging.info(f"Generated {len(chunks)} chunks from document")
        
        # 2. Embed chunks with OpenAI directly
        logging.info(f"Embedding chunks with OpenAI model: {embedding_model}")
        embeddings = embed_chunks_v2(chunks, model=embedding_model, batch_size=batch_size)
        
        # Check if embedding service returned an error
        if embeddings and isinstance(embeddings[0], dict) and "message" in embeddings[0] and "status" in embeddings[0]:
            error_msg = f"OpenAI embedding error: {embeddings[0]['message']}"
            logging.error(error_msg)
            raise ValueError(error_msg)
        
        logging.info(f"Successfully embedded {len(embeddings)} chunks with OpenAI")
        
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
        except ValueError as e:
            logging.error(f"Error upserting vectors: {e}")
            raise ValueError(f"Failed to process document embeddings: {str(e)}")
            
        return document_id
        
    except Exception as e:
        logging.exception(f"Error processing document with OpenAI: {e}")
        raise


def answer_query(
    query: str,
    document_id: Optional[str] = None,
    space_id: Optional[str] = None,
    acl_tags: Optional[List[str]] = None,
    rerank_top_n: int = 15,
    system_prompt: str = "You are a helpful assistant. Use only the provided context to answer.",
    groq_model: str = "meta-llama/llama-4-scout-17b-16e-instruct",
    openai_model: str = "gpt-4o",
    use_openai_embeddings: bool = True,
    search_by_space_only: bool = False
) -> Dict[str, Any]:
    """
    Answers a user query using hybrid search, rerank, and LLM generation.
    Returns dict with answer and citations.
    
    Args:
        query: The user's question.
        document_id: Filter results to this document ID.
        space_id: Filter results to this space ID.
        acl_tags: Optional list of ACL tags to filter by.
        rerank_top_n: Number of results to rerank.
        system_prompt: System prompt for the LLM.
        groq_model: Model to use with Groq.
        openai_model: Model to use with OpenAI (fallback).
        use_openai_embeddings: Whether to use OpenAI directly for embeddings.
        search_by_space_only: If True, search based on space_id only, ignoring document_id.
    """
    logging.info(f"Answering query: '{query}' for document {document_id if not search_by_space_only else 'None'} in space {space_id}")
    
    # Embed query using either Chonkie API or OpenAI directly
    if use_openai_embeddings:
        query_embedded = embed_query_v2(query)
    else:
        query_embedded = embed_query(query)
    
    # Check for embedding errors
    if "message" in query_embedded and "status" in query_embedded:
        error_msg = f"Query embedding failed: {query_embedded['message']}"
        logging.error(error_msg)
        return {"answer": f"Error: {error_msg}", "citations": []}
    
    query_emb = query_embedded["embedding"]
    query_sparse = query_embedded.get("sparse")

    # print('query_emb', query_emb) 
    # print('query_sparse', query_sparse)
    
    # Set document_id to None if search_by_space_only is True
    doc_id_param = None if search_by_space_only else document_id
    
    # Search using hybrid search
    hits = hybrid_search(
        query_emb=query_emb,
        query_sparse=query_sparse,
        top_k=rerank_top_n,
        doc_id=doc_id_param,
        space_id=space_id,
        acl_tags=acl_tags
    )
    # print('hits', hits)
    
    if not hits:
        return {"answer": "No relevant context found.", "citations": []}
    
    # Log hit sources if source_file is available
    for hit in hits[:3]:
        source = hit.get("metadata", {}).get("source_file", "unknown")
        logging.info(f"Top hit from source: {source}")
    
    reranked = rerank_results(query, hits, top_n=rerank_top_n)

    # print('reranked', reranked)
    
    try:
        print('generating answer with groq')
        answer, citations = generate_answer(query, reranked, system_prompt, model=groq_model)
    except Exception as e:
        logging.warning(f"Groq generation failed, falling back to OpenAI: {e}")
        answer, citations = openai_client.generate_answer(query, reranked, system_prompt, model=openai_model)
    
    return {"answer": answer, "citations": citations}


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
    
    # Get transcript
    transcript_result = wetro_service.get_transcript(youtube_url)
    if not transcript_result.get('success'):
        error_msg = f"Failed to extract transcript: {transcript_result.get('message')}"
        logging.error(error_msg)
        raise ValueError(error_msg)
    
    transcript_text = transcript_result['text']
    print('transcript_text', transcript_text)
    
    # Get video metadata
    yt_api_url = os.getenv('YT_API_URL', 'https://pdf-ocr-staging-production.up.railway.app')
    video_title = wetro_service.get_video_title(youtube_url, yt_api_url)
    thumbnail = f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg"
    
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
        
        # Embed chunks
        logging.info(f"Embedding chunks with OpenAI model: {embedding_model}")
        embeddings = embed_chunks_v2(cleaned_chunks, model=embedding_model, batch_size=32)
        
        # Check for embedding errors
        if embeddings and isinstance(embeddings[0], dict) and "message" in embeddings[0] and "status" in embeddings[0]:
            error_msg = f"OpenAI embedding error: {embeddings[0]['message']}"
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


@app.post("/process_document")
async def process_document_endpoint(
    files: List[UploadFile] = File(...),
    file_urls: str = Form(...),  # JSON string of URLs corresponding to each file
    space_id: str = Form(...),
    use_openai: bool = Form(False)
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
                document_id = process_document_with_openai(temp_file_path, metadata)
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


@app.post("/answer_query")
async def answer_query_endpoint(
    query: str = Form(...),
    document_id: str = Form(...),
    space_id: Optional[str] = Form(None),
    acl_tags: Optional[str] = Form(None),  # Comma-separated
    use_openai_embeddings: bool = Form(False),
    search_by_space_only: bool = Form(False)
) -> JSONResponse:
    """
    Endpoint to answer a query using the RAG pipeline.
    Returns the answer and citations.
    """
    acl_list = [tag.strip() for tag in acl_tags.split(",")] if acl_tags else None
    try:
        result = answer_query(
            query, 
            document_id, 
            space_id=space_id, 
            acl_tags=acl_list, 
            use_openai_embeddings=use_openai_embeddings,
            search_by_space_only=search_by_space_only
        )
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


@app.post("/generate_flashcards")
async def generate_flashcards_endpoint(
    request: FlashcardRequest
) -> JSONResponse:
    """
    Endpoint to generate flashcards from a document.
    
    Args:
        request: FlashcardRequest with document_id, space_id, etc.
        
    Returns:
        JSON response with flashcards or error message.
    """
    logging.info(f"Generate flashcards endpoint called for document: {request.document_id}")
    
    try:
        result = generate_flashcards(
            document_id=request.document_id,
            space_id=request.space_id,
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
    Endpoint to generate flashcards from a document.
    
    Args:
        request: FlashcardRequest with document_id, space_id, etc.
        
    Returns:
        JSON response with flashcards or error message.
    """
    logging.info(f"Re-Generate flashcards endpoint called for document: {request.document_id}")
    
    try:
        result = regenerate_flashcards(
            document_id=request.document_id,
            space_id=request.space_id,
            num_questions=request.num_questions,
            acl_tags=request.acl_tags
        )
        
        return JSONResponse(result)
    except Exception as e:
        logging.exception(f"Error in generate_flashcards_endpoint: {e}")
        return JSONResponse({"error": str(e)}, status_code=500) 