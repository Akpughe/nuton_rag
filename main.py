import os
from fastapi import File, UploadFile, Form, HTTPException, FastAPI, Body, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, HttpUrl
from typing import List, Optional, Dict, Tuple, Any, Union
import openai
import chromadb
from datetime import datetime
import PyPDF2
import io
from dotenv import load_dotenv
import logging
import json
import time
import httpx
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from starlette.responses import JSONResponse
from starlette.requests import Request
import tempfile
import shutil
from openai import OpenAI
from youtube_transcript_api import YouTubeTranscriptApi
import groq
import sys


from sub import QuizRequest, StreamingQuizResponse, OptimizedStudyGenerator
from pdf_handler import PDFHandler

# Import Pinecone study services
from pinecone_study_service import PineconeStudyGenerator
from pinecone_index_manager import PineconeIndexManager
from optimize_content_processing import OptimizedContentProcessor

# Import our service components
from services.document_processing import MultiFileDocumentProcessor
from services.chunking import MultiFileSemanticChunker
from services.embedding import MultiFileVectorIndexer
from services.reranking import RetrievalEngine
from services.response_generator import ResponseGenerator
from services.legacy_rag import RAGSystem
from services.youtube_processing import YouTubeTranscriptProcessor
# Import our new WetroCloud service
from services.wetrocloud_youtube import WetroCloudYouTubeService

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('app.log')
    ]
)

load_dotenv()

# Initialize the OptimizedStudyGenerator
study_generator = OptimizedStudyGenerator()

# Initialize the Pinecone-based study generator
pinecone_study_generator = PineconeStudyGenerator()
pinecone_index_manager = PineconeIndexManager()

nuton_api = "https://api.nuton.app"

class RAGRequest(BaseModel):
    query: str
    pdfIds: Optional[List[str]] = None
    ytId: Optional[str] = None
    audioIds: Optional[List[str]] = None
    use_knowledge_base: bool = False  # New parameter

class QuizGenerationRequest(BaseModel):
    pdfIds: Optional[List[str]] = None
    ytId: Optional[str] = None

class RAGResponse(BaseModel):
    response: str
    role: str
    page_references: Optional[Dict[str, List[int]]] = None

class PDFEmbeddingRequest(BaseModel):
    pdf_id: str
    text: str

class YTUploadRequest(BaseModel):
    text: str
    space_id: str
    yt_link: HttpUrl
    thumbnail: Optional[str] = None

class YTEmbeddingRequest(BaseModel):
    yt_id: str
    text: str

class GeneralEmbeddingRequest(BaseModel):
    audio_id: str
    text: str

# New integrated RAG models for request and response
class IntegratedRAGRequest(BaseModel):
    query: str
    space_id: str
    document_ids: Optional[List[str]] = None
    top_k: Optional[int] = 5
    use_external_knowledge: Optional[bool] = False

class IntegratedDocumentProcessRequest(BaseModel):
    space_id: str
    file_paths: Optional[List[str]] = None
    file_urls: Optional[List[str]] = None
class IntegratedProcessingResponse(BaseModel):
    status: str
    message: str
    document_ids: Optional[Dict[str, str]] = None
    indexed_chunks: Optional[int] = None
    total_chunks: Optional[int] = None

class YTTranscriptRequest(BaseModel):
    yt_link: str

# Create instances of our service components
document_processor = MultiFileDocumentProcessor()
chunker = MultiFileSemanticChunker()
vector_indexer = MultiFileVectorIndexer()
response_generator = ResponseGenerator()

# FastAPI App
app = FastAPI()
rag_system = RAGSystem()

# Add the custom exception handlers
@app.exception_handler(UnicodeDecodeError)
async def unicode_decode_exception_handler(request: Request, exc: UnicodeDecodeError):
    return JSONResponse(
        status_code=422,
        content={
            "detail": {
                "msg": f"Invalid binary data encountered: {str(exc)}",
                "type": "binary_data_error",
            }
        },
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    try:
        # Try the standard encoder
        detail = jsonable_encoder(exc.errors())
    except UnicodeDecodeError:
        # If that fails due to binary data, use a simplified error
        detail = [
            {
                "loc": ["binary_content"],
                "msg": "Binary data cannot be properly decoded as UTF-8",
                "type": "binary_data_error",
            }
        ]
    
    return JSONResponse(
        status_code=422,
        content={"detail": detail},
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"greeting": "Hello!", "message": "Welcome to Nuton RAG!"}

@app.get("/get-chroma-data")
async def get_chroma_data():
    try:
        # Initialize ChromaDB client
        chroma_client = chromadb.HttpClient(host=os.getenv('CHROMA_DB_CONNECTION_STRING'), port=8000)
        

        # Retrieve data from PDF embeddings
        pdf_collection = chroma_client.get_collection("pdf_embeddings")
        pdf_data = pdf_collection.get()


        # # Retrieve data from audio embeddings
        # audio_collection = chroma_client.get_collection("audio_embeddings")
        # audio_data = audio_collection.get()  # Ensure this is valid

        # Retrieve data from YouTube embeddings
        yt_collection = chroma_client.get_collection("youtube_embeddings")
        yt_data = yt_collection.get()  # Ensure this is valid

        return {
            "pdf_data": pdf_data,
            # "audio_data": audio_data,
            "yt_data": yt_data
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rag")
async def process_rag_request(request: RAGRequest):
    try:
        # Start timing for performance monitoring
        start_time = time.time()
        
        # Generate query embedding
        query_embedding = rag_system.generate_embedding(request.query)

        # Perform similarity search with page references
        search_results, page_refs = rag_system.similarity_search(
            query_embedding,
            request.pdfIds,
            request.ytId,
            request.audioIds
        )

        # Retrieve content
        if not search_results:
            combined_text = rag_system.retrieve_content(
                request.pdfIds, 
                request.ytId,
                request.audioIds
            )
        else:
            combined_text = "\n".join(search_results)

        # Check if we have multiple documents
        multiple_docs = "==== DOCUMENT:" in combined_text and len(request.pdfIds or []) > 1

        # Detect if this is a summary-type query
        summary_keywords = ["summarize", "summary", "key points", "list", "main ideas", 
                           "overview", "highlights", "important", "takeaways", "outline"]
        
        is_summary_query = any(keyword in request.query.lower() for keyword in summary_keywords)
        
        # For non-summary queries, limit the context size for faster processing
        if not is_summary_query:
            # Extract the most relevant section based on keyword matching
            query_terms = set(request.query.lower().split())
            
            # If text is very long, chunk it and find the most relevant chunk
            if len(combined_text) > 3000:
                text_chunks = rag_system.chunk_text(combined_text, max_tokens=3000)
                
                # Score chunks by term overlap with query
                chunk_scores = []
                for chunk in text_chunks:
                    chunk_terms = set(chunk.lower().split())
                    overlap = len(query_terms.intersection(chunk_terms))
                    chunk_scores.append(overlap)
                
                # Use the most relevant chunk (highest term overlap)
                if any(score > 0 for score in chunk_scores):
                    best_chunk_idx = chunk_scores.index(max(chunk_scores))
                    context = text_chunks[best_chunk_idx]
                else:
                    # If no good match, use the first chunk
                    context = text_chunks[0]
            else:
                context = combined_text
                
            # Generate a concise response for non-summary queries
            response = rag_system.generate_response_groq(
                request.query + " (Be concise, limit to 3-5 sentences)", 
                context, 
                request.use_knowledge_base
            )
        else:
            # For summary queries or multiple documents, use a more comprehensive approach
            if len(combined_text) > 6000:
                text_chunks = rag_system.chunk_text(combined_text, max_tokens=6000)
                
                # For summary queries, sample from different parts of the document
                if len(text_chunks) >= 3:
                    context = "\n\n".join([
                        text_chunks[0],
                        text_chunks[len(text_chunks)//2],
                        text_chunks[-1]
                    ])
                else:
                    context = "\n\n".join(text_chunks)
            else:
                context = combined_text
            
            # Add special instruction for comprehensive coverage
            if multiple_docs:
                modified_query = f"{request.query} (Provide a comprehensive response covering key points from ALL documents)"
            else:
                modified_query = f"{request.query} (Provide a comprehensive response covering all key points)"
                
            response = rag_system.generate_response_groq(modified_query, context, request.use_knowledge_base)
        
        # Log performance metrics
        processing_time = time.time() - start_time
        logger.info(f"RAG request processed in {processing_time:.2f} seconds")

        return RAGResponse(
            response=response, 
            role="assistant", 
            page_references=page_refs
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/process-yt-embeddings")
async def process_yt_embeddings(request: YTEmbeddingRequest):
    try:
        # Initialize ChromaDB client
        chroma_client = chromadb.HttpClient(host = os.getenv('CHROMA_DB_CONNECTION_STRING'), port=8000)
        
        # Create or get collection
        collection = chroma_client.get_or_create_collection("youtube_embeddings")
        
        # Text splitting
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=5000, 
            chunk_overlap=1000
        )
        chunks = text_splitter.split_text(request.text)
        
        # Generate embeddings and insert into ChromaDB
        for i, chunk in enumerate(chunks):
            embedding = rag_system.generate_embedding(chunk)
            collection.add(
                ids=[f"{request.yt_id}_{i}"],
                embeddings=[embedding],
                metadatas=[{
                    "yt_id": request.yt_id,
                    "content": chunk,
                    "created_at": datetime.now().isoformat()
                }],
                documents=[chunk]
            )
        
        return {"status": "success", "chunks_processed": len(chunks)}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))  

@app.post("/process-audio-embeddings")
async def process_general_embeddings(request: GeneralEmbeddingRequest):
    try:
        # Initialize ChromaDB client
        chroma_client = chromadb.HttpClient(
            host=os.getenv('CHROMA_DB_CONNECTION_STRING'), port=8000
        )
        
        # Create or get collection for general content
        collection = chroma_client.get_or_create_collection("audio_embeddings")
        
        # Text splitting
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=5000, 
            chunk_overlap=1000
        )
        chunks = text_splitter.split_text(request.text)
        
        # Generate embeddings and insert into ChromaDB
        for i, chunk in enumerate(chunks):
            embedding = rag_system.generate_embedding(chunk)
            collection.add(
                ids=[f"{request.audio_id}_{i}"],
                embeddings=[embedding],
                metadatas=[{
                    "audio_id": request.audio_id,
                    "content": chunk,
                    "created_at": datetime.now().isoformat()
                }],
                documents=[chunk]
            )
        
        return {"status": "success", "chunks_processed": len(chunks)}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))    

@app.post("/upload-yt")
async def upload_yt(request: YTUploadRequest):
    try:
        # Use text splitter to chunk the text
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=15000,  # Consistent with PDF upload 
            chunk_overlap=100
        )
        text_chunks = text_splitter.split_text(request.text)
        
        # Upload YouTube metadata to Supabase
        yt_upload = rag_system.supabase.table('yts').insert({
            'space_id': request.space_id,
            'yt_url': str(request.yt_link),
            'thumbnail': request.thumbnail,
            'extracted_text': request.text
        }).execute()
        
        # Get the inserted YouTube record ID
        yt_id = yt_upload.data[0]['id']
        
        # Process embeddings for chunks
        chroma_client = chromadb.HttpClient(host = os.getenv('CHROMA_DB_CONNECTION_STRING'), port=8000)
        collection = chroma_client.get_or_create_collection("youtube_embeddings")
        
        for i, chunk in enumerate(text_chunks):
            try:
                # Generate embedding for each chunk
                embedding = rag_system.generate_embedding(chunk)
                
                # Add to ChromaDB
                collection.add(
                    ids=[f"{yt_id}_{i}"],
                    embeddings=[embedding],
                    metadatas=[{
                        "yt_id": yt_id,
                        "chunk_index": i,
                        "created_at": datetime.now().isoformat()
                    }],
                    documents=[chunk]
                )
            except Exception as chunk_error:
                print(f"Error processing YouTube chunk {i}: {chunk_error}")
        
        return {
            "status": "success", 
            "yt_id": yt_id, 
            "yt_link": str(request.yt_link), 
            "chunks_processed": len(text_chunks)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))    

pdf_handler = PDFHandler()

@app.post("/upload-document")
async def upload_document(file: UploadFile = File(...), space_id: str = Form(None), file_path: str = Form(None)):
    """
    Handles upload of PDF, PPTX, and DOCX files.
    Automatically detects file type and uses appropriate handler.
    """
    if not file:
        raise HTTPException(
            status_code=400,
            detail="No file provided for upload."
        )
    
    # Get file extension (lowercase)
    try:
        file_extension = file.filename.lower().split('.')[-1]
    except (AttributeError, IndexError):
        raise HTTPException(
            status_code=400,
            detail="Invalid filename format. Please ensure the file has a proper extension."
        )
    
    # Validate file type
    supported_extensions = {'pdf', 'pptx', 'docx', 'doc'}
    if file_extension not in supported_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: '{file_extension}'. Only {', '.join(supported_extensions)} files are supported."
        )
    
    try:
        # Check if file content is valid
        file_content = await file.read()
        if not file_content:
            raise HTTPException(
                status_code=400,
                detail="Empty file content. Please upload a valid file."
            )
        
        # Reset file position for handlers to read it again
        await file.seek(0)
        
        # Process file based on its extension
        if file_extension == 'pdf':
            return await pdf_handler.handle_pdf_upload(file, space_id, rag_system, nuton_api, file_path)
        elif file_extension in ['pptx']:
            return await pdf_handler.handle_pptx_upload(file, space_id, rag_system, nuton_api, file_path)
        else:  # docx or doc
            return await pdf_handler.handle_docx_upload(file, space_id, rag_system, nuton_api, file_path)
            
    except HTTPException:
        # Re-raise HTTP exceptions without modification
        raise
    except Exception as e:
        logger.error(f"Error processing file {file.filename}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing file: {str(e)}"
        )

@app.post("/upload-multiple-documents")
async def upload_multiple_documents(files: List[UploadFile] = File(...), space_id: str = Form(None), file_path: List[str] = Form(None)):
    """
    Handles upload of multiple files of different types (PDF, PPTX, DOCX, DOC).
    Processes each file based on its extension.
    """
    if not files:
        raise HTTPException(
            status_code=400,
            detail="No files provided for upload."
        )
    
    supported_extensions = {'pdf', 'pptx', 'docx', 'doc'}
    results = []
    errors = []
    
    # Initialize file_path list if None
    if file_path is None:
        file_path = [None] * len(files)
    
    # Ensure file_path has the same length as files
    if len(file_path) < len(files):
        file_path.extend([None] * (len(files) - len(file_path)))
    
    for index, file in enumerate(files):
        try:
            # Get file extension (lowercase)
            try:
                file_extension = file.filename.lower().split('.')[-1]
            except (AttributeError, IndexError):
                errors.append({
                    "filename": getattr(file, "filename", "unknown"),
                    "error": "Invalid filename format. Please ensure the file has a proper extension."
                })
                continue
            
            # Validate file type
            if file_extension not in supported_extensions:
                errors.append({
                    "filename": file.filename,
                    "error": f"Unsupported file type: '{file_extension}'. Only {', '.join(supported_extensions)} files are supported."
                })
                continue
            
            # Check if file content is valid
            file_content = await file.read()
            if not file_content:
                errors.append({
                    "filename": file.filename,
                    "error": "Empty file content. Please upload a valid file."
                })
                continue
            
            # Reset file position for handlers to read it again
            await file.seek(0)
            
            # Get the corresponding file path for this file
            current_file_path = file_path[index] if index < len(file_path) else None
            
            # Process file based on its extension
            if file_extension == 'pdf':
                result = await pdf_handler.handle_pdf_upload(file, space_id, rag_system, nuton_api, current_file_path)
            elif file_extension in ['pptx']:
                result = await pdf_handler.handle_pptx_upload(file, space_id, rag_system, nuton_api, current_file_path)
            else:  # docx or doc
                result = await pdf_handler.handle_docx_upload(file, space_id, rag_system, nuton_api, current_file_path)
            
            # Add filename to result for clarity
            result["filename"] = file.filename
            results.append(result)
            
        except Exception as e:
            logger.error(f"Error processing file {getattr(file, 'filename', 'unknown')}: {str(e)}")
            errors.append({
                "filename": getattr(file, "filename", "unknown"),
                "error": str(e)
            })
    
    # Return summary of processed files and errors
    return {
        "status": "completed",
        "successful_uploads": results,
        "failed_uploads": errors,
        "total_files": len(files),
        "successful_count": len(results),
        "failed_count": len(errors)
    }

# Keep the old endpoint for backward compatibility
@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...), space_id: str = Form(None), file_path: str = Form(None)):
    return await upload_document(file, space_id, file_path)

@app.post("/generate-quiz-stream")
async def create_quiz_stream(space_id:str, request: QuizRequest):
    try:
        start_time = time.time()
        content, references = await study_generator.get_relevant_content_parallel(
            request.pdf_ids,
            request.yt_ids,
            request.audio_ids
        )
        logger.info(f"Content retrieval time: {time.time() - start_time:.2f}s")

        if not content:
            raise HTTPException(status_code=404, detail="No content found")

        rag_system.supabase.table('generated_content').update({
            'quiz': []
        }).eq('space_id', space_id).execute()

        async def stream():
            accumulated_questions: List[Dict[Any, Any]] = []

            async for batch in study_generator.generate_questions_stream(
                content,
                request.num_questions,
                request.difficulty,
                request.batch_size
            ):
                # Parse the batch JSON for cleaner logging
                try:
                    batch_data = json.loads(batch)

                    logger.info(f"Streaming batch {batch_data.get('current_batch', 'unknown')}: {json.dumps(batch_data, indent=2)}")

                    # Extract questions from the batch and add to accumulated list
                    if 'questions' in batch_data:
                        accumulated_questions.extend(batch_data['questions'])

                        # Update Supabase with the accumulated questions
                        rag_system.supabase.table('generated_content').update({
                            'quiz': accumulated_questions
                        }).eq('space_id', space_id).execute()
                        
                        logger.info(f"Updated Supabase with {len(accumulated_questions)} total questions")

                except json.JSONDecodeError:
                    logger.error("Failed to decode batch JSON")
                
                yield batch

        return StreamingResponse(stream(), media_type="application/x-ndjson")

    except Exception as e:
        logger.error(f"Error in create_quiz_stream: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-flashcards")
async def create_flashcards(space_id: str, request: QuizRequest):
    try:
        start_time = time.time()
        content, references = await study_generator.get_relevant_content_parallel(
            request.pdf_ids,
            request.yt_ids,
            request.audio_ids
        )
        logger.info(f"Content retrieval time: {time.time() - start_time:.2f}s")

        if not content:
            raise HTTPException(status_code=404, detail="No content found")

        # Update the flashcards column in the generated_content table
        rag_system.supabase.table('generated_content').update({
            'flashcards': []
        }).eq('space_id', space_id).execute()

        async def stream():
            accumulated_flashcards: List[Dict[str, str]] = []

            async for batch in study_generator.generate_flashcards_stream(
                content,
                request.num_questions,  # Assuming num_questions is the number of flashcards to generate
                request.difficulty,
                request.batch_size
            ):
                # Parse the batch JSON for cleaner logging
                try:
                    batch_data = json.loads(batch)

                    logger.info(f"Streaming batch {batch_data.get('current_batch', 'unknown')}: {json.dumps(batch_data, indent=2)}")

                    # Extract flashcards from the batch and add to accumulated list
                    if 'flashcards' in batch_data:
                        accumulated_flashcards.extend(batch_data['flashcards'])

                        # Update Supabase with the accumulated flashcards
                        rag_system.supabase.table('generated_content').update({
                            'flashcards': accumulated_flashcards
                        }).eq('space_id', space_id).execute()
                        
                        logger.info(f"Updated Supabase with {len(accumulated_flashcards)} total flashcards")

                except json.JSONDecodeError:
                    logger.error("Failed to decode batch JSON")
                
                yield batch

        return StreamingResponse(stream(), media_type="application/x-ndjson")

    except Exception as e:
        logger.error(f"Error in create_flashcards: {e}")
        raise HTTPException(status_code=500, detail=str(e))



@app.get("/yt-transcript")
async def get_yt_transcript(request: YTTranscriptRequest):
    try:
        # Initialize WetroCloud YouTube service
        wetrocloud_service = WetroCloudYouTubeService()
        
        # Extract video ID for thumbnail
        video_id = wetrocloud_service.extract_video_id(request.yt_link)
        if not video_id:
            raise HTTPException(status_code=400, detail="Invalid YouTube URL format")
        
        # Get transcript using WetroCloud API
        transcript_result = wetrocloud_service.get_transcript(request.yt_link)
        
        if not transcript_result.get('success'):
            return {
                "status": "error",
                "message": transcript_result.get('message', "Failed to get transcript for video")
            }
        
        # Get thumbnail URL
        thumbnail = f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg"
        
        return {
            "status": "success",
            "video_id": video_id,
            "thumbnail": thumbnail,
            "transcript": transcript_result['text']
        }
    
    except Exception as e:
        logger.error(f"Error extracting YouTube transcript: {str(e)}")
        return {
            "status": "error",
            "message": f"An error occurred during transcript extraction: {str(e)}"
        }

# New endpoint for integrated document processing
@app.post("/integrated/process-documents", response_model=IntegratedProcessingResponse)
async def integrated_process_documents(
    request: Optional[IntegratedDocumentProcessRequest] = None,
    files: Optional[List[UploadFile]] = File(None),
    space_id: Optional[str] = Form(None),
    file_urls: Optional[List[str]] = Form(None)
):
    temp_dir = None
    try:
        # Handle both JSON and form data requests
        if request is not None:
            # JSON request with file paths
            if not request.file_paths:
                return IntegratedProcessingResponse(
                    status="error",
                    message="No file paths provided for processing"
                )
            
            # Process files using paths
            processing_result = document_processor.process_files(request.file_paths, request.space_id, request.file_urls)
        elif files:
            # Form data with file uploads
            if not space_id:
                return IntegratedProcessingResponse(
                    status="error",
                    message="space_id is required when uploading files"
                )
            
            # Create a temporary directory
            temp_dir = tempfile.mkdtemp()
            
            # Save uploaded files to temp directory
            temp_paths = []
            form_file_urls = []  # List to store file URLs from form data
            
            # Convert form-based file_urls to list if it's provided
            if file_urls:
                if isinstance(file_urls, str):
                    form_file_urls = [file_urls]
                else:
                    form_file_urls = file_urls
            
            for i, file in enumerate(files):
                # Handle binary filename
                safe_filename = getattr(file, "filename", "unknown")
                if isinstance(safe_filename, bytes):
                    try:
                        safe_filename = safe_filename.decode('utf-8', errors='replace')
                    except:
                        safe_filename = f"file_{len(temp_paths)}.bin"
                
                # Create safe filename
                safe_filename = PDFHandler.clean_filename(safe_filename)
                temp_path = os.path.join(temp_dir, safe_filename)
                
                # Get the file URL if provided in the form, otherwise generate one
                current_file_url = form_file_urls[i] if i < len(form_file_urls) else f"uploads/{safe_filename}"
                
                # Save file content
                content = await file.read()
                with open(temp_path, "wb") as f:
                    f.write(content)
                
                temp_paths.append(temp_path)
            
            # Process the saved files
            processing_result = document_processor.process_files(temp_paths, space_id, form_file_urls)
        else:
            return IntegratedProcessingResponse(
                status="error",
                message="Either request body with file_paths or file uploads are required"
            )
        
        if not processing_result.get('documents'):
            return IntegratedProcessingResponse(
                status="error",
                message=f"Document processing failed: {processing_result.get('message', 'Unknown error')}"
            )
        
        # 2. Chunk the processed documents
        chunks = chunker.chunk_documents(processing_result['documents'])
        
        # 3. Create embeddings for the chunks and index them
        indexing_result = vector_indexer.embed_and_index_chunks(chunks)
        
        return IntegratedProcessingResponse(
            status="success",
            message=f"Successfully processed {len(processing_result['document_ids'])} documents and indexed {indexing_result['indexed_chunks']} chunks",
            document_ids=processing_result['document_ids'],
            indexed_chunks=indexing_result['indexed_chunks'],
            total_chunks=indexing_result['total_chunks']
        )
    
    except Exception as e:
        logging.error(f"Error in integrated document processing: {str(e)}")
        return IntegratedProcessingResponse(
            status="error",
            message=f"An error occurred during processing: {str(e)}"
        )
    finally:
        # Clean up temp directory
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                logging.error(f"Error cleaning up temporary directory: {e}")

# New endpoint for querying with the integrated components
@app.post("/integrated/query")
async def integrated_query(request: IntegratedRAGRequest):
    """
    Retrieve relevant information and generate a response based on the query.
    Works with all indexed content including documents (PDFs, DOCX, PPTX) and YouTube video transcripts.
    """
    try:
        # Add logging and timing
        start_time = time.time()
        logging.info(f"Processing query: '{request.query}' for space_id: {request.space_id}, document_ids: {request.document_ids}")
        
        # Create retrieval engine - Initialize only once if possible
        retrieval_engine = RetrievalEngine(top_k=request.top_k)
        
        # Create cache key for embedding
        cache_key = f"query_embedding:{request.query}"
        
        # Check if embedding is already in cache
        query_embedding = None
        
        # Try to get embedding from cache
        if hasattr(app, 'embedding_cache') and cache_key in app.embedding_cache:
            query_embedding = app.embedding_cache[cache_key]
            logging.info(f"Using cached query embedding")
        else:
            # Generate embedding for the query using the same model as the indexer
            embedding_model = "text-embedding-ada-002"  # Match the model in MultiFileVectorIndexer
            logging.info(f"Generating query embedding with model: {embedding_model}")
            
            # Create OpenAI client
            client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            
            # Generate embedding with new client
            response = client.embeddings.create(
                model=embedding_model,
                input=request.query
            )
            query_embedding = response.data[0].embedding
            
            # Cache the embedding
            if not hasattr(app, 'embedding_cache'):
                app.embedding_cache = {}
            app.embedding_cache[cache_key] = query_embedding
            
        embedding_time = time.time()
        logging.info(f"Embedding generation took {embedding_time - start_time:.2f} seconds")
        
        # Retrieve and rerank documents (works with both document and YouTube video content)
        logging.info("Retrieving and reranking content from documents and videos...")
        rerank_results = retrieval_engine.retrieve_and_rerank(
            request.query, 
            [{'embedding': query_embedding}],
            document_ids=request.document_ids,
            space_id=request.space_id
        )
        
        retrieval_time = time.time()
        logging.info(f"Content retrieval took {retrieval_time - embedding_time:.2f} seconds")
        
        # Check if we got any results
        if not hasattr(rerank_results, 'results') or not rerank_results.results:
            logging.warning("No results found in Pinecone for this query and filters")
            return {
                "query": request.query,
                "response": "I couldn't find any relevant information to answer your query. Please try a different question or upload more relevant documents.",
                "contexts": [],
                "space_id": request.space_id,
                "document_ids": request.document_ids,
                "sources": []
            }
        
        # Extract the text contexts from reranked results
        logging.info(f"Found {len(rerank_results.results)} results, extracting contexts")
        contexts = []
        for result in rerank_results.results:
            context = result.document
            # Add metadata to each context
            metadata = result.metadata if hasattr(result, 'metadata') else {}
            contexts.append({
                "text": context,
                "metadata": metadata
            })
        
        # Prepare source attribution
        sources = []
        for result in rerank_results.results[:3]:  # Get top 3 sources for citation
            metadata = result.metadata if hasattr(result, 'metadata') else {}
            
            if metadata.get('source_type') == 'youtube_video':
                # YouTube video source
                sources.append({
                    "type": "youtube",
                    "id": metadata.get('video_id'),
                    "url": metadata.get('source_url'),
                    "title": metadata.get('title', f"YouTube Video {metadata.get('youtube_id', '')}"),
                    "thumbnail": metadata.get('thumbnail')
                })
            elif metadata.get('document_id'):
                # Document source
                sources.append({
                    "type": "document",
                    "id": metadata.get('document_id'),
                    "file_type": metadata.get('file_type', 'unknown'),
                    "title": metadata.get('source_file', 'Document')
                })
        
        # Deduplicate sources by id
        unique_sources = []
        seen_ids = set()
        for source in sources:
            if source['id'] not in seen_ids:
                seen_ids.add(source['id'])
                unique_sources.append(source)
        
        # Generate response using the response generator
        logging.info("Generating final response with contexts")
        response_text = response_generator.generate_response(
            request.query,
            [c["text"] for c in contexts],
            use_external_knowledge=request.use_external_knowledge
        )
        
        generation_time = time.time()
        logging.info(f"Response generation took {generation_time - retrieval_time:.2f} seconds")
        logging.info(f"Total query processing time: {generation_time - start_time:.2f} seconds")
        
        return {
            "query": request.query,
            "response": response_text,
            "contexts": contexts[:3],  # Return top 3 contexts for reference, now with metadata
            "space_id": request.space_id,
            "document_ids": request.document_ids,
            "sources": unique_sources
        }
    
    except Exception as e:
        logging.error(f"Error in integrated query: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred during query processing: {str(e)}"
        )

# Request model for Pinecone-based quiz and flashcard generation
class PineconeStudyRequest(BaseModel):
    document_ids: List[str]
    num_questions: int = 15
    difficulty: str = "medium"
    batch_size: int = 3
    space_id: Optional[str] = None

# New model for integrated quiz generation
class IntegratedQuizRequest(BaseModel):
    document_ids: List[str]
    space_id: str
    num_questions: int = 15
    difficulty: str = "medium"
    batch_size: int = 3
    question_types: Optional[List[str]] = None  # ["mcq", "true_false"] or subset
    
# Add a new endpoint for integrated quiz generation
@app.post("/integrated/generate-quiz")
async def integrated_generate_quiz(request: IntegratedQuizRequest, background_tasks: BackgroundTasks):
    """Generate quiz questions using the integrated approach to avoid Pinecone filtering issues"""
    try:
        start_time = time.time()
        logging.info(f"Processing quiz generation for space_id: {request.space_id}, document_ids: {request.document_ids}")
        
        # Process question type preferences
        question_types = request.question_types or ["mcq", "true_false"]  # Default to both if not specified
        logging.info(f"Question types selected: {question_types}")
        
        # Validate question types
        valid_types = ["mcq", "true_false"]
        for q_type in question_types:
            if q_type.lower() not in valid_types:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Invalid question type: {q_type}. Supported types are: {', '.join(valid_types)}"
                )
        
        # Create retrieval engine - Initialize only once if possible
        retrieval_engine = RetrievalEngine(top_k=20)  # Larger top_k for more content
        
        # Generate a semantic query to get relevant content for quiz generation
        semantic_query = "key concepts, definitions, important facts, and core principles"
        logging.info(f"Using semantic query: '{semantic_query}' to retrieve content")
        
        # Generate embedding for the semantic query
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        embedding_response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=semantic_query
        )
        query_embedding = embedding_response.data[0].embedding
        
        embedding_time = time.time()
        logging.info(f"Embedding generation took {embedding_time - start_time:.2f} seconds")
        
        # Retrieve documents
        logging.info("Retrieving relevant content for quiz generation...")
        rerank_results = retrieval_engine.retrieve_and_rerank(
            semantic_query, 
            [{'embedding': query_embedding}],
            document_ids=request.document_ids,
            space_id=request.space_id
        )
        
        retrieval_time = time.time()
        logging.info(f"Content retrieval took {retrieval_time - embedding_time:.2f} seconds")
        
        # Check if we got any results
        if not hasattr(rerank_results, 'results') or not rerank_results.results:
            logging.warning("No content found in Pinecone for these document IDs")
            raise HTTPException(
                status_code=404, 
                detail="No content found for the provided document IDs. Please ensure documents are properly indexed."
            )
        
        # Extract the text from reranked results
        logging.info(f"Found {len(rerank_results.results)} chunks of content")
        content_chunks = []
        for result in rerank_results.results:
            content_chunks.append(result.document)
            
        # Combine content for quiz generation
        combined_content = "\n\n".join(content_chunks)
        
        # If space_id is provided, prepare for saving to Supabase
        if request.space_id:
            try:
                # Clear existing quiz data
                rag_system.supabase.table('generated_content').update({
                    'quiz': [],
                    'updated_at': datetime.now().isoformat()
                }).or_(f"pdf_id.eq.{request.document_ids[0]},yt_id.eq.{request.document_ids[0]}").execute()
            except Exception as e:
                logger.error(f"Error clearing existing quiz data: {e}")

        # Stream back quiz questions
        async def stream():
            accumulated_questions = []
            # Use the pinecone_study_generator's question generation function
            # but with our retrieved content instead of relying on its content retrieval
            async for batch in pinecone_study_generator.generate_questions_stream(
                combined_content,
                request.num_questions,
                request.difficulty,
                request.batch_size,
                question_types=question_types  # Pass question types to the generator
            ):
                try:
                    batch_data = json.loads(batch)
                    logger.info(f"Streaming batch {batch_data.get('current_batch', 'unknown')} with {len(batch_data.get('questions', []))} questions")
                    
                    # Add questions to our collection for saving later
                    if 'questions' in batch_data:
                        accumulated_questions.extend(batch_data.get('questions', []))
                    
                    # If space_id is provided, update Supabase
                    if request.space_id:
                        try:
                            rag_system.supabase.table('generated_content').update({
                                'quiz': accumulated_questions,
                                'updated_at': datetime.now().isoformat()
                            }).or_(f"pdf_id.eq.{request.document_ids[0]},yt_id.eq.{request.document_ids[0]}").execute()
                            logger.info(f"Updated Supabase with {len(accumulated_questions)} total questions")
                        except Exception as e:
                            logger.error(f"Error updating Supabase: {e}")
                    
                    # If this is the last batch, save all questions to database
                    if batch_data.get('is_complete', False):
                        # Use background task to avoid blocking the response
                        background_tasks.add_task(
                            pinecone_study_generator.save_quiz_to_db,
                            request.document_ids, 
                            accumulated_questions
                        )
                        
                except json.JSONDecodeError:
                    logger.error("Failed to decode batch JSON")
                
                yield batch

        # Return streaming response
        return StreamingResponse(stream(), media_type="application/x-ndjson")

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error in integrated quiz generation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/pinecone/generate-flashcards")
async def create_pinecone_flashcards(request: PineconeStudyRequest, background_tasks: BackgroundTasks):
    """Generate flashcards using Pinecone for document retrieval"""
    try:
        start_time = time.time()
        content, references = await pinecone_study_generator.get_relevant_content(request.document_ids)
        logger.info(f"Pinecone content retrieval time: {time.time() - start_time:.2f}s")

        if not content:
            raise HTTPException(status_code=404, detail="No content found for the provided document IDs")

        # If space_id is provided, prepare for saving to Supabase
        if request.space_id:
            try:
                # Clear existing flashcards data
                rag_system.supabase.table('generated_content').update({
                    'flashcards': [],
                    'updated_at': datetime.now().isoformat()
                }).or_(f"pdf_id.eq.{request.document_ids[0]},yt_id.eq.{request.document_ids[0]}").execute()
            except Exception as e:
                logger.error(f"Error clearing existing flashcards data: {e}")

        async def stream():
            accumulated_flashcards = []
            async for batch in pinecone_study_generator.generate_flashcards_stream(
                content,
                request.num_questions,
                request.difficulty,
                request.batch_size
            ):
                try:
                    batch_data = json.loads(batch)
                    logger.info(f"Streaming batch {batch_data.get('current_batch', 'unknown')} with {len(batch_data.get('flashcards', []))} flashcards")
                    
                    # Add flashcards to our collection for saving later
                    if 'flashcards' in batch_data:
                        accumulated_flashcards.extend(batch_data.get('flashcards', []))
                    
                    # If space_id is provided, update Supabase
                    if request.space_id:
                        try:
                            rag_system.supabase.table('generated_content').update({
                                'flashcards': accumulated_flashcards,
                                'updated_at': datetime.now().isoformat()
                            }).or_(f"pdf_id.eq.{request.document_ids[0]},yt_id.eq.{request.document_ids[0]}").execute()
                            logger.info(f"Updated Supabase with {len(accumulated_flashcards)} total flashcards")
                        except Exception as e:
                            logger.error(f"Error updating Supabase: {e}")
                    
                    # If this is the last batch, save all flashcards to database
                    if batch_data.get('is_complete', False):
                        # Use background task to avoid blocking the response
                        background_tasks.add_task(
                            pinecone_study_generator.save_flashcards_to_db,
                            request.document_ids, 
                            accumulated_flashcards
                        )
                        
                except json.JSONDecodeError:
                    logger.error("Failed to decode batch JSON")
                
                yield batch

        return StreamingResponse(stream(), media_type="application/x-ndjson")

    except Exception as e:
        logger.error(f"Error in create_pinecone_flashcards: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/pinecone/index-document")
async def index_document_with_pinecone(
    file: UploadFile = File(...), 
    document_id: Optional[str] = Form(None),
    space_id: Optional[str] = Form(None)
):
    """Index a document using Pinecone for vector storage"""
    try:
        # Read and process the file
        content = await file.read()
        file_extension = file.filename.lower().split('.')[-1]
        
        # Generate a document ID if not provided
        if not document_id:
            document_id = f"{int(time.time())}_{file.filename.replace('.', '_')}"
        
        # Process based on file type
        if file_extension == 'pdf':
            # Process PDF file
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(content))
            text_content = ""
            
            # Extract text from each page
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text_content += page.extract_text() + "\n\n"
                
            # Chunk the text
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100
            )
            chunks = text_splitter.split_text(text_content)
            
            # Format chunks with page metadata
            document_chunks = []
            for i, chunk in enumerate(chunks):
                # Estimate page number based on position in document
                estimated_page = min(i // 2 + 1, len(pdf_reader.pages))
                document_chunks.append({
                    "text": chunk,
                    "page": estimated_page
                })
        else:
            # For other document types, process as plain text
            text_content = content.decode('utf-8', errors='replace')
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100
            )
            chunks = text_splitter.split_text(text_content)
            document_chunks = [{"text": chunk, "page": "1"} for chunk in chunks]
        
        # Index the document chunks to Pinecone
        pinecone_index_manager.index_document_chunks(document_id, document_chunks)
        
        # If space_id is provided, save document metadata to Supabase
        if space_id:
            try:
                rag_system.supabase.table('documents').insert({
                    'id': document_id,
                    'space_id': space_id,
                    'filename': file.filename,
                    'content_type': file.content_type,
                    'indexed_at': datetime.now().isoformat(),
                    'chunk_count': len(document_chunks)
                }).execute()
            except Exception as e:
                logger.error(f"Error saving document metadata to Supabase: {e}")
        
        return {
            "status": "success",
            "message": f"Document indexed successfully with {len(document_chunks)} chunks",
            "document_id": document_id
        }
        
    except Exception as e:
        logger.error(f"Error indexing document with Pinecone: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/pinecone/search-documents")
async def search_documents_with_pinecone(
    query: str,
    document_ids: Optional[List[str]] = None,
    top_k: int = 10
):
    """Search for relevant document chunks using Pinecone"""
    try:
        start_time = time.time()
        
        # Search documents
        results = pinecone_index_manager.search_documents(query, document_ids, top_k)
        
        # Process results
        processed_results = []
        for match in results:
            processed_results.append({
                "text": match.metadata.get("text", ""),
                "document_id": match.metadata.get("document_id", ""),
                "page": match.metadata.get("page", "unknown"),
                "score": match.score
            })
        
        logger.info(f"Pinecone search completed in {time.time() - start_time:.2f}s with {len(processed_results)} results")
        
        return {
            "query": query,
            "results": processed_results,
            "total_results": len(processed_results),
            "search_time_seconds": time.time() - start_time
        }
        
    except Exception as e:
        logger.error(f"Error searching documents with Pinecone: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# New model for integrated flashcard generation
class IntegratedFlashcardRequest(BaseModel):
    document_ids: List[str]
    space_id: str
    num_flashcards: int = 15
    difficulty: str = "medium"
    batch_size: int = 3
    flashcard_types: Optional[List[str]] = None  # Default: ["detailed"] (optional: "basic", "cloze")

@app.post("/integrated/generate-flashcards")
async def integrated_generate_flashcards(request: IntegratedFlashcardRequest, background_tasks: BackgroundTasks):
    """Generate flashcards using the integrated approach to avoid Pinecone filtering issues"""
    try:
        start_time = time.time()
        logging.info(f"Processing flashcard generation for space_id: {request.space_id}, document_ids: {request.document_ids}")
        
        # Process flashcard type preferences - default to only detailed flashcards
        flashcard_types = request.flashcard_types or ["detailed"]  # Default to detailed flashcards only
        logging.info(f"Flashcard types selected: {flashcard_types}")
        
        # Validate flashcard types
        valid_types = ["basic", "detailed", "cloze"]
        for f_type in flashcard_types:
            if f_type.lower() not in valid_types:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Invalid flashcard type: {f_type}. Supported types are: {', '.join(valid_types)}"
                )
        
        # Create retrieval engine - Initialize only once if possible
        retrieval_engine = RetrievalEngine(top_k=20)  # Larger top_k for more content
        
        # Generate a semantic query to get relevant content for flashcard generation
        semantic_query = "key concepts, definitions, important facts, and principles worth memorizing"
        logging.info(f"Using semantic query: '{semantic_query}' to retrieve content")
        
        # Generate embedding for the semantic query
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        embedding_response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=semantic_query
        )
        query_embedding = embedding_response.data[0].embedding
        
        embedding_time = time.time()
        logging.info(f"Embedding generation took {embedding_time - start_time:.2f} seconds")
        
        # Retrieve documents
        logging.info("Retrieving relevant content for flashcard generation...")
        rerank_results = retrieval_engine.retrieve_and_rerank(
            semantic_query, 
            [{'embedding': query_embedding}],
            document_ids=request.document_ids,
            space_id=request.space_id
        )
        
        retrieval_time = time.time()
        logging.info(f"Content retrieval took {retrieval_time - embedding_time:.2f} seconds")
        
        # Check if we got any results
        if not hasattr(rerank_results, 'results') or not rerank_results.results:
            logging.warning("No content found in Pinecone for these document IDs")
            raise HTTPException(
                status_code=404, 
                detail="No content found for the provided document IDs. Please ensure documents are properly indexed."
            )
        
        # Extract the text from reranked results
        logging.info(f"Found {len(rerank_results.results)} chunks of content")
        content_chunks = []
        for result in rerank_results.results:
            content_chunks.append(result.document)
            
        # Combine content for flashcard generation
        combined_content = "\n\n".join(content_chunks)
        
        # If space_id is provided, prepare for saving to Supabase
        if request.space_id:
            try:
                # Clear existing flashcards data
                rag_system.supabase.table('generated_content').update({
                    'flashcards': []
                }).or_(f"pdf_id.eq.{request.document_ids[0]},yt_id.eq.{request.document_ids[0]}").execute()
            except Exception as e:
                logger.error(f"Error clearing existing flashcards data: {e}")

        # Stream back flashcards
        async def stream():
            accumulated_flashcards = []
            # Use the pinecone_study_generator's flashcard generation function
            async for batch in pinecone_study_generator.generate_flashcards_stream(
                combined_content,
                request.num_flashcards,
                request.difficulty,
                request.batch_size,
                flashcard_types=flashcard_types  # Pass flashcard types to the generator
            ):
                try:
                    batch_data = json.loads(batch)
                    logger.info(f"Streaming batch {batch_data.get('current_batch', 'unknown')} with {len(batch_data.get('flashcards', []))} flashcards")
                    
                    # Add flashcards to our collection for saving later
                    if 'flashcards' in batch_data:
                        accumulated_flashcards.extend(batch_data.get('flashcards', []))
                    
                    # If space_id is provided, update Supabase
                    if request.space_id:
                        try:
                            rag_system.supabase.table('generated_content').update({
                                'flashcards': accumulated_flashcards,
                                'updated_at': datetime.now().isoformat()
                            }).or_(f"pdf_id.eq.{request.document_ids[0]},yt_id.eq.{request.document_ids[0]}").execute()
                            logger.info(f"Updated Supabase with {len(accumulated_flashcards)} total flashcards")
                        except Exception as e:
                            logger.error(f"Error updating Supabase: {e}")
                    
                    # If this is the last batch, save all flashcards to database
                    if batch_data.get('is_complete', False):
                        # Use background task to avoid blocking the response
                        background_tasks.add_task(
                            pinecone_study_generator.save_flashcards_to_db,
                            request.document_ids, 
                            accumulated_flashcards
                        )
                        
                except json.JSONDecodeError:
                    logger.error("Failed to decode batch JSON")
                
                yield batch

        return StreamingResponse(stream(), media_type="application/x-ndjson")

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error in integrated flashcard generation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

class ProcessYouTubeVideosRequest(BaseModel):
    video_urls: List[str]
    space_id: str

class YouTubeTranscriptExtractRequest(BaseModel):
    video_url: str

@app.post("/integrated/extract-youtube-transcript")
async def extract_youtube_transcript(request: YouTubeTranscriptExtractRequest):
    """
    Extract transcript from a YouTube video without processing or indexing
    """
    try:
        # Initialize WetroCloud YouTube service
        wetrocloud_service = WetroCloudYouTubeService()
        
        # Extract video ID from URL
        video_id = wetrocloud_service.extract_video_id(request.video_url)
        if not video_id:
            return {
                "status": "error",
                "message": f"Invalid YouTube URL format: {request.video_url}"
            }
        
        # Get video thumbnail
        thumbnail = f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg"
        
        # Get transcript using WetroCloud API
        transcript_result = wetrocloud_service.get_transcript(request.video_url)
        
        if not transcript_result.get('success'):
            return {
                "status": "error",
                "message": transcript_result.get('message', "Failed to get transcript for video")
            }
        
        return {
            "status": "success",
            "video_id": video_id,
            "thumbnail": thumbnail,
            "transcript": transcript_result['text']
        }
    
    except Exception as e:
        logger.error(f"Error extracting YouTube transcript: {str(e)}")
        return {
            "status": "error",
            "message": f"An error occurred during transcript extraction: {str(e)}"
        }

@app.post("/integrated/process-youtube-videos", response_model=IntegratedProcessingResponse)
async def process_youtube_videos(request: ProcessYouTubeVideosRequest):
    """
    Process multiple YouTube videos, extract transcripts, and index them for RAG
    """
    try:
        # Initialize YouTube processor
        youtube_processor = YouTubeTranscriptProcessor()
        
        # Process the videos and extract transcripts
        processing_result = youtube_processor.process_videos(request.video_urls, request.space_id)
        
        if not processing_result.get('documents'):
            return IntegratedProcessingResponse(
                status="error",
                message=f"YouTube video processing failed: {processing_result.get('message', 'Unknown error')}"
            )
        
        # 2. Chunk the processed documents using the same chunker as document processing
        chunks = chunker.chunk_documents(processing_result['documents'])
        
        # Log metadata of first chunk for debugging
        if chunks and len(chunks) > 0:
            logger.info(f"First chunk metadata: {chunks[0].metadata}")
        
        # 3. Create embeddings for the chunks and index them
        indexing_result = vector_indexer.embed_and_index_chunks(chunks)
        
        return IntegratedProcessingResponse(
            status="success",
            message=f"Successfully processed {len(processing_result['document_ids'])} videos and indexed {indexing_result['indexed_chunks']} chunks",
            document_ids=processing_result['document_ids'],  # This was previously video_ids, now using document_ids
            indexed_chunks=indexing_result['indexed_chunks'],
            total_chunks=indexing_result['total_chunks']
        )
    
    except Exception as e:
        logging.error(f"Error in YouTube video processing: {str(e)}")
        return IntegratedProcessingResponse(
            status="error",
            message=f"An error occurred during processing: {str(e)}"
        )

class YouTubeUploadRequest(BaseModel):
    video_urls: List[str]
    space_id: str

@app.post("/integrated/upload-youtube-videos")
async def upload_youtube_videos(request: YouTubeUploadRequest):
    """
    Upload multiple YouTube videos, extract transcripts, and process for RAG.
    Similar to batch document uploading but specifically for YouTube videos.
    """
    if not request.video_urls:
        raise HTTPException(
            status_code=400,
            detail="No video URLs provided for upload."
        )
    
    youtube_processor = YouTubeTranscriptProcessor()
    results = []
    errors = []
    
    # Process each video URL
    for video_url in request.video_urls:
        try:
            # Extract video ID and validate URL
            video_id = youtube_processor._extract_video_id(video_url)
            if not video_id:
                errors.append({
                    "video_url": video_url,
                    "error": "Invalid YouTube URL format"
                })
                continue
            
            # Process single video
            result = youtube_processor.process_single_video(video_url, request.space_id)
            
            if result.get('status') == 'success':
                # Add to successful results
                results.append({
                    "video_url": video_url,
                    "document_id": result.get('document_id'),  # Changed from video_id to document_id
                    "thumbnail": result.get('thumbnail')
                })
                
                # Get the document for chunking and indexing
                if result.get('documents'):
                    # Chunk the document
                    chunks = chunker.chunk_documents(result['documents'])
                    
                    # Log first chunk metadata for debugging
                    if chunks and len(chunks) > 0:
                        logger.info(f"First chunk metadata for {video_url}: {chunks[0].metadata}")
                    
                    # Create embeddings and index chunks
                    index_result = vector_indexer.embed_and_index_chunks(chunks)
                    logger.info(f"Indexing result for {video_url}: {index_result}")
            else:
                errors.append({
                    "video_url": video_url,
                    "error": result.get('message', 'Unknown error')
                })
                
        except Exception as e:
            logger.error(f"Error processing video {video_url}: {str(e)}")
            errors.append({
                "video_url": video_url,
                "error": str(e)
            })
    
    # Return summary of processed videos and errors
    return {
        "status": "completed",
        "successful_uploads": results,
        "failed_uploads": errors,
        "total_videos": len(request.video_urls),
        "successful_count": len(results),
        "failed_count": len(errors)
    }

@app.get("/test/youtube-query/{space_id}/{video_id}")
async def test_youtube_query(space_id: str, video_id: str):
    """
    Test endpoint to verify YouTube videos can be properly queried
    """
    try:
        # Create a simple test query
        test_query = "summarize the video"
        
        # Log test information
        logger.info(f"Running test query '{test_query}' for video_id: {video_id} in space: {space_id}")
        
        # Initialize the retrieval engine
        retrieval_engine = RetrievalEngine(top_k=5)
        
        # Generate embedding for the query
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        embedding_response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=test_query
        )
        query_embedding = embedding_response.data[0].embedding
        
        # Retrieve content 
        rerank_results = retrieval_engine.retrieve_and_rerank(
            test_query, 
            [{'embedding': query_embedding}],
            document_ids=[video_id],
            space_id=space_id
        )
        
        # Check if we got results
        if not hasattr(rerank_results, 'results') or not rerank_results.results:
            logger.warning(f"No results found for video {video_id} in space {space_id}")
            
            # Try querying Pinecone directly to check if the video exists
            vector_indexer = MultiFileVectorIndexer()
            index_stats = vector_indexer.index.describe_index_stats()
            
            # Try querying with metadata filtering only
            filter_query = {
                "document_id": {"$eq": video_id}
            }
            
            try:
                # Direct Pinecone query with metadata filter
                direct_results = vector_indexer.index.query(
                    vector=query_embedding,
                    filter=filter_query,
                    top_k=5,
                    include_metadata=True
                )
                
                return {
                    "status": "direct_query_results",
                    "message": "No results found via RetrievalEngine, but direct Pinecone query found results",
                    "direct_results": direct_results,
                    "index_stats": index_stats
                }
            except Exception as e:
                logger.error(f"Error in direct Pinecone query: {str(e)}")
                
                # Try with a different filter as a last resort
                try:
                    # Try using the video_id field instead
                    video_filter = {
                        "video_id": {"$eq": video_id}
                    }
                    
                    video_results = vector_indexer.index.query(
                        vector=query_embedding,
                        filter=video_filter,
                        top_k=5,
                        include_metadata=True
                    )
                    
                    return {
                        "status": "video_id_query_results",
                        "message": "Found results using video_id filter instead of document_id",
                        "video_results": video_results,
                        "index_stats": index_stats
                    }
                except Exception as e2:
                    logger.error(f"Error in video_id filter query: {str(e2)}")
                
                return {
                    "status": "no_results",
                    "message": "No results found for this video",
                    "index_stats": index_stats,
                    "error": str(e)
                }
            
        # Extract contexts from results
        contexts = []
        for result in rerank_results.results:
            context = result.document
            # Add metadata to each context
            metadata = result.metadata if hasattr(result, 'metadata') else {}
            contexts.append({
                "text": context,
                "metadata": metadata
            })
            
        # Generate response
        response_text = response_generator.generate_response(
            test_query,
            [c["text"] for c in contexts],
            use_external_knowledge=False
        )
        
        return {
            "status": "success",
            "query": test_query,
            "response": response_text,
            "contexts": contexts[:3],  # Return top 3 contexts for reference, now with metadata
            "space_id": request.space_id,
            "document_ids": request.document_ids,
            "sources": []
        }
    
    except Exception as e:
        logger.error(f"Error in test query: {str(e)}", exc_info=True)
        return {
            "status": "error",
            "message": f"An error occurred: {str(e)}"
        }

@app.get("/debug/youtube-video/{video_id}")
async def debug_youtube_video(video_id: str):
    """
    Diagnostic endpoint to check YouTube video processing and indexing
    """
    try:
        # Get video data from Supabase
        youtube_processor = YouTubeTranscriptProcessor()
        
        try:
            supabase_result = youtube_processor.supabase.table('yts').select('*').eq('id', video_id).execute()
            video_data = supabase_result.data[0] if supabase_result.data else None
        except Exception as e:
            logger.error(f"Error getting video from Supabase: {str(e)}")
            video_data = None
            
        # Check Pinecone indexing
        vector_indexer = MultiFileVectorIndexer()
        
        # Try different filters to find the video
        filters = [
            {"document_id": video_id},
            {"video_id": video_id}
        ]
        
        pinecone_results = {}
        
        for i, filter_dict in enumerate(filters):
            filter_name = f"filter_{i+1}"
            try:
                # Query without using an embedding
                # This is just to find if the document exists in the index
                stats = vector_indexer.index.describe_index_stats()
                
                # Create a dummy query vector with the right dimension
                dimension = stats.get('dimension', 1536)
                dummy_vector = [0.0] * dimension
                
                results = vector_indexer.index.query(
                    vector=dummy_vector,
                    filter=filter_dict,
                    top_k=5,
                    include_metadata=True
                )
                
                pinecone_results[filter_name] = {
                    "filter": filter_dict,
                    "count": len(results.get('matches', [])),
                    "sample": results.get('matches', [])[:2]  # First two matches
                }
            except Exception as e:
                logger.error(f"Error checking Pinecone with filter {filter_dict}: {str(e)}")
                pinecone_results[filter_name] = {
                    "filter": filter_dict,
                    "error": str(e)
                }
        
        # Get index stats
        try:
            index_stats = vector_indexer.index.describe_index_stats()
        except Exception as e:
            logger.error(f"Error getting index stats: {str(e)}")
            index_stats = {"error": str(e)}
            
        return {
            "status": "success",
            "video_id": video_id,
            "supabase_data": video_data,
            "pinecone_results": pinecone_results,
            "index_stats": index_stats,
            "suggestions": [
                "If the video exists in Supabase but not in Pinecone, try reprocessing with /integrated/upload-youtube-videos",
                "If filters return different results, there might be inconsistent metadata between document_id and video_id fields",
                "Check the logs for embedding and indexing errors during processing"
            ]
        }
    
    except Exception as e:
        logger.error(f"Error in debug endpoint: {str(e)}", exc_info=True)
        return {
            "status": "error",
            "message": f"An error occurred: {str(e)}"
        }

class ReprocessVideoRequest(BaseModel):
    video_id: str
    space_id: str

@app.post("/debug/reprocess-youtube-video")
async def reprocess_youtube_video(request: ReprocessVideoRequest):
    """
    Reprocess a YouTube video that failed to index properly
    """
    try:
        # Get video data from Supabase
        youtube_processor = YouTubeTranscriptProcessor()
        
        # Get video data
        try:
            supabase_result = youtube_processor.supabase.table('yts').select('*').eq('id', request.video_id).execute()
            
            if not supabase_result.data:
                return {
                    "status": "error",
                    "message": f"Video with ID {request.video_id} not found in Supabase"
                }
                
            video_data = supabase_result.data[0]
            video_url = video_data.get('yt_url')
            
            if not video_url:
                return {
                    "status": "error",
                    "message": f"Video URL not found for video ID {request.video_id}"
                }
                
            logger.info(f"Found video {request.video_id} with URL {video_url}")
            
        except Exception as e:
            logger.error(f"Error getting video from Supabase: {str(e)}")
            return {
                "status": "error",
                "message": f"Error retrieving video data: {str(e)}"
            }
        
        # Create a document from the video data
        extracted_text = video_data.get('extracted_text')
        if not extracted_text:
            return {
                "status": "error",
                "message": "No extracted text found for this video"
            }
            
        # Get YouTube ID from URL
        youtube_id = youtube_processor._extract_video_id(video_url)
        if not youtube_id:
            return {
                "status": "error",
                "message": f"Could not extract YouTube ID from URL: {video_url}"
            }
            
        # Create thumbnail URL
        thumbnail = f"https://img.youtube.com/vi/{youtube_id}/maxresdefault.jpg"
        
        # Create title
        video_title = video_data.get('title', f"YouTube Video: {youtube_id}")
        
        # Create a Document object
        doc = Document(
            page_content=extracted_text,
            metadata={
                'source_url': video_url,
                'document_id': request.video_id,
                'video_id': request.video_id,
                'youtube_id': youtube_id,
                'space_id': request.space_id,
                'content_type': 'youtube_transcript',
                'title': video_title,
                'source': 'youtube',
                'source_type': 'youtube_video',
                'thumbnail': thumbnail,
                'file_type': 'youtube'
            }
        )
        
        # Process the document (chunk and index)
        chunks = chunker.chunk_documents([doc])
        
        if not chunks:
            return {
                "status": "error",
                "message": "Failed to create chunks from video transcript"
            }
            
        # Log first chunk metadata for debugging
        if chunks and len(chunks) > 0:
            logger.info(f"First chunk metadata: {chunks[0].metadata}")
        
        # Index the chunks
        indexing_result = vector_indexer.embed_and_index_chunks(chunks)
        
        return {
            "status": "success",
            "message": f"Successfully reprocessed video {request.video_id}",
            "video_url": video_url,
            "chunks_created": len(chunks),
            "indexing_result": indexing_result
        }
        
    except Exception as e:
        logger.error(f"Error reprocessing video: {str(e)}", exc_info=True)
        return {
            "status": "error",
            "message": f"An error occurred: {str(e)}"
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)