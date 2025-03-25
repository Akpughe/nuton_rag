import os
from fastapi import File, UploadFile, Form, HTTPException, FastAPI, Body
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

from sub import QuizRequest, StreamingQuizResponse, OptimizedStudyGenerator
from pdf_handler import PDFHandler

# Import our service components
from services.document_processing import MultiFileDocumentProcessor
from services.chunking import MultiFileSemanticChunker
from services.embedding import MultiFileVectorIndexer
from services.reranking import RetrievalEngine
from services.response_generator import ResponseGenerator
from services.legacy_rag import RAGSystem

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

load_dotenv()

# Initialize the OptimizedStudyGenerator
study_generator = OptimizedStudyGenerator()

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

class IntegratedProcessingResponse(BaseModel):
    status: str
    message: str
    document_ids: Optional[Dict[str, str]] = None
    indexed_chunks: Optional[int] = None
    total_chunks: Optional[int] = None

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

# New endpoint for integrated document processing
@app.post("/integrated/process-documents", response_model=IntegratedProcessingResponse)
async def integrated_process_documents(
    request: Optional[IntegratedDocumentProcessRequest] = None,
    files: Optional[List[UploadFile]] = File(None),
    space_id: Optional[str] = Form(None)
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
            processing_result = document_processor.process_files(request.file_paths, request.space_id)
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
            for file in files:
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
                
                # Save file content
                content = await file.read()
                with open(temp_path, "wb") as f:
                    f.write(content)
                
                temp_paths.append(temp_path)
            
            # Process the saved files
            processing_result = document_processor.process_files(temp_paths, space_id)
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
        
        # Retrieve and rerank documents
        logging.info("Retrieving and reranking documents...")
        rerank_results = retrieval_engine.retrieve_and_rerank(
            request.query, 
            [{'embedding': query_embedding}],
            document_ids=request.document_ids,
            space_id=request.space_id
        )
        
        retrieval_time = time.time()
        logging.info(f"Document retrieval took {retrieval_time - embedding_time:.2f} seconds")
        
        # Check if we got any results
        if not hasattr(rerank_results, 'results') or not rerank_results.results:
            logging.warning("No results found in Pinecone for this query and filters")
            return {
                "query": request.query,
                "response": "I couldn't find any relevant information to answer your query. Please try a different question or upload more relevant documents.",
                "contexts": [],
                "space_id": request.space_id,
                "document_ids": request.document_ids
            }
        
        # Extract the text contexts from reranked results
        logging.info(f"Found {len(rerank_results.results)} results, extracting contexts")
        contexts = []
        for result in rerank_results.results:
            context = result.document
            contexts.append(context)
            
        # Generate response using the response generator
        logging.info("Generating final response with contexts")
        response_text = response_generator.generate_response(
            request.query,
            contexts,
            use_external_knowledge=request.use_external_knowledge
        )
        
        generation_time = time.time()
        logging.info(f"Response generation took {generation_time - retrieval_time:.2f} seconds")
        logging.info(f"Total query processing time: {generation_time - start_time:.2f} seconds")
        
        return {
            "query": request.query,
            "response": response_text,
            "contexts": contexts[:3],  # Return top 3 contexts for reference
            "space_id": request.space_id,
            "document_ids": request.document_ids
        }
    
    except Exception as e:
        logging.error(f"Error in integrated query: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred during query processing: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8081)