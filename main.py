import os
from fastapi import File, UploadFile, Form, HTTPException, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, HttpUrl
from typing import List, Optional, Dict, Tuple
import openai
import chromadb
from supabase import create_client, Client
from langchain.text_splitter import RecursiveCharacterTextSplitter
from datetime import datetime
import PyPDF2
import io
from dotenv import load_dotenv
import logging
import re
import json
import time

from sub import QuizRequest, StreamingQuizResponse, OptimizedStudyGenerator


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

load_dotenv()

# Initialize the OptimizedStudyGenerator
study_generator = OptimizedStudyGenerator()

class RAGRequest(BaseModel):
    query: str
    pdfIds: Optional[List[str]] = None
    ytId: Optional[str] = None
    audioIds: Optional[List[str]] = None
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

class RAGSystem:
    def __init__(self):
        try:
            # Initialize clients
            openai.api_key = os.getenv('OPENAI_API_KEY')
            if not openai.api_key:
                raise ValueError("OpenAI API key is missing")

            self.supabase: Client = create_client(
                os.getenv('SUPABASE_URL'), 
                os.getenv('SUPABASE_KEY')
            )
            
            # ChromaDB configuration from environment
            chroma_host = os.getenv('CHROMA_HOST',  os.getenv('CHROMA_DB_CONNECTION_STRING'))
            chroma_port = int(os.getenv('CHROMA_PORT', 8000))
            self.chroma_client = chromadb.HttpClient(host=chroma_host, port=chroma_port)

        except Exception as e:
            logger.error(f"Initialization error: {e}")
            raise

    def generate_embedding(self, text: str):
        """Generate embedding for text"""
        response = openai.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        return response.data[0].embedding

    def similarity_search(self, query_embedding, pdf_ids=None, yt_id=None, audio_ids=None):
        """Perform similarity search across collections with page tracking"""
        try:
            results = []
            page_references = {}
            
            # PDF search
            if pdf_ids:
                pdf_collection = self.chroma_client.get_collection("pdf_embeddings")
                pdf_results = pdf_collection.query(
                    query_embeddings=[query_embedding],
                    where={"pdf_id": {"$in": pdf_ids}},
                    n_results=5
                )
                
                documents = pdf_results.get('documents', [[]])
                metadatas = pdf_results.get('metadatas', [[]])
                
                # Flatten documents and track page references
                for pdf_metadata_list in metadatas:
                    for metadata in pdf_metadata_list:
                        pdf_id = metadata.get('pdf_id', 'unknown')
                        page = metadata.get('page', 'unknown')
                        
                        if pdf_id not in page_references:
                            page_references[pdf_id] = []
                        
                        if page not in page_references[pdf_id] and page != 'unknown':
                            page_references[pdf_id].append(page)
                
                # Flatten documents
                results = [doc for sublist in documents for doc in (sublist if isinstance(sublist, list) else [sublist])]
            
            if audio_ids:
                audio_collection = self.chroma_client.get_collection("audio_embeddings")
                audio_results = audio_collection.query(
                    query_embeddings=[query_embedding],
                    where={"audio_id": {"$in": audio_ids}},
                    n_results=5
                )

                documents = audio_results.get('documents', [[]])

                if documents:
                    # Flatten the list of lists into a single list of strings
                    results = [doc for sublist in documents for doc in (sublist if isinstance(sublist, list) else [sublist])]
                

            # YouTube search
            if yt_id:
                yt_collection = self.chroma_client.get_collection("youtube_embeddings")

                yt_results = yt_collection.query(
                    query_embeddings=[query_embedding],
                    where={"yt_id": yt_id},
                    n_results=5
                )
                yt_documents = yt_results.get('documents', [])
                if yt_documents:
                    # Flatten the list of lists into a single list of strings
                    results.extend([doc for sublist in yt_documents for doc in (sublist if isinstance(sublist, list) else [sublist])])    
            
            return results, page_references
        
        except Exception as e:
            print(f"Similarity search error: {e}")
            return [], {}

    def retrieve_content(self, pdf_ids=None, yt_id=None, audio_ids=None):
        """Retrieve content from Supabase"""
        combined_text = ""

        # Retrieve PDFs
        if pdf_ids:
            pdfs = self.supabase.table('pdfs').select('extracted_text').in_('id', pdf_ids).execute()
            combined_text += "\n".join([item['extracted_text'] for item in pdfs.data])

        # Retrieve YouTube transcripts
        if yt_id:
            yts = self.supabase.table('yts').select('extracted_text').eq('id', yt_id).execute()
            combined_text += "\n".join([item['extracted_text'] for item in yts.data])

        # Retrieve Audio transcripts
        if audio_ids:
            audio = self.supabase.table('recordings').select('extracted_text').in_('id', audio_ids).execute()
            combined_text += "\n".join([item['extracted_text'] for item in audio.data])

        return combined_text

    def chunk_text(self, text: str, max_tokens: int = 10000):
        """Split text into chunks"""
        chunks = []
        words = text.split()
        current_chunk = []

        for word in words:
            current_chunk.append(word)
            if len(' '.join(current_chunk)) > max_tokens:
                chunks.append(' '.join(current_chunk[:-1]))
                current_chunk = [word]

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks

    def generate_response(self, query: str, context: str):
        """Generate response using OpenAI"""
        try:
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo-16k",
                messages=[
                    {"role": "system", "content": "Using the Feynman technique, answer the question directly, clearly, and concisely. Focus on explaining the key ideas or concepts in a simple way without unnecessary repetition or generalizations."},
                    {"role": "user", "content": f"Context: {context}\n\nQuery: {query}"}
                ]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Response generation error: {e}")
            return "I apologize, but I couldn't generate a response."
        

# FastAPI App
app = FastAPI()
rag_system = RAGSystem()

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

@app.post("/rag")
async def process_rag_request(request: RAGRequest):
    try:
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

        # Chunk text if too long
        text_chunks = rag_system.chunk_text(combined_text)

        # Generate response
        response = rag_system.generate_response(request.query, text_chunks[0])

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
            chunk_size=5000,  # Consistent with PDF upload 
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


def clean_filename(filename):
    """
    Cleans the filename by replacing special characters and spaces with underscores.
    Converts the filename to lowercase.
    """
    # Replace special characters and spaces with underscores
    cleaned_name = re.sub(r'[^a-zA-Z0-9._-]', '_', filename)
    # Convert to lowercase
    cleaned_name = cleaned_name.lower()
    # Replace multiple underscores with a single underscore
    cleaned_name = re.sub(r'_+', '_', cleaned_name)
    return cleaned_name


@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...), space_id: str = Form(None)):
    try:
        # Read PDF
        pdf_content = await file.read()
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_content))
        
        # Extract text with advanced page tracking
        full_text = ""
        page_text_map = []  # Store text and page number for each section
        
        for page_num, page in enumerate(pdf_reader.pages, start=1):
            try:
                # Alternative text extraction methods
                try:
                    # First, try the standard extract_text()
                    page_content = page.extract_text()
                except Exception as standard_extract_error:
                    # If standard extraction fails, try alternative approaches
                    try:
                        # Extract text from page objects more directly
                        page_content = page.get_text()
                    except Exception as alternative_extract_error:
                        # If both fail, log the error and skip this page
                        print(f"Failed to extract text from page {page_num}")
                        print(f"Standard extract error: {standard_extract_error}")
                        print(f"Alternative extract error: {alternative_extract_error}")
                        page_content = ""
                
                # Remove any problematic characters
                page_content = ''.join(char for char in page_content if char.isprintable())
                
                # Store start position, text, and page number
                page_text_map.append({
                    'start': len(full_text),
                    'text': page_content,
                    'page_num': page_num
                })
                full_text += page_content + "\n"
            
            except Exception as page_error:
                print(f"Unexpected error extracting text from page {page_num}: {page_error}")
                traceback.print_exc()  # Print full stack trace for debugging
                continue
        
        # If no text was extracted at all, raise an error
        if not full_text.strip():
            raise ValueError("No text could be extracted from the PDF")
        
        # Compress and upload PDF to Supabase Storage
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cleaned_filename = clean_filename(file.filename)
        file_path = f"pdf_files/{timestamp}_{cleaned_filename}"
        
        # Use the Supabase client from the RAGSystem instance
        storage_upload = rag_system.supabase.storage.from_('pdf_files').upload(
            file_path, 
            pdf_content, 
            file_options={"content-type": file.content_type}
        )
        
        # Get public URL
        public_url = rag_system.supabase.storage.from_('pdf_files').get_public_url(file_path)
        
        # Save PDF metadata to database
        pdf_upload = rag_system.supabase.table('pdfs').insert({
            'space_id': space_id,  # Passed from frontend
            'file_path': public_url,
            'extracted_text': full_text
        }).execute()
        
        # Get the inserted PDF record ID
        pdf_id = pdf_upload.data[0]['id']
        
        # Use text splitter to chunk the text
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=10000,  # Reduced chunk size for more precise page tracking
            chunk_overlap=100
        )
        text_chunks = text_splitter.split_text(full_text)
        
        # Process embeddings for each chunk
        chroma_client = chromadb.HttpClient(host = os.getenv('CHROMA_DB_CONNECTION_STRING'), port=8000)
        collection = chroma_client.get_or_create_collection("pdf_embeddings")
        
        for i, chunk in enumerate(text_chunks):
            try:
                # Determine page for chunk
                page_number = determine_page_for_chunk(chunk, page_text_map)
                
                # Generate embedding for each chunk
                embedding = rag_system.generate_embedding(chunk)
                
                # Add to ChromaDB with page information
                collection.add(
                    ids=[f"{pdf_id}_{i}"],
                    embeddings=[embedding],
                    metadatas=[{
                        "pdf_id": pdf_id,
                        "chunk_index": i,
                        "page": page_number,
                        "created_at": datetime.now().isoformat()
                    }],
                    documents=[chunk]
                )
            except Exception as chunk_error:
                print(f"Error processing chunk {i}: {chunk_error}")
        
        return {
            "status": "success", 
            "pdf_id": pdf_id, 
            "file_url": public_url, 
            "extracted_text": full_text, 
            "chunks_processed": len(text_chunks)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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


def update_quiz (space_id, content) :
    rag_system.supabase.table('generated_content').update(content).eq("space_id",space_id).execute()

def determine_page_for_chunk(chunk: str, page_text_map: List[Dict]) -> int:
    """
    Determine the page number for a given text chunk
    
    Args:
    - chunk: Text chunk to locate
    - page_text_map: List of page text mappings
    
    Returns:
    - Page number where the chunk is most likely located
    """
    # Iterate through pages and check if chunk is in page text
    for page_info in page_text_map:
        # Check if chunk is in the page text
        if chunk in page_info['text']:
            return page_info['page_num']
    
    # If no exact match, return the last page
    return page_text_map[-1]['page_num']

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)