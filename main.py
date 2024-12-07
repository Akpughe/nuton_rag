import os
from fastapi import File, UploadFile, Form, HTTPException, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from typing import List, Optional
import openai
import chromadb
from supabase import create_client, Client
from langchain.text_splitter import RecursiveCharacterTextSplitter
from datetime import datetime
import PyPDF2
import io
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

load_dotenv()


class RAGRequest(BaseModel):
    query: str
    pdfIds: Optional[List[str]] = None
    ytId: Optional[str] = None

class RAGResponse(BaseModel):
    response: str
    role: str

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
            chroma_host = os.getenv('CHROMA_HOST', 'https://thirsty-christian-akpughe-0d8a1b81.koyeb.app')
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

    def similarity_search(self, query_embedding, pdf_ids=None, yt_id=None):
        """Perform similarity search across collections"""
        try:
            # Prepare collections
            pdf_collection = self.chroma_client.get_collection("pdf_embeddings")
            yt_collection = self.chroma_client.get_collection("youtube_embeddings")

            results = []
            
            # PDF search
            if pdf_ids:
                pdf_results = pdf_collection.query(
                    query_embeddings=[query_embedding],
                    where={"pdf_id": {"$in": pdf_ids}},
                    n_results=5
                )
                documents = pdf_results.get('documents', [])
            if documents:
                # Flatten the list of lists into a single list of strings
                results = [doc for sublist in documents for doc in (sublist if isinstance(sublist, list) else [sublist])]

            # YouTube search
            if yt_id:
                yt_results = yt_collection.query(
                    query_embeddings=[query_embedding],
                    where={"yt_id": yt_id},
                    n_results=5
                )
                yt_documents = yt_results.get('documents', [])
                if yt_documents:
                    # Flatten the list of lists into a single list of strings
                    results.extend([doc for sublist in yt_documents for doc in (sublist if isinstance(sublist, list) else [sublist])])    
            
            return results
        
        except Exception as e:
            print(f"Similarity search error: {e}")
            return []

    def retrieve_content(self, pdf_ids=None, yt_id=None):
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
                    {"role": "system", "content": "Use the provided context to answer the query precisely."},
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
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
async def root():
    return {"greeting": "Hello!", "message": "Welcome to Nuton RAG!"}

@app.post("/rag")
async def process_rag_request(request: RAGRequest):
    try:
        # Generate query embedding
        query_embedding = rag_system.generate_embedding(request.query)

        # Perform similarity search
        search_results = rag_system.similarity_search(
            query_embedding, 
            request.pdfIds, 
            request.ytId
        )

        # print(f"result", search_results)

        # Retrieve content
        if not search_results:
            combined_text = rag_system.retrieve_content(
                request.pdfIds, 
                request.ytId
            )
        else:
            combined_text = "\n".join(search_results)

        # Chunk text if too long
        text_chunks = rag_system.chunk_text(combined_text)

        # Generate response
        response = rag_system.generate_response(request.query, text_chunks[0])

        return RAGResponse(response=response, role="assistant")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/process-pdf-embeddings")
async def process_pdf_embeddings(request: PDFEmbeddingRequest):
    try:
        # Initialize ChromaDB client
        chroma_client = chromadb.HttpClient(host='https://thirsty-christian-akpughe-0d8a1b81.koyeb.app', port=8000)
        
        # Create or get collection
        collection = chroma_client.get_or_create_collection("pdf_embeddings")
        
        # Text splitting
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=150000, 
            chunk_overlap=1000
        )
        chunks = text_splitter.split_text(request.text)
        
        # Generate embeddings and insert into ChromaDB
        for i, chunk in enumerate(chunks):
            embedding = generate_embedding(chunk)
            collection.add(
                ids=[f"{request.pdf_id}_{i}"],
                embeddings=[embedding],
                metadatas=[{
                    "pdf_id": request.pdf_id,
                    "content": chunk,
                    "created_at": datetime.now().isoformat()
                }],
                documents=[chunk]
            )
        
        return {"status": "success", "chunks_processed": len(chunks)}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/process-yt-embeddings")
async def process_yt_embeddings(request: YTEmbeddingRequest):
    try:
        # Initialize ChromaDB client
        chroma_client = chromadb.HttpClient(host='https://thirsty-christian-akpughe-0d8a1b81.koyeb.app', port=8000)
        
        # Create or get collection
        collection = chroma_client.get_or_create_collection("youtube_embeddings")
        
        # Text splitting
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=150000, 
            chunk_overlap=1000
        )
        chunks = text_splitter.split_text(request.text)
        
        # Generate embeddings and insert into ChromaDB
        for i, chunk in enumerate(chunks):
            embedding = generate_embedding(chunk)
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

def generate_embedding(text: str):
    """Generate embedding for text"""
    response = openai.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response.data[0].embedding

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...), space_id: str = Form(None)):
    try:
        # Read PDF
        pdf_content = await file.read()
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_content))
        
        # Extract text
        full_text = ""
        for page in pdf_reader.pages:
            full_text += page.extract_text() + "\n"
        
        # Compress and upload PDF to Supabase Storage
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = f"pdf_files/{timestamp}_{file.filename}"
        
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
            chunk_size=150000,  # Reduced chunk size 
            chunk_overlap=100
        )
        text_chunks = text_splitter.split_text(full_text)
        
        # Process embeddings for each chunk
        chroma_client = chromadb.HttpClient(host='https://thirsty-christian-akpughe-0d8a1b81.koyeb.app', port=8000)
        collection = chroma_client.get_or_create_collection("pdf_embeddings")
        
        for i, chunk in enumerate(text_chunks):
            try:
                # Generate embedding for each chunk
                embedding = rag_system.generate_embedding(chunk)
                
                # Add to ChromaDB
                collection.add(
                    ids=[f"{pdf_id}_{i}"],
                    embeddings=[embedding],
                    metadatas=[{
                        "pdf_id": pdf_id,
                        "chunk_index": i,
                        "created_at": datetime.now().isoformat()
                    }],
                    documents=[chunk]
                )
            except Exception as chunk_error:
                print(f"Error processing chunk {i}: {chunk_error}")
        
        return {"status": "success", "pdf_id": pdf_id, "file_url": public_url, "extracted_text": full_text, "chunks_processed": len(text_chunks)}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@app.post("/upload-yt")
async def upload_yt(request: YTUploadRequest):
    try:
        # Use text splitter to chunk the text
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=150000,  # Consistent with PDF upload 
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
        chroma_client = chromadb.HttpClient(host='https://thirsty-christian-akpughe-0d8a1b81.koyeb.app', port=8000)
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
   

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
