import os
from fastapi import File, UploadFile, Form, HTTPException, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, HttpUrl
from typing import List, Optional, Dict, Tuple
import openai
from groq import Groq
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
import httpx

from sub import QuizRequest, StreamingQuizResponse, OptimizedStudyGenerator
from pdf_handler import PDFHandler


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

class RAGSystem:
    def __init__(self):
        try:
            # Initialize clients
            openai.api_key = os.getenv('OPENAI_API_KEY')
            if not openai.api_key:
                raise ValueError("OpenAI API key is missing")

            self.groq = Groq(
                api_key=os.environ.get("GROQ_API_KEY"),
            )

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

    # def generate_response(self, query: str, context: str):
    #     """Generate response using OpenAI"""
    #     try:
    #         response = openai.chat.completions.create(
    #             model="gpt-3.5-turbo-16k",
    #             messages=[
    #                 {"role": "system", "content": "Using the Feynman technique, answer the question directly, clearly, and concisely. Focus on explaining the key ideas or concepts in a simple way without unnecessary repetition or generalizations."},
    #                 {"role": "user", "content": f"Context: {context}\n\nQuery: {query}"}
    #             ]
    #         )
    #         return response.choices[0].message.content.strip()
    #     except Exception as e:
    #         print(f"Response generation error: {e}")
    #         return "I apologize, but I couldn't generate a response."

    def generate_response_groq(self, query: str, context: str, use_knowledge_base: bool):
        """Generate response using Groq"""
        try:
            if use_knowledge_base:
                # Only check relevance when using knowledge base
                relevance_check_message = (
                    "You are a relevance checker. Determine if the query is related to the context. "
                    "Respond with only 'RELEVANT' or 'NOT_RELEVANT'. "
                    "Example: If context is about AI and query asks about AI models - respond 'RELEVANT'. "
                    "If context is about AI but query asks about cooking recipes - respond 'NOT_RELEVANT'."
                )
                
                relevance_check = self.groq.chat.completions.create(
                    model="mixtral-8x7b-32768",  # Using Mixtral for relevance check
                    messages=[
                        {"role": "system", "content": relevance_check_message},
                        {"role": "user", "content": f"Context: {context}\nQuery: {query}"}
                    ],
                    temperature=0,
                    max_tokens=10
                )
                
                is_relevant = relevance_check.choices[0].message.content.strip() == "RELEVANT"
                
                if not is_relevant:
                    return (
                        "I apologize, but your query appears to be unrelated to the provided context. "
                        "Please ensure your question is relevant to the context provided."
                    )
                
                system_message = (
                    "Answer the following question based on the provided context and external knowledge. "
                    "Keep the response concise yet detailed enough for deep understanding. "
                    "Use plain simple english. "
                    "Use an engaging, conversational, yet formal tone. "
                    "Keep responses under 3 sentences where possible. "
                    "Include only the most relevant technical details. "
                    "Prioritize clarity over comprehensiveness. "
                    "Clearly indicate when you're using information beyond the context"
                )
                
                user_message = (
                    f"Context: {context}\n\n"
                    f"Query: {query}\n\n"
                    "You may reference information beyond the context when helpful, "
                    "but always prioritize the context provided."
                )
            else:
                system_message = (
                    "You are a helpful assistant limited to the provided context only. "
                    "Follow these rules strictly:\n"
                    "1. ONLY use information from the provided context\n"
                    "2. If the context doesn't contain enough information to answer the query, "
                    "state this explicitly\n"
                    "3. Do not add any external knowledge or assumptions\n"
                    "4. If terms are unclear and not explained in the context, "
                    "ask for clarification instead of explaining them"
                )
                
                user_message = (
                    f"Context: {context}\n\n"
                    f"Query: {query}\n\n"
                    "IMPORTANT: Use ONLY the information in the context above. "
                    "Do not add any external knowledge."
                )

            # Generate the response using Groq
            response = self.groq.chat.completions.create(
                model="mixtral-8x7b-32768",  # Using Mixtral model
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.7,
                max_tokens=1024
            )

            # Extract and clean the response
            generated_response = response.choices[0].message.content.strip()

            # Ensure the response stays relevant and avoids hallucinations
            if not use_knowledge_base:
                context_words = set(word.lower() for word in context.split())
                response_words = set(word.lower() for word in generated_response.split())
                
                context_overlap = response_words.intersection(context_words)
                if len(context_overlap) < 2:
                    return (
                        "I apologize, but I couldn't generate a response that stays within "
                        "the bounds of the provided context. Please try rephrasing your query."
                    )
                
            return generated_response

        except Exception as e:
            logger.error(f"Response generation error: {e}")
            return "I apologize, but I couldn't generate a response."
    
    
    def generate_response(self, query: str, context: str, use_knowledge_base: bool):
        """Generate response using OpenAI"""
        try:
            if use_knowledge_base:
                # Only check relevance when using knowledge base
                relevance_check_message = (
                    "You are a relevance checker. Determine if the query is related to the context. "
                    "Respond with only 'RELEVANT' or 'NOT_RELEVANT'. "
                    "Example: If context is about AI and query asks about AI models - respond 'RELEVANT'. "
                    "If context is about AI but query asks about cooking recipes - respond 'NOT_RELEVANT'."
                )
                
                relevance_check = openai.chat.completions.create(
                    model="gpt-3.5-turbo-16k",
                    messages=[
                        {"role": "system", "content": relevance_check_message},
                        {"role": "user", "content": f"Context: {context}\nQuery: {query}"}
                    ],
                    temperature=0,
                    max_tokens=10
                )
                
                is_relevant = relevance_check.choices[0].message.content.strip() == "RELEVANT"
                
                if not is_relevant:
                    return (
                        "I apologize, but your query appears to be unrelated to the provided context. "
                        "Please ensure your question is relevant to the context provided."
                )
                
                system_message = (
                    "Answer the following question based on the provided context and external knowledge. "
                    "Keep the response concise yet detailed enough for deep understanding. "
                    "Use plain simple english."
                    "Use an engaging, conversational, yet formal tone. "
                    "Keep responses under 3 sentences where possible. "
                    "Include only the most relevant technical details. "
                    "Prioritize clarity over comprehensiveness."
                    # "Provide external references if relevant for further learning." 
                    "Clearly indicate when you're using information beyond the context"
                )
                
                user_message = (
                    f"Context: {context}\n\n"
                    f"Query: {query}\n\n"
                    "You may reference information beyond the context when helpful, "
                    "but always prioritize the context provided."
                )
            else:
                system_message = (
                    "You are a helpful assistant limited to the provided context only. "
                    "Follow these rules strictly:\n"
                    "1. ONLY use information from the provided context\n"
                    "2. If the context doesn't contain enough information to answer the query, "
                    "state this explicitly\n"
                    "3. Do not add any external knowledge or assumptions\n"
                    "4. If terms are unclear and not explained in the context, "
                    "ask for clarification instead of explaining them"
                )
                
                user_message = (
                    f"Context: {context}\n\n"
                    f"Query: {query}\n\n"
                    "IMPORTANT: Use ONLY the information in the context above. "
                    "Do not add any external knowledge."
                )

            # Generate the response
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ]
            )

            # Extract and clean the response
            generated_response = response.choices[0].message.content.strip()

            # Ensure the response stays relevant and avoids hallucinations
            if not use_knowledge_base:
                context_words = set(word.lower() for word in context.split())
                response_words = set(word.lower() for word in generated_response.split())
                
                context_overlap = response_words.intersection(context_words)
                if len(context_overlap) < 2:
                    return (
                        "I apologize, but I couldn't generate a response that stays within "
                        "the bounds of the provided context. Please try rephrasing your query."
                    )
                
            return generated_response

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
        response = rag_system.generate_response_groq(request.query, text_chunks[0], request.use_knowledge_base)

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
async def upload_document(file: UploadFile = File(...), space_id: str = Form(None)):
    """
    Handles upload of PDF, PPTX, and DOCX files.
    Automatically detects file type and uses appropriate handler.
    """
    # Get file extension (lowercase)
    file_extension = file.filename.lower().split('.')[-1]
    
    # Validate file type
    supported_extensions = {'pdf', 'pptx', 'docx', 'doc'}
    if file_extension not in supported_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Only {', '.join(supported_extensions)} files are supported."
        )
    
    try:
        if file_extension == 'pdf':
            return await pdf_handler.handle_pdf_upload(file, space_id, rag_system, nuton_api)
        elif file_extension in ['pptx']:
            return await pdf_handler.handle_pptx_upload(file, space_id, rag_system, nuton_api)
        else:  # docx or doc
            return await pdf_handler.handle_docx_upload(file, space_id, rag_system, nuton_api)
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Keep the old endpoint for backward compatibility
@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...), space_id: str = Form(None)):
    return await upload_document(file, space_id)

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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)