import os
from fastapi import File, UploadFile, Form, HTTPException, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, HttpUrl
from typing import List, Optional, Dict, Tuple, Any, Union
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
import numpy as np
from cachetools import TTLCache, LRUCache
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi

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
                os.getenv('SUPABASE_URL_DEV'), 
                os.getenv('SUPABASE_KEY_DEV')
            )

            print("supabase url", os.getenv('SUPABASE_URL_DEV'))
            print("supabase key", os.getenv('SUPABASE_KEY_DEV'))
            
            # ChromaDB configuration from environment
            chroma_host = os.getenv('CHROMA_HOST',  os.getenv('CHROMA_DB_CONNECTION_STRING'))
            chroma_port = int(os.getenv('CHROMA_PORT', 8000))
            self.chroma_client = chromadb.HttpClient(host=chroma_host, port=chroma_port)
            
            # Load configuration
            try:
                with open('rag_config.json', 'r') as f:
                    self.config = json.load(f)
                    self.advanced_config = self.config.get('advanced_rag', {})
            except Exception as e:
                logger.warning(f"Could not load config file: {e}. Using default values.")
                self.config = {}
                self.advanced_config = {}
            
            # Initialize caches
            cache_ttl = self.advanced_config.get('embedding_cache_ttl', 3600)
            cache_size = self.advanced_config.get('result_cache_size', 500)
            self.embedding_cache = TTLCache(maxsize=1000, ttl=cache_ttl)
            self.result_cache = LRUCache(maxsize=cache_size)
            
            # Initialize cross-encoder for reranking
            use_reranker = self.advanced_config.get('use_reranker', True)
            reranker_model = self.advanced_config.get('reranker_model', 'cross-encoder/ms-marco-MiniLM-L-6-v2')
            
            if use_reranker:
                try:
                    self.cross_encoder = CrossEncoder(reranker_model)
                    self.use_reranker = True
                except Exception as e:
                    logger.warning(f"Could not initialize cross-encoder: {e}. Reranking will be disabled.")
                    self.use_reranker = False
            else:
                self.use_reranker = False
                
            # Hybrid search weights
            self.use_hybrid = self.advanced_config.get('hybrid_search', True)
            self.bm25_weight = self.advanced_config.get('bm25_weight', 0.3)
            self.vector_weight = self.advanced_config.get('vector_weight', 0.7)
            
            # BM25 index cache
            self.bm25_indices = {}

        except Exception as e:
            logger.error(f"Initialization error: {e}")
            raise

    def generate_embedding(self, text: str):
        """Generate embedding for text with caching"""
        # Create a hash of the text to use as cache key
        cache_key = hash(text)
        
        # Check if embedding is in cache
        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]
        
        # Generate embedding if not in cache
        response = openai.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        embedding = response.data[0].embedding
        
        # Store in cache
        self.embedding_cache[cache_key] = embedding
        
        return embedding

    def build_bm25_index(self, documents):
        """Build a BM25 index from documents"""
        # Tokenize documents
        tokenized_docs = [doc.lower().split() for doc in documents]
        return BM25Okapi(tokenized_docs)
    
    def get_bm25_scores(self, query, documents):
        """Get BM25 scores for documents"""
        # Create a cache key for this document set
        cache_key = hash(tuple(documents))
        
        # Check if we already have a BM25 index for these documents
        if cache_key not in self.bm25_indices:
            self.bm25_indices[cache_key] = self.build_bm25_index(documents)
        
        # Get BM25 scores
        bm25 = self.bm25_indices[cache_key]
        tokenized_query = query.lower().split()
        return bm25.get_scores(tokenized_query)

    def hybrid_search(self, query: str, query_embedding, pdf_ids=None, yt_id=None, audio_ids=None):
        """Perform hybrid search combining vector search with keyword search"""
        try:
            # Create cache key
            cache_key = f"{query}_{str(pdf_ids)}_{yt_id}_{str(audio_ids)}"
            
            # Check cache first
            if cache_key in self.result_cache:
                return self.result_cache[cache_key]
            
            results = []
            page_references = {}
            all_documents = []
            
            # PDF search
            if pdf_ids:
                pdf_collection = self.chroma_client.get_collection("pdf_embeddings")
                
                # Vector search
                pdf_results = pdf_collection.query(
                    query_embeddings=[query_embedding],
                    where={"pdf_id": {"$in": pdf_ids}},
                    n_results=20  # Retrieve more results for hybrid search
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
                vector_results = [doc for sublist in documents for doc in (sublist if isinstance(sublist, list) else [sublist])]
                all_documents.extend(vector_results)
            
            if audio_ids:
                audio_collection = self.chroma_client.get_collection("audio_embeddings")
                audio_results = audio_collection.query(
                    query_embeddings=[query_embedding],
                    where={"audio_id": {"$in": audio_ids}},
                    n_results=20
                )

                documents = audio_results.get('documents', [[]])

                if documents:
                    # Flatten the list of lists into a single list of strings
                    audio_docs = [doc for sublist in documents for doc in (sublist if isinstance(sublist, list) else [sublist])]
                    all_documents.extend(audio_docs)

            # YouTube search
            if yt_id:
                yt_collection = self.chroma_client.get_collection("youtube_embeddings")

                yt_results = yt_collection.query(
                    query_embeddings=[query_embedding],
                    where={"yt_id": yt_id},
                    n_results=20
                )
                yt_documents = yt_results.get('documents', [])
                if yt_documents:
                    # Flatten the list of lists into a single list of strings
                    yt_docs = [doc for sublist in yt_documents for doc in (sublist if isinstance(sublist, list) else [sublist])]
                    all_documents.extend(yt_docs)
            
            # If we have documents and hybrid search is enabled, combine vector and BM25 scores
            if all_documents and self.use_hybrid:
                # Get BM25 scores
                bm25_scores = self.get_bm25_scores(query, all_documents)
                
                # Normalize BM25 scores
                if max(bm25_scores) > 0:
                    bm25_scores = [score / max(bm25_scores) for score in bm25_scores]
                
                # Get vector scores (we don't have them directly, so we'll use position as a proxy)
                # This assumes the vector results are already sorted by relevance
                vector_scores = [1.0 - (i / len(all_documents)) for i in range(len(all_documents))]
                
                # Combine scores
                combined_scores = [
                    (self.bm25_weight * bm25_score) + (self.vector_weight * vector_score)
                    for bm25_score, vector_score in zip(bm25_scores, vector_scores)
                ]
                
                # Sort documents by combined score
                sorted_results = [doc for _, doc in sorted(zip(combined_scores, all_documents), key=lambda x: x[0], reverse=True)]
                
                # Take top results
                results = sorted_results[:10]
            else:
                results = all_documents[:10]
            
            # Rerank results if we have the cross-encoder available
            if self.use_reranker and results:
                # Create pairs of (query, document) for reranking
                pairs = [[query, doc] for doc in results]
                
                # Get scores from cross-encoder
                scores = self.cross_encoder.predict(pairs)
                
                # Sort results by score
                sorted_results = [result for _, result in sorted(zip(scores, results), key=lambda x: x[0], reverse=True)]
                
                # Take top 5 results
                results = sorted_results[:5]
            
            # Cache the results
            self.result_cache[cache_key] = (results, page_references)
            
            return results, page_references
        
        except Exception as e:
            logger.error(f"Hybrid search error: {e}")
            return [], {}

    def similarity_search(self, query_embedding, pdf_ids=None, yt_id=None, audio_ids=None):
        """Legacy similarity search method - kept for backward compatibility"""
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
        """Retrieve content from Supabase with improved handling for multiple PDFs - optimized for speed"""
        combined_text = ""
        pdf_contents = {}
        
        # Retrieve PDFs - optimize by fetching all PDFs in a single query
        if pdf_ids and len(pdf_ids) > 0:
            # Get all PDFs in a single query instead of multiple queries
            pdfs = self.supabase.table('pdfs').select('id, extracted_text').in_('id', pdf_ids).execute()
            
            # Process the results
            if pdfs.data:
                for pdf in pdfs.data:
                    pdf_id = pdf.get('id')
                    pdf_text = pdf.get('extracted_text', '')
                    
                    if pdf_text:
                        # Store the PDF content with its ID for reference
                        pdf_contents[pdf_id] = pdf_text
        
        # If we have multiple PDFs, create a structured combined text
        if len(pdf_contents) > 1:
            # Create a structured document with clear section markers
            for pdf_id, content in pdf_contents.items():
                # Add a clear document separator with PDF ID
                combined_text += f"\n\n==== DOCUMENT: PDF {pdf_id} ====\n\n"
                
                # If content is very long, sample it instead of processing everything
                if len(content) > 10000:
                    # Take samples from beginning, middle, and end
                    intro = content[:1500]
                    middle = content[len(content)//2-750:len(content)//2+750]
                    conclusion = content[-1500:]
                    
                    combined_text += f"{intro}\n\n[...middle content...]\n\n{middle}\n\n[...more content...]\n\n{conclusion}"
                else:
                    combined_text += content
        elif len(pdf_contents) == 1:
            # Just one PDF, use its content directly
            combined_text += list(pdf_contents.values())[0]
        
        # Retrieve YouTube and Audio transcripts in parallel if needed
        yt_content = ""
        audio_content = ""
        
        # Only fetch if IDs are provided
        if yt_id or audio_ids:
            # Use a more efficient approach for the remaining content
            tasks = []
            
            if yt_id:
                yt_content = "\n\n==== YOUTUBE TRANSCRIPT ====\n\n"
                yts = self.supabase.table('yts').select('extracted_text').eq('id', yt_id).execute()
                yt_content += "\n".join([item['extracted_text'] for item in yts.data])
                
            if audio_ids:
                audio_content = "\n\n==== AUDIO TRANSCRIPT ====\n\n"
                audio = self.supabase.table('recordings').select('extracted_text').in_('id', audio_ids).execute()
                audio_content += "\n".join([item['extracted_text'] for item in audio.data])
        
        # Combine all content
        if combined_text and yt_content:
            combined_text += yt_content
        elif yt_content:
            combined_text = yt_content
        
        if combined_text and audio_content:
            combined_text += audio_content
        elif audio_content:
            combined_text = audio_content

        return combined_text

    def chunk_text(self, text: str, max_tokens: int = 3000):
        """Split text into chunks - optimized for speed"""
        # For very short texts, don't bother chunking
        if len(text) < max_tokens:
            return [text]
        
        chunks = []
        # Use a faster approach with approximate token counting
        # Assuming average of 4 chars per token
        approx_chars_per_chunk = max_tokens * 4
        
        # Split by paragraphs first
        paragraphs = text.split('\n\n')
        current_chunk = []
        current_length = 0
        
        for para in paragraphs:
            para_length = len(para)
            
            # If adding this paragraph would exceed the limit
            if current_length + para_length > approx_chars_per_chunk and current_chunk:
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = [para]
                current_length = para_length
            else:
                current_chunk.append(para)
                current_length += para_length
        
        # Add the last chunk if there's anything left
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
        
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
        """Generate response using Groq - optimized for speed and relevance"""
        try:
            # Skip relevance check for non-knowledge base queries to save time
            if use_knowledge_base:
                # Simplified system message
                system_message = (
                    "Answer based on the provided context and external knowledge. "
                    "Be concise and clear. Prioritize information directly relevant to the query."
                )
                
                user_message = (
                    f"Context: {context}\n\n"
                    f"Query: {query}"
                )
            else:
                # Even more simplified system message for context-only queries
                system_message = (
                    "Answer using ONLY the provided context. Be concise and direct. "
                    "Focus only on information directly relevant to the query."
                )
                
                user_message = (
                    f"Context: {context}\n\n"
                    f"Query: {query}"
                )

            # Determine if this is a request for a short response
            is_concise_request = "Be concise" in query or "limit to 2-3 sentences" in query
            
            # Set max tokens based on request type
            max_tokens = 256 if is_concise_request else 768
            
            # Generate the response using Groq with appropriate token limit
            response = self.groq.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.7,
                max_tokens=max_tokens
            )

            return response.choices[0].message.content.strip()

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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)