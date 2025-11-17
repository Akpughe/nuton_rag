import os
import logging
import time
import json
import openai
from groq import Groq
import chromadb
from supabase import create_client, Client
from langchain_text_splitters import RecursiveCharacterTextSplitter
from datetime import datetime
from cachetools import TTLCache, LRUCache
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger(__name__)

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
                model="meta-llama/llama-4-scout-17b-16e-instruct",
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