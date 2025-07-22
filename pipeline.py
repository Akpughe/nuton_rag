from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any, Optional
import os
import shutil
import logging
import tempfile
import json
import time
from functools import lru_cache

from chonkie_client import chunk_document, embed_chunks, embed_chunks_v2, embed_query, embed_query_v2
from pinecone_client import upsert_vectors, hybrid_search, hybrid_search_parallel, rerank_results, hybrid_search_document_aware, rerank_results_document_aware
from supabase_client import insert_pdf_record, insert_yts_record, get_documents_in_space
from groq_client import generate_answer, generate_answer_document_aware
import openai_client
from services.wetrocloud_youtube import WetroCloudYouTubeService
from flashcard_process import generate_flashcards, regenerate_flashcards
from pydantic import BaseModel
from typing import Optional, List
from quiz_process import generate_quiz, regenerate_quiz

from prompts import main_prompt, general_knowledge_prompt, simple_general_knowledge_prompt, no_docs_in_space_prompt, no_relevant_in_scope_prompt, additional_space_only_prompt
from intelligent_enrichment import create_enriched_system_prompt
from enrichment_examples import create_few_shot_enhanced_prompt
from enhanced_prompts import get_domain_from_context
from websearch_client import analyze_document_context, generate_contextual_search_queries, perform_contextual_websearch, synthesize_rag_and_web_results


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

logging.basicConfig(level=logging.INFO)


class FlashcardRequest(BaseModel):
    document_id: str
    space_id: Optional[str] = None
    num_questions: Optional[int] = None
    acl_tags: Optional[List[str]] = None


class QuizRequest(BaseModel):
    document_id: str
    space_id: Optional[str] = None
    question_type: str = "both"
    num_questions: int = 30
    acl_tags: Optional[str] = None
    rerank_top_n: int = 50
    use_openai_embeddings: bool = True
    set_id: int = 1
    title: Optional[str] = None
    description: Optional[str] = None


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
        # print('chunks', chunks)
        
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


# Add caching for query embeddings
@lru_cache(maxsize=100)
def get_query_embedding(query: str, use_openai: bool = True):
    """Cache query embeddings to avoid recalculating for repeated queries."""
    start_time = time.time()
    if use_openai:
        result = embed_query_v2(query)
    else:
        result = embed_query(query)
    logging.info(f"Query embedding took {time.time() - start_time:.2f}s")
    return result


def answer_query(
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
    educational_mode: bool = False  # Enable tutoring/educational approach
) -> Dict[str, Any]:
    """
    Optimized function to answer a user query using hybrid search, rerank, and LLM generation.
    Returns dict with answer and citations.
    
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
    """
    start_time = time.time()
    logging.info(f"Answering query: '{query}' for document {document_id if not search_by_space_only else 'None'} in space {space_id}, allow_general_knowledge: {allow_general_knowledge}, enable_websearch: {enable_websearch}, enrichment_mode: {enrichment_mode}, learning_style: {learning_style}, educational_mode: {educational_mode}")
    
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
                    answer, citations = openai_client.generate_answer(query, [], fallback_prompt, model=effective_model)
                else:
                    try:
                        answer, citations = generate_answer(query, [], fallback_prompt, model=effective_model)
                    except Exception as e:
                        logging.warning(f"Groq generation failed for general knowledge fallback, trying OpenAI: {e}")
                        answer, citations = openai_client.generate_answer(query, [], fallback_prompt, model=openai_model)
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
                    answer, citations = openai_client.generate_answer(query, [], fallback_prompt, model=effective_model)
                else:
                    try:
                        answer, citations = generate_answer(query, [], fallback_prompt, model=effective_model)
                    except Exception as e:
                        logging.warning(f"Groq generation failed for general knowledge fallback, trying OpenAI: {e}")
                        answer, citations = openai_client.generate_answer(query, [], fallback_prompt, model=openai_model)
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
            
            answer, citations = openai_client.generate_answer(query, enhanced_context, system_prompt, model=effective_model)
        else:
            try:
                answer, citations = generate_answer_document_aware(
                    query=query,
                    context_chunks=limited_context,
                    space_documents=space_documents,
                    system_prompt=system_prompt,
                    model=effective_model
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
                
                answer, citations = openai_client.generate_answer(query, enhanced_context, system_prompt, model=openai_model)
        
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
                    answer, citations = openai_client.generate_answer(query, [], fallback_prompt, model=effective_model)
                else:
                    try:
                        answer, citations = generate_answer(query, [], fallback_prompt, model=effective_model)
                    except Exception as e:
                        logging.warning(f"Groq generation failed for general knowledge fallback, trying OpenAI: {e}")
                        answer, citations = openai_client.generate_answer(query, [], fallback_prompt, model=openai_model)
                return {"answer": answer, "citations": citations}
            else:
                return {"answer": "No relevant context found.", "citations": []}
        
        # Rerank results (faster with optimized parameters)
        rerank_start = time.time()
        reranked = rerank_results(query, hits, top_n=rerank_top_n)
        logging.info(f"Reranking took {time.time() - rerank_start:.2f}s")
        
        # Limit context to the top max_context_chunks chunks to reduce LLM input size
        limited_context = reranked[:max_context_chunks]
        
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
            answer, citations = openai_client.generate_answer(query, limited_context, system_prompt, model=effective_model)
        else:
            try:
                answer, citations = generate_answer(query, limited_context, system_prompt, model=effective_model)
            except Exception as e:
                logging.warning(f"Groq generation failed, falling back to OpenAI: {e}")
                answer, citations = openai_client.generate_answer(query, limited_context, system_prompt, model=openai_model)
        
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

            # Analyze document context to understand domain and guide search
            context_analysis = analyze_document_context(rag_context, query)

            # Generate targeted search queries based on document context
            search_queries = generate_contextual_search_queries(query, context_analysis)

            # Perform contextual web search
            web_results = perform_contextual_websearch(search_queries)
            
            if web_results:
                # Synthesize RAG results with web search results using GPT-4o
                synthesized_answer, combined_sources = synthesize_rag_and_web_results(
                    query=query,
                    rag_context=rag_context,
                    web_results=web_results,
                    context_analysis=context_analysis,
                    system_prompt=system_prompt,
                    has_general_knowledge=allow_general_knowledge
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
    # print('transcript_text', transcript_text)
    
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
    use_openai_embeddings: bool = Form(True),
    search_by_space_only: bool = Form(False),
    rerank_top_n: Optional[int] = Form(10),
    max_context_chunks: Optional[int] = Form(5),
    fast_mode: bool = Form(False),  # New parameter for faster but potentially lower quality results
    allow_general_knowledge: bool = Form(False),  # New parameter for allowing general knowledge supplementation
    enable_websearch: bool = Form(False),  # New parameter for enabling contextual web search
    model: str = Form("meta-llama/llama-4-scout-17b-16e-instruct"),  # User-selectable model parameter
    enrichment_mode: str = Form("simple"),  # "simple" (default) or "advanced" for intelligent enrichment
    learning_style: Optional[str] = Form(None),  # Learning style for personalized educational responses
    educational_mode: bool = Form(False)  # Enable tutoring/educational approach
) -> JSONResponse:
    """
    Optimized endpoint to answer a query using the RAG pipeline.
    Returns the answer and citations.
    
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
        model: The model to use for generation. Auto-switches to GPT-4o when websearch is enabled.
        enrichment_mode: "simple" (default, classic enrichment) or "advanced" (intelligent domain-aware enrichment)
        learning_style: Learning style for personalized educational responses ("academic_focus", "deep_dive", "quick_practical", "exploratory_curious", "narrative_reader", "default", or None)
        educational_mode: If True, enables tutoring/educational approach with context-rich responses
    """
    start_time = time.time()
    acl_list = [tag.strip() for tag in acl_tags.split(",")] if acl_tags else None
    
    # If fast_mode is enabled, adjust parameters for faster response
    if fast_mode:
        rerank_top_n = min(rerank_top_n, 5)  # Limit reranking in fast mode
        max_context_chunks = min(max_context_chunks, 3)  # Use fewer chunks in context
    
    try:
        result = answer_query(
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
            educational_mode=educational_mode
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)