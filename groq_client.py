import os
from typing import List, Dict, Tuple, Any
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def generate_answer(
    query: str,
    context_chunks: List[Dict],
    system_prompt: str,
    model: str = "meta-llama/llama-4-scout-17b-16e-instruct"
) -> Tuple[str, List[Dict]]:
    """
    Generate an answer using Groq Llama 4, given a query and context chunks.
    Args:
        query: The user query string.
        context_chunks: List of dicts with 'text' and citation info.
        system_prompt: System prompt for LLM.
        model: Model name (default: Llama 4 Scout).
    Returns:
        Tuple of (answer string, list of cited chunks/passages).
    Raises:
        Exception if the API call fails.
    """
    client = Groq(api_key=GROQ_API_KEY)
    
    # Extract text from either direct "text" field or from "metadata.text"
    context_texts = []
    for chunk in context_chunks:
        if "text" in chunk:
            context_texts.append(chunk["text"])
        elif "metadata" in chunk and "text" in chunk["metadata"]:
            context_texts.append(chunk["metadata"]["text"])
        else:
            # Add an empty string as fallback to avoid breaking the joining operation
            context_texts.append("")
    
    context = "\n\n".join(context_texts)
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{query}\n\nContext:\n{context}"}
    ]
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=False
        )
        answer = response.choices[0].message.content
        return answer, context_chunks
    except Exception as e:
        raise Exception(f"Groq Llama 4 failed: {e}")


def generate_answer_document_aware(
    query: str,
    context_chunks: List[Dict],
    space_documents: Dict[str, List[Dict[str, Any]]],
    system_prompt: str,
    model: str = "meta-llama/llama-4-scout-17b-16e-instruct"
) -> Tuple[str, List[Dict]]:
    """
    Generate an answer using Groq with document-aware formatting.
    This version structures the context to show which document each chunk comes from
    and prompts the LLM to provide document-specific insights.
    
    Args:
        query: The user query string.
        context_chunks: List of dicts with 'text' and 'metadata' including 'source_document_id'.
        space_documents: Dict with document metadata from get_documents_in_space().
        system_prompt: System prompt for LLM.
        model: Model name (default: Llama 4 Scout).
        
    Returns:
        Tuple of (answer string, list of cited chunks/passages).
    """
    client = Groq(api_key=GROQ_API_KEY)
    
    # Create a mapping of document IDs to document names for better context
    doc_id_to_name = {}
    
    # Process PDF documents
    for pdf_doc in space_documents.get("pdfs", []):
        doc_id_to_name[pdf_doc["id"]] = {
            "name": pdf_doc.get("file_name", "Unknown PDF"),
            "type": "PDF"
        }
    
    # Process YouTube documents
    for yt_doc in space_documents.get("yts", []):
        doc_id_to_name[yt_doc["id"]] = {
            "name": yt_doc.get("file_name", "Unknown Video"),
            "type": "YouTube Video"
        }
    
    # Group chunks by document
    chunks_by_document = {}
    for chunk in context_chunks:
        doc_id = chunk.get("metadata", {}).get("source_document_id", "unknown")
        if doc_id not in chunks_by_document:
            chunks_by_document[doc_id] = []
        chunks_by_document[doc_id].append(chunk)
    
    # Build document-aware context
    document_sections = []
    for doc_id, doc_chunks in chunks_by_document.items():
        doc_info = doc_id_to_name.get(doc_id, {"name": f"Document {doc_id}", "type": "Unknown"})
        doc_name = doc_info["name"]
        doc_type = doc_info["type"]
        
        # Extract text from chunks
        chunk_texts = []
        for chunk in doc_chunks:
            if "text" in chunk:
                chunk_texts.append(chunk["text"])
            elif "metadata" in chunk and "text" in chunk["metadata"]:
                chunk_texts.append(chunk["metadata"]["text"])
        
        if chunk_texts:
            combined_text = "\n\n".join(chunk_texts)
            document_sections.append(f"=== {doc_type}: {doc_name} ===\n{combined_text}")
    
    # Create the structured context
    structured_context = "\n\n" + "="*50 + "\n\n".join(document_sections)
    
    # Enhanced system prompt for document-aware responses
    enhanced_system_prompt = f"""{system_prompt}

When responding, please:
1. Analyze information from ALL available documents
2. If relevant, structure your response to show insights from different sources
3. Use document names when referencing specific information
4. For broad questions (like "summarize this space"), provide a comprehensive overview covering all documents
5. For specific questions, draw from the most relevant sources but mention if other documents have related information

The context below is organized by document source to help you provide a comprehensive, well-structured response."""
    
    messages = [
        {"role": "system", "content": enhanced_system_prompt},
        {"role": "user", "content": f"{query}\n\nDocument-Organized Context:{structured_context}"}
    ]
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=False
        )
        answer = response.choices[0].message.content
        return answer, context_chunks
    except Exception as e:
        raise Exception(f"Groq document-aware generation failed: {e}") 