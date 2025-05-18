import os
from typing import Dict, Any, List
from dotenv import load_dotenv
from supabase import create_client, Client
from datetime import datetime

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def insert_pdf_record(metadata: Dict[str, Any]) -> str:
    """
    Insert a new record into the 'pdfs' table and return the inserted id.
    Args:
        metadata: Dict of fields to insert (e.g., filename, user, space_id, etc.)
    Returns:
        The id of the inserted row as a string.
    Raises:
        Exception if insertion fails or id is not returned.
    """
    response = supabase.table("pdfs").insert(metadata).execute()
    if not response.data or "id" not in response.data[0]:
        raise Exception(f"Supabase insert failed or id not returned: {response}")
    return str(response.data[0]["id"])

def insert_yts_record(metadata: Dict[str, Any]) -> str:
    """
    Insert a new record into the 'yts' table and return the inserted id.
    Args:
        metadata: Dict of fields to insert (e.g., space_id, yt_url, extracted_text, etc.)
    Returns:
        The id of the inserted row as a string.
    Raises:
        Exception if insertion fails or id is not returned.
    """
    response = supabase.table("yts").insert(metadata).execute()
    if not response.data or "id" not in response.data[0]:
        raise Exception(f"Supabase insert failed or id not returned: {response}")
    return str(response.data[0]["id"])

# get id of generated_content by pdf_id or yt_id
def get_generated_content_id(document_id: str) -> str:
    response = supabase.table('generated_content').select('id').or_(f"pdf_id.eq.{document_id},yt_id.eq.{document_id}").execute()
    if not response.data or "id" not in response.data[0]:
        raise Exception(f"Supabase get failed or id not returned: {response}")
    return str(response.data[0]["id"])

def update_generated_content(document_id: str, content: Dict[str, Any]) -> None:
    print('document_id', document_id)
    print('content', content)

    supabase.table('generated_content').update({
        'flashcards': content['flashcards'],
        'updated_at': datetime.now().isoformat()
    }).or_(f"pdf_id.eq.{document_id},yt_id.eq.{document_id}").execute()

    print('added to supabase')

def insert_flashcard_set(content_id: str, flashcards: List[Dict[str, Any]], set_number: int) -> str:
    """
    Insert or update a batch of flashcards in the 'flashcard_sets' table.
    If a record with the same content_id and set_number exists, it will update the flashcards.
    Otherwise, it will insert a new record.
    
    Args:
        content_id: UUID of the generated_content record
        flashcards: List of flashcard objects to insert
        set_number: Batch number for this set of flashcards
        
    Returns:
        The id of the inserted/updated flashcard set as a string
        
    Raises:
        Exception if operation fails or id is not returned
    """
    # First check if a record with this content_id and set_number already exists
    check_response = supabase.table("flashcard_sets").select("id").eq("content_id", content_id).eq("set_number", set_number).execute()
    
    if check_response.data and len(check_response.data) > 0:
        # Record exists, update it
        existing_id = check_response.data[0]["id"]
        print(f"Updating existing flashcard set {existing_id} (content_id: {content_id}, set: {set_number})")
        
        response = supabase.table("flashcard_sets").update({
            "flashcards": flashcards,
        }).eq("id", existing_id).execute()
        
        if not response.data or len(response.data) == 0:
            raise Exception(f"Flashcard set update failed: {response}")
            
        return existing_id
    else:
        # No existing record, insert a new one
        print(f"Creating new flashcard set (content_id: {content_id}, set: {set_number})")
        response = supabase.table("flashcard_sets").insert({
            "content_id": content_id,
            "flashcards": flashcards,
            "set_number": set_number
        }).execute()
        
        if not response.data or "id" not in response.data[0]:
            raise Exception(f"Flashcard set insertion failed: {response}")
            
        return str(response.data[0]["id"])

# update flashcard