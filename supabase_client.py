import os
from typing import Dict, Any
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

def update_generated_content(document_id: str, content: Dict[str, Any]) -> None:
    print('document_id', document_id)
    print('content', content)

    supabase.table('generated_content').update({
        'flashcards': content['flashcards'],
        'updated_at': datetime.now().isoformat()
    }).or_(f"pdf_id.eq.{document_id},yt_id.eq.{document_id}").execute()

    print('added to supabase')
            


# def update_generated_content(document_id: str, content: Dict[str, Any]) -> None:
#     """
#     Update the generated_content table with content like flashcards.
#     If entry doesn't exist, it will be created.
    
#     Args:
#         document_id: The document ID (from either pdfs or yts table)
#         content: Dictionary of content to update (e.g., flashcards, status, etc.)
        
#     Raises:
#         Exception if the update fails
#     """
#     try:
#         # Check if entry exists
#         check = supabase.table("generated_content") \
#             .select("id") \
#             .or_(f"pdf_id.eq.{document_id},yt_id.eq.{document_id}") \
#             .execute()
        
#         if check.data and len(check.data) > 0:
#             # Update existing entry
#             response = supabase.table("generated_content") \
#                 .update(content) \
#                 .or_(f"pdf_id.eq.{document_id},yt_id.eq.{document_id}") \
#                 .execute()
#         else:
#             # Create new entry
#             # Determine if this is a PDF or YouTube document
#             pdf_check = supabase.table("pdfs").select("id").eq("id", document_id).execute()
            
#             if pdf_check.data and len(pdf_check.data) > 0:
#                 # It's a PDF document
#                 new_entry = {"pdf_id": document_id, **content}
#             else:
#                 # Assume it's a YouTube document
#                 new_entry = {"yt_id": document_id, **content}
                
#             response = supabase.table("generated_content").insert(new_entry).execute()
        
#         # Check if the operation was successful
#         if not response.data:
#             raise Exception(f"No data returned from Supabase operation")
            
#     except Exception as e:
#         raise Exception(f"Failed to update generated content: {str(e)}") 