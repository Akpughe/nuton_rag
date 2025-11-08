import os
from typing import Dict, Any, List, Tuple, Optional
from dotenv import load_dotenv
from supabase import create_client, Client
from datetime import datetime
import logging

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
    print('response', response)
    if not response.data or len(response.data) == 0:
        raise Exception(f"Supabase get failed or no content found for document: {document_id}")
    if "id" not in response.data[0]:
        raise Exception(f"Supabase get failed or id not returned: {response}")
    return str(response.data[0]["id"])

def update_generated_content(document_id: str, content: Dict[str, Any]) -> None:
    # print('document_id', document_id)
    # print('content', content)

    supabase.table('generated_content').update({
        'flashcards': content['flashcards'],
        'updated_at': datetime.now().isoformat()
    }).or_(f"pdf_id.eq.{document_id},yt_id.eq.{document_id}").execute()

    print('added to supabase')

def update_generated_content_quiz(document_id: str, content: Dict[str, Any]) -> None:
    # print('document_id', document_id)
    # print('content', content)

    supabase.table('generated_content').update({
        'quiz': content['quiz'],
        'updated_at': datetime.now().isoformat()
    }).or_(f"pdf_id.eq.{document_id},yt_id.eq.{document_id}").execute()

    print('added to supabase')

def insert_flashcard_set(
    content_id: str, 
    flashcards: List[Dict[str, Any]], 
    set_number: int,
    created_by: Optional[str] = None,
    is_shared: bool = False
) -> str:
    """
    Insert or update a batch of flashcards in the 'flashcard_sets' table.
    If a record with the same content_id and set_number exists, it will update the flashcards.
    Otherwise, it will insert a new record.
    
    Args:
        content_id: UUID of the generated_content record
        flashcards: List of flashcard objects to insert
        set_number: Batch number for this set of flashcards
        created_by: UUID of the user who created this set (for ownership tracking)
        is_shared: Boolean indicating if set is shared with all space members or private to creator
        
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
        logging.info(f"Updating existing flashcard set {existing_id} (content_id: {content_id}, set: {set_number})")
        
        update_data = {
            "flashcards": flashcards,
        }
        # Always update created_by if provided (don't use conditional)
        if created_by is not None:
            update_data["created_by"] = created_by
        # Always update is_shared if provided (explicit not None check)
        if is_shared is not None:
            update_data["is_shared"] = is_shared
        
        response = supabase.table("flashcard_sets").update(update_data).eq("id", existing_id).execute()
        
        if not response.data or len(response.data) == 0:
            raise Exception(f"Flashcard set update failed: {response}")
            
        logging.info(f"Flashcard set updated successfully with created_by: {created_by}, is_shared: {is_shared}")
        return existing_id
    else:
        # No existing record, insert a new one
        logging.info(f"Creating new flashcard set (content_id: {content_id}, set: {set_number}, created_by: {created_by}, is_shared: {is_shared})")
        insert_data = {
            "content_id": content_id,
            "flashcards": flashcards,
            "set_number": set_number
        }
        # Always set created_by if provided (don't use conditional)
        if created_by is not None:
            insert_data["created_by"] = created_by
        # Always set is_shared (don't use conditional)
        insert_data["is_shared"] = is_shared if is_shared is not None else False
        
        response = supabase.table("flashcard_sets").insert(insert_data).execute()
        
        if not response.data or "id" not in response.data[0]:
            raise Exception(f"Flashcard set insertion failed: {response}")
        
        logging.info(f"Flashcard set created successfully with id: {response.data[0]['id']}, created_by: {created_by}, is_shared: {is_shared}")
        return str(response.data[0]["id"])

def get_existing_flashcards(content_id: str) -> List[Dict[str, Any]]:
    """
    Retrieves existing flashcards for a given content_id.
    
    Args:
        content_id: The content ID to retrieve flashcards for.
        
    Returns:
        List of flashcard sets with their cards.
    """
    try:
        # First get the generated_content entry to retrieve existing flashcards
        response = supabase.table("generated_content").select("flashcards").eq("id", content_id).execute()
        if not response.data or len(response.data) == 0:
            return []
            
        # Extract flashcards from the response
        content_data = response.data[0]
        existing_flashcards = content_data.get("flashcards", [])
        
        return existing_flashcards
    except Exception as e:
        logging.error(f"Error retrieving existing flashcards: {e}")
        return []

def get_visible_flashcard_sets(content_id: str, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Retrieves flashcard sets with visibility filtering based on user ownership and sharing status.
    
    A user can see:
    - All sets where is_shared=true (shared with all space members)
    - All sets where created_by matches the user_id (their own sets)
    
    Args:
        content_id: The content ID to retrieve flashcards for.
        user_id: Optional UUID of the user requesting the flashcards. If None, only shared sets are returned.
        
    Returns:
        List of visible flashcard sets with their cards.
    """
    try:
        # Query flashcard_sets table with content_id
        response = supabase.table("flashcard_sets")\
            .select("set_number, flashcards, created_by, is_shared")\
            .eq("content_id", content_id)\
            .order("set_number")\
            .execute()
        
        if not response.data or len(response.data) == 0:
            return []
        
        visible_sets = []
        for row in response.data:
            is_shared = row.get("is_shared", False)
            created_by = row.get("created_by")
            
            # Include if shared OR if user is the creator
            if is_shared or (user_id and created_by and created_by == user_id):
                visible_sets.append({
                    "set_id": row.get("set_number"),
                    "cards": row.get("flashcards", []),
                    "created_by": created_by,
                    "is_shared": is_shared
                })
        
        logging.info(f"Retrieved {len(visible_sets)} visible flashcard sets for user {user_id} (total available: {len(response.data)})")
        return visible_sets
        
    except Exception as e:
        logging.error(f"Error retrieving visible flashcard sets: {e}")
        return []

def get_existing_quizzes(content_id: str) -> List[Dict[str, Any]]:
    """
    Retrieves existing quizzes for a given content_id from the quiz_sets table.
    
    Args:
        content_id: The content ID to retrieve quizzes for.
        
    Returns:
        List of quiz sets with their questions.
    """
    try:
        # Query the quiz_sets table for all sets associated with this content_id
        response = supabase.table("quiz_sets").select("quiz, set_number, title, description").eq("content_id", content_id).order("set_number").execute()
        
        if not response.data or len(response.data) == 0:
            return []
            
        # Extract quiz sets from the response
        quiz_sets = []
        for row in response.data:
            quiz_data = row.get("quiz", {})
            if quiz_data and isinstance(quiz_data, dict):
                # Extract questions from the quiz object
                questions = quiz_data.get("questions", [])
                quiz_sets.append({
                    "set_number": row.get("set_number"),
                    "title": row.get("title"),
                    "description": row.get("description"),
                    "questions": questions,
                    "total_questions": len(questions)
                })
        
        return quiz_sets
    except Exception as e:
        logging.error(f"Error retrieving existing quizzes from quiz_sets table: {e}")
        return []

def get_visible_quiz_sets(content_id: str, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Retrieves quiz sets with visibility filtering based on user ownership and sharing status.
    
    A user can see:
    - All sets where is_shared=true (shared with all space members)
    - All sets where created_by matches the user_id (their own sets)
    
    Args:
        content_id: The content ID to retrieve quizzes for.
        user_id: Optional UUID of the user requesting the quizzes. If None, only shared sets are returned.
        
    Returns:
        List of visible quiz sets with their questions.
    """
    try:
        # Query quiz_sets table with content_id
        response = supabase.table("quiz_sets")\
            .select("quiz, set_number, title, description, created_by, is_shared")\
            .eq("content_id", content_id)\
            .order("set_number")\
            .execute()
        
        if not response.data or len(response.data) == 0:
            return []
        
        visible_sets = []
        for row in response.data:
            is_shared = row.get("is_shared", False)
            created_by = row.get("created_by")
            
            # Include if shared OR if user is the creator
            if is_shared or (user_id and created_by and created_by == user_id):
                quiz_data = row.get("quiz", {})
                if quiz_data and isinstance(quiz_data, dict):
                    questions = quiz_data.get("questions", [])
                    visible_sets.append({
                        "set_number": row.get("set_number"),
                        "title": row.get("title"),
                        "description": row.get("description"),
                        "questions": questions,
                        "total_questions": len(questions),
                        "created_by": created_by,
                        "is_shared": is_shared
                    })
        
        logging.info(f"Retrieved {len(visible_sets)} visible quiz sets for user {user_id} (total available: {len(response.data)})")
        return visible_sets
        
    except Exception as e:
        logging.error(f"Error retrieving visible quiz sets: {e}")
        return []

def insert_quiz_set(
    content_id: str, 
    quiz_obj: Dict[str, Any], 
    set_number: int, 
    title: str = None, 
    description: str = None,
    created_by: Optional[str] = None,
    is_shared: bool = False
) -> str:
    """
    Insert or update a quiz in the 'quiz_sets' table.
    If a record with the same content_id and set_number exists, it will update the quiz.
    Otherwise, it will insert a new record.
    
    Args:
        content_id: UUID of the generated_content record
        quiz_obj: Quiz object with questions and metadata
        set_number: Set number for this quiz
        title: Optional title for the quiz
        description: Optional description for the quiz
        created_by: UUID of the user who created this set (for ownership tracking)
        is_shared: Boolean indicating if set is shared with all space members or private to creator
        
    Returns:
        The id of the inserted/updated quiz set as a string
        
    Raises:
        Exception if operation fails or id is not returned
    """
    # First check if a record with this content_id and set_number already exists
    check_response = supabase.table("quiz_sets").select("id").eq("content_id", content_id).eq("set_number", set_number).execute()
    
    if check_response.data and len(check_response.data) > 0:
        # Record exists, update it
        existing_id = check_response.data[0]["id"]
        logging.info(f"Updating existing quiz set {existing_id} (content_id: {content_id}, set: {set_number})")
        
        update_data = {
            "quiz": quiz_obj,
        }
        if title:
            update_data["title"] = title
        if description:
            update_data["description"] = description
        # Always update created_by if provided (don't use conditional)
        if created_by is not None:
            update_data["created_by"] = created_by
        # Always update is_shared if provided
        if is_shared is not None:
            update_data["is_shared"] = is_shared
            
        response = supabase.table("quiz_sets").update(update_data).eq("id", existing_id).execute()
        
        if not response.data or len(response.data) == 0:
            raise Exception(f"Quiz set update failed: {response}")
        
        logging.info(f"Quiz set updated successfully with created_by: {created_by}, is_shared: {is_shared}")
        return existing_id
    else:
        # No existing record, insert a new one
        logging.info(f"Creating new quiz set (content_id: {content_id}, set: {set_number}, created_by: {created_by}, is_shared: {is_shared})")
        insert_data = {
            "content_id": content_id,
            "quiz": quiz_obj,
            "set_number": set_number
        }
        if title:
            insert_data["title"] = title
        if description:
            insert_data["description"] = description
        # Always set created_by if provided (don't use conditional)
        if created_by is not None:
            insert_data["created_by"] = created_by
        # Always set is_shared
        insert_data["is_shared"] = is_shared if is_shared is not None else False
            
        response = supabase.table("quiz_sets").insert(insert_data).execute()
        
        if not response.data or "id" not in response.data[0]:
            raise Exception(f"Quiz set insertion failed: {response}")
        
        logging.info(f"Quiz set created successfully with id: {response.data[0]['id']}, created_by: {created_by}, is_shared: {is_shared}")
            
        return str(response.data[0]["id"])

def determine_shared_status(user_id: str, content_id: str) -> bool:
    """
    Determines if content (flashcards/quiz) should be shared based on ownership.
    
    Returns True if user is space owner (shared with all space members).
    Returns False if user is not owner (private to user only).
    
    Args:
        user_id: UUID of the user creating the content
        content_id: UUID of the generated_content record
        
    Returns:
        bool: True if shared, False if private
    """
    try:
        if not user_id:
            return False  # Safe default if no user provided
        
        # Get space_id from generated_content
        content_response = supabase.table('generated_content')\
            .select('space_id')\
            .eq('id', content_id)\
            .execute()
        
        if not content_response.data or len(content_response.data) == 0:
            logging.warning(f"Could not find generated_content with id {content_id}")
            return False
        
        space_id = content_response.data[0]['space_id']
        
        # Get space owner info
        space_response = supabase.table('spaces')\
            .select('user_id, created_by')\
            .eq('id', space_id)\
            .execute()
        
        if not space_response.data or len(space_response.data) == 0:
            logging.warning(f"Could not find space with id {space_id}")
            return False
        
        space_data = space_response.data[0]
        space_owner_id = space_data.get('user_id') or space_data.get('created_by')
        
        # Check if user is the space owner
        is_owner = (user_id == space_owner_id)
        
        logging.info(f"Ownership check: user={user_id}, owner={space_owner_id}, is_owner={is_owner}")
        
        return is_owner
        
    except Exception as e:
        logging.error(f"Error determining shared status for user {user_id} and content {content_id}: {e}")
        return False  # Safe default on error


# update flashcard

def check_document_type(document_id: str) -> Tuple[str, str]:
    """
    Check if a document_id exists in pdfs or yts table and return its type.
    
    Args:
        document_id: The document ID to check
        
    Returns:
        Tuple of (document_type, document_id) where document_type is "pdf" or "youtube"
        
    Raises:
        Exception if document is not found in either table
    """
    try:
        # Check if document exists in pdfs table
        pdf_response = supabase.table("pdfs").select("id").eq("id", document_id).execute()
        if pdf_response.data and len(pdf_response.data) > 0:
            return ("pdf", document_id)
        
        # Check if document exists in yts table
        yts_response = supabase.table("yts").select("id").eq("id", document_id).execute()
        if yts_response.data and len(yts_response.data) > 0:
            return ("youtube", document_id)
        
        # Document not found in either table
        raise Exception(f"Document {document_id} not found in pdfs or yts table")
        
    except Exception as e:
        logging.error(f"Error checking document type for {document_id}: {e}")
        raise

def upsert_generated_content_notes(document_id: str, notes_markdown: str, space_id: str, is_youtube: bool) -> str:
    """
    Upsert notes to the generated_content table.
    If a record exists (by pdf_id or yts_id), update it. Otherwise, insert a new record.
    
    Args:
        document_id: The document ID (from pdfs or yts table)
        notes_markdown: The generated notes markdown content
        space_id: The space ID
        is_youtube: True if document is a YouTube video, False if PDF
        
    Returns:
        The id of the generated_content record (existing or newly created)
    """
    try:
        # Determine which column to use for lookup and insertion
        if is_youtube:
            lookup_column = "yts_id"
            insert_data = {
                "yts_id": document_id,
                "space_id": space_id,
                "new_note": notes_markdown,
                "updated_at": datetime.now().isoformat()
            }
        else:
            lookup_column = "pdf_id"
            insert_data = {
                "pdf_id": document_id,
                "space_id": space_id,
                "new_note": notes_markdown,
                "updated_at": datetime.now().isoformat()
            }
        
        # Check if record already exists
        check_response = supabase.table("generated_content").select("id").eq(lookup_column, document_id).execute()
        
        if check_response.data and len(check_response.data) > 0:
            # Record exists, update it
            existing_id = check_response.data[0]["id"]
            logging.info(f"Updating existing generated_content record {existing_id} for document {document_id}")
            
            update_data = {
                "new_note": notes_markdown,
                "updated_at": datetime.now().isoformat()
            }
            
            response = supabase.table("generated_content").update(update_data).eq("id", existing_id).execute()
            
            if not response.data or len(response.data) == 0:
                raise Exception(f"Failed to update generated_content record: {response}")
            
            return existing_id
        else:
            # Record doesn't exist, insert new one
            logging.info(f"Creating new generated_content record for document {document_id}")
            insert_data["created_at"] = datetime.now().isoformat()
            
            response = supabase.table("generated_content").insert(insert_data).execute()
            
            if not response.data or "id" not in response.data[0]:
                raise Exception(f"Failed to insert generated_content record: {response}")
            
            return str(response.data[0]["id"])
            
    except Exception as e:
        logging.error(f"Error upserting notes to generated_content for document {document_id}: {e}")
        raise

def get_documents_in_space(space_id: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Get all documents (PDFs and YouTube videos) in a specific space.
    
    Args:
        space_id: The space ID to query documents for
        
    Returns:
        Dictionary with 'pdfs' and 'yts' keys containing lists of document metadata
    """
    try:
        # Get PDF documents in the space
        pdf_response = supabase.table("pdfs").select("id, file_name, file_path, file_type").eq("space_id", space_id).execute()
        pdf_documents = pdf_response.data if pdf_response.data else []
        
        # Get YouTube videos in the space
        yts_response = supabase.table("yts").select("id, file_name, yt_url, thumbnail").eq("space_id", space_id).execute()
        yts_documents = yts_response.data if yts_response.data else []
        
        logging.info(f"Found {len(pdf_documents)} PDFs and {len(yts_documents)} YouTube videos in space {space_id}")
        
        return {
            "pdfs": pdf_documents,
            "yts": yts_documents,
            "total_count": len(pdf_documents) + len(yts_documents)
        }
        
    except Exception as e:
        logging.error(f"Error getting documents in space {space_id}: {e}")
        return {
            "pdfs": [],
            "yts": [],
            "total_count": 0
        }