"""
Backfill script to populate created_by field for flashcard_sets and quiz_sets
that have NULL values.

This script identifies orphaned flashcard/quiz sets and attempts to assign them
to the space owner (as a safe default for initially-created sets).
"""

import os
from dotenv import load_dotenv
from supabase import create_client, Client
import logging

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def backfill_flashcard_created_by():
    """
    Backfill created_by for flashcard_sets with NULL values.
    Assigns them to the space owner based on the generated_content -> space_id relationship.
    """
    try:
        # Find all flashcard_sets with NULL created_by
        response = supabase.table("flashcard_sets")\
            .select("id, content_id, set_number, created_by")\
            .is_("created_by", "null")\
            .execute()
        
        if not response.data:
            logger.info("No flashcard_sets with NULL created_by found.")
            return
        
        logger.info(f"Found {len(response.data)} flashcard_sets with NULL created_by")
        
        for row in response.data:
            flashcard_set_id = row["id"]
            content_id = row["content_id"]
            set_number = row["set_number"]
            
            # Get the generated_content to find space_id
            content_response = supabase.table("generated_content")\
                .select("space_id")\
                .eq("id", content_id)\
                .execute()
            
            if not content_response.data:
                logger.warning(f"No generated_content found for content_id: {content_id}")
                continue
            
            space_id = content_response.data[0]["space_id"]
            
            # Get the space owner
            space_response = supabase.table("spaces")\
                .select("user_id, created_by")\
                .eq("id", space_id)\
                .execute()
            
            if not space_response.data:
                logger.warning(f"No space found for space_id: {space_id}")
                continue
            
            space_data = space_response.data[0]
            owner_id = space_data.get("user_id") or space_data.get("created_by")
            
            if not owner_id:
                logger.warning(f"Could not determine owner for space_id: {space_id}")
                continue
            
            # Update the flashcard_set with the owner_id
            update_response = supabase.table("flashcard_sets")\
                .update({"created_by": owner_id})\
                .eq("id", flashcard_set_id)\
                .execute()
            
            logger.info(f"Updated flashcard_set {flashcard_set_id} (set#{set_number}) "
                       f"with created_by={owner_id}")
        
        logger.info("Flashcard backfill completed!")
        
    except Exception as e:
        logger.error(f"Error backfilling flashcard_sets: {e}")
        raise


def backfill_quiz_created_by():
    """
    Backfill created_by for quiz_sets with NULL values.
    Assigns them to the space owner based on the generated_content -> space_id relationship.
    """
    try:
        # Find all quiz_sets with NULL created_by
        response = supabase.table("quiz_sets")\
            .select("id, content_id, set_number, created_by")\
            .is_("created_by", "null")\
            .execute()
        
        if not response.data:
            logger.info("No quiz_sets with NULL created_by found.")
            return
        
        logger.info(f"Found {len(response.data)} quiz_sets with NULL created_by")
        
        for row in response.data:
            quiz_set_id = row["id"]
            content_id = row["content_id"]
            set_number = row["set_number"]
            
            # Get the generated_content to find space_id
            content_response = supabase.table("generated_content")\
                .select("space_id")\
                .eq("id", content_id)\
                .execute()
            
            if not content_response.data:
                logger.warning(f"No generated_content found for content_id: {content_id}")
                continue
            
            space_id = content_response.data[0]["space_id"]
            
            # Get the space owner
            space_response = supabase.table("spaces")\
                .select("user_id, created_by")\
                .eq("id", space_id)\
                .execute()
            
            if not space_response.data:
                logger.warning(f"No space found for space_id: {space_id}")
                continue
            
            space_data = space_response.data[0]
            owner_id = space_data.get("user_id") or space_data.get("created_by")
            
            if not owner_id:
                logger.warning(f"Could not determine owner for space_id: {space_id}")
                continue
            
            # Update the quiz_set with the owner_id
            update_response = supabase.table("quiz_sets")\
                .update({"created_by": owner_id})\
                .eq("id", quiz_set_id)\
                .execute()
            
            logger.info(f"Updated quiz_set {quiz_set_id} (set#{set_number}) "
                       f"with created_by={owner_id}")
        
        logger.info("Quiz backfill completed!")
        
    except Exception as e:
        logger.error(f"Error backfilling quiz_sets: {e}")
        raise


if __name__ == "__main__":
    logger.info("Starting backfill of created_by fields...")
    
    try:
        backfill_flashcard_created_by()
        backfill_quiz_created_by()
        logger.info("✅ Backfill completed successfully!")
    except Exception as e:
        logger.error(f"❌ Backfill failed: {e}")
        exit(1)

