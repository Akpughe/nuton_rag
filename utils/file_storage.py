"""
Storage utilities for Course Generation.
Supabase-backed storage for courses, chapters, profiles, and progress.
Local JSON still used for generation logs.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime
import logging
import uuid

from clients.supabase_client import (
    get_supabase,
    get_learning_profile,
    upsert_learning_profile,
    upsert_course,
    get_course_by_id,
    get_course_by_slug,
    list_courses_by_user,
    is_slug_taken,
    upsert_chapter,
    get_chapter_by_order,
    get_chapters_by_course,
    upsert_chapter_progress,
    get_chapter_progress_for_course,
    get_all_progress_for_user,
    get_all_quiz_attempts_for_user,
    insert_course_quiz_attempt,
    get_course_quiz_attempts,
)

logger = logging.getLogger(__name__)

# Base paths (only used for generation logs now)
BASE_DIR = Path(__file__).parent.parent
GENERATION_LOGS_FILE = BASE_DIR / "course_generation_logs.json"


def generate_uuid() -> str:
    """Generate unique ID for courses/chapters"""
    return str(uuid.uuid4())


def generate_slug(title: str) -> str:
    """
    Generate a URL-friendly slug from a course title.
    e.g. "Exploring Modern AI & Machine Learning" -> "exploring-modern-ai-machine-learning"
    Ensures uniqueness by appending -2, -3, etc. if slug is taken.
    """
    import re
    # Lowercase and replace non-alphanumeric with hyphens
    slug = title.lower().strip()
    slug = re.sub(r'[^a-z0-9]+', '-', slug)
    # Remove leading/trailing hyphens
    slug = slug.strip('-')
    # Truncate to 80 chars at a word boundary
    if len(slug) > 80:
        slug = slug[:80].rsplit('-', 1)[0]

    # Ensure uniqueness
    base_slug = slug
    counter = 2
    while is_slug_taken(slug):
        slug = f"{base_slug}-{counter}"
        counter += 1

    return slug


def read_json_file(filepath: Path) -> Optional[Dict[str, Any]]:
    """Read JSON file, return None if not found or invalid"""
    try:
        if not filepath.exists():
            return None
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {filepath}: {e}")
        return None
    except Exception as e:
        logger.error(f"Error reading {filepath}: {e}")
        return None


def write_json_file(filepath: Path, data: Dict[str, Any]) -> bool:
    """Write data to JSON file atomically"""
    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        temp_file = filepath.with_suffix('.tmp')
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)
        temp_file.rename(filepath)
        return True
    except Exception as e:
        logger.error(f"Error writing {filepath}: {e}")
        return False


def append_to_json_list(filepath: Path, item: Dict[str, Any]) -> bool:
    """Append item to JSON list file (creates if not exists)"""
    try:
        data = read_json_file(filepath) or {"items": []}
        if "items" not in data:
            data["items"] = []
        data["items"].append(item)
        return write_json_file(filepath, data)
    except Exception as e:
        logger.error(f"Error appending to {filepath}: {e}")
        return False


# Learning Profile Operations
class LearningProfileStorage:
    """Handle learning profile storage via Supabase"""

    @staticmethod
    def get_profile(user_id: str) -> Optional[Dict[str, Any]]:
        """Get learning profile for user from Supabase"""
        try:
            return get_learning_profile(user_id)
        except Exception as e:
            logger.error(f"Error getting learning profile for {user_id}: {e}")
            return None

    @staticmethod
    def save_profile(profile_data: Dict[str, Any]) -> bool:
        """Save or update learning profile in Supabase"""
        try:
            upsert_learning_profile(profile_data)
            return True
        except Exception as e:
            logger.error(f"Error saving learning profile: {e}")
            return False


# Course Storage Operations
class CourseStorage:
    """Handle course storage via Supabase"""

    @staticmethod
    def save_course(course_data: Dict[str, Any]) -> bool:
        """Save or update course in Supabase via upsert"""
        try:
            upsert_course(course_data)
            return True
        except Exception as e:
            logger.error(f"Error saving course: {e}")
            return False

    @staticmethod
    def save_chapter(course_id: str, chapter_data: Dict[str, Any]) -> bool:
        """Save individual chapter to Supabase via upsert"""
        try:
            data = dict(chapter_data)  # shallow copy to avoid mutating caller's dict
            data["course_id"] = course_id
            upsert_chapter(data)
            return True
        except Exception as e:
            logger.error(f"Error saving chapter: {e}")
            return False

    @staticmethod
    def get_course(course_id: str) -> Optional[Dict[str, Any]]:
        """Load full course with chapters from Supabase"""
        try:
            course_data = get_course_by_id(course_id)
            if not course_data:
                return None
            chapters = get_chapters_by_course(course_id)
            course_data["chapters"] = chapters
            return course_data
        except Exception as e:
            logger.error(f"Error getting course {course_id}: {e}")
            return None

    @staticmethod
    def get_chapter(course_id: str, chapter_order: int) -> Optional[Dict[str, Any]]:
        """Load specific chapter from Supabase"""
        try:
            return get_chapter_by_order(course_id, chapter_order)
        except Exception as e:
            logger.error(f"Error getting chapter {chapter_order} for course {course_id}: {e}")
            return None

    @staticmethod
    def list_courses(user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List courses from Supabase, optionally filtered by user"""
        try:
            if not user_id:
                return []
            return list_courses_by_user(user_id)
        except Exception as e:
            logger.error(f"Error listing courses: {e}")
            return []


# Progress Storage Operations
class ProgressStorage:
    """Handle progress tracking via Supabase"""

    @staticmethod
    def load_progress(user_id: str, course_id: str) -> Optional[Dict[str, Any]]:
        """Load progress for user and course from Supabase.
        Assembles normalized DB rows into the nested dict shape
        expected by routes: {user_id, course_id, chapter_progress: [...], overall_progress: {...}}
        """
        try:
            rows = get_chapter_progress_for_course(user_id, course_id)
            if not rows:
                return None

            # Get course for total_chapters
            course = get_course_by_id(course_id)
            total_chapters = course.get("total_chapters", 0) if course else 0

            chapter_progress = []
            for row in rows:
                chapter_id = row.get("chapter_id")
                # Fetch quiz attempts for this chapter
                quiz_attempts_raw = get_course_quiz_attempts(user_id, chapter_id)
                quiz_attempts = [
                    {
                        "attempt_id": i + 1,
                        "score": float(qa.get("score", 0)),
                        "completed_at": qa.get("completed_at"),
                    }
                    for i, qa in enumerate(quiz_attempts_raw)
                ]

                chapter_progress.append({
                    "chapter_id": chapter_id,
                    "completed": row.get("completed", False),
                    "completed_at": row.get("completed_at"),
                    "quiz_attempts": quiz_attempts,
                    "time_spent_minutes": row.get("time_spent_minutes", 0),
                })

            completed_count = sum(1 for cp in chapter_progress if cp["completed"])
            percentage = round((completed_count / total_chapters) * 100) if total_chapters > 0 else 0

            # Find earliest and latest timestamps
            started_at = None
            last_activity = None
            for row in rows:
                created = row.get("created_at")
                updated = row.get("updated_at")
                if created and (started_at is None or created < started_at):
                    started_at = created
                if updated and (last_activity is None or updated > last_activity):
                    last_activity = updated

            return {
                "user_id": user_id,
                "course_id": course_id,
                "chapter_progress": chapter_progress,
                "overall_progress": {
                    "completed_chapters": completed_count,
                    "total_chapters": total_chapters,
                    "percentage": percentage,
                    "started_at": started_at,
                    "last_activity": last_activity,
                }
            }
        except Exception as e:
            logger.error(f"Error loading progress for {user_id}/{course_id}: {e}")
            return None

    @staticmethod
    def load_all_progress(user_id: str, courses: List[Dict[str, Any]]) -> Dict[str, Optional[Dict[str, Any]]]:
        """Load progress for ALL courses in 2 queries (instead of N+1+N*M).
        Returns {course_id: progress_dict_or_None}.
        """
        try:
            # 2 bulk queries instead of N*M individual ones
            all_progress = get_all_progress_for_user(user_id)
            all_quiz = get_all_quiz_attempts_for_user(user_id)

            # Index quiz attempts by chapter_id
            quiz_by_chapter: Dict[str, List[Dict]] = {}
            for qa in all_quiz:
                cid = qa.get("chapter_id")
                quiz_by_chapter.setdefault(cid, []).append(qa)

            # Group progress rows by course_id
            progress_by_course: Dict[str, List[Dict]] = {}
            for row in all_progress:
                cid = row.get("course_id")
                progress_by_course.setdefault(cid, []).append(row)

            # Build per-course total_chapters lookup
            total_chapters_map = {c["id"]: c.get("total_chapters", 0) for c in courses}

            result: Dict[str, Optional[Dict[str, Any]]] = {}
            for course in courses:
                course_id = course["id"]
                rows = progress_by_course.get(course_id, [])
                if not rows:
                    result[course_id] = None
                    continue

                total_chapters = total_chapters_map.get(course_id, 0)

                chapter_progress = []
                for row in rows:
                    chapter_id = row.get("chapter_id")
                    quiz_raw = quiz_by_chapter.get(chapter_id, [])
                    quiz_attempts = [
                        {
                            "attempt_id": i + 1,
                            "score": float(qa.get("score", 0)),
                            "completed_at": qa.get("completed_at"),
                        }
                        for i, qa in enumerate(quiz_raw)
                    ]
                    chapter_progress.append({
                        "chapter_id": chapter_id,
                        "completed": row.get("completed", False),
                        "completed_at": row.get("completed_at"),
                        "quiz_attempts": quiz_attempts,
                        "time_spent_minutes": row.get("time_spent_minutes", 0),
                    })

                completed_count = sum(1 for cp in chapter_progress if cp["completed"])
                percentage = round((completed_count / total_chapters) * 100) if total_chapters > 0 else 0

                started_at = None
                last_activity = None
                for row in rows:
                    created = row.get("created_at")
                    updated = row.get("updated_at")
                    if created and (started_at is None or created < started_at):
                        started_at = created
                    if updated and (last_activity is None or updated > last_activity):
                        last_activity = updated

                result[course_id] = {
                    "user_id": user_id,
                    "course_id": course_id,
                    "chapter_progress": chapter_progress,
                    "overall_progress": {
                        "completed_chapters": completed_count,
                        "total_chapters": total_chapters,
                        "percentage": percentage,
                        "started_at": started_at,
                        "last_activity": last_activity,
                    }
                }

            return result
        except Exception as e:
            logger.error(f"Error loading all progress for {user_id}: {e}")
            return {c["id"]: None for c in courses}

    @staticmethod
    def save_progress(progress_data: Dict[str, Any]) -> bool:
        """Save progress update by decomposing into individual upsert calls."""
        try:
            user_id = progress_data["user_id"]
            course_id = progress_data["course_id"]

            for cp in progress_data.get("chapter_progress", []):
                upsert_chapter_progress(
                    user_id=user_id,
                    course_id=course_id,
                    chapter_id=cp["chapter_id"],
                    completed=cp.get("completed", False),
                    time_spent_minutes=cp.get("time_spent_minutes", 0),
                    completed_at=cp.get("completed_at"),
                )
            return True
        except Exception as e:
            logger.error(f"Error saving progress: {e}")
            return False


# Study Guide & Flashcards Storage (stored on courses table)
class StudyGuideStorage:
    """Save/get study guide from courses.study_guide JSONB column"""

    @staticmethod
    def save_study_guide(course_id: str, study_guide: Dict[str, Any]) -> bool:
        try:
            get_supabase().table("courses").update(
                {"study_guide": study_guide}
            ).eq("id", course_id).execute()
            return True
        except Exception as e:
            logger.error(f"Error saving study guide for {course_id}: {e}")
            return False

    @staticmethod
    def get_study_guide(course_id: str) -> Optional[Dict[str, Any]]:
        try:
            response = get_supabase().table("courses").select("study_guide").eq("id", course_id).execute()
            if response.data and response.data[0].get("study_guide"):
                return response.data[0]["study_guide"]
            return None
        except Exception as e:
            logger.error(f"Error getting study guide for {course_id}: {e}")
            return None


class FlashcardStorage:
    """Save/get course flashcards from courses.flashcards JSONB column"""

    @staticmethod
    def save_flashcards(course_id: str, flashcards: List[Dict[str, Any]]) -> bool:
        try:
            get_supabase().table("courses").update(
                {"flashcards": flashcards}
            ).eq("id", course_id).execute()
            return True
        except Exception as e:
            logger.error(f"Error saving flashcards for {course_id}: {e}")
            return False

    @staticmethod
    def get_flashcards(course_id: str) -> Optional[List[Dict[str, Any]]]:
        try:
            response = get_supabase().table("courses").select("flashcards").eq("id", course_id).execute()
            if response.data and response.data[0].get("flashcards"):
                return response.data[0]["flashcards"]
            return None
        except Exception as e:
            logger.error(f"Error getting flashcards for {course_id}: {e}")
            return None


class ExamStorage:
    """Save/get exams from course_exams table"""

    @staticmethod
    def save_exam(course_id: str, user_id: str, exam_size: int, exam_data: Dict[str, Any]) -> str:
        try:
            data = {
                "course_id": course_id,
                "user_id": user_id,
                "exam_size": exam_size,
                "mcq": exam_data.get("mcq", []),
                "fill_in_gap": exam_data.get("fill_in_gap", []),
                "theory": exam_data.get("theory", []),
            }
            response = get_supabase().table("course_exams").insert(data).execute()
            if not response.data:
                raise Exception(f"Failed to insert exam: {response}")
            return str(response.data[0]["id"])
        except Exception as e:
            logger.error(f"Error saving exam for {course_id}: {e}")
            raise

    @staticmethod
    def get_exam(course_id: str, user_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        try:
            query = get_supabase().table("course_exams").select("*").eq("course_id", course_id)
            if user_id:
                query = query.eq("user_id", user_id)
            response = query.order("created_at", desc=True).limit(1).execute()
            if response.data and len(response.data) > 0:
                return response.data[0]
            return None
        except Exception as e:
            logger.error(f"Error getting exam for {course_id}: {e}")
            return None

    @staticmethod
    def get_exam_by_id(exam_id: str) -> Optional[Dict[str, Any]]:
        try:
            response = get_supabase().table("course_exams").select("*").eq("id", exam_id).limit(1).execute()
            if response.data and len(response.data) > 0:
                return response.data[0]
            return None
        except Exception as e:
            logger.error(f"Error getting exam by id {exam_id}: {e}")
            return None


class ExamAttemptStorage:
    """Save/get exam attempt results from course_exam_attempts table"""

    @staticmethod
    def save_attempt(
        exam_id: str,
        user_id: str,
        answers: Dict[str, Any],
        results: Dict[str, Any],
        score: float,
        mcq_score: Optional[float],
        fill_in_gap_score: Optional[float],
        theory_score: Optional[float],
        time_taken_seconds: Optional[int] = None
    ) -> str:
        try:
            data = {
                "exam_id": exam_id,
                "user_id": user_id,
                "answers": answers,
                "results": results,
                "score": score,
                "mcq_score": mcq_score,
                "fill_in_gap_score": fill_in_gap_score,
                "theory_score": theory_score,
                "time_taken_seconds": time_taken_seconds,
            }
            response = get_supabase().table("course_exam_attempts").insert(data).execute()
            if not response.data:
                raise Exception(f"Failed to insert exam attempt: {response}")
            return str(response.data[0]["id"])
        except Exception as e:
            logger.error(f"Error saving exam attempt for exam {exam_id}: {e}")
            raise

    @staticmethod
    def get_attempts(exam_id: str, user_id: str) -> List[Dict[str, Any]]:
        try:
            response = get_supabase().table("course_exam_attempts") \
                .select("id, exam_id, score, mcq_score, fill_in_gap_score, theory_score, time_taken_seconds, created_at") \
                .eq("exam_id", exam_id) \
                .eq("user_id", user_id) \
                .order("created_at", desc=True) \
                .execute()
            return response.data or []
        except Exception as e:
            logger.error(f"Error getting attempts for exam {exam_id}: {e}")
            return []

    @staticmethod
    def get_attempt_by_id(attempt_id: str) -> Optional[Dict[str, Any]]:
        try:
            response = get_supabase().table("course_exam_attempts") \
                .select("*") \
                .eq("id", attempt_id) \
                .limit(1) \
                .execute()
            if response.data and len(response.data) > 0:
                return response.data[0]
            return None
        except Exception as e:
            logger.error(f"Error getting attempt {attempt_id}: {e}")
            return None


class ChatStorage:
    """Save/get chat messages from course_chat_messages table"""

    @staticmethod
    def save_message(course_id: str, user_id: str, role: str, content: str, sources: Optional[List] = None) -> bool:
        try:
            data = {
                "course_id": course_id,
                "user_id": user_id,
                "role": role,
                "content": content,
            }
            if sources:
                data["sources"] = sources
            get_supabase().table("course_chat_messages").insert(data).execute()
            return True
        except Exception as e:
            logger.error(f"Error saving chat message for {course_id}: {e}")
            return False

    @staticmethod
    def get_messages(course_id: str, user_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        try:
            response = get_supabase().table("course_chat_messages") \
                .select("role, content, sources, created_at") \
                .eq("course_id", course_id) \
                .eq("user_id", user_id) \
                .order("created_at", desc=True) \
                .limit(limit) \
                .execute()
            messages = response.data or []
            messages.reverse()  # Oldest first
            return messages
        except Exception as e:
            logger.error(f"Error getting chat messages for {course_id}: {e}")
            return []

    @staticmethod
    def clear_messages(course_id: str, user_id: str) -> bool:
        try:
            get_supabase().table("course_chat_messages") \
                .delete() \
                .eq("course_id", course_id) \
                .eq("user_id", user_id) \
                .execute()
            return True
        except Exception as e:
            logger.error(f"Error clearing chat messages for {course_id}: {e}")
            return False


# Generation Logging (stays local — no DB table)
class GenerationLogger:
    """Log course generation for debugging and cost tracking"""

    @staticmethod
    def log_generation(log_entry: Dict[str, Any]) -> bool:
        """Log a generation attempt"""
        log_entry["timestamp"] = datetime.utcnow().isoformat()
        return append_to_json_list(GENERATION_LOGS_FILE, log_entry)
