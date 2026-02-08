"""
JSON file storage utilities for Course Generation POC.
Simple, robust file operations following KISS principle.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime
import logging
import uuid

logger = logging.getLogger(__name__)

# Base paths
BASE_DIR = Path("/Users/davak/Documents/nuton_rag")
COURSES_DIR = BASE_DIR / "courses"
LEARNING_PROFILES_FILE = BASE_DIR / "learning_profiles.json"
COURSE_INDEX_FILE = COURSES_DIR / "index.json"
GENERATION_LOGS_FILE = BASE_DIR / "course_generation_logs.json"


def ensure_directories():
    """Ensure all required directories exist"""
    COURSES_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Ensured courses directory: {COURSES_DIR}")


def generate_uuid() -> str:
    """Generate unique ID for courses/chapters"""
    return str(uuid.uuid4())


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
    """Handle learning profile storage"""
    
    @staticmethod
    def get_profile(user_id: str) -> Optional[Dict[str, Any]]:
        """Get learning profile for user"""
        data = read_json_file(LEARNING_PROFILES_FILE)
        if not data or "profiles" not in data:
            return None
        
        for profile in data["profiles"]:
            if profile.get("user_id") == user_id:
                return profile
        return None
    
    @staticmethod
    def save_profile(profile_data: Dict[str, Any]) -> bool:
        """Save or update learning profile"""
        data = read_json_file(LEARNING_PROFILES_FILE) or {"profiles": []}
        
        # Update existing or add new
        updated = False
        for i, existing in enumerate(data["profiles"]):
            if existing.get("user_id") == profile_data["user_id"]:
                data["profiles"][i] = profile_data
                updated = True
                break
        
        if not updated:
            data["profiles"].append(profile_data)
        
        return write_json_file(LEARNING_PROFILES_FILE, data)


# Course Storage Operations
class CourseStorage:
    """Handle course file storage"""
    
    @staticmethod
    def create_course_directory(course_id: str) -> Path:
        """Create and return course directory path"""
        course_dir = COURSES_DIR / f"course_{course_id}"
        course_dir.mkdir(parents=True, exist_ok=True)
        return course_dir
    
    @staticmethod
    def save_course(course_data: Dict[str, Any]) -> bool:
        """Save course metadata"""
        course_id = course_data["id"]
        course_dir = CourseStorage.create_course_directory(course_id)
        
        # Save course.json
        success = write_json_file(course_dir / "course.json", course_data)
        
        if success:
            # Update index
            CourseStorage._update_index(course_data)
        
        return success
    
    @staticmethod
    def save_chapter(course_id: str, chapter_data: Dict[str, Any]) -> bool:
        """Save individual chapter"""
        course_dir = COURSES_DIR / f"course_{course_id}"
        chapter_order = chapter_data["order"]
        
        return write_json_file(
            course_dir / f"chapter_{chapter_order}.json", 
            chapter_data
        )
    
    @staticmethod
    def get_course(course_id: str) -> Optional[Dict[str, Any]]:
        """Load full course with chapters"""
        course_dir = COURSES_DIR / f"course_{course_id}"
        course_file = course_dir / "course.json"
        
        course_data = read_json_file(course_file)
        if not course_data:
            return None
        
        # Load chapters
        chapters = []
        for i in range(1, course_data.get("total_chapters", 0) + 1):
            chapter_file = course_dir / f"chapter_{i}.json"
            chapter = read_json_file(chapter_file)
            if chapter:
                chapters.append(chapter)
        
        course_data["chapters"] = chapters
        return course_data
    
    @staticmethod
    def get_chapter(course_id: str, chapter_order: int) -> Optional[Dict[str, Any]]:
        """Load specific chapter"""
        chapter_file = COURSES_DIR / f"course_{course_id}" / f"chapter_{chapter_order}.json"
        return read_json_file(chapter_file)
    
    @staticmethod
    def list_courses(user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all courses, optionally filtered by user"""
        index = read_json_file(COURSE_INDEX_FILE)
        if not index or "courses" not in index:
            return []
        
        courses = index["courses"]
        if user_id:
            courses = [c for c in courses if c.get("user_id") == user_id]
        
        return courses
    
    @staticmethod
    def _update_index(course_data: Dict[str, Any]):
        """Update course index"""
        index = read_json_file(COURSE_INDEX_FILE) or {"courses": []}
        
        # Update or add entry
        updated = False
        for i, existing in enumerate(index["courses"]):
            if existing.get("id") == course_data["id"]:
                index["courses"][i] = {
                    "id": course_data["id"],
                    "user_id": course_data["user_id"],
                    "title": course_data["title"],
                    "topic": course_data["topic"],
                    "status": course_data["status"],
                    "total_chapters": course_data["total_chapters"],
                    "created_at": course_data["created_at"]
                }
                updated = True
                break
        
        if not updated:
            index["courses"].append({
                "id": course_data["id"],
                "user_id": course_data["user_id"],
                "title": course_data["title"],
                "topic": course_data["topic"],
                "status": course_data["status"],
                "total_chapters": course_data["total_chapters"],
                "created_at": course_data["created_at"]
            })
        
        write_json_file(COURSE_INDEX_FILE, index)


# Progress Storage Operations
class ProgressStorage:
    """Handle progress tracking"""
    
    @staticmethod
    def get_progress_file(course_id: str) -> Path:
        """Get path to progress file for course"""
        return COURSES_DIR / f"course_{course_id}" / "progress.json"
    
    @staticmethod
    def load_progress(user_id: str, course_id: str) -> Optional[Dict[str, Any]]:
        """Load progress for user and course"""
        progress_file = ProgressStorage.get_progress_file(course_id)
        data = read_json_file(progress_file)
        
        if not data or "user_progress" not in data:
            return None
        
        for progress in data["user_progress"]:
            if progress.get("user_id") == user_id:
                return progress
        
        return None
    
    @staticmethod
    def save_progress(progress_data: Dict[str, Any]) -> bool:
        """Save progress update"""
        course_id = progress_data["course_id"]
        progress_file = ProgressStorage.get_progress_file(course_id)
        
        data = read_json_file(progress_file) or {"user_progress": []}
        
        # Update or add
        updated = False
        for i, existing in enumerate(data["user_progress"]):
            if existing.get("user_id") == progress_data["user_id"]:
                data["user_progress"][i] = progress_data
                updated = True
                break
        
        if not updated:
            data["user_progress"].append(progress_data)
        
        return write_json_file(progress_file, data)


# Generation Logging
class GenerationLogger:
    """Log course generation for debugging and cost tracking"""
    
    @staticmethod
    def log_generation(log_entry: Dict[str, Any]) -> bool:
        """Log a generation attempt"""
        log_entry["timestamp"] = datetime.utcnow().isoformat()
        return append_to_json_list(GENERATION_LOGS_FILE, log_entry)


# Initialize on import
ensure_directories()
