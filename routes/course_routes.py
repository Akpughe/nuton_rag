"""
FastAPI routes for Course Generation.
Clean, well-documented endpoints following REST conventions.
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from typing import List, Optional
import logging
import time

from services.course_service import CourseService, CourseGenerationError
from utils.file_storage import ProgressStorage
from models.course_models import (
    LearningProfileRequest, CourseFromTopicRequest, ProgressUpdateRequest
)

# File processing imports
from processors.mistral_ocr_extractor import MistralOCRExtractor, MistralOCRConfig
from prompts.course_prompts import build_topic_extraction_prompt
from utils.model_config import ModelConfig
import os
import tempfile

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["courses"])

# Initialize service
course_service = CourseService()


# Learning Profile Endpoints
@router.post("/learning-profile", status_code=201)
async def create_learning_profile(request: LearningProfileRequest):
    """
    Save or update user learning preferences.
    
    The 5 questions from PRD:
    1. format_pref: reading/listening/testing
    2. depth_pref: quick/detailed/conversational/academic  
    3. role: student/professional/graduate_student
    4. learning_goal: exams/career/curiosity/supplement
    5. example_pref: real_world/technical/stories/analogies
    """
    try:
        success = course_service.save_learning_profile(request.dict())
        if not success:
            raise HTTPException(status_code=500, detail="Failed to save profile")
        
        return {
            "success": True,
            "profile": request.dict(),
            "message": "Learning preferences saved"
        }
    except Exception as e:
        logger.error(f"Error saving learning profile: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/learning-profile/{user_id}")
async def get_learning_profile(user_id: str):
    """Retrieve user's learning preferences"""
    profile = course_service.get_learning_profile(user_id)
    
    if not profile:
        return {
            "exists": False,
            "message": "No profile found. User will use defaults."
        }
    
    return {
        "exists": True,
        "profile": profile
    }


# Course Generation Endpoints
@router.post("/courses/from-topic", status_code=201)
async def create_course_from_topic(request: CourseFromTopicRequest):
    """
    Generate complete course from topic string.
    
    **Blocking operation** - Returns full course in 45-60 seconds.
    
    Models available:
    - claude-sonnet-4 (default): Best quality, supports web search
    - gpt-4o: Good quality, no web search
    - llama-4-scout: Fastest, cheapest, no web search
    """
    try:
        # Validate model if provided
        if request.model and request.model not in ModelConfig.get_available_models():
            available = ModelConfig.get_available_models()
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid model. Available: {available}"
            )
        
        result = await course_service.create_course_from_topic(
            user_id=request.user_id,
            topic=request.topic,
            context=request.context,
            model=request.model
        )
        
        return result
        
    except CourseGenerationError as e:
        logger.error(f"Course generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/courses/from-files", status_code=201)
async def create_course_from_files(
    files: List[UploadFile] = File(...),
    user_id: str = Form(...),
    organization: str = Form("auto"),
    model: Optional[str] = Form(None)
):
    """
    Generate course from uploaded PDF/PPT files.
    
    **Multi-file support included:**
    - Auto-detects topic relationships
    - Recommends organization strategy
    - 1-10 files supported
    
    Organization options:
    - auto: Let system decide based on topic similarity
    - thematic_bridge: Unified course showing connections
    - sequential_sections: Separate sections within one course
    - separate_courses: Create multiple independent courses
    """
    try:
        # Validate files
        if len(files) > 10:
            raise HTTPException(status_code=400, detail="Maximum 10 files allowed")
        
        if len(files) < 1:
            raise HTTPException(status_code=400, detail="At least 1 file required")
        
        # Validate model
        if model and model not in ModelConfig.get_available_models():
            available = ModelConfig.get_available_models()
            raise HTTPException(
                status_code=400,
                detail=f"Invalid model. Available: {available}"
            )
        
        # Process files
        processed_files = await _process_uploaded_files(files, model=model)
        
        # Generate course
        result = await course_service.create_course_from_files(
            user_id=user_id,
            files=processed_files,
            organization=organization,
            model=model
        )
        
        return result
        
    except CourseGenerationError as e:
        logger.error(f"File course generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Course Access Endpoints
@router.get("/courses/{course_id}")
async def get_course(course_id: str):
    """
    Retrieve full course with all chapters.
    
    Returns complete course data including:
    - Course metadata
    - Chapter summaries
    - Progress information (if available)
    """
    course = course_service.get_course(course_id)
    
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")
    
    # Get progress if available
    progress_storage = ProgressStorage()
    progress = progress_storage.load_progress(course["user_id"], course_id)
    
    return {
        "course": course,
        "progress": progress
    }


@router.get("/courses/{course_id}/chapters/{chapter_order}")
async def get_chapter(course_id: str, chapter_order: int):
    """
    Retrieve specific chapter by order number (1-indexed).
    
    Returns full chapter content including:
    - Markdown content
    - Quiz questions
    - Source citations
    """
    chapter = course_service.get_chapter(course_id, chapter_order)
    
    if not chapter:
        raise HTTPException(status_code=404, detail="Chapter not found")
    
    return {"chapter": chapter}


@router.get("/users/{user_id}/courses")
async def list_user_courses(user_id: str):
    """List all courses for a user"""
    courses = course_service.list_user_courses(user_id)
    return {
        "user_id": user_id,
        "course_count": len(courses),
        "courses": courses
    }


# Progress Endpoints
@router.post("/courses/{course_id}/progress")
async def update_progress(course_id: str, request: ProgressUpdateRequest):
    """
    Update chapter completion and quiz scores.
    
    Track:
    - Chapter completion status
    - Quiz scores
    - Time spent
    """
    try:
        progress_storage = ProgressStorage()
        
        # Load existing or create new
        course = course_service.get_course(course_id)
        if not course:
            raise HTTPException(status_code=404, detail="Course not found")
        
        user_id = course["user_id"]
        progress_data = progress_storage.load_progress(user_id, course_id) or {
            "user_id": user_id,
            "course_id": course_id,
            "chapter_progress": [],
            "overall_progress": {
                "completed_chapters": 0,
                "total_chapters": course["total_chapters"],
                "percentage": 0,
                "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ")
            }
        }
        
        # Update chapter progress
        chapter_found = False
        for cp in progress_data["chapter_progress"]:
            if cp["chapter_id"] == request.chapter_id:
                cp["completed"] = request.completed
                cp["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ") if request.completed else None
                if request.quiz_score is not None:
                    cp.setdefault("quiz_attempts", []).append({
                        "attempt_id": len(cp.get("quiz_attempts", [])) + 1,
                        "score": request.quiz_score,
                        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ")
                    })
                if request.time_spent_minutes:
                    cp["time_spent_minutes"] = request.time_spent_minutes
                chapter_found = True
                break
        
        if not chapter_found:
            progress_data["chapter_progress"].append({
                "chapter_id": request.chapter_id,
                "completed": request.completed,
                "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ") if request.completed else None,
                "quiz_attempts": [{"attempt_id": 1, "score": request.quiz_score, "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ")}] if request.quiz_score else [],
                "time_spent_minutes": request.time_spent_minutes or 0
            })
        
        # Recalculate overall progress
        completed = sum(1 for cp in progress_data["chapter_progress"] if cp["completed"])
        total = course["total_chapters"]
        progress_data["overall_progress"]["completed_chapters"] = completed
        progress_data["overall_progress"]["total_chapters"] = total
        progress_data["overall_progress"]["percentage"] = round((completed / total) * 100) if total > 0 else 0
        progress_data["overall_progress"]["last_activity"] = time.strftime("%Y-%m-%dT%H:%M:%SZ")
        
        # Save
        success = progress_storage.save_progress(progress_data)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to save progress")
        
        return {
            "success": True,
            "overall_progress": progress_data["overall_progress"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating progress: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/users/{user_id}/progress")
async def get_user_progress(user_id: str):
    """Get learning progress across all courses"""
    courses = course_service.list_user_courses(user_id)
    
    progress_storage = ProgressStorage()
    courses_with_progress = []
    
    for course in courses:
        progress = progress_storage.load_progress(user_id, course["id"])
        courses_with_progress.append({
            "course": course,
            "progress": progress
        })
    
    # Calculate stats
    completed_courses = [c for c in courses_with_progress if c["progress"] and c["progress"]["overall_progress"]["percentage"] == 100]
    in_progress = [c for c in courses_with_progress if c["progress"] and c["progress"]["overall_progress"]["percentage"] > 0 and c["progress"]["overall_progress"]["percentage"] < 100]
    
    return {
        "user_id": user_id,
        "total_courses": len(courses),
        "completed_courses": len(completed_courses),
        "in_progress": len(in_progress),
        "courses": courses_with_progress
    }


# Course Q&A Endpoint
@router.post("/courses/{course_id}/ask")
async def ask_course_question(course_id: str, question: str = Form(...), model: Optional[str] = Form(None)):
    """
    Ask a question about a course. Answers are grounded in the uploaded source material.
    Uses Pinecone vector search to find relevant chunks from the original documents.
    """
    try:
        result = await course_service.query_course(
            course_id=course_id,
            question=question,
            model=model
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Course Q&A error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Helper functions
ALLOWED_EXTENSIONS = {".pdf", ".pptx", ".ppt", ".docx", ".doc", ".txt", ".md"}
MAX_FILE_SIZE_MB = 50


async def _process_uploaded_files(files: List[UploadFile], model: Optional[str] = None) -> List[dict]:
    """Process uploaded files with OCR and topic extraction"""
    processed = []

    # Initialize Mistral OCR
    mistral_config = MistralOCRConfig(
        enhance_metadata_with_llm=True,
        fallback_method="legacy",
        include_images=False
    )
    extractor = MistralOCRExtractor(config=mistral_config)

    for file in files:
        # Validate file extension
        ext = os.path.splitext(file.filename)[1].lower() if file.filename else ""
        if ext not in ALLOWED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {ext}. Allowed: {ALLOWED_EXTENSIONS}")

        # Validate file size
        content = await file.read()
        await file.seek(0)  # Reset for later read

        size_mb = len(content) / (1024 * 1024)
        if size_mb > MAX_FILE_SIZE_MB:
            raise ValueError(f"File {file.filename} is {size_mb:.1f}MB. Max: {MAX_FILE_SIZE_MB}MB")

        # Handle plaintext files directly (no OCR needed)
        plaintext_extensions = {".txt", ".md"}
        if ext in plaintext_extensions:
            text = content.decode("utf-8", errors="replace")

            if not text.strip():
                raise ValueError(f"No text extracted from {file.filename}")

            topic = await _extract_topic(text[:2000], model=model)

            processed.append({
                "filename": file.filename,
                "topic": topic,
                "extracted_text": text,
                "pages": 1,
                "char_count": len(text)
            })

            logger.info(f"Processed {file.filename}: {topic} ({len(text)} chars, plaintext)")
            continue

        # Save temp file for OCR-based extraction
        temp_path = _save_temp_file(file)

        try:
            # Extract text via OCR
            extraction = extractor.process_document(temp_path)
            text = extraction.get('full_text', '')

            if not text:
                raise ValueError(f"No text extracted from {file.filename}")

            # Extract topic using Claude
            topic = await _extract_topic(text[:2000], model=model)

            processed.append({
                "filename": file.filename,
                "topic": topic,
                "extracted_text": text,  # FULL text - no truncation
                "pages": extraction.get('total_pages', 0),
                "char_count": len(text)
            })

            logger.info(f"Processed {file.filename}: {topic} ({len(text)} chars, {extraction.get('total_pages', 0)} pages)")

        finally:
            # Cleanup
            if os.path.exists(temp_path):
                os.remove(temp_path)

    return processed


def _save_temp_file(file: UploadFile) -> str:
    """Save uploaded file to temp location"""
    suffix = os.path.splitext(file.filename)[1] if file.filename else ""
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = file.file.read()
        if not content:
            raise ValueError(f"Empty file: {file.filename}")
        tmp.write(content)
        return tmp.name


async def _extract_topic(text: str, model: Optional[str] = None) -> str:
    """Extract main topic from text using LLM"""
    prompt = build_topic_extraction_prompt(text)
    model_config = ModelConfig.get_config(model)
    
    # Import here to avoid circular dependency
    from services.course_service import CourseService
    service = CourseService()
    
    # Quick call to get topic
    response = await service._call_model(prompt, model_config, expect_json=False)
    topic = response.get("content", "").strip()
    
    # Clean up
    topic = topic.replace('"', '').replace("Topic:", "").strip()
    
    return topic or "General Topic"


# Model info endpoint
@router.get("/models")
async def get_available_models():
    """List all available AI models for course generation"""
    from utils.model_config import MODEL_CONFIGS, estimate_course_cost
    
    models = []
    for key, config in MODEL_CONFIGS.items():
        models.append({
            "id": key,
            "name": config["model"],
            "provider": config["provider"],
            "supports_web_search": config["supports_search"],
            "estimated_cost_per_course": estimate_course_cost(key, 4),
            "default": key == "claude-sonnet-4"
        })
    
    return {"models": models}
