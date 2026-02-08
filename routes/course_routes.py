"""
FastAPI routes for Course Generation.
Clean, well-documented endpoints following REST conventions.
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from typing import List, Optional
import json
import logging
import re
import time

from services.course_service import CourseService, CourseGenerationError
from services.wetrocloud_youtube import WetroCloudYouTubeService
from clients.jina_reader_client import extract_web_content
from utils.file_storage import ProgressStorage
from clients.supabase_client import (
    upsert_chapter_progress,
    insert_course_quiz_attempt,
    get_chapter_progress_for_course,
)
from models.course_models import (
    LearningProfileRequest, ProgressUpdateRequest
)

# File processing imports
from processors.mistral_ocr_extractor import MistralOCRExtractor, MistralOCRConfig
from prompts.course_prompts import build_topic_extraction_prompt
from utils.model_config import ModelConfig
import os
import tempfile

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["courses"])

# Initialize services
course_service = CourseService()
yt_service = WetroCloudYouTubeService()


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
async def create_course_from_topic(
    user_id: str = Form(...),
    topic: str = Form(...),
    model: Optional[str] = Form(None),
    files: Optional[List[UploadFile]] = File(None),
    youtube_urls: Optional[str] = Form(None),
    web_urls: Optional[str] = Form(None)
):
    """
    Generate complete course from topic string with optional supplementary sources.

    **Blocking operation** - Returns full course in 30-60 seconds.

    Accepts Form data (not JSON) to support optional file uploads.
    Topic drives the course structure; files, YouTube videos, and web URLs
    provide supplementary context.

    Source params (all optional, can be combined):
    - files: PDF/PPTX/DOCX/TXT/MD uploads (max 10, 50MB each)
    - youtube_urls: JSON array of YouTube URLs, e.g. '["https://youtube.com/watch?v=abc"]'
    - web_urls: JSON array of web URLs, e.g. '["https://example.com/article"]'

    Models available:
    - claude-haiku-4-5 (default): Fast, affordable, native web search
    - claude-sonnet-4-5: High quality, native web search
    - claude-opus-4-6: Highest quality, native web search
    - gpt-4o: Good quality, web search via Responses API
    - gpt-5-mini: Fast, affordable, web search via Responses API
    - gpt-5.2: High quality, web search via Responses API
    - llama-4-scout: Fastest, cheapest, Perplexity search fallback
    - llama-4-maverick: Fast, affordable, Perplexity search fallback
    """
    try:
        # Validate model if provided
        if model and model not in ModelConfig.get_available_models():
            available = ModelConfig.get_available_models()
            raise HTTPException(
                status_code=400,
                detail=f"Invalid model. Available: {available}"
            )

        # Parse URL lists
        try:
            yt_urls = _parse_url_list(youtube_urls)
            w_urls = _parse_url_list(web_urls)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        # Validate file count
        has_files = files and len(files) > 0 and files[0].filename
        if has_files and len(files) > 10:
            raise HTTPException(status_code=400, detail="Maximum 10 files allowed")

        # Build merged source list
        all_processed = []

        if has_files:
            all_processed.extend(await _process_uploaded_files(files, model=model))

        if yt_urls:
            all_processed.extend(await _process_youtube_urls(yt_urls, model=model))

        if w_urls:
            all_processed.extend(await _process_web_urls(w_urls, model=model))

        # Route to appropriate service method
        if all_processed:
            result = await course_service.create_course_from_topic_with_files(
                user_id=user_id,
                topic=topic,
                files=all_processed,
                model=model
            )
        else:
            # Pure topic path
            result = await course_service.create_course_from_topic(
                user_id=user_id,
                topic=topic,
                context={},
                model=model
            )

        return result

    except HTTPException:
        raise
    except CourseGenerationError as e:
        logger.error(f"Course generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/courses/from-files", status_code=201)
async def create_course_from_files(
    files: Optional[List[UploadFile]] = File(None),
    user_id: str = Form(...),
    organization: str = Form("auto"),
    model: Optional[str] = Form(None),
    youtube_urls: Optional[str] = Form(None),
    web_urls: Optional[str] = Form(None)
):
    """
    Generate course from uploaded files, YouTube videos, and/or web URLs.

    At least one source is required (file, YouTube URL, or web URL).

    **Multi-source support:**
    - Files: PDF/PPTX/DOCX/TXT/MD (max 10, 50MB each)
    - youtube_urls: JSON array of YouTube URLs, e.g. '["https://youtube.com/watch?v=abc"]'
    - web_urls: JSON array of web URLs, e.g. '["https://example.com/article"]'

    Organization options:
    - auto: Let system decide based on topic similarity
    - thematic_bridge: Unified course showing connections
    - sequential_sections: Separate sections within one course
    - separate_courses: Create multiple independent courses

    Response format varies by organization:
    - Single course: `{course_id, status, course, generation_time_seconds, ...}`
    - Separate courses: `{organization: "separate_courses", total_courses, courses: [...]}`
    """
    try:
        # Validate model
        if model and model not in ModelConfig.get_available_models():
            available = ModelConfig.get_available_models()
            raise HTTPException(
                status_code=400,
                detail=f"Invalid model. Available: {available}"
            )

        # Parse URL lists
        try:
            yt_urls = _parse_url_list(youtube_urls)
            w_urls = _parse_url_list(web_urls)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        # Validate file count
        has_files = files and len(files) > 0 and files[0].filename
        if has_files and len(files) > 10:
            raise HTTPException(status_code=400, detail="Maximum 10 files allowed")

        # Require at least one source
        if not has_files and not yt_urls and not w_urls:
            raise HTTPException(
                status_code=400,
                detail="At least 1 source required (file, YouTube URL, or web URL)"
            )

        # Build merged source list
        all_processed = []

        if has_files:
            all_processed.extend(await _process_uploaded_files(files, model=model))

        if yt_urls:
            all_processed.extend(await _process_youtube_urls(yt_urls, model=model))

        if w_urls:
            all_processed.extend(await _process_web_urls(w_urls, model=model))

        if not all_processed:
            raise HTTPException(
                status_code=400,
                detail="No usable content extracted from provided sources"
            )

        # Generate course
        result = await course_service.create_course_from_files(
            user_id=user_id,
            files=all_processed,
            organization=organization,
            model=model
        )

        return result

    except HTTPException:
        raise
    except CourseGenerationError as e:
        logger.error(f"File course generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# SSE Streaming Helpers

def _sse_event(data: dict) -> str:
    """Format a dict as an SSE data line."""
    return f"data: {json.dumps(data, default=str)}\n\n"


# SSE Streaming Endpoints

@router.post("/courses/from-topic/stream")
async def create_course_from_topic_stream(
    user_id: str = Form(...),
    topic: str = Form(...),
    model: Optional[str] = Form(None),
    files: Optional[List[UploadFile]] = File(None),
    youtube_urls: Optional[str] = Form(None),
    web_urls: Optional[str] = Form(None)
):
    """
    SSE streaming version of /courses/from-topic.
    Returns a text/event-stream that progressively emits:
    - processing_sources: when source extraction begins
    - outline_ready: course outline + metadata (~10s)
    - chapter_ready: each chapter as it finishes
    - course_complete: all done with timing
    - error: on failure
    """
    # Pre-stream validation (can still return HTTP errors)
    if model and model not in ModelConfig.get_available_models():
        available = ModelConfig.get_available_models()
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model. Available: {available}"
        )

    try:
        yt_urls = _parse_url_list(youtube_urls)
        w_urls = _parse_url_list(web_urls)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    has_files = files and len(files) > 0 and files[0].filename
    if has_files and len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 files allowed")

    async def event_generator():
        try:
            # Process sources
            all_processed = []
            source_count = 0

            if has_files:
                source_count += len(files)
            if yt_urls:
                source_count += len(yt_urls)
            if w_urls:
                source_count += len(w_urls)

            if source_count > 0:
                yield _sse_event({
                    "type": "processing_sources",
                    "message": f"Processing {source_count} source(s)...",
                    "source_count": source_count
                })

            if has_files:
                all_processed.extend(await _process_uploaded_files(files, model=model))
            if yt_urls:
                all_processed.extend(await _process_youtube_urls(yt_urls, model=model))
            if w_urls:
                all_processed.extend(await _process_web_urls(w_urls, model=model))

            # Route to streaming service method
            if all_processed:
                stream = course_service.create_course_from_topic_with_files_stream(
                    user_id=user_id,
                    topic=topic,
                    files=all_processed,
                    model=model
                )
            else:
                stream = course_service.create_course_from_topic_stream(
                    user_id=user_id,
                    topic=topic,
                    context={},
                    model=model
                )

            async for event in stream:
                yield _sse_event(event)

        except Exception as e:
            logger.error(f"SSE stream error: {e}")
            yield _sse_event({
                "type": "error",
                "message": str(e),
                "phase": "stream"
            })

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.post("/courses/from-files/stream")
async def create_course_from_files_stream(
    files: Optional[List[UploadFile]] = File(None),
    user_id: str = Form(...),
    organization: str = Form("auto"),
    model: Optional[str] = Form(None),
    youtube_urls: Optional[str] = Form(None),
    web_urls: Optional[str] = Form(None)
):
    """
    SSE streaming version of /courses/from-files.
    Returns a text/event-stream that progressively emits events.
    """
    # Pre-stream validation
    if model and model not in ModelConfig.get_available_models():
        available = ModelConfig.get_available_models()
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model. Available: {available}"
        )

    try:
        yt_urls = _parse_url_list(youtube_urls)
        w_urls = _parse_url_list(web_urls)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    has_files = files and len(files) > 0 and files[0].filename
    if has_files and len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 files allowed")

    if not has_files and not yt_urls and not w_urls:
        raise HTTPException(
            status_code=400,
            detail="At least 1 source required (file, YouTube URL, or web URL)"
        )

    async def event_generator():
        try:
            # Process sources
            all_processed = []
            source_count = (len(files) if has_files else 0) + len(yt_urls) + len(w_urls)

            yield _sse_event({
                "type": "processing_sources",
                "message": f"Processing {source_count} source(s)...",
                "source_count": source_count
            })

            if has_files:
                all_processed.extend(await _process_uploaded_files(files, model=model))
            if yt_urls:
                all_processed.extend(await _process_youtube_urls(yt_urls, model=model))
            if w_urls:
                all_processed.extend(await _process_web_urls(w_urls, model=model))

            if not all_processed:
                yield _sse_event({
                    "type": "error",
                    "message": "No usable content extracted from provided sources",
                    "phase": "source_processing"
                })
                return

            # Stream course generation
            async for event in course_service.create_course_from_files_stream(
                user_id=user_id,
                files=all_processed,
                organization=organization,
                model=model
            ):
                yield _sse_event(event)

        except Exception as e:
            logger.error(f"SSE stream error: {e}")
            yield _sse_event({
                "type": "error",
                "message": str(e),
                "phase": "stream"
            })

    return StreamingResponse(event_generator(), media_type="text/event-stream")


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
        course = course_service.get_course(course_id)
        if not course:
            raise HTTPException(status_code=404, detail="Course not found")

        user_id = course["user_id"]
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ")

        # Upsert chapter progress directly to Supabase
        upsert_chapter_progress(
            user_id=user_id,
            course_id=course_id,
            chapter_id=request.chapter_id,
            completed=request.completed,
            time_spent_minutes=request.time_spent_minutes or 0,
            completed_at=now if request.completed else None,
        )

        # Insert quiz attempt if score provided
        if request.quiz_score is not None:
            insert_course_quiz_attempt(
                user_id=user_id,
                chapter_id=request.chapter_id,
                score=request.quiz_score,
                completed_at=now,
            )

        # Re-query to compute overall stats
        rows = get_chapter_progress_for_course(user_id, course_id)
        total = course["total_chapters"]
        completed_count = sum(1 for r in rows if r.get("completed"))
        percentage = round((completed_count / total) * 100) if total > 0 else 0

        return {
            "success": True,
            "overall_progress": {
                "completed_chapters": completed_count,
                "total_chapters": total,
                "percentage": percentage,
                "last_activity": now,
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating progress: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/users/{user_id}/progress")
async def get_user_progress(user_id: str):
    """Get learning progress across all courses (batched — 3 queries total)"""
    courses = course_service.list_user_courses(user_id)

    # Batch: 2 queries for ALL progress + quiz attempts (instead of N+1+N*M)
    progress_storage = ProgressStorage()
    progress_map = progress_storage.load_all_progress(user_id, courses)

    courses_with_progress = []
    for course in courses:
        courses_with_progress.append({
            "course": course,
            "progress": progress_map.get(course["id"])
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


def _parse_url_list(urls_json: Optional[str]) -> List[str]:
    """Parse a JSON string into a list of URL strings."""
    if not urls_json:
        return []
    try:
        urls = json.loads(urls_json)
        if not isinstance(urls, list):
            raise ValueError("Expected a JSON array of URLs")
        return [str(u) for u in urls]
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON for URLs: {e}")


async def _process_youtube_urls(urls: List[str], model: Optional[str] = None) -> List[dict]:
    """Process YouTube URLs by extracting transcripts and packaging as source dicts.

    Follows the same pattern as process_youtube() in pipeline.py:
    uses WetroCloudYouTubeService with extract_video_id + get_transcript + get_video_title.
    """
    processed = []

    for url in urls:
        try:
            # Extract video ID first (validates the URL)
            video_id = yt_service.extract_video_id(url)
            if not video_id:
                logger.warning(f"Invalid YouTube URL, could not extract video ID: {url}")
                continue

            # Get transcript (WetroCloud -> Vcyon -> yt-dlp fallback chain)
            result = yt_service.get_transcript(url)
            if not result.get("success"):
                logger.warning(f"YouTube transcript failed for {url}: {result.get('message', 'Unknown error')}")
                continue

            text = result.get("text", "")
            if len(text) < 50:
                logger.warning(f"YouTube transcript too short for {url}: {len(text)} chars")
                continue

            # Get video title — matches pipeline.py pattern
            if "video_title" in result and result["video_title"]:
                video_title = result["video_title"]
            else:
                video_title = yt_service.get_video_title(url)

            # Fall back to LLM topic extraction if title is still missing or generic
            if not video_title or video_title.startswith("YouTube Video:") or len(video_title) < 5:
                video_title = await _extract_topic(text[:2000], model=model)

            processed.append({
                "filename": f"youtube_{video_id}.txt",
                "topic": video_title,
                "extracted_text": text,
                "pages": 1,
                "char_count": len(text),
                "source_url": url,
                "source_type": "youtube",
            })

            logger.info(f"Processed YouTube {video_id}: {video_title} ({len(text)} chars, method={result.get('method', 'unknown')})")

        except Exception as e:
            logger.warning(f"YouTube processing error for {url}: {e}")
            continue

    return processed


async def _process_web_urls(urls: List[str], model: Optional[str] = None) -> List[dict]:
    """Process web URLs by extracting content via Jina Reader and packaging as source dicts."""
    processed = []

    for url in urls:
        try:
            result = extract_web_content(url)
            if not result.get("success"):
                logger.warning(f"Web extraction failed for {url}: {result.get('message', 'Unknown error')}")
                continue

            text = result.get("text", "")
            if len(text) < 50:
                logger.warning(f"Web content too short for {url}: {len(text)} chars")
                continue

            # Extract topic from content via LLM
            topic = await _extract_topic(text[:2000], model=model)

            # Build safe filename from domain
            safe_domain = re.sub(r"[^a-zA-Z0-9]", "_", url.split("//")[-1].split("/")[0])

            processed.append({
                "filename": f"web_{safe_domain}.txt",
                "topic": topic,
                "extracted_text": text,
                "pages": 1,
                "char_count": len(text),
                "source_url": url,
                "source_type": "web",
            })

            logger.info(f"Processed web URL {url}: {topic} ({len(text)} chars)")

        except Exception as e:
            logger.warning(f"Web URL processing error for {url}: {e}")
            continue

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

    # Use module-level singleton instead of creating new instances
    response = await course_service._call_model(prompt, model_config, expect_json=False)
    topic = response.get("content", "").strip()
    
    # Clean up
    topic = topic.replace('"', '').replace("Topic:", "").strip()
    
    return topic or "General Topic"


# Model info endpoint
@router.get("/models")
async def get_available_models():
    """List all available AI models for course generation"""
    from utils.model_config import MODEL_CONFIGS, DEFAULT_MODEL, estimate_course_cost, get_search_mode

    models = []
    for key, config in MODEL_CONFIGS.items():
        models.append({
            "id": key,
            "name": config["model"],
            "provider": config["provider"],
            "supports_web_search": config["supports_search"],
            "search_mode": get_search_mode(key),
            "estimated_cost_per_course": estimate_course_cost(key, 4),
            "default": key == DEFAULT_MODEL
        })

    return {"models": models}
