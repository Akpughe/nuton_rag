"""
FastAPI routes for Space-level operations.

  POST /api/v1/spaces/{space_id}/courses
      — Batch-add courses to a space (updates Pinecone vector metadata).

  POST /api/v1/spaces/{space_id}/query/stream
      — Stream a RAG query across all courses in a space.
        Searches courses in parallel, streams LLM tokens, cites sources,
        and falls back to web search / general knowledge if nothing relevant is found.
"""

import json
import logging
from typing import List, Optional

from fastapi import APIRouter, Body
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from services.space_service import SpaceService, _sse
from utils.exceptions import NutonError, ValidationError
from utils.model_config import ModelConfig

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["spaces"])

space_service = SpaceService()


# ─── Request models ────────────────────────────────────────────────────────────

class AddCoursesRequest(BaseModel):
    course_ids: List[str] = Field(..., min_length=1, max_length=50)


class SpaceQueryRequest(BaseModel):
    course_ids: List[str] = Field(..., min_length=1, max_length=50)
    query: str
    user_id: str
    model: Optional[str] = None


# ─── SSE error helper ─────────────────────────────────────────────────────────

def _sse_error(exc: Exception) -> str:
    if isinstance(exc, NutonError):
        return _sse({
            "type": "error",
            "error": exc.error_code,
            "message": exc.message,
            "status_code": exc.status_code,
        })
    return _sse({"type": "error", "error": "INTERNAL_ERROR", "message": str(exc), "status_code": 500})


# ─── Endpoints ────────────────────────────────────────────────────────────────

@router.post("/spaces/{space_id}/courses")
async def add_courses_to_space(
    space_id: str,
    body: AddCoursesRequest = Body(...),
):
    """
    Batch-add courses to a space by updating their Pinecone vector metadata
    to include this space_id. The Supabase side (space_courses table) is
    managed by the frontend — this only updates Pinecone for queryability.

    Body:
        course_ids: List of course UUIDs to add (1–50).

    Returns a summary of how many vectors were updated per course.
    """
    result = await space_service.add_courses_to_space(
        space_id=space_id,
        course_ids=body.course_ids,
    )
    return result


@router.post("/spaces/{space_id}/query/stream")
async def query_space_stream(
    space_id: str,
    body: SpaceQueryRequest = Body(...),
):
    """
    Stream a question across all courses in a space.

    Searches all provided courses in parallel, filters by relevance,
    and streams a personalized, citation-aware LLM response.

    If no relevant content is found in any course, falls back to a
    web-search-powered answer for Claude models (clearly labelled as external),
    or general-knowledge answer for other providers.

    SSE event types:
      status       — phase transitions (searching, synthesizing, no_results)
      course_found — emitted as each course returns relevant chunks
      token        — streaming LLM text delta
      citations    — full list of source citations after the response
      done         — signals end of stream
      error        — on failure

    Body:
        course_ids: List of course UUIDs to search (1–50).
        query:      The learner's question.
        user_id:    Used for personalization and chat history.
        model:      Optional model override (defaults to system default).
    """
    if not body.query.strip():
        raise ValidationError(
            "query must not be empty",
            error_code="INVALID_REQUEST",
            context={"space_id": space_id},
        )
    if body.model and body.model not in ModelConfig.get_available_models():
        raise ValidationError(
            f"Invalid model. Available: {ModelConfig.get_available_models()}",
            error_code="INVALID_MODEL",
            context={"model": body.model},
        )

    async def event_generator():
        try:
            async for event in space_service.query_space_stream(
                space_id=space_id,
                course_ids=body.course_ids,
                query=body.query,
                user_id=body.user_id,
                model=body.model,
            ):
                yield event
        except Exception as e:
            logger.error(f"Space query SSE error: {e}", exc_info=True)
            yield _sse_error(e)

    return StreamingResponse(event_generator(), media_type="text/event-stream")
