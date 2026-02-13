"""
Unified exception hierarchy for Nuton RAG.

All domain exceptions inherit from NutonError and carry:
- error_code: machine-readable string (e.g. "COURSE_NOT_FOUND")
- status_code: HTTP status code
- message: human-readable description
- context: optional structured metadata dict
"""

from typing import Optional, Dict, Any


class NutonError(Exception):
    """Base exception for all Nuton domain errors."""

    def __init__(
        self,
        message: str,
        error_code: str = "INTERNAL_ERROR",
        status_code: int = 500,
        context: Optional[Dict[str, Any]] = None,
    ):
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        self.context = context or {}
        super().__init__(message)


class ValidationError(NutonError):
    """400-level validation / bad-request errors."""

    def __init__(
        self,
        message: str,
        error_code: str = "INVALID_JSON",
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, error_code=error_code, status_code=400, context=context)


class NotFoundError(NutonError):
    """404 resource-not-found errors."""

    def __init__(
        self,
        message: str,
        error_code: str = "COURSE_NOT_FOUND",
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, error_code=error_code, status_code=404, context=context)


class GenerationError(NutonError):
    """500-level generation failures."""

    def __init__(
        self,
        message: str,
        error_code: str = "GENERATION_FAILED",
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, error_code=error_code, status_code=500, context=context)


class OutlineGenerationError(GenerationError):
    """Failed to generate course outline."""

    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, error_code="OUTLINE_GENERATION_FAILED", context=context)


class ChapterGenerationError(GenerationError):
    """Failed to generate a chapter."""

    def __init__(self, chapter_num: int, message: str, context: Optional[Dict[str, Any]] = None):
        self.chapter_num = chapter_num
        ctx = {"chapter_num": chapter_num}
        if context:
            ctx.update(context)
        super().__init__(
            f"Chapter {chapter_num} generation failed: {message}",
            error_code="CHAPTER_GENERATION_FAILED",
            context=ctx,
        )


class StorageError(NutonError):
    """500-level database / storage failures."""

    def __init__(
        self,
        message: str,
        error_code: str = "STORAGE_ERROR",
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, error_code=error_code, status_code=500, context=context)
