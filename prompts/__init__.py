# Prompts module initialization

# Course Generation Prompts
from .course_prompts import (
    build_outline_generation_prompt,
    build_chapter_content_prompt,
    build_topic_extraction_prompt,
    build_multi_file_analysis_prompt,
    RETRY_PROMPT_ADDENDUM
)

__all__ = [
    'build_outline_generation_prompt',
    'build_chapter_content_prompt',
    'build_topic_extraction_prompt',
    'build_multi_file_analysis_prompt',
    'RETRY_PROMPT_ADDENDUM'
]
