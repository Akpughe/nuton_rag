"""
Perplexity API client for web search fallback.
Used by Groq/Llama models that lack native web search.
Perplexity API is OpenAI-compatible (same SDK, different base_url).
"""

import os
import logging
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")


def _get_client():
    """Get Perplexity client (OpenAI-compatible)"""
    from openai import OpenAI
    api_key = PERPLEXITY_API_KEY or os.getenv("PERPLEXITY_API_KEY")
    if not api_key:
        logger.warning("PERPLEXITY_API_KEY not set. Search fallback unavailable.")
        return None
    return OpenAI(api_key=api_key, base_url="https://api.perplexity.ai")


def search_for_chapter(
    chapter_title: str,
    chapter_concepts: List[str],
    course_topic: str,
    model: str = "sonar-pro"
) -> Dict[str, Any]:
    """
    Search for authoritative sources relevant to a chapter.

    Returns:
        {"sources": [{"title": str, "url": str, "excerpt": str}]}
    """
    client = _get_client()
    if not client:
        return {"sources": []}

    concepts_text = ", ".join(chapter_concepts) if chapter_concepts else chapter_title
    prompt = (
        f"Find 3-5 authoritative sources about '{chapter_title}' "
        f"in the context of {course_topic}. Key concepts: {concepts_text}. "
        f"For each source, provide the title, URL, and a brief excerpt of "
        f"the most relevant information. Prioritize academic papers, "
        f"official documentation, and reputable educational sources."
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a research assistant. Return factual information "
                        "with real, verifiable sources. Format each source as:\n"
                        "TITLE: <title>\nURL: <url>\nEXCERPT: <relevant excerpt>\n---"
                    )
                },
                {"role": "user", "content": prompt}
            ],
            max_tokens=2000,
            temperature=0.3
        )

        content = response.choices[0].message.content or ""
        sources = _parse_sources(content)
        logger.info(f"Perplexity search for '{chapter_title}': {len(sources)} sources found")
        return {"sources": sources}

    except Exception as e:
        logger.warning(f"Perplexity search failed for '{chapter_title}': {e}")
        return {"sources": []}


def search_for_chapters_parallel(
    chapters: List[Dict[str, Any]],
    course_topic: str,
    max_workers: int = 4
) -> Dict[int, Dict[str, Any]]:
    """
    Search for sources for multiple chapters in parallel.

    Args:
        chapters: List of chapter outlines with "order", "title", "key_concepts"
        course_topic: Overall course topic
        max_workers: Max concurrent Perplexity calls

    Returns:
        Dict mapping chapter_order -> {"sources": [...]}
    """
    results: Dict[int, Dict[str, Any]] = {}

    if not PERPLEXITY_API_KEY and not os.getenv("PERPLEXITY_API_KEY"):
        logger.warning("PERPLEXITY_API_KEY not set. Skipping parallel search.")
        for ch in chapters:
            results[ch["order"]] = {"sources": []}
        return results

    def _search_single(chapter):
        return (
            chapter["order"],
            search_for_chapter(
                chapter_title=chapter["title"],
                chapter_concepts=chapter.get("key_concepts", []),
                course_topic=course_topic
            )
        )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_search_single, ch) for ch in chapters]
        for future in as_completed(futures):
            try:
                order, result = future.result()
                results[order] = result
            except Exception as e:
                logger.warning(f"Parallel search task failed: {e}")

    # Fill in any missing chapters with empty sources
    for ch in chapters:
        if ch["order"] not in results:
            results[ch["order"]] = {"sources": []}

    return results


def _parse_sources(text: str) -> List[Dict[str, str]]:
    """Parse Perplexity response into structured source list"""
    sources = []
    current = {}

    for line in text.split("\n"):
        line = line.strip()
        if not line or line == "---":
            if current.get("title") and current.get("url"):
                sources.append(current)
            if line == "---":
                current = {}
            continue

        if line.upper().startswith("TITLE:"):
            current["title"] = line.split(":", 1)[1].strip()
        elif line.upper().startswith("URL:"):
            current["url"] = line.split(":", 1)[1].strip()
            # Fix URL that was split on the colon in "https:"
            if current["url"].startswith("//"):
                current["url"] = "https:" + current["url"]
            elif not current["url"].startswith("http"):
                # Try reconstructing from the full line
                url_part = line[4:].strip()  # Skip "URL:"
                if "http" in url_part:
                    current["url"] = url_part[url_part.index("http"):]
        elif line.upper().startswith("EXCERPT:"):
            current["excerpt"] = line.split(":", 1)[1].strip()

    # Don't forget the last entry
    if current.get("title") and current.get("url"):
        sources.append(current)

    return sources
