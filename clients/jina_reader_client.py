"""
Jina Reader API client for extracting clean content from web URLs.
Uses r.jina.ai to convert any URL to clean markdown text.
"""

import os
import re
import logging
from typing import Dict, Any

import requests

logger = logging.getLogger(__name__)

JINA_API_KEY = os.getenv("JINA_API_KEY")


def extract_web_content(url: str, timeout: int = 30) -> Dict[str, Any]:
    """
    Extract clean text content from a web URL using Jina Reader API.

    Args:
        url: The web URL to extract content from
        timeout: Request timeout in seconds

    Returns:
        On success: {"success": True, "text": str, "url": str, "title": str, "char_count": int}
        On failure: {"success": False, "message": str, "url": str}
    """
    api_key = JINA_API_KEY or os.getenv("JINA_API_KEY")
    if not api_key:
        logger.warning("JINA_API_KEY not set. Web URL extraction unavailable.")
        return {"success": False, "message": "JINA_API_KEY not configured", "url": url}

    try:
        response = requests.get(
            f"https://r.jina.ai/{url}",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Accept": "text/markdown",
            },
            timeout=timeout,
        )
        response.raise_for_status()

        text = response.text.strip()
        if not text:
            return {"success": False, "message": "Empty content returned", "url": url}

        # Extract title from first markdown heading
        title = _extract_title(text) or url

        logger.info(f"Jina Reader extracted {len(text)} chars from {url}")
        return {
            "success": True,
            "text": text,
            "url": url,
            "title": title,
            "char_count": len(text),
        }

    except requests.Timeout:
        logger.warning(f"Jina Reader timeout for {url}")
        return {"success": False, "message": f"Timeout after {timeout}s", "url": url}
    except requests.HTTPError as e:
        logger.warning(f"Jina Reader HTTP error for {url}: {e}")
        return {"success": False, "message": f"HTTP error: {e.response.status_code}", "url": url}
    except Exception as e:
        logger.warning(f"Jina Reader error for {url}: {e}")
        return {"success": False, "message": str(e), "url": url}


def _extract_title(markdown: str) -> str:
    """Extract title from the first # heading in markdown content."""
    match = re.search(r"^#\s+(.+)$", markdown, re.MULTILINE)
    if match:
        return match.group(1).strip()
    return ""
