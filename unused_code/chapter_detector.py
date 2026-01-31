import json
import logging
from typing import List, Dict, Any, Optional, Tuple
import asyncio
from groq_client import generate_answer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set chapter detector to DEBUG level for detailed logging
logger.setLevel(logging.DEBUG)


def _clean_json_string(json_str: str) -> str:
    """
    Clean JSON string by removing/fixing control characters and formatting issues.
    Handles common issues from PDF extraction and AI responses.
    """
    import re

    # Remove common control characters that break JSON
    # Keep only newlines temporarily for processing
    control_chars = {
        '\x00': '',  # Null
        '\x01': '',  # Start of heading
        '\x02': '',  # Start of text
        '\x03': '',  # End of text
        '\x04': '',  # End of transmission
        '\x05': '',  # Enquiry
        '\x06': '',  # Acknowledge
        '\x07': '',  # Bell
        '\x08': '',  # Backspace
        '\x0b': ' ',  # Vertical tab
        '\x0c': ' ',  # Form feed (common in PDFs)
        '\x0e': '',  # Shift out
        '\x0f': '',  # Shift in
        '\x10': '',  # Data link escape
        '\x11': '',  # Device control 1
        '\x12': '',  # Device control 2
        '\x13': '',  # Device control 3
        '\x14': '',  # Device control 4
        '\x15': '',  # Negative acknowledge
        '\x16': '',  # Synchronous idle
        '\x17': '',  # End of transmission block
        '\x18': '',  # Cancel
        '\x19': '',  # End of medium
        '\x1a': '',  # Substitute
        '\x1b': '',  # Escape
        '\x1c': '',  # File separator
        '\x1d': '',  # Group separator
        '\x1e': '',  # Record separator
        '\x1f': '',  # Unit separator
    }

    for char, replacement in control_chars.items():
        json_str = json_str.replace(char, replacement)

    # Also handle escaped control characters that might appear in JSON
    json_str = json_str.replace('\\x0c', ' ')
    json_str = json_str.replace('\\x0b', ' ')

    # Fix line breaks within JSON string values
    # Strategy: Find strings that contain newlines and fix them
    # We'll do this character by character to handle properly

    result = []
    in_string = False
    escape_next = False
    i = 0

    while i < len(json_str):
        char = json_str[i]

        # Handle escape sequences
        if escape_next:
            result.append(char)
            escape_next = False
            i += 1
            continue

        if char == '\\':
            escape_next = True
            result.append(char)
            i += 1
            continue

        # Track whether we're inside a string
        if char == '"':
            in_string = not in_string
            result.append(char)
            i += 1
            continue

        # If we're inside a string and hit a newline, replace with space
        if in_string and char == '\n':
            # Skip the newline and any following whitespace
            i += 1
            while i < len(json_str) and json_str[i] in ' \t\r':
                i += 1
            # Add single space
            result.append(' ')
            continue

        # For everything else, just append
        result.append(char)
        i += 1

    # Join and clean up excessive spaces
    json_str = ''.join(result)

    # Clean up multiple spaces (but preserve JSON structure)
    # Only within string values (this is safe now that newlines are fixed)
    json_str = re.sub(r'  +', ' ', json_str)

    return json_str


async def detect_chapters_with_ai(
    full_text: str,
    model: str = "llama-3.1-8b-instant",
    timeout: int = 10
) -> List[Dict[str, Any]]:
    """
    Use Groq AI to detect chapters in a document.

    Args:
        full_text: Complete document text
        model: Groq model to use (default: fast model for speed)
        timeout: Max seconds to wait for AI response

    Returns:
        List of chapters with chapter_number, chapter_title, and first_words

    Raises:
        TimeoutError: If AI takes too long
        Exception: If AI fails or returns invalid JSON
    """
    logger.info(f"Starting AI chapter detection on document ({len(full_text)} chars)")

    # Handle documents within Groq's context window (~400k chars for safety)
    if len(full_text) < 400000:
        return await _detect_full_document(full_text, model, timeout)
    else:
        logger.warning(f"Document too large ({len(full_text)} chars), using sliding window")
        return await _detect_large_document(full_text, model, timeout)


async def _detect_full_document(
    full_text: str,
    model: str,
    timeout: int
) -> List[Dict[str, Any]]:
    """Detect chapters in a document that fits in context window."""

    prompt = f"""Analyze this ENTIRE document and identify ALL chapters, sections, or major divisions.

Include: numbered chapters, unnumbered sections, introductions, prefaces, epilogues, appendices, etc.

For each chapter/section, provide:
1. chapter_number: The number/identifier (use "intro", "preface", "epilogue", "appendix_a" etc. for unnumbered sections)
2. chapter_title: The actual title/heading text
3. first_words: Copy EXACTLY the first 10-20 words of the actual content that appears after the chapter heading. This must be verbatim text from the document, not a summary.

CRITICAL for "first_words":
- Copy the EXACT text as it appears (word-for-word, including punctuation)
- Get the text that comes AFTER the heading/title
- This will be used to locate the chapter in the document, so accuracy is essential
- If the chapter starts with a paragraph, copy the first 10-20 words of that paragraph

COMPLETE DOCUMENT:
{full_text}

Return ONLY a valid JSON array with this exact structure:
[
  {{
    "chapter_number": "1",
    "chapter_title": "Introduction to Machine Learning",
    "first_words": "Machine learning is a subset of artificial intelligence that enables"
  }}
]

Return ONLY the JSON array, no explanation or markdown formatting."""

    try:
        # Use asyncio timeout
        async def get_ai_response():
            # Note: generate_answer is synchronous, wrap in executor
            loop = asyncio.get_event_loop()
            response, _ = await loop.run_in_executor(
                None,
                lambda: generate_answer(
                    query=prompt,
                    context_chunks=[],
                    system_prompt="You are a precise document analyzer. Return only valid JSON.",
                    model=model
                )
            )
            return response

        response = await asyncio.wait_for(get_ai_response(), timeout=timeout)

        # Clean response - sometimes models add markdown
        response = response.strip()
        if response.startswith("```json"):
            response = response.replace("```json", "").replace("```", "").strip()
        elif response.startswith("```"):
            response = response.replace("```", "").strip()

        # Clean control characters and fix line breaks in JSON
        response = _clean_json_string(response)
        logger.debug(f"Cleaned response (first 300 chars): {response[:300]}")

        # Parse JSON
        chapters = json.loads(response)

        if not isinstance(chapters, list):
            logger.error(f"AI returned non-list: {type(chapters)}")
            return []

        logger.info(f"AI detected {len(chapters)} chapters")

        # Debug: Log chapter detection results
        for chapter in chapters[:3]:  # Log first 3 chapters for debugging
            logger.debug(f"Detected: Ch {chapter.get('chapter_number')} - {chapter.get('chapter_title')} - First words: '{chapter.get('first_words', '')[:60]}...'")

        return chapters

    except asyncio.TimeoutError:
        logger.error(f"Chapter detection timed out after {timeout}s")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse AI response as JSON: {e}")
        logger.debug(f"Original response (first 500 chars): {response[:500]}")
        # Try to show the problematic part
        try:
            error_pos = e.pos if hasattr(e, 'pos') else 0
            context_start = max(0, error_pos - 100)
            context_end = min(len(response), error_pos + 100)
            logger.debug(f"Context around error position {error_pos}: ...{response[context_start:context_end]}...")
        except:
            pass
        raise
    except Exception as e:
        logger.error(f"Chapter detection failed: {e}")
        raise


async def _detect_large_document(
    full_text: str,
    model: str,
    timeout: int
) -> List[Dict[str, Any]]:
    """
    Detect chapters in very large documents using sliding window.
    Splits document into overlapping chunks and deduplicates results.
    """

    window_size = 300000  # ~75k tokens
    overlap = 50000       # Overlap to catch chapters at boundaries

    all_chapters = []
    position = 0
    window_count = 0

    while position < len(full_text):
        window_count += 1
        chunk = full_text[position:position + window_size]

        logger.info(f"Processing window {window_count} (position {position})")

        prompt = f"""Find ALL chapter headings in this document section.

Document section (starting at character position {position}):
{chunk}

Return JSON array with:
- chapter_number
- chapter_title
- first_words (first 15-25 words of chapter content)

Return ONLY valid JSON array, no explanation."""

        try:
            loop = asyncio.get_event_loop()
            response, _ = await loop.run_in_executor(
                None,
                lambda: generate_answer(
                    query=prompt,
                    context_chunks=[],
                    system_prompt="Return only valid JSON.",
                    model=model
                )
            )

            # Clean and parse
            response = response.strip()
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                response = response.split("```")[1].split("```")[0].strip()

            # Clean control characters and fix line breaks
            response = _clean_json_string(response)

            chapters = json.loads(response)

            if isinstance(chapters, list):
                # Add position offset for deduplication
                for ch in chapters:
                    ch['_window_position'] = position
                all_chapters.extend(chapters)
                logger.info(f"Found {len(chapters)} chapters in window {window_count}")

        except Exception as e:
            logger.warning(f"Failed to process window {window_count}: {e}")

        position += (window_size - overlap)

    # Deduplicate chapters found in overlapping regions
    deduplicated = _deduplicate_chapters(all_chapters)
    logger.info(f"Total chapters after deduplication: {len(deduplicated)}")

    return deduplicated


def _deduplicate_chapters(chapters: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Remove duplicate chapters found in overlapping windows.
    Uses chapter_number and title similarity for matching.
    """
    if not chapters:
        return []

    unique_chapters = []
    seen_signatures = set()

    for chapter in chapters:
        # Create signature from chapter number and title
        signature = f"{chapter.get('chapter_number', '')}:{chapter.get('chapter_title', '')[:50]}"

        if signature not in seen_signatures:
            seen_signatures.add(signature)
            # Remove internal tracking field
            if '_window_position' in chapter:
                del chapter['_window_position']
            unique_chapters.append(chapter)

    return unique_chapters


def assign_chapters_to_chunks(
    chunks: List[Dict[str, Any]],
    detected_chapters: List[Dict[str, Any]],
    full_text: str
) -> List[Dict[str, Any]]:
    """
    Assign chapter metadata to chunks using hybrid position + content matching.

    Args:
        chunks: List of text chunks from chunking process
        detected_chapters: Chapters detected by AI
        full_text: Complete document text (for position validation)

    Returns:
        Chunks with added chapter_number and chapter_title fields
    """

    if not detected_chapters:
        logger.warning("No chapters detected, assigning all chunks to default chapter")
        for chunk in chunks:
            chunk['chapter_number'] = "1"
            chunk['chapter_title'] = "Full Document"
        return chunks

    logger.info(f"Assigning {len(chunks)} chunks to {len(detected_chapters)} chapters")

    # Step 1: Find exact positions of chapters using content matching
    chapters_with_positions = _find_chapter_positions(detected_chapters, full_text)

    # Step 2: Assign each chunk to a chapter
    current_chapter_idx = 0

    for chunk in chunks:
        chunk_text = chunk.get('text', '')
        chunk_start = chunk.get('start_index', 0)

        # Check if this chunk contains a NEW chapter start (content-based)
        for i in range(current_chapter_idx, len(chapters_with_positions)):
            chapter = chapters_with_positions[i]

            # Match by content signature (first_words)
            first_words = chapter.get('first_words', '').strip()
            if first_words and len(first_words) > 10:
                # Normalize for matching (lowercase, remove extra spaces)
                normalized_first_words = ' '.join(first_words.lower().split())
                normalized_chunk = ' '.join(chunk_text.lower().split())

                if normalized_first_words in normalized_chunk:
                    current_chapter_idx = i
                    logger.info(f"Found Chapter {chapter['chapter_number']} in chunk (content match)")
                    break

        # Also validate with position if available
        if 'exact_position' in chapters_with_positions[current_chapter_idx]:
            chapter_pos = chapters_with_positions[current_chapter_idx]['exact_position']
            # If chunk is before current chapter, move back
            if chunk_start < chapter_pos and current_chapter_idx > 0:
                current_chapter_idx -= 1

        # Assign current chapter to this chunk
        current_chapter = chapters_with_positions[current_chapter_idx]
        chunk['chapter_number'] = current_chapter['chapter_number']
        chunk['chapter_title'] = current_chapter['chapter_title']
        chunk['chapter_assignment_method'] = current_chapter.get('match_method', 'sequential')

    # Validation: log chapter distribution
    _log_chapter_distribution(chunks)

    return chunks


def _find_chapter_positions(
    chapters: List[Dict[str, Any]],
    full_text: str
) -> List[Dict[str, Any]]:
    """
    Find exact positions of chapters in full text using content matching.
    Returns chapters with 'exact_position' and 'match_method' fields.
    """

    chapters_with_positions = []

    for chapter in chapters:
        first_words = chapter.get('first_words', '').strip()

        if first_words:
            # Try multiple matching strategies
            pos, match_method = _find_text_position(first_words, full_text, chapter['chapter_number'])

            if pos != -1:
                chapters_with_positions.append({
                    **chapter,
                    'exact_position': pos,
                    'match_method': match_method
                })
                logger.debug(f"Chapter {chapter['chapter_number']} found at position {pos} using {match_method}")
            else:
                # Could not find - use sequential assumption
                logger.warning(f"Could not locate Chapter {chapter['chapter_number']} by content. First words: '{first_words[:50]}...'")
                chapters_with_positions.append({
                    **chapter,
                    'match_method': 'sequential'
                })
        else:
            logger.warning(f"Chapter {chapter['chapter_number']} missing first_words")
            chapters_with_positions.append({
                **chapter,
                'match_method': 'sequential'
            })

    # Sort by position (if available) or maintain order
    chapters_with_positions.sort(
        key=lambda x: x.get('exact_position', float('inf'))
    )

    return chapters_with_positions


def _find_text_position(search_text: str, full_text: str, chapter_num: str) -> Tuple[int, str]:
    """
    Try multiple strategies to find text position in document.
    Returns (position, match_method) or (-1, '') if not found.
    """

    # Strategy 1: Exact match (case-insensitive)
    pos = full_text.lower().find(search_text.lower())
    if pos != -1:
        return pos, 'exact_match'

    # Strategy 2: Normalized match (remove extra whitespace)
    normalized_search = ' '.join(search_text.lower().split())
    normalized_text = ' '.join(full_text.lower().split())
    pos = normalized_text.find(normalized_search)
    if pos != -1:
        # Map back to approximate position in original text
        return pos, 'normalized_match'

    # Strategy 3: Partial match - try first 50 characters
    if len(search_text) > 50:
        partial_search = search_text[:50].lower()
        pos = full_text.lower().find(partial_search)
        if pos != -1:
            return pos, 'partial_match'

    # Strategy 4: Token-based fuzzy match - try first 10 words
    search_words = search_text.lower().split()[:10]
    if len(search_words) >= 5:
        # Try to find first 5 words in sequence
        search_phrase = ' '.join(search_words[:5])
        pos = full_text.lower().find(search_phrase)
        if pos != -1:
            return pos, 'fuzzy_match'

    # Strategy 5: Very aggressive - try just first 3 words
    if len(search_words) >= 3:
        search_phrase = ' '.join(search_words[:3])
        pos = full_text.lower().find(search_phrase)
        if pos != -1:
            logger.debug(f"Chapter {chapter_num} found using only first 3 words")
            return pos, 'aggressive_match'

    return -1, ''


def _log_chapter_distribution(chunks: List[Dict[str, Any]]) -> None:
    """Log how chunks are distributed across chapters for validation."""

    chapter_counts = {}
    for chunk in chunks:
        ch_num = chunk.get('chapter_number', 'unknown')
        chapter_counts[ch_num] = chapter_counts.get(ch_num, 0) + 1

    logger.info("Chapter assignment distribution:")
    for ch_num, count in sorted(chapter_counts.items()):
        logger.info(f"  Chapter {ch_num}: {count} chunks")


def get_fallback_chapter() -> List[Dict[str, Any]]:
    """
    Return a single default chapter for when detection fails.
    """
    return [{
        "chapter_number": "1",
        "chapter_title": "Full Document",
        "first_words": ""
    }]
