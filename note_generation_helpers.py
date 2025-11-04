"""
Note Generation Helper Functions
Utilities for organizing chunks, analyzing document structure, and preparing content for note generation.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def organize_chunks_by_hierarchy(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Organize chunks into hierarchical structure based on chapters, sections, and pages.

    Args:
        chunks: List of chunk dictionaries from Pinecone with metadata

    Returns:
        Organized structure:
        {
            "chapters": [
                {
                    "chapter_number": 1,
                    "chapter_title": "Introduction",
                    "sections": [...],
                    "chunks": [...]
                }
            ],
            "metadata": {
                "total_pages": 120,
                "total_chunks": 350,
                "has_chapters": True,
                "heading_hierarchy": [...]
            }
        }
    """
    logger.info(f"Organizing {len(chunks)} chunks by hierarchy...")

    # Sort chunks by position in document
    sorted_chunks = sort_chunks_by_position(chunks)

    # Extract metadata
    metadata = extract_document_metadata(sorted_chunks)

    # Check if document has chapters
    has_chapters = any(chunk.get("metadata", {}).get("chapter_number") for chunk in sorted_chunks)

    if has_chapters:
        chapters = organize_by_chapters(sorted_chunks)
    else:
        # If no chapters, organize by pages or sections
        chapters = organize_by_pages(sorted_chunks)

    # Build heading hierarchy
    heading_hierarchy = extract_heading_hierarchy(sorted_chunks)

    metadata.update({
        "has_chapters": has_chapters,
        "heading_hierarchy": heading_hierarchy,
        "total_chapters": len(chapters)
    })

    pages_info = (
        f"{metadata['total_pages']} pages"
        if metadata.get("total_pages", 0)
        else "no page metadata"
    )

    coverage_span = metadata.get("total_character_span")
    span_info = (
        f"characters {metadata['min_start_index']}–{metadata['max_end_index']} (~{coverage_span:,} chars)"
        if coverage_span
        else "no start/end index coverage"
    )

    indexed_chunks = metadata.get("chunks_with_position_indices", 0)

    logger.info(
        "✅ Organized into %s chapters/sections (%s; %s across %s indexed chunks)",
        len(chapters),
        pages_info,
        span_info,
        indexed_chunks,
    )

    return {
        "chapters": chapters,
        "metadata": metadata
    }


def sort_chunks_by_position(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Sort chunks by their position in the document.
    Priority: chapter_number → page_number → start_index

    Args:
        chunks: List of chunk dictionaries

    Returns:
        Sorted list of chunks
    """
    def get_sort_key(chunk: Dict[str, Any]) -> Tuple:
        """Generate sort key for chunk."""
        metadata = chunk.get("metadata", {})

        # Get chapter number (default to 0 if not present)
        chapter_num = metadata.get("chapter_number", 0)
        if isinstance(chapter_num, str):
            try:
                chapter_num = int(chapter_num)
            except (ValueError, TypeError):
                chapter_num = 0

        # Get page number
        page_num = metadata.get("page_number", "0")
        if isinstance(page_num, str):
            # Handle comma-separated page numbers (take first)
            page_num = page_num.split(',')[0] if page_num else "0"
            try:
                page_num = int(page_num)
            except (ValueError, TypeError):
                page_num = 0

        # Get start index (for chunks within same page)
        start_idx = chunk.get("start_index", 0)
        if not isinstance(start_idx, (int, float)):
            start_idx = 0

        return (chapter_num, page_num, start_idx)

    return sorted(chunks, key=get_sort_key)


def extract_document_metadata(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Extract document-level metadata from chunks.

    Args:
        chunks: List of chunk dictionaries

    Returns:
        Metadata dictionary
    """
    total_chunks = len(chunks)

    # Extract unique pages
    pages = set()
    for chunk in chunks:
        page_num = chunk.get("metadata", {}).get("page_number")
        if page_num:
            if isinstance(page_num, str):
                # Handle comma-separated page numbers
                for p in page_num.split(','):
                    if p and p not in {"", "false", "0"}:
                        pages.add(p)
            else:
                pages.add(str(page_num))

    total_pages = len(pages) if pages else 0

    # Extract source file
    source_file = None
    for chunk in chunks:
        source_file = chunk.get("metadata", {}).get("source_file")
        if source_file:
            break

    # Calculate total characters
    total_chars = sum(len(chunk.get("metadata", {}).get("text", "")) for chunk in chunks)

    # Track coverage using start_index / end_index
    min_start_index: Optional[int] = None
    max_end_index: Optional[int] = None
    chunks_with_indices = 0

    for chunk in chunks:
        metadata = chunk.get("metadata", {})

        start_raw = metadata.get("start_index")
        end_raw = metadata.get("end_index")

        start_idx = None
        end_idx = None

        if start_raw is not None:
            try:
                start_idx = int(float(start_raw))
            except (TypeError, ValueError):
                start_idx = None

        if end_raw is not None:
            try:
                end_idx = int(float(end_raw))
            except (TypeError, ValueError):
                end_idx = None

        if start_idx is not None:
            if min_start_index is None or start_idx < min_start_index:
                min_start_index = start_idx

        if end_idx is not None:
            if max_end_index is None or end_idx > max_end_index:
                max_end_index = end_idx

        if (
            start_idx is not None
            and end_idx is not None
            and end_idx > start_idx
        ):
            chunks_with_indices += 1

    total_character_span = None
    if (
        min_start_index is not None
        and max_end_index is not None
        and max_end_index > min_start_index
    ):
        total_character_span = max_end_index - min_start_index

    return {
        "total_chunks": total_chunks,
        "total_pages": total_pages,
        "source_file": source_file,
        "total_characters": total_chars,
        "min_start_index": min_start_index,
        "max_end_index": max_end_index,
        "total_character_span": total_character_span,
        "chunks_with_position_indices": chunks_with_indices,
    }


def organize_by_chapters(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Organize chunks by chapters.

    Args:
        chunks: Sorted list of chunk dictionaries

    Returns:
        List of chapter dictionaries
    """
    chapters_dict = defaultdict(lambda: {
        "chapter_number": 0,
        "chapter_title": "Unknown Chapter",
        "chunks": [],
        "sections": []
    })

    for chunk in chunks:
        metadata = chunk.get("metadata", {})

        chapter_num = metadata.get("chapter_number", 0)
        if isinstance(chapter_num, str):
            try:
                chapter_num = int(chapter_num)
            except (ValueError, TypeError):
                chapter_num = 0
        elif isinstance(chapter_num, (tuple, list)):
            # Handle tuple/list - take first element
            chapter_num = chapter_num[0] if chapter_num else 0
            try:
                chapter_num = int(chapter_num)
            except (ValueError, TypeError):
                chapter_num = 0

        chapter_title = metadata.get("chapter_title", f"Chapter {chapter_num}")
        # Ensure chapter_title is a string
        if not isinstance(chapter_title, str):
            chapter_title = str(chapter_title) if chapter_title is not None else f"Chapter {chapter_num}"

        if chapter_num not in chapters_dict or chapters_dict[chapter_num]["chapter_title"] == "Unknown Chapter":
            chapters_dict[chapter_num]["chapter_number"] = chapter_num
            chapters_dict[chapter_num]["chapter_title"] = chapter_title

        chapters_dict[chapter_num]["chunks"].append(chunk)

    # Convert to sorted list
    chapters = []
    for chapter_num in sorted(chapters_dict.keys()):
        chapter_data = chapters_dict[chapter_num]

        # Organize sections within chapter (if heading_path exists)
        chapter_data["sections"] = organize_sections_in_chapter(chapter_data["chunks"])

        chapters.append(chapter_data)

    return chapters


def organize_by_pages(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Organize chunks by pages when no chapter information is available.
    Groups pages into logical sections based on content breaks.

    Args:
        chunks: Sorted list of chunk dictionaries

    Returns:
        List of pseudo-chapter dictionaries
    """
    # Group chunks by pages (every 10-20 pages = 1 section)
    section_size = 15  # pages per section

    pages_dict = defaultdict(list)
    for chunk in chunks:
        page_num = chunk.get("metadata", {}).get("page_number", "0")
        if isinstance(page_num, str):
            page_num = page_num.split(',')[0] if page_num else "0"
            try:
                page_num = int(page_num)
            except (ValueError, TypeError):
                page_num = 0
        elif isinstance(page_num, (tuple, list)):
            # Handle tuple/list - take first element
            page_num = page_num[0] if page_num else "0"
            try:
                page_num = int(page_num)
            except (ValueError, TypeError):
                page_num = 0
        elif not isinstance(page_num, int):
            # Convert any other type to int
            try:
                page_num = int(page_num)
            except (ValueError, TypeError):
                page_num = 0

        pages_dict[page_num].append(chunk)

    # Create pseudo-chapters
    sorted_pages = sorted(pages_dict.keys())
    chapters = []

    section_num = 1
    current_section_chunks = []
    start_page = None
    end_page = None

    for page_num in sorted_pages:
        if start_page is None:
            start_page = page_num

        current_section_chunks.extend(pages_dict[page_num])
        end_page = page_num

        # Create section every section_size pages
        if end_page - start_page >= section_size or page_num == sorted_pages[-1]:
            chapters.append({
                "chapter_number": section_num,
                "chapter_title": f"Pages {start_page}-{end_page}",
                "chunks": current_section_chunks,
                "sections": []
            })

            section_num += 1
            current_section_chunks = []
            start_page = None
            end_page = None

    return chapters


def organize_sections_in_chapter(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Organize chunks within a chapter into sections based on heading_path.

    Args:
        chunks: List of chunks within a chapter

    Returns:
        List of section dictionaries
    """
    sections_dict = defaultdict(lambda: {
        "section_title": "Main Content",
        "chunks": []
    })

    for chunk in chunks:
        metadata = chunk.get("metadata", {})

        # Try to get section from heading_path
        heading_path = metadata.get("heading_path", "")
        if heading_path:
            # Parse heading_path (usually comma-separated)
            if isinstance(heading_path, str):
                headings = [h.strip() for h in heading_path.split(",")]
            else:
                headings = []

            # Use last heading as section title
            section_title = headings[-1] if headings else "Main Content"
        else:
            section_title = "Main Content"

        sections_dict[section_title]["section_title"] = section_title
        sections_dict[section_title]["chunks"].append(chunk)

    # Convert to list
    return [sections_dict[title] for title in sections_dict.keys()]


def extract_heading_hierarchy(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Extract heading hierarchy from chunks.

    Args:
        chunks: List of chunk dictionaries

    Returns:
        List of heading dictionaries
    """
    headings = []
    seen_headings = set()

    for chunk in chunks:
        metadata = chunk.get("metadata", {})
        heading_path = metadata.get("heading_path", "")

        if heading_path and isinstance(heading_path, str):
            # Parse heading path
            path_parts = [h.strip() for h in heading_path.split(",")]

            for level, heading_text in enumerate(path_parts, start=1):
                if heading_text and heading_text not in seen_headings:
                    headings.append({
                        "text": heading_text,
                        "level": level,
                        "chapter": metadata.get("chapter_number"),
                    })
                    seen_headings.add(heading_text)

    return headings


def extract_text_from_chunks(chunks: List[Dict[str, Any]]) -> str:
    """
    Extract and concatenate text from all chunks.

    Args:
        chunks: List of chunk dictionaries

    Returns:
        Concatenated text string
    """
    text_parts = []

    for chunk in chunks:
        # Get text from metadata
        text = chunk.get("metadata", {}).get("text", "")
        if text:
            text_parts.append(text.strip())

    return "\n\n".join(text_parts)


def get_chunk_context(chunk: Dict[str, Any], all_chunks: List[Dict[str, Any]], context_window: int = 1) -> str:
    """
    Get surrounding context for a chunk (previous and next chunks).

    Args:
        chunk: Target chunk
        all_chunks: All chunks in order
        context_window: Number of chunks before/after to include

    Returns:
        Context string
    """
    chunk_id = chunk.get("id")

    # Find chunk index
    chunk_idx = None
    for idx, c in enumerate(all_chunks):
        if c.get("id") == chunk_id:
            chunk_idx = idx
            break

    if chunk_idx is None:
        return chunk.get("metadata", {}).get("text", "")

    # Get context chunks
    start_idx = max(0, chunk_idx - context_window)
    end_idx = min(len(all_chunks), chunk_idx + context_window + 1)

    context_chunks = all_chunks[start_idx:end_idx]

    return extract_text_from_chunks(context_chunks)


def calculate_section_stats(section: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate statistics for a section.

    Args:
        section: Section dictionary with chunks

    Returns:
        Statistics dictionary
    """
    chunks = section.get("chunks", [])

    total_chunks = len(chunks)
    total_chars = sum(len(c.get("metadata", {}).get("text", "")) for c in chunks)

    # Extract pages
    pages = set()
    for chunk in chunks:
        page_num = chunk.get("metadata", {}).get("page_number")
        if page_num:
            if isinstance(page_num, str):
                for p in page_num.split(','):
                    if p and p != "false":
                        pages.add(p)
            else:
                pages.add(str(page_num))

    return {
        "total_chunks": total_chunks,
        "total_characters": total_chars,
        "total_pages": len(pages),
        "avg_chunk_size": total_chars // total_chunks if total_chunks > 0 else 0
    }


# Testing
if __name__ == "__main__":
    # Example test data
    test_chunks = [
        {
            "id": "doc1::chunk_0",
            "metadata": {
                "text": "Introduction to the topic...",
                "chapter_number": "1",
                "chapter_title": "Introduction",
                "page_number": "1",
                "heading_path": "Chapter 1, Introduction"
            }
        },
        {
            "id": "doc1::chunk_1",
            "metadata": {
                "text": "More introductory content...",
                "chapter_number": "1",
                "chapter_title": "Introduction",
                "page_number": "2",
                "heading_path": "Chapter 1, Introduction, Background"
            }
        },
        {
            "id": "doc1::chunk_2",
            "metadata": {
                "text": "Chapter 2 begins here...",
                "chapter_number": "2",
                "chapter_title": "Methods",
                "page_number": "5",
                "heading_path": "Chapter 2, Methods"
            }
        }
    ]

    print("Testing chunk organization...")
    organized = organize_chunks_by_hierarchy(test_chunks)

    print(f"\nChapters: {len(organized['chapters'])}")
    for chapter in organized['chapters']:
        print(f"  - Chapter {chapter['chapter_number']}: {chapter['chapter_title']} ({len(chapter['chunks'])} chunks)")

    print(f"\nMetadata:")
    for key, value in organized['metadata'].items():
        print(f"  - {key}: {value}")
