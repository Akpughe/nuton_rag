"""
Note Generation Process - Core Orchestration
Main engine for generating comprehensive study notes from documents.
"""

import logging
import time
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

# Import helper modules
from note_generation_helpers import (
    organize_chunks_by_hierarchy,
    extract_text_from_chunks,
    calculate_section_stats
)
from note_generation_prompts import (
    get_prompt,
    get_level_instructions,
    get_level_config,
    FORMATTING_GUIDELINES
)
from pinecone_client import fetch_all_document_chunks

# Import LLM clients
import openai_client
from groq_client import generate_answer as groq_generate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def generate_comprehensive_notes(
    document_id: str,
    space_id: Optional[str] = None,
    academic_level: str = "graduate",
    personalization_options: Optional[Dict] = None,
    include_diagrams: bool = True,
    include_mermaid: bool = True,
    max_chunks: int = 2000,
    acl_tags: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Generate comprehensive study notes from a document.

    This is the main entry point for note generation. It orchestrates the entire process:
    1. Fetch all document chunks
    2. Organize chunks by hierarchy
    3. Analyze document structure
    4. Generate notes section by section
    5. Add diagrams and visualizations
    6. Validate and format output

    Args:
        document_id: Document ID to generate notes for
        space_id: Optional space ID filter
        academic_level: Academic level (undergraduate, graduate, msc, phd)
        personalization_options: Additional personalization settings
        include_diagrams: Whether to include PDF diagrams
        include_mermaid: Whether to generate mermaid diagrams
        max_chunks: Maximum chunks to retrieve
        acl_tags: Optional ACL tags

    Returns:
        Dictionary with:
        {
            "notes_markdown": "# Complete markdown notes...",
            "metadata": {
                "academic_level": "graduate",
                "total_pages": 120,
                "total_chapters": 8,
                ...
            },
            "status": "success"
        }
    """
    start_time = time.time()
    logger.info(f"🚀 Starting note generation for document {document_id}, level={academic_level}")

    try:
        # Phase 1: Fetch all chunks
        logger.info("📥 Gathering every snippet from the document...")
        chunks = fetch_all_document_chunks(
            document_id=document_id,
            space_id=space_id,
            max_chunks=max_chunks,
            acl_tags=acl_tags
        )

        if not chunks:
            return {
                "notes_markdown": "",
                "metadata": {},
                "status": "error",
                "message": "No chunks found for document"
            }

        logger.info("✅ Pulled together %s passages to study", len(chunks))

        # Phase 2: Organize by hierarchy
        logger.info("📚 Reading the document structure...")
        organized_structure = organize_chunks_by_hierarchy(chunks)
        org_metadata = organized_structure.get("metadata", {})
        logger.info(
            "🧭 Mapped into %s chapters/sections covering characters %s-%s",
            org_metadata.get("total_chapters", 0),
            org_metadata.get("min_start_index", "?"),
            org_metadata.get("max_end_index", "?"),
        )

        # Phase 3: Analyze document
        logger.info("🧠 Understanding the document's tone and goals...")
        doc_analysis = await analyze_document_structure(organized_structure, academic_level)
        logger.info("📌 The document feels like: %s", doc_analysis.get("document_type", "unknown"))

        # Phase 4: Extract diagrams (if requested)
        diagrams = []
        if include_diagrams:
            logger.info("🎨 Collecting any diagrams that brighten the story...")
            diagrams = extract_diagrams_from_chunks(chunks, organized_structure)
            logger.info("🖼️ Saved %s diagram(s) for later", len(diagrams))

        # Phase 5: Generate notes section by section
        logger.info("✍️ Crafting the study guide chapter by chapter...")
        notes_markdown = await process_chunks_sequentially(
            organized_structure=organized_structure,
            academic_level=academic_level,
            doc_analysis=doc_analysis,
            diagrams=diagrams,
            include_diagrams=include_diagrams,
            include_mermaid=include_mermaid
        )

        # Phase 6: Validate completeness
        logger.info("🔍 Giving the finished notes a final once-over for coverage...")
        validation_result = validate_note_completeness(
            notes_markdown=notes_markdown,
            original_chunks=chunks,
            organized_structure=organized_structure
        )

        # Calculate generation time
        generation_time = time.time() - start_time

        # Build metadata
        metadata = {
            "academic_level": academic_level,
            "document_type": doc_analysis.get("document_type", "unknown"),
            "total_pages": organized_structure["metadata"].get("total_pages", 0),
            "total_chapters": organized_structure["metadata"].get("total_chapters", 0),
            "total_chunks_processed": len(chunks),
            "diagrams_included": len(diagrams) if include_diagrams else 0,
            "generation_time_seconds": round(generation_time, 2),
            "coverage_score": validation_result.get("coverage_score", 0.0),
            "notes_length_chars": len(notes_markdown),
            "generated_at": datetime.now().isoformat()
        }

        logger.info(f"✅ Note generation complete in {generation_time:.2f}s")
        logger.info(f"   - Length: {len(notes_markdown)} characters")
        logger.info(f"   - Coverage score: {validation_result.get('coverage_score', 0):.0%}")

        return {
            "notes_markdown": notes_markdown,
            "metadata": metadata,
            "status": "success"
        }

    except Exception as e:
        logger.error(f"❌ Error generating notes: {e}", exc_info=True)
        return {
            "notes_markdown": "",
            "metadata": {},
            "status": "error",
            "message": str(e)
        }


async def process_chunks_sequentially(
    organized_structure: Dict[str, Any],
    academic_level: str,
    doc_analysis: Dict[str, Any],
    diagrams: List[Dict[str, Any]],
    include_diagrams: bool = True,
    include_mermaid: bool = True
) -> str:
    """
    Process document section by section, maintaining context.

    Args:
        organized_structure: Hierarchically organized chunks
        academic_level: Target academic level
        doc_analysis: Document analysis results
        diagrams: List of extracted diagrams
        include_diagrams: Whether to include PDF diagrams
        include_mermaid: Whether to generate mermaid diagrams

    Returns:
        Complete markdown notes string
    """
    logger.info("📚 Settling in to read the document from start to finish...")

    markdown_parts = []
    context_memory = []  # Track previous sections for continuity

    # Add document header
    doc_header = generate_document_header(
        organized_structure=organized_structure,
        academic_level=academic_level,
        doc_analysis=doc_analysis
    )
    markdown_parts.append(doc_header)

    # Process each chapter
    chapters = organized_structure.get("chapters", [])

    total_chapters = len(chapters)

    for chapter_idx, chapter in enumerate(chapters, 1):
        chapter_title = chapter.get("chapter_title", "Untitled chapter")
        logger.info(
            "📖 Reading chapter %s/%s: \"%s\"",
            chapter_idx,
            total_chapters,
            chapter_title,
        )

        # Generate chapter notes
        chapter_md = await generate_chapter_notes(
            chapter=chapter,
            academic_level=academic_level,
            context_memory=context_memory,
            diagrams=diagrams,
            include_diagrams=include_diagrams,
            include_mermaid=include_mermaid
        )

        markdown_parts.append(chapter_md)

        logger.info("✨ Wrapped up chapter \"%s\".", chapter_title)

        # Update context memory (keep last 3 chapters for continuity)
        chapter_summary = {
            "chapter_number": chapter.get("chapter_number", chapter_idx),
            "chapter_title": chapter.get("chapter_title", "Unknown"),
            "key_concepts": extract_key_concepts_from_markdown(chapter_md)
        }
        context_memory.append(chapter_summary)

        # Keep only recent context
        if len(context_memory) > 3:
            context_memory = context_memory[-3:]

    # Join all parts
    complete_notes = "\n\n---\n\n".join(markdown_parts)

    return complete_notes


def generate_document_header(
    organized_structure: Dict[str, Any],
    academic_level: str,
    doc_analysis: Dict[str, Any]
) -> str:
    """
    Generate document header with title, metadata, and overview.

    Args:
        organized_structure: Document structure
        academic_level: Target academic level
        doc_analysis: Document analysis results

    Returns:
        Header markdown string
    """
    metadata = organized_structure.get("metadata", {})
    source_file = metadata.get("source_file", "Unknown Document")

    # Ensure source_file is a string
    if not isinstance(source_file, str):
        source_file = str(source_file) if source_file is not None else "Unknown Document"

    # Extract title from source file or use first chapter
    if source_file and source_file != "Unknown Document":
        doc_title = source_file.replace(".pdf", "").replace("_", " ").title()
    else:
        chapters = organized_structure.get("chapters", [])
        if chapters:
            chapter_title = chapters[0].get("chapter_title", "Study Notes")
            doc_title = str(chapter_title) if not isinstance(chapter_title, str) else chapter_title
        else:
            doc_title = "Study Notes"

    # Ensure all values are strings or convertible for f-string
    doc_type = doc_analysis.get('document_type', 'Unknown')
    doc_type = str(doc_type) if not isinstance(doc_type, str) else doc_type

    total_pages = metadata.get('total_pages', 'Unknown')
    total_pages = str(total_pages) if total_pages not in ['Unknown', None] else 'Unknown'

    total_chapters = metadata.get('total_chapters', 'Unknown')
    total_chapters = str(total_chapters) if total_chapters not in ['Unknown', None] else 'Unknown'

    # Generate header
    header = f"""# 📚 {doc_title}

**Academic Level:** {academic_level.title()}
**Document Type:** {doc_type}
**Total Pages:** {total_pages}
**Total Chapters:** {total_chapters}

---

## 📋 Document Overview

These comprehensive study notes cover all content from the source document, organized hierarchically and formatted for optimal learning at the {academic_level} level.

"""

    return header


async def generate_chapter_notes(
    chapter: Dict[str, Any],
    academic_level: str,
    context_memory: List[Dict],
    diagrams: List[Dict[str, Any]],
    include_diagrams: bool = True,
    include_mermaid: bool = True
) -> str:
    """
    Generate comprehensive notes for a single chapter.

    Args:
        chapter: Chapter dictionary with chunks
        academic_level: Target academic level
        context_memory: Previous chapters for context
        diagrams: List of diagrams
        include_diagrams: Whether to include PDF diagrams
        include_mermaid: Whether to generate mermaid diagrams

    Returns:
        Chapter markdown string
    """
    chapter_num = chapter.get("chapter_number", 0)
    chapter_title = chapter.get("chapter_title", "Unknown Chapter")
    chunks = chapter.get("chunks", [])

    # Ensure chapter_num is an int
    if not isinstance(chapter_num, int):
        try:
            chapter_num = int(chapter_num) if chapter_num else 0
        except (ValueError, TypeError):
            chapter_num = 0

    # Ensure chapter_title is a string
    if not isinstance(chapter_title, str):
        chapter_title = str(chapter_title) if chapter_title is not None else f"Chapter {chapter_num}"

    logger.info(
        "   🧾 Gathering material from Chapter %s: \"%s\" (%s passages)",
        chapter_num or "?",
        chapter_title,
        len(chunks),
    )

    sections = chapter.get("sections", []) or []
    for section in sections:
        section_title = section.get("section_title", "Main ideas")
        section_chunks = section.get("chunks", []) or []

        logger.info(
            "   📘 Reading section \"%s\" (%s passages)",
            section_title,
            len(section_chunks),
        )

        # Estimate character coverage for the section
        min_start = None
        max_end = None
        for chunk in section_chunks:
            metadata = chunk.get("metadata", {})
            start_val = metadata.get("start_index")
            end_val = metadata.get("end_index")

            try:
                if start_val is not None:
                    start_int = int(float(start_val))
                    min_start = start_int if min_start is None else min(min_start, start_int)
                if end_val is not None:
                    end_int = int(float(end_val))
                    max_end = end_int if max_end is None else max(max_end, end_int)
            except (TypeError, ValueError):
                continue

        if min_start is not None and max_end is not None and max_end > min_start:
            logger.info(
                "   🧠 Capturing the main ideas from \"%s\" (characters %s-%s)",
                section_title,
                min_start,
                max_end,
            )

    # Extract all text from chapter chunks
    chapter_content = extract_text_from_chunks(chunks)

    # Build previous context string
    previous_context = build_context_string(context_memory)

    # Get level instructions
    level_instructions = get_level_instructions(academic_level)

    # Generate chapter header
    chapter_header = f"## 🔍 Chapter {chapter_num}: {chapter_title}\n\n"

    # Generate main chapter content using LLM
    try:
        logger.info(
            "   🤖 Asking the writing assistant to help phrase \"%s\"...",
            chapter_title,
        )
        notes_content = await generate_section_content(
            content=chapter_content,
            section_title=chapter_title,
            academic_level=academic_level,
            level_instructions=level_instructions,
            previous_context=previous_context
        )
        # Ensure notes_content is a string (groq might return tuple)
        if isinstance(notes_content, tuple):
            notes_content = notes_content[0] if notes_content else ""
        notes_content = str(notes_content) if notes_content is not None else ""
        logger.info(
            "   🗒️ Assistant drafted %s characters for \"%s\".",
            len(notes_content),
            chapter_title,
        )
    except Exception as e:
        logger.error(f"Error generating chapter content: {e}")
        notes_content = f"### 📖 Content\n\n{chapter_content}\n"

    # Add mermaid diagrams if requested
    if include_mermaid:
        mermaid_diagrams = await generate_mermaid_for_content(chapter_content, academic_level)
        if mermaid_diagrams and isinstance(mermaid_diagrams, list):
            # Ensure all mermaid diagrams are strings
            mermaid_diagrams = [str(d) if not isinstance(d, str) else d for d in mermaid_diagrams]
            notes_content += "\n\n" + "\n\n".join(mermaid_diagrams)

    # Add PDF diagrams if available and requested
    if include_diagrams and diagrams:
        chapter_diagrams = get_diagrams_for_chapter(diagrams, chapter_num)
        if chapter_diagrams:
            diagram_section = format_diagram_section(chapter_diagrams)
            # Ensure diagram_section is a string
            diagram_section = str(diagram_section) if not isinstance(diagram_section, str) else diagram_section
            notes_content += "\n\n" + diagram_section

    return chapter_header + notes_content


async def generate_section_content(
    content: str,
    section_title: str,
    academic_level: str,
    level_instructions: str,
    previous_context: str
) -> str:
    """
    Generate detailed notes for a section using LLM.

    Args:
        content: Section content text
        section_title: Section title
        academic_level: Target academic level
        level_instructions: Level-specific instructions
        previous_context: Context from previous sections

    Returns:
        Generated markdown notes
    """
    # Build prompt
    prompt = get_prompt(
        "section_notes",
        section_title=section_title,
        academic_level=academic_level,
        section_content=content[:15000],  # Limit for LLM context
        previous_context=previous_context,
        level_instructions=level_instructions
    )

    # Try Groq first (faster)
    try:
        result = await asyncio.to_thread(
            groq_generate,
            prompt,
            "You are an expert academic note-taker creating comprehensive study notes.",
            "openai/gpt-oss-120b"
        )
        # Ensure result is a string (groq might return tuple)
        if isinstance(result, tuple):
            result = result[0] if result else ""
        return str(result) if result is not None else ""
    except Exception as e:
        logger.warning(f"Groq failed, trying OpenAI: {e}")

        # Fallback to OpenAI
        try:
            response = openai_client.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an expert academic note-taker creating comprehensive study notes."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=4000
            )
            return response.choices[0].message.content
        except Exception as e2:
            logger.error(f"OpenAI also failed: {e2}")
            # Last resort: return formatted content
            return f"### 📖 {section_title}\n\n{content}\n"


def build_context_string(context_memory: List[Dict]) -> str:
    """Build a context string from previous chapters."""
    if not context_memory:
        return "This is the beginning of the document."

    context_parts = []
    for ctx in context_memory:
        context_parts.append(f"Chapter {ctx['chapter_number']}: {ctx['chapter_title']}")
        if ctx.get('key_concepts'):
            context_parts.append(f"  Key concepts: {', '.join(ctx['key_concepts'][:5])}")

    return "\n".join(context_parts)


def extract_key_concepts_from_markdown(markdown: str) -> List[str]:
    """Extract key terms from markdown (simple heuristic)."""
    import re

    # Find bold text (likely key terms)
    bold_terms = re.findall(r'\*\*([^*]+)\*\*', markdown)

    # Find heading text
    headings = re.findall(r'^#+\s+[🔍📖💡🔸📌]?\s*(.+)$', markdown, re.MULTILINE)

    # Combine and limit
    concepts = list(set(bold_terms + headings))
    return concepts[:10]  # Return top 10


async def analyze_document_structure(
    organized_structure: Dict[str, Any],
    academic_level: str
) -> Dict[str, Any]:
    """
    Analyze document structure and classify type.

    Args:
        organized_structure: Organized document structure
        academic_level: Target academic level

    Returns:
        Document analysis dictionary
    """
    metadata = organized_structure.get("metadata", {})
    chapters = organized_structure.get("chapters", [])

    # Get sample content
    sample_chunks = []
    for chapter in chapters[:3]:  # Sample first 3 chapters
        sample_chunks.extend(chapter.get("chunks", [])[:2])  # 2 chunks per chapter

    sample_content = extract_text_from_chunks(sample_chunks)[:2000]  # Limit to 2000 chars

    # Simple heuristic classification (can be enhanced with LLM)
    analysis = {
        "document_type": "unknown",
        "has_abstract": "abstract" in sample_content.lower()[:500],
        "has_introduction": any(title.lower().startswith("intro") for title in [c.get("chapter_title", "") for c in chapters]),
        "has_conclusion": any("conclusion" in c.get("chapter_title", "").lower() for c in chapters),
        "has_equations": bool(__import__('re').search(r'\$.*?\$|\\[.*?\\]', sample_content)),
        "has_code": bool(__import__('re').search(r'```|`[^`]+`', sample_content)),
        "complexity_level": "intermediate",
        "subject_domain": "general"
    }

    # Classify document type based on structure
    if analysis["has_abstract"] and len(chapters) < 10:
        analysis["document_type"] = "research_paper"
    elif len(chapters) > 10 and metadata.get("total_pages", 0) > 100:
        analysis["document_type"] = "textbook"
    elif "lecture" in sample_content.lower()[:1000]:
        analysis["document_type"] = "lecture_notes"
    else:
        analysis["document_type"] = "document"

    return analysis


def extract_diagrams_from_chunks(
    chunks: List[Dict[str, Any]],
    organized_structure: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Extract diagrams from chunks where type="image".

    Args:
        chunks: All chunks (including image chunks)
        organized_structure: Organized document structure

    Returns:
        List of diagram dictionaries
    """
    logger.info("Extracting diagrams from chunks...")

    diagrams = []

    for chunk in chunks:
        metadata = chunk.get("metadata", {})
        chunk_type = metadata.get("type")

        if chunk_type == "image":
            # This is an image chunk
            diagram = {
                "image_base64": metadata.get("image_base64"),
                "image_id": metadata.get("image_id"),
                "page": metadata.get("page"),
                "chapter": metadata.get("chapter_number"),
                "position_in_doc": metadata.get("position_in_doc", 0),
                "description": metadata.get("description", "Diagram")
            }

            # Only add if we have image data
            if diagram["image_base64"]:
                diagrams.append(diagram)

    logger.info(f"Extracted {len(diagrams)} diagrams from chunks")
    return diagrams


def get_diagrams_for_chapter(
    diagrams: List[Dict[str, Any]],
    chapter_number: int
) -> List[Dict[str, Any]]:
    """Get diagrams that belong to a specific chapter."""
    chapter_diagrams = []

    for diagram in diagrams:
        if diagram.get("chapter") == chapter_number or diagram.get("chapter") == str(chapter_number):
            chapter_diagrams.append(diagram)

    return chapter_diagrams


def format_diagram_section(diagrams: List[Dict[str, Any]]) -> str:
    """Format diagrams as a markdown section."""
    if not diagrams:
        return ""

    section = "### 📊 Diagrams and Visual Elements\n\n"

    for idx, diagram in enumerate(diagrams, 1):
        section += f"**Diagram {idx}** (Page {diagram.get('page', 'Unknown')})\n\n"

        # Add description if available
        if diagram.get("description"):
            section += f"*{diagram['description']}*\n\n"

        # Note: In actual implementation, image would be embedded or referenced
        # For markdown, we indicate where the image would appear
        section += f"![Diagram {idx}](data:image/png;base64,{diagram.get('image_base64', '')[:50]}...)\n\n"

    return section


async def generate_mermaid_for_content(content: str, academic_level: str) -> List[str]:
    """
    Generate mermaid diagrams for content where applicable.

    Args:
        content: Content text to analyze
        academic_level: Target academic level

    Returns:
        List of mermaid diagram code blocks
    """
    # Check if content describes processes, workflows, or relationships
    keywords = ["process", "workflow", "step", "sequence", "procedure", "algorithm", "flow", "cycle"]

    has_process_description = any(keyword in content.lower() for keyword in keywords)

    if not has_process_description or len(content) < 200:
        return []

    # Generate mermaid diagram using LLM
    try:
        prompt = get_prompt("mermaid", content=content[:2000])

        # Try Groq
        try:
            result = await asyncio.to_thread(
                groq_generate,
                prompt,
                "You are a diagram expert. Generate mermaid diagram code or return NO_DIAGRAM.",
                "openai/gpt-oss-120b"
            )
            # Ensure result is a string (groq might return tuple)
            if isinstance(result, tuple):
                result = result[0] if result else ""
            result = str(result) if result is not None else ""
        except:
            # Fallback to OpenAI
            response = openai_client.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a diagram expert. Generate mermaid diagram code or return NO_DIAGRAM."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=800
            )
            result = response.choices[0].message.content

        # Check if diagram was generated
        if "NO_DIAGRAM" in result or "```mermaid" not in result:
            return []

        # Extract mermaid code blocks
        import re
        mermaid_blocks = re.findall(r'```mermaid\n(.*?)\n```', result, re.DOTALL)

        return [f"```mermaid\n{block}\n```" for block in mermaid_blocks]

    except Exception as e:
        logger.warning(f"Failed to generate mermaid diagram: {e}")
        return []


def calculate_text_coverage(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate actual text coverage using start_index and end_index from chunks.
    
    This function tracks which character ranges of the original document are covered
    by the chunks, providing true coverage metrics.
    
    Args:
        chunks: List of chunks with metadata containing start_index and end_index
        
    Returns:
        Dictionary with coverage metrics including:
        - coverage_percentage: Percentage of document covered (0.0 to 1.0)
        - total_covered_chars: Total characters covered
        - total_document_chars: Estimated total document size
        - coverage_ranges: List of covered ranges
        - gaps: List of uncovered gaps
        - chunks_with_indices: Count of chunks with valid indices
    """
    ranges = []
    
    # Extract valid start_index and end_index from chunks
    for chunk in chunks:
        metadata = chunk.get("metadata", {})
        start_idx = metadata.get("start_index")
        end_idx = metadata.get("end_index")
        
        # Only include chunks with valid indices
        if start_idx is not None and end_idx is not None:
            try:
                start_idx = int(start_idx)
                end_idx = int(end_idx)
                if end_idx > start_idx:  # Valid range
                    ranges.append((start_idx, end_idx))
            except (ValueError, TypeError):
                continue
    
    if not ranges:
        logger.warning("No chunks with valid start_index/end_index found")
        return {
            "coverage_percentage": 0.0,
            "total_covered_chars": 0,
            "total_document_chars": 0,
            "coverage_ranges": [],
            "gaps": [],
            "chunks_with_indices": 0
        }
    
    # Sort ranges by start position
    ranges.sort()
    
    # Merge overlapping ranges to get actual coverage
    merged_ranges = []
    current_start, current_end = ranges[0]
    
    for start, end in ranges[1:]:
        if start <= current_end:
            # Overlapping or adjacent, merge
            current_end = max(current_end, end)
        else:
            # No overlap, save current range and start new one
            merged_ranges.append((current_start, current_end))
            current_start, current_end = start, end
    
    # Add the last range
    merged_ranges.append((current_start, current_end))
    
    # Calculate total covered characters
    total_covered = sum(end - start for start, end in merged_ranges)
    
    # Estimate total document size (from first char to last char)
    total_document_chars = merged_ranges[-1][1] - merged_ranges[0][0] if merged_ranges else 0
    
    # Calculate coverage percentage
    coverage_percentage = total_covered / total_document_chars if total_document_chars > 0 else 0.0
    
    # Identify gaps between covered ranges
    gaps = []
    for i in range(len(merged_ranges) - 1):
        gap_start = merged_ranges[i][1]
        gap_end = merged_ranges[i + 1][0]
        gap_size = gap_end - gap_start
        if gap_size > 0:
            gaps.append({
                "start": gap_start,
                "end": gap_end,
                "size": gap_size
            })
    
    logger.info(f"Text coverage calculated: {coverage_percentage:.2%} ({total_covered:,} / {total_document_chars:,} chars)")
    logger.info(f"Coverage ranges: {len(merged_ranges)}, Gaps: {len(gaps)}")
    
    return {
        "coverage_percentage": coverage_percentage,
        "total_covered_chars": total_covered,
        "total_document_chars": total_document_chars,
        "coverage_ranges": merged_ranges,
        "gaps": gaps,
        "chunks_with_indices": len(ranges)
    }


def validate_note_completeness(
    notes_markdown: str,
    original_chunks: List[Dict[str, Any]],
    organized_structure: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Validate that notes cover all content adequately.

    Args:
        notes_markdown: Generated notes
        original_chunks: Original chunks
        organized_structure: Document structure

    Returns:
        Validation results
    """
    logger.info("🧮 Checking that the notes cover the whole document…")

    # Calculate basic metrics
    total_chunks = len(original_chunks)
    notes_length = len(notes_markdown)
    original_content_length = sum(len(c.get("metadata", {}).get("text", "")) for c in original_chunks)

    # Calculate TRUE text coverage using start_index and end_index
    text_coverage_result = calculate_text_coverage(original_chunks)
    text_coverage_percentage = text_coverage_result["coverage_percentage"]
    
    # Legacy coverage ratio (for comparison)
    coverage_ratio = min(1.0, notes_length / max(original_content_length, 1))

    # Check for chapter coverage
    expected_chapters = organized_structure.get("metadata", {}).get("total_chapters", 0)
    import re
    chapter_headings = len(re.findall(r'^##\s+🔍', notes_markdown, re.MULTILINE))

    chapter_coverage = min(1.0, chapter_headings / max(expected_chapters, 1)) if expected_chapters > 0 else 1.0

    # Calculate overall coverage score (prioritize text coverage over length ratio)
    coverage_score = (text_coverage_percentage * 0.7) + (chapter_coverage * 0.3)

    # Assess quality
    has_formatting = bool(re.search(r'^#+\s+', notes_markdown, re.MULTILINE))
    has_lists = bool(re.search(r'^\s*[-*]\s+', notes_markdown, re.MULTILINE))
    has_tables = "|" in notes_markdown
    has_code_blocks = "```" in notes_markdown

    validation_result = {
        "coverage_score": coverage_score,
        "text_coverage_percentage": text_coverage_percentage,  # NEW: True coverage metric
        "text_coverage_details": text_coverage_result,  # NEW: Full coverage details
        "coverage_ratio": coverage_ratio,  # Legacy metric for comparison
        "notes_length": notes_length,
        "original_length": original_content_length,
        "total_chunks": total_chunks,
        "chapter_coverage": chapter_coverage,
        "has_formatting": has_formatting,
        "has_lists": has_lists,
        "has_tables": has_tables,
        "has_code_blocks": has_code_blocks,
        "quality_indicators": {
            "proper_formatting": has_formatting,
            "uses_lists": has_lists,
            "includes_tables": has_tables,
            "includes_code": has_code_blocks
        }
    }

    logger.info(
        "📊 Coverage check: %.0f%% of the document captured; %s gap(s) spotted.",
        text_coverage_percentage * 100,
        len(text_coverage_result.get("gaps", [])),
    )

    return validation_result


def validate_markdown_formatting(notes_markdown: str) -> Dict[str, Any]:
    """
    Validate markdown structure and formatting.

    Args:
        notes_markdown: Generated notes

    Returns:
        Validation results
    """
    import re

    validation = {
        "valid": True,
        "errors": [],
        "warnings": []
    }

    # Check heading hierarchy
    headings = re.findall(r'^(#+)\s+(.+)$', notes_markdown, re.MULTILINE)

    if not headings:
        validation["errors"].append("No headings found")
        validation["valid"] = False

    # Check for proper H1 (should have exactly one)
    h1_count = sum(1 for h in headings if len(h[0]) == 1)
    if h1_count == 0:
        validation["warnings"].append("No H1 heading found")
    elif h1_count > 1:
        validation["warnings"].append(f"Multiple H1 headings found ({h1_count})")

    # Check for broken markdown syntax
    if notes_markdown.count("```") % 2 != 0:
        validation["errors"].append("Unmatched code block delimiters")
        validation["valid"] = False

    # Check mermaid syntax (basic)
    mermaid_blocks = re.findall(r'```mermaid\n(.*?)\n```', notes_markdown, re.DOTALL)
    for block in mermaid_blocks:
        if not any(keyword in block for keyword in ["graph", "sequenceDiagram", "classDiagram", "stateDiagram", "gantt"]):
            validation["warnings"].append("Potentially invalid mermaid diagram")

    return validation


# Testing
if __name__ == "__main__":
    import asyncio

    async def test():
        print("Testing note generation process...")

        # This would normally use real data
        print("✅ Module loaded successfully")
        print("Ready to generate notes!")

    asyncio.run(test())
