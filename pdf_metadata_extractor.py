"""
Enhanced PDF Metadata Extractor for RAG Systems
Extracts comprehensive metadata: pages, chapters, sections, headings, structure.
Designed for Pinecone vector DB integration.
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import pdfplumber
from collections import defaultdict
import logging

# Import text cleaner
from pdf_text_cleaner import PDFTextCleaner

# Import multi-strategy extraction
from pdf_extraction_strategies import PDFExtractionStrategy

logging.basicConfig(level=logging.INFO)


class PDFMetadataExtractor:
    """
    Extract comprehensive metadata from PDFs for RAG systems.

    Features:
    - Page tracking
    - Chapter/section detection
    - Heading hierarchy
    - Font-based structure detection
    - Table/figure references
    - Position tracking
    """

    def __init__(self):
        # Initialize text cleaner
        self.text_cleaner = PDFTextCleaner()

        # Initialize multi-strategy extraction
        self.extraction_strategy = PDFExtractionStrategy()

        # Chapter/section patterns (ordered by priority)
        self.chapter_patterns = [
            # "Chapter 1: Title" or "Chapter I: Title"
            (r'(?i)^chapter\s+(\d+|[ivxlcdm]+)[\s:]+(.+?)$', 1),
            # "CHAPTER 1" or "CHAPTER ONE"
            (r'(?i)^chapter\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten).*$', 1),
            # "1. Title" or "1 Title" at start of line
            (r'^(\d+)\.?\s+([A-Z][A-Za-z\s]+)$', 1),
            # "Section 1.1: Title"
            (r'(?i)^section\s+(\d+\.?\d*)[\s:]+(.+?)$', 2),
            # "1.1 Title" (subsection)
            (r'^(\d+\.\d+)\.?\s+([A-Z][A-Za-z\s]+)$', 2),
            # ALL CAPS HEADINGS (at least 3 words)
            (r'^([A-Z][A-Z\s]{10,})$', 1),
        ]

        # Common section names
        self.standard_sections = [
            'abstract', 'introduction', 'background', 'related work',
            'methodology', 'methods', 'approach', 'implementation',
            'results', 'discussion', 'conclusion', 'future work',
            'references', 'bibliography', 'appendix'
        ]

    def extract_full_metadata(
        self,
        pdf_path: str,
        detect_chapters: bool = True,
        detect_fonts: bool = True,
        detect_structure: bool = True
    ) -> Dict[str, Any]:
        """
        Extract complete metadata from a PDF.

        Args:
            pdf_path: Path to PDF file
            detect_chapters: Detect chapters/sections
            detect_fonts: Extract font information
            detect_structure: Detect document structure

        Returns:
            Dict with pages, text, and metadata
        """
        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        logging.info(f"Extracting metadata from: {pdf_path.name}")

        result = {
            'file_name': pdf_path.name,
            'total_pages': 0,
            'pages': [],
            'full_text': '',
            'chapters': [],
            'headings': [],
            'structure': {},
            'metadata_quality': {}
        }

        # Step 1: Extract text using best-quality strategy
        logging.info("Using multi-strategy extraction...")
        extraction_result = self.extraction_strategy.extract_best(str(pdf_path))

        raw_text = extraction_result['text']
        extraction_strategy_used = extraction_result['strategy']
        extraction_quality_score = extraction_result['quality_score']

        logging.info(f"Extraction strategy: {extraction_strategy_used}")
        logging.info(f"Extraction quality: {extraction_quality_score}/100")

        # Step 2: Clean the extracted text
        cleaned_text = self.text_cleaner.clean(raw_text, aggressive=True)

        # Step 3: Analyze text quality after cleaning
        text_quality_after = self.text_cleaner.detect_quality(cleaned_text)

        # Step 4: Extract page-level metadata using pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            result['total_pages'] = len(pdf.pages)

            full_text = []
            char_position = 0

            for page_num, page in enumerate(pdf.pages, 1):
                logging.info(f"Processing page {page_num}/{len(pdf.pages)}")

                # Get this page's text from the cleaned text
                # We'll use a simple heuristic: split by page count
                page_start = int((page_num - 1) * len(cleaned_text) / len(pdf.pages))
                page_end = int(page_num * len(cleaned_text) / len(pdf.pages))
                page_text = cleaned_text[page_start:page_end]

                # Extract page metadata
                page_data = {
                    'page_num': page_num,
                    'text': page_text,
                    'start_char': char_position,
                    'end_char': char_position + len(page_text),
                    'has_tables': False,
                    'has_images': False,
                    'headings': []
                }

                # Detect tables
                tables = page.extract_tables()
                if tables:
                    page_data['has_tables'] = True
                    page_data['table_count'] = len(tables)

                # Detect images
                if page.images:
                    page_data['has_images'] = True
                    page_data['image_count'] = len(page.images)

                # Extract font information if requested
                if detect_fonts and page.chars:
                    font_info = self._extract_font_info(page)
                    page_data['font_info'] = font_info

                    # Detect headings by font size
                    headings = self._detect_headings_by_font(page, page_text, char_position)
                    page_data['headings'] = headings
                    result['headings'].extend(headings)

                result['pages'].append(page_data)
                full_text.append(page_text)
                char_position += len(page_text) + 1  # +1 for newline

            result['full_text'] = cleaned_text

        # Store comprehensive text quality metrics
        result['text_quality'] = {
            **text_quality_after,
            'extraction_strategy': extraction_strategy_used,
            'extraction_quality_raw': extraction_quality_score,
        }

        logging.info(f"Text quality score (after cleaning): {text_quality_after.get('quality_score', 0)}/100")
        logging.info(f"Broken words detected: {text_quality_after.get('broken_word_count', 0)}")

        # Detect chapters/sections
        if detect_chapters:
            chapters = self._detect_chapters(result)
            result['chapters'] = chapters

        # Detect document structure
        if detect_structure:
            structure = self._detect_structure(result)
            result['structure'] = structure

        # Calculate metadata quality score
        result['metadata_quality'] = self._calculate_quality_score(result)

        logging.info(f"Extracted metadata: {len(result['pages'])} pages, "
                    f"{len(result['chapters'])} chapters, "
                    f"{len(result['headings'])} headings")

        return result

    def _extract_font_info(self, page) -> Dict[str, Any]:
        """Extract font information from page."""
        if not page.chars:
            return {}

        font_sizes = [char.get('size', 0) for char in page.chars if char.get('size')]

        if not font_sizes:
            return {}

        return {
            'avg_size': sum(font_sizes) / len(font_sizes),
            'min_size': min(font_sizes),
            'max_size': max(font_sizes),
            'unique_sizes': len(set(font_sizes))
        }

    def _detect_headings_by_font(
        self,
        page,
        page_text: str,
        start_char: int
    ) -> List[Dict[str, Any]]:
        """Detect headings based on font size."""
        if not page.chars:
            return []

        # Get average font size
        font_sizes = [char.get('size', 0) for char in page.chars if char.get('size')]
        if not font_sizes:
            return []

        avg_font = sum(font_sizes) / len(font_sizes)

        # Group characters by line
        lines = defaultdict(list)
        for char in page.chars:
            if 'size' in char and 'text' in char:
                # Round y-position to group into lines
                line_y = round(char.get('top', 0))
                lines[line_y].append(char)

        headings = []
        for line_y, chars in lines.items():
            # Check if this line has larger font
            line_sizes = [c.get('size', 0) for c in chars]
            if not line_sizes:
                continue

            avg_line_size = sum(line_sizes) / len(line_sizes)

            # Heading if 20% larger than average
            if avg_line_size > avg_font * 1.2:
                line_text = ''.join(c.get('text', '') for c in chars).strip()

                if len(line_text) > 3:  # Skip very short text
                    heading_level = self._determine_heading_level(avg_line_size, avg_font)

                    headings.append({
                        'text': line_text,
                        'level': heading_level,
                        'font_size': avg_line_size,
                        'position': start_char + page_text.find(line_text) if line_text in page_text else start_char
                    })

        return headings

    def _determine_heading_level(self, font_size: float, avg_font: float) -> int:
        """Determine heading level based on font size ratio."""
        ratio = font_size / avg_font

        if ratio > 2.0:
            return 1  # H1
        elif ratio > 1.6:
            return 2  # H2
        elif ratio > 1.3:
            return 3  # H3
        else:
            return 4  # H4

    def _detect_chapters(self, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect chapters and sections using pattern matching."""
        text = result['full_text']
        lines = text.split('\n')

        chapters = []

        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue

            # Try each pattern
            for pattern, level in self.chapter_patterns:
                match = re.match(pattern, line)
                if match:
                    # Find position in full text
                    position = text.find(line)

                    chapter = {
                        'title': line,
                        'level': level,
                        'position': position,
                        'line_number': i,
                        'type': 'chapter' if level == 1 else 'section'
                    }

                    # Extract chapter number if available
                    if match.groups():
                        chapter['number'] = match.group(1)

                    # Determine page number
                    chapter['page'] = self._position_to_page(position, result['pages'])

                    chapters.append(chapter)
                    break

        # Also detect standard sections
        for section_name in self.standard_sections:
            pattern = rf'(?i)^{re.escape(section_name)}s?\s*$'
            for i, line in enumerate(lines):
                if re.match(pattern, line.strip()):
                    position = text.find(line)
                    chapters.append({
                        'title': line.strip(),
                        'level': 1,
                        'position': position,
                        'line_number': i,
                        'type': 'standard_section',
                        'page': self._position_to_page(position, result['pages'])
                    })

        # Sort by position
        chapters.sort(key=lambda x: x['position'])

        # Remove duplicates (same position)
        unique_chapters = []
        seen_positions = set()
        for chapter in chapters:
            if chapter['position'] not in seen_positions:
                unique_chapters.append(chapter)
                seen_positions.add(chapter['position'])

        return unique_chapters

    def _detect_structure(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Detect overall document structure."""
        structure = {
            'has_abstract': False,
            'has_introduction': False,
            'has_conclusion': False,
            'has_references': False,
            'has_appendix': False,
            'sections': []
        }

        text_lower = result['full_text'].lower()

        # Detect standard sections
        if 'abstract' in text_lower:
            structure['has_abstract'] = True
        if 'introduction' in text_lower:
            structure['has_introduction'] = True
        if 'conclusion' in text_lower or 'concluding' in text_lower:
            structure['has_conclusion'] = True
        if 'references' in text_lower or 'bibliography' in text_lower:
            structure['has_references'] = True
        if 'appendix' in text_lower:
            structure['has_appendix'] = True

        # Build section hierarchy
        for chapter in result.get('chapters', []):
            structure['sections'].append({
                'title': chapter['title'],
                'level': chapter['level'],
                'page': chapter['page']
            })

        return structure

    def _position_to_page(self, position: int, pages: List[Dict]) -> int:
        """Convert character position to page number."""
        for page in pages:
            if page['start_char'] <= position <= page['end_char']:
                return page['page_num']
        return 1  # Default to first page

    def _calculate_quality_score(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate metadata extraction quality score."""
        score = {
            'has_chapters': len(result.get('chapters', [])) > 0,
            'has_headings': len(result.get('headings', [])) > 0,
            'has_structure': any(result.get('structure', {}).values()),
            'chapter_count': len(result.get('chapters', [])),
            'heading_count': len(result.get('headings', [])),
            'pages_with_tables': sum(1 for p in result['pages'] if p.get('has_tables')),
            'pages_with_images': sum(1 for p in result['pages'] if p.get('has_images')),
        }

        # Overall quality score (0-100)
        quality = 0
        if score['has_chapters']:
            quality += 30
        if score['has_headings']:
            quality += 20
        if score['has_structure']:
            quality += 20
        if score['chapter_count'] >= 3:
            quality += 15
        if score['heading_count'] >= 5:
            quality += 15

        score['overall_quality'] = min(quality, 100)

        return score

    def map_chunks_to_metadata(
        self,
        chunks: List[Dict[str, Any]],
        pdf_metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Map chunks to extracted metadata.

        Args:
            chunks: List of text chunks from chunker
            pdf_metadata: Metadata extracted from PDF

        Returns:
            Chunks enhanced with metadata
        """
        enhanced_chunks = []

        for chunk in chunks:
            enhanced = chunk.copy()

            start_pos = chunk.get('start_index', 0)
            end_pos = chunk.get('end_index', 0)

            # Find pages this chunk spans
            pages = self._get_chunk_pages(start_pos, end_pos, pdf_metadata['pages'])
            enhanced['pages'] = pages

            # Find chapter/section
            chapter = self._get_chunk_chapter(start_pos, pdf_metadata.get('chapters', []))
            if chapter:
                enhanced['chapter'] = chapter['title']
                enhanced['chapter_number'] = chapter.get('number')
                enhanced['section_level'] = chapter['level']

            # Find nearest heading
            heading = self._get_nearest_heading(start_pos, pdf_metadata.get('headings', []))
            if heading:
                enhanced['heading'] = heading['text']
                enhanced['heading_level'] = heading['level']

            # Position in document (0-1)
            enhanced['position_in_doc'] = start_pos / len(pdf_metadata['full_text']) if pdf_metadata['full_text'] else 0

            # Check for tables/figures in chunk
            enhanced['has_tables'] = any(
                pdf_metadata['pages'][p-1].get('has_tables', False)
                for p in pages if 0 < p <= len(pdf_metadata['pages'])
            )
            enhanced['has_images'] = any(
                pdf_metadata['pages'][p-1].get('has_images', False)
                for p in pages if 0 < p <= len(pdf_metadata['pages'])
            )

            # Extract figure/table references from text
            enhanced['figure_refs'] = self._extract_figure_refs(chunk['text'])
            enhanced['table_refs'] = self._extract_table_refs(chunk['text'])

            enhanced_chunks.append(enhanced)

        return enhanced_chunks

    def _get_chunk_pages(
        self,
        start_pos: int,
        end_pos: int,
        pages: List[Dict[str, Any]]
    ) -> List[int]:
        """Get pages that a chunk spans."""
        chunk_pages = []
        for page in pages:
            # Check if chunk overlaps with this page
            if not (end_pos < page['start_char'] or start_pos > page['end_char']):
                chunk_pages.append(page['page_num'])
        return chunk_pages

    def _get_chunk_chapter(
        self,
        position: int,
        chapters: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Get the chapter/section this chunk belongs to."""
        # Find the chapter that starts before this position
        relevant_chapter = None
        for chapter in reversed(chapters):  # Check from end
            if chapter['position'] <= position:
                relevant_chapter = chapter
                break
        return relevant_chapter

    def _get_nearest_heading(
        self,
        position: int,
        headings: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Get the nearest heading before this position."""
        relevant_heading = None
        for heading in reversed(headings):
            if heading['position'] <= position:
                relevant_heading = heading
                break
        return relevant_heading

    def _extract_figure_refs(self, text: str) -> List[str]:
        """Extract figure references from text."""
        pattern = r'(?i)(figure|fig\.?)\s+(\d+\.?\d*)'
        matches = re.findall(pattern, text)
        return [f"Figure {num}" for _, num in matches]

    def _extract_table_refs(self, text: str) -> List[str]:
        """Extract table references from text."""
        pattern = r'(?i)table\s+(\d+\.?\d*)'
        matches = re.findall(pattern, text)
        return [f"Table {num}" for num in matches]


# Convenience function
def extract_pdf_with_metadata(
    pdf_path: str,
    detect_chapters: bool = True,
    detect_fonts: bool = True,
    detect_structure: bool = True
) -> Dict[str, Any]:
    """
    Extract PDF with full metadata.

    Args:
        pdf_path: Path to PDF file
        detect_chapters: Detect chapters/sections
        detect_fonts: Extract font information
        detect_structure: Detect document structure

    Returns:
        Dictionary with full PDF metadata
    """
    extractor = PDFMetadataExtractor()
    return extractor.extract_full_metadata(
        pdf_path,
        detect_chapters=detect_chapters,
        detect_fonts=detect_fonts,
        detect_structure=detect_structure
    )
