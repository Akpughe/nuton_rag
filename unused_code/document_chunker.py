"""
Document Chunker - Large Document Splitting and Processing
Handles documents that exceed size/page limits for single-pass processing.

Features:
- Smart page-based splitting (configurable pages per chunk)
- Size-based splitting (configurable MB per chunk)
- Async parallel processing of chunks
- Result merging with page offset tracking
- Support for PDF documents

Author: RAG System Integration
Date: 2025
"""

import os
import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import PyPDF2
    from PyPDF2 import PdfReader, PdfWriter
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False
    logging.warning("PyPDF2 not available. Install with: pip install PyPDF2")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChunkInfo:
    """Information about a document chunk."""

    def __init__(
        self,
        chunk_id: int,
        file_path: str,
        start_page: int,
        end_page: int,
        page_count: int,
        is_temp: bool = False
    ):
        self.chunk_id = chunk_id
        self.file_path = file_path
        self.start_page = start_page
        self.end_page = end_page
        self.page_count = page_count
        self.is_temp = is_temp

    def __repr__(self):
        return (f"ChunkInfo(id={self.chunk_id}, pages={self.start_page}-{self.end_page}, "
                f"file={Path(self.file_path).name})")


class LargeDocumentChunker:
    """
    Handles splitting and processing of large documents.

    Automatically splits documents that exceed thresholds into
    manageable chunks for processing.
    """

    def __init__(
        self,
        max_pages: int = 10,
        max_mb: int = 10,
        temp_dir: Optional[str] = None
    ):
        """
        Initialize document chunker.

        Args:
            max_pages: Maximum pages per chunk
            max_mb: Maximum MB per chunk
            temp_dir: Temporary directory for chunk files (defaults to system temp)
        """
        self.max_pages = max_pages
        self.max_mb = max_mb
        self.temp_dir = temp_dir or tempfile.gettempdir()

        if not PYPDF2_AVAILABLE:
            raise ImportError("PyPDF2 required for document chunking. Install with: pip install PyPDF2")

        logger.info(f"Initialized LargeDocumentChunker: max_pages={max_pages}, max_mb={max_mb}")

    def should_chunk(self, file_path: str) -> bool:
        """
        Determine if document should be chunked.

        Args:
            file_path: Path to document

        Returns:
            True if document exceeds thresholds
        """
        file_path = Path(file_path)

        # Check file size
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        if file_size_mb > self.max_mb:
            logger.info(f"Document size ({file_size_mb:.1f}MB) exceeds threshold ({self.max_mb}MB)")
            return True

        # Check page count (PDF only)
        if file_path.suffix.lower() == '.pdf':
            try:
                with open(file_path, 'rb') as f:
                    reader = PdfReader(f)
                    page_count = len(reader.pages)

                if page_count > self.max_pages:
                    logger.info(f"Document pages ({page_count}) exceed threshold ({self.max_pages})")
                    return True
            except Exception as e:
                logger.warning(f"Could not check page count: {e}")

        return False

    def split_pdf(self, pdf_path: str) -> List[ChunkInfo]:
        """
        Split a PDF into chunks.

        Args:
            pdf_path: Path to PDF file

        Returns:
            List of ChunkInfo objects for each chunk
        """
        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        logger.info(f"Splitting PDF: {pdf_path.name}")

        # Read PDF
        with open(pdf_path, 'rb') as f:
            reader = PdfReader(f)
            total_pages = len(reader.pages)

        logger.info(f"Total pages: {total_pages}, splitting into chunks of {self.max_pages} pages")

        chunks = []

        # Split by pages
        for i in range(0, total_pages, self.max_pages):
            start_page = i
            end_page = min(i + self.max_pages, total_pages)

            # Create chunk file
            chunk_writer = PdfWriter()

            # Add pages to chunk
            with open(pdf_path, 'rb') as f:
                reader = PdfReader(f)
                for page_num in range(start_page, end_page):
                    chunk_writer.add_page(reader.pages[page_num])

            # Save chunk to temp file
            chunk_filename = f"{pdf_path.stem}_chunk_{i//self.max_pages}.pdf"
            chunk_path = Path(self.temp_dir) / chunk_filename

            with open(chunk_path, 'wb') as f:
                chunk_writer.write(f)

            # Create ChunkInfo
            chunk_info = ChunkInfo(
                chunk_id=i // self.max_pages,
                file_path=str(chunk_path),
                start_page=start_page + 1,  # 1-indexed
                end_page=end_page,  # 1-indexed
                page_count=end_page - start_page,
                is_temp=True
            )

            chunks.append(chunk_info)
            logger.info(f"Created {chunk_info}")

        logger.info(f"✅ Split into {len(chunks)} chunks")
        return chunks

    async def process_chunks_async(
        self,
        chunks: List[ChunkInfo],
        process_func,
        max_concurrent: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Process chunks asynchronously in parallel.

        Args:
            chunks: List of ChunkInfo objects
            process_func: Function to process each chunk (takes file_path, returns dict)
            max_concurrent: Maximum concurrent chunk processing

        Returns:
            List of processing results (one per chunk, in order)
        """
        logger.info(f"Processing {len(chunks)} chunks with max_concurrent={max_concurrent}")

        results = [None] * len(chunks)

        # Use ThreadPoolExecutor for CPU-bound tasks
        with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            # Submit all tasks
            future_to_chunk = {
                executor.submit(process_func, chunk.file_path): chunk
                for chunk in chunks
            }

            # Collect results as they complete
            for future in as_completed(future_to_chunk):
                chunk = future_to_chunk[future]

                try:
                    result = future.result()
                    results[chunk.chunk_id] = {
                        **result,
                        'chunk_id': chunk.chunk_id,
                        'chunk_start_page': chunk.start_page,
                        'chunk_end_page': chunk.end_page,
                    }
                    logger.info(f"✅ Processed {chunk}")

                except Exception as e:
                    logger.error(f"❌ Error processing {chunk}: {e}")
                    results[chunk.chunk_id] = {
                        'error': str(e),
                        'chunk_id': chunk.chunk_id,
                        'chunk_start_page': chunk.start_page,
                        'chunk_end_page': chunk.end_page,
                    }

        logger.info(f"✅ All {len(chunks)} chunks processed")
        return results

    def merge_results(
        self,
        chunk_results: List[Dict[str, Any]],
        original_filename: str
    ) -> Dict[str, Any]:
        """
        Merge results from multiple chunks into a single result.

        Args:
            chunk_results: List of results from process_chunks_async
            original_filename: Original document filename

        Returns:
            Merged result dictionary
        """
        logger.info(f"Merging results from {len(chunk_results)} chunks")

        # Initialize merged result
        merged = {
            'file_name': original_filename,
            'total_pages': 0,
            'full_text': '',
            'pages': [],
            'chapters': [],
            'headings': [],
            'images': [],
            'structure': {},
            'metadata_quality': {},
            'extraction_method': 'chunked',
            'chunk_count': len(chunk_results),
        }

        # Merge text and pages
        page_offset = 0

        for chunk_result in chunk_results:
            if 'error' in chunk_result:
                logger.warning(f"Skipping chunk {chunk_result['chunk_id']} due to error")
                continue

            # Merge text
            chunk_text = chunk_result.get('full_text', '')
            merged['full_text'] += chunk_text + '\n\n'

            # Merge pages with offset correction
            chunk_pages = chunk_result.get('pages', [])
            for page in chunk_pages:
                # Adjust page numbers
                page_copy = page.copy()
                page_copy['page_num'] += page_offset
                merged['pages'].append(page_copy)

            # Merge chapters with page offset
            chunk_chapters = chunk_result.get('chapters', [])
            for chapter in chunk_chapters:
                chapter_copy = chapter.copy()
                if 'page' in chapter_copy:
                    chapter_copy['page'] += page_offset
                merged['chapters'].append(chapter_copy)

            # Merge headings with page offset
            chunk_headings = chunk_result.get('headings', [])
            for heading in chunk_headings:
                heading_copy = heading.copy()
                # Adjust position if available
                merged['headings'].append(heading_copy)

            # Merge images with page offset
            chunk_images = chunk_result.get('images', [])
            for image in chunk_images:
                image_copy = image.copy()
                if 'page' in image_copy:
                    image_copy['page'] += page_offset
                merged['images'].append(image_copy)

            # Update page offset
            page_offset += len(chunk_pages)

        # Set total pages
        merged['total_pages'] = len(merged['pages'])

        # Merge structure
        merged['structure'] = {
            'has_abstract': any(c.get('title', '').lower() == 'abstract' for c in merged['chapters']),
            'has_introduction': any('introduction' in c.get('title', '').lower() for c in merged['chapters']),
            'has_conclusion': any('conclusion' in c.get('title', '').lower() for c in merged['chapters']),
            'has_references': any('reference' in c.get('title', '').lower() for c in merged['chapters']),
        }

        # Calculate quality score
        merged['metadata_quality'] = {
            'overall_quality': 70,  # Base quality for chunked processing
            'has_chapters': len(merged['chapters']) > 0,
            'has_headings': len(merged['headings']) > 0,
            'has_images': len(merged['images']) > 0,
            'chapter_count': len(merged['chapters']),
            'heading_count': len(merged['headings']),
            'image_count': len(merged['images']),
        }

        logger.info(f"✅ Merged result: {merged['total_pages']} pages, "
                   f"{len(merged['chapters'])} chapters, "
                   f"{len(merged['images'])} images")

        return merged

    def cleanup_temp_files(self, chunks: List[ChunkInfo]) -> None:
        """
        Clean up temporary chunk files.

        Args:
            chunks: List of ChunkInfo objects
        """
        logger.info(f"Cleaning up {len(chunks)} temporary chunk files")

        for chunk in chunks:
            if chunk.is_temp and Path(chunk.file_path).exists():
                try:
                    os.remove(chunk.file_path)
                    logger.debug(f"Deleted {Path(chunk.file_path).name}")
                except Exception as e:
                    logger.warning(f"Could not delete {chunk.file_path}: {e}")

        logger.info("✅ Cleanup complete")


# Convenience function
def chunk_and_process_pdf(
    pdf_path: str,
    process_func,
    max_pages: int = 10,
    max_concurrent: int = 5,
    cleanup: bool = True
) -> Dict[str, Any]:
    """
    Convenience function to chunk and process a large PDF.

    Args:
        pdf_path: Path to PDF file
        process_func: Function to process each chunk
        max_pages: Maximum pages per chunk
        max_concurrent: Maximum concurrent processing
        cleanup: Clean up temp files after processing

    Returns:
        Merged processing result

    Example:
        def my_process_func(file_path):
            # Extract and process chunk
            return {'full_text': '...', 'pages': [...]}

        result = chunk_and_process_pdf('large_doc.pdf', my_process_func)
    """
    chunker = LargeDocumentChunker(max_pages=max_pages)

    # Check if chunking needed
    if not chunker.should_chunk(pdf_path):
        logger.info("Document doesn't need chunking, processing directly")
        return process_func(pdf_path)

    # Split into chunks
    chunks = chunker.split_pdf(pdf_path)

    try:
        # Process chunks
        import asyncio
        loop = asyncio.get_event_loop()
        chunk_results = loop.run_until_complete(
            chunker.process_chunks_async(chunks, process_func, max_concurrent)
        )

        # Merge results
        merged_result = chunker.merge_results(chunk_results, Path(pdf_path).name)

        return merged_result

    finally:
        # Cleanup
        if cleanup:
            chunker.cleanup_temp_files(chunks)


# Testing
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]

        print(f"\n{'='*80}")
        print(f"Testing Document Chunker")
        print(f"{'='*80}\n")

        chunker = LargeDocumentChunker(max_pages=10)

        # Check if needs chunking
        needs_chunking = chunker.should_chunk(pdf_path)
        print(f"Document needs chunking: {needs_chunking}")

        if needs_chunking:
            # Split
            chunks = chunker.split_pdf(pdf_path)

            print(f"\n✅ Split into {len(chunks)} chunks:")
            for chunk in chunks:
                print(f"   {chunk}")

            # Cleanup
            print(f"\nCleaning up...")
            chunker.cleanup_temp_files(chunks)
            print(f"✅ Done")

    else:
        print("Usage: python document_chunker.py <pdf_file>")
        print("\nExample:")
        print("  python document_chunker.py large_document.pdf")
