import logging
from typing import Optional
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_text_from_pdf(file_path: str) -> str:
    """
    Extract full text from a PDF file.

    Args:
        file_path: Path to the PDF file

    Returns:
        Complete text content of the PDF

    Raises:
        Exception: If extraction fails
    """
    logger.info(f"Extracting text from PDF: {file_path}")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF file not found: {file_path}")

    try:
        # Try pypdf first (more modern)
        try:
            from pypdf import PdfReader
            reader = PdfReader(file_path)
            text_parts = []

            for page_num, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)

            full_text = "\n".join(text_parts)
            logger.info(f"Extracted {len(full_text)} characters from {len(reader.pages)} pages using pypdf")
            return full_text

        except ImportError:
            # Fallback to PyPDF2
            logger.info("pypdf not available, trying PyPDF2")
            from PyPDF2 import PdfReader
            reader = PdfReader(file_path)
            text_parts = []

            for page_num, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)

            full_text = "\n".join(text_parts)
            logger.info(f"Extracted {len(full_text)} characters from {len(reader.pages)} pages using PyPDF2")
            return full_text

    except ImportError as e:
        error_msg = "PDF extraction library not available. Install pypdf or PyPDF2: pip install pypdf"
        logger.error(error_msg)
        raise ImportError(error_msg)

    except Exception as e:
        logger.error(f"Failed to extract text from PDF: {e}")
        raise


def extract_text_from_file(file_path: str) -> str:
    """
    Extract text from various file types.

    Args:
        file_path: Path to the file

    Returns:
        Complete text content

    Raises:
        Exception: If extraction fails or file type not supported
    """
    file_ext = os.path.splitext(file_path)[1].lower()

    if file_ext == '.pdf':
        return extract_text_from_pdf(file_path)
    elif file_ext == '.txt':
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    elif file_ext in ['.md', '.markdown']:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    else:
        # For other file types, try to read as text
        logger.warning(f"Unsupported file type {file_ext}, attempting to read as text")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            raise Exception(f"Failed to extract text from {file_ext} file: {e}")


def validate_extracted_text(text: str, min_length: int = 100) -> bool:
    """
    Validate that extracted text is meaningful.

    Args:
        text: Extracted text to validate
        min_length: Minimum expected length

    Returns:
        True if text appears valid, False otherwise
    """
    if not text or len(text.strip()) < min_length:
        logger.warning(f"Extracted text too short: {len(text) if text else 0} chars")
        return False

    # Check if text is mostly garbage characters
    printable_ratio = sum(c.isprintable() or c.isspace() for c in text) / len(text)
    if printable_ratio < 0.8:
        logger.warning(f"Extracted text has low printable ratio: {printable_ratio:.2f}")
        return False

    return True
