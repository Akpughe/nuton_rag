"""
PDF Text Extraction Strategies with Quality Scoring
Implements multiple extraction methods and automatically selects the best one.

Strategies:
1. PyMuPDF (fitz) - Best for most PDFs, superior font handling
2. pdfplumber default - Good fallback, no layout inference
3. pdfplumber layout - Last resort only, can cause mid-word spaces

Quality Scoring:
- Detects broken words (mid-word spaces)
- Analyzes word length distribution
- Identifies repeated fragments
- Scores 0-100 (higher is better)
"""

import re
import logging
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    logging.warning("PyMuPDF not available, falling back to pdfplumber only")

import pdfplumber

logging.basicConfig(level=logging.INFO)


class PDFExtractionStrategy:
    """Multi-strategy PDF text extraction with quality scoring."""

    def __init__(self):
        self.strategies = []

        # Register available strategies in priority order
        if PYMUPDF_AVAILABLE:
            self.strategies.append(('pymupdf', self.extract_with_pymupdf))

        self.strategies.append(('pdfplumber_default', self.extract_with_pdfplumber_default))
        self.strategies.append(('pdfplumber_layout', self.extract_with_pdfplumber_layout))

    def extract_best(self, pdf_path: str) -> Dict[str, Any]:
        """
        Try all extraction strategies and return the best one.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Dict with:
                - text: Extracted text
                - strategy: Strategy name used
                - quality_score: Quality score (0-100)
                - all_scores: Scores from all strategies
        """
        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        results = []

        # Try each strategy
        for strategy_name, extract_func in self.strategies:
            try:
                logging.info(f"Trying extraction strategy: {strategy_name}")
                text = extract_func(str(pdf_path))

                if not text or len(text.strip()) < 100:
                    logging.warning(f"{strategy_name}: Extracted text too short, skipping")
                    continue

                # Score quality
                quality_score = self.score_extraction_quality(text)

                results.append({
                    'strategy': strategy_name,
                    'text': text,
                    'quality_score': quality_score
                })

                logging.info(f"{strategy_name}: quality score = {quality_score}/100")

            except Exception as e:
                logging.warning(f"{strategy_name} failed: {e}")
                continue

        if not results:
            raise Exception("All extraction strategies failed")

        # Select best result
        best_result = max(results, key=lambda x: x['quality_score'])

        logging.info(f"Selected strategy: {best_result['strategy']} (score: {best_result['quality_score']}/100)")

        return {
            'text': best_result['text'],
            'strategy': best_result['strategy'],
            'quality_score': best_result['quality_score'],
            'all_scores': {r['strategy']: r['quality_score'] for r in results}
        }

    def extract_with_pymupdf(self, pdf_path: str) -> str:
        """
        Extract text using PyMuPDF (fitz).

        Best for:
        - PDFs with embedded fonts
        - Complex layouts
        - Most general-purpose extraction

        Args:
            pdf_path: Path to PDF file

        Returns:
            Extracted text
        """
        if not PYMUPDF_AVAILABLE:
            raise ImportError("PyMuPDF not available")

        doc = fitz.open(pdf_path)
        pages_text = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            # Extract text with default settings
            # PyMuPDF handles fonts much better than pdfplumber
            text = page.get_text()
            pages_text.append(text)

        doc.close()

        return '\n'.join(pages_text)

    def extract_with_pdfplumber_default(self, pdf_path: str) -> str:
        """
        Extract text using pdfplumber without layout mode.

        Best for:
        - Simple PDFs
        - When PyMuPDF fails
        - Fallback extraction

        Args:
            pdf_path: Path to PDF file

        Returns:
            Extracted text
        """
        pages_text = []

        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                # Extract WITHOUT layout mode to avoid mid-word spaces
                text = page.extract_text(layout=False) or ""
                pages_text.append(text)

        return '\n'.join(pages_text)

    def extract_with_pdfplumber_layout(self, pdf_path: str) -> str:
        """
        Extract text using pdfplumber WITH layout mode.

        WARNING: Can cause mid-word spaces for PDFs with custom fonts!
        Only use as last resort.

        Best for:
        - PDFs with complex spatial layouts
        - When other methods fail completely

        Args:
            pdf_path: Path to PDF file

        Returns:
            Extracted text
        """
        pages_text = []

        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                # Extract WITH layout mode
                # This can cause mid-word spacing issues!
                text = page.extract_text(layout=True) or ""
                pages_text.append(text)

        return '\n'.join(pages_text)

    def score_extraction_quality(self, text: str) -> int:
        """
        Score extraction quality (0-100).

        Detects common extraction problems:
        - Mid-word spaces ("ver er", "whic ic")
        - Excessive short words
        - Repeated fragments ("tion tion")
        - Abnormal word length distribution

        Args:
            text: Extracted text to score

        Returns:
            Quality score (0-100, higher is better)
        """
        if not text or len(text.strip()) < 100:
            return 0

        score = 100
        words = text.split()

        if not words:
            return 0

        # 1. Detect mid-word space pattern
        # Matches: "ver er", "whic ic", "arter erosus"
        # Pattern: 3-5 chars + space + 2-3 chars
        broken_word_pattern = r'\b\w{3,5}\s\w{2,3}\b'
        broken_words = len(re.findall(broken_word_pattern, text))

        if broken_words > 0:
            # Penalize heavily - this is a clear sign of extraction issues
            score -= min(broken_words * 5, 40)
            logging.debug(f"Found {broken_words} potential broken words")

        # 2. Check short word ratio
        short_words = sum(1 for w in words if len(w) <= 3)
        short_ratio = short_words / len(words)

        # Normal English: ~15-20% short words (a, the, is, of, etc.)
        # Broken text: >30% short words (spaces inserted everywhere)
        if short_ratio > 0.30:
            score -= 30
            logging.debug(f"High short word ratio: {short_ratio:.2%}")
        elif short_ratio > 0.25:
            score -= 15

        # 3. Check average word length
        avg_word_length = sum(len(w) for w in words) / len(words)

        # Normal English: 4.5-5.5 characters average
        # Broken text: <4 characters (spaces breaking words)
        if avg_word_length < 4.0:
            score -= 40
            logging.debug(f"Low average word length: {avg_word_length:.2f}")
        elif avg_word_length < 4.5:
            score -= 20

        # 4. Detect repeated fragments (strong indicator of broken text)
        # Matches: "tion tion", "er er", "ic ic"
        repeated_fragment_pattern = r'\b(\w{2,3})\s\1\b'
        repeated_fragments = len(re.findall(repeated_fragment_pattern, text))

        if repeated_fragments > 0:
            # This is almost always a sign of broken extraction
            score -= min(repeated_fragments * 15, 30)
            logging.debug(f"Found {repeated_fragments} repeated fragments")

        # 5. Check for excessive single/double character "words"
        very_short_words = sum(1 for w in words if len(w) <= 2 and w.isalpha())
        very_short_ratio = very_short_words / len(words)

        if very_short_ratio > 0.20:
            score -= 20
            logging.debug(f"High very short word ratio: {very_short_ratio:.2%}")

        return max(score, 0)


# Convenience function
def extract_pdf_text_best_quality(pdf_path: str) -> Dict[str, Any]:
    """
    Extract text from PDF using the best available strategy.

    Args:
        pdf_path: Path to PDF file

    Returns:
        Dict with extracted text and quality metrics

    Example:
        >>> result = extract_pdf_text_best_quality("document.pdf")
        >>> print(result['text'])
        >>> print(f"Quality: {result['quality_score']}/100")
        >>> print(f"Strategy: {result['strategy']}")
    """
    extractor = PDFExtractionStrategy()
    return extractor.extract_best(pdf_path)


# For testing
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]

        print(f"\nTesting PDF extraction strategies on: {pdf_path}\n")
        print("=" * 80)

        result = extract_pdf_text_best_quality(pdf_path)

        print(f"\nBest Strategy: {result['strategy']}")
        print(f"Quality Score: {result['quality_score']}/100")
        print(f"\nAll Strategy Scores:")
        for strategy, score in result['all_scores'].items():
            print(f"  {strategy}: {score}/100")

        print(f"\nFirst 500 characters of extracted text:")
        print("-" * 80)
        print(result['text'][:500])
        print("-" * 80)
    else:
        print("Usage: python pdf_extraction_strategies.py <pdf_file>")
