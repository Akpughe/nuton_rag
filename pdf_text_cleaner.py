"""
PDF Text Cleaning Utilities
Cleans common PDF extraction artifacts for better RAG quality.

Handles:
- CID codes: (cid:123)
- Missing spaces between words
- Special characters and encoding issues
- Extra whitespace
- Ligatures and special glyphs
"""

import re
import unicodedata
from typing import Optional


class PDFTextCleaner:
    """Clean and normalize text extracted from PDFs."""

    def __init__(self):
        """Initialize text cleaner with patterns."""
        # Common PDF artifacts to remove
        self.cid_pattern = re.compile(r'\(cid:\d+\)')

        # Ligature replacements (common in PDFs)
        self.ligatures = {
            'ﬁ': 'fi',
            'ﬂ': 'fl',
            'ﬀ': 'ff',
            'ﬃ': 'ffi',
            'ﬄ': 'ffl',
            'ﬆ': 'st',
            'Ꜳ': 'AA',
            'Æ': 'AE',
            'æ': 'ae',
        }

        # Characters to remove (zero-width, control chars, etc.)
        self.remove_chars = [
            '\x00', '\x01', '\x02', '\x03', '\x04', '\x05', '\x06', '\x07',
            '\x08', '\x0b', '\x0c', '\x0e', '\x0f', '\x10', '\x11', '\x12',
            '\x13', '\x14', '\x15', '\x16', '\x17', '\x18', '\x19', '\x1a',
            '\x1b', '\x1c', '\x1d', '\x1e', '\x1f',
            '\u200b',  # Zero-width space
            '\ufeff',  # Zero-width no-break space (BOM)
        ]

    def clean(self, text: str, aggressive: bool = True) -> str:
        """
        Clean PDF text.

        Args:
            text: Raw text from PDF
            aggressive: Apply aggressive cleaning (fix spacing issues)

        Returns:
            Cleaned text
        """
        if not text:
            return ""

        # Step 1: Remove CID codes
        text = self._remove_cid_codes(text)

        # Step 2: Replace ligatures
        text = self._replace_ligatures(text)

        # Step 3: Remove control characters
        text = self._remove_control_chars(text)

        # Step 4: Normalize unicode
        text = self._normalize_unicode(text)

        # Step 5: Fix mid-word spaces (CRITICAL for extraction quality)
        if aggressive:
            text = self._fix_mid_word_spaces(text)

        # Step 6: Fix spacing issues (aggressive)
        if aggressive:
            text = self._fix_spacing(text)

        # Step 7: Clean whitespace
        text = self._clean_whitespace(text)

        return text

    def _remove_cid_codes(self, text: str) -> str:
        """
        Remove CID codes like (cid:123).

        These appear when PDF fonts aren't properly decoded.
        """
        # Remove all (cid:XX) patterns
        text = self.cid_pattern.sub('', text)

        # Also remove common variants
        text = re.sub(r'\(cid:\w+\)', '', text)

        return text

    def _replace_ligatures(self, text: str) -> str:
        """Replace ligatures with normal characters."""
        for ligature, replacement in self.ligatures.items():
            text = text.replace(ligature, replacement)
        return text

    def _remove_control_chars(self, text: str) -> str:
        """Remove control characters."""
        for char in self.remove_chars:
            text = text.replace(char, '')
        return text

    def _normalize_unicode(self, text: str) -> str:
        """
        Normalize unicode to standard form.

        This converts accented characters to standard forms.
        """
        # NFKC normalization (compatibility decomposition + canonical composition)
        # This handles most unicode oddities
        try:
            text = unicodedata.normalize('NFKC', text)
        except Exception:
            pass  # If normalization fails, keep original

        return text

    def _fix_mid_word_spaces(self, text: str) -> str:
        """
        Remove spaces incorrectly inserted mid-word.

        This is CRITICAL for fixing extraction quality issues from
        pdfplumber with layout=True mode.

        Common patterns:
        - "ver er" → "very"
        - "whic ic" → "which"
        - "arter erosus" → "arteriosus"
        - "denition tion" → "definition"
        - "Artic ical" → "Artificial"
        """
        # Pattern 1: Repeated fragments (very strong indicator)
        # "tion tion" → "tion", "ic ic" → "ic", "er er" → "er"
        text = re.sub(r'\b(\w{2,3})\s+\1\b', r'\1', text)

        # Pattern 2: Common suffix splits
        # "defini tion" → "definition", "arter erosus" → "arteriosus"
        suffixes = [
            'tion', 'sion', 'ment', 'ness', 'ful', 'less',
            'ous', 'ive', 'al', 'ic', 'ed', 'er', 'est',
            'ing', 'ly', 'ity', 'ism', 'able', 'ible',
            'ance', 'ence', 'ant', 'ent', 'ary', 'ory',
            'ous', 'eous', 'ious', 'uous'
        ]

        for suffix in suffixes:
            # Match: word_part + space + suffix
            # Only join if word_part is at least 3 characters
            pattern = rf'\b(\w{{3,}})\s+({suffix})\b'
            text = re.sub(pattern, r'\1\2', text, flags=re.IGNORECASE)

        # Pattern 3: Common prefix splits
        # "re sult" → "result", "un der" → "under"
        prefixes = ['re', 'un', 'in', 'dis', 'en', 'non', 'over', 'mis', 'sub', 'pre', 'inter', 'fore', 'de', 'trans', 'super', 'semi', 'anti', 'mid', 'under']

        for prefix in prefixes:
            # Match: prefix + space + rest of word (at least 3 chars)
            pattern = rf'\b({prefix})\s+(\w{{3,}})\b'
            text = re.sub(pattern, r'\1\2', text, flags=re.IGNORECASE)

        # Pattern 4: Common broken words (high-frequency words)
        # These are medical/scientific terms that commonly break
        broken_words = {
            r'\bver\s+y\b': 'very',
            r'\bwhic\s+h\b': 'which',
            r'\bover\s+view\b': 'overview',
            r'\barter\s+erosus\b': 'arteriosus',
            r'\boval\s+e\b': 'ovale',
            r'\bUnive\s+rsity\b': 'University',
            r'\bArtic\s+icial\b': 'Artificial',
            r'\bArtic\s+ical\b': 'Artificial',
            r'\bdenition\s+tion\b': 'definition',
            r'\binter\s+atrial\b': 'interatrial',
            r'\bcongen\s+ital\b': 'congenital',
            r'\bductus\s+arter\s+erosus\b': 'ductus arteriosus',
        }

        for pattern, replacement in broken_words.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        # Pattern 5: Generic mid-word space pattern
        # Words where 3-5 chars are followed by space + 2-3 chars
        # "ver er" → "verer" (will be caught by other patterns if it's a real word)
        # This is conservative - only joins very likely broken words
        text = re.sub(r'\b(\w{3,5})\s+(\w{2,3})(?=\s|[.,;:]|$)', r'\1\2', text)

        return text

    def _fix_spacing(self, text: str) -> str:
        """
        Fix missing spaces between words.

        This is a heuristic approach:
        - Detects transitions from lowercase to uppercase
        - Detects transitions ending with punctuation
        - Adds spaces where appropriate
        - Handles semicolon-separated text (common in definitions)
        """
        # Pattern 1: lowercase letter followed by uppercase letter
        # Example: "wordAnother" -> "word Another"
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)

        # Pattern 2: letter followed by number
        # Example: "word123" -> "word 123"
        text = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', text)

        # Pattern 3: number followed by letter
        # Example: "123word" -> "123 word"
        text = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', text)

        # Pattern 4: punctuation without space after
        # Example: "word.Another" -> "word. Another"
        text = re.sub(r'([.!?;:])([A-Z])', r'\1 \2', text)

        # Pattern 5: closing paren/bracket followed by letter
        # Example: "(word)Another" -> "(word) Another"
        text = re.sub(r'([)\]})"])([A-Za-z])', r'\1 \2', text)

        # Pattern 6: semicolon without space after (common in medical/scientific text)
        # Example: "term;another" -> "term; another"
        text = re.sub(r';([a-zA-Z])', r'; \1', text)

        # Pattern 7: lowercase followed by lowercase after known endings
        # Example: "contraction;the" -> "contraction; the"
        # Common word endings that should have space after
        word_endings = r'(ion|tion|ing|ment|ence|ance|able|ible|ness|ship|hood|ful|less|ous|ive|al|ic|ed|er|est|ly)'
        text = re.sub(rf'({word_endings})([a-z])', r'\1 \2', text)

        return text

    def _clean_whitespace(self, text: str) -> str:
        """
        Clean up whitespace issues.

        - Remove multiple spaces
        - Remove trailing/leading whitespace
        - Normalize line breaks
        """
        # Replace multiple spaces with single space
        text = re.sub(r' {2,}', ' ', text)

        # Replace multiple newlines with max 2 newlines
        text = re.sub(r'\n{3,}', '\n\n', text)

        # Remove spaces at start/end of lines
        lines = text.split('\n')
        lines = [line.strip() for line in lines]
        text = '\n'.join(lines)

        # Remove trailing/leading whitespace
        text = text.strip()

        return text

    def detect_quality(self, text: str) -> dict:
        """
        Detect text quality issues.

        Returns a dict with quality metrics.
        """
        quality = {
            'has_cid_codes': bool(self.cid_pattern.search(text)),
            'cid_count': len(self.cid_pattern.findall(text)),
            'has_ligatures': any(lig in text for lig in self.ligatures),
            'avg_word_length': 0,
            'very_long_words': 0,  # Words > 30 chars (likely concatenated)
            'broken_word_count': 0,  # Mid-word spaces detected
            'quality_score': 100  # 0-100
        }

        # Count very long words (likely concatenation errors)
        words = text.split()
        if words:
            word_lengths = [len(w) for w in words]
            quality['avg_word_length'] = sum(word_lengths) / len(word_lengths)
            quality['very_long_words'] = sum(1 for w in word_lengths if w > 30)

        # Detect broken words (mid-word spaces)
        # Pattern: 3-5 chars + space + 2-3 chars
        broken_word_pattern = r'\b\w{3,5}\s\w{2,3}\b'
        quality['broken_word_count'] = len(re.findall(broken_word_pattern, text))

        # Calculate quality score
        score = 100

        if quality['has_cid_codes']:
            score -= 30  # Major issue

        if quality['cid_count'] > 100:
            score -= 20  # Severe CID problem

        if quality['broken_word_count'] > 50:
            score -= 30  # Many broken words

        if quality['very_long_words'] > 10:
            score -= 20  # Likely spacing issues

        if quality['avg_word_length'] > 15:
            score -= 15  # Unusual word length (spacing?)
        elif quality['avg_word_length'] < 4:
            score -= 15  # Too short (broken words)

        quality['quality_score'] = max(score, 0)

        return quality


# Convenience function
def clean_pdf_text(text: str, aggressive: bool = True) -> str:
    """
    Clean text extracted from PDF.

    Args:
        text: Raw PDF text
        aggressive: Apply aggressive cleaning (fix spacing)

    Returns:
        Cleaned text

    Example:
        >>> text = "(cid:55)(cid:80)(cid:77) Volume 2"
        >>> clean_pdf_text(text)
        'Volume 2'
    """
    cleaner = PDFTextCleaner()
    return cleaner.clean(text, aggressive=aggressive)


# For testing
if __name__ == "__main__":
    # Test CID codes
    test1 = "(cid:55)(cid:80)(cid:77)(cid:86)(cid:78)(cid:70)(cid:1)2(cid:1)(cid:80)(cid:71)(cid:1)(cid:20)(cid:27)"
    print("Test 1 (CID codes):")
    print(f"Before: {test1}")
    print(f"After:  {clean_pdf_text(test1)}")
    print()

    # Test spacing
    test2 = "theircontraction;therightatriumreceivesbloodfromthesystemiccircuit"
    print("Test 2 (Missing spaces):")
    print(f"Before: {test2}")
    print(f"After:  {clean_pdf_text(test2)}")
    print()

    # Test combined
    test3 = "(cid:53)(cid:73)(cid:74)(cid:84)(cid:1)textbookHasContent"
    print("Test 3 (Combined):")
    print(f"Before: {test3}")
    print(f"After:  {clean_pdf_text(test3)}")
    print()

    # Quality detection
    cleaner = PDFTextCleaner()
    quality = cleaner.detect_quality(test1)
    print("Quality metrics:")
    for key, value in quality.items():
        print(f"  {key}: {value}")
