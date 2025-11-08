# Migration Guide: Chonkie API → Chonkie OSS

## Why Switch?

| Feature | Chonkie API (Current) | Chonkie OSS (New) |
|---------|----------------------|-------------------|
| **Cost** | Paid API (requires key) | ✅ **FREE** (MIT License) |
| **Speed** | Network latency | ✅ **33x faster** (local) |
| **Dependencies** | API calls | ✅ Lightweight (505KB) |
| **Privacy** | Data sent to API | ✅ Fully local |
| **Availability** | Requires internet | ✅ Works offline |

## Installation

```bash
# Install Chonkie OSS with all features
pip install "chonkie[all]"

# Optional: For PDF support, also install
pip install pdfplumber
# or: pip install PyPDF2
# or: pip install pymupdf
```

## Code Comparison

### Before (API Version)
```python
from chonkie_client import chunk_document, embed_chunks

# Requires CHONKIE_API_KEY
chunks = chunk_document(
    text="Your text here",
    chunk_size=512,
    overlap_tokens=80,
    tokenizer="gpt2",
    recipe="markdown"  # API-specific parameter
)

# Uses Chonkie API for embedding (costs money)
embedded = embed_chunks(chunks)
```

### After (OSS Version)
```python
from chonkie_oss_client import chunk_document, embed_chunks

# No API key needed!
chunks = chunk_document(
    text="Your text here",
    chunk_size=512,
    overlap_tokens=80,
    tokenizer="gpt2",
    chunker_type="recursive"  # More flexible
)

# Uses OpenAI API directly (you already have this)
embedded = embed_chunks(chunks)
```

## Key Differences

### 1. **Chunker Types** (OSS is more flexible)
```python
# Recursive chunker (best for general text)
chunks = chunk_document(text=text, chunker_type="recursive")

# Token chunker (fixed-size chunks)
chunks = chunk_document(text=text, chunker_type="token")

# Available in full Chonkie OSS:
# - SemanticChunker (similarity-based)
# - SentenceChunker (sentence-level)
# - CodeChunker (for code files)
# - NeuralChunker (ML-based)
```

### 2. **Tokenizer Options**
```python
# GPT-2 (default)
tokenizer="gpt2"

# GPT-4 / GPT-3.5
tokenizer="cl100k_base"

# GPT-3
tokenizer="p50k_base"
```

### 3. **File Support**
```python
# Text files (no change)
chunks = chunk_document(file_path="document.txt")

# Markdown files (no change)
chunks = chunk_document(file_path="README.md")

# PDF files (now requires pdfplumber)
chunks = chunk_document(file_path="document.pdf")
```

## Performance Comparison

### Speed
- **Chonkie API**: Network round-trip (~200-500ms per request)
- **Chonkie OSS**: Local processing (~10-15ms for same text)
- **Result**: ✅ **10-50x faster** depending on network

### Cost
- **Chonkie API**: Pay per API call
- **Chonkie OSS**: ✅ **$0.00** (free forever)

### Memory
- **Chonkie API**: Minimal (sends data to API)
- **Chonkie OSS**: Slightly higher (tokenizer in memory) but still lightweight

## Migration Steps

1. **Install Chonkie OSS**
   ```bash
   pip install "chonkie[all]"
   ```

2. **Update imports**
   ```python
   # Change this:
   from chonkie_client import chunk_document, embed_chunks

   # To this:
   from chonkie_oss_client import chunk_document, embed_chunks
   ```

3. **Remove Chonkie API key** from `.env` (optional)
   ```bash
   # CHONKIE_API_KEY=xxx  # No longer needed!
   OPENAI_API_KEY=xxx     # Still needed for embeddings
   ```

4. **Test the migration**
   ```bash
   python chonkie_oss_client.py
   ```

## What About dots.ocr?

**dots.ocr is NOT a replacement for Chonkie** - it's an OCR/document parser.

### When to use dots.ocr:
- ✅ Extracting text from scanned PDFs
- ✅ Extracting text from images
- ✅ Parsing complex document layouts (tables, formulas)
- ✅ Multi-language OCR

### When NOT to use dots.ocr:
- ❌ Chunking already-extracted text (use Chonkie OSS)
- ❌ Simple text processing (too heavy - requires GPU)
- ❌ Fast processing (it's a 1.7B parameter model)

### Hybrid approach (if you process scanned PDFs):
```python
# 1. Extract text from scanned PDF using dots.ocr
extracted_text = dots_ocr.extract("scanned_document.pdf")

# 2. Chunk the extracted text using Chonkie OSS
from chonkie_oss_client import chunk_document
chunks = chunk_document(text=extracted_text)
```

## Advanced: Semantic Chunking

Chonkie OSS supports advanced chunking beyond what the API offers:

```python
from chonkie import SemanticChunker
from sentence_transformers import SentenceTransformer

# Chunk based on semantic similarity
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
chunker = SemanticChunker(
    embedding_model=embedding_model,
    chunk_size=512,
    similarity_threshold=0.5
)

chunks = chunker.chunk(text)
```

## Recommendation

✅ **Switch to Chonkie OSS** - it's:
- Free
- Faster
- More flexible
- Same company/ecosystem
- Easy migration

❌ **Skip dots.ocr** unless you specifically need OCR for scanned documents.
