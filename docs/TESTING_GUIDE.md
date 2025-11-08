# Chonkie OSS Testing Guide

## 🎯 Two Ways to Test

### Option 1: Web Interface (Easiest)
### Option 2: Command Line (Quick)

---

## 🌐 Option 1: Web Interface (Recommended)

### Start the server:

```bash
source venv/bin/activate
python test_chunking_api.py
```

### Then open in browser:
```
http://localhost:8000
```

### Features:
- ✅ Beautiful web interface
- ✅ Upload PDFs via drag-and-drop
- ✅ Real-time results
- ✅ See processing time
- ✅ View all chunks with formatting
- ✅ Adjust chunk size and overlap
- ✅ Choose chunker type

### Screenshot of what you'll see:
```
┌─────────────────────────────────────────────────┐
│   🚀 Chonkie OSS Test                          │
│   Upload a PDF and see the chunked output      │
│                                                 │
│   📄 PDF File: [Choose File]                   │
│   📏 Chunk Size: [512] tokens                  │
│   🔄 Overlap: [80] tokens                      │
│   ⚙️ Chunker Type: [Recursive ▼]               │
│                                                 │
│   [🚀 Chunk Document]                          │
│                                                 │
│   📊 Results                                   │
│   ┌──────────┬──────────┬──────────┬─────────┐│
│   │  250ms   │    5     │  2,456   │  491    ││
│   │Processing│  Chunks  │  Tokens  │Avg/Chunk││
│   └──────────┴──────────┴──────────┴─────────┘│
│                                                 │
│   📝 Chunks                                    │
│   ┌─────────────────────────────────────────┐ │
│   │ Chunk #1              491 tokens         │ │
│   │ [Chunk text preview...]                  │ │
│   └─────────────────────────────────────────┘ │
└─────────────────────────────────────────────────┘
```

---

## 💻 Option 2: Command Line

### Quick test:

```bash
source venv/bin/activate
python test_pdf_chunking.py /path/to/your/document.pdf
```

### With custom parameters:

```bash
python test_pdf_chunking.py document.pdf 512 80 recursive
#                           │          │   │   │
#                           │          │   │   └─ Chunker type
#                           │          │   └───── Overlap
#                           │          └───────── Chunk size
#                           └──────────────────── PDF path
```

### Interactive mode (step-by-step):

```bash
python test_pdf_chunking.py

# Then enter:
📁 PDF file path: /path/to/document.pdf
📏 Chunk size (default 512): 512
🔄 Overlap (default 80): 80
⚙️  Chunker type [recursive/token] (default recursive): recursive
```

### Output example:

```
==============================================================================
🚀 CHONKIE OSS PDF CHUNKING TEST
==============================================================================

📁 File: research_paper.pdf
📏 Chunk Size: 512 tokens
🔄 Overlap: 80 tokens
⚙️  Chunker: recursive

──────────────────────────────────────────────────────────────────────────────
Processing...
──────────────────────────────────────────────────────────────────────────────

==============================================================================
📊 CHUNKING RESULTS
==============================================================================

⏱️  Processing Time: 245.32ms (0.245s)
📄 Total Chunks: 5
🔢 Total Tokens: 2,456
📊 Average Tokens per Chunk: 491.2

==============================================================================
📝 CHUNK DETAILS
==============================================================================

────────────────────────────────────────────────────────────────────────────
Chunk #1
────────────────────────────────────────────────────────────────────────────
Token Count: 489
Character Range: 0 - 2640

Text Preview (first 300 chars):
┌┌┌┌┌┌┌┌┌┌┌┌┌┌┌┌┌┌┌┌┌┌┌┌┌┌┌┌┌┌┌┌┌┌┌┌┌┌┌┌┌┌┌┌┌┌┌┌┌┌┌┌┌┌┌┌┌┌┌┌┌┌┌┌┌┌┌┌┌┌┌┌┌┌┌┌┌┌┌┌
Abstract

Machine learning has revolutionized many fields in computer science.
This paper presents a comprehensive survey of recent advances in deep
learning architectures, focusing on transformer models and their
applications in natural language processing...
└└└└└└└└└└└└└└└└└└└└└└└└└└└└└└└└└└└└└└└└└└└└└└└└└└└└└└└└└└└└└└└└└└└└└└└└└└└└└└

▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼

────────────────────────────────────────────────────────────────────────────
Chunk #2
────────────────────────────────────────────────────────────────────────────
Token Count: 497
Character Range: 2520 - 5210

Text Preview (first 300 chars):
┌┌┌┌┌┌┌┌┌┌┌┌┌┌┌┌┌┌┌┌┌┌┌┌┌┌┌┌┌┌┌┌┌┌┌┌┌┌┌┌┌┌┌┌┌┌┌┌┌┌┌┌┌┌┌┌┌┌┌┌┌┌┌┌┌┌┌┌┌┌┌┌┌┌┌┌┌┌┌┌
1. Introduction

Recent breakthroughs in neural network architectures have enabled
unprecedented performance in various tasks. The transformer
architecture, introduced in 2017, has become the foundation for
state-of-the-art models...
└└└└└└└└└└└└└└└└└└└└└└└└└└└└└└└└└└└└└└└└└└└└└└└└└└└└└└└└└└└└└└└└└└└└└└└└└└└└└└

[... more chunks ...]

==============================================================================
✅ CHUNKING COMPLETE
==============================================================================

💾 Results saved to: research_paper_chunks.json
```

---

## 📋 What Gets Saved

Both methods save results to a JSON file:

```json
{
  "source_file": "research_paper.pdf",
  "processing_time_ms": 245.32,
  "chunk_size": 512,
  "overlap": 80,
  "chunker_type": "recursive",
  "total_chunks": 5,
  "chunks": [
    {
      "text": "Full chunk text here...",
      "start_index": 0,
      "end_index": 2640,
      "token_count": 489
    },
    ...
  ]
}
```

You can use this JSON for:
- ✅ Inspecting chunk quality
- ✅ Feeding into your RAG pipeline
- ✅ Comparing different chunking strategies

---

## 🎨 Chunker Types Explained

### 1. Recursive (Recommended)
```python
chunker_type="recursive"
```
- **Best for**: General text, documents, articles
- **How it works**: Splits on semantic boundaries (paragraphs → sentences → words)
- **Pros**: Preserves context and meaning
- **Use when**: You want high-quality chunks for RAG

### 2. Token
```python
chunker_type="token"
```
- **Best for**: Fixed-size requirements
- **How it works**: Splits at exact token counts with overlap
- **Pros**: Predictable chunk sizes
- **Use when**: You need consistent token counts

---

## 💡 Tips for Testing

### Test with Different Chunk Sizes:
```bash
# Small chunks (better for specific retrieval)
python test_pdf_chunking.py doc.pdf 256 50 recursive

# Medium chunks (balanced)
python test_pdf_chunking.py doc.pdf 512 80 recursive

# Large chunks (more context)
python test_pdf_chunking.py doc.pdf 1024 150 recursive
```

### Test with Different Overlaps:
```bash
# No overlap (faster, less redundancy)
python test_pdf_chunking.py doc.pdf 512 0 recursive

# Medium overlap (recommended)
python test_pdf_chunking.py doc.pdf 512 80 recursive

# High overlap (better context preservation)
python test_pdf_chunking.py doc.pdf 512 150 recursive
```

### Compare Chunker Types:
```bash
# Test recursive
python test_pdf_chunking.py doc.pdf 512 80 recursive

# Test token
python test_pdf_chunking.py doc.pdf 512 80 token

# Compare the outputs!
```

---

## 📊 What to Look For

### Good Chunks Should:
- ✅ Preserve semantic meaning
- ✅ Not split in the middle of sentences
- ✅ Have appropriate token counts
- ✅ Include enough context for retrieval

### Bad Chunks:
- ❌ Cut off mid-sentence
- ❌ Mix unrelated topics
- ❌ Too short (< 100 tokens)
- ❌ Too long (> 1500 tokens)

### Example Good Chunk:
```
Introduction

Machine learning has transformed the field of artificial
intelligence. Modern deep learning approaches leverage
neural networks with multiple layers to learn hierarchical
representations of data. This enables unprecedented
performance on tasks such as image recognition, natural
language processing, and game playing.
```

### Example Bad Chunk:
```
...al learning approaches leverage
neural networks with multiple layers to learn hierarchical
representations of data. This enables unprecedented
performance on tasks such as image recognition, nat
```

---

## 🔧 Troubleshooting

### "PDF support requires pdfplumber"
Install pdfplumber:
```bash
source venv/bin/activate
pip install pdfplumber
```

### "Processing too slow"
- Try smaller PDFs first
- Use `token` chunker (faster than `recursive`)
- Reduce chunk size

### "Chunks look weird"
- Try different chunker types
- Adjust overlap
- Check if PDF has extractable text (not scanned)

---

## 🚀 Quick Start (Choose One)

### Web Interface (Recommended):
```bash
source venv/bin/activate
python test_chunking_api.py
# Open: http://localhost:8000
```

### Command Line:
```bash
source venv/bin/activate
python test_pdf_chunking.py /path/to/your.pdf
```

---

## 📝 Next Steps After Testing

Once you're happy with the chunking quality:

1. **Update your main code** to use Chonkie OSS:
   ```python
   from chonkie_oss_client import chunk_document
   ```

2. **Remove Chonkie API key** from `.env`:
   ```bash
   # CHONKIE_API_KEY=xxx  # No longer needed!
   ```

3. **Save money** and enjoy faster, free chunking! 🎉
