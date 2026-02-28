# Building a Production RAG Pipeline with FastAPI: Architecture and Design Decisions

When building a Retrieval-Augmented Generation (RAG) system that needs to handle real users, you quickly realize that the academic examples don't cut it. You need resilient extraction, smart chunking, multi-modal support, and graceful fallbacks. In this article, I'll walk you through the architecture of a production RAG pipeline built with FastAPI, sharing the design decisions that emerged from real-world constraints.

## The Big Picture

At its core, our RAG pipeline does three things:

1. **Ingest documents** - Extract text, chunk it intelligently, embed it, and store vectors
2. **Process queries** - Embed the query, search vectors, rerank results, and generate answers
3. **Handle edge cases** - Fallback chains, error handling, and graceful degradation

Here's the high-level architecture:

```
                    ┌─────────────────────────────────────────────────────────┐
                    │                    FastAPI Server                        │
                    │                     (pipeline.py)                        │
                    └─────────────────────────────────────────────────────────┘
                                              │
              ┌───────────────────────────────┼───────────────────────────────┐
              │                               │                               │
              ▼                               ▼                               ▼
    ┌─────────────────┐           ┌─────────────────┐           ┌─────────────────┐
    │  Document       │           │  Query          │           │  Content        │
    │  Ingestion      │           │  Answering      │           │  Generation     │
    └─────────────────┘           └─────────────────┘           └─────────────────┘
              │                               │                               │
    ┌─────────┴─────────┐         ┌──────────┴──────────┐         ┌──────────┴──────────┐
    │                   │         │                     │         │                     │
    ▼                   ▼         ▼                     ▼         ▼                     ▼
┌────────┐        ┌────────┐  ┌────────┐         ┌────────┐  ┌────────┐          ┌────────┐
│Mistral │        │Chonkie │  │Hybrid  │         │Rerank  │  │Flash-  │          │Quiz    │
│OCR     │        │Chunker │  │Search  │         │(BGE)   │  │cards   │          │Gen     │
└────────┘        └────────┘  └────────┘         └────────┘  └────────┘          └────────┘
    │                   │         │                     │
    └─────────┬─────────┘         └──────────┬──────────┘
              │                              │
              ▼                              ▼
    ┌─────────────────┐           ┌─────────────────┐
    │  Jina CLIP-v2   │           │  Groq / OpenAI  │
    │  (Embeddings)   │           │  (Generation)   │
    └─────────────────┘           └─────────────────┘
              │                              │
              └──────────────┬───────────────┘
                             ▼
                  ┌─────────────────┐
                  │    Pinecone     │
                  │  (Vector Store) │
                  └─────────────────┘
```

## Why FastAPI?

We chose FastAPI for several reasons:

```python
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

**1. Native async support** - Document processing and LLM calls are I/O bound. FastAPI's async support lets us handle concurrent requests efficiently.

**2. Automatic OpenAPI docs** - Every endpoint gets documented automatically, making frontend integration straightforward.

**3. Pydantic models** - Request validation is built-in, reducing boilerplate and catching errors early.

```python
class QuizRequest(BaseModel):
    document_id: str
    space_id: Optional[str] = None
    user_id: str
    question_type: str = "both"
    num_questions: int = 30
    target_coverage: float = 0.80
    enable_gap_filling: bool = True
```

**4. Form data handling** - RAG endpoints often need both files and metadata. FastAPI handles this elegantly:

```python
@app.post("/process_document")
async def process_document_endpoint(
    files: List[UploadFile] = File(...),
    file_urls: str = Form(...),  # JSON string of URLs
    space_id: str = Form(...),
    use_openai: bool = Form(True)
) -> JSONResponse:
```

## Document Ingestion: The Three-Stage Pipeline

The document processing endpoint orchestrates three critical stages: extraction, embedding, and upserting. Let's walk through each stage with the actual implementation.

### Stage 1: Mistral OCR Extraction

We use Mistral's Pixtral model for document extraction. It handles PDFs, images, and Office documents with impressive accuracy:

```python
@app.post("/process_document")
async def process_document_endpoint(
    files: List[UploadFile] = File(...),
    space_id: str = Form(...),
    use_openai: bool = Form(True)
) -> JSONResponse:
    document_ids = []

    for file in files:
        # Save uploaded file
        file_path = f"/tmp/{file.filename}"
        with open(file_path, "wb") as f:
            f.write(await file.read())

        # Extract with Mistral OCR
        mistral_config = MistralOCRConfig(
            enhance_metadata_with_llm=True,
            include_images=True,
            include_image_base64=True,
        )

        extractor = MistralOCRExtractor(config=mistral_config)
        extraction_result = extractor.process_document(file_path)

        # Process extraction result
        if extraction_result.get("success"):
            extracted_text = extraction_result["text"]
            images_with_data = extraction_result.get("images", [])
            metadata = extraction_result.get("metadata", {})
```

**Why Mistral OCR?**

- Handles complex layouts (tables, multi-column, diagrams)
- Extracts images with positional context
- Works across formats (PDF, PPTX, DOCX, images)
- Returns structured metadata automatically

### Stage 2: Multimodal Embedding

Once we have text and images, we embed them using Jina CLIP-v2. This creates a unified vector space for both modalities:

```python
# Chunk the extracted text
chunks = chunk_document(
    text=extracted_text,
    chunk_size=512,
    recipe="markdown"
)

# Embed text chunks with multimodal embeddings
embeddings = embed_chunks_multimodal(chunks, batch_size=64)

# Embed extracted images separately
if images_with_data:
    embedder = MultimodalEmbedder(model="jina-clip-v2")
    image_embeddings = embedder.embed_images(
        [img["base64_data"] for img in images_with_data],
        normalize=True
    )
```

The key insight: **text and images share the same 1024-dimensional embedding space**. This means users can search with text and retrieve relevant images, or vice versa.

### Stage 3: Upserting to Pinecone

Finally, we store the vectors in Pinecone with rich metadata for filtering:

```python
# Upsert text chunks
upsert_vectors(
    doc_id=document_id,
    space_id=space_id,
    embeddings=embeddings,
    chunks=chunks,
    metadata={
        "title": metadata.get("title", file.filename),
        "extraction_method": "mistral_ocr",
        "created_at": datetime.utcnow().isoformat()
    }
)

# Upsert image vectors with positional metadata
if images_with_data and image_embeddings:
    upsert_image_vectors(
        doc_id=document_id,
        space_id=space_id,
        images=images_with_data,
        embeddings=image_embeddings
    )
```

Each vector gets metadata for filtering by document, space, or content type. This enables precise retrieval later.

## Query Flow: Hybrid Search + Reranking

The query answering flow combines dense vector search with sparse keyword search, then reranks results for optimal relevance. Here's how each stage works:

### Step 1: Query Embedding

First, we embed the user's query using the same multimodal embedder:

```python
@app.post("/answer_query")
async def answer_query_endpoint(request: QueryRequest):
    # Embed query (cached for repeated queries)
    query_embedded = get_query_embedding(request.query)
    query_emb = query_embedded["embedding"]  # Dense vector (1024 dims)
    query_sparse = query_embedded.get("sparse")  # Sparse BM25 vector
```

### Step 2: Hybrid Search

We search using both dense and sparse vectors in parallel, then merge results:

```python
def hybrid_search_parallel(
    query_emb: List[float],
    query_sparse: Optional[Dict],
    top_k: int,
    doc_id: Optional[str] = None,
    space_id: Optional[str] = None
) -> List[Dict]:
    """
    Parallel hybrid search combining dense + sparse retrieval.
    """
    with ThreadPoolExecutor(max_workers=2) as executor:
        # Dense search (semantic similarity)
        dense_future = executor.submit(
            pinecone_index.query,
            vector=query_emb,
            top_k=top_k,
            filter={"space_id": space_id} if space_id else None,
            include_metadata=True
        )

        # Sparse search (keyword matching)
        sparse_future = executor.submit(
            pinecone_index.query,
            sparse_vector=query_sparse,
            top_k=top_k,
            filter={"space_id": space_id} if space_id else None,
            include_metadata=True
        )

        dense_results = dense_future.result()
        sparse_results = sparse_future.result()

    # Merge with reciprocal rank fusion
    merged = reciprocal_rank_fusion(dense_results, sparse_results)
    return merged
```

**Why hybrid search?**

- Dense vectors capture semantic meaning ("car" matches "automobile")
- Sparse vectors capture exact keywords (important for technical terms)
- Fusion combines the best of both approaches

### Step 3: Reranking with Cross-Encoders

Hybrid search gives us candidates; reranking refines them using a cross-encoder model:

```python
def rerank_results(
    query: str,
    hits: List[Dict],
    top_n: int = 10
) -> List[Dict]:
    """
    Rerank search results using BGE reranker for better relevance.
    """
    from sentence_transformers import CrossEncoder

    reranker = CrossEncoder('BAAI/bge-reranker-base')

    # Create query-document pairs
    pairs = [(query, hit["metadata"]["text"]) for hit in hits]

    # Score all pairs
    scores = reranker.predict(pairs)

    # Sort by reranker scores
    for hit, score in zip(hits, scores):
        hit["rerank_score"] = float(score)

    reranked = sorted(hits, key=lambda x: x["rerank_score"], reverse=True)
    return reranked[:top_n]
```

### Step 4: Answer Generation

Finally, we use the top reranked chunks as context for the LLM:

```python
# Take top 5 chunks after reranking
limited_context = reranked[:5]

# Build context string
context = "\n\n".join([
    f"[Source {i+1}]\n{chunk['metadata']['text']}"
    for i, chunk in enumerate(limited_context)
])

# Generate answer with Groq
answer, citations = groq_client.generate_answer(
    query=request.query,
    context=context,
    model="meta-llama/llama-4-scout-17b-16e-instruct"
)

return {
    "answer": answer,
    "citations": citations,
    "sources": limited_context
}
```

### Query Embedding Caching

A simple but effective optimization - caching query embeddings:

```python
from functools import lru_cache

@lru_cache(maxsize=100)
def get_query_embedding(query: str, use_openai: bool = True):
    """Cache query embeddings to avoid recalculating for repeated queries."""
    start_time = time.time()
    result = embed_query_multimodal(query)
    logging.info(f"Query embedding took {time.time() - start_time:.2f}s")
    return result
```

This prevents redundant API calls when users ask variations of the same question.

### Conversation History Management

For chat-like interfaces, we validate and truncate conversation history to balance context with cost:

```python
def _validate_and_truncate_history(
    history: Optional[List[Dict[str, str]]],
    max_messages: int = 10,
    max_tokens: int = 2000
) -> Optional[List[Dict[str, str]]]:
    """Validate and truncate conversation history for speed."""

    if not history:
        return None

    # Validate structure
    validated = []
    for msg in history:
        if isinstance(msg, dict) and "role" in msg and "content" in msg:
            if msg["role"] in ["user", "assistant"]:
                validated.append(msg)

    # Truncate to last N messages
    validated = validated[-max_messages:]

    # Fast token estimation (4 chars ≈ 1 token)
    total_chars = sum(len(msg["content"]) for msg in validated)
    estimated_tokens = total_chars // 4

    # Remove oldest if over budget
    while estimated_tokens > max_tokens and len(validated) > 2:
        validated.pop(0)
        total_chars = sum(len(msg["content"]) for msg in validated)
        estimated_tokens = total_chars // 4

    return validated
```

## LLM Fallback Chains

Never trust a single LLM provider. We implement a fallback chain:

```python
# Primary: Groq (fast, free tier)
try:
    answer, citations = generate_answer(
        query, limited_context, system_prompt,
        model="meta-llama/llama-4-scout-17b-16e-instruct"
    )
except Exception as e:
    logging.warning(f"Groq failed, falling back to OpenAI: {e}")
    # Fallback: OpenAI (reliable, paid)
    answer, citations = openai_client.generate_answer(
        query, limited_context, system_prompt,
        model="gpt-4o-mini"
    )
```

When websearch is enabled, we automatically switch to GPT-4o for better synthesis:

```python
if enable_websearch:
    effective_model = "gpt-4o"
    use_openai = True
    # Auto-enable general knowledge for richer synthesis
    if not allow_general_knowledge:
        allow_general_knowledge = True
```

## Multi-Source Support: YouTube Integration

RAG isn't just about PDFs. We support YouTube videos with a similar fallback pattern:

```python
def process_youtube(youtube_url: str, space_id: str) -> str:
    # Initialize service with fallbacks
    wetro_service = WetroCloudYouTubeService()

    # Get transcript (with automatic fallback)
    transcript_result = wetro_service.get_transcript(youtube_url)

    # Chunk the transcript
    chunks = chunk_document(
        text=transcript_text,  # Pass text directly
        chunk_size=512,
        recipe="markdown"
    )

    # Embed and store
    embeddings = embed_chunks_multimodal(cleaned_chunks)
    upsert_vectors(doc_id=document_id, embeddings=embeddings, chunks=cleaned_chunks)
```

The key insight: transcripts and documents share the same embedding and storage infrastructure. The only difference is the extraction step.

## Performance Optimizations Summary

Throughout the pipeline, we apply several optimizations:

| Optimization           | Impact                    | Implementation                     |
| ---------------------- | ------------------------- | ---------------------------------- |
| Query embedding cache  | Avoid redundant API calls | `@lru_cache(maxsize=100)`          |
| Parallel hybrid search | Faster retrieval          | `ThreadPoolExecutor`               |
| History truncation     | Controlled context size   | `max_messages=10, max_tokens=2000` |
| Batch embeddings       | Fewer API calls           | `batch_size=64`                    |
| Context limiting       | Lower generation costs    | `max_context_chunks=5`             |

## Error Handling Philosophy

Our error handling follows a simple principle: **never fail silently, always provide useful feedback**.

```python
try:
    result = await answer_query(...)
    result["time_ms"] = int((time.time() - start_time) * 1000)
    return JSONResponse(result)
except Exception as e:
    # Log the full exception for debugging
    logging.exception(f"Error in answer_query: {e}")
    # Return a user-friendly message
    return JSONResponse({"error": str(e)}, status_code=500)
```

For ingestion, we continue processing even if individual files fail:

```python
document_ids = []
errors = []

for file in files:
    try:
        document_id = await process_document_with_openai(file_path, metadata)
        document_ids.append({"file": file.filename, "document_id": document_id})
    except Exception as e:
        errors.append({"file": file.filename, "error": str(e)})

# Return partial success with errors
if document_ids:
    result = {"document_ids": document_ids}
    if errors:
        result["errors"] = errors
    return JSONResponse(result)
```

## Conclusion

Building a production RAG pipeline is about making pragmatic choices:

1. **Use fallback chains everywhere** - Extraction, LLMs, transcript services
2. **Optimize selectively** - Don't correct every chunk, cache repeated operations
3. **Go multimodal early** - Unified embeddings simplify architecture
4. **Stream for UX** - Users prefer progressive feedback over loading spinners
5. **Log extensively** - You'll need it when debugging production issues

The architecture we've outlined handles thousands of documents and queries reliably. It's not the simplest possible RAG system, but it's one that survives contact with real users and real documents.

In the next article, we'll dive deep into the text extraction layer - exploring why we need multiple strategies and how to score extraction quality automatically.

---

_This is Part 1 of a 12-part series on building production RAG systems. Next up: "The Art of PDF Text Extraction: Multi-Strategy Approaches with Quality Scoring"_
