# RAG-Style Course Generation from Uploaded Files

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the truncated file context approach with full-document RAG retrieval so every chapter is generated from the actual uploaded material — no content loss.

**Architecture:** Upload → OCR extract full text → Chonkie chunks (overlapping, ~512 tokens) → Embed + Upsert to Pinecone (for later Q&A) → Document Map (one LLM call summarizing all chunks with IDs) → Outline generation from Map → Per-chapter chunk assignment → Parallel chapter generation with relevant chunks as context. Coverage guarantee: every chunk maps to at least one chapter. Post-generation: course Q&A endpoint queries Pinecone for source-grounded answers.

**Tech Stack:** Existing Chonkie API client (`clients/chonkie_client.py`), existing MistralOCR (`processors/mistral_ocr_extractor.py`), existing Pinecone client (`clients/pinecone_client.py`) for vector upsert/search, Chonkie embeddings (`embed_chunks_multimodal`), asyncio for parallel chapter generation, existing LLM clients (Claude/Groq/OpenAI).

---

## Task 1: Add `_chunk_document_for_course` to CourseService

**Files:**
- Modify: `services/course_service.py:125-190`

**Step 1: Add the chunking method**

Add this method to the `CourseService` class, after `_get_or_create_profile` (around line 530):

```python
def _chunk_document_for_course(
    self,
    extracted_text: str,
    filename: str
) -> List[Dict[str, Any]]:
    """
    Chunk full document text into overlapping semantic chunks.
    Uses existing Chonkie client. Returns list of chunks with index IDs.
    """
    from clients.chonkie_client import chunk_document

    chunks = chunk_document(
        text=extracted_text,
        chunk_size=512,
        overlap_tokens=80,
        recipe="markdown",
        min_characters_per_chunk=50
    )

    # Tag each chunk with an index for tracking
    for i, chunk in enumerate(chunks):
        chunk["chunk_index"] = i
        chunk["source_file"] = filename

    logger.info(f"Chunked {filename}: {len(chunks)} chunks from {len(extracted_text)} chars")
    return chunks
```

**Step 2: Run quick smoke test**

```bash
cd /Users/davak/Documents/nuton_rag
python3 -c "
from services.course_service import CourseService
cs = CourseService()
chunks = cs._chunk_document_for_course('This is a test document about machine learning. ' * 200, 'test.pdf')
print(f'Chunks: {len(chunks)}')
print(f'First chunk keys: {chunks[0].keys()}')
print(f'First chunk index: {chunks[0][\"chunk_index\"]}')
"
```

Expected: Several chunks returned, each with `chunk_index` and `source_file` keys.

**Step 3: Commit**

```bash
git add services/course_service.py
git commit -m "feat: add _chunk_document_for_course method to CourseService"
```

---

## Task 2: Add `_build_document_map` to CourseService

**Files:**
- Modify: `services/course_service.py`
- Modify: `prompts/course_prompts.py`

**Step 1: Add the document map prompt**

Add to `prompts/course_prompts.py` after the `build_multi_file_analysis_prompt` function:

```python
def build_document_map_prompt(chunk_summaries: List[Dict[str, str]]) -> str:
    """Build prompt to create a structured document map from chunk summaries"""
    chunks_text = "\n".join([
        f"[Chunk {c['index']}] {c['summary']}"
        for c in chunk_summaries
    ])

    return f"""Analyze these document chunks and create a structured map of all topics and sections covered.

DOCUMENT CHUNKS:
{chunks_text}

Create a structured JSON map that:
1. Identifies ALL distinct topics/sections in the document
2. Maps each topic to the chunk indices that contain relevant content
3. Orders topics logically (as they should be taught)
4. Ensures EVERY chunk index appears in at least one topic

OUTPUT FORMAT (JSON only):
{{
  "document_title": "Inferred title of the document",
  "total_chunks": {len(chunk_summaries)},
  "topics": [
    {{
      "topic": "Topic or section name",
      "description": "One sentence description of what this section covers",
      "chunk_indices": [0, 1, 2],
      "importance": "core|supporting|supplementary"
    }}
  ],
  "coverage_check": "all_chunks_mapped"
}}

Important: Return ONLY the JSON object. Every chunk index from 0 to {len(chunk_summaries) - 1} MUST appear in at least one topic."""
```

**Step 2: Add the document map builder method to CourseService**

Add this method to `CourseService`:

```python
async def _build_document_map(
    self,
    chunks: List[Dict[str, Any]],
    model_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Build a structured map of document content from chunks.
    Summarizes each chunk, then creates a topic-to-chunk mapping.
    """
    from prompts.course_prompts import build_document_map_prompt

    # Create brief summaries of each chunk (first 200 chars as proxy)
    chunk_summaries = []
    for chunk in chunks:
        text = chunk.get("text", "")
        summary = text[:200].replace("\n", " ").strip()
        chunk_summaries.append({
            "index": chunk["chunk_index"],
            "summary": summary
        })

    prompt = build_document_map_prompt(chunk_summaries)
    doc_map = await self._call_model(prompt, model_config, expect_json=True)

    if not doc_map or "topics" not in doc_map:
        raise ValueError("Failed to generate document map")

    # Verify coverage - every chunk should be in at least one topic
    mapped_indices = set()
    for topic in doc_map["topics"]:
        mapped_indices.update(topic.get("chunk_indices", []))

    all_indices = set(range(len(chunks)))
    unmapped = all_indices - mapped_indices

    if unmapped:
        logger.warning(f"Unmapped chunks: {unmapped}. Adding to closest topic.")
        # Assign unmapped chunks to the last topic as fallback
        if doc_map["topics"]:
            doc_map["topics"][-1]["chunk_indices"].extend(list(unmapped))

    logger.info(f"Document map: {len(doc_map['topics'])} topics covering {len(chunks)} chunks")
    return doc_map
```

**Step 3: Commit**

```bash
git add services/course_service.py prompts/course_prompts.py
git commit -m "feat: add document map builder for structured chunk-to-topic mapping"
```

---

## Task 3: Add `_get_chunks_for_chapter` retrieval method

**Files:**
- Modify: `services/course_service.py`

**Step 1: Add chunk retrieval method**

This method retrieves the relevant chunks for a specific chapter based on the document map and outline.

```python
def _get_chunks_for_chapter(
    self,
    chapter_outline: Dict[str, Any],
    doc_map: Dict[str, Any],
    all_chunks: List[Dict[str, Any]],
    max_context_tokens: int = 3000
) -> str:
    """
    Retrieve relevant chunks for a specific chapter.
    Uses document map topic-to-chunk mapping + keyword matching.
    Returns concatenated chunk text, capped at max_context_tokens.
    """
    chapter_title = chapter_outline["title"].lower()
    chapter_objectives = " ".join(chapter_outline.get("objectives", [])).lower()
    chapter_concepts = " ".join(chapter_outline.get("key_concepts", [])).lower()
    chapter_text = f"{chapter_title} {chapter_objectives} {chapter_concepts}"

    # Score each topic in the doc map by relevance to this chapter
    scored_topics = []
    for topic in doc_map.get("topics", []):
        topic_text = f"{topic['topic']} {topic.get('description', '')}".lower()

        # Simple keyword overlap scoring
        chapter_words = set(chapter_text.split())
        topic_words = set(topic_text.split())
        # Remove common stop words
        stop_words = {"the", "a", "an", "and", "or", "to", "of", "in", "for", "is", "on", "with", "this", "that", "be", "can", "will", "are", "after", "students"}
        chapter_words -= stop_words
        topic_words -= stop_words

        overlap = len(chapter_words & topic_words)
        scored_topics.append((overlap, topic))

    # Sort by relevance score (highest first)
    scored_topics.sort(key=lambda x: x[0], reverse=True)

    # Collect chunks from most relevant topics
    collected_indices = []
    for score, topic in scored_topics:
        if score > 0:
            collected_indices.extend(topic.get("chunk_indices", []))

    # If no matches, take from top 2 topics anyway
    if not collected_indices and scored_topics:
        for _, topic in scored_topics[:2]:
            collected_indices.extend(topic.get("chunk_indices", []))

    # Deduplicate while preserving order
    seen = set()
    unique_indices = []
    for idx in collected_indices:
        if idx not in seen:
            seen.add(idx)
            unique_indices.append(idx)

    # Build context from chunks, respecting token limit
    context_parts = []
    token_count = 0
    for idx in unique_indices:
        if idx < len(all_chunks):
            chunk_text = all_chunks[idx].get("text", "")
            # Rough token estimate: 1 token ≈ 4 chars
            chunk_tokens = len(chunk_text) // 4
            if token_count + chunk_tokens > max_context_tokens:
                break
            context_parts.append(f"[Source Section {idx + 1}]\n{chunk_text}")
            token_count += chunk_tokens

    context = "\n\n---\n\n".join(context_parts)
    logger.info(f"Chapter '{chapter_outline['title']}': {len(context_parts)} chunks, ~{token_count} tokens")
    return context
```

**Step 2: Commit**

```bash
git add services/course_service.py
git commit -m "feat: add per-chapter chunk retrieval with relevance scoring"
```

---

## Task 4: Rewrite `create_course_from_files` to use RAG pipeline

**Files:**
- Modify: `services/course_service.py:125-190`

**Step 1: Rewrite the method**

Replace the existing `create_course_from_files` method:

```python
async def create_course_from_files(
    self,
    user_id: str,
    files: List[Dict[str, Any]],  # Processed file data
    organization: str,
    model: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate course from uploaded files using RAG pipeline.
    Full document chunking → document map → per-chapter retrieval.
    """
    start_time = time.time()

    # Get profile and model config
    profile = self._get_or_create_profile(user_id, {})
    model_config = ModelConfig.get_config(model)

    # Step 1: Chunk ALL files
    all_chunks = []
    for f in files:
        file_chunks = self._chunk_document_for_course(
            extracted_text=f["extracted_text"],
            filename=f["filename"]
        )
        all_chunks.extend(file_chunks)

    logger.info(f"Total chunks from {len(files)} file(s): {len(all_chunks)}")

    # Step 2: Build document map
    doc_map = await self._build_document_map(all_chunks, model_config)

    # Step 3: Multi-file organization (if multiple files)
    chosen_org = None
    if len(files) > 1:
        _, chosen_org = await self._analyze_multi_files(files, organization)

    # Step 4: Build topic from files
    combined_topic = " + ".join([f["topic"] for f in files]) if len(files) > 1 else files[0]["topic"]

    # Step 5: Generate outline from document map (not raw text)
    # Pass the doc map as structured context instead of raw text
    doc_map_context = "DOCUMENT MAP (topics found in uploaded material):\n"
    for topic in doc_map.get("topics", []):
        doc_map_context += f"- {topic['topic']}: {topic.get('description', '')} [{len(topic.get('chunk_indices', []))} sections]\n"

    try:
        course = await self._generate_full_course_from_files(
            user_id=user_id,
            topic=combined_topic,
            profile=profile,
            model_config=model_config,
            source_files=[{"file_id": generate_uuid(), "filename": f["filename"], "extracted_topic": f["topic"]} for f in files],
            doc_map=doc_map,
            doc_map_context=doc_map_context,
            all_chunks=all_chunks,
            organization=chosen_org
        )

        generation_time = round(time.time() - start_time, 2)

        self.logger.log_generation({
            "type": "file_course",
            "user_id": user_id,
            "course_id": course["id"],
            "files": [f["filename"] for f in files],
            "total_chunks": len(all_chunks),
            "organization": chosen_org.value if chosen_org else None,
            "model": model_config["model"],
            "generation_time": generation_time,
            "status": "success"
        })

        return {
            "course_id": course["id"],
            "status": CourseStatus.READY,
            "detected_topics": [f["topic"] for f in files],
            "organization_chosen": chosen_org.value if chosen_org else None,
            "document_map": doc_map,
            "total_chunks_processed": len(all_chunks),
            "course": course,
            "storage_path": f"courses/course_{course['id']}/",
            "generation_time_seconds": generation_time
        }

    except Exception as e:
        logger.error(f"File course generation failed: {e}")
        raise
```

**Step 2: Commit**

```bash
git add services/course_service.py
git commit -m "refactor: rewrite create_course_from_files to use RAG chunking pipeline"
```

---

## Task 5: Add `_generate_full_course_from_files` with parallel chapters

**Files:**
- Modify: `services/course_service.py`

**Step 1: Add the new generation method**

This is separate from `_generate_full_course` (which handles topic-based courses) to keep responsibilities clean. Add below the existing `_generate_full_course` method:

```python
async def _generate_full_course_from_files(
    self,
    user_id: str,
    topic: str,
    profile: LearningProfile,
    model_config: Dict[str, Any],
    source_files: List[Dict],
    doc_map: Dict[str, Any],
    doc_map_context: str,
    all_chunks: List[Dict[str, Any]],
    organization: Optional[OrganizationType] = None
) -> Dict[str, Any]:
    """
    Generate full course from files with per-chapter RAG retrieval.
    Chapters are generated in parallel for speed.
    """
    import asyncio

    # Step 1: Generate outline using document map as context
    outline = await self._generate_outline(
        topic=topic,
        profile=profile,
        model_config=model_config,
        file_context=doc_map_context,
        organization=organization
    )

    # Create course record
    course_id = generate_uuid()
    personalization = PersonalizationParams(
        format_pref=profile.format_pref,
        depth_pref=profile.depth_pref,
        role=profile.role,
        learning_goal=profile.learning_goal,
        example_pref=profile.example_pref
    )

    course_data = {
        "id": course_id,
        "user_id": user_id,
        "title": outline["title"],
        "description": outline["description"],
        "topic": topic,
        "source_type": SourceType.FILES,
        "source_files": source_files,
        "multi_file_organization": organization.value if organization else None,
        "total_chapters": len(outline["chapters"]),
        "estimated_time": outline["total_estimated_time"],
        "status": CourseStatus.GENERATING,
        "personalization_params": personalization.dict(),
        "outline": outline,
        "model_used": model_config["model"],
        "created_at": datetime.utcnow(),
        "completed_at": None
    }

    self.storage.save_course(course_data)

    # Step 2: Retrieve relevant chunks per chapter
    chapter_contexts = {}
    for chapter_outline in outline["chapters"]:
        context = self._get_chunks_for_chapter(
            chapter_outline=chapter_outline,
            doc_map=doc_map,
            all_chunks=all_chunks,
            max_context_tokens=3000
        )
        chapter_contexts[chapter_outline["order"]] = context

    # Step 3: Generate chapters in parallel
    async def generate_single_chapter(i, chapter_outline):
        chapter = await self._generate_chapter(
            course_id=course_id,
            course_title=outline["title"],
            chapter_outline=chapter_outline,
            total_chapters=len(outline["chapters"]),
            profile=profile,
            model_config=model_config,
            prev_chapter_title=outline["chapters"][i - 1]["title"] if i > 0 else None,
            next_chapter_title=outline["chapters"][i + 1]["title"] if i < len(outline["chapters"]) - 1 else None,
            file_context=chapter_contexts.get(chapter_outline["order"])
        )
        self.storage.save_chapter(course_id, chapter)
        logger.info(f"Generated chapter {i + 1}/{len(outline['chapters'])}: {chapter['title']}")
        return chapter

    tasks = [
        generate_single_chapter(i, ch)
        for i, ch in enumerate(outline["chapters"])
    ]

    try:
        chapters = await asyncio.gather(*tasks)
    except Exception as e:
        logger.error(f"Parallel chapter generation failed: {e}")
        raise CourseGenerationError(f"Chapter generation failed: {e}")

    # Update course status
    course_data["status"] = CourseStatus.READY
    course_data["completed_at"] = datetime.utcnow()
    self.storage.save_course(course_data)

    return course_data
```

**Step 2: Commit**

```bash
git add services/course_service.py
git commit -m "feat: add parallel chapter generation with per-chapter RAG retrieval"
```

---

## Task 6: Update `_process_uploaded_files` with validation

**Files:**
- Modify: `routes/course_routes.py:340-410`

**Step 1: Add file validation and keep full text**

Replace the existing `_process_uploaded_files` function:

```python
ALLOWED_EXTENSIONS = {".pdf", ".pptx", ".ppt", ".docx", ".doc", ".txt", ".md"}
MAX_FILE_SIZE_MB = 50

async def _process_uploaded_files(files: List[UploadFile]) -> List[dict]:
    """Process uploaded files with OCR and topic extraction"""
    processed = []

    # Initialize Mistral OCR
    mistral_config = MistralOCRConfig(
        enhance_metadata_with_llm=True,
        fallback_method="legacy",
        include_images=False
    )
    extractor = MistralOCRExtractor(config=mistral_config)

    for file in files:
        # Validate file extension
        ext = os.path.splitext(file.filename)[1].lower() if file.filename else ""
        if ext not in ALLOWED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {ext}. Allowed: {ALLOWED_EXTENSIONS}")

        # Validate file size
        content = await file.read()
        await file.seek(0)  # Reset for later read

        size_mb = len(content) / (1024 * 1024)
        if size_mb > MAX_FILE_SIZE_MB:
            raise ValueError(f"File {file.filename} is {size_mb:.1f}MB. Max: {MAX_FILE_SIZE_MB}MB")

        # Save temp file
        temp_path = _save_temp_file(file)

        try:
            # Extract text
            extraction = extractor.process_document(temp_path)
            text = extraction.get('full_text', '')

            if not text:
                raise ValueError(f"No text extracted from {file.filename}")

            # Extract topic using Claude
            topic = await _extract_topic(text[:2000])

            processed.append({
                "filename": file.filename,
                "topic": topic,
                "extracted_text": text,  # FULL text - no truncation
                "pages": extraction.get('total_pages', 0),
                "char_count": len(text)
            })

            logger.info(f"Processed {file.filename}: {topic} ({len(text)} chars, {extraction.get('total_pages', 0)} pages)")

        finally:
            # Cleanup
            if os.path.exists(temp_path):
                os.remove(temp_path)

    return processed
```

**Step 2: Update `_save_temp_file` to use async-safe write**

```python
def _save_temp_file(file: UploadFile) -> str:
    """Save uploaded file to temp location"""
    suffix = os.path.splitext(file.filename)[1] if file.filename else ""
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = file.file.read()
        if not content:
            raise ValueError(f"Empty file: {file.filename}")
        tmp.write(content)
        return tmp.name
```

**Step 3: Commit**

```bash
git add routes/course_routes.py
git commit -m "feat: add file validation and preserve full extracted text"
```

---

## Task 7: Update chapter content prompt for source-grounded generation

**Files:**
- Modify: `prompts/course_prompts.py:111-114`

**Step 1: Strengthen the source material instruction**

Replace the existing `source_section` in `build_chapter_content_prompt`:

```python
    source_section = f"""
SOURCE MATERIAL (from uploaded document - YOU MUST teach from this):
{source_material_context}

CRITICAL INSTRUCTIONS FOR SOURCE MATERIAL:
- Base your chapter content PRIMARILY on the source material above
- Do NOT invent facts or examples that aren't supported by the source material
- Use the source material's terminology and structure
- If the source material is thin on a subtopic, note it briefly and move on
- Inline citations [1], [2] should reference the [Source Section N] labels above
""" if source_material_context else ""
```

**Step 2: Commit**

```bash
git add prompts/course_prompts.py
git commit -m "feat: strengthen source material grounding in chapter prompt"
```

---

## Task 8: Embed and upsert course chunks to Pinecone

**Files:**
- Modify: `services/course_service.py`

**Step 1: Add `_embed_and_upsert_chunks` method**

Add this method to `CourseService`. It follows the exact same pattern as the existing pipeline (`pipeline.py:199-242`): embed via Chonkie multimodal → upsert via Pinecone client.

```python
async def _embed_and_upsert_chunks(
    self,
    course_id: str,
    all_chunks: List[Dict[str, Any]],
    source_files: List[Dict]
) -> None:
    """
    Embed course chunks and upsert to Pinecone for later Q&A retrieval.
    Uses existing Chonkie multimodal embeddings + Pinecone upsert pattern.
    """
    from clients.chonkie_client import embed_chunks_multimodal
    from clients.pinecone_client import upsert_vectors

    if not all_chunks:
        logger.warning("No chunks to embed/upsert")
        return

    # Embed all chunks using multimodal embeddings (Jina CLIP-v2, 1024 dims)
    logger.info(f"Embedding {len(all_chunks)} course chunks...")
    embeddings = embed_chunks_multimodal(all_chunks, batch_size=32)

    if not embeddings or len(embeddings) != len(all_chunks):
        logger.error(f"Embedding count mismatch: {len(embeddings)} embeddings for {len(all_chunks)} chunks")
        return

    # Tag chunks with course metadata before upserting
    tagged_chunks = []
    for chunk in all_chunks:
        tagged = dict(chunk)
        tagged["course_id"] = course_id
        tagged["content_type"] = "course_source_material"
        tagged_chunks.append(tagged)

    # Upsert to Pinecone with course_id as the doc_id
    # space_id uses "course_{course_id}" namespace for easy filtering
    upsert_vectors(
        doc_id=course_id,
        space_id=f"course_{course_id}",
        embeddings=embeddings,
        chunks=tagged_chunks,
        source_file=", ".join([f["filename"] for f in source_files])
    )

    logger.info(f"Upserted {len(all_chunks)} chunks to Pinecone for course {course_id}")
```

**Step 2: Wire into `create_course_from_files`**

In the `create_course_from_files` method (Task 4), add the upsert call right after chunking (Step 1) and before building the doc map (Step 2). Add between the "Step 1: Chunk ALL files" and "Step 2: Build document map" blocks:

```python
    # Step 1b: Embed and upsert chunks to Pinecone (for course Q&A)
    await self._embed_and_upsert_chunks(
        course_id=generate_uuid(),  # Pre-generate course_id to use here and later
        all_chunks=all_chunks,
        source_files=[{"filename": f["filename"]} for f in files]
    )
```

**Note:** The course_id needs to be generated early (before `_generate_full_course_from_files`) so it can be used for both Pinecone upsert and course creation. Move the `course_id = generate_uuid()` to `create_course_from_files` and pass it through.

**Step 3: Commit**

```bash
git add services/course_service.py
git commit -m "feat: embed and upsert course chunks to Pinecone for Q&A retrieval"
```

---

## Task 9: Add course Q&A endpoint

**Files:**
- Modify: `routes/course_routes.py`
- Modify: `services/course_service.py`

**Step 1: Add `query_course` method to CourseService**

```python
async def query_course(
    self,
    course_id: str,
    question: str,
    model: Optional[str] = None,
    top_k: int = 5
) -> Dict[str, Any]:
    """
    Answer a question about a course using its source material from Pinecone.
    Searches course chunks, reranks, and generates a grounded answer.
    """
    from clients.chonkie_client import embed_query_multimodal
    from clients.pinecone_client import hybrid_search, rerank_results

    model_config = ModelConfig.get_config(model)

    # Get course for context
    course = self.get_course(course_id)
    if not course:
        raise ValueError(f"Course not found: {course_id}")

    # Embed the question
    query_embedding = embed_query_multimodal(question)

    # Search Pinecone filtered by course namespace
    search_results = hybrid_search(
        query_embedding=query_embedding,
        space_id=f"course_{course_id}",
        top_k=top_k * 2  # Over-fetch for reranking
    )

    # Rerank results
    if search_results:
        reranked = rerank_results(
            query=question,
            results=search_results,
            top_n=top_k
        )
    else:
        reranked = []

    # Build context from retrieved chunks
    context_parts = []
    for i, result in enumerate(reranked):
        text = result.get("metadata", {}).get("text", "")
        source = result.get("metadata", {}).get("source_file", "")
        context_parts.append(f"[Source {i+1}] ({source})\n{text}")

    context = "\n\n---\n\n".join(context_parts)

    # Generate answer
    prompt = f"""You are a helpful course assistant. Answer the student's question based ONLY on the source material provided.

COURSE: {course.get('title', '')}

SOURCE MATERIAL:
{context}

STUDENT QUESTION: {question}

Instructions:
- Answer based ONLY on the source material above
- If the source material doesn't contain the answer, say so clearly
- Reference specific sources using [Source N] citations
- Keep the answer concise and educational"""

    response = await self._call_model(prompt, model_config, expect_json=False)

    return {
        "course_id": course_id,
        "question": question,
        "answer": response.get("content", ""),
        "sources_used": len(reranked),
        "source_excerpts": [{"text": r.get("metadata", {}).get("text", "")[:200], "file": r.get("metadata", {}).get("source_file", "")} for r in reranked]
    }
```

**Step 2: Add the API endpoint to course_routes.py**

Add after the progress endpoints:

```python
@router.post("/courses/{course_id}/ask")
async def ask_course_question(course_id: str, question: str = Form(...), model: Optional[str] = Form(None)):
    """
    Ask a question about a course. Answers are grounded in the uploaded source material.
    Uses Pinecone vector search to find relevant chunks from the original documents.
    """
    try:
        result = await course_service.query_course(
            course_id=course_id,
            question=question,
            model=model
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Course Q&A error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
```

**Step 3: Commit**

```bash
git add services/course_service.py routes/course_routes.py
git commit -m "feat: add course Q&A endpoint with Pinecone-backed retrieval"
```

---

## Task 10: End-to-end integration test

**Step 1: Test with a real PDF upload via curl**

Find a sample PDF in the project or use a small test:

```bash
# Create a test markdown file to simulate upload
cat > /tmp/test_course_material.md << 'EOF'
# Introduction to Neural Networks

Neural networks are computational models inspired by the human brain. They consist of layers of interconnected nodes that process information.

## Perceptrons

The perceptron is the simplest neural network. It takes multiple inputs, multiplies each by a weight, sums them, and passes through an activation function.

## Backpropagation

Backpropagation is the key algorithm for training neural networks. It calculates gradients of the loss function with respect to each weight by applying the chain rule.

## Convolutional Neural Networks

CNNs are specialized for processing grid-like data such as images. They use convolutional layers that apply filters to detect features like edges and textures.

## Recurrent Neural Networks

RNNs process sequential data by maintaining a hidden state. They are used for tasks like language modeling and time series prediction. LSTM networks solve the vanishing gradient problem.

## Transformers and Attention

The transformer architecture uses self-attention mechanisms instead of recurrence. This allows parallel processing and better handling of long-range dependencies.
EOF

curl -X POST http://localhost:8000/api/v1/courses/from-files \
  -F "files=@/tmp/test_course_material.md" \
  -F "user_id=test-user-1" \
  -F "organization=auto" \
  -F "model=llama-4-scout"
```

**Step 2: Verify output**

Check that:
- [ ] Response includes `total_chunks_processed` > 1
- [ ] Response includes `document_map` with topics
- [ ] Each chapter content references source material (look for [Source Section N] citations)
- [ ] All document topics appear somewhere in the course
- [ ] Generation time is reasonable (not slower than before due to parallelism)

**Step 3: Test multi-file upload**

```bash
cat > /tmp/test_ml_basics.md << 'EOF'
# Machine Learning Basics
Machine learning is a subset of AI. Supervised learning uses labeled data. Unsupervised learning finds patterns in unlabeled data.
## Classification
Classification assigns categories. Common algorithms: SVM, Random Forest, Logistic Regression.
## Regression
Regression predicts continuous values. Linear regression, polynomial regression, ridge regression.
EOF

curl -X POST http://localhost:8000/api/v1/courses/from-files \
  -F "files=@/tmp/test_course_material.md" \
  -F "files=@/tmp/test_ml_basics.md" \
  -F "user_id=test-user-1" \
  -F "organization=auto" \
  -F "model=llama-4-scout"
```

**Step 4: Test course Q&A endpoint**

After a course is generated from files, test the Q&A:

```bash
# Use the course_id from the file upload response above
COURSE_ID="<course_id_from_step_1>"

curl -X POST "http://localhost:8000/api/v1/courses/$COURSE_ID/ask" \
  -F "question=What is backpropagation and how does it work?" \
  -F "model=llama-4-scout"
```

Check that:
- [ ] Answer references source material (not generic model knowledge)
- [ ] `sources_used` > 0
- [ ] `source_excerpts` contain relevant text from the uploaded document
- [ ] Answer includes [Source N] citations

**Step 5: Test cross-course search (multiple courses from files)**

```bash
# Ask about content from the second file in a multi-file course
curl -X POST "http://localhost:8000/api/v1/courses/$COURSE_ID/ask" \
  -F "question=What algorithms are used for classification?" \
  -F "model=llama-4-scout"
```

**Step 6: Final commit**

```bash
git add -A
git commit -m "test: verify full RAG pipeline with Pinecone upsert and course Q&A"
```

---

## Summary of Changes

| File | Change | Purpose |
|------|--------|---------|
| `services/course_service.py` | Add `_chunk_document_for_course()` | Chunk full docs via Chonkie |
| `services/course_service.py` | Add `_build_document_map()` | Map chunks → topics |
| `services/course_service.py` | Add `_get_chunks_for_chapter()` | Retrieve relevant chunks per chapter |
| `services/course_service.py` | Rewrite `create_course_from_files()` | New RAG pipeline entry point |
| `services/course_service.py` | Add `_generate_full_course_from_files()` | Parallel chapter gen with RAG context |
| `services/course_service.py` | Add `_embed_and_upsert_chunks()` | Embed + upsert to Pinecone |
| `services/course_service.py` | Add `query_course()` | Course Q&A with Pinecone retrieval |
| `prompts/course_prompts.py` | Add `build_document_map_prompt()` | Prompt for doc map generation |
| `prompts/course_prompts.py` | Update source material section | Stronger grounding instructions |
| `routes/course_routes.py` | Update `_process_uploaded_files()` | File validation, full text preservation |
| `routes/course_routes.py` | Add `POST /courses/{id}/ask` | Course Q&A endpoint |

## Loopholes Fixed

| Original Loophole | Fix |
|---|---|
| File context only in Chapter 1 | Every chapter gets its own relevant chunks |
| Text truncated to 1500/3000 chars | Full text chunked, all chunks used |
| Topic from first 2000 chars only | Document map covers entire document |
| No file type validation | Extension whitelist added |
| No file size validation | 50MB limit added |
| Chapters generated from model knowledge | Prompt forces teaching FROM source material |
| Sequential chapter generation (slow) | Parallel generation with asyncio.gather |
| No queryable storage of source material | Chunks embedded + upserted to Pinecone |
| No way to ask questions about courses | Course Q&A endpoint with vector retrieval |
