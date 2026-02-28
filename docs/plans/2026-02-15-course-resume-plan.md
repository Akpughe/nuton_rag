# Course Resume Endpoint Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add `POST /courses/{course_id}/resume` and `/resume/stream` endpoints that detect missing/errored chapters and generate only what's needed to complete a partially-generated course.

**Architecture:** Standalone resume endpoints call a new `resume_course()` service method. It performs gap analysis (outline chapters vs existing DB chapters), generates missing chapters using the existing `_generate_chapter()` method in batched parallel, fills in missing study guide/flashcards, then finalizes the course status. For file-based courses, chunks are retrieved from Pinecone.

**Tech Stack:** FastAPI, Supabase (Postgres), Pinecone (vector store), existing LLM dispatch (`_call_model`)

---

### Task 1: Add `resume_course` Service Method (Blocking)

**Files:**
- Modify: `services/course_service.py` (add method after line ~131, after `create_course_from_topic`)

**Step 1: Add the `resume_course` method to `CourseService`**

Add this method to the `CourseService` class. It handles gap analysis, chapter generation, extras, and finalization:

```python
async def resume_course(
    self,
    course_id: str,
    model: Optional[str] = None
) -> Dict[str, Any]:
    """
    Resume generation of a partially-completed course.
    Detects missing/errored chapters and generates only the gaps.
    Re-entrant: safe to call multiple times.
    """
    start_time = time.time()

    # --- Gap Analysis ---
    course = self.storage.get_course(course_id)
    if not course:
        raise NotFoundError(
            "Course not found",
            error_code="COURSE_NOT_FOUND",
            context={"course_id": course_id},
        )

    outline = course.get("outline")
    if not outline or "chapters" not in outline:
        raise ValidationError(
            "Course has no outline — cannot resume. Please regenerate the course.",
            error_code="NO_OUTLINE",
            context={"course_id": course_id},
        )

    # Model config: use override or original
    model_to_use = model or course.get("model_used")
    model_config = ModelConfig.get_config(model_to_use)

    # Get existing chapters and find gaps
    existing_chapters = course.get("chapters", [])
    existing_ready = {
        ch["order"] for ch in existing_chapters if ch.get("status") == "ready"
    }
    outline_orders = {ch["order"] for ch in outline["chapters"]}
    missing_orders = sorted(outline_orders - existing_ready)

    # Check extras
    needs_study_guide = course.get("study_guide") is None
    needs_flashcards = course.get("flashcards") is None

    # Early return if complete
    if not missing_orders and not needs_study_guide and not needs_flashcards:
        return {
            "course_id": course_id,
            "status": course.get("status", "ready"),
            "course": course,
            "resume_summary": {
                "chapters_total": len(outline["chapters"]),
                "chapters_existed": len(existing_ready),
                "chapters_generated": 0,
                "chapters_failed": [],
                "study_guide_generated": False,
                "flashcards_generated": False,
                "already_complete": True,
            },
        }

    # --- Reconstruct context ---
    user_id = course.get("user_id", "")
    profile = self._get_or_create_profile(user_id, {})
    topic = course.get("topic", "")

    # Determine search mode
    model_key = None
    from utils.model_config import MODEL_CONFIGS
    for key, cfg in MODEL_CONFIGS.items():
        if cfg["model"] == model_config["model"]:
            model_key = key
            break
    search_mode = get_search_mode(model_key)

    # File context from Pinecone (if file-based course)
    file_contexts: Dict[int, str] = {}
    source_type = course.get("source_type", "")
    if source_type in ("files", "mixed"):
        try:
            file_contexts = await self._retrieve_chunks_for_resume(
                course_id, outline["chapters"], missing_orders
            )
        except Exception as e:
            logger.warning(f"Pinecone chunk retrieval failed for resume, proceeding without file context: {e}")

    # Pre-fetch web sources for Perplexity if needed
    chapter_web_sources: Dict[int, Dict] = {}
    if search_mode == "perplexity":
        from clients.perplexity_client import search_for_chapters_parallel
        missing_outlines = [ch for ch in outline["chapters"] if ch["order"] in missing_orders]
        chapter_web_sources = search_for_chapters_parallel(
            chapters=missing_outlines,
            course_topic=topic
        )

    # --- Generate missing chapters (batched parallel) ---
    BATCH_SIZE = 4
    chapters_generated = 0
    chapters_failed = []
    outline_lookup = {ch["order"]: ch for ch in outline["chapters"]}

    missing_items = []
    for order in missing_orders:
        ch_outline = outline_lookup[order]
        idx = order - 1  # 0-based index for prev/next lookup
        missing_items.append((idx, ch_outline))

    try:
        for batch_start in range(0, len(missing_items), BATCH_SIZE):
            batch = missing_items[batch_start:batch_start + BATCH_SIZE]
            tasks = []
            for idx, ch_outline in batch:
                ch_order = ch_outline["order"]
                ch_search_mode = search_mode
                if search_mode == "perplexity" and chapter_web_sources.get(ch_order, {}).get("sources"):
                    ch_search_mode = "provided"
                ch_web_sources = chapter_web_sources.get(ch_order, {}).get("sources") if search_mode == "perplexity" else None
                ch_file_context = file_contexts.get(ch_order)
                ch_use_search = (search_mode == "native" and model_config["provider"] == "openai")

                tasks.append(self._generate_chapter(
                    course_id=course_id,
                    course_title=outline["title"],
                    chapter_outline=ch_outline,
                    total_chapters=len(outline["chapters"]),
                    profile=profile,
                    model_config=model_config,
                    prev_chapter_title=outline["chapters"][idx - 1]["title"] if idx > 0 else None,
                    next_chapter_title=outline["chapters"][idx + 1]["title"] if idx < len(outline["chapters"]) - 1 else None,
                    file_context=ch_file_context,
                    search_mode=ch_search_mode,
                    web_sources=ch_web_sources,
                    use_search=ch_use_search,
                ))

            results = await asyncio.gather(*tasks, return_exceptions=True)
            for (_, ch_outline), result in zip(batch, results):
                if isinstance(result, Exception):
                    logger.error(f"Resume: chapter {ch_outline['order']} failed: {result}")
                    chapters_failed.append(ch_outline["order"])
                else:
                    self.storage.save_chapter(course_id, result)
                    chapters_generated += 1
                    logger.info(f"Resume: generated chapter {ch_outline['order']}: {result['title']}")

            if batch_start + BATCH_SIZE < len(missing_items):
                await asyncio.sleep(1)
    except Exception as e:
        logger.error(f"Resume batched chapter generation failed: {e}")
        raise GenerationError(
            f"Chapter generation failed during resume: {e}",
            error_code="CHAPTER_GENERATION_FAILED",
        )

    # --- Generate missing extras ---
    sg_generated = False
    fc_generated = False
    if needs_study_guide or needs_flashcards:
        all_chapters = self.storage.get_course(course_id).get("chapters", [])
        try:
            extras_tasks = []
            if needs_study_guide:
                extras_tasks.append(("sg", self._generate_study_guide(
                    course_id, all_chapters, profile, model_config, topic
                )))
            if needs_flashcards:
                extras_tasks.append(("fc", self._generate_course_flashcards(
                    course_id, all_chapters, profile, model_config
                )))

            results = await asyncio.gather(
                *[t for _, t in extras_tasks], return_exceptions=True
            )
            for (label, _), result in zip(extras_tasks, results):
                if isinstance(result, Exception):
                    logger.warning(f"Resume: {label} generation failed (non-fatal): {result}")
                elif result:
                    if label == "sg":
                        sg_generated = True
                    else:
                        fc_generated = True
        except Exception as e:
            logger.warning(f"Resume: extras generation error (non-fatal): {e}")

    # --- Finalize ---
    if not chapters_failed:
        course_update = {
            "id": course_id,
            "status": CourseStatus.READY,
            "completed_at": datetime.utcnow(),
        }
        self.storage.save_course(course_update)

    # Return full course
    final_course = self.storage.get_course(course_id)
    generation_time = round(time.time() - start_time, 2)

    return {
        "course_id": course_id,
        "status": final_course.get("status", "generating"),
        "course": final_course,
        "generation_time_seconds": generation_time,
        "resume_summary": {
            "chapters_total": len(outline["chapters"]),
            "chapters_existed": len(existing_ready),
            "chapters_generated": chapters_generated,
            "chapters_failed": chapters_failed,
            "study_guide_generated": sg_generated,
            "flashcards_generated": fc_generated,
            "already_complete": False,
        },
    }
```

**Step 2: Add the `_retrieve_chunks_for_resume` helper**

Add this private method to `CourseService` (after the `resume_course` method). It queries Pinecone for stored chunks by course_id and maps them to chapter outlines:

```python
async def _retrieve_chunks_for_resume(
    self,
    course_id: str,
    chapter_outlines: List[Dict[str, Any]],
    missing_orders: List[int],
) -> Dict[int, str]:
    """
    Retrieve stored chunks from Pinecone for a file-based course resume.
    Returns a dict mapping chapter order -> context string.
    Only fetches context for missing chapters.
    """
    from clients.pinecone_client import get_pinecone_index
    import os

    index = get_pinecone_index()
    namespace = os.getenv("PINECONE_NAMESPACE", "nuton-courses")

    # Query all vectors for this course_id
    # Use a dummy query vector to fetch by metadata filter
    results = index.query(
        vector=[0.0] * 1024,  # Jina CLIP-v2 dimension
        top_k=200,
        filter={"course_id": {"$eq": course_id}},
        include_metadata=True,
        namespace=namespace,
    )

    if not results.matches:
        logger.warning(f"No Pinecone chunks found for course {course_id}")
        return {}

    # Build chunks list from Pinecone metadata
    all_chunks = []
    for match in results.matches:
        meta = match.metadata or {}
        all_chunks.append({
            "text": meta.get("text", ""),
            "index": meta.get("chunk_index", 0),
            "filename": meta.get("filename", ""),
        })

    # Map chunks to missing chapters using keyword overlap from outlines
    chapter_contexts: Dict[int, str] = {}
    for ch_outline in chapter_outlines:
        if ch_outline["order"] not in missing_orders:
            continue

        # Simple keyword matching: chapter title + key_concepts
        keywords = set()
        keywords.update(ch_outline.get("title", "").lower().split())
        for concept in ch_outline.get("key_concepts", []):
            keywords.update(concept.lower().split())

        # Score each chunk by keyword overlap
        scored = []
        for chunk in all_chunks:
            chunk_words = set(chunk["text"].lower().split())
            overlap = len(keywords & chunk_words)
            if overlap > 0:
                scored.append((overlap, chunk))

        scored.sort(key=lambda x: x[0], reverse=True)

        # Take top chunks up to ~3000 tokens
        context_parts = []
        token_count = 0
        for _, chunk in scored[:10]:
            chunk_tokens = len(chunk["text"].split()) * 1.3  # rough token estimate
            if token_count + chunk_tokens > 3000:
                break
            context_parts.append(f"[Source Section]\n{chunk['text']}")
            token_count += chunk_tokens

        if context_parts:
            chapter_contexts[ch_outline["order"]] = "\n\n".join(context_parts)

    return chapter_contexts
```

**Step 3: Add the `import asyncio` if not present at top of course_service.py**

Check line 1 of `services/course_service.py` — `asyncio` is used in `_generate_full_course_stream` (line 461) via local import. The `resume_course` method uses `asyncio.gather` so add `import asyncio` to the top-level imports if not already there.

```python
# At the top of course_service.py, after existing imports (around line 11)
import asyncio
```

**Step 4: Verify it runs**

Run: `cd /Users/davak/Documents/nuton_rag && python3 -c "from services.course_service import CourseService; print('OK')"`
Expected: `OK`

**Step 5: Commit**

```bash
git add services/course_service.py
git commit -m "feat(service): add resume_course method for partial course completion"
```

---

### Task 2: Add `resume_course_stream` Service Method (Streaming)

**Files:**
- Modify: `services/course_service.py` (add method after `resume_course`, before existing streaming methods around line ~1666)

**Step 1: Add the streaming variant**

```python
async def resume_course_stream(
    self,
    course_id: str,
    model: Optional[str] = None
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Streaming version of resume_course. Yields SSE events as chapters complete.
    """
    start_time = time.time()

    # --- Gap Analysis (same as blocking) ---
    course = self.storage.get_course(course_id)
    if not course:
        raise NotFoundError(
            "Course not found",
            error_code="COURSE_NOT_FOUND",
            context={"course_id": course_id},
        )

    outline = course.get("outline")
    if not outline or "chapters" not in outline:
        raise ValidationError(
            "Course has no outline — cannot resume. Please regenerate the course.",
            error_code="NO_OUTLINE",
            context={"course_id": course_id},
        )

    model_to_use = model or course.get("model_used")
    model_config = ModelConfig.get_config(model_to_use)

    existing_chapters = course.get("chapters", [])
    existing_ready = {
        ch["order"] for ch in existing_chapters if ch.get("status") == "ready"
    }
    outline_orders = {ch["order"] for ch in outline["chapters"]}
    missing_orders = sorted(outline_orders - existing_ready)

    needs_study_guide = course.get("study_guide") is None
    needs_flashcards = course.get("flashcards") is None

    # Early return
    if not missing_orders and not needs_study_guide and not needs_flashcards:
        yield {
            "type": "course_complete",
            "course_id": course_id,
            "status": "ready",
            "generation_time_seconds": 0,
            "resume_summary": {
                "chapters_total": len(outline["chapters"]),
                "chapters_existed": len(existing_ready),
                "chapters_generated": 0,
                "chapters_failed": [],
                "study_guide_generated": False,
                "flashcards_generated": False,
                "already_complete": True,
            },
        }
        return

    # Emit resume_started
    yield {
        "type": "resume_started",
        "course_id": course_id,
        "missing_chapters": missing_orders,
        "total_to_generate": len(missing_orders),
        "needs_study_guide": needs_study_guide,
        "needs_flashcards": needs_flashcards,
    }

    # Reconstruct context
    user_id = course.get("user_id", "")
    profile = self._get_or_create_profile(user_id, {})
    topic = course.get("topic", "")

    model_key = None
    from utils.model_config import MODEL_CONFIGS
    for key, cfg in MODEL_CONFIGS.items():
        if cfg["model"] == model_config["model"]:
            model_key = key
            break
    search_mode = get_search_mode(model_key)

    file_contexts: Dict[int, str] = {}
    source_type = course.get("source_type", "")
    if source_type in ("files", "mixed"):
        try:
            file_contexts = await self._retrieve_chunks_for_resume(
                course_id, outline["chapters"], missing_orders
            )
        except Exception as e:
            logger.warning(f"Pinecone chunk retrieval failed for resume stream: {e}")

    chapter_web_sources: Dict[int, Dict] = {}
    if search_mode == "perplexity":
        from clients.perplexity_client import search_for_chapters_parallel
        missing_outlines = [ch for ch in outline["chapters"] if ch["order"] in missing_orders]
        chapter_web_sources = search_for_chapters_parallel(
            chapters=missing_outlines,
            course_topic=topic
        )

    # Generate missing chapters
    BATCH_SIZE = 4
    chapters_generated = 0
    chapters_failed = []
    outline_lookup = {ch["order"]: ch for ch in outline["chapters"]}

    missing_items = []
    for order in missing_orders:
        ch_outline = outline_lookup[order]
        idx = order - 1
        missing_items.append((idx, ch_outline))

    try:
        for batch_start in range(0, len(missing_items), BATCH_SIZE):
            batch = missing_items[batch_start:batch_start + BATCH_SIZE]
            tasks = []
            for idx, ch_outline in batch:
                ch_order = ch_outline["order"]
                ch_search_mode = search_mode
                if search_mode == "perplexity" and chapter_web_sources.get(ch_order, {}).get("sources"):
                    ch_search_mode = "provided"
                ch_web_sources = chapter_web_sources.get(ch_order, {}).get("sources") if search_mode == "perplexity" else None
                ch_file_context = file_contexts.get(ch_order)
                ch_use_search = (search_mode == "native" and model_config["provider"] == "openai")

                tasks.append(self._generate_chapter(
                    course_id=course_id,
                    course_title=outline["title"],
                    chapter_outline=ch_outline,
                    total_chapters=len(outline["chapters"]),
                    profile=profile,
                    model_config=model_config,
                    prev_chapter_title=outline["chapters"][idx - 1]["title"] if idx > 0 else None,
                    next_chapter_title=outline["chapters"][idx + 1]["title"] if idx < len(outline["chapters"]) - 1 else None,
                    file_context=ch_file_context,
                    search_mode=ch_search_mode,
                    web_sources=ch_web_sources,
                    use_search=ch_use_search,
                ))

            for future in asyncio.as_completed(tasks):
                try:
                    chapter = await future
                    self.storage.save_chapter(course_id, chapter)
                    chapters_generated += 1
                    yield {
                        "type": "chapter_ready",
                        "course_id": course_id,
                        "chapter_order": chapter["order"],
                        "chapter_title": chapter["title"],
                        "total_chapters": len(outline["chapters"]),
                        "chapter": chapter,
                    }
                except Exception as e:
                    logger.error(f"Resume stream: chapter generation failed: {e}")
                    chapters_failed.append(batch[0][1]["order"])  # approximate
                    yield {
                        "type": "error",
                        "error": "CHAPTER_GENERATION_FAILED",
                        "message": f"Chapter generation failed: {e}",
                        "status_code": 500,
                        "course_id": course_id,
                        "phase": "chapter_generation",
                        "context": None,
                    }
                    return

            if batch_start + BATCH_SIZE < len(missing_items):
                await asyncio.sleep(1)
    except Exception as e:
        logger.error(f"Resume stream batched generation failed: {e}")
        yield {
            "type": "error",
            "error": "CHAPTER_GENERATION_FAILED",
            "message": f"Batched chapter generation failed: {e}",
            "status_code": 500,
            "course_id": course_id,
            "phase": "chapter_generation",
            "context": None,
        }
        return

    # Generate extras
    sg_generated = False
    fc_generated = False
    if needs_study_guide or needs_flashcards:
        all_chapters = self.storage.get_course(course_id).get("chapters", [])
        try:
            extras_tasks = []
            if needs_study_guide:
                extras_tasks.append(("sg", self._generate_study_guide(
                    course_id, all_chapters, profile, model_config, topic
                )))
            if needs_flashcards:
                extras_tasks.append(("fc", self._generate_course_flashcards(
                    course_id, all_chapters, profile, model_config
                )))
            results = await asyncio.gather(
                *[t for _, t in extras_tasks], return_exceptions=True
            )
            for (label, _), result in zip(extras_tasks, results):
                if isinstance(result, Exception):
                    logger.warning(f"Resume stream: {label} failed (non-fatal): {result}")
                elif result:
                    if label == "sg":
                        sg_generated = True
                        yield {"type": "study_guide_ready", "course_id": course_id}
                    else:
                        fc_generated = True
                        yield {"type": "flashcards_ready", "course_id": course_id}
        except Exception as e:
            logger.warning(f"Resume stream: extras error (non-fatal): {e}")

    # Finalize
    if not chapters_failed:
        course_update = {
            "id": course_id,
            "status": CourseStatus.READY,
            "completed_at": datetime.utcnow(),
        }
        self.storage.save_course(course_update)

    generation_time = round(time.time() - start_time, 2)
    yield {
        "type": "course_complete",
        "course_id": course_id,
        "status": CourseStatus.READY if not chapters_failed else "generating",
        "generation_time_seconds": generation_time,
        "resume_summary": {
            "chapters_total": len(outline["chapters"]),
            "chapters_existed": len(existing_ready),
            "chapters_generated": chapters_generated,
            "chapters_failed": chapters_failed,
            "study_guide_generated": sg_generated,
            "flashcards_generated": fc_generated,
            "already_complete": False,
        },
    }
```

**Step 2: Verify import**

Run: `cd /Users/davak/Documents/nuton_rag && python3 -c "from services.course_service import CourseService; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add services/course_service.py
git commit -m "feat(service): add resume_course_stream for SSE-based resume"
```

---

### Task 3: Add Route Handlers

**Files:**
- Modify: `routes/course_routes.py` (add 2 endpoints after the `/courses/{course_id}` GET endpoint, around line 505)

**Step 1: Add the blocking resume endpoint**

Insert after the `get_course` endpoint (after line 504):

```python
@router.post("/courses/{course_id}/resume")
async def resume_course(course_id: str, model: Optional[str] = None):
    """
    Resume generation of a partially-completed course.

    Detects missing or errored chapters and generates only the gaps.
    Also generates study guide and flashcards if missing.
    Re-entrant: safe to call multiple times.

    Query params:
        model: Optional model override (defaults to course's original model)
    """
    result = await course_service.resume_course(
        course_id=course_id,
        model=model,
    )
    return result
```

**Step 2: Add the streaming resume endpoint**

Insert right after the blocking endpoint:

```python
@router.post("/courses/{course_id}/resume/stream")
async def resume_course_stream(course_id: str, model: Optional[str] = None):
    """
    SSE streaming version of course resume.

    Emits events:
    - resume_started: gap analysis results
    - chapter_ready: each generated chapter
    - study_guide_ready / flashcards_ready: extras
    - course_complete: final summary with resume_summary
    - error: on failure
    """
    # Pre-stream validation: check course exists and has outline
    course = course_service.get_course(course_id)
    if not course:
        raise NotFoundError(
            "Course not found",
            error_code="COURSE_NOT_FOUND",
            context={"course_id": course_id},
        )
    if not course.get("outline"):
        raise ValidationError(
            "Course has no outline — cannot resume. Please regenerate the course.",
            error_code="NO_OUTLINE",
            context={"course_id": course_id},
        )

    async def event_generator():
        try:
            async for event in course_service.resume_course_stream(
                course_id=course_id,
                model=model,
            ):
                yield _sse_event(event)
                if event.get("type") == "error":
                    return
        except Exception as e:
            logger.error(f"Resume SSE stream error: {e}")
            yield _sse_error_event(e, phase="resume")

    return StreamingResponse(event_generator(), media_type="text/event-stream")
```

**Step 3: Verify the `get_course` method is available on `course_service`**

The route calls `course_service.get_course(course_id)` for pre-stream validation. Check that `CourseService` has a `get_course` method. If not, it's available via `course_service.storage.get_course(course_id)` — adjust accordingly.

Run: `cd /Users/davak/Documents/nuton_rag && python3 -c "from services.course_service import CourseService; cs = CourseService(); print(hasattr(cs, 'get_course'))"`

If `False`, use `course_service.storage.get_course(course_id)` instead, or check if there's a wrapper method.

**Step 4: Verify the server starts**

Run: `cd /Users/davak/Documents/nuton_rag && timeout 5 python3 -c "from routes.course_routes import router; print('Routes OK')" || true`
Expected: `Routes OK`

**Step 5: Commit**

```bash
git add routes/course_routes.py
git commit -m "feat(routes): add POST /courses/{course_id}/resume and /resume/stream endpoints"
```

---

### Task 4: Verify Pinecone Client Integration

**Files:**
- Read: `clients/pinecone_client.py` (verify `get_pinecone_index` exists and signature)

**Step 1: Verify Pinecone client has the function we need**

Run: `cd /Users/davak/Documents/nuton_rag && grep -n "def get_pinecone_index\|def query" clients/pinecone_client.py`

If `get_pinecone_index` doesn't exist, check what's available (e.g., `get_index`, `pinecone_index`, etc.) and update the import in `_retrieve_chunks_for_resume` accordingly.

**Step 2: Verify Pinecone namespace and dimension match**

Check the existing upsert code to confirm dimension (1024) and namespace:

Run: `cd /Users/davak/Documents/nuton_rag && grep -n "namespace\|1024\|dimension" clients/pinecone_client.py services/course_service.py | head -20`

Update `_retrieve_chunks_for_resume` if dimension or namespace differs.

**Step 3: Verify metadata fields stored in Pinecone**

Check what metadata is stored when chunks are upserted (look for `metadata` dict construction in the embeddings code):

Run: `cd /Users/davak/Documents/nuton_rag && grep -A 5 "metadata" services/course_service.py | grep -i "text\|chunk_index\|filename\|course_id" | head -10`

Update the metadata field names in `_retrieve_chunks_for_resume` if they differ from `text`, `chunk_index`, `filename`.

**Step 4: Commit any fixes**

```bash
git add services/course_service.py
git commit -m "fix(service): align Pinecone integration in resume chunk retrieval"
```

---

### Task 5: End-to-End Smoke Test

**Step 1: Start the server**

Run: `cd /Users/davak/Documents/nuton_rag && python3 -m uvicorn main:app --host 0.0.0.0 --port 8000 &`

**Step 2: Test with a completed course (idempotent case)**

Find an existing course ID from the database:

Run: `curl -s http://localhost:8000/api/v1/courses/<COURSE_ID> | python3 -m json.tool | head -5`

Then test resume:

Run: `curl -s -X POST "http://localhost:8000/api/v1/courses/<COURSE_ID>/resume" | python3 -m json.tool`

Expected: Response with `"already_complete": true` in `resume_summary`.

**Step 3: Test streaming endpoint**

Run: `curl -s -N -X POST "http://localhost:8000/api/v1/courses/<COURSE_ID>/resume/stream"`

Expected: SSE event with `course_complete` and `already_complete: true`.

**Step 4: Test error cases**

Run: `curl -s -X POST "http://localhost:8000/api/v1/courses/00000000-0000-0000-0000-000000000000/resume" | python3 -m json.tool`

Expected: 404 error response.

**Step 5: Kill server**

Run: `kill %1`

**Step 6: Commit if any fixes needed**

```bash
git add -A
git commit -m "fix: address issues found during resume endpoint smoke test"
```

---
