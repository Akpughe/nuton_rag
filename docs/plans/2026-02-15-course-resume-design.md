# Course Resume Endpoint Design

## Problem

Course generation can fail mid-way, leaving courses in `status="generating"` with partial chapters. There is no mechanism to complete these courses without regenerating from scratch.

## Solution

Add standalone resume endpoints that detect what's missing and generate only the gaps.

## Endpoints

### `POST /api/v1/courses/{course_id}/resume` (blocking)

- Path param: `course_id` (UUID)
- Optional query/body param: `model` (override original `model_used`)
- Returns: Full course with all chapters + `resume_summary`

### `POST /api/v1/courses/{course_id}/resume/stream` (streaming)

- Same inputs
- SSE events: `resume_started`, `chapter_ready`, `course_complete`

### Validation

- Course must exist (404)
- Course must have an outline (400) — if outline generation failed, user must regenerate
- If course is already complete with all extras, return as-is (idempotent)

## Resume Logic

### Step 1: Gap Analysis

- Fetch course (outline, total_chapters, status, study_guide, flashcards)
- Fetch existing chapters
- Identify gaps: missing chapters (no DB record) or chapters with `status='error'`
- Check if study_guide/flashcards are null

### Step 2: Early Return

- If zero chapter gaps AND study_guide AND flashcards exist → return course with `resume_summary: { already_complete: true }`

### Step 3: File Context Retrieval

- If `source_type` involves files → query Pinecone by `course_id` for stored chunks
- Build lightweight doc map for per-chapter context
- If Pinecone retrieval fails → fall back to generating without file context (log warning)

### Step 4: Generate Missing Chapters

- Reuse `_generate_chapter()` for each gap
- Batched parallel (BATCH_SIZE=4, 1s sleep between batches)
- Each chapter gets outline entry from stored outline
- Upsert to DB as each completes

### Step 5: Generate Missing Extras

- If study_guide is null → generate (using ALL chapters)
- If flashcards are null → generate
- Parallel, non-fatal (same pattern as existing)

### Step 6: Finalize

- Set course `status = "ready"`, `completed_at = now()`
- Return full course with resume_summary

## Response Shape

### Resume Summary

```json
{
  "chapters_total": 10,
  "chapters_existed": 7,
  "chapters_generated": 3,
  "chapters_failed": [],
  "study_guide_generated": true,
  "flashcards_generated": false,
  "already_complete": false
}
```

### Streaming Events

1. `resume_started` — `{"type": "resume_started", "course_id": "...", "missing_chapters": [3, 8, 9], "total_to_generate": 3}`
2. `chapter_ready` — same shape as existing
3. `course_complete` — same as existing, with `resume_summary`

## Error Handling

- Course not found → `NotFoundError` (404)
- No outline → `ValidationError` (400)
- Chapter failure during resume → stops at failure point, persisted chapters remain, user can call `/resume` again
- Pinecone failure → fall back to no file context, log warning
- Resume is re-entrant: call as many times as needed

## Files Modified

1. `routes/course_routes.py` — 2 new route handlers
2. `services/course_service.py` — `resume_course()` + streaming variant + Pinecone chunk retrieval helper
3. `clients/supabase_client.py` — verify `get_chapters_by_course()` returns status, add helper if needed
