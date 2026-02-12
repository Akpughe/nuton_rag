"""
Course Generation POC - Quick Start Guide
=========================================

## What's Been Built

Complete POC for AI-powered course generation with the following features:

### Core Features (All Complete)
1. **Topic-to-Course**: Generate course from text topic
2. **File-to-Course**: Generate from PDF/PPT uploads with OCR
3. **Multi-File Support**: Auto-detect topic relationships, recommend organization
4. **5-Question Personalization**: Format, depth, role, goal, examples
5. **Model Switching**: Claude Haiku 4.5, Claude Sonnet 4.5, GPT-4o, GPT-5-mini, Llama-4 Scout/Maverick
6. **Progress Tracking**: Chapter completion, quiz scores, time tracking
7. **Source Citations**: Claude native search + Perplexity fallback
8. **Chapter-Specific Quizzes**: 3-5 questions with explanations
9. **Course Q&A + Chat History**: Ask questions with persistent per-course conversation (Redis cache + Supabase)
10. **SSE Streaming**: Real-time progress via Server-Sent Events (outline -> chapters -> study_guide -> flashcards -> complete)
11. **Study Guide**: Auto-generated comprehensive study aid (core concepts, key terms, comparisons, common mistakes)
12. **Flashcards**: Auto-generated flashcards covering all chapters (5-8 per chapter)
13. **Final Exam**: On-demand exam generation (MCQ + fill-in-the-gap + theory, configurable 30 or 50 questions)
15. **Exam Grading**: Submit answers for automated grading — instant MCQ/fill-in-gap, LLM-based theory with per-rubric-point breakdown, persistent attempt history
14. **Parallel Document Processing**: Files, YouTube URLs, and web URLs processed concurrently with semaphores
16. **Course Notes Generation**: On-demand comprehensive study notes from source materials (space-based or file-based), stored in `courses.summary_md`. Auto-updates course title/topic/slug from document analysis, generates study guide.
17. **Notes Flashcards**: On-demand spaced-repetition flashcards from source materials. Text-based LLM output parsed into JSON. Format mix: cloze, application, compare/contrast, cause-effect, reversal. Saved to both `courses.flashcards` JSONB and `flashcard_sets` table.
18. **Notes Quiz**: On-demand comprehensive quiz from source materials. Bloom's taxonomy distribution, MCQ + fill-in-gap + scenario question types. Saved to `quiz_sets` table.

### File Structure
```
nuton_rag/
├── models/course_models.py           # Pydantic schemas
├── services/course_service.py        # Core generation logic
├── routes/course_routes.py           # FastAPI endpoints
├── clients/supabase_client.py        # Supabase DB operations
├── clients/redis_client.py           # Async Redis client for chat caching
├── utils/file_storage.py             # Storage layer (Supabase-backed)
├── utils/model_config.py             # Model switching
├── prompts/course_prompts.py         # All prompts
├── supabase/migrations/              # DB schema migrations
└── pipeline.py                       # Main app (updated with routes)
```

## API Endpoints

### Learning Profile
- `POST /api/v1/learning-profile` - Save 5-question preferences
- `GET /api/v1/learning-profile/{user_id}` - Retrieve profile

### Course Generation
- `POST /api/v1/courses/from-topic` - Generate from topic (45-60s)
- `POST /api/v1/courses/from-files` - Generate from PDF/PPT

### Course Generation (SSE Streaming)
- `POST /api/v1/courses/from-topic/stream` - Stream course generation from topic via SSE
- `POST /api/v1/courses/from-files/stream` - Stream course generation from files via SSE

### Course Access
- `GET /api/v1/courses/{course_id}` - Full course with chapters
- `GET /api/v1/courses/by-slug/{slug}` - Full course by URL-friendly slug
- `GET /api/v1/courses/{course_id}/chapters/{chapter_order}` - Single chapter
- `GET /api/v1/users/{user_id}/courses` - List user's courses (includes slug)

### Course Q&A + Chat History
- `POST /api/v1/courses/{course_id}/ask` - Ask questions (with optional `user_id` for persistent chat)
- `GET /api/v1/courses/{course_id}/chat-history?user_id=X` - Get chat history
- `DELETE /api/v1/courses/{course_id}/chat-history?user_id=X` - Clear chat history

### Course Notes (on-demand from source materials)
- `POST /api/v1/courses/{course_id}/generate-notes` - Generate comprehensive notes from source materials (30-120s)
- `GET /api/v1/courses/{course_id}/notes` - Get stored notes (markdown)

### Notes Flashcards & Quiz (on-demand from source materials)
- `POST /api/v1/courses/{course_id}/generate-notes-flashcards` - Generate spaced-repetition flashcards (incrementally saved per section)
- `GET /api/v1/courses/{course_id}/notes-flashcards` - Get flashcards (partial during generation, full after)
- `POST /api/v1/courses/{course_id}/generate-notes-quiz` - Generate Bloom's taxonomy quiz (incrementally saved per section)
- `GET /api/v1/courses/{course_id}/notes-quiz` - Get quiz questions (partial during generation, full after)

### Study Guide & Flashcards (auto-generated during course creation)
- `GET /api/v1/courses/{course_id}/study-guide` - Get study guide
- `GET /api/v1/courses/{course_id}/flashcards` - Get flashcards

### Final Exam (on-demand)
- `POST /api/v1/courses/{course_id}/generate-exam` - Generate exam (30 or 50 questions)
- `GET /api/v1/courses/{course_id}/exam` - Get existing exam
- `POST /api/v1/courses/{course_id}/exam/{exam_id}/submit` - Submit answers for grading
- `GET /api/v1/courses/{course_id}/exam/{exam_id}/attempts` - List graded attempts
- `GET /api/v1/courses/{course_id}/exam/{exam_id}/attempts/{attempt_id}` - Get full attempt detail

### Progress
- `POST /api/v1/courses/{course_id}/progress` - Update completion
- `GET /api/v1/users/{user_id}/progress` - All progress stats

### Models
- `GET /api/v1/models` - List available models

## Example Requests

### 1. Save Learning Profile
```bash
curl -X POST http://localhost:8000/api/v1/learning-profile \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user-123",
    "format_pref": "reading",
    "depth_pref": "detailed",
    "role": "student",
    "learning_goal": "exams",
    "example_pref": "real_world"
  }'
```

### 2. Generate Course from Topic
```bash
curl -X POST http://localhost:8000/api/v1/courses/from-topic \
  -F "user_id=user-123" \
  -F "topic=quantum computing basics" \
  -F "model=claude-haiku-4-5"
```

Optional source parameters (can be combined with topic):
```bash
curl -X POST http://localhost:8000/api/v1/courses/from-topic \
  -F "user_id=user-123" \
  -F "topic=quantum computing basics" \
  -F "model=claude-haiku-4-5" \
  -F "files=@notes.pdf" \
  -F 'youtube_urls=["https://youtube.com/watch?v=abc123"]' \
  -F 'web_urls=["https://example.com/article"]'
```

**Response** (45-60s):
```json
{
  "course_id": "uuid",
  "status": "ready",
  "course": {
    "title": "Understanding Quantum Computing",
    "description": "...",
    "total_chapters": 4,
    "chapters": [...]
  },
  "generation_time_seconds": 52
}
```

### 3. Generate from Files
```bash
curl -X POST http://localhost:8000/api/v1/courses/from-files \
  -F "files=@physics_lecture.pdf" \
  -F "files=@chemistry_notes.pdf" \
  -F "user_id=user-123" \
  -F "organization=auto" \
  -F "model=claude-haiku-4-5"
```

With YouTube and web URL sources (at least one source required):
```bash
curl -X POST http://localhost:8000/api/v1/courses/from-files \
  -F "files=@physics_lecture.pdf" \
  -F "user_id=user-123" \
  -F "organization=auto" \
  -F "model=claude-haiku-4-5" \
  -F 'youtube_urls=["https://youtube.com/watch?v=abc123"]' \
  -F 'web_urls=["https://example.com/article"]'
```

### 4. Get Course
```bash
curl http://localhost:8000/api/v1/courses/{course_id}

# Or by slug (URL-friendly identifier auto-generated from title)
curl http://localhost:8000/api/v1/courses/by-slug/understanding-quantum-computing
```

### 5. Update Progress
```bash
curl -X POST http://localhost:8000/api/v1/courses/{course_id}/progress \
  -H "Content-Type: application/json" \
  -d '{
    "chapter_id": "chapter-uuid",
    "completed": true,
    "quiz_score": 85,
    "time_spent_minutes": 20
  }'
```

### 6. Stream Course from Topic (SSE)
```bash
curl -N -X POST http://localhost:8000/api/v1/courses/from-topic/stream \
  -F "user_id=user-123" \
  -F "topic=quantum computing basics" \
  -F "model=claude-haiku-4-5"
```

**SSE Events** (streamed progressively):
```
data: {"type": "processing_sources", "message": "Processing 2 source(s)...", "source_count": 2}

data: {"type": "outline_ready", "course_id": "uuid", "title": "...", "chapters": [...]}

data: {"type": "chapter_ready", "chapter_order": 1, "title": "...", "content": "..."}

data: {"type": "chapter_ready", "chapter_order": 2, "title": "...", "content": "..."}

data: {"type": "study_guide_ready", "course_id": "uuid"}

data: {"type": "flashcards_ready", "course_id": "uuid"}

data: {"type": "course_complete", "course_id": "uuid", "generation_time_seconds": 52}
```

### 7. Stream Course from Files (SSE)
```bash
curl -N -X POST http://localhost:8000/api/v1/courses/from-files/stream \
  -F "files=@physics_lecture.pdf" \
  -F "files=@chemistry_notes.pdf" \
  -F "user_id=user-123" \
  -F "organization=auto" \
  -F "model=claude-haiku-4-5"
```

### 8. Ask a Question (with Chat History)
```bash
curl -X POST http://localhost:8000/api/v1/courses/{course_id}/ask \
  -F "question=Explain quantum entanglement in simpler terms" \
  -F "user_id=user-123" \
  -F "model=claude-haiku-4-5"
```

Subsequent questions with the same `user_id` will include prior conversation context.

### 9. Get Chat History
```bash
curl http://localhost:8000/api/v1/courses/{course_id}/chat-history?user_id=user-123&limit=20
```

### 10. Clear Chat History
```bash
curl -X DELETE http://localhost:8000/api/v1/courses/{course_id}/chat-history?user_id=user-123
```

### 11. Get Study Guide
```bash
curl http://localhost:8000/api/v1/courses/{course_id}/study-guide
```

**Response**:
```json
{
  "course_id": "uuid",
  "study_guide": {
    "core_concepts": [{"concept": "...", "definition": "...", "chapter": "...", "why_it_matters": "..."}],
    "must_remember": [{"topic": "...", "points": ["..."]}],
    "key_terms": [{"term": "...", "definition": "...", "used_in_context": "...", "dont_confuse_with": "..."}],
    "key_comparisons": [{"items": ["X", "Y"], "similarities": [...], "differences": [...]}],
    "common_mistakes": [{"mistake": "...", "why_wrong": "...", "instead": "..."}],
    "timeline": null,
    "processes": null,
    "formulas": null
  }
}
```

### 12. Get Flashcards
```bash
curl http://localhost:8000/api/v1/courses/{course_id}/flashcards
```

**Response**:
```json
{
  "course_id": "uuid",
  "total": 24,
  "flashcards": [
    {
      "id": "fc_1",
      "front": "What is quantum superposition?",
      "back": "The ability of a quantum system to exist in multiple states simultaneously...",
      "hint": "Think about Schrodinger's cat",
      "chapter_ref": "Chapter Title",
      "concept": "Superposition",
      "difficulty": "basic"
    }
  ]
}
```

### 13. Generate Final Exam
```bash
curl -X POST http://localhost:8000/api/v1/courses/{course_id}/generate-exam \
  -F "user_id=user-123" \
  -F "exam_size=30" \
  -F "model=claude-haiku-4-5"
```

**Response**:
```json
{
  "id": "exam-uuid",
  "course_id": "uuid",
  "exam_size": 30,
  "total_questions": 30,
  "mcq": [
    {"question": "...", "options": ["A", "B", "C", "D"], "correct_answer": 0, "explanation": "...", "chapter_ref": "...", "difficulty": "medium"}
  ],
  "fill_in_gap": [
    {"sentence_with_gap": "The process of _____ ...", "correct_answer": "...", "alternatives": [...], "explanation": "...", "chapter_ref": "...", "difficulty": "easy"}
  ],
  "theory": [
    {"question": "Explain how...", "model_answer": "...", "rubric": ["Key point 1", "Key point 2"], "chapter_ref": "...", "difficulty": "hard"}
  ]
}
```

Exam sizes: `30` (15 MCQ / 8 fill-in / 7 theory) or `50` (25 MCQ / 15 fill-in / 10 theory)

### 14. Get Existing Exam
```bash
curl http://localhost:8000/api/v1/courses/{course_id}/exam?user_id=user-123
```

### 15. Submit Exam Answers for Grading
```bash
curl -X POST http://localhost:8000/api/v1/courses/{course_id}/exam/{exam_id}/submit \
  -F "user_id=user-123" \
  -F 'answers={"mcq":[2,0,3,1,0],"fill_in_gap":["data transformation","neural network"],"theory":["Theory explains...","The key difference is..."]}' \
  -F "model=claude-haiku-4-5" \
  -F "time_taken_seconds=1800"
```

**Response**:
```json
{
  "attempt_id": "uuid",
  "score": 72.5,
  "mcq_score": 80.0,
  "fill_in_gap_score": 75.0,
  "theory_score": 60.0,
  "results": {
    "mcq": [{"question_index": 0, "selected": 2, "correct": 0, "is_correct": false, "explanation": "..."}],
    "fill_in_gap": [{"question_index": 0, "answer": "data transformation", "correct_answer": "data transformation", "is_correct": true, "match_type": "exact"}],
    "theory": [{"question_index": 0, "rubric_breakdown": [{"point": "...", "status": "covered", "feedback": "..."}], "score": 7, "max_score": 10, "feedback": "..."}]
  }
}
```

Scoring: MCQ 40%, Fill-in-gap 25%, Theory 35%. Theory uses LLM-based per-rubric-point grading (covered/partial/missed).

### 16. Get Exam Attempts
```bash
curl "http://localhost:8000/api/v1/courses/{course_id}/exam/{exam_id}/attempts?user_id=user-123"
```

### 17. Get Attempt Detail (with rubric breakdowns)
```bash
curl http://localhost:8000/api/v1/courses/{course_id}/exam/{exam_id}/attempts/{attempt_id}
```

### 18. Generate Course Notes
```bash
curl -X POST http://localhost:8000/api/v1/courses/{course_id}/generate-notes \
  -F "user_id=user-123" \
  -F "model=claude-haiku-4-5"
```

**Response** (30-120s depending on source material size):
```json
{
  "course_id": "uuid",
  "title": "Updated Title from Document Analysis",
  "slug": "updated-title-from-document-analysis",
  "topic": "Main Topic",
  "notes_length": 15000,
  "sections_generated": 8,
  "model_used": "claude-haiku-4-5",
  "has_study_guide": true,
  "generation_time_seconds": 65.2,
  "summary_md": "# Full Markdown Notes\n\n## Section 1: ...\n\n..."
}
```

Notes generation flow:
1. Fetches all source chunks from Pinecone (space-based or file-based courses)
2. Builds a document map (topic-to-chunk mapping)
3. Updates course title, topic, and slug from document analysis
4. Generates intro + sections (batched parallel, 4 at a time) + conclusion with sources
5. Saves to `courses.summary_md` column
6. Generates a study guide from the notes content

### 19. Get Course Notes
```bash
curl http://localhost:8000/api/v1/courses/{course_id}/notes
```

**Response**:
```json
{
  "course_id": "uuid",
  "title": "Course Title",
  "notes_length": 15000,
  "summary_md": "# Full Markdown Notes\n\n## Table of Contents\n..."
}
```

### 20. Generate Notes Flashcards
```bash
curl -X POST http://localhost:8000/api/v1/courses/{course_id}/generate-notes-flashcards \
  -F "user_id=user-123" \
  -F "model=claude-haiku-4-5"
```

**Response**:
```json
{
  "course_id": "uuid",
  "flashcard_count": 42,
  "model_used": "claude-haiku-4-5",
  "generation_time_seconds": 35.1
}
```

Flashcard format mix: cloze (~40%), application (~25%), compare/contrast (~15%), cause-effect (~10%), reversal (~10%). Each card includes front, back, hint, concept, difficulty, and section reference. Saved to both `courses.flashcards` JSONB and `flashcard_sets` table.

### 21. Generate Notes Quiz
```bash
curl -X POST http://localhost:8000/api/v1/courses/{course_id}/generate-notes-quiz \
  -F "user_id=user-123" \
  -F "model=claude-haiku-4-5"
```

**Response**:
```json
{
  "course_id": "uuid",
  "total_questions": 28,
  "by_type": {"mcq": 14, "fill_in_gap": 7, "scenario": 7},
  "by_bloom": {"remember": 4, "understand": 7, "apply": 9, "analyze": 6, "evaluate": 2},
  "by_difficulty": {"easy": 8, "medium": 14, "hard": 6},
  "model_used": "claude-haiku-4-5",
  "generation_time_seconds": 40.3
}
```

Quiz uses Bloom's taxonomy distribution calibrated to learner expertise. Question types: MCQ (~50%), fill-in-gap (~25%), scenario-based (~25%). Saved to `quiz_sets` table.

### 22. Get Notes Flashcards (incremental)
```bash
curl http://localhost:8000/api/v1/courses/{course_id}/notes-flashcards
```

**Response** (partial results appear during generation, full set after):
```json
{
  "course_id": "uuid",
  "total": 42,
  "sets_count": 8,
  "flashcards": [
    {"front": "...", "back": "...", "hint": "...", "concept": "...", "difficulty": "medium", "section_ref": "Section Title"}
  ]
}
```

### 23. Get Notes Quiz (incremental)
```bash
curl http://localhost:8000/api/v1/courses/{course_id}/notes-quiz
```

**Response** (partial results appear during generation, full set after):
```json
{
  "course_id": "uuid",
  "total_questions": 28,
  "sets_count": 8,
  "by_type": {"mcq": 14, "fill_in_gap": 7, "scenario": 7},
  "quiz": {
    "mcq": [...],
    "fill_in_gap": [...],
    "scenario": [...]
  }
}
```

## Key Features

### Course Notes Generation (On-Demand)
Generates comprehensive study notes from course source materials stored in Pinecone:
- **Space-based courses**: Fetches chunks from all documents in the space (PDFs + YouTube), tags vectors with `course_id`
- **File-based courses**: Fetches chunks stored during course creation with `document_id=course_id`
- **Document map**: Builds structured topic-to-chunk mapping, then generates notes section-by-section
- **3-phase generation**: Intro (with TOC) + sections (batched parallel, 4 at a time) + conclusion (with sources)
- **Personalized**: Full learner profile integration (12 role-goal combos, expertise/depth/example/format modifiers)
- **Auto-updates**: Course title, topic, and slug derived from document analysis
- **Study guide**: Automatically generated from the notes content

### Notes Flashcards (On-Demand from Source Materials)
Generates spaced-repetition optimized flashcards directly from course source materials:
- **Incremental insertion**: Each section's flashcards are saved to `flashcard_sets` immediately as they complete — poll `GET /notes-flashcards` to see partial progress
- **Format mix**: Cloze (~40%), application (~25%), compare/contrast (~15%), cause-effect (~10%), reversal (~10%)
- **Anti-patterns enforced**: No "What is X?" definitional cards, no yes/no, no enumeration/listing cards
- **Per-card fields**: Front, back, hint (activates recall without giving away answer), concept, difficulty, type
- **Dual storage**: `courses.flashcards` JSONB column (full set at end) + `flashcard_sets` table (one row per section, incremental)
- **Text-to-JSON parsing**: LLM generates structured text, parsed via regex into JSON (more reliable than direct JSON generation)
- **Re-generation safe**: Previous sets are deleted before new generation starts

### Notes Quiz (On-Demand from Source Materials)
Generates comprehensive assessment quizzes using Bloom's cognitive taxonomy:
- **Incremental insertion**: Each section's quiz is saved to `quiz_sets` immediately as it completes — poll `GET /notes-quiz` to see partial progress
- **Bloom's distribution** (calibrated to expertise): Remember 15%, Understand 25%, Apply 30%, Analyze 20%, Evaluate 10%
- **Question types**: MCQ (~50%), fill-in-gap (~25%), scenario-based (~25%)
- **Cross-section interleaving**: Questions reference connections between topics for deeper learning
- **MCQ quality**: All distractors test common misconceptions, explanations cover every option
- **Saved to** `quiz_sets` table with one row per section (incremental) and full metadata (by_bloom, by_type, by_difficulty breakdowns)
- **Re-generation safe**: Previous sets are deleted before new generation starts

### Parallel Document Processing
Files, YouTube URLs, and web URLs are processed concurrently using `asyncio.gather` with semaphores:
- Files: Semaphore(3) for OCR rate limiting
- YouTube: Semaphore(4) for transcript extraction
- Web: Semaphore(4) for content extraction
- All three source types also run concurrently with each other

### Study Guide + Flashcards (Auto-Generated)
After all chapters complete, study guide and flashcards are generated **in parallel** via `asyncio.gather`:
- **Study Guide**: Core concepts, must remember, key terms, key comparisons, common mistakes. Conditional sections (timeline, processes, formulas) included when relevant.
- **Flashcards**: 5-8 per chapter covering definitions, concepts, application scenarios, and misconceptions. Mix of basic/intermediate/advanced difficulty.

### Final Exam (On-Demand)
Generated with 3 parallel LLM calls (one per question type):
- **MCQ**: 4 options, explanations for all options
- **Fill-in-the-gap**: Sentence with blank, correct answer + alternatives
- **Theory**: Open-ended, model answer + grading rubric
- Difficulty mix: 30% easy, 50% medium, 20% hard
- Questions distributed proportionally across all chapters

### Chat History
Persistent per-course, per-user conversation:
- **Redis cache** (24h TTL) for fast access to last 20 messages
- **Supabase fallback** when Redis is unavailable
- Subsequent questions include conversation context for coherent follow-ups
- Clear chat history via DELETE endpoint

### Multi-File Organization
Automatically detects relationship between uploaded files:
- **Thematic Bridge** (>0.75 similarity): Topics are closely related
- **Sequential Sections** (0.4-0.75): Topics are distinct but complementary
- **Separate Courses** (<0.4): Topics are unrelated

### Model Configuration
```
Default: claude-haiku-4-5
Available models:
- claude-haiku-4-5: Fastest Claude, web search support
- claude-sonnet-4-5: Balanced Claude, web search support
- claude-opus-4-6: Most capable Claude, web search support
- gpt-4o: OpenAI, web search support
- gpt-5-mini: OpenAI, web search support
- gpt-5.2: OpenAI, web search support
- llama-4-scout: Groq, Perplexity search fallback
- llama-4-maverick: Groq, Perplexity search fallback
```

### Cost Tracking
Each generation is logged with estimated cost:
- Outline: $0.02
- Each chapter (1000 words): $0.02
- Search: $0.01

### Storage
All data stored in Supabase (PostgreSQL):

| Table | Purpose |
|-------|---------|
| `learning_profiles` | User learning preferences (format, depth, role, goal, examples) |
| `courses` | Course metadata, outline, personalization params, study_guide (jsonb), flashcards (jsonb), summary_md (text) |
| `chapters` | Full chapter content, quiz, sources, key concepts |
| `course_progress` | Per-chapter completion status and time tracking |
| `course_quiz_attempts` | Individual quiz attempt scores and answers |
| `course_exams` | On-demand final exams (MCQ + fill-in-gap + theory) |
| `course_chat_messages` | Persistent chat history per course per user |
| `course_exam_attempts` | Graded exam submissions with scores and rubric breakdowns |
| `flashcard_sets` | Standalone flashcard sets (course-based via `course_id` or content-based via `content_id`) |
| `quiz_sets` | Standalone quiz sets with Bloom's taxonomy metadata |

Generation logs are still stored locally in `course_generation_logs.json` for debugging.

## Next Steps

### To Run:
1. Ensure all dependencies installed: `pip install -r requirements.txt`
2. Set Supabase credentials in `.env` (`SUPABASE_URL`, `SUPABASE_KEY`)
3. Optional: Set `REDIS_URL` in `.env` for chat caching (defaults to `redis://localhost:6379`, works without Redis)
4. Apply migration: `supabase db push` (or run the SQL in `supabase/migrations/`)
5. Start server: `uvicorn pipeline:app --reload`
6. Test endpoints using examples above

### To Test:
1. Save a learning profile
2. Generate a course from topic
3. Verify course and chapters appear in Supabase tables
4. Check study guide and flashcards are populated after generation
5. Generate a final exam and verify question counts
6. Ask 3+ questions with same user_id and verify conversation context

### To Extend:
1. Add audio generation endpoint (separate from text)
2. Add course sharing with public links
3. Add recommendation engine using Pinecone

## Architecture Principles Applied

### DRY (Don't Repeat Yourself)
- All prompts in `course_prompts.py`
- All models in `course_models.py`
- All storage in `file_storage.py` (backed by Supabase)
- Single `CourseService` handles all generation

### KISS (Keep It Simple, Stupid)
- Supabase for persistent storage (7 normalized tables)
- Redis for optional caching with graceful fallback
- Clear separation: models -> service -> routes
- Simple model switching via config dict
"""
