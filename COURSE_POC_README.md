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

## Key Features

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
| `courses` | Course metadata, outline, personalization params, study_guide (jsonb), flashcards (jsonb) |
| `chapters` | Full chapter content, quiz, sources, key concepts |
| `course_progress` | Per-chapter completion status and time tracking |
| `course_quiz_attempts` | Individual quiz attempt scores and answers |
| `course_exams` | On-demand final exams (MCQ + fill-in-gap + theory) |
| `course_chat_messages` | Persistent chat history per course per user |
| `course_exam_attempts` | Graded exam submissions with scores and rubric breakdowns |

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
