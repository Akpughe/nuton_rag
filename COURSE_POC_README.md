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
9. **Course Q&A**: Ask questions about course content per chapter
10. **SSE Streaming**: Real-time progress via Server-Sent Events (outline → chapters → complete)

### File Structure
```
nuton_rag/
├── models/course_models.py           # Pydantic schemas
├── services/course_service.py        # Core generation logic
├── routes/course_routes.py           # FastAPI endpoints
├── clients/supabase_client.py        # Supabase DB operations
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

### Course Q&A
- `POST /api/v1/courses/{course_id}/ask` - Ask questions about course content

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

data: {"type": "course_complete", "course_id": "uuid", "generation_time_seconds": 52}
```

Optional source parameters (can be combined with topic):
```bash
curl -N -X POST http://localhost:8000/api/v1/courses/from-topic/stream \
  -F "user_id=user-123" \
  -F "topic=quantum computing basics" \
  -F "files=@notes.pdf" \
  -F "youtube_urls=https://youtube.com/watch?v=abc123" \
  -F "web_urls=https://example.com/article"
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

### 8. Ask a Question about a Course
```bash
curl -X POST http://localhost:8000/api/v1/courses/{course_id}/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Explain quantum entanglement in simpler terms",
    "chapter_order": 2,
    "user_id": "user-123"
  }'
```

## Key Features

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
| `courses` | Course metadata, outline, personalization params |
| `chapters` | Full chapter content, quiz, sources, key concepts |
| `course_progress` | Per-chapter completion status and time tracking |
| `course_quiz_attempts` | Individual quiz attempt scores and answers |

Generation logs are still stored locally in `course_generation_logs.json` for debugging.

## Next Steps

### To Run:
1. Ensure all dependencies installed: `pip install -r requirements.txt`
2. Set Supabase credentials in `.env` (`SUPABASE_URL`, `SUPABASE_KEY`)
3. Apply migration: `supabase db push` (or run the SQL in `supabase/migrations/`)
4. Start server: `uvicorn pipeline:app --reload`
5. Test endpoints using examples above

### To Test:
1. Save a learning profile
2. Generate a course from topic
3. Verify course and chapters appear in Supabase tables
4. Retrieve course and verify chapters

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
- Supabase for persistent storage (5 normalized tables)
- Synchronous generation (45-60s is acceptable for POC)
- Clear separation: models -> service -> routes
- Simple model switching via config dict
"""
