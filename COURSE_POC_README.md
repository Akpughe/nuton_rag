"""
Course Generation POC - Quick Start Guide
=========================================

## What's Been Built

✅ Complete POC for AI-powered course generation with the following features:

### Core Features (All Complete)
1. **Topic-to-Course**: Generate course from text topic
2. **File-to-Course**: Generate from PDF/PPT uploads with OCR
3. **Multi-File Support**: Auto-detect topic relationships, recommend organization
4. **5-Question Personalization**: Format, depth, role, goal, examples
5. **Model Switching**: Claude Sonnet 4, GPT-4o, Llama-4 Scout
6. **Progress Tracking**: Chapter completion, quiz scores, time tracking
7. **Source Citations**: Claude native search + Perplexity fallback
8. **Chapter-Specific Quizzes**: 3-5 questions with explanations

### File Structure
```
nuton_rag/
├── courses/                          # Course storage (JSON files)
├── models/course_models.py           # Pydantic schemas (DRY)
├── services/course_service.py        # Core generation logic
├── routes/course_routes.py           # FastAPI endpoints
├── utils/file_storage.py             # JSON storage layer
├── utils/model_config.py             # Model switching
├── prompts/course_prompts.py         # All prompts (KISS)
└── pipeline.py                       # Main app (updated with routes)
```

## API Endpoints

### Learning Profile
- `POST /api/v1/learning-profile` - Save 5-question preferences
- `GET /api/v1/learning-profile/{user_id}` - Retrieve profile

### Course Generation
- `POST /api/v1/courses/from-topic` - Generate from topic (45-60s)
- `POST /api/v1/courses/from-files` - Generate from PDF/PPT

### Course Access
- `GET /api/v1/courses/{course_id}` - Full course with chapters
- `GET /api/v1/courses/{course_id}/chapters/{chapter_order}` - Single chapter
- `GET /api/v1/users/{user_id}/courses` - List user's courses

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
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user-123",
    "topic": "quantum computing basics",
    "context": {
      "expertise": "beginner",
      "time_available": 60
    },
    "model": "claude-sonnet-4"
  }'
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
  "storage_path": "courses/course_uuid/",
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
  -F "model=claude-sonnet-4"
```

### 4. Get Course
```bash
curl http://localhost:8000/api/v1/courses/{course_id}
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

## Key Features

### Multi-File Organization
Automatically detects relationship between uploaded files:
- **Thematic Bridge** (>0.75 similarity): Topics are closely related
- **Sequential Sections** (0.4-0.75): Topics are distinct but complementary
- **Separate Courses** (<0.4): Topics are unrelated

### Model Configuration
```python
# Default: claude-sonnet-4
# Available models:
- claude-sonnet-4: $0.11/course, web search support
- gpt-4o: $0.17/course, no web search
- llama-4-scout: $0.05/course, fastest
```

### Cost Tracking
Each generation is logged with estimated cost:
- Outline: $0.02
- Each chapter (1000 words): $0.02
- Search: $0.01

### Storage
All data stored in JSON files:
- `learning_profiles.json` - User preferences
- `courses/course_{uuid}/` - Course content
- `courses/index.json` - Course catalog
- `course_generation_logs.json` - Generation history

## Next Steps

### To Run:
1. Ensure all dependencies installed: `pip install -r requirements.txt`
2. Start server: `uvicorn pipeline:app --reload`
3. Test endpoints using examples above

### To Test:
1. Save a learning profile
2. Generate a course from topic
3. Check `courses/` directory for generated JSON files
4. Retrieve course and verify chapters

### To Extend:
1. Add audio generation endpoint (separate from text)
2. Add course sharing with public links
3. Add recommendation engine using Pinecone
4. Add database migration when ready

## Architecture Principles Applied

### DRY (Don't Repeat Yourself)
- All prompts in `course_prompts.py`
- All models in `course_models.py`
- All storage in `file_storage.py`
- Single `CourseService` handles all generation

### KISS (Keep It Simple, Stupid)
- JSON file storage (no database complexity)
- Synchronous generation (45-60s is acceptable for POC)
- Clear separation: models → service → routes
- Simple model switching via config dict

## Files Created

1. `models/course_models.py` - 200 lines: All Pydantic models
2. `utils/file_storage.py` - 200 lines: JSON storage layer
3. `utils/model_config.py` - 100 lines: Model switching
4. `prompts/course_prompts.py` - 200 lines: All prompts
5. `services/course_service.py` - 400 lines: Core logic
6. `routes/course_routes.py` - 300 lines: API endpoints
7. `.opencode/plans/POC_DESIGN.md` - 500 lines: Full design doc

**Total: ~1900 lines of production-ready POC code**
