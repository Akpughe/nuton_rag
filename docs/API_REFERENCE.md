# Nuton RAG API Reference

Base URL: `http://localhost:8000`

---

## Table of Contents

- [Health Check](#health-check)
- [Course Generation](#course-generation)
  - [Generate Course from Topic](#generate-course-from-topic)
  - [Generate Course from Files](#generate-course-from-files)
  - [Get Course](#get-course)
  - [Get Chapter](#get-chapter)
  - [List User Courses](#list-user-courses)
  - [Ask Course Question (Q&A)](#ask-course-question)
- [Learning Profiles](#learning-profiles)
  - [Create/Update Learning Profile](#createupdate-learning-profile)
  - [Get Learning Profile](#get-learning-profile)
- [Progress Tracking](#progress-tracking)
  - [Update Progress](#update-progress)
  - [Get User Progress](#get-user-progress)
- [Models](#models)
  - [List Available Models](#list-available-models)
- [Document Processing (RAG)](#document-processing-rag)
  - [Process Document](#process-document)
  - [Answer Query](#answer-query)
  - [Answer Query (Streaming)](#answer-query-streaming)
- [YouTube Processing](#youtube-processing)
  - [Process YouTube Videos](#process-youtube-videos)
  - [Extract Transcript (yt-dlp)](#extract-transcript-yt-dlp)
  - [Extract Transcript (youtube-transcript-api)](#extract-transcript-youtube-transcript-api)
  - [Get YouTube Video Info](#get-youtube-video-info)
  - [Get Transcript Languages](#get-transcript-languages)
  - [YouTube Proxy Status](#youtube-proxy-status)
  - [Test Vcyon Transcript](#test-vcyon-transcript)
- [Study Materials](#study-materials)
  - [Generate Flashcards](#generate-flashcards)
  - [Regenerate Flashcards](#regenerate-flashcards)
  - [Generate Quiz](#generate-quiz)
  - [Regenerate Quiz](#regenerate-quiz)
  - [Generate Notes](#generate-notes)
- [Google Drive](#google-drive)
  - [List Drive Files](#list-drive-files)
  - [Import Drive Files](#import-drive-files)

---

## Health Check

### `GET /`

Simple health check endpoint.

**Response:**

```json
{
  "greeting": "Hello!",
  "message": "Welcome to Nuton RAG Pipeline!"
}
```

---

## Course Generation

### Generate Course from Topic

`POST /api/v1/courses/from-topic`

Generate a complete course from a topic string. Optionally attach supplementary files (PDFs, PPTX, etc.) that provide reference material while the topic drives the course structure.

**Content-Type:** `multipart/form-data`

| Field          | Type           | Required | Description                                        |
|----------------|----------------|----------|----------------------------------------------------|
| `user_id`      | string         | Yes      | User identifier                                    |
| `topic`        | string         | Yes      | Topic to generate course about                     |
| `model`        | string         | No       | Model key (default: `claude-haiku-4-5`)            |
| `files`        | file(s)        | No       | Optional supplementary PDF/PPTX/DOCX/TXT/MD files |
| `youtube_urls` | string (JSON)  | No       | JSON array of YouTube URLs                         |
| `web_urls`     | string (JSON)  | No       | JSON array of web page URLs                        |

**Supported file types:** `.pdf`, `.pptx`, `.ppt`, `.docx`, `.doc`, `.txt`, `.md` (max 50MB each, max 10 files)

**Use cases:**

1. **Pure topic** -- generate a course entirely from the model's knowledge + web search
2. **Topic + files** -- topic drives the structure, uploaded files provide supplementary context
3. **Topic + YouTube** -- topic drives the structure, video transcripts provide supplementary context
4. **Topic + web URLs** -- topic drives the structure, web page content provides supplementary context
5. **Mixed** -- combine any of the above sources

#### Example: Pure topic course

```bash
curl -X POST http://localhost:8000/api/v1/courses/from-topic \
  -F "user_id=user-123" \
  -F "topic=Introduction to Machine Learning and Neural Networks" \
  -F "model=claude-haiku-4-5"
```

**Response (201):**

```json
{
  "course_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "status": "ready",
  "course": {
    "id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
    "user_id": "user-123",
    "title": "Mastering Machine Learning and Neural Networks",
    "description": "A comprehensive introduction to machine learning fundamentals, from linear models to deep neural networks, with practical applications.",
    "topic": "Introduction to Machine Learning and Neural Networks",
    "source_type": "topic",
    "source_files": [],
    "multi_file_organization": null,
    "total_chapters": 6,
    "estimated_time": 90,
    "status": "ready",
    "personalization_params": {
      "format_pref": "reading",
      "depth_pref": "detailed",
      "role": "student",
      "learning_goal": "curiosity",
      "example_pref": "real_world"
    },
    "outline": {
      "title": "Mastering Machine Learning and Neural Networks",
      "description": "...",
      "learning_objectives": ["..."],
      "chapters": [
        {
          "order": 1,
          "title": "The Foundations of Machine Learning",
          "objectives": ["Define ML and distinguish supervised, unsupervised, and reinforcement learning"],
          "key_concepts": ["Supervised Learning", "Feature Engineering", "Training Data"],
          "estimated_time": 15,
          "prerequisites": []
        }
      ],
      "total_estimated_time": 90
    },
    "model_used": "claude-haiku-4-5-20251001",
    "created_at": "2026-02-08T12:00:00Z",
    "completed_at": "2026-02-08T12:00:35Z"
  },
  "storage_path": "courses/course_a1b2c3d4-e5f6-7890-abcd-ef1234567890/",
  "generation_time_seconds": 34.52
}
```

#### Example: Topic + YouTube videos

```bash
curl -X POST http://localhost:8000/api/v1/courses/from-topic \
  -F "user_id=user-123" \
  -F "topic=Neural Networks" \
  -F 'youtube_urls=["https://www.youtube.com/watch?v=aircAruvnKk"]'
```

#### Example: Topic + web URLs

```bash
curl -X POST http://localhost:8000/api/v1/courses/from-topic \
  -F "user_id=user-123" \
  -F "topic=Machine Learning" \
  -F 'web_urls=["https://en.wikipedia.org/wiki/Machine_learning"]'
```

#### Example: Topic + supplementary files

```bash
curl -X POST http://localhost:8000/api/v1/courses/from-topic \
  -F "user_id=user-123" \
  -F "topic=Artificial Intelligence" \
  -F "files=@ai_textbook.pdf" \
  -F "files=@ml_notes.pdf" \
  -F "model=claude-haiku-4-5"
```

**Response (201):**

```json
{
  "course_id": "b2c3d4e5-f6a7-8901-bcde-f12345678901",
  "status": "ready",
  "topic": "Artificial Intelligence",
  "supplementary_files": ["ai_textbook.pdf", "ml_notes.pdf"],
  "document_map": {
    "document_title": "Artificial Intelligence Textbook",
    "total_chunks": 48,
    "topics": [
      {
        "topic": "Search Algorithms",
        "description": "BFS, DFS, A* and heuristic search methods",
        "chunk_indices": [0, 1, 2, 3],
        "importance": "core"
      }
    ]
  },
  "total_chunks_processed": 48,
  "course": { "..." },
  "storage_path": "courses/course_b2c3d4e5-f6a7-8901-bcde-f12345678901/",
  "generation_time_seconds": 52.18
}
```

**Error responses:**

| Status | Detail |
|--------|--------|
| 400    | `"Invalid model. Available: [...]"` |
| 400    | `"Maximum 10 files allowed"` |
| 400    | `"Invalid JSON for URLs: ..."` |
| 400    | `"At least 1 source required (file, YouTube URL, or web URL)"` (from-files only) |
| 400    | `"No usable content extracted from provided sources"` |
| 500    | `"Chapter generation failed: ..."` |

---

### Generate Course from Files

`POST /api/v1/courses/from-files`

Generate a course from uploaded documents, YouTube videos, and/or web URLs. The system extracts topics from sources, builds a document map, and generates chapters grounded in the source material.

At least one source is required (file, YouTube URL, or web URL). All source types can be combined.

**Content-Type:** `multipart/form-data`

| Field          | Type            | Required | Default             | Description                              |
|----------------|-----------------|----------|---------------------|------------------------------------------|
| `files`        | file(s)         | No       | --                  | 1-10 PDF/PPTX/DOCX/TXT/MD files         |
| `user_id`      | string          | Yes      | --                  | User identifier                          |
| `organization` | string          | No       | `auto`              | `auto`, `thematic_bridge`, `sequential_sections`, `separate_courses` |
| `model`        | string          | No       | `claude-haiku-4-5`  | Model key                                |
| `youtube_urls` | string (JSON)   | No       | --                  | JSON array of YouTube URLs               |
| `web_urls`     | string (JSON)   | No       | --                  | JSON array of web page URLs              |

#### Example: Single file

```bash
curl -X POST http://localhost:8000/api/v1/courses/from-files \
  -F "user_id=user-123" \
  -F "files=@quantum_physics.pdf" \
  -F "model=claude-haiku-4-5"
```

**Response (201) -- single course:**

```json
{
  "course_id": "c3d4e5f6-a7b8-9012-cdef-123456789012",
  "status": "ready",
  "detected_topics": ["Quantum Physics"],
  "organization_chosen": null,
  "document_map": {
    "document_title": "Introduction to Quantum Physics",
    "total_chunks": 32,
    "topics": [
      {
        "topic": "Wave-Particle Duality",
        "description": "Exploration of light and matter exhibiting both wave and particle properties",
        "chunk_indices": [0, 1, 2, 3, 4],
        "importance": "core"
      },
      {
        "topic": "Schrodinger Equation",
        "description": "The fundamental equation of quantum mechanics",
        "chunk_indices": [5, 6, 7, 8],
        "importance": "core"
      }
    ],
    "coverage_check": "all_chunks_mapped"
  },
  "total_chunks_processed": 32,
  "course": { "..." },
  "storage_path": "courses/course_c3d4e5f6-a7b8-9012-cdef-123456789012/",
  "generation_time_seconds": 48.33
}
```

#### Example: YouTube videos only

```bash
curl -X POST http://localhost:8000/api/v1/courses/from-files \
  -F "user_id=user-123" \
  -F 'youtube_urls=["https://www.youtube.com/watch?v=aircAruvnKk"]'
```

#### Example: Web URLs only

```bash
curl -X POST http://localhost:8000/api/v1/courses/from-files \
  -F "user_id=user-123" \
  -F 'web_urls=["https://en.wikipedia.org/wiki/Machine_learning"]'
```

#### Example: Mixed sources (files + YouTube + web)

```bash
curl -X POST http://localhost:8000/api/v1/courses/from-files \
  -F "user_id=user-123" \
  -F "files=@lecture.pdf" \
  -F 'youtube_urls=["https://youtube.com/watch?v=abc"]' \
  -F 'web_urls=["https://example.com/article"]'
```

#### Example: Multiple files with separate courses

```bash
curl -X POST http://localhost:8000/api/v1/courses/from-files \
  -F "user_id=user-123" \
  -F "files=@biology.pdf" \
  -F "files=@history.pdf" \
  -F "organization=separate_courses"
```

**Response (201) -- separate courses:**

```json
{
  "organization": "separate_courses",
  "total_courses": 2,
  "courses": [
    {
      "id": "d4e5f6a7-b8c9-0123-defa-234567890123",
      "title": "Foundations of Cell Biology",
      "topic": "Cell Biology",
      "status": "ready",
      "total_chapters": 5,
      "estimated_time": 60,
      "storage_path": "courses/course_d4e5f6a7-b8c9-0123-defa-234567890123/"
    },
    {
      "id": "e5f6a7b8-c9d0-1234-efab-345678901234",
      "title": "The Modern World: A Historical Overview",
      "topic": "Modern History",
      "status": "ready",
      "total_chapters": 7,
      "estimated_time": 90,
      "storage_path": "courses/course_e5f6a7b8-c9d0-1234-efab-345678901234/"
    }
  ],
  "generation_time_seconds": 95.12
}
```

---

### Get Course

`GET /api/v1/courses/{course_id}`

Retrieve a full course with all chapters and progress.

```bash
curl http://localhost:8000/api/v1/courses/a1b2c3d4-e5f6-7890-abcd-ef1234567890
```

**Response (200):**

```json
{
  "course": {
    "id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
    "user_id": "user-123",
    "title": "Mastering Machine Learning and Neural Networks",
    "description": "...",
    "topic": "Introduction to Machine Learning and Neural Networks",
    "source_type": "topic",
    "total_chapters": 6,
    "estimated_time": 90,
    "status": "ready",
    "outline": { "..." },
    "model_used": "claude-haiku-4-5-20251001",
    "created_at": "2026-02-08T12:00:00Z",
    "completed_at": "2026-02-08T12:00:35Z"
  },
  "progress": null
}
```

**Error:** `404 "Course not found"`

---

### Get Chapter

`GET /api/v1/courses/{course_id}/chapters/{chapter_order}`

Retrieve a specific chapter by order number (1-indexed).

```bash
curl http://localhost:8000/api/v1/courses/a1b2c3d4/chapters/1
```

**Response (200):**

```json
{
  "chapter": {
    "id": "f6a7b8c9-d0e1-2345-fabc-456789012345",
    "course_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
    "order": 1,
    "title": "The Foundations of Machine Learning",
    "learning_objectives": [
      "Define machine learning and distinguish supervised, unsupervised, and reinforcement learning",
      "Identify real-world applications of each ML paradigm"
    ],
    "content": "# The Foundations of Machine Learning\n\n## Why This Matters\n\nMachine learning powers everything from...\n\n## Core Content\n\n### What Is Machine Learning?\n\nMachine learning is a subset of artificial intelligence...\n\n## Key Takeaways\n\n- Machine learning enables computers to learn from data...\n- Supervised learning uses labeled data...",
    "content_format": "markdown",
    "estimated_time": 15,
    "key_concepts": ["Supervised Learning", "Feature Engineering", "Training Data"],
    "sources": [
      {
        "number": 1,
        "title": "A Few Useful Things to Know About Machine Learning",
        "url": "https://homes.cs.washington.edu/~pedrod/papers/cacm12.pdf",
        "date": "2012-10-01",
        "source_type": "academic",
        "relevance": "Foundational paper on ML best practices"
      }
    ],
    "quiz": {
      "questions": [
        {
          "id": "q1",
          "type": "multiple_choice",
          "question": "Which type of learning uses labeled training data to make predictions?",
          "options": [
            "Unsupervised learning",
            "Supervised learning",
            "Reinforcement learning",
            "Transfer learning"
          ],
          "correct_answer": 1,
          "explanation": "Supervised learning uses labeled datasets where input-output pairs are provided. Unsupervised learning works with unlabeled data, reinforcement learning uses rewards, and transfer learning adapts pre-trained models."
        },
        {
          "id": "q2",
          "type": "true_false",
          "question": "Feature engineering is the process of selecting and transforming input variables to improve model performance.",
          "options": ["True", "False"],
          "correct_answer": 0,
          "explanation": "True. Feature engineering involves creating, selecting, and transforming features to help the model better capture patterns in data."
        }
      ]
    },
    "status": "ready",
    "generated_at": "2026-02-08T12:00:10Z",
    "word_count": 1050
  }
}
```

**Error:** `404 "Chapter not found"`

---

### List User Courses

`GET /api/v1/users/{user_id}/courses`

List all courses belonging to a user.

```bash
curl http://localhost:8000/api/v1/users/user-123/courses
```

**Response (200):**

```json
{
  "user_id": "user-123",
  "course_count": 3,
  "courses": [
    {
      "id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
      "title": "Mastering Machine Learning and Neural Networks",
      "topic": "Introduction to Machine Learning and Neural Networks",
      "total_chapters": 6,
      "estimated_time": 90,
      "status": "ready",
      "created_at": "2026-02-08T12:00:00Z"
    }
  ]
}
```

---

### Ask Course Question

`POST /api/v1/courses/{course_id}/ask`

Ask a question about a course. Answers are grounded in the uploaded source material using Pinecone vector search. Only available for courses generated from files or topic+files.

**Content-Type:** `multipart/form-data`

| Field      | Type   | Required | Description                        |
|------------|--------|----------|------------------------------------|
| `question` | string | Yes      | Question about the course content  |
| `model`    | string | No       | Model key for answer generation    |

```bash
curl -X POST http://localhost:8000/api/v1/courses/a1b2c3d4/ask \
  -F "question=What is backpropagation?" \
  -F "model=claude-haiku-4-5"
```

**Response (200):**

```json
{
  "course_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "question": "What is backpropagation?",
  "answer": "Based on the source material, backpropagation is an algorithm used to train neural networks by computing gradients of the loss function... [Source 1] [Source 2]",
  "sources_used": 3,
  "source_excerpts": [
    {
      "text": "Backpropagation computes the gradient of the loss function with respect to each weight...",
      "file": "ml_textbook.pdf"
    }
  ]
}
```

**Error:** `404 "Course not found: {course_id}"`

---

## Learning Profiles

### Create/Update Learning Profile

`POST /api/v1/learning-profile`

Save or update user learning preferences. These preferences personalize all future course generation.

**Content-Type:** `application/json`

```json
{
  "user_id": "user-123",
  "format_pref": "reading",
  "depth_pref": "detailed",
  "role": "student",
  "learning_goal": "curiosity",
  "example_pref": "real_world"
}
```

| Field          | Options                                          |
|----------------|--------------------------------------------------|
| `format_pref`  | `reading`, `listening`, `testing`, `mixed`       |
| `depth_pref`   | `quick`, `detailed`, `conversational`, `academic`|
| `role`         | `student`, `professional`, `graduate_student`    |
| `learning_goal`| `exams`, `career`, `curiosity`, `supplement`     |
| `example_pref` | `real_world`, `technical`, `stories`, `analogies`|

**Response (201):**

```json
{
  "success": true,
  "profile": {
    "user_id": "user-123",
    "format_pref": "reading",
    "depth_pref": "detailed",
    "role": "student",
    "learning_goal": "curiosity",
    "example_pref": "real_world"
  },
  "message": "Learning preferences saved"
}
```

---

### Get Learning Profile

`GET /api/v1/learning-profile/{user_id}`

```bash
curl http://localhost:8000/api/v1/learning-profile/user-123
```

**Response (200) -- profile exists:**

```json
{
  "exists": true,
  "profile": {
    "user_id": "user-123",
    "format_pref": "reading",
    "depth_pref": "detailed",
    "role": "student",
    "learning_goal": "curiosity",
    "example_pref": "real_world",
    "created_at": "2026-02-08T10:00:00Z",
    "updated_at": "2026-02-08T10:00:00Z"
  }
}
```

**Response (200) -- no profile:**

```json
{
  "exists": false,
  "message": "No profile found. User will use defaults."
}
```

---

## Progress Tracking

### Update Progress

`POST /api/v1/courses/{course_id}/progress`

Update chapter completion status and quiz scores.

**Content-Type:** `application/json`

```json
{
  "chapter_id": "f6a7b8c9-d0e1-2345-fabc-456789012345",
  "completed": true,
  "quiz_score": 85,
  "time_spent_minutes": 18
}
```

**Response (200):**

```json
{
  "success": true,
  "overall_progress": {
    "completed_chapters": 2,
    "total_chapters": 6,
    "percentage": 33,
    "started_at": "2026-02-08T13:00:00Z",
    "last_activity": "2026-02-08T14:30:00Z"
  }
}
```

---

### Get User Progress

`GET /api/v1/users/{user_id}/progress`

Get learning progress across all courses.

```bash
curl http://localhost:8000/api/v1/users/user-123/progress
```

**Response (200):**

```json
{
  "user_id": "user-123",
  "total_courses": 3,
  "completed_courses": 1,
  "in_progress": 1,
  "courses": [
    {
      "course": {
        "id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
        "title": "Mastering Machine Learning and Neural Networks",
        "total_chapters": 6,
        "status": "ready"
      },
      "progress": {
        "user_id": "user-123",
        "course_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
        "chapter_progress": [
          {
            "chapter_id": "f6a7b8c9-d0e1-2345-fabc-456789012345",
            "completed": true,
            "completed_at": "2026-02-08T14:00:00Z",
            "quiz_attempts": [
              {
                "attempt_id": 1,
                "score": 85,
                "completed_at": "2026-02-08T14:00:00Z"
              }
            ],
            "time_spent_minutes": 18
          }
        ],
        "overall_progress": {
          "completed_chapters": 1,
          "total_chapters": 6,
          "percentage": 17,
          "started_at": "2026-02-08T13:00:00Z",
          "last_activity": "2026-02-08T14:00:00Z"
        }
      }
    }
  ]
}
```

---

## Models

### List Available Models

`GET /api/v1/models`

List all available AI models for course generation with their capabilities and costs.

```bash
curl http://localhost:8000/api/v1/models
```

**Response (200):**

```json
{
  "models": [
    {
      "id": "claude-haiku-4-5",
      "name": "claude-haiku-4-5-20251001",
      "provider": "anthropic",
      "supports_web_search": true,
      "search_mode": "native",
      "estimated_cost_per_course": 0.06,
      "default": true
    },
    {
      "id": "claude-sonnet-4-5",
      "name": "claude-sonnet-4-5-20250929",
      "provider": "anthropic",
      "supports_web_search": true,
      "search_mode": "native",
      "estimated_cost_per_course": 0.11,
      "default": false
    },
    {
      "id": "claude-opus-4-6",
      "name": "claude-opus-4-6",
      "provider": "anthropic",
      "supports_web_search": true,
      "search_mode": "native",
      "estimated_cost_per_course": 0.21,
      "default": false
    },
    {
      "id": "gpt-4o",
      "name": "gpt-4o",
      "provider": "openai",
      "supports_web_search": true,
      "search_mode": "native",
      "estimated_cost_per_course": 0.11,
      "default": false
    },
    {
      "id": "gpt-5-mini",
      "name": "gpt-5-mini",
      "provider": "openai",
      "supports_web_search": true,
      "search_mode": "native",
      "estimated_cost_per_course": 0.02,
      "default": false
    },
    {
      "id": "gpt-5.2",
      "name": "gpt-5.2",
      "provider": "openai",
      "supports_web_search": true,
      "search_mode": "native",
      "estimated_cost_per_course": 0.11,
      "default": false
    },
    {
      "id": "llama-4-scout",
      "name": "meta-llama/llama-4-scout-17b-16e-instruct",
      "provider": "groq",
      "supports_web_search": false,
      "search_mode": "perplexity",
      "estimated_cost_per_course": 0.02,
      "default": false
    },
    {
      "id": "llama-4-maverick",
      "name": "meta-llama/llama-4-maverick-17b-128e-instruct",
      "provider": "groq",
      "supports_web_search": false,
      "search_mode": "perplexity",
      "estimated_cost_per_course": 0.02,
      "default": false
    }
  ]
}
```

**Search mode descriptions:**

| Mode        | Behavior                                                        |
|-------------|-----------------------------------------------------------------|
| `native`    | Model uses built-in web search (Claude tool, GPT Responses API) |
| `perplexity`| Pre-fetches sources via Perplexity API before chapter generation |
| `none`      | No web search; cites from training knowledge only               |

---

## Document Processing (RAG)

### Process Document

`POST /process_document`

Upload and process documents: extract text via OCR, chunk, embed, and index in Pinecone for later retrieval.

**Content-Type:** `multipart/form-data`

| Field        | Type     | Required | Description                               |
|--------------|----------|----------|-------------------------------------------|
| `files`      | file(s)  | Yes      | Documents to process                      |
| `file_urls`  | string   | Yes      | JSON array of URLs corresponding to files |
| `space_id`   | string   | Yes      | Space to associate documents with         |
| `use_openai` | boolean  | No       | Use OpenAI embeddings (default: `true`)   |

```bash
curl -X POST http://localhost:8000/process_document \
  -F "files=@lecture.pdf" \
  -F 'file_urls=["https://example.com/lecture.pdf"]' \
  -F "space_id=space-abc"
```

**Response (200):**

```json
{
  "document_ids": [
    {
      "file": "lecture.pdf",
      "document_id": "doc-uuid-1234",
      "url": "https://example.com/lecture.pdf"
    }
  ]
}
```

**With partial errors:**

```json
{
  "document_ids": [
    { "file": "good.pdf", "document_id": "doc-uuid-1234", "url": "..." }
  ],
  "errors": [
    { "file": "bad.pdf", "error": "OCR extraction failed" }
  ]
}
```

---

### Answer Query

`POST /answer_query`

Answer a question using the RAG pipeline. Searches indexed documents, reranks results, and generates a grounded answer.

**Content-Type:** `multipart/form-data`

| Field                    | Type    | Required | Default              | Description                                      |
|--------------------------|---------|----------|----------------------|--------------------------------------------------|
| `query`                  | string  | Yes      | --                   | User's question                                  |
| `document_id`            | string  | Yes      | --                   | Document to search within                        |
| `space_id`               | string  | No       | `null`               | Space ID filter                                  |
| `acl_tags`               | string  | No       | `null`               | Comma-separated ACL tags                         |
| `use_openai_embeddings`  | boolean | No       | `true`               | Use OpenAI for embeddings                        |
| `search_by_space_only`   | boolean | No       | `false`              | Search by space only, ignoring document_id       |
| `rerank_top_n`           | int     | No       | `10`                 | Number of results to rerank                      |
| `max_context_chunks`     | int     | No       | `5`                  | Max chunks in context                            |
| `fast_mode`              | boolean | No       | `false`              | Optimized settings for speed                     |
| `allow_general_knowledge`| boolean | No       | `false`              | Supplement with general knowledge                |
| `enable_websearch`       | boolean | No       | `false`              | Enable contextual web search                     |
| `model`                  | string  | No       | `openai/gpt-oss-120b`| Model for generation                             |
| `enrichment_mode`        | string  | No       | `simple`             | `simple` or `advanced`                           |
| `learning_style`         | string  | No       | `null`               | `academic_focus`, `deep_dive`, `quick_practical`, `exploratory_curious`, `narrative_reader` |
| `educational_mode`       | boolean | No       | `false`              | Enable tutoring approach                         |
| `conversation_history`   | string  | No       | `null`               | JSON string `[{"role":"user","content":"..."}]`  |
| `include_diagrams`       | boolean | No       | `true`               | Include diagrams in response                     |
| `max_diagrams`           | int     | No       | `3`                  | Max diagrams to return                           |

```bash
curl -X POST http://localhost:8000/answer_query \
  -F "query=What is photosynthesis?" \
  -F "document_id=doc-uuid-1234" \
  -F "space_id=space-abc"
```

**Response (200):**

```json
{
  "answer": "Photosynthesis is the process by which green plants convert sunlight into chemical energy...",
  "citations": [
    {
      "text": "Photosynthesis occurs in the chloroplasts...",
      "source_file": "biology.pdf",
      "score": 0.92
    }
  ],
  "diagrams": [],
  "time_ms": 1250
}
```

---

### Answer Query (Streaming)

`POST /answer_query_stream`

Same parameters as `/answer_query` but returns a Server-Sent Events (SSE) stream with real-time status updates and progressive answer chunks.

**Response:** `Content-Type: text/event-stream`

```
data: {"type": "status", "message": "Searching documents..."}

data: {"type": "status", "message": "Reranking results..."}

data: {"type": "chunk", "content": "Photosynthesis is "}

data: {"type": "chunk", "content": "the process by which..."}

data: {"type": "done", "citations": [...], "time_ms": 1350}
```

---

## YouTube Processing

### Process YouTube Videos

`POST /process_youtube`

Extract transcripts from YouTube videos, chunk, embed, and index for RAG.

**Content-Type:** `multipart/form-data`

| Field             | Type    | Required | Default                    | Description                    |
|-------------------|---------|----------|----------------------------|--------------------------------|
| `youtube_urls`    | string  | Yes      | --                         | JSON array of YouTube URLs     |
| `space_id`        | string  | Yes      | --                         | Space to associate with        |
| `embedding_model` | string  | No       | `text-embedding-ada-002`   | Embedding model                |
| `chunk_size`      | int     | No       | `512`                      | Chunk size in tokens           |
| `overlap_tokens`  | int     | No       | `80`                       | Token overlap between chunks   |

```bash
curl -X POST http://localhost:8000/process_youtube \
  -F 'youtube_urls=["https://youtube.com/watch?v=abc123"]' \
  -F "space_id=space-abc"
```

**Response (200):**

```json
{
  "document_ids": [
    {
      "youtube_url": "https://youtube.com/watch?v=abc123",
      "document_id": "yt-doc-uuid-5678"
    }
  ]
}
```

---

### Extract Transcript (yt-dlp)

`POST /extract_transcript_ytdlp`

Extract transcript using yt-dlp. Most reliable method -- works without proxies.

| Field       | Type   | Required | Default | Description                  |
|-------------|--------|----------|---------|------------------------------|
| `video_url` | string | Yes      | --      | YouTube URL or video ID      |
| `languages` | string | No       | `en`    | Comma-separated lang codes   |

```bash
curl -X POST http://localhost:8000/extract_transcript_ytdlp \
  -F "video_url=https://youtube.com/watch?v=abc123"
```

**Response (200):**

```json
{
  "success": true,
  "video_id": "abc123",
  "video_title": "Introduction to Machine Learning",
  "transcript": "Welcome to this lecture on machine learning...",
  "thumbnail": "https://img.youtube.com/vi/abc123/maxresdefault.jpg",
  "language": "en",
  "is_automatic": false,
  "transcript_entries": [
    { "text": "Welcome to this lecture", "start": 0.0, "duration": 3.5 }
  ],
  "method": "yt-dlp"
}
```

---

### Extract Transcript (youtube-transcript-api)

`POST /extract_youtube_transcript`

Extract transcript using youtube-transcript-api. Supports WebShare proxy for cloud deployments.

| Field       | Type    | Required | Default | Description                        |
|-------------|---------|----------|---------|------------------------------------|
| `video_url` | string  | Yes      | --      | YouTube URL or video ID            |
| `use_proxy` | boolean | No       | `false` | Use WebShare proxy                 |
| `languages` | string  | No       | `en`    | Comma-separated language codes     |

---

### Get YouTube Video Info

`GET /youtube_info_ytdlp`

Get video metadata and available subtitles using yt-dlp.

| Param       | Type   | Required | Description             |
|-------------|--------|----------|-------------------------|
| `video_url` | string | Yes      | YouTube URL or video ID |

```bash
curl "http://localhost:8000/youtube_info_ytdlp?video_url=https://youtube.com/watch?v=abc123"
```

**Response (200):**

```json
{
  "success": true,
  "video_id": "abc123",
  "title": "Introduction to Machine Learning",
  "description": "This lecture covers the basics of ML...",
  "duration": 3600,
  "channel": "MIT OpenCourseWare",
  "upload_date": "20250115",
  "view_count": 1500000,
  "thumbnail": "https://img.youtube.com/vi/abc123/maxresdefault.jpg",
  "video_url": "https://www.youtube.com/watch?v=abc123",
  "available_subtitles": ["en", "es", "fr"],
  "method": "yt-dlp"
}
```

---

### Get Transcript Languages

`GET /youtube_transcript_info`

List available transcript languages for a video.

| Param       | Type    | Required | Default | Description             |
|-------------|---------|----------|---------|-------------------------|
| `video_url` | string  | Yes      | --      | YouTube URL or video ID |
| `use_proxy` | boolean | No       | `false` | Use WebShare proxy      |

**Response (200):**

```json
{
  "success": true,
  "video_id": "abc123",
  "video_url": "https://www.youtube.com/watch?v=abc123",
  "thumbnail": "https://img.youtube.com/vi/abc123/maxresdefault.jpg",
  "available_transcripts": [
    { "language": "English", "language_code": "en", "is_generated": false },
    { "language": "Spanish", "language_code": "es", "is_generated": true }
  ]
}
```

---

### YouTube Proxy Status

`GET /youtube_proxy_status`

Check if WebShare proxy is configured for YouTube transcript extraction.

**Response (200):**

```json
{
  "proxy_configured": true,
  "has_username": true,
  "has_password": true,
  "message": "Proxy fully configured",
  "instructions": {
    "step_1": "Sign up at https://www.webshare.io/",
    "step_2": "Purchase RESIDENTIAL proxy package (not Static or Proxy Server)",
    "step_3": "Get your proxy username and password from dashboard",
    "step_4": "Set environment variables: WEBSHARE_PROXY_USERNAME and WEBSHARE_PROXY_PASSWORD",
    "step_5": "Restart the server"
  }
}
```

---

### Test Vcyon Transcript

`POST /test_vcyon_transcript`

Test the Vcyon API for YouTube transcript extraction.

| Field       | Type   | Required | Default | Description              |
|-------------|--------|----------|---------|--------------------------|
| `video_url` | string | Yes      | --      | YouTube URL or video ID  |
| `languages` | string | No       | `en`    | Comma-separated langs    |

---

## Study Materials

### Generate Flashcards

`POST /generate_flashcards`

Generate flashcards from an indexed document with comprehensive coverage.

**Content-Type:** `application/json`

```json
{
  "document_id": "doc-uuid-1234",
  "space_id": "space-abc",
  "user_id": "user-123",
  "num_questions": 20,
  "max_chunks": 1000,
  "target_coverage": 0.80,
  "enable_gap_filling": true
}
```

**Response (200):**

```json
{
  "flashcards": [
    {
      "question": "What is the primary function of mitochondria?",
      "answer": "To produce ATP through cellular respiration",
      "hint": "Think about the 'powerhouse' of the cell",
      "explanation": "Mitochondria convert nutrients into ATP through oxidative phosphorylation..."
    }
  ],
  "metadata": {
    "total_flashcards": 20,
    "chunks_used": 45,
    "coverage_score": 0.85
  }
}
```

---

### Regenerate Flashcards

`POST /regenerate_flashcards`

Generate a new set of flashcards covering different parts of the document. Same request/response format as [Generate Flashcards](#generate-flashcards).

---

### Generate Quiz

`POST /generate_quiz`

Generate quiz questions from an indexed document.

**Content-Type:** `application/json`

```json
{
  "document_id": "doc-uuid-1234",
  "space_id": "space-abc",
  "user_id": "user-123",
  "question_type": "both",
  "num_questions": 30,
  "max_chunks": 1000,
  "target_coverage": 0.80,
  "enable_gap_filling": true,
  "set_id": 1,
  "title": "Biology Midterm Review",
  "description": "Quiz covering chapters 1-5"
}
```

| `question_type` | Description                         |
|------------------|-------------------------------------|
| `mcq`            | Multiple choice only                |
| `true_false`     | True/false only                     |
| `both`           | Mix of multiple choice and true/false|

**Response (200):**

```json
{
  "quiz": {
    "title": "Biology Midterm Review",
    "description": "Quiz covering chapters 1-5",
    "set_id": 1,
    "questions": [
      {
        "id": "q1",
        "type": "mcq",
        "question": "Which organelle is responsible for protein synthesis?",
        "options": ["Mitochondria", "Ribosome", "Golgi apparatus", "Lysosome"],
        "correct_answer": 1,
        "explanation": "Ribosomes are the cellular machinery for translating mRNA into proteins."
      }
    ]
  },
  "metadata": {
    "total_questions": 30,
    "coverage_score": 0.82,
    "chunks_processed": 120
  }
}
```

---

### Regenerate Quiz

`POST /regenerate_quiz`

Generate a new quiz set avoiding duplicate questions from previous sets. Same request format as [Generate Quiz](#generate-quiz) -- `set_id` is auto-incremented.

---

### Generate Notes

`POST /generate_notes`

Generate comprehensive markdown study notes from an indexed document.

**Content-Type:** `multipart/form-data`

| Field               | Type    | Required | Default      | Description                                |
|---------------------|---------|----------|--------------|--------------------------------------------|
| `document_id`       | string  | Yes      | --           | Document to generate notes for             |
| `space_id`          | string  | No       | `null`       | Space ID (needed for database save)        |
| `academic_level`    | string  | No       | `graduate`   | `undergraduate`, `graduate`, `msc`, `phd`  |
| `include_diagrams`  | boolean | No       | `true`       | Include diagrams from PDF                  |
| `include_mermaid`   | boolean | No       | `true`       | Generate mermaid diagrams                  |
| `max_chunks`        | int     | No       | `2000`       | Max chunks to retrieve                     |
| `target_coverage`   | float   | No       | `0.85`       | Target coverage (0.0-1.0)                  |
| `enable_gap_filling`| boolean | No       | `true`       | Enable intelligent gap-filling             |
| `acl_tags`          | string  | No       | `null`       | Comma-separated ACL tags                   |

```bash
curl -X POST http://localhost:8000/generate_notes \
  -F "document_id=doc-uuid-1234" \
  -F "space_id=space-abc" \
  -F "academic_level=graduate" \
  -F "target_coverage=0.90"
```

**Response (200):**

```json
{
  "notes_markdown": "# Comprehensive Study Notes: Cell Biology\n\n## Chapter 1: Cell Structure\n\n### 1.1 The Cell Membrane\n\nThe cell membrane is a selectively permeable barrier...\n\n```mermaid\ngraph TD\n    A[Cell] --> B[Nucleus]\n    A --> C[Cytoplasm]\n```\n\n...",
  "metadata": {
    "academic_level": "graduate",
    "total_pages": 120,
    "total_chapters": 8,
    "total_chunks_processed": 350,
    "diagrams_included": 12,
    "generation_time_seconds": 145.2,
    "coverage_score": 0.98,
    "text_coverage_percentage": 0.95,
    "notes_length_chars": 50000,
    "generated_at": "2026-02-08T15:30:00Z"
  },
  "status": "success",
  "saved_to_database": true,
  "content_id": "content-uuid-9999",
  "saved_to_file": "note/doc-uuid-1234_graduate_20260208_153000.md"
}
```

---

## Google Drive

### List Drive Files

`POST /api/google-drive/files`

List files from a user's Google Drive.

**Content-Type:** `application/json`

```json
{
  "access_token": "ya29.a0...",
  "refresh_token": "1//0e...",
  "folder_id": null,
  "file_types": ["pdf", "doc", "docx"],
  "max_results": 100
}
```

**Response (200):**

```json
{
  "files": [
    {
      "id": "1AbCdEfGhIjKlMnOpQrStUvWxYz",
      "name": "Lecture Notes.pdf",
      "mimeType": "application/pdf",
      "size": 2048000,
      "modifiedTime": "2026-02-01T10:00:00Z"
    }
  ],
  "updated_tokens": {
    "access_token": "ya29.new...",
    "refresh_token": "1//0e..."
  }
}
```

---

### Import Drive Files

`POST /api/google-drive/import`

Import selected Google Drive files into the RAG pipeline for indexing.

**Content-Type:** `application/json`

```json
{
  "file_ids": ["1AbCdEfGhIjKlMnOpQrStUvWxYz"],
  "space_id": "space-abc",
  "access_token": "ya29.a0...",
  "refresh_token": "1//0e..."
}
```

**Response (200):**

```json
{
  "status": "completed",
  "message": "Successfully processed all 1 files",
  "processed_files": [
    {
      "file_id": "1AbCdEfGhIjKlMnOpQrStUvWxYz",
      "document_id": "doc-uuid-7890",
      "filename": "Lecture Notes.pdf"
    }
  ],
  "errors": [],
  "updated_tokens": {
    "access_token": "ya29.new...",
    "refresh_token": "1//0e..."
  }
}
```

**Partial success:**

```json
{
  "status": "partial_success",
  "message": "Processed 2/3 files successfully",
  "processed_files": ["..."],
  "errors": [
    { "file_id": "...", "error": "File too large" }
  ],
  "updated_tokens": { "..." }
}
```
