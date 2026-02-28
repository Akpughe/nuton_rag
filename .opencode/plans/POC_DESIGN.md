# Nuton AI Course Generation - POC Design Document

## Overview
POC implementation of AI-powered course generation with JSON file storage (transitioning to database later).

---

## 1. File Storage Structure

```
nuton_rag/
├── courses/                          # Course storage folder
│   ├── course_{uuid}/                # Each course gets a folder
│   │   ├── course.json               # Course metadata & outline
│   │   ├── chapter_{1..N}.json       # Individual chapter content
│   │   ├── sources.json              # Aggregated citations
│   │   └── quiz_{chapter_id}.json    # Chapter-specific quizzes
│   └── index.json                    # Index of all courses
├── learning_profiles.json            # User personalization data
└── course_generation_logs.json       # Generation logs & errors
```

---

## 2. Database Schema (JSON Structure)

### 2.1 Learning Profile Schema

```json
{
  "user_id": "uuid-string",
  "format_pref": "reading|listening|testing|mixed",
  "depth_pref": "quick|detailed|conversational|academic",
  "role": "student|professional|graduate_student",
  "learning_goal": "exams|career|curiosity|supplement",
  "example_pref": "real_world|technical|stories|analogies",
  "created_at": "2026-02-01T10:00:00Z",
  "updated_at": "2026-02-01T10:00:00Z"
}
```

**5-Question Onboarding Flow:**
1. "How do you prefer to learn?" → format_pref
2. "What depth do you want?" → depth_pref  
3. "What's your current role?" → role
4. "Why are you learning?" → learning_goal
5. "What examples help you most?" → example_pref

### 2.2 Course Schema

```json
{
  "id": "course_uuid",
  "user_id": "user_uuid",
  "title": "Understanding Quantum Computing",
  "description": "A beginner-friendly introduction to quantum computing principles...",
  "topic": "quantum computing basics",
  "source_type": "topic|files",
  "source_files": [
    {
      "file_id": "doc_uuid",
      "filename": "physics_lecture.pdf",
      "extracted_topic": "Quantum Mechanics"
    }
  ],
  "multi_file_organization": "thematic_bridge|sequential_sections|separate_courses",
  "total_chapters": 4,
  "estimated_time": 60,
  "status": "generating|ready|error",
  "personalization_params": {
    "format_pref": "reading",
    "depth_pref": "detailed",
    "role": "student",
    "learning_goal": "exams",
    "example_pref": "real_world"
  },
  "outline": {
    "chapters": [
      {
        "order": 1,
        "id": "chapter_uuid_1",
        "title": "What is Quantum Computing?",
        "objectives": ["Define quantum computing", "Contrast with classical computing"],
        "key_concepts": ["Qubit", "Superposition"],
        "estimated_time": 15
      }
    ]
  },
  "model_used": "claude-sonnet-4|gpt-4o|llama-4-scout",
  "created_at": "2026-02-01T10:00:00Z",
  "completed_at": "2026-02-01T10:45:00Z",
  "progress": {
    "completed_chapters": 2,
    "total_chapters": 4,
    "percentage": 50
  }
}
```

### 2.3 Chapter Schema

```json
{
  "id": "chapter_uuid",
  "course_id": "course_uuid",
  "order": 1,
  "title": "What is Quantum Computing?",
  "learning_objectives": [
    "Define quantum computing in simple terms",
    "Explain how it differs from classical computing"
  ],
  "content": "# What is Quantum Computing?\n\nQuantum computing represents a fundamental shift...",
  "content_format": "markdown",
  "estimated_time": 15,
  "key_concepts": ["Qubit", "Superposition", "Entanglement"],
  "sources": [
    {
      "number": 1,
      "title": "Introduction to Quantum Computing",
      "url": "https://example.com/quantum-intro",
      "date": "2025-01-15",
      "source_type": "academic|news|documentation",
      "cited_in_content": ["paragraph_2", "paragraph_5"]
    }
  ],
  "quiz": {
    "questions": [
      {
        "id": "q1",
        "type": "multiple_choice",
        "question": "What is the basic unit of quantum information?",
        "options": ["Bit", "Qubit", "Byte", "Quantum bit"],
        "correct_answer": 1,
        "explanation": "A qubit (quantum bit) is the fundamental unit..."
      }
    ]
  },
  "status": "ready",
  "generated_at": "2026-02-01T10:15:00Z",
  "word_count": 1050
}
```

### 2.4 Progress Schema

```json
{
  "user_id": "user_uuid",
  "course_id": "course_uuid",
  "chapter_progress": [
    {
      "chapter_id": "chapter_uuid_1",
      "completed": true,
      "completed_at": "2026-02-01T11:00:00Z",
      "quiz_attempts": [
        {
          "attempt_id": 1,
          "score": 80,
          "answers": ["b", "c", "a"],
          "completed_at": "2026-02-01T11:15:00Z"
        }
      ],
      "time_spent_minutes": 18
    }
  ],
  "overall_progress": {
    "completed_chapters": 1,
    "total_chapters": 4,
    "percentage": 25,
    "started_at": "2026-02-01T10:30:00Z",
    "last_activity": "2026-02-01T11:15:00Z"
  }
}
```

---

## 3. FastAPI Endpoints

### 3.1 Learning Profile Endpoints

**POST** `/api/v1/learning-profile`
```json
Request:
{
  "user_id": "uuid",
  "format_pref": "reading",
  "depth_pref": "detailed", 
  "role": "student",
  "learning_goal": "exams",
  "example_pref": "real_world"
}

Response:
{
  "success": true,
  "profile": {...},
  "message": "Learning preferences saved"
}
```

**GET** `/api/v1/learning-profile/{user_id}`
```json
Response:
{
  "profile": {...},
  "exists": true
}
```

### 3.2 Course Generation Endpoints

**POST** `/api/v1/courses/from-topic`
```json
Request:
{
  "user_id": "uuid",
  "topic": "quantum computing basics",
  "context": {
    "expertise": "beginner|intermediate|advanced",
    "time_available": 60
  },
  "model": "claude-sonnet-4|gpt-4o|llama-4-scout"  // Optional, has default
}

Response (45-60s - blocks until complete):
{
  "course_id": "uuid",
  "status": "ready",
  "course": {
    "title": "...",
    "description": "...",
    "chapters": [...],
    "total_time": 60
  },
  "storage_path": "courses/course_{uuid}/",
  "generation_time_seconds": 52
}
```

**POST** `/api/v1/courses/from-files`
```json
Request:
{
  "files": [UploadFile],  // 1-10 PDF/PPT files
  "user_id": "uuid",
  "organization": "auto|thematic_bridge|sequential_sections|separate_courses",
  "model": "claude-sonnet-4|gpt-4o|llama-4-scout"  // Optional
}

Response (depends on file size + generation):
{
  "course_id": "uuid",
  "status": "ready",
  "detected_topics": ["Quantum Mechanics", "Linear Algebra"],
  "organization_chosen": "sequential_sections",
  "course": {...},
  "storage_path": "courses/course_{uuid}/"
}
```

### 3.3 Course Access Endpoints

**GET** `/api/v1/courses/{course_id}`
```json
Response:
{
  "course": {...},
  "chapters": [
    {"id": "uuid", "title": "...", "order": 1, "is_complete": true}
  ],
  "progress": {...}
}
```

**GET** `/api/v1/courses/{course_id}/chapters/{chapter_id}`
```json
Response:
{
  "chapter": {
    "id": "uuid",
    "order": 1,
    "title": "...",
    "content": "# Markdown...",
    "quiz": {...},
    "sources": [...]
  }
}
```

### 3.4 Progress Endpoints

**POST** `/api/v1/courses/{course_id}/progress`
```json
Request:
{
  "chapter_id": "uuid",
  "completed": true,
  "quiz_score": 85,
  "time_spent_minutes": 20
}

Response:
{
  "success": true,
  "overall_progress": {
    "completed_chapters": 2,
    "total_chapters": 4,
    "percentage": 50
  }
}
```

**GET** `/api/v1/users/{user_id}/progress`
```json
Response:
{
  "courses_in_progress": [...],
  "completed_courses": [...],
  "total_time_spent_hours": 12.5
}
```

---

## 4. Multi-File Detection Logic

```python
async def detect_multi_file_organization(files: List[UploadFile]):
    """
    1. Extract text from each file using Mistral OCR
    2. Use Claude to identify main topic of each (2-4 words)
    3. Calculate semantic similarity between topics
    4. Recommend organization strategy
    """
    
    # Step 1: Extract topics
    topics = []
    for file in files:
        text = await extract_text_with_mistral(file)
        topic = await claude.extract_topic(text[:2000])  # "Quantum Mechanics"
        topics.append({"file": file.filename, "topic": topic})
    
    # Step 2: Calculate similarities
    if len(topics) == 2:
        similarity = await calculate_semantic_similarity(
            topics[0]["topic"], 
            topics[1]["topic"]
        )
        
        if similarity > 0.75:
            return {
                "organization": "thematic_bridge",
                "reason": "Topics are closely related",
                "topics": topics,
                "similarity": similarity
            }
        elif similarity > 0.4:
            return {
                "organization": "ask_user",  # Show options
                "reason": "Topics may be related",
                "topics": topics,
                "similarity": similarity,
                "options": ["thematic_bridge", "sequential_sections", "separate_courses"]
            }
        else:
            return {
                "organization": "sequential_sections",
                "reason": "Topics are different subjects",
                "topics": topics,
                "similarity": similarity
            }
```

---

## 5. Prompt Engineering

### 5.1 Course Outline Generation Prompt

```python
COURSE_OUTLINE_SYSTEM_PROMPT = """
You are an expert curriculum designer and educator. Create a structured, pedagogically sound course outline.

USER CONTEXT:
- Topic: {topic}
- Expertise Level: {expertise} (beginner/intermediate/advanced)
- Time Available: {time_available} minutes
- Learning Format Preference: {format_pref}
- Explanation Depth: {depth_pref}
- User Role: {role}
- Learning Goal: {learning_goal}
- Example Preference: {example_pref}

{file_context}

REQUIREMENTS:
1. Generate 3-5 chapters that progressively build knowledge
2. Structure: Foundation → Core Concepts → Applications → Synthesis
3. Each chapter must have:
   - Clear, descriptive title
   - 2-4 specific learning objectives (measurable)
   - 2-3 key concepts covered
   - Estimated time (minutes)
4. Total time should approximate {time_available} minutes
5. Chapter progression must be logical (prerequisites first)

{organization_instructions}

OUTPUT FORMAT (JSON):
{{
  "title": "Engaging course title",
  "description": "2-3 sentence overview of what student will learn",
  "learning_objectives": ["By the end, students will be able to..."],
  "chapters": [
    {{
      "order": 1,
      "title": "Chapter Title",
      "objectives": ["Specific objective 1", "Specific objective 2"],
      "key_concepts": ["Concept 1", "Concept 2"],
      "estimated_time": 15,
      "prerequisites": []
    }}
  ],
  "total_estimated_time": 60
}}
"""
```

### 5.2 Chapter Content Generation Prompt

```python
CHAPTER_CONTENT_SYSTEM_PROMPT = """
You are an expert educator creating high-quality educational content.

CHAPTER CONTEXT:
- Course: {course_title}
- Chapter {chapter_num} of {total_chapters}: {chapter_title}
- Previous Chapter: {prev_chapter_title} (if applicable)
- Next Chapter: {next_chapter_title} (if applicable)

USER PERSONALIZATION:
- Expertise Level: {expertise}
- Learning Format: {format_pref}
- Depth Preference: {depth_pref}
- Role: {role}
- Learning Goal: {learning_goal}
- Example Preference: {example_pref}

LEARNING OBJECTIVES FOR THIS CHAPTER:
{objectives}

{source_material_context}

CONTENT REQUIREMENTS:
1. Length: 800-1200 words
2. Format: Markdown with clear hierarchy (# ## ###)
3. Structure:
   - Hook/Why This Matters (2-3 sentences)
   - Core content with {example_pref} examples
   - Practical applications or implications
   - Key Takeaways section (bullet points)
4. Tone: Match {depth_pref} style
   - "quick": Concise, bullet points, minimal fluff
   - "detailed": Comprehensive, thorough explanations
   - "conversational": Friendly, engaging, accessible
   - "academic": Formal, rigorous, precise
5. Include inline citations [1], [2], [3] for:
   - Factual claims
   - Statistics or data
   - Research findings
   - Historical events

QUIZ REQUIREMENTS:
Generate 3-5 questions:
- Mix of multiple choice and true/false
- Test understanding, not memorization
- Include explanations for correct AND incorrect answers

SEARCH REQUIREMENTS:
Use web search to find 3-5 authoritative sources on key claims.

OUTPUT FORMAT (JSON):
{{
  "content": "Full markdown content with inline citations [1], [2]...",
  "word_count": 1050,
  "key_concepts_explained": ["Qubit", "Superposition"],
  "quiz": {{
    "questions": [
      {{
        "id": "q1",
        "type": "multiple_choice",
        "question": "Clear question text?",
        "options": ["Option A", "Option B", "Option C", "Option D"],
        "correct_answer": 1,
        "explanation": "Why B is correct and why others are wrong"
      }}
    ]
  }},
  "sources": [
    {{
      "number": 1,
      "title": "Source Title",
      "url": "https://...",
      "date": "2025-01-15",
      "source_type": "academic|news|documentation|book",
      "relevance": "Brief note on what this source verifies"
    }}
  ]
}}
"""
```

---

## 6. Search Strategy (Claude + Perplexity)

```python
async def search_with_fallback(query: str, num_results: int = 5):
    """
    Primary: Claude native web search
    Fallback: Perplexity API
    """
    
    # Try Claude first (free, bundled)
    try:
        response = await claude.messages.create(
            model="claude-sonnet-4-20250514",
            messages=[{"role": "user", "content": f"Search for authoritative sources on: {query}"}],
            tools=[{"type": "web_search"}],
            max_tokens=2000
        )
        
        if response.citations:
            return parse_claude_citations(response)
    except Exception as e:
        logger.warning(f"Claude search failed: {e}")
    
    # Fallback to Perplexity
    try:
        response = await perplexity.chat.completions.create(
            model="sonar-large-128k-online",
            messages=[{"role": "user", "content": query}]
        )
        return parse_perplexity_citations(response)
    except Exception as e:
        logger.error(f"Perplexity search failed: {e}")
        return []
```

---

## 7. Model Switching Configuration

```python
# Model configuration
MODEL_CONFIG = {
    "default": "claude-sonnet-4",
    "models": {
        "claude-sonnet-4": {
            "provider": "anthropic",
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 4000,
            "cost_per_1k_input": 0.003,
            "cost_per_1k_output": 0.015,
            "supports_search": True
        },
        "gpt-4o": {
            "provider": "openai",
            "model": "gpt-4o",
            "max_tokens": 4000,
            "cost_per_1k_input": 0.005,
            "cost_per_1k_output": 0.015,
            "supports_search": False
        },
        "llama-4-scout": {
            "provider": "groq",
            "model": "meta-llama/llama-4-scout-17b-16e-instruct",
            "max_tokens": 4000,
            "cost_per_1k_input": 0.0005,
            "cost_per_1k_output": 0.001,
            "supports_search": False
        }
    }
}

# Usage in endpoint
async def generate_course(request: CourseRequest):
    model_key = request.model or MODEL_CONFIG["default"]
    model_config = MODEL_CONFIG["models"][model_key]
    
    if model_config["provider"] == "anthropic":
        return await generate_with_claude(request, model_config)
    elif model_config["provider"] == "openai":
        return await generate_with_openai(request, model_config)
    elif model_config["provider"] == "groq":
        return await generate_with_groq(request, model_config)
```

---

## 8. File Processing Pipeline

```python
async def process_uploaded_files(files: List[UploadFile]) -> List[ProcessedFile]:
    """
    1. Validate file types (PDF, PPTX, DOCX)
    2. Extract text using Mistral OCR
    3. Identify topic using Claude
    4. Store extracted content
    """
    processed = []
    
    for file in files:
        # Validate
        if not validate_file_type(file.filename):
            raise ValueError(f"Unsupported file type: {file.filename}")
        
        # Save temp
        temp_path = save_temp_file(file)
        
        try:
            # Extract with Mistral OCR
            extraction = await mistral_ocr.extract(temp_path)
            
            # Get topic
            topic = await extract_topic(extraction.text[:2000])
            
            processed.append({
                "filename": file.filename,
                "topic": topic,
                "extracted_text": extraction.text,
                "pages": extraction.pages,
                "images": extraction.images,
                "temp_path": temp_path
            })
        finally:
            cleanup_temp_file(temp_path)
    
    return processed
```

---

## 9. Implementation Priority

### Phase 1: Core Structure (Week 1)
- [ ] Create `courses/` directory structure
- [ ] Implement JSON file storage layer
- [ ] Create Pydantic models for all schemas
- [ ] Build learning profile endpoints
- [ ] Model switching infrastructure

### Phase 2: Topic-to-Course (Week 1-2)
- [ ] Course outline generation endpoint
- [ ] Chapter content generation
- [ ] Quiz generation (chapter-specific)
- [ ] Claude + Perplexity search integration
- [ ] Source citation handling

### Phase 3: File Processing (Week 2)
- [ ] Multi-file upload endpoint
- [ ] Mistral OCR integration (reuse existing)
- [ ] Topic extraction from files
- [ ] Multi-file organization detection
- [ ] File-to-course generation

### Phase 4: Progress & Polish (Week 3)
- [ ] Progress tracking endpoints
- [ ] Course retrieval endpoints
- [ ] Error handling & retries
- [ ] Cost tracking per course
- [ ] Testing & optimization

---

## 10. Cost Estimation per Course

| Component | Claude Sonnet 4 | GPT-4o | Llama-4 |
|-----------|----------------|--------|---------|
| Outline Generation | $0.02 | $0.03 | $0.005 |
| 4 Chapters @ 1000 words each | $0.08 | $0.12 | $0.02 |
| Source Search (Claude free + Perplexity backup) | $0.01 | $0.02 | $0.02 |
| Quiz Generation (included in chapter) | - | - | - |
| **Total per Course** | **~$0.11** | **~$0.17** | **~$0.045** |

---

## 11. Error Handling Strategy

```python
class CourseGenerationError(Exception):
    """Base exception for course generation"""
    pass

class OutlineGenerationError(CourseGenerationError):
    """Failed to generate outline"""
    pass

class ChapterGenerationError(CourseGenerationError):
    """Failed to generate chapter"""
    def __init__(self, chapter_num, underlying_error):
        self.chapter_num = chapter_num
        self.underlying_error = underlying_error

class SearchError(CourseGenerationError):
    """Source search failed"""
    pass

# Recovery strategies
RECOVERY_STRATEGIES = {
    "claude_timeout": {
        "retry": 2,
        "backoff": "exponential",
        "fallback_model": "gpt-4o"
    },
    "chapter_generation_fail": {
        "retry": 1,
        "regenerate": True,
        "accept_partial": True
    },
    "search_fail": {
        "continue_without_sources": True,
        "add_disclaimer": True
    }
}
```

---

## 12. Testing Checklist

- [ ] Single file PDF upload
- [ ] Multiple file upload (2-3 files)
- [ ] Topic-only generation
- [ ] Each model (Claude, GPT-4o, Llama-4)
- [ ] 5-question personalization flow
- [ ] Progress tracking
- [ ] Quiz completion
- [ ] Source citations present
- [ ] Error recovery (timeout, API failure)
- [ ] Multi-file organization detection
- [ ] Cost tracking accuracy

---

**Next Steps:**
1. Review this design document
2. Approve or request changes
3. Begin Phase 1 implementation

**Files to Create:**
- `services/course_service.py` - Core generation logic
- `routes/course_routes.py` - FastAPI endpoints  
- `models/course_models.py` - Pydantic schemas
- `prompts/course_prompts.py` - Prompt templates
- `utils/file_storage.py` - JSON file operations
- `utils/search_clients.py` - Claude + Perplexity integration

**Integration Points with Existing Code:**
- Reuse `process_document_with_openai()` for file extraction
- Reuse `mistral_ocr` integration
- Reuse `hybrid_search` for related course recommendations (future)
