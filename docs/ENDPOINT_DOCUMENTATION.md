# Flashcard & Quiz Generation Endpoints - Updated Documentation

## Overview

These endpoints now support **shared spaces with user-based data isolation**. The key requirement is that all requests must include a `user_id` field to determine ownership and visibility (`is_shared` flag).

---

## Core Concept: `is_shared` Logic

```
if user_id == space_owner_id:
    is_shared = true   # Shared with all space members
else:
    is_shared = false  # Private to this user only
```

---

## Updated Pydantic Models

### `FlashcardRequest` (UPDATED)

```python
class FlashcardRequest(BaseModel):
    document_id: str
    space_id: Optional[str] = None
    user_id: str                          # ✨ NEW: Required for ownership tracking
    num_questions: Optional[int] = None
    acl_tags: Optional[List[str]] = None
```

**Changes from current:**
- ✅ Add `user_id: str` field (required)

### `QuizRequest` (UPDATED)

```python
class QuizRequest(BaseModel):
    document_id: str
    space_id: Optional[str] = None
    user_id: str                          # ✨ NEW: Required for ownership tracking
    question_type: str = "both"
    num_questions: int = 30
    acl_tags: Optional[str] = None
    rerank_top_n: int = 50
    use_openai_embeddings: bool = True
    set_id: int = 1
    title: Optional[str] = None
    description: Optional[str] = None
```

**Changes from current:**
- ✅ Add `user_id: str` field (required)

---

## Endpoint 1: Generate Flashcards

### Request

**POST** `/generate_flashcards`

```json
{
  "document_id": "uuid-of-document",
  "space_id": "uuid-of-space",
  "user_id": "uuid-of-current-user",
  "num_questions": 30,
  "acl_tags": null
}
```

### Response Success (200)

```json
{
  "flashcards": [
    {
      "set_id": 1,
      "cards": [
        {
          "question": "What is photosynthesis?",
          "answer": "Process by which plants convert light energy into chemical energy",
          "hint": "Involves chlorophyll and sunlight",
          "explanation": "Photosynthesis occurs in plant cells..."
        }
      ]
    }
  ],
  "status": "success",
  "elapsed_seconds": 45.2,
  "total_flashcards": 30,
  "num_questions": 30
}
```

### Response Error (500)

```json
{
  "error": "Query embedding failed: Invalid API key"
}
```

### Backend Flow

1. ✅ Receive `user_id` in request
2. ✅ Call `generate_flashcards(document_id, space_id, user_id, ...)`
3. ✅ In `flashcard_process.py`, call `determine_shared_status(user_id, content_id)`
4. ✅ Call `insert_flashcard_set(content_id, flashcards, set_number, created_by=user_id, is_shared=shared_status)`
5. ✅ Return flashcards with metadata

**Database Result:**
```sql
INSERT INTO flashcard_sets (content_id, flashcards, set_number, created_by, is_shared)
VALUES ('...', [...], 1, 'user-uuid', false);  -- is_shared=false if not owner
```

---

## Endpoint 2: Regenerate Flashcards

### Request

**POST** `/regenerate_flashcards`

```json
{
  "document_id": "uuid-of-document",
  "space_id": "uuid-of-space",
  "user_id": "uuid-of-current-user",
  "num_questions": 30,
  "acl_tags": null
}
```

### Response Success (200)

```json
{
  "flashcards": [
    {
      "set_id": 2,
      "cards": [
        {
          "question": "What is photosynthesis?",
          "answer": "Process by which plants convert light energy into chemical energy",
          "hint": "Involves chlorophyll and sunlight",
          "explanation": "Photosynthesis occurs in plant cells..."
        }
      ]
    }
  ],
  "status": "success",
  "elapsed_seconds": 45.2,
  "total_flashcards": 30,
  "num_questions": 30
}
```

### Backend Flow

1. ✅ Receive `user_id` in request
2. ✅ Call `regenerate_flashcards(document_id, space_id, user_id, ...)`
3. ✅ Get existing flashcards to avoid duplicates
4. ✅ Calculate next `set_number`
5. ✅ Get ownership status: `determine_shared_status(user_id, content_id)`
6. ✅ Call `insert_flashcard_set(content_id, flashcards, next_set_number, created_by=user_id, is_shared=shared_status)`

**Database Result:**
```sql
-- New set created with same shared status as user
INSERT INTO flashcard_sets (content_id, flashcards, set_number, created_by, is_shared)
VALUES ('...', [...], 2, 'user-uuid', false);  -- Preserves is_shared from ownership
```

---

## Endpoint 3: Generate Quiz

### Request

**POST** `/generate_quiz`

```json
{
  "document_id": "uuid-of-document",
  "space_id": "uuid-of-space",
  "user_id": "uuid-of-current-user",
  "question_type": "both",
  "num_questions": 30,
  "acl_tags": null,
  "rerank_top_n": 50,
  "use_openai_embeddings": true,
  "set_id": 1,
  "title": "Biology Quiz 1",
  "description": "Quiz on photosynthesis and cellular respiration"
}
```

### Response Success (200)

```json
{
  "quiz": {
    "set_id": 1,
    "total_questions": 30,
    "questions": [
      {
        "type": "mcq",
        "question_id": "q1",
        "question_text": "What is the primary pigment in photosynthesis?",
        "correct_option": "A",
        "options": [
          {"a": "Chlorophyll"},
          {"b": "Hemoglobin"},
          {"c": "Melanin"},
          {"d": "Carotenoid"}
        ],
        "explanation": "Chlorophyll absorbs light energy..."
      }
    ]
  },
  "status": "success",
  "elapsed_seconds": 52.1,
  "total_questions": 30,
  "question_type": "both",
  "mcq_count": 21,
  "tf_count": 9,
  "set_number": 1
}
```

### Response Error (500)

```json
{
  "error": "No relevant content found."
}
```

### Backend Flow

1. ✅ Receive `user_id` in request
2. ✅ Call `generate_quiz(document_id, space_id, user_id, ...)`
3. ✅ Call `determine_shared_status(user_id, content_id)`
4. ✅ Call `insert_quiz_set(content_id, quiz_obj, set_number, created_by=user_id, is_shared=shared_status, title, description)`
5. ✅ Return quiz with metadata

**Database Result:**
```sql
INSERT INTO quiz_sets (content_id, quiz, set_number, created_by, is_shared, title, description)
VALUES ('...', {...}, 1, 'user-uuid', false, 'Biology Quiz 1', '...');
```

---

## Endpoint 4: Regenerate Quiz

### Request

**POST** `/regenerate_quiz`

```json
{
  "document_id": "uuid-of-document",
  "space_id": "uuid-of-space",
  "user_id": "uuid-of-current-user",
  "question_type": "both",
  "num_questions": 30,
  "acl_tags": null,
  "rerank_top_n": 50,
  "use_openai_embeddings": true,
  "set_id": 2,
  "title": "Biology Quiz 2",
  "description": "Additional quiz questions"
}
```

### Response Success (200)

```json
{
  "quiz": {
    "set_id": 2,
    "total_questions": 30,
    "questions": [...]
  },
  "status": "success",
  "elapsed_seconds": 52.1,
  "total_questions": 30,
  "question_type": "both",
  "mcq_count": 21,
  "tf_count": 9,
  "set_number": 2
}
```

### Backend Flow

1. ✅ Receive `user_id` in request
2. ✅ Call `regenerate_quiz(document_id, space_id, user_id, ...)`
3. ✅ Get existing quizzes to avoid duplicates
4. ✅ Calculate next `set_number`
5. ✅ Get ownership status: `determine_shared_status(user_id, content_id)`
6. ✅ Call `insert_quiz_set(content_id, quiz_obj, next_set_number, created_by=user_id, is_shared=shared_status, title, description)`

**Database Result:**
```sql
INSERT INTO quiz_sets (content_id, quiz, set_number, created_by, is_shared, title, description)
VALUES ('...', {...}, 2, 'user-uuid', false, 'Biology Quiz 2', '...');
```

---

## Implementation Checklist

### 1. Update Pydantic Models in `pipeline.py`

```python
class FlashcardRequest(BaseModel):
    document_id: str
    space_id: Optional[str] = None
    user_id: str                          # ✨ ADD THIS
    num_questions: Optional[int] = None
    acl_tags: Optional[List[str]] = None

class QuizRequest(BaseModel):
    document_id: str
    space_id: Optional[str] = None
    user_id: str                          # ✨ ADD THIS
    question_type: str = "both"
    num_questions: int = 30
    acl_tags: Optional[str] = None
    rerank_top_n: int = 50
    use_openai_embeddings: bool = True
    set_id: int = 1
    title: Optional[str] = None
    description: Optional[str] = None
```

### 2. Update Endpoint Handlers in `pipeline.py`

**For `/generate_flashcards` endpoint (line 1929):**
```python
@app.post("/generate_flashcards")
async def generate_flashcards_endpoint(
    request: FlashcardRequest
) -> JSONResponse:
    logging.info(f"Generate flashcards endpoint called for document: {request.document_id}, user: {request.user_id}")
    
    try:
        result = generate_flashcards(
            document_id=request.document_id,
            space_id=request.space_id,
            user_id=request.user_id,              # ✨ PASS THIS
            num_questions=request.num_questions,
            acl_tags=request.acl_tags
        )
        return JSONResponse(result)
    except Exception as e:
        logging.exception(f"Error in generate_flashcards_endpoint: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)
```

**For `/regenerate_flashcards` endpoint (line 1958):**
```python
@app.post("/regenerate_flashcards")
async def regenerate_flashcards_endpoint(
    request: FlashcardRequest
) -> JSONResponse:
    logging.info(f"Re-Generate flashcards endpoint called for document: {request.document_id}, user: {request.user_id}")
    
    try:
        result = regenerate_flashcards(
            document_id=request.document_id,
            space_id=request.space_id,
            user_id=request.user_id,              # ✨ PASS THIS
            num_questions=request.num_questions,
            acl_tags=request.acl_tags
        )
        return JSONResponse(result)
    except Exception as e:
        logging.exception(f"Error in regenerate_flashcards_endpoint: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)
```

**For `/generate_quiz` endpoint (line 1987):**
```python
@app.post("/generate_quiz")
async def generate_quiz_endpoint(
    request: QuizRequest
) -> JSONResponse:
    if request.question_type not in ["mcq", "true_false", "both"]:
        return JSONResponse({"error": "question_type must be one of 'mcq', 'true_false', or 'both'"}, status_code=400)
        
    acl_list = [tag.strip() for tag in request.acl_tags.split(",")] if request.acl_tags else None
    try:
        result = generate_quiz(
            document_id=request.document_id,
            space_id=request.space_id,
            user_id=request.user_id,              # ✨ PASS THIS
            question_type=request.question_type,
            num_questions=request.num_questions,
            acl_tags=acl_list,
            rerank_top_n=request.rerank_top_n,
            use_openai_embeddings=request.use_openai_embeddings,
            set_id=request.set_id,
            title=request.title,
            description=request.description
        )
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
```

**For `/regenerate_quiz` endpoint (line 2030):**
```python
@app.post("/regenerate_quiz")
async def regenerate_quiz_endpoint(
    request: QuizRequest
) -> JSONResponse:
    if request.question_type not in ["mcq", "true_false", "both"]:
        return JSONResponse({"error": "question_type must be one of 'mcq', 'true_false', or 'both'"}, status_code=400)
        
    acl_list = [tag.strip() for tag in request.acl_tags.split(",")] if request.acl_tags else None
    try:
        result = regenerate_quiz(
            document_id=request.document_id,
            space_id=request.space_id,
            user_id=request.user_id,              # ✨ PASS THIS
            question_type=request.question_type,
            num_questions=request.num_questions,
            acl_tags=acl_list,
            rerank_top_n=request.rerank_top_n,
            use_openai_embeddings=request.use_openai_embeddings,
            title=request.title,
            description=request.description
        )
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
```

### 3. Update Function Signatures

**In `flashcard_process.py`:**
```python
def generate_flashcards(
    document_id: str,
    space_id: Optional[str] = None,
    user_id: Optional[str] = None,           # ✨ ADD THIS
    num_questions: Optional[int] = None,
    acl_tags: Optional[List[str]] = None,
    rerank_top_n: int = 50,
    use_openai_embeddings: bool = False
) -> Dict[str, Any]:
    # ... implementation

def regenerate_flashcards(
    document_id: str,
    space_id: Optional[str] = None,
    user_id: Optional[str] = None,           # ✨ ADD THIS
    num_questions: Optional[int] = None,
    acl_tags: Optional[List[str]] = None,
    rerank_top_n: int = 50,
    use_openai_embeddings: bool = False
) -> Dict[str, Any]:
    # ... implementation
```

**In `quiz_process.py`:**
```python
def generate_quiz(
    document_id: str,
    space_id: Optional[str] = None,
    user_id: Optional[str] = None,           # ✨ ADD THIS
    question_type: str = "both",
    num_questions: int = 10,
    acl_tags: Optional[List[str]] = None,
    rerank_top_n: int = 50,
    use_openai_embeddings: bool = False,
    set_id: int = 1,
    title: Optional[str] = None,
    description: Optional[str] = None
) -> Dict[str, Any]:
    # ... implementation

def regenerate_quiz(
    document_id: str,
    space_id: Optional[str] = None,
    user_id: Optional[str] = None,           # ✨ ADD THIS
    question_type: str = "both",
    num_questions: int = 10,
    acl_tags: Optional[List[str]] = None,
    rerank_top_n: int = 50,
    use_openai_embeddings: bool = False,
    title: Optional[str] = None,
    description: Optional[str] = None
) -> Dict[str, Any]:
    # ... implementation
```

---

## Key Implementation Details

### When `user_id` is `None` (Backward Compatibility)

For safety, if `user_id` is not provided:
```python
is_shared = False  # Default to private
```

### Ownership Check Logic

```python
def determine_shared_status(user_id: str, content_id: str) -> bool:
    """
    True if user is space owner (shared with all)
    False if user is not owner (private to user only)
    """
    if not user_id:
        return False  # Safe default
    
    # Get space_id from generated_content
    # Get space owner from spaces table
    # Return: user_id == space_owner
```

### Database Constraints

Ensure these columns exist (you've confirmed they do):
- ✅ `flashcard_sets.created_by` (UUID)
- ✅ `flashcard_sets.is_shared` (BOOLEAN DEFAULT false)
- ✅ `quiz_sets.created_by` (UUID)
- ✅ `quiz_sets.is_shared` (BOOLEAN DEFAULT false)

---

## Frontend Integration Example

```typescript
// Frontend calling the endpoint
const generateFlashcards = async (documentId, spaceId, userId) => {
  const response = await fetch('/generate_flashcards', {
    method: 'POST',
    body: JSON.stringify({
      document_id: documentId,
      space_id: spaceId,
      user_id: userId,              // ✨ PASS CURRENT USER ID
      num_questions: 30
    })
  });
  return response.json();
};
```

---

## Testing Scenarios

### Scenario 1: Space Owner Generates
```json
{
  "document_id": "doc-123",
  "space_id": "space-456",
  "user_id": "user-owner-789"
}
```
✅ Result: `created_by: user-owner-789`, `is_shared: true`

### Scenario 2: Non-Owner Generates
```json
{
  "document_id": "doc-123",
  "space_id": "space-456",
  "user_id": "user-viewer-999"
}
```
✅ Result: `created_by: user-viewer-999`, `is_shared: false`

### Scenario 3: User Regenerates (Preserves Status)
```json
{
  "document_id": "doc-123",
  "space_id": "space-456",
  "user_id": "user-owner-789"
}
```
✅ Result: New set with `set_id: 2`, preserves `is_shared: true`

---

This documentation should guide your implementation. Ready to proceed? 🚀

