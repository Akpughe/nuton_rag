# Pinecone Study Generator

A FastAPI service for generating quizzes and flashcards based on documents stored in Pinecone vector database.

## Features

- Generate multiple-choice quiz questions from embedded documents
- Create flashcards with questions, answers and hints
- Stream results for immediate UI feedback
- Optimized content processing for faster generation
- Automatic saving of generated content

## Setup

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file with the following variables:
   ```
   OPENAI_API_KEY=your_openai_api_key
   PINECONE_API_KEY=your_pinecone_api_key
   PINECONE_ENVIRONMENT=your_pinecone_environment
   PINECONE_INDEX_NAME=your_index_name
   ```

## Usage

### Start the server

```bash
uvicorn pinecone_study_service:app --host 0.0.0.0 --port 8082
```

### API Endpoints

#### Generate Quiz Questions

```
POST /generate-quiz-stream
```

Request body:

```json
{
  "document_ids": ["doc1", "doc2"],
  "num_questions": 15,
  "difficulty": "medium",
  "batch_size": 3
}
```

Response:
Streaming response of batches of quiz questions.

#### Generate Flashcards

```
POST /generate-flashcards
```

Request body:

```json
{
  "document_ids": ["doc1", "doc2"],
  "num_questions": 15,
  "difficulty": "medium",
  "batch_size": 3
}
```

Response:
Streaming response of batches of flashcards.

## API Endpoints (pipeline.py)

### 1. Process Document

**POST** `/process_document`

Process and index one or more uploaded documents (PDFs, etc.) with optional OpenAI embeddings.

**Parameters:**

- `files` (List[UploadFile], required): One or more files to upload.
- `file_urls` (str, required): JSON string array of URLs corresponding to each file (for storage reference).
- `space_id` (str, required): Space ID to associate with all documents.
- `use_openai` (bool, optional): Whether to use OpenAI for embeddings (default: False).

**Example Request (multipart/form-data):**

```
files: [file1.pdf, file2.pdf]
file_urls: ["https://example.com/file1.pdf", "https://example.com/file2.pdf"]
space_id: myspace123
use_openai: true
```

**Response:**

```
{
  "document_ids": [
    {"file": "file1.pdf", "document_id": "abc123", "url": "https://example.com/file1.pdf"},
    {"file": "file2.pdf", "document_id": "def456", "url": "https://example.com/file2.pdf"}
  ],
  "errors": [
    {"file": "file2.pdf", "error": "Failed to process..."}
  ]
}
```

---

### 2. Answer Query

**POST** `/answer_query`

Answer a user query using the RAG pipeline (hybrid search, rerank, LLM answer generation).

**Parameters:**

- `query` (str, required): User's question.
- `document_id` (str, required): Document ID to search within.
- `space_id` (str, optional): Space ID to filter by.
- `acl_tags` (str, optional): Comma-separated ACL tags to filter by.
- `use_openai_embeddings` (bool, optional): Use OpenAI for embeddings (default: False).
- `search_by_space_only` (bool, optional): Search by space only, ignore document_id (default: False).
- `rerank_top_n` (int, optional): Number of results to rerank (default: 10).
- `max_context_chunks` (int, optional): Max context chunks for LLM (default: 5).
- `fast_mode` (bool, optional): Use faster settings (default: False).

**Example Request (form-data):**

```
query: "What is the main idea?"
document_id: "abc123"
space_id: "myspace123"
acl_tags: "tag1,tag2"
use_openai_embeddings: true
fast_mode: true
```

**Response:**

```
{
  "answer": "The main idea is...",
  "citations": [...],
  "time_ms": 1234
}
```

---

### 3. Process YouTube

**POST** `/process_youtube`

Extract, chunk, embed, and index transcripts from one or more YouTube videos.

**Parameters:**

- `youtube_urls` (str, required): JSON string array of YouTube URLs.
- `space_id` (str, required): Space ID to associate with all videos.
- `embedding_model` (str, optional): Embedding model to use (default: "text-embedding-ada-002").
- `chunk_size` (int, optional): Chunk size in tokens (default: 512).
- `overlap_tokens` (int, optional): Overlap tokens between chunks (default: 80).

**Example Request (form-data):**

```
youtube_urls: ["https://youtube.com/watch?v=abc123"]
space_id: "myspace123"
embedding_model: "text-embedding-ada-002"
```

**Response:**

```
{
  "document_ids": [
    {"youtube_url": "https://youtube.com/watch?v=abc123", "document_id": "yt_abc123"}
  ],
  "errors": [
    {"youtube_url": "https://youtube.com/watch?v=def456", "error": "Failed to process..."}
  ]
}
```

---

### 4. Generate Flashcards

**POST** `/generate_flashcards`

Generate flashcards from a document.

**Request Body (JSON):**

```
{
  "document_id": "abc123",
  "space_id": "myspace123",
  "num_questions": 10,
  "acl_tags": ["tag1", "tag2"]
}
```

**Response:**

```
{
  "flashcards": [...]
}
```

---

### 5. Regenerate Flashcards

**POST** `/regenerate_flashcards`

Regenerate flashcards from a document (same as above, but for re-generation).

**Request Body (JSON):**

```
{
  "document_id": "abc123",
  "space_id": "myspace123",
  "num_questions": 10,
  "acl_tags": ["tag1", "tag2"]
}
```

**Response:**

```
{
  "flashcards": [...]
}
```

---

### 6. Generate Quiz

**POST** `/generate_quiz`

Generate a quiz from a document.

**Request Body (JSON):**

```
{
  "document_id": "abc123",
  "space_id": "myspace123",
  "question_type": "both", // "mcq", "true_false", or "both"
  "num_questions": 30,
  "acl_tags": "tag1,tag2",
  "rerank_top_n": 50,
  "use_openai_embeddings": true,
  "set_id": 1,
  "title": "Quiz Title",
  "description": "Quiz description."
}
```

**Response:**

```
{
  "quiz": [...]
}
```

## Components

- `pinecone_study_service.py`: Main FastAPI application
- `pinecone_index_manager.py`: Utility for managing Pinecone index and document embeddings
- `optimize_content_processing.py`: Utilities for efficient content retrieval and processing

## Performance Optimizations

- Multi-query retrieval for better content coverage
- Parallel processing of document queries
- Content summarization for large documents
- Background tasks for database operations
- Response streaming for immediate UI feedback

## Example

```python
import requests
import json

url = "http://localhost:8082/generate-quiz-stream"
payload = {
    "document_ids": ["doc123"],
    "num_questions": 10,
    "difficulty": "hard",
    "batch_size": 3
}

response = requests.post(url, json=payload, stream=True)

for line in response.iter_lines():
    if line:
        batch = json.loads(line)
        print(f"Received batch {batch['current_batch']} with {len(batch['questions'])} questions")
        for question in batch['questions']:
            print(f"Q: {question['question']}")
            print(f"A: {question['answer']}")
            print()
```
