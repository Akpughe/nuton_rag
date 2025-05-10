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
