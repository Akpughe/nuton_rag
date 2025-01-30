import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Union, AsyncGenerator
import openai
import chromadb
from dotenv import load_dotenv
import logging
import json
import asyncio
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import time
from typing import AsyncGenerator
from openai import AsyncOpenAI


# Load environment variables and setup logging
load_dotenv()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class QuizRequest(BaseModel):
    pdf_ids: Optional[List[str]] = None
    yt_ids: Optional[List[str]] = None
    audio_ids: Optional[List[str]] = None
    num_questions: int = 15
    difficulty: str = "medium"
    batch_size: int = 3  # Number of questions in initial batch

class StreamingQuizResponse(BaseModel):
    questions: List[Dict]
    is_complete: bool
    total_questions: int
    current_batch: int

    def to_json(self) -> str:
        """Convert the response to JSON string"""
        return json.dumps(self.model_dump())

@lru_cache(maxsize=1000)
def get_embedding_cached(text: str):
    """Cached version of embedding generation"""
    response = openai.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response.data[0].embedding

class OptimizedStudyGenerator:
    def __init__(self, executor=None):
        self.openai_key = os.getenv('OPENAI_API_KEY')
        if not self.openai_key:
            raise ValueError("OpenAI API key is missing")
        
        chroma_host = os.getenv('CHROMA_HOST', os.getenv('CHROMA_DB_CONNECTION_STRING'))
        chroma_port = int(os.getenv('CHROMA_PORT', 8000))
        self.chroma_client = chromadb.HttpClient(host=chroma_host, port=chroma_port)
        # self.executor = ThreadPoolExecutor(max_workers=3)
        self.executor = executor
        self.client = AsyncOpenAI()  # Use 

    async def get_relevant_content_parallel(self, pdf_ids=None, yt_ids=None, audio_ids=None):
        """Retrieve content from different sources in parallel"""
        content = []
        references = {}
        
        query_embedding = get_embedding_cached("main concepts, key points, important definitions")
        
        async def query_collection(collection_name, source_id, id_field):
            try:
                collection = self.chroma_client.get_collection(collection_name)
                results = collection.query(
                    query_embeddings=[query_embedding],
                    where={id_field: source_id},
                    n_results=3
                )
                return results, collection_name, source_id
            except Exception as e:
                logger.error(f"Error querying {collection_name}: {e}")
                return None, collection_name, source_id

        tasks = []
        if pdf_ids:
            tasks.extend([
                query_collection("pdf_embeddings", pdf_id, "pdf_id")
                for pdf_id in pdf_ids
            ])
        if yt_ids:
               tasks.extend([
                query_collection("youtube_embeddings", yt_id, "yt_id")
                for yt_id in yt_ids
            ])
        if audio_ids:
            tasks.extend([
                query_collection("audio_embeddings", audio_id, "audio_id")
                for audio_id in audio_ids
            ])

        results = await asyncio.gather(*tasks)
        
        for result, collection_name, source_id in results:
            if result and result['documents']:
                content.extend(result['documents'][0])
                ref_key = f"{collection_name.split('_')[0]}_{source_id}"
                references[ref_key] = (
                    [meta.get('page') for meta in result['metadatas'][0]]
                    if collection_name == "pdf_embeddings"
                    else ["content"]
                )

        return "\n".join(content), references

    async def generate_questions_stream(
        self, content: str, num_questions: int, 
        difficulty: str, batch_size: int
    ) -> AsyncGenerator[str, None]:
        """Generate quiz questions in batches and yield JSON strings"""
        questions_generated = 0
        batch_num = 1

        while questions_generated < num_questions:
            current_batch = min(batch_size, num_questions - questions_generated)
            
            prompt = f"""
            Based on the following content, generate {current_batch} multiple-choice questions
            at {difficulty} difficulty level.

             **Requirements:**
          - Each question must have **4 answer options** labeled A, B, C, and D.
          - Indicate the **correct answer** clearly for each question.
          - The questions should test **deep understanding** of the text, focusing on key concepts, terms, or crucial details.
          - Ensure there are no redundant questions, and the questions span a variety of topics from the text.
          - Ensure answers are evenly distributed among the options (A, B, C, D) instead of being repetitive (e.g., all “B” or “C”).

          
            Return in JSON format:
            {{
                "questions": [
                    {{
                        "question": "Question text here",
                        "options": {{
                        "A": "Option A text",
                        "B": "Option B text",
                        "C": "Option C text",
                        "D": "Option D text"
                        }},
                        "answer": "A"
                    }}
                ]
            }}
            
            Content:
            {content}
            """

            try:
                # Use OpenAI's async streaming API
                response = await self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are creating quiz questions."},
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"},
                    stream=True  # Enable streaming
                )

                response_text = ""
                async for chunk in response:
                    if chunk.choices[0].delta.content:
                        response_text += chunk.choices[0].delta.content

                batch_questions = json.loads(response_text).get("questions", [])
                questions_generated += len(batch_questions)

                # logger.info(f"Received batch {batch_num}: {len(batch_questions)} questions.")

                response_data = StreamingQuizResponse(
                    questions=batch_questions,
                    is_complete=(questions_generated >= num_questions),
                    total_questions=num_questions,
                    current_batch=batch_num
                )

                yield response_data.to_json() + "\n"
                batch_num += 1

            except Exception as e:
                logger.error(f"Error generating questions batch {batch_num}: {e}")
                continue

# FastAPI App
app = FastAPI()
study_generator = OptimizedStudyGenerator()

@app.post("/generate-quiz-stream")
async def create_quiz_stream(request: QuizRequest):
    try:
        start_time = time.time()
        content, references = await study_generator.get_relevant_content_parallel(
            request.pdf_ids,
            request.yt_ids,
            request.audio_ids
        )
        logger.info(f"Content retrieval time: {time.time() - start_time:.2f}s")

        if not content:
            raise HTTPException(status_code=404, detail="No content found")

        async def stream():
            async for batch in study_generator.generate_questions_stream(
                content,
                request.num_questions,
                request.difficulty,
                request.batch_size
            ):
                # Parse the batch JSON for cleaner logging
                try:
                    batch_data = json.loads(batch)
                    logger.info(f"Streaming batch {batch_data.get('current_batch', 'unknown')}: {json.dumps(batch_data, indent=2)}")
                except json.JSONDecodeError:
                    logger.error("Failed to decode batch JSON")
                
                yield batch

        return StreamingResponse(stream(), media_type="application/x-ndjson")

    except Exception as e:
        logger.error(f"Error in create_quiz_stream: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8081)