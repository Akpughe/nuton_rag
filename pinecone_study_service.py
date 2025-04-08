import os
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Union, AsyncGenerator
import pinecone
from openai import AsyncOpenAI
from groq import AsyncGroq
from dotenv import load_dotenv
import logging
import json
import asyncio
from functools import lru_cache
import time
from optimize_content_processing import OptimizedContentProcessor

# Load environment variables and setup logging
load_dotenv()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class QuizRequest(BaseModel):
    document_ids: List[str]
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
        
    @classmethod
    def validate_questions(cls, questions: List[Dict]) -> List[Dict]:
        """Validate that questions have the required fields and types"""
        validated = []
        for q in questions:
            # Ensure each question has a type field
            if "type" not in q:
                if "options" in q:
                    q["type"] = "mcq"
                else:
                    q["type"] = "true_false"
                
            # Ensure MCQ questions have options
            if q["type"] == "mcq" and "options" not in q:
                logger.warning(f"Skipping invalid MCQ question without options: {q.get('question', 'Unknown')}")
                continue
                
            # Remove options from true/false questions
            if q["type"] == "true_false" and "options" in q:
                del q["options"]
                
            validated.append(q)
        return validated

class StreamingFlashcardResponse(BaseModel):
    flashcards: List[Dict[str, str]]
    is_complete: bool
    total_flashcards: int
    current_batch: int

    def to_json(self) -> str:
        """Convert the response to JSON string"""
        return json.dumps(self.model_dump())
        
    @classmethod
    def validate_flashcards(cls, flashcards: List[Dict]) -> List[Dict]:
        """Validate that flashcards have the required fields for detailed flashcards"""
        validated = []
        for card in flashcards:
            # Set type to detailed
            card["type"] = "detailed"
                
            # Make sure detailed flashcards have a hint
            if "hint" not in card:
                card["hint"] = "Think about the key concepts related to this topic."
                
            # Skip flashcards without question or answer
            if "question" not in card or "answer" not in card:
                logger.warning(f"Skipping invalid flashcard missing question or answer")
                continue
            
            # Enforce answer length limits for brevity and speed
            if "answer" in card and len(card["answer"].split()) > 30:
                # Truncate long answers to about 2 sentences (approx 30 words)
                words = card["answer"].split()
                truncated = " ".join(words[:30])
                # Try to find the last sentence boundary
                last_period = truncated.rfind(".")
                if last_period > len(truncated) * 0.5:  # If we can find a good breakpoint
                    card["answer"] = truncated[:last_period+1]
                else:
                    card["answer"] = truncated + "..."
                
            # Also truncate long hints
            if "hint" in card and len(card["hint"].split()) > 15:
                words = card["hint"].split()
                card["hint"] = " ".join(words[:15])
                if not card["hint"].endswith("."):
                    card["hint"] += "."
                
            validated.append(card)
        return validated

@lru_cache(maxsize=1000)
def get_embedding_cached(text: str):
    """Cached version of embedding generation"""
    client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response.data[0].embedding

class PineconeStudyGenerator:
    def __init__(self):
        self.openai_key = os.getenv('OPENAI_API_KEY')
        if not self.openai_key:
            raise ValueError("OpenAI API key is missing")
        
        self.groq_key = os.getenv('GROQ_API_KEY')
        
        self.content_processor = OptimizedContentProcessor()
        self.client = AsyncOpenAI(api_key=self.openai_key)
        
        # Initialize Groq client if API key is available
        if self.groq_key:
            self.groq_client = AsyncGroq(api_key=self.groq_key)
            self.use_groq = True
            logger.info("Groq client initialized successfully")
            
            # Define Groq models to try with fallback
            self.groq_models = [
                "meta-llama/llama-4-scout-17b-16e-instruct",  # Fast model
                "llama-3.1-8b-instant",  # Backup
                "meta-llama/llama-4-scout-17b-16e-instruct"      # Second fallback
            ]
        else:
            self.use_groq = False
            logger.warning("No Groq API key found. Using OpenAI only.")
        
        # Initialize database connection for saving results
        # This would depend on your database setup

    async def get_relevant_content(self, document_ids: List[str]):
        """Retrieve optimized content from documents using the content processor"""
        try:
            # Log the document IDs being processed
            logger.info(f"Retrieving content for document IDs: {document_ids}")
            
            # Check if document IDs exist in Pinecone
            document_stats = {}
            for doc_id in document_ids:
                try:
                    # Get stats for each document
                    stats = self.content_processor.index.describe_index_stats(
                        filter={"document_id": doc_id}
                    )
                    vector_count = stats.namespaces.get('', {}).get('vector_count', 0)
                    document_stats[doc_id] = vector_count
                    logger.info(f"Document {doc_id} has {vector_count} vectors in Pinecone")
                except Exception as e:
                    logger.error(f"Error getting stats for document {doc_id}: {e}")
                    document_stats[doc_id] = 0
            
            # Check if any documents have vectors
            if not any(document_stats.values()):
                logger.error(f"No vectors found for any of the document IDs: {document_ids}")
                # Try a different approach - direct query with minimal filter
                try:
                    # Get stats for the entire index
                    index_stats = self.content_processor.index.describe_index_stats()
                    logger.info(f"Index stats: {index_stats.total_vector_count} total vectors")
                    
                    # If we have vectors but none match our document IDs, the IDs might be incorrect
                    if index_stats.total_vector_count > 0:
                        logger.warning("Index has vectors but none match the provided document IDs")
                except Exception as e:
                    logger.error(f"Error getting index stats: {e}")
                
                return "", {}
            
            # First try multi-query approach for better retrieval
            query_types = [
                "main concepts and key points",
                "important definitions and terminology",
                "examples and applications",
                "critical insights and conclusions"
            ]
            
            # Process in parallel for speed
            batch_results = await self.content_processor.batch_process_document_queries(
                document_ids, query_types
            )
            
            # Check if we got any results
            result_count = sum(len(results) for results in batch_results.values())
            logger.info(f"Retrieved {result_count} total results from batch processing")
            
            if result_count == 0:
                logger.warning(f"No results from batch processing for document IDs: {document_ids}")
                # Try direct content extraction as fallback
                content, references = await self.content_processor.extract_relevant_content(
                    document_ids, "comprehensive content overview"
                )
                if content:
                    logger.info(f"Retrieved content via fallback method, length: {len(content)}")
                    return content, references
                else:
                    logger.error("Fallback content retrieval also failed")
                    return "", {}
            
            # Combine results from different query types, prioritizing by score
            all_chunks = []
            for query_type, results in batch_results.items():
                # Sort by score and take top results
                sorted_results = sorted(results, key=lambda x: x.get('score', 0), reverse=True)
                top_results = sorted_results[:5]  # Take top 5 from each query type
                
                for result in top_results:
                    if result.get('text') and result.get('text') not in [c.get('text') for c in all_chunks]:
                        all_chunks.append(result)
            
            # Sort by score for final selection
            all_chunks = sorted(all_chunks, key=lambda x: x.get('score', 0), reverse=True)
            
            # Extract content and references
            content = "\n\n".join([chunk.get('text', '') for chunk in all_chunks])
            
            # Create references dictionary
            references = {}
            for chunk in all_chunks:
                doc_id = chunk.get('document_id')
                if doc_id:
                    page = chunk.get('page', 'unknown')
                    if doc_id not in references:
                        references[doc_id] = []
                    if page not in references[doc_id]:
                        references[doc_id].append(page)
            
            logger.info(f"Generated content with length: {len(content)}")
            
            # If content is very long, summarize it to stay within token limits
            if len(content.split()) > 3000:
                content = await self.content_processor.summarize_content(content)
            
            return content, references
        except Exception as e:
            logger.error(f"Error in get_relevant_content: {str(e)}", exc_info=True)
            return "", {}

    async def generate_questions_stream(
        self, content: str, num_questions: int, 
        difficulty: str, batch_size: int,
        question_types: List[str] = None
    ) -> AsyncGenerator[str, None]:
        """
        Generate quiz questions in batches and yield JSON strings
        
        Args:
            content: The content to generate questions from
            num_questions: Total number of questions to generate
            difficulty: Difficulty level (easy, medium, hard)
            batch_size: Number of questions per batch
            question_types: List of question types to generate ("mcq", "true_false")
        """
        questions_generated = 0
        batch_num = 1
        
        # Default to both question types if not specified
        if not question_types:
            question_types = ["mcq", "true_false"]
        
        # Convert to lowercase for consistency
        question_types = [qt.lower() for qt in question_types]
        
        # Log the question types being generated
        logger.info(f"Generating questions with types: {question_types}")
        
        # Optimize content for generation if it's very long
        if len(content.split()) > 2000:
            logger.info(f"Content is long ({len(content.split())} words) - optimizing for speed")
            try:
                # Extract key points and reduce content size
                summarization_prompt = f"""
                Extract the key points, definitions, and important concepts from the following content.
                Be concise and focus only on the most important information that would be useful for quiz questions.
                Organize the information in a way that preserves the most important facts and relationships.
                
                Content:
                {content[:5000]}  # Using only the first 5000 chars to avoid token limits
                """
                
                summary_response = await self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "Summarize the content into key points for quiz creation."},
                        {"role": "user", "content": summarization_prompt}
                    ],
                    max_tokens=1000,
                    temperature=0.3
                )
                
                optimized_content = summary_response.choices[0].message.content
                if optimized_content and len(optimized_content) > 300:  # Ensure we got a valid summary
                    content = optimized_content
                    logger.info(f"Content optimized to {len(content.split())} words")
            except Exception as e:
                logger.warning(f"Error optimizing content: {e}. Using original content.")

        while questions_generated < num_questions:
            current_batch = min(batch_size, num_questions - questions_generated)
            
            # Determine the distribution of question types for this batch
            mcq_count = 0
            tf_count = 0
            
            if len(question_types) == 1:
                # If only one type is requested, all questions are that type
                if question_types[0] == "mcq":
                    mcq_count = current_batch
                else:
                    tf_count = current_batch
            else:
                # If both types are requested, split evenly with any remainder going to MCQ
                mcq_count = current_batch // 2 + (current_batch % 2)
                tf_count = current_batch // 2
                
            logger.info(f"Batch {batch_num}: Generating {mcq_count} MCQ and {tf_count} True/False questions")
            
            # Build a clearer and more structured prompt
            prompt = f"""
            TASK: Generate exactly {current_batch} quiz questions based on the provided content.
            Difficulty level: {difficulty.upper()}
            
            QUESTION DISTRIBUTION:
            - Multiple-choice questions: {mcq_count}
            - True/False questions: {tf_count}
            
            FORMATTING REQUIREMENTS:
            
            1. For MULTIPLE-CHOICE questions ({mcq_count}):
               - Each question MUST have the field "type": "mcq"
               - Each question must have 4 options labeled A, B, C, and D
               - Include ONLY ONE correct answer
               - Indicate the correct answer with the field "answer": "A" (or B, C, D)
               - Evenly distribute correct answers (don't make all answers "C")
               
            2. For TRUE/FALSE questions ({tf_count}):
               - Each question MUST have the field "type": "true_false"
               - Format as a statement that is definitively true or false
               - The answer should ONLY be "True" or "False" (not T/F, not Yes/No)
               - The question field should be a complete statement, not a question
               - Try to have a balance of true and false statements
            
            CONTENT REQUIREMENTS:
            - Questions should be based ONLY on the content provided
            - Focus on key concepts, important facts, and significant details
            - Ensure variety across the questions to cover different aspects of the content
            - Make questions challenging and test deep understanding, not just memorization
            - Avoid ambiguous or opinion-based questions that could have multiple correct answers
            
            JSON STRUCTURE:
            Return a properly formatted JSON with the following structure:
            {{
                "questions": [
                    {{
                        "question": "What is the capital of France?",
                        "type": "mcq",
                        "options": {{
                            "A": "London",
                            "B": "Berlin",
                            "C": "Paris",
                            "D": "Madrid"
                        }},
                        "answer": "C"
                    }},
                    {{
                        "question": "Paris is the capital of France.",
                        "type": "true_false",
                        "answer": "True"
                    }}
                ]
            }}
            
            Only include options for multiple-choice questions, not for true/false questions.
            Make sure every question has the correct "type" field.
            
            Content:
            {content}
            """

            try:
                response_text = ""
                # First try with Groq if available
                if self.use_groq:
                    for model in self.groq_models:
                        try:
                            logger.info(f"Generating questions with Groq model: {model}")
                            
                            async_response = await self.groq_client.chat.completions.create(
                                model=model,
                                messages=[
                                    {
                                        "role": "system", 
                                        "content": "You are a specialized quiz creator. Your task is to generate questions in the exact format specified, following all the instructions precisely. Every question must include a 'type' field that specifies whether it's 'mcq' or 'true_false'."
                                    },
                                    {"role": "user", "content": prompt}
                                ],
                                response_format={"type": "json_object"},
                                temperature=0.7,
                                max_tokens=1500,
                                stream=True
                            )
                            
                            # Process the streaming response
                            async for chunk in async_response:
                                if chunk.choices[0].delta.content:
                                    response_text += chunk.choices[0].delta.content
                            
                            # If we got a valid response, break out of the loop
                            if response_text and '{"questions"' in response_text:
                                logger.info(f"Successfully generated questions with Groq model: {model}")
                                break
                            else:
                                logger.warning(f"Empty or invalid response from Groq model: {model}")
                        
                        except Exception as e:
                            logger.warning(f"Error with Groq model {model}: {e}")
                            continue
                
                # Fall back to OpenAI if Groq failed or is not available
                if not response_text or '{"questions"' not in response_text:
                    logger.info("Falling back to OpenAI for question generation")
                    async_response = await self.client.chat.completions.create(
                        model="gpt-4o-mini",  # Using a faster model
                        messages=[
                            {
                                "role": "system", 
                                "content": "You are a specialized quiz creator. Your task is to generate questions in the exact format specified, following all the instructions precisely. Every question must include a 'type' field that specifies whether it's 'mcq' or 'true_false'."
                            },
                            {"role": "user", "content": prompt}
                        ],
                        response_format={"type": "json_object"},
                        stream=True
                    )
                    
                    # Clear any partial response from Groq
                    response_text = ""
                    
                    # Process the streaming response
                    async for chunk in async_response:
                        if chunk.choices[0].delta.content:
                            response_text += chunk.choices[0].delta.content

                # Parse and validate the response
                response_data = json.loads(response_text)
                batch_questions = response_data.get("questions", [])
                
                # Validate each question has the required type field
                validated_questions = []
                for q in batch_questions:
                    if "type" not in q:
                        # If type is missing, try to infer it
                        if "options" in q:
                            q["type"] = "mcq"
                        else:
                            q["type"] = "true_false"
                    
                    # Make sure true_false questions don't have options
                    if q["type"] == "true_false" and "options" in q:
                        del q["options"]
                    
                    # Verify MCQ questions have options and a valid answer
                    if q["type"] == "mcq" and "options" not in q:
                        # Skip invalid MCQ questions
                        logger.warning(f"Skipping invalid MCQ question without options: {q.get('question', 'Unknown')}")
                        continue
                    
                    validated_questions.append(q)
                
                # Check if we have the right balance of question types
                actual_mcq = sum(1 for q in validated_questions if q["type"] == "mcq")
                actual_tf = sum(1 for q in validated_questions if q["type"] == "true_false")
                
                logger.info(f"Generated {actual_mcq} MCQ and {actual_tf} True/False questions (requested {mcq_count} MCQ, {tf_count} True/False)")
                
                questions_generated += len(validated_questions)

                response_data = StreamingQuizResponse(
                    questions=validated_questions,
                    is_complete=(questions_generated >= num_questions),
                    total_questions=num_questions,
                    current_batch=batch_num
                )

                yield response_data.to_json() + "\n"
                batch_num += 1

            except Exception as e:
                logger.error(f"Error generating questions batch {batch_num}: {e}")
                continue

    async def generate_flashcards_stream(
        self, content: str, num_flashcards: int, 
        difficulty: str, batch_size: int,
        flashcard_types: List[str] = None
    ) -> AsyncGenerator[str, None]:
        """
        Generate flashcards in batches and yield JSON strings
        
        Args:
            content: The content to generate flashcards from
            num_flashcards: Total number of flashcards to generate
            difficulty: Difficulty level (easy, medium, hard)
            batch_size: Number of flashcards per batch
            flashcard_types: Not used - we only generate detailed flashcards
        """
        flashcards_generated = 0
        batch_num = 1
        
        # We only support detailed flashcards
        logger.info("Generating detailed flashcards only")
        
        # Optimize content for generation
        if len(content.split()) > 2000:
            logger.info(f"Content is long ({len(content.split())} words) - optimizing for speed")
            try:
                # Extract key points and reduce content size
                summarization_prompt = f"""
                Extract the key points, definitions, and important concepts from the following content.
                Be concise and focus only on the most important information that would be useful for flashcards.
                Organize the information in a way that preserves the most important facts and relationships.
                
                Content:
                {content[:5000]}  # Using only the first 5000 chars to avoid token limits
                """
                
                summary_response = await self.groq_client.chat.completions.create(
                    model=self.groq_models[0],
                    messages=[
                        {"role": "system", "content": "Summarize the content into key points for flashcard creation."},
                        {"role": "user", "content": summarization_prompt}
                    ],
                    max_tokens=1000,
                    temperature=0.3
                )
                
                optimized_content = summary_response.choices[0].message.content
                if optimized_content and len(optimized_content) > 300:  # Ensure we got a valid summary
                    content = optimized_content
                    logger.info(f"Content optimized to {len(content.split())} words")
            except Exception as e:
                logger.warning(f"Error optimizing content: {e}. Using original content.")

        while flashcards_generated < num_flashcards:
            current_batch = min(batch_size, num_flashcards - flashcards_generated)
            
            logger.info(f"Batch {batch_num}: Generating {current_batch} detailed flashcards")
            
            # Build a simplified prompt only for detailed flashcards
            prompt = f"""
            TASK: Generate exactly {current_batch} detailed flashcards based on the provided content.
            Difficulty level: {difficulty.upper()}
            
            IMPORTANT: ALL ANSWERS MUST BE CONCISE - MAXIMUM 1-2 SENTENCES. Keep flashcards brief and focused.
            
            FORMATTING REQUIREMENTS:
            - Each flashcard must have a thought-provoking question in the "question" field
            - Include a CONCISE answer (1 sentences max) in the "answer" field
            - Include a brief hint (1 sentence) in the "hint" field that guides without giving away the answer
            - Each flashcard must have the field "type": "detailed"
            
            CONTENT REQUIREMENTS:
            - Flashcards should be based ONLY on the content provided
            - Focus on key concepts, important facts, and significant details
            - BREVITY IS CRITICAL - keep all answers short and to the point
            - Avoid ambiguous answers or questions with multiple possible answers
            
            JSON STRUCTURE:
            Return a properly formatted JSON with the following structure:
            {{
                "flashcards": [
                    {{
                        "question": "What is the relationship between energy and wavelength in electromagnetic radiation?",
                        "answer": "Energy is inversely proportional to wavelength, so shorter wavelengths have higher energy.",
                        "hint": "Think about the formula E=hc/λ in physics.",
                        "type": "detailed"
                    }}
                ]
            }}
            
            Content:
            {content}
            """

            try:
                response_text = ""
                # First try with Groq if available
                if self.use_groq:
                    for model in self.groq_models:
                        try:
                            logger.info(f"Generating flashcards with Groq model: {model}")
                            
                            async_response = await self.groq_client.chat.completions.create(
                                model=model,
                                messages=[
                                    {
                                        "role": "system", 
                                        "content": "You are a specialized flashcard creator that prioritizes brevity and speed. Create concise detailed flashcards with short answers (1-2 sentences maximum)."
                                    },
                                    {"role": "user", "content": prompt}
                                ],
                                response_format={"type": "json_object"},
                                temperature=0.7,
                                max_tokens=1500,
                                stream=True
                            )
                            
                            # Process the streaming response
                            async for chunk in async_response:
                                if chunk.choices[0].delta.content:
                                    response_text += chunk.choices[0].delta.content
                            
                            # If we got a valid response, break out of the loop
                            if response_text and '{"flashcards"' in response_text:
                                logger.info(f"Successfully generated flashcards with Groq model: {model}")
                                break
                            else:
                                logger.warning(f"Empty or invalid response from Groq model: {model}")
                        
                        except Exception as e:
                            logger.warning(f"Error with Groq model {model}: {e}")
                            continue
                
                # Fall back to OpenAI if Groq failed or is not available
                if not response_text or '{"flashcards"' not in response_text:
                    logger.info("Falling back to OpenAI for flashcard generation")
                    async_response = await self.client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {
                                "role": "system", 
                                "content": "You are a specialized flashcard creator that prioritizes brevity and speed. Create concise detailed flashcards with short answers (1-2 sentences maximum)."
                            },
                            {"role": "user", "content": prompt}
                        ],
                        response_format={"type": "json_object"},
                        temperature=0.7,
                        max_tokens=1500,
                        stream=True
                    )
                    
                    # Clear any partial response from Groq
                    response_text = ""
                    
                    # Process the streaming response
                    async for chunk in async_response:
                        if chunk.choices[0].delta.content:
                            response_text += chunk.choices[0].delta.content

                # Parse and validate the response
                response_data = json.loads(response_text)
                batch_flashcards = response_data.get("flashcards", [])
                
                # Validate the flashcards - all should be detailed type
                validated_flashcards = []
                for card in batch_flashcards:
                    # Set type to detailed if not specified
                    card["type"] = "detailed"
                    
                    # Make sure each flashcard has a hint
                    if "hint" not in card:
                        card["hint"] = "Think about the key concepts related to this topic."
                    
                    # Skip flashcards without question or answer
                    if "question" not in card or "answer" not in card:
                        logger.warning(f"Skipping invalid flashcard missing question or answer")
                        continue
                    
                    # Enforce answer length limits
                    if len(card["answer"].split()) > 30:
                        words = card["answer"].split()
                        truncated = " ".join(words[:30])
                        last_period = truncated.rfind(".")
                        if last_period > len(truncated) * 0.5:
                            card["answer"] = truncated[:last_period+1]
                        else:
                            card["answer"] = truncated + "..."
                    
                    # Also truncate long hints
                    if len(card["hint"].split()) > 15:
                        words = card["hint"].split()
                        card["hint"] = " ".join(words[:15])
                        if not card["hint"].endswith("."):
                            card["hint"] += "."
                    
                    validated_flashcards.append(card)
                
                flashcards_generated += len(validated_flashcards)
                logger.info(f"Generated {len(validated_flashcards)} detailed flashcards")

                response_data = StreamingFlashcardResponse(
                    flashcards=validated_flashcards,
                    is_complete=(flashcards_generated >= num_flashcards),
                    total_flashcards=num_flashcards,
                    current_batch=batch_num
                )

                yield response_data.to_json() + "\n"
                batch_num += 1

            except Exception as e:
                logger.error(f"Error generating flashcards batch {batch_num}: {e}")
                continue

    async def save_quiz_to_db(self, document_ids: List[str], questions: List[Dict]):
        """Save generated quiz questions to database"""
        # This would be implemented based on your database schema
        try:
            # Create a unique ID for this quiz
            quiz_id = f"quiz_{int(time.time())}_{document_ids[0]}"
            
            # Store quiz metadata
            quiz_data = {
                "id": quiz_id,
                "document_ids": document_ids,
                "timestamp": time.time(),
                "num_questions": len(questions),
                "questions": questions
            }
            
            # Implementation depends on your database structure
            logger.info(f"Saved quiz {quiz_id} with {len(questions)} questions")
            return quiz_id
        except Exception as e:
            logger.error(f"Error saving quiz to database: {e}")
            return None

    async def save_flashcards_to_db(self, document_ids: List[str], flashcards: List[Dict]):
        """Save generated flashcards to database"""
        # This would be implemented based on your database schema
        try:
            # Create a unique ID for this flashcard set
            set_id = f"flashcards_{int(time.time())}_{document_ids[0]}"
            
            # Store flashcard set metadata
            flashcard_data = {
                "id": set_id,
                "document_ids": document_ids,
                "timestamp": time.time(),
                "num_flashcards": len(flashcards),
                "flashcards": flashcards
            }
            
            # Implementation depends on your database structure
            logger.info(f"Saved flashcard set {set_id} with {len(flashcards)} cards")
            return set_id
        except Exception as e:
            logger.error(f"Error saving flashcards to database: {e}")
            return None

# FastAPI App
app = FastAPI()
study_generator = PineconeStudyGenerator()

@app.post("/generate-quiz-stream")
async def create_quiz_stream(request: QuizRequest, background_tasks: BackgroundTasks):
    try:
        start_time = time.time()
        content, references = await study_generator.get_relevant_content(request.document_ids)
        logger.info(f"Content retrieval time: {time.time() - start_time:.2f}s")

        if not content:
            raise HTTPException(status_code=404, detail="No content found for the provided document IDs")

        async def stream():
            all_questions = []
            async for batch in study_generator.generate_questions_stream(
                content,
                request.num_questions,
                request.difficulty,
                request.batch_size
            ):
                # Parse the batch JSON for cleaner logging and storing
                try:
                    batch_data = json.loads(batch)
                    logger.info(f"Streaming batch {batch_data.get('current_batch', 'unknown')} with {len(batch_data.get('questions', []))} questions")
                    
                    # Add questions to our collection for saving later
                    all_questions.extend(batch_data.get('questions', []))
                    
                    # If this is the last batch, save all questions to database
                    if batch_data.get('is_complete', False):
                        # Use background task to avoid blocking the response
                        background_tasks.add_task(
                            study_generator.save_quiz_to_db,
                            request.document_ids, 
                            all_questions
                        )
                        
                except json.JSONDecodeError:
                    logger.error("Failed to decode batch JSON")
                
                yield batch

        return StreamingResponse(stream(), media_type="application/x-ndjson")

    except Exception as e:
        logger.error(f"Error in create_quiz_stream: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-flashcards")
async def create_flashcards_stream(request: QuizRequest, background_tasks: BackgroundTasks):
    try:
        start_time = time.time()
        content, references = await study_generator.get_relevant_content(request.document_ids)
        logger.info(f"Content retrieval time: {time.time() - start_time:.2f}s")

        if not content:
            raise HTTPException(status_code=404, detail="No content found for the provided document IDs")

        async def stream():
            all_flashcards = []
            async for batch in study_generator.generate_flashcards_stream(
                content,
                request.num_questions,
                request.difficulty,
                request.batch_size
            ):
                try:
                    batch_data = json.loads(batch)
                    logger.info(f"Streaming batch {batch_data.get('current_batch', 'unknown')} with {len(batch_data.get('flashcards', []))} flashcards")
                    
                    # Add flashcards to our collection for saving later
                    all_flashcards.extend(batch_data.get('flashcards', []))
                    
                    # If this is the last batch, save all flashcards to database
                    if batch_data.get('is_complete', False):
                        # Use background task to avoid blocking the response
                        background_tasks.add_task(
                            study_generator.save_flashcards_to_db,
                            request.document_ids, 
                            all_flashcards
                        )
                        
                except json.JSONDecodeError:
                    logger.error("Failed to decode batch JSON")
                
                yield batch

        return StreamingResponse(stream(), media_type="application/x-ndjson")

    except Exception as e:
        logger.error(f"Error in create_flashcards_stream: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8082) 