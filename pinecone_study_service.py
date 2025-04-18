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

def parse_text_flashcards(text_content: str) -> List[Dict[str, str]]:
    """
    Parse flashcards from a delimited text format into structured dictionaries
    
    Format:
    ---
    Question: What is X?
    Answer: X is Y
    Hint: Think about Z
    Explanation: Additional details about X
    ---
    """
    flashcards = []
    # Split by the '---' delimiter, ignoring empty entries
    segments = [s.strip() for s in text_content.split('---') if s.strip()]
    
    for segment in segments:
        card = {"type": "detailed"}
        
        # Parse each line to extract fields
        lines = segment.strip().split('\n')
        for line in lines:
            if not line.strip():
                continue
                
            # Find the first colon which separates field name from content
            colon_index = line.find(':')
            if colon_index > 0:
                field = line[:colon_index].strip().lower()
                value = line[colon_index+1:].strip()
                
                # Map fields to our flashcard structure
                if field == 'question':
                    card['question'] = value
                elif field == 'answer':
                    card['answer'] = value
                elif field == 'hint':
                    card['hint'] = value
                elif field == 'explanation':
                    # Store explanation if we want to use it
                    card['explanation'] = value
        
        # Only add cards with at least a question and answer
        if 'question' in card and 'answer' in card:
            # Add default hint if missing
            if 'hint' not in card:
                card['hint'] = "Think about the key concepts related to this topic."
            flashcards.append(card)
        else:
            logger.warning(f"Skipping invalid flashcard missing question or answer")
    
    return flashcards

def parse_text_questions(text_content: str) -> List[Dict]:
    """
    Parse quiz questions from a delimited text format into structured dictionaries
    
    Format:
    ---
    Question: What is the capital of France?
    Type: mcq
    Options:
    A. London  
    B. Berlin  
    C. Paris  
    D. Madrid  
    Answer: C  
    Explanation: Paris is the capital city of France.
    ---
    Question: Paris is the capital of France.
    Type: true_false  
    Answer: True  
    Explanation: Paris is officially recognized as the capital of France.
    ---
    """
    questions = []
    # Split by the '---' delimiter, ignoring empty entries
    segments = [s.strip() for s in text_content.split('---') if s.strip()]
    
    for segment in segments:
        question = {}
        
        # Parse each line to extract fields
        lines = segment.strip().split('\n')
        
        # Process options separately
        in_options = False
        options = {}
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if we're entering or leaving options section
            if line.lower() == 'options:':
                in_options = True
                continue
            
            # If we're in options section, parse options
            if in_options:
                # If we find a line that doesn't look like an option, exit options mode
                if not line[0].isalpha() or not line[1:2] in [')', '.', ':']:
                    in_options = False
                else:
                    # Extract option letter and text
                    option_letter = line[0]
                    # Find the separator (., :, or ))
                    separator_idx = max(line.find('.'), max(line.find(':'), line.find(')')))
                    if separator_idx > 0:
                        option_text = line[separator_idx+1:].strip()
                        options[option_letter] = option_text
                    continue
            
            # If we're not in options mode, parse other fields
            colon_index = line.find(':')
            if colon_index > 0:
                field = line[:colon_index].strip().lower()
                value = line[colon_index+1:].strip()
                
                # Map fields to our question structure
                if field == 'question':
                    question['question'] = value
                elif field == 'type':
                    question['type'] = value.lower()
                elif field == 'answer':
                    question['answer'] = value
                elif field == 'explanation':
                    question['explanation'] = value
        
        # Add options to the question if we found any
        if options and len(options) > 0:
            question['options'] = options
        
        # Validate question has required fields
        if 'question' in question and 'answer' in question:
            # Make sure type is set
            if 'type' not in question:
                if 'options' in question:
                    question['type'] = 'mcq'
                else:
                    question['type'] = 'true_false'
                    
            # Normalize types
            if question['type'].lower() in ['multiple choice', 'multiple-choice', 'multiplechoice']:
                question['type'] = 'mcq'
            elif question['type'].lower() in ['true/false', 'truefalse', 't/f', 'tf']:
                question['type'] = 'true_false'
            
            questions.append(question)
        else:
            logger.warning(f"Skipping invalid question missing question or answer")
    
    return questions

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
            
            # Build a clearer and more structured prompt for text-based questions
            prompt = f"""
            TASK: Generate exactly {current_batch} quiz questions based on the provided content.
            Difficulty level: {difficulty.upper()}
            
            QUESTION DISTRIBUTION:
            - Multiple-choice questions: {mcq_count}
            - True/False questions: {tf_count}
            
            FORMATTING REQUIREMENTS:
            - Format each question exactly like these examples, with fields on separate lines:
            
            ---
            Question: What is the capital of France?
            Type: mcq
            Options:
            A. London  
            B. Berlin  
            C. Paris  
            D. Madrid  
            Answer: C  
            Explanation: Paris is the capital city of France, known for its landmarks like the Eiffel Tower and the Louvre.
            ---
            Question: Paris is the capital of France.
            Type: true_false  
            Answer: True  
            Explanation: Paris is officially recognized as the capital of France and is also the country's largest city.
            ---
            
            - Start and end each question with "---" on its own line
            - Include the following fields for each question:
              * Question: The actual question text
              * Type: Either "mcq" or "true_false"
              * Options: For MCQ questions only, list all options with letters A-D
              * Answer: For MCQ, just the letter (A, B, C, D). For True/False, either "True" or "False"
              * Explanation: Brief explanation of the correct answer
            
            1. For MULTIPLE-CHOICE questions ({mcq_count}):
              - Include ONLY ONE correct answer
              - Provide 4 options labeled A, B, C, and D
              - Make sure options are realistic and not obviously wrong
              - Evenly distribute correct answers (don't make all answers "C")
               
            2. For TRUE/FALSE questions ({tf_count}):
              - Format as a statement that is definitively true or false
              - The question field should be a complete statement, not a question
              - Try to have a balance of true and false statements
            
            CONTENT REQUIREMENTS:
            - Questions should be based ONLY on the content provided
            - Focus on key concepts, important facts, and significant details
            - Ensure variety across the questions to cover different aspects of the content
            - Make questions challenging and test deep understanding, not just memorization
            - Avoid ambiguous or opinion-based questions that could have multiple correct answers
            
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
                                        "content": "You are a specialized quiz creator. Your task is to generate questions in the exact format specified, following all the instructions precisely."
                                    },
                                    {"role": "user", "content": prompt}
                                ],
                                temperature=0.7,
                                max_tokens=1500,
                                stream=True
                            )
                            
                            # Process the streaming response
                            async for chunk in async_response:
                                if chunk.choices[0].delta.content:
                                    response_text += chunk.choices[0].delta.content
                        
                            # If we got a valid response, break out of the loop
                            if response_text and '---' in response_text:
                                logger.info(f"Successfully generated questions with Groq model: {model}")
                                break
                            else:
                                logger.warning(f"Empty or invalid response from Groq model: {model}")
                        
                        except Exception as e:
                            logger.warning(f"Error with Groq model {model}: {e}")
                            continue
                
                # Fall back to OpenAI if Groq failed or is not available
                if not response_text or '---' not in response_text:
                    logger.info("Falling back to OpenAI for question generation")
                    async_response = await self.client.chat.completions.create(
                        model="gpt-4o-mini",  # Using a faster model
                        messages=[
                            {
                                "role": "system", 
                                "content": "You are a specialized quiz creator. Your task is to generate questions in the exact format specified, following all the instructions precisely."
                            },
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.7,
                        stream=True
                    )
                    
                    # Clear any partial response from Groq
                    response_text = ""
                    
                    # Process the streaming response
                    async for chunk in async_response:
                        if chunk.choices[0].delta.content:
                            response_text += chunk.choices[0].delta.content

                # Parse the text response into structured questions
                parsed_questions = parse_text_questions(response_text)
                
                # Apply additional validation if needed
                validated_questions = []
                for q in parsed_questions:
                    # Skip if we've reached our target count
                    if len(validated_questions) >= current_batch:
                        break
                        
                    # Verify that question types match what's expected
                    if q["type"] not in ["mcq", "true_false"]:
                        if "options" in q:
                            q["type"] = "mcq"
                        else:
                            q["type"] = "true_false"
                    
                    # Make sure MCQ questions have options
                    if q["type"] == "mcq" and "options" not in q:
                        logger.warning(f"Skipping invalid MCQ question without options: {q.get('question', 'Unknown')}")
                        continue
                    
                    # Make sure true_false questions don't have options
                    if q["type"] == "true_false" and "options" in q:
                        del q["options"]
                    
                    validated_questions.append(q)
                
                num_questions_generated = len(validated_questions)
                questions_generated += num_questions_generated

                # Check the distribution of question types in this batch
                actual_mcq = sum(1 for q in validated_questions if q["type"] == "mcq")
                actual_tf = sum(1 for q in validated_questions if q["type"] == "true_false")
                logger.info(f"Generated {actual_mcq} MCQ and {actual_tf} True/False questions")

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
            
            # Build a simplified prompt for text-based flashcards
            prompt = f"""
            TASK: Generate exactly {current_batch} detailed flashcards based on the provided content.
            Difficulty level: {difficulty.upper()}
            
            IMPORTANT: ALL ANSWERS MUST BE CONCISE - MAXIMUM 1-2 SENTENCES. Keep flashcards brief and focused.
            
            FORMATTING REQUIREMENTS:
            - Format each flashcard exactly like this example, with fields on separate lines:
            ---
            Question: What is the powerhouse of the cell?
            Answer: Mitochondria
            Hint: It's often associated with energy production.
            Explanation: The mitochondria is responsible for producing ATP, the energy currency of the cell.
            ---
            
            - Start and end each flashcard with "---" on its own line
            - Include the following fields for each flashcard:
              * Question: A thought-provoking question
              * Answer: A CONCISE answer (1-2 sentences max)
              * Hint: A brief clue that guides without giving away the answer
              * Explanation: A short explanation that provides additional context (optional)
            
            CONTENT REQUIREMENTS:
            - Flashcards should be based ONLY on the content provided
            - Focus on key concepts, important facts, and significant details
            - BREVITY IS CRITICAL - keep all answers short and to the point
            - Avoid ambiguous answers or questions with multiple possible answers
            
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
                                temperature=0.7,
                                max_tokens=1500,
                                stream=True
                            )
                            
                            # Process the streaming response
                            async for chunk in async_response:
                                if chunk.choices[0].delta.content:
                                    response_text += chunk.choices[0].delta.content
                        
                            # If we got a valid response, break out of the loop
                            if response_text and '---' in response_text:
                                logger.info(f"Successfully generated flashcards with Groq model: {model}")
                                break
                            else:
                                logger.warning(f"Empty or invalid response from Groq model: {model}")
                        
                        except Exception as e:
                            logger.warning(f"Error with Groq model {model}: {e}")
                            continue
                
                # Fall back to OpenAI if Groq failed or is not available
                if not response_text or '---' not in response_text:
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

                # Parse the text response into structured flashcards
                parsed_flashcards = parse_text_flashcards(response_text)
                
                # Apply validation rules for consistency
                validated_flashcards = []
                for card in parsed_flashcards:
                    # Skip if we've reached our target count
                    if len(validated_flashcards) >= current_batch:
                        break
                        
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
                
                num_cards = len(validated_flashcards)
                flashcards_generated += num_cards
                logger.info(f"Generated {num_cards} detailed flashcards")

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