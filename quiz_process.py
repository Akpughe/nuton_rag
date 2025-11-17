import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import concurrent.futures
from functools import partial
import random

from chonkie_client import embed_query, embed_query_v2, embed_query_multimodal
from pinecone_client import hybrid_search, fetch_all_document_chunks, rerank_results
from groq import Groq
import os
from dotenv import load_dotenv
from supabase_client import update_generated_content, get_generated_content_id, insert_quiz_set, update_generated_content_quiz, get_existing_quizzes, determine_shared_status

load_dotenv()

# Initialize Groq client
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Define the quiz prompt template once
QUIZ_PROMPT_TEMPLATE = '''
OBJECTIVE:

You are an expert academic quiz creator. Your task is to generate a high-quality quiz from the input material to help students study effectively through active recall and critical thinking.

⸻

QUESTION DISTRIBUTION:
{question_distribution}

⸻

FORMATTING REQUIREMENTS:

Each quiz item must be formatted exactly as shown below:

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

Each question must include:
	•	Question: The quiz question or statement
	•	Type: "mcq" or "true_false"
	•	Options (MCQ only): Four labeled options (A, B, C, D)
	•	Answer: One correct letter (for MCQ) or True/False
	•	Explanation: Brief explanation of the answer

⸻

QUALITY RULES FOR QUESTION GENERATION:

For ALL questions:
	1.	No Duplicates: Every question must cover a unique concept. Avoid overlapping or rephrased duplicates.
	2.	Vary the Cognitive Load: Include a mix of:
	•	Basic factual recall
	•	Concept understanding
	•	Application and comparison

⸻

MULTIPLE-CHOICE QUESTIONS:
	•	Include only one correct answer per question.
	•	Provide realistic distractors (wrong answers). Avoid:
	•	Clearly absurd answers
	•	Repetitive phrasing
	•	"All of the above" or "None of the above"
	•	Balance correct answers across A–D as evenly as possible throughout the quiz.
	•	Good Distractor Example:
For a question about the boiling point of water, "90°C" is a better distractor than "1000°C".

⸻

TRUE/FALSE QUESTIONS:
	•	Write as full, factual statements, not questions.
	•	Maintain a balanced mix:
	•	~50% True
	•	~50% False
	•	False statements must be plausible but definitively incorrect, not trivially false.
	•	Example (Good): "Water boils at 120°C at sea level." → False.

⸻

INPUT HANDLING:

If the source material is:
	•	Unstructured (notes, slides, bullets): Reorganize into logical topics before generating questions.
	•	Long: Process content in batches or sections and extract quiz items from each.

⸻

TARGET AUDIENCE:
	•	Students aged 16–25
	•	Preparing for academic exams, tests, or concept reinforcement

⸻

INPUT MATERIAL:
{context}
'''

def get_question_counts(question_type: str, num_questions: int) -> Tuple[int, int]:
    """
    Determine how many questions of each type to generate based on question_type.
    Args:
        question_type: One of "mcq", "true_false", or "both"
        num_questions: Total number of questions to generate
    Returns:
        Tuple of (mcq_count, tf_count)
    """
    if question_type == "mcq":
        return num_questions, 0
    elif question_type == "true_false":
        return 0, num_questions
    else:  # "both"
        # Implement 70% MCQ, 30% True/False split
        mcq_count = int(round(num_questions * 0.7))
        tf_count = num_questions - mcq_count  # Ensures we get exactly num_questions total
        
        # Ensure at least one of each type if we have more than 1 question
        if num_questions > 1:
            if mcq_count == 0:
                mcq_count = 1
                tf_count = num_questions - 1
            elif tf_count == 0:
                tf_count = 1
                mcq_count = num_questions - 1
                
        return mcq_count, tf_count

def get_question_distribution_text(mcq_count: int, tf_count: int) -> str:
    """
    Generate the question distribution text for the prompt with a 70% MCQ / 30% TF split.
    Args:
        mcq_count: Number of MCQ questions
        tf_count: Number of True/False questions
    Returns:
        Formatted text for the prompt
    """
    lines = []
    if mcq_count > 0:
        lines.append(f"•	Multiple-choice questions: {mcq_count} ({mcq_count / (mcq_count + tf_count) * 100:.2f}%)")
    if tf_count > 0:
        lines.append(f"•	True/False questions: {tf_count} ({tf_count / (mcq_count + tf_count) * 100:.2f}%)")
    return "\n".join(lines)

def generate_quiz(
    document_id: str,
    space_id: Optional[str] = None,
    user_id: Optional[str] = None,
    question_type: str = "both",  # one of "mcq", "true_false", or "both"
    num_questions: int = 10,
    acl_tags: Optional[List[str]] = None,
    max_chunks: int = 1000,
    target_coverage: float = 0.80,
    enable_gap_filling: bool = True,
    rerank_top_n: int = 50,
    use_openai_embeddings: bool = False,
    set_id: int = 1,
    title: Optional[str] = None,
    description: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generates a quiz from a document with comprehensive coverage across the entire document.

    Uses fetch_all_document_chunks with gap-filling to ensure quiz questions are distributed
    throughout the document, not just from semantically similar sections.

    Args:
        document_id: Filter results to this document ID.
        space_id: Filter results to this space ID.
        user_id: UUID of the user creating the quiz (for ownership tracking).
        question_type: Type of questions to generate ("mcq", "true_false", or "both").
        num_questions: Total number of questions to generate.
        acl_tags: Optional list of ACL tags to filter by.
        max_chunks: Maximum chunks to retrieve (default: 1000).
        target_coverage: Target coverage percentage (0.0-1.0, default: 0.80).
        enable_gap_filling: Enable intelligent gap-filling (default: True).
        rerank_top_n: Number of results to rerank (deprecated).
        use_openai_embeddings: Whether to use OpenAI directly for embeddings (deprecated).
        set_id: Quiz set number.
        title: Optional quiz title.
        description: Optional quiz description.
    Returns:
        Dict with quiz, status, and coverage metadata.
    """
    # Validate question_type
    if question_type not in ["mcq", "true_false", "both"]:
        raise ValueError("question_type must be one of 'mcq', 'true_false', or 'both'")
    
    # Validate minimum number of questions
    if num_questions < 1:
        raise ValueError("num_questions must be at least 1")
    
    # Determine number of each question type
    mcq_count, tf_count = get_question_counts(question_type, num_questions)
    
    logging.info(f"Generating quiz for document {document_id}, space_id: {space_id}, user_id: {user_id}, type: {question_type} ({mcq_count} MCQ, {tf_count} TF), target_coverage: {target_coverage:.0%}")

    # Log the start of processing (no need to update generated_content.quiz since we're using quiz_sets)
    logging.info(f"Starting quiz generation for document {document_id}, set #{set_id}")

    try:
        start_time = datetime.now()

        # Fetch all chunks with gap-filling for comprehensive coverage
        # This ensures quiz questions are distributed throughout the document
        logging.info(f"Fetching all chunks with target coverage: {target_coverage:.0%}, gap_filling: {enable_gap_filling}")

        chunks = fetch_all_document_chunks(
            document_id=document_id,
            space_id=space_id,
            max_chunks=max_chunks,
            acl_tags=acl_tags,
            target_coverage=target_coverage,
            enable_gap_filling=enable_gap_filling
        )

        if not chunks:
            error_msg = "No chunks found for document."
            return {"quiz": [], "status": "error", "message": error_msg}

        logging.info(f"Retrieved {len(chunks)} chunks (target coverage: {target_coverage:.0%})")

        # Calculate actual coverage from chunks
        from pinecone_client import calculate_coverage_from_chunks
        coverage_result = calculate_coverage_from_chunks(chunks)
        actual_coverage = coverage_result.get("coverage_percentage", 0.0)
        coverage_gaps = len(coverage_result.get("gaps", []))

        logging.info(f"Actual coverage: {actual_coverage:.0%}, gaps: {coverage_gaps}")

        # Extract text from chunks (already sorted by position: chapter→page→start_index)
        context_chunks = []
        for chunk in chunks:
            if "metadata" in chunk and "text" in chunk["metadata"]:
                context_chunks.append(chunk["metadata"]["text"])

        if not context_chunks:
            error_msg = "No text content found in chunks."
            return {"quiz": [], "status": "error", "message": error_msg}
        
        # Generate quiz questions in parallel batches
        quiz_questions = generate_quiz_from_chunks_parallel(
            context_chunks,
            mcq_count=mcq_count,
            tf_count=tf_count
        )
        
        # Verify we have exactly the requested number of questions
        if len(quiz_questions) != num_questions:
            error_msg = f"Generated {len(quiz_questions)} questions but {num_questions} were requested."
            logging.error(error_msg)
            # This should never happen due to the assert in generate_quiz_from_chunks_parallel,
            # but adding this check for extra safety
            return {"quiz": [], "status": "error", "message": error_msg}
        
        # Format for DB - this is the format expected by the quiz_sets table
        quiz_obj = {
            "set_id": set_id,
            "total_questions": len(quiz_questions),
            "questions": quiz_questions
        }
        
        # Insert into quiz_sets table (primary storage)
        try:
            content_id = get_generated_content_id(document_id)
            
            # Determine if this quiz set should be shared based on user ownership
            is_shared = determine_shared_status(user_id, content_id)
            
            logging.info(f"Saving quiz set #{set_id} to quiz_sets table, created_by: {user_id}, is_shared: {is_shared}")
            insert_quiz_set(content_id, quiz_obj, set_id, title, description, created_by=user_id, is_shared=is_shared)
            logging.info(f"Successfully saved quiz set #{set_id} to quiz_sets table")
        except Exception as e:
            logging.error(f"Error storing quiz set: {e}")
            return {"quiz": [], "status": "error", "message": f"Failed to save quiz: {str(e)}"}
        
        elapsed_time = (datetime.now() - start_time).total_seconds()
        logging.info(f"Quiz generation completed in {elapsed_time:.2f} seconds. Generated {len(quiz_questions)} questions.")
        logging.info(f"Coverage: {actual_coverage:.0%}, chunks processed: {len(chunks)}, gaps: {coverage_gaps}")

        return {
            "quiz": quiz_obj,
            "status": "success",
            "elapsed_seconds": elapsed_time,
            "total_questions": len(quiz_questions),
            "question_type": question_type,
            "mcq_count": mcq_count,
            "tf_count": tf_count,
            "set_number": set_id,
            "coverage_metadata": {
                "chunks_processed": len(chunks),
                "text_coverage_percentage": actual_coverage,
                "coverage_gaps": coverage_gaps,
                "target_coverage": target_coverage,
                "gap_filling_enabled": enable_gap_filling
            }
        }
    except Exception as e:
        logging.exception(f"Error generating quiz: {e}")
        return {"quiz": [], "status": "error", "message": str(e)}

def generate_quiz_from_chunks_parallel(
    context_chunks: List[str],
    mcq_count: int = 5,
    tf_count: int = 5
) -> List[Dict[str, Any]]:
    """
    Generate quiz questions from context chunks in parallel.
    Ensures exactly mcq_count + tf_count questions are returned.
    """
    # For simplicity, split MCQ and TF evenly across batches
    chunk_count = len(context_chunks)
    batch_size = 3 if chunk_count > 10 else 2
    batches = []
    for i in range(0, chunk_count, batch_size):
        batch_chunks = context_chunks[i:i+batch_size]
        batch_context = "\n\n=== NEXT SECTION ===\n\n".join(batch_chunks)
        batches.append(batch_context)
    
    # No questions to generate for a specific type
    if mcq_count == 0 and tf_count == 0:
        return []
    
    # Request more questions than needed to account for potential deduplication
    # This helps ensure we'll have enough unique questions
    safety_factor = 1.5
    target_mcq = int(mcq_count * safety_factor) if mcq_count > 0 else 0
    target_tf = int(tf_count * safety_factor) if tf_count > 0 else 0
    
    # Distribute MCQ and TF counts across batches
    mcq_per_batch = max(1, target_mcq // len(batches)) if target_mcq > 0 else 0
    tf_per_batch = max(1, target_tf // len(batches)) if target_tf > 0 else 0
    
    # Handle remainder questions
    mcq_remainder = target_mcq - (mcq_per_batch * len(batches))
    tf_remainder = target_tf - (tf_per_batch * len(batches))
    
    all_questions = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, len(batches))) as executor:
        future_to_batch = {}
        
        # Submit tasks for each batch with appropriate question counts
        for i, batch_context in enumerate(batches):
            # Add remainder questions to early batches
            batch_mcq = mcq_per_batch + (1 if i < mcq_remainder else 0)
            batch_tf = tf_per_batch + (1 if i < tf_remainder else 0)
            
            # Skip if no questions to generate for this batch
            if batch_mcq == 0 and batch_tf == 0:
                continue
                
            future = executor.submit(
                generate_quiz_batch,
                batch_context,
                batch_mcq,
                batch_tf
            )
            future_to_batch[future] = i
        
        # Collect results
        for future in concurrent.futures.as_completed(future_to_batch):
            batch_questions = future.result()
            all_questions.extend(batch_questions)
    
    # Deduplicate by question text
    seen = set()
    deduped = []
    for q in all_questions:
        key = q.get("question_text", "")
        if key and key not in seen:
            seen.add(key)
            deduped.append(q)
    
    # Separate by type
    mcq_questions = [q for q in deduped if q.get("type") == "mcq"]
    tf_questions = [q for q in deduped if q.get("type") == "true_false"]
    
    # Check if we have enough questions of each type
    if len(mcq_questions) < mcq_count or len(tf_questions) < tf_count:
        # Generate additional batches until we have enough questions
        additional_batches_needed = True
        max_attempts = 3  # Limit the number of additional attempts
        attempt = 0
        
        while additional_batches_needed and attempt < max_attempts:
            attempt += 1
            logging.info(f"Generating additional questions (attempt {attempt}): " 
                        f"Have {len(mcq_questions)}/{mcq_count} MCQ, {len(tf_questions)}/{tf_count} TF")
            
            # Calculate how many more questions we need
            more_mcq_needed = max(0, mcq_count - len(mcq_questions))
            more_tf_needed = max(0, tf_count - len(tf_questions))
            
            if more_mcq_needed == 0 and more_tf_needed == 0:
                additional_batches_needed = False
                break
                
            # Use random chunks for additional questions to get variety
            random_batches = random.sample(batches, min(3, len(batches)))
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                futures = []
                for batch_context in random_batches:
                    future = executor.submit(
                        generate_quiz_batch,
                        batch_context,
                        more_mcq_needed,
                        more_tf_needed
                    )
                    futures.append(future)
                
                for future in concurrent.futures.as_completed(futures):
                    extra_questions = future.result()
                    for q in extra_questions:
                        key = q.get("question_text", "")
                        if key and key not in seen:
                            seen.add(key)
                            deduped.append(q)
                            if q.get("type") == "mcq" and len(mcq_questions) < mcq_count:
                                mcq_questions.append(q)
                            elif q.get("type") == "true_false" and len(tf_questions) < tf_count:
                                tf_questions.append(q)
            
            # Check if we have enough after this attempt
            additional_batches_needed = (len(mcq_questions) < mcq_count or len(tf_questions) < tf_count)
    
    # If we still don't have enough questions, duplicate some existing ones with slight modifications
    # This is a last resort to ensure we return exactly the requested number
    if len(mcq_questions) < mcq_count:
        orig_mcq = mcq_questions.copy()
        while len(mcq_questions) < mcq_count:
            # Take a random question and create a variation
            source_q = random.choice(orig_mcq)
            new_q = source_q.copy()
            new_q["question_id"] = f"q{len(deduped) + 1}"
            # Add a prefix to make it unique
            new_q["question_text"] = f"Regarding the same topic: {source_q['question_text']}"
            mcq_questions.append(new_q)
            deduped.append(new_q)
    
    if len(tf_questions) < tf_count:
        orig_tf = tf_questions.copy()
        while len(tf_questions) < tf_count:
            # Take a random question and create a variation
            source_q = random.choice(orig_tf)
            new_q = source_q.copy()
            new_q["question_id"] = f"q{len(deduped) + 1}"
            # Add a prefix to make it unique
            new_q["question_text"] = f"Regarding the same concept: {source_q['question_text']}"
            tf_questions.append(new_q)
            deduped.append(new_q)
    
    # Trim to requested counts (in case we have extras)
    final_mcq = mcq_questions[:mcq_count]
    final_tf = tf_questions[:tf_count]
    
    # Combine and shuffle to mix MCQ and True/False questions
    combined_questions = final_mcq + final_tf
    random.shuffle(combined_questions)
    
    # Verify we have exactly the requested number of questions
    assert len(combined_questions) == mcq_count + tf_count, f"Expected {mcq_count + tf_count} questions, got {len(combined_questions)}"
    
    return combined_questions

def generate_quiz_batch(context: str, mcq_count: int, tf_count: int) -> List[Dict[str, Any]]:
    """
    Generate a batch of quiz questions from a context string.
    """
    # Create the question distribution text
    question_distribution = get_question_distribution_text(mcq_count, tf_count)

    # Generate the prompt for this batch
    prompt = QUIZ_PROMPT_TEMPLATE.format(
        question_distribution=question_distribution,
        context=context
    )

    # Call LLM with Groq
    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": "You are an expert quiz generator."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=2048
    )
    text = response.choices[0].message.content

    # Parse questions
    questions = parse_quiz_questions(text)

    # DIAGNOSTIC: Log if we got 0 questions to debug parsing issues
    if len(questions) == 0:
        import re
        logging.warning(f"⚠️ Quiz batch generated 0 questions. Diagnostic info:")
        logging.warning(f"Response length: {len(text)} chars")
        logging.warning(f"First 600 chars:\n{text[:600]}")
        logging.warning(f"Contains '---': {len(re.findall(r'---+', text))} separators")
        logging.warning(f"Contains 'Type:': {len(re.findall(r'Type:', text, re.I))}")
        logging.warning(f"Contains 'Question:': {len(re.findall(r'Question:', text, re.I))}")
        logging.warning(f"Contains 'Answer:': {len(re.findall(r'Answer:', text, re.I))}")
        logging.warning(f"Contains 'Explanation:': {len(re.findall(r'Explanation:', text, re.I))}")

    return questions

def parse_quiz_questions(text: str) -> List[Dict[str, Any]]:
    """
    Parse quiz questions from LLM output.
    Supports multiline field values for better LLM response handling.
    """
    questions = []
    import re
    blocks = re.split(r'---+', text)
    qid = 1

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        # Parse type using case-insensitive regex
        type_match = re.search(r'Type:\s*(mcq|true_false)', block, re.IGNORECASE | re.DOTALL)
        qtype = type_match.group(1).lower() if type_match else None

        # Parse question - use DOTALL to match multiline content, strip markdown
        # Handle both uppercase and lowercase option letters in boundary
        q_match = re.search(r'Question:\s*(.+?)(?=\n[A-Da-d]\.|Answer:|Explanation:|Type:|Options:|$)', block, re.DOTALL | re.IGNORECASE)
        if q_match:
            question_text = q_match.group(1).strip().replace('\n', ' ')
            # Strip markdown formatting from content
            question_text = re.sub(r'^\*+|\*+$', '', question_text).strip()
        else:
            question_text = None

        # Parse options (for MCQ) - handle multiline options, strip markdown
        # Case-insensitive to handle both A./a., B./b., etc.
        options = []
        if qtype == 'mcq':
            for opt in ['A', 'B', 'C', 'D']:
                # Match from option letter to next option, Answer, or Explanation
                # Case-insensitive to handle lowercase option letters
                opt_pattern = rf'{opt}\.\s*(.+?)(?=\n[A-Da-d]\.|Answer:|Explanation:|Type:|$)'
                opt_match = re.search(opt_pattern, block, re.DOTALL | re.IGNORECASE)
                if opt_match:
                    opt_text = opt_match.group(1).strip().replace('\n', ' ')
                    # Strip markdown formatting
                    opt_text = re.sub(r'^\*+|\*+$', '', opt_text).strip()
                    options.append({opt.lower(): opt_text})

        # Parse answer - keep this single line as answers should be short
        ans_match = re.search(r'Answer:\s*([A-D]|True|False)', block, re.IGNORECASE)
        if ans_match:
            raw_answer = ans_match.group(1)
            # Normalize answer format for consistency
            if raw_answer.upper() in ['A', 'B', 'C', 'D']:
                correct_option = raw_answer.upper()  # MCQ: always uppercase
            elif raw_answer.lower() == 'true':
                correct_option = 'True'  # Boolean: capitalize
            elif raw_answer.lower() == 'false':
                correct_option = 'False'  # Boolean: capitalize
            else:
                correct_option = raw_answer  # Fallback: keep as-is
        else:
            correct_option = None

        # Parse explanation - handle multiline explanations, strip markdown
        exp_match = re.search(r'Explanation:\s*(.+?)(?=\n(?:Type:|Question:|---)|$)', block, re.DOTALL)
        if exp_match:
            explanation = exp_match.group(1).strip().replace('\n', ' ')
            # Strip markdown formatting
            explanation = re.sub(r'^\*+|\*+$', '', explanation).strip()
        else:
            explanation = None

        # Create question object if we have minimum required fields
        if qtype and question_text and correct_option:
            # Additional validation for MCQ questions
            if qtype == 'mcq':
                # Ensure we have at least 2 options (ideally 4)
                if len(options) < 2:
                    logging.warning(f"Skipping MCQ question with insufficient options ({len(options)}): {question_text[:50]}...")
                    continue

                # Verify correct_option exists in options
                correct_key = correct_option.lower()
                option_keys = [list(opt.keys())[0] for opt in options]
                if correct_key not in option_keys:
                    logging.warning(f"Skipping MCQ question - correct answer '{correct_option}' not in options {option_keys}: {question_text[:50]}...")
                    continue

            # Fill in explanation if missing
            if not explanation:
                explanation = "Review the material for context."

            q = {
                "type": qtype,
                "question_id": f"q{qid}",
                "question_text": question_text,
                "correct_option": correct_option,
                "explanation": explanation
            }
            if qtype == 'mcq':
                q["options"] = options
            questions.append(q)
            qid += 1

    return questions

def simplify_quiz_text(text: str) -> str:
    """
    Simplify text for comparison by removing punctuation, extra whitespace,
    and converting to lowercase.
    
    Args:
        text: The text to simplify.
        
    Returns:
        Simplified text.
    """
    import re
    # Remove punctuation and extra whitespace
    simplified = re.sub(r'[^\w\s]', '', text.lower())
    simplified = re.sub(r'\s+', ' ', simplified).strip()
    return simplified


def jaccard_similarity_quiz(set1: set, set2: set) -> float:
    """
    Calculate Jaccard similarity between two sets.
    More efficient implementation for word sets.
    
    Args:
        set1: First set of words.
        set2: Second set of words.
        
    Returns:
        Similarity score between 0 and 1.
    """
    if not set1 or not set2:
        return 0.0
    
    intersection = len(set1.intersection(set2))
    union = len(set1) + len(set2) - intersection
    
    return intersection / union


def is_duplicate_quiz_question(question: Dict[str, Any], existing_questions: List[Dict[str, Any]]) -> bool:
    """
    Check if a quiz question is too similar to any existing question.
    Compares both question text and answer options for MCQ questions.
    
    Args:
        question: The quiz question to check.
        existing_questions: List of existing quiz questions to compare against.
        
    Returns:
        True if the question is a duplicate, False otherwise.
    """
    if not question or not existing_questions:
        return False
        
    question_text = question.get("question_text", "")
    question_type = question.get("type", "")
    
    if not question_text:
        return False
        
    # Simplify for comparison
    simple_question = simplify_quiz_text(question_text)
    question_words = set(simple_question.split())
    
    # Get options for MCQ questions
    question_options = []
    if question_type == "mcq" and "options" in question:
        for opt_dict in question["options"]:
            for opt_text in opt_dict.values():
                question_options.extend(simplify_quiz_text(opt_text).split())
    
    question_options_set = set(question_options)
    
    # Check against all existing questions
    for existing in existing_questions:
        existing_question_text = simplify_quiz_text(existing.get("question_text", ""))
        existing_type = existing.get("type", "")
        
        existing_q_words = set(existing_question_text.split())
        
        # Check question text similarity
        question_similarity = jaccard_similarity_quiz(question_words, existing_q_words)
        
        # If question text is very similar, it's likely a duplicate
        if question_similarity > 0.7:
            return True
        
        # For MCQ questions, also compare options
        if question_type == "mcq" and existing_type == "mcq" and question_options_set:
            existing_options = []
            if "options" in existing:
                for opt_dict in existing["options"]:
                    for opt_text in opt_dict.values():
                        existing_options.extend(simplify_quiz_text(opt_text).split())
            
            existing_options_set = set(existing_options)
            
            # Check options similarity
            if existing_options_set:
                options_similarity = jaccard_similarity_quiz(question_options_set, existing_options_set)
                
                # If both question and options are moderately similar, consider it a duplicate
                if question_similarity > 0.5 and options_similarity > 0.6:
                    return True
    
    return False


def regenerate_quiz(
    document_id: str,
    space_id: Optional[str] = None,
    user_id: Optional[str] = None,
    question_type: str = "both",
    num_questions: int = 10,
    acl_tags: Optional[List[str]] = None,
    rerank_top_n: int = 50,
    use_openai_embeddings: bool = False,
    title: Optional[str] = None,
    description: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generates additional quiz questions from a document, avoiding duplicates from previous sets.
    
    Args:
        document_id: Filter results to this document ID.
        space_id: Filter results to this space ID.
        user_id: UUID of the user creating the quiz (for ownership tracking).
        question_type: Type of questions to generate ("mcq", "true_false", or "both").
        num_questions: Number of additional questions to generate.
        acl_tags: Optional list of ACL tags to filter by.
        rerank_top_n: Number of results to rerank.
        use_openai_embeddings: Whether to use OpenAI directly for embeddings.
        title: Optional quiz title.
        description: Optional quiz description.
    
    Returns:
        Dict with new quiz questions and status.
    """
    # Validate question_type
    if question_type not in ["mcq", "true_false", "both"]:
        raise ValueError("question_type must be one of 'mcq', 'true_false', or 'both'")
    
    # Validate minimum number of questions
    if num_questions < 1:
        raise ValueError("num_questions must be at least 1")
    
    # Determine number of each question type
    mcq_count, tf_count = get_question_counts(question_type, num_questions)
    
    logging.info(f"Regenerating quiz for document {document_id}, space_id: {space_id}, user_id: {user_id}, type: {question_type} ({mcq_count} MCQ, {tf_count} TF)")
    
    # Get content_id for this document
    content_id = get_generated_content_id(document_id)
    
    # Get existing quiz sets and determine the next set number
    existing_quiz_sets = get_existing_quizzes(content_id)
    
    existing_questions = []
    next_set_number = 1
    
    if existing_quiz_sets:
        # Extract all questions from all sets to check for duplicates
        for quiz_set in existing_quiz_sets:
            if isinstance(quiz_set, dict):
                questions = quiz_set.get("questions", [])
                existing_questions.extend(questions)
                
                # Update next_set_number
                set_num = quiz_set.get("set_number", 0)
                if set_num >= next_set_number:
                    next_set_number = set_num + 1
    
    logging.info(f"Found {len(existing_questions)} existing quiz questions across {len(existing_quiz_sets)} sets. Next set number: {next_set_number}")
    
    # Initialize status in database (we don't need to update generated_content.quiz anymore since we're using quiz_sets)
    # Just log the start of processing
    logging.info(f"Starting quiz regeneration for document {document_id}, set #{next_set_number}")
    
    try:
        start_time = datetime.now()
        
        # Create enhanced queries to dig deeper into content
        base_query = "Extract comprehensive quiz-worthy information from this document including all key concepts, facts, and relationships."
        enhanced_queries = [
            f"{base_query} Focus on detailed concepts and specific facts.",
            f"{base_query} Emphasize relationships between topics and underlying principles.",
            f"{base_query} Include practical applications and real-world examples."
        ]
        
        all_hits = []
        for query in enhanced_queries:
            # ALWAYS use multimodal embeddings (1024 dims) to match the document embeddings in Pinecone
            # The use_openai_embeddings parameter is deprecated but kept for backwards compatibility
            query_embedded = embed_query_multimodal(query)
            if isinstance(query_embedded, dict) and "message" in query_embedded and "status" in query_embedded:
                continue  # Skip this query if embedding fails
            
            query_emb = query_embedded["embedding"]
            query_sparse = query_embedded.get("sparse")
            
            # Search using hybrid search with increased top_k for broader coverage
            hits = hybrid_search(
                query_emb=query_emb,
                query_sparse=query_sparse,
                top_k=rerank_top_n * 2,  # Increased for more diverse content
                doc_id=document_id,
                space_id=space_id,
                acl_tags=acl_tags
            )
            if hits:
                all_hits.extend(hits)
        
        if not all_hits:
            error_msg = "No relevant content found for quiz regeneration."
            return {"quiz": [], "status": "error", "message": error_msg}
        
        # Remove duplicates and rerank with enhanced parameters
        unique_hits = []
        seen_texts = set()
        for hit in all_hits:
            text = hit.get("metadata", {}).get("text", "")
            if text and text not in seen_texts:
                seen_texts.add(text)
                unique_hits.append(hit)
        
        # Use the first enhanced query for reranking
        reranked = rerank_results(enhanced_queries[0], unique_hits, top_n=min(len(unique_hits), rerank_top_n * 2))
        
        context_chunks = []
        for item in reranked:
            if "metadata" in item and "text" in item["metadata"]:
                context_chunks.append(item["metadata"]["text"])
        
        if not context_chunks:
            error_msg = "No text content found in search results for quiz regeneration."
            return {"quiz": [], "status": "error", "message": error_msg}
        
        # Generate new quiz questions, avoiding duplicates with existing ones
        new_questions = generate_quiz_from_chunks_parallel_with_existing(
            context_chunks,
            mcq_count=mcq_count,
            tf_count=tf_count,
            existing_questions=existing_questions
        )
        
        # Verify we have exactly the requested number of questions
        if len(new_questions) != num_questions:
            error_msg = f"Generated {len(new_questions)} questions but {num_questions} were requested."
            logging.error(error_msg)
            return {"quiz": [], "status": "error", "message": error_msg}
        
        # Format for DB - this is the format expected by the quiz_sets table
        quiz_obj = {
            "set_id": next_set_number,
            "total_questions": len(new_questions),
            "questions": new_questions
        }
        
        # Insert the new set to quiz_sets table
        try:
            # Determine if this quiz set should be shared based on user ownership
            is_shared = determine_shared_status(user_id, content_id)
            
            logging.info(f"Inserting new quiz set #{next_set_number} with {len(new_questions)} questions, created_by: {user_id}, is_shared: {is_shared}")
            insert_quiz_set(content_id, quiz_obj, next_set_number, title, description, created_by=user_id, is_shared=is_shared)
        except Exception as e:
            logging.error(f"Error storing new quiz set: {e}")
        
        elapsed_time = (datetime.now() - start_time).total_seconds()
        logging.info(f"Additional quiz generation completed in {elapsed_time:.2f} seconds. Generated {len(new_questions)} new questions.")
        
        return {
            "quiz": quiz_obj,
            "status": "success",
            "elapsed_seconds": elapsed_time,
            "total_questions": len(new_questions),
            "question_type": question_type,
            "mcq_count": mcq_count,
            "tf_count": tf_count,
            "set_number": next_set_number
        }
        
    except Exception as e:
        logging.exception(f"Error regenerating quiz: {e}")
        return {"quiz": [], "status": "error", "message": str(e)}


def generate_quiz_from_chunks_parallel_with_existing(
    context_chunks: List[str],
    mcq_count: int = 5,
    tf_count: int = 5,
    existing_questions: Optional[List[Dict[str, Any]]] = None
) -> List[Dict[str, Any]]:
    """
    Generate quiz questions from context chunks in parallel, avoiding duplicates with existing questions.
    Ensures exactly mcq_count + tf_count questions are returned.
    """
    if existing_questions is None:
        existing_questions = []
    
    # Use the enhanced version of the original function with duplicate checking
    chunk_count = len(context_chunks)
    batch_size = 3 if chunk_count > 10 else 2
    batches = []
    for i in range(0, chunk_count, batch_size):
        batch_chunks = context_chunks[i:i+batch_size]
        batch_context = "\n\n=== NEXT SECTION ===\n\n".join(batch_chunks)
        batches.append(batch_context)
    
    if mcq_count == 0 and tf_count == 0:
        return []
    
    # Request more questions than needed to account for deduplication
    safety_factor = 2.0  # Increased safety factor for regeneration
    target_mcq = int(mcq_count * safety_factor) if mcq_count > 0 else 0
    target_tf = int(tf_count * safety_factor) if tf_count > 0 else 0
    
    # Distribute counts across batches
    mcq_per_batch = max(1, target_mcq // len(batches)) if target_mcq > 0 else 0
    tf_per_batch = max(1, target_tf // len(batches)) if target_tf > 0 else 0
    
    # Handle remainder questions
    mcq_remainder = target_mcq - (mcq_per_batch * len(batches))
    tf_remainder = target_tf - (tf_per_batch * len(batches))
    
    all_questions = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, len(batches))) as executor:
        future_to_batch = {}
        
        for i, batch_context in enumerate(batches):
            batch_mcq = mcq_per_batch + (1 if i < mcq_remainder else 0)
            batch_tf = tf_per_batch + (1 if i < tf_remainder else 0)
            
            if batch_mcq == 0 and batch_tf == 0:
                continue
                
            future = executor.submit(
                generate_quiz_batch_with_existing,
                batch_context,
                batch_mcq,
                batch_tf,
                existing_questions
            )
            future_to_batch[future] = i
        
        for future in concurrent.futures.as_completed(future_to_batch):
            batch_questions = future.result()
            all_questions.extend(batch_questions)
    
    # Deduplicate and filter out questions similar to existing ones
    seen = set()
    deduped = []
    for q in all_questions:
        key = q.get("question_text", "")
        if key and key not in seen and not is_duplicate_quiz_question(q, existing_questions):
            seen.add(key)
            deduped.append(q)
    
    # Separate by type
    mcq_questions = [q for q in deduped if q.get("type") == "mcq"]
    tf_questions = [q for q in deduped if q.get("type") == "true_false"]
    
    # Generate additional questions if needed
    max_attempts = 3
    attempt = 0
    
    while (len(mcq_questions) < mcq_count or len(tf_questions) < tf_count) and attempt < max_attempts:
        attempt += 1
        logging.info(f"Generating additional questions for regeneration (attempt {attempt}): " 
                    f"Have {len(mcq_questions)}/{mcq_count} MCQ, {len(tf_questions)}/{tf_count} TF")
        
        more_mcq_needed = max(0, mcq_count - len(mcq_questions))
        more_tf_needed = max(0, tf_count - len(tf_questions))
        
        if more_mcq_needed == 0 and more_tf_needed == 0:
            break
        
        # Use random chunks for variety
        random_batches = random.sample(batches, min(3, len(batches)))
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = []
            for batch_context in random_batches:
                future = executor.submit(
                    generate_quiz_batch_with_existing,
                    batch_context,
                    more_mcq_needed,
                    more_tf_needed,
                    existing_questions + deduped  # Include already generated questions
                )
                futures.append(future)
            
            for future in concurrent.futures.as_completed(futures):
                extra_questions = future.result()
                for q in extra_questions:
                    key = q.get("question_text", "")
                    if key and key not in seen and not is_duplicate_quiz_question(q, existing_questions + deduped):
                        seen.add(key)
                        deduped.append(q)
                        if q.get("type") == "mcq" and len(mcq_questions) < mcq_count:
                            mcq_questions.append(q)
                        elif q.get("type") == "true_false" and len(tf_questions) < tf_count:
                            tf_questions.append(q)
    
    # Final selection and shuffling
    final_mcq = mcq_questions[:mcq_count]
    final_tf = tf_questions[:tf_count]
    
    combined_questions = final_mcq + final_tf
    random.shuffle(combined_questions)
    
    logging.info(f"Regeneration produced {len(combined_questions)} unique questions avoiding {len(existing_questions)} existing ones")
    
    return combined_questions


def generate_quiz_batch_with_existing(
    context: str, 
    mcq_count: int, 
    tf_count: int, 
    existing_questions: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Generate a batch of quiz questions from a context string, avoiding duplicates with existing questions.
    """
    # Create enhanced question distribution text with duplicate avoidance
    question_distribution = get_question_distribution_text(mcq_count, tf_count)
    
    # Add duplicate avoidance instruction
    duplicate_avoidance_prompt = ""
    if existing_questions:
        duplicate_avoidance_prompt = f"\n\nIMPORTANT: Generate COMPLETELY DIFFERENT questions from the {len(existing_questions)} existing questions. Focus on different concepts, facts, and details that haven't been covered yet. Avoid creating questions about the same topics or using similar phrasing."
    
    # Generate the enhanced prompt for this batch
    prompt = QUIZ_PROMPT_TEMPLATE.format(
        question_distribution=question_distribution + duplicate_avoidance_prompt,
        context=context
    )
    
    # Call LLM with enhanced instructions using Groq
    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": "You are an expert quiz generator specializing in creating unique, non-duplicate questions."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.8,  # Slightly higher temperature for more variety
        max_tokens=2048
    )
    text = response.choices[0].message.content
    
    # Parse questions
    return parse_quiz_questions(text) 