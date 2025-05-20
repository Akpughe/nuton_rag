import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import concurrent.futures
from functools import partial
import random

from chonkie_client import embed_query, embed_query_v2
from pinecone_client import hybrid_search, rerank_results
import openai_client
from supabase_client import update_generated_content, get_generated_content_id, insert_quiz_set, update_generated_content_quiz

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
    question_type: str = "both",  # one of "mcq", "true_false", or "both"
    num_questions: int = 10,
    acl_tags: Optional[List[str]] = None,
    rerank_top_n: int = 50,
    use_openai_embeddings: bool = True,
    set_id: int = 1,
    title: Optional[str] = None,
    description: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generates a quiz from a document using hybrid search, rerank, and GPT-4o.
    Args:
        document_id: Filter results to this document ID.
        space_id: Filter results to this space ID.
        question_type: Type of questions to generate ("mcq", "true_false", or "both").
        num_questions: Total number of questions to generate.
        acl_tags: Optional list of ACL tags to filter by.
        rerank_top_n: Number of results to rerank.
        use_openai_embeddings: Whether to use OpenAI directly for embeddings.
        set_id: Quiz set number.
        title: Optional quiz title.
        description: Optional quiz description.
    Returns:
        Dict with quiz and status.
    """
    # Validate question_type
    if question_type not in ["mcq", "true_false", "both"]:
        raise ValueError("question_type must be one of 'mcq', 'true_false', or 'both'")
    
    # Determine number of each question type
    mcq_count, tf_count = get_question_counts(question_type, num_questions)
    
    logging.info(f"Generating quiz for document {document_id}, space_id: {space_id}, type: {question_type} ({mcq_count} MCQ, {tf_count} TF)")
    
    # Initialize status in database with empty quiz
    update_generated_content_quiz(
        document_id,
        {"quiz": [], "status": "processing", "updated_at": datetime.now().isoformat()}
    )
    
    try:
        start_time = datetime.now()
        
        # Embed query using OpenAI directly
        query = "Extract all key concepts, facts, and relationships from this document for quiz generation."
        query_embedded = embed_query_v2(query) if use_openai_embeddings else embed_query(query)
        if isinstance(query_embedded, dict) and "message" in query_embedded and "status" in query_embedded:
            error_msg = f"Query embedding failed: {query_embedded['message']}"
            logging.error(error_msg)
            update_generated_content_quiz(document_id, {"quiz": [], "status": "error", "updated_at": datetime.now().isoformat()})
            return {"quiz": [], "status": "error", "message": error_msg}
        query_emb = query_embedded["embedding"]
        query_sparse = query_embedded.get("sparse")
        
        # Search using hybrid search to gather relevant content
        hits = hybrid_search(
            query_emb=query_emb,
            query_sparse=query_sparse,
            top_k=rerank_top_n,
            doc_id=document_id,
            space_id=space_id,
            acl_tags=acl_tags
        )
        if not hits:
            error_msg = "No relevant content found."
            update_generated_content_quiz(document_id, {"quiz": [], "status": "error", "updated_at": datetime.now().isoformat()})
            return {"quiz": [], "status": "error", "message": error_msg}
        
        reranked = rerank_results(query, hits, top_n=rerank_top_n)
        context_chunks = []
        for item in reranked:
            if "metadata" in item and "text" in item["metadata"]:
                context_chunks.append(item["metadata"]["text"])
        
        if not context_chunks:
            error_msg = "No text content found in search results."
            update_generated_content_quiz(document_id, {"quiz": [], "status": "error", "updated_at": datetime.now().isoformat()})
            return {"quiz": [], "status": "error", "message": error_msg}
        
        # Generate quiz questions in parallel batches
        quiz_questions = generate_quiz_from_chunks_parallel(
            context_chunks,
            mcq_count=mcq_count,
            tf_count=tf_count
        )
        
        # Format for DB - this is the format expected by the quiz_sets table
        quiz_obj = {
            "set_id": set_id,
            "total_questions": len(quiz_questions),
            "questions": quiz_questions
        }
        
        # Update generated_content table - use the quiz_questions directly as an array
        update_generated_content_quiz(
            document_id,
            {"quiz": quiz_questions, "status": "completed", "updated_at": datetime.now().isoformat()}
        )
        
        # Insert into quiz_sets table
        try:
            content_id = get_generated_content_id(document_id)
            insert_quiz_set(content_id, quiz_obj, set_id, title, description)
        except Exception as e:
            logging.error(f"Error storing quiz set: {e}")
        
        elapsed_time = (datetime.now() - start_time).total_seconds()
        logging.info(f"Quiz generation completed in {elapsed_time:.2f} seconds. Generated {len(quiz_questions)} questions.")
        
        return {
            "quiz": quiz_obj,
            "status": "success",
            "elapsed_seconds": elapsed_time,
            "total_questions": len(quiz_questions),
            "question_type": question_type,
            "mcq_count": mcq_count,
            "tf_count": tf_count
        }
    except Exception as e:
        logging.exception(f"Error generating quiz: {e}")
        update_generated_content_quiz(document_id, {"quiz": [], "status": "error", "updated_at": datetime.now().isoformat()})
        return {"quiz": [], "status": "error", "message": str(e)}

def generate_quiz_from_chunks_parallel(
    context_chunks: List[str],
    mcq_count: int = 5,
    tf_count: int = 5
) -> List[Dict[str, Any]]:
    """
    Generate quiz questions from context chunks in parallel.
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
    
    # Distribute MCQ and TF counts across batches
    mcq_per_batch = max(1, mcq_count // len(batches)) if mcq_count > 0 else 0
    tf_per_batch = max(1, tf_count // len(batches)) if tf_count > 0 else 0
    
    # Handle remainder questions
    mcq_remainder = mcq_count - (mcq_per_batch * len(batches))
    tf_remainder = tf_count - (tf_per_batch * len(batches))
    
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
    
    # Rebalance MCQ and TF to match requested counts
    mcq_questions = [q for q in deduped if q.get("type") == "mcq"]
    tf_questions = [q for q in deduped if q.get("type") == "true_false"]
    
    # Trim to requested counts
    final_mcq = mcq_questions[:mcq_count]
    final_tf = tf_questions[:tf_count]
    
    # Combine and shuffle to mix MCQ and True/False questions
    combined_questions = final_mcq + final_tf
    random.shuffle(combined_questions)
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
    
    # Call LLM
    response = openai_client.client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an expert quiz generator."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=2048
    )
    text = response.choices[0].message.content
    
    # Parse questions
    return parse_quiz_questions(text)

def parse_quiz_questions(text: str) -> List[Dict[str, Any]]:
    """
    Parse quiz questions from LLM output.
    """
    questions = []
    import re
    blocks = re.split(r'---+', text)
    qid = 1
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        
        # Parse type
        type_match = re.search(r'Type:\s*(mcq|true_false)', block, re.IGNORECASE)
        qtype = type_match.group(1).lower() if type_match else None
        
        # Parse question
        q_match = re.search(r'Question:\s*(.+)', block)
        question_text = q_match.group(1).strip() if q_match else None
        
        # Parse options (for MCQ)
        options = []
        if qtype == 'mcq':
            for opt in ['A', 'B', 'C', 'D']:
                opt_match = re.search(rf'{opt}\.\s*(.+)', block)
                if opt_match:
                    options.append({opt.lower(): opt_match.group(1).strip()})
        
        # Parse answer
        ans_match = re.search(r'Answer:\s*([A-D]|True|False)', block)
        correct_option = ans_match.group(1) if ans_match else None
        
        # Parse explanation
        exp_match = re.search(r'Explanation:\s*(.+)', block)
        explanation = exp_match.group(1).strip() if exp_match else None
        
        if qtype and question_text and correct_option and explanation:
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