import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import asyncio
import concurrent.futures
from functools import partial

from chonkie_client import embed_query, embed_query_v2, embed_query_multimodal
from pinecone_client import hybrid_search
from pinecone_client import rerank_results
import openai_client
from supabase_client import update_generated_content, get_generated_content_id, insert_flashcard_set, get_existing_flashcards

def generate_flashcards(
    document_id: str,
    space_id: Optional[str] = None,
    num_questions: Optional[int] = None,
    acl_tags: Optional[List[str]] = None,
    rerank_top_n: int = 50,
    use_openai_embeddings: bool = False
) -> Dict[str, Any]:
    """
    Generates flashcards from a document using hybrid search, rerank, and GPT-4o.
    
    Args:
        document_id: Filter results to this document ID.
        space_id: Filter results to this space ID.
        num_questions: Optional number of flashcards to generate.
        acl_tags: Optional list of ACL tags to filter by.
        rerank_top_n: Number of results to rerank.
        use_openai_embeddings: Whether to use OpenAI directly for embeddings.
    
    Returns:
        Dict with flashcards and status.
    """
    logging.info(f"Generating flashcards for document {document_id}, space_id: {space_id}")
    
    # Initialize status in database with empty set
    update_generated_content(
        document_id,
        {"flashcards": [{"set_id": 1, "cards": []}], "status": "processing", "updated_at": datetime.now().isoformat()}
    )

    try:
        start_time = datetime.now()

        # Create a query to gather content for flashcards
        query = "Extract comprehensive information from this document including all key concepts, facts, definitions, examples, relationships between topics, methodologies, processes, theories, historical context, practical applications, edge cases, and underlying principles. Cover all sections and subtopics thoroughly to ensure complete coverage of the material for high-quality and diverse flashcard generation."

        # Embed query using multimodal embeddings (1024 dims) by default, or OpenAI (1536 dims) if specified
        query_embedded = embed_query_v2(query) if use_openai_embeddings else embed_query_multimodal(query)
        
        # Check for embedding errors
        if isinstance(query_embedded, dict) and "message" in query_embedded and "status" in query_embedded:
            error_msg = f"Query embedding failed: {query_embedded['message']}"
            logging.error(error_msg)
            update_error_status(document_id, error_msg)
            return {"flashcards": [], "status": "error", "message": error_msg}
        
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
            update_error_status(document_id, error_msg)
            return {"flashcards": [], "status": "error", "message": error_msg}
        
        # Rerank results to get most relevant content
        reranked = rerank_results(query, hits, top_n=rerank_top_n)
        
        # Prepare context for flashcard generation
        context_chunks = []
        for item in reranked:
            if "metadata" in item and "text" in item["metadata"]:
                context_chunks.append(item["metadata"]["text"])
        
        if not context_chunks:
            error_msg = "No text content found in search results."
            update_error_status(document_id, error_msg)
            return {"flashcards": [], "status": "error", "message": error_msg}
        
        # Generate flashcards chunk by chunk (parallel processing)
        logging.info(f"Processing {len(context_chunks)} chunks for flashcard generation")
        
        # Create a shared state for tracking accumulated flashcards and database updates
        shared_state = {
            "accumulated_cards": [],
            "set_id": 1,
            "update_threshold": 5,  # Update DB every 5 flashcards
            "last_update_count": 0,
            "document_id": document_id,
            "last_update_time": datetime.now()
        }
        
        flashcards = generate_flashcards_from_chunks_parallel(
            context_chunks, 
            num_questions=num_questions,
            shared_state=shared_state
        )
        
        # Final update to the database with all flashcards (after deduplication)
        # Use the new format
        update_generated_content(
            document_id,
            {
                "flashcards": [
                    {
                        "set_id": shared_state["set_id"],
                        "cards": flashcards
                    }
                ], 
                "status": "completed", 
                "updated_at": datetime.now().isoformat()
            }
        )
        
        # Now insert the complete set to flashcard_sets
        try:
            # Get content_id for this document
            content_id = get_generated_content_id(document_id)
            
            # Insert the complete set
            logging.info(f"Inserting complete flashcard set with {len(flashcards)} cards")
            insert_flashcard_set(content_id, flashcards, shared_state["set_id"])
        except Exception as e:
            logging.error(f"Error storing complete flashcard set: {e}")
        
        elapsed_time = (datetime.now() - start_time).total_seconds()
        logging.info(f"Flashcard generation completed in {elapsed_time:.2f} seconds. Generated {len(flashcards)} flashcards.")
        
        return {
            "flashcards": [
                {
                    "set_id": shared_state["set_id"],
                    "cards": flashcards
                }
            ], 
            "status": "success", 
            "elapsed_seconds": elapsed_time, 
            "total_flashcards": len(flashcards), 
            "num_questions": num_questions
        }
        
    except Exception as e:
        logging.exception(f"Error generating flashcards: {e}")
        update_error_status(document_id, str(e))
        return {"flashcards": [], "status": "error", "message": str(e)}


def generate_flashcards_from_chunks_parallel(
    context_chunks: List[str], 
    num_questions: Optional[int] = None,
    shared_state: Optional[Dict[str, Any]] = None
) -> List[Dict[str, str]]:
    """
    Process chunks in parallel to generate flashcards more efficiently.
    
    Args:
        context_chunks: List of text chunks to process.
        num_questions: Optional target number of flashcards to generate.
        shared_state: Optional shared state for incremental DB updates.
        
    Returns:
        List of flashcard objects with question, answer, hint, and explanation.
    """
    min_flashcards = 30
    if num_questions and num_questions > min_flashcards:
        min_flashcards = num_questions
    
    # Optimize batch size based on number of chunks
    chunk_count = len(context_chunks)
    
    # Adaptive batch sizing - larger batches for fewer chunks, smaller for many chunks
    if chunk_count <= 10:
        batch_size = 5  # Larger batches for small documents
    elif chunk_count <= 20:
        batch_size = 4  # Medium batch size
    else:
        batch_size = 3  # Smaller batches for large documents
    
    batches_count = (chunk_count + batch_size - 1) // batch_size  # Ceiling division
    
    # Request slightly more flashcards than needed since some will be duplicates
    cards_per_batch = max(5, (min_flashcards + batches_count - 1) // batches_count)
    # Add extra for deduplication loss
    cards_per_batch = int(cards_per_batch * 1.2)  
    
    logging.info(f"Using batch size of {batch_size} chunks, targeting {cards_per_batch} flashcards per batch across {batches_count} batches")
    
    # Prepare batches
    batches = []
    for i in range(0, chunk_count, batch_size):
        end_idx = min(i + batch_size, chunk_count)
        batch_chunks = context_chunks[i:end_idx]
        batch_context = "\n\n=== NEXT SECTION ===\n\n".join(batch_chunks)
        batches.append({
            "context": batch_context,
            "batch_number": len(batches) + 1,
            "cards_per_batch": cards_per_batch
        })
    
    all_flashcards = []
    
    # Process batches in parallel using ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, len(batches))) as executor:
        # Create a partial function with the constant parameters
        process_batch_fn = partial(process_batch, shared_state=shared_state)
        
        # Map the function to all batches
        batch_results = list(executor.map(
            process_batch_fn, 
            batches,
            [batches_count] * len(batches)  # Total batches count for progress info
        ))
        
        # Collect all results
        for result in batch_results:
            all_flashcards.extend(result)
    
    # Optimized deduplication with progressive filtering
    deduplicated_flashcards = fast_deduplicate_flashcards(all_flashcards)
    
    # If we still have more flashcards than requested, trim to the target number
    if num_questions and len(deduplicated_flashcards) > num_questions:
        deduplicated_flashcards = deduplicated_flashcards[:num_questions]
    
    logging.info(f"Final flashcard count after deduplication: {len(deduplicated_flashcards)}")
    
    return deduplicated_flashcards


def process_batch(
    batch_data: Dict[str, Any], 
    total_batches: int,
    shared_state: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Process a single batch of content to generate flashcards.
    Designed to be run in parallel.
    
    Args:
        batch_data: Dictionary containing batch information.
        total_batches: Total number of batches for progress tracking.
        shared_state: Optional shared state for incremental DB updates.
        
    Returns:
        List of flashcard objects for this batch.
    """
    batch_number = batch_data["batch_number"]
    context = batch_data["context"]
    cards_per_batch = batch_data["cards_per_batch"]
    
    # Use progress numbering in the prompt to help track completeness
    progress_info = f"[Processing batch {batch_number} of {total_batches}]"
    logging.info(f"Generating flashcards for batch {batch_number}/{total_batches}")
    
    # Check if we need to avoid duplicates
    existing_flashcards = shared_state.get("existing_flashcards", []) if shared_state else []
    
    # Generate flashcards for this batch using streaming
    batch_flashcards = []
    
    def card_callback(cards):
        """Callback function to process cards as they're generated and update the database."""
        nonlocal batch_flashcards
        if not cards:
            return
            
        # Filter out duplicates if needed
        if existing_flashcards:
            unique_cards = []
            for card in cards:
                # Skip if this card is too similar to any existing card
                if not is_duplicate_of_existing(card, existing_flashcards):
                    unique_cards.append(card)
            
            # Only add unique cards
            batch_flashcards.extend(unique_cards)
        else:
            batch_flashcards.extend(cards)
        
        # If we have shared state, update the accumulated flashcards and possibly the DB
        if shared_state:
            # Add to accumulated cards (either filtered or all)
            if existing_flashcards:
                # We've already filtered the cards above
                cards_to_add = batch_flashcards[-len(unique_cards if 'unique_cards' in locals() else cards):]
                shared_state["accumulated_cards"].extend(cards_to_add)
            else:
                shared_state["accumulated_cards"].extend(cards)
                
            current_count = len(shared_state["accumulated_cards"])
            
            # Check if we should update the database
            cards_since_last_update = current_count - shared_state.get("last_update_count", 0)
            time_since_last_update = (datetime.now() - shared_state.get("last_update_time", datetime.min)).total_seconds()
            
            # Update DB if we have enough new cards or enough time has passed
            if (cards_since_last_update >= shared_state["update_threshold"] or time_since_last_update > 10):
                try:
                    # Update the database with the current accumulated flashcards
                    logging.info(f"Incremental DB update: {current_count} flashcards accumulated so far")
                    
                    # Get existing flashcards data
                    document_id = shared_state["document_id"]
                    content_id = get_generated_content_id(document_id)
                    existing_flashcards_data = get_existing_flashcards(content_id)
                    
                    # Create the updated flashcards structure
                    if existing_flashcards_data:
                        # Get sets before the current one
                        previous_sets = [s for s in existing_flashcards_data if s.get("set_id", 0) < shared_state["set_id"]]
                        
                        # Add the current in-progress set
                        flashcards_structure = previous_sets + [{
                            "set_id": shared_state["set_id"],
                            "cards": shared_state["accumulated_cards"]
                        }]
                    else:
                        # If no existing sets, just create one
                        flashcards_structure = [{
                            "set_id": shared_state["set_id"],
                            "cards": shared_state["accumulated_cards"]
                        }]
                    
                    # Update the generated_content table with new format (in progress status)
                    update_generated_content(
                        shared_state["document_id"],
                        {
                            "flashcards": flashcards_structure,
                            "status": "processing", 
                            "updated_at": datetime.now().isoformat()
                        }
                    )
                    
                    # Update tracking info
                    shared_state["last_update_count"] = current_count
                    shared_state["last_update_time"] = datetime.now()
                except Exception as e:
                    logging.error(f"Error during incremental database update: {e}")
    
    # Generate flashcards using the streaming API
    from openai import Stream
    
    # Add additional instruction if we have existing flashcards
    duplicate_avoidance_prompt = ""
    if existing_flashcards:
        duplicate_avoidance_prompt = f"\n\nIMPORTANT: Generate COMPLETELY DIFFERENT flashcards from the previous set. Avoid creating cards that cover the same concepts or facts already covered in the {len(existing_flashcards)} existing flashcards."
    
    system_prompt = f"You are an AI flashcard generator specialized in creating precise, high-quality flashcards for specific sections of content. Generate EXACTLY {cards_per_batch} unique flashcards from the provided content section.{duplicate_avoidance_prompt}"
    
    # Define the flashcard generation prompt
    FLASHCARD_PROMPT = f"""
:books: ADVANCED FLASHCARD GENERATOR — OPTIMIZED FOR STUDY RETENTION {progress_info}

You are an elite academic tutor and flashcard specialist. Your task is to extract exactly {cards_per_batch} unique and high-quality flashcards from the input material, designed to maximize retention, comprehension, and long-term learning.

⸻

OBJECTIVES:
	•	Extract key concepts, facts, principles, and comparisons.
	•	Ensure variety in cognitive depth (recall, understanding, application).
	•	Every flashcard must cover a distinct idea within THIS SPECIFIC CONTENT SECTION.
	•	Focus on quality and specificity - create cards that precisely capture the content of this section.
{duplicate_avoidance_prompt}

⸻

FORMAT — STRICTLY FOLLOW THIS STRUCTURE FOR EACH FLASHCARD:

---
Question: [Clear, specific, and quiz-ready. Can be multiple choice, true/false, or open-ended.]
Answer: [Concise, 1–2 sentences max. Exact and unambiguous.]
Hint: [A precise clue to aid memory. Should make the student think, not give away the answer.]
Explanation: [Brief (1–3 sentences) but powerful clarification. Can include examples, context, or why it matters.]
---

Every flashcard must contain all four fields. No field should be left blank.

⸻

INTELLIGENCE RULES FOR QUALITY:
	1.	Focus on extracting SPECIFIC information from THIS section, not general knowledge.
	2.	Create a diverse mix of factual, conceptual, and applied knowledge cards.
	3.	Don't create cards for information not contained in this section.
	4.	Hint Quality:
	•	Good: "It's the process used in cell division."
	•	Bad: "Think hard" or "This is easy."
	5.	Explanation Quality:
	•	Should clarify why the answer is correct.
	•	May include quick analogies or real-world relevance.

⸻

TONE & AUDIENCE:
	•	Suitable for motivated learners, ages 16–25.
	•	Avoid academic jargon unless necessary.
	•	Be clear, focused, and study-driven.

⸻

INPUT MATERIAL:
{context}

⸻
"""
    # Process streaming response to detect complete flashcards
    response_text = ""
    accumulated_card_text = ""
    card_count = 0
    pending_batch = []
    
    # Start the streaming generation
    response = openai_client.client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": FLASHCARD_PROMPT}
        ],
        temperature=0.8,
        max_tokens=8000,
        stream=True
    )
    
    for chunk in response:
        if chunk.choices[0].delta.content:
            delta = chunk.choices[0].delta.content
            response_text += delta
            accumulated_card_text += delta
            
            # Look for completed flashcards - they end with "---"
            if "---" in delta:
                # Extract flashcards from accumulated text
                cards = parse_streaming_content(accumulated_card_text)
                if cards:
                    card_count += len(cards)
                    pending_batch.extend(cards)
                    
                    # Call callback every 3-5 cards
                    if len(pending_batch) >= 5:
                        card_callback(pending_batch)
                        pending_batch = []
                    
                    # Reset accumulated text but keep any content after the last "---"
                    parts = accumulated_card_text.split("---")
                    accumulated_card_text = "" if len(parts) <= 1 else f"---{parts[-1]}"
    
    # Process any remaining cards in the final batch
    if accumulated_card_text:
        final_cards = parse_streaming_content(accumulated_card_text)
        if final_cards:
            pending_batch.extend(final_cards)
    
    # Send any remaining pending cards
    if pending_batch:
        card_callback(pending_batch)
    
    # Final parsing of the complete response to ensure we catch everything
    all_cards = openai_client.parse_flashcards(response_text)
    
    # Filter out duplicates of existing flashcards if needed
    if existing_flashcards:
        all_cards = [card for card in all_cards if not is_duplicate_of_existing(card, existing_flashcards)]
    
    logging.info(f"Generated {len(all_cards)} flashcards from batch {batch_number}")
    return all_cards


def parse_streaming_content(text: str) -> List[Dict[str, str]]:
    """
    Parse streaming content to extract complete flashcards.
    
    Args:
        text: The accumulated text from streaming.
        
    Returns:
        List of complete flashcard objects.
    """
    # Split by the "---" marker
    parts = text.split("---")
    
    # Skip the first part (it's usually empty or intro text)
    parts = parts[1:] if len(parts) > 1 else []
    
    flashcards = []
    
    for i in range(len(parts) - 1):  # Skip the last part as it might be incomplete
        card_text = parts[i].strip()
        if not card_text:
            continue
            
        card = {}
        
        # Check for required fields
        has_question = "Question:" in card_text
        has_answer = "Answer:" in card_text
        has_hint = "Hint:" in card_text
        has_explanation = "Explanation:" in card_text
        
        if has_question and has_answer and has_hint and has_explanation:
            # Extract fields
            lines = card_text.split("\n")
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                if line.startswith("Question:"):
                    card["question"] = line.replace("Question:", "", 1).strip()
                elif line.startswith("Answer:"):
                    card["answer"] = line.replace("Answer:", "", 1).strip()
                elif line.startswith("Hint:"):
                    card["hint"] = line.replace("Hint:", "", 1).strip()
                elif line.startswith("Explanation:"):
                    card["explanation"] = line.replace("Explanation:", "", 1).strip()
            
            # Only add complete cards
            if all(k in card for k in ["question", "answer", "hint", "explanation"]):
                flashcards.append(card)
    
    return flashcards


def fast_deduplicate_flashcards(flashcards: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Optimized deduplication with better performance. Uses a multi-stage approach
    with exact matching first, then similarity for edge cases.
    
    Args:
        flashcards: List of flashcard objects to deduplicate.
        
    Returns:
        Deduplicated list of flashcard objects.
    """
    if not flashcards:
        return []
    
    # Phase 1: Quick exact duplicate removal using sets
    unique_questions = set()
    phase1_cards = []
    
    for card in flashcards:
        # Create simplified version of question for exact matching
        simple_question = simplify_text(card.get("question", ""))
        
        # Skip exact duplicates
        if simple_question and simple_question not in unique_questions:
            unique_questions.add(simple_question)
            phase1_cards.append(card)
    
    logging.info(f"Phase 1 deduplication: {len(flashcards)} → {len(phase1_cards)}")
    
    # Phase 2: For smaller result sets, do full similarity check
    # Skip the expensive similarity check if we have a lot of unique cards
    if len(phase1_cards) <= 100:  # Only do similarity check for reasonable sizes
        unique_flashcards = []
        question_vectors = {}  # Cache for word vectors
        
        for card in phase1_cards:
            question = card.get("question", "")
            if not question:
                continue
                
            # Get vector for current question
            if question not in question_vectors:
                simple_q = simplify_text(question)
                word_set = set(simple_q.split())
                question_vectors[question] = word_set
            
            current_vector = question_vectors[question]
            
            # Check for high similarity with existing cards
            is_duplicate = False
            for existing_card in unique_flashcards:
                existing_q = existing_card.get("question", "")
                
                # Get vector for existing question
                if existing_q not in question_vectors:
                    simple_existing = simplify_text(existing_q)
                    word_set = set(simple_existing.split())
                    question_vectors[existing_q] = word_set
                
                existing_vector = question_vectors[existing_q]
                
                # Calculate Jaccard similarity
                if jaccard_similarity(current_vector, existing_vector) > 0.7:  # Slightly lower threshold
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_flashcards.append(card)
        
        logging.info(f"Phase 2 deduplication: {len(phase1_cards)} → {len(unique_flashcards)}")
        return unique_flashcards
    else:
        # Skip phase 2 for large result sets to avoid performance issues
        logging.info(f"Skipping similarity deduplication for large result set ({len(phase1_cards)} cards)")
        return phase1_cards


def simplify_text(text: str) -> str:
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


def jaccard_similarity(set1: set, set2: set) -> float:
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


def update_error_status(document_id: str, error_message: str) -> None:
    """
    Update the database with an error status.
    
    Args:
        document_id: The document ID to update.
        error_message: The error message to store.
    """
    update_generated_content(
        document_id,
        {
            "flashcards": [{"set_id": 1, "cards": []}],
            "status": "error", 
            "error_message": error_message, 
            "updated_at": datetime.now().isoformat()
        }
    )


def is_duplicate_of_existing(card: Dict[str, str], existing_cards: List[Dict[str, str]]) -> bool:
    """
    Check if a card is too similar to any existing card.
    
    Args:
        card: The flashcard to check.
        existing_cards: List of existing flashcards to compare against.
        
    Returns:
        True if the card is a duplicate, False otherwise.
    """
    if not card or not existing_cards:
        return False
        
    question = card.get("question", "")
    answer = card.get("answer", "")
    
    if not question or not answer:
        return False
        
    # Simplify for comparison
    simple_question = simplify_text(question)
    simple_answer = simplify_text(answer)
    
    question_words = set(simple_question.split())
    answer_words = set(simple_answer.split())
    
    # Check against all existing cards
    for existing in existing_cards:
        existing_question = simplify_text(existing.get("question", ""))
        existing_answer = simplify_text(existing.get("answer", ""))
        
        existing_q_words = set(existing_question.split())
        existing_a_words = set(existing_answer.split())
        
        # Check for high similarity in either question or answer
        question_similarity = jaccard_similarity(question_words, existing_q_words)
        answer_similarity = jaccard_similarity(answer_words, existing_a_words)
        
        # If either question or answer is very similar, consider it a duplicate
        if question_similarity > 0.7 or answer_similarity > 0.7:
            return True
    
    return False


def regenerate_flashcards(
    document_id: str,
    space_id: Optional[str] = None,
    num_questions: Optional[int] = None,
    acl_tags: Optional[List[str]] = None,
    rerank_top_n: int = 50,
    use_openai_embeddings: bool = False
) -> Dict[str, Any]:
    """
    Generates additional flashcards from a document, avoiding duplicates from previous sets.
    
    Args:
        document_id: Filter results to this document ID.
        space_id: Filter results to this space ID.
        num_questions: Optional number of additional flashcards to generate.
        acl_tags: Optional list of ACL tags to filter by.
        rerank_top_n: Number of results to rerank.
        use_openai_embeddings: Whether to use OpenAI directly for embeddings.
    
    Returns:
        Dict with new flashcards and status.
    """
    logging.info(f"Regenerating flashcards for document {document_id}, space_id: {space_id}")
    
    # Get content_id for this document
    content_id = get_generated_content_id(document_id)
    
    # Get existing flashcards and determine the next set number
    existing_flashcards_data = get_existing_flashcards(content_id)
    
    existing_flashcards = []
    next_set_number = 1
    
    if existing_flashcards_data:
        # Extract all flashcards to check for duplicates
        for set_data in existing_flashcards_data:
            existing_flashcards.extend(set_data.get("cards", []))
            
            # Update next_set_number if needed
            if set_data.get("set_id", 0) >= next_set_number:
                next_set_number = set_data.get("set_id", 0) + 1
    
    logging.info(f"Found {len(existing_flashcards)} existing flashcards. Next set number: {next_set_number}")
    
    # Initialize status in database with empty set for the new set_id
    current_flashcards = existing_flashcards_data if existing_flashcards_data else []
    current_flashcards.append({"set_id": next_set_number, "cards": []})
    
    update_generated_content(
        document_id,
        {"flashcards": current_flashcards, "status": "processing", "updated_at": datetime.now().isoformat()}
    )
    
    try:
        start_time = datetime.now()

        # Create a query to gather content for flashcards
        query = "Extract comprehensive information from this document including all key concepts, facts, definitions, examples, relationships between topics, methodologies, processes, theories, historical context, practical applications, edge cases, and underlying principles. Cover all sections and subtopics thoroughly to ensure complete coverage of the material for high-quality and diverse flashcard generation."

        # Embed query using multimodal embeddings (1024 dims) by default, or OpenAI (1536 dims) if specified
        query_embedded = embed_query_v2(query) if use_openai_embeddings else embed_query_multimodal(query)

        # Check for embedding errors
        if isinstance(query_embedded, dict) and "message" in query_embedded and "status" in query_embedded:
            error_msg = f"Query embedding failed: {query_embedded['message']}"
            logging.error(error_msg)
            update_error_status(document_id, error_msg)
            return {"flashcards": [], "status": "error", "message": error_msg}
        
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
            update_error_status(document_id, error_msg)
            return {"flashcards": [], "status": "error", "message": error_msg}
        
        # Rerank results to get most relevant content
        reranked = rerank_results(query, hits, top_n=rerank_top_n)
        
        # Prepare context for flashcard generation
        context_chunks = []
        for item in reranked:
            if "metadata" in item and "text" in item["metadata"]:
                context_chunks.append(item["metadata"]["text"])
        
        if not context_chunks:
            error_msg = "No text content found in search results."
            update_error_status(document_id, error_msg)
            return {"flashcards": [], "status": "error", "message": error_msg}
        
        # Generate flashcards chunk by chunk (parallel processing)
        logging.info(f"Processing {len(context_chunks)} chunks for additional flashcard generation")
        
        # Create a shared state for tracking accumulated flashcards and database updates
        shared_state = {
            "accumulated_cards": [],
            "set_id": next_set_number,
            "update_threshold": 5,  # Update DB every 5 flashcards
            "last_update_count": 0,
            "document_id": document_id,
            "last_update_time": datetime.now(),
            "existing_flashcards": existing_flashcards  # Pass existing flashcards to avoid duplicates
        }
        
        # Generate new flashcards, avoiding duplicates with existing ones
        new_flashcards = generate_flashcards_from_chunks_parallel(
            context_chunks, 
            num_questions=num_questions,
            shared_state=shared_state
        )
        
        # Final update to the database with all flashcards (after deduplication)
        # Keep existing sets and append the new set
        final_flashcards = existing_flashcards_data if existing_flashcards_data else []
        final_flashcards.append({
            "set_id": next_set_number,
            "cards": new_flashcards
        })
        
        update_generated_content(
            document_id,
            {
                "flashcards": final_flashcards, 
                "status": "completed", 
                "updated_at": datetime.now().isoformat()
            }
        )
        
        # Insert the new set to flashcard_sets
        try:
            logging.info(f"Inserting new flashcard set #{next_set_number} with {len(new_flashcards)} cards")
            insert_flashcard_set(content_id, new_flashcards, next_set_number)
        except Exception as e:
            logging.error(f"Error storing new flashcard set: {e}")
        
        elapsed_time = (datetime.now() - start_time).total_seconds()
        logging.info(f"Additional flashcard generation completed in {elapsed_time:.2f} seconds. Generated {len(new_flashcards)} new flashcards.")
        
        return {
            "flashcards": final_flashcards,
            "status": "success", 
            "elapsed_seconds": elapsed_time, 
            "total_flashcards": len(new_flashcards), 
            "num_questions": num_questions
        }
        
    except Exception as e:
        logging.exception(f"Error generating additional flashcards: {e}")
        update_error_status(document_id, str(e))
        return {"flashcards": [], "status": "error", "message": str(e)} 