import os
from typing import List, Dict, Tuple, Any
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize the client
client = OpenAI(api_key=OPENAI_API_KEY)

def generate_answer(
    query: str,
    context_chunks: List[Dict],
    system_prompt: str,
    model: str = "gpt-4o"
) -> Tuple[str, List[Dict]]:
    """
    Generate an answer using OpenAI GPT-4o, given a query and context chunks.
    Args:
        query: The user query string.
        context_chunks: List of dicts with 'text' and citation info.
        system_prompt: System prompt for LLM.
        model: Model name (default: gpt-4o).
    Returns:
        Tuple of (answer string, list of cited chunks/passages).
    Raises:
        Exception if the API call fails.
    """
    # Extract text from either direct "text" field or from "metadata.text"
    context_texts = []
    for chunk in context_chunks:
        if "text" in chunk:
            context_texts.append(chunk["text"])
        elif "metadata" in chunk and "text" in chunk["metadata"]:
            context_texts.append(chunk["metadata"]["text"])
        else:
            # Add an empty string as fallback to avoid breaking the joining operation
            context_texts.append("")
    
    context = "\n\n".join(context_texts)
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{query}\n\nContext:\n{context}"}
    ]
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages
        )
        answer = response.choices[0].message.content
        return answer, context_chunks
    except Exception as e:
        raise Exception(f"OpenAI GPT-4o failed: {e}")


def generate_flashcards(
    context: str,
    system_prompt: str,
    user_prompt: str,
    model: str = "gpt-4o",
) -> List[Dict[str, Any]]:
    """
    Generate flashcards using OpenAI GPT-4o, given context material.
    
    Args:
        context: The document content to generate flashcards from.
        system_prompt: System prompt for LLM.
        user_prompt: The prompt template to use for flashcard generation.
        model: Model name (default: gpt-4o).
        
    Returns:
        List of flashcard objects with question, answer, hint, and explanation.
        
    Raises:
        Exception if the API call fails.
    """
    try:
        # Format the user prompt with the context
        formatted_prompt = user_prompt.format(context=context)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": formatted_prompt}
        ]
        
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.8,  # Increased for more creative and diverse flashcards
            max_tokens=8000,   # Doubled to allow for many more flashcards
            stream=True
        )
        # print('response', response)

        response_text = ""

        for chunk in response:
            if chunk.choices[0].delta.content:
                response_text += chunk.choices[0].delta.content


        # Extract the response content
        # response_text = response.choices[0].message.content

        
        # Parse the response text into flashcard objects
        flashcards = parse_flashcards(response_text)
        print('flashcards', flashcards)
        
        return flashcards
        
    except Exception as e:
        raise Exception(f"OpenAI flashcard generation failed: {e}")


def parse_flashcards(response_text: str) -> List[Dict[str, str]]:
    """
    Parse the raw text response from OpenAI into structured flashcard objects.
    
    Args:
        response_text: The raw text response from OpenAI.
        
    Returns:
        List of flashcard objects with question, answer, hint, and explanation.
    """
    flashcards = []
    # Split by the flashcard separator (triple dash)
    cards_raw = response_text.split("---")
    
    current_card = {}
    for section in cards_raw:
        section = section.strip()
        if not section:
            continue
            
        # Check if this section has flashcard fields
        has_question = "Question:" in section
        has_answer = "Answer:" in section
        
        if has_question and has_answer:
            current_card = {}
            
            # Extract each field
            for line in section.split("\n"):
                line = line.strip()
                if not line:
                    continue
                    
                if line.startswith("Question:"):
                    current_card["question"] = line.replace("Question:", "", 1).strip()
                elif line.startswith("Answer:"):
                    current_card["answer"] = line.replace("Answer:", "", 1).strip()
                elif line.startswith("Hint:"):
                    current_card["hint"] = line.replace("Hint:", "", 1).strip()
                elif line.startswith("Explanation:"):
                    current_card["explanation"] = line.replace("Explanation:", "", 1).strip()
            
            # Only add complete cards
            if all(k in current_card for k in ["question", "answer", "hint", "explanation"]):
                flashcards.append(current_card)
    
    return flashcards 