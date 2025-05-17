import os
from typing import List, Dict, Tuple
from dotenv import load_dotenv
import openai

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

openai.api_key = OPENAI_API_KEY

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
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            stream=False
        )
        answer = response["choices"][0]["message"]["content"]
        return answer, context_chunks
    except Exception as e:
        raise Exception(f"OpenAI GPT-4o failed: {e}") 