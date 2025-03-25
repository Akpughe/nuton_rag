import openai
import os
from groq import Groq
import logging

logger = logging.getLogger(__name__)

class ResponseGenerator:
    def __init__(self):
        openai.api_key = os.getenv('OPENAI_API_KEY')
        self.groq = Groq(
                api_key=os.environ.get("GROQ_API_KEY"),
            )
        # Define fallback models
        self.groq_models = [
            "llama-3.3-70b-versatile",  # First choice
            "llama-3.1-70b-versatile",  # First fallback 
            "llama-3.1-8b-versatile",   # Second fallback (smaller model)
            "mixtral-8x7b-32768"        # Third fallback
        ]
    
    def generate_response(self, query, retrieved_contexts, use_external_knowledge=False):
        """
        Generate a response to a query using the provided contexts.
        
        Args:
            query (str): The user's query
            retrieved_contexts (List[str]): List of relevant text contexts
            use_external_knowledge (bool): Whether to allow using knowledge beyond the context
        
        Returns:
            str: The generated response
        """
        # Combine contexts into a single string
        context_text = "\n\n".join(retrieved_contexts)
        
        # Determine if we have multiple distinct documents
        multiple_docs = len(retrieved_contexts) > 1
        
        # Check if query might need a longer response
        needs_longer_response = any(term in query.lower() for term in [
            "explain", "describe", "elaborate", "summarize", "overview", 
            "compare", "contrast", "analyze", "detail", "comprehensive"
        ])
        
        if use_external_knowledge:
            system_message = "You are a helpful assistant. Provide clear, concise answers based on the given context and your knowledge."
            instruction = (
                f"{'Provide a more detailed response since this query requires elaboration. ' if needs_longer_response else 'Keep your answer brief and to the point unless more detail is clearly needed. '}"
                f"{'When answering, make sure to cover information from ALL provided documents, not just one. Balance your response to include key points from each document. ' if multiple_docs else ''}"
                f"You may use external knowledge when the context is insufficient, but clearly indicate when you do so."
            )
        else:
            system_message = "You are a helpful assistant limited to the provided context only."
            instruction = (
                f"{'Provide a more detailed response since this query requires elaboration. ' if needs_longer_response else 'Keep your answer brief and to the point unless more detail is clearly needed. '}"
                f"{'When answering, make sure to cover information from ALL provided documents, not just one. Balance your response to include key points from each document. ' if multiple_docs else ''}"
                f"Only use information found in the context. If the context doesn't contain sufficient information, say 'Insufficient context.'"
            )
        
        prompt = f"""
        Context: {context_text}
        Question: {query}
        
        {instruction}
        """

        # Try Groq models with fallback options
        for model in self.groq_models:
            try:
                logger.info(f"Attempting to generate response with Groq model: {model}")
                response = self.groq.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": prompt}
                    ]
                )
                logger.info(f"Successfully generated response with Groq model: {model}")
                return response.choices[0].message.content
            except Exception as e:
                logger.warning(f"Failed to generate response with Groq model {model}: {str(e)}")
                continue
        
        # If all Groq models fail, fall back to OpenAI
        try:
            logger.info("Falling back to OpenAI model")
            openai_client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",  # More widely available model
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ]
            )
            logger.info("Successfully generated response with OpenAI fallback")
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"All LLM providers failed: {str(e)}")
            return "I apologize, but I couldn't generate a response due to technical difficulties. Please try again later."