import os
import logging
from typing import List, Dict, Optional, Union
from groq import Groq
from openai import OpenAI

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ResponseGenerator:
    """
    A class to generate responses using different LLM providers with fallback options.
    Supports Groq as primary provider with OpenAI as backup.
    """
    
    def __init__(self):
        """Initialize the response generator with API keys and model configurations."""
        # Validate API keys
        openai_api_key = os.getenv('OPENAI_API_KEY')
        groq_api_key = os.getenv('GROQ_API_KEY')
        
        if not openai_api_key:
            logger.warning("OPENAI_API_KEY not found in environment variables")
        if not groq_api_key:
            logger.warning("GROQ_API_KEY not found in environment variables")
            
        # Initialize clients
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.groq_client = Groq(api_key=groq_api_key)
        
        # Define fallback models with hierarchy
        self.groq_models = [
            "llama-3.3-70b-versatile",  # First choice
            "llama-3.1-70b-versatile",  # First fallback 
            "llama-3.1-8b-versatile",   # Second fallback (smaller model)
            "mixtral-8x7b-32768"        # Third fallback
        ]
        
        logger.info("ResponseGenerator initialized with Groq and OpenAI fallback")
    
    def generate_response(self, 
                         query: str, 
                         retrieved_contexts: List[str], 
                         use_external_knowledge: bool = False) -> str:
        """
        Generate a response to a query using the provided contexts.
        
        Args:
            query: The user's query
            retrieved_contexts: List of relevant text contexts
            use_external_knowledge: Whether to allow using knowledge beyond the context
        
        Returns:
            The generated response text
        """
        # Prepare context and determine response parameters
        context_text = self._prepare_context(retrieved_contexts)
        system_message, instruction = self._create_prompt_components(
            query, 
            retrieved_contexts, 
            use_external_knowledge
        )
        
        prompt = f"""
        Context: {context_text}
        Question: {query}
        
        {instruction}
        """

        # Try Groq models with fallback options
        response = self._try_groq_models(system_message, prompt)
        
        # If all Groq models fail, fall back to OpenAI
        if not response:
            response = self._try_openai_fallback(system_message, prompt)
        
        # Final fallback message if all providers fail
        if not response:
            response = "I apologize, but I couldn't generate a response due to technical difficulties. Please try again later."
            
        return response
    
    def _prepare_context(self, retrieved_contexts: List[str]) -> str:
        """Combine contexts into a properly formatted string."""
        if not retrieved_contexts:
            return "No relevant context available."
        return "\n\n".join(retrieved_contexts)
    
    def _create_prompt_components(self, 
                                 query: str, 
                                 retrieved_contexts: List[str], 
                                 use_external_knowledge: bool) -> tuple:
        """Create appropriate system message and instruction based on query and settings."""
        # Check if we have multiple distinct documents
        multiple_docs = len(retrieved_contexts) > 1
        
        # Check if query might need a longer response
        needs_longer_response = any(term in query.lower() for term in [
            "explain", "describe", "elaborate", "summarize", "overview", 
            "compare", "contrast", "analyze", "detail", "comprehensive"
        ])
        
        # Create appropriate system message based on settings
        if use_external_knowledge:
            system_message = "You are a helpful assistant. Provide clear, concise answers based on the given context and your knowledge."
        else:
            system_message = "You are a helpful assistant limited to the provided context only."
        
        # Create appropriate instruction based on query characteristics
        instruction_parts = []
        
        # Response length instruction
        if needs_longer_response:
            instruction_parts.append("Provide a more detailed response since this query requires elaboration.")
        else:
            instruction_parts.append("Keep your answer brief and to the point unless more detail is clearly needed.")
        
        # Multiple documents instruction
        if multiple_docs:
            instruction_parts.append("When answering, make sure to cover information from ALL provided documents, not just one. Balance your response to include key points from each document.")
        
        # External knowledge instruction
        if use_external_knowledge:
            instruction_parts.append("You may use external knowledge when the context is insufficient, but clearly indicate when you do so.")
        else:
            instruction_parts.append("Only use information found in the context. If the context doesn't contain sufficient information, say 'Insufficient context.'")
        
        # Combine instructions
        instruction = " ".join(instruction_parts)
        
        return system_message, instruction
    
    def _try_groq_models(self, system_message: str, prompt: str) -> Optional[str]:
        """Try generating response with each Groq model until one succeeds."""
        for model in self.groq_models:
            try:
                logger.info(f"Attempting to generate response with Groq model: {model}")
                response = self.groq_client.chat.completions.create(
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
        
        logger.error("All Groq models failed")
        return None
    
    def _try_openai_fallback(self, system_message: str, prompt: str) -> Optional[str]:
        """Try generating response with OpenAI as fallback."""
        try:
            logger.info("Falling back to OpenAI model")
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",  # More widely available model
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ]
            )
            logger.info("Successfully generated response with OpenAI fallback")
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI fallback failed: {str(e)}")
            return None