import openai
import os

class ResponseGenerator:
    def __init__(self):
        openai.api_key = os.getenv('OPENAI_API_KEY')
    
    def generate_response(self, query, retrieved_contexts):
        """
        Generate a response to a query using the provided contexts.
        
        Args:
            query (str): The user's query
            retrieved_contexts (List[str]): List of relevant text contexts
        
        Returns:
            str: The generated response
        """
        # Combine contexts into a single string
        context_text = "\n\n".join(retrieved_contexts)
        
        prompt = f"""
        Context: {context_text}
        Question: {query}
        
        Provide a comprehensive answer using ONLY the given context.
        If the context doesn't contain sufficient information, say "Insufficient context."
        """
        
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        
        return response.choices[0].message.content