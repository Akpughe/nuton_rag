"""
Diagram Explainer - Context-based diagram description generation
Uses Groq's openai/gpt-oss-120b model with Gemini fallback for diagram descriptions.

Features:
- Context-based diagram description (using surrounding text)
- Groq model integration (openai/gpt-oss-120b)
- Gemini fallback for reliability
- Smart metadata extraction
- Handles both inline and reference diagrams

Author: RAG System Integration
Date: 2025
"""

import os
import logging
from typing import Dict, Any, List, Optional, Tuple
from dotenv import load_dotenv

# Import Groq
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    logging.warning("Groq not available. Install with: pip install groq")

# Import Gemini
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logging.warning("Google Gemini not available. Install with: pip install google-generativeai")

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DiagramExplainer:
    """
    Context-based diagram explainer using Groq and Gemini.

    Since openai/gpt-oss-120b may not have native vision capabilities,
    this class uses surrounding text context to intelligently describe diagrams.
    """

    def __init__(
        self,
        groq_api_key: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
        primary_model: str = "openai/gpt-oss-120b",
        fallback_model: str = "gemini-1.5-flash"
    ):
        """
        Initialize diagram explainer.

        Args:
            groq_api_key: Groq API key (defaults to env var)
            gemini_api_key: Gemini API key (defaults to env var)
            primary_model: Primary model for Groq (default: openai/gpt-oss-120b)
            fallback_model: Fallback Gemini model (default: gemini-1.5-flash)
        """
        self.groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        self.primary_model = primary_model
        self.fallback_model = fallback_model

        # Initialize Groq client
        self.groq_client = None
        if GROQ_AVAILABLE and self.groq_api_key:
            try:
                self.groq_client = Groq(api_key=self.groq_api_key)
                logger.info(f"Groq client initialized with model: {primary_model}")
            except Exception as e:
                logger.error(f"Failed to initialize Groq client: {e}")

        # Initialize Gemini
        self.gemini_model = None
        if GEMINI_AVAILABLE and self.gemini_api_key:
            try:
                genai.configure(api_key=self.gemini_api_key)
                self.gemini_model = genai.GenerativeModel(fallback_model)
                logger.info(f"Gemini client initialized with model: {fallback_model}")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini client: {e}")

    def explain_diagram_from_context(
        self,
        query: str,
        diagram_metadata: Dict[str, Any],
        surrounding_text: str,
        use_fallback: bool = False
    ) -> Dict[str, Any]:
        """
        Generate diagram description using context and metadata.

        Args:
            query: User's query
            diagram_metadata: Diagram metadata (page, position, etc.)
            surrounding_text: Text chunks surrounding the diagram
            use_fallback: Force use of Gemini fallback

        Returns:
            Dict with:
                - description: Generated description
                - model_used: Which model generated the description
                - confidence: Confidence score (if available)
                - success: Whether generation succeeded
        """
        logger.info(f"Generating diagram description for page {diagram_metadata.get('page', 'unknown')}")

        # Try Groq first (unless fallback is forced)
        if not use_fallback and self.groq_client:
            try:
                return self._explain_with_groq(query, diagram_metadata, surrounding_text)
            except Exception as e:
                logger.warning(f"Groq explanation failed: {e}, trying Gemini fallback...")

        # Fallback to Gemini
        if self.gemini_model:
            try:
                return self._explain_with_gemini(query, diagram_metadata, surrounding_text)
            except Exception as e:
                logger.error(f"Gemini explanation also failed: {e}")
                return {
                    "description": self._generate_fallback_description(diagram_metadata),
                    "model_used": "fallback",
                    "confidence": 0.3,
                    "success": False
                }

        # Ultimate fallback
        return {
            "description": self._generate_fallback_description(diagram_metadata),
            "model_used": "fallback",
            "confidence": 0.3,
            "success": False
        }

    def _explain_with_groq(
        self,
        query: str,
        diagram_metadata: Dict[str, Any],
        surrounding_text: str
    ) -> Dict[str, Any]:
        """
        Generate diagram description using Groq openai/gpt-oss-120b.

        Args:
            query: User's query
            diagram_metadata: Diagram metadata
            surrounding_text: Surrounding text context

        Returns:
            Description result dict
        """
        # Build smart prompt for context-based diagram description
        prompt = self._build_description_prompt(query, diagram_metadata, surrounding_text)

        # Call Groq
        response = self.groq_client.chat.completions.create(
            model=self.primary_model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert at analyzing academic and technical documents. "
                               "Based on the context provided, describe what diagrams and figures illustrate. "
                               "Be specific, concise, and accurate."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,  # Low temperature for factual descriptions
            max_tokens=200  # Concise descriptions
        )

        description = response.choices[0].message.content.strip()

        logger.info(f"✅ Groq generated diagram description ({len(description)} chars)")

        return {
            "description": description,
            "model_used": self.primary_model,
            "confidence": 0.85,  # High confidence for Groq
            "success": True
        }

    def _explain_with_gemini(
        self,
        query: str,
        diagram_metadata: Dict[str, Any],
        surrounding_text: str
    ) -> Dict[str, Any]:
        """
        Generate diagram description using Gemini (fallback).

        Args:
            query: User's query
            diagram_metadata: Diagram metadata
            surrounding_text: Surrounding text context

        Returns:
            Description result dict
        """
        # Build smart prompt
        prompt = self._build_description_prompt(query, diagram_metadata, surrounding_text)

        # Call Gemini
        response = self.gemini_model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.3,
                max_output_tokens=200
            )
        )

        description = response.text.strip()

        logger.info(f"✅ Gemini generated diagram description ({len(description)} chars)")

        return {
            "description": description,
            "model_used": self.fallback_model,
            "confidence": 0.8,  # Slightly lower confidence for fallback
            "success": True
        }

    def _build_description_prompt(
        self,
        query: str,
        diagram_metadata: Dict[str, Any],
        surrounding_text: str
    ) -> str:
        """
        Build intelligent prompt for diagram description.

        Args:
            query: User's query
            diagram_metadata: Diagram metadata
            surrounding_text: Surrounding text

        Returns:
            Formatted prompt string
        """
        page = diagram_metadata.get('page', 'unknown')
        position = diagram_metadata.get('position_in_doc', 'unknown')
        source_file = diagram_metadata.get('source_file', 'document')

        # Truncate surrounding text if too long
        max_context_chars = 1500
        if len(surrounding_text) > max_context_chars:
            surrounding_text = surrounding_text[:max_context_chars] + "..."

        prompt = f"""Based on the context below, describe what the diagram/figure on page {page} likely illustrates.

**User Query:** {query}

**Document:** {source_file}
**Diagram Location:** Page {page}, position {position}

**Context from surrounding text:**
{surrounding_text}

**Task:** Provide a clear, concise description (2-3 sentences) of what this diagram likely shows, based on the context. Focus on:
1. What type of diagram it is (circuit, flowchart, graph, illustration, etc.)
2. What concept or data it illustrates
3. How it relates to the user's query

Description:"""

        return prompt

    def _generate_fallback_description(self, diagram_metadata: Dict[str, Any]) -> str:
        """
        Generate basic fallback description when models fail.

        Args:
            diagram_metadata: Diagram metadata

        Returns:
            Basic description string
        """
        page = diagram_metadata.get('page', 'unknown')
        source_file = diagram_metadata.get('source_file', 'the document')

        return f"Diagram from {source_file}, page {page}. (Description generation unavailable)"


def explain_diagrams_batch(
    diagrams: List[Dict[str, Any]],
    query: str,
    text_chunks: List[Dict[str, Any]],
    max_diagrams: int = 3,
    groq_api_key: Optional[str] = None,
    gemini_api_key: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Batch process multiple diagrams for description generation.

    Args:
        diagrams: List of diagram chunks from search results
        query: User's query
        text_chunks: Text chunks for context extraction
        max_diagrams: Maximum number of diagrams to process (default: 3)
        groq_api_key: Groq API key
        gemini_api_key: Gemini API key

    Returns:
        List of diagram dicts with descriptions added
    """
    # Initialize explainer
    explainer = DiagramExplainer(groq_api_key=groq_api_key, gemini_api_key=gemini_api_key)

    # Limit to max_diagrams
    diagrams_to_process = diagrams[:max_diagrams]

    logger.info(f"Processing {len(diagrams_to_process)} diagrams (max: {max_diagrams})")

    enriched_diagrams = []

    for diagram in diagrams_to_process:
        try:
            # Extract metadata
            metadata = diagram.get("metadata", {})
            page = metadata.get("page", 0)

            # Find surrounding text context
            surrounding_text = _extract_surrounding_context(
                diagram, text_chunks, page, context_window=3
            )

            # Generate description
            description_result = explainer.explain_diagram_from_context(
                query=query,
                diagram_metadata=metadata,
                surrounding_text=surrounding_text
            )

            # Build enriched diagram object
            enriched_diagram = {
                "image_base64": metadata.get("image_base64"),
                "image_reference": None,  # Will be set if diagram is large
                "page": page,
                "source_file": metadata.get("source_file", "unknown"),
                "description": description_result["description"],
                "model_used": description_result["model_used"],
                "relevance_score": diagram.get("rerank_score", diagram.get("score", 0)),
                "storage_type": metadata.get("image_storage", "unknown"),
                "position_in_doc": metadata.get("position_in_doc", 0)
            }

            # Handle large images (>30KB) - set reference instead of base64
            if metadata.get("image_storage") == "reference":
                enriched_diagram["image_base64"] = None
                enriched_diagram["image_reference"] = f"Large diagram from {enriched_diagram['source_file']}, page {page}"

            enriched_diagrams.append(enriched_diagram)

            logger.info(f"✅ Processed diagram from page {page} using {description_result['model_used']}")

        except Exception as e:
            logger.error(f"Error processing diagram: {e}")
            # Add diagram with basic info even if description fails
            metadata = diagram.get("metadata", {})
            enriched_diagrams.append({
                "image_base64": metadata.get("image_base64"),
                "image_reference": None,
                "page": metadata.get("page", 0),
                "source_file": metadata.get("source_file", "unknown"),
                "description": f"Diagram from page {metadata.get('page', 'unknown')} (description unavailable)",
                "model_used": "fallback",
                "relevance_score": diagram.get("rerank_score", diagram.get("score", 0)),
                "storage_type": metadata.get("image_storage", "unknown"),
                "position_in_doc": metadata.get("position_in_doc", 0)
            })

    return enriched_diagrams


def _extract_surrounding_context(
    diagram_chunk: Dict[str, Any],
    text_chunks: List[Dict[str, Any]],
    diagram_page: int,
    context_window: int = 3
) -> str:
    """
    Extract surrounding text context for a diagram.

    Args:
        diagram_chunk: Diagram chunk data
        text_chunks: All text chunks from search results
        diagram_page: Page number of the diagram
        context_window: Number of chunks before/after to include

    Returns:
        Concatenated surrounding text
    """
    # Filter text chunks from same document and nearby pages
    diagram_doc_id = diagram_chunk.get("metadata", {}).get("document_id", "")

    relevant_chunks = []
    for chunk in text_chunks:
        chunk_metadata = chunk.get("metadata", {})
        chunk_doc_id = chunk_metadata.get("document_id", "")
        chunk_page = chunk_metadata.get("page", 0)

        # Same document and within context window
        if chunk_doc_id == diagram_doc_id:
            # Try to get page from page_number field or parse from chunk text
            try:
                if isinstance(chunk_page, str):
                    chunk_page = int(chunk_page.split(',')[0]) if chunk_page else 0
                elif isinstance(chunk_page, list):
                    chunk_page = chunk_page[0] if chunk_page else 0
            except:
                chunk_page = 0

            # Include chunks from nearby pages
            if abs(chunk_page - diagram_page) <= context_window:
                chunk_text = chunk.get("text") or chunk_metadata.get("text", "")
                if chunk_text:
                    relevant_chunks.append(chunk_text)

    # Concatenate and limit length
    surrounding_text = "\n\n".join(relevant_chunks[:10])  # Max 10 chunks

    # If no surrounding text found, use a generic message
    if not surrounding_text:
        surrounding_text = f"Text context near page {diagram_page} (limited context available)"

    return surrounding_text[:2000]  # Limit to 2000 chars


# Convenience function
def explain_diagram(
    query: str,
    diagram_metadata: Dict[str, Any],
    surrounding_text: str
) -> str:
    """
    Quick function to explain a single diagram.

    Args:
        query: User's query
        diagram_metadata: Diagram metadata
        surrounding_text: Surrounding text context

    Returns:
        Description string
    """
    explainer = DiagramExplainer()
    result = explainer.explain_diagram_from_context(query, diagram_metadata, surrounding_text)
    return result["description"]


# Testing
if __name__ == "__main__":
    print(f"\n{'='*80}")
    print(f"Testing Diagram Explainer")
    print(f"{'='*80}\n")

    # Test with sample data
    test_query = "Explain how voltage dividers work"
    test_metadata = {
        "page": 42,
        "source_file": "electronics_textbook.pdf",
        "position_in_doc": 0.45
    }
    test_context = """
    A voltage divider is a passive linear circuit that produces an output voltage (Vout)
    that is a fraction of its input voltage (Vin). Voltage division is the result of
    distributing the input voltage among the components of the divider.

    The circuit consists of two resistors in series. The input voltage is applied across
    the series combination, and the output voltage is the voltage across one of the resistors.

    Figure 3.2 shows a basic voltage divider circuit with two resistors R1 and R2.
    """

    try:
        explainer = DiagramExplainer()
        result = explainer.explain_diagram_from_context(
            query=test_query,
            diagram_metadata=test_metadata,
            surrounding_text=test_context
        )

        print(f"✅ Diagram description generated successfully!")
        print(f"   Model: {result['model_used']}")
        print(f"   Confidence: {result['confidence']}")
        print(f"   Success: {result['success']}")
        print(f"\n   Description:")
        print(f"   {result['description']}")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n{'='*80}\n")
