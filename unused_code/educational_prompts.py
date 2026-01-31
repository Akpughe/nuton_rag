"""
Educational Prompt Templates for Learning Style Integration
Provides enhanced prompting that transforms RAG responses into educational, tutoring-focused interactions.
"""

from typing import Dict, List, Any, Optional
from learning_styles import LearningStyleManager, LearningContext, LearningStyleType


class EducationalPromptBuilder:
    """Builds educational prompts that integrate with existing RAG system"""
    
    def __init__(self):
        self.style_manager = LearningStyleManager()
    
    def create_educational_system_prompt(
        self,
        learning_style: str,
        query: str,
        document_context: str,
        base_prompt: str = None,
        **context_kwargs
    ) -> tuple[str, Dict[str, Any]]:
        """
        Create educational system prompt for learning-style-aware responses.
        
        Args:
            learning_style: Learning style identifier
            query: User's question
            document_context: Content from documents
            base_prompt: Base prompt to enhance (optional)
            **context_kwargs: Additional context parameters
            
        Returns:
            Tuple of (educational_prompt, metadata)
        """
        
        # Get appropriate persona
        persona = self.style_manager.get_persona(learning_style)
        
        # Create learning context
        learning_context = self.style_manager.create_learning_context(
            query=query,
            document_context=document_context,
            **context_kwargs
        )
        
        # Generate persona-specific prompt
        educational_prompt = persona.create_system_prompt(learning_context)
        
        # Add base prompt integration if provided
        if base_prompt:
            educational_prompt = self._integrate_base_prompt(educational_prompt, base_prompt)
        
        # Get enrichment context
        enrichment_data = persona.enrich_context(learning_context)
        
        # Prepare metadata
        metadata = {
            "learning_style": learning_style,
            "persona_name": persona.name,
            "persona_goal": persona.goal,
            "enrichment_data": enrichment_data,
            "context_provided": bool(document_context),
            "educational_mode": True
        }
        
        return educational_prompt, metadata
    
    def _integrate_base_prompt(self, educational_prompt: str, base_prompt: str) -> str:
        """Integrate base system prompt with educational enhancements"""
        
        integration_note = f"""
## BASE SYSTEM INTEGRATION:
The following base guidelines also apply: {base_prompt}

However, prioritize the educational approach above - always provide context and act as a tutor rather than just answering directly.
"""
        
        return educational_prompt + integration_note
    
    def create_context_aware_prompt(
        self,
        learning_style: str,
        query: str,
        document_context: str,
        source_metadata: Dict[str, Any] = None,
        study_mode: bool = False,
        quiz_mode: bool = False,
        has_web_results: bool = False
    ) -> str:
        """
        Create context-aware educational prompt with source-specific guidance.
        """
        
        # Determine source type from metadata
        source_type = "document"
        if source_metadata:
            if "youtube" in source_metadata.get("source", "").lower():
                source_type = "video"
            elif "audio" in source_metadata.get("type", "").lower():
                source_type = "audio"
        
        # Create educational prompt
        prompt, metadata = self.create_educational_system_prompt(
            learning_style=learning_style,
            query=query,
            document_context=document_context,
            source_type=source_type,
            study_mode=study_mode,
            quiz_mode=quiz_mode,
            has_web_results=has_web_results
        )
        
        # Add source-specific context
        if source_metadata:
            source_context = self._create_source_context(source_metadata, source_type)
            prompt += f"\n\n## SOURCE CONTEXT:\n{source_context}"
        
        return prompt
    
    def _create_source_context(self, source_metadata: Dict[str, Any], source_type: str) -> str:
        """Create source-specific context for educational responses"""
        
        context_parts = []
        
        if source_type == "video":
            context_parts.extend([
                "• Video Content: When referencing video material, include timestamps when relevant",
                "• Speaker Context: Preserve the speaker's examples and teaching style where appropriate",
                "• Visual Elements: Note any important visual components mentioned in the transcript"
            ])
        
        if source_metadata.get("file_name"):
            context_parts.append(f"• Source Title: '{source_metadata['file_name']}'")
        
        if source_metadata.get("domain"):
            context_parts.append(f"• Subject Domain: {source_metadata['domain']}")
        
        # Add educational context based on source type
        context_parts.extend([
            "• Educational Approach: Always provide background context before diving into specific answers",
            "• Learning Support: Frame responses to help users understand concepts deeply, not just get answers",
            "• Context Integration: Help users see how this information connects to broader knowledge"
        ])
        
        return "\n".join(context_parts)


class TutoringPromptEnhancer:
    """Enhances existing prompts with tutoring-focused additions"""
    
    @staticmethod
    def add_tutoring_context(base_prompt: str, learning_style: str = "default") -> str:
        """Add tutoring context to existing prompts"""
        
        tutoring_addition = f"""
## EDUCATIONAL ENHANCEMENT:

Transform your response approach from information delivery to tutoring:

### Before Answering:
1. **Provide Context**: Explain where this information comes from and why it matters
2. **Set Learning Frame**: Help users understand what they'll learn from your response
3. **Connect to Prior Knowledge**: Link to concepts users likely already understand

### During Response:
4. **Explain, Don't Just Answer**: Provide the 'why' behind information, not just the 'what'
5. **Use Progressive Complexity**: Start simple and build to more complex concepts
6. **Include Examples**: Use concrete examples to illustrate abstract concepts

### After Main Content:
7. **Synthesize Learning**: Help users understand key takeaways
8. **Encourage Exploration**: Suggest related questions or areas for further learning

**Learning Style Focus**: Apply {learning_style} approach while maintaining educational depth.

**Remember**: You're a tutor, not just an information source. Help users learn and grow.
"""
        
        return base_prompt + tutoring_addition
    
    @staticmethod
    def enhance_general_knowledge_prompt(base_prompt: str, learning_style: str) -> str:
        """Enhance general knowledge prompts with learning style awareness"""
        
        style_guidance = {
            "academic_focus": "Focus on structured, exam-ready knowledge with clear definitions and testable concepts.",
            "deep_dive": "Provide comprehensive analysis from multiple perspectives with authoritative sources.",
            "quick_practical": "Emphasize actionable insights and real-world applications with concrete steps.",
            "exploratory_curious": "Make knowledge exciting with fascinating facts and unexpected connections.",
            "narrative_reader": "Present information in readable, article-style format with natural flow.",
            "default": "Use clear structure with balanced depth and accessibility."
        }
        
        specific_guidance = style_guidance.get(learning_style, style_guidance["default"])
        
        enhancement = f"""
## LEARNING STYLE INTEGRATION:

**Style-Specific Approach**: {specific_guidance}

**Educational Context Rule**: 
- Always begin responses with sufficient context about the source and topic
- Frame answers as teaching moments that build understanding
- Help users see connections between document content and broader knowledge
- Maintain the role of an expert tutor throughout the response

**General Knowledge Integration**:
- When adding knowledge beyond documents, clearly signal this enrichment
- Ensure all additions directly serve the user's learning objectives  
- Use the {learning_style} approach to structure and present enriched content
"""
        
        return base_prompt + enhancement


def get_educational_prompt_for_style(
    learning_style: str,
    query: str,
    document_context: str,
    base_system_prompt: str = None,
    **context_options
) -> tuple[str, Dict[str, Any]]:
    """
    Convenience function to get educational prompt for a specific learning style.
    
    Args:
        learning_style: Learning style identifier
        query: User's question
        document_context: Document content
        base_system_prompt: Optional base prompt to enhance
        **context_options: Additional context parameters
        
    Returns:
        Tuple of (educational_prompt, metadata)
    """
    
    builder = EducationalPromptBuilder()
    return builder.create_educational_system_prompt(
        learning_style=learning_style,
        query=query,
        document_context=document_context,
        base_prompt=base_system_prompt,
        **context_options
    )


def enhance_existing_prompt_with_learning_style(
    existing_prompt: str,
    learning_style: str = "default",
    add_tutoring_context: bool = True
) -> str:
    """
    Enhance an existing prompt with learning style and tutoring context.
    
    Args:
        existing_prompt: Current system prompt
        learning_style: Learning style to apply
        add_tutoring_context: Whether to add general tutoring context
        
    Returns:
        Enhanced prompt with learning style integration
    """
    
    enhanced_prompt = existing_prompt
    
    if add_tutoring_context:
        enhanced_prompt = TutoringPromptEnhancer.add_tutoring_context(
            enhanced_prompt, learning_style
        )
    
    # Add learning style specific enhancements
    enhanced_prompt = TutoringPromptEnhancer.enhance_general_knowledge_prompt(
        enhanced_prompt, learning_style
    )
    
    return enhanced_prompt


# Pre-built educational prompt templates for common scenarios
EDUCATIONAL_PROMPT_TEMPLATES = {
    "document_qa": {
        "academic_focus": """You are an Academic Tutor helping students understand document content for exam preparation.

Before answering, establish what subject area this document covers and why this information matters for academic success. Structure your response with clear definitions, key concepts, and study-ready organization.""",

        "deep_dive": """You are a Research Analyst helping users gain comprehensive understanding of document content.

Before answering, establish the broader intellectual context of this material. Explore the topic from multiple angles, connect to related fields, and provide the analytical depth needed for mastery.""",

        "quick_practical": """You are a Business Consultant extracting actionable insights from document content.

Before answering, quickly establish what practical value this document provides. Focus on immediate applications, concrete steps, and real-world implementation of the concepts discussed.""",

        "exploratory_curious": """You are an Enthusiastic Educator making document content fascinating and engaging.

Before answering, hook attention with the most interesting aspects of this material. Create wonder and curiosity about the broader topic while making learning an exciting journey.""",

        "narrative_reader": """You are a Storyteller/Writer converting document content into readable, engaging format.

Before answering, set the scene for what readers will discover. Transform the information into article-style content that flows naturally while preserving educational value.""",

        "default": """You are a Knowledge Architect helping users understand document content systematically.

Before answering, provide clear context about the document's educational value. Structure your response with logical organization that supports understanding and retention."""
    }
}


def get_template_prompt(scenario: str, learning_style: str) -> Optional[str]:
    """Get pre-built template prompt for specific scenario and learning style"""
    return EDUCATIONAL_PROMPT_TEMPLATES.get(scenario, {}).get(learning_style)