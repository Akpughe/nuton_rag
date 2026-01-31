"""
Tutoring Context Enhancement System
Provides educational context enrichment that transforms RAG responses into tutoring experiences.
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import re


@dataclass
class TutoringContext:
    """Context information for tutoring-enhanced responses"""
    source_summary: str
    learning_objective: str
    background_needed: List[str]
    key_concepts: List[str]
    practical_applications: List[str]
    connection_opportunities: List[str]
    learning_style_adaptations: Dict[str, str]


class ContextAnalyzer:
    """Analyzes content to determine educational context needs"""
    
    def analyze_educational_needs(
        self,
        query: str,
        document_context: str,
        source_type: str = "document"
    ) -> TutoringContext:
        """
        Analyze query and content to determine educational context needs.
        
        Args:
            query: User's question
            document_context: Content from documents
            source_type: Type of source (document, video, etc.)
            
        Returns:
            TutoringContext with educational enhancement recommendations
        """
        
        # Generate source summary
        source_summary = self._create_source_summary(document_context, source_type)
        
        # Determine learning objective
        learning_objective = self._extract_learning_objective(query, document_context)
        
        # Identify background concepts needed
        background_needed = self._identify_background_needs(query, document_context)
        
        # Extract key concepts
        key_concepts = self._extract_key_concepts(document_context)
        
        # Identify practical applications
        practical_applications = self._identify_practical_applications(document_context)
        
        # Find connection opportunities
        connection_opportunities = self._find_connection_opportunities(query, document_context)
        
        # Create learning style adaptations
        learning_style_adaptations = self._create_style_adaptations(
            query, document_context, source_type
        )
        
        return TutoringContext(
            source_summary=source_summary,
            learning_objective=learning_objective,
            background_needed=background_needed,
            key_concepts=key_concepts,
            practical_applications=practical_applications,
            connection_opportunities=connection_opportunities,
            learning_style_adaptations=learning_style_adaptations
        )
    
    def _create_source_summary(self, content: str, source_type: str) -> str:
        """Create educational summary of source material"""
        if not content:
            return f"No {source_type} content provided for educational analysis."
        
        # Simple heuristic for source summary
        content_length = len(content.split())
        
        if content_length < 50:
            scope = "brief overview"
        elif content_length < 200:
            scope = "focused discussion"
        elif content_length < 500:
            scope = "comprehensive coverage"
        else:
            scope = "detailed exploration"
        
        return f"This {source_type} provides a {scope} of the topic, covering approximately {content_length} words of educational content."
    
    def _extract_learning_objective(self, query: str, content: str) -> str:
        """Extract or infer learning objective from query and content"""
        
        # Query-based learning objectives
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["what is", "define", "meaning"]):
            return "Understand definitions and core concepts"
        elif any(word in query_lower for word in ["how to", "how do", "steps"]):
            return "Learn procedures and implementation methods"
        elif any(word in query_lower for word in ["why", "reason", "cause"]):
            return "Understand reasoning and causal relationships"
        elif any(word in query_lower for word in ["compare", "difference", "versus"]):
            return "Analyze differences and make comparisons"
        elif any(word in query_lower for word in ["example", "instance", "case"]):
            return "Apply concepts through concrete examples"
        else:
            return "Gain comprehensive understanding of the topic"
    
    def _identify_background_needs(self, query: str, content: str) -> List[str]:
        """Identify background concepts that may need explanation"""
        
        background_needs = []
        
        # Technical terms that might need definition
        technical_patterns = [
            r'\b[A-Z]{2,}\b',  # Acronyms
            r'\b\w+(?:tion|sion|ment|ness|ity|ing)\b',  # Technical suffixes
            r'\b(?:algorithm|method|process|system|framework|model)\b',  # Technical terms
        ]
        
        for pattern in technical_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                background_needs.extend([match.lower() for match in matches[:3]])  # Limit to avoid noise
        
        # Domain-specific background needs
        if any(word in content.lower() for word in ["code", "programming", "software"]):
            background_needs.append("programming fundamentals")
        
        if any(word in content.lower() for word in ["analysis", "research", "study"]):
            background_needs.append("research methodology")
        
        if any(word in content.lower() for word in ["business", "strategy", "market"]):
            background_needs.append("business concepts")
        
        return list(set(background_needs[:5]))  # Remove duplicates, limit to 5
    
    def _extract_key_concepts(self, content: str) -> List[str]:
        """Extract key concepts from content for emphasis"""
        
        # Simple keyword extraction based on frequency and importance indicators
        key_concepts = []
        
        # Look for concepts that appear multiple times
        words = content.lower().split()
        word_freq = {}
        
        for word in words:
            if len(word) > 4:  # Focus on substantial words
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Get frequently mentioned concepts
        frequent_concepts = [word for word, freq in word_freq.items() if freq > 1]
        key_concepts.extend(frequent_concepts[:3])
        
        # Look for definition patterns
        definition_patterns = [
            r'(\w+)\s+(?:is|are|means|refers to)',
            r'(?:define|definition of)\s+(\w+)',
            r'(\w+):\s+[A-Z]',  # Word followed by colon and definition
        ]
        
        for pattern in definition_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            key_concepts.extend([match.lower() for match in matches[:2]])
        
        return list(set(key_concepts[:5]))  # Remove duplicates, limit to 5
    
    def _identify_practical_applications(self, content: str) -> List[str]:
        """Identify practical applications mentioned in content"""
        
        applications = []
        
        # Look for application indicators
        application_patterns = [
            r'(?:use|apply|implement|utilize)\s+(\w+(?:\s+\w+){0,2})',
            r'(?:example|instance|case study):\s*([^.]+)',
            r'(?:in practice|real-world|applied)\s+([^.]+)',
        ]
        
        for pattern in application_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            applications.extend([match.strip()[:50] for match in matches[:2]])  # Limit length
        
        # Domain-specific applications
        content_lower = content.lower()
        
        if "code" in content_lower or "programming" in content_lower:
            applications.append("software development applications")
        
        if "business" in content_lower or "management" in content_lower:
            applications.append("business implementation strategies")
        
        if "research" in content_lower or "analysis" in content_lower:
            applications.append("research and analysis applications")
        
        return applications[:3]  # Limit to 3 applications
    
    def _find_connection_opportunities(self, query: str, content: str) -> List[str]:
        """Find opportunities to connect this content to broader knowledge"""
        
        connections = []
        
        # Domain connections
        content_lower = content.lower()
        query_lower = query.lower()
        combined = content_lower + " " + query_lower
        
        # Technical connections
        if any(word in combined for word in ["code", "programming", "software", "algorithm"]):
            connections.append("computer science principles")
        
        # Business connections
        if any(word in combined for word in ["business", "strategy", "market", "organization"]):
            connections.append("business strategy frameworks")
        
        # Scientific connections
        if any(word in combined for word in ["research", "study", "analysis", "method"]):
            connections.append("research methodology and scientific thinking")
        
        # Learning connections
        if any(word in combined for word in ["learn", "study", "understand", "concept"]):
            connections.append("learning theory and cognitive science")
        
        # Historical/contextual connections
        connections.append("historical development and evolution")
        connections.append("current trends and future implications")
        
        return connections[:4]  # Limit to 4 connections
    
    def _create_style_adaptations(
        self,
        query: str,
        content: str,
        source_type: str
    ) -> Dict[str, str]:
        """Create learning style specific adaptations"""
        
        adaptations = {
            "academic_focus": f"Structure content for exam preparation with clear definitions and testable concepts from this {source_type}",
            "deep_dive": f"Provide comprehensive analysis connecting this {source_type} content to broader theoretical frameworks",
            "quick_practical": f"Extract actionable insights and implementation steps from this {source_type}",
            "exploratory_curious": f"Highlight fascinating aspects and unexpected connections in this {source_type}",
            "narrative_reader": f"Convert this {source_type} content into engaging, readable narrative format",
            "default": f"Present this {source_type} content in clear, systematic organization for optimal learning"
        }
        
        return adaptations


class TutoringContextEnhancer:
    """Enhances responses with tutoring context"""
    
    def __init__(self):
        self.analyzer = ContextAnalyzer()
    
    def create_educational_context_prompt(
        self,
        query: str,
        document_context: str,
        learning_style: str = "default",
        source_type: str = "document"
    ) -> str:
        """
        Create educational context prompt for tutoring-enhanced responses.
        
        Args:
            query: User's question
            document_context: Content from documents
            learning_style: Selected learning style
            source_type: Type of source material
            
        Returns:
            Educational context prompt
        """
        
        # Analyze educational needs
        tutoring_context = self.analyzer.analyze_educational_needs(
            query, document_context, source_type
        )
        
        # Build context-aware prompt
        context_prompt = f"""
## EDUCATIONAL CONTEXT ANALYSIS:

**Source Overview**: {tutoring_context.source_summary}

**Learning Objective**: {tutoring_context.learning_objective}

**Background Concepts to Address**: {', '.join(tutoring_context.background_needed) if tutoring_context.background_needed else 'None identified'}

**Key Concepts to Emphasize**: {', '.join(tutoring_context.key_concepts) if tutoring_context.key_concepts else 'Extract from content'}

**Practical Applications**: {'; '.join(tutoring_context.practical_applications) if tutoring_context.practical_applications else 'Identify relevant applications'}

**Connection Opportunities**: {'; '.join(tutoring_context.connection_opportunities)}

## TUTORING APPROACH FOR THIS QUERY:

### Context-First Strategy:
1. **Establish Source Context**: Begin by explaining what educational value this {source_type} provides
2. **Set Learning Frame**: Help user understand what they'll learn and why it matters
3. **Provide Necessary Background**: Address any prerequisite concepts users need to understand

### Educational Enhancement:
4. **Progressive Explanation**: Build understanding step by step, connecting new concepts to familiar ones
5. **Multiple Perspectives**: Show how this information connects to broader knowledge areas
6. **Practical Relevance**: Help users see real-world applications and significance

### Learning Style Adaptation:
{tutoring_context.learning_style_adaptations.get(learning_style, tutoring_context.learning_style_adaptations['default'])}

## RESPONSE QUALITY STANDARDS:
- Always provide sufficient context before diving into answers
- Act as a knowledgeable tutor, not just an information provider
- Help users understand concepts deeply, not just get quick answers
- Make connections explicit between document content and broader knowledge
- Encourage further learning and exploration
"""
        
        return context_prompt
    
    def enhance_response_with_context(
        self,
        base_response: str,
        tutoring_context: TutoringContext,
        learning_style: str = "default"
    ) -> str:
        """
        Enhance an existing response with tutoring context.
        
        Args:
            base_response: Original response
            tutoring_context: Tutoring context information
            learning_style: Learning style for adaptation
            
        Returns:
            Enhanced response with educational context
        """
        
        # This would be used if we need to post-process responses
        # For now, we focus on prompt-based enhancement
        
        context_introduction = f"""
**Learning Context**: {tutoring_context.learning_objective}

**Key Concepts**: {', '.join(tutoring_context.key_concepts) if tutoring_context.key_concepts else 'See below'}

**Educational Enhancement**:
"""
        
        enhanced_response = context_introduction + base_response
        
        if tutoring_context.connection_opportunities:
            enhanced_response += f"""

**Broader Connections**: This relates to {', '.join(tutoring_context.connection_opportunities[:2])}.
"""
        
        return enhanced_response


def create_tutoring_enhanced_prompt(
    query: str,
    document_context: str,
    learning_style: str = "default",
    source_type: str = "document",
    base_prompt: str = None
) -> Tuple[str, Dict[str, Any]]:
    """
    Convenience function to create tutoring-enhanced prompt.
    
    Args:
        query: User's question
        document_context: Document content
        learning_style: Selected learning style
        source_type: Type of source material
        base_prompt: Optional base prompt to enhance
        
    Returns:
        Tuple of (enhanced_prompt, context_metadata)
    """
    
    enhancer = TutoringContextEnhancer()
    
    # Create educational context prompt
    context_prompt = enhancer.create_educational_context_prompt(
        query, document_context, learning_style, source_type
    )
    
    # Combine with base prompt if provided
    if base_prompt:
        enhanced_prompt = base_prompt + "\n\n" + context_prompt
    else:
        enhanced_prompt = context_prompt
    
    # Analyze tutoring context for metadata
    tutoring_context = enhancer.analyzer.analyze_educational_needs(
        query, document_context, source_type
    )
    
    metadata = {
        "tutoring_context": tutoring_context,
        "learning_style": learning_style,
        "source_type": source_type,
        "educational_enhancements": {
            "background_concepts": len(tutoring_context.background_needed),
            "key_concepts": len(tutoring_context.key_concepts),
            "practical_applications": len(tutoring_context.practical_applications),
            "connection_opportunities": len(tutoring_context.connection_opportunities)
        }
    }
    
    return enhanced_prompt, metadata


class StudyModeEnhancer:
    """Special enhancements for study mode contexts"""
    
    @staticmethod
    def create_study_context_prompt(
        query: str,
        document_context: str,
        learning_style: str,
        study_type: str = "general"  # general, exam_prep, concept_review, etc.
    ) -> str:
        """Create study-specific context prompt"""
        
        study_enhancements = {
            "exam_prep": """
## EXAM PREPARATION FOCUS:
- Structure information for optimal memorization and recall
- Highlight testable concepts and key definitions
- Provide memory aids and study techniques
- Create clear review points and self-assessment opportunities
""",
            "concept_review": """
## CONCEPT REVIEW FOCUS:
- Build clear conceptual frameworks and relationships
- Connect new information to previously learned material
- Provide multiple examples and applications
- Emphasize understanding over memorization
""",
            "skill_building": """
## SKILL BUILDING FOCUS:
- Emphasize practical application and hands-on learning
- Provide step-by-step procedures and methods
- Include practice opportunities and implementation guidance
- Connect theory to real-world application
""",
            "general": """
## STUDY ENHANCEMENT FOCUS:
- Support comprehensive understanding and retention
- Balance theoretical knowledge with practical applications
- Provide clear organization for effective study
- Encourage active learning and engagement
"""
        }
        
        study_prompt = study_enhancements.get(study_type, study_enhancements["general"])
        
        base_context = create_tutoring_enhanced_prompt(
            query, document_context, learning_style
        )[0]
        
        return base_context + study_prompt + """
**Study Mode Reminder**: Structure your response to maximize learning effectiveness and retention for study purposes.
"""