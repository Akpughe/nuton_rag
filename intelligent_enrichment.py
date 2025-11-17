"""
Intelligent knowledge enrichment integration system.
Combines gap analysis with enhanced prompting for optimal general knowledge integration.
Now includes learning style integration for personalized educational responses.
"""

from typing import Dict, List, Any, Optional
from enhanced_prompts import (
    generate_enhanced_general_knowledge_prompt,
    get_domain_from_context,
    assess_user_expertise_level,
    determine_enrichment_strategy
)
from knowledge_enrichment import (
    KnowledgeGapAnalyzer,
    EnrichmentQualityController,
    EnrichmentOpportunity
)
from prompts import general_knowledge_prompt
from educational_prompts import EducationalPromptBuilder, enhance_existing_prompt_with_learning_style
from tutoring_context import create_tutoring_enhanced_prompt
from learning_styles import LearningStyleManager

class IntelligentEnrichmentEngine:
    """
    Main engine for intelligent general knowledge enrichment.
    Analyzes context and determines optimal enrichment strategy.
    Now includes learning style integration for personalized educational responses.
    """
    
    def __init__(self):
        self.gap_analyzer = KnowledgeGapAnalyzer()
        self.quality_controller = EnrichmentQualityController()
        self.learning_style_manager = LearningStyleManager()
        self.educational_prompt_builder = EducationalPromptBuilder()
    
    def should_enrich_response(
        self,
        query: str,
        document_context: str,
        allow_general_knowledge: bool,
        learning_style: Optional[str] = None,
        educational_mode: bool = False
    ) -> Dict[str, Any]:
        """
        Analyze whether and how to enrich the response with general knowledge.
        
        Args:
            query: User's question
            document_context: Content from user's documents  
            allow_general_knowledge: Whether enrichment is enabled
            learning_style: Learning style for personalized responses
            educational_mode: Whether to use educational/tutoring approach
            
        Returns:
            Enrichment analysis and recommendations
        """
        
        # Handle educational mode and learning styles
        if educational_mode and learning_style:
            return self._create_educational_enrichment(
                query, document_context, learning_style, allow_general_knowledge
            )
        
        if not allow_general_knowledge:
            # Even if general knowledge is disabled, we can still provide educational structure
            if educational_mode and learning_style:
                educational_prompt, metadata = create_tutoring_enhanced_prompt(
                    query, document_context, learning_style
                )
                return {
                    "should_enrich": True,
                    "reason": "Educational structure provided without general knowledge enrichment",
                    "prompt": educational_prompt,
                    "strategy": metadata,
                    "educational_mode": True
                }
            
            return {
                "should_enrich": False,
                "reason": "General knowledge enrichment disabled",
                "prompt": None,
                "strategy": None
            }
        
        # Analyze document context
        domain = get_domain_from_context(document_context, query)
        user_level = assess_user_expertise_level(document_context, query)
        enrichment_strategy = determine_enrichment_strategy(query, document_context)
        
        # Identify enrichment opportunities
        document_chunks = [document_context] if document_context else []
        opportunities = self.gap_analyzer.analyze_content_gaps(query, document_chunks, domain)
        
        # Determine if enrichment should proceed
        should_enrich = self.quality_controller.should_enrich(
            document_context, query, opportunities
        )
        
        if not should_enrich:
            return {
                "should_enrich": False,
                "reason": "No valuable enrichment opportunities identified",
                "prompt": general_knowledge_prompt,  # Use basic prompt as fallback
                "strategy": {
                    "domain": domain,
                    "user_level": user_level,
                    "opportunities": len(opportunities)
                }
            }
        
        # Generate enrichment plan
        enrichment_plan = self.quality_controller.generate_enrichment_plan(opportunities)
        
        # Create enhanced prompt
        enhanced_prompt = self._create_contextual_prompt(
            query, document_context, domain, user_level, 
            enrichment_strategy, opportunities, enrichment_plan
        )
        
        return {
            "should_enrich": True,
            "reason": f"Identified {len(opportunities)} enrichment opportunities",
            "prompt": enhanced_prompt,
            "strategy": {
                "domain": domain,
                "user_level": user_level,
                "enrichment_strategy": enrichment_strategy,
                "opportunities": opportunities,
                "plan": enrichment_plan
            }
        }
    
    def _create_contextual_prompt(
        self,
        query: str,
        document_context: str,
        domain: str,
        user_level: str,
        enrichment_strategy: str,
        opportunities: List[EnrichmentOpportunity],
        enrichment_plan: Dict[str, Any]
    ) -> str:
        """Create a contextually-aware enrichment prompt."""
        
        # Start with base enhanced prompt
        base_prompt = generate_enhanced_general_knowledge_prompt(
            query, document_context, domain, user_level, enrichment_strategy
        )
        
        # Add specific guidance based on identified opportunities
        specific_guidance = self._generate_opportunity_guidance(opportunities, domain)
        
        # Add transparency context
        transparency_context = self._generate_transparency_context(enrichment_plan)
        
        # Combine into final prompt
        final_prompt = f"""{base_prompt}

## SPECIFIC ENRICHMENT GUIDANCE FOR THIS QUERY:

{specific_guidance}

## TRANSPARENCY CONTEXT:
{transparency_context}

Focus your enrichment on the specific opportunities identified while maintaining the highest quality standards."""
        
        return final_prompt
    
    def _create_educational_enrichment(
        self,
        query: str,
        document_context: str,
        learning_style: str,
        allow_general_knowledge: bool
    ) -> Dict[str, Any]:
        """Create educational enrichment with learning style integration"""
        
        # Auto-detect learning style if needed
        if not learning_style or learning_style == "auto":
            learning_style = self.learning_style_manager.auto_detect_learning_style(query, document_context)
        
        # Create educational prompt
        educational_prompt, educational_metadata = self.educational_prompt_builder.create_educational_system_prompt(
            learning_style=learning_style,
            query=query,
            document_context=document_context,
            source_type="document",  # Default, can be enhanced later
            study_mode=False,  # Can be parameterized
            quiz_mode=False,   # Can be parameterized
            has_web_results=False  # Will be updated based on websearch
        )
        
        # If general knowledge is allowed, enhance with traditional enrichment
        if allow_general_knowledge:
            # Get traditional enrichment analysis
            traditional_analysis = self._get_traditional_enrichment_analysis(query, document_context)
            
            # Combine educational and general knowledge approaches
            enhanced_prompt = self._combine_educational_and_general_knowledge(
                educational_prompt, traditional_analysis, learning_style
            )
            
            return {
                "should_enrich": True,
                "reason": f"Educational approach with {learning_style} learning style plus general knowledge enrichment",
                "prompt": enhanced_prompt,
                "strategy": {
                    **educational_metadata,
                    "traditional_enrichment": traditional_analysis,
                    "combined_approach": True
                },
                "educational_mode": True
            }
        else:
            # Pure educational approach without general knowledge
            return {
                "should_enrich": True,
                "reason": f"Educational approach with {learning_style} learning style",
                "prompt": educational_prompt,
                "strategy": educational_metadata,
                "educational_mode": True
            }
    
    def _get_traditional_enrichment_analysis(self, query: str, document_context: str) -> Dict[str, Any]:
        """Get traditional enrichment analysis for integration with educational approach"""
        
        domain = get_domain_from_context(document_context, query)
        user_level = assess_user_expertise_level(document_context, query)
        enrichment_strategy = determine_enrichment_strategy(query, document_context)
        
        # Identify enrichment opportunities
        document_chunks = [document_context] if document_context else []
        opportunities = self.gap_analyzer.analyze_content_gaps(query, document_chunks, domain)
        
        # Generate enrichment plan
        enrichment_plan = self.quality_controller.generate_enrichment_plan(opportunities)
        
        return {
            "domain": domain,
            "user_level": user_level,
            "enrichment_strategy": enrichment_strategy,
            "opportunities": opportunities,
            "plan": enrichment_plan
        }
    
    def _combine_educational_and_general_knowledge(
        self,
        educational_prompt: str,
        traditional_analysis: Dict[str, Any],
        learning_style: str
    ) -> str:
        """Combine educational approach with traditional general knowledge enrichment"""
        
        # Extract key elements from traditional analysis
        domain = traditional_analysis["domain"]
        opportunities = traditional_analysis["opportunities"]
        
        # Create integration guidance
        integration_guidance = f"""
## GENERAL KNOWLEDGE INTEGRATION:

**Domain Context**: This query relates to {domain} domain knowledge.

**Identified Enhancement Opportunities**:
{self._format_opportunities_for_educational_context(opportunities)}

**Integration Strategy for {learning_style.title().replace('_', ' ')} Learning Style**:
• Seamlessly weave general knowledge into the educational structure
• Maintain the persona and approach of your learning style
• Use general knowledge to enhance understanding rather than replace document content
• Ensure all enrichments serve the educational learning objectives

**Quality Standards**:
• All general knowledge additions must align with the {learning_style} learning approach
• Maintain clear attribution between document content and general knowledge
• Focus enrichments on gaps that genuinely enhance the learning experience
"""
        
        return educational_prompt + integration_guidance
    
    def _format_opportunities_for_educational_context(self, opportunities: List) -> str:
        """Format enrichment opportunities for educational integration"""
        
        if not opportunities:
            return "• No specific opportunities identified - use general educational enhancement"
        
        formatted = []
        for i, opp in enumerate(opportunities[:3]):  # Limit to top 3
            formatted.append(f"• {opp.description} (Priority: {opp.priority})")
        
        return "\n".join(formatted)
    
    def _generate_opportunity_guidance(
        self, 
        opportunities: List[EnrichmentOpportunity], 
        domain: str
    ) -> str:
        """Generate specific guidance based on identified opportunities."""
        
        if not opportunities:
            return "No specific enrichment opportunities identified. Use standard enhancement approach."
        
        guidance_parts = []
        
        # Group opportunities by type
        opportunity_groups = {}
        for opp in opportunities[:5]:  # Limit to top 5 opportunities
            if opp.gap_type not in opportunity_groups:
                opportunity_groups[opp.gap_type] = []
            opportunity_groups[opp.gap_type].append(opp)
        
        # Generate guidance for each type
        type_instructions = {
            "background": "🔧 **Background Enhancement Needed:**",
            "context": "🌐 **Context Enhancement Needed:**", 
            "implications": "⚡ **Practical Implications Needed:**",
            "current_developments": "📈 **Current Context Needed:**"
        }
        
        for gap_type, opps in opportunity_groups.items():
            if gap_type in type_instructions:
                guidance_parts.append(type_instructions[gap_type])
                for opp in opps:
                    guidance_parts.append(f"   • {opp.description}")
                    guidance_parts.append(f"     Suggested: {opp.suggested_content}")
                    guidance_parts.append(f"     Priority: {opp.priority.upper()}")
                guidance_parts.append("")
        
        return "\n".join(guidance_parts)
    
    def _generate_transparency_context(self, enrichment_plan: Dict[str, Any]) -> str:
        """Generate transparency context for the user."""
        
        context_parts = [
            f"• Total enrichment opportunities: {enrichment_plan['total_opportunities']}",
            f"• High-priority enhancements: {enrichment_plan['high_priority']}"
        ]
        
        if enrichment_plan['transparency_notes']:
            context_parts.append("• Enrichment focus:")
            for note in enrichment_plan['transparency_notes']:
                context_parts.append(f"  - {note}")
        
        context_parts.append("")
        context_parts.append("Make sure to clearly communicate to the user what enrichment you're adding and why it's valuable.")
        
        return "\n".join(context_parts)

def create_enriched_system_prompt(
    query: str,
    document_context: str,
    allow_general_knowledge: bool,
    base_system_prompt: str = None,
    learning_style: Optional[str] = None,
    educational_mode: bool = False
) -> tuple[str, Dict[str, Any]]:
    """
    Create an enriched system prompt with intelligent knowledge enhancement.
    
    Args:
        query: User's question
        document_context: Content from user's documents
        allow_general_knowledge: Whether enrichment is enabled
        base_system_prompt: Base system prompt to enhance
        learning_style: Learning style for personalized responses
        educational_mode: Whether to use educational/tutoring approach
        
    Returns:
        Tuple of (enhanced_system_prompt, enrichment_metadata)
    """
    
    engine = IntelligentEnrichmentEngine()
    enrichment_analysis = engine.should_enrich_response(
        query, document_context, allow_general_knowledge, learning_style, educational_mode
    )
    
    if enrichment_analysis["should_enrich"]:
        enhanced_prompt = enrichment_analysis["prompt"]
        metadata = enrichment_analysis["strategy"]
        metadata["enrichment_applied"] = True
        metadata["reason"] = enrichment_analysis["reason"]
        metadata["educational_mode"] = enrichment_analysis.get("educational_mode", False)
        metadata["learning_style"] = learning_style
    else:
        # Use base prompt or standard general knowledge prompt
        enhanced_prompt = enrichment_analysis["prompt"] or base_system_prompt
        metadata = enrichment_analysis.get("strategy", {})
        metadata["enrichment_applied"] = False
        metadata["reason"] = enrichment_analysis["reason"]
        metadata["educational_mode"] = False
        metadata["learning_style"] = learning_style
    
    return enhanced_prompt, metadata

# For backward compatibility
def get_intelligent_general_knowledge_prompt(
    query: str,
    document_context: str,
    domain: str = None,
    user_level: str = None
) -> str:
    """
    Backward compatible function to get an intelligent general knowledge prompt.
    """
    
    if domain is None:
        domain = get_domain_from_context(document_context, query)
    
    if user_level is None:
        user_level = assess_user_expertise_level(document_context, query)
    
    enrichment_strategy = determine_enrichment_strategy(query, document_context)
    
    return generate_enhanced_general_knowledge_prompt(
        query, document_context, domain, user_level, enrichment_strategy
    )