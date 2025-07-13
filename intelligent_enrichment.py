"""
Intelligent knowledge enrichment integration system.
Combines gap analysis with enhanced prompting for optimal general knowledge integration.
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

class IntelligentEnrichmentEngine:
    """
    Main engine for intelligent general knowledge enrichment.
    Analyzes context and determines optimal enrichment strategy.
    """
    
    def __init__(self):
        self.gap_analyzer = KnowledgeGapAnalyzer()
        self.quality_controller = EnrichmentQualityController()
    
    def should_enrich_response(
        self,
        query: str,
        document_context: str,
        allow_general_knowledge: bool
    ) -> Dict[str, Any]:
        """
        Analyze whether and how to enrich the response with general knowledge.
        
        Args:
            query: User's question
            document_context: Content from user's documents  
            allow_general_knowledge: Whether enrichment is enabled
            
        Returns:
            Enrichment analysis and recommendations
        """
        
        if not allow_general_knowledge:
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
    base_system_prompt: str = None
) -> tuple[str, Dict[str, Any]]:
    """
    Create an enriched system prompt with intelligent knowledge enhancement.
    
    Args:
        query: User's question
        document_context: Content from user's documents
        allow_general_knowledge: Whether enrichment is enabled
        base_system_prompt: Base system prompt to enhance
        
    Returns:
        Tuple of (enhanced_system_prompt, enrichment_metadata)
    """
    
    engine = IntelligentEnrichmentEngine()
    enrichment_analysis = engine.should_enrich_response(
        query, document_context, allow_general_knowledge
    )
    
    if enrichment_analysis["should_enrich"]:
        enhanced_prompt = enrichment_analysis["prompt"]
        metadata = enrichment_analysis["strategy"]
        metadata["enrichment_applied"] = True
        metadata["reason"] = enrichment_analysis["reason"]
    else:
        # Use base prompt or standard general knowledge prompt
        enhanced_prompt = enrichment_analysis["prompt"] or base_system_prompt
        metadata = enrichment_analysis.get("strategy", {})
        metadata["enrichment_applied"] = False
        metadata["reason"] = enrichment_analysis["reason"]
    
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