"""
Intelligent knowledge enrichment system for analyzing when and how 
to enrich document-based responses with general knowledge.
"""

import re
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass

@dataclass
class EnrichmentOpportunity:
    """Represents a specific opportunity for knowledge enrichment."""
    gap_type: str  # "background", "context", "implications", "current_developments"
    description: str
    priority: str  # "high", "medium", "low"
    suggested_content: str
    rationale: str

class KnowledgeGapAnalyzer:
    """Analyzes document content to identify opportunities for general knowledge enrichment."""
    
    def __init__(self):
        self.gap_patterns = {
            "missing_background": [
                r"(?:uses?|mentions?|refers? to) ([A-Z][a-zA-Z\s]+) (?:but|without|lacking)",
                r"assumes? (?:knowledge of|familiarity with) ([^.]+)",
                r"(?:based on|using) ([A-Z][a-zA-Z\s]+) (?:principles?|methods?|approaches?)"
            ],
            "incomplete_context": [
                r"(?:in the context of|within) ([^,]+), but",
                r"specifically (?:for|in) ([^,]+), however",
                r"(?:this|these) (?:applies?|works?) in ([^,]+) situations?"
            ],
            "missing_implications": [
                r"(?:this means|this indicates|this suggests) that ([^.]+)\.",
                r"(?:the result is|consequently|therefore) ([^.]+)\.",
                r"(?:impact|effect|consequence) (?:on|of) ([^.]+)"
            ],
            "outdated_references": [
                r"(?:as of|since|until) (\d{4})",
                r"(?:recent|latest|current) (?:studies?|research|findings?)",
                r"(?:traditional|conventional|standard) (?:approach|method|practice)"
            ]
        }
    
    def analyze_content_gaps(self, query: str, document_chunks: List[str], domain: str) -> List[EnrichmentOpportunity]:
        """
        Analyze document content to identify specific gaps that could benefit from enrichment.
        
        Args:
            query: User's question
            document_chunks: List of document text chunks
            domain: Subject domain
            
        Returns:
            List of enrichment opportunities ranked by priority
        """
        opportunities = []
        combined_content = " ".join(document_chunks)
        
        # Check for different types of gaps
        opportunities.extend(self._find_background_gaps(query, combined_content, domain))
        opportunities.extend(self._find_context_gaps(query, combined_content, domain))
        opportunities.extend(self._find_implication_gaps(query, combined_content, domain))
        opportunities.extend(self._find_currency_gaps(query, combined_content, domain))
        
        # Rank opportunities by priority
        return self._prioritize_opportunities(opportunities, query)
    
    def _find_background_gaps(self, query: str, content: str, domain: str) -> List[EnrichmentOpportunity]:
        """Find gaps where background knowledge would be helpful."""
        gaps = []
        
        # Check for technical terms without definition
        technical_terms = re.findall(r'\b[A-Z]{2,}(?:[A-Z][a-z]*)*\b', content)
        for term in set(technical_terms):
            if term not in content.lower() and len(term) > 2:
                gaps.append(EnrichmentOpportunity(
                    gap_type="background",
                    description=f"Technical term '{term}' mentioned without definition",
                    priority="medium",
                    suggested_content=f"Provide background on {term}",
                    rationale=f"User may benefit from understanding what {term} means in {domain} context"
                ))
        
        # Check for assumed knowledge
        for pattern in self.gap_patterns["missing_background"]:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                gaps.append(EnrichmentOpportunity(
                    gap_type="background",
                    description=f"Assumed knowledge of '{match}'",
                    priority="high",
                    suggested_content=f"Explain foundational concepts of {match}",
                    rationale=f"Document assumes familiarity with {match} without explanation"
                ))
        
        return gaps
    
    def _find_context_gaps(self, query: str, content: str, domain: str) -> List[EnrichmentOpportunity]:
        """Find gaps where broader context would be valuable."""
        gaps = []
        
        # Check for isolated information
        if len(content.split('.')) < 3:  # Very brief content
            gaps.append(EnrichmentOpportunity(
                gap_type="context",
                description="Limited document context for comprehensive understanding",
                priority="high", 
                suggested_content="Provide broader domain context and related concepts",
                rationale="Brief document content would benefit from additional context"
            ))
        
        # Check for domain-specific context needs
        context_needs = {
            "technical": ["architecture", "implementation", "best practices", "security"],
            "medical": ["patient safety", "contraindications", "evidence base", "guidelines"],
            "legal": ["jurisdiction", "precedents", "compliance", "recent changes"],
            "business": ["market context", "strategic implications", "ROI", "implementation"]
        }
        
        if domain in context_needs:
            for concept in context_needs[domain]:
                if concept.lower() not in content.lower() and concept.lower() in query.lower():
                    gaps.append(EnrichmentOpportunity(
                        gap_type="context",
                        description=f"Missing {domain} context: {concept}",
                        priority="medium",
                        suggested_content=f"Add {concept} considerations relevant to {domain}",
                        rationale=f"Query suggests interest in {concept} but document lacks this context"
                    ))
        
        return gaps
    
    def _find_implication_gaps(self, query: str, content: str, domain: str) -> List[EnrichmentOpportunity]:
        """Find gaps where practical implications would be valuable."""
        gaps = []
        
        # Check for actionable query without practical guidance
        action_words = ["how to", "implement", "apply", "use", "deploy", "execute"]
        if any(word in query.lower() for word in action_words):
            if not any(practical in content.lower() for practical in ["step", "process", "method", "approach", "procedure"]):
                gaps.append(EnrichmentOpportunity(
                    gap_type="implications",
                    description="Query seeks practical guidance but document lacks implementation details",
                    priority="high",
                    suggested_content="Add practical implementation considerations and common challenges",
                    rationale="User wants actionable guidance beyond what document provides"
                ))
        
        # Check for missing risk/consideration analysis
        if domain in ["medical", "legal", "technical"] and "risk" not in content.lower():
            gaps.append(EnrichmentOpportunity(
                gap_type="implications", 
                description=f"Missing risk considerations for {domain} domain",
                priority="medium",
                suggested_content=f"Add relevant risk factors and mitigation strategies for {domain}",
                rationale=f"{domain} decisions typically require risk assessment"
            ))
        
        return gaps
    
    def _find_currency_gaps(self, query: str, content: str, domain: str) -> List[EnrichmentOpportunity]:
        """Find gaps where current developments would be valuable."""
        gaps = []
        
        # Check for outdated references
        years = re.findall(r'\b(19|20)\d{2}\b', content)
        if years:
            latest_year = max(int(year) for year in years)
            if latest_year < 2023:  # Potentially outdated
                gaps.append(EnrichmentOpportunity(
                    gap_type="current_developments",
                    description=f"Document references may be from {latest_year}, potentially outdated",
                    priority="medium",
                    suggested_content="Add current developments and recent updates in the field",
                    rationale="User may benefit from more recent developments"
                ))
        
        # Check for queries about current state
        current_terms = ["latest", "current", "recent", "new", "modern", "today", "now"]
        if any(term in query.lower() for term in current_terms):
            gaps.append(EnrichmentOpportunity(
                gap_type="current_developments",
                description="Query asks about current state but document may not have latest information",
                priority="high",
                suggested_content="Provide recent developments and current best practices",
                rationale="User specifically seeks current information"
            ))
        
        return gaps
    
    def _prioritize_opportunities(self, opportunities: List[EnrichmentOpportunity], query: str) -> List[EnrichmentOpportunity]:
        """Prioritize enrichment opportunities based on relevance to query."""
        
        # Score opportunities based on various factors
        scored_opportunities = []
        query_words = set(query.lower().split())
        
        for opp in opportunities:
            score = 0
            
            # Priority weight
            priority_weights = {"high": 3, "medium": 2, "low": 1}
            score += priority_weights.get(opp.priority, 1)
            
            # Relevance to query
            opp_words = set(opp.description.lower().split())
            relevance = len(query_words.intersection(opp_words)) / len(query_words) if query_words else 0
            score += relevance * 2
            
            # Gap type importance
            gap_importance = {
                "background": 2.5,  # Often critical for understanding
                "implications": 2.0,  # Important for actionable guidance  
                "context": 1.5,     # Valuable for broader understanding
                "current_developments": 1.0  # Nice to have
            }
            score += gap_importance.get(opp.gap_type, 1)
            
            scored_opportunities.append((score, opp))
        
        # Sort by score (highest first) and return opportunities
        scored_opportunities.sort(key=lambda x: x[0], reverse=True)
        return [opp for score, opp in scored_opportunities]

class EnrichmentQualityController:
    """Ensures enrichment meets quality standards and doesn't duplicate content."""
    
    def __init__(self):
        self.redundancy_threshold = 0.7  # Similarity threshold for redundancy detection
    
    def should_enrich(self, document_content: str, query: str, enrichment_opportunities: List[EnrichmentOpportunity]) -> bool:
        """
        Determine if enrichment should be applied based on quality criteria.
        
        Args:
            document_content: Original document text
            query: User's query
            enrichment_opportunities: Identified opportunities
            
        Returns:
            Boolean indicating whether enrichment should proceed
        """
        
        # Check if document is already comprehensive
        if self._is_content_comprehensive(document_content, query):
            # Only enrich with high-priority opportunities
            high_priority_count = sum(1 for opp in enrichment_opportunities if opp.priority == "high")
            return high_priority_count > 0
        
        # Check if enrichment opportunities are valuable
        return len(enrichment_opportunities) > 0 and any(
            opp.priority in ["high", "medium"] for opp in enrichment_opportunities
        )
    
    def _is_content_comprehensive(self, content: str, query: str) -> bool:
        """Check if document content already comprehensively addresses the query."""
        
        # Simple heuristic - could be enhanced with semantic similarity
        query_words = set(query.lower().split())
        content_words = set(content.lower().split())
        
        # Remove common words
        common_words = {"the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
        query_words -= common_words
        content_words -= common_words
        
        if not query_words:
            return False
        
        coverage = len(query_words.intersection(content_words)) / len(query_words)
        word_density = len(content.split()) / max(len(query.split()), 1)
        
        # Consider comprehensive if high coverage and substantial content
        return coverage > 0.8 and word_density > 20
    
    def generate_enrichment_plan(self, opportunities: List[EnrichmentOpportunity]) -> Dict[str, Any]:
        """
        Generate a structured plan for applying enrichment.
        
        Returns:
            Dictionary with enrichment strategy and specific actions
        """
        
        plan = {
            "total_opportunities": len(opportunities),
            "high_priority": len([o for o in opportunities if o.priority == "high"]),
            "enrichment_layers": {},
            "transparency_notes": []
        }
        
        # Group opportunities by type
        for opp in opportunities:
            if opp.gap_type not in plan["enrichment_layers"]:
                plan["enrichment_layers"][opp.gap_type] = []
            plan["enrichment_layers"][opp.gap_type].append(opp)
        
        # Add transparency notes
        if plan["high_priority"] > 0:
            plan["transparency_notes"].append(f"Adding {plan['high_priority']} high-priority knowledge enhancements")
        
        if "background" in plan["enrichment_layers"]:
            plan["transparency_notes"].append("Providing foundational context for better understanding")
        
        if "implications" in plan["enrichment_layers"]:
            plan["transparency_notes"].append("Including practical implications and considerations")
        
        return plan