"""
Enhanced prompting system for intelligent general knowledge enrichment.
Provides domain-aware, context-sensitive knowledge integration.
"""

def generate_enhanced_general_knowledge_prompt(
    query: str,
    document_context: str,
    domain: str = "general",
    user_level: str = "intermediate",
    enrichment_strategy: str = "comprehensive"
) -> str:
    """
    Generate a dynamic, context-aware prompt for general knowledge enrichment.
    
    Args:
        query: User's question
        document_context: Content from user's documents
        domain: Subject domain (technical, medical, legal, business, etc.)
        user_level: User expertise (beginner, intermediate, advanced)
        enrichment_strategy: Type of enrichment needed
        
    Returns:
        Customized prompt for optimal knowledge enrichment
    """
    
    # Domain-specific enrichment guidelines
    domain_guidelines = {
        "technical": {
            "authorities": "IEEE standards, RFC specifications, industry best practices",
            "enrichment_focus": "implementation considerations, security implications, scalability factors",
            "citation_style": "technical standards and established frameworks"
        },
        "medical": {
            "authorities": "WHO guidelines, peer-reviewed research, clinical standards",
            "enrichment_focus": "safety considerations, evidence hierarchy, current treatment protocols",
            "citation_style": "evidence-based medical standards"
        },
        "legal": {
            "authorities": "statutory requirements, case law precedents, regulatory guidelines",
            "enrichment_focus": "jurisdictional considerations, recent legal developments, compliance requirements",
            "citation_style": "legal authorities and established precedents"
        },
        "business": {
            "authorities": "industry reports, established frameworks (like ISO, COSO), market research",
            "enrichment_focus": "strategic implications, market context, implementation considerations",
            "citation_style": "business frameworks and industry standards"
        },
        "general": {
            "authorities": "established academic sources, authoritative references, consensus research",
            "enrichment_focus": "conceptual foundations, broader context, practical applications",
            "citation_style": "authoritative sources and established principles"
        }
    }
    
    domain_info = domain_guidelines.get(domain, domain_guidelines["general"])
    
    # User level adaptations
    level_adaptations = {
        "beginner": {
            "approach": "Build foundational understanding with clear explanations",
            "complexity": "Avoid technical jargon, provide step-by-step context",
            "depth": "Focus on core concepts and practical applications"
        },
        "intermediate": {
            "approach": "Bridge current knowledge with advanced concepts", 
            "complexity": "Moderate technical detail with clear explanations",
            "depth": "Connect concepts and explore implications"
        },
        "advanced": {
            "approach": "Provide nuanced insights and cutting-edge context",
            "complexity": "Full technical depth with professional language",
            "depth": "Explore complex relationships and future implications"
        }
    }
    
    level_info = level_adaptations.get(user_level, level_adaptations["intermediate"])
    
    return f"""You are an expert knowledge synthesis assistant specializing in {domain} domain.

ENRICHMENT MISSION:
Your goal is to intelligently enhance the user's document-based answer with valuable general knowledge that adds depth, context, and actionable insights without redundancy.

USER CONTEXT:
- Query: {query}
- Domain: {domain}
- Expertise Level: {user_level}
- Enrichment Strategy: {enrichment_strategy}

PROGRESSIVE KNOWLEDGE INTEGRATION FRAMEWORK:

## PHASE 1: DOCUMENT FOUNDATION (Always Start Here)
1. **Lead with Document Facts**
   • Begin with: "Based on your documents, [key findings]..."
   • Establish what the user already knows as the foundation

## PHASE 2: INTELLIGENT ENRICHMENT (Add Value Strategically)
2. **Gap Analysis & Enhancement Strategy**
   • Identify what's missing or could benefit from broader context
   • {level_info['approach']}
   • Focus on: {domain_info['enrichment_focus']}

3. **Knowledge Enrichment Layers** (Apply selectively based on relevance):

   **Layer A: Foundational Context** (When documents lack background)
   • Introduce with: "To provide essential context, {domain_info['citation_style']} establish that..."
   • Add prerequisites, definitions, or historical background
   • {level_info['complexity']}

   **Layer B: Broader Connections** (When documents are isolated)
   • Introduce with: "This connects to broader {domain} principles where..."
   • Link to established frameworks, related concepts, methodologies
   • Show how document content fits into larger knowledge landscape

   **Layer C: Practical Implications** (When documents lack actionable insight)
   • Introduce with: "Drawing on {domain_info['authorities']}, practical considerations include..."
   • Add implementation guidance, common pitfalls, success factors
   • {level_info['depth']}

   **Layer D: Current Developments** (When documents may be outdated)
   • Introduce with: "Recent developments in {domain} indicate..."
   • Add emerging trends, updated best practices, new research
   • Note any potential changes since document creation

## PHASE 3: SYNTHESIS & INTEGRATION
4. **Cohesive Knowledge Weaving**
   • Create seamless flow from document foundation through enrichment layers
   • Ensure each enrichment clearly adds value beyond document content
   • Maintain clear source attribution throughout

## CRITICAL QUALITY CONTROLS:

**Relevance Filter:**
- Only add general knowledge that directly enhances understanding of the user's query
- Avoid tangential information or generic background
- Each enrichment should answer: "How does this help the user better understand or act on their question?"

**Redundancy Prevention:**
- Never repeat information already covered in the documents
- If documents and general knowledge conflict, acknowledge both perspectives
- Focus on complementary information, not overlapping content

**Transparency Requirements:**
- Use {domain_info['citation_style']} for general knowledge enrichment
- Clearly signal transitions between document content and enrichment
- If uncertain about any enrichment, state confidence level

**Domain-Specific Quality Standards:**
- {domain_info['enrichment_focus']}
- Reference {domain_info['authorities']} when appropriate
- Maintain {domain} professional standards and terminology

## RESPONSE STRUCTURE:
Format your response with clear sections:

**📋 Document Summary:** What your documents tell us [with citations]

**🔍 Enhanced Understanding:** Relevant enrichment that adds value [with clear source attribution]

**💡 Key Insights:** Synthesis of document + general knowledge 

**🎯 Practical Takeaways:** Actionable guidance for the user

## ENRICHMENT DECISION MATRIX:
Before adding any general knowledge, assess:
- ✅ Does this fill a genuine gap in the document content?
- ✅ Does this help the user better understand or act on their query?
- ✅ Is this information reliable and from authoritative sources?
- ✅ Does this complement rather than repeat document content?
- ✅ Is the source attribution clear and appropriate?

Only proceed with enrichment if ALL criteria are met.

**WEB SEARCH INTEGRATION:**
When web search results are provided alongside document content:
- Include relevant web links directly in your response text where they add value
- Format links as: [Link Text](URL) or mention URLs directly in context
- Integrate web sources naturally into your explanation, not as a separate list
- Use web results to provide current examples, tutorials, or additional resources that complement document knowledge

Remember: Your goal is to make the user's document-based knowledge more powerful and actionable, not to replace it with generic information."""

def get_domain_from_context(document_context: str, query: str) -> str:
    """
    Analyze document context and query to determine the domain.
    Returns appropriate domain classification.
    """
    # This could be enhanced with ML classification
    domain_keywords = {
        "technical": ["code", "software", "programming", "algorithm", "system", "technical", "engineering"],
        "medical": ["patient", "treatment", "diagnosis", "medical", "health", "clinical", "therapy"],
        "legal": ["law", "legal", "contract", "regulation", "compliance", "statute", "court"],
        "business": ["business", "strategy", "market", "financial", "management", "organization", "revenue"]
    }
    
    context_text = (document_context + " " + query).lower()
    
    for domain, keywords in domain_keywords.items():
        if any(keyword in context_text for keyword in keywords):
            return domain
    
    return "general"

def assess_user_expertise_level(document_context: str, query: str) -> str:
    """
    Analyze user's query and documents to estimate expertise level.
    """
    # Simple heuristic - could be enhanced with more sophisticated analysis
    beginner_indicators = ["what is", "how to", "basic", "simple", "introduction", "beginner"]
    advanced_indicators = ["optimization", "advanced", "complex", "sophisticated", "integration", "framework"]
    
    query_lower = query.lower()
    context_lower = document_context.lower()
    combined_text = query_lower + " " + context_lower
    
    beginner_score = sum(1 for indicator in beginner_indicators if indicator in combined_text)
    advanced_score = sum(1 for indicator in advanced_indicators if indicator in combined_text)
    
    if beginner_score > advanced_score:
        return "beginner"
    elif advanced_score > beginner_score * 2:
        return "advanced"
    else:
        return "intermediate"

def determine_enrichment_strategy(query: str, document_context: str) -> str:
    """
    Determine what type of enrichment would be most valuable.
    """
    query_lower = query.lower()
    
    if any(word in query_lower for word in ["how to", "steps", "guide", "process"]):
        return "procedural"
    elif any(word in query_lower for word in ["what is", "define", "meaning", "concept"]):
        return "conceptual"
    elif any(word in query_lower for word in ["why", "reason", "cause", "analysis"]):
        return "analytical"
    elif any(word in query_lower for word in ["best", "recommend", "should", "advice"]):
        return "advisory"
    else:
        return "comprehensive"