"""
Few-shot examples for general knowledge enrichment to demonstrate best practices.
These examples show the LLM how to properly enrich document-based responses.
"""

ENRICHMENT_EXAMPLES = {
    "technical": {
        "query": "How do I implement caching in my web application?",
        "document_content": "The application uses Redis for session storage. Redis is configured with 2GB memory limit.",
        "poor_enrichment": """
Based on your documents, your application uses Redis for session storage with 2GB memory limit. [Doc 1]

Caching is a technique used to store frequently accessed data in memory for faster retrieval. There are many types of caching like browser caching, CDN caching, database caching, and application-level caching. Redis is an in-memory data structure store that can be used as a database, cache, and message broker.
        """,
        "good_enrichment": """
📋 **Document Summary:** Your application currently uses Redis for session storage with a 2GB memory limit. [Doc 1]

🔍 **Enhanced Understanding:** Since you already have Redis infrastructure, this provides an excellent foundation for application-level caching. Drawing on established web architecture principles, Redis excels as a caching layer due to its in-memory performance and data structure versatility.

💡 **Key Insights:** Your existing Redis setup can be expanded for multi-purpose caching:
• **Session data** (current use) + **Application cache** (new use)
• **Memory allocation strategy**: Consider partitioning your 2GB between sessions and cache data
• **Data structure optimization**: Redis supports strings, hashes, lists, and sets - choose based on your data patterns

🎯 **Practical Takeaways:**
• Implement cache-aside pattern for database query results
• Set appropriate TTL (time-to-live) values based on data volatility
• Monitor memory usage to prevent eviction of session data
• Consider Redis clustering if you need to scale beyond 2GB

According to Redis best practices, start with simple string caching for database queries and expand to more complex structures as needed.
        """
    },
    
    "medical": {
        "query": "What are the side effects of the medication mentioned in my report?",
        "document_content": "Patient prescribed Metformin 500mg twice daily for type 2 diabetes management. No current contraindications noted.",
        "poor_enrichment": """
Based on your documents, you were prescribed Metformin 500mg twice daily for type 2 diabetes. [Doc 1]

Metformin is a medication used to treat type 2 diabetes. It works by decreasing glucose production in the liver. Side effects can include nausea, vomiting, diarrhea, and lactic acidosis.
        """,
        "good_enrichment": """
📋 **Document Summary:** Your report shows a prescription for Metformin 500mg twice daily for type 2 diabetes management, with no current contraindications noted. [Doc 1]

🔍 **Enhanced Understanding:** According to current clinical guidelines from the American Diabetes Association, Metformin is the first-line treatment for type 2 diabetes due to its proven efficacy and safety profile.

💡 **Key Side Effect Categories:**

**Common (affect 10-25% of patients):**
• Gastrointestinal: nausea, diarrhea, stomach upset
• Usually occur in first few weeks and often resolve with continued use

**Less Common but Important:**
• Vitamin B12 deficiency with long-term use
• Metallic taste in mouth

**Rare but Serious:**
• Lactic acidosis (extremely rare: <1 in 100,000 patient-years)

🎯 **Practical Management:**
• Take with food to minimize GI symptoms
• Start with lower dose and gradually increase (your 500mg twice daily is standard)
• Annual B12 level monitoring recommended for long-term use
• Report persistent severe GI symptoms to your healthcare provider

**Important**: Your document notes no contraindications, which is appropriate since major contraindications include severe kidney disease and certain heart conditions.
        """
    },
    
    "business": {
        "query": "How should we approach our market expansion strategy?",
        "document_content": "Company currently has 15% market share in the domestic market. Revenue growth has been 8% annually. Profit margin is 12%.",
        "poor_enrichment": """
Based on your documents, your company has 15% market share domestically with 8% annual revenue growth and 12% profit margin. [Doc 1]

Market expansion can be done through geographic expansion, product diversification, or market penetration. Companies need to consider their resources, competition, and market conditions.
        """,
        "good_enrichment": """
📋 **Document Summary:** Your company currently holds 15% domestic market share with 8% annual revenue growth and 12% profit margins. [Doc 1]

🔍 **Enhanced Understanding:** Drawing on strategic management frameworks, your metrics indicate a strong foundation for expansion. According to Ansoff's Growth Matrix, your position suggests multiple viable expansion pathways.

💡 **Strategic Assessment Based on Your Metrics:**

**Market Position Strength:**
• 15% market share indicates established brand presence
• 8% growth rate (if above market average) shows competitive advantage
• 12% profit margin provides expansion capital and buffer

**Recommended Strategic Framework:**
According to BCG growth strategies, companies with your profile typically succeed with:

1. **Market Penetration** (lowest risk): Increase share in current market
   - Your 15% share suggests significant room for growth
   - Leverage existing customer relationships and distribution

2. **Geographic Expansion** (medium risk): New territories, same products
   - Strong domestic performance indicates scalable business model
   - Consider markets with similar demographics/regulations first

3. **Product Development** (higher risk): New products, current market
   - 12% margins suggest R&D investment capacity

🎯 **Practical Next Steps:**
• Conduct market saturation analysis in current territory
• Assess competitive landscape in target expansion areas
• Stress-test financial projections with 15-20% lower margins for new markets
• Consider partnership/acquisition vs. organic growth based on timeline and resources

Industry benchmarks suggest companies with similar profiles often achieve 20-30% market share before geographic expansion becomes more attractive than market penetration.
        """
    }
}

def get_few_shot_examples(domain: str, max_examples: int = 1) -> str:
    """
    Get few-shot examples for a specific domain to include in prompts.
    
    Args:
        domain: The domain to get examples for
        max_examples: Maximum number of examples to include
        
    Returns:
        Formatted few-shot examples string
    """
    
    if domain not in ENRICHMENT_EXAMPLES:
        domain = "technical"  # Default fallback
    
    example = ENRICHMENT_EXAMPLES[domain]
    
    return f"""
## EXAMPLE OF EXCELLENT ENRICHMENT:

**Query:** {example['query']}

**Document Content:** {example['document_content']}

**❌ Poor Enrichment (avoid this approach):**
{example['poor_enrichment'].strip()}

**✅ Excellent Enrichment (follow this approach):**
{example['good_enrichment'].strip()}

**Why the excellent example works:**
• Starts with clear document summary with citations
• Adds relevant background that enhances understanding
• Provides structured, actionable insights
• Uses authoritative sources and frameworks
• Maintains clear separation between document and enrichment content
• Focuses on making the user's knowledge more powerful and actionable

---

Now apply this same approach to the user's actual query and documents:
"""

def get_domain_specific_enrichment_guidelines(domain: str) -> str:
    """
    Get domain-specific guidelines for enrichment quality.
    
    Args:
        domain: The subject domain
        
    Returns:
        Domain-specific enrichment guidelines
    """
    
    guidelines = {
        "technical": """
**Technical Domain Enrichment Guidelines:**
• Reference established standards (IEEE, RFC, ISO) when relevant
• Include security, performance, and scalability considerations
• Provide practical implementation guidance and common pitfalls
• Consider compatibility and integration aspects
• Mention testing and validation approaches when applicable
        """,
        
        "medical": """
**Medical Domain Enrichment Guidelines:**
• Prioritize patient safety and evidence-based information
• Reference clinical guidelines (AMA, WHO, specialty organizations)
• Include contraindications and monitoring requirements
• Provide context for clinical decision-making
• Note when professional consultation is recommended
• Maintain appropriate medical disclaimers
        """,
        
        "legal": """
**Legal Domain Enrichment Guidelines:**
• Reference relevant statutes, regulations, and case law
• Consider jurisdictional variations and recent changes
• Include compliance and risk management perspectives
• Provide practical implementation guidance for legal requirements
• Note when professional legal counsel is recommended
• Maintain appropriate legal disclaimers
        """,
        
        "business": """
**Business Domain Enrichment Guidelines:**
• Reference established business frameworks (Porter's Five Forces, BCG Matrix, etc.)
• Include strategic, operational, and financial perspectives
• Consider market dynamics and competitive landscape
• Provide actionable implementation guidance
• Include risk assessment and mitigation strategies
• Reference industry benchmarks and best practices
        """,
        
        "general": """
**General Domain Enrichment Guidelines:**
• Use authoritative academic and professional sources
• Provide balanced perspectives on complex topics
• Include practical applications and real-world context
• Consider multiple stakeholder viewpoints
• Provide actionable guidance where appropriate
• Maintain objectivity and professional tone
        """
    }
    
    return guidelines.get(domain, guidelines["general"])

def create_few_shot_enhanced_prompt(base_prompt: str, domain: str) -> str:
    """
    Enhance a base prompt with few-shot examples and domain-specific guidelines.
    
    Args:
        base_prompt: The base enrichment prompt
        domain: The subject domain
        
    Returns:
        Enhanced prompt with examples and guidelines
    """
    
    few_shot_examples = get_few_shot_examples(domain)
    domain_guidelines = get_domain_specific_enrichment_guidelines(domain)
    
    enhanced_prompt = f"""{base_prompt}

{few_shot_examples}

{domain_guidelines}

Remember: Your goal is to create responses as excellent as the example above, tailored to the user's specific query and documents."""
    
    return enhanced_prompt