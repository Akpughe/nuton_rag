import os
import json
import requests
from typing import List, Dict, Any, Tuple, Optional
from dotenv import load_dotenv
from openai import OpenAI
from exa_py import Exa

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EXA_API_KEY = os.getenv("EXA_API_KEY")  # For Exa semantic search API

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Initialize Exa client
exa_client = Exa(api_key=EXA_API_KEY) if EXA_API_KEY else None

# Quality domains for educational content
QUALITY_DOMAINS = [
    # Video & Courses
    "youtube.com", "coursera.org", "udemy.com", "edx.org",
    "khanacademy.org", "skillshare.com", "brilliant.org", "pluralsight.com",

    # Tech & Programming Learning
    "freecodecamp.org", "w3schools.com", "geeksforgeeks.org", "tutorialspoint.com",
    "codecademy.com", "css-tricks.com", "hashnode.com", "dev.to",

    # Blogs & Articles
    "medium.com", "towardsdatascience.com", "dzone.com",
    "infoq.com", "smashingmagazine.com",

    # Academic & Research
    "mit.edu", "stanford.edu", "harvard.edu", "cam.ac.uk",
    "nature.com", "sciencedirect.com", "springer.com",
    "arxiv.org", "researchgate.net", "jstor.org",

    # Developer Resources
    "github.com", "gitlab.com", "bitbucket.org",
    "stackoverflow.com", "superuser.com",

    # General Knowledge & Trusted Media
    "wikipedia.org", "britannica.com", "nationalgeographic.com",
    "npr.org", "bbc.com", "nytimes.com"
]


def analyze_document_context(rag_context: str, query: str) -> Dict[str, Any]:
    """
    Analyze the RAG context and query to understand intent and generate targeted search guidance.
    
    Args:
        rag_context: The context extracted from user's documents
        query: The user's original query
        
    Returns:
        Dict containing intent analysis and search guidance
    """
    analysis_prompt = f"""
    Analyze the user's query and their document context to understand their true intent and needs:
    
    User Query: {query}
    
    Document Context:
    {rag_context}
    
    Determine:
    1. Query Intent: What is the user really trying to achieve? (learn, build, implement, understand, become, etc.)
    2. Action Level: Are they seeking definitions, tutorials, guides, or implementation help?
    3. Context Domain: What specific field/domain is this about?
    4. User's Current Level: Based on their documents, what's their expertise level?
    5. Gap Analysis: What specific knowledge or guidance would help them most?
    
    Provide a JSON response with:
    {{
        "query_intent": "what the user is really trying to achieve",
        "intent_type": "definition/tutorial/guide/implementation/becoming/learning",
        "action_focus": "specific actionable outcome they want",
        "domain": "specific domain/field",
        "user_level": "beginner/intermediate/advanced",
        "knowledge_gaps": ["specific gap1", "specific gap2", ...],
        "search_focus": "what type of content would be most helpful",
        "key_concepts": ["concept1", "concept2", ...],
        "preferred_formats": ["tutorials", "guides", "videos", "courses", "articles"]
    }}
    """

    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert at analyzing educational content and understanding learning contexts. Respond only with valid JSON."},
                {"role": "user", "content": analysis_prompt}
            ],
            temperature=0.3
        )
        response_content = response.choices[0].message.content.strip()

        
        # Try to extract JSON if it's wrapped in markdown code blocks
        if response_content.startswith('```json'):
            response_content = response_content.replace('```json', '').replace('```', '').strip()
        elif response_content.startswith('```'):
            response_content = response_content.replace('```', '').strip()
            
        analysis = json.loads(response_content)

        return analysis
    except Exception as e:
        print(f"Error in analyze_document_context: {e}")
        # Fallback analysis based on query keywords
        intent_type = "learning"
        if "how to" in query.lower() or "how can" in query.lower():
            intent_type = "tutorial"
        elif "what is" in query.lower() or "define" in query.lower():
            intent_type = "definition"
        elif "build" in query.lower() or "create" in query.lower():
            intent_type = "implementation"
        elif "become" in query.lower():
            intent_type = "becoming"
            
        return {
            "query_intent": f"Understanding and learning about the topic in the query",
            "intent_type": intent_type,
            "action_focus": "gain practical knowledge and understanding",
            "domain": "general",
            "user_level": "intermediate",
            "knowledge_gaps": ["current developments", "practical applications"],
            "search_focus": "actionable guides and tutorials",
            "key_concepts": [],
            "preferred_formats": ["tutorials", "guides", "articles"]
        }

def generate_contextual_search_queries(query: str, context_analysis: Dict[str, Any]) -> List[str]:
    """
    Generate purposeful, intent-driven search queries that return actionable resources.
    
    Args:
        query: The user's original query
        context_analysis: Intent analysis from analyze_document_context
        
    Returns:
        List of targeted search queries focused on user's true intent
    """
    intent_type = context_analysis.get("intent_type", "learning")
    action_focus = context_analysis.get("action_focus", "")
    domain = context_analysis.get("domain", "general")
    search_focus = context_analysis.get("search_focus", "")
    user_level = context_analysis.get("user_level", "intermediate")
    preferred_formats = context_analysis.get("preferred_formats", ["guides", "tutorials"])
    
    query_generation_prompt = f"""
    Generate 3-5 purposeful search queries that will return actionable, high-quality resources to help the user achieve their true intent.
    
    Original Query: {query}
    User's True Intent: {context_analysis.get("query_intent", "")}
    Intent Type: {intent_type}
    Action Focus: {action_focus}
    Domain: {domain}
    User Level: {user_level}
    Search Focus: {search_focus}
    Preferred Formats: {', '.join(preferred_formats)}
    
    Create search queries that:
    1. Target the user's TRUE INTENT, not just keywords from their query
    2. Return resources that help them TAKE ACTION or ACHIEVE their goal
    3. Find step-by-step guides, tutorials, and practical resources
    4. Prioritize authoritative, educational content over generic definitions
    5. Match their skill level and provide progression paths
    
    Search Strategy Examples:
    - If intent is "becoming": Search for "how to become [X]", career guides, step-by-step paths
    - If intent is "implementation": Search for tutorials, code examples, practical guides
    - If intent is "learning": Search for comprehensive guides, courses, structured learning paths
    - If intent is "definition" but user seems ready for more: Search for in-depth explanations and applications
    
    Focus on queries that will return:
    - YouTube tutorials and walkthroughs
    - Step-by-step guides and blog posts
    - Courses and structured learning materials
    - Real-world examples and case studies
    - Tools and resources for implementation
    
    Return as a JSON array of search query strings:
    ["query1", "query2", "query3", ...]
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Generate targeted search queries that complement existing document knowledge. Respond only with valid JSON array."},
                {"role": "user", "content": query_generation_prompt}
            ],
            temperature=0.5
        )
        
        response_content = response.choices[0].message.content.strip()
        
        # Try to extract JSON if it's wrapped in markdown code blocks
        if response_content.startswith('```json'):
            response_content = response_content.replace('```json', '').replace('```', '').strip()
        elif response_content.startswith('```'):
            response_content = response_content.replace('```', '').strip()
            
        queries = json.loads(response_content)
        return queries[:5]  # Limit to 5 queries max
    except Exception as e:
        print(f"Error in generate_contextual_search_queries: {e}")
        # Generate intelligent fallback queries based on intent
        fallback_queries = []
        
        if intent_type == "becoming":
            fallback_queries = [
                f"how to become {query.replace('how can i become', '').replace('what is', '').strip()}",
                f"step by step guide {query}",
                f"career path {domain}"
            ]
        elif intent_type == "tutorial" or intent_type == "implementation":
            fallback_queries = [
                f"tutorial {query}",
                f"step by step {query}",
                f"guide {query}"
            ]
        elif intent_type == "definition":
            fallback_queries = [
                f"{query} explained simply",
                f"{query} beginner guide",
                f"{query} examples"
            ]
        else:
            fallback_queries = [
                f"learn {query}",
                f"{query} tutorial",
                f"{query} guide"
            ]
            
        return fallback_queries[:3]

def perform_web_search(
    search_query: str,
    category: Optional[str] = None,
    intent_type: Optional[str] = None
) -> List[Dict[str, str]]:
    """
    Perform web search using Exa SDK (semantic search optimized for LLMs).

    Uses "auto" search type (intelligently blends neural + keyword),
    highlights for context-relevant excerpts, and domain filtering for quality.

    Args:
        search_query: The search query string
        category: Optional Exa category filter ("tutorial", "github", "research paper", etc.)
        intent_type: Optional intent type to map to category if not provided

    Returns:
        List of search results with title, url, and snippet (from highlights)
    """
    if not exa_client:
        print("No EXA_API_KEY found or Exa client not initialized")
        return []

    try:
        # Map intent type to Exa category if category not explicitly provided
        if not category and intent_type:
            category_mapping = {
                # Learning & Guides
                "tutorial": "tutorial",
                "guide": "tutorial",
                "how-to": "tutorial",
                "learning": "course",
                "becoming": "career path",

                # Technical Implementation
                "implementation": "github",
                "code": "github",
                "repository": "github",
                "demo": "example",

                # Knowledge Types
                "definition": None,  # auto-handle with dictionary / encyclopedia
                "concept": "concept",
                "theory": "concept",
                "principle": "concept",

                # Academic / Research
                "research": "research paper",
                "study": "research paper",
                "thesis": "research paper",
                "publication": "research paper",

                # Practical Application
                "application": "application",
                "case study": "application",
                "example": "application",
                "use-case": "application",

                # Technology / Ecosystem
                "tool": "tool",
                "framework": "framework",
                "library": "library",
                "package": "library",
                "module": "library",
                "sdk": "library",
                "database": "database",
                "api": "api",
                "service": "api",

                # General Knowledge
                "article": "article",
                "blog": "article",
                "documentation": "docs",
                "faq": "docs",
                "wiki": "docs",

                # Multimedia
                "video": "video",
                "lecture": "video",
                "course": "course",
                "podcast": "podcast",
            }

            category = category_mapping.get(intent_type)

        # Build search parameters
        search_params = {
            "query": search_query,
            "type": "auto",  # Intelligently blends neural + keyword (Exa's recommendation)
            "num_results": 5,
            "use_autoprompt": True,  # Let Exa optimize the query
            "include_domains": QUALITY_DOMAINS,  # Filter at API level for speed
            "highlights": {
                "query": search_query,
                "num_sentences": 3,
                "highlights_per_url": 1
            }
        }

        # Add category filter if available
        if category:
            search_params["category"] = category

        # Perform search with the Exa SDK
        results = exa_client.search_and_contents(**search_params)

        # Extract and format results
        formatted_results = []
        for result in results.results:
            # Use highlights if available, fall back to text excerpt
            snippet = ""
            if result.highlights and len(result.highlights) > 0:
                snippet = " ".join(result.highlights)
            elif result.text:
                snippet = result.text[:500]

            formatted_results.append({
                "title": result.title or "",
                "url": result.url or "",
                "snippet": snippet
            })

        return formatted_results

    except Exception as e:
        print(f"Exa SDK search failed: {e}")
        # Fallback to empty results
        return []

def perform_contextual_websearch(
    search_queries: List[str],
    context_analysis: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Perform multiple web searches and aggregate results with intent-aware filtering.

    Args:
        search_queries: List of search query strings
        context_analysis: Optional context analysis with intent information

    Returns:
        List of aggregated search results (max 5)
    """
    all_results = []

    # Extract intent type from context analysis
    intent_type = context_analysis.get("intent_type") if context_analysis else None

    for query in search_queries:
        # Pass intent type to enable category filtering at API level
        results = perform_web_search(query, intent_type=intent_type)
        for result in results:
            result["search_query"] = query
            all_results.append(result)

    # Remove duplicates based on URL
    seen_urls = set()
    unique_results = []
    for result in all_results:
        url = result.get("url", "")
        if url and url not in seen_urls:
            seen_urls.add(url)
            unique_results.append(result)

    # Return top 5 results (already filtered by domain at API level, so less ranking needed)
    return unique_results[:5]

def rank_and_filter_results(results: List[Dict[str, Any]], search_queries: List[str]) -> List[Dict[str, Any]]:
    """
    Rank and filter search results to prioritize actionable, high-quality content.
    
    Args:
        results: List of search results
        search_queries: Original search queries for context
        
    Returns:
        Ranked and filtered results prioritizing actionable content
    """
    if not results:
        return results
    
    # Score each result based on actionable content indicators
    scored_results = []
    
    for result in results:
        title = result.get("title", "").lower()
        snippet = result.get("snippet", "").lower()
        url = result.get("url", "").lower()
        
        score = 0
        
        # Prioritize actionable content
        actionable_keywords = [
            "how to", "tutorial", "guide", "step by step", "walkthrough", 
            "learn", "course", "training", "beginner", "complete guide",
            "implementation", "build", "create", "develop", "become"
        ]
        
        for keyword in actionable_keywords:
            if keyword in title:
                score += 3
            if keyword in snippet:
                score += 2
        
        # Prioritize educational platforms and quality sources
        quality_domains = [
            "youtube.com", "coursera.org", "udemy.com", "medium.com",
            "towards", "dev.to", "freecodecamp", "khan", "mit.edu",
            "stanford.edu", "github.com", "stackoverflow.com"
        ]
        
        for domain in quality_domains:
            if domain in url:
                score += 4
                break
        
        # Boost video content for tutorials
        if "youtube" in url or "video" in title or "watch" in title:
            score += 2
        
        # Penalize overly promotional content
        promotional_keywords = ["buy", "sale", "discount", "price", "cheap"]
        for keyword in promotional_keywords:
            if keyword in title or keyword in snippet:
                score -= 2
        
        scored_results.append((score, result))
    
    # Sort by score (highest first) and return results
    scored_results.sort(key=lambda x: x[0], reverse=True)
    return [result for score, result in scored_results]

def synthesize_rag_and_web_results(
    query: str,
    rag_context: str,
    web_results: List[Dict[str, Any]],
    context_analysis: Dict[str, Any],
    system_prompt: str,
    has_general_knowledge: bool = True,
    conversation_history: Optional[List[Dict[str, str]]] = None
) -> Tuple[str, List[Dict]]:
    """
    Synthesize RAG results with web search results using GPT-4o, leveraging general knowledge.
    
    Args:
        query: The user's original query
        rag_context: Context from user's documents
        web_results: Results from web search
        context_analysis: Analysis of document context
        system_prompt: Base system prompt
        has_general_knowledge: Whether general knowledge is enabled for richer synthesis
        
    Returns:
        Tuple of (synthesized answer, combined sources)
    """
    # Format web results for context
    web_context = ""
    if web_results:
        web_context = "\n".join([
            f"• {result['title']}\n  {result['snippet']}\n  Source: {result['url']}"
            for result in web_results
        ])
    
    # Enhanced system prompt for intent-driven synthesis
    user_intent = context_analysis.get("query_intent", "learning")
    intent_type = context_analysis.get("intent_type", "learning")
    action_focus = context_analysis.get("action_focus", "")
    
    # Build knowledge sources description based on general knowledge availability
    if has_general_knowledge:
        knowledge_sources = """three powerful knowledge sources:
1. The user's uploaded documents (their foundation knowledge)
2. Your built-in general knowledge (established concepts and frameworks)
3. Curated web search results (current, actionable resources and guidance)"""
        
        general_knowledge_guidance = """- Enrich the response with your domain expertise to provide foundational understanding, broader context, and connections
- Create a cohesive response that builds from document foundation, through expert insights, to current applications (web results)
- Use your knowledge to explain concepts, provide historical context, and make connections that enhance understanding"""
        
        integration_strategy = """KNOWLEDGE ENRICHMENT STRATEGY:
- Start with the user's document content as the foundation
- Enhance with your domain expertise to provide depth, context, and connections
- Bridge concepts with foundational knowledge and professional insights"""
    else:
        knowledge_sources = """two powerful knowledge sources:
1. The user's uploaded documents (their foundation knowledge)
2. Curated web search results (current, actionable resources and guidance)"""
        
        general_knowledge_guidance = """- Create a cohesive response that builds from document context to current applications (web results)"""
        
        integration_strategy = """INTEGRATION STRATEGY:
- Start with the user's document context as the foundation"""
    
    synthesis_prompt = f"""{system_prompt}

You are synthesizing information to help the user achieve their specific intent and goals.

USER'S TRUE INTENT: {user_intent}
INTENT TYPE: {intent_type}
ACTION FOCUS: {action_focus}

You are combining {knowledge_sources}

SYNTHESIS GUIDELINES:
- Focus on helping the user ACHIEVE their specific intent, not just providing information
- If they want to "become" something, provide clear pathways and next steps
- If they want to "learn" something, structure the response as a learning journey
- If they want to "implement" something, provide actionable steps and resources
- Use document content as context for their current understanding
{general_knowledge_guidance}
- Use web results to fill knowledge gaps with current, actionable next steps
- Prioritize resources that match their intent (tutorials for learning, career guides for becoming, etc.)

{integration_strategy}
- Bridge to the user's document context to show relevance
- Culminate with current, actionable resources from web search
- Ensure the response flows logically from theory to practice

Domain Context: {context_analysis.get('domain', 'general')}
User Level: {context_analysis.get('user_level', 'intermediate')}
Key Concepts: {', '.join(context_analysis.get('key_concepts', []))}
"""

    user_message = f"""USER'S QUERY: {query}
USER'S TRUE INTENT: {user_intent}

=== FOUNDATION KNOWLEDGE (From User's Documents) ===
{rag_context}

=== ACTIONABLE RESOURCES & GUIDANCE (From Curated Web Search) ===
{web_context if web_context else "No additional web results found."}

TASK: Help the user achieve their specific intent by:
1. Building on their existing document knowledge as the foundation
2. Providing actionable next steps and resources from the web results
3. Creating a clear pathway toward their goal
4. Focusing on practical, implementable guidance rather than just information

Provide a response that helps them take concrete action toward their intent."""

    # Build messages array with conversation history
    messages = [{"role": "system", "content": synthesis_prompt}]

    # Insert conversation history for context continuity
    if conversation_history:
        messages.extend(conversation_history)

    # Add current user message
    messages.append({"role": "user", "content": user_message})

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.3
        )

        answer = response.choices[0].message.content
        
        # Combine sources for citation tracking
        combined_sources = []
        
        # Add RAG sources (this would need to be passed from the calling function)
        # For now, we'll add web sources
        for result in web_results:
            combined_sources.append({
                "title": result["title"],
                "url": result["url"],
                "snippet": result["snippet"],
                "source_type": "web"
            })
        
        return answer, combined_sources
        
    except Exception as e:
        raise Exception(f"Synthesis with GPT-4o failed: {e}")