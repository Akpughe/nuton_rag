import os
import json
import requests
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERP_API_KEY = os.getenv("SERP_API_KEY")  # For Google Search API
BING_API_KEY = os.getenv("BING_API_KEY")  # Alternative search API

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

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

def perform_web_search(search_query: str) -> List[Dict[str, str]]:
    """
    Perform web search using available search APIs.
    
    Args:
        search_query: The search query string
        
    Returns:
        List of search results with title, url, and snippet
    """
    results = []
    
    # Try SerpAPI first (Google Search)
    if SERP_API_KEY:
        try:
            url = "https://serpapi.com/search"
            params = {
                "q": search_query,
                "api_key": SERP_API_KEY,
                "engine": "google",
                "num": 5
            }
            
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            for result in data.get("organic_results", [])[:5]:
                results.append({
                    "title": result.get("title", ""),
                    "url": result.get("link", ""),
                    "snippet": result.get("snippet", "")
                })
                
        except Exception as e:
            print(f"SerpAPI search failed: {e}")
    
    # Try Bing Search API as fallback
    elif BING_API_KEY:
        try:
            url = "https://api.bing.microsoft.com/v7.0/search"
            headers = {"Ocp-Apim-Subscription-Key": BING_API_KEY}
            params = {"q": search_query, "count": 5}
            
            response = requests.get(url, headers=headers, params=params, timeout=10)
            data = response.json()
            
            for result in data.get("webPages", {}).get("value", [])[:5]:
                results.append({
                    "title": result.get("name", ""),
                    "url": result.get("url", ""),
                    "snippet": result.get("snippet", "")
                })
                
        except Exception as e:
            print(f"Bing search failed: {e}")
    
    # If no API keys or all failed, return empty results
    if not results:
        print(f"No search results for: {search_query}")
    
    return results

def perform_contextual_websearch(search_queries: List[str]) -> List[Dict[str, Any]]:
    """
    Perform multiple web searches and aggregate results.
    
    Args:
        search_queries: List of search query strings
        
    Returns:
        List of aggregated search results
    """
    all_results = []
    
    for query in search_queries:
        results = perform_web_search(query)
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
    
    return rank_and_filter_results(unique_results, search_queries)[:5]  # Limit to 5 best results

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
    system_prompt: str
) -> Tuple[str, List[Dict]]:
    """
    Synthesize RAG results with web search results using GPT-4o.
    
    Args:
        query: The user's original query
        rag_context: Context from user's documents
        web_results: Results from web search
        context_analysis: Analysis of document context
        system_prompt: Base system prompt
        
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
    
    synthesis_prompt = f"""{system_prompt}

You are synthesizing information to help the user achieve their specific intent and goals.

USER'S TRUE INTENT: {user_intent}
INTENT TYPE: {intent_type}
ACTION FOCUS: {action_focus}

You are combining two sources:
1. The user's uploaded documents (their foundation knowledge)
2. Curated web search results (actionable resources and guidance)

SYNTHESIS GUIDELINES:
- Focus on helping the user ACHIEVE their specific intent, not just providing information
- If they want to "become" something, provide clear pathways and next steps
- If they want to "learn" something, structure the response as a learning journey
- If they want to "implement" something, provide actionable steps and resources
- Use the document content as context for their current understanding
- Leverage web results to fill knowledge gaps and provide actionable next steps
- Prioritize resources that match their intent (tutorials for learning, career guides for becoming, etc.)

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

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": synthesis_prompt},
                {"role": "user", "content": user_message}
            ],
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