"""
Prompts for space-level querying across multiple courses.
"""

from typing import Dict, Any, List


# ─── Personalization maps (shared with course prompts) ─────────────────────
_EXPERTISE_MAP = {
    "beginner": "The learner is a beginner — use plain language, define technical terms, and lean on analogies.",
    "intermediate": "The learner has foundational knowledge — build on it, clarify nuances, skip basic definitions.",
    "advanced": "The learner is advanced — skip basics entirely, focus on trade-offs, edge cases, and depth.",
}

_ROLE_MAP = {
    "student": "They are a student — reinforce concepts with structure and encourage deeper understanding.",
    "professional": "They are a working professional — be direct, actionable, and skip academic fluff.",
    "graduate_student": "They are a graduate student — engage rigorously, reference theory, connect to research.",
}

_GOAL_MAP = {
    "exams": "They're studying for exams — emphasize testable facts, definitions, and precision.",
    "curiosity": "They're learning out of curiosity — connect ideas to big-picture themes and make it engaging.",
    "career": "They're building career skills — tie concepts to real-world use and industry practice.",
    "supplement": "They're supplementing existing coursework — fill gaps and reinforce key points concisely.",
}

_DEPTH_MAP = {
    "quick": "Keep answers tight and direct. One strong example per concept max.",
    "detailed": "Explain the 'why' behind the 'what'. Multiple examples welcome when the topic warrants it.",
    "conversational": "Be conversational — use 'you' and 'we', pose rhetorical questions, make it a dialogue.",
    "academic": "Maintain academic rigor — precise language, structured reasoning, formal tone.",
}

_EXAMPLE_MAP = {
    "real_world": "Favor real-world, hands-on examples that show concepts in practice.",
    "technical": "Use technical examples, code snippets, or formal demonstrations.",
    "stories": "Use mini-stories and narrative examples to make concepts memorable.",
    "analogies": "Use extended analogies that map abstract concepts to familiar, everyday systems.",
}


def _personalization_block(profile: Dict[str, Any]) -> str:
    expertise = profile.get("expertise", "intermediate")
    role = profile.get("role", "student")
    goal = profile.get("learning_goal", "curiosity")
    depth = profile.get("depth_pref", "detailed")
    example_pref = profile.get("example_pref", "real_world")
    return "\n".join([
        f"- {_EXPERTISE_MAP.get(expertise, _EXPERTISE_MAP['intermediate'])}",
        f"- {_ROLE_MAP.get(role, _ROLE_MAP['student'])}",
        f"- {_GOAL_MAP.get(goal, _GOAL_MAP['curiosity'])}",
        f"- {_DEPTH_MAP.get(depth, _DEPTH_MAP['detailed'])}",
        f"- {_EXAMPLE_MAP.get(example_pref, _EXAMPLE_MAP['real_world'])}",
    ])


def _history_block(chat_history: List[Dict[str, str]]) -> str:
    if not chat_history:
        return ""
    lines = []
    for msg in chat_history:
        label = "Learner" if msg["role"] == "user" else "Assistant"
        lines.append(f"{label}: {msg['content']}")
    return "CONVERSATION HISTORY:\n" + "\n\n".join(lines) + "\n\n"


def build_space_query_prompt(
    query: str,
    course_contexts: List[Dict[str, Any]],
    profile: Dict[str, Any],
    chat_history: List[Dict[str, str]],
) -> str:
    """
    Build a personalized, citation-aware prompt for a space-level query
    that spans multiple courses.

    Args:
        query: The learner's question.
        course_contexts: List of dicts, each with:
            - course_id: str
            - course_title: str
            - chunks: List[Dict] — each chunk has 'text', 'source_file', 'chapter_title' (optional)
        profile: Learning profile dict with expertise, role, learning_goal, depth_pref, example_pref.
        chat_history: Prior messages [{role, content}].

    Returns:
        Prompt string for the LLM.
    """
    # Build the multi-course source material block with clear citations
    source_sections = []
    citation_index = 1
    for ctx in course_contexts:
        course_title = ctx["course_title"]
        chunks = ctx["chunks"]
        course_chunks_text = []
        for chunk in chunks:
            text = chunk.get("text", "").strip()
            chapter = chunk.get("chapter_title", "")
            source_file = chunk.get("source_file", "")
            location = chapter or source_file
            location_label = f" ({location})" if location else ""
            course_chunks_text.append(
                f"[Source {citation_index}]{location_label}\n{text}"
            )
            citation_index += 1
        if course_chunks_text:
            section = f"── {course_title} ──\n" + "\n\n".join(course_chunks_text)
            source_sections.append(section)

    sources_block = "\n\n".join(source_sections)
    personalization = _personalization_block(profile)
    history = _history_block(chat_history)

    course_list = "\n".join(f"  • {ctx['course_title']}" for ctx in course_contexts)

    return f"""You are a knowledgeable learning assistant helping a student query across their study space, which contains the following courses:
{course_list}

LEARNER PROFILE:
{personalization}

{history}SOURCE MATERIAL (retrieved from the courses above):
{sources_block}

QUESTION: {query}

RESPONSE GUIDELINES:
- Answer directly from the source material above. Cite each source inline as [Source N] whenever you reference it.
- If the answer spans multiple courses, clearly indicate which course each piece of information comes from.
- Be CONCISE by default. For simple factual questions, 1-3 sentences suffice. For "explain" or "break down" requests, go deeper with structure.
- Use bullet points or numbered lists for multi-part answers.
- Adapt your tone, depth, and examples to the learner profile above.
- If the source material does not fully answer the question, say so clearly at the end (e.g., "The courses don't cover X in detail."). Do NOT fabricate.
- Reference conversation history naturally when it's relevant.
- Do NOT add filler phrases like "Great question!" or restate the question before answering."""


def build_space_web_fallback_prompt(
    query: str,
    profile: Dict[str, Any],
    chat_history: List[Dict[str, str]],
    space_name: str = "your study space",
    has_web_search: bool = True,
) -> str:
    """
    Build a prompt for the fallback path when no course material is relevant.

    Args:
        query: The learner's question.
        profile: Learning profile dict.
        chat_history: Prior messages.
        space_name: Human-readable name of the space (optional).
        has_web_search: True if the model supports live web search (Claude with tool).
                        False means the model will answer from general knowledge only.

    Returns:
        Prompt string for the LLM.
    """
    personalization = _personalization_block(profile)
    history = _history_block(chat_history)

    if has_web_search:
        source_description = "external web sources"
        job_instructions = (
            "1. Search the web for accurate, relevant information about this question.\n"
            "2. Provide a clear, helpful answer based on what you find.\n"
            "3. Cite your sources naturally (e.g., \"According to [source]...\").\n"
            "4. Do not fabricate. If web search doesn't yield clear results, say so honestly."
        )
    else:
        source_description = "general knowledge (no live web search available for this model)"
        job_instructions = (
            "1. Answer from your general knowledge as accurately as possible.\n"
            "2. Be clear about what you know vs. what you're less certain about.\n"
            "3. Do not fabricate facts. If you're unsure, say so."
        )

    return f"""You are a knowledgeable learning assistant. The learner asked a question that was NOT found in any of their course materials in {space_name}.

⚠️ This answer is not from your course materials — it's based on {source_description}.

Your job:
{job_instructions}

LEARNER PROFILE:
{personalization}

{history}QUESTION: {query}

RESPONSE GUIDELINES:
- Start your response with the disclaimer line above (already provided), then answer directly.
- Adapt depth, tone, and examples to the learner profile above.
- Be concise by default; go deeper only if the question warrants it."""
