"""
Prompt templates for Course Generation.
Well-structured prompts following PRD specifications.
"""

from typing import Dict, Any, Optional, List


def build_outline_generation_prompt(
    topic: str,
    expertise: str,
    time_available: int,
    format_pref: str,
    depth_pref: str,
    role: str,
    learning_goal: str,
    example_pref: str,
    file_context: Optional[str] = None,
    organization_instructions: Optional[str] = None
) -> str:
    """Build prompt for course outline generation"""
    
    file_section = f"""
SOURCE MATERIAL CONTEXT:
{file_context}
""" if file_context else ""

    org_section = f"""
MULTI-FILE ORGANIZATION:
{organization_instructions}
""" if organization_instructions else ""

    return f"""You are an expert curriculum designer and educator. Create a structured, pedagogically sound course outline.

USER CONTEXT:
- Topic: {topic}
- Expertise Level: {expertise} (beginner/intermediate/advanced)
- Time Available: {time_available} minutes
- Learning Format Preference: {format_pref}
- Explanation Depth: {depth_pref}
- User Role: {role}
- Learning Goal: {learning_goal}
- Example Preference: {example_pref}
{file_section}
{org_section}
REQUIREMENTS:
1. Generate 3-5 chapters that progressively build knowledge
2. Structure: Foundation → Core Concepts → Applications → Synthesis
3. Each chapter must have:
   - Clear, descriptive title
   - 2-4 specific learning objectives (measurable, action-oriented)
   - 2-3 key concepts covered
   - Estimated time (minutes)
   - Prerequisites (if any)
4. Total time should approximate {time_available} minutes
5. Chapter progression must be logical (prerequisites first)
6. Make titles engaging but descriptive

OUTPUT FORMAT (JSON - no markdown formatting):
{{
  "title": "Engaging course title (5-8 words)",
  "description": "2-3 sentence overview of what student will learn and why it matters",
  "learning_objectives": [
    "By the end of this course, students will be able to...",
    "Specific measurable outcome 2",
    "Specific measurable outcome 3"
  ],
  "chapters": [
    {{
      "order": 1,
      "title": "Chapter Title (descriptive, engaging)",
      "objectives": [
        "After this chapter, students can...",
        "Specific measurable objective"
      ],
      "key_concepts": ["Concept 1", "Concept 2", "Concept 3"],
      "estimated_time": 15,
      "prerequisites": []
    }}
  ],
  "total_estimated_time": {time_available}
}}

Important: Return ONLY the JSON object, no markdown code blocks or additional text."""


def build_chapter_content_prompt(
    course_title: str,
    chapter_num: int,
    total_chapters: int,
    chapter_title: str,
    objectives: list,
    expertise: str,
    format_pref: str,
    depth_pref: str,
    role: str,
    learning_goal: str,
    example_pref: str,
    prev_chapter_title: Optional[str] = None,
    next_chapter_title: Optional[str] = None,
    source_material_context: Optional[str] = None
) -> str:
    """Build prompt for chapter content generation"""
    
    prev_section = f"""
- Previous Chapter: {prev_chapter_title}""" if prev_chapter_title else ""
    
    next_section = f"""
- Next Chapter: {next_chapter_title}""" if next_chapter_title else ""
    
    source_section = f"""
SOURCE MATERIAL (from uploaded document - YOU MUST teach from this):
{source_material_context}

CRITICAL INSTRUCTIONS FOR SOURCE MATERIAL:
- Base your chapter content PRIMARILY on the source material above
- Do NOT invent facts or examples that aren't supported by the source material
- Use the source material's terminology and structure
- If the source material is thin on a subtopic, note it briefly and move on
- Inline citations [1], [2] should reference the [Source Section N] labels above
""" if source_material_context else ""

    objectives_formatted = "\n".join([f"- {obj}" for obj in objectives])
    
    depth_instructions = {
        "quick": "Be concise, use bullet points, get to the point quickly. Focus on essentials only.",
        "detailed": "Provide comprehensive, thorough explanations. Include nuances and edge cases.",
        "conversational": "Write in a friendly, engaging, accessible tone. Use 'you' and 'we'. Make it feel like a conversation.",
        "academic": "Use formal, rigorous, precise language. Include technical terminology. Maintain scholarly tone."
    }.get(depth_pref, "Provide clear, comprehensive explanations.")

    return f"""You are an expert educator creating high-quality educational content.

COURSE CONTEXT:
- Course: {course_title}
- Chapter {chapter_num} of {total_chapters}: {chapter_title}{prev_section}{next_section}

USER PERSONALIZATION:
- Expertise Level: {expertise}
- Learning Format: {format_pref}
- Depth Preference: {depth_pref}
- Role: {role}
- Learning Goal: {learning_goal}
- Example Preference: {example_pref}

LEARNING OBJECTIVES FOR THIS CHAPTER:
{objectives_formatted}
{source_section}
CONTENT REQUIREMENTS:
1. Length: 800-1200 words (aim for depth over brevity)
2. Format: Markdown with clear hierarchy (# for title, ## for sections, ### for subsections)
3. Structure:
   - **Hook/Why This Matters** (2-3 sentences explaining relevance)
   - **Core Content** with {example_pref} examples integrated throughout
   - **Practical Applications** or real-world implications
   - **Key Takeaways** section with 4-6 bullet points summarizing main points
4. Tone: {depth_instructions}
5. Include inline citations [1], [2], [3] for:
   - Factual claims and statistics
   - Research findings
   - Historical events or dates
   - Technical specifications

QUIZ REQUIREMENTS:
Generate 3-5 questions that test UNDERSTANDING, not memorization:
- Mix of multiple choice (4 options) and true/false
- Questions should require applying knowledge, not just recalling facts
- Include explanations for correct AND incorrect answers
- Make distractors (wrong answers) plausible

SEARCH REQUIREMENTS:
Use web search to find 3-5 authoritative sources on key claims. Prioritize:
- Academic papers (.edu sources)
- Official documentation
- Reputable news sources
- Educational institutions

OUTPUT FORMAT (JSON - no markdown):
{{
  "content": "Full markdown content with inline citations [1], [2]...",
  "word_count": 1050,
  "key_concepts_explained": ["Concept 1", "Concept 2"],
  "quiz": {{
    "questions": [
      {{
        "id": "q1",
        "type": "multiple_choice",
        "question": "Clear question text that tests understanding?",
        "options": ["Option A", "Option B", "Option C", "Option D"],
        "correct_answer": 1,
        "explanation": "Detailed explanation: Why B is correct and why A, C, and D are incorrect"
      }},
      {{
        "id": "q2", 
        "type": "true_false",
        "question": "Statement to evaluate?",
        "options": ["True", "False"],
        "correct_answer": 0,
        "explanation": "Explanation of why this is true/false"
      }}
    ]
  }},
  "sources": [
    {{
      "number": 1,
      "title": "Source Title",
      "url": "https://...",
      "date": "2025-01-15",
      "source_type": "academic|news|documentation|book",
      "relevance": "Brief note on what this source verifies in the content"
    }}
  ]
}}

Important: Return ONLY the JSON object. Ensure all citations [1], [2], etc. in the content match sources in the sources array."""


def build_topic_extraction_prompt(text: str) -> str:
    """Extract main topic from text"""
    return f"""Analyze this text and identify the main topic in 2-4 words.

TEXT (first 2000 characters):
{text[:2000]}

Return ONLY the topic name, nothing else. Examples:
- "Quantum Mechanics"
- "Machine Learning"
- "Organic Chemistry"
- "World War II"
- "Cellular Biology"

Topic:"""


def build_multi_file_analysis_prompt(topics: list) -> str:
    """Analyze relationship between multiple file topics"""
    topics_text = "\n".join([f"- {t['filename']}: {t['topic']}" for t in topics])
    
    return f"""Analyze these topics from uploaded files and recommend organization strategy:

FILES AND TOPICS:
{topics_text}

ORGANIZATION OPTIONS:
1. **thematic_bridge**: Topics are closely related and should be taught together showing connections
2. **sequential_sections**: Topics are distinct but can be in one course as separate sections
3. **separate_courses**: Topics are unrelated and should be separate courses

ANALYSIS INSTRUCTIONS:
- If topics are the same field (e.g., both physics topics): thematic_bridge
- If topics are complementary (e.g., math + physics): thematic_bridge or sequential_sections
- If topics are unrelated (e.g., history + biology): separate_courses
- Consider: Do these topics build on each other? Do they share concepts? Are they from same course?

Return a brief analysis (2-3 sentences) recommending the best organization and why."""


def build_document_map_prompt(chunk_summaries: List[Dict[str, str]]) -> str:
    """Build prompt to create a structured document map from chunk summaries"""
    chunks_text = "\n".join([
        f"[Chunk {c['index']}] {c['summary']}"
        for c in chunk_summaries
    ])

    return f"""Analyze these document chunks and create a structured map of all topics and sections covered.

DOCUMENT CHUNKS:
{chunks_text}

Create a structured JSON map that:
1. Identifies ALL distinct topics/sections in the document
2. Maps each topic to the chunk indices that contain relevant content
3. Orders topics logically (as they should be taught)
4. Ensures EVERY chunk index appears in at least one topic

OUTPUT FORMAT (JSON only):
{{
  "document_title": "Inferred title of the document",
  "total_chunks": {len(chunk_summaries)},
  "topics": [
    {{
      "topic": "Topic or section name",
      "description": "One sentence description of what this section covers",
      "chunk_indices": [0, 1, 2],
      "importance": "core|supporting|supplementary"
    }}
  ],
  "coverage_check": "all_chunks_mapped"
}}

Important: Return ONLY the JSON object. Every chunk index from 0 to {len(chunk_summaries) - 1} MUST appear in at least one topic."""


# Error recovery prompts
RETRY_PROMPT_ADDENDUM = """

Note: A previous generation attempt had issues. Please:
1. Ensure all content is factually accurate
2. Double-check that citations match the content
3. Make sure quiz questions are clear and unambiguous
4. Verify that explanations are thorough and educational
"""
