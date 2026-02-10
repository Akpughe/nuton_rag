"""
Prompt templates for Course Generation.
Well-structured prompts following PRD specifications.
"""

from typing import Dict, Any, Optional, List


def _build_personalization_strategy(
    expertise: str,
    role: str,
    learning_goal: str,
    depth_pref: str,
    example_pref: str,
    format_pref: str
) -> str:
    """Build detailed personalization instructions from user preferences."""

    # Role + Goal combined strategy
    role_goal_strategies = {
        ("student", "exams"): {
            "voice": "supportive academic coach preparing you for success",
            "focus": "Emphasize testable concepts, common exam patterns, and memory anchors. Flag 'this is frequently tested' where relevant. End sections with quick self-check questions.",
            "structure": "Build from definitions -> worked examples -> practice scenarios -> edge cases examiners love"
        },
        ("student", "curiosity"): {
            "voice": "enthusiastic guide revealing fascinating ideas",
            "focus": "Lead with 'why this is amazing' hooks. Connect to broader questions. Include surprising facts and unsolved mysteries.",
            "structure": "Start with an intriguing question -> explore the answer -> reveal deeper layers -> connect to big picture"
        },
        ("student", "supplement"): {
            "voice": "patient tutor filling in gaps from class",
            "focus": "Assume student has partial understanding from lectures. Fill gaps, clarify confusions, reinforce key points. Reference 'as you may have covered in class' framing.",
            "structure": "Recap core idea briefly -> identify common confusion points -> clear them up -> reinforce with examples"
        },
        ("student", "career"): {
            "voice": "mentor bridging academic knowledge to workplace skills",
            "focus": "Connect theory to industry applications. Mention relevant tools, frameworks, and real companies that use these concepts.",
            "structure": "Theory essentials -> how industry applies this -> skills employers value -> portfolio/project ideas"
        },
        ("professional", "career"): {
            "voice": "experienced colleague sharing practical expertise",
            "focus": "Skip basics the reader likely knows. Focus on actionable patterns, common pitfalls in production, and decision frameworks. Use industry terminology naturally.",
            "structure": "Context/problem statement -> proven approaches -> trade-offs -> implementation checklist"
        },
        ("professional", "curiosity"): {
            "voice": "knowledgeable peer exploring an interesting domain",
            "focus": "Respect existing expertise while introducing new domain. Draw parallels to professional experience. Focus on transferable concepts.",
            "structure": "Why this matters for your work -> key concepts with professional analogies -> deeper exploration -> practical connections"
        },
        ("graduate_student", "exams"): {
            "voice": "rigorous academic preparing you for advanced assessment",
            "focus": "Emphasize theoretical foundations, proof strategies, edge cases, and connections to research literature. Include the 'why' behind formulas and methods.",
            "structure": "Formal foundations -> derivations/proofs -> applications -> common pitfalls in advanced problems"
        },
        ("graduate_student", "curiosity"): {
            "voice": "research colleague exploring cutting-edge ideas",
            "focus": "Engage with the intellectual frontier. Reference recent papers and open problems. Encourage critical thinking about assumptions and limitations.",
            "structure": "Current state of knowledge -> key debates -> methodological considerations -> open questions"
        },
        ("professional", "exams"): {
            "voice": "focused certification prep coach who respects your experience",
            "focus": "Target exam-specific concepts and common test pitfalls. Skip unnecessary theory — focus on what's tested. Use professional context to anchor abstract exam topics.",
            "structure": "Exam topic overview -> key concepts with professional context -> common exam traps -> practice application"
        },
        ("professional", "supplement"): {
            "voice": "pragmatic advisor filling knowledge gaps for work",
            "focus": "Identify what the professional needs to know right now. Skip academic depth — focus on practical understanding sufficient for the job. Be direct and efficient.",
            "structure": "What you need to know -> core concepts explained practically -> how to apply this at work -> quick reference summary"
        },
        ("graduate_student", "career"): {
            "voice": "mentor bridging academia and industry",
            "focus": "Translate academic concepts into industry-valued skills. Show how research methods apply to real-world problems. Highlight what employers look for beyond publications.",
            "structure": "Academic foundation -> industry application -> transferable skills -> portfolio and interview relevance"
        },
        ("graduate_student", "supplement"): {
            "voice": "advanced tutor reinforcing graduate coursework",
            "focus": "Fill gaps from lectures and readings at graduate level. Provide rigorous explanations with proofs or derivations where needed. Connect to the broader research landscape.",
            "structure": "Recap core theorem/concept -> detailed walkthrough -> edge cases and extensions -> connections to related coursework"
        },
    }

    # Get the best match, or fall back to a generic
    strategy = role_goal_strategies.get(
        (role, learning_goal),
        {
            "voice": "clear, knowledgeable educator",
            "focus": "Balance theory and practice. Provide clear explanations with relevant examples.",
            "structure": "Foundation concepts -> detailed exploration -> practical applications -> synthesis"
        }
    )

    # Expertise level modifiers
    expertise_instructions = {
        "beginner": (
            "EXPERTISE ADAPTATION (Beginner):\n"
            "- Define every technical term when first introduced (bold it, then explain in plain language)\n"
            "- Use analogies to everyday objects/experiences before introducing formal concepts\n"
            "- Limit to ONE new concept per section before reinforcing it\n"
            "- Provide 'check your understanding' moments: 'If you can explain X in your own words, you're ready for the next section'\n"
            "- Never say 'simply' or 'obviously' — nothing is obvious to a beginner"
        ),
        "intermediate": (
            "EXPERTISE ADAPTATION (Intermediate):\n"
            "- Assume foundational vocabulary is known, but still clarify nuanced or ambiguous terms\n"
            "- Build on 'you already know X — here's how it connects to Y'\n"
            "- Include 'common misconception' callouts for things intermediate learners often get wrong\n"
            "- Provide both the intuitive explanation AND the precise definition\n"
            "- Challenge with 'what would happen if...' thought experiments"
        ),
        "advanced": (
            "EXPERTISE ADAPTATION (Advanced):\n"
            "- Skip introductory definitions; reference them briefly only when needed for precision\n"
            "- Focus on nuances, edge cases, trade-offs, and connections between concepts\n"
            "- Include comparisons to alternative approaches or competing theories\n"
            "- Reference primary sources and seminal papers where relevant\n"
            "- Engage with limitations and open questions in the field"
        )
    }

    # Depth preference modifiers
    depth_instructions = {
        "quick": (
            "DEPTH STYLE (Quick & Focused):\n"
            "- Lead with the key insight in the first sentence of each section\n"
            "- Use bullet points for supporting details\n"
            "- One example per concept (the best one, not three okay ones)\n"
            "- Cut all tangents — if it doesn't directly serve the learning objective, omit it\n"
            "- Target: the reader should 'get it' in the minimum number of words"
        ),
        "detailed": (
            "DEPTH STYLE (Thorough & Comprehensive):\n"
            "- Explain the 'why' behind every 'what' — don't just state facts, explain mechanisms\n"
            "- Include multiple examples showing the same concept in different contexts\n"
            "- Address edge cases and exceptions explicitly\n"
            "- Show the reasoning process, not just the conclusion\n"
            "- Connect each concept to both prerequisites and downstream applications"
        ),
        "conversational": (
            "DEPTH STYLE (Conversational & Engaging):\n"
            "- Write as if explaining to a smart friend over coffee\n"
            "- Use 'you' and 'we' and 'let's think about this'\n"
            "- Include rhetorical questions that guide thinking: 'But wait — what happens when...?'\n"
            "- Share the narrative of discovery: how was this figured out? What problem was someone trying to solve?\n"
            "- Make it feel like a dialogue, not a lecture"
        ),
        "academic": (
            "DEPTH STYLE (Rigorous & Scholarly):\n"
            "- Use precise academic terminology with proper definitions\n"
            "- Structure arguments formally: premise, evidence, conclusion\n"
            "- Reference established frameworks and taxonomies\n"
            "- Distinguish between established consensus, emerging evidence, and speculation\n"
            "- Maintain scholarly objectivity — present competing views where they exist"
        )
    }

    # Example preference modifiers
    example_instructions = {
        "real_world": "EXAMPLES: Use real companies, events, technologies, and case studies. Name specific products, dates, and outcomes. The reader should think 'I've seen this in my own life.'",
        "technical": "EXAMPLES: Use code snippets, formulas, diagrams described in text, and technical specifications. Walk through computations step-by-step. Show inputs and outputs.",
        "stories": "EXAMPLES: Frame examples as mini-narratives with characters, problems, and resolutions. 'Imagine a researcher named Dr. Patel who discovered...' Make concepts memorable through story.",
        "analogies": "EXAMPLES: Build extended analogies that map abstract concepts to concrete, familiar systems. 'Think of X like a postal system where...' Carry the analogy through multiple aspects of the concept."
    }

    # Format preference modifiers
    format_instructions = {
        "reading": "FORMAT: Optimize for reading — use clear prose, markdown headers, and visual hierarchy. Vary paragraph length to maintain rhythm.",
        "listening": "FORMAT: Optimize for spoken delivery — use shorter sentences, conversational flow, and avoid visual-only elements like complex tables or code blocks.",
        "testing": "FORMAT: Optimize for active practice — include frequent self-check questions, fill-in-the-blank prompts, and hands-on exercises woven throughout the explanation.",
        "mixed": "FORMAT: Mix reading, practice questions, and conversational explanation. Vary the format section by section to maintain engagement."
    }

    sections = [
        f"TEACHING VOICE: Write as a {strategy['voice']}.",
        f"CONTENT FOCUS: {strategy['focus']}",
        f"STRUCTURAL APPROACH: {strategy['structure']}",
        expertise_instructions.get(expertise, expertise_instructions["intermediate"]),
        depth_instructions.get(depth_pref, depth_instructions["detailed"]),
        example_instructions.get(example_pref, example_instructions["real_world"]),
        format_instructions.get(format_pref, format_instructions["reading"])
    ]

    return "\n\n".join(sections)


def _build_bloom_taxonomy_instruction(expertise: str) -> str:
    """Return Bloom's taxonomy guidance matched to expertise level."""
    if expertise == "beginner":
        return (
            "LEARNING OBJECTIVES (use Bloom's Taxonomy — lower levels for beginners):\n"
            "- Use verbs: Define, Identify, Describe, Explain, Summarize\n"
            "- Each objective should be testable: 'After this chapter, the student can [verb] [specific thing]'\n"
            "- Avoid: Analyze, Evaluate, Create (too advanced for beginners)"
        )
    elif expertise == "advanced":
        return (
            "LEARNING OBJECTIVES (use Bloom's Taxonomy — higher levels for advanced):\n"
            "- Use verbs: Analyze, Evaluate, Compare, Design, Critique, Synthesize\n"
            "- Each objective should require applying knowledge to new situations\n"
            "- Avoid: Define, List, Identify (too basic for advanced learners)"
        )
    else:  # intermediate
        return (
            "LEARNING OBJECTIVES (use Bloom's Taxonomy — mid levels for intermediate):\n"
            "- Use verbs: Apply, Compare, Explain why, Predict, Classify, Distinguish\n"
            "- Each objective should go beyond recall into application and analysis\n"
            "- Mix some higher-level objectives in: one Analyze or Evaluate per chapter"
        )


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
    organization_instructions: Optional[str] = None,
    suggested_chapter_count: Optional[int] = None,
    structured_topic_constraint: Optional[str] = None
) -> str:
    """Build prompt for course outline generation"""

    # Use structured constraint instead of flat file_context when available
    if structured_topic_constraint:
        file_section = f"""
DOCUMENT STRUCTURE CONSTRAINT:
{structured_topic_constraint}
"""
    elif file_context:
        file_section = f"""
SOURCE MATERIAL CONTEXT:
{file_context}
"""
    else:
        file_section = ""

    org_section = f"""
MULTI-FILE ORGANIZATION:
{organization_instructions}
""" if organization_instructions else ""

    # Dynamic chapter count instruction
    if suggested_chapter_count:
        chapter_instruction = f"Generate EXACTLY {suggested_chapter_count} chapters that progressively build knowledge"
    else:
        chapter_instruction = "Generate 3-7 chapters that progressively build knowledge"

    bloom_instruction = _build_bloom_taxonomy_instruction(expertise)

    # Goal-aware framing
    goal_framing = {
        "exams": f"This student is preparing for exams on {topic}. The outline should prioritize testable concepts, build from foundational to complex, and ensure every chapter covers material likely to be assessed.",
        "career": f"This learner wants to apply {topic} professionally. The outline should balance essential theory with practical application, building toward real-world competence.",
        "curiosity": f"This learner is exploring {topic} out of genuine interest. The outline should tell a compelling intellectual story — each chapter should answer a question that naturally leads to the next.",
        "supplement": f"This student is supplementing classroom learning on {topic}. The outline should mirror a typical course structure but focus on the concepts students most commonly struggle with."
    }.get(learning_goal, f"Create a well-structured course on {topic} that builds understanding progressively.")

    return f"""You are a world-class curriculum designer who creates courses that students describe as "the best explanation I've ever found." Your courses are not generic — they are thoughtfully structured for the specific learner.

LEARNER PROFILE:
- Topic: {topic}
- Expertise Level: {expertise}
- Role: {role}
- Learning Goal: {learning_goal}
- Time Budget: {time_available} minutes
- Depth Preference: {depth_pref}
- Example Style: {example_pref}

COURSE FRAMING:
{goal_framing}
{file_section}
{org_section}
CURRICULUM DESIGN REQUIREMENTS:

1. {chapter_instruction}

2. CHAPTER SEQUENCING — Choose the structure that fits the topic's nature:
   - For CONCEPTUAL topics (physics, philosophy, math): Build prerequisite chains. Each chapter's core concept must be necessary for understanding the next.
   - For PROCEDURAL topics (programming, cooking, lab techniques): Follow the workflow. Chapters should mirror the order someone would actually do things.
   - For HISTORICAL/NARRATIVE topics (history, biography, evolution of ideas): Follow chronological or thematic arcs that tell a story.
   - For SURVEY topics (introduction to a broad field): Group by sub-domain, but start with the unifying principles that connect everything.

3. CHAPTER DESIGN:
   - Title: Must hint at the "aha moment" — what the student will understand after this chapter (not just the topic label). Bad: "Neural Networks." Good: "How Machines Learn to See Patterns."
   - {bloom_instruction}
   - Key Concepts: 2-4 concepts per chapter. These should be specific and concrete (not vague like "understanding basics").
   - Prerequisites: Be explicit. If Chapter 3 requires concepts from Chapter 1, say so.
   - Time estimates must sum to approximately {time_available} minutes.

4. COURSE-LEVEL LEARNING OBJECTIVES:
   - Write 3-4 objectives that describe what the student CAN DO after completing the full course.
   - These should be specific and measurable, not vague ("understand the basics" is too weak).
   - At least one objective should involve applying knowledge to a new situation.

5. COURSE DESCRIPTION:
   - 2-3 sentences. First sentence: what the student will learn. Second sentence: why it matters to THEM specifically (given their role and goal). Third sentence (optional): what makes this course different from a textbook.

OUTPUT FORMAT (JSON - no markdown formatting):
{{
  "title": "Engaging course title that promises a transformation (5-10 words)",
  "description": "2-3 sentences: what you'll learn, why it matters to you, what's unique about this course",
  "learning_objectives": [
    "After this course, you will be able to [specific measurable outcome]",
    "You will be able to [apply/analyze/create something specific]",
    "You will understand [specific concept] well enough to [concrete action]"
  ],
  "chapters": [
    {{
      "order": 1,
      "title": "Chapter Title That Promises an Insight",
      "objectives": [
        "After this chapter, you can [specific verb] [specific thing]",
        "You will be able to [another specific outcome]"
      ],
      "key_concepts": ["Specific Concept 1", "Specific Concept 2", "Specific Concept 3"],
      "estimated_time": 15,
      "prerequisites": []
    }}
  ],
  "total_estimated_time": {time_available}
}}

Important: Return ONLY the JSON object, no markdown code blocks or additional text."""


def _build_search_section(search_mode: str, web_sources: Optional[List[Dict]] = None) -> str:
    """Build the search/citation section of the chapter prompt based on model capabilities."""
    if search_mode == "native":
        return """SEARCH REQUIREMENTS:
Use web search to find 3-5 authoritative sources on key claims. Prioritize:
- Academic papers (.edu sources)
- Official documentation
- Reputable news sources
- Educational institutions"""

    if search_mode == "provided" and web_sources:
        sources_text = ""
        for i, src in enumerate(web_sources, 1):
            sources_text += f"[{i}] {src.get('title', 'Source')}\n"
            sources_text += f"    URL: {src.get('url', 'N/A')}\n"
            if src.get('excerpt'):
                sources_text += f"    Excerpt: {src['excerpt']}\n"
            sources_text += "\n"
        return f"""WEB RESEARCH SOURCES (pre-fetched):
{sources_text}
Use these sources for inline citations [1], [2], etc. Reference them accurately.
If additional facts are needed beyond these sources, cite from your training knowledge and note the source."""

    # search_mode == "none"
    return """CITATION GUIDANCE:
No web search available. Cite from training knowledge. Note real sources you know exist
(textbooks, papers, official docs) rather than fabricating URLs. Use format:
[1] Author, "Title", Publication (Year) — or similar known references."""


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
    source_material_context: Optional[str] = None,
    search_mode: str = "native",
    web_sources: Optional[List[Dict]] = None
) -> str:
    """Build prompt for chapter content generation"""

    prev_section = f"\n- Previous Chapter: \"{prev_chapter_title}\"" if prev_chapter_title else ""
    next_section = f"\n- Next Chapter: \"{next_chapter_title}\"" if next_chapter_title else ""

    source_section = f"""
SOURCE MATERIAL (from uploaded document — base your teaching on this):
{source_material_context}

SOURCE MATERIAL RULES:
- Base your chapter content PRIMARILY on the source material above
- Do NOT invent facts not supported by the source material
- Use the source material's terminology and structure
- If the source material is thin on a subtopic, note it briefly and move on
- Inline citations [1], [2] should reference the [Source Section N] labels above
""" if source_material_context else ""

    objectives_formatted = "\n".join([f"  {i+1}. {obj}" for i, obj in enumerate(objectives)])

    personalization = _build_personalization_strategy(
        expertise=expertise,
        role=role,
        learning_goal=learning_goal,
        depth_pref=depth_pref,
        example_pref=example_pref,
        format_pref=format_pref
    )

    # Chapter position awareness
    if total_chapters == 1:
        position_guidance = (
            "CHAPTER POSITION: This is a STANDALONE single-chapter course.\n"
            "- Open with a compelling hook that makes the reader care about the topic\n"
            "- Cover the essential concepts thoroughly since this is the only chapter\n"
            "- End with synthesis, practical next steps, and resources for deeper learning"
        )
    elif chapter_num == 1:
        position_guidance = (
            "CHAPTER POSITION: This is the OPENING chapter.\n"
            "- Start with a compelling hook that makes the reader care about the entire course topic\n"
            "- Establish the 'big question' or 'big problem' that the course will answer\n"
            "- Build foundational vocabulary and mental models needed for later chapters\n"
            "- End by creating anticipation for what comes next"
        )
    elif chapter_num == total_chapters:
        position_guidance = (
            "CHAPTER POSITION: This is the FINAL chapter.\n"
            "- Synthesize ideas from across the entire course into a cohesive understanding\n"
            "- Show how individual concepts connect into a bigger picture\n"
            "- End with: what to learn next, open questions, or how to apply this knowledge\n"
            "- Give the reader a sense of accomplishment and direction"
        )
    else:
        prev_ref = f' ("{prev_chapter_title}")' if prev_chapter_title else ""
        next_ref = f' ("{next_chapter_title}")' if next_chapter_title else ""
        position_guidance = (
            f"CHAPTER POSITION: Chapter {chapter_num} of {total_chapters} (middle of course).\n"
            f"- Bridge from previous chapter{prev_ref} — open with a 1-2 sentence connection\n"
            "- Build on established concepts while introducing new ones\n"
            f"- Set up what comes next{next_ref} with a forward-looking closing"
        )

    return f"""You are creating a chapter that a student will describe as "this finally made it click." You don't just explain topics — you teach. You anticipate confusion, build understanding step by step, and make complex ideas feel approachable.

COURSE CONTEXT:
- Course: "{course_title}"
- Chapter {chapter_num} of {total_chapters}: "{chapter_title}"{prev_section}{next_section}

{position_guidance}

LEARNING OBJECTIVES FOR THIS CHAPTER:
{objectives_formatted}
Every section of your chapter must directly serve at least one of these objectives. If a section doesn't serve an objective, cut it.
{source_section}
PERSONALIZATION INSTRUCTIONS:
{personalization}

CHAPTER STRUCTURE (follow this arc):

1. **HOOK** (2-3 sentences)
   - Open with a concrete scenario, question, or surprising fact that makes the reader think "I need to know this"
   - Connect to the reader's world based on their role ({role}) and goal ({learning_goal})

2. **CORE TEACHING** (the main body — this is where learning happens)
   - Introduce ONE concept at a time. Fully explain it before moving to the next.
   - For each concept:
     a) State what it is in one clear sentence
     b) Explain WHY it works that way or WHY it matters
     c) Give a concrete example that makes it tangible
     d) Address the most common misconception or confusion point ("A common mistake is thinking X, but actually...")
   - Use transitions that show how concepts connect: "Now that you understand X, you can see why Y works the way it does"

3. **PRACTICAL APPLICATION** (show it in action)
   - One extended example, case study, or worked problem that uses MULTIPLE concepts from this chapter together
   - Walk through it step by step — don't just show the answer, show the thinking process

4. **KEY TAKEAWAYS** (4-6 bullets)
   - Each bullet should be a complete, standalone insight (not "Chapter covered X")
   - Format: "[Concept]: [What to remember about it]"
   - A student should be able to read ONLY the takeaways and recall the full chapter

CONTENT QUALITY RULES:
- Length: 1000-1500 words. Depth over breadth — better to explain 3 concepts well than 6 concepts poorly.
- Format: Markdown with clear hierarchy (# for title, ## for sections, ### for subsections)
- Include inline citations [1], [2], [3] for factual claims, statistics, and research findings
- NEVER pad with filler phrases like "In today's rapidly evolving world" or "It's important to note that"
- Every paragraph must teach something. If a paragraph just restates what was already said, delete it.

QUIZ REQUIREMENTS:
Generate 3-5 questions that test the learning objectives above. Design them like this:

- **At least 1 APPLICATION question**: Give a scenario the student hasn't seen and ask them to apply a concept from this chapter. ("A company wants to X. Based on what you learned about Y, what should they do?")
- **At least 1 MISCONCEPTION question**: The wrong answers should represent common misunderstandings. The explanation should teach WHY each wrong answer is wrong.
- **At least 1 CONNECTION question**: Test whether the student can relate concepts within this chapter or to previous chapters.
- Mix of multiple choice (4 options) and true/false.
- Write explanations for EVERY option (correct and incorrect), not just the right answer.

{_build_search_section(search_mode, web_sources)}

OUTPUT FORMAT (JSON - no markdown):
{{
  "content": "Full markdown content with inline citations [1], [2]...",
  "word_count": 1200,
  "key_concepts_explained": ["Concept 1", "Concept 2"],
  "quiz": {{
    "questions": [
      {{
        "id": "q1",
        "type": "multiple_choice",
        "question": "Scenario-based question that tests application?",
        "options": ["Option A", "Option B", "Option C", "Option D"],
        "correct_answer": 1,
        "explanation": "B is correct because [reason]. A is wrong because [common misconception]. C is wrong because [different error]. D is wrong because [plausible but incorrect reasoning]."
      }},
      {{
        "id": "q2",
        "type": "true_false",
        "question": "Statement that tests a common misconception?",
        "options": ["True", "False"],
        "correct_answer": 0,
        "explanation": "True. This is correct because [explanation]. Many learners incorrectly think [misconception] because [why it's tempting]."
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
      "relevance": "Supports the claim that [specific claim from content]"
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


def build_study_guide_prompt(
    course_title: str,
    topic: str,
    chapters_summary: str,
    expertise: str,
    role: str,
    learning_goal: str
) -> str:
    """Build prompt for comprehensive study guide generation."""
    return f"""You are creating a comprehensive study guide for the course "{course_title}" on the topic of {topic}.

LEARNER PROFILE:
- Expertise: {expertise}
- Role: {role}
- Learning Goal: {learning_goal}

COURSE CHAPTERS AND CONTENT:
{chapters_summary}

Generate a comprehensive, unified study guide that covers ALL chapters. The study guide must help the learner review and retain 90%+ of the course's key concepts.

REQUIRED SECTIONS (always include):

1. **Core Concepts**: A table mapping each core concept to its definition, which chapter it appears in, and why it matters.

2. **Must Remember**: For each major topic area, a bulleted list of the most critical facts/rules/principles that MUST be committed to memory. Group by topic.

3. **Key Terms**: Each term should include:
   - Definition
   - "Used in context" — a sentence showing how the term is used
   - "Don't confuse with" — a similar term that learners often mix up

4. **Key Comparisons**: Tables comparing similar or contrasting concepts side by side (e.g., X vs Y with columns for similarities, differences, when to use each).

5. **Common Mistakes**: For each mistake, provide:
   - The mistake (what learners commonly get wrong)
   - Why it's wrong (the root cause of the confusion)
   - Instead (the correct understanding)

CONDITIONAL SECTIONS (include ONLY if relevant to the topic):

6. **Key Events Timeline**: Include ONLY if the topic is historical or chronological. List events in order with dates and significance.

7. **Key Processes**: Include ONLY if the topic involves procedures or step-by-step processes. Show each process as numbered steps with brief explanations.

8. **Key Formulas**: Include ONLY if the topic is STEM/math-related. List each formula with variable definitions and when to use it.

OUTPUT FORMAT (JSON only):
{{
  "core_concepts": [
    {{"concept": "Name", "definition": "...", "chapter": "Chapter Title", "why_it_matters": "..."}}
  ],
  "must_remember": [
    {{"topic": "Topic Area", "points": ["Point 1", "Point 2", "Point 3"]}}
  ],
  "key_terms": [
    {{"term": "Term", "definition": "...", "used_in_context": "...", "dont_confuse_with": "Similar term — because..."}}
  ],
  "key_comparisons": [
    {{"items": ["X", "Y"], "similarities": ["..."], "differences": ["..."], "when_to_use": {{"X": "...", "Y": "..."}}}}
  ],
  "common_mistakes": [
    {{"mistake": "...", "why_wrong": "...", "instead": "..."}}
  ],
  "timeline": null,
  "processes": null,
  "formulas": null
}}

Set conditional sections to null if not relevant. Return ONLY the JSON object."""


def build_course_flashcards_prompt(
    course_title: str,
    chapters_summary: str,
    total_chapters: int
) -> str:
    """Build prompt for course-wide flashcard generation."""
    target_per_chapter = 6
    total_target = target_per_chapter * total_chapters

    return f"""Generate flashcards for the course "{course_title}".

COURSE CHAPTERS AND CONTENT:
{chapters_summary}

Generate {total_target} flashcards ({target_per_chapter} per chapter) that cover:
- Key terms and definitions from each chapter
- Important concepts and their explanations
- Application scenarios (front: scenario, back: correct approach)
- Common misconceptions (front: misconception statement, back: why it's wrong + correct answer)

REQUIREMENTS:
- Every chapter's key concepts must be represented
- Mix of difficulty levels: 40% basic, 40% intermediate, 20% advanced
- Front should be a clear question or prompt
- Back should be a concise, complete answer
- Hint should give a nudge without revealing the answer

OUTPUT FORMAT (JSON only):
{{
  "flashcards": [
    {{
      "id": "fc_1",
      "front": "Question or prompt text",
      "back": "Complete answer",
      "hint": "A helpful nudge",
      "chapter_ref": "Chapter Title",
      "concept": "Key concept this tests",
      "difficulty": "basic|intermediate|advanced"
    }}
  ]
}}

Return ONLY the JSON object."""


def build_final_exam_prompt(
    question_type: str,
    course_title: str,
    chapters_summary: str,
    num_questions: int,
    total_chapters: int
) -> str:
    """Build prompt for a specific section of the final exam."""

    if question_type == "mcq":
        return f"""Generate {num_questions} multiple-choice questions for the final exam of "{course_title}".

COURSE CONTENT:
{chapters_summary}

REQUIREMENTS:
- 4 options per question, exactly 1 correct
- Distribute questions proportionally across all {total_chapters} chapters (every chapter must have at least 1 question)
- Difficulty mix: 30% easy, 50% medium, 20% hard
- Include explanation for why each option is correct or incorrect
- Test application and understanding, not just recall

OUTPUT FORMAT (JSON only):
{{
  "mcq": [
    {{
      "question": "Question text?",
      "options": ["A) ...", "B) ...", "C) ...", "D) ..."],
      "correct_answer": 0,
      "explanation": "A is correct because... B is wrong because... C is wrong because... D is wrong because...",
      "chapter_ref": "Chapter Title",
      "difficulty": "easy|medium|hard"
    }}
  ]
}}

Return ONLY the JSON object."""

    elif question_type == "fill_in_gap":
        return f"""Generate {num_questions} fill-in-the-gap questions for the final exam of "{course_title}".

COURSE CONTENT:
{chapters_summary}

REQUIREMENTS:
- Each question is a sentence with ONE blank (marked as _____)
- Provide the correct answer and 1-2 acceptable alternatives
- Distribute across all {total_chapters} chapters
- Difficulty mix: 30% easy, 50% medium, 20% hard
- Test key terminology and concepts

OUTPUT FORMAT (JSON only):
{{
  "fill_in_gap": [
    {{
      "sentence_with_gap": "The process of _____ converts raw data into structured information.",
      "correct_answer": "data transformation",
      "alternatives": ["data processing", "ETL"],
      "explanation": "Data transformation is the specific term for...",
      "chapter_ref": "Chapter Title",
      "difficulty": "easy|medium|hard"
    }}
  ]
}}

Return ONLY the JSON object."""

    else:  # theory
        return f"""Generate {num_questions} open-ended theory questions for the final exam of "{course_title}".

COURSE CONTENT:
{chapters_summary}

REQUIREMENTS:
- Questions should require 2-3 paragraph answers
- Provide a model answer and grading rubric (key points to cover)
- Distribute across all {total_chapters} chapters
- Difficulty mix: 30% easy, 50% medium, 20% hard
- Test deep understanding, synthesis, and application

OUTPUT FORMAT (JSON only):
{{
  "theory": [
    {{
      "question": "Explain how... and why...",
      "model_answer": "A comprehensive 2-3 paragraph answer...",
      "rubric": ["Key point 1 to cover", "Key point 2 to cover", "Key point 3 to cover"],
      "chapter_ref": "Chapter Title",
      "difficulty": "easy|medium|hard"
    }}
  ]
}}

Return ONLY the JSON object."""


def build_theory_grading_prompt(
    question: str,
    student_answer: str,
    model_answer: str,
    rubric: List[str]
) -> str:
    """Build prompt for LLM-based theory answer grading with per-rubric-point breakdown."""
    rubric_text = "\n".join([f"  {i+1}. {point}" for i, point in enumerate(rubric)])
    max_score = 10
    points_per_rubric = round(max_score / len(rubric), 2) if rubric else 0

    return f"""You are an expert exam grader. Grade the student's answer against the rubric below.

QUESTION:
{question}

MODEL ANSWER:
{model_answer}

RUBRIC POINTS (each worth ~{points_per_rubric} points, total {max_score}):
{rubric_text}

STUDENT ANSWER:
{student_answer}

GRADING INSTRUCTIONS:
- Evaluate each rubric point independently
- Status: "covered" (full points), "partial" (half points), "missed" (0 points)
- Be fair but rigorous — partial credit for incomplete but relevant discussion
- Provide specific feedback for each point explaining why covered/partial/missed
- Overall feedback should be constructive and actionable

OUTPUT FORMAT (JSON only):
{{
  "rubric_breakdown": [
    {{"point": "Rubric point text", "status": "covered|partial|missed", "feedback": "Specific feedback"}}
  ],
  "score": <numeric score out of {max_score}>,
  "max_score": {max_score},
  "feedback": "Overall assessment with suggestions for improvement"
}}

Return ONLY the JSON object."""


def build_course_chat_prompt(
    course_title: str,
    rag_context: str,
    chat_history: List[Dict[str, str]],
    question: str
) -> str:
    """Build prompt for course chat with history and RAG context."""
    history_text = ""
    if chat_history:
        for msg in chat_history:
            role_label = "Student" if msg["role"] == "user" else "Assistant"
            history_text += f"{role_label}: {msg['content']}\n\n"

    history_section = f"""CONVERSATION HISTORY:
{history_text}""" if history_text else ""

    return f"""You are a helpful, knowledgeable course assistant for "{course_title}". Answer the student's question based on the source material and conversation history.

{history_section}
SOURCE MATERIAL:
{rag_context}

CURRENT QUESTION: {question}

INSTRUCTIONS:
- Answer based PRIMARILY on the source material above
- If the conversation history provides relevant context, reference it naturally
- If the source material doesn't contain the answer, say so clearly
- Reference specific sources using [Source N] citations
- Keep the answer concise, educational, and helpful
- If the student is following up on a previous question, connect your answer to the prior discussion"""


# Error recovery prompts
RETRY_PROMPT_ADDENDUM = """

Note: A previous generation attempt had issues. Please:
1. Ensure all content is factually accurate
2. Double-check that citations match the content
3. Make sure quiz questions are clear and unambiguous
4. Verify that explanations are thorough and educational
"""
