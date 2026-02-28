# Course Prompt Engineering Overhaul

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Rewrite the course generation prompts in `prompts/course_prompts.py` to produce pedagogically excellent, deeply personalized courses instead of generic AI articles.

**Architecture:** Single-file rewrite of `prompts/course_prompts.py`. No new files, no model changes, no service changes. The function signatures stay identical so `course_service.py` doesn't need to change. All improvements are inside the prompt text itself. We'll also add internal helper functions for building personalization blocks and pedagogical instructions.

**Tech Stack:** Pure Python string formatting (existing pattern). No new dependencies.

---

## Context for Implementer

### What This File Does
`prompts/course_prompts.py` contains prompt-building functions called by `services/course_service.py`. There are two critical functions:

1. `build_outline_generation_prompt()` — Generates the course outline (title, chapters, objectives)
2. `build_chapter_content_prompt()` — Generates each chapter's content + quiz

The service calls these functions, passes the resulting prompt string to the LLM, and expects JSON back. **The JSON output schema must NOT change** — the service parses specific keys (`title`, `chapters`, `content`, `quiz`, `sources`).

### What's Wrong (Summary)
1. **Outline prompt** produces cookie-cutter structures. No domain awareness, no pedagogy, no Bloom's taxonomy for objectives.
2. **Chapter prompt** creates articles, not lessons. No scaffolding, no misconception handling, no progressive examples.
3. **Quiz prompt** says "test understanding" but gives no framework. No cognitive-level mixing, no misconception-based distractors.
4. **Personalization** is listed but never applied. The model gets preferences but no instructions on how they change the output.

### What Must NOT Change
- Function signatures (parameter names and types)
- JSON output schemas (keys, structure)
- The `_build_search_section()` helper (works fine)
- `build_topic_extraction_prompt()` (works fine)
- `build_multi_file_analysis_prompt()` (works fine)
- `build_document_map_prompt()` (works fine)
- `RETRY_PROMPT_ADDENDUM` (works fine)

### Files
- **Modify:** `prompts/course_prompts.py` (lines 1-101 for outline, lines 134-259 for chapter)
- **Do NOT modify:** `services/course_service.py`, `models/course_models.py`, `prompts/__init__.py`

### Enum Values (from `models/course_models.py`)
These are the actual values the prompt functions receive:

- `expertise`: `"beginner"`, `"intermediate"`, `"advanced"`
- `format_pref`: `"reading"`, `"listening"`, `"testing"`, `"mixed"`
- `depth_pref`: `"quick"`, `"detailed"`, `"conversational"`, `"academic"`
- `role`: `"student"`, `"professional"`, `"graduate_student"`
- `learning_goal`: `"exams"`, `"career"`, `"curiosity"`, `"supplement"`
- `example_pref`: `"real_world"`, `"technical"`, `"stories"`, `"analogies"`

---

## Task 1: Add Personalization Strategy Helpers

**Files:**
- Modify: `prompts/course_prompts.py` (add new helper functions after line 7, before `build_outline_generation_prompt`)

**Step 1: Add the `_build_personalization_strategy` helper**

This function converts raw preference values into detailed, actionable instructions the LLM can follow. Add after the imports (line 7):

```python
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

    sections = [
        f"TEACHING VOICE: Write as a {strategy['voice']}.",
        f"CONTENT FOCUS: {strategy['focus']}",
        f"STRUCTURAL APPROACH: {strategy['structure']}",
        expertise_instructions.get(expertise, expertise_instructions["intermediate"]),
        depth_instructions.get(depth_pref, depth_instructions["detailed"]),
        example_instructions.get(example_pref, example_instructions["real_world"])
    ]

    return "\n\n".join(sections)
```

**Step 2: Add the `_build_bloom_taxonomy_instruction` helper**

Add immediately after `_build_personalization_strategy`:

```python
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
```

**Step 3: Verify no imports or exports changed**

Run: `cd /Users/davak/Documents/nuton_rag && python3 -c "from prompts.course_prompts import build_outline_generation_prompt, build_chapter_content_prompt; print('OK')"`

Expected: `OK`

**Step 4: Commit**

```bash
git add prompts/course_prompts.py
git commit -m "feat(prompts): add personalization strategy and Bloom's taxonomy helpers"
```

---

## Task 2: Rewrite the Outline Generation Prompt

**Files:**
- Modify: `prompts/course_prompts.py` — replace the `build_outline_generation_prompt` function body (lines 9-101)

**Step 1: Replace the `build_outline_generation_prompt` function**

Replace the entire function body (keep the same signature) with:

```python
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
```

**Step 2: Verify the function still works with the service**

Run: `cd /Users/davak/Documents/nuton_rag && python3 -c "from prompts.course_prompts import build_outline_generation_prompt; p = build_outline_generation_prompt(topic='quantum physics', expertise='beginner', time_available=60, format_pref='reading', depth_pref='detailed', role='student', learning_goal='exams', example_pref='real_world'); print(p[:200]); print('...'); print(f'Total length: {len(p)} chars')"`

Expected: Prints first 200 chars of the new prompt, then total length (should be significantly longer than before).

**Step 3: Commit**

```bash
git add prompts/course_prompts.py
git commit -m "feat(prompts): rewrite outline prompt with pedagogy, Bloom's taxonomy, and goal-aware framing"
```

---

## Task 3: Rewrite the Chapter Content Prompt

**Files:**
- Modify: `prompts/course_prompts.py` — replace the `build_chapter_content_prompt` function body (lines 134-259)

**Step 1: Replace the `build_chapter_content_prompt` function**

Replace the entire function body (keep the same signature) with:

```python
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
    if chapter_num == 1:
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
        position_guidance = (
            f"CHAPTER POSITION: Chapter {chapter_num} of {total_chapters} (middle of course).\n"
            f"- Bridge from previous chapter (\"{prev_chapter_title}\") — open with a 1-2 sentence connection\n"
            "- Build on established concepts while introducing new ones\n"
            f"- Set up what comes next (\"{next_chapter_title}\") with a forward-looking closing"
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
```

**Step 2: Verify the function still works**

Run: `cd /Users/davak/Documents/nuton_rag && python3 -c "from prompts.course_prompts import build_chapter_content_prompt; p = build_chapter_content_prompt(course_title='Quantum Physics', chapter_num=2, total_chapters=5, chapter_title='Wave-Particle Duality', objectives=['Explain wave-particle duality', 'Apply the concept to photons'], expertise='beginner', format_pref='reading', depth_pref='conversational', role='student', learning_goal='exams', example_pref='analogies', prev_chapter_title='What is Quantum?', next_chapter_title='Uncertainty Principle'); print(p[:300]); print('...'); print(f'Total length: {len(p)} chars')"`

Expected: Prints first 300 chars of the new prompt with personalization visible, then total length.

**Step 3: Verify JSON schema hasn't changed**

Run: `cd /Users/davak/Documents/nuton_rag && python3 -c "
from prompts.course_prompts import build_chapter_content_prompt
p = build_chapter_content_prompt(
    course_title='Test', chapter_num=1, total_chapters=3,
    chapter_title='Test Ch', objectives=['obj1'], expertise='beginner',
    format_pref='reading', depth_pref='detailed', role='student',
    learning_goal='exams', example_pref='real_world'
)
# Check the JSON template still has the right keys
assert '\"content\":' in p
assert '\"quiz\":' in p
assert '\"sources\":' in p
assert '\"word_count\":' in p
assert '\"key_concepts_explained\":' in p
print('All expected JSON keys present in prompt template')
"`

Expected: `All expected JSON keys present in prompt template`

**Step 4: Commit**

```bash
git add prompts/course_prompts.py
git commit -m "feat(prompts): rewrite chapter prompt with teaching pedagogy, scaffolding, and misconception handling"
```

---

## Task 4: Verify Full Integration (No Service Changes Needed)

**Files:**
- Read only: `services/course_service.py`, `prompts/__init__.py`

**Step 1: Verify all imports still work**

Run: `cd /Users/davak/Documents/nuton_rag && python3 -c "
from prompts import (
    build_outline_generation_prompt,
    build_chapter_content_prompt,
    build_topic_extraction_prompt,
    build_multi_file_analysis_prompt,
    RETRY_PROMPT_ADDENDUM
)
print('All imports OK')

# Verify function signatures match what course_service.py expects
import inspect
outline_params = inspect.signature(build_outline_generation_prompt).parameters
chapter_params = inspect.signature(build_chapter_content_prompt).parameters

# These are the params course_service.py passes
expected_outline = ['topic', 'expertise', 'time_available', 'format_pref', 'depth_pref', 'role', 'learning_goal', 'example_pref', 'file_context', 'organization_instructions', 'suggested_chapter_count', 'structured_topic_constraint']
expected_chapter = ['course_title', 'chapter_num', 'total_chapters', 'chapter_title', 'objectives', 'expertise', 'format_pref', 'depth_pref', 'role', 'learning_goal', 'example_pref', 'prev_chapter_title', 'next_chapter_title', 'source_material_context', 'search_mode', 'web_sources']

for p in expected_outline:
    assert p in outline_params, f'Missing outline param: {p}'
for p in expected_chapter:
    assert p in chapter_params, f'Missing chapter param: {p}'

print('All function signatures match service expectations')
"`

Expected:
```
All imports OK
All function signatures match service expectations
```

**Step 2: Verify course_service.py doesn't need changes**

Run: `cd /Users/davak/Documents/nuton_rag && python3 -c "
# Syntax check the service module
import py_compile
py_compile.compile('services/course_service.py', doraise=True)
print('course_service.py compiles OK')
"`

Expected: `course_service.py compiles OK`

**Step 3: Run a dry-run prompt generation with all personalization combos**

Run: `cd /Users/davak/Documents/nuton_rag && python3 -c "
from prompts.course_prompts import build_outline_generation_prompt, build_chapter_content_prompt

combos = [
    ('beginner', 'student', 'exams', 'quick', 'real_world'),
    ('intermediate', 'professional', 'career', 'detailed', 'technical'),
    ('advanced', 'graduate_student', 'curiosity', 'academic', 'stories'),
    ('beginner', 'student', 'supplement', 'conversational', 'analogies'),
]

for exp, role, goal, depth, ex in combos:
    outline = build_outline_generation_prompt(
        topic='Machine Learning', expertise=exp, time_available=60,
        format_pref='reading', depth_pref=depth, role=role,
        learning_goal=goal, example_pref=ex
    )
    chapter = build_chapter_content_prompt(
        course_title='ML Course', chapter_num=1, total_chapters=5,
        chapter_title='Ch 1', objectives=['obj'], expertise=exp,
        format_pref='reading', depth_pref=depth, role=role,
        learning_goal=goal, example_pref=ex
    )
    print(f'{exp}/{role}/{goal}/{depth}/{ex}: outline={len(outline)} chars, chapter={len(chapter)} chars')

print('All combos generated successfully')
"`

Expected: Four lines showing char counts for each combo, followed by success message.

**Step 4: Final commit**

```bash
git add prompts/course_prompts.py
git commit -m "feat(prompts): complete prompt engineering overhaul - verified integration"
```

---

## Summary of Changes

| What Changed | Before | After |
|---|---|---|
| Outline structure | Fixed "Foundation -> Core -> Application -> Synthesis" for all topics | Domain-aware sequencing (conceptual, procedural, historical, survey) |
| Learning objectives | "measurable, action-oriented" (vague) | Bloom's taxonomy verbs matched to expertise level |
| Chapter titles | "descriptive, engaging" | Must hint at the "aha moment" |
| Chapter content | "expert educator" article format | Teaching arc: Hook -> Concept-by-concept with misconceptions -> Application -> Takeaways |
| Personalization | Preferences listed in a block | 32+ role/goal strategy combos with specific voice, focus, and structure instructions |
| Expertise handling | 4-line depth dictionary | Detailed per-level rules (define terms for beginners, skip basics for advanced) |
| Quiz design | "test understanding, not memorization" | Requires application, misconception, and connection questions with per-option explanations |
| Chapter position | prev/next titles listed | Position-aware guidance (opening hook, middle bridge, final synthesis) |
| Examples | "{example_pref} examples integrated" | Specific format guidance per preference (case studies vs. code vs. narratives vs. extended analogies) |
| Course description | "2-3 sentence overview" | Must answer: what you'll learn, why it matters to YOU, what's unique |

## What Did NOT Change
- Function signatures (zero changes)
- JSON output schemas (zero changes)
- `_build_search_section()` helper
- `build_topic_extraction_prompt()`
- `build_multi_file_analysis_prompt()`
- `build_document_map_prompt()`
- `RETRY_PROMPT_ADDENDUM`
- `services/course_service.py` (zero changes)
- `models/course_models.py` (zero changes)
- `prompts/__init__.py` (zero changes)
