"""
Note Generation Prompt Templates
Comprehensive prompts for generating academic study notes at different levels.
"""

# Academic Level Configurations
ACADEMIC_LEVELS = {
    "undergraduate": {
        "depth": "foundational",
        "style": "explanatory with examples",
        "complexity": "moderate",
        "emoji_style": "friendly and engaging",
        "description": "Clear explanations with real-world examples and accessible language"
    },
    "graduate": {
        "depth": "advanced",
        "style": "analytical and research-oriented",
        "complexity": "high",
        "emoji_style": "professional",
        "description": "In-depth analysis with theoretical frameworks and practical applications"
    },
    "msc": {
        "depth": "specialized",
        "style": "technical and methodological",
        "complexity": "very high",
        "emoji_style": "minimal, professional",
        "description": "Technical precision, methodologies, and specialized knowledge"
    },
    "phd": {
        "depth": "exhaustive",
        "style": "critical analysis with research gaps",
        "complexity": "expert level",
        "emoji_style": "academic",
        "description": "Critical analysis, research frontiers, and interdisciplinary connections"
    }
}

# Level-Specific Instructions
LEVEL_INSTRUCTIONS = {
    "undergraduate": """
**UNDERGRADUATE LEVEL APPROACH:**
- Explain concepts clearly with real-world examples
- Define all technical terms immediately when introduced
- Use analogies and comparisons to aid understanding
- Focus on foundational understanding before complexity
- Break down complex processes into simple steps
- Include "Why this matters" explanations
- Use friendly, engaging language
- Provide plenty of examples and applications
""",
    "graduate": """
**GRADUATE LEVEL APPROACH:**
- Provide in-depth analytical perspectives
- Include theoretical frameworks and models
- Discuss practical applications and research implications
- Reference methodological considerations
- Connect concepts to broader academic discourse
- Include critical analysis where appropriate
- Balance theory with practical application
- Assume strong foundational knowledge
""",
    "msc": """
**M.SC LEVEL APPROACH:**
- Focus on technical precision and specialized knowledge
- Include advanced methodological details
- Discuss research methodologies and design considerations
- Provide in-depth technical explanations
- Include advanced problem-solving approaches
- Reference current research and techniques
- Assume expert-level foundational knowledge
- Focus on specialized, technical depth
""",
    "phd": """
**PhD LEVEL APPROACH:**
- Critical analysis of current paradigms and theories
- Identify research gaps and opportunities
- Include methodological critiques and debates
- Discuss theoretical controversies and competing frameworks
- Highlight future research directions
- Make interdisciplinary connections
- Question assumptions and examine limitations
- Focus on pushing knowledge boundaries
"""
}

# Formatting Guidelines
FORMATTING_GUIDELINES = """
**MARKDOWN FORMATTING REQUIREMENTS:**

1. **Headings with Emojis:**
   - # 📚 Main Topic (H1 - use once for document title)
   - ## 🔍 Major Section (H2 - for chapters/major divisions)
   - ### 📖 Subsection (H3 - for sections within chapters)
   - #### 💡 Key Concept (H4 - for important concepts)
   - ##### 🔸 Sub-concept (H5 - for details)
   - ###### 📌 Specific Detail (H6 - for fine-grained info)

2. **Lists:**
   - Use **bullet points** (- or •) for related items without specific order
   - Use **numbered lists** (1., 2., 3.) for sequences, steps, or priorities
   - Nest lists for hierarchical information (max 3 levels deep)

3. **Emphasis:**
   - Use **bold** for key terms, important concepts, definitions
   - Use *italics* for emphasis, foreign terms, or first use of terms
   - Use `code formatting` for technical terms, variables, formulas

4. **Tables:**
   Use tables for comparisons, feature matrices, or structured data:
   ```markdown
   | Category | Feature A | Feature B |
   |----------|-----------|-----------|
   | Item 1   | Detail    | Detail    |
   | Item 2   | Detail    | Detail    |
   ```

5. **Code Blocks:**
   Use for formulas, algorithms, code examples:
   ```language
   Code or formula here
   ```

6. **Blockquotes:**
   Use > for definitions, key statements, important notes:
   > **Definition:** A clear definition of the concept

7. **Mermaid Diagrams:**
   Use for processes, workflows, relationships, hierarchies:
   ```mermaid
   graph TD
       A[Start] --> B[Process]
       B --> C[Decision]
       C -->|Yes| D[End]
       C -->|No| B
   ```

8. **Visual Markers:**
   - ✅ For correct/positive points
   - ❌ For incorrect/negative points
   - ⚠️ For warnings or important notes
   - 💡 For tips or insights
   - 🔑 For key takeaways
   - 📊 For data or statistics
"""

# Master Note Generation Prompt
MASTER_NOTE_GENERATION_PROMPT = """
You are an expert academic note-taker creating comprehensive, extensive study notes for {academic_level} students.

{level_instructions}

**OBJECTIVE:**
Generate thorough, hierarchical markdown notes covering EVERY detail in the provided content. Leave no concept, fact, definition, or example undocumented.

**COVERAGE REQUIREMENTS:**
✅ Extract and document EVERY concept, fact, definition, example, and explanation
✅ Maintain the original document hierarchy (chapters → sections → subsections)
✅ Cover all topics, subtopics, and sub-subtopics completely
✅ Include all examples, case studies, and applications mentioned
✅ No information should be omitted or summarized away

{formatting_guidelines}

**CONTENT TO PROCESS:**
{content_text}

**CONTEXT FROM PREVIOUS SECTIONS (for continuity):**
{previous_context}

**YOUR TASK:**
Generate comprehensive notes for this section that:
1. Follow the hierarchy from the source material
2. Use appropriate markdown formatting with emojis
3. Include ALL information from the content
4. Add mermaid diagrams where processes or relationships are described
5. Use tables for comparisons or structured data
6. Include code blocks for formulas or technical content
7. Maintain academic level-appropriate depth and style

Begin generating the notes now. Remember: COMPREHENSIVE COVERAGE is the priority!
"""

# Section-Specific Prompts

CHAPTER_HEADER_PROMPT = """
Generate a comprehensive chapter header for:

**Chapter Number:** {chapter_number}
**Chapter Title:** {chapter_title}
**Academic Level:** {academic_level}

Create a markdown header section (H2) that includes:
1. Emoji-enhanced chapter title
2. Brief chapter overview (2-3 sentences)
3. Key topics covered (bullet list)

Format:
```markdown
## 🔍 Chapter {chapter_number}: {chapter_title}

**Overview:** [Brief description of what this chapter covers]

**Key Topics:**
- Topic 1
- Topic 2
- Topic 3
```
"""

SECTION_NOTES_PROMPT = """
Generate comprehensive notes for this section:

**Section Title:** {section_title}
**Academic Level:** {academic_level}
**Content:**
{section_content}

**Previous Context:**
{previous_context}

{level_instructions}

Generate detailed notes covering ALL information in this section. Use proper markdown formatting with:
- Hierarchical headings (H3, H4, H5)
- Bullet points and numbered lists
- Tables for comparisons
- Code blocks for formulas
- Blockquotes for definitions
- Mermaid diagrams for processes/relationships

Ensure COMPLETE coverage - no detail should be omitted!
"""

MERMAID_DIAGRAM_PROMPT = """
Analyze this content and generate mermaid diagram code if applicable:

**Content:**
{content}

**Diagram Types to Consider:**
- graph TD (flowcharts, processes)
- graph LR (horizontal workflows)
- sequenceDiagram (interactions, sequences)
- classDiagram (relationships, hierarchies)
- stateDiagram (state transitions)
- gantt (timelines, schedules)

**Instructions:**
1. Identify if a diagram would enhance understanding
2. Choose the most appropriate mermaid diagram type
3. Generate clean, properly formatted mermaid code
4. Return ONLY the mermaid code block, or "NO_DIAGRAM" if not applicable

Example format:
```mermaid
graph TD
    A[Concept A] --> B[Concept B]
    B --> C[Concept C]
    B --> D[Concept D]
```
"""

DIAGRAM_DESCRIPTION_PROMPT = """
Generate a clear description for this diagram based on the surrounding context:

**Diagram Page:** {page}
**Surrounding Context:**
{context}

**Instructions:**
Generate a 1-2 sentence description of what this diagram likely illustrates based on the context.

Example: "Circuit diagram showing a voltage divider with resistors R1 and R2 in series between Vin and ground."
"""

CHAPTER_SUMMARY_PROMPT = """
Generate a comprehensive summary for this chapter:

**Chapter:** {chapter_title}
**Academic Level:** {academic_level}
**Chapter Content:**
{chapter_content}

{level_instructions}

Create a summary section that includes:
1. **Key Concepts:** List of main concepts covered (bullet points)
2. **Important Points:** 3-5 critical takeaways
3. **Connections:** How this chapter relates to previous/next material

Format as a markdown section titled "### 🔑 Chapter Summary"
"""

# Document Analysis Prompts

DOCUMENT_TYPE_CLASSIFICATION_PROMPT = """
Analyze this document sample and classify its type:

**Sample Content:**
{sample_content}

**Possible Types:**
- Textbook chapter
- Research paper
- Lecture notes
- Technical manual
- Academic article
- Tutorial/guide
- Reference documentation
- Report

Return ONLY the document type (no explanation needed).
"""

DOCUMENT_STRUCTURE_ANALYSIS_PROMPT = """
Analyze this document structure and extract metadata:

**Document Info:**
- Total pages: {total_pages}
- Total chunks: {total_chunks}
- Has chapters: {has_chapters}

**Sample Content:**
{sample_content}

**Task:**
Identify the document structure and return a JSON object:
```json
{{
    "document_type": "textbook|paper|manual|etc",
    "has_abstract": true/false,
    "has_introduction": true/false,
    "has_conclusion": true/false,
    "has_references": true/false,
    "has_equations": true/false,
    "has_code": true/false,
    "complexity_level": "beginner|intermediate|advanced|expert",
    "subject_domain": "general description"
}}
```
"""

# Validation Prompts

COMPLETENESS_CHECK_PROMPT = """
Review these generated notes and assess completeness:

**Original Content Chunks:** {num_chunks}
**Generated Notes Length:** {notes_length} characters

**Notes Preview:**
{notes_preview}

Rate completeness on a scale of 0-100 and provide brief assessment:
- 90-100: Exceptional coverage, all details included
- 70-89: Good coverage, minor gaps
- 50-69: Moderate coverage, some gaps
- Below 50: Insufficient coverage, major gaps

Return format:
```
Score: XX
Assessment: [Brief explanation]
Missing: [Any obvious gaps identified]
```
"""

# Helper function to get prompt
def get_prompt(prompt_type: str, **kwargs) -> str:
    """
    Get a formatted prompt by type.

    Args:
        prompt_type: Type of prompt to retrieve
        **kwargs: Variables to format into the prompt

    Returns:
        Formatted prompt string
    """
    prompts = {
        "master": MASTER_NOTE_GENERATION_PROMPT,
        "chapter_header": CHAPTER_HEADER_PROMPT,
        "section_notes": SECTION_NOTES_PROMPT,
        "mermaid": MERMAID_DIAGRAM_PROMPT,
        "diagram_description": DIAGRAM_DESCRIPTION_PROMPT,
        "chapter_summary": CHAPTER_SUMMARY_PROMPT,
        "document_type": DOCUMENT_TYPE_CLASSIFICATION_PROMPT,
        "document_structure": DOCUMENT_STRUCTURE_ANALYSIS_PROMPT,
        "completeness_check": COMPLETENESS_CHECK_PROMPT
    }

    if prompt_type not in prompts:
        raise ValueError(f"Unknown prompt type: {prompt_type}")

    return prompts[prompt_type].format(**kwargs)


def get_level_instructions(academic_level: str) -> str:
    """Get instructions for a specific academic level."""
    return LEVEL_INSTRUCTIONS.get(academic_level, LEVEL_INSTRUCTIONS["graduate"])


def get_level_config(academic_level: str) -> dict[str, str]:
    """Get configuration for a specific academic level."""
    return ACADEMIC_LEVELS.get(academic_level, ACADEMIC_LEVELS["graduate"])


# Testing
if __name__ == "__main__":
    print("Testing prompt templates...")

    print("\n1. Master Prompt for Graduate Level:")
    print("=" * 80)
    master = get_prompt(
        "master",
        academic_level="graduate",
        level_instructions=get_level_instructions("graduate"),
        formatting_guidelines=FORMATTING_GUIDELINES,
        content_text="Sample content about machine learning algorithms...",
        previous_context="Introduction to AI and basic concepts..."
    )
    print(master[:500] + "...")

    print("\n2. Mermaid Diagram Prompt:")
    print("=" * 80)
    mermaid = get_prompt(
        "mermaid",
        content="The process starts with data collection, then preprocessing, followed by model training and evaluation."
    )
    print(mermaid[:300] + "...")

    print("\n3. Level Configurations:")
    print("=" * 80)
    for level, config in ACADEMIC_LEVELS.items():
        print(f"  - {level}: {config['description']}")
