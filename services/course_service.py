"""
Core course generation service.
Orchestrates outline generation, chapter creation, and storage.
Following DRY principle - modular, reusable components.
"""

import json
import time
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

from models.course_models import (
    Course, Chapter, LearningProfile, CourseStatus, SourceType,
    PersonalizationParams, SourceFile, OrganizationType
)
from utils.file_storage import (
    CourseStorage, LearningProfileStorage, GenerationLogger, generate_uuid
)
from utils.model_config import ModelConfig, estimate_course_cost
from prompts.course_prompts import (
    build_outline_generation_prompt,
    build_chapter_content_prompt,
    build_topic_extraction_prompt,
    build_multi_file_analysis_prompt
)

# Import existing clients
import clients.openai_client as openai_client
from clients.groq_client import generate_answer

logger = logging.getLogger(__name__)


class CourseGenerationError(Exception):
    """Base exception for course generation errors"""
    pass


class OutlineGenerationError(CourseGenerationError):
    """Failed to generate course outline"""
    pass


class ChapterGenerationError(CourseGenerationError):
    """Failed to generate chapter"""
    def __init__(self, chapter_num: int, message: str):
        self.chapter_num = chapter_num
        super().__init__(f"Chapter {chapter_num} generation failed: {message}")


class CourseService:
    """Main service for course generation"""
    
    def __init__(self):
        self.storage = CourseStorage()
        self.profile_storage = LearningProfileStorage()
        self.logger = GenerationLogger()
    
    async def create_course_from_topic(
        self,
        user_id: str,
        topic: str,
        context: Dict[str, Any],
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate complete course from topic string.
        Blocking operation - returns full course in 45-60s.
        """
        start_time = time.time()
        
        # Get or create learning profile
        profile = self._get_or_create_profile(user_id, context)
        
        # Get model config
        model_config = ModelConfig.get_config(model)
        
        # Generate course
        try:
            course = await self._generate_full_course(
                user_id=user_id,
                topic=topic,
                source_type=SourceType.TOPIC,
                profile=profile,
                model_config=model_config,
                source_files=None
            )
            
            generation_time = round(time.time() - start_time, 2)
            
            # Log success
            self.logger.log_generation({
                "type": "topic_course",
                "user_id": user_id,
                "course_id": course["id"],
                "topic": topic,
                "model": model_config["model"],
                "chapters": course["total_chapters"],
                "generation_time": generation_time,
                "status": "success",
                "estimated_cost": estimate_course_cost(model_config["model"], course["total_chapters"])
            })
            
            return {
                "course_id": course["id"],
                "status": CourseStatus.READY,
                "course": course,
                "storage_path": f"courses/course_{course['id']}/",
                "generation_time_seconds": generation_time
            }
            
        except Exception as e:
            logger.error(f"Course generation failed: {e}")
            self.logger.log_generation({
                "type": "topic_course",
                "user_id": user_id,
                "topic": topic,
                "model": model_config["model"],
                "status": "error",
                "error": str(e)
            })
            raise
    
    async def create_course_from_files(
        self,
        user_id: str,
        files: List[Dict[str, Any]],  # Processed file data
        organization: str,
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate course from uploaded files.
        Multi-file detection and organization included.
        """
        start_time = time.time()
        
        # Get profile
        profile = self._get_or_create_profile(user_id, {})
        model_config = ModelConfig.get_config(model)
        
        # Multi-file analysis
        if len(files) > 1:
            file_context, chosen_org = await self._analyze_multi_files(files, organization)
        else:
            file_context = files[0]["extracted_text"][:3000] if files else ""
            chosen_org = None
        
        # Build topic from files
        combined_topic = " + ".join([f["topic"] for f in files]) if len(files) > 1 else files[0]["topic"]
        
        # Generate course
        try:
            course = await self._generate_full_course(
                user_id=user_id,
                topic=combined_topic,
                source_type=SourceType.FILES,
                profile=profile,
                model_config=model_config,
                source_files=[{"file_id": generate_uuid(), "filename": f["filename"], "extracted_topic": f["topic"]} for f in files],
                file_context=file_context,
                organization=chosen_org
            )
            
            generation_time = round(time.time() - start_time, 2)
            
            self.logger.log_generation({
                "type": "file_course",
                "user_id": user_id,
                "course_id": course["id"],
                "files": [f["filename"] for f in files],
                "organization": chosen_org.value if chosen_org else None,
                "model": model_config["model"],
                "generation_time": generation_time,
                "status": "success"
            })
            
            return {
                "course_id": course["id"],
                "status": CourseStatus.READY,
                "detected_topics": [f["topic"] for f in files],
                "organization_chosen": chosen_org.value if chosen_org else None,
                "course": course,
                "storage_path": f"courses/course_{course['id']}/",
                "generation_time_seconds": generation_time
            }
            
        except Exception as e:
            logger.error(f"File course generation failed: {e}")
            raise
    
    async def _generate_full_course(
        self,
        user_id: str,
        topic: str,
        source_type: SourceType,
        profile: LearningProfile,
        model_config: Dict[str, Any],
        source_files: Optional[List[Dict]] = None,
        file_context: Optional[str] = None,
        organization: Optional[OrganizationType] = None
    ) -> Dict[str, Any]:
        """Internal method: Generate complete course with all chapters"""
        
        # Step 1: Generate outline
        outline = await self._generate_outline(
            topic=topic,
            profile=profile,
            model_config=model_config,
            file_context=file_context,
            organization=organization
        )
        
        # Create course record
        course_id = generate_uuid()
        personalization = PersonalizationParams(
            format_pref=profile.format_pref,
            depth_pref=profile.depth_pref,
            role=profile.role,
            learning_goal=profile.learning_goal,
            example_pref=profile.example_pref
        )
        
        course_data = {
            "id": course_id,
            "user_id": user_id,
            "title": outline["title"],
            "description": outline["description"],
            "topic": topic,
            "source_type": source_type,
            "source_files": source_files or [],
            "multi_file_organization": organization.value if organization else None,
            "total_chapters": len(outline["chapters"]),
            "estimated_time": outline["total_estimated_time"],
            "status": CourseStatus.GENERATING,
            "personalization_params": personalization.dict(),
            "outline": outline,
            "model_used": model_config["model"],
            "created_at": datetime.utcnow(),
            "completed_at": None
        }
        
        # Save initial course
        self.storage.save_course(course_data)
        
        # Step 2: Generate chapters sequentially
        chapters = []
        for i, chapter_outline in enumerate(outline["chapters"]):
            try:
                chapter = await self._generate_chapter(
                    course_id=course_id,
                    course_title=outline["title"],
                    chapter_outline=chapter_outline,
                    total_chapters=len(outline["chapters"]),
                    profile=profile,
                    model_config=model_config,
                    prev_chapter_title=chapters[-1]["title"] if chapters else None,
                    next_chapter_title=outline["chapters"][i + 1]["title"] if i < len(outline["chapters"]) - 1 else None,
                    file_context=file_context if i == 0 else None  # Only use file context for first chapter
                )
                
                chapters.append(chapter)
                self.storage.save_chapter(course_id, chapter)
                logger.info(f"Generated chapter {i + 1}/{len(outline['chapters'])}: {chapter['title']}")
                
            except Exception as e:
                logger.error(f"Failed to generate chapter {i + 1}: {e}")
                raise ChapterGenerationError(i + 1, str(e))
        
        # Update course status
        course_data["status"] = CourseStatus.READY
        course_data["completed_at"] = datetime.utcnow()
        self.storage.save_course(course_data)
        
        return course_data
    
    async def _generate_outline(
        self,
        topic: str,
        profile: LearningProfile,
        model_config: Dict[str, Any],
        file_context: Optional[str] = None,
        organization: Optional[OrganizationType] = None
    ) -> Dict[str, Any]:
        """Generate course outline"""
        
        # Build organization instructions if multi-file
        org_instructions = None
        if organization:
            org_instructions = f"""This course uses '{organization.value}' organization:
- Thematic Bridge: Create unified narrative connecting all topics
- Sequential Sections: Teach topics as distinct but related sections
- Separate Courses: Not applicable here"""
        
        prompt = build_outline_generation_prompt(
            topic=topic,
            expertise=profile.expertise if hasattr(profile, 'expertise') else 'beginner',
            time_available=60,  # Default, could be personalized
            format_pref=profile.format_pref.value,
            depth_pref=profile.depth_pref.value,
            role=profile.role.value,
            learning_goal=profile.learning_goal.value,
            example_pref=profile.example_pref.value,
            file_context=file_context,
            organization_instructions=org_instructions
        )
        
        # Call appropriate model
        response = await self._call_model(prompt, model_config, expect_json=True)
        
        if not response or "chapters" not in response:
            raise OutlineGenerationError("Invalid outline response from model")
        
        return response
    
    async def _generate_chapter(
        self,
        course_id: str,
        course_title: str,
        chapter_outline: Dict[str, Any],
        total_chapters: int,
        profile: LearningProfile,
        model_config: Dict[str, Any],
        prev_chapter_title: Optional[str] = None,
        next_chapter_title: Optional[str] = None,
        file_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate single chapter with content and quiz"""
        
        chapter_id = generate_uuid()
        
        prompt = build_chapter_content_prompt(
            course_title=course_title,
            chapter_num=chapter_outline["order"],
            total_chapters=total_chapters,
            chapter_title=chapter_outline["title"],
            objectives=chapter_outline["objectives"],
            expertise=profile.expertise if hasattr(profile, 'expertise') else 'beginner',
            format_pref=profile.format_pref.value,
            depth_pref=profile.depth_pref.value,
            role=profile.role.value,
            learning_goal=profile.learning_goal.value,
            example_pref=profile.example_pref.value,
            prev_chapter_title=prev_chapter_title,
            next_chapter_title=next_chapter_title,
            source_material_context=file_context
        )
        
        response = await self._call_model(prompt, model_config, expect_json=True)
        
        if not response or "content" not in response:
            raise ChapterGenerationError(chapter_outline["order"], "Invalid chapter response")
        
        # Build chapter data
        chapter = {
            "id": chapter_id,
            "course_id": course_id,
            "order": chapter_outline["order"],
            "title": chapter_outline["title"],
            "learning_objectives": chapter_outline["objectives"],
            "content": response["content"],
            "content_format": "markdown",
            "estimated_time": chapter_outline["estimated_time"],
            "key_concepts": chapter_outline["key_concepts"],
            "sources": response.get("sources", []),
            "quiz": response.get("quiz", {"questions": []}),
            "status": "ready",
            "generated_at": datetime.utcnow(),
            "word_count": response.get("word_count", len(response["content"].split()))
        }
        
        return chapter
    
    async def _call_model(
        self,
        prompt: str,
        model_config: Dict[str, Any],
        expect_json: bool = True
    ) -> Dict[str, Any]:
        """Call appropriate model based on configuration"""
        
        provider = model_config["provider"]
        
        if provider == "anthropic":
            return await self._call_claude(prompt, model_config, expect_json)
        elif provider == "openai":
            return await self._call_openai(prompt, model_config, expect_json)
        elif provider == "groq":
            return await self._call_groq(prompt, model_config, expect_json)
        else:
            raise ValueError(f"Unknown provider: {provider}")
    
    async def _call_claude(
        self,
        prompt: str,
        model_config: Dict[str, Any],
        expect_json: bool
    ) -> Dict[str, Any]:
        """Call Claude API"""
        try:
            # Use existing openai_client with Anthropic
            import anthropic
            
            client = anthropic.Anthropic()
            
            # Build messages
            messages = [{"role": "user", "content": prompt}]
            
            # Add web search if supported
            tools = [{"type": "web_search"}] if model_config.get("supports_search") else None
            
            response = client.messages.create(
                model=model_config["model"],
                max_tokens=model_config["max_tokens"],
                temperature=model_config.get("temperature", 0.7),
                messages=messages,
                tools=tools
            )
            
            content = response.content[0].text if response.content else ""
            
            if expect_json:
                return self._extract_json(content)
            return {"content": content}
            
        except Exception as e:
            logger.error(f"Claude API error: {e}")
            raise
    
    async def _call_openai(
        self,
        prompt: str,
        model_config: Dict[str, Any],
        expect_json: bool
    ) -> Dict[str, Any]:
        """Call OpenAI API"""
        try:
            system_prompt = "You be an expert educational content creator."
            if expect_json:
                system_prompt += " You MUST respond with ONLY a valid JSON object. No markdown code blocks, no extra text. Start your response with { and end with }."

            response = openai_client.generate_answer(
                query=prompt,
                context_chunks=[],
                system_prompt=system_prompt,
                model=model_config["model"]
            )
            
            if expect_json and isinstance(response, tuple):
                answer = response[0]
                return self._extract_json(answer)
            elif isinstance(response, tuple):
                return {"content": response[0]}
            else:
                return response
                
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise
    
    async def _call_groq(
        self,
        prompt: str,
        model_config: Dict[str, Any],
        expect_json: bool
    ) -> Dict[str, Any]:
        """Call Groq API directly with response_format support"""
        import os
        from groq import Groq

        try:
            client = Groq(api_key=os.getenv("GROQ_API_KEY"))

            system_prompt = "You are an expert educational content creator."
            if expect_json:
                system_prompt += " Always respond with valid JSON."

            completion_params = {
                "model": model_config["model"],
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": model_config.get("max_tokens", 8192),
                "temperature": model_config.get("temperature", 0.7),
                "stream": False
            }

            if expect_json:
                completion_params["response_format"] = {"type": "json_object"}

            response = client.chat.completions.create(**completion_params)
            answer = response.choices[0].message.content

            # Log if response was truncated
            if response.choices[0].finish_reason == "length":
                logger.warning(f"Groq response truncated (finish_reason=length). Response length: {len(answer)}")

            if expect_json:
                return self._extract_json(answer)
            return {"content": answer}

        except Exception as e:
            logger.error(f"Groq API error: {e}")
            raise
    
    def _extract_json(self, text: str) -> Dict[str, Any]:
        """Extract JSON from text response"""
        try:
            # Try direct parsing first
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        # Try to find JSON block
        import re
        
        # Look for code blocks
        patterns = [
            r'```json\s*(.*?)\s*```',
            r'```\s*(.*?)\s*```',
            r'\{.*\}'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                try:
                    return json.loads(match)
                except:
                    continue
        
        logger.error(f"Could not extract JSON from: {text[:500]}")
        raise ValueError("Failed to parse JSON response")
    
    def _get_or_create_profile(
        self,
        user_id: str,
        context: Dict[str, Any]
    ) -> LearningProfile:
        """Get existing profile or create default"""
        
        profile_data = self.profile_storage.get_profile(user_id)
        
        if profile_data:
            return LearningProfile(**profile_data)
        
        # Create default profile from context or use defaults
        default_profile = LearningProfile(
            user_id=user_id,
            format_pref=context.get("format_pref", "reading"),
            depth_pref=context.get("depth_pref", "detailed"),
            role=context.get("role", "student"),
            learning_goal=context.get("learning_goal", "curiosity"),
            example_pref=context.get("example_pref", "real_world")
        )
        
        # Save it
        self.profile_storage.save_profile(default_profile.dict())
        
        return default_profile
    
    def _chunk_document_for_course(
        self,
        extracted_text: str,
        filename: str
    ) -> List[Dict[str, Any]]:
        """
        Chunk full document text into overlapping semantic chunks.
        Uses existing Chonkie client. Returns list of chunks with index IDs.
        """
        from clients.chonkie_client import chunk_document

        chunks = chunk_document(
            text=extracted_text,
            chunk_size=512,
            overlap_tokens=80,
            recipe="markdown",
            min_characters_per_chunk=50
        )

        # Tag each chunk with an index for tracking
        for i, chunk in enumerate(chunks):
            chunk["chunk_index"] = i
            chunk["source_file"] = filename

        logger.info(f"Chunked {filename}: {len(chunks)} chunks from {len(extracted_text)} chars")
        return chunks

    async def _build_document_map(
        self,
        chunks: List[Dict[str, Any]],
        model_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Build a structured map of document content from chunks.
        Summarizes each chunk, then creates a topic-to-chunk mapping.
        """
        from prompts.course_prompts import build_document_map_prompt

        # Create brief summaries of each chunk (first 200 chars as proxy)
        chunk_summaries = []
        for chunk in chunks:
            text = chunk.get("text", "")
            summary = text[:200].replace("\n", " ").strip()
            chunk_summaries.append({
                "index": chunk["chunk_index"],
                "summary": summary
            })

        prompt = build_document_map_prompt(chunk_summaries)
        doc_map = await self._call_model(prompt, model_config, expect_json=True)

        if not doc_map or "topics" not in doc_map:
            raise ValueError("Failed to generate document map")

        # Verify coverage - every chunk should be in at least one topic
        mapped_indices = set()
        for topic in doc_map["topics"]:
            mapped_indices.update(topic.get("chunk_indices", []))

        all_indices = set(range(len(chunks)))
        unmapped = all_indices - mapped_indices

        if unmapped:
            logger.warning(f"Unmapped chunks: {unmapped}. Adding to closest topic.")
            # Assign unmapped chunks to the last topic as fallback
            if doc_map["topics"]:
                doc_map["topics"][-1]["chunk_indices"].extend(list(unmapped))

        logger.info(f"Document map: {len(doc_map['topics'])} topics covering {len(chunks)} chunks")
        return doc_map

    def _get_chunks_for_chapter(
        self,
        chapter_outline: Dict[str, Any],
        doc_map: Dict[str, Any],
        all_chunks: List[Dict[str, Any]],
        max_context_tokens: int = 3000
    ) -> str:
        """
        Retrieve relevant chunks for a specific chapter.
        Uses document map topic-to-chunk mapping + keyword matching.
        Returns concatenated chunk text, capped at max_context_tokens.
        """
        chapter_title = chapter_outline["title"].lower()
        chapter_objectives = " ".join(chapter_outline.get("objectives", [])).lower()
        chapter_concepts = " ".join(chapter_outline.get("key_concepts", [])).lower()
        chapter_text = f"{chapter_title} {chapter_objectives} {chapter_concepts}"

        # Score each topic in the doc map by relevance to this chapter
        scored_topics = []
        for topic in doc_map.get("topics", []):
            topic_text = f"{topic['topic']} {topic.get('description', '')}".lower()

            # Simple keyword overlap scoring
            chapter_words = set(chapter_text.split())
            topic_words = set(topic_text.split())
            # Remove common stop words
            stop_words = {"the", "a", "an", "and", "or", "to", "of", "in", "for", "is", "on", "with", "this", "that", "be", "can", "will", "are", "after", "students"}
            chapter_words -= stop_words
            topic_words -= stop_words

            overlap = len(chapter_words & topic_words)
            scored_topics.append((overlap, topic))

        # Sort by relevance score (highest first)
        scored_topics.sort(key=lambda x: x[0], reverse=True)

        # Collect chunks from most relevant topics
        collected_indices = []
        for score, topic in scored_topics:
            if score > 0:
                collected_indices.extend(topic.get("chunk_indices", []))

        # If no matches, take from top 2 topics anyway
        if not collected_indices and scored_topics:
            for _, topic in scored_topics[:2]:
                collected_indices.extend(topic.get("chunk_indices", []))

        # Deduplicate while preserving order
        seen = set()
        unique_indices = []
        for idx in collected_indices:
            if idx not in seen:
                seen.add(idx)
                unique_indices.append(idx)

        # Build context from chunks, respecting token limit
        context_parts = []
        token_count = 0
        for idx in unique_indices:
            if idx < len(all_chunks):
                chunk_text = all_chunks[idx].get("text", "")
                # Rough token estimate: 1 token ~ 4 chars
                chunk_tokens = len(chunk_text) // 4
                if token_count + chunk_tokens > max_context_tokens:
                    break
                context_parts.append(f"[Source Section {idx + 1}]\n{chunk_text}")
                token_count += chunk_tokens

        context = "\n\n---\n\n".join(context_parts)
        logger.info(f"Chapter '{chapter_outline['title']}': {len(context_parts)} chunks, ~{token_count} tokens")
        return context

    async def _analyze_multi_files(
        self,
        files: List[Dict[str, Any]],
        requested_org: str
    ) -> tuple:
        """Analyze multiple files and determine organization"""
        
        # Build topics list
        topics = [{"filename": f["filename"], "topic": f["topic"]} for f in files]
        
        if requested_org != "auto":
            # Use requested organization
            org_map = {
                "thematic_bridge": OrganizationType.THEMATIC_BRIDGE,
                "sequential_sections": OrganizationType.SEQUENTIAL_SECTIONS,
                "separate_courses": OrganizationType.SEPARATE_COURSES
            }
            chosen_org = org_map.get(requested_org, OrganizationType.SEQUENTIAL_SECTIONS)
            
            # Build file context
            combined_text = "\n\n---\n\n".join([f["extracted_text"][:1500] for f in files])
            return combined_text, chosen_org
        
        # Auto-detect: Use Claude to analyze
        prompt = build_multi_file_analysis_prompt(topics)
        model_config = ModelConfig.get_config("claude-sonnet-4")
        
        analysis = await self._call_model(prompt, model_config, expect_json=False)
        analysis_text = analysis.get("content", "")
        
        # Simple keyword-based selection (could be more sophisticated)
        if "thematic_bridge" in analysis_text.lower() or "closely related" in analysis_text.lower():
            chosen_org = OrganizationType.THEMATIC_BRIDGE
        elif "separate" in analysis_text.lower() or "unrelated" in analysis_text.lower():
            chosen_org = OrganizationType.SEPARATE_COURSES
        else:
            chosen_org = OrganizationType.SEQUENTIAL_SECTIONS
        
        # Build file context
        if chosen_org == OrganizationType.THEMATIC_BRIDGE:
            # Interleave content to show connections
            combined_text = "\n\n---\n\n".join([f["extracted_text"][:1500] for f in files])
        else:
            # Sequential - keep separate
            combined_text = "\n\n=== SECTION BREAK ===\n\n".join(
                [f"**{f['topic']}**\n\n{f['extracted_text'][:1500]}" for f in files]
            )
        
        return combined_text, chosen_org
    
    # Public API methods
    
    def get_course(self, course_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve full course with chapters"""
        return self.storage.get_course(course_id)
    
    def get_chapter(self, course_id: str, chapter_order: int) -> Optional[Dict[str, Any]]:
        """Retrieve specific chapter"""
        return self.storage.get_chapter(course_id, chapter_order)
    
    def list_user_courses(self, user_id: str) -> List[Dict[str, Any]]:
        """List all courses for user"""
        return self.storage.list_courses(user_id)
    
    def get_learning_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user's learning profile"""
        return self.profile_storage.get_profile(user_id)
    
    def save_learning_profile(self, profile_data: Dict[str, Any]) -> bool:
        """Save learning profile"""
        profile = LearningProfile(**profile_data)
        return self.profile_storage.save_profile(profile.dict())
