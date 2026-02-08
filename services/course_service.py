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
from utils.model_config import ModelConfig, estimate_course_cost, get_search_mode
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
        Generate course from uploaded files using RAG pipeline.
        Full document chunking -> document map -> per-chapter retrieval.

        For separate_courses organization: generates one course per file
        and returns a multi-course response.
        """
        import asyncio
        start_time = time.time()

        # Get profile and model config
        profile = self._get_or_create_profile(user_id, {})
        model_config = ModelConfig.get_config(model)

        # Multi-file organization analysis (if multiple files)
        chosen_org = None
        if len(files) > 1:
            _, chosen_org = await self._analyze_multi_files(files, organization, model_config)

        # Branch: SEPARATE_COURSES - generate one course per file
        if chosen_org == OrganizationType.SEPARATE_COURSES:
            courses_info = []
            for f in files:
                course = await self._generate_single_file_course(
                    user_id=user_id,
                    file_data=f,
                    profile=profile,
                    model_config=model_config
                )
                courses_info.append({
                    "id": course["id"],
                    "title": course["title"],
                    "topic": course["topic"],
                    "status": CourseStatus.READY,
                    "total_chapters": course["total_chapters"],
                    "estimated_time": course["estimated_time"],
                    "storage_path": f"courses/course_{course['id']}/"
                })

            generation_time = round(time.time() - start_time, 2)

            self.logger.log_generation({
                "type": "separate_courses",
                "user_id": user_id,
                "files": [f["filename"] for f in files],
                "total_courses": len(courses_info),
                "model": model_config["model"],
                "generation_time": generation_time,
                "status": "success"
            })

            return {
                "organization": "separate_courses",
                "total_courses": len(courses_info),
                "courses": courses_info,
                "generation_time_seconds": generation_time
            }

        # Single-course path (thematic_bridge, sequential_sections, or single file)

        # Step 1: Chunk ALL files
        all_chunks = []
        for f in files:
            file_chunks = self._chunk_document_for_course(
                extracted_text=f["extracted_text"],
                filename=f["filename"]
            )
            all_chunks.extend(file_chunks)

        logger.info(f"Total chunks from {len(files)} file(s): {len(all_chunks)}")

        # Step 1b+2: Embed/upsert and build doc map in parallel (independent after chunking)
        course_id = generate_uuid()
        embed_task = self._embed_and_upsert_chunks(
            course_id=course_id,
            all_chunks=all_chunks,
            source_files=[{"filename": f["filename"]} for f in files]
        )
        doc_map_task = self._build_document_map(all_chunks, model_config)
        _, doc_map = await asyncio.gather(embed_task, doc_map_task)

        # Step 3: Build topic from files
        combined_topic = " + ".join([f["topic"] for f in files]) if len(files) > 1 else files[0]["topic"]

        # Step 4: Build doc map context (flat text fallback)
        doc_map_context = "DOCUMENT MAP (topics found in uploaded material):\n"
        for topic in doc_map.get("topics", []):
            doc_map_context += f"- {topic['topic']}: {topic.get('description', '')} [{len(topic.get('chunk_indices', []))} sections]\n"

        try:
            course = await self._generate_full_course_from_files(
                user_id=user_id,
                course_id=course_id,
                topic=combined_topic,
                profile=profile,
                model_config=model_config,
                source_files=[{"file_id": generate_uuid(), "filename": f["filename"], "extracted_topic": f["topic"]} for f in files],
                doc_map=doc_map,
                doc_map_context=doc_map_context,
                all_chunks=all_chunks,
                organization=chosen_org
            )

            generation_time = round(time.time() - start_time, 2)

            self.logger.log_generation({
                "type": "file_course",
                "user_id": user_id,
                "course_id": course["id"],
                "files": [f["filename"] for f in files],
                "total_chunks": len(all_chunks),
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
                "document_map": doc_map,
                "total_chunks_processed": len(all_chunks),
                "course": course,
                "storage_path": f"courses/course_{course['id']}/",
                "generation_time_seconds": generation_time
            }

        except Exception as e:
            logger.error(f"File course generation failed: {e}")
            raise

    async def create_course_from_topic_with_files(
        self,
        user_id: str,
        topic: str,
        files: List[Dict[str, Any]],
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate course from topic + supplementary files.
        Topic drives the structure, files provide supplementary context.
        """
        import asyncio
        start_time = time.time()

        profile = self._get_or_create_profile(user_id, {})
        model_config = ModelConfig.get_config(model)

        # Step 1: Chunk all files
        all_chunks = []
        for f in files:
            file_chunks = self._chunk_document_for_course(
                extracted_text=f["extracted_text"],
                filename=f["filename"]
            )
            all_chunks.extend(file_chunks)

        logger.info(f"Topic+files: {len(all_chunks)} chunks from {len(files)} file(s)")

        # Step 2: Embed/upsert and build doc map in parallel
        course_id = generate_uuid()
        embed_task = self._embed_and_upsert_chunks(
            course_id=course_id,
            all_chunks=all_chunks,
            source_files=[{"filename": f["filename"]} for f in files]
        )
        doc_map_task = self._build_document_map(all_chunks, model_config)
        _, doc_map = await asyncio.gather(embed_task, doc_map_task)

        # Build doc map context
        doc_map_context = "SUPPLEMENTARY MATERIAL MAP (from uploaded files):\n"
        for t in doc_map.get("topics", []):
            doc_map_context += f"- {t['topic']}: {t.get('description', '')} [{len(t.get('chunk_indices', []))} sections]\n"

        source_files = [
            {"file_id": generate_uuid(), "filename": f["filename"], "extracted_topic": f.get("topic", "")}
            for f in files
        ]

        try:
            # Step 3: Generate course — topic drives structure, files supplement
            course = await self._generate_full_course(
                user_id=user_id,
                topic=topic,
                source_type=SourceType.TOPIC,
                profile=profile,
                model_config=model_config,
                source_files=source_files,
                doc_map=doc_map,
                doc_map_context=doc_map_context,
                all_chunks=all_chunks,
                course_id=course_id  # Align with Pinecone upsert
            )

            generation_time = round(time.time() - start_time, 2)

            self.logger.log_generation({
                "type": "topic_with_files_course",
                "user_id": user_id,
                "course_id": course["id"],
                "topic": topic,
                "supplementary_files": [f["filename"] for f in files],
                "total_chunks": len(all_chunks),
                "model": model_config["model"],
                "generation_time": generation_time,
                "status": "success"
            })

            return {
                "course_id": course["id"],
                "status": CourseStatus.READY,
                "topic": topic,
                "supplementary_files": [f["filename"] for f in files],
                "document_map": doc_map,
                "total_chunks_processed": len(all_chunks),
                "course": course,
                "storage_path": f"courses/course_{course['id']}/",
                "generation_time_seconds": generation_time
            }

        except Exception as e:
            logger.error(f"Topic+files course generation failed: {e}")
            raise

    async def _assess_topic_complexity(
        self,
        topic: str,
        model_config: Dict[str, Any]
    ) -> tuple:
        """
        Quick LLM call to assess topic complexity.
        Returns (suggested_chapters: int, time_estimate: int).
        Fallback on error: (5, 60).
        """
        prompt = f"""Assess the complexity of the following topic for an educational course.

TOPIC: {topic}

Consider:
- Breadth of sub-topics
- Depth of foundational concepts needed
- Whether the topic is narrow/focused or broad/multidisciplinary

Return ONLY a JSON object:
{{
  "suggested_chapters": <int between 3 and 10>,
  "time_estimate_minutes": <int between 30 and 180>,
  "reasoning": "<one sentence>"
}}"""

        try:
            result = await self._call_model(prompt, model_config, expect_json=True)
            chapters = max(3, min(10, int(result.get("suggested_chapters", 5))))
            time_est = max(30, min(180, int(result.get("time_estimate_minutes", 60))))
            logger.info(f"Topic complexity for '{topic}': {chapters} chapters, {time_est} min — {result.get('reasoning', '')}")
            return chapters, time_est
        except Exception as e:
            logger.warning(f"Topic complexity assessment failed: {e}. Using defaults.")
            return 5, 60

    async def _generate_full_course(
        self,
        user_id: str,
        topic: str,
        source_type: SourceType,
        profile: LearningProfile,
        model_config: Dict[str, Any],
        source_files: Optional[List[Dict]] = None,
        file_context: Optional[str] = None,
        organization: Optional[OrganizationType] = None,
        doc_map: Optional[Dict[str, Any]] = None,
        doc_map_context: Optional[str] = None,
        all_chunks: Optional[List[Dict[str, Any]]] = None,
        course_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Internal method: Generate complete course with all chapters.
        Supports dynamic chapter counts, parallel generation, and model-aware search.
        """
        import asyncio

        # Step 1: Determine chapter count and time
        # For topic+files hybrid, use doc map if available
        if doc_map:
            suggested_chapters, dynamic_time = self._calculate_chapter_count_from_doc_map(doc_map)
            structured_constraint = self._build_structured_topic_constraint(doc_map)
        else:
            # Pure topic course: use LLM complexity assessment
            suggested_chapters, dynamic_time = await self._assess_topic_complexity(topic, model_config)
            structured_constraint = None

        # Step 2: Generate outline with dynamic params
        outline = await self._generate_outline(
            topic=topic,
            profile=profile,
            model_config=model_config,
            file_context=file_context or doc_map_context,
            organization=organization,
            suggested_chapter_count=suggested_chapters,
            dynamic_time=dynamic_time,
            structured_topic_constraint=structured_constraint
        )

        # Step 3: Determine search mode from model config
        model_key = None
        from utils.model_config import MODEL_CONFIGS
        for key, cfg in MODEL_CONFIGS.items():
            if cfg["model"] == model_config["model"]:
                model_key = key
                break
        search_mode = get_search_mode(model_key)

        # Step 4: Pre-fetch web sources for Perplexity (Groq/Llama models)
        chapter_web_sources: Dict[int, Dict] = {}
        if search_mode == "perplexity":
            from clients.perplexity_client import search_for_chapters_parallel
            chapter_web_sources = search_for_chapters_parallel(
                chapters=outline["chapters"],
                course_topic=topic
            )

        # Step 5: Pre-fetch file chunks per chapter if hybrid mode
        chapter_contexts: Dict[int, str] = {}
        if all_chunks and doc_map:
            for chapter_outline in outline["chapters"]:
                context = self._get_chunks_for_chapter(
                    chapter_outline=chapter_outline,
                    doc_map=doc_map,
                    all_chunks=all_chunks,
                    max_context_tokens=3000
                )
                chapter_contexts[chapter_outline["order"]] = context

        # Create course record (use pre-generated ID if provided, e.g. for Pinecone alignment)
        course_id = course_id or generate_uuid()
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

        self.storage.save_course(course_data)

        # Step 6: Generate chapters in batched parallel
        BATCH_SIZE = 4

        # Determine per-chapter search params
        def get_chapter_search_mode(chapter_order: int) -> str:
            if search_mode == "perplexity" and chapter_web_sources.get(chapter_order, {}).get("sources"):
                return "provided"
            return search_mode

        def get_chapter_web_sources(chapter_order: int) -> Optional[List[Dict]]:
            if search_mode == "perplexity":
                return chapter_web_sources.get(chapter_order, {}).get("sources")
            return None

        async def generate_single_chapter(i, chapter_outline):
            ch_order = chapter_outline["order"]
            ch_search_mode = get_chapter_search_mode(ch_order)
            ch_web_sources = get_chapter_web_sources(ch_order)

            # File context: from pre-fetched chunks or passed file_context
            ch_file_context = chapter_contexts.get(ch_order) if chapter_contexts else (file_context if i == 0 else None)

            # For OpenAI native search, pass use_search=True
            ch_use_search = (search_mode == "native" and model_config["provider"] == "openai")

            chapter = await self._generate_chapter(
                course_id=course_id,
                course_title=outline["title"],
                chapter_outline=chapter_outline,
                total_chapters=len(outline["chapters"]),
                profile=profile,
                model_config=model_config,
                prev_chapter_title=outline["chapters"][i - 1]["title"] if i > 0 else None,
                next_chapter_title=outline["chapters"][i + 1]["title"] if i < len(outline["chapters"]) - 1 else None,
                file_context=ch_file_context,
                search_mode=ch_search_mode,
                web_sources=ch_web_sources,
                use_search=ch_use_search
            )
            self.storage.save_chapter(course_id, chapter)
            logger.info(f"Generated chapter {i + 1}/{len(outline['chapters'])}: {chapter['title']}")
            return chapter

        all_chapters_indexed = list(enumerate(outline["chapters"]))
        chapters = []

        try:
            for batch_start in range(0, len(all_chapters_indexed), BATCH_SIZE):
                batch = all_chapters_indexed[batch_start:batch_start + BATCH_SIZE]
                batch_tasks = [generate_single_chapter(i, ch) for i, ch in batch]
                batch_results = await asyncio.gather(*batch_tasks)
                chapters.extend(batch_results)
                if batch_start + BATCH_SIZE < len(all_chapters_indexed):
                    await asyncio.sleep(1)
        except Exception as e:
            logger.error(f"Batched chapter generation failed: {e}")
            raise CourseGenerationError(f"Chapter generation failed: {e}")

        # Update course status
        course_data["status"] = CourseStatus.READY
        course_data["completed_at"] = datetime.utcnow()
        self.storage.save_course(course_data)

        return course_data

    async def _generate_full_course_from_files(
        self,
        user_id: str,
        course_id: str,
        topic: str,
        profile: LearningProfile,
        model_config: Dict[str, Any],
        source_files: List[Dict],
        doc_map: Dict[str, Any],
        doc_map_context: str,
        all_chunks: List[Dict[str, Any]],
        organization: Optional[OrganizationType] = None
    ) -> Dict[str, Any]:
        """
        Generate full course from files with per-chapter RAG retrieval.
        Chapters are generated in batches of 4 for rate-limit safety.
        """
        import asyncio

        # Calculate dynamic chapter count and time from doc map
        suggested_chapters, dynamic_time = self._calculate_chapter_count_from_doc_map(doc_map)
        structured_constraint = self._build_structured_topic_constraint(doc_map)

        # Step 1: Generate outline using structured constraints
        outline = await self._generate_outline(
            topic=topic,
            profile=profile,
            model_config=model_config,
            file_context=doc_map_context,
            organization=organization,
            suggested_chapter_count=suggested_chapters,
            dynamic_time=dynamic_time,
            structured_topic_constraint=structured_constraint
        )

        # Use pre-generated course_id (shared with Pinecone upsert)
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
            "source_type": SourceType.FILES,
            "source_files": source_files,
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

        self.storage.save_course(course_data)

        # Step 2: Retrieve relevant chunks per chapter
        chapter_contexts = {}
        for chapter_outline in outline["chapters"]:
            context = self._get_chunks_for_chapter(
                chapter_outline=chapter_outline,
                doc_map=doc_map,
                all_chunks=all_chunks,
                max_context_tokens=3000
            )
            chapter_contexts[chapter_outline["order"]] = context

        # Determine search mode for file-based chapters
        file_model_key = None
        from utils.model_config import MODEL_CONFIGS
        for key, cfg in MODEL_CONFIGS.items():
            if cfg["model"] == model_config["model"]:
                file_model_key = key
                break
        file_search_mode = get_search_mode(file_model_key)
        # For file-based courses, don't ask models to web search — content comes from files
        # Only use "none" for models without search; native models can still supplement
        if file_search_mode == "perplexity":
            file_search_mode = "none"

        # Step 3: Generate chapters in batches of 4 (rate-limit safety)
        BATCH_SIZE = 4

        async def generate_single_chapter(i, chapter_outline):
            chapter = await self._generate_chapter(
                course_id=course_id,
                course_title=outline["title"],
                chapter_outline=chapter_outline,
                total_chapters=len(outline["chapters"]),
                profile=profile,
                model_config=model_config,
                prev_chapter_title=outline["chapters"][i - 1]["title"] if i > 0 else None,
                next_chapter_title=outline["chapters"][i + 1]["title"] if i < len(outline["chapters"]) - 1 else None,
                file_context=chapter_contexts.get(chapter_outline["order"]),
                search_mode=file_search_mode
            )
            self.storage.save_chapter(course_id, chapter)
            logger.info(f"Generated chapter {i + 1}/{len(outline['chapters'])}: {chapter['title']}")
            return chapter

        all_chapters_indexed = list(enumerate(outline["chapters"]))
        chapters = []

        try:
            for batch_start in range(0, len(all_chapters_indexed), BATCH_SIZE):
                batch = all_chapters_indexed[batch_start:batch_start + BATCH_SIZE]
                batch_tasks = [generate_single_chapter(i, ch) for i, ch in batch]
                batch_results = await asyncio.gather(*batch_tasks)
                chapters.extend(batch_results)
                # Sleep between batches to avoid rate limits (skip after last batch)
                if batch_start + BATCH_SIZE < len(all_chapters_indexed):
                    await asyncio.sleep(1)
        except Exception as e:
            logger.error(f"Batched chapter generation failed: {e}")
            raise CourseGenerationError(f"Chapter generation failed: {e}")

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
        organization: Optional[OrganizationType] = None,
        suggested_chapter_count: Optional[int] = None,
        dynamic_time: Optional[int] = None,
        structured_topic_constraint: Optional[str] = None
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
            time_available=dynamic_time or 60,
            format_pref=profile.format_pref.value,
            depth_pref=profile.depth_pref.value,
            role=profile.role.value,
            learning_goal=profile.learning_goal.value,
            example_pref=profile.example_pref.value,
            file_context=file_context,
            organization_instructions=org_instructions,
            suggested_chapter_count=suggested_chapter_count,
            structured_topic_constraint=structured_topic_constraint
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
        file_context: Optional[str] = None,
        search_mode: str = "native",
        web_sources: Optional[List[Dict]] = None,
        use_search: bool = False
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
            source_material_context=file_context,
            search_mode=search_mode,
            web_sources=web_sources
        )

        response = await self._call_model(prompt, model_config, expect_json=True, use_search=use_search)
        
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
        expect_json: bool = True,
        use_search: bool = False
    ) -> Dict[str, Any]:
        """Call appropriate model based on configuration"""

        provider = model_config["provider"]

        if provider == "anthropic":
            return await self._call_claude(prompt, model_config, expect_json)
        elif provider == "openai":
            return await self._call_openai(prompt, model_config, expect_json, use_search=use_search)
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
            
            # Concatenate only text blocks — web_search returns mixed blocks
            content = ""
            for block in response.content:
                if hasattr(block, 'text'):
                    content += block.text
            
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
        expect_json: bool,
        use_search: bool = False
    ) -> Dict[str, Any]:
        """Call OpenAI API. Uses Responses API when use_search=True."""
        try:
            # Responses API path for web search
            if use_search and model_config.get("supports_search"):
                return await self._call_openai_responses(prompt, model_config, expect_json)

            # Standard Chat Completions path
            system_prompt = "You are an expert educational content creator."
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

    async def _call_openai_responses(
        self,
        prompt: str,
        model_config: Dict[str, Any],
        expect_json: bool
    ) -> Dict[str, Any]:
        """Call OpenAI Responses API with web_search_preview tool."""
        import os
        from openai import OpenAI

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        system_prompt = "You are an expert educational content creator."
        if expect_json:
            system_prompt += " You MUST respond with ONLY a valid JSON object. No markdown code blocks, no extra text. Start your response with { and end with }."

        try:
            response = client.responses.create(
                model=model_config["model"],
                tools=[{"type": "web_search_preview"}],
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ]
            )

            # Parse response.output — list of content items
            content = ""
            for item in response.output:
                if hasattr(item, 'content'):
                    # Message item with content blocks
                    for block in item.content:
                        if hasattr(block, 'text'):
                            content += block.text
                elif hasattr(item, 'text'):
                    content += item.text

            if expect_json:
                return self._extract_json(content)
            return {"content": content}

        except Exception as e:
            logger.warning(f"OpenAI Responses API failed, falling back to Chat Completions: {e}")
            # Fallback to standard Chat Completions
            return await self._call_openai(prompt, model_config, expect_json, use_search=False)
    
    async def _call_groq(
        self,
        prompt: str,
        model_config: Dict[str, Any],
        expect_json: bool
    ) -> Dict[str, Any]:
        """Call Groq API directly with response_format support and retry logic"""
        import os
        import asyncio
        from groq import Groq

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

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(**completion_params)
                answer = response.choices[0].message.content

                # Log if response was truncated
                if response.choices[0].finish_reason == "length":
                    logger.warning(f"Groq response truncated (finish_reason=length). Response length: {len(answer)}")

                if expect_json:
                    return self._extract_json(answer)
                return {"content": answer}

            except Exception as e:
                error_str = str(e).lower()
                is_rate_limit = "429" in str(e) or "rate_limit" in error_str or "rate limit" in error_str
                if is_rate_limit and attempt < max_retries - 1:
                    backoff = 2 ** (attempt + 1)  # 2s, 4s, 8s
                    logger.warning(f"Groq rate limit hit (attempt {attempt + 1}/{max_retries}). Retrying in {backoff}s...")
                    await asyncio.sleep(backoff)
                    continue
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
        Uses local Chonkie RecursiveChunker (no API key needed).
        Returns list of chunks with index IDs.
        """
        from chonkie import RecursiveChunker
        from chonkie.tokenizer import AutoTokenizer

        tokenizer = AutoTokenizer("cl100k_base")
        chunker = RecursiveChunker(
            tokenizer=tokenizer,
            chunk_size=512,
            min_characters_per_chunk=50
        )

        chunk_objects = chunker.chunk(extracted_text)

        chunks = []
        for i, chunk_obj in enumerate(chunk_objects):
            chunks.append({
                "text": chunk_obj.text,
                "start_index": chunk_obj.start_index,
                "end_index": chunk_obj.end_index,
                "token_count": chunk_obj.token_count,
                "chunk_index": i,
                "source_file": filename
            })

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

    async def _embed_and_upsert_chunks(
        self,
        course_id: str,
        all_chunks: List[Dict[str, Any]],
        source_files: List[Dict]
    ) -> None:
        """
        Embed course chunks and upsert to Pinecone for later Q&A retrieval.
        Uses existing Chonkie multimodal embeddings + Pinecone upsert pattern.
        """
        from clients.chonkie_client import embed_chunks_multimodal
        from clients.pinecone_client import upsert_vectors

        if not all_chunks:
            logger.warning("No chunks to embed/upsert")
            return

        # Embed all chunks using multimodal embeddings (Jina CLIP-v2, 1024 dims)
        logger.info(f"Embedding {len(all_chunks)} course chunks...")
        embeddings = embed_chunks_multimodal(all_chunks, batch_size=32)

        if not embeddings or len(embeddings) != len(all_chunks):
            logger.error(f"Embedding count mismatch: {len(embeddings)} embeddings for {len(all_chunks)} chunks")
            return

        # Tag chunks with course metadata before upserting
        tagged_chunks = []
        for chunk in all_chunks:
            tagged = dict(chunk)
            tagged["course_id"] = course_id
            tagged["content_type"] = "course_source_material"
            tagged_chunks.append(tagged)

        # Upsert to Pinecone with course_id as the doc_id
        upsert_vectors(
            doc_id=course_id,
            space_id=f"course_{course_id}",
            embeddings=embeddings,
            chunks=tagged_chunks,
            source_file=", ".join([f["filename"] for f in source_files])
        )

        logger.info(f"Upserted {len(all_chunks)} chunks to Pinecone for course {course_id}")

    async def _analyze_multi_files(
        self,
        files: List[Dict[str, Any]],
        requested_org: str,
        model_config: Optional[Dict[str, Any]] = None
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
        
        # Auto-detect: Use LLM to analyze
        prompt = build_multi_file_analysis_prompt(topics)
        if model_config is None:
            model_config = ModelConfig.get_config(None)
        
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

    def _calculate_chapter_count_from_doc_map(
        self,
        doc_map: Dict[str, Any]
    ) -> tuple:
        """
        Calculate suggested chapter count and dynamic time from doc map topics.
        Returns (suggested_chapter_count, dynamic_time_minutes).
        """
        topics = doc_map.get("topics", [])
        core_topics = sum(1 for t in topics if t.get("importance") == "core")
        supporting_topics = sum(1 for t in topics if t.get("importance") == "supporting")

        # Chapter count: core topics + half the supporting topics, clamped 3-10
        suggested_chapters = max(3, min(10, core_topics + (supporting_topics + 1) // 2))

        # Dynamic time: core topics get 10 min each, supporting get 5 min, clamped 30-180
        dynamic_time = core_topics * 10 + supporting_topics * 5
        dynamic_time = max(30, min(180, dynamic_time))

        logger.info(
            f"Doc map analysis: {core_topics} core, {supporting_topics} supporting topics "
            f"-> {suggested_chapters} chapters, {dynamic_time} min"
        )
        return suggested_chapters, dynamic_time

    def _build_structured_topic_constraint(
        self,
        doc_map: Dict[str, Any]
    ) -> str:
        """
        Convert doc map into a structured prompt constraint that forces
        the LLM to cover all topics.
        """
        topics = doc_map.get("topics", [])

        core_lines = []
        supporting_lines = []
        supplementary_lines = []

        for t in topics:
            importance = t.get("importance", "supplementary")
            section_count = len(t.get("chunk_indices", []))
            line = f"  - {t['topic']}: {t.get('description', '')} [{section_count} sections]"
            if importance == "core":
                core_lines.append(line)
            elif importance == "supporting":
                supporting_lines.append(line)
            else:
                supplementary_lines.append(line)

        parts = []
        if core_lines:
            parts.append("CORE TOPICS (must be primary chapter focus):\n" + "\n".join(core_lines))
        if supporting_lines:
            parts.append("SUPPORTING TOPICS (must be sub-sections):\n" + "\n".join(supporting_lines))
        if supplementary_lines:
            parts.append("SUPPLEMENTARY TOPICS (include where relevant):\n" + "\n".join(supplementary_lines))

        parts.append(
            "RULES:\n"
            "- Every CORE topic MUST appear as a chapter or major section\n"
            "- Every SUPPORTING topic MUST appear as a sub-section\n"
            "- Do NOT skip any core or supporting topic\n"
            "- Supplementary topics should be woven in where they fit naturally"
        )

        return "\n\n".join(parts)

    async def _generate_single_file_course(
        self,
        user_id: str,
        file_data: Dict[str, Any],
        profile: 'LearningProfile',
        model_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate a course from a single file (used by separate_courses path).
        Handles chunking, embedding, doc map, and full course generation.
        """
        # Chunk the single file
        file_chunks = self._chunk_document_for_course(
            extracted_text=file_data["extracted_text"],
            filename=file_data["filename"]
        )

        # Generate course_id, embed+upsert
        course_id = generate_uuid()
        await self._embed_and_upsert_chunks(
            course_id=course_id,
            all_chunks=file_chunks,
            source_files=[{"filename": file_data["filename"]}]
        )

        # Build document map
        doc_map = await self._build_document_map(file_chunks, model_config)

        # Build doc_map_context (flat text fallback, used internally)
        doc_map_context = "DOCUMENT MAP (topics found in uploaded material):\n"
        for topic in doc_map.get("topics", []):
            doc_map_context += f"- {topic['topic']}: {topic.get('description', '')} [{len(topic.get('chunk_indices', []))} sections]\n"

        source_files = [{"file_id": generate_uuid(), "filename": file_data["filename"], "extracted_topic": file_data["topic"]}]

        course = await self._generate_full_course_from_files(
            user_id=user_id,
            course_id=course_id,
            topic=file_data["topic"],
            profile=profile,
            model_config=model_config,
            source_files=source_files,
            doc_map=doc_map,
            doc_map_context=doc_map_context,
            all_chunks=file_chunks,
            organization=None
        )

        return course

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

    async def query_course(
        self,
        course_id: str,
        question: str,
        model: Optional[str] = None,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Answer a question about a course using its source material from Pinecone.
        Searches course chunks, reranks, and generates a grounded answer.
        """
        from clients.chonkie_client import embed_query_multimodal
        from clients.pinecone_client import hybrid_search, rerank_results

        model_config = ModelConfig.get_config(model)

        # Get course for context
        course = self.get_course(course_id)
        if not course:
            raise ValueError(f"Course not found: {course_id}")

        # Embed the question
        query_result = embed_query_multimodal(question)
        query_emb = query_result["embedding"]

        # Search Pinecone filtered by course namespace
        search_results = hybrid_search(
            query_emb=query_emb,
            space_id=f"course_{course_id}",
            top_k=top_k * 2  # Over-fetch for reranking
        )

        # Rerank results
        if search_results:
            reranked = rerank_results(
                query=question,
                hits=search_results,
                top_n=top_k
            )
        else:
            reranked = []

        # Build context from retrieved chunks
        context_parts = []
        for i, result in enumerate(reranked):
            text = result.get("metadata", {}).get("text", "")
            source = result.get("metadata", {}).get("source_file", "")
            context_parts.append(f"[Source {i+1}] ({source})\n{text}")

        context = "\n\n---\n\n".join(context_parts)

        # Generate answer
        prompt = f"""You are a helpful course assistant. Answer the student's question based ONLY on the source material provided.

COURSE: {course.get('title', '')}

SOURCE MATERIAL:
{context}

STUDENT QUESTION: {question}

Instructions:
- Answer based ONLY on the source material above
- If the source material doesn't contain the answer, say so clearly
- Reference specific sources using [Source N] citations
- Keep the answer concise and educational"""

        response = await self._call_model(prompt, model_config, expect_json=False)

        return {
            "course_id": course_id,
            "question": question,
            "answer": response.get("content", ""),
            "sources_used": len(reranked),
            "source_excerpts": [
                {
                    "text": r.get("metadata", {}).get("text", "")[:200],
                    "file": r.get("metadata", {}).get("source_file", "")
                }
                for r in reranked
            ]
        }
