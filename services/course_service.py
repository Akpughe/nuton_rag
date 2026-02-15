"""
Core course generation service.
Orchestrates outline generation, chapter creation, and storage.
Following DRY principle - modular, reusable components.
"""

import json
import time
import logging
from typing import Dict, Any, Optional, List, AsyncGenerator
from datetime import datetime

from models.course_models import (
    Course, Chapter, LearningProfile, CourseStatus, SourceType,
    PersonalizationParams, SourceFile, OrganizationType
)
from utils.file_storage import (
    CourseStorage, LearningProfileStorage, GenerationLogger,
    StudyGuideStorage, FlashcardStorage, ExamStorage, ExamAttemptStorage, ChatStorage,
    generate_uuid, generate_slug
)
from utils.model_config import ModelConfig, estimate_course_cost, get_search_mode
from prompts.course_prompts import (
    build_outline_generation_prompt,
    build_chapter_content_prompt,
    build_topic_extraction_prompt,
    build_multi_file_analysis_prompt,
    build_study_guide_prompt,
    build_course_flashcards_prompt,
    build_final_exam_prompt,
    build_theory_grading_prompt,
    build_course_chat_prompt,
    build_topic_course_chat_prompt,
    build_course_notes_intro_prompt,
    build_course_notes_section_prompt,
    build_course_notes_conclusion_prompt,
    build_notes_flashcards_prompt,
    build_notes_quiz_prompt
)

# Import existing clients
import clients.openai_client as openai_client
from clients.groq_client import generate_answer

from utils.exceptions import (
    NutonError, NotFoundError, ValidationError, GenerationError,
    OutlineGenerationError, ChapterGenerationError, StorageError,
)

# Backward-compat alias used by routes
CourseGenerationError = GenerationError

logger = logging.getLogger(__name__)


class CourseService:
    """Main service for course generation"""
    
    def __init__(self):
        self.storage = CourseStorage()
        self.profile_storage = LearningProfileStorage()
        self.logger = GenerationLogger()
        self.study_guide_storage = StudyGuideStorage()
        self.flashcard_storage = FlashcardStorage()
        self.exam_storage = ExamStorage()
        self.exam_attempt_storage = ExamAttemptStorage()
        self.chat_storage = ChatStorage()
    
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
                "estimated_cost": estimate_course_cost(model or "llama-4-scout", course["total_chapters"])
            })
            
            return {
                "course_id": course["id"],
                "status": CourseStatus.READY,
                "course": course,
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

    async def resume_course(
        self,
        course_id: str,
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Resume a partially-generated course by generating only missing/errored chapters.
        Blocking operation - returns full course when all gaps are filled.
        """
        import asyncio
        start_time = time.time()

        # ── 1. Load course and validate ──────────────────────────────
        course = self.storage.get_course(course_id)
        if not course:
            raise NotFoundError(f"Course {course_id} not found")

        outline = course.get("outline")
        if not outline or not outline.get("chapters"):
            raise ValidationError(f"Course {course_id} has no outline — cannot resume")

        existing_chapters = course.get("chapters", [])
        ready_orders = {
            ch["order"] for ch in existing_chapters if ch.get("status") == "ready"
        }
        all_outline_orders = {ch["order"] for ch in outline["chapters"]}
        missing_orders = all_outline_orders - ready_orders

        # Check extras status before deciding if truly complete
        needs_study_guide = self.study_guide_storage.get_study_guide(course_id) is None
        needs_flashcards = self.flashcard_storage.get_flashcards(course_id) is None

        # Early return if already complete (chapters + extras)
        if not missing_orders and not needs_study_guide and not needs_flashcards:
            logger.info(f"Course {course_id} is already complete ({len(ready_orders)} chapters)")
            return {
                "course_id": course_id,
                "status": course.get("status", CourseStatus.READY),
                "course": course,
                "resume_summary": {
                    "already_complete": True,
                    "chapters_total": len(all_outline_orders),
                    "chapters_existed": len(all_outline_orders),
                    "chapters_generated": 0,
                    "chapters_failed": [],
                    "study_guide_generated": False,
                    "flashcards_generated": False,
                },
            }

        logger.info(
            f"Resuming course {course_id}: {len(missing_orders)} missing chapters "
            f"out of {len(all_outline_orders)} total"
        )

        # ── 2. Reconstruct context ──────────────────────────────────
        # Profile
        user_id = course.get("user_id", "unknown")
        personalization = course.get("personalization_params", {})
        profile = self._get_or_create_profile(user_id, personalization)

        # Model config
        model_config = ModelConfig.get_config(model or course.get("model_used"))

        # Search mode
        model_key = None
        from utils.model_config import MODEL_CONFIGS
        for key, cfg in MODEL_CONFIGS.items():
            if cfg["model"] == model_config["model"]:
                model_key = key
                break
        search_mode = get_search_mode(model_key)

        # File contexts from Pinecone (for file-based courses)
        missing_outlines = [
            ch for ch in outline["chapters"] if ch["order"] in missing_orders
        ]
        chapter_contexts: Dict[int, str] = {}
        source_type = course.get("source_type", "")
        if source_type in ("files", "mixed"):
            chapter_contexts = self._retrieve_chunks_for_resume(course_id, missing_outlines)

        # Web sources for Perplexity models
        chapter_web_sources: Dict[int, Dict] = {}
        if search_mode == "perplexity":
            from clients.perplexity_client import search_for_chapters_parallel
            chapter_web_sources = search_for_chapters_parallel(
                chapters=missing_outlines,
                course_topic=course.get("topic", course.get("title", ""))
            )

        # ── 3. Generate missing chapters in batched parallel ────────
        course_title = outline["title"]
        total_chapters = len(outline["chapters"])
        all_chapter_outlines = {ch["order"]: ch for ch in outline["chapters"]}

        # Update status to generating
        course["status"] = CourseStatus.GENERATING
        self.storage.save_course(course)

        def get_chapter_search_mode(chapter_order: int) -> str:
            if search_mode == "perplexity" and chapter_web_sources.get(chapter_order, {}).get("sources"):
                return "provided"
            return search_mode

        def get_chapter_web_sources_fn(chapter_order: int) -> Optional[List[Dict]]:
            if search_mode == "perplexity":
                return chapter_web_sources.get(chapter_order, {}).get("sources")
            return None

        async def generate_single_chapter(chapter_outline):
            ch_order = chapter_outline["order"]
            ch_search_mode = get_chapter_search_mode(ch_order)
            ch_web_sources = get_chapter_web_sources_fn(ch_order)
            ch_file_context = chapter_contexts.get(ch_order)
            ch_use_search = (search_mode == "native" and model_config["provider"] == "openai")

            prev_order = ch_order - 1
            next_order = ch_order + 1
            prev_title = all_chapter_outlines[prev_order]["title"] if prev_order in all_chapter_outlines else None
            next_title = all_chapter_outlines[next_order]["title"] if next_order in all_chapter_outlines else None

            chapter = await self._generate_chapter(
                course_id=course_id,
                course_title=course_title,
                chapter_outline=chapter_outline,
                total_chapters=total_chapters,
                profile=profile,
                model_config=model_config,
                prev_chapter_title=prev_title,
                next_chapter_title=next_title,
                file_context=ch_file_context,
                search_mode=ch_search_mode,
                web_sources=ch_web_sources,
                use_search=ch_use_search,
            )
            self.storage.save_chapter(course_id, chapter)
            logger.info(f"[resume] Generated chapter {ch_order}/{total_chapters}: {chapter['title']}")
            return chapter

        BATCH_SIZE = 4
        generated_chapters = []
        failed_orders = []

        try:
            sorted_missing = sorted(missing_outlines, key=lambda ch: ch["order"])
            for batch_start in range(0, len(sorted_missing), BATCH_SIZE):
                batch = sorted_missing[batch_start:batch_start + BATCH_SIZE]
                batch_tasks = [generate_single_chapter(ch) for ch in batch]
                results = await asyncio.gather(*batch_tasks, return_exceptions=True)

                for ch_outline, result in zip(batch, results):
                    if isinstance(result, Exception):
                        failed_orders.append(ch_outline["order"])
                        logger.error(f"[resume] Chapter {ch_outline['order']} generation failed: {result}")
                    else:
                        generated_chapters.append(result)

                if batch_start + BATCH_SIZE < len(sorted_missing):
                    await asyncio.sleep(1)
        except Exception as e:
            logger.error(f"[resume] Batched chapter generation failed: {e}")
            raise GenerationError(f"Resume chapter generation failed: {e}")

        # ── 4. Generate study guide & flashcards if missing ─────────
        sg_generated = False
        fc_generated = False
        try:
            refreshed_course = self.storage.get_course(course_id)
            all_chapters = refreshed_course.get("chapters", [])
            topic = course.get("topic", course.get("title", ""))

            extras_tasks = []
            extras_labels = []
            if needs_study_guide:
                extras_tasks.append(
                    self._generate_study_guide(course_id, all_chapters, profile, model_config, topic)
                )
                extras_labels.append("study_guide")
            if needs_flashcards:
                extras_tasks.append(
                    self._generate_course_flashcards(course_id, all_chapters, profile, model_config)
                )
                extras_labels.append("flashcards")

            if extras_tasks:
                extras_results = await asyncio.gather(*extras_tasks, return_exceptions=True)
                for label, r in zip(extras_labels, extras_results):
                    if isinstance(r, Exception):
                        logger.warning(f"[resume] Extras generation failed (non-fatal): {r}")
                    elif label == "study_guide":
                        sg_generated = True
                    elif label == "flashcards":
                        fc_generated = True
        except Exception as e:
            logger.warning(f"[resume] Study guide / flashcard generation error (non-fatal): {e}")

        # ── 5. Finalize ─────────────────────────────────────────────
        if not failed_orders:
            self.storage.save_course({
                "id": course_id,
                "status": CourseStatus.READY,
                "completed_at": datetime.utcnow(),
            })
            course["status"] = CourseStatus.READY
        else:
            logger.warning(
                f"[resume] {len(failed_orders)} chapters failed (orders: {failed_orders}), "
                f"leaving course {course_id} in GENERATING status"
            )

        generation_time = round(time.time() - start_time, 2)

        # Reload final course state
        final_course = self.storage.get_course(course_id)

        self.logger.log_generation({
            "type": "resume_course",
            "user_id": user_id,
            "course_id": course_id,
            "model": model_config["model"],
            "chapters_resumed": len(generated_chapters),
            "total_chapters": total_chapters,
            "generation_time": generation_time,
            "status": "success",
        })

        return {
            "course_id": course_id,
            "status": course["status"],
            "course": final_course,
            "generation_time_seconds": generation_time,
            "resume_summary": {
                "already_complete": False,
                "chapters_total": total_chapters,
                "chapters_existed": len(ready_orders),
                "chapters_generated": len(generated_chapters),
                "chapters_failed": failed_orders,
                "study_guide_generated": sg_generated,
                "flashcards_generated": fc_generated,
            },
        }

    def _retrieve_chunks_for_resume(
        self,
        course_id: str,
        missing_chapter_outlines: List[Dict[str, Any]],
        max_context_tokens: int = 3000
    ) -> Dict[int, str]:
        """
        Query Pinecone by course_id to retrieve stored chunks,
        then map them to missing chapter outlines using keyword overlap.

        Returns:
            Dict mapping chapter order -> context string.
            Empty dict if Pinecone has no data for this course.
        """
        try:
            from clients.pinecone_client import fetch_all_document_chunks
        except ImportError:
            logger.warning("[resume] Pinecone client not available, skipping chunk retrieval")
            return {}

        try:
            chunks = fetch_all_document_chunks(
                document_id=course_id,
                space_id=f"course_{course_id}",
                max_chunks=500,
                enable_gap_filling=False,
            )
        except Exception as e:
            logger.warning(f"[resume] Failed to fetch chunks from Pinecone: {e}")
            return {}

        if not chunks:
            logger.info(f"[resume] No Pinecone chunks found for course {course_id} (topic-only course)")
            return {}

        logger.info(f"[resume] Retrieved {len(chunks)} chunks from Pinecone for course {course_id}")

        # Map chunks to missing chapters using keyword overlap
        stop_words = {
            "the", "a", "an", "and", "or", "to", "of", "in", "for", "is",
            "on", "with", "this", "that", "be", "can", "will", "are", "after",
            "students", "chapter", "learn", "understand",
        }

        chapter_contexts: Dict[int, str] = {}
        for chapter_outline in missing_chapter_outlines:
            ch_order = chapter_outline["order"]
            ch_title = chapter_outline["title"].lower()
            ch_objectives = " ".join(chapter_outline.get("objectives", [])).lower()
            ch_concepts = " ".join(chapter_outline.get("key_concepts", [])).lower()
            chapter_text = f"{ch_title} {ch_objectives} {ch_concepts}"
            chapter_words = set(chapter_text.split()) - stop_words

            # Score each chunk by keyword overlap with this chapter
            scored_chunks = []
            for chunk in chunks:
                chunk_text = chunk.get("metadata", {}).get("text", "")
                if not chunk_text:
                    continue
                chunk_words = set(chunk_text.lower().split()[:200]) - stop_words
                overlap = len(chapter_words & chunk_words)
                if overlap > 0:
                    scored_chunks.append((overlap, chunk_text))

            scored_chunks.sort(key=lambda x: x[0], reverse=True)

            # Build context string, respecting token limit
            context_parts = []
            token_count = 0
            for _, text in scored_chunks:
                chunk_tokens = len(text) // 4  # rough estimate: 1 token ~ 4 chars
                if token_count + chunk_tokens > max_context_tokens:
                    break
                context_parts.append(text)
                token_count += chunk_tokens

            if context_parts:
                chapter_contexts[ch_order] = "\n\n---\n\n".join(context_parts)

        logger.info(
            f"[resume] Mapped chunks to {len(chapter_contexts)}/{len(missing_chapter_outlines)} chapters"
        )
        return chapter_contexts

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
                source_files=[{
                    "file_id": generate_uuid(),
                    "filename": f["filename"],
                    "extracted_topic": f["topic"],
                    "source_url": f.get("source_url"),
                    "source_type": f.get("source_type", "pdf"),
                } for f in files],
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
            {
                "file_id": generate_uuid(),
                "filename": f["filename"],
                "extracted_topic": f.get("topic", ""),
                "source_url": f.get("source_url"),
                "source_type": f.get("source_type", "pdf"),
            }
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
        """Non-streaming wrapper — consumes generator, returns final course_data."""
        result_course_id = None
        async for event in self._generate_full_course_stream(
            user_id=user_id, topic=topic, source_type=source_type,
            profile=profile, model_config=model_config,
            source_files=source_files, file_context=file_context,
            organization=organization, doc_map=doc_map,
            doc_map_context=doc_map_context, all_chunks=all_chunks,
            course_id=course_id
        ):
            if event["type"] == "error":
                raise CourseGenerationError(event["message"])
            if event["type"] == "outline_ready":
                result_course_id = event["course_id"]
        return self.storage.get_course(result_course_id)

    async def _generate_full_course_stream(
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
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Internal async generator: Generate complete course with all chapters.
        Yields outline_ready and chapter_ready events as they complete.
        """
        import asyncio

        # Step 1: Determine chapter count and time
        if doc_map:
            suggested_chapters, dynamic_time = self._calculate_chapter_count_from_doc_map(doc_map)
            structured_constraint = self._build_structured_topic_constraint(doc_map)
        else:
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

        # Create course record
        course_id = course_id or generate_uuid()
        personalization = PersonalizationParams(
            expertise=profile.expertise,
            format_pref=profile.format_pref,
            depth_pref=profile.depth_pref,
            role=profile.role,
            learning_goal=profile.learning_goal,
            example_pref=profile.example_pref
        )

        slug = generate_slug(outline["title"])

        course_data = {
            "id": course_id,
            "user_id": user_id,
            "slug": slug,
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

        # Yield outline_ready event
        yield {
            "type": "outline_ready",
            "course_id": course_id,
            "slug": slug,
            "title": outline["title"],
            "total_chapters": len(outline["chapters"]),
            "estimated_time": outline["total_estimated_time"],
            "outline": outline
        }

        # Step 6: Generate chapters in batched parallel
        BATCH_SIZE = 4

        def get_chapter_search_mode(chapter_order: int) -> str:
            if search_mode == "perplexity" and chapter_web_sources.get(chapter_order, {}).get("sources"):
                return "provided"
            return search_mode

        def get_chapter_web_sources_fn(chapter_order: int) -> Optional[List[Dict]]:
            if search_mode == "perplexity":
                return chapter_web_sources.get(chapter_order, {}).get("sources")
            return None

        async def generate_single_chapter(i, chapter_outline):
            ch_order = chapter_outline["order"]
            ch_search_mode = get_chapter_search_mode(ch_order)
            ch_web_sources = get_chapter_web_sources_fn(ch_order)
            ch_file_context = chapter_contexts.get(ch_order) if chapter_contexts else (file_context if i == 0 else None)
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

        try:
            for batch_start in range(0, len(all_chapters_indexed), BATCH_SIZE):
                batch = all_chapters_indexed[batch_start:batch_start + BATCH_SIZE]
                batch_tasks = [generate_single_chapter(i, ch) for i, ch in batch]
                for future in asyncio.as_completed(batch_tasks):
                    chapter = await future
                    yield {
                        "type": "chapter_ready",
                        "course_id": course_id,
                        "chapter_order": chapter["order"],
                        "chapter_title": chapter["title"],
                        "total_chapters": len(outline["chapters"]),
                        "chapter": chapter
                    }
                if batch_start + BATCH_SIZE < len(all_chapters_indexed):
                    await asyncio.sleep(1)
        except Exception as e:
            logger.error(f"Batched chapter generation failed: {e}")
            yield {
                "type": "error",
                "error": "CHAPTER_GENERATION_FAILED",
                "message": f"Chapter generation failed: {e}",
                "status_code": 500,
                "course_id": course_id,
                "phase": "chapter_generation",
                "context": None,
            }
            return

        # Generate study guide + flashcards in parallel (after all chapters)
        try:
            chapters_for_extras = self.storage.get_course(course_id).get("chapters", [])
            sg_task = self._generate_study_guide(course_id, chapters_for_extras, profile, model_config, topic)
            fc_task = self._generate_course_flashcards(course_id, chapters_for_extras, profile, model_config)
            sg_result, fc_result = await asyncio.gather(sg_task, fc_task, return_exceptions=True)

            if not isinstance(sg_result, Exception) and sg_result:
                yield {"type": "study_guide_ready", "course_id": course_id}
            else:
                logger.warning(f"Study guide generation failed: {sg_result}")

            if not isinstance(fc_result, Exception) and fc_result:
                yield {"type": "flashcards_ready", "course_id": course_id}
            else:
                logger.warning(f"Flashcard generation failed: {fc_result}")
        except Exception as e:
            logger.warning(f"Study guide / flashcard generation error (non-fatal): {e}")

        # Update course status
        course_data["status"] = CourseStatus.READY
        course_data["completed_at"] = datetime.utcnow()
        self.storage.save_course(course_data)

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
        """Non-streaming wrapper — consumes generator, returns final course_data."""
        result_course_id = None
        async for event in self._generate_full_course_from_files_stream(
            user_id=user_id, course_id=course_id, topic=topic,
            profile=profile, model_config=model_config,
            source_files=source_files, doc_map=doc_map,
            doc_map_context=doc_map_context, all_chunks=all_chunks,
            organization=organization
        ):
            if event["type"] == "error":
                raise CourseGenerationError(event["message"])
            if event["type"] == "outline_ready":
                result_course_id = event["course_id"]
        return self.storage.get_course(result_course_id)

    async def _generate_full_course_from_files_stream(
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
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Async generator: Generate full course from files with per-chapter RAG retrieval.
        Yields outline_ready and chapter_ready events.
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
            expertise=profile.expertise,
            format_pref=profile.format_pref,
            depth_pref=profile.depth_pref,
            role=profile.role,
            learning_goal=profile.learning_goal,
            example_pref=profile.example_pref
        )

        slug = generate_slug(outline["title"])

        course_data = {
            "id": course_id,
            "user_id": user_id,
            "slug": slug,
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

        # Yield outline_ready event
        yield {
            "type": "outline_ready",
            "course_id": course_id,
            "slug": slug,
            "title": outline["title"],
            "total_chapters": len(outline["chapters"]),
            "estimated_time": outline["total_estimated_time"],
            "outline": outline
        }

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

        try:
            for batch_start in range(0, len(all_chapters_indexed), BATCH_SIZE):
                batch = all_chapters_indexed[batch_start:batch_start + BATCH_SIZE]
                batch_tasks = [generate_single_chapter(i, ch) for i, ch in batch]
                for future in asyncio.as_completed(batch_tasks):
                    chapter = await future
                    yield {
                        "type": "chapter_ready",
                        "course_id": course_id,
                        "chapter_order": chapter["order"],
                        "chapter_title": chapter["title"],
                        "total_chapters": len(outline["chapters"]),
                        "chapter": chapter
                    }
                if batch_start + BATCH_SIZE < len(all_chapters_indexed):
                    await asyncio.sleep(1)
        except Exception as e:
            logger.error(f"Batched chapter generation failed: {e}")
            yield {
                "type": "error",
                "error": "CHAPTER_GENERATION_FAILED",
                "message": f"Chapter generation failed: {e}",
                "status_code": 500,
                "course_id": course_id,
                "phase": "chapter_generation",
                "context": None,
            }
            return

        # Generate study guide + flashcards in parallel (after all chapters)
        try:
            chapters_for_extras = self.storage.get_course(course_id).get("chapters", [])
            sg_task = self._generate_study_guide(course_id, chapters_for_extras, profile, model_config, topic)
            fc_task = self._generate_course_flashcards(course_id, chapters_for_extras, profile, model_config)
            sg_result, fc_result = await asyncio.gather(sg_task, fc_task, return_exceptions=True)

            if not isinstance(sg_result, Exception) and sg_result:
                yield {"type": "study_guide_ready", "course_id": course_id}
            else:
                logger.warning(f"Study guide generation failed: {sg_result}")

            if not isinstance(fc_result, Exception) and fc_result:
                yield {"type": "flashcards_ready", "course_id": course_id}
            else:
                logger.warning(f"Flashcard generation failed: {fc_result}")
        except Exception as e:
            logger.warning(f"Study guide / flashcard generation error (non-fatal): {e}")

        # Update course status
        course_data["status"] = CourseStatus.READY
        course_data["completed_at"] = datetime.utcnow()
        self.storage.save_course(course_data)

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
            expertise=profile.expertise.value,
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
            expertise=profile.expertise.value,
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
        use_search: bool = False,
        max_tokens_override: Optional[int] = None
    ) -> Dict[str, Any]:
        """Call appropriate model based on configuration"""

        if max_tokens_override:
            model_config = {**model_config, "max_tokens": max_tokens_override}

        provider = model_config["provider"]

        if provider == "anthropic":
            return await self._call_claude(prompt, model_config, expect_json)
        elif provider == "openai":
            return await self._call_openai(prompt, model_config, expect_json, use_search=use_search)
        elif provider == "groq":
            return await self._call_groq(prompt, model_config, expect_json)
        else:
            raise ValidationError(f"Unknown provider: {provider}", error_code="INVALID_MODEL")
    
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
            tools = [{"type": "web_search_20250305", "name": "web_search"}] if model_config.get("supports_search") else None
            
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

            system_prompt = "You are an expert educational content creator."
            if expect_json:
                system_prompt += " You MUST respond with ONLY a valid JSON object. No markdown code blocks, no extra text. Start your response with { and end with }."

            # JSON path: call OpenAI directly with response_format for reliability
            if expect_json:
                import os
                from openai import OpenAI
                client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

                params = {
                    "model": model_config["model"],
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    "response_format": {"type": "json_object"},
                }
                if model_config.get("max_tokens"):
                    params["max_tokens"] = model_config["max_tokens"]

                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        response = client.chat.completions.create(**params)
                        answer = response.choices[0].message.content
                        return self._extract_json(answer)
                    except GenerationError:
                        if attempt < max_retries - 1:
                            import asyncio
                            backoff = 2 ** (attempt + 1)
                            logger.warning(f"OpenAI JSON parse failed (attempt {attempt + 1}/{max_retries}). Retrying in {backoff}s...")
                            await asyncio.sleep(backoff)
                            continue
                        raise

            # Non-JSON path: use existing client wrapper
            response = openai_client.generate_answer(
                query=prompt,
                context_chunks=[],
                system_prompt=system_prompt,
                model=model_config["model"],
                max_tokens=model_config.get("max_tokens")
            )

            if isinstance(response, tuple):
                return {"content": response[0]}
            return response

        except GenerationError:
            raise
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
            system_prompt += " Always respond with valid JSON. Ensure all string values have properly escaped inner quotes."

        temperature = model_config.get("temperature", 0.7)
        completion_params = {
            "model": model_config["model"],
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": model_config.get("max_tokens", 8192),
            "temperature": temperature,
            "stream": False
        }

        if expect_json:
            completion_params["response_format"] = {"type": "json_object"}

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(**completion_params)
                answer = response.choices[0].message.content

                # Truncated response — JSON is guaranteed invalid
                if response.choices[0].finish_reason == "length":
                    if attempt < max_retries - 1:
                        completion_params["temperature"] = max(0.3, temperature - 0.2)
                        logger.warning(f"Groq response truncated (attempt {attempt + 1}/{max_retries}). Retrying with lower temperature...")
                        await asyncio.sleep(2 ** (attempt + 1))
                        continue
                    logger.warning(f"Groq response truncated after all retries. Response length: {len(answer)}")

                if expect_json:
                    return self._extract_json(answer)
                return {"content": answer}

            except GenerationError:
                # _extract_json failed to parse — retry with backoff
                if attempt < max_retries - 1:
                    backoff = 2 ** (attempt + 1)
                    logger.warning(f"Groq JSON parse failed (attempt {attempt + 1}/{max_retries}). Retrying in {backoff}s...")
                    await asyncio.sleep(backoff)
                    continue
                raise

            except Exception as e:
                error_str = str(e).lower()
                is_rate_limit = "429" in str(e) or "rate_limit" in error_str or "rate limit" in error_str
                is_json_fail = "json_validate_failed" in error_str or "json_validate" in error_str

                if (is_rate_limit or is_json_fail) and attempt < max_retries - 1:
                    backoff = 2 ** (attempt + 1)
                    reason = "rate limit" if is_rate_limit else "json_validate_failed"
                    logger.warning(f"Groq {reason} (attempt {attempt + 1}/{max_retries}). Retrying in {backoff}s...")
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
                except (json.JSONDecodeError, ValueError):
                    continue
        
        logger.error(f"Could not extract JSON from: {text[:500]}")
        raise GenerationError("Failed to parse JSON response", error_code="GENERATION_FAILED")
    
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
            expertise=context.get("expertise", "beginner"),
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
            raise GenerationError("Failed to generate document map", error_code="GENERATION_FAILED")

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
        """Non-streaming wrapper — consumes generator, returns final course_data."""
        result_course_id = None
        async for event in self._generate_single_file_course_stream(
            user_id=user_id, file_data=file_data,
            profile=profile, model_config=model_config
        ):
            if event["type"] == "error":
                raise CourseGenerationError(event["message"])
            if event["type"] == "outline_ready":
                result_course_id = event["course_id"]
        return self.storage.get_course(result_course_id)

    async def _generate_single_file_course_stream(
        self,
        user_id: str,
        file_data: Dict[str, Any],
        profile: 'LearningProfile',
        model_config: Dict[str, Any]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Async generator for single-file course (used by separate_courses path).
        Forwards events from _generate_full_course_from_files_stream.
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

        # Build doc_map_context
        doc_map_context = "DOCUMENT MAP (topics found in uploaded material):\n"
        for topic in doc_map.get("topics", []):
            doc_map_context += f"- {topic['topic']}: {topic.get('description', '')} [{len(topic.get('chunk_indices', []))} sections]\n"

        source_files = [{
            "file_id": generate_uuid(),
            "filename": file_data["filename"],
            "extracted_topic": file_data["topic"],
            "source_url": file_data.get("source_url"),
            "source_type": file_data.get("source_type", "pdf"),
        }]

        async for event in self._generate_full_course_from_files_stream(
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
        ):
            yield event

    # Streaming public methods

    async def create_course_from_topic_stream(
        self,
        user_id: str,
        topic: str,
        context: Dict[str, Any],
        model: Optional[str] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Streaming version of create_course_from_topic. Yields SSE events."""
        start_time = time.time()
        profile = self._get_or_create_profile(user_id, context)
        model_config = ModelConfig.get_config(model)
        course_id = None
        total_chapters = None

        try:
            async for event in self._generate_full_course_stream(
                user_id=user_id,
                topic=topic,
                source_type=SourceType.TOPIC,
                profile=profile,
                model_config=model_config,
                source_files=None
            ):
                if event["type"] == "outline_ready":
                    course_id = event["course_id"]
                    total_chapters = event["total_chapters"]
                yield event
                if event["type"] == "error":
                    return

            generation_time = round(time.time() - start_time, 2)
            yield {
                "type": "course_complete",
                "course_id": course_id,
                "status": CourseStatus.READY,
                "total_chapters": total_chapters,
                "generation_time_seconds": generation_time
            }
        except Exception as e:
            logger.error(f"Streaming course generation failed: {e}")
            yield {
                "type": "error",
                "error": "GENERATION_FAILED",
                "message": str(e),
                "status_code": 500,
                "course_id": course_id,
                "phase": "generation",
                "context": None,
            }

    async def create_course_from_topic_with_files_stream(
        self,
        user_id: str,
        topic: str,
        files: List[Dict[str, Any]],
        model: Optional[str] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Streaming version of create_course_from_topic_with_files."""
        import asyncio
        start_time = time.time()

        profile = self._get_or_create_profile(user_id, {})
        model_config = ModelConfig.get_config(model)
        course_id = None
        total_chapters = None

        try:
            # Chunk all files
            all_chunks = []
            for f in files:
                file_chunks = self._chunk_document_for_course(
                    extracted_text=f["extracted_text"],
                    filename=f["filename"]
                )
                all_chunks.extend(file_chunks)

            # Embed/upsert and build doc map in parallel
            pre_course_id = generate_uuid()
            embed_task = self._embed_and_upsert_chunks(
                course_id=pre_course_id,
                all_chunks=all_chunks,
                source_files=[{"filename": f["filename"]} for f in files]
            )
            doc_map_task = self._build_document_map(all_chunks, model_config)
            _, doc_map = await asyncio.gather(embed_task, doc_map_task)

            doc_map_context = "SUPPLEMENTARY MATERIAL MAP (from uploaded files):\n"
            for t in doc_map.get("topics", []):
                doc_map_context += f"- {t['topic']}: {t.get('description', '')} [{len(t.get('chunk_indices', []))} sections]\n"

            source_files = [
                {"file_id": generate_uuid(), "filename": f["filename"], "extracted_topic": f.get("topic", ""), "source_url": f.get("source_url"), "source_type": f.get("source_type", "pdf")}
                for f in files
            ]

            async for event in self._generate_full_course_stream(
                user_id=user_id,
                topic=topic,
                source_type=SourceType.TOPIC,
                profile=profile,
                model_config=model_config,
                source_files=source_files,
                doc_map=doc_map,
                doc_map_context=doc_map_context,
                all_chunks=all_chunks,
                course_id=pre_course_id
            ):
                if event["type"] == "outline_ready":
                    course_id = event["course_id"]
                    total_chapters = event["total_chapters"]
                yield event
                if event["type"] == "error":
                    return

            generation_time = round(time.time() - start_time, 2)
            yield {
                "type": "course_complete",
                "course_id": course_id,
                "status": CourseStatus.READY,
                "total_chapters": total_chapters,
                "generation_time_seconds": generation_time
            }
        except Exception as e:
            logger.error(f"Streaming topic+files course generation failed: {e}")
            yield {
                "type": "error",
                "error": "GENERATION_FAILED",
                "message": str(e),
                "status_code": 500,
                "course_id": course_id,
                "phase": "generation",
                "context": None,
            }

    async def create_course_from_files_stream(
        self,
        user_id: str,
        files: List[Dict[str, Any]],
        organization: str,
        model: Optional[str] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Streaming version of create_course_from_files."""
        import asyncio
        start_time = time.time()

        profile = self._get_or_create_profile(user_id, {})
        model_config = ModelConfig.get_config(model)
        course_id = None
        total_chapters = None

        try:
            # Multi-file organization analysis
            chosen_org = None
            if len(files) > 1:
                _, chosen_org = await self._analyze_multi_files(files, organization, model_config)

            # Branch: SEPARATE_COURSES
            if chosen_org == OrganizationType.SEPARATE_COURSES:
                for idx, f in enumerate(files):
                    yield {
                        "type": "course_start",
                        "course_index": idx + 1,
                        "total_courses": len(files),
                        "filename": f["filename"]
                    }
                    async for event in self._generate_single_file_course_stream(
                        user_id=user_id,
                        file_data=f,
                        profile=profile,
                        model_config=model_config
                    ):
                        yield event
                        if event["type"] == "error":
                            return

                generation_time = round(time.time() - start_time, 2)
                yield {
                    "type": "course_complete",
                    "status": CourseStatus.READY,
                    "organization": "separate_courses",
                    "total_courses": len(files),
                    "generation_time_seconds": generation_time
                }
                return

            # Single-course path
            all_chunks = []
            for f in files:
                file_chunks = self._chunk_document_for_course(
                    extracted_text=f["extracted_text"],
                    filename=f["filename"]
                )
                all_chunks.extend(file_chunks)

            pre_course_id = generate_uuid()
            embed_task = self._embed_and_upsert_chunks(
                course_id=pre_course_id,
                all_chunks=all_chunks,
                source_files=[{"filename": f["filename"]} for f in files]
            )
            doc_map_task = self._build_document_map(all_chunks, model_config)
            _, doc_map = await asyncio.gather(embed_task, doc_map_task)

            combined_topic = " + ".join([f["topic"] for f in files]) if len(files) > 1 else files[0]["topic"]
            doc_map_context = "DOCUMENT MAP (topics found in uploaded material):\n"
            for topic in doc_map.get("topics", []):
                doc_map_context += f"- {topic['topic']}: {topic.get('description', '')} [{len(topic.get('chunk_indices', []))} sections]\n"

            async for event in self._generate_full_course_from_files_stream(
                user_id=user_id,
                course_id=pre_course_id,
                topic=combined_topic,
                profile=profile,
                model_config=model_config,
                source_files=[{"file_id": generate_uuid(), "filename": f["filename"], "extracted_topic": f["topic"], "source_url": f.get("source_url"), "source_type": f.get("source_type", "pdf")} for f in files],
                doc_map=doc_map,
                doc_map_context=doc_map_context,
                all_chunks=all_chunks,
                organization=chosen_org
            ):
                if event["type"] == "outline_ready":
                    course_id = event["course_id"]
                    total_chapters = event["total_chapters"]
                yield event
                if event["type"] == "error":
                    return

            generation_time = round(time.time() - start_time, 2)
            yield {
                "type": "course_complete",
                "course_id": course_id,
                "status": CourseStatus.READY,
                "total_chapters": total_chapters,
                "generation_time_seconds": generation_time
            }
        except Exception as e:
            logger.error(f"Streaming file course generation failed: {e}")
            yield {
                "type": "error",
                "error": "GENERATION_FAILED",
                "message": str(e),
                "status_code": 500,
                "course_id": course_id,
                "phase": "generation",
                "context": None,
            }

    async def resume_course_stream(
        self,
        course_id: str,
        model: Optional[str] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Streaming variant of resume_course.
        Yields SSE events as missing chapters are generated.
        Raises NotFoundError / ValidationError BEFORE any yield so the route
        handler can catch them before the stream starts.
        """
        import asyncio
        start_time = time.time()

        # ── 1. Load course and validate (pre-yield — exceptions propagate) ──
        course = self.storage.get_course(course_id)
        if not course:
            raise NotFoundError(f"Course {course_id} not found")

        outline = course.get("outline")
        if not outline or not outline.get("chapters"):
            raise ValidationError(f"Course {course_id} has no outline — cannot resume")

        existing_chapters = course.get("chapters", [])
        ready_orders = {
            ch["order"] for ch in existing_chapters if ch.get("status") == "ready"
        }
        all_outline_orders = {ch["order"] for ch in outline["chapters"]}
        missing_orders = all_outline_orders - ready_orders

        # Check extras status before deciding if truly complete
        has_study_guide = self.study_guide_storage.get_study_guide(course_id) is not None
        has_flashcards = self.flashcard_storage.get_flashcards(course_id) is not None

        # Early return if already complete (chapters + extras)
        if not missing_orders and has_study_guide and has_flashcards:
            logger.info(f"Course {course_id} is already complete ({len(ready_orders)} chapters)")
            yield {
                "type": "course_complete",
                "course_id": course_id,
                "status": course.get("status", CourseStatus.READY),
                "total_chapters": len(all_outline_orders),
                "generation_time_seconds": 0,
                "resume_summary": {
                    "already_complete": True,
                    "chapters_total": len(all_outline_orders),
                    "chapters_existed": len(all_outline_orders),
                    "chapters_generated": 0,
                    "chapters_failed": [],
                    "study_guide_generated": False,
                    "flashcards_generated": False,
                },
            }
            return

        logger.info(
            f"[resume-stream] Resuming course {course_id}: {len(missing_orders)} missing "
            f"chapters out of {len(all_outline_orders)} total"
        )

        # ── 2. Reconstruct context ───────────────────────────────────────
        user_id = course.get("user_id", "unknown")
        personalization = course.get("personalization_params", {})
        profile = self._get_or_create_profile(user_id, personalization)

        model_config = ModelConfig.get_config(model or course.get("model_used"))

        model_key = None
        from utils.model_config import MODEL_CONFIGS
        for key, cfg in MODEL_CONFIGS.items():
            if cfg["model"] == model_config["model"]:
                model_key = key
                break
        search_mode = get_search_mode(model_key)

        # File contexts from Pinecone (for file-based courses)
        missing_outlines = [
            ch for ch in outline["chapters"] if ch["order"] in missing_orders
        ]
        chapter_contexts: Dict[int, str] = {}
        source_type = course.get("source_type", "")
        if source_type in ("files", "mixed"):
            chapter_contexts = self._retrieve_chunks_for_resume(course_id, missing_outlines)

        # Web sources for Perplexity models
        chapter_web_sources: Dict[int, Dict] = {}
        if search_mode == "perplexity":
            from clients.perplexity_client import search_for_chapters_parallel
            chapter_web_sources = search_for_chapters_parallel(
                chapters=missing_outlines,
                course_topic=course.get("topic", course.get("title", ""))
            )

        # Yield resume_started event
        sorted_missing = sorted(missing_orders)
        yield {
            "type": "resume_started",
            "course_id": course_id,
            "missing_chapters": sorted_missing,
            "total_to_generate": len(sorted_missing),
            "needs_study_guide": not has_study_guide,
            "needs_flashcards": not has_flashcards,
        }

        # ── 4. Generate missing chapters in batched parallel ──────────────
        course_title = outline["title"]
        total_chapters = len(outline["chapters"])
        all_chapter_outlines = {ch["order"]: ch for ch in outline["chapters"]}

        # Update status to generating
        course["status"] = CourseStatus.GENERATING
        self.storage.save_course(course)

        def get_chapter_search_mode(chapter_order: int) -> str:
            if search_mode == "perplexity" and chapter_web_sources.get(chapter_order, {}).get("sources"):
                return "provided"
            return search_mode

        def get_chapter_web_sources_fn(chapter_order: int) -> Optional[List[Dict]]:
            if search_mode == "perplexity":
                return chapter_web_sources.get(chapter_order, {}).get("sources")
            return None

        async def generate_single_chapter(chapter_outline):
            ch_order = chapter_outline["order"]
            ch_search_mode = get_chapter_search_mode(ch_order)
            ch_web_sources = get_chapter_web_sources_fn(ch_order)
            ch_file_context = chapter_contexts.get(ch_order)
            ch_use_search = (search_mode == "native" and model_config["provider"] == "openai")

            prev_order = ch_order - 1
            next_order = ch_order + 1
            prev_title = all_chapter_outlines[prev_order]["title"] if prev_order in all_chapter_outlines else None
            next_title = all_chapter_outlines[next_order]["title"] if next_order in all_chapter_outlines else None

            chapter = await self._generate_chapter(
                course_id=course_id,
                course_title=course_title,
                chapter_outline=chapter_outline,
                total_chapters=total_chapters,
                profile=profile,
                model_config=model_config,
                prev_chapter_title=prev_title,
                next_chapter_title=next_title,
                file_context=ch_file_context,
                search_mode=ch_search_mode,
                web_sources=ch_web_sources,
                use_search=ch_use_search,
            )
            self.storage.save_chapter(course_id, chapter)
            logger.info(f"[resume-stream] Generated chapter {ch_order}/{total_chapters}: {chapter['title']}")
            return chapter

        BATCH_SIZE = 4
        generated_chapters = []
        failed_orders: List[int] = []

        try:
            sorted_missing_outlines = sorted(missing_outlines, key=lambda ch: ch["order"])
            for batch_start in range(0, len(sorted_missing_outlines), BATCH_SIZE):
                batch = sorted_missing_outlines[batch_start:batch_start + BATCH_SIZE]
                batch_tasks = [generate_single_chapter(ch) for ch in batch]
                results = await asyncio.gather(*batch_tasks, return_exceptions=True)

                for ch_outline, result in zip(batch, results):
                    if isinstance(result, Exception):
                        failed_orders.append(ch_outline["order"])
                        logger.error(f"[resume-stream] Chapter {ch_outline['order']} generation failed: {result}")
                    else:
                        generated_chapters.append(result)
                        yield {
                            "type": "chapter_ready",
                            "course_id": course_id,
                            "chapter_order": result["order"],
                            "chapter_title": result["title"],
                            "total_chapters": total_chapters,
                            "chapter": result,
                        }

                if batch_start + BATCH_SIZE < len(sorted_missing_outlines):
                    await asyncio.sleep(1)
        except Exception as e:
            logger.error(f"[resume-stream] Batched chapter generation failed: {e}")
            yield {
                "type": "error",
                "error": "CHAPTER_GENERATION_FAILED",
                "message": f"Resume chapter generation failed: {e}",
                "status_code": 500,
                "course_id": course_id,
                "phase": "chapter_generation",
                "context": None,
            }
            return

        # ── 5. Generate study guide & flashcards if missing (non-fatal) ───
        sg_generated = False
        fc_generated = False
        try:
            refreshed_course = self.storage.get_course(course_id)
            all_chapters = refreshed_course.get("chapters", [])
            topic = course.get("topic", course.get("title", ""))

            extras_tasks = []
            extras_labels = []
            if not has_study_guide:
                extras_tasks.append(
                    self._generate_study_guide(course_id, all_chapters, profile, model_config, topic)
                )
                extras_labels.append("study_guide")
            if not has_flashcards:
                extras_tasks.append(
                    self._generate_course_flashcards(course_id, all_chapters, profile, model_config)
                )
                extras_labels.append("flashcards")

            if extras_tasks:
                extras_results = await asyncio.gather(*extras_tasks, return_exceptions=True)
                for label, r in zip(extras_labels, extras_results):
                    if isinstance(r, Exception):
                        logger.warning(f"[resume-stream] Extras generation failed (non-fatal): {r}")
                    elif label == "study_guide":
                        sg_generated = True
                        yield {"type": "study_guide_ready", "course_id": course_id}
                    elif label == "flashcards":
                        fc_generated = True
                        yield {"type": "flashcards_ready", "course_id": course_id}
        except Exception as e:
            logger.warning(f"[resume-stream] Study guide / flashcard generation error (non-fatal): {e}")

        # ── 6. Finalize ──────────────────────────────────────────────────
        if not failed_orders:
            self.storage.save_course({
                "id": course_id,
                "status": CourseStatus.READY,
                "completed_at": datetime.utcnow(),
            })
            course["status"] = CourseStatus.READY
        else:
            logger.warning(
                f"[resume-stream] {len(failed_orders)} chapters failed "
                f"(orders: {failed_orders}), leaving course {course_id} in GENERATING status"
            )

        generation_time = round(time.time() - start_time, 2)

        self.logger.log_generation({
            "type": "resume_course_stream",
            "user_id": user_id,
            "course_id": course_id,
            "model": model_config["model"],
            "chapters_resumed": len(generated_chapters),
            "total_chapters": total_chapters,
            "generation_time": generation_time,
            "status": "success",
        })

        yield {
            "type": "course_complete",
            "course_id": course_id,
            "status": course["status"],
            "total_chapters": total_chapters,
            "generation_time_seconds": generation_time,
            "resume_summary": {
                "already_complete": False,
                "chapters_total": total_chapters,
                "chapters_existed": len(ready_orders),
                "chapters_generated": len(generated_chapters),
                "chapters_failed": failed_orders,
                "study_guide_generated": sg_generated,
                "flashcards_generated": fc_generated,
            },
        }

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

    def _build_chapters_summary(self, chapters: List[Dict[str, Any]]) -> str:
        """Build a text summary of all chapters for use in study guide / flashcard / exam prompts."""
        parts = []
        for ch in chapters:
            title = ch.get("title", "Untitled")
            order = ch.get("order", ch.get("order_index", "?"))
            objectives = ch.get("learning_objectives", [])
            key_concepts = ch.get("key_concepts", [])
            content = ch.get("content", "")
            # Use first 500 chars of content as summary
            content_preview = content[:500] if content else ""

            part = f"""Chapter {order}: {title}
  Objectives: {', '.join(objectives)}
  Key Concepts: {', '.join(key_concepts)}
  Content Preview: {content_preview}"""
            parts.append(part)
        return "\n\n".join(parts)

    async def _generate_study_guide(
        self,
        course_id: str,
        chapters: List[Dict[str, Any]],
        profile: 'LearningProfile',
        model_config: Dict[str, Any],
        topic: str
    ) -> bool:
        """Generate a comprehensive study guide from all chapters and save to DB."""
        course = self.storage.get_course(course_id)
        if not course:
            return False

        chapters_summary = self._build_chapters_summary(chapters)
        prompt = build_study_guide_prompt(
            course_title=course.get("title", ""),
            topic=topic,
            chapters_summary=chapters_summary,
            expertise=profile.expertise.value,
            role=profile.role.value,
            learning_goal=profile.learning_goal.value
        )

        result = await self._call_model(prompt, model_config, expect_json=True)
        if result:
            self.study_guide_storage.save_study_guide(course_id, result)
            logger.info(f"Study guide generated for course {course_id}")
            return True
        return False

    async def _generate_course_flashcards(
        self,
        course_id: str,
        chapters: List[Dict[str, Any]],
        profile: 'LearningProfile',
        model_config: Dict[str, Any]
    ) -> bool:
        """Generate course-wide flashcards, save to DB, and distribute to individual chapters."""
        course = self.storage.get_course(course_id)
        if not course:
            return False

        chapters_summary = self._build_chapters_summary(chapters)
        prompt = build_course_flashcards_prompt(
            course_title=course.get("title", ""),
            chapters_summary=chapters_summary,
            total_chapters=len(chapters)
        )

        result = await self._call_model(prompt, model_config, expect_json=True)
        flashcards = result.get("flashcards", []) if result else []
        if flashcards:
            # Save course-level flashcards
            self.flashcard_storage.save_flashcards(course_id, flashcards)
            logger.info(f"Generated {len(flashcards)} flashcards for course {course_id}")

            # Distribute flashcards to individual chapters by matching chapter_ref to title
            self._distribute_flashcards_to_chapters(course_id, chapters, flashcards)
            return True
        return False

    def _distribute_flashcards_to_chapters(
        self,
        course_id: str,
        chapters: List[Dict[str, Any]],
        flashcards: List[Dict[str, Any]]
    ):
        """Distribute course-level flashcards to individual chapters based on chapter_ref."""
        from clients.supabase_client import get_supabase

        # Build lookup: normalized chapter title -> chapter record
        chapter_lookup = {}
        for ch in chapters:
            title = ch.get("title", "")
            chapter_lookup[title.lower().strip()] = ch

        # Group flashcards by chapter_ref
        chapter_flashcards: Dict[str, List[Dict[str, Any]]] = {}
        unmatched = []
        for fc in flashcards:
            ref = (fc.get("chapter_ref") or "").lower().strip()
            matched = False
            # Try exact match first
            if ref in chapter_lookup:
                ch_id = chapter_lookup[ref]["id"]
                chapter_flashcards.setdefault(ch_id, []).append(fc)
                matched = True
            else:
                # Fuzzy match: check if ref is a substring of any chapter title or vice versa
                for title_lower, ch in chapter_lookup.items():
                    if ref and (ref in title_lower or title_lower in ref):
                        ch_id = ch["id"]
                        chapter_flashcards.setdefault(ch_id, []).append(fc)
                        matched = True
                        break
            if not matched:
                unmatched.append(fc)

        # Save per-chapter flashcards
        for ch_id, ch_fcs in chapter_flashcards.items():
            try:
                get_supabase().table("chapters").update(
                    {"flashcards": ch_fcs}
                ).eq("id", ch_id).execute()
            except Exception as e:
                logger.warning(f"Failed to save flashcards for chapter {ch_id}: {e}")

        if unmatched:
            logger.info(f"{len(unmatched)} flashcards could not be matched to chapters (kept in course-level only)")

    async def generate_final_exam(
        self,
        course_id: str,
        user_id: str,
        exam_size: int = 30,
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate an on-demand final exam with MCQ + fill-in-gap + theory questions."""
        import asyncio

        model_config = ModelConfig.get_config(model)
        course = self.storage.get_course(course_id)
        if not course:
            raise NotFoundError("Course not found", error_code="COURSE_NOT_FOUND", context={"course_id": course_id})

        chapters = course.get("chapters", [])
        if not chapters:
            raise NotFoundError("Course has no chapters", error_code="CHAPTER_NOT_FOUND", context={"course_id": course_id})

        chapters_summary = self._build_chapters_summary(chapters)

        # Determine question counts
        if exam_size == 50:
            mcq_count, fill_count, theory_count = 25, 15, 10
        else:
            mcq_count, fill_count, theory_count = 15, 8, 7

        # Generate all 3 question types in parallel
        mcq_task = self._call_model(
            build_final_exam_prompt("mcq", course["title"], chapters_summary, mcq_count, len(chapters)),
            model_config, expect_json=True
        )
        fill_task = self._call_model(
            build_final_exam_prompt("fill_in_gap", course["title"], chapters_summary, fill_count, len(chapters)),
            model_config, expect_json=True
        )
        theory_task = self._call_model(
            build_final_exam_prompt("theory", course["title"], chapters_summary, theory_count, len(chapters)),
            model_config, expect_json=True
        )

        results = await asyncio.gather(mcq_task, fill_task, theory_task, return_exceptions=True)

        # Check for failures
        for r in results:
            if isinstance(r, Exception):
                raise CourseGenerationError(f"Exam question generation failed: {r}")

        mcq_result, fill_result, theory_result = results

        exam_data = {
            "mcq": mcq_result.get("mcq", []),
            "fill_in_gap": fill_result.get("fill_in_gap", []),
            "theory": theory_result.get("theory", []),
        }
        exam_data["total_questions"] = len(exam_data["mcq"]) + len(exam_data["fill_in_gap"]) + len(exam_data["theory"])

        # Save to DB
        exam_id = self.exam_storage.save_exam(course_id, user_id, exam_size, exam_data)
        exam_data["id"] = exam_id
        exam_data["course_id"] = course_id
        exam_data["exam_size"] = exam_size

        logger.info(f"Final exam generated for course {course_id}: {exam_data['total_questions']} questions")
        return exam_data

    async def grade_exam_submission(
        self,
        exam_id: str,
        user_id: str,
        answers: Dict[str, Any],
        model: Optional[str] = None,
        time_taken_seconds: Optional[int] = None
    ) -> Dict[str, Any]:
        """Grade a submitted exam with MCQ, fill-in-gap, and theory answers."""
        import asyncio

        # Fetch exam
        exam = self.exam_storage.get_exam_by_id(exam_id)
        if not exam:
            raise NotFoundError("Exam not found", error_code="EXAM_NOT_FOUND", context={"exam_id": exam_id})

        model_config = ModelConfig.get_config(model)

        mcq_questions = exam.get("mcq", [])
        fill_questions = exam.get("fill_in_gap", [])
        theory_questions = exam.get("theory", [])

        mcq_answers = answers.get("mcq", [])
        fill_answers = answers.get("fill_in_gap", [])
        theory_answers = answers.get("theory", [])

        # --- Grade MCQ (instant) ---
        mcq_results = []
        mcq_correct = 0
        for i, q in enumerate(mcq_questions):
            selected = mcq_answers[i] if i < len(mcq_answers) else None
            correct = q.get("correct_answer")
            is_correct = selected == correct
            if is_correct:
                mcq_correct += 1
            mcq_results.append({
                "question_index": i,
                "selected": selected,
                "correct": correct,
                "is_correct": is_correct,
                "explanation": q.get("explanation", "")
            })

        mcq_score = (mcq_correct / len(mcq_questions) * 100) if mcq_questions else None

        # --- Grade Fill-in-gap (instant) ---
        fill_results = []
        fill_correct = 0
        for i, q in enumerate(fill_questions):
            raw_answer = fill_answers[i] if i < len(fill_answers) else ""
            student_answer = str(raw_answer).strip() if raw_answer is not None else ""
            correct_answer = q.get("correct_answer", "")
            alternatives = q.get("alternatives", [])

            student_lower = student_answer.lower().strip()
            correct_lower = correct_answer.lower().strip()

            if student_lower == correct_lower:
                match_type = "exact"
                is_correct = True
            elif any(student_lower == alt.lower().strip() for alt in alternatives):
                match_type = "alternative"
                is_correct = True
            else:
                match_type = "no_match"
                is_correct = False

            if is_correct:
                fill_correct += 1

            fill_results.append({
                "question_index": i,
                "answer": student_answer,
                "correct_answer": correct_answer,
                "is_correct": is_correct,
                "match_type": match_type
            })

        fill_score = (fill_correct / len(fill_questions) * 100) if fill_questions else None

        # --- Grade Theory (parallel LLM calls) ---
        theory_results = []
        theory_score = None

        if theory_questions:
            async def grade_single_theory(i, q):
                raw_answer = theory_answers[i] if i < len(theory_answers) else ""
                student_answer = str(raw_answer) if raw_answer is not None else ""
                if not student_answer.strip():
                    rubric = q.get("rubric", [])
                    return {
                        "question_index": i,
                        "rubric_breakdown": [
                            {"point": p, "status": "missed", "feedback": "No answer provided"}
                            for p in rubric
                        ],
                        "score": 0,
                        "max_score": 10,
                        "feedback": "No answer was provided."
                    }

                prompt = build_theory_grading_prompt(
                    question=q.get("question", ""),
                    student_answer=student_answer,
                    model_answer=q.get("model_answer", ""),
                    rubric=q.get("rubric", [])
                )

                result = await self._call_model(prompt, model_config, expect_json=True)
                result["question_index"] = i
                return result

            tasks = [grade_single_theory(i, q) for i, q in enumerate(theory_questions)]
            graded = await asyncio.gather(*tasks, return_exceptions=True)

            total_theory_score = 0
            total_theory_max = 0
            for i, result in enumerate(graded):
                if isinstance(result, Exception):
                    logger.error(f"Theory grading failed for question {i}: {result}")
                    rubric = theory_questions[i].get("rubric", [])
                    result = {
                        "question_index": i,
                        "rubric_breakdown": [
                            {"point": p, "status": "missed", "feedback": "Grading error"}
                            for p in rubric
                        ],
                        "score": 0,
                        "max_score": 10,
                        "feedback": "An error occurred during grading."
                    }
                theory_results.append(result)
                total_theory_score += result.get("score", 0)
                total_theory_max += result.get("max_score", 10)

            theory_score = (total_theory_score / total_theory_max * 100) if total_theory_max > 0 else 0

        # --- Overall score (weighted) ---
        weights = {"mcq": 0.40, "fill_in_gap": 0.25, "theory": 0.35}
        weighted_sum = 0
        weight_total = 0

        if mcq_score is not None:
            weighted_sum += mcq_score * weights["mcq"]
            weight_total += weights["mcq"]
        if fill_score is not None:
            weighted_sum += fill_score * weights["fill_in_gap"]
            weight_total += weights["fill_in_gap"]
        if theory_score is not None:
            weighted_sum += theory_score * weights["theory"]
            weight_total += weights["theory"]

        overall_score = round(weighted_sum / weight_total, 2) if weight_total > 0 else 0

        results = {
            "mcq": mcq_results,
            "fill_in_gap": fill_results,
            "theory": theory_results
        }

        # Save attempt
        attempt_id = self.exam_attempt_storage.save_attempt(
            exam_id=exam_id,
            user_id=user_id,
            answers=answers,
            results=results,
            score=overall_score,
            mcq_score=round(mcq_score, 2) if mcq_score is not None else None,
            fill_in_gap_score=round(fill_score, 2) if fill_score is not None else None,
            theory_score=round(theory_score, 2) if theory_score is not None else None,
            time_taken_seconds=time_taken_seconds
        )

        logger.info(f"Exam {exam_id} graded for user {user_id}: {overall_score}%")

        return {
            "attempt_id": attempt_id,
            "score": overall_score,
            "mcq_score": round(mcq_score, 2) if mcq_score is not None else None,
            "fill_in_gap_score": round(fill_score, 2) if fill_score is not None else None,
            "theory_score": round(theory_score, 2) if theory_score is not None else None,
            "results": results
        }

    async def query_course(
        self,
        course_id: str,
        question: str,
        model: Optional[str] = None,
        top_k: int = 3,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Answer a question about a course using its source material from Pinecone.
        Searches course chunks, reranks, and generates a grounded answer.
        Supports persistent chat history when user_id is provided.
        """
        from clients.chonkie_client import embed_query_multimodal
        from clients.pinecone_client import hybrid_search, rerank_results

        model_config = ModelConfig.get_config(model)

        # Get course for context
        course = self.get_course(course_id)
        if not course:
            raise NotFoundError("Course not found", error_code="COURSE_NOT_FOUND", context={"course_id": course_id})

        # Determine the correct space_id for Pinecone search
        # Space-based courses store chunks under the original space_id,
        # while file-based courses use "course_{course_id}"
        original_space_id = course.get("space_id")
        pinecone_space_id = original_space_id if original_space_id else f"course_{course_id}"

        # Fetch chat history if user_id provided
        chat_history = []
        if user_id:
            try:
                from clients.redis_client import get_chat_history, push_messages
                cached = await get_chat_history(course_id, user_id)
                if cached:
                    chat_history = cached
                else:
                    db_messages = self.chat_storage.get_messages(course_id, user_id, limit=20)
                    if db_messages:
                        chat_history = [{"role": m["role"], "content": m["content"]} for m in db_messages]
                        # Warm Redis cache
                        await push_messages(course_id, user_id, chat_history)
            except Exception as e:
                logger.warning(f"Chat history fetch failed (non-fatal): {e}")

        # Embed the question
        query_result = embed_query_multimodal(question)
        query_emb = query_result["embedding"]

        # Search Pinecone filtered by course's space
        search_results = hybrid_search(
            query_emb=query_emb,
            space_id=pinecone_space_id,
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

        rag_context = "\n\n---\n\n".join(context_parts)

        # Build prompt — topic courses get expanded prompt with domain context + broader knowledge
        source_type = course.get("source_type", "files")
        if source_type == "topic":
            outline = course.get("outline", {})
            chapter_titles = [ch.get("title", "") for ch in outline.get("chapters", [])]
            personalization = course.get("personalization_params", {})

            prompt = build_topic_course_chat_prompt(
                course_title=course.get("title", ""),
                topic=course.get("topic", ""),
                description=outline.get("description", course.get("description", "")),
                outline_chapters=chapter_titles,
                personalization=personalization,
                rag_context=rag_context,
                chat_history=chat_history[-10:],
                question=question
            )
        else:
            prompt = build_course_chat_prompt(
                course_title=course.get("title", ""),
                rag_context=rag_context,
                chat_history=chat_history[-10:],
                question=question
            )

        response = await self._call_model(prompt, model_config, expect_json=False, max_tokens_override=2048)
        answer = response.get("content", "")

        source_excerpts = [
            {
                "text": r.get("metadata", {}).get("text", "")[:200],
                "file": r.get("metadata", {}).get("source_file", "")
            }
            for r in reranked
        ]

        # Save chat messages to DB and Redis if user_id provided
        if user_id:
            try:
                self.chat_storage.save_message(course_id, user_id, "user", question)
                self.chat_storage.save_message(course_id, user_id, "assistant", answer, sources=source_excerpts)
                from clients.redis_client import push_messages
                await push_messages(course_id, user_id, [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer}
                ])
            except Exception as e:
                logger.warning(f"Chat history save failed (non-fatal): {e}")

        return {
            "course_id": course_id,
            "question": question,
            "answer": answer,
            "sources_used": len(reranked),
            "source_excerpts": source_excerpts
        }

    # ================================================================
    # Notes Generation
    # ================================================================

    async def _auto_ingest_documents(
        self,
        docs: Dict[str, Any],
        space_id: str
    ) -> None:
        """
        Auto-ingest documents that exist in a space but haven't been
        chunked/embedded/upserted to Pinecone yet.

        Processes YouTube videos (via extracted_text) and PDFs (via Mistral OCR).
        """
        from clients.chonkie_client import embed_chunks_multimodal
        from clients.pinecone_client import upsert_vectors
        from clients.supabase_client import get_yt_extracted_text

        # Process YouTube videos
        for yt in docs.get("yts", []):
            yt_id = yt["id"]
            yt_name = yt.get("file_name", "YouTube Video")
            logger.info(f"Auto-ingesting YouTube video: {yt_name} ({yt_id})")

            extracted_text = get_yt_extracted_text(yt_id)
            if not extracted_text:
                logger.warning(f"No extracted_text for YT {yt_id}, skipping")
                continue

            chunks = self._chunk_document_for_course(extracted_text, yt_name)
            if not chunks:
                logger.warning(f"Chunking produced 0 chunks for YT {yt_id}, skipping")
                continue

            embedded = embed_chunks_multimodal(chunks)
            if embedded and isinstance(embedded[0], dict) and "message" in embedded[0]:
                logger.error(f"Embedding failed for YT {yt_id}: {embedded[0]}")
                continue

            upsert_vectors(
                doc_id=yt_id,
                space_id=space_id,
                embeddings=embedded,
                chunks=chunks,
                source_file=yt.get("yt_url", yt_name)
            )
            logger.info(f"Ingested YT {yt_id}: {len(chunks)} chunks")

        # Process PDFs
        for pdf in docs.get("pdfs", []):
            pdf_id = pdf["id"]
            pdf_name = pdf.get("file_name", "PDF Document")
            file_path = pdf.get("file_path", "")
            logger.info(f"Auto-ingesting PDF: {pdf_name} ({pdf_id})")

            if not file_path:
                logger.warning(f"No file_path for PDF {pdf_id}, skipping")
                continue

            try:
                from processors.mistral_ocr_extractor import MistralOCRExtractor
                ocr_result = MistralOCRExtractor().process_document(file_path)
                full_text = ocr_result.get("full_text", "")
            except Exception as e:
                logger.error(f"OCR extraction failed for PDF {pdf_id}: {e}")
                continue

            if not full_text:
                logger.warning(f"OCR produced empty text for PDF {pdf_id}, skipping")
                continue

            chunks = self._chunk_document_for_course(full_text, pdf_name)
            if not chunks:
                logger.warning(f"Chunking produced 0 chunks for PDF {pdf_id}, skipping")
                continue

            embedded = embed_chunks_multimodal(chunks)
            if embedded and isinstance(embedded[0], dict) and "message" in embedded[0]:
                logger.error(f"Embedding failed for PDF {pdf_id}: {embedded[0]}")
                continue

            upsert_vectors(
                doc_id=pdf_id,
                space_id=space_id,
                embeddings=embedded,
                chunks=chunks,
                source_file=pdf_name
            )
            logger.info(f"Ingested PDF {pdf_id}: {len(chunks)} chunks")

    async def generate_notes(
        self,
        course_id: str,
        user_id: str,
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive study notes from course source materials.
        Stores result in courses.summary_md and generates a study guide.

        Flow:
        1. Validate course exists and user owns it
        2. Fetch all source chunks from Pinecone (space-based or file-based)
        3. Build document map from chunks
        4. Update course title/topic/slug from document map
        5. Generate notes: intro + sections (batched parallel) + conclusion
        6. Save summary_md
        7. Generate study guide
        """
        import asyncio
        from clients.pinecone_client import (
            fetch_all_document_chunks, fetch_chunks_by_space, update_vector_metadata
        )
        from clients.supabase_client import (
            get_course_by_id, get_documents_in_space, update_course_summary_md
        )

        start_time = time.time()
        model_config = ModelConfig.get_config(model)

        # 1. Validate course
        course = get_course_by_id(course_id)
        if not course:
            raise NotFoundError("Course not found", error_code="COURSE_NOT_FOUND", context={"course_id": course_id})
        if course.get("user_id") != user_id:
            raise ValidationError("Not authorized to generate notes for this course", error_code="MISSING_REQUIRED_FIELD")

        space_id = course.get("space_id")
        source_urls: List[Dict[str, str]] = []

        # 2. Fetch source chunks
        if space_id:
            # Space-based course: fetch chunks from all documents in the space
            docs = get_documents_in_space(space_id)
            doc_ids = []

            for pdf in docs.get("pdfs", []):
                doc_ids.append(pdf["id"])
                url = pdf.get("file_path", "")
                name = pdf.get("file_name", "PDF Document")
                if url:
                    source_urls.append({"name": name, "url": url})

            for yt in docs.get("yts", []):
                doc_ids.append(yt["id"])
                url = yt.get("yt_url", "")
                name = yt.get("file_name", "YouTube Video")
                if url:
                    source_urls.append({"name": name, "url": url})

            if not doc_ids:
                raise NotFoundError(f"No documents found in space {space_id}", error_code="MISSING_SOURCE", context={"space_id": space_id})

            raw_chunks = fetch_chunks_by_space(
                space_id=space_id,
                document_ids=doc_ids,
                max_chunks_per_doc=2000,
                target_coverage=0.80
            )

            # Tag vectors with course_id (best-effort)
            vector_ids = [c.get("id") for c in raw_chunks if c.get("id")]
            if vector_ids:
                update_vector_metadata(vector_ids, {"course_id": course_id})
        else:
            # File-based course: chunks stored with document_id=course_id
            raw_chunks = fetch_all_document_chunks(
                document_id=course_id,
                space_id=f"course_{course_id}",
                max_chunks=2000,
                target_coverage=0.80
            )

            # Collect source URLs from course.source_files
            for sf in (course.get("source_files") or []):
                url = sf.get("source_url", "")
                name = sf.get("filename", "Source")
                if url:
                    source_urls.append({"name": name, "url": url})

        if not raw_chunks and space_id:
            logger.info("No chunks in Pinecone, auto-ingesting documents...")
            await self._auto_ingest_documents(docs, space_id)

            # Re-fetch chunks after ingestion
            raw_chunks = fetch_chunks_by_space(
                space_id=space_id,
                document_ids=doc_ids,
                max_chunks_per_doc=2000,
                target_coverage=0.80
            )

        if not raw_chunks:
            raise NotFoundError("No source material found for this course. Upload files or add to a space first.", error_code="MISSING_SOURCE", context={"course_id": course_id})

        logger.info(f"Fetched {len(raw_chunks)} raw chunks for notes generation")

        # 3. Normalize chunks for document map
        normalized_chunks = []
        for i, chunk in enumerate(raw_chunks):
            metadata = chunk.get("metadata", {})
            text = metadata.get("text", "")
            source_file = metadata.get("source_file", "")
            normalized_chunks.append({
                "chunk_index": i,
                "text": text,
                "source_file": source_file
            })

        # 4. Build document map
        doc_map = await self._build_document_map(normalized_chunks, model_config)
        topics = doc_map.get("topics", [])
        if not topics:
            raise GenerationError("Document map produced no topics", error_code="NOTES_GENERATION_FAILED", context={"course_id": course_id})

        # 5. Update course title, topic, and slug from document map
        new_title = doc_map.get("document_title", course.get("title", ""))
        new_topic = topics[0].get("topic", course.get("topic", "")) if topics else course.get("topic", "")
        new_slug = generate_slug(new_title)

        # Use partial update (not upsert) to avoid overwriting other columns
        from clients.supabase_client import get_supabase
        get_supabase().table("courses").update({
            "title": new_title,
            "topic": new_topic,
            "slug": new_slug,
        }).eq("id", course_id).execute()
        logger.info(f"Updated course metadata: title='{new_title}', topic='{new_topic}', slug='{new_slug}'")

        # 6. Get learning profile for personalization
        profile = self._get_or_create_profile(user_id, {})

        # 7. Generate notes in 3 phases
        markdown_parts = []

        # 7a. Intro
        logger.info("Generating notes intro...")
        intro_prompt = build_course_notes_intro_prompt(
            document_title=new_title,
            topics=topics,
            expertise=profile.expertise.value,
            role=profile.role.value,
            learning_goal=profile.learning_goal.value,
            depth_pref=profile.depth_pref.value,
            example_pref=profile.example_pref.value,
            format_pref=profile.format_pref.value
        )
        intro_result = await self._call_model(intro_prompt, model_config, expect_json=False)
        markdown_parts.append(intro_result.get("content", ""))

        # 7b. Sections (batched parallel)
        BATCH_SIZE = 4
        section_markdowns = [""] * len(topics)

        async def generate_section(idx: int, topic: Dict[str, Any]) -> tuple:
            section_context = self._get_chunks_for_notes_section(
                topic=topic,
                all_chunks=normalized_chunks,
                max_context_tokens=6000
            )
            prev_title = topics[idx - 1]["topic"] if idx > 0 else None
            next_title = topics[idx + 1]["topic"] if idx < len(topics) - 1 else None

            prompt = build_course_notes_section_prompt(
                section_title=topic["topic"],
                section_description=topic.get("description", ""),
                source_material=section_context,
                section_number=idx + 1,
                total_sections=len(topics),
                prev_section_title=prev_title,
                next_section_title=next_title,
                expertise=profile.expertise.value,
                role=profile.role.value,
                learning_goal=profile.learning_goal.value,
                depth_pref=profile.depth_pref.value,
                example_pref=profile.example_pref.value,
                format_pref=profile.format_pref.value
            )
            result = await self._call_model(prompt, model_config, expect_json=False)
            return idx, result.get("content", "")

        logger.info(f"Generating {len(topics)} note sections in batches of {BATCH_SIZE}...")
        all_sections_indexed = list(enumerate(topics))
        for batch_start in range(0, len(all_sections_indexed), BATCH_SIZE):
            batch = all_sections_indexed[batch_start:batch_start + BATCH_SIZE]
            batch_tasks = [generate_section(idx, topic) for idx, topic in batch]
            results = await asyncio.gather(*batch_tasks)
            for idx, content in results:
                section_markdowns[idx] = content
                logger.info(f"Generated section {idx + 1}/{len(topics)}: {topics[idx]['topic']}")
            if batch_start + BATCH_SIZE < len(all_sections_indexed):
                await asyncio.sleep(1)

        markdown_parts.extend(section_markdowns)

        # 7c. Conclusion + Sources
        logger.info("Generating notes conclusion...")
        conclusion_prompt = build_course_notes_conclusion_prompt(
            document_title=new_title,
            topics=topics,
            source_urls=source_urls,
            expertise=profile.expertise.value,
            role=profile.role.value,
            learning_goal=profile.learning_goal.value
        )
        conclusion_result = await self._call_model(conclusion_prompt, model_config, expect_json=False)
        markdown_parts.append(conclusion_result.get("content", ""))

        # 8. Concatenate and save
        full_markdown = "\n\n---\n\n".join([p for p in markdown_parts if p.strip()])
        update_course_summary_md(course_id, full_markdown)
        logger.info(f"Saved summary_md for course {course_id} ({len(full_markdown)} chars)")

        # 9. Generate study guide
        has_study_guide = False
        try:
            chapters_summary = self._build_notes_chapters_summary(topics, normalized_chunks)
            sg_prompt = build_study_guide_prompt(
                course_title=new_title,
                topic=new_topic,
                chapters_summary=chapters_summary,
                expertise=profile.expertise.value,
                role=profile.role.value,
                learning_goal=profile.learning_goal.value
            )
            sg_result = await self._call_model(sg_prompt, model_config, expect_json=True)
            if sg_result:
                self.study_guide_storage.save_study_guide(course_id, sg_result)
                has_study_guide = True
                logger.info(f"Study guide generated for notes-based course {course_id}")
        except Exception as e:
            logger.warning(f"Study guide generation failed (non-fatal): {e}")

        generation_time = round(time.time() - start_time, 2)

        return {
            "course_id": course_id,
            "title": new_title,
            "slug": new_slug,
            "topic": new_topic,
            "notes_length": len(full_markdown),
            "sections_generated": len(topics),
            "model_used": model_config["model"],
            "has_study_guide": has_study_guide,
            "generation_time_seconds": generation_time,
            "summary_md": full_markdown
        }

    def _get_chunks_for_notes_section(
        self,
        topic: Dict[str, Any],
        all_chunks: List[Dict[str, Any]],
        max_context_tokens: int = 6000
    ) -> str:
        """
        Retrieve relevant chunks for a notes section using chunk_indices from doc map.
        Simpler than _get_chunks_for_chapter — uses direct index mapping.
        """
        chunk_indices = topic.get("chunk_indices", [])

        context_parts = []
        token_count = 0
        for idx in chunk_indices:
            if idx < len(all_chunks):
                chunk_text = all_chunks[idx].get("text", "")
                chunk_tokens = len(chunk_text) // 4  # Rough estimate
                if token_count + chunk_tokens > max_context_tokens:
                    break
                context_parts.append(f"[Source Section {idx + 1}]\n{chunk_text}")
                token_count += chunk_tokens

        context = "\n\n---\n\n".join(context_parts)
        logger.info(f"Notes section '{topic['topic']}': {len(context_parts)} chunks, ~{token_count} tokens")
        return context

    def _build_notes_chapters_summary(
        self,
        topics: List[Dict[str, Any]],
        all_chunks: List[Dict[str, Any]]
    ) -> str:
        """
        Convert doc map topics + chunk text into a chapters_summary string
        that build_study_guide_prompt() expects.
        """
        parts = []
        for i, topic in enumerate(topics, 1):
            chunk_indices = topic.get("chunk_indices", [])
            # Build a content preview from the first few chunks
            preview_parts = []
            for idx in chunk_indices[:3]:
                if idx < len(all_chunks):
                    text = all_chunks[idx].get("text", "")
                    preview_parts.append(text[:300])
            content_preview = " ".join(preview_parts)[:500]

            part = f"""Chapter {i}: {topic['topic']}
  Objectives: Understand {topic.get('description', topic['topic'])}
  Key Concepts: {topic.get('description', '')}
  Content Preview: {content_preview}"""
            parts.append(part)
        return "\n\n".join(parts)

    # ================================================================
    # Flashcard & Quiz Text Parsers
    # ================================================================

    def _parse_flashcards_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Parse structured text output from LLM into flashcard JSON objects.
        Splits on --- delimiter, extracts labeled fields via regex.
        Returns list of flashcard dicts. Skips malformed cards.
        """
        import re

        cards = []
        raw_blocks = re.split(r'\n\s*---\s*\n|\n\s*---\s*$|^\s*---\s*\n', text.strip())

        for block in raw_blocks:
            block = block.strip()
            if not block or "CARD" not in block.upper():
                continue

            # Extract fields — case-insensitive, handles multi-line values
            field_pattern = r'^\s*(Type|Front|Back|Hint|Concept|Difficulty)\s*:\s*(.+?)(?=\n\s*(?:Type|Front|Back|Hint|Concept|Difficulty)\s*:|$)'
            matches = re.findall(field_pattern, block, re.MULTILINE | re.DOTALL | re.IGNORECASE)

            fields = {}
            for key, value in matches:
                fields[key.strip().lower()] = value.strip()

            front = fields.get("front", "")
            back = fields.get("back", "")
            if not front or not back:
                logger.warning(f"Skipping malformed flashcard block (missing front/back): {block[:200]}")
                continue

            cards.append({
                "front": front,
                "back": back,
                "hint": fields.get("hint", ""),
                "concept": fields.get("concept", ""),
                "difficulty": fields.get("difficulty", "intermediate"),
                "type": fields.get("type", "application"),
            })

        return cards

    def _parse_quiz_text(self, text: str) -> Dict[str, Any]:
        """
        Parse structured text output from LLM into quiz JSON object.
        Splits on --- delimiter, extracts fields, groups by question type.
        Returns {mcq: [...], fill_in_gap: [...], scenario: [...]}.
        """
        import re

        questions = {"mcq": [], "fill_in_gap": [], "scenario": []}
        raw_blocks = re.split(r'\n\s*---\s*\n|\n\s*---\s*$|^\s*---\s*\n', text.strip())

        def _parse_correct_answer(raw: str, options: List[str]) -> int:
            """Parse correct answer from letter, number, or text. Returns 0-based index."""
            raw = raw.strip()
            # Try single letter: A, B, C, D
            if len(raw) == 1 and raw.upper().isalpha():
                return ord(raw.upper()) - ord('A')
            # Try letter with paren/period: "A)" or "A."
            letter_match = re.match(r'^([A-Da-d])[).\s]', raw)
            if letter_match:
                return ord(letter_match.group(1).upper()) - ord('A')
            # Try numeric: "0", "1", "2", "3"
            if raw.isdigit():
                return int(raw)
            # Fallback: try matching against option text
            for i, opt in enumerate(options):
                if raw.lower() in opt.lower() or opt.lower() in raw.lower():
                    return i
            return 0

        for block in raw_blocks:
            block = block.strip()
            if not block or "QUESTION" not in block.upper():
                continue

            # Case-insensitive field extraction
            field_pattern = r'^\s*(Type|Bloom|Question|Options|Correct|Answer|Explanation|Section|Difficulty)\s*:\s*(.+?)(?=\n\s*(?:Type|Bloom|Question|Options|Correct|Answer|Explanation|Section|Difficulty)\s*:|$)'
            matches = re.findall(field_pattern, block, re.MULTILINE | re.DOTALL | re.IGNORECASE)

            fields = {}
            for key, value in matches:
                fields[key.strip().lower()] = value.strip()

            q_type = fields.get("type", "mcq").lower().strip()
            question_text = fields.get("question", "")
            if not question_text:
                logger.warning(f"Skipping quiz block with no question text: {block[:200]}")
                continue

            base_q = {
                "question": question_text,
                "bloom": fields.get("bloom", "understand"),
                "explanation": fields.get("explanation", ""),
                "section": fields.get("section", ""),
                "difficulty": fields.get("difficulty", "medium"),
            }

            if q_type == "mcq":
                options_raw = fields.get("options", "")
                options = [o.strip() for o in re.split(r'\s*\|\s*', options_raw) if o.strip()]
                if not options:
                    options = [o.strip() for o in options_raw.split('\n') if o.strip()]

                correct_raw = fields.get("correct", "A")
                correct_index = _parse_correct_answer(correct_raw, options)

                base_q["options"] = options
                base_q["correct_answer"] = min(correct_index, len(options) - 1) if options else 0
                questions["mcq"].append(base_q)

            elif q_type in ("fill_in_gap", "fill_in_the_gap", "fill-in-gap", "fill"):
                base_q["correct_answer"] = fields.get("answer", "")
                questions["fill_in_gap"].append(base_q)

            elif q_type == "scenario":
                base_q["correct_answer"] = fields.get("answer", "")
                questions["scenario"].append(base_q)

            else:
                if fields.get("options"):
                    options_raw = fields.get("options", "")
                    options = [o.strip() for o in re.split(r'\s*\|\s*', options_raw) if o.strip()]
                    correct_raw = fields.get("correct", "A")
                    correct_index = _parse_correct_answer(correct_raw, options)
                    base_q["options"] = options
                    base_q["correct_answer"] = min(correct_index, len(options) - 1) if options else 0
                    questions["mcq"].append(base_q)
                else:
                    base_q["correct_answer"] = fields.get("answer", "")
                    questions["scenario"].append(base_q)

        return questions

    # ================================================================
    # Notes Flashcard & Quiz Generation
    # ================================================================

    async def _generate_notes_flashcards(
        self,
        course_id: str,
        user_id: str,
        topics: List[Dict[str, Any]],
        all_chunks: List[Dict[str, Any]],
        profile: 'LearningProfile',
        model_config: Dict[str, Any],
        course_title: str
    ) -> int:
        """
        Generate flashcards for notes-based course using per-section text prompts.
        Inserts each section's flashcards to DB incrementally (one set_number per section).
        Returns total flashcard count.
        """
        import asyncio
        from clients.supabase_client import insert_course_flashcard_set, delete_course_flashcard_sets

        BATCH_SIZE = 4
        all_flashcards = []

        # Clear previous flashcard sets for this course before regenerating
        try:
            delete_course_flashcard_sets(course_id)
        except Exception as e:
            logger.warning(f"Failed to delete old flashcard sets for course {course_id}: {e}")

        async def generate_section_flashcards(idx: int, topic: Dict[str, Any]) -> List[Dict[str, Any]]:
            section_context = self._get_chunks_for_notes_section(
                topic=topic,
                all_chunks=all_chunks,
                max_context_tokens=6000
            )

            prompt = build_notes_flashcards_prompt(
                section_title=topic["topic"],
                section_content=section_context,
                section_number=idx + 1,
                total_sections=len(topics),
                expertise=profile.expertise.value,
                role=profile.role.value,
                learning_goal=profile.learning_goal.value
            )

            result = await self._call_model(prompt, model_config, expect_json=False)
            text = result.get("content", "")
            cards = self._parse_flashcards_text(text)

            if not cards:
                logger.warning(f"No flashcards parsed from LLM output for section '{topic['topic']}' (raw length: {len(text)})")
                return []

            for card in cards:
                card["section_ref"] = topic["topic"]

            logger.info(f"Flashcards for section '{topic['topic']}': {len(cards)} cards parsed")

            # Incremental insert: save this section's cards immediately
            try:
                insert_course_flashcard_set(
                    course_id=course_id,
                    flashcards=cards,
                    set_number=idx + 1,
                    created_by=user_id,
                    is_shared=True
                )
            except Exception as e:
                logger.warning(f"Failed to insert flashcard set {idx+1} for course {course_id}: {e}")

            return cards

        all_sections_indexed = list(enumerate(topics))
        for batch_start in range(0, len(all_sections_indexed), BATCH_SIZE):
            batch = all_sections_indexed[batch_start:batch_start + BATCH_SIZE]
            batch_tasks = [generate_section_flashcards(idx, topic) for idx, topic in batch]
            results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, Exception):
                    logger.warning(f"Flashcard generation failed for a section: {result}")
                    continue
                all_flashcards.extend(result)
            if batch_start + BATCH_SIZE < len(all_sections_indexed):
                await asyncio.sleep(1)

        if not all_flashcards:
            logger.warning("No flashcards generated across any section")
            return 0

        for i, card in enumerate(all_flashcards, 1):
            card["id"] = f"fc_{i}"

        # Still write full set to courses.flashcards JSONB for backward compat
        self.flashcard_storage.save_flashcards(course_id, all_flashcards)

        return len(all_flashcards)

    async def _generate_notes_quiz(
        self,
        course_id: str,
        user_id: str,
        topics: List[Dict[str, Any]],
        all_chunks: List[Dict[str, Any]],
        profile: 'LearningProfile',
        model_config: Dict[str, Any],
        course_title: str
    ) -> Optional[Dict[str, Any]]:
        """
        Generate quiz for notes-based course using per-section text prompts.
        Inserts each section's quiz to DB incrementally (one set_number per section).
        Returns quiz metadata dict or None on failure.
        """
        import asyncio
        from clients.supabase_client import insert_course_quiz_set, delete_course_quiz_sets

        BATCH_SIZE = 4
        all_questions = {"mcq": [], "fill_in_gap": [], "scenario": []}

        # Clear previous quiz sets for this course before regenerating
        try:
            delete_course_quiz_sets(course_id)
        except Exception as e:
            logger.warning(f"Failed to delete old quiz sets for course {course_id}: {e}")

        all_topics_summary = "\n".join([
            f"{i+1}. {t['topic']}: {t.get('description', '')}"
            for i, t in enumerate(topics)
        ])

        async def generate_section_quiz(idx: int, topic: Dict[str, Any]) -> Dict[str, Any]:
            section_context = self._get_chunks_for_notes_section(
                topic=topic,
                all_chunks=all_chunks,
                max_context_tokens=6000
            )

            prompt = build_notes_quiz_prompt(
                section_title=topic["topic"],
                section_content=section_context,
                all_topics_summary=all_topics_summary,
                section_number=idx + 1,
                total_sections=len(topics),
                expertise=profile.expertise.value,
                role=profile.role.value,
                learning_goal=profile.learning_goal.value
            )

            result = await self._call_model(prompt, model_config, expect_json=False)
            text = result.get("content", "")
            parsed = self._parse_quiz_text(text)

            total = sum(len(v) for v in parsed.values())
            if total == 0:
                logger.warning(f"No quiz questions parsed from LLM output for section '{topic['topic']}' (raw length: {len(text)})")
            else:
                logger.info(f"Quiz for section '{topic['topic']}': {total} questions parsed")

            # Incremental insert: save this section's quiz immediately
            if total > 0:
                section_quiz_obj = {
                    **parsed,
                    "metadata": {
                        "total": total,
                        "by_type": {k: len(v) for k, v in parsed.items()},
                    }
                }
                try:
                    insert_course_quiz_set(
                        course_id=course_id,
                        quiz_obj=section_quiz_obj,
                        set_number=idx + 1,
                        title=f"Section {idx+1}: {topic['topic']}",
                        created_by=user_id,
                        is_shared=True
                    )
                except Exception as e:
                    logger.warning(f"Failed to insert quiz set {idx+1} for course {course_id}: {e}")

            return parsed

        all_sections_indexed = list(enumerate(topics))
        for batch_start in range(0, len(all_sections_indexed), BATCH_SIZE):
            batch = all_sections_indexed[batch_start:batch_start + BATCH_SIZE]
            batch_tasks = [generate_section_quiz(idx, topic) for idx, topic in batch]
            results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, Exception):
                    logger.warning(f"Quiz generation failed for a section: {result}")
                    continue
                for q_type in ("mcq", "fill_in_gap", "scenario"):
                    all_questions[q_type].extend(result.get(q_type, []))
            if batch_start + BATCH_SIZE < len(all_sections_indexed):
                await asyncio.sleep(1)

        total_questions = sum(len(v) for v in all_questions.values())
        if total_questions == 0:
            logger.warning("No quiz questions generated across any section")
            return None

        bloom_counts = {}
        difficulty_counts = {}
        for q_type_questions in all_questions.values():
            for q in q_type_questions:
                bloom = q.get("bloom", "understand")
                difficulty = q.get("difficulty", "medium")
                bloom_counts[bloom] = bloom_counts.get(bloom, 0) + 1
                difficulty_counts[difficulty] = difficulty_counts.get(difficulty, 0) + 1

        return {
            "total": total_questions,
            "by_type": {k: len(v) for k, v in all_questions.items()},
            "by_bloom": bloom_counts,
            "by_difficulty": difficulty_counts
        }

    async def _fetch_notes_context(
        self,
        course_id: str,
        user_id: str,
        model_config: Dict[str, Any]
    ) -> tuple:
        """
        Shared setup for notes-based flashcard/quiz generation.
        Fetches chunks, builds doc map, returns (course, topics, normalized_chunks, profile).
        """
        from clients.pinecone_client import (
            fetch_all_document_chunks, fetch_chunks_by_space
        )
        from clients.supabase_client import (
            get_course_by_id, get_documents_in_space
        )

        course = get_course_by_id(course_id)
        if not course:
            raise NotFoundError("Course not found", error_code="COURSE_NOT_FOUND", context={"course_id": course_id})
        if course.get("user_id") != user_id:
            raise ValidationError("Not authorized for this course", error_code="MISSING_REQUIRED_FIELD")

        space_id = course.get("space_id")

        if space_id:
            docs = get_documents_in_space(space_id)
            doc_ids = [pdf["id"] for pdf in docs.get("pdfs", [])]
            doc_ids += [yt["id"] for yt in docs.get("yts", [])]
            if not doc_ids:
                raise NotFoundError(f"No documents found in space {space_id}", error_code="MISSING_SOURCE", context={"space_id": space_id})
            raw_chunks = fetch_chunks_by_space(
                space_id=space_id,
                document_ids=doc_ids,
                max_chunks_per_doc=2000,
                target_coverage=0.80
            )
        else:
            raw_chunks = fetch_all_document_chunks(
                document_id=course_id,
                space_id=f"course_{course_id}",
                max_chunks=2000,
                target_coverage=0.80
            )

        if not raw_chunks:
            raise NotFoundError("No source material found for this course.", error_code="MISSING_SOURCE", context={"course_id": course_id})

        normalized_chunks = []
        for i, chunk in enumerate(raw_chunks):
            metadata = chunk.get("metadata", {})
            normalized_chunks.append({
                "chunk_index": i,
                "text": metadata.get("text", ""),
                "source_file": metadata.get("source_file", "")
            })

        doc_map = await self._build_document_map(normalized_chunks, model_config)
        topics = doc_map.get("topics", [])
        if not topics:
            raise GenerationError("Document map produced no topics", error_code="NOTES_GENERATION_FAILED", context={"course_id": course_id})

        profile = self._get_or_create_profile(user_id, {})
        title = course.get("title", "")

        return course, topics, normalized_chunks, profile, title

    async def generate_notes_flashcards(
        self,
        course_id: str,
        user_id: str,
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Public method: generate flashcards from course source materials.
        Standalone endpoint — does not require generate_notes() to run first.
        """
        start_time = time.time()
        model_config = ModelConfig.get_config(model)

        course, topics, normalized_chunks, profile, title = await self._fetch_notes_context(
            course_id, user_id, model_config
        )

        flashcard_count = await self._generate_notes_flashcards(
            course_id, user_id, topics, normalized_chunks, profile, model_config, title
        )

        generation_time = round(time.time() - start_time, 2)
        logger.info(f"Generated {flashcard_count} flashcards for course {course_id} in {generation_time}s")

        return {
            "course_id": course_id,
            "flashcard_count": flashcard_count,
            "sections_processed": len(topics),
            "model_used": model_config["model"],
            "generation_time_seconds": generation_time
        }

    async def generate_notes_quiz(
        self,
        course_id: str,
        user_id: str,
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Public method: generate quiz from course source materials.
        Standalone endpoint — does not require generate_notes() to run first.
        """
        start_time = time.time()
        model_config = ModelConfig.get_config(model)

        course, topics, normalized_chunks, profile, title = await self._fetch_notes_context(
            course_id, user_id, model_config
        )

        quiz_meta = await self._generate_notes_quiz(
            course_id, user_id, topics, normalized_chunks, profile, model_config, title
        )

        generation_time = round(time.time() - start_time, 2)
        logger.info(f"Generated quiz for course {course_id} in {generation_time}s: {quiz_meta}")

        return {
            "course_id": course_id,
            "has_quiz": quiz_meta is not None,
            "quiz_summary": quiz_meta,
            "sections_processed": len(topics),
            "model_used": model_config["model"],
            "generation_time_seconds": generation_time
        }
