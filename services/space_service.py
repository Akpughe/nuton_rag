"""
SpaceService: handles space-level operations.

  1. add_courses_to_space — batch Pinecone metadata update (adds space_id to vectors)
  2. query_space_stream   — parallel multi-course RAG query with streaming LLM response,
                           personalization, citations, and web fallback.
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, Dict, List, Optional

from clients.supabase_client import get_supabase
from prompts.space_prompts import build_space_query_prompt, build_space_web_fallback_prompt
from utils.file_storage import LearningProfileStorage, SpaceConversationStorage
from utils.model_config import ModelConfig

logger = logging.getLogger(__name__)

# Relevance score threshold: courses whose best chunk scores below this are excluded.
RELEVANCE_THRESHOLD = 0.25

# Top-K chunks to fetch per course for the query.
TOP_K_PER_COURSE = 6


def _sse(data: Dict[str, Any]) -> str:
    """Format a dict as an SSE data line."""
    return f"data: {json.dumps(data, default=str)}\n\n"


def _resolve_source_link(filename: str, source_files: List[Dict[str, Any]]) -> Optional[str]:
    """
    Resolve a clickable URL from a Pinecone source_file filename.

    Priority:
      1. Stored source_url in the course's source_files array (e.g. S3/PPTX uploads).
      2. YouTube pattern  — youtube_{VIDEO_ID}.txt
      3. Web scrape pattern — web_{domain_parts}.txt  (underscores → dots)
      4. None  — plain PDFs with no stored URL.
    """
    if not filename:
        return None

    # 1. Stored source_url
    for sf in source_files:
        if sf.get("filename") == filename and sf.get("source_url"):
            return sf["source_url"]

    # 2. YouTube: youtube_{VIDEO_ID}.txt
    if filename.startswith("youtube_") and filename.endswith(".txt"):
        video_id = filename[len("youtube_"):-len(".txt")]
        if video_id:
            return f"https://www.youtube.com/watch?v={video_id}"

    # 3. Web scrape: web_{domain_with_underscores}.txt → reconstruct domain
    if filename.startswith("web_") and filename.endswith(".txt"):
        domain_raw = filename[len("web_"):-len(".txt")]
        if domain_raw:
            return f"https://{'.'.join(domain_raw.split('_'))}"

    return None


class SpaceService:

    def __init__(self):
        self.profile_storage = LearningProfileStorage()
        self.conversation_storage = SpaceConversationStorage()

    # =========================================================================
    # Endpoint 1 — Add courses to a space (Pinecone metadata update)
    # =========================================================================

    async def add_courses_to_space(
        self,
        space_id: str,
        course_ids: List[str],
    ) -> Dict[str, Any]:
        """
        For each course_id, fetch all Pinecone vector IDs for that course and
        patch their metadata to record this space_id.

        All courses are processed in parallel. Returns a summary of updated counts.
        """
        from clients.qdrant_client import fetch_document_vector_ids, update_vector_metadata

        async def _update_single_course(course_id: str) -> Dict[str, Any]:
            try:
                vector_ids = await asyncio.to_thread(fetch_document_vector_ids, course_id)
                if not vector_ids:
                    logger.warning(f"add_courses_to_space: no vectors found for course {course_id}")
                    return {"course_id": course_id, "vectors_updated": 0, "status": "no_vectors"}

                updated = await asyncio.to_thread(
                    update_vector_metadata, vector_ids, {"nuton_space_id": space_id}
                )
                logger.info(
                    f"add_courses_to_space: updated {updated}/{len(vector_ids)} vectors for course {course_id}"
                )
                return {"course_id": course_id, "vectors_updated": updated, "status": "ok"}
            except Exception as e:
                logger.error(f"add_courses_to_space: failed for course {course_id}: {e}")
                return {"course_id": course_id, "vectors_updated": 0, "status": "error", "error": str(e)}

        results = await asyncio.gather(*[_update_single_course(cid) for cid in course_ids])
        total = sum(r["vectors_updated"] for r in results)
        return {
            "space_id": space_id,
            "courses_processed": len(course_ids),
            "total_vectors_updated": total,
            "results": list(results),
        }

    # =========================================================================
    # Endpoint 2 — Stream query across all courses in a space
    # =========================================================================

    async def query_space_stream(
        self,
        space_id: str,
        course_ids: List[str],
        query: str,
        user_id: str,
        model: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Streaming space-level Q&A across multiple courses.

        SSE event shapes:
          {"type": "status",       "phase": "searching",    "total_courses": N}
          {"type": "course_found", "course_id": "...",      "course_title": "...", "chunks_found": N}
          {"type": "status",       "phase": "synthesizing"}
          {"type": "status",       "phase": "no_results",   "falling_back_to_web": bool, "message": "..."}
          {"type": "token",        "text": "..."}
          {"type": "citations",    "sources": [...]}
          {"type": "done",         "from_web": bool,        "sources_used": N}
          {"type": "error",        "message": "..."}
        """
        model_config = ModelConfig.get_config(model)
        provider = model_config.get("provider", "anthropic")
        supports_search = model_config.get("supports_search", False) and provider == "anthropic"

        # ── Phase 0: embed query + fetch course titles + chat history (parallel) ──
        try:
            query_emb, courses_info, chat_history, raw_profile = await asyncio.gather(
                asyncio.to_thread(self._embed_query, query),
                asyncio.to_thread(self._fetch_courses_info, course_ids),
                asyncio.to_thread(self.conversation_storage.get_messages, space_id, user_id, 20),
                asyncio.to_thread(self.profile_storage.get_profile, user_id),
            )
        except Exception as e:
            yield _sse({"type": "error", "message": f"Setup failed: {e}"})
            return

        profile = raw_profile or {}

        # ── Phase 1: search each course in parallel, stream progress events ──
        yield _sse({"type": "status", "phase": "searching", "total_courses": len(course_ids)})

        queue: asyncio.Queue = asyncio.Queue()

        async def _search_one(course_id: str, course_title: str) -> None:
            try:
                results = await asyncio.to_thread(
                    self._search_course, query, query_emb, course_id, TOP_K_PER_COURSE
                )
                await queue.put((course_id, course_title, results, None))
            except Exception as e:
                await queue.put((course_id, course_title, [], str(e)))

        search_tasks = [
            asyncio.create_task(
                _search_one(cid, courses_info.get(cid, {}).get("title", cid))
            )
            for cid in course_ids
        ]

        course_contexts: List[Dict[str, Any]] = []
        all_citations: List[Dict[str, Any]] = []

        for _ in range(len(course_ids)):
            course_id, course_title, results, err = await queue.get()
            if err:
                logger.warning(f"query_space_stream: search failed for course {course_id}: {err}")
                continue
            if not results:
                continue

            relevant = [r for r in results if r.get("score", 0) >= RELEVANCE_THRESHOLD]
            if not relevant:
                continue

            chunks_for_prompt = []
            for r in relevant:
                meta = r.get("metadata", {})
                chunks_for_prompt.append({
                    "text": meta.get("text", ""),
                    "source_file": meta.get("source_file", ""),
                    "chapter_title": meta.get("chapter_title", ""),
                })
                all_citations.append({
                    "course_id": course_id,
                    "course_title": course_title,
                    "source_file": meta.get("source_file", ""),
                    "chapter_title": meta.get("chapter_title", ""),
                    "score": round(r.get("score", 0), 4),
                })

            course_contexts.append({
                "course_id": course_id,
                "course_title": course_title,
                "chunks": chunks_for_prompt,
            })

            yield _sse({
                "type": "course_found",
                "course_id": course_id,
                "course_title": course_title,
                "chunks_found": len(relevant),
            })

        # Ensure all search tasks are fully done before proceeding
        await asyncio.gather(*search_tasks, return_exceptions=True)

        # ── Phase 2: Generate answer ──
        full_response = ""
        from_web = False

        if course_contexts:
            yield _sse({"type": "status", "phase": "synthesizing"})
            prompt = build_space_query_prompt(
                query=query,
                course_contexts=course_contexts,
                profile=profile,
                chat_history=chat_history,
            )
            async for token in self._stream_response(prompt, model_config):
                full_response += token
                yield _sse({"type": "token", "text": token})

        else:
            from_web = True
            space_name = await asyncio.to_thread(self._get_space_name, space_id) or "your study space"
            fallback_msg = (
                "No relevant content found in your courses. Searching the web..."
                if supports_search
                else "No relevant content found in your courses. Answering from general knowledge..."
            )
            yield _sse({
                "type": "status",
                "phase": "no_results",
                "falling_back_to_web": supports_search,
                "message": fallback_msg,
            })
            prompt = build_space_web_fallback_prompt(
                query=query,
                profile=profile,
                chat_history=chat_history,
                space_name=space_name,
                has_web_search=supports_search,
            )
            # For Claude + web search: pass use_web_search=True so the tool is included.
            # For all other providers: stream normally without web search.
            async for token in self._stream_response(
                prompt, model_config, use_web_search=supports_search
            ):
                full_response += token
                yield _sse({"type": "token", "text": token})

        # ── Phase 3: Deduplicate + enrich citations ──
        deduped_citations: List[Dict[str, Any]] = []
        source_map: Dict[str, int] = {}
        if all_citations:
            # Collapse (course_id, source_file) duplicates, keeping highest-scoring chunk.
            seen: Dict[tuple, Dict[str, Any]] = {}
            for citation in all_citations:
                key = (citation["course_id"], citation.get("source_file", ""))
                if key not in seen or citation["score"] > seen[key]["score"]:
                    seen[key] = citation
            deduped_citations = list(seen.values())

            # Enrich each citation with clickable links.
            base_url = os.getenv("FRONTEND_URL", "https://nuton.app")
            for c in deduped_citations:
                course_data = courses_info.get(c["course_id"], {})
                slug = course_data.get("slug") or ""
                c["course_link"] = (
                    f"{base_url}/learn/{slug}?space={space_id}" if slug else None
                )
                c["source_link"] = _resolve_source_link(
                    c.get("source_file", ""),
                    course_data.get("source_files") or [],
                )

            # Build source_map: "[Source N]" number → deduped_citations index.
            # all_citations[i] corresponds to [Source i+1] in the LLM text.
            key_to_deduped_idx = {
                (c["course_id"], c.get("source_file", "")): i
                for i, c in enumerate(deduped_citations)
            }
            for i, raw in enumerate(all_citations):
                key = (raw["course_id"], raw.get("source_file", ""))
                idx = key_to_deduped_idx.get(key)
                if idx is not None:
                    source_map[str(i + 1)] = idx

        # ── Phase 4: Persist conversation + emit citations + done ──
        # Save BEFORE yielding `done` — code after the final yield in an async
        # generator is unreliable because the client may disconnect after `done`,
        # causing FastAPI to cancel the generator before post-yield code runs.
        now = datetime.now(timezone.utc).isoformat()
        messages_to_save = [
            {"role": "user", "content": query, "created_at": now},
            {
                "role": "assistant",
                "content": full_response,
                "sources": deduped_citations,
                "from_web": from_web,
                "created_at": now,
            },
        ]
        try:
            await asyncio.to_thread(
                self.conversation_storage.save_messages, space_id, user_id, messages_to_save
            )
        except Exception as e:
            logger.warning(f"Space conversation save failed (non-fatal): {e}")

        if deduped_citations:
            yield _sse({
                "type": "citations",
                "sources": deduped_citations,
                "source_map": source_map,
            })

        yield _sse({"type": "done", "from_web": from_web, "sources_used": len(deduped_citations)})

    # =========================================================================
    # Internal helpers
    # =========================================================================

    @staticmethod
    def _embed_query(query: str) -> List[float]:
        from clients.chonkie_client import embed_query_multimodal
        return embed_query_multimodal(query)["embedding"]

    @staticmethod
    def _search_course(
        query_text: str, query_emb: List[float], course_id: str, top_k: int
    ) -> List[Dict[str, Any]]:
        from clients.qdrant_client import hybrid_search, rerank_results
        results = hybrid_search(query_emb=query_emb, query_text=query_text, doc_id=course_id, top_k=top_k * 2)
        if not results:
            return []
        return rerank_results(query=query_text, hits=results, top_n=top_k)

    @staticmethod
    def _fetch_courses_info(course_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """Fetch course titles from Supabase. Returns {course_id: row_dict}."""
        try:
            response = (
                get_supabase()
                .table("courses")
                .select("id, title, topic, personalization_params, slug, source_files")
                .in_("id", course_ids)
                .execute()
            )
            return {row["id"]: row for row in (response.data or [])}
        except Exception as e:
            logger.error(f"_fetch_courses_info error: {e}")
            return {}

    @staticmethod
    def _get_space_name(space_id: str) -> Optional[str]:
        try:
            response = (
                get_supabase()
                .table("spaces_v2")
                .select("name")
                .eq("id", space_id)
                .limit(1)
                .execute()
            )
            if response.data:
                return response.data[0].get("name")
        except Exception as e:
            logger.error(f"_get_space_name error: {e}")
        return None

    # ── Streaming helpers (one per provider) ─────────────────────────────────

    @staticmethod
    async def _stream_response(
        prompt: str,
        model_config: Dict[str, Any],
        use_web_search: bool = False,
    ) -> AsyncGenerator[str, None]:
        """Dispatch to the correct streaming helper based on model provider."""
        provider = model_config.get("provider", "anthropic")
        if provider == "anthropic":
            async for token in SpaceService._stream_claude(prompt, model_config, use_web_search):
                yield token
        elif provider == "openai":
            async for token in SpaceService._stream_openai(prompt, model_config):
                yield token
        elif provider == "groq":
            async for token in SpaceService._stream_groq(prompt, model_config):
                yield token
        else:
            logger.error(f"_stream_response: unsupported provider '{provider}'")
            yield f"\n\n[Unsupported model provider: {provider}]"

    @staticmethod
    async def _stream_claude(
        prompt: str,
        model_config: Dict[str, Any],
        use_web_search: bool = False,
    ) -> AsyncGenerator[str, None]:
        """Stream tokens from Claude (Anthropic async client)."""
        import anthropic
        client = anthropic.AsyncAnthropic()
        kwargs: Dict[str, Any] = {
            "model": model_config.get("model", "claude-haiku-4-5-20251001"),
            "max_tokens": model_config.get("max_tokens", 2048),
            "messages": [{"role": "user", "content": prompt}],
        }
        # Only add web_search tool for Claude models that support it
        if use_web_search and model_config.get("supports_search"):
            kwargs["tools"] = [{"type": "web_search_20250305", "name": "web_search"}]
        try:
            async with client.messages.stream(**kwargs) as stream:
                async for text in stream.text_stream:
                    yield text
        except Exception as e:
            logger.error(f"_stream_claude error: {e}")
            yield f"\n\n[Error generating response: {e}]"

    @staticmethod
    async def _stream_openai(
        prompt: str,
        model_config: Dict[str, Any],
    ) -> AsyncGenerator[str, None]:
        """Stream tokens from OpenAI (async client)."""
        from openai import AsyncOpenAI
        client = AsyncOpenAI()
        try:
            stream = await client.chat.completions.create(
                model=model_config.get("model", "gpt-4o"),
                max_tokens=model_config.get("max_tokens", 2048),
                messages=[{"role": "user", "content": prompt}],
                stream=True,
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        except Exception as e:
            logger.error(f"_stream_openai error: {e}")
            yield f"\n\n[Error generating response: {e}]"

    @staticmethod
    async def _stream_groq(
        prompt: str,
        model_config: Dict[str, Any],
    ) -> AsyncGenerator[str, None]:
        """Stream tokens from Groq (async client)."""
        from groq import AsyncGroq
        client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))
        try:
            stream = await client.chat.completions.create(
                model=model_config.get("model", "meta-llama/llama-4-scout-17b-16e-instruct"),
                max_tokens=model_config.get("max_tokens", 2048),
                messages=[{"role": "user", "content": prompt}],
                stream=True,
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        except Exception as e:
            logger.error(f"_stream_groq error: {e}")
            yield f"\n\n[Error generating response: {e}]"
