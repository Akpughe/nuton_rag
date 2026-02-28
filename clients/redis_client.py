"""
Async Redis client for course chat history caching.
Provides get/push/clear operations with graceful fallback on Redis failure.
"""

import os
import json
import asyncio
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_redis_client = None
_redis_available = None
_redis_lock = asyncio.Lock()

CHAT_KEY_PREFIX = "chat"
CHAT_TTL_SECONDS = 86400  # 24 hours
MAX_CACHED_MESSAGES = 20


async def _get_redis():
    """Lazy-init async Redis connection singleton with lock to prevent race conditions."""
    global _redis_client, _redis_available
    if _redis_available is False:
        return None
    if _redis_client is not None:
        return _redis_client
    async with _redis_lock:
        # Double-check after acquiring lock
        if _redis_client is not None:
            return _redis_client
        if _redis_available is False:
            return None
        try:
            import redis.asyncio as aioredis
            url = os.getenv("REDIS_URL", "redis://localhost:6379")
            _redis_client = aioredis.from_url(url, decode_responses=True)
            await _redis_client.ping()
            _redis_available = True
            logger.info(f"Redis connected: {url}")
            return _redis_client
        except Exception as e:
            logger.warning(f"Redis unavailable, chat will use DB only: {e}")
            _redis_available = False
            _redis_client = None
            return None


def _chat_key(course_id: str, user_id: str) -> str:
    return f"{CHAT_KEY_PREFIX}:{course_id}:{user_id}"


async def get_chat_history(course_id: str, user_id: str, limit: int = MAX_CACHED_MESSAGES) -> Optional[List[Dict]]:
    """Get cached chat history. Returns None on miss or Redis failure."""
    try:
        r = await _get_redis()
        if r is None:
            return None
        key = _chat_key(course_id, user_id)
        raw = await r.lrange(key, -limit, -1)
        if not raw:
            return None
        return [json.loads(msg) for msg in raw]
    except Exception as e:
        logger.warning(f"Redis get_chat_history failed: {e}")
        return None


async def push_messages(course_id: str, user_id: str, messages: List[Dict]) -> bool:
    """Push messages to cache and trim to MAX_CACHED_MESSAGES."""
    try:
        r = await _get_redis()
        if r is None:
            return False
        key = _chat_key(course_id, user_id)
        pipe = r.pipeline()
        for msg in messages:
            pipe.rpush(key, json.dumps(msg, default=str))
        pipe.ltrim(key, -MAX_CACHED_MESSAGES, -1)
        pipe.expire(key, CHAT_TTL_SECONDS)
        await pipe.execute()
        return True
    except Exception as e:
        logger.warning(f"Redis push_messages failed: {e}")
        return False


async def clear_chat(course_id: str, user_id: str) -> bool:
    """Clear cached chat history."""
    try:
        r = await _get_redis()
        if r is None:
            return False
        await r.delete(_chat_key(course_id, user_id))
        return True
    except Exception as e:
        logger.warning(f"Redis clear_chat failed: {e}")
        return False
