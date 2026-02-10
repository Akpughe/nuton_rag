"""
S3 client for uploading source files (PDFs, etc.) for persistent storage.
Uploads run in the background so they don't block course generation.
"""

import os
import logging
import asyncio
from typing import Optional

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

# Config from environment
S3_BUCKET = os.getenv("S3_BUCKET", "nuton-source-files")
S3_REGION = os.getenv("AWS_REGION", "eu-west-1")
S3_PREFIX = os.getenv("S3_PREFIX", "sources")

_s3_client = None


def _get_s3_client():
    global _s3_client
    if _s3_client is None:
        _s3_client = boto3.client("s3", region_name=S3_REGION)
    return _s3_client


def get_s3_url(key: str) -> str:
    """Pre-compute the public S3 URL for a given key."""
    return f"https://{S3_BUCKET}.s3.{S3_REGION}.amazonaws.com/{key}"


def build_s3_key(file_id: str, filename: str) -> str:
    """Build a deterministic S3 key: sources/{file_id}/{filename}"""
    return f"{S3_PREFIX}/{file_id}/{filename}"


def upload_bytes_to_s3(key: str, data: bytes, content_type: str = "application/octet-stream") -> bool:
    """Synchronous S3 upload. Returns True on success."""
    try:
        _get_s3_client().put_object(
            Bucket=S3_BUCKET,
            Key=key,
            Body=data,
            ContentType=content_type,
        )
        logger.info(f"S3 upload success: {key} ({len(data)} bytes)")
        return True
    except ClientError as e:
        logger.error(f"S3 upload failed for {key}: {e}")
        return False
    except Exception as e:
        logger.error(f"S3 upload unexpected error for {key}: {e}")
        return False


async def upload_bytes_to_s3_async(key: str, data: bytes, content_type: str = "application/octet-stream") -> bool:
    """Async wrapper — runs the sync upload in a thread."""
    return await asyncio.to_thread(upload_bytes_to_s3, key, data, content_type)


def fire_and_forget_upload(key: str, data: bytes, content_type: str = "application/octet-stream"):
    """
    Start an S3 upload as a fire-and-forget background task.
    Does not block the caller. Logs errors but never raises.
    """
    try:
        loop = asyncio.get_running_loop()
        task = loop.create_task(upload_bytes_to_s3_async(key, data, content_type))
        task.add_done_callback(_log_upload_result)
    except RuntimeError:
        # No running loop — fall back to thread
        import threading
        t = threading.Thread(target=upload_bytes_to_s3, args=(key, data, content_type), daemon=True)
        t.start()


def _log_upload_result(task: asyncio.Task):
    """Callback for fire-and-forget upload tasks."""
    try:
        result = task.result()
        if not result:
            logger.warning("Background S3 upload returned failure")
    except Exception as e:
        logger.error(f"Background S3 upload exception: {e}")


# Content type mapping
CONTENT_TYPES = {
    ".pdf": "application/pdf",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".ppt": "application/vnd.ms-powerpoint",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".doc": "application/msword",
    ".txt": "text/plain",
    ".md": "text/markdown",
}


def get_content_type(filename: str) -> str:
    """Get content type from filename extension."""
    ext = os.path.splitext(filename)[1].lower() if filename else ""
    return CONTENT_TYPES.get(ext, "application/octet-stream")
