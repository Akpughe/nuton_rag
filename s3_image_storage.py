"""
S3 Image Storage - Upload and manage images for multimodal embeddings.

This module handles uploading document images to AWS S3 and generating
public URLs that can be used with Jina CLIP v2 embeddings API.

Author: RAG System Integration
Date: 2025
"""

import os
import io
import base64
import hashlib
import logging
from typing import List, Dict, Any, Optional
import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


class S3ImageStorage:
    """
    Upload and manage images in AWS S3 for multimodal embeddings.

    Features:
    - Upload base64-encoded images to S3
    - Generate public URLs for Jina CLIP v2 API
    - Organize by document ID and space ID
    - Handle image deduplication via content hashing
    """

    def __init__(
        self,
        bucket_name: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_region: Optional[str] = None
    ):
        """
        Initialize S3 image storage.

        Args:
            bucket_name: S3 bucket name (defaults to env var)
            aws_access_key_id: AWS access key (defaults to env var)
            aws_secret_access_key: AWS secret key (defaults to env var)
            aws_region: AWS region (defaults to env var)
        """
        self.bucket_name = bucket_name or os.getenv("AWS_S3_BUCKET", "nuton-rag-images")
        self.aws_region = aws_region or os.getenv("AWS_REGION", "us-east-1")

        # Initialize S3 client
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_access_key_id or os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=aws_secret_access_key or os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=self.aws_region
        )

        logger.info(f"Initialized S3 image storage: bucket={self.bucket_name}, region={self.aws_region}")

        # Ensure bucket exists
        self._ensure_bucket_exists()

    def _ensure_bucket_exists(self):
        """Create bucket if it doesn't exist."""
        try:
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            logger.info(f"✅ S3 bucket '{self.bucket_name}' exists")
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                logger.info(f"Creating S3 bucket: {self.bucket_name}")
                try:
                    if self.aws_region == 'us-east-1':
                        # us-east-1 doesn't need LocationConstraint
                        self.s3_client.create_bucket(Bucket=self.bucket_name)
                    else:
                        self.s3_client.create_bucket(
                            Bucket=self.bucket_name,
                            CreateBucketConfiguration={'LocationConstraint': self.aws_region}
                        )

                    # Make bucket public-read for images
                    self.s3_client.put_bucket_acl(
                        Bucket=self.bucket_name,
                        ACL='public-read'
                    )

                    logger.info(f"✅ Created S3 bucket: {self.bucket_name}")
                except ClientError as create_error:
                    logger.error(f"Failed to create bucket: {create_error}")
                    raise
            else:
                raise

    def _get_content_hash(self, image_data: bytes) -> str:
        """Generate SHA256 hash of image content for deduplication."""
        return hashlib.sha256(image_data).hexdigest()[:16]

    def upload_image(
        self,
        image_base64: str,
        document_id: str,
        image_id: str,
        space_id: Optional[str] = None
    ) -> str:
        """
        Upload a single image to S3 and return public URL.

        Args:
            image_base64: Base64-encoded image (with or without data URI prefix)
            document_id: Document ID
            image_id: Unique image identifier
            space_id: Optional space ID for organization

        Returns:
            Public URL to the uploaded image
        """
        # Strip data URI prefix if present
        if image_base64.startswith('data:'):
            # Format: data:image/jpeg;base64,<data>
            image_base64 = image_base64.split(',', 1)[1]

        # Decode base64 to bytes
        try:
            image_bytes = base64.b64decode(image_base64)
        except Exception as e:
            logger.error(f"Failed to decode base64 image: {e}")
            raise ValueError(f"Invalid base64 image data: {e}")

        # Generate content hash for deduplication
        content_hash = self._get_content_hash(image_bytes)

        # Determine image format from bytes
        image_format = self._detect_image_format(image_bytes)

        # Generate S3 key (path)
        if space_id:
            s3_key = f"images/{space_id}/{document_id}/{image_id}_{content_hash}.{image_format}"
        else:
            s3_key = f"images/{document_id}/{image_id}_{content_hash}.{image_format}"

        # Upload to S3
        try:
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=image_bytes,
                ContentType=f'image/{image_format}',
                ACL='public-read'  # Make image publicly accessible
            )

            # Generate public URL
            url = f"https://{self.bucket_name}.s3.{self.aws_region}.amazonaws.com/{s3_key}"

            logger.debug(f"Uploaded image: {s3_key} -> {url}")
            return url

        except ClientError as e:
            logger.error(f"Failed to upload image to S3: {e}")
            raise

    def upload_images_batch(
        self,
        images: List[Dict[str, Any]],
        document_id: str,
        space_id: Optional[str] = None
    ) -> List[str]:
        """
        Upload multiple images to S3 and return list of URLs.

        Args:
            images: List of image dicts with 'image_base64' and 'id' keys
            document_id: Document ID
            space_id: Optional space ID

        Returns:
            List of public URLs (same order as input)
        """
        urls = []

        for img in images:
            image_base64 = img.get('image_base64')
            image_id = img.get('id', img.get('image_id', f"img_{len(urls)}"))

            if not image_base64:
                logger.warning(f"Skipping image {image_id}: no base64 data")
                continue

            try:
                url = self.upload_image(
                    image_base64=image_base64,
                    document_id=document_id,
                    image_id=image_id,
                    space_id=space_id
                )
                urls.append(url)
            except Exception as e:
                logger.error(f"Failed to upload image {image_id}: {e}")
                # Continue with other images
                continue

        logger.info(f"✅ Uploaded {len(urls)}/{len(images)} images to S3")
        return urls

    def _detect_image_format(self, image_bytes: bytes) -> str:
        """Detect image format from bytes (magic numbers)."""
        # Check magic numbers
        if image_bytes.startswith(b'\xff\xd8\xff'):
            return 'jpeg'
        elif image_bytes.startswith(b'\x89PNG'):
            return 'png'
        elif image_bytes.startswith(b'GIF87a') or image_bytes.startswith(b'GIF89a'):
            return 'gif'
        elif image_bytes.startswith(b'RIFF') and b'WEBP' in image_bytes[:12]:
            return 'webp'
        else:
            # Default to jpeg if unknown
            logger.warning("Unknown image format, defaulting to jpeg")
            return 'jpeg'

    def delete_document_images(
        self,
        document_id: str,
        space_id: Optional[str] = None
    ) -> int:
        """
        Delete all images for a document from S3.

        Args:
            document_id: Document ID
            space_id: Optional space ID

        Returns:
            Number of images deleted
        """
        # Determine prefix
        if space_id:
            prefix = f"images/{space_id}/{document_id}/"
        else:
            prefix = f"images/{document_id}/"

        # List objects with prefix
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix
            )

            if 'Contents' not in response:
                logger.info(f"No images found for document {document_id}")
                return 0

            # Delete objects
            objects_to_delete = [{'Key': obj['Key']} for obj in response['Contents']]

            self.s3_client.delete_objects(
                Bucket=self.bucket_name,
                Delete={'Objects': objects_to_delete}
            )

            deleted_count = len(objects_to_delete)
            logger.info(f"Deleted {deleted_count} images for document {document_id}")
            return deleted_count

        except ClientError as e:
            logger.error(f"Failed to delete images: {e}")
            raise


# Convenience functions
def upload_images_to_s3(
    images: List[Dict[str, Any]],
    document_id: str,
    space_id: Optional[str] = None
) -> List[str]:
    """
    Quick function to upload images to S3 and get URLs.

    Args:
        images: List of image dicts with 'image_base64'
        document_id: Document ID
        space_id: Optional space ID

    Returns:
        List of public URLs
    """
    storage = S3ImageStorage()
    return storage.upload_images_batch(images, document_id, space_id)


# Testing
if __name__ == "__main__":
    print(f"\n{'='*80}")
    print(f"Testing S3 Image Storage")
    print(f"{'='*80}\n")

    # Create a tiny test image (1x1 red pixel PNG)
    tiny_png_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="

    test_images = [
        {
            'id': 'test_image_1',
            'image_base64': f"data:image/png;base64,{tiny_png_base64}"
        }
    ]

    try:
        storage = S3ImageStorage()
        urls = storage.upload_images_batch(
            images=test_images,
            document_id="test_doc_123",
            space_id="test_space"
        )

        print(f"✅ Upload successful!")
        print(f"   URLs: {urls}")

    except Exception as e:
        print(f"❌ Upload failed: {e}")

    print(f"\n{'='*80}\n")
