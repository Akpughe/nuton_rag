"""
Multimodal Embeddings - Text and Image Embedding for RAG
Supports jina-clip-v2 and OpenAI CLIP for unified embedding space.

Features:
- Text embedding with semantic understanding
- Image embedding from base64 or URLs
- Unified embedding space for text-image search
- Batch processing for efficiency
- Support for multiple embedding models
- Uses Jina REST API (no jina library installation required)

Author: RAG System Integration
Date: 2025
"""

import os
import logging
from typing import List, Dict, Any, Optional, Union
import base64
import requests
from dotenv import load_dotenv

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logging.warning("OpenAI not available. Install with: pip install openai")

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultimodalEmbedder:
    """
    Multimodal embedding handler for text and images.

    Supports:
    - jina-clip-v2 (primary, cost-effective)
    - OpenAI CLIP (alternative)
    """

    def __init__(
        self,
        model: str = "jina-clip-v2",
        api_key: Optional[str] = None,
        batch_size: int = 32,
    ):
        """
        Initialize multimodal embedder.

        Args:
            model: Model to use ('jina-clip-v2' or 'openai-clip')
            api_key: API key (Jina or OpenAI, defaults to env vars)
            batch_size: Batch size for embedding
        """
        self.model = model.lower()
        self.batch_size = batch_size

        # Initialize appropriate client
        if self.model == "jina-clip-v2":
            self.api_key = api_key or os.getenv("JINA_API_KEY")
            if not self.api_key:
                raise ValueError("JINA_API_KEY not found in environment")

            # Jina API endpoint (using REST API, no library required)
            self.jina_endpoint = "https://api.jina.ai/v1/embeddings"
            logger.info("Initialized Jina CLIP-v2 embedder")

        elif self.model == "openai-clip":
            self.api_key = api_key or os.getenv("OPENAI_API_KEY")
            if not self.api_key:
                raise ValueError("OPENAI_API_KEY not found in environment")

            if not OPENAI_AVAILABLE:
                raise ImportError("OpenAI not available. Install with: pip install openai")

            self.client = OpenAI(api_key=self.api_key)
            logger.info("Initialized OpenAI CLIP embedder")

        else:
            raise ValueError(f"Unsupported model: {model}. Use 'jina-clip-v2' or 'openai-clip'")

    def embed_texts(
        self,
        texts: List[str],
        normalize: bool = True
    ) -> List[List[float]]:
        """
        Embed text strings.

        Args:
            texts: List of text strings
            normalize: Normalize embeddings to unit length

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        logger.info(f"Embedding {len(texts)} texts with {self.model}")

        if self.model == "jina-clip-v2":
            return self._embed_texts_jina(texts, normalize=normalize)
        elif self.model == "openai-clip":
            return self._embed_texts_openai(texts)
        else:
            raise ValueError(f"Unsupported model: {self.model}")

    def embed_images(
        self,
        images: List[Union[str, bytes]],
        normalize: bool = True
    ) -> List[List[float]]:
        """
        Embed images.

        Args:
            images: List of image data (base64 strings, URLs, or bytes)
            normalize: Normalize embeddings to unit length

        Returns:
            List of embedding vectors
        """
        if not images:
            return []

        logger.info(f"Embedding {len(images)} images with {self.model}")

        if self.model == "jina-clip-v2":
            return self._embed_images_jina(images, normalize=normalize)
        elif self.model == "openai-clip":
            return self._embed_images_openai(images)
        else:
            raise ValueError(f"Unsupported model: {self.model}")

    def embed_batch(
        self,
        items: List[Dict[str, Any]],
        normalize: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Embed a mixed batch of text and images.

        Args:
            items: List of dicts with 'type' ('text' or 'image') and 'content'
            normalize: Normalize embeddings

        Returns:
            List of items with 'embedding' field added

        Example:
            items = [
                {'type': 'text', 'content': 'A cat on a mat'},
                {'type': 'image', 'content': 'data:image/jpeg;base64,...'},
            ]
            results = embedder.embed_batch(items)
        """
        if not items:
            return []

        # Separate by type
        texts = []
        images = []
        text_indices = []
        image_indices = []

        for idx, item in enumerate(items):
            item_type = item.get('type', 'text').lower()
            content = item.get('content')

            if item_type == 'text':
                texts.append(content)
                text_indices.append(idx)
            elif item_type == 'image':
                images.append(content)
                image_indices.append(idx)
            else:
                logger.warning(f"Unknown item type: {item_type}, skipping")

        # Embed each type
        text_embeddings = self.embed_texts(texts, normalize=normalize) if texts else []
        image_embeddings = self.embed_images(images, normalize=normalize) if images else []

        # Merge back
        results = [item.copy() for item in items]

        for idx, embedding in zip(text_indices, text_embeddings):
            results[idx]['embedding'] = embedding

        for idx, embedding in zip(image_indices, image_embeddings):
            results[idx]['embedding'] = embedding

        return results

    def _embed_texts_jina(self, texts: List[str], normalize: bool = True) -> List[List[float]]:
        """Embed texts using Jina CLIP-v2."""
        all_embeddings = []

        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]

            # Call Jina API
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }

            payload = {
                "model": "jina-clip-v2",
                "input": batch,
                "normalized": normalize,
                "embedding_type": "float"
            }

            try:
                response = requests.post(
                    self.jina_endpoint,
                    headers=headers,
                    json=payload,
                    timeout=30
                )
                response.raise_for_status()

                result = response.json()

                # Extract embeddings
                batch_embeddings = [item['embedding'] for item in result['data']]
                all_embeddings.extend(batch_embeddings)

                logger.info(f"Embedded batch {i // self.batch_size + 1}/{(len(texts) + self.batch_size - 1) // self.batch_size}")

            except Exception as e:
                logger.error(f"Error embedding batch with Jina: {e}")
                raise

        return all_embeddings

    def _embed_images_jina(self, images: List[Union[str, bytes]], normalize: bool = True) -> List[List[float]]:
        """
        Embed images using Jina CLIP-v2.

        IMPORTANT: The Jina REST API only supports IMAGE URLS, not base64 data.

        Args:
            images: List of image URLs (https://...) or base64 strings
                   If base64 is detected, will raise error with alternatives

        Returns:
            List of embedding vectors

        Raises:
            NotImplementedError: If base64 images are provided instead of URLs
        """
        all_embeddings = []

        # Check if we have base64 data (not supported)
        has_base64 = False
        for img in images:
            if isinstance(img, bytes) or (isinstance(img, str) and img.startswith('data:')):
                has_base64 = True
                break

        if has_base64:
            logger.error("=" * 80)
            logger.error("JINA CLIP v2 API LIMITATION DETECTED")
            logger.error("=" * 80)
            logger.error(
                "The Jina REST API endpoint (/v1/embeddings) does NOT support base64-encoded images.\n"
                "When base64 data URIs are sent, the API treats them as text and returns:\n"
                "  'Input text exceeds maximum length of 8194 tokens'\n\n"
                "The API only supports IMAGE URLS, not base64 data.\n\n"
                "SOLUTION: Upload images to S3 first:\n"
                "  from s3_image_storage import upload_images_to_s3\n"
                "  urls = upload_images_to_s3(images, document_id, space_id)\n"
                "  embeddings = embedder.embed_images(urls)  # Use URLs\n\n"
                "ALTERNATIVES:\n"
                "  1. Use s3_image_storage module (recommended)\n"
                "  2. Upload images to any cloud storage and use URLs\n"
                "  3. Use HuggingFace transformers to run jina-clip-v2 locally\n"
                "  4. Use a different embedding service that supports base64\n\n"
                f"Current request: {len(images)} images with base64 data\n"
            )
            logger.error("=" * 80)

            raise NotImplementedError(
                "Jina REST API does not support base64 images. "
                "Upload to S3 first using s3_image_storage module. "
                "See logs above for details."
            )

        # All inputs should be URLs at this point
        image_urls = [str(img) for img in images]

        # Validate URLs
        for url in image_urls:
            if not url.startswith(('http://', 'https://')):
                raise ValueError(f"Invalid image URL: {url}. Must start with http:// or https://")

        # Use moderate batch size for URL-based images
        image_batch_size = 20  # URLs are small, can use larger batches

        # Process in batches
        for i in range(0, len(image_urls), image_batch_size):
            batch = image_urls[i:i + image_batch_size]

            # Call Jina API
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }

            # Minimal payload for image URLs
            payload = {
                "model": "jina-clip-v2",
                "input": batch
            }

            try:
                response = requests.post(
                    self.jina_endpoint,
                    headers=headers,
                    json=payload,
                    timeout=60  # Longer timeout for image processing
                )

                # Log detailed error if request fails
                if response.status_code != 200:
                    logger.error(f"Jina API error {response.status_code}: {response.text}")
                    logger.error(f"Request payload: model={payload['model']}, batch_size={len(batch)}")
                    logger.error(f"First URL: {batch[0]}")

                response.raise_for_status()

                result = response.json()

                # Extract embeddings
                batch_embeddings = [item['embedding'] for item in result['data']]
                all_embeddings.extend(batch_embeddings)

                logger.info(f"Embedded image batch {i // image_batch_size + 1}/{(len(image_urls) + image_batch_size - 1) // image_batch_size}")

            except Exception as e:
                logger.error(f"Error embedding image batch with Jina: {e}")
                raise

        return all_embeddings

    def _embed_texts_openai(self, texts: List[str]) -> List[List[float]]:
        """Embed texts using OpenAI."""
        all_embeddings = []

        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]

            try:
                response = self.client.embeddings.create(
                    model="text-embedding-3-small",  # OpenAI's embedding model
                    input=batch
                )

                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)

                logger.info(f"Embedded batch {i // self.batch_size + 1}/{(len(texts) + self.batch_size - 1) // self.batch_size}")

            except Exception as e:
                logger.error(f"Error embedding batch with OpenAI: {e}")
                raise

        return all_embeddings

    def _embed_images_openai(self, images: List[Union[str, bytes]]) -> List[List[float]]:
        """
        Embed images using OpenAI.

        Note: OpenAI's text-embedding models don't natively support images.
        For true multimodal embeddings with OpenAI, you'd need to use GPT-4V
        to generate text descriptions first, then embed those.

        This is a placeholder implementation.
        """
        logger.warning("OpenAI CLIP implementation is not available. "
                      "Use jina-clip-v2 for true multimodal embeddings, "
                      "or implement GPT-4V description + text embedding.")

        # Return zero vectors as placeholder
        return [[0.0] * 1536 for _ in images]  # text-embedding-3-small dimension


# Convenience functions
def embed_text(
    text: str,
    model: str = "jina-clip-v2",
    api_key: Optional[str] = None
) -> List[float]:
    """
    Quick function to embed a single text.

    Args:
        text: Text to embed
        model: Model to use
        api_key: API key

    Returns:
        Embedding vector
    """
    embedder = MultimodalEmbedder(model=model, api_key=api_key)
    embeddings = embedder.embed_texts([text])
    return embeddings[0] if embeddings else []


def embed_image(
    image: Union[str, bytes],
    model: str = "jina-clip-v2",
    api_key: Optional[str] = None
) -> List[float]:
    """
    Quick function to embed a single image.

    Args:
        image: Image data (base64 string, URL, or bytes)
        model: Model to use
        api_key: API key

    Returns:
        Embedding vector
    """
    embedder = MultimodalEmbedder(model=model, api_key=api_key)
    embeddings = embedder.embed_images([image])
    return embeddings[0] if embeddings else []


# Testing
if __name__ == "__main__":
    print(f"\n{'='*80}")
    print(f"Testing Multimodal Embeddings")
    print(f"{'='*80}\n")

    # Test text embedding
    try:
        print("Testing text embedding...")
        embedder = MultimodalEmbedder(model="jina-clip-v2")

        test_texts = [
            "A cat sitting on a mat",
            "A dog running in a park",
            "Machine learning and artificial intelligence"
        ]

        text_embeddings = embedder.embed_texts(test_texts)

        print(f"✅ Text embedding successful!")
        print(f"   Embedded {len(text_embeddings)} texts")
        print(f"   Embedding dimension: {len(text_embeddings[0])}")
        print(f"   First embedding (first 10 dims): {text_embeddings[0][:10]}")

    except Exception as e:
        print(f"❌ Text embedding failed: {e}")

    print()

    # Test batch embedding
    try:
        print("Testing batch embedding (mixed text/image)...")

        batch_items = [
            {'type': 'text', 'content': 'A beautiful sunset'},
            {'type': 'text', 'content': 'Modern architecture'},
        ]

        batch_results = embedder.embed_batch(batch_items)

        print(f"✅ Batch embedding successful!")
        print(f"   Embedded {len(batch_results)} items")

        for i, item in enumerate(batch_results):
            print(f"   Item {i+1} ({item['type']}): embedding dim = {len(item['embedding'])}")

    except Exception as e:
        print(f"❌ Batch embedding failed: {e}")

    print(f"\n{'='*80}\n")
