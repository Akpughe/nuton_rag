"""
Debug script to test different Jina API configurations for image embedding.
"""

import os
import requests
import json

# Create a tiny test image (1x1 red pixel PNG)
TINY_PNG_BASE64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="

JINA_ENDPOINT = "https://api.jina.ai/v1/embeddings"
JINA_API_KEY = os.getenv("JINA_API_KEY")

def test_config(config_name, payload):
    """Test a specific configuration."""
    print(f"\n{'='*80}")
    print(f"Testing: {config_name}")
    print(f"{'='*80}")
    print(f"Payload: {json.dumps(payload, indent=2)[:500]}")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {JINA_API_KEY}"
    }

    try:
        response = requests.post(JINA_ENDPOINT, headers=headers, json=payload, timeout=30)

        print(f"\nStatus: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print(f"✅ SUCCESS!")
            print(f"   Embeddings returned: {len(result.get('data', []))}")
            if result.get('data'):
                print(f"   Embedding dimension: {len(result['data'][0].get('embedding', []))}")
            return True
        else:
            print(f"❌ FAILED")
            print(f"   Error response: {response.text[:500]}")
            return False

    except Exception as e:
        print(f"❌ EXCEPTION: {e}")
        return False


def main():
    print("\n🔍 JINA IMAGE EMBEDDING DEBUG - Systematic Testing\n")

    if not JINA_API_KEY:
        print("❌ Error: JINA_API_KEY not found")
        return

    img_data_uri = f"data:image/png;base64,{TINY_PNG_BASE64}"

    # Test 1: Text embedding (baseline - should work)
    test_config("Text embedding (baseline)", {
        "model": "jina-clip-v2",
        "input": ["A cat on a mat"]
    })

    # Test 2: Image with minimal payload
    test_config("Image - minimal payload", {
        "model": "jina-clip-v2",
        "input": [img_data_uri]
    })

    # Test 3: Image with input_type
    test_config("Image - with input_type", {
        "model": "jina-clip-v2",
        "input": [img_data_uri],
        "input_type": "image"
    })

    # Test 4: Image with task parameter
    test_config("Image - with task=retrieval.passage", {
        "model": "jina-clip-v2",
        "input": [img_data_uri],
        "task": "retrieval.passage"
    })

    # Test 5: Try without data URI prefix
    test_config("Image - raw base64", {
        "model": "jina-clip-v2",
        "input": [TINY_PNG_BASE64]
    })

    # Test 6: Try different model identifier
    test_config("Image - model: jina-clip-v1", {
        "model": "jina-clip-v1",
        "input": [img_data_uri]
    })

    # Test 7: Mixed text and image
    test_config("Mixed - text + image", {
        "model": "jina-clip-v2",
        "input": ["A cat on a mat", img_data_uri]
    })

    # Test 8: URL to public image
    test_config("Image - URL", {
        "model": "jina-clip-v2",
        "input": ["https://picsum.photos/200"]
    })

    print(f"\n{'='*80}")
    print("Testing complete")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
