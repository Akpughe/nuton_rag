#!/usr/bin/env python3
"""
Example usage of the YouTube Transcript Service.

This script demonstrates how to use the YouTubeTranscriptService to extract
transcripts from YouTube videos with optional WebShare proxy support.
"""

import os
from services.youtube_transcript_service import YouTubeTranscriptService


def example_basic_usage():
    """Example: Basic transcript extraction without proxy."""
    print("=" * 60)
    print("Example 1: Basic Transcript Extraction (No Proxy)")
    print("=" * 60)

    # Initialize service without proxy
    yt_service = YouTubeTranscriptService(use_proxy=False)

    # Extract transcript from a video
    video_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    result = yt_service.get_transcript(video_url)

    if result['success']:
        print(f"✅ Success!")
        print(f"Video ID: {result['video_id']}")
        print(f"Language: {result['language']}")
        print(f"Thumbnail: {result['thumbnail']}")
        print(f"\nTranscript (first 500 chars):")
        print(result['text'][:500] + "...")
    else:
        print(f"❌ Error: {result['message']}")

    print()


def example_with_proxy():
    """Example: Transcript extraction with WebShare proxy (for cloud deployments)."""
    print("=" * 60)
    print("Example 2: Transcript Extraction with WebShare Proxy")
    print("=" * 60)

    # Set WebShare credentials (you need to set these in your environment)
    # os.environ['WEBSHARE_PROXY_USERNAME'] = 'your-username'
    # os.environ['WEBSHARE_PROXY_PASSWORD'] = 'your-password'

    # Initialize service with proxy
    yt_service = YouTubeTranscriptService(use_proxy=True)

    video_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    result = yt_service.get_transcript(video_url)

    if result['success']:
        print(f"✅ Success with proxy!")
        print(f"Video ID: {result['video_id']}")
        print(f"Transcript length: {len(result['text'])} characters")
    else:
        print(f"❌ Error: {result['message']}")

    print()


def example_multiple_languages():
    """Example: Request transcript in multiple languages with fallback."""
    print("=" * 60)
    print("Example 3: Multi-language Transcript Extraction")
    print("=" * 60)

    yt_service = YouTubeTranscriptService(use_proxy=False)

    video_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

    # Try Spanish first, then English as fallback
    result = yt_service.get_transcript(video_url, languages=['es', 'en'])

    if result['success']:
        print(f"✅ Transcript retrieved in: {result['language']}")
        print(f"Video ID: {result['video_id']}")
    else:
        print(f"❌ Error: {result['message']}")

    print()


def example_available_transcripts():
    """Example: Get list of available transcript languages."""
    print("=" * 60)
    print("Example 4: List Available Transcripts")
    print("=" * 60)

    yt_service = YouTubeTranscriptService(use_proxy=False)

    video_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    result = yt_service.get_available_transcripts(video_url)

    if result['success']:
        print(f"✅ Available transcripts for video: {result['video_id']}")
        print("\nLanguages:")
        for transcript in result['available_transcripts']:
            status = "🤖 Auto-generated" if transcript['is_generated'] else "👤 Manual"
            print(f"  - {transcript['language']} ({transcript['language_code']}) {status}")
    else:
        print(f"❌ Error: {result['message']}")

    print()


def example_video_info():
    """Example: Get basic video information."""
    print("=" * 60)
    print("Example 5: Get Video Information")
    print("=" * 60)

    yt_service = YouTubeTranscriptService(use_proxy=False)

    video_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    result = yt_service.get_video_info(video_url)

    if result['success']:
        print(f"✅ Video Information:")
        print(f"  Video ID: {result['video_id']}")
        print(f"  Video URL: {result['video_url']}")
        print(f"  Thumbnail: {result['thumbnail']}")
        print(f"  HD Thumbnail: {result['thumbnail_maxres']}")
    else:
        print(f"❌ Error: {result['message']}")

    print()


def example_error_handling():
    """Example: Proper error handling for different scenarios."""
    print("=" * 60)
    print("Example 6: Error Handling")
    print("=" * 60)

    yt_service = YouTubeTranscriptService(use_proxy=False)

    # Test cases that might fail
    test_cases = [
        ("Invalid URL", "not-a-valid-url"),
        ("Invalid Video ID", "invalid-video-id-123"),
        ("Valid Video", "https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    ]

    for name, url in test_cases:
        print(f"\nTesting: {name}")
        result = yt_service.get_transcript(url)
        if result['success']:
            print(f"  ✅ Success - got {len(result['text'])} characters")
        else:
            print(f"  ❌ Failed - {result['message']}")

    print()


if __name__ == "__main__":
    print("\n🎥 YouTube Transcript Service Examples\n")

    # Run all examples
    example_basic_usage()
    example_with_proxy()
    example_multiple_languages()
    example_available_transcripts()
    example_video_info()
    example_error_handling()

    print("=" * 60)
    print("Examples completed!")
    print("=" * 60)
    print("\n📝 Notes:")
    print("  - For cloud deployments (AWS, GCP, Azure), use WebShare proxy")
    print("  - Set WEBSHARE_PROXY_USERNAME and WEBSHARE_PROXY_PASSWORD env vars")
    print("  - Get WebShare credentials at https://www.webshare.io/")
    print("  - Make sure to purchase 'Residential' proxy package, not 'Proxy Server'")
