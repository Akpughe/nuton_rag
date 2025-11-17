#!/usr/bin/env python3
"""
Quick test script for yt-dlp transcript service.
"""

from services.ytdlp_transcript_service import YTDLPTranscriptService

def test_ytdlp():
    print("🧪 Testing yt-dlp transcript service...\n")

    # Initialize service
    ytdlp = YTDLPTranscriptService()

    # Test video URL
    video_url = "https://youtu.be/eTHft0-OSWA"

    print(f"📹 Testing with video: {video_url}\n")

    # Get transcript
    result = ytdlp.get_transcript(video_url, languages=["en"])

    if result['success']:
        print("✅ SUCCESS!\n")
        print(f"Video ID: {result['video_id']}")
        print(f"Title: {result['video_title']}")
        print(f"Language: {result['language']}")
        print(f"Is Automatic: {result['is_automatic']}")
        print(f"Method: {result['method']}")
        print(f"\nTranscript Preview (first 500 chars):")
        print(result['text'][:500] + "...")
        print(f"\nTotal transcript length: {len(result['text'])} characters")
        print(f"Number of entries: {len(result['transcript_entries'])}")
    else:
        print(f"❌ FAILED: {result['message']}")

    print("\n" + "="*60)
    print("Test completed!")

if __name__ == "__main__":
    test_ytdlp()
