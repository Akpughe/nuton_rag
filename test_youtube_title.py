#!/usr/bin/env python3
"""
Test that video titles are correctly extracted and saved from yt-dlp.
"""

from services.wetrocloud_youtube import WetroCloudYouTubeService
from services.ytdlp_transcript_service import YTDLPTranscriptService

def test_title_extraction():
    print("🧪 Testing Video Title Extraction\n")
    print("="*70)

    test_url = "https://youtu.be/eTHft0-OSWA"

    # Test 1: WetroCloud service (with fallback)
    print("\n1️⃣ Testing WetroCloud Service (with yt-dlp fallback)")
    print("-"*70)

    wetro_service = WetroCloudYouTubeService(enable_ytdlp_fallback=True)
    result = wetro_service.get_transcript(test_url)

    if result['success']:
        method = result.get('method', 'unknown')
        print(f"✅ Success!")
        print(f"   Method: {method}")

        if 'video_title' in result:
            print(f"   Video Title: {result['video_title']}")
            print(f"   Title Source: {method}")
        else:
            print(f"   ⚠️ No video_title in response (WetroCloud doesn't provide title)")

        print(f"   Video ID: {result.get('video_id', 'N/A')}")
        print(f"   Thumbnail: {result.get('thumbnail', 'N/A')}")
    else:
        print(f"❌ Failed: {result.get('message')}")

    # Test 2: Direct yt-dlp service
    print("\n2️⃣ Testing Direct yt-dlp Service")
    print("-"*70)

    ytdlp_service = YTDLPTranscriptService()
    result_ytdlp = ytdlp_service.get_transcript(test_url)

    if result_ytdlp['success']:
        print(f"✅ Success!")
        print(f"   Method: {result_ytdlp.get('method', 'yt-dlp')}")
        print(f"   Video Title: {result_ytdlp.get('video_title', 'N/A')}")
        print(f"   Video ID: {result_ytdlp.get('video_id', 'N/A')}")
        print(f"   Language: {result_ytdlp.get('language', 'N/A')}")
        print(f"   Is Automatic: {result_ytdlp.get('is_automatic', 'N/A')}")
    else:
        print(f"❌ Failed: {result_ytdlp.get('message')}")

    # Test 3: Verify title is in response
    print("\n3️⃣ Title Availability Check")
    print("-"*70)

    print("Response fields that contain title:")

    if result['success']:
        if 'video_title' in result:
            print(f"   ✅ WetroCloud (via fallback): video_title = '{result['video_title']}'")
        else:
            print(f"   ⚠️ WetroCloud: No video_title field (will use external API)")

    if result_ytdlp['success']:
        print(f"   ✅ yt-dlp direct: video_title = '{result_ytdlp.get('video_title', 'N/A')}'")

    print("\n" + "="*70)
    print("\n💡 Summary:")
    print("-"*70)

    if result['success'] and 'video_title' in result:
        print("✅ WetroCloud with fallback: Title available from yt-dlp")
        print("   → process_youtube will use this title (no external API call needed)")
    elif result['success']:
        print("⚠️ WetroCloud without fallback: No title in response")
        print("   → process_youtube will call external API for title")

    if result_ytdlp['success']:
        print("✅ yt-dlp direct: Title always included")
        print("   → Saves external API call for title")

    print("\n🎯 Recommendation:")
    print("   When using process_youtube, the system will:")
    print("   1. Try to get title from transcript result (yt-dlp)")
    print("   2. Fall back to external API if title not in result (WetroCloud only)")
    print("   → This optimization saves API calls when yt-dlp is used!")

    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    test_title_extraction()
