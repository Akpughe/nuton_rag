#!/usr/bin/env python3
"""
Test script for WetroCloud YouTube service with yt-dlp fallback.
"""

import os
from services.wetrocloud_youtube import WetroCloudYouTubeService

def test_wetrocloud_with_fallback():
    print("🧪 Testing WetroCloud YouTube Service with yt-dlp Fallback\n")
    print("="*70)

    # Test video URL
    video_url = "https://youtu.be/eTHft0-OSWA"

    print(f"\n📹 Test Video: {video_url}\n")

    # Test 1: With fallback enabled (default)
    print("Test 1: WetroCloud with yt-dlp fallback ENABLED")
    print("-" * 70)

    service_with_fallback = WetroCloudYouTubeService(enable_ytdlp_fallback=True)
    result = service_with_fallback.get_transcript(video_url)

    if result['success']:
        method = result.get('method', 'unknown')
        print(f"✅ SUCCESS!")
        print(f"   Method used: {method}")
        print(f"   Video ID: {result.get('video_id', 'N/A')}")

        if method == 'wetrocloud':
            print(f"   WetroCloud API worked directly")
            print(f"   Tokens used: {result.get('tokens', 0)}")
        elif method == 'ytdlp-fallback':
            print(f"   yt-dlp fallback was used")
            print(f"   WetroCloud error: {result.get('wetrocloud_error', 'N/A')}")
            print(f"   Video title: {result.get('video_title', 'N/A')}")

        print(f"   Transcript length: {len(result.get('text', ''))} characters")
        print(f"   Preview: {result.get('text', '')[:150]}...")
    else:
        print(f"❌ FAILED: {result.get('message', 'Unknown error')}")

    print("\n" + "="*70)

    # Test 2: Without fallback (to show the difference)
    print("\nTest 2: WetroCloud WITHOUT yt-dlp fallback")
    print("-" * 70)

    service_no_fallback = WetroCloudYouTubeService(enable_ytdlp_fallback=False)
    result_no_fallback = service_no_fallback.get_transcript(video_url)

    if result_no_fallback['success']:
        print(f"✅ SUCCESS!")
        print(f"   Method: {result_no_fallback.get('method', 'wetrocloud')}")
        print(f"   WetroCloud API worked")
    else:
        print(f"❌ FAILED (as expected if WetroCloud is down)")
        print(f"   Error: {result_no_fallback.get('message', 'Unknown error')}")
        print(f"   Note: No fallback available, so it failed")

    print("\n" + "="*70)
    print("\n🎯 Test Summary:")
    print("-" * 70)

    if result['success']:
        if result.get('method') == 'wetrocloud':
            print("✅ WetroCloud API is working - no fallback needed")
        elif result.get('method') == 'ytdlp-fallback':
            print("✅ yt-dlp fallback worked when WetroCloud failed")
    else:
        print("❌ Both methods failed (check network/credentials)")

    print("\n💡 Recommendation:")
    if result['success'] and result.get('method') == 'ytdlp-fallback':
        print("   WetroCloud is having issues, but yt-dlp fallback ensures reliability!")
        print("   The system will automatically recover when WetroCloud is back online.")
    elif result['success'] and result.get('method') == 'wetrocloud':
        print("   Everything is working perfectly with WetroCloud!")
        print("   yt-dlp is ready as a backup if needed.")
    else:
        print("   Check your network connection and API credentials.")

    print("\n" + "="*70 + "\n")


def test_specific_scenarios():
    """Test specific error scenarios."""
    print("🔬 Testing Specific Scenarios\n")
    print("="*70)

    service = WetroCloudYouTubeService(enable_ytdlp_fallback=True)

    # Scenario 1: Invalid video URL
    print("\nScenario 1: Invalid Video URL")
    print("-" * 70)
    result = service.get_transcript("https://youtube.com/invalid")
    print(f"Result: {'✅ Handled gracefully' if not result['success'] else '❌ Unexpected success'}")
    print(f"Message: {result.get('message', 'N/A')}")

    # Scenario 2: Video with no transcript
    print("\n\nScenario 2: Video ID that might not exist")
    print("-" * 70)
    result = service.get_transcript("https://youtube.com/watch?v=INVALIDVIDEO123")
    print(f"Result: {'✅ Handled gracefully' if not result['success'] else '❌ Unexpected success'}")
    if not result['success']:
        print(f"Message: {result.get('message', 'N/A')[:100]}...")

    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    print("\n" + "🎬 " * 20)
    print("WetroCloud YouTube Service - Fallback System Test")
    print("🎬 " * 20 + "\n")

    # Main test
    test_wetrocloud_with_fallback()

    # Additional scenarios
    print("\n" + "━" * 70 + "\n")
    test_specific_scenarios()

    print("✨ All tests completed!\n")
