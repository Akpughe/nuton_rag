#!/usr/bin/env python3
"""
Test script for Vcyon API integration with WetroCloudYouTubeService
"""

import sys
import os

# Add the parent directory to the path to import services
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.wetrocloud_youtube import WetroCloudYouTubeService

def test_video(service, video_url, video_name):
    """Test a single video URL"""
    print(f"\n{'='*80}")
    print(f"Testing: {video_name}")
    print(f"URL: {video_url}")
    print(f"{'='*80}")

    # Test video ID extraction
    video_id = service.extract_video_id(video_url)
    print(f"\n✓ Video ID extracted: {video_id}")

    # Test video title retrieval
    print("\n--- Testing Video Title Retrieval ---")
    title = service.get_video_title(video_url)
    print(f"✓ Video Title: {title}")

    # Test transcript extraction
    print("\n--- Testing Transcript Extraction ---")
    result = service.get_transcript(video_url)

    if result['success']:
        print(f"✅ SUCCESS using method: {result.get('method', 'unknown')}")
        print(f"✓ Video ID: {result.get('video_id', 'N/A')}")
        print(f"✓ Thumbnail: {result.get('thumbnail', 'N/A')}")

        # Show transcript preview
        text = result.get('text', '')
        if text:
            lines = text.split('\n')
            preview_lines = lines[:5]  # Show first 5 lines
            print(f"\n✓ Transcript Preview ({len(lines)} total lines):")
            for line in preview_lines:
                print(f"  {line}")
            if len(lines) > 5:
                print(f"  ... ({len(lines) - 5} more lines)")

            # Show character count
            print(f"\n✓ Total transcript length: {len(text)} characters")
        else:
            print("⚠️  No transcript text found")

        # Show transcript entry count
        entries = result.get('transcript_entries', [])
        print(f"✓ Number of transcript entries: {len(entries)}")

        # Show additional metadata
        if 'language' in result:
            print(f"✓ Language: {result['language']}")
        if 'tokens' in result:
            print(f"✓ Tokens used: {result['tokens']}")

    else:
        print(f"❌ FAILED: {result.get('message', 'Unknown error')}")

    return result['success']

def main():
    """Main test function"""
    print("\n" + "="*80)
    print("VCYON API INTEGRATION TEST")
    print("="*80)

    # Initialize service with all fallbacks enabled
    print("\nInitializing WetroCloudYouTubeService with Vcyon and yt-dlp fallback...")
    service = WetroCloudYouTubeService(
        enable_vcyon_fallback=True,
        enable_ytdlp_fallback=True
    )
    print("✓ Service initialized")

    # Test videos
    test_videos = [
        {
            'url': 'https://youtu.be/FDEVzFsPrkU?si=Ok9O0XmF1c2kCdu0',
            'name': 'Video 1'
        },
        {
            'url': 'https://youtu.be/2SEgQiu8XaU?si=fpK4NnQPYPE7RjzT',
            'name': 'Video 2'
        }
    ]

    results = []
    for video in test_videos:
        success = test_video(service, video['url'], video['name'])
        results.append({
            'name': video['name'],
            'url': video['url'],
            'success': success
        })

    # Summary
    print(f"\n{'='*80}")
    print("TEST SUMMARY")
    print(f"{'='*80}")

    for result in results:
        status = "✅ PASSED" if result['success'] else "❌ FAILED"
        print(f"{status} - {result['name']}")
        print(f"  URL: {result['url']}")

    total = len(results)
    passed = sum(1 for r in results if r['success'])
    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
