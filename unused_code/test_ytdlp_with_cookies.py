#!/usr/bin/env python3
"""
Test script for YTDLPTranscriptService with browser cookie support.

This script demonstrates how to use the updated YTDLPTranscriptService
that includes automatic browser cookie extraction to bypass YouTube's
bot detection.
"""

import sys
import os

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import directly from the module file to avoid __init__.py dependencies
import importlib.util
spec = importlib.util.spec_from_file_location(
    "ytdlp_transcript_service",
    os.path.join(os.path.dirname(__file__), "services", "ytdlp_transcript_service.py")
)
ytdlp_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ytdlp_module)
YTDLPTranscriptService = ytdlp_module.YTDLPTranscriptService


def test_with_auto_browser():
    """Test with automatic browser detection (tries Chrome by default)."""
    print("=" * 70)
    print("Test 1: Using Auto-Detected Browser Cookies (Chrome)")
    print("=" * 70)

    # Initialize service - will auto-detect and use Chrome cookies
    ytdlp = YTDLPTranscriptService()

    # Test with the problematic video
    video_url = "https://www.youtube.com/watch?v=44eFf-tRiSg"

    print(f"\nAttempting to extract transcript from: {video_url}")
    result = ytdlp.get_transcript(video_url)

    if result['success']:
        print("\n✅ SUCCESS!")
        print(f"Video Title: {result['video_title']}")
        print(f"Video ID: {result['video_id']}")
        print(f"Language: {result['language']}")
        print(f"Auto-generated: {result['is_automatic']}")
        print(f"Method: {result['method']}")
        print(f"\nTranscript length: {len(result['text'])} characters")
        print(f"\nFirst 500 characters:\n{result['text'][:500]}...")
        return True
    else:
        print(f"\n❌ FAILED: {result['message']}")
        return False


def test_with_specific_browser(browser_name):
    """Test with a specific browser."""
    print("\n" + "=" * 70)
    print(f"Test 2: Using Specific Browser Cookies ({browser_name})")
    print("=" * 70)

    # Initialize service with specific browser
    ytdlp = YTDLPTranscriptService(browser=browser_name)

    video_url = "https://www.youtube.com/watch?v=44eFf-tRiSg"

    print(f"\nAttempting to extract transcript from: {video_url}")
    result = ytdlp.get_transcript(video_url)

    if result['success']:
        print("\n✅ SUCCESS!")
        print(f"Video Title: {result['video_title']}")
        print(f"Transcript length: {len(result['text'])} characters")
        return True
    else:
        print(f"\n❌ FAILED: {result['message']}")
        return False


def get_video_info_test():
    """Test getting video info with cookies."""
    print("\n" + "=" * 70)
    print("Test 3: Getting Video Info")
    print("=" * 70)

    ytdlp = YTDLPTranscriptService()
    video_url = "https://www.youtube.com/watch?v=44eFf-tRiSg"

    print(f"\nGetting info for: {video_url}")
    result = ytdlp.get_video_info(video_url)

    if result['success']:
        print("\n✅ SUCCESS!")
        print(f"Video ID: {result['video_id']}")
        print(f"Title: {result['title']}")
        print(f"Channel: {result['channel']}")
        print(f"Duration: {result['duration']} seconds")
        print(f"View Count: {result['view_count']:,}")
        print(f"Upload Date: {result['upload_date']}")
        return True
    else:
        print(f"\n❌ FAILED: {result['message']}")
        return False


if __name__ == "__main__":
    print("\n🎥 YTDLPTranscriptService - Cookie Authentication Test\n")

    # Test with auto-detected browser (Chrome by default)
    success1 = test_with_auto_browser()

    # If you want to test with a different browser, uncomment below:
    # Available browsers: chrome, firefox, safari, edge, brave, opera, etc.
    # success2 = test_with_specific_browser('firefox')

    # Test getting video info
    success3 = get_video_info_test()

    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    print(f"Auto-detect browser: {'✅ PASSED' if success1 else '❌ FAILED'}")
    print(f"Get video info: {'✅ PASSED' if success3 else '❌ FAILED'}")

    print("\n📝 Notes:")
    print("  - The service automatically extracts cookies from your browser")
    print("  - Make sure you are logged into YouTube in your browser")
    print("  - If using Chrome, make sure it's installed and you've visited YouTube")
    print("  - Supported browsers: chrome, firefox, safari, edge, brave, opera")
    print("  - If one browser fails, try specifying a different one:")
    print("    YTDLPTranscriptService(browser='firefox')")

    if not success1:
        print("\n⚠️ Troubleshooting:")
        print("  1. Make sure you're logged into YouTube in your browser")
        print("  2. Try visiting the video URL in your browser first")
        print("  3. Close your browser and try again (to release the cookie database)")
        print("  4. Try a different browser by specifying it explicitly")
        print("  5. Check if you have the browser installed")

    sys.exit(0 if success1 else 1)
