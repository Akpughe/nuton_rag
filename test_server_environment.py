#!/usr/bin/env python3
"""
Test script to simulate server environment behavior.

Tests:
1. Without cookies (use_cookies=False) - simulates server environment
2. With environment variable YTDLP_USE_COOKIES=false
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import directly
import importlib.util
spec = importlib.util.spec_from_file_location(
    "ytdlp_transcript_service",
    os.path.join(os.path.dirname(__file__), "services", "ytdlp_transcript_service.py")
)
ytdlp_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ytdlp_module)
YTDLPTranscriptService = ytdlp_module.YTDLPTranscriptService


def test_without_cookies():
    """Test 1: Explicit use_cookies=False (server mode)"""
    print("=" * 70)
    print("Test 1: Server Environment Simulation (use_cookies=False)")
    print("=" * 70)

    # This is what should be used in server deployments
    service = YTDLPTranscriptService(use_cookies=False)

    video_url = "https://www.youtube.com/watch?v=44eFf-tRiSg"

    print(f"\nExtracting transcript from: {video_url}")
    print("(Without cookies - simulating server/Docker environment)\n")

    result = service.get_transcript(video_url)

    if result['success']:
        print("\n✅ SUCCESS!")
        print(f"Video Title: {result['video_title']}")
        print(f"Transcript length: {len(result['text'])} characters")
        print(f"\nFirst 300 characters:\n{result['text'][:300]}...")
        return True
    else:
        print(f"\n❌ FAILED: {result['message']}")
        return False


def test_with_env_variable():
    """Test 2: Using YTDLP_USE_COOKIES environment variable"""
    print("\n" + "=" * 70)
    print("Test 2: Environment Variable Override (YTDLP_USE_COOKIES=false)")
    print("=" * 70)

    # Set environment variable (simulating Docker/container config)
    os.environ['YTDLP_USE_COOKIES'] = 'false'

    # Initialize without explicit use_cookies parameter
    # (service should read from env var)
    service = YTDLPTranscriptService()

    video_url = "https://www.youtube.com/watch?v=44eFf-tRiSg"

    print(f"\nExtracting transcript from: {video_url}")
    print("(Environment variable YTDLP_USE_COOKIES=false)\n")

    result = service.get_transcript(video_url)

    # Clean up env var
    del os.environ['YTDLP_USE_COOKIES']

    if result['success']:
        print("\n✅ SUCCESS!")
        print(f"Video Title: {result['video_title']}")
        print(f"Transcript length: {len(result['text'])} characters")
        return True
    else:
        print(f"\n❌ FAILED: {result['message']}")
        return False


def test_backward_compatibility():
    """Test 3: Existing code (no parameters) should work everywhere"""
    print("\n" + "=" * 70)
    print("Test 3: Backward Compatibility (Default Initialization)")
    print("=" * 70)

    # This is how ALL existing production code calls it
    # It should work with cookies locally, fall back automatically on servers
    service = YTDLPTranscriptService()

    video_url = "https://www.youtube.com/watch?v=44eFf-tRiSg"

    print(f"\nExtracting transcript from: {video_url}")
    print("(Default initialization - same as production code)\n")

    result = service.get_transcript(video_url)

    if result['success']:
        print("\n✅ SUCCESS!")
        print(f"Video Title: {result['video_title']}")
        print(f"Method: {result['method']}")
        print(f"Transcript length: {len(result['text'])} characters")
        return True
    else:
        print(f"\n❌ FAILED: {result['message']}")
        return False


if __name__ == "__main__":
    print("\n🧪 Server Environment Testing\n")

    # Run all tests
    test1 = test_without_cookies()
    test2 = test_with_env_variable()
    test3 = test_backward_compatibility()

    # Summary
    print("\n" + "=" * 70)
    print("Test Results Summary")
    print("=" * 70)
    print(f"Test 1 (use_cookies=False):      {'✅ PASSED' if test1 else '❌ FAILED'}")
    print(f"Test 2 (YTDLP_USE_COOKIES=false): {'✅ PASSED' if test2 else '❌ FAILED'}")
    print(f"Test 3 (Backward Compatible):     {'✅ PASSED' if test3 else '❌ FAILED'}")

    print("\n" + "=" * 70)
    print("Production Deployment Guide")
    print("=" * 70)
    print("\n✅ Option 1: Set environment variable (RECOMMENDED for containers)")
    print("   export YTDLP_USE_COOKIES=false")
    print("   # Then use YTDLPTranscriptService() normally")

    print("\n✅ Option 2: Explicit parameter")
    print("   service = YTDLPTranscriptService(use_cookies=False)")

    print("\n✅ Option 3: Do nothing (AUTO-FALLBACK)")
    print("   service = YTDLPTranscriptService()  # Works with auto-retry!")

    all_passed = test1 and test2 and test3
    sys.exit(0 if all_passed else 1)
