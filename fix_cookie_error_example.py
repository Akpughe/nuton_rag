#!/usr/bin/env python3
"""
Example showing how to fix the YouTube cookie error for server/cloud environments.

ERROR: could not find chrome cookies database in "/root/.config/google-chrome"

This error occurs when running in environments without browsers (servers, containers, etc.)
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


def example_server_environment():
    """
    Example for server/cloud environments WITHOUT browsers.
    Use use_cookies=False to skip browser cookie extraction.
    """
    print("=" * 70)
    print("Server Environment Fix (No Browser)")
    print("=" * 70)

    # SOLUTION: Disable cookie extraction for server environments
    ytdlp = YTDLPTranscriptService(use_cookies=False)

    # Test with a video
    video_url = "https://www.youtube.com/watch?v=44eFf-tRiSg"

    print(f"\nAttempting to extract transcript (without cookies)...")
    result = ytdlp.get_transcript(video_url)

    if result['success']:
        print("\n✅ SUCCESS!")
        print(f"Video Title: {result['video_title']}")
        print(f"Transcript length: {len(result['text'])} characters")
        print(f"\nFirst 300 characters:\n{result['text'][:300]}...")
        return True
    else:
        print(f"\n❌ FAILED: {result['message']}")
        return False


def example_local_with_browser():
    """
    Example for local development WITH browsers.
    Uses default behavior (auto-detect cookies).
    """
    print("\n" + "=" * 70)
    print("Local Development (With Browser)")
    print("=" * 70)

    # Default behavior - will auto-detect and use browser cookies
    ytdlp = YTDLPTranscriptService()

    video_url = "https://www.youtube.com/watch?v=44eFf-tRiSg"

    print(f"\nAttempting to extract transcript (with cookies)...")
    result = ytdlp.get_transcript(video_url)

    if result['success']:
        print("\n✅ SUCCESS!")
        print(f"Video Title: {result['video_title']}")
        print(f"Transcript length: {len(result['text'])} characters")
        return True
    else:
        print(f"\n❌ FAILED: {result['message']}")
        return False


def example_environment_detection():
    """
    Example: Auto-detect environment and adjust settings.
    """
    print("\n" + "=" * 70)
    print("Auto-Detect Environment")
    print("=" * 70)

    # Check if we're in a server environment
    is_server = (
        os.path.exists('/root') or  # Running as root (common in containers)
        os.environ.get('CONTAINER') == 'true' or  # Explicit container flag
        os.environ.get('ENVIRONMENT') in ['production', 'staging']  # Env variable
    )

    print(f"\nDetected environment: {'Server/Container' if is_server else 'Local Development'}")

    # Adjust cookie usage based on environment
    use_cookies = not is_server
    ytdlp = YTDLPTranscriptService(use_cookies=use_cookies)

    video_url = "https://www.youtube.com/watch?v=44eFf-tRiSg"

    print(f"Attempting to extract transcript (use_cookies={use_cookies})...")
    result = ytdlp.get_transcript(video_url)

    if result['success']:
        print("\n✅ SUCCESS!")
        print(f"Video Title: {result['video_title']}")
        return True
    else:
        print(f"\n❌ FAILED: {result['message']}")
        return False


if __name__ == "__main__":
    print("\n🔧 YouTube Cookie Error Fix Examples\n")

    # Determine which example to run based on environment
    # Check if we're likely in a server environment
    is_likely_server = os.path.exists('/root') or os.environ.get('CONTAINER') == 'true'

    if is_likely_server:
        print("Detected server environment - running server example\n")
        success = example_server_environment()
    else:
        print("Detected local environment - running local example\n")
        success = example_local_with_browser()

    # Also show the environment detection example
    example_environment_detection()

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    print("\n📝 Quick Reference:")
    print("\n1. Server/Cloud/Container (NO browser):")
    print("   ytdlp = YTDLPTranscriptService(use_cookies=False)")

    print("\n2. Local Development (WITH browser):")
    print("   ytdlp = YTDLPTranscriptService()  # Default, auto-detects")

    print("\n3. Specific Browser:")
    print("   ytdlp = YTDLPTranscriptService(browser='firefox')")

    print("\n4. Environment Detection:")
    print("   is_server = os.environ.get('ENVIRONMENT') == 'production'")
    print("   ytdlp = YTDLPTranscriptService(use_cookies=not is_server)")

    print("\n" + "=" * 70)

    sys.exit(0 if success else 1)
