#!/usr/bin/env python3
"""
Test script for the Vcyon API endpoint
"""

import requests
import json

# Base URL for the API
BASE_URL = "http://localhost:8000"

# Test videos
TEST_VIDEOS = [
    "https://youtu.be/FDEVzFsPrkU?si=Ok9O0XmF1c2kCdu0",
    "https://youtu.be/2SEgQiu8XaU?si=fpK4NnQPYPE7RjzT"
]

def test_vcyon_endpoint(video_url):
    """Test the Vcyon API endpoint with a video URL"""
    print(f"\n{'='*80}")
    print(f"Testing Vcyon API with: {video_url}")
    print(f"{'='*80}\n")

    try:
        # Make POST request to the endpoint
        response = requests.post(
            f"{BASE_URL}/test_vcyon_transcript",
            data={
                "video_url": video_url,
                "languages": "en"
            },
            timeout=60
        )

        # Parse response
        result = response.json()

        # Display results
        if result.get('success'):
            print("✅ SUCCESS")
            print(f"\n🎬 Video ID: {result.get('video_id')}")
            print(f"📝 Method: {result.get('method')}")
            print(f"🌐 Language: {result.get('language')}")
            print(f"📊 Transcript Length: {result.get('transcript_length')} characters")
            print(f"📄 Number of Entries: {result.get('transcript_entries_count')}")

            # Video info
            if result.get('video_info'):
                video_info = result['video_info']
                print(f"\n📹 Video Information:")
                print(f"  Title: {video_info.get('title')}")
                print(f"  Author: {video_info.get('author')}")
                print(f"  Duration: {video_info.get('duration')} seconds")
                print(f"  Views: {video_info.get('view_count'):,}")
                print(f"  Thumbnails: {len(video_info.get('thumbnails', []))} available")

            # Transcript preview
            if result.get('transcript_preview'):
                print(f"\n📜 Transcript Preview:")
                print("-" * 60)
                print(result['transcript_preview'])
                print("-" * 60)

            return True
        else:
            print("❌ FAILED")
            print(f"Error: {result.get('error')}")
            print(f"Video ID: {result.get('video_id')}")

            # Video info might still be available even if transcript failed
            if result.get('video_info'):
                video_info = result['video_info']
                print(f"\n📹 Video Information (still available):")
                print(f"  Title: {video_info.get('title')}")
                print(f"  Author: {video_info.get('author')}")

            return False

    except requests.exceptions.ConnectionError:
        print("❌ ERROR: Could not connect to the server")
        print("Make sure the FastAPI server is running on http://localhost:8000")
        print("Run: uvicorn pipeline:app --reload")
        return False
    except requests.exceptions.Timeout:
        print("❌ ERROR: Request timed out")
        return False
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

def main():
    """Main test function"""
    print("\n" + "="*80)
    print("VCYON API ENDPOINT TEST")
    print("="*80)
    print(f"\nTesting endpoint: {BASE_URL}/test_vcyon_transcript")

    results = []

    for video_url in TEST_VIDEOS:
        success = test_vcyon_endpoint(video_url)
        results.append({
            'url': video_url,
            'success': success
        })

    # Summary
    print(f"\n{'='*80}")
    print("TEST SUMMARY")
    print(f"{'='*80}\n")

    passed = sum(1 for r in results if r['success'])
    total = len(results)

    for result in results:
        status = "✅ PASSED" if result['success'] else "❌ FAILED"
        print(f"{status} - {result['url']}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1

if __name__ == '__main__':
    import sys
    exit_code = main()
    sys.exit(exit_code)
