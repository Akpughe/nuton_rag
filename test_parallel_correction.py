"""
Test Parallel LLM Correction on Problematic PDF

This script tests the new parallel quality correction pipeline on the
AI Overview PDF that has known text quality issues ("Artific ical", "Per ereption", etc.)
"""

import asyncio
import sys
import time
from pathlib import Path
from dotenv import load_dotenv
from hybrid_pdf_processor import extract_and_chunk_pdf_async

# Load environment variables from .env file
load_dotenv()


async def test_parallel_correction():
    """Test parallel LLM correction on the problematic AI PDF."""

    # This PDF has known quality issues
    pdf_path = "/Users/davak/Documents/_study/Artificial Intelligence_An Overview.pdf"

    print("=" * 80)
    print("🚀 PARALLEL LLM CORRECTION TEST")
    print("=" * 80)
    print(f"\n📄 Testing PDF: {Path(pdf_path).name}")
    print(f"🎯 Goal: Fix broken text like 'Artific ical' → 'Artificial'\n")

    # Test WITH correction
    print("=" * 80)
    print("TEST 1: WITH Parallel LLM Correction (ENABLED)")
    print("=" * 80)

    start_time = time.time()

    try:
        chunks_corrected = await extract_and_chunk_pdf_async(
            pdf_path=pdf_path,
            chunk_size=512,
            overlap_tokens=80,
            enable_llm_correction=True  # ENABLED
        )

        elapsed_corrected = time.time() - start_time

        print(f"\n✅ WITH Correction: {len(chunks_corrected)} chunks in {elapsed_corrected:.2f}s")

        # Analyze first chunk
        if chunks_corrected:
            first_chunk = chunks_corrected[0]
            text = first_chunk.get('text', '')
            was_corrected = first_chunk.get('was_llm_corrected', False)

            print(f"\n📊 First Chunk Analysis:")
            print(f"  - Length: {len(text)} chars")
            print(f"  - Was corrected by LLM: {'✅ YES' if was_corrected else '❌ NO'}")

            # Check for known broken patterns
            broken_patterns = [
                "Artific ical",
                "Over eriew",
                "Per ereption",
                "definition tion",
                "Computation tionl"
            ]

            issues_found = []
            for pattern in broken_patterns:
                if pattern.lower() in text[:1000].lower():
                    issues_found.append(pattern)

            print(f"\n🔍 Text Quality Check:")
            if issues_found:
                print(f"  ❌ Still has broken patterns: {issues_found}")
            else:
                print(f"  ✅ No broken patterns detected!")

            # Show first 300 chars
            print(f"\n📖 First 300 chars:")
            print("-" * 80)
            print(text[:300])
            print("-" * 80)

        # Count how many were corrected
        corrected_count = sum(1 for c in chunks_corrected if c.get('was_llm_corrected'))
        print(f"\n📈 Correction Statistics:")
        print(f"  - Total chunks: {len(chunks_corrected)}")
        print(f"  - Corrected by LLM: {corrected_count}")
        print(f"  - Kept original: {len(chunks_corrected) - corrected_count}")
        print(f"  - Correction rate: {(corrected_count/len(chunks_corrected)*100):.1f}%")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test WITHOUT correction for comparison
    print("\n" + "=" * 80)
    print("TEST 2: WITHOUT Correction (DISABLED) - for comparison")
    print("=" * 80)

    start_time = time.time()

    try:
        chunks_uncorrected = await extract_and_chunk_pdf_async(
            pdf_path=pdf_path,
            chunk_size=512,
            overlap_tokens=80,
            enable_llm_correction=False  # DISABLED
        )

        elapsed_uncorrected = time.time() - start_time

        print(f"\n✅ WITHOUT Correction: {len(chunks_uncorrected)} chunks in {elapsed_uncorrected:.2f}s")

        # Show broken text
        if chunks_uncorrected:
            text_uncorrected = chunks_uncorrected[0].get('text', '')

            print(f"\n📖 First 300 chars (uncorrected):")
            print("-" * 80)
            print(text_uncorrected[:300])
            print("-" * 80)

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

    # Performance comparison
    print("\n" + "=" * 80)
    print("⚡ PERFORMANCE COMPARISON")
    print("=" * 80)

    print(f"\nProcessing Time:")
    print(f"  - WITH correction:    {elapsed_corrected:.2f}s")
    print(f"  - WITHOUT correction: {elapsed_uncorrected:.2f}s")
    print(f"  - Overhead:           {elapsed_corrected - elapsed_uncorrected:.2f}s")
    print(f"  - Worth it?           {'✅ YES!' if corrected_count > 0 else '🤔 Maybe not needed'}")

    # Final verdict
    print("\n" + "=" * 80)
    print("🎯 FINAL VERDICT")
    print("=" * 80)

    if corrected_count > 0 and not issues_found:
        print("\n✅ SUCCESS! Parallel LLM correction fixed the broken text!")
        print(f"   - {corrected_count} chunks were automatically corrected")
        print(f"   - Processing overhead: only {elapsed_corrected - elapsed_uncorrected:.2f}s")
        print(f"   - Text quality: EXCELLENT")
        return True
    elif corrected_count == 0:
        print("\n⚠️  No corrections applied (chunks were already clean)")
        print(f"   - This is actually good - no wasted processing!")
        return True
    else:
        print("\n❌ Corrections applied but issues remain")
        print(f"   - May need prompt tuning or different model")
        return False


if __name__ == "__main__":
    print("\n🚀 Starting parallel correction test...\n")

    success = asyncio.run(test_parallel_correction())

    sys.exit(0 if success else 1)
