"""
Test script for selective chunk quality correction.

Demonstrates the optimization: only correct chunks below quality threshold.
"""

import asyncio
from chunk_quality_corrector import (
    calculate_quality_score,
    process_chunks_in_parallel
)

# Sample chunks with varying quality levels
TEST_CHUNKS = [
    {
        "text": "This is a perfectly clean chunk of text with no issues at all. It should score very high and not need correction.",
        "chunk_id": 1
    },
    {
        "text": "Artific ial Intellig ence is a field of computer er science that focuses on creat ing intell igent machines.",
        "chunk_id": 2
    },
    {
        "text": "Machine learning enables computers to learn without being explicitly programmed.",
        "chunk_id": 3
    },
    {
        "text": "Neural net nets are comput ing sys stems ins pired by bio logical neural net works.",
        "chunk_id": 4
    },
    {
        "text": "Deep learning uses multiple layers to progressively extract higher-level features.",
        "chunk_id": 5
    },
    {
        "text": "definition tion of artific ial intel ligence includes reas oning, learn ing, and per erception.",
        "chunk_id": 6
    },
]


async def test_selective_correction():
    """Test selective correction with different thresholds."""

    print("=" * 80)
    print("SELECTIVE CHUNK CORRECTION DEMO")
    print("=" * 80)

    # First, show quality scores for all chunks
    print("\n📊 QUALITY SCORES:")
    print("-" * 80)
    for chunk in TEST_CHUNKS:
        score = calculate_quality_score(chunk['text'])
        preview = chunk['text'][:60] + "..." if len(chunk['text']) > 60 else chunk['text']
        print(f"Chunk {chunk['chunk_id']}: {score:.3f} - {preview}")

    # Test with different thresholds
    thresholds = [0.5, 0.65, 0.8, 1.0]

    for threshold in thresholds:
        print(f"\n{'=' * 80}")
        print(f"TESTING WITH THRESHOLD = {threshold}")
        print(f"{'=' * 80}")

        # Make a copy to avoid modifying originals
        test_chunks = [chunk.copy() for chunk in TEST_CHUNKS]

        # Process with this threshold
        result = await process_chunks_in_parallel(
            test_chunks,
            enable_correction=True,
            quality_threshold=threshold,
            max_concurrent=3
        )

        # Show results
        print("\n📋 RESULTS:")
        corrected_count = sum(1 for c in result if c.get('was_llm_corrected'))
        skipped_count = sum(1 for c in result if c.get('skip_reason') == 'high_quality')

        print(f"✅ Corrected: {corrected_count} chunks")
        print(f"⏭️  Skipped: {skipped_count} chunks")
        print(f"💰 Cost savings: {(skipped_count / len(TEST_CHUNKS) * 100):.1f}%")


async def test_quality_scoring():
    """Test quality scoring function."""

    print("\n" + "=" * 80)
    print("QUALITY SCORING EXAMPLES")
    print("=" * 80)

    examples = [
        ("Perfect text with no issues", "This is a clean, well-formatted piece of text."),
        ("Broken words", "Artific ial intell igence is amaz ing technol ogy"),
        ("Repeated fragments", "definition tion of under ertand ing and learn ing"),
        ("CID codes", "This text has (cid:123) embedded (cid:456) codes"),
        ("Special chars", "Text with ∂∫∑π many ≈≠±÷ special chars"),
        ("Very short words", "a b c d e f g h i j k l m"),
    ]

    for name, text in examples:
        score = calculate_quality_score(text)
        needs_correction = score < 0.65
        status = "❌ NEEDS CORRECTION" if needs_correction else "✅ CLEAN"
        print(f"\n{name}:")
        print(f"  Score: {score:.3f} {status}")
        print(f"  Text: {text[:60]}...")


if __name__ == "__main__":
    # Run quality scoring test first
    asyncio.run(test_quality_scoring())

    # Then run selective correction test
    asyncio.run(test_selective_correction())
