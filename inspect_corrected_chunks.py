"""
Inspect Corrected Chunks

Show before/after comparison of chunks that were successfully corrected.
"""

import asyncio
from dotenv import load_dotenv
from hybrid_pdf_processor import extract_and_chunk_pdf_async

load_dotenv()


async def inspect_corrections():
    """Show before/after for corrected chunks."""

    pdf_path = "/Users/davak/Documents/_study/Artificial Intelligence_An Overview.pdf"

    print("\n" + "="*80)
    print("📊 CORRECTED CHUNKS INSPECTION")
    print("="*80 + "\n")

    # Get corrected chunks
    chunks_corrected = await extract_and_chunk_pdf_async(
        pdf_path=pdf_path,
        chunk_size=512,
        overlap_tokens=80,
        enable_llm_correction=True
    )

    # Get uncorrected chunks for comparison
    chunks_uncorrected = await extract_and_chunk_pdf_async(
        pdf_path=pdf_path,
        chunk_size=512,
        overlap_tokens=80,
        enable_llm_correction=False
    )

    # Find chunks that were corrected
    corrected_indices = [
        i for i, chunk in enumerate(chunks_corrected)
        if chunk.get('was_llm_corrected', False)
    ]

    print(f"Found {len(corrected_indices)} corrected chunks\n")

    # Show first 2 corrected chunks
    for idx in corrected_indices[:2]:
        print("="*80)
        print(f"CHUNK #{idx + 1}")
        print("="*80)

        original = chunks_uncorrected[idx].get('text', '')
        corrected = chunks_corrected[idx].get('text', '')

        print("\n🔴 BEFORE (Original):")
        print("-"*80)
        print(original[:500])
        print("-"*80)

        print("\n✅ AFTER (Corrected):")
        print("-"*80)
        print(corrected[:500])
        print("-"*80)

        # Calculate improvement
        original_len = len(original)
        corrected_len = len(corrected)

        print(f"\n📊 Stats:")
        print(f"  - Original length: {original_len} chars")
        print(f"  - Corrected length: {corrected_len} chars")
        print(f"  - Change: {corrected_len - original_len:+d} chars ({(corrected_len/original_len - 1)*100:+.1f}%)")
        print()


if __name__ == "__main__":
    asyncio.run(inspect_corrections())
