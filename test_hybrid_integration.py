"""
Test the hybrid PDF processor integration with pipeline.py

This script tests:
1. Direct hybrid processor call
2. Enhanced metadata presence
3. Clean text extraction (no broken words)
"""

import sys
from pathlib import Path
from hybrid_pdf_processor import extract_and_chunk_pdf

def test_hybrid_processor():
    """Test hybrid PDF processor on AI Overview PDF."""

    pdf_path = "/Users/davak/Documents/_study/Artificial Intelligence_An Overview.pdf"

    print("=" * 80)
    print("Testing Hybrid PDF Processor Integration")
    print("=" * 80)
    print(f"\nProcessing: {Path(pdf_path).name}\n")

    try:
        # Call hybrid processor
        chunks = extract_and_chunk_pdf(
            pdf_path=pdf_path,
            chunk_size=512,
            overlap_tokens=80,
            tokenizer="gpt2",
            recipe="markdown",
            lang="en",
            min_characters_per_chunk=12
        )

        print(f"✅ Generated {len(chunks)} chunks\n")

        # Check first chunk
        if chunks:
            first_chunk = chunks[0]

            print("=" * 80)
            print("FIRST CHUNK ANALYSIS")
            print("=" * 80)

            # Check for Chonkie fields
            print("\n📊 CHONKIE FIELDS (Standard):")
            print(f"  ✓ text: {len(first_chunk.get('text', ''))} chars")
            print(f"  ✓ start_index: {first_chunk.get('start_index', 'N/A')}")
            print(f"  ✓ end_index: {first_chunk.get('end_index', 'N/A')}")
            print(f"  ✓ token_count: {first_chunk.get('token_count', 'N/A')}")

            # Check for enhanced metadata
            print("\n🚀 ENHANCED METADATA (New):")
            print(f"  ✓ extraction_method: {first_chunk.get('extraction_method', 'N/A')}")
            print(f"  ✓ extraction_quality: {first_chunk.get('extraction_quality', 'N/A')}")
            print(f"  ✓ heading_path: {first_chunk.get('heading_path', [])}")
            print(f"  ✓ node_types: {first_chunk.get('node_types', [])}")

            # Check text quality
            print("\n📝 TEXT QUALITY CHECK:")
            text = first_chunk.get('text', '')

            # Known broken patterns from previous extraction
            broken_patterns = [
                "Artific ical",
                "Over eriew",
                "definition tion",
                "Computation tionl",
                "Unive ivesity"
            ]

            issues_found = []
            for pattern in broken_patterns:
                if pattern.lower() in text.lower():
                    issues_found.append(pattern)

            if issues_found:
                print(f"  ❌ BROKEN PATTERNS FOUND: {issues_found}")
            else:
                print(f"  ✅ No broken patterns detected!")

            # Show first 300 chars
            print("\n📖 FIRST 300 CHARACTERS:")
            print("-" * 80)
            print(text[:300])
            print("-" * 80)

            # Show markdown context
            markdown_context = first_chunk.get('markdown_context', '')
            if markdown_context:
                print("\n📚 MARKDOWN CONTEXT (First 200 chars):")
                print("-" * 80)
                print(markdown_context[:200])
                print("-" * 80)

            print("\n" + "=" * 80)
            print("✅ INTEGRATION TEST COMPLETE")
            print("=" * 80)

            # Summary
            print("\n📋 SUMMARY:")
            print(f"  • Total chunks: {len(chunks)}")
            print(f"  • Extraction method: {first_chunk.get('extraction_method', 'unknown')}")
            print(f"  • Quality score: {first_chunk.get('extraction_quality', 'N/A')}")
            print(f"  • Text quality: {'CLEAN ✅' if not issues_found else 'BROKEN ❌'}")
            print(f"  • Enhanced metadata: {'PRESENT ✅' if first_chunk.get('extraction_method') else 'MISSING ❌'}")

            return not issues_found
        else:
            print("❌ No chunks generated!")
            return False

    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_hybrid_processor()
    sys.exit(0 if success else 1)
