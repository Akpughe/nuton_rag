"""
Test DocChunker extraction quality on problematic PDF.

Checks if DocChunker can extract text without mid-word spacing issues.
"""

from docchunker import DocChunker
import sys
import json
from pathlib import Path

def test_extraction():
    """Test DocChunker on the AI Overview PDF."""

    pdf_path = "/Users/davak/Documents/_study/anatomy+phys+vol2a.pdf"
    # pdf_path = "/Users/davak/Documents/_study/Artificial Intelligence_An Overview.pdf"

    print("=" * 80)
    print("Testing DocChunker Extraction Quality")
    print("=" * 80)
    print(f"\nProcessing: {pdf_path}\n")

    try:
        # Initialize DocChunker with default settings
        chunker = DocChunker(chunk_size=1000, num_overlapping_elements=0)

        # Process the PDF
        print("Extracting text with DocChunker...")
        chunks = chunker.process_document(pdf_path)

        if not chunks:
            print("❌ ERROR: No chunks returned!")
            return False

        print(f"✅ Extracted {len(chunks)} chunks\n")

        # Get first chunk's text
        first_chunk = chunks[0]

        # Chunk is an object, not a dict - access text attribute directly
        text = first_chunk.text if hasattr(first_chunk, 'text') else str(first_chunk)

        if not text:
            print("❌ ERROR: First chunk has no text!")
            return False

        # Display first 800 characters
        print("=" * 80)
        print("FIRST 800 CHARACTERS OF EXTRACTED TEXT:")
        print("=" * 80)
        print(text[:800])
        print("=" * 80)
        print()

        # Check for broken patterns
        print("QUALITY CHECKS:")
        print("-" * 80)

        broken_patterns = {
            "Artific ical": "Artificial",
            "Over eriew": "Overview",
            "definition tion": "definition",
            "Computation tionl": "Computational",
            "Unive ivesity": "University",
            "inter atrial": "interatrial",
            "al al": "all",
            "er er": "error",
            "ic ic": "ici",
            "tion tion": "tion",
        }

        issues_found = []
        for broken, correct in broken_patterns.items():
            if broken in text[:1000]:  # Check first 1000 chars
                issues_found.append(f"Found '{broken}' (should be '{correct}')")

        if issues_found:
            print("❌ BROKEN TEXT DETECTED:")
            for issue in issues_found:
                print(f"   • {issue}")
            print()
            return False
        else:
            print("✅ No broken word patterns detected!")
            print()

        # Check for correct patterns
        correct_patterns = ["Artificial", "Intelligence", "Overview"]
        found_correct = []

        for pattern in correct_patterns:
            if pattern in text[:500]:
                found_correct.append(pattern)

        if len(found_correct) >= 2:
            print(f"✅ Found correct patterns: {', '.join(found_correct)}")
            print()

        # Display chunk metadata
        print("CHUNK METADATA:")
        print("-" * 80)
        # Display Chunk object attributes
        print(f"  Type: {type(first_chunk)}")
        if hasattr(first_chunk, '__dict__'):
            for key, value in first_chunk.__dict__.items():
                if key != 'text':  # Skip text (already displayed)
                    print(f"  {key}: {value}")
        print()

        # Save extraction to file
        print("=" * 80)
        print("SAVING EXTRACTION RESULTS...")
        print("=" * 80)

        # Prepare output data
        output_data = {
            "source_pdf": pdf_path,
            "total_chunks": len(chunks),
            "extraction_quality": "clean" if not issues_found else "broken",
            "chunks": []
        }

        # Convert chunks to serializable format
        for i, chunk in enumerate(chunks):
            chunk_data = {
                "chunk_index": i,
                "text": chunk.text,
                "metadata": chunk.metadata if hasattr(chunk, 'metadata') else {}
            }
            output_data["chunks"].append(chunk_data)

        # Save JSON file
        output_dir = Path("docchunker_outputs")
        output_dir.mkdir(exist_ok=True)

        json_path = output_dir / "Artificial_Intelligence_docchunker_extraction.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"✅ Saved JSON extraction to: {json_path}")

        # Save readable text file (first 10 chunks)
        text_path = output_dir / "Artificial_Intelligence_docchunker_sample.txt"
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("DocChunker Extraction Results\n")
            f.write(f"Source: {pdf_path}\n")
            f.write(f"Total Chunks: {len(chunks)}\n")
            f.write("=" * 80 + "\n\n")

            # Write first 10 chunks as sample
            for i in range(min(10, len(chunks))):
                chunk = chunks[i]
                f.write(f"\n{'='*80}\n")
                f.write(f"CHUNK {i+1}/{len(chunks)}\n")
                f.write(f"{'='*80}\n")

                # Write metadata
                if hasattr(chunk, 'metadata'):
                    f.write("METADATA:\n")
                    f.write(f"  Node Type: {chunk.metadata.get('node_type', 'N/A')}\n")
                    f.write(f"  Headings: {chunk.metadata.get('headings', [])}\n")
                    f.write(f"  Chars: {chunk.metadata.get('num_chars', 0)}\n")
                    f.write("\n")

                f.write("TEXT:\n")
                f.write("-" * 80 + "\n")
                f.write(chunk.text)
                f.write("\n" + "-" * 80 + "\n")

        print(f"✅ Saved text sample to: {text_path}")
        print()

        # Final verdict
        print("=" * 80)
        print("VERDICT: ✅ DocChunker extracts clean text!")
        print("=" * 80)
        print(f"\n📁 Files saved in: {output_dir.absolute()}")
        return True

    except Exception as e:
        print(f"❌ ERROR during extraction: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_extraction()
    sys.exit(0 if success else 1)
