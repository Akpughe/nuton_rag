#!/usr/bin/env python3
"""
Test script for Chonkie OSS PDF chunking.
Upload a PDF and see the chunked output with timing.
"""

import sys
import time
import json
from pathlib import Path
from chonkie_oss_client import chunk_document


def format_chunk_output(chunks, elapsed_time):
    """Format chunks for readable display."""
    print("\n" + "="*80)
    print(f"📊 CHUNKING RESULTS")
    print("="*80)

    print(f"\n⏱️  Processing Time: {elapsed_time*1000:.2f}ms ({elapsed_time:.3f}s)")
    print(f"📄 Total Chunks: {len(chunks)}")

    if chunks:
        total_tokens = sum(chunk.get('token_count', 0) for chunk in chunks)
        avg_tokens = total_tokens / len(chunks)
        print(f"🔢 Total Tokens: {total_tokens:,}")
        print(f"📊 Average Tokens per Chunk: {avg_tokens:.1f}")

    print("\n" + "="*80)
    print("📝 CHUNK DETAILS")
    print("="*80)

    for i, chunk in enumerate(chunks, 1):
        print(f"\n{'─'*80}")
        print(f"Chunk #{i}")
        print(f"{'─'*80}")
        print(f"Token Count: {chunk.get('token_count', 'N/A')}")
        print(f"Character Range: {chunk.get('start_index', 0)} - {chunk.get('end_index', 0)}")
        print(f"\nText Preview (first 300 chars):")
        print(f"{'┌'*80}")
        text = chunk.get('text', '')
        preview = text[:300] + ("..." if len(text) > 300 else "")
        print(preview)
        print(f"{'└'*80}")

        if i < len(chunks):
            print(f"\n{'▼'*80}\n")

    print("\n" + "="*80)
    print("✅ CHUNKING COMPLETE")
    print("="*80)


def test_pdf_chunking(pdf_path, chunk_size=512, overlap=80, chunker_type="recursive"):
    """
    Test PDF chunking with Chonkie OSS.

    Args:
        pdf_path: Path to PDF file
        chunk_size: Target chunk size in tokens
        overlap: Overlap between chunks in tokens
        chunker_type: Type of chunker ('recursive', 'token', 'semantic')
    """
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        print(f"❌ Error: File not found: {pdf_path}")
        return None

    if not pdf_path.suffix.lower() == '.pdf':
        print(f"❌ Error: Not a PDF file: {pdf_path}")
        return None

    print("\n" + "="*80)
    print("🚀 CHONKIE OSS PDF CHUNKING TEST")
    print("="*80)
    print(f"\n📁 File: {pdf_path.name}")
    print(f"📏 Chunk Size: {chunk_size} tokens")
    print(f"🔄 Overlap: {overlap} tokens")
    print(f"⚙️  Chunker: {chunker_type}")
    print(f"\n{'─'*80}")
    print("Processing...")
    print(f"{'─'*80}\n")

    # Time the chunking
    start_time = time.time()

    try:
        chunks = chunk_document(
            file_path=str(pdf_path),
            chunk_size=chunk_size,
            overlap_tokens=overlap,
            tokenizer="cl100k_base",  # GPT-4 tokenizer
            chunker_type=chunker_type
        )

        elapsed = time.time() - start_time

        # Display results
        format_chunk_output(chunks, elapsed)

        # Save to JSON file
        output_file = pdf_path.parent / f"{pdf_path.stem}_chunks.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'source_file': str(pdf_path),
                'processing_time_ms': elapsed * 1000,
                'chunk_size': chunk_size,
                'overlap': overlap,
                'chunker_type': chunker_type,
                'total_chunks': len(chunks),
                'chunks': chunks
            }, f, indent=2, ensure_ascii=False)

        print(f"\n💾 Results saved to: {output_file}")

        return chunks

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n❌ Error after {elapsed:.2f}s: {e}")
        import traceback
        traceback.print_exc()
        return None


def interactive_mode():
    """Interactive mode for testing multiple PDFs."""
    print("\n" + "="*80)
    print("📚 INTERACTIVE PDF CHUNKING TEST")
    print("="*80)
    print("\nEnter PDF path (or 'quit' to exit)")

    while True:
        print("\n" + "─"*80)
        pdf_path = input("\n📁 PDF file path: ").strip()

        if pdf_path.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Goodbye!")
            break

        # Remove quotes if user copied path with quotes
        pdf_path = pdf_path.strip('"').strip("'")

        # Get parameters
        try:
            chunk_size = input(f"📏 Chunk size (default 512): ").strip()
            chunk_size = int(chunk_size) if chunk_size else 512

            overlap = input(f"🔄 Overlap (default 80): ").strip()
            overlap = int(overlap) if overlap else 80

            chunker_type = input(f"⚙️  Chunker type [recursive/token] (default recursive): ").strip()
            chunker_type = chunker_type if chunker_type in ['recursive', 'token'] else 'recursive'

        except ValueError:
            print("❌ Invalid input, using defaults")
            chunk_size = 512
            overlap = 80
            chunker_type = 'recursive'

        # Process the PDF
        test_pdf_chunking(pdf_path, chunk_size, overlap, chunker_type)


def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        # Command line mode
        pdf_path = sys.argv[1]
        chunk_size = int(sys.argv[2]) if len(sys.argv) > 2 else 512
        overlap = int(sys.argv[3]) if len(sys.argv) > 3 else 80
        chunker_type = sys.argv[4] if len(sys.argv) > 4 else 'recursive'

        test_pdf_chunking(pdf_path, chunk_size, overlap, chunker_type)
    else:
        # Interactive mode
        print("\n💡 Usage:")
        print(f"   {sys.argv[0]} <pdf_path> [chunk_size] [overlap] [chunker_type]")
        print(f"\nExample:")
        print(f"   {sys.argv[0]} document.pdf 512 80 recursive")

        interactive_mode()


if __name__ == "__main__":
    main()
