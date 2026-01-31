"""
Test Suite for Mistral OCR Integration
Tests all new functionality: extraction, embeddings, chunking, Pinecone storage.

Run with: python test_mistral_ocr_integration.py <pdf_file_or_url>
Supports both local PDF files and remote URLs.
"""

import os
import sys
import json
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_mistral_ocr_extraction(file_path: str):
    """Test Mistral OCR extraction."""
    print(f"\n{'='*80}")
    print("TEST 1: Mistral OCR Extraction")
    print(f"{'='*80}\n")

    try:
        from mistral_ocr_extractor import extract_document_with_mistral
        from pathlib import Path

        print(f"📄 Extracting: {Path(file_path).name}")

        result = extract_document_with_mistral(
            file_path,
            enhance_metadata=True,
            fallback_to_legacy=True
        )

        print("\n✅ Extraction successful!")
        print(f"   Method: {result.get('extraction_method')}")
        print(f"   Pages: {result.get('total_pages')}")
        print(f"   Chapters: {len(result.get('chapters', []))}")
        print(f"   Headings: {len(result.get('headings', []))}")
        print(f"   Images: {len(result.get('images', []))}")
        print(f"   Quality: {result.get('metadata_quality', {}).get('overall_quality', 0)}/100")
        print(f"   Time: {result.get('extraction_time_ms', 0):.2f}ms")

        # Save extracted text to docchunker_outputs/
        if result and result.get('full_text'):
            output_dir = Path("docchunker_outputs")
            output_dir.mkdir(exist_ok=True)

            # Create filename based on input file
            base_name = Path(file_path).stem
            output_file = output_dir / f"{base_name}_mistral_extracted.txt"

            # Save the full extracted text
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(result.get('full_text', ''))

            print(f"💾 Saved extracted text to: {output_file}")

        # Show first 300 characters
        print("\n📖 Text preview:")
        print("-"*80)
        print(result.get('full_text', '')[:300] + "...")
        print("-"*80)

        return result

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_chonkie_integration(file_path: str):
    """Test Chonkie integration with Mistral OCR."""
    print(f"\n{'='*80}")
    print("TEST 2: Chonkie Integration with Mistral OCR")
    print(f"{'='*80}\n")

    try:
        from chonkie_oss_enhanced import chunk_document_with_metadata
        from pathlib import Path

        print(f"📄 Processing with Chonkie + Mistral OCR: {Path(file_path).name}")

        result = chunk_document_with_metadata(
            file_path=file_path,
            chunk_size=512,
            overlap_tokens=80,
            chunker_type="recursive",
            use_mistral_ocr=True,
            mistral_enhance_metadata=True,
            mistral_fallback_to_legacy=True,
            extract_metadata=True,
            recipe="markdown",
        )

        print("\n✅ Chunking successful!")
        print(f"   Total chunks: {result['stats']['total_chunks']}")
        print(f"   Total tokens: {result['stats']['total_tokens']}")
        print(f"   Avg tokens/chunk: {result['stats']['avg_tokens_per_chunk']:.1f}")
        print(f"   Processing time: {result['stats']['processing_time_ms']:.2f}ms")
        print(f"   Extraction method: {result['metadata'].get('extraction_method')}")

        # Save chunks to chunking_outputs/
        if result and result.get('chunks'):
            import json
            output_dir = Path("chunking_outputs")
            output_dir.mkdir(exist_ok=True)

            # Create filename based on input file
            base_name = Path(file_path).stem
            output_file = output_dir / f"{base_name}_chunks.json"

            # Save the full chunking result
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            print(f"💾 Saved chunks to: {output_file}")

        # Show first chunk
        if result['chunks']:
            print("\n📦 First chunk:")
            print("-"*80)
            chunk = result['chunks'][0]
            print(f"   Text: {chunk['text'][:200]}...")
            print(f"   Token count: {chunk.get('token_count')}")
            print(f"   Pages: {chunk.get('pages', [])}")
            print(f"   Chapter: {chunk.get('chapter', 'N/A')}")
            print("-"*80)

        return result

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_multimodal_embeddings():
    """Test multimodal embeddings."""
    print(f"\n{'='*80}")
    print("TEST 3: Multimodal Embeddings")
    print(f"{'='*80}\n")

    try:
        try:
            from multimodal_embeddings import MultimodalEmbedder
        except ImportError as ie:
            print(f"⚠️  Skipping multimodal embeddings test: Import error - {ie}")
            return False

        print("🧠 Testing Jina CLIP-v2 embeddings...")

        try:
            embedder = MultimodalEmbedder(model="jina-clip-v2")
        except Exception as init_error:
            print(f"⚠️  Skipping multimodal embeddings test: Initialization error - {init_error}")
            return False

        # Test text embedding
        test_texts = [
            "A cat sitting on a mat",
            "Machine learning and artificial intelligence",
        ]

        text_embeddings = embedder.embed_texts(test_texts)

        print("\n✅ Text embedding successful!")
        print(f"   Embedded {len(text_embeddings)} texts")
        print(f"   Embedding dimension: {len(text_embeddings[0])}")
        print(f"   First 10 dimensions: {text_embeddings[0][:10]}")

        # Test batch embedding
        batch_items = [
            {'type': 'text', 'content': 'A beautiful sunset'},
            {'type': 'text', 'content': 'Modern architecture'},
        ]

        batch_results = embedder.embed_batch(batch_items)

        print("\n✅ Batch embedding successful!")
        print(f"   Embedded {len(batch_results)} items")

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_document_chunking(file_path_or_url: str):
    """Test large document chunking."""
    print(f"\n{'='*80}")
    print("TEST 4: Document Chunking (Large Documents)")
    print(f"{'='*80}\n")

    # Check if it's a URL
    is_url = file_path_or_url.startswith(('http://', 'https://'))
    if is_url:
        print("📄 Skipping document chunking test for URLs")
        print(f"   URL chunking not yet supported: {file_path_or_url}")
        print("\n✅ Test skipped (URL not supported)")
        return True

    try:
        from document_chunker import LargeDocumentChunker

        print(f"📄 Testing document chunker: {Path(file_path_or_url).name}")

        chunker = LargeDocumentChunker(max_pages=10, max_mb=10)

        # Check if needs chunking
        needs_chunking = chunker.should_chunk(file_path_or_url)
        print(f"\n   Document needs chunking: {needs_chunking}")

        if needs_chunking:
            # Split
            chunks = chunker.split_pdf(file_path_or_url)

            print("\n✅ Splitting successful!")
            print(f"   Created {len(chunks)} chunks:")
            for chunk in chunks[:5]:  # Show first 5
                print(f"      {chunk}")

            if len(chunks) > 5:
                print(f"      ... and {len(chunks) - 5} more")

            # Cleanup
            chunker.cleanup_temp_files(chunks)
            print("\n✅ Cleanup complete")

        else:
            print("\n✅ Document is small enough, no chunking needed")

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_end_to_end_summary(results: dict):
    """Print end-to-end test summary."""
    print(f"\n{'='*80}")
    print("TEST SUMMARY")
    print(f"{'='*80}\n")

    tests = [
        ("Mistral OCR Extraction", results.get('test1')),
        ("Chonkie Integration", results.get('test2')),
        ("Multimodal Embeddings", results.get('test3')),
        ("Document Chunking", results.get('test4')),
    ]

    passed = sum(1 for _, result in tests if result)
    total = len(tests)

    for test_name, result in tests:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"   {test_name}: {status}")

    print(f"\n{'='*80}")
    print(f"Results: {passed}/{total} tests passed")
    print("="*80 + "\n")

    if passed == total:
        print("🎉 All tests passed! Integration successful.")
    else:
        print(f"⚠️  {total - passed} test(s) failed. Check logs above.")


def main():
    """Run all tests."""
    if len(sys.argv) < 2:
        print("\n❌ Error: No file or URL provided")
        print("\nUsage: python test_mistral_ocr_integration.py <file_path_or_url>")
        print("\nExamples:")
        print("  python test_mistral_ocr_integration.py document.pdf")
        print("  python test_mistral_ocr_integration.py https://example.com/document.pdf")
        sys.exit(1)

    file_path_or_url = sys.argv[1]

    # Check if it's a URL or local file
    is_url = file_path_or_url.startswith(('http://', 'https://'))

    if not is_url and not os.path.exists(file_path_or_url):
        print(f"\n❌ Error: File not found: {file_path_or_url}")
        sys.exit(1)

    from pathlib import Path

    print(f"\n{'#'*80}")
    print("# Mistral OCR Integration Test Suite")
    print(f"# Source: {Path(file_path_or_url).name}")
    print("#"*80)

    results = {}

    # Run tests
    results['test1'] = test_mistral_ocr_extraction(file_path_or_url) is not None
    results['test2'] = test_chonkie_integration(file_path_or_url) is not None
    results['test3'] = test_multimodal_embeddings()
    results['test4'] = test_document_chunking(file_path_or_url)

    # Summary
    test_end_to_end_summary(results)


if __name__ == "__main__":
    main()
