#!/usr/bin/env python3
"""
Quick verification script to test that enhanced metadata extraction is working.
"""
import json
from pathlib import Path

# Check if any test PDFs exist
test_pdf = Path("GPTV_System_Card.pdf")

if not test_pdf.exists():
    print("❌ Test PDF not found. Please ensure GPTV_System_Card.pdf is in the directory.")
    exit(1)

print("🔍 Testing enhanced metadata extraction...")
print("=" * 80)

# Import the enhanced chunker
try:
    from chonkie_oss_enhanced import chunk_document_with_metadata
    print("✅ Enhanced module imported successfully")
except ImportError as e:
    print(f"❌ Failed to import enhanced module: {e}")
    exit(1)

# Test chunking with metadata
print(f"\n📄 Processing: {test_pdf}")
print("⏱️  This may take a few seconds...\n")

try:
    result = chunk_document_with_metadata(
        file_path=str(test_pdf),
        chunk_size=512,
        overlap_tokens=80,
        tokenizer="cl100k_base",
        chunker_type="recursive",
        # METADATA EXTRACTION
        extract_metadata=True,
        detect_chapters=True,
        detect_fonts=True,
        detect_structure=True,
        pinecone_format=False
    )

    print("✅ Chunking completed successfully!")
    print(f"⏱️  Processing time: {result['stats']['processing_time_ms']:.0f}ms")
    print(f"📊 Total chunks: {result['stats']['total_chunks']}")
    print(f"📚 Total tokens: {result['stats']['total_tokens']}")

    # Verify first chunk has enhanced metadata
    print("\n" + "=" * 80)
    print("🔍 VERIFYING FIRST CHUNK METADATA:")
    print("=" * 80)

    first_chunk = result['chunks'][0]

    # Check for required fields
    required_fields = [
        'text', 'token_count', 'start_index', 'end_index',
        'pages', 'chapter', 'heading', 'position_in_doc',
        'has_tables', 'has_images', 'figure_refs', 'table_refs'
    ]

    missing_fields = []
    present_fields = []

    for field in required_fields:
        if field in first_chunk:
            present_fields.append(field)
            print(f"✅ {field:20s}: {first_chunk[field]}")
        else:
            missing_fields.append(field)
            print(f"❌ {field:20s}: MISSING")

    # Check document-level metadata
    print("\n" + "=" * 80)
    print("📋 DOCUMENT-LEVEL METADATA:")
    print("=" * 80)

    if 'metadata' in result and result['metadata']:
        metadata = result['metadata']
        print(f"✅ file_name: {metadata.get('file_name', 'N/A')}")
        print(f"✅ total_pages: {metadata.get('total_pages', 'N/A')}")
        print(f"✅ chapters detected: {len(metadata.get('chapters', []))}")

        if 'quality_score' in metadata:
            qs = metadata['quality_score']
            print(f"✅ quality_score: {qs.get('overall_quality', 'N/A')}/100")
        else:
            print("❌ quality_score: MISSING")
    else:
        print("❌ No document-level metadata found")

    # Summary
    print("\n" + "=" * 80)
    print("📊 SUMMARY:")
    print("=" * 80)

    if not missing_fields:
        print("🎉 SUCCESS! All enhanced metadata fields are present!")
        print("\n✅ The fix is working correctly.")
        print("✅ Web interface will now extract full metadata on new uploads.")
    else:
        print(f"⚠️  WARNING: {len(missing_fields)} fields are missing:")
        for field in missing_fields:
            print(f"   - {field}")
        print("\n❌ The enhanced metadata extraction may not be working correctly.")

    # Save test output
    output_file = Path("verify_metadata_output.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'first_chunk': first_chunk,
            'metadata': result.get('metadata', {}),
            'stats': result.get('stats', {})
        }, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Test output saved to: {output_file.absolute()}")

except Exception as e:
    print(f"❌ Error during chunking: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
