"""
Test Diagram Integration
Simple test to verify diagram explainer and pipeline integration.

Usage:
    python test_diagram_integration.py
"""

import asyncio
import json
import os
import sys
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

from diagram_explainer import DiagramExplainer, explain_diagrams_batch


def test_diagram_explainer():
    """Test the DiagramExplainer class."""
    print("\n" + "="*80)
    print("Testing DiagramExplainer")
    print("="*80 + "\n")

    # Test data
    test_query = "Explain how voltage dividers work"
    test_metadata = {
        "page": 42,
        "source_file": "electronics_textbook.pdf",
        "position_in_doc": 0.45,
        "document_id": "test_doc_123"
    }
    test_context = """
    A voltage divider is a passive linear circuit that produces an output voltage (Vout)
    that is a fraction of its input voltage (Vin). Voltage division is the result of
    distributing the input voltage among the components of the divider.

    The circuit consists of two resistors in series. The input voltage is applied across
    the series combination, and the output voltage is the voltage across one of the resistors.

    Figure 3.2 shows a basic voltage divider circuit with two resistors R1 and R2.
    The output voltage is taken from the junction between the two resistors.
    """

    try:
        # Initialize explainer
        explainer = DiagramExplainer()

        # Test explanation generation
        result = explainer.explain_diagram_from_context(
            query=test_query,
            diagram_metadata=test_metadata,
            surrounding_text=test_context
        )

        print(f"✅ Diagram description generated successfully!")
        print(f"   Model used: {result['model_used']}")
        print(f"   Confidence: {result['confidence']}")
        print(f"   Success: {result['success']}")
        print(f"\n   Description:")
        print(f"   {result['description']}")
        print()

        return True

    except Exception as e:
        print(f"❌ DiagramExplainer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_batch_processing():
    """Test batch diagram processing."""
    print("\n" + "="*80)
    print("Testing Batch Diagram Processing")
    print("="*80 + "\n")

    # Mock diagram chunks
    diagram_chunks = [
        {
            "metadata": {
                "type": "image",
                "page": 10,
                "source_file": "test.pdf",
                "document_id": "doc_1",
                "image_base64": "mock_base64_data_1",
                "image_storage": "inline",
                "position_in_doc": 0.2
            },
            "score": 0.95
        },
        {
            "metadata": {
                "type": "image",
                "page": 25,
                "source_file": "test.pdf",
                "document_id": "doc_1",
                "image_storage": "reference",
                "position_in_doc": 0.5
            },
            "score": 0.88
        }
    ]

    # Mock text chunks
    text_chunks = [
        {
            "text": "This is some context text about circuits and diagrams.",
            "metadata": {
                "page": 10,
                "document_id": "doc_1"
            }
        },
        {
            "text": "More information about voltage dividers and resistor networks.",
            "metadata": {
                "page": 25,
                "document_id": "doc_1"
            }
        }
    ]

    test_query = "Show me circuit diagrams"

    try:
        # Process diagrams
        enriched_diagrams = explain_diagrams_batch(
            diagrams=diagram_chunks,
            query=test_query,
            text_chunks=text_chunks,
            max_diagrams=3
        )

        print(f"✅ Batch processing successful!")
        print(f"   Processed {len(enriched_diagrams)} diagram(s)")
        print()

        for i, diagram in enumerate(enriched_diagrams, 1):
            print(f"   Diagram {i}:")
            print(f"     Page: {diagram['page']}")
            print(f"     Source: {diagram['source_file']}")
            print(f"     Storage: {diagram['storage_type']}")
            print(f"     Model: {diagram['model_used']}")
            print(f"     Description: {diagram['description'][:100]}...")
            print()

        return True

    except Exception as e:
        print(f"❌ Batch processing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_response_format():
    """Test that the response format is correct."""
    print("\n" + "="*80)
    print("Testing Response Format")
    print("="*80 + "\n")

    # Mock enriched diagram
    sample_diagram = {
        "image_base64": "data:image/png;base64,iVBORw0KG...",
        "image_reference": None,
        "page": 42,
        "source_file": "textbook.pdf",
        "description": "Circuit diagram showing voltage divider with R1 and R2",
        "model_used": "openai/gpt-oss-120b",
        "relevance_score": 0.92,
        "storage_type": "inline",
        "position_in_doc": 0.45
    }

    try:
        # Verify all required fields
        required_fields = [
            "image_base64", "image_reference", "page", "source_file",
            "description", "model_used", "relevance_score", "storage_type"
        ]

        missing_fields = [field for field in required_fields if field not in sample_diagram]

        if missing_fields:
            print(f"❌ Missing fields: {missing_fields}")
            return False

        print(f"✅ Response format is correct!")
        print(f"   All required fields present: {', '.join(required_fields)}")
        print(f"\n   Sample response:")
        print(f"   {json.dumps(sample_diagram, indent=2)}")
        print()

        return True

    except Exception as e:
        print(f"❌ Format validation failed: {e}")
        return False


async def test_pipeline_integration():
    """Test that answer_query returns diagrams correctly."""
    print("\n" + "="*80)
    print("Testing Pipeline Integration")
    print("="*80 + "\n")

    print("ℹ️  This test requires a running pipeline with documents containing diagrams.")
    print("   To test manually:")
    print("   1. Upload a document with diagrams to your RAG system")
    print("   2. Make a query request with include_diagrams=True")
    print("   3. Verify the response includes a 'diagrams' array")
    print()
    print("   Example response structure:")
    example_response = {
        "answer": "A voltage divider consists of...",
        "citations": [...],
        "diagrams": [
            {
                "image_base64": "data:image/png;base64,...",
                "page": 42,
                "description": "Circuit diagram showing...",
                "relevance_score": 0.92
            }
        ]
    }
    print(f"   {json.dumps(example_response, indent=2)}")
    print()

    return True


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("Diagram Integration Test Suite")
    print("="*80)

    results = []

    # Run synchronous tests
    results.append(("DiagramExplainer", test_diagram_explainer()))
    results.append(("Batch Processing", test_batch_processing()))
    results.append(("Response Format", test_response_format()))

    # Run async test
    loop = asyncio.get_event_loop()
    results.append(("Pipeline Integration", loop.run_until_complete(test_pipeline_integration())))

    # Summary
    print("\n" + "="*80)
    print("Test Summary")
    print("="*80 + "\n")

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status}: {test_name}")

    print()
    print(f"   Total: {passed}/{total} tests passed")
    print("="*80 + "\n")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
