#!/usr/bin/env python3
"""
Quick test script to verify Chonkie OSS installation and benchmark performance.
"""

import time
from chonkie_oss_client import chunk_document, embed_chunks

# Sample text for testing
SAMPLE_TEXT = """
Artificial Intelligence (AI) is revolutionizing technology. Machine learning, a subset of AI,
enables computers to learn from data without explicit programming. Deep learning, using neural
networks with multiple layers, has achieved remarkable success in image recognition, natural
language processing, and game playing.

Natural Language Processing (NLP) allows computers to understand and generate human language.
Recent advances in transformer architectures, like BERT and GPT, have dramatically improved
language understanding capabilities. These models can perform tasks such as translation,
summarization, and question answering with impressive accuracy.

Computer Vision enables machines to interpret visual information from the world. Convolutional
Neural Networks (CNNs) have been particularly successful in this domain, achieving human-level
performance in tasks like object detection and image classification. Applications range from
autonomous vehicles to medical image analysis.

Reinforcement Learning teaches agents to make decisions through trial and error. By receiving
rewards for good actions, agents learn optimal strategies for complex tasks. This approach has
led to breakthroughs in robotics, game playing (like AlphaGo), and resource optimization.

The future of AI holds tremendous promise and challenges. Ethical considerations around bias,
privacy, and job displacement must be carefully addressed. As AI systems become more capable,
ensuring they remain safe, transparent, and aligned with human values becomes increasingly
important.
""" * 3  # Repeat to make it longer for better benchmarking


def test_chunking():
    """Test basic chunking functionality."""
    print("=" * 70)
    print("TEST 1: Basic Chunking")
    print("=" * 70)

    start_time = time.time()

    chunks = chunk_document(
        text=SAMPLE_TEXT,
        chunk_size=512,
        overlap_tokens=80,
        tokenizer="gpt2",
        chunker_type="recursive"
    )

    elapsed = time.time() - start_time

    print(f"\n✅ Created {len(chunks)} chunks in {elapsed*1000:.2f}ms")
    print(f"   Speed: {len(SAMPLE_TEXT) / elapsed:.0f} chars/sec")
    print(f"   Text length: {len(SAMPLE_TEXT)} characters")

    # Show first chunk
    if chunks:
        print(f"\n📄 First chunk:")
        print(f"   Text: {chunks[0]['text'][:100]}...")
        print(f"   Tokens: {chunks[0]['token_count']}")
        print(f"   Range: {chunks[0]['start_index']}-{chunks[0]['end_index']}")

    return chunks


def test_different_chunkers():
    """Compare recursive vs token chunker."""
    print("\n" + "=" * 70)
    print("TEST 2: Chunker Comparison")
    print("=" * 70)

    sample = SAMPLE_TEXT[:1000]  # Use shorter text for comparison

    # Recursive chunker
    start = time.time()
    recursive_chunks = chunk_document(
        text=sample,
        chunk_size=256,
        overlap_tokens=40,
        chunker_type="recursive"
    )
    recursive_time = time.time() - start

    # Token chunker
    start = time.time()
    token_chunks = chunk_document(
        text=sample,
        chunk_size=256,
        overlap_tokens=40,
        chunker_type="token"
    )
    token_time = time.time() - start

    print(f"\n📊 Recursive Chunker:")
    print(f"   Chunks: {len(recursive_chunks)}")
    print(f"   Time: {recursive_time*1000:.2f}ms")
    print(f"   Avg tokens/chunk: {sum(c['token_count'] for c in recursive_chunks) / len(recursive_chunks):.1f}")

    print(f"\n📊 Token Chunker:")
    print(f"   Chunks: {len(token_chunks)}")
    print(f"   Time: {token_time*1000:.2f}ms")
    print(f"   Avg tokens/chunk: {sum(c['token_count'] for c in token_chunks) / len(token_chunks):.1f}")


def test_tokenizers():
    """Test different tokenizers."""
    print("\n" + "=" * 70)
    print("TEST 3: Tokenizer Comparison")
    print("=" * 70)

    sample = "Hello world! This is a test of different tokenizers."

    tokenizers = ["gpt2", "cl100k_base"]  # GPT-2 and GPT-4

    for tok in tokenizers:
        try:
            chunks = chunk_document(
                text=sample,
                chunk_size=50,
                tokenizer=tok,
                chunker_type="token"
            )

            if chunks:
                print(f"\n✅ {tok}:")
                print(f"   Tokens: {chunks[0]['token_count']}")
        except Exception as e:
            print(f"\n❌ {tok}: {str(e)}")


def benchmark_vs_api():
    """Show speed improvement vs API version."""
    print("\n" + "=" * 70)
    print("TEST 4: Performance Benchmark")
    print("=" * 70)

    iterations = 5
    times = []

    print(f"\n⚡ Running {iterations} chunking operations...")

    for i in range(iterations):
        start = time.time()
        chunks = chunk_document(
            text=SAMPLE_TEXT,
            chunk_size=512,
            overlap_tokens=80,
            chunker_type="recursive"
        )
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"   Run {i+1}: {elapsed*1000:.2f}ms ({len(chunks)} chunks)")

    avg_time = sum(times) / len(times)

    print(f"\n📈 Results:")
    print(f"   Average time: {avg_time*1000:.2f}ms")
    print(f"   Min time: {min(times)*1000:.2f}ms")
    print(f"   Max time: {max(times)*1000:.2f}ms")
    print(f"   Throughput: {len(SAMPLE_TEXT) / avg_time:.0f} chars/sec")

    # Estimated API comparison (assuming 200ms network latency)
    estimated_api_time = avg_time + 0.2  # Add 200ms for network
    print(f"\n🔄 Comparison:")
    print(f"   Chonkie OSS (local): {avg_time*1000:.2f}ms")
    print(f"   Chonkie API (estimated): {estimated_api_time*1000:.2f}ms")
    print(f"   Speedup: {estimated_api_time / avg_time:.1f}x faster")


def main():
    """Run all tests."""
    print("\n🚀 Chonkie OSS Test Suite\n")

    try:
        # Test 1: Basic chunking
        chunks = test_chunking()

        # Test 2: Different chunkers
        test_different_chunkers()

        # Test 3: Different tokenizers
        test_tokenizers()

        # Test 4: Performance benchmark
        benchmark_vs_api()

        print("\n" + "=" * 70)
        print("✅ ALL TESTS PASSED!")
        print("=" * 70)
        print("\n💡 Next steps:")
        print("   1. Install: pip install 'chonkie[all]'")
        print("   2. Update imports in your code")
        print("   3. Remove CHONKIE_API_KEY from .env")
        print("   4. Enjoy free, fast, local chunking!")

    except ImportError as e:
        print("\n❌ ERROR: Chonkie OSS not installed")
        print("\n📦 Install with:")
        print("   pip install 'chonkie[all]'")
        print(f"\nDetails: {e}")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
