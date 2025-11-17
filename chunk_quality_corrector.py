"""
Async Parallel Chunk Quality Corrector

This module provides fast quality detection and parallel LLM-based correction
for PDF chunks with character encoding issues.

Key features:
- Fast quality detection (regex-based, instant)
- Async parallel LLM correction (only for broken chunks)
- Groq LLM integration (blazing fast + free tier)
- Fallback to original text if LLM fails
- Comprehensive logging and metrics
"""

import re
import os
import asyncio
import logging
from typing import List, Dict, Any, Optional
from groq import AsyncGroq

logging.basicConfig(level=logging.INFO)


def calculate_quality_score(chunk_text: str) -> float:
    """
    Calculate a quality score (0-1) for chunk text.

    Higher score = better quality, less likely to need correction.
    Score below threshold (~0.65) indicates chunk needs LLM correction.

    This is VERY fast (regex-based) and runs on every chunk.

    Scoring criteria:
    - 1.0: Perfect text, no issues detected
    - 0.8-0.99: Minor issues, probably OK
    - 0.5-0.79: Moderate issues, might need correction
    - 0-0.49: Severe issues, definitely needs correction

    Args:
        chunk_text: The chunk text to score

    Returns:
        Quality score between 0 and 1
    """
    if not chunk_text or len(chunk_text) < 20:
        return 1.0  # Too short to judge, skip correction

    score = 1.0  # Start perfect
    sample = chunk_text[:1000]

    # Check 1: Mid-word spaces (broken words)
    # Look for patterns like "Artific ial", "computer er", "intell igence"
    # Pattern: lowercase word fragment + space + lowercase fragment (not a valid 2-word combo)
    broken_word_pattern = r'[a-z]{2,8}\s+[a-z]{2,8}(?=\s|$|[.,;!?])'
    potential_broken = re.findall(broken_word_pattern, sample.lower())

    # Filter out valid 2-word combinations
    # These are common valid phrases that look like broken words but aren't
    valid_two_word_combos = {
        'the a', 'the an', 'the and', 'the of', 'the to', 'the in', 'the is', 'the it', 'the for',
        'of the', 'of a', 'of an', 'of and', 'in the', 'in a', 'in an', 'to the', 'to a', 'to an',
        'is a', 'is an', 'is the', 'for the', 'for a', 'on the', 'at the', 'by the', 'from the',
        'with the', 'as a', 'as an', 'as the', 'this is', 'that is', 'it is', 'there is', 'there are',
        'will be', 'can be', 'may be', 'such as', 'as well', 'in order', 'in which', 'of which',
        'at least', 'at most', 'more than', 'less than', 'rather than', 'other than',
    }

    suspicious_broken = [w for w in potential_broken if w not in valid_two_word_combos]

    if len(suspicious_broken) >= 5:
        score -= 0.4  # Severe
    elif len(suspicious_broken) >= 3:
        score -= 0.3  # Moderate
    elif len(suspicious_broken) >= 2:
        score -= 0.2  # Minor
    elif len(suspicious_broken) >= 1:
        score -= 0.1  # Very minor

    # Check 2: Repeated fragments
    repeated_pattern = r'\b(\w{2,6})\s\1\b'
    repeated_matches = re.findall(repeated_pattern, sample)
    valid_repetitions = ['ha ha', 'no no', 'yes yes', 'ok ok', 'bye bye']
    suspicious_repeated = [r for r in repeated_matches if r.lower() not in [v.split()[0] for v in valid_repetitions]]

    if len(suspicious_repeated) >= 3:
        score -= 0.3
    elif len(suspicious_repeated) >= 1:
        score -= 0.15

    # Check 3: CID codes
    if '(cid:' in chunk_text[:500]:
        score -= 0.5  # Very bad

    # Check 4: Excessive special characters
    special_char_count = len(re.findall(r'[^\w\s.,!?;:\-\'"()]', sample))
    special_ratio = special_char_count / len(sample) if len(sample) > 0 else 0
    if special_ratio > 0.08:
        score -= 0.3
    elif special_ratio > 0.05:
        score -= 0.15

    # Check 5: Average word length (too short = fragmented)
    words = re.findall(r'\b\w+\b', sample)
    if words:
        avg_word_length = sum(len(w) for w in words) / len(words)
        if avg_word_length < 3.0:
            score -= 0.25
        elif avg_word_length < 3.5:
            score -= 0.15

    # Check 6: Chunk length heuristics
    if len(chunk_text) < 30:
        score -= 0.1  # Suspiciously short
    elif len(chunk_text) > 10000:
        score -= 0.05  # Suspiciously long

    # Check 7: High gibberish ratio (random character sequences)
    # Count sequences of 4+ consonants or unusual patterns
    gibberish_patterns = [
        r'[bcdfghjklmnpqrstvwxyz]{5,}',  # 5+ consonants
        r'[aeiou]{4,}',  # 4+ vowels
        r'\d{8,}',  # Long number sequences
    ]
    gibberish_count = sum(len(re.findall(p, sample.lower())) for p in gibberish_patterns)
    if gibberish_count > 5:
        score -= 0.2

    return max(0.0, min(1.0, score))  # Clamp to [0, 1]


def chunk_needs_correction(chunk_text: str, threshold: float = 0.65) -> bool:
    """
    Fast quality check to detect if a chunk needs LLM correction.

    This wraps calculate_quality_score() for backward compatibility.
    Uses a threshold-based approach: score < threshold = needs correction.

    Args:
        chunk_text: The chunk text to check
        threshold: Quality score threshold (default 0.65)

    Returns:
        True if chunk needs LLM correction, False if clean
    """
    score = calculate_quality_score(chunk_text)
    needs_fix = score < threshold

    if needs_fix:
        logging.debug(f"Chunk needs correction (score: {score:.2f} < {threshold})")

    return needs_fix


async def correct_chunk_with_llm(
    chunk: Dict[str, Any],
    client: AsyncGroq,
    model: Optional[str] = None,
    temperature: float = 0.1,
    timeout: int = 10
) -> Dict[str, Any]:
    """
    Fix a single chunk using async LLM call.

    This uses Groq's blazing-fast LLM with async for parallel execution.

    Args:
        chunk: Chunk dictionary with 'text' field
        client: AsyncGroq client instance
        model: Groq model to use (auto-selected if None)
        temperature: LLM temperature (low for consistency)
        timeout: Request timeout in seconds

    Returns:
        Chunk with corrected text (or original if correction fails)
    """
    original_text = chunk.get('text', '')

    if not original_text:
        return chunk

    # Default model if not specified
    if model is None:
        model = "llama-3.1-8b-instant"

    # Construct prompt
    prompt = f"""Fix OCR/character errors in this text. Follow these rules EXACTLY:

1. ONLY fix obvious character-level errors:
   - Mid-word spaces: "Artific ical" → "Artificial"
   - Repeated fragments: "Per ereption" → "Perception"
   - Broken suffixes: "definition tion" → "definition"

2. DO NOT rephrase or change meaning
3. PRESERVE all formatting (markdown headings, lists, structure)
4. Keep technical terms and proper nouns exactly as-is unless clearly broken
5. Maintain all markdown symbols (###, -, ---, etc.)

Text to fix:
{original_text}

Return ONLY the corrected text with no explanations or additional commentary."""

    try:
        # Async LLM call
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=min(len(original_text) + 500, 4096)  # Allow some expansion
            ),
            timeout=timeout
        )

        corrected_text = response.choices[0].message.content.strip()

        # Sanity checks
        if not corrected_text:
            logging.warning(f"LLM returned empty response for chunk")
            return chunk

        # Check if changed too much (likely hallucination)
        if len(corrected_text) > len(original_text) * 1.5:
            logging.warning(f"LLM changed text too much ({len(original_text)} → {len(corrected_text)}), using original")
            return chunk

        # Check if markdown structure preserved
        original_headings = len(re.findall(r'^#+\s', original_text, re.MULTILINE))
        corrected_headings = len(re.findall(r'^#+\s', corrected_text, re.MULTILINE))

        if original_headings != corrected_headings:
            logging.warning(f"LLM changed markdown structure ({original_headings} → {corrected_headings} headings), using original")
            return chunk

        # Success - update chunk
        chunk['text'] = corrected_text
        chunk['was_llm_corrected'] = True
        chunk['original_length'] = len(original_text)
        chunk['corrected_length'] = len(corrected_text)
        chunk['correction_model'] = model

        logging.info(f"✅ Corrected chunk with {model} ({len(original_text)} → {len(corrected_text)} chars)")

        return chunk

    except asyncio.TimeoutError:
        logging.error(f"LLM correction timeout after {timeout}s, using original")
        chunk['was_llm_corrected'] = False
        chunk['correction_error'] = 'timeout'
        return chunk

    except Exception as e:
        logging.error(f"LLM correction failed: {e}, using original")
        chunk['was_llm_corrected'] = False
        chunk['correction_error'] = str(e)
        return chunk


async def process_single_chunk(
    chunk: Dict[str, Any],
    client: AsyncGroq,
    enable_correction: bool = True,
    model: Optional[str] = None,
    quality_threshold: float = 0.65
) -> Dict[str, Any]:
    """
    Process a single chunk: check quality, correct if needed.

    This is the per-chunk processing unit that runs in parallel.

    OPTIMIZATION: Only corrects chunks with quality score < threshold.
    This saves 60-80% on LLM costs by skipping high-quality chunks.

    Args:
        chunk: Chunk to process
        client: AsyncGroq client
        enable_correction: If False, skip LLM correction
        model: Model to use for correction (auto-selected if None)
        quality_threshold: Only correct if score < threshold (default 0.65)

    Returns:
        Processed chunk (corrected or original)
    """
    chunk_text = chunk.get('text', '')

    # Calculate quality score (fast, always runs)
    quality_score = calculate_quality_score(chunk_text)
    chunk['quality_score'] = round(quality_score, 3)

    # Check if correction needed based on threshold
    needs_correction = quality_score < quality_threshold

    if needs_correction and enable_correction:
        # Async LLM correction
        return await correct_chunk_with_llm(chunk, client, model=model)
    else:
        # No correction needed or disabled
        chunk['was_llm_corrected'] = False
        chunk['needs_correction'] = needs_correction
        chunk['skip_reason'] = 'high_quality' if quality_score >= quality_threshold else 'correction_disabled'
        return chunk


async def process_chunks_in_parallel(
    chunks: List[Dict[str, Any]],
    enable_correction: bool = True,
    max_concurrent: int = 10,
    quality_threshold: float = 0.65
) -> List[Dict[str, Any]]:
    """
    Process all chunks in parallel with selective quality correction.

    This is the main entry point for parallel chunk processing.
    OPTIMIZATION: Only corrects chunks below quality threshold.

    Typical savings with threshold=0.65:
    - 60-80% fewer LLM calls
    - 3-5x faster processing
    - Dramatically lower costs

    Uses model rotation across multiple Groq models to distribute load
    and avoid rate limits.

    Args:
        chunks: List of chunks to process
        enable_correction: Enable LLM correction (set False to skip)
        max_concurrent: Max concurrent LLM calls (default 10)
        quality_threshold: Only correct chunks with score < threshold (default 0.65)
                          Lower = more selective (fewer corrections)
                          Higher = more aggressive (more corrections)
                          Range: 0.0 to 1.0

    Returns:
        List of processed chunks (corrected where needed)
    """
    # Model rotation: distribute load across multiple Groq free tier models
    # Each model has separate rate limits (~30 req/min)
    AVAILABLE_MODELS = [
        "llama-3.1-8b-instant",      # Fast, good quality
        "openai/gpt-oss-120b",      # Fast, good quality
        "openai/gpt-oss-20b",        # Good for complex text
        "llama-3.3-70b-versatile",   # High quality
        "meta-llama/llama-guard-4-12b",              # Alternative option
        "qwen/qwen3-32b",              # Alternative option
    ]
    if not chunks:
        return []

    # Initialize async Groq client
    groq_api_key = os.getenv("GROQ_API_KEY")

    if not groq_api_key:
        logging.warning("GROQ_API_KEY not found, skipping LLM correction")
        enable_correction = False

    if not enable_correction:
        # Fast path: no correction, just mark all as uncorrected
        for chunk in chunks:
            chunk['was_llm_corrected'] = False
        return chunks

    client = AsyncGroq(api_key=groq_api_key)

    # Pre-scan: Calculate quality scores for all chunks (fast)
    logging.info(f"🔍 Calculating quality scores for {len(chunks)} chunks...")
    quality_scores = [calculate_quality_score(c.get('text', '')) for c in chunks]

    # Quality distribution stats
    chunks_needing_correction = sum(1 for score in quality_scores if score < quality_threshold)
    avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0

    # Show quality distribution
    score_ranges = {
        'Perfect (0.9-1.0)': sum(1 for s in quality_scores if s >= 0.9),
        'Good (0.7-0.89)': sum(1 for s in quality_scores if 0.7 <= s < 0.9),
        'Fair (0.5-0.69)': sum(1 for s in quality_scores if 0.5 <= s < 0.7),
        'Poor (0-0.49)': sum(1 for s in quality_scores if s < 0.5),
    }

    logging.info(f"📊 Quality Distribution:")
    for range_name, count in score_ranges.items():
        percentage = (count / len(chunks) * 100) if chunks else 0
        logging.info(f"   {range_name}: {count} chunks ({percentage:.1f}%)")

    logging.info(f"📈 Average quality score: {avg_quality:.3f}")
    logging.info(f"🎯 Threshold: {quality_threshold} → {chunks_needing_correction}/{len(chunks)} chunks need correction")

    savings_pct = ((len(chunks) - chunks_needing_correction) / len(chunks) * 100) if chunks else 0
    logging.info(f"💰 LLM cost savings: {savings_pct:.1f}% (skipping {len(chunks) - chunks_needing_correction} high-quality chunks)")

    if chunks_needing_correction == 0:
        # Fast path: all chunks are clean
        logging.info("✅ All chunks are high quality, skipping LLM correction")
        for chunk in chunks:
            chunk['was_llm_corrected'] = False
            chunk['quality_score'] = round(calculate_quality_score(chunk.get('text', '')), 3)
        return chunks

    # Process all chunks in parallel (with concurrency limit)
    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_with_semaphore(chunk, chunk_index):
        # Rotate models in round-robin fashion to distribute load
        model = AVAILABLE_MODELS[chunk_index % len(AVAILABLE_MODELS)]
        async with semaphore:
            return await process_single_chunk(chunk, client, enable_correction, model=model, quality_threshold=quality_threshold)

    logging.info(f"🚀 Processing {len(chunks)} chunks in parallel (max {max_concurrent} concurrent)...")
    logging.info(f"📊 Using {len(AVAILABLE_MODELS)} models in rotation: {', '.join(AVAILABLE_MODELS)}")

    start_time = asyncio.get_event_loop().time()

    # Run ALL chunks in parallel with model rotation!
    corrected_chunks = await asyncio.gather(
        *[process_with_semaphore(chunk, i) for i, chunk in enumerate(chunks)],
        return_exceptions=True  # Don't fail entire batch if one fails
    )

    elapsed_time = asyncio.get_event_loop().time() - start_time

    # Handle any exceptions
    final_chunks = []
    for i, result in enumerate(corrected_chunks):
        if isinstance(result, Exception):
            logging.error(f"Chunk {i} processing failed: {result}")
            # Use original chunk
            chunks[i]['was_llm_corrected'] = False
            chunks[i]['correction_error'] = str(result)
            final_chunks.append(chunks[i])
        else:
            final_chunks.append(result)

    # Log detailed statistics
    actually_corrected = sum(1 for c in final_chunks if c.get('was_llm_corrected'))
    skipped_high_quality = sum(1 for c in final_chunks if c.get('skip_reason') == 'high_quality')
    had_errors = sum(1 for c in final_chunks if c.get('correction_error'))

    logging.info(f"✅ Parallel processing complete in {elapsed_time:.2f}s")
    logging.info(f"📊 Final Stats:")
    logging.info(f"   ✓ Corrected: {actually_corrected} chunks")
    logging.info(f"   ⏭️  Skipped (high quality): {skipped_high_quality} chunks")
    logging.info(f"   ⚠️  Errors: {had_errors} chunks")
    logging.info(f"   📈 Success rate: {(actually_corrected / chunks_needing_correction * 100):.1f}% of low-quality chunks corrected")

    # Calculate time saved
    time_per_llm_call = 0.5  # Estimate: ~500ms per LLM call
    time_saved = skipped_high_quality * time_per_llm_call
    logging.info(f"⚡ Estimated time saved by selective correction: {time_saved:.1f}s")

    return final_chunks


# Synchronous wrapper for backward compatibility
def process_chunks_sync(
    chunks: List[Dict[str, Any]],
    enable_correction: bool = True,
    quality_threshold: float = 0.65
) -> List[Dict[str, Any]]:
    """
    Synchronous wrapper for process_chunks_in_parallel.

    Use this if calling from non-async code.

    Args:
        chunks: List of chunks to process
        enable_correction: Enable LLM correction
        quality_threshold: Only correct chunks with score < threshold (default 0.65)

    Returns:
        List of processed chunks
    """
    return asyncio.run(process_chunks_in_parallel(chunks, enable_correction, quality_threshold=quality_threshold))
