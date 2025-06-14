#!/usr/bin/env python3
"""
Test script for the new 500-character chunking functionality
"""

import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from ingest import chunk_text

def test_chunking():
    """Test the chunking function with various text sizes"""
    
    print("=" * 60)
    print("TESTING 500-CHARACTER CHUNKING")
    print("=" * 60)
    
    # Test 1: Short text (should return single chunk)
    short_text = "This is a short text that should fit in one chunk."
    chunks = chunk_text(short_text)
    print(f"\nTest 1 - Short text ({len(short_text)} chars):")
    print(f"  Input: '{short_text}'")
    print(f"  Chunks: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        print(f"    Chunk {i+1}: '{chunk}' ({len(chunk)} chars)")
    
    # Test 2: Medium text (should be chunked)
    medium_text = """This is a longer text that should be split into multiple chunks. 
    It contains multiple sentences and should demonstrate how the chunking algorithm works with word boundaries. 
    The algorithm should try to break at word boundaries when possible to maintain readability and context. 
    Each chunk should be approximately 500 characters with some overlap between consecutive chunks to preserve context. 
    This helps ensure that information spanning chunk boundaries is not lost in the embedding process."""
    
    chunks = chunk_text(medium_text)
    print(f"\nTest 2 - Medium text ({len(medium_text)} chars):")
    print(f"  Chunks: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        print(f"    Chunk {i+1}: '{chunk[:50]}...' ({len(chunk)} chars)")
    
    # Test 3: Very long text (multiple chunks)
    long_text = """Artificial Intelligence (AI) represents one of the most transformative technologies of our time, fundamentally changing how we approach problem-solving, data analysis, and decision-making across numerous industries. From its early conceptual roots in the 1950s to today's sophisticated machine learning algorithms and neural networks, AI has evolved from theoretical computer science concepts into practical tools that power everything from search engines to autonomous vehicles. The field encompasses various subdomains including machine learning, natural language processing, computer vision, robotics, and expert systems, each contributing unique capabilities to the broader AI ecosystem. Modern AI systems leverage vast amounts of data and computational power to identify patterns, make predictions, and generate insights that would be impossible for humans to process manually. Deep learning, a subset of machine learning inspired by the structure and function of the human brain, has been particularly revolutionary, enabling breakthroughs in image recognition, speech processing, and language understanding. These advances have practical applications in healthcare, where AI assists in diagnostic imaging and drug discovery; in finance, where algorithms detect fraud and optimize trading strategies; in transportation, where self-driving cars navigate complex environments; and in entertainment, where recommendation systems personalize content for millions of users. However, the rapid advancement of AI also raises important ethical considerations regarding privacy, bias, job displacement, and the need for transparent and accountable AI systems that serve humanity's best interests."""
    
    chunks = chunk_text(long_text)
    print(f"\nTest 3 - Long text ({len(long_text)} chars):")
    print(f"  Chunks: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        print(f"    Chunk {i+1}: '{chunk[:50]}...' ({len(chunk)} chars)")
        if i < len(chunks) - 1:
            # Check overlap
            next_chunk = chunks[i+1]
            overlap = ""
            for j in range(min(50, len(chunk))):
                if chunk[-j:] in next_chunk[:100]:
                    overlap = chunk[-j:]
            if overlap:
                print(f"      Overlap with next: '{overlap}'")
    
    # Test 4: Edge case - very short text
    tiny_text = "Short."
    chunks = chunk_text(tiny_text)
    print(f"\nTest 4 - Tiny text ({len(tiny_text)} chars):")
    print(f"  Input: '{tiny_text}'")
    print(f"  Chunks: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        print(f"    Chunk {i+1}: '{chunk}' ({len(chunk)} chars)")
    
    # Test 5: Empty text
    empty_text = ""
    chunks = chunk_text(empty_text)
    print(f"\nTest 5 - Empty text:")
    print(f"  Chunks: {len(chunks)}")
    
    print("\n" + "=" * 60)
    print("CHUNKING TESTS COMPLETED")
    print("=" * 60)

if __name__ == "__main__":
    test_chunking()