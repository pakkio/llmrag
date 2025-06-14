#!/usr/bin/env python3
"""
Test script for LLM wrapper environment validation
Tests both OpenRouter LLM API and OpenAI embeddings API
"""

import os
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from llm_wrapper import (
    test_llm_connection, 
    test_openai_embeddings as test_embedding_api, 
    generate_embeddings,
    llm_call
)

def test_environment_variables():
    """Test if required environment variables are set"""
    print("=" * 50)
    print("ENVIRONMENT VARIABLES TEST")
    print("=" * 50)
    
    required_vars = {
        "OPENROUTER_API_KEY": "OpenRouter API key for LLM calls",
        "OPENAI_API_KEY": "OpenAI API key for embeddings"
    }
    
    optional_vars = {
        "SEMANTIC_MODEL": "LLM model to use",
        "OPENROUTER_APP_URL": "App URL for OpenRouter",
        "OPENROUTER_APP_TITLE": "App title for OpenRouter",
        "PAK_DEBUG": "Debug mode flag"
    }
    
    missing_vars = []
    
    for var, description in required_vars.items():
        value = os.environ.get(var)
        if value:
            # Show only first 10 and last 4 characters for security
            masked_value = value[:10] + "..." + value[-4:] if len(value) > 14 else "***"
            print(f"✓ {var}: {masked_value} ({description})")
        else:
            print(f"✗ {var}: NOT SET ({description})")
            missing_vars.append(var)
    
    print("\nOptional variables:")
    for var, description in optional_vars.items():
        value = os.environ.get(var)
        if value:
            print(f"✓ {var}: {value} ({description})")
        else:
            print(f"- {var}: not set ({description})")
    
    assert not missing_vars, f"Missing required environment variables: {', '.join(missing_vars)}"

def test_openrouter_llm():
    """Test OpenRouter LLM API connection"""
    print("\n" + "=" * 50)
    print("OPENROUTER LLM API TEST")
    print("=" * 50)
    
    # Test basic connection
    print("Testing basic LLM connection...")
    from llm_wrapper import test_llm_connection as test_llm_conn
    basic_result = test_llm_conn()
    assert basic_result, "Basic LLM connection test failed"
    print("✓ Basic LLM connection test passed")
    
    # Test actual call with specific content
    print("\nTesting LLM call with math question...")
    messages = [{"role": "user", "content": "What is 2+2? Answer with just the number."}]
    response, success = llm_call(messages, max_tokens=50)
    
    assert success, f"LLM call failed. Response: '{response}'"
    print(f"✓ LLM call successful. Response: '{response}'")
    
    if "4" in response:
        print("✓ LLM gave correct answer")
    else:
        print("⚠ LLM response seems unusual but call succeeded")

def test_openai_embeddings():
    """Test OpenAI embeddings API"""
    print("\n" + "=" * 50)
    print("OPENAI EMBEDDINGS API TEST")
    print("=" * 50)
    
    # Test basic embedding server
    print("Testing embedding server connection...")
    from llm_wrapper import test_openai_embeddings as test_openai_emb
    basic_result = test_openai_emb()
    assert basic_result, "Basic embedding server test failed"
    print("✓ Basic embedding server test passed")
    
    # Test actual embedding generation
    print("\nTesting embedding generation...")
    test_texts = [
        "This is a test sentence for embedding generation.",
        "Another test sentence with different content."
    ]
    
    # Test single embedding
    print("Testing single text embedding...")
    embedding = generate_embeddings(test_texts[0])
    print(f"✓ Single embedding generated. Shape: {embedding.shape}, Type: {type(embedding)}")
    
    # Test batch embeddings
    print("Testing batch embeddings...")
    embeddings = generate_embeddings(test_texts)
    print(f"✓ Batch embeddings generated. Count: {len(embeddings)}")
    
    # Test embedding properties
    print("\nTesting embedding properties...")
    print(f"✓ Embedding dimension: {len(embedding)}")
    print(f"✓ Embedding norm: {float(sum(x*x for x in embedding)**0.5):.4f}")
    
    # Test similarity between different texts
    import numpy as np
    emb1 = generate_embeddings(test_texts[0])
    emb2 = generate_embeddings(test_texts[1])
    similarity = np.dot(emb1, emb2)
    print(f"✓ Similarity between test texts: {similarity:.4f}")
    
    # Basic assertions
    assert len(embedding) == 3072, f"Expected embedding dimension 3072, got {len(embedding)}"
    assert len(embeddings) == 2, f"Expected 2 embeddings, got {len(embeddings)}"

def test_llm_connection():
    """Test LLM connection (pytest wrapper)"""
    from llm_wrapper import test_llm_connection as test_llm_conn
    result = test_llm_conn()
    assert result, "LLM connection test failed"

def test_embedding_api():
    """Test embedding API (pytest wrapper)"""
    from llm_wrapper import test_openai_embeddings as test_openai_emb
    result = test_openai_emb()
    assert result, "Embedding API test failed"

def run_comprehensive_test():
    """Run all tests and provide summary"""
    print("LLM WRAPPER ENVIRONMENT TEST")
    print("=" * 50)
    
    # Test environment variables
    env_ok = test_environment_variables()
    
    # Test LLM API
    llm_ok = test_openrouter_llm() if env_ok else False
    
    # Test embeddings API
    embeddings_ok = test_openai_embeddings() if env_ok else False
    
    # Final summary
    print("\n" + "=" * 50)
    print("FINAL SUMMARY")
    print("=" * 50)
    
    if env_ok and llm_ok and embeddings_ok:
        print("🎉 ALL TESTS PASSED! Your environment is ready.")
        print("\nYou can now use:")
        print("- python ingest.py (to process documents)")
        print("- python query.py (to search documents)")
        print("- python gradio_browser.py (to launch web interface)")
        return True
    else:
        print("❌ SOME TESTS FAILED!")
        if not env_ok:
            print("- Environment variables need to be configured")
        if not llm_ok:
            print("- OpenRouter LLM API connection failed")
        if not embeddings_ok:
            print("- OpenAI embeddings API connection failed")
        print("\nPlease check your .env file and API keys.")
        return False

if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)