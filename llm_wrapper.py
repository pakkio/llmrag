import os
import requests
import json
import logging
import re
import numpy as np
from typing import List, Dict, Optional, Tuple, Union
from pathlib import Path
from openai import OpenAI

try:
    from dotenv import load_dotenv
    
    script_dir = Path(__file__).parent
    env_paths = [
        Path.cwd() / ".env",
        script_dir / ".env"
    ]
    
    for env_path in env_paths:
        if env_path.exists():
            load_dotenv(env_path)
            if os.environ.get('PAK_DEBUG') == 'true':
                print(f"llm_wrapper: Loaded .env from {env_path}")
            break
    else:
        if os.environ.get('PAK_DEBUG') == 'true':
            print(f"llm_wrapper: No .env file found in {[str(p) for p in env_paths]}")
            
except ImportError:
    pass

def llm_call(messages: List[Dict[str, str]],
            model: Optional[str] = None,
            max_tokens: int = 4000,
            temperature: float = 0.1,
            response_format: Optional[Dict] = None) -> Tuple[str, bool]:
    """
    Simplified LLM wrapper for semantic compression.
    Returns (response_text, success_flag)
    """
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        logging.error("OPENROUTER_API_KEY not found in environment")
        return "[ERROR: API key missing]", False

    if model is None:
        model = os.environ.get("SEMANTIC_MODEL", "anthropic/claude-3-haiku:beta")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": os.environ.get("OPENROUTER_APP_URL", "http://localhost"),
        "X-Title": os.environ.get("OPENROUTER_APP_TITLE", "Pak4SemanticCompressor"),
    }

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False
    }
    
    # Add response_format if provided (for structured outputs)
    if response_format:
        payload["response_format"] = response_format

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=120
        )

        if response.status_code != 200:
            logging.error(f"API call failed: {response.status_code} - {response.text}")
            return f"[ERROR: API {response.status_code}]", False

        data = response.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")

        if not content:
            logging.error(f"Empty response from API: {data}")
            return "[ERROR: Empty response]", False

        return content.strip(), True

    except requests.exceptions.RequestException as e:
        logging.error(f"Request failed: {e}")
        return f"[ERROR: {str(e)}]", False
    except json.JSONDecodeError as e:
        logging.error(f"JSON decode error: {e}")
        return "[ERROR: Invalid JSON response]", False
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        return f"[ERROR: {str(e)}]", False

def test_llm_connection() -> bool:
    """Test LLM connection with a simple prompt"""
    messages = [{"role": "user", "content": "Reply with exactly: 'CONNECTION_OK'"}]
    response, success = llm_call(messages)
    return success and "CONNECTION_OK" in response

def generate_embeddings(texts: Union[str, List[str]], 
                        normalize: bool = True) -> Union[np.ndarray, List[np.ndarray]]:
    """
    Generate embeddings using OpenAI's text-embedding-3-large model.
    
    Args:
        texts: Single text string or list of text strings
        normalize: Whether to normalize embeddings (default True)
        
    Returns:
        Single numpy array for single text, or list of arrays for multiple texts
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise Exception("OPENAI_API_KEY not found in environment variables")
    
    client = OpenAI(api_key=api_key)
    
    is_single = isinstance(texts, str)
    if is_single:
        texts = [texts]
    
    embeddings = []
    
    for text in texts:
        try:
            # Clean text to ensure UTF-8 compatibility
            cleaned_text = text.encode('utf-8', errors='replace').decode('utf-8')
            
            # OpenAI text-embedding-3-large can handle up to 8191 tokens
            # Roughly 4 chars per token, so ~32k characters max
            if len(cleaned_text) > 32000:
                # Find last space before 32000 to avoid cutting words in half
                truncation_point = cleaned_text.rfind(' ', 0, 32000)
                if truncation_point > 31000:  # Ensure we don't truncate too much (less than 1000 chars lost)
                    cleaned_text = cleaned_text[:truncation_point]
                else:
                    # Fallback to hard truncation if no suitable space found
                    cleaned_text = cleaned_text[:32000]
                logging.warning(f"Truncated text from {len(text)} to {len(cleaned_text)} characters")
            
            # Basic text cleaning
            cleaned_text = ''.join(char for char in cleaned_text if ord(char) >= 32 or char in '\t\n\r')
            cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
            
            # Ensure text is not empty
            if not cleaned_text.strip():
                raise Exception("Text is empty after cleaning")
            
            # Generate embedding using OpenAI API
            embedding_model = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-large")
            response = client.embeddings.create(
                model=embedding_model,
                input=cleaned_text,
                encoding_format="float"
            )
            
            embedding_values = response.data[0].embedding
            embedding = np.array(embedding_values, dtype=np.float32)
            
            # Normalize if requested
            if normalize:
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm
            
            embeddings.append(embedding)
            
        except Exception as e:
            logging.error(f"Error generating embedding: {e}")
            raise
    
    if is_single:
        return embeddings[0]
    return embeddings

def test_openai_embeddings() -> bool:
    """Test if OpenAI embedding API is working"""
    try:
        test_text = "Test embedding generation"
        embedding = generate_embeddings(test_text)
        return len(embedding) > 0
    except Exception as e:
        logging.error(f"Embedding test failed: {e}")
        return False

def check_openai_api() -> bool:
    """Check if OpenAI API key is available and working"""
    api_key = os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        logging.error("OPENAI_API_KEY not found in environment variables")
        return False
    
    # Test if embedding generation works
    return test_openai_embeddings()

def llm_call_structured_query_enhancement(original_query: str, enhancement_instruction: str = None) -> Tuple[Dict, bool]:
    """
    Enhanced query processing with structured JSON output using OpenRouter's JSON Schema support.
    Returns (enhancement_dict, success_flag)
    """
    if enhancement_instruction is None:
        enhancement_instruction = """Focus on BALANCED enhancement:
- Translate to English if needed
- Add relevant synonyms and related terms
- Include conceptually related vocabulary
- Maintain focus while expanding search potential"""

    # JSON Schema for structured output
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "query_enhancement",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "original_query": {"type": "string"},
                    "detected_language": {"type": "string"},
                    "translation": {"type": "string"},
                    "enhanced_query": {"type": "string"},
                    "synonyms": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "related_terms": {
                        "type": "array", 
                        "items": {"type": "string"}
                    },
                    "search_strategy": {"type": "string"}
                },
                "required": ["original_query", "detected_language", "translation", "enhanced_query", "synonyms", "related_terms", "search_strategy"],
                "additionalProperties": False
            }
        }
    }

    messages = [
        {
            "role": "user",
            "content": f"""You are a search query enhancement expert. Your task is to improve this search query for better document retrieval.

Original Query: "{original_query}"

Enhancement Level Instructions:
{enhancement_instruction}

CRITICAL: You MUST respond with valid JSON in the exact schema format. Do not use numbered lists, markdown, or any other format.

Your JSON response must contain these exact fields:
- original_query: the original query exactly as provided
- detected_language: name of detected language (e.g. "italian", "english", "spanish")
- translation: English translation if needed, or original query if already English
- enhanced_query: improved searchable version with appropriate expansion
- synonyms: array of alternative terms (adjust quantity based on enhancement level)
- related_terms: array of conceptually related terms (adjust scope based on enhancement level)
- search_strategy: brief explanation of enhancement approach

Examples by enhancement level:
MINIMAL: "Quanto è grande il Sole?" → "How big Sun? size"
FULL: "Cos'è una supernova?" → "What supernova? stellar explosion massive star core collapse" 
MAXIMUM: "Differenza tra pianeta e stella" → "Difference planet star? celestial bodies stellar objects planetary formation stellar evolution mass composition"

Focus on terms that would likely appear in academic, technical, or reference documents.

Respond ONLY with valid JSON, no markdown, no explanations."""
        }
    ]

    try:
        logging.debug(f"Calling LLM for structured query enhancement. Original query: '{original_query}'")
        response, success = llm_call(messages, max_tokens=500, response_format=response_format)
        
        if success:
            try:
                enhancement_data = json.loads(response)
                logging.info(f"Structured query enhancement successful for '{original_query}'. Strategy: {enhancement_data.get('search_strategy', 'N/A')}")
                return enhancement_data, True
            except json.JSONDecodeError as json_e:
                logging.error(f"JSONDecodeError in structured output for '{original_query}': {json_e}")
                logging.error(f"Problematic response: ```{response}```")
                return {}, False
        else:
            logging.warning(f"LLM call failed for structured query enhancement: '{original_query}'")
            return {}, False

    except Exception as e:
        logging.error(f"Unexpected error during structured query enhancement for '{original_query}': {e}", exc_info=True)
        return {}, False

if __name__ == "__main__":
    print("Testing LLM connection...")
    if test_llm_connection():
        print("✓ LLM connection successful")
    else:
        print("✗ LLM connection failed")
    
    print("\nTesting OpenAI embedding API...")
    if test_openai_embeddings():
        print("✓ OpenAI embedding API connection successful")
    else:
        print("✗ OpenAI embedding API connection failed")