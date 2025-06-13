import os
import requests
import json
import logging
import numpy as np
from typing import List, Dict, Optional, Tuple, Union
from pathlib import Path

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
            temperature: float = 0.1) -> Tuple[str, bool]:
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
                        server_url: str = "http://127.0.0.1:8080",
                        normalize: bool = True) -> Union[np.ndarray, List[np.ndarray]]:
    """
    Generate embeddings using the Qwen3 model via llama.cpp embedding binary.
    
    Args:
        texts: Single text string or list of text strings
        server_url: Not used, kept for compatibility
        normalize: Whether to normalize embeddings (default True)
        
    Returns:
        Single numpy array for single text, or list of arrays for multiple texts
    """
    import subprocess
    import tempfile
    from pathlib import Path
    
    is_single = isinstance(texts, str)
    if is_single:
        texts = [texts]
    
    embeddings = []
    
    # Path to the llama-embedding binary
    embedding_binary = Path(__file__).parent / "llama.cpp/build/bin/llama-embedding"
    model_path = Path(__file__).parent / "llama.cpp/models/qwen3-embedding-0.6b-q8_0.gguf"
    
    if not embedding_binary.exists():
        raise Exception(f"llama-embedding binary not found at {embedding_binary}")
    if not model_path.exists():
        raise Exception(f"Model file not found at {model_path}")
    
    for text in texts:
        try:
            # Clean text to ensure UTF-8 compatibility
            cleaned_text = text.encode('utf-8', errors='replace').decode('utf-8')
            
            # Truncate text if too long (roughly 3000 tokens max, ~12k chars)
            if len(cleaned_text) > 12000:
                cleaned_text = cleaned_text[:12000]
                logging.warning(f"Truncated text from {len(text)} to {len(cleaned_text)} characters")
            
            # More aggressive text cleaning for embedding generation
            # Remove control characters and replace problematic Unicode characters
            import re
            cleaned_text = ''.join(char for char in cleaned_text if ord(char) >= 32 or char in '\t\n\r')
            # Replace various Unicode spaces with regular space
            cleaned_text = re.sub(r'[\u00A0\u1680\u2000-\u200B\u202F\u205F\u3000\uFEFF]', ' ', cleaned_text)
            # Replace multiple spaces with single space
            cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
            
            # Ensure text is not empty
            if not cleaned_text.strip():
                raise Exception("Text is empty after cleaning")
            
            # Run the embedding binary with limited context and batch size
            cmd = [
                str(embedding_binary),
                "-m", str(model_path),
                "--pooling", "mean",
                "-c", "4096",
                "-b", "1",
                "-p", cleaned_text
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=120
            )
            
            if result.returncode != 0:
                logging.error(f"Embedding generation failed: {result.stderr}")
                raise Exception(f"Embedding binary failed with return code {result.returncode}")
            
            # Parse the output to extract the embedding vector
            output = result.stdout.strip()
            
            # Find the embedding line (starts with "embedding 0:")
            embedding_start = output.find('embedding 0:')
            if embedding_start == -1:
                raise Exception("Could not find 'embedding 0:' in binary output")
            
            # Extract everything after "embedding 0:" until we hit a line that starts with non-numeric content
            embedding_section = output[embedding_start + len('embedding 0:'):].strip()
            
            # Split by lines and collect all numeric values
            embedding_values = []
            lines = embedding_section.split('\n')
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                # Stop if we hit a line that starts with non-numeric content (like "build:" or "main:")
                if line and not line[0].isdigit() and line[0] not in '-.':
                    break
                    
                try:
                    # Parse all numbers in this line
                    values = [float(x) for x in line.split()]
                    embedding_values.extend(values)
                except ValueError:
                    # If we can't parse the line as numbers, we've reached the end
                    break
            
            if not embedding_values:
                # Debug: log what we actually got
                logging.error(f"Could not find embedding values. Output: {result.stdout[:500]}...")
                raise Exception("Could not find embedding output in binary result")
            
            # Debug: check if we got the expected number of values
            if len(embedding_values) != 1024:
                logging.warning(f"Expected 1024 embedding values, got {len(embedding_values)}")
            
            embedding = np.array(embedding_values, dtype=np.float32)
            
            # Normalize if requested
            if normalize:
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm
            
            embeddings.append(embedding)
            
        except subprocess.TimeoutExpired:
            logging.error("Embedding generation timed out")
            raise Exception("Embedding generation timed out")
        except Exception as e:
            logging.error(f"Unexpected error generating embedding: {e}")
            raise
    
    if is_single:
        return embeddings[0]
    return embeddings

def test_embedding_server(server_url: str = "http://127.0.0.1:8080") -> bool:
    """Test if the embedding binary is working"""
    try:
        test_text = "Test embedding generation"
        embedding = generate_embeddings(test_text, server_url)
        return len(embedding) > 0
    except Exception as e:
        logging.error(f"Embedding test failed: {e}")
        return False

def auto_start_server() -> bool:
    """Check if embedding binary is available (no server needed)"""
    from pathlib import Path
    
    embedding_binary = Path(__file__).parent / "llama.cpp/build/bin/llama-embedding"
    model_path = Path(__file__).parent / "llama.cpp/models/qwen3-embedding-0.6b-q8_0.gguf"
    
    if not embedding_binary.exists():
        logging.error(f"llama-embedding binary not found at {embedding_binary}")
        return False
    
    if not model_path.exists():
        logging.error(f"Model file not found at {model_path}")
        return False
    
    # Test if embedding generation works
    return test_embedding_server()

if __name__ == "__main__":
    print("Testing LLM connection...")
    if test_llm_connection():
        print("✓ LLM connection successful")
    else:
        print("✗ LLM connection failed")
    
    print("\nTesting embedding server...")
    if test_embedding_server():
        print("✓ Embedding server connection successful")
    else:
        print("✗ Embedding server connection failed")