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
    Generate embeddings using the Qwen3 model via llama.cpp server.
    
    Args:
        texts: Single text string or list of text strings
        server_url: URL of the llama.cpp embedding server
        normalize: Whether embeddings are already normalized by server
        
    Returns:
        Single numpy array for single text, or list of arrays for multiple texts
    """
    is_single = isinstance(texts, str)
    if is_single:
        texts = [texts]
    
    embeddings = []
    
    for text in texts:
        # Ensure text ends with the required token
        if not text.endswith('<|endoftext|>'):
            text = text + '<|endoftext|>'
        
        try:
            response = requests.post(
                f"{server_url}/v1/embeddings",
                headers={"Content-Type": "application/json"},
                json={
                    "input": text,
                    "model": "qwen3"
                },
                timeout=60
            )
            
            if response.status_code != 200:
                logging.error(f"Embedding API call failed: {response.status_code} - {response.text}")
                raise Exception(f"API call failed with status {response.status_code}")
            
            data = response.json()
            embedding = np.array(data["data"][0]["embedding"], dtype=np.float32)
            
            # The server should already normalize embeddings, but we can ensure it
            if not normalize:
                # If we want to manually normalize (though server should do this)
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm
            
            embeddings.append(embedding)
            
        except requests.exceptions.RequestException as e:
            logging.error(f"Request failed for embedding: {e}")
            raise
        except json.JSONDecodeError as e:
            logging.error(f"JSON decode error: {e}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error generating embedding: {e}")
            raise
    
    if is_single:
        return embeddings[0]
    return embeddings

def test_embedding_server(server_url: str = "http://127.0.0.1:8080") -> bool:
    """Test if the embedding server is running and responding"""
    try:
        test_text = "Test embedding generation<|endoftext|>"
        embedding = generate_embeddings(test_text, server_url)
        return len(embedding) > 0
    except Exception as e:
        logging.error(f"Embedding server test failed: {e}")
        return False

def auto_start_server() -> bool:
    """Attempt to automatically start the embedding server"""
    import subprocess
    import time
    from pathlib import Path
    
    script_path = Path(__file__).parent / "start_embedding_server.sh"
    
    if not script_path.exists():
        logging.error("start_embedding_server.sh not found")
        return False
    
    try:
        logging.info("Attempting to start embedding server automatically...")
        
        # Start server in background
        process = subprocess.Popen(
            [str(script_path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True
        )
        
        # Wait for server to start (max 30 seconds)
        for i in range(30):
            time.sleep(1)
            if test_embedding_server():
                logging.info("✅ Embedding server started successfully!")
                return True
            if i % 5 == 0 and i > 0:
                logging.info(f"Waiting for server to start... ({i+1}/30)")
        
        logging.error("Server failed to start within 30 seconds")
        return False
        
    except Exception as e:
        logging.error(f"Error auto-starting server: {e}")
        return False

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