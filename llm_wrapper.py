import os
import requests
import json
import logging
from typing import List, Dict, Optional, Tuple
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

if __name__ == "__main__":
    print("Testing LLM connection...")
    if test_llm_connection():
        print("✓ LLM connection successful")
    else:
        print("✗ LLM connection failed")