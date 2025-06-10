#!/usr/bin/env python3
"""
Utility script to check if the embedding server is running
and optionally start it automatically.
"""

import sys
import subprocess
import time
import requests
import os
from pathlib import Path

def check_server_status(url="http://127.0.0.1:8080"):
    """Check if the embedding server is responding"""
    try:
        response = requests.post(
            f"{url}/v1/embeddings",
            headers={"Content-Type": "application/json"},
            json={"input": "test<|endoftext|>", "model": "qwen3"},
            timeout=5
        )
        return response.status_code == 200
    except:
        return False

def start_server_background():
    """Start the server in background"""
    script_path = Path(__file__).parent / "start_embedding_server.sh"
    
    if not script_path.exists():
        print("❌ start_embedding_server.sh not found")
        return False
    
    try:
        # Start server in background
        process = subprocess.Popen(
            [str(script_path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True
        )
        
        print("🚀 Starting embedding server in background...")
        
        # Wait for server to start (max 30 seconds)
        for i in range(30):
            time.sleep(1)
            if check_server_status():
                print("✅ Embedding server is now running!")
                return True
            if i % 5 == 0:
                print(f"⏳ Waiting for server to start... ({i+1}/30)")
        
        print("❌ Server failed to start within 30 seconds")
        return False
        
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        return False

def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--start":
        auto_start = True
    else:
        auto_start = False
    
    print("🔍 Checking embedding server status...")
    
    if check_server_status():
        print("✅ Embedding server is running and responding")
        sys.exit(0)
    else:
        print("❌ Embedding server is not responding")
        
        if auto_start:
            if start_server_background():
                sys.exit(0)
            else:
                sys.exit(1)
        else:
            print("\n💡 To start the server manually, run:")
            print("   ./start_embedding_server.sh")
            print("\n💡 Or run this script with --start to auto-start:")
            print("   python check_server.py --start")
            sys.exit(1)

if __name__ == "__main__":
    main()