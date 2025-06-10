#!/usr/bin/env python3
import os
import sys
import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
import fitz  # PyMuPDF
import numpy as np
import chromadb
from tqdm import tqdm
from llm_wrapper import generate_embeddings, test_embedding_server, auto_start_server

def setup_logging():
    """Configure logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def extract_pdf_pages(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Extract text content from each page of a PDF file.
    Returns a list of dictionaries with page number and text content.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    pages = []
    
    try:
        doc = fitz.open(pdf_path)
        logging.info(f"Processing PDF: {pdf_path} ({len(doc)} pages)")
        
        for page_num in tqdm(range(len(doc)), desc="Extracting pages", unit="page"):
            page = doc[page_num]
            text = page.get_text()
            
            # Skip empty pages
            if text.strip():
                pages.append({
                    'page_number': page_num + 1,
                    'text': text.strip(),
                    'source_file': pdf_path
                })
                # Only log for verbose mode to avoid cluttering progress bar
                logging.debug(f"Extracted page {page_num + 1}: {len(text)} characters")
            else:
                logging.debug(f"Page {page_num + 1} is empty, skipping")
        
        doc.close()
        
    except Exception as e:
        logging.error(f"Error processing PDF: {e}")
        raise
    
    return pages

def check_embedding_server(server_url: str = "http://127.0.0.1:8080"):
    """
    Check if the Qwen3 embedding server is running, and auto-start if needed.
    """
    logging.info("Checking Qwen3 embedding server...")
    if not test_embedding_server(server_url):
        logging.warning(f"Qwen3 embedding server is not responding at {server_url}")
        logging.info("Attempting to start the server automatically...")
        
        if auto_start_server():
            logging.info("Embedding server started successfully!")
        else:
            logging.error("Failed to auto-start embedding server")
            logging.error("Please start the embedding server manually with:")
            logging.error("./start_embedding_server.sh")
            raise Exception("Embedding server not available")
    else:
        logging.info("Qwen3 embedding server is running successfully")
    
    return server_url

def generate_page_embeddings(pages: List[Dict[str, Any]], server_url: str = "http://127.0.0.1:8080") -> List[Dict[str, Any]]:
    """
    Generate embeddings for each page using the Qwen3 model via llama.cpp server.
    """
    logging.info(f"Generating embeddings for {len(pages)} pages using Qwen3 model")
    
    try:
        # Extract text content
        texts = [page['text'] for page in pages]
        
        # Generate embeddings with progress bar
        with tqdm(total=len(texts), desc="Generating embeddings", unit="page") as pbar:
            # Process in batches to show progress
            batch_size = 1  # Process one page at a time for fine-grained progress
            embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                batch_embeddings = generate_embeddings(batch_texts, server_url)
                
                # Handle single embedding vs list of embeddings
                if len(batch_texts) == 1:
                    embeddings.append(batch_embeddings)
                else:
                    embeddings.extend(batch_embeddings)
                
                pbar.update(len(batch_texts))
        
        # Add embeddings to page data
        for i, page in enumerate(pages):
            page['embedding'] = embeddings[i].tolist()  # Convert numpy array to list for JSON serialization
            page['embedding_dim'] = len(embeddings[i])
            
        logging.info(f"Generated embeddings with dimension: {len(embeddings[0])}")
        
    except Exception as e:
        logging.error(f"Error generating embeddings: {e}")
        raise
    
    return pages

def save_to_chroma(pages: List[Dict[str, Any]], pdf_path: str):
    """
    Save pages and embeddings to Chroma vector database.
    """
    try:
        pdf_name = Path(pdf_path).stem  # Get filename without extension
        
        # Initialize Chroma client (embedded mode)
        client = chromadb.PersistentClient(path="./chroma_db")
        
        # Get or create collection for this PDF
        collection_name = f"pdf_{pdf_name}"
        
        # Delete existing collection if it exists (for re-ingestion)
        try:
            client.delete_collection(name=collection_name)
            logging.info(f"Deleted existing collection: {collection_name}")
        except:
            pass  # Collection doesn't exist, that's fine
        
        collection = client.create_collection(
            name=collection_name,
            metadata={"source": pdf_path, "total_pages": len(pages)}
        )
        
        # Prepare data for batch insertion
        documents = []
        embeddings = []
        metadatas = []
        ids = []
        
        for page in pages:
            page_id = f"page_{page['page_number']}"
            
            documents.append(page['text'])
            embeddings.append(page['embedding'])
            metadatas.append({
                'page_number': page['page_number'],
                'source_file': pdf_path,
                'text_length': len(page['text'])
            })
            ids.append(page_id)
        
        # Add all pages to collection in batch with progress
        logging.info(f"Saving {len(pages)} pages to ChromaDB...")
        with tqdm(total=1, desc="Saving to ChromaDB", unit="batch") as pbar:
            collection.add(
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            pbar.update(1)
        
        logging.info(f"Saved {len(pages)} pages to Chroma collection: {collection_name}")
        logging.info(f"Chroma database location: ./chroma_db")
        
    except Exception as e:
        logging.error(f"Error saving to Chroma: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description='Extract pages and generate embeddings from PDF files')
    parser.add_argument('pdf_file', help='Path to the PDF file to process')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    setup_logging()
    
    # Validate input file
    pdf_path = Path(args.pdf_file)
    if not pdf_path.exists():
        logging.error(f"PDF file not found: {pdf_path}")
        sys.exit(1)
    
    if not pdf_path.suffix.lower() == '.pdf':
        logging.error(f"File is not a PDF: {pdf_path}")
        sys.exit(1)
    
    
    try:
        # Extract pages from PDF
        logging.info("Starting PDF page extraction...")
        pages = extract_pdf_pages(str(pdf_path))
        
        if not pages:
            logging.warning("No text content found in PDF")
            sys.exit(1)
        
        # Check embedding server
        logging.info("Checking embedding server...")
        server_url = check_embedding_server()
        
        # Generate embeddings
        logging.info("Generating embeddings...")
        pages_with_embeddings = generate_page_embeddings(pages, server_url)
        
        # Save results to Chroma
        logging.info("Saving to Chroma database...")
        save_to_chroma(pages_with_embeddings, str(pdf_path))
        
        logging.info(f"Successfully processed {len(pages)} pages from {pdf_path}")
        
    except Exception as e:
        logging.error(f"Processing failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()