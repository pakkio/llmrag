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
from sentence_transformers import SentenceTransformer

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
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            
            # Skip empty pages
            if text.strip():
                pages.append({
                    'page_number': page_num + 1,
                    'text': text.strip(),
                    'source_file': pdf_path
                })
                logging.info(f"Extracted page {page_num + 1}: {len(text)} characters")
            else:
                logging.warning(f"Page {page_num + 1} is empty, skipping")
        
        doc.close()
        
    except Exception as e:
        logging.error(f"Error processing PDF: {e}")
        raise
    
    return pages

def load_embedding_model():
    """
    Load the Qwen3-Embedding-0.6B-GGUF model using sentence-transformers.
    """
    try:
        # Use the Hugging Face model identifier
        model_name = "Qwen/Qwen3-Embedding-0.6B-GGUF"
        logging.info(f"Loading embedding model: {model_name}")
        
        model = SentenceTransformer(model_name)
        logging.info("Embedding model loaded successfully")
        return model
        
    except Exception as e:
        logging.error(f"Error loading embedding model: {e}")
        # Fallback to a smaller model if the specified one fails
        logging.info("Falling back to all-MiniLM-L6-v2 model")
        return SentenceTransformer('all-MiniLM-L6-v2')

def generate_embeddings(pages: List[Dict[str, Any]], model: SentenceTransformer) -> List[Dict[str, Any]]:
    """
    Generate embeddings for each page using the specified model.
    """
    logging.info(f"Generating embeddings for {len(pages)} pages")
    
    # Extract text content for batch processing
    texts = [page['text'] for page in pages]
    
    try:
        # Generate embeddings in batch for efficiency
        embeddings = model.encode(texts, convert_to_numpy=True)
        
        # Add embeddings to page data
        for i, page in enumerate(pages):
            page['embedding'] = embeddings[i].tolist()  # Convert numpy array to list for JSON serialization
            page['embedding_dim'] = len(embeddings[i])
            
        logging.info(f"Generated embeddings with dimension: {embeddings.shape[1]}")
        
    except Exception as e:
        logging.error(f"Error generating embeddings: {e}")
        raise
    
    return pages

def save_results(pages: List[Dict[str, Any]], pdf_path: str):
    """
    Save each page as individual text file and embedding file.
    Format: x_page_#.txt and x_page_#_embedding.npz
    """
    try:
        pdf_name = Path(pdf_path).stem  # Get filename without extension
        
        for page in pages:
            page_num = page['page_number']
            
            # Save text file
            text_filename = f"{pdf_name}_page_{page_num}.txt"
            with open(text_filename, 'w', encoding='utf-8') as f:
                f.write(page['text'])
            logging.info(f"Saved text: {text_filename}")
            
            # Save embedding as .npz file
            embedding_filename = f"{pdf_name}_page_{page_num}_embedding.npz"
            embedding_array = np.array(page['embedding'])
            np.savez_compressed(embedding_filename, embedding=embedding_array)
            logging.info(f"Saved embedding: {embedding_filename}")
        
        logging.info(f"Saved {len(pages)} pages as individual files")
        
    except Exception as e:
        logging.error(f"Error saving results: {e}")
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
        
        # Load embedding model
        logging.info("Loading embedding model...")
        model = load_embedding_model()
        
        # Generate embeddings
        logging.info("Generating embeddings...")
        pages_with_embeddings = generate_embeddings(pages, model)
        
        # Save results
        logging.info("Saving results...")
        save_results(pages_with_embeddings, str(pdf_path))
        
        logging.info(f"Successfully processed {len(pages)} pages from {pdf_path}")
        
    except Exception as e:
        logging.error(f"Processing failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()