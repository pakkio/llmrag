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
from llm_wrapper import generate_embeddings, test_openai_embeddings, check_openai_api

def setup_logging():
    """Configure logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def chunk_text(text: str, max_chars: int = 500, overlap_chars: int = 50) -> List[str]:
    """
    Split text into fixed-size character chunks with overlap to preserve context.
    Optimized for embedding generation with 500-character chunks.
    """
    if not text.strip():
        return []
    
    text = text.strip()
    
    # If text is shorter than max_chars, return as single chunk
    if len(text) <= max_chars:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        # Calculate end position for this chunk
        end = start + max_chars
        
        # If this would be the last chunk and it's very short, merge with previous
        if end >= len(text):
            chunk = text[start:]
            # If chunk is very short (less than 100 chars) and we have previous chunks,
            # merge with the last chunk
            if len(chunk) < 100 and chunks:
                last_chunk = chunks.pop()
                # Only merge if combined length doesn't exceed max_chars * 1.5
                if len(last_chunk) + len(chunk) <= max_chars * 1.5:
                    chunks.append(last_chunk + " " + chunk)
                else:
                    chunks.append(last_chunk)
                    chunks.append(chunk)
            else:
                chunks.append(chunk)
            break
        
        # Try to break at word boundary near the end position
        chunk_text = text[start:end]
        
        # Look for the last space within the chunk to avoid breaking words
        last_space = chunk_text.rfind(' ')
        if last_space > max_chars * 0.7:  # Only break at word if it's not too far back
            end = start + last_space
            chunk_text = text[start:end]
        
        chunks.append(chunk_text.strip())
        
        # Move start position with overlap
        start = end - overlap_chars
        
        # Ensure we don't go backwards
        if start <= chunks[-1].__len__() // 2:  # Prevent infinite loops
            start = end
    
    # Filter out empty chunks and ensure minimum length
    chunks = [chunk.strip() for chunk in chunks if chunk.strip() and len(chunk.strip()) >= 20]
    
    return chunks

def get_summary_pages(pdf_path: str) -> List[int]:
    """
    Identify pages that contain summary content (table of contents, scope sections, overviews).
    Returns a list of page numbers (1-based) that should be processed for summary content.
    
    Based on analysis of "Great Mythologies of the World" structure:
    - Pages 1, 10-16: Course title and table of contents
    - Pages 106, 191, 192, 269: Regional scope sections with thematic overviews
    - Pages 485+: Bibliography and reference sections
    """
    # Default summary pages based on document analysis
    summary_pages = [1]  # Title page
    
    # Table of contents pages
    summary_pages.extend(range(10, 17))  # Pages 10-16
    
    # Regional scope pages (thematic overviews)
    summary_pages.extend([106, 191, 192, 269])
    
    # Bibliography section (starting around page 485)
    summary_pages.extend(range(485, 495))  # Estimated range for bibliography
    
    return sorted(summary_pages)

def is_summary_page(page_text: str, page_number: int) -> bool:
    """
    Determine if a page contains summary/overview content based on keywords and structure.
    """
    text_lower = page_text.lower()
    
    # Keywords that indicate summary content
    summary_keywords = [
        'table of contents', 'contents', 'introduction', 'scope', 
        'overview', 'summary', 'lecture guides', 'professor biographies',
        'bibliography', 'supplemental material', 'course outline',
        'syllabus', 'index'
    ]
    
    # Check for summary keywords
    for keyword in summary_keywords:
        if keyword in text_lower:
            return True
    
    # Check for structural indicators
    if 'lecture' in text_lower and ('guide' in text_lower or 'outline' in text_lower):
        return True
    
    # Check for section headers that indicate overviews
    if any(phrase in text_lower for phrase in [
        'this section', 'this course', 'mythology of', 'mythologies of',
        'provides an introduction', 'survey of', 'explores the'
    ]):
        return True
    
    return False

def extract_pdf_pages(pdf_path: str, from_page: int = 1, to_page: int = 0, summary_only: bool = False) -> List[Dict[str, Any]]:
    """
    Extract text content from each page of a PDF file.
    Large pages are automatically chunked to prevent token limit issues.
    Returns a list of dictionaries with page number and text content.
    
    Args:
        pdf_path: Path to the PDF file
        from_page: Start processing from this page (1-based, default: 1)
        to_page: Stop processing at this page (1-based, 0 means last page)
        summary_only: If True, extract only summary/overview pages (table of contents, scope sections, etc.)
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    pages = []
    
    try:
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        
        # Validate and adjust page range
        from_page = max(1, from_page)  # Ensure from_page is at least 1
        if to_page <= 0:
            to_page = total_pages
        else:
            to_page = min(to_page, total_pages)  # Don't exceed total pages
        
        if from_page > total_pages:
            logging.warning(f"from_page ({from_page}) exceeds total pages ({total_pages})")
            doc.close()
            return []
        
        if from_page > to_page:
            logging.error(f"from_page ({from_page}) cannot be greater than to_page ({to_page})")
            doc.close()
            return []
        
        # Handle summary-only mode
        if summary_only:
            summary_pages = get_summary_pages(pdf_path)
            # Filter summary pages to be within the specified range
            summary_pages = [p for p in summary_pages if from_page <= p <= to_page]
            if not summary_pages:
                logging.warning(f"No summary pages found in range {from_page}-{to_page}")
                doc.close()
                return []
            
            logging.info(f"Processing PDF in summary mode: {pdf_path}")
            logging.info(f"Summary pages to process: {summary_pages}")
            pages_to_process = len(summary_pages)
            page_range = summary_pages
        else:
            pages_to_process = to_page - from_page + 1
            logging.info(f"Processing PDF: {pdf_path} (pages {from_page}-{to_page} of {total_pages} total)")
            page_range = range(from_page, to_page + 1)
        
        for page_num in tqdm(page_range, desc="Extracting pages", unit="page"):
            # Convert to 0-based index for PyMuPDF (page_num is 1-based)
            page_idx = page_num - 1
            page = doc[page_idx]
            text = page.get_text()
            
            # Skip empty pages
            if text.strip():
                text = text.strip()
                # Ensure UTF-8 compatibility by encoding/decoding with error handling
                text = text.encode('utf-8', errors='replace').decode('utf-8')
                
                # In summary mode, verify this is actually a summary page
                if summary_only and not is_summary_page(text, page_num):
                    logging.debug(f"Page {page_num} doesn't appear to contain summary content, skipping")
                    continue
                
                # Chunk page text into 500-character chunks for better embeddings
                chunks = chunk_text(text, max_chars=500, overlap_chars=50)
                
                if len(chunks) == 1:
                    # Page fits in one chunk (≤500 chars)
                    page_data = {
                        'page_number': page_num,
                        'text': text,
                        'source_file': pdf_path,
                        'chunk_id': None,
                        'total_chunks': 1,
                        'chunk_size': len(text)
                    }
                    if summary_only:
                        page_data['is_summary'] = True
                        page_data['content_type'] = 'summary'
                    pages.append(page_data)
                    logging.debug(f"Extracted page {page_num}: {len(text)} characters (single chunk)")
                else:
                    # Page chunked into multiple 500-char pieces
                    logging.info(f"Page {page_num} ({len(text)} chars) split into {len(chunks)} chunks of ~500 chars each")
                    for chunk_idx, chunk in enumerate(chunks):
                        chunk_data = {
                            'page_number': page_num,
                            'text': chunk,
                            'source_file': pdf_path,
                            'chunk_id': chunk_idx + 1,
                            'total_chunks': len(chunks),
                            'chunk_size': len(chunk)
                        }
                        if summary_only:
                            chunk_data['is_summary'] = True
                            chunk_data['content_type'] = 'summary'
                        pages.append(chunk_data)
                    logging.debug(f"Page {page_num} chunk {chunk_idx + 1}: {len(chunk)} characters")
            else:
                logging.debug(f"Page {page_num} is empty, skipping")
        
        doc.close()
        
    except Exception as e:
        logging.error(f"Error processing PDF: {e}")
        raise
    
    return pages

def check_embedding_system():
    """
    Check if OpenAI API key is available and embedding API is working.
    """
    logging.info("Checking OpenAI embedding API...")
    if not test_openai_embeddings():
        logging.error("OpenAI embedding API not available")
        logging.error("Please ensure OPENAI_API_KEY is set in your environment")
        raise Exception("OpenAI embedding API not available")
    else:
        logging.info("OpenAI embedding API is working successfully")

def generate_page_embeddings(pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Generate embeddings for each page using OpenAI's text-embedding-3-large.
    Includes error recovery to continue processing after failures.
    """
    logging.info(f"Generating embeddings for {len(pages)} pages using OpenAI text-embedding-3-large")
    
    # Extract text content
    texts = [page['text'] for page in pages]
    
    # Generate embeddings with progress bar and error recovery
    with tqdm(total=len(texts), desc="Generating embeddings", unit="page") as pbar:
        embeddings = [None] * len(pages)  # Initialize with None values
        successful_count = 0
        failed_pages = []
        
        for i, text in enumerate(texts):
            try:
                # Generate embedding for single text
                embedding = generate_embeddings([text])
                embeddings[i] = embedding
                successful_count += 1
                
                # Log if page was chunked
                if pages[i].get('chunk_id'):
                    logging.debug(f"Generated embedding for page {pages[i]['page_number']}, chunk {pages[i]['chunk_id']}")
                else:
                    logging.debug(f"Generated embedding for page {pages[i]['page_number']}")
                    
            except Exception as e:
                logging.warning(f"Failed to generate embedding for page {pages[i]['page_number']}: {e}")
                failed_pages.append(i)
                embeddings[i] = None
            
            pbar.update(1)
    
    if failed_pages:
        logging.warning(f"Failed to generate embeddings for {len(failed_pages)} pages/chunks")
        logging.info(f"Successfully processed {successful_count}/{len(pages)} pages/chunks")
    
    # Filter out failed pages and add embeddings to successful pages
    successful_pages = []
    embedding_dim = None
    
    for i, page in enumerate(pages):
        if embeddings[i] is not None:
            embedding = embeddings[i]
            
            # Convert numpy array to list for JSON serialization
            if hasattr(embedding, 'tolist'):
                page['embedding'] = embedding.tolist()
                page['embedding_dim'] = len(embedding)
                if embedding_dim is None:
                    embedding_dim = len(embedding)
            elif isinstance(embedding, list) and len(embedding) == 1 and hasattr(embedding[0], 'tolist'):
                # Handle case where we have [numpy_array] instead of numpy_array
                page['embedding'] = embedding[0].tolist()
                page['embedding_dim'] = len(embedding[0])
                if embedding_dim is None:
                    embedding_dim = len(embedding[0])
            else:
                # If it's already a list
                page['embedding'] = embedding
                page['embedding_dim'] = len(embedding)
                if embedding_dim is None:
                    embedding_dim = len(embedding)
            
            successful_pages.append(page)
        else:
            # Log the failed page details
            chunk_info = f", chunk {page['chunk_id']}/{page['total_chunks']}" if page.get('chunk_id') else ""
            logging.warning(f"Skipping page {page['page_number']}{chunk_info} due to embedding failure")
    
    if not successful_pages:
        raise Exception("No embeddings were generated successfully")
    
    logging.info(f"Generated embeddings with dimension: {embedding_dim}")
    logging.info(f"Successfully processed {len(successful_pages)} pages/chunks")
    
    return successful_pages

def save_to_chroma(pages: List[Dict[str, Any]], pdf_path: str, summary_only: bool = False):
    """
    Save pages and embeddings to Chroma vector database.
    If summary_only is True, adds summary pages to existing collection without deleting it.
    """
    try:
        pdf_name = Path(pdf_path).stem  # Get filename without extension
        
        # Initialize Chroma client (embedded mode)
        client = chromadb.PersistentClient(path="./chroma_db")
        
        # Get or create collection for this PDF
        # Clean the name to meet ChromaDB requirements: alphanumeric, underscores, hyphens
        clean_name = "".join(c if c.isalnum() or c in "_-" else "_" for c in pdf_name)
        collection_name = f"pdf_{clean_name}"
        
        # Handle collection creation/retrieval based on mode
        if summary_only:
            # In summary mode, try to get existing collection or create new one
            try:
                collection = client.get_collection(name=collection_name)
                logging.info(f"Using existing collection: {collection_name}")
                
                # Check for existing summary pages and remove them to avoid duplicates
                existing_summary_ids = []
                try:
                    # Get all documents with summary metadata
                    results = collection.get(where={"is_summary": True})
                    if results and results['ids']:
                        existing_summary_ids = results['ids']
                        collection.delete(ids=existing_summary_ids)
                        logging.info(f"Removed {len(existing_summary_ids)} existing summary pages to avoid duplicates")
                except Exception as e:
                    logging.debug(f"No existing summary pages found or error checking: {e}")
                
            except ValueError:
                # Collection doesn't exist, create it
                collection = client.create_collection(
                    name=collection_name,
                    metadata={"source": pdf_path, "total_pages": len(pages)}
                )
                logging.info(f"Created new collection: {collection_name}")
        else:
            # Normal mode: delete existing collection if it exists (for re-ingestion)
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
            # Create unique ID for each page/chunk
            if page.get('is_summary'):
                # Use unique prefix for summary pages to avoid conflicts
                if page.get('chunk_id'):
                    page_id = f"summary_page_{page['page_number']}_chunk_{page['chunk_id']}"
                else:
                    page_id = f"summary_page_{page['page_number']}"
            else:
                # Regular page IDs
                if page.get('chunk_id'):
                    page_id = f"page_{page['page_number']}_chunk_{page['chunk_id']}"
                else:
                    page_id = f"page_{page['page_number']}"
            
            documents.append(page['text'])
            
            embeddings.append(page['embedding'])
            metadata = {
                'page_number': page['page_number'],
                'source_file': pdf_path,
                'text_length': len(page['text']),
                'chunk_size': page.get('chunk_size', len(page['text']))
            }
            
            # Add chunk information if available
            if page.get('chunk_id'):
                metadata['chunk_id'] = page['chunk_id']
                metadata['total_chunks'] = page['total_chunks']
                metadata['is_chunked'] = True
                metadata['chunk_type'] = '500char_chunk'
            else:
                metadata['is_chunked'] = False
                metadata['chunk_type'] = 'single_page'
            
            # Add summary information if available
            if page.get('is_summary'):
                metadata['is_summary'] = True
                metadata['content_type'] = page.get('content_type', 'summary')
            else:
                metadata['is_summary'] = False
                metadata['content_type'] = 'regular'
                
            metadatas.append(metadata)
            ids.append(page_id)
        
        # Add all pages to collection in batch with progress
        mode_desc = "summary pages" if summary_only else "pages"
        action_desc = "Adding" if summary_only else "Saving"
        logging.info(f"{action_desc} {len(pages)} {mode_desc} to ChromaDB...")
        
        with tqdm(total=1, desc=f"{action_desc} to ChromaDB", unit="batch") as pbar:
            collection.add(
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            pbar.update(1)
        
        if summary_only:
            logging.info(f"Added {len(pages)} summary pages to existing Chroma collection: {collection_name}")
        else:
            logging.info(f"Saved {len(pages)} pages to Chroma collection: {collection_name}")
        logging.info(f"Chroma database location: ./chroma_db")
        
    except Exception as e:
        logging.error(f"Error saving to Chroma: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description='Extract pages and generate embeddings from PDF files')
    parser.add_argument('pdf_file', help='Path to the PDF file to process')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose logging')
    parser.add_argument('-p', '--pages', type=int, default=0, help='Number of pages to process from the selected range (0 means all pages in range)')
    parser.add_argument('--from-page', type=int, default=1, help='Start processing from this page number (1-based, default: 1)')
    parser.add_argument('--to-page', type=int, default=0, help='Stop processing at this page number (1-based, 0 means last page)')
    parser.add_argument('--summary', action='store_true', help='Extract only summary/overview pages (table of contents, scope sections, etc.)')
    
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
        # Validate page range arguments
        if args.from_page < 1:
            logging.error("--from-page must be 1 or greater")
            sys.exit(1)
        
        if args.to_page > 0 and args.to_page < args.from_page:
            logging.error("--to-page cannot be less than --from-page")
            sys.exit(1)
        
        # Warn about potentially confusing option combinations
        if args.pages > 0 and (args.from_page > 1 or args.to_page > 0):
            logging.warning("Using --pages with --from-page/--to-page: --pages will limit results AFTER page range filtering")
        
        # Extract pages from PDF
        logging.info("Starting PDF page extraction...")
        pages = extract_pdf_pages(str(pdf_path), args.from_page, args.to_page, args.summary)
        
        if not pages:
            logging.warning("No text content found in PDF")
            sys.exit(1)
        
        # Limit pages if specified (applies after page range filtering)
        if args.pages > 0:
            original_count = len(pages)
            pages = pages[:args.pages]
            if len(pages) < original_count:
                logging.info(f"Limited to first {len(pages)} pages from the selected range")
        
        # Check embedding system
        logging.info("Checking embedding system...")
        check_embedding_system()
        
        # Generate embeddings
        logging.info("Generating embeddings...")
        pages_with_embeddings = generate_page_embeddings(pages)
        
        # Save results to Chroma
        logging.info("Saving to Chroma database...")
        save_to_chroma(pages_with_embeddings, str(pdf_path), args.summary)
        
        logging.info(f"Successfully processed {len(pages)} pages from {pdf_path}")
        
    except Exception as e:
        logging.error(f"Processing failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()