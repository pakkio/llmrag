#!/usr/bin/env python3
import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any
import chromadb
from datetime import datetime

def setup_logging():
    """Configure logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def get_chroma_client():
    """Initialize and return ChromaDB client"""
    return chromadb.PersistentClient(path="./chroma_db")

def show_page_content(collection, page_number: int):
    """Show content of a specific page"""
    try:
        # Query for specific page
        results = collection.get(
            include=['documents', 'metadatas'],
            where={"page_number": page_number}
        )
        
        if not results['ids']:
            print(f"Page {page_number} not found in collection.")
            return
        
        print(f"\nPage {page_number} Content:")
        print("=" * 60)
        
        # Handle multiple chunks for the same page
        for i, (doc_id, document, metadata) in enumerate(zip(results['ids'], results['documents'], results['metadatas'])):
            if len(results['ids']) > 1:
                chunk_info = f" (Chunk {metadata.get('chunk_id', i+1)}/{metadata.get('total_chunks', len(results['ids']))})"
                print(f"\n--- Page {page_number}{chunk_info} ---")
            
            print(f"Text Length: {len(document)} characters")
            if metadata.get('is_chunked'):
                print(f"Chunked: Yes ({metadata.get('chunk_id')}/{metadata.get('total_chunks')})")
            print()
            
            # Show content with line wrapping
            import textwrap
            wrapped_text = textwrap.fill(document, width=80)
            print(wrapped_text)
            
            if i < len(results['ids']) - 1:
                print("\n" + "-" * 40)
        
        print("\n" + "=" * 60)
        
    except Exception as e:
        logging.error(f"Error showing page content: {e}")

def verify_collection(client, collection_name: str):
    """Verify collection integrity and report issues"""
    try:
        # Handle both full collection name, display name, and numeric index
        collection = None
        
        # Check if it's a numeric index
        if collection_name.isdigit():
            collections = client.list_collections()
            pdf_collections = [col for col in collections if col.name.startswith('pdf_')]
            
            index = int(collection_name) - 1  # Convert to 0-based index
            if 0 <= index < len(pdf_collections):
                collection = pdf_collections[index]
                collection_name = collection.name
            else:
                print(f"Collection index {collection_name} not found.")
                print(f"Available collections: 1-{len(pdf_collections)}")
                return
        else:
            # Handle collection name
            if not collection_name.startswith('pdf_'):
                clean_name = "".join(c if c.isalnum() or c in "_-" else "_" for c in collection_name.lower())
                collection_name = f"pdf_{clean_name}"
            
            try:
                collection = client.get_collection(name=collection_name)
            except Exception:
                # Try to find collection by partial match
                collections = client.list_collections()
                pdf_collections = [col for col in collections if col.name.startswith('pdf_')]
                matches = [c for c in pdf_collections if collection_name.lower() in c.name.lower()]
                
                if not matches:
                    print(f"Collection '{collection_name}' not found.")
                    return
                elif len(matches) > 1:
                    print(f"Multiple collections match '{collection_name}':")
                    for match in matches:
                        print(f"  - {match.name}")
                    return
                else:
                    collection = matches[0]
                    collection_name = collection.name
        
        print(f"\nVerifying collection: {collection_name}")
        print("=" * 60)
        
        issues = []
        warnings = []
        
        # Basic collection info
        count = collection.count()
        metadata = collection.metadata or {}
        
        print(f"Total documents: {count}")
        
        if count == 0:
            issues.append("Collection is empty")
        else:
            # Get all documents for analysis
            results = collection.get(include=['documents', 'embeddings', 'metadatas'])
            
            if not results['ids']:
                issues.append("Failed to retrieve documents")
            else:
                documents = results['documents']
                embeddings = results['embeddings']
                metadatas = results['metadatas']
                
                # Check for missing data
                if not all(documents):
                    missing_docs = sum(1 for doc in documents if not doc or not doc.strip())
                    if missing_docs > 0:
                        issues.append(f"{missing_docs} documents have empty content")
                
                if not all(embeddings):
                    missing_emb = sum(1 for emb in embeddings if not emb)
                    if missing_emb > 0:
                        issues.append(f"{missing_emb} documents missing embeddings")
                
                if not all(metadatas):
                    missing_meta = sum(1 for meta in metadatas if not meta)
                    if missing_meta > 0:
                        issues.append(f"{missing_meta} documents missing metadata")
                
                # Check embedding dimensions consistency and validity
                valid_embeddings = [emb for emb in embeddings if emb]
                if valid_embeddings:
                    embedding_dims = [len(emb) for emb in valid_embeddings]
                    unique_dims = set(embedding_dims)
                    if len(unique_dims) > 1:
                        issues.append(f"Inconsistent embedding dimensions: {unique_dims}")
                    else:
                        print(f"Embedding dimension: {list(unique_dims)[0]}")
                    
                    # Check for zero vectors or invalid values
                    zero_vectors = 0
                    invalid_vectors = 0
                    
                    for i, emb in enumerate(embeddings):
                        if emb:
                            # Check if all values are zero
                            if all(val == 0.0 for val in emb):
                                zero_vectors += 1
                            
                            # Check for NaN or infinite values
                            import math
                            if any(math.isnan(val) or math.isinf(val) for val in emb):
                                invalid_vectors += 1
                    
                    if zero_vectors > 0:
                        issues.append(f"{zero_vectors} documents have zero-valued embeddings")
                    
                    if invalid_vectors > 0:
                        issues.append(f"{invalid_vectors} documents have invalid embeddings (NaN/Inf)")
                    
                    # Calculate embedding quality metrics
                    if len(valid_embeddings) > 0:
                        # Calculate average magnitude
                        import numpy as np
                        magnitudes = [np.linalg.norm(emb) for emb in valid_embeddings[:100]]  # Sample first 100
                        avg_magnitude = sum(magnitudes) / len(magnitudes)
                        min_magnitude = min(magnitudes)
                        max_magnitude = max(magnitudes)
                        
                        print(f"Embedding quality - Avg magnitude: {avg_magnitude:.4f}, Range: {min_magnitude:.4f}-{max_magnitude:.4f}")
                        
                        # Warn about suspicious magnitudes
                        very_small = sum(1 for mag in magnitudes if mag < 0.1)
                        very_large = sum(1 for mag in magnitudes if mag > 100.0)
                        
                        if very_small > len(magnitudes) * 0.1:  # More than 10%
                            warnings.append(f"{very_small} embeddings have unusually small magnitude (<0.1)")
                        
                        if very_large > 0:
                            warnings.append(f"{very_large} embeddings have unusually large magnitude (>100)")
                else:
                    issues.append("No valid embeddings found in collection")
                
                # Analyze page sequence
                pages = [m.get('page_number', 0) for m in metadatas if m and m.get('page_number')]
                if pages:
                    unique_pages = sorted(set(pages))
                    min_page = min(pages)
                    max_page = max(pages)
                    expected_pages = set(range(min_page, max_page + 1))
                    missing_pages = expected_pages - set(unique_pages)
                    
                    print(f"Page range: {min_page}-{max_page} ({len(unique_pages)} unique pages)")
                    
                    if missing_pages:
                        missing_list = sorted(list(missing_pages))
                        if len(missing_list) <= 10:
                            warnings.append(f"Missing pages: {missing_list}")
                        else:
                            warnings.append(f"Missing {len(missing_list)} pages: {missing_list[:5]}...{missing_list[-5:]}")
                    
                    # Check for duplicate pages (non-chunked)
                    non_chunked_pages = [m.get('page_number') for m in metadatas 
                                       if m and m.get('page_number') and not m.get('is_chunked', False)]
                    if non_chunked_pages:
                        from collections import Counter
                        page_counts = Counter(non_chunked_pages)
                        duplicates = {page: count for page, count in page_counts.items() if count > 1}
                        if duplicates:
                            warnings.append(f"Duplicate non-chunked pages: {dict(list(duplicates.items())[:5])}")
                
                # Check text length distribution
                text_lengths = [len(doc) for doc in documents if doc]
                if text_lengths:
                    avg_length = sum(text_lengths) / len(text_lengths)
                    very_short = sum(1 for length in text_lengths if length < 50)
                    very_long = sum(1 for length in text_lengths if length > 10000)
                    
                    print(f"Average text length: {avg_length:.0f} characters")
                    
                    if very_short > 0:
                        warnings.append(f"{very_short} documents are very short (<50 chars)")
                    if very_long > 0:
                        warnings.append(f"{very_long} documents are very long (>10k chars)")
        
        # Report results
        print("\nVerification Results:")
        if not issues and not warnings:
            print("✅ Collection appears to be in good condition")
        else:
            if issues:
                print(f"\n❌ Issues found ({len(issues)}):")
                for i, issue in enumerate(issues, 1):
                    print(f"  {i}. {issue}")
            
            if warnings:
                print(f"\n⚠️  Warnings ({len(warnings)}):")
                for i, warning in enumerate(warnings, 1):
                    print(f"  {i}. {warning}")
        
        print("=" * 60)
        
    except Exception as e:
        logging.error(f"Error verifying collection: {e}")
        sys.exit(1)

def list_collections(client):
    """List all PDF collections in the database"""
    try:
        collections = client.list_collections()
        pdf_collections = [col for col in collections if col.name.startswith('pdf_')]
        
        if not pdf_collections:
            print("No PDF collections found in database.")
            return
        
        print(f"\nFound {len(pdf_collections)} PDF collection(s):")
        print("-" * 100)
        print(f"{'#':<3} {'Collection Name':<25} {'Documents':<10} {'Pages':<15} {'Source File':<40}")
        print("-" * 100)
        
        for idx, collection in enumerate(pdf_collections, 1):
            collection_name = collection.name
            count = collection.count()
            
            # Try to get source file from metadata
            metadata = collection.metadata
            source_file = metadata.get('source', 'Unknown') if metadata else 'Unknown'
            
            # Get page range information
            page_info = "Unknown"
            if count > 0:
                try:
                    # Get sample to determine page range
                    results = collection.get(limit=count, include=['metadatas'])
                    if results and results['metadatas']:
                        pages = [m.get('page_number', 0) for m in results['metadatas'] if m.get('page_number')]
                        if pages:
                            min_page = min(pages)
                            max_page = max(pages)
                            unique_pages = len(set(pages))
                            if min_page == max_page:
                                page_info = f"Page {min_page}"
                            else:
                                page_info = f"{min_page}-{max_page} ({unique_pages} unique)"
                except:
                    page_info = "Error reading"
            
            # Clean up the display name (remove pdf_ prefix if present)
            display_name = collection_name
            if collection_name.startswith('pdf_'):
                display_name = collection_name[4:].replace('_', ' ')
            
            print(f"{idx:<3} {display_name:<25} {count:<10} {page_info:<15} {Path(source_file).name:<40}")
        
        print("-" * 100)
        
    except Exception as e:
        logging.error(f"Error listing collections: {e}")
        sys.exit(1)

def get_collection_info(client, collection_name: str, page_number: int = None):
    """Get detailed information about a specific collection"""
    try:
        # Handle both full collection name, display name, and numeric index
        collection = None
        
        # Check if it's a numeric index
        if collection_name.isdigit():
            collections = client.list_collections()
            pdf_collections = [col for col in collections if col.name.startswith('pdf_')]
            
            index = int(collection_name) - 1  # Convert to 0-based index
            if 0 <= index < len(pdf_collections):
                collection = pdf_collections[index]
            else:
                print(f"Collection index {collection_name} not found.")
                print(f"Available collections: 1-{len(pdf_collections)}")
                return
        else:
            # Handle collection name
            if not collection_name.startswith('pdf_'):
                # Try to find collection by cleaning the name
                clean_name = "".join(c if c.isalnum() or c in "_-" else "_" for c in collection_name.lower())
                collection_name = f"pdf_{clean_name}"
            
            try:
                collection = client.get_collection(name=collection_name)
            except Exception:
                # Try to find collection by partial match
                collections = client.list_collections()
                matches = [c for c in collections if collection_name.lower() in c.name.lower()]
                
                if not matches:
                    print(f"Collection '{collection_name}' not found.")
                    available = [c.name for c in collections]
                    if available:
                        print(f"Available collections: {', '.join(available)}")
                    return
                elif len(matches) > 1:
                    print(f"Multiple collections match '{collection_name}':")
                    for match in matches:
                        print(f"  - {match.name}")
                    return
                else:
                    collection = matches[0]
        
        # If specific page requested, show page content
        if page_number is not None:
            show_page_content(collection, page_number)
            return
        
        # Get collection metadata
        metadata = collection.metadata or {}
        count = collection.count()
        
        print(f"\nCollection Information: {collection.name}")
        print("=" * 60)
        print(f"Documents: {count}")
        print(f"Source File: {metadata.get('source', 'Unknown')}")
        
        if count > 0:
            # Get all documents to analyze
            results = collection.get(include=['metadatas'])
            
            if results and results['metadatas']:
                metadatas = results['metadatas']
                
                # Analyze page distribution
                pages = [m.get('page_number', 0) for m in metadatas if m.get('page_number')]
                if pages:
                    unique_pages = sorted(set(pages))
                    print(f"Page Range: {min(pages)} - {max(pages)} ({len(unique_pages)} unique pages)")
                    
                    # Check for missing pages
                    expected_pages = set(range(min(pages), max(pages) + 1))
                    missing_pages = expected_pages - set(unique_pages)
                    if missing_pages:
                        missing_list = sorted(list(missing_pages))
                        print(f"Missing Pages: {missing_list}")
                
                # Count chunked vs regular pages
                chunked_count = sum(1 for m in metadatas if m.get('is_chunked', False))
                regular_count = len(metadatas) - chunked_count
                
                if chunked_count > 0:
                    print(f"Chunked Documents: {chunked_count}")
                    print(f"Regular Documents: {regular_count}")
                
                # Average text length
                text_lengths = [m.get('text_length', 0) for m in metadatas if m.get('text_length')]
                if text_lengths:
                    avg_length = sum(text_lengths) / len(text_lengths)
                    min_length = min(text_lengths)
                    max_length = max(text_lengths)
                    print(f"Text Length - Avg: {avg_length:.0f}, Min: {min_length}, Max: {max_length} chars")
        
        print("=" * 60)
        
    except Exception as e:
        logging.error(f"Error getting collection info: {e}")
        sys.exit(1)

def remove_collection(client, collection_name: str):
    """Remove a PDF collection from the database"""
    try:
        # Handle both full collection name and display name
        if not collection_name.startswith('pdf_'):
            # Try to find collection by cleaning the name
            clean_name = "".join(c if c.isalnum() or c in "_-" else "_" for c in collection_name.lower())
            collection_name = f"pdf_{clean_name}"
        
        try:
            collection = client.get_collection(name=collection_name)
        except Exception:
            # Try to find collection by partial match
            collections = client.list_collections()
            matches = [c for c in collections if collection_name.lower() in c.name.lower()]
            
            if not matches:
                print(f"Collection '{collection_name}' not found.")
                available = [c.name for c in collections]
                if available:
                    print(f"Available collections: {', '.join(available)}")
                return
            elif len(matches) > 1:
                print(f"Multiple collections match '{collection_name}':")
                for match in matches:
                    print(f"  - {match.name}")
                print("Please be more specific.")
                return
            else:
                collection = matches[0]
                collection_name = collection.name
        
        # Confirm deletion
        count = collection.count()
        metadata = collection.metadata or {}
        source_file = metadata.get('source', 'Unknown')
        
        print(f"\nAbout to delete collection: {collection_name}")
        print(f"Source: {source_file}")
        print(f"Documents: {count}")
        
        confirm = input("\nAre you sure you want to delete this collection? (yes/no): ")
        if confirm.lower() not in ['yes', 'y']:
            print("Deletion cancelled.")
            return
        
        client.delete_collection(name=collection_name)
        print(f"Successfully deleted collection: {collection_name}")
        
    except Exception as e:
        logging.error(f"Error removing collection: {e}")
        sys.exit(1)

def rename_collection(client, old_name: str, new_name: str):
    """Rename a PDF collection"""
    try:
        # Handle both full collection name and display name for old_name
        if not old_name.startswith('pdf_'):
            clean_old_name = "".join(c if c.isalnum() or c in "_-" else "_" for c in old_name.lower())
            old_name = f"pdf_{clean_old_name}"
        
        # Clean new name
        clean_new_name = "".join(c if c.isalnum() or c in "_-" else "_" for c in new_name)
        if not clean_new_name.startswith('pdf_'):
            new_collection_name = f"pdf_{clean_new_name}"
        else:
            new_collection_name = clean_new_name
        
        try:
            old_collection = client.get_collection(name=old_name)
        except Exception:
            # Try to find collection by partial match
            collections = client.list_collections()
            matches = [c for c in collections if old_name.lower() in c.name.lower()]
            
            if not matches:
                print(f"Collection '{old_name}' not found.")
                available = [c.name for c in collections]
                if available:
                    print(f"Available collections: {', '.join(available)}")
                return
            elif len(matches) > 1:
                print(f"Multiple collections match '{old_name}':")
                for match in matches:
                    print(f"  - {match.name}")
                print("Please be more specific.")
                return
            else:
                old_collection = matches[0]
                old_name = old_collection.name
        
        # Check if new name already exists
        try:
            client.get_collection(name=new_collection_name)
            print(f"Collection '{new_collection_name}' already exists.")
            return
        except:
            pass  # Good, new name doesn't exist
        
        print(f"Renaming collection from '{old_name}' to '{new_collection_name}'...")
        
        # Get all data from old collection
        results = old_collection.get(include=['documents', 'embeddings', 'metadatas'])
        
        if not results['ids']:
            print("Collection is empty, just renaming...")
            
        # Create new collection with updated metadata
        old_metadata = old_collection.metadata or {}
        new_metadata = old_metadata.copy()
        
        new_collection = client.create_collection(
            name=new_collection_name,
            metadata=new_metadata
        )
        
        # Copy data if collection has documents
        if results['ids']:
            print(f"Copying {len(results['ids'])} documents...")
            new_collection.add(
                ids=results['ids'],
                documents=results['documents'],
                embeddings=results['embeddings'],
                metadatas=results['metadatas']
            )
        
        # Delete old collection
        client.delete_collection(name=old_name)
        
        print(f"Successfully renamed collection to: {new_collection_name}")
        
    except Exception as e:
        logging.error(f"Error renaming collection: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='Manage PDF collections in ChromaDB')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose logging')
    
    # Create mutually exclusive group for commands
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--list', action='store_true', help='List all PDF collections')
    group.add_argument('--info', type=str, metavar='COLLECTION', help='Show detailed info for a collection (name or index #)')
    group.add_argument('--remove', type=str, metavar='COLLECTION', help='Remove a collection')
    group.add_argument('--rename', nargs=2, metavar=('OLD_NAME', 'NEW_NAME'), help='Rename a collection')
    group.add_argument('--verify', type=str, metavar='COLLECTION', help='Verify collection integrity')
    
    # Optional arguments for --info
    parser.add_argument('--page', type=int, metavar='PAGE_NUM', help='Show content of specific page (use with --info)')
    
    # If no arguments provided, show help
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    setup_logging()
    
    # Check if database exists
    db_path = Path("./chroma_db")
    if not db_path.exists():
        logging.error("ChromaDB database not found at './chroma_db'")
        logging.error("Run ingest.py first to create the database")
        sys.exit(1)
    
    try:
        client = get_chroma_client()
        
        # Validate argument combinations
        if args.page and not args.info:
            parser.error("--page can only be used with --info")
        
        if args.list:
            list_collections(client)
        elif args.info:
            get_collection_info(client, args.info, args.page)
        elif args.remove:
            remove_collection(client, args.remove)
        elif args.rename:
            rename_collection(client, args.rename[0], args.rename[1])
        elif args.verify:
            verify_collection(client, args.verify)
            
    except Exception as e:
        logging.error(f"Database operation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()