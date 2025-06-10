#!/usr/bin/env python3
import chromadb
import logging
from pathlib import Path

def setup_logging():
    """Configure logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def get_pdf_records_info():
    """Get information about records stored for each PDF in ChromaDB"""
    try:
        # Initialize Chroma client
        client = chromadb.PersistentClient(path="./chroma_db")
        
        # Get all collections
        collections = client.list_collections()
        pdf_collections = [col for col in collections if col.name.startswith('pdf_')]
        
        if not pdf_collections:
            print("No PDF collections found in ChromaDB.")
            print("Run 'python ingest.py <pdf_file>' to process some PDFs first.")
            return
        
        print(f"📊 ChromaDB Information")
        print("=" * 60)
        print(f"Database location: ./chroma_db")
        print(f"Total PDF collections: {len(pdf_collections)}")
        print()
        
        total_records = 0
        
        for collection in pdf_collections:
            pdf_name = collection.name.replace('pdf_', '')
            
            # Get collection details
            col_obj = client.get_collection(name=collection.name)
            record_count = col_obj.count()
            
            # Get metadata if available
            metadata = col_obj.metadata or {}
            source_file = metadata.get('source', 'Unknown')
            total_pages = metadata.get('total_pages', 'Unknown')
            
            print(f"📄 PDF: {pdf_name}")
            print(f"   Records: {record_count}")
            print(f"   Source: {source_file}")
            print(f"   Total pages: {total_pages}")
            print()
            
            total_records += record_count
        
        print("=" * 60)
        print(f"📊 Summary:")
        print(f"   Total PDFs: {len(pdf_collections)}")
        print(f"   Total records: {total_records}")
        print(f"   Average records per PDF: {total_records / len(pdf_collections):.1f}")
        
    except Exception as e:
        logging.error(f"Error accessing ChromaDB: {e}")
        print(f"Error: {e}")
        print("Make sure ChromaDB is properly initialized.")

def main():
    setup_logging()
    get_pdf_records_info()

if __name__ == "__main__":
    main()