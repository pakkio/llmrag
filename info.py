#!/usr/bin/env python3
"""
ChromaDB Document Information Tool
Usage:
  ./info.py --list                    # List all documents
  ./info.py 1                         # Show document 1 info
  ./info.py 1 --page 34               # Show page 34 of document 1
  ./info.py 1 --page 34 --chunk 2     # Show chunk 2 of page 34 in document 1
"""

import argparse
import chromadb
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

def setup_logging(verbose: bool = False):
    """Configure logging"""
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

class DocumentInfo:
    def __init__(self, db_path: str = "./chroma_db"):
        self.client = chromadb.PersistentClient(path=db_path)
        self.db_path = db_path
    
    def list_documents(self) -> List[Dict[str, Any]]:
        """List all PDF documents in the database"""
        try:
            collections = self.client.list_collections()
            pdf_collections = [col for col in collections if col.name.startswith('pdf_')]
            
            if not pdf_collections:
                return []
            
            documents = []
            for i, collection in enumerate(pdf_collections):
                col_obj = self.client.get_collection(name=collection.name)
                record_count = col_obj.count()
                
                # Get metadata
                metadata = col_obj.metadata or {}
                source_file = metadata.get('source', 'Unknown')
                total_pages = metadata.get('total_pages', 'Unknown')
                
                # Analyze chunk distribution
                if record_count > 0:
                    sample_results = col_obj.get(
                        limit=min(100, record_count), 
                        include=['metadatas']
                    )
                    
                    chunk_analysis = self._analyze_chunks(sample_results['metadatas'])
                else:
                    chunk_analysis = {
                        'total_records': 0,
                        'chunked_records': 0,
                        'single_page_records': 0,
                        'avg_chunk_size': 0,
                        'unique_pages': 0
                    }
                
                documents.append({
                    'index': i + 1,
                    'name': collection.name.replace('pdf_', ''),
                    'collection_name': collection.name,
                    'source_file': source_file,
                    'total_pages': total_pages,
                    'total_records': record_count,
                    'chunk_analysis': chunk_analysis
                })
            
            return documents
            
        except Exception as e:
            logging.error(f"Error listing documents: {e}")
            return []
    
    def _analyze_chunks(self, metadatas: List[Dict]) -> Dict[str, Any]:
        """Analyze chunk distribution from metadata"""
        chunked_records = 0
        single_page_records = 0
        chunk_sizes = []
        unique_pages = set()
        
        for meta in metadatas:
            unique_pages.add(meta.get('page_number', 0))
            
            if meta.get('is_chunked', False):
                chunked_records += 1
            else:
                single_page_records += 1
            
            chunk_size = meta.get('chunk_size', meta.get('text_length', 0))
            if chunk_size > 0:
                chunk_sizes.append(chunk_size)
        
        avg_chunk_size = sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0
        
        return {
            'total_records': len(metadatas),
            'chunked_records': chunked_records,
            'single_page_records': single_page_records,
            'avg_chunk_size': int(avg_chunk_size),
            'unique_pages': len(unique_pages)
        }
    
    def get_document_info(self, doc_index: int) -> Optional[Dict[str, Any]]:
        """Get detailed information for a specific document"""
        documents = self.list_documents()
        if not documents or doc_index < 1 or doc_index > len(documents):
            return None
        
        doc = documents[doc_index - 1]
        
        # Get full analysis
        collection = self.client.get_collection(name=doc['collection_name'])
        all_results = collection.get(include=['metadatas'])
        
        # Build page structure
        pages = {}
        for i, meta in enumerate(all_results['metadatas']):
            page_num = meta.get('page_number', 0)
            
            if page_num not in pages:
                pages[page_num] = {
                    'page_number': page_num,
                    'chunks': [],
                    'is_chunked': False,
                    'total_chunks': 1
                }
            
            chunk_info = {
                'record_index': i,
                'chunk_id': meta.get('chunk_id'),
                'chunk_size': meta.get('chunk_size', meta.get('text_length', 0)),
                'is_chunked': meta.get('is_chunked', False)
            }
            
            pages[page_num]['chunks'].append(chunk_info)
            if meta.get('is_chunked', False):
                pages[page_num]['is_chunked'] = True
                pages[page_num]['total_chunks'] = meta.get('total_chunks', 1)
        
        doc['pages'] = dict(sorted(pages.items()))
        return doc
    
    def get_page_content(self, doc_index: int, page_number: int) -> Optional[Dict[str, Any]]:
        """Get content for a specific page with all its chunks"""
        doc_info = self.get_document_info(doc_index)
        if not doc_info or page_number not in doc_info['pages']:
            return None
        
        collection = self.client.get_collection(name=doc_info['collection_name'])
        page_info = doc_info['pages'][page_number]
        
        # Get all records for this page
        page_records = []
        all_results = collection.get(include=['documents', 'metadatas'])
        
        for i, meta in enumerate(all_results['metadatas']):
            if meta.get('page_number') == page_number:
                page_records.append({
                    'chunk_id': meta.get('chunk_id'),
                    'text': all_results['documents'][i],
                    'chunk_size': meta.get('chunk_size', len(all_results['documents'][i])),
                    'is_chunked': meta.get('is_chunked', False),
                    'total_chunks': meta.get('total_chunks', 1)
                })
        
        # Sort chunks by chunk_id if chunked
        if page_info['is_chunked']:
            page_records.sort(key=lambda x: x['chunk_id'] or 0)
        
        return {
            'page_number': page_number,
            'document': doc_info['name'],
            'is_chunked': page_info['is_chunked'],
            'total_chunks': page_info['total_chunks'],
            'chunks': page_records
        }
    
    def get_chunk_content(self, doc_index: int, page_number: int, chunk_id: int) -> Optional[Dict[str, Any]]:
        """Get content for a specific chunk"""
        page_content = self.get_page_content(doc_index, page_number)
        if not page_content:
            return None
        
        # Find the specific chunk
        for chunk in page_content['chunks']:
            if chunk['chunk_id'] == chunk_id:
                return {
                    'document': page_content['document'],
                    'page_number': page_number,
                    'chunk_id': chunk_id,
                    'total_chunks': page_content['total_chunks'],
                    'text': chunk['text'],
                    'chunk_size': chunk['chunk_size']
                }
        
        return None

def print_document_list(documents: List[Dict[str, Any]]):
    """Print formatted list of documents"""
    if not documents:
        print("❌ No documents found in ChromaDB.")
        print("   Run 'python ingest.py <pdf_file>' to add documents.")
        return
    
    print(f"📚 ChromaDB Documents ({len(documents)} total)")
    print("=" * 80)
    print(f"{'#':<3} {'Document':<30} {'Pages':<8} {'Records':<9} {'Avg Size':<9} {'Type':<12}")
    print("-" * 80)
    
    for doc in documents:
        analysis = doc['chunk_analysis']
        chunk_ratio = (analysis['chunked_records'] / max(analysis['total_records'], 1)) * 100
        
        if chunk_ratio > 50:
            doc_type = "Chunked"
        elif chunk_ratio > 0:
            doc_type = "Mixed"
        else:
            doc_type = "Single Pages"
        
        print(f"{doc['index']:<3} {doc['name'][:29]:<30} {doc['total_pages']:<8} "
              f"{doc['total_records']:<9} {analysis['avg_chunk_size']:<9} {doc_type:<12}")
    
    print("-" * 80)
    print(f"Database location: ./chroma_db")
    print(f"\nUsage:")
    print(f"  ./info.py 1                    # Show document 1 details")
    print(f"  ./info.py 1 --page 34          # Show page 34 of document 1")
    print(f"  ./info.py 1 --page 34 --chunk 2  # Show chunk 2 of page 34")

def print_document_info(doc_info: Dict[str, Any]):
    """Print detailed document information"""
    print(f"📄 Document {doc_info['index']}: {doc_info['name']}")
    print("=" * 60)
    print(f"Source: {doc_info['source_file']}")
    print(f"Total Pages: {doc_info['total_pages']}")
    print(f"Total Records: {doc_info['total_records']}")
    
    analysis = doc_info['chunk_analysis']
    print(f"Chunked Records: {analysis['chunked_records']}")
    print(f"Single Page Records: {analysis['single_page_records']}")
    print(f"Average Chunk Size: {analysis['avg_chunk_size']} characters")
    
    print(f"\n📋 Page Structure:")
    print("-" * 60)
    print(f"{'Page':<6} {'Chunks':<8} {'Type':<12} {'Size Range':<15}")
    print("-" * 60)
    
    for page_num, page_info in doc_info['pages'].items():
        chunk_count = len(page_info['chunks'])
        page_type = "Chunked" if page_info['is_chunked'] else "Single"
        
        # Calculate size range for this page
        sizes = [chunk['chunk_size'] for chunk in page_info['chunks']]
        if sizes:
            size_range = f"{min(sizes)}-{max(sizes)}" if len(sizes) > 1 else str(sizes[0])
        else:
            size_range = "0"
        
        print(f"{page_num:<6} {chunk_count:<8} {page_type:<12} {size_range:<15}")
    
    print("-" * 60)
    print(f"Use --page <num> to view specific page content")

def print_page_content(page_content: Dict[str, Any]):
    """Print page content with all chunks"""
    print(f"📖 Page {page_content['page_number']} - {page_content['document']}")
    print("=" * 80)
    
    if page_content['is_chunked']:
        print(f"Page is chunked into {page_content['total_chunks']} parts")
        print("-" * 80)
        
        for i, chunk in enumerate(page_content['chunks']):
            chunk_num = chunk['chunk_id'] or (i + 1)
            print(f"\n🔸 Chunk {chunk_num}/{page_content['total_chunks']} ({chunk['chunk_size']} chars)")
            print("-" * 40)
            print(chunk['text'])
    else:
        chunk = page_content['chunks'][0]
        print(f"Single page content ({chunk['chunk_size']} chars)")
        print("-" * 80)
        print(chunk['text'])
    
    print("\n" + "=" * 80)
    if page_content['is_chunked']:
        print(f"Use --chunk <num> to view specific chunk only")

def print_chunk_content(chunk_content: Dict[str, Any]):
    """Print specific chunk content"""
    print(f"🔸 Chunk {chunk_content['chunk_id']}/{chunk_content['total_chunks']} - "
          f"Page {chunk_content['page_number']} - {chunk_content['document']}")
    print("=" * 80)
    print(f"Content ({chunk_content['chunk_size']} characters):")
    print("-" * 80)
    print(chunk_content['text'])
    print("=" * 80)

def main():
    parser = argparse.ArgumentParser(
        description='ChromaDB Document Information Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  ./info.py --list                    # List all documents
  ./info.py 1                         # Show document 1 info
  ./info.py 1 --page 34               # Show page 34 of document 1
  ./info.py 1 --page 34 --chunk 2     # Show chunk 2 of page 34 in document 1
        """
    )
    
    parser.add_argument('document', type=int, nargs='?', 
                       help='Document index (1-based)')
    parser.add_argument('--list', action='store_true',
                       help='List all documents')
    parser.add_argument('--page', type=int,
                       help='Page number to display')
    parser.add_argument('--chunk', type=int,
                       help='Chunk ID within the page')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    
    try:
        doc_info = DocumentInfo()
        
        # List documents
        if args.list:
            documents = doc_info.list_documents()
            print_document_list(documents)
            return
        
        # Validate document index
        if args.document is None:
            print("❌ Error: Document index required (or use --list)")
            print("Usage: ./info.py <document_index> [--page <page>] [--chunk <chunk>]")
            sys.exit(1)
        
        # Show specific chunk
        if args.page and args.chunk:
            chunk_content = doc_info.get_chunk_content(args.document, args.page, args.chunk)
            if chunk_content:
                print_chunk_content(chunk_content)
            else:
                print(f"❌ Chunk {args.chunk} not found on page {args.page} of document {args.document}")
                sys.exit(1)
        
        # Show specific page
        elif args.page:
            page_content = doc_info.get_page_content(args.document, args.page)
            if page_content:
                print_page_content(page_content)
            else:
                print(f"❌ Page {args.page} not found in document {args.document}")
                sys.exit(1)
        
        # Show document info
        else:
            document = doc_info.get_document_info(args.document)
            if document:
                print_document_info(document)
            else:
                documents = doc_info.list_documents()
                if documents:
                    print(f"❌ Document {args.document} not found. Available: 1-{len(documents)}")
                else:
                    print("❌ No documents found in ChromaDB")
                sys.exit(1)
    
    except Exception as e:
        logging.error(f"Error: {e}")
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()