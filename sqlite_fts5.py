#!/usr/bin/env python3
"""
SQLite FTS5 module for keyword-based search complementing ChromaDB semantic search.
Provides BM25-like ranking with full-text search capabilities.
"""

import os
import sqlite3
import logging
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import re


class SQLiteFTS5Manager:
    """
    Manages SQLite FTS5 full-text search database for BM25-like keyword search
    """
    
    def __init__(self, db_path: str = "./hybrid_search.db"):
        """
        Initialize SQLite FTS5 manager
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.conn = None
        self._initialize_db()
    
    def _initialize_db(self):
        """Initialize SQLite database with FTS5 table"""
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row  # Enable column access by name
            
            # Create FTS5 virtual table with porter stemming
            self.conn.execute('''
                CREATE VIRTUAL TABLE IF NOT EXISTS documents USING fts5(
                    content,                 -- Document text content
                    pdf_name,               -- Source PDF filename  
                    page_number,            -- Page number in PDF
                    chunk_id,               -- Chunk ID if text was chunked
                    total_chunks,           -- Total chunks for this page
                    is_summary,             -- Whether this is summary content
                    content_type,           -- Type: regular, summary
                    source_file,            -- Full path to source file
                    tokenize='porter'       -- Porter stemming for better matching
                )
            ''')
            
            # Create metadata table for additional info
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS document_metadata (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pdf_name TEXT NOT NULL,
                    source_file TEXT NOT NULL,
                    total_pages INTEGER,
                    ingestion_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    content_type TEXT DEFAULT 'regular'
                )
            ''')
            
            self.conn.commit()
            logging.debug(f"Initialized SQLite FTS5 database: {self.db_path}")
            
        except sqlite3.Error as e:
            logging.error(f"Error initializing SQLite FTS5 database: {e}")
            raise
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
    
    def clear_pdf_documents(self, pdf_name: str, summary_only: bool = False):
        """
        Remove existing documents for a PDF to avoid duplicates
        
        Args:
            pdf_name: Name of the PDF file (without extension)
            summary_only: If True, only remove summary documents
        """
        try:
            if summary_only:
                # Remove only summary documents
                cursor = self.conn.execute(
                    "DELETE FROM documents WHERE pdf_name = ? AND is_summary = 'true'",
                    (pdf_name,)
                )
            else:
                # Remove all documents for this PDF
                cursor = self.conn.execute(
                    "DELETE FROM documents WHERE pdf_name = ?", 
                    (pdf_name,)
                )
                # Also remove from metadata table
                self.conn.execute(
                    "DELETE FROM document_metadata WHERE pdf_name = ?",
                    (pdf_name,)
                )
            
            deleted_count = cursor.rowcount
            self.conn.commit()
            
            if deleted_count > 0:
                doc_type = "summary documents" if summary_only else "documents"
                logging.info(f"Removed {deleted_count} existing {doc_type} for {pdf_name}")
                
        except sqlite3.Error as e:
            logging.error(f"Error clearing documents for {pdf_name}: {e}")
            raise
    
    def add_documents(self, pages: List[Dict[str, Any]], pdf_path: str, summary_only: bool = False):
        """
        Add documents to FTS5 index
        
        Args:
            pages: List of page dictionaries with text and metadata
            pdf_path: Path to source PDF file
            summary_only: Whether adding summary-only content
        """
        try:
            pdf_name = Path(pdf_path).stem
            
            # Clear existing documents to avoid duplicates
            self.clear_pdf_documents(pdf_name, summary_only)
            
            # Add metadata entry if not summary-only mode
            if not summary_only:
                self.conn.execute('''
                    INSERT OR REPLACE INTO document_metadata 
                    (pdf_name, source_file, total_pages, content_type)
                    VALUES (?, ?, ?, ?)
                ''', (pdf_name, pdf_path, len(pages), 'regular'))
            
            # Prepare batch insert data
            documents_data = []
            
            for page in pages:
                # Clean and prepare text content
                content = self._clean_text_for_fts(page['text'])
                
                # Extract metadata
                page_number = page.get('page_number', 0)
                chunk_id = page.get('chunk_id', '')
                total_chunks = page.get('total_chunks', 1)
                is_summary = 'true' if page.get('is_summary', False) else 'false'
                content_type = page.get('content_type', 'regular')
                
                documents_data.append((
                    content,
                    pdf_name,
                    page_number,
                    chunk_id,
                    total_chunks,
                    is_summary,
                    content_type,
                    pdf_path
                ))
            
            # Batch insert all documents
            self.conn.executemany('''
                INSERT INTO documents 
                (content, pdf_name, page_number, chunk_id, total_chunks, 
                 is_summary, content_type, source_file)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', documents_data)
            
            self.conn.commit()
            
            doc_type = "summary pages" if summary_only else "pages"
            logging.info(f"Added {len(pages)} {doc_type} to SQLite FTS5 index for {pdf_name}")
            
        except sqlite3.Error as e:
            logging.error(f"Error adding documents to FTS5: {e}")
            raise
    
    def _clean_text_for_fts(self, text: str) -> str:
        """
        Clean text content for better FTS5 indexing
        
        Args:
            text: Raw text content
            
        Returns:
            Cleaned text suitable for FTS5 indexing
        """
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove problematic characters that might interfere with FTS5
        text = re.sub(r'[^\w\s\-.,;:!?()\[\]{}"\'/]', ' ', text)
        
        # Normalize quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        
        return text
    
    def search(self, query: str, pdf_name: Optional[str] = None, 
               limit: int = 10, include_summary: bool = True) -> List[Tuple[str, int, str, float, Dict[str, Any]]]:
        """
        Search documents using FTS5 BM25-style ranking
        
        Args:
            query: Search query
            pdf_name: Optional PDF name to limit search to specific document
            limit: Maximum number of results to return
            include_summary: Whether to include summary pages in results
            
        Returns:
            List of tuples: (pdf_name, page_number, content, score, metadata)
        """
        try:
            # Clean query for FTS5
            query = self._clean_query_for_fts(query)
            if not query.strip():
                return []
            
            # Build WHERE clause
            where_conditions = []
            params = [query]
            
            if pdf_name:
                where_conditions.append("pdf_name = ?")
                params.append(pdf_name)
            
            if not include_summary:
                where_conditions.append("is_summary = 'false'")
            
            where_clause = ""
            if where_conditions:
                where_clause = " AND " + " AND ".join(where_conditions)
            
            # Execute FTS5 search with BM25 ranking
            sql_query = f'''
                SELECT 
                    pdf_name,
                    page_number,
                    chunk_id,
                    total_chunks,
                    content,
                    is_summary,
                    content_type,
                    source_file,
                    bm25(documents) as score
                FROM documents 
                WHERE documents MATCH ?{where_clause}
                ORDER BY score
                LIMIT ?
            '''
            
            params.append(limit)
            cursor = self.conn.execute(sql_query, params)
            results = cursor.fetchall()
            
            # Format results with rank-based scoring
            formatted_results = []
            for rank, row in enumerate(results):
                metadata = {
                    'chunk_id': row['chunk_id'] if row['chunk_id'] else None,
                    'total_chunks': row['total_chunks'],
                    'is_summary': row['is_summary'] == 'true',
                    'content_type': row['content_type'],
                    'source_file': row['source_file']
                }
                
                # Use rank-based scoring: 1st result gets highest score
                # This is more reliable than trying to interpret tiny BM25 values
                rank_score = 1.0 - (rank / len(results)) if len(results) > 0 else 0.0
                
                formatted_results.append((
                    row['pdf_name'],
                    row['page_number'],
                    row['content'],
                    rank_score,
                    metadata
                ))
            
            logging.debug(f"FTS5 search for '{query}' returned {len(results)} results")
            return formatted_results
            
        except sqlite3.Error as e:
            logging.error(f"Error searching FTS5 database: {e}")
            return []
    
    def _clean_query_for_fts(self, query: str) -> str:
        """
        Clean and prepare query for FTS5 search
        
        Args:
            query: Raw search query
            
        Returns:
            Cleaned query suitable for FTS5
        """
        if not query:
            return ""
        
        # Remove special FTS5 characters that might cause issues
        query = re.sub(r'[^\w\s\-]', ' ', query)
        
        # Split into terms and clean
        terms = query.split()
        cleaned_terms = []
        
        for term in terms:
            term = term.strip()
            if len(term) >= 2:  # Minimum term length
                cleaned_terms.append(term)
        
        if not cleaned_terms:
            return ""
        
        # Join terms with OR for better recall
        return " OR ".join(cleaned_terms)
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about indexed documents
        
        Returns:
            Dictionary with collection statistics
        """
        try:
            # Get document counts
            cursor = self.conn.execute("SELECT COUNT(*) as total_docs FROM documents")
            total_docs = cursor.fetchone()['total_docs']
            
            # Get PDF counts
            cursor = self.conn.execute("SELECT COUNT(DISTINCT pdf_name) as total_pdfs FROM documents")
            total_pdfs = cursor.fetchone()['total_pdfs']
            
            # Get summary document count
            cursor = self.conn.execute("SELECT COUNT(*) as summary_docs FROM documents WHERE is_summary = 'true'")
            summary_docs = cursor.fetchone()['summary_docs']
            
            # Get PDFs list
            cursor = self.conn.execute("SELECT DISTINCT pdf_name FROM documents ORDER BY pdf_name")
            pdf_names = [row['pdf_name'] for row in cursor.fetchall()]
            
            return {
                'total_documents': total_docs,
                'total_pdfs': total_pdfs,
                'summary_documents': summary_docs,
                'regular_documents': total_docs - summary_docs,
                'pdf_names': pdf_names,
                'database_path': self.db_path
            }
            
        except sqlite3.Error as e:
            logging.error(f"Error getting collection stats: {e}")
            return {}
    
    def list_collections(self) -> List[str]:
        """
        List available PDF collections (for compatibility with ChromaDB interface)
        
        Returns:
            List of PDF collection names
        """
        try:
            cursor = self.conn.execute("SELECT DISTINCT pdf_name FROM documents ORDER BY pdf_name")
            return [f"pdf_{row['pdf_name']}" for row in cursor.fetchall()]
        except sqlite3.Error as e:
            logging.error(f"Error listing collections: {e}")
            return []