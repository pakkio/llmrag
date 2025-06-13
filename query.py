#!/usr/bin/env python3
import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import chromadb
from llm_wrapper import llm_call, generate_embeddings, test_embedding_server, auto_start_server

def setup_logging():
    """Configure logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def check_embedding_system():
    """Check if the Qwen3 embedding binary is available"""
    logging.info("Checking Qwen3 embedding system...")
    if not test_embedding_server():
        logging.error("Embedding system not available")
        logging.error("Please ensure llama.cpp binary and model are properly set up")
        raise Exception("Embedding system not available")
    else:
        logging.info("Qwen3 embedding system is working successfully")
    
    return True

def list_available_collections() -> List[Tuple[str, int]]:
    """List all available PDF collections with page counts"""
    try:
        client = chromadb.PersistentClient(path="./chroma_db")
        collections = client.list_collections()
        pdf_collections = []
        
        for col in collections:
            if col.name.startswith('pdf_'):
                collection = client.get_collection(name=col.name)
                count = collection.count()
                book_name = col.name.replace('pdf_', '')
                pdf_collections.append((book_name, count))
        
        return pdf_collections
    except Exception as e:
        logging.error(f"Error listing collections: {e}")
        return []

def query_chroma_collections(query_embedding: np.ndarray, top_k: int = 10, pdf_name: str = None) -> List[Tuple[str, int, str, float]]:
    """
    Query Chroma collections for similar pages.
    If pdf_name is provided, search only that collection. Otherwise search all collections.
    Returns list of (pdf_name, page_number, text_content, similarity_score)
    """
    try:
        # Initialize Chroma client
        client = chromadb.PersistentClient(path="./chroma_db")
        
        # Determine which collections to search
        if pdf_name:
            collection_names = [f"pdf_{pdf_name}"]
        else:
            # Search all PDF collections
            collections = client.list_collections()
            collection_names = [col.name for col in collections if col.name.startswith('pdf_')]
        
        if not collection_names:
            logging.error("No PDF collections found. Make sure you've run ingest.py first.")
            return []
        
        all_similarities = []
        
        # Query each collection
        for collection_name in collection_names:
            try:
                collection = client.get_collection(name=collection_name)
                pdf_name_from_collection = collection_name.replace('pdf_', '')
                
                # Query the collection
                results = collection.query(
                    query_embeddings=[query_embedding.tolist()],
                    n_results=top_k,
                    include=['documents', 'metadatas', 'distances']
                )
                
                # Process results
                if results['documents'] and results['documents'][0]:
                    for i, (doc, metadata, distance) in enumerate(zip(
                        results['documents'][0],
                        results['metadatas'][0], 
                        results['distances'][0]
                    )):
                        # Convert distance to similarity (Chroma returns squared euclidean distance)
                        # For normalized embeddings, cosine similarity ≈ 1 - (euclidean_distance²/2)
                        similarity = max(0, 1 - (distance / 2))
                        
                        page_number = metadata['page_number']
                        all_similarities.append((pdf_name_from_collection, page_number, doc, similarity))
                        
            except Exception as e:
                logging.warning(f"Error querying collection {collection_name}: {e}")
                continue
        
        # Sort all results by similarity (highest first)
        all_similarities.sort(key=lambda x: x[3], reverse=True)
        
        logging.info(f"Found {len(all_similarities)} results from {len(collection_names)} collections")
        return all_similarities
        
    except Exception as e:
        logging.error(f"Error querying Chroma collections: {e}")
        return []

def generate_query_embedding(query: str) -> np.ndarray:
    """Generate embedding for the query text using Qwen3 model"""
    try:
        logging.info(f"Generating embedding for query: '{query[:50]}...'")
        embedding = generate_embeddings(query, normalize=True)
        return embedding
    except Exception as e:
        logging.error(f"Error generating query embedding: {e}")
        raise


def detect_language(text: str) -> str:
    """Detect the language of the text using LLM"""
    sample_text = text[:500]  # Use first 500 chars for detection
    messages = [
        {
            "role": "user",
            "content": f"""Detect the language of this text and respond with only the language name in English:

{sample_text}

Respond with only one word: the language name (e.g., "Italian", "English", "French", "Spanish", etc.)"""
        }
    ]
    
    language, success = llm_call(messages, max_tokens=10)
    if success and language.strip():
        return language.strip().lower()
    return "english"  # fallback

def highlight_relevant_text(query: str, page_text: str) -> str:
    """Use LLM to find and highlight relevant parts of the text and explain relevance"""
    # Detect the language of the page text
    detected_language = detect_language(page_text)
    
    # Create language-specific instructions
    if detected_language == "italian":
        language_instruction = "Provide all explanations in Italian. Use natural, fluent Italian for the [EXPLAIN] sections."
        example_format = "Testo qui [HIGHLIGHT]persone che si riuniscono nella piazza[/HIGHLIGHT] [EXPLAIN]Questo si collega alla query su \"riunione\" perché mostra persone che si incontrano in uno spazio pubblico[/EXPLAIN] altro testo."
    elif detected_language == "spanish":
        language_instruction = "Provide all explanations in Spanish. Use natural, fluent Spanish for the [EXPLAIN] sections."
        example_format = "Texto aquí [HIGHLIGHT]personas reuniéndose en la plaza[/HIGHLIGHT] [EXPLAIN]Esto se relaciona con la consulta sobre \"reunión\" porque muestra personas juntándose en un espacio público[/EXPLAIN] más texto."
    elif detected_language == "french":
        language_instruction = "Provide all explanations in French. Use natural, fluent French for the [EXPLAIN] sections."
        example_format = "Texte ici [HIGHLIGHT]personnes se rassemblant sur la place[/HIGHLIGHT] [EXPLAIN]Ceci se rapporte à la requête sur \"réunion\" car cela montre des personnes se rassemblant dans un espace public[/EXPLAIN] plus de texte."
    else:
        language_instruction = "Provide all explanations in English."
        example_format = "Some text here [HIGHLIGHT]people gathering in the square[/HIGHLIGHT] [EXPLAIN]This relates to the query about \"meeting\" because it shows people coming together in a public space[/EXPLAIN] more text."
    
    messages = [
        {
            "role": "user", 
            "content": f"""Given this query: "{query}"

And this page text:
{page_text}

This text was found by semantic search as being related to the query. Your task is to:
1. Identify parts of the text that could be semantically or contextually related to the query (even without exact word matches)
2. Wrap these related parts with [HIGHLIGHT] and [/HIGHLIGHT] tags
3. After each highlighted section, add an explanation wrapped in [EXPLAIN] and [/EXPLAIN] tags explaining WHY this content is semantically related

{language_instruction}

Look for meaningful connections between the query and text:
- Direct word matches or synonyms
- Conceptual connections (e.g., if query is "meeting", highlight gatherings, planning sessions, coordination)
- Contextual relationships (e.g., if query is "conflict", highlight tension, disagreements, confrontations)
- Thematic similarities (e.g., if query is "leadership", highlight decision-making, authority, guidance)
- Emotional or situational parallels

Return the ENTIRE original text with highlights and explanations added.

Example format:
{example_format}

Rules:
1. Return the complete original text
2. Always provide at least one highlight and explanation, even if the connection is weak or tenuous
3. Explain the semantic/conceptual connection in each [EXPLAIN] tag
4. Look beyond exact word matches to find thematic and contextual relevance
5. If only weak connections exist, be honest about their tenuous nature in the explanation
6. Don't change any of the original text content
7. For weak matches, use phrases like "weak connection", "tenuous link", "possibly related" in explanations
8. {language_instruction}"""
        }
    ]
    
    highlighted_text, success = llm_call(messages, max_tokens=6000)
    
    if not success:
        return page_text
    
    # Replace highlight tags with ANSI color codes for yellow background
    highlighted_text = highlighted_text.replace('[HIGHLIGHT]', '\033[43m\033[30m')  # Yellow background, black text
    highlighted_text = highlighted_text.replace('[/HIGHLIGHT]', '\033[0m')  # Reset color
    
    # Replace explanation tags with green color
    highlighted_text = highlighted_text.replace('[EXPLAIN]', '\033[32m')  # Green text
    highlighted_text = highlighted_text.replace('[/EXPLAIN]', '\033[0m')  # Reset color
    
    return highlighted_text

def create_page_border(title: str, width: int = 70) -> Tuple[str, str, str]:
    """Create a decorative border for page display"""
    horizontal = "─" * (width - 2)
    top = f"╭{horizontal}╮"
    middle = f"│ {title:<{width-4}} │"
    bottom = f"╰{horizontal}╯"
    return top, middle, bottom

def process_highlighted_text(highlighted_text: str) -> Tuple[str, List[str]]:
    """Separate the main text from explanations and add footnote numbering"""
    import re
    
    # Extract explanations
    explanations = []
    explanation_pattern = r'\033\[32m(.*?)\033\[0m'
    explanations = re.findall(explanation_pattern, highlighted_text)
    
    # Replace explanation sections with footnote numbers sequentially
    clean_text = highlighted_text
    footnote_counter = 1
    
    # Replace each explanation pattern with footnote reference
    def replace_explanation(match):
        nonlocal footnote_counter
        footnote_ref = f'\033[94m[{footnote_counter}]\033[0m'  # Blue footnote number
        footnote_counter += 1
        return footnote_ref
    
    clean_text = re.sub(explanation_pattern, replace_explanation, clean_text)
    
    return clean_text.strip(), explanations

def display_results(similarities: List[Tuple[str, int, str, float]], 
                   query: str,
                   top_k: int = 3, 
                   min_similarity: float = 0.0,
                   show_text: bool = True):
    """Display the search results with improved formatting"""
    
    # Main header
    header_top, header_middle, header_bottom = create_page_border(f"✻ SEARCH RESULTS (Top {top_k} matches for: '{query}')", 80)
    print(f"\n{header_top}")
    print(header_middle)
    print(header_bottom)
    
    count = 0
    for pdf_name, page_number, text_content, similarity in similarities:
        if similarity >= min_similarity and count < top_k:
            # Page result border
            page_title = f"📄 {pdf_name} - Page {page_number} │ Similarity: {similarity:.4f} │ Rank {count + 1}"
            page_top, page_middle, page_bottom = create_page_border(page_title, 80)
            
            print(f"\n{page_top}")
            print(page_middle)
            print(page_bottom)
            
            if show_text:
                try:
                    # Content border
                    content_top, content_middle, content_bottom = create_page_border("📖 CONTENT", 80)
                    print(f"\n{content_top}")
                    print(content_middle)
                    print(content_bottom)
                    
                    # Use LLM to highlight relevant parts
                    highlighted_text = highlight_relevant_text(query, text_content)
                    clean_text, explanations = process_highlighted_text(highlighted_text)
                    
                    # Display the main content with highlights
                    print(clean_text)
                    
                    # Display explanations separately if any
                    if explanations:
                        explain_top, explain_middle, explain_bottom = create_page_border("💡 RELEVANCE ANALYSIS", 80)
                        print(f"\n{explain_top}")
                        print(explain_middle)
                        print(explain_bottom)
                        
                        for i, explanation in enumerate(explanations, 1):
                            print(f"\n\033[94m[{i}]\033[0m \033[36m{explanation}\033[0m")  # Blue footnote number + cyan explanation
                    
                except Exception as e:
                    print(f"Error processing text content: {e}")
            
            count += 1
    
    if count == 0:
        print(f"No results found with similarity >= {min_similarity}")
    
    # Summary border
    summary_info = f"📊 Total results: {len(similarities)}"
    if similarities:
        best_result = similarities[0]
        worst_result = similarities[-1]
        summary_info += f" │ Best: {best_result[0]} Page {best_result[1]} ({best_result[3]:.4f}) │ Worst: {worst_result[0]} Page {worst_result[1]} ({worst_result[3]:.4f})"
    
    summary_top, summary_middle, summary_bottom = create_page_border(summary_info, 100)
    print(f"\n{summary_top}")
    print(summary_middle)
    print(summary_bottom)

def main():
    parser = argparse.ArgumentParser(description='Query PDF pages using semantic similarity')
    parser.add_argument('query', nargs='?', help='Query text to search for')
    parser.add_argument('--pdf', '-p', help='Name of specific PDF to search (optional, searches all by default)')
    parser.add_argument('-k', '--top-k', type=int, default=3, help='Number of top results to show (default: 3)')
    parser.add_argument('-s', '--min-similarity', type=float, default=0.0, 
                       help='Minimum similarity threshold (default: 0.0)')
    parser.add_argument('--no-text', action='store_true', 
                       help='Hide text content (text shown by default)')
    parser.add_argument('--list', action='store_true', 
                       help='List available PDF collections and exit')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    setup_logging()
    
    # Handle --list option
    if args.list:
        collections = list_available_collections()
        if collections:
            print("📚 Available books in database:")
            print("=" * 60)
            total_pages = 0
            for book_name, page_count in collections:
                print(f"  📖 {book_name:<40} {page_count:>6} pages")
                total_pages += page_count
            print("=" * 60)
            print(f"  📊 Total: {len(collections)} books, {total_pages} pages")
        else:
            print("No PDF collections found. Run ingest.py first to process documents.")
        sys.exit(0)
    
    # Check if query is provided (required unless using --list)
    if not args.query:
        parser.error("Query text is required unless using --list")
    
    try:
        # Check embedding system
        logging.info("Checking embedding system...")
        check_embedding_system()
        
        # Generate query embedding
        query_embedding = generate_query_embedding(args.query)
        
        # Query Chroma collections
        logging.info("Querying Chroma collections...")
        similarities = query_chroma_collections(query_embedding, max(args.top_k, 20), args.pdf)
        
        if not similarities:
            logging.error("No results found. Make sure you've run ingest.py first.")
            sys.exit(1)
        
        # Display results (show text by default unless --no-text is specified)
        show_text = not args.no_text
        display_results(similarities, args.query, args.top_k, args.min_similarity, show_text)
        
    except Exception as e:
        logging.error(f"Query failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()