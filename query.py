#!/usr/bin/env python3
import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from llm_wrapper import llm_call

def setup_logging():
    """Configure logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def load_embedding_model():
    """Load the same embedding model used in ingestion"""
    try:
        model_name = "all-MiniLM-L6-v2"  # Use the fallback model that worked
        logging.info(f"Loading embedding model: {model_name}")
        model = SentenceTransformer(model_name)
        logging.info("Embedding model loaded successfully")
        return model
    except Exception as e:
        logging.error(f"Error loading embedding model: {e}")
        raise

def find_page_embeddings(pdf_name: str) -> List[Tuple[int, str, np.ndarray]]:
    """
    Find all page embeddings for a given PDF name.
    Returns list of (page_number, text_file_path, embedding_array)
    """
    embeddings = []
    
    # Look for all embedding files matching the pattern
    current_dir = Path('.')
    embedding_files = list(current_dir.glob(f"{pdf_name}_page_*_embedding.npz"))
    
    if not embedding_files:
        logging.error(f"No embedding files found for PDF: {pdf_name}")
        return embeddings
    
    logging.info(f"Found {len(embedding_files)} embedding files")
    
    for embedding_file in embedding_files:
        try:
            # Extract page number from filename
            filename = embedding_file.stem  # Remove .npz extension
            page_part = filename.split('_page_')[1].split('_embedding')[0]
            page_number = int(page_part)
            
            # Load embedding
            embedding_data = np.load(embedding_file)
            embedding = embedding_data['embedding']
            
            # Find corresponding text file
            text_file = f"{pdf_name}_page_{page_number}.txt"
            
            if Path(text_file).exists():
                embeddings.append((page_number, text_file, embedding))
            else:
                logging.warning(f"Text file not found: {text_file}")
                
        except Exception as e:
            logging.error(f"Error processing {embedding_file}: {e}")
            continue
    
    # Sort by page number
    embeddings.sort(key=lambda x: x[0])
    return embeddings

def generate_query_embedding(query: str, model: SentenceTransformer) -> np.ndarray:
    """Generate embedding for the query text"""
    try:
        logging.info(f"Generating embedding for query: '{query[:50]}...'")
        embedding = model.encode([query], convert_to_numpy=True)
        return embedding[0]  # Return single embedding array
    except Exception as e:
        logging.error(f"Error generating query embedding: {e}")
        raise

def calculate_similarities(query_embedding: np.ndarray, 
                         page_embeddings: List[Tuple[int, str, np.ndarray]]) -> List[Tuple[int, str, float]]:
    """
    Calculate cosine similarities between query and all page embeddings.
    Returns list of (page_number, text_file, similarity_score) sorted by similarity.
    """
    similarities = []
    
    for page_number, text_file, page_embedding in page_embeddings:
        try:
            # Calculate cosine similarity
            similarity = cosine_similarity([query_embedding], [page_embedding])[0][0]
            similarities.append((page_number, text_file, similarity))
        except Exception as e:
            logging.error(f"Error calculating similarity for page {page_number}: {e}")
            continue
    
    # Sort by similarity score (highest first)
    similarities.sort(key=lambda x: x[2], reverse=True)
    return similarities

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

def display_results(similarities: List[Tuple[int, str, float]], 
                   query: str,
                   top_k: int = 5, 
                   min_similarity: float = 0.0,
                   show_text: bool = False):
    """Display the search results with improved formatting"""
    
    # Main header
    header_top, header_middle, header_bottom = create_page_border(f"✻ SEARCH RESULTS (Top {top_k} matches for: '{query}')", 80)
    print(f"\n{header_top}")
    print(header_middle)
    print(header_bottom)
    
    count = 0
    for page_number, text_file, similarity in similarities:
        if similarity >= min_similarity and count < top_k:
            # Page result border
            page_title = f"📄 Page {page_number} │ Similarity: {similarity:.4f} │ Rank {count + 1}"
            page_top, page_middle, page_bottom = create_page_border(page_title, 80)
            
            print(f"\n{page_top}")
            print(page_middle)
            print(page_bottom)
            
            if show_text:
                try:
                    with open(text_file, 'r', encoding='utf-8') as f:
                        text = f.read().strip()
                    
                    # Content border
                    content_top, content_middle, content_bottom = create_page_border("📖 CONTENT", 80)
                    print(f"\n{content_top}")
                    print(content_middle)
                    print(content_bottom)
                    
                    # Use LLM to highlight relevant parts
                    highlighted_text = highlight_relevant_text(query, text)
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
                    print(f"Error reading text file: {e}")
            
            count += 1
    
    if count == 0:
        print(f"No results found with similarity >= {min_similarity}")
    
    # Summary border
    summary_info = f"📊 Total pages: {len(similarities)}"
    if similarities:
        summary_info += f" │ Best: Page {similarities[0][0]} ({similarities[0][2]:.4f}) │ Worst: Page {similarities[-1][0]} ({similarities[-1][2]:.4f})"
    
    summary_top, summary_middle, summary_bottom = create_page_border(summary_info, 100)
    print(f"\n{summary_top}")
    print(summary_middle)
    print(summary_bottom)

def main():
    parser = argparse.ArgumentParser(description='Query PDF pages using semantic similarity')
    parser.add_argument('pdf_name', help='Name of the PDF (without .pdf extension)')
    parser.add_argument('query', help='Query text to search for')
    parser.add_argument('-k', '--top-k', type=int, default=5, help='Number of top results to show (default: 5)')
    parser.add_argument('-s', '--min-similarity', type=float, default=0.0, 
                       help='Minimum similarity threshold (default: 0.0)')
    parser.add_argument('-t', '--show-text', action='store_true', 
                       help='Show text content preview for results')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    setup_logging()
    
    try:
        # Load embedding model
        logging.info("Loading embedding model...")
        model = load_embedding_model()
        
        # Find page embeddings
        logging.info(f"Loading page embeddings for: {args.pdf_name}")
        page_embeddings = find_page_embeddings(args.pdf_name)
        
        if not page_embeddings:
            logging.error("No page embeddings found. Make sure you've run ingest.py first.")
            sys.exit(1)
        
        # Generate query embedding
        query_embedding = generate_query_embedding(args.query, model)
        
        # Calculate similarities
        logging.info("Calculating similarities...")
        similarities = calculate_similarities(query_embedding, page_embeddings)
        
        # Display results
        display_results(similarities, args.query, args.top_k, args.min_similarity, args.show_text)
        
    except Exception as e:
        logging.error(f"Query failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()