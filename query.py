#!/usr/bin/env python3
import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import re
import chromadb
from llm_wrapper import llm_call, generate_embeddings, test_openai_embeddings, check_openai_api
from sqlite_fts5 import SQLiteFTS5Manager
from llm_reranker import llm_rerank_results

def setup_logging():
    """Configure logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def check_embedding_system():
    """Check if OpenAI API key is available and embedding API is working"""
    logging.info("Checking OpenAI embedding API...")
    if not test_openai_embeddings():
        logging.error("OpenAI embedding API not available")
        logging.error("Please ensure OPENAI_API_KEY is set in your environment")
        raise Exception("OpenAI embedding API not available")
    else:
        logging.info("OpenAI embedding API is working successfully")
    
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
    """Generate embedding for the query text using OpenAI text-embedding-3-large"""
    try:
        logging.info(f"Generating embedding for query: '{query[:50]}...'")
        embedding = generate_embeddings(query, normalize=True)
        return embedding
    except Exception as e:
        logging.error(f"Error generating query embedding: {e}")
        raise

def query_fts5_collections(query: str, top_k: int = 10, pdf_name: str = None, use_enhancement: bool = True) -> List[Tuple[str, int, str, float]]:
    """
    Query SQLite FTS5 collections for keyword-based matches with optional query enhancement.
    Returns list of (pdf_name, page_number, text_content, bm25_score)
    """
    try:
        search_query = query
        
        # Try query enhancement for better keyword matching
        if use_enhancement:
            try:
                logging.info(f"Enhancing query: '{query}'")
                enhancement = enhance_query(query)
                enhanced_query = enhancement.get('enhanced_query', query)
                translation = enhancement.get('translation', query)
                synonyms = enhancement.get('synonyms', [])
                
                # Create expanded search query with translation and synonyms
                # FTS5 syntax: use OR without quotes for multi-term queries
                search_terms = [translation] + synonyms[:3]  # Use translation and top synonyms
                # Clean terms and remove duplicates
                clean_terms = []
                for term in search_terms:
                    if term and term.strip() and term not in clean_terms:
                        # Split multi-word terms and use individual words
                        words = term.split()
                        clean_terms.extend([word for word in words if len(word) > 2])  # Skip very short words
                
                if clean_terms:
                    # Join terms with spaces - FTS5Manager will handle OR logic internally
                    search_query = ' '.join(clean_terms[:5])  # Limit to 5 terms to avoid complexity
                else:
                    search_query = query  # Fallback to original
                
                logging.info(f"Enhanced search query: {search_query}")
                logging.info(f"Enhancement strategy: {enhancement.get('search_strategy', 'unknown')}")
                
            except Exception as e:
                logging.warning(f"Query enhancement failed, using original: {e}")
                search_query = query
        
        fts5_manager = SQLiteFTS5Manager()
        
        # Search using FTS5 BM25-style ranking with enhanced query
        results = fts5_manager.search(
            query=search_query,
            pdf_name=pdf_name,
            limit=top_k,
            include_summary=True
        )
        
        fts5_manager.close()
        
        logging.info(f"Found {len(results)} FTS5 keyword results (enhanced: {use_enhancement})")
        
        # Convert to expected format: (pdf_name, page_number, text_content, score)
        formatted_results = []
        for pdf_name, page_number, content, score, metadata in results:
            formatted_results.append((pdf_name, page_number, content, score))
        
        return formatted_results
        
    except Exception as e:
        logging.error(f"Error querying FTS5 collections: {e}")
        return []

def hybrid_search(query: str, top_k: int = 10, pdf_name: str = None, 
                 semantic_weight: float = 0.6, bm25_weight: float = 0.4, use_enhancement: bool = True, use_reranking: bool = False, language: str = "auto") -> List[Tuple[str, int, str, float]]:
    """
    Perform hybrid search combining semantic similarity (ChromaDB) and keyword matching (FTS5).
    
    Args:
        query: Search query
        top_k: Number of results to return
        pdf_name: Optional specific PDF to search
        semantic_weight: Weight for semantic search results (default: 0.6)
        bm25_weight: Weight for BM25 keyword search results (default: 0.4)
        use_enhancement: Whether to use query enhancement (default: True)
        use_reranking: Whether to apply LLM reranking to results (default: False)
        language: Language hint for LLM reranking (default: "auto")
    
    Returns:
        List of (pdf_name, page_number, text_content, combined_score) sorted by combined score
    """
    try:
        # Ensure weights sum to 1.0
        total_weight = semantic_weight + bm25_weight
        if total_weight > 0:
            semantic_weight = semantic_weight / total_weight
            bm25_weight = bm25_weight / total_weight
        
        logging.info(f"Performing hybrid search with {semantic_weight:.2f} semantic + {bm25_weight:.2f} keyword weighting")
        
        # Get semantic search results
        query_embedding = generate_query_embedding(query)
        semantic_results = query_chroma_collections(query_embedding, top_k * 2, pdf_name)
        
        # Get keyword search results with enhancement
        keyword_results = query_fts5_collections(query, top_k * 2, pdf_name, use_enhancement=use_enhancement)
        
        # Remove duplicates and normalize scores for fair combination
        def deduplicate_results(results: List[Tuple[str, int, str, float]]) -> List[Tuple[str, int, str, float]]:
            """Remove duplicate results, keeping the best score for each unique (pdf, page, chunk) combination"""
            seen = {}  # (pdf_name, page_number, text_hash) -> best_score
            unique_results = []
            
            for pdf_name, page_number, text, score in results:
                # Create a simple hash of the text content to identify duplicates
                text_hash = hash(text.strip()[:100])  # Use first 100 chars for hash
                key = (pdf_name, page_number, text_hash)
                
                if key not in seen or score > seen[key][3]:
                    seen[key] = (pdf_name, page_number, text, score)
            
            return list(seen.values())

        def normalize_scores(results: List[Tuple[str, int, str, float]]) -> List[Tuple[str, int, str, float]]:
            if not results:
                return []
            
            scores = [score for _, _, _, score in results]
            max_score = max(scores) if scores else 1.0
            min_score = min(scores) if scores else 0.0
            
            # Avoid division by zero
            if max_score == min_score:
                return [(pdf, page, text, 0.5) for pdf, page, text, _ in results]
            
            normalized = []
            for pdf, page, text, score in results:
                # Min-max normalization to [0, 1]
                norm_score = (score - min_score) / (max_score - min_score)
                normalized.append((pdf, page, text, norm_score))
            
            return normalized
        
        # Deduplicate and normalize both result sets
        semantic_deduped = deduplicate_results(semantic_results)
        keyword_deduped = deduplicate_results(keyword_results)
        
        semantic_normalized = normalize_scores(semantic_deduped)
        keyword_normalized = normalize_scores(keyword_deduped)
        
        # Combine results using weighted scoring
        combined_scores = {}  # (pdf_name, page_number, text_snippet) -> combined_score
        
        # Add semantic scores
        for pdf_name, page_number, text, norm_score in semantic_normalized:
            key = (pdf_name, page_number, text[:50])  # Use text snippet as part of key
            combined_scores[key] = {
                'text': text,
                'semantic_score': norm_score,
                'keyword_score': 0.0,
                'combined_score': semantic_weight * norm_score
            }
        
        # Add keyword scores
        for pdf_name, page_number, text, norm_score in keyword_normalized:
            key = (pdf_name, page_number, text[:50])
            if key in combined_scores:
                # Update existing entry
                combined_scores[key]['keyword_score'] = norm_score
                combined_scores[key]['combined_score'] += bm25_weight * norm_score
            else:
                # Create new entry (keyword-only result)
                combined_scores[key] = {
                    'text': text,
                    'semantic_score': 0.0,
                    'keyword_score': norm_score,
                    'combined_score': bm25_weight * norm_score
                }
        
        # Convert back to list format and sort by combined score
        hybrid_results = []
        for (pdf_name, page_number, _), scores_data in combined_scores.items():
            hybrid_results.append((pdf_name, page_number, scores_data['text'], scores_data['combined_score']))
        
        # Sort by combined score (highest first)
        hybrid_results.sort(key=lambda x: x[3], reverse=True)
        
        logging.info(f"Hybrid search: {len(semantic_results)} semantic → {len(semantic_deduped)} deduped")
        logging.info(f"Hybrid search: {len(keyword_results)} keyword → {len(keyword_deduped)} deduped")
        logging.info(f"Final hybrid results: {len(hybrid_results)} unique combinations")
        
        if hybrid_results and logging.getLogger().isEnabledFor(logging.DEBUG):
            logging.debug("Top 3 hybrid scores:")
            for i, (pdf, page, text, score) in enumerate(hybrid_results[:3]):
                logging.debug(f"  {i+1}. {pdf} p.{page}: {score:.4f} - {text[:50]}...")
        
        # Apply LLM reranking if requested and we have enough candidates
        if use_reranking and len(hybrid_results) >= 8:
            try:
                logging.info(f"Applying LLM reranking to {len(hybrid_results)} hybrid results")
                reranked_results = llm_rerank_results(
                    query=query,
                    candidates=hybrid_results[:25],  # Limit to 25 for cost control
                    language=language
                )
                logging.info("LLM reranking completed successfully")
                return reranked_results[:top_k]
            except Exception as e:
                logging.warning(f"LLM reranking failed, using hybrid scores: {e}")
        
        return hybrid_results[:top_k]
        
    except Exception as e:
        logging.error(f"Error in hybrid search: {e}")
        # Fallback to semantic search only
        logging.warning("Falling back to semantic search only")
        try:
            query_embedding = generate_query_embedding(query)
            return query_chroma_collections(query_embedding, top_k, pdf_name)
        except Exception as fallback_error:
            logging.error(f"Fallback semantic search also failed: {fallback_error}")
            return []


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

def enhance_query(original_query: str, target_language: str = "english") -> dict:
    """
    Enhance and translate query using LLM for better search performance.
    
    Args:
        original_query: The original search query
        target_language: Target language for translation (default: english)
    
    Returns:
        dict with enhanced_query, translations, synonyms, and related_terms
    """
    messages = [
        {
            "role": "user",
            "content": f"""You are a search query enhancement expert. Your task is to improve this search query for better document retrieval.

Original Query: "{original_query}"

Please provide:
1. TRANSLATION: If the query is not in English, translate it to English
2. ENHANCED_QUERY: Improve the query by adding relevant terms, synonyms, and alternative phrasings
3. SYNONYMS: List alternative terms and synonyms that could be used
4. RELATED_TERMS: Add conceptually related terms that might appear in relevant documents

Format your response as JSON:
{{
    "original_query": "{original_query}",
    "detected_language": "language_name",
    "translation": "english translation if needed",
    "enhanced_query": "improved searchable version with multiple terms",
    "synonyms": ["synonym1", "synonym2", "synonym3"],
    "related_terms": ["related1", "related2", "related3"],
    "search_strategy": "brief explanation of enhancement approach"
}}

Examples:
- "Nettuno" → enhance with "Neptune planet solar system eighth gas giant blue methane atmosphere"
- "strategie di marketing" → enhance with "marketing strategies business promotional tactics advertising campaigns"
- "machine learning" → enhance with "artificial intelligence AI neural networks deep learning algorithms"

Focus on terms that would likely appear in academic, technical, or reference documents."""
        }
    ]
    
    try:
        response, success = llm_call(messages, max_tokens=500)
        if success:
            import json
            # Clean the response to extract JSON
            response = response.strip()
            if response.startswith('```json'):
                response = response[7:]
            if response.endswith('```'):
                response = response[:-3]
            
            enhancement_data = json.loads(response.strip())
            return enhancement_data
        else:
            # Fallback: basic structure
            return {
                "original_query": original_query,
                "detected_language": "unknown",
                "translation": original_query,
                "enhanced_query": original_query,
                "synonyms": [],
                "related_terms": [],
                "search_strategy": "fallback - no enhancement applied"
            }
    except Exception as e:
        logging.warning(f"Query enhancement failed: {e}")
        # Fallback: return original query
        return {
            "original_query": original_query,
            "detected_language": "unknown", 
            "translation": original_query,
            "enhanced_query": original_query,
            "synonyms": [],
            "related_terms": [],
            "search_strategy": "fallback - enhancement error"
        }

def highlight_relevant_text_batch(query: str, results: List[Tuple[str, int, str, float]], output_format_ansi: bool = True, force_language: str = None) -> List[str]:
    """Use LLM to batch highlight relevant parts of multiple texts and explain relevance"""
    if not results:
        return []
    
    # Use forced language or detect language from first result
    if force_language:
        detected_language = force_language.lower()
    else:
        detected_language = detect_language(results[0][2])
    
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
    
    # Build batch content
    batch_content = f'Given this query: "{query}"\n\n'
    batch_content += "I will provide multiple texts found by semantic search. For each text, identify parts that could be semantically or contextually related to the query and add highlights and explanations.\n\n"
    
    for i, (pdf_name, page_number, text_content, similarity) in enumerate(results, 1):
        batch_content += f"--- TEXT {i} (Source: {pdf_name}, Page: {page_number}) ---\n"
        batch_content += f"{text_content}\n\n"
    
    batch_content += f"""For each text above, your task is to:
1. Identify parts that could be semantically or contextually related to the query (even without exact word matches)
2. Wrap these related parts with [HIGHLIGHT] and [/HIGHLIGHT] tags
3. After each highlighted section, add an explanation wrapped in [EXPLAIN] and [/EXPLAIN] tags explaining WHY this content is semantically related

{language_instruction}

Look for meaningful connections between the query and text:
- Direct word matches or synonyms
- Conceptual connections (e.g., if query is "meeting", highlight gatherings, planning sessions, coordination)
- Contextual relationships (e.g., if query is "conflict", highlight tension, disagreements, confrontations)
- Thematic similarities (e.g., if query is "leadership", highlight decision-making, authority, guidance)
- Emotional or situational parallels

Return each text with its highlights and explanations, keeping the same order and using the format:

--- HIGHLIGHTED TEXT 1 ---
[highlighted version of text 1]

--- HIGHLIGHTED TEXT 2 ---
[highlighted version of text 2]

... and so on.

Example format for highlighting:
{example_format}

Rules:
1. Return the complete original text for each
2. Always provide at least one highlight and explanation per text, even if the connection is weak or tenuous
3. Explain the semantic/conceptual connection in each [EXPLAIN] tag
4. Look beyond exact word matches to find thematic and contextual relevance
5. If only weak connections exist, be honest about their tenuous nature in the explanation
6. Don't change any of the original text content
7. For weak matches, use phrases like "weak connection", "tenuous link", "possibly related" in explanations
8. {language_instruction}"""

    messages = [{"role": "user", "content": batch_content}]
    
    highlighted_batch, success = llm_call(messages, max_tokens=12000)
    
    if not success:
        # Fallback to individual highlighting
        return [highlight_relevant_text(query, text_content, output_format_ansi=output_format_ansi, force_language=force_language) for _, _, text_content, _ in results]
    
    # Parse the batch response to extract individual highlighted texts
    highlighted_texts = []
    sections = highlighted_batch.split("--- HIGHLIGHTED TEXT")
    
    for i, section in enumerate(sections[1:], 1):  # Skip first empty section
        # Extract content after the header line
        lines = section.strip().split('\n')
        if len(lines) > 1:
            content = '\n'.join(lines[1:]).strip()
            if content.startswith('---'):
                content = '\n'.join(content.split('\n')[1:]).strip()
            
            if output_format_ansi:
                # Apply color formatting
                content = content.replace('[HIGHLIGHT]', '\033[43m\033[30m')  # Yellow background, black text
                content = content.replace('[/HIGHLIGHT]', '\033[0m')  # Reset color
                content = content.replace('[EXPLAIN]', '\033[32m')  # Green text
                content = content.replace('[/EXPLAIN]', '\033[0m')  # Reset color
            
            highlighted_texts.append(content)
        else:
            # Fallback for malformed sections
            highlighted_texts.append(results[i-1][2] if i-1 < len(results) else "")
    
    # Ensure we have the right number of results
    while len(highlighted_texts) < len(results):
        idx = len(highlighted_texts)
        highlighted_texts.append(highlight_relevant_text(query, results[idx][2], output_format_ansi=output_format_ansi, force_language=force_language))
    
    return highlighted_texts[:len(results)]

def highlight_relevant_text(query: str, page_text: str, output_format_ansi: bool = True, force_language: str = None) -> str:
    """Use LLM to find and highlight relevant parts of the text and explain relevance (single text version)"""
    # Use forced language or detect the language of the page text
    if force_language:
        detected_language = force_language.lower()
    else:
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
    
    if output_format_ansi:
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

def analyze_individual_result(query: str, result_text: str, pdf_name: str, page_number: int, similarity: float, force_language: str = None) -> str:
    """Use LLM to analyze and comment on individual search result relevance"""
    if force_language:
        detected_language = force_language.lower()
    else:
        detected_language = detect_language(result_text)
    
    # Create language-specific instructions
    if detected_language == "italian":
        language_instruction = "Rispondi in italiano con un'analisi fluente e naturale."
    elif detected_language == "spanish":
        language_instruction = "Responde en español con un análisis fluente y natural."
    elif detected_language == "french":
        language_instruction = "Répondez en français avec une analyse fluide et naturelle."
    else:
        language_instruction = "Respond in English with fluent, natural analysis."
    
    messages = [
        {
            "role": "user",
            "content": f"""Analyze this search result for the query: "{query}"

Document: {pdf_name}
Page: {page_number}
Similarity Score: {similarity:.4f}

Text Content:
{result_text}

Provide a brief analytical comment (2-3 sentences) explaining:
1. Why this result is relevant to the query
2. What key concepts or themes connect it to the search
3. How useful this result would be for someone researching this topic

{language_instruction}

Be concise but insightful. Focus on semantic connections and practical value."""
        }
    ]
    
    analysis, success = llm_call(messages, max_tokens=200)
    if success and analysis.strip():
        return analysis.strip()
    return "Analysis unavailable for this result."

def synthesize_results(query: str, results: List[Tuple[str, int, str, float]], max_results: int = 5, force_language: str = None) -> str:
    """Use LLM to synthesize and conglomerate multiple search results"""
    if not results:
        return "No results to synthesize."
    
    # Take top results for synthesis
    top_results = results[:max_results]
    
    # Use forced language or detect language from first result
    if force_language:
        detected_language = force_language.lower()
    else:
        detected_language = detect_language(top_results[0][2])
    
    # Create language-specific instructions
    if detected_language == "italian":
        language_instruction = "Fornisci la sintesi in italiano con analisi fluente e naturale."
        synthesis_prompt = "Sintesi dei Risultati di Ricerca"
    elif detected_language == "spanish":
        language_instruction = "Proporciona la síntesis en español con análisis fluente y natural."
        synthesis_prompt = "Síntesis de Resultados de Búsqueda"
    elif detected_language == "french":
        language_instruction = "Fournissez la synthèse en français avec une analyse fluide et naturelle."
        synthesis_prompt = "Synthèse des Résultats de Recherche"
    else:
        language_instruction = "Provide synthesis in English with fluent, natural analysis."
        synthesis_prompt = "Search Results Synthesis"
    
    # Build context from all results
    results_context = ""
    for i, (pdf_name, page_num, text, similarity) in enumerate(top_results, 1):
        results_context += f"\n--- Result {i} (Similarity: {similarity:.4f}) ---\n"
        results_context += f"Source: {pdf_name}, Page {page_num}\n"
        results_context += f"Content: {text[:400]}{'...' if len(text) > 400 else ''}\n"
    
    messages = [
        {
            "role": "user",
            "content": f"""Query: "{query}"

Here are the top {len(top_results)} search results:
{results_context}

Please provide a comprehensive synthesis that:

1. **Overall Assessment**: How well do these results address the query?
2. **Key Themes**: What main topics, concepts, or patterns emerge across results?
3. **Cross-References**: How do the different results complement or contradict each other?
4. **Research Value**: What insights can be drawn from viewing these results together?
5. **Gaps & Limitations**: What aspects of the query might need additional sources?

{language_instruction}

Provide a structured analysis (3-4 paragraphs) that helps someone understand the collective meaning and value of these search results."""
        }
    ]
    
    synthesis, success = llm_call(messages, max_tokens=800)
    if success and synthesis.strip():
        return synthesis.strip()
    return "Synthesis unavailable for these results."

def generate_direct_answer(query: str, results: List[Tuple[str, int, str, float]], max_results: int = 5, force_language: str = None) -> str:
    """Generate a comprehensive, detailed answer to the query based on search results with source attribution"""
    if not results:
        return "No information found to answer the query."
    
    # Take top results for answer generation
    top_results = results[:max_results]
    
    # Use forced language or detect language from first result
    if force_language:
        detected_language = force_language.lower()
    else:
        detected_language = detect_language(top_results[0][2])
    
    # Create language-specific instructions
    if detected_language == "italian":
        language_instruction = "Rispondi in italiano con una risposta dettagliata e completa. Distingui chiaramente tra informazioni dalle fonti documentali e conoscenza generale."
        source_format_instruction = "Per le informazioni dalle fonti usa: **testo dalle fonti** *(nome_documento, p.XX)* \nPer la conoscenza generale usa: [testo generale...]"
    elif detected_language == "spanish":
        language_instruction = "Responde en español con una respuesta detallada y completa. Distingue claramente entre información de las fuentes documentales y conocimiento general."
        source_format_instruction = "Para información de fuentes usa: **texto de fuentes** *(nombre_documento, p.XX)* \nPara conocimiento general usa: [texto general...]"
    elif detected_language == "french":
        language_instruction = "Répondez en français avec una réponse détaillée et complète. Distinguez clairement entre les informations des sources documentaires et les connaissances générales."
        source_format_instruction = "Pour les informations des sources utilisez: **texte des sources** *(nom_document, p.XX)* \nPour les connaissances générales utilisez: [texte général...]"
    else:
        language_instruction = "Respond in English with a detailed and comprehensive answer. Clearly distinguish between information from documentary sources and general knowledge."
        source_format_instruction = "For information from sources use: **text from sources** *(document_name, p.XX)* \nFor general knowledge use: [general text...]"
    
    # Build context from all results
    results_context = ""
    for i, (pdf_name, page_num, text, similarity) in enumerate(top_results, 1):
        results_context += f"\n--- Source {i}: {pdf_name}, Page {page_num} ---\n"
        results_context += f"{text}\n"
    
    messages = [
        {
            "role": "user",
            "content": f"""Query: "{query}"

Based on the following sources, provide a comprehensive and detailed answer to the query. Your response should be thorough and informative while staying focused on the question.

Sources:
{results_context}

CRITICAL FORMATTING REQUIREMENTS:
{source_format_instruction}

Instructions:
1. CLEARLY DISTINGUISH between information from sources and general knowledge
2. For ANY information taken directly from the provided sources, use **bold text** followed by source citation *(document_name, p.XX)*
3. For general knowledge or contextual information not in the sources, use [square brackets...]
4. Include specific details, examples, and explanations when available
5. Organize your response logically with clear structure
6. Synthesize information from multiple sources when relevant
7. Include key facts, concepts, and context that help fully answer the question
8. Use bullet points or numbered lists when appropriate for clarity
9. Do NOT include meta-analysis about search quality or methodology
10. Focus entirely on delivering substantive content that answers the query

{language_instruction}

Provide a detailed, comprehensive answer of 3-8 paragraphs that thoroughly addresses the query using the available information with proper source attribution."""
        }
    ]
    
    direct_answer, success = llm_call(messages, max_tokens=1200)
    if success and direct_answer.strip():
        return direct_answer.strip()
    return "Unable to generate a comprehensive answer from the available sources."

def process_highlighted_text(highlighted_text: str) -> Tuple[str, List[str]]:
    """Separate the main text from explanations and add footnote numbering"""
    
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
                   show_text: bool = True,
                   enhanced_analysis: bool = True,
                   dual_answer: bool = False,
                   force_language: str = None):
    """Display the search results with improved formatting and LLM analysis"""
    
    # Store results for synthesis
    displayed_results = []
    count = 0
    
    # Collect displayed results first
    for pdf_name, page_number, text_content, similarity in similarities:
        if similarity >= min_similarity and count < top_k:
            displayed_results.append((pdf_name, page_number, text_content, similarity))
            count += 1
    
    if count == 0:
        print(f"No results found with similarity >= {min_similarity}")
        return
    
    # If dual answer mode is enabled, show both parts
    if dual_answer:
        # PART 1: Direct Answer (no meta-analysis)
        direct_top, direct_middle, direct_bottom = create_page_border("🎯 DIRECT ANSWER", 80)
        print(f"\n{direct_top}")
        print(direct_middle)  
        print(direct_bottom)
        
        try:
            direct_answer = generate_direct_answer(query, displayed_results, force_language=force_language)
            print(f"\n\033[92m{direct_answer}\033[0m")  # Green for direct answer
        except Exception as e:
            print(f"\nError generating direct answer: {e}")
        
        # Separator
        print(f"\n{'='*80}")
        
        # PART 2: Quality Analysis & Dynamic Results (existing detailed analysis)
        analysis_header_top, analysis_header_middle, analysis_header_bottom = create_page_border("🔍 DETAILED ANALYSIS & SEARCH RESULTS", 80)
        print(f"\n{analysis_header_top}")
        print(analysis_header_middle)
        print(analysis_header_bottom)
    
    # Main header (for non-dual mode or Part 2 of dual mode)
    if not dual_answer:
        header_top, header_middle, header_bottom = create_page_border(f"✻ SEARCH RESULTS (Top {top_k} matches for: '{query}')", 80)
        print(f"\n{header_top}")
        print(header_middle)
        print(header_bottom)
    
    # Batch process highlighting for all results at once (if show_text is enabled)
    highlighted_texts = []
    if show_text:
        try:
            # Use batch highlighting for better performance
            # For terminal output, ANSI is desired.
            highlighted_texts = highlight_relevant_text_batch(query, displayed_results, output_format_ansi=True, force_language=force_language)
        except Exception as e:
            print(f"Error in batch highlighting, falling back to individual processing: {e}")
            # Fallback to individual highlighting
            highlighted_texts = [highlight_relevant_text(query, text_content, output_format_ansi=True, force_language=force_language) for _, _, text_content, _ in displayed_results]
    
    # Display individual results with analysis
    for i, (pdf_name, page_number, text_content, similarity) in enumerate(displayed_results):
        # Page result border
        page_title = f"📄 {pdf_name} - Page {page_number} │ Similarity: {similarity:.4f} │ Rank {i + 1}"
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
                
                # Use pre-processed highlighted text
                highlighted_text = highlighted_texts[i] if i < len(highlighted_texts) else text_content
                clean_text, explanations = process_highlighted_text(highlighted_text)
                
                # Display the main content with highlights
                print(clean_text)
                
                # Display explanations separately if any
                if explanations:
                    explain_top, explain_middle, explain_bottom = create_page_border("💡 RELEVANCE ANALYSIS", 80)
                    print(f"\n{explain_top}")
                    print(explain_middle)
                    print(explain_bottom)
                    
                    for j, explanation in enumerate(explanations, 1):
                        print(f"\n\033[94m[{j}]\033[0m \033[36m{explanation}\033[0m")  # Blue footnote number + cyan explanation
                
                # Individual result LLM analysis
                if enhanced_analysis:
                    analysis_top, analysis_middle, analysis_bottom = create_page_border("🧠 LLM ANALYSIS", 80)
                    print(f"\n{analysis_top}")
                    print(analysis_middle)
                    print(analysis_bottom)
                    
                    individual_analysis = analyze_individual_result(query, text_content, pdf_name, page_number, similarity, force_language=force_language)
                    print(f"\n\033[95m{individual_analysis}\033[0m")  # Magenta for analysis
                
            except Exception as e:
                print(f"Error processing text content: {e}")
    
    # Results synthesis (only if enhanced analysis is enabled)
    if enhanced_analysis and displayed_results:
        synthesis_top, synthesis_middle, synthesis_bottom = create_page_border("🔬 COMPREHENSIVE SYNTHESIS", 90)
        print(f"\n{synthesis_top}")
        print(synthesis_middle)
        print(synthesis_bottom)
        
        try:
            synthesis = synthesize_results(query, displayed_results, force_language=force_language)
            print(f"\n\033[93m{synthesis}\033[0m")  # Yellow for synthesis
        except Exception as e:
            print(f"\nError generating synthesis: {e}")
    
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
    parser = argparse.ArgumentParser(description='Query PDF pages using semantic similarity, keyword search, or hybrid search')
    parser.add_argument('query', nargs='?', help='Query text to search for')
    parser.add_argument('--pdf', '-p', help='Name of specific PDF to search (optional, searches all by default)')
    parser.add_argument('-k', '--top-k', type=int, default=3, help='Number of top results to show (default: 3)')
    parser.add_argument('-s', '--min-similarity', type=float, default=0.0, 
                       help='Minimum similarity threshold (default: 0.0)')
    
    # Search method options
    search_group = parser.add_mutually_exclusive_group()
    search_group.add_argument('--bm25', action='store_true',
                             help='Use BM25 keyword search instead of semantic search')
    search_group.add_argument('--hybrid', action='store_true', default=True,
                             help='Use hybrid search combining semantic + keyword (default)')
    search_group.add_argument('--semantic', action='store_true',
                             help='Use semantic search only (ChromaDB)')
    
    # Hybrid search weighting
    parser.add_argument('--semantic-weight', type=float, default=0.6,
                       help='Weight for semantic search in hybrid mode (default: 0.6)')
    parser.add_argument('--keyword-weight', type=float, default=0.4,
                       help='Weight for keyword search in hybrid mode (default: 0.4)')
    
    # Display options
    parser.add_argument('--no-text', action='store_true', 
                       help='Hide text content (text shown by default)')
    parser.add_argument('--no-analysis', action='store_true',
                       help='Disable enhanced LLM analysis and synthesis (enabled by default)')
    parser.add_argument('--dual-answer', action='store_true', default=True,
                       help='Enable dual answer mode: direct answer + detailed analysis (enabled by default)')
    parser.add_argument('--no-dual', action='store_true',
                       help='Disable dual answer mode and show only detailed analysis')
    parser.add_argument('--list', action='store_true', 
                       help='List available PDF collections and exit')
    parser.add_argument('--language', type=str, default=None,
                       help='Force response language (italian, spanish, french, english). Default: auto-detect')
    parser.add_argument('--no-enhancement', action='store_true',
                       help='Disable query enhancement (translation and expansion). Enhancement enabled by default')
    parser.add_argument('--rerank', action='store_true',
                       help='Enable LLM reranking for improved result quality (~2s with Gemini Flash 1.5)')
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
        # Determine enhancement setting
        use_enhancement = not args.no_enhancement
        enhancement_status = "enabled" if use_enhancement else "disabled"
        
        # Determine search method
        if args.bm25:
            rerank_suffix = " + reranking" if args.rerank else ""
            search_method = f"BM25 keyword search (enhancement {enhancement_status}{rerank_suffix})"
            # Only check embedding system if needed for analysis features
            if not args.no_analysis:
                check_embedding_system()
            similarities = query_fts5_collections(args.query, max(args.top_k, 20), args.pdf, use_enhancement=use_enhancement)
            
            # Apply reranking to BM25 results if requested
            if args.rerank and len(similarities) >= 8:
                try:
                    similarities = llm_rerank_results(args.query, similarities, language=args.language or "auto")
                    logging.info("Applied LLM reranking to BM25 results")
                except Exception as e:
                    logging.warning(f"BM25 reranking failed: {e}")
        elif args.semantic:
            rerank_suffix = " + reranking" if args.rerank else ""
            search_method = f"Semantic search{rerank_suffix}"
            check_embedding_system()
            query_embedding = generate_query_embedding(args.query)
            similarities = query_chroma_collections(query_embedding, max(args.top_k, 20), args.pdf)
            
            # Apply reranking to semantic results if requested
            if args.rerank and len(similarities) >= 8:
                try:
                    similarities = llm_rerank_results(args.query, similarities, language=args.language or "auto")
                    logging.info("Applied LLM reranking to semantic results")
                except Exception as e:
                    logging.warning(f"Semantic reranking failed: {e}")
        else:
            # Default to hybrid search
            rerank_suffix = " + reranking" if args.rerank else ""
            search_method = f"Hybrid search ({args.semantic_weight:.1f} semantic + {args.keyword_weight:.1f} keyword, enhancement {enhancement_status}{rerank_suffix})"
            check_embedding_system()
            similarities = hybrid_search(
                args.query, 
                max(args.top_k, 20), 
                args.pdf,
                args.semantic_weight,
                args.keyword_weight,
                use_enhancement=use_enhancement,
                use_reranking=args.rerank,
                language=args.language or "auto"
            )
        
        logging.info(f"Using {search_method}")
        
        if not similarities:
            logging.error("No results found. Make sure you've run ingest.py first.")
            sys.exit(1)
        
        # Display results (show text by default unless --no-text is specified)
        show_text = not args.no_text
        enhanced_analysis = not args.no_analysis
        dual_answer = args.dual_answer and not args.no_dual
        display_results(similarities, args.query, args.top_k, args.min_similarity, show_text, enhanced_analysis, dual_answer, force_language=args.language)
        
    except Exception as e:
        logging.error(f"Query failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()