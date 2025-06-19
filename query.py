#!/usr/bin/env python3
import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
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

def query_fts5_collections(query: str, top_k: int = 10, pdf_name: str = None, enhancement_mode: str = "full") -> List[Tuple[str, int, str, float]]:
    """
    Query SQLite FTS5 collections for keyword-based matches with optional query enhancement.
    Returns list of (pdf_name, page_number, text_content, bm25_score)
    """
    try:
        search_query = query
        
        # Try query enhancement for better keyword matching
        if enhancement_mode != "off":
            try:
                logging.info(f"Enhancing query with mode '{enhancement_mode}': '{query}'")
                enhancement = enhance_query_adaptive(query, enhancement_mode)
                enhanced_query = enhancement.get('enhanced_query', query)
                translation = enhancement.get('translation', query)
                synonyms = enhancement.get('synonyms', [])
                
                # Create expanded search query with translation and synonyms
                # Adjust expansion based on enhancement mode
                if enhancement_mode == "minimal" or (enhancement_mode == "auto" and classify_query_type(query) == "factual"):
                    # Minimal expansion: translation + 1-2 direct synonyms
                    search_terms = [translation] + synonyms[:1]
                    max_terms = 3
                elif enhancement_mode == "maximum" or (enhancement_mode == "auto" and classify_query_type(query) == "comparative"):
                    # Maximum expansion: translation + many synonyms + related terms
                    related_terms = enhancement.get('related_terms', [])
                    search_terms = [translation] + synonyms[:4] + related_terms[:2]
                    max_terms = 8
                else:
                    # Full/balanced expansion
                    search_terms = [translation] + synonyms[:3]
                    max_terms = 5
                
                # Clean terms and remove duplicates
                clean_terms = []
                for term in search_terms:
                    if term and term.strip() and term not in clean_terms:
                        # Split multi-word terms and use individual words
                        words = term.split()
                        clean_terms.extend([word for word in words if len(word) > 2])  # Skip very short words
                
                if clean_terms:
                    # Join terms with spaces - FTS5Manager will handle OR logic internally
                    search_query = ' '.join(clean_terms[:max_terms])
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
        
        logging.info(f"Found {len(results)} FTS5 keyword results (enhancement: {enhancement_mode})")
        
        # Convert to expected format: (pdf_name, page_number, text_content, score)
        formatted_results = []
        for pdf_name, page_number, content, score, metadata in results:
            formatted_results.append((pdf_name, page_number, content, score))
        
        return formatted_results
        
    except Exception as e:
        logging.error(f"Error querying FTS5 collections: {e}")
        return []

def apply_fusion_strategy(semantic_results: List[Tuple[str, int, str, float]], 
                         keyword_results: List[Tuple[str, int, str, float]], 
                         strategy: str, 
                         semantic_weight: float = 0.6, 
                         keyword_weight: float = 0.4,
                         query: str = "") -> List[Tuple[str, int, str, float]]:
    """
    Apply different fusion strategies to combine semantic and keyword search results.
    
    Args:
        semantic_results: Results from semantic search
        keyword_results: Results from keyword search  
        strategy: Fusion strategy ("weighted", "rrf", "comb_sum", "comb_mnz", "adaptive")
        semantic_weight: Weight for semantic results (used in weighted strategy)
        keyword_weight: Weight for keyword results (used in weighted strategy)
        query: Original query (used in adaptive strategy)
    
    Returns:
        Combined and sorted results
    """
    
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
    
    # Create unified result mapping
    all_results = {}  # (pdf_name, page_number, text_snippet) -> result_data
    
    # Add semantic results
    for i, (pdf_name, page_number, text, norm_score) in enumerate(semantic_normalized):
        key = (pdf_name, page_number, text[:50])
        all_results[key] = {
            'text': text,
            'semantic_score': norm_score,
            'semantic_rank': i + 1,
            'keyword_score': 0.0,
            'keyword_rank': len(keyword_normalized) + 1,  # Worst possible rank
            'in_both': False
        }
    
    # Add keyword results
    for i, (pdf_name, page_number, text, norm_score) in enumerate(keyword_normalized):
        key = (pdf_name, page_number, text[:50])
        if key in all_results:
            # Update existing entry
            all_results[key]['keyword_score'] = norm_score
            all_results[key]['keyword_rank'] = i + 1
            all_results[key]['in_both'] = True
        else:
            # Create new entry (keyword-only result)
            all_results[key] = {
                'text': text,
                'semantic_score': 0.0,
                'semantic_rank': len(semantic_normalized) + 1,  # Worst possible rank
                'keyword_score': norm_score,
                'keyword_rank': i + 1,
                'in_both': False
            }
    
    # Apply fusion strategy
    if strategy == "weighted":
        # Ensure weights sum to 1.0
        total_weight = semantic_weight + keyword_weight
        if total_weight > 0:
            semantic_weight = semantic_weight / total_weight
            keyword_weight = keyword_weight / total_weight
        
        for key, data in all_results.items():
            data['final_score'] = semantic_weight * data['semantic_score'] + keyword_weight * data['keyword_score']
            
    elif strategy == "rrf":
        # Reciprocal Rank Fusion (k=60 is standard)
        k = 60
        for key, data in all_results.items():
            rrf_score = (1 / (k + data['semantic_rank'])) + (1 / (k + data['keyword_rank']))
            data['final_score'] = rrf_score
            
    elif strategy == "comb_sum":
        # Simple sum of normalized scores
        for key, data in all_results.items():
            data['final_score'] = data['semantic_score'] + data['keyword_score']
            
    elif strategy == "comb_mnz":
        # CombMNZ: CombSUM * number of systems that returned the document
        for key, data in all_results.items():
            systems_count = (1 if data['semantic_score'] > 0 else 0) + (1 if data['keyword_score'] > 0 else 0)
            data['final_score'] = (data['semantic_score'] + data['keyword_score']) * systems_count
            
    elif strategy == "adaptive":
        # Query-adaptive weighting based on query characteristics
        def is_factual_query(q: str) -> bool:
            factual_indicators = ['what', 'who', 'when', 'where', 'define', 'definition', 'meaning']
            return any(indicator in q.lower() for indicator in factual_indicators)
        
        def is_conceptual_query(q: str) -> bool:
            conceptual_indicators = ['why', 'how', 'explain', 'describe', 'analysis', 'relationship', 'compare']
            return any(indicator in q.lower() for indicator in conceptual_indicators)
        
        if is_factual_query(query):
            # Favor keyword matching for factual queries
            adaptive_semantic_weight, adaptive_keyword_weight = 0.3, 0.7
        elif is_conceptual_query(query):
            # Favor semantic matching for conceptual queries
            adaptive_semantic_weight, adaptive_keyword_weight = 0.8, 0.2
        else:
            # Default balanced approach
            adaptive_semantic_weight, adaptive_keyword_weight = 0.6, 0.4
        
        for key, data in all_results.items():
            data['final_score'] = adaptive_semantic_weight * data['semantic_score'] + adaptive_keyword_weight * data['keyword_score']
    
    else:
        # Fallback to weighted if unknown strategy
        logging.warning(f"Unknown fusion strategy '{strategy}', falling back to weighted")
        for key, data in all_results.items():
            data['final_score'] = semantic_weight * data['semantic_score'] + keyword_weight * data['keyword_score']
    
    # Convert back to list format and sort by final score
    final_results = []
    for (pdf_name, page_number, _), data in all_results.items():
        final_results.append((pdf_name, page_number, data['text'], data['final_score']))
    
    # Sort by final score (highest first)
    final_results.sort(key=lambda x: x[3], reverse=True)
    
    logging.info(f"Fusion strategy '{strategy}': {len(semantic_results)} semantic → {len(semantic_deduped)} deduped")
    logging.info(f"Fusion strategy '{strategy}': {len(keyword_results)} keyword → {len(keyword_deduped)} deduped")
    logging.info(f"Final fused results: {len(final_results)} unique combinations")
    
    return final_results

def get_full_page_content(pdf_name: str, page_number: int) -> Optional[str]:
    """
    Retrieve complete page content by reconstructing all chunks for a given page.
    
    Args:
        pdf_name: Name of the PDF document
        page_number: Page number to retrieve
    
    Returns:
        Complete page text or None if not found
    """
    try:
        import chromadb
        client = chromadb.PersistentClient(path="./chroma_db")
        
        # Try to get collection (handle both with and without 'pdf_' prefix)
        collection_name = f"pdf_{pdf_name}"
        try:
            collection = client.get_collection(name=collection_name)
        except:
            # Try without prefix in case it was stored differently
            collection = client.get_collection(name=pdf_name)
        
        # Get all chunks for this page
        results = collection.get(
            include=['documents', 'metadatas'],
            where={"page_number": page_number}
        )
        
        if not results['documents']:
            return None
        
        # Sort chunks by chunk_id to maintain proper order
        chunks = []
        for doc, meta in zip(results['documents'], results['metadatas']):
            chunk_id = meta.get('chunk_id', 1)
            # Handle None chunk_id (single-chunk pages)
            if chunk_id is None:
                chunk_id = 1
            chunks.append((chunk_id, doc))
        
        # Sort by chunk_id and reconstruct full text
        chunks.sort(key=lambda x: x[0])
        full_text = ''.join([chunk[1] for chunk in chunks])
        
        return full_text.strip()
        
    except Exception as e:
        logging.warning(f"Failed to retrieve full page content for {pdf_name} page {page_number}: {e}")
        return None

def enrich_chunks_to_pages(chunk_results: List[Tuple[str, int, str, float]], 
                          include_previous_page: bool = False,
                          include_next_page: bool = False,
                          max_enriched_results: int = 50) -> List[Tuple[str, int, str, float]]:
    """
    Expand chunk results to full page content with optional adjacent page context.
    
    Args:
        chunk_results: List of (pdf_name, page_number, chunk_text, score)
        include_previous_page: Include previous page for additional context
        include_next_page: Include next page for additional context  
        max_enriched_results: Maximum number of results to enrich (performance control)
    
    Returns:
        List of (pdf_name, page_number, enriched_page_text, score) with full page content
    """
    if not chunk_results:
        return []
    
    logging.info(f"Enriching {min(len(chunk_results), max_enriched_results)} chunk results to full pages")
    if include_previous_page or include_next_page:
        context_info = []
        if include_previous_page:
            context_info.append("previous page")
        if include_next_page:
            context_info.append("next page")
        logging.info(f"Including {' + '.join(context_info)} for additional context")
    
    enriched_results = []
    page_cache = {}  # Cache to avoid retrieving same page multiple times
    
    # Limit processing for performance
    results_to_process = chunk_results[:max_enriched_results]
    
    for pdf_name, page_number, chunk_text, score in results_to_process:
        page_content_parts = []
        
        # Previous page (if requested and page > 1)
        if include_previous_page and page_number > 1:
            prev_page_key = (pdf_name, page_number - 1)
            if prev_page_key not in page_cache:
                page_cache[prev_page_key] = get_full_page_content(pdf_name, page_number - 1)
            
            prev_page_content = page_cache[prev_page_key]
            if prev_page_content:
                page_content_parts.append(f"[Previous Page {page_number-1}]\n{prev_page_content}\n")
        
        # Current page (always include)
        current_page_key = (pdf_name, page_number)
        if current_page_key not in page_cache:
            page_cache[current_page_key] = get_full_page_content(pdf_name, page_number)
        
        current_page_content = page_cache[current_page_key]
        if current_page_content:
            page_content_parts.append(f"[Page {page_number}]\n{current_page_content}")
        else:
            # Fallback to original chunk if full page retrieval fails
            page_content_parts.append(f"[Page {page_number} - Chunk Only]\n{chunk_text}")
        
        # Next page (if requested)
        if include_next_page:
            next_page_key = (pdf_name, page_number + 1)
            if next_page_key not in page_cache:
                page_cache[next_page_key] = get_full_page_content(pdf_name, page_number + 1)
            
            next_page_content = page_cache[next_page_key]
            if next_page_content:
                page_content_parts.append(f"\n[Next Page {page_number+1}]\n{next_page_content}")
        
        # Combine all page content
        enriched_text = "\n".join(page_content_parts)
        enriched_results.append((pdf_name, page_number, enriched_text, score))
        
        logging.debug(f"Enriched {pdf_name} page {page_number}: {len(chunk_text)} chars → {len(enriched_text)} chars")
    
    logging.info(f"Page enrichment completed: {len(enriched_results)} results with full page context")
    return enriched_results

def smart_deduplicate_pages(enriched_results: List[Tuple[str, int, str, float]]) -> List[Tuple[str, int, str, float]]:
    """
    Deduplicate results that refer to the same page after enrichment.
    Keeps the result with the highest score for each unique (pdf_name, page_number) combination.
    
    Args:
        enriched_results: List of (pdf_name, page_number, enriched_text, score)
    
    Returns:
        Deduplicated list with best score per page
    """
    if not enriched_results:
        return []
    
    page_map = {}  # (pdf_name, page_number) -> best_result
    
    for result in enriched_results:
        pdf_name, page_number, text, score = result
        key = (pdf_name, page_number)
        
        if key not in page_map or score > page_map[key][3]:
            page_map[key] = result
    
    # Convert back to list and sort by score (highest first)
    deduplicated_results = list(page_map.values())
    deduplicated_results.sort(key=lambda x: x[3], reverse=True)
    
    logging.info(f"Smart deduplication: {len(enriched_results)} → {len(deduplicated_results)} unique pages")
    return deduplicated_results

def hybrid_search(query: str, top_k: int = 10, pdf_name: str = None, 
                 semantic_weight: float = 0.6, bm25_weight: float = 0.4, enhancement_mode: str = "full", use_reranking: bool = False, language: str = "auto", fusion_strategy: str = "weighted", use_page_enrichment: bool = False, include_previous_page: bool = False, include_next_page: bool = False) -> List[Tuple[str, int, str, float]]:
    """
    Perform hybrid search combining semantic similarity (ChromaDB) and keyword matching (FTS5).
    
    Args:
        query: Search query
        top_k: Number of results to return
        pdf_name: Optional specific PDF to search
        semantic_weight: Weight for semantic search results (default: 0.6)
        bm25_weight: Weight for BM25 keyword search results (default: 0.4)
        enhancement_mode: Enhancement mode ("auto", "minimal", "full", "maximum", "off") (default: "full")
        use_reranking: Whether to apply LLM reranking to results (default: False)
        language: Language hint for LLM reranking (default: "auto")
        fusion_strategy: Strategy for combining results ("weighted", "rrf", "comb_sum", "comb_mnz", "adaptive") (default: "weighted")
        use_page_enrichment: Whether to expand chunks to full pages (default: False)
        include_previous_page: Include previous page for additional context (default: False)
        include_next_page: Include next page for additional context (default: False)
    
    Returns:
        List of (pdf_name, page_number, text_content, combined_score) sorted by combined score
    """
    try:
        # Ensure weights sum to 1.0
        total_weight = semantic_weight + bm25_weight
        if total_weight > 0:
            semantic_weight = semantic_weight / total_weight
            bm25_weight = bm25_weight / total_weight
        
        logging.info(f"Performing hybrid search with fusion strategy: {fusion_strategy}")
        if fusion_strategy == "weighted":
            logging.info(f"Using {semantic_weight:.2f} semantic + {bm25_weight:.2f} keyword weighting")
        
        # Get semantic search results
        query_embedding = generate_query_embedding(query)
        semantic_results = query_chroma_collections(query_embedding, top_k * 2, pdf_name)
        
        # Get keyword search results with enhancement
        keyword_results = query_fts5_collections(query, top_k * 2, pdf_name, enhancement_mode=enhancement_mode)
        
        # Apply fusion strategy to combine results
        hybrid_results = apply_fusion_strategy(
            semantic_results=semantic_results,
            keyword_results=keyword_results,
            strategy=fusion_strategy,
            semantic_weight=semantic_weight,
            keyword_weight=bm25_weight,
            query=query
        )
        
        if hybrid_results and logging.getLogger().isEnabledFor(logging.DEBUG):
            logging.debug("Top 3 hybrid scores:")
            for i, (pdf, page, text, score) in enumerate(hybrid_results[:3]):
                logging.debug(f"  {i+1}. {pdf} p.{page}: {score:.4f} - {text[:50]}...")
        
        # Page enrichment pipeline (expand chunks to full pages)
        if use_page_enrichment:
            try:
                # Step 4: Enrich chunks to full pages (with optional adjacent pages)
                logging.info("Starting page enrichment pipeline")
                enriched_results = enrich_chunks_to_pages(
                    chunk_results=hybrid_results,
                    include_previous_page=include_previous_page,
                    include_next_page=include_next_page,
                    max_enriched_results=50  # Limit for performance
                )
                
                # Step 5: Smart deduplication (remove duplicate pages, keep best score)
                final_results = smart_deduplicate_pages(enriched_results)
                
                # Use enriched results for reranking and final output
                results_for_processing = final_results
                logging.info(f"Page enrichment pipeline completed: {len(hybrid_results)} chunks → {len(final_results)} enriched pages")
            except Exception as e:
                logging.warning(f"Page enrichment failed, using original chunk results: {e}")
                results_for_processing = hybrid_results
        else:
            results_for_processing = hybrid_results
        
        # Step 6: Apply LLM reranking if requested and we have enough candidates
        if use_reranking and len(results_for_processing) >= 8:
            try:
                context_info = " with enriched pages" if use_page_enrichment else " with chunks"
                logging.info(f"Applying LLM reranking to {len(results_for_processing)} hybrid results{context_info}")
                reranked_results = llm_rerank_results(
                    query=query,
                    candidates=results_for_processing[:25],  # Limit to 25 for cost control
                    language=language
                )
                logging.info("LLM reranking completed successfully")
                return reranked_results[:top_k]
            except Exception as e:
                logging.warning(f"LLM reranking failed, using fusion scores: {e}")
        
        return results_for_processing[:top_k]
        
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

def classify_query_type(query: str) -> str:
    """
    Classify query type for adaptive enhancement.
    
    Args:
        query: The search query text
    
    Returns:
        Query type: "factual", "conceptual", "comparative", or "default"
    """
    query_lower = query.lower()
    
    # Factual patterns (need minimal enhancement - direct answers)
    factual_patterns = [
        r'\bquanto\b', r'\bquando\b', r'\bdove\b', r'\bchi\b', r'\bche\s+distanza\b',
        r'\bche\s+dimensioni?\b', r'\bche\s+grandezza\b', r'\bche\s+misura\b',
        r'\bhow\s+big\b', r'\bhow\s+far\b', r'\bhow\s+many\b', r'\bwhat\s+is\s+the\s+size\b',
        r'\bwhen\s+did\b', r'\bwhere\s+is\b', r'\bwho\s+was\b', r'\bwhat\s+year\b'
    ]
    
    # Comparative patterns (need maximum enhancement - relationships and contrasts)
    comparative_patterns = [
        r'\bdifferenza\b', r'\bconfronto\b', r'\brelazione\b', r'\bparagonare\b',
        r'\bversus\b', r'\bvs\b', r'\bdifference\b', r'\bcompare\b', r'\bcompared\s+to\b',
        r'\bbetween\b.*\band\b', r'\brather\s+than\b', r'\binstead\s+of\b'
    ]
    
    # Conceptual patterns (need full enhancement - complex explanations)
    conceptual_patterns = [
        r'\bcos[\'è]\b', r'\bcome\b', r'\bperch[eé]\b', r'\bspiegare?\b',
        r'\bwhat\s+is\b', r'\bhow\s+does\b', r'\bwhy\s+does\b', r'\bexplain\b',
        r'\bdescribe\b', r'\btell\s+me\s+about\b', r'\bwhat\s+are\b'
    ]
    
    # Check patterns in order of specificity
    if any(re.search(pattern, query_lower) for pattern in factual_patterns):
        return "factual"
    elif any(re.search(pattern, query_lower) for pattern in comparative_patterns):
        return "comparative"
    elif any(re.search(pattern, query_lower) for pattern in conceptual_patterns):
        return "conceptual"
    
    return "default"

def enhance_query_adaptive(original_query: str, enhancement_mode: str = "full") -> dict:
    """
    Enhance query with adaptive expansion based on query type or explicit mode.
    
    Args:
        original_query: The original search query
        enhancement_mode: "auto", "minimal", "full", "maximum", or "off"
    
    Returns:
        dict with enhanced_query, translations, synonyms, and related_terms
    """
    if enhancement_mode == "off":
        return {
            "original_query": original_query,
            "detected_language": "unknown",
            "translation": original_query,
            "enhanced_query": original_query,
            "synonyms": [],
            "related_terms": [],
            "search_strategy": "enhancement disabled"
        }
    
    # Determine enhancement level
    if enhancement_mode == "auto":
        query_type = classify_query_type(original_query)
        if query_type == "factual":
            enhancement_level = "minimal"
        elif query_type == "comparative":
            enhancement_level = "maximum"
        elif query_type == "conceptual":
            enhancement_level = "full"
        else:
            enhancement_level = "full"  # default
        
        logging.info(f"Auto-detected query type: '{query_type}' → using '{enhancement_level}' enhancement")
    else:
        enhancement_level = enhancement_mode
        logging.info(f"Using explicit enhancement level: '{enhancement_level}'")
    
    # Create enhancement instructions based on level
    if enhancement_level == "minimal":
        enhancement_instruction = """Focus on MINIMAL enhancement:
- Translate to English if needed
- Add only direct synonyms (1-2 terms max)
- Avoid expanding scope or adding conceptual terms
- Keep the query focused and precise"""
        
    elif enhancement_level == "maximum":
        enhancement_instruction = """Focus on MAXIMUM enhancement:
- Translate to English if needed
- Add extensive synonyms and alternative phrasings
- Include related concepts and comparative terms
- Add domain-specific terminology
- Expand to capture relationships and contrasts"""
        
    else:  # "full" or default
        enhancement_instruction = """Focus on BALANCED enhancement:
- Translate to English if needed
- Add relevant synonyms and related terms
- Include conceptually related vocabulary
- Maintain focus while expanding search potential"""
    
    return enhance_query(original_query, enhancement_instruction)

def enhance_query(original_query: str, enhancement_instruction: str = None) -> dict:
    """
    Enhance and translate query using LLM for better search performance.
    
    Args:
        original_query: The original search query
        enhancement_instruction: Specific instructions for enhancement level
    
    Returns:
        dict with enhanced_query, translations, synonyms, and related_terms
    """
    
    # Default enhancement instruction if none provided
    if enhancement_instruction is None:
        enhancement_instruction = """Focus on BALANCED enhancement:
- Translate to English if needed
- Add relevant synonyms and related terms
- Include conceptually related vocabulary
- Maintain focus while expanding search potential"""
    
    messages = [
        {
            "role": "user",
            "content": f"""You are a search query enhancement expert. Your task is to improve this search query for better document retrieval.

Original Query: "{original_query}"

Enhancement Level Instructions:
{enhancement_instruction}

Please provide:
1. TRANSLATION: If the query is not in English, translate it to English
2. ENHANCED_QUERY: Improve the query according to the enhancement level instructions
3. SYNONYMS: List alternative terms and synonyms (adjust quantity based on enhancement level)
4. RELATED_TERMS: Add conceptually related terms (adjust scope based on enhancement level)

Format your response as JSON:
{{
    "original_query": "{original_query}",
    "detected_language": "language_name",
    "translation": "english translation if needed",
    "enhanced_query": "improved searchable version with appropriate expansion",
    "synonyms": ["synonym1", "synonym2", "synonym3"],
    "related_terms": ["related1", "related2", "related3"],
    "search_strategy": "brief explanation of enhancement approach"
}}

Examples by enhancement level:
MINIMAL: "Quanto è grande il Sole?" → "How big Sun? size"
FULL: "Cos'è una supernova?" → "What supernova? stellar explosion massive star core collapse"
MAXIMUM: "Differenza tra pianeta e stella" → "Difference planet star? celestial bodies stellar objects planetary formation stellar evolution mass composition"

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
    parser.add_argument('--fusion-strategy', type=str, default='weighted',
                       choices=['weighted', 'rrf', 'comb_sum', 'comb_mnz', 'adaptive'],
                       help='Fusion strategy for combining semantic and keyword results (default: weighted)')
    
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
    # Enhancement options (mutually exclusive group)
    enhancement_group = parser.add_mutually_exclusive_group()
    enhancement_group.add_argument('--enhancement', type=str, default='auto',
                       choices=['auto', 'minimal', 'full', 'maximum', 'off'],
                       help='Enhancement mode: auto (adaptive based on query type), minimal (factual queries), full (balanced), maximum (comparative queries), off (disabled). Default: auto')
    enhancement_group.add_argument('--no-enhancement', action='store_true',
                       help='Disable query enhancement (equivalent to --enhancement=off)')
    parser.add_argument('--rerank', action='store_true',
                       help='Enable LLM reranking for improved result quality (~2s with Gemini Flash 1.5)')
    
    # Page enrichment options
    parser.add_argument('--enrich-pages', action='store_true',
                       help='Expand chunk results to full page content for better context and reranking')
    parser.add_argument('--include-previous-page', action='store_true',
                       help='Include previous page content for additional narrative context (requires --enrich-pages)')
    parser.add_argument('--include-next-page', action='store_true',
                       help='Include next page content for additional narrative context (requires --enrich-pages)')
    
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
        # Determine enhancement mode
        if args.no_enhancement:
            enhancement_mode = "off"
        else:
            enhancement_mode = args.enhancement
        
        enhancement_status = f"mode: {enhancement_mode}"
        
        # Determine search method
        if args.bm25:
            rerank_suffix = " + reranking" if args.rerank else ""
            search_method = f"BM25 keyword search (enhancement {enhancement_status}{rerank_suffix})"
            # Only check embedding system if needed for analysis features
            if not args.no_analysis:
                check_embedding_system()
            similarities = query_fts5_collections(args.query, max(args.top_k, 20), args.pdf, enhancement_mode=enhancement_mode)
            
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
            page_enrichment_info = ""
            if args.enrich_pages:
                page_enrichment_info = " + page enrichment"
                if args.include_previous_page or args.include_next_page:
                    adjacent_pages = []
                    if args.include_previous_page:
                        adjacent_pages.append("prev")
                    if args.include_next_page:
                        adjacent_pages.append("next")
                    page_enrichment_info += f" ({'+'.join(adjacent_pages)})"
            
            search_method = f"Hybrid search ({args.fusion_strategy}, {args.semantic_weight:.1f} semantic + {args.keyword_weight:.1f} keyword, enhancement {enhancement_status}{rerank_suffix}{page_enrichment_info})"
            check_embedding_system()
            similarities = hybrid_search(
                args.query, 
                max(args.top_k, 20), 
                args.pdf,
                args.semantic_weight,
                args.keyword_weight,
                enhancement_mode=enhancement_mode,
                use_reranking=args.rerank,
                language=args.language or "auto",
                fusion_strategy=args.fusion_strategy,
                use_page_enrichment=args.enrich_pages,
                include_previous_page=args.include_previous_page,
                include_next_page=args.include_next_page
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