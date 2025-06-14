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

def highlight_relevant_text_batch(query: str, results: List[Tuple[str, int, str, float]], output_format_ansi: bool = True) -> List[str]:
    """Use LLM to batch highlight relevant parts of multiple texts and explain relevance"""
    if not results:
        return []
    
    # Detect language from first result
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
        return [highlight_relevant_text(query, text_content, output_format_ansi=output_format_ansi) for _, _, text_content, _ in results]
    
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
        highlighted_texts.append(highlight_relevant_text(query, results[idx][2], output_format_ansi=output_format_ansi))
    
    return highlighted_texts[:len(results)]

def highlight_relevant_text(query: str, page_text: str, output_format_ansi: bool = True) -> str:
    """Use LLM to find and highlight relevant parts of the text and explain relevance (single text version)"""
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

def analyze_individual_result(query: str, result_text: str, pdf_name: str, page_number: int, similarity: float) -> str:
    """Use LLM to analyze and comment on individual search result relevance"""
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

def synthesize_results(query: str, results: List[Tuple[str, int, str, float]], max_results: int = 5) -> str:
    """Use LLM to synthesize and conglomerate multiple search results"""
    if not results:
        return "No results to synthesize."
    
    # Take top results for synthesis
    top_results = results[:max_results]
    
    # Detect language from first result
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

def generate_direct_answer(query: str, results: List[Tuple[str, int, str, float]], max_results: int = 5) -> str:
    """Generate a comprehensive, detailed answer to the query based on search results with source attribution"""
    if not results:
        return "No information found to answer the query."
    
    # Take top results for answer generation
    top_results = results[:max_results]
    
    # Detect language from first result
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
                   dual_answer: bool = False):
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
            direct_answer = generate_direct_answer(query, displayed_results)
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
            highlighted_texts = highlight_relevant_text_batch(query, displayed_results, output_format_ansi=True)
        except Exception as e:
            print(f"Error in batch highlighting, falling back to individual processing: {e}")
            # Fallback to individual highlighting
            highlighted_texts = [highlight_relevant_text(query, text_content, output_format_ansi=True) for _, _, text_content, _ in displayed_results]
    
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
                    
                    individual_analysis = analyze_individual_result(query, text_content, pdf_name, page_number, similarity)
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
            synthesis = synthesize_results(query, displayed_results)
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
    parser = argparse.ArgumentParser(description='Query PDF pages using semantic similarity')
    parser.add_argument('query', nargs='?', help='Query text to search for')
    parser.add_argument('--pdf', '-p', help='Name of specific PDF to search (optional, searches all by default)')
    parser.add_argument('-k', '--top-k', type=int, default=3, help='Number of top results to show (default: 3)')
    parser.add_argument('-s', '--min-similarity', type=float, default=0.0, 
                       help='Minimum similarity threshold (default: 0.0)')
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
        enhanced_analysis = not args.no_analysis
        dual_answer = args.dual_answer and not args.no_dual
        display_results(similarities, args.query, args.top_k, args.min_similarity, show_text, enhanced_analysis, dual_answer)
        
    except Exception as e:
        logging.error(f"Query failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()