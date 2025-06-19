#!/usr/bin/env python3
"""
Gradio-based web interface for LLM RAG with advanced text highlighting
"""
import gradio as gr
import os
import sys
from typing import List, Tuple, Dict, Any
import json
import re
import markdown # For Markdown in direct answer/synthesis if needed
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from query import (
    list_available_collections, query_chroma_collections, query_fts5_collections, hybrid_search,
    highlight_relevant_text_batch, detect_language, generate_direct_answer, generate_query_embedding,
    enhance_query_adaptive, # Explicitly using the adaptive version
    calculate_confidence_score # Added for confidence calculation
)
from llm_wrapper import llm_call, generate_embeddings, test_openai_embeddings
from llm_reranker import llm_rerank_results

def create_web_border(title: str, style_class: str = "border-default") -> str:
    """Create a web-friendly bordered section with title"""
    return f'<div class="{style_class}"><div class="border-title">{title}</div><div class="border-content">'

def close_web_border() -> str:
    """Close a web bordered section"""
    return '</div></div>'

def process_highlighted_text_for_html(highlighted_text: str) -> Tuple[str, List[str]]:
    """Convert query.py style highlighting to HTML and separate explanations"""
    clean_text = highlighted_text
    # Convert [[explanation]] to <em class="explanation-text">explanation</em>
    clean_text = re.sub(r'\[\[(.*?)]]', r'<em class="explanation-text">\1</em>', clean_text)
    # Convert [HIGHLIGHT] and [/HIGHLIGHT] to spans
    clean_text = clean_text.replace('[HIGHLIGHT]', '<span class="highlight-text">')
    clean_text = clean_text.replace('[/HIGHLIGHT]', '</span>')
    return clean_text, []


def analyze_individual_result_web(query: str, text_content: str, pdf_name: str, page_number: int, similarity: float, force_language: str = None) -> str:
    """Generate individual result analysis for web display (from query.py)"""
    if force_language:
        detected_language = force_language.lower()
    else:
        detected_language = detect_language(text_content)

    if detected_language == "italian":
        language_instruction = "Rispondi in italiano."
    elif detected_language == "spanish":
        language_instruction = "Responde en español."
    elif detected_language == "french":
        language_instruction = "Répondez en français."
    else:
        language_instruction = "Respond in English."

    messages = [
        {
            "role": "user",
            "content": f"""Query: "{query}"
Source: {pdf_name}, Page {page_number}
Similarity Score: {similarity:.4f}
Content: {text_content[:500]}{'...' if len(text_content) > 500 else ''}

Analyze this search result:
1. How relevant is this result to the query?
2. What key concepts connect this result to the search?
3. What value does this result provide for someone researching this topic?
{language_instruction}
Provide a concise analysis (2-3 sentences) focusing on relevance and research value."""
        }
    ]
    analysis, success = llm_call(messages, max_tokens=300)
    if success and analysis.strip():
        return analysis.strip()
    return "Analysis not available for this result."


def synthesize_results_web(query: str, results: List[Tuple[str, int, str, float]], force_language: str = None) -> str:
    """Generate comprehensive synthesis for web display (from query.py)"""
    if not results:
        return "No results to synthesize."

    if force_language:
        detected_language = force_language.lower()
    else:
        detected_language = detect_language(results[0][2]) # Detect from first result

    if detected_language == "italian":
        language_instruction = "Rispondi in italiano con un'analisi strutturata."
    elif detected_language == "spanish":
        language_instruction = "Responde en español con un análisis estructurado."
    elif detected_language == "french":
        language_instruction = "Répondez en français avec une analyse structurée."
    else:
        language_instruction = "Respond in English with a structured analysis."

    results_context = ""
    for i, (pdf_name, page_num, text, similarity) in enumerate(results, 1):
        results_context += f"\n--- Result {i} (Similarity: {similarity:.4f}) ---\n"
        results_context += f"Source: {pdf_name}, Page {page_num}\n"
        results_context += f"Content: {text[:400]}{'...' if len(text) > 400 else ''}\n"

    messages = [
        {
            "role": "user",
            "content": f"""Query: "{query}"
Here are the top {len(results)} search results:
{results_context}

Please provide a comprehensive synthesis that:
1. **Overall Assessment**: How well do these results address the query?
2. **Key Themes**: What main topics, concepts, or patterns emerge across results?
3. **Cross-References**: How do the different results complement or contradict each other?
4. **Research Value**: What insights can be drawn from viewing these results together?
5. **Gaps & Limitations**: What aspects of the query might need additional sources?
{language_instruction}
Use [[...]] to highlight any inferred information or assumptions, and format them as italic text inline."""
        }
    ]
    synthesis, success = llm_call(messages, max_tokens=800)
    if success and synthesis.strip():
        return synthesis.strip()
    return "Synthesis unavailable for these results."

def process_highlighting_for_gradio(query: str, results: List[Tuple[str, int, str, float]], force_language: str = None) -> List[Tuple[str, List[str]]]:
    """Use query.py's highlighting and convert to HTML for gradio"""
    if not results:
        return []
    try:
        highlighted_texts = highlight_relevant_text_batch(query, results, output_format_ansi=False, force_language=force_language)
    except Exception as e:
        # Fallback to unhighlighted text if batch highlighting fails
        highlighted_texts = [text_content for _, _, text_content, _ in results]

    processed_results = []
    for highlighted_text in highlighted_texts:
        html_text, explanations = process_highlighted_text_for_html(highlighted_text)
        processed_results.append((html_text, explanations))
    return processed_results

def create_custom_css() -> str:
    """Create CSS with clean, readable color scheme using white backgrounds with blue/green text"""
    return """
    <style>
    /* Light theme defaults with white backgrounds and blue/green text */
    .results-container {
        background: #ffffff; /* White background */
        color: #2c3e50; /* Dark blue-gray text */
        border: 2px solid #bdc3c7; /* Light gray border */
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .direct-answer-container {
        background: #f8f9fa; /* Very light gray background */
        border: 2px solid #28a745; /* Green border for emphasis */
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(40, 167, 69, 0.2); /* Greenish shadow */
    }
    .direct-answer-container h3 {
        margin-top: 0;
        margin-bottom: 1rem;
        color: #196f3d; /* Darker green for title */
        border-bottom: 1px solid #a9dfbf; /* Lighter green separator */
        padding-bottom: 0.5rem;
    }

    /* Confidence Score Display */
    .confidence-display-container {
        margin-bottom: 1rem;
        padding: 0.75rem 1rem; /* Adjusted padding */
        border-radius: 6px;
        border: 1px solid; /* color set by specific class */
        font-size: 0.95em; /* Slightly smaller font */
        text-align: center; /* Center the confidence text */
    }
    .confidence-low {
        border-color: #e74c3c; /* Red border */
        background-color: #fdedec; /* Light red background */
        color: #c0392b; /* Darker red text */
    }
    .confidence-medium {
        border-color: #f39c12; /* Orange border */
        background-color: #fef5e7; /* Light orange background */
        color: #d35400; /* Darker orange text */
    }
    .confidence-high {
        border-color: #2ecc71; /* Green border */
        background-color: #eafaf1; /* Light green background */
        color: #27ae60; /* Darker green text */
    }
    .confidence-display-container strong {
        font-weight: 600; /* Medium bold */
    }


    .border-default {
        background: #ffffff;
        border: 2px solid #3498db; /* Blue border */
        border-radius: 8px;
        margin: 1rem 0;
        overflow: hidden;
    }
    .border-content { /* Content within a default border */
        background: #fdfefe; /* Slightly off-white */
        border: 2px solid #aed6f1; /* Lighter blue */
        border-radius: 8px;
        margin: 1rem 0; /* Margin inside parent border */
        padding: 1rem; /* Padding for content */
        overflow: auto; /* Changed from hidden to auto */
    }
    .border-analysis { /* Analysis sections */
        background: #f5f5f5; /* Light gray for analysis */
        border: 2px solid #5dade2; /* Medium blue */
        border-radius: 8px;
        margin: 1rem 0;
        padding: 1rem;
        overflow: auto;
    }
    .border-synthesis { /* Synthesis section */
        background: #e8f8f5; /* Light teal/green */
        border: 2px solid #1abc9c; /* Teal/green border */
        border-radius: 8px;
        margin: 1rem 0;
        padding: 1rem;
        overflow: auto;
    }

    .border-title {
        background: #3498db; /* Blue background for default title */
        color: #ffffff; /* White text */
        padding: 0.75rem 1rem;
        font-weight: 600;
        font-size: 1.1em;
        border-bottom: 2px solid #217dbb; /* Darker blue bottom border */
    }
    .border-content .border-title { /* Title for content within default */
        background: #aed6f1; /* Lighter blue */
        color: #1b4f72; /* Dark blue text */
        border-bottom: 2px solid #85c1e9; /* Medium light blue */
    }
    .border-analysis .border-title {
        background: #5dade2; /* Medium blue for analysis title */
        color: #ffffff;
        border-bottom: 2px solid #3498db; /* Blue bottom border */
    }
    .border-synthesis .border-title {
        background: #1abc9c; /* Teal/green for synthesis title */
        color: #ffffff;
        border-bottom: 2px solid #16a085; /* Darker teal/green */
    }

    .highlight-text {
        background-color: #f1c40f; /* Yellow background for highlight */
        color: #333333; /* Dark gray text for contrast */
        padding: 2px 4px;
        border-radius: 3px;
        font-weight: 700;
        border: 1px solid #f39c12; /* Orange border */
    }
    .explanation-text {
        color: #2980b9; /* Blue for explanations */
        font-style: italic;
    }
    .logs-container {
        border: 2px dashed #7f8c8d; /* Dashed gray border */
        padding: 1rem;
        margin: 1rem 0;
        background: #ecf0f1; /* Very light gray background */
        font-family: monospace;
        white-space: pre-wrap;
        color: #34495e; /* Dark blue-gray text */
    }

    /* Dark theme overrides */
    @media (prefers-color-scheme: dark) {
        .results-container {
            background: #2c3e50; /* Dark blue-gray */
            color: #ecf0f1; /* Light gray text */
            border: 2px solid #34495e; /* Slightly lighter border */
        }
        .border-default {
            background: #34495e; /* Darker blue-gray */
            border: 2px solid #2980b9; /* Blue border */
        }
        .border-content {
            background: #2c3e50; /* Dark blue-gray */
            border: 2px solid #5dade2; /* Lighter blue */
            padding: 1rem;
        }
        .border-analysis {
            background: #283747; /* Very dark blue */
            border: 2px solid #5dade2; /* Medium blue */
            padding: 1rem;
        }
        .border-synthesis {
            background: #1f6153; /* Dark teal */
            border: 2px solid #1abc9c; /* Teal/green */
            padding: 1rem;
        }
        .border-title {
            background: #2980b9; /* Blue */
            color: #ffffff;
            border-bottom: 2px solid #1f618d; /* Darker blue */
        }
        .border-content .border-title {
            background: #5dade2; /* Lighter blue */
            color: #f0f0f0; /* Light text */
            border-bottom: 2px solid #3498db;
        }
        .border-analysis .border-title {
            background: #5dade2; /* Medium blue */
            color: #ffffff;
            border-bottom: 2px solid #2980b9;
        }
        .border-synthesis .border-title {
            background: #1abc9c; /* Teal/green */
            color: #ffffff;
            border-bottom: 2px solid #117a65; /* Darker teal */
        }

        .direct-answer-container {
            background: #212f3c; /* Darker blue */
            border: 2px solid #2ecc71; /* Bright green border */
            box-shadow: 0 2px 8px rgba(46, 204, 113, 0.3); /* Greenish shadow */
        }
        .direct-answer-container h3 {
            color: #abebc6; /* Light green title */
            border-bottom: 1px solid #58d68d; /* Medium green separator */
        }

        /* Dark mode confidence */
        .confidence-low {
            border-color: #c0392b; 
            background-color: #572323; 
            color: #f5b7b1; 
        }
        .confidence-medium {
            border-color: #d35400;
            background-color: #5f3b0d;
            color: #fdebd0;
        }
        .confidence-high {
            border-color: #27ae60;
            background-color: #145a32;
            color: #d1f2eb;
        }

        .highlight-text {
            background-color: #f1c40f; /* Yellow, same as light */
            color: #1c1c1c; /* Very dark gray for contrast */
            border: 1px solid #b8860b; /* Darker yellow/brown border */
        }
        .explanation-text {
            color: #5dade2; /* Lighter blue for explanations */
            font-style: italic;
        }
        .logs-container {
            border: 2px dashed #566573; /* Darker dashed gray */
            background: #1c2833; /* Very dark blue */
            color: #bdc3c7; /* Light gray text */
        }
    }
    </style>
    """

def search_and_highlight(query: str, collection_names: str = "all", language_selection: str = "Auto-detect", search_method: str = "Hybrid", fusion_strategy: str = "weighted", semantic_weight: float = 0.6, keyword_weight: float = 0.4, use_reranking: bool = False, use_page_enrichment: bool = False, include_previous_page: bool = False, include_next_page: bool = False, enhancement_mode: str = "auto"):
    """
    Main search function as a generator.
    Yields a tuple: (direct_answer_md, html_results, synthesis_md, logs_html)
    """
    if language_selection == "Auto-detect":
        force_language = None
    else:
        force_language = language_selection.lower()

    logs = []
    direct_answer_md = ""
    html_results = ""
    synthesis_md = ""
    confidence_score = None # Initialize confidence score

    def get_logs_html():
        return f'<div class="logs-container"><div class="border-title">📝 Logs</div><pre>{"\n".join(logs)}</pre></div>'

    logs.append(f"[search_and_highlight] Received query: '{query}' | collections: '{collection_names}'")
    yield (direct_answer_md, html_results, synthesis_md, get_logs_html())

    if not query.strip():
        logs.append("[search_and_highlight] No query provided.")
        yield (direct_answer_md, html_results, synthesis_md, get_logs_html())
        return

    try:
        logs.append("[search_and_highlight] Testing OpenAI embeddings...")
        yield (direct_answer_md, html_results, synthesis_md, get_logs_html())
        if not test_openai_embeddings():
            logs.append("[search_and_highlight] OpenAI embedding API not available.")
            yield (direct_answer_md, html_results, synthesis_md, get_logs_html())
            return

        logs.append("[search_and_highlight] Listing available collections...")
        yield (direct_answer_md, html_results, synthesis_md, get_logs_html())
        collections = list_available_collections()
        if not collections:
            logs.append("[search_and_highlight] No collections found.")
            yield (direct_answer_md, html_results, synthesis_md, get_logs_html())
            return

        available_book_names = [book_name for book_name, _ in collections]
        if collection_names.lower() == "all":
            selected_pdf_name = None
            logs.append("[search_and_highlight] Using all collections.")
        else:
            selected_names = [name.strip() for name in collection_names.split(",")]
            valid_names = [name for name in selected_names if name in available_book_names]
            selected_pdf_name = valid_names[0] if valid_names else None # Takes the first valid name
            logs.append(f"[search_and_highlight] Selected collection: {selected_pdf_name}")
        yield (direct_answer_md, html_results, synthesis_md, get_logs_html())

        logs.append(f"[search_and_highlight] Using {search_method} search...")
        yield (direct_answer_md, html_results, synthesis_md, get_logs_html())

        all_results = [] # Initialize all_results

        if search_method == "BM25":
            rerank_suffix = " + reranking" if use_reranking else ""
            logs.append(f"[search_and_highlight] Performing BM25 keyword search with {enhancement_mode} enhancement{rerank_suffix}...")
            yield (direct_answer_md, html_results, synthesis_md, get_logs_html())
            all_results = query_fts5_collections(query, top_k=10, pdf_name=selected_pdf_name, enhancement_mode=enhancement_mode)
            if use_reranking and len(all_results) >= 8:
                try:
                    logs.append("[search_and_highlight] Applying LLM reranking to BM25 results...")
                    yield (direct_answer_md, html_results, synthesis_md, get_logs_html())
                    all_results = llm_rerank_results(query, all_results, language=force_language or "auto")
                    logs.append("[search_and_highlight] LLM reranking completed successfully")
                except Exception as e:
                    logs.append(f"[search_and_highlight] BM25 reranking failed: {e}")
                yield (direct_answer_md, html_results, synthesis_md, get_logs_html())

            # Calculate confidence for BM25
            enhancement_data_bm25 = {}
            if enhancement_mode != "off":
                try:
                    enhancement_data_bm25 = enhance_query_adaptive(query, enhancement_mode)
                except Exception as e:
                    logs.append(f"[search_and_highlight] BM25 - Error getting enhancement data for confidence: {e}")
                    enhancement_data_bm25 = {'detected_language': 'unknown', 'synonyms': [], 'related_terms': [], 'search_strategy': 'fallback - error'}
            else: # enhancement_mode == "off"
                enhancement_data_bm25 = {'detected_language': 'unknown', 'synonyms': [], 'related_terms': [], 'search_strategy': 'enhancement_off'}
            confidence_score = calculate_confidence_score(query, enhancement_data_bm25, all_results, reranked=use_reranking)
            logs.append(f"[search_and_highlight] BM25 confidence: {confidence_score:.2f}")


        elif search_method == "Semantic":
            rerank_suffix = " + reranking" if use_reranking else ""
            logs.append(f"[search_and_highlight] Generating query embedding for semantic search{rerank_suffix}...")
            yield (direct_answer_md, html_results, synthesis_md, get_logs_html())
            query_embedding = generate_query_embedding(query)
            logs.append("[search_and_highlight] Performing semantic search...")
            yield (direct_answer_md, html_results, synthesis_md, get_logs_html())
            all_results = query_chroma_collections(query_embedding, top_k=10, pdf_name=selected_pdf_name)
            if use_reranking and len(all_results) >= 8:
                try:
                    logs.append("[search_and_highlight] Applying LLM reranking to semantic results...")
                    yield (direct_answer_md, html_results, synthesis_md, get_logs_html())
                    all_results = llm_rerank_results(query, all_results, language=force_language or "auto")
                    logs.append("[search_and_highlight] LLM reranking completed successfully")
                except Exception as e:
                    logs.append(f"[search_and_highlight] Semantic reranking failed: {e}")
                yield (direct_answer_md, html_results, synthesis_md, get_logs_html())

            # Calculate confidence for Semantic
            # For pure semantic, enhancement_data is mostly for language context if query isn't English.
            # Using a simplified enhancement_data for this path for confidence calculation.
            enhancement_data_semantic = {'detected_language': 'english', 'synonyms': [], 'related_terms': [], 'search_strategy': 'semantic_default'}
            try: # Try to get actual detected language
                lang_for_confidence = detect_language(query) if not force_language else force_language
                enhancement_data_semantic['detected_language'] = lang_for_confidence
            except:
                pass # Keep default 'english' if detection fails
            confidence_score = calculate_confidence_score(query, enhancement_data_semantic, all_results, reranked=use_reranking)
            logs.append(f"[search_and_highlight] Semantic confidence: {confidence_score:.2f}")

        else: # Hybrid search
            rerank_suffix = " + reranking" if use_reranking else ""
            page_enrichment_info = ""
            if use_page_enrichment:
                page_enrichment_info = " + page enrichment"
                if include_previous_page or include_next_page:
                    adjacent_pages = []
                    if include_previous_page: adjacent_pages.append("prev")
                    if include_next_page: adjacent_pages.append("next")
                    page_enrichment_info += f" ({'+'.join(adjacent_pages)})"

            logs.append(f"[search_and_highlight] Performing hybrid search ({fusion_strategy}, {semantic_weight:.1f} semantic + {keyword_weight:.1f} keyword) with {enhancement_mode} enhancement{rerank_suffix}{page_enrichment_info}...")
            yield (direct_answer_md, html_results, synthesis_md, get_logs_html())
            hybrid_output = hybrid_search(query, top_k=10, pdf_name=selected_pdf_name,
                                          semantic_weight=semantic_weight, bm25_weight=keyword_weight, enhancement_mode=enhancement_mode,
                                          use_reranking=use_reranking, language=force_language or "auto", fusion_strategy=fusion_strategy,
                                          use_page_enrichment=use_page_enrichment, include_previous_page=include_previous_page, include_next_page=include_next_page)

            if isinstance(hybrid_output, tuple) and len(hybrid_output) == 2:
                all_results, confidence_score_hybrid = hybrid_output
                confidence_score = confidence_score_hybrid # Set the main confidence_score
                logs.append(f"[search_and_highlight] Hybrid search confidence: {confidence_score:.2f}" if confidence_score is not None else "[search_and_highlight] Hybrid search confidence: N/A")
            else: # Fallback if hybrid_search doesn't return tuple (should not happen with current query.py)
                all_results = hybrid_output # Assuming hybrid_output is just the results list
                enhancement_data_hybrid = {}
                if enhancement_mode != "off":
                    try:
                        enhancement_data_hybrid = enhance_query_adaptive(query, enhancement_mode)
                    except Exception as e:
                        logs.append(f"[search_and_highlight] Hybrid (fallback) - Error getting enhancement data for confidence: {e}")
                        enhancement_data_hybrid = {'detected_language': 'unknown', 'synonyms': [], 'related_terms': [], 'search_strategy': 'fallback - error'}
                else: # enhancement_mode == "off"
                    enhancement_data_hybrid = {'detected_language': 'unknown', 'synonyms': [], 'related_terms': [], 'search_strategy': 'enhancement_off'}
                confidence_score = calculate_confidence_score(query, enhancement_data_hybrid, all_results, reranked=use_reranking)
                logs.append(f"[search_and_highlight] Warning: Hybrid search did not return a tuple. Calculated confidence: {confidence_score:.2f}")


        if not all_results:
            logs.append("[search_and_highlight] No results found for query.")
            yield (direct_answer_md, html_results, synthesis_md, get_logs_html())
            return

        top_results = all_results[:5] # Use top 5 for direct answer and detailed display
        html_sections = []

        # Generate confidence HTML
        confidence_html = ""
        if confidence_score is not None:
            confidence_percent = confidence_score * 100
            confidence_class = ""
            confidence_level_text = "" # Renamed for clarity
            if confidence_percent >= 90:
                confidence_level_text = "STRONGLY CONFIDENT"
                confidence_class = "confidence-high"
            elif confidence_percent >= 80:
                confidence_level_text = "MEDIUM CONFIDENCE"
                confidence_class = "confidence-medium"
            else:
                confidence_level_text = "LOW CONFIDENCE"
                confidence_class = "confidence-low"

            confidence_html = f"""
            <div class="confidence-display-container {confidence_class}">
                <strong>Confidence: {confidence_percent:.0f}% ({confidence_level_text})</strong>
            </div>
            """
        else:
            logs.append("[search_and_highlight] Confidence score is None, skipping confidence display.")


        try:
            logs.append("[search_and_highlight] Generating direct answer...")
            yield (direct_answer_md, html_results, synthesis_md, get_logs_html()) # Yield before potentially long LLM call
            direct_answer_content = generate_direct_answer(query, top_results, force_language=force_language)
            # Use Markdown component for direct answer, so convert to Markdown string
            direct_answer_container_content = markdown.markdown(direct_answer_content)
            direct_answer_container = f"""<div class="direct-answer-container">\n{direct_answer_container_content}\n</div>"""
            direct_answer_md = confidence_html + direct_answer_container # Prepend confidence
        except Exception as e:
            logs.append(f"[search_and_highlight] Error generating direct answer: {e}")
            direct_answer_container_error_content = markdown.markdown(f"Error generating direct answer: {str(e)}")
            direct_answer_container_error = f"""<div class="direct-answer-container">\n{direct_answer_container_error_content}\n</div>"""
            direct_answer_md = confidence_html + direct_answer_container_error # Prepend confidence even on error

        yield (direct_answer_md, html_results, synthesis_md, get_logs_html())


        html_sections.append('<div class="detailed-analysis-header">📊 Detailed Analysis & Search Results</div>')

        def flatten_results(results_list): # Renamed for clarity
            flat = []
            for r_item in results_list: # Renamed for clarity
                if isinstance(r_item, (list, tuple)) and len(r_item) == 4 and isinstance(r_item[3], (float, int)):
                    flat.append(tuple(r_item))
                elif isinstance(r_item, (list, tuple)) and all(isinstance(x, (list, tuple)) and len(x) == 4 and isinstance(x[3], (float, int)) for x in r_item):
                    flat.extend([tuple(x) for x in r_item])
                else:
                    logs.append(f"[search_and_highlight] Skipping malformed result: {repr(r_item)}")
            return flat

        top_results = flatten_results(top_results) # Ensure top_results is flat for highlighting

        try:
            logs.append("[search_and_highlight] Processing highlights for Gradio...")
            yield (direct_answer_md, "".join(html_sections), synthesis_md, get_logs_html())
            processed_highlights = process_highlighting_for_gradio(query, top_results, force_language=force_language)
        except Exception as e:
            logs.append(f"[search_and_highlight] Error in highlighting: {e}")
            processed_highlights = [(text, []) for _, _, text, _ in top_results]

        yield (direct_answer_md, "".join(html_sections), synthesis_md, get_logs_html())


        for i, (pdf_name, page_number, page_text, similarity) in enumerate(top_results, 1):
            logs.append(f"[search_and_highlight] Processing result {i}: {pdf_name} page {page_number} (similarity {similarity:.4f})")
            yield (direct_answer_md, "".join(html_sections), synthesis_md, get_logs_html())

            if i - 1 < len(processed_highlights):
                highlighted_text, explanations = processed_highlights[i-1]
            else:
                highlighted_text, explanations = page_text, [] # Fallback

            # Convert page_text to HTML (e.g., nl2br) if it's plain text for display
            # For highlighted_text, it's already HTML from process_highlighted_text_for_html
            # Ensure page_text for LLM analysis is the raw text, not HTML.

            result_html = f"""
{create_web_border(f"📄 {pdf_name} - Page {page_number} │ Similarity: {similarity:.4f} │ Rank {i}", "border-default")}
{create_web_border("📖 CONTENT", "border-content")}
{highlighted_text} 
{close_web_border()}
"""
            if explanations:
                explanations_html = '<div class="explanations-list">'
                for j, explanation in enumerate(explanations, 1):
                    explanations_html += f'<div class="explanation-item"><span class="explanation-number">[{j}]</span>{markdown.markdown(explanation)}</div>'
                explanations_html += '</div>'
                result_html += f"""
{create_web_border("💡 RELEVANCE ANALYSIS (from highlighting)", "border-analysis")}
{explanations_html}
{close_web_border()}
"""
            try:
                logs.append(f"[search_and_highlight] Analyzing individual result {i}...")
                yield (direct_answer_md, "".join(html_sections), synthesis_md, get_logs_html())
                individual_analysis_raw = analyze_individual_result_web(query, page_text, pdf_name, page_number, similarity, force_language=force_language)
                individual_analysis_html = markdown.markdown(individual_analysis_raw)
                result_html += f"""
{create_web_border("🧠 LLM ANALYSIS", "border-analysis")}
<div class="individual-analysis">{individual_analysis_html}</div>
{close_web_border()}
"""
            except Exception as e:
                logs.append(f"[search_and_highlight] Error in individual analysis for result {i}: {e}")
                individual_analysis_error_html = markdown.markdown(f"Error generating analysis: {str(e)}")
                result_html += f"""
{create_web_border("🧠 LLM ANALYSIS", "border-analysis")}
<div class="individual-analysis">{individual_analysis_error_html}</div>
{close_web_border()}
"""
            result_html += f"{close_web_border()}<div class='section-separator'></div>"
            html_sections.append(result_html)
            yield (direct_answer_md, "".join(html_sections), synthesis_md, get_logs_html())

        try:
            logs.append("[search_and_highlight] Synthesizing results...")
            yield (direct_answer_md, "".join(html_sections), synthesis_md, get_logs_html())
            synthesis_raw = synthesize_results_web(query, top_results, force_language=force_language)
            # Synthesis output is Markdown, so it's fine for gr.Markdown
            synthesis_md = f"### 🧩 Comprehensive Synthesis\n\n{synthesis_raw}"
        except Exception as e:
            logs.append(f"[search_and_highlight] Error in synthesis: {e}")
            synthesis_md = f"### 🧩 Comprehensive Synthesis\n\nError: {str(e)}"

        yield (direct_answer_md, "".join(html_sections), synthesis_md, get_logs_html())

        summary_info = f"📊 <b>Total results found:</b> {len(all_results)}"
        if top_results:
            best_result, worst_of_top = top_results[0], top_results[-1]
            summary_info += f" | <b>Best (Top {len(top_results)}):</b> {best_result[0]} Page {best_result[1]} ({best_result[3]:.4f}) | <b>Worst (Top {len(top_results)}):</b> {worst_of_top[0]} Page {worst_of_top[1]} ({worst_of_top[3]:.4f})"

        html_sections.append(f"<div style='margin:1em 0; text-align:center; font-size:0.9em;'>{summary_info}</div>")
        html_results = "".join(html_sections)

        yield (direct_answer_md, html_results, synthesis_md, get_logs_html())
        logs.append("[search_and_highlight] Done. Returning results.")
        yield (direct_answer_md, html_results, synthesis_md, get_logs_html())

    except Exception as e:
        logs.append(f"[search_and_highlight] Fatal error: {e}")
        import traceback
        logs.append(traceback.format_exc())
        error_html = f"<h2>An error occurred:</h2><p>{str(e)}</p><p>Please check the logs for more details.</p>"
        # Ensure confidence HTML is prepended if available, even on fatal error
        final_direct_answer_md = confidence_html + markdown.markdown(f"Error: {str(e)}") if 'confidence_html' in locals() else markdown.markdown(f"Error: {str(e)}")
        yield (final_direct_answer_md, error_html, "", get_logs_html())


def create_interface():
    """Create the Gradio interface"""
    with gr.Blocks(
            title="🔍 LLM RAG Advanced Search",
            theme=gr.themes.Default(),
            css="""
        .gradio-container {
            max-width: 1200px !important; 
            margin: auto;
        }
        .search-box { 
            font-size: 16px !important; 
        }
        """ + create_custom_css()
    ) as interface:
        gr.HTML("""
        <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #4a5568 0%, #2d3748 100%); border-radius: 15px; margin-bottom: 2rem;">
            <h1 style="color: white; margin: 0; font-size: 2.5em;">🔍 LLM RAG Advanced Search</h1>
            <p style="color: rgba(255,255,255,0.9); font-size: 1.2em; margin: 0.5rem 0 0 0;">
                Intelligent document search with semantic highlighting and LLM-powered insights
            </p>
        </div>
        """)

        with gr.Row():
            with gr.Column(scale=4):
                query_input = gr.Textbox(
                    label="🔍 Search Query",
                    placeholder="Enter your search query here...",
                    elem_classes=["search-box"]
                )
            with gr.Column(scale=1):
                collections_input = gr.Textbox(
                    label="📚 Collections",
                    value="all",
                    placeholder="all or comma-separated names"
                )
            with gr.Column(scale=1):
                language_input = gr.Dropdown(
                    label="🌐 Language",
                    choices=["Auto-detect", "English", "Italian", "Spanish", "French"],
                    value="Italian",
                    interactive=True
                )

        with gr.Row():
            with gr.Column(scale=1):
                search_method_input = gr.Dropdown(
                    label="⚙️ Search Method",
                    choices=["Hybrid", "Semantic", "BM25"],
                    value="Hybrid",
                    interactive=True
                )
            with gr.Column(scale=1):
                fusion_strategy_input = gr.Dropdown(
                    label="🔗 Fusion Strategy",
                    choices=["weighted", "rrf", "comb_sum", "comb_mnz", "adaptive"],
                    value="rrf",
                    interactive=True,
                    visible=True,
                    info="Strategy for combining semantic + keyword results (for Hybrid)"
                )
            with gr.Column(scale=1):
                semantic_weight_input = gr.Slider(
                    label="🧠 Semantic Weight",
                    minimum=0.0,
                    maximum=1.0,
                    step=0.1,
                    value=0.6,
                    visible=True
                )
            with gr.Column(scale=1):
                keyword_weight_input = gr.Slider(
                    label="🔑 Keyword Weight",
                    minimum=0.0,
                    maximum=1.0,
                    step=0.1,
                    value=0.4,
                    visible=True
                )
            with gr.Column(scale=1):
                reranking_input = gr.Checkbox(
                    label="✨ LLM Reranking",
                    value=False,
                    info="Use Gemini Flash 1.5 to intelligently reorder results (~2s)"
                )

        with gr.Row():
            with gr.Column(scale=1):
                enhancement_mode_input = gr.Dropdown(
                    label="💡 Query Enhancement",
                    choices=["auto", "minimal", "full", "maximum", "off"],
                    value="auto",
                    interactive=True,
                    info="AI-powered query translation and expansion"
                )
            with gr.Column(scale=1):
                page_enrichment_input = gr.Checkbox(
                    label="📄 Page Enrichment",
                    value=False,
                    info="Expand chunks to full pages for better context"
                )
            with gr.Column(scale=1):
                include_previous_input = gr.Checkbox(
                    label="⏪ Include Previous Page",
                    value=False,
                    visible=False,
                    info="Add previous page for narrative context"
                )
            with gr.Column(scale=1):
                include_next_input = gr.Checkbox(
                    label="⏩ Include Next Page",
                    value=False,
                    visible=False,
                    info="Add next page for narrative context"
                )
            with gr.Column(scale=1):
                gr.HTML("<div style='padding: 1rem; color: #666; font-size: 0.9em;'>💡 Enhancement: 'auto' adapts to query type</div>")


        search_button = gr.Button("🚀 Search", variant="primary", size="lg")

        direct_answer_output = gr.HTML( # Changed to HTML to allow confidence display
            label="🎯 Direct Answer & Confidence",
            value="<p>Enter a query and click search to see results.</p>"
        )

        results_output = gr.HTML(
            label="📊 Detailed Analysis & Search Results",
            value=""
        )

        synthesis_output = gr.Markdown( # Synthesis can remain Markdown
            label="🧩 Comprehensive Synthesis",
            value="Synthesis will appear here..."
        )

        logs_output = gr.HTML(
            label="📝 Logs",
            value="<div>No logs yet.</div>"
        )

        def update_slider_visibility(search_method_val):
            if search_method_val == "Hybrid":
                return (
                    gr.update(visible=True),
                    gr.update(visible=True),
                    gr.update(visible=True)
                )
            else:
                return (
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False)
                )

        def normalize_semantic_weight(sem_weight):
            key_weight = round(1.0 - sem_weight, 1) # Ensure one decimal place
            return gr.update(value=key_weight)

        def normalize_keyword_weight(key_weight):
            sem_weight = round(1.0 - key_weight, 1) # Ensure one decimal place
            return gr.update(value=sem_weight)

        def update_page_enrichment_visibility(page_enrich_enabled):
            if page_enrich_enabled:
                return gr.update(visible=True), gr.update(visible=True)
            else:
                return gr.update(visible=False, value=False), gr.update(visible=False, value=False)


        search_method_input.change(
            fn=update_slider_visibility,
            inputs=[search_method_input],
            outputs=[fusion_strategy_input, semantic_weight_input, keyword_weight_input]
        )
        semantic_weight_input.change(
            fn=normalize_semantic_weight,
            inputs=[semantic_weight_input],
            outputs=[keyword_weight_input]
        )
        keyword_weight_input.change(
            fn=normalize_keyword_weight,
            inputs=[keyword_weight_input],
            outputs=[semantic_weight_input]
        )
        page_enrichment_input.change(
            fn=update_page_enrichment_visibility,
            inputs=[page_enrichment_input],
            outputs=[include_previous_input, include_next_input]
        )

        search_inputs = [
            query_input, collections_input, language_input,
            search_method_input, fusion_strategy_input,
            semantic_weight_input, keyword_weight_input,
            reranking_input, page_enrichment_input,
            include_previous_input, include_next_input,
            enhancement_mode_input
        ]
        search_outputs = [direct_answer_output, results_output, synthesis_output, logs_output]

        search_button.click(
            fn=search_and_highlight,
            inputs=search_inputs,
            outputs=search_outputs,
            show_progress="full"
        )
        query_input.submit(
            fn=search_and_highlight,
            inputs=search_inputs,
            outputs=search_outputs,
            show_progress="full"
        )

        gr.HTML("""
        <div style="text-align: center; margin-top: 2rem; padding: 1rem;">
            <h3 style="color: var(--body-text-color, #666);">🎨 Highlighting Legend</h3>
            <div style="display: flex; justify-content: center; flex-wrap: wrap; gap: 1rem;">
                <span style="background: #f1c40f; padding: 4px 8px; border-radius: 4px; color: #333333; font-weight: 600;">Highlighted Text</span>
                <span style="color: #2980b9; font-style: italic;">Explanations in italics</span>
            </div>
        </div>
        """)
    return interface

if __name__ == "__main__":
    interface = create_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_api=False
    )