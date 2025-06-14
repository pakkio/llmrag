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
import markdown

# Add the current directory to Python path to import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from query import (
    list_available_collections, query_chroma_collections,
    highlight_relevant_text_batch, detect_language, generate_direct_answer
)
from llm_wrapper import llm_call, generate_embeddings, test_openai_embeddings

def create_web_border(title: str, style_class: str = "border-default") -> str:
    """Create a web-friendly bordered section with title"""
    return f'<div class="{style_class}"><div class="border-title">{title}</div><div class="border-content">'

def close_web_border() -> str:
    """Close a web bordered section"""
    return '</div></div>'

def process_highlighted_text_for_html(highlighted_text: str) -> Tuple[str, List[str]]:
    """Convert query.py style highlighting to HTML and separate explanations"""
    # Extract explanations first (before converting to HTML)
    explanations = []
    explanation_pattern = r'\[EXPLAIN\](.*?)\[/EXPLAIN\]'
    explanations = re.findall(explanation_pattern, highlighted_text)
    
    # Remove explanations from main text and add footnote numbers
    clean_text = highlighted_text
    for i, explanation in enumerate(explanations, 1):
        # Replace explanation with footnote number
        clean_text = re.sub(r'\[EXPLAIN\].*?\[/EXPLAIN\]', f'<sup class="footnote-number">[{i}]</sup>', clean_text, count=1)
    
    # Convert [HIGHLIGHT] tags to HTML
    clean_text = clean_text.replace('[HIGHLIGHT]', '<span class="highlight-text">')
    clean_text = clean_text.replace('[/HIGHLIGHT]', '</span>')
    
    return clean_text, explanations

def analyze_individual_result_web(query: str, text_content: str, pdf_name: str, page_number: int, similarity: float) -> str:
    """Generate individual result analysis for web display (from query.py)"""
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

def synthesize_results_web(query: str, results: List[Tuple[str, int, str, float]]) -> str:
    """Generate comprehensive synthesis for web display (from query.py)"""
    if not results:
        return "No results to synthesize."
    
    detected_language = detect_language(results[0][2])
    
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

Provide a structured analysis (3-4 paragraphs) that helps someone understand the collective meaning and value of these search results."""
        }
    ]
    
    synthesis, success = llm_call(messages, max_tokens=800)
    if success and synthesis.strip():
        return synthesis.strip()
    return "Synthesis unavailable for these results."


def process_highlighting_for_gradio(query: str, results: List[Tuple[str, int, str, float]]) -> List[Tuple[str, List[str]]]:
    """Use query.py's highlighting and convert to HTML for gradio"""
    if not results:
        return []
    
    # Get raw highlighted texts with [HIGHLIGHT] tags (reuse from query.py)
    try:
        highlighted_texts = highlight_relevant_text_batch(query, results)
    except Exception as e:
        # Fallback to original texts
        highlighted_texts = [text_content for _, _, text_content, _ in results]
    
    # Convert to HTML format and separate explanations
    processed_results = []
    for highlighted_text in highlighted_texts:
        html_text, explanations = process_highlighted_text_for_html(highlighted_text)
        processed_results.append((html_text, explanations))
    
    return processed_results



def create_custom_css() -> str:
    """Create CSS with proper dark theme support"""
    return """
    <style>
    /* Light theme defaults */
    .results-container {
        background: #ffffff;
        color: #2d3748;
        border: 2px solid #e2e8f0;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Bordered sections */
    .border-default {
        background: #f8fafc;
        border: 2px solid #cbd5e0;
        border-radius: 8px;
        margin: 1rem 0;
        overflow: hidden;
    }
    
    .border-content {
        background: #f8fafc;
        border: 2px solid #3182ce;
        border-radius: 8px;
        margin: 1rem 0;
        overflow: hidden;
    }
    
    .border-analysis {
        background: #fef5e7;
        border: 2px solid #d69e2e;
        border-radius: 8px;
        margin: 1rem 0;
        overflow: hidden;
    }
    
    .border-synthesis {
        background: #f0fff4;
        border: 2px solid #38a169;
        border-radius: 8px;
        margin: 1rem 0;
        overflow: hidden;
    }
    
    .border-title {
        background: #4a5568;
        color: white;
        padding: 0.75rem 1rem;
        font-weight: 600;
        font-size: 1.1em;
        border-bottom: 2px solid #2d3748;
    }
    
    .border-content .border-title {
        background: #3182ce;
        border-bottom: 2px solid #2c5282;
    }
    
    .border-analysis .border-title {
        background: #d69e2e;
        color: #1a202c;
        border-bottom: 2px solid #b7791f;
    }
    
    .border-synthesis .border-title {
        background: #38a169;
        border-bottom: 2px solid #2f855a;
    }
    
    .border-content > div:last-child {
        padding: 1.5rem;
    }
    
    .footnote-number {
        background: #3182ce;
        color: white;
        padding: 2px 6px;
        border-radius: 50%;
        font-size: 0.8em;
        font-weight: 600;
        margin-left: 2px;
    }
    
    .explanations-list {
        background: #f7fafc;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .explanation-item {
        margin: 0.5rem 0;
        padding: 0.5rem;
        background: white;
        border-radius: 4px;
        border-left: 4px solid #3182ce;
    }
    
    .explanation-number {
        color: #3182ce;
        font-weight: 600;
        margin-right: 0.5rem;
    }
    
    .individual-analysis {
        background: #fef5e7;
        color: #744210;
        padding: 1rem;
        border-radius: 6px;
        margin: 1rem 0;
        font-style: italic;
    }
    
    .synthesis-content {
        background: #f0fff4;
        color: #276749;
        padding: 1.5rem;
        border-radius: 6px;
        margin: 1rem 0;
        line-height: 1.6;
    }
    
    .highlight-text {
        background-color: #fbbf24;
        color: #1a202c;
        padding: 2px 4px;
        border-radius: 3px;
        font-weight: 600;
    }
    
    .explain-text {
        font-size: 0.9em;
        color: #3182ce;
        font-style: italic;
        margin-left: 3px;
    }
    
    .result-header {
        font-size: 1.1em;
        font-weight: 600;
        color: #2b6cb0;
        margin-bottom: 0.5rem;
        border-bottom: 2px solid #cbd5e0;
        padding-bottom: 0.5rem;
    }
    
    .relevance-score {
        background: #3182ce;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8em;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 1rem;
    }
    
    .source-info {
        font-size: 0.9em;
        color: #718096;
        margin-top: 1rem;
        padding-top: 0.5rem;
        border-top: 1px solid #e2e8f0;
    }
    
    .direct-answer-container {
        background: #f0fff4;
        border: 2px solid #38a169;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 2rem 0;
    }
    
    .direct-answer-header {
        font-size: 1.3em;
        font-weight: 700;
        color: #276749;
        margin-bottom: 1rem;
    }
    
    .direct-answer-content {
        font-size: 1.0em;
        color: #276749;
        line-height: 1.6;
    }
    
    .direct-answer-content h1, .direct-answer-content h2, .direct-answer-content h3, 
    .direct-answer-content h4, .direct-answer-content h5, .direct-answer-content h6 {
        color: #2c7a2c;
        margin: 1em 0 0.5em 0;
    }
    
    .direct-answer-content ul, .direct-answer-content ol {
        margin: 0.5em 0;
        padding-left: 1.5em;
    }
    
    .direct-answer-content li {
        margin: 0.3em 0;
    }
    
    .direct-answer-content p {
        margin: 0.8em 0;
    }
    
    .direct-answer-content code {
        background: #e8f5e8;
        padding: 2px 4px;
        border-radius: 3px;
        font-family: 'Courier New', monospace;
        color: #2c7a2c;
    }
    
    .direct-answer-content pre {
        background: #e8f5e8;
        padding: 1em;
        border-radius: 5px;
        overflow-x: auto;
        margin: 1em 0;
    }
    
    .direct-answer-content blockquote {
        border-left: 4px solid #38a169;
        margin: 1em 0;
        padding-left: 1em;
        font-style: italic;
        color: #2c7a2c;
    }
    
    .direct-answer-content strong, .direct-answer-content b {
        font-weight: 700;
        color: #2c7a2c;
    }
    
    .direct-answer-content em, .direct-answer-content i {
        font-style: italic;
        color: #2c7a2c;
    }
    
    .section-separator {
        height: 2px;
        background: #3182ce;
        margin: 2rem 0;
        border-radius: 1px;
    }
    
    .detailed-analysis-header {
        font-size: 1.2em;
        font-weight: 600;
        color: #2b6cb0;
        margin: 2rem 0 1rem 0;
        padding: 1rem;
        background: #f7fafc;
        border-radius: 8px;
        border-left: 4px solid #3182ce;
        text-align: center;
    }
    
    /* Dark theme overrides */
    @media (prefers-color-scheme: dark) {
        .results-container {
            background: #2d3748;
            color: #e2e8f0;
            border: 2px solid #4a5568;
        }
        
        .highlight-text {
            background-color: #d69e2e;
            color: #1a202c;
        }
        
        .explain-text {
            color: #63b3ed;
        }
        
        .result-header {
            color: #63b3ed;
            border-bottom: 2px solid #4a5568;
        }
        
        .relevance-score {
            background: #4299e1;
        }
        
        .source-info {
            color: #a0aec0;
            border-top: 1px solid #4a5568;
        }
        
        .direct-answer-container {
            background: #1a202c;
            border: 2px solid #48bb78;
        }
        
        .direct-answer-header {
            color: #68d391;
        }
        
        .direct-answer-content {
            color: #c6f6d5;
        }
        
        .direct-answer-content h1, .direct-answer-content h2, .direct-answer-content h3, 
        .direct-answer-content h4, .direct-answer-content h5, .direct-answer-content h6 {
            color: #68d391;
        }
        
        .direct-answer-content code {
            background: #2d3748;
            color: #68d391;
        }
        
        .direct-answer-content pre {
            background: #2d3748;
        }
        
        .direct-answer-content blockquote {
            border-left: 4px solid #48bb78;
            color: #68d391;
        }
        
        .direct-answer-content strong, .direct-answer-content b {
            color: #68d391;
        }
        
        .direct-answer-content em, .direct-answer-content i {
            color: #68d391;
        }
        
        .section-separator {
            background: #4299e1;
        }
        
        .detailed-analysis-header {
            color: #63b3ed;
            background: #1a202c;
            border-left: 4px solid #4299e1;
        }
    }
    
    /* Gradio dark theme specific overrides */
    [data-theme="dark"] .results-container {
        background: #2d3748;
        color: #e2e8f0;
        border: 2px solid #4a5568;
    }
    
    [data-theme="dark"] .highlight-text {
        background-color: #d69e2e;
        color: #1a202c;
    }
    
    [data-theme="dark"] .explain-text {
        color: #63b3ed;
    }
    
    [data-theme="dark"] .result-header {
        color: #63b3ed;
        border-bottom: 2px solid #4a5568;
    }
    
    [data-theme="dark"] .source-info {
        color: #a0aec0;
        border-top: 1px solid #4a5568;
    }
    
    [data-theme="dark"] .direct-answer-container {
        background: #1a202c;
        border: 2px solid #48bb78;
    }
    
    [data-theme="dark"] .direct-answer-header {
        color: #68d391;
    }
    
    [data-theme="dark"] .direct-answer-content {
        color: #c6f6d5;
    }
    
    [data-theme="dark"] .direct-answer-content h1, [data-theme="dark"] .direct-answer-content h2, 
    [data-theme="dark"] .direct-answer-content h3, [data-theme="dark"] .direct-answer-content h4, 
    [data-theme="dark"] .direct-answer-content h5, [data-theme="dark"] .direct-answer-content h6 {
        color: #68d391;
    }
    
    [data-theme="dark"] .direct-answer-content code {
        background: #2d3748;
        color: #68d391;
    }
    
    [data-theme="dark"] .direct-answer-content pre {
        background: #2d3748;
    }
    
    [data-theme="dark"] .direct-answer-content blockquote {
        border-left: 4px solid #48bb78;
        color: #68d391;
    }
    
    [data-theme="dark"] .direct-answer-content strong, [data-theme="dark"] .direct-answer-content b {
        color: #68d391;
    }
    
    [data-theme="dark"] .direct-answer-content em, [data-theme="dark"] .direct-answer-content i {
        color: #68d391;
    }
    
    [data-theme="dark"] .section-separator {
        background: #4299e1;
    }
    
    [data-theme="dark"] .detailed-analysis-header {
        color: #63b3ed;
        background: #1a202c;
        border-left: 4px solid #4299e1;
    }
    </style>
    """

def search_and_highlight(query: str, collection_names: str = "all") -> str:
    """Main search function with markdown output"""
    if not query.strip():
        return "Please enter a search query."
    
    try:
        # Check if OpenAI embedding API is available
        if not test_openai_embeddings():
            return "OpenAI embedding API not available. Please check OPENAI_API_KEY."
        
        # Get available collections
        collections = list_available_collections()
        if not collections:
            return "No collections found. Please run ingestion first."
        
        # Parse collection names
        available_book_names = [book_name for book_name, _ in collections]
        if collection_names.lower() == "all":
            selected_pdf_name = None  # Search all collections
        else:
            # For simplicity, use the first valid collection name
            selected_names = [name.strip() for name in collection_names.split(",")]
            valid_names = [name for name in selected_names if name in available_book_names]
            selected_pdf_name = valid_names[0] if valid_names else None
        
        # Generate query embedding using OpenAI API
        query_embedding = generate_embeddings(query, normalize=True)
        
        # Perform search
        all_results = query_chroma_collections(query_embedding, top_k=5, pdf_name=selected_pdf_name)
        
        if not all_results:
            return "No results found for your query."
        
        top_results = all_results[:5]
        
        # Create markdown results
        markdown_results = []
        
        # PART 1: Direct Answer Section
        try:
            direct_answer = generate_direct_answer(query, top_results)
            markdown_results.append(f"## 🎯 Direct Answer\n\n{direct_answer}\n\n---\n")
        except Exception as e:
            markdown_results.append(f"## 🎯 Direct Answer\n\nError generating direct answer: {str(e)}\n\n---\n")
        
        # PART 2: Detailed Analysis Section
        markdown_results.append("## 🔍 Detailed Analysis & Search Results\n\n")
        
        # Get highlighted texts using query.py's highlighting functionality
        try:
            processed_highlights = process_highlighting_for_gradio(query, top_results)
        except Exception as e:
            # Fallback to original text if highlighting fails
            processed_highlights = [(text, []) for _, _, text, _ in top_results]
        
        for i, (pdf_name, page_number, page_text, similarity) in enumerate(top_results, 1):
            # Calculate relevance percentage  
            relevance = max(0, min(100, int(similarity * 100)))
            
            # Get highlighted text and explanations
            if i-1 < len(processed_highlights):
                highlighted_text, explanations = processed_highlights[i-1]
            else:
                highlighted_text, explanations = page_text, []
            
            # Create result with rich formatting
            result_markdown = f"""
{create_web_border(f"📄 {pdf_name} - Page {page_number} │ Similarity: {similarity:.4f} │ Rank {i}", "border-default")}

{create_web_border("📖 CONTENT", "border-content")}
{highlighted_text}
{close_web_border()}
"""
            
            # Add explanations if any
            if explanations:
                explanations_html = '<div class="explanations-list">'
                for j, explanation in enumerate(explanations, 1):
                    explanations_html += f'<div class="explanation-item"><span class="explanation-number">[{j}]</span>{explanation}</div>'
                explanations_html += '</div>'
                
                result_markdown += f"""
{create_web_border("💡 RELEVANCE ANALYSIS", "border-analysis")}
{explanations_html}
{close_web_border()}
"""
            
            # Add individual LLM analysis
            try:
                individual_analysis = analyze_individual_result_web(query, page_text, pdf_name, page_number, similarity)
                result_markdown += f"""
{create_web_border("🧠 LLM ANALYSIS", "border-analysis")}
<div class="individual-analysis">{individual_analysis}</div>
{close_web_border()}
"""
            except Exception as e:
                result_markdown += f"""
{create_web_border("🧠 LLM ANALYSIS", "border-analysis")}
<div class="individual-analysis">Error generating analysis: {str(e)}</div>
{close_web_border()}
"""
            
            result_markdown += f"{close_web_border()}\n\n---\n\n"
            markdown_results.append(result_markdown)
        
        # Add comprehensive synthesis
        try:
            synthesis = synthesize_results_web(query, top_results)
            synthesis_section = f"""
{create_web_border("🔬 COMPREHENSIVE SYNTHESIS", "border-synthesis")}
<div class="synthesis-content">{synthesis}</div>
{close_web_border()}

---
"""
            markdown_results.append(synthesis_section)
        except Exception as e:
            markdown_results.append(f"**Synthesis Error:** {str(e)}\n\n---\n\n")
        
        # Add summary stats
        best_result = top_results[0] if top_results else None
        worst_result = top_results[-1] if top_results else None
        summary_info = f"📊 **Total results:** {len(all_results)}"
        if best_result and worst_result:
            summary_info += f" | **Best:** {best_result[0]} Page {best_result[1]} ({best_result[3]:.4f}) | **Worst:** {worst_result[0]} Page {worst_result[1]} ({worst_result[3]:.4f})"
        markdown_results.append(f"\n{summary_info}\n")
        
        # Add CSS styling for proper display
        css_styles = create_custom_css()
        final_result = css_styles + "\n\n" + "".join(markdown_results)
        
        return final_result
        
    except Exception as e:
        return f"Error: {str(e)}"

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
        """
    ) as interface:
        
        gr.HTML("""
        <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #4a5568 0%, #2d3748 100%); border-radius: 15px; margin-bottom: 2rem;">
            <h1 style="color: white; margin: 0; font-size: 2.5em;">🔍 LLM RAG Advanced Search</h1>
            <p style="color: rgba(255,255,255,0.9); font-size: 1.2em; margin: 0.5rem 0 0 0;">
                Intelligent document search with semantic highlighting
            </p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=3):
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
        
        search_button = gr.Button("🚀 Search", variant="primary", size="lg")
        
        results_output = gr.Markdown(
            label="Search Results",
            value="Enter a query and click search to see results."
        )
        
        # Set up the search functionality
        search_button.click(
            fn=search_and_highlight,
            inputs=[query_input, collections_input],
            outputs=results_output,
            show_progress=True
        )
        
        # Also trigger search on Enter key
        query_input.submit(
            fn=search_and_highlight,
            inputs=[query_input, collections_input],
            outputs=results_output,
            show_progress=True
        )
        
        gr.HTML("""
        <div style="text-align: center; margin-top: 2rem; padding: 1rem;">
            <h3 style="color: var(--body-text-color, #666);">🎨 Highlighting Legend</h3>
            <div style="display: flex; justify-content: center; flex-wrap: wrap; gap: 1rem;">
                <span style="background: #fbbf24; padding: 4px 8px; border-radius: 4px; color: #1a202c; font-weight: 600;">Highlighted Text</span>
                <span style="color: #3182ce; font-style: italic;">[Explanations in italics]</span>
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