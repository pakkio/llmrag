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

# Add the current directory to Python path to import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from query import (
    load_collections, search_collections, highlight_relevant_text,
    llm_call, detect_language
)

def create_advanced_highlight_prompt(query: str, text: str) -> str:
    """Create an enhanced prompt for sophisticated multi-color highlighting"""
    detected_language = detect_language(text)
    
    if detected_language == "italian":
        lang_instruction = "Rispondi in italiano naturale."
        colors_explanation = """
        - HIGHLIGHT_PRIMARY: Corrispondenze dirette o molto rilevanti (giallo)
        - HIGHLIGHT_SECONDARY: Connessioni concettuali forti (arancione)  
        - HIGHLIGHT_WEAK: Collegamenti deboli o contestuali (blu chiaro)
        - HIGHLIGHT_CONTEXT: Contesto importante che supporta la comprensione (verde chiaro)
        """
    else:
        lang_instruction = "Respond in natural English."
        colors_explanation = """
        - HIGHLIGHT_PRIMARY: Direct matches or highly relevant (yellow)
        - HIGHLIGHT_SECONDARY: Strong conceptual connections (orange)
        - HIGHLIGHT_WEAK: Weak or contextual links (light blue)  
        - HIGHLIGHT_CONTEXT: Important supporting context (light green)
        """
    
    return f"""Given query: "{query}"
Text to analyze:
{text}

Create sophisticated multi-level highlighting using these tags:
{colors_explanation}

{lang_instruction}

Rules:
1. Use different highlight levels based on relevance strength
2. Add brief explanations after each highlight: [EXPLAIN_TYPE]reason[/EXPLAIN_TYPE]
3. Be creative with semantic connections
4. Return the complete original text with highlights
5. Use varied highlight types to create visual interest

Example:
Text [HIGHLIGHT_PRIMARY]direct match[/HIGHLIGHT_PRIMARY] [EXPLAIN_PRIMARY]Exact keyword match[/EXPLAIN_PRIMARY] 
more text [HIGHLIGHT_SECONDARY]related concept[/HIGHLIGHT_SECONDARY] [EXPLAIN_SECONDARY]Conceptually connected[/EXPLAIN_SECONDARY]
context [HIGHLIGHT_CONTEXT]supporting info[/HIGHLIGHT_CONTEXT] [EXPLAIN_CONTEXT]Provides important context[/EXPLAIN_CONTEXT]
"""

def process_enhanced_highlighting(query: str, text: str) -> str:
    """Process text with advanced multi-color highlighting"""
    messages = [{"role": "user", "content": create_advanced_highlight_prompt(query, text)}]
    
    highlighted_text, success = llm_call(messages, max_tokens=8000)
    if not success:
        return text
    
    # Convert to sophisticated HTML with CSS classes
    color_mappings = {
        'HIGHLIGHT_PRIMARY': ('<span class="highlight-primary">', '</span>'),
        'HIGHLIGHT_SECONDARY': ('<span class="highlight-secondary">', '</span>'),
        'HIGHLIGHT_WEAK': ('<span class="highlight-weak">', '</span>'),
        'HIGHLIGHT_CONTEXT': ('<span class="highlight-context">', '</span>'),
        'EXPLAIN_PRIMARY': ('<span class="explain explain-primary">', '</span>'),
        'EXPLAIN_SECONDARY': ('<span class="explain explain-secondary">', '</span>'),
        'EXPLAIN_WEAK': ('<span class="explain explain-weak">', '</span>'),
        'EXPLAIN_CONTEXT': ('<span class="explain explain-context">', '</span>'),
    }
    
    for tag, (open_html, close_html) in color_mappings.items():
        highlighted_text = highlighted_text.replace(f'[{tag}]', open_html)
        highlighted_text = highlighted_text.replace(f'[/{tag}]', close_html)
    
    return highlighted_text

def create_custom_css() -> str:
    """Create sophisticated CSS for the highlighting interface"""
    return """
    <style>
    .search-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }
    
    .results-container {
        background: #ffffff;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    
    .highlight-primary {
        background: linear-gradient(120deg, #ffd700, #ffed4e);
        color: #333;
        padding: 2px 4px;
        border-radius: 4px;
        font-weight: 600;
        box-shadow: 0 2px 4px rgba(255,215,0,0.3);
    }
    
    .highlight-secondary {
        background: linear-gradient(120deg, #ff8c00, #ffa500);
        color: white;
        padding: 2px 4px;
        border-radius: 4px;
        font-weight: 500;
        box-shadow: 0 2px 4px rgba(255,140,0,0.3);
    }
    
    .highlight-weak {
        background: linear-gradient(120deg, #87ceeb, #add8e6);
        color: #333;
        padding: 2px 4px;
        border-radius: 4px;
        font-style: italic;
        box-shadow: 0 2px 4px rgba(135,206,235,0.3);
    }
    
    .highlight-context {
        background: linear-gradient(120deg, #90ee90, #98fb98);
        color: #333;
        padding: 2px 4px;
        border-radius: 4px;
        box-shadow: 0 2px 4px rgba(144,238,144,0.3);
    }
    
    .explain {
        font-size: 0.85em;
        margin-left: 5px;
        padding: 2px 6px;
        border-radius: 10px;
        display: inline-block;
    }
    
    .explain-primary {
        background: rgba(255,215,0,0.2);
        color: #b8860b;
        border: 1px solid rgba(255,215,0,0.4);
    }
    
    .explain-secondary {
        background: rgba(255,140,0,0.2);
        color: #cc6600;
        border: 1px solid rgba(255,140,0,0.4);
    }
    
    .explain-weak {
        background: rgba(135,206,235,0.2);
        color: #4682b4;
        border: 1px solid rgba(135,206,235,0.4);
    }
    
    .explain-context {
        background: rgba(144,238,144,0.2);
        color: #228b22;
        border: 1px solid rgba(144,238,144,0.4);
    }
    
    .result-header {
        font-size: 1.1em;
        font-weight: 600;
        color: #667eea;
        margin-bottom: 0.5rem;
        border-bottom: 2px solid #f0f0f0;
        padding-bottom: 0.5rem;
    }
    
    .relevance-score {
        background: linear-gradient(45deg, #667eea, #764ba2);
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
        color: #666;
        margin-top: 1rem;
        padding-top: 0.5rem;
        border-top: 1px solid #eee;
    }
    </style>
    """

def search_and_highlight(query: str, collection_names: str = "all") -> str:
    """Main search function with enhanced highlighting"""
    if not query.strip():
        return "<p>Please enter a search query.</p>"
    
    try:
        # Load collections
        collections = load_collections()
        if not collections:
            return "<p>No collections found. Please run ingestion first.</p>"
        
        # Parse collection names
        if collection_names.lower() == "all":
            selected_collections = list(collections.keys())
        else:
            selected_collections = [name.strip() for name in collection_names.split(",")]
            selected_collections = [name for name in selected_collections if name in collections]
        
        if not selected_collections:
            return "<p>No valid collections specified.</p>"
        
        # Perform search
        all_results = search_collections(query, selected_collections, collections, top_k=5)
        
        if not all_results:
            return "<p>No results found for your query.</p>"
        
        # Create enhanced HTML results
        html_results = [create_custom_css()]
        
        for i, (distance, page_text, metadata) in enumerate(all_results[:5], 1):
            # Process with enhanced highlighting
            highlighted_text = process_enhanced_highlighting(query, page_text)
            
            # Calculate relevance percentage
            relevance = max(0, min(100, int((1 - distance) * 100)))
            
            # Create result HTML
            source = metadata.get('source', 'Unknown')
            page = metadata.get('page', 'N/A')
            
            result_html = f"""
            <div class="results-container">
                <div class="result-header">Result #{i}</div>
                <div class="relevance-score">Relevance: {relevance}%</div>
                <div class="highlighted-content">{highlighted_text}</div>
                <div class="source-info">
                    <strong>Source:</strong> {source} | <strong>Page:</strong> {page}
                </div>
            </div>
            """
            html_results.append(result_html)
        
        return "".join(html_results)
        
    except Exception as e:
        return f"<p>Error: {str(e)}</p>"

def create_interface():
    """Create the Gradio interface"""
    
    with gr.Blocks(
        title="🔍 LLM RAG Advanced Search",
        theme=gr.themes.Soft(),
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
        <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; margin-bottom: 2rem;">
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
        
        results_output = gr.HTML(
            label="Search Results",
            value="<p>Enter a query and click search to see results.</p>"
        )
        
        # Set up the search functionality
        search_button.click(
            fn=search_and_highlight,
            inputs=[query_input, collections_input],
            outputs=results_output
        )
        
        # Also trigger search on Enter key
        query_input.submit(
            fn=search_and_highlight,
            inputs=[query_input, collections_input],
            outputs=results_output
        )
        
        gr.HTML("""
        <div style="text-align: center; margin-top: 2rem; padding: 1rem; color: #666;">
            <h3>🎨 Highlighting Legend</h3>
            <div style="display: flex; justify-content: center; flex-wrap: wrap; gap: 1rem;">
                <span style="background: linear-gradient(120deg, #ffd700, #ffed4e); padding: 4px 8px; border-radius: 4px; color: #333; font-weight: 600;">Primary Match</span>
                <span style="background: linear-gradient(120deg, #ff8c00, #ffa500); padding: 4px 8px; border-radius: 4px; color: white; font-weight: 500;">Conceptual</span>
                <span style="background: linear-gradient(120deg, #87ceeb, #add8e6); padding: 4px 8px; border-radius: 4px; color: #333; font-style: italic;">Weak Link</span>
                <span style="background: linear-gradient(120deg, #90ee90, #98fb98); padding: 4px 8px; border-radius: 4px; color: #333;">Supporting Context</span>
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