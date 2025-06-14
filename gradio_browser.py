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
    # Convert [[...]] to italic text inline
    clean_text = highlighted_text
    clean_text = re.sub(r'\[\[(.*?)\]\]', r'<em class="explanation-text">\1</em>', clean_text)
    
    # Convert [HIGHLIGHT] tags to HTML
    clean_text = clean_text.replace('[HIGHLIGHT]', '<span class="highlight-text">')
    clean_text = clean_text.replace('[/HIGHLIGHT]', '</span>')
    
    return clean_text, []  # Explanations are now inline, no separate list needed

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

Use [[...]] to highlight any inferred information or assumptions, and format them as italic text inline."""
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
        # Request raw tags, not ANSI codes
        highlighted_texts = highlight_relevant_text_batch(query, results, output_format_ansi=False)
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
    """Create CSS with clean, readable color scheme using white backgrounds with blue/green text"""
    return """
    <style>
    /* Light theme defaults with white backgrounds and blue/green text */
    .results-container {
        background: #ffffff;
        color: #222;
        border: 2px solid #e2e8f0;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .border-default {
        background: #ffffff;
        border: 2px solid #bbb;
        border-radius: 8px;
        margin: 1rem 0;
        overflow: hidden;
    }
    .border-content {
        background: #ffffff;
        border: 2px solid #1a73e8;
        border-radius: 8px;
        margin: 1rem 0;
        overflow: hidden;
    }
    .border-analysis {
        background: #ffffff;
        border: 2px solid #ff9800;
        border-radius: 8px;
        margin: 1rem 0;
        overflow: hidden;
    }
    .border-synthesis {
        background: #ffffff;
        border: 2px solid #00b300;
        border-radius: 8px;
        margin: 1rem 0;
        overflow: hidden.
    }
    .border-title {
        background: #1a73e8;
        color: #fff;
        padding: 0.75rem 1rem;
        font-weight: 600;
        font-size: 1.1em;
        border-bottom: 2px solid #1a73e8;
    }
    .border-content .border-title {
        background: #1a73e8;
        border-bottom: 2px solid #1a73e8;
    }
    .border-analysis .border-title {
        background: #ff9800;
        color: #222;
        border-bottom: 2px solid #ff9800;
    }
    .border-synthesis .border-title {
        background: #00b300;
        color: #fff;
        border-bottom: 2px solid #00b300;
    }
    .highlight-text {
        background-color: #e8f0fe !important;
        color: #1a73e8 !important;
        padding: 2px 4px;
        border-radius: 3px;
        font-weight: 700;
        border: 1px solid #1a73e8;
    }
    .explanation-text {
        color: #ff5722 !important;
        font-style: italic;
    }
    .logs-container {
        border: 2px dashed #888;
        padding: 1rem;
        margin: 1rem 0;
        background: #f5f5f5;
        font-family: monospace;
        white-space: pre-wrap;
        color: #222;
    }
    /* Dark theme overrides */
    @media (prefers-color-scheme: dark) {
        .results-container {
            background: #1e1e1e;
            color: #f8f8f8;
            border: 2px solid #333;
        }
        .border-default {
            background: #2d2d2d;
            border: 2px solid #444;
        }
        .border-content {
            background: #2d2d2d;
            border: 2px solid #4d90fe;
        }
        .border-analysis {
            background: #2d2d2d;
            border: 2px solid #ffb300;
        }
        .border-synthesis {
            background: #2d2d2d;
            border: 2px solid #00e676;
        }
        .border-title {
            background: #1e1e1e;
            color: #fff;
            border-bottom: 2px solid #333;
        }
        .border-content .border-title {
            background: #4d90fe;
            border-bottom: 2px solid #4d90fe;
        }
        .border-analysis .border-title {
            background: #ffb300;
            color: #1e1e1e;
            border-bottom: 2px solid #ffb300;
        }
        .border-synthesis .border-title {
            background: #00e676;
            color: #1e1e1e;
            border-bottom: 2px solid #00e676.
        }
        .highlight-text {
            background-color: #193c69 !important;
            color: #78b0fd !important;
            border: 1px solid #4d90fe;
        }
        .explanation-text {
            color: #ff7043 !important;
            font-style: italic;
        }
        .logs-container {
            border: 2px dashed #666;
            background: #2d2d2d;
            color: #fafafa;
        }
    }
    </style>
    """

def search_and_highlight(query: str, collection_names: str = "all"):
    """
    Main search function as a generator.
    Yields a tuple: (direct_answer_md, html_results, synthesis_md, logs_html)
    """
    logs = []
    direct_answer_md = ""
    html_results = ""
    synthesis_md = ""

    def get_logs_html():
        return f'<div class="logs-container"><div class="border-title">📜 Logs</div><pre>{"\n".join(logs)}</pre></div>'

    # Start: update with initial log
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
            selected_pdf_name = valid_names[0] if valid_names else None
            logs.append(f"[search_and_highlight] Selected collection: {selected_pdf_name}")
        yield (direct_answer_md, html_results, synthesis_md, get_logs_html())

        logs.append("[search_and_highlight] Generating query embedding...")
        yield (direct_answer_md, html_results, synthesis_md, get_logs_html())
        query_embedding = generate_embeddings(query, normalize=True)

        logs.append("[search_and_highlight] Querying Chroma collections...")
        yield (direct_answer_md, html_results, synthesis_md, get_logs_html())
        all_results = query_chroma_collections(query_embedding, top_k=5, pdf_name=selected_pdf_name)
        if not all_results:
            logs.append("[search_and_highlight] No results found for query.")
            yield (direct_answer_md, html_results, synthesis_md, get_logs_html())
            return

        top_results = all_results[:5]
        html_sections = []

        # PART 1: Direct Answer Section (Highly Emphasized Markdown with new CSS class)
        try:
            logs.append("[search_and_highlight] Generating direct answer...")
            yield (direct_answer_md, html_results, synthesis_md, get_logs_html())
            direct_answer = generate_direct_answer(query, top_results)
            direct_answer_md = f"""<div class="direct-answer-container">
### 🎯 Direct Answer
{direct_answer}
</div>"""
        except Exception as e:
            logs.append(f"[search_and_highlight] Error generating direct answer: {e}")
            direct_answer_md = f"""<div class="direct-answer-container">
### 🎯 Direct Answer
Error generating direct answer: {str(e)}
</div>"""
        yield (direct_answer_md, html_results, synthesis_md, get_logs_html())

        # PART 2: Detailed Analysis Section (HTML excluding synthesis)
        html_sections.append('<div class="detailed-analysis-header">🔍 Detailed Analysis &amp; Search Results</div>')
        try:
            logs.append("[search_and_highlight] Processing highlights for Gradio...")
            yield (direct_answer_md, html_results, synthesis_md, get_logs_html())
            processed_highlights = process_highlighting_for_gradio(query, top_results)
        except Exception as e:
            logs.append(f"[search_and_highlight] Error in highlighting: {e}")
            processed_highlights = [(text, []) for _, _, text, _ in top_results]
        yield (direct_answer_md, html_results, synthesis_md, get_logs_html())

        for i, (pdf_name, page_number, page_text, similarity) in enumerate(top_results, 1):
            logs.append(f"[search_and_highlight] Processing result {i}: {pdf_name} page {page_number} (similarity {similarity:.4f})")
            yield (direct_answer_md, html_results, synthesis_md, get_logs_html())
            if i - 1 < len(processed_highlights):
                highlighted_text, explanations = processed_highlights[i-1]
            else:
                highlighted_text, explanations = page_text, []

            result_html = f"""
{create_web_border(f"📄 {pdf_name} - Page {page_number} │ Similarity: {similarity:.4f} │ Rank {i}", "border-default")}
{create_web_border("📖 CONTENT", "border-content")}
{highlighted_text}
{close_web_border()}
"""
            if explanations:
                explanations_html = '<div class="explanations-list">'
                for j, explanation in enumerate(explanations, 1):
                    explanations_html += f'<div class="explanation-item"><span class="explanation-number">[{j}]</span>{explanation}</div>'
                explanations_html += '</div>'
                result_html += f"""
{create_web_border("💡 RELEVANCE ANALYSIS", "border-analysis")}
{explanations_html}
{close_web_border()}
"""
            try:
                logs.append(f"[search_and_highlight] Analyzing individual result {i}...")
                yield (direct_answer_md, html_results, synthesis_md, get_logs_html())
                individual_analysis = analyze_individual_result_web(query, page_text, pdf_name, page_number, similarity)
                result_html += f"""
{create_web_border("🧠 LLM ANALYSIS", "border-analysis")}
<div class="individual-analysis">{individual_analysis}</div>
{close_web_border()}
"""
            except Exception as e:
                logs.append(f"[search_and_highlight] Error in individual analysis for result {i}: {e}")
                result_html += f"""
{create_web_border("🧠 LLM ANALYSIS", "border-analysis")}
<div class="individual-analysis">Error generating analysis: {str(e)}</div>
{close_web_border()}
"""
            result_html += f"{close_web_border()}<div class='section-separator'></div>"
            html_sections.append(result_html)
            yield (direct_answer_md, "".join(html_sections), synthesis_md, get_logs_html())

        # PART 3: Comprehensive Synthesis as Dedicated Markdown Field
        try:
            logs.append("[search_and_highlight] Synthesizing results...")
            yield (direct_answer_md, "".join(html_sections), synthesis_md, get_logs_html())
            synthesis = synthesize_results_web(query, top_results)
            synthesis_md = f"### 🔬 Comprehensive Synthesis\n\n{synthesis}"
        except Exception as e:
            logs.append(f"[search_and_highlight] Error in synthesis: {e}")
            synthesis_md = f"### 🔬 Comprehensive Synthesis\n\nError: {str(e)}"
        yield (direct_answer_md, "".join(html_sections), synthesis_md, get_logs_html())

        summary_info = f"📊 <b>Total results:</b> {len(all_results)}"
        if top_results:
            best_result, worst_result = top_results[0], top_results[-1]
            summary_info += f" | <b>Best:</b> {best_result[0]} Page {best_result[1]} ({best_result[3]:.4f}) | <b>Worst:</b> {worst_result[0]} Page {worst_result[1]} ({worst_result[3]:.4f})"
        html_sections.append(f"<div style='margin:1em 0'>{summary_info}</div>")
        html_results = "".join(html_sections)
        yield (direct_answer_md, html_results, synthesis_md, get_logs_html())

        logs.append("[search_and_highlight] Done. Returning results.")
        yield (direct_answer_md, html_results, synthesis_md, get_logs_html())

    except Exception as e:
        logs.append(f"[search_and_highlight] Fatal error: {e}")
        yield ("", f"<div>Error: {str(e)}</div>", "", get_logs_html())


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

        # Outputs: direct answer (Markdown), detailed analysis (HTML), comprehensive synthesis (Markdown), logs (HTML)
        direct_answer_output = gr.Markdown(
            label="Direct Answer",
            value="Enter a query and click search to see results."
        )
        results_output = gr.HTML(
            label="Detailed Analysis",
            value=""
        )
        synthesis_output = gr.Markdown(
            label="Comprehensive Synthesis",
            value="Synthesis will appear here..."
        )
        logs_output = gr.HTML(
            label="Logs",
            value="<div>No logs yet.</div>"
        )

        search_button.click(
            fn=search_and_highlight,
            inputs=[query_input, collections_input],
            outputs=[direct_answer_output, results_output, synthesis_output, logs_output],
            show_progress=True  # removed stream=True
        )

        query_input.submit(
            fn=search_and_highlight,
            inputs=[query_input, collections_input],
            outputs=[direct_answer_output, results_output, synthesis_output, logs_output],
            show_progress=True  # removed stream=True
        )

        gr.HTML("""
        <div style="text-align: center; margin-top: 2rem; padding: 1rem;">
            <h3 style="color: var(--body-text-color, #666);">🎨 Highlighting Legend</h3>
            <div style="display: flex; justify-content: center; flex-wrap: wrap; gap: 1rem;">
                <span style="background: #fbbf24; padding: 4px 8px; border-radius: 4px; color: #1a202c; font-weight: 600;">Highlighted Text</span>
                <span style="color: #3182ce; font-style: italic;">Explanations in italics</span>
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