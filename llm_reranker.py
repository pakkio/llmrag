import os
import logging
import re
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path

from llm_wrapper import llm_call

try:
    from dotenv import load_dotenv
    
    script_dir = Path(__file__).parent
    env_paths = [
        Path.cwd() / ".env",
        script_dir / ".env"
    ]
    
    for env_path in env_paths:
        if env_path.exists():
            load_dotenv(env_path)
            if os.environ.get('PAK_DEBUG') == 'true':
                print(f"llm_reranker: Loaded .env from {env_path}")
            break
    else:
        if os.environ.get('PAK_DEBUG') == 'true':
            print(f"llm_reranker: No .env file found in {[str(p) for p in env_paths]}")
            
except ImportError:
    pass

# Configuration constants
RERANKING_CONFIG = {
    "min_candidates": 8,      # Minimum candidates needed for reranking
    "max_candidates": 25,     # Maximum candidates to send to LLM (cost control)
    "fallback_timeout": 10.0, # Timeout for LLM reranking (seconds)
    "default_model": "google/gemini-flash-1.5",  # Fast and cost-effective
    "max_text_length": 300,   # Max text length per candidate in prompt
}

def build_reranking_prompt(query: str, candidates: List[Tuple[str, int, str, float]], language: str = "auto") -> str:
    """
    Build a sophisticated reranking prompt for LLM evaluation
    
    Args:
        query: Original search query
        candidates: List of (pdf_name, page_num, text, score) tuples
        language: Language for instructions (auto, english, italian, spanish, french)
    
    Returns:
        Formatted prompt string for LLM reranking
    """
    
    # Language-specific instructions
    language_prompts = {
        "italian": {
            "intro": f'Riordina questi risultati di ricerca per rilevanza rispetto alla query: "{query}"',
            "criteria": """Criteri di ranking (in ordine di importanza):
1. **Rilevanza diretta**: Quanto il contenuto risponde direttamente alla query
2. **Completezza**: Informazioni complete e dettagliate vs parziali o vaghe
3. **Specificità**: Dettagli specifici e concreti vs informazioni generiche
4. **Autorevolezza**: Definizioni primarie vs riferimenti secondari
5. **Contesto**: Informazioni contestualizzate vs isolate""",
            "instruction": "Rispondi SOLO con l'ordine di ranking preferito come lista numerata:",
            "no_explanation": "Non fornire spiegazioni, solo la lista ordinata dal più rilevante al meno rilevante."
        },
        "english": {
            "intro": f'Rerank these search results by relevance to the query: "{query}"',
            "criteria": """Ranking criteria (in order of importance):
1. **Direct relevance**: How well the content directly answers the query
2. **Completeness**: Complete and detailed information vs partial or vague
3. **Specificity**: Specific concrete details vs generic information
4. **Authority**: Primary definitions vs secondary references
5. **Context**: Contextualized information vs isolated facts""",
            "instruction": "Respond ONLY with the preferred ranking order as a numbered list:",
            "no_explanation": "Do not provide explanations, only the ordered list from most to least relevant."
        },
        "spanish": {
            "intro": f'Reordena estos resultados de búsqueda por relevancia respecto a la consulta: "{query}"',
            "criteria": """Criterios de ranking (en orden de importancia):
1. **Relevancia directa**: Qué tan bien el contenido responde directamente a la consulta
2. **Completitud**: Información completa y detallada vs parcial o vaga
3. **Especificidad**: Detalles específicos y concretos vs información genérica
4. **Autoridad**: Definiciones primarias vs referencias secundarias
5. **Contexto**: Información contextualizada vs aislada""",
            "instruction": "Responde SOLO con el orden de ranking preferido como lista numerada:",
            "no_explanation": "No proporciones explicaciones, solo la lista ordenada de más a menos relevante."
        },
        "french": {
            "intro": f'Reclassez ces résultats de recherche par pertinence par rapport à la requête: "{query}"',
            "criteria": """Critères de classement (par ordre d'importance):
1. **Pertinence directe**: Dans quelle mesure le contenu répond directement à la requête
2. **Complétude**: Informations complètes et détaillées vs partielles ou vagues
3. **Spécificité**: Détails spécifiques et concrets vs informations génériques
4. **Autorité**: Définitions primaires vs références secondaires
5. **Contexte**: Informations contextualisées vs isolées""",
            "instruction": "Répondez UNIQUEMENT avec l'ordre de classement préféré sous forme de liste numérotée:",
            "no_explanation": "Ne fournissez pas d'explications, seulement la liste ordonnée du plus au moins pertinent."
        }
    }
    
    # Default to English if language not supported or auto
    lang_key = language.lower() if language.lower() in language_prompts else "english"
    prompts = language_prompts[lang_key]
    
    # Build the prompt
    prompt = f"""{prompts['intro']}

{prompts['criteria']}

Candidati da riordinare:
"""
    
    # Add candidates with truncated text
    for i, (pdf_name, page_num, text, score) in enumerate(candidates, 1):
        # Truncate text for prompt efficiency
        truncated_text = text[:RERANKING_CONFIG["max_text_length"]]
        if len(text) > RERANKING_CONFIG["max_text_length"]:
            truncated_text += "..."
        
        # Clean text for better LLM processing
        truncated_text = re.sub(r'\s+', ' ', truncated_text.strip())
        
        prompt += f"\n[{i}] {pdf_name} p.{page_num} (hybrid_score: {score:.3f})\n"
        prompt += f"Contenuto: {truncated_text}\n"
    
    prompt += f"""
{prompts['instruction']}
1. [numero_candidato]
2. [numero_candidato]
3. [numero_candidato]
...

{prompts['no_explanation']}"""
    
    return prompt

def parse_ranking_response(response: str, original_candidates: List[Tuple]) -> List[Tuple]:
    """
    Parse LLM ranking response and reconstruct ordered results
    
    Args:
        response: LLM response with numbered ranking
        original_candidates: Original candidate list
    
    Returns:
        Reordered list of candidates based on LLM ranking
    """
    try:
        # Extract numbers from response using regex
        number_pattern = r'^\s*(\d+)\.\s*\[?(\d+)\]?'
        rankings = []
        
        for line in response.strip().split('\n'):
            match = re.match(number_pattern, line.strip())
            if match:
                rank_position = int(match.group(1))
                candidate_number = int(match.group(2))
                rankings.append((rank_position, candidate_number))
        
        # Sort by rank position and validate candidate numbers
        rankings.sort(key=lambda x: x[0])
        reordered_candidates = []
        used_indices = set()
        
        for rank_pos, candidate_num in rankings:
            # Convert to 0-based index
            candidate_index = candidate_num - 1
            
            # Validate index and avoid duplicates
            if (0 <= candidate_index < len(original_candidates) and 
                candidate_index not in used_indices):
                reordered_candidates.append(original_candidates[candidate_index])
                used_indices.add(candidate_index)
        
        # Add any remaining candidates that weren't ranked
        for i, candidate in enumerate(original_candidates):
            if i not in used_indices:
                reordered_candidates.append(candidate)
        
        return reordered_candidates
        
    except Exception as e:
        logging.warning(f"Failed to parse LLM ranking response: {e}")
        logging.debug(f"Response was: {response}")
        # Return original order if parsing fails
        return original_candidates

def llm_rerank_results(query: str, 
                      candidates: List[Tuple[str, int, str, float]], 
                      max_candidates: Optional[int] = None,
                      language: str = "auto",
                      model: Optional[str] = None) -> List[Tuple[str, int, str, float]]:
    """
    Rerank search results using LLM understanding of relevance
    
    Args:
        query: Original search query
        candidates: List of (pdf_name, page_num, text, score) tuples
        max_candidates: Maximum candidates to rerank (cost control)
        language: Language for LLM instructions
        model: LLM model to use for reranking
    
    Returns:
        Reordered list of candidates based on LLM assessment
    """
    
    # Validate input
    if not candidates:
        return []
    
    # Apply max_candidates limit
    if max_candidates is None:
        max_candidates = RERANKING_CONFIG["max_candidates"]
    
    candidates_to_rank = candidates[:max_candidates]
    remaining_candidates = candidates[max_candidates:] if len(candidates) > max_candidates else []
    
    # Check if we have enough candidates to justify reranking
    if len(candidates_to_rank) < RERANKING_CONFIG["min_candidates"]:
        logging.info(f"Skipping reranking: only {len(candidates_to_rank)} candidates (min: {RERANKING_CONFIG['min_candidates']})")
        return candidates
    
    try:
        # Build reranking prompt
        prompt = build_reranking_prompt(query, candidates_to_rank, language)
        
        # Prepare LLM call
        messages = [{"role": "user", "content": prompt}]
        
        # Use specified model or default
        rerank_model = model or RERANKING_CONFIG["default_model"]
        
        # Call LLM for reranking
        logging.info(f"LLM reranking {len(candidates_to_rank)} candidates using {rerank_model}")
        response, success = llm_call(
            messages, 
            model=rerank_model, 
            max_tokens=1000,
            temperature=0.1  # Low temperature for consistent ranking
        )
        
        if not success:
            logging.warning("LLM reranking failed, returning original order")
            return candidates
        
        # Parse response and reorder candidates
        reranked_candidates = parse_ranking_response(response, candidates_to_rank)
        
        # Add remaining candidates at the end
        final_results = reranked_candidates + remaining_candidates
        
        logging.info(f"LLM reranking completed successfully for {len(candidates_to_rank)} candidates")
        
        return final_results
        
    except Exception as e:
        logging.error(f"LLM reranking failed: {e}")
        return candidates

def test_reranker():
    """Test the LLM reranker with sample data"""
    
    # Sample test data
    test_query = "equinozio primavera"
    test_candidates = [
        ("astronomy_book.pdf", 24, "L'equinozio di primavera segna il momento in cui il sole attraversa l'equatore celeste. Questo fenomeno astronomico determina l'inizio della stagione primaverile nell'emisfero settentrionale.", 0.85),
        ("general_science.pdf", 156, "Gli equinozi sono due momenti dell'anno in cui il giorno e la notte hanno la stessa durata. Il termine deriva dal latino aequinoctium.", 0.78),
        ("calendar_history.pdf", 89, "Il calendario gregoriano tiene conto della precessione degli equinozi per mantenere allineate le stagioni con le date.", 0.72),
        ("physics_textbook.pdf", 445, "La meccanica celeste studia i moti dei corpi celesti sotto l'influenza delle forze gravitazionali.", 0.65),
        ("weather_patterns.pdf", 12, "I cambiamenti stagionali sono dovuti all'inclinazione dell'asse terrestre rispetto al piano dell'orbita.", 0.82)
    ]
    
    print("Testing LLM reranker...")
    print(f"Query: {test_query}")
    print(f"Original candidates: {len(test_candidates)}")
    
    # Test reranking
    reranked = llm_rerank_results(test_query, test_candidates, language="italian")
    
    print("\nOriginal order:")
    for i, (pdf, page, text, score) in enumerate(test_candidates, 1):
        print(f"{i}. {pdf} p.{page} (score: {score:.3f})")
        print(f"   {text[:100]}...")
    
    print("\nReranked order:")
    for i, (pdf, page, text, score) in enumerate(reranked, 1):
        print(f"{i}. {pdf} p.{page} (original_score: {score:.3f})")
        print(f"   {text[:100]}...")
    
    return reranked

if __name__ == "__main__":
    # Enable debug logging
    logging.basicConfig(level=logging.INFO)
    
    # Test the reranker
    test_reranker()