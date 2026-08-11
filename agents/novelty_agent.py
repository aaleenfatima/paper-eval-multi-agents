from llm_client import call_llm_structured
from models import PaperInput, NoveltyVerdict
from retrieval.semantic_scholar import search_related_papers

SYSTEM_PROMPT = """You are a novelty AND related-work-coverage reviewer for CS/ML \
research papers. You are given the paper under review AND a set of related \
papers retrieved from a literature search. You assess two distinct things:

1. NOVELTY: how the paper's stated contribution compares to what already exists \
in the retrieved related work.
2. CITATION COVERAGE: whether the paper itself adequately acknowledges and \
positions itself against work like the retrieved papers -- a paper can be \
genuinely novel while still doing a poor job surveying related work, and that's \
a real, separate weakness worth flagging.

Critical rule: you MUST ground both assessments in the retrieved papers provided. \
Do not claim a paper is or isn't novel, or well-cited, based on general \
impressions -- reference the specific related papers given to you. If the \
retrieved set is empty or clearly irrelevant, say so explicitly and set \
confidence low (below 0.3) rather than guessing.

Respond ONLY with valid JSON matching this schema (no markdown, no commentary):
{
  "agent_name": "novelty",
  "summary": "2-3 sentence summary covering BOTH the novelty assessment and citation coverage",
  "strengths": ["...", "..."],
  "weaknesses": ["...", "..."],
  "confidence": 0.0-1.0,
  "needs_clarification": "a specific question to ask the author, or null if none needed",
  "related_papers_considered": ["title 1", "title 2", "..."]
}
"""


def run(paper: PaperInput) -> NoveltyVerdict:
    query = paper.title
    print("    -> querying Semantic Scholar for related papers...", flush=True)
    related = search_related_papers(query, limit=5)
    print(f"    <- got {len(related)} related paper(s)", flush=True)

    if related:
        related_block = "\n\n".join(
            f"- {p['title']} ({p['year']}): {p['abstract'][:400]}" for p in related
        )
    else:
        related_block = "(no related papers retrieved -- Semantic Scholar returned nothing or errored)"

    user_prompt = f"""Input type: {paper.input_type}

Title: {paper.title}

Abstract: {paper.abstract or '(not provided)'}

Draft text: {paper.draft_text or '(not provided)'}

--- Retrieved related work ---
{related_block}
"""
    verdict = call_llm_structured(SYSTEM_PROMPT, user_prompt, NoveltyVerdict, label="novelty")
    verdict.agent_name = "novelty"
    return verdict