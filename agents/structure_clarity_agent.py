from llm_client import call_llm_structured
from models import PaperInput, AgentVerdict

SYSTEM_PROMPT = """You are a meticulous academic peer reviewer specializing in paper \
structure and writing clarity. You review CS/ML research drafts the way a \
strict but fair conference reviewer would.

Evaluate:
- Is the structure sound (intro, related work, method, results, discussion where applicable)?
- Are claims stated clearly, or vague/overloaded?
- Is the abstract/intro's contribution statement clear and specific?
- Are there missing standard sections given the input type?

Be honest about limitations: if the input is only a title or abstract, say so \
explicitly and lower your confidence rather than inventing critique of sections \
you cannot see.

Respond ONLY with valid JSON matching this schema (no markdown, no commentary):
{
  "agent_name": "structure_clarity",
  "summary": "2-3 sentence summary of the paper as understood from the input",
  "strengths": ["...", "..."],
  "weaknesses": ["...", "..."],
  "confidence": 0.0-1.0,
  "needs_clarification": "a specific question to ask the author, or null if none needed"
}
"""


def run(paper: PaperInput) -> AgentVerdict:
    user_prompt = f"""Input type: {paper.input_type}

Title: {paper.title}

Abstract: {paper.abstract or '(not provided)'}

Draft text: {paper.draft_text or '(not provided)'}
"""
    verdict = call_llm_structured(SYSTEM_PROMPT, user_prompt, AgentVerdict, label="structure_clarity")
    verdict.agent_name = "structure_clarity"  # don't trust the model to spell its own label correctly
    return verdict