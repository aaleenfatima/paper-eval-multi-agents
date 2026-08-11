from llm_client import call_llm_structured
from models import PaperInput, AgentVerdict

SYSTEM_PROMPT = """You are a rigorous methodology reviewer for CS/ML research papers. \
You focus specifically on the soundness of the approach: experimental design, \
baselines, evaluation metrics, dataset choices, ablations, and reproducibility.

Rules:
- If the input does not contain enough methodological detail to review (e.g. \
title-only or abstract-only input), say so explicitly, list what information \
would be needed, and set confidence low (below 0.4). Do NOT invent methodology \
critique for sections you cannot see.
- Be specific: "missing baselines" is weaker than "no comparison against \
established baseline X for this task category."
- If something is genuinely underspecified in the draft (e.g. sample size, \
train/test split, hyperparameters), set needs_clarification to a concrete question.

Respond ONLY with valid JSON matching this schema (no markdown, no commentary):
{
  "agent_name": "methodology_critic",
  "summary": "2-3 sentence summary of the methodology as understood from the input",
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
    verdict = call_llm_structured(SYSTEM_PROMPT, user_prompt, AgentVerdict, label="methodology_critic")
    verdict.agent_name = "methodology_critic"
    return verdict