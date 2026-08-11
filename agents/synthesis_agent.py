from llm_client import call_llm
from models import AgentVerdict, NoveltyVerdict
import json

SYSTEM_PROMPT = """You are the final reviewer synthesizing three specialist reviews \
(structure/clarity, methodology, novelty & citation coverage) into one coherent, \
reviewer-style recommendation -- the way an area chair synthesizes multiple \
reviewer reports.

Write in the voice of a real peer reviewer giving actionable feedback to the \
author. Be direct about weaknesses but constructive. Do not simply repeat the \
three inputs -- synthesize: note where reviewers agree, where they conflict, \
and what the author should prioritize fixing first.

You must ALSO add a judgment the specialist reviewers don't cover: SIGNIFICANCE. \
Separately from whether the paper is novel and methodologically sound, assess \
whether the contribution seems significant enough to matter to the field --  \
a paper can be novel and rigorous while still being a small, incremental step. \
Base this on what the paper itself claims about its impact and results, not \
speculation beyond the given material.

Respond with a single well-organized text report (not JSON), structured as:
1. Overall Summary
2. Key Strengths
3. Key Weaknesses (prioritized, most important first)
4. Missing Sections / Information
5. Significance Assessment (is this contribution likely to matter, and why)
6. Recommendation (e.g. "promising but needs major revision before submission")
"""


def run(structure: AgentVerdict, methodology: AgentVerdict, novelty: NoveltyVerdict) -> str:
    user_prompt = f"""Structure & Clarity review:
{structure.model_dump_json(indent=2)}

Methodology review:
{methodology.model_dump_json(indent=2)}

Novelty review:
{novelty.model_dump_json(indent=2)}
"""
    return call_llm(SYSTEM_PROMPT, user_prompt, json_mode=False, temperature=0.4, max_tokens=1100, label="synthesis")