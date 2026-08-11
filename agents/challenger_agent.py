from llm_client import call_llm_structured
from models import AgentVerdict, PaperInput, ChallengeOutput

SYSTEM_PROMPT = """You are a devil's-advocate reviewer. You are given a summary \
of another reviewer's verdict on a research paper, along with the paper itself. \
Your job is to challenge that verdict -- find gaps, overclaims, or missed \
issues in the original review. You are not reviewing the paper from scratch; \
you are critiquing the REVIEW.

Rules:
- If the original review genuinely missed something, name it specifically.
- If the original review is actually solid and you can't find a real gap, say \
so honestly -- do not manufacture a fake objection just to seem thorough. \
Your challenger_confidence should be LOW (below 0.3) in that case.
- Be specific and reference the original review's actual claims, not generic \
critique.

IMPORTANT: Your output uses a DIFFERENT schema than the review you are shown. \
Do not copy the review's field names (agent_name, summary, strengths, \
weaknesses, confidence). Respond with ONLY these two fields, exactly, and \
nothing else -- no markdown, no extra keys, no commentary:
{
  "challenge": "your specific critique of the original review, or a note that it holds up",
  "challenger_confidence": 0.0-1.0
}
"""


def _verdict_as_plain_text(v: AgentVerdict) -> str:
    """Render the verdict as prose rather than JSON, so a small local model
    isn't tempted to just echo the same JSON shape back as its own output."""
    lines = [f"Summary: {v.summary}"]
    if v.strengths:
        lines.append("Strengths noted: " + "; ".join(v.strengths))
    if v.weaknesses:
        lines.append("Weaknesses noted: " + "; ".join(v.weaknesses))
    lines.append(f"Reviewer's self-rated confidence: {v.confidence}")
    if v.needs_clarification:
        lines.append(f"Reviewer's open question: {v.needs_clarification}")
    return "\n".join(lines)


def run(original_verdict: AgentVerdict, paper: PaperInput) -> ChallengeOutput:
    user_prompt = f"""Paper title: {paper.title}
Paper abstract: {paper.abstract or '(not provided)'}
Paper draft: {paper.draft_text or '(not provided)'}

--- Original review to challenge (plain-text summary, not the schema to copy) ---
{_verdict_as_plain_text(original_verdict)}

Now produce your challenge using ONLY the "challenge" and "challenger_confidence" fields.
"""
    return call_llm_structured(SYSTEM_PROMPT, user_prompt, ChallengeOutput, label="challenger")