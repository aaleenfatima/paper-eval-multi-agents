"""
LLM-as-judge: compares PaperPilot's flagged weaknesses against the real human
reviewer's comments for the same paper, and scores how well they agree.

This is deliberately a SEPARATE model call, not something PaperPilot's own
agents do -- an agent grading its own homework is a well-known bias risk
(self-preference bias). Using the same underlying model family as judge and
subject isn't ideal either, but it's the honest, defensible choice given
local-only constraints; this is worth naming explicitly as a limitation in
your writeup rather than glossing over.
"""

from pydantic import BaseModel, Field, field_validator
from llm_client import call_llm_structured
from models import _coerce_confidence  # reuse the same word->number defensive coercion


class JudgeVerdict(BaseModel):
    agreement_score: float = Field(ge=0.0, le=1.0, description="how well PaperPilot's critique overlaps with the real human review, 0=no overlap, 1=captures the same core issues")
    matched_criticisms: list[str] = Field(default_factory=list, description="issues both PaperPilot and the human reviewer raised")
    missed_by_paperpilot: list[str] = Field(default_factory=list, description="real issues the human reviewer raised that PaperPilot did NOT catch")
    paperpilot_flagged_extra: list[str] = Field(default_factory=list, description="issues PaperPilot raised that the human reviewer did not mention (not necessarily wrong, just extra)")
    reasoning: str = Field(description="2-3 sentence explanation of the score")

    @field_validator("agreement_score", mode="before")
    @classmethod
    def _coerce_score(cls, v):
        return _coerce_confidence(v)


SYSTEM_PROMPT = """You are an impartial evaluator comparing an AI paper-review \
system's critique against a REAL human peer reviewer's comments on the same \
paper. Your job is ONLY to judge overlap and agreement -- not to re-review the \
paper yourself.

Read both critiques carefully. Identify:
1. Criticisms that BOTH raised (even if worded differently -- match by meaning, not exact phrasing)
2. Real issues the human reviewer raised that the AI system's critique MISSED
3. Issues the AI system raised that the human reviewer did NOT mention (this
   isn't necessarily a flaw in the AI critique -- just note it)

Score agreement_score based on whether the AI system caught the human
reviewer's MOST IMPORTANT points, not on raw count of overlapping items.
Missing the human reviewer's single biggest concern should hurt the score
more than missing a minor point they raised in passing.

Keep your lists to the 3-5 most important items each, not exhaustive. Keep
reasoning to 2-3 sentences.

Respond ONLY with valid JSON matching this schema (no markdown, no commentary):
{
  "agreement_score": 0.0-1.0,
  "matched_criticisms": ["...", "..."],
  "missed_by_paperpilot": ["...", "..."],
  "paperpilot_flagged_extra": ["...", "..."],
  "reasoning": "2-3 sentences explaining the score"
}
"""

MAX_REVIEW_CHARS = 3000  # some PeerRead papers have multiple reviewers' comments
                          # concatenated (15,000+ chars) -- the judge only needs
                          # enough to identify the reviewer's main points, not
                          # every reviewer's full text verbatim


def run(paperpilot_report_text: str, human_review_text: str) -> JudgeVerdict:
    trimmed_review = human_review_text[:MAX_REVIEW_CHARS]
    user_prompt = f"""--- AI SYSTEM'S CRITIQUE (PaperPilot) ---
{paperpilot_report_text}

--- REAL HUMAN REVIEWER'S COMMENTS ---
{trimmed_review}
"""
    return call_llm_structured(SYSTEM_PROMPT, user_prompt, JudgeVerdict, label="judge", max_tokens=500)