"""
Structured schemas for PaperPilot agent I/O.

Every agent returns one of these instead of free text. This is what makes
the orchestrator's job possible -- it can inspect .confidence and .flags
instead of trying to parse prose.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional

_WORD_CONFIDENCE_MAP = {
    "very low": 0.1, "low": 0.25, "moderate": 0.5, "medium": 0.5,
    "fairly high": 0.7, "high": 0.8, "very high": 0.9,
}


def _coerce_confidence(v):
    """Small local models occasionally write 'high'/'moderate' instead of a
    number despite instructions. Coerce known words to a float rather than
    hard-failing the whole pipeline run over a formatting slip."""
    if isinstance(v, (int, float)):
        return v
    if isinstance(v, str):
        cleaned = v.strip().lower()
        if cleaned in _WORD_CONFIDENCE_MAP:
            return _WORD_CONFIDENCE_MAP[cleaned]
        try:
            return float(cleaned)
        except ValueError:
            pass
    return v  # let pydantic raise its normal validation error if truly unparseable


class PaperInput(BaseModel):
    title: str
    abstract: Optional[str] = None
    draft_text: Optional[str] = None  # full draft, if provided

    @property
    def input_type(self) -> str:
        if self.draft_text:
            return "full_draft"
        if self.abstract:
            return "abstract_only"
        return "title_only"


class AgentVerdict(BaseModel):
    """Common shape every critic agent returns."""
    agent_name: str
    summary: str
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, description="agent's self-rated confidence in this verdict")
    needs_clarification: Optional[str] = None  # question to ask the author, if any
    was_debated: bool = False
    pre_debate_confidence: Optional[float] = None  # confidence before challenger revision, if debated

    @field_validator("confidence", "pre_debate_confidence", mode="before")
    @classmethod
    def _coerce_confidence_field(cls, v):
        if v is None:
            return v
        return _coerce_confidence(v)


class NoveltyVerdict(AgentVerdict):
    related_papers_considered: list[str] = Field(default_factory=list)


class ChallengeOutput(BaseModel):
    """What the devil's-advocate agent produces when challenging a verdict."""
    challenge: str
    challenger_confidence: float = Field(ge=0.0, le=1.0, description="how confident the challenger is that the original verdict has real gaps")

    @field_validator("challenger_confidence", mode="before")
    @classmethod
    def _coerce_challenger_confidence(cls, v):
        return _coerce_confidence(v)


class DebateRound(BaseModel):
    """Record of a challenge + resolution between two agent passes."""
    agent_name: str
    original_verdict: AgentVerdict
    challenger_critique: str
    resolution_summary: str
    final_confidence: float
    confidence_changed: bool


class FinalReport(BaseModel):
    input_type: str
    structure_clarity: AgentVerdict
    methodology: AgentVerdict
    novelty: NoveltyVerdict
    debate_log: list[DebateRound] = Field(default_factory=list)
    clarifying_questions_asked: list[str] = Field(default_factory=list)
    final_recommendation: str
    overall_confidence: float
