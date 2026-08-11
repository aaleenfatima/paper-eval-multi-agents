"""
Orchestrator, Phase 2.

Split into explicit stages so the Streamlit app can pause between them for
user input (clarifying questions) rather than running everything blind in
one shot:

  1. run_initial_agents(paper)        -> verdicts + clarifying questions
  2. incorporate_clarifications(...)  -> updated PaperInput
  3. run_debate(paper, verdicts)      -> challenged/resolved verdicts + debate log
  4. finalize(...)                    -> FinalReport via synthesis agent

This still uses the confidence-based re-invoke from Phase 1 inside stage 1.
"""

from models import PaperInput, FinalReport, DebateRound, AgentVerdict, NoveltyVerdict
from agents import (
    structure_clarity_agent,
    methodology_critic_agent,
    novelty_agent,
    challenger_agent,
    synthesis_agent,
)

CONFIDENCE_THRESHOLD = 0.45
DEBATE_TRIGGER_THRESHOLD = 0.35  # challenger confidence above this triggers a re-review


def _reinvoke_if_low_confidence(verdict, agent_module, paper: PaperInput):
    if verdict.confidence >= CONFIDENCE_THRESHOLD:
        return verdict

    # Special case: if novelty's low confidence is because retrieval genuinely
    # returned nothing, re-invoking won't help -- the model can't ground itself
    # in papers that don't exist. Re-running here is pure wasted compute (one
    # more model call + one more throttled network call), so accept the
    # single-pass verdict as-is instead.
    if isinstance(verdict, NoveltyVerdict) and not verdict.related_papers_considered:
        return verdict

    nudge_note = (
        "\n\n[ORCHESTRATOR NOTE: your previous review had low confidence "
        f"({verdict.confidence}). If the input genuinely lacks enough detail, "
        "state that clearly and explain exactly what's missing rather than "
        "guessing. If you can extract more signal from what's given, do so.]"
    )
    patched = paper.model_copy()
    patched.draft_text = (patched.draft_text or "") + nudge_note
    return agent_module.run(patched)


def run_initial_agents(paper: PaperInput):
    """Stage 1: run all three critics, re-invoking on low confidence."""
    structure_verdict = structure_clarity_agent.run(paper)
    structure_verdict = _reinvoke_if_low_confidence(structure_verdict, structure_clarity_agent, paper)

    methodology_verdict = methodology_critic_agent.run(paper)
    methodology_verdict = _reinvoke_if_low_confidence(methodology_verdict, methodology_critic_agent, paper)

    novelty_verdict = novelty_agent.run(paper)
    novelty_verdict = _reinvoke_if_low_confidence(novelty_verdict, novelty_agent, paper)

    clarifying_questions = [
        v.needs_clarification
        for v in (structure_verdict, methodology_verdict, novelty_verdict)
        if v.needs_clarification
    ]

    return {
        "structure": structure_verdict,
        "methodology": methodology_verdict,
        "novelty": novelty_verdict,
        "clarifying_questions": clarifying_questions,
    }


def rerun_agents_after_clarification(paper: PaperInput, stage1_before: dict) -> dict:
    """Re-run ONLY the agents that raised a clarifying question, not the full
    trio -- a full re-run re-sends the entire draft to all 3 agents even
    when only 1 actually needed the extra context, which is wasted cost on
    every call given how much the draft dominates prompt size."""
    structure_verdict = stage1_before["structure"]
    methodology_verdict = stage1_before["methodology"]
    novelty_verdict = stage1_before["novelty"]

    if structure_verdict.needs_clarification:
        structure_verdict = structure_clarity_agent.run(paper)
        structure_verdict = _reinvoke_if_low_confidence(structure_verdict, structure_clarity_agent, paper)

    if methodology_verdict.needs_clarification:
        methodology_verdict = methodology_critic_agent.run(paper)
        methodology_verdict = _reinvoke_if_low_confidence(methodology_verdict, methodology_critic_agent, paper)

    if novelty_verdict.needs_clarification:
        novelty_verdict = novelty_agent.run(paper)
        novelty_verdict = _reinvoke_if_low_confidence(novelty_verdict, novelty_agent, paper)

    return {
        "structure": structure_verdict,
        "methodology": methodology_verdict,
        "novelty": novelty_verdict,
        "clarifying_questions": stage1_before["clarifying_questions"],
    }


def incorporate_clarifications(paper: PaperInput, qa_pairs: list[tuple[str, str]]) -> PaperInput:
    """Stage 2: fold the author's answers back into the paper context so a
    re-run of stage 1 reflects the new information."""
    if not qa_pairs:
        return paper
    clarification_block = "\n\n[AUTHOR CLARIFICATIONS]\n" + "\n".join(
        f"Q: {q}\nA: {a}" for q, a in qa_pairs
    )
    patched = paper.model_copy()
    patched.draft_text = (patched.draft_text or "") + clarification_block
    return patched


def _debate_one(agent_name: str, verdict: AgentVerdict, agent_module, paper: PaperInput):
    """Challenge one verdict; if the challenger finds real issues, re-invoke
    the original agent with the challenge folded in and keep the revised verdict."""
    challenge = challenger_agent.run(verdict, paper)

    if challenge.challenger_confidence < DEBATE_TRIGGER_THRESHOLD:
        verdict.was_debated = True
        verdict.pre_debate_confidence = verdict.confidence
        debate_round = DebateRound(
            agent_name=agent_name,
            original_verdict=verdict,
            challenger_critique=challenge.challenge,
            resolution_summary="Challenge did not surface a substantive gap; original verdict retained.",
            final_confidence=verdict.confidence,
            confidence_changed=False,
        )
        return verdict, debate_round

    nudge = paper.model_copy()
    nudge.draft_text = (nudge.draft_text or "") + (
        f"\n\n[DEVIL'S ADVOCATE CHALLENGE to previous {agent_name} review: "
        f"{challenge.challenge}]\nRevise your review if this challenge is valid; "
        f"otherwise explain why it doesn't hold."
    )
    revised = agent_module.run(nudge)
    revised.was_debated = True
    revised.pre_debate_confidence = verdict.confidence

    debate_round = DebateRound(
        agent_name=agent_name,
        original_verdict=verdict,
        challenger_critique=challenge.challenge,
        resolution_summary=revised.summary,
        final_confidence=revised.confidence,
        confidence_changed=(revised.confidence != verdict.confidence),
    )
    return revised, debate_round


def run_debate(paper: PaperInput, methodology_verdict: AgentVerdict, novelty_verdict: AgentVerdict):
    """Stage 3: challenge the methodology and novelty verdicts -- these are
    the two most prone to overclaiming or missed gaps. Structure/clarity is
    comparatively objective and skipped to keep runtime reasonable."""
    debate_log = []

    resolved_methodology, round1 = _debate_one("methodology", methodology_verdict, methodology_critic_agent, paper)
    debate_log.append(round1)

    resolved_novelty, round2 = _debate_one("novelty", novelty_verdict, novelty_agent, paper)
    debate_log.append(round2)

    return resolved_methodology, resolved_novelty, debate_log


def finalize(paper: PaperInput, structure_verdict, methodology_verdict, novelty_verdict,
             debate_log: list[DebateRound], clarifying_questions_asked: list[str]) -> FinalReport:
    """Stage 4: synthesis agent produces the final human-readable report."""
    final_text = synthesis_agent.run(structure_verdict, methodology_verdict, novelty_verdict)

    overall_confidence = round(
        (structure_verdict.confidence + methodology_verdict.confidence + novelty_verdict.confidence) / 3,
        2,
    )

    return FinalReport(
        input_type=paper.input_type,
        structure_clarity=structure_verdict,
        methodology=methodology_verdict,
        novelty=novelty_verdict,
        debate_log=debate_log,
        clarifying_questions_asked=clarifying_questions_asked,
        final_recommendation=final_text,
        overall_confidence=overall_confidence,
    )