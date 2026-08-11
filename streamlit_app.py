"""
PaperPilot Streamlit UI, Phase 2.

Flow:
  1. Author submits title/abstract/draft
  2. Stage 1 agents run; if any raised a clarifying question, show input boxes
     for the author to answer before continuing
  3. Once clarifications are answered (or skipped), stage 1 re-runs with the
     new context, then the debate loop challenges methodology + novelty
  4. Final synthesis report is shown, along with a visible trace of the
     debate log so the agentic behavior isn't hidden inside a black box
"""

import streamlit as st
from models import PaperInput
import orchestrator

st.set_page_config(page_title="PaperPilot", layout="centered")
st.title("PaperPilot")
st.caption("A multi-agent pre-submission reviewer for research drafts")

if "stage" not in st.session_state:
    st.session_state.stage = "input"  # input -> clarify -> done


def reset():
    for key in ["stage", "paper", "stage1", "report"]:
        st.session_state.pop(key, None)
    st.session_state.stage = "input"


# ---------- Stage: input ----------
if st.session_state.stage == "input":
    title = st.text_input("Paper title", placeholder="e.g. GazeGuard: Detecting Interview Anxiety via Multimodal Features")
    abstract = st.text_area("Abstract (optional)", height=100)
    draft = st.text_area("Full draft text (optional)", height=250,
                          placeholder="Paste your introduction, method, results, etc.")

    if st.button("Run PaperPilot", type="primary"):
        if not title.strip():
            st.error("Title is required.")
        else:
            paper = PaperInput(
                title=title.strip(),
                abstract=abstract.strip() or None,
                draft_text=draft.strip() or None,
            )
            with st.spinner("Running structure, methodology, and novelty review..."):
                stage1 = orchestrator.run_initial_agents(paper)
            st.session_state.paper = paper
            st.session_state.stage1 = stage1
            st.session_state.stage = "clarify" if stage1["clarifying_questions"] else "debate"
            st.rerun()

# ---------- Stage: clarify ----------
elif st.session_state.stage == "clarify":
    st.subheader("A few clarifying questions before finishing the review")
    st.caption("The reviewer agents flagged these as underspecified. Answer what you can, or leave blank to skip.")

    questions = st.session_state.stage1["clarifying_questions"]
    answers = []
    for i, q in enumerate(questions):
        ans = st.text_area(q, key=f"clarify_{i}", height=80)
        answers.append((q, ans.strip() if ans.strip() else "(no additional information provided)"))

    col1, col2 = st.columns(2)
    if col1.button("Submit answers", type="primary"):
        updated_paper = orchestrator.incorporate_clarifications(st.session_state.paper, answers)
        with st.spinner("Re-running the agents that asked questions..."):
            stage1 = orchestrator.rerun_agents_after_clarification(updated_paper, st.session_state.stage1)
        st.session_state.paper = updated_paper
        st.session_state.stage1 = stage1
        st.session_state.stage = "debate"
        st.rerun()
    if col2.button("Skip all"):
        st.session_state.stage = "debate"
        st.rerun()

# ---------- Stage: debate + finalize (runs automatically) ----------
elif st.session_state.stage == "debate":
    with st.spinner("Running debate loop on methodology and novelty verdicts..."):
        resolved_methodology, resolved_novelty, debate_log = orchestrator.run_debate(
            st.session_state.paper,
            st.session_state.stage1["methodology"],
            st.session_state.stage1["novelty"],
        )
        report = orchestrator.finalize(
            st.session_state.paper,
            st.session_state.stage1["structure"],
            resolved_methodology,
            resolved_novelty,
            debate_log,
            st.session_state.stage1["clarifying_questions"],
        )
    st.session_state.report = report
    st.session_state.stage = "done"
    st.rerun()

# ---------- Stage: done ----------
elif st.session_state.stage == "done":
    report = st.session_state.report

    st.metric("Overall confidence", f"{report.overall_confidence:.2f}")

    st.subheader("Final report")
    st.markdown(report.final_recommendation)

    with st.expander("Agent reasoning trace (debate log)"):
        if not report.debate_log:
            st.caption("No debate rounds were triggered.")
        for round_ in report.debate_log:
            st.markdown(f"**{round_.agent_name.title()}**")
            st.markdown(f"- Original confidence: `{round_.original_verdict.confidence}`")
            st.markdown(f"- Challenger's critique: {round_.challenger_critique}")
            st.markdown(f"- Resolution: {round_.resolution_summary}")
            st.markdown(f"- Final confidence: `{round_.final_confidence}` "
                         f"({'changed' if round_.confidence_changed else 'unchanged'})")
            st.divider()

    with st.expander("Individual agent verdicts (raw)"):
        st.markdown("**Structure & clarity**")
        st.json(report.structure_clarity.model_dump())
        st.markdown("**Methodology**")
        st.json(report.methodology.model_dump())
        st.markdown("**Novelty**")
        st.json(report.novelty.model_dump())

    st.button("Review another paper", on_click=reset)
