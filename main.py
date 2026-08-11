"""
Quick non-UI test runner for Phase 2. This auto-skips clarifying questions
(answers them with "(no additional information provided)") so you can smoke-test
the debate loop without typing -- use the Streamlit app for real interactive use.

Usage:
    python main.py sample_papers/example_full_draft.json
"""

import sys
import json
from models import PaperInput
import orchestrator


def main():
    if len(sys.argv) != 2:
        print("Usage: python main.py <path_to_paper_json>")
        sys.exit(1)

    with open(sys.argv[1]) as f:
        paper = PaperInput(**json.load(f))

    print(f"Running PaperPilot on: {paper.title}  (input_type={paper.input_type})\n")

    stage1 = orchestrator.run_initial_agents(paper)
    print(f"Clarifying questions raised: {len(stage1['clarifying_questions'])}")
    for q in stage1["clarifying_questions"]:
        print(f"  - {q}")

    # Auto-skip clarifications for CLI smoke-testing
    qa_pairs = [(q, "(no additional information provided)") for q in stage1["clarifying_questions"]]
    updated_paper = orchestrator.incorporate_clarifications(paper, qa_pairs)

    if qa_pairs:
        print("\nRe-running only the agents that asked questions...")
        stage1 = orchestrator.rerun_agents_after_clarification(updated_paper, stage1)

    print("\nRunning debate loop on methodology + novelty verdicts...")
    resolved_methodology, resolved_novelty, debate_log = orchestrator.run_debate(
        updated_paper, stage1["methodology"], stage1["novelty"]
    )

    for round_ in debate_log:
        print(f"\n--- Debate: {round_.agent_name} ---")
        print(f"Challenge: {round_.challenger_critique}")
        print(f"Confidence changed: {round_.confidence_changed} (final: {round_.final_confidence})")

    report = orchestrator.finalize(
        updated_paper,
        stage1["structure"],
        resolved_methodology,
        resolved_novelty,
        debate_log,
        stage1["clarifying_questions"],
    )

    print("\n" + "=" * 70)
    print(f"FINAL REPORT (overall confidence: {report.overall_confidence})")
    print("=" * 70)
    print(report.final_recommendation)


if __name__ == "__main__":
    main()
