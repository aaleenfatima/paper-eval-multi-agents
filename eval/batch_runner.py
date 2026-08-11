"""
Batch eval runner. Runs the FULL PaperPilot pipeline (including debate) on
every paper in eval/data/, checkpointing each paper's result to its own file
in eval/results/ as soon as it completes.

This is resumable by design: if the run crashes, gets killed, or you close
your laptop halfway through, just re-run this script -- it skips any
paper_id that already has a result file and picks up where it left off.
Nothing is lost except whatever paper was mid-flight when it stopped.

Usage:
    python eval/batch_runner.py
    python eval/batch_runner.py --limit 5      # quick test on first 5 papers
    python eval/batch_runner.py --force         # re-run everything, ignore existing results

Clarifying questions are auto-answered with a placeholder rather than
prompted interactively -- there's no human in the loop for a 30-50 paper
batch. This is a real, honest limitation worth naming in your writeup:
the eval measures PaperPilot's ungrounded first-pass-plus-debate behavior,
not its multi-turn clarification behavior, since that needs a real author.
"""

import sys
import json
import argparse
import time
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # so `import orchestrator` works regardless of cwd

import orchestrator
from eval.data_loader import load_eval_papers

RESULTS_DIR = Path(__file__).resolve().parent / "results"
DATA_DIR = Path(__file__).resolve().parent / "data"

AUTO_CLARIFICATION_ANSWER = "(no additional information provided -- automated eval run, no human author available)"


def _result_path(paper_id: str) -> Path:
    return RESULTS_DIR / f"{paper_id}.json"


def _already_done(paper_id: str) -> bool:
    return _result_path(paper_id).exists()


def run_one_paper(paper_id: str, paper_input, human_review_text: str) -> dict:
    """Runs the full pipeline on one paper and returns a JSON-serializable result dict."""
    t0 = time.time()

    stage1 = orchestrator.run_initial_agents(paper_input)

    qa_pairs = [(q, AUTO_CLARIFICATION_ANSWER) for q in stage1["clarifying_questions"]]
    updated_paper = orchestrator.incorporate_clarifications(paper_input, qa_pairs)
    if qa_pairs:
        stage1 = orchestrator.rerun_agents_after_clarification(updated_paper, stage1)

    resolved_methodology, resolved_novelty, debate_log = orchestrator.run_debate(
        updated_paper, stage1["methodology"], stage1["novelty"]
    )

    report = orchestrator.finalize(
        updated_paper,
        stage1["structure"],
        resolved_methodology,
        resolved_novelty,
        debate_log,
        stage1["clarifying_questions"],
    )

    elapsed = round(time.time() - t0, 1)

    return {
        "paper_id": paper_id,
        "title": paper_input.title,
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": elapsed,
        "human_review_text": human_review_text,
        "report": report.model_dump(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="only process the first N papers (for quick testing)")
    parser.add_argument("--force", action="store_true", help="re-process papers even if a result already exists")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)

    papers = load_eval_papers(str(DATA_DIR))
    if args.limit:
        papers = papers[:args.limit]

    print(f"Loaded {len(papers)} papers from {DATA_DIR}")

    done_count = sum(1 for p in papers if _already_done(p["paper_id"])) if not args.force else 0
    if done_count:
        print(f"{done_count} already have results and will be SKIPPED (use --force to re-run them)")

    remaining = [p for p in papers if args.force or not _already_done(p["paper_id"])]
    print(f"{len(remaining)} papers to process this run.\n")

    for i, p in enumerate(remaining, 1):
        paper_id = p["paper_id"]
        print(f"[{i}/{len(remaining)}] {paper_id} -- {p['paper_input'].title[:60]}...", flush=True)

        try:
            result = run_one_paper(paper_id, p["paper_input"], p["human_review_text"])
            with open(_result_path(paper_id), "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)
            print(f"    done in {result['elapsed_seconds']}s, saved -> {_result_path(paper_id).name}")
        except Exception as e:
            # Log the failure but keep going -- one bad paper shouldn't kill a multi-hour run.
            error_path = RESULTS_DIR / f"{paper_id}.ERROR.txt"
            with open(error_path, "w", encoding="utf-8") as f:
                f.write(f"{datetime.now(timezone.utc).isoformat()}\n{repr(e)}")
            print(f"    FAILED: {e}  (logged to {error_path.name}, continuing to next paper)")

    print("\nBatch run complete.")
    print(f"Results directory: {RESULTS_DIR}")
    print("Re-run this script anytime to pick up any papers that failed or were skipped due to interruption.")


if __name__ == "__main__":
    main()
