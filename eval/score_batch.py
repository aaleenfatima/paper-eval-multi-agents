"""
Scores every completed PaperPilot result in eval/results/ against its real
human review, using an LLM-as-judge. Checkpointed per paper (same pattern as
batch_runner.py) so it's safe to run now, safe to interrupt, and safe to
re-run later after any result gets updated (e.g. once novelty retrieval is
backfilled) -- just delete that paper's score file and re-run.

Usage:
    python eval/score_batch.py
    python eval/score_batch.py --limit 5
    python eval/score_batch.py --force
    python eval/score_batch.py --rescore iclr2017dev_316   # re-score one specific paper
"""

import sys
import json
import argparse
import time
import statistics
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.judge_agent import run as run_judge

RESULTS_DIR = Path(__file__).resolve().parent / "results"
SCORES_DIR = Path(__file__).resolve().parent / "scores"


def _score_path(paper_id: str) -> Path:
    return SCORES_DIR / f"{paper_id}.json"


def _build_paperpilot_critique_text(report: dict) -> str:
    """Flattens PaperPilot's structured verdicts into one text block for the
    judge -- using the raw weaknesses lists (not just the prose synthesis
    report) so the judge is comparing actual flagged issues, not prose style."""
    parts = [f"Final recommendation:\n{report['final_recommendation']}\n"]
    for agent_key in ["structure_clarity", "methodology", "novelty"]:
        v = report[agent_key]
        parts.append(f"\n{agent_key} weaknesses flagged: {'; '.join(v['weaknesses']) if v['weaknesses'] else '(none)'}")
    return "\n".join(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true", help="re-score papers that already have a score file")
    parser.add_argument("--rescore", type=str, default=None, help="re-score just this one paper_id")
    args = parser.parse_args()

    SCORES_DIR.mkdir(exist_ok=True)

    result_files = sorted(RESULTS_DIR.glob("*.json"))
    result_files = [f for f in result_files if not f.name.endswith(".ERROR.txt")]

    if args.rescore:
        result_files = [f for f in result_files if f.stem == args.rescore]
        if not result_files:
            print(f"No result found for paper_id '{args.rescore}'")
            return

    if args.limit:
        result_files = result_files[:args.limit]

    print(f"Found {len(result_files)} completed results in {RESULTS_DIR}")

    to_process = []
    for f in result_files:
        if args.force or args.rescore or not _score_path(f.stem).exists():
            to_process.append(f)

    skipped = len(result_files) - len(to_process)
    if skipped:
        print(f"{skipped} already scored, skipping (use --force to re-score everything)")
    print(f"{len(to_process)} to score this run.\n")

    for i, f in enumerate(to_process, 1):
        with open(f, encoding="utf-8") as fh:
            result = json.load(fh)

        paper_id = result["paper_id"]
        human_review = result.get("human_review_text", "")

        if not human_review.strip():
            print(f"[{i}/{len(to_process)}] {paper_id} -- SKIPPING, no human_review_text present")
            continue

        print(f"[{i}/{len(to_process)}] {paper_id} -- {result['title'][:50]}...", flush=True)

        critique_text = _build_paperpilot_critique_text(result["report"])

        try:
            t0 = time.time()
            judge_verdict = run_judge(critique_text, human_review)
            elapsed = round(time.time() - t0, 1)

            score_record = {
                "paper_id": paper_id,
                "scored_at": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": elapsed,
                "judge_verdict": judge_verdict.model_dump(),
            }
            with open(_score_path(paper_id), "w", encoding="utf-8") as out:
                json.dump(score_record, out, indent=2)

            print(f"    agreement_score={judge_verdict.agreement_score} ({elapsed}s)")
        except Exception as e:
            error_path = SCORES_DIR / f"{paper_id}.ERROR.txt"
            with open(error_path, "w", encoding="utf-8") as out:
                out.write(f"{datetime.now(timezone.utc).isoformat()}\n{repr(e)}")
            print(f"    FAILED: {e} (logged, continuing)")

    # Aggregate summary over everything scored so far (not just this run)
    all_scores = []
    for f in SCORES_DIR.glob("*.json"):
        with open(f, encoding="utf-8") as fh:
            all_scores.append(json.load(fh)["judge_verdict"]["agreement_score"])

    if all_scores:
        print("\n" + "=" * 50)
        print(f"AGGREGATE across {len(all_scores)} scored papers:")
        print(f"  mean agreement_score:   {statistics.mean(all_scores):.3f}")
        print(f"  median agreement_score: {statistics.median(all_scores):.3f}")
        if len(all_scores) > 1:
            print(f"  stdev:                  {statistics.stdev(all_scores):.3f}")
        print("=" * 50)


if __name__ == "__main__":
    main()