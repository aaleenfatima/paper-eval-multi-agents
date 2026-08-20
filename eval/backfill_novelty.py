"""
Backfills grounded novelty verdicts into existing eval/results/ files.

Context: the full 50-paper batch ran BEFORE Semantic Scholar retrieval was
working (SSL interception + rate limiting), so every result has an ungrounded
novelty verdict (empty related_papers_considered). Retrieval is fixed now.
Redoing the full pipeline (structure + methodology + debate + synthesis) per
paper would cost another ~6-8 min/paper for no reason -- structure, methodology,
and the debate log are unaffected by retrieval and don't need to change.

This script, for each existing result:
  1. Re-runs ONLY the novelty agent (now grounded), single-pass -- no debate
     challenge on novelty specifically, since debate mechanics were already
     validated on methodology in the original run (see pre_debate_confidence
     there); re-running debate here would roughly double backfill time for
     largely redundant evidence
  2. Drops the old (ungrounded-verdict) novelty debate round from the log,
     since it challenged a verdict that no longer exists
  3. Re-runs ONLY the synthesis agent so the final report reflects the new
     novelty verdict
  4. Overwrites the result file in place, preserving everything else

Safe to interrupt and re-run -- use --force to redo a paper that already has
a backfilled novelty verdict (marked via the "novelty_backfilled" flag added
to each result), otherwise already-backfilled papers are skipped.

Usage:
    python eval/backfill_novelty.py --limit 3
    python eval/backfill_novelty.py
"""

import sys
import json
import argparse
import time
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models import PaperInput, FinalReport, AgentVerdict, NoveltyVerdict, DebateRound
from agents import novelty_agent, synthesis_agent

RESULTS_DIR = Path(__file__).resolve().parent / "results"
DATA_DIR = Path(__file__).resolve().parent / "data"


def _rebuild_paper_input(result: dict) -> PaperInput:
    """Reconstruct the ORIGINAL PaperInput (title + abstract + draft_text) by
    looking up the matching file in eval/data/ via paper_id. The result file
    itself only stored the title, not the full input -- using title alone
    here would make novelty's re-review worse than the original run, not
    better, since it'd lose all the actual paper content to critique against."""
    paper_id = result["paper_id"]
    data_path = DATA_DIR / f"{paper_id}.json"
    if not data_path.exists():
        raise FileNotFoundError(
            f"Cannot backfill {paper_id}: original input not found at {data_path}. "
            f"Was eval/data/ cleared or regenerated with different paper_ids since the batch ran?"
        )
    with open(data_path, encoding="utf-8") as f:
        original = json.load(f)
    return PaperInput(
        title=original["title"],
        abstract=original.get("abstract"),
        draft_text=original.get("draft_text"),
    )


def backfill_one(result_path: Path) -> bool:
    with open(result_path, encoding="utf-8") as f:
        result = json.load(f)

    if result.get("novelty_backfilled"):
        return False  # already done

    report = result["report"]
    paper = _rebuild_paper_input(result)

    # Re-run novelty, now grounded. NOTE: debate/challenge is intentionally
    # skipped here -- the debate mechanism was already validated on
    # methodology verdicts in the original 50-paper run (see debate_log /
    # pre_debate_confidence there). This backfill's job is specifically to
    # add missing retrieval grounding, not to re-prove debate works a second
    # time on a different agent; re-running debate here would roughly double
    # backfill time for a largely redundant piece of evidence.
    new_novelty = novelty_agent.run(paper)
    new_novelty.was_debated = False
    new_novelty.pre_debate_confidence = None

    debate_log = [DebateRound(**d) for d in report.get("debate_log", []) if d["agent_name"] != "novelty"]
    # Old ungrounded novelty debate round is dropped (it challenged a verdict
    # that no longer exists); no new novelty debate round replaces it.

    # Re-run synthesis with the updated novelty verdict
    structure_verdict = AgentVerdict(**report["structure_clarity"])
    methodology_verdict = AgentVerdict(**report["methodology"])
    final_text = synthesis_agent.run(structure_verdict, methodology_verdict, new_novelty)

    overall_confidence = round(
        (structure_verdict.confidence + methodology_verdict.confidence + new_novelty.confidence) / 3, 2
    )

    report["novelty"] = new_novelty.model_dump()
    report["debate_log"] = [d.model_dump() for d in debate_log]
    report["final_recommendation"] = final_text
    report["overall_confidence"] = overall_confidence

    result["report"] = report
    result["novelty_backfilled"] = True
    result["backfilled_at"] = datetime.now(timezone.utc).isoformat()

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    result_files = sorted([f for f in RESULTS_DIR.glob("*.json")])
    if args.limit:
        result_files = result_files[:args.limit]

    print(f"Found {len(result_files)} result files")

    processed = 0
    skipped = 0
    for i, f in enumerate(result_files, 1):
        if args.force:
            with open(f, encoding="utf-8") as fh:
                r = json.load(fh)
            r["novelty_backfilled"] = False
            with open(f, "w", encoding="utf-8") as fh:
                json.dump(r, fh, indent=2)

        t0 = time.time()
        print(f"[{i}/{len(result_files)}] {f.stem}...", flush=True)
        try:
            did_work = backfill_one(f)
            if did_work:
                print(f"    backfilled in {time.time() - t0:.1f}s")
                processed += 1
            else:
                print(f"    already backfilled, skipping")
                skipped += 1
        except Exception as e:
            print(f"    FAILED: {e}")

    print(f"\nDone. Backfilled {processed}, skipped {skipped} already-done.")


if __name__ == "__main__":
    main()