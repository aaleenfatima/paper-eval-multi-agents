"""
Loads papers for the eval batch. Expects a directory of JSON files, one per
paper, each shaped like:

{
  "paper_id": "peerread_0001",
  "title": "...",
  "abstract": "...",
  "draft_text": "... (optional, full text if available)",
  "human_review_text": "... (the real reviewer's comments, for later comparison in Phase 4)"
}

paper_id is what drives checkpoint/resume -- it must be unique and stable
across runs, or resume logic will re-run or skip the wrong papers.

If you're pulling from PeerRead directly: their raw format nests things
differently (reviews as a list of dicts under different keys depending on
venue). Write a one-off conversion script that flattens PeerRead's format
into this shape and drop the converted files into eval/data/ -- don't try
to parse PeerRead's raw format inline here, keep this loader dumb and simple.
"""

import json
from pathlib import Path
from models import PaperInput


def load_eval_papers(data_dir: str) -> list[dict]:
    """Returns a list of dicts: {paper_id, paper_input, human_review_text}."""
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Eval data directory not found: {data_dir}\n"
            f"Create it and add paper JSON files (see docstring in this file for the expected shape)."
        )

    papers = []
    for f in sorted(data_path.glob("*.json")):
        with open(f, encoding="utf-8") as fh:
            data = json.load(fh)

        if "paper_id" not in data or "title" not in data:
            print(f"SKIPPING {f.name}: missing required 'paper_id' or 'title' field")
            continue

        paper_input = PaperInput(
            title=data["title"],
            abstract=data.get("abstract"),
            draft_text=data.get("draft_text"),
        )
        papers.append({
            "paper_id": data["paper_id"],
            "paper_input": paper_input,
            "human_review_text": data.get("human_review_text", ""),
        })

    return papers
