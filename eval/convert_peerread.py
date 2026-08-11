"""
One-off converter: PeerRead's raw review + parsed-PDF JSON files -> PaperPilot's
eval data shape (see eval/data_loader.py docstring).

PeerRead splits data into two parallel folders per paper (same numeric filename
stem, e.g. "316"):
  - reviews/{id}.json       -> {title, abstract, reviews: [{comments, IS_META_REVIEW}, ...]}
  - parsed_pdfs/{id}.pdf.json -> {metadata: {sections: [{heading, text}, ...], ...}}

We join both files by their shared numeric ID to build a full-draft paper
(title + abstract + concatenated section text) alongside the real human
review, so PaperPilot's Methodology and Structure agents have actual content
to critique instead of just a one-paragraph abstract.

Usage:
    python eval/convert_peerread.py <reviews_dir> <parsed_pdfs_dir> <output_dir> [--limit N]

Example:
    python eval/convert_peerread.py \\
        peerread_extracted/PeerRead-master/data/iclr_2017/train/reviews \\
        peerread_extracted/PeerRead-master/data/iclr_2017/train/parsed_pdfs \\
        eval/data --limit 40
"""

import sys
import json
import argparse
from pathlib import Path

MAX_DRAFT_CHARS = 6000  # first ~intro+method+results carries most critique-worthy content;
                          # halving this roughly halves prefill cost on every call, and every
                          # call in the pipeline re-sends the full draft, so this multiplies out fast


def _load_full_text(parsed_pdfs_dir: Path, paper_stem: str) -> str | None:
    """Loads and reassembles section text from a parsed_pdfs/{stem}.pdf.json file."""
    parsed_path = parsed_pdfs_dir / f"{paper_stem}.pdf.json"
    if not parsed_path.exists():
        return None
    try:
        with open(parsed_path, encoding="utf-8") as f:
            parsed = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None

    sections = parsed.get("metadata", {}).get("sections") or []
    if not sections:
        return None

    parts = []
    for s in sections:
        heading = (s.get("heading") or "").strip()
        text = (s.get("text") or "").strip()
        if not text:
            continue
        parts.append(f"{heading}\n{text}" if heading else text)

    full_text = "\n\n".join(parts)
    if not full_text:
        return None
    return full_text[:MAX_DRAFT_CHARS]


def convert_one(raw_path: Path, parsed_pdfs_dir: Path, venue_tag: str) -> dict | None:
    with open(raw_path, encoding="utf-8") as f:
        data = json.load(f)

    title = data.get("title", "").strip()
    abstract = data.get("abstract", "").strip()
    reviews = data.get("reviews", [])

    if not title or not abstract:
        return None  # skip incomplete entries rather than write junk

    non_meta = [r.get("comments", "").strip() for r in reviews if not r.get("IS_META_REVIEW") and r.get("comments")]
    meta = [r.get("comments", "").strip() for r in reviews if r.get("IS_META_REVIEW") and r.get("comments")]
    review_texts = non_meta if non_meta else meta

    if not review_texts:
        return None  # no usable review content, skip

    human_review_text = "\n\n---\n\n".join(review_texts)

    draft_text = _load_full_text(parsed_pdfs_dir, raw_path.stem)  # None if no parsed PDF available -> falls back to abstract-only

    paper_id = f"{venue_tag}_{raw_path.stem}"

    return {
        "paper_id": paper_id,
        "title": title,
        "abstract": abstract,
        "draft_text": draft_text,
        "human_review_text": human_review_text,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("reviews_dir", help="path to a PeerRead .../reviews/ directory")
    parser.add_argument("parsed_pdfs_dir", help="path to the matching .../parsed_pdfs/ directory")
    parser.add_argument("output_dir", help="where to write converted eval data files")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--venue-tag", default="iclr2017", help="short tag prefixed to paper_id")
    args = parser.parse_args()

    reviews_dir = Path(args.reviews_dir)
    parsed_pdfs_dir = Path(args.parsed_pdfs_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_files = sorted(reviews_dir.glob("*.json"))
    if args.limit:
        raw_files = raw_files[:args.limit * 2]  # over-fetch since some will be skipped for missing data

    converted = 0
    skipped = 0
    full_draft_count = 0
    for raw_path in raw_files:
        if args.limit and converted >= args.limit:
            break
        result = convert_one(raw_path, parsed_pdfs_dir, args.venue_tag)
        if result is None:
            skipped += 1
            continue
        if result["draft_text"]:
            full_draft_count += 1
        out_path = output_dir / f"{result['paper_id']}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        converted += 1

    print(f"Converted {converted} papers -> {output_dir}")
    print(f"  {full_draft_count} with full draft text, {converted - full_draft_count} abstract-only (no matching parsed PDF found)")
    print(f"Skipped {skipped} (missing title/abstract/reviews)")


if __name__ == "__main__":
    main()
