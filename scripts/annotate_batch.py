#!/usr/bin/env python3
"""Batch human annotation tool for eval traces.

Presents trace summaries and collects PASS/FAIL judgments.
Writes to both SQLite DB and calibration/annotations.jsonl.

Usage:
    python scripts/annotate_batch.py --experiment-id 28
    python scripts/annotate_batch.py --experiment-id 28 --limit 10
    python scripts/annotate_batch.py --experiment-id 28 --unannotated-only
"""

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent  # repo root (scripts/ lives directly under it)
ANNOTATIONS_FILE = _REPO_ROOT / "calibration" / "annotations.jsonl"
DB_PATH = _REPO_ROOT / "backend" / "eval.db"


def load_traces(db_path: str, experiment_id: int, unannotated_only: bool = False, limit: int = 0) -> list[dict]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    sql = """
        SELECT id, experiment_id, case_key, case_type, query,
               guide_text, final_score, final_pass,
               composite_scores, duration_s, error_events,
               human_pass, human_scores
        FROM traces
        WHERE experiment_id = ? AND status IN ('done', 'graded', 'collected')
              AND guide_text IS NOT NULL AND length(guide_text) > 0
    """
    params: list = [experiment_id]

    if unannotated_only:
        sql += " AND human_pass IS NULL"

    sql += " ORDER BY id"

    if limit > 0:
        sql += f" LIMIT {limit}"

    rows = conn.execute(sql, params).fetchall()
    conn.close()

    traces = []
    for r in rows:
        traces.append({
            "id": r["id"],
            "experiment_id": r["experiment_id"],
            "case_key": r["case_key"],
            "case_type": r["case_type"] or "",
            "query": r["query"] or "",
            "guide_text": r["guide_text"] or "",
            "final_score": r["final_score"] or 0,
            "final_pass": bool(r["final_pass"]),
            "composite_scores": json.loads(r["composite_scores"] or "{}"),
            "duration_s": r["duration_s"] or 0,
            "error_events": json.loads(r["error_events"] or "[]"),
            "human_pass": r["human_pass"],
            "human_scores": json.loads(r["human_scores"] or "{}"),
        })
    return traces


def present_trace(trace: dict, index: int, total: int):
    """Print trace summary for human review.

    Machine results are left out so they can't anchor the human's verdict;
    present_auto_result() prints them once the verdict is saved.
    """
    print(f"\n{'='*70}")
    print(f"  Trace {trace['id']}  ({index+1}/{total})  [{trace['case_type']}]")
    print(f"{'='*70}")
    print(f"  Query: {trace['query'][:200]}")
    print(f"  Duration: {trace['duration_s']:.1f}s")

    if trace['error_events']:
        print(f"  Errors: {len(trace['error_events'])}")

    # Guide excerpt
    guide = trace['guide_text']
    print(f"\n  --- Guide ({len(guide)} chars) ---")
    # Show first 800 chars
    excerpt = guide[:800]
    for line in excerpt.split('\n'):
        print(f"  | {line}")
    if len(guide) > 800:
        print(f"  | ... ({len(guide) - 800} more chars)")

    # Previous annotation
    if trace['human_pass'] is not None:
        prev = "PASS" if trace['human_pass'] else "FAIL"
        print(f"\n  Previous annotation: {prev}")


def present_auto_result(trace: dict):
    """Print the automated verdict and grader scores, after the human has judged."""
    print(f"  Auto Score: {trace['final_score']:.1f}  Auto Pass: {trace['final_pass']}")

    # Auto grader scores
    cs = trace['composite_scores']
    if cs:
        print(f"\n  --- Auto Grader Scores ---")
        for name, data in sorted(cs.items()):
            if isinstance(data, dict):
                score = data.get('score', '?')
                print(f"    {name:25s}: {score}")


def collect_annotation(trace: dict) -> dict | None:
    """Collect human annotation interactively."""
    while True:
        raw = input("\n  Verdict [P]ass / [F]ail / [S]kip / [Q]uit: ").strip().lower()
        if raw in ('q', 'quit'):
            return None
        if raw in ('s', 'skip'):
            return {"skipped": True}
        if raw in ('p', 'pass'):
            human_pass = True
            break
        if raw in ('f', 'fail'):
            human_pass = False
            break
        print("  Invalid input. Enter P, F, S, or Q.")

    # Optional: per-grader scores
    human_scores = {}
    do_grader = input("  Score individual graders? [y/N]: ").strip().lower()
    if do_grader == 'y':
        for grader_name in ["rubric_compliance", "groundedness", "actionability", "trap_detection"]:
            raw = input(f"    {grader_name} (0-100, Enter=skip): ").strip()
            if raw and raw.isdigit():
                human_scores[grader_name] = int(raw)

    return {
        "trace_id": trace["id"],
        "experiment_id": trace["experiment_id"],
        "case_key": trace["case_key"],
        "case_type": trace["case_type"],
        "human_pass": human_pass,
        "human_scores": human_scores,
        "auto_score": trace["final_score"],
        "auto_pass": trace["final_pass"],
        "annotator": "human",
        "timestamp": time.time(),
    }


def save_annotation(annotation: dict, db_path: str):
    """Save to both DB and JSONL file."""
    # Append to JSONL
    ANNOTATIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(ANNOTATIONS_FILE, 'a') as f:
        f.write(json.dumps(annotation, default=str) + '\n')

    # Update DB
    conn = sqlite3.connect(db_path)
    conn.execute(
        "UPDATE traces SET human_pass = ?, human_scores = ? WHERE id = ?",
        (int(annotation["human_pass"]), json.dumps(annotation["human_scores"]), annotation["trace_id"]),
    )
    conn.commit()
    conn.close()


def main():
    parser = argparse.ArgumentParser(description="Batch annotate eval traces")
    parser.add_argument("--experiment-id", type=int, required=True)
    parser.add_argument("--db", default=str(DB_PATH))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--unannotated-only", action="store_true")
    args = parser.parse_args()

    traces = load_traces(args.db, args.experiment_id, args.unannotated_only, args.limit)
    if not traces:
        print("No traces found for annotation.")
        return

    print(f"Found {len(traces)} traces for experiment #{args.experiment_id}")
    annotated = 0
    skipped = 0

    for i, trace in enumerate(traces):
        present_trace(trace, i, len(traces))
        result = collect_annotation(trace)

        if result is None:  # Quit
            break
        if result.get("skipped"):
            skipped += 1
            continue

        save_annotation(result, args.db)
        annotated += 1
        verdict = "PASS" if result["human_pass"] else "FAIL"
        print(f"  → Saved: {verdict}")
        present_auto_result(trace)

    print(f"\nDone. Annotated: {annotated}, Skipped: {skipped}")
    print(f"Annotations file: {ANNOTATIONS_FILE}")


if __name__ == "__main__":
    main()
