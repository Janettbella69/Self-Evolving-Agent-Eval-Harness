#!/usr/bin/env python3
"""Extract few-shot examples from training split for LLM judge calibration.

Reads human-annotated traces from the train split, selects the best examples
(1 clear PASS + 1 clear FAIL per grader), and writes to calibration/few_shots/.

Usage:
    python scripts/extract_few_shots.py
    python scripts/extract_few_shots.py --db backend/eval.db
"""

import argparse
import json
import sqlite3
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent  # repo root
# Must match graders/llm_graders.py:_CALIBRATION_DIR — the judge reads few-shots from here.
CALIBRATION_DIR = _REPO_ROOT / "calibration"
FEW_SHOTS_DIR = CALIBRATION_DIR / "few_shots"
SPLITS_FILE = CALIBRATION_DIR / "splits.json"
DB_PATH = _REPO_ROOT / "backend" / "eval.db"

# LLM graders that need few-shot examples
LLM_GRADERS = ["rubric_compliance", "groundedness", "actionability", "trap_detection"]


def load_train_traces(db_path: str) -> list[dict]:
    """Load traces in the training split with human annotations."""
    if not SPLITS_FILE.exists():
        print(f"No splits.json found at {SPLITS_FILE}")
        print("Run calibration_loop.py --create-splits first.")
        return []

    splits = json.loads(SPLITS_FILE.read_text())
    train_ids = splits.get("train", [])
    if not train_ids:
        print("No train IDs in splits.json")
        return []

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    placeholders = ",".join("?" * len(train_ids))
    rows = conn.execute(f"""
        SELECT id, case_key, case_type, query, guide_text, final_score, final_pass,
               human_pass, human_scores, composite_scores
        FROM traces
        WHERE id IN ({placeholders}) AND human_pass IS NOT NULL
    """, train_ids).fetchall()
    conn.close()

    return [dict(r) for r in rows]


def select_examples(traces: list[dict], grader_name: str) -> list[dict]:
    """Select 1 PASS + 1 FAIL example for a grader.

    Criteria: highest human-judge agreement (both agree on verdict).
    """
    examples = []

    pass_candidates = []
    fail_candidates = []

    for t in traces:
        human_pass = bool(t["human_pass"])
        composite = json.loads(t["composite_scores"] or "{}")
        grader_data = composite.get(grader_name, {})
        if not isinstance(grader_data, dict):
            continue

        grader_score = grader_data.get("score", 0)
        grader_pass = grader_score >= 70

        query = t["query"] or ""
        guide_excerpt = (t["guide_text"] or "")[:300]

        example = {
            "trace_id": t["id"],
            "input_summary": f"Query: {query[:150]}. Guide excerpt: {guide_excerpt[:150]}...",
            "score": grader_score,
            "verdict": "PASS" if human_pass else "FAIL",
            "reasoning": grader_data.get("details", {}).get("reasoning", f"Auto score: {grader_score}"),
            "human_pass": human_pass,
            "auto_pass": grader_pass,
            "agreement": human_pass == grader_pass,
        }

        if human_pass:
            pass_candidates.append(example)
        else:
            fail_candidates.append(example)

    # Prefer examples where human and auto agree (high-confidence examples)
    pass_candidates.sort(key=lambda x: (x["agreement"], x["score"]), reverse=True)
    fail_candidates.sort(key=lambda x: (x["agreement"], -x["score"]))

    if pass_candidates:
        examples.append(pass_candidates[0])
    if fail_candidates:
        examples.append(fail_candidates[0])

    return examples


def main():
    parser = argparse.ArgumentParser(description="Extract few-shot examples from train split")
    parser.add_argument("--db", default=str(DB_PATH))
    args = parser.parse_args()

    FEW_SHOTS_DIR.mkdir(parents=True, exist_ok=True)

    traces = load_train_traces(args.db)
    if not traces:
        print("No annotated train traces. Annotate traces first with annotate_batch.py")
        return

    print(f"Loaded {len(traces)} annotated train traces")

    for grader in LLM_GRADERS:
        examples = select_examples(traces, grader)
        output_path = FEW_SHOTS_DIR / f"{grader}.json"
        output_path.write_text(json.dumps(examples, indent=2, default=str))
        print(f"  {grader}: {len(examples)} examples → {output_path.name}")

    print(f"\nFew-shots written to {FEW_SHOTS_DIR}")


if __name__ == "__main__":
    main()
