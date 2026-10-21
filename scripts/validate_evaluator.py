#!/usr/bin/env python3
"""Validate evaluator — calibrate LLM judges against human labels.

Based on Hamel Husain's validate-evaluator skill:
https://hamel.dev/blog/posts/evals-skills/

Failure is the positive class (TPR = share of human-failed traces the judge
fails too). Splits come from calibration/splits.json when it exists, otherwise
they are made in memory by case, so no case lands in two splits. The default
report covers dev only; the test split is held out for one final check.

Usage:
    python scripts/validate_evaluator.py [--db backend/eval.db]
    python scripts/validate_evaluator.py --per-grader
    python scripts/validate_evaluator.py --experiment-id 5
    python scripts/validate_evaluator.py --split test        # final check, prompts frozen
"""

import argparse
import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "backend"))

from graders.validation import (  # noqa: E402
    MIN_LABELS_PER_CLASS,
    VALIDATION_TARGET,
    VALIDATION_TNR_THRESHOLD,
    VALIDATION_TPR_THRESHOLD,
    confusion_from_pairs,
    group_stratified_split,
    human_grader_passed,
    judge_passed,
    rogan_gladen_correction,
)

SPLITS_FILE = _REPO_ROOT / "calibration" / "splits.json"


def _row_to_labeled(r) -> dict:
    return {
        "id": r["id"],
        "case_key": r["case_key"],
        "final_pass": bool(r["final_pass"]),
        "human_pass": bool(r["human_pass"]),
        "final_score": r["final_score"] or 0,
        "composite_scores": json.loads(r["composite_scores"] or "{}"),
        "human_scores": json.loads(r["human_scores"] or "{}"),
        "case_type": r["case_type"] or "",
    }


def load_labeled_traces(db_path: str, experiment_id: int | None = None) -> list[dict]:
    """Load traces with human PASS/FAIL annotations."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    sql = """
        SELECT id, case_key, final_pass, human_pass, final_score,
               composite_scores, human_scores, case_type
        FROM traces
        WHERE human_pass IS NOT NULL AND status IN ('done', 'graded')
    """
    params = []
    if experiment_id:
        sql += " AND experiment_id = ?"
        params.append(experiment_id)

    rows = conn.execute(sql, params).fetchall()
    conn.close()
    return [_row_to_labeled(r) for r in rows]


def load_all_traces(db_path: str) -> list[dict]:
    """Load ALL traces (for bias correction denominator)."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT final_pass FROM traces WHERE status IN ('done', 'graded')"
    ).fetchall()
    conn.close()
    return [{"final_pass": bool(r["final_pass"])} for r in rows]


def compute_metrics(traces: list[dict], auto_key="final_pass", human_key="human_pass"):
    """Compute TPR, TNR, confusion matrix (failure = positive)."""
    m = confusion_from_pairs((t[auto_key], t[human_key]) for t in traces)
    m["total"] = len(traces)
    m["human_pass_rate"] = round(m["human_pass"] / len(traces), 3) if traces else 0
    m["auto_pass_rate"] = round((m["tn"] + m["fn"]) / len(traces), 3) if traces else 0
    return m


def bootstrap_ci(traces: list[dict], p_obs: float, n_bootstrap=2000, seed=42):
    """Bootstrap 95% CI for corrected success rate."""
    rng = np.random.RandomState(seed)
    n = len(traces)
    if n < 10:
        return None, None

    estimates = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        m = confusion_from_pairs((traces[i]["final_pass"], traces[i]["human_pass"]) for i in idx)
        if m["tpr"] is None or m["tnr"] is None:
            continue
        corrected = rogan_gladen_correction(p_obs, m["tpr"], m["tnr"])
        if corrected is not None:
            estimates.append(corrected)

    if len(estimates) < 100:
        return None, None

    return float(np.percentile(estimates, 2.5)), float(np.percentile(estimates, 97.5))


def compute_per_grader_alignment(traces: list[dict]):
    """Compute TPR/TNR per LLM judge against the human label for that grader.

    The judge's verdict (or score >= 70 for old entries) is compared with
    human_scores[grader] (>= 70 = Pass). Comparing a single judge with the
    overall human verdict would blame it for failures it isn't meant to catch.
    """
    pairs: dict[str, list[tuple[bool, bool]]] = defaultdict(list)
    for t in traces:
        for grader, entry in t["composite_scores"].items():
            if not isinstance(entry, dict) or entry.get("category") != "llm":
                continue
            judge = judge_passed(entry)
            label = human_grader_passed(t["human_scores"], grader)
            if judge is not None and label is not None:
                pairs[grader].append((judge, label))
    return {grader: confusion_from_pairs(p) for grader, p in sorted(pairs.items())}


def print_report(
    splits: dict, eval_name: str, metrics: dict,
    all_traces, per_grader, corrected_rate, ci_lower, ci_upper, per_grader_requested: bool,
):
    """Print structured validation report."""
    print("=" * 60)
    print("  EVALUATOR VALIDATION REPORT")
    print("  Based on Hamel Husain's validate-evaluator framework")
    print("=" * 60)

    # Data splits
    print(f"\n## Data Splits (by case)")
    for name, part in splits.items():
        p = sum(1 for t in part if t["human_pass"])
        cases = len({t["case_key"] for t in part})
        print(f"  {name + ':':<6} {len(part):>4} traces, {cases:>3} cases ({p} Pass, {len(part) - p} Fail)")

    print(f"\n## {eval_name.capitalize()} Set Alignment")
    _print_confusion(metrics)
    if eval_name == "dev":
        print("  Test split held out — run with --split test once prompts are frozen.")

    # Targets
    print(f"\n## Target Check")
    tpr = metrics["tpr"]
    tnr = metrics["tnr"]
    goal = f"(target: ≥{VALIDATION_TARGET:.2f}, min: ≥{VALIDATION_TPR_THRESHOLD:.2f})"
    print(f"  TPR: {tpr:.3f} {_mark(tpr, VALIDATION_TPR_THRESHOLD)} {goal}" if tpr is not None else "  TPR: N/A")
    print(f"  TNR: {tnr:.3f} {_mark(tnr, VALIDATION_TNR_THRESHOLD)} {goal}" if tnr is not None else "  TNR: N/A")

    # Bias correction
    if corrected_rate is not None:
        all_count = len(all_traces)
        all_pass = sum(1 for t in all_traces if t["final_pass"])
        p_obs = all_pass / all_count if all_count else 0

        print(f"\n## Rogan-Gladen Bias Correction (on {all_count} total traces)")
        print(f"  Observed pass rate:  {p_obs:.3f}")
        print(f"  Corrected pass rate: {corrected_rate:.3f}")
        if ci_lower is not None:
            print(f"  95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")
        else:
            print(f"  95% CI: insufficient data for bootstrap")

    # Per-grader
    if per_grader_requested:
        print(f"\n## Per-Judge Alignment (judge verdict vs human label for the same grader)")
        if not per_grader:
            print("  No per-grader human labels. Label graders in the trace page's Scores tab")
            print("  or with annotate_batch.py, then re-run.")
        else:
            print(f"  {'Grader':<25} {'TPR':>6} {'TNR':>6} {'TP':>4} {'FP':>4} {'TN':>4} {'FN':>4}  Pass/Fail labels")
            print(f"  {'-'*25} {'-'*6} {'-'*6} {'-'*4} {'-'*4} {'-'*4} {'-'*4}  {'-'*16}")
            for name, m in per_grader.items():
                tpr_s = f"{m['tpr']:.3f}" if m['tpr'] is not None else "N/A"
                tnr_s = f"{m['tnr']:.3f}" if m['tnr'] is not None else "N/A"
                few = "  ⚠ few" if min(m["human_pass"], m["human_fail"]) < MIN_LABELS_PER_CLASS else ""
                print(f"  {name:<25} {tpr_s:>6} {tnr_s:>6} {m['tp']:>4} {m['fp']:>4} {m['tn']:>4} {m['fn']:>4}"
                      f"  {m['human_pass']}/{m['human_fail']}{few}")

    # Recommendations
    print(f"\n## Recommendations")
    labeled = sum(len(part) for part in splits.values())
    if labeled < 50:
        print(f"  ⚠ Only {labeled} labeled traces — need ~100 for reliable calibration")
        print(f"    Collect more via: POST /api/traces/{{id}}/annotate-pass")
    if min(metrics["human_pass"], metrics["human_fail"]) < MIN_LABELS_PER_CLASS:
        print(f"  ⚠ {eval_name} has {metrics['human_pass']} Pass / {metrics['human_fail']} Fail labels — "
              f"aim for {MIN_LABELS_PER_CLASS}-50 of each")
    if tpr is not None and tpr < VALIDATION_TPR_THRESHOLD:
        print(f"  ⚠ TPR too low ({tpr:.3f}) — judge misses real failures (too lenient)")
        print(f"    → Sharpen FAIL definitions and add failing edge-case examples")
    if tnr is not None and tnr < VALIDATION_TNR_THRESHOLD:
        print(f"  ⚠ TNR too low ({tnr:.3f}) — judge fails good outputs (too strict)")
        print(f"    → Clarify PASS definitions and add passing examples")
    if labeled >= 50 and tpr and tnr and tpr >= VALIDATION_TARGET and tnr >= VALIDATION_TARGET:
        if eval_name == "test":
            print(f"  ✅ Judge meets alignment targets on the held-out test split.")
        else:
            print(f"  ✅ Judge meets alignment targets on {eval_name}. Confirm once with --split test.")


def _mark(rate: float | None, minimum: float) -> str:
    if rate is None:
        return ""
    return "✅" if rate >= VALIDATION_TARGET else "❌" if rate >= minimum else "🔴"


def _print_confusion(m):
    print(f"  Confusion Matrix:")
    print(f"                  Judge PASS  Judge FAIL")
    print(f"    Human PASS    {m['tn']:>6}      {m['fp']:>6}")
    print(f"    Human FAIL    {m['fn']:>6}      {m['tp']:>6}")
    print(f"  TPR (failures caught): {m['tpr']:.3f}" if m['tpr'] is not None else "  TPR: N/A")
    print(f"  TNR (passes kept):     {m['tnr']:.3f}" if m['tnr'] is not None else "  TNR: N/A")


def load_split_traces(db_path: str, split_name: str) -> list[dict]:
    """Load traces from a specific split (using calibration/splits.json)."""
    if not SPLITS_FILE.exists():
        return []
    splits = json.loads(SPLITS_FILE.read_text())
    trace_ids = splits.get(split_name, [])
    if not trace_ids:
        return []

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    placeholders = ",".join("?" * len(trace_ids))
    rows = conn.execute(f"""
        SELECT id, case_key, final_pass, human_pass, final_score,
               composite_scores, human_scores, case_type
        FROM traces
        WHERE id IN ({placeholders}) AND human_pass IS NOT NULL
    """, trace_ids).fetchall()
    conn.close()

    return [_row_to_labeled(r) for r in rows]


def main():
    parser = argparse.ArgumentParser(description="Validate LLM evaluator against human labels")
    parser.add_argument("--db", default="backend/eval.db", help="Path to SQLite DB")
    parser.add_argument("--experiment-id", type=int, help="Filter to specific experiment")
    parser.add_argument("--per-grader", action="store_true", help="Show per-grader alignment")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for in-memory splits")
    parser.add_argument("--split", choices=["dev", "test"],
                        help="Report this split of calibration/splits.json (default: dev)")
    args = parser.parse_args()

    db_path = args.db
    if not Path(db_path).exists():
        print(f"Error: Database not found at {db_path}")
        sys.exit(1)

    if args.split:
        # One split from calibration/splits.json
        eval_traces = load_split_traces(db_path, args.split)
        if not eval_traces:
            print(f"No annotated traces in {args.split} split. Run calibration_loop.py --create-splits first.")
            sys.exit(0)
        print(f"Using {args.split} split: {len(eval_traces)} traces")
        splits = {args.split: eval_traces}
        eval_name = args.split
    else:
        traces = load_labeled_traces(db_path, args.experiment_id)
        if not traces:
            print(f"No human-labeled traces found.")
            print(f"Label traces via: python scripts/annotate_batch.py --experiment-id N")
            sys.exit(0)

        print(f"Found {len(traces)} human-labeled traces.")

        if len(traces) < 20:
            print(f"Warning: Only {len(traces)} labeled traces. Need ~100 for reliable results.")
            print(f"Running with full dataset (no train/dev/test split).\n")
            splits = {"all": traces}
            eval_traces, eval_name = traces, "all"
        elif SPLITS_FILE.exists():
            # Reuse the calibration splits so few-shot (train) traces never count as dev
            ids = json.loads(SPLITS_FILE.read_text())
            by_id = {t["id"]: t for t in traces}
            splits = {name: [by_id[i] for i in ids.get(name, []) if i in by_id]
                      for name in ("train", "dev", "test")}
            print(f"Using splits from {SPLITS_FILE} (labeled traces outside it are ignored).")
            eval_traces, eval_name = splits["dev"], "dev"
        else:
            train, dev, test = group_stratified_split(traces, seed=args.seed)
            splits = {"train": train, "dev": dev, "test": test}
            eval_traces, eval_name = dev, "dev"

    metrics = compute_metrics(eval_traces)

    all_traces = load_all_traces(db_path)
    corrected_rate = ci_lower = ci_upper = None
    if metrics["tpr"] is not None and metrics["tnr"] is not None and all_traces:
        all_pass = sum(1 for t in all_traces if t["final_pass"])
        p_obs = all_pass / len(all_traces) if all_traces else 0
        corrected_rate = rogan_gladen_correction(p_obs, metrics["tpr"], metrics["tnr"])
        ci_lower, ci_upper = bootstrap_ci(eval_traces, p_obs, seed=args.seed)

    per_grader = compute_per_grader_alignment(eval_traces) if args.per_grader else {}
    print_report(splits, eval_name, metrics, all_traces, per_grader,
                 corrected_rate, ci_lower, ci_upper, args.per_grader)


if __name__ == "__main__":
    main()
