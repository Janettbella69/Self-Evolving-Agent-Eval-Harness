#!/usr/bin/env python3
"""Validate evaluator — calibrate LLM judges against human labels.

Based on Hamel Husain's validate-evaluator skill:
https://hamel.dev/blog/posts/evals-skills/

Usage:
    python scripts/validate_evaluator.py [--db backend/eval.db]
    python scripts/validate_evaluator.py --per-grader
    python scripts/validate_evaluator.py --experiment-id 5
"""

import argparse
import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


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

    traces = []
    for r in rows:
        traces.append({
            "id": r["id"],
            "case_key": r["case_key"],
            "final_pass": bool(r["final_pass"]),
            "human_pass": bool(r["human_pass"]),
            "final_score": r["final_score"] or 0,
            "composite_scores": json.loads(r["composite_scores"] or "{}"),
            "human_scores": json.loads(r["human_scores"] or "{}"),
            "case_type": r["case_type"] or "",
        })
    return traces


def load_all_traces(db_path: str) -> list[dict]:
    """Load ALL traces (for bias correction denominator)."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT final_pass FROM traces WHERE status IN ('done', 'graded')"
    ).fetchall()
    conn.close()
    return [{"final_pass": bool(r["final_pass"])} for r in rows]


def stratified_split(traces: list[dict], train_frac=0.15, dev_frac=0.45, seed=42):
    """Split labeled data into train/dev/test, stratified by human_pass."""
    rng = np.random.RandomState(seed)

    pos = [t for t in traces if t["human_pass"]]
    neg = [t for t in traces if not t["human_pass"]]

    def _split(data):
        idx = rng.permutation(len(data))
        n_train = max(1, int(len(data) * train_frac))
        n_dev = max(1, int(len(data) * dev_frac))
        train = [data[i] for i in idx[:n_train]]
        dev = [data[i] for i in idx[n_train:n_train + n_dev]]
        test = [data[i] for i in idx[n_train + n_dev:]]
        return train, dev, test

    pos_train, pos_dev, pos_test = _split(pos)
    neg_train, neg_dev, neg_test = _split(neg)

    return (
        pos_train + neg_train,
        pos_dev + neg_dev,
        pos_test + neg_test,
    )


def compute_metrics(traces: list[dict], auto_key="final_pass", human_key="human_pass"):
    """Compute TPR, TNR, confusion matrix."""
    tp = fp = tn = fn = 0
    for t in traces:
        human = t[human_key]
        auto = t[auto_key]
        if human and auto:
            tp += 1
        elif human and not auto:
            fn += 1
        elif not human and auto:
            fp += 1
        else:
            tn += 1

    tpr = tp / (tp + fn) if (tp + fn) > 0 else None
    tnr = tn / (tn + fp) if (tn + fp) > 0 else None

    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "tpr": round(tpr, 3) if tpr is not None else None,
        "tnr": round(tnr, 3) if tnr is not None else None,
        "total": len(traces),
        "human_pass_rate": round((tp + fn) / len(traces), 3) if traces else 0,
        "auto_pass_rate": round((tp + fp) / len(traces), 3) if traces else 0,
    }


def rogan_gladen(p_obs: float, tpr: float, tnr: float) -> float | None:
    """Rogan-Gladen bias correction for true pass rate."""
    denom = tpr + tnr - 1
    if abs(denom) < 1e-6:
        return None  # Judge is no better than random
    corrected = (p_obs + tnr - 1) / denom
    return max(0.0, min(1.0, corrected))


def bootstrap_ci(traces: list[dict], p_obs: float, n_bootstrap=2000, seed=42):
    """Bootstrap 95% CI for corrected success rate."""
    rng = np.random.RandomState(seed)
    n = len(traces)
    if n < 10:
        return None, None

    estimates = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        sample = [traces[i] for i in idx]

        tp = sum(1 for t in sample if t["human_pass"] and t["final_pass"])
        fn = sum(1 for t in sample if t["human_pass"] and not t["final_pass"])
        tn = sum(1 for t in sample if not t["human_pass"] and not t["final_pass"])
        fp = sum(1 for t in sample if not t["human_pass"] and t["final_pass"])

        tpr_b = tp / (tp + fn) if (tp + fn) > 0 else 0
        tnr_b = tn / (tn + fp) if (tn + fp) > 0 else 0
        denom = tpr_b + tnr_b - 1

        if abs(denom) < 1e-6:
            continue
        theta = (p_obs + tnr_b - 1) / denom
        estimates.append(np.clip(theta, 0, 1))

    if len(estimates) < 100:
        return None, None

    return float(np.percentile(estimates, 2.5)), float(np.percentile(estimates, 97.5))


def compute_per_grader_alignment(traces: list[dict], threshold=70.0):
    """Compute TPR/TNR per grader dimension.

    For each grader, binarize its score (>= threshold → PASS) and compare
    against human_pass to get per-grader TPR/TNR.
    """
    grader_names = set()
    for t in traces:
        if t["composite_scores"]:
            grader_names.update(t["composite_scores"].keys())

    results = {}
    for grader in sorted(grader_names):
        tp = fp = tn = fn = 0
        for t in traces:
            scores = t["composite_scores"]
            if grader not in scores:
                continue
            grader_data = scores[grader]
            if not isinstance(grader_data, dict):
                continue

            grader_pass = grader_data.get("score", 0) >= threshold
            human = t["human_pass"]

            if human and grader_pass:
                tp += 1
            elif human and not grader_pass:
                fn += 1
            elif not human and grader_pass:
                fp += 1
            else:
                tn += 1

        total = tp + fp + tn + fn
        if total == 0:
            continue

        tpr = tp / (tp + fn) if (tp + fn) > 0 else None
        tnr = tn / (tn + fp) if (tn + fp) > 0 else None

        results[grader] = {
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "tpr": round(tpr, 3) if tpr is not None else None,
            "tnr": round(tnr, 3) if tnr is not None else None,
            "total": total,
        }

    return results


def print_report(
    train, dev, test, dev_metrics, test_metrics,
    all_traces, per_grader, corrected_rate, ci_lower, ci_upper,
):
    """Print structured validation report."""
    print("=" * 60)
    print("  EVALUATOR VALIDATION REPORT")
    print("  Based on Hamel Husain's validate-evaluator framework")
    print("=" * 60)

    # Data splits
    print(f"\n## Data Splits")
    print(f"  Train: {len(train):>4} ({sum(1 for t in train if t['human_pass'])} Pass, {sum(1 for t in train if not t['human_pass'])} Fail)")
    print(f"  Dev:   {len(dev):>4} ({sum(1 for t in dev if t['human_pass'])} Pass, {sum(1 for t in dev if not t['human_pass'])} Fail)")
    print(f"  Test:  {len(test):>4} ({sum(1 for t in test if t['human_pass'])} Pass, {sum(1 for t in test if not t['human_pass'])} Fail)")

    # Dev metrics
    print(f"\n## Dev Set Alignment")
    _print_confusion(dev_metrics)

    # Test metrics
    print(f"\n## Test Set Alignment (FINAL)")
    _print_confusion(test_metrics)

    # Targets
    print(f"\n## Target Check")
    tpr = test_metrics["tpr"]
    tnr = test_metrics["tnr"]
    tpr_ok = "✅" if tpr and tpr >= 0.9 else "❌" if tpr and tpr >= 0.8 else "🔴"
    tnr_ok = "✅" if tnr and tnr >= 0.9 else "❌" if tnr and tnr >= 0.8 else "🔴"
    print(f"  TPR: {tpr:.3f} {tpr_ok} (target: >0.90, min: >0.80)" if tpr else "  TPR: N/A")
    print(f"  TNR: {tnr:.3f} {tnr_ok} (target: >0.90, min: >0.80)" if tnr else "  TNR: N/A")

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
    if per_grader:
        print(f"\n## Per-Grader Alignment (threshold=70)")
        print(f"  {'Grader':<25} {'TPR':>6} {'TNR':>6} {'TP':>4} {'FP':>4} {'TN':>4} {'FN':>4}")
        print(f"  {'-'*25} {'-'*6} {'-'*6} {'-'*4} {'-'*4} {'-'*4} {'-'*4}")
        for name, m in per_grader.items():
            tpr_s = f"{m['tpr']:.3f}" if m['tpr'] is not None else "N/A"
            tnr_s = f"{m['tnr']:.3f}" if m['tnr'] is not None else "N/A"
            print(f"  {name:<25} {tpr_s:>6} {tnr_s:>6} {m['tp']:>4} {m['fp']:>4} {m['tn']:>4} {m['fn']:>4}")

    # Recommendations
    print(f"\n## Recommendations")
    labeled = len(train) + len(dev) + len(test)
    if labeled < 50:
        print(f"  ⚠ Only {labeled} labeled traces — need ~100 for reliable calibration")
        print(f"    Collect more via: POST /api/traces/{{id}}/annotate-pass")
    if tpr and tpr < 0.8:
        print(f"  ⚠ TPR too low ({tpr:.3f}) — judge is too strict, missing real passes")
        print(f"    → Strengthen PASS definitions in grader prompts")
    if tnr and tnr < 0.8:
        print(f"  ⚠ TNR too low ({tnr:.3f}) — judge is too lenient, missing real fails")
        print(f"    → Strengthen FAIL definitions and add edge-case examples")
    if labeled >= 50 and tpr and tnr and tpr >= 0.9 and tnr >= 0.9:
        print(f"  ✅ Judge meets alignment targets! Safe to use for automated eval.")


def _print_confusion(m):
    print(f"  Confusion Matrix:")
    print(f"                  Judge PASS  Judge FAIL")
    print(f"    Human PASS    {m['tp']:>6}      {m['fn']:>6}")
    print(f"    Human FAIL    {m['fp']:>6}      {m['tn']:>6}")
    print(f"  TPR (sensitivity): {m['tpr']:.3f}" if m['tpr'] is not None else "  TPR: N/A")
    print(f"  TNR (specificity): {m['tnr']:.3f}" if m['tnr'] is not None else "  TNR: N/A")


def load_split_traces(db_path: str, split_name: str) -> list[dict]:
    """Load traces from a specific split (using calibration/splits.json)."""
    splits_file = Path(__file__).parent.parent / "calibration" / "splits.json"
    if not splits_file.exists():
        return []
    splits = json.loads(splits_file.read_text())
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

    return [{
        "id": r["id"], "case_key": r["case_key"],
        "final_pass": bool(r["final_pass"]), "human_pass": bool(r["human_pass"]),
        "final_score": r["final_score"] or 0,
        "composite_scores": json.loads(r["composite_scores"] or "{}"),
        "human_scores": json.loads(r["human_scores"] or "{}"),
        "case_type": r["case_type"] or "",
    } for r in rows]


def main():
    parser = argparse.ArgumentParser(description="Validate LLM evaluator against human labels")
    parser.add_argument("--db", default="backend/eval.db", help="Path to SQLite DB")
    parser.add_argument("--experiment-id", type=int, help="Filter to specific experiment")
    parser.add_argument("--per-grader", action="store_true", help="Show per-grader alignment")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splits")
    parser.add_argument("--split", choices=["dev", "test"], help="Use calibration split instead of auto-split")
    args = parser.parse_args()

    db_path = args.db
    if not Path(db_path).exists():
        print(f"Error: Database not found at {db_path}")
        sys.exit(1)

    # If --split is specified, use calibration splits
    if args.split:
        split_traces = load_split_traces(db_path, args.split)
        if not split_traces:
            print(f"No annotated traces in {args.split} split. Run calibration_loop.py --create-splits first.")
            sys.exit(0)
        print(f"Using {args.split} split: {len(split_traces)} traces")
        train, dev, test = [], split_traces, split_traces
        dev_metrics = compute_metrics(dev)
        test_metrics = dev_metrics  # same data for single-split mode

        all_traces = load_all_traces(db_path)
        corrected_rate = ci_lower = ci_upper = None
        if test_metrics["tpr"] and test_metrics["tnr"] and all_traces:
            all_pass = sum(1 for t in all_traces if t["final_pass"])
            p_obs = all_pass / len(all_traces) if all_traces else 0
            corrected_rate = rogan_gladen(p_obs, test_metrics["tpr"], test_metrics["tnr"])
            ci_lower, ci_upper = bootstrap_ci(split_traces, p_obs, seed=args.seed)

        per_grader = compute_per_grader_alignment(split_traces) if args.per_grader else {}
        print_report(train, dev, test, dev_metrics, test_metrics,
                     all_traces, per_grader, corrected_rate, ci_lower, ci_upper)
        return

    # Default: load all labeled traces and auto-split
    traces = load_labeled_traces(db_path, args.experiment_id)
    if not traces:
        print(f"No human-labeled traces found.")
        print(f"Label traces via: python scripts/annotate_batch.py --experiment-id N")
        sys.exit(0)

    print(f"Found {len(traces)} human-labeled traces.")

    if len(traces) < 20:
        print(f"Warning: Only {len(traces)} labeled traces. Need ~100 for reliable results.")
        print(f"Running with full dataset (no train/dev/test split).\n")
        train, dev, test = [], traces, traces
    else:
        train, dev, test = stratified_split(traces, seed=args.seed)

    dev_metrics = compute_metrics(dev)
    test_metrics = compute_metrics(test)

    all_traces = load_all_traces(db_path)
    corrected_rate = ci_lower = ci_upper = None
    if test_metrics["tpr"] and test_metrics["tnr"] and all_traces:
        all_pass = sum(1 for t in all_traces if t["final_pass"])
        p_obs = all_pass / len(all_traces) if all_traces else 0
        corrected_rate = rogan_gladen(p_obs, test_metrics["tpr"], test_metrics["tnr"])
        ci_lower, ci_upper = bootstrap_ci(test, p_obs, seed=args.seed)

    per_grader = compute_per_grader_alignment(test) if args.per_grader else {}
    print_report(train, dev, test, dev_metrics, test_metrics,
                 all_traces, per_grader, corrected_rate, ci_lower, ci_upper)


if __name__ == "__main__":
    main()
