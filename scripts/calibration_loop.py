#!/usr/bin/env python3
"""Judge calibration loop — regrade dev set, compute TPR/TNR, show FP/FN cases.

Implements the Hamel Husain validate-evaluator workflow:
1. Create train/dev/test splits from annotated traces
2. Regrade dev set with current prompts
3. Compute TPR/TNR
4. Show FP/FN cases for prompt iteration
5. Log iteration results

Usage:
    python scripts/calibration_loop.py --create-splits          # First: create splits
    python scripts/calibration_loop.py --run-dev                # Regrade dev + report
    python scripts/calibration_loop.py --run-test               # Final test evaluation
    python scripts/calibration_loop.py --show-errors dev        # Show FP/FN on dev
    python scripts/calibration_loop.py --history                # Show iteration history
"""

import argparse
import asyncio
import json
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent  # repo root
CALIBRATION_DIR = _REPO_ROOT / "calibration"
SPLITS_FILE = CALIBRATION_DIR / "splits.json"
LOG_FILE = CALIBRATION_DIR / "calibration_log.jsonl"
DB_PATH = _REPO_ROOT / "backend" / "eval.db"

# Add eval backend to path for regrade
sys.path.insert(0, str(_REPO_ROOT / "backend"))


def load_annotated_traces(db_path: str) -> list[dict]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT id, case_key, case_type, final_pass, human_pass, final_score,
               composite_scores, experiment_id
        FROM traces
        WHERE human_pass IS NOT NULL AND status IN ('done', 'graded')
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def create_splits(db_path: str, train_frac=0.15, dev_frac=0.45, seed=42):
    """Create stratified train/dev/test splits and save to splits.json."""
    traces = load_annotated_traces(db_path)
    if not traces:
        print("No annotated traces found. Run annotate_batch.py first.")
        return

    rng = np.random.RandomState(seed)

    pos = [t for t in traces if t["human_pass"]]
    neg = [t for t in traces if not t["human_pass"]]

    print(f"Annotated traces: {len(traces)} ({len(pos)} PASS, {len(neg)} FAIL)")

    def _split_ids(data):
        ids = [t["id"] for t in data]
        idx = rng.permutation(len(ids))
        n_train = max(1, int(len(ids) * train_frac))
        n_dev = max(1, int(len(ids) * dev_frac))
        return (
            [ids[i] for i in idx[:n_train]],
            [ids[i] for i in idx[n_train:n_train + n_dev]],
            [ids[i] for i in idx[n_train + n_dev:]],
        )

    pos_train, pos_dev, pos_test = _split_ids(pos)
    neg_train, neg_dev, neg_test = _split_ids(neg)

    splits = {
        "train": pos_train + neg_train,
        "dev": pos_dev + neg_dev,
        "test": pos_test + neg_test,
        "created_at": time.time(),
        "seed": seed,
        "total": len(traces),
    }

    CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
    SPLITS_FILE.write_text(json.dumps(splits, indent=2))

    print(f"Splits saved to {SPLITS_FILE}")
    print(f"  Train: {len(splits['train'])} ({len(pos_train)} P, {len(neg_train)} F)")
    print(f"  Dev:   {len(splits['dev'])} ({len(pos_dev)} P, {len(neg_dev)} F)")
    print(f"  Test:  {len(splits['test'])} ({len(pos_test)} P, {len(neg_test)} F)")


def compute_metrics(db_path: str, trace_ids: list[int]) -> dict:
    """Compute TPR/TNR for a set of trace IDs."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    placeholders = ",".join("?" * len(trace_ids))
    rows = conn.execute(f"""
        SELECT id, case_key, final_pass, human_pass, final_score, composite_scores
        FROM traces WHERE id IN ({placeholders})
    """, trace_ids).fetchall()
    conn.close()

    tp = fp = tn = fn = 0
    fp_cases = []
    fn_cases = []

    for r in rows:
        human = bool(r["human_pass"])
        auto = bool(r["final_pass"])

        if human and auto:
            tp += 1
        elif human and not auto:
            fn += 1
            fn_cases.append(dict(r))
        elif not human and auto:
            fp += 1
            fp_cases.append(dict(r))
        else:
            tn += 1

    tpr = tp / (tp + fn) if (tp + fn) > 0 else None
    tnr = tn / (tn + fp) if (tn + fp) > 0 else None

    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "tpr": round(tpr, 4) if tpr is not None else None,
        "tnr": round(tnr, 4) if tnr is not None else None,
        "total": len(rows),
        "fp_cases": fp_cases,
        "fn_cases": fn_cases,
    }


def print_metrics(metrics: dict, split_name: str):
    """Print confusion matrix and TPR/TNR."""
    print(f"\n{'='*50}")
    print(f"  {split_name.upper()} SET METRICS")
    print(f"{'='*50}")
    print(f"  Confusion Matrix (n={metrics['total']}):")
    print(f"                    Auto PASS  Auto FAIL")
    print(f"    Human PASS      {metrics['tp']:>6}     {metrics['fn']:>6}")
    print(f"    Human FAIL      {metrics['fp']:>6}     {metrics['tn']:>6}")
    print()

    tpr = metrics["tpr"]
    tnr = metrics["tnr"]
    tpr_status = "✅" if tpr and tpr >= 0.9 else "⚠️" if tpr and tpr >= 0.8 else "❌"
    tnr_status = "✅" if tnr and tnr >= 0.9 else "⚠️" if tnr and tnr >= 0.8 else "❌"

    print(f"  TPR (sensitivity): {tpr:.4f} {tpr_status}  (target ≥ 0.90)" if tpr else "  TPR: N/A")
    print(f"  TNR (specificity): {tnr:.4f} {tnr_status}  (target ≥ 0.90)" if tnr else "  TNR: N/A")


def show_error_cases(metrics: dict, split_name: str):
    """Show FP and FN cases for debugging."""
    if metrics["fp_cases"]:
        print(f"\n  --- FALSE POSITIVES (Auto=PASS, Human=FAIL) ---")
        print(f"  Judge is too LENIENT on these {len(metrics['fp_cases'])} cases:")
        for c in metrics["fp_cases"][:5]:
            print(f"    Trace {c['id']}: {c['case_key']}  auto_score={c['final_score']}")
            cs = json.loads(c.get("composite_scores") or "{}")
            for name, data in sorted(cs.items()):
                if isinstance(data, dict) and data.get("score", 0) >= 70:
                    print(f"      {name}: {data['score']} (should be lower)")

    if metrics["fn_cases"]:
        print(f"\n  --- FALSE NEGATIVES (Auto=FAIL, Human=PASS) ---")
        print(f"  Judge is too STRICT on these {len(metrics['fn_cases'])} cases:")
        for c in metrics["fn_cases"][:5]:
            print(f"    Trace {c['id']}: {c['case_key']}  auto_score={c['final_score']}")
            cs = json.loads(c.get("composite_scores") or "{}")
            for name, data in sorted(cs.items()):
                if isinstance(data, dict) and data.get("score", 0) < 70:
                    print(f"      {name}: {data['score']} (should be higher)")


def log_iteration(split_name: str, metrics: dict, notes: str = ""):
    """Append iteration result to calibration log."""
    CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": time.time(),
        "split": split_name,
        "tpr": metrics["tpr"],
        "tnr": metrics["tnr"],
        "tp": metrics["tp"],
        "fp": metrics["fp"],
        "tn": metrics["tn"],
        "fn": metrics["fn"],
        "notes": notes,
    }
    with open(LOG_FILE, 'a') as f:
        f.write(json.dumps(entry) + '\n')
    print(f"\n  Logged to {LOG_FILE}")


def show_history():
    """Print calibration iteration history."""
    if not LOG_FILE.exists():
        print("No calibration history.")
        return

    entries = [json.loads(line) for line in LOG_FILE.read_text().strip().split('\n') if line]
    print(f"\n{'Iter':>4}  {'Split':>5}  {'TPR':>6}  {'TNR':>6}  {'Date':>12}  Notes")
    print("-" * 70)
    for i, e in enumerate(entries, 1):
        date = time.strftime("%Y-%m-%d", time.localtime(e["timestamp"]))
        tpr = f"{e['tpr']:.4f}" if e['tpr'] is not None else "N/A"
        tnr = f"{e['tnr']:.4f}" if e['tnr'] is not None else "N/A"
        print(f"{i:4d}  {e['split']:>5}  {tpr:>6}  {tnr:>6}  {date:>12}  {e.get('notes','')[:30]}")


async def regrade_split(db_path: str, trace_ids: list[int]):
    """Regrade traces in a split using current grading pipeline."""
    from storage.database import init_db
    from storage import queries
    from runner.executor import _grade_single_trace

    await init_db()

    print(f"  Regrading {len(trace_ids)} traces...")
    regraded = 0

    for tid in trace_ids:
        trace = await queries.get_trace(tid)
        if not trace or not trace.guide_text:
            continue

        # Reconstruct case from DB
        all_cases = await queries.get_cases_by_keys([trace.case_key])
        case = {"key": trace.case_key, "query": trace.query, "type": trace.case_type,
                "constraints": {}, "golden_data": {}}
        if all_cases:
            c = all_cases[0]
            case["constraints"] = c.constraints
            case["golden_data"] = c.golden_data

        try:
            grades = await _grade_single_trace(trace, case)
            await queries.update_trace(tid, **{
                "status": "done",
                "l0_scores": grades.get("l0", {}),
                "l1_scores": grades.get("l1", {}).get("breakdown", {}),
                "l2_scores": grades.get("l2"),
                "final_score": grades.get("final_score", 0),
                "final_pass": int(grades.get("final_pass", False)),
                "composite_scores": grades.get("composite_scores", {}),
                "failure_funnel": grades.get("failure_funnel", {}),
                "error_types": grades.get("error_types", []),
                "grading_duration_s": grades.get("grading_duration_s", 0),
                "grading_log": grades.get("grading_log", []),
            })
            regraded += 1
        except Exception as e:
            print(f"    Error regrading trace {tid}: {e}")

    print(f"  Regraded {regraded}/{len(trace_ids)} traces")


def main():
    parser = argparse.ArgumentParser(description="Judge calibration loop")
    parser.add_argument("--db", default=str(DB_PATH))
    parser.add_argument("--create-splits", action="store_true", help="Create train/dev/test splits")
    parser.add_argument("--run-dev", action="store_true", help="Regrade dev set + compute TPR/TNR")
    parser.add_argument("--run-test", action="store_true", help="Run FINAL test evaluation")
    parser.add_argument("--show-errors", choices=["dev", "test"], help="Show FP/FN cases")
    parser.add_argument("--history", action="store_true", help="Show calibration history")
    parser.add_argument("--notes", default="", help="Notes for this iteration")
    args = parser.parse_args()

    if args.create_splits:
        create_splits(args.db)
        return

    if args.history:
        show_history()
        return

    if not SPLITS_FILE.exists():
        print("No splits found. Run --create-splits first.")
        return

    splits = json.loads(SPLITS_FILE.read_text())

    if args.run_dev:
        dev_ids = splits["dev"]
        print(f"Dev set: {len(dev_ids)} traces")
        asyncio.run(regrade_split(args.db, dev_ids))
        metrics = compute_metrics(args.db, dev_ids)
        print_metrics(metrics, "dev")
        show_error_cases(metrics, "dev")
        log_iteration("dev", metrics, args.notes)

    elif args.run_test:
        test_ids = splits["test"]
        print(f"Test set: {len(test_ids)} traces (FINAL — run only when prompts are frozen)")
        confirm = input("Proceed? [y/N]: ").strip().lower()
        if confirm != 'y':
            print("Aborted.")
            return
        asyncio.run(regrade_split(args.db, test_ids))
        metrics = compute_metrics(args.db, test_ids)
        print_metrics(metrics, "test")
        show_error_cases(metrics, "test")
        log_iteration("test", metrics, args.notes)

    elif args.show_errors:
        split_name = args.show_errors
        ids = splits[split_name]
        metrics = compute_metrics(args.db, ids)
        print_metrics(metrics, split_name)
        show_error_cases(metrics, split_name)


if __name__ == "__main__":
    main()
