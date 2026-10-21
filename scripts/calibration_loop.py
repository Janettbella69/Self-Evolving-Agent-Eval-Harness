#!/usr/bin/env python3
"""Judge calibration loop — regrade dev set, compute TPR/TNR, show disagreements.

Implements the Hamel Husain validate-evaluator workflow:
1. Create train/dev/test splits from annotated traces (by case: no case spans two splits)
2. Regrade dev set with current prompts
3. Compute TPR/TNR, overall and per LLM judge (failure is the positive class)
4. Show missed failures and false alarms for prompt iteration
5. Log iteration results

Usage:
    python scripts/calibration_loop.py --create-splits          # First: create splits
    python scripts/calibration_loop.py --run-dev                # Regrade dev + report
    python scripts/calibration_loop.py --run-test               # Final test evaluation
    python scripts/calibration_loop.py --show-errors dev        # Show disagreements on dev
    python scripts/calibration_loop.py --history                # Show iteration history
"""

import argparse
import asyncio
import json
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent  # repo root
CALIBRATION_DIR = _REPO_ROOT / "calibration"
SPLITS_FILE = CALIBRATION_DIR / "splits.json"
LOG_FILE = CALIBRATION_DIR / "calibration_log.jsonl"
DB_PATH = _REPO_ROOT / "backend" / "eval.db"

# Add eval backend to path for regrade
sys.path.insert(0, str(_REPO_ROOT / "backend"))

from graders.validation import (  # noqa: E402
    MIN_LABELS_PER_CLASS,
    VALIDATION_TARGET,
    VALIDATION_TNR_THRESHOLD,
    VALIDATION_TPR_THRESHOLD,
    cases_in_multiple_splits,
    confusion_from_pairs,
    group_stratified_split,
    human_grader_passed,
    judge_passed,
)


def _loads(raw, default):
    if not raw:
        return default
    if isinstance(raw, (dict, list)):
        return raw
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return default


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
    """Create train/dev/test splits by case, stratified by label, and save to splits.json."""
    traces = load_annotated_traces(db_path)
    if not traces:
        print("No annotated traces found. Run annotate_batch.py first.")
        return

    n_pass = sum(1 for t in traces if t["human_pass"])
    n_cases = len({t["case_key"] for t in traces})
    print(f"Annotated traces: {len(traces)} ({n_pass} PASS, {len(traces) - n_pass} FAIL) "
          f"from {n_cases} cases")

    train, dev, test = group_stratified_split(
        traces, train_frac=train_frac, dev_frac=dev_frac, seed=seed,
    )

    splits = {
        "train": [t["id"] for t in train],
        "dev": [t["id"] for t in dev],
        "test": [t["id"] for t in test],
        "group_by": "case_key",
        "created_at": time.time(),
        "seed": seed,
        "total": len(traces),
    }

    CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
    SPLITS_FILE.write_text(json.dumps(splits, indent=2))

    print(f"Splits saved to {SPLITS_FILE}")
    for name, part in (("Train", train), ("Dev", dev), ("Test", test)):
        p = sum(1 for t in part if t["human_pass"])
        cases = len({t["case_key"] for t in part})
        print(f"  {name + ':':<6} {len(part)} traces, {cases} cases ({p} P, {len(part) - p} F)")
    for name, part in (("Dev", dev), ("Test", test)):
        p = sum(1 for t in part if t["human_pass"])
        if min(p, len(part) - p) < MIN_LABELS_PER_CLASS:
            print(f"  ⚠ {name} has {p} Pass / {len(part) - p} Fail labels; the FAQ suggests "
                  f"{MIN_LABELS_PER_CLASS}-50 of each before trusting its TPR/TNR")


def load_splits(db_path: str) -> dict | None:
    """Read splits.json and warn when a case's traces sit in more than one split."""
    if not SPLITS_FILE.exists():
        print("No splits found. Run --create-splits first.")
        return None
    splits = json.loads(SPLITS_FILE.read_text())

    conn = sqlite3.connect(db_path)
    case_of = dict(conn.execute("SELECT id, case_key FROM traces").fetchall())
    conn.close()
    leaked = cases_in_multiple_splits(splits, case_of)
    if leaked:
        print(f"⚠ {len(leaked)} case(s) have traces in more than one split "
              f"(e.g. {', '.join(leaked[:3])}). Few-shots from train can then leak into dev/test.")
        print("  Re-run --create-splits: splits are now made by case.")
    return splits


def compute_metrics(db_path: str, trace_ids: list[int]) -> dict:
    """Compute TPR/TNR for a set of trace IDs, overall and per LLM judge.

    Failure is the positive class: a missed failure is Auto=PASS on Human=FAIL,
    a false alarm is Auto=FAIL on Human=PASS. Per-judge rows compare a judge's
    verdict with the human's label for that same grader (human_scores).
    """
    rows: list[dict] = []
    if trace_ids:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        placeholders = ",".join("?" * len(trace_ids))
        rows = [dict(r) for r in conn.execute(f"""
            SELECT * FROM traces WHERE id IN ({placeholders}) AND human_pass IS NOT NULL
        """, trace_ids).fetchall()]
        conn.close()

    overall = confusion_from_pairs((bool(r["final_pass"]), bool(r["human_pass"])) for r in rows)

    judge_pairs: dict[str, list[tuple[bool, bool]]] = defaultdict(list)
    for r in rows:
        human_scores = _loads(r.get("human_scores"), {})
        for name, entry in _loads(r.get("composite_scores"), {}).items():
            if not isinstance(entry, dict) or entry.get("category") != "llm":
                continue
            judge = judge_passed(entry)
            label = human_grader_passed(human_scores, name)
            if judge is not None and label is not None:
                judge_pairs[name].append((judge, label))

    return {
        **overall,
        "total": len(rows),
        "missed": [r for r in rows if not r["human_pass"] and r["final_pass"]],
        "false_alarms": [r for r in rows if r["human_pass"] and not r["final_pass"]],
        "per_judge": {name: confusion_from_pairs(pairs) for name, pairs in sorted(judge_pairs.items())},
    }


def _status(rate: float | None, minimum: float) -> str:
    if rate is None:
        return ""
    return "✅" if rate >= VALIDATION_TARGET else "⚠️" if rate >= minimum else "❌"


def print_metrics(metrics: dict, split_name: str):
    """Print confusion matrix and TPR/TNR."""
    print(f"\n{'='*50}")
    print(f"  {split_name.upper()} SET METRICS")
    print(f"{'='*50}")
    print(f"  Confusion Matrix (n={metrics['total']}):")
    print(f"                    Auto PASS  Auto FAIL")
    print(f"    Human PASS      {metrics['tn']:>6}     {metrics['fp']:>6}")
    print(f"    Human FAIL      {metrics['fn']:>6}     {metrics['tp']:>6}")
    print()

    tpr = metrics["tpr"]
    tnr = metrics["tnr"]
    target = f"(target ≥ {VALIDATION_TARGET:.2f}, minimum {VALIDATION_TPR_THRESHOLD:.2f})"
    if tpr is not None:
        print(f"  TPR (failures caught): {tpr:.4f} {_status(tpr, VALIDATION_TPR_THRESHOLD)}  {target}")
    else:
        print("  TPR: N/A (no Human FAIL labels)")
    if tnr is not None:
        print(f"  TNR (passes kept):     {tnr:.4f} {_status(tnr, VALIDATION_TNR_THRESHOLD)}  {target}")
    else:
        print("  TNR: N/A (no Human PASS labels)")
    if min(metrics["human_pass"], metrics["human_fail"]) < MIN_LABELS_PER_CLASS:
        print(f"  ⚠ {metrics['human_pass']} Pass / {metrics['human_fail']} Fail labels; the FAQ suggests "
              f"{MIN_LABELS_PER_CLASS}-50 of each, so treat these rates as rough")

    print_per_judge(metrics["per_judge"])


def print_per_judge(per_judge: dict):
    """Print TPR/TNR for each LLM judge against the human's label for that grader."""
    print(f"\n  Per-judge alignment (judge verdict vs human label for the same grader):")
    if not per_judge:
        print("    No per-grader human labels in this split. Label individual graders in the")
        print("    trace page's Scores tab or with annotate_batch.py to calibrate judges one by one.")
        return
    print(f"    {'Judge':<20} {'n':>4} {'TPR':>6} {'TNR':>6}  Pass/Fail labels")
    for name, m in per_judge.items():
        tpr = f"{m['tpr']:.3f}" if m["tpr"] is not None else "N/A"
        tnr = f"{m['tnr']:.3f}" if m["tnr"] is not None else "N/A"
        few = "  ⚠ few labels" if min(m["human_pass"], m["human_fail"]) < MIN_LABELS_PER_CLASS else ""
        print(f"    {name:<20} {m['n']:>4} {tpr:>6} {tnr:>6}  {m['human_pass']}/{m['human_fail']}{few}")


def show_error_cases(metrics: dict, split_name: str):
    """Show missed failures and false alarms for debugging."""
    if metrics["missed"]:
        print(f"\n  --- MISSED FAILURES (Auto=PASS, Human=FAIL) ---")
        print(f"  Judge is too LENIENT on these {len(metrics['missed'])} cases:")
        for c in metrics["missed"][:5]:
            print(f"    Trace {c['id']}: {c['case_key']}  auto_score={c['final_score']}")
            for name, data in sorted(_loads(c.get("composite_scores"), {}).items()):
                if isinstance(data, dict) and data.get("score", 0) >= 70:
                    print(f"      {name}: {data['score']} (should be lower)")

    if metrics["false_alarms"]:
        print(f"\n  --- FALSE ALARMS (Auto=FAIL, Human=PASS) ---")
        print(f"  Judge is too STRICT on these {len(metrics['false_alarms'])} cases:")
        for c in metrics["false_alarms"][:5]:
            print(f"    Trace {c['id']}: {c['case_key']}  auto_score={c['final_score']}")
            failed_checks = sorted(n for n, ok in _loads(c.get("checks"), {}).items() if not ok)
            if failed_checks:
                for name in failed_checks:
                    print(f"      {name}: failed (should pass)")
                continue
            for name, data in sorted(_loads(c.get("composite_scores"), {}).items()):
                if isinstance(data, dict) and data.get("score", 0) < 70:
                    print(f"      {name}: {data['score']} (should be higher)")


def log_iteration(split_name: str, metrics: dict, notes: str = ""):
    """Append iteration result to calibration log."""
    CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": time.time(),
        "split": split_name,
        "positive_class": "fail",
        "tpr": metrics["tpr"],
        "tnr": metrics["tnr"],
        "tp": metrics["tp"],
        "fp": metrics["fp"],
        "tn": metrics["tn"],
        "fn": metrics["fn"],
        "per_judge": {
            name: {k: m[k] for k in ("tpr", "tnr", "n", "human_pass", "human_fail")}
            for name, m in metrics["per_judge"].items()
        },
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
    print(f"\n{'Iter':>4}  {'Split':>5}  {'TPR':>7}  {'TNR':>7}  {'Date':>12}  Notes")
    print("-" * 70)
    swapped = False
    for i, e in enumerate(entries, 1):
        date = time.strftime("%Y-%m-%d", time.localtime(e["timestamp"]))
        tpr_v, tnr_v, mark = e["tpr"], e["tnr"], " "
        if e.get("positive_class") != "fail":
            # Logged while Pass was the positive class: swap to today's meaning.
            tpr_v, tnr_v, mark, swapped = e["tnr"], e["tpr"], "*", True
        tpr = f"{tpr_v:.4f}" if tpr_v is not None else "N/A"
        tnr = f"{tnr_v:.4f}" if tnr_v is not None else "N/A"
        print(f"{i:4d}  {e['split']:>5}  {tpr:>6}{mark}  {tnr:>6}{mark}  {date:>12}  {e.get('notes','')[:30]}")
    if swapped:
        print("\n* logged with Pass as the positive class; TPR/TNR shown swapped to today's meaning")


async def regrade_split(db_path: str, trace_ids: list[int]):
    """Regrade traces in a split using current grading pipeline."""
    from storage.database import init_db
    from storage import queries
    from runner.executor import _grade_single_trace, _grades_to_trace_update

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
            await queries.update_trace(tid, **_grades_to_trace_update(grades))
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
    parser.add_argument("--show-errors", choices=["dev", "test"], help="Show missed failures and false alarms")
    parser.add_argument("--history", action="store_true", help="Show calibration history")
    parser.add_argument("--notes", default="", help="Notes for this iteration")
    args = parser.parse_args()

    if args.create_splits:
        create_splits(args.db)
        return

    if args.history:
        show_history()
        return

    splits = load_splits(args.db)
    if splits is None:
        return

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
