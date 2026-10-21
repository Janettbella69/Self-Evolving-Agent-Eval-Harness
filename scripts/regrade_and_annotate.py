#!/usr/bin/env python3
"""Regrade existing traces (code-only) + sample traces for human annotation.

Phase 1: Code-only regrade
  - Re-runs code graders on all done/collected traces
  - Skips LLM graders (JUDGE_ENABLED=false) to avoid API costs
  - Updates composite_scores with code-only weights

Phase 2: Annotation sampling
  - Stratified sampling across score ranges + case types + languages
  - Outputs annotation queue as JSON for batch review

Usage:
    python scripts/regrade_and_annotate.py regrade --experiment-id 66
    python scripts/regrade_and_annotate.py regrade --all
    python scripts/regrade_and_annotate.py sample --n 50 [--experiment-id 66]
    python scripts/regrade_and_annotate.py annotate --trace-id 285 --verdict pass
    python scripts/regrade_and_annotate.py annotate --batch annotations.json
"""

import argparse
import asyncio
import json
import os
import sqlite3
import sys
from pathlib import Path

# Add eval backend to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))


# ── Phase 1: Code-only Regrade ──────────────────────────────────────

async def regrade_code_only(experiment_id: int | None = None, limit: int = 0):
    """Re-run code graders on existing traces (skip LLM graders).

    This is fast and free — no API calls needed.
    """
    # Force code-only grading BEFORE any config import
    os.environ["JUDGE_ENABLED"] = "false"

    from storage import queries, database
    from graders.pipeline import grade_trace as grade_trace_pipeline
    from runner.collector import CollectedResult

    # Monkey-patch config to ensure LLM graders are disabled
    # (load_dotenv(override=True) in config.py may have re-enabled it from .env)
    import config
    config.JUDGE_ENABLED = False

    await database.init_db()

    # Build case map from all datasets
    db = await database.get_db()

    if experiment_id:
        rows = await db.execute_fetchall(
            "SELECT * FROM traces WHERE experiment_id=? AND status IN ('done', 'collected', 'graded') ORDER BY id",
            (experiment_id,),
        )
    else:
        rows = await db.execute_fetchall(
            "SELECT * FROM traces WHERE status IN ('done', 'collected') ORDER BY id",
        )

    if limit:
        rows = rows[:limit]

    # Load all cases for golden_data
    case_rows = await db.execute_fetchall("SELECT * FROM cases")
    case_map = {}
    for cr in case_rows:
        cr_dict = dict(cr)
        case_map[cr_dict["key"]] = {
            "key": cr_dict["key"],
            "query": cr_dict.get("query", ""),
            "type": cr_dict.get("type", "clear_en"),
            "constraints": json.loads(cr_dict.get("constraints", "{}") or "{}"),
            "golden_data": json.loads(cr_dict.get("golden_data", "{}") or "{}"),
        }

    total = len(rows)
    print(f"Regrading {total} traces (code-only)...")

    success = 0
    errors = 0
    scores = []

    for i, row in enumerate(rows):
        row_dict = dict(row)
        trace_id = row_dict["id"]
        case_key = row_dict.get("case_key", "")

        case = case_map.get(case_key, {
            "key": case_key,
            "query": row_dict.get("query", ""),
            "type": row_dict.get("case_type", "clear_en"),
            "constraints": {},
            "golden_data": {},
        })

        try:
            result = CollectedResult(
                guide_text=row_dict.get("guide_text", ""),
                products=json.loads(row_dict.get("products", "[]") or "[]"),
                sources=json.loads(row_dict.get("sources", "[]") or "[]"),
                events=json.loads(row_dict.get("events", "[]") or "[]"),
                hook_metrics=json.loads(row_dict.get("hook_metrics", "{}") or "{}"),
                error_events=json.loads(row_dict.get("error_events", "[]") or "[]"),
                clarification=json.loads(row_dict.get("clarification", "null") or "null"),
                duration_s=row_dict.get("duration_s", 0) or 0,
                input_tokens=row_dict.get("input_tokens", 0) or 0,
                output_tokens=row_dict.get("output_tokens", 0) or 0,
                turn_count=row_dict.get("turn_count", 0) or 0,
                tool_names=json.loads(row_dict.get("tool_names", "[]") or "[]"),
            )

            grades = await grade_trace_pipeline(result, case)

            # Update trace
            await db.execute(
                """UPDATE traces SET
                    status='graded',
                    final_score=?, final_pass=?,
                    checks=?,
                    composite_scores=?,
                    failure_funnel=?,
                    error_types=?,
                    grading_duration_s=?,
                    grading_log=?
                WHERE id=?""",
                (
                    grades.get("final_score", 0),
                    int(grades.get("final_pass", False)),
                    json.dumps(grades.get("checks", {})),
                    json.dumps(grades.get("composite_scores", {})),
                    json.dumps(grades.get("failure_funnel", {})),
                    json.dumps(grades.get("error_types", [])),
                    grades.get("grading_duration_s", 0),
                    json.dumps(grades.get("grading_log", [])),
                    trace_id,
                ),
            )
            await db.commit()

            score = grades.get("final_score", 0)
            scores.append(score)
            success += 1

            if (i + 1) % 20 == 0 or i == total - 1:
                avg = sum(scores) / len(scores) if scores else 0
                print(f"  [{i+1}/{total}] avg_score={avg:.1f}, pass_rate={sum(1 for s in scores if s >= 60)/len(scores)*100:.0f}%")

        except Exception as e:
            errors += 1
            print(f"  Error grading trace {trace_id}: {e}")

    print(f"\nDone: {success} regraded, {errors} errors")
    if scores:
        avg = sum(scores) / len(scores)
        pass_rate = sum(1 for s in scores if s >= 60) / len(scores) * 100
        print(f"Score stats: avg={avg:.1f}, pass_rate={pass_rate:.0f}%, min={min(scores):.1f}, max={max(scores):.1f}")
    else:
        print("No traces were successfully graded.")


# ── Phase 2: Annotation Sampling ────────────────────────────────────

def sample_for_annotation(n: int = 50, experiment_id: int | None = None):
    """Stratified sampling of traces for human annotation.

    Strategy:
    - 20% random (discover unknown issues)
    - 20% highest scores (verify true positives)
    - 20% lowest scores (verify true negatives)
    - 20% near threshold (calibrate boundary, scores 50-70)
    - 20% diverse case types + languages

    Returns list of trace IDs with metadata.
    """
    import random
    random.seed(42)

    db_path = str(Path(__file__).resolve().parent.parent / "backend" / "eval.db")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    where = "WHERE status IN ('done', 'graded') AND guide_text IS NOT NULL AND LENGTH(guide_text) > 200"
    params = []
    if experiment_id:
        where += " AND experiment_id = ?"
        params.append(experiment_id)

    rows = conn.execute(f"""
        SELECT id, case_key, case_type, final_score, final_pass,
               experiment_id, query, human_pass,
               LENGTH(guide_text) as guide_len
        FROM traces {where}
        ORDER BY id
    """, params).fetchall()

    if not rows:
        print("No eligible traces found.")
        return []

    rows = [dict(r) for r in rows]

    # Exclude already annotated
    unannotated = [r for r in rows if r["human_pass"] is None]
    already = len(rows) - len(unannotated)
    if already:
        print(f"  Skipping {already} already-annotated traces")

    if len(unannotated) < n:
        print(f"  Only {len(unannotated)} unannotated traces available (requested {n})")
        n = len(unannotated)

    bucket_size = n // 5

    # Sort by score
    scored = [r for r in unannotated if r["final_score"] and r["final_score"] > 0]
    scored.sort(key=lambda r: r["final_score"])

    selected_ids = set()

    # Bucket 1: Highest scores (verify true positives)
    top = scored[-bucket_size:] if len(scored) >= bucket_size else scored[-max(1, len(scored)//4):]
    for r in top:
        selected_ids.add(r["id"])

    # Bucket 2: Lowest scores (verify true negatives)
    bottom = scored[:bucket_size] if len(scored) >= bucket_size else scored[:max(1, len(scored)//4)]
    for r in bottom:
        selected_ids.add(r["id"])

    # Bucket 3: Near threshold (calibrate boundary)
    threshold = 60
    near = [r for r in scored if abs(r["final_score"] - threshold) <= 15]
    random.shuffle(near)
    for r in near[:bucket_size]:
        selected_ids.add(r["id"])

    # Bucket 4: Diverse case types
    by_type = {}
    for r in unannotated:
        ct = r["case_type"] or "unknown"
        by_type.setdefault(ct, []).append(r)
    for ct in sorted(by_type.keys()):
        if len(selected_ids) >= n - bucket_size:
            break
        candidates = [r for r in by_type[ct] if r["id"] not in selected_ids]
        if candidates:
            pick = random.choice(candidates)
            selected_ids.add(pick["id"])

    # Bucket 5: Random fill
    remaining = [r for r in unannotated if r["id"] not in selected_ids]
    random.shuffle(remaining)
    for r in remaining:
        if len(selected_ids) >= n:
            break
        selected_ids.add(r["id"])

    # Build output
    selected = [r for r in unannotated if r["id"] in selected_ids]
    selected.sort(key=lambda r: r["final_score"] or 0, reverse=True)

    conn.close()
    return selected


def print_annotation_queue(traces: list[dict], output_file: str | None = None):
    """Print annotation queue as a formatted table + optional JSON export."""
    print(f"\n{'='*80}")
    print(f"  ANNOTATION QUEUE: {len(traces)} traces")
    print(f"{'='*80}")
    print(f"{'ID':>5} {'Score':>6} {'Type':>12} {'Guide':>6} {'Query':>45}")
    print(f"{'-'*5:>5} {'-'*6:>6} {'-'*12:>12} {'-'*6:>6} {'-'*45}")

    for t in traces:
        query = (t.get("query", "") or "")[:45]
        print(f"{t['id']:>5} {t.get('final_score', 0):>6.1f} {(t.get('case_type', '') or ''):>12} {t.get('guide_len', 0):>6} {query}")

    if output_file:
        Path(output_file).write_text(json.dumps(
            [{"id": t["id"], "query": t.get("query", ""), "case_type": t.get("case_type", ""),
              "score": t.get("final_score", 0)} for t in traces],
            indent=2, ensure_ascii=False,
        ))
        print(f"\nExported to {output_file}")

    score_dist = [t.get("final_score", 0) for t in traces if t.get("final_score")]
    if score_dist:
        print(f"\nScore distribution: min={min(score_dist):.1f}, avg={sum(score_dist)/len(score_dist):.1f}, max={max(score_dist):.1f}")

    types = {}
    for t in traces:
        ct = t.get("case_type", "unknown") or "unknown"
        types[ct] = types.get(ct, 0) + 1
    print(f"Case types: {dict(sorted(types.items(), key=lambda x: -x[1]))}")


# ── Phase 3: Annotation ────────────────────────────────────────────

def annotate_single(trace_id: int, verdict: str):
    """Annotate a single trace with PASS/FAIL."""
    db_path = str(Path(__file__).resolve().parent.parent / "backend" / "eval.db")
    conn = sqlite3.connect(db_path)

    human_pass = 1 if verdict.lower() in ("pass", "1", "true", "yes") else 0
    conn.execute(
        "UPDATE traces SET human_pass=? WHERE id=?",
        (human_pass, trace_id),
    )
    conn.commit()
    print(f"Trace {trace_id}: human_pass={human_pass}")
    conn.close()


def annotate_batch(json_file: str):
    """Batch annotate from JSON file: [{"id": 1, "verdict": "pass"}, ...]"""
    data = json.loads(Path(json_file).read_text())
    db_path = str(Path(__file__).resolve().parent.parent / "backend" / "eval.db")
    conn = sqlite3.connect(db_path)

    count = 0
    for item in data:
        trace_id = item["id"]
        verdict = item.get("verdict", item.get("human_pass", ""))
        human_pass = 1 if str(verdict).lower() in ("pass", "1", "true", "yes") else 0
        conn.execute("UPDATE traces SET human_pass=? WHERE id=?", (human_pass, trace_id))
        count += 1

    conn.commit()
    print(f"Annotated {count} traces from {json_file}")
    conn.close()


# ── Interactive annotation helper ───────────────────────────────────

def annotate_interactive(trace_ids: list[int]):
    """Interactive annotation: show trace summary, prompt for PASS/FAIL."""
    db_path = str(Path(__file__).resolve().parent.parent / "backend" / "eval.db")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    count = 0
    for trace_id in trace_ids:
        row = conn.execute(
            "SELECT id, query, case_type, guide_text, final_score, products, sources FROM traces WHERE id=?",
            (trace_id,),
        ).fetchone()
        if not row:
            print(f"Trace {trace_id} not found, skipping")
            continue

        row = dict(row)
        products = json.loads(row.get("products", "[]") or "[]")
        sources = json.loads(row.get("sources", "[]") or "[]")
        guide = row.get("guide_text", "")

        print(f"\n{'='*80}")
        print(f"Trace #{row['id']} | Score: {row.get('final_score', 'N/A')} | Type: {row.get('case_type', 'N/A')}")
        print(f"Query: {row.get('query', 'N/A')}")
        print(f"Products: {len(products)} | Sources: {len(sources)} | Guide: {len(guide)} chars")
        print(f"{'-'*80}")
        # Show first 800 chars of guide
        print(guide[:800])
        if len(guide) > 800:
            print(f"... [{len(guide) - 800} more chars]")
        print(f"{'-'*80}")
        # Show products
        for i, p in enumerate(products[:5], 1):
            name = p.get("name", "?") if isinstance(p, dict) else str(p)
            price = p.get("price", "N/A") if isinstance(p, dict) else "N/A"
            print(f"  {i}. {name} — {price}")
        print(f"{'-'*80}")

        while True:
            verdict = input(f"Verdict (pass/fail/skip/quit): ").strip().lower()
            if verdict in ("pass", "fail", "p", "f"):
                human_pass = 1 if verdict in ("pass", "p") else 0
                conn.execute("UPDATE traces SET human_pass=? WHERE id=?", (human_pass, trace_id))
                conn.commit()
                count += 1
                print(f"  → {'PASS' if human_pass else 'FAIL'}")
                break
            elif verdict == "skip":
                break
            elif verdict in ("quit", "q"):
                print(f"\nAnnotated {count} traces total.")
                conn.close()
                return
            else:
                print("  Enter: pass, fail, skip, or quit")

    print(f"\nAnnotated {count} traces total.")
    conn.close()


# ── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Regrade traces + sample for annotation")
    sub = parser.add_subparsers(dest="command")

    # Regrade
    rg = sub.add_parser("regrade", help="Re-run code graders on traces")
    rg.add_argument("--experiment-id", type=int, help="Specific experiment")
    rg.add_argument("--all", action="store_true", help="Regrade all done/collected traces")
    rg.add_argument("--limit", type=int, default=0, help="Max traces to regrade")

    # Sample
    sp = sub.add_parser("sample", help="Sample traces for annotation")
    sp.add_argument("--n", type=int, default=50, help="Number of traces to sample")
    sp.add_argument("--experiment-id", type=int, help="Filter to experiment")
    sp.add_argument("--output", type=str, help="Export JSON file path")

    # Annotate
    an = sub.add_parser("annotate", help="Annotate traces")
    an.add_argument("--trace-id", type=int, help="Single trace ID")
    an.add_argument("--verdict", type=str, help="pass or fail")
    an.add_argument("--batch", type=str, help="JSON file for batch annotation")
    an.add_argument("--interactive", type=str, help="Comma-separated trace IDs for interactive annotation")

    args = parser.parse_args()

    if args.command == "regrade":
        exp_id = args.experiment_id if not args.all else None
        if not args.all and not args.experiment_id:
            parser.error("Specify --experiment-id or --all")
        asyncio.run(regrade_code_only(exp_id, args.limit))

    elif args.command == "sample":
        traces = sample_for_annotation(args.n, args.experiment_id)
        print_annotation_queue(traces, args.output)

    elif args.command == "annotate":
        if args.batch:
            annotate_batch(args.batch)
        elif args.trace_id and args.verdict:
            annotate_single(args.trace_id, args.verdict)
        elif args.interactive:
            ids = [int(x.strip()) for x in args.interactive.split(",")]
            annotate_interactive(ids)
        else:
            parser.error("Specify --trace-id + --verdict, --batch, or --interactive")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
