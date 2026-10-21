#!/usr/bin/env python3
"""Regrade traces WITH LLM judges enabled (full pipeline).

Only targets traces with actual guide text (>500 chars).
Uses JUDGE_ENABLED=true from .env — does NOT override it.

Usage:
    python scripts/regrade_with_llm.py [--limit N] [--dry-run]
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)  # optional repo-root .env
load_dotenv(Path(__file__).resolve().parent.parent / "backend" / ".env", override=True)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
os.environ.pop("CLAUDECODE", None)


async def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    # Do NOT override JUDGE_ENABLED — let .env value (true) take effect
    from storage import database
    from graders.pipeline import grade_trace as grade_trace_pipeline
    from runner.collector import CollectedResult
    from config import JUDGE_ENABLED, GRADING_MODEL, PASS_THRESHOLD

    print(f"Config: JUDGE_ENABLED={JUDGE_ENABLED}, MODEL={GRADING_MODEL}, THRESHOLD={PASS_THRESHOLD}")

    if not JUDGE_ENABLED:
        print("ERROR: JUDGE_ENABLED is False. Set JUDGE_ENABLED=true in .env")
        sys.exit(1)

    await database.init_db()
    db = await database.get_db()

    # Only traces with actual output
    rows = await db.execute_fetchall(
        "SELECT * FROM traces WHERE LENGTH(guide_text) > 500 ORDER BY id"
    )
    if args.limit:
        rows = rows[:args.limit]

    # Load cases for golden_data
    case_rows = await db.execute_fetchall("SELECT * FROM cases")
    case_map = {}
    for cr in case_rows:
        cr = dict(cr)
        case_map[cr["key"]] = {
            "key": cr["key"], "query": cr.get("query", ""),
            "type": cr.get("type", "clear_en"),
            "constraints": json.loads(cr.get("constraints", "{}") or "{}"),
            "golden_data": json.loads(cr.get("golden_data", "{}") or "{}"),
        }

    total = len(rows)
    print(f"\nRegrading {total} traces with FULL pipeline (code + LLM)...")
    print(f"Estimated time: ~{total * 2 * 30 // 60} min\n")

    if args.dry_run:
        print("DRY RUN — no changes made")
        return

    success = 0
    errors = 0
    scores = []
    t0 = time.time()

    for i, row in enumerate(rows):
        row = dict(row)
        trace_id = row["id"]
        case_key = row.get("case_key", "")
        case = case_map.get(case_key, {
            "key": case_key, "query": row.get("query", ""),
            "type": row.get("case_type", "clear_en"),
            "constraints": {}, "golden_data": {},
        })

        try:
            result = CollectedResult(
                guide_text=row.get("guide_text", ""),
                products=json.loads(row.get("products", "[]") or "[]"),
                sources=json.loads(row.get("sources", "[]") or "[]"),
                events=json.loads(row.get("events", "[]") or "[]"),
                hook_metrics=json.loads(row.get("hook_metrics", "{}") or "{}"),
                error_events=json.loads(row.get("error_events", "[]") or "[]"),
                clarification=json.loads(row.get("clarification", "null") or "null"),
                duration_s=row.get("duration_s", 0) or 0,
                input_tokens=row.get("input_tokens", 0) or 0,
                output_tokens=row.get("output_tokens", 0) or 0,
                turn_count=row.get("turn_count", 0) or 0,
                tool_names=json.loads(row.get("tool_names", "[]") or "[]"),
            )

            gt = time.time()
            grades = await grade_trace_pipeline(result, case)
            duration = time.time() - gt

            score = grades.get("final_score", 0)
            passed = int(grades.get("final_pass", False))

            # Extract LLM grader info from grading_log
            grading_log = grades.get("grading_log", [])
            llm_entries = [e for e in grading_log if e.get("category") == "llm"]
            llm_summary = " ".join(
                f"{e['step']}={e.get('score', '?'):.0f}" if isinstance(e.get('score'), (int, float)) else f"{e['step']}={e.get('status', '?')}"
                for e in llm_entries
            )

            await db.execute(
                """UPDATE traces SET status='graded', final_score=?, final_pass=?, checks=?,
                   composite_scores=?, failure_funnel=?, error_types=?,
                   grading_duration_s=?, grading_log=? WHERE id=?""",
                (score, passed,
                 json.dumps(grades.get("checks", {})),
                 json.dumps(grades.get("composite_scores", {})),
                 json.dumps(grades.get("failure_funnel", {})),
                 json.dumps(grades.get("error_types", [])),
                 grades.get("grading_duration_s", 0),
                 json.dumps(grading_log),
                 trace_id))
            await db.commit()

            scores.append((trace_id, score, passed))
            success += 1

            elapsed = time.time() - t0
            eta = (elapsed / (i + 1)) * (total - i - 1)
            status = "PASS" if passed else "FAIL"
            print(f"  [{i+1}/{total}] #{trace_id} {score:>5.1f} {status} ({duration:.0f}s) LLM:[{llm_summary}] ETA:{eta/60:.0f}m", flush=True)

        except Exception as e:
            errors += 1
            print(f"  [{i+1}/{total}] #{trace_id} ERROR: {str(e)[:100]}", flush=True)

    # Summary
    elapsed = time.time() - t0
    sc = [s for _, s, _ in scores]
    passes = sum(1 for _, _, p in scores if p)

    print(f"\n{'=' * 65}")
    print(f"FULL PIPELINE REGRADE COMPLETE")
    print(f"{'=' * 65}")
    print(f"Traces: {success}/{total} ({errors} errors)")
    print(f"Time: {elapsed/60:.1f} min ({elapsed/max(success,1):.1f}s per trace)")
    if sc:
        print(f"Scores: avg={sum(sc)/len(sc):.1f}, pass={passes}/{len(sc)}, range=[{min(sc):.1f}, {max(sc):.1f}]")
    else:
        print("Scores: no traces were successfully graded")
    print(f"\nPer-trace results:")
    for tid, sc_val, pa in sorted(scores, key=lambda x: -x[1]):
        print(f"  {tid:>5} {sc_val:>6.1f} {'PASS' if pa else 'FAIL'}")


if __name__ == "__main__":
    asyncio.run(main())
