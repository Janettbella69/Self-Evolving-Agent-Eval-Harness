"""Re-grade all collected/graded traces with current model config.

Runs the grading pipeline directly (no eval backend needed).
Uses GPT-5.4 for LLM judges via direct OpenAI API.

Usage:
    python3 regrade_all.py [--experiment-id 28] [--dry-run]
"""

import argparse
import asyncio
import json
import os
import sqlite3
import sys
import time
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)  # optional repo-root .env
load_dotenv(Path(__file__).resolve().parent.parent / "backend" / ".env", override=True)

# Prevent nested Agent SDK session conflict (critical when run from Claude Code)
os.environ.pop("CLAUDECODE", None)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-id", type=int, help="Regrade specific experiment")
    parser.add_argument("--limit", type=int, default=100, help="Max traces to regrade")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be regraded")
    parser.add_argument("--code-only", action="store_true", help="Only run code graders (no LLM)")
    args = parser.parse_args()

    # Import after path setup
    from storage.database import init_db
    from storage import queries
    from graders.pipeline import grade_trace as grade_trace_pipeline
    from runner.collector import CollectedResult

    await init_db()

    # Find traces to regrade
    db = await queries.get_db()
    if args.experiment_id:
        rows = await db.execute_fetchall(
            """SELECT t.id, t.experiment_id, t.case_key, t.query, t.case_type,
                      t.guide_text, t.products, t.sources, t.events,
                      t.hook_metrics, t.error_events, t.duration_s,
                      t.prompt_version, t.model, t.input_tokens, t.output_tokens,
                      t.turn_count, t.system_prompt, t.tool_names, t.clarification,
                      t.final_score, t.final_pass
               FROM traces t
               WHERE t.experiment_id = ? AND t.status IN ('done', 'graded', 'collected')
               LIMIT ?""",
            (args.experiment_id, args.limit),
        )
    else:
        rows = await db.execute_fetchall(
            """SELECT t.id, t.experiment_id, t.case_key, t.query, t.case_type,
                      t.guide_text, t.products, t.sources, t.events,
                      t.hook_metrics, t.error_events, t.duration_s,
                      t.prompt_version, t.model, t.input_tokens, t.output_tokens,
                      t.turn_count, t.system_prompt, t.tool_names, t.clarification,
                      t.final_score, t.final_pass
               FROM traces t
               WHERE t.status IN ('done', 'graded', 'collected')
                 AND LENGTH(t.guide_text) > 500
               ORDER BY t.id
               LIMIT ?""",
            (args.limit,),
        )

    print(f"Found {len(rows)} traces to regrade")

    if args.dry_run:
        for r in rows[:10]:
            print(f"  #{r['id']} [{r['case_type']}] {r['query'][:60]}  "
                  f"(score={r['final_score']}, pass={r['final_pass']})")
        return

    # Get case data (golden_data) for each trace
    case_cache = {}

    regraded = 0
    errors = 0
    score_changes = []

    for i, r in enumerate(rows):
        trace_id = r["id"]
        old_score = r["final_score"] or 0
        old_pass = bool(r["final_pass"])

        # Build CollectedResult
        result = CollectedResult(
            guide_text=r["guide_text"] or "",
            products=json.loads(r["products"] or "[]"),
            sources=json.loads(r["sources"] or "[]"),
            events=json.loads(r["events"] or "[]"),
            hook_metrics=json.loads(r["hook_metrics"] or "{}"),
            error_events=json.loads(r["error_events"] or "[]"),
            duration_s=r["duration_s"] or 0,
            prompt_version=r["prompt_version"] or "",
            model=r["model"] or "",
            input_tokens=r["input_tokens"] or 0,
            output_tokens=r["output_tokens"] or 0,
            turn_count=r["turn_count"] or 0,
            system_prompt=r["system_prompt"] or "",
            tool_names=json.loads(r["tool_names"] or "[]"),
            clarification=json.loads(r["clarification"] or "null"),
        )

        # Get case golden_data
        cache_key = (r["experiment_id"], r["case_key"])
        if cache_key not in case_cache:
            exp = await queries.get_experiment(r["experiment_id"])
            if exp:
                cases = await queries.get_cases(exp.dataset_id)
                for c in cases:
                    case_cache[(r["experiment_id"], c.key)] = {
                        "key": c.key, "query": c.query, "type": c.type,
                        "constraints": c.constraints, "golden_data": c.golden_data,
                    }

        case = case_cache.get(cache_key, {
            "key": r["case_key"], "query": r["query"],
            "type": r["case_type"], "constraints": {}, "golden_data": {},
        })

        if args.code_only:
            # Override: disable LLM graders
            os.environ["JUDGE_ENABLED"] = ""

        try:
            t0 = time.time()
            grades = await grade_trace_pipeline(result, case)
            elapsed = time.time() - t0

            new_score = grades.get("final_score", 0)
            new_pass = grades.get("final_pass", False)

            # Update trace
            update = {
                "status": "graded",
                "final_score": new_score,
                "final_pass": new_pass,
                "l0_scores": grades.get("l0", {}),
                "l1_scores": grades.get("l1", {}),
                "composite_scores": grades.get("composite_scores", {}),
                "failure_funnel": grades.get("failure_funnel", {}),
                "error_types": grades.get("error_types", []),
                "grading_duration_s": elapsed,
                "grading_log": grades.get("grading_log", []),
                "judge_prompts": grades.get("judge_prompts", {}),
            }
            if grades.get("judge_prompt_version"):
                update["judge_prompt_version"] = grades["judge_prompt_version"]

            await queries.update_trace(trace_id, **update)
            regraded += 1

            delta = new_score - old_score
            score_changes.append(delta)
            marker = "↑" if delta > 0 else ("↓" if delta < 0 else "=")
            pass_change = ""
            if new_pass != old_pass:
                pass_change = f" {'FAIL→PASS' if new_pass else 'PASS→FAIL'}"

            print(f"  [{i+1}/{len(rows)}] #{trace_id}: {old_score:.0f} → {new_score:.0f} {marker}{pass_change} ({elapsed:.1f}s)")

        except Exception as e:
            errors += 1
            print(f"  [{i+1}/{len(rows)}] #{trace_id}: ERROR — {str(e)[:100]}")

    # Recompute experiment summaries
    exp_ids = set(r["experiment_id"] for r in rows)
    for eid in exp_ids:
        try:
            summary = await queries.compute_experiment_summary(eid)
            await queries.update_experiment_status(
                eid, "complete",
                summary=summary.model_dump(),
                finished_at=time.time(),
            )
        except Exception:
            pass

    # Summary
    print(f"\n{'='*50}")
    print(f"Regraded: {regraded}, Errors: {errors}")
    if score_changes:
        avg_delta = sum(score_changes) / len(score_changes)
        improved = sum(1 for d in score_changes if d > 0)
        worsened = sum(1 for d in score_changes if d < 0)
        unchanged = sum(1 for d in score_changes if d == 0)
        print(f"Score changes: avg={avg_delta:+.1f}, improved={improved}, worsened={worsened}, unchanged={unchanged}")


if __name__ == "__main__":
    asyncio.run(main())
