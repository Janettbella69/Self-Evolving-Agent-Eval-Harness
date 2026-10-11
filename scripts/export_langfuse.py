#!/usr/bin/env python3
"""One-time export: LangFuse → Eval platform SQLite.

Fetches all real user traces from LangFuse, converts to the eval format,
and imports into the eval platform's SQLite database for grading/analysis.

Usage:
    # Set keys temporarily (they won't be saved)
    export LANGFUSE_SECRET_KEY=sk-lf-...
    export LANGFUSE_PUBLIC_KEY=pk-lf-...

    # Run export
    python3 scripts/export_langfuse.py
    python3 scripts/export_langfuse.py --limit 100 --days 30
    python3 scripts/export_langfuse.py --tag shopping-research --grade

After export, you can safely uninstall langfuse:
    pip uninstall langfuse langsmith opentelemetry-api
"""

import argparse
import asyncio
import json
import os
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))


def get_langfuse_client():
    """Create LangFuse client from environment variables."""
    secret = os.environ.get("LANGFUSE_SECRET_KEY", "")
    public = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
    host = os.environ.get("LANGFUSE_BASE_URL", "https://us.cloud.langfuse.com")

    if not secret or not public:
        print("ERROR: Set LANGFUSE_SECRET_KEY and LANGFUSE_PUBLIC_KEY environment variables.")
        print("")
        print("  export LANGFUSE_SECRET_KEY=sk-lf-...")
        print("  export LANGFUSE_PUBLIC_KEY=pk-lf-...")
        print("  python3 scripts/export_langfuse.py")
        sys.exit(1)

    try:
        from langfuse import Langfuse
        client = Langfuse(
            secret_key=secret,
            public_key=public,
            host=host,
        )
        # Test connection
        client.fetch_traces(limit=1)
        print(f"Connected to LangFuse ({host})")
        return client
    except Exception as e:
        print(f"ERROR: Cannot connect to LangFuse: {e}")
        sys.exit(1)


def fetch_all_traces(client, tag: str, days: int, limit: int) -> list:
    """Fetch traces from LangFuse with pagination."""
    from_ts = datetime.now(timezone.utc) - timedelta(days=days)
    all_traces = []
    page = 1

    print(f"Fetching traces (tag={tag}, last {days} days, limit={limit})...")

    while len(all_traces) < limit:
        batch_size = min(50, limit - len(all_traces))
        try:
            result = client.fetch_traces(
                limit=batch_size,
                page=page,
                tags=[tag] if tag else None,
                from_timestamp=from_ts,
            )
            if not result.data:
                break
            all_traces.extend(result.data)
            print(f"  Page {page}: +{len(result.data)} traces (total: {len(all_traces)})")
            if len(result.data) < batch_size:
                break
            page += 1
        except Exception as e:
            print(f"  Fetch error on page {page}: {e}")
            break

    print(f"Fetched {len(all_traces)} traces total")
    return all_traces[:limit]


def make_case_key(query: str) -> str:
    """Generate a case key from a query string."""
    key = query.lower().strip()
    key = re.sub(r"[^a-z0-9\u4e00-\u9fff\s-]", "", key)
    key = re.sub(r"\s+", "-", key)
    return key[:80]


def convert_trace(lf_trace) -> dict | None:
    """Convert a LangFuse trace to the format expected by the eval platform."""
    output = lf_trace.output or {}
    metadata = lf_trace.metadata or {}
    input_data = lf_trace.input or {}

    # Extract query
    query = ""
    if isinstance(input_data, dict):
        query = input_data.get("query", input_data.get("message", ""))
    elif isinstance(input_data, str):
        query = input_data

    if not query:
        return None

    guide_text = output.get("guide", "")
    if not guide_text:
        return None

    # Products and sources (with JSON string guard)
    products = output.get("products", [])
    sources = output.get("sources", [])
    if isinstance(products, str):
        try:
            products = json.loads(products)
        except (json.JSONDecodeError, ValueError):
            products = []
    if isinstance(sources, str):
        try:
            sources = json.loads(sources)
        except (json.JSONDecodeError, ValueError):
            sources = []

    hook_metrics = {
        "search_count": metadata.get("search_count", 0),
        "entity_count": metadata.get("entity_count", 0),
        "entities": metadata.get("entities", []),
        "dimensions_explored": metadata.get("dimensions_explored", 0),
        "dimension_coverage": metadata.get("dimension_coverage", {}),
        "source_domain_count": metadata.get("source_domain_count", 0),
        "source_domains": metadata.get("source_domains", []),
        "failure_count": metadata.get("failure_count", 0),
        "product_count": len(products) if isinstance(products, list) else 0,
    }

    usage = getattr(lf_trace, "usage", None) or {}
    input_tokens = 0
    output_tokens = 0
    if isinstance(usage, dict):
        input_tokens = usage.get("input", usage.get("promptTokens", 0)) or 0
        output_tokens = usage.get("output", usage.get("completionTokens", 0)) or 0

    duration = output.get("elapsed_s", 0) or 0
    lf_id = getattr(lf_trace, "id", "")
    created = getattr(lf_trace, "timestamp", None)
    model = getattr(lf_trace, "model", "") or ""

    return {
        "case_key": make_case_key(query),
        "query": query,
        "duration_s": duration,
        "guide_text": guide_text,
        "products": products,
        "sources": sources,
        "hook_metrics": hook_metrics,
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "turn_count": 0,
        "tool_names": [],
        "langfuse_trace_id": lf_id,
        "created_at": created.timestamp() if created else time.time(),
    }


async def import_to_db(traces: list[dict], dataset_name: str, run_grading: bool) -> int:
    """Import converted traces into the SQLite database."""
    from storage.database import init_db
    from storage import queries

    await init_db()

    # Create dataset
    existing = await queries.get_dataset_by_name(dataset_name)
    if existing:
        dataset_name = f"{dataset_name}-{int(time.time()) % 10000}"

    dataset_id = await queries.create_dataset(
        name=dataset_name,
        description=f"Production traces exported from LangFuse ({len(traces)} traces)",
        suite_type="capability",
    )

    exp_tag = f"langfuse-export-{datetime.now().strftime('%Y%m%d-%H%M')}"
    config = {
        "cases": None,
        "trials": 1,
        "concurrency": 1,
        "judge_enabled": False,
        "source": "langfuse-export",
        "mode": "production",
    }
    experiment_id = await queries.create_experiment(dataset_id, exp_tag, config)

    imported = 0
    skipped = 0

    for trace_data in traces:
        try:
            case_key = trace_data["case_key"]
            await queries.upsert_case(
                dataset_id=dataset_id,
                key=case_key,
                query=trace_data["query"],
                case_type="production",
                constraints={},
                golden_data={},
            )
            trace_id = await queries.create_trace(
                experiment_id=experiment_id,
                case_key=case_key,
                trial_num=1,
                query=trace_data["query"],
                case_type="production",
            )
            await queries.update_trace(trace_id, **{
                "status": "collected",
                "duration_s": trace_data["duration_s"],
                "guide_text": trace_data["guide_text"],
                "products": trace_data["products"],
                "sources": trace_data["sources"],
                "events": [],
                "hook_metrics": trace_data["hook_metrics"],
                "error_events": [],
                "clarification": None,
                "prompt_version": "",
                "model": trace_data.get("model", ""),
                "input_tokens": trace_data.get("input_tokens", 0),
                "output_tokens": trace_data.get("output_tokens", 0),
                "turn_count": trace_data.get("turn_count", 0),
                "system_prompt": "",
                "tool_names": trace_data.get("tool_names", []),
            })
            imported += 1
        except Exception as e:
            print(f"  Skip: {trace_data['case_key'][:40]} - {e}")
            skipped += 1

    print(f"\nImported {imported} traces (skipped {skipped})")
    print(f"  Dataset: '{dataset_name}' (id={dataset_id})")
    print(f"  Experiment: #{experiment_id} (tag={exp_tag})")

    if run_grading and imported > 0:
        print(f"\nRunning grading pipeline on {imported} traces...")
        from runner.executor import regrade_experiment
        count = await regrade_experiment(experiment_id)
        summary = await queries.compute_experiment_summary(experiment_id)
        print(f"  Graded {count} traces")
        print(f"  Avg score: {summary.avg_score}")
        print(f"  Pass rate: {summary.passed}/{summary.total_cases}")
    else:
        summary = await queries.compute_experiment_summary(experiment_id)
        await queries.update_experiment_status(
            experiment_id, "complete",
            summary=summary.model_dump(),
            finished_at=time.time(),
        )

    # Save JSON backup
    backup_path = Path(__file__).parent / "langfuse_export.json"
    with open(backup_path, "w") as f:
        json.dump([{
            "case_key": t["case_key"],
            "query": t["query"],
            "duration_s": t["duration_s"],
            "guide_text": t["guide_text"][:5000],
            "products_count": len(t["products"]),
            "sources_count": len(t["sources"]),
            "model": t.get("model", ""),
            "langfuse_trace_id": t.get("langfuse_trace_id", ""),
        } for t in traces], f, indent=2, ensure_ascii=False)
    print(f"\nJSON backup saved: {backup_path}")

    return experiment_id


def main():
    parser = argparse.ArgumentParser(description="Export LangFuse traces to eval platform")
    parser.add_argument("--limit", type=int, default=200, help="Max traces to export")
    parser.add_argument("--days", type=int, default=90, help="Look back N days")
    parser.add_argument("--tag", default="shopping-research", help="LangFuse tag filter")
    parser.add_argument("--dataset-name", default="", help="Dataset name (auto if empty)")
    parser.add_argument("--grade", action="store_true", help="Run grading after import")
    args = parser.parse_args()

    print("=" * 50)
    print("  LangFuse -> Eval Platform Export")
    print("=" * 50)

    client = get_langfuse_client()
    lf_traces = fetch_all_traces(client, args.tag, args.days, args.limit)
    if not lf_traces:
        print("No traces found. Nothing to export.")
        return

    converted = []
    for lf_trace in lf_traces:
        data = convert_trace(lf_trace)
        if data:
            converted.append(data)

    print(f"\nConverted {len(converted)}/{len(lf_traces)} traces")

    if not converted:
        print("No valid traces to import.")
        return

    print("\nSample queries:")
    for t in converted[:5]:
        print(f"  - {t['query'][:70]}  ({len(t['products'])} products, {len(t['sources'])} sources)")

    dataset_name = args.dataset_name or f"langfuse-export-{datetime.now().strftime('%Y%m%d')}"
    asyncio.run(import_to_db(converted, dataset_name, args.grade))

    print("\n" + "=" * 50)
    print("  Export complete!")
    print("  View in the eval UI (frontend-v2: npm run dev → http://localhost:5200)")
    print("=" * 50)


if __name__ == "__main__":
    main()
