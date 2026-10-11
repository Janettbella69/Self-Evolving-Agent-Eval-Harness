"""GPT-5.4 pre-labeling for human review.

Reads completed traces from eval.db, sends each to GPT-5.4 for a binary
Pass/Fail verdict + reasoning. Outputs a JSON file for human review.

Usage:
    python3 prelabel_traces.py [--limit 100] [--output prelabels.json]
"""

import argparse
import asyncio
import json
import os
import sqlite3
import sys
import time
from pathlib import Path

# Add backend to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)  # optional repo-root .env
load_dotenv(Path(__file__).resolve().parent.parent / "backend" / ".env", override=True)


PRELABEL_PROMPT = """You are a shopping guide quality evaluator. You will see a buyer's guide produced by an AI shopping research agent, along with the original query and any products/sources found.

## Task
Determine: Is this a GOOD buying guide that helps the user make a purchase decision?

## PASS Definition
The guide:
- Addresses the user's actual query/needs
- Recommends specific products with prices
- Cites credible sources (not just marketing)
- Provides enough detail to make a purchase decision
- Mentions trade-offs or limitations

## FAIL Definition
The guide fails if ANY of these are true:
- Wrong product category or doesn't address the query
- No specific products recommended
- No prices or purchase paths
- Mostly vague/generic content without specific evidence
- Critical factual errors visible (wrong specs, impossible claims)
- Guide is too short (<500 chars) or clearly truncated

## Output
Respond with EXACTLY this JSON format (no other text):
{
  "verdict": "PASS" or "FAIL",
  "confidence": "high" or "medium" or "low",
  "reasoning": "1-2 sentence explanation",
  "failure_type": "none" or one of: wrong_category, no_products, no_prices, vague_content, factual_error, truncated, other"
}"""


async def prelabel_one(client, trace: dict, model: str) -> dict:
    """Pre-label a single trace with GPT-5.4."""
    guide = trace["guide_text"][:8000] if trace["guide_text"] else "(empty)"
    products = trace["products"][:5] if trace["products"] else []
    sources_count = len(trace["sources"]) if trace["sources"] else 0

    products_summary = ""
    for i, p in enumerate(products, 1):
        if isinstance(p, dict):
            name = p.get("name", "?")
            price = p.get("price", "N/A")
            products_summary += f"{i}. {name} — {price}\n"

    user_msg = f"""## Query
{trace['query']}

## Products Found ({len(products)} shown, {trace['product_count']} total)
{products_summary or '(none)'}

## Sources: {sources_count} cited

## Guide Text ({len(trace['guide_text'] or '')} chars)
{guide}"""

    try:
        resp = await client.chat.completions.create(
            model=model,
            max_completion_tokens=300,
            temperature=0.1,
            messages=[
                {"role": "system", "content": PRELABEL_PROMPT},
                {"role": "user", "content": user_msg},
            ],
        )
        text = resp.choices[0].message.content.strip()

        # Parse JSON from response
        # Handle potential markdown code blocks
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        result = json.loads(text)

        return {
            "trace_id": trace["id"],
            "case_key": trace["case_key"],
            "query": trace["query"][:200],
            "agent_verdict": "PASS" if trace["final_pass"] else "FAIL",
            "agent_score": trace["final_score"],
            "prelabel": result,
            "guide_length": len(trace["guide_text"] or ""),
            "product_count": trace["product_count"],
            "source_count": sources_count,
        }
    except Exception as e:
        return {
            "trace_id": trace["id"],
            "case_key": trace["case_key"],
            "query": trace["query"][:200],
            "agent_verdict": "PASS" if trace["final_pass"] else "FAIL",
            "agent_score": trace["final_score"],
            "prelabel": {"verdict": "ERROR", "reasoning": str(e)[:200]},
            "error": True,
        }


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument("--output", default="prelabels.json")
    parser.add_argument("--concurrency", type=int, default=5)
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set")
        sys.exit(1)

    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=api_key)

    # Load traces from DB
    db_path = Path(__file__).resolve().parent.parent / "backend" / "eval.db"
    db = sqlite3.connect(str(db_path))
    db.row_factory = sqlite3.Row

    rows = db.execute("""
        SELECT id, case_key, query, status, final_score, final_pass,
               guide_text, products, sources
        FROM traces
        WHERE status IN ('done', 'graded', 'collected')
        ORDER BY id
        LIMIT ?
    """, (args.limit,)).fetchall()
    db.close()

    traces = []
    for r in rows:
        products = json.loads(r["products"] or "[]")
        sources = json.loads(r["sources"] or "[]")
        traces.append({
            "id": r["id"],
            "case_key": r["case_key"],
            "query": r["query"],
            "final_score": r["final_score"],
            "final_pass": bool(r["final_pass"]),
            "guide_text": r["guide_text"] or "",
            "products": products if isinstance(products, list) else [],
            "sources": sources if isinstance(sources, list) else [],
            "product_count": len(products) if isinstance(products, list) else 0,
        })

    print(f"Pre-labeling {len(traces)} traces with {args.model}...")
    print(f"Concurrency: {args.concurrency}")

    # Process in batches
    semaphore = asyncio.Semaphore(args.concurrency)
    results = []
    start = time.time()

    async def process(trace):
        async with semaphore:
            return await prelabel_one(client, trace, args.model)

    tasks = [process(t) for t in traces]
    for i, coro in enumerate(asyncio.as_completed(tasks)):
        result = await coro
        results.append(result)
        verdict = result.get("prelabel", {}).get("verdict", "?")
        print(f"  [{i+1}/{len(traces)}] trace {result['trace_id']}: "
              f"agent={result['agent_verdict']} → prelabel={verdict}")

    elapsed = time.time() - start

    # Sort by trace_id
    results.sort(key=lambda x: x["trace_id"])

    # Summary
    verdicts = [r["prelabel"].get("verdict", "ERROR") for r in results]
    pass_count = verdicts.count("PASS")
    fail_count = verdicts.count("FAIL")
    error_count = verdicts.count("ERROR")

    # Disagreements (where prelabel != agent verdict)
    disagreements = [
        r for r in results
        if r.get("prelabel", {}).get("verdict") in ("PASS", "FAIL")
        and r["prelabel"]["verdict"] != r["agent_verdict"]
    ]

    summary = {
        "model": args.model,
        "total": len(results),
        "pass": pass_count,
        "fail": fail_count,
        "errors": error_count,
        "disagreements": len(disagreements),
        "elapsed_s": round(elapsed, 1),
    }

    output = {
        "summary": summary,
        "results": results,
        "disagreements": [
            {
                "trace_id": d["trace_id"],
                "case_key": d["case_key"],
                "query": d["query"],
                "agent": d["agent_verdict"],
                "prelabel": d["prelabel"]["verdict"],
                "reasoning": d["prelabel"].get("reasoning", ""),
            }
            for d in disagreements
        ],
    }

    out_path = Path(__file__).resolve().parent / args.output
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2))

    print(f"\n{'='*50}")
    print(f"Done in {elapsed:.1f}s")
    print(f"Results: {pass_count} PASS / {fail_count} FAIL / {error_count} ERROR")
    print(f"Disagreements with agent: {len(disagreements)} ({len(disagreements)*100//max(len(results),1)}%)")
    print(f"Output: {out_path}")
    print(f"\nNext step: Human review the {len(disagreements)} disagreements first (highest value).")


if __name__ == "__main__":
    asyncio.run(main())
