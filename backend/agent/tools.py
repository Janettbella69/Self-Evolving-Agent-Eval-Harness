"""Eval agent tools — score recording and URL verification.

Tools are registered via @tool and served as an in-process MCP server.
The eval agent calls these tools during its analysis of a shopping guide.

Score tools write into a per-run `scores` dict owned by the caller (see
create_eval_tools_server). Nothing is module-level on purpose: the executor
grades several traces concurrently, and a shared store would let one judge
run clear or overwrite another run's verdict.
"""

import re
from typing import Any

import httpx
from claude_agent_sdk import tool, create_sdk_mcp_server


def _parse_score(raw) -> float | None:
    """Parse score from various formats: 67, "67", "67%", "0.67"."""
    if raw is None:
        return None
    s = str(raw).strip().rstrip("%")
    try:
        val = float(s)
        # If value looks like a ratio (0.0-1.0), convert to percentage
        if 0 < val < 1:
            val = val * 100
        return val
    except (TypeError, ValueError):
        return None


def build_eval_tools(scores: dict) -> list:
    """Build the score-recording tools bound to `scores` (one dict per judge run)."""

    @tool(
        "score_grader",
        "Record your evaluation verdict for this grading dimension. "
        "Call with result='Pass' or result='Fail' for binary evaluation. "
        "IMPORTANT: Also provide a numeric 'score' (0-100) reflecting the degree. "
        "Example: result='Fail', score=65 means 'failed but close'; result='Fail', score=10 means 'badly failed'. "
        "You can call this multiple times — the LAST call is your final answer. "
        "Include detailed reasoning with specific evidence from the guide.",
        {
            "grader_name": str,
            "result": str,
            "score": str,
            "reasoning": str,
        },
    )
    async def score_grader(args: dict[str, Any]) -> dict[str, Any]:
        grader_name = args.get("grader_name", "").strip().lower()
        reasoning = args.get("reasoning", "")

        # Binary result + optional numeric score
        # Judge returns Pass/Fail verdict; score field provides granularity within verdict.
        # FAIL with score=67 (67% grounded) > FAIL with score=20 (20% grounded).
        result_raw = str(args.get("result", "")).strip()
        explicit_score = args.get("score")
        result = ""

        if result_raw.lower() in ("pass", "true", "yes"):
            result = "Pass"
            parsed = _parse_score(explicit_score)
            if parsed is not None:
                score = max(70, min(100, parsed))  # Pass scores: 70-100
            else:
                score = 100.0
        elif result_raw.lower() in ("fail", "false", "no"):
            result = "Fail"
            parsed = _parse_score(explicit_score)
            if parsed is not None and parsed > 0:
                score = max(0, min(69, parsed))  # Fail scores: 0-69
            elif parsed == 0:
                # LLM sent score=0 — check reasoning for a numeric ratio as fallback
                ratio_match = re.search(r'(?:grounding_ratio|ratio|grounded)[:\s]*(\d+)%', reasoning, re.IGNORECASE)
                if ratio_match:
                    score = max(0, min(69, float(ratio_match.group(1))))
                else:
                    score = 30.0  # Default Fail score
            else:
                score = 30.0
        else:
            # Fallback: accept numeric score for backward compat
            raw_score = args.get("score", args.get("result", 0))
            parsed = _parse_score(raw_score)
            score = max(0, min(100, parsed)) if parsed is not None else 0.0
            # Infer result from numeric score
            result = "Pass" if score >= 70 else "Fail"

        # Track revision history — last call wins
        existing = scores.get(grader_name)
        revision_num = 1
        revisions: list[dict] = []
        if existing and isinstance(existing, dict):
            revision_num = existing.get("revision_num", 1) + 1
            revisions = list(existing.get("revisions", []))
            revisions.append({
                "score": existing.get("score", 0),
                "reasoning": existing.get("reasoning", ""),
                "result": existing.get("result", ""),
            })

        scores[grader_name] = {
            "score": score,
            "result": result,
            "reasoning": reasoning,
            "revision_num": revision_num,
            "revisions": revisions,
        }

        response: dict[str, Any] = {
            "status": "recorded",
            "grader_name": grader_name,
            "result": result,
            "score": score,
        }
        if revision_num > 1:
            response["revised"] = True
            response["revision_num"] = revision_num
            response["previous_result"] = revisions[-1].get("result", "")
        return response

    # Legacy tool — kept for backward compatibility with existing L2 judge
    @tool(
        "score_dimension",
        "Record your PASS/FAIL verdict for a quality dimension. "
        "You MUST call this exactly once for each of the 5 dimensions: "
        "relevance, actionability, evidence_quality, completeness, objectivity. "
        "Include brief reasoning for your verdict.",
        {
            "dimension": str,
            "verdict": str,
            "reasoning": str,
        },
    )
    async def score_dimension(args: dict[str, Any]) -> dict[str, Any]:
        dimension = args.get("dimension", "").strip().lower()
        verdict = args.get("verdict", "UNKNOWN").strip().upper()
        reasoning = args.get("reasoning", "")

        valid_dimensions = {"relevance", "actionability", "evidence_quality", "completeness", "objectivity"}
        if dimension not in valid_dimensions:
            return {"error": f"Invalid dimension '{dimension}'. Must be one of: {', '.join(sorted(valid_dimensions))}"}

        if verdict not in ("PASS", "FAIL", "UNKNOWN"):
            verdict = "UNKNOWN"

        scores[dimension] = {"verdict": verdict, "reasoning": reasoning}
        return {"status": "recorded", "dimension": dimension, "verdict": verdict}

    return [score_grader, score_dimension, verify_url]


@tool(
    "verify_url",
    "Check if a source URL is accessible. Returns HTTP status code. "
    "Use this to spot-check 2-3 key sources when evaluating evidence_quality. "
    "Do NOT check every URL — just the most important ones.",
    {
        "url": str,
    },
)
async def verify_url(args: dict[str, Any]) -> dict[str, Any]:
    url = args.get("url", "")
    if not url:
        return {"error": "url is required"}

    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(8.0, connect=5.0),
            follow_redirects=True,
        ) as client:
            resp = await client.head(url)
            return {
                "url": url,
                "status": resp.status_code,
                "accessible": resp.status_code < 400,
            }
    except Exception as e:
        return {
            "url": url,
            "status": 0,
            "accessible": False,
            "error": str(e)[:100],
        }


def create_eval_tools_server(scores: dict):
    """Create an in-process MCP server whose score tools record into `scores`.

    Pass a fresh dict for every judge run and read the verdicts back from it
    once the run finishes.
    """
    return create_sdk_mcp_server(
        name="eval-tools",
        version="1.0.0",
        tools=build_eval_tools(scores),
    )
