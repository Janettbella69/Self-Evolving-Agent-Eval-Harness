"""Agent API routes — programmatic access for the shopping agent to submit and query eval results.

POST /api/agent/submit-result   Submit guide + products + sources for grading (no collect phase)
POST /api/agent/query-scores    Query scores for a case across experiments/trials
POST /api/agent/analyze-trace   Get AI-powered analysis of a trace's strengths/weaknesses
"""

from pydantic import BaseModel

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from runner.collector import CollectedResult
from graders.pipeline import grade_trace as grade_trace_pipeline
from storage import queries

router = APIRouter(prefix="/api/agent", tags=["agent"])


# ── Request/Response Models ──────────────────────

class SubmitResultRequest(BaseModel):
    dataset_id: int
    case_key: str
    guide_text: str
    products: list[dict] = []
    sources: list[dict] = []
    events: list[dict] = []
    duration_s: float = 0
    tag: str = ""
    trial_num: int = 1


class QueryScoresRequest(BaseModel):
    experiment_id: int | None = None
    case_key: str | None = None
    dataset_id: int | None = None
    limit: int = 20


class AnalyzeTraceRequest(BaseModel):
    trace_id: int


# ── Endpoints ────────────────────────────────────

@router.post("/submit-result")
async def submit_result(body: SubmitResultRequest):
    """Submit an agent result for immediate grading.

    Creates a one-off experiment + trace, runs the grading pipeline,
    and returns scores inline. Useful for agent self-evaluation.
    """
    # Validate dataset and case exist
    case = await queries.get_case_by_key(body.dataset_id, body.case_key)
    if not case:
        return JSONResponse(status_code=404, content={
            "detail": f"Case '{body.case_key}' not found in dataset {body.dataset_id}."
        })

    # Create experiment for this submission
    config = {"cases": [body.case_key], "trials": 1, "concurrency": 1, "agent_submitted": True}
    experiment_id = await queries.create_experiment(
        body.dataset_id, body.tag or "agent-submit", config,
    )

    # Create trace
    trace_id = await queries.create_trace(
        experiment_id, body.case_key, body.trial_num, case.query, case.type,
    )

    # Build CollectedResult from submitted data
    result = CollectedResult(
        guide_text=body.guide_text,
        products=body.products,
        sources=body.sources,
        events=body.events,
        duration_s=body.duration_s,
    )

    # Run grading pipeline
    case_dict = {
        "key": case.key,
        "query": case.query,
        "type": case.type,
        "constraints": case.constraints,
        "golden_data": case.golden_data,
    }
    grades = await grade_trace_pipeline(result, case_dict)

    # Save trace
    await queries.update_trace(
        trace_id,
        status="done",
        duration_s=body.duration_s,
        guide_text=body.guide_text,
        products=body.products,
        sources=body.sources,
        events=body.events,
        l0_scores=grades["l0"],
        l1_scores=grades["l1"],
        l2_scores=grades.get("l2"),
        final_score=grades["final_score"],
        final_pass=grades["final_pass"],
        checks=grades.get("checks", {}),
        composite_scores=grades["composite_scores"],
        failure_funnel=grades["failure_funnel"],
        error_types=grades["error_types"],
        grading_duration_s=grades["grading_duration_s"],
        grading_log=grades.get("grading_log", []),
    )

    # Update experiment
    summary = await queries.compute_experiment_summary(experiment_id)
    await queries.update_experiment_status(
        experiment_id, "complete", summary.model_dump(),
    )

    # Build response with scores
    return {
        "trace_id": trace_id,
        "experiment_id": experiment_id,
        "final_score": grades["final_score"],
        "final_pass": grades["final_pass"],
        "checks": grades.get("checks", {}),
        "composite_scores": grades["composite_scores"],
        "failure_funnel": grades["failure_funnel"],
        "error_types": grades["error_types"],
        "gate": grades.get("gate"),
        "grading_duration_s": grades["grading_duration_s"],
    }


@router.post("/query-scores")
async def query_scores(body: QueryScoresRequest):
    """Query scores for a case across experiments and trials.

    Returns per-trial scores with pass@k statistics.
    """
    if body.experiment_id:
        traces = await queries.get_experiment_traces(body.experiment_id)
        if body.case_key:
            traces = [t for t in traces if t.case_key == body.case_key]
    elif body.case_key and body.dataset_id:
        # Find all experiments for this dataset and filter by case
        experiments = await queries.list_experiments()
        traces = []
        for exp in experiments:
            if exp.dataset_id == body.dataset_id:
                exp_traces = await queries.get_experiment_traces(exp.id)
                traces.extend(t for t in exp_traces if t.case_key == body.case_key)
    else:
        return JSONResponse(status_code=422, content={
            "detail": "Provide experiment_id, or both case_key and dataset_id."
        })

    traces = traces[:body.limit]

    # Compute pass@k stats
    trial_results = [
        {
            "trace_id": t.id,
            "experiment_id": t.experiment_id,
            "trial_num": t.trial_num,
            "score": t.final_score,
            "pass": t.final_pass,
            "duration_s": t.duration_s,
            "composite_scores": t.composite_scores,
        }
        for t in traces
    ]

    n = len(trial_results)
    n_pass = sum(1 for r in trial_results if r["pass"])

    return {
        "case_key": body.case_key,
        "trials": trial_results,
        "total_trials": n,
        "pass_at_1": n_pass / n if n else 0,
        "pass_all": 1.0 if (n > 0 and n_pass == n) else 0.0,
        "avg_score": sum(r["score"] for r in trial_results) / n if n else 0,
    }


@router.post("/analyze-trace")
async def analyze_trace(body: AnalyzeTraceRequest):
    """Analyze a trace and return actionable diagnosis + suggestions.

    Uses composite scores and failure funnel to generate analysis
    without requiring an LLM call (deterministic analysis).
    """
    trace = await queries.get_trace(body.trace_id)
    if not trace:
        return JSONResponse(status_code=404, content={"detail": "Trace not found."})

    diagnosis_parts = []
    suggestions = []

    # Analyze gate
    gate = trace.composite_scores.get("_gate") or {}
    if isinstance(gate, dict) and not gate.get("passed", True):
        failures = [k for k, v in gate.get("results", {}).items() if not v]
        diagnosis_parts.append(f"Gate failed: {', '.join(failures)}")

    # Analyze failure funnel
    funnel = trace.failure_funnel
    if isinstance(funnel, dict) and funnel.get("stage"):
        stage = funnel["stage"]
        reason = funnel.get("reason", "")
        diagnosis_parts.append(f"First failure at '{stage}' stage: {reason}")
        funnel_suggestions = {
            "understand": "Agent may not have understood the query. Try rephrasing or adding constraints.",
            "search": "Agent searched but found no relevant results. Consider different search strategies or keywords.",
            "extract": "Agent found sources but couldn't extract product data. Check if scraping is working correctly.",
            "match_rubric": "Products were found but don't match expected products. Review product matching logic.",
            "generate": "Products found but guide generation was poor. Check output format and content quality.",
        }
        if stage in funnel_suggestions:
            suggestions.append(funnel_suggestions[stage])

    # Analyze weak graders
    if trace.composite_scores:
        weak = [
            (name, data)
            for name, data in trace.composite_scores.items()
            if isinstance(data, dict) and data.get("score", 100) < 50
        ]
        weak.sort(key=lambda x: x[1].get("score", 0))
        for name, data in weak[:3]:
            score = data.get("score", 0)
            diagnosis_parts.append(f"Low score on '{name}': {score:.0f}/100")
            # Specific suggestions per grader
            grader_suggestions = {
                "rubric_coverage": "Improve coverage of rubric keywords. Check if guide addresses all scene requirements.",
                "product_matching": "Ensure recommended products match golden product list. Check product name fuzzy matching.",
                "source_quality": "Use more authoritative review sources (T1/T2 tier). Diversify source domains.",
                "output_format": "Add comparison table, pros/cons section, and more inline citations.",
                "trap_detection": "Agent missed the trap. Ensure guide warns about hidden risks/issues.",
                "actionability": "Add clear purchase recommendations with prices and buy links.",
                "constraint_compliance": "Check that all user constraints (budget, brand, features) are satisfied.",
                "efficiency": "Too many searches or excessive duration. Optimize search strategy.",
            }
            if name in grader_suggestions:
                suggestions.append(grader_suggestions[name])

    # Analyze error types
    if trace.error_types:
        diagnosis_parts.append(f"Error types: {', '.join(trace.error_types[:5])}")

    # Summary
    if not diagnosis_parts:
        if trace.final_pass:
            diagnosis_parts.append("Trace passed all checks with no significant issues.")
        else:
            diagnosis_parts.append("Trace failed but no specific failure point identified from composite scores.")

    return {
        "trace_id": trace.id,
        "case_key": trace.case_key,
        "final_score": trace.final_score,
        "final_pass": trace.final_pass,
        "diagnosis": " | ".join(diagnosis_parts),
        "suggestions": suggestions,
        "weak_graders": [
            {"name": name, "score": data.get("score", 0)}
            for name, data in (trace.composite_scores or {}).items()
            if isinstance(data, dict) and data.get("score", 100) < 50
        ],
        "error_types": trace.error_types,
    }
