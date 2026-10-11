"""Experiment executor: streaming collect→grade pipeline with circuit breaker.

Architecture: collect workers → asyncio.Queue → grade workers (overlapping phases)
Grade workers start consuming immediately as traces are collected, reducing total time.
"""

import asyncio
import logging
import time

from runner.collector import collect_sse, CollectedResult
from config import USER_SIMULATOR_MODEL
from graders.pipeline import grade_trace as grade_trace_pipeline
from storage import queries
from runner.progress import get_manager

logger = logging.getLogger(__name__)

MAX_AUTO_CLARIFICATIONS = 2
CIRCUIT_BREAKER_THRESHOLD = 3  # consecutive errors → auto-pause
INCREMENTAL_SUMMARY_INTERVAL = 5  # broadcast summary every N graded traces


def _pick_clarification_answer(clarification: dict) -> str:
    """Pick a default answer: first option, or generic fallback."""
    options = clarification.get("options", [])
    if options:
        first = options[0]
        return first if isinstance(first, str) else str(first)
    return "give me general recommendations"


async def _simulate_user_answer(original_query: str, clarification: dict) -> str:
    """Use LLM to simulate a realistic user response to a clarification question.

    Anthropic eval blog: "conversational agents often require a second LLM to simulate
    the user" for multi-turn evaluation. This simulates a knowledgeable shopper who
    answers concisely based on their original intent.

    Falls back to _pick_clarification_answer on any error (fail-open).
    """
    question = clarification.get("question", "")
    options = clarification.get("options", [])
    question_type = clarification.get("question_type", "radio")

    if not question:
        return _pick_clarification_answer(clarification)

    try:
        import anthropic
        import os

        auth_token = os.getenv("ANTHROPIC_AUTH_TOKEN", "")
        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        base_url = os.getenv("ANTHROPIC_BASE_URL")

        client = anthropic.AsyncAnthropic(
            api_key=auth_token or api_key,
            timeout=10.0,
            **({"base_url": base_url} if base_url else {}),
        )

        options_text = "\n".join(f"- {o}" for o in options) if options else "(no predefined options)"

        resp = await client.messages.create(
            model=USER_SIMULATOR_MODEL,
            max_tokens=100,
            system=(
                "You are simulating a real shopper answering a clarification question. "
                "Based on the original shopping query, pick the most reasonable answer. "
                "If options are provided, pick one (or write a brief custom answer). "
                "Reply with ONLY the answer text, nothing else. Keep it under 20 words."
            ),
            messages=[{
                "role": "user",
                "content": f"Original query: {original_query}\n\nClarification: {question}\n\nOptions:\n{options_text}\n\nYour answer:",
            }],
        )

        # Extract text from response (handle minimax thinking blocks)
        text = ""
        thinking = ""
        for block in resp.content:
            if hasattr(block, "text") and block.type == "text":
                text += block.text
            elif block.type == "thinking" and hasattr(block, "thinking"):
                thinking += block.thinking

        answer = (text.strip() or thinking.strip())[:200]
        if answer:
            logger.info(f"User simulator: Q='{question[:40]}' A='{answer[:40]}'")
            return answer

        await client.close()
    except Exception as e:
        logger.warning(f"User simulator failed: {e} — falling back to first option")

    return _pick_clarification_answer(clarification)


async def collect_single_case(
    case: dict,
    trial_num: int = 1,
    auto_clarify: bool = True,
    model: str = "",
    system_prompt: str = "",
    ablation_flags: dict[str, bool] | None = None,
) -> CollectedResult:
    """Collect SSE events for a single case (Phase 1 only — no grading).

    Args:
        case: {"key", "query", "type", "constraints", "golden_data"}
        trial_num: Trial number (1-indexed).
        auto_clarify: Auto-answer clarification questions.
        model: Override orchestrator model for this run.
        system_prompt: Override system prompt for this run.
        ablation_flags: Disable agent components for ablation experiments.

    Returns:
        CollectedResult with raw SSE data.
    """
    query = case["query"]
    history: list[dict] = []

    result: CollectedResult | None = None
    clarification_count = 0

    while True:
        result = await collect_sse(query, history, model=model, system_prompt=system_prompt, ablation_flags=ablation_flags)

        # Auto-answer clarifications with LLM user simulator
        if result.clarification and auto_clarify and clarification_count < MAX_AUTO_CLARIFICATIONS:
            clarification_count += 1
            answer = await _simulate_user_answer(query, result.clarification)
            history.append({"role": "assistant", "content": f"[clarification] {result.clarification.get('question', '')}"})
            history.append({"role": "user", "content": answer})
            continue

        break

    return result


def _result_to_trace_data(result: CollectedResult, case: dict, trial_num: int) -> dict:
    """Convert CollectedResult to trace data dict for DB storage."""
    # Merge latency timestamps into hook_metrics for storage
    hm = dict(result.hook_metrics)
    hm["time_to_first_search"] = result.time_to_first_search
    hm["time_to_first_product"] = result.time_to_first_product
    hm["time_to_first_text"] = result.time_to_first_text

    return {
        "case_key": case["key"],
        "trial_num": trial_num,
        "query": case["query"],
        "case_type": case.get("type", "clear_en"),
        "status": "collected",
        "duration_s": result.duration_s,
        "guide_text": result.guide_text,
        "products": result.products,
        "sources": result.sources,
        "events": result.events,
        "hook_metrics": hm,
        "error_events": result.error_events,
        "clarification": result.clarification,
        "prompt_version": result.prompt_version,
        "model": result.model,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "turn_count": result.turn_count,
        "system_prompt": result.system_prompt,
        "tool_names": result.tool_names,
    }


def _grades_to_trace_update(grades: dict) -> dict:
    """Convert grading pipeline output to trace update kwargs."""
    return {
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
        "judge_prompt_version": grades.get("judge_prompt_version", ""),
    }


async def _grade_single_trace(trace: queries.Trace, case: dict) -> dict:
    """Run grading pipeline on a collected trace.

    Reconstructs a CollectedResult from trace data and runs the full pipeline.
    """
    result = CollectedResult(
        guide_text=trace.guide_text,
        products=trace.products,
        sources=trace.sources,
        events=trace.events,
        hook_metrics=trace.hook_metrics,
        error_events=trace.error_events,
        clarification=trace.clarification,
        duration_s=trace.duration_s,
        input_tokens=trace.input_tokens or 0,
        output_tokens=trace.output_tokens or 0,
        turn_count=trace.turn_count or 0,
        tool_names=trace.tool_names or [],
    )

    grades = await grade_trace_pipeline(result, case)
    return grades


# In-flight experiment cancellation tokens
_cancel_flags: dict[int, bool] = {}


def request_stop(experiment_id: int):
    _cancel_flags[experiment_id] = True


async def run_experiment(
    experiment_id: int,
    cases: list[dict],
    concurrency: int = 1,
    trials: int = 1,
    grade_concurrency: int = 2,
    model: str = "",
    system_prompt: str = "",
    ablation_flags: dict[str, bool] | None = None,
):
    """Run all cases: streaming collect→grade pipeline with overlapping phases.

    Architecture:
      collect_workers (Semaphore) → grade_queue → grade_workers (Semaphore)
      Grade workers start consuming immediately as traces are collected.

    Args:
        model: Override orchestrator model for all cases in this experiment.
        system_prompt: Override system prompt for all cases.
        ablation_flags: Disable agent components for ablation experiments.

    Updates DB traces and broadcasts progress via WebSocket.
    """
    _cancel_flags[experiment_id] = False
    manager = get_manager()
    collect_sem = asyncio.Semaphore(concurrency)

    await queries.update_experiment_status(experiment_id, "running")
    total = len(cases) * trials
    manager.broadcast(experiment_id, {"type": "experiment_start", "total": total})

    # ── Shared state ──
    case_map: dict[str, dict] = {c["key"]: c for c in cases}
    consecutive_errors = 0
    graded_count = 0
    graded_scores: list[float] = []
    graded_passes: list[bool] = []

    # ── Grade Queue: bridge between collect and grade phases ──
    grade_queue: asyncio.Queue[int | None] = asyncio.Queue()

    # ── Collect Worker ──
    async def collect_one(case: dict, trial_num: int):
        nonlocal consecutive_errors

        if _cancel_flags.get(experiment_id):
            return

        trace_id = await queries.create_trace(
            experiment_id, case["key"], trial_num, case["query"], case.get("type", "clear_en"),
        )
        manager.broadcast(experiment_id, {
            "type": "case_start",
            "case_key": case["key"],
            "trial_num": trial_num,
            "trace_id": trace_id,
        })

        try:
            async with collect_sem:
                result = await collect_single_case(case, trial_num, model=model, system_prompt=system_prompt, ablation_flags=ablation_flags)

            trace_data = _result_to_trace_data(result, case, trial_num)
            await queries.update_trace(trace_id, **trace_data)
            consecutive_errors = 0

            manager.broadcast(experiment_id, {
                "type": "case_collected",
                "case_key": case["key"],
                "trial_num": trial_num,
                "trace_id": trace_id,
                "duration_s": result.duration_s,
            })

            # Enqueue for grading immediately
            await grade_queue.put(trace_id)

        except Exception as e:
            await queries.update_trace(trace_id, **{
                "status": "error",
                "error_events": [{"type": "error", "message": str(e)[:500]}],
                "final_score": 0,
                "final_pass": 0,
            })
            consecutive_errors += 1

            manager.broadcast(experiment_id, {
                "type": "case_error",
                "case_key": case["key"],
                "trial_num": trial_num,
                "trace_id": trace_id,
                "error": str(e)[:200],
            })

            # Circuit breaker
            if consecutive_errors >= CIRCUIT_BREAKER_THRESHOLD:
                reason = f"Circuit breaker: {consecutive_errors} consecutive errors"
                await queries.update_experiment_status(experiment_id, "paused")
                manager.broadcast(experiment_id, {
                    "type": "circuit_break",
                    "reason": reason,
                    "consecutive_errors": consecutive_errors,
                })
                _cancel_flags[experiment_id] = True

    # ── Grade Worker ──
    async def grade_worker():
        nonlocal graded_count

        while True:
            trace_id = await grade_queue.get()
            if trace_id is None:  # Sentinel: no more work
                grade_queue.task_done()
                break

            trace = await queries.get_trace(trace_id)
            if not trace or trace.status == "error":
                grade_queue.task_done()
                continue

            case = case_map.get(trace.case_key, {
                "key": trace.case_key, "query": trace.query,
                "type": trace.case_type, "constraints": {}, "golden_data": {},
            })

            manager.broadcast(experiment_id, {
                "type": "case_grading",
                "case_key": trace.case_key,
                "trial_num": trace.trial_num,
                "trace_id": trace_id,
            })

            try:
                grades = await _grade_single_trace(trace, case)
                await queries.update_trace(trace_id, **_grades_to_trace_update(grades))

                score = grades.get("final_score", 0)
                passed = bool(grades.get("final_pass"))
                graded_count += 1
                graded_scores.append(score)
                graded_passes.append(passed)

                manager.broadcast(experiment_id, {
                    "type": "case_graded",
                    "case_key": trace.case_key,
                    "trial_num": trace.trial_num,
                    "trace_id": trace_id,
                    "score": score,
                    "passed": passed,
                })

                # Incremental summary
                if graded_count % INCREMENTAL_SUMMARY_INTERVAL == 0:
                    avg = sum(graded_scores) / len(graded_scores) if graded_scores else 0
                    pr = sum(graded_passes) / len(graded_passes) if graded_passes else 0
                    manager.broadcast(experiment_id, {
                        "type": "incremental_summary",
                        "graded_count": graded_count,
                        "avg_score": round(avg, 1),
                        "pass_rate": round(pr * 100, 1),
                    })

            except Exception as e:
                logger.error(f"Grading error for trace {trace_id}: {e}")
                await queries.update_trace(trace_id, status="done", final_score=0, final_pass=0)

            grade_queue.task_done()

    # ── Start grade workers FIRST (they block on empty queue) ──
    grade_workers = [asyncio.create_task(grade_worker()) for _ in range(grade_concurrency)]

    # ── Start all collect tasks ──
    collect_tasks = []
    for case in cases:
        for trial in range(1, trials + 1):
            collect_tasks.append(collect_one(case, trial))

    await asyncio.gather(*collect_tasks, return_exceptions=True)

    # ── Signal grade workers to stop ──
    for _ in range(grade_concurrency):
        await grade_queue.put(None)

    # ── Wait for all grading to complete ──
    await asyncio.gather(*grade_workers)

    # ── Phase 3: Final Aggregate ──
    summary = await queries.compute_experiment_summary(experiment_id)

    status = "complete"
    if _cancel_flags.get(experiment_id) and consecutive_errors >= CIRCUIT_BREAKER_THRESHOLD:
        status = "paused"

    await queries.update_experiment_status(
        experiment_id, status,
        summary=summary.model_dump(),
        finished_at=time.time(),
    )
    manager.broadcast(experiment_id, {
        "type": "experiment_done",
        "summary": summary.model_dump(),
    })
    _cancel_flags.pop(experiment_id, None)


async def regrade_experiment(experiment_id: int) -> int:
    """Re-grade all collected/graded traces in an experiment.

    Returns the number of traces re-graded.
    """
    experiment = await queries.get_experiment(experiment_id)
    if not experiment:
        return 0

    # Get all cases for golden_data
    all_cases = await queries.get_cases(experiment.dataset_id)
    case_map = {c.key: {"key": c.key, "query": c.query, "type": c.type,
                        "constraints": c.constraints, "golden_data": c.golden_data}
                for c in all_cases}

    traces = await queries.get_collected_traces(experiment_id)
    count = 0

    for trace in traces:
        case = case_map.get(trace.case_key, {"key": trace.case_key, "query": trace.query,
                                              "type": trace.case_type, "constraints": {}, "golden_data": {}})
        try:
            grades = await _grade_single_trace(trace, case)
            await queries.update_trace(trace.id, **_grades_to_trace_update(grades))
            count += 1
        except Exception:
            continue

    # Recompute summary
    summary = await queries.compute_experiment_summary(experiment_id)
    await queries.update_experiment_status(
        experiment_id, "complete",
        summary=summary.model_dump(),
        finished_at=time.time(),
    )

    return count
