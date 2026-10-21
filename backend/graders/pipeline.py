"""Full grading pipeline: Gate → Code Graders → LLM Graders → Funnel → Composite → Pass.

Two-phase design:
  Phase 1 (collect): Saves raw trace with status="collected"
  Phase 2 (grade): Runs this pipeline, updates trace with scores

The pass decision follows config.PASS_RULE (see graders/scoring.py).
This pipeline can be re-run on existing traces (re-grade).
"""

import time

from runner.collector import CollectedResult
from graders.types import GraderResult, GRADER_DEFS, NEGATIVE_TYPES, TRAP_TYPES
from graders.scoring import build_checks, compute_pass_fail_gate, decide_pass
from graders.composite import compute_composite_score, get_effective_weights
from graders.failure_funnel import detect_failure_stage
from graders.search_quality import grade_search_quality
from graders.code_graders import (
    grade_rubric_coverage,
    grade_product_matching,
    grade_source_authority,
    grade_output_format,
    grade_efficiency,
    grade_tool_calls,
    grade_transcript,
    grade_state_check,
)
from graders.retrieval_grader import grade_retrieval_quality
from graders.l0_structure import grade_structure
from graders.l0_constraints import grade_constraints
from graders.l1_metrics import compute_l1_score


async def grade_trace(result: CollectedResult, case: dict) -> dict:
    """Run full grading pipeline on a collected result.

    Args:
        result: CollectedResult from SSE collection
        case: {"key", "query", "type", "constraints", "golden_data"}

    Returns dict with all grading details for DB storage.
    """
    from config import PASS_RULE

    t0 = time.time()
    grading_log: list[dict] = []  # Per-step execution log
    case_type = case.get("type", "clear_en")
    constraints = case.get("constraints", {})
    golden_data = case.get("golden_data", {})
    has_golden = bool(golden_data and (golden_data.get("scene_list") or golden_data.get("trap_rubric")))

    # ── Legacy L0/L1/L2 scores (backward compatibility) ──
    l0 = grade_structure(result, case_type)
    l0c = grade_constraints(result, constraints)
    l1_score, l1_breakdown = compute_l1_score(result, case_type)

    # L2 legacy (generic dimensions; off unless LEGACY_L2_ENABLED)
    l2 = None
    try:
        from config import JUDGE_ENABLED, LEGACY_L2_ENABLED
        if JUDGE_ENABLED and LEGACY_L2_ENABLED:
            from graders.l2_judge import grade_l2 as _grade_l2
            l2 = await _grade_l2(result, case, {"structure": l0}, l0c, l1_breakdown)
    except Exception:
        pass

    # ── Pass/Fail Gate ──
    gate_results, gate_passed = compute_pass_fail_gate(result, case_type)

    # ── Negative cases: special scoring ──
    if case_type in NEGATIVE_TYPES:
        from graders.scoring import compute_final_score
        final, composite_pass = compute_final_score(l0, l1_score, l2, case_type)
        # Negative cases are checked by their own L0 rules: brief answer, few searches, no error
        checks = {f"l0.{name}": bool(ok) for name, ok in l0.items()}
        return {
            "l0": {"structure": l0, "constraints": l0c},
            "l1": {"score": l1_score, "breakdown": l1_breakdown},
            "l2": l2,
            "final_score": final,
            "final_pass": decide_pass(checks, composite_pass, PASS_RULE),
            "checks": checks,
            "composite_scores": {},
            "failure_funnel": {},
            "error_types": [],
            "grading_duration_s": round(time.time() - t0, 2),
        }

    # ── Code Graders ──
    has_llm = False  # Will be set True if LLM graders run
    try:
        from config import JUDGE_ENABLED
        has_llm = JUDGE_ENABLED
    except Exception:
        pass

    effective_weights = get_effective_weights(case_type, has_golden, has_llm)

    grader_results: list[GraderResult] = []

    def _run_code_grader(name: str, fn, *args):
        """Run a code grader and log its execution."""
        gt = time.time()
        try:
            r = fn(*args)
            r.weight = effective_weights[name]
            grader_results.append(r)
            grading_log.append({
                "step": name, "category": "code", "score": r.score,
                "weight": r.weight, "duration_s": round(time.time() - gt, 3),
                "error_types": r.error_types, "status": "ok",
            })
        except Exception as e:
            grading_log.append({
                "step": name, "category": "code", "score": None,
                "duration_s": round(time.time() - gt, 3),
                "status": "error", "error": str(e)[:200],
            })

    if effective_weights.get("rubric_coverage", 0) > 0:
        _run_code_grader("rubric_coverage", grade_rubric_coverage, result, golden_data)
    if effective_weights.get("product_matching", 0) > 0:
        _run_code_grader("product_matching", grade_product_matching, result, golden_data)
    if effective_weights.get("source_authority", 0) > 0:
        _run_code_grader("source_authority", grade_source_authority, result, golden_data)
    if effective_weights.get("output_format", 0) > 0:
        _run_code_grader("output_format", grade_output_format, result)
    if effective_weights.get("efficiency", 0) > 0:
        _run_code_grader("efficiency", grade_efficiency, result)
    if effective_weights.get("tool_calls", 0) > 0:
        _run_code_grader("tool_calls", grade_tool_calls, result, case.get("query", ""))
    if effective_weights.get("transcript", 0) > 0:
        _run_code_grader("transcript", grade_transcript, result)
    if effective_weights.get("state_check", 0) > 0:
        _run_code_grader("state_check", grade_state_check, result)
    if effective_weights.get("retrieval_quality", 0) > 0:
        _run_code_grader("retrieval_quality", grade_retrieval_quality, result, case.get("query", ""))
    if effective_weights.get("search_quality", 0) > 0:
        _run_code_grader("search_quality", grade_search_quality, result, case.get("query", ""))

    # ── LLM Graders (graceful degradation) ──
    judge_prompts: dict[str, str] = {}
    if has_llm:
        try:
            from graders.llm_graders import (
                grade_rubric_compliance,
                grade_trap_detection,
                grade_actionability,
                grade_groundedness,
            )

            async def _run_llm_grader_logged(name: str, fn, *args):
                gt = time.time()
                try:
                    r = await fn(*args)
                    # Detect silently-failed LLM graders (returned score=0 with skipped=True)
                    skipped = r.details.get("skipped", False)
                    if skipped:
                        # Don't add to grader_results — let composite rescale without this grader
                        grading_log.append({
                            "step": name, "category": "llm", "score": 0,
                            "weight": 0, "duration_s": round(time.time() - gt, 3),
                            "status": "skipped", "reason": r.details.get("reason", "unknown"),
                        })
                        return
                    r.weight = effective_weights[name]
                    grader_results.append(r)
                    log_entry = {
                        "step": name, "category": "llm", "score": r.score,
                        "weight": r.weight, "duration_s": round(time.time() - gt, 3),
                        "error_types": r.error_types, "status": "ok",
                        "num_turns": r.details.get("num_turns", 0),
                    }
                    if r.details.get("revision_count", 1) > 1:
                        log_entry["revisions"] = r.details.get("revision_count", 1)
                    if r.details.get("reasoning"):
                        log_entry["reasoning_preview"] = str(r.details["reasoning"])[:300]
                    grading_log.append(log_entry)
                    # Capture judge prompt for traceability
                    if r.details.get("system_prompt"):
                        judge_prompts[name] = str(r.details["system_prompt"])[:5000]
                except Exception as e:
                    grading_log.append({
                        "step": name, "category": "llm", "score": None,
                        "duration_s": round(time.time() - gt, 3),
                        "status": "error", "error": str(e)[:200],
                    })

            if effective_weights.get("rubric_compliance", 0) > 0:
                await _run_llm_grader_logged("rubric_compliance", grade_rubric_compliance, result, golden_data)

            if effective_weights.get("trap_detection", 0) > 0 and case_type in TRAP_TYPES:
                await _run_llm_grader_logged("trap_detection", grade_trap_detection, result, golden_data)

            if effective_weights.get("actionability", 0) > 0:
                await _run_llm_grader_logged("actionability", grade_actionability, result)

            if effective_weights.get("groundedness", 0) > 0:
                await _run_llm_grader_logged("groundedness", grade_groundedness, result)
        except Exception as e:
            # LLM graders failed — log for diagnostics instead of silent swallow
            import logging
            logging.getLogger(__name__).error(f"LLM graders setup/run failed: {e}", exc_info=True)
            grading_log.append({
                "step": "llm_graders_setup", "category": "llm",
                "status": "error", "error": str(e)[:300],
            })

    # ── Composite Score ──
    composite_score, composite_pass = compute_composite_score(grader_results, case_type)

    # Override: gate failure forces fail
    if not gate_passed:
        composite_pass = False

    # ── Pass decision: every check (pass_all) or the weighted score (composite) ──
    checks = build_checks(gate_results, grader_results)
    is_pass = decide_pass(checks, composite_pass, PASS_RULE)

    grading_log.append({
        "step": "gate", "category": "system",
        "passed": gate_passed, "results": gate_results,
    })
    grading_log.append({
        "step": "composite", "category": "system",
        "score": composite_score, "is_pass": composite_pass,
        "n_graders": len(grader_results),
    })
    grading_log.append({
        "step": "pass", "category": "system",
        "rule": PASS_RULE, "is_pass": is_pass,
        "failed_checks": sorted(name for name, ok in checks.items() if not ok),
    })

    # ── Failure Funnel ──
    grader_score_map = {g.name: g.score for g in grader_results}
    funnel = detect_failure_stage(result, golden_data, grader_score_map)

    # ── Aggregate error types ──
    all_error_types = []
    for g in grader_results:
        all_error_types.extend(g.error_types)

    # ── Build composite_scores dict ──
    composite_dict = {}
    for g in grader_results:
        composite_dict[g.name] = {
            "score": g.score,
            "weight": g.weight,
            "category": g.category,
            "details": g.details,
            "error_types": g.error_types,
        }

    # ── Also compute legacy final_score for backward compatibility ──
    # Use composite score as the new final_score
    final_score = composite_score

    total_duration = round(time.time() - t0, 2)
    grading_log.append({
        "step": "total", "category": "system",
        "duration_s": total_duration,
    })

    # Include judge prompt version for traceability
    try:
        from config import JUDGE_PROMPT_VERSION
        judge_pv = JUDGE_PROMPT_VERSION
    except ImportError:
        judge_pv = ""

    return {
        "l0": {"structure": l0, "constraints": l0c},
        "l1": {"score": l1_score, "breakdown": l1_breakdown},
        "l2": l2,
        "final_score": final_score,
        "final_pass": is_pass,
        "checks": checks,
        "composite_scores": composite_dict,
        "failure_funnel": {
            "stage": funnel.stage,
            "reason": funnel.reason,
            "stages": funnel.stages,
        },
        "error_types": all_error_types,
        "grading_duration_s": total_duration,
        "grading_log": grading_log,
        "gate": {"results": gate_results, "passed": gate_passed},
        "judge_prompt_version": judge_pv,
        "judge_prompts": judge_prompts,
    }
