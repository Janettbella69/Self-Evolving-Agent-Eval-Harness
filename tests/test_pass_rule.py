"""final_pass under PASS_RULE: every binary check (pass_all) or the weighted score (composite)."""

import asyncio

import config
from graders import l2_judge, llm_graders
from graders.pipeline import grade_trace
from graders.scoring import build_checks, decide_pass
from graders.types import GraderResult
from runner.collector import CollectedResult
from storage import queries
from storage.database import close_db, init_db


def _judge(name: str, verdict: str, score: float) -> GraderResult:
    return GraderResult(name=name, score=score, weight=0.2, category="llm", details={"verdict": verdict})


def _good_result() -> CollectedResult:
    return CollectedResult(
        guide_text="A long buying guide. " * 60,
        products=[{"name": "Sony WF-C700N", "price": "$78"}],
        sources=[{"url": "https://www.rtings.com/headphones", "title": "RTINGS"}],
    )


def _with_judges(monkeypatch, groundedness: str):
    """Turn the judge on with stub LLM graders; the legacy generic-dimension judge must not run."""
    monkeypatch.setattr(config, "JUDGE_ENABLED", True)

    async def actionability(result):
        return _judge("actionability", "Pass", 95)

    async def grounded(result):
        return _judge("groundedness", groundedness, 65 if groundedness == "Fail" else 95)

    async def legacy_l2(*args, **kwargs):
        raise AssertionError("legacy L2 ran although LEGACY_L2_ENABLED is off")

    monkeypatch.setattr(llm_graders, "grade_actionability", actionability)
    monkeypatch.setattr(llm_graders, "grade_groundedness", grounded)
    monkeypatch.setattr(l2_judge, "grade_l2", legacy_l2)


def _case(case_type: str = "clear_en") -> dict:
    return {"key": "k", "query": "best wireless earbuds under $100", "type": case_type,
            "constraints": {}, "golden_data": {}}


def test_checks_are_gate_results_plus_judge_verdicts():
    graders = [
        _judge("groundedness", "Fail", 65),
        _judge("actionability", "Pass", 90),
        GraderResult(name="output_format", score=40, weight=0.07, category="code"),
    ]
    checks = build_checks({"has_guide": True, "no_error": True}, graders)
    assert checks == {"gate.has_guide": True, "gate.no_error": True,
                      "groundedness": False, "actionability": True}
    assert decide_pass(checks, composite_pass=True, rule="pass_all") is False
    assert decide_pass(checks, composite_pass=True, rule="composite") is True


def test_one_failed_judge_fails_the_trace_under_pass_all(monkeypatch):
    _with_judges(monkeypatch, groundedness="Fail")
    grades = asyncio.run(grade_trace(_good_result(), _case()))

    assert grades["checks"]["groundedness"] is False
    assert grades["checks"]["actionability"] is True
    assert all(v for k, v in grades["checks"].items() if k.startswith("gate."))
    assert grades["final_pass"] is False
    assert grades["l2"] is None
    pass_step = next(s for s in grades["grading_log"] if s["step"] == "pass")
    assert pass_step["rule"] == "pass_all" and pass_step["failed_checks"] == ["groundedness"]


def test_composite_rule_keeps_the_weighted_score_decision(monkeypatch):
    _with_judges(monkeypatch, groundedness="Fail")
    monkeypatch.setattr(config, "PASS_RULE", "composite")
    grades = asyncio.run(grade_trace(_good_result(), _case()))

    composite_step = next(s for s in grades["grading_log"] if s["step"] == "composite")
    assert grades["final_pass"] is composite_step["is_pass"]
    assert grades["checks"]["groundedness"] is False  # still recorded for drill-down


def test_negative_cases_use_their_own_l0_checks(monkeypatch):
    monkeypatch.setattr(config, "JUDGE_ENABLED", False)
    # Off-topic question answered with a 3000-char guide: too long for a negative case.
    long_answer = CollectedResult(guide_text="x" * 3000)

    grades = asyncio.run(grade_trace(long_answer, _case("negative")))
    assert grades["checks"] == {"l0.kept_brief": False, "l0.few_searches": True, "l0.no_error": True}
    assert grades["final_pass"] is False

    # The weighted score alone (old rule) would have passed it.
    monkeypatch.setattr(config, "PASS_RULE", "composite")
    assert asyncio.run(grade_trace(long_answer, _case("negative")))["final_pass"] is True


def test_summary_reports_pass_rate_per_check():
    async def scenario():
        await init_db()
        try:
            dataset_id = await queries.create_dataset("checks")
            exp_id = await queries.create_experiment(dataset_id, "checks", {})
            rows = [
                {"gate.has_guide": True, "groundedness": True},
                {"gate.has_guide": True, "groundedness": False},
                {"gate.has_guide": False},  # judge skipped on this one
            ]
            for i, checks in enumerate(rows):
                tid = await queries.create_trace(exp_id, f"c{i}", 1, "q", "clear_en")
                await queries.update_trace(tid, status="done", checks=checks,
                                           final_pass=int(all(checks.values())))
            return await queries.compute_experiment_summary(exp_id)
        finally:
            await close_db()

    summary = asyncio.run(scenario())
    assert summary.pass_rule == "pass_all"
    assert summary.check_pass_rates == {
        "gate.has_guide": {"passed": 2, "total": 3, "rate": 0.667},
        "groundedness": {"passed": 1, "total": 2, "rate": 0.5},
    }
