"""The judge's score tools record into the dict they were built with, nothing global."""

import asyncio

from agent.tools import build_eval_tools


def _tools(scores: dict) -> dict:
    return {t.name: t for t in build_eval_tools(scores)}


def _call(tool, **args) -> dict:
    return asyncio.run(tool.handler(args))


def test_each_tool_set_writes_only_to_its_own_store():
    a, b = {}, {}
    _call(_tools(a)["score_grader"], grader_name="groundedness", result="Pass", score="90",
          reasoning="13 of 15 claims grounded, no critical fabrications")
    _call(_tools(b)["score_grader"], grader_name="groundedness", result="Fail", score="40",
          reasoning="price contradicted by product data")
    _call(_tools(b)["score_dimension"], dimension="relevance", verdict="pass", reasoning="on topic")

    assert a == {"groundedness": {"score": 90, "result": "Pass", "revision_num": 1, "revisions": [],
                                  "reasoning": "13 of 15 claims grounded, no critical fabrications"}}
    assert b["groundedness"]["result"] == "Fail" and b["groundedness"]["score"] == 40
    assert b["relevance"] == {"verdict": "PASS", "reasoning": "on topic"}


def test_verdict_bands_and_revision_history_are_unchanged():
    scores: dict = {}
    grader = _tools(scores)["score_grader"]

    # A Pass is clamped into 70-100, a Fail into 0-69; the last call wins.
    _call(grader, grader_name="Actionability", result="Pass", score="40", reasoning="prices and links present")
    assert scores["actionability"]["score"] == 70

    second = _call(grader, grader_name="actionability", result="Fail", score="85", reasoning="no buy links at all")
    assert scores["actionability"]["score"] == 69
    assert scores["actionability"]["revision_num"] == 2
    assert scores["actionability"]["revisions"] == [
        {"score": 70, "reasoning": "prices and links present", "result": "Pass"},
    ]
    assert second["revised"] is True and second["previous_result"] == "Pass"

    # score=0 on a Fail falls back to the ratio quoted in the reasoning.
    _call(grader, grader_name="groundedness", result="Fail", score="0", reasoning="grounding_ratio: 55%")
    assert scores["groundedness"]["score"] == 55
