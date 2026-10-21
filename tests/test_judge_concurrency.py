"""Judge runs that overlap must each get their own verdict back.

The executor grades several traces at once, so two `run_eval_grader` calls for
the same grader are routinely in flight together. The Agent SDK client is faked
here; what is under test is how eval_agent wires a run to its score store.
"""

import asyncio

from claude_agent_sdk import ResultMessage

from agent import eval_agent
from agent.tools import build_eval_tools


class _FakeJudge:
    """Stand-in for ClaudeSDKClient.

    Reads which verdict to give from the user message, yields to the event loop
    so the two runs interleave, then records the verdict through the score tools
    of the run it belongs to.
    """

    def __init__(self, options):
        self._scores = options.mcp_servers["eval-tools"]  # see _use_fake_judge
        self._message = ""

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def query(self, message: str):
        self._message = message

    async def receive_response(self):
        tools = {t.name: t for t in build_eval_tools(self._scores)}
        passing = "VERDICT=PASS" in self._message
        await asyncio.sleep(0.02 if passing else 0.01)  # the other run starts in this gap
        if "score all 5 dimensions" in self._message:  # legacy L2 judge
            for dim in eval_agent.L2_DIMENSIONS:
                await tools["score_dimension"].handler(
                    {"dimension": dim.name, "verdict": "PASS" if passing else "FAIL", "reasoning": "fake"})
        else:
            await tools["score_grader"].handler({
                "grader_name": "groundedness",
                "result": "Pass" if passing else "Fail",
                "score": "92" if passing else "35",
                "reasoning": "fake judge reasoning",
            })
        await asyncio.sleep(0.02 if passing else 0.01)  # …and finishes in this one
        yield ResultMessage(subtype="success", duration_ms=1, duration_api_ms=1,
                            is_error=False, num_turns=2, session_id="fake")


def _use_fake_judge(monkeypatch):
    # The "MCP server" handed to the fake client is just the run's score dict.
    monkeypatch.setattr(eval_agent, "create_eval_tools_server", lambda scores: scores)
    monkeypatch.setattr(eval_agent, "ClaudeSDKClient", lambda options: _FakeJudge(options))


def test_overlapping_grader_runs_keep_their_own_verdicts(monkeypatch):
    _use_fake_judge(monkeypatch)

    async def scenario():
        return await asyncio.gather(
            eval_agent.run_eval_grader("judge prompt", "VERDICT=PASS", "groundedness"),
            eval_agent.run_eval_grader("judge prompt", "VERDICT=FAIL", "groundedness"),
        )

    passed, failed = asyncio.run(scenario())
    assert (passed["result"], passed["score"]) == ("Pass", 92)
    assert (failed["result"], failed["score"]) == ("Fail", 35)


def test_overlapping_legacy_l2_runs_keep_their_own_verdicts(monkeypatch):
    _use_fake_judge(monkeypatch)

    async def scenario():
        return await asyncio.gather(
            eval_agent.run_eval_agent("VERDICT=PASS", "guide", "evidence", [], []),
            eval_agent.run_eval_agent("VERDICT=FAIL", "guide", "evidence", [], []),
        )

    passed, failed = asyncio.run(scenario())
    assert passed["l2_score"] == 100.0 and set(passed["dimensions"].values()) == {"PASS"}
    assert failed["l2_score"] == 0.0 and set(failed["dimensions"].values()) == {"FAIL"}


def test_grader_run_that_records_nothing_returns_none(monkeypatch):
    """A run with no score_grader call must not pick up a verdict left by another run."""
    _use_fake_judge(monkeypatch)

    class _SilentJudge(_FakeJudge):
        async def receive_response(self):
            await asyncio.sleep(0.03)
            yield ResultMessage(subtype="success", duration_ms=1, duration_api_ms=1,
                                is_error=False, num_turns=1, session_id="fake")

    def pick(options):
        return _SilentJudge(options) if "SILENT" in options.system_prompt else _FakeJudge(options)

    monkeypatch.setattr(eval_agent, "ClaudeSDKClient", lambda options: pick(options))

    async def scenario():
        return await asyncio.gather(
            eval_agent.run_eval_grader("SILENT judge", "VERDICT=PASS", "groundedness"),
            eval_agent.run_eval_grader("judge prompt", "VERDICT=PASS", "groundedness"),
        )

    silent, spoken = asyncio.run(scenario())
    assert silent is None
    assert spoken["result"] == "Pass"
