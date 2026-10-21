"""Judge options: only the tools the judge needs, loaded upfront, with a pinned effort."""

import config
from agent import eval_agent
from agent.eval_agent import _build_options
from agent.tools import create_eval_tools_server


def _options(is_l2: bool = False):
    scores: dict = {}
    return _build_options("judge prompt", create_eval_tools_server(scores), scores, is_l2=is_l2)


def test_tool_search_is_off_so_score_grader_is_never_deferred():
    options = _options()

    # Behind an ANTHROPIC_BASE_URL gateway, Claude Code 2.1.304+ defers MCP tools
    # behind ToolSearch unless this is set.
    assert options.env.get("ENABLE_TOOL_SEARCH") == "false"
    assert "mcp__eval-tools__score_grader" in options.allowed_tools


def test_basic_mode_loads_no_built_in_tools():
    for is_l2 in (False, True):
        options = _options(is_l2)
        assert options.tools == []
        assert all(name.startswith("mcp__eval-tools__") for name in options.allowed_tools)


def test_preset_mode_loads_only_its_six_built_in_tools(monkeypatch):
    monkeypatch.setattr(eval_agent, "JUDGE_PRESET", True)
    options = _options()
    assert options.tools == ["WebSearch", "WebFetch", "Bash", "Read", "Grep", "Glob"]
    assert set(options.tools) <= set(options.allowed_tools)


def test_effort_and_models_come_from_config():
    # Compared with config rather than literals: backend/.env may override the defaults.
    options = _options()
    assert options.effort == config.JUDGE_EFFORT
    assert options.model == config.GRADING_MODEL
    assert options.fallback_model == config.GRADING_FALLBACK_MODEL
