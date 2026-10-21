"""Judge options keep the eval tools loaded upfront, whatever the gateway default."""

from agent.eval_agent import _build_options
from agent.tools import create_eval_tools_server


def test_tool_search_is_off_so_score_grader_is_never_deferred():
    scores: dict = {}
    options = _build_options("judge prompt", create_eval_tools_server(scores), scores)

    # Behind an ANTHROPIC_BASE_URL gateway, Claude Code 2.1.304+ defers MCP tools
    # behind ToolSearch unless this is set.
    assert options.env.get("ENABLE_TOOL_SEARCH") == "false"
    assert "mcp__eval-tools__score_grader" in options.allowed_tools
