"""Eval agent — Claude Agent SDK powered evaluation.

Supports two execution modes (controlled by JUDGE_PRESET config):
1. Preset mode (default): Uses preset="claude_code" for full Claude Code capabilities
   — WebSearch, WebFetch, Bash, Read, Grep, Glob + eval tools
   — Deep verification: spot-checks prices, cross-references sources, analyzes codebase
2. Basic mode: Custom system prompt with only eval tools (score_grader + verify_url)
   — Lightweight, faster, cheaper

Both modes support:
- L2 Judge (legacy): Holistic 5-dimension PASS/FAIL evaluation
- Single Grader: Focused 0-100 numeric scoring for a specific dimension
"""

import asyncio
import logging
import os

from claude_agent_sdk import (
    ClaudeSDKClient,
    ClaudeAgentOptions,
    ResultMessage,
    ThinkingConfigAdaptive,
    HookMatcher,
)

logger = logging.getLogger(__name__)

from agent.tools import create_eval_tools_server


# ── Eval Agent Hooks (Harness Engineering) ────────────────────────────
#
# Hooks prevent the eval agent from making known mistakes:
# 1. Stop: if agent finishes without calling score_grader → force a verdict
# 2. PreToolUse(score_grader): validate verdict format before accepting
# 3. PostToolUseFailure: log and guide agent to retry

def _make_eval_stop_hook(scores: dict):
    """Stop hook: detect when agent finishes without scoring.

    If no score_grader call was made, inject a system message forcing
    the agent to make a decision. This is the #1 eval agent failure mode.
    Bound to this run's score store so concurrent runs don't see each other.
    """

    async def _on_eval_stop(input_data: dict, tool_use_id, context) -> dict:
        if scores:
            return {}
        logger.warning("Eval agent Stop hook: no score recorded — injecting reminder")
        return {
            "systemMessage": (
                "CRITICAL: You have NOT called score_grader yet. "
                "You MUST call score_grader before finishing. "
                "Based on your analysis so far, make your best judgment and call "
                "score_grader with result='Pass' or result='Fail' and your reasoning."
            ),
            "stopReason": "tool_use",  # Force agent to continue
        }

    return _on_eval_stop


async def _on_eval_pre_score(input_data: dict, tool_use_id, context) -> dict:
    """PreToolUse(score_grader): validate the verdict before accepting."""
    tool_input = input_data.get("tool_input", {})
    result = str(tool_input.get("result", "")).strip().lower()
    reasoning = str(tool_input.get("reasoning", "")).strip()

    if result not in ("pass", "fail", "true", "false", "yes", "no"):
        return {
            "systemMessage": (
                f"Invalid verdict '{result}'. Use result='Pass' or result='Fail'. "
                "Call score_grader again with a valid verdict."
            ),
        }

    if len(reasoning) < 10:
        return {
            "systemMessage": (
                "Reasoning is too brief. Provide at least 1-2 sentences explaining "
                "your verdict with specific evidence. Call score_grader again."
            ),
        }

    return {}


async def _on_eval_tool_failure(input_data: dict, tool_use_id, context) -> dict:
    """PostToolUseFailure: guide agent to retry or skip verification."""
    tool_name = input_data.get("tool_name", "")
    error = input_data.get("error", "")
    return {
        "systemMessage": (
            f"Tool '{tool_name}' failed: {str(error)[:100]}. "
            "This is not critical — proceed with your evaluation based on "
            "available information. Call score_grader with your verdict."
        ),
    }


def _get_eval_hooks(scores: dict) -> dict:
    """Build hooks config for one eval agent run."""
    return {
        "Stop": [
            HookMatcher(hooks=[_make_eval_stop_hook(scores)]),
        ],
        "PreToolUse": [
            HookMatcher(
                matcher="mcp__eval-tools__score_grader",
                hooks=[_on_eval_pre_score],
                timeout=10,
            ),
        ],
        "PostToolUseFailure": [
            HookMatcher(hooks=[_on_eval_tool_failure]),
        ],
    }
from config import (
    GRADING_MODEL,
    GRADING_FALLBACK_MODEL,
    JUDGE_PRESET,
    JUDGE_MAX_TURNS_PRESET,
    JUDGE_MAX_TURNS_BASIC,
    PRODUCT_CODE_PATH,
)
from graders.types import L2_DIMENSIONS


# ── Shared capabilities block (appended in preset mode) ──────────────

_JUDGE_CAPABILITIES_TEMPLATE = """
## You Are a Deep-Investigation Judge

You have FULL Claude Code capabilities — WebSearch, WebFetch, Bash, Read, Grep, Glob.
You are not a passive text reviewer. You are an autonomous investigator with tools.

### Your Information Advantage ("God's Eye View")
You see things the end-user never sees:
1. **Agent Operation Log** — every search query, URL fetched, error, timing breakdown
2. **Live Web Access** — verify claims against current reality
3. **Bash/Code** — write analysis scripts for quantitative checks
{source_code_ref}
## Autonomous Investigation Protocol

### Phase 1: Analyze Operation Log
The user message contains a structured "Agent Operation Log" section. Use it to:
- Understand the agent's research strategy (what it searched, in what order)
- Identify gaps (searches it should have done but didn't)
- Spot failures (errors, failed URLs, low product count)
- Assess efficiency (too many searches? too few? right balance?)

### Phase 2: Initial Score
Call `score_grader` with your first assessment based on the guide + operation log.

### Phase 3: Deep Verification (use your tools)
- **WebSearch** 2-3 product names to verify existence and price accuracy
- **verify_url** on 2-3 critical source URLs
- **WebFetch** the most important source to check content supports the guide's claims
- **Read** `prompts.py` to check if the agent followed its quality gate
- **Bash** for quantitative analysis (count products, measure section balance, etc.)

### Phase 4: Self-Correction Loop
After verification, check:
- Did verification CONTRADICT my initial score? → call `score_grader` again
- Is there a GAP between what the operation log shows (agent's effort) and the output?
  e.g., agent searched extensively but guide is shallow → output quality issue
  e.g., agent barely searched but guide is detailed → hallucination risk
- Did I overlook error events that explain quality gaps?

If your revised score differs from initial by >10 points, explain WHAT CHANGED.
If initial holds up after verification, do NOT revise — one accurate score beats padding.

### Budget
- 5-8 tool calls per evaluation (searches + reads + fetches)
- Focus on claims that are central to the guide's value
- If something is easily verifiable and important, verify it
"""

_SOURCE_CODE_REF_TEMPLATE = """
### Source Code Reference
The product agent's codebase lives at `{path}` (configured via PRODUCT_CODE_PATH).
Reading it shows what the agent was INSTRUCTED to do — prompts, hooks, quality
gates — so you can judge: did the agent follow its own rules? Did it pass its
quality gate? Did it use the right research strategy?
"""


def judge_capabilities() -> str:
    """Capabilities block for preset mode; includes the product source-code
    section only when PRODUCT_CODE_PATH is configured."""
    ref = ""
    if PRODUCT_CODE_PATH:
        ref = _SOURCE_CODE_REF_TEMPLATE.format(path=PRODUCT_CODE_PATH)
    return _JUDGE_CAPABILITIES_TEMPLATE.format(source_code_ref=ref)


# ── Tool lists ────────────────────────────────────────────────────────

_BASIC_TOOLS_L2 = [
    "mcp__eval-tools__score_dimension",
    "mcp__eval-tools__verify_url",
]

_BASIC_TOOLS_GRADER = [
    "mcp__eval-tools__score_grader",
    "mcp__eval-tools__verify_url",
]

# Claude Code preset tools — the preset provides Read/Bash/Grep/Glob/WebSearch/WebFetch
# automatically; we just need to also allow our MCP eval tools.
_PRESET_EXTRA_TOOLS = [
    "WebSearch",
    "WebFetch",
    "Bash",
    "Read",
    "Grep",
    "Glob",
]


def _build_options(
    prompt_text: str,
    tools_server,
    scores: dict,
    is_l2: bool = False,
) -> ClaudeAgentOptions:
    """Build ClaudeAgentOptions for either preset or basic mode.

    In preset mode:
      - system_prompt = {"type": "preset", "preset": "claude_code", "append": prompt_text + capabilities}
      - allowed_tools = eval MCP tools + Claude Code built-in tools
      - max_turns = JUDGE_MAX_TURNS_PRESET (default 20)

    In basic mode:
      - system_prompt = prompt_text (plain string)
      - allowed_tools = only eval MCP tools
      - max_turns = JUDGE_MAX_TURNS_BASIC (default 6)
    """
    base_tools = _BASIC_TOOLS_L2 if is_l2 else _BASIC_TOOLS_GRADER

    if JUDGE_PRESET:
        system_prompt = {
            "type": "preset",
            "preset": "claude_code",
            "append": prompt_text + "\n" + judge_capabilities(),
        }
        allowed_tools = base_tools + _PRESET_EXTRA_TOOLS
        max_turns = JUDGE_MAX_TURNS_PRESET if is_l2 else min(JUDGE_MAX_TURNS_PRESET, 15)
    else:
        system_prompt = prompt_text
        allowed_tools = base_tools
        max_turns = 10 if is_l2 else JUDGE_MAX_TURNS_BASIC

    # Fallback model: if GRADING_MODEL fails, try a cheaper alternative (config.GRADING_FALLBACK_MODEL)
    fallback = GRADING_FALLBACK_MODEL

    return ClaudeAgentOptions(
        system_prompt=system_prompt,
        model=GRADING_MODEL,
        fallback_model=fallback,
        max_turns=max_turns,
        permission_mode="bypassPermissions",
        allowed_tools=allowed_tools,
        mcp_servers={"eval-tools": tools_server},
        thinking=ThinkingConfigAdaptive(type="adaptive"),
        hooks=_get_eval_hooks(scores),
        # Claude Code 2.1.304+ turns tool search on behind an ANTHROPIC_BASE_URL gateway,
        # which hides score_grader behind ToolSearch and costs the judge an extra turn
        # to find the one tool it must call. Load the eval tools upfront, as before.
        env={"ENABLE_TOOL_SEARCH": "false"},
    )


# ── L2 Judge (legacy PASS/FAIL mode) ─────────────────────────────────

EVAL_AGENT_PROMPT = """You are an evaluation agent for a shopping research system.

Your task: evaluate a buyer's guide across 5 quality dimensions and record PASS/FAIL verdicts.

## Dimensions

{dimensions_block}

## Rules

1. Read the guide holistically before scoring any dimension.
2. Use the `verify_url` tool to spot-check 2-3 key source URLs for evidence_quality.
   - Don't check every URL. Focus on the most important claims.
   - If URLs are broken, factor that into your evidence_quality verdict.
3. Call `score_dimension` for each dimension (5 calls total).
4. PASS = meets criteria adequately for a helpful shopping guide.
5. FAIL = clearly deficient on this dimension.
6. UNKNOWN = insufficient information to judge (e.g., empty guide).
7. Base your judgment on the guide content AND the automated evidence provided.
8. Be fair but rigorous. A guide doesn't need to be perfect to PASS.
9. After scoring all 5 dimensions, briefly reflect: do your verdicts feel consistent
   with each other and with the evidence? If not, revise the affected dimension(s).
10. Output a brief overall assessment (2-3 sentences).
"""


def _build_dimensions_block() -> str:
    lines = []
    for i, dim in enumerate(L2_DIMENSIONS, 1):
        lines.append(f"{i}. **{dim.name}**: {dim.criteria}")
    return "\n".join(lines)


def _build_eval_message(query, guide_text, evidence_summary, products_summary, sources_summary) -> str:
    return f"""Evaluate this shopping research guide.

## User Query
{query}

## Automated Evidence (from L0/L1 graders)
{evidence_summary}

## Products Found
{products_summary}

## Sources Referenced
{sources_summary}

## Guide Text (first 8000 chars)
{guide_text[:8000]}

---

Now evaluate the guide. Verify a few key URLs, then score all 5 dimensions using the score_dimension tool."""


def _format_products(products: list) -> str:
    if not products:
        return "No products found."
    lines = []
    for i, p in enumerate(products[:10], 1):
        name = p.get("name", "Unknown") if isinstance(p, dict) else str(p)
        price = p.get("price", "N/A") if isinstance(p, dict) else "N/A"
        brand = p.get("brand", "") if isinstance(p, dict) else ""
        lines.append(f"{i}. {name} ({brand}) — {price}")
    if len(products) > 10:
        lines.append(f"   ... and {len(products) - 10} more")
    return "\n".join(lines)


def _format_sources(sources: list) -> str:
    if not sources:
        return "No sources found."
    lines = []
    for i, s in enumerate(sources[:15], 1):
        if isinstance(s, dict):
            title = s.get("title", "Untitled")
            url = s.get("url", "")
            domain = s.get("domain", "")
            lines.append(f"{i}. [{title}]({url}) — {domain}")
        else:
            lines.append(f"{i}. {s}")
    if len(sources) > 15:
        lines.append(f"   ... and {len(sources) - 15} more")
    return "\n".join(lines)


async def run_eval_agent(
    query: str,
    guide_text: str,
    evidence_summary: str,
    products: list,
    sources: list,
) -> dict:
    """Run the L2 eval agent (legacy PASS/FAIL mode).

    Returns:
        {"dimensions": {"name": "PASS"|"FAIL"|"UNKNOWN"}, "l2_score": float | None}
    """
    scores: dict = {}  # per-run score store (see agent/tools.py)

    prompt_text = EVAL_AGENT_PROMPT.format(
        dimensions_block=_build_dimensions_block(),
    )

    tools_server = create_eval_tools_server(scores)
    options = _build_options(prompt_text, tools_server, scores, is_l2=True)

    message = _build_eval_message(
        query=query,
        guide_text=guide_text,
        evidence_summary=evidence_summary,
        products_summary=_format_products(products),
        sources_summary=_format_sources(sources),
    )

    env_backup = os.environ.pop("CLAUDECODE", None)
    try:
        async with ClaudeSDKClient(options=options) as client:
            await client.query(message)
            async for msg in client.receive_response():
                if isinstance(msg, ResultMessage):
                    break
    except Exception as e:
        logger.error(f"L2 eval agent error: {e}")
    finally:
        if env_backup is not None:
            os.environ["CLAUDECODE"] = env_backup

    # Collect scores from tool calls
    raw_scores = scores

    dimensions = {}
    for dim in L2_DIMENSIONS:
        score_data = raw_scores.get(dim.name, {})
        if isinstance(score_data, dict):
            dimensions[dim.name] = score_data.get("verdict", "UNKNOWN")
        else:
            dimensions[dim.name] = "UNKNOWN"

    # Score: passes / (total - unknowns) * 100
    valid = [v for v in dimensions.values() if v != "UNKNOWN"]
    passes = sum(1 for v in valid if v == "PASS")
    l2_score = round(passes / len(valid) * 100, 1) if valid else None

    return {"dimensions": dimensions, "l2_score": l2_score}


# ── Single Grader (new 0-100 mode) ───────────────────────────────────

_GRADER_TIMEOUT_S = 120  # Max seconds per LLM grader call


async def run_eval_grader(
    system_prompt: str,
    user_message: str,
    grader_name: str,
) -> dict | None:
    """Run a single LLM grader that returns a binary Pass/Fail verdict.

    The agent calls score_grader(grader_name, result="Pass"/"Fail", reasoning).

    In preset mode, the grader's system_prompt becomes the "append" portion of the
    Claude Code preset — giving the grader full capabilities to verify claims.

    Returns:
        {"score": float, "result": str, "reasoning": str, ...} or None on failure.
    """
    scores: dict = {}  # per-run score store (see agent/tools.py)

    # Prevent nested Claude Code session blocking Agent SDK
    env_backup = os.environ.pop("CLAUDECODE", None)

    tools_server = create_eval_tools_server(scores)
    options = _build_options(system_prompt, tools_server, scores, is_l2=False)

    num_turns = 0
    try:
        async def _run():
            nonlocal num_turns
            async with ClaudeSDKClient(options=options) as client:
                await client.query(user_message)
                async for msg in client.receive_response():
                    if isinstance(msg, ResultMessage):
                        num_turns = getattr(msg, "num_turns", 0) or 0
                        break

        await asyncio.wait_for(_run(), timeout=_GRADER_TIMEOUT_S)
    except asyncio.TimeoutError:
        logger.error(f"LLM grader '{grader_name}' timed out after {_GRADER_TIMEOUT_S}s")
        return None
    except Exception as e:
        logger.error(f"LLM grader '{grader_name}' Agent SDK error: {e}")
        return None
    finally:
        if env_backup is not None:
            os.environ["CLAUDECODE"] = env_backup

    # Case-insensitive lookup — LLM may send "Groundedness" instead of "groundedness"
    raw_scores = scores
    grader_data = raw_scores.get(grader_name)
    if not grader_data:
        # Fallback: case-insensitive search
        for key, val in raw_scores.items():
            if key.lower() == grader_name.lower():
                grader_data = val
                break

    if not grader_data or not isinstance(grader_data, dict):
        logger.warning(
            f"LLM grader '{grader_name}' completed (turns={num_turns}) but no score recorded. "
            f"Available keys: {list(raw_scores.keys())}"
        )
        return None

    return {
        "score": grader_data.get("score", 0),
        "result": grader_data.get("result", ""),
        "reasoning": grader_data.get("reasoning", ""),
        "details": {},
        "revision_num": grader_data.get("revision_num", 1),
        "revisions": grader_data.get("revisions", []),
        "num_turns": num_turns,
    }
