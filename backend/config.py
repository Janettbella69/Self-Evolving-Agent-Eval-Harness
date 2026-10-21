"""Eval platform configuration — independent of product backend."""

import hashlib
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)

# Product backend URL (the system under test)
# Default to eval-dedicated backend (port 8001, started via docker-compose.eval.yml).
# Override with PRODUCT_API_URL=http://localhost:8000 to test against production.
PRODUCT_API_URL = os.getenv("PRODUCT_API_URL", "http://localhost:8001")

# Shared secret for /api/internal/research
EVAL_API_KEY = os.getenv("EVAL_API_KEY", "")

# Anthropic API key — used by Claude Agent SDK (reads from env automatically)
# Kept here for reference; Agent SDK picks up ANTHROPIC_API_KEY from os.environ.
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# Optional Anthropic-compatible gateway (e.g. OpenRouter). The Agent SDK and the
# anthropic client read ANTHROPIC_BASE_URL / ANTHROPIC_AUTH_TOKEN from the environment;
# non-Anthropic model ids such as "openai/gpt-6-luna" only resolve through such a gateway.
ANTHROPIC_BASE_URL = os.getenv("ANTHROPIC_BASE_URL", "")

# L2 grading model. Start with a capable model so the judge prompt is easy to get right,
# then try cheaper ones and keep whichever still agrees with human labels
# (validate_evaluator.py). Per the evals FAQ, the product's own model is usually fine:
# what matters is agreement with humans, not relative strength.
# Default since 2026-10-21: openai/gpt-6-luna, $0.10 in / $0.50 out per M tokens on
# OpenRouter vs $2.50 / $15 for the previous openai/gpt-5.4. In Hamel Husain's
# 2026-10-20 decision-model test, OpenAI Decisions (backed by GPT-6 Luna) had the best
# F1 at flagging failed agent traces. Not yet validated on our own human labels.
# Default assumes an OpenRouter-style gateway (see ANTHROPIC_BASE_URL above);
# with a plain Anthropic key, set GRADING_MODEL to a Claude model.
GRADING_MODEL = os.getenv("GRADING_MODEL", "openai/gpt-6-luna")

# Fallback judge model when GRADING_MODEL fails (Agent SDK fallback_model).
GRADING_FALLBACK_MODEL = os.getenv("GRADING_FALLBACK_MODEL", "minimax/minimax-m2.7-highspeed")

# Model that answers the product agent's clarification questions on the user's behalf
# (runner/executor.py). Only has to pick an option, so a cheap model is fine.
USER_SIMULATOR_MODEL = os.getenv("USER_SIMULATOR_MODEL", "minimax/minimax-m2.7")

# Scoring thresholds
# Calibrated 2026-03-28: human=Pass at 59.2, human=Fail at 51.6 → threshold 60
# Re-calibrate after 50+ annotations (Hamel audit action #3)
# Decides pass/fail only under PASS_RULE=composite.
PASS_THRESHOLD = int(os.getenv("PASS_THRESHOLD", "60"))

# How a trace passes. "pass_all" (default): every binary check passes, i.e. the gate
# checks plus each LLM judge's own Pass/Fail (the evals FAQ combines binary evals
# into a pass-all rate). "composite": the weighted 0-100 score must reach
# PASS_THRESHOLD, the rule before 2026-10-21. The weighted score is stored either way.
PASS_RULE = os.getenv("PASS_RULE", "pass_all").strip().lower()
if PASS_RULE not in ("pass_all", "composite"):
    raise ValueError(f"PASS_RULE must be 'pass_all' or 'composite', got {PASS_RULE!r}")

# L2 judge toggle
JUDGE_ENABLED = os.getenv("JUDGE_ENABLED", "").lower() in ("1", "true", "yes")

# Legacy L2 judge (graders/l2_judge.py): one more judge call per trace rating five
# generic dimensions (relevance, completeness, objectivity, ...). Off by default; its
# result is only stored in l2_scores and used for negative cases under PASS_RULE=composite.
LEGACY_L2_ENABLED = os.getenv("LEGACY_L2_ENABLED", "").lower() in ("1", "true", "yes")

# Judge preset mode — Claude Code preset gives full autonomy (WebSearch, Bash, Read, etc.)
# but adds ~60s startup overhead and can timeout with OpenRouter models.
# Default: False (basic mode) — faster, reliable with OpenRouter, sufficient for scoring.
JUDGE_PRESET = os.getenv("JUDGE_PRESET", "false").lower() in ("1", "true", "yes")

# Judge max turns — preset mode needs more turns for verification workflows
JUDGE_MAX_TURNS_PRESET = int(os.getenv("JUDGE_MAX_TURNS_PRESET", "20"))
JUDGE_MAX_TURNS_BASIC = int(os.getenv("JUDGE_MAX_TURNS_BASIC", "10"))

# LangFuse — production trace import
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "")
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_BASE_URL = os.getenv("LANGFUSE_BASE_URL", "https://us.cloud.langfuse.com")

# Product agent source code location (optional) — when set, the preset-mode
# judge is told to read the product's code to check rule/quality-gate compliance.
PRODUCT_CODE_PATH = os.getenv("PRODUCT_CODE_PATH", "")

# Database
DB_PATH = Path(os.getenv("DB_PATH", str(Path(__file__).parent / "eval.db")))

# Collector timeout (seconds)
COLLECT_TIMEOUT = int(os.getenv("COLLECT_TIMEOUT", "900"))

# Server
HOST = os.getenv("EVAL_HOST", "0.0.0.0")
PORT = int(os.getenv("EVAL_PORT", "8100"))


# Judge prompt version hash — computed from grader prompts at startup
def _compute_judge_prompt_hash() -> str:
    """SHA-256 of all judge-related prompt files, truncated to 12 hex chars."""
    h = hashlib.sha256()
    grader_dir = Path(__file__).parent
    for filename in sorted(["graders/llm_graders.py", "agent/eval_agent.py"]):
        filepath = grader_dir / filename
        try:
            h.update(filepath.read_bytes())
        except FileNotFoundError:
            pass
    return h.hexdigest()[:12]

JUDGE_PROMPT_VERSION = _compute_judge_prompt_hash()
