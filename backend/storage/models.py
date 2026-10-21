"""Pydantic models for eval platform data."""

from pydantic import BaseModel


class CaseIn(BaseModel):
    key: str
    query: str
    type: str = "clear_en"
    constraints: dict = {}
    golden_data: dict = {}
    reference_output: dict | None = None


class DatasetIn(BaseModel):
    name: str
    description: str = ""
    suite_type: str = "capability"  # capability | regression
    cases: list[CaseIn] = []


class Dataset(BaseModel):
    id: int
    name: str
    description: str = ""
    suite_type: str = "capability"
    version: int = 1
    case_count: int = 0
    created_at: float = 0


class Case(BaseModel):
    id: int
    dataset_id: int
    key: str
    query: str
    type: str
    constraints: dict = {}
    golden_data: dict = {}
    reference_output: dict | None = None
    last_validated_at: float = 0  # Unix timestamp of last human validation


class ExperimentIn(BaseModel):
    dataset_id: int
    tag: str = ""
    cases: list[str] | None = None  # subset of case keys; None = all
    trials: int = 1
    concurrency: int = 1
    judge_enabled: bool = False
    notes: str = ""                 # free-text notes about this experiment
    mode: str = "benchmark"         # "benchmark" (with hints, 5min) or "product" (no hints, 10min)
    model: str = ""                 # orchestrator model name (for A/B testing)
    grading_model: str = ""         # LLM judge model name
    system_prompt: str = ""         # optional system prompt override (for prompt A/B testing)
    auto_run: bool = False          # if True, start running immediately after creation
    ablation_flags: dict[str, bool] = {}  # disable components for ablation experiments


class Experiment(BaseModel):
    id: int
    dataset_id: int
    dataset_version: int = 0
    tag: str = ""
    status: str = "pending"
    config: dict = {}
    summary: dict = {}
    created_at: float = 0
    finished_at: float | None = None


class Trace(BaseModel):
    id: int
    experiment_id: int
    case_key: str
    trial_num: int = 1
    query: str = ""
    case_type: str = ""
    status: str = "pending"
    duration_s: float = 0
    guide_text: str = ""
    products: list = []
    sources: list = []
    events: list = []
    hook_metrics: dict = {}
    error_events: list = []
    clarification: dict | None = None
    l0_scores: dict = {}
    l1_scores: dict = {}
    l2_scores: dict | None = None
    final_score: float = 0
    final_pass: bool = False
    checks: dict = {}               # binary checks behind final_pass, {name: passed}
    composite_scores: dict = {}
    failure_funnel: dict = {}
    error_types: list[str] = []
    grading_duration_s: float = 0
    human_scores: dict = {}
    grading_log: list = []
    prompt_version: str = ""
    model: str = ""
    judge_prompt_version: str = ""
    human_pass: bool | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    turn_count: int = 0
    system_prompt: str = ""
    tool_names: list[str] = []
    judge_prompts: dict = {}
    open_codes: list[str] = []      # qualitative labels (e.g. "formula-error", "multi-sheet-misunderstanding")
    review_status: str = "pending"  # pending | reviewed | flagged
    reviewed_at: float | None = None
    review_notes: str = ""
    created_at: float = 0


class ExperimentSummary(BaseModel):
    total_cases: int = 0
    completed: int = 0
    passed: int = 0
    failed: int = 0
    errored: int = 0
    avg_score: float = 0
    median_score: float = 0
    avg_duration: float = 0
    avg_turns: float = 0           # average turn_count
    avg_tokens: float = 0          # average total tokens (input + output)
    avg_toolcalls: float = 0       # average tool call count
    pass_rate: float = 0           # pass@1
    pass_all_rate: float = 0       # pass^k (all trials pass)
    consistency_rate: float = 0    # consistency@k (score std dev)
    failure_funnel_dist: dict = {}  # {understand: 3, search: 5, ...}
    grader_averages: dict = {}      # {rubric_coverage: 72.3, ...}
    check_pass_rates: dict = {}     # {"groundedness": {"passed": 7, "total": 10, "rate": 0.7}, ...}
    pass_rule: str = ""             # PASS_RULE when summarized: pass_all | composite
    pass_at_k: dict = {}            # {"pass@1": 0.75, "pass@3": 0.92, ...}
    pass_pow_k: dict = {}           # {"pass^1": 0.75, "pass^3": 0.42, ...}
    suite_type: str = "capability"  # capability | regression (from dataset)
    primary_metric: str = ""        # e.g. "pass@1: 0.75" or "pass^3: 1.0"
