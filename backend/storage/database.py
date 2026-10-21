"""SQLite database with WAL mode for concurrent read/write."""

import aiosqlite

from config import DB_PATH

_db: aiosqlite.Connection | None = None

# Use strftime for compatibility with SQLite < 3.38 (no unixepoch)
_TABLES = [
    """CREATE TABLE IF NOT EXISTS datasets (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL UNIQUE,
        description TEXT DEFAULT '',
        created_at REAL NOT NULL DEFAULT (strftime('%s', 'now'))
    )""",
    """CREATE TABLE IF NOT EXISTS cases (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        dataset_id INTEGER NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
        key TEXT NOT NULL,
        query TEXT NOT NULL,
        type TEXT NOT NULL DEFAULT 'clear_en',
        constraints TEXT DEFAULT '{}',
        golden_data TEXT DEFAULT '{}',
        UNIQUE(dataset_id, key)
    )""",
    """CREATE TABLE IF NOT EXISTS experiments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        dataset_id INTEGER NOT NULL REFERENCES datasets(id),
        tag TEXT DEFAULT '',
        status TEXT NOT NULL DEFAULT 'pending',
        config TEXT DEFAULT '{}',
        summary TEXT DEFAULT '{}',
        created_at REAL NOT NULL DEFAULT (strftime('%s', 'now')),
        finished_at REAL
    )""",
    """CREATE TABLE IF NOT EXISTS traces (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        experiment_id INTEGER NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
        case_key TEXT NOT NULL,
        trial_num INTEGER NOT NULL DEFAULT 1,
        query TEXT NOT NULL,
        case_type TEXT NOT NULL DEFAULT 'clear_en',
        status TEXT NOT NULL DEFAULT 'pending',
        duration_s REAL DEFAULT 0,
        guide_text TEXT DEFAULT '',
        products TEXT DEFAULT '[]',
        sources TEXT DEFAULT '[]',
        events TEXT DEFAULT '[]',
        hook_metrics TEXT DEFAULT '{}',
        error_events TEXT DEFAULT '[]',
        clarification TEXT,
        l0_scores TEXT DEFAULT '{}',
        l1_scores TEXT DEFAULT '{}',
        l2_scores TEXT,
        final_score REAL DEFAULT 0,
        final_pass INTEGER DEFAULT 0,
        composite_scores TEXT DEFAULT '{}',
        failure_funnel TEXT DEFAULT '{}',
        error_types TEXT DEFAULT '[]',
        grading_duration_s REAL DEFAULT 0,
        created_at REAL NOT NULL DEFAULT (strftime('%s', 'now'))
    )""",
    "CREATE INDEX IF NOT EXISTS idx_traces_experiment ON traces(experiment_id)",
    "CREATE INDEX IF NOT EXISTS idx_cases_dataset ON cases(dataset_id)",
]

# Migrations for existing databases (run after table creation)
_MIGRATIONS = [
    # Add golden_data to cases
    "ALTER TABLE cases ADD COLUMN golden_data TEXT DEFAULT '{}'",
    # Add new grading columns to traces
    "ALTER TABLE traces ADD COLUMN composite_scores TEXT DEFAULT '{}'",
    "ALTER TABLE traces ADD COLUMN failure_funnel TEXT DEFAULT '{}'",
    "ALTER TABLE traces ADD COLUMN error_types TEXT DEFAULT '[]'",
    "ALTER TABLE traces ADD COLUMN grading_duration_s REAL DEFAULT 0",
    # Phase 2: suite_type on datasets (capability / regression)
    "ALTER TABLE datasets ADD COLUMN suite_type TEXT DEFAULT 'capability'",
    # Phase 2: reference_output on cases (known-good output for grader calibration)
    "ALTER TABLE cases ADD COLUMN reference_output TEXT",
    # Phase 2: human_scores on traces (human annotation for LLM-judge calibration)
    "ALTER TABLE traces ADD COLUMN human_scores TEXT DEFAULT '{}'",
    # Phase 2: grading_log on traces (per-grader execution log)
    "ALTER TABLE traces ADD COLUMN grading_log TEXT DEFAULT '[]'",
    # Phase 3: prompt version tracking + human pass/fail
    "ALTER TABLE traces ADD COLUMN prompt_version TEXT DEFAULT ''",
    "ALTER TABLE traces ADD COLUMN model TEXT DEFAULT ''",
    "ALTER TABLE traces ADD COLUMN judge_prompt_version TEXT DEFAULT ''",
    "ALTER TABLE traces ADD COLUMN human_pass INTEGER",
    # Phase 3: Agent trace metadata (tokens, turns, prompts, tools)
    "ALTER TABLE traces ADD COLUMN input_tokens INTEGER DEFAULT 0",
    "ALTER TABLE traces ADD COLUMN output_tokens INTEGER DEFAULT 0",
    "ALTER TABLE traces ADD COLUMN turn_count INTEGER DEFAULT 0",
    "ALTER TABLE traces ADD COLUMN system_prompt TEXT DEFAULT ''",
    "ALTER TABLE traces ADD COLUMN tool_names TEXT DEFAULT '[]'",
    "ALTER TABLE traces ADD COLUMN judge_prompts TEXT DEFAULT '{}'",
    # Module 10: Open/Axial Coding — qualitative failure labels
    "ALTER TABLE traces ADD COLUMN open_codes TEXT DEFAULT '[]'",
    # Dataset staleness tracking
    "ALTER TABLE cases ADD COLUMN last_validated_at REAL DEFAULT 0",
    # Transcript review workflow
    "ALTER TABLE traces ADD COLUMN review_status TEXT DEFAULT 'pending'",
    "ALTER TABLE traces ADD COLUMN reviewed_at REAL",
    "ALTER TABLE traces ADD COLUMN review_notes TEXT DEFAULT ''",
    # Dataset versioning: auto-increment on case changes
    "ALTER TABLE datasets ADD COLUMN version INTEGER DEFAULT 1",
    # Experiment → dataset version snapshot (which version was tested)
    "ALTER TABLE experiments ADD COLUMN dataset_version INTEGER DEFAULT 0",
    # Binary checks behind final_pass ({"gate.has_guide": true, "groundedness": false, ...})
    "ALTER TABLE traces ADD COLUMN checks TEXT DEFAULT '{}'",
    # Grader validation history (TPR/TNR per judge prompt version)
    """CREATE TABLE IF NOT EXISTS grader_validations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        grader_name TEXT NOT NULL,
        tpr REAL,
        tnr REAL,
        tp INTEGER DEFAULT 0,
        fp INTEGER DEFAULT 0,
        tn INTEGER DEFAULT 0,
        fn INTEGER DEFAULT 0,
        n_samples INTEGER DEFAULT 0,
        threshold_met INTEGER DEFAULT 0,
        created_at REAL NOT NULL DEFAULT (strftime('%s', 'now'))
    )""",
    # Judge prompt management: evaluator prompts as data, not code
    """CREATE TABLE IF NOT EXISTS judge_prompts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        grader_name TEXT NOT NULL,
        version INTEGER NOT NULL DEFAULT 1,
        system_prompt TEXT NOT NULL,
        few_shots TEXT DEFAULT '[]',
        is_active INTEGER DEFAULT 1,
        notes TEXT DEFAULT '',
        created_at REAL NOT NULL DEFAULT (strftime('%s', 'now')),
        UNIQUE(grader_name, version)
    )""",
]


async def init_db() -> None:
    """Initialize database connection and create tables."""
    global _db
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = aiosqlite.connect(str(DB_PATH))
    # aiosqlite.Connection is a non-daemon Thread — daemonize it so one-shot
    # entrypoints (CLI, scripts) that skip close_db() can still exit.
    conn.daemon = True
    _db = await conn
    _db.row_factory = aiosqlite.Row
    await _db.execute("PRAGMA journal_mode=WAL")
    await _db.execute("PRAGMA foreign_keys=ON")
    for sql in _TABLES:
        await _db.execute(sql)
    # Run migrations (safe: ALTER TABLE ADD COLUMN is no-op if column exists)
    for sql in _MIGRATIONS:
        try:
            await _db.execute(sql)
        except Exception:
            pass  # Column already exists
    await _db.commit()


async def get_db() -> aiosqlite.Connection:
    """Get the shared database connection."""
    if _db is None:
        await init_db()
    return _db  # type: ignore


async def close_db() -> None:
    """Close the shared connection. One-shot entrypoints (CLI, scripts) call
    this on exit; the server keeps the connection for its whole lifetime."""
    global _db
    if _db is not None:
        await _db.close()
        _db = None
