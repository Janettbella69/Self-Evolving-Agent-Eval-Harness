"""Calibration splits keep each case in one split; TPR/TNR treat a failure as the positive class."""

import asyncio
import importlib.util
import json
import sqlite3
from pathlib import Path

from graders.validation import (
    cases_in_multiple_splits,
    confusion_from_pairs,
    group_stratified_split,
    human_grader_passed,
    judge_passed,
    rogan_gladen_correction,
)
from storage import queries
from storage.database import close_db, init_db


def _load_script(name: str):
    path = Path(__file__).resolve().parent.parent / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _labeled(n_cases: int, trials: int = 2) -> list[dict]:
    """Cases cycle through all-Pass, all-Fail and mixed labels across trials."""
    items, next_id = [], 1
    for i in range(n_cases):
        for trial in range(trials):
            passed = i % 3 == 0 or (i % 3 == 2 and trial == 0)
            items.append({"id": next_id, "case_key": f"case{i}", "human_pass": passed})
            next_id += 1
    return items


def _seed(rows: list[dict]) -> list[int]:
    """Insert traces into the test database; each row's keys go to update_trace."""
    async def scenario():
        await init_db()
        try:
            dataset_id = await queries.create_dataset("calibration")
            exp_id = await queries.create_experiment(dataset_id, "calibration", {})
            ids = []
            for row in rows:
                row = dict(row)
                tid = await queries.create_trace(exp_id, row.pop("case_key"), row.pop("trial_num", 1), "q", "clear_en")
                await queries.update_trace(tid, status="done", **row)
                ids.append(tid)
            return ids
        finally:
            await close_db()
    return asyncio.run(scenario())


def test_split_keeps_every_case_in_one_split():
    items = _labeled(60)
    train, dev, test = group_stratified_split(items, seed=7)

    keys = [{t["case_key"] for t in part} for part in (train, dev, test)]
    assert not keys[0] & keys[1] and not keys[0] & keys[2] and not keys[1] & keys[2]
    assert len(train) + len(dev) + len(test) == len(items)
    for part in (dev, test):
        assert {t["human_pass"] for t in part} == {True, False}


def test_split_is_reproducible_for_a_seed():
    items = _labeled(30)
    assert group_stratified_split(items, seed=3) == group_stratified_split(items, seed=3)


def test_overlap_check_flags_a_case_in_two_splits():
    case_of = {1: "a", 2: "a", 3: "b"}
    assert cases_in_multiple_splits({"train": [1], "dev": [2], "test": [3]}, case_of) == ["a"]
    assert cases_in_multiple_splits({"train": [1, 2], "dev": [3], "test": []}, case_of) == []


def test_failure_is_the_positive_class():
    # The FAQ's example: humans fail 10 traces and the judge catches 8;
    # humans pass 100 and the judge passes 95. Pairs are (judge_passed, human_passed).
    pairs = [(False, False)] * 8 + [(True, False)] * 2 + [(True, True)] * 95 + [(False, True)] * 5
    m = confusion_from_pairs(pairs)
    assert (m["tp"], m["fn"], m["tn"], m["fp"]) == (8, 2, 95, 5)
    assert (m["tpr"], m["tnr"]) == (0.8, 0.95)
    assert (m["human_fail"], m["human_pass"]) == (10, 100)


def test_rogan_gladen_recovers_the_true_pass_rate():
    # True pass rate 0.6; the judge catches 80% of failures and passes 90% of good outputs.
    observed = 0.6 * 0.9 + 0.4 * (1 - 0.8)
    assert rogan_gladen_correction(observed, tpr=0.8, tnr=0.9) == 0.6
    assert rogan_gladen_correction(0.5, tpr=0.5, tnr=0.5) is None


def test_verdict_and_label_helpers():
    assert judge_passed({"score": 65, "details": {"verdict": "Pass"}}) is True
    assert judge_passed({"score": 90, "details": {"verdict": "Fail"}}) is False
    assert judge_passed({"score": 70, "details": {}}) is True
    assert judge_passed({"details": {}}) is None
    assert human_grader_passed({"groundedness": {"score": 40, "reasoning": ""}}, "groundedness") is False
    assert human_grader_passed({"groundedness": 85}, "groundedness") is True
    assert human_grader_passed({}, "groundedness") is None


def test_create_splits_puts_both_trials_of_a_case_together(eval_db, tmp_path, monkeypatch):
    loop = _load_script("calibration_loop")
    monkeypatch.setattr(loop, "CALIBRATION_DIR", tmp_path / "calibration")
    monkeypatch.setattr(loop, "SPLITS_FILE", tmp_path / "calibration" / "splits.json")
    _seed([
        {"case_key": f"case{i}", "trial_num": trial, "final_pass": 1, "human_pass": int(i % 2 == 0)}
        for i in range(40) for trial in (1, 2)
    ])

    loop.create_splits(str(eval_db))

    splits = json.loads((tmp_path / "calibration" / "splits.json").read_text())
    assert splits["group_by"] == "case_key"
    conn = sqlite3.connect(eval_db)
    case_of = dict(conn.execute("SELECT id, case_key FROM traces").fetchall())
    conn.close()
    assert cases_in_multiple_splits(splits, case_of) == []
    assert sum(len(splits[name]) for name in ("train", "dev", "test")) == 80


def test_load_splits_warns_about_trace_level_splits(eval_db, tmp_path, monkeypatch, capsys):
    loop = _load_script("calibration_loop")
    splits_file = tmp_path / "splits.json"
    monkeypatch.setattr(loop, "SPLITS_FILE", splits_file)
    ids = _seed([{"case_key": "same", "trial_num": 1, "human_pass": 1},
                 {"case_key": "same", "trial_num": 2, "human_pass": 1}])
    splits_file.write_text(json.dumps({"train": [ids[0]], "dev": [], "test": [ids[1]]}))

    loop.load_splits(str(eval_db))

    assert "more than one split" in capsys.readouterr().out


def test_per_judge_metrics_use_the_human_label_for_that_grader(eval_db):
    loop = _load_script("calibration_loop")

    def judge(verdict: str) -> dict:
        return {"score": 80 if verdict == "Pass" else 20, "category": "llm", "details": {"verdict": verdict}}

    ids = _seed([
        # Human fails the trace for another reason; groundedness itself was fine.
        {"case_key": "a", "final_pass": 1, "human_pass": 0,
         "composite_scores": {"groundedness": judge("Pass")}, "human_scores": {"groundedness": {"score": 85}}},
        {"case_key": "b", "final_pass": 0, "human_pass": 1,
         "composite_scores": {"groundedness": judge("Fail")}, "human_scores": {"groundedness": {"score": 90}}},
        {"case_key": "c", "final_pass": 0, "human_pass": 0,
         "composite_scores": {"groundedness": judge("Fail")}, "human_scores": {"groundedness": {"score": 20}}},
        # No per-grader label: counts overall, not per judge.
        {"case_key": "d", "final_pass": 1, "human_pass": 1,
         "composite_scores": {"actionability": judge("Pass")}},
    ])

    m = loop.compute_metrics(str(eval_db), ids)

    assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (1, 1, 1, 1)
    assert [c["case_key"] for c in m["missed"]] == ["a"]
    assert [c["case_key"] for c in m["false_alarms"]] == ["b"]
    assert list(m["per_judge"]) == ["groundedness"]
    g = m["per_judge"]["groundedness"]
    assert (g["tp"], g["fp"], g["tn"], g["fn"]) == (1, 1, 1, 0)
    assert (g["tpr"], g["tnr"]) == (1.0, 0.5)
