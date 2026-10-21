"""A stop request or a tripped circuit breaker must keep queued cases from starting."""

import asyncio

from runner import executor
from runner.collector import CollectedResult
from runner.progress import get_manager
from storage import queries
from storage.database import close_db, init_db


def _cases(n: int) -> list[dict]:
    return [{"key": f"c{i}", "query": f"query {i}", "type": "clear_en", "constraints": {}, "golden_data": {}}
            for i in range(n)]


def _collected() -> CollectedResult:
    return CollectedResult(guide_text="g" * 600, products=[{"name": "p"}], sources=[{"url": "https://example.com"}])


async def _fake_grade(trace, case) -> dict:
    return {
        "l0": {}, "l1": {"breakdown": {}}, "l2": None,
        "final_score": 80.0, "final_pass": True,
        "composite_scores": {}, "failure_funnel": {}, "error_types": [],
        "grading_duration_s": 0, "grading_log": [], "judge_prompt_version": "test",
    }


async def _new_experiment(name: str) -> int:
    dataset_id = await queries.create_dataset(name)
    return await queries.create_experiment(dataset_id, name, {})


def _drain(q: asyncio.Queue) -> list[dict]:
    out = []
    while not q.empty():
        out.append(q.get_nowait())
    return out


def _run(scenario):
    """Run `scenario()` against a fresh database and always close the connection."""
    async def wrapped():
        await init_db()
        try:
            return await scenario()
        finally:
            await close_db()
    return asyncio.run(wrapped())


async def _finish(exp_id: int, messages: asyncio.Queue):
    return (
        await queries.get_experiment(exp_id),
        await queries.get_experiment_traces(exp_id),
        _drain(messages),
    )


def test_stop_keeps_queued_cases_from_starting(monkeypatch):
    started: list[str] = []
    monkeypatch.setattr(executor, "_grade_single_trace", _fake_grade)

    async def scenario():
        first_started, release = asyncio.Event(), asyncio.Event()

        async def fake_collect(case, trial_num=1, **kwargs):
            started.append(case["key"])
            first_started.set()
            await release.wait()
            return _collected()

        monkeypatch.setattr(executor, "collect_single_case", fake_collect)
        exp_id = await _new_experiment("stop")
        messages: asyncio.Queue = asyncio.Queue()
        get_manager().subscribe(exp_id, messages)

        run = asyncio.create_task(executor.run_experiment(exp_id, _cases(4), concurrency=1))
        await asyncio.wait_for(first_started.wait(), timeout=5)  # c0 in flight, c1..c3 queued
        executor.request_stop(exp_id)
        release.set()  # the in-flight case is allowed to finish and be graded
        await asyncio.wait_for(run, timeout=10)
        return await _finish(exp_id, messages)

    exp, traces, messages = _run(scenario)

    assert started == ["c0"]
    assert [(t.case_key, t.status, t.final_score) for t in traces] == [("c0", "done", 80.0)]
    assert exp.status == "stopped"
    done = next(m for m in messages if m["type"] == "experiment_done")
    assert (done["status"], done["skipped"]) == ("stopped", 3)
    assert [m["case_key"] for m in messages if m["type"] == "case_start"] == ["c0"]
    assert exp.id not in executor._cancel_flags  # the request is cleared once the run ends


def test_stop_after_every_case_started_is_still_a_complete_run(monkeypatch):
    monkeypatch.setattr(executor, "_grade_single_trace", _fake_grade)

    async def scenario():
        all_started, release = asyncio.Event(), asyncio.Event()
        n_started = 0

        async def fake_collect(case, trial_num=1, **kwargs):
            nonlocal n_started
            n_started += 1
            if n_started == 3:
                all_started.set()
            await release.wait()
            return _collected()

        monkeypatch.setattr(executor, "collect_single_case", fake_collect)
        exp_id = await _new_experiment("late-stop")
        messages: asyncio.Queue = asyncio.Queue()
        get_manager().subscribe(exp_id, messages)

        run = asyncio.create_task(executor.run_experiment(exp_id, _cases(3), concurrency=3))
        await asyncio.wait_for(all_started.wait(), timeout=5)
        executor.request_stop(exp_id)  # nothing left to cancel
        release.set()
        await asyncio.wait_for(run, timeout=10)
        return await _finish(exp_id, messages)

    exp, traces, messages = _run(scenario)

    assert sorted(t.status for t in traces) == ["done", "done", "done"]
    assert exp.status == "complete"
    assert next(m for m in messages if m["type"] == "experiment_done")["skipped"] == 0


def test_stale_stop_request_does_not_cancel_the_next_run(monkeypatch):
    monkeypatch.setattr(executor, "_grade_single_trace", _fake_grade)

    async def fake_collect(case, trial_num=1, **kwargs):
        return _collected()

    monkeypatch.setattr(executor, "collect_single_case", fake_collect)

    async def scenario():
        exp_id = await _new_experiment("stale-stop")
        executor.request_stop(exp_id)  # experiment is not running yet
        await asyncio.wait_for(executor.run_experiment(exp_id, _cases(2), concurrency=1), timeout=10)
        return await _finish(exp_id, asyncio.Queue())

    exp, traces, _ = _run(scenario)
    assert exp.status == "complete" and len(traces) == 2


def test_circuit_breaker_stops_the_queue_after_threshold_errors(monkeypatch):
    async def failing_collect(case, trial_num=1, **kwargs):
        raise RuntimeError("product backend down")

    monkeypatch.setattr(executor, "collect_single_case", failing_collect)

    async def scenario():
        exp_id = await _new_experiment("breaker")
        messages: asyncio.Queue = asyncio.Queue()
        get_manager().subscribe(exp_id, messages)
        await asyncio.wait_for(executor.run_experiment(exp_id, _cases(8), concurrency=1), timeout=10)
        return await _finish(exp_id, messages)

    exp, traces, messages = _run(scenario)
    threshold = executor.CIRCUIT_BREAKER_THRESHOLD

    assert [t.status for t in traces] == ["error"] * threshold
    assert exp.status == "paused"
    assert len([m for m in messages if m["type"] == "circuit_break"]) == 1
    done = next(m for m in messages if m["type"] == "experiment_done")
    assert (done["status"], done["skipped"]) == ("paused", 8 - threshold)


def test_circuit_breaker_with_parallel_collection_lets_only_in_flight_cases_finish(monkeypatch):
    async def failing_collect(case, trial_num=1, **kwargs):
        await asyncio.sleep(0.01)
        raise RuntimeError("product backend down")

    monkeypatch.setattr(executor, "collect_single_case", failing_collect)
    concurrency = 3

    async def scenario():
        exp_id = await _new_experiment("breaker-parallel")
        messages: asyncio.Queue = asyncio.Queue()
        get_manager().subscribe(exp_id, messages)
        await asyncio.wait_for(executor.run_experiment(exp_id, _cases(12), concurrency=concurrency), timeout=10)
        return await _finish(exp_id, messages)

    exp, traces, messages = _run(scenario)
    threshold = executor.CIRCUIT_BREAKER_THRESHOLD

    # Cases already holding a slot when the breaker trips still run; nothing queued does.
    assert threshold <= len(traces) <= threshold + concurrency - 1
    assert {t.status for t in traces} == {"error"}
    assert exp.status == "paused"
    assert len([m for m in messages if m["type"] == "circuit_break"]) == 1
    assert next(m for m in messages if m["type"] == "experiment_done")["skipped"] == 12 - len(traces)


def test_a_success_resets_the_consecutive_error_count(monkeypatch):
    monkeypatch.setattr(executor, "_grade_single_trace", _fake_grade)
    fails = {"c0", "c1", "c3", "c4"}  # never three in a row

    async def flaky_collect(case, trial_num=1, **kwargs):
        if case["key"] in fails:
            raise RuntimeError("timeout")
        return _collected()

    monkeypatch.setattr(executor, "collect_single_case", flaky_collect)

    async def scenario():
        exp_id = await _new_experiment("flaky")
        await asyncio.wait_for(executor.run_experiment(exp_id, _cases(6), concurrency=1), timeout=10)
        return await _finish(exp_id, asyncio.Queue())

    exp, traces, _ = _run(scenario)
    assert sorted(t.status for t in traces) == ["done", "done", "error", "error", "error", "error"]
    assert exp.status == "complete"


def test_regrading_a_stopped_run_does_not_relabel_it_complete(monkeypatch):
    monkeypatch.setattr(executor, "_grade_single_trace", _fake_grade)

    async def scenario():
        first_started, release = asyncio.Event(), asyncio.Event()

        async def fake_collect(case, trial_num=1, **kwargs):
            first_started.set()
            await release.wait()
            return _collected()

        monkeypatch.setattr(executor, "collect_single_case", fake_collect)
        exp_id = await _new_experiment("regrade-stopped")
        run = asyncio.create_task(executor.run_experiment(exp_id, _cases(3), concurrency=1))
        await asyncio.wait_for(first_started.wait(), timeout=5)
        executor.request_stop(exp_id)
        release.set()
        await asyncio.wait_for(run, timeout=10)

        regraded = await executor.regrade_experiment(exp_id)
        return regraded, await queries.get_experiment(exp_id)

    regraded, exp = _run(scenario)
    assert regraded == 1
    assert exp.status == "stopped"
