"""Dataset management routes."""

import json
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from storage import queries
from storage.models import DatasetIn

router = APIRouter(prefix="/api/datasets", tags=["datasets"])

DATASETS_DIR = Path(__file__).parent.parent.parent / "datasets"


@router.get("")
async def list_datasets():
    datasets = await queries.list_datasets()
    return [d.model_dump() for d in datasets]


@router.get("/{dataset_id}")
async def get_dataset(dataset_id: int):
    dataset = await queries.get_dataset(dataset_id)
    if not dataset:
        return JSONResponse(status_code=404, content={"detail": "Dataset not found."})
    cases = await queries.get_cases(dataset_id)
    return {
        **dataset.model_dump(),
        "cases": [c.model_dump() for c in cases],
    }


@router.post("/import")
async def import_dataset(body: DatasetIn | None = None, name: str | None = None):
    """Import a dataset from JSON file or request body.

    If `name` is provided (query param), loads from datasets/{name}.json.
    If body is provided, uses inline cases.
    """
    if name:
        # Load from file
        file_path = DATASETS_DIR / f"{name}.json"
        if not file_path.exists():
            return JSONResponse(status_code=404, content={"detail": f"Dataset file not found: {name}.json"})
        data = json.loads(file_path.read_text())
        dataset_name = data.get("name", name)
        description = data.get("description", "")
        cases = data.get("cases", [])
    elif body:
        dataset_name = body.name
        description = body.description
        cases = [c.model_dump() for c in body.cases]
    else:
        return JSONResponse(status_code=422, content={"detail": "Provide name (query param) or body."})

    suite_type = "capability"
    if body and hasattr(body, "suite_type"):
        suite_type = body.suite_type
    elif name:
        suite_type = data.get("suite_type", "capability")

    # Upsert dataset
    existing = await queries.get_dataset_by_name(dataset_name)
    if existing:
        dataset_id = existing.id
        await queries.update_dataset(dataset_id, suite_type=suite_type)
    else:
        dataset_id = await queries.create_dataset(dataset_name, description, suite_type)

    # Upsert cases
    count = 0
    for case in cases:
        await queries.upsert_case(
            dataset_id,
            case["key"],
            case["query"],
            case.get("type", "clear_en"),
            case.get("constraints", {}),
            case.get("golden_data", {}),
            case.get("reference_output"),
        )
        count += 1

    # Bump dataset version once after batch import
    if count > 0:
        version = await queries.bump_dataset_version(dataset_id)
    else:
        ds = await queries.get_dataset(dataset_id)
        version = ds.version if ds else 1

    return {"dataset_id": dataset_id, "name": dataset_name, "cases_imported": count, "version": version}


@router.post("/from-experiment")
async def create_dataset_from_experiment(body: dict):
    """Create a debug dataset by filtering cases from an experiment.

    Body:
        experiment_id: int — source experiment
        name: str — new dataset name
        filter: str — "failed" | "passed" | "regressed" | "improved"
        compare_to: int | None — base experiment for regressed/improved filter
        description: str — optional
    """
    experiment_id = body.get("experiment_id")
    name = body.get("name")
    filter_type = body.get("filter", "failed")
    compare_to = body.get("compare_to")
    description = body.get("description", "")

    if not experiment_id or not name:
        return JSONResponse(status_code=422, content={"detail": "experiment_id and name are required."})

    if filter_type in ("regressed", "improved") and not compare_to:
        return JSONResponse(status_code=422, content={
            "detail": f"compare_to is required for filter type '{filter_type}'."
        })

    result = await queries.create_dataset_from_experiment(
        experiment_id=experiment_id,
        name=name,
        filter_type=filter_type,
        compare_to=compare_to,
        description=description,
    )

    if "error" in result:
        return JSONResponse(status_code=404, content={"detail": result["error"]})

    return result


@router.patch("/{dataset_id}")
async def update_dataset(dataset_id: int, body: dict):
    """Update dataset metadata (suite_type, description)."""
    dataset = await queries.get_dataset(dataset_id)
    if not dataset:
        return JSONResponse(status_code=404, content={"detail": "Dataset not found."})

    allowed = {"suite_type", "description"}
    updates = {k: v for k, v in body.items() if k in allowed}
    if not updates:
        return JSONResponse(status_code=422, content={"detail": "No valid fields."})

    await queries.update_dataset(dataset_id, **updates)
    return {"ok": True, "updated": list(updates.keys())}


@router.get("/{dataset_id}/staleness")
async def get_dataset_staleness(dataset_id: int, max_age_days: int = 30):
    """Check which cases have stale or never-validated golden data.

    Query params:
        max_age_days: Cases older than this are "stale" (default 30).
    """
    dataset = await queries.get_dataset(dataset_id)
    if not dataset:
        return JSONResponse(status_code=404, content={"detail": "Dataset not found."})
    return await queries.get_staleness_report(dataset_id, max_age_days)


@router.post("/{dataset_id}/validate")
async def validate_dataset_cases(dataset_id: int, body: dict):
    """Mark cases as freshly validated.

    Body:
        case_keys: list[str] — keys to mark as validated
        all: bool — if true, validate all cases in the dataset
    """
    dataset = await queries.get_dataset(dataset_id)
    if not dataset:
        return JSONResponse(status_code=404, content={"detail": "Dataset not found."})

    if body.get("all"):
        cases = await queries.get_cases(dataset_id)
        case_keys = [c.key for c in cases]
    else:
        case_keys = body.get("case_keys", [])

    if not case_keys:
        return JSONResponse(status_code=422, content={"detail": "No case_keys provided."})

    count = await queries.validate_cases(dataset_id, case_keys)
    return {"ok": True, "validated_count": count}


@router.get("/{dataset_id}/saturation")
async def get_saturation(dataset_id: int):
    """Per-case saturation summary — identifies saturated (100% pass) cases."""
    cases = await queries.get_saturation_summary(dataset_id)
    saturated = sum(1 for c in cases if c["saturated"])
    return {
        "dataset_id": dataset_id,
        "total_cases": len(cases),
        "saturated_cases": saturated,
        "cases": cases,
    }


@router.get("/cases/history")
async def get_case_history_endpoint(case_key: str, dataset_id: int | None = None):
    """Score history for a case across experiments (saturation tracking)."""
    history = await queries.get_case_history(case_key, dataset_id)
    return {"case_key": case_key, "history": history}


@router.post("/import-shoppingcomp")
async def import_shoppingcomp_endpoint(file_path: str, name: str, description: str = ""):
    """Import a ShoppingComp JSONL file via API."""
    from loaders.shoppingcomp import import_shoppingcomp

    path = Path(file_path)
    if not path.is_absolute():
        path = DATASETS_DIR.parent.parent / path

    if not path.exists():
        return JSONResponse(status_code=404, content={"detail": f"File not found: {path}"})

    result = await import_shoppingcomp(path, name, description)
    return {
        "dataset_id": result.dataset_id,
        "imported": result.imported,
        "skipped": result.skipped,
        "types": result.types,
        "categories": result.categories,
    }


@router.post("/import-shoppingcomp-all")
async def import_shoppingcomp_all():
    """Import all 4 ShoppingComp JSONL files (EN/ZH × regular/trap)."""
    from loaders.shoppingcomp import import_shoppingcomp

    # ShoppingComp files live at <repo_root>/datasets/ShoppingComp/
    sc_dir = DATASETS_DIR.parent.parent / "datasets" / "ShoppingComp"

    files = [
        {
            "path": sc_dir / "ShoppingComp_97_20260127.en.jsonl",
            "name": "ShoppingComp EN",
            "description": "ShoppingComp benchmark — 97 English cases with expert-annotated golden data",
        },
        {
            "path": sc_dir / "ShoppingComp_97_20260127.zh.jsonl",
            "name": "ShoppingComp ZH",
            "description": "ShoppingComp benchmark — 97 Chinese cases with expert-annotated golden data",
        },
        {
            "path": sc_dir / "ShoppingComp_traps_48_20260127.en.jsonl",
            "name": "ShoppingComp Traps EN",
            "description": "ShoppingComp trap cases — 48 English adversarial queries",
        },
        {
            "path": sc_dir / "ShoppingComp_traps_48_20260127.zh.jsonl",
            "name": "ShoppingComp Traps ZH",
            "description": "ShoppingComp trap cases — 48 Chinese adversarial queries",
        },
    ]

    results = []
    for f in files:
        if not f["path"].exists():
            results.append({"name": f["name"], "error": f"File not found: {f['path']}"})
            continue
        try:
            r = await import_shoppingcomp(f["path"], f["name"], f["description"])
            results.append({
                "name": f["name"],
                "dataset_id": r.dataset_id,
                "imported": r.imported,
                "skipped": r.skipped,
                "types": r.types,
                "categories": r.categories,
            })
        except Exception as e:
            results.append({"name": f["name"], "error": str(e)})

    total_imported = sum(r.get("imported", 0) for r in results)
    return {"results": results, "total_imported": total_imported}


# ── Trace Backflow (replace LangFuse) ────────────────────────────────

@router.post("/{dataset_id}/backflow")
async def trace_backflow(dataset_id: int, body: dict):
    """Save a trace as a new case in a dataset (Trace data backflow).

    Per Coze Loop pattern: production traces become ground truth for future evaluation.
    Replaces the LangFuse → eval import pipeline with direct backflow.

    Body:
        query: str — the original user query (becomes case.query)
        guide_text: str — agent's output guide
        products: list — found products
        sources: list — cited sources
        events: list — raw SSE events (becomes trajectory reference)
        hook_metrics: dict — agent metrics
        trace_id: str — optional, for dedup
        category: str — optional, product category tag
    """
    dataset = await queries.get_dataset(dataset_id)
    if not dataset:
        return JSONResponse(status_code=404, content={"detail": "Dataset not found."})

    query = body.get("query", "")
    if not query:
        return JSONResponse(status_code=422, content={"detail": "query is required."})

    # Generate case key from query hash (or use trace_id)
    import hashlib
    trace_id = body.get("trace_id", "")
    case_key = trace_id[:8] if trace_id else hashlib.sha256(query.encode()).hexdigest()[:8]

    # Build golden_data from agent output (this trace becomes the reference)
    golden_data = {
        "category": body.get("category", "other"),
        "source": "backflow",
        "reference_guide": body.get("guide_text", "")[:5000],
        "reference_products": body.get("products", [])[:10],
        "reference_sources": body.get("sources", [])[:20],
    }

    # Build reference_output with full trajectory
    reference_output = {
        "guide_text": body.get("guide_text", ""),
        "products": body.get("products", []),
        "sources": body.get("sources", []),
        "events": body.get("events", []),
        "hook_metrics": body.get("hook_metrics", {}),
    }

    await queries.upsert_case(
        dataset_id=dataset_id,
        key=case_key,
        query=query,
        case_type="backflow",
        constraints={},
        golden_data=golden_data,
        reference_output=reference_output,
    )

    await queries.bump_dataset_version(dataset_id)

    return {
        "ok": True,
        "dataset_id": dataset_id,
        "case_key": case_key,
        "query": query[:80],
    }
