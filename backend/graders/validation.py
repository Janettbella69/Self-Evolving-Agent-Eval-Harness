"""Judge validation: TPR/TNR measurement, Rogan-Gladen bias correction, and
leak-free calibration splits.

Follows the evals FAQ (Hamel Husain & Shreya Shankar), where the positive class
is a failure:
- TPR (True Positive Rate): of the traces a human failed, how many did the judge fail?
- TNR (True Negative Rate): of the traces a human passed, how many did the judge pass?
- Threshold: TPR >= 80% AND TNR >= 80% before trusting the judge
- Rogan-Gladen corrects aggregate pass rates for known judge bias

Records written before 2026-10-21 treated Pass as the positive class, so their
TPR and TNR are swapped relative to these definitions.
"""

import logging
import random
from collections import defaultdict
from typing import Iterable

logger = logging.getLogger(__name__)

VALIDATION_TPR_THRESHOLD = 0.80
VALIDATION_TNR_THRESHOLD = 0.80
# Rates at or above the target are on target; between threshold and target the
# judge is usable but worth improving.
VALIDATION_TARGET = 0.90
# The FAQ asks for 30-50 Pass and 30-50 Fail labels in each of dev and test.
MIN_LABELS_PER_CLASS = 30
# Pass line for 0-100 scores, shared by human per-grader scores and judge scores
# (agent/tools.py maps a Pass verdict to 70-100 and a Fail to 0-69).
PASS_SCORE = 70


def _round(value: float | None) -> float | None:
    return round(value, 4) if value is not None else None


def confusion_from_pairs(pairs: Iterable[tuple[bool, bool]]) -> dict:
    """Confusion matrix from (judge_passed, human_passed) pairs, failure = positive.

    Returns tp, fp, tn, fn, tpr, tnr, precision, recall, f1, n and the number of
    human Pass / Fail labels. Precision, recall and F1 describe the judge's Fail flags.
    """
    tp = fp = tn = fn = 0
    for judge_passed, human_passed in pairs:
        if not human_passed and not judge_passed:
            tp += 1   # failure caught
        elif not human_passed:
            fn += 1   # failure missed
        elif not judge_passed:
            fp += 1   # false alarm on a good output
        else:
            tn += 1

    tpr = tp / (tp + fn) if (tp + fn) > 0 else None
    tnr = tn / (tn + fp) if (tn + fp) > 0 else None
    precision = tp / (tp + fp) if (tp + fp) > 0 else None
    f1 = (2 * precision * tpr / (precision + tpr)) if precision and tpr else None

    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "tpr": _round(tpr),
        "tnr": _round(tnr),
        "precision": _round(precision),
        "recall": _round(tpr),
        "f1": _round(f1),
        "n": tp + fp + tn + fn,
        "human_fail": tp + fn,
        "human_pass": tn + fp,
    }


def compute_confusion_matrix(
    predictions: list[str],
    labels: list[str],
) -> dict:
    """Compute confusion matrix from predicted and human-labeled verdicts.

    Args:
        predictions: Judge verdicts ("Pass" or "Fail")
        labels: Human ground truth ("Pass" or "Fail")

    Returns the dict from confusion_from_pairs.
    """
    return confusion_from_pairs(
        (pred.lower().startswith("pass"), label.lower().startswith("pass"))
        for pred, label in zip(predictions, labels)
    )


def rogan_gladen_correction(p_obs: float, tpr: float, tnr: float) -> float | None:
    """Rogan-Gladen bias correction for aggregate pass rate.

    p_obs is the judge's observed pass rate. With failure as the positive class,
    the observed pass rate is p_true * TNR + (1 - p_true) * (1 - TPR), so

        corrected = (p_obs + TPR - 1) / (TPR + TNR - 1)

    Returns None if denominator is zero (useless classifier).
    """
    denom = tpr + tnr - 1
    if abs(denom) < 1e-6:
        return None
    corrected = (p_obs + tpr - 1) / denom
    return round(max(0, min(1, corrected)), 4)


def check_threshold(tpr: float | None, tnr: float | None) -> bool:
    """Check if TPR and TNR meet minimum deployment thresholds."""
    if tpr is None or tnr is None:
        return False
    return tpr >= VALIDATION_TPR_THRESHOLD and tnr >= VALIDATION_TNR_THRESHOLD


def judge_passed(grader_entry) -> bool | None:
    """The Pass/Fail a judge recorded in one composite_scores entry.

    Uses details.verdict, falling back to score >= PASS_SCORE for entries
    written without a verdict. None when the entry has neither.
    """
    if not isinstance(grader_entry, dict):
        return None
    verdict = str((grader_entry.get("details") or {}).get("verdict", "")).strip().lower()
    if verdict.startswith("pass"):
        return True
    if verdict.startswith("fail"):
        return False
    score = grader_entry.get("score")
    if isinstance(score, (int, float)):
        return score >= PASS_SCORE
    return None


def human_grader_passed(human_scores: dict, grader_name: str) -> bool | None:
    """A human's Pass/Fail for one grader, from the 0-100 score they entered.

    human_scores entries are {"score": x, "reasoning": ...} from the UI, or a bare
    number from scripts/annotate_batch.py. None when the grader wasn't labeled.
    """
    entry = (human_scores or {}).get(grader_name)
    if entry is None or entry == "":
        return None
    score = entry.get("score") if isinstance(entry, dict) else entry
    try:
        return float(score) >= PASS_SCORE
    except (TypeError, ValueError):
        return None


def group_stratified_split(
    items: list[dict],
    train_frac: float = 0.15,
    dev_frac: float = 0.45,
    seed: int = 42,
    group_key: str = "case_key",
    label_key: str = "human_pass",
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split labeled traces into train/dev/test with each case in exactly one split.

    Trials and re-runs of a case share its query, so a trace-level split can put
    one copy in train (where it may become a few-shot example) and another in
    dev or test. Cases are stratified by their labels (all Pass, all Fail, mixed)
    so every split keeps both classes. The default fractions follow the FAQ:
    10-20% train, 40-45% dev, the rest test.
    """
    groups: dict[str, list[dict]] = defaultdict(list)
    for item in items:
        groups[str(item[group_key])].append(item)

    strata: dict[str, list[str]] = defaultdict(list)
    for key in sorted(groups):
        labels = {bool(item[label_key]) for item in groups[key]}
        if len(labels) > 1:
            strata["mixed"].append(key)
        else:
            strata["pass" if True in labels else "fail"].append(key)

    rng = random.Random(seed)
    train: list[dict] = []
    dev: list[dict] = []
    test: list[dict] = []
    for name in ("pass", "fail", "mixed"):
        keys = strata.get(name, [])
        if not keys:
            continue
        rng.shuffle(keys)
        n_train = max(1, int(len(keys) * train_frac))
        n_dev = max(1, int(len(keys) * dev_frac))
        for rank, key in enumerate(keys):
            if rank < n_train:
                train.extend(groups[key])
            elif rank < n_train + n_dev:
                dev.extend(groups[key])
            else:
                test.extend(groups[key])
    return train, dev, test


def cases_in_multiple_splits(splits: dict, case_of: dict) -> list[str]:
    """Case keys whose traces sit in more than one of train/dev/test.

    case_of maps trace id -> case_key. A non-empty result means the splits leak.
    """
    seen: dict[str, set[str]] = defaultdict(set)
    for name in ("train", "dev", "test"):
        for trace_id in splits.get(name, []):
            key = case_of.get(trace_id)
            if key is not None:
                seen[key].add(name)
    return sorted(key for key, names in seen.items() if len(names) > 1)


async def validate_grader_from_annotations(grader_name: str) -> dict:
    """Validate a grader using existing human annotations.

    Collects all traces where human_scores[grader_name] exists,
    extracts auto verdict from composite_scores[grader_name].details.verdict,
    and computes confusion matrix.

    Returns validation result dict.
    """
    from storage import queries
    import json

    db = await queries.get_db()
    rows = await db.execute_fetchall(
        """SELECT id, composite_scores, human_scores
           FROM traces
           WHERE human_scores IS NOT NULL
             AND human_scores != '{}'
             AND composite_scores IS NOT NULL
             AND composite_scores != '{}'""",
    )

    pairs: list[tuple[bool, bool]] = []

    for row in rows:
        try:
            composite = json.loads(row[1]) if isinstance(row[1], str) else row[1]
            human = json.loads(row[2]) if isinstance(row[2], str) else row[2]
        except (json.JSONDecodeError, TypeError):
            continue

        judge = judge_passed(composite.get(grader_name))
        label = human_grader_passed(human, grader_name)
        if judge is None or label is None:
            continue
        pairs.append((judge, label))

    if len(pairs) < 5:
        return {
            "grader_name": grader_name,
            "n": len(pairs),
            "error": f"Insufficient annotations ({len(pairs)}). Need at least 5.",
            "threshold_met": False,
        }

    matrix = confusion_from_pairs(pairs)
    matrix["grader_name"] = grader_name
    matrix["positive_class"] = "fail"
    matrix["threshold_met"] = check_threshold(matrix["tpr"], matrix["tnr"])

    # Compute corrected rate if possible
    if matrix["tpr"] is not None and matrix["tnr"] is not None:
        p_obs = sum(1 for judge, _ in pairs if judge) / len(pairs)
        matrix["observed_pass_rate"] = round(p_obs, 4)
        matrix["corrected_pass_rate"] = rogan_gladen_correction(p_obs, matrix["tpr"], matrix["tnr"])

    return matrix
