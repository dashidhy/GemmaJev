"""Pure Python scoring of semantic-ID predictions; no model or dataset required."""

from __future__ import annotations

import math
from collections import Counter, defaultdict

FIXED_LABEL_TASKS = {
    "ag_news",
    "boolq",
    "anli_r3",
    "emotion",
    "banking20",
    "newsgroups",
    "injection",
    "wanli",
}


def validate(row: dict) -> None:
    for field in ("model", "id", "dataset", "group_id", "gold_id", "predicted_id"):
        if not isinstance(row.get(field), str) or not row[field]:
            raise ValueError(f"Missing {field}")
    p = row["probabilities"]
    if not isinstance(p, dict) or not 2 <= len(p) <= 26:
        raise ValueError("Expected 2–26 semantic probabilities")
    if any(
        type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1
        for value in p.values()
    ):
        raise ValueError("Invalid probability")
    if not math.isclose(math.fsum(p.values()), 1, abs_tol=1e-6, rel_tol=0):
        raise ValueError("Probabilities must sum to one")
    if row["gold_id"] not in p or row["predicted_id"] not in p:
        raise ValueError("Unknown selected or gold ID")
    if p[row["predicted_id"]] < max(p.values()) - 1e-12:
        raise ValueError("Selected ID is not an argmax")
    if "option_ids" in row and set(row["option_ids"]) != set(p):
        raise ValueError("Probability option set differs from source")
    if row.get("primitive") == "score":
        values = row["score_values"]
        if set(values) != set(p) or any(not math.isfinite(v) for v in values.values()):
            raise ValueError("Score scale differs from options")
        if not math.isfinite(row["gold_score"]) or not min(values.values()) <= row[
            "gold_score"
        ] <= max(values.values()):
            raise ValueError("Gold score outside scale")
    latency = row.get("request_ms")
    if latency is not None and (not math.isfinite(latency) or latency < 0):
        raise ValueError("Invalid request duration")


def ranks(values):
    order = sorted(range(len(values)), key=values.__getitem__)
    result = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        for i in order[start:end]:
            result[i] = (start + 1 + end) / 2
        start = end
    return result


def spearman(left, right):
    if len(left) != len(right) or len(left) < 2:
        return None
    center = (len(left) + 1) / 2
    a, b = [[r - center for r in ranks(values)] for values in (left, right)]
    denominator = math.sqrt(math.fsum(x * x for x in a) * math.fsum(x * x for x in b))
    return (
        math.fsum(x * y for x, y in zip(a, b, strict=True)) / denominator if denominator else None
    )


def auroc(probabilities, labels):
    positives, negatives = sum(labels), len(labels) - sum(labels)
    if not positives or not negatives:
        return None
    numerator = math.fsum(
        rank for rank, label in zip(ranks(probabilities), labels, strict=True) if label
    )
    return (numerator - positives * (positives + 1) / 2) / (positives * negatives)


def quantile(values, fraction):
    ordered = sorted(values)
    position = (len(values) - 1) * fraction
    low = int(position)
    return ordered[low] + (ordered[min(low + 1, len(values) - 1)] - ordered[low]) * (position - low)


def summarize(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("No predictions")
    seen, groups = set(), defaultdict(list)
    for row in rows:
        validate(row)
        key = row["model"], row["dataset"], row["id"]
        if key in seen:
            raise ValueError("Duplicate prediction")
        seen.add(key)
        groups[row["model"], row["dataset"]].append(row)
    result = {}
    for (model, dataset), items in sorted(groups.items()):
        correct = sum(r["gold_id"] == r["predicted_id"] for r in items)
        summary = {
            "n": len(items),
            "correct": correct,
            "accuracy": correct / len(items),
            "source_groups": len({r["group_id"] for r in items}),
        }
        if dataset in FIXED_LABEL_TASKS:
            gold = Counter(r["gold_id"] for r in items)
            chosen = Counter(r["predicted_id"] for r in items)
            tp = Counter(r["gold_id"] for r in items if r["gold_id"] == r["predicted_id"])
            labels = gold.keys() | chosen.keys()
            summary["macro_f1"] = math.fsum(
                2 * tp[k] / (gold[k] + chosen[k]) for k in labels
            ) / len(labels)
        scores = [r for r in items if r.get("primitive") == "score"]
        if scores:
            expected = [
                math.fsum(r["probabilities"][k] * v for k, v in r["score_values"].items())
                for r in scores
            ]
            targets = [r["gold_score"] for r in scores]
            summary.update(
                score_n=len(scores),
                score_mae=math.fsum(abs(a - b) for a, b in zip(expected, targets, strict=True))
                / len(scores),
            )
            if dataset == "essay_or_readability":
                summary["spearman"] = spearman(expected, targets)
        if dataset == "boolq":
            summary["auroc"] = auroc(
                [r["probabilities"]["yes"] for r in items], [r["gold_id"] == "yes" for r in items]
            )
        latencies = [r["request_ms"] for r in items if r.get("request_ms") is not None]
        if latencies:
            summary["latency_ms"] = {
                "n": len(latencies),
                "mean": math.fsum(latencies) / len(latencies),
                "p50": quantile(latencies, 0.5),
                "p95": quantile(latencies, 0.95),
            }
        result.setdefault(model, {})[dataset] = summary
    return result
