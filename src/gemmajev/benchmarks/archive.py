"""Verify and render the compact published measurements, fully offline."""

from __future__ import annotations

import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

from .common import CONFIG_ROOT, PACKAGE, ROOT
from .data import COUNTS
from .metrics import summarize

RESULTS = ROOT / "docs/results"
if not RESULTS.is_dir():
    RESULTS = PACKAGE / "results"
TITLES = {
    "ag_news": "AG News",
    "boolq": "BoolQ",
    "mmlu": "MMLU",
    "logiqa": "LogiQA",
    "anli_r3": "ANLI R3",
    "emotion": "Emotion",
    "essay_or_readability": "HelpSteer2",
    "banking20": "Banking20",
    "newsgroups": "20 Newsgroups",
    "injection": "Prompt injection",
    "typed_decisions": "Typed decisions",
    "authored144": "Authored144",
    "wanli": "WANLI",
}


def read_jsonl(path: Path) -> list[dict]:
    opener = gzip.open if path.suffix == ".gz" else Path.open
    with opener(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def verify(directory: Path = RESULTS) -> dict:
    """Recompute every local table cell and verify source manifest/reference hashes."""
    summary = json.loads((directory / "summary.json").read_text())
    for section in ("archive", "references"):
        spec = summary[section]
        data = (directory / spec["file"]).read_bytes()
        if hashlib.sha256(data).hexdigest() != spec["sha256"]:
            raise ValueError(f"{section} checksum mismatch")
    for name, expected in summary["dataset_manifests"].items():
        if hashlib.sha256((CONFIG_ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Dataset manifest changed: {name}")
    rows = read_jsonl(directory / summary["archive"]["file"])
    if len(rows) != summary["archive"]["rows"] or len(rows) != 2 * sum(COUNTS.values()):
        raise ValueError("Incomplete published archive")
    for model in ("e4b", "12b"):
        if Counter(r["dataset"] for r in rows if r["model"] == model) != COUNTS:
            raise ValueError("Incomplete model/dataset panel")
    if {r["model"] for r in rows} != {"e4b", "12b"}:
        raise ValueError("Unexpected model in archive")
    computed = summarize(rows)
    if computed != summary["results"]:
        raise ValueError("Recomputed metrics differ from the published summary")
    return summary


def primary_metric(dataset: str) -> str:
    if dataset in {"banking20", "newsgroups", "injection"}:
        return "macro_f1"
    return "spearman" if dataset == "essay_or_readability" else "accuracy"


METRIC_LABELS = {
    "accuracy": "Accuracy ↑",
    "macro_f1": "Macro-F1 ↑",
    "spearman": "Spearman ↑",
    "score_mae": "Score MAE ↓",
}


def format_metric(metric: str, value: float) -> str:
    return f"{100 * value:.2f}%" if metric in {"accuracy", "macro_f1"} else f"{value:.4f}"


def markdown(summary: dict, references: list[dict]) -> str:
    lines = ["| Dataset | Metric | N | E4B | 12B |", "| --- | --- | ---: | ---: | ---: |"]
    for dataset in COUNTS:
        panels = [summary["results"][model][dataset] for model in ("e4b", "12b")]
        metrics = [primary_metric(dataset)]
        if all("score_mae" in panel for panel in panels):
            metrics.append("score_mae")
        for metric in metrics:
            count_key = "score_n" if metric == "score_mae" else "n"
            if panels[0][count_key] != panels[1][count_key]:
                raise ValueError("Compared local metric denominators differ")
            cells = [format_metric(metric, panel[metric]) for panel in panels]
            lines.append(
                f"| {TITLES[dataset]} | {METRIC_LABELS[metric]} | {panels[0][count_key]} | {' | '.join(cells)} |"
            )
    lines.extend(
        [
            "",
            "| Reference configuration | Dataset | Metric | N | Value |",
            "| --- | --- | --- | ---: | ---: |",
        ]
    )
    for row in references:
        dataset, measured = row["dataset"], row["metrics"]
        metrics = [primary_metric(dataset)]
        if "score_mae" in measured:
            metrics.append("score_mae")
        provider = row.get("provider", "")
        system = row["system"]
        label = f"{system if system.startswith(provider) else provider + ' ' + system} / {row['configuration']}"
        for metric in metrics:
            if metric not in measured:
                raise ValueError(
                    f"Published reference lacks the comparison metric: {dataset}/{metric}"
                )
            count_key = "score_n" if metric == "score_mae" else "n"
            count = measured.get(count_key, "—")
            value = format_metric(metric, measured[metric])
            lines.append(
                f"| [{label}]({row['source']['url']}) | {TITLES[dataset]} | {METRIC_LABELS[metric]} | {count} | {value} |"
            )
    return "\n".join(lines) + "\n"
