"""Build final, self-contained API inputs from pinned dataset recipes."""

from __future__ import annotations

from . import anyjev, jev, semif, wanli
from .common import cache_root

DATASETS = (*jev.TASKS, *anyjev.TASKS, "authored144", "wanli")
COUNTS = {
    **dict.fromkeys(jev.TASKS, 200),
    **dict.fromkeys(anyjev.TASKS, 300),
    "typed_decisions": 2000,
    "authored144": 144,
    "wanli": 144,
}


def request_for(row: dict) -> dict:
    """Only four public fields enter the model; gold and provenance stay outside."""
    if row["dataset"] == "wanli":
        task, state_label, query_label = wanli.TASK, "Evidence", "Question"
    else:
        task, state_label, query_label = row["caller_task"], row["state_label"], row["query_label"]
    if state_label.lower() not in task.lower() or query_label.lower() not in task.lower():
        raise ValueError("Caller task must name the same state and query terms")
    options = {option["id"]: option["description"] for option in row["options"]}
    if len(options) != len(row["options"]) or any(len(key) < 2 or key.isdigit() for key in options):
        raise ValueError("Benchmark option IDs must be unique semantic names")
    return {
        "task": task,
        "state": state_label + ":\n" + row["state"],
        "query": query_label + ":\n" + row["question"],
        "options": options,
    }


def load(name: str = "all", *, workspace=None, download: bool = False) -> list[dict]:
    if name != "all" and name not in DATASETS:
        raise ValueError(f"Unknown dataset: {name}")
    selected = DATASETS if name == "all" else (name,)
    root = cache_root(workspace)
    rows = []
    if set(selected) & set(jev.TASKS):
        items, _ = jev.load_suite(
            root, download=download, tasks=tuple(t for t in selected if t in jev.TASKS)
        )
        rows.extend(row for row in items if row["dataset"] in selected)
    for task in anyjev.TASKS:
        if task in selected:
            items, _ = anyjev.load_dataset(root, task, download=download)
            rows.extend(items)
    if "authored144" in selected:
        items, _ = semif.load_dataset(root, download=download)
        rows.extend(items)
    if "wanli" in selected:
        items, _ = wanli.load_dataset(root, download=download)
        rows.extend(
            {**row, "dataset": "wanli", "primitive": "choice"}
            for row in items
            if row["partition"] == "test"
        )
    for task in selected:
        if sum(row["dataset"] == task for row in rows) != COUNTS[task]:
            raise ValueError(f"Incomplete dataset: {task}")
    for row in rows:
        request_for(row)
    return rows
