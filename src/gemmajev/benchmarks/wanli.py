"""Outcome-blind, group-disjoint WANLI dev/test slices for NLI transfer checks.

Only state/question/options are model inputs. Gold, split and source-group
metadata are evaluation-only. The starting pool is SemIf's pinned 256-row
selection; no model predictions participate in our 192-row sample or split.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from .common import CONFIG_ROOT, _download_source, _verified_bytes

CONFIG_PATH = CONFIG_ROOT / "wanli.json"
REPOSITORY = "alisawuffles/WANLI"
REVISION = "61c95318fd71c55b6ba355d76253254615f387ec"
SEMIF_REVISION = "1f2dea3e25379f9dfc98cb83c324f00ab5deda37"
POOL_PATH = "benchmarks/manifests/source-selection.jsonl"
SELECTION_SEED = 20260923
MAX_ARTIFACT_BYTES = 4 * 1024 * 1024
LABELS = {"entailment": "supported", "contradiction": "contradicted", "neutral": "insufficient"}
OPTIONS = (
    {"id": "supported", "description": "The evidence establishes the claim"},
    {"id": "contradicted", "description": "The evidence establishes the opposite"},
    {"id": "insufficient", "description": "The evidence does not establish either"},
)
GROUPING = "full-test connected components of whitespace-normalized casefolded premise or pairID"
TASK = (
    "Read the evidence and answer the question using only the evidence provided.\n"
    "Distinguish evidence that contradicts a claim from missing information.\n"
    "Choose the option whose description best matches the evidence and question."
)


def _load_spec() -> dict:
    spec = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if spec.get("schema_version") != 1 or spec.get("dataset") != "wanli":
        raise ValueError("Unsupported WANLI manifest schema")
    if spec["repository"] != REPOSITORY or spec["revision"] != REVISION:
        raise ValueError("WANLI must use the pinned author repository and revision")
    if (
        spec["source"]["url"]
        != f"https://huggingface.co/datasets/{REPOSITORY}/raw/{REVISION}/test.jsonl"
    ):
        raise ValueError("WANLI source URL must use the pinned test file")
    expected_pool = (
        f"https://raw.githubusercontent.com/TheoLeeCJ/SemIf/{SEMIF_REVISION}/{POOL_PATH}"
    )
    if spec["pool"]["revision"] != SEMIF_REVISION or spec["pool"]["url"] != expected_pool:
        raise ValueError("WANLI candidate pool must use the pinned SemIf selection")
    if spec["license"]["spdx"] != "CC-BY-4.0":
        raise ValueError("Expected the upstream WANLI CC-BY-4.0 license")
    for artifact in (spec["source"], spec["pool"]):
        if not isinstance(artifact["sha256"], str) or not re.fullmatch(
            r"[0-9a-f]{64}", artifact["sha256"]
        ):
            raise ValueError("WANLI artifacts require complete SHA-256 pins")
        if (
            type(artifact["size_bytes"]) is not int
            or not 0 < artifact["size_bytes"] <= MAX_ARTIFACT_BYTES
        ):
            raise ValueError("WANLI artifact exceeds its bounded download allowance")
    selection = spec["selection"]
    if selection["seed"] != SELECTION_SEED or selection["grouping"] != GROUPING:
        raise ValueError("WANLI selection seed/grouping differs from the frozen protocol")
    for field in ("dev_per_class", "test_per_class"):
        if type(selection[field]) is not int or selection[field] < 1:
            raise ValueError("WANLI partitions require a positive per-class count")
    return spec


def _jsonl(data: bytes) -> list[dict]:
    rows = []
    for index, line in enumerate(data.decode("utf-8").split("\n"), start=1):
        if line.strip():
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"WANLI artifact row {index} is not an object")
            rows.append(row)
    return rows


def _source_id(value) -> str:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, str))
        or not re.fullmatch(r"[0-9]+", str(value))
    ):
        raise ValueError("WANLI source IDs must be nonnegative integers")
    return str(int(value))


def _text(value, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"WANLI {field} must be nonempty text")
    return value


def _indexed_source(source_rows: list[dict]) -> dict[str, dict]:
    source = {}
    for row in source_rows:
        identifier = _source_id(row.get("id"))
        if identifier in source:
            raise ValueError("Duplicate WANLI source ID")
        _text(row.get("premise"), "premise")
        _text(row.get("hypothesis"), "hypothesis")
        if row.get("gold") not in LABELS:
            raise ValueError(f"Unknown WANLI gold label for {identifier}")
        if row.get("pairID") is None or not str(row["pairID"]).strip():
            raise ValueError(f"Missing WANLI pairID for {identifier}")
        source[identifier] = row
    return source


def _connected_groups(source: dict[str, dict]) -> dict[str, str]:
    """Connect the full official test source, including rows outside the pool."""
    parents = {identifier: identifier for identifier in source}

    def root(identifier):
        while parents[identifier] != identifier:
            parents[identifier] = parents[parents[identifier]]
            identifier = parents[identifier]
        return identifier

    anchors = {}
    for identifier, row in source.items():
        keys = (
            ("premise", " ".join(row["premise"].split()).casefold()),
            ("pairID", str(row["pairID"]).strip()),
        )
        for key in keys:
            if key in anchors:
                parents[root(identifier)] = root(anchors[key])
            else:
                anchors[key] = identifier
    members = defaultdict(list)
    for identifier in source:
        members[root(identifier)].append(identifier)
    groups = {}
    for identifiers in members.values():
        canonical = "\n".join(sorted(identifiers, key=int))
        group_id = "wanli-group-" + hashlib.sha256(canonical.encode()).hexdigest()[:20]
        for identifier in identifiers:
            groups[identifier] = group_id
    return groups


def _pool_ids(pool_rows: list[dict], source: dict[str, dict], spec: dict) -> list[str]:
    identifiers = []
    for row in pool_rows:
        if row.get("source") != "wanli":
            continue
        upstream = row["upstream"]
        if upstream["revision"] != REVISION or upstream["split"] != "test":
            raise ValueError("SemIf pool references a different WANLI revision or split")
        identifier = _source_id(upstream["source_id"])
        if identifier not in source or str(source[identifier]["pairID"]) != str(
            upstream["seed_id"]
        ):
            raise ValueError("SemIf pool source ID/pairID does not match WANLI")
        identifiers.append(identifier)
    if len(identifiers) != spec["pool"]["selected_count"] or len(set(identifiers)) != len(
        identifiers
    ):
        raise ValueError("WANLI pool size or unique IDs differ from the pinned manifest")
    return identifiers


def _partition_ids(source, pool_ids, groups, selection) -> tuple[dict[str, list[str]], list[str]]:
    # One representative per connected source group, chosen by source ID before
    # looking at labels. This never uses a model's answer or confidence.
    representatives = []
    seen = set()
    for identifier in sorted(pool_ids, key=int):
        if groups[identifier] not in seen:
            representatives.append(identifier)
            seen.add(groups[identifier])
    rng = random.Random(selection["seed"])
    partitions = {"dev": [], "test": []}
    n_dev, n_test = selection["dev_per_class"], selection["test_per_class"]
    for label in LABELS:
        candidates = [
            identifier for identifier in representatives if source[identifier]["gold"] == label
        ]
        if len(candidates) < n_dev + n_test:
            raise ValueError(f"Not enough independent WANLI groups for class {label}")
        rng.shuffle(candidates)
        partitions["dev"].extend(candidates[:n_dev])
        partitions["test"].extend(candidates[n_dev : n_dev + n_test])
    for identifiers in partitions.values():
        rng.shuffle(identifiers)
    return partitions, representatives


def _normalize(source_data: bytes, pool_data: bytes, spec: dict) -> tuple[list[dict], dict]:
    source_rows, pool_rows = _jsonl(source_data), _jsonl(pool_data)
    if (
        len(source_rows) != spec["source"]["count"]
        or len(pool_rows) != spec["pool"]["manifest_count"]
    ):
        raise ValueError("WANLI source/selection-manifest counts changed")
    source = _indexed_source(source_rows)
    pool_ids = _pool_ids(pool_rows, source, spec)
    groups = _connected_groups(source)
    partitions, eligible_ids = _partition_ids(source, pool_ids, groups, spec["selection"])
    if partitions != spec["selection"]["source_partitions"]:
        raise ValueError("WANLI split IDs/order differ from the frozen selection")
    selected = partitions["dev"] + partitions["test"]
    if [f"wanli-{identifier}" for identifier in selected] != spec["selection"][
        "selected_ids"
    ] or len(eligible_ids) != spec["selection"]["eligible_group_count"]:
        raise ValueError("WANLI selected IDs or independent pool groups differ from the manifest")
    if len({groups[identifier] for identifier in selected}) != len(selected):
        raise ValueError("WANLI connected source groups overlap across selected rows")
    rows = []
    splits = {}
    for partition, identifiers in partitions.items():
        partition_rows = []
        for identifier in identifiers:
            original = source[identifier]
            gold_id = LABELS[original["gold"]]
            row = {
                "id": f"wanli-{identifier}",
                "group_id": groups[identifier],
                "family": "evidence_interpretation",
                "task": "nli",
                "partition": partition,
                "state": original["premise"],
                "question": "Assess the claim using only the supplied evidence: "
                + original["hypothesis"],
                "options": [dict(option) for option in OPTIONS],
                "gold_index": next(
                    index for index, option in enumerate(OPTIONS) if option["id"] == gold_id
                ),
                "gold_id": gold_id,
                "variant": "original",
            }
            partition_rows.append(row)
        rows.extend(partition_rows)
        splits[partition] = {
            "count": len(partition_rows),
            "selected_ids": [row["id"] for row in partition_rows],
            "source_ids": list(identifiers),
            "group_ids": [row["group_id"] for row in partition_rows],
            "label_counts": dict(Counter(row["gold_id"] for row in partition_rows)),
        }
        if splits[partition]["label_counts"] != spec["selection"]["label_counts"][partition]:
            raise ValueError("WANLI partition label distribution differs from the frozen selection")
    return rows, {
        "dataset": "wanli",
        "task": "nli",
        "repository": REPOSITORY,
        "revision": REVISION,
        "source_url": spec["source"]["url"],
        "source_sha256": spec["source"]["sha256"],
        "source_size_bytes": spec["source"]["size_bytes"],
        "source_count": len(source),
        "source_official_split": "test",
        "pool": dict(spec["pool"]),
        "pool_count": len(pool_ids),
        "eligible_group_count": len(eligible_ids),
        "selected_count": len(rows),
        "group_count": len(selected),
        "selected_ids": [row["id"] for row in rows],
        "source_selected_ids": selected,
        "selection_seed": spec["selection"]["seed"],
        "grouping": GROUPING,
        "selection_method": spec["selection"]["method"],
        "splits": splits,
        "label_counts": dict(Counter(row["gold_id"] for row in rows)),
        "license": dict(spec["license"]),
        "docs": dict(spec["docs"]),
        "annotation_status": spec["annotation_status"],
    }


def load_dataset(cache_root: Path, download: bool = False) -> tuple[list[dict], dict]:
    """Load fixed dev48/test144 slices, preparing pinned bytes only on request."""
    spec = _load_spec()
    directory = Path(cache_root) / "wanli" / REVISION
    artifacts = []
    for filename, record in (
        ("test.jsonl", spec["source"]),
        (f"semif-source-selection-{SEMIF_REVISION}.jsonl", spec["pool"]),
    ):
        path = directory / filename
        if not path.exists() and not path.is_symlink():
            if not download:
                raise FileNotFoundError(
                    f"Missing dataset file: {path}. Run: uv run --extra data gemmajev benchmark prepare --dataset wanli"
                )
            _download_source(record, path)
        artifacts.append(_verified_bytes(path, record))
    return _normalize(artifacts[0], artifacts[1], spec)
