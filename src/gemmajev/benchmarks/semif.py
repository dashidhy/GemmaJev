"""Complete pinned SemIf authored144 and strictly aligned public predictions."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from .common import CONFIG_ROOT, _download_source, _verified_bytes

CONFIG_PATH = CONFIG_ROOT / "semif.json"
REVISION = "1f2dea3e25379f9dfc98cb83c324f00ab5deda37"
REPOSITORY = "TheoLeeCJ/SemIf"
FAMILIES = ("evidence_interpretation", "rule_application", "candidate_selection")
CALLER_TASK = (
    "Read the Evidence and answer the Question using only the supplied information "
    "and option descriptions."
)


def _encoded(rows):
    return b"".join(
        json.dumps(
            row, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
        + b"\n"
        for row in rows
    )


def _text(value, field):
    if not isinstance(value, str) or not value.strip() or "\0" in value:
        raise ValueError(f"SemIf {field} must be nonempty text without NUL")
    return value


def _load_spec():
    spec = json.loads(CONFIG_PATH.read_text())
    if (
        spec.get("schema_version") != 1
        or spec["repository"] != REPOSITORY
        or spec["revision"] != REVISION
        or spec["count"] != 144
        or spec["groups"] != 36
    ):
        raise ValueError("Unexpected SemIf authored144 manifest")
    if spec["family_counts"] != dict.fromkeys(FAMILIES, 48):
        raise ValueError("SemIf must retain all three complete families")
    for artifact in [spec["source"], *spec["references"]]:
        if (
            artifact["url"]
            != f"https://raw.githubusercontent.com/{REPOSITORY}/{REVISION}/{artifact['path']}"
        ):
            raise ValueError("SemIf source URL differs from its pinned path")
        if not re.fullmatch(r"[0-9a-f]{64}", artifact["sha256"]):
            raise ValueError("SemIf artifacts require full SHA-256 pins")
        if type(artifact["size_bytes"]) is not int or not 0 < artifact["size_bytes"] <= 1024 * 1024:
            raise ValueError("SemIf artifact exceeds its bounded size")
        if Path(artifact["cache_file"]).name != artifact["cache_file"]:
            raise ValueError("SemIf cache files must be plain filenames")
    return spec


def _read(cache, artifact, download):
    path = Path(cache) / "semif" / REVISION / artifact["cache_file"]
    if not path.exists() and not path.is_symlink():
        if not download:
            raise FileNotFoundError(
                f"Missing dataset file: {path}. Run: uv run --extra data gemmajev benchmark prepare --dataset authored144"
            )
        _download_source(artifact, path)
    data = _verified_bytes(path, artifact)
    rows = [json.loads(line) for line in data.decode().split("\n") if line.strip()]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError("SemIf JSONL must contain objects")
    return rows


def _normalize(source, spec):
    if len(source) != spec["count"] or len({r["id"] for r in source}) != len(source):
        raise ValueError("SemIf source count or unique IDs changed")
    if Counter(r["family"] for r in source) != spec["family_counts"]:
        raise ValueError("SemIf family population changed")
    groups = Counter(r["group_id"] for r in source)
    if len(groups) != spec["groups"] or any(n != spec["rows_per_group"] for n in groups.values()):
        raise ValueError("SemIf source groups must retain every related variant")
    rows = []
    for original in source:
        if original["split"] != "test":
            raise ValueError("SemIf authored panel must be the original test population")
        state = original["state"]
        if not isinstance(state, str):
            state = json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        original_ids = [option["id"] for option in original["options"]]
        if not 2 <= len(original_ids) <= 26 or len(set(original_ids)) != len(original_ids):
            raise ValueError("SemIf semantic option IDs must be unique")
        mapping = spec["semantic_id_aliases"].get(original["family"], {})
        options = [
            {
                "id": mapping.get(o["id"], o["id"]),
                "description": _text(o["description"], "option description"),
            }
            for o in original["options"]
        ]
        semantic_ids = [o["id"] for o in options]
        if len(set(semantic_ids)) != len(options) or any(
            len(_text(i, "option ID")) == 1 or i.isdigit() for i in semantic_ids
        ):
            raise ValueError("SemIf aliases must be unique semantic names, not answer letters")
        label = original["label"]
        if type(label) is not int or not 0 <= label < len(options):
            raise ValueError("SemIf gold label is outside the original option order")
        rows.append(
            {
                "id": _text(original["id"], "id"),
                "group_id": _text(original["group_id"], "group_id"),
                "dataset": "authored144",
                "family": original["family"],
                "task": "classification",
                "primitive": "choice",
                "partition": "test",
                "variant": original["provenance"]["variant"],
                "state": _text(state, "state"),
                "question": _text(original["question"], "question"),
                "options": options,
                "gold_index": label,
                "gold_id": options[label]["id"],
                "upstream_option_ids": original_ids,
                "caller_task": CALLER_TASK,
                "state_label": "Evidence",
                "query_label": "Question",
            }
        )
    return rows


def load_dataset(cache_root: Path, download: bool = False) -> tuple[list[dict], dict]:
    """Return all144 rows in source order, with private annotation metadata removed."""
    spec = _load_spec()
    rows = _normalize(_read(cache_root, spec["source"], download), spec)
    digest = hashlib.sha256(_encoded(rows)).hexdigest()
    if digest != spec["normalized_sha256"]:
        raise ValueError("SemIf normalized content differs from the frozen snapshot")
    return rows, {
        "dataset": "authored144",
        "repository": REPOSITORY,
        "revision": REVISION,
        "test_count": len(rows),
        "source_groups": len({r["group_id"] for r in rows}),
        "family_counts": dict(Counter(r["family"] for r in rows)),
        "label_counts": dict(Counter(r["gold_id"] for r in rows)),
        "variant_counts": dict(Counter(r["variant"] for r in rows)),
        "max_state_chars": max(len(r["state"]) for r in rows),
        "max_question_chars": max(len(r["question"]) for r in rows),
        "normalized_sha256": digest,
        "source": spec["source"],
        "license": spec["license"],
        "annotation_status": spec["annotation_status"],
        "semantic_id_aliases": spec["semantic_id_aliases"],
        "scope": "Complete authored144; three families, 36 source groups with four related variants each.",
        "prompt_boundary": "Send caller_task, labelled state/question and options only. Gold/provenance/source IDs never enter input.",
    }
