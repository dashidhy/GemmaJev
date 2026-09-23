"""Pinned AnyJev task reconstruction.

Only state/question/options and the caller-owned labels/instructions are input.
Gold distributions, factors, calibration IDs and provenance never form a prompt.
No upstream Python is executed. Parquet is needed only to build the verified
normalized cache; subsequent offline reads require only the standard library.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import tempfile
from collections import Counter
from pathlib import Path

from .common import CONFIG_ROOT, _download_source, _verified_bytes

CONFIG_PATH = CONFIG_ROOT / "anyjev.json"
REPOSITORY = "nokia-applied-research/AnyJev"
REVISION = "78ca550268eefa4b4ed7b34f377b475597e4f51b"
TASKS = ("banking20", "newsgroups", "injection", "typed_decisions")
MAX_BYTES = 16 * 1024 * 1024
ALIGNMENT = (
    "Deterministic published loader/seed reconstructed on pinned datasets last modified before "
    "the upstream runs. Upstream did not publish evaluated item IDs or per-item predictions; "
    "aggregate references cannot support paired cross-system intervals or error intersections."
)


def _json_bytes(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def _encoded(rows: list[dict]) -> bytes:
    return b"".join(_json_bytes(row) + b"\n" for row in rows)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _text(value, name):
    if not isinstance(value, str) or not value.strip() or "\0" in value:
        raise ValueError(f"AnyJev {name} must be nonempty text without NUL")
    return value


def _artifact(record):
    if not re.fullmatch(r"[0-9a-f]{64}", record.get("sha256", "")):
        raise ValueError("AnyJev artifacts need a full SHA-256")
    if type(record.get("size_bytes")) is not int or not 0 < record["size_bytes"] <= MAX_BYTES:
        raise ValueError("AnyJev artifact exceeds its size bound")
    path = Path(record["cache_file"])
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("AnyJev cache path must stay under its dataset directory")


def _load_spec():
    spec = json.loads(CONFIG_PATH.read_text())
    if spec.get("schema_version") != 1 or spec["upstream"]["repository"] != REPOSITORY:
        raise ValueError("Unexpected AnyJev manifest")
    if spec["upstream"]["revision"] != REVISION or set(spec["tasks"]) != set(TASKS):
        raise ValueError("AnyJev revision/tasks changed")
    for source in spec["sources"].values():
        _artifact(source)
        if not re.fullmatch(r"[0-9a-f]{40}", source["revision"]):
            raise ValueError("Dataset revision must be immutable")
        expected = (
            f"https://huggingface.co/datasets/{source['repository']}/resolve/"
            f"{source['revision']}/{source['path']}"
        )
        if source["url"] != expected:
            raise ValueError("Dataset URL differs from its pinned source")
    for reference in spec["references"]:
        _artifact(reference)
        if reference["url"] != (
            f"https://raw.githubusercontent.com/{REPOSITORY}/{REVISION}/{reference['path']}"
        ):
            raise ValueError("Reference URL differs from the pinned AnyJev source")
    for name, task in spec["tasks"].items():
        _artifact(task["normalized"])
        if task["test_count"] != (2000 if name == "typed_decisions" else 300):
            raise ValueError("AnyJev must retain each complete published test panel")
        if name != "typed_decisions" and (task["seed"], task["calibration_count"]) != (0, 200):
            raise ValueError("AnyJev sampling/calibration recipe changed")
    return spec


def _publish(data, path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".anyjev-", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(data)
    try:
        _verified_bytes(temporary, record)
        try:
            os.link(temporary, path)
        except FileExistsError:
            _verified_bytes(path, record)
    finally:
        temporary.unlink(missing_ok=True)


def _raw_rows(root, record, download):
    data = _obtain(root, record, download)
    if record["format"] == "jsonl":
        rows = [json.loads(line) for line in data.decode().split("\n") if line.strip()]
    elif record["format"] == "parquet":
        try:
            from pyarrow import parquet
        except ImportError as error:
            raise RuntimeError(
                "Preparing AnyJev Parquet needs pyarrow; use uv run --extra data. "
                "Verified normalized caches can be loaded without pyarrow."
            ) from error
        rows = parquet.read_table(root / record["cache_file"]).to_pylist()
    else:
        raise ValueError("Unknown source format")
    if len(rows) != record["row_count"]:
        raise ValueError("Source row count differs from the pinned manifest")
    return rows


def _question_key(primitive, question, descriptions, *, score=False, legacy=False):
    payload = {
        "kind": primitive,
        "text": question,
        "options": descriptions,
        "scale": [0.0, float(len(descriptions) - 1)] if score else [0.0, 1.0],
        "centers": [float(i) for i in range(len(descriptions))] if score else None,
    }
    if legacy:
        payload.pop("centers")  # Published 2026-09-20 task hashes predate this field.
    # Matches AnyJev Question.key, including its ordinary JSON spaces.
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()[:16]


def _base(name, spec, identifier, group, state, question, options, gold_index, primitive):
    return {
        "id": identifier,
        "group_id": group,
        "dataset": name,
        "family": f"anyjev_{name}",
        "task": "typed" if name == "typed_decisions" else "classification",
        "partition": "test",
        "variant": "original",
        "primitive": primitive,
        "state": _text(state, "state"),
        "question": _text(question, "question"),
        "options": [dict(option) for option in options],
        "gold_index": gold_index,
        "gold_id": options[gold_index]["id"],
        "caller_task": spec["caller_task"],
        "state_label": spec["state_label"],
        "query_label": "Question",
    }


def _classification(name, raw, spec):
    pool = []
    options = spec["options"]
    if name == "banking20":
        train = raw["banking_train"]
        names = {int(row["label"]): row["label_text"] for row in train}
        top = sorted(label for label, _ in Counter(int(r["label"]) for r in train).most_common(20))
        if [names[label] for label in top] != spec["upstream_option_ids"]:
            raise ValueError("Banking top-20 label order changed")
        remap = {label: i for i, label in enumerate(top)}
        for i, row in enumerate(raw["banking_test"]):
            if int(row["label"]) in remap:
                pool.append((f"banking77:test:{i}", row["text"], remap[int(row["label"])]))
    elif name == "newsgroups":
        remap = {label: i for i, label in enumerate(spec["upstream_option_ids"])}
        for i, row in enumerate(raw["newsgroups_test"]):
            text = (row["text"] or "").strip()
            if len(text) >= 40:
                pool.append((f"newsgroups:test:{i}", text[:2000], remap[row["label_text"]]))
    elif name == "injection":
        for split in ("train", "test"):
            for i, row in enumerate(raw[f"injection_{split}"]):
                if type(row["label"]) is not int or row["label"] not in (0, 1):
                    raise ValueError("Invalid upstream injection label")
                pool.append((f"injection:{split}:{i}", row["text"], 0 if row["label"] else 1))
    else:
        raise ValueError("Unknown classification recipe")
    if len(pool) != spec["pool_count"]:
        raise ValueError("Eligible pool count changed")
    indices = list(range(len(pool)))
    random.Random(spec["seed"]).shuffle(indices)
    n = spec["test_count"]
    selected = [pool[i] for i in indices[:n]]
    calibration = [pool[i] for i in indices[n : n + spec["calibration_count"]]]
    if len(calibration) != spec["calibration_count"]:
        raise ValueError("Incomplete calibration population")
    qhash = _question_key(
        spec["primitive"], spec["question"], [o["description"] for o in options], legacy=True
    )
    if qhash != spec["upstream_question_key"]:
        raise ValueError("Question/option descriptions differ from published AnyJev hash")
    rows = []
    for source_id, text, gold in selected:
        identifier = f"anyjev:{name}:{source_id}"
        group = f"anyjev:{name}:text:{_sha(text.encode())}"
        row = _base(
            name, spec, identifier, group, text, spec["question"], options, gold, spec["primitive"]
        )
        row.update(source_id=source_id, question_key=qhash)
        rows.append(row)
    test_texts = {r[1] for r in selected}
    return rows, {
        "eligible_pool_count": len(pool),
        "seed": spec["seed"],
        "selected_source_ids": [r[0] for r in selected],
        "calibration_source_ids": [r[0] for r in calibration],
        "calibration_inference": False,
        "test_calibration_identical_texts": len(test_texts & {r[1] for r in calibration}),
        "upstream_question_key": qhash,
        "upstream_question_key_schema": "pre-centers field; matches published 2026-09-20 hash",
    }


def _render_state(state):
    if isinstance(state, str):
        return state
    if (
        isinstance(state, list)
        and state
        and all(isinstance(m, dict) and "role" in m and "content" in m for m in state)
    ):
        return "\n".join(f"{m['role']}: {m['content']}" for m in state)
    return json.dumps(state, indent=2, ensure_ascii=False)


def _typed(raw, spec):
    rows, cases, question_keys = [], set(), set()
    for case in raw["typed_test"]:
        case_id = _text(case["id"], "case ID")
        if case_id in cases or case["split"] != "test":
            raise ValueError("Duplicate case ID or non-test typed case")
        cases.add(case_id)
        state = _render_state(json.loads(case["state"]))
        questions, golds = json.loads(case["questions"]), json.loads(case["gold"])
        if len(questions) != 5 or questions.keys() != golds.keys():
            raise ValueError("Typed cases must retain all five questions and labels")
        for qname, question in questions.items():
            primitive, criteria = question["type"], question.get("criteria")
            text = question["instructions"]
            if primitive == "choice":
                keys = list(criteria) if isinstance(criteria, dict) else [str(c) for c in criteria]
                descriptions = (
                    [f"{k}: {criteria[k]}" for k in keys] if isinstance(criteria, dict) else keys
                )
                semantic = keys
            elif primitive == "noul":
                keys, semantic, descriptions = ["true", "false"], ["yes", "no"], ["Yes", "No"]
                text = "Is the following statement true? " + text
                if isinstance(criteria, dict):
                    text += f"\nYes means: {criteria.get('true', '')}\nNo means: {criteria.get('false', '')}"
            elif primitive == "score":
                descriptions = (
                    list(criteria.values())
                    if isinstance(criteria, dict)
                    else [str(c) for c in criteria]
                )
                keys = [str(i) for i in range(len(descriptions))]
                semantic = spec["score_ids"][qname]
                if len(semantic) != len(keys):
                    raise ValueError("Semantic score IDs must retain every original rubric level")
            else:
                raise ValueError("Unknown typed primitive")
            options = [
                {"id": k, "description": d} for k, d in zip(semantic, descriptions, strict=True)
            ]
            gold = golds[qname]
            if set(gold["probabilities"]) != set(keys):
                raise ValueError("Gold probabilities differ from the question options")
            probabilities = [float(gold["probabilities"][k]) for k in keys]
            if (
                any(not math.isfinite(v) or not 0 <= v <= 1 for v in probabilities)
                or abs(sum(probabilities) - 1) > 1e-5
            ):
                raise ValueError("Invalid upstream soft gold")
            gi = keys.index(str(gold["label"]))
            row = _base(
                "typed_decisions",
                spec,
                f"typed:{case_id}:{qname}",
                case_id,
                state,
                text,
                options,
                gi,
                primitive,
            )
            qhash = _question_key(primitive, text, descriptions, score=primitive == "score")
            question_keys.add(qhash)
            row.update(
                workflow=case["workflow"],
                question_name=qname,
                question_key=qhash,
                source_id=case_id,
                upstream_option_ids=keys,
                gold_probs=dict(zip(semantic, probabilities, strict=True)),
            )
            if primitive == "score":
                value = float(gold["score"])
                if not math.isfinite(value) or not 0 <= value <= len(keys) - 1:
                    raise ValueError("Invalid score target")
                row.update(
                    gold_score=value, score_values={k: float(i) for i, k in enumerate(semantic)}
                )
            rows.append(row)
    return rows, {
        "case_count": len(cases),
        "question_count": len(question_keys),
        "calibration_inference": False,
        "upstream_L1_calibration": "First 50 train cases per workflow (200 cases/1000 decisions); not needed or loaded for this run.",
    }


def _validate_rows(rows, spec):
    if len(rows) != spec["test_count"] or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Normalized test population changed")
    for row in rows:
        for key in (
            "id",
            "group_id",
            "state",
            "question",
            "caller_task",
            "state_label",
            "query_label",
        ):
            _text(row[key], key)
        opts = row["options"]
        ids = [o["id"] for o in opts]
        if not 2 <= len(ids) <= 26 or len(set(ids)) != len(ids):
            raise ValueError("Invalid normalized semantic options")
        for option in opts:
            if set(option) != {"id", "description"}:
                raise ValueError("Option contains evaluation metadata")
            _text(option["id"], "option ID")
            _text(option["description"], "description")
            if len(option["id"]) == 1 or option["id"].isdigit():
                raise ValueError("Semantic option IDs must not be letters or bare numbers")
        if (
            row["partition"] != "test"
            or type(row["gold_index"]) is not int
            or not 0 <= row["gold_index"] < len(ids)
            or ids[row["gold_index"]] != row["gold_id"]
        ):
            raise ValueError("Normalized gold/partition mismatch")
        if any(k in row for k in ("factors", "label_agreement", "gold", "rationale")):
            raise ValueError("Unneeded label-generation metadata in normalized rows")


def _normalize(name, raw, spec):
    rows, selection = (
        _typed(raw, spec) if name == "typed_decisions" else _classification(name, raw, spec)
    )
    _validate_rows(rows, spec)
    metadata = {
        "dataset": name,
        "test_count": len(rows),
        "source_groups": len({r["group_id"] for r in rows}),
        "label_counts": dict(Counter(r["gold_id"] for r in rows)),
        "primitive_counts": dict(Counter(r["primitive"] for r in rows)),
        "workflow_counts": dict(Counter(r.get("workflow", name) for r in rows)),
        "max_state_chars": max(len(r["state"]) for r in rows),
        "max_question_chars": max(len(r["question"]) for r in rows),
        "soft_gold_tied_max_rows": sum(
            sum(abs(p - max(r["gold_probs"].values())) < 1e-12 for p in r["gold_probs"].values())
            > 1
            for r in rows
            if "gold_probs" in r
        ),
        "selection": selection,
    }
    return rows, metadata


def load_dataset(cache_root: Path, task: str, download: bool = False) -> tuple[list[dict], dict]:
    """Return a complete test panel; never score or return calibration examples."""
    if task not in TASKS:
        raise ValueError(f"task must be one of {TASKS}")
    spec = _load_spec()
    task_spec = spec["tasks"][task]
    root = Path(cache_root) / "anyjev" / REVISION
    cached = root / task_spec["normalized"]["cache_file"]
    if cached.exists() or cached.is_symlink():
        data = _verified_bytes(cached, task_spec["normalized"])
        rows = [json.loads(line) for line in data.decode().split("\n") if line.strip()]
        _validate_rows(rows, task_spec)
    else:
        sources = {
            key: _raw_rows(root, spec["sources"][key], download) for key in task_spec["sources"]
        }
        rows, selection = _normalize(task, sources, task_spec)
        if selection != task_spec["expected_metadata"]:
            raise ValueError("Rebuilt selection differs from the frozen metadata")
        _publish(_encoded(rows), cached, task_spec["normalized"])
    manifest = {
        **task_spec["expected_metadata"],
        "anyjev_revision": REVISION,
        "sources": {key: spec["sources"][key] for key in task_spec["sources"]},
        "normalized_snapshot": task_spec["normalized"],
        "license": task_spec["license"],
        "annotation_status": task_spec["annotation_status"],
        "transformations": task_spec["transformations"],
        "reference_alignment": ALIGNMENT,
        "prompt_boundary": "Only caller_task, labelled state, labelled question and options may be sent. Gold/score targets/source IDs are evaluation-only.",
    }
    return rows, manifest


def _obtain(root, record, download):
    path = root / record["cache_file"]
    if not path.exists():
        if not download:
            raise FileNotFoundError(
                f"Missing dataset file: {path}. Run: uv run --extra data gemmajev benchmark prepare --dataset all"
            )
        _download_source(record, path)
    return _verified_bytes(path, record)
