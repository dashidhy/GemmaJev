"""Pinned, CPU-only reconstruction of seven published Omar/Jev selection rules.

Historical upstream inputs were not hashed or dataset-revision-pinned. This
freezes a new reconstruction; it does not assert historical item identity.
``upstream_item`` and all gold/provenance fields are evaluation-only. The public
API receives only the fields allowlisted by ``build_suite_request``.
"""

from __future__ import annotations

import copy
import gzip
import hashlib
import io
import json
import random
import re
from collections import Counter
from pathlib import Path

from .common import CONFIG_ROOT, _download_source, _verified_bytes

TASKS = ("ag_news", "boolq", "mmlu", "logiqa", "anli_r3", "emotion", "essay_or_readability")
CONFIG_PATH = CONFIG_ROOT / "jev.json"
SUITE_ID = "jev-bench-v1"
UPSTREAM_REVISION = "0883fa781729ab571005253f6b288ec67f49dd29"
MAX_SOURCE_BYTES = 32 * 1024 * 1024
SCORE_IDS = (
    "not_helpful",
    "barely_helpful",
    "partially_helpful",
    "mostly_helpful",
    "fully_helpful",
)
QUERIES = {
    "ag_news": "Which news section best matches `article`?",
    "boolq": "According to `passage`, is the answer to `question` yes?",
    "mmlu": "Which provided answer correctly answers `question` in `subject`?",
    "logiqa": "Which provided answer correctly answers `question`, using `passage` alone?",
    "anli_r3": "Which relationship between `premise` and `hypothesis` is justified by `premise`?",
    "emotion": "Which emotion is mainly expressed in `message`?",
    "essay_or_readability": "How helpful is `response` to the user who wrote `prompt`?",
}
COMPARISON_SCOPE = (
    "Rules reconstructed from the pinned published builder. Upstream did not publish "
    "input snapshots/hashes or immutable HF dataset revisions; equality to its historical "
    "200 items is unverified. No selection uses model predictions."
)


def canonical_bytes(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _hash(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _text(value, name):
    _require(isinstance(value, str) and value.strip() and "\0" not in value, f"Invalid {name}")
    return value


def _load_spec():
    spec = json.loads(CONFIG_PATH.read_text())
    _require(
        spec.get("schema_version") == 1 and spec.get("suite_id") == SUITE_ID,
        "Unsupported suite manifest",
    )
    _require(tuple(spec["tasks"]) == TASKS, "Frozen task order differs")
    _require(spec["upstream"]["revision"] == UPSTREAM_REVISION, "Builder revision differs")
    sources = list(spec.get("repair_sources", []))
    for name, task in spec["tasks"].items():
        _require(task["count"] == 200 and task["seed"] == 7, f"Frozen selection differs: {name}")
        sources.extend(task["sources"])
    for source in sources:
        revision, path = source["revision"], source["path"]
        _require(
            re.fullmatch(r"[0-9a-f]{40}", revision) is not None, "Source revision is not pinned"
        )
        _require(
            re.fullmatch(r"[0-9a-f]{64}", source["sha256"]) is not None, "Missing source SHA-256"
        )
        _require(
            type(source["size_bytes"]) is int and 0 < source["size_bytes"] <= MAX_SOURCE_BYTES,
            "Source exceeds size limit",
        )
        _require(
            not Path(path).is_absolute() and ".." not in Path(path).parts, "Invalid source path"
        )
        expected = (
            f"https://raw.githubusercontent.com/{source['repository']}/{revision}/{path}"
            if source.get("provider") == "github"
            else f"https://huggingface.co/datasets/{source['repository']}/resolve/{revision}/{path}"
        )
        _require(source["url"] == expected, "Source URL differs from pinned HF file")
        cache_file = f"raw/{source['repository'].replace('/', '--')}/{revision}/{path}"
        _require(source["cache_file"] == cache_file, "Source cache path differs")
    return spec


def _artifact_bytes(cache_root, source, download):
    path = Path(cache_root) / SUITE_ID / source["cache_file"]
    if not path.exists() and not path.is_symlink():
        if not download:
            raise FileNotFoundError(
                f"Missing dataset file: {path}. Run: uv run --extra data gemmajev benchmark prepare --dataset all"
            )
        _download_source(source, path)
    return _verified_bytes(path, source)


def _read_source(cache_root, source, download):
    data = _artifact_bytes(cache_root, source, download)
    if source["format"] == "parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError as error:
            raise ImportError(
                "Parquet dataset loading requires the project's data extra"
            ) from error
        rows = pq.read_table(io.BytesIO(data)).to_pylist()
    elif source["format"] == "jsonl.gz":
        with gzip.GzipFile(fileobj=io.BytesIO(data)) as stream:
            expanded = stream.read(64 * 1024 * 1024 + 1)
        _require(len(expanded) <= 64 * 1024 * 1024, "Expanded dataset exceeds size limit")
        rows = [json.loads(line) for line in expanded.decode().split("\n") if line.strip()]
    else:
        raise ValueError("Unsupported source format")
    _require(
        len(rows) == source["row_count"] and all(isinstance(r, dict) for r in rows),
        "Source row count/schema differs",
    )
    return rows


def _source_rows(cache_root, spec, download, tasks=TASKS):
    result = {}
    for name in tasks:
        sources = spec["tasks"][name]["sources"]
        values = {s["dataset_config"]: _read_source(cache_root, s, download) for s in sources}
        result[name] = values if name == "mmlu" else next(iter(values.values()))
    return result


def _logiqa_sentence(text):
    """Transcription of the pinned HF loader's sentence formatting, without imports."""
    text = text.replace("\n", "")
    result = ""
    for sentence in text.split("."):
        if not sentence:
            continue
        result += sentence if not result else ("." if sentence[0].isnumeric() else ". ") + sentence
    result = result.replace("  ", " ").replace("\\'", "'")
    while result.endswith(" "):
        result = result[:-1]
    if re.match(r"^[A-Z][\w\s]+[?.!]$", result) is None:
        result += "."
    return result.replace("?.", "?").replace("!.", "!").replace("..", ".")


def logiqa_repair_options(raw_text: str, frozen_rows: list[dict]) -> dict[int, list[str]]:
    """Require full legacy-parser equality before correcting anchored option prefixes."""
    lines = raw_text.replace("\r\n", "\n").replace("\r", "\n").splitlines(keepends=True)
    formatted = [_logiqa_sentence(line) for line in lines]
    legacy, repaired = [], {}
    for index in range(len(formatted) // 8):
        block = formatted[index * 8 : (index + 1) * 8]
        options = block[4:8]
        legacy.append(
            {
                "context": block[2],
                "query": block[3],
                "options": [
                    option[3:] if option.startswith(tuple("ABCD")) else option for option in options
                ],
                "correct_option": "abcd".index(block[1].replace(".", "")),
            }
        )
        repaired[index] = [re.sub(r"^[A-D]\.\s*", "", option, count=1) for option in options]
    _require(
        legacy == frozen_rows, "LogiQA original Test.txt does not reproduce every frozen HF row"
    )
    return repaired


def load_logiqa_repairs(cache_root, spec, source_rows, *, download=False):
    sources = spec.get("repair_sources", [])
    _require(len(sources) == 1, "Expected the pinned LogiQA repair source")
    source = sources[0]
    _require(
        source["repository"] == "lgw863/LogiQA-dataset" and source["path"] == "Test.txt",
        "Unexpected repair source",
    )
    raw = _artifact_bytes(cache_root, source, download)
    return logiqa_repair_options(raw.decode("utf-8"), source_rows)


def _balanced(rng, pools, count, labels):
    selected = []
    for label in labels:
        pool = list(pools[label])
        rng.shuffle(pool)
        _require(len(pool) >= count, f"Insufficient selection pool: {label}")
        selected.extend((label, value) for value in pool[:count])
    rng.shuffle(selected)
    return selected


def _pages(rng, rows, labels, count, classify):
    offsets = list(range(0, len(rows), 100))
    rng.shuffle(offsets)
    pools, used, page_count = {label: [] for label in labels}, 0, 24
    while used < len(offsets):
        for offset in offsets[used : used + page_count]:
            for index in range(offset, min(offset + 100, len(rows))):
                label = classify(rows[index])
                if label in pools:
                    pools[label].append((index, rows[index]))
        used += page_count
        if all(len(pool) >= count for pool in pools.values()):
            break
        page_count = max(8, page_count // 2)
    return _balanced(rng, pools, count, labels)


def _choice(identifier, state, row, index, instructions, rng, *, subject=None):
    options = [str(value) for value in row["choices" if subject is not None else "options"]]
    gold = int(row["answer" if subject is not None else "correct_option"])
    _require(0 <= gold < len(options), "Invalid source option label")
    order = list(range(len(options)))
    rng.shuffle(order)
    criteria = {chr(65 + position): options[original] for position, original in enumerate(order)}
    native = {
        "id": identifier,
        "state": state,
        "questions": {"q": {"type": "choice", "instructions": instructions, "criteria": criteria}},
        "gold": {"q": chr(65 + order.index(gold))},
    }
    return native, {
        "source_index": index,
        "subject": subject,
        "choice_original_options": options,
        "choice_original_gold_index": gold,
        "choice_shuffle_original_indices": order,
    }


def reconstruct_task(name, data, task):
    """Faithfully reproduce RNG consumption, filtering, ordering, native items and gold."""
    rng = random.Random(task["seed"])
    selected = []
    if name in ("ag_news", "boolq"):
        labels = task["labels"]

        def classify(row):
            if name == "ag_news":
                return labels[row["label"]] if row["text"].strip() else None
            if (
                not row["question"].strip()
                or not row["passage"].strip()
                or len(row["passage"]) > 6000
            ):
                return None
            return "yes" if row["answer"] else "no"

        picked = _pages(rng, data, labels, task["per_class"], classify)
        for ordinal, (label, (index, row)) in enumerate(picked):
            state = (
                {"article": row["text"].strip()}
                if name == "ag_news"
                else {"passage": row["passage"].strip(), "question": row["question"].strip()}
            )
            selected.append(
                (
                    {
                        "id": f"{name}-{ordinal}",
                        "state": state,
                        "gold": label if name == "ag_news" else label == "yes",
                    },
                    {"source_index": index},
                )
            )
    elif name == "emotion":
        labels = task["labels"]
        pools = {label: [] for label in labels}
        for index, row in enumerate(data):
            if row["text"].strip():
                pools[labels[row["label"]]].append((index, row["text"].strip()))
        count = min(33, min(map(len, pools.values())))
        picked = _balanced(rng, pools, count, labels)
        used = {value[0] for _, value in picked}
        spare = [
            (label, value) for label in labels for value in pools[label] if value[0] not in used
        ]
        rng.shuffle(spare)
        picked.extend(spare[: 200 - len(picked)])
        rng.shuffle(picked)
        selected = [
            (
                {"id": f"emotion-{i}", "state": {"message": value[1]}, "gold": label},
                {"source_index": value[0]},
            )
            for i, (label, value) in enumerate(picked)
        ]
    elif name == "anli_r3":
        labels = ("entailment", "neutral", "contradiction")
        pools = {label: [] for label in labels}
        for index, row in enumerate(data):
            label = int(row["label"])
            if 0 <= label < 3 and row.get("premise") and row.get("hypothesis"):
                pools[labels[label]].append((index, row))
        for label, count in zip(labels, (67, 67, 66), strict=True):
            for index, row in rng.sample(pools[label], count):
                selected.append(
                    (
                        {
                            "state": {"premise": row["premise"], "hypothesis": row["hypothesis"]},
                            "gold": label,
                        },
                        {"source_index": index},
                    )
                )
        rng.shuffle(selected)
        for ordinal, (item, _) in enumerate(selected):
            item["id"] = f"anli-{ordinal}"
    elif name == "mmlu":
        for subject in task["subjects"]:
            pool = [
                (i, row)
                for i, row in enumerate(data[subject])
                if row["question"] and len(row["choices"]) >= 2
            ]
            for ordinal, (index, row) in enumerate(rng.sample(pool, 8)):
                selected.append(
                    _choice(
                        f"mmlu-{subject}-{ordinal}",
                        {"subject": subject.replace("_", " "), "question": row["question"]},
                        row,
                        index,
                        task["instructions"],
                        rng,
                        subject=subject,
                    )
                )
        rng.shuffle(selected)
    elif name == "logiqa":
        pool = [
            (i, row)
            for i, row in enumerate(data)
            if row.get("query")
            and len(row.get("options", [])) >= 2
            and 0 <= int(row["correct_option"]) < len(row["options"])
        ]
        for ordinal, (index, row) in enumerate(rng.sample(pool, 200)):
            selected.append(
                _choice(
                    f"logiqa-{ordinal}",
                    {"passage": row["context"], "question": row["query"]},
                    row,
                    index,
                    task["instructions"],
                    rng,
                )
            )
    elif name == "essay_or_readability":
        pools = {label: [] for label in range(5)}
        for index, row in enumerate(data[:1000]):
            helpfulness = row.get("helpfulness")
            if helpfulness is None or row.get("response") is None or row.get("prompt") is None:
                continue
            if len(json.dumps(row["prompt"])) + len(json.dumps(row["response"])) > 40000:
                continue
            pools[int(helpfulness)].append((index, row))
        for pool in pools.values():
            rng.shuffle(pool)
        need, picked = 200, []
        order = sorted(pools, key=lambda label: len(pools[label]))
        for remaining, label in zip(range(len(order), 0, -1), order, strict=True):
            quota = min(len(pools[label]), -(-need // remaining))
            picked.extend(pools[label][:quota])
            need -= quota
        _require(len(picked) == 200, "Insufficient HelpSteer2 candidates")
        rng.shuffle(picked)
        selected = [
            (
                {
                    "id": f"essay_or_readability-{i}",
                    "state": {"prompt": row["prompt"], "response": row["response"]},
                    "gold": int(row["helpfulness"]),
                },
                {"source_index": index, "source_helpfulness": row["helpfulness"]},
            )
            for i, (index, row) in enumerate(picked)
        ]
    else:
        raise ValueError(f"Unknown dataset {name}")
    _require(len(selected) == 200, f"Reconstruction count differs: {name}")
    return selected


def _answer_ids(descriptions):
    seen, result = Counter(), []
    stop = {
        "a",
        "an",
        "the",
        "of",
        "to",
        "in",
        "on",
        "and",
        "or",
        "is",
        "are",
        "it",
        "that",
        "this",
        "for",
        "with",
    }
    for description in descriptions:
        normalized = description.strip().casefold().rstrip(".")
        if re.fullmatch(r"[-+]?\d+(?:\.\d+)?(?:/\d+)?%?", normalized):
            numeric = normalized.replace("-", "negative_").replace("+", "positive_")
            numeric = (
                numeric.replace(".", "_point_").replace("/", "_over_").replace("%", "_percent")
            )
            identifier = "numeric_value_" + numeric[:22]
        else:
            words = re.findall(r"[a-z0-9]+", normalized)
            meaningful = [word for word in words if word not in stop] or words
            identifier = "_".join(meaningful[:4])[:36].strip("_") or "expression"
            if len(identifier) < 2 or identifier.isdigit():
                identifier = "answer_" + identifier
        base = identifier
        if base in seen:
            suffix = hashlib.sha256(description.encode()).hexdigest()[:4]
            identifier = base[:31] + "_" + suffix
            if identifier in seen:
                identifier = base[:28] + f"_copy{seen[base] + 1}"
        seen[base] += 1
        if identifier != base:
            seen[identifier] += 1
        result.append(identifier)
    return result


def option_reference_audit(rows):
    patterns = {
        "above_below_reference": r"\b(?:all|none|both|any)\s+of\s+(?:the\s+)?(?:above|below|following|these|them)\b",
        "explicit_option_letter": r"\b(?:option|choice|answer)s?\s+[A-D]\b",
        "combined_letters": r"\b[A-D]\s+(?:and|or)\s+[A-D]\b",
    }
    result = []
    for row in rows:
        if row["dataset"] not in ("mmlu", "logiqa"):
            continue
        descriptions = [o["description"] for o in row["options"]]
        for index, description in enumerate(descriptions):
            matches = [
                name
                for name, pattern in patterns.items()
                if re.search(pattern, description, re.IGNORECASE)
            ]
            if descriptions.count(description) > 1:
                matches.append("duplicate_answer_text")
            if matches:
                result.append(
                    {
                        "id": row["id"],
                        "source_id": row["source_id"],
                        "shuffled_index": index,
                        "original_index": row["choice_shuffle_original_indices"][index],
                        "description": description,
                        "flags": matches,
                    }
                )
    return result


def normalize_task(name, selected, task, *, logiqa_options=None, repair_source=None):
    rows = []
    for native, provenance in selected:
        question = native["questions"]["q"] if "questions" in native else task["upstream_question"]
        primitive, criteria = question["type"], question["criteria"]
        gold = native["gold"]["q"] if "questions" in native else native["gold"]
        if primitive == "score":
            options = [
                {"id": key, "description": value}
                for key, value in zip(SCORE_IDS, criteria, strict=True)
            ]
            gold_id = SCORE_IDS[gold]
        elif primitive == "noul":
            options = [
                {"id": "yes", "description": criteria["true"]},
                {"id": "no", "description": criteria["false"]},
            ]
            gold_id = "yes" if gold else "no"
        elif "questions" in native:
            descriptions = list(criteria.values())
            repairs = []
            for position, (description, original_index) in enumerate(
                zip(descriptions, provenance["choice_shuffle_original_indices"], strict=True)
            ):
                changed, kind, reason = description, None, None
                originals = provenance["choice_original_options"]
                if name == "mmlu":
                    if original_index == len(originals) - 1 and re.fullmatch(
                        r"\s*(all|none) of the above\.?\s*", description, re.IGNORECASE
                    ):
                        mode = description.strip().split()[0].casefold()
                        changed = mode.capitalize() + " of the other answer choices."
                        kind, reason = (
                            "positional_" + mode + "_above",
                            "Expand the original final option without depending on its shuffled position.",
                        )
                    elif re.fullmatch(r"\s*both A and B\.?\s*", description, re.IGNORECASE):
                        changed = f'Both "{originals[0]}" and "{originals[1]}".'
                        kind, reason = (
                            "explicit_option_reference",
                            "Expand original options0/1; current internal answer letters have a different order.",
                        )
                elif name == "logiqa" and logiqa_options is not None:
                    changed = logiqa_options[provenance["source_index"]][original_index]
                    if changed != description:
                        kind, reason = (
                            "logiqa_option_prefix",
                            "Replace legacy fixed[3:] slicing with anchored removal of the actual A–D prefix, verified against original Test.txt.",
                        )
                if changed != description:
                    source = (
                        repair_source
                        if name == "logiqa"
                        else next(
                            s
                            for s in task["sources"]
                            if s["dataset_config"] == provenance["subject"]
                        )
                    )
                    source_reference = {
                        key: source[key] for key in ("repository", "revision", "path", "sha256")
                    }
                    source_reference["source_index"] = provenance["source_index"]
                    if name == "logiqa":
                        source_reference["source_line"] = (
                            provenance["source_index"] * 8 + 5 + original_index
                        )
                    repairs.append(
                        {
                            "type": kind,
                            "field": "options",
                            "original_index": original_index,
                            "shuffled_index": position,
                            "before": description,
                            "after": changed,
                            "before_sha256": hashlib.sha256(description.encode()).hexdigest(),
                            "after_sha256": hashlib.sha256(changed.encode()).hexdigest(),
                            "reason": reason,
                            "source": source_reference,
                        }
                    )
                    descriptions[position] = changed
            identifiers = _answer_ids(descriptions)
            options = [
                {"id": key, "description": value}
                for key, value in zip(identifiers, descriptions, strict=True)
            ]
            gold_id = identifiers[list(criteria).index(gold)]
        else:
            options = [{"id": key, "description": value} for key, value in criteria.items()]
            gold_id = gold
        state = native["state"]
        group_input = state.get("prompt", state.get("premise", state.get("passage", state)))
        subject = provenance.get("subject")
        source = next(
            s for s in task["sources"] if subject is None or s["dataset_config"] == subject
        )
        source_id = f"{source['dataset_config']}/{source['split']}/{provenance['source_index']}"
        api_instructions = question["instructions"]
        if name in ("mmlu", "logiqa"):
            api_instructions = api_instructions.replace(
                ", copied verbatim from the exam", ""
            ).replace(", copied verbatim", "")
        if name == "boolq":
            statement = "The answer to `question` is yes, according to `passage`."
            _require(statement in api_instructions, "BoolQ native proposition changed")
            api_instructions = api_instructions.replace(
                statement,
                "Decide whether the answer to `question` is yes according to `passage`.",
            )
        row = {
            "id": f"{SUITE_ID}:{name}:{native['id']}",
            "upstream_id": native["id"],
            "group_id": f"{SUITE_ID}:{name}:group:{_hash(group_input)[:24]}",
            "dataset": name,
            "task": name,
            "family": f"omar_{name}",
            "primitive": primitive,
            "partition": "test",
            "variant": "original",
            "source_id": source_id,
            "source_split": source["split"],
            "source_index": provenance["source_index"],
            "gold_id": gold_id,
            "gold_index": next(i for i, o in enumerate(options) if o["id"] == gold_id),
            "options": options,
            "state": json.dumps(state, ensure_ascii=False, indent=2),
            "question": QUERIES[name],
            "state_label": "Evidence",
            "query_label": "Question",
            "caller_task": api_instructions
            + "\n\nThe named source fields are explicitly provided in Evidence. Answer Question using the supplied options.",
            "upstream_item": copy.deepcopy(native),
        }
        if name == "boolq":
            row["api_adaptations"] = ["noul_statement_to_question"]
        if "questions" in native and repairs:
            row["input_repairs"] = repairs
        for field in (
            "subject",
            "choice_original_options",
            "choice_original_gold_index",
            "choice_shuffle_original_indices",
        ):
            if provenance.get(field) is not None:
                row[field] = provenance[field]
        if primitive == "score":
            row["score_values"] = dict(zip(SCORE_IDS, range(5), strict=True))
            _require(
                provenance["source_helpfulness"] == gold,
                "Fractional helpfulness differs from upstream integer gold",
            )
            row["gold_score"] = provenance["source_helpfulness"]
        for field in re.findall(r"`([^`]+)`", question["instructions"] + row["question"]):
            _require(field in state, f"Task references an absent source field: {name}/{field}")
        _require(len(options) == len({o["id"] for o in options}), "Semantic option IDs collide")
        for option in options:
            _text(option["description"], "option description")
        _text(row["state"], "state")
        rows.append(row)
    return rows


def load_suite(cache_root: Path, *, download: bool = False, tasks=TASKS) -> tuple[list[dict], dict]:
    """Load all1400 frozen rows, verifying source bytes and normalized selection hashes."""
    spec = _load_spec()
    if not tasks or not set(tasks) <= set(TASKS):
        raise ValueError("Unknown or empty task selection")
    data = _source_rows(cache_root, spec, download, tasks)
    logiqa_options = (
        load_logiqa_repairs(cache_root, spec, data["logiqa"], download=download)
        if "logiqa" in tasks
        else None
    )
    rows, datasets = [], {}
    for name in tasks:
        task = spec["tasks"][name]
        selected = reconstruct_task(name, data[name], task)
        normalized = normalize_task(
            name,
            selected,
            task,
            logiqa_options=logiqa_options,
            repair_source=spec["repair_sources"][0],
        )
        frozen = task["selection"]
        _require(
            [r["source_id"] for r in normalized] == frozen["source_ids"],
            f"Selected source order differs: {name}",
        )
        _require(
            _hash(normalized) == frozen["normalized_sha256"],
            f"Normalized frozen inputs differ: {name}",
        )
        actual_repairs = [
            {
                "id": r["id"],
                "repairs": [
                    {k: v for k, v in repair.items() if k not in ("before", "after")}
                    for repair in r["input_repairs"]
                ],
            }
            for r in normalized
            if r.get("input_repairs")
        ]
        _require(
            actual_repairs == task.get("approved_input_repairs", []),
            f"Frozen input repair map differs: {name}",
        )
        groups = Counter(r["group_id"] for r in normalized)
        datasets[name] = {
            "count": len(normalized),
            "group_count": len(groups),
            "max_group_size": max(groups.values()),
            "label_counts": dict(Counter(r["gold_id"] for r in normalized)),
            "selected_ids": [r["id"] for r in normalized],
            "normalized_sha256": _hash(normalized),
            "upstream_question": task.get("upstream_question"),
            "upstream_items_sha256": _hash([native for native, _ in selected]),
            "choice_reference_flagged_ids": sorted(
                {r["id"] for r in option_reference_audit(normalized)}
            ),
            "repaired_rows": len(actual_repairs),
        }
        rows.extend(normalized)
    _require(len({r["id"] for r in rows}) == len(rows), "Duplicate suite IDs")
    return rows, {
        "schema_version": 1,
        "suite_id": SUITE_ID,
        "manifest": spec,
        "datasets": datasets,
        "references": spec["references"],
        "comparison_scope": COMPARISON_SCOPE,
    }
