"""Download pinned data, run serial evaluations, and verify published scores."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shlex
import time
from pathlib import Path

from ..assets import file_sha256, model_config, runtime_path, workspace_root
from ..request import render_prompt
from .archive import RESULTS, markdown, read_jsonl, verify
from .common import CONFIG_ROOT
from .data import COUNTS, DATASETS, load, request_for
from .metrics import summarize, validate


def canonical(value) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


SCORING_SOURCES = (
    "engine.py",
    "assets.py",
    "models.toml",
    "request.py",
    "benchmarks/cli.py",
    "benchmarks/metrics.py",
    "benchmarks/data.py",
    "benchmarks/common.py",
    "benchmarks/jev.py",
    "benchmarks/anyjev.py",
    "benchmarks/semif.py",
    "benchmarks/wanli.py",
)


def execution_fingerprint(weights, binary, *, package_root=None, config_root=CONFIG_ROOT):
    """Freeze the inference wrapper, assets, adapters and evaluator as well as weights."""
    package_root = package_root or Path(__file__).parents[1]
    return {
        "model_sha256": file_sha256(weights),
        "runtime_sha256": file_sha256(binary),
        "scoring_sources": {name: file_sha256(package_root / name) for name in SCORING_SOURCES},
        "dataset_manifests": {
            path.name: file_sha256(path) for path in sorted(config_root.glob("*.json"))
        },
    }


def request_hash(row):
    return hashlib.sha256(canonical(request_for(row))).hexdigest()


def _record(row, model, probabilities, duration):
    result = {k: row[k] for k in ("id", "group_id", "dataset", "primitive", "gold_id")}
    result.update(
        model=model,
        probabilities=probabilities,
        predicted_id=max(probabilities, key=probabilities.get),
        request_ms=duration,
        request_sha256=request_hash(row),
        option_ids=[o["id"] for o in row["options"]],
    )
    for key in ("score_values", "gold_score", "question_name", "workflow", "family"):
        if key in row:
            result[key] = row[key]
    validate(result)
    return result


def _write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def evaluate(
    rows,
    *,
    model,
    workspace,
    output,
    fingerprint,
    engine_factory=None,
    batch_size=16,
    batch_seconds=45.0,
    cooldown=30.0,
) -> dict:
    """Resume only an identical ordered protocol. Release the model between batches."""
    if engine_factory is None:
        from ..engine import GemmaJev

        engine_factory = GemmaJev
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "run.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("This benchmark output is already in use") from error
        plan = {
            "model": model,
            "ordered_inputs": [
                {
                    "id": r["id"],
                    "request_sha256": request_hash(r),
                    "prompt_sha256": hashlib.sha256(
                        render_prompt(request_for(r))[0].encode()
                    ).hexdigest(),
                    "gold_id": r["gold_id"],
                    "score_values": r.get("score_values"),
                    "gold_score": r.get("gold_score"),
                }
                for r in rows
            ],
            "fingerprint": fingerprint,
            "batch_size": batch_size,
            "batch_seconds": batch_seconds,
            "cooldown_seconds": cooldown,
        }
        config_path = output / "protocol.json"
        if config_path.exists():
            if json.loads(config_path.read_text()) != plan:
                raise ValueError(
                    "Existing run uses a different protocol; choose a new --output directory"
                )
        else:
            _write_json(config_path, plan)
        path = output / "predictions.jsonl"
        predictions = read_jsonl(path) if path.exists() else []
        if len(predictions) > len(rows):
            raise ValueError("Existing results exceed the planned sample")
        for row, record in zip(rows, predictions):
            validate(record)
            if (record["id"], record["model"], record["request_sha256"]) != (
                row["id"],
                model,
                request_hash(row),
            ):
                raise ValueError("Existing predictions are not an exact prefix of this run")
            if any(record.get(k) != row.get(k) for k in ("gold_id", "score_values", "gold_score")):
                raise ValueError("Existing prediction targets differ from the input")
        status = {"state": "running", "completed": len(predictions), "planned": len(rows)}
        _write_json(output / "status.json", status)
        try:
            if predictions and len(predictions) < len(rows):
                time.sleep(cooldown)
            while len(predictions) < len(rows):
                engine = engine_factory(model=model, workspace=workspace)
                try:
                    engine.load()
                    start = time.monotonic()
                    for _ in range(batch_size):
                        row = rows[len(predictions)]
                        began = time.monotonic()
                        probabilities = engine.decide(request_for(row))
                        record = _record(
                            row, model, probabilities, (time.monotonic() - began) * 1000
                        )
                        with path.open("a", encoding="utf-8") as stream:
                            stream.write(canonical(record).decode() + "\n")
                            stream.flush()
                            os.fsync(stream.fileno())
                        predictions.append(record)
                        status["completed"] = len(predictions)
                        _write_json(output / "status.json", status)
                        print(
                            f"{model}: {len(predictions)}/{len(rows)} {row['dataset']}", flush=True
                        )
                        if (
                            len(predictions) == len(rows)
                            or time.monotonic() - start >= batch_seconds
                        ):
                            break
                finally:
                    engine.close()
                if len(predictions) < len(rows):
                    time.sleep(cooldown)
            status["state"] = "complete"
        except BaseException:
            status["state"] = "interrupted"
            raise
        finally:
            _write_json(output / "status.json", status)
        result = summarize(predictions)
        _write_json(output / "metrics.json", result)
        return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="gemmajev benchmark")
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "run"):
        sub = commands.add_parser(command)
        sub.add_argument("--dataset", choices=("all", *DATASETS), default="all")
        sub.add_argument("--workspace", type=Path)
        if command == "run":
            sub.add_argument("--model", choices=("e4b", "12b"), required=True)
            sub.add_argument("--limit", type=int, help="First N examples per dataset; smoke only")
            sub.add_argument("--output", type=Path)
    for command in ("verify", "report"):
        sub = commands.add_parser(command)
        sub.add_argument("--results", type=Path, default=RESULTS)
    args = parser.parse_args(argv)
    if args.command in ("verify", "report"):
        summary = verify(args.results)
        if args.command == "verify":
            print(
                f"Verified {summary['archive']['rows']:,} predictions across 13 datasets and two models; no model or data download."
            )
        else:
            references = json.loads((args.results / summary["references"]["file"]).read_text())
            print(markdown(summary, references), end="")
        return 0
    root = workspace_root(args.workspace)
    try:
        rows = load(args.dataset, workspace=root, download=args.command == "prepare")
    except FileNotFoundError as error:
        if args.command == "prepare":
            raise
        command = f"uv run --extra data gemmajev benchmark prepare --dataset {args.dataset}"
        if args.workspace is not None:
            command += " --workspace " + shlex.quote(str(root))
        raise FileNotFoundError(f"The requested dataset is not prepared. Run: {command}") from error
    if args.command == "prepare":
        target = root / ".datasets/prepared"
        target.mkdir(exist_ok=True, parents=True)
        for dataset in COUNTS:
            selected = [r for r in rows if r["dataset"] == dataset]
            if selected:
                _write_json(
                    target / (dataset + ".json"),
                    {
                        "count": len(selected),
                        "request_hashes": {r["id"]: request_hash(r) for r in selected},
                    },
                )
        print(f"Prepared {len(rows):,} examples under {root / '.datasets'}")
        return 0
    if args.limit is not None:
        if args.limit < 1:
            parser.error("--limit must be positive")
        rows = [
            r
            for dataset in DATASETS
            for r in [x for x in rows if x["dataset"] == dataset][: args.limit]
        ]
    config = model_config(args.model, root)
    weight = Path(config["model_path"])
    binary = runtime_path(root)
    if not weight.is_file() or not binary.is_file():
        raise FileNotFoundError(f"Run gemmajev setup --model {args.model} before evaluation")
    fingerprint = execution_fingerprint(weight, binary)
    if fingerprint["model_sha256"] != config["model_sha256"]:
        raise ValueError("Model bytes do not match the fixed checkpoint")
    suffix = f"-smoke-{args.limit}" if args.limit else ""
    output = args.output or root / ".runtime/benchmarks" / args.model / (args.dataset + suffix)
    result = evaluate(
        rows, model=args.model, workspace=root, output=output, fingerprint=fingerprint
    )
    print(json.dumps(result, indent=2))
    return 0
