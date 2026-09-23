"""Commands for local setup, decision scoring, the playground and benchmarks."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

from .assets import MODELS, build_runtime, check_platform, prepare_model, workspace_root


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def read_request(path: str) -> dict:
    text = sys.stdin.read() if path == "-" else Path(path).read_text(encoding="utf-8")
    return json.loads(text, object_pairs_hook=_unique_object)


def _run(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "benchmark":
        from .benchmarks import main as benchmark_main

        return benchmark_main(argv[1:])
    parser = argparse.ArgumentParser(description="Local decision probabilities with Gemma 4.")
    commands = parser.add_subparsers(dest="command", required=True)
    setup = commands.add_parser("setup", help="Build the runtime and download a verified model")
    setup.add_argument("--model", choices=(*MODELS, "all"), default="12b")
    setup.add_argument("--workspace", type=Path)
    commands.add_parser("build", help="Download and compile the pinned text runtime")
    decide = commands.add_parser("decide", help="Score an API JSON request; '-' reads stdin")
    decide.add_argument("request", help="Request JSON file, or -")
    decide.add_argument("--model", choices=MODELS, default="12b")
    decide.add_argument("--workspace", type=Path)
    demo = commands.add_parser("demo", help="Open the local interactive Playground")
    demo.add_argument("--model", choices=MODELS, default="12b")
    demo.add_argument("--workspace", type=Path)
    demo.add_argument("--port", type=int, default=7860)
    demo.add_argument("--no-browser", action="store_true")
    commands.add_parser("benchmark", help="Prepare, run, verify and report the text benchmarks")
    args = parser.parse_args(argv)
    if args.command == "build":
        print(build_runtime())
    elif args.command == "setup":
        hardware = check_platform()
        print(f"macOS · {hardware['chip']} · {hardware['memory_bytes'] / 2**30:.0f} GiB")
        root = workspace_root(args.workspace)
        build_runtime(root)
        for model in MODELS if args.model == "all" else (args.model,):
            print(f"Preparing {model}…", flush=True)
            print(prepare_model(model, root))
        command = [
            "uv",
            "run",
            "gemmajev",
            "demo",
            "--model",
            "12b" if args.model == "all" else args.model,
        ]
        if args.workspace is not None:
            command += ["--workspace", str(root)]
        print(f"Ready. Run: {shlex.join(command)}")
    elif args.command == "decide":
        from .engine import GemmaJev
        from .request import validate_request

        request = validate_request(read_request(args.request))
        with GemmaJev(model=args.model, workspace=args.workspace) as model:
            probabilities = model.decide(request)
        print(json.dumps(probabilities, ensure_ascii=False, indent=2, allow_nan=False))
    elif args.command == "demo":
        from .demo import launch

        launch(
            model=args.model,
            workspace=args.workspace,
            port=args.port,
            open_browser=not args.no_browser,
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        return _run(argv)
    except (ValueError, RuntimeError, OSError, subprocess.CalledProcessError) as error:
        print(f"gemmajev: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
