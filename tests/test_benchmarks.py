import gzip
import json

import pytest

from gemmajev.benchmarks.archive import RESULTS, markdown, read_jsonl, verify
from gemmajev.benchmarks.data import COUNTS, request_for
from gemmajev.benchmarks.metrics import auroc, spearman, summarize, validate
from gemmajev.request import render_prompt


def prediction(**changes):
    row = {
        "model": "e4b",
        "dataset": "wanli",
        "id": "one",
        "group_id": "group",
        "primitive": "choice",
        "gold_id": "supported",
        "predicted_id": "supported",
        "request_ms": 1.0,
        "probabilities": {"supported": 0.75, "contradicted": 0.25},
    }
    return {**row, **changes}


def test_release_archive_recomputes_every_retained_metric():
    result = verify()
    assert result["counts"]["predictions"] == 9176
    assert result["results"]["12b"]["mmlu"]["accuracy"] == 0.73
    assert result["results"]["12b"]["authored144"]["correct"] == 138
    assert result["results"]["e4b"]["typed_decisions"]["score_n"] == 800
    assert result["results"]["12b"]["essay_or_readability"]["spearman"] > 0.56
    assert set(result["results"]) == {"e4b", "12b"}
    assert sum(COUNTS.values()) == 4588


def test_archive_tampering_is_rejected(tmp_path):
    for name in ("summary.json", "predictions.jsonl.gz", "references.json"):
        (tmp_path / name).write_bytes((RESULTS / name).read_bytes())
    (tmp_path / "references.json").write_text("[]")
    with pytest.raises(ValueError, match="checksum"):
        verify(tmp_path)


@pytest.mark.parametrize(
    "change",
    [
        {"probabilities": {"supported": float("nan"), "contradicted": 0.25}},
        {"probabilities": {"supported": 0.75, "contradicted": 0.5}},
        {"predicted_id": "contradicted"},
        {"gold_id": "missing"},
        {"option_ids": ["different", "ids"]},
        {"request_ms": -1},
    ],
)
def test_invalid_scores_rejected(change):
    with pytest.raises(ValueError):
        validate(prediction(**change))


def test_rank_and_tie_conventions():
    assert spearman([1, 2, 2, 4], [1, 2, 2, 4]) == pytest.approx(1.0)
    assert spearman([1, 1], [0, 1]) is None
    assert auroc([0.5, 0.5], [True, False]) == 0.5
    assert auroc([1, 0], [True, False]) == 1.0


def test_score_mae_uses_expectation_not_argmax():
    row = prediction(
        dataset="typed_decisions",
        primitive="score",
        score_values={"supported": 0.0, "contradicted": 4.0},
        gold_score=2.0,
    )
    summary = summarize([row])["e4b"]["typed_decisions"]
    assert summary["score_mae"] == 1.0
    assert summary["score_n"] == 1
    with pytest.raises(ValueError, match="Duplicate"):
        summarize([row, row])


def test_wanli_labels_and_gold_isolation():
    row = {
        "dataset": "wanli",
        "state": "All tickets require reservations.",
        "question": "Are reservations required?",
        "options": [
            {"id": "supported", "description": "The evidence establishes the claim"},
            {"id": "insufficient", "description": "The evidence does not establish either"},
        ],
        "gold_id": "supported",
        "private_teacher_explanation": "never show this",
    }
    request = request_for(row)
    prompt, _ = render_prompt(request)
    assert request["state"].startswith("Evidence:\n")
    assert request["query"].startswith("Question:\n")
    assert "never show this" not in prompt
    assert set(request) == {"task", "state", "query", "options"}
    assert prompt.index("Current state:") < prompt.index("Query:") < prompt.index("Options:")


def test_unicode_line_separator_is_not_jsonl_boundary(tmp_path):
    path = tmp_path / "tiny.jsonl.gz"
    path.write_bytes(
        gzip.compress((json.dumps({"text": "a\u2028b"}, ensure_ascii=False) + "\n").encode())
    )
    assert read_jsonl(path) == [{"text": "a\u2028b"}]


def test_report_renders_both_tables_and_boolq_threshold_accuracy():
    summary = verify()
    references = json.loads((RESULTS / "references.json").read_text())
    output = markdown(summary, references)
    assert "91.50%" in output
    assert "HelpSteer2" in output
    assert "laya-typed-decisions" in output
    boolq = next(row for row in references if row["dataset"] == "boolq")
    assert boolq["metrics"] == {"n": 200, "auroc": 0.9692, "accuracy": 0.915}


def test_report_uses_readme_metrics_for_local_and_external_systems():
    summary = verify()
    references = json.loads((RESULTS / "references.json").read_text())
    output = markdown(summary, references)
    assert "| Banking20 | Macro-F1 ↑ | 300 | 83.72% | 85.84% |" in output
    assert "| 20 Newsgroups | Macro-F1 ↑ | 300 | 64.75% | 68.31% |" in output
    assert "| Prompt injection | Macro-F1 ↑ | 300 | 82.75% | 92.88% |" in output
    assert "| Typed decisions | Score MAE ↓ | 800 | 0.4236 | 0.4530 |" in output
    assert "| HelpSteer2 | Score MAE ↓ | 200 | 1.0815 | 1.0228 |" in output
    banking_reference = next(
        line
        for line in output.splitlines()
        if "Qwen/Qwen2.5-7B-Instruct / raw" in line and "| Banking20 |" in line
    )
    assert "| Macro-F1 ↑ | 300 | 73.77% |" in banking_reference
    assert "72.33%" not in banking_reference
