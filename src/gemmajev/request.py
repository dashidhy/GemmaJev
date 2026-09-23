"""The public decision request and its single, task-independent prompt."""

from __future__ import annotations

from collections.abc import Mapping
from string import ascii_uppercase


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\0" in value:
        raise ValueError(f"{name} must be nonempty text without NUL bytes")
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as error:
        raise ValueError(f"{name} must be valid UTF-8 text without unpaired surrogates") from error
    return value


def validate_request(request: Mapping[str, object]) -> dict:
    """Return an owned copy of model input, ignoring all extra metadata.

    Options map semantic IDs to descriptions. Their insertion order determines
    the internal labels A–Z; caller strings are never stripped or rewritten.
    Use short meaningful IDs, such as ``refund`` or ``not enough evidence``.
    """
    if not isinstance(request, Mapping):
        raise ValueError("request must contain task, state, query and options")  # noqa: TRY004
    result = {key: _text(request.get(key), key) for key in ("task", "state", "query")}
    options = request.get("options")
    if not isinstance(options, Mapping) or not 2 <= len(options) <= 26:
        raise ValueError("options must map 2–26 semantic IDs to descriptions")
    copied = {}
    for option_id, description in options.items():
        option_id = _text(option_id, "option ID")
        if option_id in copied:
            raise ValueError("option IDs must be unique")
        copied[option_id] = _text(description, "option description")
    result["options"] = copied
    return result


def render_prompt(request: Mapping[str, object]) -> tuple[str, dict[str, str]]:
    """Assign internal labels and render only the four public input fields."""
    request = validate_request(request)
    labels = dict(zip(ascii_uppercase, request["options"]))
    options = "\n".join(
        f"{label} = {option_id}: {request['options'][option_id]}"
        for label, option_id in labels.items()
    )
    return (
        (
            f"{request['task']}\n\n"
            "Your response must begin immediately with one valid option label.\n"
            "Output only that label without explanation.\n\n"
            f"Current state:\n{request['state']}\n\n"
            f"Query:\n{request['query']}\n\nOptions:\n{options}"
        ),
        labels,
    )
