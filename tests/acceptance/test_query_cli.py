"""CLI query behavior tests for T20-T23 expectations."""

from __future__ import annotations

import json
import warnings

from minsync.cli import build_parser, main
from minsync.core import QueryResult


class _FakeMinSync:
    def __init__(self, results: list[QueryResult]) -> None:
        self._results = results
        self.calls: list[dict[str, object]] = []

    def query(
        self,
        query_text: str,
        *,
        k: int = 10,
        ref: str | None = None,
        filter_expr: str | None = None,
        show_score: bool = False,
    ) -> list[QueryResult]:
        self.calls.append({
            "query_text": query_text,
            "k": k,
            "ref": ref,
            "filter_expr": filter_expr,
            "show_score": show_score,
        })
        return self._results[: max(int(k), 0)]


class _FakeEmptyIndexMinSync:
    def query(
        self,
        query_text: str,
        *,
        k: int = 10,
        ref: str | None = None,
        filter_expr: str | None = None,
        show_score: bool = False,
    ) -> list[QueryResult]:
        del query_text, k, ref, filter_expr, show_score
        warnings.warn("index is empty. Run minsync sync first.", RuntimeWarning, stacklevel=2)
        return []


def test_query_parser_accepts_format_after_subcommand() -> None:
    parser = build_parser()
    args = parser.parse_args(["query", "authentication flow", "--format", "json", "--k", "3"])

    assert args.command == "query"
    assert args.format == "json"
    assert args.k == 3


def test_query_json_output_is_enveloped(capsys) -> None:
    parser = build_parser()
    args = parser.parse_args(["query", "authentication flow", "--format", "json", "--k", "1"])
    fake = _FakeMinSync([
        QueryResult(
            doc_id="doc-1",
            path="docs/auth/login.md",
            heading_path="Authentication > Login",
            chunk_type="child",
            text="The login process begins with credential validation.",
            score=0.92,
            content_commit="abc1234",
        )
    ])

    exit_code = args.handler(fake, args)
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert fake.calls[0]["query_text"] == "authentication flow"
    assert payload["query"] == "authentication flow"
    assert isinstance(payload["results"], list)
    assert len(payload["results"]) == 1
    assert payload["results"][0]["rank"] == 1
    assert payload["results"][0]["doc_id"] == "doc-1"
    assert payload["results"][0]["path"] == "docs/auth/login.md"
    assert "text" in payload["results"][0]
    assert "score" in payload["results"][0]


def test_query_empty_index_warning_is_emitted_to_stderr(capsys) -> None:
    parser = build_parser()
    args = parser.parse_args(["query", "authentication flow"])

    exit_code = args.handler(_FakeEmptyIndexMinSync(), args)
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Warning: index is empty" in captured.err


def test_query_empty_text_returns_clear_cli_error(capsys) -> None:
    exit_code = main(["query", ""])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "query text is required" in captured.err.lower()
