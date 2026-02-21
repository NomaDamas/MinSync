"""Command-line interface for MinSync."""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import warnings
from pathlib import Path
from typing import Any

from minsync.core import (
    DEFAULT_CHUNKER_ID,
    DEFAULT_EMBEDDER_ID,
    MinSync,
    MinSyncError,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="minsync")
    parser.add_argument("--version", action="version", version="minsync 0.0.1")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging output.")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress non-error output.")
    parser.add_argument("--format", choices=("text", "json"), default="text", help="Output format.")

    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="Initialize MinSync in the current git repository.")
    init_parser.add_argument("--collection")
    init_parser.add_argument("--embedder", default=DEFAULT_EMBEDDER_ID)
    init_parser.add_argument("--chunker", default=DEFAULT_CHUNKER_ID)
    init_parser.add_argument("--force", action="store_true")
    init_parser.set_defaults(handler=_handle_init)

    sync_parser = subparsers.add_parser("sync", help="Synchronize git changes into the index.")
    sync_parser.add_argument("--ref")
    sync_parser.add_argument("--full", action="store_true")
    sync_parser.add_argument("--dry-run", action="store_true")
    sync_parser.add_argument("--batch-size", type=int)
    sync_parser.add_argument("--wait", action="store_true")
    sync_parser.set_defaults(handler=_handle_sync)

    query_parser = subparsers.add_parser("query", help="Query indexed content.")
    query_parser.add_argument("query_text")
    query_parser.add_argument("--k", type=int, default=10)
    query_parser.add_argument("--ref")
    query_parser.add_argument("--filter", dest="filter_expr")
    query_parser.add_argument("--format", choices=("text", "json", "jsonl"))
    query_parser.add_argument("--show-score", action="store_true")
    query_parser.set_defaults(handler=_handle_query)

    status_parser = subparsers.add_parser("status", help="Show repository sync status.")
    status_parser.add_argument("--format", choices=("text", "json"))
    status_parser.set_defaults(handler=_handle_status)

    check_parser = subparsers.add_parser("check", help="Run dependency and environment checks.")
    check_parser.add_argument("--format", choices=("text", "json"))
    check_parser.set_defaults(handler=_handle_check)

    verify_parser = subparsers.add_parser("verify", help="Verify and optionally repair index consistency.")
    verify_parser.add_argument("--ref")
    verify_parser.add_argument("--all", action="store_true")
    verify_parser.add_argument("--fix", action="store_true")
    verify_parser.add_argument("--sample", type=int)
    verify_parser.add_argument("--format", choices=("text", "json"))
    verify_parser.set_defaults(handler=_handle_verify)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    ms = MinSync(repo_path=Path.cwd())

    try:
        return args.handler(ms, args)
    except MinSyncError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return exc.exit_code
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


def _handle_init(ms: MinSync, args: argparse.Namespace) -> int:
    result = ms.init(
        collection=args.collection,
        embedder=args.embedder,
        chunker=args.chunker,
        force=args.force,
    )
    if args.quiet:
        return 0

    if args.format == "json":
        print(json.dumps(_to_jsonable(result), indent=2, sort_keys=True))
        return 0

    print("Initialized MinSync in .minsync/")
    print(f"  repo_id:      {result.repo_id}")
    print(f"  collection:   {result.collection}")
    print(f"  chunker:      {result.chunker}")
    print(f"  embedder:     {result.embedder}")
    print(f"  vectorstore:  {result.vectorstore} (local)")
    print()
    print("Run 'minsync check' to verify your setup, then 'minsync sync' to build the initial index.")
    return 0


def _handle_sync(ms: MinSync, args: argparse.Namespace) -> int:
    result = ms.sync(
        ref=args.ref,
        full=args.full,
        dry_run=args.dry_run,
        batch_size=args.batch_size,
        wait=args.wait,
        verbose=args.verbose,
    )
    _emit_result(args, result)
    return 0


def _handle_query(ms: MinSync, args: argparse.Namespace) -> int:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = ms.query(
            args.query_text,
            k=args.k,
            ref=args.ref,
            filter_expr=args.filter_expr,
            show_score=args.show_score,
        )

    for warning in caught:
        print(f"Warning: {warning.message}", file=sys.stderr)

    output_format = args.format or "text"
    if args.quiet:
        return 0

    if output_format == "json":
        payload = {
            "query": args.query_text,
            "ref": args.ref,
            "results": [_query_result_row(item, rank) for rank, item in enumerate(result, start=1)],
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    if output_format == "jsonl":
        for rank, item in enumerate(result, start=1):
            row = _query_result_row(item, rank)
            row["query"] = args.query_text
            if args.ref is not None:
                row["ref"] = args.ref
            print(json.dumps(row, sort_keys=True))
        return 0

    _emit_query_text(query_text=args.query_text, result=result, show_score=args.show_score)
    return 0


def _handle_status(ms: MinSync, args: argparse.Namespace) -> int:
    result = ms.status()
    _emit_result(args, result)
    return 0


def _handle_check(ms: MinSync, args: argparse.Namespace) -> int:
    result = ms.check()
    _emit_result(args, result)
    return 0 if bool(getattr(result, "all_passed", False)) else 1


def _handle_verify(ms: MinSync, args: argparse.Namespace) -> int:
    result = ms.verify(ref=args.ref, all=args.all, fix=args.fix, sample=args.sample)
    _emit_result(args, result)
    return 0 if bool(getattr(result, "all_passed", False)) else 1


def _emit_result(args: argparse.Namespace, result: Any) -> None:
    if args.quiet:
        return
    if result is None:
        return
    if args.format == "json":
        print(json.dumps(_to_jsonable(result), indent=2, sort_keys=True))
        return
    print(result)


def _to_jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    if isinstance(value, dict):
        return {key: _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if hasattr(value, "__dict__"):
        return {key: _to_jsonable(item) for key, item in vars(value).items()}
    if isinstance(value, Path):
        return str(value)
    return value


def _query_result_row(item: Any, rank: int) -> dict[str, Any]:
    row = _to_jsonable(item)
    if not isinstance(row, dict):
        row = {"value": str(item)}
    row["rank"] = rank
    return row


def _emit_query_text(*, query_text: str, result: list[Any], show_score: bool) -> None:
    print(f'Found {len(result)} results for "{query_text}":')
    if not result:
        return
    print()

    for rank, item in enumerate(result, start=1):
        row = _query_result_row(item, rank)
        path = str(row.get("path") or "")
        heading = str(row.get("heading_path") or "")
        text = str(row.get("text") or "")
        score = row.get("score")

        score_suffix = ""
        if show_score and isinstance(score, int | float):
            score_suffix = f" (score: {score:.2f})"

        print(f"[{rank}] {path}{score_suffix}")
        if heading:
            print(f"    heading: {heading}")
        print("    ---")
        print(f"    {text.strip()}")
        print("    ---")

        if rank != len(result):
            print()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
