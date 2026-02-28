#!/usr/bin/env python3
"""Entrypoint script for the MinSync GitHub Action.

Runs ``minsync sync`` (and optionally ``minsync verify``), parses the JSON
output, writes GitHub Actions outputs, and produces the PR comment body.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import uuid
from typing import Any


def _run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, check=check)  # noqa: S603


def _write_output(name: str, value: str) -> None:
    output_file = os.environ.get("GITHUB_OUTPUT", "")
    if output_file:
        with open(output_file, "a") as fh:
            fh.write(f"{name}={value}\n")


def _write_multiline_output(name: str, value: str) -> None:
    output_file = os.environ.get("GITHUB_OUTPUT", "")
    if output_file:
        delimiter = uuid.uuid4().hex
        with open(output_file, "a") as fh:
            fh.write(f"{name}<<{delimiter}\n{value}\n{delimiter}\n")


def _run_sync() -> tuple[dict[str, Any] | None, str | None]:
    ref = os.environ.get("INPUT_REF", "")
    sync_args = os.environ.get("INPUT_SYNC_ARGS", "")

    sync_cmd = ["minsync", "sync", "--format", "json"]
    if ref:
        sync_cmd += ["--ref", ref]
    if sync_args:
        sync_cmd += shlex.split(sync_args)

    proc = _run(sync_cmd, check=False)

    if proc.returncode != 0:
        error = (proc.stderr or proc.stdout).strip()
        _write_output("sync-result", "error")
        return None, error

    try:
        result = json.loads(proc.stdout)
    except json.JSONDecodeError:
        error = f"Failed to parse sync output: {proc.stdout[:500]}"
        _write_output("sync-result", "error")
        return None, error

    if result.get("already_up_to_date"):
        _write_output("sync-result", "up-to-date")
    else:
        _write_output("sync-result", "completed")
    _write_output("chunks-added", str(result.get("chunks_added", 0)))
    _write_output("chunks-updated", str(result.get("chunks_updated", 0)))
    _write_output("chunks-deleted", str(result.get("chunks_deleted", 0)))
    _write_output("files-processed", str(result.get("files_processed", 0)))
    return result, None


def _run_verify() -> dict[str, Any] | None:
    ref = os.environ.get("INPUT_REF", "")
    verify_args = os.environ.get("INPUT_VERIFY_ARGS", "")

    verify_cmd = ["minsync", "verify", "--format", "json"]
    if ref:
        verify_cmd += ["--ref", ref]
    if verify_args:
        verify_cmd += shlex.split(verify_args)

    proc = _run(verify_cmd, check=False)
    try:
        result = json.loads(proc.stdout)
    except json.JSONDecodeError:
        result = {"all_passed": False, "basic_checks": {}, "file_checks": []}

    all_passed = result.get("all_passed", False)
    _write_output("verify-passed", "true" if all_passed else "false")
    _write_multiline_output("verify-result", json.dumps(result))
    return result


def main() -> int:
    from action.comment import format_comment

    do_verify = os.environ.get("INPUT_VERIFY", "true").lower() == "true"

    sync_result, sync_error = _run_sync()

    verify_result = None
    verify_skipped = not do_verify
    if do_verify and sync_error is None:
        verify_result = _run_verify()
    else:
        _write_output("verify-passed", "")
        _write_output("verify-result", "")

    comment_body = format_comment(
        sync_result=sync_result,
        sync_error=sync_error,
        verify_result=verify_result,
        verify_skipped=verify_skipped,
    )
    _write_multiline_output("comment-body", comment_body)

    if sync_error is not None:
        print(f"::error::MinSync sync failed: {sync_error}", file=sys.stderr)
        return 1

    if verify_result and not verify_result.get("all_passed", False):
        print("::warning::MinSync verify reported issues", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
