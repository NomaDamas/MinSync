"""PR comment Markdown formatter for the MinSync GitHub Action."""

from __future__ import annotations

from typing import Any

MARKER = "<!-- minsync-action -->"
_FILES_COLLAPSE_THRESHOLD = 5


def format_comment(
    *,
    sync_result: dict[str, Any] | None = None,
    sync_error: str | None = None,
    verify_result: dict[str, Any] | None = None,
    verify_skipped: bool = False,
) -> str:
    """Build the full PR comment body including the HTML marker."""
    sections: list[str] = [MARKER, "## MinSync Sync Report", ""]

    if sync_error is not None:
        sections.append(_sync_error_section(sync_error))
    elif sync_result is not None:
        sections.append(_sync_section(sync_result))
    else:
        sections.append("### Sync: No Result")

    sections.append("")

    if verify_skipped:
        sections.append("### Verify: Skipped")
    elif verify_result is not None:
        sections.append(_verify_section(verify_result))

    return "\n".join(sections).rstrip() + "\n"


def _sync_section(result: dict[str, Any]) -> str:
    if result.get("already_up_to_date"):
        return "### Sync: Already up to date\n\nNo changes detected since last sync."

    from_commit = result.get("from_commit") or "(initial)"
    to_commit = result.get("to_commit") or "unknown"
    from_short = from_commit[:8] if from_commit != "(initial)" else from_commit
    to_short = to_commit[:8]

    files_processed = result.get("files_processed", 0)
    chunks_added = result.get("chunks_added", 0)
    chunks_updated = result.get("chunks_updated", 0)
    chunks_deleted = result.get("chunks_deleted", 0)

    lines: list[str] = [
        "### Sync: Completed",
        f"**Commit range:** `{from_short}` \u2192 `{to_short}`",
        "",
        "| Metric | Count |",
        "|--------|------:|",
        f"| Files processed | {files_processed} |",
        f"| Chunks added | {chunks_added} |",
        f"| Chunks updated | {chunks_updated} |",
        f"| Chunks deleted | {chunks_deleted} |",
    ]

    paths = result.get("files_processed_paths") or []
    if paths:
        lines.append("")
        lines.append(_files_detail(paths))

    return "\n".join(lines)


def _sync_error_section(error: str) -> str:
    return f"### Sync: Error\n\n```\n{error}\n```"


def _verify_section(result: dict[str, Any]) -> str:
    all_passed = result.get("all_passed", False)
    status = "PASSED" if all_passed else "FAILED"

    lines: list[str] = [f"### Verify: {status}"]

    basic_checks = result.get("basic_checks") or {}
    if basic_checks:
        lines.append("")
        lines.append("| Check | Status |")
        lines.append("|-------|--------|")
        for key, passed in basic_checks.items():
            label = key.replace("_", " ").title()
            icon = "\u2705" if passed else "\u274c"
            lines.append(f"| {label} | {icon} |")

    file_checks = result.get("file_checks") or []
    failed_files = [fc for fc in file_checks if fc.get("status") != "OK"]
    if failed_files:
        lines.append("")
        issue_lines: list[str] = []
        for fc in failed_files:
            path = fc.get("path", "unknown")
            status_str = fc.get("status", "FAIL")
            issues = fc.get("issues") or []
            issue_str = ", ".join(str(i) for i in issues) if issues else status_str
            issue_lines.append(f"- `{path}`: {issue_str}")
        detail = "\n".join(issue_lines)
        lines.append(f"<details><summary><b>File issues ({len(failed_files)})</b></summary>\n\n{detail}\n\n</details>")

    return "\n".join(lines)


def _files_detail(paths: list[str]) -> str:
    file_lines = [f"- `{p}`" for p in paths]
    body = "\n".join(file_lines)
    count = len(paths)
    if count <= _FILES_COLLAPSE_THRESHOLD:
        return f"**Files ({count}):**\n{body}"
    return f"<details><summary><b>Files ({count})</b></summary>\n\n{body}\n\n</details>"
