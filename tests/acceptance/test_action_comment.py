"""Tests for the GitHub Action PR comment formatter."""

from __future__ import annotations

from action.comment import MARKER, format_comment


class TestFormatComment:
    """Tests for format_comment()."""

    def test_successful_sync_with_verify_passed(self):
        sync_result = {
            "already_up_to_date": False,
            "from_commit": "abc12345deadbeef",
            "to_commit": "def67890cafebabe",
            "files_processed": 3,
            "files_processed_paths": ["docs/guide.md", "docs/api.md", "README.md"],
            "chunks_added": 12,
            "chunks_updated": 3,
            "chunks_deleted": 2,
        }
        verify_result = {
            "all_passed": True,
            "basic_checks": {
                "cursor_valid": True,
                "no_pending_txn": True,
                "schema_match": True,
                "collection_alive": True,
            },
            "file_checks": [],
        }
        body = format_comment(sync_result=sync_result, verify_result=verify_result)

        assert body.startswith(MARKER)
        assert "### Sync: Completed" in body
        assert "`abc12345`" in body
        assert "`def67890`" in body
        assert "| Files processed | 3 |" in body
        assert "| Chunks added | 12 |" in body
        assert "| Chunks updated | 3 |" in body
        assert "| Chunks deleted | 2 |" in body
        assert "`docs/guide.md`" in body
        assert "### Verify: PASSED" in body
        assert "Cursor Valid" in body

    def test_already_up_to_date(self):
        sync_result = {
            "already_up_to_date": True,
            "from_commit": "abc12345",
            "to_commit": "abc12345",
            "files_processed": 0,
            "chunks_added": 0,
            "chunks_updated": 0,
            "chunks_deleted": 0,
        }
        body = format_comment(sync_result=sync_result, verify_skipped=True)

        assert "### Sync: Already up to date" in body
        assert "No changes detected" in body
        assert "### Verify: Skipped" in body

    def test_sync_error(self):
        body = format_comment(sync_error="Error: Lock file exists, another sync is running")

        assert "### Sync: Error" in body
        assert "Lock file exists" in body
        assert "```" in body

    def test_verify_failure_with_file_issues(self):
        sync_result = {
            "already_up_to_date": False,
            "from_commit": "aaa11111",
            "to_commit": "bbb22222",
            "files_processed": 2,
            "files_processed_paths": ["a.md", "b.md"],
            "chunks_added": 5,
            "chunks_updated": 0,
            "chunks_deleted": 0,
        }
        verify_result = {
            "all_passed": False,
            "basic_checks": {
                "cursor_valid": True,
                "no_pending_txn": True,
                "schema_match": False,
            },
            "file_checks": [
                {"path": "a.md", "status": "OK", "issues": []},
                {"path": "b.md", "status": "FAIL", "issues": ["chunk count mismatch"]},
            ],
        }
        body = format_comment(sync_result=sync_result, verify_result=verify_result)

        assert "### Verify: FAILED" in body
        assert "\u274c" in body  # ❌ for schema_match
        assert "\u2705" in body  # ✅ for passing checks
        assert "`b.md`" in body
        assert "chunk count mismatch" in body

    def test_verify_skipped(self):
        sync_result = {
            "already_up_to_date": False,
            "from_commit": "aaa",
            "to_commit": "bbb",
            "files_processed": 1,
            "files_processed_paths": ["x.md"],
            "chunks_added": 1,
            "chunks_updated": 0,
            "chunks_deleted": 0,
        }
        body = format_comment(sync_result=sync_result, verify_skipped=True)

        assert "### Verify: Skipped" in body
        assert "### Verify: PASSED" not in body
        assert "### Verify: FAILED" not in body

    def test_large_file_list_collapsible(self):
        paths = [f"docs/file_{i}.md" for i in range(10)]
        sync_result = {
            "already_up_to_date": False,
            "from_commit": "aaa",
            "to_commit": "bbb",
            "files_processed": 10,
            "files_processed_paths": paths,
            "chunks_added": 20,
            "chunks_updated": 0,
            "chunks_deleted": 0,
        }
        body = format_comment(sync_result=sync_result, verify_skipped=True)

        assert "<details>" in body
        assert "<summary>" in body
        assert "Files (10)" in body
        for p in paths:
            assert f"`{p}`" in body

    def test_small_file_list_not_collapsible(self):
        paths = ["a.md", "b.md"]
        sync_result = {
            "already_up_to_date": False,
            "from_commit": "aaa",
            "to_commit": "bbb",
            "files_processed": 2,
            "files_processed_paths": paths,
            "chunks_added": 4,
            "chunks_updated": 0,
            "chunks_deleted": 0,
        }
        body = format_comment(sync_result=sync_result, verify_skipped=True)

        assert "**Files (2):**" in body
        assert "<details>" not in body

    def test_empty_sync_result(self):
        sync_result = {
            "already_up_to_date": False,
            "from_commit": None,
            "to_commit": "abc12345",
            "files_processed": 0,
            "files_processed_paths": [],
            "chunks_added": 0,
            "chunks_updated": 0,
            "chunks_deleted": 0,
        }
        body = format_comment(sync_result=sync_result, verify_skipped=True)

        assert "### Sync: Completed" in body
        assert "(initial)" in body
        assert "| Files processed | 0 |" in body

    def test_marker_present(self):
        body = format_comment(sync_error="test")
        assert body.startswith(MARKER)

    def test_no_sync_result_and_no_error(self):
        body = format_comment()
        assert "### Sync: No Result" in body

    def test_initial_sync_from_commit_none(self):
        sync_result = {
            "already_up_to_date": False,
            "from_commit": None,
            "to_commit": "deadbeef12345678",
            "files_processed": 5,
            "files_processed_paths": ["a.md", "b.md", "c.md", "d.md", "e.md"],
            "chunks_added": 30,
            "chunks_updated": 0,
            "chunks_deleted": 0,
        }
        body = format_comment(sync_result=sync_result, verify_skipped=True)

        assert "(initial)" in body
        assert "`deadbeef`" in body

    def test_verify_all_basic_checks_pass(self):
        verify_result = {
            "all_passed": True,
            "basic_checks": {
                "cursor_valid": True,
                "no_pending_txn": True,
            },
            "file_checks": [],
        }
        body = format_comment(
            sync_result={"already_up_to_date": True},
            verify_result=verify_result,
        )

        assert "### Verify: PASSED" in body
        # All checks should show ✅
        assert body.count("\u2705") == 2
        assert "\u274c" not in body
