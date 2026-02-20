"""T20–T23: MinSync query — search, structured results, empty index, empty query.

TDD tests for ``MinSync.query()`` Python API.
The implementation does NOT exist yet; these tests define the expected behavior.

References:
    - ai_instruction/E2E_TEST_PLAN.md  (T20, T21, T22, T23)
    - ai_instruction/CLI_SPEC.md       (section 3: minsync query)
"""

from __future__ import annotations

import warnings

import pytest

# ============================================================================
# T20: query -- basic search
# ============================================================================


class TestT20BasicQuery:
    """T20: After init+sync with all sample files, query returns relevant results."""

    # -- T20-1: query succeeds (no exception) ---------------------------------
    def test_t20_1_query_succeeds(self, synced_repo):
        """query('login process authentication', k=5) should not raise."""
        _repo, ms = synced_repo
        results = ms.query("login process authentication", k=5)
        # No exception means success.
        assert results is not None

    # -- T20-2: at least one result returned ----------------------------------
    def test_t20_2_results_not_empty(self, synced_repo):
        """query should return at least one result from the indexed documents."""
        _repo, ms = synced_repo
        results = ms.query("login process authentication", k=5)
        assert len(results) > 0

    # -- T20-3: at least one result has 'login' in path -----------------------
    def test_t20_3_login_in_results_path(self, synced_repo):
        """At least one result should come from docs/auth/login.md."""
        _repo, ms = synced_repo
        results = ms.query("login process authentication", k=5)
        paths = [r.path for r in results]
        assert any("login" in p for p in paths), (
            f"Expected at least one result with 'login' in path, got paths: {paths}"
        )

    # -- T20-4: each result has path and text attributes ----------------------
    def test_t20_4_results_have_path_and_text(self, synced_repo):
        """Every QueryResult should expose path and text attributes."""
        _repo, ms = synced_repo
        results = ms.query("login process authentication", k=5)
        assert len(results) > 0, "Need at least one result for attribute checks"
        for r in results:
            assert hasattr(r, "path"), f"Result missing 'path' attribute: {r}"
            assert hasattr(r, "text"), f"Result missing 'text' attribute: {r}"
            assert isinstance(r.path, str) and len(r.path) > 0
            assert isinstance(r.text, str) and len(r.text) > 0


# ============================================================================
# T21: query -- structured results (JSON-like output)
# ============================================================================


class TestT21StructuredResults:
    """T21: Query results are valid structured data with expected attributes."""

    # -- T21-1: results are valid structured data (list of QueryResult) -------
    def test_t21_1_results_are_structured(self, synced_repo):
        """query should return a list of structured result objects."""
        _repo, ms = synced_repo
        results = ms.query("vector index sync engine", k=3)
        assert isinstance(results, list)
        # Each element should be a structured object (not a raw dict or string).
        for r in results:
            assert hasattr(r, "doc_id")
            assert hasattr(r, "path")
            assert hasattr(r, "text")
            assert hasattr(r, "score")

    # -- T21-2: len(results) <= k --------------------------------------------
    def test_t21_2_results_respect_k_limit(self, synced_repo):
        """Number of results should not exceed the requested k."""
        _repo, ms = synced_repo
        results = ms.query("vector index sync engine", k=3)
        assert len(results) <= 3

    # -- T21-3: each result has doc_id, path, text, score attributes ----------
    def test_t21_3_result_attributes(self, synced_repo):
        """Each result must expose doc_id, path, text, and score."""
        _repo, ms = synced_repo
        results = ms.query("vector index sync engine", k=3)
        assert len(results) > 0, "Need at least one result for attribute checks"
        for r in results:
            # doc_id
            assert hasattr(r, "doc_id")
            assert isinstance(r.doc_id, str) and len(r.doc_id) > 0
            # path
            assert hasattr(r, "path")
            assert isinstance(r.path, str) and len(r.path) > 0
            # text
            assert hasattr(r, "text")
            assert isinstance(r.text, str) and len(r.text) > 0
            # score
            assert hasattr(r, "score")
            assert isinstance(r.score, (int, float))
            assert r.score >= 0.0


# ============================================================================
# T22: query -- empty index (init but no sync)
# ============================================================================


class TestT22EmptyIndex:
    """T22: Querying an empty index (init done, sync NOT done) returns 0 results
    with a warning."""

    # -- T22-1: query succeeds (no exception) ---------------------------------
    def test_t22_1_query_on_empty_index_succeeds(self, initialized_repo):
        """query on an empty index should not raise an exception."""
        _repo, ms = initialized_repo
        results = ms.query("login process authentication", k=5)
        # Should succeed (exit code 0 equivalent in Python API = no exception).
        assert results is not None

    # -- T22-2: len(results) == 0 --------------------------------------------
    def test_t22_2_empty_results(self, initialized_repo):
        """query on an empty index should return zero results."""
        _repo, ms = initialized_repo
        results = ms.query("login process authentication", k=5)
        assert len(results) == 0

    # -- T22-3: warning about empty index -------------------------------------
    def test_t22_3_empty_index_warning(self, initialized_repo):
        """query on an empty index should emit a warning about the index being empty."""
        _repo, ms = initialized_repo
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            results = ms.query("login process authentication", k=5)

        # Accept either a Python warning or the method returning a result object
        # that carries a warning/message.  The E2E plan says "index is empty"
        # should appear somewhere.
        warning_texts = [str(w.message) for w in caught]
        has_warning_via_warnings = any("index is empty" in t.lower() for t in warning_texts)

        # Some implementations may surface the warning as an attribute on the
        # return value (e.g., result.warnings or a printed message).  We check
        # the Python warnings module first as the primary mechanism.
        has_warning_via_attr = False
        if hasattr(results, "warnings"):
            has_warning_via_attr = any("index is empty" in str(w).lower() for w in results.warnings)

        assert has_warning_via_warnings or has_warning_via_attr, (
            f"Expected a warning containing 'index is empty'. Caught warnings: {warning_texts}"
        )


# ============================================================================
# T23: query -- empty query string
# ============================================================================


class TestT23EmptyQueryString:
    """T23: Querying with an empty string should raise an error."""

    # -- T23-1: query("") raises exception ------------------------------------
    def test_t23_1_empty_query_raises(self, synced_repo):
        """query('') must raise an exception."""
        _repo, ms = synced_repo
        with pytest.raises(Exception):  # noqa: B017
            ms.query("", k=5)

    # -- T23-2: error message contains "query text is required" ---------------
    def test_t23_2_error_message(self, synced_repo):
        """The error raised by query('') must mention 'query text is required'."""
        _repo, ms = synced_repo
        with pytest.raises(Exception, match=r"(?i)query text is required"):
            ms.query("", k=5)
