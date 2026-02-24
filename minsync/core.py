"""Core Python API for MinSync."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import time
import uuid
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pygit2
import yaml

from minsync.gitbackend import GitRepo

DEFAULT_REF = "main"
DEFAULT_EMBEDDER_ID = "openai:text-embedding-3-small"
DEFAULT_CHUNKER_ID = "markdown-heading"
DEFAULT_VECTORSTORE_ID = "zvec"


class MinSyncError(Exception):
    """Base application error with an associated CLI exit code."""

    def __init__(self, message: str, *, exit_code: int = 1) -> None:
        super().__init__(message)
        self.exit_code = exit_code


class MinSyncGitError(MinSyncError):
    """Raised when git-related preconditions fail."""

    def __init__(self, message: str) -> None:
        super().__init__(message, exit_code=2)


class MinSyncNotImplementedError(MinSyncError):
    """Raised for API methods not yet implemented in this story."""

    def __init__(self, message: str) -> None:
        super().__init__(message, exit_code=1)


@dataclass(frozen=True)
class InitResult:
    repo_id: str
    collection: str
    chunker: str
    embedder: str
    vectorstore: str
    path: str


@dataclass(frozen=True)
class SyncResult:
    from_commit: str | None
    to_commit: str
    files_processed: int
    files_processed_paths: list[str] = field(default_factory=list)
    chunks_added: int = 0
    chunks_updated: int = 0
    chunks_deleted: int = 0
    dry_run: bool = False
    already_up_to_date: bool = False
    planned_files: list[str] = field(default_factory=list)
    files_planned: int = 0
    recovered: bool = False


@dataclass(frozen=True)
class QueryResult:
    doc_id: str
    path: str
    heading_path: str
    chunk_type: str
    text: str
    score: float
    content_commit: str = ""


@dataclass(frozen=True)
class StatusResult:
    repo_id: str
    ref: str
    collection: str
    chunker: str
    embedder: str
    vectorstore: str
    last_synced_commit: str | None
    current_head: str
    state: str
    commits_behind: int = 0
    pending_txn: dict[str, Any] | None = None

    def __str__(self) -> str:
        state_text = _status_text(self.state)
        if self.state == "OUT_OF_DATE":
            suffix = "commit" if self.commits_behind == 1 else "commits"
            state_text = f"{state_text} ({self.commits_behind} {suffix} behind)"
        if self.state == "INTERRUPTED":
            started = _coerce_text(self.pending_txn, "started_at")
            if started:
                state_text = f"{state_text} (txn.json found, started {started})"
            else:
                state_text = f"{state_text} (txn.json found)"

        lines = [
            "MinSync Status",
            f"  repo_id:         {_short_commit(self.repo_id)}",
            f"  ref:             {self.ref}",
            f"  collection:      {self.collection}",
            f"  chunker:         {self.chunker}",
            f"  embedder:        {self.embedder}",
            f"  vectorstore:     {self.vectorstore}",
            f"  last synced:     {_short_commit(self.last_synced_commit) if self.last_synced_commit else '(never)'}",
            f"  current HEAD:    {_short_commit(self.current_head)}",
            f"  status:          {state_text}",
        ]

        if isinstance(self.pending_txn, dict):
            from_commit = _short_commit(_coerce_text(self.pending_txn, "from_commit"))
            to_commit = _short_commit(_coerce_text(self.pending_txn, "to_commit"))
            lines.append(f"  pending txn:     {from_commit} -> {to_commit}")
            lines.append("")
            lines.append("Run 'minsync sync' to resume/recover.")
        return "\n".join(lines)


@dataclass(frozen=True)
class CheckResult:
    git_ok: bool
    embedder_ok: bool
    vectorstore_ok: bool
    errors: list[str] = field(default_factory=list)
    all_passed: bool = True
    git: dict[str, Any] = field(default_factory=dict)
    embedder: dict[str, Any] = field(default_factory=dict)
    vectorstore: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        git_info = ""
        if self.git_ok:
            repo_id = _coerce_text(self.git, "repo_id")
            ref = _coerce_text(self.git, "ref")
            head = _coerce_text(self.git, "head")
            if repo_id and ref and head:
                git_info = f" (repo_id={_short_commit(repo_id)}, ref={ref}, HEAD={_short_commit(head)})"

        embedder_id = _coerce_text(self.embedder, "id")
        embedder_info = f"{embedder_id} ... " if embedder_id else ""
        if self.embedder_ok:
            dim = self.embedder.get("dimension")
            latency_ms = self.embedder.get("latency_ms")
            if isinstance(dim, int) and isinstance(latency_ms, int):
                embedder_info += f"OK (dim={dim}, {latency_ms / 1000:.1f}s)"
            else:
                embedder_info += "OK"
        else:
            embedder_info += "FAIL"

        vectorstore_id = _coerce_text(self.vectorstore, "id")
        vectorstore_info = f"{vectorstore_id} ... " if vectorstore_id else ""
        if self.vectorstore_ok:
            doc_count = self.vectorstore.get("doc_count")
            if isinstance(doc_count, int):
                vectorstore_info += f"OK ({doc_count} docs)"
            else:
                vectorstore_info += "OK"
        else:
            vectorstore_info += "FAIL"

        lines = [
            "MinSync Health Check",
            f"  Git:          {'OK' if self.git_ok else 'FAIL'}{git_info}",
            f"  Embedder:     {embedder_info}",
            f"  VectorStore:  {vectorstore_info}",
        ]

        if self.errors:
            for message in self.errors:
                lines.append(f"    Error: {message}")

        lines.append("")
        if self.all_passed:
            lines.append("All checks passed.")
        else:
            failed_checks = int(not self.git_ok) + int(not self.embedder_ok) + int(not self.vectorstore_ok)
            lines.append(f"{failed_checks} checks failed.")
        return "\n".join(lines)


@dataclass(frozen=True)
class VerifyResult:
    all_passed: bool
    basic_checks: dict[str, bool] = field(default_factory=dict)
    file_checks: list[dict[str, Any]] = field(default_factory=list)
    ignored_stale: list[str] = field(default_factory=list)
    fixed: bool = False

    def __str__(self) -> str:
        lines = ["MinSync Verify"]
        basic_pass = all(bool(value) for value in self.basic_checks.values())
        lines.append(f"Basic checks:       {'PASS' if basic_pass else 'FAIL'}")
        for key, passed in self.basic_checks.items():
            label = key.replace("_", " ")
            lines.append(f"  {label:<17} {'OK' if passed else 'FAIL'}")

        lines.append("")
        if self.ignored_stale:
            lines.append("Ignored files check: FAIL")
            for path in self.ignored_stale:
                lines.append(f"  IGNORED_STALE {path}")
        elif self.fixed:
            lines.append("Ignored files check: FIXED")
        else:
            lines.append("Ignored files check: PASS")

        if self.file_checks:
            lines.append("")
            lines.append("File verification:")
            for check in self.file_checks:
                path = str(check.get("path") or "")
                status = str(check.get("status") or "UNKNOWN")
                issues = [str(issue) for issue in check.get("issues", [])]
                if status == "OK":
                    lines.append(f"  {path:<24} OK")
                    continue
                if status == "FIXED":
                    lines.append(f"  {path:<24} FIXED")
                    continue
                issue_text = ", ".join(issues) if issues else "FAIL"
                lines.append(f"  {path:<24} FAIL ({issue_text})")

        lines.append("")
        if self.all_passed:
            if self.fixed:
                lines.append("Result: ALL CHECKS PASSED (after fix)")
            else:
                lines.append("Result: ALL CHECKS PASSED")
        else:
            lines.append("Result: VERIFICATION FAILED")
        return "\n".join(lines)


class _DefaultChunker:
    """Fallback chunker used when no custom chunker is injected."""

    def __init__(self, chunker_id: str) -> None:
        self._chunker_id = chunker_id

    def schema_id(self) -> str:
        return self._chunker_id

    def chunk(self, text: str, path: str) -> list[Any]:
        from minsync.protocols import Chunk

        stripped = text.strip()
        if not stripped:
            return []
        return [Chunk(chunk_type="parent", text=stripped, heading_path="")]


class _DefaultEmbedder:
    """Fallback deterministic embedder used when no custom embedder is injected."""

    def __init__(self, embedder_id: str) -> None:
        self._embedder_id = embedder_id

    def id(self) -> str:
        return self._embedder_id

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            vectors.append([byte / 255.0 for byte in digest])
        return vectors


class _InMemoryVectorStore:
    """Fallback in-memory vector store used when no custom store is injected."""

    def __init__(self) -> None:
        self._docs: dict[str, dict[str, Any]] = {}

    def upsert(self, docs: list[dict[str, Any]]) -> None:
        for doc in docs:
            self._docs[str(doc["id"])] = dict(doc)

    def update(self, docs: list[dict[str, Any]]) -> None:
        for doc in docs:
            doc_id = str(doc["id"])
            if doc_id in self._docs:
                self._docs[doc_id].update(doc)

    def fetch(self, ids: list[str]) -> list[dict[str, Any]]:
        return [dict(self._docs[doc_id]) for doc_id in ids if doc_id in self._docs]

    def delete_by_filter(self, filter_expr: str) -> int:
        to_delete = [doc_id for doc_id, doc in self._docs.items() if _matches_filter(doc, filter_expr)]
        for doc_id in to_delete:
            del self._docs[doc_id]
        return len(to_delete)

    def query(self, vector: list[float], filter_expr: str | None = None, topk: int = 10) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        for doc in self._docs.values():
            if filter_expr and not _matches_filter(doc, filter_expr):
                continue
            embedding = doc.get("embedding")
            if not isinstance(embedding, list):
                continue
            score = _cosine_similarity(vector, embedding)
            candidates.append({**doc, "score": score})

        candidates.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
        return candidates[: max(topk, 0)]

    def doc_count(self) -> int:
        return len(self._docs)

    def get_all_docs(self) -> list[dict[str, Any]]:
        return [dict(doc) for doc in self._docs.values()]

    def flush(self) -> None:
        return


@dataclass(frozen=True)
class _IgnoreRule:
    regex: re.Pattern[str]
    negated: bool


class _IgnoreMatcher:
    """Minimal gitignore-style matcher for .minsyncignore patterns."""

    def __init__(self, rules: list[_IgnoreRule]) -> None:
        self._rules = rules

    @classmethod
    def from_text(cls, text: str) -> _IgnoreMatcher:
        rules: list[_IgnoreRule] = []
        for raw_line in text.splitlines():
            stripped = raw_line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            rules.append(_compile_ignore_rule(stripped))
        return cls(rules)

    @classmethod
    def empty(cls) -> _IgnoreMatcher:
        return cls([])

    def matches(self, path: str) -> bool:
        normalized = path.replace("\\", "/").lstrip("./")
        matched = False
        for rule in self._rules:
            if rule.regex.match(normalized):
                matched = not rule.negated
        return matched


class MinSync:
    """Python API surface for MinSync."""

    def __init__(self, repo_path: str | Path = ".", chunker=None, embedder=None, vector_store=None) -> None:
        self.repo_path = Path(repo_path).resolve()
        self._chunker_injected = chunker is not None
        self._embedder_injected = embedder is not None
        self._vector_store_injected = vector_store is not None
        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = vector_store if vector_store is not None else _InMemoryVectorStore()
        self._git: GitRepo | None = None

    def _git_repo(self) -> GitRepo:
        """Lazy-discover the git repository on first access."""
        if self._git is None:
            try:
                self._git = GitRepo.discover(self.repo_path)
            except (KeyError, pygit2.GitError) as exc:
                raise MinSyncGitError("not a git repository") from exc
        return self._git

    def init(
        self,
        collection: str | None = None,
        embedder: str = DEFAULT_EMBEDDER_ID,
        chunker: str = DEFAULT_CHUNKER_ID,
        force: bool = False,
    ) -> InitResult:
        """Initialize `.minsync/config.yaml` for a git repository."""
        repo_root = self._resolve_git_root()
        repo_id = self._resolve_repo_id(repo_root)
        minsync_dir = repo_root / ".minsync"

        if minsync_dir.exists():
            if not force:
                raise MinSyncError("already initialized. Use --force to reinitialize.", exit_code=1)
            self._remove_path(minsync_dir)

        minsync_dir.mkdir(parents=True, exist_ok=True)
        config = self._build_config(
            repo_id=repo_id,
            collection=collection or f"minsync_{repo_id[:8]}",
            embedder=embedder,
            chunker=chunker,
        )
        config_path = minsync_dir / "config.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

        return InitResult(
            repo_id=repo_id,
            collection=config["collection"]["name"],
            chunker=config["chunker"]["id"],
            embedder=config["embedder"]["id"],
            vectorstore=config["vectorstore"]["id"],
            path=str(minsync_dir),
        )

    def sync(
        self,
        *,
        ref: str | None = None,
        full: bool = False,
        dry_run: bool = False,
        batch_size: int | None = None,
        wait: bool = False,
        verbose: bool = False,
    ) -> SyncResult:
        del batch_size  # Reserved for a future batching implementation.

        repo_root = self._resolve_git_root()
        config = self._load_config(repo_root)
        self._ensure_vectorstore(config, repo_root)
        minsync_dir = repo_root / ".minsync"
        cursor_path = minsync_dir / "cursor.json"
        txn_path = minsync_dir / "txn.json"
        lock_path = minsync_dir / "lock"

        repo_id = str(config.get("repo_id") or self._resolve_repo_id(repo_root))
        ref_name = str(ref or config.get("ref") or DEFAULT_REF)
        config_chunker_id = str(config.get("chunker", {}).get("id", DEFAULT_CHUNKER_ID))
        config_embedder_id = str(config.get("embedder", {}).get("id", DEFAULT_EMBEDDER_ID))

        chunker = self.chunker if self.chunker is not None else self._create_chunker_from_config(config)
        embedder = self.embedder if self.embedder is not None else self._create_embedder_from_config(config)
        vector_store = self.vector_store

        chunk_schema_id = self._chunk_schema_id(chunker, config_chunker_id)
        embedder_id = self._embedder_id(embedder, config_embedder_id)

        with self._acquire_lock(lock_path, wait=wait):
            cursor = self._read_json(cursor_path) or {}
            txn = self._read_json(txn_path)
            recovered = isinstance(txn, dict)

            if cursor and not full:
                if self._has_schema_mismatch(
                    cursor=cursor,
                    chunk_schema_id=chunk_schema_id,
                    embedder_id=embedder_id,
                    config_chunker_id=config_chunker_id,
                    config_embedder_id=config_embedder_id,
                ):
                    raise MinSyncError("schema/embedder mismatch detected. Run sync with --full.", exit_code=1)

            if recovered:
                from_commit = self._coerce_optional_str(txn.get("from_commit"))
                to_commit = self._coerce_optional_str(txn.get("to_commit"))
                sync_token = str(txn.get("sync_token") or uuid.uuid4().hex)
                ref_name = str(txn.get("ref") or ref_name)
            else:
                from_commit = None if full else self._coerce_optional_str(cursor.get("last_synced_commit"))
                to_commit = None
                sync_token = uuid.uuid4().hex

            if to_commit is None:
                to_commit = self._resolve_commit(repo_root, ref_name)

            if verbose and recovered:
                print("Recovering from interrupted sync...")

            full_scan = full or from_commit is None
            if not recovered and not full_scan and from_commit == to_commit:
                return SyncResult(
                    from_commit=from_commit,
                    to_commit=to_commit,
                    files_processed=0,
                    files_processed_paths=[],
                    chunks_added=0,
                    chunks_updated=0,
                    chunks_deleted=0,
                    dry_run=dry_run,
                    already_up_to_date=True,
                    planned_files=[],
                    files_planned=0,
                    recovered=False,
                )

            planned_changes = self._collect_changes(
                repo_root=repo_root,
                from_commit=from_commit,
                to_commit=to_commit,
                full_scan=full_scan,
            )
            ignore_rules_changed = self._did_ignore_rules_change(planned_changes)
            ignore_matcher = self._load_ignore_matcher(repo_root=repo_root, to_commit=to_commit)
            planned_changes = self._apply_ignore_rules(planned_changes, ignore_matcher)
            if ignore_rules_changed and from_commit is not None and not full_scan:
                previous_ignore_matcher = self._load_ignore_matcher(repo_root=repo_root, to_commit=from_commit)
                re_included_paths = self._collect_reincluded_paths(
                    repo_root=repo_root,
                    to_commit=to_commit,
                    previous_ignore_matcher=previous_ignore_matcher,
                    current_ignore_matcher=ignore_matcher,
                )
                planned_changes = self._append_added_paths(changes=planned_changes, paths=re_included_paths)
            planned_paths = _unique_paths(planned_changes)

            if dry_run:
                return SyncResult(
                    from_commit=from_commit,
                    to_commit=to_commit,
                    files_processed=0,
                    files_processed_paths=[],
                    chunks_added=0,
                    chunks_updated=0,
                    chunks_deleted=0,
                    dry_run=True,
                    already_up_to_date=False,
                    planned_files=planned_paths,
                    files_planned=len(planned_paths),
                    recovered=recovered,
                )

            txn_payload = {
                "from_commit": from_commit,
                "to_commit": to_commit,
                "sync_token": sync_token,
                "repo_id": repo_id,
                "ref": ref_name,
            }
            self._write_json_atomic(txn_path, txn_payload)

            chunks_added = 0
            chunks_updated = 0
            chunks_deleted = 0

            if full and planned_changes:
                removed = vector_store.delete_by_filter(_repo_filter(repo_id, ref_name))
                chunks_deleted += int(removed or 0)

            for status, path in planned_changes:
                if status == "D":
                    removed = vector_store.delete_by_filter(_path_filter(repo_id, ref_name, path))
                    chunks_deleted += int(removed or 0)
                    continue

                file_text = self._read_file_at_commit(repo_root, to_commit, path)
                normalized = self._normalize_text(file_text, config.get("normalize") or {})
                chunks = chunker.chunk(normalized, path)
                docs = self._build_docs(
                    chunks=chunks,
                    repo_id=repo_id,
                    ref_name=ref_name,
                    path=path,
                    commit=to_commit,
                    sync_token=sync_token,
                    chunk_schema_id=chunk_schema_id,
                )

                doc_ids = [doc["id"] for doc in docs]
                existing = vector_store.fetch(doc_ids) if doc_ids else []
                existing_ids = {str(doc["id"]) for doc in existing}

                docs_to_upsert: list[dict[str, Any]] = []
                texts_to_embed: list[str] = []
                docs_to_update: list[dict[str, Any]] = []

                for doc in docs:
                    if doc["id"] in existing_ids:
                        docs_to_update.append({
                            "id": doc["id"],
                            "repo_id": doc["repo_id"],
                            "ref": doc["ref"],
                            "path": doc["path"],
                            "heading_path": doc["heading_path"],
                            "chunk_type": doc["chunk_type"],
                            "text": doc["text"],
                            "seen_token": doc["seen_token"],
                            "content_commit": doc["content_commit"],
                        })
                    else:
                        docs_to_upsert.append(doc)
                        texts_to_embed.append(doc["text"])

                if docs_to_upsert:
                    vectors = embedder.embed(texts_to_embed)
                    for doc, vector in zip(docs_to_upsert, vectors, strict=False):
                        doc["embedding"] = vector
                    vector_store.upsert(docs_to_upsert)
                    chunks_added += len(docs_to_upsert)

                if docs_to_update:
                    vector_store.update(docs_to_update)
                    chunks_updated += len(docs_to_update)

                removed = vector_store.delete_by_filter(_stale_path_filter(repo_id, ref_name, path, sync_token))
                chunks_deleted += int(removed or 0)

            if hasattr(vector_store, "flush"):
                vector_store.flush()
            self._persist_default_vector_store(repo_root)

            cursor_payload = {
                "last_synced_commit": to_commit,
                "chunk_schema_id": chunk_schema_id,
                "embedder_id": embedder_id,
                "config_chunker_id": config_chunker_id,
                "config_embedder_id": config_embedder_id,
                "ref": ref_name,
                "repo_id": repo_id,
            }
            self._write_json_atomic(cursor_path, cursor_payload)
            txn_path.unlink(missing_ok=True)

            return SyncResult(
                from_commit=from_commit,
                to_commit=to_commit,
                files_processed=len(planned_paths),
                files_processed_paths=planned_paths,
                chunks_added=chunks_added,
                chunks_updated=chunks_updated,
                chunks_deleted=chunks_deleted,
                dry_run=False,
                already_up_to_date=False,
                planned_files=planned_paths,
                files_planned=len(planned_paths),
                recovered=recovered,
            )

    def query(
        self,
        query_text: str,
        *,
        k: int = 10,
        ref: str | None = None,
        filter_expr: str | None = None,
        show_score: bool = False,
    ) -> list[QueryResult]:
        del show_score  # Formatting concern for CLI; Python API always carries score.

        query = query_text.strip()
        if not query:
            raise MinSyncError("query text is required.", exit_code=1)

        repo_root = self._resolve_git_root()
        config = self._load_config(repo_root)
        self._ensure_vectorstore(config, repo_root)
        cursor = self._read_json(repo_root / ".minsync" / "cursor.json") or {}
        if not self._coerce_optional_str(cursor.get("last_synced_commit")):
            warnings.warn("index is empty. Run minsync sync first.", RuntimeWarning, stacklevel=2)
            return []

        limit = max(int(k), 0)
        if limit == 0:
            return []

        embedder = self.embedder if self.embedder is not None else self._create_embedder_from_config(config)

        repo_id = str(config.get("repo_id") or cursor.get("repo_id") or self._resolve_repo_id(repo_root))
        ref_name = str(ref or cursor.get("ref") or config.get("ref") or DEFAULT_REF)
        base_filter = _repo_filter(repo_id, ref_name)
        normalized_filter = self._coerce_optional_str(filter_expr)
        effective_filter = f"{base_filter} AND {normalized_filter}" if normalized_filter else base_filter

        try:
            vectors = embedder.embed([query])
        except Exception as exc:
            raise MinSyncError(f"embedding failed: {exc}", exit_code=5) from exc

        if not vectors:
            return []

        doc_count_fn = getattr(self.vector_store, "doc_count", None)
        if callable(doc_count_fn):
            try:
                if int(doc_count_fn()) == 0:
                    warnings.warn("index is empty. Run minsync sync first.", RuntimeWarning, stacklevel=2)
                    return []
            except Exception:
                # Some store implementations may not provide count semantics.
                pass

        query_fn = getattr(self.vector_store, "query", None)
        if not callable(query_fn):
            raise MinSyncError("vector store does not support querying", exit_code=4)

        try:
            matches = query_fn(vectors[0], filter_expr=effective_filter, topk=limit)
        except Exception as exc:
            raise MinSyncError(f"vector store operation failed: {exc}", exit_code=4) from exc

        results: list[QueryResult] = []
        for match in matches:
            if not isinstance(match, dict):
                continue
            doc_id = str(match.get("doc_id") or match.get("id") or "")
            path = str(match.get("path") or "")
            text = str(match.get("text") or "")
            if not doc_id or not path or not text:
                continue
            score = float(match.get("score") or 0.0)
            results.append(
                QueryResult(
                    doc_id=doc_id,
                    path=path,
                    heading_path=str(match.get("heading_path") or ""),
                    chunk_type=str(match.get("chunk_type") or ""),
                    text=text,
                    score=score,
                    content_commit=str(match.get("content_commit") or ""),
                )
            )

        results.sort(key=lambda item: item.score, reverse=True)
        return results[:limit]

    def status(self) -> StatusResult:
        repo_root = self._resolve_git_root()
        config = self._load_config(repo_root)
        minsync_dir = repo_root / ".minsync"
        cursor = self._read_json(minsync_dir / "cursor.json") or {}
        pending_txn = self._read_json(minsync_dir / "txn.json")

        ref_name = str(cursor.get("ref") or config.get("ref") or DEFAULT_REF)
        current_head = self._resolve_commit(repo_root, ref_name)
        last_synced_commit = self._coerce_optional_str(cursor.get("last_synced_commit"))

        commits_behind = 0
        if isinstance(pending_txn, dict):
            state = "INTERRUPTED"
        elif not last_synced_commit:
            state = "NOT_SYNCED"
        elif last_synced_commit == current_head:
            state = "UP_TO_DATE"
        else:
            state = "OUT_OF_DATE"
            commits_behind = self._count_commits_between(repo_root, last_synced_commit, current_head)

        collection = str((config.get("collection") or {}).get("name") or "")
        chunker = str((config.get("chunker") or {}).get("id") or DEFAULT_CHUNKER_ID)
        embedder = str((config.get("embedder") or {}).get("id") or DEFAULT_EMBEDDER_ID)
        vectorstore = str((config.get("vectorstore") or {}).get("id") or DEFAULT_VECTORSTORE_ID)
        repo_id = str(config.get("repo_id") or cursor.get("repo_id") or self._resolve_repo_id(repo_root))

        return StatusResult(
            repo_id=repo_id,
            ref=ref_name,
            collection=collection,
            chunker=chunker,
            embedder=embedder,
            vectorstore=vectorstore,
            last_synced_commit=last_synced_commit,
            current_head=current_head,
            state=state,
            commits_behind=commits_behind,
            pending_txn=pending_txn if isinstance(pending_txn, dict) else None,
        )

    def check(self) -> CheckResult:
        repo_root = self._resolve_git_root()
        config = self._load_config(repo_root)
        self._ensure_vectorstore(config, repo_root)
        errors: list[str] = []

        ref_name = str(config.get("ref") or DEFAULT_REF)
        repo_id = str(config.get("repo_id") or self._resolve_repo_id(repo_root))

        git_details: dict[str, Any] = {"repo_id": repo_id, "ref": ref_name}
        try:
            git_details["head"] = self._resolve_commit(repo_root, ref_name)
            git_ok = True
        except MinSyncError as exc:
            git_ok = False
            errors.append(str(exc))
            git_details["error"] = str(exc)

        embedder_id = str((config.get("embedder") or {}).get("id") or DEFAULT_EMBEDDER_ID)
        embedder = self.embedder if self.embedder is not None else self._create_embedder_from_config(config)
        embedder_details: dict[str, Any] = {"id": embedder_id}
        try:
            started = time.perf_counter()
            vectors = embedder.embed(["minsync health check"])
            latency_ms = int((time.perf_counter() - started) * 1000)
            if not vectors:
                raise RuntimeError("embedder returned no vectors")
            dimension = len(vectors[0]) if isinstance(vectors[0], list) else 0
            if dimension <= 0:
                raise RuntimeError("embedder returned invalid vector dimension")
            embedder_ok = True
            embedder_details["dimension"] = dimension
            embedder_details["latency_ms"] = latency_ms
        except Exception as exc:
            embedder_ok = False
            message = f"embedder check failed: {exc}"
            errors.append(message)
            embedder_details["error"] = str(exc)

        vectorstore_id = str((config.get("vectorstore") or {}).get("id") or DEFAULT_VECTORSTORE_ID)
        vectorstore_details: dict[str, Any] = {"id": vectorstore_id}
        if not self._vector_store_injected and vectorstore_id != DEFAULT_VECTORSTORE_ID:
            vectorstore_ok = False
            message = _missing_dependency_message(vectorstore_id)
            errors.append(message)
            vectorstore_details["error"] = message
        else:
            try:
                vectorstore_details["doc_count"] = self._vector_store_doc_count()
                vectorstore_ok = True
            except Exception as exc:
                vectorstore_ok = False
                message = f"vectorstore check failed: {exc}"
                errors.append(message)
                vectorstore_details["error"] = str(exc)

        all_passed = git_ok and embedder_ok and vectorstore_ok
        return CheckResult(
            git_ok=git_ok,
            embedder_ok=embedder_ok,
            vectorstore_ok=vectorstore_ok,
            errors=errors,
            all_passed=all_passed,
            git=git_details,
            embedder=embedder_details,
            vectorstore=vectorstore_details,
        )

    def verify(
        self,
        *,
        ref: str | None = None,
        all: bool = False,
        fix: bool = False,
        sample: int | None = None,
    ) -> VerifyResult:
        report = self._verify_impl(ref=ref, verify_all=all, fix=fix, sample=sample)
        if fix and report.fixed:
            stabilized = self._verify_impl(ref=ref, verify_all=all, fix=False, sample=sample)
            return VerifyResult(
                all_passed=stabilized.all_passed,
                basic_checks=stabilized.basic_checks,
                file_checks=stabilized.file_checks,
                ignored_stale=stabilized.ignored_stale,
                fixed=True,
            )
        return report

    def _verify_impl(
        self,
        *,
        ref: str | None,
        verify_all: bool,
        fix: bool,
        sample: int | None,
    ) -> VerifyResult:
        repo_root = self._resolve_git_root()
        config = self._load_config(repo_root)
        self._ensure_vectorstore(config, repo_root)

        minsync_dir = repo_root / ".minsync"
        cursor = self._read_json(minsync_dir / "cursor.json") or {}
        pending_txn = self._read_json(minsync_dir / "txn.json")
        last_synced_commit = self._coerce_optional_str(cursor.get("last_synced_commit"))
        if not last_synced_commit:
            raise MinSyncError("never synced. Run minsync sync first.", exit_code=1)

        ref_name = str(ref or cursor.get("ref") or config.get("ref") or DEFAULT_REF)
        repo_id = str(config.get("repo_id") or cursor.get("repo_id") or self._resolve_repo_id(repo_root))
        config_chunker_id = str((config.get("chunker") or {}).get("id") or DEFAULT_CHUNKER_ID)
        config_embedder_id = str((config.get("embedder") or {}).get("id") or DEFAULT_EMBEDDER_ID)
        normalize = config.get("normalize") or {}

        chunker = self.chunker if self.chunker is not None else self._create_chunker_from_config(config)
        embedder = self.embedder if self.embedder is not None else self._create_embedder_from_config(config)
        chunk_schema_id = self._chunk_schema_id(chunker, config_chunker_id)
        embedder_id = self._embedder_id(embedder, config_embedder_id)

        basic_checks = {
            "cursor_valid": True,
            "cursor_commit_exists": self._git_commit_exists(repo_root, last_synced_commit),
            "no_pending_txn": not isinstance(pending_txn, dict),
            "schema_match": not self._has_schema_mismatch(
                cursor=cursor,
                chunk_schema_id=chunk_schema_id,
                embedder_id=embedder_id,
                config_chunker_id=config_chunker_id,
                config_embedder_id=config_embedder_id,
            ),
            "collection_alive": True,
        }

        try:
            self._vector_store_doc_count()
        except Exception:
            basic_checks["collection_alive"] = False

        fixed_any = False
        file_checks: list[dict[str, Any]] = []
        ignored_stale: list[str] = []
        repo_docs: list[dict[str, Any]] = []
        if basic_checks["collection_alive"]:
            repo_docs = self._vector_store_docs(repo_id=repo_id, ref_name=ref_name)

        docs_by_path: dict[str, list[dict[str, Any]]] = {}
        for doc in repo_docs:
            path = self._coerce_optional_str(doc.get("path"))
            if not path:
                continue
            docs_by_path.setdefault(path, []).append(doc)
        indexed_paths = sorted(docs_by_path.keys())

        ignore_matcher = self._load_worktree_ignore_matcher(repo_root)
        remaining_ignored_stale: list[str] = []
        for path in indexed_paths:
            if not ignore_matcher.matches(path):
                continue
            ignored_stale.append(path)
            if fix:
                removed = self.vector_store.delete_by_filter(_path_filter(repo_id, ref_name, path))
                if int(removed or 0) > 0:
                    fixed_any = True
                    continue
            remaining_ignored_stale.append(path)

        expected_paths: list[str] = []
        if basic_checks["cursor_commit_exists"]:
            expected_paths = self._tracked_paths_at_commit(repo_root, last_synced_commit)
            expected_paths = [
                path for path in expected_paths if not _is_internal_sync_path(path) and not ignore_matcher.matches(path)
            ]

        paths_to_check: list[str] = []
        if verify_all:
            paths_to_check = expected_paths
        else:
            effective_sample = sample if sample is not None else 10
            sample_size = max(int(effective_sample), 0)
            if sample_size > 0:
                paths_to_check = expected_paths[:sample_size]

        for path in paths_to_check:
            file_text = self._read_file_at_commit(repo_root, last_synced_commit, path)
            normalized = self._normalize_text(file_text, normalize)
            chunks = chunker.chunk(normalized, path)
            expected_docs = self._build_docs(
                chunks=chunks,
                repo_id=repo_id,
                ref_name=ref_name,
                path=path,
                commit=last_synced_commit,
                sync_token="verify",
                chunk_schema_id=chunk_schema_id,
            )
            expected_ids = {str(doc["id"]) for doc in expected_docs}
            fetched = self.vector_store.fetch(sorted(expected_ids)) if expected_ids else []
            found_ids = {str(doc.get("id") or "") for doc in fetched}
            path_docs = docs_by_path.get(path, [])
            actual_ids = {str(doc.get("id") or "") for doc in path_docs}

            missing_ids = sorted(doc_id for doc_id in (expected_ids - found_ids) if doc_id)
            stale_ids = sorted(doc_id for doc_id in (actual_ids - expected_ids) if doc_id)

            issues: list[str] = []
            if missing_ids:
                issues.append("MISSING")
            if stale_ids:
                issues.append("STALE")

            check_row: dict[str, Any] = {
                "path": path,
                "status": "OK" if not issues else "FAIL",
                "issues": issues,
                "missing_count": len(missing_ids),
                "stale_count": len(stale_ids),
            }
            if missing_ids:
                check_row["missing_ids"] = missing_ids
            if stale_ids:
                check_row["stale_ids"] = stale_ids

            if fix and issues:
                self._repair_path(
                    repo_root=repo_root,
                    config=config,
                    repo_id=repo_id,
                    ref_name=ref_name,
                    commit=last_synced_commit,
                    path=path,
                    chunker=chunker,
                    embedder=embedder,
                    chunk_schema_id=chunk_schema_id,
                )
                fixed_any = True
                check_row["status"] = "FIXED"
            file_checks.append(check_row)

        if verify_all:
            expected_set = set(expected_paths)
            ignored_set = set(ignored_stale)
            stale_deleted_paths = sorted(
                path for path in indexed_paths if path not in expected_set and path not in ignored_set
            )
            for path in stale_deleted_paths:
                stale_count = len(docs_by_path.get(path, []))
                row = {
                    "path": path,
                    "status": "FAIL",
                    "issues": ["STALE_DELETED"],
                    "missing_count": 0,
                    "stale_count": stale_count,
                }
                if fix:
                    removed = self.vector_store.delete_by_filter(_path_filter(repo_id, ref_name, path))
                    if int(removed or 0) > 0:
                        fixed_any = True
                        row["status"] = "FIXED"
                file_checks.append(row)

        basic_passed = all(bool(value) for value in basic_checks.values())
        file_checks_passed = all(str(row.get("status")) in {"OK", "FIXED"} for row in file_checks)
        all_passed = basic_passed and file_checks_passed and not remaining_ignored_stale

        if fix and fixed_any:
            self._persist_default_vector_store(repo_root)

        return VerifyResult(
            all_passed=all_passed,
            basic_checks=basic_checks,
            file_checks=file_checks,
            ignored_stale=remaining_ignored_stale,
            fixed=fixed_any,
        )

    def _repair_path(
        self,
        *,
        repo_root: Path,
        config: dict[str, Any],
        repo_id: str,
        ref_name: str,
        commit: str,
        path: str,
        chunker: Any,
        embedder: Any,
        chunk_schema_id: str = "",
    ) -> None:
        text = self._read_file_at_commit(repo_root, commit, path)
        normalized = self._normalize_text(text, config.get("normalize") or {})
        chunks = chunker.chunk(normalized, path)
        docs = self._build_docs(
            chunks=chunks,
            repo_id=repo_id,
            ref_name=ref_name,
            path=path,
            commit=commit,
            sync_token="verify-fix",
            chunk_schema_id=chunk_schema_id,
        )

        self.vector_store.delete_by_filter(_path_filter(repo_id, ref_name, path))
        if not docs:
            return

        vectors = embedder.embed([doc["text"] for doc in docs])
        for doc, vector in zip(docs, vectors, strict=False):
            doc["embedding"] = vector
        self.vector_store.upsert(docs)

    def _git_commit_exists(self, repo_root: Path, commit: str) -> bool:
        return self._git_repo().commit_exists(commit)

    def _count_commits_between(self, repo_root: Path, from_commit: str, to_commit: str) -> int:
        return self._git_repo().count_commits_between(from_commit, to_commit)

    def _tracked_paths_at_commit(self, repo_root: Path, commit: str) -> list[str]:
        try:
            return self._git_repo().list_tree_paths(commit)
        except (KeyError, pygit2.GitError) as exc:
            raise MinSyncGitError("failed to list tracked files for verification") from exc

    def _load_worktree_ignore_matcher(self, repo_root: Path) -> _IgnoreMatcher:
        ignore_path = repo_root / ".minsyncignore"
        if not ignore_path.exists():
            return _IgnoreMatcher.empty()
        return _IgnoreMatcher.from_text(ignore_path.read_text(encoding="utf-8"))

    def _vector_store_docs(self, *, repo_id: str, ref_name: str) -> list[dict[str, Any]]:
        docs: list[dict[str, Any]] = []
        get_all_docs = getattr(self.vector_store, "get_all_docs", None)
        if callable(get_all_docs):
            raw_docs = get_all_docs()
            if isinstance(raw_docs, list):
                for item in raw_docs:
                    if isinstance(item, dict):
                        docs.append(dict(item))
        elif isinstance(getattr(self.vector_store, "_docs", None), dict):
            raw_store = self.vector_store._docs
            if isinstance(raw_store, dict):
                for item in raw_store.values():
                    if isinstance(item, dict):
                        docs.append(dict(item))

        if not docs:
            fetch_by_filter = getattr(self.vector_store, "fetch_by_filter", None)
            if callable(fetch_by_filter):
                return fetch_by_filter(_repo_filter(repo_id, ref_name))

        filtered: list[dict[str, Any]] = []
        for doc in docs:
            if str(doc.get("repo_id") or "") != repo_id:
                continue
            if str(doc.get("ref") or "") != ref_name:
                continue
            filtered.append(doc)
        return filtered

    def _default_vector_store_path(self, repo_root: Path) -> Path:
        return repo_root / ".minsync" / "vector_store.json"

    def _load_default_vector_store(self, repo_root: Path) -> None:
        if self._vector_store_injected:
            return
        if not isinstance(self.vector_store, _InMemoryVectorStore):
            return

        payload = self._read_json(self._default_vector_store_path(repo_root)) or {}
        raw_docs = payload.get("docs") if isinstance(payload, dict) else None

        loaded_docs: dict[str, dict[str, Any]] = {}
        if isinstance(raw_docs, list):
            for item in raw_docs:
                if not isinstance(item, dict):
                    continue
                doc_id = self._coerce_optional_str(item.get("id"))
                if not doc_id:
                    continue
                loaded_docs[doc_id] = dict(item)

        self.vector_store._docs = loaded_docs

    def _persist_default_vector_store(self, repo_root: Path) -> None:
        if self._vector_store_injected:
            return
        if not isinstance(self.vector_store, _InMemoryVectorStore):
            return

        docs = self.vector_store.get_all_docs()
        docs.sort(key=lambda item: str(item.get("id") or ""))
        self._write_json_atomic(self._default_vector_store_path(repo_root), {"docs": docs})

    def _vector_store_doc_count(self) -> int:
        doc_count_fn = getattr(self.vector_store, "doc_count", None)
        if callable(doc_count_fn):
            return int(doc_count_fn())
        get_all_docs = getattr(self.vector_store, "get_all_docs", None)
        if callable(get_all_docs):
            docs = get_all_docs()
            if isinstance(docs, list):
                return len(docs)
        if isinstance(getattr(self.vector_store, "_docs", None), dict):
            raw_store = self.vector_store._docs
            if isinstance(raw_store, dict):
                return len(raw_store)
        return 0

    def _resolve_git_root(self) -> Path:
        return self._git_repo().workdir

    def _resolve_repo_id(self, repo_root: Path) -> str:
        try:
            return self._git_repo().resolve_repo_id()
        except (KeyError, pygit2.GitError) as exc:
            raise MinSyncGitError("repository has no commits. Create at least one commit first.") from exc

    def _resolve_commit(self, repo_root: Path, ref_name: str) -> str:
        try:
            return self._git_repo().resolve_commit(ref_name)
        except (KeyError, pygit2.GitError) as exc:
            raise MinSyncGitError(f"unable to resolve git ref: {ref_name}") from exc

    def _load_config(self, repo_root: Path) -> dict[str, Any]:
        config_path = repo_root / ".minsync" / "config.yaml"
        if not config_path.exists():
            raise MinSyncError("not initialized. Run minsync init first.", exit_code=1)

        data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            raise MinSyncError("invalid config file .minsync/config.yaml", exit_code=1)
        return data

    def _read_json(self, path: Path) -> dict[str, Any] | None:
        if not path.exists():
            return None
        text = path.read_text(encoding="utf-8")
        data = json.loads(text)
        if not isinstance(data, dict):
            return None
        return data

    def _write_json_atomic(self, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
                handle.write("\n")
            os.rename(tmp_name, path)
        except Exception:
            Path(tmp_name).unlink(missing_ok=True)
            raise

    @contextmanager
    def _acquire_lock(self, lock_path: Path, *, wait: bool) -> Iterator[None]:
        while True:
            try:
                fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            except FileExistsError:
                if self._reclaim_stale_lock(lock_path):
                    continue
                if not wait:
                    raise MinSyncError("another sync is in progress", exit_code=3) from None
                time.sleep(0.1)
                continue
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(str(os.getpid()))
            break

        try:
            yield
        finally:
            lock_path.unlink(missing_ok=True)

    def _reclaim_stale_lock(self, lock_path: Path) -> bool:
        owner_pid = self._read_lock_owner_pid(lock_path)
        if owner_pid is not None and self._is_process_alive(owner_pid):
            return False
        try:
            lock_path.unlink(missing_ok=True)
        except OSError:
            return False
        return True

    def _read_lock_owner_pid(self, lock_path: Path) -> int | None:
        try:
            payload = lock_path.read_text(encoding="utf-8").strip()
        except OSError:
            return None
        if not payload:
            return None
        if not payload.isdigit():
            return None
        pid = int(payload)
        if pid <= 0:
            return None
        return pid

    def _is_process_alive(self, pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except OSError:
            return True
        return True

    def _collect_changes(
        self,
        *,
        repo_root: Path,
        from_commit: str | None,
        to_commit: str,
        full_scan: bool,
    ) -> list[tuple[str, str]]:
        git = self._git_repo()
        if full_scan:
            try:
                paths = git.list_tree_paths(to_commit)
            except (KeyError, pygit2.GitError) as exc:
                raise MinSyncGitError("failed to list tracked files for sync target") from exc
            return [("A", p) for p in paths]

        if not from_commit:
            return []

        try:
            return git.diff_name_status(from_commit, to_commit)
        except (KeyError, pygit2.GitError) as exc:
            raise MinSyncGitError("failed to compute git diff for sync") from exc

    def _load_ignore_matcher(self, *, repo_root: Path, to_commit: str) -> _IgnoreMatcher:
        text = self._git_repo().read_file_at_commit_or_none(to_commit, ".minsyncignore")
        if text is None:
            return _IgnoreMatcher.empty()
        return _IgnoreMatcher.from_text(text)

    def _did_ignore_rules_change(self, changes: list[tuple[str, str]]) -> bool:
        return any(path == ".minsyncignore" for _status, path in changes)

    def _collect_reincluded_paths(
        self,
        *,
        repo_root: Path,
        to_commit: str,
        previous_ignore_matcher: _IgnoreMatcher,
        current_ignore_matcher: _IgnoreMatcher,
    ) -> list[str]:
        re_included: list[str] = []
        for path in self._tracked_paths_at_commit(repo_root, to_commit):
            if _is_internal_sync_path(path):
                continue
            if current_ignore_matcher.matches(path):
                continue
            if not previous_ignore_matcher.matches(path):
                continue
            re_included.append(path)
        return re_included

    def _append_added_paths(
        self,
        *,
        changes: list[tuple[str, str]],
        paths: list[str],
    ) -> list[tuple[str, str]]:
        if not paths:
            return changes

        existing_paths = {path for _status, path in changes}
        augmented = list(changes)
        for path in paths:
            if path in existing_paths:
                continue
            augmented.append(("A", path))
        return augmented

    def _apply_ignore_rules(
        self,
        changes: list[tuple[str, str]],
        ignore_matcher: _IgnoreMatcher,
    ) -> list[tuple[str, str]]:
        filtered: list[tuple[str, str]] = []
        for status, path in changes:
            if _is_internal_sync_path(path):
                continue
            if status in {"A", "M"} and ignore_matcher.matches(path):
                continue
            filtered.append((status, path))
        return filtered

    def _read_file_at_commit(self, repo_root: Path, commit: str, path: str) -> str:
        try:
            return self._git_repo().read_file_at_commit(commit, path)
        except (KeyError, pygit2.GitError) as exc:
            raise MinSyncGitError(f"failed to read file content from git: {path}") from exc

    def _normalize_text(self, text: str, normalize: dict[str, Any]) -> str:
        normalized = text

        if normalize.get("normalize_newlines", True):
            normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")

        if normalize.get("strip_frontmatter", False):
            normalized = re.sub(r"\A---\n.*?\n---\n?", "", normalized, flags=re.DOTALL)

        if normalize.get("strip_trailing_whitespace", True):
            normalized = "\n".join(line.rstrip() for line in normalized.split("\n"))

        if normalize.get("collapse_whitespace", False):
            normalized = re.sub(r"[ \t]+", " ", normalized)

        return normalized

    def _build_docs(
        self,
        *,
        chunks: list[Any],
        repo_id: str,
        ref_name: str,
        path: str,
        commit: str,
        sync_token: str,
        chunk_schema_id: str = "",
    ) -> list[dict[str, Any]]:
        docs: list[dict[str, Any]] = []
        dup_counter: dict[tuple[str, str], int] = {}

        for chunk in chunks:
            chunk_text = str(getattr(chunk, "text", ""))
            if not chunk_text.strip():
                continue
            chunk_type = str(getattr(chunk, "chunk_type", "child"))
            heading_path = str(getattr(chunk, "heading_path", ""))
            content_hash = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()

            dup_key = (content_hash, heading_path)
            dup_index = dup_counter.get(dup_key, 0)
            dup_counter[dup_key] = dup_index + 1

            doc_id = self._doc_id(
                repo_id=repo_id,
                ref_name=ref_name,
                path=path,
                schema_id=chunk_schema_id,
                chunk_type=chunk_type,
                heading_path=heading_path,
                content_hash=content_hash,
                dup_index=dup_index,
            )
            docs.append({
                "id": doc_id,
                "repo_id": repo_id,
                "ref": ref_name,
                "path": path,
                "heading_path": heading_path,
                "chunk_type": chunk_type,
                "text": chunk_text,
                "content_commit": commit,
                "seen_token": sync_token,
                "chunk_schema_id": chunk_schema_id,
                "content_hash": content_hash,
            })
        return docs

    def _doc_id(
        self,
        *,
        repo_id: str,
        ref_name: str,
        path: str,
        schema_id: str,
        chunk_type: str,
        heading_path: str,
        content_hash: str,
        dup_index: int,
    ) -> str:
        payload = "\0".join([
            repo_id,
            ref_name,
            path,
            schema_id,
            chunk_type,
            heading_path,
            content_hash,
            str(dup_index),
        ])
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _create_chunker_from_config(self, config: dict[str, Any]) -> Any:
        """Create a chunker from config, falling back to _DefaultChunker on error."""
        try:
            from minsync.factory import create_chunker

            return create_chunker(config)
        except (MinSyncError, Exception):
            config_chunker_id = str((config.get("chunker") or {}).get("id") or DEFAULT_CHUNKER_ID)
            return _DefaultChunker(config_chunker_id)

    def _create_embedder_from_config(self, config: dict[str, Any]) -> Any:
        """Create an embedder from config, falling back to _DefaultEmbedder on error."""
        try:
            from minsync.factory import create_embedder

            return create_embedder(config)
        except (MinSyncError, Exception):
            config_embedder_id = str((config.get("embedder") or {}).get("id") or DEFAULT_EMBEDDER_ID)
            return _DefaultEmbedder(config_embedder_id)

    def _create_vectorstore_from_config(self, config: dict[str, Any], repo_root: Path) -> Any:
        """Create a vector store from config, resolving relative paths."""
        try:
            from minsync.factory import create_vectorstore

            resolved = dict(config)
            vs = dict(resolved.get("vectorstore") or {})
            coll = vs.get("collection") or {}
            rel = coll.get("path", "")
            if rel and not Path(rel).is_absolute():
                vs["collection"] = {**coll, "path": str(repo_root / rel)}
                resolved["vectorstore"] = vs
            return create_vectorstore(resolved)
        except (MinSyncError, ImportError, Exception):
            return _InMemoryVectorStore()

    def _ensure_vectorstore(self, config: dict[str, Any], repo_root: Path) -> None:
        """Upgrade to a real vectorstore if available, then load persistence."""
        if not self._vector_store_injected and isinstance(self.vector_store, _InMemoryVectorStore):
            self.vector_store = self._create_vectorstore_from_config(config, repo_root)
        self._load_default_vector_store(repo_root)

    def _chunk_schema_id(self, chunker: Any, fallback: str) -> str:
        schema_fn = getattr(chunker, "schema_id", None)
        if callable(schema_fn):
            return str(schema_fn())
        return fallback

    def _embedder_id(self, embedder: Any, fallback: str) -> str:
        embedder_id_fn = getattr(embedder, "id", None)
        if callable(embedder_id_fn):
            return str(embedder_id_fn())
        return fallback

    def _has_schema_mismatch(
        self,
        *,
        cursor: dict[str, Any],
        chunk_schema_id: str,
        embedder_id: str,
        config_chunker_id: str,
        config_embedder_id: str,
    ) -> bool:
        prev_schema = self._coerce_optional_str(cursor.get("chunk_schema_id"))
        prev_embedder = self._coerce_optional_str(cursor.get("embedder_id"))
        prev_config_schema = self._coerce_optional_str(cursor.get("config_chunker_id"))
        prev_config_embedder = self._coerce_optional_str(cursor.get("config_embedder_id"))

        runtime_schema_mismatch = prev_schema is not None and prev_schema != chunk_schema_id
        runtime_embedder_mismatch = prev_embedder is not None and prev_embedder != embedder_id
        config_schema_mismatch = prev_config_schema is not None and prev_config_schema != config_chunker_id
        config_embedder_mismatch = prev_config_embedder is not None and prev_config_embedder != config_embedder_id

        return (
            runtime_schema_mismatch or runtime_embedder_mismatch or config_schema_mismatch or config_embedder_mismatch
        )

    def _coerce_optional_str(self, value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    def _build_config(self, *, repo_id: str, collection: str, embedder: str, chunker: str) -> dict[str, Any]:
        return {
            "version": 1,
            "repo_id": repo_id,
            "ref": DEFAULT_REF,
            "collection": {
                "name": collection,
                "path": ".minsync/zvec_data",
            },
            "chunker": {
                "id": chunker,
                "options": {
                    "max_chunk_size": 1000,
                    "overlap": 100,
                },
            },
            "embedder": {
                "id": embedder,
                "batch_size": 64,
            },
            "vectorstore": {
                "id": DEFAULT_VECTORSTORE_ID,
                "options": {},
            },
            "normalize": {
                "strip_trailing_whitespace": True,
                "normalize_newlines": True,
                "collapse_whitespace": False,
                "strip_frontmatter": False,
            },
        }

    def _remove_path(self, path: Path) -> None:
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
            return
        path.unlink(missing_ok=True)


def _status_text(state: str) -> str:
    mapping = {
        "UP_TO_DATE": "UP TO DATE",
        "OUT_OF_DATE": "OUT OF DATE",
        "NOT_SYNCED": "NOT SYNCED",
        "INTERRUPTED": "INTERRUPTED",
    }
    return mapping.get(state, state)


def _short_commit(commit: str | None) -> str:
    if not commit:
        return "?"
    return str(commit)[:8]


def _coerce_text(payload: dict[str, Any] | None, key: str) -> str | None:
    if not isinstance(payload, dict):
        return None
    value = payload.get(key)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _missing_dependency_message(vectorstore_id: str) -> str:
    if vectorstore_id == "weaviate":
        return (
            "'weaviate-client' package not found for configured vectorstore 'weaviate'. "
            "Install with: pip install weaviate-client langchain-weaviate"
        )
    if vectorstore_id == "chroma":
        return (
            "'chromadb' package not found for configured vectorstore 'chroma'. "
            "Install with: pip install chromadb langchain-chroma"
        )
    if vectorstore_id == "qdrant":
        return (
            "'qdrant-client' package not found for configured vectorstore 'qdrant'. "
            "Install with: pip install qdrant-client langchain-qdrant"
        )
    return (
        f"vectorstore '{vectorstore_id}' is configured but not available in this environment. "
        "Install the required package(s) and retry."
    )


def _repo_filter(repo_id: str, ref_name: str) -> str:
    return f"repo_id == '{repo_id}' AND ref == '{ref_name}'"


def _path_filter(repo_id: str, ref_name: str, path: str) -> str:
    return f"repo_id == '{repo_id}' AND ref == '{ref_name}' AND path == '{path}'"


def _stale_path_filter(repo_id: str, ref_name: str, path: str, sync_token: str) -> str:
    return f"repo_id == '{repo_id}' AND ref == '{ref_name}' AND path == '{path}' AND seen_token != '{sync_token}'"


def _matches_filter(doc: dict[str, Any], filter_expr: str) -> bool:
    clauses = [clause.strip() for clause in filter_expr.split(" AND ")]
    for clause in clauses:
        if "!=" in clause:
            key, val = clause.split("!=", 1)
            key = key.strip()
            val = val.strip().strip("'\"")
            if str(doc.get(key, "")) == val:
                return False
            continue
        if "==" in clause:
            key, val = clause.split("==", 1)
            key = key.strip()
            val = val.strip().strip("'\"")
            if str(doc.get(key, "")) != val:
                return False
    return True


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        return 0.0
    dot = sum(lval * rval for lval, rval in zip(left, right, strict=False))
    left_norm = sum(value * value for value in left) ** 0.5
    right_norm = sum(value * value for value in right) ** 0.5
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def _unique_paths(changes: list[tuple[str, str]]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for _status, path in changes:
        if path in seen:
            continue
        seen.add(path)
        ordered.append(path)
    return ordered


def _is_internal_sync_path(path: str) -> bool:
    normalized = path.replace("\\", "/")
    if normalized.startswith("./"):
        normalized = normalized[2:]
    return (
        normalized == ".minsyncignore"
        or normalized == ".minsync"
        or normalized.startswith(".minsync/")
        or normalized == ".git"
        or normalized.startswith(".git/")
    )


def _compile_ignore_rule(raw_pattern: str) -> _IgnoreRule:
    pattern = raw_pattern
    negated = False
    if pattern.startswith("!"):
        negated = True
        pattern = pattern[1:]

    anchored = False
    if pattern.startswith("/"):
        anchored = True
        pattern = pattern[1:]

    directory_only = pattern.endswith("/")
    if directory_only:
        pattern = pattern[:-1]

    has_slash = "/" in pattern
    body = _glob_to_regex(pattern)

    if directory_only:
        if has_slash or anchored:
            regex = re.compile(f"^{body}(?:/.*)?$")
        else:
            regex = re.compile(f"^(?:.*/)?{body}(?:/.*)?$")
    elif has_slash or anchored:
        regex = re.compile(f"^{body}(?:/.*)?$")
    else:
        regex = re.compile(f"^(?:.*/)?{body}(?:/.*)?$")

    return _IgnoreRule(regex=regex, negated=negated)


def _glob_to_regex(pattern: str) -> str:
    pieces: list[str] = []
    idx = 0
    while idx < len(pattern):
        char = pattern[idx]
        if char == "*":
            if idx + 1 < len(pattern) and pattern[idx + 1] == "*":
                idx += 1
                if idx + 1 < len(pattern) and pattern[idx + 1] == "/":
                    idx += 1
                    pieces.append("(?:.*/)?")
                else:
                    pieces.append(".*")
            else:
                pieces.append("[^/]*")
        elif char == "?":
            pieces.append("[^/]")
        else:
            pieces.append(re.escape(char))
        idx += 1
    return "".join(pieces)
