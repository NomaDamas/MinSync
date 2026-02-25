"""Native git backend using pygit2 (libgit2 bindings)."""

from __future__ import annotations

from pathlib import Path

import pygit2


class GitRepo:
    """Thin wrapper around pygit2.Repository exposing only the operations MinSync needs."""

    def __init__(self, repo: pygit2.Repository) -> None:
        self._repo = repo

    @classmethod
    def discover(cls, path: Path) -> GitRepo:
        """Find and open the git repository containing *path*.

        Raises ``KeyError`` when *path* is not inside a git work-tree.
        """
        repo_path = pygit2.discover_repository(str(path))
        if repo_path is None:
            raise KeyError(path)
        return cls(pygit2.Repository(repo_path))

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def workdir(self) -> Path:
        """Return the resolved repository work-tree root."""
        wd = self._repo.workdir
        if wd is None:
            raise KeyError("bare repository has no workdir")
        return Path(wd).resolve()

    # ------------------------------------------------------------------
    # Commit helpers
    # ------------------------------------------------------------------

    def resolve_repo_id(self) -> str:
        """Return the OID of the root commit (max-parents=0).

        When there are multiple roots the *last* one (sorted chronologically
        by the walker, same as ``git rev-list --max-parents=0 HEAD``) is returned.
        """
        walker = self._repo.walk(self._repo.head.target, pygit2.enums.SortMode.TOPOLOGICAL)
        root_oid: str | None = None
        for commit in walker:
            if not commit.parents:
                root_oid = str(commit.id)
        if root_oid is None:
            raise KeyError("no root commit found")
        return root_oid

    def resolve_commit(self, ref: str) -> str:
        """Resolve *ref* (branch name, tag, SHA) to a full hex commit SHA."""
        obj = self._repo.revparse_single(ref)
        commit = obj.peel(pygit2.Commit)
        return str(commit.id)

    def commit_exists(self, sha: str) -> bool:
        """Return True if *sha* identifies a valid commit object."""
        try:
            oid = pygit2.Oid(hex=sha)
            obj = self._repo.get(oid)
            if obj is None:
                return False
            obj.peel(pygit2.Commit)
        except (ValueError, pygit2.GitError):
            return False
        else:
            return True

    def count_commits_between(self, from_sha: str, to_sha: str) -> int:
        """Return the number of commits in ``from_sha..to_sha`` (same semantics as
        ``git rev-list --count from..to``)."""
        try:
            from_oid = pygit2.Oid(hex=from_sha)
            to_oid = pygit2.Oid(hex=to_sha)
            ahead, _behind = self._repo.ahead_behind(to_oid, from_oid)
            return max(ahead, 0)
        except (ValueError, pygit2.GitError):
            return 0

    # ------------------------------------------------------------------
    # Tree / file access
    # ------------------------------------------------------------------

    def list_tree_paths(self, sha: str) -> list[str]:
        """Return every blob path reachable from the tree of *sha*, sorted."""
        commit = self._repo.revparse_single(sha).peel(pygit2.Commit)
        tree = commit.tree
        paths: list[str] = []
        stack: list[tuple[str, pygit2.Tree]] = [("", tree)]
        while stack:
            prefix, current_tree = stack.pop()
            for entry in current_tree:
                full = f"{prefix}{entry.name}" if not prefix else f"{prefix}/{entry.name}"
                if entry.type_str == "tree":
                    subtree = self._repo.get(entry.id)
                    if subtree is not None and isinstance(subtree, pygit2.Tree):
                        stack.append((full, subtree))
                else:
                    paths.append(full)
        return sorted(paths)

    def read_file_at_commit(self, sha: str, path: str) -> str:
        """Read *path* from the tree of commit *sha*; raise on missing."""
        commit = self._repo.revparse_single(sha).peel(pygit2.Commit)
        tree = commit.tree
        entry = tree[path]
        obj = self._repo.get(entry.id)
        if obj is None or obj.type != pygit2.GIT_OBJECT_BLOB:
            raise KeyError(path)
        blob: pygit2.Blob = obj.peel(pygit2.Blob)
        return blob.data.decode("utf-8", errors="replace")

    def read_file_at_commit_or_none(self, sha: str, path: str) -> str | None:
        """Like :meth:`read_file_at_commit` but returns ``None`` when the path
        does not exist."""
        try:
            return self.read_file_at_commit(sha, path)
        except (KeyError, pygit2.GitError):
            return None

    # ------------------------------------------------------------------
    # Diff
    # ------------------------------------------------------------------

    def diff_name_status(self, from_sha: str, to_sha: str) -> list[tuple[str, str]]:
        """Return ``[(status, path), ...]`` matching ``git diff --name-status --find-renames``."""
        from_commit = self._repo.revparse_single(from_sha).peel(pygit2.Commit)
        to_commit = self._repo.revparse_single(to_sha).peel(pygit2.Commit)
        diff = self._repo.diff(a=from_commit, b=to_commit)
        diff.find_similar(flags=pygit2.enums.DiffFind.FIND_RENAMES)

        changes: list[tuple[str, str]] = []
        for delta in diff.deltas:
            status = delta.status
            if status == pygit2.GIT_DELTA_RENAMED:
                old_path = delta.old_file.path
                new_path = delta.new_file.path
                if old_path:
                    changes.append(("D", old_path))
                if new_path:
                    changes.append(("A", new_path))
            elif status == pygit2.GIT_DELTA_ADDED:
                changes.append(("A", delta.new_file.path))
            elif status == pygit2.GIT_DELTA_DELETED:
                changes.append(("D", delta.old_file.path))
            else:
                # Modified, copied, etc. → treat as modification
                changes.append(("M", delta.new_file.path))
        return changes
