#!/usr/bin/env python3
"""Reproducible benchmark harness for MinSync issue #71.

Compares a *baseline* MinSync binary against a *candidate* binary on the exact
condition from issue #71: a LanceDB store left with thousands of fragments /
versions by sync runs that were killed before the final `flush()`. Every chunk
is its own LanceDB commit, so each file costs seconds once the store is
fragmented.

No real embedding model is used: `fake_tei.py` serves deterministic,
L2-normalised 16-d vectors (sha256-derived), so both binaries produce
identical indexes for identical workloads.

Usage
-----
    python3 scripts/bench_issue71/bench.py --scenario all \
        --baseline /tmp/bench/minsync-main \
        --candidate target/release/minsync \
        --out scripts/bench_issue71/results.md

Scenarios
---------
    initial      fresh initial sync of N files
    incremental  baseline initial sync, then modify 30% of files, incremental sync
    fragmented   build a >=K-fragment store with killed baseline runs, then let
                 both binaries finish on separate copies (the issue's condition)
    resume       candidate only: kill mid-sync, rerun; check work is not redone
    all          initial + incremental + fragmented + resume

Standard library only (python3).
"""

import argparse
import json
import os
import random
import re
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
FAKE_TEI = HERE / "fake_tei.py"

DEFAULT_BASELINE = "/tmp/bench/minsync-main"
DEFAULT_CANDIDATE = "target/release/minsync"

# Vocabulary for the generated corpus. Every paragraph contains the token
# "lorem" so a BM25 query for it matches the whole store (used for sanity +
# row counting via the vector path, which needs no FTS index).
WORDS = (
    "lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod "
    "tempor incididunt ut labore et dolore magna aliqua enim ad minim veniam "
    "quis nostrud exercitation ullamco laboris nisi aliquip ex ea commodo"
).split()

# A different vocabulary for mutations, so an edited paragraph is textually new
# while keeping the exact same byte length (chunk boundaries stay stable).
MUT_WORDS = (
    "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi "
    "omicron pi rho sigma tau upsilon phi chi psi omega"
).split()

PARAGRAPHS_PER_FILE = 44
QUERY_K = 10_000_000


def rng(*parts) -> random.Random:
    """Deterministic RNG (python 3.14 rejects tuple seeds)."""
    return random.Random(":".join(str(part) for part in parts))


# --------------------------------------------------------------------------- #
# fake TEI server
# --------------------------------------------------------------------------- #
def free_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


class FakeServer:
    def __init__(self, dim: int, latency_ms: float, port=None):
        self.dim = dim
        self.latency_ms = latency_ms
        self.port = port or free_port()
        self.proc = None

    def start(self, timeout=20.0) -> None:
        cmd = [
            sys.executable,
            str(FAKE_TEI),
            "--port",
            str(self.port),
            "--dim",
            str(self.dim),
            "--latency-ms",
            str(self.latency_ms),
        ]
        self.proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                out = self.proc.stdout.read() if self.proc.stdout else ""
                raise RuntimeError(f"fake_tei died on startup:\n{out}")
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{self.port}/health", timeout=0.5
                ) as resp:
                    if resp.status == 200:
                        return
            except Exception:
                time.sleep(0.05)
        raise RuntimeError("fake_tei did not become ready in time")

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def stats(self) -> dict:
        with urllib.request.urlopen(f"{self.base_url}/stats", timeout=2) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def inputs(self) -> int:
        try:
            return int(self.stats().get("inputs", 0))
        except Exception:
            return 0

    def reset(self) -> None:
        req = urllib.request.Request(f"{self.base_url}/reset", data=b"", method="POST")
        with urllib.request.urlopen(req, timeout=2) as resp:
            resp.read()

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()


# --------------------------------------------------------------------------- #
# corpus generation + mutation
# --------------------------------------------------------------------------- #
def make_docs(n: int, seed: int) -> list:
    """Deterministic corpus description: list of {name, paras}."""
    docs = []
    for i in range(n):
        rnd = rng("doc", seed, i)
        paras = []
        for _ in range(PARAGRAPHS_PER_FILE):
            nwords = rnd.randint(80, 120)
            paras.append("lorem " + " ".join(rnd.choice(WORDS) for _ in range(nwords)))
        docs.append({"name": f"doc_{i:04d}.md", "paras": paras})
    return docs


def render_doc(doc: dict) -> str:
    return f"# Document {doc['name']}\n" + "\n".join(doc["paras"]) + "\n"


def write_corpus(ws: Path, docs: list) -> None:
    ws.mkdir(parents=True, exist_ok=True)
    for doc in docs:
        (ws / doc["name"]).write_text(render_doc(doc))


def _same_length_replacement(old: str, rnd: random.Random) -> str:
    new = ""
    while len(new) < len(old):
        new += rnd.choice(MUT_WORDS) + " "
    new = new[: len(old)]
    # keep it valid and not byte-identical to the old paragraph
    if new.endswith(" "):
        new = new[:-1] + "x"
    if new == old:
        new = new[:-1] + ("y" if new[-1] != "y" else "z")
    assert len(new) == len(old), (len(old), len(new))
    return new


def mutate_files(
    ws: Path,
    n: int,
    seed: int,
    tag,
    frac: float,
    add: int = 0,
    delete: int = 0,
) -> dict:
    """Edit ONE paragraph (same byte length) in `frac` of the files.

    Same-length edits keep every chunk boundary stable, so all chunks except
    the one holding the edited paragraph keep their doc IDs -> the
    update/reuse path, not a full re-embed.
    """
    rnd = rng("mutate", seed, tag)
    count = max(1, int(round(n * frac))) if frac > 0 else 0
    indices = sorted(rnd.sample(range(n), min(count, n)))
    existing = [p for p in ws.glob("doc_*.md")]
    names = sorted(p.name for p in existing)

    mutated = []
    for i in indices:
        name = f"doc_{i:04d}.md"
        path = ws / name
        if not path.exists():
            continue
        lines = path.read_text().split("\n")
        candidates = [idx for idx, line in enumerate(lines) if line and idx > 0]
        if not candidates:
            continue
        para_idx = rnd.choice(candidates)
        lines[para_idx] = _same_length_replacement(lines[para_idx], rnd)
        path.write_text("\n".join(lines))
        mutated.append(name)

    added, deleted = [], []
    for j in range(add):
        name = f"extra_{tag}_{j:03d}.md"
        sub = rng("add", seed, tag, j)
        paras = []
        for _ in range(PARAGRAPHS_PER_FILE):
            paras.append(
                "lorem "
                + " ".join(sub.choice(WORDS) for _ in range(sub.randint(80, 120)))
            )
        (ws / name).write_text(f"# Document {name}\n" + "\n".join(paras) + "\n")
        added.append(name)
    for j in range(delete):
        idx = len(names) - 1 - j
        if idx < 0:
            break
        victim = ws / names[idx]
        if victim.exists():
            victim.unlink()
            deleted.append(names[idx])

    return {"mutated": len(mutated), "added": added, "deleted": deleted}


# --------------------------------------------------------------------------- #
# workspace helpers
# --------------------------------------------------------------------------- #
def patch_config(ws: Path, port: int, dim: int, batch_size: int, max_concurrent: int,
                 ann_build_threshold: int, ann_optimize_threshold: int) -> None:
    path = ws / ".minsync" / "config.toml"
    out = []
    section = None
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("["):
            section = stripped
        if section == "[embedder]" and stripped.startswith("id ="):
            out.append(line)
            out.append(f'base_url = "http://127.0.0.1:{port}"')
            continue
        if section == "[embedder]" and stripped.startswith("batch_size ="):
            out.append(f"batch_size = {batch_size}")
            continue
        if section == "[embedder]" and stripped.startswith("max_concurrent ="):
            out.append(f"max_concurrent = {max_concurrent}")
            continue
        if section == "[vectorstore.options]" and stripped.startswith("dimension ="):
            out.append(f"dimension = {dim}")
            # Keep the vector search exhaustive/flat so row counts are exact
            # (an IVF-HNSW index makes large-k queries approximate) and so ANN
            # maintenance cost, which varies run-to-run, does not pollute the
            # sync timing. Raise via --ann-* flags to restore defaults.
            out.append(f"index_build_threshold = {ann_build_threshold}")
            out.append(f"index_optimize_delta_threshold = {ann_optimize_threshold}")
            continue
        out.append(line)
    path.write_text("\n".join(out) + "\n")


def init_workspace(ws: Path, binary: str, server: FakeServer, cfg: dict) -> None:
    ws.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [binary, "init", "--embedder", "tei:bench/fake", "--chunker", "recursive"],
        cwd=ws,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    patch_config(
        ws, server.port, cfg["dim"], cfg["batch_size"], cfg["max_concurrent"],
        cfg["ann_build_threshold"], cfg["ann_optimize_threshold"],
    )
    assert (ws / ".minsync" / "config.toml").exists()


def copy_ws(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, symlinks=True)


def store_stats(ws: Path) -> dict:
    base = ws / ".minsync" / "store" / "documents.lance"
    stats = {}
    for name in ("_versions", "data", "_deletions"):
        d = base / name
        stats[name] = sum(1 for _ in d.iterdir()) if d.is_dir() else 0
    size = 0
    if base.is_dir():
        for root, _dirs, files in os.walk(base):
            for f in files:
                try:
                    size += (Path(root) / f).stat().st_size
                except OSError:
                    pass
    stats["size_bytes"] = size
    stats["fragments"] = stats["data"]
    stats["versions"] = stats["_versions"]
    stats["max_rail"] = max(stats["data"], stats["_versions"])
    return stats


def read_source_id(ws: Path) -> str:
    cfg = (ws / ".minsync" / "config.toml").read_text()
    return re.search(r'source_id = "([^"]+)"', cfg).group(1)


def write_probe_cursor(ws: Path) -> Path:
    """Create a cursor.json if absent so `minsync query` can read the store.

    Never done before a *timed* run; only for post-hoc row counting on stores
    that a killed run left without a cursor. The probe file is removed again.
    """
    cur = ws / ".minsync" / "cursor.json"
    if not cur.exists():
        cur.write_text(
            json.dumps(
                {
                    "source_id": read_source_id(ws),
                    "last_synced_at": "2000-01-01T00:00:00Z",
                    "manifest_hash": "probe",
                    "chunk_schema_id": "probe",
                    "embedder_id": "tei:bench/fake",
                    "collection_path": "store",
                    "lexical_language": "simple",
                }
            )
        )
    return cur


# --------------------------------------------------------------------------- #
# running MinSync
# --------------------------------------------------------------------------- #
def parse_json_output(text: str):
    text = text.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
        return None


def run_min_sync(
    binary: str,
    ws: Path,
    log_dir: Path,
    timeout=None,
    kill_after=None,
    kill_inputs=None,
    server: FakeServer = None,
    tag: str = "run",
) -> dict:
    """Run `minsync --format json sync` in `ws`.

    timeout      -> give up (SIGKILL) after N seconds, mark TIMEOUT
    kill_after   -> deliberately SIGKILL after N seconds (no TIMEOUT flag)
    kill_inputs  -> deliberately SIGKILL once the fake server has embedded
                    >= N texts in total (mid-sync kill)
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    out_path = log_dir / f"{tag}.out"
    err_path = log_dir / f"{tag}.err"
    cmd = [str(binary), "--format", "json", "sync"]
    start = time.monotonic()
    with open(out_path, "w") as outf, open(err_path, "w") as errf:
        proc = subprocess.Popen(
            cmd,
            cwd=ws,
            stdout=outf,
            stderr=errf,
            text=True,
            start_new_session=True,
        )
        timed_out = killed = False
        deadline = start + timeout if timeout else None
        while proc.poll() is None:
            now = time.monotonic()
            do_kill = False
            if kill_inputs is not None and server is not None:
                if server.inputs() >= kill_inputs:
                    killed = True
                    do_kill = True
            if not do_kill and kill_after is not None and (now - start) >= kill_after:
                killed = True
                do_kill = True
            if not do_kill and deadline is not None and now >= deadline:
                timed_out = True
                do_kill = True
            if do_kill:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
                break
            time.sleep(0.05)
        proc.wait()
    wall = time.monotonic() - start
    out_text = out_path.read_text()
    err_text = err_path.read_text()
    return {
        "wall_s": wall,
        "exit_code": proc.returncode,
        "timed_out": timed_out,
        "killed": killed,
        "json": parse_json_output(out_text),
        "stdout": out_text,
        "stderr": err_text,
        "cmd": " ".join(cmd),
    }


def run_query(binary: str, ws: Path, mode: str, k: int = QUERY_K, text: str = "lorem"):
    cmd = [str(binary), "--format", "json", "query", text, "--k", str(k), "--mode", mode]
    try:
        proc = subprocess.run(
            cmd, cwd=ws, capture_output=True, text=True, timeout=300
        )
    except subprocess.TimeoutExpired:
        return None, "query timeout"
    if proc.returncode != 0:
        return None, (proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else f"exit {proc.returncode}")
    try:
        return json.loads(proc.stdout), None
    except json.JSONDecodeError as error:
        return None, f"bad json: {error}"


def row_count(binary: str, ws: Path) -> tuple:
    """Total number of rows in the current store (via an exhaustive vector query).

    Writes a throwaway probe cursor only when the workspace has none (killed
    runs), then removes it again.
    """
    queries = store_queries(binary, ws)
    return queries["vector"], queries.get("vector_err")


def store_queries(binary: str, ws: Path) -> dict:
    """`minsync query` sanity check in both bm25 and vector modes."""
    cur = ws / ".minsync" / "cursor.json"
    created = False
    if not cur.exists():
        write_probe_cursor(ws)
        created = True
    try:
        out = {}
        for mode in ("vector", "bm25"):
            rows, err = run_query(binary, ws, mode)
            out[mode] = len(rows) if rows is not None else None
            out[f"{mode}_err"] = err
        return out
    finally:
        if created:
            cur.unlink(missing_ok=True)


def dump_ids(binary: str, ws: Path) -> tuple:
    cur = ws / ".minsync" / "cursor.json"
    created = False
    if not cur.exists():
        write_probe_cursor(ws)
        created = True
    try:
        rows, err = run_query(binary, ws, "vector")
        if rows is None:
            return None, err
        return sorted((r["path"], r["doc_id"]) for r in rows), None
    finally:
        if created:
            cur.unlink(missing_ok=True)


# --------------------------------------------------------------------------- #
# scenario runner
# --------------------------------------------------------------------------- #
class Bench:
    def __init__(self, args):
        self.args = args
        self.baseline = str(Path(args.baseline).resolve())
        self.candidate = str(Path(args.candidate).resolve()) if args.candidate else None
        self.candidate_exists = bool(self.candidate) and Path(self.candidate).exists()
        self.workdir = (
            Path(args.workdir).resolve()
            if args.workdir
            else Path(tempfile.mkdtemp(prefix="minsync_bench71_"))
        )
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.log_dir = self.workdir / "logs"
        self.server = FakeServer(args.dim, args.latency_ms, port=args.fake_port or None)
        self.docs = make_docs(args.n, args.seed)
        self.cfg = {
            "dim": args.dim,
            "batch_size": args.batch_size,
            "max_concurrent": args.max_concurrent,
            "ann_build_threshold": args.ann_build_threshold,
            "ann_optimize_threshold": args.ann_optimize_threshold,
        }
        self.results = []

    # -- helpers ---------------------------------------------------------- #
    def medians(self, runs):
        return statistics.median([r["wall_s"] for r in runs]) if runs else None

    def summarize_runs(self, runs, ws_after, binary):
        completed = [r for r in runs if not r["timed_out"] and not r["killed"] and r["exit_code"] == 0]
        last = runs[-1]
        js = last["json"] or {}
        queries = store_queries(binary, ws_after)
        st = store_stats(ws_after)
        return {
            "runs": runs,
            "median_s": self.medians(runs),
            "completed": len(completed) == len(runs),
            "timed_out": last["timed_out"],
            "killed": last["killed"],
            "stats_after": st,
            "rows": queries["vector"],
            "row_err": queries.get("vector_err"),
            "queries": queries,
            "json": js,
        }

    def fresh_workspace(self, name: str) -> Path:
        """Corpus + config, never synced."""
        ws = self.workdir / name
        copy_ws(self.workdir / "base_unsynced", ws)
        return ws

    def prepare_base_unsynced(self) -> None:
        base = self.workdir / "base_unsynced"
        if base.exists():
            shutil.rmtree(base)
        init_workspace(base, self.baseline, self.server, self.cfg)
        write_corpus(base, self.docs)

    # -- scenarios -------------------------------------------------------- #
    def scenario_initial(self) -> dict:
        a = self.args
        base_runs, cand_runs = [], []
        base_ws = cand_ws = None
        for rep in range(a.repeats):
            tag = f"initial_rep{rep}"
            base_ws = self.workdir / f"ws_initial_base_{rep}"
            cand_ws = self.workdir / f"ws_initial_cand_{rep}"
            copy_ws(self.workdir / "base_unsynced", base_ws)
            copy_ws(self.workdir / "base_unsynced", cand_ws)
            self.server.reset()
            base_runs.append(run_min_sync(self.baseline, base_ws, self.log_dir, timeout=a.timeout, server=self.server, tag=f"{tag}_base"))
            if self.candidate_exists:
                self.server.reset()
                cand_runs.append(run_min_sync(self.candidate, cand_ws, self.log_dir, timeout=a.timeout, server=self.server, tag=f"{tag}_cand"))
        base = self.summarize_runs(base_runs, base_ws, self.baseline)
        cand = None
        if self.candidate_exists:
            cand = self.summarize_runs(cand_runs, cand_ws, self.candidate)
        correctness = self.compare_workspaces(base_ws, cand_ws, base, cand) if cand else "candidate missing"
        return self.pack("initial", base, cand, correctness)

    def scenario_incremental(self) -> dict:
        a = self.args
        # One baseline initial sync to produce a synced store to mutate.
        prepared = self.workdir / "base_synced"
        copy_ws(self.workdir / "base_unsynced", prepared)
        self.server.reset()
        prim = run_min_sync(self.baseline, prepared, self.log_dir, timeout=a.timeout, server=self.server, tag="incremental_prime")
        if prim["exit_code"] != 0 or prim["timed_out"]:
            return {"scenario": "incremental", "error": f"priming sync failed: exit={prim['exit_code']} timed_out={prim['timed_out']}"}

        base_runs, cand_runs = [], []
        base_ws = cand_ws = None
        for rep in range(a.repeats):
            base_ws = self.workdir / f"ws_inc_base_{rep}"
            cand_ws = self.workdir / f"ws_inc_cand_{rep}"
            copy_ws(prepared, base_ws)
            copy_ws(prepared, cand_ws)
            mutate_files(base_ws, a.n, a.seed, rep, a.incremental_frac, a.add_files, a.delete_files)
            mutate_files(cand_ws, a.n, a.seed, rep, a.incremental_frac, a.add_files, a.delete_files)
            self.server.reset()
            base_runs.append(run_min_sync(self.baseline, base_ws, self.log_dir, timeout=a.timeout, server=self.server, tag=f"inc_rep{rep}_base"))
            if self.candidate_exists:
                self.server.reset()
                cand_runs.append(run_min_sync(self.candidate, cand_ws, self.log_dir, timeout=a.timeout, server=self.server, tag=f"inc_rep{rep}_cand"))
        base = self.summarize_runs(base_runs, base_ws, self.baseline)
        cand = self.summarize_runs(cand_runs, cand_ws, self.candidate) if self.candidate_exists else None
        correctness = self.compare_workspaces(base_ws, cand_ws, base, cand) if cand else "candidate missing"
        return self.pack("incremental", base, cand, correctness)

    def scenario_fragmented(self) -> dict:
        a = self.args
        frag_ws = self.workdir / "base_fragmented"
        copy_ws(self.workdir / "base_unsynced", frag_ws)

        iterations = 0
        start = time.monotonic()
        st = store_stats(frag_ws)
        while st["max_rail"] < a.fragments and iterations < a.frag_max_iters:
            if time.monotonic() - start > a.frag_budget:
                break
            mutate_files(frag_ws, a.n, a.seed, ("frag", iterations), a.frag_modify_frac)
            run_min_sync(
                self.baseline, frag_ws, self.log_dir,
                kill_after=a.kill_timeout, server=self.server,
                tag=f"frag_build_{iterations}",
            )
            iterations += 1
            st = store_stats(frag_ws)
        build_info = {
            "iterations": iterations,
            "build_seconds": time.monotonic() - start,
            "fragments": st["fragments"],
            "versions": st["versions"],
            "max_rail": st["max_rail"],
        }

        snap = self.workdir / "snapshot_fragmented"
        copy_ws(frag_ws, snap)
        snap_stats = store_stats(snap)

        base_ws = self.workdir / "ws_frag_base"
        cand_ws = self.workdir / "ws_frag_cand"
        copy_ws(snap, base_ws)
        copy_ws(snap, cand_ws)

        self.server.reset()
        base_run = run_min_sync(self.baseline, base_ws, self.log_dir, timeout=a.timeout, server=self.server, tag="frag_complete_base")
        cand_run = None
        if self.candidate_exists:
            self.server.reset()
            cand_run = run_min_sync(self.candidate, cand_ws, self.log_dir, timeout=a.timeout, server=self.server, tag="frag_complete_cand")

        base = self.summarize_runs([base_run], base_ws, self.baseline)
        cand = self.summarize_runs([cand_run], cand_ws, self.candidate) if cand_run else None
        correctness = self.compare_workspaces(base_ws, cand_ws, base, cand) if (cand and not base["timed_out"] and not cand["timed_out"]) else (
            "not compared (timeout)" if cand else "candidate missing"
        )
        out = self.pack("fragmented", base, cand, correctness)
        out["build"] = build_info
        out["snapshot"] = {"fragments": snap_stats["fragments"], "versions": snap_stats["versions"], "max_rail": snap_stats["max_rail"], "size_bytes": snap_stats["size_bytes"]}
        return out

    def scenario_resume(self) -> dict:
        a = self.args
        if not self.candidate_exists:
            return {"scenario": "resume", "error": "candidate binary not found"}
        # 1. uninterrupted candidate run
        single = self.workdir / "resume_single"
        copy_ws(self.workdir / "base_unsynced", single)
        self.server.reset()
        single_run = run_min_sync(self.candidate, single, self.log_dir, timeout=a.timeout, server=self.server, tag="resume_single")
        if single_run["exit_code"] != 0:
            return {"scenario": "resume", "error": f"single run failed exit={single_run['exit_code']}"}
        embedded_single = self.server.inputs()
        rows_single, err_single = row_count(self.candidate, single)

        # 2. killed-then-resumed candidate run
        res = self.workdir / "resume_split"
        copy_ws(self.workdir / "base_unsynced", res)
        self.server.reset()
        kill_inputs = max(1, int(embedded_single * a.resume_kill_frac))
        first = run_min_sync(self.candidate, res, self.log_dir, timeout=a.timeout, kill_inputs=kill_inputs, server=self.server, tag="resume_first_killed")
        after_first_inputs = self.server.inputs()
        second = run_min_sync(self.candidate, res, self.log_dir, timeout=a.timeout, server=self.server, tag="resume_second")
        embedded_total = self.server.inputs()
        rows_final, err_final = row_count(self.candidate, res)

        ratio = (embedded_total / embedded_single) if embedded_single else None
        rows_match = (rows_final is not None and rows_single is not None and rows_final == rows_single)
        passed = (
            second["exit_code"] == 0
            and ratio is not None
            and ratio < 1.5
            and rows_match
        )
        return {
            "scenario": "resume",
            "single": {"wall_s": single_run["wall_s"], "embedded": embedded_single, "rows": rows_single, "row_err": err_single},
            "split": {
                "killed": first["killed"],
                "first_wall_s": first["wall_s"],
                "after_first_inputs": after_first_inputs,
                "second_wall_s": second["wall_s"],
                "second_exit": second["exit_code"],
                "embedded_total": embedded_total,
                "rows": rows_final,
                "row_err": err_final,
            },
            "embedded_ratio": ratio,
            "rows_match": rows_match,
            "passed": passed,
        }

    # -- comparison / packing --------------------------------------------- #
    def compare_workspaces(self, base_ws: Path, cand_ws: Path, base: dict = None, cand: dict = None) -> str:
        if cand_ws is None or cand is None:
            return "candidate missing"
        base_ids, err_b = dump_ids(self.baseline, base_ws)
        cand_ids, err_c = dump_ids(self.candidate, cand_ws)
        if base_ids is None or cand_ids is None:
            return f"id dump failed (base={err_b} cand={err_c})"
        if base_ids != cand_ids:
            bset, cset = set(base_ids), set(cand_ids)
            return (
                f"ID MISMATCH base={len(base_ids)} cand={len(cand_ids)} "
                f"only_base={len(bset - cset)} only_cand={len(cset - bset)}"
            )
        # counter comparison from the last JSON result of each side
        bj = (base.get("json") or {}) if base else {}
        cj = (cand.get("json") or {}) if cand else {}
        keys = ["files_processed", "files_added", "files_modified", "files_deleted",
                "chunks_added", "chunks_updated", "chunks_deleted", "initial_sync"]
        mismatched = [k for k in keys if k in bj and k in cj and bj[k] != cj[k]]
        rows_note = ""
        if base and cand and base.get("rows") is not None and cand.get("rows") is not None:
            if base["rows"] != cand["rows"]:
                return f"ROW COUNT MISMATCH base={base['rows']} cand={cand['rows']} (ids identical)"
            rows_note = f", {base['rows']} rows"
        if mismatched:
            detail = ", ".join(f"{k}: base={bj.get(k)} cand={cj.get(k)}" for k in mismatched)
            return f"COUNTER MISMATCH ({detail})"
        return f"ok ({len(base_ids)} ids identical{rows_note})"

    def pack(self, name: str, base: dict, cand: dict, correctness: str) -> dict:
        speedup = None
        if cand and base.get("median_s") and cand.get("median_s") and cand["median_s"] > 0:
            speedup = base["median_s"] / cand["median_s"]
        return {
            "scenario": name,
            "baseline": base,
            "candidate": cand,
            "speedup": speedup,
            "correctness": correctness,
        }

    # -- top level -------------------------------------------------------- #
    def run(self) -> None:
        a = self.args
        print(f"baseline : {self.baseline}")
        print(f"candidate: {self.candidate if self.candidate_exists else '(missing)'} {self.candidate or ''}")
        print(f"workdir  : {self.workdir}")
        print(f"N        : {a.n} files x ~{PARAGRAPHS_PER_FILE} paragraphs")
        print()

        self.server.start()
        try:
            self.prepare_base_unsynced()
            scenarios = ["initial", "incremental", "fragmented", "resume"] if a.scenario == "all" else [a.scenario]
            for name in scenarios:
                print(f"=== scenario: {name} ===", flush=True)
                t0 = time.monotonic()
                if name == "initial":
                    res = self.scenario_initial()
                elif name == "incremental":
                    res = self.scenario_incremental()
                elif name == "fragmented":
                    res = self.scenario_fragmented()
                elif name == "resume":
                    res = self.scenario_resume()
                else:
                    raise SystemExit(f"unknown scenario {name}")
                res["scenario_seconds"] = time.monotonic() - t0
                self.results.append(res)
                print(f"  done in {res['scenario_seconds']:.1f}s: {self.result_line(res)}", flush=True)
        finally:
            self.server.stop()

        report = render_report(self.results, a)
        out_path = Path(a.out) if a.out else None
        if out_path:
            out_path.write_text(report)
            print(f"\nwrote {out_path}")
        print("\n" + report)
        if a.workdir is None and not a.keep:
            shutil.rmtree(self.workdir, ignore_errors=True)

    def result_line(self, res: dict) -> str:
        if "error" in res:
            return res["error"]
        if res.get("scenario") == "resume":
            return (
                f"PASS={res['passed']} embedded_ratio={res['embedded_ratio']} "
                f"rows_match={res['rows_match']}"
            )
        base = res["baseline"]
        cand = res["candidate"]
        b = f"{base['median_s']:.2f}s" if base and base.get("median_s") else "n/a"
        c = f"{cand['median_s']:.2f}s" if cand and cand.get("median_s") else "n/a"
        sp = f"speedup {res['speedup']:.2f}x" if res.get("speedup") else "speedup n/a"
        return f"base={b} cand={c} {sp} correctness={res['correctness']}"


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def fmt_seconds(value):
    return f"{value:.2f}" if isinstance(value, (int, float)) else "n/a"


def fmt_ratio(value):
    return f"{value:.2f}x" if isinstance(value, (int, float)) else "n/a"


def fmt_rail(stats):
    if not stats:
        return "n/a"
    return f"{stats['fragments']}/{stats['versions']}"


def render_report(results, args) -> str:
    lines = []
    lines.append("# MinSync issue #71 benchmark\n")
    lines.append(f"- baseline binary: `{args.baseline}`")
    lines.append(f"- candidate binary: `{args.candidate}`")
    lines.append(f"- N = {args.n} files, repeats = {args.repeats} (scenarios 1-2), seed = {args.seed}")
    lines.append(f"- fake embedder: dim {args.dim}, latency {args.latency_ms} ms, batch_size {args.batch_size}, max_concurrent {args.max_concurrent}")
    lines.append(f"- timeout per run: {args.timeout}s, fragmented kill interval: {args.kill_timeout}s, fragment target K={args.fragments}")
    lines.append("")

    lines.append("## Summary\n")
    lines.append("| Scenario | N | Baseline (s) | Candidate (s) | Speedup | Baseline frag/ver | Candidate frag/ver | Correctness | Notes |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for res in results:
        name = res.get("scenario")
        if "error" in res:
            lines.append(f"| {name} | {args.n} | n/a | n/a | n/a | n/a | n/a | error | {res['error']} |")
            continue
        if name == "resume":
            single = res["single"]
            notes = f"embedded_ratio={res['embedded_ratio']:.3f}, rows_match={res['rows_match']}"
            lines.append(
                f"| resume | {args.n} | n/a | {fmt_seconds(single['wall_s'])} | n/a | n/a | n/a | "
                f"{'PASS' if res['passed'] else 'FAIL'} | {notes} |"
            )
            continue
        base = res["baseline"]
        cand = res["candidate"]
        b_stats = base["stats_after"]
        c_stats = cand["stats_after"] if cand else None
        notes = []
        if base.get("timed_out"):
            notes.append("baseline TIMEOUT")
        if cand and cand.get("timed_out"):
            notes.append("candidate TIMEOUT")
        if res["scenario"] == "fragmented" and "build" in res:
            notes.append(
                f"built {res['build']['fragments']} frag/{res['build']['versions']} ver "
                f"in {res['build']['iterations']} kills ({res['build']['build_seconds']:.0f}s)"
            )
        lines.append(
            f"| {name} | {args.n} | {fmt_seconds(base.get('median_s'))} | "
            f"{fmt_seconds(cand.get('median_s')) if cand else 'n/a'} | {fmt_ratio(res.get('speedup'))} | "
            f"{fmt_rail(b_stats)} | {fmt_rail(c_stats) if c_stats else 'n/a'} | "
            f"{res['correctness']} | {'; '.join(notes)} |"
        )
    lines.append("")

    for res in results:
        if "error" in res:
            continue
        name = res["scenario"]
        lines.append(f"## Scenario: {name}\n")
        if name == "resume":
            lines.append(f"- single uninterrupted: {json.dumps(res['single'])}")
            lines.append(f"- killed+resumed: {json.dumps(res['split'])}")
            lines.append(f"- embedded work ratio (total/single): {res['embedded_ratio']}")
            lines.append(f"- row counts match: {res['rows_match']}")
            lines.append("- **PASS**" if res["passed"] else "- **FAIL**")
            lines.append("")
            continue
        for label, side, binary in (("baseline", res["baseline"], args.baseline), ("candidate", res["candidate"], args.candidate)):
            if side is None:
                lines.append(f"- **{label}**: n/a")
                continue
            runs = side["runs"]
            walls = ", ".join(f"{r['wall_s']:.2f}" for r in runs)
            st = side["stats_after"]
            js = side.get("json") or {}
            lines.append(f"- **{label}** (`{binary}`):")
            lines.append(f"    - wall seconds per run: [{walls}] -> median {fmt_seconds(side['median_s'])}")
            lines.append(f"    - status: {'TIMEOUT' if side.get('timed_out') else ('KILLED' if side.get('killed') else 'ok')}, exit {runs[-1]['exit_code']}")
            lines.append(f"    - fragments/versions/deletions: {st['fragments']}/{st['versions']}/{st['_deletions']}, store size {st['size_bytes']/1e6:.1f} MB")
            q = side.get("queries") or {}
            lines.append(f"    - rows after: {side.get('rows')}" + (f" (err: {side['row_err']})" if side.get("row_err") else ""))
            lines.append(
                f"    - query sanity (k={QUERY_K}): vector={q.get('vector')} (err: {q.get('vector_err')})"
                f", bm25={q.get('bm25')} (err: {q.get('bm25_err')})"
            )
            if js:
                keys = ["files_processed", "files_added", "files_modified", "files_deleted",
                        "chunks_added", "chunks_updated", "chunks_deleted",
                        "embedded_texts", "embedding_api_calls", "elapsed_seconds"]
                lines.append("    - last sync JSON: " + json.dumps({k: js.get(k) for k in keys if k in js}))
        # query sanity for the last completed workspace
        lines.append(f"- correctness: {res['correctness']}")
        if "build" in res:
            lines.append(f"- fragmentation build: {json.dumps(res['build'])}")
            lines.append(f"- snapshot: {json.dumps(res['snapshot'])}")
        lines.append("")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="MinSync issue #71 benchmark harness")
    p.add_argument("--baseline", default=DEFAULT_BASELINE, help="baseline binary path")
    p.add_argument("--candidate", default=DEFAULT_CANDIDATE, help="candidate binary path")
    p.add_argument("--scenario", default="all", choices=["initial", "incremental", "fragmented", "resume", "all"])
    p.add_argument("--n", type=int, default=600, help="number of corpus files")
    p.add_argument("--repeats", type=int, default=3, help="timing repeats for initial/incremental")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--timeout", type=float, default=900.0, help="per-run timeout (s); TIMEOUT reported on expiry")
    p.add_argument("--kill-timeout", type=float, default=3.0, help="fragmented builder: SIGKILL after N s per run")
    p.add_argument("--fragments", type=int, default=3000, help="fragment target K for the fragmented scenario")
    p.add_argument("--frag-max-iters", type=int, default=200, help="max kill iterations when building fragments")
    p.add_argument("--frag-budget", type=float, default=600.0, help="wall budget (s) for building fragments")
    p.add_argument("--frag-modify-frac", type=float, default=0.15, help="fraction of files edited between kills")
    p.add_argument("--incremental-frac", type=float, default=0.30, help="fraction of files modified in the incremental scenario")
    p.add_argument("--add-files", type=int, default=0, help="files added during incremental mutation")
    p.add_argument("--delete-files", type=int, default=0, help="files deleted during incremental mutation")
    p.add_argument("--resume-kill-frac", type=float, default=0.4, help="kill the first resume run after this fraction of embeddings")
    p.add_argument("--latency-ms", type=float, default=0.0, help="artificial per-request fake-TEI latency")
    p.add_argument("--dim", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-concurrent", type=int, default=1)
    p.add_argument("--fake-port", type=int, default=0, help="fixed fake-TEI port (0 = auto)")
    p.add_argument("--workdir", default=None, help="work directory (default: temp dir)")
    p.add_argument("--keep", action="store_true", help="keep an auto-created temp work directory (ignored with --workdir)")
    p.add_argument("--ann-build-threshold", type=int, default=1_000_000_000,
                   help="vectorstore.options.index_build_threshold (huge = keep vector search exact)")
    p.add_argument("--ann-optimize-threshold", type=int, default=1_000_000_000,
                   help="vectorstore.options.index_optimize_delta_threshold")
    p.add_argument("--out", default=None, help="write the markdown report to this path as well as stdout")
    return p


def main() -> int:
    args = build_parser().parse_args()
    if not Path(args.baseline).exists():
        print(f"baseline binary not found: {args.baseline}", file=sys.stderr)
        return 2
    if not FAKE_TEI.exists():
        print(f"missing {FAKE_TEI}", file=sys.stderr)
        return 2
    bench = Bench(args)
    bench.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
