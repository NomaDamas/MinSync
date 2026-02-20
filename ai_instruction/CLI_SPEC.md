# MinSync CLI 명세서

> 버전: v0.2-draft
> 최종 갱신: 2026-02-20

---

## 0. 설계 원칙

1. **CLI = Python API의 Thin Wrapper**: 모든 CLI 커맨드는 대응하는 Python API 메서드를 호출한다. CLI 전용 로직은 출력 포매팅과 인자 파싱뿐이다.
2. **Pre-built vs Custom**: `config.yaml`에서는 pre-built 메서드만 선택 가능. 커스텀 chunker/embedder/vectorstore는 Python API를 통해 객체를 직접 전달하는 방식으로만 사용.
3. **LangChain 연동, 의존성 미포함**: minsync는 LangChain 기반 어댑터 인터페이스를 정의하지만, 구체적 벡터 DB/임베딩 모델의 패키지는 번들하지 않는다. 사용자가 직접 설치해야 하며, 미설치 시 명확한 에러를 반환한다.
4. **상태 디렉토리**: `.minsync/` (`.ragit/` 대신)
5. **인덱싱 대상**: git-tracked 모든 파일이 기본 대상. `.gitignore`에 의해 untracked된 파일은 `git ls-tree`/`git diff`에 나타나지 않으므로 자동으로 제외된다. 추가로 `.minsyncignore` 파일로 git-tracked이지만 인덱싱은 원치 않는 파일을 제외 관리한다 (`.gitignore`와 동일 문법).
6. **CI/CD 우선 설계**: GitHub Actions 등 비대화형 환경에서 실행되는 것을 전제로 설계.

---

## 0.1 진입점

```
minsync <command> [options]
```

`pyproject.toml`의 `[project.scripts]`에 등록:
```toml
[project.scripts]
minsync = "minsync.cli:main"
```

모든 커맨드는 **현재 디렉토리가 git 저장소 내부**에 있어야 하며, 아닐 경우 에러를 반환한다.

### 공통 옵션

| 옵션 | 설명 |
|---|---|
| `--help`, `-h` | 해당 커맨드의 도움말 출력 |
| `--version` | MinSync 버전 출력 |
| `--verbose`, `-v` | 상세 로그 출력 (DEBUG 레벨) |
| `--quiet`, `-q` | 에러 외 출력 억제 |
| `--format <fmt>` | 출력 포맷: `text` (기본), `json` |

### 종료 코드 규칙

| 코드 | 의미 |
|---|---|
| `0` | 성공 |
| `1` | 일반 에러 (인자 오류, 설정 오류, 정합성 불일치 등) |
| `2` | Git 관련 에러 (git 저장소가 아님, ref를 찾을 수 없음 등) |
| `3` | Lock 획득 실패 (다른 sync가 진행 중) |
| `4` | Vector DB 연결/조작 에러 |
| `5` | 임베딩 에러 (모델 호출 실패, 패키지 미설치 등) |

---

## 0.2 Python API 개요

모든 CLI 기능은 `MinSync` 클래스를 통해 Python에서 동일하게 사용 가능하다.

```python
from minsync import MinSync

# 기본 사용 (config.yaml의 pre-built 설정 사용)
ms = MinSync(repo_path="/path/to/repo")

# 커스텀 컴포넌트 사용
ms = MinSync(
    repo_path="/path/to/repo",
    chunker=MyCustomChunker(),       # Chunker 인터페이스 구현
    embedder=MyCustomEmbedder(),     # Embedder 인터페이스 구현
    vector_store=MyCustomStore(),    # VectorStore 인터페이스 구현
)

# CLI 커맨드에 대응하는 메서드
ms.init()                                   # minsync init
ms.sync()                                   # minsync sync
ms.sync(full=True)                          # minsync sync --full
ms.sync(dry_run=True)                       # minsync sync --dry-run
results = ms.query("search text", k=10)     # minsync query "text" --k 10
status = ms.status()                        # minsync status
report = ms.verify(all=True)                # minsync verify --all
report = ms.verify(fix=True)                # minsync verify --fix
health = ms.check()                         # minsync check
```

`MinSync` 생성자에 `chunker`, `embedder`, `vector_store`를 전달하면 config.yaml의 해당 설정을 무시하고 전달된 객체를 사용한다. 전달하지 않으면 config.yaml의 pre-built 설정에서 인스턴스를 생성한다.

---

## 1. `minsync init`

### 용도
현재 git 저장소에 MinSync를 초기화한다. `.minsync/` 디렉토리와 설정 파일을 생성한다.

### CLI 사용법
```bash
minsync init [--collection <name>] [--embedder <embedder_id>] [--chunker <chunker_id>] [--force]
```

### Python API
```python
ms.init(collection="my_collection", embedder="openai:text-embedding-3-small",
        chunker="markdown-heading", force=False)
```

### 옵션

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--collection <name>` | `minsync_{repo_id[:8]}` | 벡터 DB 컬렉션 이름 |
| `--embedder <id>` | `openai:text-embedding-3-small` | 사용할 임베더 식별자 (pre-built) |
| `--chunker <id>` | `markdown-heading` | 사용할 청커 식별자 (pre-built) |
| `--force` | false | 기존 `.minsync/`가 있어도 덮어쓰기 |

### 동작

1. **git 저장소 확인**: 현재 디렉토리가 git 저장소인지 확인 → 아니면 exit 2
2. **repo_id 결정**: `git rev-list --max-parents=0 HEAD | tail -n 1` 실행
   - HEAD가 없는 빈 저장소면 exit 2 + "커밋이 최소 1개 필요합니다" 메시지
3. **기존 `.minsync/` 확인**:
   - 이미 존재하고 `--force` 없으면 exit 1 + "이미 초기화되어 있습니다. --force로 재초기화하세요"
   - `--force` 시 기존 `.minsync/` 삭제 후 재생성
4. **디렉토리 및 파일 생성**:
   - `.minsync/` 디렉토리 생성
   - `.minsync/config.yaml` 생성 (아래 스키마)
   - `cursor.json`은 아직 생성하지 않음 (sync 전이므로)

### 생성되는 `.minsync/config.yaml`

```yaml
version: 1
repo_id: "<root-commit-hash>"
ref: "main"

collection:
  name: "minsync_abcd1234"
  path: ".minsync/zvec_data"   # 로컬 Zvec 기본 경로

chunker:
  id: "markdown-heading"
  options:
    max_chunk_size: 1000
    overlap: 100

embedder:
  id: "openai:text-embedding-3-small"
  batch_size: 64

vectorstore:
  id: "zvec"
  options: {}

normalize:
  strip_trailing_whitespace: true
  normalize_newlines: true
  collapse_whitespace: false
  strip_frontmatter: false
```

> **참고**: `include` 패턴은 없다. git-tracked 모든 파일이 기본 인덱싱 대상이다. `.gitignore`에 의해 untracked된 파일은 git 자체에서 제외되므로 minsync에도 보이지 않는다. git-tracked이지만 인덱싱하고 싶지 않은 파일은 `.minsyncignore`로 제외한다.

### 출력 (stdout)

```
Initialized MinSync in .minsync/
  repo_id:      abc123def456...
  collection:   minsync_abc123de
  chunker:      markdown-heading
  embedder:     openai:text-embedding-3-small
  vectorstore:  zvec (local)

Run 'minsync check' to verify your setup, then 'minsync sync' to build the initial index.
```

### 에러 케이스

| 상황 | 메시지 | 종료 코드 |
|---|---|---|
| git 저장소가 아님 | `Error: not a git repository` | 2 |
| 커밋이 없는 빈 저장소 | `Error: repository has no commits. Create at least one commit first.` | 2 |
| `.minsync/` 이미 존재 (--force 없음) | `Error: already initialized. Use --force to reinitialize.` | 1 |

---

## 2. `minsync sync`

### 용도
git diff를 기반으로 벡터 인덱스를 증분 동기화한다. 최초 실행 시 전체 인덱싱(full index)을 수행한다.

### CLI 사용법
```bash
minsync sync [--ref <branch>] [--full] [--dry-run] [--verbose] [--batch-size <n>] [--wait]
```

### Python API
```python
result = ms.sync(ref="main", full=False, dry_run=False, batch_size=64, wait=False)
# result: SyncResult(from_commit, to_commit, files_processed, chunks_added, ...)
```

### 옵션

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--ref <branch>` | config의 `ref` (기본 `main`) | 동기화 대상 브랜치/ref |
| `--full` | false | cursor 무시, 전체 재구축 |
| `--dry-run` | false | 실제 DB 변경 없이 변경 계획만 출력 |
| `--batch-size <n>` | config의 `embedder.batch_size` | 임베딩 배치 크기 오버라이드 |
| `--wait` | false | 락 점유 시 해제될 때까지 대기 (기본: 즉시 종료) |

### 동작 — 증분 모드 (기본)

1. **전제 조건 확인**
   - `.minsync/config.yaml` 존재 확인 → 없으면 exit 1 + "minsync init을 먼저 실행하세요"
   - git 저장소 확인
   - schema/embedder 불일치 확인: cursor.json이 존재하고, chunk_schema_id 또는 embedder_id가 현재 config와 다르면 `--full` 필수 → exit 1 + 경고

2. **락 획득**
   - `.minsync/lock` 파일 생성 시도 (exclusive)
   - 실패 시: `--wait`면 polling 대기, 아니면 exit 3

3. **미완료 트랜잭션(txn.json) 복구 확인**
   - `.minsync/txn.json` 존재 시:
     - stdout에 `Recovering from interrupted sync...` 출력
     - txn의 `from_commit`과 `to_commit`을 사용해 동일 범위로 재처리

4. **목표 커밋 결정**
   - `to_commit = git rev-parse <ref>`
   - `from_commit = cursor.last_synced_commit` (없으면 None → full index)

5. **변경 파일 목록 산출**
   - `from_commit`이 None (최초 sync):
     - `git ls-tree -r --name-only to_commit` 으로 전체 파일 목록 → 모두 A(Added) 취급
   - `from_commit` 존재:
     - `git diff --name-status from_commit..to_commit`
     - A/M/D/R 상태 파싱
   - **`.minsyncignore` 필터링 적용**: git-tracked 파일 중 `.minsyncignore` 패턴에 매칭되는 파일을 제외

6. **txn.json 기록**
   - `sync_token = uuid4()` 생성
   - txn.json 작성

7. **파일 단위 처리**

   각 파일에 대해 (파일 상태에 따라):

   **D (Deleted)**:
   - `delete_by_filter("repo_id == X AND ref == Y AND path == Z")`

   **A / M / R (Added / Modified / Renamed)**:
   - a. `git show to_commit:<path>` 로 파일 내용 읽기
   - b. normalize 적용
   - c. chunker로 청크 생성
   - d. 각 청크의 `doc_id` 계산 (결정적 ID 알고리즘)
   - e. `fetch(ids)` → 존재하는 ID / 없는 ID 분류
   - f. 존재하는 ID: `update([{id, seen_token, path, heading_path}])` (메타데이터만)
   - g. 없는 ID: embedding 계산 → `upsert([{id, embedding, text, ..., seen_token}])`
   - h. **Sweep**: `delete_by_filter("repo_id==X AND ref==Y AND path==Z AND seen_token != sync_token")`

   진행률 출력: `[3/15] Processing docs/guide.md (A) ... 5 chunks (2 new, 3 unchanged)`

8. **커밋 단계**
   - `collection.flush()`
   - cursor.json 원자적 갱신 (temp write → fsync → rename)
   - txn.json 삭제
   - lock 해제

9. **dry-run 모드**
   - 1~5 단계까지만 실행
   - 변경 파일 목록과 예상 동작을 출력하고 종료 (DB 변경 없음)

### 출력 — 정상 완료

```
MinSync sync completed.
  ref:         main
  from_commit: abc1234 (short)
  to_commit:   def5678 (short)
  files:       15 processed (3 added, 10 modified, 2 deleted)
  chunks:      142 total (28 new, 98 unchanged, 16 deleted)
  duration:    4.2s
```

### 출력 — 이미 최신

```
Already up to date.
  current: def5678
  ref:     main
```

### 출력 — dry-run

```
[DRY RUN] Sync plan: abc1234 → def5678

  Added:
    docs/new-feature.md
  Modified:
    docs/guide.md
    docs/api.md
  Deleted:
    docs/old-page.md
  Ignored (.minsyncignore):
    src/main.py
    src/utils.py

  Estimated: 3 files to process, 1 file to delete
```

### 에러 케이스

| 상황 | 메시지 | 종료 코드 |
|---|---|---|
| 초기화 안 됨 | `Error: not initialized. Run 'minsync init' first.` | 1 |
| lock 획득 실패 | `Error: another sync is in progress. Use --wait to wait.` | 3 |
| ref를 찾을 수 없음 | `Error: ref 'feature/x' not found.` | 2 |
| 임베딩 실패 | `Error: embedding failed: <detail>` | 5 |
| 임베딩 패키지 미설치 | `Error: '<package>' is required for embedder '<id>'. Install it with: pip install <package>` | 5 |
| vector DB 에러 | `Error: vector store operation failed: <detail>` | 4 |
| vector DB 패키지 미설치 | `Error: '<package>' is required for vectorstore '<id>'. Install it with: pip install <package>` | 4 |
| schema/embedder 불일치 | `Error: schema/embedder mismatch detected. Run 'minsync sync --full' to rebuild.` | 1 |

---

## 3. `minsync query`

### 용도
텍스트 쿼리를 벡터 검색하여 관련 문서 청크를 반환한다.

### CLI 사용법
```bash
minsync query "<text>" [--ref <branch>] [--k <n>] [--filter <expr>] [--format <fmt>] [--show-score]
```

### Python API
```python
results = ms.query("search text", ref="main", k=10, filter_expr=None)
# results: list[QueryResult(doc_id, path, heading_path, chunk_type, text, score, content_commit)]
```

### 옵션

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--ref <branch>` | config의 `ref` | 검색 범위 ref |
| `--k <n>` | `10` | 반환할 최대 결과 수 |
| `--filter <expr>` | (없음) | 추가 필터 표현식 (예: `path == "docs/*"`) |
| `--format <fmt>` | `text` | 출력 포맷: `text`, `json`, `jsonl` |
| `--show-score` | false | 유사도 점수 표시 |

### 동작

1. 초기화 확인 + cursor.json 존재 확인 (sync 한번도 안 했으면 경고)
2. 쿼리 텍스트를 embedder로 임베딩
3. `collection.query(vector, filter="repo_id==X AND ref==Y" + 사용자 필터, topk=k)`
4. 결과를 format에 맞게 출력

### 출력 — text 포맷 (기본)

```
Found 5 results for "authentication flow":

[1] docs/auth/login.md (score: 0.92)
    heading: ## Login Process
    ---
    The login process begins with the user entering their credentials...
    ---

[2] docs/auth/oauth.md (score: 0.87)
    heading: ## OAuth2 Integration
    ---
    For third-party authentication, we use OAuth2 with...
    ---
```

### 출력 — json 포맷

```json
{
  "query": "authentication flow",
  "ref": "main",
  "results": [
    {
      "rank": 1,
      "doc_id": "sha256...",
      "path": "docs/auth/login.md",
      "heading_path": "## Login Process",
      "chunk_type": "child",
      "text": "The login process begins with...",
      "score": 0.92,
      "content_commit": "abc1234"
    }
  ]
}
```

### 에러 케이스

| 상황 | 메시지 | 종료 코드 |
|---|---|---|
| 초기화 안 됨 | `Error: not initialized. Run 'minsync init' first.` | 1 |
| 인덱스가 비어있음 (sync 미실행) | `Warning: index is empty. Run 'minsync sync' first.` (exit 0, 결과 0건) | 0 |
| 빈 쿼리 문자열 | `Error: query text is required.` | 1 |
| 임베딩 실패 | `Error: embedding failed: <detail>` | 5 |

---

## 4. `minsync status`

### 용도
현재 MinSync 동기화 상태를 표시한다.

### CLI 사용법
```bash
minsync status [--format <fmt>]
```

### Python API
```python
status = ms.status()
# status: StatusResult(repo_id, ref, last_synced_commit, current_head, state, pending_txn, ...)
# state: "UP_TO_DATE" | "OUT_OF_DATE" | "NOT_SYNCED" | "INTERRUPTED"
```

### 동작

1. `.minsync/config.yaml` 읽기
2. `.minsync/cursor.json` 읽기 (없으면 "never synced")
3. `to_commit = git rev-parse <ref>` 실행
4. cursor.last_synced_commit과 to_commit 비교
5. `.minsync/txn.json` 존재 여부 확인 (incomplete sync 감지)

### 출력 — 각 상태별

**UP TO DATE:**
```
MinSync Status
  repo_id:         abc123def456...
  ref:             main
  collection:      minsync_abc123de
  chunker:         markdown-heading (schema: v1_md_heading)
  embedder:        openai:text-embedding-3-small
  vectorstore:     zvec (local)

Sync State:
  last_synced:     def5678 (2026-02-20T10:30:00Z)
  current_head:    def5678
  status:          UP TO DATE
  pending txn:     none
```

**OUT OF DATE:**
```
Sync State:
  last_synced:     abc1234 (2026-02-19T15:00:00Z)
  current_head:    ghi9012
  status:          OUT OF DATE (2 commits behind)
  pending txn:     none

Run 'minsync sync' to update the index.
```

**NOT SYNCED (sync 미실행):**
```
Sync State:
  last_synced:     (never)
  current_head:    def5678
  status:          NOT SYNCED

Run 'minsync sync' to build the initial index.
```

**INTERRUPTED (txn 잔존):**
```
Sync State:
  last_synced:     abc1234 (2026-02-19T15:00:00Z)
  current_head:    ghi9012
  status:          INTERRUPTED (txn.json found, started 2026-02-20T09:00:00Z)
  pending txn:     abc1234 → def5678

Run 'minsync sync' to resume/recover.
```

### JSON 출력

```json
{
  "repo_id": "abc123def456...",
  "ref": "main",
  "collection": "minsync_abc123de",
  "chunker": "markdown-heading",
  "embedder": "openai:text-embedding-3-small",
  "vectorstore": "zvec",
  "last_synced_commit": "abc1234...",
  "current_head": "ghi9012...",
  "state": "OUT_OF_DATE",
  "commits_behind": 2,
  "pending_txn": null
}
```

### 에러 케이스

| 상황 | 메시지 | 종료 코드 |
|---|---|---|
| 초기화 안 됨 | `Error: not initialized. Run 'minsync init' first.` | 1 |

---

## 5. `minsync check`

### 용도
현재 설정된 임베딩 모델과 벡터 DB의 연결 상태를 확인한다. CI/CD 파이프라인에서 sync 전 사전 검증에 사용.

### CLI 사용법
```bash
minsync check [--format <fmt>]
```

### Python API
```python
health = ms.check()
# health: CheckResult(embedder_ok, vectorstore_ok, git_ok, errors=[...])
```

### 동작

1. **Git 확인**: 저장소 유효성, repo_id, ref, HEAD 확인
2. **Embedder 확인**: 테스트 텍스트로 임베딩 호출 시도. 성공하면 dimension과 응답시간 표시.
3. **VectorStore 확인**: 컬렉션 접근 시도. 존재하면 문서 수 표시, 없으면 "will be created" 표시.

### 출력 — 정상

```
MinSync Health Check
  Git:          OK (repo_id=abc123, ref=main, HEAD=def456)
  Embedder:     openai:text-embedding-3-small ... OK (dim=1536, 0.3s)
  VectorStore:  zvec (local) ... OK (collection exists, 142 docs)

All checks passed.
```

### 출력 — 실패

```
MinSync Health Check
  Git:          OK (repo_id=abc123, ref=main, HEAD=def456)
  Embedder:     openai:text-embedding-3-small ... FAIL
    Error: OpenAI API key not set. Set OPENAI_API_KEY environment variable.
  VectorStore:  weaviate (http://localhost:8080) ... FAIL
    Error: Connection refused. Is Weaviate running?

2 checks failed.
```

### 출력 — 패키지 미설치

```
MinSync Health Check
  Git:          OK
  Embedder:     openai:text-embedding-3-small ... FAIL
    Error: 'langchain-openai' package not found. Install with: pip install langchain-openai
  VectorStore:  weaviate ... FAIL
    Error: 'weaviate-client' package not found. Install with: pip install weaviate-client langchain-weaviate

2 checks failed.
```

### JSON 출력

```json
{
  "git": {"status": "ok", "repo_id": "abc123...", "ref": "main", "head": "def456..."},
  "embedder": {"status": "ok", "id": "openai:text-embedding-3-small", "dimension": 1536, "latency_ms": 300},
  "vectorstore": {"status": "ok", "id": "zvec", "doc_count": 142},
  "all_passed": true
}
```

### 종료 코드

| 상황 | 종료 코드 |
|---|---|
| 모든 체크 통과 | `0` |
| 1개 이상 실패 | `1` |
| 초기화 안 됨 | `1` |

---

## 6. `minsync verify`

### 용도
인덱스의 정합성을 검증한다. 현재 git 스냅샷과 벡터 DB 내용이 일치하는지 확인한다.
`.minsyncignore`에 해당하는 파일의 잔존 청크도 감지하여 삭제한다.

### CLI 사용법
```bash
minsync verify [--ref <branch>] [--sample <n>] [--all] [--fix] [--format <fmt>]
```

### Python API
```python
report = ms.verify(ref="main", sample=10, all=False, fix=False)
# report: VerifyResult(basic_checks, file_checks, ignored_stale, all_passed)
```

### 옵션

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--ref <branch>` | config의 `ref` | 검증 대상 ref |
| `--sample <n>` | `10` | 랜덤 샘플링할 파일 수 |
| `--all` | false | 전체 파일 검증 (sample 무시) |
| `--fix` | false | 불일치 발견 시 자동 수정 |
| `--format <fmt>` | `text` | 출력 포맷 |

### 동작

1. **기본 검증** (항상 수행):
   - cursor.json 존재 및 유효성 확인
   - cursor.last_synced_commit이 실제 git에 존재하는지 확인
   - txn.json 잔존 여부 확인 (있으면 "interrupted" 경고)
   - chunk_schema_id, embedder_id가 현재 config와 일치하는지 확인
   - 컬렉션 접근 가능 여부 및 기본 통계 (문서 수)

2. **`.minsyncignore` 잔존 검사** (항상 수행):
   - 현재 `.minsyncignore` 패턴을 로드
   - 벡터 DB에서 해당 repo_id/ref의 모든 고유 path 목록 조회
   - 각 path가 `.minsyncignore`에 매칭되는지 확인
   - 매칭되는 path가 있으면 "IGNORED_STALE" 보고
   - `--fix` 시: 해당 path의 모든 청크를 `delete_by_filter`로 삭제

3. **샘플 검증** (`--sample N` 또는 `--all`):
   - cursor.last_synced_commit 시점의 파일 목록 수집 (`.minsyncignore` 적용)
   - N개 파일을 랜덤 선택 (또는 `--all`이면 전체)
   - 각 파일에 대해:
     a. `git show last_synced_commit:<path>`로 내용 읽기
     b. 청킹 → doc_id 목록 생성
     c. `fetch(ids)` 로 DB에서 해당 ID 존재 확인
     d. 누락된 ID가 있으면 "MISSING" 보고
     e. DB에서 해당 path의 전체 doc 목록 조회
     f. 예상 ID 목록에 없는 doc이 있으면 "STALE" 보고

4. **삭제된 파일 검증** (`--all` 시):
   - git에 없는 path가 DB에 남아있는지 확인

### 출력 — 정상

```
MinSync Verify
  ref:              main
  synced_commit:    def5678
  collection docs:  142

Basic checks:       PASS
  cursor valid:     OK
  no pending txn:   OK
  schema match:     OK
  collection alive: OK (142 documents)

Ignored files check: PASS
  no stale ignored files found

Sample verification (10 files):
  docs/guide.md          OK (12 chunks)
  docs/api.md            OK (8 chunks)
  docs/auth/login.md     OK (5 chunks)
  ...

Result: ALL CHECKS PASSED
```

### 출력 — .minsyncignore 잔존 감지

```
MinSync Verify
  ...

Ignored files check: FAIL
  2 ignored paths still have chunks in the index:
    src/main.py          3 stale chunks
    src/utils.py         2 stale chunks

  Run 'minsync verify --fix' to remove stale ignored chunks.

Result: VERIFICATION FAILED
  5 stale chunks from ignored files.
```

### 출력 — --fix 실행

```
MinSync Verify (with fix)
  ...

Ignored files check: FIXED
  Removed 5 stale chunks from 2 ignored paths:
    src/main.py          3 chunks removed
    src/utils.py         2 chunks removed

Sample verification (10 files): PASS

Result: ALL CHECKS PASSED (after fix)
```

### 에러 케이스

| 상황 | 메시지 | 종료 코드 |
|---|---|---|
| 초기화 안 됨 | `Error: not initialized.` | 1 |
| sync 한번도 안 됨 | `Error: never synced. Run 'minsync sync' first.` | 1 |
| 불일치 발견 (fix 미적용) | 상세 보고 출력 | 1 |
| 모든 검증 통과 | | 0 |
| fix로 수정 완료 | | 0 |

---

## 7. `.minsyncignore`

### 위치
git 저장소 루트에 위치: `<repo_root>/.minsyncignore`

### 문법
`.gitignore`와 100% 동일한 문법을 따른다:

```
# 주석
*.pyc
__pycache__/
src/**/*.py
!src/important.py    # 네거티브 패턴 (제외 대상에서 다시 포함)
dist/
build/
*.png
*.jpg
*.bin
vendor/
```

### 자동 제외 대상 (하드코딩)
`.minsyncignore` 파일 유무와 관계없이 항상 제외되는 항목:
- `.minsync/` 디렉토리 자체
- `.minsyncignore` 파일 자체
- `.git/` 디렉토리

### 동작 원칙

1. **1차 필터 (.gitignore)**: `.gitignore`에 해당하는 파일은 git에 의해 untracked 상태이므로 `git ls-tree`와 `git diff`의 출력에 포함되지 않는다. minsync는 별도 처리 없이도 이 파일들을 보지 못한다.
2. **2차 필터 (.minsyncignore)**: git-tracked이지만 인덱싱을 원치 않는 파일(소스 코드, 바이너리 등)을 추가로 제외한다. `git ls-tree`/`git diff` 결과에서 `.minsyncignore` 패턴에 매칭되는 파일을 걸러낸다.
3. **최종 인덱싱 대상** = (git-tracked 파일) − (.minsyncignore 매칭 파일) − (하드코딩 제외 대상)
4. **`.minsyncignore` 변경**: `--full` 재인덱싱 불필요. 증분 sync와 verify의 조합으로 수렴.
   - 새로 ignore된 파일: 다음 verify에서 잔존 청크 감지 → 삭제
   - ignore 해제된 파일: 다음 sync에서 신규 파일로 인덱싱됨

---

## 부록: 전체 `.minsync/` 디렉토리 구조

```
.minsync/
├── config.yaml          # 설정 (init 시 생성, 사용자 편집 가능 — pre-built만)
├── cursor.json          # 마지막 sync 상태 (sync 완료 후 생성)
├── txn.json             # 진행 중 트랜잭션 (sync 중에만 존재)
├── lock                 # 파일 락 (sync 중에만 존재)
└── zvec_data/           # 로컬 Zvec 데이터 디렉토리 (기본 vectorstore 경로)
```

## 부록: Pre-built 컴포넌트 목록

### Chunker

| ID | 설명 | 필수 의존성 |
|---|---|---|
| `markdown-heading` | Markdown heading 기반 parent/child 분할 | (내장) |
| `sliding-window` | 고정 크기 슬라이딩 윈도우 | (내장) |

### Embedder

| ID | 설명 | 필수 의존성 |
|---|---|---|
| `openai:text-embedding-3-small` | OpenAI 임베딩 (소형) | `langchain-openai` |
| `openai:text-embedding-3-large` | OpenAI 임베딩 (대형) | `langchain-openai` |
| `openai:text-embedding-ada-002` | OpenAI 임베딩 (레거시) | `langchain-openai` |
| `huggingface:<model_name>` | HuggingFace 모델 | `langchain-huggingface`, `sentence-transformers` |

### VectorStore

| ID | 설명 | 필수 의존성 |
|---|---|---|
| `zvec` | 로컬 Zvec (기본) | (내장 또는 `zvec` 패키지) |
| `weaviate` | Weaviate | `weaviate-client`, `langchain-weaviate` |
| `chroma` | ChromaDB | `chromadb`, `langchain-chroma` |
| `qdrant` | Qdrant | `qdrant-client`, `langchain-qdrant` |

> 위 목록 외의 벡터 DB/임베딩 모델을 사용하려면 Python API로 커스텀 객체를 전달해야 한다.
