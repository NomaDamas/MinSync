# MinSync 유저 시나리오

> 버전: v0.2-draft
> 최종 갱신: 2026-02-20

이 문서는 MinSync의 모든 기능을 엔드 유저 관점에서 시나리오로 기술한다.
각 시나리오는 "누가, 어떤 상황에서, 무엇을 하고, 무엇을 기대하는가"를 명시한다.

## 설계 원칙 (시나리오 전반에 적용)

1. **CLI = Python API의 Thin Wrapper**: 모든 CLI 기능은 동일한 Python API를 호출한다. 사용자는 CLI 또는 Python 코드 어디서든 동일한 결과를 얻는다.
2. **커스텀 플러그인은 Python으로**: `config.yaml`에서는 pre-built 메서드만 선택. 커스텀 chunker/embedder/vectorstore는 Python 코드로 등록한다.
3. **LangChain 의존성은 사용자 책임**: minsync는 LangChain 기반 어댑터 인터페이스를 제공하지만, 구체적 벡터 DB나 임베딩 모델의 dependency는 번들하지 않는다.
4. **상태 디렉토리**: `.minsync/` (프로젝트명 반영)
5. **인덱싱 대상**: git에 트래킹되는 모든 파일이 기본 대상. `.gitignore`에 의해 untracked된 파일은 git 자체에서 제외되므로 minsync에도 보이지 않는다. 추가로 `.minsyncignore`(``.gitignore``와 동일 문법)로 git-tracked이지만 인덱싱을 원치 않는 파일을 제외한다.
6. **CI/CD 우선 설계**: GitHub Actions 등에서 실행되는 것을 전제로 설계.

---

## 시나리오 1: 최초 설정 및 전체 인덱싱

### 상황
개발자 A가 마크다운 문서로 구성된 지식 베이스 저장소를 clone한 직후.
저장소에는 `docs/` 아래 5개의 마크다운 파일과 `src/` 아래 Python 소스가 있다.

### 행동 순서

```bash
# 1. 저장소 clone
git clone https://github.com/team/knowledge-base.git
cd knowledge-base

# 2. MinSync 초기화
minsync init

# 3. (선택) .minsyncignore 작성 — 인덱싱 제외 대상 지정
cat > .minsyncignore << 'EOF'
*.pyc
__pycache__/
node_modules/
*.lock
EOF

# 4. 초기 인덱스 구축
minsync sync

# 5. 상태 확인
minsync status

# 6. 검색 테스트
minsync query "인증 프로세스"
```

### 동일 작업을 Python API로 수행

```python
from minsync import MinSync

ms = MinSync(repo_path="/path/to/knowledge-base")
ms.init()
ms.sync()
status = ms.status()
results = ms.query("인증 프로세스", k=10)
```

### 기대 결과

- `minsync init`:
  - `.minsync/` 디렉토리가 생성됨
  - `.minsync/config.yaml`이 생성되며 repo_id, ref, chunker, embedder 설정 포함
  - "Initialized MinSync" 메시지가 출력됨

- `minsync sync`:
  - cursor.json이 없으므로 전체 인덱싱(full index) 모드로 실행됨
  - git에 트래킹되는 모든 파일이 대상 (`.gitignore`에 의해 untracked된 파일은 자동 제외)
  - 그 중 `.minsyncignore`에 해당하지 않는 파일들만 처리
  - 각 파일의 청크가 벡터 DB에 upsert됨
  - 완료 후 `.minsync/cursor.json`이 생성됨 (last_synced_commit = 현재 HEAD)
  - `.minsync/txn.json`은 삭제됨
  - "MinSync sync completed" 메시지에 처리 통계가 포함됨

- `minsync status`:
  - "UP TO DATE" 표시
  - last_synced와 current_head가 동일한 커밋을 가리킴

- `minsync query "인증 프로세스"`:
  - 관련 청크가 유사도 점수와 함께 반환됨
  - 결과에 path, heading, 텍스트 미리보기가 포함됨

---

## 시나리오 2: 증분 동기화 (문서 추가/수정/삭제)

### 상황
시나리오 1 이후, 팀원 B가 문서를 수정하여 push했고, 개발자 A가 pull한 상태.
변경 내용: 1개 파일 추가, 1개 파일 수정, 1개 파일 삭제.

### 행동 순서

```bash
# 1. 최신 변경 pull
git pull origin main

# 2. 증분 동기화
minsync sync

# 3. 상태 확인
minsync status

# 4. 삭제된 문서 검색
minsync query "삭제된 문서 제목의 키워드"
```

### 기대 결과

- `minsync sync`:
  - cursor.json의 last_synced_commit과 현재 HEAD 사이의 diff만 처리
  - 추가된 파일: 새 청크가 upsert됨
  - 수정된 파일: 변경된 청크만 새로 임베딩, 불변 청크는 seen_token만 갱신, stale 청크는 sweep으로 삭제
  - 삭제된 파일: 해당 path의 모든 청크가 delete_by_filter로 삭제됨
  - 출력에 "3 added, 10 modified, 2 deleted" 등의 통계

- 삭제된 문서 검색: 해당 문서의 청크가 결과에 포함되지 않아야 함

---

## 시나리오 3: 파일 이름 변경 (Rename) 처리

### 상황
`docs/old-name.md` 파일이 `docs/new-name.md`로 rename되어 커밋됨. 내용은 동일.

### 행동 순서

```bash
git pull origin main
minsync sync
minsync query "old-name 파일의 내용 키워드"
```

### 기대 결과

- `minsync sync`:
  - git diff가 R (Rename) 상태를 감지
  - rename인 경우 Added/Modified와 동일하게 처리 (새 path로 청크 생성)
  - **mark+sweep** 메커니즘에 의해:
    - 새 path (`docs/new-name.md`)의 청크들은 seen_token이 찍힘
    - 구 path (`docs/old-name.md`)의 청크들은 삭제됨 (D 처리 또는 sweep)
  - 내용이 동일하므로 content_hash가 같지만, path가 달라 doc_id는 새로 생성됨
  - 새 doc_id에 대해 임베딩 계산 필요 (path가 ID 계산에 포함되므로)

- 검색 결과에서 path가 `docs/new-name.md`로 표시됨

---

## 시나리오 4: dry-run으로 변경 사항 미리보기

### 상황
대규모 변경이 있었고, sync 전에 어떤 파일이 처리될지 미리 확인하고 싶음.

### 행동 순서

```bash
git pull origin main
minsync sync --dry-run
# 확인 후 실제 sync
minsync sync
```

### 기대 결과

- `minsync sync --dry-run`:
  - 벡터 DB에 어떤 변경도 가해지지 않음
  - cursor.json이 갱신되지 않음
  - txn.json이 생성되지 않음
  - Added/Modified/Deleted 파일 목록이 출력됨
  - 예상 처리 파일 수가 표시됨

- 이후 `minsync sync`는 동일한 변경을 실제로 수행

---

## 시나리오 5: 크래시 복구 (Crash Recovery)

### 상황
`minsync sync` 실행 중 프로세스가 강제 종료됨 (SIGKILL, OOM, CI timeout 등).
5개 파일 중 3개 처리 후 크래시.

### 행동 순서

```bash
# sync 중 크래시 발생 (외부 요인)
minsync sync   # 이 과정에서 kill됨

# 재실행
minsync status  # 상태 확인
minsync sync    # 복구 실행
minsync verify  # 정합성 검증
```

### 기대 결과

- 크래시 시점:
  - `.minsync/txn.json` 존재 (status: "running")
  - `.minsync/cursor.json`은 **갱신되지 않음** (이전 상태 유지)
  - `.minsync/lock` 파일이 남아있을 수 있음 (stale lock)

- `minsync status`:
  - "INTERRUPTED" 상태 표시
  - txn.json의 from/to 커밋 정보 표시

- `minsync sync` (재실행):
  - stale lock 감지 및 회수 (프로세스가 살아있지 않으면)
  - txn.json 감지 → "Recovering from interrupted sync..." 메시지
  - cursor의 last_synced_commit부터 다시 전체 처리
  - **mark+sweep 알고리즘** 덕분에:
    - 이미 처리된 3개 파일의 청크: 존재 확인 → seen_token 갱신만 (임베딩 재계산 불필요)
    - 미처리 2개 파일: 정상 처리
    - 모든 파일에 대해 sweep 수행 → stale 청크 제거
  - 결과는 크래시 없이 실행한 것과 **동일**해야 함

- `minsync verify`:
  - "ALL CHECKS PASSED" 표시

---

## 시나리오 6: 전체 재구축 (Full Rebuild)

### 상황
embedder 설정을 변경했거나, 인덱스가 손상된 것으로 의심될 때.

### 행동 순서

```bash
# config에서 embedder 변경
vi .minsync/config.yaml  # embedder.id를 변경

# 전체 재구축
minsync sync --full

# 검증
minsync verify --all
```

### 기대 결과

- `minsync sync --full`:
  - cursor.json 무시
  - 해당 repo_id/ref 범위의 기존 벡터를 전부 삭제
  - 모든 파일을 처음부터 처리 (전체 인덱싱)
  - 완료 후 cursor.json 갱신 (chunk_schema_id, embedder_id도 업데이트)

- `minsync verify --all`:
  - 전체 파일에 대해 검증 수행
  - 모든 청크가 일치해야 함

---

## 시나리오 7: 결정적 ID 보장 (Deterministic ID)

### 상황
동일한 저장소를 두 곳에서 clone하고, 같은 커밋에서 sync를 실행했을 때 ID가 동일한지 확인.

### 행동 순서

```bash
# 위치 1
cd /tmp/repo-a
git clone https://github.com/team/knowledge-base.git .
minsync init && minsync sync

# 위치 2
cd /tmp/repo-b
git clone https://github.com/team/knowledge-base.git .
minsync init && minsync sync

# 두 위치의 doc_id 집합 비교
```

### 기대 결과

- 두 위치에서 생성된 모든 doc_id가 완전히 동일
- repo_id가 동일 (root commit hash 기반)
- 같은 텍스트 → 같은 content_hash → 같은 doc_id
- 순서에 무관하게 결정적

---

## 시나리오 8: 동시 실행 방지 (Lock)

### 상황
터미널 2개에서 동시에 `minsync sync`를 실행.

### 행동 순서

```bash
# 터미널 1
minsync sync  # 정상 진행 중

# 터미널 2 (동시에)
minsync sync         # 즉시 에러
minsync sync --wait  # 대기 후 실행
```

### 기대 결과

- 터미널 1: 정상 진행
- 터미널 2 (`minsync sync`):
  - exit code 3
  - "Error: another sync is in progress. Use --wait to wait." 메시지
- 터미널 2 (`minsync sync --wait`):
  - "Waiting for lock..." 메시지 출력
  - 터미널 1 완료 후 lock 획득하여 정상 진행
  - 이미 최신이면 "Already up to date." 출력

---

## 시나리오 9: .minsyncignore를 통한 인덱싱 제외

### 상황
`.gitignore`에 의해 untracked된 파일(빌드 산출물 등)은 minsync에 보이지 않아 자동 제외된다.
그러나 git-tracked이면서 인덱싱 대상에서 제외하고 싶은 파일(소스 코드, 바이너리 등)이 있을 수 있다.

### 행동 순서

```bash
# .minsyncignore 작성 (.gitignore와 동일 문법)
cat > .minsyncignore << 'EOF'
# 소스코드 제외
src/**/*.py
# 빌드 산출물
dist/
build/
# 이미지/바이너리
*.png
*.jpg
*.bin
# 특정 디렉토리
vendor/
EOF

# sync 실행
minsync sync
```

### 기대 결과

- `.minsyncignore`에 매칭되는 파일은 인덱싱되지 않음
- git에 트래킹되는 파일 중 `.minsyncignore`에 해당하지 않는 파일만 처리됨
- `.minsyncignore` 자체는 인덱싱 대상이 아님 (자동 제외)
- `.gitignore`에 이미 명시된 파일은 git-tracked가 아니므로 minsync에 보이지 않음 → `.minsyncignore`에 중복 기재할 필요 없음

### .minsyncignore 변경 후 시나리오

```bash
# 이전에 인덱싱된 파일을 .minsyncignore에 추가
echo "docs/internal/**" >> .minsyncignore

# 일반 sync만으로 충분 (--full 불필요)
minsync sync
```

- `minsync sync`: 증분 sync에서 `.minsyncignore` 변경을 감지
  - 새로 ignore된 파일의 기존 청크: 이번 sync의 처리 대상에서 제외 → sweep에서 걸러짐
  - **`--full` 없이도 정상 동작**: verify에서 잔존 청크를 감지하고 삭제

```bash
# verify가 .minsyncignore에 해당하는 잔존 청크를 감지 및 삭제
minsync verify
```

- `minsync verify`: `.minsyncignore`에 해당하는 path의 문서가 벡터 DB에 남아있으면 경고 + 자동 삭제

---

## 시나리오 10: Python API를 통한 커스텀 파이프라인

### 상황
기본 제공 chunker/embedder 대신, 자체 구현한 커스텀 컴포넌트를 사용하고 싶음.

### 행동 순서 (Python)

```python
from minsync import MinSync
from minsync.chunker import Chunker, Chunk
from minsync.embedder import Embedder
from langchain_community.vectorstores import Weaviate  # 사용자가 직접 설치

# 커스텀 Chunker 구현
class MyChunker(Chunker):
    def schema_id(self) -> str:
        return "my-custom-chunker-v1"

    def chunk(self, text: str, path: str) -> list[Chunk]:
        # 커스텀 청킹 로직
        ...

# 커스텀 Embedder 구현
class MyEmbedder(Embedder):
    def id(self) -> str:
        return "my-custom-embedder-v1"

    def embed(self, texts: list[str]) -> list[list[float]]:
        # 커스텀 임베딩 로직
        ...

# 커스텀 컴포넌트로 MinSync 실행
ms = MinSync(
    repo_path="/path/to/repo",
    chunker=MyChunker(),
    embedder=MyEmbedder(),
    # vector_store도 커스텀 가능
)
ms.init()
ms.sync()
results = ms.query("search text")
```

### 기대 결과

- config.yaml의 설정 대신 Python 코드로 전달된 컴포넌트가 사용됨
- 커스텀 chunker의 `schema_id()`가 cursor에 기록됨
- 커스텀 embedder의 `id()`가 cursor에 기록됨
- 나머지 sync 알고리즘 (lock, mark+sweep, crash recovery)은 동일하게 작동

---

## 시나리오 11: LangChain 벡터 DB 연동 (외부 의존성)

### 상황
Weaviate를 벡터 DB로 사용하고 싶음. minsync에는 Weaviate dependency가 포함되어 있지 않으므로 직접 설치 필요.

### 행동 순서

```bash
# 1. 사용자가 직접 의존성 설치
pip install weaviate-client langchain-weaviate

# 2. config.yaml에서 pre-built vectorstore로 weaviate 선택
cat .minsync/config.yaml
```

```yaml
vectorstore:
  id: "weaviate"
  options:
    url: "http://localhost:8080"
    collection_name: "minsync_docs"
```

```bash
# 3. health check로 연결 확인
minsync check

# 4. sync 실행
minsync sync
```

### 기대 결과

- `minsync check`:
  - Weaviate 연결 확인
  - 임베딩 모델 호출 테스트
  - "All checks passed" 또는 구체적 에러 메시지

- weaviate-client가 설치되지 않은 경우:
  - `minsync sync` 또는 `minsync check` 실행 시 명확한 에러:
    "Error: 'weaviate-client' package is required for weaviate vectorstore. Install it with: pip install weaviate-client langchain-weaviate"

---

## 시나리오 12: Health Check (minsync check)

### 상황
sync 전에 임베딩 모델과 벡터 DB 연결이 정상인지 확인하고 싶음.

### 행동 순서

```bash
minsync check
```

### 기대 결과 — 정상

```
MinSync Health Check
  Embedder:     openai:text-embedding-3-small ... OK (dim=1536, 0.3s)
  VectorStore:  local (zvec) ... OK (collection exists, 142 docs)
  Git:          OK (repo_id=abc123, ref=main, HEAD=def456)

All checks passed.
```

### 기대 결과 — 임베딩 실패

```
MinSync Health Check
  Embedder:     openai:text-embedding-3-small ... FAIL
    Error: OpenAI API key not set. Set OPENAI_API_KEY environment variable.
  VectorStore:  local (zvec) ... OK
  Git:          OK

1 check failed.
```

### 기대 결과 — 벡터 DB 연결 실패

```
MinSync Health Check
  Embedder:     openai:text-embedding-3-small ... OK
  VectorStore:  weaviate (http://localhost:8080) ... FAIL
    Error: Connection refused. Is Weaviate running?
  Git:          OK

1 check failed.
```

---

## 시나리오 13: GitHub Actions CI/CD 파이프라인

### 상황
팀에서 문서 저장소에 push할 때마다 자동으로 벡터 인덱스를 동기화하고 싶음.

### GitHub Actions 워크플로우 예제

```yaml
# .github/workflows/minsync.yml
name: MinSync Index Sync

on:
  push:
    branches: [main]

jobs:
  sync:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0  # 전체 히스토리 필요 (git diff용)

      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Install MinSync and dependencies
        run: |
          pip install minsync
          pip install langchain-openai  # 임베딩 모델 의존성

      - name: Initialize MinSync (first run only)
        run: |
          if [ ! -d ".minsync" ]; then
            minsync init
          fi

      - name: Health Check
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        run: minsync check

      - name: Sync Index
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        run: minsync sync --verbose

      - name: Verify Index
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        run: minsync verify --sample 5

      - name: Commit cursor update
        run: |
          git config user.name "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
          git add .minsync/cursor.json
          git diff --cached --quiet || git commit -m "chore: update minsync cursor"
          git push
```

### 기대 결과

- push마다 자동으로 증분 sync 실행
- `.minsync/cursor.json`이 저장소에 커밋되어 다음 실행에서 증분 기준점으로 사용
- health check로 API 키/벡터 DB 연결 사전 검증
- verify로 sync 결과의 정합성 확인

---

## 시나리오 14: GitHub Actions — 실패 후 복구

### 상황
이전 CI 실행에서 `minsync sync`가 중간에 실패함 (네트워크 에러, 타임아웃 등).
다음 실행에서 자동으로 복구되어야 함.

### GitHub Actions 워크플로우 (복구 포함)

```yaml
# .github/workflows/minsync-with-recovery.yml
name: MinSync Sync with Recovery

on:
  push:
    branches: [main]

jobs:
  sync:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Install dependencies
        run: pip install minsync langchain-openai

      - name: Initialize if needed
        run: |
          if [ ! -d ".minsync" ]; then
            minsync init
          fi

      - name: Check status and recover if needed
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        run: |
          echo "=== Current Status ==="
          minsync status

          # status의 exit code와 출력으로 상태 판단
          # txn.json이 남아있으면 interrupted 상태 → sync가 자동 복구

      - name: Health Check
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        run: minsync check

      - name: Sync (자동 복구 포함)
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        run: |
          # sync는 txn.json이 있으면 자동으로 recovery 모드로 진입
          minsync sync --verbose

      - name: Verify integrity
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        run: |
          # 정합성 검증 — 실패 시 자동 수정
          minsync verify --all --fix

      - name: Final status
        run: minsync status

      - name: Commit state
        run: |
          git config user.name "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
          git add .minsync/cursor.json
          git diff --cached --quiet || git commit -m "chore: update minsync cursor"
          git push
```

### 복구 흐름 요약

```
Actions 실행 시작
  ↓
minsync status          ← 현재 상태 확인 (INTERRUPTED / OUT OF DATE / UP TO DATE)
  ↓
minsync check           ← 임베딩/벡터 DB health check
  ↓
minsync sync --verbose  ← txn.json 존재 시 자동 복구, 아니면 정상 증분 sync
  ↓
minsync verify --all --fix  ← 정합성 검증 + 불일치 시 자동 수정
  ↓
minsync status          ← 최종 상태 확인 (UP TO DATE여야 함)
  ↓
cursor.json 커밋/push
```

### 기대 결과

- 이전 실행이 실패했더라도 다음 실행에서 자동 복구됨
- `minsync status` → `minsync check` → `minsync sync` → `minsync verify --all --fix` → `minsync status` 순서로 **상태 확인 → 사전 검증 → 동기화 → 정합성 검증 → 최종 확인** 전체 흐름이 수행됨
- txn.json이 남아있는 경우 sync가 자동으로 복구 모드로 진입
- verify --fix가 남은 불일치를 자동 수정
- 최종 status에서 "UP TO DATE" 확인

---

## 시나리오 15: 설정 변경 — embedder/chunker 변경

### 상황
임베딩 모델을 교체하고 싶음 (비용/성능 조정).

### 행동 순서

```bash
# config에서 embedder 변경
vi .minsync/config.yaml  # embedder.id를 변경

# 전체 재구축 필요
minsync sync --full
```

### 기대 결과

- `chunk_schema_id`나 `embedder_id`가 바뀌면 `--full`이 **필수**
- 현재 cursor의 schema/embedder와 config가 다르면 sync 시 경고:
  "Warning: schema/embedder mismatch. Use --full to rebuild."
  (경고만 하고 증분 sync는 거부, exit 1)

### 주의: .minsyncignore 변경은 --full 불필요

```bash
echo "new-excluded-dir/" >> .minsyncignore
minsync sync     # --full 없이도 OK
minsync verify   # 잔존 청크 감지 및 삭제
```

---

## 시나리오 16: 대규모 저장소에서의 증분 성능

### 상황
1000개 이상의 파일이 있는 저장소에서 3개 파일만 변경된 후 sync.

### 행동 순서

```bash
git pull origin main  # 3개 파일만 변경됨
minsync sync --verbose
```

### 기대 결과

- git diff로 변경된 3개 파일만 식별
- 3개 파일만 처리 (나머지는 건드리지 않음)
- 임베딩 API 호출은 새로운 청크에 대해서만 발생
- verbose 모드에서 처리 대상 파일만 로그 확인
- 처리 시간이 전체 재구축 대비 극히 짧음

---

## 시나리오 17: 빈 결과 및 경계 조건

### 케이스

**17-a: 변경 없이 sync**
```bash
minsync sync  # 이미 최신 상태에서 실행
```
→ "Already up to date." 출력, exit 0

**17-b: 빈 쿼리**
```bash
minsync query ""
```
→ "Error: query text is required." exit 1

**17-c: 초기화 전 sync 시도**
```bash
cd /tmp/new-repo
git init && git commit --allow-empty -m "init"
minsync sync
```
→ "Error: not initialized. Run 'minsync init' first." exit 1

**17-d: 모든 파일이 .minsyncignore에 해당**
```bash
echo "*" > .minsyncignore
minsync sync
```
→ sync는 성공하지만 "0 files processed" 경고

---

## 시나리오 18: verify를 통한 정합성 검증

### 상황
sync가 정상적으로 완료되었는지 신뢰할 수 없는 상황.

### 행동 순서

```bash
# 기본 검증 (샘플 10개)
minsync verify

# 전체 검증
minsync verify --all

# 불일치 발견 시 자동 수정
minsync verify --fix
```

### 기대 결과

- `minsync verify`: 10개 파일 샘플링하여 doc_id 존재 확인
- `minsync verify --all`: 전체 파일 검증
- 불일치 시: 누락/잔존 청크 목록 출력, exit 1
- `minsync verify --fix`: full sync 실행하여 자동 복구
- `.minsyncignore`에 해당하는 파일의 잔존 청크도 감지 및 삭제

---

## 시나리오 19: JSON 출력으로 프로그래밍 연동

### 상황
CI/CD 파이프라인이나 다른 도구에서 MinSync 결과를 파싱해야 함.

### 행동 순서

```bash
# 상태를 JSON으로
minsync status --format json

# 검색 결과를 JSON으로
minsync query "API 설계" --format json --k 5

# 검증 결과를 JSON으로
minsync verify --format json

# health check를 JSON으로
minsync check --format json
```

### 기대 결과

- 모든 출력이 valid JSON으로 반환됨
- jq 등으로 파싱 가능
- CI/CD 스크립트에서 프로그래밍적으로 결과 활용 가능

---

## 시나리오 흐름 요약

```
clone → init → [.minsyncignore 작성] → check → sync (full) → verify
                                          ↓
                              [수정/커밋] → sync (incremental) → verify
                                          ↓
                              [크래시 발생] → status → sync (recovery) → verify
                                          ↓
                              [설정 변경] → sync --full → verify
                                          ↓
                              [.minsyncignore 변경] → sync → verify (잔존 삭제)

GitHub Actions:
  status → check → sync → verify --all --fix → status → cursor commit
```

### CLI ↔ Python API 대응표

| CLI 커맨드 | Python API |
|------------|------------|
| `minsync init` | `ms.init()` |
| `minsync sync` | `ms.sync()` |
| `minsync sync --full` | `ms.sync(full=True)` |
| `minsync sync --dry-run` | `ms.sync(dry_run=True)` |
| `minsync query "text"` | `ms.query("text")` |
| `minsync status` | `ms.status()` |
| `minsync verify` | `ms.verify()` |
| `minsync verify --all --fix` | `ms.verify(all=True, fix=True)` |
| `minsync check` | `ms.check()` |
