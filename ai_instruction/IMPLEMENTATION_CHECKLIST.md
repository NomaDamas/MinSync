# MinSync 구현 완료 체크리스트

> 버전: v0.2-draft
> 최종 갱신: 2026-02-20

모든 항목이 체크되어야 v0.1 릴리스 준비 완료로 간주한다.

---

## 1. 프로젝트 인프라

- [ ] `pyproject.toml`에 `[project.scripts]` 엔트리 등록 (`minsync = "minsync.cli:main"`)
- [ ] CLI 엔트리포인트 (`minsync/cli.py`) 구현 — argparse 또는 click 기반
- [ ] 공통 옵션 처리: `--help`, `--version`, `--verbose`, `--quiet`, `--format`
- [ ] 종료 코드 체계 구현 (0~5)
- [ ] `.minsync/` 디렉토리 경로 결정 로직 (git repo root 기준)
- [ ] Python API 진입점 (`minsync/__init__.py`에서 `MinSync` 클래스 export)

---

## 2. Python API (`MinSync` 클래스)

### 코어 클래스
- [ ] `MinSync` 클래스 정의 (`minsync/core.py` 또는 `minsync/minsync.py`)
- [ ] 생성자: `repo_path`, `chunker=None`, `embedder=None`, `vector_store=None`
  - [ ] `chunker`/`embedder`/`vector_store`가 전달되면 config.yaml의 해당 설정 무시
  - [ ] 전달되지 않으면 config.yaml에서 pre-built 인스턴스 자동 생성
- [ ] `init()` 메서드 — CLI `minsync init`과 동일
- [ ] `sync()` 메서드 — CLI `minsync sync`와 동일, `SyncResult` 반환
- [ ] `query()` 메서드 — CLI `minsync query`와 동일, `list[QueryResult]` 반환
- [ ] `status()` 메서드 — CLI `minsync status`와 동일, `StatusResult` 반환
- [ ] `verify()` 메서드 — CLI `minsync verify`와 동일, `VerifyResult` 반환
- [ ] `check()` 메서드 — CLI `minsync check`와 동일, `CheckResult` 반환

### 결과 데이터 클래스
- [ ] `SyncResult`: from_commit, to_commit, files_processed, chunks_added, chunks_deleted, duration 등
- [ ] `QueryResult`: doc_id, path, heading_path, chunk_type, text, score, content_commit
- [ ] `StatusResult`: repo_id, ref, last_synced_commit, current_head, state, pending_txn 등
- [ ] `VerifyResult`: basic_checks, file_checks, ignored_stale, all_passed
- [ ] `CheckResult`: embedder_ok, vectorstore_ok, git_ok, errors

### CLI = Thin Wrapper
- [ ] CLI의 각 커맨드는 `MinSync` 메서드를 호출하고 결과를 format에 맞게 출력하는 역할만 수행
- [ ] Python API에서 발생하는 예외를 CLI에서 종료 코드로 변환

### 테스트 통과
- [ ] Python API로 init/sync/query/status/verify/check를 호출하여 CLI와 동일 결과 확인

---

## 3. `minsync init`

### 핵심 기능
- [ ] git 저장소 여부 확인 로직
- [ ] root commit hash (repo_id) 추출: `git rev-list --max-parents=0 HEAD`
- [ ] 빈 저장소 (커밋 없음) 감지 및 에러 처리
- [ ] `.minsync/` 디렉토리 생성
- [ ] `.minsync/config.yaml` 생성 (기본값 포함, include 없음)
- [ ] 기존 `.minsync/` 존재 시 에러 + `--force` 옵션으로 덮어쓰기
- [ ] 초기화 완료 메시지 출력 (repo_id, collection, chunker, embedder, vectorstore 표시)

### CLI 옵션
- [ ] `--collection <name>`
- [ ] `--embedder <id>` (pre-built ID만)
- [ ] `--chunker <id>` (pre-built ID만)
- [ ] `--force`

### 테스트 통과
- [ ] T01: 기본 초기화
- [ ] T02-a: 비 git 저장소 에러
- [ ] T02-b: 중복 초기화 에러
- [ ] T02-c: --force 재초기화
- [ ] T02-d: 빈 저장소 에러

---

## 4. `minsync sync`

### 핵심 알고리즘
- [ ] config.yaml 로드 및 유효성 검사
- [ ] **Lock 메커니즘**: `.minsync/lock` 파일 기반 exclusive lock
  - [ ] 락 획득 성공/실패 처리
  - [ ] `--wait` 옵션 (polling 대기)
  - [ ] stale lock 감지 (프로세스 생존 확인)
  - [ ] 정상 종료 시 lock 해제
  - [ ] 비정상 종료 시 lock 잔존 허용 (다음 실행에서 감지)
- [ ] **목표 커밋 결정**: `git rev-parse <ref>`
- [ ] **from_commit 결정**: cursor.json 읽기 (없으면 None → full index)
- [ ] **schema/embedder 불일치 감지**: cursor.json과 config 비교, 불일치 시 --full 필수 (exit 1)
- [ ] **변경 파일 목록**: `git diff --name-status from..to` 파싱
  - [ ] A (Added) 처리
  - [ ] M (Modified) 처리
  - [ ] D (Deleted) 처리
  - [ ] R (Renamed) 처리
  - [ ] `.minsyncignore` 필터링 적용 (include 없음, ignore 기반 제외만)
- [ ] **최초 sync** (cursor 없음): `git ls-tree -r --name-only` 으로 전체 파일 → 모두 A 취급
- [ ] **txn.json 기록**: sync_token, from/to commit, status 등
- [ ] **파일 내용 읽기**: `git show to_commit:<path>` (working tree가 아닌 git 스냅샷)

### Normalize
- [ ] 줄바꿈 통일 (`\r\n` → `\n`)
- [ ] 트레일링 공백 제거
- [ ] (옵션) 연속 공백 축약
- [ ] (옵션) Markdown frontmatter 처리

### Chunking
- [ ] Chunker 인터페이스 정의 (`schema_id() -> str`, `chunk(text, path) -> list[Chunk]`)
- [ ] Chunk 데이터 클래스: `chunk_type, text, heading_path, ...`
- [ ] 기본 Markdown chunker 구현 (heading 기반 parent/child)
- [ ] 기본 Text chunker 구현 (sliding window)
- [ ] LangChain base class 활용/상속

### 결정적 ID (Deterministic ID)
- [ ] content_hash 계산: `sha256(normalized_chunk_text)`
- [ ] doc_id 계산: `sha256(null_byte_join(repo_id, ref, path, chunk_schema_id, chunk_type, heading_path, content_hash, dup_index))`
- [ ] dup_index 계산 (동일 content_hash + heading_path 내 중복 시)

### Embedding
- [ ] Embedder 인터페이스 정의 (`id() -> str`, `embed(texts) -> list[list[float]]`)
- [ ] Pre-built embedder 팩토리: config의 id → 인스턴스 생성
- [ ] LangChain Embeddings 어댑터
- [ ] 배치 임베딩 (batch_size 설정)
- [ ] 패키지 미설치 시 명확한 에러 메시지

### Vector Store 연동
- [ ] VectorStore 인터페이스 정의
  - [ ] `upsert(docs)`
  - [ ] `update(docs)` — metadata-only 갱신
  - [ ] `fetch(ids)` — 존재 확인
  - [ ] `delete_by_filter(filter)`
  - [ ] `query(vector, filter, topk)`
  - [ ] `flush()`
- [ ] Pre-built vectorstore 팩토리: config의 id → 인스턴스 생성
- [ ] Zvec 어댑터 구현
- [ ] 패키지 미설치 시 명확한 에러 메시지

### .minsyncignore
- [ ] `.minsyncignore` 파일 파싱 (`.gitignore` 문법 100% 호환)
- [ ] 하드코딩 제외: `.minsync/`, `.minsyncignore`, `.git/`
- [ ] git-tracked 파일 목록에서 ignore 패턴 필터링 (`.gitignore`에 의한 untracked 파일은 `git ls-tree`/`git diff`에 나타나지 않으므로 별도 처리 불필요)
- [ ] git diff 결과에서도 ignore 패턴 필터링
- [ ] `.minsyncignore` 변경 시 `--full` 불필요 (증분 sync + verify로 수렴)

### Sync 파일 단위 처리
- [ ] **Deleted**: `delete_by_filter(repo_id AND ref AND path)`
- [ ] **Added/Modified/Renamed**:
  - [ ] 파일 내용 로드 (`git show`)
  - [ ] normalize 적용
  - [ ] 청킹
  - [ ] doc_id 목록 생성
  - [ ] `fetch(ids)` → 존재/미존재 분류
  - [ ] 존재: `update(seen_token, path, heading_path)`
  - [ ] 미존재: embed → `upsert(전체 필드)`
  - [ ] **Sweep**: `delete_by_filter(path AND seen_token != sync_token)`

### Commit (커밋 단계)
- [ ] `collection.flush()`
- [ ] cursor.json 원자적 갱신 (temp → fsync → rename)
- [ ] txn.json 삭제
- [ ] lock 해제

### 크래시 복구
- [ ] txn.json 존재 감지 → 복구 모드 진입
- [ ] stale lock 처리 (PID 기반 또는 타임아웃)
- [ ] 복구 시 cursor의 from_commit부터 재처리
- [ ] mark+sweep으로 수렴 보장

### CLI 옵션
- [ ] `--ref <branch>`
- [ ] `--full`
- [ ] `--dry-run`
- [ ] `--verbose`
- [ ] `--batch-size <n>`
- [ ] `--wait`

### dry-run 모드
- [ ] DB 변경 없이 계획만 출력
- [ ] cursor/txn 불변 보장
- [ ] `.minsyncignore`로 제외된 파일도 "Ignored" 섹션에 표시

### full 모드
- [ ] cursor 무시
- [ ] 기존 repo_id/ref 범위 전체 삭제
- [ ] 전체 파일 재인덱싱

### 출력
- [ ] 정상 완료 통계 (files, chunks, duration)
- [ ] "Already up to date" 메시지
- [ ] dry-run 출력 (파일 목록 + 제외 목록)
- [ ] 진행률 표시

### 에러 처리
- [ ] 미초기화 감지 (exit 1)
- [ ] lock 충돌 (exit 3)
- [ ] ref 미발견 (exit 2)
- [ ] 임베딩 실패 (exit 5)
- [ ] 임베딩 패키지 미설치 (exit 5, 설치 안내 메시지)
- [ ] 벡터 DB 에러 (exit 4)
- [ ] 벡터 DB 패키지 미설치 (exit 4, 설치 안내 메시지)
- [ ] schema/embedder 불일치 (exit 1, --full 안내)

### 테스트 통과
- [ ] T03: 최초 전체 인덱싱
- [ ] T04: 증분 - 파일 추가
- [ ] T05: 증분 - 파일 수정
- [ ] T06: 증분 - 파일 삭제
- [ ] T07: 증분 - 파일 rename
- [ ] T08: 복합 변경
- [ ] T09: 다중 커밋 증분
- [ ] T10: dry-run
- [ ] T11: full rebuild
- [ ] T12: 결정적 ID (다른 위치)
- [ ] T13: 결정적 ID (rebuild)
- [ ] T14: 크래시 복구 (중간 중단)
- [ ] T15: 크래시 복구 (flush 전)
- [ ] T16: 크래시 복구 (cursor 전)
- [ ] T17: Lock 동시 실행 방지
- [ ] T18: 이미 최신
- [ ] T19: 미초기화 에러
- [ ] T28: mark+sweep 수렴
- [ ] T29: 대규모 증분 성능
- [ ] T30: .minsyncignore 필터링
- [ ] T31: schema/embedder 불일치
- [ ] T32: .minsyncignore 변경 후 증분 sync (full 불필요)

---

## 5. `minsync query`

### 핵심 기능
- [ ] 쿼리 텍스트 임베딩
- [ ] `collection.query(vector, filter, topk)` 호출
- [ ] 결과 정렬 및 포매팅

### 출력 포맷
- [ ] text 포맷 (기본): rank, path, heading, score, 텍스트 미리보기
- [ ] json 포맷: valid JSON
- [ ] jsonl 포맷
- [ ] `--show-score` 옵션

### CLI 옵션
- [ ] 위치 인자: 쿼리 텍스트
- [ ] `--ref <branch>`
- [ ] `--k <n>`
- [ ] `--filter <expr>`
- [ ] `--format <fmt>`
- [ ] `--show-score`

### 에러 처리
- [ ] 미초기화 (exit 1)
- [ ] 빈 쿼리 문자열 (exit 1)
- [ ] 빈 인덱스 경고 (exit 0, 0건)
- [ ] 임베딩 실패 (exit 5)

### 테스트 통과
- [ ] T20: 기본 검색
- [ ] T21: JSON 출력
- [ ] T22: 빈 인덱스
- [ ] T23: 빈 쿼리

---

## 6. `minsync status`

### 핵심 기능
- [ ] config.yaml 로드
- [ ] cursor.json 로드 (없으면 "never synced")
- [ ] 현재 HEAD 조회 (`git rev-parse`)
- [ ] cursor와 HEAD 비교 → 상태 결정
- [ ] txn.json 잔존 여부 확인
- [ ] 상태 문자열: `UP_TO_DATE`, `OUT_OF_DATE`, `NOT_SYNCED`, `INTERRUPTED`

### 출력 포맷
- [ ] text 포맷 (기본): 사람 읽기 좋은 형식
- [ ] json 포맷: valid JSON

### 에러 처리
- [ ] 미초기화 (exit 1)

### 테스트 통과
- [ ] T24-a: NOT_SYNCED 상태
- [ ] T24-b: UP_TO_DATE 상태
- [ ] T24-c: OUT_OF_DATE 상태
- [ ] T24-d: JSON 출력

---

## 7. `minsync check`

### 핵심 기능
- [ ] **Git 확인**: 저장소 유효성, repo_id, ref, HEAD
- [ ] **Embedder 확인**: 테스트 텍스트로 임베딩 호출 시도
  - [ ] 성공: dimension, 응답시간 표시
  - [ ] 실패: 에러 메시지 (API 키 미설정, 패키지 미설치 등)
- [ ] **VectorStore 확인**: 컬렉션 접근 시도
  - [ ] 성공: 문서 수 표시
  - [ ] 컬렉션 미존재: "will be created on first sync" 표시
  - [ ] 실패: 에러 메시지 (연결 거부, 패키지 미설치 등)

### 출력 포맷
- [ ] text 포맷 (기본)
- [ ] json 포맷: valid JSON

### 종료 코드
- [ ] 모든 체크 통과: exit 0
- [ ] 1개 이상 실패: exit 1

### 테스트 통과
- [ ] T33: 정상 health check
- [ ] T34: 임베딩 실패 감지
- [ ] T35: 벡터 DB 연결 실패 감지
- [ ] T36: 패키지 미설치 감지

---

## 8. `minsync verify`

### 핵심 기능
- [ ] **기본 검증**:
  - [ ] cursor.json 유효성
  - [ ] cursor의 commit이 git에 존재하는지
  - [ ] txn.json 잔존 여부
  - [ ] chunk_schema_id / embedder_id 일치 확인
  - [ ] 컬렉션 접근 및 기본 통계
- [ ] **`.minsyncignore` 잔존 검사** (항상 수행):
  - [ ] 현재 `.minsyncignore` 패턴 로드
  - [ ] 벡터 DB의 모든 고유 path 조회
  - [ ] ignore 패턴에 매칭되는 path 검출
  - [ ] 매칭 시 "IGNORED_STALE" 보고
  - [ ] `--fix` 시 해당 path의 청크 삭제 (`delete_by_filter`)
- [ ] **샘플 검증** (`--sample N`):
  - [ ] N개 파일 랜덤 선택
  - [ ] 파일 → 청킹 → doc_id 산출 → fetch → 존재 확인
  - [ ] 해당 path의 stale doc 확인
- [ ] **전체 검증** (`--all`):
  - [ ] 모든 대상 파일에 대해 샘플 검증과 동일 수행
  - [ ] 삭제된 파일의 잔존 doc 확인

### CLI 옵션
- [ ] `--ref <branch>`
- [ ] `--sample <n>`
- [ ] `--all`
- [ ] `--fix`
- [ ] `--format <fmt>`

### --fix 동작
- [ ] `.minsyncignore` 잔존 청크 삭제
- [ ] 일반 불일치 발견 시 `sync --full` 동등 동작 실행
- [ ] 수정 후 재검증

### 테스트 통과
- [ ] T25: 정상 검증
- [ ] T26: 불일치 감지
- [ ] T27: --fix 자동 수정
- [ ] T37: .minsyncignore 잔존 감지
- [ ] T38: .minsyncignore 잔존 --fix 삭제

---

## 9. 인덱싱 대상 결정 및 `.minsyncignore`

### .gitignore (1차 필터 — 자동)
- [ ] minsync는 `git ls-tree`와 `git diff`로 파일 목록을 얻으므로, `.gitignore`에 의해 untracked된 파일은 자동으로 보이지 않음 (별도 구현 불필요, git이 처리)
- [ ] 이 동작이 문서에 명시되어 있어야 함 (유저가 `.minsyncignore`에 중복 기재하지 않도록)

### .minsyncignore (2차 필터 — 명시적)
- [ ] `.gitignore` 문법 100% 호환 파서 구현 (또는 `pathspec` 라이브러리 활용)
- [ ] 네거티브 패턴 (`!`) 지원
- [ ] 디렉토리 패턴 (`dir/`) 지원
- [ ] 글로브 패턴 (`**/*.py`, `*.md`) 지원
- [ ] 하드코딩 제외: `.minsync/`, `.minsyncignore`, `.git/`
- [ ] 변경 시 `--full` 불필요: sync + verify 조합으로 수렴
- [ ] 최종 인덱싱 대상 = (git-tracked 파일) − (.minsyncignore 매칭) − (하드코딩 제외)

### 테스트 통과
- [ ] T30: .minsyncignore 기본 필터링
- [ ] T32: .minsyncignore 변경 후 sync (full 불필요)
- [ ] T37: verify에서 잔존 감지
- [ ] T38: verify --fix로 잔존 삭제

---

## 10. 데이터 모델

### config.yaml 스키마
- [ ] version 필드
- [ ] repo_id
- [ ] ref
- [ ] collection (name, path)
- [ ] chunker (id, options) — pre-built ID만
- [ ] embedder (id, batch_size) — pre-built ID만
- [ ] vectorstore (id, options) — pre-built ID만
- [ ] normalize 옵션
- [ ] ~~include~~ (없음 — git-tracked 전체가 기본. `.gitignore`에 의해 untracked된 파일은 git 자체에서 제외되므로 minsync에서 별도 처리 불필요)

### cursor.json 스키마
- [ ] repo_id
- [ ] ref
- [ ] last_synced_commit
- [ ] chunk_schema_id
- [ ] embedder_id
- [ ] collection_path
- [ ] updated_at (ISO8601)

### txn.json 스키마
- [ ] repo_id
- [ ] ref
- [ ] from_commit
- [ ] to_commit
- [ ] sync_token
- [ ] chunk_schema_id
- [ ] embedder_id
- [ ] status ("running" | "failed")
- [ ] started_at
- [ ] last_progress_at

### Vector DB Document 스키마
- [ ] id (doc_id)
- [ ] embedding
- [ ] text
- [ ] repo_id
- [ ] ref
- [ ] path
- [ ] ext
- [ ] chunk_schema_id
- [ ] chunk_type ("parent" / "child")
- [ ] heading_path (optional)
- [ ] content_hash
- [ ] content_commit
- [ ] seen_token

---

## 11. 코어 인터페이스 (플러그 아키텍처)

### Chunker
- [ ] 추상 인터페이스: `schema_id() -> str`, `chunk(text, path) -> list[Chunk]`
- [ ] Chunk 데이터 클래스: `chunk_type, text, heading_path, ...`
- [ ] LangChain base class 활용/상속
- [ ] Pre-built: MarkdownHeadingChunker
- [ ] Pre-built: SlidingWindowChunker
- [ ] Python API로 커스텀 Chunker 전달 가능

### Embedder
- [ ] 추상 인터페이스: `id() -> str`, `embed(texts) -> list[list[float]]`
- [ ] LangChain Embeddings 어댑터
- [ ] Pre-built embedder 팩토리 (config id → 인스턴스)
- [ ] 배치 처리 지원
- [ ] 패키지 미설치 시 명확한 에러 (설치 안내 포함)
- [ ] Python API로 커스텀 Embedder 전달 가능

### VectorStore
- [ ] 추상 인터페이스: `upsert`, `update`, `fetch`, `delete_by_filter`, `query`, `flush`
- [ ] Pre-built vectorstore 팩토리 (config id → 인스턴스)
- [ ] Zvec 어댑터 구현
- [ ] LangChain VectorStore 어댑터 (Weaviate, Chroma, Qdrant 등)
- [ ] scalar 필드 인덱스 설정 (repo_id, ref, path, chunk_schema_id, seen_token)
- [ ] 패키지 미설치 시 명확한 에러 (설치 안내 포함)
- [ ] Python API로 커스텀 VectorStore 전달 가능

---

## 12. Git 연동

- [ ] git repo root 탐지 (`git rev-parse --show-toplevel`)
- [ ] repo_id 추출 (`git rev-list --max-parents=0 HEAD`)
- [ ] HEAD / ref 해석 (`git rev-parse`)
- [ ] diff 파싱 (`git diff --name-status`)
- [ ] 파일 내용 읽기 (`git show commit:path`)
- [ ] 파일 목록 (`git ls-tree -r --name-only`)
- [ ] 커밋 수 계산 (`git rev-list --count from..to`)

---

## 13. CI/CD 지원

- [ ] 비대화형 환경에서 모든 커맨드가 정상 동작
- [ ] `--format json` 출력으로 스크립트 파싱 지원
- [ ] GitHub Actions 워크플로우 예제 문서화
- [ ] 복구 흐름: `status → check → sync → verify --all --fix → status`
- [ ] exit code 기반 분기 처리 가능

---

## 14. 품질 기준

### 정합성
- [ ] 모든 sync 후 verify --all 통과
- [ ] 삭제된 파일의 청크가 0개
- [ ] 살아있는 파일은 최신 스냅샷 청크만 존재
- [ ] `.minsyncignore`에 해당하는 파일의 청크가 verify 후 0개

### 결정성
- [ ] 같은 커밋에서 rebuild → 동일 doc_id 집합
- [ ] 다른 clone 위치에서 → 동일 doc_id 집합

### Crash-safe
- [ ] 어느 시점에서 중단되어도 재실행으로 수렴
- [ ] cursor는 완전한 sync 후에만 갱신

### 성능
- [ ] 변경 파일만 처리 (증분)
- [ ] 배치 임베딩
- [ ] 배치 fetch/update/upsert

### 외부 의존성
- [ ] minsync 코어에 LangChain 구체 구현체 의존성 없음
- [ ] 패키지 미설치 시 명확한 에러 + 설치 안내

---

## 진행 상황 요약

| 영역 | 전체 항목 수 | 완료 | 진행률 |
|------|------------|------|--------|
| 프로젝트 인프라 | 6 | 0 | 0% |
| Python API | 15 | 0 | 0% |
| init | 12 | 0 | 0% |
| sync | 66 | 0 | 0% |
| query | 15 | 0 | 0% |
| status | 9 | 0 | 0% |
| check | 11 | 0 | 0% |
| verify | 19 | 0 | 0% |
| .minsyncignore | 10 | 0 | 0% |
| 데이터 모델 | 28 | 0 | 0% |
| 코어 인터페이스 | 18 | 0 | 0% |
| Git 연동 | 7 | 0 | 0% |
| CI/CD 지원 | 5 | 0 | 0% |
| 품질 기준 | 11 | 0 | 0% |
| **합계** | **232** | **0** | **0%** |
