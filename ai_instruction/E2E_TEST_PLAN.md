# MinSync E2E 테스트 계획

> 버전: v0.2-draft
> 최종 갱신: 2026-02-20

이 문서는 MinSync의 모든 기능을 **엔드 투 엔드**로 검증하기 위한 테스트 계획이다.
각 테스트는 실제 git 저장소를 생성하고, 문서를 커밋하고, MinSync CLI (및 Python API)를 실행하여 결과를 검증한다.

---

## 설계 원칙

1. 모든 테스트는 CLI와 Python API 양쪽으로 검증 가능해야 한다.
2. `.ragit/`가 아닌 `.minsync/`를 사용한다.
3. `include` 패턴은 없다. git-tracked 모든 파일이 기본 대상이다. `.gitignore`에 의해 untracked된 파일은 git 자체에서 제외되므로 minsync에 보이지 않는다. 추가로 `.minsyncignore`로 git-tracked이지만 인덱싱 제외할 파일을 관리한다.
4. 외부 의존성(LangChain 벡터 DB 등)이 필요한 테스트는 mock 또는 로컬 벡터 DB를 사용한다.
5. CI/CD 시나리오 테스트를 포함한다.

---

## 테스트 환경 구성

### 공통 Fixture: 테스트 git 저장소

모든 테스트는 아래의 임시 git 저장소를 기반으로 한다.
`pytest`의 `tmp_path` 또는 `tempfile.mkdtemp()`를 이용해 격리된 환경에서 실행한다.

```
test_repo/
├── docs/
│   ├── guide.md
│   ├── api.md
│   ├── auth/
│   │   ├── login.md
│   │   └── oauth.md
│   └── faq.md
├── src/
│   ├── main.py
│   └── utils.py
├── notes/
│   └── meeting.txt
├── .minsyncignore        # (일부 테스트에서 생성)
└── README.md
```

### 샘플 문서 내용

**docs/guide.md**
```markdown
# User Guide

## Getting Started

MinSync is a git-native vector index sync engine.
It detects changes via git diff and incrementally updates the vector database.

## Installation

Install MinSync using pip:

pip install minsync

## Configuration

After installation, run `minsync init` in your git repository.
```

**docs/api.md**
```markdown
# API Reference

## Sync API

The sync command processes file changes and updates the vector index.

### Parameters

- `--ref`: Target branch (default: main)
- `--full`: Force full rebuild

## Query API

Search the vector index with natural language queries.
```

**docs/auth/login.md**
```markdown
# Authentication

## Login Process

The login process begins with the user entering their credentials.
The system validates the credentials against the authentication provider.

## Session Management

After successful login, a session token is issued.
```

**docs/auth/oauth.md**
```markdown
# OAuth Integration

## OAuth2 Flow

For third-party authentication, we use OAuth2.
The flow follows the authorization code grant type.

## Provider Configuration

Supported providers: Google, GitHub, Microsoft.
```

**docs/faq.md**
```markdown
# FAQ

## How does sync work?

MinSync uses git diff to detect file changes between commits.

## Is it safe to run during CI?

Yes, MinSync uses file locking to prevent concurrent access.
```

**src/main.py**
```python
def main():
    print("Hello, world!")

if __name__ == "__main__":
    main()
```

**src/utils.py**
```python
def helper():
    return 42
```

### 헬퍼 함수

```python
def create_test_repo(tmp_path, files: dict[str, str]) -> Path:
    """임시 git 저장소 생성 및 초기 커밋"""

def add_commit(repo_path, files: dict[str, str], message: str) -> str:
    """파일 추가/수정 후 커밋, commit hash 반환"""

def delete_commit(repo_path, paths: list[str], message: str) -> str:
    """파일 삭제 후 커밋"""

def rename_commit(repo_path, old: str, new: str, message: str) -> str:
    """파일 이름 변경 후 커밋"""

def run_minsync(repo_path, args: list[str]) -> CompletedProcess:
    """minsync CLI 실행, stdout/stderr/returncode 캡처"""

def run_minsync_api(repo_path, method: str, **kwargs) -> Any:
    """Python API로 MinSync 메서드 호출"""

def get_all_doc_ids(repo_path) -> set[str]:
    """벡터 DB에서 현재 저장된 모든 doc_id 반환"""

def get_docs_by_path(repo_path, path: str) -> list[dict]:
    """특정 path의 모든 문서 반환"""

def get_all_paths_in_db(repo_path) -> set[str]:
    """벡터 DB에 저장된 모든 고유 path 반환"""

def get_cursor(repo_path) -> dict:
    """cursor.json 읽기"""

def write_minsyncignore(repo_path, content: str):
    """.minsyncignore 파일 작성"""
```

---

## T01: init — 기본 초기화

### 테스트 환경 구성
```
git init → 샘플 문서들 추가 → git commit
```

### 실행
```bash
minsync init
```

### 검증 항목

| # | 검증 | 기대값 | 검증 방법 |
|---|------|--------|----------|
| T01-1 | 종료 코드 | `0` | `assert result.returncode == 0` |
| T01-2 | `.minsync/` 디렉토리 존재 | 존재 | `assert (repo / ".minsync").is_dir()` |
| T01-3 | `.minsync/config.yaml` 존재 | 존재 | `assert (repo / ".minsync/config.yaml").is_file()` |
| T01-4 | config에 repo_id 존재 | root commit hash와 동일 | `assert config["repo_id"] == git_root_commit(repo)` |
| T01-5 | config에 ref 존재 | `"main"` | `assert config["ref"] == "main"` |
| T01-6 | config에 chunker.id 존재 | `"markdown-heading"` | 값 확인 |
| T01-7 | config에 embedder.id 존재 | 임베더 ID 문자열 | 존재 확인 |
| T01-8 | config에 vectorstore.id 존재 | `"zvec"` | 값 확인 |
| T01-9 | config에 include 없음 (git-tracked 전체가 기본, .gitignore 파일은 git이 자동 제외) | 키 부존재 | `assert "include" not in config` |
| T01-10 | cursor.json 부존재 | 존재하지 않음 | `assert not path.exists()` |
| T01-11 | stdout에 "Initialized" 포함 | 포함 | stdout 검사 |

### Python API 검증
```python
ms = MinSync(repo_path=repo)
ms.init()
assert (repo / ".minsync/config.yaml").is_file()
```

---

## T02: init — 에러 케이스

### T02-a: git 저장소가 아닌 곳에서 init

| # | 검증 | 기대값 |
|---|------|--------|
| T02-a-1 | 종료 코드 | `2` |
| T02-a-2 | stderr에 "not a git repository" 포함 | 포함 |
| T02-a-3 | `.minsync/` 미생성 | 미존재 |

### T02-b: 이미 초기화된 상태에서 init

| # | 검증 | 기대값 |
|---|------|--------|
| T02-b-1 | 종료 코드 | `1` |
| T02-b-2 | stderr에 "already initialized" 포함 | 포함 |

### T02-c: --force로 재초기화

| # | 검증 | 기대값 |
|---|------|--------|
| T02-c-1 | 종료 코드 | `0` |
| T02-c-2 | config.yaml이 새로 생성됨 | 파일 mtime이 갱신됨 |

### T02-d: 커밋이 없는 빈 저장소에서 init

| # | 검증 | 기대값 |
|---|------|--------|
| T02-d-1 | 종료 코드 | `2` |
| T02-d-2 | stderr에 "no commits" 포함 | 포함 |

---

## T03: sync — 최초 전체 인덱싱

### 테스트 환경 구성
```
git init
→ 샘플 문서 추가 (guide.md, api.md, login.md, oauth.md, faq.md, main.py, utils.py, meeting.txt, README.md)
→ git commit
→ minsync init
(모든 git-tracked 파일이 인덱싱 대상. .gitignore에 의해 untracked된 파일은 자동 제외)
```

### 실행
```bash
minsync sync
```

### 검증 항목

| # | 검증 | 기대값 | 검증 방법 |
|---|------|--------|----------|
| T03-1 | 종료 코드 | `0` | returncode |
| T03-2 | cursor.json 생성됨 | 존재 | file exists |
| T03-3 | cursor.last_synced_commit | HEAD와 동일 | cursor 검사 |
| T03-4 | txn.json 부존재 | 존재하지 않음 | file not exists |
| T03-5 | lock 파일 부존재 | 존재하지 않음 | file not exists |
| T03-6 | 벡터 DB에 문서 존재 | 청크 수 > 0 | `len(get_all_doc_ids(repo)) > 0` |
| T03-7 | 모든 git-tracked 파일의 청크 존재 | 각 파일 1개 이상 | 각 path에 대해 doc 수 검사 |
| T03-8 | 모든 doc의 repo_id 일치 | config의 repo_id와 동일 | 전체 문서 검사 |
| T03-9 | 모든 doc의 ref 일치 | `"main"` | 전체 문서 검사 |
| T03-10 | stdout에 "completed" 포함 | 포함 | stdout 검사 |

### Python API 검증
```python
ms = MinSync(repo_path=repo)
result = ms.sync()
assert result.files_processed > 0
```

---

## T04: sync — 증분 동기화 (파일 추가)

### 테스트 환경 구성
```
T03 완료 상태에서:
→ docs/tutorial.md 추가
→ git commit
```

### 검증 항목

| # | 검증 | 기대값 | 검증 방법 |
|---|------|--------|----------|
| T04-1 | 종료 코드 | `0` | returncode |
| T04-2 | cursor 갱신됨 | 새 HEAD와 동일 | cursor 검사 |
| T04-3 | tutorial.md 청크 존재 | 1개 이상 | path 검사 |
| T04-4 | 기존 파일 청크 유지 | 이전과 동일한 수 | 각 path의 doc 수 비교 |
| T04-5 | 전체 청크 수 증가 | T03 + tutorial 청크 수 | 총 doc 수 비교 |

---

## T05: sync — 증분 동기화 (파일 수정)

### 테스트 환경 구성
```
T04 완료 상태에서:
→ docs/guide.md 수정 (새 섹션 "## Troubleshooting" 추가)
→ git commit
```

### 검증 항목

| # | 검증 | 기대값 | 검증 방법 |
|---|------|--------|----------|
| T05-1 | 종료 코드 | `0` | returncode |
| T05-2 | cursor 갱신됨 | 새 HEAD | cursor 검사 |
| T05-3 | guide.md 청크에 "Troubleshooting" 포함 | 존재 | text 검사 |
| T05-4 | guide.md의 stale 청크 없음 | 현재 스냅샷과 일치 | 기대 doc_id 집합과 실제 일치 |
| T05-5 | 다른 파일 청크 불변 | 변경 없음 | 다른 path의 doc 수 동일 |

---

## T06: sync — 증분 동기화 (파일 삭제)

### 테스트 환경 구성
```
T05 완료 상태에서:
→ docs/faq.md 삭제
→ git commit
```

### 검증 항목

| # | 검증 | 기대값 | 검증 방법 |
|---|------|--------|----------|
| T06-1 | 종료 코드 | `0` | returncode |
| T06-2 | faq.md 청크 0개 | 0 | `len(get_docs_by_path("docs/faq.md")) == 0` |
| T06-3 | 다른 파일 청크 유지 | 불변 | 다른 path 확인 |
| T06-4 | 전체 청크 수 감소 | T05 - faq 청크 수 | 총 doc 수 비교 |

---

## T07: sync — 파일 이름 변경 (Rename)

### 테스트 환경 구성
```
T06 완료 상태에서:
→ git mv docs/api.md docs/reference.md
→ git commit
```

### 검증 항목

| # | 검증 | 기대값 | 검증 방법 |
|---|------|--------|----------|
| T07-1 | 종료 코드 | `0` | returncode |
| T07-2 | api.md 청크 0개 | 0 | path 검사 |
| T07-3 | reference.md 청크 존재 | > 0 | path 검사 |
| T07-4 | reference.md 청크 내용 | api.md와 동일 텍스트 | 텍스트 비교 |

---

## T08: sync — 복합 변경 (추가+수정+삭제 동시)

### 테스트 환경 구성
```
초기 상태에서 단일 커밋으로:
→ docs/new-feature.md 추가
→ docs/guide.md 수정
→ docs/oauth.md 삭제
```

### 검증 항목

| # | 검증 | 기대값 | 검증 방법 |
|---|------|--------|----------|
| T08-1 | 종료 코드 | `0` | returncode |
| T08-2 | new-feature.md 청크 존재 | > 0 | path 검사 |
| T08-3 | guide.md 최신 반영 | 수정된 내용의 청크만 존재 | doc_id 일치 |
| T08-4 | oauth.md 청크 0개 | 0 | path 검사 |
| T08-5 | login.md, faq.md 불변 | 동일 | doc 수 비교 |

---

## T09: sync — 여러 커밋에 걸친 증분

### 테스트 환경 구성
```
초기 sync 완료 후, 3개의 커밋을 연속으로 생성:
  commit A: docs/a.md 추가
  commit B: docs/b.md 추가, docs/guide.md 수정
  commit C: docs/a.md 삭제
(최종 상태: b.md 추가됨, guide.md 수정됨, a.md는 추가 후 삭제되어 없음)
```

### 검증 항목

| # | 검증 | 기대값 | 검증 방법 |
|---|------|--------|----------|
| T09-1 | 종료 코드 | `0` | returncode |
| T09-2 | cursor가 commit C를 가리킴 | commit C hash | cursor 검사 |
| T09-3 | a.md 청크 0개 | 0 | path 검사 |
| T09-4 | b.md 청크 존재 | > 0 | path 검사 |
| T09-5 | guide.md 최신 반영 | commit B의 수정본 | 내용 검사 |

---

## T10: sync --dry-run

### 테스트 환경 구성
```
초기 sync 완료 후:
→ docs/new.md 추가
→ docs/guide.md 수정
→ git commit
```

### 검증 항목

| # | 검증 | 기대값 | 검증 방법 |
|---|------|--------|----------|
| T10-1 | 종료 코드 | `0` | returncode |
| T10-2 | cursor.json 불변 | dry-run 전과 동일 | cursor 비교 |
| T10-3 | txn.json 미생성 | 부존재 | file not exists |
| T10-4 | 벡터 DB 불변 | doc 수 동일 | doc 수 비교 |
| T10-5 | stdout에 "DRY RUN" 포함 | 포함 | stdout 검사 |
| T10-6 | stdout에 파일 목록 포함 | new.md, guide.md 표시 | stdout 검사 |
| T10-7 | 이후 실제 sync가 정상 동작 | `minsync sync` 성공 | 이후 sync 실행 |

### Python API 검증
```python
result = ms.sync(dry_run=True)
assert result.dry_run is True
assert result.files_planned > 0
# DB는 변경되지 않아야 함
```

---

## T11: sync --full (전체 재구축)

### 테스트 환경 구성
```
T03 완료 상태
```

### 검증 항목

| # | 검증 | 기대값 | 검증 방법 |
|---|------|--------|----------|
| T11-1 | 종료 코드 | `0` | returncode |
| T11-2 | cursor 갱신됨 | HEAD | cursor 검사 |
| T11-3 | 전체 doc_id 집합 동일 | T03과 동일한 doc_id 집합 | 집합 비교 |
| T11-4 | 모든 doc의 seen_token 통일 | 전부 같은 sync_token | 문서 검사 |

---

## T12: 결정적 ID (Deterministic ID) — 다른 위치

### 테스트 환경 구성
```
동일 저장소를 두 임시 경로에 clone:
  /tmp/repo-a/  →  minsync init && minsync sync
  /tmp/repo-b/  →  minsync init && minsync sync
```

### 검증 항목

| # | 검증 | 기대값 | 검증 방법 |
|---|------|--------|----------|
| T12-1 | 두 위치의 repo_id 동일 | 동일 | 비교 |
| T12-2 | 두 위치의 doc_id 집합 동일 | 완전 일치 | `set_a == set_b` |
| T12-3 | 각 doc_id의 content_hash 동일 | 모두 일치 | 개별 비교 |

---

## T13: 결정적 ID — rebuild 후 동일

### 테스트 환경 구성
```
  minsync sync  → doc_id 집합 A
  minsync sync --full  → doc_id 집합 B
```

### 검증 항목

| # | 검증 | 기대값 | 검증 방법 |
|---|------|--------|----------|
| T13-1 | 집합 A == 집합 B | 완전 일치 | `set_a == set_b` |

---

## T14: 크래시 복구 — sync 도중 중단

### 테스트 환경 구성
```
초기 sync 완료 후:
→ 5개 파일 추가 → git commit
sync 중간에 강제 중단 (mock으로 3번째 파일 처리 후 예외 발생)
```

### 검증 항목

| # | 검증 | 기대값 | 검증 방법 |
|---|------|--------|----------|
| T14-1 | 크래시 후 cursor.json | 이전 상태 유지 | cursor 비교 |
| T14-2 | 크래시 후 txn.json | 존재 | file exists |
| T14-3 | 복구 sync 종료 코드 | `0` | returncode |
| T14-4 | 복구 후 cursor 갱신 | 최신 HEAD | cursor 검사 |
| T14-5 | 복구 후 txn.json 삭제 | 부존재 | file not exists |
| T14-6 | 복구 후 벡터 DB 정합 | 크래시 없이 sync한 것과 동일 doc_id 집합 | 집합 비교 |
| T14-7 | verify 통과 | exit 0, ALL PASSED | verify 결과 |

---

## T15: 크래시 복구 — flush 직전 중단

### 테스트 환경 구성
```
모든 파일 처리 완료 후, collection.flush() 호출 전에 중단
```

### 검증 항목

| # | 검증 | 기대값 | 검증 방법 |
|---|------|--------|----------|
| T15-1 | 복구 sync 후 모든 청크 존재 | 완전 일치 | doc_id 집합 비교 |
| T15-2 | stale 청크 없음 | 0 | sweep 후 검사 |

---

## T16: 크래시 복구 — cursor 갱신 직전 중단

### 테스트 환경 구성
```
flush() 성공 후, cursor.json 갱신 전에 중단
```

### 검증 항목

| # | 검증 | 기대값 | 검증 방법 |
|---|------|--------|----------|
| T16-1 | cursor는 이전 상태 | 갱신 안 됨 | cursor 검사 |
| T16-2 | 복구 sync 실행 | 성공 | returncode 0 |
| T16-3 | 최종 결과 동일 | 정상 sync와 동일 | doc_id 집합 비교 |

---

## T17: Lock — 동시 실행 방지

### 테스트 환경 구성
```
minsync sync를 2개의 프로세스/스레드에서 동시 실행
```

### 검증 항목

| # | 검증 | 기대값 | 검증 방법 |
|---|------|--------|----------|
| T17-1 | 한쪽은 성공 | exit 0 | returncode |
| T17-2 | 다른 쪽은 lock 에러 | exit 3 | returncode |
| T17-3 | lock 에러 메시지 | "another sync is in progress" | stderr 검사 |
| T17-4 | 완료 후 lock 해제 | lock 파일 부존재 | file not exists |

---

## T18: sync — 이미 최신 상태

### 검증 항목

| # | 검증 | 기대값 | 검증 방법 |
|---|------|--------|----------|
| T18-1 | 종료 코드 | `0` | returncode |
| T18-2 | stdout에 "Already up to date" | 포함 | stdout 검사 |
| T18-3 | cursor 불변 | 동일 | cursor 비교 |
| T18-4 | 벡터 DB 불변 | doc 수 동일 | doc 수 비교 |

---

## T19: sync — 초기화 안 된 상태

### 검증 항목

| # | 검증 | 기대값 | 검증 방법 |
|---|------|--------|----------|
| T19-1 | 종료 코드 | `1` | returncode |
| T19-2 | stderr에 "not initialized" | 포함 | stderr 검사 |

---

## T20: query — 기본 검색

### 테스트 환경 구성
```
전체 인덱싱 완료 상태
```

### 검증 항목

| # | 검증 | 기대값 | 검증 방법 |
|---|------|--------|----------|
| T20-1 | 종료 코드 | `0` | returncode |
| T20-2 | 결과 1개 이상 | true | 결과 파싱 |
| T20-3 | 상위 결과에 login.md 포함 | true | path 확인 |
| T20-4 | 각 결과에 path, text 포함 | true | 구조 검사 |

### Python API 검증
```python
results = ms.query("login process authentication", k=5)
assert len(results) > 0
assert any("login" in r.path for r in results)
```

---

## T21: query — JSON 출력

### 검증 항목

| # | 검증 | 기대값 | 검증 방법 |
|---|------|--------|----------|
| T21-1 | stdout가 valid JSON | true | `json.loads(result.stdout)` |
| T21-2 | results 배열 길이 ≤ 3 | true | `len(data["results"]) <= 3` |
| T21-3 | 각 결과에 doc_id, path, text, score 존재 | true | 키 검사 |

---

## T22: query — 빈 인덱스

### 검증 항목

| # | 검증 | 기대값 | 검증 방법 |
|---|------|--------|----------|
| T22-1 | 종료 코드 | `0` | returncode |
| T22-2 | 결과 0건 | true | 결과 파싱 |
| T22-3 | 경고 메시지 | "index is empty" 포함 | stdout/stderr |

---

## T23: query — 빈 쿼리 문자열

### 검증 항목

| # | 검증 | 기대값 | 검증 방법 |
|---|------|--------|----------|
| T23-1 | 종료 코드 | `1` | returncode |
| T23-2 | 에러 메시지 | "query text is required" | stderr |

---

## T24: status — 각 상태별 출력

### T24-a: NOT_SYNCED 상태 (초기화만)

| # | 검증 | 기대값 |
|---|------|--------|
| T24-a-1 | stdout에 "NOT SYNCED" | 포함 |
| T24-a-2 | stdout에 "(never)" | 포함 |

### T24-b: UP_TO_DATE 상태 (sync 완료 직후)

| # | 검증 | 기대값 |
|---|------|--------|
| T24-b-1 | stdout에 "UP TO DATE" | 포함 |
| T24-b-2 | last_synced와 current_head 동일 | 동일 커밋 hash |

### T24-c: OUT_OF_DATE 상태 (새 커밋 발생)

| # | 검증 | 기대값 |
|---|------|--------|
| T24-c-1 | stdout에 "OUT OF DATE" | 포함 |
| T24-c-2 | "commits behind" 표시 | 숫자 포함 |

### T24-d: JSON 출력

| # | 검증 | 기대값 |
|---|------|--------|
| T24-d-1 | valid JSON | `json.loads()` 성공 |
| T24-d-2 | "state" 키 존재 | 문자열 값 |

### Python API 검증
```python
status = ms.status()
assert status.state in ("UP_TO_DATE", "OUT_OF_DATE", "NOT_SYNCED", "INTERRUPTED")
```

---

## T25: verify — 정상 상태 검증

### 검증 항목

| # | 검증 | 기대값 | 검증 방법 |
|---|------|--------|----------|
| T25-1 | 종료 코드 | `0` | returncode |
| T25-2 | "ALL CHECKS PASSED" 출력 | 포함 | stdout |
| T25-3 | 모든 파일 OK | 각 파일 "OK" | stdout 파싱 |

---

## T26: verify — 불일치 감지

### 테스트 환경 구성
```
sync 완료 후, 벡터 DB에서 일부 문서를 직접 삭제 (의도적 손상)
```

### 검증 항목

| # | 검증 | 기대값 | 검증 방법 |
|---|------|--------|----------|
| T26-1 | 종료 코드 | `1` | returncode |
| T26-2 | "FAIL" 출력 | 포함 | stdout |
| T26-3 | "MISSING" 표시 | 삭제된 문서에 해당 | stdout 파싱 |

---

## T27: verify --fix

### 테스트 환경 구성
```
T26과 동일 (의도적 손상 상태)
```

### 검증 항목

| # | 검증 | 기대값 | 검증 방법 |
|---|------|--------|----------|
| T27-1 | fix 후 종료 코드 | `0` | returncode |
| T27-2 | 이후 verify 통과 | "ALL CHECKS PASSED" | stdout |

---

## T28: mark+sweep 수렴 보장

### 테스트 환경 구성
```
1. 파일 A를 추가 → sync
2. 파일 A를 크게 수정 (청크 구조 변경) → sync
3. 파일 A를 다시 수정 → sync
```

### 검증 항목

| # | 검증 | 기대값 | 검증 방법 |
|---|------|--------|----------|
| T28-1 | 매 sync 후 파일 A의 청크 | 현재 스냅샷의 기대 청크만 존재 | doc_id 집합 비교 |
| T28-2 | stale 청크 없음 | 이전 버전의 청크가 남아있지 않음 | doc_id 비교 |

---

## T29: 대규모 증분 성능

### 테스트 환경 구성
```
100개 파일 생성 → sync → 3개 파일만 수정 → 커밋
```

### 검증 항목

| # | 검증 | 기대값 | 검증 방법 |
|---|------|--------|----------|
| T29-1 | 처리 파일 수 | 3개만 표시 | stdout 파싱 |
| T29-2 | 임베딩 호출 횟수 | 3개 파일의 새 청크에 대해서만 | mock 카운터 |
| T29-3 | 나머지 97개 불변 | doc_id 동일 | 비교 |

---

## T30: .minsyncignore 기본 필터링

### 테스트 환경 구성
```
저장소에 다양한 파일 유형이 git-tracked:
  docs/guide.md, docs/api.md, src/main.py, src/utils.py, notes/meeting.txt

.minsyncignore:
  src/**/*.py
  notes/
```

### 검증 항목

| # | 검증 | 기대값 | 검증 방법 |
|---|------|--------|----------|
| T30-1 | docs/guide.md 인덱싱됨 | 청크 존재 | path 검사 |
| T30-2 | docs/api.md 인덱싱됨 | 청크 존재 | path 검사 |
| T30-3 | src/main.py 미인덱싱 | 청크 0개 | path 검사 |
| T30-4 | src/utils.py 미인덱싱 | 청크 0개 | path 검사 |
| T30-5 | notes/meeting.txt 미인덱싱 | 청크 0개 | path 검사 |
| T30-6 | README.md 인덱싱됨 | 청크 존재 | path 검사 |

---

## T31: schema/embedder 불일치 감지

### 테스트 환경 구성
```
sync 완료 후, config.yaml에서 chunker.id를 변경
```

### 검증 항목

| # | 검증 | 기대값 | 검증 방법 |
|---|------|--------|----------|
| T31-1 | 종료 코드 | `1` | returncode |
| T31-2 | 에러 메시지 | "schema/embedder mismatch" | stderr |
| T31-3 | cursor 불변 | 이전 상태 | cursor 비교 |
| T31-4 | `--full`로 재실행 시 성공 | exit 0 | returncode |

---

## T32: .minsyncignore 변경 후 증분 sync (full 불필요)

### 테스트 환경 구성
```
1. 초기 sync 완료 (src/main.py, src/utils.py 포함 인덱싱)
2. .minsyncignore에 "src/**/*.py" 추가 → git commit
3. minsync sync (--full 없이)
4. minsync verify
```

### 검증 항목

| # | 검증 | 기대값 | 검증 방법 |
|---|------|--------|----------|
| T32-1 | sync 종료 코드 | `0` | returncode |
| T32-2 | sync에서 src/*.py를 처리 대상에서 제외 | 처리 파일에 미포함 | stdout 검사 |
| T32-3 | verify에서 src/*.py 잔존 감지 | "IGNORED_STALE" 보고 | stdout 검사 |
| T32-4 | verify --fix 후 src/*.py 청크 0개 | 0 | path 검사 |
| T32-5 | --full 없이 정상 동작 | 전체 과정에서 --full 미사용 | 실행 로그 확인 |

### 반대 방향: ignore 해제

```
5. .minsyncignore에서 "src/**/*.py" 제거 → git commit
6. minsync sync
```

| # | 검증 | 기대값 | 검증 방법 |
|---|------|--------|----------|
| T32-6 | sync에서 src/*.py가 신규 파일로 처리 | 청크 생성 | path 검사 |
| T32-7 | src/main.py 청크 존재 | > 0 | path 검사 |

---

## T33: check — 정상 health check

### 테스트 환경 구성
```
init 완료, 임베딩/벡터 DB가 정상 연결된 상태
```

### 검증 항목

| # | 검증 | 기대값 | 검증 방법 |
|---|------|--------|----------|
| T33-1 | 종료 코드 | `0` | returncode |
| T33-2 | stdout에 "All checks passed" | 포함 | stdout 검사 |
| T33-3 | Git 체크 OK | "OK" | stdout 파싱 |
| T33-4 | Embedder 체크 OK | "OK" | stdout 파싱 |
| T33-5 | VectorStore 체크 OK | "OK" | stdout 파싱 |

### Python API 검증
```python
health = ms.check()
assert health.embedder_ok is True
assert health.vectorstore_ok is True
assert health.git_ok is True
```

---

## T34: check — 임베딩 실패 감지

### 테스트 환경 구성
```
API 키가 설정되지 않은 상태 또는 잘못된 키
```

### 검증 항목

| # | 검증 | 기대값 | 검증 방법 |
|---|------|--------|----------|
| T34-1 | 종료 코드 | `1` | returncode |
| T34-2 | Embedder 체크 FAIL | "FAIL" | stdout 파싱 |
| T34-3 | 에러 메시지에 원인 포함 | API key 관련 메시지 | stdout 검사 |

---

## T35: check — 벡터 DB 연결 실패 감지

### 테스트 환경 구성
```
config.yaml에 접근 불가능한 벡터 DB URL 설정
```

### 검증 항목

| # | 검증 | 기대값 | 검증 방법 |
|---|------|--------|----------|
| T35-1 | 종료 코드 | `1` | returncode |
| T35-2 | VectorStore 체크 FAIL | "FAIL" | stdout 파싱 |
| T35-3 | 에러 메시지에 원인 포함 | 연결 거부 메시지 | stdout 검사 |

---

## T36: check — 패키지 미설치 감지

### 테스트 환경 구성
```
config.yaml에 weaviate를 vectorstore로 설정, weaviate-client 미설치
```

### 검증 항목

| # | 검증 | 기대값 | 검증 방법 |
|---|------|--------|----------|
| T36-1 | 종료 코드 | `1` | returncode |
| T36-2 | 에러 메시지에 설치 안내 포함 | "pip install" 포함 | stdout/stderr |

---

## T37: verify — .minsyncignore 잔존 감지

### 테스트 환경 구성
```
1. src/main.py, src/utils.py 포함하여 전체 sync 완료
2. .minsyncignore에 "src/**" 추가
3. minsync verify (sync 없이 바로 verify)
```

### 검증 항목

| # | 검증 | 기대값 | 검증 방법 |
|---|------|--------|----------|
| T37-1 | 종료 코드 | `1` | returncode |
| T37-2 | "IGNORED_STALE" 보고 | src/main.py, src/utils.py 언급 | stdout 파싱 |
| T37-3 | stale 청크 수 표시 | > 0 | stdout 파싱 |

---

## T38: verify --fix — .minsyncignore 잔존 삭제

### 테스트 환경 구성
```
T37과 동일 상태에서 --fix 적용
```

### 검증 항목

| # | 검증 | 기대값 | 검증 방법 |
|---|------|--------|----------|
| T38-1 | 종료 코드 | `0` | returncode |
| T38-2 | src/main.py 청크 0개 | 0 | path 검사 |
| T38-3 | src/utils.py 청크 0개 | 0 | path 검사 |
| T38-4 | 다른 파일 청크 불변 | 동일 | doc 수 비교 |
| T38-5 | 이후 verify 통과 | "ALL CHECKS PASSED" | stdout |

---

## T39: Python API — 커스텀 컴포넌트

### 테스트 환경 구성
```python
class MockChunker(Chunker):
    def schema_id(self): return "mock-chunker-v1"
    def chunk(self, text, path): return [Chunk(chunk_type="parent", text=text)]

class MockEmbedder(Embedder):
    def id(self): return "mock-embedder-v1"
    def embed(self, texts): return [[0.1] * 128 for _ in texts]

ms = MinSync(repo_path=repo, chunker=MockChunker(), embedder=MockEmbedder())
```

### 검증 항목

| # | 검증 | 기대값 | 검증 방법 |
|---|------|--------|----------|
| T39-1 | init 성공 | 예외 없음 | 실행 확인 |
| T39-2 | sync 성공 | SyncResult 반환 | result 검사 |
| T39-3 | cursor에 커스텀 schema_id 기록 | "mock-chunker-v1" | cursor 검사 |
| T39-4 | cursor에 커스텀 embedder_id 기록 | "mock-embedder-v1" | cursor 검사 |
| T39-5 | query 성공 | 결과 반환 | result 검사 |
| T39-6 | verify 통과 | all_passed = True | report 검사 |

---

## T40: Python API — CLI와 동일 결과

### 테스트 환경 구성
```
동일 저장소에서 CLI와 Python API 각각 실행
```

### 검증 항목

| # | 검증 | 기대값 | 검증 방법 |
|---|------|--------|----------|
| T40-1 | CLI sync 후 doc_id 집합 | Python API sync 후 doc_id 집합과 동일 | 집합 비교 |
| T40-2 | CLI status 결과 | Python API status와 동일 | 필드 비교 |
| T40-3 | CLI verify 결과 | Python API verify와 동일 | 결과 비교 |

---

## T41: CI/CD 시뮬레이션 — 정상 흐름

### 테스트 환경 구성
```
GitHub Actions와 유사한 순서로 실행:
1. git clone (fetch-depth 0)
2. minsync init (또는 .minsync가 이미 있으면 skip)
3. minsync check
4. minsync sync --verbose
5. minsync verify --sample 5
6. cursor.json commit
```

### 검증 항목

| # | 검증 | 기대값 | 검증 방법 |
|---|------|--------|----------|
| T41-1 | 모든 단계 exit 0 | 성공 | returncode |
| T41-2 | cursor.json 갱신됨 | 최신 HEAD | cursor 검사 |
| T41-3 | verify 통과 | PASS | stdout 검사 |

---

## T42: CI/CD 시뮬레이션 — 실패 후 복구

### 테스트 환경 구성
```
1. 첫 번째 CI 실행: sync 중간에 강제 중단 (크래시 시뮬)
2. 두 번째 CI 실행: 복구 흐름
   a. minsync status (INTERRUPTED 상태 확인)
   b. minsync check
   c. minsync sync --verbose (자동 복구)
   d. minsync verify --all --fix
   e. minsync status (UP_TO_DATE 확인)
```

### 검증 항목

| # | 검증 | 기대값 | 검증 방법 |
|---|------|--------|----------|
| T42-1 | 첫 실행 후 status | INTERRUPTED | status 출력 검사 |
| T42-2 | 두 번째 실행 — check 통과 | exit 0 | returncode |
| T42-3 | 두 번째 실행 — sync 복구 성공 | exit 0 | returncode |
| T42-4 | 두 번째 실행 — sync 출력에 "Recovering" | 포함 | stdout 검사 |
| T42-5 | 두 번째 실행 — verify 통과 | ALL PASSED | stdout 검사 |
| T42-6 | 두 번째 실행 — 최종 status | UP_TO_DATE | status 출력 검사 |
| T42-7 | 최종 doc_id 집합 | 정상 sync와 동일 | 집합 비교 |

---

## T43: .gitignore에 의한 자동 제외

### 테스트 환경 구성
```
git init
→ .gitignore 작성:
    build/
    *.log
    __pycache__/
→ git-tracked 파일: docs/guide.md, src/main.py, README.md
→ git-untracked 파일 (디스크에만 존재): build/output.bin, app.log, __pycache__/cache.pyc
→ git commit (tracked 파일만 커밋됨)
→ minsync init && minsync sync
```

### 검증 항목

| # | 검증 | 기대값 | 검증 방법 |
|---|------|--------|----------|
| T43-1 | docs/guide.md 인덱싱됨 | 청크 존재 | path 검사 |
| T43-2 | src/main.py 인덱싱됨 | 청크 존재 | path 검사 |
| T43-3 | README.md 인덱싱됨 | 청크 존재 | path 검사 |
| T43-4 | build/output.bin 미인덱싱 | DB에 path 부존재 | `"build/output.bin" not in get_all_paths_in_db()` |
| T43-5 | app.log 미인덱싱 | DB에 path 부존재 | path 검사 |
| T43-6 | __pycache__/cache.pyc 미인덱싱 | DB에 path 부존재 | path 검사 |
| T43-7 | .gitignore된 파일은 .minsyncignore에 기재하지 않아도 자동 제외 | .minsyncignore 없이도 위 결과 동일 | .minsyncignore 파일 부존재 확인 |

### 핵심 포인트
이 테스트는 `.minsyncignore` 없이도 `.gitignore`에 의해 untracked된 파일이 minsync에 보이지 않음을 검증한다. minsync가 `git ls-tree`로 파일 목록을 가져오므로, git이 무시하는 파일은 자연스럽게 인덱싱 대상에서 제외된다.

---

## T44: .gitignore + .minsyncignore 조합

### 테스트 환경 구성
```
.gitignore:
    build/
    *.log

.minsyncignore:
    src/**/*.py
    *.txt

git-tracked 파일: docs/guide.md, src/main.py, notes/meeting.txt, README.md
git-untracked: build/output.bin, app.log
```

### 검증 항목

| # | 검증 | 기대값 | 검증 방법 |
|---|------|--------|----------|
| T44-1 | docs/guide.md 인덱싱됨 | 청크 존재 | 두 ignore 모두에 해당하지 않음 |
| T44-2 | README.md 인덱싱됨 | 청크 존재 | .md 파일, ignore 미해당 |
| T44-3 | src/main.py 미인덱싱 | 0개 | .minsyncignore에 의해 제외 |
| T44-4 | notes/meeting.txt 미인덱싱 | 0개 | .minsyncignore `*.txt`에 의해 제외 |
| T44-5 | build/output.bin 미인덱싱 | DB에 부존재 | .gitignore에 의해 untracked → 자동 제외 |
| T44-6 | app.log 미인덱싱 | DB에 부존재 | .gitignore에 의해 untracked → 자동 제외 |

### 핵심 포인트
최종 인덱싱 대상 = (git-tracked 파일) − (.minsyncignore 매칭) − (하드코딩 제외)
`.gitignore`는 git 레벨에서 처리, `.minsyncignore`는 minsync 레벨에서 처리. 두 계층이 독립적으로 작동한다.

---

## 테스트 매트릭스 요약

| 테스트 | 커맨드 | 카테고리 | 핵심 검증 |
|--------|--------|----------|-----------|
| T01 | init | 정상 | 디렉토리/파일 생성, include 없음 |
| T02 | init | 에러 | 비 git, 중복, 빈 repo |
| T03 | sync | 최초 동기화 | 전체 인덱싱 (git-tracked 전체) |
| T04 | sync | 증분 - 추가 | 새 파일 청크 추가 |
| T05 | sync | 증분 - 수정 | 변경 청크 갱신 |
| T06 | sync | 증분 - 삭제 | 삭제 파일 청크 제거 |
| T07 | sync | 증분 - rename | 구 path 삭제, 신 path 생성 |
| T08 | sync | 증분 - 복합 | 추가+수정+삭제 동시 |
| T09 | sync | 증분 - 다중 커밋 | 여러 커밋 한 번에 |
| T10 | sync | dry-run | DB 불변 |
| T11 | sync | full rebuild | 결과 동일성 |
| T12 | sync | 결정성 | 다른 위치 동일 ID |
| T13 | sync | 결정성 | rebuild 동일 ID |
| T14 | sync | 크래시 복구 | 중간 중단 |
| T15 | sync | 크래시 복구 | flush 전 |
| T16 | sync | 크래시 복구 | cursor 전 |
| T17 | sync | Lock | 동시 실행 방지 |
| T18 | sync | 이미 최신 | no-op |
| T19 | sync | 에러 | 미초기화 |
| T20 | query | 정상 검색 | 관련 결과 반환 |
| T21 | query | JSON 출력 | valid JSON |
| T22 | query | 빈 인덱스 | 경고 + 0건 |
| T23 | query | 에러 | 빈 쿼리 |
| T24 | status | 상태별 출력 | 4가지 상태 |
| T25 | verify | 정상 | ALL PASSED |
| T26 | verify | 불일치 | FAIL 감지 |
| T27 | verify --fix | 자동 수정 | 수정 후 PASS |
| T28 | sync | 수렴 보장 | stale 없음 |
| T29 | sync | 성능 | 변경분만 처리 |
| T30 | sync | .minsyncignore | 기본 필터링 |
| T31 | sync | 설정 변경 | 불일치 감지 |
| T32 | sync+verify | .minsyncignore 변경 | full 불필요 |
| T33 | check | 정상 | All passed |
| T34 | check | 임베딩 실패 | FAIL 감지 |
| T35 | check | 벡터 DB 실패 | FAIL 감지 |
| T36 | check | 패키지 미설치 | 설치 안내 |
| T37 | verify | .minsyncignore 잔존 | IGNORED_STALE |
| T38 | verify --fix | .minsyncignore 잔존 삭제 | 청크 제거 |
| T39 | Python API | 커스텀 컴포넌트 | 커스텀 chunker/embedder |
| T40 | Python API | CLI 동일성 | 동일 결과 |
| T41 | CI/CD | 정상 흐름 | 전체 파이프라인 |
| T42 | CI/CD | 실패 복구 | status→check→sync→verify |
| T43 | sync | .gitignore 자동 제외 | untracked 파일 미인덱싱 |
| T44 | sync | .gitignore+.minsyncignore 조합 | 2계층 필터링 |
