PRD: (Project Name Placeholder) — Git-native Vector Index Sync Engine
1. 목적

Git 레포지토리(문서/지식베이스)의 변경사항을 git diff로 감지하고, 벡터 DB(Zvec)를 증분 갱신하여 “검색 가능한 인덱스”를 항상 최신 상태로 유지한다.

2. 핵심 가치

Deterministic chunk_id: clone/rebuild해도 동일 입력이면 동일 ID

Crash-safe / Resume-safe: 중간 실패해도 재실행하면 반드시 수렴

Atomic cursor commit: 모든 반영이 끝난 뒤에만 cursor 갱신

Backend / Embedder / Chunker 플러그형: 초기엔 Zvec + LangChain wrapper 제공

3. 범위
포함

Git 기반 변경 감지(git diff --name-status)

파일 내용 로드(기본: 텍스트/마크다운)

청킹(기본 청커 1개 + 커스텀 인터페이스)

임베딩(기본: LangChain Embeddings 어댑터)

Zvec 컬렉션에 upsert/update/delete_by_filter로 동기화

로컬 상태: .ragit/ 아래 작은 파일 몇 개(cursor/txn/lock/config)

제외(Non-goals)

권한/팀 협업/감사로그 UI

Git hosting(GitHub) 연동 UI

멀티모달/PDF 고급 처리(추후 확장 포인트로만 설계)

4. 사용자 스토리

개발자/에이전트가 레포를 clone하고 tool init && tool sync를 실행하면, Zvec에 인덱스가 생성된다.

문서가 바뀌고 커밋/머지되면 tool sync만으로 변경분이 반영된다.

sync 도중 크래시가 나도, 재실행하면 인덱스가 올바른 상태로 복구된다.

사용자는 chunk 규칙/임베딩 모델/Zvec 경로를 config로 바꿀 수 있다(단 schema 변경 시 rebuild 요구).

5. 데이터 모델
5.1 Repo ID

repo_id: root commit hash(예: git rev-list --max-parents=0 HEAD | tail -n 1)

clone 위치가 달라도 동일 레포면 동일하게 유지되는 “결정적 ID”

5.2 Cursor (옵션B)

.ragit/cursor.json

{
  "repo_id": "...",
  "ref": "main",
  "last_synced_commit": "...",
  "chunk_schema_id": "...",
  "embedder_id": "...",
  "collection_path": "...",
  "updated_at": "ISO8601"
}
5.3 Inflight transaction

.ragit/txn.json

{
  "repo_id": "...",
  "ref": "main",
  "from_commit": "...",
  "to_commit": "...",
  "sync_token": "...",
  "chunk_schema_id": "...",
  "embedder_id": "...",
  "status": "running|failed",
  "started_at": "...",
  "last_progress_at": "..."
}
5.4 Vector DB Document Schema (Zvec)

필수 필드(권장):

id (문자열, doc_id)

embedding (vector field)

text (원문 chunk)

repo_id (string)

ref (string) — 초기엔 "main" 고정

path (string) — full path

ext (string)

chunk_schema_id (string)

chunk_type (string) — "parent" / "child" 등

heading_path (string, optional)

content_hash (string) — sha256(normalized_text)

content_commit (string) — 이 chunk_id가 처음 생성(변경)된 커밋 (best-effort)

seen_token (string) — 이번 sync에서 “살아있음” 마킹용

인덱스(권장):

scalar 필드(repo_id/ref/path/chunk_schema_id/seen_token)에 inverted index

Zvec는 scalar에 inverted index를 만들 수 있다고 명시.

6. 결정적 ID 설계
6.1 Normalization

줄바꿈 통일(\r\n → \n)

트레일링 공백 제거

(옵션) 연속 공백 축약

(옵션) Markdown frontmatter 제거/보존을 config로 선택

6.2 content_hash

content_hash = sha256(normalized_chunk_text)

6.3 chunk_id (doc_id)

중복/재현성을 위해 아래를 결합해 hash:

repo_id

ref

path

chunk_schema_id

chunk_type

heading_path (있으면)

content_hash

dup_index (동일 content_hash가 같은 heading_path 내 여러 번 등장할 때 0,1,2…)

doc_id = sha256(join_with_null_bytes([...]))

포인트: dup_index는 “동일 텍스트 반복”에서만 영향. 일반 문서에선 거의 0.

7. Sync 알고리즘 (정합성 보장 버전)
7.1 락

.ragit/lock 파일 락(동시 sync 방지)

락 획득 실패 시 즉시 종료 또는 --wait 옵션

7.2 목표 커밋 결정

to_commit = git rev-parse <ref> (기본 main)

from_commit = cursor.last_synced_commit (없으면 full index 모드)

7.3 변경 파일 목록

git diff --name-status from_commit..to_commit -- <include_paths>

상태: A/M/D/R(가능하면)

7.4 파일 단위 처리 규칙 (핵심)
공통

sync_token = uuid4() (실행당 1개)

txn.json 기록 후 시작

Deleted (D)

delete_by_filter("repo_id == ... AND ref == ... AND path == ...")

path 기반 벌크 삭제는 Zvec에서 delete_by_filter로 가능

Added/Modified/Renamed (A/M/R)

파일 내용 로드

결정성 위해 working tree가 아니라 git show to_commit:path로 읽기(기본)

청킹

chunker에서 parent/child 생성

doc_id 목록 생성

존재 여부 확인

fetch(ids)로 기존 문서 존재 확인

Zvec fetch는 ID로 문서를 가져오고, 없는 ID는 결과에서 누락됨

업데이트/삽입

existing ids: update([{id, seen_token=sync_token, path=..., heading_path=... (필요 시)}])

Zvec update는 지정한 필드만 업데이트하고 나머지는 유지

missing ids: 임베딩 계산 후 upsert([{id, embedding, text, ..., content_commit=to_commit, seen_token=sync_token}])

Sweep (stale chunk 제거, 파일 단위 수렴 보장)

delete_by_filter("repo_id==... AND ref==... AND path==... AND seen_token != '<sync_token>'")

이렇게 하면 “이 파일에 대해 이번 실행에서 마킹되지 않은 모든 청크”가 제거되어,
DB가 이 파일의 최신 스냅샷과 정확히 일치하게 됨.

이 mark+sweep 덕분에, 중간에 일부 delete 로직이 누락되거나 rename 감지가 흔들려도
결국 “현재 파일 스냅샷”만 남게 수렴한다.

7.5 커밋(Commit) 단계

모든 파일 처리 성공 후:

collection.flush() 호출(내구성)

cursor.json을 원자적 갱신

temp 파일에 쓰기 → fsync → rename(atomic)

txn.json 삭제

실패 시:

cursor.json 갱신 금지

txn.json은 남겨서 디버깅 가능(또는 status=failed)

8. Chunker/Embedder/VectorStore 인터페이스
8.1 Chunker (플러그형)
class Chunker:
    def schema_id(self) -> str: ...
    def chunk(self, text: str, path: str) -> list[Chunk]:
        # Chunk: {chunk_type, text, heading_path?, ...}

기본 chunker:

Markdown: heading 기반 section(parent) + paragraph(child)

Text: sliding window(설정 기반)

8.2 Embedder

코어 인터페이스는 단순하게 유지:

class Embedder:
    def id(self) -> str: ...
    def embed(self, texts: list[str]) -> list[list[float]]: ...

LangChain wrapper 제공(선택)

8.3 VectorStore (Zvec Adapter v1)

필수:

upsert(docs)

update(docs) ← metadata-only 갱신에 사용

fetch(ids) ← 존재 확인/재시작 최적화

delete_by_filter(filter)

query(vector, filter, topk)

9. CLI 스펙

tool init

.ragit/config.yaml, .ragit/cursor.json(없음), .ragit/ 생성

tool sync [--ref main] [--full] [--dry-run] [--verbose]

기본: cursor 기반 증분

--full: cursor 무시하고 전체 재구축(기존 repo_id/ref 범위 delete 후 재생성)

tool query "<text>" [--ref main] [--k 10] [--filter "..."]

임베딩 후 Zvec query 실행

tool status

cursor commit, to_commit, dirty 여부, 마지막 sync 시간

tool verify [--ref main] [--sample N|--all]

(v1) 최소: cursor와 to_commit 비교, txn 존재 여부, 컬렉션 통계/기본 체크

(v2) 확장: 파일 샘플을 골라 현재 스냅샷 doc_id들을 fetch로 확인

10. 품질/테스트 요구사항 (중요)
Crash/Resume 테스트 (필수)

sync 중간에 강제 종료(SIGKILL) 후 재실행 → 최종 상태가 동일해야 함

“파일 처리 도중” / “flush 직전” / “cursor 갱신 직전” 등 구간별로 반복

정합성 테스트 (필수)

문서 추가/수정/삭제/rename(감지 실패 포함) 시에도 최종적으로

삭제된 path의 doc이 0개

살아있는 path는 “최신 스냅샷 chunk”만 남음(mark+sweep로 보장)

결정성 테스트 (필수)

같은 커밋에서 두 번 rebuild → 생성되는 doc_id 집합이 동일해야 함

11. 성능 요구사항 (v1 기준)

대규모 repo에서도 “변경 파일만” 처리

배치 임베딩(batch_size 설정)

fetch/update/upsert도 배치로 처리


추가 노트 :


자 프로젝트 이름은 MinSync다

embedding, vectorstore, chunk는 기본적으로 Langchain의 base class를 ‘그대로’ 활용. ‘필요한 경우’ base class를 상속한 클래스를 만들어서, 거기에 멤버 함수를 추가하는 방식으로 확장 가능.

성능 균형을 위하여 ‘코어한’ 부분은 rust로 작성, 실제 사용은 python을 통해 end user는 사용하게 됨.
