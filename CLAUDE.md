# MinSync

Git 저장소의 변경사항을 `git diff`로 감지하여 벡터 DB를 증분 갱신하는 도구.

## 핵심 원칙

- **CLI = Python API Thin Wrapper**: 모든 기능은 `MinSync` 클래스 메서드. CLI는 인자 파싱+출력만.
- **config.yaml = pre-built만**: 커스텀 chunker/embedder/vectorstore는 Python API로 객체 주입.
- **LangChain 인터페이스 제공, 의존성 미포함**: 외부 패키지(weaviate-client 등)는 사용자가 직접 설치.
- **상태 디렉토리**: `.minsync/` (config.yaml, cursor.json, txn.json, lock)
- **인덱싱 대상**: git-tracked 전체 → `.gitignore`는 git이 자동 제외 → `.minsyncignore`로 추가 제외 (`.gitignore` 문법). `include` 없음.
- **`.minsyncignore` 변경 시 `--full` 불필요**: sync + verify 조합으로 수렴.
- **CI/CD 우선 설계**: GitHub Actions 비대화형 실행이 기본 전제.
- **Crash-safe**: cursor는 모든 처리 완료 후에만 갱신. mark+sweep으로 수렴 보장.
- **Deterministic ID**: `sha256(repo_id+ref+path+schema_id+chunk_type+heading_path+content_hash+dup_index)`.

## CLI 커맨드

`init` / `sync` / `query` / `status` / `check` / `verify`

## 상세 문서

- `ai_instruction/CLI_SPEC.md` — CLI 및 Python API 명세
- `ai_instruction/USER_SCENARIOS.md` — 유저 시나리오 19개
- `ai_instruction/E2E_TEST_PLAN.md` — E2E 테스트 44개
- `ai_instruction/IMPLEMENTATION_CHECKLIST.md` — 구현 체크리스트 232항목
- `PRD.md` — 원본 PRD
