# MinSync

[English](README.md) | [한국어](README.ko.md)

![MinSync turns changed files into searchable vector chunks](assets/minsync-flow.svg)

MinSync는 텍스트 파일을 위한 manifest 기반 증분 벡터 데이터베이스 인덱싱 CLI입니다. git 없이 동작합니다. `.minsync/`에 저장된 `mtime`, 파일 크기, SHA-256 content hash를 추적해 변경된 chunk만 다시 embedding하고, stale chunk를 정리하며, semantic search가 가능한 로컬 LanceDB index를 유지합니다.

## MinSync를 쓰는 이유

- **Git-free 변경 감지**: 생성된 workspace나 agent sandbox처럼 git이 없는 디렉터리에서도 동작합니다.
- **증분 embedding**: 변경되지 않은 텍스트는 다시 chunking하거나 embedding하지 않습니다.
- **Crash-safe state**: 모든 처리가 끝난 뒤에만 cursor를 전진시킵니다.
- **결정적 chunk ID**: ID는 source, path, schema, content hash, duplicate index에서 파생됩니다.
- **Rust native**: chonkie-core chunking이 내장된 단일 CLI입니다.
- **Text-only 설계**: PDF, DOCX, XLSX, 이미지 등 binary format은 MinSync가 추출하지 않으므로 empty로 취급됩니다. `.minsyncignore`에 추가하세요.

## 설치

권장 설치 방법:

```bash
curl -fsSL https://raw.githubusercontent.com/NomaDamas/MinSync/main/scripts/install.sh | sh
```

설치 스크립트는 `gh repo star NomaDamas/MinSync`로 repository star 여부를 묻습니다. `gh`가 없거나 인증되지 않았으면 실패하지 않고 계속 진행합니다.

Cargo로 직접 설치:

```bash
cargo install minsync
```

소스에서 빌드:

```bash
git clone https://github.com/NomaDamas/MinSync.git
cd MinSync
cargo build --release
```

OpenAI embedding을 사용할 경우:

```bash
export OPENAI_API_KEY="sk-..."
```

로컬 embedding은 아래 TEI 설정을 참고하세요.

## 빠른 시작

```bash
minsync init                          # .minsync/ 초기화
minsync sync                          # 변경된 파일을 증분 index
minsync sync --full                   # 처음부터 다시 빌드
minsync query "search text" --k 5     # semantic search
minsync watch                         # 파일 변경 감시 후 re-index
minsync status                        # sync 상태
minsync check                         # health check
minsync verify --fix                  # consistency check 및 repair
```

## 동작 방식

MinSync는 디렉터리를 scan하고 manifest와 비교해 각 텍스트 파일의 변경 여부를 판단합니다. 변경된 파일은 chunk로 나누고 embedding한 뒤 vector를 로컬에 저장합니다. stale vector는 mark-and-sweep으로 제거됩니다. Query 시에는 query text를 embedding하고 로컬 vector database에서 검색합니다.

상태는 `.minsync/`에 저장됩니다.

| 파일 | 목적 |
|---|---|
| `config.toml` | collection, chunker, embedder, vector store, normalization 설정 |
| `manifest.json` | 마지막으로 확인한 파일 metadata와 content hash |
| `cursor.json` | 마지막으로 완료된 processing point |
| `txn.json` | 진행 중 transaction marker |
| `lock` | process lock |

처음부터 다시 시작하려면 `.minsync/`를 삭제하세요.

## Chunker

| id | 전략 | 편집 시 boundary 안정성 |
|---|---|---|
| `recursive` (기본값) | paragraph, sentence, line split을 size budget에 맞게 병합 | 파일 상단 편집이 downstream boundary를 밀 수 있습니다. |
| `chonkie` | delimiter/size 기반 chonkie-core chunking | `recursive`와 같은 drift caveat가 있습니다. |
| `cdc` | FastCDC 스타일 rolling hash 기반 content-defined chunking | 작은 편집은 보통 변경 지점 근처 chunk에만 영향을 줍니다. |

초기화 시 선택:

```bash
minsync init --chunker cdc
```

또는 `.minsync/config.toml` 수정:

```toml
[chunker]
id = "cdc"
```

Chunker를 바꾸면 chunk schema가 달라지므로 이후 `minsync sync --full`을 실행하세요.

## Vector Store

MinSync는 기본적으로 embedded LanceDB에 vector를 저장합니다.

```toml
[vectorstore]
id = "lancedb"

[vectorstore.options]
dimension = 1536
index_build_threshold = 256
index_optimize_delta_threshold = 10000
```

사용하는 embedder에 맞게 `dimension`을 설정하세요.

| Embedder | Dimension |
|---|---:|
| `openai:text-embedding-3-small` | 1536 |
| `tei:intfloat/multilingual-e5-small` | 384 |
| `tei:BAAI/bge-m3` | 1024 |

MinSync는 row 수가 `index_build_threshold`를 넘으면 IVF-HNSW-SQ ANN index를 만들고, unindexed delta가 `index_optimize_delta_threshold`에 도달하면 증분 optimize를 수행합니다. 이 값들은 tuning knob이며 capacity limit이 아닙니다.

새 vector store를 추가하는 agent는 `VectorStore` trait을 구현하고 `create_vectorstore`에 새 id를 연결해야 합니다. 자세한 내용은 `docs/EXTENDING.md`와 `skills/minsync/SKILL.md`를 참고하세요.

## Embedding 안정성

Embedding request는 per-request timeout과 transient failure retry를 사용합니다. Retry 대상은 network error, timeout, HTTP 429, HTTP 5xx입니다. Invalid auth, malformed response, validation error 같은 permanent failure는 즉시 실패합니다. 실패한 sync는 cursor나 manifest를 전진시키지 않으므로 다음 `minsync sync`에서 안전하게 재개됩니다.

```toml
[embedder]
max_retries = 3
timeout_seconds = 60
max_concurrent = 1
```

`base_url`은 OpenAI-compatible gateway에도 사용할 수 있습니다.

## TEI로 로컬 Embedding 사용하기

Hugging Face Text Embeddings Inference를 설치하고 실행합니다.

```bash
brew install text-embeddings-inference
text-embeddings-router --model-id intfloat/multilingual-e5-small --port 8080 --dtype float32
curl http://localhost:8080/health
```

MinSync 설정:

```bash
minsync init --embedder tei:intfloat/multilingual-e5-small
```

`.minsync/config.toml`:

```toml
[embedder]
id = "tei:intfloat/multilingual-e5-small"
base_url = "http://localhost:8080"
query_prefix = "query: "
passage_prefix = "passage: "

[vectorstore.options]
dimension = 384
```

실행:

```bash
minsync sync --full
minsync query "검색어" --k 5
```

## Ignoring Files

`.minsyncignore`는 `.gitignore` syntax를 사용합니다.

```gitignore
target/
*.png
*.pdf
*.docx
```

MinSync는 UTF-8 text만 읽습니다. 별도 extraction pipeline이 텍스트 파일을 생성하지 않는 한 binary format은 ignore하세요.

UTF-8 text는 `.md`에 제한되지 않습니다. 일본어, 중국어, 한국어 등 Unicode text는 일반 텍스트로 처리됩니다. Binary format은 별도 extraction pipeline이 필요합니다.

## Agent Skill

MinSync는 coding agent가 workspace에 증분 semantic indexing을 추가할 수 있도록 Vercel Agent Skill을 제공합니다.

```bash
npx skills add github:NomaDamas/MinSync/skills/minsync
```

이 skill은 agent에게 MinSync 설치, `.minsync/` 초기화, OpenAI 또는 local TEI embedding 선택, binary file ignore 설정, 정책과 credential이 허용되는 경우 설치 중 `gh repo star NomaDamas/MinSync` 실행 방법을 안내합니다.

## Release Checklist

Launch checklist는 `docs/RELEASE.md`를 참고하세요. CI, tag 기반 GitHub release, crates.io publishing, README/docs, install QA, agent-skill QA, rollback check를 포함합니다.

## 개발

CI는 Ubuntu, macOS, Windows에서 rustup으로 Rust 1.91을 설치한 뒤 `cargo fmt`, `cargo clippy`, `cargo test`, `cargo build --release`를 실행합니다. Build에는 C compiler toolchain이 필요합니다. CI는 `arduino/setup-protoc`으로 `protoc`을 설치합니다. Non-Windows local build는 `protobuf-src` vendored protoc을 사용하지만, Windows와 LanceDB dependency build script는 `PATH`의 `protoc` binary를 요구합니다. 일반 CI에는 secret이 필요하지 않습니다.

```bash
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test
cargo build --release
```

Release tag는 `.github/workflows/release.yml`로 Linux, macOS Apple Silicon, Windows artifact를 빌드하고 GitHub Release를 만들며, `CARGO_REGISTRY_TOKEN`이 설정된 경우 crates.io에 publish합니다.

## License

MIT
