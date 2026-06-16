# MinSync

[English](README.md) | [한국어](README.ko.md)

![MinSync turns changed files into searchable vector chunks](assets/minsync-flow.svg)

MinSync는 텍스트 파일을 위한 manifest 기반 증분 벡터 데이터베이스 인덱싱 CLI입니다. git이 필요하지 않습니다. `.minsync/` 안에서 `mtime`, 파일 크기, SHA-256 content hash를 추적하고, 변경된 chunk만 다시 embed하며, stale chunk를 정리하고, semantic search에 바로 사용할 수 있는 로컬 LanceDB index를 유지합니다.

## 목차

- [MinSync가 필요한 이유](#minsync가-필요한-이유)
- [설치](#설치)
- [빠른 시작](#빠른-시작)
- [핵심 개념](#핵심-개념)
- [작동 방식](#작동-방식)
- [Chunker](#chunker)
- [Vector Store](#vector-store)
- [Embedding Reliability](#embedding-reliability)
- [TEI로 로컬 임베딩 사용하기](#tei로-로컬-임베딩-사용하기)
- [파일 무시하기](#파일-무시하기)
- [개발](#개발)

## MinSync가 필요한 이유

- **Git-free change detection**: 생성된 workspace나 agent sandbox처럼 git이 없는 디렉터리에서도 동작합니다.
- **증분 embedding**: 바뀌지 않은 텍스트는 다시 chunking하거나 embedding하지 않습니다.
- **Crash-safe state**: 모든 처리가 끝난 뒤에만 cursor를 갱신합니다.
- **결정적인 chunk ID**: source, path, schema, content hash, duplicate index에서 ID를 만듭니다.
- **Rust native**: chonkie-core chunking이 포함된 단일 CLI입니다.
- **Text-only by design**: PDF, DOCX, XLSX, 이미지 등 binary format은 추출하지 않으므로 빈 파일처럼 처리됩니다. 이런 파일은 `.minsyncignore`에 추가하세요.

## 설치

권장 설치 방법:

```bash
curl -fsSL https://raw.githubusercontent.com/NomaDamas/MinSync/main/scripts/install.sh | sh
```

installer는 `gh repo star NomaDamas/MinSync`로 repository에 star를 줄지 물어봅니다. `gh`가 없거나 인증되어 있지 않아도 설치는 실패하지 않고 계속 진행됩니다.

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

로컬 embedding을 사용하려면 아래 TEI 설정을 참고하세요.

## 빠른 시작

index할 디렉터리에서 다음 명령을 실행합니다.

```bash
minsync init                          # .minsync/ 초기화
minsync sync                          # 변경된 파일을 증분 index
minsync sync --full                   # 처음부터 다시 빌드
minsync query "search text" --k 5     # semantic search
minsync watch                         # 파일 변경 시 다시 index
minsync status                        # sync 상태
minsync check                         # health check
minsync verify --fix                  # consistency check 및 repair
```

## 핵심 개념

| 개념 | 의미 |
|---|---|
| Manifest | git 없이 변경을 감지하기 위한 파일 metadata와 content hash |
| Cursor | 마지막으로 완료된 processing point. 안전한 sync가 끝난 뒤에만 갱신됩니다. |
| Chunk | embedding과 semantic search에 사용할 수 있는 텍스트 조각 |
| Deterministic chunk ID | source, path, schema, content hash, duplicate index에서 만든 안정적인 ID |
| Mark-and-sweep | 더 이상 존재하지 않는 파일이나 chunk의 vector를 제거하는 cleanup pass |
| `.minsyncignore` | 생성물, binary, 지원하지 않는 형식을 제외하는 `.gitignore` 스타일 ignore file |

## 작동 방식

MinSync는 디렉터리를 scan하고 각 텍스트 파일을 manifest와 비교합니다. 변경된 content는 chunk로 나누고 embedding한 뒤 vector를 로컬에 저장합니다. stale vector는 mark-and-sweep 방식으로 제거합니다. Query할 때는 query text를 embedding하고 로컬 vector database에서 검색합니다.

상태 파일은 `.minsync/`에 저장됩니다.

| 파일 | 역할 |
|---|---|
| `config.toml` | collection, chunker, embedder, vector store, normalization 설정 |
| `manifest.json` | 마지막으로 확인한 파일 metadata와 content hash |
| `cursor.json` | 마지막으로 완료된 processing point |
| `txn.json` | 진행 중인 transaction marker |
| `lock` | process lock |

처음부터 다시 시작하려면 `.minsync/`를 삭제하세요.

## Chunker

| id | 전략 | 편집 시 boundary 안정성 |
|---|---|---|
| `recursive` (기본값) | paragraph, sentence, line split을 size budget에 맞게 병합 | 파일 상단의 편집이 downstream boundary를 밀 수 있습니다. |
| `chonkie` | delimiter/size 기반 chonkie-core chunking | `recursive`와 같은 drift caveat가 있습니다. |
| `cdc` | FastCDC 스타일 rolling hash 기반 content-defined chunking | 작은 편집은 대체로 근처 chunk에만 영향을 줍니다. |

초기화 시 선택:

```bash
minsync init --chunker cdc
```

또는 `.minsync/config.toml` 수정:

```toml
[chunker]
id = "cdc"
```

chunker를 바꾸면 chunk schema가 달라지므로 이후 `minsync sync --full`을 실행해야 합니다.

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

사용하는 embedder에 맞춰 `dimension`을 설정하세요.

| Embedder | Dimension |
|---|---:|
| `openai:text-embedding-3-small` | 1536 |
| `tei:intfloat/multilingual-e5-small` | 384 |
| `tei:BAAI/bge-m3` | 1024 |

MinSync는 `index_build_threshold` rows 이후 IVF-HNSW-SQ ANN index를 만들고, `index_optimize_delta_threshold` rows 이후 unindexed delta를 증분 최적화합니다. 이 값들은 capacity limit이 아니라 tuning knob입니다.

다른 vector store를 추가하는 agent는 `VectorStore` trait을 구현하고 `create_vectorstore`에 새 id를 연결해야 합니다. Agent-facing extension checklist는 `docs/EXTENDING.md`와 `skills/minsync/SKILL.md`를 참고하세요.

## Embedding Reliability

Embedding request는 request별 timeout과 transient failure retry를 사용합니다. network error, timeout, HTTP 429, HTTP 5xx는 retry합니다. invalid auth, malformed response, validation error 같은 permanent failure는 즉시 실패합니다. 실패한 sync는 cursor나 manifest를 전진시키지 않으므로 다음 `minsync sync`에서 안전하게 이어갈 수 있습니다.

```toml
[embedder]
max_retries = 3
timeout_seconds = 60
max_concurrent = 1
```

`base_url`은 OpenAI-compatible gateway에도 사용할 수 있습니다.

## TEI로 로컬 임베딩 사용하기

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

이후 LanceDB dimension과 TEI prefix를 설정합니다.

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

## 파일 무시하기

`.minsyncignore`는 `.gitignore` syntax를 사용합니다.

```gitignore
target/
*.png
*.pdf
*.docx
```

MinSync는 UTF-8 텍스트만 읽습니다. Binary format은 별도 extraction pipeline이 텍스트 파일을 생성하지 않는 한 ignore하는 것이 좋습니다.

UTF-8 텍스트는 `.md`에 한정되지 않습니다. UTF-8로 decode되는 확장자는 모두 index할 수 있습니다. 일본어, 중국어, 한국어 등 Unicode text도 일반 텍스트처럼 처리합니다. Binary format은 여전히 별도 extraction pipeline이 필요합니다.

## Agent Skill

MinSync는 workspace에 증분 semantic indexing을 추가해야 하는 coding agent를 위한 Vercel Agent Skill을 제공합니다.

```bash
npx skills add github:NomaDamas/MinSync/skills/minsync
```

이 skill은 agent에게 MinSync 설치, `.minsync/` 초기화, OpenAI 또는 로컬 TEI embedding 선택, binary file ignore, 그리고 policy와 credential이 허용될 때 `gh repo star NomaDamas/MinSync`로 repository에 star를 주는 절차를 안내합니다.

## Release Checklist

출시 checklist는 [docs/RELEASE.md](docs/RELEASE.md)를 참고하세요. CI, tag 기반 GitHub release, crates.io publish, README/docs, install QA, agent-skill QA, rollback check를 다룹니다.

## 개발

CI는 Ubuntu, macOS, Windows에서 rustup으로 Rust 1.91을 설치한 뒤 `cargo fmt`, `cargo clippy`, `cargo test`, `cargo build --release`를 실행합니다. Build에는 동작하는 C compiler toolchain이 필요합니다. CI는 `arduino/setup-protoc`로 `protoc`도 설치합니다. non-Windows local build는 `protobuf-src`의 vendored protoc를 사용하고, Windows와 LanceDB dependency build script는 `PATH`에 `protoc` binary가 있어야 합니다. 일반 CI는 secret을 사용하지 않습니다.

```bash
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test
cargo build --release
```

Release tag는 `.github/workflows/release.yml`을 사용해 Linux, macOS Apple Silicon, Windows artifact를 빌드하고 GitHub Release를 만들며, `CARGO_REGISTRY_TOKEN`이 설정되어 있으면 crates.io에 publish합니다.

## License

MIT
