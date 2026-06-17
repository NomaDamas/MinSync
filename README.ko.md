# MinSync

[English](README.md) | [한국어](README.ko.md)

MinSync는 Rust로 작성된 manifest 기반 증분 벡터 DB 인덱싱 CLI입니다. git 없이 동작합니다.

## 목차

- [설치](#설치)
- [사용법](#사용법)
- [작동 방식](#작동-방식)
- [Chunker](#chunker)
- [Vector store](#vector-store)
- [Embedding network reliability](#embedding-network-reliability)
- [Hugging Face TEI로 로컬 embedding 사용하기](#hugging-face-tei로-로컬-embedding-사용하기)
- [.minsyncignore](#minsyncignore)
- [개발](#개발)

## 설치

```bash
git clone https://github.com/NomaDamas/MinSync.git
cd MinSync
cargo build --release
export OPENAI_API_KEY="sk-..."
```

## 사용법

```bash
minsync init                          # .minsync/ 초기화
minsync sync                          # 파일 증분 index
minsync sync --full                   # 처음부터 다시 빌드
minsync query "search text" --k 5     # semantic search
minsync watch                         # .md/.txt 변경 감시 및 증분 re-index
minsync status                        # sync 상태
minsync check                         # health check
minsync verify --fix                  # consistency check 및 repair
```

## 작동 방식

MinSync는 디렉터리를 scan하고 manifest 비교(mtime + size + SHA-256 content hash)로 파일 변경을 감지합니다. 변경된 파일은 [chonkie-core](https://github.com/chonkie-inc/chunk)의 split/merge primitive를 기반으로 한 recursive chunker로 나눕니다. paragraph, sentence, line boundary를 size budget에 맞춰 병합하고, OpenAI로 embedding한 뒤 vector를 로컬에 저장합니다. 변경된 content만 다시 embedding하며 stale chunk는 자동으로 정리됩니다.

상태 파일은 `.minsync/`에 저장됩니다. 처음부터 다시 시작하려면 이 디렉터리를 삭제하세요.

### Chunker

| id | 전략 | 편집 시 boundary 안정성 |
|---|---|---|
| `recursive` (기본값) | paragraph, sentence, line split을 size budget에 맞게 병합 | 파일 상단의 편집이 모든 downstream boundary를 밀 수 있습니다. |
| `chonkie` | delimiter/size 기반 chonkie-core chunking | `recursive`와 같은 drift caveat가 있습니다. |
| `cdc` | FastCDC 스타일 gear rolling hash 기반 content-defined chunking | content로 boundary를 고르므로 작은 편집은 대체로 변경된 chunk 근처만 다시 embedding합니다. |

초기화 시 `minsync init --chunker cdc`로 선택하거나 `.minsync/config.toml`의 `chunker.id`로 설정할 수 있습니다. `cdc`는 `chunker.options.max_chunk_size`에서 size window를 계산합니다. 최대값은 `max_chunk_size`, 평균은 절반, 최소값은 1/8입니다.

re-embedding 비용이 중요한 크고 자주 바뀌는 파일에는 `cdc`가 적합합니다. sentence/paragraph 의미 단위에 맞춘 chunk가 중요하면 `recursive`가 적합합니다. chunker를 바꾸면 chunk schema가 달라지므로 이후 `minsync sync --full`을 실행해야 합니다.

### Vector store

MinSync는 기본적으로 embedded [LanceDB](https://github.com/lancedb/lancedb) database에 vector를 저장합니다. 기본 설정은 `vectorstore.id = "lancedb"`입니다. MinSync는 non-Windows target의 자체 build script를 위해 `protobuf-src`를 통해 `protoc`을 vendoring합니다. Windows build와 LanceDB dependency build script에는 `PATH`에 `protoc` binary가 필요합니다.

`.minsync/config.toml`에서 사용하는 embedder에 맞춰 embedding dimension을 설정하세요.

```toml
[vectorstore]
id = "lancedb"
[vectorstore.options]
dimension = 1536   # openai:text-embedding-3-small; e5-small은 384, bge-m3는 1024
```

**ANN indexing.** 새 table은 exhaustive flat scan으로 검색됩니다. 정확도는 100%지만 O(n)입니다. Table이 `index_build_threshold` chunk(기본값 256)를 넘으면 다음 `sync`/`flush` 때 MinSync가 cosine distance에 고정된 IVF-HNSW-SQ approximate-nearest-neighbour index를 만들어 similarity search를 가속합니다.

새로 sync된 chunk는 LanceDB의 indexed + flat-over-delta search로 즉시 검색 가능합니다. unindexed delta가 `index_optimize_delta_threshold`(기본값 10,000)에 도달하면 MinSync는 full rebuild나 k-means retrain 없이 기존 index에 증분 최적화를 수행합니다. 수동 작업은 필요 없습니다.

두 threshold는 독립적인 tuning knob입니다. min/max pair도 아니고 capacity limit도 아닙니다. 원하는 개수의 vector를 저장하고 검색할 수 있습니다.

```toml
[vectorstore.options]
dimension = 1536
index_build_threshold = 256              # total rows가 이 값에 도달하면 ANN index 생성
index_optimize_delta_threshold = 50000   # 높이면 optimize 빈도가 줄고, 낮추면 query latency를 더 타이트하게 유지
```

## Embedding network reliability

Embedding request(OpenAI와 TEI)는 request별 timeout을 사용하고, network error, timeout, HTTP 429, HTTP 5xx 같은 transient failure에는 exponential backoff와 jitter로 retry합니다. Invalid auth, 4xx validation error, malformed response 같은 permanent failure는 즉시 실패합니다. 실패한 sync는 cursor나 manifest를 전진시키지 않으므로 다음 `minsync sync`에서 정확히 이어서 처리할 수 있습니다.

`.minsync/config.toml`에서 조정할 수 있습니다.

```toml
[embedder]
max_retries = 3       # 첫 시도 이후 retry 횟수(total attempts = max_retries + 1)
timeout_seconds = 60  # request별 HTTP timeout
max_concurrent = 1    # 하나의 sync embedding call 안에서 concurrent batch request 수
```

`base_url`은 OpenAI-compatible proxy/gateway를 위해 `openai:` embedder에도 적용됩니다.

## Hugging Face TEI로 로컬 embedding 사용하기

로컬 [Text Embeddings Inference](https://github.com/huggingface/text-embeddings-inference) server를 사용하면 `OPENAI_API_KEY` 없이 MinSync를 완전히 offline으로 실행할 수 있습니다.

### 1. TEI 설치 및 실행(macOS Apple Silicon)

```bash
brew install text-embeddings-inference
```

```bash
text-embeddings-router --model-id intfloat/multilingual-e5-small --port 8080 --dtype float32
```

첫 실행에서는 model 약 470 MB를 `~/.cache/huggingface`에 다운로드합니다. 준비되면 다음 명령으로 확인합니다.

```bash
curl http://localhost:8080/health   # 200을 반환해야 합니다.
```

### 2. MinSync 설정

`minsync init --embedder tei:intfloat/multilingual-e5-small`을 실행한 뒤 `.minsync/config.toml`을 수정하거나, `[embedder]` section을 직접 설정합니다.

```toml
[embedder]
id = "tei:intfloat/multilingual-e5-small"
base_url = "http://localhost:8080"
query_prefix = "query: "
passage_prefix = "passage: "
```

`e5` model은 retrieval 품질을 위해 input prefix가 필요합니다. 검색 query에는 `query: `가, index할 document에는 `passage: `가 붙습니다. MinSync는 config에 따라 이를 자동 적용합니다.

### 3. LanceDB dimension 설정

`intfloat/multilingual-e5-small`은 384-dimensional vector를 생성합니다. Embedder에 맞춰 dimension을 설정하세요.

```toml
[vectorstore]
id = "lancedb"
[vectorstore.options]
dimension = 384
```

### 4. 실행

```bash
minsync sync --full          # 첫 sync(init이 manifest baseline을 만들기 때문에 --full 사용)
minsync query "검색어" --k 5
minsync watch
```

### 대체 model

| Model | Dim | Notes |
|---|---|---|
| `intfloat/multilingual-e5-small` | 384 | 한국어를 포함한 multilingual model. 권장 기본값 |
| `dragonkue/multilingual-e5-small-ko-v2` | 384 | Korean-tuned e5-small variant |
| `BAAI/bge-m3` | 1024 | Prefix 불필요. 더 큰 model |

## .minsyncignore

`.gitignore` syntax로 index에서 제외할 파일을 지정합니다.

```gitignore
target/
*.png
*.pdf
```

## 개발

CI는 Ubuntu, macOS, Windows의 표준 GitHub-hosted runner를 가정하고, rustup으로 Rust 1.91을 설치합니다.
Build에는 동작하는 C compiler toolchain, MinSync non-Windows build script를 위한 protobuf-src vendored protoc, Windows와 LanceDB dependency build script를 위한 setup-protoc, CI에서 빌드되는 LanceDB native dependency가 필요합니다. 일반 CI에는 secret이 필요하지 않습니다.

```bash
cargo test            # full test suite
cargo clippy          # lint
cargo fmt             # format
```

## License

MIT
