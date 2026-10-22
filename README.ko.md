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

MinSync는 기본적으로 [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)를 프로세스 안에서 embedding합니다. API key, TEI 서버, Hugging Face 게이트가 필요 없습니다. 첫 `minsync sync`에서 모델을 로컬 캐시로 받습니다.

모델 없이 BM25만 쓰려면 embedder, API 키, 모델 다운로드, 네트워크가 모두 필요 없는 `none` 모드를 사용하세요.

```bash
minsync init --embedder none
minsync sync                          # chunk와 full-text 인덱스만 생성하며 임베딩은 하지 않음
minsync query "정확한 단어"           # 이 workspace에서는 --mode 기본값이 bm25
```

`--embedder none`은 chunk마다 1차원 placeholder 벡터를 저장하고(LanceDB 스키마에 벡터 컬럼이 필요) ANN 인덱스는 만들지 않습니다. `--mode vector`와 `--mode hybrid`는 `--mode bm25`를 안내하며 실패합니다. 나중에 벡터 검색을 추가하려면 `minsync init --force --embedder <id>` 후 다시 sync하세요.

OpenAI embedding을 사용할 경우:

```bash
export OPENAI_API_KEY="sk-..."
minsync init --embedder openai:text-embedding-3-small
```

## 빠른 시작

```bash
minsync init                          # .minsync/ 초기화
minsync sync                          # 변경된 파일을 증분 index
minsync sync --full                   # 처음부터 다시 빌드
minsync query "검색어" --mode vector --k 5
minsync query "정확한 단어" --mode bm25 --k 5
minsync query "검색어" --mode hybrid --k 5
minsync watch                         # 초기 sync 후 파일 변경 감시 및 re-index
minsync watch --watch-on-sync-error   # 초기 sync 오류 후에도 유지하며 재시도
minsync status                        # sync 상태
minsync check                         # health check
minsync verify --fix                  # consistency check 및 repair
minsync optimize                      # store compact + re-index + prune
```

초기화할 때 BM25 tokenizer preset을 선택할 수 있습니다.

```bash
minsync init --language ko
```

지원 값은 `simple`, `ko`, `ja`, `zh`, `ar`, `multilingual`입니다.
한국어는 `kiwi-rs` Rust binding을 통한 Kiwi, 일본어는 사전이 내장된
Lindera, 중국어는 `jieba-rs`, 아랍어는 Discrawl PR #180을 따른
in-process light stemmer를 사용합니다.
`[lexical].language`를 바꾸면 다음 sync에서 full rebuild가 수행됩니다.

`minsync watch`는 filesystem event를 기다리기 전에 초기 sync를 수행합니다.
기본값에서는 초기 embedding 또는 vector store 오류가 `minsync sync`와 같이
즉시 종료됩니다. `--watch-on-sync-error`를 사용하면 watcher를 유지하면서
이후 파일 event에서 다시 시도합니다. 오류는 로그에 남고, sync가 성공한 뒤에만
cursor가 전진합니다.

Sync 출력은 파일 단위 변경과 chunk 단위 저장 결과를 분리합니다.
`files added/modified/deleted`는 원본 파일 변경을, `chunks
inserted/reused/removed`는 content-addressed index row의 삽입·재사용·삭제를
뜻합니다.
JSON sync 출력에는 `files_checked`, `elapsed_seconds`,
`freshness_check_only`, `query_ready`도 포함됩니다. 변경되지 않은 증분
sync는 보수적인 metadata fingerprint로 변경 없는 파일의 재해시를 건너뛴 뒤,
read, chunk, embedding, write, flush, index row rebuild는 수행하지 않고
반환합니다.
따라서 이때 elapsed time은 freshness-check 비용을 뜻합니다. 변경 sync와
full sync의 elapsed time에는 indexing과 LanceDB maintenance가 포함됩니다.
Credential 없이 로컬에서 비교하려면 `cargo run --quiet -- sync --format json`을
두 번 실행하고, 추적 중인 text 파일 하나를 수정해 다시 실행한 다음
`cargo run --quiet -- sync --full --format json`을 실행하세요. Query process
startup 비용은 이 sync 측정에 포함되지 않습니다.

Credential 없이 실행할 수 있는 저장된 scale benchmark는 다음과 같습니다.

```bash
cargo test --test issue_41_benchmark -- --nocapture
```

이 benchmark는 동일한 local corpus를 1x, 2x, 4x, 8x로 생성하고 full,
변경 없음, 파일 변경 sync의 millisecond와 BM25, vector, hybrid 검색의
p50/p95를 출력합니다. Query scaling과 LanceDB index maintenance를
분리하기 위해 query 측정은 의도적으로 in-memory store를 사용하며, LanceDB의
ANN/FTS threshold와 unindexed delta 동작은 live index test가 검증합니다.
측정값은 관찰 결과이며 machine-specific latency SLO가 아닙니다.

한국어 tokenizer에는 공식 Kiwi native library와 base model이 필요합니다.
`KIWI_LIBRARY_PATH`는 `libkiwi`, `KIWI_MODEL_PATH`는 압축 해제한
`models/cong/base`를 가리켜야 합니다. `kiwi-rs`(LGPL-2.1-or-later)는 정적으로
링크되며, 고지와 수정본으로 다시 링크하는 방법은 `THIRD_PARTY_NOTICES.md`를 참고하세요.
macOS/Linux에서는 `bash scripts/install-kiwi.sh`가 공식 자산을 내려받고
필요한 환경변수를 출력합니다.
Windows에서는 `powershell -ExecutionPolicy Bypass -File
scripts/install-kiwi.ps1`를 실행한 뒤 출력된 경로를 환경변수로 설정합니다.
Rust binding의 남은 `kiwi_config_t` ABI 차이가 upstream에서 해결될 때까지
Windows CI는 공식 Kiwi 0.22.2 자산을 사용합니다.

## 동작 방식

MinSync는 디렉터리를 scan하고 manifest와 비교해 각 텍스트 파일의 변경 여부를 판단합니다. 변경된 파일은 한 번만 chunk로 나누며, 같은 stable chunk ID를 LanceDB의 vector 검색과 BM25 검색이 공유합니다. stale row는 mark-and-sweep으로 제거됩니다. vector 모드는 query를 embedding하고, BM25 모드는 embedding 요청 없이 LanceDB full-text search를 사용하며, hybrid 모드는 두 순위를 결정적인 RRF(`k=60`)로 결합합니다.

선택한 analyzer가 문서와 query를 같은 방식으로 `lexical_text`로 변환한
뒤 LanceDB BM25 index에서 검색합니다.

상태는 `.minsync/`에 저장됩니다.

| 파일 | 목적 |
|---|---|
| `config.toml` | collection, chunker, embedder, vector store, normalization 설정 |
| `manifest.json` | 마지막으로 확인한 파일 metadata와 content hash |
| `cursor.json` | 마지막으로 완료된 processing point |
| `txn.json` | 진행 중 transaction marker |
| `progress.jsonl` | 중단된 sync가 이미 commit한 파일 목록(resume log) |
| `lock` | process lock |

처음부터 다시 시작하려면 `.minsync/`를 삭제하세요.

### 읽을 수 없는 항목

scan이 읽지 못한 디렉터리·파일(`EACCES`,
`EPERM`, `EIO`, `EBUSY`, `ETIMEDOUT`, `ESTALE`, symlink loop 등)은 scan을 중단시키지 않고 건너뛰며 sync
summary의 `scan_errors`(`path`, `kind` = `transient` 또는 `permanent`, `error`)에
기록하고 항목당 경고를 한 번 출력합니다. scan 도중 사라진 경로(`ENOENT`, `ENOTDIR`)는
삭제로 처리합니다. `status`와 `verify`도 같은 규칙을 적용하므로 `verify --fix`가 읽을 수 없는
경로의 chunk를 삭제하지 않습니다.

`[sync] on_file_error = "skip"`(기본값)이면 읽을 수 없는 항목은 `files_failed`(stage `read`)에도 기록되어 sync가 `partial`(종료 코드 6)이 되고, `"fail"`이면 첫 항목에서 경로를 밝히며 중단합니다.

**보장:** 오류가 난 항목은 *삭제*가 아니라 *unknown*입니다. 이전에 index된 chunk는
`--full` sweep 중에도 유지되며, 이후 scan이 해당 경로를 다시 읽거나 사라졌음을
확인할 때까지 남습니다. root 디렉터리 자체를 읽을 수 없으면 어떤 파일도 판단할 수
없으므로 sync는 여전히 실패합니다.

## Chunker

| id | 전략 | 편집 시 boundary 안정성 |
|---|---|---|
| `recursive` (기본값) | paragraph, sentence, line split을 size budget에 맞게 병합 | 파일 상단 편집이 downstream boundary를 밀 수 있습니다. |
| `chonkie` | delimiter/size 기반 chonkie-core chunking | `recursive`와 같은 drift caveat가 있습니다. |
| `cdc` | FastCDC 스타일 rolling hash 기반 content-defined chunking | 작은 편집은 보통 변경 지점 근처 chunk에만 영향을 줍니다. |

`recursive` chunker는 paragraph break, sentence delimiter(`. `, `? `,
`! `), line break를 기준으로 나눈 뒤 `chunker.options.max_chunk_size`
이하로 병합합니다. 설정한 크기는 일반 delimiter 기반 텍스트, delimiter가
없는 긴 줄, table chunk 모두에 적용되는 hard cap입니다. 초과하는 내용은
문자 경계에서 나뉘며 원래 길이를 포함한 warning을 기록합니다.

연속된 Markdown table line(앞쪽 공백은 허용하며 `|`로 시작하고 다른
`|`도 포함해야 함)은 하나의 table block으로 처리합니다. 유효한 table
header와 separator는 data row와 함께 유지하고, row가 cap 안에 들어오면
table chunk는 row 사이에서만 나눕니다. table이 여러 chunk가 되면 첫 chunk
이후의 모든 chunk가 header와 separator를 반복합니다. row 하나와 반복
header(또는 header만)가 cap을 넘으면 그 chunk는 문자 경계에서 나뉘므로
어떤 chunk도 `max_chunk_size`를 넘지 않으며, 나뉜 조각은 더 이상 유효한
table 조각이 아닙니다. table 내부에서 boundary를 선택할 때도 warning을
기록합니다.

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
dimension = 1024
index_build_threshold = 256
index_optimize_delta_threshold = 10000
```

`minsync init`은 알려진 embedder에 대해 `dimension`을 자동으로 설정합니다. 다른 모델을 사용할 경우 embedder에 맞게 `dimension`을 설정하세요.

| Embedder | Dimension |
|---|---:|
| `native:Qwen/Qwen3-Embedding-0.6B` (기본값) | 1024 |
| `tei:google/embeddinggemma-300m` | 768 |
| `openai:text-embedding-3-small` | 1536 |
| `tei:intfloat/multilingual-e5-small` | 384 |
| `tei:BAAI/bge-m3` | 1024 |

MinSync는 row 수가 `index_build_threshold`를 넘으면 IVF-HNSW-SQ ANN index를 만들고, unindexed delta가 `index_optimize_delta_threshold`에 도달하면 증분 optimize를 수행합니다. 이 값들은 tuning knob이며 capacity limit이 아닙니다.

새 vector store를 추가하는 agent는 `VectorStore` trait을 구현하고 `create_vectorstore`에 새 id를 연결해야 합니다. 자세한 내용은 `docs/EXTENDING.md`와 `skills/minsync/SKILL.md`를 참고하세요.

### 쓰기 경로, checkpoint, `optimize`

sync는 `[sync] batch_files`(기본 25)개 파일씩 묶어서 씁니다. batch마다 id만 읽는 존재 확인 1회, embedding 호출 1회, 재사용 chunk의 metadata update 1회, 새 chunk append 1회, stale chunk sweep 1회만 수행하므로 commit 수가 batch 안의 파일 수와 무관합니다. `id`와 `path`에 BTree index를 유지해 이 filter들이 index lookup으로 처리되게 하고, `[sync] checkpoint_every_files`(기본 50)개 파일마다 fragment가 `compact_fragment_threshold`(기본 64) 이상이면 compact하고 오래된 version을 prune합니다. 실행 중인 sync는 checkpoint마다 `sync progress`를 로그로 남깁니다. 중단되어도 최대 한 batch만 다시 처리합니다.

```toml
[sync]
batch_files = 25              # store commit당 파일 수; 1이면 파일별 commit
checkpoint_every_files = 50   # 0이면 중간 checkpoint 비활성화

[vectorstore.options]
compact_fragment_threshold = 64
```

`minsync optimize`는 compact + re-index + prune만 수행합니다(embedding, 파일 읽기 없음). 중단된 sync 때문에 조각난 store를 복구할 때 사용하세요.

## 건너뛴 파일

기본 설정에서는 파일 하나가 실패해도 sync 전체가 중단되지 않습니다. read, chunk, token 수 계산, embed 단계에서 실패한 파일(EACCES, EBUSY, EIO, parser 오류, embedder의 입력 거부 등)은 건너뛰고 기록합니다. batch commit이 실패하면 그 batch를 파일 단위로 다시 시도해 원인 파일만 건너뜁니다. 건너뛴 파일은 완료로 기록되지 않습니다. 새 파일은 저장되는 manifest에서 빠지고, 이전에 색인된 파일은 이전 manifest 항목을 유지하므로 다음 `minsync sync`가 자동으로 다시 시도하며, 성공할 때까지 기존 chunk가 (full sync 중에도) 검색됩니다.

```json
"files_failed": [
  { "path": "notes/broken.pdf", "stage": "read", "error": "io error: notes/broken.pdf: Permission denied (os error 13)" }
],
"partial": true
```

`stage`는 `read`, `chunk`, `truncate`, `embed`, `store` 중 하나입니다. `partial`이 true이면 CLI는 종료 코드 **6**(완료, 일부 파일 건너뜀)으로 끝나고, text 출력은 건너뛴 경로를 stderr에 나열합니다. 하드 실패의 종료 코드(1, 3, 4, 5)는 그대로입니다. 아직 아무것도 색인하지 못한 실행에서 처음 몇 개 파일이 embed/store 단계에서 연속 실패하면(예: embedder 중단) 파일 문제가 아니라 장애로 보고, 파일 경로가 포함된 오류(종료 코드 5/4)로 sync를 중단합니다.

```toml
[sync]
on_file_error = "skip"   # "fail"이면 첫 실패 파일에서 중단(기존 동작)
```

## Embedding 안정성

Embedding request는 per-request timeout과 transient failure retry를 사용합니다. Retry 대상은 network error, timeout, HTTP 429, HTTP 5xx입니다. Invalid auth, malformed response, validation error 같은 permanent failure는 즉시 실패합니다. 실패하거나 강제 종료된 sync는 cursor나 manifest를 전진시키지 않지만 resume log(`progress.jsonl`)를 남깁니다. 다음 `minsync sync`는 중단된 실행이 이미 commit한 파일을 건너뛰고 이어서 처리합니다. 그 사이 파일, chunker, normalizer, embedder, lexical language가 바뀌면 log는 폐기됩니다.

```toml
[embedder]
max_retries = 3
timeout_seconds = 60
max_concurrent = 1
```

`base_url`은 OpenAI-compatible gateway에도 사용할 수 있습니다.

## 로컬 Embedding

기본 embedder `native:Qwen/Qwen3-Embedding-0.6B`는 [candle](https://github.com/huggingface/candle)로 프로세스 안에서 실행됩니다(모델 코드는 [fastembed-rs](https://github.com/Anush008/fastembed-rs)에서 가져왔으며 `THIRD_PARTY_NOTICES.md`에 표기했습니다). `minsync init`은 `dimension = 1024`를 쓰고 query/passage prefix는 비웁니다. Apple Silicon에서는 Metal, 그 외에는 CPU를 사용합니다.

첫 embedding 호출이 Apache-2.0 모델을 Hugging Face Hub에서 `~/.cache/minsync/models`로 받습니다. `config.toml`의 `model_cache_dir`, `FASTEMBED_CACHE_DIR`, `HF_HOME`으로 경로를 바꿀 수 있습니다. 이후 실행은 캐시에서 로드합니다.

```bash
minsync init
minsync sync
minsync query "검색어" --k 5
```

768차원 TEI 인덱스는 1024차원 벡터와 섞으면 안 됩니다. embedder를 바꾼 뒤에는 전체를 다시 임베딩하세요.

```bash
minsync sync --full
```

### Native 런타임

`[embedder]`의 선택 설정 두 가지로 native 런타임을 조정합니다. 대부분은 기본값이면 충분합니다.

```toml
[embedder]
dtype = "auto"      # auto | f32 | f16 | bf16
attention = "auto"  # auto | eager | tiled | fused
```

- `dtype`: `auto`는 Metal에서 `f16`, 그 외에는 `f32`입니다. 반정밀도는 모델 메모리를 약 3분의 1 줄이며, 벡터는 `f32` 결과와 코사인 유사도 0.0001 이내로 유지되고 테스트한 검색 결과는 바뀌지 않았습니다. 전체 정밀도가 필요하면 `dtype = "f32"`로 두세요. 기존 인덱스는 그대로 유효합니다.
- `attention`: `auto`는 장치의 fused 커널(Metal scaled-dot-product attention, CPU flash attention)을 쓰고, 그 외 장치(CUDA 빌드)에서는 점수 행렬을 블록 단위로 계산해 chunk 길이의 제곱에 비례하던 피크 메모리를 없앱니다. `eager`는 점수 행렬 전체를 만드는 기준 구현(긴 chunk에서 수 GiB)이고, `tiled`는 어느 장치에서나 블록 단위로 계산하며, `fused`는 장치에 fused 커널이 없으면 오류를 냅니다.

Apple M4 Max(macOS, Metal)에서 기본 chunk 크기로 `minsync sync`를 세 번 실행한 중앙값이며, 이전 릴리스(f32, eager attention)와 비교한 결과입니다.

| 코퍼스 | 이전 | 이후(기본값) | 이후(`dtype = "f32"`) |
|---|---|---|---|
| 한국어 문단 600개(약 550자) | 125초, 6.3 GiB | 58초, 3.0 GiB | 61초, 4.6 GiB |
| 영어 초록 1000개 | 173초, 4.9 GiB | 83초, 3.0 GiB | 89초, 4.5 GiB |
| 약 3.7k자 chunk 48개 | 97초, 7.3 GiB | 26초, 3.2 GiB | 29초, 4.9 GiB |

### Node.js에서 사용

`@nomadamas/minsync-native`([`bindings/node`](bindings/node))는 같은 native embedder를 프로세스 내부 Node-API addon으로 제공합니다. 오래 실행되는 Node/Electron 호스트가 별도 프로세스나 호출마다의 콜드 모델 로드 없이 임베딩할 수 있습니다. 이 런타임과 같은 가중치, 캐시 디렉터리, 배치 planner, deadline을 쓰며, 같은 텍스트·장치·배치 구성에서는 CLI와 비트 단위로 동일한 벡터를 반환합니다.

```ts
import { NativeEmbedder } from '@nomadamas/minsync-native'

const embedder = new NativeEmbedder({ idleUnloadSeconds: 600 })
const { id, dimension, dtype, device } = await embedder.identity()
const passages = await embedder.embed(['첫 번째 문단', '두 번째 문단']) // Float32Array[]
const query = await embedder.embedQuery('어느 문단?')
await embedder.dispose()
```

Apple M 시리즈 Mac에서 모델은 한 번(약 2.2초) 로드되고, 이후 warm 쿼리는 약 17ms이며, `unload()`는 RSS 약 2.7 GB를 반환합니다. 옵션과 abort·deadline 동작은 [`bindings/node/README.md`](bindings/node/README.md)를 참고하세요.

### 다른 provider

`openai:`와 `tei:`는 그대로 사용할 수 있습니다. 로컬 TEI + EmbeddingGemma를 유지하려면:

```bash
brew install text-embeddings-inference
export HF_TOKEN="hf_..."   # https://huggingface.co/google/embeddinggemma-300m 약관 동의 후
text-embeddings-router --model-id google/embeddinggemma-300m --port 8080 --dtype float32
curl http://localhost:8080/health
minsync init --embedder tei:google/embeddinggemma-300m
```

EmbeddingGemma는 float16을 지원하지 않으므로 `--dtype float32`(또는 `bfloat16`)를 유지하세요. `minsync init --embedder tei:google/embeddinggemma-300m`은 Gemma retrieval prefix와 `dimension = 768`을 씁니다.

다른 TEI 모델을 사용하려면 init 시점에 id를 넘기고 `.minsync/config.toml`에서 dimension과 prompt prefix를 설정하세요.

```bash
minsync init --embedder tei:intfloat/multilingual-e5-small
```

```toml
[embedder]
id = "tei:intfloat/multilingual-e5-small"
base_url = "http://localhost:8080"
query_prefix = "query: "
passage_prefix = "passage: "

[vectorstore.options]
dimension = 384
```

### Embedding 서버

`minsync serve`는 native embedder를 한 번만 로드해 loopback으로 제공합니다. 다른 도구와 반복되는 CLI 호출이 모델을 각자 로드하지 않고 하나를 공유합니다.

```bash
minsync serve                    # workspace config가 있으면 그 설정, 없으면 기본 모델
minsync serve --port 8089 --idle-timeout 600
```

| Endpoint | 요청 | 응답 |
|---|---|---|
| `POST /embed` (TEI) | `{"inputs": [...], "truncate"?: bool}` | `[[f32, ...], ...]` |
| `POST /v1/embeddings` (OpenAI) | `{"model", "input": [...], "encoding_format"?: "float" \| "base64"}` | `{"object": "list", "data": [{"index", "embedding"}], "model", "usage"}` |
| `GET /health` | | model id, dimension, dtype, device, pid, port |

- 같은 텍스트·device·dtype이면 in-process `native:`와 같은 vector를 돌려줍니다. 요청은 `sync`와 같은 batch planner, padded-token 상한, pass별 deadline을 거칩니다.
- Prefix는 호출자가 붙입니다. 두 endpoint 모두 `"input_type": "query" | "passage"`를 받으면 설정된 `query_prefix` / `passage_prefix`를 붙입니다.
- 모델 context를 넘는 입력은 `truncate: true`(TEI)가 아니면 `413`입니다. `--max-inputs`(기본 512)개 초과 입력이나 `--max-request-bytes`(기본 32 MiB) 초과 body도 `413`, 잘못된 JSON은 `400`/`422`입니다. forward pass가 deadline을 넘기면 `503`을 돌려주고 서버는 exit code 7로 종료합니다(멈춘 thread가 모델을 잡고 있기 때문).
- 인증이 없으므로 `127.0.0.1`(또는 `::1`)에만 bind하고 다른 `--host`는 거부합니다. `--port 0`(기본)은 빈 port를 고릅니다.
- `pid`, `host`, `port` 파일은 `~/.cache/minsync/serve/<model instance>/`에 있습니다(`--state-dir`로 변경). 같은 model·device·dtype·attention·`max_length`로 두 번째 `minsync serve`를 실행하면 모델을 로드하지 않고 실행 중인 서버를 출력(`--format json`이면 `"event": "adopted"`)한 뒤 종료합니다.
- `--idle-timeout`초(기본 300, `0`은 비활성) 동안 embedding 요청이 없으면 종료하고 모델을 해제합니다. `SIGTERM`/Ctrl-C도 정상 종료하며, 두 경우 모두 discovery 파일을 지웁니다.

`native:` workspace는 실행 중인 서버를 자동으로 사용합니다. `sync`, `query`, `check`, `verify`, `watch`는 첫 embedding 호출 때 같은 model instance의 discovery 파일을 찾아 텍스트를 서버로 보냅니다(workspace prefix 적용, in-process 엔진처럼 `max_length`에서 truncate). 서버가 없거나 응답을 멈추면 이전처럼 in-process로 embedding합니다. M 시리즈 Mac에서 warm 서버는 `query`를 약 2.3 s / 2.4 GB에서 약 0.22 s / 59 MB로 줄입니다. `MINSYNC_SERVER=off`면 항상 in-process로 embedding합니다. 서버가 발견되려면 기본 `--state-dir`을 써야 합니다.

TEI 또는 OpenAI 호환 client를 이 서버로 지정하면 됩니다. 예: 다른 workspace

```toml
[embedder]
id = "tei:Qwen/Qwen3-Embedding-0.6B"
base_url = "http://127.0.0.1:8089"

[vectorstore.options]
dimension = 1024
```

OpenAI 호환 client는 `/v1` base URL을 씁니다. 예: discrawl (API key env가 필요하지만 서버는 무시합니다)

```toml
[search.embeddings]
enabled = true
provider = "openai_compatible"
model = "Qwen/Qwen3-Embedding-0.6B"
base_url = "http://127.0.0.1:8089/v1"
api_key_env = "OPENAI_API_KEY"   # 비어 있지 않은 아무 값
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

## Architecture Scope

MinSync는 Markdown 중심 UTF-8 text를 증분 인덱싱하는 작고 효율적인
cross-platform Rust 도구로 유지합니다. 범용 증분 dataflow framework를
목표로 하지 않습니다. CocoIndex 비교, extension 경계, 채택된 범위 결정은
[docs/ARCHITECTURE_SCOPE.md](docs/ARCHITECTURE_SCOPE.md)를 참고하세요.

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
