# MinSync Release Checklist

Use this checklist before cutting a public release.

## Release note: pre-#59 macOS/Metal indexes

The native macOS/Metal embedder bug fixed by [#59](https://github.com/NomaDamas/MinSync/pull/59)
could silently write corrupt, finite vectors when long chunks exceeded the affected
attention-memory limit described in [#58](https://github.com/NomaDamas/MinSync/issues/58).
Indexes created before that fix may therefore contain vectors that do not represent
their stored text. As the manifest treats unchanged files as unchanged, an
incremental sync does not repair those vectors.

For the diagnostic added in [#61](https://github.com/NomaDamas/MinSync/issues/61),
run `minsync verify`. If it reports suspicious duplicate-vector pairs, or if the
native embedder was used on macOS before #59, run `minsync sync --full` once to
re-embed the index.

## 1. Version and Metadata

- [ ] Confirm `Cargo.toml` version is the intended release version.
- [ ] Confirm package metadata: description, license, repository, readme, keywords, and categories.
- [ ] Run `cargo package --list` and check that `README.md`, `assets/minsync-flow.svg`, `docs/RELEASE.md`, and `skills/minsync/SKILL.md` are present.
- [ ] Confirm no private agent artifacts, local state, secrets, or generated run logs are included.

## 2. Documentation

- [ ] README is English, starts with the MinSync explanatory image, and clearly explains install, quick start, state files, chunkers, LanceDB, OpenAI, TEI, `.minsyncignore`, and development.
- [ ] README install path uses `scripts/install.sh` and documents the optional `gh repo star NomaDamas/MinSync` prompt.
- [ ] README documents the Vercel Agent Skill installation command.
- [ ] `docs/RELEASE.md` reflects the current release workflow and supported platforms.

## 3. CI

- [ ] `cargo fmt --all -- --check`
- [ ] `cargo clippy --all-targets --all-features -- -D warnings`
- [ ] `cargo test`
- [ ] `cargo build --release`
- [ ] GitHub Actions CI passes on Ubuntu, macOS, and Windows.
- [ ] CI does not require secrets for pull requests or normal pushes.

## 4. Install QA

- [ ] `scripts/install.sh --dry-run` prints the cargo install command without changing the system.
- [ ] Pressing Enter at the star prompt skips starring and continues.
- [ ] Pressing `y` runs `gh repo star NomaDamas/MinSync` when `gh` is installed and authenticated.
- [ ] Missing `gh` does not fail installation.
- [ ] `cargo install minsync` remains documented as the direct non-prompt path.

## 5. Agent Skill QA

- [ ] `skills/minsync/SKILL.md` has Vercel-compatible YAML frontmatter with `name` and `description`.
- [ ] The skill includes install, repository star, initialization, ignore-file, OpenAI, TEI, sync, query, watch, and troubleshooting instructions.
- [ ] The skill tells agents not to index binary files directly.
- [ ] The skill can be installed with:

```bash
npx skills add github:NomaDamas/MinSync/skills/minsync
```

## 6. Release Automation

- [ ] Push a signed or reviewed tag named `vX.Y.Z`.
- [ ] `.github/workflows/release.yml` builds:
  - `x86_64-unknown-linux-gnu`
  - `aarch64-apple-darwin`
  - `x86_64-pc-windows-msvc`
- [ ] GitHub Release artifacts are uploaded.
- [ ] crates.io publishing succeeds with `CARGO_REGISTRY_TOKEN`.
- [ ] Release notes mention text-only indexing and `.minsyncignore`.

## 7. Post-Release Smoke Test

Use a clean temp directory:

```bash
cargo install minsync
mkdir /tmp/minsync-smoke
cd /tmp/minsync-smoke
printf 'MinSync indexes changed text only.\n' > notes.md
minsync init --chunker cdc
minsync status
```

For OpenAI or TEI environments, also run:

```bash
minsync sync --full
minsync query "changed text"
```

## 8. Rollback

- [ ] If GitHub Release artifacts are bad, delete the broken release and tag after announcing the issue.
- [ ] If the crate is bad, yank the crates.io version with `cargo yank --vers X.Y.Z minsync`.
- [ ] Open a follow-up issue with the failing command, platform, and artifact name.
