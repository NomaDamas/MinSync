#!/usr/bin/env sh
set -eu

REPO="NomaDamas/MinSync"
CRATE="minsync"
DRY_RUN=0
YES_STAR=0
NO_STAR=0

usage() {
  cat <<'USAGE'
Usage: install.sh [--dry-run] [--yes-star] [--no-star]

Installs MinSync with cargo install minsync.

Options:
  --dry-run    Print the install command without running it.
  --yes-star   Star NomaDamas/MinSync with gh repo star when possible.
  --no-star    Skip the star prompt.
  -h, --help   Show this help.
USAGE
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      ;;
    --yes-star)
      YES_STAR=1
      ;;
    --no-star)
      NO_STAR=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

star_repo() {
  if ! command -v gh >/dev/null 2>&1; then
    echo "GitHub CLI not found; skipping optional repo star."
    return 0
  fi

  if [ "$DRY_RUN" -eq 1 ]; then
    echo "dry-run: gh repo star $REPO"
    return 0
  fi

  if gh repo star "$REPO"; then
    echo "Starred github.com/$REPO."
  else
    echo "Could not star github.com/$REPO; continuing install."
  fi
}

if [ "$YES_STAR" -eq 1 ] && [ "$NO_STAR" -eq 1 ]; then
  echo "--yes-star and --no-star cannot be used together" >&2
  exit 2
fi

if [ "$YES_STAR" -eq 1 ]; then
  star_repo
elif [ "$NO_STAR" -eq 0 ]; then
  printf "Star github.com/%s with 'gh repo star' before installing? [y/N] " "$REPO"
  IFS= read -r answer || answer=""
  case "$answer" in
    y|Y|yes|YES|Yes)
      star_repo
      ;;
    *)
      echo "Skipping repo star."
      ;;
  esac
else
  echo "Skipping repo star."
fi

if ! command -v cargo >/dev/null 2>&1; then
  echo "cargo is required. Install Rust from https://rustup.rs/ and rerun this script." >&2
  exit 1
fi

if [ "$DRY_RUN" -eq 1 ]; then
  echo "dry-run: cargo install $CRATE"
else
  cargo install "$CRATE"
fi
