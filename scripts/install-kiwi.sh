#!/usr/bin/env bash
set -euo pipefail

version="${KIWI_VERSION:-0.23.2}"
prefix="${KIWI_PREFIX:-$HOME/.local/kiwi}"
case "$(uname -s):$(uname -m)" in
  Darwin:arm64) asset="kiwi_mac_arm64_v${version}.tgz" ;;
  Darwin:x86_64) asset="kiwi_mac_x86_64_v${version}.tgz" ;;
  Linux:x86_64) asset="kiwi_lnx_x86_64_v${version}.tgz" ;;
  Linux:aarch64|Linux:arm64) asset="kiwi_lnx_aarch64_v${version}.tgz" ;;
  *) echo "unsupported Kiwi platform: $(uname -s) $(uname -m)" >&2; exit 1 ;;
esac

base="https://github.com/bab2min/Kiwi/releases/download/v${version}"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
mkdir -p "$prefix"
curl -fsSL "$base/$asset" | tar -xz -C "$prefix"
curl -fsSL "$base/kiwi_model_v${version}_base.tgz" | tar -xz -C "$prefix"

if [[ "$(uname -s)" == Darwin ]]; then
  library="$prefix/lib/libkiwi.dylib"
else
  library="$prefix/lib/libkiwi.so"
fi
printf 'KIWI_LIBRARY_PATH=%s\nKIWI_MODEL_PATH=%s\n' \
  "$library" "$prefix/models/cong/base"
