#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PKG_NAME="TileMEM_TilePO_V0_1_20260611"
PKG_DIR="publish/$PKG_NAME"
TARBALL="publish/$PKG_NAME.tar.gz"

rm -rf "$PKG_DIR" "$TARBALL" "$TARBALL.sha256"
mkdir -p "$PKG_DIR"

rsync -a \
  .gitattributes README.md PRIORITY_DISCLOSURE.md CLAIM_CHECKLIST.md RELEASE_NOTES.md \
  CITATION.cff LICENSE Makefile pyproject.toml CMakeLists.txt REPRODUCE.md \
  "$PKG_DIR/"

for dir in SKILL tilemem tilepo tools configs docs evidence paper scripts examples kernels include src tests; do
  if [[ -e "$dir" ]]; then
    rsync -a --exclude='__pycache__/' --exclude='*.pyc' --exclude='superpowers/' "$dir" "$PKG_DIR/"
  fi
done

find "$PKG_DIR" -type f ! -name SHA256SUMS -print0 \
  | sort -z \
  | xargs -0 sha256sum \
  | sed "s#  $PKG_DIR/#  #" \
  > "$PKG_DIR/SHA256SUMS"

tar -C publish -czf "$TARBALL" "$PKG_NAME"
(cd publish && sha256sum "$PKG_NAME.tar.gz" > "$PKG_NAME.tar.gz.sha256")

echo "$TARBALL"
echo "$TARBALL.sha256"
