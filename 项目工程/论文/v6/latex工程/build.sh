#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
FINAL_DIR="$(cd "$ROOT/../../.." && pwd)/【最终版】/v6"

(
  cd "$ROOT/论文"
  tectonic --keep-logs 论文.tex
)

(
  cd "$ROOT/开题汇报"
  tectonic --keep-logs 开题汇报.tex
)

(
  cd "$ROOT/开题报告"
  tectonic --keep-logs 开题报告.tex
)

mkdir -p "$FINAL_DIR"
cp -p "$ROOT/论文/论文.tex" "$ROOT/论文/论文.pdf" "$FINAL_DIR/"
cp -p "$ROOT/开题报告/开题报告.tex" "$ROOT/开题报告/开题报告.pdf" "$FINAL_DIR/"
cp -p "$ROOT/开题汇报/开题汇报.tex" "$ROOT/开题汇报/开题汇报.pdf" "$FINAL_DIR/"

echo "Generated:"
echo "  $FINAL_DIR/论文.tex"
echo "  $FINAL_DIR/论文.pdf"
echo "  $FINAL_DIR/开题报告.tex"
echo "  $FINAL_DIR/开题报告.pdf"
echo "  $FINAL_DIR/开题汇报.tex"
echo "  $FINAL_DIR/开题汇报.pdf"
