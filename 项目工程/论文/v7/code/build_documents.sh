#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PROJECT="$(cd "$ROOT/../.." && pwd)"
FINAL="$PROJECT/【最终版】/v7"

for name in 论文 开题报告 开题汇报; do
  (
    cd "$ROOT/latex工程/$name"
    tectonic --keep-logs "$name.tex"
  )
done
(
  cd "$ROOT"
  tectonic --keep-logs "论文_v7_精简版.tex"
  tectonic --keep-logs "论文_v7_极限精简版.tex"
)
mkdir -p "$FINAL"
for name in 论文 开题报告 开题汇报; do
  cp -p "$ROOT/latex工程/$name/$name.tex" "$ROOT/latex工程/$name/$name.pdf" "$FINAL/"
done
cp -p "$ROOT/latex工程/论文/论文.pdf" "$ROOT/论文_v7.pdf"
printf 'Documents built in %s\n' "$FINAL"
