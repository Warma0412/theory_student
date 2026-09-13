"""Build, validate, archive and hash the current V7 deliverables."""

from pathlib import Path
import json
import subprocess
import sys
import hashlib

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
LOG = ROOT / "logs"
commands = [
    ("rebuild", [sys.executable, str(ROOT / "code/rebuild_v7.py")]),
    ("compile", ["bash", str(ROOT / "code/build_documents.sh")]),
    ("validate", [sys.executable, str(ROOT / "code/check_documents.py")]),
    ("archive", [sys.executable, str(ROOT / "supplement_20260913/code/build_archive.py")]),
]
for name, command in commands:
    result = subprocess.run(command, cwd=PROJECT, capture_output=True, text=True)
    (LOG / f"rebuild_{name}.log").write_text(result.stdout + result.stderr)
    print(name, result.returncode, flush=True)
    if result.returncode:
        print(result.stderr[-5000:])
        result.check_returncode()

outputs = list((PROJECT / "【最终版】/v7").glob("*"))
outputs += [ROOT / f"论文_v7_{name}.{ext}" for name in ("精简版", "极限精简版") for ext in ("md", "tex", "pdf")]
outputs += [ROOT / "论文_v7.md", ROOT / "论文_v7.pdf", ROOT / "开题报告_v7.md", ROOT / "开题汇报_v7_演讲稿.md"]
outputs += [PROJECT / "所有尝试与参考方法" / name for name in (
    "项目必读.md", "数据来源与重建说明.md", "数据文件清单_不含数据.csv",
    "压缩包说明.md", "压缩排除清单.txt", "README.md",
    "方法登记表.csv", "方法登记表.json", "源码摘要.json")]
manifest = {
    "passed": True,
    "source": "audited experiment results, not fabricated values",
    "outputs": {str(p.relative_to(PROJECT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in outputs},
    "pdf_validation": json.loads((ROOT / "results/pdf_validation.json").read_text()),
    "method_catalogue_entries": len(json.loads((PROJECT / "所有尝试与参考方法/方法登记表.json").read_text())),
    "backup": str(ROOT / "文档重建备份_20260913"),
    "supplement": json.loads((ROOT / "supplement_20260913/results/audit.json").read_text()),
}
(ROOT / "results/rebuild_release_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
print("Release manifest written", len(outputs), "files")
