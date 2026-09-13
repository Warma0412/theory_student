"""Check generated PDF text and page bounds, and render contact sheets."""

from pathlib import Path
import json
import re

import pymupdf
from PIL import Image, ImageDraw

from common import ROOT, save_json, digest_file

PROJECT = ROOT.parents[1]
RENDER = ROOT / "results/pdf_checks"
RENDER.mkdir(exist_ok=True)


def inspect(path, name):
    doc = pymupdf.open(path)
    text = "\n".join(page.get_text() for page in doc)
    errors = []
    for number, page in enumerate(doc):
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                for span in line["spans"]:
                    x0, y0, x1, y1 = span["bbox"]
                    if span["text"].strip() and (
                        x0 < -1 or y0 < -1 or x1 > page.rect.width + 1 or y1 > page.rect.height + 1
                    ):
                        errors.append({"page": number + 1, "text": span["text"], "bbox": span["bbox"]})
    assert "@@" not in text, path
    assert "??" not in text, path
    assert "吴优" in text if name in ("thesis", "report", "slides") else True
    assert "1.8443" in text and "1.8625" in text, path
    if name == "slides":
        assert len(doc) == 20
        compact = re.sub(r"\s+", "", text)
        for expected in ("共60个半样本", "共100个外层扰动", "不把60个半样本视为独立",
                         "17.8784%", "后续验证"):
            assert expected in compact, expected
    if name == "brief":
        assert 4 <= len(doc) <= 5
    if name in ("thesis", "report"):
        assert "2025213385" in text and "张吕欧" in text
    cols, width = (4, 420) if name == "slides" else (4, 280)
    thumbnails = []
    for number, page in enumerate(doc):
        scale = width / page.rect.width
        pix = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False)
        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        tile = Image.new("RGB", (width + 12, pix.height + 30), "#e5e5e5")
        tile.paste(image, (6, 22))
        ImageDraw.Draw(tile).text((8, 5), f"{name}  p.{number+1}", fill="#222222")
        thumbnails.append(tile)
    height = max(t.height for t in thumbnails)
    for part in range((len(thumbnails) + 23) // 24):
        current = thumbnails[part * 24:(part + 1) * 24]
        sheet = Image.new("RGB", (cols * (width + 12), height * ((len(current) + cols - 1) // cols)), "white")
        for i, tile in enumerate(current):
            sheet.paste(tile, ((i % cols) * (width + 12), (i // cols) * height))
        sheet.save(RENDER / f"{name}_contact_{part+1}.png")
    for number, page in enumerate(doc):
        words = page.get_text()
        if number == 0 or any(s in words for s in (
            "同样12项", "12项预算下的统一", "Top-12名单的样本稳定性",
            "六场景Top-12", "原生选择规则的平均", "31项候选指标的范围",
        )):
            page.get_pixmap(matrix=pymupdf.Matrix(1.4, 1.4), alpha=False).save(
                RENDER / f"{name}_page_{number+1:02d}.png")
    return {"path": str(path), "pages": len(doc), "sha256": digest_file(path),
            "characters": len(text), "out_of_page_spans": errors}


def main():
    files = {
        "thesis": ROOT / "latex工程/论文/论文.pdf",
        "report": ROOT / "latex工程/开题报告/开题报告.pdf",
        "slides": ROOT / "latex工程/开题汇报/开题汇报.pdf",
        "condensed": ROOT / "论文_v7_精简版.pdf",
        "brief": ROOT / "论文_v7_极限精简版.pdf",
    }
    reports = {name: inspect(path, name) for name, path in files.items()}
    final = PROJECT / "【最终版】/v7"
    assert sorted(p.name for p in final.iterdir()) == sorted(
        name + ext for name in ("论文", "开题报告", "开题汇报") for ext in (".tex", ".pdf"))
    for name in ("论文", "开题报告", "开题汇报"):
        assert digest_file(final / f"{name}.pdf") == digest_file(ROOT / f"latex工程/{name}/{name}.pdf")
    save_json(ROOT / "results/pdf_validation.json", reports)
    print(json.dumps({name: {"pages": r["pages"], "out_of_page_spans": len(r["out_of_page_spans"])}
                      for name, r in reports.items()}, indent=2))
    assert not any(r["out_of_page_spans"] for r in reports.values()), "Inspect out-of-page spans"


if __name__ == "__main__":
    main()
