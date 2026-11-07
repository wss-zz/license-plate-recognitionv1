# -*- coding: utf-8 -*-
"""把 REPORT_MAIN.md + REPORT_BODY.md 转成符合模板版式的 Word 文档。

用法: python md2docx.py
输出: report/车牌识别一体闸机系统-设计报告.docx
"""
from pathlib import Path
import re

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(r"D:\desktop\车辆闸机项目\runs_2026-11-05")
SRC = [ROOT / "report/REPORT_MAIN.md", ROOT / "report/REPORT_BODY.md"]
OUT = ROOT / "report/车牌识别一体闸机系统-设计报告.docx"

BODY_FONT = "宋体"
HEAD_FONT = "黑体"


def set_font(run, name=BODY_FONT, size=12, bold=False):
    run.font.name = name
    run.font.size = Pt(size)
    run.font.bold = bold
    run._element.rPr.rFonts.set(qn("w:eastAsia"), name)


def style_doc(doc):
    st = doc.styles["Normal"]
    st.font.name = BODY_FONT
    st.font.size = Pt(12)
    st._element.rPr.rFonts.set(qn("w:eastAsia"), BODY_FONT)
    pf = st.paragraph_format
    pf.line_spacing = 1.25
    pf.space_before = Pt(0)
    pf.space_after = Pt(0)
    for i, (size, name) in enumerate([(16, "Heading 1"), (14, "Heading 2"), (12, "Heading 3")]):
        s = doc.styles[name]
        s.font.name = HEAD_FONT
        s.font.size = Pt(size)
        s.font.bold = True
        s.font.color.rgb = RGBColor(0, 0, 0)
        s._element.rPr.rFonts.set(qn("w:eastAsia"), HEAD_FONT)
        s.paragraph_format.space_before = Pt(12 if i == 0 else 8)
        s.paragraph_format.space_after = Pt(8 if i == 0 else 6)
        s.paragraph_format.first_line_indent = Pt(0)
        s.paragraph_format.line_spacing = 1.25


def add_cover(doc):
    for _ in range(3):
        doc.add_paragraph()
    for text, size, bold, gap in [("电子信息工程专业科研训练", 22, True, 10),
                                  ("设 计 报 告", 26, True, 40),
                                  ("设计题目：车牌识别一体闸机系统", 18, True, 30)]:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.first_line_indent = Pt(0)
        p.paragraph_format.space_after = Pt(gap)
        set_font(p.add_run(text), HEAD_FONT, size, bold)
    for label in ["专    业：", "班    级：", "学    号：", "姓    名：", "指导教师："]:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.first_line_indent = Pt(0)
        p.paragraph_format.space_before = Pt(8)
        set_font(p.add_run(label + " " * 18), BODY_FONT, 14, False)
    doc.add_paragraph()
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.first_line_indent = Pt(0)
    set_font(p.add_run("2026 年 11 月"), BODY_FONT, 14, False)
    doc.add_page_break()


def add_caption(doc, text, keep=True):
    p = doc.add_paragraph()
    p.paragraph_format.keep_with_next = keep
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.first_line_indent = Pt(0)
    p.paragraph_format.space_before = Pt(3)
    p.paragraph_format.space_after = Pt(6)
    set_font(p.add_run(text), BODY_FONT, 10.5, False)


def add_picture(doc, rel, cap):
    path = (ROOT / rel).resolve()
    if not path.exists():
        add_caption(doc, f"[缺图] {rel}")
        return
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.first_line_indent = Pt(0)
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(2)
    p.add_run().add_picture(str(path), width=Cm(13.6))
    add_caption(doc, cap)


def _widths(rows, total_cm=15.4):
    ncol = len(rows[0])
    w = []
    for j in range(ncol):
        need = max(4, min(26, max(len(r[j]) for r in rows)))
        w.append(need)
    s = sum(w)
    lo = min(1.9, total_cm / ncol * 0.85)
    return [min(6.2, max(lo, total_cm * x / s)) for x in w]


def add_table(doc, rows):
    rows = [[str(c).replace("**", "").replace("`", "") for c in r] for r in rows]
    ncol = len(rows[0])
    t = doc.add_table(rows=len(rows), cols=ncol)
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    widths = _widths(rows)
    for j, cw in enumerate(widths):
        t.columns[j].width = Cm(cw)
    for r in t.rows:
        r.allow_break_across_pages = False
    # 表头行跨页重复
    trPr = t.rows[0]._tr.get_or_add_trPr()
    tblHeader = trPr.makeelement(qn("w:tblHeader"), {})
    trPr.append(tblHeader)
    for i, row in enumerate(rows):
        for j, cell in enumerate(row):
            c = t.cell(i, j)
            c.text = ""
            p = c.paragraphs[0]
            p.paragraph_format.first_line_indent = Pt(0)
            p.paragraph_format.line_spacing = 1.0
            short = max(len(r[j]) for r in rows) <= 10
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if (i == 0 or (j > 0 and short)) else WD_ALIGN_PARAGRAPH.LEFT
            set_font(p.add_run(cell), BODY_FONT, 8.5, i == 0)
            c.width = Cm(widths[j])
            if len(rows) <= 10 and i < len(rows) - 1:
                p.paragraph_format.keep_with_next = True
    # 表后留白
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def parse_table(lines, idx):
    rows = []
    while idx < len(lines) and lines[idx].strip().startswith("|"):
        raw = lines[idx].strip().strip("|")
        cells = [c.strip() for c in raw.split("|")]
        if not all(re.fullmatch(r":?-{2,}:?", c) for c in cells):
            rows.append(cells)
        idx += 1
    return rows, idx


def build():
    doc = Document()
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21), Cm(29.7)
    sec.top_margin = sec.bottom_margin = Cm(2.5)
    sec.left_margin = sec.right_margin = Cm(2.6)
    style_doc(doc)
    add_cover(doc)

    lines = []
    for f in SRC:
        lines += f.read_text(encoding="utf-8").splitlines() + [""]

    i = 0
    while i < len(lines):
        ln = lines[i].rstrip()
        s = ln.strip()
        if not s:
            i += 1
            continue
        if s.startswith("!fig|"):
            _, rel, cap = s.split("|", 2)
            add_picture(doc, rel, cap)
            i += 1
            continue
        if s.startswith("!tab|"):
            cap = s.split("|", 1)[1]
            rows, j = parse_table(lines, i + 1)
            add_caption(doc, cap)
            add_table(doc, rows)
            i = j
            continue
        if s.startswith("### "):
            doc.add_heading(s[4:], level=3)
        elif s.startswith("## "):
            doc.add_heading(s[3:], level=2)
        elif s.startswith("# "):
            doc.add_heading(s[2:], level=1)
        elif s.startswith(">"):
            body = s[1:].strip()
            while i + 1 < len(lines):
                nxt = lines[i + 1].strip()
                if not nxt or nxt[0] in "#|!":
                    break
                if nxt.startswith(">"):
                    nxt = nxt[1:].strip()
                body += nxt
                i += 1
            p = doc.add_paragraph()
            p.paragraph_format.first_line_indent = Pt(0)
            p.paragraph_format.left_indent = Pt(24)
            p.paragraph_format.line_spacing = 1.15
            for k, chunk in enumerate(re.split(r"\*\*", body.replace("`", ""))):
                if chunk:
                    set_font(p.add_run(chunk), BODY_FONT, 10.5, bold=(k % 2 == 1))
        elif s.startswith("- "):
            p = doc.add_paragraph(style="List Bullet")
            p.paragraph_format.first_line_indent = Pt(0)
            p.paragraph_format.line_spacing = 1.25
            set_font(p.add_run(s[2:]), BODY_FONT, 12)
        elif s.startswith("|"):
            rows, j = parse_table(lines, i)
            if rows:
                add_table(doc, rows)
            i = j
            continue
        else:
            # 续行合并：直到遇到空行或下一个块级标记
            while i + 1 < len(lines):
                nxt = lines[i + 1].strip()
                if not nxt or nxt[0] in "#-|!>" or (nxt[:2].rstrip(".").isdigit() and nxt[1:3] == ". "):
                    break
                s += nxt
                i += 1
            p = doc.add_paragraph()
            p.paragraph_format.first_line_indent = Pt(24)
            p.paragraph_format.line_spacing = 1.25
            p.alignment = (WD_ALIGN_PARAGRAPH.LEFT if "http" in s else WD_ALIGN_PARAGRAPH.JUSTIFY)
            # **bold** 片段
            s = s.replace("`", "")
            for seg in re.split(r"(<sup>.*?</sup>)", s):
                if not seg:
                    continue
                if seg.startswith("<sup>"):
                    r = p.add_run(seg[5:-6])
                    set_font(r, BODY_FONT, 12, False)
                    r.font.superscript = True
                    continue
                for k, chunk in enumerate(re.split(r"\*\*", seg)):
                    if not chunk:
                        continue
                    set_font(p.add_run(chunk), BODY_FONT, 12, bold=(k % 2 == 1))
        i += 1

    doc.save(OUT)
    print("saved:", OUT)


if __name__ == "__main__":
    build()
