"""Build the project report as a Word document in BML Munjal University's report format.

The Markdown source lives in ``docs/report/bmu/`` (see ``00-conventions.md`` there). This script reads
``abstract.md``, ``part1.md``, ``part2.md`` and ``part3.md`` and writes a ``.docx`` with the template's layout:
the title page, the candidate's and the supervisor's declarations, the acknowledgement, the table of contents,
the abstract, the lists of figures, tables and abbreviations, six numbered chapters, the bibliography (IEEE
style), the appendices and the plagiarism-check page. Only the standard library and Pillow are used.

Page numbers and the table of contents are Word fields. They hold placeholders until Word updates them;
``scripts/finish_bmu_report.ps1`` opens the file in Word, updates every field and saves it.

    ./.venv/Scripts/python.exe scripts/build_bmu_report.py --out docs/report/Nightingale-Project-Report-BMU.docx
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
import sys
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from xml.sax.saxutils import escape

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
BMU = ROOT / "docs" / "report" / "bmu"
LOGO = ROOT / "docs" / "report" / "bmu" / "img" / "bmu_logo.png"

PROJECT_TITLE = (
    "Nightingale — Clinical Intelligence & Reasoning System: "
    "A Red-Flag-Aware Decision-Support Prototype for Acute Chest Pain"
)
TEXT_WIDTH = 9045  # twips, A4 less the margins below
PAGE_MAR = dict(top=1700, right=860, bottom=1300, left=2015, header=720, footer=600)
TIMES = "Times New Roman"
SANS = "Calibri"
W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
NSDECL = (
    f'xmlns:w="{W_NS}" xmlns:r="{R_NS}" '
    'xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing" '
    'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
    'xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture"'
)

PLACEHOLDER_NOTE = "highlighted text in [[brackets]] is for the team to fill in or resolve"


# --------------------------------------------------------------------------------------------------
# Inline text -> runs
# --------------------------------------------------------------------------------------------------
@dataclass
class Ctx:
    """State shared while the document is assembled."""

    cites: list[str] = field(default_factory=list)
    refs: dict[str, str] = field(default_factory=dict)
    media: list[tuple[str, bytes]] = field(default_factory=list)  # (name, bytes)
    rels: list[tuple[str, str]] = field(default_factory=list)  # (rId, target)
    bm_id: int = 0
    draw_id: int = 0
    num_instances: list[int] = field(default_factory=list)
    figures: list[tuple[str, str, str]] = field(default_factory=list)  # (label, text, bookmark)
    tables: list[tuple[str, str, str]] = field(default_factory=list)
    abbreviations: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    chapter: int = 0

    def cite(self, key: str) -> int:
        if key not in self.refs:
            self.warnings.append(f"unknown citation key: {key}")
            return 0
        if key not in self.cites:
            self.cites.append(key)
        return self.cites.index(key) + 1


def rpr(
    bold: bool = False,
    italic: bool = False,
    size: float | None = None,
    font: str | None = None,
    highlight: str | None = None,
    sup: bool = False,
    sub: bool = False,
    color: str | None = None,
) -> str:
    parts = []
    if font:
        parts.append(f'<w:rFonts w:ascii="{font}" w:hAnsi="{font}" w:cs="{font}"/>')
    if bold:
        parts.append("<w:b/><w:bCs/>")
    if italic:
        parts.append("<w:i/><w:iCs/>")
    if color:
        parts.append(f'<w:color w:val="{color}"/>')
    if size:
        half = int(round(size * 2))
        parts.append(f'<w:sz w:val="{half}"/><w:szCs w:val="{half}"/>')
    if highlight:
        parts.append(f'<w:highlight w:val="{highlight}"/>')
    if sup:
        parts.append('<w:vertAlign w:val="superscript"/>')
    if sub:
        parts.append('<w:vertAlign w:val="subscript"/>')
    return f"<w:rPr>{''.join(parts)}</w:rPr>" if parts else ""


def run(text: str, **props) -> str:
    if text == "":
        return ""
    return f'<w:r>{rpr(**props)}<w:t xml:space="preserve">{escape(text)}</w:t></w:r>'


INLINE = re.compile(
    r"(?P<ph>\[\[(?P<phtext>.+?)\]\])"
    r"|(?P<cite>\[@(?P<keys>[^\]]+)\])"
    r"|(?P<bold>\*\*(?P<boldtext>.+?)\*\*)"
    r"|(?P<ital>(?<![\*\w])\*(?P<italtext>[^\s*][^*]*?)\*(?![\*\w]))"
    r"|(?P<code>`(?P<codetext>[^`]+)`)"
    r"|(?P<sup>\^(?P<suptext>[^\^\s]+)\^)"
    r"|(?P<sub>~(?P<subtext>[^~\s]+)~)"
)


def inline(text: str, ctx: Ctx, **base) -> str:
    """Convert inline Markdown to a string of runs."""
    out = []
    pos = 0
    for m in INLINE.finditer(text):
        out.append(run(text[pos : m.start()], **base))
        pos = m.end()
        if m.group("ph"):
            out.append(run("[" + m.group("phtext") + "]", **{**base, "highlight": "yellow"}))
        elif m.group("cite"):
            nums = []
            for key in re.split(r"[;,]\s*", m.group("keys")):
                key = key.strip().lstrip("@")
                if key:
                    n = ctx.cite(key)
                    nums.append(f"[{n}]" if n else f"[?{key}]")
            out.append(run(", ".join(nums), **base))
        elif m.group("bold"):
            out.append(inline(m.group("boldtext"), ctx, **{**base, "bold": True}))
        elif m.group("ital"):
            out.append(inline(m.group("italtext"), ctx, **{**base, "italic": True}))
        elif m.group("code"):
            out.append(
                run(
                    m.group("codetext"),
                    **{**base, "font": "Consolas", "size": (base.get("size") or 12) - 1.5},
                )
            )
        elif m.group("sup"):
            out.append(run(m.group("suptext"), **{**base, "sup": True}))
        elif m.group("sub"):
            out.append(run(m.group("subtext"), **{**base, "sub": True}))
    out.append(run(text[pos:], **base))
    return "".join(out)


# --------------------------------------------------------------------------------------------------
# Block builders
# --------------------------------------------------------------------------------------------------
def para(
    content: str,
    style: str | None = None,
    jc: str | None = None,
    before: int | None = None,
    after: int | None = None,
    line: int | None = None,
    ind_left: int | None = None,
    hanging: int | None = None,
    keep_next: bool = False,
    keep_lines: bool = False,
    page_break_before: bool = False,
    border_bottom: bool = False,
    shade: str | None = None,
    tabs: list[tuple[str, int, str | None]] | None = None,
    sect: str | None = None,
) -> str:
    p = []
    if style:
        p.append(f'<w:pStyle w:val="{style}"/>')
    if keep_next:
        p.append("<w:keepNext/>")
    if keep_lines:
        p.append("<w:keepLines/>")
    if page_break_before:
        p.append("<w:pageBreakBefore/>")
    if border_bottom:
        p.append(
            '<w:pBdr><w:bottom w:val="single" w:sz="8" w:space="4" w:color="000000"/></w:pBdr>'
        )
    if shade:
        p.append(f'<w:shd w:val="clear" w:color="auto" w:fill="{shade}"/>')
    if tabs:
        t = "".join(
            f'<w:tab w:val="{kind}" w:pos="{pos}"' + (f' w:leader="{lead}"' if lead else "") + "/>"
            for kind, pos, lead in tabs
        )
        p.append(f"<w:tabs>{t}</w:tabs>")
    sp = []
    if before is not None:
        sp.append(f'w:before="{before}"')
    if after is not None:
        sp.append(f'w:after="{after}"')
    if line is not None:
        sp.append(f'w:line="{line}" w:lineRule="auto"')
    if sp:
        p.append(f"<w:spacing {' '.join(sp)}/>")
    if ind_left is not None or hanging is not None:
        a = f' w:left="{ind_left or 0}"'
        if hanging:
            a += f' w:hanging="{hanging}"'
        p.append(f"<w:ind{a}/>")
    if jc:
        p.append(f'<w:jc w:val="{jc}"/>')
    if sect:
        p.append(sect)
    ppr = f"<w:pPr>{''.join(p)}</w:pPr>" if p else ""
    return f"<w:p>{ppr}{content}</w:p>"


def bookmark(ctx: Ctx, name: str, inner: str) -> str:
    ctx.bm_id += 1
    return f'<w:bookmarkStart w:id="{ctx.bm_id}" w:name="{name}"/>{inner}<w:bookmarkEnd w:id="{ctx.bm_id}"/>'


def field_xml(instr: str, cached: str, **props) -> str:
    rp = rpr(**props)
    return (
        f'<w:r>{rp}<w:fldChar w:fldCharType="begin"/></w:r>'
        f'<w:r>{rp}<w:instrText xml:space="preserve"> {instr} </w:instrText></w:r>'
        f'<w:r>{rp}<w:fldChar w:fldCharType="separate"/></w:r>'
        f"<w:r>{rp}<w:t>{escape(cached)}</w:t></w:r>"
        f'<w:r>{rp}<w:fldChar w:fldCharType="end"/></w:r>'
    )


def heading1_chapter(num: int, title: str) -> str:
    label = run(f"Chapter {num}", font=SANS, bold=True, size=17)
    body = run(title, font=TIMES, bold=True, size=17)
    return para(
        label + "<w:r><w:br/></w:r>" + body,
        style="Heading1",
        page_break_before=True,
        keep_next=True,
    )


def heading1_plain(title: str) -> str:
    return para(
        run(title, font=SANS, bold=True, size=17),
        style="Heading1",
        page_break_before=True,
        keep_next=True,
    )


def heading2(text: str, ctx: Ctx) -> str:
    m = re.match(r"^(\S+)\s+(.*)$", text)
    num, title = (
        (m.group(1), m.group(2)) if m and re.match(r"^[\dA-Z][\d.]*$", m.group(1)) else ("", text)
    )
    content = (run(num, font=TIMES, bold=True) + "<w:r><w:tab/></w:r>" if num else "") + inline(
        title, ctx, bold=True, font=TIMES
    )
    return para(content, style="Heading2", keep_next=True)


def heading3(text: str, ctx: Ctx) -> str:
    m = re.match(r"^(\d[\d.]*)\s+(.*)$", text)
    num, title = (m.group(1), m.group(2)) if m else ("", text)
    content = (
        run(num, font=TIMES, bold=True, italic=True) + "<w:r><w:tab/></w:r>" if num else ""
    ) + inline(title, ctx, bold=True, italic=True, font=TIMES)
    return para(content, style="Heading3", keep_next=True)


def body_para(text: str, ctx: Ctx) -> str:
    return para(inline(text, ctx, font=TIMES), style="BodyText", jc="both")


def list_block(items: list[tuple[int, str]], ordered: bool, ctx: Ctx) -> str:
    if ordered:
        num_id = 10 + len(ctx.num_instances)
        ctx.num_instances.append(num_id)
    else:
        num_id = 1
    out = []
    for level, text in items:
        ppr = f'<w:numPr><w:ilvl w:val="{min(level, 2)}"/><w:numId w:val="{num_id}"/></w:numPr>'
        out.append(
            f'<w:p><w:pPr><w:pStyle w:val="ListParagraph"/>{ppr}<w:spacing w:after="60" w:line="360" w:lineRule="auto"/>'
            f'<w:jc w:val="both"/></w:pPr>{inline(text, ctx, font=TIMES)}</w:p>'
        )
    return "".join(out)


def code_block(text: str) -> str:
    lines = text.rstrip("\n").split("\n")
    runs = []
    for i, line in enumerate(lines):
        if i:
            runs.append("<w:r><w:br/></w:r>")
        runs.append(run(line, font="Consolas", size=8.5))
    return para("".join(runs), style="CodeBlock", shade="F2F2F2", keep_lines=len(lines) <= 30)


def split_row(line: str) -> list[str]:
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|"):
        line = line[:-1]
    cells, cur, esc_next = [], "", False
    for ch in line:
        if esc_next:
            cur += ch
            esc_next = False
        elif ch == "\\":
            esc_next = True
        elif ch == "|":
            cells.append(cur.strip())
            cur = ""
        else:
            cur += ch
    cells.append(cur.strip())
    return cells


def column_widths(rows: list[list[str]], total: int) -> list[int]:
    n = max(len(r) for r in rows)
    weights = []
    for c in range(n):
        lens = [len(re.sub(r"[*`^~\[\]@]", "", r[c])) for r in rows if c < len(r)]
        longest = max(lens) if lens else 4
        mean = sum(lens) / max(len(lens), 1)
        weights.append(max(6.0, min(46.0, 0.55 * longest + 0.45 * mean)))
    scale = total / sum(weights)
    widths = [max(700, int(w * scale)) for w in weights]
    excess = sum(widths) - total
    if excess > 0:
        widths[widths.index(max(widths))] -= excess
    return widths


def table_block(rows: list[list[str]], ctx: Ctx) -> str:
    n = max(len(r) for r in rows)
    rows = [r + [""] * (n - len(r)) for r in rows]
    widths = column_widths(rows, TEXT_WIDTH)
    size = 9.5 if n >= 5 else 10
    grid = "".join(f'<w:gridCol w:w="{w}"/>' for w in widths)
    borders = "".join(
        f'<w:{side} w:val="single" w:sz="4" w:space="0" w:color="7F7F7F"/>'
        for side in ("top", "left", "bottom", "right", "insideH", "insideV")
    )
    out = [
        f'<w:tbl><w:tblPr><w:tblW w:w="{TEXT_WIDTH}" w:type="dxa"/><w:tblBorders>{borders}</w:tblBorders>'
        '<w:tblLayout w:type="fixed"/><w:tblCellMar><w:top w:w="30" w:type="dxa"/><w:left w:w="70" w:type="dxa"/>'
        '<w:bottom w:w="30" w:type="dxa"/><w:right w:w="70" w:type="dxa"/></w:tblCellMar></w:tblPr>'
        f"<w:tblGrid>{grid}</w:tblGrid>"
    ]
    for ri, row in enumerate(rows):
        header = ri == 0
        trpr = "<w:trPr><w:cantSplit/>" + ("<w:tblHeader/>" if header else "") + "</w:trPr>"
        cells = []
        for ci, text in enumerate(row):
            shade = '<w:shd w:val="clear" w:color="auto" w:fill="E7E6E6"/>' if header else ""
            content = inline(text, ctx, font=TIMES, size=size, bold=header)
            cells.append(
                f'<w:tc><w:tcPr><w:tcW w:w="{widths[ci]}" w:type="dxa"/>{shade}</w:tcPr>'
                f'<w:p><w:pPr><w:pStyle w:val="TableText"/>{"<w:keepNext/>" if ri < len(rows) - 1 and len(rows) <= 8 else ""}'
                f"</w:pPr>{content}</w:p></w:tc>"
            )
        out.append(f"<w:tr>{trpr}{''.join(cells)}</w:tr>")
    out.append("</w:tbl>")
    out.append(para("", after=60, line=240))
    return "".join(out)


def caption_para(label: str, text: str, ctx: Ctx, kind: str, jc: str | None) -> str:
    safe = re.sub(r"[^A-Za-z0-9]+", "_", label)
    name = f"{kind}_{safe}"
    registry = ctx.figures if kind == "fig" else ctx.tables
    registry.append((label, re.sub(r"[`*^~]", "", text), name))
    inner = run(label + ". ", font=TIMES, bold=True, size=11) + inline(
        text, ctx, font=TIMES, size=11
    )
    return para(
        bookmark(ctx, name, inner),
        style="Caption",
        jc=jc,
        keep_next=(kind == "tab"),
    )


def image_block(path: Path, ctx: Ctx, max_w_cm: float = 15.0, max_h_cm: float = 19.5) -> str | None:
    if not path.exists():
        return None
    with Image.open(path) as im:
        w_px, h_px = im.size
    w_cm = min(max_w_cm, 15.0)
    h_cm = w_cm * h_px / w_px
    if h_cm > max_h_cm:
        h_cm = max_h_cm
        w_cm = h_cm * w_px / h_px
    cx, cy = int(w_cm * 360000), int(h_cm * 360000)
    name = f"image{len(ctx.media) + 1}.png"
    ctx.media.append((name, path.read_bytes()))
    rid = f"rIdImg{len(ctx.media)}"
    ctx.rels.append((rid, f"media/{name}"))
    ctx.draw_id += 1
    drawing = (
        f'<w:r><w:drawing><wp:inline distT="0" distB="0" distL="0" distR="0"><wp:extent cx="{cx}" cy="{cy}"/>'
        f'<wp:docPr id="{ctx.draw_id}" name="Picture {ctx.draw_id}" descr="{escape(path.stem)}"/>'
        '<wp:cNvGraphicFramePr><a:graphicFrameLocks noChangeAspect="1"/></wp:cNvGraphicFramePr>'
        '<a:graphic><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture"><pic:pic>'
        f'<pic:nvPicPr><pic:cNvPr id="{ctx.draw_id}" name="{escape(name)}"/><pic:cNvPicPr/></pic:nvPicPr>'
        f'<pic:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>'
        f'<pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm>'
        '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr></pic:pic></a:graphicData></a:graphic>'
        "</wp:inline></w:drawing></w:r>"
    )
    return para(drawing, jc="center", keep_next=True, before=120, after=60)


# --------------------------------------------------------------------------------------------------
# Markdown -> blocks -> XML
# --------------------------------------------------------------------------------------------------
LIST_RE = re.compile(r"^(\s*)([-*]|\d+\.)\s+(.*)$")
TABLE_CAP = re.compile(r"^\*\*(Table \d+\.\d+)\.\*\*\s+(.*)$")
FIG_RE = re.compile(r"^!\[(?P<alt>.*)\]\((?P<path>[^)]+)\)\s*$")


def render_markdown(md: str, ctx: Ctx, base: Path) -> list[str]:
    lines = md.replace("\r\n", "\n").split("\n")
    out: list[str] = []
    i = 0
    para_buf: list[str] = []
    pending_caption: tuple[str, str] | None = None

    def flush_para() -> None:
        nonlocal para_buf, pending_caption
        if para_buf:
            text = " ".join(s.strip() for s in para_buf).strip()
            para_buf = []
            m = TABLE_CAP.match(text)
            if m:
                pending_caption = (m.group(1), m.group(2))
                return
            if pending_caption:
                out.append(caption_para(*pending_caption, ctx, "tab", None))
                pending_caption = None
            out.append(body_para(text, ctx))

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if stripped.startswith("<!--"):
            while i < len(lines) and "-->" not in lines[i]:
                i += 1
            i += 1
            continue
        if stripped.startswith("```"):
            flush_para()
            lang = stripped[3:].strip()
            i += 1
            buf = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                buf.append(lines[i])
                i += 1
            i += 1
            if lang == "abbreviations":
                for row in buf:
                    if "=" in row:
                        k, v = row.split("=", 1)
                        ctx.abbreviations.setdefault(k.strip(), v.strip())
            else:
                out.append(code_block("\n".join(buf)))
            continue
        if not stripped:
            flush_para()
            i += 1
            continue
        h = re.match(r"^(#{1,3})\s+(.*)$", stripped)
        if h:
            flush_para()
            level, text = len(h.group(1)), h.group(2).strip()
            if level == 1:
                ctx.chapter += 1
                out.append(heading1_chapter(ctx.chapter, text))
            elif level == 2 and text.startswith("Appendix"):
                out.append(heading1_plain(text))
            elif level == 2:
                out.append(heading2(text, ctx))
            else:
                out.append(heading3(text, ctx))
            i += 1
            continue
        fm = FIG_RE.match(stripped)
        if fm:
            flush_para()
            alt = fm.group("alt")
            cm = re.match(r"^(Figure \d+\.\d+)\.\s*(.*)$", alt)
            label, text = (cm.group(1), cm.group(2)) if cm else ("Figure", alt)
            img_path = (base / fm.group("path")).resolve()
            block = image_block(img_path, ctx)
            if block is None:
                ctx.warnings.append(f"missing image: {fm.group('path')}")
                block = para(
                    run(
                        f"[Figure to be inserted: {fm.group('path')}]",
                        font=TIMES,
                        highlight="yellow",
                    ),
                    jc="center",
                    keep_next=True,
                    before=120,
                    after=60,
                )
            out.append(block)
            out.append(caption_para(label, text, ctx, "fig", "center"))
            i += 1
            continue
        if stripped.startswith("|"):
            flush_para()
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                row = split_row(lines[i])
                if not all(re.fullmatch(r":?-{2,}:?", c.replace(" ", "")) for c in row):
                    rows.append(row)
                i += 1
            if pending_caption:
                out.append(caption_para(*pending_caption, ctx, "tab", None))
                pending_caption = None
            if rows:
                out.append(table_block(rows, ctx))
            continue
        lm = LIST_RE.match(line)
        if lm:
            flush_para()
            ordered = lm.group(2)[0].isdigit()
            items: list[tuple[int, str]] = []
            while i < len(lines):
                m2 = LIST_RE.match(lines[i])
                if m2 and m2.group(2)[0].isdigit() == ordered:
                    items.append((len(m2.group(1)) // 2, m2.group(3).strip()))
                    i += 1
                elif (
                    lines[i].startswith("  ")
                    and lines[i].strip()
                    and items
                    and not LIST_RE.match(lines[i])
                ):
                    items[-1] = (items[-1][0], items[-1][1] + " " + lines[i].strip())
                    i += 1
                else:
                    break
            out.append(list_block(items, ordered, ctx))
            continue
        para_buf.append(line)
        i += 1
    flush_para()
    if pending_caption:
        out.append(caption_para(*pending_caption, ctx, "tab", None))
    return out


# --------------------------------------------------------------------------------------------------
# Front matter, lists, bibliography
# --------------------------------------------------------------------------------------------------
def centered(
    text: str, size: float, bold: bool = False, highlight: str | None = None, after: int = 60
) -> str:
    return para(
        run(text, font=TIMES, size=size, bold=bold, highlight=highlight),
        jc="center",
        after=after,
        line=276,
    )


def title_page(ctx: Ctx) -> list[str]:
    today = dt.date.today().strftime("%B %Y")
    out = [
        centered("A", 18),
        centered("PROJECT-I REPORT", 22),
        centered("on", 18),
        para(run(PROJECT_TITLE, font=TIMES, size=22), jc="center", after=120, line=276),
        centered("Submitted by:", 18),
        centered("[Student name(s)]", 22, highlight="yellow", after=20),
        centered("[Enrollment No(s).]", 18, highlight="yellow"),
        centered("under mentorship of", 18),
        centered("Prof. Ashish Kumar", 22, after=20),
        centered(
            "[Assistant / Associate / Professor: confirm the title and designation]",
            18,
            highlight="yellow",
        ),
    ]
    if LOGO.exists():
        out.append(image_block(LOGO, ctx, max_w_cm=6.5, max_h_cm=4.0) or "")
    out += [
        para("", after=200),
        centered("Department of Computer Science Engineering", 18, after=0),
        centered("School of Engineering and Technology", 18, after=0),
        centered("BML MUNJAL UNIVERSITY, GURUGRAM (INDIA)", 18, after=120),
        centered(today, 18),
    ]
    return out


def sect_pr(fmt: str | None, start: int | None, footer: bool, margins: dict | None = None) -> str:
    m = margins or PAGE_MAR
    foot = '<w:footerReference w:type="default" r:id="rIdFooter1"/>' if footer else ""
    num = f'<w:pgNumType w:fmt="{fmt}" w:start="{start}"/>' if fmt else ""
    return (
        f'<w:sectPr>{foot}<w:pgSz w:w="11906" w:h="16838"/>'
        f'<w:pgMar w:top="{m["top"]}" w:right="{m["right"]}" w:bottom="{m["bottom"]}" w:left="{m["left"]}" '
        f'w:header="{m["header"]}" w:footer="{m["footer"]}" w:gutter="0"/>{num}</w:sectPr>'
    )


def front_title(text: str, first: bool = False) -> str:
    return para(
        run(text, font=SANS, bold=True, size=14),
        jc="center",
        border_bottom=True,
        after=240,
        keep_next=True,
        page_break_before=not first,
    )


def declarations() -> list[str]:
    big = dict(font=TIMES, bold=True, size=17)
    cert = (
        run("I hereby certify that the work on the project entitled, “", font=TIMES)
        + run(PROJECT_TITLE, font=TIMES, bold=True)
        + run(
            "”, in partial fulfillment of requirements for the award of Degree of ",
            font=TIMES,
        )
        + run("Bachelor of Technology", font=TIMES, bold=True)
        + run(
            " in School of Engineering and Technology at BML Munjal University, having University Roll No. ",
            font=TIMES,
        )
        + run("[University Roll No.]", font=TIMES, highlight="yellow")
        + run(
            ", is an authentic record of my own work carried out during a period from ", font=TIMES
        )
        + run("[month year]", font=TIMES, highlight="yellow")
        + run(" to ", font=TIMES)
        + run("[month year]", font=TIMES, highlight="yellow")
        + run(" under the supervision of ", font=TIMES)
        + run("[supervisor name]", font=TIMES, highlight="yellow")
        + run(".", font=TIMES)
    )
    return [
        para(run("CANDIDATE’S DECLARATION", **big), jc="center", after=360),
        para(cert, style="BodyText", jc="both", after=240),
        para(run("[Name of the candidate]", font=TIMES, bold=True, highlight="yellow"), after=480),
        para(run("SUPERVISOR’S DECLARATION", **big), jc="center", before=240, after=360),
        para(
            run(
                "This is to certify that the above statement made by the candidate is correct to the best of my knowledge.",
                font=TIMES,
            ),
            style="BodyText",
            jc="both",
            after=480,
        ),
        para(
            run("Faculty Supervisor Name: ", font=TIMES, bold=True)
            + run("[name]", font=TIMES, highlight="yellow"),
            after=60,
        ),
        para(run("Signature:", font=TIMES, bold=True), after=60),
    ]


def acknowledgement() -> list[str]:
    hl = dict(font=TIMES, highlight="yellow")
    return [
        front_title("ACKNOWLEDGEMENT"),
        para(
            run("I am highly grateful to ", font=TIMES)
            + run("Prof. Ashish Kumar", font=TIMES, bold=True)
            + run(", ", font=TIMES)
            + run("[designation]", **hl)
            + run(
                ", BML Munjal University, Gurugram, for providing supervision to carry out this project from ",
                font=TIMES,
            )
            + run("[month year]", **hl)
            + run(" to ", font=TIMES)
            + run("[month year]", **hl)
            + run(".", font=TIMES),
            style="BodyText",
            jc="both",
        ),
        para(
            run("Prof. Ashish Kumar", font=TIMES, bold=True)
            + run(
                " has provided great help in carrying out our work and is acknowledged with reverential thanks. "
                "Without wise counsel and able guidance, it would have been impossible to complete the project in this manner.",
                font=TIMES,
            ),
            style="BodyText",
            jc="both",
        ),
        para(
            run(
                "We would like to express thanks profusely to our supervisor for stimulating us from time to time. "
                "We would also like to thank the entire team at BML Munjal University, and the friends who devoted their "
                "valuable time and helped us in all possible ways toward the successful completion of this work.",
                font=TIMES,
            ),
            style="BodyText",
            jc="both",
        ),
        para(
            run(
                "[Acknowledgement of any AI assistance used in the project: wording to be agreed by the team.]",
                **hl,
            ),
            style="BodyText",
            jc="both",
        ),
        para(
            run("[Names of the team members]", font=TIMES, bold=True, highlight="yellow"),
            jc="right",
            before=480,
        ),
    ]


def toc_page() -> list[str]:
    toc = (
        '<w:r><w:fldChar w:fldCharType="begin"/></w:r>'
        '<w:r><w:instrText xml:space="preserve"> TOC \\o "1-2" \\h \\z \\u </w:instrText></w:r>'
        '<w:r><w:fldChar w:fldCharType="separate"/></w:r>'
        f"{run('The table of contents is built by Word: right-click here and choose Update Field.', font=TIMES)}"
        '<w:r><w:fldChar w:fldCharType="end"/></w:r>'
    )
    return [front_title("TABLE OF CONTENTS"), para(toc)]


def lined_table(
    rows: list[list[str]], widths: list[int], header: list[str], ctx: Ctx, pageref: bool
) -> str:
    grid = "".join(f'<w:gridCol w:w="{w}"/>' for w in widths)
    rule = '<w:top w:val="single" w:sz="6" w:space="0" w:color="000000"/><w:bottom w:val="single" w:sz="6" w:space="0" w:color="000000"/>'
    out = [
        f'<w:tbl><w:tblPr><w:tblW w:w="{sum(widths)}" w:type="dxa"/><w:tblLayout w:type="fixed"/>'
        '<w:tblCellMar><w:top w:w="40" w:type="dxa"/><w:left w:w="60" w:type="dxa"/><w:bottom w:w="40" w:type="dxa"/>'
        f'<w:right w:w="60" w:type="dxa"/></w:tblCellMar></w:tblPr><w:tblGrid>{grid}</w:tblGrid>'
    ]
    hdr = "".join(
        f'<w:tc><w:tcPr><w:tcW w:w="{widths[i]}" w:type="dxa"/><w:tcBorders>{rule}</w:tcBorders></w:tcPr>'
        f'<w:p><w:pPr><w:pStyle w:val="TableText"/></w:pPr>{run(h, font=TIMES, bold=True, size=11)}</w:p></w:tc>'
        for i, h in enumerate(header)
    )
    out.append(f"<w:tr><w:trPr><w:tblHeader/></w:trPr>{hdr}</w:tr>")
    for row in rows:
        cells = []
        for i, text in enumerate(row):
            if pageref and i == len(row) - 1:
                content = field_xml(f"PAGEREF {text} \\h", "0", font=TIMES, size=11)
            else:
                content = inline(text, ctx, font=TIMES, size=11)
            cells.append(
                f'<w:tc><w:tcPr><w:tcW w:w="{widths[i]}" w:type="dxa"/></w:tcPr>'
                f'<w:p><w:pPr><w:pStyle w:val="TableText"/></w:pPr>{content}</w:p></w:tc>'
            )
        out.append(f"<w:tr><w:trPr><w:cantSplit/></w:trPr>{''.join(cells)}</w:tr>")
    out.append("</w:tbl>")
    return "".join(out)


def list_pages(ctx: Ctx) -> list[str]:
    out = [front_title("LIST OF FIGURES")]
    w = [1300, 6745, 1000]
    figs = [[lab, txt, bm] for lab, txt, bm in ctx.figures]
    out.append(
        lined_table(figs, w, ["Figure No.", "Figure Description", "Page No."], ctx, True)
        if figs
        else para("")
    )
    out.append(front_title("LIST OF TABLES"))
    tabs = [[lab, txt, bm] for lab, txt, bm in ctx.tables]
    out.append(
        lined_table(tabs, w, ["Table No.", "Table Description", "Page No."], ctx, True)
        if tabs
        else para("")
    )
    out.append(front_title("LIST OF ABBREVIATIONS"))
    abbr = sorted(ctx.abbreviations.items(), key=lambda kv: kv[0].lower())
    rows = [[k, v] for k, v in abbr]
    out.append(
        lined_table(rows, [2200, 6845], ["Abbreviation", "Full Form"], ctx, False)
        if rows
        else para("")
    )
    return out


def bibliography(ctx: Ctx) -> list[str]:
    out = [heading1_plain("Bibliography")]
    for n, key in enumerate(ctx.cites, start=1):
        text = ctx.refs[key]
        out.append(
            para(
                run(f"[{n}]", font=TIMES, size=11)
                + "<w:r><w:tab/></w:r>"
                + inline(text, ctx, font=TIMES, size=11),
                jc="both",
                hanging=560,
                ind_left=560,
                after=100,
                line=300,
                tabs=[("left", 560, None)],
            )
        )
    return out


def plagiarism_page() -> list[str]:
    return [
        heading1_plain("Plagiarism Check Report"),
        para(
            run(
                "[Attach the plagiarism check report here. The similarity index must be less than 10%.]",
                font=TIMES,
                highlight="yellow",
            ),
            style="BodyText",
        ),
    ]


def load_refs(src: Path) -> dict[str, str]:
    refs = {}
    for line in (src / "refs.md").read_text(encoding="utf-8").splitlines():
        m = re.match(r"^([a-z0-9_]+)\s*\|\s*(.+)$", line)
        if m:
            refs[m.group(1)] = m.group(2).strip()
    return refs


# --------------------------------------------------------------------------------------------------
# Package parts
# --------------------------------------------------------------------------------------------------
def styles_xml() -> str:
    def pstyle(sid, name, based="Normal", ppr="", rp="", nxt=None, qf=True):
        return (
            f'<w:style w:type="paragraph" w:styleId="{sid}"><w:name w:val="{name}"/>'
            + (f'<w:basedOn w:val="{based}"/>' if based else "")
            + (f'<w:next w:val="{nxt}"/>' if nxt else "")
            + ("<w:qFormat/>" if qf else "")
            + (f"<w:pPr>{ppr}</w:pPr>" if ppr else "")
            + (f"<w:rPr>{rp}</w:rPr>" if rp else "")
            + "</w:style>"
        )

    times = f'<w:rFonts w:ascii="{TIMES}" w:hAnsi="{TIMES}" w:cs="{TIMES}" w:eastAsia="{TIMES}"/>'
    return (
        f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:styles {NSDECL}>'
        f'<w:docDefaults><w:rPrDefault><w:rPr>{times}<w:sz w:val="24"/><w:szCs w:val="24"/>'
        '<w:lang w:val="en-GB" w:eastAsia="en-US" w:bidi="ar-SA"/></w:rPr></w:rPrDefault>'
        '<w:pPrDefault><w:pPr><w:spacing w:after="0" w:line="240" w:lineRule="auto"/></w:pPr></w:pPrDefault></w:docDefaults>'
        '<w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/><w:qFormat/></w:style>'
        + pstyle(
            "BodyText",
            "Body Text",
            ppr='<w:spacing w:after="120" w:line="360" w:lineRule="auto"/><w:jc w:val="both"/>',
        )
        + pstyle(
            "Heading1",
            "heading 1",
            nxt="BodyText",
            ppr='<w:keepNext/><w:spacing w:before="0" w:after="240" w:line="370" w:lineRule="exact"/><w:outlineLvl w:val="0"/>',
            rp='<w:b/><w:bCs/><w:sz w:val="34"/><w:szCs w:val="34"/>',
        )
        + pstyle(
            "Heading2",
            "heading 2",
            nxt="BodyText",
            ppr='<w:keepNext/><w:tabs><w:tab w:val="left" w:pos="780"/></w:tabs><w:spacing w:before="240" w:after="120"/>'
            '<w:ind w:left="780" w:hanging="780"/><w:outlineLvl w:val="1"/>',
            rp='<w:b/><w:bCs/><w:sz w:val="24"/><w:szCs w:val="24"/>',
        )
        + pstyle(
            "Heading3",
            "heading 3",
            nxt="BodyText",
            ppr='<w:keepNext/><w:tabs><w:tab w:val="left" w:pos="780"/></w:tabs><w:spacing w:before="180" w:after="100"/>'
            '<w:ind w:left="780" w:hanging="780"/><w:outlineLvl w:val="2"/>',
            rp='<w:b/><w:bCs/><w:i/><w:iCs/><w:sz w:val="24"/><w:szCs w:val="24"/>',
        )
        + pstyle("ListParagraph", "List Paragraph", ppr='<w:ind w:left="720"/>')
        + pstyle(
            "Caption",
            "caption",
            ppr='<w:keepLines/><w:spacing w:before="80" w:after="200" w:line="260" w:lineRule="auto"/>',
            rp='<w:sz w:val="22"/><w:szCs w:val="22"/>',
        )
        + pstyle(
            "TableText",
            "Table Text",
            ppr='<w:spacing w:before="0" w:after="0" w:line="250" w:lineRule="auto"/>',
            rp='<w:sz w:val="20"/><w:szCs w:val="20"/>',
        )
        + pstyle(
            "CodeBlock",
            "Code Block",
            ppr='<w:spacing w:before="60" w:after="160" w:line="240" w:lineRule="auto"/><w:ind w:left="120" w:right="120"/>',
            rp='<w:rFonts w:ascii="Consolas" w:hAnsi="Consolas" w:cs="Consolas"/><w:sz w:val="17"/><w:szCs w:val="17"/>',
        )
        + pstyle(
            "TOC1",
            "toc 1",
            ppr=f'<w:tabs><w:tab w:val="right" w:leader="dot" w:pos="{TEXT_WIDTH}"/></w:tabs><w:spacing w:before="120" w:after="40"/>',
            rp="<w:b/><w:bCs/>",
            qf=False,
        )
        + pstyle(
            "TOC2",
            "toc 2",
            ppr=f'<w:tabs><w:tab w:val="right" w:leader="dot" w:pos="{TEXT_WIDTH}"/></w:tabs><w:spacing w:after="20"/><w:ind w:left="360"/>',
            qf=False,
        )
        + '<w:style w:type="character" w:styleId="Hyperlink"><w:name w:val="Hyperlink"/></w:style>'
        + "</w:styles>"
    )


def numbering_xml(ordered_ids: list[int]) -> str:
    bullet_lvls = "".join(
        f'<w:lvl w:ilvl="{i}"><w:start w:val="1"/><w:numFmt w:val="bullet"/><w:lvlText w:val="{c}"/><w:lvlJc w:val="left"/>'
        f'<w:pPr><w:ind w:left="{720 + 360 * i}" w:hanging="360"/></w:pPr>'
        f'<w:rPr><w:rFonts w:ascii="{TIMES}" w:hAnsi="{TIMES}" w:hint="default"/></w:rPr></w:lvl>'
        for i, c in enumerate(["•", "–", "•"])
    )
    dec_lvls = "".join(
        f'<w:lvl w:ilvl="{i}"><w:start w:val="1"/><w:numFmt w:val="{fmt}"/><w:lvlText w:val="%{i + 1}{"." if i == 0 else ")"}"/>'
        f'<w:lvlJc w:val="left"/><w:pPr><w:ind w:left="{720 + 360 * i}" w:hanging="360"/></w:pPr></w:lvl>'
        for i, fmt in enumerate(["decimal", "lowerLetter", "lowerRoman"])
    )
    nums = '<w:num w:numId="1"><w:abstractNumId w:val="0"/></w:num>'
    for nid in ordered_ids:
        nums += (
            f'<w:num w:numId="{nid}"><w:abstractNumId w:val="1"/><w:lvlOverride w:ilvl="0"><w:startOverride w:val="1"/>'
            "</w:lvlOverride></w:num>"
        )
    return (
        f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:numbering {NSDECL}>'
        f'<w:abstractNum w:abstractNumId="0"><w:multiLevelType w:val="hybridMultilevel"/>{bullet_lvls}</w:abstractNum>'
        f'<w:abstractNum w:abstractNumId="1"><w:multiLevelType w:val="hybridMultilevel"/>{dec_lvls}</w:abstractNum>{nums}'
        "</w:numbering>"
    )


def footer_xml() -> str:
    page = field_xml("PAGE", "1", font=SANS, size=10)
    return (
        f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:ftr {NSDECL}>'
        f'<w:p><w:pPr><w:jc w:val="center"/></w:pPr>{page}</w:p></w:ftr>'
    )


def settings_xml() -> str:
    return (
        f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:settings {NSDECL}>'
        '<w:zoom w:percent="100"/><w:defaultTabStop w:val="720"/><w:characterSpacingControl w:val="doNotCompress"/>'
        '<w:compat><w:compatSetting w:name="compatibilityMode" w:uri="http://schemas.microsoft.com/office/word" w:val="15"/></w:compat>'
        "</w:settings>"
    )


def assemble(out_path: Path, src: Path = BMU) -> list[str]:
    ctx = Ctx(refs=load_refs(src))
    parts_md = {}
    for name in ("abstract", "part1", "part2", "part3"):
        p = src / f"{name}.md"
        parts_md[name] = p.read_text(encoding="utf-8") if p.exists() else ""
        if not parts_md[name]:
            ctx.warnings.append(f"missing source: {p.name}")

    # the abstract comes first in reading order, so its citations are numbered first; the lists of figures,
    # tables and abbreviations need the whole body, so the body is rendered before they are built
    abstract_blocks = [front_title("ABSTRACT")] + render_markdown(parts_md["abstract"], ctx, src)
    body_blocks: list[str] = []
    for name in ("part1", "part2", "part3"):
        body_blocks += render_markdown(parts_md[name], ctx, src)
    biblio = bibliography(ctx)
    front_matter: list[str] = (
        declarations() + acknowledgement() + toc_page() + abstract_blocks + list_pages(ctx)
    )

    title = title_page(ctx)
    title_last = para(
        "",
        sect=sect_pr(
            None, None, False, dict(PAGE_MAR, top=1440, bottom=1200, left=1440, right=1440)
        ),
    )
    front_last = para("", sect=sect_pr("lowerRoman", 1, True))
    final_sect = sect_pr("decimal", 1, True)
    body = (
        "".join(title)
        + title_last
        + "".join(front_matter)
        + front_last
        + "".join(body_blocks)
        + "".join(biblio)
        + "".join(plagiarism_page())
    )
    document = f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:document {NSDECL}><w:body>{body}{final_sect}</w:body></w:document>'

    rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rIdStyles" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
        '<Relationship Id="rIdNum" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/numbering" Target="numbering.xml"/>'
        '<Relationship Id="rIdSettings" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/settings" Target="settings.xml"/>'
        '<Relationship Id="rIdFooter1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/footer" Target="footer1.xml"/>'
        + "".join(
            f'<Relationship Id="{rid}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="{target}"/>'
            for rid, target in ctx.rels
        )
        + "</Relationships>"
    )
    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/><Default Extension="png" ContentType="image/png"/>'
        '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
        '<Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>'
        '<Override PartName="/word/numbering.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.numbering+xml"/>'
        '<Override PartName="/word/settings.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.settings+xml"/>'
        '<Override PartName="/word/footer1.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.footer+xml"/>'
        '<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>'
        "</Types>"
    )
    root_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>'
        '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>'
        "</Relationships>"
    )
    now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    core = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><cp:coreProperties '
        'xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" '
        'xmlns:dcterms="http://purl.org/dc/terms/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
        f"<dc:title>{escape(PROJECT_TITLE)}</dc:title><dc:creator></dc:creator>"
        f'<dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created></cp:coreProperties>'
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", content_types)
        z.writestr("_rels/.rels", root_rels)
        z.writestr("word/document.xml", document)
        z.writestr("word/_rels/document.xml.rels", rels)
        z.writestr("word/styles.xml", styles_xml())
        z.writestr("word/numbering.xml", numbering_xml(ctx.num_instances))
        z.writestr("word/settings.xml", settings_xml())
        z.writestr("word/footer1.xml", footer_xml())
        z.writestr("docProps/core.xml", core)
        for name, data in ctx.media:
            z.writestr(f"word/media/{name}", data)
    stats = [
        f"chapters: {ctx.chapter}",
        f"figures: {len(ctx.figures)}, tables: {len(ctx.tables)}, citations: {len(ctx.cites)}",
        f"abbreviations: {len(ctx.abbreviations)}",
    ]
    return stats + [f"WARNING: {w}" for w in ctx.warnings]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="docs/report/Nightingale-Project-Report-BMU.docx")
    parser.add_argument(
        "--src", default=str(BMU), help="folder with abstract.md, part1-3.md, refs.md"
    )
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    for line in assemble(ROOT / args.out, Path(args.src)):
        print(line)
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
