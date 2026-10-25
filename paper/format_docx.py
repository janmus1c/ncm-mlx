import argparse
import copy
import re
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from zipfile import ZipFile

NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "m": "http://schemas.openxmlformats.org/officeDocument/2006/math",
}
for prefix, uri in NS.items():
    ET.register_namespace(prefix, uri)


def tag(name):
    prefix, local = name.split(":")
    return f"{{{NS[prefix]}}}{local}"


def element(parent, name, **attributes):
    node = parent.find(name, NS)
    if node is None:
        node = ET.Element(tag(name))
        if name.endswith("Pr"):
            parent.insert(0, node)
        else:
            parent.append(node)
    node.attrib.update({tag(f"w:{key}"): str(value) for key, value in attributes.items()})
    return node


def plain_text(node):
    return "".join(t.text or "" for t in node.findall(".//w:t", NS))


def repair_underbraces(document):
    repaired = 0
    for outer in document.findall(".//m:limLow", NS):
        content = outer.find("m:e", NS)
        inner = None if content is None else content.find("m:limLow", NS)
        if inner is None or len(content) != 1:
            continue
        brace = inner.find("m:lim", NS)
        if brace is None or "".join(brace.itertext()) != "⏟":
            continue
        expression = inner.find("m:e", NS)
        if expression is None:
            continue
        group = ET.Element(tag("m:groupChr"))
        properties = ET.SubElement(group, tag("m:groupChrPr"))
        for name, value in (("chr", "⏟"), ("pos", "bot"), ("vertJc", "top")):
            ET.SubElement(properties, tag(f"m:{name}"), {tag("m:val"): value})
        group.append(copy.deepcopy(expression))
        content.remove(inner)
        content.append(group)
        repaired += 1
    return repaired


def table_layout(headers):
    if headers == ["Term", "Meaning"]:
        return [0.33, 0.67], 22
    if len(headers) == 3 and headers[0] == "Component":
        return [0.25, 0.37, 0.38], 21
    if len(headers) == 3 and headers[0] == "":
        return [0.46, 0.27, 0.27], 21
    if headers == ["Model", "Params", "Easy", "Medium", "Hard"]:
        return [0.48, 0.10, 0.14, 0.14, 0.14], 21
    if headers == ["Run", "Updates × batch", "Presentations", "Best step", "Train seconds"]:
        return [0.18, 0.23, 0.21, 0.17, 0.21], 20
    if headers == ["Run", "Question", "Change from the previous run", "Result", "What we learned"]:
        return [0.10, 0.22, 0.24, 0.19, 0.25], 20
    if headers == [
        "Run",
        "Levels",
        "Codes",
        "Codec",
        "Codec steps (lr)",
        "Generator",
        "Gen. steps (lr)",
        "Other",
    ]:
        return [0.06, 0.15, 0.08, 0.18, 0.11, 0.13, 0.12, 0.17], 19
    if headers == ["Run", "Generator", "Steps", "Batch", "Learning rate", "Corruption"]:
        return [0.18, 0.16, 0.16, 0.11, 0.17, 0.22], 21
    if headers == ["Version", "Date", "Changes"]:
        return [0.12, 0.18, 0.70], 21
    return [1 / len(headers)] * len(headers), 21


def format_tables(document):
    section = document.find(".//w:sectPr", NS)
    page = section.find("w:pgSz", NS)
    margins = section.find("w:pgMar", NS)
    page_width = 12240 if page is None else int(page.get(tag("w:w"), 12240))
    available = page_width - sum(
        1440 if margins is None else int(margins.get(tag(f"w:{side}"), 1440))
        for side in ("left", "right")
    )
    tables = document.findall(".//w:tbl", NS)
    for table in tables:
        rows = table.findall("w:tr", NS)
        headers = [re.sub(r"\s+", " ", plain_text(cell)).strip() for cell in rows[0].findall("w:tc", NS)]
        proportions, font_size = table_layout(headers)
        total = sum(proportions)
        widths = [round(available * value / total) for value in proportions]
        widths[-1] = available - sum(widths[:-1])
        properties = element(table, "w:tblPr")
        element(properties, "w:tblW", w=available, type="dxa")
        element(properties, "w:tblLayout", type="fixed")
        element(properties, "w:tblInd", w=0, type="dxa")
        cell_margins = element(properties, "w:tblCellMar")
        for side, width in (("left", 100), ("right", 100), ("top", 65), ("bottom", 65)):
            element(cell_margins, f"w:{side}", w=width, type="dxa")
        grid = element(table, "w:tblGrid")
        grid.clear()
        for width in widths:
            ET.SubElement(grid, tag("w:gridCol"), {tag("w:w"): str(width)})
        for row_index, row in enumerate(rows):
            row_properties = element(row, "w:trPr")
            element(row_properties, "w:cantSplit")
            if row_index == 0:
                element(row_properties, "w:tblHeader")
            for column, cell in enumerate(row.findall("w:tc", NS)):
                cell_properties = element(cell, "w:tcPr")
                element(cell_properties, "w:tcW", w=widths[column], type="dxa")
                element(cell_properties, "w:vAlign", val="top")
                if row_index == 0:
                    borders = element(cell_properties, "w:tcBorders")
                    element(borders, "w:bottom", val="single", sz=6, color="8C9DA5")
                for paragraph in cell.findall("w:p", NS):
                    paragraph_properties = element(paragraph, "w:pPr")
                    element(paragraph_properties, "w:spacing", before=0, after=30, line=240, lineRule="auto")
                    keep = paragraph_properties.find("w:keepNext", NS)
                    if keep is not None:
                        paragraph_properties.remove(keep)
                    if row_index == 0:
                        element(paragraph_properties, "w:keepNext")
                    if headers[0] == "Run" and headers[-1] == "Train seconds" and column > 0:
                        element(paragraph_properties, "w:jc", val="right")
                    for run in paragraph.findall(".//w:r", NS):
                        run_properties = element(run, "w:rPr")
                        element(run_properties, "w:sz", val=font_size)
                        element(run_properties, "w:szCs", val=font_size)
                        if row_index == 0:
                            element(run_properties, "w:b")
                if "Levels" in headers and headers[column] == "Levels":
                    for text in cell.findall(".//w:t", NS):
                        if text.text:
                            text.text = re.sub(r",(?=\d)", ", ", text.text)
    return len(tables)


def format_docx(path):
    with ZipFile(path) as source:
        entries = [(item, source.read(item.filename)) for item in source.infolist()]
    document = ET.fromstring({item.filename: data for item, data in entries}["word/document.xml"])
    equations = repair_underbraces(document)
    tables = format_tables(document)
    updated = ET.tostring(document, encoding="utf-8", xml_declaration=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".docx", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        with ZipFile(temporary, "w") as target:
            for item, data in entries:
                target.writestr(item, updated if item.filename == "word/document.xml" else data)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Formatted {tables} tables and {equations} native equation annotations")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    format_docx(parser.parse_args().path)
