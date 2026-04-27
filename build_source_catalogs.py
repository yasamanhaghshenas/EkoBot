import argparse
import json
import re
import sys
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path


NS = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def normalize_key(value):
    value = (value or "").strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def load_shared_strings(zf):
    shared = []
    if "xl/sharedStrings.xml" not in zf.namelist():
        return shared
    root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
    for si in root.findall("a:si", NS):
        shared.append("".join(t.text or "" for t in si.iterfind(".//a:t", NS)))
    return shared


def cell_text(cell, shared):
    cell_type = cell.attrib.get("t")
    value = cell.find("a:v", NS)
    if cell_type == "s" and value is not None:
        try:
            return shared[int(value.text)]
        except Exception:
            return ""
    if cell_type == "inlineStr":
        return "".join(t.text or "" for t in cell.iterfind(".//a:t", NS))
    return value.text if value is not None else ""


def read_workbook_tables(path):
    """
    Read an .xlsx workbook into a list of tables.
    Each table is a dict with `sheet` and `rows`.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    tables = []
    with zipfile.ZipFile(path) as zf:
        workbook = ET.fromstring(zf.read("xl/workbook.xml"))
        rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        rel_map = {rel.attrib["Id"]: rel.attrib["Target"] for rel in rels}
        shared = load_shared_strings(zf)

        for sheet in workbook.findall("a:sheets/a:sheet", NS):
            rel_id = sheet.attrib.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id")
            target = rel_map.get(rel_id)
            if not target or not target.startswith("worksheets/sheet"):
                continue

            sheet_name = sheet.attrib.get("name") or target
            root = ET.fromstring(zf.read(f"xl/{target}"))
            rows = []
            for row in root.findall(".//a:sheetData/a:row", NS):
                values = [cell_text(cell, shared).strip() for cell in row.findall("a:c", NS)]
                if any(values):
                    rows.append(values)

            tables.append({"sheet": sheet_name, "rows": rows})

    return tables


def rows_to_dicts(rows):
    if not rows:
        return []

    headers = [normalize_key(h) for h in rows[0]]
    out = []
    for values in rows[1:]:
        item = {}
        for idx, header in enumerate(headers):
            if not header:
                continue
            if idx >= len(values):
                break
            value = values[idx].strip() if idx < len(values) and values[idx] else ""
            if value:
                item[header] = value
        if item:
            out.append(item)
    return out


def build_ieu_url_catalog(xlsx_path):
    tables = read_workbook_tables(xlsx_path)
    out = []
    for table in tables:
        for item in rows_to_dicts(table["rows"]):
            item["source_workbook"] = "IEU URLs.xlsx"
            item["sheet"] = table["sheet"]
            out.append(item)
    return out


def build_ecobot_rule_catalog(xlsx_path):
    tables = read_workbook_tables(xlsx_path)
    out = []
    for table in tables:
        for item in rows_to_dicts(table["rows"]):
            item["source_workbook"] = "EcoBot Rules.xlsx"
            item["sheet"] = table["sheet"]
            out.append(item)
    return out


def default_downloads_path(filename):
    return Path.home() / "Downloads" / filename


def main():
    parser = argparse.ArgumentParser(description="Build JSON catalogs from the IEU workbook sources.")
    parser.add_argument(
        "--ieu-urls",
        default=str(default_downloads_path("IEU URLs.xlsx")),
        help="Path to the IEU URLs workbook.",
    )
    parser.add_argument(
        "--eco-rules",
        default=str(default_downloads_path("EcoBot Rules.xlsx")),
        help="Path to the EcoBot Rules workbook.",
    )
    parser.add_argument(
        "--out-dir",
        default=".",
        help="Directory to write the generated JSON catalogs into.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ieu_catalog = build_ieu_url_catalog(args.ieu_urls)
    eco_catalog = build_ecobot_rule_catalog(args.eco_rules)

    ieu_out = out_dir / "ieu_url_catalog.json"
    eco_out = out_dir / "ecobot_rules_sources.json"

    ieu_out.write_text(json.dumps(ieu_catalog, ensure_ascii=False, indent=2), encoding="utf-8")
    eco_out.write_text(json.dumps(eco_catalog, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote {len(ieu_catalog)} IEU URL rows -> {ieu_out}")
    print(f"Wrote {len(eco_catalog)} EcoBot rule rows -> {eco_out}")


if __name__ == "__main__":
    main()
