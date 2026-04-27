import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from catalog import fetch_web_page_text, load_web_sources


def build_cache(max_workers=8):
    sources = load_web_sources()
    items = []
    if not sources:
        return items

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {}
        for source in sources:
            url = (source.get("url") or "").strip()
            if not url:
                continue
            future_map[executor.submit(fetch_web_page_text, url)] = source

        for future in as_completed(future_map):
            source = future_map[future]
            url = (source.get("url") or "").strip()
            title = (source.get("title") or "").strip()
            description = (source.get("description") or "").strip()
            try:
                text = future.result()
            except Exception:
                text = None

            items.append(
                {
                    "title": title,
                    "url": url,
                    "description": description,
                    "text": (text or "").strip(),
                    "source_workbook": source.get("source_workbook"),
                    "sheet": source.get("sheet"),
                }
            )

    items.sort(key=lambda item: (item.get("title") or "").lower())
    return items


def main():
    parser = argparse.ArgumentParser(description="Build cached text for the IEU web source catalog.")
    parser.add_argument(
        "--output",
        default="ecobot_rules_pages.json",
        help="Path to write the cached page text JSON.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of concurrent fetch workers.",
    )
    args = parser.parse_args()

    items = build_cache(max_workers=max(args.workers, 1))
    out_path = Path(args.output)
    out_path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(items)} cached pages -> {out_path}")


if __name__ == "__main__":
    main()
