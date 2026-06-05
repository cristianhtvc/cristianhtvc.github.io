from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from content_review import review_markdown


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
DATA_PATH = ROOT / "data" / "papers.json"
NOTES_ROOT = ROOT / "data" / "notes" / "orl"
GENERATED_ROOT_ITEMS = {
    ".nojekyll",
    "404.html",
    "assets",
    "embodied_llm",
    "index.html",
    "javascripts",
    "offline_rl",
    "search",
    "sitemap.xml",
    "sitemap.xml.gz",
    "stylesheets",
    "tech-stack",
}

SOURCE_TO_VENUE = {
    "ICLR": "iclr",
    "ICML": "icml",
    "NEURIPS": "neurips",
    "NIPS": "neurips",
    "AAMAS": "aamas",
}

JOURNAL_SOURCES = ("JMLR", "TMLR", "TPAMI", "AIJ", "NATURE", "SCIENCE")


@dataclass
class UploadResult:
    saved_path: str
    page_path: str
    title: str
    venue: str
    year: int
    review: dict[str, Any]
    build_log: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "saved_path": self.saved_path,
            "page_path": self.page_path,
            "title": self.title,
            "venue": self.venue,
            "year": self.year,
            "review": self.review,
            "build_log": self.build_log,
        }


def slugify(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-")[:90] or "paper"


def infer_venue_id(source: str) -> str:
    upper = source.upper()
    for token, venue_id in SOURCE_TO_VENUE.items():
        if token in upper:
            return venue_id
    if any(token in upper for token in JOURNAL_SOURCES):
        return "journals"
    return "other_conferences"


def clean_filename(name: str, title: str, source: str, year: int) -> str:
    name = Path(name or "").name
    if not name.lower().endswith(".md"):
        name = f"[{source.replace(' ', '_')}] {title}.md"
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name or f"{slugify(title)}.md"


def load_data() -> dict[str, Any]:
    return json.loads(DATA_PATH.read_text(encoding="utf-8"))


def save_data(data: dict[str, Any]) -> None:
    DATA_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def find_venue(data: dict[str, Any], venue_id: str) -> dict[str, Any]:
    for venue in data["offline_rl"]["venues"]:
        if venue["id"] == venue_id:
            return venue
    raise ValueError(f"papers.json 中没有 venue `{venue_id}`。")


def ensure_year_block(venue: dict[str, Any], year: int) -> dict[str, Any]:
    for year_block in venue["years"]:
        if int(year_block["year"]) == year:
            return year_block
    year_block = {"year": year, "papers": []}
    venue["years"].append(year_block)
    venue["years"].sort(key=lambda item: int(item["year"]), reverse=True)
    return year_block


def relative_to_root(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def paper_page_path(venue_id: str, year: int, title: str) -> str:
    return f"offline_rl/{venue_id}/{year}/{slugify(title)}/"


def paper_entry(metadata: dict[str, Any], source_markdown: str) -> dict[str, Any]:
    title = str(metadata["title"])
    summary = str(metadata.get("summary") or title)
    if len(summary) > 240:
        summary = summary[:237].rstrip() + "..."
    return {
        "title": title,
        "authors": str(metadata.get("author", "详见笔记")),
        "venue": str(metadata.get("source", "")),
        "type": "精读",
        "status": "已整理",
        "summary": summary,
        "tags": metadata.get("tags", ["Offline RL"]),
        "source_markdown": source_markdown,
    }


def upsert_paper(data: dict[str, Any], venue_id: str, year: int, entry: dict[str, Any], replace: bool) -> None:
    venue = find_venue(data, venue_id)
    year_block = ensure_year_block(venue, year)
    papers = year_block["papers"]
    for index, existing in enumerate(papers):
        if existing.get("title", "").casefold() == entry["title"].casefold():
            if not replace:
                raise ValueError(f"`{entry['title']}` 已存在；如需覆盖，请开启 replace。")
            papers[index] = entry
            return
    papers.append(entry)
    papers.sort(key=lambda item: item["title"].casefold())


def run_command(command: list[str], cwd: Path) -> str:
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return completed.stdout.strip()


def build_site() -> list[str]:
    logs = []
    logs.append(run_command([sys.executable, "scripts/generate_site.py"], ROOT))
    logs.append(run_command([sys.executable, "-m", "mkdocs", "build", "--clean", "--strict"], ROOT))
    return [log for log in logs if log]


def publish_site() -> None:
    built = ROOT / "site"
    if not built.exists():
        raise FileNotFoundError("未找到 mkdocs-source/site，请先构建站点。")
    for item_name in GENERATED_ROOT_ITEMS:
        target = REPO_ROOT / item_name
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()
    for child in built.iterdir():
        target = REPO_ROOT / child.name
        if child.is_dir():
            shutil.copytree(child, target, dirs_exist_ok=True)
        else:
            shutil.copy2(child, target)
    (REPO_ROOT / ".nojekyll").write_text("", encoding="utf-8")


def upload_markdown(
    markdown: str,
    original_name: str = "",
    *,
    replace: bool = False,
    build: bool = True,
    publish: bool = True,
) -> UploadResult:
    review = review_markdown(markdown)
    if not review.passed:
        raise ValueError("内容审查未通过：" + "；".join(review.errors))

    metadata = review.metadata
    title = str(metadata["title"])
    source = str(metadata["source"])
    year = int(str(metadata["year"]))
    venue_id = infer_venue_id(source)
    filename = clean_filename(original_name, title, source, year)
    note_dir = NOTES_ROOT / venue_id / str(year)
    note_dir.mkdir(parents=True, exist_ok=True)
    note_path = note_dir / filename

    if note_path.exists() and not replace:
        raise ValueError(f"`{relative_to_root(note_path)}` 已存在；如需覆盖，请开启 replace。")

    build_log: list[str] = []
    old_note = note_path.read_text(encoding="utf-8") if note_path.exists() else None
    old_data = DATA_PATH.read_text(encoding="utf-8")
    source_markdown = relative_to_root(note_path)

    try:
        note_path.write_text(markdown.rstrip() + "\n", encoding="utf-8")

        data = load_data()
        upsert_paper(data, venue_id, year, paper_entry(metadata, source_markdown), replace)
        save_data(data)

        if build:
            build_log = build_site()
            if publish:
                publish_site()
    except Exception:
        if old_note is None:
            if note_path.exists():
                note_path.unlink()
        else:
            note_path.write_text(old_note, encoding="utf-8")
        DATA_PATH.write_text(old_data, encoding="utf-8")
        raise

    return UploadResult(
        saved_path=source_markdown,
        page_path=paper_page_path(venue_id, year, title),
        title=title,
        venue=venue_id,
        year=year,
        review=review.to_dict(),
        build_log=build_log,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Upload a reviewed ORL Markdown note into the MkDocs site.")
    parser.add_argument("file", type=Path)
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--no-build", action="store_true")
    parser.add_argument("--no-publish", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    try:
        result = upload_markdown(
            args.file.read_text(encoding="utf-8"),
            args.file.name,
            replace=args.replace,
            build=not args.no_build,
            publish=not args.no_publish,
        )
    except Exception as exc:
        if args.json:
            print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False, indent=2))
        else:
            print(f"上传失败：{exc}")
        return 1

    if args.json:
        print(json.dumps({"ok": True, "result": result.to_dict()}, ensure_ascii=False, indent=2))
    else:
        print(f"上传成功：{result.saved_path}")
        print(f"页面路径：{result.page_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
