from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REQUIRED_FRONTMATTER = ("title", "author", "affiliations", "year", "source", "tags")

REQUIRED_SECTIONS = (
    "第一作者相关信息",
    "研究问题",
    "背景知识",
    "问题分析",
    "思想与方法",
    "算法与伪代码",
    "实验与消融",
    "展望",
)

SECTION_PATTERNS = tuple(
    re.compile(rf"^##\s*{index}\s*[.．、]?\s*{re.escape(title)}\s*$")
    for index, title in enumerate(REQUIRED_SECTIONS, start=1)
)


@dataclass
class ReviewResult:
    passed: bool
    errors: list[str]
    warnings: list[str]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "errors": self.errors,
            "warnings": self.warnings,
            "metadata": self.metadata,
        }


def strip_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


def parse_frontmatter(markdown: str) -> tuple[dict[str, Any], str, list[str]]:
    errors: list[str] = []
    lines = markdown.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, markdown, ["文件必须以 YAML front matter 开头，第一行应为 `---`。"]

    closing_index = None
    for index, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            closing_index = index
            break

    if closing_index is None:
        return {}, markdown, ["YAML front matter 缺少结束行 `---`。"]

    metadata: dict[str, Any] = {}
    current_key: str | None = None
    for raw_line in lines[1:closing_index]:
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue

        key_match = re.match(r"^([A-Za-z_][A-Za-z0-9_-]*)\s*:\s*(.*)$", raw_line)
        list_match = re.match(r"^\s*-\s*(.+?)\s*$", raw_line)

        if key_match:
            current_key = key_match.group(1).strip()
            value = key_match.group(2).strip()
            if value:
                if value.startswith("[") and value.endswith("]"):
                    metadata[current_key] = [
                        strip_quotes(item)
                        for item in re.split(r"\s*,\s*", value[1:-1])
                        if item.strip()
                    ]
                else:
                    metadata[current_key] = strip_quotes(value)
            else:
                metadata[current_key] = []
            continue

        if list_match and current_key:
            metadata.setdefault(current_key, [])
            if not isinstance(metadata[current_key], list):
                errors.append(f"`{current_key}` 同时出现标量和值列表，请改成一种格式。")
                continue
            metadata[current_key].append(strip_quotes(list_match.group(1)))
            continue

        errors.append(f"无法解析 front matter 行：`{raw_line}`。")

    body = "\n".join(lines[closing_index + 1 :]).lstrip()
    return metadata, body, errors


def normalize_title(value: str) -> str:
    return re.sub(r"[\W_]+", "", value, flags=re.UNICODE).lower()


def markdown_plain_text(value: str) -> str:
    value = re.sub(r"\[(.*?)\]\(.*?\)", r"\1", value)
    value = re.sub(r"[*_`>#]", "", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip(" ：:;-")


def first_h1(body: str) -> tuple[str | None, int | None]:
    for index, line in enumerate(body.splitlines()):
        if line.startswith("# ") and not line.startswith("## "):
            return line[2:].strip(), index
        if line.strip():
            return None, index
    return None, None


def find_first_h2(lines: list[str], start: int) -> int:
    for index in range(start, len(lines)):
        if lines[index].startswith("## "):
            return index
    return len(lines)


def extract_intro(lines: list[str], h1_index: int) -> str:
    first_h2_index = find_first_h2(lines, h1_index + 1)
    return "\n".join(lines[h1_index + 1 : first_h2_index]).strip()


def extract_summary_from_intro(intro: str) -> str:
    for line in intro.splitlines():
        if "一句话概括" not in line:
            continue
        text = re.split(r"一句话概括\s*[：:]", line, maxsplit=1)
        return markdown_plain_text(text[-1] if text else line)
    return ""


def h2_lines(lines: list[str]) -> list[tuple[int, str]]:
    return [(index, line.strip()) for index, line in enumerate(lines) if line.startswith("## ")]


def review_frontmatter(metadata: dict[str, Any]) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    for key in REQUIRED_FRONTMATTER:
        value = metadata.get(key)
        if value is None or value == "" or value == []:
            errors.append(f"front matter 缺少必填字段 `{key}`。")

    year = str(metadata.get("year", "")).strip()
    if year and not re.match(r"^(19|20)\d{2}$", year):
        errors.append("`year` 必须是四位年份，例如 `2025`。")

    tags = metadata.get("tags")
    if tags and not isinstance(tags, list):
        if isinstance(tags, str):
            metadata["tags"] = [item.strip() for item in tags.split(",") if item.strip()]
        else:
            errors.append("`tags` 必须是 YAML 列表或逗号分隔字符串。")

    if isinstance(metadata.get("tags"), list) and len(metadata["tags"]) < 1:
        errors.append("`tags` 至少需要包含一个标签。")
    if isinstance(metadata.get("tags"), list) and len(metadata["tags"]) > 8:
        warnings.append("`tags` 多于 8 个，建议控制在 3-6 个以便网页展示。")

    return errors, warnings


def review_structure(metadata: dict[str, Any], body: str) -> tuple[list[str], list[str], dict[str, Any]]:
    errors: list[str] = []
    warnings: list[str] = []
    extracted: dict[str, Any] = {}
    lines = body.splitlines()

    h1_title, h1_index = first_h1(body)
    if not h1_title or h1_index is None:
        errors.append("front matter 后第一个非空内容必须是一级标题 `# 论文标题`。")
        return errors, warnings, extracted

    extracted["h1_title"] = h1_title
    if metadata.get("title") and normalize_title(str(metadata["title"])) != normalize_title(h1_title):
        errors.append("一级标题必须与 front matter 的 `title` 保持一致。")

    intro = extract_intro(lines, h1_index)
    extracted["summary"] = extract_summary_from_intro(intro)
    if "一句话概括" not in intro:
        errors.append("一级标题后必须紧接包含 `一句话概括` 的段落。")
    if "定位" not in intro:
        errors.append("一级标题后必须紧接包含论文定位的段落，例如 `论文的定位` 或 `我对这篇论文的定位`。")
    if not extracted["summary"]:
        warnings.append("未能从 `一句话概括` 行提取摘要，上传时会使用标题作为兜底摘要。")

    headings = h2_lines(lines)
    heading_cursor = 0
    for section_index, pattern in enumerate(SECTION_PATTERNS):
        if heading_cursor >= len(headings):
            errors.append(f"缺少第 {section_index + 1} 段：`{REQUIRED_SECTIONS[section_index]}`。")
            continue
        line_number, heading = headings[heading_cursor]
        if not pattern.match(heading):
            errors.append(
                f"第 {section_index + 1} 个二级标题应为 `## {section_index + 1}. "
                f"{REQUIRED_SECTIONS[section_index]}`，实际为第 {line_number + 1} 行 `{heading}`。"
            )
        heading_cursor += 1

    if len(headings) <= len(REQUIRED_SECTIONS):
        errors.append("文章最后必须包含 `## Links` 链接段。")
        return errors, warnings, extracted

    links_line_number, links_heading = headings[len(REQUIRED_SECTIONS)]
    if not re.match(r"^##\s+Links\s*$", links_heading, flags=re.IGNORECASE):
        errors.append(f"八段正文之后必须立即放置 `## Links`，实际为第 {links_line_number + 1} 行 `{links_heading}`。")

    links_start = links_line_number + 1
    links_end = headings[len(REQUIRED_SECTIONS) + 1][0] if len(headings) > len(REQUIRED_SECTIONS) + 1 else len(lines)
    links_body = "\n".join(lines[links_start:links_end])
    if not re.search(r"https?://\S+", links_body):
        errors.append("`## Links` 段必须包含至少一个可点击 URL。")
    if not re.search(r"(Paper|论文|OpenReview|arXiv|PDF)", links_body, flags=re.IGNORECASE):
        errors.append("`## Links` 段必须包含论文链接条目。")
    if not re.search(r"(Code|代码|GitHub|GitLab)", links_body, flags=re.IGNORECASE):
        errors.append("`## Links` 段必须包含代码链接条目。")

    return errors, warnings, extracted


def review_markdown(markdown: str) -> ReviewResult:
    metadata, body, parse_errors = parse_frontmatter(markdown)
    frontmatter_errors, frontmatter_warnings = review_frontmatter(metadata)
    structure_errors, structure_warnings, extracted = review_structure(metadata, body)

    result_metadata = dict(metadata)
    result_metadata.update(extracted)
    if "tags" in result_metadata and isinstance(result_metadata["tags"], str):
        result_metadata["tags"] = [item.strip() for item in result_metadata["tags"].split(",") if item.strip()]

    errors = parse_errors + frontmatter_errors + structure_errors
    warnings = frontmatter_warnings + structure_warnings
    return ReviewResult(passed=not errors, errors=errors, warnings=warnings, metadata=result_metadata)


def review_file(path: Path) -> ReviewResult:
    return review_markdown(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Review ORL Markdown notes before publishing.")
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    args = parser.parse_args()

    results = {str(path): review_file(path).to_dict() for path in args.paths}
    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        for path, result in results.items():
            status = "PASS" if result["passed"] else "FAIL"
            print(f"{status} {path}")
            for message in result["errors"]:
                print(f"  ERROR: {message}")
            for message in result["warnings"]:
                print(f"  WARN: {message}")
    return 0 if all(result["passed"] for result in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
