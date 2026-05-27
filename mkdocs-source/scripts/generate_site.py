from __future__ import annotations

import json
import re
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "papers.json"
DOCS = ROOT / "docs"
PRESERVED_DOCS_DIRS = {"stylesheets", "assets", "javascripts"}
PRESERVED_DOCS_FILES = {".nojekyll"}


def slugify(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-")[:90] or "paper"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write(path: Path, content: str) -> None:
    ensure_dir(path.parent)
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def yaml_string(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def paper_path(venue_id: str, year: int, title: str) -> str:
    return f"offline_rl/{venue_id}/{year}/{slugify(title)}.md"


def flatten_orl_papers(data: dict) -> list[dict]:
    rows = []
    for venue in data["offline_rl"]["venues"]:
        for year_block in venue["years"]:
            for paper in year_block["papers"]:
                rows.append({"venue": venue, "year": year_block["year"], "paper": paper})
    return rows


def count_orl_years(data: dict) -> int:
    return sum(len(venue["years"]) for venue in data["offline_rl"]["venues"])


def build_mkdocs(data: dict) -> str:
    site = data["site"]
    lines = [
        f"site_name: {yaml_string(site['name'])}",
        f"site_description: {yaml_string(site['description'])}",
        f"site_url: {yaml_string(site['url'])}",
        "docs_dir: docs",
        "site_dir: site",
        "theme:",
        "  name: material",
        "  language: zh",
        "  favicon: assets/favicon.svg",
        "  icon:",
        "    logo: material/library-shelves",
        "    repo: fontawesome/brands/github",
        "  features:",
        "    - navigation.tabs",
        "    - navigation.indexes",
        "    - navigation.top",
        "    - navigation.tracking",
        "    - search.suggest",
        "    - search.highlight",
        "    - content.code.copy",
        "    - toc.follow",
        "  palette:",
        "    - media: '(prefers-color-scheme: light)'",
        "      scheme: default",
        "      primary: custom",
        "      accent: custom",
        "      toggle:",
        "        icon: material/weather-night",
        "        name: 切换到深色模式",
        "    - media: '(prefers-color-scheme: dark)'",
        "      scheme: slate",
        "      primary: custom",
        "      accent: custom",
        "      toggle:",
        "        icon: material/weather-sunny",
        "        name: 切换到浅色模式",
        "markdown_extensions:",
        "  - admonition",
        "  - attr_list",
        "  - md_in_html",
        "  - pymdownx.details",
        "  - pymdownx.superfences",
        "  - pymdownx.tabbed:",
        "      alternate_style: true",
        "  - pymdownx.highlight:",
        "      anchor_linenums: true",
        "  - pymdownx.arithmatex:",
        "      generic: true",
        "plugins:",
        "  - search:",
        "      lang:",
        "        - zh",
        "        - en",
        "extra_css:",
        "  - stylesheets/extra.css",
        "extra_javascript:",
        "  - javascripts/mathjax.js",
        "  - https://unpkg.com/mathjax@3/es5/tex-mml-chtml.js",
        "nav:",
        "  - 首页: index.md",
        "  - 技术栈: tech-stack.md",
        "  - 离线强化学习:",
        "      - 总览: offline_rl/index.md",
    ]

    if site.get("repo_url"):
        lines.insert(3, f"repo_url: {yaml_string(site['repo_url'])}")
        lines.insert(4, 'repo_name: "GitHub"')

    for venue in data["offline_rl"]["venues"]:
        lines.append(f"      - {venue['title']}:")
        lines.append(f"          - 总览: offline_rl/{venue['id']}/index.md")
        for year_block in venue["years"]:
            year = year_block["year"]
            lines.append(f"          - {year}:")
            lines.append(f"              - 年度总览: offline_rl/{venue['id']}/{year}/index.md")
            for paper in year_block["papers"]:
                lines.append(
                    f"              - {yaml_string(paper['title'])}: "
                    f"{paper_path(venue['id'], year, paper['title'])}"
                )

    lines.extend(
        [
            "  - 具身智能与 LLM:",
            "      - 总览: embodied_llm/index.md",
        ]
    )
    for section in data["embodied_llm"]["sections"]:
        lines.append(f"      - {section['title']}: embodied_llm/{section['id']}.md")
    return "\n".join(lines)


def render_home(data: dict) -> str:
    site = data["site"]
    orl_papers = flatten_orl_papers(data)
    active_venues = sum(
        1
        for venue in data["offline_rl"]["venues"]
        if any(year_block["papers"] for year_block in venue["years"])
    )

    return f"""---
title: "首页"
description: "{site['description']}"
search:
  exclude: true
hide:
  - toc
  - navigation
---

<section class="landing-hero" markdown>
<div class="hero-copy" markdown>
<p class="eyebrow">Offline RL · Embodied AI · LLM</p>

# {site['name']}

<p class="hero-subtitle">{site['description']}<br>{site['tagline']}</p>

<div class="hero-actions">
<a class="primary-action" href="offline_rl/">进入 ORL 主阵地</a>
<a class="secondary-action" href="embodied_llm/">观察具身智能与 LLM</a>
</div>
</div>

<div class="hero-visual">
<img src="assets/paper-atlas-cover.svg" alt="由论文卡片、标签和引用线组成的研究地图示意图">
</div>
</section>

<section class="metrics-row">
<div><strong>{len(orl_papers)}</strong><span>篇已上传 ORL 笔记</span></div>
<div><strong>{len(data['offline_rl']['venues'])}</strong><span>类 ORL 来源</span></div>
<div><strong>{count_orl_years(data)}</strong><span>个年份入口</span></div>
<div><strong>{active_venues}</strong><span>个已有内容入口</span></div>
</section>

<section class="section-head" markdown>

## 阅读地图

{site['daily_goal']}
</section>

<section class="conf-grid">
<article class="conf-card">
<p class="eyebrow">Main Track</p>
<h2><a href="offline_rl/">离线强化学习</a></h2>
<p>按 ICLR、ICML、NeurIPS、AAMAS、其他会议和期刊文章组织，并固定 2022-2026 年入口。</p>
<div class="chip-row">
<a class="chip" href="offline_rl/iclr/2025/">ICLR 2025 <span>{len(data['offline_rl']['venues'][0]['years'][1]['papers'])}</span></a>
<a class="chip" href="offline_rl/">全部 ORL 入口</a>
</div>
</article>
<article class="conf-card">
<p class="eyebrow">Frontier Watch</p>
<h2><a href="embodied_llm/">具身智能与 LLM</a></h2>
<p>保留行业动态与相关论文两个入口，用来追踪大模型、智能体和具身系统的最新进展。</p>
<div class="chip-row">
<a class="chip" href="embodied_llm/industry_updates/">行业最新动态</a>
<a class="chip" href="embodied_llm/related_papers/">相关论文</a>
</div>
</article>
</section>
"""


def render_tech_stack(data: dict) -> str:
    return """---
title: "技术栈"
description: "个人论文笔记站的建站技术栈与维护流程"
---

# 技术栈

这个站点采用 **MkDocs Material + Markdown + JSON 数据源**。它很适合论文笔记站：页面轻、搜索快、Markdown 友好，并且可以自然托管在 GitHub Pages 或 Cloudflare Pages 上。

## 为什么这样选

| 层级 | 选择 | 用途 |
| --- | --- | --- |
| 静态站生成 | MkDocs | 把 Markdown 笔记编译成静态网页 |
| 主题系统 | MkDocs Material | 提供导航、搜索、深浅色、目录和响应式布局 |
| 公式渲染 | pymdownx.arithmatex + MathJax | 渲染论文笔记中的行内公式与块级公式 |
| 内容格式 | Markdown + YAML front matter | 单篇论文笔记可直接编辑、审阅和版本管理 |
| 数据源 | `data/papers.json` | 统一维护板块、会议、年份和论文元数据 |
| 原始笔记 | `data/notes/` | 保存长文笔记原稿，由脚本挂载到网页中 |
| 页面生成 | `scripts/generate_site.py` | 从 JSON 自动生成首页、导航、索引页和论文页 |
| 样式扩展 | `docs/stylesheets/extra.css` | 定义个人站点配色、卡片和论文页面细节 |

## 内容维护流程

1. 把新论文笔记放入 `data/notes/`。
2. 在 `data/papers.json` 中新增论文元数据，并指向 `source_markdown`。
3. 运行 `python scripts/generate_site.py` 更新 Markdown 和导航。
4. 运行 `mkdocs serve` 本地预览，确认公式、标题和目录正常。
5. 运行 `mkdocs build` 生成 `site/` 静态目录并部署。
"""


def render_offline_overview(data: dict) -> str:
    block = data["offline_rl"]
    venue_cards = "\n".join(
        f"""<article class="area-card">
<p class="eyebrow">Offline RL</p>
<h2><a href="{venue['id']}/">{venue['title']}</a></h2>
<p>{venue['description']}</p>
<div class="chip-row">{render_year_chips(venue)}</div>
</article>"""
        for venue in block["venues"]
    )
    return f"""---
title: "{block['title']}"
description: "{block['description']}"
---

# {block['title']}

<p class="lead">{block['description']}</p>

<section class="area-grid">
{venue_cards}
</section>
"""


def render_year_chips(venue: dict) -> str:
    return "".join(
        f'<a class="chip" href="{venue["id"]}/{year_block["year"]}/">{year_block["year"]} <span>{len(year_block["papers"])}</span></a>'
        for year_block in venue["years"]
    )


def render_venue_page(venue: dict) -> str:
    year_cards = "\n".join(
        f"""<article class="area-card">
<p class="eyebrow">{venue['title']}</p>
<h2><a href="{year_block['year']}/">{year_block['year']}</a></h2>
<p>{len(year_block['papers'])} 篇已整理笔记。</p>
</article>"""
        for year_block in venue["years"]
    )
    return f"""---
title: "{venue['title']}"
description: "{venue['title']} 离线强化学习论文笔记"
---

# {venue['title']}

<p class="lead">{venue['description']}</p>

<section class="area-grid">
{year_cards}
</section>
"""


def render_year_page(venue: dict, year_block: dict) -> str:
    papers = year_block["papers"]
    if papers:
        paper_cards = "\n".join(render_paper_card(venue, year_block["year"], paper) for paper in papers)
    else:
        paper_cards = """<article class="empty-card">
<h2>等待填充</h2>
<p>这个年份入口已经建好，后续把论文笔记加入 `data/papers.json` 后会自动显示在这里。</p>
</article>"""

    return f"""---
title: "{venue['title']} {year_block['year']}"
description: "{venue['title']} {year_block['year']} 离线强化学习论文笔记"
---

# {venue['title']} {year_block['year']}

<p class="lead">{venue['description']}</p>

<section class="paper-list">
{paper_cards}
</section>
"""


def render_paper_card(venue: dict, year: int, paper: dict) -> str:
    return f"""<article class="paper-card">
<p class="eyebrow">{paper.get('type', '笔记')} · {paper.get('status', '待整理')}</p>
<h2><a href="{slugify(paper['title'])}/">{paper['title']}</a></h2>
<p class="paper-meta">{paper.get('authors', 'Unknown authors')} · {paper.get('venue', venue['title'] + ' ' + str(year))}</p>
<p>{paper.get('summary', '')}</p>
<div class="tag-row">{render_tags(paper.get('tags', []))}</div>
</article>"""


def render_tags(tags: list[str]) -> str:
    return "".join(f"<span>{tag}</span>" for tag in tags)


def strip_first_h1(markdown: str) -> str:
    lines = markdown.splitlines()
    if lines and lines[0].startswith("# "):
        return "\n".join(lines[1:]).lstrip()
    return markdown


def render_paper_page(venue: dict, year: int, paper: dict) -> str:
    tags_yaml = "\n".join(f"  - {tag}" for tag in paper.get("tags", [])) or "  - paper"
    tag_html = render_tags(paper.get("tags", []))
    source = paper.get("source_markdown")
    if source:
        source_path = ROOT / source
        body = strip_first_h1(source_path.read_text(encoding="utf-8"))
    else:
        body = paper.get("summary", "")

    return f"""---
title: "{paper['title']}"
description: "{paper.get('summary', '')}"
tags:
{tags_yaml}
---

# {paper['title']}

<div class="paper-hero">
<p class="paper-meta">{paper.get('authors', 'Unknown authors')} · {paper.get('venue', venue['title'] + ' ' + str(year))} · {venue['title']} / {year}</p>
<div class="tag-row">{tag_html}</div>
</div>

<article class="note-body" markdown>

{body}

</article>
"""


def render_embodied_overview(data: dict) -> str:
    block = data["embodied_llm"]
    section_cards = "\n".join(
        f"""<article class="area-card">
<p class="eyebrow">Frontier Watch</p>
<h2><a href="{section['id']}/">{section['title']}</a></h2>
<p>{section['description']}</p>
<span>{len(section['items'])} 条记录</span>
</article>"""
        for section in block["sections"]
    )
    return f"""---
title: "{block['title']}"
description: "{block['description']}"
---

# {block['title']}

<p class="lead">{block['description']}</p>

<section class="area-grid">
{section_cards}
</section>
"""


def render_embodied_section(section: dict) -> str:
    if section["items"]:
        items = "\n".join(
            f"""<article class="paper-card">
<h2>{item['title']}</h2>
<p>{item.get('summary', '')}</p>
</article>"""
            for item in section["items"]
        )
    else:
        items = """<article class="empty-card">
<h2>等待填充</h2>
<p>这里会记录具身智能、LLM agent、多模态模型和行业产品动态。先把入口留好，后续可以按日期持续补充。</p>
</article>"""

    return f"""---
title: "{section['title']}"
description: "{section['description']}"
---

# {section['title']}

<p class="lead">{section['description']}</p>

<section class="paper-list">
{items}
</section>
"""


def clean_generated_docs() -> None:
    ensure_dir(DOCS)
    for child in DOCS.iterdir():
        if child.name in PRESERVED_DOCS_DIRS:
            continue
        if child.name in PRESERVED_DOCS_FILES:
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def main() -> None:
    data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    clean_generated_docs()

    write(ROOT / "mkdocs.yml", build_mkdocs(data))
    write(DOCS / "index.md", render_home(data))
    write(DOCS / "tech-stack.md", render_tech_stack(data))
    write(DOCS / "offline_rl" / "index.md", render_offline_overview(data))
    write(DOCS / "embodied_llm" / "index.md", render_embodied_overview(data))

    for venue in data["offline_rl"]["venues"]:
        write(DOCS / "offline_rl" / venue["id"] / "index.md", render_venue_page(venue))
        for year_block in venue["years"]:
            year = year_block["year"]
            write(DOCS / "offline_rl" / venue["id"] / str(year) / "index.md", render_year_page(venue, year_block))
            for paper in year_block["papers"]:
                write(
                    DOCS / "offline_rl" / venue["id"] / str(year) / f"{slugify(paper['title'])}.md",
                    render_paper_page(venue, year, paper),
                )

    for section in data["embodied_llm"]["sections"]:
        write(DOCS / "embodied_llm" / f"{section['id']}.md", render_embodied_section(section))


if __name__ == "__main__":
    main()
