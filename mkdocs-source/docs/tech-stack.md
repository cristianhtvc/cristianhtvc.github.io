---
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
