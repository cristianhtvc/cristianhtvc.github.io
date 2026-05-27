# 论文笔记站维护手册

这个仓库用于发布个人 GitHub Pages 站点：`https://cristianhtvc.github.io/`。

仓库根目录保存的是已经构建好的静态网页，`mkdocs-source/` 保存的是可维护源码。以后新增论文笔记时，优先修改 `mkdocs-source/` 中的内容，再重新构建并把结果发布到仓库根目录。

## 目录说明

```text
.
├── index.html                 # GitHub Pages 首页，自动生成
├── offline_rl/                # 已发布的离线强化学习网页
├── embodied_llm/              # 已发布的具身智能与 LLM 网页
├── assets/                    # 已发布的主题资源
├── stylesheets/               # 已发布的自定义 CSS
├── javascripts/               # 已发布的 MathJax 配置
├── mkdocs-source/             # 可维护源码
│   ├── data/papers.json       # 板块、会议、年份、论文元数据
│   ├── data/notes/            # 原始 Markdown 长文笔记
│   ├── docs/                  # MkDocs 页面与样式源文件
│   ├── scripts/generate_site.py
│   ├── mkdocs.yml
│   └── requirements.txt
└── MAINTENANCE.md             # 本维护手册
```

## 新增一篇 ORL 论文笔记

1. 把你的 Markdown 笔记复制到 `mkdocs-source/data/notes/` 下的合适目录，例如：

```text
mkdocs-source/data/notes/orl/iclr/2025/my-new-paper.md
```

2. 打开 `mkdocs-source/data/papers.json`，在对应会议和年份下新增一条论文记录。示例：

```json
{
  "title": "My New Offline RL Paper",
  "authors": "Author A, Author B",
  "venue": "ICLR 2025",
  "type": "精读",
  "status": "已整理",
  "summary": "一句话说明这篇论文解决什么问题、用了什么方法。",
  "tags": ["Offline RL", "Conservatism"],
  "source_markdown": "data/notes/orl/iclr/2025/my-new-paper.md"
}
```

3. 在 PowerShell 中进入源码目录：

```powershell
cd C:\Users\chenwy\Documents\GitHub\cristianhtvc.github.io\mkdocs-source
```

4. 安装依赖。如果已经装过，可以跳过：

```powershell
python -m pip install -r requirements.txt
```

5. 重新生成 MkDocs 页面并构建：

```powershell
python scripts\generate_site.py
python -m mkdocs build --clean --strict
```

6. 把构建结果发布到仓库根目录。推荐用 PowerShell：

```powershell
$repo = "C:\Users\chenwy\Documents\GitHub\cristianhtvc.github.io"
$built = "$repo\mkdocs-source\site"
Get-ChildItem -LiteralPath $repo -Force |
  Where-Object { $_.Name -notin @(".git", "mkdocs-source", "MAINTENANCE.md") } |
  Remove-Item -Recurse -Force
Copy-Item -Path "$built\*" -Destination $repo -Recurse -Force
New-Item -ItemType File -Force -Path "$repo\.nojekyll" | Out-Null
```

7. 用本地服务器预览发布结果：

```powershell
cd C:\Users\chenwy\Documents\GitHub\cristianhtvc.github.io
python -m http.server 8080
```

浏览器打开：

```text
http://127.0.0.1:8080/
```

8. 在 GitHub Desktop 中提交并推送：

- Summary 可以写：`Update paper notes site`
- 确认变更包含 `index.html`、相关目录和 `mkdocs-source/`
- 点击 `Commit to main`
- 点击 `Push origin`

## 更新已有笔记

1. 修改 `mkdocs-source/data/notes/` 中对应 Markdown 文件。
2. 如需改标题、作者、标签或摘要，同时修改 `mkdocs-source/data/papers.json`。
3. 重新执行：

```powershell
cd C:\Users\chenwy\Documents\GitHub\cristianhtvc.github.io\mkdocs-source
python scripts\generate_site.py
python -m mkdocs build --clean --strict
```

4. 按“新增一篇 ORL 论文笔记”的第 6-8 步重新发布。

## 公式显示检查

站点使用 `pymdownx.arithmatex + MathJax` 渲染公式。

推荐写法：

```markdown
行内公式：\\(Q^\pi(s,a)\\)

块级公式：

$$
J(\\pi)=\\mathbb{E}_{\\tau\\sim\\pi}\\left[\\sum_{t=0}^{T} r(s_t,a_t)\\right]
$$
```

发布前请检查：

- 页面没有出现原始的 `$$`。
- 长公式不会把页面撑出横向滚动条。
- 标题层级 `##`、`###`、`####` 能明显区分。

## 修改首页信息

首页标题、自我介绍、板块结构主要在：

```text
mkdocs-source/data/papers.json
```

常改字段：

- `site.name`
- `site.description`
- `site.tagline`
- `site.daily_goal`
- `offline_rl.venues`
- `embodied_llm.sections`

修改后重新生成和发布即可。

## 常见问题

### 访问页面时出现 404

通常是还没有把 `mkdocs-source/site/` 的构建结果复制到仓库根目录，或 GitHub Desktop 没有 push。

检查仓库根目录是否有：

```text
index.html
offline_rl/
assets/
stylesheets/
```

### GitHub Pages 仍显示旧页面

GitHub Pages 更新需要几十秒到几分钟。等待后使用 `Ctrl + F5` 强制刷新浏览器。

### GitHub Desktop 没有 Push origin 按钮

说明本地仓库还没有发布到 GitHub。点击 GitHub Desktop 顶部的 `Publish repository`，仓库名保持：

```text
cristianhtvc.github.io
```

发布后，浏览器访问：

```text
https://cristianhtvc.github.io/
```
