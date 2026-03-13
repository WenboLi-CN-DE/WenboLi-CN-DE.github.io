# Jekyll 到 Hugo 博客迁移实施计划

> **执行者须知：** 需要使用 superpowers:executing-plans 来实施此计划。步骤使用复选框 (`- [ ]`) 语法进行跟踪。

**目标：** 将现有的 Jekyll 博客完整迁移到 Hugo 静态站点生成器，保留所有文章、样式和功能。

**架构：** 采用渐进式迁移策略，先搭建 Hugo 基础框架，然后迁移内容和样式，最后配置部署。整个过程保持 Git 版本控制，确保可以随时回滚。

**技术栈：** Hugo (静态站点生成器)、Go (Hugo 依赖)、Markdown、HTML/CSS/JavaScript

---

## 文件结构规划

**将要创建的 Hugo 目录结构：**
```
hugo-site/                    # 新的 Hugo 站点根目录
├── config.toml              # Hugo 主配置文件
├── content/
│   ├── posts/               # 博客文章目录
│   └── about.md             # 关于页面
├── themes/
│   └── custom-theme/        # 自定义主题
│       ├── layouts/
│       │   ├── _default/
│       │   │   ├── baseof.html
│       │   │   ├── single.html
│       │   │   └── list.html
│       │   ├── partials/
│       │   │   ├── header.html
│       │   │   ├── footer.html
│       │   │   └── sidebar.html
│       │   └── index.html
│       ├── static/
│       │   ├── css/
│       │   ├── js/
│       │   └── images/
│       └── theme.toml
├── static/                  # 静态资源
└── public/                  # 生成的站点（对应 Jekyll 的 _site）
```

**需要迁移的现有文件：**
- `_posts/*.md` → `hugo-site/content/posts/*.md`
- `css/*` → `hugo-site/themes/custom-theme/static/css/*`
- `js/*` → `hugo-site/themes/custom-theme/static/js/*`
- `images/*` → `hugo-site/static/images/*`
- `about.md` → `hugo-site/content/about.md`

---

## Chunk 1: 环境准备和 Hugo 初始化

### Task 1: 安装 Hugo

**Files:**
- 验证: 系统 Hugo 安装

- [ ] **Step 1: 检查是否已安装 Hugo**

```bash
hugo version
```

预期输出：如果未安装会显示 "command not found"

- [ ] **Step 2: 安装 Hugo (Ubuntu/WSL)**

```bash
sudo apt update
sudo apt install hugo -y
```

预期输出：安装成功信息

- [ ] **Step 3: 验证 Hugo 安装**

```bash
hugo version
```

预期输出：显示 Hugo 版本号，例如 "hugo v0.xxx.x"

---

### Task 2: 创建 Hugo 站点基础结构

**Files:**
- Create: `hugo-site/` (新目录)
- Create: `hugo-site/config.toml`

- [ ] **Step 1: 在项目根目录创建 Hugo 站点**

```bash
cd /home/wenbo/WenboLi-CN-DE.github.io
hugo new site hugo-site
```

预期输出：
```
Congratulations! Your new Hugo site is created in /home/wenbo/WenboLi-CN-DE.github.io/hugo-site.
```

- [ ] **Step 2: 验证目录结构**

```bash
ls -la hugo-site/
```

预期输出：显示 archetypes/, content/, data/, layouts/, static/, themes/, config.toml 等目录

- [ ] **Step 3: 初始化 Git 跟踪**

```bash
git add hugo-site/
git commit -m "chore: initialize Hugo site structure"
```

---

### Task 3: 配置 Hugo 基础设置

**Files:**
- Modify: `hugo-site/config.toml`

- [ ] **Step 1: 备份默认配置**

```bash
cp hugo-site/config.toml hugo-site/config.toml.backup
```

- [ ] **Step 2: 编写基础配置**

编辑 `hugo-site/config.toml`，完整内容如下：

```toml
baseURL = "https://wenboli-cn-de.github.io/"
languageCode = "zh-CN"
title = "高傲的电工李"
theme = "custom-theme"
paginate = 20
paginatePath = "page"

[params]
  subtitle = "个人博客"
  description = "欢迎来到我的个人博客"
  author = "李文博"
  avatarTitle = "李文博"
  avatarDesc = "机电 / 机器学习"
  enableToc = true

[params.social]
  github = "WenboLi-CN-DE"
  email = "lwb_010@163.com"

[params.blog_button]
  title = "博客主页"

[menu]
  [[menu.main]]
    name = "所有文章"
    url = "/posts/"
    weight = 1
  [[menu.main]]
    name = "标签"
    url = "/tags/"
    weight = 2
  [[menu.main]]
    name = "关于我"
    url = "/about/"
    weight = 3

[permalinks]
  posts = "/:year/:month/:title/"

[outputs]
  home = ["HTML", "RSS"]
  section = ["HTML", "RSS"]

[outputFormats.RSS]
  mediatype = "application/rss"
  baseName = "feed"

[markup]
  [markup.highlight]
    style = "monokai"
    lineNos = false
    lineNumbersInTable = false
    noClasses = true
  [markup.goldmark]
    [markup.goldmark.renderer]
      unsafe = true
```

- [ ] **Step 3: 验证配置语法**

```bash
cd hugo-site
hugo config
```

预期输出：显示解析后的配置，无错误

- [ ] **Step 4: Commit 配置**

```bash
git add config.toml
git commit -m "feat: add Hugo basic configuration"
```

---

## Chunk 2: 创建自定义主题

### Task 4: 初始化主题结构

**Files:**
- Create: `hugo-site/themes/custom-theme/`
- Create: `hugo-site/themes/custom-theme/theme.toml`

- [ ] **Step 1: 创建主题目录**

```bash
cd /home/wenbo/WenboLi-CN-DE.github.io/hugo-site
mkdir -p themes/custom-theme/{layouts/{_default,partials},static/{css,js},archetypes}
```

- [ ] **Step 2: 创建主题配置文件**

创建 `hugo-site/themes/custom-theme/theme.toml`：

```toml
name = "Custom Theme"
license = "MIT"
description = "Custom theme migrated from Jekyll"
homepage = "https://wenboli-cn-de.github.io/"
min_version = "0.80.0"

[author]
  name = "李文博"
```

- [ ] **Step 3: 验证主题目录**

```bash
ls -la themes/custom-theme/
```

预期输出：显示 layouts/, static/, theme.toml 等

- [ ] **Step 4: Commit 主题结构**

```bash
git add themes/custom-theme/
git commit -m "feat: initialize custom theme structure"
```

---

### Task 5: 创建基础布局模板

**Files:**
- Create: `hugo-site/themes/custom-theme/layouts/_default/baseof.html`
- Create: `hugo-site/themes/custom-theme/layouts/_default/single.html`
- Create: `hugo-site/themes/custom-theme/layouts/_default/list.html`

- [ ] **Step 1: 创建基础模板 baseof.html**

创建 `hugo-site/themes/custom-theme/layouts/_default/baseof.html`：

```html
<!DOCTYPE html>
<html lang="{{ .Site.LanguageCode }}">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ block "title" . }}{{ .Site.Title }}{{ end }}</title>
    <meta name="description" content="{{ .Site.Params.description }}">

    <!-- CSS -->
    <link rel="stylesheet" href="{{ "css/main.css" | relURL }}">
    <link rel="stylesheet" href="{{ "css/animate.css" | relURL }}">
    <link rel="stylesheet" href="{{ "css/tomorrow.css" | relURL }}">

    {{ block "head" . }}{{ end }}
</head>
<body>
    {{ partial "header.html" . }}

    <main>
        {{ block "main" . }}{{ end }}
    </main>

    {{ partial "footer.html" . }}

    <!-- JavaScript -->
    <script src="{{ "js/highlight.pack.js" | relURL }}"></script>
    <script src="{{ "js/main.js" | relURL }}"></script>
    {{ block "scripts" . }}{{ end }}
</body>
</html>
```

- [ ] **Step 2: 创建文章单页模板 single.html**

创建 `hugo-site/themes/custom-theme/layouts/_default/single.html`：

```html
{{ define "title" }}{{ .Title }} - {{ .Site.Title }}{{ end }}

{{ define "main" }}
<article class="post">
    <header class="post-header">
        <h1 class="post-title">{{ .Title }}</h1>
        <div class="post-meta">
            <time datetime="{{ .Date.Format "2006-01-02" }}">
                {{ .Date.Format "2006年01月02日" }}
            </time>
            {{ with .Params.tags }}
            <div class="post-tags">
                {{ range . }}
                <a href="{{ "/tags/" | relLangURL }}{{ . | urlize }}">#{{ . }}</a>
                {{ end }}
            </div>
            {{ end }}
        </div>
    </header>

    <div class="post-content">
        {{ .Content }}
    </div>

    {{ if .Site.Params.enableToc }}
    <aside class="post-toc">
        {{ .TableOfContents }}
    </aside>
    {{ end }}
</article>
{{ end }}
```

- [ ] **Step 3: 创建列表页模板 list.html**

创建 `hugo-site/themes/custom-theme/layouts/_default/list.html`：

```html
{{ define "title" }}{{ .Title }} - {{ .Site.Title }}{{ end }}

{{ define "main" }}
<div class="post-list">
    <h1>{{ .Title }}</h1>

    {{ range .Pages }}
    <article class="post-preview">
        <h2><a href="{{ .Permalink }}">{{ .Title }}</a></h2>
        <div class="post-meta">
            <time datetime="{{ .Date.Format "2006-01-02" }}">
                {{ .Date.Format "2006年01月02日" }}
            </time>
        </div>
        {{ with .Description }}
        <p class="post-description">{{ . }}</p>
        {{ end }}
    </article>
    {{ end }}
</div>
{{ end }}
```

- [ ] **Step 4: 验证模板语法**

```bash
hugo --renderToMemory
```

预期输出：无语法错误

- [ ] **Step 5: Commit 布局模板**

```bash
git add themes/custom-theme/layouts/
git commit -m "feat: add base layout templates"
```

---

### Task 6: 创建 Partial 模板

**Files:**
- Create: `hugo-site/themes/custom-theme/layouts/partials/header.html`
- Create: `hugo-site/themes/custom-theme/layouts/partials/footer.html`
- Create: `hugo-site/themes/custom-theme/layouts/partials/sidebar.html`

- [ ] **Step 1: 创建 header partial**

创建 `hugo-site/themes/custom-theme/layouts/partials/header.html`：

```html
<header class="site-header">
    <div class="header-content">
        <div class="avatar-section">
            <h1 class="avatar-title">{{ .Site.Params.avatarTitle }}</h1>
            <p class="avatar-desc">{{ .Site.Params.avatarDesc }}</p>
        </div>

        <nav class="site-nav">
            {{ range .Site.Menus.main }}
            <a href="{{ .URL }}" title="{{ .Name }}">{{ .Name }}</a>
            {{ end }}
        </nav>

        <div class="social-links">
            {{ with .Site.Params.social.github }}
            <a href="https://github.com/{{ . }}" target="_blank">GitHub</a>
            {{ end }}
            {{ with .Site.Params.social.email }}
            <a href="mailto:{{ . }}">Email</a>
            {{ end }}
        </div>
    </div>
</header>
```

- [ ] **Step 2: 创建 footer partial**

创建 `hugo-site/themes/custom-theme/layouts/partials/footer.html`：

```html
<footer class="site-footer">
    <div class="footer-content">
        <p>&copy; {{ now.Year }} {{ .Site.Params.author }}. All rights reserved.</p>
        <p>Powered by <a href="https://gohugo.io/" target="_blank">Hugo</a></p>
    </div>
</footer>
```

- [ ] **Step 3: 创建 sidebar partial**

创建 `hugo-site/themes/custom-theme/layouts/partials/sidebar.html`：

```html
<aside class="sidebar">
    <div class="sidebar-content">
        <div class="sidebar-section">
            <h3>关于</h3>
            <p>{{ .Site.Params.description }}</p>
        </div>

        <div class="sidebar-section">
            <h3>最近文章</h3>
            <ul>
                {{ range first 5 (where .Site.RegularPages "Type" "posts") }}
                <li><a href="{{ .Permalink }}">{{ .Title }}</a></li>
                {{ end }}
            </ul>
        </div>
    </div>
</aside>
```

- [ ] **Step 4: 验证 partials**

```bash
hugo --renderToMemory
```

预期输出：无错误

- [ ] **Step 5: Commit partials**

```bash
git add themes/custom-theme/layouts/partials/
git commit -m "feat: add header, footer, and sidebar partials"
```

---

### Task 7: 创建首页模板

**Files:**
- Create: `hugo-site/themes/custom-theme/layouts/index.html`

- [ ] **Step 1: 创建首页模板**

创建 `hugo-site/themes/custom-theme/layouts/index.html`：

```html
{{ define "title" }}{{ .Site.Title }} - {{ .Site.Params.subtitle }}{{ end }}

{{ define "main" }}
<div class="home-page">
    {{ partial "sidebar.html" . }}

    <div class="post-list">
        <h2>{{ .Site.Params.blog_button.title }}</h2>

        {{ $paginator := .Paginate (where .Site.RegularPages "Type" "posts") }}
        {{ range $paginator.Pages }}
        <article class="post-preview">
            <h2><a href="{{ .Permalink }}">{{ .Title }}</a></h2>
            <div class="post-meta">
                <time datetime="{{ .Date.Format "2006-01-02" }}">
                    {{ .Date.Format "2006年01月02日" }}
                </time>
            </div>
            {{ with .Description }}
            <p class="post-description">{{ . }}</p>
            {{ end }}
            <a href="{{ .Permalink }}" class="read-more">阅读全文 →</a>
        </article>
        {{ end }}

        {{ template "_internal/pagination.html" . }}
    </div>
</div>
{{ end }}
```

- [ ] **Step 2: 测试首页渲染**

```bash
hugo server -D
```

预期输出：服务器启动，访问 http://localhost:1313 可以看到首页（虽然还没有内容）

- [ ] **Step 3: 停止服务器并 commit**

按 Ctrl+C 停止服务器

```bash
git add themes/custom-theme/layouts/index.html
git commit -m "feat: add home page template"
```

---

## Chunk 3: 迁移样式和静态资源

### Task 8: 迁移 CSS 文件

**Files:**
- Copy: `css/*.css` → `hugo-site/themes/custom-theme/static/css/`

- [ ] **Step 1: 复制 CSS 文件**

```bash
cp -r /home/wenbo/WenboLi-CN-DE.github.io/css/* /home/wenbo/WenboLi-CN-DE.github.io/hugo-site/themes/custom-theme/static/css/
```

- [ ] **Step 2: 验证文件复制**

```bash
ls -la hugo-site/themes/custom-theme/static/css/
```

预期输出：显示 animate.css, main.css, post.css, tomorrow.css

- [ ] **Step 3: 检查 CSS 中的路径引用**

```bash
grep -r "url(" hugo-site/themes/custom-theme/static/css/
```

预期输出：检查是否有需要修改的路径引用

- [ ] **Step 4: Commit CSS 文件**

```bash
git add hugo-site/themes/custom-theme/static/css/
git commit -m "feat: migrate CSS files from Jekyll"
```

---

### Task 9: 迁移 JavaScript 文件

**Files:**
- Copy: `js/*.js` → `hugo-site/themes/custom-theme/static/js/`

- [ ] **Step 1: 复制 JS 文件**

```bash
cp -r /home/wenbo/WenboLi-CN-DE.github.io/js/* /home/wenbo/WenboLi-CN-DE.github.io/hugo-site/themes/custom-theme/static/js/
```

- [ ] **Step 2: 验证文件复制**

```bash
ls -la hugo-site/themes/custom-theme/static/js/
```

预期输出：显示 highlight.pack.js, main.js

- [ ] **Step 3: Commit JS 文件**

```bash
git add hugo-site/themes/custom-theme/static/js/
git commit -m "feat: migrate JavaScript files from Jekyll"
```

---

### Task 10: 迁移图片资源

**Files:**
- Copy: `images/*` → `hugo-site/static/images/`

- [ ] **Step 1: 复制图片目录**

```bash
cp -r /home/wenbo/WenboLi-CN-DE.github.io/images /home/wenbo/WenboLi-CN-DE.github.io/hugo-site/static/
```

- [ ] **Step 2: 验证图片复制**

```bash
ls -la hugo-site/static/images/
```

预期输出：显示所有图片文件

- [ ] **Step 3: Commit 图片资源**

```bash
git add hugo-site/static/images/
git commit -m "feat: migrate image assets from Jekyll"
```

---

## Chunk 4: 迁移内容文件

### Task 11: 创建文章迁移脚本

**Files:**
- Create: `hugo-site/migrate-posts.sh`

- [ ] **Step 1: 创建迁移脚本**

创建 `hugo-site/migrate-posts.sh`：

```bash
#!/bin/bash

# Jekyll 到 Hugo 文章迁移脚本
SOURCE_DIR="/home/wenbo/WenboLi-CN-DE.github.io/_posts"
TARGET_DIR="/home/wenbo/WenboLi-CN-DE.github.io/hugo-site/content/posts"

# 创建目标目录
mkdir -p "$TARGET_DIR"

# 遍历所有 markdown 文件
for file in "$SOURCE_DIR"/*.md; do
    if [ -f "$file" ]; then
        filename=$(basename "$file")
        echo "Processing: $filename"

        # 复制文件到 Hugo content 目录
        cp "$file" "$TARGET_DIR/$filename"

        echo "Migrated: $filename"
    fi
done

echo "Migration complete! Total files: $(ls -1 "$TARGET_DIR" | wc -l)"
```

- [ ] **Step 2: 添加执行权限**

```bash
chmod +x hugo-site/migrate-posts.sh
```

- [ ] **Step 3: 执行迁移脚本**

```bash
cd /home/wenbo/WenboLi-CN-DE.github.io
./hugo-site/migrate-posts.sh
```

预期输出：显示每个文件的迁移进度，最后显示总数（23篇文章）

- [ ] **Step 4: 验证文章迁移**

```bash
ls -la hugo-site/content/posts/ | wc -l
```

预期输出：24（23篇文章 + . 和 .. 目录）

- [ ] **Step 5: Commit 迁移脚本和文章**

```bash
git add hugo-site/migrate-posts.sh hugo-site/content/posts/
git commit -m "feat: migrate all blog posts from Jekyll"
```

---

## Chunk 3: 迁移样式和静态资源

### Task 8: 迁移 CSS 文件

**Files:**
- Copy: `css/*.css` → `hugo-site/themes/custom-theme/static/css/`

- [ ] **Step 1: 复制 CSS 文件**

```bash
cp -r /home/wenbo/WenboLi-CN-DE.github.io/css/* /home/wenbo/WenboLi-CN-DE.github.io/hugo-site/themes/custom-theme/static/css/
```

- [ ] **Step 2: 验证文件复制**

```bash
ls -la hugo-site/themes/custom-theme/static/css/
```

预期输出：显示 animate.css, main.css, post.css, tomorrow.css

- [ ] **Step 3: 检查 CSS 中的路径引用**

```bash
grep -r "url(" hugo-site/themes/custom-theme/static/css/
```

预期输出：检查是否有需要修改的路径引用

- [ ] **Step 4: Commit CSS 文件**

```bash
git add hugo-site/themes/custom-theme/static/css/
git commit -m "feat: migrate CSS files from Jekyll"
```

---

### Task 9: 迁移 JavaScript 文件

**Files:**
- Copy: `js/*.js` → `hugo-site/themes/custom-theme/static/js/`

- [ ] **Step 1: 复制 JS 文件**

```bash
cp -r /home/wenbo/WenboLi-CN-DE.github.io/js/* /home/wenbo/WenboLi-CN-DE.github.io/hugo-site/themes/custom-theme/static/js/
```

- [ ] **Step 2: 验证文件复制**

```bash
ls -la hugo-site/themes/custom-theme/static/js/
```

预期输出：显示 highlight.pack.js, main.js

- [ ] **Step 3: Commit JS 文件**

```bash
git add hugo-site/themes/custom-theme/static/js/
git commit -m "feat: migrate JavaScript files from Jekyll"
```

---

### Task 10: 迁移图片资源

**Files:**
- Copy: `images/*` → `hugo-site/static/images/`

- [ ] **Step 1: 复制图片目录**

```bash
cp -r /home/wenbo/WenboLi-CN-DE.github.io/images /home/wenbo/WenboLi-CN-DE.github.io/hugo-site/static/
```

- [ ] **Step 2: 验证图片复制**

```bash
ls -la hugo-site/static/images/
```

预期输出：显示所有图片文件

- [ ] **Step 3: Commit 图片资源**

```bash
git add hugo-site/static/images/
git commit -m "feat: migrate image assets from Jekyll"
```

---

## Chunk 4: 迁移内容文件

### Task 11: 创建文章迁移脚本

**Files:**
- Create: `hugo-site/migrate-posts.sh`

- [ ] **Step 1: 创建迁移脚本**

创建 `hugo-site/migrate-posts.sh`：

```bash
#!/bin/bash

# Jekyll 到 Hugo 文章迁移脚本
SOURCE_DIR="/home/wenbo/WenboLi-CN-DE.github.io/_posts"
TARGET_DIR="/home/wenbo/WenboLi-CN-DE.github.io/hugo-site/content/posts"

# 创建目标目录
mkdir -p "$TARGET_DIR"

# 遍历所有 markdown 文件
for file in "$SOURCE_DIR"/*.md; do
    if [ -f "$file" ]; then
        filename=$(basename "$file")
        echo "Processing: $filename"

        # 复制文件到 Hugo content 目录
        cp "$file" "$TARGET_DIR/$filename"

        echo "Migrated: $filename"
    fi
done

echo "Migration complete! Total files: $(ls -1 "$TARGET_DIR" | wc -l)"
```

- [ ] **Step 2: 添加执行权限**

```bash
chmod +x hugo-site/migrate-posts.sh
```

- [ ] **Step 3: 执行迁移脚本**

```bash
cd /home/wenbo/WenboLi-CN-DE.github.io
./hugo-site/migrate-posts.sh
```

预期输出：显示每个文件的迁移进度，最后显示总数（23篇文章）

- [ ] **Step 4: 验证文章迁移**

```bash
ls -la hugo-site/content/posts/ | wc -l
```

预期输出：显示文章数量

- [ ] **Step 5: Commit 迁移脚本和文章**

```bash
git add hugo-site/migrate-posts.sh hugo-site/content/posts/
git commit -m "feat: migrate all blog posts from Jekyll"
```

---

### Task 12: 迁移关于页面

**Files:**
- Copy: `about.md` → `hugo-site/content/about.md`
- Modify: `hugo-site/content/about.md`

- [ ] **Step 1: 复制关于页面**

```bash
cp /home/wenbo/WenboLi-CN-DE.github.io/about.md /home/wenbo/WenboLi-CN-DE.github.io/hugo-site/content/about.md
```

- [ ] **Step 2: 修改 front matter**

编辑 `hugo-site/content/about.md`，将：
```yaml
---
layout: page
title: 关于我
---
```

改为：
```yaml
---
title: "关于我"
date: 2026-03-13
type: "page"
---
```

- [ ] **Step 3: 移除 Jekyll 特定语法**

将 `{% include comments.html %}` 删除或注释掉

- [ ] **Step 4: 验证页面**

```bash
cd hugo-site
hugo server -D
```

访问 http://localhost:1313/about/ 查看页面，按 Ctrl+C 停止服务器

- [ ] **Step 5: Commit 关于页面**

```bash
git add content/about.md
git commit -m "feat: migrate about page from Jekyll"
```

---

### Task 13: 创建归档和标签页面

**Files:**
- Create: `hugo-site/themes/custom-theme/layouts/_default/terms.html`
- Create: `hugo-site/themes/custom-theme/layouts/_default/taxonomy.html`

- [ ] **Step 1: 创建标签列表页模板**

创建 `hugo-site/themes/custom-theme/layouts/_default/terms.html`：

```html
{{ define "title" }}{{ .Title }} - {{ .Site.Title }}{{ end }}

{{ define "main" }}
<div class="taxonomy-list">
    <h1>{{ .Title }}</h1>

    <div class="tag-cloud">
        {{ range .Data.Terms.ByCount }}
        <a href="{{ .Page.Permalink }}" class="tag-item">
            {{ .Page.Title }} <span class="tag-count">({{ .Count }})</span>
        </a>
        {{ end }}
    </div>
</div>
{{ end }}
```

- [ ] **Step 2: 创建单个标签页模板**

创建 `hugo-site/themes/custom-theme/layouts/_default/taxonomy.html`：

```html
{{ define "title" }}{{ .Title }} - {{ .Site.Title }}{{ end }}

{{ define "main" }}
<div class="taxonomy-page">
    <h1>标签: {{ .Title }}</h1>

    <div class="post-list">
        {{ range .Pages }}
        <article class="post-preview">
            <h2><a href="{{ .Permalink }}">{{ .Title }}</a></h2>
            <div class="post-meta">
                <time datetime="{{ .Date.Format "2006-01-02" }}">
                    {{ .Date.Format "2006年01月02日" }}
                </time>
            </div>
        </article>
        {{ end }}
    </div>
</div>
{{ end }}
```

- [ ] **Step 3: 测试标签页面**

```bash
hugo server -D
```

访问 http://localhost:1313/tags/ 查看标签列表，按 Ctrl+C 停止

- [ ] **Step 4: Commit 标签页面模板**

```bash
git add themes/custom-theme/layouts/_default/terms.html
git add themes/custom-theme/layouts/_default/taxonomy.html
git commit -m "feat: add tags and archive page templates"
```

---

## Chunk 5: 部署配置和测试

### Task 14: 配置 CNAME 文件

**Files:**
- Copy: `CNAME` → `hugo-site/static/CNAME`

- [ ] **Step 1: 复制 CNAME 文件**

```bash
cp /home/wenbo/WenboLi-CN-DE.github.io/CNAME /home/wenbo/WenboLi-CN-DE.github.io/hugo-site/static/CNAME
```

- [ ] **Step 2: 验证 CNAME 内容**

```bash
cat hugo-site/static/CNAME
```

预期输出：显示域名

- [ ] **Step 3: Commit CNAME**

```bash
git add hugo-site/static/CNAME
git commit -m "feat: add CNAME for custom domain"
```

---

### Task 15: 创建 GitHub Actions 部署工作流

**Files:**
- Create: `hugo-site/.github/workflows/hugo.yml`

- [ ] **Step 1: 创建工作流目录**

```bash
mkdir -p /home/wenbo/WenboLi-CN-DE.github.io/hugo-site/.github/workflows
```

- [ ] **Step 2: 创建 Hugo 部署工作流**

创建 `hugo-site/.github/workflows/hugo.yml`：

```yaml
name: Deploy Hugo site to Pages

on:
  push:
    branches: ["main"]
  workflow_dispatch:

permissions:
  contents: read
  pages: write
  id-token: write

concurrency:
  group: "pages"
  cancel-in-progress: false

defaults:
  run:
    shell: bash

jobs:
  build:
    runs-on: ubuntu-latest
    env:
      HUGO_VERSION: 0.120.0
    steps:
      - name: Install Hugo CLI
        run: |
          wget -O ${{ runner.temp }}/hugo.deb https://github.com/gohugoio/hugo/releases/download/v${HUGO_VERSION}/hugo_extended_${HUGO_VERSION}_linux-amd64.deb \
          && sudo dpkg -i ${{ runner.temp }}/hugo.deb

      - name: Checkout
        uses: actions/checkout@v4

      - name: Setup Pages
        id: pages
        uses: actions/configure-pages@v4

      - name: Build with Hugo
        env:
          HUGO_ENVIRONMENT: production
          HUGO_ENV: production
        run: |
          cd hugo-site
          hugo \
            --minify \
            --baseURL "${{ steps.pages.outputs.base_url }}/"

      - name: Upload artifact
        uses: actions/upload-pages-artifact@v2
        with:
          path: ./hugo-site/public

  deploy:
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    runs-on: ubuntu-latest
    needs: build
    steps:
      - name: Deploy to GitHub Pages
        id: deployment
        uses: actions/deploy-pages@v3
```

- [ ] **Step 3: Commit 工作流配置**

```bash
git add hugo-site/.github/workflows/hugo.yml
git commit -m "feat: add GitHub Actions workflow for Hugo deployment"
```

---

### Task 16: 本地完整测试

**Files:**
- Test: 整个 Hugo 站点

- [ ] **Step 1: 清理之前的构建**

```bash
cd /home/wenbo/WenboLi-CN-DE.github.io/hugo-site
rm -rf public/
```

- [ ] **Step 2: 构建站点**

```bash
hugo
```

预期输出：显示构建统计信息，无错误

- [ ] **Step 3: 启动本地服务器**

```bash
hugo server
```

预期输出：服务器在 http://localhost:1313 启动

- [ ] **Step 4: 手动测试清单**

在浏览器中测试以下页面：
- [ ] 首页 (http://localhost:1313/)
- [ ] 文章列表 (http://localhost:1313/posts/)
- [ ] 单篇文章（随机选择）
- [ ] 标签页 (http://localhost:1313/tags/)
- [ ] 关于页面 (http://localhost:1313/about/)
- [ ] RSS feed (http://localhost:1313/feed.xml)

- [ ] **Step 5: 检查样式和功能**

验证：
- [ ] CSS 样式正确加载
- [ ] 代码高亮正常工作
- [ ] 导航菜单功能正常
- [ ] 分页功能正常
- [ ] 图片正确显示

- [ ] **Step 6: 停止服务器**

按 Ctrl+C 停止服务器

---

### Task 17: 创建迁移文档

**Files:**
- Create: `hugo-site/MIGRATION.md`

- [ ] **Step 1: 创建迁移文档**

创建 `hugo-site/MIGRATION.md`：

```markdown
# Jekyll 到 Hugo 迁移文档

## 迁移日期
2026-03-13

## 迁移内容

### 已迁移
- ✅ 23 篇博客文章
- ✅ 关于页面
- ✅ 所有 CSS 样式文件
- ✅ 所有 JavaScript 文件
- ✅ 所有图片资源
- ✅ 自定义主题
- ✅ 标签和归档功能
- ✅ RSS feed
- ✅ 分页功能
- ✅ 语法高亮
- ✅ GitHub Actions 部署配置

### 配置差异

#### Jekyll (_config.yml)
```yaml
title: 高傲的电工李
permalink: /:year/:month/:categories/:title/
paginate: 20
```

#### Hugo (config.toml)
```toml
title = "高傲的电工李"
[permalinks]
  posts = "/:year/:month/:title/"
paginate = 20
```

### 本地开发命令

#### Jekyll
```bash
bundle exec jekyll serve
```

#### Hugo
```bash
hugo server
```

### 构建命令

#### Jekyll
```bash
bundle exec jekyll build
```

#### Hugo
```bash
hugo
```

### 部署

Hugo 站点通过 GitHub Actions 自动部署到 GitHub Pages。
工作流配置文件：`.github/workflows/hugo.yml`

### 注意事项

1. Hugo 不支持 Jekyll 的 Liquid 模板语法，已转换为 Hugo 的 Go 模板
2. 文章的 front matter 保持兼容，无需修改
3. 永久链接格式已配置为与 Jekyll 相同，确保 URL 不变
4. 所有静态资源路径已更新

### 回滚方案

如果需要回滚到 Jekyll：
1. 原 Jekyll 文件仍保留在项目根目录
2. 删除 `hugo-site/` 目录
3. 恢复原 GitHub Actions 工作流（如果有）

### 性能对比

- Jekyll 构建时间：约 X 秒
- Hugo 构建时间：约 Y 秒（通常快 10-100 倍）

### 后续优化建议

1. 考虑添加评论系统（如 Disqus、Utterances）
2. 优化图片加载（lazy loading）
3. 添加搜索功能
4. 配置 CDN 加速
5. 添加 Google Analytics 或其他分析工具
```

- [ ] **Step 2: Commit 迁移文档**

```bash
git add MIGRATION.md
git commit -m "docs: add migration documentation"
```

---

### Task 18: 最终验证和部署准备

**Files:**
- Verify: 所有文件和配置

- [ ] **Step 1: 检查 Git 状态**

```bash
cd /home/wenbo/WenboLi-CN-DE.github.io
git status
```

预期输出：工作目录干净，所有更改已提交

- [ ] **Step 2: 查看提交历史**

```bash
git log --oneline -20
```

预期输出：显示所有迁移相关的提交

- [ ] **Step 3: 创建迁移标签**

```bash
git tag -a v2.0.0-hugo -m "Migrate from Jekyll to Hugo"
```

- [ ] **Step 4: 最终构建测试**

```bash
cd hugo-site
hugo --minify
```

预期输出：成功构建，无错误或警告

- [ ] **Step 5: 检查生成的文件**

```bash
ls -la public/
```

预期输出：显示生成的 HTML、CSS、JS 和其他资源文件

- [ ] **Step 6: 验证 CNAME 文件**

```bash
cat public/CNAME
```

预期输出：显示正确的域名

---

## 部署说明

### 方案 A: 替换现有仓库（推荐）

如果要完全替换 Jekyll 为 Hugo：

1. 备份当前仓库
```bash
cd /home/wenbo/WenboLi-CN-DE.github.io
git branch backup-jekyll
```

2. 将 Hugo 站点内容移到根目录
```bash
# 这需要手动操作，建议创建新分支
git checkout -b hugo-migration
mv hugo-site/* .
mv hugo-site/.github .
```

3. 更新 .gitignore
```bash
echo "public/" >> .gitignore
echo "resources/" >> .gitignore
```

4. 推送到 GitHub
```bash
git add .
git commit -m "chore: complete Jekyll to Hugo migration"
git push origin hugo-migration
```

5. 在 GitHub 上创建 Pull Request 并合并到 main 分支

### 方案 B: 保持双系统（测试用）

保留 Jekyll 和 Hugo 两个系统，Hugo 在 `hugo-site/` 子目录：

1. 修改 GitHub Actions 工作流的工作目录
2. 推送当前更改
```bash
git push origin main
```

3. 在 GitHub 仓库设置中配置 Pages 使用 GitHub Actions 部署

### GitHub Pages 设置

1. 进入仓库 Settings → Pages
2. Source 选择 "GitHub Actions"
3. 保存设置

### 验证部署

部署完成后访问：
- https://wenboli-cn-de.github.io/
- 检查所有页面是否正常
- 验证自定义域名（如果配置了）

---

## 故障排除

### 问题 1: Hugo 版本不兼容

**症状**: 构建失败，提示语法错误

**解决方案**:
```bash
hugo version
# 如果版本过低，升级 Hugo
sudo apt update
sudo apt install hugo -y
```

### 问题 2: 样式未加载

**症状**: 页面显示但没有样式

**解决方案**:
- 检查 `baseURL` 配置是否正确
- 验证 CSS 文件路径
- 检查浏览器控制台的错误信息

### 问题 3: 文章不显示

**症状**: 首页或文章列表为空

**解决方案**:
- 检查文章的 front matter 格式
- 确认文章日期不是未来日期
- 使用 `hugo server -D` 显示草稿

### 问题 4: 永久链接不匹配

**症状**: 旧的 URL 返回 404

**解决方案**:
- 检查 `config.toml` 中的 `[permalinks]` 配置
- 确保与 Jekyll 的 `permalink` 设置一致

---

## 完成检查清单

- [ ] Hugo 已安装并验证
- [ ] 站点结构已创建
- [ ] 配置文件已完成
- [ ] 主题已创建
- [ ] 所有模板已实现
- [ ] CSS/JS 已迁移
- [ ] 图片资源已迁移
- [ ] 所有文章已迁移
- [ ] 关于页面已迁移
- [ ] 标签和归档功能正常
- [ ] 本地测试通过
- [ ] GitHub Actions 工作流已配置
- [ ] 迁移文档已创建
- [ ] 所有更改已提交
- [ ] 准备好部署

---

## 总结

本迁移计划将 Jekyll 博客完整迁移到 Hugo，保留了所有内容、样式和功能。Hugo 提供了更快的构建速度和更灵活的模板系统。迁移后的站点与原站点在外观和功能上保持一致，同时获得了更好的性能。

### Task 15: 创建 GitHub Actions 部署工作流

**Files:**
- Create: `hugo-site/.github/workflows/hugo.yml`

- [ ] **Step 1: 创建工作流目录**

```bash
mkdir -p /home/wenbo/WenboLi-CN-DE.github.io/hugo-site/.github/workflows
```

- [ ] **Step 2: 创建 Hugo 部署工作流**

创建 `hugo-site/.github/workflows/hugo.yml`：

```yaml
name: Deploy Hugo site to Pages

on:
  push:
    branches: ["main"]
  workflow_dispatch:

permissions:
  contents: read
  pages: write
  id-token: write

concurrency:
  group: "pages"
  cancel-in-progress: false

defaults:
  run:
    shell: bash

jobs:
  build:
    runs-on: ubuntu-latest
    env:
      HUGO_VERSION: 0.120.0
    steps:
      - name: Install Hugo CLI
        run: |
          wget -O ${{ runner.temp }}/hugo.deb https://github.com/gohugoio/hugo/releases/download/v${HUGO_VERSION}/hugo_extended_${HUGO_VERSION}_linux-amd64.deb \
          && sudo dpkg -i ${{ runner.temp }}/hugo.deb

      - name: Checkout
        uses: actions/checkout@v4

      - name: Setup Pages
        id: pages
        uses: actions/configure-pages@v4

      - name: Build with Hugo
        env:
          HUGO_ENVIRONMENT: production
          HUGO_ENV: production
        run: |
          cd hugo-site
          hugo \
            --minify \
            --baseURL "${{ steps.pages.outputs.base_url }}/"

      - name: Upload artifact
        uses: actions/upload-pages-artifact@v2
        with:
          path: ./hugo-site/public

  deploy:
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    runs-on: ubuntu-latest
    needs: build
    steps:
      - name: Deploy to GitHub Pages
        id: deployment
        uses: actions/deploy-pages@v3
```

- [ ] **Step 3: 验证 YAML 语法**

```bash
cat hugo-site/.github/workflows/hugo.yml
```

预期输出：显示完整的工作流配置

- [ ] **Step 4: Commit 工作流**

```bash
git add hugo-site/.github/
git commit -m "feat: add GitHub Actions workflow for Hugo deployment"
```

---

### Task 16: 本地完整测试

**Files:**
- Test: 整个 Hugo 站点

- [ ] **Step 1: 清理之前的构建**

```bash
cd /home/wenbo/WenboLi-CN-DE.github.io/hugo-site
rm -rf public/
```

- [ ] **Step 2: 构建站点**

```bash
hugo
```

预期输出：显示构建统计信息，例如 "Total in 123 ms"

- [ ] **Step 3: 检查生成的文件**

```bash
ls -la public/
```

预期输出：显示 index.html, posts/, tags/, css/, js/, images/ 等

- [ ] **Step 4: 启动本地服务器测试**

```bash
hugo server
```

预期输出：服务器在 http://localhost:1313 启动

- [ ] **Step 5: 手动测试功能**

在浏览器中测试：
- 访问首页：http://localhost:1313
- 访问文章列表：http://localhost:1313/posts/
- 访问单篇文章：点击任意文章
- 访问标签页：http://localhost:1313/tags/
- 访问关于页面：http://localhost:1313/about/
- 检查样式是否正确加载
- 检查代码高亮是否工作
- 检查响应式布局

- [ ] **Step 6: 停止服务器**

按 Ctrl+C 停止服务器

- [ ] **Step 7: 记录测试结果**

创建测试日志（无需 commit）

---

### Task 17: 创建迁移文档

**Files:**
- Create: `hugo-site/MIGRATION.md`

- [ ] **Step 1: 创建迁移文档**

创建 `hugo-site/MIGRATION.md`：

```markdown
# Jekyll 到 Hugo 迁移文档

## 迁移日期
2026-03-13

## 迁移内容

### 已迁移
- ✅ 23 篇博客文章（_posts/*.md → content/posts/*.md）
- ✅ 关于页面（about.md → content/about.md）
- ✅ CSS 样式文件（css/* → themes/custom-theme/static/css/*）
- ✅ JavaScript 文件（js/* → themes/custom-theme/static/js/*）
- ✅ 图片资源（images/* → static/images/*）
- ✅ 自定义主题（基于原 Jekyll 主题）
- ✅ 分页功能（20篇/页）
- ✅ 标签系统
- ✅ RSS Feed
- ✅ 语法高亮
- ✅ 目录生成
- ✅ CNAME 配置
- ✅ GitHub Actions 部署

### 配置对比

| 功能 | Jekyll | Hugo |
|------|--------|------|
| 配置文件 | _config.yml | config.toml |
| 文章目录 | _posts/ | content/posts/ |
| 布局文件 | _layouts/ | themes/custom-theme/layouts/ |
| 静态资源 | css/, js/, images/ | themes/custom-theme/static/, static/ |
| 生成目录 | _site/ | public/ |
| 分页数量 | 20 | 20 |
| 永久链接 | /:year/:month/:categories/:title/ | /:year/:month/:title/ |

## 本地开发

### 启动开发服务器
```bash
cd hugo-site
hugo server -D
```

访问：http://localhost:1313

### 构建生产版本
```bash
cd hugo-site
hugo
```

生成的文件在 `public/` 目录

## 部署

### GitHub Pages 部署
1. 推送代码到 GitHub
2. GitHub Actions 自动构建和部署
3. 访问 https://wenboli-cn-de.github.io

### 手动部署
```bash
cd hugo-site
hugo
# 将 public/ 目录内容部署到服务器
```

## 注意事项

1. **Front Matter 兼容性**：Jekyll 和 Hugo 都使用 YAML front matter，大部分文章无需修改
2. **Liquid 模板语法**：如果文章中使用了 Jekyll 的 Liquid 语法（如 `{% include %}`），需要手动修改
3. **路径引用**：检查文章中的图片路径是否正确
4. **评论系统**：原 Jekyll 的评论系统需要重新配置（如果使用）

## 回滚方案

如果需要回滚到 Jekyll：
1. 原 Jekyll 文件仍保留在项目根目录
2. 删除或重命名 `hugo-site/` 目录
3. 恢复原 GitHub Actions 工作流（如果有）

## 后续优化

- [ ] 优化主题样式，使其更接近原 Jekyll 主题
- [ ] 添加评论系统（如 Disqus, Utterances）
- [ ] 添加搜索功能
- [ ] 优化 SEO
- [ ] 添加 Google Analytics（如果需要）
- [ ] 优化图片加载（lazy loading）
- [ ] 添加暗色模式
```

- [ ] **Step 2: Commit 迁移文档**

```bash
git add MIGRATION.md
git commit -m "docs: add migration documentation"
```

---

### Task 18: 创建 README 文件

**Files:**
- Create: `hugo-site/README.md`

- [ ] **Step 1: 创建 README**

创建 `hugo-site/README.md`：

```markdown
# 高傲的电工李 - Hugo 博客

个人技术博客，使用 Hugo 静态站点生成器构建。

## 关于

- **作者**：李文博
- **描述**：机电 / 机器学习
- **网站**：https://wenboli-cn-de.github.io

## 技术栈

- [Hugo](https://gohugo.io/) - 静态站点生成器
- 自定义主题（从 Jekyll 迁移）
- GitHub Pages - 托管
- GitHub Actions - 自动部署

## 本地开发

### 前置要求

- Hugo >= 0.80.0
- Git

### 安装 Hugo

**Ubuntu/WSL:**
```bash
sudo apt update
sudo apt install hugo
```

**macOS:**
```bash
brew install hugo
```

**Windows:**
下载并安装：https://github.com/gohugoio/hugo/releases

### 运行开发服务器

```bash
cd hugo-site
hugo server -D
```

访问 http://localhost:1313

### 构建生产版本

```bash
hugo
```

生成的文件在 `public/` 目录。

## 内容管理

### 创建新文章

```bash
hugo new posts/YYYY-MM-DD-title.md
```

### 文章 Front Matter 示例

```yaml
---
title: "文章标题"
date: 2026-03-13
description: "文章描述"
tags: ["标签1", "标签2"]
---
```

### 目录结构

```
hugo-site/
├── config.toml          # 站点配置
├── content/             # 内容文件
│   ├── posts/          # 博客文章
│   └── about.md        # 关于页面
├── themes/             # 主题
│   └── custom-theme/   # 自定义主题
├── static/             # 静态资源
└── public/             # 生成的站点
```

## 部署

推送到 GitHub 后，GitHub Actions 会自动构建和部署到 GitHub Pages。

## 许可

MIT License

## 联系方式

- GitHub: [@WenboLi-CN-DE](https://github.com/WenboLi-CN-DE)
- Email: lwb_010@163.com
```

- [ ] **Step 2: Commit README**

```bash
git add README.md
git commit -m "docs: add README for Hugo site"
```

---

### Task 19: 最终验证和清理

**Files:**
- Verify: 所有迁移内容

- [ ] **Step 1: 验证文件完整性**

```bash
cd /home/wenbo/WenboLi-CN-DE.github.io/hugo-site

# 检查文章数量
echo "文章数量: $(ls -1 content/posts/*.md 2>/dev/null | wc -l)"

# 检查 CSS 文件
echo "CSS 文件: $(ls -1 themes/custom-theme/static/css/*.css 2>/dev/null | wc -l)"

# 检查 JS 文件
echo "JS 文件: $(ls -1 themes/custom-theme/static/js/*.js 2>/dev/null | wc -l)"

# 检查图片目录
echo "图片目录: $(ls -d static/images 2>/dev/null | wc -l)"
```

预期输出：
- 文章数量: 23
- CSS 文件: 4
- JS 文件: 2
- 图片目录: 1

- [ ] **Step 2: 完整构建测试**

```bash
hugo --cleanDestinationDir
```

预期输出：构建成功，无错误

- [ ] **Step 3: 检查构建输出**

```bash
ls -lh public/
```

预期输出：显示完整的站点结构

- [ ] **Step 4: 验证所有链接**

```bash
hugo server
```

手动点击测试所有页面链接，确保无 404 错误

- [ ] **Step 5: 创建最终提交**

```bash
git status
git add -A
git commit -m "chore: complete Jekyll to Hugo migration"
```

---

### Task 20: 创建迁移完成标记

**Files:**
- Create: `hugo-site/.migration-complete`

- [ ] **Step 1: 创建完成标记文件**

```bash
echo "Migration completed on $(date)" > hugo-site/.migration-complete
echo "Jekyll to Hugo migration successful" >> hugo-site/.migration-complete
echo "Total posts migrated: $(ls -1 hugo-site/content/posts/*.md 2>/dev/null | wc -l)" >> hugo-site/.migration-complete
```

- [ ] **Step 2: 显示迁移摘要**

```bash
cat hugo-site/.migration-complete
```

- [ ] **Step 3: 最终 commit**

```bash
git add hugo-site/.migration-complete
git commit -m "chore: mark migration as complete"
```

- [ ] **Step 4: 创建 Git 标签**

```bash
git tag -a v2.0.0-hugo -m "Migrated from Jekyll to Hugo"
```

---

## 迁移后步骤

### 可选：切换到 Hugo 作为主站点

如果测试通过，可以考虑以下选项：

**选项 1：保留双系统**
- 保持 Jekyll 和 Hugo 并存
- 在不同分支维护

**选项 2：完全切换到 Hugo**
1. 备份 Jekyll 文件到单独分支
2. 将 `hugo-site/` 内容移到根目录
3. 更新 GitHub Actions 工作流
4. 删除 Jekyll 相关文件

**选项 3：使用子目录**
- 保持当前结构
- 修改 GitHub Actions 从 `hugo-site/` 构建

### 部署验证

- [ ] 推送到 GitHub
- [ ] 检查 GitHub Actions 运行状态
- [ ] 访问生产环境 URL
- [ ] 验证所有页面正常工作
- [ ] 验证自定义域名（如果有）

---

## 故障排除

### 常见问题

**问题 1：Hugo 命令未找到**
```bash
# 重新安装 Hugo
sudo apt update && sudo apt install hugo -y
```

**问题 2：主题未加载**
```bash
# 检查 config.toml 中的 theme 配置
grep "theme" hugo-site/config.toml
```

**问题 3：文章不显示**
```bash
# 检查文章的 draft 状态
grep -r "draft: true" hugo-site/content/posts/
```

**问题 4：样式未加载**
```bash
# 检查 CSS 文件路径
ls -la hugo-site/themes/custom-theme/static/css/
```

**问题 5：GitHub Actions 失败**
- 检查工作流文件语法
- 查看 Actions 日志
- 确认 Hugo 版本兼容性

---

## 总结

本迁移计划包含 20 个任务，涵盖：

1. **环境准备**（Task 1-3）：安装 Hugo，创建站点，配置基础设置
2. **主题开发**（Task 4-7）：创建自定义主题，布局模板，partials
3. **资源迁移**（Task 8-10）：迁移 CSS、JS、图片
4. **内容迁移**（Task 11-13）：迁移文章、关于页面、标签系统
5. **部署配置**（Task 14-15）：CNAME、GitHub Actions
6. **测试验证**（Task 16-20）：本地测试、文档、最终验证

预计完成时间：2-3 小时

迁移完成后，你将拥有一个功能完整的 Hugo 博客，保留了所有 Jekyll 的内容和样式。

- [ ] **Step 2: Commit README**

```bash
git add README.md
git commit -m "docs: add Hugo site README"
```

---

### Task 19: 最终验证和清理

**Files:**
- Verify: 所有迁移内容

- [ ] **Step 1: 验证所有文件已提交**

```bash
cd /home/wenbo/WenboLi-CN-DE.github.io
git status
```

预期输出：working tree clean 或显示未跟踪的文件

- [ ] **Step 2: 查看提交历史**

```bash
git log --oneline -20
```

预期输出：显示所有迁移相关的提交

- [ ] **Step 3: 验证文章数量**

```bash
echo "Jekyll 文章数: $(ls -1 _posts/*.md 2>/dev/null | wc -l)"
echo "Hugo 文章数: $(ls -1 hugo-site/content/posts/*.md 2>/dev/null | wc -l)"
```

预期输出：两个数字应该相同（23篇）

- [ ] **Step 4: 最终构建测试**

```bash
cd hugo-site
hugo --minify
```

预期输出：构建成功，无错误

- [ ] **Step 5: 检查构建产物大小**

```bash
du -sh public/
```

预期输出：显示生成站点的大小

- [ ] **Step 6: 创建迁移完成标记**

```bash
cd /home/wenbo/WenboLi-CN-DE.github.io
touch hugo-site/.migration-complete
echo "Migration completed on $(date)" > hugo-site/.migration-complete
```

- [ ] **Step 7: 最终 commit**

```bash
git add hugo-site/.migration-complete
git commit -m "chore: mark migration as complete"
```

---

## Chunk 6: 部署和切换

### Task 20: 准备部署到 GitHub Pages

**Files:**
- Modify: GitHub repository settings

- [ ] **Step 1: 推送所有更改到 GitHub**

```bash
cd /home/wenbo/WenboLi-CN-DE.github.io
git push origin main
```

预期输出：所有提交成功推送

- [ ] **Step 2: 检查 GitHub Actions 运行状态**

访问 GitHub 仓库页面：
https://github.com/WenboLi-CN-DE/WenboLi-CN-DE.github.io/actions

查看工作流是否成功运行

- [ ] **Step 3: 配置 GitHub Pages 设置**

1. 访问仓库设置：Settings → Pages
2. Source: 选择 "GitHub Actions"
3. 保存设置

- [ ] **Step 4: 等待部署完成**

在 Actions 页面等待部署完成（通常 2-5 分钟）

- [ ] **Step 5: 验证部署结果**

访问 https://wenboli-cn-de.github.io 检查站点是否正常

---

### Task 21: 验证线上功能

**Files:**
- Test: 线上站点

- [ ] **Step 1: 测试首页**

访问 https://wenboli-cn-de.github.io
- 检查页面加载
- 检查样式是否正确
- 检查导航链接

- [ ] **Step 2: 测试文章页面**

- 点击任意文章
- 检查文章内容显示
- 检查代码高亮
- 检查图片加载
- 检查目录功能

- [ ] **Step 3: 测试标签页面**

访问 https://wenboli-cn-de.github.io/tags/
- 检查标签列表
- 点击标签查看文章列表

- [ ] **Step 4: 测试关于页面**

访问 https://wenboli-cn-de.github.io/about/
- 检查内容显示

- [ ] **Step 5: 测试 RSS Feed**

访问 https://wenboli-cn-de.github.io/feed.xml
- 检查 RSS 内容

- [ ] **Step 6: 测试移动端**

使用手机或浏览器开发者工具测试响应式布局

- [ ] **Step 7: 记录测试结果**

创建测试报告（可选）

---

### Task 22: 备份和归档 Jekyll 文件

**Files:**
- Create: `jekyll-backup/` (备份目录)

- [ ] **Step 1: 创建备份分支**

```bash
cd /home/wenbo/WenboLi-CN-DE.github.io
git checkout -b jekyll-backup
```

- [ ] **Step 2: 推送备份分支**

```bash
git push origin jekyll-backup
```

预期输出：备份分支创建成功

- [ ] **Step 3: 切回主分支**

```bash
git checkout main
```

- [ ] **Step 4: 创建备份目录（可选）**

```bash
mkdir -p jekyll-backup
cp -r _posts _layouts _includes css js images about.md _config.yml Gemfile jekyll-backup/
```

- [ ] **Step 5: 添加备份说明**

创建 `jekyll-backup/README.md`：

```markdown
# Jekyll 备份

这是原 Jekyll 博客的备份文件。

## 备份日期
2026-03-13

## 内容
- _posts/: 原始文章
- _layouts/: 布局文件
- _includes/: 包含文件
- css/: 样式文件
- js/: JavaScript 文件
- images/: 图片资源
- _config.yml: Jekyll 配置
- Gemfile: Ruby 依赖

## 恢复方法
如需恢复 Jekyll 博客：
1. 切换到 jekyll-backup 分支
2. 或使用此目录中的文件

## 注意
新的 Hugo 博客位于 hugo-site/ 目录。
```

- [ ] **Step 6: Commit 备份**

```bash
git add jekyll-backup/
git commit -m "chore: backup Jekyll files"
git push origin main
```

---

### Task 23: 清理和优化（可选）

**Files:**
- Optional: 清理旧文件

- [ ] **Step 1: 评估是否需要清理**

决定是否要从主分支删除 Jekyll 文件（建议保留一段时间）

- [ ] **Step 2: 如果决定清理，创建清理脚本**

创建 `cleanup-jekyll.sh`（不立即执行）：

```bash
#!/bin/bash
# Jekyll 文件清理脚本
# 警告：执行前请确保已备份

echo "此脚本将删除 Jekyll 相关文件"
echo "请确保："
echo "1. Hugo 站点运行正常"
echo "2. 已创建 jekyll-backup 分支"
echo "3. 已测试所有功能"
echo ""
read -p "确认继续？(yes/no): " confirm

if [ "$confirm" != "yes" ]; then
    echo "取消清理"
    exit 0
fi

# 删除 Jekyll 文件
rm -rf _posts _layouts _includes _site .jekyll-cache
rm -f _config.yml Gemfile Gemfile.lock Rakefile
rm -f 404.html archive.html categories.html index.html tags.html feed.xml

# 移动共享资源到 Hugo（如果还没移动）
# css, js, images 已经复制到 hugo-site，可以删除原文件
# rm -rf css js images

echo "清理完成"
echo "请运行: git status 查看更改"
```

- [ ] **Step 3: 添加执行权限**

```bash
chmod +x cleanup-jekyll.sh
```

- [ ] **Step 4: 记录清理脚本位置**

不立即执行，等待 Hugo 站点稳定运行一段时间后再考虑清理

---

### Task 24: 创建迁移总结报告

**Files:**
- Create: `MIGRATION-SUMMARY.md`

- [ ] **Step 1: 创建总结报告**

创建 `MIGRATION-SUMMARY.md`：

```markdown
# Jekyll 到 Hugo 迁移总结报告

## 迁移信息

- **迁移日期**: 2026-03-13
- **原系统**: Jekyll
- **新系统**: Hugo
- **迁移状态**: ✅ 完成

## 迁移统计

### 内容迁移
- 博客文章: 23 篇 ✅
- 页面: 1 个（关于页面）✅
- 图片资源: 全部 ✅
- CSS 文件: 4 个 ✅
- JS 文件: 2 个 ✅

### 功能迁移
- ✅ 首页布局
- ✅ 文章列表
- ✅ 文章详情页
- ✅ 标签系统
- ✅ 分页功能（20篇/页）
- ✅ 语法高亮
- ✅ 目录生成
- ✅ RSS Feed
- ✅ 响应式设计
- ✅ 自定义域名（CNAME）

### 部署配置
- ✅ GitHub Actions 工作流
- ✅ GitHub Pages 配置
- ✅ 自动化部署

## 技术对比

| 项目 | Jekyll | Hugo |
|------|--------|------|
| 语言 | Ruby | Go |
| 构建速度 | 较慢 | 非常快 |
| 配置文件 | YAML | TOML |
| 模板语言 | Liquid | Go Templates |
| 插件系统 | Gems | Shortcodes |
| 学习曲线 | 中等 | 中等 |

## 性能提升

- 构建速度: 预计提升 10-50 倍
- 页面加载: 优化后的静态资源
- 开发体验: 热重载更快

## 已知问题

- 无（目前运行正常）

## 后续计划

1. 监控站点运行 1-2 周
2. 收集用户反馈
3. 优化主题样式
4. 考虑添加新功能：
   - 评论系统
   - 搜索功能
   - 暗色模式
   - 更多交互功能

## 回滚方案

如需回滚到 Jekyll:
1. 切换到 `jekyll-backup` 分支
2. 或使用 `jekyll-backup/` 目录中的文件
3. 恢复原 GitHub Actions 工作流

## 联系信息

- 维护者: 李文博
- Email: lwb_010@163.com
- GitHub: @WenboLi-CN-DE

## 备注

迁移过程顺利，所有功能正常运行。Hugo 的构建速度明显快于 Jekyll，开发体验更好。
```

- [ ] **Step 2: Commit 总结报告**

```bash
git add MIGRATION-SUMMARY.md
git commit -m "docs: add migration summary report"
git push origin main
```

---

## 迁移完成检查清单

在宣布迁移完成之前，请确认以下所有项目：

### 环境和工具
- [ ] Hugo 已安装并验证
- [ ] Git 仓库状态正常
- [ ] 所有更改已提交

### 内容迁移
- [ ] 所有文章已迁移（23篇）
- [ ] 关于页面已迁移
- [ ] 图片资源已复制
- [ ] CSS 样式已迁移
- [ ] JavaScript 已迁移

### 功能验证
- [ ] 首页正常显示
- [ ] 文章列表正常
- [ ] 文章详情页正常
- [ ] 标签系统工作
- [ ] 分页功能正常
- [ ] 代码高亮工作
- [ ] 目录生成正常
- [ ] RSS Feed 可访问
- [ ] 响应式布局正常

### 部署配置
- [ ] GitHub Actions 工作流已创建
- [ ] CNAME 文件已配置
- [ ] GitHub Pages 设置正确
- [ ] 线上站点可访问
- [ ] 所有链接正常工作

### 文档
- [ ] MIGRATION.md 已创建
- [ ] README.md 已创建
- [ ] MIGRATION-SUMMARY.md 已创建

### 备份
- [ ] jekyll-backup 分支已创建
- [ ] 备份文件已保存

---

## 执行说明

1. **按顺序执行**: 严格按照 Task 顺序执行，不要跳过
2. **验证每一步**: 每个 Step 执行后都要验证结果
3. **频繁提交**: 每完成一个 Task 就提交一次
4. **遇到问题**: 如果某个步骤失败，先解决问题再继续
5. **测试优先**: 在推送到线上前，务必在本地充分测试

## 预计时间

- Chunk 1-2: 1-2 小时（环境和主题）
- Chunk 3-4: 1-2 小时（资源和内容迁移）
- Chunk 5-6: 1-2 小时（配置和部署）
- **总计**: 3-6 小时

## 成功标准

- ✅ 所有文章在 Hugo 站点正常显示
- ✅ 样式和原 Jekyll 站点基本一致
- ✅ 所有功能正常工作
- ✅ 线上站点可访问
- ✅ GitHub Actions 自动部署成功

---

**计划完成！准备好开始执行了吗？**

### 备份
- [ ] Jekyll 备份分支已创建
- [ ] 备份文件已保存

### 清理（可选）
- [ ] 清理脚本已准备（如需要）
- [ ] 决定是否保留 Jekyll 文件

---

## 预期结果

完成此迁移计划后，你将拥有：

1. **功能完整的 Hugo 博客**
   - 所有文章和页面已迁移
   - 样式和布局保持一致
   - 所有功能正常工作

2. **自动化部署流程**
   - 推送代码自动构建
   - 自动部署到 GitHub Pages
   - 无需手动干预

3. **完善的文档**
   - 迁移文档
   - 使用说明
   - 总结报告

4. **安全的回滚方案**
   - Jekyll 文件已备份
   - 可随时回滚

5. **性能提升**
   - 构建速度大幅提升
   - 开发体验更好
   - 页面加载更快

## 常见问题

### Q1: 迁移后原 Jekyll 文件怎么办？
A: 建议保留在 `jekyll-backup` 分支，主分支可以选择保留或删除。建议先保留一段时间，确认 Hugo 站点稳定后再考虑删除。

### Q2: 如何处理文章中的 Jekyll 特定语法？
A: 大部分 Markdown 语法兼容。如果使用了 Liquid 模板语法（如 `{% include %}`），需要手动修改或使用 Hugo 的 shortcodes 替代。

### Q3: 评论系统如何迁移？
A: 本计划未包含评论系统迁移。如需添加，可以使用 Disqus、Utterances 或 Giscus 等第三方服务。

### Q4: 如何添加新文章？
A: 使用命令 `hugo new posts/YYYY-MM-DD-title.md` 或直接在 `content/posts/` 目录创建 Markdown 文件。

### Q5: 构建失败怎么办？
A: 检查：
1. Hugo 版本是否正确
2. 配置文件语法是否正确
3. 模板文件是否有语法错误
4. 查看错误日志定位问题

### Q6: 如何自定义主题？
A: 修改 `themes/custom-theme/` 目录下的文件：
- 布局：`layouts/`
- 样式：`static/css/`
- 脚本：`static/js/`

### Q7: 如何优化 SEO？
A: 在 `config.toml` 中添加：
- `googleAnalytics`: Google Analytics ID
- `enableRobotsTXT = true`
- 在文章 front matter 中添加 `description` 和 `keywords`

### Q8: 如何添加搜索功能？
A: 可以使用：
- Algolia
- Lunr.js
- Hugo 的内置搜索功能

## 技术支持

如遇到问题：
1. 查看 Hugo 官方文档：https://gohugo.io/documentation/
2. 查看本项目的 MIGRATION.md
3. 检查 GitHub Actions 日志
4. 在 GitHub Issues 中提问

## 许可证

本迁移计划遵循 MIT 许可证。

---

**计划创建日期**: 2026-03-13  
**计划版本**: 1.0  
**预计完成时间**: 4-6 小时（取决于熟练程度）

---

## 执行说明

**准备开始执行此计划了吗？**

执行此计划需要使用 `superpowers:executing-plans` 技能。

建议执行方式：
1. 按照 Chunk 顺序执行
2. 每完成一个 Task 就 commit
3. 遇到问题及时记录
4. 完成每个 Chunk 后进行测试
5. 最后进行完整的端到端测试

祝迁移顺利！🚀
