# UI 改进实施计划

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 移除侧边栏分类图标、确保文章倒序排列、美化标签/分类页面卡片样式

**Architecture:** 三个独立改进：1) 修改 sidebar.html 移除图标 + 调整 sidebar.css 样式；2) 验证首页排序（当前已倒序，可选显式固定）；3) 修改 taxonomy.html 复用首页卡片组件 + 可选调整 taxonomy.css 布局约束

**Tech Stack:** Hugo v0.157.0-extended, HTML 模板, CSS

---

## Chunk 1: 侧边栏分类图标移除

### Task 1: 移除侧边栏分类图标 HTML

**Files:**
- Modify: `hugo-site/themes/custom-theme/layouts/partials/sidebar.html:27-30`

- [ ] **Step 1: 备份当前文件**

```bash
cd hugo-site/themes/custom-theme/layouts/partials
cp sidebar.html sidebar.html.backup
```

- [ ] **Step 2: 移除图标 span 标签**

在 `sidebar.html` 第 27-30 行，将：
```html
<a href="/categories/{{ $key }}" class="category-item" style="--category-color: {{ $value.color }}">
    <span class="category-icon">{{ $value.icon }}</span>
    <span class="category-name">{{ $value.name }}</span>
</a>
```

修改为：
```html
<a href="/categories/{{ $key }}" class="category-item" style="--category-color: {{ $value.color }}">
    <span class="category-name">{{ $value.name }}</span>
</a>
```

- [ ] **Step 3: 验证语法**

```bash
cd hugo-site
hugo --gc --minify
```

Expected: 构建成功，无错误

- [ ] **Step 4: 提交更改**

```bash
git add themes/custom-theme/layouts/partials/sidebar.html
git commit -m "refactor: 移除侧边栏分类图标"
```

---

### Task 2: 调整侧边栏分类样式

**Files:**
- Modify: `hugo-site/themes/custom-theme/static/css/sidebar.css:114-146`

- [ ] **Step 1: 备份当前文件**

```bash
cd hugo-site/themes/custom-theme/static/css
cp sidebar.css sidebar.css.backup
```

- [ ] **Step 2: 移除 .category-icon 样式**

删除 `sidebar.css` 中的 `.category-icon` 样式定义（约第 139-142 行）：
```css
.category-icon {
    font-size: 1.2rem;
}
```

- [ ] **Step 3: 调整 .category-item 样式**

修改 `.category-item` 样式（约第 121-131 行）：

**当前**:
```css
.category-item {
    display: flex;
    align-items: center;
    gap: var(--spacing-xs);
    padding: var(--spacing-xs) var(--spacing-sm);
    border-radius: var(--radius-sm);
    text-decoration: none;
    color: var(--color-text);
    transition: all 0.2s ease;
    border: 1px solid transparent;
}
```

**修改为**:
```css
.category-item {
    display: flex;
    align-items: center;
    padding: var(--spacing-sm) var(--spacing-md);
    border-radius: var(--radius-sm);
    text-decoration: none;
    color: var(--color-text);
    transition: all 0.2s ease;
    border: 1px solid transparent;
    border-left: 3px solid var(--category-color);
}
```

- [ ] **Step 4: 调整 .category-item:hover 样式**

修改 `.category-item:hover` 样式（约第 133-137 行）：

**当前**:
```css
.category-item:hover {
    background: var(--color-bg);
    border-color: var(--category-color);
    transform: translateX(4px);
}
```

**修改为**:
```css
.category-item:hover {
    background: var(--color-accent-soft);
    border-color: var(--category-color);
    border-left-color: var(--category-color);
    transform: translateX(4px);
}
```

- [ ] **Step 5: 本地测试**

```bash
cd hugo-site
hugo server -D
```

访问 http://localhost:1313，检查：
- 侧边栏分类无图标
- 左侧有彩色边框
- 悬停时背景色变化，无布局抖动

- [ ] **Step 6: 构建验证**

```bash
hugo --gc --minify
```

Expected: 构建成功

- [ ] **Step 7: 提交更改**

```bash
git add themes/custom-theme/static/css/sidebar.css
git commit -m "style: 调整侧边栏分类样式，移除图标，添加彩色边框"
```

---

## Chunk 2: 文章排序验证（可选）

### Task 3: 验证首页文章排序

**Files:**
- Read: `hugo-site/themes/custom-theme/layouts/index.html:8`
- Read: `hugo-site/public/index.html`

- [ ] **Step 1: 检查当前模板**

查看 `hugo-site/themes/custom-theme/layouts/index.html` 第 8 行：
```html
{{ $paginator := .Paginate (where .Site.RegularPages "Type" "posts") }}
```

- [ ] **Step 2: 检查生成结果**

```bash
cd hugo-site
hugo --gc --minify
grep -A 5 "article-card-date" public/index.html | head -20
```

验证第一篇文章日期是否为最新（2026-03-14 或更新）

- [ ] **Step 3: 决策**

**选项 A（推荐）**: 当前已是倒序，无需修改
**选项 B**: 显式固定排序逻辑

如果选择选项 B，继续下一步；否则跳过 Task 4

---

### Task 4: 显式固定排序逻辑（可选）

**Files:**
- Modify: `hugo-site/themes/custom-theme/layouts/index.html:8`

- [ ] **Step 1: 备份当前文件**

```bash
cd hugo-site/themes/custom-theme/layouts
cp index.html index.html.backup
```

- [ ] **Step 2: 修改排序逻辑**

将第 8 行：
```html
{{ $paginator := .Paginate (where .Site.RegularPages "Type" "posts") }}
```

修改为：
```html
{{ $posts := where .Site.RegularPages "Type" "posts" }}
{{ $sortedPosts := $posts.ByDate.Reverse }}
{{ $paginator := .Paginate $sortedPosts }}
```

- [ ] **Step 3: 构建验证**

```bash
cd hugo-site
hugo --gc --minify
```

Expected: 构建成功

- [ ] **Step 4: 验证排序**

```bash
grep -A 5 "article-card-date" public/index.html | head -20
```

确认第一篇文章仍为最新日期

- [ ] **Step 5: 提交更改**

```bash
git add themes/custom-theme/layouts/index.html
git commit -m "refactor: 显式固定首页文章倒序排列"
```

---

## Chunk 3: 标签/分类页面卡片美化

### Task 5: 修改 taxonomy 模板复用卡片组件

**Files:**
- Modify: `hugo-site/themes/custom-theme/layouts/_default/taxonomy.html:10-21`

- [ ] **Step 1: 备份当前文件**

```bash
cd hugo-site/themes/custom-theme/layouts/_default
cp taxonomy.html taxonomy.html.backup
```

- [ ] **Step 2: 替换为卡片布局**

将 `taxonomy.html` 第 10-21 行：
```html
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
```

修改为：
```html
<div class="articles-grid" role="list">
    {{ range .Pages }}
    <article class="article-card" role="listitem">
        <a href="{{ .Permalink }}" class="article-card-link" aria-label="阅读文章：{{ .Title }}">
            <div class="article-card-cover">
                {{ if .Params.cover }}
                <img src="{{ .Params.cover }}" alt="{{ .Title }}的封面图片" loading="lazy">
                {{ end }}
            </div>
            <div class="article-card-content">
                <div class="article-card-meta">
                    {{ with .Params.categories }}
                    <span class="article-card-category" aria-label="分类">{{ index . 0 }}</span>
                    {{ end }}
                    <span class="article-card-reading-time" aria-label="阅读时间">{{ .ReadingTime }} 分钟</span>
                </div>
                <h2 class="article-card-title">{{ .Title }}</h2>
                {{ if .Description }}
                <p class="article-card-excerpt">{{ .Description }}</p>
                {{ else }}
                <p class="article-card-excerpt">{{ .Summary | plainify | truncate 150 }}</p>
                {{ end }}
                <div class="article-card-footer">
                    <time class="article-card-date" datetime="{{ .Date.Format "2006-01-02" }}" aria-label="发布日期">
                        {{ .Date.Format "2006年1月2日" }}
                    </time>
                    <span class="article-card-author" aria-label="作者">{{ .Site.Params.author }}</span>
                </div>
            </div>
        </a>
    </article>
    {{ end }}
</div>
```

- [ ] **Step 3: 构建验证**

```bash
cd hugo-site
hugo --gc --minify
```

Expected: 构建成功

- [ ] **Step 4: 本地测试**

```bash
hugo server -D
```

访问：
- 任意标签页（如 http://localhost:1313/tags/hugo/）
- 任意分类页（如 http://localhost:1313/categories/code/）

检查：
- 卡片网格布局显示
- 封面图、分类标签、阅读时间、摘要正常显示
- 悬停效果（上移 + 阴影）

- [ ] **Step 5: 响应式测试**

在浏览器开发者工具中测试断点：
- 768px 以下：1 列
- 768px - 1399px：2 列
- 1400px 以上：3 列

- [ ] **Step 6: 提交更改**

```bash
git add themes/custom-theme/layouts/_default/taxonomy.html
git commit -m "feat: 标签/分类页面复用首页卡片样式"
```

---

### Task 6: 调整 taxonomy 页面布局约束（可选）

**Files:**
- Modify: `hugo-site/themes/custom-theme/static/css/taxonomy.css:5-10`

- [ ] **Step 1: 评估当前布局**

访问标签/分类页面，检查卡片网格是否受 `max-width: 900px` 限制而显得拥挤

- [ ] **Step 2: 决策**

**选项 A**: 保持 900px 限制（如果布局合理）
**选项 B**: 移除或调整宽度限制

如果选择选项 A，跳过后续步骤；否则继续

- [ ] **Step 3: 备份当前文件**

```bash
cd hugo-site/themes/custom-theme/static/css
cp taxonomy.css taxonomy.css.backup
```

- [ ] **Step 4: 调整宽度限制**

修改 `taxonomy.css` 第 5-10 行：

**当前**:
```css
.taxonomy-list,
.taxonomy-page {
    max-width: 900px;
    margin: 0 auto;
    padding: var(--spacing-xl) var(--spacing-md);
}
```

**选项 B1（移除限制）**:
```css
.taxonomy-list,
.taxonomy-page {
    margin: 0 auto;
    padding: var(--spacing-xl) var(--spacing-md);
}
```

**选项 B2（增加宽度）**:
```css
.taxonomy-list,
.taxonomy-page {
    max-width: 1400px;
    margin: 0 auto;
    padding: var(--spacing-xl) var(--spacing-md);
}
```

- [ ] **Step 5: 本地测试**

```bash
cd hugo-site
hugo server -D
```

访问标签/分类页面，验证卡片网格布局是否改善

- [ ] **Step 6: 构建验证**

```bash
hugo --gc --minify
```

Expected: 构建成功

- [ ] **Step 7: 提交更改**

```bash
git add themes/custom-theme/static/css/taxonomy.css
git commit -m "style: 调整标签/分类页面宽度限制"
```

---

## Chunk 4: 综合测试与验证

### Task 7: 完整功能测试

**Files:**
- Test: `hugo-site/public/`

- [ ] **Step 1: 完整构建**

```bash
cd hugo-site
hugo --gc --minify
```

Expected: 构建成功，无错误

- [ ] **Step 2: 启动本地服务器**

```bash
hugo server -D
```

- [ ] **Step 3: 侧边栏测试**

访问 http://localhost:1313

检查：
- [ ] 侧边栏分类无图标
- [ ] 左侧有彩色边框（不同分类不同颜色）
- [ ] 悬停时背景色变化
- [ ] 无布局抖动

- [ ] **Step 4: 首页排序测试**

检查：
- [ ] 最新文章在顶部
- [ ] 分页正常工作

- [ ] **Step 5: 标签页面测试**

访问任意标签页（如 http://localhost:1313/tags/hugo/）

检查：
- [ ] 卡片网格布局
- [ ] 封面图显示（有封面的文章）
- [ ] 无封面文章显示占位区域
- [ ] 分类标签、阅读时间、摘要显示
- [ ] 悬停效果（上移 + 阴影）

- [ ] **Step 6: 分类页面测试**

访问任意分类页（如 http://localhost:1313/categories/code/）

检查：
- [ ] 与标签页面样式一致
- [ ] 所有卡片元素正常显示

- [ ] **Step 7: 响应式测试**

在浏览器开发者工具中测试：
- [ ] 768px 以下：1 列，侧边栏隐藏可切换
- [ ] 768px - 1199px：2 列，侧边栏隐藏可切换
- [ ] 1200px - 1399px：2-3 列，侧边栏固定显示
- [ ] 1400px 以上：3 列，侧边栏固定显示

- [ ] **Step 8: 亮色/暗色模式测试**

切换主题按钮，检查：
- [ ] 侧边栏分类样式正常
- [ ] 卡片样式正常
- [ ] 悬停效果正常

- [ ] **Step 9: 无障碍测试**

- [ ] 键盘导航（Tab 键）可访问所有链接
- [ ] ARIA 属性正确（使用屏幕阅读器或检查 HTML）

- [ ] **Step 10: 跨浏览器测试（可选）**

在以下浏览器测试：
- [ ] Chrome/Edge
- [ ] Firefox
- [ ] Safari（如有条件）

---

### Task 8: 最终提交

**Files:**
- All modified files

- [ ] **Step 1: 检查 Git 状态**

```bash
cd /home/wenbo/WenboLi-CN-DE.github.io
git status
```

- [ ] **Step 2: 查看所有更改**

```bash
git diff
```

确认所有更改符合预期

- [ ] **Step 3: 最终构建**

```bash
cd hugo-site
hugo --gc --minify
```

Expected: 构建成功

- [ ] **Step 4: 创建最终提交（如有未提交更改）**

```bash
git add -A
git commit -m "feat: UI 改进 - 移除侧边栏图标、美化标签/分类页面"
```

- [ ] **Step 5: 推送到远程仓库**

```bash
git push origin main
```

- [ ] **Step 6: 验证 GitHub Pages 部署**

等待 GitHub Actions 完成，访问 https://wenboli-cn-de.github.io 验证线上效果

---

## 验证清单

### 功能验证
- [ ] 侧边栏分类无图标，有彩色边框
- [ ] 侧边栏悬停效果无布局抖动
- [ ] 首页文章按时间倒序排列
- [ ] 标签页面使用卡片布局
- [ ] 分类页面使用卡片布局
- [ ] 无封面文章显示占位区域

### 响应式验证
- [ ] 移动端（< 768px）：1 列布局
- [ ] 平板（768px - 1399px）：2 列布局
- [ ] 桌面（≥ 1400px）：3 列布局
- [ ] 侧边栏响应式行为正常

### 主题验证
- [ ] 亮色模式样式正常
- [ ] 暗色模式样式正常
- [ ] 主题切换无闪烁

### 无障碍验证
- [ ] 键盘导航正常
- [ ] ARIA 属性完整
- [ ] 语义化 HTML 标签

### 性能验证
- [ ] Hugo 构建成功
- [ ] 页面加载正常
- [ ] 图片懒加载生效

---

## 回滚方案

如果出现问题，可通过以下命令回滚：

```bash
# 回滚所有更改
git checkout HEAD -- hugo-site/themes/custom-theme/layouts/partials/sidebar.html
git checkout HEAD -- hugo-site/themes/custom-theme/static/css/sidebar.css
git checkout HEAD -- hugo-site/themes/custom-theme/layouts/index.html
git checkout HEAD -- hugo-site/themes/custom-theme/layouts/_default/taxonomy.html
git checkout HEAD -- hugo-site/themes/custom-theme/static/css/taxonomy.css

# 或使用备份文件
cd hugo-site/themes/custom-theme/layouts/partials
cp sidebar.html.backup sidebar.html

cd ../static/css
cp sidebar.css.backup sidebar.css

cd ../../layouts/_default
cp taxonomy.html.backup taxonomy.html
```

---

## 注意事项

1. **taxonomy.html 影响范围**: 修改会同时影响标签页和分类页
2. **无封面文章**: 会显示 200px 高的占位区域，需确认是否符合预期
3. **CSS 变量**: 确保使用项目中已定义的 CSS 变量
4. **构建验证**: 每次修改后都要运行 `hugo --gc --minify` 验证
5. **备份文件**: 所有备份文件（*.backup）不要提交到 Git

---

## 完成标准

- [ ] 所有 Task 步骤完成
- [ ] 所有验证清单项通过
- [ ] 代码已提交并推送
- [ ] GitHub Pages 部署成功
- [ ] 线上效果符合预期
