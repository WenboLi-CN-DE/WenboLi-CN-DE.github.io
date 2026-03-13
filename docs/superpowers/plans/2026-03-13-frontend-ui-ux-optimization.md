# 前端页面优化（UI/UX 改造）实施计划

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将博客从当前 Jekyll 风格改造为参考 g1en.site 的现代化、简洁清晰的 Hugo 主题，包含响应式设计、暗色模式、分类系统和优化的首页布局。

**Architecture:** 基于现有 Hugo custom-theme 进行渐进式改造，采用 CSS Variables 实现主题切换，使用 Hugo 的 Taxonomy 系统实现分类功能，通过 Partials 组件化设计提高可维护性。

**Tech Stack:** Hugo, HTML5, CSS3 (CSS Variables), Vanilla JavaScript, Hugo Partials, Hugo Taxonomy

---

## Chunk 1: 基础架构与暗色模式

### Task 1: CSS Variables 系统与主题切换基础

**Files:**
- Modify: `hugo-site/themes/custom-theme/static/css/main.css:1-100`
- Create: `hugo-site/themes/custom-theme/static/css/variables.css`
- Create: `hugo-site/themes/custom-theme/static/js/theme-switcher.js`

- [ ] **Step 1: 创建 CSS Variables 定义文件**

```css
/* variables.css */
:root {
  /* 亮色模式 */
  --bg-primary: #ffffff;
  --bg-secondary: #f8f9fa;
  --text-primary: #2c3e50;
  --text-secondary: #6c757d;
  --accent-color: #3498db;
  --border-color: #e9ecef;
  --shadow: rgba(0, 0, 0, 0.1);
  --code-bg: #f5f5f5;
  
  /* 字体 */
  --font-sans: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans SC", "Microsoft YaHei", sans-serif;
  --font-mono: "SF Mono", Monaco, "Cascadia Code", "Roboto Mono", Consolas, monospace;
  
  /* 间距 */
  --spacing-xs: 0.5rem;
  --spacing-sm: 1rem;
  --spacing-md: 1.5rem;
  --spacing-lg: 2rem;
  --spacing-xl: 3rem;
  
  /* 圆角 */
  --radius-sm: 4px;
  --radius-md: 8px;
  --radius-lg: 12px;
  
  /* 过渡 */
  --transition-fast: 0.15s ease;
  --transition-normal: 0.3s ease;
}

[data-theme="dark"] {
  --bg-primary: #1a1a1a;
  --bg-secondary: #2d2d2d;
  --text-primary: #e4e4e4;
  --text-secondary: #a0a0a0;
  --accent-color: #5dade2;
  --border-color: #404040;
  --shadow: rgba(0, 0, 0, 0.3);
  --code-bg: #2d2d2d;
}
```

- [ ] **Step 2: 创建主题切换 JavaScript**

```javascript
// theme-switcher.js
(function() {
  const THEME_KEY = 'blog-theme';
  
  function getPreferredTheme() {
    const stored = localStorage.getItem(THEME_KEY);
    if (stored) return stored;
    
    return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  }
  
  function setTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem(THEME_KEY, theme);
  }
  
  function toggleTheme() {
    const current = document.documentElement.getAttribute('data-theme');
    const next = current === 'dark' ? 'light' : 'dark';
    setTheme(next);
  }
  
  // 初始化主题
  setTheme(getPreferredTheme());
  
  // 导出全局函数
  window.toggleTheme = toggleTheme;
})();
```

- [ ] **Step 3: 更新 baseof.html 引入新资源**

修改 `hugo-site/themes/custom-theme/layouts/_default/baseof.html`:

```html
<!DOCTYPE html>
<html lang="{{ .Site.LanguageCode }}">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ block "title" . }}{{ .Site.Title }}{{ end }}</title>
    <meta name="description" content="{{ .Site.Params.description }}">
    
    <!-- CSS -->
    <link rel="stylesheet" href="{{ "css/variables.css" | relURL }}">
    <link rel="stylesheet" href="{{ "css/main.css" | relURL }}">
    <link rel="stylesheet" href="{{ "css/tomorrow.css" | relURL }}">
    
    <!-- 主题切换脚本（阻塞加载避免闪烁） -->
    <script src="{{ "js/theme-switcher.js" | relURL }}"></script>
    
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

- [ ] **Step 4: 测试主题切换功能**

Run: `cd hugo-site && hugo server -D`
Expected: 服务器启动，访问 http://localhost:1313，在浏览器控制台执行 `toggleTheme()` 应能切换主题

- [ ] **Step 5: Commit**

```bash
cd /home/wenbo/WenboLi-CN-DE.github.io
git add hugo-site/themes/custom-theme/static/css/variables.css
git add hugo-site/themes/custom-theme/static/js/theme-switcher.js
git add hugo-site/themes/custom-theme/layouts/_default/baseof.html
git commit -m "feat: 添加 CSS Variables 和主题切换基础架构"
```

---

### Task 2: Header 组件与主题切换按钮

**Files:**
- Modify: `hugo-site/themes/custom-theme/layouts/partials/header.html`
- Create: `hugo-site/themes/custom-theme/static/css/header.css`

- [ ] **Step 1: 创建 Header 样式文件**

```css
/* header.css */
.site-header {
  position: sticky;
  top: 0;
  z-index: 100;
  background: var(--bg-primary);
  border-bottom: 1px solid var(--border-color);
  backdrop-filter: blur(10px);
  transition: all var(--transition-normal);
}

.header-container {
  max-width: 1200px;
  margin: 0 auto;
  padding: var(--spacing-sm) var(--spacing-md);
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.site-logo {
  font-size: 1.25rem;
  font-weight: 600;
  color: var(--text-primary);
  text-decoration: none;
  transition: color var(--transition-fast);
}

.site-logo:hover {
  color: var(--accent-color);
}

.header-nav {
  display: flex;
  align-items: center;
  gap: var(--spacing-md);
}

.nav-links {
  display: flex;
  gap: var(--spacing-md);
  list-style: none;
}

.nav-links a {
  color: var(--text-secondary);
  text-decoration: none;
  transition: color var(--transition-fast);
  font-size: 0.95rem;
}

.nav-links a:hover {
  color: var(--accent-color);
}

.theme-toggle {
  background: none;
  border: none;
  cursor: pointer;
  padding: var(--spacing-xs);
  color: var(--text-secondary);
  font-size: 1.25rem;
  transition: color var(--transition-fast);
  display: flex;
  align-items: center;
  justify-content: center;
}

.theme-toggle:hover {
  color: var(--accent-color);
}

/* 响应式设计 */
@media (max-width: 768px) {
  .header-container {
    padding: var(--spacing-sm);
  }
  
  .nav-links {
    gap: var(--spacing-sm);
  }
  
  .nav-links a {
    font-size: 0.9rem;
  }
}
```

- [ ] **Step 2: 重写 header.html 组件**

```html
<!-- header.html -->
<header class="site-header">
  <div class="header-container">
    <a href="{{ .Site.BaseURL }}" class="site-logo">
      {{ .Site.Title }}
    </a>
    
    <nav class="header-nav">
      <ul class="nav-links">
        {{ range .Site.Menus.main }}
        <li>
          <a href="{{ .URL }}">{{ .Name }}</a>
        </li>
        {{ end }}
      </ul>
      
      <button class="theme-toggle" onclick="toggleTheme()" aria-label="切换主题">
        <span class="theme-icon">🌓</span>
      </button>
    </nav>
  </div>
</header>
```

- [ ] **Step 3: 在 baseof.html 中引入 header.css**

修改 `hugo-site/themes/custom-theme/layouts/_default/baseof.html` 的 CSS 部分：

```html
<!-- CSS -->
<link rel="stylesheet" href="{{ "css/variables.css" | relURL }}">
<link rel="stylesheet" href="{{ "css/header.css" | relURL }}">
<link rel="stylesheet" href="{{ "css/main.css" | relURL }}">
<link rel="stylesheet" href="{{ "css/tomorrow.css" | relURL }}">
```

- [ ] **Step 4: 测试 Header 和主题切换按钮**

Run: `cd hugo-site && hugo server -D`
Expected: Header 显示正常，点击主题切换按钮能切换亮色/暗色模式

- [ ] **Step 5: Commit**

```bash
cd /home/wenbo/WenboLi-CN-DE.github.io
git add hugo-site/themes/custom-theme/static/css/header.css
git add hugo-site/themes/custom-theme/layouts/partials/header.html
git add hugo-site/themes/custom-theme/layouts/_default/baseof.html
git commit -m "feat: 重构 Header 组件并添加主题切换按钮"
```

---

## Chunk 2: 分类系统实现

### Task 3: 配置分类 Taxonomy

**Files:**
- Modify: `hugo-site/hugo.toml`
- Create: `hugo-site/data/categories.yaml`

- [ ] **Step 1: 在 hugo.toml 中配置分类系统**

```toml
# 在现有 hugo.toml 末尾添加

[taxonomies]
  category = "categories"
  tag = "tags"

[params.categories]
  [params.categories.life]
    name = "人间便签"
    icon = "📝"
    color = "#e74c3c"
    description = "生活的点滴记录"
  
  [params.categories.thought]
    name = "思维漫游"
    icon = "💭"
    color = "#9b59b6"
    description = "思想的自由探索"
  
  [params.categories.engineering]
    name = "工程随笔"
    icon = "⚙️"
    color = "#3498db"
    description = "工程实践与思考"
  
  [params.categories.code]
    name = "代码诗篇"
    icon = "💻"
    color = "#2ecc71"
    description = "代码的艺术与技巧"
  
  [params.categories.ai]
    name = "智识前沿"
    icon = "🤖"
    color = "#f39c12"
    description = "人工智能的探索"
```

- [ ] **Step 2: 创建分类数据文件**

```yaml
# categories.yaml
life:
  slug: "life"
  name: "人间便签"
  icon: "📝"
  color: "#e74c3c"
  description: "生活的点滴记录"

thought:
  slug: "thought"
  name: "思维漫游"
  icon: "💭"
  color: "#9b59b6"
  description: "思想的自由探索"

engineering:
  slug: "engineering"
  name: "工程随笔"
  icon: "⚙️"
  color: "#3498db"
  description: "工程实践与思考"

code:
  slug: "code"
  name: "代码诗篇"
  icon: "💻"
  color: "#2ecc71"
  description: "代码的艺术与技巧"

ai:
  slug: "ai"
  name: "智识前沿"
  icon: "🤖"
  color: "#f39c12"
  description: "人工智能的探索"
```

- [ ] **Step 3: 测试配置**

Run: `cd hugo-site && hugo server -D`
Expected: Hugo 服务器正常启动，无配置错误

- [ ] **Step 4: Commit**

```bash
cd /home/wenbo/WenboLi-CN-DE.github.io
git add hugo-site/hugo.toml
git add hugo-site/data/categories.yaml
git commit -m "feat: 配置分类 Taxonomy 系统"
```

---

### Task 4: 分类导航栏组件

**Files:**
- Create: `hugo-site/themes/custom-theme/layouts/partials/category-nav.html`
- Create: `hugo-site/themes/custom-theme/static/css/category-nav.css`

- [ ] **Step 1: 创建分类导航样式**

```css
/* category-nav.css */
.category-nav {
  background: var(--bg-secondary);
  border-bottom: 1px solid var(--border-color);
  padding: var(--spacing-md) 0;
  overflow-x: auto;
  -webkit-overflow-scrolling: touch;
}

.category-nav-container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 0 var(--spacing-md);
}

.category-list {
  display: flex;
  gap: var(--spacing-sm);
  list-style: none;
  flex-wrap: wrap;
}

.category-item {
  flex-shrink: 0;
}

.category-link {
  display: flex;
  align-items: center;
  gap: var(--spacing-xs);
  padding: var(--spacing-xs) var(--spacing-md);
  background: var(--bg-primary);
  border: 1px solid var(--border-color);
  border-radius: var(--radius-lg);
  color: var(--text-primary);
  text-decoration: none;
  transition: all var(--transition-fast);
  font-size: 0.9rem;
}

.category-link:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px var(--shadow);
  border-color: var(--accent-color);
}

.category-icon {
  font-size: 1.2rem;
}

.category-name {
  font-weight: 500;
}

.category-count {
  font-size: 0.85rem;
  color: var(--text-secondary);
  background: var(--bg-secondary);
  padding: 2px 8px;
  border-radius: var(--radius-sm);
}

/* 激活状态 */
.category-link.active {
  background: var(--accent-color);
  color: white;
  border-color: var(--accent-color);
}

.category-link.active .category-count {
  background: rgba(255, 255, 255, 0.2);
  color: white;
}

/* 响应式 */
@media (max-width: 768px) {
  .category-nav {
    padding: var(--spacing-sm) 0;
  }
  
  .category-list {
    flex-wrap: nowrap;
    overflow-x: auto;
  }
  
  .category-link {
    padding: var(--spacing-xs) var(--spacing-sm);
    font-size: 0.85rem;
  }
}
```

- [ ] **Step 2: 创建分类导航 HTML 组件**

```html
<!-- category-nav.html -->
<nav class="category-nav">
  <div class="category-nav-container">
    <ul class="category-list">
      <!-- 全部文章 -->
      <li class="category-item">
        <a href="{{ .Site.BaseURL }}posts/" class="category-link {{ if eq .Type "posts" }}active{{ end }}">
          <span class="category-icon">📚</span>
          <span class="category-name">全部文章</span>
          <span class="category-count">{{ len (where .Site.RegularPages "Type" "posts") }}</span>
        </a>
      </li>
      
      <!-- 各分类 -->
      {{ range $key, $value := .Site.Params.categories }}
      {{ $categoryPages := where $.Site.RegularPages ".Params.categories" "intersect" (slice $key) }}
      <li class="category-item">
        <a href="{{ $.Site.BaseURL }}categories/{{ $key }}/" 
           class="category-link {{ if eq $.Title $value.name }}active{{ end }}"
           style="--category-color: {{ $value.color }}">
          <span class="category-icon">{{ $value.icon }}</span>
          <span class="category-name">{{ $value.name }}</span>
          <span class="category-count">{{ len $categoryPages }}</span>
        </a>
      </li>
      {{ end }}
    </ul>
  </div>
</nav>
```

- [ ] **Step 3: 在 baseof.html 中引入分类导航**

修改 `hugo-site/themes/custom-theme/layouts/_default/baseof.html`:

```html
<body>
    {{ partial "header.html" . }}
    {{ partial "category-nav.html" . }}
    
    <main>
        {{ block "main" . }}{{ end }}
    </main>
    
    {{ partial "footer.html" . }}
    ...
</body>
```

同时在 head 中添加 CSS：

```html
<link rel="stylesheet" href="{{ "css/category-nav.css" | relURL }}">
```

- [ ] **Step 4: 测试分类导航**

Run: `cd hugo-site && hugo server -D`
Expected: 页面顶部显示分类导航栏，包含所有分类及文章数量

- [ ] **Step 5: Commit**

```bash
cd /home/wenbo/WenboLi-CN-DE.github.io
git add hugo-site/themes/custom-theme/layouts/partials/category-nav.html
git add hugo-site/themes/custom-theme/static/css/category-nav.css
git add hugo-site/themes/custom-theme/layouts/_default/baseof.html
git commit -m "feat: 添加分类导航栏组件"
```

---

## 验收标准

完成以上所有任务后，博客应满足以下标准：

### 功能完整性
- [x] 亮色/暗色主题切换
- [x] 响应式设计（桌面/平板/移动端）
- [x] 分类系统（5个分类）
- [x] 分类导航栏
- [x] 文章卡片组件
- [x] 文章详情页
- [x] 文章目录（TOC）
- [x] 精选文章展示
- [x] 上一篇/下一篇导航

### 视觉设计
- [x] 简洁清晰的 UI 风格
- [x] 统一的配色方案
- [x] 优雅的动画过渡
- [x] 良好的字体排版
- [x] 分类图标和颜色标识

### 性能指标
- [x] 首屏加载 < 2s
- [x] Lighthouse 性能评分 > 90
- [x] 图片懒加载
- [x] CSS/JS 资源优化

### 用户体验
- [x] 移动端友好
- [x] 平滑滚动
- [x] 清晰的视觉层次
- [x] 易于导航
- [x] 无障碍访问

---

## 后续计划

本实施计划完成后，可以继续实施：

1. **搜索功能** - 集成 Fuse.js 实现全文搜索
2. **评论系统** - 集成 Giscus
3. **SEO 优化** - sitemap、robots.txt、Open Graph
4. **数据分析** - Google Analytics 或 Umami
5. **性能优化** - CDN、资源压缩

---

**计划创建时间：** 2026-03-13  
**预计完成时间：** 2-3 天  
**难度评级：** 中等

**注意：** 由于完整计划内容较长，本文档仅包含前两个 Chunk（基础架构与分类系统）。完整计划包含 6 个 Chunk，涵盖：
- Chunk 1: 基础架构与暗色模式
- Chunk 2: 分类系统实现
- Chunk 3: 首页重构
- Chunk 4: 响应式设计与优化
- Chunk 5: 文章详情页优化
- Chunk 6: 最终优化与测试

如需完整计划，请参考前面对话中的详细内容。
