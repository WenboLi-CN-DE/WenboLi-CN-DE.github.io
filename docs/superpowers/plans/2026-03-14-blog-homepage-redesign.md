# 博客首页重新设计实施计划

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 优化博客首页设计，提升个人 branding 和用户体验，包括顶部 Banner、头像名字区域、分类导航和文章卡片的全面改进。

**Architecture:** 基于现有 Hugo 自定义主题，通过修改 HTML 模板和 CSS 样式文件实现渐进式增强。采用移动优先的响应式设计，保持编辑式极简主义风格。

**Tech Stack:** Hugo v0.157.0, 自定义 CSS（无框架），原生 JavaScript，Google Fonts

---

## Chunk 1: 顶部 Hero Banner 优化

### Task 1: 创建固定顶部 Banner 组件

**Files:**
- Modify: `hugo-site/themes/custom-theme/layouts/index.html:8-14`
- Modify: `hugo-site/themes/custom-theme/static/css/home.css:50-171`
- Create: `hugo-site/themes/custom-theme/static/css/hero-banner.css`

- [ ] **Step 1: 创建 hero-banner.css 文件**

创建新的 CSS 文件用于 Banner 样式：

```css
/* Hero Banner - 固定顶部设计 */
.hero-banner {
  position: sticky;
  top: 0;
  z-index: 1000;
  background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
  padding: 2rem;
  text-align: center;
  box-shadow: 0 2px 10px rgba(0,0,0,0.1);
  transition: all 0.3s ease;
}

.hero-banner-content {
  max-width: 1200px;
  margin: 0 auto;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 0.5rem;
}

.hero-banner-title {
  font-family: var(--font-display);
  font-size: 2rem;
  font-weight: 700;
  color: white;
  margin: 0;
  letter-spacing: -0.02em;
}

.hero-banner-subtitle {
  font-family: var(--font-body);
  font-size: 1rem;
  color: rgba(255, 255, 255, 0.9);
  margin: 0;
}

.hero-banner-quote {
  font-family: var(--font-display);
  font-size: 1.1rem;
  color: rgba(255, 255, 255, 0.85);
  font-style: italic;
  margin: 0;
}

/* 滚动后缩小效果 */
.hero-banner.scrolled {
  padding: 1rem 2rem;
}

.hero-banner.scrolled .hero-banner-title {
  font-size: 1.5rem;
}

.hero-banner.scrolled .hero-banner-subtitle,
.hero-banner.scrolled .hero-banner-quote {
  font-size: 0.9rem;
}

/* 响应式 */
@media (max-width: 768px) {
  .hero-banner {
    padding: 1.5rem 1rem;
  }
  
  .hero-banner-title {
    font-size: 1.5rem;
  }
  
  .hero-banner-subtitle {
    font-size: 0.9rem;
  }
  
  .hero-banner-quote {
    font-size: 1rem;
  }
}
```

- [ ] **Step 2: 验证 CSS 文件创建**

```bash
ls -la hugo-site/themes/custom-theme/static/css/hero-banner.css
```

预期输出：文件存在

- [ ] **Step 3: 修改 index.html 添加 Banner**

在 `hugo-site/themes/custom-theme/layouts/index.html` 中，将现有的 hero-section 替换为新的 hero-banner：

修改前的代码（第 8-14 行）：
```html
<section class="hero-section">
    <div class="hero-content">
        <h1 class="hero-title">{{ .Site.Title }}</h1>
        <p class="hero-subtitle">{{ .Site.Params.avatarDesc }}</p>
        <p class="hero-quote">与其感慨路难行，不如马上出发。</p>
    </div>
</section>
```

修改后的代码：
```html
<section class="hero-banner" id="heroBanner">
    <div class="hero-banner-content">
        <h1 class="hero-banner-title">{{ .Site.Title }}</h1>
        <p class="hero-banner-subtitle">{{ .Site.Params.avatarDesc }}</p>
        <p class="hero-banner-quote">与其感慨路难行，不如马上出发</p>
    </div>
</section>
```

- [ ] **Step 4: 验证 HTML 修改**

```bash
grep -A 5 "hero-banner" hugo-site/themes/custom-theme/layouts/index.html
```

预期输出：包含新的 hero-banner 结构

- [ ] **Step 5: 在 baseof.html 中引入 hero-banner.css**

检查并修改 `hugo-site/themes/custom-theme/layouts/_default/baseof.html`，在 head 部分添加：

```html
<link rel="stylesheet" href="/css/hero-banner.css">
```

- [ ] **Step 6: 创建 JavaScript 实现滚动效果**

创建 `hugo-site/themes/custom-theme/static/js/hero-banner.js`：

```javascript
document.addEventListener('DOMContentLoaded', function() {
  const heroBanner = document.getElementById('heroBanner');
  
  if (heroBanner) {
    let lastScrollTop = 0;
    
    window.addEventListener('scroll', function() {
      const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
      
      if (scrollTop > 100) {
        heroBanner.classList.add('scrolled');
      } else {
        heroBanner.classList.remove('scrolled');
      }
      
      lastScrollTop = scrollTop;
    });
  }
});
```

- [ ] **Step 7: 验证 JavaScript 文件创建**

```bash
ls -la hugo-site/themes/custom-theme/static/js/hero-banner.js
```

预期输出：文件存在

- [ ] **Step 8: 在 baseof.html 中引入 JavaScript**

在 `baseof.html` 的 body 结束标签前添加：

```html
<script src="/js/hero-banner.js"></script>
```

- [ ] **Step 9: 测试 Banner 显示效果**

```bash
cd hugo-site && hugo server -D
```

访问 http://localhost:1313，验证：
- Banner 显示在页面顶部
- 包含标题、副标题和引言
- 滚动时 Banner 缩小

- [ ] **Step 10: 提交 Task 1 更改**

```bash
git add hugo-site/themes/custom-theme/static/css/hero-banner.css
git add hugo-site/themes/custom-theme/static/js/hero-banner.js
git add hugo-site/themes/custom-theme/layouts/index.html
git add hugo-site/themes/custom-theme/layouts/_default/baseof.html
git commit -m "feat: 添加固定顶部 Hero Banner 组件"
```

---

### Task 2: 优化 Banner 响应式设计

**Files:**
- Modify: `hugo-site/themes/custom-theme/static/css/hero-banner.css:60-80`

- [ ] **Step 1: 添加平板端样式**

在 `hero-banner.css` 文件末尾添加平板端适配：

```css
@media (max-width: 1024px) and (min-width: 769px) {
  .hero-banner {
    padding: 1.75rem 1.5rem;
  }
  
  .hero-banner-title {
    font-size: 1.75rem;
  }
  
  .hero-banner-subtitle {
    font-size: 0.95rem;
  }
  
  .hero-banner-quote {
    font-size: 1.05rem;
  }
}
```

- [ ] **Step 2: 添加小屏幕手机适配**

```css
@media (max-width: 480px) {
  .hero-banner {
    padding: 1rem 0.75rem;
  }
  
  .hero-banner-title {
    font-size: 1.25rem;
  }
  
  .hero-banner-subtitle {
    font-size: 0.85rem;
  }
  
  .hero-banner-quote {
    font-size: 0.9rem;
    display: none;
  }
  
  .hero-banner.scrolled {
    padding: 0.75rem;
  }
}
```

- [ ] **Step 3: 测试响应式效果**

```bash
cd hugo-site && hugo server -D
```

使用浏览器开发者工具测试不同屏幕尺寸：
- 桌面端（>1024px）
- 平板端（768-1024px）
- 手机端（<768px）
- 小屏手机（<480px）

- [ ] **Step 4: 提交响应式优化**

```bash
git add hugo-site/themes/custom-theme/static/css/hero-banner.css
git commit -m "feat: 优化 Hero Banner 响应式设计"
```

---

## Chunk 2: 个人头像与名字区域重新设计

### Task 3: 重新设计侧边栏头像区域

**Files:**
- Modify: `hugo-site/themes/custom-theme/layouts/partials/sidebar.html:3-10`
- Modify: `hugo-site/themes/custom-theme/static/css/sidebar.css:30-64`

- [ ] **Step 1: 修改侧边栏 HTML 结构**

在 `sidebar.html` 中，将头像区域（第 3-10 行）修改为：

```html
<div class="sidebar-avatar">
    <div class="avatar-wrapper">
        <img src="/images/avatar.jpg" alt="{{ .Site.Params.author }}" class="avatar-img">
    </div>
    <h3 class="sidebar-name">高傲的电工李</h3>
    <p class="sidebar-title">机电工程师 | 机器学习爱好者</p>
    <div class="sidebar-contact">
        <span class="contact-item">📍 中国</span>
        <span class="contact-item">📧 {{ .Site.Params.social.email }}</span>
    </div>
</div>
```

- [ ] **Step 2: 更新侧边栏 CSS 样式**

修改 `sidebar.css` 中的头像区域样式（第 30-64 行）：

```css
/* Avatar Section */
.sidebar-avatar {
    text-align: center;
    padding-bottom: var(--spacing-md);
    border-bottom: 1px solid var(--color-border);
}

.avatar-wrapper {
    width: 120px;
    height: 120px;
    margin: 0 auto var(--spacing-sm);
    border-radius: 50%;
    overflow: hidden;
    border: 3px solid var(--color-accent);
    box-shadow: var(--shadow-md);
    transition: transform 0.3s ease;
}

.avatar-wrapper:hover {
    transform: scale(1.05) rotate(5deg);
}

.avatar-img {
    width: 100%;
    height: 100%;
    object-fit: cover;
}

.sidebar-name {
    font-family: var(--font-display);
    font-size: 1.75rem;
    font-weight: 700;
    color: var(--color-text);
    margin-bottom: var(--spacing-xs);
    line-height: 1.2;
}

.sidebar-title {
    font-size: 1rem;
    font-weight: 400;
    color: var(--color-text-secondary);
    margin-bottom: var(--spacing-sm);
    line-height: 1.5;
}

.sidebar-contact {
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
    font-size: 0.875rem;
    color: var(--color-text-secondary);
}

.contact-item {
    display: inline-flex;
    align-items: center;
    gap: 0.25rem;
}
```

- [ ] **Step 3: 验证 HTML 修改**

```bash
grep -A 10 "sidebar-avatar" hugo-site/themes/custom-theme/layouts/partials/sidebar.html
```

预期输出：包含新的头像区域结构

- [ ] **Step 4: 测试头像区域显示**

```bash
cd hugo-site && hugo server -D
```

验证：
- 头像尺寸为 120x120px
- 姓名字体大小为 1.75rem，加粗
- 身份标签显示正确
- 联系方式显示正确
- hover 时头像有放大和旋转效果

- [ ] **Step 5: 提交头像区域更改**

```bash
git add hugo-site/themes/custom-theme/layouts/partials/sidebar.html
git add hugo-site/themes/custom-theme/static/css/sidebar.css
git commit -m "feat: 重新设计侧边栏头像和个人信息区域"
```

---

### Task 4: 更新配色方案

**Files:**
- Modify: `hugo-site/themes/custom-theme/static/css/variables.css:1-43`

- [ ] **Step 1: 更新 variables.css 配色**

修改 `variables.css` 文件，替换为新的配色方案：

```css
:root {
  /* 主色调 */
  --primary-color: #2c3e50;
  --secondary-color: #34495e;
  --accent-color: #3498db;
  --text-color: #2c3e50;
  --light-gray: #95a5a6;
  
  /* 背景色 */
  --bg-primary: #ffffff;
  --bg-secondary: #f8f9fa;
  
  /* 文字颜色 */
  --text-primary: #2c3e50;
  --text-secondary: #6c757d;
  
  /* 边框和阴影 */
  --border-color: #e9ecef;
  --shadow: rgba(0, 0, 0, 0.1);
  --shadow-md: 0 4px 6px rgba(0, 0, 0, 0.1);
  
  /* 代码块 */
  --code-bg: #f5f5f5;
  
  /* 字体 */
  --font-sans: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans SC", "Microsoft YaHei", sans-serif;
  --font-mono: "SF Mono", Monaco, "Cascadia Code", "Roboto Mono", Consolas, monospace;
  --font-display: 'Playfair Display', 'Noto Serif SC', serif;
  --font-body: 'IBM Plex Sans', 'Noto Sans SC', sans-serif;
  
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
  
  /* 分类颜色（从 hugo.toml 同步） */
  --color-life: #e74c3c;
  --color-thought: #9b59b6;
  --color-engineering: #3498db;
  --color-code: #2ecc71;
  --color-ai: #f39c12;
}

[data-theme="dark"] {
  --primary-color: #5dade2;
  --secondary-color: #85c1e9;
  --accent-color: #5dade2;
  --text-color: #e4e4e4;
  --light-gray: #7f8c8d;
  
  --bg-primary: #1a1a1a;
  --bg-secondary: #2d2d2d;
  --text-primary: #e4e4e4;
  --text-secondary: #a0a0a0;
  --border-color: #404040;
  --shadow: rgba(0, 0, 0, 0.3);
  --shadow-md: 0 4px 6px rgba(0, 0, 0, 0.3);
  --code-bg: #2d2d2d;
}
```

- [ ] **Step 2: 验证配色更新**

```bash
grep "primary-color" hugo-site/themes/custom-theme/static/css/variables.css
```

预期输出：包含新的配色定义

- [ ] **Step 3: 测试配色效果**

```bash
cd hugo-site && hugo server -D
```

验证：
- 主色调为深蓝灰（#2c3e50）
- 强调色为蓝色（#3498db）
- 暗色模式配色正确

- [ ] **Step 4: 提交配色方案更新**

```bash
git add hugo-site/themes/custom-theme/static/css/variables.css
git commit -m "feat: 更新配色方案，统一视觉风格"
```

---

## Chunk 3: 分类导航优化

### Task 5: 移除分类图标（方案 A）

**Files:**
- Modify: `hugo-site/themes/custom-theme/layouts/partials/category-nav.html:8,21`
- Modify: `hugo-site/themes/custom-theme/static/css/category-nav.css:54-57`

- [ ] **Step 1: 修改 category-nav.html 移除图标**

在 `category-nav.html` 中，注释掉或删除图标元素：

修改前（第 8 行）：
```html
<span class="category-icon">📚</span>
```

修改后：
```html
<!-- 图标已移除 -->
```

修改前（第 21 行）：
```html
<span class="category-icon">{{ $value.icon }}</span>
```

修改后：
```html
<!-- 图标已移除 -->
```

- [ ] **Step 2: 更新 category-nav.css 样式**

修改 `category-nav.css`，移除图标相关样式（第 54-57 行）：

```css
/* 图标样式已移除 */
```

同时更新 `.category-link` 样式，移除 gap：

```css
.category-link {
  display: flex;
  align-items: center;
  gap: 0.25rem;  /* 从 0.5rem 改为 0.25rem */
  padding: 0.5rem 1.25rem;
  background: var(--color-bg);
  border: 1px solid var(--color-border);
  border-radius: 999px;
  color: var(--color-text);
  text-decoration: none;
  transition: all 0.2s ease;
  font-size: 0.9rem;
  white-space: nowrap;
}
```

- [ ] **Step 3: 增强分类名字体样式**

在 `category-nav.css` 中更新 `.category-name` 样式：

```css
.category-name {
  font-weight: 600;  /* 从 500 改为 600 */
  font-size: 0.95rem;  /* 增加字号 */
  letter-spacing: 0.02em;
}
```

- [ ] **Step 4: 验证修改**

```bash
grep -v "category-icon" hugo-site/themes/custom-theme/layouts/partials/category-nav.html | grep -c "category-name"
```

预期输出：2（两个 category-name 元素）

- [ ] **Step 5: 测试分类导航显示**

```bash
cd hugo-site && hugo server -D
```

验证：
- 分类导航不显示图标
- 分类名字体加粗
- hover 效果正常

- [ ] **Step 6: 提交分类导航优化**

```bash
git add hugo-site/themes/custom-theme/layouts/partials/category-nav.html
git add hugo-site/themes/custom-theme/static/css/category-nav.css
git commit -m "feat: 移除分类图标，优化文字样式"
```

---

### Task 6: 添加分类导航 hover 状态优化

**Files:**
- Modify: `hugo-site/themes/custom-theme/static/css/category-nav.css:47-52`

- [ ] **Step 1: 增强 hover 效果**

修改 `category-nav.css` 中的 hover 样式：

```css
.category-link:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
  border-color: var(--category-color, var(--accent-color));
  background: var(--color-surface);
  font-weight: 600;
}
```

- [ ] **Step 2: 添加 active 状态增强**

更新 `.category-link.active` 样式：

```css
.category-link.active {
  background: var(--accent-color);
  color: white;
  border-color: var(--accent-color);
  font-weight: 700;
  box-shadow: 0 2px 8px rgba(52, 152, 219, 0.3);
}

.category-link.active .category-count {
  background: rgba(255, 255, 255, 0.25);
  color: white;
  font-weight: 600;
}
```

- [ ] **Step 3: 测试交互效果**

```bash
cd hugo-site && hugo server -D
```

验证：
- hover 时有明显的阴影和位移效果
- active 状态有明显的视觉区分
- 过渡动画流畅

- [ ] **Step 4: 提交 hover 状态优化**

```bash
git add hugo-site/themes/custom-theme/static/css/category-nav.css
git commit -m "feat: 增强分类导航 hover 和 active 状态"
```

---

## Chunk 4: 首页文章卡片设计

### Task 7: 创建卡片式布局组件

**Files:**
- Modify: `hugo-site/themes/custom-theme/layouts/index.html:17-37`
- Create: `hugo-site/themes/custom-theme/static/css/article-card.css`

- [ ] **Step 1: 创建 article-card.css 文件**

创建新的 CSS 文件用于文章卡片样式：

```css
/* Article Card - 卡片式设计 */
.articles-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
  gap: 2rem;
  padding: 2rem 0;
  max-width: 1400px;
  margin: 0 auto;
}

.article-card {
  background: white;
  border-radius: 12px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.08);
  transition: transform 0.3s ease, box-shadow 0.3s ease;
  overflow: hidden;
  display: flex;
  flex-direction: column;
  height: 100%;
}

.article-card:hover {
  transform: translateY(-4px);
  box-shadow: 0 8px 24px rgba(0,0,0,0.12);
}

.article-card-cover {
  width: 100%;
  height: 200px;
  overflow: hidden;
  background: var(--color-accent-soft, #f4e8e3);
}

.article-card-cover img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  transition: transform 0.3s ease;
}

.article-card:hover .article-card-cover img {
  transform: scale(1.05);
}

.article-card-content {
  padding: 1.5rem;
  flex: 1;
  display: flex;
  flex-direction: column;
}

.article-card-meta {
  display: flex;
  align-items: center;
  gap: 1rem;
  margin-bottom: 1rem;
  font-size: 0.875rem;
}

.article-card-category {
  display: inline-flex;
  align-items: center;
  padding: 0.25rem 0.75rem;
  background: var(--accent-color);
  color: white;
  border-radius: 999px;
  font-weight: 500;
  font-size: 0.8rem;
}

.article-card-reading-time {
  color: var(--text-secondary);
  display: flex;
  align-items: center;
  gap: 0.25rem;
}

.article-card-title {
  font-family: var(--font-display);
  font-size: 1.5rem;
  font-weight: 600;
  color: var(--text-primary);
  margin-bottom: 0.75rem;
  line-height: 1.3;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

.article-card-excerpt {
  font-size: 0.95rem;
  color: var(--text-secondary);
  line-height: 1.6;
  margin-bottom: 1rem;
  flex: 1;
  display: -webkit-box;
  -webkit-line-clamp: 3;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

.article-card-footer {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding-top: 1rem;
  border-top: 1px solid var(--border-color);
  font-size: 0.875rem;
  color: var(--text-secondary);
}

.article-card-date {
  display: flex;
  align-items: center;
  gap: 0.25rem;
}

.article-card-author {
  font-weight: 500;
}

/* 响应式 */
@media (max-width: 768px) {
  .articles-grid {
    grid-template-columns: 1fr;
    gap: 1.5rem;
    padding: 1.5rem 1rem;
  }
  
  .article-card-title {
    font-size: 1.25rem;
  }
}

@media (max-width: 480px) {
  .article-card-content {
    padding: 1rem;
  }
  
  .article-card-title {
    font-size: 1.15rem;
  }
  
  .article-card-excerpt {
    font-size: 0.9rem;
  }
}
```

- [ ] **Step 2: 验证 CSS 文件创建**

```bash
ls -la hugo-site/themes/custom-theme/static/css/article-card.css
```

预期输出：文件存在

- [ ] **Step 3: 修改 index.html 使用卡片布局**

在 `hugo-site/themes/custom-theme/layouts/index.html` 中，将 posts-list 部分（第 17-37 行）修改为：

```html
<section class="posts-section">
    {{ $paginator := .Paginate (where .Site.RegularPages "Type" "posts") }}
    <div class="articles-grid">
        {{ range $paginator.Pages }}
        <article class="article-card">
            <a href="{{ .Permalink }}" style="text-decoration: none; color: inherit;">
                {{ if .Params.cover }}
                <div class="article-card-cover">
                    <img src="{{ .Params.cover }}" alt="{{ .Title }}" loading="lazy">
                </div>
                {{ end }}
                
                <div class="article-card-content">
                    <div class="article-card-meta">
                        {{ with .Params.categories }}
                        <span class="article-card-category">{{ index . 0 }}</span>
                        {{ end }}
                        {{ if .Params.readingTime }}
                        <span class="article-card-reading-time">📖 {{ .Params.readingTime }} 分钟</span>
                        {{ end }}
                    </div>
                    
                    <h2 class="article-card-title">{{ .Title }}</h2>
                    
                    <p class="article-card-excerpt">
                        {{ if .Params.description }}
                            {{ .Params.description }}
                        {{ else }}
                            {{ .Summary | truncate 150 }}
                        {{ end }}
                    </p>
                    
                    <div class="article-card-footer">
                        <time class="article-card-date" datetime="{{ .Date.Format "2006-01-02" }}">
                            📅 {{ .Date.Format "2006年1月2日" }}
                        </time>
                        <span class="article-card-author">{{ .Site.Params.author }}</span>
                    </div>
                </div>
            </a>
        </article>
        {{ end }}
    </div>
    
    {{ if gt $paginator.TotalPages 1 }}
    <nav class="pagination">
        {{ if $paginator.HasPrev }}
        <a href="{{ $paginator.Prev.URL }}" class="pagination-link">← 上一页</a>
        {{ end }}
        <span class="pagination-info">第 {{ $paginator.PageNumber }} 页 / 共 {{ $paginator.TotalPages }} 页</span>
        {{ if $paginator.HasNext }}
        <a href="{{ $paginator.Next.URL }}" class="pagination-link">下一页 →</a>
        {{ end }}
    </nav>
    {{ end }}
</section>
```

- [ ] **Step 4: 在 baseof.html 中引入 article-card.css**

在 `baseof.html` 的 head 部分添加：

```html
<link rel="stylesheet" href="/css/article-card.css">
```

- [ ] **Step 5: 测试卡片布局显示**

```bash
cd hugo-site && hugo server -D
```

验证：
- 文章以卡片形式显示
- 网格布局响应式
- 封面图显示正确
- 分类标签显示
- hover 效果正常

- [ ] **Step 6: 提交卡片布局更改**

```bash
git add hugo-site/themes/custom-theme/static/css/article-card.css
git add hugo-site/themes/custom-theme/layouts/index.html
git add hugo-site/themes/custom-theme/layouts/_default/baseof.html
git commit -m "feat: 实现文章卡片式布局"
```

---

### Task 8: 添加封面图懒加载和占位符

**Files:**
- Modify: `hugo-site/themes/custom-theme/static/css/article-card.css:24-30`
- Create: `hugo-site/themes/custom-theme/static/js/lazy-load.js`

- [ ] **Step 1: 添加占位符样式**

在 `article-card.css` 中更新封面图样式：

```css
.article-card-cover {
  width: 100%;
  height: 200px;
  overflow: hidden;
  background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
  position: relative;
}

.article-card-cover::before {
  content: '📷';
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  font-size: 3rem;
  opacity: 0.3;
}

.article-card-cover img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  transition: transform 0.3s ease, opacity 0.3s ease;
  opacity: 0;
}

.article-card-cover img.loaded {
  opacity: 1;
}
```

- [ ] **Step 2: 创建懒加载 JavaScript**

创建 `hugo-site/themes/custom-theme/static/js/lazy-load.js`：

```javascript
document.addEventListener('DOMContentLoaded', function() {
  const images = document.querySelectorAll('.article-card-cover img[loading="lazy"]');
  
  if ('IntersectionObserver' in window) {
    const imageObserver = new IntersectionObserver((entries, observer) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          const img = entry.target;
          img.addEventListener('load', function() {
            img.classList.add('loaded');
          });
          
          if (img.complete) {
            img.classList.add('loaded');
          }
          
          observer.unobserve(img);
        }
      });
    });
    
    images.forEach(img => imageObserver.observe(img));
  } else {
    images.forEach(img => {
      img.addEventListener('load', function() {
        img.classList.add('loaded');
      });
      if (img.complete) {
        img.classList.add('loaded');
      }
    });
  }
});
```

- [ ] **Step 3: 在 baseof.html 中引入 JavaScript**

在 `baseof.html` 的 body 结束标签前添加：

```html
<script src="/js/lazy-load.js"></script>
```

- [ ] **Step 4: 测试懒加载效果**

```bash
cd hugo-site && hugo server -D
```

验证：
- 图片加载前显示占位符
- 图片加载后平滑显示
- 滚动时图片按需加载

- [ ] **Step 5: 提交懒加载功能**

```bash
git add hugo-site/themes/custom-theme/static/css/article-card.css
git add hugo-site/themes/custom-theme/static/js/lazy-load.js
git add hugo-site/themes/custom-theme/layouts/_default/baseof.html
git commit -m "feat: 添加封面图懒加载和占位符"
```

---

## Chunk 5: 整体测试和优化

### Task 9: 响应式布局全面测试

**Files:**
- Test: 所有修改的 HTML 和 CSS 文件

- [ ] **Step 1: 桌面端测试（>1400px）**

```bash
cd hugo-site && hugo server -D
```

使用浏览器访问 http://localhost:1313，测试：
- Hero Banner 显示完整
- 侧边栏固定显示
- 文章卡片 3-4 列网格
- 所有交互效果正常

- [ ] **Step 2: 平板端测试（768-1399px）**

使用浏览器开发者工具切换到平板视图，测试：
- Hero Banner 适配正确
- 侧边栏可切换
- 文章卡片 2-3 列网格
- 分类导航横向滚动

- [ ] **Step 3: 手机端测试（<768px）**

切换到手机视图，测试：
- Hero Banner 缩小显示
- 侧边栏全屏显示
- 文章卡片单列布局
- 所有文字可读

- [ ] **Step 4: 小屏手机测试（<480px）**

切换到小屏手机视图（iPhone SE），测试：
- Hero Banner 引言隐藏
- 卡片内容紧凑
- 按钮和链接可点击
- 无横向滚动

- [ ] **Step 5: 记录测试结果**

创建测试报告：

```bash
cat > hugo-site/RESPONSIVE_TEST_REPORT.md << 'TESTEOF'
# 响应式测试报告

## 测试日期
2026-03-14

## 测试设备
- 桌面端：Chrome 1920x1080
- 平板端：iPad 768x1024
- 手机端：iPhone 12 390x844
- 小屏手机：iPhone SE 375x667

## 测试结果
- [ ] 桌面端：通过
- [ ] 平板端：通过
- [ ] 手机端：通过
- [ ] 小屏手机：通过

## 发现的问题
（记录测试中发现的问题）

## 修复建议
（记录需要修复的内容）
TESTEOF
```

- [ ] **Step 6: 提交测试报告**

```bash
git add hugo-site/RESPONSIVE_TEST_REPORT.md
git commit -m "docs: 添加响应式测试报告"
```

---

### Task 10: 性能优化

**Files:**
- Modify: `hugo-site/themes/custom-theme/static/css/*.css`
- Create: `hugo-site/themes/custom-theme/static/css/critical.css`

- [ ] **Step 1: 提取关键 CSS**

创建 `critical.css` 包含首屏必需的样式：

```css
/* Critical CSS - 首屏加载 */
body {
  margin: 0;
  font-family: var(--font-sans);
  background: var(--bg-primary);
  color: var(--text-primary);
}

.hero-banner {
  position: sticky;
  top: 0;
  z-index: 1000;
  background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
  padding: 2rem;
  text-align: center;
}

.hero-banner-title {
  font-size: 2rem;
  font-weight: 700;
  color: white;
  margin: 0;
}
```

- [ ] **Step 2: 优化 CSS 加载顺序**

在 `baseof.html` 中调整 CSS 加载顺序：

```html
<head>
  <!-- 关键 CSS 内联 -->
  <style>
    /* 从 critical.css 复制内容 */
  </style>
  
  <!-- 其他 CSS 异步加载 -->
  <link rel="preload" href="/css/variables.css" as="style" onload="this.onload=null;this.rel='stylesheet'">
  <link rel="preload" href="/css/main.css" as="style" onload="this.onload=null;this.rel='stylesheet'">
  <link rel="preload" href="/css/hero-banner.css" as="style" onload="this.onload=null;this.rel='stylesheet'">
  <link rel="preload" href="/css/article-card.css" as="style" onload="this.onload=null;this.rel='stylesheet'">
  
  <!-- 降级支持 -->
  <noscript>
    <link rel="stylesheet" href="/css/variables.css">
    <link rel="stylesheet" href="/css/main.css">
    <link rel="stylesheet" href="/css/hero-banner.css">
    <link rel="stylesheet" href="/css/article-card.css">
  </noscript>
</head>
```

- [ ] **Step 3: 压缩图片资源**

检查并优化图片大小：

```bash
cd hugo-site/static/images
ls -lh *.jpg *.png
```

如果图片过大（>500KB），建议压缩或转换为 WebP 格式。

- [ ] **Step 4: 添加字体预加载**

在 `baseof.html` 中添加字体预加载：

```html
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="preload" href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap" as="style" onload="this.onload=null;this.rel='stylesheet'">
```

- [ ] **Step 5: 运行性能测试**

```bash
cd hugo-site && hugo server -D
```

使用 Chrome DevTools Lighthouse 运行性能测试：
- Performance: 目标 >90
- Accessibility: 目标 >90
- Best Practices: 目标 >90
- SEO: 目标 >90

- [ ] **Step 6: 提交性能优化**

```bash
git add hugo-site/themes/custom-theme/static/css/critical.css
git add hugo-site/themes/custom-theme/layouts/_default/baseof.html
git commit -m "perf: 优化 CSS 加载和字体预加载"
```

---

### Task 11: 暗色模式适配

**Files:**
- Modify: `hugo-site/themes/custom-theme/static/css/hero-banner.css`
- Modify: `hugo-site/themes/custom-theme/static/css/article-card.css`
- Create: `hugo-site/themes/custom-theme/static/js/theme-toggle.js`

- [ ] **Step 1: 添加暗色模式 CSS 变量**

在 `hero-banner.css` 中添加暗色模式支持：

```css
[data-theme="dark"] .hero-banner {
  background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
}

[data-theme="dark"] .hero-banner-title {
  color: #e4e4e4;
}

[data-theme="dark"] .hero-banner-subtitle,
[data-theme="dark"] .hero-banner-quote {
  color: rgba(228, 228, 228, 0.85);
}
```

在 `article-card.css` 中添加：

```css
[data-theme="dark"] .article-card {
  background: #2d2d2d;
  box-shadow: 0 2px 8px rgba(0,0,0,0.3);
}

[data-theme="dark"] .article-card:hover {
  box-shadow: 0 8px 24px rgba(0,0,0,0.4);
}

[data-theme="dark"] .article-card-title {
  color: #e4e4e4;
}

[data-theme="dark"] .article-card-excerpt {
  color: #a0a0a0;
}

[data-theme="dark"] .article-card-footer {
  border-top-color: #404040;
  color: #a0a0a0;
}
```

- [ ] **Step 2: 创建主题切换 JavaScript**

创建 `hugo-site/themes/custom-theme/static/js/theme-toggle.js`：

```javascript
document.addEventListener('DOMContentLoaded', function() {
  const themeToggle = document.getElementById('themeToggle');
  const html = document.documentElement;
  
  // 读取保存的主题
  const savedTheme = localStorage.getItem('theme') || 'light';
  html.setAttribute('data-theme', savedTheme);
  
  if (themeToggle) {
    themeToggle.addEventListener('click', function() {
      const currentTheme = html.getAttribute('data-theme');
      const newTheme = currentTheme === 'light' ? 'dark' : 'light';
      
      html.setAttribute('data-theme', newTheme);
      localStorage.setItem('theme', newTheme);
      
      // 更新按钮图标
      this.textContent = newTheme === 'light' ? '🌙' : '☀️';
    });
    
    // 设置初始图标
    themeToggle.textContent = savedTheme === 'light' ? '🌙' : '☀️';
  }
});
```

- [ ] **Step 3: 添加主题切换按钮**

在 `baseof.html` 的 body 中添加切换按钮：

```html
<button id="themeToggle" class="theme-toggle" aria-label="切换主题">
  🌙
</button>
```

添加按钮样式到 `main.css`：

```css
.theme-toggle {
  position: fixed;
  bottom: 2rem;
  right: 2rem;
  width: 50px;
  height: 50px;
  border-radius: 50%;
  background: var(--color-surface);
  border: 1px solid var(--color-border);
  font-size: 1.5rem;
  cursor: pointer;
  z-index: 1000;
  box-shadow: var(--shadow-md);
  transition: all 0.3s ease;
}

.theme-toggle:hover {
  transform: scale(1.1) rotate(15deg);
  box-shadow: 0 4px 12px rgba(0,0,0,0.2);
}
```

- [ ] **Step 4: 在 baseof.html 中引入 JavaScript**

```html
<script src="/js/theme-toggle.js"></script>
```

- [ ] **Step 5: 测试暗色模式**

```bash
cd hugo-site && hugo server -D
```

验证：
- 点击切换按钮可切换主题
- 暗色模式样式正确
- 主题选择被保存
- 页面刷新后主题保持

- [ ] **Step 6: 提交暗色模式功能**

```bash
git add hugo-site/themes/custom-theme/static/css/hero-banner.css
git add hugo-site/themes/custom-theme/static/css/article-card.css
git add hugo-site/themes/custom-theme/static/css/main.css
git add hugo-site/themes/custom-theme/static/js/theme-toggle.js
git add hugo-site/themes/custom-theme/layouts/_default/baseof.html
git commit -m "feat: 添加暗色模式支持和主题切换功能"
```

---

### Task 12: 可访问性优化

**Files:**
- Modify: `hugo-site/themes/custom-theme/layouts/index.html`
- Modify: `hugo-site/themes/custom-theme/layouts/partials/sidebar.html`
- Modify: `hugo-site/themes/custom-theme/static/css/main.css`

- [ ] **Step 1: 添加 ARIA 标签**

在 `index.html` 中添加语义化标签和 ARIA 属性：

```html
<section class="hero-banner" id="heroBanner" role="banner" aria-label="网站标题">
  <div class="hero-banner-content">
    <h1 class="hero-banner-title">{{ .Site.Title }}</h1>
    <p class="hero-banner-subtitle">{{ .Site.Params.avatarDesc }}</p>
    <p class="hero-banner-quote">与其感慨路难行，不如马上出发</p>
  </div>
</section>

<section class="posts-section" role="main" aria-label="文章列表">
  <div class="articles-grid" role="list">
    {{ range $paginator.Pages }}
    <article class="article-card" role="listitem">
      <a href="{{ .Permalink }}" aria-label="阅读文章：{{ .Title }}">
        <!-- 卡片内容 -->
      </a>
    </article>
    {{ end }}
  </div>
</section>
```

- [ ] **Step 2: 优化键盘导航**

在 `main.css` 中添加焦点样式：

```css
/* 键盘导航焦点样式 */
a:focus-visible,
button:focus-visible {
  outline: 3px solid var(--accent-color);
  outline-offset: 3px;
  border-radius: 4px;
}

.article-card a:focus-visible {
  outline: 3px solid var(--accent-color);
  outline-offset: -3px;
}

/* 跳过导航链接 */
.skip-to-content {
  position: absolute;
  top: -40px;
  left: 0;
  background: var(--accent-color);
  color: white;
  padding: 0.5rem 1rem;
  text-decoration: none;
  z-index: 10000;
}

.skip-to-content:focus {
  top: 0;
}
```

- [ ] **Step 3: 添加跳过导航链接**

在 `baseof.html` 的 body 开始处添加：

```html
<a href="#main-content" class="skip-to-content">跳到主内容</a>
```

在 `index.html` 的 posts-section 添加 id：

```html
<section class="posts-section" id="main-content" role="main">
```

- [ ] **Step 4: 优化图片 alt 文本**

确保所有图片都有描述性的 alt 文本：

```html
<img src="{{ .Params.cover }}" alt="{{ .Title }} - 文章封面图" loading="lazy">
```

- [ ] **Step 5: 测试可访问性**

```bash
cd hugo-site && hugo server -D
```

使用键盘测试：
- Tab 键可以导航所有交互元素
- Enter 键可以激活链接和按钮
- 焦点样式清晰可见
- 跳过导航链接工作正常

使用 Lighthouse 测试可访问性评分，目标 >90。

- [ ] **Step 6: 提交可访问性优化**

```bash
git add hugo-site/themes/custom-theme/layouts/index.html
git add hugo-site/themes/custom-theme/layouts/partials/sidebar.html
git add hugo-site/themes/custom-theme/layouts/_default/baseof.html
git add hugo-site/themes/custom-theme/static/css/main.css
git commit -m "a11y: 优化可访问性，添加 ARIA 标签和键盘导航"
```

---

## Chunk 6: 文档和部署

### Task 13: 创建示例文档

**Files:**
- Create: `hugo-site/content/posts/2026-03-14-example-post.md`

- [ ] **Step 1: 创建示例文章**

创建一篇包含所有 front-matter 字段的示例文章：

```bash
cat > hugo-site/content/posts/2026-03-14-example-post.md << 'EXAMPLEEOF'
---
title: "示例文章：展示卡片样式"
date: 2026-03-14
description: "这是一篇示例文章，用于展示新的文章卡片设计。包含封面图、分类标签、阅读时间等完整元素。"
categories: ["code"]
tags: ["Hugo", "设计", "前端"]
cover: "/images/background-cover.jpg"
readingTime: 5
featured: true
---

# 示例文章

这是一篇示例文章，用于展示博客的新设计。

## 卡片元素

文章卡片包含以下元素：

1. 封面图（可选）
2. 分类标签
3. 阅读时间
4. 文章标题
5. 文章摘要
6. 发布日期
7. 作者信息

## 设计特点

- 卡片式布局
- 响应式网格
- hover 动画效果
- 懒加载图片
- 暗色模式支持

## 使用方法

在文章的 front-matter 中添加以下字段：

```yaml
cover: /images/your-cover.jpg
readingTime: 5
featured: true
```

这样就可以显示完整的卡片效果了。
EXAMPLEEOF
```

- [ ] **Step 2: 验证示例文章**

```bash
ls -la hugo-site/content/posts/2026-03-14-example-post.md
```

预期输出：文件存在

- [ ] **Step 3: 测试示例文章显示**

```bash
cd hugo-site && hugo server -D
```

访问首页，验证示例文章卡片显示完整。

- [ ] **Step 4: 提交示例文档**

```bash
git add hugo-site/content/posts/2026-03-14-example-post.md
git commit -m "docs: 添加示例文章展示卡片样式"
```

---

### Task 14: 更新项目文档

**Files:**
- Create: `hugo-site/HOMEPAGE_REDESIGN.md`
- Modify: `hugo-site/README.md`

- [ ] **Step 1: 创建重新设计文档**

创建详细的设计文档：

```bash
cat > hugo-site/HOMEPAGE_REDESIGN.md << 'DESIGNEOF'
# 首页重新设计文档

## 概述

本次重新设计优化了博客首页的用户体验和视觉呈现，包括以下主要改进：

1. 固定顶部 Hero Banner
2. 优化的个人信息展示
3. 简化的分类导航
4. 卡片式文章布局
5. 暗色模式支持
6. 性能和可访问性优化

## 设计原则

- 编辑式极简主义
- 移动优先的响应式设计
- 注重可访问性
- 性能优化

## 技术实现

### 1. Hero Banner

- 位置：页面顶部，sticky 定位
- 内容：网站标题、副标题、个人 slogan
- 特性：滚动时自动缩小

### 2. 文章卡片

- 布局：响应式网格（auto-fill）
- 元素：封面图、分类、标题、摘要、日期
- 交互：hover 动画、懒加载

### 3. 响应式断点

- 桌面端：>1400px（3-4 列）
- 平板端：768-1399px（2-3 列）
- 手机端：<768px（单列）
- 小屏手机：<480px（紧凑布局）

## 文件结构

新增文件：
- static/css/hero-banner.css
- static/css/article-card.css
- static/css/critical.css
- static/js/hero-banner.js
- static/js/lazy-load.js
- static/js/theme-toggle.js

修改文件：
- layouts/index.html
- layouts/partials/sidebar.html
- layouts/_default/baseof.html
- static/css/variables.css
- static/css/sidebar.css
- static/css/category-nav.css

## 使用指南

### 文章 Front Matter

```yaml
---
title: "文章标题"
date: 2026-03-14
description: "文章描述（用于卡片摘要）"
categories: ["code"]
tags: ["标签1", "标签2"]
cover: "/images/cover.jpg"  # 封面图（可选）
readingTime: 5              # 阅读时间（可选）
featured: true              # 精选文章（可选）
---
```

### 主题切换

用户可以通过右下角的按钮切换亮色/暗色模式，选择会保存在 localStorage 中。

## 性能指标

目标 Lighthouse 评分：
- Performance: >90
- Accessibility: >90
- Best Practices: >90
- SEO: >90

## 浏览器兼容性

- Chrome/Edge: 最新版本
- Firefox: 最新版本
- Safari: 最新版本
- 移动浏览器：iOS Safari, Chrome Mobile

## 维护说明

### 添加新分类

在 `hugo.toml` 中的 `[params.categories]` 部分添加：

```toml
[params.categories.new-category]
  name = "分类名称"
  icon = "🎯"
  color = "#ff6b6b"
  description = "分类描述"
```

### 自定义配色

修改 `static/css/variables.css` 中的 CSS 变量。

### 优化图片

建议使用 WebP 格式，尺寸控制在 800x600px 以内。

## 已知问题

（记录已知的问题和待改进项）

## 更新日志

- 2026-03-14: 完成首页重新设计
DESIGNEOF
```

- [ ] **Step 2: 更新 README.md**

在 `hugo-site/README.md` 中添加新功能说明：

```bash
cat >> hugo-site/README.md << 'READMEEOF'

## 最新更新

### 首页重新设计（2026-03-14）

- ✨ 新增固定顶部 Hero Banner
- 🎨 重新设计文章卡片布局
- 🌓 支持亮色/暗色模式切换
- 📱 优化移动端响应式设计
- ♿ 改进可访问性支持
- ⚡ 性能优化（懒加载、关键 CSS）

详细信息请查看 [HOMEPAGE_REDESIGN.md](HOMEPAGE_REDESIGN.md)

## 文章 Front Matter 字段

```yaml
---
title: "文章标题"
date: 2026-03-14
description: "文章描述"
categories: ["code"]
tags: ["标签1", "标签2"]
cover: "/images/cover.jpg"  # 可选
readingTime: 5              # 可选
featured: true              # 可选
---
```
READMEEOF
```

- [ ] **Step 3: 验证文档创建**

```bash
ls -la hugo-site/HOMEPAGE_REDESIGN.md
grep "最新更新" hugo-site/README.md
```

预期输出：文件存在，README 包含更新说明

- [ ] **Step 4: 提交文档更新**

```bash
git add hugo-site/HOMEPAGE_REDESIGN.md
git add hugo-site/README.md
git commit -m "docs: 添加首页重新设计文档"
```

---

### Task 15: 最终验证和部署准备

**Files:**
- Test: 所有修改的文件

- [ ] **Step 1: 运行完整构建**

```bash
cd hugo-site
hugo --gc --minify
```

预期输出：构建成功，无错误

- [ ] **Step 2: 检查构建输出**

```bash
ls -la hugo-site/public/
du -sh hugo-site/public/
```

验证：
- public 目录存在
- 包含所有必需的文件
- 总大小合理（<50MB）

- [ ] **Step 3: 本地预览生产版本**

```bash
cd hugo-site/public
python3 -m http.server 8000
```

访问 http://localhost:8000，验证：
- 所有页面正常显示
- CSS 和 JS 正确加载
- 图片显示正常
- 链接可点击

- [ ] **Step 4: 运行 Lighthouse 审计**

使用 Chrome DevTools 对首页运行 Lighthouse 审计，记录评分：

```bash
cat > hugo-site/LIGHTHOUSE_REPORT.md << 'LIGHTHOUSEEOF'
# Lighthouse 审计报告

## 测试日期
2026-03-14

## 测试页面
- 首页: /
- 文章页: /posts/example-post/
- 分类页: /categories/code/

## 评分结果

### 首页
- Performance: __/100
- Accessibility: __/100
- Best Practices: __/100
- SEO: __/100

### 文章页
- Performance: __/100
- Accessibility: __/100
- Best Practices: __/100
- SEO: __/100

### 分类页
- Performance: __/100
- Accessibility: __/100
- Best Practices: __/100
- SEO: __/100

## 改进建议
（记录 Lighthouse 提出的改进建议）

## 已修复问题
（记录已经修复的问题）
LIGHTHOUSEEOF
```

- [ ] **Step 5: 检查 Git 状态**

```bash
cd /home/wenbo/WenboLi-CN-DE.github.io
git status
git log --oneline -10
```

验证：
- 所有更改已提交
- 提交信息清晰
- 无未跟踪的重要文件

- [ ] **Step 6: 创建功能分支（可选）**

如果需要在合并前进行审查：

```bash
git checkout -b feature/homepage-redesign
git push -u origin feature/homepage-redesign
```

- [ ] **Step 7: 合并到主分支**

如果直接部署：

```bash
git checkout main
git merge feature/homepage-redesign
git push origin main
```

- [ ] **Step 8: 验证 GitHub Actions 部署**

访问 GitHub Actions 页面，确认：
- 工作流自动触发
- 构建成功
- 部署到 GitHub Pages

- [ ] **Step 9: 验证线上网站**

访问 https://wenboli-cn-de.github.io，验证：
- 首页显示正确
- 所有功能正常
- 响应式布局工作
- 暗色模式可切换

- [ ] **Step 10: 最终提交**

```bash
git add hugo-site/LIGHTHOUSE_REPORT.md
git commit -m "docs: 添加 Lighthouse 审计报告"
git push origin main
```

---

## 总结

### 完成的任务

- [x] Task 1: 创建固定顶部 Banner 组件
- [x] Task 2: 优化 Banner 响应式设计
- [x] Task 3: 重新设计侧边栏头像区域
- [x] Task 4: 更新配色方案
- [x] Task 5: 移除分类图标
- [x] Task 6: 添加分类导航 hover 状态优化
- [x] Task 7: 创建卡片式布局组件
- [x] Task 8: 添加封面图懒加载和占位符
- [x] Task 9: 响应式布局全面测试
- [x] Task 10: 性能优化
- [x] Task 11: 暗色模式适配
- [x] Task 12: 可访问性优化
- [x] Task 13: 创建示例文档
- [x] Task 14: 更新项目文档
- [x] Task 15: 最终验证和部署准备

### 关键文件清单

**新增文件：**
- `themes/custom-theme/static/css/hero-banner.css`
- `themes/custom-theme/static/css/article-card.css`
- `themes/custom-theme/static/css/critical.css`
- `themes/custom-theme/static/js/hero-banner.js`
- `themes/custom-theme/static/js/lazy-load.js`
- `themes/custom-theme/static/js/theme-toggle.js`
- `content/posts/2026-03-14-example-post.md`
- `HOMEPAGE_REDESIGN.md`
- `RESPONSIVE_TEST_REPORT.md`
- `LIGHTHOUSE_REPORT.md`

**修改文件：**
- `themes/custom-theme/layouts/index.html`
- `themes/custom-theme/layouts/partials/sidebar.html`
- `themes/custom-theme/layouts/partials/category-nav.html`
- `themes/custom-theme/layouts/_default/baseof.html`
- `themes/custom-theme/static/css/variables.css`
- `themes/custom-theme/static/css/sidebar.css`
- `themes/custom-theme/static/css/category-nav.css`
- `themes/custom-theme/static/css/main.css`
- `README.md`

### 预期效果

1. **视觉改进**：现代化的卡片式布局，清晰的视觉层次
2. **用户体验**：流畅的交互动画，直观的导航
3. **响应式设计**：完美适配各种设备尺寸
4. **性能优化**：快速加载，优秀的 Lighthouse 评分
5. **可访问性**：支持键盘导航和屏幕阅读器

### 后续建议

1. 定期更新文章封面图
2. 监控网站性能指标
3. 收集用户反馈进行迭代
4. 考虑添加搜索功能
5. 优化 SEO 元数据

---

**计划创建完成！准备执行。**
