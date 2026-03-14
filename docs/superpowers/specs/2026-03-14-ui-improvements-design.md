# UI 改进设计文档

**日期**: 2026-03-14  
**项目**: Hugo 博客站点  
**状态**: 已修订（根据 Oracle 审查意见）

## 概述

对博客站点进行三项 UI 改进：
1. 移除侧边栏分类图标（简化视觉）
2. 验证并确保文章按时间倒序排列（新→旧）
3. 标签页面和分类页面卡片美化（复用首页样式）

## 背景

### 当前问题

1. **侧边栏分类图标**: 用户反馈 emoji 图标"太丑了"，与编辑式极简主义设计风格不符
2. **文章排序**: 需要验证首页文章是否按时间倒序排列（最新文章在顶部），并在必要时显式固定排序逻辑
3. **标签/分类页面**: 当前使用简陋的 `.post-preview` 样式，与首页精美的 `.article-card` 样式不一致。注意：`taxonomy.html` 模板同时服务于标签页和分类页

### 设计目标

- 强化"编辑式极简主义"设计原则
- 提升视觉一致性
- 改善用户体验

## 设计方案

### 1. 侧边栏分类图标移除

#### 修改范围

**文件**: `hugo-site/themes/custom-theme/layouts/partials/sidebar.html`

**当前实现**:
```html
<a href="/categories/{{ $key }}" class="category-item" style="--category-color: {{ $value.color }}">
    <span class="category-icon">{{ $value.icon }}</span>
    <span class="category-name">{{ $value.name }}</span>
</a>
```

**修改后**:
```html
<a href="/categories/{{ $key }}" class="category-item" style="--category-color: {{ $value.color }}">
    <span class="category-name">{{ $value.name }}</span>
</a>
```

#### CSS 调整

**文件**: `hugo-site/themes/custom-theme/static/css/sidebar.css`

**修改内容**:
1. 移除 `.category-icon` 样式定义
2. 调整 `.category-item` 布局：
   - 移除 `gap` 属性（不再需要图标与文字间距）
   - 添加左侧彩色边框：`border-left: 3px solid var(--category-color)`
   - 调整 padding：`padding: var(--spacing-sm) var(--spacing-md)`
3. 增强悬停效果（避免布局抖动）：
   - **不改变边框宽度**（保持 3px）
   - 背景色变化：`background: var(--color-accent-soft)`
   - 边框颜色加深：`border-left-color: var(--category-color)`
   - 可选：添加 `box-shadow` 或改变文字颜色

**预期效果**:
- 纯文字分类列表
- 左侧彩色边框作为视觉区分
- 悬停时背景色变化，无布局抖动

---

### 2. 文章时间倒序排列

#### 当前状态验证

**观察结果**: 检查 `hugo-site/public/index.html` 发现首页当前**已经是最新文章在前**（倒序排列）。

**结论**: Hugo 的 `.Site.RegularPages` 默认按日期倒序排列，当前实现已满足需求。

#### 修改方案

**选项 A（推荐）**: 无需修改，当前行为已符合预期

**选项 B**: 为代码可读性，显式添加排序逻辑

**文件**: `hugo-site/themes/custom-theme/layouts/index.html`

**当前实现**:
```html
{{ $paginator := .Paginate (where .Site.RegularPages "Type" "posts") }}
```

**显式排序版本**:
```html
{{ $posts := where .Site.RegularPages "Type" "posts" }}
{{ $sortedPosts := $posts.ByDate.Reverse }}
{{ $paginator := .Paginate $sortedPosts }}
```

#### 技术说明

- Hugo 的 `.Site.RegularPages` 默认按 `date` 字段倒序（新→旧）
- `.ByDate` 方法返回升序（旧→新）
- `.Reverse` 反转为倒序（新→旧）
- 如果当前已是倒序，添加显式排序仅为提高代码可读性

**推荐**: 保持当前实现不变，除非需要显式固定排序逻辑以防未来 Hugo 版本行为变化

---

### 3. 标签页面和分类页面卡片美化

#### 修改范围

**文件**: `hugo-site/themes/custom-theme/layouts/_default/taxonomy.html`

**重要说明**: 此模板同时服务于**标签页**（`/tags/xxx/`）和**分类页**（`/categories/xxx/`），修改将影响两者。

**当前实现**:
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

**修改后**:
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

#### 技术说明

- **完全复用首页卡片组件**: 使用 `.article-card` 和 `.articles-grid`
- **卡片样式可直接复用**: `baseof.html` 已全局加载 `article-card.css`
- **页面布局需确认**: `taxonomy.css` 中 `.taxonomy-page` 有 `max-width: 900px` 限制，可能影响卡片网格宽度。建议测试后决定是否调整或移除此限制
- **响应式布局**: 自动适配 1-3 列网格（移动端 1 列，平板 2 列，桌面 3 列）
- **无障碍支持**: 保留 ARIA 属性

**卡片组件包含**:
- 封面图（200px 高，懒加载）
- 分类标签
- 阅读时间
- 文章标题
- 摘要（150 字截断）
- 发布日期 + 作者

**预期效果**:
- 标签页面和分类页面与首页视觉一致
- 卡片式网格布局
- 悬停效果（上移 + 阴影）
- 无封面文章将显示占位封面区域（需确认是否符合预期）

---

## 实现清单

### 文件修改列表

1. `hugo-site/themes/custom-theme/layouts/partials/sidebar.html` - 移除图标 span
2. `hugo-site/themes/custom-theme/static/css/sidebar.css` - 调整分类样式（避免布局抖动）
3. `hugo-site/themes/custom-theme/layouts/index.html` - 可选：显式添加排序逻辑（当前已是倒序）
4. `hugo-site/themes/custom-theme/layouts/_default/taxonomy.html` - 替换为卡片布局
5. `hugo-site/themes/custom-theme/static/css/taxonomy.css` - 可选：调整 `.taxonomy-page` 宽度限制

### 测试验证

**构建验证**:
```bash
cd hugo-site
hugo --gc --minify
```

**功能测试**:

1. **侧边栏分类**:
   - 检查图标是否完全移除
   - 验证左侧彩色边框显示
   - 测试悬停效果（无布局抖动）

2. **文章排序**:
   - 访问首页，确认最新文章在顶部
   - 检查分页是否正常工作

3. **标签页面**:
   - 访问任意标签页（如 `/tags/机器学习/`）
   - 验证卡片网格布局
   - 测试响应式断点（768px, 1200px, 1400px）
   - 检查封面图懒加载
   - 验证无封面文章的占位封面显示

4. **分类页面**（新增）:
   - 访问任意分类页（如 `/categories/code/`）
   - 验证卡片布局与标签页一致
   - 测试响应式布局

5. **跨浏览器测试**:
   - Chrome/Edge
   - Firefox
   - Safari（如有条件）

6. **亮色/暗色模式**:
   - 切换主题，验证样式正常

7. **无障碍测试**:
   - 键盘导航（Tab 键）
   - 屏幕阅读器 ARIA 属性

---

## 风险与注意事项

### 潜在问题

1. **taxonomy 模板影响范围**: `layouts/_default/taxonomy.html` 同时服务于标签页和分类页
   - **缓解措施**: 测试时需同时验证标签页和分类页

2. **页面布局约束**: `taxonomy.css` 中 `.taxonomy-page` 的 `max-width: 900px` 可能限制卡片网格宽度
   - **缓解措施**: 实施后测试，必要时调整或移除宽度限制

3. **无封面文章**: 卡片会保留 `.article-card-cover` 占位区域（200px 高）
   - **缓解措施**: 需确认此行为是否符合预期，或考虑为无封面文章添加默认占位图

4. **CSS 冲突**: `.post-preview` 样式可能在其他页面使用
   - **缓解措施**: 检查 `list.html` 和其他模板是否依赖该样式

5. **图标配置残留**: `hugo.toml` 中仍配置了分类图标
   - **处理方式**: 保留配置不影响功能，未来可选择性清理

### 回滚方案

如果修改后出现问题，可通过 Git 回滚：
```bash
git checkout HEAD -- hugo-site/themes/custom-theme/layouts/partials/sidebar.html
git checkout HEAD -- hugo-site/themes/custom-theme/static/css/sidebar.css
git checkout HEAD -- hugo-site/themes/custom-theme/layouts/index.html
git checkout HEAD -- hugo-site/themes/custom-theme/layouts/_default/taxonomy.html
```

---

## 设计原则符合性

### 编辑式极简主义

- ✅ 移除不必要的视觉元素（emoji 图标）
- ✅ 强化排版和留白
- ✅ 保持视觉一致性（标签页面复用首页样式）

### 响应式设计

- ✅ 卡片网格自适应 1-3 列
- ✅ 移动端优先布局
- ✅ 触摸友好的交互区域

### 无障碍访问

- ✅ 保留 ARIA 属性
- ✅ 语义化 HTML 标签
- ✅ 键盘导航支持

---

## 后续优化建议

1. **分类页面**: 考虑同样应用卡片布局（当前未在需求中）
2. **归档页面**: 评估是否需要类似改进
3. **性能监控**: 使用 Lighthouse 测试页面性能
4. **A/B 测试**: 收集用户反馈，评估改进效果

---

## 审批

- [ ] 设计方案审查通过
- [ ] 用户确认需求符合预期
- [ ] 准备进入实施阶段
