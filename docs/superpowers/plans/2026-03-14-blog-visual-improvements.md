# 博客视觉优化实施计划

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 优化博客四个视觉问题：sidebar name 样式、contact item emoji、分页导航按钮、文章卡片底色

**Architecture:** 纯 CSS 修改，无需 JavaScript。修改四个独立的样式文件和一个 HTML 模板文件。每个改进都是独立的，可以单独测试和回滚。

**Tech Stack:** Hugo 模板、CSS3、CSS 变量

**Spec Document:** `docs/superpowers/specs/2026-03-14-blog-visual-improvements-design.md`

---

## Chunk 1: 准备工作和备份

### Task 1: 创建备份和准备环境

**Files:**
- Backup: `hugo-site/themes/custom-theme/static/css/sidebar.css`
- Backup: `hugo-site/themes/custom-theme/static/css/home.css`
- Backup: `hugo-site/themes/custom-theme/static/css/article-card.css`
- Backup: `hugo-site/themes/custom-theme/layouts/partials/sidebar.html`

- [ ] **Step 1: 创建备份分支**

```bash
cd /home/wenbo/WenboLi-CN-DE.github.io
git checkout -b feature/visual-improvements
```

Expected: 切换到新分支 `feature/visual-improvements`

- [ ] **Step 2: 备份原始文件**

```bash
cd hugo-site/themes/custom-theme
cp static/css/sidebar.css static/css/sidebar.css.backup
cp static/css/home.css static/css/home.css.backup
cp static/css/article-card.css static/css/article-card.css.backup
cp layouts/partials/sidebar.html layouts/partials/sidebar.html.backup
```

Expected: 创建 4 个备份文件

- [ ] **Step 3: 验证当前 Hugo 服务器可以运行**

```bash
cd /home/wenbo/WenboLi-CN-DE.github.io/hugo-site
hugo server -D
```

Expected: 服务器启动成功，访问 http://localhost:1313 显示正常

按 Ctrl+C 停止服务器

- [ ] **Step 4: 提交备份**

```bash
cd /home/wenbo/WenboLi-CN-DE.github.io
git add hugo-site/themes/custom-theme/static/css/*.backup
git add hugo-site/themes/custom-theme/layouts/partials/*.backup
git commit -m "backup: 创建视觉优化前的备份文件"
```

Expected: 提交成功

---

## Chunk 2: Sidebar Name 优化

### Task 2: 修改 Sidebar Name 样式

**Files:**
- Modify: `hugo-site/themes/custom-theme/static/css/sidebar.css:58-65`

**参考设计文档**: Section 1 - Sidebar Name 优化

- [ ] **Step 1: 打开 sidebar.css 文件**

```bash
cd /home/wenbo/WenboLi-CN-DE.github.io/hugo-site/themes/custom-theme/static/css
```

- [ ] **Step 2: 替换 .sidebar-name 样式（第 58-65 行）**

找到：
```css
.sidebar-name {
    font-family: var(--font-display);
    font-size: 1.75rem;
    font-weight: 700;
    color: var(--color-text);
    margin-bottom: var(--spacing-xs);
    line-height: 1.2;
}
```

替换为：
```css
.sidebar-name {
    font-family: var(--font-display);
    font-size: 1.5rem;
    font-weight: 600;
    color: var(--color-text);
    margin-bottom: var(--spacing-sm);
    line-height: 1.3;
    letter-spacing: -0.02em;
    position: relative;
    padding-bottom: 0.5rem;
}

.sidebar-name::after {
    content: '';
    position: absolute;
    bottom: 0;
    left: 50%;
    transform: translateX(-50%);
    width: 60px;
    height: 2px;
    background: linear-gradient(90deg,
        transparent,
        var(--color-accent) 20%,
        var(--color-accent) 80%,
        transparent
    );
}

[data-theme="dark"] .sidebar-name {
    color: var(--color-text);
}
```

- [ ] **Step 3: 保存文件并测试**

```bash
cd /home/wenbo/WenboLi-CN-DE.github.io/hugo-site
hugo server -D
```

Expected: 
- 服务器启动成功
- 访问 http://localhost:1313
- Sidebar 中的名字"高傲的电工李"下方出现橙红色渐变装饰线
- 字体大小略小，字重适中
- 无蓝色底色

按 Ctrl+C 停止服务器

- [ ] **Step 4: 测试暗色模式**

在浏览器中切换到暗色模式，验证名字颜色正确显示

- [ ] **Step 5: 测试响应式（移动端）**

在浏览器开发者工具中切换到移动设备视图（375px 宽度），验证样式正常

- [ ] **Step 6: 提交更改**

```bash
cd /home/wenbo/WenboLi-CN-DE.github.io
git add hugo-site/themes/custom-theme/static/css/sidebar.css
git commit -m "feat: 优化 sidebar name 样式，添加优雅装饰线"
```

Expected: 提交成功

---

## Chunk 3: Contact Item 优雅化

### Task 3: 修改 Contact Item HTML

**Files:**
- Modify: `hugo-site/themes/custom-theme/layouts/partials/sidebar.html:10-13`

**参考设计文档**: Section 2 - Contact Item 优雅化

- [ ] **Step 1: 打开 sidebar.html 文件**

```bash
cd /home/wenbo/WenboLi-CN-DE.github.io/hugo-site/themes/custom-theme/layouts/partials
```

- [ ] **Step 2: 修改 contact-item HTML（第 10-13 行）**

找到：
```html
<div class="sidebar-contact">
    <span class="contact-item">📍 中国</span>
    <span class="contact-item">📧 {{ .Site.Params.social.email }}</span>
</div>
```

替换为：
```html
<div class="sidebar-contact">
    <span class="contact-item contact-location">中国</span>
    <span class="contact-item contact-email">{{ .Site.Params.social.email }}</span>
</div>
```

- [ ] **Step 3: 保存文件**

Expected: HTML 文件已保存，emoji 已移除，添加了 CSS 类名

### Task 4: 修改 Contact Item CSS

**Files:**
- Modify: `hugo-site/themes/custom-theme/static/css/sidebar.css:82-88`

- [ ] **Step 1: 打开 sidebar.css 文件**

```bash
cd /home/wenbo/WenboLi-CN-DE.github.io/hugo-site/themes/custom-theme/static/css
```

- [ ] **Step 2: 替换 .contact-item 样式（第 82-88 行）**

找到：
```css
.contact-item {
    font-size: 0.875rem;
    color: var(--color-text-secondary);
    display: flex;
    align-items: center;
    gap: 0.25rem;
}
```

替换为：
```css
.contact-item {
    font-size: 0.875rem;
    color: var(--color-text-secondary);
    display: flex;
    align-items: center;
    gap: 0.5rem;
    position: relative;
    padding-left: 1.25rem;
}

.contact-item::before {
    content: '';
    position: absolute;
    left: 0;
    width: 4px;
    height: 4px;
    background: var(--color-accent);
    border-radius: 50%;
}

.contact-location::before {
    background: var(--color-accent);
}

.contact-email::before {
    background: var(--color-text-secondary);
}
```

- [ ] **Step 3: 保存文件并测试**

```bash
cd /home/wenbo/WenboLi-CN-DE.github.io/hugo-site
hugo server -D
```

Expected:
- 服务器启动成功
- 访问 http://localhost:1313
- Sidebar 联系信息中 emoji 已消失
- "中国"前有橙红色小圆点
- 邮箱前有灰色小圆点
- 视觉更简洁优雅

按 Ctrl+C 停止服务器

- [ ] **Step 4: 测试暗色模式**

切换到暗色模式，验证圆点颜色正确显示

- [ ] **Step 5: 测试响应式**

在移动设备视图中验证样式正常

- [ ] **Step 6: 提交更改**

```bash
cd /home/wenbo/WenboLi-CN-DE.github.io
git add hugo-site/themes/custom-theme/layouts/partials/sidebar.html
git add hugo-site/themes/custom-theme/static/css/sidebar.css
git commit -m "feat: 用 CSS 圆点替代 contact item emoji"
```

Expected: 提交成功

---

## Chunk 4: 分页导航重设计

### Task 5: 修改分页导航样式

**Files:**
- Modify: `hugo-site/themes/custom-theme/static/css/home.css:400-470`

**参考设计文档**: Section 3 - 分页导航重设计

- [ ] **Step 1: 打开 home.css 文件**

```bash
cd /home/wenbo/WenboLi-CN-DE.github.io/hugo-site/themes/custom-theme/static/css
```

- [ ] **Step 2: 找到分页导航样式区域（约第 400-470 行）**

查找包含 `.pagination` 和 `.pagination-link` 的样式块

- [ ] **Step 3: 替换整个分页导航样式块**

删除原有的 `.pagination`、`.pagination-link`、`.pagination-info` 相关样式

添加新样式：
```css
/* ============================================
   Pagination Navigation - Editorial Style
   ============================================ */

.pagination {
    display: flex;
    justify-content: center;
    align-items: center;
    gap: var(--spacing-lg);
    margin-top: var(--spacing-xl);
    padding: var(--spacing-lg) 0;
    font-family: var(--font-body);
}

.pagination-link {
    display: inline-flex;
    align-items: center;
    padding: 0.75rem 1.5rem;
    color: var(--color-text);
    text-decoration: none;
    font-size: 0.9rem;
    font-weight: 500;
    letter-spacing: 0.05em;
    border: 1px solid var(--color-border);
    border-radius: 0;
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}

.pagination-link::before {
    content: '';
    position: absolute;
    bottom: 0;
    left: 0;
    width: 0;
    height: 1px;
    background: var(--color-accent);
    transition: width 0.3s ease;
}

.pagination-link:hover {
    color: var(--color-accent);
    border-color: var(--color-accent);
}

.pagination-link:hover::before {
    width: 100%;
}

.pagination-info {
    font-size: 0.875rem;
    color: var(--color-text-secondary);
    font-weight: 400;
    letter-spacing: 0.02em;
}

[data-theme="dark"] .pagination-link {
    border-color: var(--color-border);
    color: var(--color-text);
}

[data-theme="dark"] .pagination-link:hover {
    color: var(--color-accent);
    border-color: var(--color-accent);
}

@media (max-width: 768px) {
    .pagination {
        gap: var(--spacing-sm);
        flex-wrap: wrap;
    }

    .pagination-link {
        padding: 0.625rem 1.25rem;
        font-size: 0.85rem;
    }
}
```

- [ ] **Step 4: 保存文件并测试**

```bash
cd /home/wenbo/WenboLi-CN-DE.github.io/hugo-site
hugo server -D
```

Expected:
- 服务器启动成功
- 访问 http://localhost:1313
- 如果有多页文章，滚动到底部查看分页导航
- 按钮使用细线边框，无圆角
- 悬停时底部出现橙红色下划线动画
- 文字和边框颜色变为橙红色

按 Ctrl+C 停止服务器

- [ ] **Step 5: 测试分页功能**

如果文章数量不足以触发分页，临时修改 `hugo.toml` 中的 `paginate` 值为 2：

```bash
cd /home/wenbo/WenboLi-CN-DE.github.io/hugo-site
# 编辑 hugo.toml，找到 paginate 行，改为 paginate = 2
hugo server -D
```

验证分页导航显示和功能正常，然后恢复 `paginate` 值

- [ ] **Step 6: 测试暗色模式**

切换到暗色模式，验证按钮样式正确

- [ ] **Step 7: 测试响应式**

在移动设备视图（375px）中验证按钮换行和间距正常

- [ ] **Step 8: 提交更改**

```bash
cd /home/wenbo/WenboLi-CN-DE.github.io
git add hugo-site/themes/custom-theme/static/css/home.css
git commit -m "feat: 重设计分页导航，使用细线边框和悬停动画"
```

Expected: 提交成功

---

## Chunk 5: 文章卡片底色修正

### Task 6: 修改文章卡片背景色

**Files:**
- Modify: `hugo-site/themes/custom-theme/static/css/article-card.css:13-31,176-204`

**参考设计文档**: Section 4 - 文章卡片底色修正

- [ ] **Step 1: 打开 article-card.css 文件**

```bash
cd /home/wenbo/WenboLi-CN-DE.github.io/hugo-site/themes/custom-theme/static/css
```

- [ ] **Step 2: 修改 .article-card 基础样式（约第 13-31 行）**

找到：
```css
.article-card {
  background: var(--bg-secondary);
  border-radius: var(--radius-lg);
  overflow: hidden;
  transition: transform 0.3s ease, box-shadow 0.3s ease;
  display: flex;
  flex-direction: column;
  height: 100%;
  border: 1px solid var(--border-color);
}

.article-card:hover {
  transform: translateY(-4px);
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.12);
}

[data-theme="dark"] .article-card:hover {
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
}
```

替换为：
```css
.article-card {
  background: #ffffff;
  border-radius: var(--radius-lg);
  overflow: hidden;
  transition: transform 0.3s ease, box-shadow 0.3s ease;
  display: flex;
  flex-direction: column;
  height: 100%;
  border: 1px solid var(--border-color);
  box-shadow: 0 1px 3px rgba(26, 22, 20, 0.04);
}

.article-card:hover {
  transform: translateY(-4px);
  box-shadow: 0 8px 24px rgba(26, 22, 20, 0.08);
}
```

- [ ] **Step 3: 修改暗色模式样式（约第 176-204 行）**

找到：
```css
[data-theme="dark"] .article-card {
  background: #1e1e1e;
  border-color: #333;
}

[data-theme="dark"] .article-card:hover {
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
}

[data-theme="dark"] .article-card-cover {
  background: linear-gradient(135deg, #4a5568 0%, #2d3748 100%);
}

[data-theme="dark"] .article-card-title {
  color: #e0e0e0;
}

[data-theme="dark"] .article-card:hover .article-card-title {
  color: var(--accent-color);
}

[data-theme="dark"] .article-card-excerpt {
  color: #b0b0b0;
}

[data-theme="dark"] .article-card-footer {
  border-top-color: #333;
  color: #888;
}
```

替换为：
```css
[data-theme="dark"] .article-card {
  background: #252220;
  border-color: #3d3935;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.2);
}

[data-theme="dark"] .article-card:hover {
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
}

[data-theme="dark"] .article-card-cover {
  background: linear-gradient(135deg, #4a5568 0%, #2d3748 100%);
}

[data-theme="dark"] .article-card-title {
  color: #f5f3f0;
}

[data-theme="dark"] .article-card:hover .article-card-title {
  color: var(--accent-color);
}

[data-theme="dark"] .article-card-excerpt {
  color: #a8a29e;
}

[data-theme="dark"] .article-card-footer {
  border-top-color: #3d3935;
  color: #6b6560;
}
```

- [ ] **Step 4: 保存文件并测试**

```bash
cd /home/wenbo/WenboLi-CN-DE.github.io/hugo-site
hugo server -D
```

Expected:
- 服务器启动成功
- 访问 http://localhost:1313
- 文章卡片背景为纯白色（#ffffff）
- 卡片有微妙的阴影效果
- 悬停时阴影加深
- 整体视觉更温暖柔和

按 Ctrl+C 停止服务器

- [ ] **Step 5: 测试暗色模式**

切换到暗色模式，验证：
- 卡片背景为深灰棕色（#252220）
- 边框颜色为中灰棕色（#3d3935）
- 文字颜色协调
- 悬停效果正常

- [ ] **Step 6: 测试响应式**

在不同屏幕尺寸下验证卡片布局和样式：
- 桌面（1400px+）：3 列网格
- 平板（769px-1024px）：2 列网格
- 手机（<768px）：1 列网格

- [ ] **Step 7: 提交更改**

```bash
cd /home/wenbo/WenboLi-CN-DE.github.io
git add hugo-site/themes/custom-theme/static/css/article-card.css
git commit -m "feat: 优化文章卡片背景色，使用温暖柔和的配色"
```

Expected: 提交成功

---

## Chunk 6: 最终测试和清理

### Task 7: 综合测试

**测试清单**:

- [ ] **Step 1: 完整的亮色模式测试**

```bash
cd /home/wenbo/WenboLi-CN-DE.github.io/hugo-site
hugo server -D
```

访问 http://localhost:1313，验证：
- ✅ Sidebar name 有优雅装饰线，无蓝色底色
- ✅ Contact item 使用圆点，无 emoji
- ✅ 分页导航使用细线边框，悬停有动画
- ✅ 文章卡片背景为纯白色，阴影柔和

- [ ] **Step 2: 完整的暗色模式测试**

切换到暗色模式，验证所有四个改进在暗色模式下正常显示

- [ ] **Step 3: 响应式测试**

测试三个断点：
- 桌面（1400px）
- 平板（768px）
- 手机（375px）

验证所有改进在不同屏幕尺寸下正常显示

- [ ] **Step 4: 浏览器兼容性测试**

在以下浏览器中测试（如果可用）：
- Chrome/Edge
- Firefox
- Safari

- [ ] **Step 5: 性能检查**

```bash
cd /home/wenbo/WenboLi-CN-DE.github.io/hugo-site
hugo --gc --minify
```

Expected: 构建成功，无错误或警告

### Task 8: 清理和文档

- [ ] **Step 1: 删除备份文件**

```bash
cd /home/wenbo/WenboLi-CN-DE.github.io/hugo-site/themes/custom-theme
rm static/css/sidebar.css.backup
rm static/css/home.css.backup
rm static/css/article-card.css.backup
rm layouts/partials/sidebar.html.backup
```

Expected: 备份文件已删除

- [ ] **Step 2: 更新 AGENTS.md（如果需要）**

如果有新的设计模式或重要变更，更新项目知识库

- [ ] **Step 3: 最终提交**

```bash
cd /home/wenbo/WenboLi-CN-DE.github.io
git add -A
git commit -m "chore: 清理备份文件，完成视觉优化"
```

Expected: 提交成功

- [ ] **Step 4: 查看提交历史**

```bash
git log --oneline -10
```

Expected: 看到所有相关提交记录

### Task 9: 合并到主分支（可选）

- [ ] **Step 1: 切换到主分支**

```bash
cd /home/wenbo/WenboLi-CN-DE.github.io
git checkout main
```

- [ ] **Step 2: 合并功能分支**

```bash
git merge feature/visual-improvements
```

Expected: 合并成功，无冲突

- [ ] **Step 3: 推送到远程仓库**

```bash
git push origin main
```

Expected: 推送成功

- [ ] **Step 4: 删除功能分支（可选）**

```bash
git branch -d feature/visual-improvements
```

Expected: 分支已删除

---

## 回滚方案

如果需要回滚任何更改：

### 回滚单个文件

```bash
cd /home/wenbo/WenboLi-CN-DE.github.io
git checkout HEAD~1 -- hugo-site/themes/custom-theme/static/css/sidebar.css
```

### 回滚整个功能分支

```bash
git checkout main
git reset --hard HEAD~N  # N 是要回滚的提交数量
```

### 使用备份文件恢复

```bash
cd /home/wenbo/WenboLi-CN-DE.github.io/hugo-site/themes/custom-theme
cp static/css/sidebar.css.backup static/css/sidebar.css
```

---

## 验收标准

所有改进完成后，应满足以下标准：

1. ✅ **Sidebar Name**: 无蓝色底色，有优雅的橙红色渐变装饰线
2. ✅ **Contact Item**: 无 emoji，使用 CSS 圆点装饰
3. ✅ **分页导航**: 细线边框，无圆角，悬停时有底部下划线动画
4. ✅ **文章卡片**: 亮色模式使用纯白背景，暗色模式使用深灰棕背景
5. ✅ **响应式**: 所有改进在移动端、平板、桌面端都正常显示
6. ✅ **暗色模式**: 所有改进在暗色模式下都正常显示
7. ✅ **性能**: Hugo 构建成功，无错误或警告
8. ✅ **代码质量**: 所有更改已提交，提交信息清晰

---

## 注意事项

1. **CSS 变量优先级**: 注意 `home.css` 和 `variables.css` 中定义了不同的 CSS 变量。优先使用 `home.css` 中的变量（如 `--color-accent`），因为它们更符合编辑式极简主义风格。

2. **浏览器缓存**: 测试时如果样式未生效，清除浏览器缓存或使用无痕模式。

3. **Hugo 版本**: 确保使用 Hugo v0.157.0-extended 或更高版本。

4. **Git 提交**: 每个任务完成后立即提交，便于回滚和追踪。

5. **测试顺序**: 建议按照任务顺序逐个测试，确保每个改进独立工作正常。
