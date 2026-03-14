# AGENTS.md - 代码规范与开发指南

本文档为 AI 编码助手提供项目规范和开发指南。

## 项目概述

这是一个使用 Hugo 构建的个人技术博客，主要内容包括机器学习、机器视觉、工程技术等领域的文章。

- **站点**: https://wenboli-cn-de.github.io
- **Hugo 版本**: v0.157.0-extended
- **环境**: WSL Ubuntu 24.04
- **语言**: 中文（zh-CN）

## 构建与测试命令

### 开发服务器
```bash
cd hugo-site
hugo server -D
# 访问 http://localhost:1313
```

### 构建生产版本
```bash
cd hugo-site
hugo --gc --minify
# 输出到 hugo-site/public/
```

### 检查 Hugo 版本
```bash
hugo version
```

### 创建新文章
```bash
cd hugo-site
hugo new posts/YYYY-MM-DD-title.md
```

### 清理构建缓存
```bash
cd hugo-site
hugo --cleanDestinationDir
```

## 项目结构

```
hugo-site/
├── hugo.toml              # 站点配置
├── content/
│   ├── posts/            # 博客文章（Markdown）
│   ├── about.md          # 关于页面
│   └── archive.md        # 归档页面
├── themes/custom-theme/
│   ├── layouts/          # HTML 模板
│   │   ├── index.html           # 首页
│   │   ├── _default/single.html # 文章页
│   │   └── partials/            # 组件
│   └── static/
│       ├── css/          # 样式文件
│       └── js/           # JavaScript
├── static/               # 静态资源（图片等）
└── public/              # 构建输出（不提交到 Git）
```

## 代码风格指南

### HTML/模板（Hugo Templates）

- 使用 Hugo 模板语法，不是 Liquid
- 模板文件使用 2 空格缩进
- 使用语义化 HTML5 标签
- 添加适当的 ARIA 属性以支持无障碍访问

```html
{{ define "main" }}
<article class="post">
  <h1>{{ .Title }}</h1>
  <time datetime="{{ .Date.Format "2006-01-02" }}">
    {{ .Date.Format "2006年1月2日" }}
  </time>
</article>
{{ end }}
```

### CSS

- 使用 CSS 变量（定义在 `variables.css`）
- 2 空格缩进
- 类名使用 kebab-case（如 `post-title`）
- 支持亮色/暗色模式（通过 `[data-theme="dark"]`）
- 移动优先的响应式设计

```css
.post-title {
  color: var(--text-primary);
  font-size: 2rem;
  margin-bottom: var(--spacing-md);
}

@media (max-width: 768px) {
  .post-title {
    font-size: 1.5rem;
  }
}
```

### JavaScript

- 使用现代 ES6+ 语法
- 2 空格缩进
- 使用 `const` 和 `let`，避免 `var`
- 添加事件监听器前检查元素是否存在
- 使用 `DOMContentLoaded` 确保 DOM 加载完成

```javascript
document.addEventListener('DOMContentLoaded', function() {
  const sidebar = document.querySelector('.sidebar');
  
  if (sidebar) {
    sidebar.addEventListener('click', function(event) {
      // 处理点击事件
    });
  }
});
```

### Markdown（文章内容）

- 文件命名: `YYYY-MM-DD-title.md`
- 使用 YAML Front Matter
- 支持 LaTeX 数学公式（`$...$` 行内，`$$...$$` 块级）
- 代码块使用三个反引号并指定语言

**Front Matter 模板**:
```yaml
---
title: "文章标题"
date: 2026-03-13
description: "文章简介"
categories: ["code"]  # life, thought, engineering, code, ai
tags: ["标签1", "标签2"]
cover: "/images/cover.jpg"  # 可选
---
```

## 命名约定

- **文件名**: kebab-case（如 `sidebar.js`, `post-item.css`）
- **CSS 类名**: kebab-case（如 `.post-title`, `.hero-section`）
- **JavaScript 变量**: camelCase（如 `toggleButton`, `sidebarActive`）
- **Hugo 配置**: camelCase（如 `enableToc`, `avatarTitle`）

## 分类系统

博客使用五个预定义分类（在 `hugo.toml` 中配置）：

- `life` - 人间便签 📝（生活记录）
- `thought` - 思维漫游 💭（思想探索）
- `engineering` - 工程随笔 ⚙️（工程实践）
- `code` - 代码诗篇 💻（编程技术）
- `ai` - 智识前沿 🤖（人工智能）

## 设计原则

### 编辑式极简主义（Editorial Minimalism）

- **排版**: 精致的字体系统（Crimson Pro, Inter, JetBrains Mono）
- **布局**: 卡片式设计，响应式网格
- **动画**: 微妙的悬停效果和过渡
- **配色**: 支持亮色/暗色模式切换

### 响应式断点

- 桌面: ≥ 1400px（侧边栏固定）
- 平板: 768px - 1399px（侧边栏可切换）
- 手机: < 768px（单列布局）

## Git 工作流

### 提交信息规范

使用简洁的中文描述：

```
添加新文章：机器学习基础
修复侧边栏在移动端的显示问题
优化首页加载性能
更新关于页面内容
```

### 部署流程

1. 推送到 `main` 分支
2. GitHub Actions 自动触发构建
3. 部署到 GitHub Pages

**不要手动修改 `public/` 目录**，它由 Hugo 自动生成。

## 常见任务

### 添加新文章

1. 创建文件: `hugo new posts/2026-03-14-title.md`
2. 编辑 Front Matter 和内容
3. 本地预览: `hugo server -D`
4. 提交并推送

### 修改主题样式

1. 编辑 `themes/custom-theme/static/css/` 中的 CSS 文件
2. 使用 CSS 变量保持一致性
3. 测试亮色/暗色模式
4. 测试响应式布局

### 添加新页面

1. 创建 Markdown: `hugo new about.md`
2. 在 `hugo.toml` 中添加菜单项
3. 创建对应的布局模板（如需要）

## 性能优化

- 图片使用 `loading="lazy"` 懒加载
- CSS 动画使用 GPU 加速（`transform`, `opacity`）
- 避免大型 JavaScript 库
- 使用 Hugo 的 `--minify` 压缩输出

## 注意事项

- **不要使用 Jekyll 语法**（如 `{% include %}`），使用 Hugo 模板语法
- **数学公式**: 使用 `$...$` 或 `$$...$$`，已配置 Goldmark passthrough
- **中文排版**: 注意中英文之间的空格，使用全角标点
- **图片路径**: 相对于 `static/` 目录，引用时使用 `/images/...`
- **构建时间**: Hugo 构建非常快（约 36ms），无需担心性能

## 故障排查

### Hugo 命令找不到
```bash
export PATH="$HOME/.local/bin:$PATH"
```

### 样式未生效
- 清除浏览器缓存
- 检查 CSS 文件路径
- 运行 `hugo --cleanDestinationDir`

### 数学公式不显示
- 确认 `hugo.toml` 中 `markup.goldmark.extensions.passthrough` 已启用
- 使用正确的分隔符（`$...$` 或 `$$...$$`）

## 参考文档

- Hugo 官方文档: https://gohugo.io/documentation/
- 项目设计说明: `hugo-site/DESIGN_NOTES.md`
- 迁移文档: `hugo-site/MIGRATION.md`
