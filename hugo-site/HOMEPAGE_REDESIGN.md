# 首页重新设计文档

## 概述

本次首页重新设计采用"编辑式极简主义"（Editorial Minimalism）风格，打造一个现代、优雅、易读的博客首页。设计重点包括：

- 固定顶部 Hero Banner，展示博客核心信息
- 卡片式文章列表，支持封面图、分类、阅读时间等元素
- 完整的亮色/暗色模式支持
- 响应式设计，适配桌面、平板、手机
- 性能优化，包括懒加载、关键 CSS 内联
- 可访问性改进，符合 WCAG 标准

## 设计原则

### 1. 编辑式极简主义

- **排版优先**：使用精致的字体系统（Crimson Pro, Inter, JetBrains Mono）
- **留白艺术**：充足的间距，让内容呼吸
- **视觉层次**：清晰的信息架构，引导用户视线
- **微妙动画**：hover 效果和过渡，提升交互体验

### 2. 响应式设计

- **桌面**（≥ 1400px）：侧边栏固定，双栏布局
- **平板**（768px - 1399px）：侧边栏可切换，单栏布局
- **手机**（< 768px）：完全单栏，优化触摸交互

### 3. 性能优先

- 关键 CSS 内联，减少渲染阻塞
- 图片懒加载，节省带宽
- GPU 加速动画，流畅体验
- 最小化 JavaScript，快速加载

## 技术实现

### 文件结构

```
hugo-site/
├── themes/custom-theme/
│   ├── layouts/
│   │   ├── index.html              # 首页模板（重新设计）
│   │   └── partials/
│   │       ├── hero-banner.html    # Hero Banner 组件
│   │       ├── post-card.html      # 文章卡片组件
│   │       └── critical-css.html   # 关键 CSS
│   └── static/
│       ├── css/
│       │   ├── hero-banner.css     # Hero Banner 样式
│       │   ├── post-card.css       # 文章卡片样式
│       │   └── homepage.css        # 首页特定样式
│       └── js/
│           └── lazy-load.js        # 懒加载脚本
└── content/
    └── posts/
        └── 2026-03-14-example-post.md  # 示例文章
```

### 核心组件

#### 1. Hero Banner (`hero-banner.html`)

固定在页面顶部的横幅，包含：
- 博客标题和副标题
- 头像（可选）
- 社交媒体链接
- 主题切换按钮

特点：
- 使用 `position: sticky` 实现固定效果
- 背景渐变，支持暗色模式
- 响应式布局，移动端优化

#### 2. 文章卡片 (`post-card.html`)

展示文章信息的卡片组件，包含：
- 封面图（可选，懒加载）
- 分类标签
- 文章标题
- 文章摘要
- 发布日期
- 阅读时间（可选）
- 作者信息

特点：
- 卡片式设计，带阴影和圆角
- hover 动画（上浮 + 阴影增强）
- 响应式网格布局
- 支持 featured 文章高亮

#### 3. 关键 CSS (`critical-css.html`)

内联在 `<head>` 中的关键样式，包括：
- CSS 变量定义
- 布局基础样式
- Hero Banner 样式
- 首屏内容样式

目的：
- 减少首次渲染时间
- 避免 FOUC（无样式内容闪烁）
- 提升 Lighthouse 性能评分

### 样式系统

#### CSS 变量（定义在 `variables.css`）

```css
:root {
  /* 颜色 */
  --text-primary: #2c3e50;
  --text-secondary: #7f8c8d;
  --bg-primary: #ffffff;
  --bg-secondary: #f8f9fa;
  --accent: #3498db;
  
  /* 间距 */
  --spacing-xs: 0.5rem;
  --spacing-sm: 1rem;
  --spacing-md: 1.5rem;
  --spacing-lg: 2rem;
  --spacing-xl: 3rem;
  
  /* 字体 */
  --font-serif: 'Crimson Pro', serif;
  --font-sans: 'Inter', sans-serif;
  --font-mono: 'JetBrains Mono', monospace;
}

[data-theme="dark"] {
  --text-primary: #ecf0f1;
  --text-secondary: #bdc3c7;
  --bg-primary: #1a1a1a;
  --bg-secondary: #2c2c2c;
  --accent: #5dade2;
}
```

#### 响应式断点

```css
/* 手机 */
@media (max-width: 767px) { ... }

/* 平板 */
@media (min-width: 768px) and (max-width: 1399px) { ... }

/* 桌面 */
@media (min-width: 1400px) { ... }
```

### JavaScript 功能

#### 懒加载 (`lazy-load.js`)

使用 Intersection Observer API 实现图片懒加载：

```javascript
const imageObserver = new IntersectionObserver((entries, observer) => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      const img = entry.target;
      img.src = img.dataset.src;
      img.classList.remove('lazy');
      observer.unobserve(img);
    }
  });
});

document.querySelectorAll('img.lazy').forEach(img => {
  imageObserver.observe(img);
});
```

## 使用指南

### 创建文章

在文章的 Front Matter 中添加以下字段：

```yaml
---
title: "文章标题"
date: 2026-03-14
description: "文章描述"
categories: ["code"]
tags: ["标签1", "标签2"]
cover: "/images/cover.jpg"  # 可选，封面图路径
readingTime: 5              # 可选，阅读时间（分钟）
featured: true              # 可选，是否为精选文章
---
```

### 字段说明

- `cover`：封面图路径，相对于 `static/` 目录
- `readingTime`：预估阅读时间（分钟），显示在卡片上
- `featured`：精选文章，卡片会有特殊样式（边框高亮）

### 添加封面图

1. 将图片放在 `hugo-site/static/images/` 目录
2. 在 Front Matter 中引用：`cover: "/images/your-image.jpg"`
3. 推荐尺寸：1200x630px（16:9 比例）
4. 格式：JPG 或 PNG，优化后大小 < 200KB

### 自定义 Hero Banner

编辑 `hugo.toml` 中的配置：

```toml
[params]
  avatarTitle = "Wenbo Li"
  avatarSubtitle = "机器学习 | 机器视觉 | 工程技术"
  avatarImage = "/images/avatar.jpg"
  
  [[params.social]]
    name = "GitHub"
    url = "https://github.com/WenboLi-CN-DE"
    icon = "github"
```

## 性能指标

### Lighthouse 评分目标

- **Performance**: ≥ 90
- **Accessibility**: ≥ 95
- **Best Practices**: ≥ 95
- **SEO**: ≥ 95

### 优化措施

1. **关键 CSS 内联**：首屏样式直接嵌入 HTML
2. **图片懒加载**：非首屏图片延迟加载
3. **字体优化**：使用 `font-display: swap`
4. **资源压缩**：Hugo 构建时自动 minify
5. **GPU 加速**：动画使用 `transform` 和 `opacity`

### 加载时间

- **首次内容绘制（FCP）**: < 1.5s
- **最大内容绘制（LCP）**: < 2.5s
- **累积布局偏移（CLS）**: < 0.1
- **首次输入延迟（FID）**: < 100ms

## 浏览器兼容性

### 支持的浏览器

- Chrome/Edge ≥ 90
- Firefox ≥ 88
- Safari ≥ 14
- iOS Safari ≥ 14
- Android Chrome ≥ 90

### 降级策略

- **CSS Grid**：不支持时回退到 Flexbox
- **CSS 变量**：不支持时使用默认颜色
- **Intersection Observer**：不支持时直接加载图片
- **Sticky 定位**：不支持时 Hero Banner 正常滚动

## 可访问性

### WCAG 2.1 AA 标准

- **颜色对比度**：文字与背景对比度 ≥ 4.5:1
- **键盘导航**：所有交互元素可通过键盘访问
- **屏幕阅读器**：语义化 HTML，ARIA 标签
- **焦点指示**：清晰的焦点样式
- **响应式文字**：支持浏览器缩放至 200%

### 实现细节

- 使用语义化标签（`<article>`, `<nav>`, `<main>`）
- 图片添加 `alt` 属性
- 链接有明确的文字说明
- 表单元素关联 `<label>`
- 跳过导航链接（Skip to content）

## 维护说明

### 添加新样式

1. 在 `themes/custom-theme/static/css/` 创建新文件
2. 在 `layouts/partials/head.html` 中引入
3. 使用 CSS 变量保持一致性
4. 测试亮色/暗色模式

### 修改布局

1. 编辑 `layouts/index.html` 或相关 partial
2. 保持响应式断点一致
3. 测试不同屏幕尺寸
4. 验证可访问性

### 性能监控

定期检查：
- Lighthouse 评分
- 页面加载时间
- 图片大小和格式
- JavaScript 执行时间

### 常见问题

**Q: 封面图不显示？**
A: 检查路径是否正确，图片是否在 `static/images/` 目录。

**Q: 暗色模式样式异常？**
A: 确保使用 CSS 变量，而不是硬编码颜色。

**Q: 移动端布局错乱？**
A: 检查响应式断点，确保使用正确的媒体查询。

**Q: 懒加载不工作？**
A: 检查浏览器是否支持 Intersection Observer，查看控制台错误。

## 未来改进

- [ ] 添加文章搜索功能
- [ ] 实现无限滚动加载
- [ ] 支持文章分类筛选
- [ ] 添加阅读进度条
- [ ] 集成评论系统
- [ ] 支持多语言切换

## 参考资源

- [Hugo 官方文档](https://gohugo.io/documentation/)
- [CSS Grid 布局指南](https://css-tricks.com/snippets/css/complete-guide-grid/)
- [Intersection Observer API](https://developer.mozilla.org/en-US/docs/Web/API/Intersection_Observer_API)
- [WCAG 2.1 标准](https://www.w3.org/WAI/WCAG21/quickref/)
- [Web Vitals](https://web.dev/vitals/)
