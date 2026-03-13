# 高傲的电工李 - Hugo 博客

个人技术博客，使用 Hugo 静态站点生成器构建。

## 关于

- **作者**：李文博
- **描述**：机电 / 机器学习
- **网站**：https://wenboli-cn-de.github.io 或 https://wenboli-cn-de.com

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
# 下载预编译版本
wget https://github.com/gohugoio/hugo/releases/download/v0.120.0/hugo_extended_0.120.0_linux-amd64.tar.gz
tar -xzf hugo_extended_0.120.0_linux-amd64.tar.gz
mkdir -p ~/.local/bin
mv hugo ~/.local/bin/
export PATH="$HOME/.local/bin:$PATH"
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
cd hugo-site
hugo
```

生成的文件在 `public/` 目录。

## 内容管理

### 创建新文章

```bash
cd hugo-site
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
├── hugo.toml            # 站点配置
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

工作流文件：`.github/workflows/hugo.yml`

## 迁移说明

本站点从 Jekyll 迁移到 Hugo。详细迁移文档请查看 [MIGRATION.md](MIGRATION.md)。

## 许可

MIT License

## 联系方式

- GitHub: [@WenboLi-CN-DE](https://github.com/WenboLi-CN-DE)
- Email: lwb_010@163.com

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
