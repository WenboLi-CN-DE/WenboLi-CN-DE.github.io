# 高傲的电工李 - 个人博客

这是李文博的个人博客，使用 Hugo 静态站点生成器构建。

## 博客链接

- [博客主页](https://wenboli-cn-de.github.io)

## 技术栈

- **静态站点生成器**: Hugo v0.120.0+
- **主题**: 自定义主题 (custom-theme)
- **部署**: GitHub Pages
- **语言**: 中文 (zh-CN)

## 最近更新

### 前端页面优化 (2026-03-13)

完成了博客前端的现代化改造：

- ✅ CSS Variables 系统与主题切换
- ✅ 响应式 Header 组件
- ✅ 分类系统（人间便签、思维漫游、工程随笔、代码诗篇、智识前沿）
- ✅ 分类导航栏组件
- ✅ 亮色/暗色主题切换

## 项目结构

```
.
├── hugo-site/              # Hugo 站点目录
│   ├── content/           # 文章内容
│   ├── themes/            # 主题文件
│   │   └── custom-theme/  # 自定义主题
│   ├── static/            # 静态资源
│   ├── layouts/           # 布局模板
│   └── hugo.toml          # Hugo 配置文件
├── docs/                  # 文档目录
└── README.md              # 本文件
```

## 本地开发

### 环境要求

- Hugo v0.120.0 或更高版本（需要 extended 版本）
- Git

### 运行本地服务器

```bash
cd hugo-site
hugo server -D
```

访问 http://localhost:1313 查看效果。

### 构建生产版本

```bash
cd hugo-site
hugo --minify
```

构建结果将输出到 `hugo-site/public/` 目录。

## 文章分类

博客文章按以下分类组织：

- 📝 **人间便签** - 生活的点滴记录
- 💭 **思维漫游** - 思想的自由探索
- ⚙️ **工程随笔** - 工程实践与思考
- 💻 **代码诗篇** - 代码的艺术与技巧
- 🤖 **智识前沿** - 人工智能的探索

## 功能特性

- 🎨 现代化 UI 设计
- 🌓 亮色/暗色主题切换
- 📱 响应式设计，完美适配移动端
- 🏷️ 分类和标签系统
- 🔍 文章搜索（规划中）
- 💬 评论系统（规划中）

## 迁移历史

本博客原使用 Jekyll 构建，已于 2026-03 完成向 Hugo 的迁移。详见 [迁移文档](hugo-site/MIGRATION.md)。

## 作者

**李文博**
- GitHub: [@WenboLi-CN-DE](https://github.com/WenboLi-CN-DE)
- Email: lwb_010@163.com
- 领域: 机电 / 机器学习

## 许可证

本项目内容采用 [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) 许可协议。

---

最后更新: 2026-03-13
