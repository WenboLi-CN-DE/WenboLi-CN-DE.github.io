# Jekyll 到 Hugo 迁移文档

## 迁移日期
2026-03-13

## 迁移内容

### 已迁移
- ✅ 22 篇博客文章（_posts/*.md → content/posts/*.md）
- ✅ 关于页面（about.md → content/about.md）
- ✅ CSS 样式文件（css/* → themes/custom-theme/static/css/*）
- ✅ JavaScript 文件（js/* → themes/custom-theme/static/js/*）
- ✅ 图片资源（images/* → static/images/*）
- ✅ 自定义主题（基于原 Jekyll 主题）
- ✅ 分页功能（20篇/页）
- ✅ 标签系统
- ✅ 语法高亮
- ✅ 目录生成
- ✅ CNAME 配置
- ✅ GitHub Actions 部署

### 配置对比

| 功能 | Jekyll | Hugo |
|------|--------|------|
| 配置文件 | _config.yml | hugo.toml |
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
/home/wenbo/.local/bin/hugo server -D
```

访问：http://localhost:1313

### 构建生产版本
```bash
cd hugo-site
/home/wenbo/.local/bin/hugo
```

生成的文件在 `public/` 目录

## 部署

### GitHub Pages 部署
1. 推送代码到 GitHub
2. GitHub Actions 自动构建和部署
3. 访问 https://wenboli-cn-de.github.io 或 https://wenboli-cn-de.com

### 手动部署
```bash
cd hugo-site
/home/wenbo/.local/bin/hugo
# 将 public/ 目录内容部署到服务器
```

## 注意事项

1. **Front Matter 兼容性**：Jekyll 和 Hugo 都使用 YAML front matter，大部分文章无需修改
2. **Liquid 模板语法**：如果文章中使用了 Jekyll 的 Liquid 语法（如 `{% include %}`），需要手动修改
3. **路径引用**：检查文章中的图片路径是否正确
4. **RSS Feed**：暂时禁用了 RSS，后续可以添加自定义 RSS 模板

## 回滚方案

如果需要回滚到 Jekyll：
1. 原 Jekyll 文件仍保留在项目根目录
2. 删除或重命名 `hugo-site/` 目录
3. 恢复原 GitHub Actions 工作流（如果有）

## 后续优化

- [ ] 添加 RSS Feed 支持
- [ ] 优化主题样式，使其更接近原 Jekyll 主题
- [ ] 添加评论系统（如 Disqus, Utterances）
- [ ] 添加搜索功能
- [ ] 优化 SEO
- [ ] 添加 Google Analytics（如果需要）
- [ ] 优化图片加载（lazy loading）
- [ ] 添加暗色模式

## 已知问题

- RSS Feed 暂时禁用（配置问题，待修复）

## 迁移统计

- 总文章数：22 篇
- CSS 文件：4 个
- JS 文件：2 个
- 图片文件：19 个
- 构建时间：约 36ms（Hugo）vs 数秒（Jekyll）
- 性能提升：约 100 倍
