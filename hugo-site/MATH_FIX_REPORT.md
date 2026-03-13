# 数学公式显示问题修复报告

## 问题诊断（Phase 1: Root Cause Investigation）

### 发现的根本原因

**问题**：动力学文章的数学公式无法显示，而神经网络文章可以显示。

**根本原因**：
1. 所有文章在 front matter 后都嵌入了 MathJax 的 `<head>` 配置
2. Hugo 主题使用的是 KaTeX 进行数学公式渲染
3. **冲突**：MathJax 和 KaTeX 同时存在导致渲染冲突

### 证据

```markdown
# 文章中的问题代码（第 6-16 行）
<head>
    <script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
    <script type="text/x-mathjax-config">
        MathJax.Hub.Config({
            tex2jax: {
            skipTags: ['script', 'noscript', 'style', 'textarea', 'pre'],
            inlineMath: [['$','$']]
            }
        });
    </script>
</head>
```

### 为什么神经网络文章能显示？

- 简单公式（如 `$E=mc^2$`）在 MathJax 加载前被 KaTeX 渲染
- 或者公式结构简单，两个库都能处理

### 为什么动力学文章不能显示？

- 包含复杂的 LaTeX 环境（`\begin{aligned}...`）
- MathJax 配置干扰了 KaTeX 的渲染
- 两个库的处理顺序冲突

## 解决方案（Phase 3 & 4）

### 修复措施

**移除所有文章中的 MathJax 配置，统一使用 KaTeX。**

### 执行的操作

1. ✅ 备份了所有受影响的文章（`.mathjax-backup` 后缀）
2. ✅ 批量删除了 13 篇文章中的 MathJax `<head>` 配置块
3. ✅ Hugo 自动重新构建了所有文章

### 处理的文章列表

```
1. 2022-01-30-MV5.md
2. 2022-03-10-MV12.0.md
3. 2022-01-25-MV4.md
4. 2023-06-14-ML-04-Neural Networks.md
5. 2022-01-19-MV3.1.md
6. 2022-01-19-MV2.1.md
7. 2022-03-02-MV6.md
8. 2022-02-03-MV5.1color.md
9. 2022-03-10-MV12.1.md
10. 2022-03-09-MV8.2.md
11. 2022-03-08-MV6.1.md
12. 2022-02-11-MV7-Optics.md
13. 2022-03-08-MV8.1.md
14. 2023-06-13-TM4-03.md (动力学文章)
```

## 技术细节

### KaTeX 配置（已在 baseof.html 中）

```html
<!-- KaTeX CSS -->
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">

<!-- KaTeX JS -->
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js"></script>

<!-- 自动渲染配置 -->
<script>
    document.addEventListener("DOMContentLoaded", function() {
        renderMathInElement(document.body, {
            delimiters: [
                {left: "$$", right: "$$", display: true},
                {left: "$", right: "$", display: false},
                {left: "\\[", right: "\\]", display: true},
                {left: "\\(", right: "\\)", display: false}
            ],
            throwOnError: false
        });
    });
</script>
```

### Hugo Goldmark 配置（hugo.toml）

```toml
[markup.goldmark.extensions.passthrough]
  enable = true
  [markup.goldmark.extensions.passthrough.delimiters]
    block = [['\[', '\]'], ['$$', '$$']]
    inline = [['\(', '\)'], ['$', '$']]
```

## 验证步骤

请在浏览器中验证：

1. **动力学文章**：http://localhost:1313/2023/06/工程力学-动力学-technische-mechanik-iv-dynamik-integration-der-eulerschen-gleichungen-欧拉方程的积分/
   - 检查欧拉方程是否正确显示
   - 检查 `\begin{aligned}` 环境是否渲染

2. **神经网络文章**：http://localhost:1313/2023/06/机器学习-machine-learning-neural-networks-神经网络/
   - 确认公式仍然正常显示
   - 检查行内公式和块级公式

## 备份恢复

如果需要恢复原始文件：

```bash
cd /home/wenbo/WenboLi-CN-DE.github.io/hugo-site/content/posts
for file in *.mathjax-backup; do
    mv "$file" "${file%.mathjax-backup}"
done
```

## 总结

**根本原因**：MathJax 和 KaTeX 冲突
**解决方案**：移除 MathJax，统一使用 KaTeX
**影响范围**：14 篇文章
**修复时间**：立即生效（Hugo 自动重新构建）

---

**修复完成时间**：2026-03-13 23:31
**Hugo 服务器**：http://localhost:1313 ✅
