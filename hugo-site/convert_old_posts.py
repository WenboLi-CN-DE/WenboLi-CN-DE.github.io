#!/usr/bin/env python3
"""Hugo 博客旧文章批量转换脚本"""

import re
from pathlib import Path

POSTS_DIR = "content/posts"
DRY_RUN = False

CATEGORY_KEYWORDS = {
    "engineering": [
        "机器视觉",
        "汽车视觉",
        "图像处理",
        "边缘检测",
        "曲线拟合",
        "图像分割",
        "模式识别",
        "相机",
        "光学",
        "Optics",
        "Vision",
        "SLAM",
        "定位",
        "地图构建",
        "道路识别",
        "行为识别",
        "追踪",
        "Tracking",
        "Binocular",
        "Stereo",
    ],
    "ai": [
        "机器学习",
        "深度学习",
        "神经网络",
        "强化学习",
        "学习",
        "Machine Learning",
        "Deep Learning",
        "Neural",
        "Reinforcement",
        "算法",
        "模型训练",
        "梯度下降",
    ],
    "code": [
        "编程",
        "代码",
        "系统",
        "Linux",
        "Win",
        "文件互传",
        "软件",
        "工具",
        "开发",
    ],
    "life": ["生活", "记录", "随笔", "日常"],
    "thought": ["思考", "想法", "观点", "思考"],
}


def detect_category(title: str, content: str) -> str:
    text = f"{title} {content}".lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        for keyword in keywords:
            if keyword.lower() in text:
                return category
    return "engineering"


def parse_frontmatter(content: str) -> tuple[dict, str, str]:
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)", content, re.DOTALL)
    if not match:
        return {}, "", content

    fm_raw = match.group(1)
    body = match.group(2)

    fm_dict = {}
    for line in fm_raw.split("\n"):
        if ":" in line:
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            if value.startswith("[") and value.endswith("]"):
                items = value[1:-1].split(",")
                fm_dict[key] = [
                    item.strip().strip("\"'") for item in items if item.strip()
                ]
            else:
                fm_dict[key] = value.strip("\"'")

    return fm_dict, fm_raw, body


def format_value(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, list):
        if len(value) == 0:
            return "[]"
        items = ", ".join(f'"{item}"' for item in value)
        return f"[{items}]"
    elif isinstance(value, str):
        return f'"{value}"'
    return str(value)


def convert_article(file_path: Path) -> dict:
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    fm_dict, fm_raw, body = parse_frontmatter(content)

    if not fm_dict:
        return {
            "file": file_path.name,
            "status": "skipped",
            "reason": "无 Front Matter",
        }

    changes = []

    if "layout" in fm_dict and fm_dict["layout"] == "post":
        del fm_dict["layout"]
        changes.append("移除 layout: post")

    if "tag" in fm_dict:
        fm_dict["tags"] = fm_dict.pop("tag")
        changes.append("tag → tags")

    if "tags" in fm_dict and not isinstance(fm_dict["tags"], list):
        fm_dict["tags"] = [fm_dict["tags"]]

    if "categories" in fm_dict:
        if isinstance(fm_dict["categories"], str):
            cat_value = fm_dict["categories"]
            if cat_value in CATEGORY_KEYWORDS:
                fm_dict["categories"] = [cat_value]
            else:
                fm_dict["categories"] = [
                    detect_category(fm_dict.get("title", ""), body)
                ]
            changes.append("修正 categories 格式")
    else:
        title = fm_dict.get("title", "")
        fm_dict["categories"] = [detect_category(title, body)]
        changes.append(f"添加 categories: {fm_dict['categories']}")

    if isinstance(fm_dict["categories"], list):
        valid_categories = []
        for cat in fm_dict["categories"]:
            if cat in CATEGORY_KEYWORDS:
                valid_categories.append(cat)
            else:
                valid_categories.append("engineering")
        fm_dict["categories"] = valid_categories

    field_order = [
        "title",
        "date",
        "description",
        "categories",
        "tags",
        "cover",
        "mathjax",
    ]
    new_fm_lines = []

    for field in field_order:
        if field in fm_dict:
            new_fm_lines.append(f"{field}: {format_value(fm_dict[field])}")

    for field, value in fm_dict.items():
        if field not in field_order:
            new_fm_lines.append(f"{field}: {format_value(value)}")

    new_fm = "---\n" + "\n".join(new_fm_lines) + "\n---\n\n"
    new_content = new_fm + body

    return {
        "file": file_path.name,
        "status": "changed",
        "changes": changes,
        "new_content": new_content,
    }


def main():
    posts_path = Path(POSTS_DIR)

    if not posts_path.exists():
        print(f"错误：目录 {POSTS_DIR} 不存在")
        return

    md_files = list(posts_path.glob("*.md"))
    print(f"找到 {len(md_files)} 个 Markdown 文件\n")

    results = []

    for file_path in md_files:
        if "example" in file_path.name:
            continue

        result = convert_article(file_path)
        results.append(result)

    print("=" * 60)
    print("转换报告")
    print("=" * 60)

    changed = [r for r in results if r["status"] == "changed"]
    skipped = [r for r in results if r["status"] == "skipped"]

    print(f"\n总计：{len(results)} 个文件")
    print(f"需要修改：{len(changed)} 个")
    print(f"跳过：{len(skipped)} 个\n")

    if DRY_RUN:
        print("🔍 DRY RUN 模式 - 未实际修改文件\n")

    for result in changed:
        print(f"📄 {result['file']}")
        for change in result["changes"]:
            print(f"   ✓ {change}")
        print()

    if not DRY_RUN:
        print("\n开始执行转换...")
        for result in changed:
            file_path = posts_path / result["file"]
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(result["new_content"])
        print("✅ 转换完成！")
    else:
        print("\n" + "=" * 60)
        print("提示：预览模式结束。确认无误后，修改脚本中 DRY_RUN = False 再次运行")
        print("=" * 60)
    print("转换报告")
    print("=" * 60)

    changed = [r for r in results if r["status"] == "changed"]
    skipped = [r for r in results if r["status"] == "skipped"]

    print(f"\n总计：{len(results)} 个文件")
    print(f"需要修改：{len(changed)} 个")
    print(f"跳过：{len(skipped)} 个\n")

    if DRY_RUN:
        print("🔍 DRY RUN 模式 - 未实际修改文件\n")

    # 详细报告
    for result in changed:
        print(f"📄 {result['file']}")
        for change in result["changes"]:
            print(f"   ✓ {change}")
        print()

    # 实际执行
    if not DRY_RUN:
        print("\n开始执行转换...")
        for result in changed:
            file_path = posts_path / result["file"]
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(result["new_content"])
        print("✅ 转换完成！")
    else:
        print("\n" + "=" * 60)
        print("提示：预览模式结束。确认无误后，修改脚本中 DRY_RUN = False 再次运行")
        print("=" * 60)


if __name__ == "__main__":
    main()
