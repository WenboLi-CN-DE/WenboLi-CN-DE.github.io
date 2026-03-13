#!/bin/bash

# Jekyll 到 Hugo 文章迁移脚本
SOURCE_DIR="/home/wenbo/WenboLi-CN-DE.github.io/_posts"
TARGET_DIR="/home/wenbo/WenboLi-CN-DE.github.io/hugo-site/content/posts"

# 创建目标目录
mkdir -p "$TARGET_DIR"

# 遍历所有 markdown 文件
for file in "$SOURCE_DIR"/*.md; do
    if [ -f "$file" ]; then
        filename=$(basename "$file")
        echo "Processing: $filename"
        
        # 复制文件到 Hugo content 目录
        cp "$file" "$TARGET_DIR/$filename"
        
        echo "Migrated: $filename"
    fi
done

echo "Migration complete! Total files: $(ls -1 "$TARGET_DIR" | wc -l)"
