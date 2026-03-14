/**
 * FlexSearch 中文搜索实现
 * 针对中文优化的客户端搜索
 */

document.addEventListener('DOMContentLoaded', async function() {
    const searchInput = document.getElementById('search-input');
    const searchResults = document.getElementById('search-results');
    
    if (!searchInput || !searchResults) return;
    
    // 创建 FlexSearch 文档索引（full 模式匹配分词后的中文索引）
    const index = new FlexSearch.Document({
        tokenize: 'full',
        charset: 'unicode:default',
        optimize: true,
        cache: true,
        document: {
            id: 'id',
            index: ['title', 'content', 'tags', 'categories', 'description']
        }
    });
    
    // 加载搜索索引
    let indexLoaded = false;
    let data = []; // 在外层定义，供 renderResults 使用
    try {
        const response = await fetch('/search-index.json');
        if (!response.ok) {
            throw new Error('索引文件不存在');
        }
        data = await response.json();
        
        // 批量添加到索引（处理空字符串，避免 FlexSearch tokenize 错误）
        data.forEach(item => {
            // 确保 tags 和 categories 是有效的字符串（空字符串会导致 FlexSearch 内部错误）
            const normalizedItem = {
                ...item,
                tags: item.tags || 'untagged',
                categories: item.categories || 'uncategorized'
            };
            index.add(normalizedItem);
        });
        
        indexLoaded = true;
        console.log(`✅ 搜索索引加载完成：${data.length} 篇文章`);
        
    } catch (error) {
        console.error('❌ 加载搜索索引失败:', error);
        searchResults.innerHTML = `
            <div class="search-error">
                <p>搜索索引加载失败</p>
                <p class="error-detail">${error.message}</p>
            </div>
        `;
        return;
    }
    
    // 搜索输入处理（带防抖）
    let debounceTimer;
    searchInput.addEventListener('input', function(e) {
        clearTimeout(debounceTimer);
        const query = e.target.value.trim();
        
        // 空查询处理
        if (query.length < 1) {
            searchResults.innerHTML = '<p class="search-hint">输入关键词开始搜索</p>';
            return;
        }
        
        // 防抖：150ms 后执行搜索
        debounceTimer = setTimeout(() => {
            if (!indexLoaded) {
                searchResults.innerHTML = '<p class="search-hint">索引加载中，请稍候...</p>';
                return;
            }
            
            // 执行搜索
            const results = index.search(query, {
                limit: 20,
                enrich: true
            });
            
            // 合并结果并去重
            const allResults = [];
            const seen = new Set();
            
            results.forEach(result => {
                result.result.forEach(id => {
                    if (!seen.has(id)) {
                        seen.add(id);
                        // 从原始数据中查找
                        const doc = data.find(d => d.id === id);
                        if (doc) allResults.push(doc);
                    }
                });
            });
            
            // 渲染结果
            renderResults(allResults, query);
        }, 150);
    });
    
    // 渲染搜索结果
    function renderResults(results, query) {
        if (results.length === 0) {
            searchResults.innerHTML = `
                <p class="no-results">未找到匹配的结果</p>
                <p class="search-tip">建议：尝试其他关键词或减少关键词数量</p>
            `;
            return;
        }
        
        const resultsHTML = results.map(item => {
            const snippet = generateSnippet(item.content || '', query);
            // 处理 tags 和 categories（可能是空字符串或已分词的字符串）
            const tags = typeof item.tags === 'string' ? item.tags.split(' ').filter(Boolean) : [];
            const categories = typeof item.categories === 'string' ? item.categories.split(' ').filter(Boolean) : [];
            
            return `
                <a href="${item.permalink}" class="result-item">
                    <div class="result-header">
                        <h3 class="result-title">${highlight(item.title, query)}</h3>
                        <span class="result-date">${item.date}</span>
                    </div>
                    ${item.description ? `<p class="result-description">${highlight(item.description, query)}</p>` : ''}
                    ${snippet ? `<p class="result-snippet">${snippet}</p>` : ''}
                    <div class="result-meta">
                        ${categories.length > 0 ? `<span class="result-category">${categories[0]}</span>` : ''}
                        ${tags.length > 0 ? `<span class="result-tags">${tags.slice(0, 3).join(' · ')}</span>` : ''}
                    </div>
                </a>
            `;
        }).join('');
        
        searchResults.innerHTML = `
            <p class="results-count">找到 ${results.length} 个结果</p>
            <div class="results-list">${resultsHTML}</div>
        `;
    }
    
    // 生成摘要片段（带关键词高亮）
    function generateSnippet(content, query, length = 150) {
        if (!content) return '';
        
        const lowerContent = content.toLowerCase();
        const lowerQuery = query.toLowerCase();
        
        // 查找查询词位置
        const matchIndex = lowerContent.indexOf(lowerQuery);
        
        if (matchIndex !== -1) {
            // 从匹配位置前后截取
            const start = Math.max(0, matchIndex - 50);
            const end = Math.min(content.length, matchIndex + query.length + 100);
            const snippet = content.substring(start, end);
            return (start > 0 ? '...' : '') + highlight(snippet, query) + (end < content.length ? '...' : '');
        }
        
        // 没找到匹配，返回开头
        return highlight(content.substring(0, length) + '...', query);
    }
    
    // 关键词高亮
    function highlight(text, query) {
        if (!query || !text) return text;
        
        // 转义 HTML 特殊字符
        const escapeHtml = (str) => {
            return str
                .replace(/&/g, '&amp;')
                .replace(/</g, '&lt;')
                .replace(/>/g, '&gt;')
                .replace(/"/g, '&quot;')
                .replace(/'/g, '&#039;');
        };
        
        // 创建不区分大小写的正则
        const regex = new RegExp(`(${escapeHtml(query)})`, 'gi');
        return escapeHtml(text).replace(regex, '<mark>$1</mark>');
    }
});
