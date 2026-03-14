/**
 * 使用 nodejieba 对 Hugo 搜索索引进行中文分词
 * 生成优化后的搜索索引文件
 */

const { Jieba } = require('@node-rs/jieba');
const fs = require('fs');
const path = require('path');

// 配置
const INPUT_FILE = path.join(__dirname, 'public', 'search-index.json');
const OUTPUT_FILE = path.join(__dirname, 'public', 'search-index.json');

// 创建结巴分词实例
const jieba = new Jieba();

console.log('🔪 开始中文分词处理...');
console.log(`📄 输入文件：${INPUT_FILE}`);
console.log(`📝 输出文件：${OUTPUT_FILE}`);

// 读取原始索引
let data;
try {
  const content = fs.readFileSync(INPUT_FILE, 'utf-8');
  data = JSON.parse(content);
  console.log(`✅ 读取索引完成：${data.length} 篇文章`);
} catch (error) {
  console.error('❌ 读取索引失败:', error.message);
  process.exit(1);
}

// 分词处理
console.log('🔪 正在进行中文分词...');
const startTime = Date.now();

const segmentedData = data.map((item, index) => {
  // 进度显示
  if ((index + 1) % 10 === 0 || index === data.length - 1) {
    process.stdout.write(`\r  处理进度：${index + 1}/${data.length} (${Math.round((index + 1) / data.length * 100)}%)`);
  }

  return {
    ...item,
    // 对标题进行分词
    title: jieba.cut(item.title || '').join(' '),
    // 对内容进行分词
    content: jieba.cut(item.content || '').join(' '),
    // 对描述进行分词
    description: jieba.cut(item.description || '').join(' '),
    // tags 和 categories 保持原样
    tags: item.tags,
    categories: item.categories
  };
});

process.stdout.write('\n');
const endTime = Date.now();
console.log(`✅ 分词完成，耗时：${endTime - startTime}ms`);

// 写入分词后的索引
try {
  const output = JSON.stringify(segmentedData, null, 2);
  fs.writeFileSync(OUTPUT_FILE, output, 'utf-8');
  
  // 计算文件大小
  const stats = fs.statSync(OUTPUT_FILE);
  const sizeKB = (stats.size / 1024).toFixed(2);
  
  console.log(`✅ 索引文件已保存`);
  console.log(`📊 文件大小：${sizeKB} KB`);
  console.log(`📈 平均文章大小：${(sizeKB / data.length).toFixed(2)} KB`);
  
  // 显示示例
  console.log('\n📋 分词示例:');
  const sample = segmentedData[0];
  if (sample) {
    console.log(`   标题：${sample.title.substring(0, 100)}...`);
    console.log(`   标签：${sample.tags || '无'}`);
    console.log(`   分类：${sample.categories || '无'}`);
  }
  
  console.log('\n✨ 中文分词处理完成！');
} catch (error) {
  console.error('❌ 保存索引失败:', error.message);
  process.exit(1);
}
