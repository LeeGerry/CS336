import { defineConfig } from 'vitepress'
import { withMermaid } from 'vitepress-plugin-mermaid'
import cjkFriendly from 'markdown-it-cjk-friendly'

const base = process.env.BASE_PATH || '/'

export default withMermaid(defineConfig({
  base,
  lang: 'zh-CN',
  title: '从零构建大语言模型',
  description: '面向移动端工程师的 Stanford CS336 中文学习笔记',
  cleanUrls: true,
  lastUpdated: true,
  markdown: { math: true, lineNumbers: false, config: (md) => { md.use(cjkFriendly) } },
  head: [['link', { rel: 'icon', href: `${base}favicon.svg` }]],
  themeConfig: {
    nav: [
      { text: '开始阅读', link: '/guide' },
      { text: '交互演示', link: '/gallery' },
      { text: '原课程', link: 'https://cs336.stanford.edu/' },
    ],
    sidebar: [
      { text: '开始', items: [
        { text: '导读：怎么用这本书', link: '/guide' },
        { text: '第 0 章 预备知识速成', link: '/chapters/ch00-primer' },
        { text: '交互演示总览', link: '/gallery' },
      ]},
      { text: '第一部分 · 基础', items: [
        { text: '第 1 章 课程总览与分词', link: '/chapters/ch01-overview-tokenization' },
        { text: '第 2 章 PyTorch 与资源核算', link: '/chapters/ch02-pytorch-resource' },
        { text: '第 3 章 模型架构', link: '/chapters/ch03-architectures' },
        { text: '第 4 章 注意力替代方案与 MoE', link: '/chapters/ch04-attention-alternatives' },
      ]},
      { text: '第二部分 · 系统', items: [
        { text: '第 5 章 GPU 与 TPU', link: '/chapters/ch05-gpus-tpus' },
        { text: '第 6 章 Kernel 与 Triton', link: '/chapters/ch06-kernels-triton' },
        { text: '第 7 章 并行训练（上）', link: '/chapters/ch07-parallelism-1' },
        { text: '第 8 章 并行训练（下）', link: '/chapters/ch08-parallelism-2' },
      ]},
      { text: '第三部分 · 缩放定律与推理', items: [
        { text: '第 9 章 缩放定律（上）', link: '/chapters/ch09-scaling-laws-1' },
        { text: '第 10 章 推理', link: '/chapters/ch10-inference' },
        { text: '第 11 章 缩放定律（下）', link: '/chapters/ch11-scaling-laws-2' },
      ]},
      { text: '第四部分 · 评估与数据', items: [
        { text: '第 12 章 评估', link: '/chapters/ch12-evaluation' },
        { text: '第 13 章 数据来源与数据集', link: '/chapters/ch13-data-sources' },
        { text: '第 14 章 数据处理', link: '/chapters/ch14-data-processing' },
      ]},
      { text: '第五部分 · 后训练与对齐', items: [
        { text: '第 15 章 中训练与后训练', link: '/chapters/ch15-mid-post-training' },
        { text: '第 16 章 强化学习 RLVR', link: '/chapters/ch16-rlvr' },
        { text: '第 17 章 多模态模型', link: '/chapters/ch17-alignment-multimodality' },
      ]},
      { text: '嘉宾讲座', items: [
        { text: '第 18 章 从推理系统看全栈创新（Dan Fu）', link: '/chapters/ch18-guest-dan-fu' },
      ]},
      { text: '附录', items: [
        { text: '附录 A 作业实战路线', link: '/chapters/appendix-a-assignments' },
        { text: '附录 B 术语表', link: '/chapters/appendix-b-glossary' },
      ]},
    ],
    outline: { level: [2, 3], label: '本页目录' },
    search: { provider: 'local' },
    docFooter: { prev: '上一章', next: '下一章' },
    socialLinks: [{ icon: 'github', link: 'https://github.com/LeeGerry/CS336' }],
    footer: { message: '基于 Stanford CS336（Spring 2026）公开课程整理的学习笔记，非官方。', copyright: '原课程版权归 Stanford CS336 课程组所有' },
  },
  mermaid: {},
}))
