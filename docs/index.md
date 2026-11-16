---
layout: home
hero:
  name: 从零构建大语言模型
  text: 面向移动端工程师的 CS336 中文学习笔记
  tagline: 18 讲 Stanford CS336（Spring 2026）全部重讲一遍：从分词、Transformer、GPU、并行、缩放定律，到数据、RLHF 与 RLVR。只需要一点微积分和一点 Python。
  actions:
    - theme: brand
      text: 开始阅读 →
      link: /guide
    - theme: alt
      text: 先补数学（第 0 章）
      link: /chapters/ch00-primer
    - theme: alt
      text: 交互演示
      link: /gallery
features:
  - icon: 🧱
    title: 第一部分 · 基础
    details: 分词（BPE）、PyTorch 与 FLOPs 核算、Transformer 架构、注意力的替代方案。学完能自己写出并训练一个小 LLM。
    link: /chapters/ch01-overview-tokenization
  - icon: ⚡
    title: 第二部分 · 系统
    details: GPU/TPU 原理、Kernel 与 Triton、数据/张量/流水线并行。学会让硬件跑满。
    link: /chapters/ch05-gpus-tpus
  - icon: 📈
    title: 第三部分 · 缩放定律与推理
    details: 用小实验预测大模型：Chinchilla、IsoFLOP、超参数迁移；以及推理与 KV Cache。花一千万美元之前先算清楚。
    link: /chapters/ch09-scaling-laws-1
  - icon: 🗂️
    title: 第四部分 · 评估与数据
    details: 怎么衡量模型好坏，数据从哪来，怎么过滤、去重、配比。
    link: /chapters/ch12-evaluation
  - icon: 🎯
    title: 第五部分 · 后训练
    details: SFT、DPO、RLHF、GRPO 与可验证奖励强化学习，以及多模态。
    link: /chapters/ch15-mid-post-training
  - icon: 🕹️
    title: 每章都有交互演示
    details: BPE 合并动画、注意力热力图、Roofline、集合通信、KV Cache、缩放曲线拟合、MinHash 去重、GRPO 循环……拖一拖就懂。
    link: /gallery
---
