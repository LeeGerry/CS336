# 从零构建大语言模型 · 面向移动端工程师的 Stanford CS336 中文学习笔记

📖 **在线阅读：<https://leegerry.github.io/CS336/>**

这是 Stanford [CS336《Language Modeling from Scratch》](https://cs336.stanford.edu/)（Spring 2026）全部 18 讲的中文重讲版。读者设定为 Android / iOS 工程师，只要求一点微积分和一点 Python。

- **19 章 + 2 个附录**，每一讲对应一章，另有一章从零补齐所需数学（第 0 章）
- **48 个交互演示**：BPE 合并动画、自注意力单步计算、Roofline、kernel 融合、集合通信、ZeRO 显存、流水线气泡、KV Cache、投机解码、缩放定律拟合、IsoFLOP、μP、困惑度、MinHash 去重、DPO、GRPO、ViT patch 等
- 每章开头嵌入原课视频，正文关键处附时间戳链接，可以跳到对应位置
- 附录 A 是五个作业的实战路线，附录 B 是 130 条术语表

## 目录

| 章 | 主题 | 原课 |
|---|---|---|
| [0](docs/chapters/ch00-primer.md) | 预备知识速成 | 原创 |
| [1](docs/chapters/ch01-overview-tokenization.md) | 课程总览与分词（BPE） | L1 |
| [2](docs/chapters/ch02-pytorch-resource.md) | PyTorch 与资源核算 | L2 |
| [3](docs/chapters/ch03-architectures.md) | Transformer 架构与现代变体 | L3 |
| [4](docs/chapters/ch04-attention-alternatives.md) | 注意力替代方案与 MoE | L4 |
| [5](docs/chapters/ch05-gpus-tpus.md) | GPU 与 TPU | L5 |
| [6](docs/chapters/ch06-kernels-triton.md) | Kernel 与 Triton | L6 |
| [7](docs/chapters/ch07-parallelism-1.md) | 并行训练（上） | L7 |
| [8](docs/chapters/ch08-parallelism-2.md) | 并行训练（下） | L8 |
| [9](docs/chapters/ch09-scaling-laws-1.md) | 缩放定律（上） | L9 |
| [10](docs/chapters/ch10-inference.md) | 推理 | L10 |
| [11](docs/chapters/ch11-scaling-laws-2.md) | 缩放定律（下） | L11 |
| [12](docs/chapters/ch12-evaluation.md) | 评测 | L12 |
| [13](docs/chapters/ch13-data-sources.md) | 数据（上）：来源与数据集 | L13 |
| [14](docs/chapters/ch14-data-processing.md) | 数据（下）：处理流水线 | L14 |
| [15](docs/chapters/ch15-mid-post-training.md) | 中训练、SFT 与 RLHF | L15 |
| [16](docs/chapters/ch16-rlvr.md) | 可验证奖励强化学习（GRPO） | L16 |
| [17](docs/chapters/ch17-alignment-multimodality.md) | 多模态模型 | L17 |
| [18](docs/chapters/ch18-guest-dan-fu.md) | 嘉宾讲座：Dan Fu | Guest |
| [A](docs/chapters/appendix-a-assignments.md) | 作业实战路线 | — |
| [B](docs/chapters/appendix-b-glossary.md) | 术语表 | — |

## 本地运行

```bash
npm install
npm run dev      # http://localhost:5173
npm run build    # 输出到 docs/.vitepress/dist
```

推送到 `main` 分支后，GitHub Actions 会自动部署到 GitHub Pages。

## 声明

本项目是个人学习笔记，非官方出品。课程内容、讲义与视频版权归 Stanford CS336 课程组（Percy Liang、Tatsunori Hashimoto 等）所有。本仓库不包含官方幻灯片和图片，只链接原始资源。如有理解错误，欢迎提 Issue。
