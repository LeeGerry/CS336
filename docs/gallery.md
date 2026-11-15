# 交互演示总览

书中所有交互演示都汇总在下面。每个演示都是一个独立网页，点击"打开"可以全屏使用，点击章节名可以回到对应章节阅读上下文。

| 演示 | 所在章节 | |
|---|---|---|
| 作业路线规划器：按硬件条件看每道题在哪做 | [附录 A 作业实战路线：五个作业怎么做、在哪做、怎么自测](./chapters/appendix-a-assignments) | <a href="./demos/appA-roadmap.html" target="_blank">打开 ↗</a> |
| 反向传播单步演示：前向算值，反向传梯度 | [第 0 章 预备知识速成](./chapters/ch00-primer) | <a href="./demos/ch00-backprop.html" target="_blank">打开 ↗</a> |
| 浮点数格式对比：同一个数在 fp32 / fp16 / bf16 里变成什么 | [第 0 章 预备知识速成](./chapters/ch00-primer) | <a href="./demos/ch00-float.html" target="_blank">打开 ↗</a> |
| 梯度下降：小球滑坡 | [第 0 章 预备知识速成](./chapters/ch00-primer) | <a href="./demos/ch00-gradient-descent.html" target="_blank">打开 ↗</a> |
| 矩阵乘法：形状规则 + 逐元素计算 | [第 0 章 预备知识速成](./chapters/ch00-primer) | <a href="./demos/ch00-matmul.html" target="_blank">打开 ↗</a> |
| softmax、温度、交叉熵与困惑度 | [第 0 章 预备知识速成](./chapters/ch00-primer) | <a href="./demos/ch00-softmax.html" target="_blank">打开 ↗</a> |
| BPE 训练逐步合并动画 | [第 1 章 课程总览与分词（Tokenization）](./chapters/ch01-overview-tokenization) | <a href="./demos/ch01-bpe.html" target="_blank">打开 ↗</a> |
| 字符 / 字节 / 单词 / BPE 四种分词对比 | [第 1 章 课程总览与分词（Tokenization）](./chapters/ch01-overview-tokenization) | <a href="./demos/ch01-tokenizers.html" target="_blank">打开 ↗</a> |
| einops 维度重排可视化 | [第 2 章 PyTorch 与资源核算](./chapters/ch02-pytorch-resource) | <a href="./demos/ch02-einops.html" target="_blank">打开 ↗</a> |
| 训练资源计算器：6ND 与每参数字节数 | [第 2 章 PyTorch 与资源核算](./chapters/ch02-pytorch-resource) | <a href="./demos/ch02-resource-calculator.html" target="_blank">打开 ↗</a> |
| 算术强度与 Roofline | [第 2 章 PyTorch 与资源核算](./chapters/ch02-pytorch-resource) | <a href="./demos/ch02-roofline.html" target="_blank">打开 ↗</a> |
| Transformer block 数据流 + 参数计算器 | [第 3 章 模型架构：从零搭一个 Transformer，再看现代 LLM 都改了什么](./chapters/ch03-architectures) | <a href="./demos/ch03-block-flow.html" target="_blank">打开 ↗</a> |
| RoPE 旋转可视化：平移整句，点积不变 | [第 3 章 模型架构：从零搭一个 Transformer，再看现代 LLM 都改了什么](./chapters/ch03-architectures) | <a href="./demos/ch03-rope.html" target="_blank">打开 ↗</a> |
| 自注意力逐步计算（带因果掩码开关） | [第 3 章 模型架构：从零搭一个 Transformer，再看现代 LLM 都改了什么](./chapters/ch03-architectures) | <a href="./demos/ch03-self-attention.html" target="_blank">打开 ↗</a> |
| 全注意力 vs 滑动窗口 vs 线性注意力 vs 混合：成本随序列长度变化 | [第 4 章 注意力的替代方案与混合专家（MoE）](./chapters/ch04-attention-alternatives) | <a href="./demos/ch04-attention-cost.html" target="_blank">打开 ↗</a> |
| 状态递推动画：线性注意力 / Mamba-2 / Gated DeltaNet | [第 4 章 注意力的替代方案与混合专家（MoE）](./chapters/ch04-attention-alternatives) | <a href="./demos/ch04-linear-recurrence.html" target="_blank">打开 ↗</a> |
| MoE 路由玩具模拟：负载均衡与专家塌缩 | [第 4 章 注意力的替代方案与混合专家（MoE）](./chapters/ch04-attention-alternatives) | <a href="./demos/ch04-moe-routing.html" target="_blank">打开 ↗</a> |
| GPU 内存层级：数据从哪儿来 | [第 5 章 GPU 与 TPU：硬件如何决定模型跑多快](./chapters/ch05-gpus-tpus) | <a href="./demos/ch05-memory-hierarchy.html" target="_blank">打开 ↗</a> |
| Roofline 屋顶线：拖动算术强度看瓶颈 | [第 5 章 GPU 与 TPU：硬件如何决定模型跑多快](./chapters/ch05-gpus-tpus) | <a href="./demos/ch05-roofline.html" target="_blank">打开 ↗</a> |
| 波量化：矩阵大 1，反而慢一截 | [第 5 章 GPU 与 TPU：硬件如何决定模型跑多快](./chapters/ch05-gpus-tpus) | <a href="./demos/ch05-wave-quantization.html" target="_blank">打开 ↗</a> |
| 算子融合：数一数往返 HBM 的次数 | [第 6 章 Kernel 与 Triton：亲手写 GPU 程序](./chapters/ch06-kernels-triton) | <a href="./demos/ch06-fusion.html" target="_blank">打开 ↗</a> |
| 分块矩阵乘法：一块搬进共享内存，反复使用 | [第 6 章 Kernel 与 Triton：亲手写 GPU 程序](./chapters/ch06-kernels-triton) | <a href="./demos/ch06-tiling.html" target="_blank">打开 ↗</a> |
| 集合通信：4 张 GPU 间的数据流动画 | [第 7 章 并行训练（上）：集合通信与三种基本切分](./chapters/ch07-parallelism-1) | <a href="./demos/ch07-collectives.html" target="_blank">打开 ↗</a> |
| 梯度 all-reduce 耗时计算器 | [第 7 章 并行训练（上）：集合通信与三种基本切分](./chapters/ch07-parallelism-1) | <a href="./demos/ch07-comm-cost.html" target="_blank">打开 ↗</a> |
| 六种切法：每卡存什么、要多少显存 | [第 8 章 并行训练（下）：ZeRO/FSDP、模型并行与 3D 并行](./chapters/ch08-parallelism-2) | <a href="./demos/ch08-parallel-memory.html" target="_blank">打开 ↗</a> |
| 流水线并行的气泡：四种调度的时间线 | [第 8 章 并行训练（下）：ZeRO/FSDP、模型并行与 3D 并行](./chapters/ch08-parallelism-2) | <a href="./demos/ch08-pipeline-bubble.html" target="_blank">打开 ↗</a> |
| Chinchilla IsoFLOP：固定算力，滑动模型大小找最优 | [第 9 章 缩放定律（上）：用小模型预测大模型](./chapters/ch09-scaling-laws-1) | <a href="./demos/ch09-isoflop.html" target="_blank">打开 ↗</a> |
| 缩放定律拟合与外推：log-log 直线、噪声与渐近线 | [第 9 章 缩放定律（上）：用小模型预测大模型](./chapters/ch09-scaling-laws-1) | <a href="./demos/ch09-powerlaw-fit.html" target="_blank">打开 ↗</a> |
| Prefill vs Decode：KV cache 增长动画与显存计算器 | [第 10 章 推理：KV cache、访存瓶颈与加速技巧](./chapters/ch10-inference) | <a href="./demos/ch10-kv-cache.html" target="_blank">打开 ↗</a> |
| 投机解码：猜测-验证动画与无损性抽样验证 | [第 10 章 推理：KV cache、访存瓶颈与加速技巧](./chapters/ch10-inference) | <a href="./demos/ch10-speculative.html" target="_blank">打开 ↗</a> |
| μP：一步更新后的激活变化 & 最优学习率随宽度的漂移 | [第 11 章 缩放定律（下）：真实配方、学习率调度、优化器与 μP](./chapters/ch11-scaling-laws-2) | <a href="./demos/ch11-mup.html" target="_blank">打开 ↗</a> |
| 余弦 vs WSD：数据扫描的成本与损失曲线 | [第 11 章 缩放定律（下）：真实配方、学习率调度、优化器与 μP](./chapters/ch11-scaling-laws-2) | <a href="./demos/ch11-wsd.html" target="_blank">打开 ↗</a> |
| 基准测试的噪声与污染：差 2 分的两个模型分得出来吗？ | [第 12 章 评测：怎样才算一个"好"模型？](./chapters/ch12-evaluation) | <a href="./demos/ch12-benchmark-noise.html" target="_blank">打开 ↗</a> |
| 竞技场 Elo 排名：从稀疏的两两投票拟合分数，以及长度偏好如何扭曲排名 | [第 12 章 评测：怎样才算一个"好"模型？](./chapters/ch12-evaluation) | <a href="./demos/ch12-elo-arena.html" target="_blank">打开 ↗</a> |
| 困惑度计算器：逐 token 概率 → 交叉熵 → 困惑度 | [第 12 章 评测：怎样才算一个"好"模型？](./chapters/ch12-evaluation) | <a href="./demos/ch12-perplexity.html" target="_blank">打开 ↗</a> |
| 预训练数据集年表：来源、规模、过滤方式 | [第 13 章 数据（上）：数据从哪里来——来源、版权与经典数据集](./chapters/ch13-data-sources) | <a href="./demos/ch13-dataset-timeline.html" target="_blank">打开 ↗</a> |
| 预训练数据漏斗：每一步还剩多少 | [第 13 章 数据（上）：数据从哪里来——来源、版权与经典数据集](./chapters/ch13-data-sources) | <a href="./demos/ch13-pipeline-funnel.html" target="_blank">打开 ↗</a> |
| 数据配比计算器：权重、epoch、UniMax 与模拟 epoch | [第 14 章 数据（下）：处理流水线——转换、过滤、去重、配比与后训练数据](./chapters/ch14-data-processing) | <a href="./demos/ch14-data-mixing.html" target="_blank">打开 ↗</a> |
| MinHash + LSH 近似去重 | [第 14 章 数据（下）：处理流水线——转换、过滤、去重、配比与后训练数据](./chapters/ch14-data-processing) | <a href="./demos/ch14-minhash-lsh.html" target="_blank">打开 ↗</a> |
| 质量分类器：训练、打分、按阈值筛选 | [第 14 章 数据（下）：处理流水线——转换、过滤、去重、配比与后训练数据](./chapters/ch14-data-processing) | <a href="./demos/ch14-quality-filter.html" target="_blank">打开 ↗</a> |
| DPO：偏好对的概率如何变化 | [第 15 章 预训练之后：中训练、SFT 与 RLHF](./chapters/ch15-mid-post-training) | <a href="./demos/ch15-dpo.html" target="_blank">打开 ↗</a> |
| SFT 对话模板与损失掩码 | [第 15 章 预训练之后：中训练、SFT 与 RLHF](./chapters/ch15-mid-post-training) | <a href="./demos/ch15-sft-loss-mask.html" target="_blank">打开 ↗</a> |
| GRPO 循环：采样 → 验证打分 → 组内归一化 → 更新 | [第 16 章 可验证奖励强化学习（RLVR）：从 PPO 到 GRPO，再到 R1、Kimi 与 Qwen](./chapters/ch16-rlvr) | <a href="./demos/ch16-grpo.html" target="_blank">打开 ↗</a> |
| GRPO 的两个偏差：长度归一化与标准差归一化 | [第 16 章 可验证奖励强化学习（RLVR）：从 PPO 到 GRPO，再到 R1、Kimi 与 Qwen](./chapters/ch16-rlvr) | <a href="./demos/ch16-length-bias.html" target="_blank">打开 ↗</a> |
| CLIP 与 SigLIP：图文相似度矩阵的对比学习 | [第 17 章 多模态模型：让语言模型看见图像](./chapters/ch17-alignment-multimodality) | <a href="./demos/ch17-clip-matrix.html" target="_blank">打开 ↗</a> |
| 图像如何变成 token：ViT 切 patch、中心裁剪、AnyRes 与动态分辨率 | [第 17 章 多模态模型：让语言模型看见图像](./chapters/ch17-alignment-multimodality) | <a href="./demos/ch17-vit-patches.html" target="_blank">打开 ↗</a> |
| 循环模型：谱半径决定会不会爆炸 | [第 18 章 嘉宾讲座：从推理系统看全栈创新（Dan Fu）](./chapters/ch18-guest-dan-fu) | <a href="./demos/ch18-looped-stability.html" target="_blank">打开 ↗</a> |
| 逐算子 kernel vs 巨型 kernel：SM 时间线 | [第 18 章 嘉宾讲座：从推理系统看全栈创新（Dan Fu）](./chapters/ch18-guest-dan-fu) | <a href="./demos/ch18-megakernel.html" target="_blank">打开 ↗</a> |
