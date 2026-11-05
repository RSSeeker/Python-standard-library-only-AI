# 路线图：从「组件补齐」到「范式覆盖」

> 依据：2026-11-05 的十七节功能清单（网络层 / 损失 / 优化器 / 调度 / 正则化 / 初始化 /
> 数据管道 / 分词 / 指标 / 可解释性 / 推理部署 / 生成模型 / 自监督与 RL / 训练基础设施 /
> 测试 / 领域应用 / 研究前沿）。
>
> 现状：`AI.py` 13121 行、22 个分节、677 项测试全绿、74 个公开 API；`AI_C` 与网页版
> 已同步到「LLM 主线」（因果 Transformer + 采样 + 存档 + 交叉语言逐位对拍）。

---

## 1. 规划口径

1. **优先级**沿用清单里的 P0→P3；清单里的「轮次」列不参与排期（它表示可以落在哪几轮，
   不是先后要求）。
2. **按族归并**：同一族的条目合并成一个实现单元，避免一个参数一个提交。例如
   - `GeGLU / ReGLU / Bilinear GLU` = 给 `SwiGLU` 加一个 `gate` 参数（4 种模式）
   - `CosineAnnealingWarmRestarts / OneCycle / Noam / WSD / Cyclic / Polynomial / …` = 一个调度器族
   - `LeCun / Orthogonal / Sparse / TruncatedNormal` = 一个 `init_*` 族
   - `NDCG / MAP / MRR / P@k / R@k / HR / Coverage` = 一个排序指标族
   - 归并后约 **240 个实现单元**（原表约 400 行）。
3. **每个单元的验收标准**（缺一不可）：
   - 前向形状/数值与手算或参考实现一致；
   - 反向：解析梯度 vs 中心差分，相对误差 ≤ 1e-8（小张量）或 ≤ 1e-6（深链）；
   - **默认路径逐位不变**：所有新特性默认关闭，`probe_default_paths.py` 必须仍是
     `FAILS: none`（这是与 C 引擎逐位对拍的前提）；
   - 测试：每族 1 个测试类、3–6 个用例；收集数变化要同步 README 两处计数；
   - README：组件表一行 + 需要时加小节；
   - 一个单元或一族一个提交，提交信息写清「验证数字」。
4. **C / Web 同步**默认不做。C 侧只维持「LLM 主线 + 已对拍组件」的逐位一致；新特性
   属于 Python 侧扩展，需要时再单独排期（否则每轮都要改三份，无法收敛）。
5. 每波结束跑一次：`pytest`、`python AI.py`（14 段 demo）、`probe_default_paths.py`。

---

## 2. 明确不做的（写进文档当「已知边界」）

| 条目 | 原因 |
|---|---|
| 张量并行 / 流水并行 / 数据并行 | 单进程纯 Python，做出来只是概念演示，没有加速 |
| Flash Attention 的真实收益 | 无 CUDA kernel；只实现分块算法做教学，标注「不更快」 |
| 真 float16 / BF16 模拟 | Python float 仍是 double，除非全程 struct 打包，收益为负 |
| 多 worker 数据加载 | `multiprocessing` 在 Windows 上的启动开销 + 序列化成本大于收益 |
| Paged Attention 的性能意义 | 分页是为了显存碎片，纯 Python 无显存管理；只做数据结构演示 |
| SparseAdam / SM3 | 稀疏梯度在纯 Python 的稠密 list 表示下没有收益 |
| FID / BERTScore / LPIPS 的「原版」 | 依赖 Inception / BERT 权重；给简化版并明确标注口径 |
| QLoRA 的 4-bit 基座 | 无 int4 存储与反量化 kernel；只做「量化 + LoRA」的接口与等效性验证 |

**已具备、只需接线或标注**：MQA（`num_kv_heads=1` 就是）、LoRA、Mixup/CutMix/Cutout
（函数已有，缺 `fit` 入口）、SmoothL1（`huber_loss` 就是）、非结构化剪枝
（`prune_magnitude` 已有）、滑动窗口、KV cache。

---

## 3. 波次总表

| 波次 | 主题 | 覆盖清单 | 关键产出 | 依赖 |
|---|---|---|---|---|
| W0 | 训练基础设施 | §14 P0 + §5 Mixup 接线 | `set_seed` / Callback / `resume_from` / `infer_shapes` / `predict_batch` / 逐层梯度范数 / 激活统计 / 死单元检测 | — |
| W1 | 损失：回归与对齐 | §2 P0 | L1 / LogCosh / Quantile / DPO·IPO·KTO·ORPO·SimPO·CPO / KD 特征对齐 / SoftCE·OHEM·CB·LDAM / 多任务权重 | W0 |
| W2 | 卷积与视觉块 | §1 P0/P1 | DepthwiseSeparable / ResNet Basic·Bottleneck / DenseNet / SE·ECA·CBAM / ConvNeXt / Conv3d·Transpose1d·Transpose3d / 池化家族 / AdaIN·FiLM | W0 |
| W3 | 归一化·激活·初始化 | §1 P1 + §6 | WeightNorm·SpectralNorm·SwitchNorm·FRN·ScaleNorm / PReLU·Maxout·DynamicReLU·FReLU·Snake / GLU 家族 / 初始化族 / 残差零初始化 | W2 |
| W4 | 视觉骨干 | §1 P0/P1 | PatchEmbedding + ViT / U-Net / FPN / SegNet / Swin·MLP-Mixer（简化）/ AttentionPooling·GeM | W2, W3 |
| W5 | 注意力与位置编码 | §1 P1/P2 | MLA / 可学习位置编码·T5·xPos·YaRN·NTK-RoPE / Linear Attention·Performer·Linformer·BigBird·Longformer·Reformer / FlashAttention 分块 | W3 |
| W6 | 图神经网络 | §1 P1/P2 | GCN·GAT·GIN·GraphSAGE·MPNN·GCNII·APPNP·JK-Net / 图池化 TopK·DiffPool·SAGPool | W3 |
| W7 | LLM 微调与推理 | §1 P0/P1 + §11 P1 | Prefix·P-tuning·Prompt Tuning / QLoRA·DoRA·AdaLoRA / 约束解码 / 采样扩展 / 投机解码 / 序列打包 / Prompt cache / 连续批处理 | W5 |
| W8 | 生成模型 | §12 | DDIM·v-pred·x0-pred·Min-SNR / VQ-VAE / Flow Matching / GAN 变体族 / Score-SDE·Consistency·Flows·PixelCNN·Latent Diffusion·EBM | W2, W4 |
| W9 | 优化器与调度 | §3 + §4 | Nesterov·SGDW / Lookahead·SWA·EMA / AdaGrad·AdaDelta·Nadam·RAdam·Ranger·LAMB·LARS / Prodigy·Sophia·Shampoo·CAME·Adan·L-BFGS / 调度器族 10+ 个 | W0 |
| W10 | 正则与训练技巧 | §5 | Dropout 变体 / DropBlock·DropConnect / 对抗训练 FGSM·PGD·TRADES·MART·FreeLB·SMART / 课程学习 / 伪标签·FixMatch·MixMatch·MeanTeacher / 梯度检查点 | W0, W1 |
| W11 | 数据管道 | §7 | Dataset 抽象 / Sampler 体系 / bucketing / 编码器族 / 增强管道 Compose / GroupKFold·时序切分 / Scaler 族 / Imputer / TargetEncoder / PCA·t-SNE / SMOTE | W0 |
| W12 | 分词与批处理 | §8 | 特殊 token 统一 / padding·truncation·mask / SentencePiece Unigram / Word-level·Whitespace·Regex | W7 |
| W13 | 指标扩展 | §9 | 回归·效率·分类·排序·分割·文本·语音·聚类·校准·检测·图像·一致性·公平性 共 12 族 | — |
| W14 | 可解释性 | §10 | Saliency·Grad×Input·SmoothGrad·IG·DeepLIFT / Attention Rollout·Flow / Permutation / feature_maps_ascii / GradCAM 族 / SHAP·LIME 简化 / TCAV | W2, W4 |
| W15 | 推理与部署 | §11 | 结构化剪枝 / 特征蒸馏 / QAT / 集成 Voting·Stacking·Bagging·Boosting / 投机解码 / 序列打包 / PTQ 族（教学） | W7, W9 |
| W16 | 自监督·多模态·RL | §13 | SimCLR·MoCo·BYOL·Barlow·VICReg·SwAV·DINO / MAE·BEiT·SimMIM / CLIP / DQN 族·PPO·SAC·TD3 / RLHF·奖励模型 / 元学习·联邦 | W4, W11 |
| W17 | 领域应用 | §16 | 时序 TCN·Informer·PatchTST / 推荐 MF·NCF·Wide&Deep·DeepFM / 风格迁移·超分·去噪 / 向量检索 / 语音·3D·视频·OCR·检测（简化） | W4, W8 |
| W18 | 测试与数值稳定 | §15 | grad_check 自动遍历 / 稳定性套件（1e300·NaN·Inf·全零·空） / 回归基线快照 JSON / 集成测试 / doctest | 贯穿 |
| W19 | 工程化收尾 | §14 剩余 | 实验追踪 JSON / 模型元数据 / CLI 子命令 / YAML·TOML / logging / 内存监控 / 模型注册表 | W0 |
| W20 | 研究前沿 | §17 | DP-SGD / 成员推断 / Lottery Ticket / MoE-路由动态·Early Exit / 神经符号（文档 + 最小演示） | W15 |

---

## 4. 逐波清单

### W0 训练基础设施（P0，最先做，杠杆最大）
- [ ] `set_seed(seed, deterministic=True)`：一次性固定 `random` / 采样器 / 初始化，并提供
      「进入-退出」上下文，避免污染调用方
- [ ] `TrainingCallback` 基类（`on_train_begin / on_epoch_begin / on_epoch_end /
      on_batch_end / on_train_end`）+ `Model.fit(callbacks=[...])`
- [ ] 内置 `EarlyStopping` / `ModelCheckpoint` / `LoggingCallback`（接 `logging`）/ `HistoryCallback`
- [ ] `fit(resume_from=...)`：恢复 epoch、history、最佳指标、调度器与优化器状态（现在
      `load(restore_optimizer=True)` 只恢复参数，epoch 仍从 1 开始）
- [ ] `infer_shapes(model, input_shape)`：真跑一次前向拿每层形状（区别于 `estimate_flops` 的解析估算）
- [ ] `predict_batch(model, xs, batch_size=32)`：批推理
- [ ] 逐层梯度范数进 `history['layer_grad_norm']`
- [ ] `activation_stats(model, x)` / `dead_unit_report(model, x)`：用已有 hook 统计逐层均值/方差/死单元
- [ ] `fit(train_augment=...)`：把 mixup / cutmix / cutout 接进训练循环

### W1 损失：回归与偏好对齐（P0）
- [ ] 回归族：`l1_loss`、`log_cosh_loss`、`quantile_loss`（= pinball）、`smooth_l1`（标注 `huber_loss` 即同类）
- [ ] 偏好对齐族：`preference_loss(chosen_logp, rejected_logp, ref_..., method=)` 支持
      DPO / IPO / KTO / ORPO / SimPO / CPO，配「参考模型冻结」的等效性验证
- [ ] 蒸馏：`distill_loss(student_logits, teacher_logits, T)` + 隐藏层对齐（`hint_layers`）
- [ ] 类别不平衡族：`soft_ce`、`ohem_loss`、`class_balanced_loss`、`ldam_loss`
- [ ] 多任务权重：`uncertainty_weighting`、`gradnorm`、`pcgrad`
- [ ] 已有项标注：`huber_loss`（SmoothL1）、`kl_divergence`、`dice_loss`、`ctc_loss`、`triplet_loss`、`info_nce_loss`

### W2 卷积与视觉块（P0/P1）
- [ ] `DepthwiseSeparableConv`（= `Conv2d(groups=C)` + 1×1）
- [ ] `ResNetBasicBlock` / `ResNetBottleneck`（1×1→3×3→1×1 + 残差）
- [ ] `DenseBlock` / 稠密连接
- [ ] `SEBlock` / `ECABlock` / `CBAMBlock`（通道 + 空间注意力）
- [ ] `ConvNeXtBlock`（7×7 depthwise + LayerNorm + 1×1 升/降维）
- [ ] 卷积家族：`Conv3d`、`ConvTranspose1d`、`ConvTranspose3d`
- [ ] 池化家族：`MaxPool1d/3d`、`AvgPool1d/3d`、`LPPool2d`、`AdaptiveMaxPool2d`、`GlobalMaxPool2d`、
      `FractionalMaxPool2d`、`StochasticPool2d`
- [ ] `AdaIN` / `FiLM` / `ConditionalNorm`

### W3 归一化·激活·初始化（P1 + §6）
- [ ] 归一化族：`WeightNorm`、`SpectralNorm`（幂迭代）、`SwitchNorm`、`FilterResponseNorm`、`ScaleNorm`
- [ ] 激活族：`PReLU`（可学习斜率）、`Maxout`、`DynamicReLU`、`FReLU`、`Snake`
- [ ] GLU 族：`SwiGLU` 加 `gate="swish|gelu|relu|bilinear"`（GeGLU / ReGLU / Bilinear GLU）
- [ ] 初始化族：`init_lecun`、`init_orthogonal`、`init_sparse`、`init_trunc_normal`、零初始化 bias、
      残差分支零初始化（`zero_init_residual`）
- [ ] 数据驱动初始化：LSUV、Fixup/T-Fixup、Mimic Init（NET2NET）
- [ ] 残差变体：Pre-LN / Post-LN / ReZero / SkipInit / Highway

### W4 视觉骨干（P0/P1）
- [ ] `PatchEmbedding`（= `Conv2d(in=3, out=d_model, kernel=p, stride=p)`）+ `ViT`（含 CLS token、位置编码、分类头）
- [ ] `UNet`（编码-解码 + 跳跃连接）、`FPN`、`SegNet`（简化）
- [ ] `SwinTransformer`（窗口注意力，简化）、`MLPMixer`
- [ ] `AttentionPooling` / `GeMPooling`

### W5 注意力与位置编码（P1/P2）
- [ ] `MLA`（低秩 KV 压缩）；`MQA` 标注为 `num_kv_heads=1` 特例
- [ ] 位置编码：可学习位置编码、T5 相对位置偏置、xPos、YaRN / NTK-aware RoPE（改 `rope_freqs`）
- [ ] 线性注意力族：`LinearAttention`、`Performer`（随机特征）、`Linformer`、`BigBird`（稀疏）、
      `Longformer`（滑窗+全局）、`Reformer`（LSH）
- [ ] `FlashAttention` 分块算法（模拟，明确「不更快」）

### W6 图神经网络（§1 全新范式）
- [ ] `GCNLayer` / `GATLayer` / `GINLayer` / `GraphSAGELayer` / `MPNNLayer` / `GCNIILayer` / `APPNPLayer` / `JKNet`
- [ ] 图池化：`TopKPool`、`DiffPool`（简化）、`SAGPool`

### W7 LLM 微调与推理（P0/P1）
- [ ] 提示微调：`PrefixTuning` / `P_Tuning` / `PromptTuning`（可学习前缀拼在 Embedding 前）
- [ ] 参数高效：`QLoRA`（量化基座 + LoRA 接口）/ `DoRA`（幅度+方向分解）/ `AdaLoRA`（重要性分配秩）
- [ ] 采样扩展：typical / mirostat / eta / top-a / tail-free / contrastive search / diverse beam
- [ ] 约束解码：JSON Schema / 正则 / CFG（`_sample` 接 mask 回调）
- [ ] 解码加速：投机解码（草稿模型 + 验证）、自投机/Medusa/EAGLE（概念）、连续批处理、`PagedAttention`（数据结构演示）
- [ ] `sequence_packing`（segment_ids，复用 `pad_mask`）、`PromptCache` / `PrefixCache` 持久化

### W8 生成模型（§12）
- [ ] 扩散：`DDIM`、`v-prediction` / `x0-prediction` / `Min-SNR` 加权（DDPM 已有）
- [ ] `VQVAE`（码本 + straight-through）
- [ ] `FlowMatching` / `RectifiedFlow`
- [ ] GAN 变体：DCGAN / WGAN / WGAN-GP / LSGAN / Hinge / R1-R2 / StyleGAN（简化）/ CycleGAN（简化）/ Pix2Pix / Progressive
- [ ] 其它：Score-based（NCSN）/ SDE、Consistency Models、RealNVP·Glow、PixelCNN·WaveNet（概念）、Latent Diffusion（概念）、EBM

### W9 优化器与调度（§3/§4）
- [ ] `SGDSGDW`（解耦 wd）/ Nesterov / `Lookahead`（外挂任意优化器）/ `SWA` / `EMA`
- [ ] 自适应族：AdaGrad / AdaDelta / Nadam / RAdam / Ranger / LAMB / LARS
- [ ] 进阶：Prodigy（无 lr）/ Sophia / Shampoo / CAME / Adan / L-BFGS（小模型）
- [ ] 调度器族：CosineAnnealingWarmRestarts / OneCycle / InverseSqrt / Noam / CyclicLR（triangular·triangular2·exp_range）/
      Polynomial / Lambda / MultiStep / Multiplicative / WSD / ChainedScheduler / SequentialLR
- [ ] 标注不做：SparseAdam / SM3

### W10 正则与训练技巧（§5）
- [ ] Dropout 变体：SpatialDropout / AlphaDropout / GaussianDropout / R-Drop
- [ ] 结构化：DropBlock / DropConnect
- [ ] 对抗训练：FGSM / PGD / TRADES / MART / FreeLB / SMART
- [ ] 噪声注入（权重/梯度/输入）、课程学习 `CurriculumSampler`、Manifold Mixup、Snapshot Ensembles
- [ ] 半监督：VAT、伪标签、FixMatch、MixMatch、MeanTeacher、一致性正则
- [ ] 梯度检查点（activation checkpointing）、混合精度模拟（标注无真收益）

### W11 数据管道（§7）
- [ ] `Dataset` 抽象 + 流式/懒加载；`Sampler` 体系（Weighted / Stratified / Curriculum / Subset / BatchSampler）
- [ ] 长度分桶 `bucket_by_length`；预处理缓存（文件哈希命中）
- [ ] 编码器：`OneHotEncoder` / `LabelEncoder` / `OrdinalEncoder` / `TargetEncoder` / `FrequencyEncoder` / `WOEEncoder`
- [ ] 图像增强管道：`Compose` / `RandomCrop` / `RandomResizedCrop` / `RandomRotation` / `ColorJitter` / `GaussianBlur` / `Normalize`
- [ ] 切分扩展：`GroupKFold` / 时间序列切分
- [ ] Scaler 族：Robust / MaxAbs / QuantileTransformer / PowerTransformer
- [ ] 缺失值：`SimpleImputer` / `KNNImputer` / `IterativeImputer`
- [ ] 降维：`PCA` / `t-SNE`（简化）；`SMOTE` / `ADASYN`；特征选择（VarianceThreshold / SelectKBest / RFE / L1）
- [ ] 文本增强（同义词替换 / EDA / Token Dropout）；音频增强（概念）

### W12 分词与批处理（§8）
- [ ] 特殊 token 统一管理（BOS/EOS/PAD/UNK/SEP/CLS/MASK）
- [ ] 批处理：`pad_sequences` / `truncate` / `attention_mask` / `tokenize_batch`
- [ ] `SentencePieceTokenizer`（Unigram LM 简化版）；`WordTokenizer` / `WhitespaceTokenizer` / `RegexTokenizer`

### W13 指标扩展（§9）
- [ ] 回归：MedAE / MSLE / RMSLE / ExplainedVariance / AdjustedR² / Quantile
- [ ] 效率：`inference_benchmark`（时间 / 吞吐 / 峰值内存）
- [ ] 分类：BalancedAccuracy / CohenKappa / Brier / LogLoss / PR-AUC / Specificity / Sensitivity / F-beta / Hamming / Jaccard
- [ ] 排序：NDCG@k / MAP@k / MRR / P@k / R@k / HitRate / Coverage
- [ ] 分割：mIoU / Dice / HD95 / BoundaryF1 / PixelAcc / FW-IoU
- [ ] 文本：语料级 BLEU / ROUGE-1/2 / METEOR（简化）/ CIDEr（简化）/ Self-BLEU / Perplexity
- [ ] 语音：WER / CER / SER（编辑距离）
- [ ] 聚类：Silhouette / Calinski-Harabasz / Davies-Bouldin / ARI / NMI / AMI
- [ ] 校准：ECE / MCE / ReliabilityDiagram
- [ ] 检测：mAP / mAP@0.5 / mAP@[.5:.95] / AR（简化）
- [ ] 图像：PSNR / SSIM / MS-SSIM；FID / KID / IS（简化并标注口径）
- [ ] 一致性：Fleiss Kappa / Krippendorff Alpha；公平性：Demographic Parity / Equal Opportunity / Disparate Impact

### W14 可解释性（§10）
- [ ] 归因：Saliency / Gradient×Input / SmoothGrad / IntegratedGradients / DeepLIFT（简化）
- [ ] 注意力：Attention Rollout / Attention Flow（在 `attention_maps` 上做矩阵连乘）
- [ ] `permutation_importance`；`feature_maps_ascii`（`attention_ascii` 扩展）
- [ ] CNN：GradCAM / GradCAM++ / ScoreCAM / LayerCAM
- [ ] 嵌入可视化（复用 t-SNE）；SHAP（KernelSHAP 简化）/ LIME（简化）/ Anchors
- [ ] TCAV；Activation Maximization / Network Dissection；Counterfactual / CEM；Prototypes / Influence（简化）

### W15 推理与部署（§11）
- [ ] 结构化剪枝：通道 / 头 / 层（现有 `prune_magnitude` 是非结构化）
- [ ] 特征蒸馏（hint layer）；QAT（量化感知训练）；PTQ / GPTQ / AWQ / SmoothQuant（教学，标注 int8 未真存）
- [ ] 集成：Voting / Stacking / Bagging / Boosting
- [ ] 推理优化：投机解码（已在 W7 实现，此处做集成）、序列打包、连续批处理、并行概念（文档）

### W16 自监督·多模态·RL（§13）
- [ ] 对比学习训练器：SimCLR / MoCo（队列）/ BYOL / BarlowTwins / VICReg / SwAV / DINO
- [ ] 掩码建模：MAE / BeiT / SimMIM（视觉侧）
- [ ] 多模态：CLIP（双塔 + InfoNCE）、跨模态融合（CrossAttention 扩展）
- [ ] 预测式自监督：Rotation / Jigsaw
- [ ] RL 环境：`GridWorld` / `CartPole`（内置）
- [ ] RL 算法：DQN / Double / Dueling / Rainbow（简化）/ REINFORCE / A2C / PPO / SAC / TD3 / DDPG
- [ ] RL 基础设施：ReplayBuffer / ε-greedy / UCB / Thompson；RLHF（PPO-based）/ 奖励模型
- [ ] 元学习：MAML / Reptile / ProtoNet / Matching Networks
- [ ] 联邦学习：FedAvg / FedProx / FedBN

### W17 领域应用（§16）
- [ ] 时序：TCN / Informer（简化）/ Autoformer（简化）/ PatchTST
- [ ] 推荐：MF / NCF / Wide&Deep / DeepFM / DIN
- [ ] 图像：风格迁移（NST）/ 超分（SRCNN / SRGAN）/ 去噪（DnCNN）
- [ ] 检索：Faiss 式暴力 + HNSW（简化）/ IVF（概念）
- [ ] 语音 Conformer（简化）/ Mel 滤波器；3D PointNet（简化）；视频 3D-CNN / Two-Stream；OCR CRNN；检测 YOLO（简化）；深度/姿态估计（概念）

### W18 测试与数值稳定性（§15，贯穿各波）
- [ ] `grad_check_all(model, ...)`：自动遍历所有带参数的子模块
- [ ] 数值稳定性套件：1e300 / 1e-300 / NaN / Inf / 全零 / 空序列 不静默出 NaN
- [ ] 回归基线快照：固定 seed 下 forward/backward 输出存 JSON，CI 比对
- [ ] 集成测试（toy 数据集训练到达标）；doctest；属性测试（随机输入验证不变量）

### W19 工程化收尾（§14 剩余）
- [ ] 实验追踪（JSON：配置/指标/曲线/耗时）；模型元数据（created_at/git_hash/config/metrics）
- [ ] CLI 子命令：train / predict / eval / export / finetune
- [ ] 配置支持 YAML/TOML；`logging` 集成；内存监控（tracemalloc）；模型注册表/版本管理

### W20 研究前沿（§17，文档 + 最小演示）
- [ ] DP-SGD（梯度裁剪 + 高斯噪声 + 隐私预算记账）
- [ ] 成员推断攻击与防御（最简版）
- [ ] Lottery Ticket（迭代剪枝 + 重初始化）
- [ ] Mixture-of-Depths / Early Exit
- [ ] 神经符号（规则约束的候选过滤）
- [ ] 文档化：NTK / 双下降 / grokking 作为「已知边界」

---

## 5. 节奏与里程碑

| 里程碑 | 内容 | 累计测试（估） |
|---|---|---|
| M1 | W0 + W1 + W9（基础设施、损失、优化器调度） | 677 → ~740 |
| M2 | W2 + W3 + W4（视觉骨干可用） | ~800 |
| M3 | W5 + W7（注意力扩展 + LLM 微调推理） | ~860 |
| M4 | W8 + W10 + W11（生成模型 + 正则 + 数据管道） | ~930 |
| M5 | W12 + W13 + W14（文本 + 指标 + 可解释性） | ~1000 |
| M6 | W15 + W16 + W17（部署 + 自监督/RL + 领域） | ~1080 |
| M7 | W18 + W19 + W20（稳定性 + 工程化 + 前沿） | ~1120 |

按过去的实测节奏（一次会话 5–15 个单元），240 个单元约需 **25–35 个会话**。
每个会话结束都保证：测试全绿、demo exit 0、`probe_default_paths.py` 无差异、工作区干净。

---

## 6. 风险与对策

1. **范围爆炸**：每波开始时先把该波的单元数写进清单，超出就拆波，不硬塞。
2. **默认路径漂移**：任何改动都要跑 `probe_default_paths.py`；它比对的是上一个提交的
   `AI.py`（`git show HEAD:AI.py`），一旦出现非零差异就回退。
3. **测试变慢**：新测试一律小规模（≤200 步、d_model ≤ 32），慢测试标 `@pytest.mark.slow`。
4. **README 计数漂移**：已在提交前固定检查（正文 + 文件结构树两处）。
5. **C/Web 失同步**：本路线图明确 Python 优先；C 侧只保留「LLM 主线逐位一致」，
   新增特性不同步，除非单独要求。
