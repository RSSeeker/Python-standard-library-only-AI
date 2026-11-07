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
4. **C / Web 同步**：能同步的尽量同步，但按「可逐位对拍」分级（见第 7 节）。判据很简单
   —— 纯标量数学（激活、损失、优化器、调度器、指标、初始化）默认进 A 档必同步；
   需要新张量布局或新子系统的进 B 档（排进同波的后半段）；只跟宿主环境有关的进 C 档
   不同步。Python 永远先做完并验证，C/Web 再跟。
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

**状态：已完成（提交 `W0` 批次）**。偏差见每条的括注——都是有意收窄，不是遗漏。

- [x] `set_seed(seed)`：固定全局 `random`（只做这一件事；`fit(seed=)` 仍独立控制打乱，
      两者分工写在 docstring 里）
- [x] `Callback` 基类（`on_train_begin / on_epoch_begin / on_epoch_end / on_train_end`）
      + `Model.fit(callbacks=[...])`；钩子抛异常会中断训练（不吞异常）
      （**未做** `on_batch_end`：batch 级钩子会进最内层循环，留到确有需求时再加）
- [x] 内置 `EarlyStopping` / `ModelCheckpoint` / `LoggingCallback` / `HistoryCallback`；
      `EarlyStopping` 只发停止请求（`should_stop`），回滚最优权重仍用
      `fit(restore_best_weights=True)`
- [x] `fit(resume_from=...)`：恢复权重 + 优化器状态（含 `step_count`），`epochs=0` 即「只恢复」
      （**未做** epoch/history/调度器续接：现在的 optimizer `step_count` 足以保证逐位续训，
      history 拼接属展示层，等有真实需求再补）
- [x] `infer_shapes(model, x)`：真跑一次前向拿每层形状（入参是样本而不是 shape，
      省得再猜 dtype/布局）
- [x] `predict_batch(model, xs, batch_size=64)`：与 `predict` 逐位一致
- [x] 逐层梯度范数进 `history['layer_grad_norm']`（`record_layer_norms=True`，只记录不改训练）
- [x] `activation_stats(model, x)` / `dead_units(model, x)`：逐层 mean/std/min/max/zero_ratio
      与死单元下标、比例
- [x] `fit(train_augment=fn(xs, ys) -> (xs, ys))`：批级回调，**在 Mixup / PGD 之前**执行；
      样本数必须不变，默认 None 时数值与旧版逐位相同；探针证明「增强 + mixup」与「手工预处理 + mixup」逐位一致，Cutout/CutMix 由此可按正确顺序接入（CutMix 需自定义回调）

### W1 损失：回归与偏好对齐（P0）
- [x] 回归族：`l1_loss`、`log_cosh_loss`、`quantile_loss`（= pinball）、`smooth_l1`（= `huber_loss` 别名，逐位等价）（`c37cdf5`；梯度差分 ≤2.6e-10，`quantile(0.5)` 逐位等于 `l1/2`）
- [x] 偏好对齐族：`preference_loss(chosen_logp, rejected_logp, ref_..., method=)` 支持
      DPO / IPO / KTO / ORPO / SimPO / CPO + 单边 `kto_loss(logp, ref_logp, desirable=)`
      （W1-D 提交；五种 method 闭式解与双向中心差分吻合 ≤1e-8，`β=50` 极端情形 loss < 1e-12 不溢出；
      ORPO 在 logit 空间做、`p>=1` 报错；**不进 `_LOSS_MAP`**——吃一对 logp 而非逐样本 `(pred, target)`）
- [x] 蒸馏：`distill_loss(student_logits, teacher_logits, T=2.0)`（`T²·KL`，对学生 logits 的梯度 `T·(p_s−p_t)`）+ `hint_loss(student_feat, teacher_feat, weight)`（隐藏层均方误差）；`_LOSS_MAP` 增 `'distill'`、输出层要求 `linear`（梯度差分 ≤9.4e-10，`T=1` 时与手算 KL 逐位一致）
- [x] 类别不平衡族：`ohem_loss`（批级，不进 `Model`）、`class_balanced_loss`、`ldam_loss`；`soft_ce` **不再单独实现**——`cross_entropy_loss` 已支持任意软标签与 `label_smoothing`，再写一个只是重复（W1-B 提交；CB/LDAM 梯度差分 ≤3.5e-9）
- [x] 多任务权重：`uncertainty_weighting`（Kendall，`d_log_vars = 1 - exp(-s_i)·L_i`）、`gradnorm_weights`（`target_i = Ḡ·r_i`，权重归一化均值 1；只有 `r_i ≡ 1` 时才有不动点，一般情形用返回的 `gradients = sign(G-target)·G0/r`）、`pcgrad`（两两冲突投影，嵌套 list 支持）；W1-E 提交，双向差分 worst_rel 1.54e-10，11 类非法输入全报错
- [x] 已有项标注：`huber_loss`（SmoothL1）、`kl_divergence`、`dice_loss`、`ctc_loss`、`triplet_loss`、`info_nce_loss` —— README 损失表（约 725–734 行）逐条给了公式与适用场景，`smooth_l1_loss` 是 huber 的逐位等价别名

### W2 卷积与视觉块（P0/P1）
- [x] `DepthwiseSeparableConv2d`（W2-B：`Conv2d(groups=C)` + 1×1 点卷积，参数量 `C·k·k + C + C·out + out`；dx 中心差分 ≤1e-8，`sgd_step`/`adam_step` 真改权重）
- [x] `BasicBlock2d`（W2-B：`conv3x3 → BN → ReLU → conv3x3 → BN → (+x) → ReLU`，stride=1 恒等捷径；全零卷积时退化为 `relu(x)`，eval 模式 dx 中心差分 ≤1e-7）
- [x] `Bottleneck2d`（W2-C：`1x1 降维(mid=out//4) → 3x3 → 1x1 升维` + 残差；stride≠1 或通道变化时用 `1x1 Conv + BN` 投影捷径，BN 梯度收在 `bn_grads`，eval 模式 dx 中心差分 ≤1e-7）
- [x] `DenseBlock` / 稠密连接（`BN → ReLU → 3x3 Conv` 逐层拼接，`(Cin,H,W) → (Cin+growth*num_layers,H,W)`；`dx`/conv `grad_w`/BN `gamma,beta` 中心差分 ≤1e-7，eval 走 running 统计量）
- [x] `SEBlock`（W2-A 提交：GAP → 瓶颈 MLP → sigmoid 乘回输入，dx 与四组参数梯度中心差分 ≤1e-9）
- [x] `ECABlock`（W2-B：GAP 后一条奇数宽 1D 卷积跨通道混权重，只多 `kernel_size` 个参数；dx/`grad_w` 中心差分 ≤1e-8）
- [x] `CBAMBlock`（通道注意力 + 7×7 空间注意力，比 ECA 多个空间分支；空间反卷积下标写反过，`dx` 中心差分 1.187e-01 → 修后 7.2e-11；顺带给 `SEBlock`/`CBAMBlock` 补了 `sgd_step`/`adam_step`）
- [x] `ConvNeXtBlock`（7×7 depthwise + 逐位置跨通道 LayerNorm + 1×1 升/降维 + GELU + 残差；`xhat` 方差 `var/(var+eps)`、`pw2` 全零时输出等于输入、`dx`/LN 参数/depthwise 权重中心差分 ≤1e-7）
- [x] `ConvTranspose1d`（W2-D）：`(Cin,L) -> (Cout,(L-1)*s-2p+k+op)`，前向把每个输入点按 stride 摆到输出网格累加，`dx` 与 `grad_w` 中心差分 ≤1e-8
- [x] `Conv3d` / `ConvTranspose3d`（W2-E）：三轴独立 `kernel/stride/padding`，`(D-1)*s-2p+k+op` 的输出尺寸，`dx` 与 `grad_w` 中心差分 ≤1e-8，中心 tap 退化为恒等映射、单点散射位置逐个校验
- [x] 池化家族（部分，W2-A 提交）：`GlobalMaxPool2d`（只回传最大值位置）、`AdaptiveMaxPool2d`（分箱与
      `AdaptiveAvgPool2d` 一致，`(1,1)` 与全局最大逐位相同）、`LPPool2d`（`p=1` 在非负输入上等于
      `AvgPool2d`，随 p 增大单调逼近 `MaxPool2d`，梯度差分 ≤1e-9）
- [x] 池化家族（1D / 3D，W2-D）：`MaxPool1d` / `AvgPool1d` / `MaxPool3d` / `AvgPool3d`（floor 语义丢弃尾窗、最大池化只回传最大值位置、平均池化均分、padding 不接收梯度）
- [x] 池化家族（2D 特殊，W2-H）：`FractionalMaxPool2d`（比例重叠窗口、起点随机可注入 `rng`，alpha 为整数时窗口与 seed 无关）、`StochasticPool2d`（训练按 `relu(x)/Σrelu` 采样并乘 `p`，`eval()` 取概率加权期望；非正窗口退化为均匀）；梯度总量守恒 32/非零位置 32、反向 `d*p` 差分 ≤1e-12
- [x] `AdaIN`（W2-C：按通道实例归一化后乘风格向量，`set_style()`；`d_gamma`/`d_beta`/`dx` 中心差分 ≤1e-6，单向量输入 mean=x ⇒ `dx=0`）
- [x] `FiLM`（W2-C：`y = gamma ⊙ x + beta`，`groups>1` 时特征分段共享调制参数）
- [x] `ConditionalNorm`（W2-H：`gamma = 1 + W_g·c + b_g`、`beta = W_b·c + b_b`，`set_cond()` 设条件；`dx`/`b_g,b_b`/`W_g,W_b`/`grad_cond` 五组中心差分 ≤1e-7）

### W3 归一化·激活·初始化（P1 + §6）
- [x] 归一化族（W3-A）：`ScaleNorm`（只有标量 g）、`WeightNorm`（`w = g⊙v/‖v‖`，反向含解耦项，`dv`/`dg`/`db` 中心差分 ≤1e-7）、`SpectralNorm`（幂迭代估计 σ，默认 σ 视为常量，`freeze_sigma()` 后 `dv` 与中心差分一致）
- [x] 归一化族（W3-D）：`SwitchNorm`（instance/batch/layer 三种统计量按 `softmax(logits)` 软加权，
      本实现 instance 与 batch 同源；`dx` 与 `gamma`/`beta`/`logits` 四组中心差分 ≤1e-7）、
      `FRN`（跨通道 ν²，batch=1 可用；TLU 截断位置的 `dout` 只流向 `tau`）
      —— 两处推导坑：`dx` 的 layer 项必须对**全部通道**求和（只算本通道时误差 1.503），
      而 `grad_logits` 的 `S`/`Sx` 只需**本通道**（用全局量时 `dL/dw2` 差 2.03×）。
- [x] 激活族（W3-B）：`PReLU`（逐特征可学习斜率，负分支才累计 `da`）、`Snake`（`y = x + sin²(ax)/a`，`a→0` 走极限分支；`dx`/`da` 中心差分 ≤1e-7）
- [x] 激活族（W3-E）：`Maxout`（k 条仿射支路取最大，argmax 路由，`dx`/`grad_a` 中心差分 ≤7.1e-11）、
      `DynamicReLU`（斜率由 `mean(x)` 线性预测；`dx` 的耦合项是 `(1/d)Σ_{x_i≤0} dout_i·x_i·w_i`，
      对**所有**特征都非零 —— 我第一版把它写成 `w_j` 且只加到负分支，差分误差 1.017）、
      `FReLU`（`y = max(x, T(x))`，`T` 是逐通道 k×k 深度卷积；零初始化严格等于 ReLU）
- [x] GLU 族（W3-B）：`SwiGLU(gate="swish|silu|gelu|relu|bilinear")` → SwiGLU / GeGLU / ReGLU / Bilinear GLU；默认 `swish` 与原实现逐位一致，各 gate 的 `grad_Wg` 中心差分 ≤1e-6
- [x] 初始化族：`init_lecun_normal`/`init_lecun_uniform`、`init_orthogonal`、`init_sparse`、
      `init_trunc_normal`、`zero_bias`、`zero_init_residual`（W3-C）
      —— 正交实现里 Gram-Schmidt 曾漏除 `|prev|²`（行内积 3.997），且 `zero_init_residual` 曾零掉
      捷径的 `short_bn` 而不是分支的 `bn3`；两处都由中心差分/恒等性质测出。
- [x] 数据驱动初始化（W3-F）：`lsuv_init`（预激活拉成 mean 0 / std 1；**必须在预激活上做**，
      拿 ReLU 输出算 std 时偏置平移会把一半压成 0，收敛到 0.79 就到头了）、
      `fixup_init`（残差分支最后一层 `1/√n` 缩放 + 偏置置零；Conv2d 权重是 4 层嵌套要逐叶缩放）、
      `mimic_init`（NET2NET 宽度/深度迁移，激活不一致报错）
- [x] 残差变体：Pre-LN / Post-LN / ReZero / SkipInit / Highway（ReZero/SkipInit/PreNormResidual/PostNormResidual/Highway；dx/参数梯度中心差分 ≤1e-6，ReZero 的 alpha=0 逐位恒等；顺带补了 _dense_param_count/_dense_apply_sgd/_dense_apply_adam/_norm_* 六个助手，因为 Layer/LayerNorm 本身没有 param_count()/sgd_step()）

### W4 视觉骨干（P0/P1）
- [x] `PatchEmbedding`（`Conv2d(kernel=stride=p)` + 行优先摊平，`backward` 还原特征图；`dx`/`conv.grad_w` 中心差分 ≤1e-7）+ `ViT`（CLS token + 位置编码 + 2 层编码器 + 分类头；`dx` 与 CLS/`pos`/head/`LayerNorm` 四组梯度中心差分 ≤1e-7；梯度累加语义与 `zero_grad`；2 类小任务 15 轮收敛并全对；只有 `sgd_step`，与 Transformer 系列一致）
- [x] `UNet`（2 级下采样 + 瓶颈 + 2 级上采样 + 通道拼接跳跃；可选 BN；`dx` 与编码/转置卷积/BN 三组参数梯度中心差分 ≤1e-9；Adam 40 步 MSE 0.67 → <0.22）
- [x] `FPN`（1x1 横向 + 最近邻上采样相加 + 3x3 平滑；逐级 `dx` 与横向/平滑卷积 `grad_w` 中心差分 ≤1e-9，要求相邻尺寸严格 2 倍）
- [x] `SegNet`（简化 2 级；解码用 `MaxPool2d.mask` 的**窗口内相对位置**+ stride 换算出绝对坐标做 unpool —— 直接当绝对位置用会让 `dx` 差 4 倍，`_unpool2`/`_unpool2_back` 已按此修；`dx` 与 `conv1` 梯度中心差分 ≤1e-7，40 步 Adam loss 1.01 → 0.02）
- [x] `SwinTransformer`（简化）：单 stage、交替普通/移位窗口；移位用循环 roll 且**不做 mask**；窗口内注意力手写（不复用 `MultiHeadAttention`，避免其单份 cache 被逐窗口调用覆盖）；`dx` 1.51e-09、block 内 q/k/v/o/f1/f2 逐项 ≤1e-5、150 步 Adam 收敛，898 passed
- [x] `MLPMixer`：token 混合 + 通道混合，两段 pre-LN 残差；**踩过的坑**：同一 `Layer`/`LayerNorm` 实例在循环里逐通道/逐 token 调用会互相覆盖 cache（`Layer.backward` 只读最后一次的 `input`/`z`），实测 `t1` 解析梯度 ≈ -1.19e-18 而数值是 -0.242284；改用按行批量的 `_dense_rows_forward/backward`、`_norm_rows_forward/backward` 后，block 内四组权重、两级 LayerNorm、head、`dx` 的中心差分全部 ≤1e-5 相对误差，893 passed
- [x] `AttentionPooling`（`q` 可学，`q=0` 退化为平均；`dx`/`grad_q` 中心差分 ≤1e-9）、`GeMPooling`（`p=1` 平均 / `p=2` 均方根 / `p=2000` 仍不溢出，log 空间实现）

### W5 注意力与位置编码（P1/P2）
- [x] `MLA`（低秩 KV 压缩）由 `MLAttention` 承载（`T·d_c` 的 KV cache，见 W5 第 178 行）；`MQA` 是 `num_kv_heads=1` 特例，README 第 553–557 行写明
- [x] `LearnedPositionalEncoding`（可学习位置编码表，`out=x+table`，`sgd_step`/`adam_step`，超长自动扩表）
- [x] RoPE 长度外推 `rope_inv_freqs`：`linear`（频率 /factor）、`ntk`（`base·factor^(d_k/(d_k-2))`）、`yarn`（按 `rotations=original_max·f/(2π)` 与 `beta_fast/beta_slow` 做 smoothstep 分段）；`MultiHeadAttention(rope_scaling=..., rope_factor=..., rope_original_max=...)` 接线，默认 None 与旧版逐位一致；缩放后 `W_q/W_k/W_v/W_o` 梯度与中心差分 ≤1e-6（未实现 YaRN 的 mscale 温度，docstring 注明）
- [x] `MLAttention`（MLA 简化版：低秩 KV 潜在向量 `d_c`，KV cache 从 `T·2·H·d_k` 降到 `T·d_c`；
      T=4/H=2/d_k=4 时压缩 8 倍；五组权重与偏置梯度中心差分 ≤1e-5）**W5-B4**
- [x] 位置编码剩余：T5 相对位置偏置、xPos —— `T5RelativeBias(num_heads, num_buckets=32, max_distance=128, bidirectional=True)`（`(num_buckets, num_heads)` 可学习表，`t5_relative_buckets` 按 T5 原公式：正负各占一半桶、每半区再分近端线性/远端对数）+ `xpos_factors(seq_len, d_k, base=10000.0, alpha=1.0, offset=0.0)`（位置衰减 `(t+1+offset)^(-alpha*theta_i)`，位置 0 归 1、`alpha=0` 等于关闭）。踩过的坑：①分桶若把远端也算进本半区低桶，会出现「距离 3 → 桶 3、距离 4 → 桶 0」的非单调，改成每半区再二分后单调；②`_la_apply_adam` 的签名是 `(param, grad, m, v, lr, c1, c2, beta1, beta2, eps)`，没有 `t` 参数
- [x] `LinearAttention`（`phi=elu+1`，O(T·d²)；`causal=True` 用前缀和、非 causal 用全长和；
      `W_q/W_k/W_v/W_o` 与输入梯度中心差分 ≤1e-5，120 步 Adam 收敛）
- [x] `LinformerAttention`（K/V 低秩投影 `E`/`F`，O(T·k·d)；不支持 causal；
      `k=T` 且 `E=F=I` 时与标准注意力逐位等价 <1e-12）
- [x] `PerformerAttention`（FAVOR+ 随机特征 `phi(x)=exp(Wx-|x|²/2)/sqrt(m)`，O(T·d·m)；
      iid 高斯特征，实测 m=4 误差 92% / m=64 误差 35%，是省算力而非逼近 softmax）**W5-B3**
- [x] 稀疏注意力族（W5-C）：`BigBirdAttention`（窗口+全局+随机连边）、`LongformerAttention`（滑窗+全局 token，全局位看全序列）、`ReformerAttention`（随机投影 argmax 分桶，桶内注意力）——全连通时与手写稠密注意力逐位等价，参数/输入梯度中心差分 ≤1e-5；
      修掉一个只有 causal 才暴露的真 bug：`ds == 0` 的短路把 `dv` 累加也跳过了（causal 下 t=0 只有一个可见位置，ds 恒为 0 而 dv 非零）
- [x] `FlashAttention` 分块算法（模拟，明确「不更快」）—— 在线 softmax + cache 只存 `q/k/v/m/l`（不物化 T×T）；反向按 `(m,l)` 重算概率；`block_size` 1/2/3/4/8 与稠密一致 <1e-12、块大小不影响结果 <1e-15。踩过的坑：`tot = Σ_j p_ij dp_ij` 原来只在单个键块内累加，单块恰好对、多块漏项（梯度差 6.55e-02），改成每个查询块两遍扫描后归零

### W6 图神经网络（§1 全新范式）
- [x] `GCNLayer` / `GINLayer` / `GraphSAGELayer` + `GraphReadout`（W6-A，提交在 W6-A 这一批）：`adj` 用 `N×N` 密集矩阵（教学规模足够，公式最直观），GCN 是 `D^{-1/2}(A+I)D^{-1/2}` 归一化、GraphSAGE 是均值聚合 + 可选拼接、GIN 是 `MLP((1+eps)x + Σ邻居)` 且 `eps` 可学；踩过的坑：①激活导数是**行向量**，不能与上游梯度直接 `*`（`list*list` 直接 TypeError），必须逐元素相乘；②GIN 的邻居散射要用 forward 缓存的邻接矩阵，否则 `NameError`；③无边 + 无自环时 `Â=0`，输出只剩偏置（不是「退化为逐节点全连接」）
- [x] `GATLayer` / `APPNPLayer` / `JKNet`（W6-B）—— 注意力 `dx`/`dW`/`da_src`/`da_dst` 中心差分 ≤1e-5、`alpha=0` 退化为 `Â^K H0` 且 `alpha=1` 输出恒为 `H0`、JKNet 三种汇总与手工双层堆叠黄金对照 1e-12；踩到：GAT 前向错用 `z` 而非 `ctx`（注意力形同无效）、`slope` 未乘进注意力梯度、JKNet 逐层 `dx` 宽度不同要先累加再回传
- [x] `MPNNLayer` / `GCNIILayer` + 图池化 `TopKPool` / `SAGPool`（W6-C）—— MPNN 的 `dx`/`dW` 中心差分 1.46e-11、GCNII 的 `dx`/`gW`/`grad_h0` 均 ≤3.1e-11、两种池化的全节点梯度 ≤1.3e-10；踩到：`_g_matmul` 只支持方阵邻接（GCNII 要通用矩阵乘）、池化的 score 路径梯度一开始漏加（5.6e-01）；`DiffPool` 需要软分配矩阵，留作后续
- [x] `DiffPool`（软分配池化，W6 补完）—— `X' = Sᵀ Z`、`A' = Sᵀ A S`，`S` 是逐节点 softmax（行和 1），两层 GCN（embed + pool）都会收到梯度；`dX`/`embed.W`/`pool.W`/两个偏置全部对中心差分 ≤1e-6；`A'` 对输入 `A` 的梯度是 `S dA' Sᵀ`，按 `GCNLayer` 的约定不向下传（README 写明）
- [x] 图池化汇总：`TopKPool`（可学习分数 + sigmoid gate）、`SAGPool`（GCN 打分 + tanh）、`DiffPool`（软分配）——三种的 `dX` 都与中心差分对照过；硬池化的未选中节点梯度为 0、软池化所有节点都有梯度，这个差异写进了 README 的对照说明

### W7 LLM 微调与推理（P0/P1）
- [x] 提示微调：`PromptTuning`（软提示）/ `P_Tuning`（MLP 生成前缀）——`g_prompt` 与 `anchor`/`W1`/`W2` 全部中心差分 ≤2.2e-11；`PrefixTuning`（每层 KV 前缀）留到接进 MHA 时再做，当前用 `PrefixCache` 承载缓存
- [x] 参数高效：`DoRALinear`（幅度+方向分解，`B=0` 与原层逐位一致，`g_V/g_m/g_A/g_B` 中心差分 ≤9.4e-7）/ `AdaLoRALinear`（每秩重要性 EMA + `prune(keep)` 剪秩，剪后前向等于存活秩之和）
- [x] `QLoRA` 说明：本库只有 float32 张量、没有 4-bit 量化基座，因此**不做**量化路径；`DoRALinear`/`AdaLoRALinear` 的 `params()`/`merge()`/`denormalized()` 就是 QLoRA 训练侧的同一套约定
- [x] 采样扩展：`typical_filter` / `mirostat_filter` + `mirostat_update` / `eta_filter` / `top_a_filter` / `tail_free_filter`（`tft` 0.1/0.3/0.5/0.7/0.9 实测保留 5/4/4/4/3 个 token，单调不增）
- [x] 采样扩展剩余：contrastive search / diverse beam（需要模型前向，留作生成器集成）
  - 对比搜索：`contrastive_select` 按**候选自己的隐状态**算退化惩罚（共用一个隐状态会退化成 argmax，探针抓到过）
  - 多样束：`diverse_beam_select` 的组间扣分 + `diverse_beam_search`（模型解耦）+ `diverse_beam_generate`
  - 起手 beam 全为同一前缀、命中 `eos_id` 立即进 finished；`num_return > num_beams` 报错
- [x] 约束解码：`token_id_mask`（任意 id 白名单）+ `token_prefix_mask`（合法序列前缀约束，含 `eos_id` 收尾）；JSON Schema / 正则 / CFG 的完整语法器留作后续
- [x] 约束解码剩余：JSON Schema / 正则 / CFG 的语法器（本项已有可叠加的 mask 原语）
  - `RegexGrammar`：自写正则解析（字面量/`.`/`*+?`/`[a-z]`/`[^...]`/`\d`/`|`/分组）+ Thompson NFA，`allowed`/`accepts`/`viable`
  - `CFGGrammar`：Earley chart，非终结符为 `productions` 的键，终结符支持字符串精确匹配与谓词函数
  - `json_grammar()` + `json_tokens()`：结构文法（`STR`/`NUM` 类别终结符），语义由 `json.loads` 收尾校验，**不含 JSON Schema 校验**
  - `grammar_mask`/`grammar_pick`/`grammar_generate`：掩码可直接叠到 logits；无可用 token 时显式报错而不是产出非法串
- [x] 解码加速：投机解码的接受判定 `speculative_accept`（`min(1, p_t/p_d)` 拒绝采样 + `max(0, p_t-p_d)` 残差重采样，实测接受率 0.7885 ≈ Σmin(p,q)=0.8）
- [x] `ContinuousBatcher`：请求随时进出、每步按容量重组批与补位；记账按**准入时**的提示词长度（旧版扣生成后长度会让 `used_tokens` 变负数）；测试 4 例、计数 1014→1018
- [x] `PagedKVCache`：PagedAttention 的块表数据结构（按 `block_size` 分块、逻辑→物理块映射、释放即复用；3 token 用 2 块、fragmentation=0.25）
- [x] 解码加速剩余：完整草稿-验证循环、自投机/Medusa/EAGLE（概念）
  - `speculative_decode(draft_probs_fn, target_probs_fn, ...)`：两个纯回调，含 `rounds/proposed/accepted/target_calls/accept_rate/tokens_per_target_call`
  - `self_speculative_probs(model, draft_layers)` 用前 n 个编码器块做草稿，`target_probs(model)` 做验证；层数越界构造时即报错
  - 探针用 4000 次试验核对「gamma=1 与 gamma=3 的输出分布都对目标分布最大偏差 <0.03」
  - Medusa / EAGLE 只说明机制不实现（需要额外训练的多头 / 特征级草稿）
- [x] `sequence_packing`（返回 `segment_ids` / `position_ids`）+ `packed_causal_mask` + `PrefixCache`（逐层 KV、可 `merge`/`truncate`、JSON 往返）

### W8 生成模型（§12）
- [x] 扩散：`DDPM(prediction="eps"|"v"|"x0")` + `target`/`pred_to_x0_eps` 互逆、`ddim_sample(steps=, eta=)` 确定性子序列采样、`train_step(min_snr_gamma=)` 的 `min(SNR,γ)/SNR` 加权（加权后 loss 恰为 `w×` 原 loss，实测 1.2174 → 0.8197）
- [x] `VQVAE` 向量量化：最近邻码本 + straight-through 反向 + commitment + `warmup` 只训重构；**已知限制**：无 EMA 码本更新 / dead-code restart，小数据上码本会塌缩（`fit(warmup=)` 只缓解）
- [x] `FlowMatching` / `RectifiedFlow`：直线插值 `x_t=(1-t)x0+t·noise` + 速度场回归（目标 `noise-x0`），`euler`/`heun` 两种积分器；梯度对中心差分全部通过
- [x] GAN 损失变体：`GAN(variant="bce"|"lsgan"|"wgan"|"hinge")` + `r1_penalty` 诊断
  - `bce` 与旧实现 20 步逐位一致（`d2` 仍是 sigmoid）；非 bce 用裸分数，`score()` 拿原始输出、`discriminate()` 仍返回概率
  - **WGAN 必须裁剪**：不裁时 400 步判别器损失到 -71.96、样本距离 3.21（发散）；裁剪到 `±0.01` 后稳定
  - R1/R2 只返回数值：手写单阶反向没有二阶图，无法把罚项回传进参数，故为诊断量（PyTorch 的 `autograd.grad` 才行）
  - 实测样本平均最近距离（4 角点 / hidden=16 / 1200 步）：hinge 1.03、lsgan 1.30、wgan 2.41–2.61（小容量下几乎学不动）、bce 1.21
- [ ] GAN 变体剩余：DCGAN / WGAN-GP / StyleGAN（简化）/ CycleGAN（简化）/ Pix2Pix / Progressive
- [x] `RealNVP` 仿射耦合流：`log|det J| = Σs` **精确**、`inverse` 与 `forward` 互逆（实测 4.4e-16）、`log_prob` 是真密度（2 维网格积分 0.9989）、`nll` 最大似然训练 3.2951 → 1.4170（120 轮 lr=3e-3）
  - 实现坑：`Layer` 不支持 `tanh`（改用 sigmoid 打底、外面再套 `tanh` 限幅）；`Layer` 没有 `rng` 参数（临时 `random.seed` 再恢复 `getstate`）
  - `nll = 0.5|y|² + 0.5d·log2π − logdet` ⇒ `dL/dlogdet = -1`，符号写反会让训练朝反方向走
- [x] Glow 的两个可逆组件：`ActNorm(dim)`（`data_init` 按首批统计量初始化，`logdet=Σlog_s` 与输入无关）+ `InvertibleLinear(dim)`（`log|det J|=log|det W|` 精确，LU 部分主元求 det、高斯消元求逆，反向含 `g_logdet·W^{-T}` 项，实测 1e-5 内）
  - 踩坑：`ActNorm.g_log_s` 的符号（`dy/dlog_s = y` 不是 `-y`），中心差分立刻抓到（解析 -3.64 vs 数值 5.24）
- [ ] 其它：Score-based（NCSN）/ SDE、Consistency Models、多尺度 Glow 组合（把上面两件与 RealNVP 耦合串成完整 Glow）、PixelCNN·WaveNet（概念）、Latent Diffusion（概念）、EBM

### W9 优化器与调度（§3/§4）
- [x] `SGDW`（解耦 wd）/ Nesterov / `Lookahead`（外挂任意优化器）/ `SWA` / `EMA`
      - `SGDW`：`p *= 1-lr·wd` 解耦衰减 + 动量/Nesterov 前瞻；`wd=0, momentum=0` 时与 `SGD` **逐位相同**（探针 diff 0.0e+00）
      - `NesterovSGD`：`SGDW(wd=0, momentum=0.9, nesterov=True)` 便捷写法；三步与手写参考 worst 0.0e+00
      - `Lookahead`：包任意优化器，每 `k` 步 `slow += alpha·(fast-slow)`；`alpha=0.5` 恰为中点（0.0e+00），`state_dict` 续训逐位一致
      - `EMA` / `SWA`：`shadow`/`avg` + `apply_to()`；EMA 三步闭式解 `0.966274244912`、warmup 有效衰减 `d0=0.1, d1=2/11`；SWA 增量均值 == 算术均值（0.0e+00）
      - 回调 `EMACallback` / `SWACallback`：每 epoch 一次，6 轮 → `step=6` / `n=6`；**`fit` 传给回调的是 `Model` 包装器**，内部要 `getattr(model, 'model', model)` 取 MLP（否则 `AttributeError: 'Model' object has no attribute 'layers'`）
      - 已知边界：`Lookahead` 只包「有 `step`/`.model` 的优化器」；`EMA`/`SWA` 的状态不从 `save_model(optimizer=)` 走，要自己 `state_dict()`；Adam 族的矩存在 `layer` 上，断点续训必须连 `m_w/v_w/m_b/v_b` 与 `step_count` 一起搬
- [x] 自适应族前半：`AdaGrad` / `AdaDelta` / `Nadam` / `RAdam`（W9-A）
  - 与手写参考实现逐步**逐位相同**；AdaGrad 步长单调不增、AdaDelta 对梯度整体缩放不变、
    Nadam 在 `b1=0` 退化成 RMSprop 式、RAdam 前几步纯动量后自动打开修正。
  - 踩坑：`state_dict()` 最初只深拷贝了 list，**没拷贝 dict** ⇒ 状态字典被原样返回，
    两个优化器共享同一批 list，一方 `step` 会改掉另一方要用的状态（续训不再逐位一致）。
  - 已知：状态挂在优化器实例上，`save_model(optimizer=...)` 只序列化 Adam 一族的 layer 状态，
    这四个要走 `state_dict()` / `load_state_dict()` 单独存。
- [x] 自适应族后半：Ranger / LAMB / LARS（W9-B，`LARS`/`LAMB`/`Ranger` 三个类）
  - `LARS` 的 `local_lr = lr·tc·‖W‖/(‖g‖+wd·‖W‖+eps)`、`LAMB` 的 `trust = ‖W‖/‖u‖`（截到 `trust_clip`）、`Ranger` = RAdam + Lookahead（每 `k` 步 `slow += alpha(fast-slow)` 再 `fast = slow`）
  - 三个都与手写参考**两步逐位一致**；`Ranger` 第一步每个参数满足 `p-p0 = -lr·g`（纯动量，worst 8.674e-17）；三者 `state_dict()` 续训逐位一致
  - 已知边界：只支持二维 `Layer` 权重（与 `Adam` 一致）；`LARS` 偏置恒用 `lr`；`LAMB` 的 `trust_clip` 默认 10.0 是经验值（更小时收敛更慢但更稳）
- [x] 进阶：Prodigy（无 lr）/ Sophia / Shampoo / CAME / Adan / L-BFGS（小模型）
  - Shampoo：`L^{-1/4} G R^{-1/4}` 预条件；超过 `max_dim` 的层退化为逐元素 Adam；
    逆根用循环 Jacobi（`_jacobi_eigh`，重建误差 8.9e-16、`A·M⁴ == I` 5.6e-16）。
  - Sophia：`clip(m / max(ρh, ε), ±1)`；`h` 依次取 `hessian_diags` > `hessian_fn` >
    `β₂h + (1-β₂)g²`，并明写「不是真 Hessian」（纯梯度接口拿不到二阶信息）。
  - L-BFGS：逐层分块的两循环递归 + 固定步长（无 line search）；位移对（`s`/`y`）与 `prev_p`/`prev_g` 一并进 `state_dict`，往后续训逐位一致。
  - 探针 `probe_w9e2.py` 28 项、`TestAdvancedOptimizers2` 11 例；找到两个真 bug：
    L-BFGS 的 `prev_p` 建成嵌套矩阵（`_advance` 要扁平列表，直接 TypeError）、
    状态校验按嵌套形状校验扁平列表（`load_state_dict` 把合法存档判成形状错）。
  - E1（已完成）：`Adan` / `Prodigy` / `CAME` —— Adan 用「梯度差 + 参数差」的三阶矩（第一步没有位移信息，退化为带偏差校正的 Adam）；Prodigy 不需要 lr，`d` 按「上一步是否真的下降」乘性自适应（与原论文的累积投影估计不同，docstring 与 README 都写明）；CAME 用 `r`/`c` 行/列**向量**因子化二阶矩。
  - 三个都与手写参考 5 步**逐位一致**（maxdiff 0.000e+00）；`state_dict()` 往返续训逐位一致（Prodigy 额外存 `d`/`p_prev`/`g_prev`）；400 步 XOR 收敛到 9.4e-4 / 5e-6 / 8.3e-4。
  - 踩坑：CAME 最初把 `r`/`c` 都做成逐元素矩阵，`v = r·c` 只有 `|g|⁴` 量级 ⇒ 步长被放大 `1/|g|` 倍，`clip_threshold=1e6`（等于关掉校正）时参数飞到 9.68e+02；改成 Adafactor 式行/列均值 `v_ij = r_i·c_j/mean(r)` 才把量纲对齐。另：偏置是扁平 `list[float]`，`param[j]` 的嵌套写法会直接 `TypeError`。
  - E2（已完成）：`Shampoo` / `Sophia` / `L-BFGS` —— Shampoo 走 `L^{-1/4} G R^{-1/4}`；Sophia 的曲率按 `hessian_diags` > `hessian_fn` > `g²` 取值；L-BFGS 逐层分块、无 line search。三者 `state_dict()` 往后续训逐位一致，11 例测试全绿。
- [x] 调度器族：CosineAnnealingWarmRestarts / OneCycle / InverseSqrt / Noam / CyclicLR（triangular·triangular2·exp_range）/
      - 12 个调度器（Lambda / Multiplicative / MultiStep / 余弦重启 / Cyclic / OneCycle / InverseSqrt / Noam / Polynomial / WSD / Chained / Sequential）；闭式解逐点对照、周期与里程碑边界、step(0) 夹到 1、与 `fit(lr_scheduler=)` 12 种集成；**StepLR 家族是 epoch 门控的**（step(5) 且 5%2≠0 时 lr 不变），单独喂非整除轮次不会变。
      Polynomial / Lambda / MultiStep / Multiplicative / WSD / ChainedScheduler / SequentialLR
- 明确不做：SparseAdam / SM3 —— 稀疏梯度在纯 Python 的稠密 `list` 表示下没有收益，保留为说明项，不计入完成度。

### W10 正则与训练技巧（§5）
- [x] Dropout 变体：SpatialDropout / AlphaDropout / GaussianDropout / R-Drop
  - `SpatialDropout1d`（`(T,C)` 按通道）、`SpatialDropout2d`（`(C,H,W)` 按通道）：一个通道一个掩码，保留通道乘 `1/(1-p)`；
  - `AlphaDropout`：丢弃值填 `α' = -λα ≈ -1.7581`，再仿射 `a = (1-p+p·α'²)^(-1/2)`、`b = -a·α'·p`；实测输入标准正态时输出均值 -0.006、方差 0.914；
  - `GaussianDropout`：`y = x·N(1, sqrt(p/(1-p)))`，实测 `E=1.0015`、`Var=0.2512`（理论 1 与 0.25）；
  - `r_drop_loss(logits_a, logits_b, alpha=0.5)`：对称 KL 正则项，同分布为 0、对称、`alpha` 线性；
  - 探针 `probe_w10a.py` 全绿，`TestDropoutVariants` 10 例；越界 `p`/非负 `α`/非方阵特征图都在入口报错，未 forward 就 backward 也报错。
- [x] 结构化：DropBlock / DropConnect
      - `DropBlock2d(p, block_size=3)`：合法种子位置上以 `γ = p/(bs²)·(H·W)/((H-bs+1)(W-bs+1))` 掷硬币，命中则整块置零，最后按实际保留比例放大；实测丢弃比例 0.2925（目标 0.3）、100% 的丢零元都有 ≥2 个丢零邻居；
      - `DropConnect(in, out, act, p=0.5)` 直接继承 `Layer`（复用权重与 Adam 状态）：`W⊙M/(1-p)`，正向与反向都用掩码权重；`eval()` 下与同权重 `Layer` 逐位相同，`next_delta` 用掩码权重（有限差分 -1.641944 vs -1.641944）；
      - 找到的真 bug：`DropBlock2d` 的 `γ` 一开始写成 `(1-p)/(bs²)·…`（把 `p` 当保留率），实测丢弃比例 0.534 而目标 0.3 ⇒ 改成 `p/(bs²)·…`（期望覆盖面积 `种子数·bs²·γ/(H·W) = p`）；
      - 探针 `probe_w10b.py` 24 项全绿，`TestStructuredRegularization` 10 例；越界 `p`/`block_size`、非 `(C,H,W)`、未 forward 就 backward 都在入口报错。
- [x] 对抗训练：FGSM / PGD / TRADES / MART / FreeLB / SMART
      - `pgd_examples` / `fgsm_examples` / `freelb_examples`：输入空间攻击，每步投影回 `‖x_adv−x‖∞ ≤ eps`，`alpha` 默认 `2.5·eps/steps`；单步 PGD 与 FGSM 逐位相同，FGSM 用 linear+mse 的闭式解核对过；
      - `trades_regularizer` / `mart_regularizer` / `smart_regularizer`：都是「干净 logits + 对抗 logits」的批级函数，手算 KL 与 `(1−p_max)` 加权逐项核对；SMART 只实现输出项（诚实标注）；
      - `Model.fit(adv_eps=, adv_steps=, adv_alpha=)`：每 batch 先用当前权重攻击再训练；`adv_eps=0`（默认）时权重与 loss 曲线**逐位不变**；与 `mixup_alpha` 互斥并报错；
      - 探针 `probe_w10c.py` 全绿，`TestAdversarialTraining` 12 例，全库 1153 passed。
- [x] 噪声注入（权重/梯度/输入）、课程学习 `CurriculumSampler`、Manifold Mixup、Snapshot Ensembles
      - `inject_weight_noise` / `add_gradient_noise` / `input_noise`：`sigma=0` 数值不变；200 次加噪实测均值 0.0002、方差 0.0101（`sigma=0.1`）；同种子可复现；
      - `curriculum_fraction`（linear / exp / step）+ `CurriculumSampler`：按 `difficulty` 排序，第 1 轮只给最简单 `min_fraction`（默认 0.1）比例，末轮全量；`set_epoch` + 同种子可复现；
      - `manifold_mixup(fa, fb, lam)`：嵌套结构校验 + 端点/中点闭式核对；
      - `SnapshotEnsemble` + `SnapshotCallback(every=)`：单快照预测与普通前向逐位相同，`predict` 结束后权重恢复（无副作用）；
      - 探针 `probe_w10d.py` 全绿，`TestNoiseCurriculumEnsemble` 10 例，全库 1163 passed。
- [x] 半监督：VAT、伪标签、FixMatch、MixMatch、MeanTeacher、一致性正则
      - `consistency_loss` / `consistency_regularizer`（`mse` / `kl`）+ `softmax_rows`：两个视图的预测互相靠拢，`sigma` 噪声或自定义 `aug`；
      - `vat_perturbation` / `vat_regularizer`：随机方向 + `power` 次幂迭代求扰动（`K L(p(x)‖p(x+r_adv))`），两条分支分别反传累加；实测扰动 L2 范数恰为 `eps`，冻结扰动后权重梯度 vs 有限差分 2.808382e-04 / 2.808381e-04；
      - `pseudo_label_targets` / `fixmatch_targets` / `fixmatch_regularizer`：argmax 硬标签 + 置信度阈值（不足则权重 0，全部丢弃时损失 0、梯度全零）；
      - `mixmatch_targets`（多次增强平均后按 `T` 锐化）+ `soft_target_ce`（软标签 CE 与梯度）；
      - `MeanTeacher` + `mean_teacher_regularizer`：教师权重 EMA（`update`）、教师前向用「换入 → 前向 → 换回」实现；**真 bug**：`_swap_in` 原先用 `list(layer.weights)` 保存现场，行对象是同一份引用，换回时学生权重已被教师权重覆盖 —— 改成 `_semi_copy` 深拷贝后修复；
      - 探针 `probe_w10e.py` 全绿（闭式梯度 vs 有限差分：`mse` 1.0e-11、`kl` 4.6e-11），`TestSemiSupervised` 10 例，全库 1173 passed。
- [x] 梯度检查点（activation checkpointing）、混合精度模拟（标注无真收益）
  - `checkpoint_grads`：前向只留段边界、反向逐段重放；`segments=1/2/3` 与普通反传
    逐位相同。诚实边界：本库本来就只缓存 O(层数) 份输入，**不会真省内存**。
  - `fp16_round` / `bf16_round`：IEEE 半精度（`struct` `'e'`）/ bfloat16 RNE 截断；
    `|v| > 65504` 溢出为 `inf`，低于次正规归 0。
  - `quantize_model`（原地舍入，幂等）、`mixed_precision_grads`（loss scaling 后
    舍入再除回；实测 `1e-9` 的梯度不缩放会被舍成 0，`scale=8192` 时保住）。
  - 混合精度**只模拟舍入**，不做半精度矩阵乘，因此没有真实加速。

### W11 数据管道（§7）
- [x] `Dataset` 抽象 + 流式/懒加载；`Sampler` 体系（Weighted / Stratified / Curriculum / Subset / BatchSampler）
  - `Dataset` / `ListDataset` / `Subset` / `LazyDataset`（FIFO 缓存 + 命中计数）/
    `IterableDataset`（流式，无需长度）。
  - `Sampler` / `SequentialSampler` / `RandomSampler` / `WeightedSampler`（无放回用
    Gumbel 指数键）/ `StratifiedSampler` / `SubsetSampler` / `BatchSampler`。
  - `DataLoader` 新增 `sampler=` / `batch_sampler=`（默认路径仍与旧版逐位相同）。
- [x] 长度分桶 `bucket_by_length`；预处理缓存（文件哈希命中）
  - `bucket_by_length`：乱序 → 按长度排序 → 桶内再乱序 → 切 batch，`drop_last` 可选。
  - `PreprocessCache`：键 = 源文件内容哈希（或 size+mtime）+ 参数指纹，命中直接读
    JSON，未命中构建后先写临时文件再 `os.replace`。
- [x] 编码器：`OneHotEncoder` / `LabelEncoder` / `OrdinalEncoder` / `TargetEncoder` / `FrequencyEncoder` / `WOEEncoder`
  - `LabelEncoder`（排序类别 + 往返）、`OrdinalEncoder`（逐列独立）、
    `OneHotEncoder`（`drop='first'`、`handle_unknown='error'|'ignore'`、
    `get_feature_names`）。
  - `TargetEncoder`（平滑到全局均值，docstring 警告只能 fit 训练集）、
    `FrequencyEncoder`、`WOEEncoder`（`log(bad/good)` + 平滑，未见类别 0）。
- [x] 图像增强管道：`Compose` / `RandomCrop` / `RandomResizedCrop` / `RandomRotation` / `ColorJitter` / `GaussianBlur` / `Normalize`
  - 统一吃 `(C, H, W)` 嵌套 list，`seed` 决定一条可复现的随机流（同实例连续调用继续抽）。
  - `RandomResizedCrop` 10 次采样失败退化为中心裁剪；`RandomRotation` 逆映射双线性 + `fill`。
  - `ColorJitter` 的 hue 用行和恒为 1 的亮度保持矩阵（灰度图抖动后仍是同一灰度）。
  - `GaussianBlur` 边界按实际权重重新归一化 ⇒ 常量图模糊不变；核/`sigma` 非法报错。
- [x] 切分扩展：`GroupKFold` / 时间序列切分 / Purged K-Fold
  - `group_k_fold_indices`：同组不跨折，组按样本数降序塞进最空的折（折间极差 ≤ 1）。
  - `time_series_split_indices`：测试块恒在训练块之后；`expanding` 扩张 / 滚动窗口，
    `gap` 留空档；唯一约束是首折训练集非空。
  - `purged_kfold_indices`：连续分块 + `purge`/`embargo` 剔除测试块附近样本，
    清洗后训练集为空会报错而不是静默返回空折。
- [x] Scaler 族：Robust / MaxAbs / QuantileTransformer / PowerTransformer
  - `RobustScaler`（中位数 + IQR，可分别关掉中心化/缩放）、`MaxAbsScaler`（不中心化）。
  - `QuantileTransformer`：分段线性分位数映射，`uniform` / `normal` 两种输出；
    两端点在 normal 模式有 1e-8 量级逆变换残差（已写进 docstring）。
  - `PowerTransformer`：Yeo-Johnson / Box-Cox，lambda 用黄金分割搜索近似 MLE。
- [x] 缺失值：`SimpleImputer` / `KNNImputer` / `IterativeImputer`
  - `SimpleImputer`：`mean` / `median` / `most_frequent` / `constant`（`constant` 必须给 `fill_value`）；
    整列全缺失填 `0.0`，不静默丢列。
  - `KNNImputer`：只在**共同观测到**的列上算距离；`weights='distance'` 按 `1/(d+eps)` 加权；
    没有任何邻居共享观测列时退回该列 fit 均值。
  - `IterativeImputer`：初始值取列均值，轮转岭回归（特征标准化 + `alpha` 惩罚），只回填原本缺失的位置；
    最大变化 < `tol` 提前停，`n_iter_` 记录实际轮数。docstring 写明是单变量轮转近似，无 `sample_posterior`。
  - 探针 50 项：三种策略闭式值、k=1/k=2 与 distance 权重 0.5、`y=2x` 与多列缺失恢复、
    观测值不改写、`n_iter_` 夹紧、列主元消元精确解与奇异返回 None、22 条非法输入。

- [x] 降维：`PCA` / `t-SNE`（简化）；`SMOTE` / `ADASYN`；特征选择（VarianceThreshold / SelectKBest / RFE / L1）
  - `PCA`：协方差 + 幂迭代取前 k 个特征向量（符号归一），`explained_variance_ratio_`、白化、`inverse_transform` 齐备。
  - `TSNE`：精确 O(n²) 版；按行二分 sigma 使熵 = `log(perplexity)`，对称化后 Student-t 核梯度下降，
    前 50 轮 `early_exaggeration=12`、动量 0.5→0.8。**真 bug**：二分搜索方向写反（熵偏大时该缩上界却放了下界），
    导致 P 近似均匀、三簇完全不分离（intra 116 > inter 111）→ 修后 5.8 vs 92.8。
  - `SMOTE` / `ADASYN`：`_w11g_neighbors(x, i, candidates, k)` 原先把 k 当候选集合（`for other in idx` 报 `int not iterable`），
    改成显式候选列表；ADASYN 的难度比在全样本上算、合成只在少数类内部取邻居。
  - `VarianceThreshold` / `SelectKBest`（`f_classif` 与 `pearson`）/ `RFE`（岭回归系数递归消除）/ `L1Selector`（坐标下降 Lasso）。
    **真 bug**：`f_classif` 里误用标签均值当特征总均值（F 从 8 变 40），修正后常量列得 0 而不是 inf。
  - 探针 74 项：PCA 秩一/往返/白化、t-SNE 簇内<簇间且同种子可复现、SMOTE 4:4 与线段约束、
    ADASYN 加权、四个选择器的支持集与分数、28 条非法输入。

- [x] 文本增强（同义词替换 / EDA / Token Dropout）；音频增强（概念）
  - `synonym_replace` / `random_insert` / `random_swap` / `random_delete` / `token_dropout` / `eda_augment`，
    以及 `audio_gain` / `audio_add_noise` / `audio_time_shift` / `audio_clip` / `audio_polarity_invert`。
  - 判定修正：同义词表里的空列表本来被我判成非法，实际语义应是「这个 token 没有同义词」，已放宽为合法（
    探针因此改一条期望值，不是改实现去凑数）。
  - 探针 46 项：替换/插入/交换/删除的确定性与不变量、EDA 的词表闭包与 `p_rd=1` 仍非空、
    加噪实测 SNR 10.0000 dB、6 dB 增益 1.9953、平移与限幅闭式值、20 条非法输入。

### W12 分词与批处理（§8）
- [x] 特殊 token 统一管理（BOS/EOS/PAD/UNK/SEP/CLS/MASK）
  - `SpecialTokens`（默认与 `WordPieceTokenizer` 兼容）+ 模块级 `SPECIAL_TOKENS`，
    `tokens()` / `as_dict()` / `id_of(name, vocab)`，validate 覆盖空串、首尾空白、重复、未知字段。
  - 探针 8 项（含 `from_dict` 局部覆盖与 `id_of` 对 list/dict 两种词表）。
- [x] 批处理：`pad_sequences` / `truncate` / `attention_mask` / `tokenize_batch`
  - 另加 `batch_sequences(...)`：返回 `lengths` 并按**真实长度**生成掩码，避免「pad 值也是
    合法 token」时 `attention_mask` 按值判断的歧义；两者都保留、docstring 写明差别。
  - **真 bug**：`pad_sequences` 的参数名 `truncate=` 把模块函数 `truncate` 遮蔽了（
    `TypeError: 'str' object is not callable`）；抽出内部 `_w12_truncate` 供两者共用。
  - 探针 31 项：特殊 token、裁剪两侧、pre/post 补齐、`max_len` 截断、按值掩码、
    `batch_sequences` 的 lengths 与两种掩码、歧义对照、真实 `WordPieceTokenizer` 端到端、22 条非法输入。

- [x] `SentencePieceTokenizer`（Unigram LM 简化版）；`WordTokenizer` / `WhitespaceTokenizer` / `RegexTokenizer`
  - 四个分词器统一 `fit/encode/decode` 契约，词表固定 `[PAD]=0`、`[UNK]=1`，可直接喂 `batch_sequences`。
  - `SentencePieceTokenizer`：子串频次建 unigram + Viterbi 切分，强制收录语料里的全部单字符，
    未登录字符退 `<unk>`；docstring 写明与真 SentencePiece 的四点差别（无 EM、无字节回退、
    只支持 `▁`、无 dummy prefix 开关）。
  - 为避免引入新依赖，`RegexTokenizer` 用的是标准库 `re`（`AI.py` 顶部新增 `import re`）。
  - 探针 51 项：四种分词器的词表顺序与往返、标点/大小写/正则切分、SentencePiece 的 id 固定、
    子词复现、`max_piece_len` 约束、未登录与中文单字、与 `tokenize_batch` 的协作、30 条非法输入。

### W13 指标扩展（§9）
- [x] 回归：MedAE / MSLE / RMSLE / ExplainedVariance / AdjustedR² / Quantile
  - `median_absolute_error` / `mean_squared_log_error` / `root_mean_squared_log_error` /
    `explained_variance_score` / `adjusted_r2_score(..., n_features)` / `mean_pinball_loss`。
  - MSLE 要求目标与预测都非负（负值直接 `ValueError`）；`explained_variance_score` 不惩罚
    系统性偏移，与 `r2_score` 的差别在 README 里用「预测整体 +5」的例子说明。
  - 探针 49 项：六个指标的闭式/手算对照、常量目标约定、与 `quantile_loss` 的数值一致、
    以及全部非法输入。
- [x] 效率：`inference_benchmark`（时间 / 吞吐 / 峰值内存）
  - 返回 `total_seconds` / `per_call_ms` / `throughput_per_s` / `peak_memory_bytes` / `params`。
  - `peak_memory_bytes` 用 `tracemalloc`，只统计 Python 堆（本库全是 `list[float]`），
    不含解释器常驻内存；`params` 是 weights/biases 标量总数，非 MLP 层记 0。
  - `AI.py` 顶部新增 `import tracemalloc`。
- [x] 分类：BalancedAccuracy / CohenKappa / Brier / LogLoss / PR-AUC / Specificity / Sensitivity / F-beta / Hamming / Jaccard（10 个；`_binary_counts` 统一二元计数、`_proba_matrix` 统一概率矩阵校验；探针 47 项全过，含 kappa 的 pe≥1 无定义、PR-AUC 阶梯面积、log_loss 与 `cross_entropy_loss` 同值）
  - 单位口径：Brier 单列按 `(p−y)²`、多列按 one-hot 求和平方；`log_loss` 单列时标签必须 0/1
- [x] 排序：NDCG@k / MAP@k / MRR / P@k / R@k / HitRate / Coverage（10 个函数：单查询 6 + 批量聚合 3 + coverage；`_rank_query`/`_rank_batches` 统一校验，按 `(-score, index)` 稳定排序；探针 48 项全过，含分级相关性 NDCG、AP 分母 `min(rel, k)`、coverage 并列口径）
  - 口径：`k=None` 取全部、`k>n` 截到 n；P@k 分母是 `min(k, n)`；NDCG 增益 `2^rel−1`
- [x] 分割：mIoU / Dice / HD95 / BoundaryF1 / PixelAcc / FW-IoU（6 个；`_seg_confusion` 统一混淆矩阵与 `ignore_index`，`_seg_boundary` 用四邻域差分取边界；探针 37 项全过，含手算 IoU/Dice、tolerance=0/1 两档 BoundaryF1、HD95=1.0、一侧空边界的两种处理）
  - 口径：边界只在图像内部判定（图外邻域不制造边界）；HD95 是双向距离合并后取第 95 百分位
- [x] 文本：语料级 BLEU / ROUGE-1/2 / METEOR（简化）/ CIDEr（简化）/ Self-BLEU / Perplexity（6 个新函数；`perplexity` 早已存在、是 `CausalLM` 的方法；`_text_ref_sets` 统一参考写法、`_meteor_chunks` 做贪心对齐；探针 36 项全过）
  - 口径：多参考必须写「列表的列表」；`cider_score` 的 IDF 只在本条候选的参考集内统计；smooth 只对「该阶零命中」生效
- [x] 语音：WER / CER / SER（编辑距离）（`_edit_ops` 做 DP 回溯给出替换/删除/插入三元组、`_speech_check_empty` 统一空参考；探针 26 项全过，含中文字符级 CER 与 SER 的多出槽位）
  - 口径：参考为空抛 ValueError，假设可为空；SER 的分子包含「多出的槽位」
- [x] 聚类：Silhouette / Calinski-Harabasz / Davies-Bouldin / ARI / NMI / AMI
  - silhouette 每点 `(b-a)/max(a,b)`，单点簇记 0，要求 `2 <= k < n`
  - CH 与 DB 只依赖质心距离，常数维样本会报错而不是除以 0
  - ARI 用组合数做机会校正；AMI 用超几何期望（`lgamma` 实现），小样本可为负（`[0,0,1,1]` vs `[0,1,0,1]` = -0.5）
  - NMI/AMI 支持 arithmetic / geometric / min / max 四种归一化
  - 探针 33 项（含独立参考实现逐步对照 silhouette 与 DB）
- [x] 校准：ECE / MCE / ReliabilityDiagram
  - 置信度取最大类概率，单列概率按二分类正类处理（`pred = p > 0.5`）
  - `strategy='uniform'` 等宽分箱；`'quantile'` 按分位点切，重复值会去重
  - 只统计非空箱；MCE 是各箱偏差绝对值的最大值
  - 探针 30 项（手算 ECE 0.14 / MCE 0.15 + 单列与两列等价 + 十条非法输入）
- [x] 检测：mAP / mAP@0.5 / mAP@[.5:.95] / AR（简化）
  - 匹配按分数降序贪心，同图同类未占用框中取 IoU 最大者
  - AP 用全点插值（precision 包络 + Δrecall 加权）；重复命中算 FP
  - 不做 NMS / 旋转框 / 忽略区域；mAP@[.5:.95] 是 10 档阈值的均值
  - 探针 31 项（IoU 手算、FP 在前的 2/3、阈值 0.5 vs 0.3、AR 截断、十二条非法输入）
- [x] 图像：PSNR / SSIM / MS-SSIM；FID / KID / IS（简化并标注口径）
  - SSIM 用均匀窗口（无高斯加权），窗口大于图像时退化为全局统计
  - MS-SSIM 用 2×2 平均池化，尺寸不足时减少级数并重新归一化权重
  - FID/KID 直接吃特征向量（无预训练 Inception）；FID 不做样本量校正，协方差维数 > `max_dim` 报错并提示先 PCA
  - KID 是无偏 MMD²，小样本可为负值（docstring 与 README 都写明）
  - 探针 35 项（SSIM 与独立参考实现逐窗口对照、FID 三种方差组合、KID 与参考 MMD 一致）
- [x] 一致性：Fleiss Kappa / Krippendorff Alpha；公平性：Demographic Parity / Equal Opportunity / Disparate Impact

  - 已实现：`fleiss_kappa`（固定类别、不允许缺失）、`krippendorff_alpha`（`None` 缺失 +
    nominal/ordinal/interval 三种距离，重合矩阵 `1/(mₐ−1)` 摊分）、
    `demographic_parity_difference` / `equal_opportunity_difference` / `disparate_impact_ratio`。
  - 三项公平性只是描述性比较，无显著性检验；`equal_opportunity_difference` 在
    「含正样本的分组少于 2 个」时抛错而不是返回 0。
  - 探针 52 项；`fleiss_kappa` 的 Fleiss(1971) 经典例手算核对为 **0.2199**（不是常被
    引用的 0.210，那是另一张表），`krippendorff_alpha` 另写一份按定义直接枚举的方案做
    独立参考对照。

### W14 可解释性（§10）
- [x] 归因：Saliency / Gradient×Input / SmoothGrad / IntegratedGradients / DeepLIFT（简化）

  - 已实现：`saliency_map` / `gradient_x_input` / `smooth_grad` /
    `integrated_gradients`（中点黎曼和）/ `deep_lift`（Rescale 乘子回传）。
  - `deep_lift` 只支持逐元素激活，`softmax` 直接报错（规则限制，不是实现省略）；
    线性网络与 ReLU 网络上完备性 `Σ attr = f(x) − f(baseline)` 精确到 1e-12。
  - 探针 41 项；两处首跑失败都是我的期望值错（`smooth_grad` 在小 sigma 下采到同一
    ReLU 分片、`Model` 包装器的 saliency 期望写成两类都有梯度）。
- [x] 注意力：Attention Rollout / Attention Flow（在 `attention_maps` 上做矩阵连乘）

  - 已实现：`attention_rollout`（`Ã = 0.5A + 0.5I` 连乘）、`attention_flow`
    （按列归一后连乘，简化版不含梯度项）；输入支持每层 `T×T`、`heads×T×T`
    或 `attention_maps` 返回值，头部融合 `mean/max/min`。
  - Rollout 与 Flow 在均匀注意力上本来就不同（前者含单位项），README 写明。
  - 探针 37 项；首跑 4 条失败全是我的期望值错：以为 flow 也加单位项、
    把 `discard_ratio` 的期望方向写反、`max`/`min` 融合的两处算术错。
- [x] `permutation_importance`；`feature_maps_ascii`（`attention_ascii` 扩展）
  - `permutation_importance`：本地 `random.Random(seed)` 打乱、默认 `model.score`、返回 baseline/mean/std/names，探针 39 项
  - `feature_maps_ascii`：单张 `H×W` 与 `(C,H,W)` 两种输入、每通道独立归一、块均值下采样
  - 探针首跑 3 条失败都是我自己的期望值错（重要性的符号、峰值字符、零点字符表）
- [x] CNN：GradCAM / GradCAM++ / ScoreCAM / LayerCAM
  - `cam_heatmap`（GradCAM / GradCAM++ / LayerCAM）+ `scorecam_heatmap` + `gradcam_weights` + `cam_ascii`
  - 复用 `Sequential` 的反向协议（`delta_index` 取 delta），定位到非卷积层会明确报错
  - 探针 54 项；首跑 2 条失败是我自己的期望值错（全负热图归一化后必为 0、`cam_ascii` 的表头带通道名）
- [x] 嵌入可视化（复用 t-SNE）；SHAP（KernelSHAP 简化）/ LIME（简化）/ Anchors
  - 已实现：`kernel_shap` / `lime_explain` / `anchor_rule` / `scatter_ascii` / `embedding_visualize`（复用 `TSNE` / `PCA`）
  - 真 bug：二值掩码只有 `2^m` 种，`n_samples` 未截断时「随机抽 + 去重」循环永不结束（探针直接挂死）；已截断并加回归测试
  - 探针 49 项；三者都标注为简化实现（非 TreeSHAP、不建模特征相关性、Anchors 无 beam search）
- [x] TCAV；Activation Maximization / Network Dissection；Counterfactual / CEM；Prototypes / Influence（简化）
  - `tcav`：CAV = 概念均值 − 参照均值归一化，敏感度取 `dScore/dA · CAV` 的符号比例；
  - `activation_maximization` / `network_dissection` / `prototype_examples`；
  - `counterfactual_example`（目标类 one-hot 减原类 one-hot 的单次回传，非 CEM）与 `influence_scores`（一阶梯度余弦，非留一重训）；
  - 回传到输入的三件只支持 `Layer` 组成的 MLP，卷积请用 `cam_heatmap`（已在 README 写明）。

### W15 推理与部署（§11）
- [x] 结构化剪枝：通道 / 头 / 层（现有 `prune_magnitude` 是非结构化）
  - `channel_scores` / `prune_channels`（掩码式置零，形状不变）/ `head_scores` / `prune_heads` / `prune_layers`（真删中间层）/ `sparsity_report`；
  - 删层要求相邻维度兼容，否则直接 `ValueError`，不会悄悄改形状。
- [x] 特征蒸馏（hint layer）；QAT（量化感知训练）；PTQ / GPTQ / AWQ / SmoothQuant（教学，标注 int8 未真存）
  - 特征蒸馏复用 W1-C 的 `hint_loss`；QAT 提供 `qat_grads`（直通估计，只给梯度）；
  - PTQ 用 `quantize_weights` 真存 int 编码 + 每层一个 scale，`quantization_error` 给误差上界；
  - SmoothQuant 只提供 `smooth_quant_scale` 与 `smooth_quant_apply`（激活端除 `s` 由调用方做）；GPTQ / AWQ 未实现，README 已写明。
- [x] 集成：Voting / Stacking / Bagging / Boosting
  - `VotingEnsemble`（soft 平均概率 / hard 多数票，类别数不一致直接报错）、`BaggingEnsemble`（有放回重采样 + 局部 RNG）、`stack_features` + `StackingEnsemble`（元特征再训元模型）、`AdaBoostStumps`（二分类桩，`alpha = lr·½ln((1−err)/err)`）。
  - 实现修正：`StackingEnsemble.fit` 一开始给 `Model.fit` 传了 `lr=`，而 `fit` 没有这个参数（学习率在 `Model` 构造时给）⇒ 改成写进 `meta_model.optimizer.lr`。
  - 探针 39 项；测试 `TestEnsembles` 10 例。
- [x] 推理优化：投机解码（已在 W7 实现，此处做集成）、序列打包、连续批处理（W7 已实现 `ContinuousBatcher`）、并行概念（文档）
  - `pack_decode_batch` 把 `sequence_packing` + `packed_causal_mask` 组装成一次前向的入参（`tokens`/`segment_ids`/`position_ids`/`attention_mask`/`lengths`）；
  - `speculative_generate(draft, target, ...)` 是草稿-验证循环：`gamma` 个候选、逐位置算两个模型的分布、交给 `speculative_accept` 拒绝采样，返回接受率统计；
  - 实现修正：`pack_decode_batch` 一开始把 `sequence_packing` 的**整个 dict** 当成 `tokens` 返回（掩码对、token 错），改成取 `packed["tokens"]` / `packed["segment_ids"]` / `packed["position_ids"]`；
  - 探针 25 项；测试 `TestPackingAndSpeculative` 5 例。

### W16 自监督·多模态·RL（§13）
- [x] 对比学习训练器：SimCLR / MoCo（队列）/ BYOL / BarlowTwins / VICReg / SwAV / DINO
  - `nt_xent_loss`（SimCLR）、`moco_loss` + `MoCoQueue`、`byol_loss` 三个函数返回**对嵌入**的解析梯度，探针用中心差分核对（最差 <1e-6）；
  - `barlow_twins_loss` / `vicreg_loss` / `swav_loss` / `dino_loss` + `sinkhorn` / `update_dino_center` 只返回损失值（批次统计与 Sinkhorn 的梯度容易写错，README 已写明不提供未核对的梯度）；
  - 实现修正三处：`feat = [ua[i] + ub[i] ...]` 把两行拼成 2d 长的行（应为 `ua + ub`）、BYOL 梯度漏了 `1/n`、`sinkhorn` 的行/列目标写成 `1` 与 `1`（应为 `1/n` 与 `1/k`）；
  - 探针 38 项；测试 `TestContrastiveLearning` 8 例。
- [x] 掩码建模：MAE / BeiT / SimMIM（视觉侧）
  - `patchify` / `unpatchify`、`random_masking`（`mask_ratio=0` 时一个都不遮，这条是探针逼出来的语义修正）、`block_mask`
  - `mae_loss`（被遮 patch 的 MSE）、`simmim_loss`（平滑 L1）、`beit_loss`（离散 token 交叉熵）
  - 只提供掩码与损失，编码器/解码器由用户拿 `TransformerEncoder` 自己拼；`beit_loss` 的 dVAE tokenizer 不在本库范围内
- [x] 多模态：CLIP（双塔 + InfoNCE）、跨模态融合（CrossAttention 扩展）
  - `clip_similarity` / `clip_loss`（对称 InfoNCE，两侧梯度都对中心差分）/ `contrastive_retrieval_accuracy`
  - `CLIPProjector`：两个线性投影头 + `loss_and_grads` + SGD 更新，30 步实测 loss 1.857 -> 0.376
  - `CrossModalBlock`：文本 query × 图像 memory 的交叉注意力 + 残差 + 逐行归一化，backward 同时回传图像与文本两侧；memory 只有 1 行时 `W_q` 梯度恰为 0（softmax 恒定），测试里专门记了这一条
- [x] 预测式自监督：Rotation / Jigsaw
  - `rotation_targets` / `rotate_image`（只有 90 度整数倍，非方图只能 0/180）/`rotation_classifier_head` / `rotation_loss`
  - `jigsaw_permutations`（保证非恒等、互不相同，凑不够时明确报错）/`jigsaw_shuffle` / `linear_probe_accuracy`
  - 只提供任务构造与评估，不内置训练循环；探针 42 项，测试 `TestPredictiveSelfSupervision` 7 例
- [x] RL 环境：`GridWorld` / `CartPole`（内置）
  - `GridWorld`：撞墙原地不动但照扣 `step_reward`、到达 goal 给 `goal_reward`、超过 `max_steps` 标 `truncated`
  - `CartPole`：显式欧拉 + Gym 原始常数，探针里用**独立重算**核对第一步的 `x`/`x_dot`（差 0.000e+00）
  - 附带 `discounted_returns` 与 `rollout`；测试 `TestRLEnvironments` 8 例
- [x] RL 算法：DQN / Double / Dueling / Rainbow（简化）/ REINFORCE / A2C / PPO / SAC / TD3 / DDPG
  - W16-F 已做价值侧：`ReplayBuffer`（FIFO）+ `dqn_targets` / `double_dqn_targets`
    + `huber_td_loss`（返回梯度）+ `dueling_q_values` + `DQNNetwork` + `DQNAgent`
    （`double=` / `dueling=` 开关）。Rainbow 只到 Double + Dueling，不做优先采样 / N-step / 分布式。
    `probe_w16f.py` 58 项：FIFO 覆盖、Double 与普通目标在构造样本上不同、Huber 两分支与梯度截断、
    单层权重中心差分 1e-5、单状态赌博机 150 步后 argmax = 最优动作、`sync_target` 逐位一致。
  - W16-F2 已做策略侧（离散）：`policy_gradient_loss` / `entropy_term` / `value_loss` /
    `gae_advantages` / `ppo_loss`（裁剪目标）+ `PolicyNetwork` / `ValueNetwork` / `PPOAgent`，
    probe 51 项（策略梯度与 PPO 梯度都对中心差分、裁剪分支梯度为 0、GAE 手算）。
  - 仍未做：SAC / TD3 / DDPG（连续动作，需要高斯策略的 reparameterization）与 RLHF。
- [x] RL 基础设施：ReplayBuffer / ε-greedy / UCB / Thompson；RLHF（PPO-based）/ 奖励模型
  - ReplayBuffer 在 W16-F 已做；W16-G 补 `epsilon_schedule`（linear/exp）、
    `epsilon_greedy_action`、`ucb1_action`、`BetaBernoulliBandit`（Thompson）。
  - RLHF 侧只到「奖励模型 + 奖励塑形」：`bradley_terry_loss`（成对偏好，带两侧梯度）、
    `kl_penalty`、`rlhf_reward`（`r - β·(logp - ref_logp)`）。
  - 完整的 PPO-RLHF 四模型训练循环**没有**封装（README 给了拼法），策略侧复用 W16-F2 的 `PPOAgent`。
  - probe_w16g.py 48 项：ε 单调不增、UCB 稀疏臂优先、Thompson 200 轮后最优臂后验最高、
    Bradley-Terry 在 ±50 的分数差下不溢出且梯度与中心差分一致。
- [x] 元学习：MAML / Reptile / ProtoNet / Matching Networks
  - `maml_inner_step` / `maml_outer_grads`（一阶）/ `reptile_step`
  - `prototype_embeddings` / `prototype_logits` / `prototypical_loss`
  - `matching_attention` / `matching_logits`
  - 探针 41 项；修掉 `prototypical_loss` 梯度漏除样本数（差 1/2）
- [x] 联邦学习：FedAvg / FedProx / FedBN
  - `named_params` / `load_named_params`（MLP 参数摊平与写回）
  - `fedavg`（等权/加权平均）/ `fedprox_penalty`（近端项与梯度）
  - `fedbn_split` / `fedbn_average`（按前缀区分私有参数）
  - 探针 40 项；不含客户端训练循环与通信调度（README 已写明）

### W17 领域应用（§16）
- [x] 时序：TCN / Informer（简化）/ Autoformer（简化）/ PatchTST
  - `TCN` / `TCNBlock`（因果膨胀卷积 + 残差，手写反向）
  - `prob_sparse_attention`（Informer 稀疏注意力，前向近似）
  - `series_decompose` / `auto_correlation` / `autoformer_aggregate`
  - `PatchTSTEmbedding`（patch + 线性嵌入 + RevIN）
  - 探针 47 项；修掉 `TCNBlock` 反向漏掉第二道 ReLU 掩码（conv2 梯度多算，差 1.768e-01）
- [x] 推荐：MF / NCF / Wide&Deep / DeepFM / DIN
  - `matrix_factorization` / `mf_predict` / `mf_rmse`（显式反馈，逐条 SGD）
  - `NCF`（GMF + MLP 融合，逐样本手写反向）、`WideAndDeep`（宽 + 深，logit 相加）
  - `DeepFM`（FM 二阶交叉闭式 + 深侧）、`DIN`（点积注意力池化 + 输入梯度）
  - 探针 42 + 37 项，参数梯度全部对过中心差分（< 1e-6）
  - `matrix_factorization` / `mf_predict` / `mf_rmse`（显式反馈，逐条 SGD）
  - `NCF`（GMF + MLP 融合，逐样本手写反向）、`WideAndDeep`（宽 + 深，logit 相加）
  - 探针 42 项（含 8 组参数的中心差分 < 1e-7）
  - 待做：`DeepFM`（FM 二阶交叉 + 深侧）、`DIN`（序列注意力池化）
- [x] 图像：风格迁移（NST）/ 超分（SRCNN 式）/ 去噪（DnCNN 式）
  - `upscale_bilinear` / `add_gaussian_noise`（双线性放大、可复现高斯噪声）
  - `SRCNN`：9×9 / 1×1 / 5×5 三层 same 卷积，`fit` 用 MSE 手写反传；权重梯度对中心差分 < 1e-6
  - `DnCNN`：预测噪声残差，`denoise = 输入 − 预测`；训练目标 `noisy − clean`
  - **未做 SRGAN**（需要对抗训练 + 感知损失，本库没有 VGG 特征）；探针 33 项
  - `gram_matrix` / `FeatureExtractor` / `style_transfer`：Gram 梯度沿「特征图 -> 逐层反传」回到像素，单步更新与 `-lr*dL/dx` 对齐；特征栈是随机初始化后固定的小卷积，不是 VGG 预训练权重
- [x] 检索：Faiss 式暴力 + 单层 NSW（HNSW 的简化）/ IVF
  - `BruteForceIndex`（`l2` / `ip` 两种度量，结果与精确扫描逐位一致）
  - `IVFIndex`（k-means++ 初始化 + 只在最近 `n_probe` 个簇内扫描，全探测时与暴力完全一致）
  - `NSWIndex`：**单层**近邻图 + 贪心束搜索，不是分层 HNSW（没有层级/启发式剪枝），构造数据上 recall@5 到 1.0；探针 32 项
- [x] 语音 Conformer（简化）/ Mel 滤波器；3D PointNet（简化）；视频 3D-CNN / Two-Stream；OCR CRNN；检测 YOLO（简化）；深度/姿态估计（概念）
  - 语音前端已完成：`mel_filterbank` / `frame_signal` / `dft_power`（朴素 DFT）/ `log_mel_spectrogram` / `dct_matrix` / `mfcc`；`ConformerBlock` 只做前向，无 GLU / 相对位置 / 手写反向；探针 34 项
  - 点云已完成：`farthest_point_indices` / `chamfer_distance` / `voxelize_points` / `PointNetEncoder`（逐点 MLP + 最大池化，只做编码，无 T-Net、无分类头）；探针 24 项
  - 视频已完成：`Video3DCNN`（Conv3d/ReLU/MaxPool3d 串成可反传整网 + 全局平均池化 + Layer 分类）、`TwoStreamFusion`（两路特征各投影后拼接分类，**不含光流**）；探针 17 项
  - OCR / YOLO 已完成：`CRNN`（卷积 + 逐列分类，**省略循环层**）、`ctc_greedy_decode`、`yolo_grid_targets` / `yolo_decode` / `nms`；探针 30 项
  - 深度 / 姿态估计**不做实现**：真实做法依赖大规模 3D/关键点数据集与数百 MB 的预训练骨干，本库无权重可用；只在此说明取舍，不放空壳接口。

### W18 测试与数值稳定性（§15，贯穿各波）
- [x] `grad_check_all(model, ...)`：自动遍历所有带参数的子模块
  - 已实现：沿 `__dict__` 深度优先收集 Layer 式子模块，逐个用随机输入 + 随机上游梯度做中心差分；非 Layer 参数持有者进 `skipped` 并说明原因；支持 `max_params` 抽查与 `seed` 复现；探针 34 项
- [x] 数值稳定性套件：1e300 / 1e-300 / NaN / Inf / 全零 / 空序列 不静默出 NaN
  - 已实现 `numeric_scan` / `is_all_finite` / `assert_all_finite` / `stability_cases` / `stability_probe`；据此修掉 softmax 的 +inf→NaN 与 NaN 静默、l1/log_cosh 空输入 ZeroDivisionError、`_numeric_pair` 与 `mse_loss` 的 NaN 输入、`r2_score` 的 OverflowError；模型热路径不校验输入（已写进 README）；探针 54 项
- [x] 回归基线快照：固定 seed 下 forward/backward 输出存 JSON，CI 比对
  - 已实现 `baseline_model` / `snapshot_model` / `snapshot_digest` / `save_snapshot` / `load_snapshot` /
    `compare_snapshots` / `compare_snapshot_file`；快照是纯 JSON（forward + 逐层梯度），
    比较器带路径信息，键结构差异一律判失败。
  - CI 用摘要常量当金标准（`TestBaselineSnapshot.GOLDEN`），不额外增加仓库文件；
    探针 50 项 + `TestBaselineSnapshot` 10 例。
  - 已知边界：不覆盖 BN/Dropout 随机状态、不做层间接口校验（分别由 `grad_check_all` 和集成测试负责）。
- [x] 集成测试（toy 数据集训练到达标）；doctest；属性测试（随机输入验证不变量）
  - 集成：XOR 收敛到 1e-34、三分类 ≥95%、4→2→4 自编码器重建 1e-6、CausalLM loss < 0.0004、
    `grad_check_all` 全过；踩过的坑是默认 `val_split=0.1` 会把 4 个样本里的 1 个切走，
    导致「训练损失极小但有一个样本永远没学会」。
  - doctest：`run_doctests`（带 `require_examples`，空模块报错而不是静默返回 0）+ 5 条示例；
    另有一条故意写错示例的反例测试，断言确实被抓到（依赖 `sys.modules` 注册）。
  - 属性测试：12 条不变量，固定 seed 的 `random.Random`，不引入第三方库。



### W19 工程化收尾（§14 剩余）
- [x] 实验追踪（JSON：配置/指标/曲线/耗时）；模型元数据（created_at/git_hash/config/metrics）
  - 已实现 `ExperimentTracker`（context manager、异常自动记 failed、文件名带序号不覆盖）、
    `load_experiment` / `summarize_experiments` / `model_metadata` / `save_model_metadata` /
    `load_model_metadata` / `git_hash`（取不到返回 None，不抛异常）。
  - 口径：指标只收数字（拒绝 bool）；`clock` 可注入以便测试断言耗时；不做可视化面板。
  - 探针 55 项 + `TestExperimentTracking` 8 例。
- [x] CLI 子命令：train / predict / eval / export / finetune
  - 已实现 `ai_cli.py`（新文件，纯 `argparse` + `json`）：五个子命令、三种数据形状、
    成功 0 / 出错 2 的退出码约定，stdout 只有一行 JSON，库里的中文提示走 `redirect_stdout` 收进缓冲。
  - `train` 走 `quick_train`，`finetune` 直接改 `model.optimizer.lr` 后 `fit`，
    `export` 写 `{"format": "ai-export-1", "model", "metadata"}`（元数据复用 `model_metadata`）。
  - 探针 12 项（`C:\Users\Jiang\AppData\Local\Temp\probe_w19b.py`）+ `TestCLI` 11 例；
    全库 1705 -> 1716 passed。
- [x] 配置支持 YAML/TOML；`logging` 集成；内存监控（tracemalloc）；模型注册表/版本管理
  - 已实现 `load_config` / `save_config` / `parse_config_text` / `dump_config` +
    自带 `parse_simple_yaml` / `dump_simple_yaml`（YAML 子集：嵌套映射、块列表、标量、注释；
    不支持锚点/多文档/块标量）；TOML 走标准库 `tomllib`（只读），INI 走 `configparser`。
  - `setup_logging` / `get_logger` / `log_metrics`：重复调用不叠加 handler，指标只收数字。
  - `memory_usage` / `track_memory`：`tracemalloc` 当前值与峰值，退出恢复调用方追踪状态。
  - `ModelRegistry`：`register`（`Model` 或 JSON 路径）/ `versions` / `latest` / `get` /
    `describe` / `rollback`，每次登记写 `<name>/v<n>.json` + `index.json`（含 sha256），旧版本不删。
  - 真 bug（探针抓到）：`register` 原先先 `setdefault(name, [])` 再校验，登记失败会在索引里
    留一个空壳模型名；改为「先解析 payload 再动索引」。
  - 探针 40 项（`C:\\Users\\Jiang\\AppData\\Local\\Temp\\probe_w19c.py`）+ `TestConfigRegistry` 11 例；
    全库 1716 -> 1727 passed。

### W20 研究前沿（§17，文档 + 最小演示）
- [x] DP-SGD（梯度裁剪 + 高斯噪声 + 隐私预算记账）
  - `clip_grads_global`（全局 L2 范数，跨层合并）→ `dp_grads`（裁剪后加 `N(0, (σ·C)²)`）、
    `squeeze_grads` 摊平工具；`PrivacyAccountant` 用**简单组合**上界
    `eps_step = q·sqrt(2 ln(1.25/δ))/σ`，`advanced=True` 取高级组合与简单组合的较小者。
  - 口径写在 README：这是偏松上界，**不是** RDP / 矩记账，不与 opacus 对数字。
  - 探针 43 项 + `TestPrivacyAndMembership` 9 例；全库 1727 -> 1736 passed。
- [x] 成员推断攻击与防御（最简版）
  - membership_inference_attack（按损失阈值判成员，阈值取 balanced accuracy 最优；
    返回 accuracy / tpr / fpr / Mann-Whitney AUC）与 membership_defense_delta
    （成员/非成员损失差与攻击 AUC 的防御前后对照），与 DP-SGD 同一批交付
  - 口径：单特征阈值攻击，不是 shadow-model 训练攻击；README 写明它是「下界演示」
- [x] Lottery Ticket（迭代剪枝 + 重初始化）
  - weight_mask 跨层统一阈值 + 累积掩码（第二轮起只在保留集合里排序）
  - masked_reinit 掩码内 He 初始化，掩码外清零，偏置不动
  - lottery_ticket 的 keep_ratio 是乘性收缩（0.8 后 0.64），不是每轮重新挑
  - 真 bug：早期用「全量重算掩码再取交集」，零权重参与排序，第二轮比例卡住不动

- [x] Mixture-of-Depths / Early Exit
  - `mix_of_depths_route` 容量 `ceil(n·ratio)`、同分按下标、权重在入选内归一
  - `apply_mod_block` 入选相加 / 未入选透传，返回新列表不改入参
  - `early_exit` 逐样本首达阈值即退，否则最后一层；概率和必须为 1，不静默归一
  - `early_exit_savings` 给平均深度与相对节省；布尔校验用 `isinstance(bool)`
    （`x in (True, False)` 会放过整数 1，这是本批抓到的真 bug）
- [x] 神经符号（规则约束的候选过滤）
  - `RuleSet` 每条规则是 `(名字, 谓词(prefix, token))`，名字唯一、谓词可调用
  - `check` 逐位置列出违规规则，`mask`/`filter` 给单步可用候选
  - `apply_rule_mask` 全屏蔽时报错（避免下游 softmax 变 NaN）
  - `constrained_generate` 是解码期过滤，不是可微符号层（docstring/README 写明）
- [x] 文档化：NTK / 双下降 / grokking 作为「已知边界」——`empirical_ntk`（有限宽经验核，线性模型闭式 `xᵀx+1`）、`ntk_kernel_regression`（核视角闭式解，不等价于 SGD）、`double_descent_curve`（`p≈n` 处 test_mse 峰 11.94、`p>n` 二次下降到 0.48）；grokking 只解释机制不跑实验

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

### 5.1 量级拆分（自下而上估）

单元不是同一量级，按三类分别估：

| 类别 | 判据 | 数量 | 速度 | 会话数 |
|---|---|---|---|---|
| **S** | ≤30 分钟：纯公式，几十行 + 一个测试 | 约 100 个（调度器 12、初始化 6、回归/分割损失 12、指标 50、激活 5、GLU 1、增强 4、Scaler/Encoder 8、池化 8、位置编码 4） | 12–15 个/会话 | 7–8 |
| **M** | 1–3 小时：新模块带 `forward`/`backward` + 数值验证 | 约 85 个（归一化 5、视觉块 6、标注头 2、偏好/检测损失 10、优化器 10、GNN 8、MoE 1、分词器 3、RL 算法 8、可解释性 10、数据管道 10） | 3–4 个/会话 | 21–28 |
| **L** | 半天以上：新子系统或需重新设计数据流 | 约 30 个（ViT、U-Net、FPN、Swin、MLP-Mixer、4 个线性注意力、FlashAttention、约束解码、投机解码、VQ-VAE、Flow Matching、GAN 变体 6、RL 环境 2、CLIP、MAE、PCA/t-SNE、HNSW/检测/OCR） | 1–2 个/会话 | 15–20 |

**Python 主体合计 43–56 个会话**（中位约 50）。

### 5.2 加上跨语言同步与 Web

| 项 | 估算 |
|---|---|
| C 侧 A 档同步（13 项欠账 + 各波 A 档，约 45 个单元） | +11–14 会话 |
| C 侧 B 档（MoE / 标注头 / 分词器 / 生成模型最小版） | +4–6 会话 |
| Web（采样扩展、约束解码、prefix cache） | +1 会话 |

**总计 59–77 个会话，中位约 68**；若把 C 的 B 档推迟，实际落在 **55–65**。

### 5.3 换算成日历时间

| 节奏 | 全部完成 | 只做核心（见下） |
|---|---|---|
| 1 会话/天（晚上 1–2 小时） | 约 2–2.5 个月 | 约 2 周 |
| 2 会话/天 | 约 1 个月 | 约 1 周 |
| 3–4 会话/天（全天） | 约 3 周 | 3–5 天 |

**最小可用路径**（不做视觉骨干/图/RL/领域应用，只补「现代训练 + LLM」）：
W0 + W1 + W3 + W9 + W13 + W2 的部分 = 约 8–10 个会话，就能让这个库在
「损失 / 优化器 / 调度器 / 指标 / 归一化与激活」上不再有短板。

每个会话结束都保证：测试全绿、demo exit 0、`probe_default_paths.py` 无差异、工作区干净。

---

## 6. 风险与对策

1. **范围爆炸**：每波开始时先把该波的单元数写进清单，超出就拆波，不硬塞。
2. **默认路径漂移**：任何改动都要跑 `probe_default_paths.py`；它比对的是上一个提交的
   `AI.py`（`git show HEAD:AI.py`），一旦出现非零差异就回退。
3. **测试变慢**：新测试一律小规模（≤200 步、d_model ≤ 32），慢测试标 `@pytest.mark.slow`。
4. **README 计数漂移**：已在提交前固定检查（正文 + 文件结构树两处）。
5. **C/Web 失同步**：按第 7 节的三档分级执行——A 档（纯标量数学）必须当场同步，
   B 档排进同波后半段，C 档明确不同步并写进 `c/README.md` 的已知边界。


---

## 7. 跨语言同步策略（C / Web）

### 7.1 三档分级

| 档 | 判据 | 处理 |
|---|---|---|
| **A** | 纯标量数学：激活、损失、优化器、调度器、指标、初始化、采样参数 | 当场同步；容差 0 |
| **B** | 需要新的张量布局或新子系统：`(C,H,W)` 归一化、MoE、序列标注头、生成模型、分词器、LoRA | 排进同波后半段；容差 0（单步）/ 1e-12（多步轨迹） |
| **C** | 只跟宿主环境有关：数据集管道、CLI / 日志 / 配置、剖析与可视化、超参搜索、图网络 / RL / 多模态 / 检测 | 不同步，写进 `c/README.md` 的已知边界 |

判据只有一条：**能不能用固定权重 + 固定输入，把 Python 的输出逐位复现**。能，就进 A/B；
不能（因为依赖宿主 RNG、文件系统、进程模型），就进 C。

### 7.2 同步流程（六步，缺一步不算完成）

1. Python 侧先实现并用探针验证（解析梯度 / 闭式解）；
2. 写 `c/tools/<topic>_reference.py`：固定 seed、显式导出权重与输入、把期望输出写进
   `c/<topic>_reference.json`；
3. C 侧实现（沿用 `numeric.h` 的 Neumaier 求和与既有模块风格，不引入新依赖）；
4. `c/tests/compare_<topic>.c`：注入同一份权重后逐项比对，**单步/前向容差 0**，
   多步训练轨迹允许 1e-12（msvcrt 与 ucrtbase 的 `exp`/`log` 有 1 ULP 差异）；
5. `c/tests/test_<topic>.c` 自测（形状、边界、非法输入），并确认 11 个既有二进制不回归；
6. `c/README.md` 更新（状态表 / 模块表 / 验证节 / 已知差异），两仓分别提交。

### 7.3 各波同步档位

| 波次 | C 侧 | Web 侧 |
|---|---|---|
| W0 训练基础设施 | C（回调/续训是宿主概念） | C |
| W1 损失 | **A**（L1/LogCosh/Quantile 等回归族、SoftCE 等） | C |
| W2 卷积与视觉块 | **A**（池化家族、DepthwiseSeparable、ResNet 块）+ B（Conv3d/Transpose3d） | C |
| W3 归一化·激活·初始化 | **A**（初始化族、激活族、GLU、残差变体）+ B（WeightNorm/SpectralNorm） | C |
| W4 视觉骨干 | B（ViT/U-Net 需新图结构） | C |
| W5 注意力与位置编码 | **A**（YaRN/NTK-RoPE 改 `rope_freqs`、xPos、T5 偏置）+ B（线性注意力族） | B（页面只做可展示的采样/偏置项） |
| W6 图神经网络 | C | C |
| W7 LLM 微调与推理 | B（LoRA/DoRA、约束解码、序列打包）+ A（采样扩展） | **A/B**（采样扩展、约束解码） |
| W8 生成模型 | B（VAE/GAN/DDPM 各挑最小可对拍版本） | C |
| W9 优化器与调度 | **A**（全部） | C |
| W10 正则与训练技巧 | B（DropBlock/DropConnect、对抗训练） | C |
| W11 数据管道 | C | C |
| W12 分词与批处理 | B（BPE/WordPiece、padding/mask） | C |
| W13 指标扩展 | **A**（全部，纯函数） | B（困惑度/BLEU 展示） |
| W14 可解释性 | C（可视化是宿主能力） | C |
| W15 推理与部署 | B（结构化剪枝、QAT、集成） | C |
| W16 自监督·多模态·RL | C | C |
| W17 领域应用 | C（除时序 1D 卷积） | C |
| W18 测试与数值稳定 | **A**（稳定性套件、基线快照可共用 JSON） | C |
| W19 工程化收尾 | C | C |
| W20 研究前沿 | C | C |

### 7.4 C 侧欠账清单（Python 已有、C 尚未同步）

按「先补最常用的」排序，每项都走 7.2 的六步：

- [ ] **AdamW**（C 目前只有 Adam / SGD / SGDMomentum / RMSprop）
- [ ] Lion / Adafactor
- [ ] WarmupLR / ReduceLROnPlateau（C 只有 Step / Exponential / Cosine）
- [ ] RMSNorm
- [ ] `silu` / `mish` / `elu` / `selu`（扩 `act_apply` 枚举）与 GELU 版 FFN
- [ ] 损失枚举补齐：先核对 `LossKind` 现有成员，再补 focal / bce / bce_with_logits / huber
- [ ] 指标模块：accuracy / precision / recall / f1 / confusion_matrix / mae / rmse / mape / top-k / roc_auc
- [ ] `Conv2d` 的 `groups` / `dilation`；`Conv1d` / `ConvTranspose2d`
- [ ] `BatchNorm2d` / `GroupNorm` / `InstanceNorm` / `DropPath`
- [ ] Mixup / CutMix / Cutout / RandAugment（纯张量运算，A 档）
- [ ] MoE / 序列标注头 / 多标签头（B 档）
- [ ] BPE / WordPiece 分词器（B 档）
- [ ] LoRA、EMA / SWA / Lookahead（等 Python 侧 W9 完成后同步）

### 7.5 Web 同步范围

页面是一个「单层因果 LM 教学页」，能同步的只有 LLM 相关能力：

- 已同步：因果掩码、KV cache、`top_k`/`top_p`/`min_p`、五种惩罚、滑动窗口、`ALiBi`、
  模型存档/加载、困惑度、输入校验（66→87 checks）
- 待同步（跟着 W7 走）：typical / mirostat / eta / top-a / tail-free / contrastive 采样、
  约束解码（JSON Schema / 正则）、Prefix/Prompt cache 的可视化
- 不同步：网络层扩展、图 / RL / 多模态、数据管道（页面没有「数据集」概念）

### 7.6 成本影响

A 档每项约为 Python 同类工作的 0.8–1.2 倍工时（实现 + 参考导出 + 对拍 + 自测 + 文档），
B 档 2–3 倍。折算后 A 档 +11–14 会话、B 档 +4–6 会话（见 5.2）；欠账清单（7.4）
按「AdamW → 损失 → 指标 → Conv 扩展 → 归一化 → 分词器」顺序补，约 **4–6 个会话**。
优先级不变：A 档不允许欠账，B 档允许落后一个波次。
