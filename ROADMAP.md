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
- [ ] `fit(train_augment=...)`：**只接了 Mixup**（`mixup_alpha=`，分类 + 索引型损失）；
      CutMix/Cutout 需要「先卷积再混合」的图像语义，随 W10 的增强族一起接

### W1 损失：回归与偏好对齐（P0）
- [x] 回归族：`l1_loss`、`log_cosh_loss`、`quantile_loss`（= pinball）、`smooth_l1`（= `huber_loss` 别名，逐位等价）（`c37cdf5`；梯度差分 ≤2.6e-10，`quantile(0.5)` 逐位等于 `l1/2`）
- [x] 偏好对齐族：`preference_loss(chosen_logp, rejected_logp, ref_..., method=)` 支持
      DPO / IPO / KTO / ORPO / SimPO / CPO + 单边 `kto_loss(logp, ref_logp, desirable=)`
      （W1-D 提交；五种 method 闭式解与双向中心差分吻合 ≤1e-8，`β=50` 极端情形 loss < 1e-12 不溢出；
      ORPO 在 logit 空间做、`p>=1` 报错；**不进 `_LOSS_MAP`**——吃一对 logp 而非逐样本 `(pred, target)`）
- [x] 蒸馏：`distill_loss(student_logits, teacher_logits, T=2.0)`（`T²·KL`，对学生 logits 的梯度 `T·(p_s−p_t)`）+ `hint_loss(student_feat, teacher_feat, weight)`（隐藏层均方误差）；`_LOSS_MAP` 增 `'distill'`、输出层要求 `linear`（梯度差分 ≤9.4e-10，`T=1` 时与手算 KL 逐位一致）
- [x] 类别不平衡族：`ohem_loss`（批级，不进 `Model`）、`class_balanced_loss`、`ldam_loss`；`soft_ce` **不再单独实现**——`cross_entropy_loss` 已支持任意软标签与 `label_smoothing`，再写一个只是重复（W1-B 提交；CB/LDAM 梯度差分 ≤3.5e-9）
- [x] 多任务权重：`uncertainty_weighting`（Kendall，`d_log_vars = 1 - exp(-s_i)·L_i`）、`gradnorm_weights`（`target_i = Ḡ·r_i`，权重归一化均值 1；只有 `r_i ≡ 1` 时才有不动点，一般情形用返回的 `gradients = sign(G-target)·G0/r`）、`pcgrad`（两两冲突投影，嵌套 list 支持）；W1-E 提交，双向差分 worst_rel 1.54e-10，11 类非法输入全报错
- [ ] 已有项标注：`huber_loss`（SmoothL1）、`kl_divergence`、`dice_loss`、`ctc_loss`、`triplet_loss`、`info_nce_loss`

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
- [x] 残差变体：Pre-LN / Post-LN / ReZero / SkipInit / Highway（ReZero/SkipInit/PreNormResidual/PostNormResidual/Highway；dx/参数梯度中心差分 ≤1e-6，ReZero 的 lpha=0 逐位恒等；顺带补了 _dense_param_count/_dense_apply_sgd/_dense_apply_adam/_norm_* 六个助手，因为 Layer/LayerNorm 本身没有 param_count()/sgd_step()）

### W4 视觉骨干（P0/P1）
- [x] `PatchEmbedding`（`Conv2d(kernel=stride=p)` + 行优先摊平，`backward` 还原特征图；`dx`/`conv.grad_w` 中心差分 ≤1e-7）+ `ViT`（CLS token + 位置编码 + 2 层编码器 + 分类头；`dx` 与 CLS/`pos`/head/`LayerNorm` 四组梯度中心差分 ≤1e-7；梯度累加语义与 `zero_grad`；2 类小任务 15 轮收敛并全对；只有 `sgd_step`，与 Transformer 系列一致）
- [x] `UNet`（2 级下采样 + 瓶颈 + 2 级上采样 + 通道拼接跳跃；可选 BN；`dx` 与编码/转置卷积/BN 三组参数梯度中心差分 ≤1e-9；Adam 40 步 MSE 0.67 → <0.22）
- [x] `FPN`（1x1 横向 + 最近邻上采样相加 + 3x3 平滑；逐级 `dx` 与横向/平滑卷积 `grad_w` 中心差分 ≤1e-9，要求相邻尺寸严格 2 倍）
- [x] `SegNet`（简化 2 级；解码用 `MaxPool2d.mask` 的**窗口内相对位置**+ stride 换算出绝对坐标做 unpool —— 直接当绝对位置用会让 `dx` 差 4 倍，`_unpool2`/`_unpool2_back` 已按此修；`dx` 与 `conv1` 梯度中心差分 ≤1e-7，40 步 Adam loss 1.01 → 0.02）
- [x] `SwinTransformer`（简化）：单 stage、交替普通/移位窗口；移位用循环 roll 且**不做 mask**；窗口内注意力手写（不复用 `MultiHeadAttention`，避免其单份 cache 被逐窗口调用覆盖）；`dx` 1.51e-09、block 内 q/k/v/o/f1/f2 逐项 ≤1e-5、150 步 Adam 收敛，898 passed
- [x] `MLPMixer`：token 混合 + 通道混合，两段 pre-LN 残差；**踩过的坑**：同一 `Layer`/`LayerNorm` 实例在循环里逐通道/逐 token 调用会互相覆盖 cache（`Layer.backward` 只读最后一次的 `input`/`z`），实测 `t1` 解析梯度 ≈ -1.19e-18 而数值是 -0.242284；改用按行批量的 `_dense_rows_forward/backward`、`_norm_rows_forward/backward` 后，block 内四组权重、两级 LayerNorm、head、`dx` 的中心差分全部 ≤1e-5 相对误差，893 passed
- [x] `AttentionPooling`（`q` 可学，`q=0` 退化为平均；`dx`/`grad_q` 中心差分 ≤1e-9）、`GeMPooling`（`p=1` 平均 / `p=2` 均方根 / `p=2000` 仍不溢出，log 空间实现）

### W5 注意力与位置编码（P1/P2）
- [ ] `MLA`（低秩 KV 压缩）；`MQA` 标注为 `num_kv_heads=1` 特例
- [x] `LearnedPositionalEncoding`（可学习位置编码表，`out=x+table`，`sgd_step`/`adam_step`，超长自动扩表）
- [x] RoPE 长度外推 `rope_inv_freqs`：`linear`（频率 /factor）、`ntk`（`base·factor^(d_k/(d_k-2))`）、`yarn`（按 `rotations=original_max·f/(2π)` 与 `beta_fast/beta_slow` 做 smoothstep 分段）；`MultiHeadAttention(rope_scaling=..., rope_factor=..., rope_original_max=...)` 接线，默认 None 与旧版逐位一致；缩放后 `W_q/W_k/W_v/W_o` 梯度与中心差分 ≤1e-6（未实现 YaRN 的 mscale 温度，docstring 注明）
- [ ] 位置编码剩余：T5 相对位置偏置、xPos、MLA 低秩 KV 压缩
- [x] `LinearAttention`（`phi=elu+1`，O(T·d²)；`causal=True` 用前缀和、非 causal 用全长和；
      `W_q/W_k/W_v/W_o` 与输入梯度中心差分 ≤1e-5，120 步 Adam 收敛）
- [ ] 线性注意力族剩余：`Performer`（随机特征）、`Linformer`（低秩 K/V）、`BigBird`（稀疏）、
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
