# Python-standard-library-only-AI

纯 Python 标准库实现的神经网络工具箱，**零第三方依赖**。

- **运行环境**: Python **3.12+**（3.12 起 `sum()` 改用 Neumaier 补偿求和，C 版引擎
  逐位复刻了这一行为；3.10/3.11 上跨语言对拍的末位会不一致）
- **CI**: GitHub Actions 在 Python 3.12 上跑 `flake8` + `pytest`（1040 项）

## 特性

- **零第三方依赖**: 仅依赖 `math`、`random`、`json` 等标准库
- **全栈组件**: 从全连接层到 CNN、RNN/LSTM/GRU、Transformer 一应俱全
- **双模式 API**: 底层精细控制 + 高层 sklearn 风格封装
- **完整的训练管线**: Early Stopping、梯度裁剪、学习率调度、模型保存/加载
- **反向传播有数值梯度测试守着**: `Layer` / `MultiHeadAttention` / `Conv2d` / `BatchNorm1d` 的解析梯度都与数值梯度逐项比对
- **因果注意力 / LLM**: `MultiHeadAttention(causal=True)` 因果掩码 + `CausalLM`（GPT 式字符级语言模型，训练与生成全部纯标准库）；注意力权重 dropout 同样实现了反向
- **现代 LLM 组件**: GELU / sigmoid 激活、`RMSNorm`、`Linear`、`AdamW`（解耦权重衰减）、`WarmupLR`、
  RoPE 旋转位置编码、KV cache 生成（默认开启，比整段重算快约 18 倍且结果逐位相同），
  以及 Focal / BCE / BCEWithLogits / Huber / KL 与 label smoothing；
  这些损失都能通过 `Model(loss_fn='focal', loss_kwargs={...})` 直接用，配错输出层激活会在构造时报错
- **LLM 全家桶**: `Seq2SeqLM`（encoder-decoder + teacher forcing）/ `MaskedLM`（BERT 式）、GQA/MQA、滑动窗口、ALiBi、SwiGLU、LoRA、Lion / Adafactor、以及 min-p / 重复·存在·频率惩罚 / 禁重复 n-gram 五种采样控制
- **训练工程化**: 梯度累积、断点续训（含优化器与损失配置）、分层切分、K 折、`StandardScaler` / `MinMaxScaler`、配置驱动训练、超参搜索、逐层性能剖析、学习率扫描
- **训练基础设施**: `set_seed` / `Callback` 回调（4 个钩子）/ `infer_shapes` /
  `predict_batch` / `activation_stats` / `dead_units` / `resume_from` 续训 /
  `record_layer_norms` 逐层梯度范数 / `fit(mixup_alpha=)`
- **评估与诊断**: 14 个纯标准库指标（含混淆矩阵 / PRF / MCC / AUC / BLEU / ROUGE-L）、`grad_check` 梯度自检、`estimate_flops`、`ascii_plot`、注意力可视化
- **更多模型**: 序列标注头与多标签头、`MoE` 专家混合、VAE / GAN / DDPM 三个生成模型
- **分词器**: 字节级 BPE（任意 UTF-8 无损）与 WordPiece（`##` 子词 + 最长前缀匹配）

## 快速开始

```python
import random
from AI import *

random.seed(42)   # 示例固定随机种子，输出可复现

# ── 分类：一行训练，零配置 ────────────────────
x = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9],
     [0.15, 0.25, 0.35], [0.45, 0.55, 0.65], [0.75, 0.85, 0.95]]
y = [0, 1, 2, 0, 1, 2]

model = quick_train(x, y, task='classification', epochs=300, verbose=False)
print(model.score(x, y))          # 准确率 → 1.0

# ── 回归：目标 y = [x0², x0·x1] ───────────────
vals = [0.2 + 1.1 * i / 19 for i in range(20)]        # 0.2 ~ 1.3
x_reg = [[v, v * 1.5] for v in vals]
y_reg = [[v * v, v * v * 1.5] for v in vals]

random.seed(42)
model_reg = make_regressor(input_dim=2, output_dim=2, hidden_layers=(32, 32), lr=0.03)
model_reg.fit(x_reg, y_reg, epochs=1500, patience=100, verbose=False)
print(model_reg.predict([0.6, 0.9]))   # → [0.3605, 0.5393]（真值 [0.36, 0.54]，R² ≈ 0.9999）
```

## 高层 API

### Model 类（sklearn 风格）

```python
# 手动构建（5 维输入 → 2 维输出）
x_train = [[0.1, 0.2, 0.3, 0.4, 0.5], [0.2, 0.3, 0.4, 0.5, 0.6],
           [0.3, 0.4, 0.5, 0.6, 0.7], [0.4, 0.5, 0.6, 0.7, 0.8]]
y_train = [[0.5, 1.0], [0.7, 1.4], [0.9, 1.8], [1.1, 2.2]]

model = Model(MLP([5, 16, 8, 2]), loss_fn='mse', optimizer='adam', lr=0.01)
model.summary()          # 打印模型结构
history = model.fit(x_train, y_train, epochs=300, batch_size=2, patience=30,
                    lr_scheduler=StepLR(model.optimizer, step_size=50, gamma=0.5),
                    grad_clip=1.0)
pred = model.predict(x_train)
score = model.score(x_train, y_train)   # 分类=准确率 / 回归=R²（多输出时对各输出维度取平均）

# 保存 & 加载
model.save("model.json")
model2 = Model.load("model.json")
```

### 模型瘦身与蒸馏

| 方法 / 函数 | 说明 |
|------------|------|
| MLP.param_count() | 可训练参数总数 |
| MLP.sparsity() | 权重中为 0 的比例 |
| MLP.prune_magnitude(threshold) | 幅度剪枝：\|w\| < 阈值置 0，返回剪掉个数 |
| MLP.quantize_weights(bits) | 全局 min/max 均匀分档定点量化，返回步长 |
| MLP.merge_weights(other, w1, w2) | 权重线性合并（K 折权重平均用，**各折需同初值**） |
| distill(teacher, student, x, y) | 蒸馏：用教师输出当软目标训练学生，返回一致率 |

`fit()` 新增 `early_stop_metric='val'|'train'`：默认盯验证损失；
选 'train' 时盯训练损失，**且不依赖验证集**（原来没有验证集时 patience 完全不生效）。

### 工厂函数

| 函数 | 说明 |
|------|------|
| `make_classifier(input_dim, num_classes, ...)` | 创建分类模型（Softmax + CrossEntropy） |
| `make_regressor(input_dim, output_dim, ...)` | 创建回归模型（Linear + MSE） |
| `quick_train(x, y, task, ...)` | 一行代码自动创建并训练；`seed=` 同时固定权重初始化与每轮打乱（可完全复现，且不推进全局 RNG） |

### fit() 参数

| 参数 | 说明 |
|------|------|
| `epochs` | 训练轮数 |
| `batch_size` | 批次大小（-1 = 全批量） |
| `val_split` / `val_data` | 验证集比例或自定义验证集（见下方说明） |
| `patience` | Early Stopping 耐心值（0 = 关闭） |
| `lr_scheduler` | StepLR / ExponentialLR / CosineAnnealingLR / ReduceLROnPlateau（后者要传指标：`sched.step(val_loss)`） |
| `grad_clip` | 梯度范数裁剪阈值 |
| `checkpoint_path` / `checkpoint_every` | 每 `checkpoint_every` 轮把模型（含优化器状态）存到该路径，用于断点续训 |
| `grad_accum` | 梯度累积步数（1 = 每个 batch 更新一次）；>1 时累加若干 batch 的平均梯度再更新，等效放大 batch 而不占额外内存 |
| `seed` | 打乱数据的随机种子（None = 用全局 `random`，与旧版行为一致；传整数则改用局部 `Random`：同样的 `fit` 再跑一次结果完全相同，也不会推进调用方的全局随机状态） |
| `callbacks` | 单个 `Callback` 或列表；四个钩子 `on_train_begin/on_epoch_begin/on_epoch_end/on_train_end`，钩子抛异常会直接中断训练 |
| `resume_from` | 从 checkpoint JSON 恢复权重与优化器状态（`epochs=0` 即“只恢复不训练”）；层数或形状不符直接报错 |
| `mixup_alpha` | >0 时对每个 batch 做 Mixup（Beta(α, α) 混合样本与软标签）；只支持分类任务 + 索引型损失，默认 0 关闭 |
| `record_layer_norms` | 记录每层梯度范数到 `history['layer_grad_norm']`；只记录、不影响训练数值 |

训练结束后 `model.history` 里有三个键：`train_loss`、`val_loss`、`grad_norm`（`record_layer_norms=True` 时多一个 `layer_grad_norm`，每轮一个逐层列表）
（每轮平均梯度范数，裁剪前；用来判断训练是否发散）。

`val_data` 支持三种写法（会校验维度，维度不符直接报错）：

```python
model = make_regressor(2, 1, hidden_layers=(8,))
x_train = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]]
y_train = [[0.3], [0.7], [1.1], [1.5]]
x_val, y_val = [[0.15, 0.25], [0.35, 0.45]], [[0.4], [0.8]]

model.fit(x_train, y_train, val_data={'x': x_val, 'y': y_val})   # 显式并行数组（推荐，无歧义）
model.fit(x_train, y_train, val_data=(x_val, y_val))             # 并行数组
model.fit(x_train, y_train, val_data=[([0.15, 0.25], [0.4]), ([0.35, 0.45], [0.8])])  # 样本对列表
```

> 说明：恰好 2 个元素且结构上无法区分时，`list` 按「样本对」解释、`tuple` 按「并行数组」解释；拿不准就用 dict 写法。

## 底层组件

### 层 (Layers)

| 组件 | 说明 |
|------|------|
| `Layer` | 全连接层 + 激活函数（He 初始化）；激活可选 `leaky_relu` / `relu` / `gelu` / `sigmoid` / `silu` / `mish` / `elu` / `selu` / `linear` / `softmax` |
| `Linear` | 无激活的全连接层（`Layer(..., activation='linear')` 的语义化子类） |
| `MLP` | 多层感知机容器 |
| `Conv2d` | 二维卷积层（`groups=G` 支持分组/深度卷积，`dilation=D` 支持空洞卷积） |
| `Conv1d` | 一维卷积层（`(C, L)` 输入，滑动窗口直接累加） |
| `ConvTranspose2d` | 二维转置卷积（反卷积；`output_padding` 与 PyTorch 同规则） |
| `MaxPool2d` | 二维最大池化 |
| `AvgPool2d` | 二维平均池化（接口同 `MaxPool2d`，反向把梯度均分回窗口） |
| `GlobalAvgPool2d` | 全局平均池化：`(C,H,W) → (C,)`，CNN 分类头常用 |
| `AdaptiveAvgPool2d` | 自适应平均池化：`(C,H,W) → (C,out_h,out_w)`，任意输入尺寸都能固定输出（PyTorch 分箱规则，`(1,1)` 等价于 `GlobalAvgPool2d`） |
| `Flatten` | 展平（CNN → MLP） |
| `PatchEmbedding(in_channels, patch_size, d_model)` | ViT 的第一步：用 `kernel=stride=patch_size` 的卷积把 `(C,H,W)` 切成不重叠 patch，摊平成 `gh*gw` 个 token（行优先）；`backward` 把 token 梯度还原成特征图再回传 |
| `AttentionPooling(d_model)` | 注意力池化：可学习查询向量给每个 token 打分 `softmax(q·x/√d)`，加权求和成一个向量；权重由内容决定，能学会"只看重要补丁" |
| `UNet(in_channels=1, num_classes=2, base_channels=4, use_bn=False)` | U-Net：2 级下采样 + 瓶颈 + 2 级上采样，每级把上采样结果与同级编码特征**按通道拼接**；`use_bn=True` 时每个卷积后接 `BatchNorm2d`（`train()`/`eval()` 切换）；输入 `(C,H,W)` 要求 H、W 能被 4 整除；`sgd_step`/`adam_step` 可用 |
| `ViT(in_channels=3, image_size=8, patch_size=4, d_model=16, num_heads=4, d_ff=32, num_layers=2, num_classes=10)` | 视觉 Transformer：`PatchEmbedding` 切 patch → 可学习 `CLS` + 位置编码 → `TransformerEncoder` → 取 `CLS` 过 `LayerNorm` → 线性分类头；`image_size` 支持 int 或 `(H,W)` 且必须能被 `patch_size` 整除；梯度**累加**（与编码器一致），更新前先 `zero_grad()`，`sgd_step(lr)` 一步 SGD |
| `FPN(in_channels_list, out_channels=8, smooth=True)` | 特征金字塔：每级先过 1x1 横向卷积压到统一通道，再从最粗一级往上 `P_i = lateral_i + upsample2x(P_{i+1})`，每级过 3x3 平滑卷积（`smooth=False` 可关）。要求相邻特征尺寸严格 2 倍关系；`backward` 返回与输入等长的梯度列表 |
| `SegNet(in_channels=1, num_classes=2, base_channels=4)` | 2 级编码-解码分割网络：编码 `Conv+ReLU+MaxPool`，解码用**池化索引**（`MaxPool2d.mask` 的 argmax 位置）把值放回原位置再卷积，不做跳跃拼接；要求输入 `H`、`W` 能被 4 整除 |
| `MLPMixer(in_channels=3, image_size=8, patch_size=4, d_model=16, num_layers=2, token_hidden=32, channel_hidden=32, num_classes=10)` | MLP-Mixer：每个 block 先做 **token 混合**（转置后逐通道跑同一个 MLP），再做**通道混合**（逐 token 跑 MLP），两段都是 pre-LN 残差；输出对 token 取均值池化。逐行批量前向/反向（cache 按行保存，不会互相覆盖），`backward` 梯度累加、`sgd_step`/`adam_step` 都有 |
| `SwinTransformer(in_channels=3, image_size=8, patch_size=4, d_model=16, num_heads=4, d_ff=32, num_layers=2, window_size=2, num_classes=10)` | **简化版 Swin**：PatchEmbedding → 交替（普通 / 移位 `window_size//2`）窗口块 → 全局均值池化 → 线性头。窗口内注意力手写（共享 `q/k/v/o` 投影，按窗口切分后做缩放点积 + softmax），逐行批量 helper 保证 cache 不互相覆盖。**简化点**：单 stage、无 patch merging、移位用循环 roll **且不加 attention mask**（真实 Swin 会挡住跨窗口部分） |
| `GeMPooling(p=3.0)` | 广义平均池化：`(mean(x^p))^(1/p)`，`p=1` 是平均、`p→∞` 趋近最大；前后向都在 log 空间算，`p` 取到几千也不溢出 |
| `Dropout` | Dropout 正则化（**推理前必须调 `eval()`**，见下方「训练 / 推理模式」） |
| `GlobalMaxPool2d` | 全局最大池化：`(C,H,W) → (C,)`，反向只回传给最大值位置 |
| `AdaptiveMaxPool2d` | 自适应最大池化：`(C,H,W) → (C,out_h,out_w)`，分箱与 `AdaptiveAvgPool2d` 一致（`(1,1)` 与 `GlobalMaxPool2d` 逐位相同） |
| `LPPool2d` | Lp 池化：`(Σ\|x\|^p / area)^(1/p)`；`p=1` 在非负输入上等于 `AvgPool2d`，`p` 越大越接近 `MaxPool2d` |
| `SEBlock` | Squeeze-and-Excitation 通道注意力：GAP → 瓶颈 MLP → sigmoid 通道权重乘回输入；自带 `grads`/`sgd_step`/`adam_step`/`param_count`（`CBAMBlock` 同样有这四个） |
| `CBAMBlock` | CBAM 注意力：通道支路（SE 式瓶颈 MLP）+ 空间支路（通道维 mean/max 池化 → k×k 卷积 → sigmoid），`forward((C,H,W)) -> (C,H,W)`，梯度键 `W1/b1/W2/b2/sp_w/sp_b` |
| `DenseBlock` | 稠密连接块：每层 `BN → ReLU → 3x3 Conv`，输出与输入在通道维拼接；`(Cin,H,W) → (Cin+growth*num_layers,H,W)`，BN 梯度收在 `bn_grads` |
| `ConvNeXtBlock` | ConvNeXt：`7x7 depthwise Conv → LayerNorm（逐位置跨通道）→ 1x1 升维 → GELU → 1x1 降维` + 残差 |
| `AdaptiveMaxPool2d(out_size)` | 自适应最大池化，bin 边界与 `AdaptiveAvgPool2d` 同一套规则；`(2)` 与 `MaxPool2d(2)` 逐位相同 |
| `LPPool2d(norm_type, kernel_size, stride=None)` | Lp 池化：`(Σ|x|^p/area)^(1/p)`，`norm_type=1` 退化为平均、趋向 ∞ 逼近最大 |
| `FractionalMaxPool2d(output_ratio=(0.5,0.5), rng=None)` | 分数最大池化：相邻窗口按比例重叠，起点在允许范围内随机（可注入 `random.Random` 复现） |
| `StochasticPool2d(kernel_size, stride=None, rng=None)` | 随机池化：按 `relu(x)/Σrelu` 采样一个位置输出原值（训练）或概率加权期望（`eval()`） |
| `ConditionalNorm(features, cond_dim)` | 条件归一化（CiM）：`gamma = 1 + W_g·c + b_g`、`beta = W_b·c + b_b`，反向同时给出 `grad_cond` |
| `ECABlock` | ECA 通道注意力：GAP 后用一条奇数宽的 1D 卷积（零填充）跨通道混权重，比 SE 少降维 MLP，只多 `kernel_size` 个参数 |
| `DepthwiseSeparableConv2d` | 深度可分离卷积：`Conv2d(groups=in_channels)` + 1x1 点卷积；参数量约为标准卷积的 `1/out_channels + 1/(k*k)` |
| `BasicBlock2d` | ResNet BasicBlock：`conv3x3 → BN → ReLU → conv3x3 → BN → (+x) → ReLU`（stride=1 恒等捷径），BN 梯度收在 `bn_grads` |
| `Bottleneck2d` | ResNet Bottleneck：`1x1 降维 → 3x3 → 1x1 升维` + 残差；`stride≠1` 或通道变化时用 `1x1 Conv + BN` 投影捷径 |
| `AdaIN` | 自适应实例归一化：按通道做实例归一化后用风格向量 `gamma`/`beta` 缩放平移，`set_style()` 换风格 |
| `FiLM` | 特征级线性调制：`y = gamma ⊙ x + beta`，`groups>1` 时把特征分段共享一组调制参数 |
| `ConvTranspose1d` | 一维转置卷积：`(Cin,L) → (Cout, (L-1)*s-2p+k+op)`，前向把每个输入点按 stride 摆到输出网格累加，反向对 dx 与 `grad_w` 做同样的散射 |
| `MaxPool1d` / `AvgPool1d` | 一维池化：`(C,L) → (C,(L+2p-k)//s+1)`；最大池化只回传最大值位置，平均池化把梯度均分回窗口 |
| `MaxPool3d` / `AvgPool3d` | 三维池化：`(C,D,H,W) → (C,Do,Ho,Wo)`，同样按窗口回传/均分，padding 位置不接收梯度 |
| `Conv3d` | 三维卷积：`(C,D,H,W) → (Cout,Od,Oh,Ow)`，三轴独立 kernel/stride/padding；直接滑动窗口累加（不做 im2col，小体数据更省内存） |
| `ConvTranspose3d` | 三维转置卷积：每轴 `(D-1)*s-2p+k+op`，前向按 stride 散射累加，`output_padding` 只改输出尺寸 |
| `DropPath` | 随机深度：整条残差分支按概率丢弃（inverted，`eval()` 关闭） |
| `ReZero(layer, init_alpha=0.0)` | ReZero 残差：`y = x + alpha·f(x)`，`alpha` 从 0 起可学 ⇒ 整块初始就是恒等映射，深层网络一开始不爆也不消 |
| `SkipInit(layer, num_blocks=1)` | SkipInit 残差：`y = x + beta·f(x)`，`beta = 1/√num_blocks` **固定不可学**（参数量与内层相同） |
| `PreNormResidual(layer, eps=1e-5)` | Pre-LN 残差：`y = x + f(LN(x))`，归一化在分支入口、主路是干净单位通路（主流 Transformer 摆法） |
| `PostNormResidual(layer, eps=1e-5)` | Post-LN 残差：`y = LN(x + f(x))`（Transformer 原始论文摆法，主路梯度要穿 LN） |
| `Highway(num_features)` | Highway 门控：`y = g ⊙ x + (1-g) ⊙ tanh(w ⊙ x + b)`，`g = sigmoid(W_g·x + b_g)` 逐特征门控且依赖整条输入 |
| `BatchNorm1d` | 批归一化：`forward` 同时接受单样本与 batch；训练+2D 走 batch 统计量，推理用 running 统计量 |
| `BatchNorm2d` | 二维批归一化：按通道在 H×W 上归一化，训练更新 running 统计量、`eval()` 用 running；`backward` 返回 `(dx, grad_gamma, grad_beta)`，**反向也分模式**（eval 时统计量来自 running，dx 为逐元素 `gamma/sqrt(var+eps)`） |
| `GroupNorm` / `InstanceNorm` | 分组归一化（`(C,H,W)`，小 batch 稳定）；`InstanceNorm` 是组数等于通道数的特例 |
| `LayerNorm` | 层归一化 |
| `RMSNorm` | RMS 归一化（Llama 系）：不减均值、无 beta，`y = gamma·x/sqrt(mean(x²)+eps)` |
| `ScaleNorm(num_features, eps=1e-6)` | 尺度归一化：只有标量 `g`，`y = g·x/(‖x‖/√d+eps)`（整向量缩放，不是逐特征归一化） |
| `WeightNorm(layer)` | 权重归一化：`w = g ⊙ v/‖v‖_row`，把权重的尺度与方向解耦；反向精确传到 `v`/`g`/`b`（含 `dv` 里的解耦项） |
| `SpectralNorm(layer, power_iter=1)` | 谱归一化：`w = v/σ(v)`，σ 用幂迭代估计；默认把 σ 视为常量（`freeze_sigma()` 可冻结以获得严格端到端梯度） |
| `SwitchNorm(channels, eps=1e-5)` | Switchable Norm：instance / batch / layer 三种统计量按 `softmax(logits)` 软加权（本实现 instance 与 batch 同源，权重合并）；`logits` 也可学 |
| `FRN(channels, eps=1e-6, tau=0.0)` | Filter Response Normalization：跨通道求 ν²、无 batch 依赖（batch=1 可用），后接可学习阈值的 TLU |
| `PReLU(num_features, init=0.25)` | 参数化 ReLU：逐特征可学习斜率，负分支才累计 `da` |
| `Snake(num_features, init=1.0)` | Snake 激活：`y = x + sin²(ax)/a`，逐特征可学习 `a`，`a→0` 退化为恒等（按极限 `y ≈ x + a·x²` 处理） |
| `Maxout(num_features, num_pieces=2)` | 逐特征并行 `k` 条仿射支路取最大（分段线性凸函数），梯度只走 argmax 支路；`param_count = 2·d·k` |
| `DynamicReLU(num_features, init_slope=0.25)` | 动态 ReLU：负分支斜率由输入均值线性预测 `a_i = w_i·mean(x) + b_i`；`dx` 有跨特征耦合项 `(1/d)Σ_{x_i≤0} dout_i·x_i·w_i` |
| `FReLU(channels, kernel_size=3)` | 漏斗激活 `y = max(x, T(x))`，`T` 为逐通道 `k×k` 深度卷积（零填充）；`w,b` 全零时严格等于 ReLU |
| `Embedding` | 嵌入层 |
| `ResidualBlock` | 残差块 |
| `Sequential` | 层序列容器；`Sequential(a, b)` 与 `Sequential([a, b])` 都接受，不是模块（没有 `forward`）的会在构造时报错；不含 train/eval，内部的 Dropout 需自行切换 |
| `MultiHeadAttention` | 多头注意力；`causal=True` 时加因果掩码；`window=N` 时改成滑动窗口（局部因果，只看最近 N 个位置）；`num_kv_heads=K` 时用 GQA/MQA（多个 Q 头共享一组 K/V）；`rope=True` 时对 Q/K 加旋转位置编码；`dropout>0` 时对注意力权重做 inverted dropout（`eval()` 关闭） |

> **早停与权重回滚**：`fit(patience=...)` 在指标不再改善时停止训练，但**不会自动回滚**
> 到最优那一轮；需要最优权重时传 `restore_best_weights=True`。
>
> **优化器状态默认不落盘**：`save_model` / `Model.save` 只保存权重、偏置与结构（及 task），
> 要连 Adam 的动量/二阶矩一起存，需显式传 `optimizer=`（`Model.save` 默认已传），
> 加载时再传 `restore_optimizer=True`。不恢复动量的话「加载后继续训练」不等于「不中断地训练」——
> 前几步的更新幅度会偏大（实测 2.4e-02 量级差异）；恢复后可达**逐位续训**（见下方「断点续训」）。

### 训练 / 推理模式

含随机性的组件默认都是**训练模式**，所以推理前必须显式切换，否则同一输入两次结果不同：

```python
model.eval()      # 之后预测结果确定
model.forward(x)
model.train()     # 回到训练
```

| 组件 | 说明 |
|------|------|
| `Dropout` | `train()` / `eval()`；`eval()` 时原样透传 |
| `TransformerEncoderLayer` / `TransformerEncoder` | `train()` / `eval()` 会转发到内部两个 Dropout **和注意力的权重 dropout** |
| `MultiHeadAttention` | `train()` / `eval()` 只控制注意力权重 dropout（不看模式的数学本身） |
| `BatchNorm1d` | `train()` / `eval()`；`forward` 按模式自动选 batch / running 统计量 |
| `BatchNorm2d` | `train()` / `eval()`；训练用当前样本的 H×W 统计量，推理用 running 统计量 |
| `MLP` / `Model` | **不含 Dropout**，`quick_train`、`make_classifier` 的返回值无需 `eval()` |

> `Sequential` 目前没有 train/eval，把 Dropout 放进 `Sequential` 时需要自行对子模块调用。

### 循环神经网络

| 组件 | 说明 |
|------|------|
| `RNNCell` / `RNN` | 基础 RNN（含 BPTT） |
| `LSTMCell` / `LSTM` | 长短期记忆网络 |
| `GRUCell` / `GRU` | 门控循环单元 |

### Transformer

| 组件 | 说明 |
|------|------|
| `MultiHeadAttention` | 多头注意力机制（`causal=True` → 因果掩码，位置 t 只能看到 `0..t`；`window=N` → 滑动窗口局部因果；`rope=True` → Q/K 旋转位置编码；`dropout>0` → 注意力权重 dropout） |
| `FeedForward(d_model, d_ff, activation='relu')` | 前馈网络（`activation` 可选 relu/gelu/sigmoid/silu/mish/elu/selu/linear） |
| `SwiGLU(d_model, d_ff)` | 门控前馈网络（LLaMA/PaLM）：`(silu(x·W_g) ⊙ (x·W_1))·W_2`，表达力强于 ReLU-FFN；`gate="swish|silu|gelu|relu|bilinear"` 对应 GeGLU / ReGLU / Bilinear GLU |
| `MoE(d_model, d_ff, num_experts=4, top_k=2)` | 专家混合前馈层：路由网络为每个 token 选 top-k 个专家，在被选中的专家上做 softmax 加权求和；未被选中的专家梯度恰为 0 |
| `VAE(input_dim, hidden_dim=32, latent_dim=2, beta=1.0)` | 变分自编码器：重参数化采样 `z = mu + eps·std`，损失 = 重构 MSE + `beta·KL`；`fit` / `sample` |
| `GAN(input_dim, hidden_dim=32, latent_dim=2)` | 生成对抗网络：D 用 sigmoid + BCE，G 的梯度经 D 反传（`d_out = -1/p`）；`step` 一次更新 D 与 G |
| `DDPM(data_dim, hidden_dim=32, steps=50, prediction="eps")` | 扩散模型：线性 beta 表 + 累积 `abar`；`prediction` 可选 `eps`/`v`/`x0`（`target()` 与 `pred_to_x0_eps()` 互为反函数）；`q_sample` 加噪、`sample` DDPM 反向去噪、`ddim_sample(steps=, eta=)` 确定性 DDIM（可只走子序列）；`train_step(..., min_snr_gamma=)` 按 `min(SNR, γ)/SNR` 加权 |
| `VQVAE(input_dim, hidden_dim=32, latent_dim=2, num_codes=8, beta=0.25)` | 向量量化自编码器：`encode` / `quantize` / `decode`，重构 + 码本 + commitment 三项损失，straight-through 反向；`codebook_histogram()` 看码字使用 |
| `TransformerEncoderLayer` | 单层 Encoder（`causal=True` 时自注意力带因果掩码，`rope=True` / `window=N` / `alibi=True` 透传给注意力） |
| `TransformerEncoder` | 多层 Encoder 堆叠（`causal` / `rope` / `window` 会传给每一层） |
| `CrossAttention` | 交叉注意力：Q 来自解码器、K/V 来自编码器输出 `memory`（长度可不同，无因果掩码），`backward` 返回 `(dx, dmemory)` |
| `TransformerDecoderLayer` / `TransformerDecoder` | 解码器层（因果自注意力 → 交叉注意力 → FFN，三段残差 + LayerNorm）与多层堆叠；`backward` 把梯度同时传回解码器输入和编码器输出 |
| `PositionalEncoding` | sin/cos 绝对位置编码（无参数，按需扩展表长） |
| `LearnedPositionalEncoding(d_model, max_len=5000, std=0.02)` | 可学习位置编码表（`out[t] = x[t] + table[t]`，与模型一起训练；`backward` 返回 `(dx, d_table)`，带 `sgd_step`/`adam_step`，超长输入自动扩表） |
| `LinearAttention(d_model, num_heads, causal=False, eps=1e-6)` | **线性注意力**（Katharopoulos 2020）：`phi(x)=elu(x)+1` 替代 softmax，利用加法结合律把 T×T 矩阵换成「前缀累加矩阵 + 向量」，代价从 O(T²·d) 降到 O(T·d²)。`causal=True` 用前缀和（自回归可用），非 causal 用全长和；不带注意力 dropout （权重不显式存在），`train()`/`eval()` 是空操作 |
| `LinformerAttention(d_model, num_heads, seq_len, num_proj=None)` | **Linformer**（Wang 2020）：用两个可学习投影 `E`/`F`（`k×T`）把 K/V 沿**序列维**压到 `k` 个 token，分数矩阵从 `T×T` 变成 `T×k`，代价 O(T·k·d)。`num_proj` 默认 `max(1, seq_len//4)`；**不支持 causal**（E/F 用了整条序列，做自回归会泄漏未来）、同一个实例只能处理固定长度；`k == T` 且 `E = F = I` 时与标准注意力逐位等价（测试里验证） |
| `PerformerAttention(d_model, num_heads, num_features=16, causal=False, eps=1e-6, seed=0)` | **Performer**（Choromanski 2020，FAVOR+）：`phi(x)=exp(Wx-\|x\|²/2)/sqrt(m)` 近似 softmax 核，同样用前缀累加矩阵，O(T·d·m)。`W` 是固定随机特征（不训练）、每头一份；`causal=True` 可自回归。**近似质量**：实测 d=8 时 `m=4` 相对误差约 92%、`m=64` 约 35%（iid 高斯特征，论文的正交版更好），所以它是「省算力」而不是「逼近 softmax」的组件 |
| `MLAttention(d_model, num_heads, latent_dim=None, causal=False, eps=1e-8)` | **MLA 简化版**（DeepSeek-V2）：`c_t = W_dkv x_t` 是低秩潜在向量，`k/v` 由 `W_uk c_t` / `W_uv c_t` 上投影得到——**KV cache 只需存 `T·d_c` 个标量**，而不是标准 MHA 的 `T·2·H·d_k`（`kv_cache_size` / `full_kv_cache_size` 可查）。保留全秩 Q 与标准缩放点积注意力（也支持 causal），所以它是「低秩 KV」而非完整 MLA（后者还会压缩 Q、并把 RoPE 解耦出去） |
| `BigBirdAttention(d_model, num_heads, window_size=2, num_global=1, num_random=1, causal=False, seed=0)` | **BigBird 稀疏注意力**：窗口 + 全局 + 随机三类连接，只在稀疏图上做 softmax。全连通时与稠密注意力**逐位等价**（验证 <1e-12），`seed` 决定随机连边且可复现 |
| `LongformerAttention(d_model, num_heads, window_size=2, global_tokens=None, causal=False)` | **Longformer**：滑窗 + 全局 token。全局 token（默认 `[0]`）能看见整条序列，其余位置只看窗口 ∪ 全局，代价 O(T·w) |
| `ReformerAttention(d_model, num_heads, num_buckets=2, causal=False, seed=0)` | **Reformer 的 LSH 注意力**：固定随机投影 argmax 分桶，只在桶内做注意力。简化点：单轮哈希、桶分配不可导（梯度只走注意力权重），`num_buckets=1` 时退化为稠密注意力 |
| `FlashAttention(d_model, num_heads, causal=False, block_size=2)` | 分块 + 在线 softmax 的注意力：`O(T)` 显存、**不物化 T×T 矩阵**；反向按 `(m, l)` 重算概率。注意：纯 Python 里**不会更快**，价值是把显存从 O(T²) 降到 O(T) —— 反向的 `tot = Σ_j p_ij·dp_ij` 必须在**所有键块**上求和，单块实现会漏 |
| `T5RelativeBias(num_heads, num_buckets=32, max_distance=128, bidirectional=True)` | T5 式相对位置偏置：`(num_buckets, num_heads)` 可学习表，`forward(qlen, klen)` 返回 `bias[head][i][j]` 加到 softmax 前；分桶用 `t5_relative_buckets(...)`，正负相对位置各占一半桶、近端线性远端对数 |
| `xpos_factors(seq_len, d_k, base=10000.0, alpha=1.0, offset=0.0)` | xPos 的位置衰减因子（简化版）：对第 `i` 个频率维乘 `(t+1+offset) ** (-alpha * theta_i)`，位置 0 处为 1；`alpha=0` 等于关闭 |
| `GCNLayer(in_features, out_features, activation='relu', self_loops=True)` | 图卷积 `H = σ(Â X W + b)`，`Â = D^{-1/2}(A+I)D^{-1/2}`；`adj` 是 `N×N` 密集矩阵，邻接视为常量不参与反向 |
| `GraphSAGELayer(in_features, out_features, concat=True)` | 均值聚合 `σ(W [x_i ‖ mean_{j∈N(i)} x_j])`；`concat=False` 时只用邻居均值 |
| `GINLayer(in_features, hidden, out_features, eps=0.0, train_eps=False)` | `MLP((1+eps)·x_i + Σ_{j∈N(i)} x_j)`；`train_eps=True` 时 `eps` 作标量参数参与优化 |
| `GraphReadout(mode='mean')` | 节点特征 → 图向量的读出（`mean` / `sum` / `max`），反向按模式分发（`max` 只回给取到最大值的位置） |
| `GATLayer(in_features, out_features, num_heads=1, concat=True)` | 图注意力：`s_ij = LeakyReLU(a_s·z_i + a_d·z_j)`、`alpha = softmax_{j∈N(i)} s_ij`、`ctx_i = Σ_j alpha_ij z_j`；`adj` 只用**存在性**（权重被忽略），`concat=True` 时多头拼接、否则取均值 |
| `APPNPLayer(in_features, out_features, K=2, alpha=0.1)` | 个性化 PageRank 传播：`H0 = σ(XW+b)`、`H^{k+1} = (1-alpha)·Â H^k + alpha·H0`，输出 `H^K`；`alpha=0` 退化为 `Â^K H0`，`alpha=1` 直接输出 `H0` |
| `JKNet(layers, mode='concat')` | 跳连汇总（JK-Net）：逐层前向再汇总，`concat` 拼所有层输出 / `last` 只用最后一层 / `max` 逐位置取最大；反向按堆叠语义把梯度链式回传每一层 |
| `MPNNLayer(in_features, hidden, out_features, edge_features=0)` | 消息传递：`m_ij = MLP_msg([x_i ‖ x_j ‖ e_ij])`、`agg_i = Aggregate_{j∈N(i)} m_ij`、`h_i = MLP_upd([x_i ‖ agg_i])`；`aggregation` 支持 `sum` / `mean`，`edge` 是可选的密集边特征（视为常量不反向） |
| `GCNIILayer(in_features, alpha=0.1, lam=0.5, layer_index=0)` | GCNII 的一层：`H^{l+1} = σ(((1-alpha)·Â H^l + alpha·H^0)((1-beta_l)I + beta_l W))`，`beta_l = log(lam/l + 1)`；要求各层同宽，`H^0` 由调用方传入（缺省用本层输入），`grad_h0` 暴露对初始特征的梯度 |
| `TopKPool(in_features, ratio=0.5)` | Graph U-Net 风格 Top-K 池化：`g_i = sigmoid(W x_i + b)`，保留分数最大的 `k = max(1, round(ratio·N))` 个节点、特征按 `g_i` 缩放、邻接取子块；反向同时回传 gate 路径与 score 路径 |
| `SAGPool(in_features, ratio=0.5)` | 自注意力池化：用一层 GCN 打分 `s = tanh(GCN(X, A))`，保留前 `k` 个节点并按 `s` 缩放特征 |
| `PromptTuning(num_tokens, d_model, seed=0)` | 软提示：`num_tokens` 个可学习向量拼在序列前，`forward` 返回 `prompt + seq`，反向把前缀梯度累加进 `g_prompt`、其余直通回序列 |
| `P_Tuning(num_tokens, d_model, hidden=None)` | 生成式软提示：`tanh(anchor·W1 + b1)·W2 + b2` 产生前缀，`anchor`/`W1,b1`/`W2,b2` 全可学，`g_anchor` 等暴露各段梯度 |
| `sequence_packing(sequences, pad_id=0)` | 多条序列打包成一条：返回 `tokens`/`segment_ids`/`position_ids`（每段从 0 计数）/`segments`/`length`，不插分隔符 |
| `packed_causal_mask(segment_ids, causal=True)` | 打包序列的注意力掩码：只允许同段（`causal` 时再加 `j<=i`） |
| `PrefixCache(num_layers, d_model)` | 逐层 KV 前缀缓存：`set`/`get`/`length`/`truncate`/`merge` + `to_dict`/`from_dict`/`save`/`load`（JSON 持久化） |
| `ContinuousBatcher(max_batch=2, max_total_tokens=None)` | 连续批处理调度：`add(prompt, max_new, request_id)` 入队、`batch()` 取当前批、`step({id: token})` 喂 token 并补位、`result(id)`/`stats()`；只做调度与记账，不依赖模型实现 |
| `PagedKVCache(num_layers, num_heads, head_dim, block_size=4, max_blocks=16)` | PagedAttention 的块表（数据结构演示）：`new_request`/`append`/`get`/`length`/`free`/`stats`；逻辑位置到物理块不必连续、释放后可复用 |
| `CausalLM` | 字符/词级因果语言模型（GPT 式）：Embedding → 位置编码 → 因果 Transformer → 词表头，自带 Adam、`fit` / `evaluate` / `generate`；`rope=True` 时改用旋转位置编码（默认关掉绝对位置编码），`window=N` 用滑动窗口，`generate(use_cache=True)` 走 KV cache，`top_p` / `min_p` 做截尾采样、`repetition_penalty` / `presence_penalty` / `frequency_penalty` / `no_repeat_ngram_size` 抗重复，`num_beams>1` 走束搜索；`save` / `load` 支持 JSON 存档（可选带 Adam 状态做逐位续训） |
| `MaskedLM` | 掩码语言模型（BERT 式）：双向 Transformer，随机遮 15% 位置（80% 换 `mask_id` / 10% 随机 / 10% 保留），只在被遮位置算交叉熵；自带 Adam、`fit` / `evaluate` / `predict`（填空）、`mask_tokens` |
| `Seq2SeqLM` | 序列到序列（encoder-decoder）：源端 `TransformerEncoder` → 目标端因果 `TransformerDecoder` + 交叉注意力 → 词表头；自带 Adam、`fit` / `evaluate` / `generate`，`fit` 用 teacher forcing（目标端 `BOS` 放在词表外的一个 id），`generate` 逐 token 解码并支持 `eos_id` 提前停 |

### 因果掩码与 LLM

`causal=True` 把注意力分数矩阵的上三角（`j > t`）置为 `-inf`，softmax 之后
这些权重恰好是 `0.0`，因此：

- 位置 `t` 的输出**只依赖 `0..t`** —— 一次前向就能并行算出所有位置的「下一个 token」预测；
- 反向不需要额外分支，被屏蔽位置的梯度天然为 `0`（`d_scores = attn * (...)` 里 `attn=0`）。

`CausalLM` 直接建立在这套掩码之上，训练目标就是学生成「输入 `ids` → 目标 `ids[1:]`」：

```python
lm = CausalLM(vocab_size=3, d_model=16, num_heads=2, d_ff=32, num_layers=1)
lm.fit([[0, 1, 2] * 5], epochs=120, lr=0.01)      # 学 012 循环
lm.generate([0], max_new_tokens=8)                 # → [0, 1, 2, 0, 1, 2, 0, 1, 2]
```

整个实现是**纯标准库**：没有依赖矩阵库，也没有 `torch`。训练/生成都用一个
小语料（15 个 token、1 层、d_model=16）即可在 1 秒内跑完，适合当教学样例。

> 注意：`generate()` 会把模型切到 `eval()` 模式（关闭 Dropout）并保持；
> 想继续训练请再调 `train()`。

### 现代 LLM 组件（GELU / RMSNorm / AdamW / warmup / RoPE / KV cache）

这些是当下 LLM 训练的常用配置，全部是**纯增量**：不传新参数时，所有默认路径与旧版本
**逐位一致**（`test_ai.py` 里有专门的回归与差分测试守着）。

```python
# 现代激活：GELU（tanh 近似）、sigmoid（分支写法，±800 不溢出）
mlp = MLP([8, 32, 4], activations=['gelu', 'linear'])
linear = Linear(8, 4)                     # 语义化写法 = Layer(..., activation='linear')
norm = RMSNorm(32)                        # Llama 系：不减均值、无 beta

# AdamW：解耦权重衰减（每步先把参数乘 1-lr*wd，再套 Adam）
model = Model(MLP([5, 16, 2]), loss_fn='cross_entropy', optimizer='adamw',
              lr=0.01, weight_decay=0.1)

# 线性 warmup + 现有调度器组合（前 5 轮爬坡，之后交给 CosineAnnealingLR）
sched = WarmupLR(optimizer, warmup_epochs=5,
                 base_scheduler=CosineAnnealingLR(optimizer, T_max=50))
```

| 组件 | 说明 |
|------|------|
| `Layer(activation='gelu')` | GELU（tanh 近似），导数同样手写 |
| `Layer(activation='sigmoid')` | sigmoid；用 `z>=0 ? 1/(1+e^-z) : e^z/(1+e^z)` 写法，避免 `exp` 溢出 |
| `Layer(activation='silu')` | SiLU / Swish：`z·sigmoid(z)`，LLaMA 的门控激活 |
| `Layer(activation='mish')` | Mish：`z·tanh(softplus(z))`，softplus 用 `log1p(exp(z))` 且 `z>30` 时退化成 `z` |
| `Layer(activation='elu'/'selu')` | ELU（α=1）与 SELU（λ=1.0507, α=1.6733），自带自归一化性质 |
| `MultiHeadAttention(alibi=True)` | ALiBi 线性偏置：第 h 个头加 `slope_h·(t-j)`，`slope_h=2^(-8(h+1)/num_heads)`；无需位置编码即可外推，偏置是常数、反向无额外分支 |
| `TokenClassificationHead(d_model, num_labels)` | 序列标注头：`(T,d_model)→(T,num_labels)`，loss 按有效 token 平均，`mask` 位置梯度恰 0，`predict` 逐位置取 argmax |
| `MultiLabelHead(fan_in, num_labels)` | 多标签头：逐标签 sigmoid + BCE-with-logits，标签之间不竞争，`predict(threshold=0.5)` 返回 0/1 列表 |
| `Linear(fan_in, fan_out)` | 无激活全连接层 |
| `RMSNorm(num_features, eps=1e-6)` | RMS 归一化 |
| `LoRA(layer, rank=4, alpha=None)` | 低秩适配：冻结原层，只训 `A (rank, in)` / `B (out, rank)`；`B` 初始为 0（包装后行为不变）；`merge()` 把 `scaling·BA` 折进原层权重 |
| `DoRALinear(layer, rank=4, alpha=None)` | DoRA：权重拆成「幅度 `m` × 方向 `V`」，LoRA 只改方向（`W' = m·Weff/‖Weff‖`），`V`/`m` 也可训；`B=0` 时与原层逐位一致，`merge()` 折回原层 |
| `AdaLoRALinear(layer, rank=8, ema=0.9)` | AdaLoRA（简化）：每秩维护重要性 `|A_r|·|B_:,r|` 的 EMA，`prune(keep)` 只保留最重要的 `keep` 个秩（其余 A 行/B 列置零并停梯度），`active_ranks()` 查存活秩 |
| `AdamW(model, lr, ..., weight_decay=0.01)` | 解耦权重衰减；`weight_decay=0` 时与 `Adam` **逐位相同** |
| `Lion(model, lr=1e-4, beta1=0.9, beta2=0.99)` | 符号动量优化器：更新量恒为 `±lr`（+ 解耦 wd），只存一份动量；**lr 通常比 Adam 小一个量级** |
| `Adafactor(model, lr=1e-3, eps2=1e-3, weight_decay=0.0)` | 行列因子化二阶矩，显存 O(m+n)；乘相对步长 `max(eps2, rms(g))`，所以 lr 语义比 Adam 小一个量级 |
| `WarmupLR(optimizer, warmup_epochs=5, start_factor=0.0, base_scheduler=None)` | 线性 warmup，到期后把控制权交给 `base_scheduler` |
| `ReduceLROnPlateau(opt, factor=0.5, patience=10, min_lr=0.0, mode='min')` | 指标停滞 `patience+1` 次就把 lr 乘 `factor`（可反复降档，与 PyTorch 一致）；`step()` 要传指标 |
| `MultiHeadAttention(rope=True)` | RoPE（GPT-NeoX 版式：第 i 维配 i+d_k/2）；反向用逆旋转 |
| `MultiHeadAttention(rope_scaling="yarn")` | RoPE 长度外推：`"linear"`（频率全除 factor）、`"ntk"`（改底数 `base·factor^(d_k/(d_k-2))`）、`"yarn"`（按旋转圈数做 linear/原样的 smoothstep 分段）；配套 `rope_factor` / `rope_original_max` / `rope_beta_fast` / `rope_beta_slow`，也可直接调 `rope_inv_freqs(d_k, base, scaling, factor, ...)` 拿频率表 |
| `MultiHeadAttention(window=N)` | 滑动窗口（Mistral 式局部因果）：只看最近 N 个位置，代价 O(T·window) |
| `generate(..., top_p=0.9)` | nucleus 采样：按概率累加保留头部集合（与 `top_k` 可叠加；默认 1.0 不过滤） |
| `generate(..., min_p=0.05)` | min-p 截尾：低于「最大概率 × min_p」的全丢掉（比 top_p 更稳；默认 0 关闭） |
| `typical_filter(logits, mass=0.9)` | typical 采样：保留「惊讶度最接近条件熵」的 token 直到累计概率 ≥ mass |
| `top_a_filter(logits, top_a=0.1)` | top-a：阈值 `top_a × p_max²`（`top_a=0` 不过滤） |
| `eta_filter(logits, eta=0.3)` | eta 采样：丢掉概率 ≤ eta 的 token（至少保留 argmax） |
| `mirostat_filter(logits, mu=5.0)` | Mirostat：保留 `-log p ≤ mu` 的 token，返回 `(masked, 保留集熵)`；配合 `mirostat_update(mu, 观测惊讶度)` 迭代 |
| `tail_free_filter(logits, tft=0.5)` | Tail-free 采样：按 `log p` 曲线的归一化二阶差分截断长尾（跳过头部线性区），`tft` 越小保留越多；返回带 `-inf` 的新 logits |
| `token_id_mask(logits, allowed)` | 约束解码：只保留 `allowed` 里的 token id（其余置 `-inf`），可与任意采样器叠加 |
| `token_prefix_mask(logits, generated, sequences, eos_id=None)` | 前缀约束：只允许仍是某个合法序列前缀的下一个 token；序列已完整时只允许 `eos_id` |
| `speculative_accept(draft_ids, draft_probs, target_probs)` | 投机解码的接受判定：按 `min(1, p_target/p_draft)` 拒绝采样，被拒时从 `max(0, p_target − p_draft)` 重采样，返回 `(接受个数, 重采样 token)` |
| `generate(..., repetition_penalty=1.2)` | 重复惩罚：已出现的 token，正 logit 除以它、负 logit 乘以它（默认 1.0 不惩罚） |
| `generate(..., presence_penalty=0.5, frequency_penalty=0.5)` | 存在惩罚（出现过就扣一次）/ 频率惩罚（按出现次数扣） |
| `generate(..., no_repeat_ngram_size=3)` | 禁止 n-gram 立刻重复：末尾 (n-1)-gram 再次出现时，把它后面跟过的 token 置 `-inf` |
| `MultiHeadAttention(num_kv_heads=K)` | GQA/MQA：K 个 Q 头共享一组 K/V，`W_k`/`W_v` 与 KV cache 同比例缩小 |

**RoPE + KV cache**：`CausalLM(..., rope=True)` 会默认关掉绝对位置编码（Llama 的做法），
想两者同时用可传 `use_abs_pos=True`。

**优化器与 LoRA**：`Lion`（符号动量，`lr` 取 1e-4~1e-3）和 `Adafactor`（行列因子化，显存 O(m+n)）
都能直接 `opt.step(AI._mlp_backward(model, grad))`；`LoRA` 用来冻结原层只训低秩分支：

```python
layer = MLP([64, 128]).layers[0]
lora = LoRA(layer, rank=4)          # B 初始为 0，包装后前向逐位不变
out = lora.forward(x)               # 与原层同形状
opt = Adam(lora, lr=1e-3)           # 只更新 A/B（原层被冻结）
lora.merge()                        # 训练完折回原层权重，推理零额外开销
```

```python
lm = CausalLM(vocab_size=8, d_model=32, num_heads=4, d_ff=64, num_layers=2,
              max_len=128, rope=True)
lm.fit(seqs, epochs=50, lr=3e-3)

# 默认走 KV cache：每个新 token 只前向一次（复用各层 K/V），复杂度 O(N·T)
out = lm.generate([1, 2, 3], max_new_tokens=32, top_k=5)
out = lm.generate([1, 2, 3], max_new_tokens=32, top_k=5, use_cache=False)  # 结果逐位相同
```

实测 4 层 / d_model=32 / prompt 8 / 新生成 40 个 token：`use_cache=True` 比整段重算快
**约 18 倍**，且两者输出的 token **完全一致**（有测试用例逐位比对，含窗口滑动与 RoPE 两种情形）。

> cache 依赖绝对位置，窗口超过 `max_len` 开始滑动时会自动按新窗口重建一次，
> 所以滑动场景仍然与重算版一致。

**滑动窗口**：`MultiHeadAttention(window=64)` 或 `CausalLM(..., window=64)` 让每个位置只看
最近 64 个位置（含自己），更早的权重被掩在 `-inf` 后 softmax 成 0，反向无需额外分支。
窗口**隐含因果**——未来位置一律不可见；`window=None`（默认）就是全注意力，与旧版逐位一致。
增量前向（KV cache）同样按当前绝对位置裁剪窗口，因此 `forward_step` 与整段 `forward`
的最后一行**逐位相同**（`rope=True` 时也成立，有测试守着）。

**nucleus 采样**：`generate(..., top_p=0.9)` 按温度缩放后的概率从高到低累加，保留累计概率
刚好覆盖 0.9 的头部集合再采样；`top_p=1.0`（默认）不做过滤，输出与旧版逐位一致。

**抗重复与 min-p**：采样路径另外支持五项控制，默认值全部等于「不干预」，因此不传参时输出与
旧版逐位相同：

```python
# 1) min-p：只保留概率 >= 最大概率 5% 的 token（比 top_p 更稳，2024 年起常用）
lm.generate(prompt, 80, min_p=0.05, temperature=0.9)

# 2) 重复惩罚：>1 抑制已出现的 token（正 logit 除以它、负 logit 乘以它）
lm.generate(prompt, 80, repetition_penalty=1.2)

# 3) presence / frequency：出现过扣一次 / 按出现次数扣
lm.generate(prompt, 80, presence_penalty=0.5, frequency_penalty=0.3)

# 4) 禁重复 n-gram：3-gram 不再立刻重复（末尾 2-gram 再次出现时禁掉后继）
lm.generate(prompt, 80, no_repeat_ngram_size=3)

# 5) 与束搜索、KV cache、top_p、top_k 可任意叠加
lm.generate(prompt, 80, num_beams=3, no_repeat_ngram_size=2, min_p=0.02)
```

`Seq2SeqLM.generate` 支持同样的五个参数（含义一致，作用在解码器侧）。`min_p` 要求 `[0,1)`，
`repetition_penalty` 要求 `> 0`，`no_repeat_ngram_size` 要求 `>= 2`（=1 会禁掉所有已出现的
token，没有意义，直接报错）。

**GQA / MQA**：`CausalLM(..., num_kv_heads=2)`（或 `MultiHeadAttention(num_kv_heads=2)`）
让每 `num_heads/num_kv_heads` 个 Q 头共享一组 K/V——Llama-2 70B、Mistral 都用这套。
`W_k`/`W_v` 的输出维度从 `d_model` 缩到 `num_kv_heads*d_k`（参数与 KV cache 同比例变小），
反向时同组 Q 头的 `dK`/`dV` 在 KV 头上相加。`num_kv_heads` 缺省等于 `num_heads`，
此时张量形状、随机数抽取顺序都与旧版完全一致；`num_kv_heads=1` 即 MQA。

```python
lm = CausalLM(vocab_size=32, d_model=64, num_heads=8, d_ff=128,
              num_layers=2, num_kv_heads=2)   # 8 个 Q 头 → 2 组 K/V
lm.fit(seqs, epochs=50, lr=3e-3)
out = lm.generate([1, 2, 3], max_new_tokens=32, use_cache=True)
print("困惑度:", lm.perplexity(seqs))    # exp(交叉熵)
```

**批量 padded 训练**：`fit(..., batch_size=8)` 会把一批长短不一的序列右侧补齐成同一长度，
再用 `pad_mask` 把填充位挡在注意力之外——填充位**不能当 key**（分数置 `-inf`，softmax 后
权重恰为 0），loss 只对有效目标位置求平均并除以有效 token 数，梯度按 token 数加权后
每批做一次 Adam。

```python
seqs = [[1, 4, 2, 5, 3], [2, 3], [6, 1, 2, 4, 5, 3, 2]]   # 长度 5 / 2 / 7
lm.fit(seqs, epochs=50, lr=3e-3, batch_size=2)             # 3 条 → 2 步/轮
```

补齐是**数值透明**的：同一序列「单独算」与「补齐后带 mask 算」的 logits、loss、每个梯度都
逐位相同（有测试守着）。要自己喂 padding 也可以，`forward(ids, pad_mask=[1,1,0,...])` 与
`_loss_and_backward(ids, pad_mask)` 都接受它；`batch_size=1`（默认）仍然是「每条序列一步」，
与旧版逐位一致。填充必须**靠右**——真实 token 的位置编号才不会被挪动（位置编码/ RoPE 都依赖它）。
传成左填充（`[0,0,1,1,1]`）会直接 `ValueError`：第 0 行 query 的因果可见集只有 key 0，再被 mask 掉
整行就是 `-inf`，原来会静默产出 NaN logits 和全 NaN 梯度（loss 还是普通数字，最难查的那种）。
同理 `top_k` 必须是 `>= 0` 的整数（`0.5` 会被截成 0 → 候选集为空 → 采样恒取最后一类），
token id 必须是 `[0, vocab)` 内的整数（负索引原来会被悄悄当成「最后一类」），
logits 若出现 NaN/Inf 会直接报错而不是把 loss clamp 成 27.63。

### 序列到序列（Encoder-Decoder）

```python
import AI

# 目标端 BOS 用词表外的 id（tgt_vocab），源/目标词表可以不同大小
m = AI.Seq2SeqLM(src_vocab=16, tgt_vocab=16, d_model=32, num_heads=4,
                 d_ff=64, num_layers=2, max_len=32)
m.fit([([1, 2, 3], [1, 2, 3])], epochs=40, lr=0.01)   # teacher forcing
print(m.evaluate([([1, 2, 3], [1, 2, 3])]))            # 平均交叉熵
print(m.generate([1, 2, 3], max_new_tokens=3, top_k=1))  # 贪心解码
```

解码器里三层结构各带自己的残差与 LayerNorm：**因果自注意力 → 交叉注意力（K/V 来自编码器
输出，长度可以与源序列不同）→ 前馈网络**；`backward` 同时把梯度传回解码器输入和编码器输出，
所以整条 encoder-decoder 可以端到端训练。`fit` 把目标序列右移一位当输入
（`[BOS] + tgt[:-1]`），交叉熵对目标 token 数取平均。

**批量 padded 训练**（与 `CausalLM` 的 `pad_mask` 同一套思路）：

```python
# 长度不同的样本会右填充成一批；padding 不进注意力、不计入 loss
m.fit(pairs, epochs=40, lr=0.01, batch_size=4)

# 单独调用也可以自己传掩码：0 表示右填充位置
logits = m.forward(src, tgt_in, src_mask=[1, 1, 1, 0])
m._loss_and_backward(src, tgt, src_mask=[1, 1, 1, 0], tgt_mask=[1, 1, 1, 0])
```

实现上只有两处需要屏蔽：编码器自注意力的 key（`src_mask`）与解码器交叉注意力的 memory
（同一份 `src_mask`）——softmax 后填充位置权重恰为 0，反向自动零梯度。梯度按 token 数加权
（`grad_scale = 该样本目标 token 数 / 整批目标 token 总数`），所以 `batch_size=4` 一步等于
把 4 条序列的目标 token 拉平后取平均；不传掩码时与逐条训练**逐位一致**。

### 训练工程化工具

```python
import AI

# 分层切分：类别不均衡时按比例抽（默认行为不变）
train, val, test = AI.split_data(data, 0.2, 0.1, seed=7, stratify=labels)

# K 折交叉验证：只算索引，不复制数据
for k, (tr_idx, va_idx) in enumerate(AI.k_fold_indices(len(data), 5, seed=7)):
    model.fit(x=[data[i][0] for i in tr_idx], y=[data[i][1] for i in tr_idx], epochs=20)
    print(k, model.score([data[i][0] for i in va_idx], [data[i][1] for i in va_idx]))

# 标准化 / 归一化
scaler = AI.StandardScaler().fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# 梯度自检（中心差分，纯标准库）：返回 ok / worst_name / worst_rel / checks
print(AI.grad_check(model, x_sample, y_sample)["worst_rel"])

# 训练曲线（终端里直接看）
model.fit(x_train, y_train, epochs=50, verbose=False)
print(AI.ascii_plot(model.history))
```

> `grad_check` 的 `CausalLM` / `Seq2SeqLM` 分支默认抽查 24 个标量参数（整模型逐个差分太慢），
> 可用 `max_params=` 调整；它已经把「梯度错」和「数值误差」区分开——人为改坏一个
> 梯度元素时 `worst_rel` 会到 1.0 量级。

### 断点续训

`fit` 可以边训边存，`Model.load` 可以把优化器状态一起接上，做到**逐位续训**：

```python
# 训练 4 轮，每轮落一次盘
model.fit(x_train, y_train, epochs=4, checkpoint_path="ck.json", checkpoint_every=1)

# 从中断处接着训 2 轮：等价于一次训 6 轮（权重逐位相同）
model = AI.Model.load("ck.json", task="regression", restore_optimizer=True)
model.fit(x_train, y_train, epochs=2)
```

- 只存权重时 `save_model` 的输出与旧版**逐字段相同**（无 `optimizer` 键），旧存档照常可读。
- 传 `optimizer=` 才会写入 `{"type","lr","beta1","beta2","eps","step_count","m_w","v_w","m_b","v_b"}`；
  `SGD` / `SGDMomentum` / `RMSprop` 没有状态，`_optimizer_state()` 返回 `None`。
- `restore_optimizer=True` 但存档里没有优化器状态时**报 `ValueError`**（原来静默跳过，等于白丢动量）；
  存档里有但参数块对不上也报错；类型或形状不匹配同样 `ValueError`。
- `step_count` 记的是**批次数**（样本数 / batch_size），不是轮数。
- `Model.save` 还会记下**损失函数名与 `loss_kwargs`**（例如 `focal(gamma=1.5)`、
  `CE(label_smoothing=0.1)`）：重新 `Model.load` 后损失配置不变，接着训的 loss 曲线与
  不中断训练逐位相同。显式传 `loss_fn=` / `loss_kwargs=` 会覆盖文件里的值；
  旧文件没有这两个字段，退回按 `task` 推断的默认损失。

### 语言模型存档与束搜索

```python
lm = AI.CausalLM(vocab_size=32, d_model=64, num_heads=8, d_ff=128, num_layers=2)
lm.fit(seqs, epochs=50, lr=3e-3)

lm.save("lm.json", include_optimizer=True)      # 结构 + 参数（+ Adam 状态）
lm = AI.CausalLM.load("lm.json", restore_optimizer=True)   # 可逐位接着训

# 束搜索（确定性，不抽随机数）；num_beams=1 与贪心解码逐位相同
print(lm.generate([1, 2, 3], max_new_tokens=32, num_beams=4))

# Seq2SeqLM 同样有 save / load / num_beams
m = AI.Seq2SeqLM(16, 16, d_model=32, num_heads=4)
m.save("s2s.json")
m = AI.Seq2SeqLM.load("s2s.json")
```

- 存档是纯 JSON：`{"format": "ai_lm_v1", "kind": "causal"|"seq2seq", "config": ..., "params": {名字: 值}}`；
  名字与 `_named_params()` 一一对应，所以参数块增删会被检测出来（报 `ValueError`）。
- `include_optimizer=True` 才会写 `optimizer`（`step_count` + 每个参数块的 `m`/`v`）；
  `restore_optimizer=True` 后继续训练与不中断训练**逐位相同**（未恢复则实测差 ~4.5e-02）。
- 束搜索按累计对数概率保留 `num_beams` 条前缀；`top_k>0` 时每步只展开前 `top_k` 个候选，
  否则展开 `num_beams` 个。`Seq2SeqLM` 的束搜索会把以 `eos_id` 结尾的候选放进「已完成」再统一比较。
- `top_k` 必须是 `>= 0` 的整数；`generate` 会校验源序列长度不超过 `max_len`、token id 在词表内
  （这两条原来只有 `forward` 检查，生成路径会绕过它们）。
### 优化器

| 优化器 | 说明 |
|------|------|
| `Adam` | Adam（默认） |
| `AdamW` | Adam + 解耦权重衰减（LLM 训练的默认选择；`weight_decay` 不进一阶/二阶矩） |
| `SGD` | 随机梯度下降 |
| `SGDMomentum` | 带动量的 SGD |
| `RMSprop` | RMSprop |

### 损失函数

| 函数 | 说明 |
|------|------|
| `mse_loss(pred, target)` | 均方误差；`target` 可以是序列或单个标量（`y=0.5` 也行），长度不一致会报错 |
| `cross_entropy_loss(pred, target, label_smoothing=0.0)` | 交叉熵；`target` 可以是类别索引、one-hot 或概率向量（软标签）；`label_smoothing>0` 时目标分布取 `(1-ε)·onehot + ε/K` |
| `focal_loss(pred, idx, gamma=2.0, alpha=None)` | Focal Loss：`-α(1-p_t)^γ·log p_t`，压易分样本、聚焦难样本（`gamma=0` 即普通 CE） |
| `bce_loss(pred, target)` | 二元交叉熵，`pred` 是概率（配 `sigmoid` 输出层）；返回的梯度相对概率 |
| `bce_with_logits_loss(logits, target)` | 数值稳定的 BCE：`max(z,0) - z·y + log1p(exp(-\|z\|))`，直接吃 logits |
| `huber_loss(pred, target, delta=1.0)` | Huber / Smooth L1，对离群点比 MSE 稳 |
| `smooth_l1_loss(pred, target, delta=1.0)` | `huber_loss` 的别名（PyTorch 的叫法），逐位等价 |
| `l1_loss(pred, target)` | L1 / MAE：`\|p-y\|` 的平均，梯度是 `sign(p-y)/n`（对离群点不敏感） |
| `log_cosh_loss(pred, target)` | Log-Cosh：`log(cosh(p-y))` 的平均，平滑又抗离群；实现用 `\|e\|+log1p(exp(-2\|e\|))-log2` 防溢出 |
| `quantile_loss(pred, target, quantile=0.5)` | 分位数（pinball）损失：`quantile=0.5` 恰为 MAE 的一半，取大分位数即「宁高勿低」 |
| `kl_divergence(pred, target)` | `KL(target ‖ pred) = Σ t·log(t/p)`，蒸馏/软目标常用 |
| `triplet_loss(anchor, pos, neg, margin=1.0)` | `max(0, d(a,p) − d(a,n) + margin)`（欧氏距离），返回三个梯度；已满足间隔时梯度全 0 |
| `info_nce_loss(anchor, pos, negs, temperature=0.07)` | 对比学习：余弦相似度 + 交叉熵，1 正 + N 负；返回 anchor / 正 / 每个负的梯度 |
| `dice_loss(pred, target)` | 分割常用 `1 − (2\|p∩t\|+ε)/(\|p\|+\|t\|+ε)`，支持 `(C,H,W)` 或扁平结构 |
| `ctc_loss(log_probs, targets, blank=0)` | CTC（单样本、log 空间前向后向）：输入 `(T,K)` 的 log 概率，返回对 log 概率的梯度 |
| `class_balanced_loss(pred, idx, class_counts, beta=0.999, normalize=True)` | Class-Balanced：按「有效样本数」`(1-β)/(1-β^{n_c})` 加权，`normalize=True` 时让加权均值回到 1（配 `softmax` 输出层） |
| `ldam_loss(logits, idx, class_counts, max_m=0.5, s=30.0)` | LDAM：给稀有类的 logit 加 `max_m/n_y^{1/4}` 的间隔再乘缩放 `s`，直接吃 logits（配 `linear` 输出层） |
| `ohem_loss(preds, targets, keep_ratio=0.7)` | OHEM（**批级**）：一批概率向量里只保留最难的那部分样本，被丢掉的样本梯度全 0 |
| `distill_loss(student_logits, teacher_logits, T=2.0)` | 知识蒸馏：`T²·KL(teacher_T ‖ student_T)`，对学生 logits 的梯度是 `T·(p_s − p_t)` |
| `hint_loss(student_feat, teacher_feat, weight=1.0)` | 隐藏层对齐（FitNet 式）：学生/教师特征的均方误差，`weight` 调与主损失的配比 |
| `preference_loss(chosen_logp, rejected_logp, ref_*, method='dpo', beta=0.1)` | 偏好对齐，五种：`dpo` / `ipo` / `simpo` / `cpo` / `orpo`；吃**一对**序列的 log 概率，返回对两者的梯度 |
| `kto_loss(logp, ref_logp, desirable=True, beta=0.1, baseline=0.0)` | KTO：单条样本「好/坏」二元反馈即可，不需要成对偏好数据 |

| 损失名（`Model(loss=...)`） | 需要的输出层 | 目标形式 |
|------|------|------|
| `'mse'` | 任意（通常 `linear`） | 数值向量 / 标量 |
| `'cross_entropy'` / `'ce'` | `softmax` | 类别索引 / one-hot / 软标签 |
| `'focal'` | `softmax` | 类别索引 / 严格 one-hot（软标签会被拒绝） |
| `'bce'` | `sigmoid` | 0/1 数值 |
| `'bce_with_logits'` | `linear` | 0/1 数值（喂 logits） |
| `'huber'` | 任意 | 数值向量 / 标量 |
| `'smooth_l1'` / `'l1'` / `'log_cosh'` / `'quantile'` | `linear` | 数值向量 / 标量 |
| `'class_balanced'` | `softmax` | 类别索引 / one-hot / 软标签（`loss_kwargs` 必须给 `class_counts`） |
| `'ldam'` | `linear` | 类别索引 / one-hot（`loss_kwargs` 必须给 `class_counts`） |
| `'distill'` | `linear` | 教师 logits（与输出层同维度的数值向量，`loss_kwargs` 可给 `T`） |

`preference_loss` / `kto_loss` **也不在**上面的映射表里：它们吃的是「一段序列的 log 概率」
（偏好损失还要一对），不是逐样本的 `(pred, target)`，所以只能在自定义训练循环里调用，
不能传给 `Model(loss=...)`。

`kl_divergence` **不在**上面的映射里：它要求 `pred` 是概率分布，而 `softmax` 层的
反向是 softmax+CE 的联合梯度，任何非 CE 损失叠在 softmax 上都拿到不自洽的梯度
（实测 KL 训练 loss 会从 0.35 涨到 5.78）。需要 KL 时请手动调用
`AI.kl_divergence` 并自己反传。

配错「输出层激活 × 损失」会在 `Model(...)` 构造时直接抛 `ValueError`，而不是静默训歪。

`ohem_loss` 是**批级**接口（一次吃一批概率向量、返回整批梯度），不能塞进 `Model(loss_fn=...)`
（那是逐样本调用约定）；类别不平衡的两个可选项已经接进映射表，用法：

> 所有损失都遵循和 `mse_loss` 相同的返回约定：`(loss, grad)`，
> 也都可以当成 `Model(loss_fn=...)` 的自定义可调用对象用。
> 额外超参（`focal` 的 `gamma`/`alpha`、`ce` 的 `label_smoothing`、`huber`/`smooth_l1` 的 `delta`、
> `quantile` 的 `quantile`）
> 通过 `Model(..., loss_kwargs={"gamma": 1.5})` 传入，会被转发到损失函数上。

> 搭配约束：`softmax` 输出层的反向使用的是 **softmax + cross_entropy 的联合梯度**，
> 因此只能与 `cross_entropy` 搭配。`Model(loss_fn='mse')` + softmax 输出层会**直接报错**，
> 而不是静默用错误梯度训练；需要 MSE 时请把输出层激活设为 `linear`。
>
> `Model` 的 CE 路径会自动规范化 target：整数索引、`[0.0, 1.0]` 这类浮点标签、
> 长度为 1 的列表会被转成类别索引；**长度等于类数的 one-hot / 概率向量会原样直通**
> 给 `cross_entropy_loss` 的软标签分支，因此 label smoothing / 蒸馏的软目标不会被
> argmax 硬化。软标签的联合梯度按 `Σt·p − t` 计算（全 0 目标得到全 0 梯度）。
> 该守卫对 `loss_fn='mse'`、`loss_fn=mse_loss` 与 `functools.partial(mse_loss)` 都生效。

```python
# 新损失直接用名字 + loss_kwargs
m = AI.Model(AI.MLP([8, 16, 3], activations=["relu", "softmax"]),
             loss_fn="focal", task="classification", lr=0.01,
             loss_kwargs={"gamma": 1.5, "alpha": [1.0, 2.0, 1.0]})
m.fit(data=train, epochs=20)

# 二元分类：sigmoid 输出层 + bce
m2 = AI.Model(AI.MLP([8, 16, 1], activations=["relu", "sigmoid"]),
              loss_fn="bce", task="regression", lr=0.01)

# 长尾分类：按类频次加权（class_counts 用训练集的真实计数）
m3 = AI.Model(AI.MLP([8, 16, 3], activations=["relu", "softmax"]),
              loss_fn="class_balanced", task="classification",
              loss_kwargs={"class_counts": [900, 90, 10], "beta": 0.999})
```

### 评估指标

```python
import AI

# 分类：一次拿到全部常用指标
model = AI.Model(AI.MLP([2, 8, 3], activations=["relu", "softmax"]),
                 task="classification", loss_fn="cross_entropy")
model.fit(data=train_data, epochs=50)
print(model.evaluate_all(x_test, y_test))
# {'task': 'classification', 'n_samples': 30, 'accuracy': 0.93, 'precision': 0.93,
#  'recall': 0.93, 'f1': 0.93, 'confusion_matrix': [[...]], 'mcc': 0.89,
#  'top_k_accuracy': 1.0, 'roc_auc': 0.98}

# 单独用某个指标
AI.precision_recall_f1(y_true, y_pred, average="macro")
AI.confusion_matrix(y_true, y_pred, num_classes=3)
AI.roc_auc_score(y_binary, y_score)
f1 = AI.precision_recall_f1(y_true, y_pred)["f1"]

# 回归
AI.mean_absolute_error(y_true, y_pred)
AI.r2_score(y_true, y_pred)

# 生成质量（token 列表或空白分隔的字符串都行）
AI.bleu_score(pred_tokens, ref_tokens, max_n=4)
AI.rouge_l(pred_tokens, ref_tokens)
AI.sequence_accuracy(preds, targets)      # 整条序列全对才算对
```

> 指标全部是**纯标准库实现**，输入会做防御性校验（空列表、长度不符、负类别索引、
> 非有限数值都直接报 `ValueError`，不会静默给出 `nan`）。

### 工具函数

| 函数 | 说明 |
|------|------|
| `split_data(data, val_ratio, test_ratio, seed, stratify=None)` | 数据集拆分（`stratify` 分层） |
| `DataLoader(dataset, batch_size, shuffle, seed, collate_fn=None)` | 批量迭代器（可自定义批次拼装） |
| `train(model, loss_fn, optimizer, data, ...)` | 底层训练循环（Early Stopping + 验证集） |
| `evaluate(model, loss_fn, data)` | 计算平均损失 |
| `save_model(model, path, task=None, optimizer=None, quiet=False, loss=None, loss_kwargs=None)` | MLP 的 JSON 持久化；传 `optimizer=` 时一并写入动量状态，传 `loss=` / `loss_kwargs=` 时记下损失配置；旧格式（无逐层 activation）可用 `default_output_activation='softmax'` 还原分类头 |
| `CausalLM.save(path, include_optimizer=False)` / `CausalLM.load(path, restore_optimizer=False)` | 语言模型的 JSON 存档与恢复（`Seq2SeqLM` 同名方法） |
| `save_conv_model(conv, fc, path)` / `load_conv_model(path, default_output_activation=None)` | CNN + MLP 的 JSON 持久化，旧格式同上；会保存 `groups`（分组/深度卷积可原样加载，没有该字段的旧存档按 1 处理） |
| `_optimizer_state(optimizer)` / `load_optimizer_state(optimizer, state)` | 读取 / 还原 Adam、AdamW 的动量与步数（其它优化器返回 `None`） |
| `Model.save(path)` / `Model.load(path, task='auto', restore_optimizer=False, loss_kwargs=None)` | 高层封装；`task='auto'` 时优先采用文件中记录的 task；`restore_optimizer=True` 做逐位续训；损失函数名与 `loss_kwargs` 随存档往返，显式传参可覆盖 |

| `clip_grad_by_norm(grads, max_norm)` | 梯度范数裁剪 |
| `clip_grad_by_value(grads, clip_val)` | 梯度值裁剪 |
| `compare_with_pytorch()` | 与 PyTorch 逐层数值对齐（可选，装了 torch 才跑） |
| `k_fold_indices(n, n_splits, shuffle, seed, stratify)` | K 折交叉验证索引 |
| `accuracy_score` / `precision_recall_f1` / `confusion_matrix` / `matthews_corrcoef` | 分类指标；`precision_recall_f1` 支持 `macro` / `micro` / `weighted` / `none` |
| `roc_auc_score` / `top_k_accuracy` | 二分类 AUC（并列取平均秩）、Top-k 准确率 |
| `mean_absolute_error` / `root_mean_squared_error` / `mean_absolute_percentage_error` / `r2_score` | 回归指标，多输出按全部元素平均；常数目标时 R² 返回 1.0 或 0.0 |
| `sequence_accuracy(preds, targets)` | 整条序列全对才算对（Seq2Seq 常用） |
| `bleu_score(pred, ref, max_n=4)` / `rouge_l(pred, ref)` | 生成质量：BLEU（含简短惩罚）与 ROUGE-L F1 |
| `evaluate_all(model, x, y)` | 按任务一次算完：回归 `mae/rmse/mape/r2`；分类 `accuracy/precision/recall/f1/confusion_matrix/mcc`，二分类加 `roc_auc`、多分类加 `top_k_accuracy`；`Model.evaluate_all(x, y)` 同名方法 |
| `StandardScaler` / `MinMaxScaler` | 逐特征标准化 / 归一化（`fit`/`transform`/`inverse_transform`） |
| `ascii_plot(history)` | 把训练曲线画成终端 ASCII 图 |
| `set_seed(seed)` | 设定全局 `random` 种子；注意训练可复现还需要给 `fit(seed=)`（两者作用不同） |
| `infer_shapes(model, x)` | 一次前向推断输入/输出形状、每层形状与激活名、参数量，并返回输出 |
| `predict_batch(model, x, batch_size=64)` | 分块推理，结果与 `predict` 逐位一致（大输入避免一次性占内存） |
| `activation_stats(model, x)` | 每层激活的 `mean/std/min/max/zero_ratio`，用于找塌陷的层 |
| `dead_units(model, x, atol=1e-12)` | 每层恒为 0 的单元下标与比例（ReLU 死单元诊断） |
| `Callback` | 训练回调基类，子类覆写需要的钩子即可 |
| `EarlyStopping(monitor, patience, min_delta, mode)` | 监控指标连续 `patience` 轮不改善就请求停止（`should_stop` / `stopped_epoch`），不改权重 |
| `ModelCheckpoint(path, every, monitor, save_best_only)` | 每 `every` 轮存一份（含优化器状态）；`save_best_only` 只存最优；产物可直接 `resume_from` |
| `LoggingCallback(logger)` | 每轮指标写进标准库 `logging`（默认 logger `ai.train`） |
| `HistoryCallback()` | 把每轮 logs 收进 `records`（画曲线 / 导出 CSV 用） |
| `grad_check(module, x, y=None)` | 中心差分校验解析梯度（MLP / Model / CausalLM，以及 `Seq2SeqLM`（传 `(src, tgt)` 二元组））；解析与数值都 ≈0 的参数记进 `zeros` 不参与相对误差 |
| `build_model(config, data=None)` | 按 dict（或 JSON 路径）建 `Model`：`hidden`/`activations`/`task`/`loss`/`loss_kwargs`/`optimizer`/`lr`；维度可写死也可从 `data` 推断，未知配置项直接报错 |
| `train_from_config(config, data=..., x=..., y=...)` | 按配置建模型并训练，返回已训练好的 `Model`；`seed` 同时固定初始化与打乱，`lr_scheduler` 支持 step/exponential/cosine/warmup/plateau |
| `hyperparameter_search(base, grid, data=...)` | 在配置空间里搜超参：`method="grid"` 全组合 / `"random"` 随机抽 `n_trials` 组（给 `seed` 可复现）；指标支持 `val_loss`/`train_loss`/`grad_norm`（越小越好）与 `accuracy`（越大越好） |
| `attention_maps(module, x, layer_index=None)` | 跑一次前向取回每头注意力权重；支持 `MultiHeadAttention` / `TransformerEncoderLayer` / `TransformerEncoder` / `CausalLM`，采样期间切 eval 以避开 dropout 噪声、结束后恢复原模式 |
| `attention_ascii(maps, head=0, max_width=48)` | 把注意力矩阵画成字符热图（每行按自身最大值缩放，支持 `max_width` 抽样） |
| `profile_model(model, x, backward=False)` | 逐层前向/反向墙钟耗时 + 参数量；`repeat` 取最小值降抖动（纯计时，无内存统计） |
| `lr_finder(model, x=..., y=..., start_lr, end_lr, steps)` | LR range test：等比扫 lr、记录平滑 loss，按 fastai 规则推荐「最低点 / 10」；会改动权重，结束后恢复 `optimizer.lr` |
| `mlp.register_forward_hook(i, fn)` / `register_backward_hook(i, fn)` | 第 i 层前向/反向后的回调（`fn(layer, output)` / `fn(layer, delta, grad_w, grad_b)`）；不注册时只多一次真假判断，`Model` 上同名方法转发给内部的 `MLP` |
| `estimate_flops(model, input_shape)` | 前向乘加次数（MACs）/ 参数量 / 逐层明细；支持 MLP 与 `Sequential`（`Layer`/`Conv1d`/`Conv2d`（含 `groups`）/`ConvTranspose2d`/各种池化/`BatchNorm1d`/`LayerNorm`/`GroupNorm`/`InstanceNorm`/`Dropout`/`DropPath`/`Flatten`），不认识的模块报 TypeError 而不是静默估算；`Flatten` 只接受 `(C, H, W)`，传一维/二维会明确报错（`Flatten.forward` 本来也跑不了） |
| `augment_noise(x, sigma=0.01)` | 加性高斯噪声（单样本或一批样本，不改入参） |
| `augment_image(x, flip=True, max_shift=1)` | 图像 `(C,H,W)` 的随机翻转 / 平移（越界补 0） |
| `mixup(x1, y1, x2, y2, alpha=0.2)` | Mixup：返回混合样本 + 软标签向量（可直接喂 CE 的软标签分支）；`lam ~ Beta(alpha, alpha)`，恒在 `[0,1]` 内 |
| `augment_cutmix(x1, y1, x2, y2, alpha=1.0)` | CutMix：从 x2 剪一块矩形贴到 x1，`lam` 按**实际贴入面积**重算，标签同步混合 |
| `augment_cutout(x, num_holes=1, hole_size=0.25)` | Cutout：随机抹掉若干个正方形区域（`hole_size` 是相对短边比例） |
| `augment_randaugment(x, n=2, magnitude=9.0)` | RandAugment：随机挑 n 个操作（brightness / contrast / translate / rotate90 / identity）并返回实际用到的操作名 |
| `uncertainty_weighting(losses, log_vars)` | Kendall 多任务不确定性加权：`Σ exp(-s_i)·L_i + s_i`，返回 `(total, weights, d_losses, d_log_vars)` 可直接反传 |
| `gradnorm_weights(losses, grad_norms, alpha=1.5, lr=0.025, weights=None, G0=None)` | GradNorm：按 loss 比例给出目标梯度范数并更新任务权重，最后归一化到均值 1；返回 `weights`/`targets`/`loss`/`gradients` |
| `pcgrad(grad_a, grad_b, eps=1e-12)` | PCGrad 冲突投影：两任务梯度内积为负时互相投影掉冲突分量，返回投影后的两个梯度（嵌套 list 也支持） |
| `init_lecun_normal(target, fan_in=None)` | LeCun 正态初始化（`std = sqrt(1/fan_in)`），`fan_in` 缺省取层输入维度 |
| `init_lecun_uniform(target, fan_in=None)` | LeCun 均匀初始化（`U(±sqrt(3/fan_in))`，方差与上面一致） |
| `init_orthogonal(target, gain=1.0)` | 正交初始化（Gram-Schmidt）；`fan_out > fan_in` 时改按列正交（此时每行模长不再是 `gain`） |
| `init_sparse(target, sparsity=0.9, std=None)` | 稀疏初始化：零元素比例恰为 `sparsity`，非零项按 `sqrt(1/fan_in)` 正态 |
| `init_trunc_normal(target, std=0.02, cutoff=2.0, max_tries=100)` | 截断正态（拒绝采样，全部落在 `±cutoff·std` 内），ViT 类模型的常用初始化 |
| `zero_bias(target)` | 把所有层的 `biases` 清零 |
| `zero_init_residual(block)` | 残差分支零初始化：把块里**最后一个**分支 BN 的 `gamma` 置零（`BasicBlock2d` 零 `bn2`、`Bottleneck2d` 零 `bn3`）；捷径投影的 `short_bn` 不动 |
| `lsuv_init(target, x, tol=1e-2, max_iter=20)` | LSUV：逐层把**预激活**整批拉成均值 0 / std 1（权重只整体缩放，方向不变）；支持 `Layer`/`MLP` 与单样本/批输入 |
| `fixup_init(target, num_blocks=None)` | Fixup：残差分支最后一层按 `1/√n` 缩放并把该层偏置置零；作用在 `BasicBlock2d`/`Bottleneck2d`/`DenseBlock` 上 |
| `mimic_init(student, teacher)` | NET2NET 宽度/深度迁移（MLP）：老师权重块复制到左上角，新增行列用小随机补齐；激活不一致直接报错 |

> 旧文件不含逐层 `activation` 时，输出层回退优先级为：显式参数 > 文件里的 `task`
> （`classification` 用 softmax）> `linear`。既不传参数、文件里也没有 `task` 的旧分类器
> 会被当成回归模型，请显式传 `default_output_activation='softmax'`。

### 多任务权重

多任务训练里各任务 loss 量级不同，直接相加会被大 loss 主导。三个工具分别处理
「量级」「平衡」「冲突」：

```python
# 1) 不确定性加权（Kendall & Gal）：让模型自己学每个任务该占多少权重
total, w, d_losses, d_log_vars = AI.uncertainty_weighting([1.0, 2.0], [0.0, 0.0])
# total = Σ exp(-s_i)·L_i + s_i；d_log_vars 可直接反传到 log_vars

# 2) GradNorm：按各任务 loss 的相对大小定目标梯度范数，再更新权重
res = AI.gradnorm_weights([1.0, 2.0, 4.0], [3.0, 3.0, 3.0], alpha=1.5)
print(res["weights"], res["targets"])   # 权重均值恒为 1

# 3) PCGrad：两个任务梯度打架时，把冲突分量投影掉
ga, gb = AI.pcgrad([1.0, 0.0], [-1.0, 1.0])
```

要点：`uncertainty_weighting` 的 `d_log_vars` 是 `1 - exp(-s_i)·L_i`，反传前先
`sigmoid`/`log` 变换到你自己的参数化；`gradnorm_weights` 的 `target_i = Ḡ·r_i`
只在各任务 loss 相等时才和 `grad_norms` 重合，一般情形请用返回的 `gradients`
（`sign(G-target)·G0/r`）而不是自己推；`pcgrad` 只处理**两两**冲突，多任务请循环合并。

### 学习率调度器

| 调度器 | 说明 |
|------|------|
| `StepLR(optimizer, step_size, gamma)` | 阶梯衰减 |
| `ExponentialLR(optimizer, gamma)` | 指数衰减 |
| `CosineAnnealingLR(optimizer, T_max)` | 余弦退火 |
| `WarmupLR(optimizer, warmup_epochs, start_factor, base_scheduler)` | 线性 warmup；到期后交给 `base_scheduler`（可与上面任意一个组合） |

### 文本处理

| 组件 | 说明 |
|------|------|
| `TextProcessor` | 字符级词表、one-hot 编码、滑动窗口 |
| `BytePairTokenizer` | 字节级 BPE 分词器：`train(text, vocab_size)` 学合并、`encode`/`decode` 无损往返任意 UTF-8、`encode_corpus(text, seq_len)` 直接产出 `CausalLM.fit` 用的序列、`save`/`load` 存 JSON；手工构造时每条规则只能引用**之前已定义**的 id（前向引用报 `ValueError`，不会漏成 `KeyError`） |
| `WordPieceTokenizer` | BERT 风格子词分词器：`train(text, vocab_size)` 学词表、`encode`/`decode`/`encode_pair`、`save`/`load`；词内非首字符带 `##` 前缀，词表外落到 `[UNK]` |
| `generate_text(network, tp, start, length)` | 自回归文本生成（`generate_text_with_mlp` 的别名，保留以兼容旧调用） |
| `generate_text_with_mlp(model, tp, start, length, window_size, temp)` | 自回归文本生成（滑动窗口上下文，只需 `forward`） |
| `_softmax_with_temp(logits, temperature)` | 带温度的 Softmax |
| `sample_index(probs)` | 随机采样 |

```python
tok = AI.BytePairTokenizer().train(corpus, vocab_size=512)   # 语料小会提前停
ids = tok.encode("hello 世界")            # 任意 UTF-8 都不缺字
assert tok.decode(ids) == "hello 世界"

lm = AI.CausalLM(vocab_size=tok.vocab_size, d_model=32, num_heads=4)
lm.fit(tok.encode_corpus(corpus, seq_len=64), epochs=20, lr=3e-3)
print(tok.decode(lm.generate(tok.encode("hello"), max_new_tokens=16, top_k=4)))

tok.save("bpe.json")
tok = AI.BytePairTokenizer.load("bpe.json")
```

字节级的好处是**没有词表外字符**：中文、emoji、换行都落在 256 个基础字节 id 里。
合并只在词块（字母数字 / 空白 / 其它符号的连续段）内部学习，所以不会学到
「字母+空格」这种跨词 token；语料里重复片段耗尽时会提前停下，vocab_size 只是上限。

WordPiece 走另一条路：词表里是**字符串**，词内非首字符带 `##` 前缀，
所以 `playing` 能拆成 `play + ##ing`，而 `ing` 不会和它混为一谈。
训练打分用 BERT 原始的 `频次 / (左频次 × 右频次)`，高频组合优先合并。

```python
wp = AI.WordPieceTokenizer().train(corpus, vocab_size=512)
ids = wp.encode("the quick brown fox")
print([wp.tokens[i] for i in ids])      # 常见词整体成 token，词内子词带 ##
assert wp.decode(ids) == 'the quick brown fox'

pair = wp.encode_pair('the dog', 'a fox')   # [CLS] a [SEP] b [SEP]
print(pair[0] == wp.cls_id, pair[-1] == wp.sep_id)
```

与 BPE 的分工：BPE 字节级、**永不产生 `[UNK]`**，适合从零训练语言模型；
WordPiece 有可读词表与特殊标记，适合接 BERT 式双向编码器与 `MaskedLM`。

## 训练基础设施（W0）

```python
AI.set_seed(42)

class Log(AI.Callback):
    def on_epoch_end(self, epoch, logs):
        print(epoch, round(logs["train_loss"], 4), round(logs["lr"], 6))

model = make_classifier(2, 2, hidden_layers=(8,))
model.fit(x, y, epochs=20, callbacks=[Log()], seed=42,
          record_layer_norms=True, checkpoint_path="ck.json")

# 只恢复、不继续训练
resumed = make_classifier(2, 2, hidden_layers=(8,))
resumed.fit(x, y, epochs=0, resume_from="ck.json")

AI.infer_shapes(model, x[0])          # {'input_shape': [2], 'output_shape': [2], 'params': 42, ...}
AI.predict_batch(model, x, batch_size=8)
AI.activation_stats(model, x)          # 每层 mean/std/min/max/zero_ratio
AI.dead_units(model, x)                # 恒为 0 的单元下标与比例
```

内置回调（也可自己继承 `Callback`）：

```python
cbs = [AI.EarlyStopping(patience=3),
       AI.ModelCheckpoint("ck.json", every=2),
       AI.LoggingCallback(),
       AI.HistoryCallback()]
model.fit(x, y, epochs=50, callbacks=cbs, seed=42)
model.history["early_stop"]       # 不存在；停止信息在回调对象上
cbs[0].should_stop, cbs[0].stopped_epoch
cbs[3].records[-1]                 # 最后一轮的完整指标
```

四条约定：

- `seed` 与 `set_seed` 分工不同：前者管 `fit` 内部的打乱与（配合初始化的）可复现，后者只设全局 `random`。
- `record_layer_norms` 只记录，**不改变训练数值**（与不开启时 `train_loss` 完全相同）。
- `resume_from` 恢复权重 + 优化器状态；`epochs=0` 就是“只恢复”，用于做推理或对比。
- `mixup_alpha` 只支持分类任务与索引型损失（one-hot / 概率向量标签也行）；回归任务直接报错。

## 配置驱动训练

把结构、损失、优化器、调度器和训练超参写进一个 dict（或 JSON 文件），
一次调用即可建模型并训练：

```python
cfg = {
    "input_dim": 4, "output_dim": 3, "hidden": [16, 8],
    "activations": ["relu", "relu"], "task": "classification",
    "loss": "cross_entropy", "loss_kwargs": {"label_smoothing": 0.1},
    "optimizer": "adamw", "lr": 0.01, "epochs": 80, "batch_size": 16,
    "val_split": 0.15, "patience": 0, "seed": 7,
    "lr_scheduler": {"type": "warmup", "warmup_epochs": 5,
                     "base_scheduler": {"type": "cosine", "t_max": 60}},
}
model = AI.train_from_config(cfg, data=dataset)
print(model.history['train_loss'][-1])
```

约定：**未知配置项直接报错**（避免把 `epoch` 拼错成静默忽略），
`input_dim`/`output_dim` 可以只给一个、另一个从 `data` 推断，
`seed` 同时固定权重初始化与每轮打乱，因此同配置同数据可复现。

## 专家混合（MoE）

`MoE` 用一条路由网络给每个 token 选 top-k 个专家，只让这几条路径参与计算：

```python
moe = AI.MoE(d_model=32, d_ff=64, num_experts=4, top_k=2)
out = moe.forward(hidden)            # (T, d_model)，与 FeedForward 同形
loss, dout = ..., ...               # 由上层给出对 out 的梯度
dx = moe.backward(dout)             # (T, d_model)，可继续往前传
moe.adam_step(lr=1e-3)
```

与 `FeedForward` 的区别：参数量随 `num_experts` 线性增长，但每个 token 只走
`top_k` 条路径，所以计算量按 `top_k` 增长而不是按专家数。路由权重是在
**被选中的专家**上做的 softmax（Switch/GShard 风格），未被选中的专家梯度恰为 0，
这一点在测试里单独断言过。

实现上这是教学版本：token 先按专家分组再逐组前向，没有做 scatter/gather 的
批量矩阵化。token 数不多时很快，超长序列会比稠密 FFN 慢；真实实现需要更复杂的
分组矩阵乘法。

## 生成模型：VAE / GAN / DDPM

三个生成模型都只用标准库实现，可单独训练并采样：

```python
# 变分自编码器：重建 + KL，重参数化采样
vae = AI.VAE(input_dim=2, hidden_dim=32, latent_dim=4, beta=1.0)
history = vae.fit(data, epochs=300, lr=0.01, seed=7)
samples = vae.sample(8)                     # 从 N(0,1) 采样再解码

# 生成对抗网络：D 学真伪，G 骗过 D
gan = AI.GAN(input_dim=2, hidden_dim=32, latent_dim=4)
for _ in range(500):
    d_loss, g_loss = gan.step(real_batch)

# 扩散模型：学会预测噪声，再反向去噪
ddpm = AI.DDPM(data_dim=2, hidden_dim=32, steps=100)
ddpm.fit(data, epochs=500, lr=1e-3, seed=3)
samples = ddpm.sample(8)
```

三者的梯度都用中心差分验证过（`dec2`/`enc2`、D 与 G、扩散网络的输出层），
训练曲线在探针里分别降 6.9×、把生成均值从 3.16 拉到 1.75、降 4.4×。采样端另有 `ddim_sample(steps=20, eta=0)`：`eta=0` 时完全确定性、且只需训练步数的一个子序列。DDPM 的
`train_step` 允许显式传入 `t` 与 `noise`，就是为了让测试能固定随机性做差分。

## 序列标注与多标签分类

`CausalLM` / `TransformerEncoder` 输出的是逐位置隐状态，接上分类头就能做 NER、
词性标注；单个向量接多标签头则处理「一条样本命中多个标签」的场景。

```python
head = AI.TokenClassificationHead(d_model=32, num_labels=5)
logits = head.forward(hidden)                    # (T, 5)
loss, d_logits = head.loss_grad(logits, tags, mask=valid)   # 右填充用 mask
d_hidden = head.backward(d_logits)               # (T, d_model)，可继续往前传
head.adam_step(lr=0.01)

multi = AI.MultiLabelHead(fan_in=32, num_labels=3)
probs = multi.forward(feature)                   # [0.12, 0.87, 0.44]
loss, d_logits = multi.loss_grad(probs, [0, 1, 0])
d_feature = multi.backward(d_logits)
print(multi.predict(probs))                      # [0, 1, 0]
```

要点：两个头都自带 `zero_grad` / `sgd_step` / `adam_step` / `param_count`；
序列标注的 `mask` 语义与 `CausalLM` 的 `pad_mask` 一致（右填充、False 不参与），
`predict` 对 mask 位置返回 `-1`；多标签用 BCE-with-logits 而不是 softmax+CE，
因为标签之间不该互相竞争。两者都已在 `grad_check` 覆盖范围之外，
所以测试里单独做了中心差分校验。

## 超参搜索

在配置空间上做网格或随机搜索，直接复用 `train_from_config`：

```python
res = AI.hyperparameter_search(
    base_config,
    {"lr": [1e-3, 5e-3, 1e-2], "hidden": [[16], [32, 16]],
     "activation": ["relu", "gelu"]},
    data=dataset, method="random", n_trials=8, metric="val_loss", seed=3)
print(res["best"]["params"], res["best"]["score"])
for t in res["trials"]:
    print(t["trial"], t["params"], round(t["score"], 4))
```

约定：`grid` 的键必须是合法配置项（写错直接报错，不会静默忽略）；
`metric` 决定方向（`val_loss`/`train_loss`/`grad_norm` 取最小，`accuracy` 取最大）；
用 `val_loss` 时配置里必须有验证集，用 `accuracy` 时必须同时给 `x`/`y`；
`seed` 同时控制随机搜索的抽样与每次训练的初始化，因此整轮可复现。
返回 `{"best", "trials", "metric", "mode", "params_names"}`，
每个 trial 里带自己的 `history`，方便事后画曲线。

## 底层 API 示例

### CNN

```python
conv = Conv2d(in_channels=1, out_channels=2, kernel_size=2, stride=1)
pool = MaxPool2d(kernel_size=2)
flatten = Flatten()
fc = MLP([2, 4, 2], activations=["leaky_relu", "softmax"])

# 前向: img (1, 4, 4) → conv → pool → flatten → fc
x = conv.forward(img)
x = pool.forward(x)
x = flatten.forward(x)
probs = fc.forward(x)
```

### Transformer

```python
d_model = 4
pos_enc = PositionalEncoding(d_model=d_model)
encoder = TransformerEncoder(d_model=d_model, num_heads=2, d_ff=8, num_layers=1)

x = pos_enc.forward(seq)          # 加位置编码
x = encoder.forward(x)            # 经过 Transformer
pooled = [sum(t[j] for t in x) / len(x) for j in range(d_model)]   # 全局平均池化
head = MLP([d_model, 8, 2], activations=["leaky_relu", "linear"])  # 分类头
logits = head.forward(pooled)
```

### RNN / LSTM / GRU

```python
rnn = RNN(input_size=1, hidden_size=4, num_layers=1)
lstm = LSTM(input_size=1, hidden_size=4, num_layers=1)
gru = GRU(input_size=1, hidden_size=4, num_layers=1)

outputs, _ = rnn.forward(seq)     # outputs: 每个时间步的隐藏状态
last_h = outputs[-1]              # 取最后一步做预测
```

**BPTT 写法**：`forward` 会把每一步、每一层的 cell 缓存存进 `step_cache`，
反向时用 `restore_cache` 直接恢复即可，不需要再 forward 一次来"重建"缓存：

```python
rnn = RNN(input_size=1, hidden_size=4, num_layers=1)   # 承接上例
seq = [[0.1], [0.2], [0.3], [0.4]]

outputs, _ = rnn.forward(seq)
rnn.cells[0].zero_grad()          # 整个 BPTT 只清一次
dh = [0.1] * rnn.hidden_size
for t in reversed(range(len(seq))):
    rnn.cells[0].restore_cache(rnn.step_cache[t][0])   # 复用前向缓存
    _, dh = rnn.cells[0].backward(dh)

grad_W_ih, grad_W_hh, grad_b_ih, grad_b_hh = rnn.cells[0].get_grads()
```

> 两个容易踩的点：
> 1. `backward()` 会把梯度**原地累加**到 cell 自己的缓冲区，所以整个 BPTT 只需 `zero_grad()` 一次；
> 2. `get_grads()` 返回的**就是这些缓冲区本身**（不是副本），因此**不要**再对它做手工求和/累加——
>    那等于把缓冲区加到它自己身上，梯度会按 2 的幂被放大（T=10 时实测放大 264 倍）。

### 文本生成

```python
tp = TextProcessor(text)
dataset = tp.prepare_data(text, window_size=3)

# 训练字符级语言模型...
model = MLP([tp.vocab_size * window_size, 64, tp.vocab_size],
            activations=["leaky_relu", "softmax"])

# 自回归生成
generated = generate_text(model, tp, "君不见", length=15, temp=0.8)
```

## 运行示例

```bash
python AI.py
```

依次运行 14 段演示：回归（`Model.fit`）→ 分类（`make_classifier` + Softmax/CE）→ CNN（卷积→池化→展平→MLP）→ RNN & LSTM 序列预测 → 模型保存/加载 → GRU 序列预测 → Dropout + LayerNorm → 梯度裁剪 + StepLR → `quick_train` 一行训练 → Transformer 序列分类 → 因果语言模型（`CausalLM`）→ 迷你 GPT（BPE 分词 + 存档续训 + 束搜索）→ 字符级文本生成 → PyTorch 数值对齐。

## 测试

```bash
python test_ai.py     # 等价于 pytest test_ai.py
```

1040 个测试覆盖所有组件，其中这几类是**数值校验**而不是只查形状：

| 测试类 | 校验内容 |
|--------|----------|
| `TestLayerNumericalGradients` | `Layer` / `MLP` 的 `grad_w / grad_b`、反向传播链（`dL/dx`）vs 数值梯度；含 softmax+CE 联合梯度约定 |
| `TestTransformerEncoderGradients` | `TransformerEncoderLayer` / `TransformerEncoder` 的 `d_x` 与 LayerNorm 参数梯度 vs 数值梯度（**seq_len>1 必须测**，见下方实现说明） |
| `TestAttentionGradients` | `MultiHeadAttention` 的 `W_q/W_k/W_v/W_o` 解析梯度 vs 数值梯度 |
| `TestConvDilation` | 空洞卷积的等效感受野与朴素参考实现逐元素一致、`dilation=1` 与旧行为逐位相同、`dx`/`grad_w` vs 数值梯度、非法 dilation 报错、存档带走 `dilation`/`groups` 且前向逐位一致 |
| `TestBatchNorm2d` | 训练模式输出均值 0/方差 1、running 统计量按 momentum 更新、`eval()` 用 running、`dx` vs 数值梯度、`grad_gamma`/`grad_beta` 公式、`delta_index=0`、非法构造与未 forward 就 backward 报错、`estimate_flops` 能算 |
| `TestConv2dGradients` | `Conv2d` 的 `grad_w / grad_b / dx` vs 数值梯度（含 padding 与 stride=2） |
| `TestBatchNormTraining` | `BatchNorm1d` 训练路径的 `dx / gamma / beta` vs 数值梯度，并与朴素实现逐位一致 |
| `TestStepCache` / `TestCellGradSemantics` | `restore_cache` 与「重新前向」逐位一致；一次 `zero_grad` + 逐步 backward == 各步梯度之和 |
| `TestValDataFormats` / `TestCrossEntropyTargets` | `val_data` 三种写法、CE 的 target 规范化 |
| `TestCausalMask` | 因果掩码的**因果性**（改未来 token 不改过去输出）、掩码权重精确为 0、`W_v` 梯度 vs 数值梯度 |
| `TestCausalLM` | 小语料过拟合（loss → <0.05）、贪心续写接上周期、前缀 logits 与未来无关、生成可复现、非法输入 |
| `TestMaskedLM` | 遮罩的 80/10/10 比例与 `mask_id` 在词表外、双向性（改末位影响前面的 logits）、`out.W` 梯度 vs 数值、全遮罩记忆后能填空还原、`evaluate` 可复现且不改模式、19 块参数去重、非法输入 |
| `TestAttentionMaps` | MHA 形状/行和/因果上三角、eval 两次一致、TEL 与 Encoder 多层与 `layer_index`、CausalLM 只读且模式复原、ASCII 结构与抽样、六类非法输入 |
| `TestConfigDriven` | `build_model` 结构与维度推断（部分给、全不给、显式覆盖）、八类配置错误、五种调度器构造、`train_from_config` 训练与收敛、同 seed 曲线与权重逐位一致、JSON 路径与 `data`/`x` 冲突 |
| `TestHyperparameterSearch` | 网格全组合与参数覆盖、`best` 确实是最优分、随机搜索同 seed 可复现、`accuracy` 取最大、全局 RNG 不被推进、十类非法输入（含 `val_split=0` 用 `val_loss`、缺 `x`/`y` 用 `accuracy`） |
| `TestTokenClassificationHead` | 形状与参数校验、W/b 中心差分、`mask` 只算有效 token 且被屏蔽行梯度为 0、`predict` 的 argmax 与 mask 语义、300 步训练后 loss 降 20 倍并全对 |
| `TestMultiLabelHead` | 概率范围与校验、BCE 梯度中心差分、阈值预测（含非法阈值）、400 步训练后 loss 降 3 倍并全对 |
| `TestMoE` | 形状与参数校验（含 `top_k > num_experts`、非法 `top_k`）、`top_k=1` 权重恒 1、路由权重与专家权重/输入的中心差分、未选中专家梯度恰为 0、200 步训练降 5 倍且至少两个专家被用到 |
| `TestVAE` | 重参数化 `z = mu + eps·std`、`KL ≥ 0`、`loss == 重构 + beta·KL`、`dec2`/`enc2` 权重中心差分、120 轮训练降 3 倍、采样形状与有限性、构造与输入校验 |
| `TestGAN` | D 输出落在 (0,1)、D 与 G 梯度中心差分、300 步后生成均值明显靠近数据、空 batch 报错 |
| `TestDDPM` | `q_sample` 与公式逐位一致、`t`/`x0`/`noise` 三类校验、噪声预测层梯度中心差分、150 轮训练降 2.5 倍、反向采样有限且均值在数据尺度内 |
| `TestDDIMSampling` | `eta=0` 同 seed 逐位相同、`eta=1` 引入随机、子序列 `steps` 与全步不同但有限、`n`/`eta`/`steps` 非法值 |
| `TestPredictionTypes` | `eps`/`v`/`x0` 三种参数的 `target()` 都能被 `pred_to_x0_eps()` 反解回 `(x0, eps)`、`eps` 目标就是噪声、非法 `prediction` |
| `TestMinSNR` | 权重 ∈ (0,1] 且随 t 单调、`γ` 越小越不放大、加权后 loss 恰为 `w×` 原 loss、非法 `γ` |
| `TestVQVAE` | `quantize` 最近邻与平局、三项损失闭式解、straight-through 与 `beta=0` 退化、码本与 decoder 梯度中心差分、warmup 不动码本、`histogram`、优化器、训练与非法输入 |
| `TestProfileModel` | 层数/总耗时/参数量、逐层耗时之和等于总耗时、反向统计、`Model` 自动解包、非法 repeat 与非网络报错 |
| `TestLRFinder` | history 长度与等比增长、`suggested_lr` 落在区间内、`optimizer.lr` 恢复、同 seed 可复现、全程下降时的提示、六类非法参数 |
| `TestAttentionDropout` | 注意力 dropout 的 mask 取值/缩放、`eval()` 关闭、train 随机性、**固定 mask 下 `W_v` 解析梯度 vs 数值梯度** |
| `TestModernComponents` | GELU 公式与导数、sigmoid 不溢出、`Linear`、`RMSNorm` 的 `dx / dgamma` vs 数值梯度 |
| `TestAdamWAndWarmup` | AdamW 零梯度只做衰减、`weight_decay=0` 与 Adam 逐位相同、`WarmupLR` 的爬坡/接力曲线 |
| `TestExtraLosses` | `gamma=0` 的 focal 退化为 CE、focal/CE-ls 梯度 vs 数值梯度、BCE 的 logits 稳定分支、Huber 两段、KL 公式 |
| `TestRoPE` | 增量前向 == 整段前向（逐位）、RoPE 改变输出、**Q/K 逆旋转后的梯度 vs 数值梯度** |
| `TestKVCache` | KV cache 生成 == 整段重算（含窗口滑动与 RoPE 两种情形）、训练后仍一致、生成不改参数 |
| `TestKVCacheMatrix` | KV cache == 整段重算在 7 种组合下都成立（GQA/MQA × RoPE × 滑动窗口 × max_len 8/4），束搜索在这些组合下也确定性 |
| `TestNewActivations` | SiLU/Mish/ELU/SELU 的前向数值、导数 vs 数值微分、`Layer.backward` 梯度、`FeedForward` 支持这四种激活、非法激活名仍报错 |
| `TestSwiGLU` | 门控全 0 时退化成 `b2`、六个参数的梯度 vs 数值、`dx` vs 数值、梯度累加语义、参数校验 |
| `TestCutMix` | 形状与标签混合 / 贴入像素来自第二张图且面积与 lam 一致 / 同 seed 可复现 / 非法参数 |
| `TestCutout` | 抹掉的像素数 / `num_holes=0` 等同拷贝 / 同 seed 可复现 / 非法参数 |
| `TestRandAugment` | 操作数与被修改值域 / identity 不动像素 / rotate90 四次复原 / 同 seed 可复现 / 非法参数 |
| `TestTripletLoss` | 满足间隔 ⇒ 零梯度 / 三个梯度 vs 数值 / 非法 margin 与长度 |
| `TestInfoNCELoss` | loss 与闭式公式一致 / 四个梯度 vs 数值 / 难负样本 loss 更大 / 非法参数 |
| `TestDiceLoss` | loss 与公式一致 / 梯度 vs 数值 / 完美预测≈0 / 形状不一致报错 |
| `TestCTCLoss` | loss 与穷举所有对齐路径一致 / 梯度 vs 数值 / 不可达与非法标签 |
| `TestLion` | Lion 一步手算 / 零梯度 / 符号步长 / 线性回归收敛 |
| `TestAdafactor` | 一步与独立公式一致 / 解耦衰减用更新前权重 / 收敛 |
| `TestReduceLROnPlateau` | 停滞降档 / 改善不降 / min_lr 下界 / mode=max / NaN 忽略 / 非法参数 |
| `TestLoRA` | B=0 等价原层 / A、B、dx 数值梯度 / 原层冻结 / merge 保输出 / 参数量 / 非法参数 |
| `TestDoRALinear` | B=0 等价原层 / `g_V`、`g_m`、`g_A`、`g_B` 数值梯度 / `denormalized()` 与 forward 一致 / `merge()` 保输出 / 参数量 |
| `TestAdaLoRALinear` | B=0 等价原层 / A、B 数值梯度 / `prune` 保留 top-k 且死秩全零 / 剪枝后前向等于存活秩之和 / 非法 `keep` |
| `TestSamplingExtras` | typical 含 argmax 且累计 ≥ mass / top-a 与 eta 的单调性 / mirostat 的 mu 单调与熵上界 / `mu` 更新方向与下限 / 非法参数 |
| `TestConstrainedDecoding` | token 掩码只留允许集合且概率和仍为 1 / 前缀约束的状态转移与 `eos_id` / 非前缀报错 / 与采样器叠加 / 非法输入 |
| `TestSpeculativeAccept` | `draft==target` 全接受 / 拒绝后按残差重采样 / 接受率 ≈ Σmin(p,q) / 非法输入 |
| `TestALiBi` | 输入全 0 时注意力权重等于 `softmax(slope·(t-j))` 的闭式解、因果掩码仍生效、`forward_step` 与 `forward` 末行逐位一致、`W_q` 梯度 vs 数值、`CausalLM(alibi=True)` 端到端训练、默认为关 |
| `TestFeedForwardActivation` | `FeedForward` 换 GELU/sigmoid 后的 `d_x` vs 数值梯度，`relu` 路径与旧实现逐位一致 |
| `TestGradCheck` | `grad_check` 对 MLP / Model / CausalLM 都返回 `ok`，且人为改坏梯度时能识别 |
| `TestRegressionLosses` | l1 / log-cosh / 分位数 / smooth-l1 的取值与梯度、`q=0.5` 等于半个 MAE、配 softmax 被拒 |
| `TestSeq2SeqPaddedBatch` | `Seq2SeqLM` 的 `pad_mask`：补齐后 loss/梯度逐位不变、`batch_size` 一步等于手工按 token 加权、mask 校验 |
| `TestGradCheckSeq2Seq` | `Seq2SeqLM` 的 `grad_check` 通过、覆盖解码器与输出层、零梯度（`b_k`）不误报、检查后模式复原 |
| `TestOptimizerPersistence` | 不传 `optimizer` 时存档与旧格式逐字段相同、Adam 状态往返、**续训与一次训完权重逐位相同**、类型/形状不匹配报错 |
| `TestWindowMask` | 窗口外权重恰为 0、窗口注意力保持因果性、`forward_step` == 整段前向最后一行（含 RoPE）、**带窗口时 `W_v` 梯度 vs 数值梯度** |
| `TestTopPSampling` | `top_p=1.0` 与旧行为逐位相同、nucleus 只留头部集合、cache 与重算路径一致、非法 `top_p` 报错 |
| `TestAvgPool2d` | 平均池化前向值、反向均分、`dx` vs 数值梯度、`GlobalAvgPool2d` 形状、非法 kernel/stride |
| `TestAdaptiveAvgPool2d` | 非整除分箱手算值、`(1,1)` 与 `GlobalAvgPool2d` 逐位相同、整除时与 `AvgPool2d` 一致、`dx` vs 数值梯度、非法 `out_size` |
| `TestPoolingExtras` | `GlobalMaxPool2d` 取最大与回传位置、`AdaptiveMaxPool2d` 与 `MaxPool2d`/`GlobalMaxPool2d` 逐位一致、反向、`LPPool2d` 的 `p=1` 退化为平均池化与 `p→∞` 逼近最大、梯度差分、7 类非法构造 |
| `TestSEBlock` | 形状与 `x*s` 缩放、通道权重在 `(0,1)`、`param_count`、`zero_grad`、`dx` 与参数梯度中心差分、通道数/未 `forward` 的报错 |
| `TestECABlock` | `x*s` 缩放与 `param_count`、`dx` 与 `w` 的中心差分、偶数 kernel/通道数/未 forward 报错 |
| `TestDepthwiseSeparableConv2d` | 输出形状与参数量公式（比标准卷积省）、`dx` 中心差分、`sgd_step`/`adam_step` 真的改权重 |
| `TestBasicBlock2d` | 全零卷积退化为 `relu(x)`（恒等捷径）、eval 模式 `dx` 中心差分、BN 拿到梯度、`param_count` 与非法参数 |
| `TestBatchNorm2dEvalBackward` | eval 模式 `dx` 是逐元素 `gamma/sqrt(var+eps)`（回归测试：曾误用训练公式）、`grad_gamma`/`grad_beta` 正确 |
| `TestBottleneck2d` | 输出形状与参数量（含投影捷径）、eval 模式 `dx` 中心差分、投影卷积与 `short_*` BN 拿到梯度、非法参数 |
| `TestAdaIN` | 输出逐通道均值=beta/std=gamma、`dx`/`d_gamma`/`d_beta` 中心差分、单向量输入（mean=x ⇒ dx=0）、长度校验 |
| `TestFiLM` | `gamma ⊙ x + beta` 与分组映射、`dx`/`d_gamma`/`d_beta` 解析值、`features % groups` 校验 |
| `TestConvTranspose1d` | 输出长度公式与单点铺开、`dx` 与 `grad_w` 中心差分、`output_padding >= stride` 报错 |
| `TestPool1d` | 最大/平均池化数值、floor 语义（长度不整除时丢弃尾窗）、梯度路由（最大只回一个位置、平均均分） |
| `TestPool3d` | 三维形状、第一窗口等于窗口最大/均值、两个方向梯度总量守恒、非法参数与未 forward 报错 |
| `TestConv3d` | 中心 tap 退化为平移不变的恒等映射、`param_count`、`dx` 与 `grad_w` 中心差分、非法 kernel/stride/padding 与通道数不符 |
| `TestConvTranspose3d` | 输出尺寸公式（含 `output_padding`）、单点散射位置、`dx` 与 `grad_w` 中心差分、`output_padding >= stride` 报错 |
| `TestCBAMBlock` | 空间支路退化时通道支路与 `SEBlock` 逐位相同、空间权重逐位置不同、`dx`/`sp_w`/`sp_b` 中心差分、参数量 20、`sgd_step`/`adam_step` 真的改参数、非法构造与二维输入报错 |
| `TestDenseBlock` | 拼接后通道数与前 `Cin` 通道原样保留、`dx`/conv `grad_w`/BN `gamma,beta` 中心差分、eval 用 running 统计量、非法参数与二维输入报错 |
| `TestConvNeXtBlock` | 逐位置 `xhat` 均值 0 且方差 `var/(var+eps)`、`pw2` 全零时输出等于输入、`dx`/LN `gamma,beta`/depthwise 权重中心差分、参数量公式与校验 |
| `TestScaleNorm` | 前向闭式解、`dx`/`dg` 中心差分、全零输入不崩、优化器步与 7 类非法输入 |
| `TestPReLU` | 正分支恒等、负分支乘 `a`、`da` 只在负分支累计、`dx`/`da` 中心差分、优化器步与 8 类非法输入 |
| `TestSnake` | 前向闭式解、`dx = dout·(1+sin2ax)`、`dx`/`da` 中心差分、`a→0` 退化恒等且 `da = dout·x²` |
| `TestMaxout` | 前向等于逐元素最大仿射值、argmax 路由（只有选中支路有梯度）、`dx`/`grad_a` 中心差分、优化器步与 6 类非法输入 |
| `TestDynamicReLU` | 斜率随样本变化、含耦合项的 `dx` 闭式解、`dx`/`w`/`b` 三组中心差分、优化器步与 5 类非法输入 |
| `TestFReLU` | 零初始化逐位等于 ReLU、`k×k` 卷积支路闭式解、`dx`/`grad_w` 中心差分、优化器步与 6 类非法输入 |
| `TestLSUV` | 预激活 `std→1`/`mean→0`、权重方向不变（缩放因子全局一致）、单样本与单层输入、7 类非法输入 |
| `TestFixupInit` | 只缩放残差分支最后一层且偏置置零、`1/√n` 系数（BasicBlock/Bottleneck/DenseBlock）、5 类非法输入 |
| `TestMimicInit` | 宽度复制左上角块、新增行列非零、变窄截断、迁移后仍能训练、激活不一致报错 |
| `TestSwiGLUGates` | 默认 swish 与旧实现逐位一致、gelu/relu/bilinear 门控值、各 gate 的 `grad_Wg` 中心差分、非法 gate 校验 |
| `TestInitFamily` | LeCun 正态/均匀的 `std` 与界内、正交矩阵行模长 `==gain` 且两两正交（fat 矩阵改验列正交）、稀疏零比例、截断正态界内、`zero_bias`、残差零初始化后退化为 `relu(x)`（Bottleneck 零 `bn3` 而非 `short_bn`）、非法输入、默认 He 初始化未被改动 |
| `TestWeightNorm` | 初始化即等于原层、`dv`（含 `v`/`g` 解耦项）/`dg`/`db` 中心差分、`dx = wᵀdout`、forward 不永久改写内层权重 |
| `TestSpectralNorm` | 前向 `(v/σ)x+b`、冻结 σ 时 `dv` 中心差分、未冻结时 σ 随权重变化、`power_iter`/维度/未 forward 校验 |
| `TestReZeroSkipInit` | `alpha=0` 时输出逐位等于输入、`x + 0.5·f(x)` 闭式解、`grad_alpha` 与 `dx` 中心差分、SkipInit 的 `beta=1/√n` 与不可学参数量、6 类非法输入 |
| `TestPrePostNorm` | Pre-LN 闭式解、Pre-LN 与 Post-LN 输出不同、`dx`（穿 LN）与 `norm.gamma` 中心差分、层数不符报错 |
| `TestHighway` | 闭式解、`param_count = d²+3d`、`dx`/`wg`/`w` 中心差分、`sgd_step`/`adam_step` 真改参数、4 类非法输入 |
| `TestUNet` | 输出形状与参数量、`dx` 中心差分（无 BN / 有 BN 推理模式）、编码/转置卷积/BN 三组参数梯度中心差分、Adam 40 步 MSE 降到 1/3 以下、7 类非法输入 |
| `TestViT` | patch 化与 `CLS`/位置编码拼接、`param_count`、`dx` 与 `CLS`/`pos`/head/`LayerNorm` 四组梯度中心差分、梯度累加与 `zero_grad` 语义、2 类小任务训练收敛并全对、9 类非法输入 |
| `TestFPN` | 三级输出尺寸与参数量、`smooth=False` 分支、逐级 `dx` 与横向/平滑卷积 `grad_w` 中心差分、`sgd_step`/`adam_step`、6 类非法输入 |
| `TestSegNet` | 输出形状与参数量、`dx` 与 `conv1` 梯度中心差分、`pool0.mask` 逐窗口等于 argmax 且 `_unpool2` 只在选中位置取值、40 步 Adam loss 降到 1/3 以下、8 类非法输入 |
| `TestMLPMixer` | 形状与参数量 891、`dx` 与 block 内 `t1/t2/c1/c2` 四组权重梯度中心差分、head 与 block 内 LayerNorm 梯度中心差分、两次 `backward` 线性累加且 `zero_grad` 清空、120 步 Adam 收敛且 4 张图全对、9 类非法输入 |
| `TestSwinTransformer` | 窗口划分与移位 `perm` 语义、形状与参数量 1243、`dx` 与 block 内 `q/k/v/o/f1/f2` 六组权重梯度中心差分、head 与 block 内 LayerNorm 梯度、累加语义与 `zero_grad`、150 步 Adam 收敛且 4 张图全对、10 类非法输入 |
| `TestRoPEScaling` | `rope_inv_freqs` 的 none/linear/NTK 闭式解、YaRN 两端点退化与逐项夹逼、MHA 默认频率不变、缩放后 `W_q/W_k/W_v/W_o` 梯度仍与中心差分一致 ≤1e-6、7 类非法参数 |
| `TestLearnedPositionalEncoding` | 加常数语义、`param_count`、`dx == dout`、梯度累加、`sgd_step` 精确 `lr*grad`、`adam_step` 真的更新、超长输入自动扩表、8 类非法输入 |
| `TestLinearAttention` | 形状与参数量 168、单头非 causal 闭式解、causal 的「改尾部不影响前部」、`W_q/W_k/W_v/W_o` 与输入梯度中心差分（两种模式 ≤1e-5）、梯度累加与 `zero_grad`、`sgd_step`/`adam_step` 真更新、120 步 Adam 收敛、9 类非法输入 |
| `TestLinformerAttention` | 形状与参数量 312（低秩分数矩阵是 `T×k`）、`W_q/W_k/W_v/W_o`/`E`/`F`/偏置/输入梯度中心差分 ≤1e-5、`k==T` 且 `E=F=I` 时与 `MultiHeadAttention` 逐位等价（<1e-12）、梯度累加与 `zero_grad`、Adam 120 步收敛、非法构造与长度校验 |
| `TestPerformerAttention` | 形状与参数量 288、`phi` 长度 = m、随机特征不训练，`W_*`/输入梯度中心差分 ≤1e-5（causal 与非 causal）、causal 只影响当前及之后的位置、特征数越多近似越准（m=4 → m=64）、梯度累加与两个优化器、非法输入 |
| `TestMLAttention` | 形状、`latent` 是 `T×d_c`、参数量 210、KV cache `T·d_c` 与标准 MHA 的 8 倍压缩比，五个权重/五个偏置/输入梯度中心差分 ≤1e-5（causal 与非 causal），causal 上三角权重恰为 0 且每行和为 1、梯度累加与两个优化器、Adam 120 步收敛、非法输入 |
| `TestBigBirdAttention` | 稀疏模式（窗口/全局/随机）、causal 上三角恰 0、每行和为 1、全连通时与稠密 softmax 等价、参数与输入梯度中心差分 ≤1e-5（含 causal）、累加/清零/两个优化器、8 类非法输入 |
| `TestLongformerAttention` | 全局 token 看见全部位置、普通位置只看窗口 ∪ 全局、causal 下全局行仍看全序列、参数与输入梯度 ≤1e-5、非法输入 |
| `TestReformerAttention` | 只与同桶位置有非零权重、单桶退化为稠密、哈希可复现、causal 与非 causal 的梯度 ≤1e-5、非法输入 |
| `TestFlashAttention` | 块大小与稠密一致、块不变性、`O(T)` 缓存无 T×T、causal 前缀逐位不变、四组梯度中心差分、累加与非法输入 |
| `TestT5RelativeBias` | 分桶公式（正负半区、单调、夹断）、查表一致、桶内梯度求和、优化器与非法输入 |
| `TestXPosFactors` | 形状与位置 0 归 1、随位置衰减、`alpha=0` 关闭、`offset` 语义、非法输入 |
| `TestGraphLayers` | GCN 退化语义与 `dW`/`dx` 中心差分、GraphSAGE 两种拼接、GIN 的 `eps` 梯度与「无边退化为 MLP」、三种读出与反向、训练下降、8 类非法输入 |
| `TestGATLayer` | 注意力行和为 1 / 非邻居权重为 0 / `param_count`、`concat` 与 `mean` 两档的 `dx`/`dW`/`da_src`/`da_dst` 中心差分、累加与 `zero_grad`、两档优化器、非法输入 |
| `TestAPPNPLayer` | `alpha=0` 退化为 `Â^K H0` / `alpha=1` 输出恒为 `H0`、K=1/2/3 的 `dW` 与 `dx` 中心差分、训练下降、非法输入 |
| `TestJKNet` | `concat` 宽度、`last` 与手工双层堆叠的**黄金对照**（逐层 `gW`、`dx`）、`max` 汇总的回传、非法输入 |
| `TestMPNNLayer` | 输出形状与 `param_count`、`dx`/`msg1.gW` 中心差分、`sum` 与 `mean` 不同、无边退化、边特征参与、120 步训练下降、6 类非法输入 |
| `TestGCNIILayer` | `beta = log(lam/(l+1)+1)` 与 `(1-beta)I + beta W`、`alpha=0` 忽略 `H^0`、`dx`/`gW`/`grad_h0` 三组中心差分、两档优化器、非法输入 |
| `TestGraphPooling` | `TopKPool` 保留数与子图、gate 缩放、**含 score 路径的全区节点梯度**、`SAGPool` 的 GCN 打分路径、5 类非法输入 |
| `TestPromptTuning` | 拼接语义与 `param_count`、`g_prompt` 中心差分、序列梯度直通、累加与清零、两档优化器、非法输入 |
| `TestPTuning` | MLP 生成的提示与前向一致、`anchor`/`W1`/`W2` 三组中心差分、优化器、非法输入 |
| `TestPackingAndPrefixCache` | 打包的 `segment_ids`/`position_ids`/段边界、掩码跨段为 0 与同段因果、缓存的 `set`/`get`/深拷贝/`truncate`/`merge`/字典往返、7 类非法输入 |
| `TestContinuousBatcher` | 容量与补位（短请求退休后立即补位）、`max_total_tokens` 预算只算提示词占用、`used_tokens` 记账不出现负值、`result` 与 `stats`、4 组非法输入 |
| `TestTailFreeFilter` | 保留集是概率降序前缀、`tft` 单调、入参不被修改、n≤2 原样、非法 `tft` 与 logits |
| `TestPagedKVCache` | 块分配（2 token/块）、`fragmentation`、逻辑顺序、释放与复用、块耗尽报错、多请求交错、6 组校验 |
| `TestPatchEmbedding` | patch 切分与行优先顺序、偏置整体平移、`param_count`、`dx` 中心差分、6 类非法输入 |
| `TestAttentionPooling` | `q=0` 退化为平均、softmax 权重闭式解、`dx`/`q` 中心差分、5 类非法输入 |
| `TestGeMPooling` | `p=1` 等于平均、`p=2` 等于均方根、`p=2000` 仍接近最大值且不溢出、三档 `p` 的 `dx` 中心差分、6 类非法输入 |
| `TestSwitchNorm` | 三统计量加权闭式解、`dx`/`gamma`/`beta`/`logits` 四组中心差分、批量形状、整体平移 logits 不变性、优化器步与非法输入 |
| `TestFRN` | 闭式解与 TLU 截断、`dx`/`beta`/`tau` 中心差分、eps→0 时的尺度不变性、优化器步与 6 类非法输入 |
| `TestFractionalPooling` | 输出尺寸随比例、每个输出等于被选位置原值、同 seed 逐位相同、反向只回传被选位置且总量守恒 |
| `TestStochasticPooling` | 概率和为 1、训练输出等于采样值且反向乘 `p`、`eval()` 取期望、非正窗口退化为均匀、同 seed 可复现 |
| `TestConditionalNorm` | `gamma/beta` 闭式解、`dx`/`b_g,b_b`/`W_g,W_b`/`grad_cond` 四组中心差分、优化器步与 8 类非法输入 |
| `TestGQA` | KV 头形状、`num_kv_heads` 缺省与显式传等值逐位一致、**共享 KV 下 `W_k`/`W_v` 梯度 vs 数值梯度**、`forward_step` == 整段前向、参数确实变少、非法取值 |
| `TestGroupNorm` | 组内均值/方差、组间独立、`dx`/`dgamma`/`dbeta` vs 数值梯度、`InstanceNorm == GroupNorm(G=C)`、非法参数 |
| `TestDropPath` | `p=0`/`eval()` 恒等、丢弃率≈p、保留时放大 `1/(1-p)`、backward 与 gate 一致、非法 p |
| `TestPerplexity` | `perplexity == exp(evaluate)`、溢出保护 |
| `TestLMPersistence` | 存档往返参数与 logits 逐位相同、带 Adam 状态的续训与不中断训练逐位相同、不恢复优化器时确实不同、坏存档（format/形状/NaN/缺文件）报错、`Seq2SeqLM` 往返 |
| `TestLMPersistenceMatrix` | 存档把 `rope`/`window`/`num_kv_heads` 一起带走，重载后 forward/evaluate/generate 逐位相同、`step_count` 恢复 |
| `TestPadMaskAlignment` | 左填充 `pad_mask` 报错（原来会静默产出 NaN）、右填充仍逐位正确、整行被屏蔽时 softmax 返回 0 而不是 NaN |
| `TestTopKValidation` | `top_k=0.5/-1/True` 在 `_sample`/`generate`/束搜索里都报错，合法 `top_k` 采样不变 |
| `TestLMTargetValidation` | 负/越界/布尔目标 id 报错（原来负索引被当成最后一类），合法 id 的 loss 不变 |
| `TestNonFiniteLogits` | 权重含 NaN 时 `evaluate`/`_loss_and_backward` 报错（原来 loss 被 clamp 成 27.63、perplexity 返回 1e12） |
| `TestRestoreOptimizerStrict` | 无优化器状态的存档 + `restore_optimizer=True` 报错；带状态时 `step_count` 与 Adam 矩都能恢复 |
| `TestBPEForwardRef` | BPE 前向引用/非列表 `merges` 报 `ValueError`（原来是 `KeyError`），合法规则仍可用 |
| `TestSeq2SeqGenerateValidation` | 源序列超 `max_len`、源 id 越界在 `generate` 里报错（原来只有 `forward` 检查） |
| `TestPaddedBatch` | 右侧补齐 + `pad_mask` 后 logits/loss/梯度逐位不变、`batch_size=2` 的一步等于手工按 token 加权的一步、步数 = ceil(n/batch)、非法 mask 与非法 batch_size 报错、eval 下补齐同样逐位一致 |
| `TestBytePairTokenizer` | 任意 UTF-8 往返（含 emoji/换行）、训练确定性、合并不跨词块、`encode_corpus` 分块、JSON 存档往返、非法输入与截断 UTF-8 报错、与 `CausalLM` 串起来训练与生成 |
| `TestWordPieceTokenizer` | 词表布局（特殊标记在前、## 子词）、编码解码往返与空白归一化、词级 token 与 `encode_pair`、确定性与存档往返、十类非法输入、与 BPE 的行为差异 |
| `TestBeamSearch` | `num_beams=1` 与贪心逐位相同、束搜索确定且累计对数概率不劣于贪心、非法 `num_beams` 报错、`Seq2SeqLM` 束搜索与 `eos` 提前停 |
| `TestSeq2SeqLM` | 46 个参数块（两个词表 + enc 16 + dec 26 + 输出头 2）、解码器因果性（改未来目标 token 不动前面 logits）、改源序列影响全部位置、`src_embedding` / 交叉注意力 `W_k` / `out.W` vs 数值梯度、复制作业学会（loss < 0.2×首值且贪心解码正确）、生成同 seed 可复现、非法输入 |
| `TestCrossAttention` | 输出形状、无因果掩码、改 memory 影响输出、`dx`/`dmemory`/`W_k` vs 数值梯度、dropout train/eval、非法参数 |
| `TestTransformerDecoder` | 自注意力因果但交叉注意力看全 memory、`dx`/`dmemory`/交叉 `W_k` vs 数值梯度、单层与直接调层逐位一致、多层 `dmemory` 形状、非法 `num_layers` |
| `TestImbalanceLosses` | Class-Balanced 权重公式与归一化、CB 索引/软标签梯度、LDAM 闭式对齐与梯度、OHEM 选样、Model 接线与两个激活守卫、8 类非法输入 |
| `TestDistillAndHint` | 蒸馏损失：T² 缩放与 KL 对齐、梯度差分、T 校验、hint 均方误差与 weight、Model 接线与激活守卫 |
| `TestPreferenceLoss` | DPO/IPO/SimPO/CPO/ORPO/KTO：闭式解、双向中心差分、数值稳定（β=50 不溢出）、9 类非法输入 |
| `TestMultiTaskWeighting` | 不确定性加权闭式解与双向差分、GradNorm 不动点与 `sign(G-target)·G0/r` 逐项校验、PCGrad 冲突投影与嵌套结构、11 类非法输入 |
| `TestModelLossMap` | `Model(loss=...)` 认得 focal/bce/bce_with_logits/huber、`loss_kwargs` 透传（与直接调用逐位相等）、`kl` 被拒、激活守卫、未知损失的错误信息列出新选项 |
| `TestGradAccum` | `grad_accum=1` 与改动前逐位一致、`grad_accum=2` 与 batch 翻倍等价、剩余不足 accum 的 batch 也会更新、非法取值报错 |
| `TestFlops` | `estimate_flops` 的 MACs / 参数量 / 逐层明细与手算一致、不认识的模块报 TypeError |
| `TestFlopsExtended` | 分组/深度卷积按 `in_channels/groups` 计数（不按 `in_channels` 高估）、`Conv1d`/`ConvTranspose2d`/`AdaptiveAvgPool2d`/`GroupNorm`/`DropPath` 的计数与形状校验 |
| `TestLossPersistence` | 默认存档不多写字段（旧文件逐字节不变）、`focal`/`CE(label_smoothing)` 往返后损失配置不变且续训曲线逐位相同、显式传参覆盖文件、坏 `loss` 字段忽略、自定义 callable 不写字段、与优化器状态共存 |
| `TestAugmentation` | `sigma=0` 逐位不变且不改入参、图像翻转保持元素多重集、平移补 0、Mixup 中点与软标签、非法输入报错 |
| `TestSamplingControls` | 默认参数逐位不变、`repetition_penalty` 对正/负 logit 的处理、presence 只扣一次与 frequency 按次数扣、禁 n-gram 掩码、min-p 截尾、采样路径 cache == 重算、禁 2-gram 改变贪心输出且无重复二元组、七类非法参数、Seq2SeqLM 侧同样覆盖 |
| `TestMetrics` | 混淆矩阵/准确率、macro/micro/weighted 三种平均、MCC 与 AUC（含全同分=0.5）、Top-k、MAE/RMSE/MAPE/R²（含常数目标 1.0/0.0）、BLEU 无重叠判 0、ROUGE-L、`evaluate_all` 分类与回归键齐、七类非法输入 |
| `TestReviewFixes` | `ConvTranspose2d` 偏置梯度不被 `cin/groups` 放大、`mixup` 的 `lam` 恒在 `[0,1]` 且软标签合法、分组卷积存档往返（含缺 `groups` 的旧存档）、`CrossAttention.delta_index=0`、`estimate_flops` 拒绝一维/二维 `Flatten`、`CausalLM.save` 拒绝 NaN、`summary` 显示真实激活名、`grad_check` 消息含 Seq2SeqLM、`Sequential` 收列表且拒非模块 |
| `TestConvGroups` | `groups=2` 前向等价两个单组卷积、depthwise 通道、`grad_w` vs 数值、非法 groups |
| `TestConv1d` | 形状与 stride、手算卷积、`groups` 等价、`grad_w`/`dx` vs 数值、非法参数 |
| `TestQuickTrainSeed` | `quick_train(seed=)` 同 seed 权重与曲线完全相同、不同 seed 不同、不推进全局 RNG |
| `TestFitSeed` | `fit(seed=)` 两次训练曲线/权重逐位相同、不同 seed 顺序不同、传 seed 不推进全局 RNG、`shuffle=False` 同样可复现、顶层 `train(seed=)` 可复现 |
| `TestTrainingInfraSeed` | `set_seed` 同种子同序列、非整数报错、`fit(seed=)` 不推进全局 RNG |
| `TestInferShapesAndBatch` | `infer_shapes` 的形状/参数/激活名与手算一致、`predict_batch` 与 `predict` 逐位一致、非法输入报错 |
| `TestDiagnostics` | `activation_stats` 的 `zero_ratio` 与 `dead_units` 检出的死单元下标一致、linear 层带提示 |
| `TestBuiltinCallbacks` | `EarlyStopping` 提前停并记录最优/停止轮、`ModelCheckpoint` 按轮存且可 `resume`、`save_best_only` 只存一次、`LoggingCallback` 与 `HistoryCallback` 输出 |
| `TestCallbacks` | 四个钩子的触发顺序与 payload 键、坏回调 TypeError、钩子异常会中断训练 |
| `TestResumeAndLayerNorms` | `resume_from`（`epochs=0`）恢复权重与 `step_count`、坏存档报错、`layer_grad_norm` 只记录不改训练 |
| `TestMixupInFit` | `_mixup_batch` 的软标签和为 1、`mixup_alpha` 改变训练轨迹且数值有限、回归/非法 alpha 报错 |
| `TestHooks` | 前向/反向 hook 都能看到该层张量、注销与 `clear_hooks` 生效、非法 index/不可调用报错、挂 hook 不改变训练数值（两个同权重模型逐位相同） |
| `TestConvTranspose2d` | 输出尺寸公式与 `output_padding`、单点散布核、伴随恒等 `<Cx,y>==<x,C^T y>`、梯度 vs 数值、非法参数 |

CI（GitHub Actions）执行 flake8 + pytest。

下一步的扩展计划（网络层 / 损失 / 优化器 / 数据管道 / 推理部署等 17 类，按 20 个波次排期）见 [ROADMAP.md](ROADMAP.md)。

## 文件结构

```
.
├── AI.py            # 神经网络工具箱（Python 实现，所有代码）
├── test_ai.py       # 1040 个单元测试
├── ROADMAP.md       # 后续扩展路线图（20 波、约 240 个实现单元）
└── README.md
```
