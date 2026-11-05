# Python-standard-library-only-AI

纯 Python 标准库实现的神经网络工具箱，**零第三方依赖**。

- **运行环境**: Python **3.12+**（3.12 起 `sum()` 改用 Neumaier 补偿求和，C 版引擎
  逐位复刻了这一行为；3.10/3.11 上跨语言对拍的末位会不一致）
- **CI**: GitHub Actions 在 Python 3.12 上跑 `flake8` + `pytest`（429 项）

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
| `lr_scheduler` | StepLR / ExponentialLR / CosineAnnealingLR |
| `grad_clip` | 梯度范数裁剪阈值 |
| `checkpoint_path` / `checkpoint_every` | 每 `checkpoint_every` 轮把模型（含优化器状态）存到该路径，用于断点续训 |
| `grad_accum` | 梯度累积步数（1 = 每个 batch 更新一次）；>1 时累加若干 batch 的平均梯度再更新，等效放大 batch 而不占额外内存 |
| `seed` | 打乱数据的随机种子（None = 用全局 `random`，与旧版行为一致；传整数则改用局部 `Random`：同样的 `fit` 再跑一次结果完全相同，也不会推进调用方的全局随机状态） |

训练结束后 `model.history` 里有三个键：`train_loss`、`val_loss`、`grad_norm`
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
| `Layer` | 全连接层 + 激活函数（He 初始化）；激活可选 `leaky_relu` / `relu` / `gelu` / `sigmoid` / `linear` / `softmax` |
| `Linear` | 无激活的全连接层（`Layer(..., activation='linear')` 的语义化子类） |
| `MLP` | 多层感知机容器 |
| `Conv2d` | 二维卷积层（`groups=G` 支持分组/深度卷积） |
| `Conv1d` | 一维卷积层（`(C, L)` 输入，滑动窗口直接累加） |
| `ConvTranspose2d` | 二维转置卷积（反卷积；`output_padding` 与 PyTorch 同规则） |
| `MaxPool2d` | 二维最大池化 |
| `AvgPool2d` | 二维平均池化（接口同 `MaxPool2d`，反向把梯度均分回窗口） |
| `GlobalAvgPool2d` | 全局平均池化：`(C,H,W) → (C,)`，CNN 分类头常用 |
| `AdaptiveAvgPool2d` | 自适应平均池化：`(C,H,W) → (C,out_h,out_w)`，任意输入尺寸都能固定输出（PyTorch 分箱规则，`(1,1)` 等价于 `GlobalAvgPool2d`） |
| `Flatten` | 展平（CNN → MLP） |
| `Dropout` | Dropout 正则化（**推理前必须调 `eval()`**，见下方「训练 / 推理模式」） |
| `DropPath` | 随机深度：整条残差分支按概率丢弃（inverted，`eval()` 关闭） |
| `BatchNorm1d` | 批归一化：`forward` 同时接受单样本与 batch；训练+2D 走 batch 统计量，推理用 running 统计量 |
| `GroupNorm` / `InstanceNorm` | 分组归一化（`(C,H,W)`，小 batch 稳定）；`InstanceNorm` 是组数等于通道数的特例 |
| `LayerNorm` | 层归一化 |
| `RMSNorm` | RMS 归一化（Llama 系）：不减均值、无 beta，`y = gamma·x/sqrt(mean(x²)+eps)` |
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
| `FeedForward(d_model, d_ff, activation='relu')` | 前馈网络（`activation` 可选 relu/gelu/sigmoid/linear） |
| `TransformerEncoderLayer` | 单层 Encoder（`causal=True` 时自注意力带因果掩码，`rope=True` / `window=N` 透传给注意力） |
| `TransformerEncoder` | 多层 Encoder 堆叠（`causal` / `rope` / `window` 会传给每一层） |
| `CrossAttention` | 交叉注意力：Q 来自解码器、K/V 来自编码器输出 `memory`（长度可不同，无因果掩码），`backward` 返回 `(dx, dmemory)` |
| `TransformerDecoderLayer` / `TransformerDecoder` | 解码器层（因果自注意力 → 交叉注意力 → FFN，三段残差 + LayerNorm）与多层堆叠；`backward` 把梯度同时传回解码器输入和编码器输出 |
| `PositionalEncoding` | sin/cos 绝对位置编码 |
| `CausalLM` | 字符/词级因果语言模型（GPT 式）：Embedding → 位置编码 → 因果 Transformer → 词表头，自带 Adam、`fit` / `evaluate` / `generate`；`rope=True` 时改用旋转位置编码（默认关掉绝对位置编码），`window=N` 用滑动窗口，`generate(use_cache=True)` 走 KV cache，`top_p` 做 nucleus 采样，`num_beams>1` 走束搜索；`save` / `load` 支持 JSON 存档（可选带 Adam 状态做逐位续训） |
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
| `Linear(fan_in, fan_out)` | 无激活全连接层 |
| `RMSNorm(num_features, eps=1e-6)` | RMS 归一化 |
| `AdamW(model, lr, ..., weight_decay=0.01)` | 解耦权重衰减；`weight_decay=0` 时与 `Adam` **逐位相同** |
| `WarmupLR(optimizer, warmup_epochs=5, start_factor=0.0, base_scheduler=None)` | 线性 warmup，到期后把控制权交给 `base_scheduler` |
| `MultiHeadAttention(rope=True)` | RoPE（GPT-NeoX 版式：第 i 维配 i+d_k/2）；反向用逆旋转 |
| `MultiHeadAttention(window=N)` | 滑动窗口（Mistral 式局部因果）：只看最近 N 个位置，代价 O(T·window) |
| `generate(..., top_p=0.9)` | nucleus 采样：按概率累加保留头部集合（与 `top_k` 可叠加；默认 1.0 不过滤） |
| `MultiHeadAttention(num_kv_heads=K)` | GQA/MQA：K 个 Q 头共享一组 K/V，`W_k`/`W_v` 与 KV cache 同比例缩小 |

**RoPE + KV cache**：`CausalLM(..., rope=True)` 会默认关掉绝对位置编码（Llama 的做法），
想两者同时用可传 `use_abs_pos=True`。

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
| `bce_with_logits_loss(logits, target)` | 数值稳定的 BCE：`max(z,0) - z·y + log1p(exp(-|z|))`，直接吃 logits |
| `huber_loss(pred, target, delta=1.0)` | Huber / Smooth L1，对离群点比 MSE 稳 |
| `kl_divergence(pred, target)` | `KL(target ‖ pred) = Σ t·log(t/p)`，蒸馏/软目标常用 |

| 损失名（`Model(loss=...)`） | 需要的输出层 | 目标形式 |
|------|------|------|
| `'mse'` | 任意（通常 `linear`） | 数值向量 / 标量 |
| `'cross_entropy'` / `'ce'` | `softmax` | 类别索引 / one-hot / 软标签 |
| `'focal'` | `softmax` | 类别索引 / 严格 one-hot（软标签会被拒绝） |
| `'bce'` | `sigmoid` | 0/1 数值 |
| `'bce_with_logits'` | `linear` | 0/1 数值（喂 logits） |
| `'huber'` | 任意 | 数值向量 / 标量 |

`kl_divergence` **不在**上面的映射里：它要求 `pred` 是概率分布，而 `softmax` 层的
反向是 softmax+CE 的联合梯度，任何非 CE 损失叠在 softmax 上都拿到不自洽的梯度
（实测 KL 训练 loss 会从 0.35 涨到 5.78）。需要 KL 时请手动调用
`AI.kl_divergence` 并自己反传。

配错「输出层激活 × 损失」会在 `Model(...)` 构造时直接抛 `ValueError`，而不是静默训歪。

> 所有损失都遵循和 `mse_loss` 相同的返回约定：`(loss, grad)`，
> 也都可以当成 `Model(loss_fn=...)` 的自定义可调用对象用。
> 额外超参（`focal` 的 `gamma`/`alpha`、`ce` 的 `label_smoothing`、`huber` 的 `delta`）
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
```

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
| `StandardScaler` / `MinMaxScaler` | 逐特征标准化 / 归一化（`fit`/`transform`/`inverse_transform`） |
| `ascii_plot(history)` | 把训练曲线画成终端 ASCII 图 |
| `grad_check(module, x, y=None)` | 中心差分校验解析梯度（MLP / Model / CausalLM，以及 `Seq2SeqLM`（传 `(src, tgt)` 二元组））；解析与数值都 ≈0 的参数记进 `zeros` 不参与相对误差 |
| `mlp.register_forward_hook(i, fn)` / `register_backward_hook(i, fn)` | 第 i 层前向/反向后的回调（`fn(layer, output)` / `fn(layer, delta, grad_w, grad_b)`）；不注册时只多一次真假判断，`Model` 上同名方法转发给内部的 `MLP` |
| `estimate_flops(model, input_shape)` | 前向乘加次数（MACs）/ 参数量 / 逐层明细；支持 MLP 与 `Sequential`（`Layer`/`Conv1d`/`Conv2d`（含 `groups`）/`ConvTranspose2d`/各种池化/`BatchNorm1d`/`LayerNorm`/`GroupNorm`/`InstanceNorm`/`Dropout`/`DropPath`/`Flatten`），不认识的模块报 TypeError 而不是静默估算；`Flatten` 只接受 `(C, H, W)`，传一维/二维会明确报错（`Flatten.forward` 本来也跑不了） |
| `augment_noise(x, sigma=0.01)` | 加性高斯噪声（单样本或一批样本，不改入参） |
| `augment_image(x, flip=True, max_shift=1)` | 图像 `(C,H,W)` 的随机翻转 / 平移（越界补 0） |
| `mixup(x1, y1, x2, y2, alpha=0.2)` | Mixup：返回混合样本 + 软标签向量（可直接喂 CE 的软标签分支）；`lam ~ Beta(alpha, alpha)`，恒在 `[0,1]` 内 |

> 旧文件不含逐层 `activation` 时，输出层回退优先级为：显式参数 > 文件里的 `task`
> （`classification` 用 softmax）> `linear`。既不传参数、文件里也没有 `task` 的旧分类器
> 会被当成回归模型，请显式传 `default_output_activation='softmax'`。

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
「字母+空格」这种跨词 token；语料里重复片段耗尽时会提前停下，ocab_size 只是上限。

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

523 个测试覆盖所有组件，其中这几类是**数值校验**而不是只查形状：

| 测试类 | 校验内容 |
|--------|----------|
| `TestLayerNumericalGradients` | `Layer` / `MLP` 的 `grad_w / grad_b`、反向传播链（`dL/dx`）vs 数值梯度；含 softmax+CE 联合梯度约定 |
| `TestTransformerEncoderGradients` | `TransformerEncoderLayer` / `TransformerEncoder` 的 `d_x` 与 LayerNorm 参数梯度 vs 数值梯度（**seq_len>1 必须测**，见下方实现说明） |
| `TestAttentionGradients` | `MultiHeadAttention` 的 `W_q/W_k/W_v/W_o` 解析梯度 vs 数值梯度 |
| `TestConv2dGradients` | `Conv2d` 的 `grad_w / grad_b / dx` vs 数值梯度（含 padding 与 stride=2） |
| `TestBatchNormTraining` | `BatchNorm1d` 训练路径的 `dx / gamma / beta` vs 数值梯度，并与朴素实现逐位一致 |
| `TestStepCache` / `TestCellGradSemantics` | `restore_cache` 与「重新前向」逐位一致；一次 `zero_grad` + 逐步 backward == 各步梯度之和 |
| `TestValDataFormats` / `TestCrossEntropyTargets` | `val_data` 三种写法、CE 的 target 规范化 |
| `TestCausalMask` | 因果掩码的**因果性**（改未来 token 不改过去输出）、掩码权重精确为 0、`W_v` 梯度 vs 数值梯度 |
| `TestCausalLM` | 小语料过拟合（loss → <0.05）、贪心续写接上周期、前缀 logits 与未来无关、生成可复现、非法输入 |
| `TestAttentionDropout` | 注意力 dropout 的 mask 取值/缩放、`eval()` 关闭、train 随机性、**固定 mask 下 `W_v` 解析梯度 vs 数值梯度** |
| `TestModernComponents` | GELU 公式与导数、sigmoid 不溢出、`Linear`、`RMSNorm` 的 `dx / dgamma` vs 数值梯度 |
| `TestAdamWAndWarmup` | AdamW 零梯度只做衰减、`weight_decay=0` 与 Adam 逐位相同、`WarmupLR` 的爬坡/接力曲线 |
| `TestExtraLosses` | `gamma=0` 的 focal 退化为 CE、focal/CE-ls 梯度 vs 数值梯度、BCE 的 logits 稳定分支、Huber 两段、KL 公式 |
| `TestRoPE` | 增量前向 == 整段前向（逐位）、RoPE 改变输出、**Q/K 逆旋转后的梯度 vs 数值梯度** |
| `TestKVCache` | KV cache 生成 == 整段重算（含窗口滑动与 RoPE 两种情形）、训练后仍一致、生成不改参数 |
| `TestKVCacheMatrix` | KV cache == 整段重算在 7 种组合下都成立（GQA/MQA × RoPE × 滑动窗口 × max_len 8/4），束搜索在这些组合下也确定性 |
| `TestFeedForwardActivation` | `FeedForward` 换 GELU/sigmoid 后的 `d_x` vs 数值梯度，`relu` 路径与旧实现逐位一致 |
| `TestGradCheck` | `grad_check` 对 MLP / Model / CausalLM 都返回 `ok`，且人为改坏梯度时能识别 |
| `TestSeq2SeqPaddedBatch` | `Seq2SeqLM` 的 `pad_mask`：补齐后 loss/梯度逐位不变、`batch_size` 一步等于手工按 token 加权、mask 校验 |
| `TestGradCheckSeq2Seq` | `Seq2SeqLM` 的 `grad_check` 通过、覆盖解码器与输出层、零梯度（`b_k`）不误报、检查后模式复原 |
| `TestOptimizerPersistence` | 不传 `optimizer` 时存档与旧格式逐字段相同、Adam 状态往返、**续训与一次训完权重逐位相同**、类型/形状不匹配报错 |
| `TestWindowMask` | 窗口外权重恰为 0、窗口注意力保持因果性、`forward_step` == 整段前向最后一行（含 RoPE）、**带窗口时 `W_v` 梯度 vs 数值梯度** |
| `TestTopPSampling` | `top_p=1.0` 与旧行为逐位相同、nucleus 只留头部集合、cache 与重算路径一致、非法 `top_p` 报错 |
| `TestAvgPool2d` | 平均池化前向值、反向均分、`dx` vs 数值梯度、`GlobalAvgPool2d` 形状、非法 kernel/stride |
| `TestAdaptiveAvgPool2d` | 非整除分箱手算值、`(1,1)` 与 `GlobalAvgPool2d` 逐位相同、整除时与 `AvgPool2d` 一致、`dx` vs 数值梯度、非法 `out_size` |
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
| `TestBeamSearch` | `num_beams=1` 与贪心逐位相同、束搜索确定且累计对数概率不劣于贪心、非法 `num_beams` 报错、`Seq2SeqLM` 束搜索与 `eos` 提前停 |
| `TestSeq2SeqLM` | 46 个参数块（两个词表 + enc 16 + dec 26 + 输出头 2）、解码器因果性（改未来目标 token 不动前面 logits）、改源序列影响全部位置、`src_embedding` / 交叉注意力 `W_k` / `out.W` vs 数值梯度、复制作业学会（loss < 0.2×首值且贪心解码正确）、生成同 seed 可复现、非法输入 |
| `TestCrossAttention` | 输出形状、无因果掩码、改 memory 影响输出、`dx`/`dmemory`/`W_k` vs 数值梯度、dropout train/eval、非法参数 |
| `TestTransformerDecoder` | 自注意力因果但交叉注意力看全 memory、`dx`/`dmemory`/交叉 `W_k` vs 数值梯度、单层与直接调层逐位一致、多层 `dmemory` 形状、非法 `num_layers` |
| `TestModelLossMap` | `Model(loss=...)` 认得 focal/bce/bce_with_logits/huber、`loss_kwargs` 透传（与直接调用逐位相等）、`kl` 被拒、激活守卫、未知损失的错误信息列出新选项 |
| `TestGradAccum` | `grad_accum=1` 与改动前逐位一致、`grad_accum=2` 与 batch 翻倍等价、剩余不足 accum 的 batch 也会更新、非法取值报错 |
| `TestFlops` | `estimate_flops` 的 MACs / 参数量 / 逐层明细与手算一致、不认识的模块报 TypeError |
| `TestFlopsExtended` | 分组/深度卷积按 `in_channels/groups` 计数（不按 `in_channels` 高估）、`Conv1d`/`ConvTranspose2d`/`AdaptiveAvgPool2d`/`GroupNorm`/`DropPath` 的计数与形状校验 |
| `TestLossPersistence` | 默认存档不多写字段（旧文件逐字节不变）、`focal`/`CE(label_smoothing)` 往返后损失配置不变且续训曲线逐位相同、显式传参覆盖文件、坏 `loss` 字段忽略、自定义 callable 不写字段、与优化器状态共存 |
| `TestAugmentation` | `sigma=0` 逐位不变且不改入参、图像翻转保持元素多重集、平移补 0、Mixup 中点与软标签、非法输入报错 |
| `TestReviewFixes` | `ConvTranspose2d` 偏置梯度不被 `cin/groups` 放大、`mixup` 的 `lam` 恒在 `[0,1]` 且软标签合法、分组卷积存档往返（含缺 `groups` 的旧存档）、`CrossAttention.delta_index=0`、`estimate_flops` 拒绝一维/二维 `Flatten`、`CausalLM.save` 拒绝 NaN、`summary` 显示真实激活名、`grad_check` 消息含 Seq2SeqLM、`Sequential` 收列表且拒非模块 |
| `TestConvGroups` | `groups=2` 前向等价两个单组卷积、depthwise 通道、`grad_w` vs 数值、非法 groups |
| `TestConv1d` | 形状与 stride、手算卷积、`groups` 等价、`grad_w`/`dx` vs 数值、非法参数 |
| `TestQuickTrainSeed` | `quick_train(seed=)` 同 seed 权重与曲线完全相同、不同 seed 不同、不推进全局 RNG |
| `TestFitSeed` | `fit(seed=)` 两次训练曲线/权重逐位相同、不同 seed 顺序不同、传 seed 不推进全局 RNG、`shuffle=False` 同样可复现、顶层 `train(seed=)` 可复现 |
| `TestHooks` | 前向/反向 hook 都能看到该层张量、注销与 `clear_hooks` 生效、非法 index/不可调用报错、挂 hook 不改变训练数值（两个同权重模型逐位相同） |
| `TestConvTranspose2d` | 输出尺寸公式与 `output_padding`、单点散布核、伴随恒等 `<Cx,y>==<x,C^T y>`、梯度 vs 数值、非法参数 |

CI（GitHub Actions）执行 flake8 + pytest。

## 文件结构

```
.
├── AI.py            # 神经网络工具箱（Python 实现，所有代码）
├── test_ai.py       # 523 个单元测试
└── README.md
```
