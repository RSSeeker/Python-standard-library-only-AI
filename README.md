# Python-standard-library-only-AI

纯 Python 标准库实现的神经网络工具箱，**零第三方依赖**。

## 特性

- **零第三方依赖**: 仅依赖 `math`、`random`、`json` 等标准库
- **全栈组件**: 从全连接层到 CNN、RNN/LSTM/GRU、Transformer 一应俱全
- **双模式 API**: 底层精细控制 + 高层 sklearn 风格封装
- **完整的训练管线**: Early Stopping、梯度裁剪、学习率调度、模型保存/加载
- **反向传播有数值梯度测试守着**: `Layer` / `MultiHeadAttention` / `Conv2d` / `BatchNorm1d` 的解析梯度都与数值梯度逐项比对

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
| `quick_train(x, y, task, ...)` | 一行代码自动创建并训练 |

### fit() 参数

| 参数 | 说明 |
|------|------|
| `epochs` | 训练轮数 |
| `batch_size` | 批次大小（-1 = 全批量） |
| `val_split` / `val_data` | 验证集比例或自定义验证集（见下方说明） |
| `patience` | Early Stopping 耐心值（0 = 关闭） |
| `lr_scheduler` | StepLR / ExponentialLR / CosineAnnealingLR |
| `grad_clip` | 梯度范数裁剪阈值 |

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
| `Layer` | 全连接层 + 激活函数（He 初始化） |
| `MLP` | 多层感知机容器 |
| `Conv2d` | 二维卷积层 |
| `MaxPool2d` | 二维最大池化 |
| `Flatten` | 展平（CNN → MLP） |
| `Dropout` | Dropout 正则化（**推理前必须调 `eval()`**，见下方「训练 / 推理模式」） |
| `BatchNorm1d` | 批归一化：`forward` 同时接受单样本与 batch；训练+2D 走 batch 统计量，推理用 running 统计量 |
| `LayerNorm` | 层归一化 |
| `Embedding` | 嵌入层 |
| `ResidualBlock` | 残差块 |
| `Sequential` | 层序列容器（不含 train/eval，内部的 Dropout 需自行切换） |
| `MultiHeadAttention` | 多头注意力（`dropout` 参数**尚未实现**，注意力内不做 dropout） |

> **早停与权重回滚**：`fit(patience=...)` 在指标不再改善时停止训练，但**不会自动回滚**
> 到最优那一轮；需要最优权重时传 `restore_best_weights=True`。
>
> **优化器状态不落盘**：`save_model` / `Model.save` 只保存权重、偏置与结构（及 task），
> 不保存 Adam 的动量/二阶矩。所以「加载后继续训练」不等于「不中断地训练」——
> 前几步的更新幅度会偏大。需要严格续训时请自行保存初始权重并重放相同的训练计划。

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
| `TransformerEncoderLayer` / `TransformerEncoder` | `train()` / `eval()` 会转发到内部两个 Dropout |
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
| `MultiHeadAttention` | 多头注意力机制 |
| `FeedForward` | 前馈网络 |
| `TransformerEncoderLayer` | 单层 Encoder |
| `TransformerEncoder` | 多层 Encoder 堆叠 |
| `PositionalEncoding` | 位置编码 |

### 优化器

| 优化器 | 说明 |
|------|------|
| `Adam` | Adam（默认） |
| `SGD` | 随机梯度下降 |
| `SGDMomentum` | 带动量的 SGD |
| `RMSprop` | RMSprop |

### 损失函数

| 函数 | 说明 |
|------|------|
| `mse_loss(pred, target)` | 均方误差；`target` 可以是序列或单个标量（`y=0.5` 也行），长度不一致会报错 |
| `cross_entropy_loss(pred, target)` | 交叉熵；`target` 可以是类别索引、one-hot 或概率向量（软标签） |

> 搭配约束：`softmax` 输出层的反向使用的是 **softmax + cross_entropy 的联合梯度**，
> 因此只能与 `cross_entropy` 搭配。`Model(loss_fn='mse')` + softmax 输出层会**直接报错**，
> 而不是静默用错误梯度训练；需要 MSE 时请把输出层激活设为 `linear`。
>
> `Model` 的 CE 路径会自动规范化 target：整数索引、`[0.0, 1.0]` 这类浮点标签、
> 长度为 1 的列表会被转成类别索引；**长度等于类数的 one-hot / 概率向量会原样直通**
> 给 `cross_entropy_loss` 的软标签分支，因此 label smoothing / 蒸馏的软目标不会被
> argmax 硬化。软标签的联合梯度按 `Σt·p − t` 计算（全 0 目标得到全 0 梯度）。
> 该守卫对 `loss_fn='mse'`、`loss_fn=mse_loss` 与 `functools.partial(mse_loss)` 都生效。

### 工具函数

| 函数 | 说明 |
|------|------|
| `split_data(data, val_ratio, test_ratio)` | 数据集拆分 |
| `DataLoader` | 批量迭代器 |
| `train(model, loss_fn, optimizer, data, ...)` | 底层训练循环（Early Stopping + 验证集） |
| `evaluate(model, loss_fn, data)` | 计算平均损失 |
| `save_model(model, path, task=None)` / `load_model(path, default_output_activation=None)` | MLP 的 JSON 持久化；旧格式（无逐层 activation）可用 `default_output_activation='softmax'` 还原分类头 |
| `save_conv_model(conv, fc, path)` / `load_conv_model(path, default_output_activation=None)` | CNN + MLP 的 JSON 持久化，旧格式同上 |
| `Model.save(path)` / `Model.load(path, task='auto', default_output_activation=None)` | 高层封装；`task='auto'` 时优先采用文件中记录的 task |

> 旧文件不含逐层 `activation` 时，输出层回退优先级为：显式参数 > 文件里的 `task`
> （`classification` 用 softmax）> `linear`。既不传参数、文件里也没有 `task` 的旧分类器
> 会被当成回归模型，请显式传 `default_output_activation='softmax'`。
| `clip_grad_by_norm(grads, max_norm)` | 梯度范数裁剪 |
| `clip_grad_by_value(grads, clip_val)` | 梯度值裁剪 |
| `compare_with_pytorch()` | 与 PyTorch 逐层数值对齐（可选，装了 torch 才跑） |

### 学习率调度器

| 调度器 | 说明 |
|------|------|
| `StepLR(optimizer, step_size, gamma)` | 阶梯衰减 |
| `ExponentialLR(optimizer, gamma)` | 指数衰减 |
| `CosineAnnealingLR(optimizer, T_max)` | 余弦退火 |

### 文本处理

| 组件 | 说明 |
|------|------|
| `TextProcessor` | 字符级词表、one-hot 编码、滑动窗口 |
| `generate_text(network, tp, start, length)` | 自回归文本生成（`generate_text_with_mlp` 的别名，保留以兼容旧调用） |
| `generate_text_with_mlp(model, tp, start, length, window_size, temp)` | 自回归文本生成（滑动窗口上下文，只需 `forward`） |
| `_softmax_with_temp(logits, temperature)` | 带温度的 Softmax |
| `sample_index(probs)` | 随机采样 |

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

依次运行 12 段演示：回归（`Model.fit`）→ 分类（`make_classifier` + Softmax/CE）→ CNN（卷积→池化→展平→MLP）→ RNN & LSTM 序列预测 → 模型保存/加载 → GRU 序列预测 → Dropout + LayerNorm → 梯度裁剪 + StepLR → `quick_train` 一行训练 → Transformer 序列分类 → 字符级文本生成 → PyTorch 数值对齐。

## 测试

```bash
python test_ai.py     # 等价于 pytest test_ai.py
```

257 个测试覆盖所有组件，其中这几类是**数值校验**而不是只查形状：

| 测试类 | 校验内容 |
|--------|----------|
| `TestLayerNumericalGradients` | `Layer` / `MLP` 的 `grad_w / grad_b`、反向传播链（`dL/dx`）vs 数值梯度；含 softmax+CE 联合梯度约定 |
| `TestTransformerEncoderGradients` | `TransformerEncoderLayer` / `TransformerEncoder` 的 `d_x` 与 LayerNorm 参数梯度 vs 数值梯度（**seq_len>1 必须测**，见下方实现说明） |
| `TestAttentionGradients` | `MultiHeadAttention` 的 `W_q/W_k/W_v/W_o` 解析梯度 vs 数值梯度 |
| `TestConv2dGradients` | `Conv2d` 的 `grad_w / grad_b / dx` vs 数值梯度（含 padding 与 stride=2） |
| `TestBatchNormTraining` | `BatchNorm1d` 训练路径的 `dx / gamma / beta` vs 数值梯度，并与朴素实现逐位一致 |
| `TestStepCache` / `TestCellGradSemantics` | `restore_cache` 与「重新前向」逐位一致；一次 `zero_grad` + 逐步 backward == 各步梯度之和 |
| `TestValDataFormats` / `TestCrossEntropyTargets` | `val_data` 三种写法、CE 的 target 规范化 |

CI（GitHub Actions）执行 flake8 + pytest。

## 文件结构

```
.
├── AI.py            # 神经网络工具箱（Python 实现，所有代码）
├── test_ai.py       # 257 个单元测试
└── README.md
```
