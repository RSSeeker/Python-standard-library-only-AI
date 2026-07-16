# Python-standard-library-only-AI

纯 Python 标准库实现的神经网络工具箱，**零第三方依赖**。

## 特性

- **Zero Dependencies**: 仅依赖 `math`、`random`、`json` 等标准库
- **全栈组件**: 从全连接层到 CNN、RNN/LSTM/GRU、Transformer 一应俱全
- **双模式 API**: 底层精细控制 + 高层 sklearn 风格封装
- **完整的训练管线**: Early Stopping、梯度裁剪、学习率调度、模型保存/加载

## 快速开始

```python
from AI import *

# ── 分类：一行训练，零配置 ────────────────────
x = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9],
     [0.15, 0.25, 0.35], [0.45, 0.55, 0.65], [0.75, 0.85, 0.95]]
y = [0, 1, 2, 0, 1, 2]

model = quick_train(x, y, task='classification', epochs=300)
print(model.score(x, y))          # 准确率 → 1.0

# ── 回归：两行搞定 ────────────────────────────
x_reg = [[0.2, 0.4, 0.6, 0.8, 1.0], [0.4, 0.6, 0.8, 1.0, 1.2]]
y_reg = [[0.04, 0.08], [0.16, 0.24]]

model_reg = make_regressor(input_dim=5, output_dim=2, hidden_layers=(6, 6, 4))
model_reg.fit(x_reg, y_reg, epochs=500, patience=30)
print(model_reg.predict([[0.6, 0.8, 1.0, 1.2, 1.4]]))  # → [0.36, 0.48]
```

## 高层 API

### Model 类（sklearn 风格）

```python
# 手动构建
model = Model(MLP([5, 16, 8, 2]), loss_fn='mse', optimizer='adam', lr=0.01)
model.summary()          # 打印模型结构
history = model.fit(x, y, epochs=300, batch_size=8, patience=30,
                    lr_scheduler=StepLR(...), grad_clip=1.0)
pred = model.predict(x)
score = model.score(x, y)   # 分类=准确率 / 回归=R²

# 保存 & 加载
model.save("model.json")
model2 = Model.load("model.json")
```

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
| `val_split` / `val_data` | 验证集比例或自定义验证集 |
| `patience` | Early Stopping 耐心值（0 = 关闭） |
| `lr_scheduler` | StepLR / ExponentialLR / CosineAnnealingLR |
| `grad_clip` | 梯度范数裁剪阈值 |

## 底层组件

### 层 (Layers)

| 组件 | 说明 |
|------|------|
| `Layer` | 全连接层 + 激活函数（He 初始化） |
| `MLP` | 多层感知机容器 |
| `Conv2d` | 二维卷积层 |
| `MaxPool2d` | 二维最大池化 |
| `Flatten` | 展平（CNN → MLP） |
| `Dropout` | Dropout 正则化 |
| `BatchNorm1d` | 批归一化 |
| `LayerNorm` | 层归一化 |
| `Embedding` | 嵌入层 |
| `ResidualBlock` | 残差块 |
| `Sequential` | 层序列容器 |

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
| `mse_loss(pred, target)` | 均方误差 |
| `cross_entropy_loss(pred, target_idx)` | 交叉熵 |

### 工具函数

| 函数 | 说明 |
|------|------|
| `split_data(data, val_ratio, test_ratio)` | 数据集拆分 |
| `DataLoader` | 批量迭代器 |
| `save_model(model, path)` / `load_model(path)` | JSON 持久化 |
| `clip_grad_by_norm(grads, max_norm)` | 梯度范数裁剪 |
| `clip_grad_by_value(grads, clip_val)` | 梯度值裁剪 |

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
| `generate_text(network, tp, start, length)` | 自回归文本生成 |
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
pos_enc = PositionalEncoding(d_model=4)
encoder = TransformerEncoder(d_model=4, num_heads=2, d_ff=8, num_layers=1)

x = pos_enc.forward(seq)          # 加位置编码
x = encoder.forward(x)            # 经过 Transformer
pooled = [mean over tokens]       # 全局池化
logits = classification_head(pooled)
```

### RNN / LSTM / GRU

```python
rnn = RNN(input_size=1, hidden_size=4, num_layers=1)
lstm = LSTM(input_size=1, hidden_size=4, num_layers=1)
gru = GRU(input_size=1, hidden_size=4, num_layers=1)

outputs, _ = rnn.forward(seq)     # outputs: 每个时间步的隐藏状态
last_h = outputs[-1]              # 取最后一步做预测
```

### 文本生成

```python
tp = TextProcessor(text)
dataset = tp.prepare_data(text, window_size=3)

# 训练字符级语言模型...
model = MLP([vocab_size * window_size, 64, vocab_size], 
            activations=["leaky_relu", "softmax"])

# 自回归生成
generated = generate_text(model, tp, "君不见", length=15, temp=0.8)
```

## 运行示例

```bash
python AI.py
```

将依次运行：回归 → 分类 → 保存/加载 → CNN → RNN/LSTM → GRU → Dropout+LayerNorm → Gradient Clipping → quick_train → Transformer → 文本生成。

## 测试

```bash
python test_ai.py
```

全部 108 个测试覆盖所有组件。

## 文件结构

```
.
├── AI.py         # 神经网络工具箱（所有代码）
├── test_ai.py    # 108 个单元测试
└── README.md
```
