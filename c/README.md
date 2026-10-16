# C 移植（核心数值路径）

这是 `AI.py` 的 C 移植。**已实现的部分经过跨语言数值比对验证**：在相同初始权重下，
前向传播与梯度累加和 Python 版**逐位一致**（差异 0.000e+00），5 步 Adam 后差异 1 ULP。

## 结论先说

| 问题 | 回答 |
|------|------|
| 能不能转成 C？ | 能。数值核心（全连接层、损失、优化器、训练循环）已经跑通并通过等价性验证 |
| 快多少？ | 实测 **约 58–68 倍**（MLP[64,256,128,8]、5 万参数、1000 样本 × 5 轮：C 1.3 s vs Python 77 s；空载时测到 0.7–0.9 s vs 64 s）。注意测量时机器的 CPU 负载约 63%（有其它进程在跑），绝对值偏大，比值仍可比 |
| 训练循环有额外开销吗？ | 没有：预热后 `trainer_fit` **零分配**（用 `--wrap` 统计分配次数验证，见下） |
| 全部转完要多久？ | 已完成的约 900 行覆盖了训练主路径；剩下 Conv2d / RNN 家族 / 注意力 / BatchNorm / JSON / 文本生成，估计还要 2000–3000 行 |

## 构建与运行

```bash
cd c

# 有 make
make
./demo              # 跑 README 快速开始那两段示例（分类 + 回归）
./bench 5 1000      # 性能基准（轮数、样本数）

# 没有 make（本仓库的开发环境就没有）
gcc -std=c17 -O2 -Wall -Wextra -o demo demo.c ai.c -lm
gcc -std=c17 -O2 -Wall -Wextra -o compare_python compare_python.c ai.c -lm
gcc -std=c17 -O2 -Wall -Wextra -o bench bench.c ai.c -lm
```

验证与 Python 的数值一致性：

```bash
python ../tools/c_reference.py reference.txt   # Python 侧导出参考数据
./compare_python reference.txt                 # C 侧重算并逐项比对
```

实测输出：

```
1) 前向传播（6 个样本）
  forward output               最大差异 0.000e+00  [OK]
2) 梯度累加（6 个样本，未除以 N）
  accumulated grad W           最大差异 0.000e+00  [OK]
  accumulated grad b           最大差异 0.000e+00  [OK]
3) 训练 5 步全批量 Adam(lr=0.05)
  weights after training       最大差异 2.220e-16  [OK]
  biases  after training       最大差异 5.551e-17  [OK]
```

### 训练循环零分配（与机器负载无关的检查）

计时在这台机器上不可靠（有其它进程占着 CPU），所以另外做了一项**不依赖时间**的验证：
用链接器的 `--wrap` 把 `malloc/calloc/realloc` 换成计数版，预热一次后统计第二次
`trainer_fit` 期间的分配次数。

```bash
gcc -std=c17 -O2 -Wall -Wextra -Wl,--wrap=malloc -Wl,--wrap=calloc \
    -Wl,--wrap=realloc -o check_noalloc check_noalloc.c ai.c -lm
./check_noalloc
```

```
training 3 epochs x 200 samples x MLP[64,256,128,8]
  allocations during fit (after warmup): 0  (0 bytes)
  -> zero-allocation hot path: OK
```

也就是说所有缓冲（各层的梯度/优化器状态、反向传播的 delta 乒乓缓冲、
训练器的 pred/grad/索引缓冲）都在建模型或首次调用时一次性分配好，
训练过程中不再碰堆。

## 文件

```
c/
├── ai.h            接口与设计约定
├── ai.c             实现（约 700 行）
├── demo.c           对应 README 快速开始的示例
├── bench.c          性能基准
├── compare_python.c 跨语言等价性比对
├── check_noalloc.c  训练循环零分配验证（用 --wrap 统计分配）
├── Makefile
└── reference.txt    由 tools/c_reference.py 生成（已 gitignore）
tools/
├── c_reference.py   导出 Python 参考数据
└── c_bench.py       与 C 对等的 Python 基准
```

## 已移植

| Python | C |
|--------|---|
| `Layer` / `MLP` | `Layer` / `MLP`（`act_apply` / `act_deriv` / `he_init`） |
| `mse_loss` / `cross_entropy_loss` | `mse_loss` / `mse_grad` / `ce_loss_grad` |
| `Adam` / `SGD` / `SGDMomentum` / `RMSprop` | `optimizer_init` / `optimizer_step` |
| `Model.fit`（批量、验证集切分、Early Stopping、梯度裁剪） | `trainer_fit` + `FitConfig` |
| `Model.forward` / `predict` / `score` | `mlp_forward` / `trainer_predict` / `trainer_accuracy` / `trainer_r2` |
| `save_model` / `load_model` | `mlp_save` / `mlp_load`（文本格式） |
| `clip_grad_by_norm` | `clip_by_norm`（`FitConfig.grad_clip`） |
| `DataLoader` / `split_data` | 内联进 `trainer_fit`（打乱 + 切分） |

## 未移植（以及难在哪）

| 模块 | 难度 | 说明 |
|------|------|------|
| `Conv2d` / `MaxPool2d` | 低 | 纯数组运算，`im2col` 在 C 里反而更自然 |
| `BatchNorm1d` / `LayerNorm` / `Dropout` | 低 | 逐元素运算，需要一个 `Layer` 之外的联合体或 vtable |
| `RNN` / `LSTM` / `GRU` | 中 | 需要 `_StepCacheMixin` 那套"按时间步缓存"的结构体化管理 |
| `MultiHeadAttention` / `TransformerEncoder` | 中 | 张量维度变多（seq × heads × d_k），要么引入通用 `Tensor`，要么为每种形状写专用代码 |
| `TextProcessor` / `generate_text` | 中 | 需要字符哈希表（C 没有内建 dict）与字符串拼接 |
| `Model.save` 的 **JSON** 格式 | 中 | 需要一个 JSON 解析器才能与 Python 存档互读；当前用简单文本格式替代 |
| `Sequential` / `ResidualBlock` 这类容器 | **高** | Python 靠鸭子类型（任何有 `forward`/`backward` 的对象都能塞进去）+ 元组/list 两种返回约定。C 里要用**函数指针 + 标签联合**模拟，或者放弃通用容器、改成显式的层数组 |
| `lr_scheduler` / `StepLR` 等 | 低 | 只是按 epoch 调 `lr` |

也就是说：**数值计算部分好移植（已完成并验证），难的是 Python 的动态类型那一层**。
如果只是要速度，上面的核心已经能覆盖 MLP 训练的全部热路径。

## 两个关键实现细节

### 1. 为什么能做到逐位一致：Neumaier 补偿求和

Python 3.12+ 的 `sum()` 对浮点使用 Neumaier 补偿求和（不是朴素顺序累加）。
如果 C 侧用普通 `s += a[i]*b[i]`，前向结果会差 1 ULP 量级，并随层数/步数累积。
所以 C 侧的点积与 MSE 求和照搬了同一算法：

```c
double s = 0.0, c = 0.0;
for (i...) {
    double x = a[i] * b[i];
    double t = s + x;
    if (fabs(s) >= fabs(x)) c += (s - t) + x; else c += (x - t) + s;
    s = t;
}
return s + c;
```

这正是"前向 0.000e+00、梯度 0.000e+00"的原因。Adam 里剩下的 1 ULP 差异来自
`pow(beta, step)` 等函数在不同 libm 下的末位差别，属正常范围。

### 2. softmax + cross-entropy 的联合梯度约定

Python 版 `Layer._softmax_deriv` 恒返回 1，只在与 CE 搭配时才正确（CE 直接返回
`pred - one_hot`）。C 侧刻意保留同一约定（`ACT_SOFTMAX` 的 `act_deriv` 全为 1），
因此 **softmax 输出层同样只能配 `LOSS_CROSS_ENTROPY`**。

## 内存模型与取舍

- 权重按**行主序连续数组**存放（`W[j*fan_in + i]`），而不是 Python 的嵌套 list：
  这是 C 版快 68 倍的主要来源之一（缓存友好 + 无对象头开销）
- 每个 `Layer` 自带梯度与优化器状态缓冲区，一次 `malloc` 后训练过程**零分配**
  （`layer_backward` 里的临时数组是唯一例外，见"已知问题"）
- 全部用 `double`，与 Python 的 float（f64）一致
- 只依赖 libc + libm，没有第三方库，和原项目的"零依赖"定位一致
- C 程序的**控制台输出刻意用 ASCII**，避免 Windows 代码页把中文变成乱码；
  源码注释与文档仍是中文

## 已知问题 / 待改进

1. 单线程。矩阵运算没有用 SIMD / OpenMP（这台机器的 gcc 支持 `-fopenmp`），
   加上之后大矩阵还能再快数倍。当前每个样本的 `z_j = Σ W[j][i]·x[i]` 是标量循环，
   编译器只能有限度地自动向量化。
2. `trainer_fit` 的验证集切分用的是自带 xorshift RNG，与 Python 的 MT 不同，
   所以"随机打乱后"的训练轨迹无法逐位对齐（等价性测试因此用 `shuffle=False`）。
   若需要逐位复现 Python 的 shuffle，得把 MT19937 + `random.shuffle` 的取数方式
   一并移植过来。
3. `mlp_save` / `mlp_load` 用的是简单文本格式，**不能直接读写 Python 版的 JSON**
   （互转由 `tools/c_reference.py` 负责）。要真正互通需要一个 JSON 解析器。
4. `Optimizer` 的其它超参（`beta1/beta2/eps/momentum/alpha`）目前只能在
   `optimizer_init` 之后手动改，没有像 Python 那样做成构造参数。

已修（记录一下，免得回归）：
- `layer_backward` 曾在每次调用时 malloc 两个临时数组 → 改为 Layer 上预分配
- Adam 的 `pow(beta, step)` 曾每层都算一遍 → 改为每步只算一次
- `mse_loss` 曾为补偿求和分配临时数组 → 改为内联累加
- `trainer_fit` 的索引数组、`trainer_accumulate` 的 pred/grad 缓冲 →
  全部改为懒分配的复用缓冲，最终做到训练循环零分配
