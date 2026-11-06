"""
================================================================================
  全栈神经网络工具箱（纯 Python 标准库实现，零第三方依赖）
  - Layer / MLP（全连接层 + 激活函数）
  - Conv2d（含 groups 分组/深度卷积、dilation 空洞卷积）/ Conv1d / ConvTranspose2d / MaxPool2d（卷积与池化）
  - RNN / LSTM / GRU（循环神经网络家族，含完整 BPTT）
  - Transformer（MultiHeadAttention 支持 causal / RoPE / 滑动窗口 / GQA / 注意力 dropout / KV cache，
    另有 CrossAttention / Encoder / Decoder / PositionalEncoding）
  - CausalLM（因果语言模型）/ Seq2SeqLM（编码器-解码器序列到序列），都支持 padded 批量训练
  - Dropout / DropPath / BatchNorm1d / BatchNorm2d / LayerNorm / RMSNorm / GroupNorm / InstanceNorm（正则化与归一化）
  - Embedding / ResidualBlock / Sequential（基础组件）+ MaxPool2d / AvgPool2d / GlobalAvgPool2d / AdaptiveAvgPool2d（池化）
  - Adam / AdamW / SGD / SGDMomentum / RMSprop（优化器）
  - MSE / L1 / Log-Cosh / Quantile / Smooth-L1 / CrossEntropy / Focal / BCE / Huber / KL 损失函数
  + 类别不平衡损失（Class-Balanced / LDAM / OHEM）
  + 知识蒸馏（distill_loss 软目标 KL + T² 缩放 / hint_loss 隐藏层对齐）
  + 偏好对齐（preference_loss：DPO / IPO / SimPO / CPO / ORPO；kto_loss）
  + 多任务权重（uncertainty_weighting：Kendall 不确定性加权；gradnorm_weights；pcgrad 冲突投影）
  - 梯度裁剪 / 学习率调度器（StepLR / ExponentialLR / CosineAnnealingLR / WarmupLR）
  - 训练 / 验证 / 测试集拆分（可分层）+ K 折索引 + Early Stopping + DataLoader
  - StandardScaler / MinMaxScaler 特征缩放、ascii_plot 训练曲线、grad_check 梯度自检
  - 训练基础设施：set_seed / Callback 回调 / infer_shapes / predict_batch / 激活与死单元诊断
  - 评估指标：precision / recall / F1、confusion_matrix、ROC-AUC、top-k、MCC、
    MAE / RMSE / MAPE / R²、sequence_accuracy、BLEU / ROUGE-L、evaluate_all
  - 文本处理：TextProcessor（字符级）/ BytePairTokenizer（字节级 BPE，任意 UTF-8 无损往返）
  - 模型保存 / 加载（JSON 格式，可选带优化器状态与损失配置做逐位续训；CausalLM / Seq2SeqLM 另有 save / load）
================================================================================
"""

# 目录（按文件顺序，与下面的 # ==== 分节标题一一对应）
#   1. Layer & MLP
#   2. CNN: Conv2d / MaxPool2d / Flatten
#   3. Regularization: Dropout / BatchNorm1d / LayerNorm
#   4. RNN / LSTM
#   5. Matrix helpers (用于 Transformer)
#   6. GRU (Gated Recurrent Unit)
#   7. Transformer 组件
#   8. Embedding / Residual / Sequential
#   9. Loss Functions
#  10. Optimizer: Adam
#  11. More Optimizers: SGD / SGD+Momentum / RMSprop
#  12. Gradient Clipping / LR Scheduler
#  13. Data Utilities
#  14. Text Processing（分词与文本工具）
#  15. Trainer
#  16. 评估指标（分类 / 回归 / 序列 / 生成）
#  17. 模型保存 / 加载 (JSON)
#  18. 序列标注 / 多标签分类头
#  19. PyTorch 数值对齐对比
#  20. 训练基础设施：seed / Callback / 形状推断 / 批量推理 / 诊断
#  21. High-Level API: Model 封装（sklearn 风格） + 工厂函数
#  22. 生成模型：VAE / GAN / DDPM
#  23. Main: 回归 + 分类 + CNN 完整示例


import io
import itertools
import json
import logging
import os
import sys

import math
import operator
import random
import time
from typing import List, Tuple, Optional, Callable, Any

# 热路径别名：Layer / _matmul 每次调用都要用到 operator.mul
_mul = operator.mul

# GELU 的 tanh 近似常数：sqrt(2/pi)
_GELU_C = math.sqrt(2.0 / math.pi)
# SiLU / Mish / ELU / SELU 的标量实现（Layer 与 FeedForward / SwiGLU 共用，
# 保证同一种激活在各处数值完全一致）
_SELU_LAMBDA = 1.0507009873554805
_SELU_ALPHA = 1.6732632423543772
_ELU_ALPHA = 1.0


def _silu_scalar(z: float) -> float:
    """SiLU / Swish：z * sigmoid(z)（LLaMA 的门控激活）"""
    if z >= 0.0:
        return z / (1.0 + math.exp(-z))
    e = math.exp(z)
    return z * e / (1.0 + e)


def _dsilu_scalar(z: float) -> float:
    if z >= 0.0:
        s = 1.0 / (1.0 + math.exp(-z))
    else:
        e = math.exp(z)
        s = e / (1.0 + e)
    return s * (1.0 + z * (1.0 - s))


def _softplus_scalar(z: float) -> float:
    """softplus(z) = log(1+exp(z))，z 很大时退化成 z（避免 exp 溢出）"""
    if z > 30.0:
        return z
    return math.log1p(math.exp(z))


def _mish_scalar(z: float) -> float:
    return z * math.tanh(_softplus_scalar(z))


def _dmish_scalar(z: float) -> float:
    sp = _softplus_scalar(z)
    t = math.tanh(sp)
    if z >= 0.0:
        s = 1.0 / (1.0 + math.exp(-z))
    else:
        e = math.exp(z)
        s = e / (1.0 + e)
    return t + z * (1.0 - t * t) * s


def _elu_scalar(z: float) -> float:
    return z if z > 0.0 else _ELU_ALPHA * (math.exp(z) - 1.0)


def _delu_scalar(z: float) -> float:
    return 1.0 if z > 0.0 else _ELU_ALPHA * math.exp(z)


def _selu_scalar(z: float) -> float:
    if z > 0.0:
        return _SELU_LAMBDA * z
    return _SELU_LAMBDA * _SELU_ALPHA * (math.exp(z) - 1.0)


def _dselu_scalar(z: float) -> float:
    if z > 0.0:
        return _SELU_LAMBDA
    return _SELU_LAMBDA * _SELU_ALPHA * math.exp(z)


# ============================================================================
#  Layer & MLP
# ============================================================================

class Layer:
    """一个全连接层 + 激活函数"""

    def __init__(self, fan_in: int, fan_out: int,
                 activation: str = "leaky_relu", is_output: bool = False):
        self.fan_in = fan_in
        self.fan_out = fan_out
        self.is_output = is_output

        # He 初始化
        std = math.sqrt(2.0 / fan_in)
        self.weights = [
            [random.gauss(0, std) for _ in range(fan_in)]
            for _ in range(fan_out)
        ]
        self.biases = [0.0] * fan_out

        # Adam 状态
        self.m_w = [[0.0] * fan_in for _ in range(fan_out)]
        self.v_w = [[0.0] * fan_in for _ in range(fan_out)]
        self.m_b = [0.0] * fan_out
        self.v_b = [0.0] * fan_out

        # 前向缓存
        self.input = None
        self.z = None
        self.a = None

        # 激活函数分发
        if activation == "leaky_relu":
            self.act_fn = self._leaky_relu
            self.act_deriv = self._leaky_relu_deriv
        elif activation == "relu":
            self.act_fn = self._relu
            self.act_deriv = self._relu_deriv
        elif activation == "softmax":
            self.act_fn = self._softmax
            self.act_deriv = self._softmax_deriv
        elif activation == "linear":
            self.act_fn = self._linear
            self.act_deriv = self._linear_deriv
        elif activation == "gelu":
            self.act_fn = self._gelu
            self.act_deriv = self._gelu_deriv
        elif activation == "sigmoid":
            self.act_fn = self._sigmoid
            self.act_deriv = self._sigmoid_deriv
        elif activation == "silu":
            self.act_fn = self._silu
            self.act_deriv = self._silu_deriv
        elif activation == "mish":
            self.act_fn = self._mish
            self.act_deriv = self._mish_deriv
        elif activation == "elu":
            self.act_fn = self._elu
            self.act_deriv = self._elu_deriv
        elif activation == "selu":
            self.act_fn = self._selu
            self.act_deriv = self._selu_deriv
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # 记录激活函数名字符串（save_model 持久化 / 激活与损失的搭配校验用）
        self.activation_name = activation

# ============================================================================
#  激活函数
# ============================================================================
    def _linear(self, z):
        return z

    def _linear_deriv(self, z):
        return [1.0] * len(z)

    def _relu(self, z):
        return [max(0, zi) for zi in z]

    def _relu_deriv(self, z):
        return [1.0 if zi > 0 else 0.0 for zi in z]

    def _leaky_relu(self, z):
        return [zi if zi > 0 else 0.01 * zi for zi in z]

    def _leaky_relu_deriv(self, z):
        return [1.0 if zi > 0 else 0.01 for zi in z]

    def _gelu(self, z):
        # tanh 近似：0.5*z*(1+tanh(sqrt(2/pi)*(z+0.044715 z^3)))
        # 纯标准库没有 erf，GPT-2 用的 tanh 近似与精确 erf 版差异 < 2e-3
        out = []
        for zi in z:
            u = _GELU_C * (zi + 0.044715 * zi * zi * zi)
            out.append(0.5 * zi * (1.0 + math.tanh(u)))
        return out

    def _gelu_deriv(self, z):
        out = []
        for zi in z:
            u = _GELU_C * (zi + 0.044715 * zi * zi * zi)
            t = math.tanh(u)
            du = _GELU_C * (1.0 + 3.0 * 0.044715 * zi * zi)
            out.append(0.5 * (1.0 + t) + 0.5 * zi * (1.0 - t * t) * du)
        return out

    def _sigmoid(self, z):
        # 分支写法：x 很负时 exp(x) 不溢出（C 侧同一写法，逐位一致）
        out = []
        for zi in z:
            if zi >= 0:
                out.append(1.0 / (1.0 + math.exp(-zi)))
            else:
                e = math.exp(zi)
                out.append(e / (1.0 + e))
        return out

    def _sigmoid_deriv(self, z):
        s = self._sigmoid(z)
        return [si * (1.0 - si) for si in s]

    def _silu(self, z):
        return [_silu_scalar(zi) for zi in z]

    def _silu_deriv(self, z):
        return [_dsilu_scalar(zi) for zi in z]

    def _mish(self, z):
        return [_mish_scalar(zi) for zi in z]

    def _mish_deriv(self, z):
        return [_dmish_scalar(zi) for zi in z]

    def _elu(self, z):
        return [_elu_scalar(zi) for zi in z]

    def _elu_deriv(self, z):
        return [_delu_scalar(zi) for zi in z]

    def _selu(self, z):
        return [_selu_scalar(zi) for zi in z]

    def _selu_deriv(self, z):
        return [_dselu_scalar(zi) for zi in z]

    def _softmax(self, z):
        max_z = max(z)
        exp_z = [math.exp(zi - max_z) for zi in z]
        sum_exp = sum(exp_z)
        return [e / sum_exp for e in exp_z]

    def _softmax_deriv(self, z):
        # 注意：这里返回 [1.0] 是简化处理，仅在与 cross_entropy_loss 配合时正确。
        # cross_entropy_loss 已直接返回 softmax+CE 的联合梯度 (pred - one_hot)，
        # 无需再乘 softmax 的 Jacobian。若搭配 mse_loss 使用 softmax，梯度会错误。
        return [1.0] * len(z)

# ============================================================================
#  前向 / 反向
# ============================================================================
    def forward(self, x):
        # 维度不符时 map 会静默截断（少算或多算），必须早失败
        if len(x) != self.fan_in:
            hint = ""
            if x and isinstance(x[0], (list, tuple)):
                hint = "（输入看起来是图像/批量，请先用 Flatten 展平）"
            raise ValueError(
                f"Layer.forward: 输入长度 {len(x)} 与本层 fan_in={self.fan_in} 不一致"
                f"（本层 fan_out={self.fan_out}，激活={getattr(self, 'activation_name', '?')}"
                f"，is_output={getattr(self, 'is_output', '?')}）{hint}"
            )
        # 权重形状只在对象被替换后检查一次（Adam 只改元素、不换行对象），
        # 避免每次前向都做 O(fan_out) 的遍历
        if self.__dict__.get("_w_checked_for") is not self.weights:
            bad = (len(self.weights) != self.fan_out
                   or any(len(row) != self.fan_in for row in self.weights))
            if bad:
                raise ValueError(
                    f"Layer.forward: 权重形状应为 {self.fan_out}×{self.fan_in}，"
                    f"实际 {len(self.weights)} 行（首行 {len(self.weights[0]) if self.weights else 0} 列），"
                    f"模型可能已损坏（请检查加载的文件）"
                )
            self.__dict__["_w_checked_for"] = self.weights
        self.input = x
        # sum(map(mul, row, x)) 走 C 层的 map/sum，比 sum(w*xi for ...) 快约 1.8 倍
        W, b = self.weights, self.biases
        self.z = [sum(map(_mul, W[j], x)) + b[j] for j in range(self.fan_out)]
        self.a = self.act_fn(self.z)
        return self.a

    def backward(self, delta):
        """delta: 本层输出的梯度

        返回 **(grad_w, grad_b, next_delta)** —— 三元组，最后一个元素才是传给
        上一层的 delta。容器（`Sequential` / `ResidualBlock`）通过模块的
        `delta_index` 属性决定取哪一项（默认 -1；`BatchNorm1d` 的 dx 在第 0 位）。
        """
        if self.z is None:
            raise RuntimeError("Layer.backward: 必须先调用 forward()（没有缓存的 z）")
        if len(delta) != self.fan_out:
            raise ValueError(
                f"Layer.backward: delta 长度 {len(delta)} 与本层输出维度 {self.fan_out} 不一致"
            )
        d_act = self.act_deriv(self.z)

        # delta * d_act 同时是 grad_b 与回传给上一层的权重向量，只算一次
        w = [delta[j] * d_act[j] for j in range(self.fan_out)]
        grad_w = [[s * xi for xi in self.input] for s in w]
        grad_b = list(w)

        # 传给上一层的 delta = W^T · (delta * d_act)
        # zip(*W) 在 C 层完成转置，再用 map(mul) 做点积，比双重索引的 genexpr 快约 1.9 倍
        next_delta = [sum(map(_mul, col, w)) for col in zip(*self.weights)]

        return grad_w, grad_b, next_delta


class Linear(Layer):
    """无激活的全连接层：y = Wx + b

    等价于 Layer(..., activation="linear")，单独命名是为了让"纯投影"在代码里
    一目了然（也与 PyTorch nn.Linear 的语义对齐）。
    """

    def __init__(self, fan_in: int, fan_out: int):
        super().__init__(fan_in, fan_out, activation="linear")


class MLP:
    """多层感知机（PyTorch 风格）"""

    def __init__(self, layer_sizes: List[int],
                 activations: Optional[List[str]] = None):
        if len(layer_sizes) < 2:
            raise ValueError(f"MLP: layer_sizes 至少需要 2 个维度（输入+输出），收到 {layer_sizes}")
        if any(d <= 0 for d in layer_sizes):
            raise ValueError(f"MLP: 每一维都必须 >= 1，收到 {layer_sizes}")
        self.layer_sizes = layer_sizes

        if activations is None:
            activations = ["leaky_relu"] * (len(layer_sizes) - 2)
            activations.append("linear")

        # 必须在默认填充之后校验：否则 activations=None 时这里就 len(None) 崩了
        if len(activations) != len(layer_sizes) - 1:
            raise ValueError(
                f"MLP: 需要 {len(layer_sizes) - 1} 个激活函数（层数 = 维度数-1），"
                f"收到 {len(activations)} 个"
            )

        self.layers = []
        for i in range(1, len(layer_sizes)):
            is_output = (i == len(layer_sizes) - 1)
            layer = Layer(
                fan_in=layer_sizes[i - 1],
                fan_out=layer_sizes[i],
                activation=activations[i - 1],
                is_output=is_output,
            )
            self.layers.append(layer)

        # 前向/反向 hook：{层索引: 回调}。没注册时下面只多一次真假判断，
        # 不影响任何数值（注册了也只观察，不参与计算）
        self._fhooks = {}
        self._bhooks = {}

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer.forward(x)
            if self._fhooks:
                fn = self._fhooks.get(i)
                if fn is not None:
                    fn(layer, x)
        return x

# ── hook（调试/可视化用，注册了才生效）

    def register_forward_hook(self, index: int, fn):
        """在第 index 层前向后调用 fn(layer, output)；返回 fn 本身便于保存"""
        if not callable(fn):
            raise TypeError(f"MLP.register_forward_hook: fn 必须可调用，收到 {fn!r}")
        if not isinstance(index, int) or isinstance(index, bool):
            raise TypeError(f"MLP.register_forward_hook: index 必须是整数，收到 {index!r}")
        if not 0 <= index < len(self.layers):
            raise ValueError(
                f"MLP.register_forward_hook: index={index} 越界（共 {len(self.layers)} 层）"
            )
        self._fhooks[index] = fn
        return fn

    def register_backward_hook(self, index: int, fn):
        """在第 index 层反向后调用 fn(layer, delta_in, grad_w, grad_b)"""
        if not callable(fn):
            raise TypeError(f"MLP.register_backward_hook: fn 必须可调用，收到 {fn!r}")
        if not isinstance(index, int) or isinstance(index, bool):
            raise TypeError(f"MLP.register_backward_hook: index 必须是整数，收到 {index!r}")
        if not 0 <= index < len(self.layers):
            raise ValueError(
                f"MLP.register_backward_hook: index={index} 越界（共 {len(self.layers)} 层）"
            )
        self._bhooks[index] = fn
        return fn

    def remove_hook(self, index: int, kind: str = 'forward'):
        """注销某一层的 hook（kind 取 'forward' / 'backward'）"""
        table = self._fhooks if kind == 'forward' else self._bhooks
        if kind not in ('forward', 'backward'):
            raise ValueError(f"MLP.remove_hook: kind 只能是 'forward'/'backward'，收到 {kind!r}")
        table.pop(index, None)

    def clear_hooks(self):
        """注销全部 hook"""
        self._fhooks.clear()
        self._bhooks.clear()

    def get_parameters(self):
        params = []
        for layer in self.layers:
            params.append((layer.weights, layer.biases))
        return params

# ── 参数量与模型瘦身（与 C 引擎 c_model.cpp 的实现语义一致）

    def param_count(self):
        """可训练参数总数（权重 + 偏置）"""
        return sum(l.fan_in * l.fan_out + l.fan_out for l in self.layers)

    def sparsity(self):
        """权重中恰好为 0 的比例（剪枝后用来看稀疏度）"""
        total = sum(l.fan_in * l.fan_out for l in self.layers)
        if total == 0:
            return 0.0
        zeros = sum(1 for l in self.layers for row in l.weights for w in row if w == 0.0)
        return zeros / total

    def prune_magnitude(self, threshold):
        """幅度剪枝：把 |w| < threshold 的权重置 0，返回被剪掉的个数"""
        if threshold <= 0:
            return 0
        pruned = 0
        for l in self.layers:
            for j in range(l.fan_out):
                row = l.weights[j]
                for i in range(l.fan_in):
                    if abs(row[i]) < threshold:
                        row[i] = 0.0
                        pruned += 1
        return pruned

    def quantize_weights(self, bits):
        """权重定点量化：按全局 min/max 均匀分档，返回量化步长（bits 需在 2..24）"""
        if not (2 <= bits <= 24):
            return -1.0
        vals = [w for l in self.layers for row in l.weights for w in row]
        if not vals:
            return -1.0
        lo, hi = min(vals), max(vals)
        if hi <= lo:
            return -1.0
        step = (hi - lo) / ((1 << bits) - 1)
        for l in self.layers:
            for j in range(l.fan_out):
                row = l.weights[j]
                for i in range(l.fan_in):
                    row[i] = lo + round((row[i] - lo) / step) * step
        return step

    def merge_weights(self, other, w_self=0.5, w_other=0.5):
        """权重线性合并：self = w_self*self + w_other*other（结构必须一致）

        做 K 折权重平均时各折必须从同一初值出发，否则隐藏单元排列不同，
        直接平均会得到无意义的模型（这一点在 C 引擎的 CLI 里踩过）。
        """
        if self.layer_sizes != other.layer_sizes:
            raise ValueError(f"结构不一致: {self.layer_sizes} vs {other.layer_sizes}")
        for la, lb in zip(self.layers, other.layers):
            for j in range(la.fan_out):
                ra, rb = la.weights[j], lb.weights[j]
                for i in range(la.fan_in):
                    ra[i] = w_self * ra[i] + w_other * rb[i]
                la.biases[j] = w_self * la.biases[j] + w_other * lb.biases[j]
        return self

    def __repr__(self):
        return f"MLP({self.layer_sizes})"


# ============================================================================
#  CNN: Conv2d / MaxPool2d / Flatten
# ============================================================================

class Conv2d:
    """2D 卷积层 (im2col 实现, 纯 Python 列表)"""

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size, stride=1, padding=0, groups: int = 1,
                 dilation=1):
        if not isinstance(groups, int) or isinstance(groups, bool) or groups < 1:
            raise ValueError(f"Conv2d: groups 必须是 >=1 的整数，收到 {groups!r}")
        if in_channels % groups != 0:
            raise ValueError(
                f"Conv2d: in_channels={in_channels} 必须能被 groups={groups} 整除"
            )
        if out_channels % groups != 0:
            raise ValueError(
                f"Conv2d: out_channels={out_channels} 必须能被 groups={groups} 整除"
            )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        # 每个输出通道只连本组的 C/groups 个输入通道（groups=1 时就是原来的 C）
        self.cg = in_channels // groups
        self.og = out_channels // groups
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.padding = (padding, padding) if isinstance(padding, int) else padding
        self.dilation = ((dilation, dilation) if isinstance(dilation, int)
                         else tuple(dilation))
        self.kh, self.kw = self.kernel_size
        self.sh, self.sw = self.stride
        if self.kh <= 0 or self.kw <= 0:
            raise ValueError(f"Conv2d: kernel_size 必须为正，收到 {self.kernel_size}")
        if self.sh <= 0 or self.sw <= 0:
            raise ValueError(f"Conv2d: stride 必须为正，收到 {self.stride}")
        if min(self.padding) < 0:
            raise ValueError(f"Conv2d: padding 不能为负，收到 {self.padding}")
        if len(self.dilation) != 2 or min(self.dilation) <= 0:
            raise ValueError(f"Conv2d: dilation 必须是正数（或二元组），收到 {dilation!r}")
        self.ph, self.pw = self.padding
        self.dh, self.dw = self.dilation
        # 空洞卷积的等效感受野：(k-1)*d + 1
        self.kh_eff = (self.kh - 1) * self.dh + 1
        self.kw_eff = (self.kw - 1) * self.dw + 1

        # He 初始化（groups=1 时 fan_in 与原来的 in_channels*kh*kw 相同）
        fan_in = self.cg * self.kh * self.kw
        std = math.sqrt(2.0 / fan_in)
        self.weights = [
            [[[random.gauss(0, std) for _ in range(self.kw)]
              for _ in range(self.kh)]
             for _ in range(self.cg)]
            for _ in range(out_channels)
        ]
        self.biases = [0.0] * out_channels

        # Adam 状态
        self.m_w = [[[[0.0] * self.kw for _ in range(self.kh)]
                      for _ in range(self.cg)] for _ in range(out_channels)]
        self.v_w = [[[[0.0] * self.kw for _ in range(self.kh)]
                      for _ in range(self.cg)] for _ in range(out_channels)]
        self.m_b = [0.0] * out_channels
        self.v_b = [0.0] * out_channels

        # 前向缓存
        self.x_col = None
        self.x_shape = None

        self.zero_grad()

    def zero_grad(self):
        """初始化/清零梯度缓冲区（保留前向缓存 h_out/w_out 供 backward 使用）"""
        self.grad_w = [
            [[[0.0] * self.kw for _ in range(self.kh)]
             for _ in range(self.cg)]
            for _ in range(self.out_channels)
        ]
        self.grad_b = [0.0] * self.out_channels

# ============================================================================
#  辅助
# ============================================================================
    def _img_shape(self, H, W):
        ho = (H + 2 * self.ph - self.kh_eff) // self.sh + 1
        wo = (W + 2 * self.pw - self.kw_eff) // self.sw + 1
        # 输入比卷积核还小时 ho/wo 会是 0 或负数，原实现返回空输出而不报错
        if ho <= 0 or wo <= 0:
            raise ValueError(
                f"Conv2d: 输入 {H}x{W} 配 kernel {self.kh}x{self.kw}"
                f"（dilation {self.dh}/{self.dw} → 等效 {self.kh_eff}x{self.kw_eff}）、"
                f"padding {self.ph}/{self.pw}、stride {self.sh}/{self.sw} "
                f"得到输出 {ho}x{wo}，必须都 > 0"
            )
        return ho, wo

    def _pad(self, x):
        """对 (C, H, W) 做 zero-padding"""
        C = len(x)
        H = len(x[0])
        W = len(x[0][0])
        Hp = H + 2 * self.ph
        Wp = W + 2 * self.pw
        padded = [[[0.0] * Wp for _ in range(Hp)] for _ in range(C)]
        for c in range(C):
            for h in range(H):
                for w in range(W):
                    padded[c][h + self.ph][w + self.pw] = x[c][h][w]
        return padded

    def _im2col(self, x_padded, ho, wo):
        """将 padded (C, Hp, Wp) 转为 im2col 矩阵: (C*kh*kw, ho*wo)"""
        C = len(x_padded)
        col_rows = C * self.kh * self.kw
        col_cols = ho * wo
        col = [[0.0] * col_cols for _ in range(col_rows)]

        if self.sh == 1 and self.sw == 1:
            # stride=1 时每一行就是被卷积核在宽度方向上的连续切片，
            # 直接切片赋值（每个位置的值与逐元素写法完全相同）
            for c in range(C):
                ch = x_padded[c]
                for kh in range(self.kh):
                    for kw in range(self.kw):
                        row = c * self.kh * self.kw + kh * self.kw + kw
                        dst = col[row]
                        for oh in range(ho):
                            # 空洞：第 kh 行取输入的 kh*dh 行，列偏移 kw*dw
                            dst[oh * wo:(oh + 1) * wo] = ch[oh + kh * self.dh][
                                kw * self.dw:kw * self.dw + wo]
            return col

        for c in range(C):
            ch = x_padded[c]                      # 局部化，省掉每元素的二次索引
            for kh in range(self.kh):
                for kw in range(self.kw):
                    row = c * self.kh * self.kw + kh * self.kw + kw
                    dst = col[row]
                    for oh in range(ho):
                        src = ch[oh * self.sh + kh * self.dh]
                        off = oh * wo
                        for ow in range(wo):
                            dst[off + ow] = src[ow * self.sw + kw * self.dw]
        return col

    def _col2im(self, dcol, x_padded_shape, ho, wo):
        """将 dcol 梯度转回图像梯度 (C, Hp, Wp)"""
        C, Hp, Wp = x_padded_shape
        dx = [[[0.0] * Wp for _ in range(Hp)] for _ in range(C)]
        for c in range(C):
            for kh in range(self.kh):
                for kw in range(self.kw):
                    row = c * self.kh * self.kw + kh * self.kw + kw
                    for oh in range(ho):
                        for ow in range(wo):
                            ih = oh * self.sh + kh * self.dh
                            iw = ow * self.sw + kw * self.dw
                            dx[c][ih][iw] += dcol[row][oh * wo + ow]
        return dx

    def _flatten_weights(self, oc):
        """把第 oc 个输出通道的卷积核展平成 (cg*kh*kw,)，顺序与 im2col 的行一致"""
        out = []
        for ic in range(self.cg):
            for kh in range(self.kh):
                for kw in range(self.kw):
                    out.append(self.weights[oc][ic][kh][kw])
        return out

    def _unpad(self, dx_padded, H, W):
        """去掉 padding"""
        C = len(dx_padded)
        dx = [[[0.0] * W for _ in range(H)] for _ in range(C)]
        for c in range(C):
            for h in range(H):
                for w in range(W):
                    dx[c][h][w] = dx_padded[c][h + self.ph][w + self.pw]
        return dx

# ============================================================================
#  前向 / 反向
# ============================================================================
    def forward(self, x):
        """x: (C, H, W) -> 输出 (out_channels, Ho, Wo)"""
        if len(x) != self.in_channels:
            raise ValueError(
                f"Conv2d.forward: 输入通道 {len(x)} != in_channels={self.in_channels}"
            )
        C, H, W = len(x), len(x[0]), len(x[0][0])
        ho, wo = self._img_shape(H, W)
        self.h_out, self.w_out = ho, wo
        self.x_shape = (C, H, W)

        # 分组卷积：每组只把本组的 C/groups 个输入通道卷进本组的 out/groups 个输出通道。
        # groups=1 时循环只跑一次，运算与原来逐位相同。
        self.cols = []
        outs = []
        for g in range(self.groups):
            xg = x if self.groups == 1 else x[g * self.cg:(g + 1) * self.cg]
            xg_pad = self._pad(xg) if (self.ph > 0 or self.pw > 0) else xg
            col = self._im2col(xg_pad, ho, wo)      # (cg*kh*kw, ho*wo)
            self.cols.append(col)

            # 一次 matmul 算出本组输出: (og, cg*kh*kw) × (cg*kh*kw, ho*wo)
            W_row = [self._flatten_weights(oc)
                     for oc in range(g * self.og, (g + 1) * self.og)]
            flat = _matmul(W_row, col)
            outs.append([
                [[flat[oc - g * self.og][oh * wo + ow] + self.biases[oc]
                  for ow in range(wo)]
                 for oh in range(ho)]
                for oc in range(g * self.og, (g + 1) * self.og)
            ])

        # 兼容旧代码（多处直接读 self.x_col）：单组时仍然指向那一份 col
        self.x_col = self.cols[0] if len(self.cols) == 1 else None
        if self.groups == 1:
            return outs[0]
        return [ch for grp in outs for ch in grp]

    def backward(self, dout):
        """
        dout: (out_channels, Ho, Wo) 梯度
        返回 dx: (C, H, W)
        内部累积 grad_w, grad_b
        """
        C, H, W = self.x_shape
        ho, wo = self.h_out, self.w_out

        # 把 dout 展成 (out_channels, ho*wo)
        dout_flat = []
        for oc in range(self.out_channels):
            flat = []
            for oh in range(ho):
                for ow in range(wo):
                    flat.append(dout[oc][oh][ow])
            dout_flat.append(flat)

        dx_groups = []
        for g in range(self.groups):
            col = self.cols[g]

            # grad_w: dW = dout_flat × col^T（累积到内部梯度缓冲区）
            # 即: grad_w[oc][ic][kh][kw] += sum over ho,wo: dout[oc][oh][ow] * col_row[oh*wo]
            # 点积交给 C 层的 map(mul)/sum，求和顺序与逐元素累加一致
            for oc in range(g * self.og, (g + 1) * self.og):
                d = dout_flat[oc]
                for ic in range(self.cg):
                    for kh in range(self.kh):
                        for kw in range(self.kw):
                            row = ic * self.kh * self.kw + kh * self.kw + kw
                            self.grad_w[oc][ic][kh][kw] += sum(map(_mul, d, col[row]))

            for oc in range(g * self.og, (g + 1) * self.og):
                self.grad_b[oc] += sum(dout_flat[oc])

            # dx: col2im  dcol = W_flat^T × dout_flat  (cg*kh*kw, ho*wo)
            dcol = [[0.0] * (ho * wo) for _ in range(len(col))]
            for oc in range(g * self.og, (g + 1) * self.og):
                d = dout_flat[oc]
                w_flat = self._flatten_weights(oc)
                for row in range(len(w_flat)):
                    w = w_flat[row]
                    # 累加顺序仍是「外层 oc、内层逐元素」，与原写法逐位一致
                    dcol[row] = [a + w * b for a, b in zip(dcol[row], d)]

            dx_padded = self._col2im(
                dcol, (self.cg, H + 2 * self.ph, W + 2 * self.pw), ho, wo)

            if self.ph > 0 or self.pw > 0:
                dx_groups.append(self._unpad(dx_padded, H, W))
            else:
                dx_groups.append(dx_padded)

        if self.groups == 1:
            return dx_groups[0]
        return [ch for grp in dx_groups for ch in grp]


class Conv1d:
    """1D 卷积层（输入 (C, L)，权重 [oc][cg][k]，He 初始化）

    没有走 im2col：1D 的核很小，直接滑动窗口累加比先展开成矩阵更省内存。
    groups 语义与 Conv2d 一致（每组只连本组的 C/groups 个输入通道）。
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, groups: int = 1):
        if not isinstance(groups, int) or isinstance(groups, bool) or groups < 1:
            raise ValueError(f"Conv1d: groups 必须是 >=1 的整数，收到 {groups!r}")
        if in_channels % groups != 0:
            raise ValueError(
                f"Conv1d: in_channels={in_channels} 必须能被 groups={groups} 整除"
            )
        if out_channels % groups != 0:
            raise ValueError(
                f"Conv1d: out_channels={out_channels} 必须能被 groups={groups} 整除"
            )
        if not isinstance(kernel_size, int) or isinstance(kernel_size, bool) or kernel_size <= 0:
            raise ValueError(f"Conv1d: kernel_size 必须是正整数，收到 {kernel_size!r}")
        if not isinstance(stride, int) or isinstance(stride, bool) or stride <= 0:
            raise ValueError(f"Conv1d: stride 必须是正整数，收到 {stride!r}")
        if not isinstance(padding, int) or isinstance(padding, bool) or padding < 0:
            raise ValueError(f"Conv1d: padding 必须是非负整数，收到 {padding!r}")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.cg = in_channels // groups
        self.og = out_channels // groups
        self.kernel_size = kernel_size
        self.k = kernel_size
        self.stride = stride
        self.s = stride
        self.padding = padding
        self.p = padding

        fan_in = self.cg * self.k
        std = math.sqrt(2.0 / fan_in)
        self.weights = [[[random.gauss(0, std) for _ in range(self.k)]
                         for _ in range(self.cg)]
                        for _ in range(out_channels)]
        self.biases = [0.0] * out_channels

        self.m_w = [[[0.0] * self.k for _ in range(self.cg)]
                    for _ in range(out_channels)]
        self.v_w = [[[0.0] * self.k for _ in range(self.cg)]
                    for _ in range(out_channels)]
        self.m_b = [0.0] * out_channels
        self.v_b = [0.0] * out_channels

        self.x = None
        self.l_out = 0

        self.zero_grad()

    def zero_grad(self):
        """清零梯度缓冲区"""
        self.grad_w = [[[0.0] * self.k for _ in range(self.cg)]
                       for _ in range(self.out_channels)]
        self.grad_b = [0.0] * self.out_channels

    def _out_len(self, L):
        lo = (L + 2 * self.p - self.k) // self.s + 1
        if lo <= 0:
            raise ValueError(
                f"Conv1d: 输入长度 {L} 配 kernel {self.k}、padding {self.p}、"
                f"stride {self.s} 得到输出长度 {lo}，必须 > 0"
            )
        return lo

    def forward(self, x):
        """x: (C, L) -> (out_channels, Lo)"""
        if len(x) != self.in_channels:
            raise ValueError(
                f"Conv1d.forward: 输入通道 {len(x)} != in_channels={self.in_channels}"
            )
        L = len(x[0])
        for c in range(len(x)):
            if len(x[c]) != L:
                raise ValueError("Conv1d.forward: 各通道长度必须一致")
        lo = self._out_len(L)
        self.x = x
        self.l_out = lo

        out = []
        for oc in range(self.out_channels):
            w_oc = self.weights[oc]
            ic0 = (oc // self.og) * self.cg
            row = [0.0] * lo
            for o in range(lo):
                total = self.biases[oc]
                base = o * self.s - self.p
                for ic in range(self.cg):
                    wic = w_oc[ic]
                    xc = x[ic0 + ic]
                    for k in range(self.k):
                        idx = base + k
                        if 0 <= idx < L:
                            total += wic[k] * xc[idx]
                row[o] = total
            out.append(row)
        return out

    def backward(self, dout):
        """dout: (out_channels, Lo) -> dx: (C, L)，内部累积 grad_w / grad_b"""
        if self.x is None:
            raise RuntimeError("Conv1d.backward: 请先调用 forward")
        L = len(self.x[0])
        lo = self.l_out
        dx = [[0.0] * L for _ in range(self.in_channels)]

        for oc in range(self.out_channels):
            d = dout[oc]
            self.grad_b[oc] += sum(d)
            w_oc = self.weights[oc]
            gw_oc = self.grad_w[oc]
            ic0 = (oc // self.og) * self.cg
            for o in range(lo):
                base = o * self.s - self.p
                dv = d[o]
                for ic in range(self.cg):
                    wic = w_oc[ic]
                    gwic = gw_oc[ic]
                    xc = self.x[ic0 + ic]
                    dxi = dx[ic0 + ic]
                    for k in range(self.k):
                        idx = base + k
                        if 0 <= idx < L:
                            gwic[k] += dv * xc[idx]
                            dxi[idx] += dv * wic[k]
        return dx


class ConvTranspose2d:
    """2D 转置卷积（反卷积）层：输入 (Cin, H, W) -> (Cout, Ho, Wo)

    输出尺寸 Ho = (H-1)*stride - 2*padding + kernel + output_padding。
    权重布局 [cin][og][kh][kw]，与 PyTorch 的 (in, out/groups, kh, kw) 一致。
    forward 把每个输入元素按核散开累加到输出，backward 与之对称。
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size,
                 stride=1, padding=0, output_padding=0, groups: int = 1):
        if not isinstance(groups, int) or isinstance(groups, bool) or groups < 1:
            raise ValueError(f"ConvTranspose2d: groups 必须是 >=1 的整数，收到 {groups!r}")
        if in_channels % groups != 0:
            raise ValueError(
                f"ConvTranspose2d: in_channels={in_channels} 必须能被 groups={groups} 整除"
            )
        if out_channels % groups != 0:
            raise ValueError(
                f"ConvTranspose2d: out_channels={out_channels} 必须能被 groups={groups} 整除"
            )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.cg = in_channels // groups
        self.og = out_channels // groups
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else tuple(kernel_size)
        self.stride = (stride, stride) if isinstance(stride, int) else tuple(stride)
        self.padding = (padding, padding) if isinstance(padding, int) else tuple(padding)
        self.output_padding = ((output_padding, output_padding)
                               if isinstance(output_padding, int) else tuple(output_padding))
        self.kh, self.kw = self.kernel_size
        self.sh, self.sw = self.stride
        self.ph, self.pw = self.padding
        self.oph, self.opw = self.output_padding
        if self.kh <= 0 or self.kw <= 0:
            raise ValueError(f"ConvTranspose2d: kernel_size 必须为正，收到 {self.kernel_size}")
        if self.sh <= 0 or self.sw <= 0:
            raise ValueError(f"ConvTranspose2d: stride 必须为正，收到 {self.stride}")
        if self.ph < 0 or self.pw < 0:
            raise ValueError(f"ConvTranspose2d: padding 不能为负，收到 {self.padding}")
        if self.oph < 0 or self.opw < 0:
            raise ValueError(
                f"ConvTranspose2d: output_padding 不能为负，收到 {self.output_padding}"
            )
        # 与 PyTorch 一致：output_padding 必须小于 stride
        if self.oph >= self.sh or self.opw >= self.sw:
            raise ValueError(
                f"ConvTranspose2d: output_padding {self.output_padding} "
                f"必须小于 stride {self.stride}"
            )

        fan_in = self.cg * self.kh * self.kw
        std = math.sqrt(2.0 / fan_in)
        self.weights = [
            [[[random.gauss(0, std) for _ in range(self.kw)]
              for _ in range(self.kh)]
             for _ in range(self.og)]
            for _ in range(in_channels)
        ]
        self.biases = [0.0] * out_channels

        self.m_w = [[[[0.0] * self.kw for _ in range(self.kh)]
                     for _ in range(self.og)] for _ in range(in_channels)]
        self.v_w = [[[[0.0] * self.kw for _ in range(self.kh)]
                     for _ in range(self.og)] for _ in range(in_channels)]
        self.m_b = [0.0] * out_channels
        self.v_b = [0.0] * out_channels

        self.x = None
        self.x_shape = None
        self.h_out = 0
        self.w_out = 0

        self.zero_grad()

    def zero_grad(self):
        """清零梯度缓冲区"""
        self.grad_w = [[[[0.0] * self.kw for _ in range(self.kh)]
                        for _ in range(self.og)] for _ in range(self.in_channels)]
        self.grad_b = [0.0] * self.out_channels

    def _out_shape(self, H, W):
        ho = (H - 1) * self.sh - 2 * self.ph + self.kh + self.oph
        wo = (W - 1) * self.sw - 2 * self.pw + self.kw + self.opw
        if ho <= 0 or wo <= 0:
            raise ValueError(
                f"ConvTranspose2d: 输入 {H}x{W} 配 kernel {self.kh}x{self.kw}、"
                f"stride {self.sh}/{self.sw}、padding {self.ph}/{self.pw} "
                f"得到输出 {ho}x{wo}，必须都 > 0"
            )
        return ho, wo

    def forward(self, x):
        """x: (Cin, H, W) -> (Cout, Ho, Wo)"""
        if len(x) != self.in_channels:
            raise ValueError(
                f"ConvTranspose2d.forward: 输入通道 {len(x)} != "
                f"in_channels={self.in_channels}"
            )
        H, W = len(x[0]), len(x[0][0])
        for c in range(len(x)):
            if len(x[c]) != H or len(x[c][0]) != W:
                raise ValueError("ConvTranspose2d.forward: 各通道尺寸必须一致")
        ho, wo = self._out_shape(H, W)
        self.x = x
        self.x_shape = (len(x), H, W)
        self.h_out, self.w_out = ho, wo

        out = [[[self.biases[co] for _ in range(wo)] for _ in range(ho)]
               for co in range(self.out_channels)]
        for ci in range(self.in_channels):
            co0 = (ci // self.cg) * self.og
            for co in range(self.og):
                w = self.weights[ci][co]
                out_c = out[co0 + co]
                for oh in range(H):
                    xrow = x[ci][oh]
                    base_h = oh * self.sh - self.ph
                    for ow in range(W):
                        v = xrow[ow]
                        base_w = ow * self.sw - self.pw
                        for kh in range(self.kh):
                            ih = base_h + kh
                            if ih < 0 or ih >= ho:
                                continue
                            wrow = w[kh]
                            dst = out_c[ih]
                            for kw in range(self.kw):
                                iw = base_w + kw
                                if 0 <= iw < wo:
                                    dst[iw] += v * wrow[kw]
        return out

    def backward(self, dout):
        """dout: (Cout, Ho, Wo) -> dx: (Cin, H, W)，内部累积 grad_w / grad_b"""
        if self.x_shape is None:
            raise RuntimeError("ConvTranspose2d.backward: 请先调用 forward")
        C, H, W = self.x_shape
        ho, wo = self.h_out, self.w_out

        dx = [[[0.0] * W for _ in range(H)] for _ in range(C)]
        for ci in range(C):
            co0 = (ci // self.cg) * self.og
            for co in range(self.og):
                out_c = dout[co0 + co]
                w = self.weights[ci][co]
                g_w = self.grad_w[ci][co]
                for oh in range(H):
                    xrow = self.x[ci][oh]
                    dxrow = dx[ci][oh]
                    base_h = oh * self.sh - self.ph
                    for ow in range(W):
                        base_w = ow * self.sw - self.pw
                        xv = xrow[ow]
                        acc = 0.0
                        for kh in range(self.kh):
                            ih = base_h + kh
                            if ih < 0 or ih >= ho:
                                continue
                            drow = out_c[ih]
                            g_row = g_w[kh]
                            for kw in range(self.kw):
                                iw = base_w + kw
                                if iw < 0 or iw >= wo:
                                    continue
                                dv = drow[iw]
                                g_row[kw] += dv * xv
                                acc += dv * w[kh][kw]
                        dxrow[ow] += acc
        # 偏置梯度每次反向只加一遍：同一组的输入通道共享同一张输出图，
        # 原来写在 ci 循环里会按 cin/groups 倍放大（实测 cin=2 时 2 倍、cin=3 时 3 倍）
        for _co0 in range(0, self.out_channels, self.og):
            for _co in range(self.og):
                self.grad_b[_co0 + _co] += sum(sum(row) for row in dout[_co0 + _co])
        return dx


class MaxPool2d:
    """2D 最大池化层"""

    def __init__(self, kernel_size, stride=None):
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride = self.kernel_size if stride is None else (
            (stride, stride) if isinstance(stride, int) else stride
        )
        self.kh, self.kw = self.kernel_size
        self.sh, self.sw = self.stride
        # Conv2d 有同样的校验，MaxPool2d 原先没有：kernel=0 会返回全 -inf（内层循环不执行），
        # stride=0 会除零
        if self.kh <= 0 or self.kw <= 0:
            raise ValueError(f"MaxPool2d: kernel_size 必须为正，收到 {self.kernel_size}")
        if self.sh <= 0 or self.sw <= 0:
            raise ValueError(f"MaxPool2d: stride 必须为正，收到 {self.stride}")
        self.mask = None
        self.x_shape = None

    def forward(self, x):
        """x: (C, H, W) -> (C, Ho, Wo)"""
        C, H, W = len(x), len(x[0]), len(x[0][0])
        self.x_shape = (C, H, W)
        ho = (H - self.kh) // self.sh + 1
        wo = (W - self.kw) // self.sw + 1
        if ho <= 0 or wo <= 0:
            raise ValueError(
                f"MaxPool2d: 输入 {H}x{W} 配 kernel {self.kh}x{self.kw}、"
                f"stride {self.sh}/{self.sw} 得到输出 {ho}x{wo}，必须都 > 0"
            )
        self.mask = [[[(0, 0)] * wo for _ in range(ho)] for _ in range(C)]

        out = [[[0.0] * wo for _ in range(ho)] for _ in range(C)]
        for c in range(C):
            for oh in range(ho):
                for ow in range(wo):
                    max_val = -float("inf")
                    max_pos = (0, 0)
                    for kh in range(self.kh):
                        for kw in range(self.kw):
                            ih = oh * self.sh + kh
                            iw = ow * self.sw + kw
                            if x[c][ih][iw] > max_val:
                                max_val = x[c][ih][iw]
                                max_pos = (kh, kw)
                    out[c][oh][ow] = max_val
                    self.mask[c][oh][ow] = max_pos
        return out

    def backward(self, dout):
        """dout: (C, Ho, Wo) -> dx: (C, H, W)"""
        C, H, W = self.x_shape
        ho = len(dout[0])
        wo = len(dout[0][0])
        dx = [[[0.0] * W for _ in range(H)] for _ in range(C)]
        for c in range(C):
            for oh in range(ho):
                for ow in range(wo):
                    kh, kw = self.mask[c][oh][ow]
                    ih = oh * self.sh + kh
                    iw = ow * self.sw + kw
                    dx[c][ih][iw] += dout[c][oh][ow]
        return dx


class AvgPool2d:
    """2D 平均池化层（接口与 MaxPool2d 一致：forward 收 (C,H,W)，backward 收 dout）"""

    def __init__(self, kernel_size, stride=None):
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else tuple(kernel_size)
        self.stride = self.kernel_size if stride is None else (
            (stride, stride) if isinstance(stride, int) else tuple(stride)
        )
        self.kh, self.kw = self.kernel_size
        self.sh, self.sw = self.stride
        if self.kh <= 0 or self.kw <= 0:
            raise ValueError(f"AvgPool2d: kernel_size 必须为正，收到 {self.kernel_size}")
        if self.sh <= 0 or self.sw <= 0:
            raise ValueError(f"AvgPool2d: stride 必须为正，收到 {self.stride}")
        self.x_shape = None

    def forward(self, x):
        """x: (C, H, W) -> (C, Ho, Wo)，每个窗口取均值"""
        C, H, W = len(x), len(x[0]), len(x[0][0])
        self.x_shape = (C, H, W)
        ho = (H - self.kh) // self.sh + 1
        wo = (W - self.kw) // self.sw + 1
        if ho <= 0 or wo <= 0:
            raise ValueError(
                f"AvgPool2d: 输入 {H}x{W} 配 kernel {self.kh}x{self.kw}、"
                f"stride {self.sh}/{self.sw} 得到输出 {ho}x{wo}，必须都 > 0"
            )
        area = float(self.kh * self.kw)
        out = [[[0.0] * wo for _ in range(ho)] for _ in range(C)]
        for c in range(C):
            for oh in range(ho):
                for ow in range(wo):
                    total = 0.0
                    for kh in range(self.kh):
                        for kw in range(self.kw):
                            total += x[c][oh * self.sh + kh][ow * self.sw + kw]
                    out[c][oh][ow] = total / area
        return out

    def backward(self, dout):
        """dout: (C, Ho, Wo) -> dx: (C, H, W)，每个窗口均分到所有输入位置"""
        C, H, W = self.x_shape
        area = float(self.kh * self.kw)
        dx = [[[0.0] * W for _ in range(H)] for _ in range(C)]
        for c in range(C):
            for oh in range(len(dout[0])):
                for ow in range(len(dout[0][0])):
                    g = dout[c][oh][ow] / area
                    for kh in range(self.kh):
                        for kw in range(self.kw):
                            dx[c][oh * self.sh + kh][ow * self.sw + kw] += g
        return dx


class GlobalAvgPool2d:
    """全局平均池化：(C, H, W) -> (C,)，等价于 kernel 覆盖整张特征图"""

    def __init__(self):
        self.x_shape = None

    def forward(self, x):
        C, H, W = len(x), len(x[0]), len(x[0][0])
        self.x_shape = (C, H, W)
        area = float(H * W)
        return [sum(sum(row) for row in x[c]) / area for c in range(C)]

    def backward(self, dout):
        C, H, W = self.x_shape
        area = float(H * W)
        return [[[dout[c] / area] * W for _ in range(H)] for c in range(C)]


class AdaptiveAvgPool2d:
    """自适应平均池化：(C, H, W) -> (C, out_h, out_w)，输出尺寸由参数指定。

    分箱规则与 PyTorch 一致：第 oh 个输出格子覆盖输入行区间
    [floor(oh*H/out_h), ceil((oh+1)*H/out_h))，列同理。所以当 H 不能被 out_h
    整除时各格大小不同（相邻格会重叠 1 行），每格按自己的实际面积取均值。
    对任意 H、W 都能把输出固定成 out_size，方便接在任意尺寸的卷积后面。

    `out_size=(1, 1)` 与 `GlobalAvgPool2d` 等价（数值逐位相同）。
    """

    def __init__(self, out_size):
        if isinstance(out_size, int):
            out_size = (out_size, out_size)
        out_size = tuple(out_size)
        if len(out_size) != 2:
            raise ValueError(f"AdaptiveAvgPool2d: out_size 必须是 1 个或 2 个整数，收到 {out_size}")
        oh, ow = out_size
        for name, v in (("out_h", oh), ("out_w", ow)):
            if isinstance(v, bool) or not isinstance(v, int):
                raise ValueError(f"AdaptiveAvgPool2d: {name} 必须是整数，收到 {v!r}")
            if v <= 0:
                raise ValueError(f"AdaptiveAvgPool2d: {name} 必须为正，收到 {v}")
        self.out_size = (oh, ow)
        self.x_shape = None
        self._bins = None      # ((rows, cols), ...) 每个输出格子的输入区间

    @staticmethod
    def _bin_range(size, n_out, i):
        start = (i * size) // n_out
        end = -(-((i + 1) * size) // n_out)     # ceil
        return start, end

    def forward(self, x):
        """x: (C, H, W) -> (C, out_h, out_w)"""
        if not x or not x[0] or not x[0][0]:
            raise ValueError("AdaptiveAvgPool2d: 输入不能为空")
        C, H, W = len(x), len(x[0]), len(x[0][0])
        for c in range(C):
            if len(x[c]) != H:
                raise ValueError(f"AdaptiveAvgPool2d: 第 {c} 个通道高度不一致")
            for row in x[c]:
                if len(row) != W:
                    raise ValueError(f"AdaptiveAvgPool2d: 第 {c} 个通道宽度不一致")
        oh, ow = self.out_size
        self.x_shape = (C, H, W)
        row_bins = [self._bin_range(H, oh, i) for i in range(oh)]
        col_bins = [self._bin_range(W, ow, i) for i in range(ow)]
        self._bins = (row_bins, col_bins)
        out = [[[0.0] * ow for _ in range(oh)] for _ in range(C)]
        for c in range(C):
            for i, (h0, h1) in enumerate(row_bins):
                for j, (w0, w1) in enumerate(col_bins):
                    area = float((h1 - h0) * (w1 - w0))
                    # 行内用 sum()（CPython 的补偿求和），再对行求和 —— 与
                    # GlobalAvgPool2d 的 sum(sum(row) for row in x[c]) 完全同序，
                    # 因此 out_size=(1, 1) 时两者逐位相同
                    total = sum(sum(x[c][h][w0:w1]) for h in range(h0, h1))
                    out[c][i][j] = total / area
        return out

    def backward(self, dout):
        """dout: (C, out_h, out_w) -> dx: (C, H, W)，每格把梯度按面积均分回输入位置"""
        if self.x_shape is None or self._bins is None:
            raise RuntimeError("AdaptiveAvgPool2d.backward: 必须先调用 forward")
        C, H, W = self.x_shape
        row_bins, col_bins = self._bins
        dx = [[[0.0] * W for _ in range(H)] for _ in range(C)]
        for c in range(C):
            for i, (h0, h1) in enumerate(row_bins):
                for j, (w0, w1) in enumerate(col_bins):
                    area = float((h1 - h0) * (w1 - w0))
                    g = dout[c][i][j] / area
                    for h in range(h0, h1):
                        row = dx[c][h]
                        for w in range(w0, w1):
                            row[w] += g
        return dx


def _w2_channel_means(x):
    """(C, H, W) -> 每个通道的均值列表"""
    C, H, W = len(x), len(x[0]), len(x[0][0])
    area = float(H * W)
    return [sum(sum(row) for row in x[c]) / area for c in range(C)]


def _w2_count(params):
    """统计 (name, 参数) 序列里的元素个数（支持二维/一维）"""
    total = 0
    for _, v in params:
        if v and isinstance(v[0], list):
            for row in v:
                total += len(row)
        else:
            total += len(v)
    return total


class GlobalMaxPool2d:
    """全局最大池化：(C, H, W) -> (C,)，反向只把梯度给最大值那个位置"""

    def __init__(self):
        self.x_shape = None
        self._arg = None

    def forward(self, x):
        if not x or not x[0] or not x[0][0]:
            raise ValueError("GlobalMaxPool2d: 输入不能为空")
        C, H, W = len(x), len(x[0]), len(x[0][0])
        self.x_shape = (C, H, W)
        out, arg = [], []
        for c in range(C):
            best, where = None, (0, 0)
            for h in range(H):
                row = x[c][h]
                for w in range(W):
                    if best is None or row[w] > best:
                        best, where = row[w], (h, w)
            out.append(best)
            arg.append(where)
        self._arg = arg
        return out

    def backward(self, dout):
        if self.x_shape is None:
            raise RuntimeError("GlobalMaxPool2d.backward: 必须先调用 forward")
        C, H, W = self.x_shape
        dx = [[[0.0] * W for _ in range(H)] for _ in range(C)]
        for c in range(C):
            h, w = self._arg[c]
            dx[c][h][w] = dout[c]
        return dx


class AdaptiveMaxPool2d:
    """自适应最大池化：(C, H, W) -> (C, out_h, out_w)，分箱规则与 AdaptiveAvgPool2d 一致"""

    def __init__(self, out_size):
        if isinstance(out_size, int):
            out_size = (out_size, out_size)
        out_size = tuple(out_size)
        if len(out_size) != 2:
            raise ValueError(f"AdaptiveMaxPool2d: out_size 必须是 1 个或 2 个整数，收到 {out_size}")
        oh, ow = out_size
        for n, v in (("out_h", oh), ("out_w", ow)):
            if isinstance(v, bool) or not isinstance(v, int):
                raise ValueError(f"AdaptiveMaxPool2d: {n} 必须是整数，收到 {v!r}")
            if v <= 0:
                raise ValueError(f"AdaptiveMaxPool2d: {n} 必须为正，收到 {v}")
        self.out_size = (oh, ow)
        self.x_shape = None
        self._arg = None

    _bin_range = staticmethod(AdaptiveAvgPool2d._bin_range)

    def forward(self, x):
        if not x or not x[0] or not x[0][0]:
            raise ValueError("AdaptiveMaxPool2d: 输入不能为空")
        C, H, W = len(x), len(x[0]), len(x[0][0])
        oh, ow = self.out_size
        self.x_shape = (C, H, W)
        row_bins = [self._bin_range(H, oh, i) for i in range(oh)]
        col_bins = [self._bin_range(W, ow, i) for i in range(ow)]
        out = [[[0.0] * ow for _ in range(oh)] for _ in range(C)]
        arg = [[[None] * ow for _ in range(oh)] for _ in range(C)]
        for c in range(C):
            for i, (h0, h1) in enumerate(row_bins):
                for j, (w0, w1) in enumerate(col_bins):
                    best, where = None, None
                    for h in range(h0, h1):
                        row = x[c][h]
                        for w in range(w0, w1):
                            if best is None or row[w] > best:
                                best, where = row[w], (h, w)
                    out[c][i][j] = best
                    arg[c][i][j] = where
        self._arg = arg
        return out

    def backward(self, dout):
        if self.x_shape is None:
            raise RuntimeError("AdaptiveMaxPool2d.backward: 必须先调用 forward")
        C, H, W = self.x_shape
        oh, ow = self.out_size
        dx = [[[0.0] * W for _ in range(H)] for _ in range(C)]
        for c in range(C):
            for i in range(oh):
                for j in range(ow):
                    h, w = self._arg[c][i][j]
                    dx[c][h][w] += dout[c][i][j]
        return dx


class LPPool2d:
    """Lp 池化：(Σ|x|^p / area)^(1/p)；p=1 为「平均绝对值」，p 很大时逼近最大池化"""

    def __init__(self, norm_type, kernel_size, stride=None):
        if isinstance(norm_type, bool) or not isinstance(norm_type, (int, float)):
            raise ValueError(f"LPPool2d: norm_type 必须是正数，收到 {norm_type!r}")
        v = float(norm_type)
        if not (v > 0) or v != v or v == float("inf"):
            raise ValueError(f"LPPool2d: norm_type 必须是有限正数，收到 {norm_type!r}")
        self.norm_type = v
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        for n, k in (("kernel_size", self.kernel_size), ("stride", self.stride)):
            if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
                raise ValueError(f"LPPool2d: {n} 必须是正整数，收到 {k!r}")
        self.x_shape = None
        self._out_shape = None

    def forward(self, x):
        if not x or not x[0] or not x[0][0]:
            raise ValueError("LPPool2d: 输入不能为空")
        C, H, W = len(x), len(x[0]), len(x[0][0])
        k, st, pw = self.kernel_size, self.stride, self.norm_type
        oh = (H - k) // st + 1
        ow = (W - k) // st + 1
        if oh <= 0 or ow <= 0:
            raise ValueError(f"LPPool2d: 输入 {H}x{W} 容不下 kernel {k}")
        self.x_shape = (C, H, W)
        self._out_shape = (oh, ow)
        self._x = [[[x[c][h][w] for w in range(W)] for h in range(H)] for c in range(C)]
        inv_p = 1.0 / pw
        out = [[[0.0] * ow for _ in range(oh)] for _ in range(C)]
        for c in range(C):
            for i in range(oh):
                for j in range(ow):
                    acc = 0.0
                    for h in range(i * st, i * st + k):
                        row = x[c][h]
                        for w in range(j * st, j * st + k):
                            acc += abs(row[w]) ** pw
                    out[c][i][j] = (acc / float(k * k)) ** inv_p
        return out

    def backward(self, dout):
        if self.x_shape is None:
            raise RuntimeError("LPPool2d.backward: 必须先调用 forward")
        C, H, W = self.x_shape
        oh, ow = self._out_shape
        x = self._x
        k, st, pw = self.kernel_size, self.stride, self.norm_type
        area = float(k * k)
        dx = [[[0.0] * W for _ in range(H)] for _ in range(C)]
        for c in range(C):
            for i in range(oh):
                for j in range(ow):
                    acc = 0.0
                    for h in range(i * st, i * st + k):
                        row = x[c][h]
                        for w in range(j * st, j * st + k):
                            acc += abs(row[w]) ** pw
                    y = (acc / area) ** (1.0 / pw)
                    if y == 0.0:
                        continue
                    coef = dout[c][i][j] / (area * (y ** (pw - 1.0)))
                    for h in range(i * st, i * st + k):
                        row = x[c][h]
                        drow = dx[c][h]
                        for w in range(j * st, j * st + k):
                            v = row[w]
                            drow[w] += coef * (abs(v) ** (pw - 1.0)) * (1.0 if v >= 0 else -1.0)
        return dx


class SEBlock:
    """Squeeze-and-Excitation：GAP -> 瓶颈 MLP -> sigmoid 得到通道权重，逐通道乘回输入

    `forward((C,H,W)) -> (C,H,W)`；`backward(dout) -> dx`，四个参数的梯度留在
    `self.grads`（`W1/b1/W2/b2`），`sgd_step`/`adam_step` 只更新这四个。
    """

    def __init__(self, channels, reduction=16):
        for n, v in (("channels", channels), ("reduction", reduction)):
            if isinstance(v, bool) or not isinstance(v, int) or v <= 0:
                raise ValueError(f"SEBlock: {n} 必须是正整数，收到 {v!r}")
        self.channels = channels
        self.reduction = reduction
        self.hidden = max(1, channels // reduction)
        std1 = (2.0 / channels) ** 0.5
        std2 = (2.0 / self.hidden) ** 0.5
        self.W1 = [[random.gauss(0.0, std1) for _ in range(channels)] for _ in range(self.hidden)]
        self.b1 = [0.0] * self.hidden
        self.W2 = [[random.gauss(0.0, std2) for _ in range(self.hidden)] for _ in range(channels)]
        self.b2 = [0.0] * channels
        self.training = True
        self._cache = None
        self._init_adam()
        self.zero_grad()

    def _params(self):
        return (("W1", self.W1), ("b1", self.b1), ("W2", self.W2), ("b2", self.b2))

    def _init_adam(self):
        self._adam_m = {k: _zeros_like(v) for k, v in self._params()}
        self._adam_v = {k: _zeros_like(v) for k, v in self._params()}
        self._adam_t = 0

    def zero_grad(self):
        self.grads = {k: _zeros_like(v) for k, v in self._params()}

    def param_count(self):
        return _w2_count(self._params())

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def forward(self, x):
        if not x or not x[0] or not x[0][0]:
            raise ValueError("SEBlock: 输入不能为空")
        C, H, W = len(x), len(x[0]), len(x[0][0])
        if C != self.channels:
            raise ValueError(f"SEBlock: 输入通道数 {C} 与 channels={self.channels} 不符")
        z = _w2_channel_means(x)
        a1 = [sum(map(_mul, self.W1[k], z)) + self.b1[k] for k in range(self.hidden)]
        r1 = [v if v > 0.0 else 0.0 for v in a1]
        a2 = [sum(map(_mul, self.W2[c], r1)) + self.b2[c] for c in range(C)]
        s = [_sigmoid_scalar(v) for v in a2]
        out = [[[x[c][h][w] * s[c] for w in range(W)] for h in range(H)] for c in range(C)]
        self._cache = (x, s, r1, a1)
        return out

    def backward(self, dout):
        if self._cache is None:
            raise RuntimeError("SEBlock.backward: 必须先调用 forward")
        x, s, r1, a1 = self._cache
        C, H, W = len(x), len(x[0]), len(x[0][0])
        if (len(dout) != C or len(dout[0]) != H or len(dout[0][0]) != W):
            raise ValueError(
                f"SEBlock.backward: dout 形状 {(len(dout), len(dout[0]), len(dout[0][0]))} "
                f"与输入 {(C, H, W)} 不符")
        area = float(H * W)
        da2 = [0.0] * C
        for c in range(C):
            inner = 0.0
            for h in range(H):
                for w in range(W):
                    inner += dout[c][h][w] * x[c][h][w]
            da2[c] = inner * s[c] * (1.0 - s[c])
            for k in range(self.hidden):
                self.grads["W2"][c][k] += da2[c] * r1[k]
            self.grads["b2"][c] += da2[c]
        dz = [0.0] * C
        for k in range(self.hidden):
            dr1 = sum(da2[c] * self.W2[c][k] for c in range(C))
            da1 = dr1 if a1[k] > 0.0 else 0.0
            for c in range(C):
                self.grads["W1"][k][c] += da1 * _w2_channel_means(x)[c]
            self.grads["b1"][k] += da1
            for c in range(C):
                dz[c] += da1 * self.W1[k][c]
        dx = [[[0.0] * W for _ in range(H)] for _ in range(C)]
        for c in range(C):
            g = dz[c] / area
            for h in range(H):
                for w in range(W):
                    dx[c][h][w] = dout[c][h][w] * s[c] + g
        return dx
class ECABlock:
    """ECA 通道注意力：GAP 后用一条 `kernel_size` 宽的 1D 卷积（零填充）跨通道混权重，

    再 sigmoid 乘回输入。比 SE 省掉降维 MLP，只多 `kernel_size` 个参数。
    """

    def __init__(self, channels, kernel_size=3):
        for n, v in (("channels", channels), ("kernel_size", kernel_size)):
            if isinstance(v, bool) or not isinstance(v, int) or v <= 0:
                raise ValueError(f"ECABlock: {n} 必须是正整数，收到 {v!r}")
        if kernel_size % 2 == 0:
            raise ValueError(f"ECABlock: kernel_size 必须是奇数（保证两侧对称填充），收到 {kernel_size}")
        self.channels = channels
        self.kernel_size = kernel_size
        self.half = kernel_size // 2
        self.w = [0.0] * kernel_size
        for i in range(kernel_size):
            self.w[i] = 1.0 / kernel_size
        self.training = True
        self.grads = {"w": [0.0] * kernel_size}
        self._cache = None
        self._adam_m = {"w": [0.0] * kernel_size}
        self._adam_v = {"w": [0.0] * kernel_size}
        self._adam_t = 0

    def zero_grad(self):
        self.grads = {"w": [0.0] * self.kernel_size}

    def param_count(self):
        return self.kernel_size

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def forward(self, x):
        if not x or not x[0] or not x[0][0]:
            raise ValueError("ECABlock: 输入不能为空")
        C, H, W = len(x), len(x[0]), len(x[0][0])
        if C != self.channels:
            raise ValueError(f"ECABlock: 输入通道数 {C} 与 channels={self.channels} 不符")
        z = _w2_channel_means(x)
        e = []
        for c in range(C):
            acc = 0.0
            for k in range(self.kernel_size):
                src = c + k - self.half
                if 0 <= src < C:
                    acc += self.w[k] * z[src]
            e.append(acc)
        s = [_sigmoid_scalar(v) for v in e]
        out = [[[x[c][h][w] * s[c] for w in range(W)] for h in range(H)] for c in range(C)]
        self._cache = (x, z, s)
        return out

    def backward(self, dout):
        if self._cache is None:
            raise RuntimeError("ECABlock.backward: 必须先调用 forward")
        x, z, s = self._cache
        C, H, W = len(x), len(x[0]), len(x[0][0])
        if len(dout) != C or len(dout[0]) != H or len(dout[0][0]) != W:
            raise ValueError(
                f"ECABlock.backward: dout 形状 {(len(dout), len(dout[0]), len(dout[0][0]))} "
                f"与输入 {(C, H, W)} 不符")
        area = float(H * W)
        de = [0.0] * C
        for c in range(C):
            inner = 0.0
            for h in range(H):
                for w in range(W):
                    inner += dout[c][h][w] * x[c][h][w]
            de[c] = inner * s[c] * (1.0 - s[c])
        for k in range(self.kernel_size):
            acc = 0.0
            for c in range(C):
                src = c + k - self.half
                if 0 <= src < C:
                    acc += de[c] * z[src]
            self.grads["w"][k] += acc
        dz = [0.0] * C
        for k in range(self.kernel_size):
            for c in range(C):
                src = c + k - self.half
                if 0 <= src < C:
                    dz[src] += de[c] * self.w[k]
        dx = [[[0.0] * W for _ in range(H)] for _ in range(C)]
        for c in range(C):
            g = dz[c] / area
            for h in range(H):
                for w in range(W):
                    dx[c][h][w] = dout[c][h][w] * s[c] + g
        return dx


def _conv_param_count(layer):
    """Conv2d / Conv1d / ConvTranspose2d 的参数量（权重 + 偏置）"""
    w = layer.weights

    def count(node):
        if isinstance(node, list):
            return sum(count(v) for v in node)
        return 1

    return count(w) + len(layer.biases)


def _conv_sgd(layer, lr):
    """原地 SGD 更新一个卷积层的 weights/biases"""
    W, B, gw, gb = layer.weights, layer.biases, layer.grad_w, layer.grad_b
    oc = len(W)
    for o in range(oc):
        B[o] -= lr * gb[o]
        for ic in range(len(W[o])):
            row_w, row_g = W[o][ic], gw[o][ic]
            if isinstance(row_w, list) and row_w and isinstance(row_w[0], list):
                for a in range(len(row_w)):
                    for b in range(len(row_w[a])):
                        row_w[a][b] -= lr * row_g[a][b]
            else:
                for k in range(len(row_w)):
                    row_w[k] -= lr * row_g[k]


def _conv_adam(layer, lr, beta1=0.9, beta2=0.999, eps=1e-8):
    """原地 Adam 更新一个卷积层；步数记在 layer._conv_adam_t 上"""
    t = getattr(layer, "_conv_adam_t", 0) + 1
    layer._conv_adam_t = t
    c1 = 1.0 - beta1 ** t
    c2 = 1.0 - beta2 ** t
    o1, o2 = 1.0 - beta1, 1.0 - beta2
    W, B, gw, gb = layer.weights, layer.biases, layer.grad_w, layer.grad_b
    mw, vw = layer.m_w, layer.v_w
    mb, vb = layer.m_b, layer.v_b
    for o in range(len(W)):
        g = gb[o]
        m = beta1 * mb[o] + o1 * g
        v = beta2 * vb[o] + o2 * g * g
        mb[o], vb[o] = m, v
        B[o] -= lr * (m / c1) / (math.sqrt(v / c2) + eps)
        for ic in range(len(W[o])):
            row_w, row_g = W[o][ic], gw[o][ic]
            row_m, row_v = mw[o][ic], vw[o][ic]
            if isinstance(row_w, list) and row_w and isinstance(row_w[0], list):
                for a in range(len(row_w)):
                    for b in range(len(row_w[a])):
                        g = row_g[a][b]
                        m = beta1 * row_m[a][b] + o1 * g
                        v = beta2 * row_v[a][b] + o2 * g * g
                        row_m[a][b], row_v[a][b] = m, v
                        row_w[a][b] -= lr * (m / c1) / (math.sqrt(v / c2) + eps)
            else:
                for k in range(len(row_w)):
                    g = row_g[k]
                    m = beta1 * row_m[k] + o1 * g
                    v = beta2 * row_v[k] + o2 * g * g
                    row_m[k], row_v[k] = m, v
                    row_w[k] -= lr * (m / c1) / (math.sqrt(v / c2) + eps)


class DepthwiseSeparableConv2d:
    """深度可分离卷积：`depthwise Conv2d(groups=in_channels)` + `pointwise 1x1 Conv2d`

    参数量约为标准卷积的 `1/out_channels + 1/(k*k)`；两个子卷积各自维护
    `grad_w`/`grad_b`，本层的 `zero_grad`/`sgd_step`/`adam_step` 转发给它们。
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dw = Conv2d(in_channels, in_channels, kernel_size,
                         stride=stride, padding=padding, groups=in_channels)
        self.pw = Conv2d(in_channels, out_channels, 1, stride=1, padding=0)
        self.training = True

    @property
    def grads(self):
        return self.grads_dict()

    def grads_dict(self):
        return {"dw_w": self.dw.grad_w, "dw_b": self.dw.grad_b,
                "pw_w": self.pw.grad_w, "pw_b": self.pw.grad_b}

    def zero_grad(self):
        self.dw.zero_grad()
        self.pw.zero_grad()

    def param_count(self):
        return _conv_param_count(self.dw) + _conv_param_count(self.pw)

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def sgd_step(self, lr):
        _conv_sgd(self.dw, lr)
        _conv_sgd(self.pw, lr)

    def adam_step(self, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        _conv_adam(self.dw, lr, beta1, beta2, eps)
        _conv_adam(self.pw, lr, beta1, beta2, eps)

    def forward(self, x):
        return self.pw.forward(self.dw.forward(x))

    def backward(self, dout):
        return self.dw.backward(self.pw.backward(dout))


class BasicBlock2d:
    """ResNet BasicBlock：`conv3x3 -> BN -> ReLU -> conv3x3 -> BN -> (+x) -> ReLU`

    只支持 stride=1 的恒等捷径（通道与尺寸不变）——stride>1 时需要 1x1 投影，
    那个留给 Bottleneck。BN 的 gamma/beta 梯度收在 `self.bn_grads` 里。
    """

    def __init__(self, channels, kernel_size=3, use_bn=True):
        for n, v in (("channels", channels), ("kernel_size", kernel_size)):
            if isinstance(v, bool) or not isinstance(v, int) or v <= 0:
                raise ValueError(f"BasicBlock2d: {n} 必须是正整数，收到 {v!r}")
        self.channels = channels
        self.kernel_size = kernel_size
        self.use_bn = bool(use_bn)
        pad = kernel_size // 2
        self.conv1 = Conv2d(channels, channels, kernel_size, stride=1, padding=pad)
        self.conv2 = Conv2d(channels, channels, kernel_size, stride=1, padding=pad)
        self.bn1 = BatchNorm2d(channels) if self.use_bn else None
        self.bn2 = BatchNorm2d(channels) if self.use_bn else None
        self.training = True
        self.bn_grads = {"bn1_gamma": [0.0] * channels, "bn1_beta": [0.0] * channels,
                         "bn2_gamma": [0.0] * channels, "bn2_beta": [0.0] * channels}
        self._cache = None

    def zero_grad(self):
        self.conv1.zero_grad()
        self.conv2.zero_grad()
        for k in self.bn_grads:
            self.bn_grads[k] = [0.0] * self.channels

    def param_count(self):
        total = _conv_param_count(self.conv1) + _conv_param_count(self.conv2)
        if self.use_bn:
            total += 4 * self.channels
        return total

    def train(self):
        self.training = True
        if self.bn1 is not None:
            self.bn1.train()
            self.bn2.train()

    def eval(self):
        self.training = False
        if self.bn1 is not None:
            self.bn1.eval()
            self.bn2.eval()

    def sgd_step(self, lr):
        _conv_sgd(self.conv1, lr)
        _conv_sgd(self.conv2, lr)

    def adam_step(self, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        _conv_adam(self.conv1, lr, beta1, beta2, eps)
        _conv_adam(self.conv2, lr, beta1, beta2, eps)

    def forward(self, x):
        if not x or not x[0] or not x[0][0]:
            raise ValueError("BasicBlock2d: 输入不能为空")
        C, H, W = len(x), len(x[0]), len(x[0][0])
        if C != self.channels:
            raise ValueError(f"BasicBlock2d: 输入通道数 {C} 与 channels={self.channels} 不符")
        h1 = self.conv1.forward(x)
        a1 = self.bn1.forward(h1) if self.use_bn else h1
        r1 = [[[v if v > 0.0 else 0.0 for v in row] for row in ch] for ch in a1]
        h2 = self.conv2.forward(r1)
        a2 = self.bn2.forward(h2) if self.use_bn else h2
        pre = [[[a2[c][h][w] + x[c][h][w] for w in range(W)] for h in range(H)] for c in range(C)]
        out = [[[v if v > 0.0 else 0.0 for v in row] for row in ch] for ch in pre]
        self._cache = (x, a1, r1, pre)
        return out

    def backward(self, dout):
        if self._cache is None:
            raise RuntimeError("BasicBlock2d.backward: 必须先调用 forward")
        x, a1, r1, pre = self._cache
        C, H, W = len(x), len(x[0]), len(x[0][0])
        if len(dout) != C or len(dout[0]) != H or len(dout[0][0]) != W:
            raise ValueError(
                f"BasicBlock2d.backward: dout 形状 {(len(dout), len(dout[0]), len(dout[0][0]))} "
                f"与输入 {(C, H, W)} 不符")
        dpre = [[[dout[c][h][w] if pre[c][h][w] > 0.0 else 0.0 for w in range(W)]
                 for h in range(H)] for c in range(C)]
        if self.use_bn:
            da2, g2, b2 = self.bn2.backward(dpre)
            for c in range(C):
                self.bn_grads["bn2_gamma"][c] += g2[c]
                self.bn_grads["bn2_beta"][c] += b2[c]
        else:
            da2 = dpre
        dh2 = self.conv2.backward(da2)
        dr1 = [[[dh2[c][h][w] if a1[c][h][w] > 0.0 else 0.0 for w in range(W)]
                for h in range(H)] for c in range(C)]
        if self.use_bn:
            da1, g1, b1 = self.bn1.backward(dr1)
            for c in range(C):
                self.bn_grads["bn1_gamma"][c] += g1[c]
                self.bn_grads["bn1_beta"][c] += b1[c]
        else:
            da1 = dr1
        dh1 = self.conv1.backward(da1)
        return [[[dh1[c][h][w] + dpre[c][h][w] for w in range(W)] for h in range(H)]
                for c in range(C)]
class Bottleneck2d:
    """ResNet Bottleneck：`1x1 降维 -> 3x3 -> 1x1 升维` + 残差

    `mid = max(1, out_channels // 4)`；捷径在 `stride=1 且 in==out` 时是恒等，
    否则用 `1x1 Conv2d + BatchNorm2d` 投影。三个 BN 的 gamma/beta 梯度收在
    `self.bn_grads`（键 `bn1_*`/`bn2_*`/`bn3_*`/`short_*`，没有捷径 BN 时没有 `short_*`）。
    """

    def __init__(self, in_channels, out_channels, stride=1):
        for n, v in (("in_channels", in_channels), ("out_channels", out_channels),
                     ("stride", stride)):
            if isinstance(v, bool) or not isinstance(v, int) or v <= 0:
                raise ValueError(f"Bottleneck2d: {n} 必须是正整数，收到 {v!r}")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.mid = max(1, out_channels // 4)
        self.conv1 = Conv2d(in_channels, self.mid, 1, stride=1, padding=0)
        self.bn1 = BatchNorm2d(self.mid)
        self.conv2 = Conv2d(self.mid, self.mid, 3, stride=stride, padding=1)
        self.bn2 = BatchNorm2d(self.mid)
        self.conv3 = Conv2d(self.mid, out_channels, 1, stride=1, padding=0)
        self.bn3 = BatchNorm2d(out_channels)
        self.proj = None
        self.short_bn = None
        if not (stride == 1 and in_channels == out_channels):
            self.proj = Conv2d(in_channels, out_channels, 1, stride=stride, padding=0)
            self.short_bn = BatchNorm2d(out_channels)
        self.training = True
        keys = ["bn1_gamma", "bn1_beta", "bn2_gamma", "bn2_beta", "bn3_gamma", "bn3_beta"]
        if self.short_bn is not None:
            keys += ["short_gamma", "short_beta"]
        self.bn_grads = {k: [0.0] * (self.mid if "bn1" in k or "bn2" in k else out_channels)
                         for k in keys}
        self._cache = None

    def zero_grad(self):
        self.conv1.zero_grad()
        self.conv2.zero_grad()
        self.conv3.zero_grad()
        if self.proj is not None:
            self.proj.zero_grad()
        for k in self.bn_grads:
            self.bn_grads[k] = [0.0] * len(self.bn_grads[k])

    def param_count(self):
        total = (_conv_param_count(self.conv1) + _conv_param_count(self.conv2)
                 + _conv_param_count(self.conv3))
        total += 4 * self.mid + 2 * self.out_channels
        if self.proj is not None:
            total += _conv_param_count(self.proj) + 2 * self.out_channels
        return total

    def train(self):
        self.training = True
        for bn in (self.bn1, self.bn2, self.bn3, self.short_bn):
            if bn is not None:
                bn.train()

    def eval(self):
        self.training = False
        for bn in (self.bn1, self.bn2, self.bn3, self.short_bn):
            if bn is not None:
                bn.eval()

    def sgd_step(self, lr):
        for c in (self.conv1, self.conv2, self.conv3, self.proj):
            if c is not None:
                _conv_sgd(c, lr)

    def adam_step(self, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        for c in (self.conv1, self.conv2, self.conv3, self.proj):
            if c is not None:
                _conv_adam(c, lr, beta1, beta2, eps)

    @staticmethod
    def _relu(x):
        return [[[v if v > 0.0 else 0.0 for v in row] for row in ch] for ch in x]

    def forward(self, x):
        if not x or not x[0] or not x[0][0]:
            raise ValueError("Bottleneck2d: 输入不能为空")
        C = len(x)
        if C != self.in_channels:
            raise ValueError(
                f"Bottleneck2d: 输入通道数 {C} 与 in_channels={self.in_channels} 不符")
        h1 = self.bn1.forward(self.conv1.forward(x))
        r1 = self._relu(h1)
        h2 = self.bn2.forward(self.conv2.forward(r1))
        r2 = self._relu(h2)
        h3 = self.bn3.forward(self.conv3.forward(r2))
        if self.proj is None:
            short = x
        else:
            short = self.short_bn.forward(self.proj.forward(x))
        H, W = len(h3[0]), len(h3[0][0])
        SH, SW = len(short[0]), len(short[0][0])
        if (SH, SW) != (H, W):
            raise ValueError(
                f"Bottleneck2d: 捷径空间尺寸 {(SH, SW)} 与主干 {(H, W)} 不符")
        pre = [[[h3[c][h][w] + short[c][h][w] for w in range(W)] for h in range(H)]
               for c in range(self.out_channels)]
        out = self._relu(pre)
        self._cache = (x, h1, r1, h2, r2, h3, short, pre)
        return out

    def backward(self, dout):
        if self._cache is None:
            raise RuntimeError("Bottleneck2d.backward: 必须先调用 forward")
        x, h1, r1, h2, r2, h3, short, pre = self._cache
        C_out, H, W = len(pre), len(pre[0]), len(pre[0][0])
        if len(dout) != C_out or len(dout[0]) != H or len(dout[0][0]) != W:
            raise ValueError(
                f"Bottleneck2d.backward: dout 形状 {(len(dout), len(dout[0]), len(dout[0][0]))} "
                f"与输出 {(C_out, H, W)} 不符")
        dpre = [[[dout[c][h][w] if pre[c][h][w] > 0.0 else 0.0 for w in range(W)]
                 for h in range(H)] for c in range(C_out)]
        dh3, g3, b3 = self.bn3.backward(dpre)
        for c in range(C_out):
            self.bn_grads["bn3_gamma"][c] += g3[c]
            self.bn_grads["bn3_beta"][c] += b3[c]
        dr2 = self.conv3.backward(dh3)
        dh2 = [[[dr2[c][h][w] if r2[c][h][w] > 0.0 else 0.0 for w in range(len(r2[0][0]))]
                for h in range(len(r2[0]))] for c in range(self.mid)]
        dr2b, g2, b2 = self.bn2.backward(dh2)
        for c in range(self.mid):
            self.bn_grads["bn2_gamma"][c] += g2[c]
            self.bn_grads["bn2_beta"][c] += b2[c]
        dr1 = self.conv2.backward(dr2b)
        dh1 = [[[dr1[c][h][w] if r1[c][h][w] > 0.0 else 0.0 for w in range(W)]
                for h in range(H)] for c in range(self.mid)]
        dxc, g1, b1 = self.bn1.backward(dh1)
        for c in range(self.mid):
            self.bn_grads["bn1_gamma"][c] += g1[c]
            self.bn_grads["bn1_beta"][c] += b1[c]
        dx_main = self.conv1.backward(dxc)
        if self.proj is None:
            d_short = dpre
        else:
            dshort_bn, gs, bs = self.short_bn.backward(dpre)
            for c in range(C_out):
                self.bn_grads["short_gamma"][c] += gs[c]
                self.bn_grads["short_beta"][c] += bs[c]
            d_short = self.proj.backward(dshort_bn)
        return [[[dx_main[c][h][w] + d_short[c][h][w] for w in range(len(dx_main[0][0]))]
                 for h in range(len(dx_main[0]))] for c in range(self.in_channels)]


class AdaIN:
    """Adaptive Instance Normalization：按通道做实例归一化，再用外部给的
    `gamma`/`beta`（风格向量）缩放平移：`y = gamma_c * (x-mean_c)/sqrt(var_c+eps) + beta_c`

    `x` 可以是 `(C,H,W)` 或长度为 `C` 的特征向量；`set_style(gamma, beta)` 换风格。
    `backward` 返回 `(dx, d_gamma, d_beta)` —— 梯度按「外部参数」返回，不内部累积。
    """

    def __init__(self, channels, eps=1e-5):
        if isinstance(channels, bool) or not isinstance(channels, int) or channels < 1:
            raise ValueError(f"AdaIN: channels 必须是正整数，收到 {channels!r}")
        if not (eps > 0):
            raise ValueError(f"AdaIN: eps 必须 > 0，收到 {eps!r}")
        self.channels = channels
        self.eps = eps
        self.gamma = [1.0] * channels
        self.beta = [0.0] * channels
        self._cache = None

    def set_style(self, gamma, beta):
        if len(gamma) != self.channels or len(beta) != self.channels:
            raise ValueError(
                f"AdaIN.set_style: gamma/beta 长度必须都是 {self.channels}，"
                f"收到 {len(gamma)}/{len(beta)}")
        self.gamma = [float(g) for g in gamma]
        self.beta = [float(b) for b in beta]

    def forward(self, x):
        if not x:
            raise ValueError("AdaIN.forward: 输入不能为空")
        if len(x) != self.channels:
            raise ValueError(f"AdaIN: 通道数 {len(x)} 与 channels={self.channels} 不符")
        out = []
        stats = []
        if isinstance(x[0], list):
            H, W = len(x[0]), len(x[0][0])
            n = H * W
            for c in range(self.channels):
                flat = [x[c][h][w] for h in range(H) for w in range(W)]
                mean = sum(flat) / n
                var = sum((v - mean) ** 2 for v in flat) / n
                inv = 1.0 / math.sqrt(var + self.eps)
                stats.append((mean, inv, flat))
                out.append([[self.gamma[c] * (x[c][h][w] - mean) * inv + self.beta[c]
                             for w in range(W)] for h in range(H)])
        else:
            n = 1
            for c in range(self.channels):
                flat = [x[c]]
                mean = x[c]
                var = 0.0
                inv = 1.0 / math.sqrt(var + self.eps)
                stats.append((mean, inv, flat))
                out.append(self.gamma[c] * (x[c] - mean) * inv + self.beta[c])
        self._cache = (x, stats)
        return out

    def backward(self, dout):
        if self._cache is None:
            raise RuntimeError("AdaIN.backward: 必须先调用 forward")
        x, stats = self._cache
        d_gamma = [0.0] * self.channels
        d_beta = [0.0] * self.channels
        nested = isinstance(x[0], list)
        if not nested:
            # 单向量时 mean=x、var=0 ⇒ x_hat ≡ 0，输出恒为 beta[c]：
            # dx 与 d_gamma 都是 0，只有 d_beta 有梯度
            return [0.0] * self.channels, [0.0] * self.channels, [dout[c] for c in range(self.channels)]
        H, W = len(x[0]), len(x[0][0])
        n = H * W
        dx = [[[0.0] * W for _ in range(H)] for _ in range(self.channels)]
        for c in range(self.channels):
            mean, inv, flat = stats[c]
            dh = [[dout[c][h][w] * self.gamma[c] for w in range(W)] for h in range(H)]
            d_gamma[c] = sum(dout[c][h][w] * (x[c][h][w] - mean) * inv
                             for h in range(H) for w in range(W))
            d_beta[c] = sum(dout[c][h][w] for h in range(H) for w in range(W))
            dflat = [dh[h][w] for h in range(H) for w in range(W)]
            s1 = sum(dflat)
            s2 = sum(d * (x[c][h][w] - mean) for d, h, w in
                     ((dflat[h * W + w], h, w) for h in range(H) for w in range(W)))
            for h in range(H):
                for w in range(W):
                    dx[c][h][w] = inv * (dflat[h * W + w]
                                         - s1 / n
                                         - (x[c][h][w] - mean) * inv * inv * s2 / n)
        return dx, d_gamma, d_beta


class FiLM:
    """Feature-wise Linear Modulation：`y = gamma * x + beta`（逐特征，可选分组）

    `groups=1` 时 gamma/beta 是整向量；`groups>1` 时把特征切成若干段、各段共享
    一组 gamma/beta（条件向量比特征维度短时用）。
    """

    def __init__(self, features, groups=1):
        for n, v in (("features", features), ("groups", groups)):
            if isinstance(v, bool) or not isinstance(v, int) or v < 1:
                raise ValueError(f"FiLM: {n} 必须是正整数，收到 {v!r}")
        if features % groups != 0:
            raise ValueError(f"FiLM: features={features} 必须能被 groups={groups} 整除")
        self.features = features
        self.groups = groups
        self.group_size = features // groups
        self.gamma = [1.0] * groups
        self.beta = [0.0] * groups
        self._x = None

    def set_params(self, gamma, beta):
        if len(gamma) != self.groups or len(beta) != self.groups:
            raise ValueError(
                f"FiLM.set_params: gamma/beta 长度必须都是 {self.groups}，"
                f"收到 {len(gamma)}/{len(beta)}")
        self.gamma = [float(g) for g in gamma]
        self.beta = [float(b) for b in beta]

    def forward(self, x):
        if len(x) != self.features:
            raise ValueError(f"FiLM: 输入长度 {len(x)} 与 features={self.features} 不符")
        self._x = list(x)
        out = [0.0] * self.features
        for i in range(self.features):
            g = self.gamma[i // self.group_size]
            b = self.beta[i // self.group_size]
            out[i] = g * x[i] + b
        return out

    def backward(self, dout):
        if self._x is None:
            raise RuntimeError("FiLM.backward: 必须先调用 forward")
        if len(dout) != self.features:
            raise ValueError(f"FiLM.backward: dout 长度 {len(dout)} != {self.features}")
        dx = [0.0] * self.features
        d_gamma = [0.0] * self.groups
        d_beta = [0.0] * self.groups
        for i in range(self.features):
            g_i = i // self.group_size
            dx[i] = dout[i] * self.gamma[g_i]
            d_gamma[g_i] += dout[i] * self._x[i]
            d_beta[g_i] += dout[i]
        return dx, d_gamma, d_beta
class ConvTranspose1d:
    """1D 转置卷积：输入 (Cin, L) -> (Cout, Lo)，`Lo = (L-1)*s - 2p + k + output_padding`

    前向就是「把每个输入点乘上整条核、按 stride 摆到输出网格上累加」；反向对应
    `dx` 与 `grad_w` 两次同样的累加。权重 `[out][in][k]`（转置卷积不做 groups）。
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, output_padding=0):
        for n, v, lo in (("in_channels", in_channels, 1), ("out_channels", out_channels, 1),
                         ("kernel_size", kernel_size, 1), ("stride", stride, 1),
                         ("padding", padding, 0), ("output_padding", output_padding, 0)):
            if isinstance(v, bool) or not isinstance(v, int) or v < lo:
                raise ValueError(f"ConvTranspose1d: {n} 必须是 >= {lo} 的整数，收到 {v!r}")
        if output_padding >= stride:
            raise ValueError(
                f"ConvTranspose1d: output_padding={output_padding} 必须 < stride={stride}")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k = kernel_size
        self.s = stride
        self.p = padding
        self.op = output_padding
        fan_in = in_channels * self.k
        std = math.sqrt(2.0 / fan_in)
        self.weights = [[[random.gauss(0, std) for _ in range(self.k)]
                         for _ in range(in_channels)] for _ in range(out_channels)]
        self.biases = [0.0] * out_channels
        self.m_w = [[[0.0] * self.k for _ in range(in_channels)] for _ in range(out_channels)]
        self.v_w = [[[0.0] * self.k for _ in range(in_channels)] for _ in range(out_channels)]
        self.m_b = [0.0] * out_channels
        self.v_b = [0.0] * out_channels
        self.x = None
        self.lo = 0
        self.zero_grad()

    def zero_grad(self):
        self.grad_w = [[[0.0] * self.k for _ in range(self.in_channels)]
                       for _ in range(self.out_channels)]
        self.grad_b = [0.0] * self.out_channels

    def _out_len(self, L):
        lo = (L - 1) * self.s - 2 * self.p + self.k + self.op
        if lo <= 0:
            raise ValueError(
                f"ConvTranspose1d: 输入长度 {L} 配 kernel {self.k}、stride {self.s}、"
                f"padding {self.p}、output_padding {self.op} 得到输出长度 {lo}，必须 > 0")
        return lo

    def forward(self, x):
        if not x or len(x) != self.in_channels:
            raise ValueError(
                f"ConvTranspose1d.forward: 输入通道 {len(x) if x else 0} "
                f"!= in_channels={self.in_channels}")
        L = len(x[0])
        for c in range(len(x)):
            if len(x[c]) != L:
                raise ValueError("ConvTranspose1d.forward: 各通道长度必须一致")
        lo = self._out_len(L)
        self.x = x
        self.lo = lo
        out = [[self.biases[oc]] * lo for oc in range(self.out_channels)]
        out = [list(row) for row in out]
        for oc in range(self.out_channels):
            w_oc = self.weights[oc]
            for ic in range(self.in_channels):
                w = w_oc[ic]
                xc = x[ic]
                for o in range(L):
                    v = xc[o]
                    if v == 0.0:
                        continue
                    base = o * self.s - self.p
                    for k in range(self.k):
                        idx = base + k
                        if 0 <= idx < lo:
                            out[oc][idx] += v * w[k]
        return out

    def backward(self, dout):
        if self.x is None:
            raise RuntimeError("ConvTranspose1d.backward: 请先调用 forward")
        L = len(self.x[0])
        lo = self.lo
        dx = [[0.0] * L for _ in range(self.in_channels)]
        for oc in range(self.out_channels):
            d = dout[oc]
            self.grad_b[oc] += sum(d)
            w_oc = self.weights[oc]
            gw_oc = self.grad_w[oc]
            for ic in range(self.in_channels):
                w = w_oc[ic]
                gw = gw_oc[ic]
                xc = self.x[ic]
                dxi = dx[ic]
                for o in range(L):
                    base = o * self.s - self.p
                    acc = 0.0
                    for k in range(self.k):
                        idx = base + k
                        if 0 <= idx < lo:
                            dv = d[idx]
                            acc += dv * w[k]
                            gw[k] += xc[o] * dv
                    dxi[o] += acc
        return dx


class MaxPool1d:
    """一维最大池化：`(C, L) -> (C, Lo)`，反向只回传每个窗口的最大值位置"""

    def __init__(self, kernel_size, stride=None, padding=0):
        for n, v, lo in (("kernel_size", kernel_size, 1), ("padding", padding, 0)):
            if isinstance(v, bool) or not isinstance(v, int) or v < lo:
                raise ValueError(f"MaxPool1d: {n} 必须是 >= {lo} 的整数，收到 {v!r}")
        if stride is None:
            stride = kernel_size
        if isinstance(stride, bool) or not isinstance(stride, int) or stride < 1:
            raise ValueError(f"MaxPool1d: stride 必须是正整数，收到 {stride!r}")
        if padding >= kernel_size:
            raise ValueError(
                f"MaxPool1d: padding={padding} 必须 < kernel_size={kernel_size}")
        self.k = kernel_size
        self.s = stride
        self.p = padding
        self._cache = None

    def _out_len(self, L):
        lo = (L + 2 * self.p - self.k) // self.s + 1
        if lo <= 0:
            raise ValueError(f"MaxPool1d: 输入长度 {L} 太小，输出长度 {lo}")
        return lo

    def forward(self, x):
        if not x:
            raise ValueError("MaxPool1d.forward: 输入不能为空")
        L = len(x[0])
        for c in range(len(x)):
            if len(x[c]) != L:
                raise ValueError("MaxPool1d.forward: 各通道长度必须一致")
        lo = self._out_len(L)
        out = []
        idxs = []
        for c in range(len(x)):
            row = []
            ridx = []
            for o in range(lo):
                base = o * self.s - self.p
                best = None
                best_i = -1
                for k in range(self.k):
                    i = base + k
                    if 0 <= i < L:
                        v = x[c][i]
                        if best is None or v > best:
                            best, best_i = v, i
                row.append(best if best is not None else 0.0)
                ridx.append(best_i)
            out.append(row)
            idxs.append(ridx)
        self._cache = (len(x), L, idxs)
        return out

    def backward(self, dout):
        if self._cache is None:
            raise RuntimeError("MaxPool1d.backward: 必须先调用 forward")
        C, L, idxs = self._cache
        lo = len(dout[0])
        dx = [[0.0] * L for _ in range(C)]
        for c in range(C):
            for o in range(lo):
                i = idxs[c][o]
                if 0 <= i < L:
                    dx[c][i] += dout[c][o]
        return dx


class AvgPool1d:
    """一维平均池化：`(C, L) -> (C, Lo)`，反向把梯度均分回窗口（含 padding 的零位不接收梯度）"""

    def __init__(self, kernel_size, stride=None, padding=0):
        for n, v, lo in (("kernel_size", kernel_size, 1), ("padding", padding, 0)):
            if isinstance(v, bool) or not isinstance(v, int) or v < lo:
                raise ValueError(f"AvgPool1d: {n} 必须是 >= {lo} 的整数，收到 {v!r}")
        if stride is None:
            stride = kernel_size
        if isinstance(stride, bool) or not isinstance(stride, int) or stride < 1:
            raise ValueError(f"AvgPool1d: stride 必须是正整数，收到 {stride!r}")
        if padding >= kernel_size:
            raise ValueError(
                f"AvgPool1d: padding={padding} 必须 < kernel_size={kernel_size}")
        self.k = kernel_size
        self.s = stride
        self.p = padding
        self._cache = None

    def _out_len(self, L):
        lo = (L + 2 * self.p - self.k) // self.s + 1
        if lo <= 0:
            raise ValueError(f"AvgPool1d: 输入长度 {L} 太小，输出长度 {lo}")
        return lo

    def forward(self, x):
        if not x:
            raise ValueError("AvgPool1d.forward: 输入不能为空")
        L = len(x[0])
        lo = self._out_len(L)
        out = []
        for c in range(len(x)):
            row = []
            for o in range(lo):
                base = o * self.s - self.p
                acc = 0.0
                for k in range(self.k):
                    i = base + k
                    if 0 <= i < L:
                        acc += x[c][i]
                row.append(acc / self.k)
            out.append(row)
        self._cache = (len(x), L, lo)
        return out

    def backward(self, dout):
        if self._cache is None:
            raise RuntimeError("AvgPool1d.backward: 必须先调用 forward")
        C, L, lo = self._cache
        dx = [[0.0] * L for _ in range(C)]
        inv = 1.0 / self.k
        for c in range(C):
            for o in range(lo):
                base = o * self.s - self.p
                for k in range(self.k):
                    i = base + k
                    if 0 <= i < L:
                        dx[c][i] += dout[c][o] * inv
        return dx


class MaxPool3d:
    """三维最大池化：`(C, D, H, W) -> (C, Do, Ho, Wo)`，反向只回传最大值位置"""

    def __init__(self, kernel_size, stride=None, padding=0):
        for n, v, lo in (("kernel_size", kernel_size, 1), ("padding", padding, 0)):
            if isinstance(v, bool) or not isinstance(v, int) or v < lo:
                raise ValueError(f"MaxPool3d: {n} 必须是 >= {lo} 的整数，收到 {v!r}")
        if stride is None:
            stride = kernel_size
        if isinstance(stride, bool) or not isinstance(stride, int) or stride < 1:
            raise ValueError(f"MaxPool3d: stride 必须是正整数，收到 {stride!r}")
        self.k = kernel_size
        self.s = stride
        self.p = padding
        self._cache = None

    def _out_len(self, n):
        o = (n + 2 * self.p - self.k) // self.s + 1
        if o <= 0:
            raise ValueError(f"MaxPool3d: 输入尺寸 {n} 太小，输出 {o}")
        return o

    def forward(self, x):
        if not x or not x[0] or not x[0][0]:
            raise ValueError("MaxPool3d.forward: 输入必须是 (C, D, H, W)")
        C, D, H, W = len(x), len(x[0]), len(x[0][0]), len(x[0][0][0])
        Do, Ho, Wo = self._out_len(D), self._out_len(H), self._out_len(W)
        out = [[[[0.0] * Wo for _ in range(Ho)] for _ in range(Do)] for _ in range(C)]
        idxs = [[[[(-1, -1, -1)] * Wo for _ in range(Ho)] for _ in range(Do)] for _ in range(C)]
        for c in range(C):
            for od in range(Do):
                for oh in range(Ho):
                    for ow in range(Wo):
                        best = None
                        pos = (-1, -1, -1)
                        for kd in range(self.k):
                            d = od * self.s - self.p + kd
                            if not (0 <= d < D):
                                continue
                            for kh in range(self.k):
                                h = oh * self.s - self.p + kh
                                if not (0 <= h < H):
                                    continue
                                for kw in range(self.k):
                                    w = ow * self.s - self.p + kw
                                    if not (0 <= w < W):
                                        continue
                                    v = x[c][d][h][w]
                                    if best is None or v > best:
                                        best, pos = v, (d, h, w)
                        out[c][od][oh][ow] = best if best is not None else 0.0
                        idxs[c][od][oh][ow] = pos
        self._cache = (C, D, H, W, idxs)
        return out

    def backward(self, dout):
        if self._cache is None:
            raise RuntimeError("MaxPool3d.backward: 必须先调用 forward")
        C, D, H, W, idxs = self._cache
        dx = [[[[0.0] * W for _ in range(H)] for _ in range(D)] for _ in range(C)]
        Do, Ho, Wo = len(dout[0]), len(dout[0][0]), len(dout[0][0][0])
        for c in range(C):
            for od in range(Do):
                for oh in range(Ho):
                    for ow in range(Wo):
                        d, h, w = idxs[c][od][oh][ow]
                        if d >= 0:
                            dx[c][d][h][w] += dout[c][od][oh][ow]
        return dx


class AvgPool3d:
    """三维平均池化：`(C, D, H, W) -> (C, Do, Ho, Wo)`，反向把梯度均分回窗口"""

    def __init__(self, kernel_size, stride=None, padding=0):
        for n, v, lo in (("kernel_size", kernel_size, 1), ("padding", padding, 0)):
            if isinstance(v, bool) or not isinstance(v, int) or v < lo:
                raise ValueError(f"AvgPool3d: {n} 必须是 >= {lo} 的整数，收到 {v!r}")
        if stride is None:
            stride = kernel_size
        if isinstance(stride, bool) or not isinstance(stride, int) or stride < 1:
            raise ValueError(f"AvgPool3d: stride 必须是正整数，收到 {stride!r}")
        self.k = kernel_size
        self.s = stride
        self.p = padding
        self._cache = None

    def _out_len(self, n):
        o = (n + 2 * self.p - self.k) // self.s + 1
        if o <= 0:
            raise ValueError(f"AvgPool3d: 输入尺寸 {n} 太小，输出 {o}")
        return o

    def forward(self, x):
        if not x or not x[0] or not x[0][0]:
            raise ValueError("AvgPool3d.forward: 输入必须是 (C, D, H, W)")
        C, D, H, W = len(x), len(x[0]), len(x[0][0]), len(x[0][0][0])
        Do, Ho, Wo = self._out_len(D), self._out_len(H), self._out_len(W)
        out = [[[[0.0] * Wo for _ in range(Ho)] for _ in range(Do)] for _ in range(C)]
        for c in range(C):
            for od in range(Do):
                for oh in range(Ho):
                    for ow in range(Wo):
                        acc = 0.0
                        for kd in range(self.k):
                            d = od * self.s - self.p + kd
                            if not (0 <= d < D):
                                continue
                            for kh in range(self.k):
                                h = oh * self.s - self.p + kh
                                if not (0 <= h < H):
                                    continue
                                for kw in range(self.k):
                                    w = ow * self.s - self.p + kw
                                    if 0 <= w < W:
                                        acc += x[c][d][h][w]
                        out[c][od][oh][ow] = acc / (self.k ** 3)
        self._cache = (C, D, H, W, Do, Ho, Wo)
        return out

    def backward(self, dout):
        if self._cache is None:
            raise RuntimeError("AvgPool3d.backward: 必须先调用 forward")
        C, D, H, W, Do, Ho, Wo = self._cache
        dx = [[[[0.0] * W for _ in range(H)] for _ in range(D)] for _ in range(C)]
        inv = 1.0 / (self.k ** 3)
        for c in range(C):
            for od in range(Do):
                for oh in range(Ho):
                    for ow in range(Wo):
                        g = dout[c][od][oh][ow] * inv
                        for kd in range(self.k):
                            d = od * self.s - self.p + kd
                            if not (0 <= d < D):
                                continue
                            for kh in range(self.k):
                                h = oh * self.s - self.p + kh
                                if not (0 <= h < H):
                                    continue
                                for kw in range(self.k):
                                    w = ow * self.s - self.p + kw
                                    if 0 <= w < W:
                                        dx[c][d][h][w] += g
        return dx
def _conv3d_zeros(out_channels, in_channels, kd, kh, kw):
    """构造 [oc][ic][kd][kh][kw] 的全零嵌套列表（用循环，避免推导式括号层数出错）"""
    out = []
    for _ in range(out_channels):
        per_oc = []
        for _ in range(in_channels):
            per_ic = []
            for _ in range(kd):
                per_kd = []
                for _ in range(kh):
                    per_kd.append([0.0] * kw)
                per_ic.append(per_kd)
            per_oc.append(per_ic)
        out.append(per_oc)
    return out


def _conv3d_random(out_channels, in_channels, kd, kh, kw, std):
    """He 初始化用的随机核，结构与 _conv3d_zeros 相同"""
    out = _conv3d_zeros(out_channels, in_channels, kd, kh, kw)
    for oc in range(out_channels):
        for ic in range(in_channels):
            for a in range(kd):
                for b in range(kh):
                    for c in range(kw):
                        out[oc][ic][a][b][c] = random.gauss(0, std)
    return out


class Conv3d:
    """3D 卷积层（输入 (C, D, H, W)，权重 [oc][ic][kd][kh][kw]，He 初始化）

    直接滑动窗口累加，不走 im2col：三维展开的矩阵通常是 im2col 的几十倍，
    而这层主要处理小体数据（医学影像 patch、视频片段），直接循环更省内存。
    不做 groups —— 分组三维卷积用不到，先不引入这层复杂度。
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size,
                 stride=1, padding=0):
        for name, v in (("in_channels", in_channels),
                        ("out_channels", out_channels)):
            if not isinstance(v, int) or isinstance(v, bool) or v <= 0:
                raise ValueError(f"Conv3d: {name} 必须是正整数，收到 {v!r}")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = ((kernel_size, ) * 3 if isinstance(kernel_size, int)
                            else tuple(kernel_size))
        self.stride = ((stride, ) * 3 if isinstance(stride, int) else tuple(stride))
        self.padding = ((padding, ) * 3 if isinstance(padding, int)
                        else tuple(padding))
        if len(self.kernel_size) != 3 or min(self.kernel_size) <= 0:
            raise ValueError(
                f"Conv3d: kernel_size 必须是三个正数，收到 {kernel_size!r}")
        if len(self.stride) != 3 or min(self.stride) <= 0:
            raise ValueError(f"Conv3d: stride 必须是三个正数，收到 {stride!r}")
        if len(self.padding) != 3 or min(self.padding) < 0:
            raise ValueError(f"Conv3d: padding 必须是三个非负数，收到 {padding!r}")
        self.kd, self.kh, self.kw = self.kernel_size
        self.sd, self.sh, self.sw = self.stride
        self.pd, self.ph, self.pw = self.padding

        fan_in = in_channels * self.kd * self.kh * self.kw
        std = math.sqrt(2.0 / fan_in)
        self.weights = _conv3d_random(out_channels, in_channels,
                                      self.kd, self.kh, self.kw, std)
        self.biases = [0.0] * out_channels
        self.m_w = _conv3d_zeros(out_channels, in_channels,
                                 self.kd, self.kh, self.kw)
        self.v_w = _conv3d_zeros(out_channels, in_channels,
                                 self.kd, self.kh, self.kw)
        self.m_b = [0.0] * out_channels
        self.v_b = [0.0] * out_channels
        self.x_shape = None
        self.out_shape = None
        self.zero_grad()

    def zero_grad(self):
        """清零梯度缓冲区（保留 x_shape/out_shape 供 backward 使用）"""
        self.grad_w = _conv3d_zeros(self.out_channels, self.in_channels,
                                    self.kd, self.kh, self.kw)
        self.grad_b = [0.0] * self.out_channels

    def param_count(self):
        """可训练参数个数（权重 + 偏置）"""
        return (self.out_channels * self.in_channels * self.kd * self.kh * self.kw
                + self.out_channels)

    def _out_shape(self, D, H, W):
        od = (D + 2 * self.pd - self.kd) // self.sd + 1
        oh = (H + 2 * self.ph - self.kh) // self.sh + 1
        ow = (W + 2 * self.pw - self.kw) // self.sw + 1
        if od <= 0 or oh <= 0 or ow <= 0:
            raise ValueError(
                f"Conv3d: 输入 {D}x{H}x{W} 配 kernel {self.kernel_size}、"
                f"padding {self.padding}、stride {self.stride} 得到输出 "
                f"{od}x{oh}x{ow}，必须都 > 0"
            )
        return od, oh, ow

    def _pad(self, x):
        """对 (C, D, H, W) 做 zero-padding"""
        C, D, H, W = len(x), len(x[0]), len(x[0][0]), len(x[0][0][0])
        Dp, Hp, Wp = D + 2 * self.pd, H + 2 * self.ph, W + 2 * self.pw
        padded = []
        for _ in range(C):
            per_c = []
            for _ in range(Dp):
                per_d = []
                for _ in range(Hp):
                    per_d.append([0.0] * Wp)
                per_c.append(per_d)
            padded.append(per_c)
        for c in range(C):
            for d in range(D):
                for h in range(H):
                    for w in range(W):
                        padded[c][d + self.pd][h + self.ph][w + self.pw] = x[c][d][h][w]
        return padded

    def forward(self, x):
        """x: (C, D, H, W) -> (out_channels, Od, Oh, Ow)"""
        if not x or not isinstance(x[0], list) or not isinstance(x[0][0], list):
            raise ValueError("Conv3d.forward: 输入必须是 (C, D, H, W) 嵌套列表")
        if len(x) != self.in_channels:
            raise ValueError(
                f"Conv3d.forward: 输入通道 {len(x)} != in_channels={self.in_channels}"
            )
        C, D, H, W = len(x), len(x[0]), len(x[0][0]), len(x[0][0][0])
        od, oh, ow = self._out_shape(D, H, W)
        self.x_shape = (C, D, H, W)
        self.out_shape = (self.out_channels, od, oh, ow)
        xp = self._pad(x)
        self._xp = xp

        out = []
        for oc in range(self.out_channels):
            woc = self.weights[oc]
            bias = self.biases[oc]
            plane = []
            for o_d in range(od):
                slab = []
                for o_h in range(oh):
                    row = []
                    for o_w in range(ow):
                        acc = 0.0
                        for ic in range(C):
                            wic = woc[ic]
                            xc = xp[ic]
                            for kd in range(self.kd):
                                xd = xc[o_d * self.sd + kd]
                                wd = wic[kd]
                                for kh in range(self.kh):
                                    xh = xd[o_h * self.sh + kh]
                                    wh = wd[kh]
                                    for kw in range(self.kw):
                                        acc += wh[kw] * xh[o_w * self.sw + kw]
                        row.append(acc + bias)
                    slab.append(row)
                plane.append(slab)
            out.append(plane)
        return out

    def backward(self, dout):
        """dout: (out_channels, Od, Oh, Ow) -> dx: (C, D, H, W)"""
        if self.x_shape is None:
            raise RuntimeError("Conv3d.backward: 需要先调用 forward")
        C, D, H, W = self.x_shape
        _, od, oh, ow = self.out_shape
        Dp, Hp, Wp = D + 2 * self.pd, H + 2 * self.ph, W + 2 * self.pw
        xp = self._xp
        dxp = []
        for _ in range(C):
            per_c = []
            for _ in range(Dp):
                per_d = []
                for _ in range(Hp):
                    per_d.append([0.0] * Wp)
                per_c.append(per_d)
            dxp.append(per_c)

        for oc in range(self.out_channels):
            woc = self.weights[oc]
            doc = dout[oc]
            for o_d in range(od):
                for o_h in range(oh):
                    row = doc[o_d][o_h]
                    self.grad_b[oc] += sum(row)
                    for o_w in range(ow):
                        g = row[o_w]
                        for ic in range(C):
                            wic = woc[ic]
                            gic = self.grad_w[oc][ic]
                            xc = xp[ic]
                            dc = dxp[ic]
                            for kd in range(self.kd):
                                i_d = o_d * self.sd + kd
                                xd = xc[i_d]
                                dd = dc[i_d]
                                gd = gic[kd]
                                wd = wic[kd]
                                for kh in range(self.kh):
                                    i_h = o_h * self.sh + kh
                                    xh = xd[i_h]
                                    dh = dd[i_h]
                                    gh = gd[kh]
                                    wh = wd[kh]
                                    for kw in range(self.kw):
                                        i_w = o_w * self.sw + kw
                                        gh[kw] += g * xh[i_w]
                                        dh[i_w] += g * wh[kw]

        if min(self.padding) == 0:
            return dxp
        dx = []
        for _ in range(C):
            per_c = []
            for _ in range(D):
                per_d = []
                for _ in range(H):
                    per_d.append([0.0] * W)
                per_c.append(per_d)
            dx.append(per_c)
        for c in range(C):
            for d in range(D):
                for h in range(H):
                    for w in range(W):
                        dx[c][d][h][w] = dxp[c][d + self.pd][h + self.ph][w + self.pw]
        return dx


class ConvTranspose3d:
    """3D 转置卷积（输入 (Cin, D, H, W) -> (Cout, Od, Oh, Ow)）

    每个维度的输出尺寸 `(D-1)*sd - 2*pd + kd + opd`（三轴独立）。前向把每个
    输入点乘上卷积核后按 stride 摆到输出网格累加，反向对 dx 与 grad_w 做同样
    的散射；`output_padding` 只影响输出尺寸，不引入额外参数。
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size,
                 stride=1, padding=0, output_padding=0):
        for name, v in (("in_channels", in_channels),
                        ("out_channels", out_channels)):
            if not isinstance(v, int) or isinstance(v, bool) or v <= 0:
                raise ValueError(f"ConvTranspose3d: {name} 必须是正整数，收到 {v!r}")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = ((kernel_size, ) * 3 if isinstance(kernel_size, int)
                            else tuple(kernel_size))
        self.stride = ((stride, ) * 3 if isinstance(stride, int) else tuple(stride))
        self.padding = ((padding, ) * 3 if isinstance(padding, int)
                        else tuple(padding))
        self.output_padding = ((output_padding, ) * 3
                               if isinstance(output_padding, int)
                               else tuple(output_padding))
        if len(self.kernel_size) != 3 or min(self.kernel_size) <= 0:
            raise ValueError(
                f"ConvTranspose3d: kernel_size 必须是三个正数，收到 {kernel_size!r}")
        if len(self.stride) != 3 or min(self.stride) <= 0:
            raise ValueError(
                f"ConvTranspose3d: stride 必须是三个正数，收到 {stride!r}")
        if len(self.padding) != 3 or min(self.padding) < 0:
            raise ValueError(
                f"ConvTranspose3d: padding 必须是三个非负数，收到 {padding!r}")
        if len(self.output_padding) != 3 or min(self.output_padding) < 0:
            raise ValueError(
                f"ConvTranspose3d: output_padding 必须是三个非负数，"
                f"收到 {output_padding!r}")
        self.kd, self.kh, self.kw = self.kernel_size
        self.sd, self.sh, self.sw = self.stride
        self.pd, self.ph, self.pw = self.padding
        self.opd, self.oph, self.opw = self.output_padding
        for name, op, st in (("D", self.opd, self.sd), ("H", self.oph, self.sh),
                             ("W", self.opw, self.sw)):
            if op >= st:
                raise ValueError(
                    f"ConvTranspose3d: 维度 {name} 的 output_padding={op} "
                    f"必须小于 stride={st}"
                )

        fan_in = in_channels * self.kd * self.kh * self.kw
        std = math.sqrt(2.0 / fan_in)
        self.weights = _conv3d_random(out_channels, in_channels,
                                      self.kd, self.kh, self.kw, std)
        self.biases = [0.0] * out_channels
        self.m_w = _conv3d_zeros(out_channels, in_channels,
                                 self.kd, self.kh, self.kw)
        self.v_w = _conv3d_zeros(out_channels, in_channels,
                                 self.kd, self.kh, self.kw)
        self.m_b = [0.0] * out_channels
        self.v_b = [0.0] * out_channels
        self.x_shape = None
        self.out_shape = None
        self.zero_grad()

    def zero_grad(self):
        self.grad_w = _conv3d_zeros(self.out_channels, self.in_channels,
                                    self.kd, self.kh, self.kw)
        self.grad_b = [0.0] * self.out_channels

    def param_count(self):
        return (self.out_channels * self.in_channels * self.kd * self.kh * self.kw
                + self.out_channels)

    def _out_shape(self, D, H, W):
        od = (D - 1) * self.sd - 2 * self.pd + self.kd + self.opd
        oh = (H - 1) * self.sh - 2 * self.ph + self.kh + self.oph
        ow = (W - 1) * self.sw - 2 * self.pw + self.kw + self.opw
        if od <= 0 or oh <= 0 or ow <= 0:
            raise ValueError(
                f"ConvTranspose3d: 输入 {D}x{H}x{W} 得到输出 {od}x{oh}x{ow}，"
                f"必须都 > 0（检查 kernel/stride/padding/output_padding）"
            )
        return od, oh, ow

    def forward(self, x):
        """x: (Cin, D, H, W) -> (Cout, Od, Oh, Ow)"""
        if not x or not isinstance(x[0], list) or not isinstance(x[0][0], list):
            raise ValueError(
                "ConvTranspose3d.forward: 输入必须是 (Cin, D, H, W) 嵌套列表")
        if len(x) != self.in_channels:
            raise ValueError(
                f"ConvTranspose3d.forward: 输入通道 {len(x)} "
                f"!= in_channels={self.in_channels}"
            )
        C, D, H, W = len(x), len(x[0]), len(x[0][0]), len(x[0][0][0])
        od, oh, ow = self._out_shape(D, H, W)
        self.x_shape = (C, D, H, W)
        self.out_shape = (self.out_channels, od, oh, ow)
        self._x = x

        # 每个输出位置先放上偏置，再把输入点散射累加进去
        out = []
        for oc in range(self.out_channels):
            b = float(self.biases[oc])
            per_oc = []
            for _ in range(od):
                per_d = []
                for _ in range(oh):
                    per_d.append([b] * ow)
                per_oc.append(per_d)
            out.append(per_oc)

        for ic in range(C):
            xc = x[ic]
            for i_d in range(D):
                for i_h in range(H):
                    for i_w in range(W):
                        v = xc[i_d][i_h][i_w]
                        if v == 0.0:
                            continue
                        for oc in range(self.out_channels):
                            woc = self.weights[oc][ic]
                            for kd in range(self.kd):
                                o_d = i_d * self.sd - self.pd + kd
                                if o_d < 0 or o_d >= od:
                                    continue
                                for kh in range(self.kh):
                                    o_h = i_h * self.sh - self.ph + kh
                                    if o_h < 0 or o_h >= oh:
                                        continue
                                    dd = out[oc][o_d][o_h]
                                    wr = woc[kd][kh]
                                    for kw in range(self.kw):
                                        o_w = i_w * self.sw - self.pw + kw
                                        if 0 <= o_w < ow:
                                            dd[o_w] += v * wr[kw]
        return out

    def backward(self, dout):
        """dout: (Cout, Od, Oh, Ow) -> dx: (Cin, D, H, W)"""
        if self.x_shape is None:
            raise RuntimeError("ConvTranspose3d.backward: 需要先调用 forward")
        C, D, H, W = self.x_shape
        _, od, oh, ow = self.out_shape
        x = self._x

        for oc in range(self.out_channels):
            doc = dout[oc]
            self.grad_b[oc] += sum(doc[d][h][w]
                                   for d in range(od) for h in range(oh)
                                   for w in range(ow))

        dx = []
        for _ in range(C):
            per_c = []
            for _ in range(D):
                per_d = []
                for _ in range(H):
                    per_d.append([0.0] * W)
                per_c.append(per_d)
            dx.append(per_c)

        for ic in range(C):
            xc = x[ic]
            for i_d in range(D):
                for i_h in range(H):
                    for i_w in range(W):
                        acc = 0.0
                        v = xc[i_d][i_h][i_w]
                        for oc in range(self.out_channels):
                            woc = self.weights[oc][ic]
                            doc = dout[oc]
                            wg = self.grad_w[oc][ic]
                            for kd in range(self.kd):
                                o_d = i_d * self.sd - self.pd + kd
                                if o_d < 0 or o_d >= od:
                                    continue
                                for kh in range(self.kh):
                                    o_h = i_h * self.sh - self.ph + kh
                                    if o_h < 0 or o_h >= oh:
                                        continue
                                    for kw in range(self.kw):
                                        o_w = i_w * self.sw - self.pw + kw
                                        if 0 <= o_w < ow:
                                            g = doc[o_d][o_h][o_w]
                                            acc += g * woc[kd][kh][kw]
                                            wg[kd][kh][kw] += g * v
                        dx[ic][i_d][i_h][i_w] = acc
        return dx


class Flatten:
    """将 (C, H, W) 展平为 (C*H*W,)"""

    def __init__(self):
        self.x_shape = None

    def forward(self, x):
        """x: (C, H, W) -> 一维列表"""
        C, H, W = len(x), len(x[0]), len(x[0][0])
        self.x_shape = (C, H, W)
        flat = []
        for c in range(C):
            for h in range(H):
                for w in range(W):
                    flat.append(x[c][h][w])
        return flat

    def backward(self, delta):
        """delta: 一维列表 -> (C, H, W)"""
        C, H, W = self.x_shape
        dx = [[[0.0] * W for _ in range(H)] for _ in range(C)]
        idx = 0
        for c in range(C):
            for h in range(H):
                for w in range(W):
                    dx[c][h][w] = delta[idx]
                    idx += 1
        return dx


# ============================================================================
#  Regularization: Dropout / BatchNorm1d / LayerNorm
# ============================================================================

class Dropout:
    """随机丢弃神经元（Inverted Dropout），防止过拟合

    训练时以概率 p 将神经元置零，保留的神经元乘以 1/(1-p) 保持期望不变。
    """

    def __init__(self, p: float = 0.5):
        # p 越界会产生除零（p=1）、全零（p>1）或放大（p<0），必须早失败
        if not isinstance(p, (int, float)) or not (0.0 <= p < 1.0):
            raise ValueError(f"Dropout: p 必须在 [0, 1) 内，收到 {p!r}")
        self.p = p
        self.mask = None
        self.training = True

    def train(self):
        """切到训练模式（dropout 生效）"""
        self.training = True
        return self

    def eval(self):
        """切到推理模式（dropout 关闭，输出确定）

        默认是训练模式，所以在推理前必须显式调用 eval()，
        否则同一输入的两次预测结果会不同。
        """
        self.training = False
        return self

    def forward(self, x):
        if self.training:
            scale = 1.0 / (1.0 - self.p)
            self.mask = [scale if random.random() > self.p else 0.0 for _ in x]
            # 用 zip 一次遍历，省掉 range 索引（数值与逐元素乘法完全一致）
            return [xi * m for xi, m in zip(x, self.mask)]
        self.mask = [1.0] * len(x)
        return x

    def backward(self, delta):
        return [delta[i] * self.mask[i] for i in range(len(delta))]


def _scale_nested(x, k: float):
    """把嵌套 list（向量 / 序列 / (C,H,W)）整体乘 k；k == 1.0 时原样返回"""
    if k == 1.0:
        return x
    if isinstance(x, list):
        return [_scale_nested(v, k) for v in x]
    return x * k

class DropPath:
    """Stochastic Depth / DropPath：训练时整条残差分支以概率 p 被跳过

    与 Dropout 的区别：Dropout 按元素丢，DropPath 丢**整条分支**（整个张量），
    常见于深层 Transformer / ConvNeXt 的残差块里。训练时保留的分支放大
    1/(1-p)（inverted，推理不做任何缩放）；`eval()` 后恒等。

    用法::

        out = x + drop_path(block(x))      # drop_path 在前面挡一下
    """

    def __init__(self, p: float = 0.0, scale_by_keep: bool = True):
        if not isinstance(p, (int, float)) or isinstance(p, bool) or not (0.0 <= p < 1.0):
            raise ValueError(f"DropPath: p 必须在 [0, 1) 内，收到 {p!r}")
        self.p = float(p)
        self.scale_by_keep = bool(scale_by_keep)
        self.factor = 1.0
        self.training = True

    def train(self):
        """训练模式：按概率整条丢弃"""
        self.training = True
        return self

    def eval(self):
        """推理模式：恒等映射"""
        self.training = False
        return self

    def forward(self, x):
        if not self.training or self.p == 0.0:
            self.factor = 1.0
            return x
        if random.random() < self.p:
            self.factor = 0.0
        else:
            self.factor = 1.0 / (1.0 - self.p) if self.scale_by_keep else 1.0
        return _scale_nested(x, self.factor)

    def backward(self, dout):
        return _scale_nested(dout, self.factor)


class BatchNorm1d:
    """一维批归一化

    训练: forward_batch(batch) 用 batch 统计量
    推理: forward(x) 用 running_mean / running_var
    """

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        self.d = num_features
        self.eps = eps
        self.momentum = momentum

        self.gamma = [1.0] * self.d
        self.beta = [0.0] * self.d

        self.running_mean = [0.0] * self.d
        self.running_var = [1.0] * self.d

        self.x_batch = self.x_hat = self.mean = self.var = self.inv_std = None
        self.n = 0
        self.training = True

        self.m_gamma = [0.0] * self.d; self.v_gamma = [0.0] * self.d
        self.m_beta = [0.0] * self.d; self.v_beta = [0.0] * self.d

    def forward_batch(self, x_batch):
        """x_batch: List[List[float]]  (batch_size, d)"""
        # zip(*x_batch) 会截断到最短行、多余特征被静默丢弃并污染 running 统计量
        if not x_batch:
            raise ValueError("BatchNorm1d.forward_batch: batch 不能为空")
        if any(len(row) != self.d for row in x_batch):
            raise ValueError(
                f"BatchNorm1d.forward_batch: 每个样本的特征维度都必须是 {self.d}"
            )
        self.x_batch = x_batch
        self._cache_mode = 'train'      # backward 需要知道该用 batch 还是 running 统计量
        n = self.n = len(x_batch)
        d = self.d

        # 先整体转置成列：列求和交给 C 层的 sum，比 sum(row[j] for row in x_batch) 快约 2.4 倍
        cols = list(zip(*x_batch))
        self.mean = [sum(c) / n for c in cols]
        self.var = [sum([(v - m) ** 2 for v in c]) / n for c, m in zip(cols, self.mean)]

        for j in range(d):
            self.running_mean[j] = (1 - self.momentum) * self.running_mean[j] + self.momentum * self.mean[j]
            # n=1 时无偏估计无定义，使用有偏估计作为 fallback
            if n > 1:
                self.running_var[j] = (1 - self.momentum) * self.running_var[j] + self.momentum * self.var[j] * n / (n - 1)
            else:
                self.running_var[j] = (1 - self.momentum) * self.running_var[j] + self.momentum * self.var[j]

        self.inv_std = [1.0 / math.sqrt(v + self.eps) for v in self.var]
        mean, inv_std = self.mean, self.inv_std
        self.x_hat = [[(v - m) * s for v, m, s in zip(row, mean, inv_std)] for row in x_batch]

        gamma, beta = self.gamma, self.beta
        return [[g * xh + b for g, xh, b in zip(gamma, xh_row, beta)] for xh_row in self.x_hat]

    # backward 返回 (dx, grad_gamma, grad_beta)，delta 在**第 0 位**，
    # 与 Layer/LayerNorm 的「末位是 delta」不同；Sequential/ResidualBlock 靠这个属性取
    delta_index = 0

    def train(self):
        """训练模式：forward 收到 batch 时使用 batch 统计量"""
        self.training = True
        return self

    def eval(self):
        """推理模式：forward 一律使用 running_mean / running_var"""
        self.training = False
        return self

    def forward(self, x):
        """推理：单样本；训练模式下若传入 batch 则自动走 forward_batch"""
        if x is None or len(x) == 0:
            raise ValueError("BatchNorm1d.forward: 输入不能为空")
        if isinstance(x[0], (list, tuple)):
            if len(x[0]) != self.d:
                raise ValueError(
                    f"BatchNorm1d.forward: 特征维度 {len(x[0])} 与本层 {self.d} 不一致"
                )
            if any(len(row) != self.d for row in x):
                raise ValueError("BatchNorm1d.forward: batch 内各样本长度不一致")
        elif len(x) != self.d:
            raise ValueError(
                f"BatchNorm1d.forward: 样本维度 {len(x)} 与本层 {self.d} 不一致"
            )
        if x and isinstance(x[0], (list, tuple)):
            if self.training:
                return self.forward_batch(x)
            # eval + batch：逐列用 running stats（原实现会退回单样本路径并 TypeError）
            self._cache_mode = 'eval'
            self.x_batch = [list(row) for row in x]
            return [[self.gamma[j] * ((row[j] - self.running_mean[j]) /
                    math.sqrt(self.running_var[j] + self.eps)) + self.beta[j]
                     for j in range(self.d)] for row in x]
        self._cache_mode = 'eval'
        self.x_batch = [list(x)]
        out = [0.0] * self.d
        for j in range(self.d):
            xh = (x[j] - self.running_mean[j]) / math.sqrt(self.running_var[j] + self.eps)
            out[j] = self.gamma[j] * xh + self.beta[j]
        return out

    def backward(self, dout_batch):
        """返回 (dx_batch, grad_gamma, grad_beta)

        按**上一次前向的模式**求导：
          训练模式（forward_batch）-> 用 batch 统计量的导数
          推理模式（eval / 单样本）-> 用 running 统计量的导数
        原先无条件套 batch 统计量，eval 之后反传会得到完全错误的 dx
        （实测差 4.3 倍），单样本前向后反传还会抛 TypeError。
        """
        if getattr(self, "_cache_mode", None) is None or self.x_batch is None:
            raise RuntimeError("BatchNorm1d.backward: 必须先调用 forward 或 forward_batch")
        if dout_batch and not isinstance(dout_batch[0], (list, tuple)):
            dout_batch = [dout_batch]          # 单样本 -> 当成 batch=1
        n, d = len(dout_batch), self.d
        if any(len(row) != d for row in dout_batch):
            raise ValueError("BatchNorm1d.backward: dout 的维度与本层特征数不一致")

        if self._cache_mode == 'eval':
            xs = self.x_batch
            if len(xs) != n:
                raise ValueError(
                    f"BatchNorm1d.backward: dout 有 {n} 个样本，"
                    f"但上一次前向是 {len(xs)} 个"
                )
            inv = [1.0 / math.sqrt(v + self.eps) for v in self.running_var]
            xhat = [[(x[j] - self.running_mean[j]) * inv[j] for j in range(d)] for x in xs]
            grad_gamma = [sum(dout_batch[i][j] * xhat[i][j] for i in range(n))
                          for j in range(d)]
            grad_beta = [sum(dout_batch[i][j] for i in range(n)) for j in range(d)]
            dx_batch = [[self.gamma[j] * inv[j] * dout_batch[i][j] for j in range(d)]
                        for i in range(n)]
            return dx_batch, grad_gamma, grad_beta

        n = self.n

        # 先转置成列，再交给 C 层的 sum/map（求和顺序不变，结果逐位一致）
        dout_t = list(zip(*dout_batch))
        xhat_t = list(zip(*self.x_hat))

        grad_gamma = [sum(map(_mul, dt, xt)) for dt, xt in zip(dout_t, xhat_t)]
        grad_beta = [sum(c) for c in dout_t]

        dx_hat = [[dout_batch[i][j] * self.gamma[j] for j in range(d)] for i in range(n)]

        # s1/s2 只依赖 j，必须提到 i 循环外面：
        # 原来写在 i 循环里，每个 (i, j) 都重算一遍长度为 n 的和 → O(n²·d)
        dx_hat_t = list(zip(*dx_hat))
        s1 = [sum(c) for c in dx_hat_t]
        s2 = [sum(map(_mul, c, xt)) for c, xt in zip(dx_hat_t, xhat_t)]

        dx_batch = [
            [self.inv_std[j] * (n * dx_hat[i][j] - s1[j] - self.x_hat[i][j] * s2[j]) / n
             for j in range(d)]
            for i in range(n)
        ]
        return dx_batch, grad_gamma, grad_beta


class BatchNorm2d:
    """二维批归一化：按通道在 H×W 上归一化（CNN 里紧跟 Conv2d 使用）

    训练: forward(x) 用**本样本**的 H×W 统计量，并更新 running 统计量；
    推理: forward(x) 用 running_mean / running_var（与 BatchNorm1d 语义一致）。
    """

    # backward 返回 (dx, grad_gamma, grad_beta)，delta 在第 0 位（同 BatchNorm1d）
    delta_index = 0

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        if isinstance(num_features, bool) or not isinstance(num_features, int) \
                or num_features < 1:
            raise ValueError(
                f"BatchNorm2d: num_features 必须是 >=1 的整数，收到 {num_features!r}")
        if not (0.0 <= momentum <= 1.0):
            raise ValueError(f"BatchNorm2d: momentum 必须在 [0,1] 内，收到 {momentum!r}")
        if eps <= 0:
            raise ValueError(f"BatchNorm2d: eps 必须 > 0，收到 {eps!r}")
        self.c = num_features
        self.eps = eps
        self.momentum = momentum
        self.gamma = [1.0] * self.c
        self.beta = [0.0] * self.c
        self.running_mean = [0.0] * self.c
        self.running_var = [1.0] * self.c
        self.training = True
        self.x_hat = None
        self.stats = None
        self._cache_mode = 'train'
        self.m_gamma = [0.0] * self.c
        self.v_gamma = [0.0] * self.c
        self.m_beta = [0.0] * self.c
        self.v_beta = [0.0] * self.c

    def train(self):
        """训练模式：forward 用当前样本的统计量"""
        self.training = True
        return self

    def eval(self):
        """推理模式：forward 用 running 统计量"""
        self.training = False
        return self

    def _check(self, x, who):
        if not isinstance(x, (list, tuple)) or not x:
            raise ValueError(f"{who}: 输入必须是非空的 (C, H, W)")
        if len(x) != self.c:
            raise ValueError(f"{who}: 通道数 {len(x)} != num_features={self.c}")
        h = len(x[0])
        if h == 0:
            raise ValueError(f"{who}: H 不能为 0")
        w = len(x[0][0])
        if w == 0:
            raise ValueError(f"{who}: W 不能为 0")
        for c in range(self.c):
            if len(x[c]) != h:
                raise ValueError(f"{who}: 各通道 H 不一致（通道 {c}）")
            for hh in range(h):
                if len(x[c][hh]) != w:
                    raise ValueError(f"{who}: 各通道 W 不一致（通道 {c}, 行 {hh}）")
        return h, w

    def forward(self, x):
        """(C, H, W) → (C, H, W)（训练模式更新 running 统计量）"""
        h, w = self._check(x, "BatchNorm2d.forward")
        m_size = h * w
        self._cache_mode = 'train' if self.training else 'eval'
        x_hat = []
        stats = []
        out = []
        for c in range(self.c):
            flat = [x[c][hh][ww] for hh in range(h) for ww in range(w)]
            if self.training:
                mean = sum(flat) / m_size
                var = sum((v - mean) ** 2 for v in flat) / m_size
                self.running_mean[c] = ((1 - self.momentum) * self.running_mean[c]
                                        + self.momentum * mean)
                unb = var * m_size / (m_size - 1) if m_size > 1 else var
                self.running_var[c] = ((1 - self.momentum) * self.running_var[c]
                                       + self.momentum * unb)
            else:
                mean = self.running_mean[c]
                var = self.running_var[c]
            inv = 1.0 / math.sqrt(var + self.eps)
            stats.append((mean, inv))
            xh = [(v - mean) * inv for v in flat]
            x_hat.append(xh)
            g, b = self.gamma[c], self.beta[c]
            out.append([[g * xh[hh * w + ww] + b for ww in range(w)]
                        for hh in range(h)])
        self.x_hat = x_hat
        self.stats = stats
        return out

    def backward(self, dout):
        """返回 (dx, grad_gamma, grad_beta)：dx 与该层输入同形 (C, H, W)"""
        if self.x_hat is None or self.stats is None:
            raise RuntimeError("BatchNorm2d.backward: 请先 forward")
        h, w = self._check(dout, "BatchNorm2d.backward")
        m_size = h * w
        grad_gamma = [0.0] * self.c
        grad_beta = [0.0] * self.c
        dx = []
        for c in range(self.c):
            mean, inv = self.stats[c]
            xh = self.x_hat[c]
            d_flat = [dout[c][hh][ww] for hh in range(h) for ww in range(w)]
            g = self.gamma[c]
            grad_gamma[c] = sum(d * a for d, a in zip(d_flat, xh))
            grad_beta[c] = sum(d_flat)
            dxh = [d * g for d in d_flat]
            if self._cache_mode == 'eval':
                # 推理模式的 mean/var 来自 running 统计量，与当前样本无关，
                # 反向就是逐元素的 inv*gamma（没有 m_size 那几项）——
                # 与 BatchNorm1d 的 eval 分支一致
                d_flat_out = [inv * d for d in dxh]
            else:
                sum_dxh = sum(dxh)
                sum_dxh_xh = sum(r * a for r, a in zip(dxh, xh))
                d_flat_out = [inv * (m_size * r - sum_dxh - a * sum_dxh_xh) / m_size
                              for r, a in zip(dxh, xh)]
            dx.append([[d_flat_out[hh * w + ww] for ww in range(w)] for hh in range(h)])
        return dx, grad_gamma, grad_beta


class LayerNorm:
    """层归一化：对单个样本的特征维度归一化，适合 Transformer / RNN"""

    def __init__(self, num_features: int, eps: float = 1e-5):
        self.d = num_features
        self.eps = eps
        self.gamma = [1.0] * self.d
        self.beta = [0.0] * self.d
        self.x = self.mean = self.var = self.inv_std = self.x_hat = None
        self.m_gamma = [0.0] * self.d; self.v_gamma = [0.0] * self.d
        self.m_beta = [0.0] * self.d; self.v_beta = [0.0] * self.d

    def forward(self, x):
        # 多出的特征会被静默丢弃，而 mean/var 仍是按全长算的，必须早失败
        if len(x) != self.d:
            raise ValueError(
                f"LayerNorm.forward: 特征维度 {len(x)} 与本层 {self.d} 不一致"
            )
        self.x = x
        self.mean = sum(x) / self.d
        self.var = sum((xi - self.mean) ** 2 for xi in x) / self.d
        self.inv_std = 1.0 / math.sqrt(self.var + self.eps)
        self.x_hat = [(xi - self.mean) * self.inv_std for xi in x]
        return [self.gamma[j] * self.x_hat[j] + self.beta[j] for j in range(self.d)]

    def backward(self, dout):
        if self.x_hat is None:
            raise RuntimeError("LayerNorm.backward: 必须先调用 forward()")
        if len(dout) != self.d:
            raise ValueError(
                f"LayerNorm.backward: dout 长度 {len(dout)} 与本层维度 {self.d} 不一致"
            )
        d = self.d
        grad_gamma = [dout[j] * self.x_hat[j] for j in range(d)]
        grad_beta = list(dout)

        dx_hat = [dout[j] * self.gamma[j] for j in range(d)]
        s1 = sum(dx_hat)
        s2 = sum(dx_hat[j] * self.x_hat[j] for j in range(d))

        dx = [self.inv_std * (d * dx_hat[j] - s1 - self.x_hat[j] * s2) / d for j in range(d)]
        return grad_gamma, grad_beta, dx  # 约定: 最后一个元素为传播的 delta


class GroupNorm:
    """分组归一化（GroupNorm，Wu & He 2018）：CNN 小 batch 时替代 BatchNorm

    输入是 (C, H, W)。把通道分成 num_groups 组，每组内对 (C/G, H, W) 的
    C/G*H*W 个元素一起求均值方差——与 batch 大小无关，所以 batch=1 也稳定。
    `InstanceNorm(num_channels)` 就是 `num_groups == num_channels` 的特例。
    """

    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-5):
        if isinstance(num_groups, bool) or not isinstance(num_groups, int) or num_groups < 1:
            raise ValueError(f"GroupNorm: num_groups 必须是正整数，收到 {num_groups!r}")
        if isinstance(num_channels, bool) or not isinstance(num_channels, int) or num_channels < 1:
            raise ValueError(f"GroupNorm: num_channels 必须是正整数，收到 {num_channels!r}")
        if num_channels % num_groups != 0:
            raise ValueError(
                f"GroupNorm: num_channels={num_channels} 必须能被 "
                f"num_groups={num_groups} 整除"
            )
        self.groups = num_groups
        self.channels = num_channels
        self.eps = eps
        self.gamma = [1.0] * num_channels
        self.beta = [0.0] * num_channels
        self.m_gamma = [0.0] * num_channels; self.v_gamma = [0.0] * num_channels
        self.m_beta = [0.0] * num_channels; self.v_beta = [0.0] * num_channels
        self.x = self.x_hat = None
        self.stats = []        # 每组 (mean, inv_std)

    def _check(self, x, who):
        if len(x) != self.channels:
            raise ValueError(
                f"GroupNorm.{who}: 通道数 {len(x)} 与本层 {self.channels} 不一致"
            )
        shape = (len(x[0]), len(x[0][0])) if x and x[0] else (0, 0)
        for c in range(len(x)):
            if len(x[c]) != shape[0] or any(len(r) != shape[1] for r in x[c]):
                raise ValueError(f"GroupNorm.{who}: 各通道的 H/W 必须一致")
        if shape[0] < 1 or shape[1] < 1:
            raise ValueError(f"GroupNorm.{who}: 输入不能为空")
        return shape

    def forward(self, x):
        H, W = self._check(x, "forward")
        self.x = x
        self.x_hat = [[[0.0] * W for _ in range(H)] for _ in range(self.channels)]
        self.stats = []
        per = self.channels // self.groups
        n = per * H * W
        for g in range(self.groups):
            mean = 0.0
            for c in range(g * per, (g + 1) * per):
                for h in range(H):
                    mean += sum(x[c][h])
            mean /= n
            var = 0.0
            for c in range(g * per, (g + 1) * per):
                for h in range(H):
                    for w in range(W):
                        d = x[c][h][w] - mean
                        var += d * d
            var /= n
            inv = 1.0 / math.sqrt(var + self.eps)
            self.stats.append((mean, inv))
            for c in range(g * per, (g + 1) * per):
                for h in range(H):
                    for w in range(W):
                        self.x_hat[c][h][w] = (x[c][h][w] - mean) * inv
        return [[[self.gamma[c] * self.x_hat[c][h][w] + self.beta[c]
                  for w in range(W)] for h in range(H)] for c in range(self.channels)]

    def backward(self, dout):
        if self.x_hat is None:
            raise RuntimeError("GroupNorm.backward: 必须先调用 forward()")
        H, W = self._check(dout, "backward")
        per = self.channels // self.groups
        n = per * H * W
        grad_gamma = [0.0] * self.channels
        grad_beta = [0.0] * self.channels
        for c in range(self.channels):
            s = 0.0
            for h in range(H):
                s += sum(dout[c][h])
            grad_beta[c] = s
            t = 0.0
            for h in range(H):
                for w in range(W):
                    t += dout[c][h][w] * self.x_hat[c][h][w]
            grad_gamma[c] = t
        dx = [[[0.0] * W for _ in range(H)] for _ in range(self.channels)]
        for g in range(self.groups):
            mean, inv = self.stats[g]
            s1 = 0.0     # sum(dx_hat)
            s2 = 0.0     # sum(dx_hat * x_hat)
            for c in range(g * per, (g + 1) * per):
                gam = self.gamma[c]
                for h in range(H):
                    for w in range(W):
                        d = dout[c][h][w] * gam
                        s1 += d
                        s2 += d * self.x_hat[c][h][w]
            for c in range(g * per, (g + 1) * per):
                gam = self.gamma[c]
                for h in range(H):
                    for w in range(W):
                        d = dout[c][h][w] * gam
                        dx[c][h][w] = inv * (n * d - s1 - self.x_hat[c][h][w] * s2) / n
        return grad_gamma, grad_beta, dx       # 约定：最后一个元素是 delta


class InstanceNorm(GroupNorm):
    """实例归一化 = 每组一个通道的 GroupNorm"""

    def __init__(self, num_channels: int, eps: float = 1e-5):
        super().__init__(num_channels, num_channels, eps)


class RMSNorm:
    """RMS 归一化（Llama / Qwen / Mistral 系）：y = gamma * x / sqrt(mean(x^2)+eps)

    与 LayerNorm 的差别：不减均值、没有 beta。少一趟减均值与一次加 beta，
    在 LLM 里更常见。
    """

    def __init__(self, num_features: int, eps: float = 1e-6):
        if num_features is None or int(num_features) < 1:
            raise ValueError(f"RMSNorm: num_features 必须 >= 1，收到 {num_features!r}")
        self.d = int(num_features)
        self.eps = eps
        self.gamma = [1.0] * self.d
        self.x = None
        self.inv_rms = None
        self.m_gamma = [0.0] * self.d
        self.v_gamma = [0.0] * self.d

    def forward(self, x):
        # 多出的特征会被静默丢弃，而 mean(x^2) 仍是按全长算的，必须早失败
        if len(x) != self.d:
            raise ValueError(
                f"RMSNorm.forward: 特征维度 {len(x)} 与本层 {self.d} 不一致"
            )
        self.x = x
        mean_sq = sum(xi * xi for xi in x) / self.d
        self.inv_rms = 1.0 / math.sqrt(mean_sq + self.eps)
        r = self.inv_rms
        return [self.gamma[j] * x[j] * r for j in range(self.d)]

    def backward(self, dout):
        if self.x is None:
            raise RuntimeError("RMSNorm.backward: 必须先调用 forward()")
        if len(dout) != self.d:
            raise ValueError(
                f"RMSNorm.backward: dout 长度 {len(dout)} 与本层维度 {self.d} 不一致"
            )
        d = self.d
        r = self.inv_rms
        x = self.x
        gamma = self.gamma
        grad_gamma = [dout[j] * x[j] * r for j in range(d)]
        # g = Σ_i dout_i * gamma_i * x_i；dx_j = gamma_j*dout_j/r - g*x_j/(d*r^3)
        g = sum(dout[i] * gamma[i] * x[i] for i in range(d))
        r3 = r * r * r
        dx = [gamma[j] * dout[j] * r - g * x[j] * r3 / d for j in range(d)]
        return grad_gamma, dx  # 约定: 最后一个元素为传播的 delta


# ============================================================================
#  RNN / LSTM
# ============================================================================

class _StepCacheMixin:
    """BPTT 辅助：把 cell 的前向缓存做成快照，反向时直接恢复，避免重跑一遍前向。

    cell 的 forward 会把中间量写进 self（x / h_prev / z / 各个门…），backward 依赖它们。
    若只保存 hidden state，反向时就必须对每个时间步重新 forward 一次来"恢复缓存"。
    这里把需要的属性列进 _CACHE_FIELDS，RNN/LSTM/GRU 的 forward 会把每一步、每一层的
    快照存进 self.step_cache，训练循环里直接

        cell.restore_cache(step_cache[t][layer])

    即可复用，省掉整轮重复前向。
    """

    _CACHE_FIELDS = ()

    def cache_snapshot(self):
        return tuple(getattr(self, name) for name in self._CACHE_FIELDS)

    def restore_cache(self, snapshot):
        for name, value in zip(self._CACHE_FIELDS, snapshot):
            setattr(self, name, value)


class RNNCell(_StepCacheMixin):
    """单步 RNN 单元: h_t = tanh(W_ih·x_t + b_ih + W_hh·h_{t-1} + b_hh)"""

    _CACHE_FIELDS = ("x", "h_prev", "z")

    def __init__(self, input_size: int, hidden_size: int):
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Xavier 初始化
        std_ih = math.sqrt(1.0 / input_size)
        std_hh = math.sqrt(1.0 / hidden_size)

        self.W_ih = [[random.gauss(0, std_ih) for _ in range(input_size)] for _ in range(hidden_size)]
        self.W_hh = [[random.gauss(0, std_hh) for _ in range(hidden_size)] for _ in range(hidden_size)]
        self.b_ih = [0.0] * hidden_size
        self.b_hh = [0.0] * hidden_size

        # Adam 状态
        self._init_adam()

        # 梯度缓冲区（由 zero_grad 初始化）
        self.zero_grad()

        # 缓存
        self.x = None
        self.h_prev = None
        self.z = None  # 激活前

    def zero_grad(self):
        """初始化/清零所有参数梯度"""
        self.grad_W_ih = [[0.0] * self.input_size for _ in range(self.hidden_size)]
        self.grad_W_hh = [[0.0] * self.hidden_size for _ in range(self.hidden_size)]
        self.grad_b_ih = [0.0] * self.hidden_size
        self.grad_b_hh = [0.0] * self.hidden_size

    def _init_adam(self):
        def zeros(shape):
            if isinstance(shape[0], list):
                return [zeros(row) for row in shape]
            return [0.0] * len(shape)

        self.m_W_ih = zeros(self.W_ih)
        self.v_W_ih = zeros(self.W_ih)
        self.m_W_hh = zeros(self.W_hh)
        self.v_W_hh = zeros(self.W_hh)
        self.m_b_ih = [0.0] * self.hidden_size
        self.v_b_ih = [0.0] * self.hidden_size
        self.m_b_hh = [0.0] * self.hidden_size
        self.v_b_hh = [0.0] * self.hidden_size

    def forward(self, x, h_prev=None):
        """x: (input_size,)  h_prev: (hidden_size,)  -> h: (hidden_size,)"""
        self.x = x
        self.h_prev = h_prev if h_prev is not None else [0.0] * self.hidden_size

        self.z = [0.0] * self.hidden_size
        for j in range(self.hidden_size):
            s = self.b_ih[j] + self.b_hh[j]
            for i in range(self.input_size):
                s += self.W_ih[j][i] * x[i]
            for i in range(self.hidden_size):
                s += self.W_hh[j][i] * self.h_prev[i]
            self.z[j] = s

        # tanh
        h = [math.tanh(zj) for zj in self.z]
        return h

    def backward(self, dh):
        """
        dh: (hidden_size,) 输出梯度
        返回 (dx, dh_prev), 内部累积梯度
        """
        # tanh 导数: 1 - tanh^2
        d_tanh = [1.0 - math.tanh(zj) ** 2 for zj in self.z]
        dz = [dh[j] * d_tanh[j] for j in range(self.hidden_size)]

        # dx
        dx = [0.0] * self.input_size
        for i in range(self.input_size):
            s = 0.0
            for j in range(self.hidden_size):
                s += self.W_ih[j][i] * dz[j]
            dx[i] = s

        # dh_prev
        dh_prev = [0.0] * self.hidden_size
        for i in range(self.hidden_size):
            s = 0.0
            for j in range(self.hidden_size):
                s += self.W_hh[j][i] * dz[j]
            dh_prev[i] = s

        # 累积梯度
        for j in range(self.hidden_size):
            self.grad_b_ih[j] += dz[j]
            self.grad_b_hh[j] += dz[j]
            for i in range(self.input_size):
                self.grad_W_ih[j][i] += dz[j] * self.x[i]
            for i in range(self.hidden_size):
                self.grad_W_hh[j][i] += dz[j] * self.h_prev[i]

        return dx, dh_prev

    def get_grads(self):
        return self.grad_W_ih, self.grad_W_hh, self.grad_b_ih, self.grad_b_hh


class LSTMCell(_StepCacheMixin):
    """单步 LSTM 单元 (无 peephole)"""

    _CACHE_FIELDS = ("x", "h_prev", "c_prev", "i", "f", "g", "o", "c")

    def __init__(self, input_size: int, hidden_size: int):
        self.input_size = input_size
        self.hidden_size = hidden_size

        # 合并 W_ih (4*hidden_size, input_size) 和 W_hh (4*hidden_size, hidden_size)
        gate_size = 4 * hidden_size
        std_ih = math.sqrt(1.0 / input_size)
        std_hh = math.sqrt(1.0 / hidden_size)

        self.W_ih = [[random.gauss(0, std_ih) for _ in range(input_size)] for _ in range(gate_size)]
        self.W_hh = [[random.gauss(0, std_hh) for _ in range(hidden_size)] for _ in range(gate_size)]
        self.b_ih = [0.0] * gate_size
        self.b_hh = [0.0] * gate_size

        self._init_adam()

        # 梯度缓冲区（由 zero_grad 初始化）
        self.zero_grad()

        # 缓存
        self.x = None
        self.h_prev = None
        self.c_prev = None
        self.i = self.f = self.g = self.o = None  # gates
        self.c = None  # cell state

    def zero_grad(self):
        """初始化/清零所有参数梯度"""
        hs = self.hidden_size
        gate_size = 4 * hs
        self.grad_W_ih = [[0.0] * self.input_size for _ in range(gate_size)]
        self.grad_W_hh = [[0.0] * hs for _ in range(gate_size)]
        self.grad_b_ih = [0.0] * gate_size
        self.grad_b_hh = [0.0] * gate_size

    def _init_adam(self):
        gate_size = 4 * self.hidden_size

        def zeros(shape):
            if isinstance(shape[0], list):
                return [zeros(row) for row in shape]
            return [0.0] * len(shape)

        self.m_W_ih = zeros(self.W_ih)
        self.v_W_ih = zeros(self.W_ih)
        self.m_W_hh = zeros(self.W_hh)
        self.v_W_hh = zeros(self.W_hh)
        self.m_b_ih = [0.0] * gate_size
        self.v_b_ih = [0.0] * gate_size
        self.m_b_hh = [0.0] * gate_size
        self.v_b_hh = [0.0] * gate_size

    def _sigmoid(self, x):
        # 分支写法：xi<0 时用 exp(xi)/(1+exp(xi))，避免 math.exp(-xi) 在大负数上溢出
        return [1.0 / (1.0 + math.exp(-xi)) if xi >= 0
            else math.exp(xi) / (1.0 + math.exp(xi)) for xi in x]

    def forward(self, x, h_prev=None, c_prev=None):
        """x: (input_size,) -> h: (hidden_size,), c: (hidden_size,)"""
        self.x = x
        hs = self.hidden_size
        self.h_prev = h_prev if h_prev is not None else [0.0] * hs
        self.c_prev = c_prev if c_prev is not None else [0.0] * hs

        # 预激活: gates = W_ih·x + b_ih + W_hh·h_prev + b_hh
        gates = [0.0] * (4 * hs)
        for j in range(4 * hs):
            s = self.b_ih[j] + self.b_hh[j]
            for i in range(self.input_size):
                s += self.W_ih[j][i] * x[i]
            for i in range(hs):
                s += self.W_hh[j][i] * self.h_prev[i]
            gates[j] = s

        # 分解四个门
        i_gate = gates[0:hs]
        f_gate = gates[hs:2 * hs]
        g_gate = gates[2 * hs:3 * hs]
        o_gate = gates[3 * hs:4 * hs]

        self.i = self._sigmoid(i_gate)  # input gate
        self.f = self._sigmoid(f_gate)  # forget gate
        self.g = [math.tanh(zj) for zj in g_gate]  # cell gate
        self.o = self._sigmoid(o_gate)  # output gate

        # cell state: c = f * c_prev + i * g
        self.c = [self.f[j] * self.c_prev[j] + self.i[j] * self.g[j] for j in range(hs)]

        # hidden: h = o * tanh(c)
        h = [self.o[j] * math.tanh(self.c[j]) for j in range(hs)]

        return h, self.c

    def backward(self, dh, dc_next=None):
        """
        dh: (hidden_size,) 输出梯度
        dc_next: (hidden_size,) 来自下一时间步的 cell gradient
        返回 (dx, dh_prev, dc_prev)
        """
        hs = self.hidden_size
        dc_next = dc_next if dc_next is not None else [0.0] * hs

        # dc = dh * o * (1 - tanh^2(c)) + dc_next
        dc = [0.0] * hs
        for j in range(hs):
            tanh_c = math.tanh(self.c[j])
            dc[j] = dh[j] * self.o[j] * (1.0 - tanh_c * tanh_c) + dc_next[j]

        # 各门梯度
        di = [dc[j] * self.g[j] * self.i[j] * (1.0 - self.i[j]) for j in range(hs)]
        df = [dc[j] * self.c_prev[j] * self.f[j] * (1.0 - self.f[j]) for j in range(hs)]
        dg = [dc[j] * self.i[j] * (1.0 - self.g[j] * self.g[j]) for j in range(hs)]
        do = [dh[j] * math.tanh(self.c[j]) * self.o[j] * (1.0 - self.o[j]) for j in range(hs)]

        d_gates = di + df + dg + do  # (4*hidden_size,)

        # 累积梯度
        for j in range(4 * hs):
            self.grad_b_ih[j] += d_gates[j]
            self.grad_b_hh[j] += d_gates[j]
            for i in range(self.input_size):
                self.grad_W_ih[j][i] += d_gates[j] * self.x[i]
            for i in range(hs):
                self.grad_W_hh[j][i] += d_gates[j] * self.h_prev[i]

        # dx
        dx = [0.0] * self.input_size
        for i in range(self.input_size):
            s = 0.0
            for j in range(4 * hs):
                s += self.W_ih[j][i] * d_gates[j]
            dx[i] = s

        # dh_prev
        dh_prev = [0.0] * hs
        for i in range(hs):
            s = 0.0
            for j in range(4 * hs):
                s += self.W_hh[j][i] * d_gates[j]
            dh_prev[i] = s

        # dc_prev
        dc_prev = [dc[j] * self.f[j] for j in range(hs)]

        return dx, dh_prev, dc_prev

    def get_grads(self):
        return self.grad_W_ih, self.grad_W_hh, self.grad_b_ih, self.grad_b_hh


class RNN:
    """多层 RNN（堆叠多个 RNNCell）"""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.cells = []
        for l in range(num_layers):
            sz_in = input_size if l == 0 else hidden_size
            self.cells.append(RNNCell(sz_in, hidden_size))

    def forward(self, x_seq, h0=None):
        """
        x_seq: List[(input_size,)]  时间序列
        h0: List[(hidden_size,)] 每层初始 hidden
        返回 outputs: List[(hidden_size,)]  最后一层全部时间步
        """
        if h0 is None:
            h0 = [[0.0] * self.hidden_size for _ in range(self.num_layers)]

        outputs = []
        h = list(h0)
        self.step_cache = []  # 每步每层的前向缓存快照，BPTT 时可 restore_cache 复用
        for xt in x_seq:
            layer_in = xt
            step = []
            for l in range(self.num_layers):
                h[l] = self.cells[l].forward(layer_in, h[l])
                step.append(self.cells[l].cache_snapshot())
                layer_in = h[l]
            self.step_cache.append(step)
            outputs.append(list(h[-1]))
        return outputs, h


class LSTM:
    """多层 LSTM（堆叠多个 LSTMCell）"""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.cells = []
        for l in range(num_layers):
            sz_in = input_size if l == 0 else hidden_size
            self.cells.append(LSTMCell(sz_in, hidden_size))

    def forward(self, x_seq, h0=None, c0=None):
        """
        x_seq: List[(input_size,)]
        h0, c0: List[(hidden_size,)] 每层初始状态
        返回 outputs, (h_n, c_n)
        """
        if h0 is None:
            h0 = [[0.0] * self.hidden_size for _ in range(self.num_layers)]
        if c0 is None:
            c0 = [[0.0] * self.hidden_size for _ in range(self.num_layers)]

        outputs = []
        h = list(h0)
        c = list(c0)
        self.step_cache = []  # 每步每层的前向缓存快照（含 c / 各门），BPTT 时复用
        for xt in x_seq:
            layer_in = xt
            step = []
            for l in range(self.num_layers):
                h[l], c[l] = self.cells[l].forward(layer_in, h[l], c[l])
                step.append(self.cells[l].cache_snapshot())
                layer_in = h[l]
            self.step_cache.append(step)
            outputs.append(list(h[-1]))
        return outputs, (h, c)


# ============================================================================
#  Matrix helpers (用于 Transformer)
# ============================================================================

def _matmul(A, B):
    """矩阵乘法 A(m×n) × B(n×p) → (m×p)，纯 Python 实现"""
    m, n = len(A), len(A[0])
    n2, p = len(B), len(B[0])
    if n != n2:
        # 不能用 assert：python -O 会把它去掉，错误形状会被静默截断
        raise ValueError(f"_matmul: 维度不匹配 ({m},{n}) × ({n2},{p})")
    # 先把 B 转置一次，让每一列成为元组，再用 C 层的 map(mul) / sum 做点积：
    # 与逐个元素 += 的三重循环结果完全一致，但快约 1.6 倍
    Bt = list(zip(*B))
    return [[sum(map(_mul, row, col)) for col in Bt] for row in A]


def _transpose(M):
    """转置 (m×n) → (n×m)"""
    m, n = len(M), len(M[0])
    return [[M[i][j] for i in range(m)] for j in range(n)]


def _softmax_row(v):
    """对一维列表做 softmax（整行都是 -inf 时返回全 0，避免 exp(-inf-(-inf))=NaN）"""
    mx = max(v)
    if mx == float("-inf"):
        return [0.0] * len(v)
    exps = [math.exp(x - mx) for x in v]
    s = sum(exps)
    return [e / s for e in exps]


# ============================================================================
#  GRU (Gated Recurrent Unit)
# ============================================================================

class GRUCell(_StepCacheMixin):
    """单步 GRU 单元

    z_t = σ(W_iz·x_t + b_iz + W_hz·h_{t-1} + b_hz)        (update gate)
    r_t = σ(W_ir·x_t + b_ir + W_hr·h_{t-1} + b_hr)        (reset gate)
    n_t = tanh(W_in·x_t + b_in + r_t ⊙ (W_hn·h_{t-1} + b_hn))  (new gate)
    h_t = (1 - z_t) ⊙ n_t + z_t ⊙ h_{t-1}
    """

    _CACHE_FIELDS = ("x", "h_prev", "z", "r", "n_ih", "n_hh", "z_a", "r_a", "n_val")

    def __init__(self, input_size: int, hidden_size: int):
        self.input_size = input_size
        self.hidden_size = hidden_size
        hs, ins = hidden_size, input_size

        std_ih = math.sqrt(1.0 / ins)
        std_hh = math.sqrt(1.0 / hs)

        self.W_ih = [[random.gauss(0, std_ih) for _ in range(ins)] for _ in range(3 * hs)]
        self.W_hh = [[random.gauss(0, std_hh) for _ in range(hs)] for _ in range(3 * hs)]
        self.b_ih = [0.0] * (3 * hs)
        self.b_hh = [0.0] * (3 * hs)

        self._init_adam()

        # 梯度缓冲区（由 zero_grad 初始化）
        self.zero_grad()

        # 缓存
        self.x = self.h_prev = None
        self.z = self.r = None                      # 两组门预激活值（n 门无预激活变量）
        self.z_a = self.r_a = None                   # sigmoid(z), sigmoid(r)
        self.n_val = None                            # tanh(n_ih + r_a * n_hh)
        self.n_ih = self.n_hh = None                 # n 的 ih 和 hh 分量（分离存储用于反向）

    def zero_grad(self):
        """初始化/清零所有参数梯度"""
        hs = self.hidden_size
        self.grad_W_ih = [[0.0] * self.input_size for _ in range(3 * hs)]
        self.grad_W_hh = [[0.0] * hs for _ in range(3 * hs)]
        self.grad_b_ih = [0.0] * (3 * hs)
        self.grad_b_hh = [0.0] * (3 * hs)

    def _init_adam(self):
        hs = self.hidden_size
        def zeros(shape):
            if isinstance(shape[0], list):
                return [zeros(row) for row in shape]
            return [0.0] * len(shape)
        self.m_W_ih = zeros(self.W_ih); self.v_W_ih = zeros(self.W_ih)
        self.m_W_hh = zeros(self.W_hh); self.v_W_hh = zeros(self.W_hh)
        self.m_b_ih = [0.0] * (3 * hs); self.v_b_ih = [0.0] * (3 * hs)
        self.m_b_hh = [0.0] * (3 * hs); self.v_b_hh = [0.0] * (3 * hs)

    def _sigmoid(self, x):
        # 分支写法：xi<0 时用 exp(xi)/(1+exp(xi))，避免 math.exp(-xi) 在大负数上溢出
        return [1.0 / (1.0 + math.exp(-xi)) if xi >= 0
            else math.exp(xi) / (1.0 + math.exp(xi)) for xi in x]

    def forward(self, x, h_prev=None):
        hs = self.hidden_size
        self.x = x
        self.h_prev = h_prev if h_prev is not None else [0.0] * hs

        # 分离三组门权重: [z: 0:hs], [r: hs:2*hs], [n: 2*hs:3*hs]
        def _linear(W_slice, b_slice, inp, size):
            out = [b_slice[j] for j in range(size)]
            for j in range(size):
                for i in range(len(inp)):
                    out[j] += W_slice[j][i] * inp[i]
            return out

        z_ih = _linear(self.W_ih[:hs], self.b_ih[:hs], x, hs)
        z_hh = _linear(self.W_hh[:hs], self.b_hh[:hs], self.h_prev, hs)
        self.z = [z_ih[j] + z_hh[j] for j in range(hs)]

        r_ih = _linear(self.W_ih[hs:2*hs], self.b_ih[hs:2*hs], x, hs)
        r_hh = _linear(self.W_hh[hs:2*hs], self.b_hh[hs:2*hs], self.h_prev, hs)
        self.r = [r_ih[j] + r_hh[j] for j in range(hs)]

        self.n_ih = _linear(self.W_ih[2*hs:3*hs], self.b_ih[2*hs:3*hs], x, hs)
        self.n_hh = _linear(self.W_hh[2*hs:3*hs], self.b_hh[2*hs:3*hs], self.h_prev, hs)

        self.z_a = self._sigmoid(self.z)
        self.r_a = self._sigmoid(self.r)
        self.n_val = [math.tanh(self.n_ih[j] + self.r_a[j] * self.n_hh[j]) for j in range(hs)]

        h = [(1 - self.z_a[j]) * self.n_val[j] + self.z_a[j] * self.h_prev[j] for j in range(hs)]
        return h

    def backward(self, dh):
        """BPTT 单步反向，返回 (dx, dh_prev)，内部累积梯度"""
        hs = self.hidden_size

        # h = (1 - z_a) * n_val + z_a * h_prev
        dz_a = [dh[j] * (self.h_prev[j] - self.n_val[j]) for j in range(hs)]
        dn = [dh[j] * (1 - self.z_a[j]) for j in range(hs)]
        dh_prev_direct = [dh[j] * self.z_a[j] for j in range(hs)]

        # n_val = tanh(n_ih + r_a * n_hh)
        dn_pre = [dn[j] * (1 - self.n_val[j] * self.n_val[j]) for j in range(hs)]
        d_n_ih = list(dn_pre)
        d_n_hh = [dn_pre[j] * self.r_a[j] for j in range(hs)]
        dr_a_from_n = [dn_pre[j] * self.n_hh[j] for j in range(hs)]

        # r_a = sigmoid(r)：r_a 不受 z 直接影响，梯度仅来自 n
        dr = [dr_a_from_n[j] * self.r_a[j] * (1 - self.r_a[j]) for j in range(hs)]

        # z_a = sigmoid(z)
        dz = [dz_a[j] * self.z_a[j] * (1 - self.z_a[j]) for j in range(hs)]

        # 累积参数梯度
        # 更新门 z: rows [0:hs]
        for j in range(hs):
            self.grad_b_ih[j] += dz[j]
            self.grad_b_hh[j] += dz[j]
            for i in range(self.input_size):
                self.grad_W_ih[j][i] += dz[j] * self.x[i]
            for i in range(hs):
                self.grad_W_hh[j][i] += dz[j] * self.h_prev[i]

        # 重置门 r: rows [hs:2*hs]
        for j in range(hs):
            rj = hs + j
            self.grad_b_ih[rj] += dr[j]
            self.grad_b_hh[rj] += dr[j]
            for i in range(self.input_size):
                self.grad_W_ih[rj][i] += dr[j] * self.x[i]
            for i in range(hs):
                self.grad_W_hh[rj][i] += dr[j] * self.h_prev[i]

        # 新门 n: rows [2*hs:3*hs]
        for j in range(hs):
            nj = 2 * hs + j
            self.grad_b_ih[nj] += d_n_ih[j]
            self.grad_b_hh[nj] += d_n_hh[j]
            for i in range(self.input_size):
                self.grad_W_ih[nj][i] += d_n_ih[j] * self.x[i]
            for i in range(hs):
                self.grad_W_hh[nj][i] += d_n_hh[j] * self.h_prev[i]

        # dx 和 dh_prev 通过三个门权重反传
        # 注: dx 经过 n_ih (W_ih_n)，dh_prev 经过 n_hh (W_hh_n)，需用不同的 n-门梯度分量
        d_all_gates = [0.0] * (3 * hs)
        for j in range(hs):
            d_all_gates[j] = dz[j]
            d_all_gates[hs + j] = dr[j]
            d_all_gates[2 * hs + j] = d_n_ih[j]       # dx 走 n_ih 路径
        dx = [sum(self.W_ih[j][i] * d_all_gates[j] for j in range(3 * hs)) for i in range(self.input_size)]

        # dh_prev: 直接路径 h = (1-z)*n + z*h_prev → dh * z  + 通过三个门 hh 权重反传
        for j in range(hs):
            d_all_gates[2 * hs + j] = d_n_hh[j]       # dh_prev 走 n_hh 路径（含 r_a 因子）
        dh_prev = [dh_prev_direct[i] + sum(self.W_hh[j][i] * d_all_gates[j] for j in range(3 * hs)) for i in range(hs)]

        return dx, dh_prev

    def get_grads(self):
        return self.grad_W_ih, self.grad_W_hh, self.grad_b_ih, self.grad_b_hh


class GRU:
    """多层 GRU（堆叠多个 GRUCell）"""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.cells = []
        for l in range(num_layers):
            sz_in = input_size if l == 0 else hidden_size
            self.cells.append(GRUCell(sz_in, hidden_size))

    def forward(self, x_seq, h0=None):
        if h0 is None:
            h0 = [[0.0] * self.hidden_size for _ in range(self.num_layers)]

        outputs = []
        h = list(h0)
        self.step_cache = []  # 每步每层的前向缓存快照，BPTT 时可 restore_cache 复用
        for xt in x_seq:
            layer_in = xt
            step = []
            for l in range(self.num_layers):
                h[l] = self.cells[l].forward(layer_in, h[l])
                step.append(self.cells[l].cache_snapshot())
                layer_in = h[l]
            self.step_cache.append(step)
            outputs.append(list(h[-1]))
        return outputs, h


# ============================================================================
#  Transformer 组件
# ============================================================================

class MultiHeadAttention:
    """多头自注意力 (Multi-Head Self-Attention)

    输入: (seq_len, d_model) 的序列
    输出: (seq_len, d_model)
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0,
                 causal: bool = False, rope: bool = False, rope_base: float = 10000.0,
                 window: int = None, num_kv_heads: int = None, alibi: bool = False):
        """dropout: 注意力权重上的 dropout（训练模式下生效，eval() 关闭）。

        实现方式是「对 softmax 后的权重逐元素做 inverted dropout」：训练时每个
        权重以 1/(1-p) 的概率保留并放大，推理时不缩放。反向要乘同一个 mask。

        causal: 为 True 时加因果掩码——位置 t 只能看到 0..t，看不到未来。
        这是自回归语言模型（GPT 式）的必要条件；默认 False 保持双向注意力。

        window: 滑动窗口（Mistral 式局部因果）。设了就隐含因果——位置 t 只能看到
        最近 window 个位置（含自己），更早的一律屏蔽；显存/算力从 O(T²) 降到
        O(T·window)。默认 None 表示全注意力，行为与旧版逐位一致。

        rope: 为 True 时给 Q/K 加旋转位置编码（RoPE，Llama 系）。旋转是正交变换，
        只改"方向"不改模长，attention 分数里自然带上相对位置信息，外推性比
        固定 sin/cos 绝对位置编码好得多。默认 False 时与原行为逐位一致。

        num_kv_heads: GQA / MQA 的 KV 头数（Llama-2 70B、Mistral 用）。设成小于
        num_heads 时，多个 Q 头共享同一组 K/V：`W_k`/`W_v` 的输出维度从 d_model
        缩到 num_kv_heads*d_k，KV cache 同比例变小。默认 None 等于 num_heads，
        即普通 MHA——此时所有张量形状、随机数抽取顺序都与旧版一致。

        alibi: 为 True 时给注意力分数加 ALiBi 线性偏置（Attention with Linear
        Biases）：第 h 个头加 `slope_h * (t - j)`，slope_h = 2^(-8*(h+1)/num_heads)。
        距离越远惩罚越大，天然带「近处优先」，且无需位置编码就能外推更长的序列。
        偏置是常数、不参与求导，所以反向不需要额外分支。
        """
        if not (0.0 <= dropout < 1.0):
            raise ValueError(f"MultiHeadAttention: dropout 必须在 [0,1) 内，收到 {dropout!r}")
        self.dropout = dropout
        self.causal = bool(causal)
        self.rope = bool(rope)
        self.rope_base = float(rope_base)
        # 滑动窗口（Mistral 式局部因果）：只看最近 window 个位置（含自己）。
        # 设了 window 就隐含因果——未来位置一律屏蔽；None 表示全注意力。
        if window is None:
            self.window = None
        else:
            if isinstance(window, bool) or not isinstance(window, int):
                raise ValueError(
                    f"MultiHeadAttention: window 必须是正整数或 None，收到 {window!r}"
                )
            if window < 1:
                raise ValueError(f"MultiHeadAttention: window 必须 >= 1，收到 {window!r}")
            self.window = window
        self.alibi = bool(alibi)
        self.training = True
        self.rng = random            # 可注入（测试里换成 random.Random(seed) 复现 mask）
        # 每次 forward 重建；反向只读这几个列表
        self.attn_dropped = []       # 实际参与 attn@V 的权重（含 dropout 缩放）
        self.attn_masks = []         # 对应掩码；None 表示本次没做 dropout
        if d_model % num_heads != 0:
            raise ValueError(
                f"MultiHeadAttention: d_model={d_model} 必须能被 num_heads={num_heads} 整除"
            )
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.dropout_p = dropout
        # GQA / MQA：KV 头数可以少于 Q 头数，每 num_heads/num_kv_heads 个 Q 头
        # 共享同一组 K/V。None = num_heads（普通 MHA，形状与旧版完全一致）。
        if num_kv_heads is None:
            num_kv_heads = num_heads
        if isinstance(num_kv_heads, bool) or not isinstance(num_kv_heads, int):
            raise ValueError(
                f"MultiHeadAttention: num_kv_heads 必须是正整数，收到 {num_kv_heads!r}"
            )
        if num_kv_heads < 1 or num_kv_heads > num_heads:
            raise ValueError(
                f"MultiHeadAttention: num_kv_heads 必须在 1..{num_heads} 内，收到 {num_kv_heads}"
            )
        if num_heads % num_kv_heads != 0:
            raise ValueError(
                f"MultiHeadAttention: num_heads={num_heads} 必须能被 "
                f"num_kv_heads={num_kv_heads} 整除"
            )
        self.num_kv_heads = num_kv_heads
        self.kv_dim = num_kv_heads * self.d_k   # W_k/W_v 的输出维度（KV cache 宽度）
        self.kv_group = num_heads // num_kv_heads
        # RoPE 的成对频率：第 i 对用 base^(-2i/d_k)，与 Llama 一致
        self.rope_freqs = [1.0 / (self.rope_base ** (2.0 * i / self.d_k))
                           for i in range(self.d_k // 2)]
        # ALiBi 的逐头斜率：2^(-8*(h+1)/num_heads)，与论文/CausalLM 实现一致
        self.alibi_slopes = [2.0 ** (-8.0 * (h + 1) / num_heads)
                             for h in range(num_heads)]

        std = math.sqrt(2.0 / d_model)
        self.W_q = [[random.gauss(0, std) for _ in range(d_model)] for _ in range(d_model)]
        self.W_k = [[random.gauss(0, std) for _ in range(d_model)] for _ in range(self.kv_dim)]
        self.W_v = [[random.gauss(0, std) for _ in range(d_model)] for _ in range(self.kv_dim)]
        self.W_o = [[random.gauss(0, std) for _ in range(d_model)] for _ in range(d_model)]

        self.b_q = [0.0] * d_model
        self.b_k = [0.0] * self.kv_dim
        self.b_v = [0.0] * self.kv_dim
        self.b_o = [0.0] * d_model

        self._init_adam()
        self.zero_grad()

        # 缓存
        self.x = None       # (seq_len, d_model)
        self.seq_len = 0
        self.Q = self.K = self.V = None           # per-head: List[(seq_len, d_k)] × num_heads
        self.attn_weights = None   # List[(seq_len, seq_len)] × num_heads
        self.head_out = None       # List[(seq_len, d_k)] × num_heads
        self.concat = None         # (seq_len, d_model)

    def zero_grad(self):
        """初始化/清零所有参数梯度"""
        d = self.d_model
        def zeros_2d(r, c):
            return [[0.0] * c for _ in range(r)]
        kv = self.kv_dim
        self.grads = {
            "W_q": zeros_2d(d, d), "b_q": [0.0] * d,
            "W_k": zeros_2d(kv, d), "b_k": [0.0] * kv,
            "W_v": zeros_2d(kv, d), "b_v": [0.0] * kv,
            "W_o": zeros_2d(d, d), "b_o": [0.0] * d,
        }

    def _init_adam(self):
        d = self.d_model
        def zeros_2d(r, c):
            return [[0.0] * c for _ in range(r)]
        kv = self.kv_dim
        self.m_W_q = zeros_2d(d, d); self.v_W_q = zeros_2d(d, d)
        self.m_W_k = zeros_2d(kv, d); self.v_W_k = zeros_2d(kv, d)
        self.m_W_v = zeros_2d(kv, d); self.v_W_v = zeros_2d(kv, d)
        self.m_W_o = zeros_2d(d, d); self.v_W_o = zeros_2d(d, d)
        self.m_b_q = [0.0] * d; self.v_b_q = [0.0] * d
        self.m_b_k = [0.0] * kv; self.v_b_k = [0.0] * kv
        self.m_b_v = [0.0] * kv; self.v_b_v = [0.0] * kv
        self.m_b_o = [0.0] * d; self.v_b_o = [0.0] * d

    def train(self):
        """训练模式：注意力权重 dropout 生效"""
        self.training = True
        return self

    def eval(self):
        """推理模式：关闭 dropout（与 Dropout 的约定一致）"""
        self.training = False
        return self

    def _rope_apply(self, vec, pos: int, sign: float = 1.0):
        """对单个头的向量做 RoPE 旋转；sign=-1 即逆旋转（反向传播用）

        第 i 维与第 i+half 维配对（GPT-NeoX 版式）。d_k 为奇数时中间那维不旋转。
        """
        half = self.d_k // 2
        out = list(vec)
        for i in range(half):
            theta = pos * self.rope_freqs[i]
            c = math.cos(theta)
            s = sign * math.sin(theta)
            x0 = vec[i]
            x1 = vec[i + half]
            out[i] = x0 * c - x1 * s
            out[i + half] = x0 * s + x1 * c
        return out

    def forward_step(self, x_t, cache):
        """单 token 增量前向（推理专用，不写反向缓存）

        cache: dict，本层一份；存 `K`/`V`（每头一个 [t][d_k] 列表）与 `pos`。
        只对"没在滑动窗口里"的情况保证与 forward 的最后一行逐位一致：
        位置编码/因果掩码都依赖绝对位置，窗口滑动后必须重建 cache。
        """
        d = self.d_model
        h = self.num_heads
        d_k = self.d_k
        kv = self.kv_dim
        q = [sum(map(_mul, self.W_q[j], x_t)) + self.b_q[j] for j in range(d)]
        k = [sum(map(_mul, self.W_k[j], x_t)) + self.b_k[j] for j in range(kv)]
        v = [sum(map(_mul, self.W_v[j], x_t)) + self.b_v[j] for j in range(kv)]
        pos = cache.get("pos", 0)
        cache["pos"] = pos + 1
        n_kv = self.num_kv_heads
        K = cache.setdefault("K", [[] for _ in range(n_kv)])
        V = cache.setdefault("V", [[] for _ in range(n_kv)])
        scale = 1.0 / math.sqrt(d_k)
        concat = [0.0] * d
        for head in range(h):
            off = head * d_k
            # GQA：每 kv_group 个 Q 头共享一组 K/V（cache 也按 kv 头存）
            kv_head = head // self.kv_group
            kv_off = kv_head * d_k
            qh = q[off:off + d_k]
            kh = k[kv_off:kv_off + d_k]
            vh = v[kv_off:kv_off + d_k]
            if self.rope:
                qh = self._rope_apply(qh, pos)
                kh = self._rope_apply(kh, pos)
            if len(K[kv_head]) == pos:
                # 同一组的第一个 Q 头负责写入，后续 Q 头复用（不重复 append）
                K[kv_head].append(kh)
                V[kv_head].append(vh)
            keys = K[kv_head]
            scores = [sum(map(_mul, qh, keys[j])) * scale for j in range(len(keys))]
            if self.alibi:
                slope = self.alibi_slopes[head]
                for j in range(len(scores)):
                    scores[j] += slope * (pos - j)
            if self.window is not None:
                # 增量前向只有「过去 + 自己」，窗口外的更早位置要屏蔽（与 forward 一致）
                for j in range(max(0, pos - self.window + 1)):
                    scores[j] = float("-inf")
            attn = _softmax_row(scores)
            cols = list(zip(*V[kv_head]))
            out = [sum(map(_mul, attn, col)) for col in cols]
            concat[off:off + d_k] = out
        return [sum(map(_mul, self.W_o[j], concat)) + self.b_o[j] for j in range(d)]

    def forward(self, x_seq, key_mask=None):
        """x_seq: List[List[float]]  (seq_len, d_model)

        key_mask: 长度 seq_len 的 0/1 列表，0 表示该位置是填充、不能当 key
        （分数置 -inf，softmax 后权重恰为 0，反向自动为 0）。
        None 表示全有效，行为与旧版逐位相同。
        """
        if key_mask is not None:
            if len(key_mask) != len(x_seq):
                raise ValueError(
                    f"MultiHeadAttention.forward: key_mask 长度 {len(key_mask)} "
                    f"与 seq_len={len(x_seq)} 不一致")
            if not any(key_mask):
                raise ValueError("MultiHeadAttention.forward: key_mask 至少要有 1 个有效位置")
        self.key_mask = key_mask
        # 复制一份，避免调用者之后原地修改 x_seq 污染反向传播缓存
        self.x = [list(row) for row in x_seq]
        self.seq_len = len(x_seq)
        d_k = self.d_k
        h = self.num_heads

        # 线性投影 Q, K, V
        kv = self.kv_dim
        q_full = _matmul(x_seq, _transpose(self.W_q))  # (seq_len, d_model)
        k_full = _matmul(x_seq, _transpose(self.W_k))  # (seq_len, kv_dim)
        v_full = _matmul(x_seq, _transpose(self.W_v))
        # 加 bias
        for t in range(self.seq_len):
            for j in range(self.d_model):
                q_full[t][j] += self.b_q[j]
            for j in range(kv):
                k_full[t][j] += self.b_k[j]
                v_full[t][j] += self.b_v[j]

        # 拆分为多头: (seq_len, d_model) → h × (seq_len, d_k)
        self.Q = []
        self.K = []
        self.V = []
        for head in range(h):
            offset = head * d_k
            # GQA：Q 头按组映射到 KV 头（num_kv_heads == num_heads 时即 head 本身）
            kv_off = (head // self.kv_group) * d_k
            q_h = [[q_full[t][offset + j] for j in range(d_k)] for t in range(self.seq_len)]
            k_h = [[k_full[t][kv_off + j] for j in range(d_k)] for t in range(self.seq_len)]
            v_h = [[v_full[t][kv_off + j] for j in range(d_k)] for t in range(self.seq_len)]
            if self.rope:
                # 每个位置用自己的绝对位置角旋转（q 与 k 都转，v 不动）
                q_h = [self._rope_apply(q_h[t], t) for t in range(self.seq_len)]
                k_h = [self._rope_apply(k_h[t], t) for t in range(self.seq_len)]
            self.Q.append(q_h)
            self.K.append(k_h)
            self.V.append(v_h)

        # 各头分别计算注意力
        scale = 1.0 / math.sqrt(d_k)
        self.attn_weights = []
        self.attn_dropped = []
        self.attn_masks = []
        self.head_out = []
        self.concat = [[0.0] * self.d_model for _ in range(self.seq_len)]

        for head in range(h):
            q_h, k_h, v_h = self.Q[head], self.K[head], self.V[head]

            # scores = Q @ K^T / sqrt(d_k)  (seq_len, seq_len)
            k_t = _transpose(k_h)  # (d_k, seq_len)
            scores = [[v * scale for v in row] for row in _matmul(q_h, k_t)]

            # ALiBi：按「距离」加线性偏置（常数，不参与求导）
            if self.alibi:
                slope = self.alibi_slopes[head]
                for t in range(self.seq_len):
                    row = scores[t]
                    for j in range(self.seq_len):
                        if j <= t:
                            row[j] += slope * (t - j)

            # 因果掩码：位置 t 只能看到 j <= t。屏蔽位置置 -inf，
            # softmax 后权重恰为 0；反向的 d_scores = attn * (...) 在那些位置
            # 也自动为 0，所以 backward 不需要任何额外分支。
            if self.causal or self.window is not None:
                for t in range(self.seq_len):
                    row = scores[t]
                    # 未来一律不可见；设了 window 时同样按因果处理（局部因果）
                    for j in range(t + 1, self.seq_len):
                        row[j] = float("-inf")
                    if self.window is not None:
                        # 只看最近 window 个位置：更早的也屏蔽
                        for j in range(max(0, t - self.window + 1)):
                            row[j] = float("-inf")

            # 填充位置不能当 key：所有 query 行都要屏蔽这些列
            if key_mask is not None:
                for t in range(self.seq_len):
                    row = scores[t]
                    for j in range(self.seq_len):
                        if not key_mask[j]:
                            row[j] = float("-inf")

            # softmax per row：attn 是「干净」权重，反向的 softmax 雅可比必须用它
            attn = [_softmax_row(row) for row in scores]

            # 注意力权重 dropout（训练模式且 p>0）：mask 已含 1/(1-p) 缩放。
            # 反向时梯度要乘同一个 mask——被丢掉的连接对 dV/dK/dQ 的贡献为 0。
            mask = None
            used = attn
            if self.training and self.dropout > 0.0:
                keep = 1.0 - self.dropout
                inv = 1.0 / keep
                mask = [[inv if self.rng.random() < keep else 0.0
                         for _ in range(self.seq_len)] for _ in range(self.seq_len)]
                used = [[attn[t][j] * mask[t][j] for j in range(self.seq_len)]
                        for t in range(self.seq_len)]

            # output = attn_dropped @ V  (seq_len, d_k)
            out = _matmul(used, v_h)

            self.attn_weights.append(attn)   # 干净权重（softmax 反向用）
            self.attn_dropped.append(used)   # 实际参与 attn@V 的权重
            self.attn_masks.append(mask)     # None = 本次没做 dropout
            self.head_out.append(out)

            # 拼回 d_model
            offset = head * d_k
            for t in range(self.seq_len):
                for j in range(d_k):
                    self.concat[t][offset + j] = out[t][j]

        # 输出投影
        output = _matmul(self.concat, _transpose(self.W_o))
        for t in range(self.seq_len):
            for j in range(self.d_model):
                output[t][j] += self.b_o[j]
        return output

    def backward(self, dout_seq):
        """dout_seq: (seq_len, d_model) → 返回 dx_seq: (seq_len, d_model)"""
        h = self.num_heads
        d_k = self.d_k
        d_model = self.d_model
        kv = self.kv_dim
        seq_len = self.seq_len

        # 1. d_output → d_concat (通过 W_o)
        d_concat = _matmul(dout_seq, self.W_o)  # (seq_len, d_model)

        # 前向是 output = concat @ W_o^T，即 W_o[j][i] 表示「输入 i → 输出 j」，
        # 所以 grad_W_o[j][i] = sum_t dout[t][j] * concat[t][i]，即 dout^T @ concat。
        # （数值梯度校验过：这一行与原实现一致；concat^T @ dout 会得到转置，是错的）
        grad_W_o = _matmul(_transpose(dout_seq), self.concat)
        grad_b_o = [sum(dout_seq[t][j] for t in range(seq_len)) for j in range(d_model)]

        # 2. 拆回各头
        d_head_out = []
        for head in range(h):
            offset = head * d_k
            d_h = [[d_concat[t][offset + j] for j in range(d_k)] for t in range(seq_len)]
            d_head_out.append(d_h)

        # 3. 各头分别反向传播
        dQ_list, dK_list, dV_list = [], [], []
        for head in range(h):
            attn = self.attn_weights[head]          # 干净权重（softmax 输出）
            attn_used = self.attn_dropped[head]     # 实际参与 attn@V 的权重
            mask = self.attn_masks[head]
            V_h = self.V[head]
            Q_h = self.Q[head]
            K_h = self.K[head]
            d_out = d_head_out[head]

            dV = _matmul(_transpose(attn_used), d_out)

            # d_attn = d_out @ V^T  (seq_len, seq_len)
            d_attn = _matmul(d_out, _transpose(V_h))
            if mask is not None:
                # dropout 的直通：d(attn_dropped) → d(attn_clean) 逐元素乘 mask
                for t in range(seq_len):
                    row_d, row_m = d_attn[t], mask[t]
                    for j in range(seq_len):
                        row_d[j] *= row_m[j]


            # softmax backward: d_scores[t][j] = attn[t][j] * (d_attn[t][j] - sum_i attn[t][i] * d_attn[t][i])
            # 行内点积只依赖 t，必须提到 j 循环外面（原来每个 (t, j) 都重算一遍 → O(seq²) 冗余）
            scale = 1.0 / math.sqrt(d_k)
            d_scores = []
            for t in range(seq_len):
                row_a, row_d = attn[t], d_attn[t]
                s = sum(map(_mul, row_a, row_d))
                d_scores.append([row_a[j] * (row_d[j] - s) * scale for j in range(seq_len)])

            # dQ = d_scores @ K  (seq_len, d_k)
            dQ = _matmul(d_scores, K_h)

            # dK = d_scores^T @ Q  (seq_len, d_k)
            dK = _matmul(_transpose(d_scores), Q_h)

            if self.rope:
                # 前向 q' = R·q，所以 dL/dq = R^T·dL/dq'（反向旋转）
                dQ = [self._rope_apply(dQ[t], t, -1.0) for t in range(seq_len)]
                dK = [self._rope_apply(dK[t], t, -1.0) for t in range(seq_len)]

            dQ_list.append(dQ)
            dK_list.append(dK)
            dV_list.append(dV)

        # 4. 拼接各头的 dQ, dK, dV → (seq_len, d_model)
        kv = self.kv_dim
        dQ_full = [[0.0] * d_model for _ in range(seq_len)]
        dK_full = [[0.0] * kv for _ in range(seq_len)]
        dV_full = [[0.0] * kv for _ in range(seq_len)]
        share_kv = self.num_kv_heads != h
        for head in range(h):
            offset = head * d_k
            kv_off = (head // self.kv_group) * d_k
            for t in range(seq_len):
                for j in range(d_k):
                    dQ_full[t][offset + j] = dQ_list[head][t][j]
                    if share_kv:
                        # 多个 Q 头共享同一组 K/V：它们的梯度要相加
                        dK_full[t][kv_off + j] += dK_list[head][t][j]
                        dV_full[t][kv_off + j] += dV_list[head][t][j]
                    else:
                        dK_full[t][kv_off + j] = dK_list[head][t][j]
                        dV_full[t][kv_off + j] = dV_list[head][t][j]

        # 5. 通过投影矩阵反传梯度
        # 前向是 q_full = x @ W_q^T，即 W_q[b][a] 表示「输入 a → 输出 b」，
        # 所以 grad_W_q[b][a] = sum_t dQ_full[t][b] * x[t][a]，即 dQ_full^T @ x。
        # （不能写成 x^T @ dQ_full：那是正确梯度的转置，方阵下不会报错但会静默训歪）
        grad_W_q = _matmul(_transpose(dQ_full), self.x)  # (d_model, d_model)
        grad_W_k = _matmul(_transpose(dK_full), self.x)
        grad_W_v = _matmul(_transpose(dV_full), self.x)
        grad_b_q = [sum(dQ_full[t][j] for t in range(seq_len)) for j in range(d_model)]
        grad_b_k = [sum(dK_full[t][j] for t in range(seq_len)) for j in range(kv)]
        grad_b_v = [sum(dV_full[t][j] for t in range(seq_len)) for j in range(kv)]

        # d_x = dQ_full @ W_q^T + dK_full @ W_k^T + dV_full @ W_v^T
        dx_q = _matmul(dQ_full, self.W_q)
        dx_k = _matmul(dK_full, self.W_k)
        dx_v = _matmul(dV_full, self.W_v)
        dx_seq = [[dx_q[t][j] + dx_k[t][j] + dx_v[t][j] for j in range(d_model)]
                   for t in range(seq_len)]

        # 6. 累积梯度
        for i in range(self.d_model):
            for j in range(self.d_model):
                self.grads["W_q"][i][j] += grad_W_q[i][j]
                self.grads["W_o"][i][j] += grad_W_o[i][j]
            self.grads["b_q"][i] += grad_b_q[i]
            self.grads["b_o"][i] += grad_b_o[i]
        for i in range(kv):
            for j in range(self.d_model):
                self.grads["W_k"][i][j] += grad_W_k[i][j]
                self.grads["W_v"][i][j] += grad_W_v[i][j]
            self.grads["b_k"][i] += grad_b_k[i]
            self.grads["b_v"][i] += grad_b_v[i]
        return dx_seq


class FeedForward:
    """Transformer 前馈网络: Linear → 激活 → Linear

    activation 默认 "relu"（与原实现逐位一致）；"gelu" 是 GPT/LLaMA 的常用选择。
    激活只作用于中间层，输出层仍是线性。
    """

    def __init__(self, d_model: int, d_ff: int, activation: str = "relu"):
        if activation not in ("relu", "gelu", "sigmoid", "linear",
                              "silu", "mish", "elu", "selu"):
            raise ValueError(
                f"FeedForward: 不支持的激活 {activation!r}；"
                f"可选 relu/gelu/sigmoid/linear/silu/mish/elu/selu"
            )
        self.activation = activation
        self.d_model = d_model
        self.d_ff = d_ff

        std1 = math.sqrt(2.0 / d_model)
        std2 = math.sqrt(2.0 / d_ff)
        self.W1 = [[random.gauss(0, std1) for _ in range(d_model)] for _ in range(d_ff)]
        self.b1 = [0.0] * d_ff
        self.W2 = [[random.gauss(0, std2) for _ in range(d_ff)] for _ in range(d_model)]
        self.b2 = [0.0] * d_model

        self._init_adam()
        self.zero_grad()
        self.x = self.h = None  # 缓存
        self.z = None           # 非 relu 时需要留中间层线性输出

    def _init_adam(self):
        def z2(r, c):
            return [[0.0] * c for _ in range(r)]
        self.m_W1 = z2(self.d_ff, self.d_model); self.v_W1 = z2(self.d_ff, self.d_model)
        self.m_W2 = z2(self.d_model, self.d_ff); self.v_W2 = z2(self.d_model, self.d_ff)
        self.m_b1 = [0.0] * self.d_ff; self.v_b1 = [0.0] * self.d_ff
        self.m_b2 = [0.0] * self.d_model; self.v_b2 = [0.0] * self.d_model

    def zero_grad(self):
        """初始化/清零所有参数梯度"""
        self.grads = {
            "W1": [[0.0] * self.d_model for _ in range(self.d_ff)],
            "b1": [0.0] * self.d_ff,
            "W2": [[0.0] * self.d_ff for _ in range(self.d_model)],
            "b2": [0.0] * self.d_model,
        }

    def forward(self, x_seq):
        """x_seq: (seq_len, d_model)"""
        self.x = x_seq
        self.seq_len = len(x_seq)
        # W1 @ x + b1 → 激活 → W2 @ x + b2
        h = _matmul(x_seq, _transpose(self.W1))
        if self.activation == "relu":
            for t in range(self.seq_len):
                for j in range(self.d_ff):
                    h[t][j] = max(0, h[t][j] + self.b1[j])  # ReLU
        else:
            z = [[h[t][j] + self.b1[j] for j in range(self.d_ff)]
                 for t in range(self.seq_len)]
            self.z = z
            act = self._act_scalar
            for t in range(self.seq_len):
                zt, ht = z[t], h[t]
                for j in range(self.d_ff):
                    ht[j] = act(zt[j])
        self.h = h
        out = _matmul(h, _transpose(self.W2))
        for t in range(self.seq_len):
            for j in range(self.d_model):
                out[t][j] += self.b2[j]
        return out

    def _act_scalar(self, z: float) -> float:
        """中间层激活（标量版，供非 relu 路径使用）"""
        a = self.activation
        if a == "relu":
            return z if z > 0.0 else 0.0
        if a == "gelu":
            u = _GELU_C * (z + 0.044715 * z * z * z)
            return 0.5 * z * (1.0 + math.tanh(u))
        if a == "sigmoid":
            if z >= 0.0:
                return 1.0 / (1.0 + math.exp(-z))
            e = math.exp(z)
            return e / (1.0 + e)
        if a == "silu":
            return _silu_scalar(z)
        if a == "mish":
            return _mish_scalar(z)
        if a == "elu":
            return _elu_scalar(z)
        if a == "selu":
            return _selu_scalar(z)
        return z  # linear

    def _dact_scalar(self, z: float) -> float:
        """中间层激活的导数（标量版）"""
        a = self.activation
        if a == "relu":
            return 1.0 if z > 0.0 else 0.0
        if a == "gelu":
            u = _GELU_C * (z + 0.044715 * z * z * z)
            t = math.tanh(u)
            du = _GELU_C * (1.0 + 3.0 * 0.044715 * z * z)
            return 0.5 * (1.0 + t) + 0.5 * z * (1.0 - t * t) * du
        if a == "sigmoid":
            s = self._act_scalar(z)
            return s * (1.0 - s)
        if a == "silu":
            return _dsilu_scalar(z)
        if a == "mish":
            return _dmish_scalar(z)
        if a == "elu":
            return _delu_scalar(z)
        if a == "selu":
            return _dselu_scalar(z)
        return 1.0  # linear

    def backward(self, dout_seq):
        """dout_seq: (seq_len, d_model) → dx_seq: (seq_len, d_model)"""
        seq_len = self.seq_len

        # dout → W2:  dW2 = dout^T @ h  (d_model, seq_len) @ (seq_len, d_ff) = (d_model, d_ff)
        grad_W2 = _matmul(_transpose(dout_seq), self.h)
        grad_b2 = [sum(dout_seq[t][j] for t in range(seq_len)) for j in range(self.d_model)]

        # d_h = dout @ W2
        dh = _matmul(dout_seq, self.W2)  # (seq_len, d_ff)

        # 激活反向：relu 走原分支（读 h，保持逐位一致）；其它用中间层 z
        if self.activation == "relu":
            for t in range(seq_len):
                for j in range(self.d_ff):
                    if self.h[t][j] <= 0:
                        dh[t][j] = 0.0
        else:
            dact = self._dact_scalar
            z = self.z
            for t in range(seq_len):
                zt, dht = z[t], dh[t]
                for j in range(self.d_ff):
                    dht[j] *= dact(zt[j])

        # W1 backward: dW1 = dh^T @ x  (d_ff, seq_len) @ (seq_len, d_model) = (d_ff, d_model)
        grad_W1 = _matmul(_transpose(dh), self.x)
        grad_b1 = [sum(dh[t][j] for t in range(seq_len)) for j in range(self.d_ff)]

        # dx = dh @ W1
        dx_seq = _matmul(dh, self.W1)

        # 累积梯度
        for i in range(self.d_ff):
            for j in range(self.d_model):
                self.grads["W1"][i][j] += grad_W1[i][j]
            self.grads["b1"][i] += grad_b1[i]
        for i in range(self.d_model):
            for j in range(self.d_ff):
                self.grads["W2"][i][j] += grad_W2[i][j]
            self.grads["b2"][i] += grad_b2[i]
        return dx_seq



class SwiGLU:
    """门控前馈网络（LLaMA / PaLM 用的 SwiGLU）

    out = (silu(x W_g) ⊙ (x W_1)) W_2 + b_2

    与 `FeedForward` 的区别：多一路门控投影 `W_g`，两个分支逐元素相乘
    （SwiGLU = Swish-Gated Linear Unit）。表达力更强，LLaMA 用它替代 ReLU-FFN。
    d_ff 通常取 FeedForward 的 2/3 左右效果相当（这里由调用者决定）。
    """

    def __init__(self, d_model: int, d_ff: int):
        if isinstance(d_model, bool) or not isinstance(d_model, int) or d_model < 1:
            raise ValueError(f"SwiGLU: d_model 必须是正整数，收到 {d_model!r}")
        if isinstance(d_ff, bool) or not isinstance(d_ff, int) or d_ff < 1:
            raise ValueError(f"SwiGLU: d_ff 必须是正整数，收到 {d_ff!r}")
        self.d_model = d_model
        self.d_ff = d_ff
        std1 = math.sqrt(2.0 / d_model)
        std2 = math.sqrt(2.0 / d_ff)
        self.W1 = [[random.gauss(0, std1) for _ in range(d_model)] for _ in range(d_ff)]
        self.b1 = [0.0] * d_ff
        self.Wg = [[random.gauss(0, std1) for _ in range(d_model)] for _ in range(d_ff)]
        self.bg = [0.0] * d_ff
        self.W2 = [[random.gauss(0, std2) for _ in range(d_ff)] for _ in range(d_model)]
        self.b2 = [0.0] * d_model
        self.training = True
        self._init_adam()
        self.zero_grad()
        self.x = self.h1 = self.g = self.zg = self.h = None
        self.seq_len = 0

    def _init_adam(self):
        def z2(r, c):
            return [[0.0] * c for _ in range(r)]
        self.m_W1 = z2(self.d_ff, self.d_model); self.v_W1 = z2(self.d_ff, self.d_model)
        self.m_Wg = z2(self.d_ff, self.d_model); self.v_Wg = z2(self.d_ff, self.d_model)
        self.m_W2 = z2(self.d_model, self.d_ff); self.v_W2 = z2(self.d_model, self.d_ff)
        self.m_b1 = [0.0] * self.d_ff; self.v_b1 = [0.0] * self.d_ff
        self.m_bg = [0.0] * self.d_ff; self.v_bg = [0.0] * self.d_ff
        self.m_b2 = [0.0] * self.d_model; self.v_b2 = [0.0] * self.d_model

    def zero_grad(self):
        """初始化/清零所有参数梯度"""
        self.grads = {
            "W1": [[0.0] * self.d_model for _ in range(self.d_ff)],
            "b1": [0.0] * self.d_ff,
            "Wg": [[0.0] * self.d_model for _ in range(self.d_ff)],
            "bg": [0.0] * self.d_ff,
            "W2": [[0.0] * self.d_ff for _ in range(self.d_model)],
            "b2": [0.0] * self.d_model,
        }

    def train(self):
        """训练模式（本模块无 dropout，接口对齐 Transformer 子模块）"""
        self.training = True
        return self

    def eval(self):
        self.training = False
        return self

    def forward(self, x_seq):
        """x_seq: (seq_len, d_model) → (seq_len, d_model)"""
        if not x_seq:
            raise ValueError("SwiGLU.forward: 输入序列为空")
        if len(x_seq[0]) != self.d_model:
            raise ValueError(
                f"SwiGLU.forward: 输入维度 {len(x_seq[0])} 与 d_model={self.d_model} 不一致")
        self.x = x_seq
        self.seq_len = len(x_seq)
        self.h1 = _matmul(x_seq, _transpose(self.W1))
        zg = _matmul(x_seq, _transpose(self.Wg))
        self.zg = zg
        self.g = [[_silu_scalar(zg[t][j]) + self.bg[j] for j in range(self.d_ff)]
                  for t in range(self.seq_len)]
        # 门控分支的 bias 加在 silu 之后（LLaMA 无 bias，这里是通用写法）
        for t in range(self.seq_len):
            for j in range(self.d_ff):
                self.h1[t][j] += self.b1[j]
        self.h = [[self.h1[t][j] * self.g[t][j] for j in range(self.d_ff)]
                  for t in range(self.seq_len)]
        out = _matmul(self.h, _transpose(self.W2))
        for t in range(self.seq_len):
            for j in range(self.d_model):
                out[t][j] += self.b2[j]
        return out

    def backward(self, dout_seq):
        """dout_seq: (seq_len, d_model) → dx_seq: (seq_len, d_model)"""
        if self.h is None:
            raise RuntimeError("SwiGLU.backward: 必须先调用 forward()")
        T = self.seq_len
        grad_W2 = _matmul(_transpose(dout_seq), self.h)
        grad_b2 = [sum(dout_seq[t][j] for t in range(T)) for j in range(self.d_model)]
        dh = _matmul(dout_seq, self.W2)
        # 两条分支各自的梯度
        dh1 = [[dh[t][j] * self.g[t][j] for j in range(self.d_ff)] for t in range(T)]
        dg = [[dh[t][j] * self.h1[t][j] for j in range(self.d_ff)] for t in range(T)]
        # bg 是加在 silu 之后的（g = silu(z_g) + b_g），所以它的梯度不乘 silu'
        dg_pre = [row[:] for row in dg]
        # 门控分支反向：d silu（只用于 W_g 与 dx）
        for t in range(T):
            for j in range(self.d_ff):
                dg[t][j] *= _dsilu_scalar(self.zg[t][j])
        grad_W1 = _matmul(_transpose(dh1), self.x)
        grad_b1 = [sum(dh1[t][j] for t in range(T)) for j in range(self.d_ff)]
        grad_Wg = _matmul(_transpose(dg), self.x)
        grad_bg = [sum(dg_pre[t][j] for t in range(T)) for j in range(self.d_ff)]
        dx_seq = _matmul(dh1, self.W1)
        dx_g = _matmul(dg, self.Wg)
        for t in range(T):
            for j in range(self.d_model):
                dx_seq[t][j] += dx_g[t][j]
        # 累积梯度（与 FeedForward 一致：+= 累加，配合 zero_grad）
        for i in range(self.d_ff):
            for j in range(self.d_model):
                self.grads["W1"][i][j] += grad_W1[i][j]
                self.grads["Wg"][i][j] += grad_Wg[i][j]
            self.grads["b1"][i] += grad_b1[i]
            self.grads["bg"][i] += grad_bg[i]
        for i in range(self.d_model):
            for j in range(self.d_ff):
                self.grads["W2"][i][j] += grad_W2[i][j]
            self.grads["b2"][i] += grad_b2[i]
        return dx_seq


class MoE:
    """专家混合前馈层（Mixture of Experts）

    out_t = Σ_{e ∈ top-k} q_e · FFN_e(x_t)，其中 q 是在**被选中的专家**上做的 softmax
    （Switch/GShard 风格：路由只对被选中的专家归一化，未被选中的专家梯度为 0）。

    与 `FeedForward` 的区别：多个专家并行，每个 token 只走 top_k 条路径，
    因此参数量随专家数线性增长，计算量只按 top_k 增长。

    注意：这是一个**教学级实现**——它把 token 分组后逐专家前向，没有做批量矩阵化，
    token 数少时很快，超长序列比稠密 FFN 慢。真实的大模型实现会用 scatter/gather。
    """

    def __init__(self, d_model: int, d_ff: int, num_experts: int = 4,
                 top_k: int = 2, activation: str = "gelu"):
        for name, value in (("d_model", d_model), ("d_ff", d_ff),
                            ("num_experts", num_experts), ("top_k", top_k)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"MoE: {name} 必须是 >= 1 的整数，收到 {value!r}")
        if top_k > num_experts:
            raise ValueError(f"MoE: top_k({top_k}) 不能大于 num_experts({num_experts})")
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_experts = num_experts
        self.top_k = top_k
        self.activation = activation

        self.experts = [FeedForward(d_model, d_ff, activation) for _ in range(num_experts)]
        std = math.sqrt(2.0 / d_model)
        self.W_gate = [[random.gauss(0.0, std) for _ in range(d_model)]
                       for _ in range(num_experts)]
        self.b_gate = [0.0] * num_experts

        self._init_adam()
        self.zero_grad()
        self.x = None
        self.routes = None      # 每个 token: [(expert, weight), ...]
        self.group_index = None  # 每个专家: [token 下标, ...]
        self.group_w = None

    # ---- 参数与状态
    def _named_params(self):
        for e, expert in enumerate(self.experts):
            for key in ("W1", "b1", "W2", "b2"):
                yield f"e{e}.{key}", getattr(expert, key), expert.grads[key]
        yield "W_gate", self.W_gate, self.grad_W_gate
        yield "b_gate", self.b_gate, self.grad_b_gate

    def zero_grad(self):
        for expert in self.experts:
            expert.zero_grad()
        self.grad_W_gate = _zeros_like(self.W_gate)
        self.grad_b_gate = _zeros_like(self.b_gate)

    def _init_adam(self):
        self.m = {}
        self.v = {}
        self.step_count = 0
        for expert in self.experts:
            expert._init_adam()
        self._sync_adam()

    def _sync_adam(self):
        for name, param in self._param_views():
            if self.m.get(name) is None or len(self.m[name]) != len(param):
                self.m[name] = _zeros_like(param)
                self.v[name] = _zeros_like(param)

    def _param_views(self):
        for e, expert in enumerate(self.experts):
            for key in ("W1", "b1", "W2", "b2"):
                yield f"e{e}.{key}", getattr(expert, key)
        yield "W_gate", self.W_gate
        yield "b_gate", self.b_gate

    def param_count(self):
        per = sum(len(row) for row in self.experts[0].W1) * self.d_model
        per += len(self.experts[0].b1)
        per += self.d_model * len(self.experts[0].W2[0])
        per += len(self.experts[0].b2)
        return per * self.num_experts + len(self.W_gate) * self.d_model + len(self.b_gate)

    # ---- 前向
    def forward(self, x_seq):
        if not isinstance(x_seq, (list, tuple)) or not x_seq:
            raise ValueError("MoE.forward: 输入必须是非空序列")
        T = len(x_seq)
        for t, row in enumerate(x_seq):
            if not isinstance(row, (list, tuple)) or len(row) != self.d_model:
                raise ValueError(f"MoE.forward: 第 {t} 个位置维度应为 {self.d_model}")
        self.x = [list(row) for row in x_seq]

        routes = []
        for t in range(T):
            xt = self.x[t]
            logits = [sum(map(_mul, self.W_gate[e], xt)) + self.b_gate[e]
                      for e in range(self.num_experts)]
            order = sorted(range(self.num_experts), key=lambda e: -logits[e])[:self.top_k]
            picked = [logits[e] for e in order]
            probs = _softmax_row(picked)
            routes.append([(order[k], probs[k]) for k in range(self.top_k)])
        self.routes = routes

        group_index = [[] for _ in range(self.num_experts)]
        group_w = [[] for _ in range(self.num_experts)]
        group_pos = [{} for _ in range(self.num_experts)]
        for t in range(T):
            for e, w in routes[t]:
                group_pos[e][t] = len(group_index[e])
                group_index[e].append(t)
                group_w[e].append(w)
        self.group_index = group_index
        self.group_w = group_w
        self.group_pos = group_pos

        out = [[0.0] * self.d_model for _ in range(T)]
        self._last_group_out = {}
        for e in range(self.num_experts):
            idx = group_index[e]
            if not idx:
                continue
            group_out = self.experts[e].forward([self.x[t] for t in idx])
            self._last_group_out[e] = group_out
            for k, t in enumerate(idx):
                w = group_w[e][k]
                row = group_out[k]
                target = out[t]
                for j in range(self.d_model):
                    target[j] += w * row[j]
        return out

    # ---- 反向
    def backward(self, dout_seq):
        if self.x is None:
            raise RuntimeError("MoE.backward: 必须先 forward")
        T = len(self.x)
        if not isinstance(dout_seq, (list, tuple)) or len(dout_seq) != T:
            raise ValueError(f"MoE.backward: dout 长度应为 {T}")
        for t in range(T):
            if len(dout_seq[t]) != self.d_model:
                raise ValueError(f"MoE.backward: 第 {t} 行梯度维度应为 {self.d_model}")

        dx = [[0.0] * self.d_model for _ in range(T)]

        # 1) 各专家的输入梯度：dL/dout_t 乘路由权重后按组回传
        for e in range(self.num_experts):
            idx = self.group_index[e]
            if not idx:
                continue
            dgroup = []
            for k, t in enumerate(idx):
                w = self.group_w[e][k]
                dgroup.append([w * v for v in dout_seq[t]])
            dx_group = self.experts[e].backward(dgroup)
            for k, t in enumerate(idx):
                row = dx_group[k]
                for j in range(self.d_model):
                    dx[t][j] += row[j]

        # 2) 路由梯度：被选中专家的 softmax
        for t in range(T):
            route = self.routes[t]
            c = []
            for e, _w in route:
                gpos = self.group_pos[e][t]
                c.append(sum(map(_mul, dout_seq[t], self._last_group_out[e][gpos])))
            # p 已归一化在选中集合上：dz_j = p_j (c_j - Σ_k p_k c_k)
            q = [w for _e, w in route]
            base = sum(q[k] * c[k] for k in range(len(route)))
            xt = self.x[t]
            for k, (e, _w) in enumerate(route):
                dz = q[k] * (c[k] - base)
                self.grad_b_gate[e] += dz
                gw = self.grad_W_gate[e]
                for j in range(self.d_model):
                    gw[j] += dz * xt[j]
                    dx[t][j] += dz * self.W_gate[e][j]
        return dx

    # ---- 优化
    def sgd_step(self, lr: float = 0.01):
        for _name, param, grad in self._named_params():
            if isinstance(param[0], list):
                for i in range(len(param)):
                    for j in range(len(param[i])):
                        param[i][j] -= lr * grad[i][j]
            else:
                for i in range(len(param)):
                    param[i] -= lr * grad[i]

    def adam_step(self, lr: float = 1e-3, b1: float = 0.9, b2: float = 0.999,
                  eps: float = 1e-8):
        self.step_count += 1
        t = self.step_count
        for name, param, grad in self._named_params():
            _adam_apply(param, grad, self.m[name], self.v[name], t, lr, b1, b2, eps)

    def train(self):
        for expert in self.experts:
            expert.train()
        return self

    def eval(self):
        for expert in self.experts:
            expert.eval()
        return self


def _sgd_apply(param, grad, lr):
    """就地 SGD 更新参数: 支持 1D / 2D 列表（param 与 grad 结构一致）"""
    if grad and isinstance(grad[0], (list, tuple)):
        for i in range(len(param)):
            for j in range(len(param[i])):
                param[i][j] -= lr * grad[i][j]
    else:
        for i in range(len(param)):
            param[i] -= lr * grad[i]


class TransformerEncoderLayer:
    """单层 Transformer Encoder: Self-Attn + Residual + LN + FFN + Residual + LN"""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1,
                 causal: bool = False, rope: bool = False, window: int = None,
                 num_kv_heads: int = None, alibi: bool = False):
        # causal=True 时自注意力加因果掩码（自回归语言模型用）；
        # dropout 同时作用在注意力权重（MHA 内部）和两个残差子层（dropout1/2）
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout,
                                            causal=causal, rope=rope, window=window,
                                            num_kv_heads=num_kv_heads, alibi=alibi)
        self.causal = self.self_attn.causal
        self.ffn = FeedForward(d_model, d_ff)
        self.ln1 = LayerNorm(d_model)
        self.ln2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.zero_grad()

        # 缓存残差 及 逐 token dropout mask
        self.x = self.res1 = self.res2 = None
        self.dropout1_masks = self.dropout2_masks = None

    def train(self):
        """训练模式：注意力权重 dropout 与两个 Dropout 都生效"""
        self.self_attn.train()
        self.dropout1.train()
        self.dropout2.train()
        return self

    def eval(self):
        """推理模式：关闭全部 dropout，保证同一输入输出确定"""
        self.self_attn.eval()
        self.dropout1.eval()
        self.dropout2.eval()
        return self

    def zero_grad(self):
        """初始化/清零 LayerNorm 梯度、同时重置子模块梯度"""
        d = self.self_attn.d_model
        self.grads = {
            "ln1_gamma": [0.0] * d, "ln1_beta": [0.0] * d,
            "ln2_gamma": [0.0] * d, "ln2_beta": [0.0] * d,
        }
        self.self_attn.zero_grad()
        self.ffn.zero_grad()

    def forward_step(self, x_t, cache):
        """单 token 增量前向（推理用，eval 模式下与 forward 的最后一行逐位一致）"""
        d_model = self.self_attn.d_model
        attn_out = self.self_attn.forward_step(x_t, cache)
        res1 = [x_t[j] + attn_out[j] for j in range(d_model)]
        ln1_out = self.ln1.forward(res1)
        ffn_out = self.ffn.forward([ln1_out])[0]
        res2 = [ln1_out[j] + ffn_out[j] for j in range(d_model)]
        return self.ln2.forward(res2)

    def forward(self, x_seq, key_mask=None):
        self.x = x_seq
        d_model = self.self_attn.d_model
        # Self-Attention + Dropout1（逐 token 存储 mask）
        attn_out = self.self_attn.forward(x_seq, key_mask)
        self.dropout1_masks = []
        attn_dropped = []
        for a in attn_out:
            dropped = self.dropout1.forward(a)
            attn_dropped.append(dropped)
            self.dropout1_masks.append(list(self.dropout1.mask))
        attn_out = attn_dropped
        # Residual + LN1
        self.res1 = [[x_seq[t][j] + attn_out[t][j] for j in range(d_model)]
                      for t in range(len(x_seq))]
        ln1_out = [self.ln1.forward(v) for v in self.res1]
        # FFN + Dropout2（逐 token 存储 mask）
        ffn_out = self.ffn.forward(ln1_out)
        self.dropout2_masks = []
        ffn_dropped = []
        for f in ffn_out:
            dropped = self.dropout2.forward(f)
            ffn_dropped.append(dropped)
            self.dropout2_masks.append(list(self.dropout2.mask))
        ffn_out = ffn_dropped
        # Residual + LN2
        self.res2 = [[ln1_out[t][j] + ffn_out[t][j] for j in range(d_model)]
                      for t in range(len(x_seq))]
        return [self.ln2.forward(v) for v in self.res2]

    def backward(self, dout_seq):
        seq_len = len(self.x)
        d_model = self.self_attn.d_model

        # LN2 backward（逐 token）
        # 注意：LayerNorm 只保存「最后一次 forward」的单槽缓存，而 forward 里
        # ln2 是对每个 token 各调一次（见上面的 [self.ln2.forward(v) for v in self.res2]），
        # 缓存里只剩最后一个 token。因此这里必须先把该 token 的前向缓存重放回去，
        # 否则 token 0..T-2 会用最后一个 token 的 mean/var 反传 —— 梯度整体是错的
        # （seq_len=1 时恰好正确，所以只在小序列上看不出来）。
        d_res2 = []
        g_ln2_gamma = [0.0] * d_model; g_ln2_beta = [0.0] * d_model
        for t in range(seq_len):
            self.ln2.forward(self.res2[t])          # 重放缓存
            gg, gb, dx = self.ln2.backward(dout_seq[t])
            d_res2.append(dx)
            for j in range(d_model):
                g_ln2_gamma[j] += gg[j]
                g_ln2_beta[j] += gb[j]
        # Residual 2 的梯度分叉: 一路给 FFN，一路直通
        d_ffn = list(d_res2)
        # Dropout2 backward（逐 token 应用存储的 mask）
        if self.dropout2_masks is not None:
            d_ffn = [[d_ffn[t][j] * self.dropout2_masks[t][j]
                      for j in range(d_model)] for t in range(seq_len)]
        # FFN backward
        d_ln1 = self.ffn.backward(d_ffn)
        for t in range(seq_len):
            for j in range(d_model):
                d_ln1[t][j] += d_res2[t][j]
        # LN1 backward（逐 token）：同样要重放该 token 的缓存（原因见上面的 LN2）
        d_res1 = []
        g_ln1_gamma = [0.0] * d_model; g_ln1_beta = [0.0] * d_model
        for t in range(seq_len):
            self.ln1.forward(self.res1[t])          # 重放缓存
            gg, gb, dx = self.ln1.backward(d_ln1[t])
            d_res1.append(dx)
            for j in range(d_model):
                g_ln1_gamma[j] += gg[j]
                g_ln1_beta[j] += gb[j]
        # Residual 1: 注意力分支 和 直通分支
        d_attn = list(d_res1)
        d_x = list(d_res1)
        # Dropout1 backward（逐 token 应用存储的 mask）
        if self.dropout1_masks is not None:
            d_attn = [[d_attn[t][j] * self.dropout1_masks[t][j]
                       for j in range(d_model)] for t in range(seq_len)]
        # Self-Attn backward
        d_attn_out = self.self_attn.backward(d_attn)
        for t in range(seq_len):
            for j in range(d_model):
                d_x[t][j] += d_attn_out[t][j]

        for j in range(d_model):
            self.grads["ln1_gamma"][j] += g_ln1_gamma[j]
            self.grads["ln1_beta"][j] += g_ln1_beta[j]
            self.grads["ln2_gamma"][j] += g_ln2_gamma[j]
            self.grads["ln2_beta"][j] += g_ln2_beta[j]
        return d_x

    def sgd_step(self, lr: float):
        """用累积的梯度做一次 SGD 参数更新（self_attn / ffn / LayerNorm）"""
        attn = self.self_attn
        for name in ("W_q", "W_k", "W_v", "W_o"):
            _sgd_apply(getattr(attn, name), attn.grads[name], lr)
        for name in ("b_q", "b_k", "b_v", "b_o"):
            _sgd_apply(getattr(attn, name), attn.grads[name], lr)

        ffn = self.ffn
        _sgd_apply(ffn.W1, ffn.grads["W1"], lr)
        _sgd_apply(ffn.b1, ffn.grads["b1"], lr)
        _sgd_apply(ffn.W2, ffn.grads["W2"], lr)
        _sgd_apply(ffn.b2, ffn.grads["b2"], lr)

        _sgd_apply(self.ln1.gamma, self.grads["ln1_gamma"], lr)
        _sgd_apply(self.ln1.beta, self.grads["ln1_beta"], lr)
        _sgd_apply(self.ln2.gamma, self.grads["ln2_gamma"], lr)
        _sgd_apply(self.ln2.beta, self.grads["ln2_beta"], lr)


class TransformerEncoder:
    """堆叠多个 TransformerEncoderLayer"""

    def __init__(self, d_model: int, num_heads: int, d_ff: int,
                 num_layers: int = 6, dropout: float = 0.1, causal: bool = False,
                 rope: bool = False, window: int = None, num_kv_heads: int = None,
                 alibi: bool = False):
        """causal=True: 每层自注意力都加因果掩码（供 CausalLM 使用）"""
        self.causal = bool(causal)
        self.layers = [
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout,
                                    causal=causal, rope=rope, window=window,
                                    num_kv_heads=num_kv_heads, alibi=alibi)
            for _ in range(num_layers)
        ]

    def zero_grad(self):
        """重置所有层的梯度"""
        for layer in self.layers:
            layer.zero_grad()

    def train(self):
        """训练模式（转发给每一层）"""
        for layer in self.layers:
            layer.train()
        return self

    def eval(self):
        """推理模式（转发给每一层，关闭 Dropout）"""
        for layer in self.layers:
            layer.eval()
        return self

    def forward_step(self, x_t, caches):
        """单 token 增量前向；caches 每层一个 dict（见 MHA.forward_step）"""
        for li, layer in enumerate(self.layers):
            x_t = layer.forward_step(x_t, caches[li])
        return x_t

    def forward(self, x_seq, key_mask=None):
        for layer in self.layers:
            x_seq = layer.forward(x_seq, key_mask)
        return x_seq

    def backward(self, dout_seq):
        for layer in reversed(self.layers):
            dout_seq = layer.backward(dout_seq)
        return dout_seq

    def sgd_step(self, lr: float):
        """对每一层做一次 SGD 参数更新"""
        for layer in self.layers:
            layer.sgd_step(lr)


class CrossAttention:
    """多头交叉注意力（encoder-decoder 的 decoder 端）

    Q 来自解码器自己的序列 x_seq `(Tq, d_model)`，K/V 来自编码器输出
    memory `(Tk, d_model)`：两者长度可以不同，也不需要 causal 掩码——
    解码器对整段 memory 是全可见的。所以这里没有 causal / window / rope
    参数（位置信息由各自那侧的注入方式决定）。

    backward 返回 `(dx, dmemory)`：既要把梯度传回解码器上一层，也要传回
    编码器（多层 decoder 共享同一份 memory，各层的 dmemory 相加）。

    `delta_index = 0` 显式声明「往回传的是第 0 项」：容器（Sequential 等）
    默认取 -1，会把 dmemory 当成输入梯度，静默拿错。
    """

    delta_index = 0

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        """dropout: 交叉注意力权重上的 inverted dropout（训练模式生效）"""
        if not (0.0 <= dropout < 1.0):
            raise ValueError(
                f"CrossAttention: dropout 必须在 [0,1) 内，收到 {dropout!r}")
        if d_model % num_heads != 0:
            raise ValueError(
                f"CrossAttention: d_model={d_model} 必须能被 num_heads={num_heads} 整除")
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.dropout = dropout
        self.training = True
        self.rng = random

        std = math.sqrt(2.0 / d_model)
        self.W_q = [[random.gauss(0, std) for _ in range(d_model)] for _ in range(d_model)]
        self.W_k = [[random.gauss(0, std) for _ in range(d_model)] for _ in range(d_model)]
        self.W_v = [[random.gauss(0, std) for _ in range(d_model)] for _ in range(d_model)]
        self.W_o = [[random.gauss(0, std) for _ in range(d_model)] for _ in range(d_model)]
        self.b_q = [0.0] * d_model
        self.b_k = [0.0] * d_model
        self.b_v = [0.0] * d_model
        self.b_o = [0.0] * d_model

        self._init_adam()
        self.zero_grad()

        self.x = None          # (Tq, d_model) 解码器侧输入
        self.memory = None     # (Tk, d_model) 编码器输出
        self.Q = self.K = self.V = None
        self.attn_weights = None
        self.attn_dropped = None
        self.attn_masks = None
        self.concat = None

    def zero_grad(self):
        """初始化/清零所有参数梯度"""
        d = self.d_model
        zeros_2d = lambda r, c: [[0.0] * c for _ in range(r)]
        self.grads = {
            "W_q": zeros_2d(d, d), "b_q": [0.0] * d,
            "W_k": zeros_2d(d, d), "b_k": [0.0] * d,
            "W_v": zeros_2d(d, d), "b_v": [0.0] * d,
            "W_o": zeros_2d(d, d), "b_o": [0.0] * d,
        }

    def _init_adam(self):
        d = self.d_model
        zeros_2d = lambda r, c: [[0.0] * c for _ in range(r)]
        self.m_W_q = zeros_2d(d, d); self.v_W_q = zeros_2d(d, d)
        self.m_W_k = zeros_2d(d, d); self.v_W_k = zeros_2d(d, d)
        self.m_W_v = zeros_2d(d, d); self.v_W_v = zeros_2d(d, d)
        self.m_W_o = zeros_2d(d, d); self.v_W_o = zeros_2d(d, d)
        self.m_b_q = [0.0] * d; self.v_b_q = [0.0] * d
        self.m_b_k = [0.0] * d; self.v_b_k = [0.0] * d
        self.m_b_v = [0.0] * d; self.v_b_v = [0.0] * d
        self.m_b_o = [0.0] * d; self.v_b_o = [0.0] * d

    def train(self):
        """训练模式：注意力权重 dropout 生效"""
        self.training = True
        return self

    def eval(self):
        """推理模式：关闭 dropout"""
        self.training = False
        return self

    def forward(self, x_seq, memory, memory_mask=None):
        """x_seq: (Tq, d_model) 解码器输入；memory: (Tk, d_model) 编码器输出

        memory_mask: 长度 Tk 的 0/1 列表，0 表示该位置是填充、不能参与注意力
        （批量训练时右填充的源序列）。
        """
        if len(x_seq) == 0:
            raise ValueError("CrossAttention: x_seq 不能为空")
        if len(memory) == 0:
            raise ValueError("CrossAttention: memory 不能为空")
        for t, row in enumerate(x_seq):
            if len(row) != self.d_model:
                raise ValueError(
                    f"CrossAttention: x_seq[{t}] 长度 {len(row)} != d_model {self.d_model}")
        for t, row in enumerate(memory):
            if len(row) != self.d_model:
                raise ValueError(
                    f"CrossAttention: memory[{t}] 长度 {len(row)} != d_model {self.d_model}")
        if memory_mask is not None:
            memory_mask = _check_pad_mask(memory_mask, len(memory), "CrossAttention")
        self.x = [list(row) for row in x_seq]
        self.memory = [list(row) for row in memory]
        tq, tk = len(x_seq), len(memory)
        d, h, d_k = self.d_model, self.num_heads, self.d_k

        q_full = _matmul(x_seq, _transpose(self.W_q))
        k_full = _matmul(memory, _transpose(self.W_k))
        v_full = _matmul(memory, _transpose(self.W_v))
        for t in range(tq):
            for j in range(d):
                q_full[t][j] += self.b_q[j]
        for t in range(tk):
            for j in range(d):
                k_full[t][j] += self.b_k[j]
                v_full[t][j] += self.b_v[j]

        self.Q, self.K, self.V = [], [], []
        for head in range(h):
            off = head * d_k
            self.Q.append([[q_full[t][off + j] for j in range(d_k)] for t in range(tq)])
            self.K.append([[k_full[t][off + j] for j in range(d_k)] for t in range(tk)])
            self.V.append([[v_full[t][off + j] for j in range(d_k)] for t in range(tk)])

        scale = 1.0 / math.sqrt(d_k)
        self.attn_weights, self.attn_dropped, self.attn_masks = [], [], []
        self.concat = [[0.0] * d for _ in range(tq)]
        keep = 1.0 - self.dropout
        inv = 1.0 / keep if keep > 0 else 1.0
        for head in range(h):
            q_h, k_h, v_h = self.Q[head], self.K[head], self.V[head]
            scores = [[v * scale for v in row] for row in _matmul(q_h, _transpose(k_h))]
            # 交叉注意力没有因果掩码：解码器位置能看到整段 memory；
            # 但批量训练时被填充的 memory 位置必须屏蔽（softmax 后权重恰为 0，
            # 反向无需额外分支）
            if memory_mask is not None:
                for t in range(tq):
                    row = scores[t]
                    for j in range(tk):
                        if not memory_mask[j]:
                            row[j] = float("-inf")
            attn = [_softmax_row(row) for row in scores]
            mask = None
            used = attn
            if self.training and self.dropout > 0.0:
                mask = [[inv if self.rng.random() < keep else 0.0 for _ in range(tk)]
                        for _ in range(tq)]
                used = [[attn[t][j] * mask[t][j] for j in range(tk)] for t in range(tq)]
            out = _matmul(used, v_h)
            self.attn_weights.append(attn)
            self.attn_dropped.append(used)
            self.attn_masks.append(mask)
            off = head * d_k
            for t in range(tq):
                for j in range(d_k):
                    self.concat[t][off + j] = out[t][j]

        output = _matmul(self.concat, _transpose(self.W_o))
        for t in range(tq):
            for j in range(d):
                output[t][j] += self.b_o[j]
        return output

    def backward(self, dout_seq):
        """dout_seq: (Tq, d_model) → (dx_seq (Tq, d_model), dmemory (Tk, d_model))"""
        tq, tk = len(self.x), len(self.memory)
        d, h, d_k = self.d_model, self.num_heads, self.d_k
        scale = 1.0 / math.sqrt(d_k)

        d_concat = _matmul(dout_seq, self.W_o)
        grad_W_o = _matmul(_transpose(dout_seq), self.concat)
        grad_b_o = [sum(dout_seq[t][j] for t in range(tq)) for j in range(d)]

        dQ_full = [[0.0] * d for _ in range(tq)]
        dK_full = [[0.0] * d for _ in range(tk)]
        dV_full = [[0.0] * d for _ in range(tk)]
        for head in range(h):
            off = head * d_k
            attn = self.attn_weights[head]
            used = self.attn_dropped[head]
            mask = self.attn_masks[head]
            V_h, Q_h, K_h = self.V[head], self.Q[head], self.K[head]
            d_out = [[d_concat[t][off + j] for j in range(d_k)] for t in range(tq)]

            dV = _matmul(_transpose(used), d_out)          # (Tk, d_k)
            d_attn = _matmul(d_out, _transpose(V_h))       # (Tq, Tk)
            if mask is not None:
                for t in range(tq):
                    row_d, row_m = d_attn[t], mask[t]
                    for j in range(tk):
                        row_d[j] *= row_m[j]
            d_scores = []
            for t in range(tq):
                row_a, row_d = attn[t], d_attn[t]
                s = sum(map(_mul, row_a, row_d))
                d_scores.append([row_a[j] * (row_d[j] - s) * scale for j in range(tk)])
            dQ = _matmul(d_scores, K_h)                     # (Tq, d_k)
            dK = _matmul(_transpose(d_scores), Q_h)         # (Tk, d_k)
            for t in range(tq):
                for j in range(d_k):
                    dQ_full[t][off + j] = dQ[t][j]
            for t in range(tk):
                for j in range(d_k):
                    dK_full[t][off + j] = dK[t][j]
                    dV_full[t][off + j] = dV[t][j]

        grad_W_q = _matmul(_transpose(dQ_full), self.x)
        grad_W_k = _matmul(_transpose(dK_full), self.memory)
        grad_W_v = _matmul(_transpose(dV_full), self.memory)
        grad_b_q = [sum(dQ_full[t][j] for t in range(tq)) for j in range(d)]
        grad_b_k = [sum(dK_full[t][j] for t in range(tk)) for j in range(d)]
        grad_b_v = [sum(dV_full[t][j] for t in range(tk)) for j in range(d)]

        dx_seq = _matmul(dQ_full, self.W_q)
        dmem_k = _matmul(dK_full, self.W_k)
        dmem_v = _matmul(dV_full, self.W_v)
        dmemory = [[dmem_k[t][j] + dmem_v[t][j] for j in range(d)] for t in range(tk)]

        for i in range(d):
            for j in range(d):
                self.grads["W_q"][i][j] += grad_W_q[i][j]
                self.grads["W_k"][i][j] += grad_W_k[i][j]
                self.grads["W_v"][i][j] += grad_W_v[i][j]
                self.grads["W_o"][i][j] += grad_W_o[i][j]
            self.grads["b_q"][i] += grad_b_q[i]
            self.grads["b_k"][i] += grad_b_k[i]
            self.grads["b_v"][i] += grad_b_v[i]
            self.grads["b_o"][i] += grad_b_o[i]
        return dx_seq, dmemory

    def sgd_step(self, lr: float):
        """用累积的梯度做一次 SGD 参数更新"""
        for name in ("W_q", "W_k", "W_v", "W_o"):
            _sgd_apply(getattr(self, name), self.grads[name], lr)
        for name in ("b_q", "b_k", "b_v", "b_o"):
            _sgd_apply(getattr(self, name), self.grads[name], lr)


class TransformerDecoderLayer:
    """单层 Transformer Decoder：因果自注意力 → 交叉注意力 → FFN

    三个子层各带「残差 + LayerNorm」：res1 = x + self_attn(x)；
    res2 = ln1 + cross_attn(ln1, memory)；res3 = ln2 + ffn(ln2)。
    结构与 TransformerEncoderLayer 保持一致（残差加的是子层输入），
    多出来的第三块是交叉注意力。
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1,
                 causal: bool = True, window: int = None):
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout,
                                            causal=causal, window=window)
        self.cross_attn = CrossAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff)
        self.ln1 = LayerNorm(d_model)
        self.ln2 = LayerNorm(d_model)
        self.ln3 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)
        self.d_model = d_model

        self.zero_grad()
        self.x = self.res1 = self.res2 = self.res3 = None
        self.memory = None
        self.dropout1_masks = self.dropout2_masks = self.dropout3_masks = None

    def train(self):
        """训练模式：三处 Dropout 与注意力 dropout 全部生效"""
        self.self_attn.train()
        self.cross_attn.train()
        self.dropout1.train()
        self.dropout2.train()
        self.dropout3.train()
        return self

    def eval(self):
        """推理模式：关闭全部 dropout"""
        self.self_attn.eval()
        self.cross_attn.eval()
        self.dropout1.eval()
        self.dropout2.eval()
        self.dropout3.eval()
        return self

    def zero_grad(self):
        """清零 LayerNorm 梯度，同时重置子模块梯度"""
        d = self.d_model
        self.grads = {
            "ln1_gamma": [0.0] * d, "ln1_beta": [0.0] * d,
            "ln2_gamma": [0.0] * d, "ln2_beta": [0.0] * d,
            "ln3_gamma": [0.0] * d, "ln3_beta": [0.0] * d,
        }
        self.self_attn.zero_grad()
        self.cross_attn.zero_grad()
        self.ffn.zero_grad()

    def forward(self, x_seq, memory, memory_mask=None):
        d = self.d_model
        self.x = [list(row) for row in x_seq]
        self.memory = [list(row) for row in memory]
        n = len(self.x)

        attn_out = self.self_attn.forward(self.x)
        self.dropout1_masks = []
        attn_dropped = []
        for a in attn_out:
            dropped = self.dropout1.forward(a)
            attn_dropped.append(dropped)
            self.dropout1_masks.append(list(self.dropout1.mask))
        self.res1 = [[self.x[t][j] + attn_dropped[t][j] for j in range(d)]
                     for t in range(n)]
        ln1_out = [self.ln1.forward(v) for v in self.res1]

        cross_out = self.cross_attn.forward(ln1_out, memory, memory_mask)
        self.dropout2_masks = []
        cross_dropped = []
        for c in cross_out:
            dropped = self.dropout2.forward(c)
            cross_dropped.append(dropped)
            self.dropout2_masks.append(list(self.dropout2.mask))
        self.res2 = [[ln1_out[t][j] + cross_dropped[t][j] for j in range(d)]
                     for t in range(n)]
        ln2_out = [self.ln2.forward(v) for v in self.res2]

        ffn_out = self.ffn.forward(ln2_out)
        self.dropout3_masks = []
        ffn_dropped = []
        for f in ffn_out:
            dropped = self.dropout3.forward(f)
            ffn_dropped.append(dropped)
            self.dropout3_masks.append(list(self.dropout3.mask))
        self.res3 = [[ln2_out[t][j] + ffn_dropped[t][j] for j in range(d)]
                     for t in range(n)]
        return [self.ln3.forward(v) for v in self.res3]

    def backward(self, dout_seq):
        """dout_seq: (Tq, d_model) → (dx (Tq, d_model), dmemory (Tk, d_model))"""
        n = len(self.x)
        d = self.d_model

        def ln_backward(ln, res, d_in, key):
            d_out = []
            gg = [0.0] * d
            gb = [0.0] * d
            for t in range(n):
                ln.forward(res[t])          # 重放该 token 的缓存
                g_gamma, g_beta, dx = ln.backward(d_in[t])
                d_out.append(dx)
                for j in range(d):
                    gg[j] += g_gamma[j]
                    gb[j] += g_beta[j]
            for j in range(d):
                self.grads[key + "_gamma"][j] += gg[j]
                self.grads[key + "_beta"][j] += gb[j]
            return d_out

        d_res3 = ln_backward(self.ln3, self.res3, dout_seq, "ln3")
        d_ffn = [list(row) for row in d_res3]
        if self.dropout3_masks is not None:
            d_ffn = [[d_ffn[t][j] * self.dropout3_masks[t][j] for j in range(d)]
                     for t in range(n)]
        d_ln2 = self.ffn.backward(d_ffn)
        for t in range(n):
            for j in range(d):
                d_ln2[t][j] += d_res3[t][j]

        d_res2 = ln_backward(self.ln2, self.res2, d_ln2, "ln2")
        d_cross = [list(row) for row in d_res2]
        if self.dropout2_masks is not None:
            d_cross = [[d_cross[t][j] * self.dropout2_masks[t][j] for j in range(d)]
                       for t in range(n)]
        d_ln1_cross, dmemory = self.cross_attn.backward(d_cross)
        d_ln1 = [[d_ln1_cross[t][j] + d_res2[t][j] for j in range(d)] for t in range(n)]

        d_res1 = ln_backward(self.ln1, self.res1, d_ln1, "ln1")
        d_attn = [list(row) for row in d_res1]
        if self.dropout1_masks is not None:
            d_attn = [[d_attn[t][j] * self.dropout1_masks[t][j] for j in range(d)]
                      for t in range(n)]
        d_attn_out = self.self_attn.backward(d_attn)
        d_x = [[d_res1[t][j] + d_attn_out[t][j] for j in range(d)] for t in range(n)]
        return d_x, dmemory

    def sgd_step(self, lr: float):
        """对三个子层与三个 LayerNorm 各做一次 SGD 更新"""
        attn = self.self_attn
        for name in ("W_q", "W_k", "W_v", "W_o", "b_q", "b_k", "b_v", "b_o"):
            _sgd_apply(getattr(attn, name), attn.grads[name], lr)
        cross = self.cross_attn
        for name in ("W_q", "W_k", "W_v", "W_o", "b_q", "b_k", "b_v", "b_o"):
            _sgd_apply(getattr(cross, name), cross.grads[name], lr)
        ffn = self.ffn
        for name in ("W1", "b1", "W2", "b2"):
            _sgd_apply(getattr(ffn, name), ffn.grads[name], lr)
        for ln_name, ln in (("ln1", self.ln1), ("ln2", self.ln2), ("ln3", self.ln3)):
            _sgd_apply(ln.gamma, self.grads[ln_name + "_gamma"], lr)
            _sgd_apply(ln.beta, self.grads[ln_name + "_beta"], lr)


class TransformerDecoder:
    """堆叠多个 TransformerDecoderLayer（encoder-decoder 的 decoder）"""

    def __init__(self, d_model: int, num_heads: int, d_ff: int,
                 num_layers: int = 6, dropout: float = 0.1, causal: bool = True,
                 window: int = None):
        if isinstance(num_layers, bool) or not isinstance(num_layers, int) or num_layers < 1:
            raise ValueError(
                f"TransformerDecoder: num_layers 必须是正整数，收到 {num_layers!r}")
        self.causal = bool(causal)
        self.layers = [
            TransformerDecoderLayer(d_model, num_heads, d_ff, dropout,
                                    causal=causal, window=window)
            for _ in range(num_layers)
        ]

    def zero_grad(self):
        """重置所有层的梯度"""
        for layer in self.layers:
            layer.zero_grad()

    def train(self):
        """训练模式（转发给每一层）"""
        for layer in self.layers:
            layer.train()
        return self

    def eval(self):
        """推理模式（转发给每一层）"""
        for layer in self.layers:
            layer.eval()
        return self

    def forward(self, x_seq, memory, memory_mask=None):
        for layer in self.layers:
            x_seq = layer.forward(x_seq, memory, memory_mask)
        return x_seq

    def backward(self, dout_seq):
        """返回 (dx_seq, dmemory)；各层的 dmemory 相加（共享同一份 memory）"""
        d_memory = None
        for layer in reversed(self.layers):
            dout_seq, dm = layer.backward(dout_seq)
            if d_memory is None:
                d_memory = dm
            else:
                for t in range(len(dm)):
                    for j in range(len(dm[0])):
                        d_memory[t][j] += dm[t][j]
        return dout_seq, d_memory

    def sgd_step(self, lr: float):
        """对每一层做一次 SGD 参数更新"""
        for layer in self.layers:
            layer.sgd_step(lr)


class PositionalEncoding:
    """正弦/余弦位置编码，加到输入序列上"""

    def __init__(self, d_model: int, max_len: int = 5000):
        """max_len: **初始预计算的行数（容量），不是长度上限**。

        forward 遇到更长序列时会按需扩展编码表（`_ensure_len`），不会截断。
        调大它可以提前分配、省掉运行期扩展。
        """
        self.d_model = d_model
        self.pe = [[0.0] * d_model for _ in range(max_len)]
        for pos in range(max_len):
            for i in range(0, d_model, 2):
                div = math.exp(i * -math.log(10000.0) / d_model)
                self.pe[pos][i] = math.sin(pos * div)
                if i + 1 < d_model:
                    self.pe[pos][i + 1] = math.cos(pos * div)

    def _ensure_len(self, n: int):
        """按需把位置编码表扩展到 n 行（原实现超过 max_len 会静默截断）"""
        for pos in range(len(self.pe), n):
            row = [0.0] * self.d_model
            for i in range(0, self.d_model, 2):
                div = math.exp(i * -math.log(10000.0) / self.d_model)
                row[i] = math.sin(pos * div)
                if i + 1 < self.d_model:
                    row[i + 1] = math.cos(pos * div)
            self.pe.append(row)

    def forward(self, x_seq):
        """x_seq: (seq_len, d_model) → x_seq + PE（超过 max_len 时自动扩展）"""
        for t, row in enumerate(x_seq):
            if len(row) != self.d_model:
                raise ValueError(
                    f"PositionalEncoding.forward: 第 {t} 个 token 的维度 {len(row)} "
                    f"与 d_model={self.d_model} 不一致"
                )
        self._ensure_len(len(x_seq))
        return [[x_seq[t][j] + self.pe[t][j] for j in range(self.d_model)]
                for t in range(len(x_seq))]

    def backward(self, dout_seq):
        return dout_seq  # 无参数，直通


# ============================================================================
#  Embedding / Residual / Sequential
# ============================================================================

class Embedding:
    """词嵌入层：查表操作，将索引映射为稠密向量"""

    def __init__(self, vocab_size: int, embed_dim: int):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        std = math.sqrt(2.0 / vocab_size)
        self.weight = [[random.gauss(0, std) for _ in range(embed_dim)] for _ in range(vocab_size)]
        self.zero_grad()
        self.indices = None  # 缓存查询的索引列表

    def forward(self, indices):
        """indices: List[int] → List[List[float]] (seq_len, embed_dim)"""
        # 负数索引会静默回绕到词表末尾，必须拦掉
        for t, idx in enumerate(indices):
            if not isinstance(idx, int) or not 0 <= idx < self.vocab_size:
                raise ValueError(
                    f"Embedding.forward: 第 {t} 个索引 {idx!r} 越界"
                    f"（词表大小 {self.vocab_size}）"
                )
        self.indices = indices
        return [list(self.weight[idx]) for idx in indices]

    def zero_grad(self):
        """初始化/清零梯度缓冲区"""
        self.grad_weight = [[0.0] * self.embed_dim for _ in range(self.vocab_size)]

    def backward(self, dout_seq):
        """dout_seq: (seq_len, embed_dim) → 返回 None（无输入梯度），内部累积 grad_weight"""
        for t, idx in enumerate(self.indices):
            for j in range(self.embed_dim):
                self.grad_weight[idx][j] += dout_seq[t][j]
        return None

    def sgd_step(self, lr: float):
        """对词向量做一次 SGD 更新（原实现只累积梯度、权重永远不变）"""
        for idx, row in enumerate(self.grad_weight):
            wrow = self.weight[idx]
            for j in range(self.embed_dim):
                wrow[j] -= lr * row[j]
        self.zero_grad()


# 因果语言模型 (CausalLM)：causal mask 的直接用途。
# 它组合 Embedding + PositionalEncoding + TransformerEncoder(causal=True)，
# 所以紧跟在 Embedding 之后定义（三个依赖类都在上面）。

def _zeros_like(p):
    """按参数结构生成同形状的零数组（Adam 的一阶/二阶矩用）"""
    if p and isinstance(p[0], (list, tuple)):
        return [[0.0] * len(row) for row in p]
    return [0.0] * len(p)


def _adam_apply(param, grad, m, v, t, lr, b1=0.9, b2=0.999, eps=1e-8):
    """就地 Adam 更新；param / grad / m / v 结构一致（1D 或 2D 列表）"""
    c1 = 1.0 - b1 ** t
    c2 = 1.0 - b2 ** t
    if param and isinstance(param[0], (list, tuple)):
        for i in range(len(param)):
            pi, gi, mi, vi = param[i], grad[i], m[i], v[i]
            for j in range(len(pi)):
                g = gi[j]
                mi[j] = b1 * mi[j] + (1.0 - b1) * g
                vi[j] = b2 * vi[j] + (1.0 - b2) * g * g
                pi[j] -= lr * (mi[j] / c1) / (math.sqrt(vi[j] / c2) + eps)
    else:
        for i in range(len(param)):
            g = grad[i]
            m[i] = b1 * m[i] + (1.0 - b1) * g
            v[i] = b2 * v[i] + (1.0 - b2) * g * g
            param[i] -= lr * (m[i] / c1) / (math.sqrt(v[i] / c2) + eps)


def _log_softmax_row(v):
    """对一维列表做 log-softmax（数值稳定版，供束搜索累加对数概率用）"""
    mx = max(v)
    if mx == float("-inf"):
        return [float("-inf")] * len(v)
    exps = [math.exp(x - mx) for x in v]
    log_s = math.log(sum(exps)) + mx
    return [x - log_s for x in v]


def _lm_copy_into(cur, saved, name):
    """递归把存档里的嵌套 list 写进 cur，顺带校验形状、类型与有限性"""
    if not isinstance(cur, list):
        raise ValueError(f"LM 参数 {name}: 无法写入标量槽位")
    if not isinstance(saved, list):
        raise ValueError(f"LM 参数 {name}: 期望数组，存档里是 {type(saved).__name__}")
    if len(cur) != len(saved):
        raise ValueError(
            f"LM 参数 {name}: 长度不符（模型 {len(cur)}，存档 {len(saved)}）")
    for i in range(len(cur)):
        if isinstance(cur[i], list):
            _lm_copy_into(cur[i], saved[i], name)
            continue
        val = saved[i]
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            raise ValueError(f"LM 参数 {name}: 期望数值，存档里是 {val!r}")
        val = float(val)
        if val != val or val in (float("inf"), float("-inf")):
            raise ValueError(f"LM 参数 {name}: 存档里有非有限值 {val!r}")
        cur[i] = val


def _lm_save(lm, kind, filepath, include_optimizer=False, quiet=False):
    """把 LM 的配置 + 全部参数（可选 Adam 状态）写成 JSON"""
    if not isinstance(filepath, str) or not filepath:
        raise ValueError(f"save: filepath 必须是非空字符串，收到 {filepath!r}")
    data = {
        "format": "ai_lm_v1",
        "kind": kind,
        "config": lm._config(),
        "params": {name: param for name, param, _ in lm._named_params()},
    }
    if include_optimizer:
        data["optimizer"] = {
            "step_count": int(getattr(lm, "step_count", 0)),
            "m": {k: v for k, v in getattr(lm, "_adam_m", {}).items()},
            "v": {k: v for k, v in getattr(lm, "_adam_v", {}).items()},
        }
    with io.open(filepath, "w", encoding="utf-8") as f:
        # allow_nan=False：参数发散成 NaN/Inf 时宁可保存失败，也不写出
        # 不合 RFC-4627 的 JSON（save_model 一直是这样做的）
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
    if not quiet:
        print(f"已保存 {kind} 模型: {filepath}")
    return filepath


def _lm_load(filepath, restore_optimizer=False):
    """从 `_lm_save` 写出的 JSON 恢复 LM（结构与参数逐位还原）"""
    with io.open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or data.get("format") != "ai_lm_v1":
        raise ValueError(f"load: {filepath} 不是 ai_lm_v1 格式的 LM 存档")
    kind = data.get("kind")
    classes = {"causal": CausalLM, "seq2seq": Seq2SeqLM}
    if kind not in classes:
        raise ValueError(f"load: 未知的模型类型 {kind!r}（支持 {sorted(classes)}）")
    cfg = data.get("config") or {}
    # 存档里 window/num_kv_heads 可能是 null，构造器接受 None
    try:
        model = classes[kind](**cfg)
    except TypeError as e:
        raise ValueError(f"load: 存档的配置与构造器不匹配（{e}）")
    saved_params = data.get("params") or {}
    named = {name: param for name, param, _ in model._named_params()}
    missing = sorted(set(named) - set(saved_params))
    extra = sorted(set(saved_params) - set(named))
    if missing or extra:
        raise ValueError(
            f"load: 参数块与存档不一致（缺 {missing[:3]}，多 {extra[:3]}）")
    for name, param in named.items():
        _lm_copy_into(param, saved_params[name], name)
    if restore_optimizer:
        opt = data.get("optimizer")
        if not opt:
            raise ValueError(
                "load: 存档里没有优化器状态（保存时需要 include_optimizer=True），"
                "无法 restore_optimizer=True")
        model.step_count = int(opt.get("step_count", 0))
        for key, store in (("m", model._adam_m), ("v", model._adam_v)):
            saved_bufs = opt.get(key) or {}
            store.clear()
            for name, param, _ in model._named_params():
                if name not in saved_bufs:
                    raise ValueError(f"load: 优化器状态缺少参数块 {name!r}")
                buf = _zeros_like(param)
                _lm_copy_into(buf, saved_bufs[name], f"optimizer.{key}.{name}")
                store[name] = buf
    return model


def _check_pad_mask(mask, length, who):
    """校验 pad_mask：与序列等长的 0/1 列表，至少一个有效位；返回 bool 列表

    只支持**右填充**（有效位在前、0 在后）：CausalLM 用 `valid[:-1]` 当 key_mask，
    左填充会把 key_mask 截错位置，算出来的 loss 是错的但不会报错。
    """
    if not isinstance(mask, (list, tuple)):
        raise ValueError(f"{who}: pad_mask 必须是列表，收到 {type(mask).__name__}")
    if len(mask) != length:
        raise ValueError(f"{who}: pad_mask 长度 {len(mask)} 与序列长度 {length} 不一致")
    out = []
    for i, m in enumerate(mask):
        if isinstance(m, bool):
            out.append(m)
        elif isinstance(m, int) and m in (0, 1):
            out.append(bool(m))
        else:
            raise ValueError(f"{who}: pad_mask[{i}] 必须是 0 或 1，收到 {m!r}")
    if not any(out):
        raise ValueError(f"{who}: pad_mask 至少要有 1 个有效位置")
    seen_pad = False
    for i, m in enumerate(out):
        if not m:
            seen_pad = True
        elif seen_pad:
            raise ValueError(
                f"{who}: pad_mask 只支持右填充（0 之后不能再出现 1，位置 {i}）")
    return out


def _check_top_k(top_k, who):
    """校验 top_k：非负整数（0 表示不过滤）

    float/bool/负数都会让 `order[:int(top_k)]` 静默退化成空集或差一位，
    采样结果变成「永远是最后一个 token」或丢掉最高概率项，所以一律拒绝。
    """
    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise ValueError(f"{who}: top_k 必须是 >= 0 的整数（0 表示不过滤），收到 {top_k!r}")
    if top_k < 0:
        raise ValueError(f"{who}: top_k 必须 >= 0，收到 {top_k!r}")
    return top_k


def _check_token_ids(ids, vocab_size, who):
    """校验一串 token id 都在 [0, vocab_size) 内

    Python 的负索引会让 `probs[-1]` 静默指向「最后一类」，loss 与梯度全算在错类上；
    越界的正数反而抛 IndexError。统一在这里报 ValueError。
    """
    if not isinstance(ids, (list, tuple)):
        raise ValueError(f"{who}: token id 序列必须是列表，收到 {type(ids).__name__}")
    out = []
    for i, v in enumerate(ids):
        if isinstance(v, bool) or not isinstance(v, int) or not 0 <= v < vocab_size:
            raise ValueError(
                f"{who}: 第 {i} 个 token id 非法（{v!r}），必须在 [0, {vocab_size}) 内")
        out.append(v)
    return out


def _check_finite_logits(logits, who):
    """logits 出现 NaN/Inf 说明参数已经坏了；直接报错，别让它被 loss 的 clamp 掩盖"""
    for t, row in enumerate(logits):
        for v in row:
            if v != v or v == float("inf") or v == float("-inf"):
                raise ValueError(f"{who}: 第 {t} 行 logits 含 NaN/Inf，参数可能已损坏")


def _check_penalty_args(repetition_penalty, presence_penalty, frequency_penalty,
                        no_repeat_ngram_size, who):
    """校验生成惩罚参数（重复/存在/频率/禁重复 n-gram）"""
    for name, val in (("repetition_penalty", repetition_penalty),
                      ("presence_penalty", presence_penalty),
                      ("frequency_penalty", frequency_penalty)):
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            raise ValueError(f"{who}: {name} 必须是数字，收到 {val!r}")
        if val != val or val in (float("inf"), float("-inf")):
            raise ValueError(f"{who}: {name} 必须是有限值，收到 {val!r}")
    if repetition_penalty <= 0:
        raise ValueError(
            f"{who}: repetition_penalty 必须 > 0，收到 {repetition_penalty!r}")
    if isinstance(no_repeat_ngram_size, bool) or not isinstance(no_repeat_ngram_size, int) \
            or no_repeat_ngram_size < 0:
        raise ValueError(
            f"{who}: no_repeat_ngram_size 必须是 >= 0 的整数，收到 {no_repeat_ngram_size!r}")
    if no_repeat_ngram_size == 1:
        # n=1 会把所有已出现的 token 全禁掉，等于只能输出新 token，几乎必然无解
        raise ValueError(f"{who}: no_repeat_ngram_size=1 没有意义（会禁用所有已出现的 token）")


def _mask_min_p(logits, min_p: float):
    """把概率低于 min_p * max_prob 的 token 置 -inf（min_p<=0 时原样返回）"""
    if min_p <= 0.0:
        return logits
    probs = _softmax_row(logits)
    peak = max(probs)
    thresh = min_p * peak
    return [v if probs[i] >= thresh else float("-inf") for i, v in enumerate(logits)]


def _check_min_p(min_p, who):
    """校验 min_p（(0,1) 内的阈值；0 表示不过滤）"""
    if isinstance(min_p, bool) or not isinstance(min_p, (int, float)):
        raise ValueError(f"{who}: min_p 必须是数字，收到 {min_p!r}")
    if min_p != min_p or not (0.0 <= min_p < 1.0):
        raise ValueError(f"{who}: min_p 必须在 [0,1) 内，收到 {min_p!r}")
    return float(min_p)


def _apply_generation_penalties(logits, ids, repetition_penalty: float = 1.0,
                                presence_penalty: float = 0.0,
                                frequency_penalty: float = 0.0,
                                no_repeat_ngram_size: int = 0):
    """按已生成的 token 调整 logits（默认参数下原样返回，保证旧行为逐位不变）

    - repetition_penalty: >1 抑制重复（正 logit 除以它、负 logit 乘以它）
    - presence_penalty: 出现过就扣一次
    - frequency_penalty: 按出现次数扣
    - no_repeat_ngram_size=n: 禁止让末尾 (n-1)-gram 立刻再重复一次
    """
    has_ngram = no_repeat_ngram_size and no_repeat_ngram_size >= 2
    if (repetition_penalty == 1.0 and presence_penalty == 0.0
            and frequency_penalty == 0.0 and not has_ngram):
        return logits

    out = list(logits)
    vocab = len(out)
    counts = {}
    for tok in ids:
        if isinstance(tok, int) and not isinstance(tok, bool) and 0 <= tok < vocab:
            counts[tok] = counts.get(tok, 0) + 1

    if repetition_penalty != 1.0:
        for tok in counts:
            if out[tok] > 0:
                out[tok] = out[tok] / repetition_penalty
            else:
                out[tok] = out[tok] * repetition_penalty
    if presence_penalty:
        for tok in counts:
            out[tok] -= presence_penalty
    if frequency_penalty:
        for tok, c in counts.items():
            out[tok] -= frequency_penalty * c
    if has_ngram:
        n = int(no_repeat_ngram_size)
        if len(ids) >= n - 1:
            prefix = list(ids[len(ids) - (n - 1):])
            for i in range(len(ids) - (n - 1)):
                if list(ids[i:i + n - 1]) == prefix:
                    nxt = ids[i + n - 1]
                    if isinstance(nxt, int) and not isinstance(nxt, bool) and 0 <= nxt < vocab:
                        out[nxt] = float("-inf")
    return out


class MaskedLM:
    """掩码语言模型（BERT 式）：Embedding → 位置编码 → **双向** Transformer → 词表头

    与 `CausalLM` 的区别只有两点：
      1. 注意力**不加因果掩码**，每个位置都能看到左右上下文；
      2. 训练目标是「还原被遮住的位置」，而不是预测下一个 token。

    遮罩策略与 BERT 一致：随机选 `mask_prob` 比例的位置，其中 80% 换成
    `mask_id`、10% 换成随机 token、10% 保持不变（保持不变的位置仍然要预测，
    否则模型会只学会「看到 mask 才输出」）。`mask_id` 默认取 `vocab_size`
    （额外占一行词向量），不会与真实 token 冲突。

    用法::

        lm = MaskedLM(vocab_size=10, d_model=16, num_heads=2, num_layers=1)
        lm.fit([[1, 2, 3, 4, 5, 6]], epochs=200, lr=0.01)
        lm.predict([1, 9, 3, 4, 9, 6], [1, 4])   # 填空
    """

    def __init__(self, vocab_size: int, d_model: int = 32, num_heads: int = 4,
                 d_ff: int = 64, num_layers: int = 2, max_len: int = 64,
                 dropout: float = 0.0, mask_prob: float = 0.15,
                 replace_prob: float = 0.1):
        if not isinstance(vocab_size, int) or isinstance(vocab_size, bool) or vocab_size < 1:
            raise ValueError(f"MaskedLM: vocab_size 必须是正整数，收到 {vocab_size!r}")
        if not isinstance(max_len, int) or isinstance(max_len, bool) or max_len < 1:
            raise ValueError(f"MaskedLM: max_len 必须 >= 1，收到 {max_len!r}")
        if not isinstance(num_layers, int) or isinstance(num_layers, bool) or num_layers < 1:
            raise ValueError(f"MaskedLM: num_layers 必须 >= 1，收到 {num_layers!r}")
        for name, val in (("mask_prob", mask_prob), ("replace_prob", replace_prob)):
            if isinstance(val, bool) or not isinstance(val, (int, float)) or not 0.0 < float(val) < 1.0:
                raise ValueError(f"MaskedLM: {name} 必须在 (0,1) 内，收到 {val!r}")
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_len = max_len
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.dropout = dropout
        self.mask_prob = float(mask_prob)
        self.replace_prob = float(replace_prob)
        # mask 用一个词表之外的 id，避免和真实 token 混淆
        self.mask_id = vocab_size

        self.embedding = Embedding(vocab_size + 1, d_model)
        self.pos = PositionalEncoding(d_model, max_len)
        self.encoder = TransformerEncoder(d_model, num_heads, d_ff, num_layers,
                                          dropout, causal=False)

        std = math.sqrt(2.0 / d_model)
        self.W_out = [[random.gauss(0, std) for _ in range(d_model)] for _ in range(vocab_size)]
        self.b_out = [0.0] * vocab_size
        self.zero_grad()
        self.step_count = 0
        self._adam_m = {}
        self._adam_v = {}
        self.h = None

    def train(self):
        """训练模式（Dropout 生效）"""
        self.encoder.train()
        return self

    def eval(self):
        """推理模式（关闭 Dropout，同一输入输出确定）"""
        self.encoder.eval()
        return self

    def zero_grad(self):
        """清零全部梯度"""
        self.embedding.zero_grad()
        self.encoder.zero_grad()
        self.grads = {
            "W_out": [[0.0] * self.d_model for _ in range(self.vocab_size)],
            "b_out": [0.0] * self.vocab_size,
        }

    def _named_params(self):
        """产出 (名字, 参数, 梯度)；必须在 `zero_grad()` 之后调用"""
        yield "emb.weight", self.embedding.weight, self.embedding.grad_weight
        for li, layer in enumerate(self.encoder.layers):
            attn = layer.self_attn
            for k in ("W_q", "W_k", "W_v", "W_o", "b_q", "b_k", "b_v", "b_o"):
                yield f"L{li}.attn.{k}", getattr(attn, k), attn.grads[k]
            ffn = layer.ffn
            for k in ("W1", "b1", "W2", "b2"):
                yield f"L{li}.ffn.{k}", getattr(ffn, k), ffn.grads[k]
            yield f"L{li}.ln1.gamma", layer.ln1.gamma, layer.grads["ln1_gamma"]
            yield f"L{li}.ln1.beta", layer.ln1.beta, layer.grads["ln1_beta"]
            yield f"L{li}.ln2.gamma", layer.ln2.gamma, layer.grads["ln2_gamma"]
            yield f"L{li}.ln2.beta", layer.ln2.beta, layer.grads["ln2_beta"]
        yield "out.W", self.W_out, self.grads["W_out"]
        yield "out.b", self.b_out, self.grads["b_out"]

    def param_count(self):
        """参数总数（标量个数）"""
        return sum(len(p) if not isinstance(p[0], list) else
                   sum(len(r) for r in p)
                   for _, p, _ in self._named_params())

    def mask_tokens(self, ids, rng=None):
        """按 BERT 策略遮罩，返回 (corrupted_ids, positions, targets)

        positions 是**被遮住的位置下标**（升序），targets 是对应位置的原始 token。
        至少遮一个位置：短序列随机选时可能一个都没选中，这里强制补一个。
        """
        ids = _check_token_ids(ids, self.vocab_size, "MaskedLM.mask_tokens")
        rng = random if rng is None else rng
        if not ids:
            raise ValueError("MaskedLM.mask_tokens: ids 不能为空")
        positions = [t for t in range(len(ids)) if rng.random() < self.mask_prob]
        if not positions:
            positions = [rng.randrange(len(ids))]
        corrupted = list(ids)
        for t in positions:
            r = rng.random()
            if r < 0.8:
                corrupted[t] = self.mask_id
            elif r < 0.9:
                corrupted[t] = rng.randrange(self.vocab_size)
            # 剩下 10% 保持原样（但仍计入预测目标）
        return corrupted, positions, [ids[t] for t in positions]

    def forward(self, ids):
        """ids: List[int] → logits (seq_len, vocab_size)（bidirectional，无因果掩码）"""
        if not ids:
            raise ValueError("MaskedLM.forward: ids 不能为空")
        if len(ids) > self.max_len:
            raise ValueError(
                f"MaskedLM.forward: 序列长度 {len(ids)} 超过 max_len={self.max_len}")
        ids = _check_token_ids(ids, self.vocab_size + 1, "MaskedLM.forward")
        emb = self.embedding.forward(ids)
        x = self.pos.forward(emb)
        h = self.encoder.forward(x)
        self.h = h
        logits = _matmul(h, _transpose(self.W_out))
        for t in range(len(h)):
            for v in range(self.vocab_size):
                logits[t][v] += self.b_out[v]
        return logits

    def _loss_and_backward(self, ids, positions, targets, grad_scale: float = 1.0):
        """只在 positions 上算交叉熵并反传，返回该条序列的平均 loss

        ids 是**遮罩后**的输入，targets 是对应位置上的原始 token
        （两者必须分开传：遮罩位置上的 ids 是 mask_id，不能当答案）。
        """
        if not isinstance(positions, (list, tuple)) or not positions:
            raise ValueError("MaskedLM._loss_and_backward: positions 不能为空")
        if not isinstance(targets, (list, tuple)) or len(targets) != len(positions):
            raise ValueError(
                "MaskedLM._loss_and_backward: targets 必须与 positions 等长"
                f"（{len(positions)} 个位置，收到 {targets!r}）")
        pos = []
        for p in positions:
            if isinstance(p, bool) or not isinstance(p, int):
                raise ValueError(f"MaskedLM._loss_and_backward: 位置必须是整数，收到 {p!r}")
            if not 0 <= p < len(ids):
                raise ValueError(
                    f"MaskedLM._loss_and_backward: 位置 {p} 越界（序列长 {len(ids)}）")
            pos.append(p)
        ids = _check_token_ids(ids, self.vocab_size + 1, "MaskedLM._loss_and_backward")
        logits = self.forward(ids)
        _check_finite_logits(logits, "MaskedLM._loss_and_backward")

        d_logits = [[0.0] * self.vocab_size for _ in range(len(ids))]
        loss = 0.0
        n = len(pos)
        for j, t in enumerate(pos):
            probs = _softmax_row(logits[t])
            target = targets[j]
            if isinstance(target, bool) or not isinstance(target, int) or \
                    not 0 <= target < self.vocab_size:
                raise ValueError(
                    f"MaskedLM._loss_and_backward: 位置 {t} 的目标 {target!r} 越界"
                    f"（词表 {self.vocab_size}）")
            p = probs[target]
            loss += -math.log(p if p > 1e-12 else 1e-12)
            row = d_logits[t]
            row[:] = probs
            row[target] -= 1.0
        loss /= n
        inv = grad_scale / n
        d_logits = [[v * inv for v in row] for row in d_logits]

        h = self.h
        for v in range(self.vocab_size):
            g_row = self.grads["W_out"][v]
            for t in range(len(h)):
                d = d_logits[t][v]
                if d != 0.0:
                    for j in range(self.d_model):
                        g_row[j] += d * h[t][j]
        for v in range(self.vocab_size):
            s = 0.0
            for t in range(len(h)):
                d = d_logits[t][v]
                if d != 0.0:
                    s += d
            self.grads["b_out"][v] += s

        dh = []
        for t in range(len(h)):
            row = [0.0] * self.d_model
            dl = d_logits[t]
            for j in range(self.d_model):
                s = 0.0
                for v in range(self.vocab_size):
                    d = dl[v]
                    if d != 0.0:
                        s += d * self.W_out[v][j]
                row[j] = s
            dh.append(row)

        d_emb = self.encoder.backward(dh)
        self.embedding.backward(d_emb)
        return loss

    def adam_step(self, lr: float):
        """对全部参数做一次 Adam 更新（梯度已由 `_loss_and_backward` 累积）"""
        self.step_count += 1
        t = self.step_count
        for name, param, grad in self._named_params():
            m = self._adam_m.get(name)
            v = self._adam_v.get(name)
            if m is None:
                m = _zeros_like(param)
                v = _zeros_like(param)
                self._adam_m[name] = m
                self._adam_v[name] = v
            _adam_apply(param, grad, m, v, t, lr)

    def fit(self, sentences, epochs: int = 1, lr: float = 3e-3,
            shuffle: bool = True, seed=None, verbose: bool = False):
        """在若干条 token 序列上做掩码语言建模，返回每个 epoch 的平均 loss

        每个 epoch 对每条序列**重新采样一次遮罩**（与 BERT 一样），
        所以同一句话会反复以不同遮罩出现。
        """
        seqs = [list(s) for s in sentences]
        if not seqs:
            raise ValueError("MaskedLM.fit: sentences 不能为空")
        for s in seqs:
            if not s:
                raise ValueError("MaskedLM.fit: 每条序列不能为空")
            if len(s) > self.max_len:
                raise ValueError(
                    f"MaskedLM.fit: 序列长度 {len(s)} 超过 max_len={self.max_len}")
        if epochs < 0:
            raise ValueError(f"MaskedLM.fit: epochs 不能为负，收到 {epochs!r}")
        rng = random.Random(seed) if seed is not None else random
        history = []
        self.train()
        for ep in range(int(epochs)):
            order = list(range(len(seqs)))
            if shuffle:
                rng.shuffle(order)
            total = 0.0
            for i in order:
                corrupted, positions, targets = self.mask_tokens(seqs[i], rng)
                self.zero_grad()
                total += self._loss_and_backward(corrupted, positions, targets)
                self.adam_step(lr)
            avg = total / len(seqs)
            history.append(avg)
            if verbose:
                print(f"epoch {ep + 1}/{epochs} loss={avg:.4f}")
        return history

    def evaluate(self, sentences, seed: int = 0):
        """不更新参数的掩码 loss（eval 模式）

        固定 seed 让遮罩可复现，从而不同模型/不同轮次之间可比。
        """
        seqs = [list(s) for s in sentences]
        if not seqs:
            raise ValueError("MaskedLM.evaluate: sentences 不能为空")
        rng = random.Random(seed)
        was_training = self.encoder.layers[0].self_attn.training
        self.eval()
        try:
            total = 0.0
            for s in seqs:
                corrupted, positions, _targets = self.mask_tokens(s, random.Random(seed))
                logits = self.forward(corrupted)
                _check_finite_logits(logits, "MaskedLM.evaluate")
                for t, p in enumerate(positions):
                    probs = _softmax_row(logits[p])
                    target = s[p]
                    total += -math.log(probs[target] if probs[target] > 1e-12 else 1e-12)
            return total / max(1, sum(1 for _ in seqs))
        finally:
            if was_training:
                self.train()
            else:
                self.eval()

    def predict(self, ids, positions):
        """把 positions 处的 token 预测出来（argmax），返回 List[int]"""
        if not isinstance(positions, (list, tuple)) or not positions:
            raise ValueError("MaskedLM.predict: positions 不能为空")
        logits = self.forward(ids)
        out = []
        for p in positions:
            if isinstance(p, bool) or not isinstance(p, int) or not 0 <= p < len(ids):
                raise ValueError(f"MaskedLM.predict: 位置 {p!r} 非法")
            row = logits[p]
            best = 0
            for v in range(1, self.vocab_size):
                if row[v] > row[best]:
                    best = v
            out.append(best)
        return out


class CausalLM:
    """字符/词级因果语言模型（GPT 式）：Embedding → 位置编码 → 因果 Transformer → 词表头

    关键在 `MultiHeadAttention(causal=True)`：位置 t 只能看到 0..t，
    因此**一次前向就能并行算出所有位置的「下一个 token」预测**，
    不需要 RNN 那样串行递推 —— 这正是 LLM 预训练的做法。

    训练目标：给定 ids，位置 t 的 logits 与 ids[t+1] 做交叉熵。
    参数：Adam（内部实现，超参 0.9/0.999/1e-8）。

    用法::

        lm = CausalLM(vocab_size=3, d_model=16, num_heads=2, num_layers=1)
        lm.fit([[0, 1, 2, 0, 1, 2]], epochs=60, lr=0.01)
        lm.generate([0], max_new_tokens=6)
    """

    def __init__(self, vocab_size: int, d_model: int = 32, num_heads: int = 4,
                 d_ff: int = 64, num_layers: int = 2, max_len: int = 256,
                 dropout: float = 0.0, rope: bool = False, use_abs_pos=None,
                 window: int = None, num_kv_heads: int = None,
                 alibi: bool = False):
        if not isinstance(vocab_size, int) or isinstance(vocab_size, bool) or vocab_size < 1:
            raise ValueError(f"CausalLM: vocab_size 必须是正整数，收到 {vocab_size!r}")
        if not isinstance(max_len, int) or isinstance(max_len, bool) or max_len < 1:
            raise ValueError(f"CausalLM: max_len 必须 >= 1，收到 {max_len!r}")
        if not isinstance(num_layers, int) or isinstance(num_layers, bool) or num_layers < 1:
            raise ValueError(f"CausalLM: num_layers 必须 >= 1，收到 {num_layers!r}")
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_len = max_len
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.dropout = dropout
        self.window = window
        self.num_kv_heads = num_kv_heads

        self.rope = bool(rope)
        # RoPE 已自带相对位置信息，默认就不再叠加绝对位置编码（Llama 的做法）；
        # 想两者都用可以显式传 use_abs_pos=True
        self.use_abs_pos = (not self.rope) if use_abs_pos is None else bool(use_abs_pos)

        self.embedding = Embedding(vocab_size, d_model)
        self.pos = PositionalEncoding(d_model, max_len)
        self.alibi = bool(alibi)
        self.encoder = TransformerEncoder(d_model, num_heads, d_ff, num_layers,
                                          dropout, causal=True, rope=self.rope,
                                          window=window, num_kv_heads=num_kv_heads,
                                          alibi=self.alibi)

        std = math.sqrt(2.0 / d_model)
        self.W_out = [[random.gauss(0, std) for _ in range(d_model)] for _ in range(vocab_size)]
        self.b_out = [0.0] * vocab_size
        self.zero_grad()
        self.step_count = 0
        self._adam_m = {}
        self._adam_v = {}
        self.h = None    # 最近一次 forward 的编码器输出（反向要用）

    def train(self):
        """训练模式（Dropout 生效）"""
        self.encoder.train()
        return self

    def eval(self):
        """推理模式（关闭 Dropout，同一输入输出确定）"""
        self.encoder.eval()
        return self

    def zero_grad(self):
        """清零全部梯度（词向量、各层、输出头）"""
        self.embedding.zero_grad()
        self.encoder.zero_grad()
        self.grads = {
            "W_out": [[0.0] * self.d_model for _ in range(self.vocab_size)],
            "b_out": [0.0] * self.vocab_size,
        }

    def _named_params(self):
        """产出 (名字, 参数, 梯度)；名字让 Adam 的一阶/二阶矩跨步保持

        必须在 `zero_grad()` 之后调用：各子模块的 grads 是重建成的新列表，
        直接持有旧引用会更新到过期的缓冲区上。
        """
        yield "emb.weight", self.embedding.weight, self.embedding.grad_weight
        for li, layer in enumerate(self.encoder.layers):
            attn = layer.self_attn
            for k in ("W_q", "W_k", "W_v", "W_o", "b_q", "b_k", "b_v", "b_o"):
                yield f"L{li}.attn.{k}", getattr(attn, k), attn.grads[k]
            ffn = layer.ffn
            for k in ("W1", "b1", "W2", "b2"):
                yield f"L{li}.ffn.{k}", getattr(ffn, k), ffn.grads[k]
            yield f"L{li}.ln1.gamma", layer.ln1.gamma, layer.grads["ln1_gamma"]
            yield f"L{li}.ln1.beta", layer.ln1.beta, layer.grads["ln1_beta"]
            yield f"L{li}.ln2.gamma", layer.ln2.gamma, layer.grads["ln2_gamma"]
            yield f"L{li}.ln2.beta", layer.ln2.beta, layer.grads["ln2_beta"]
        yield "out.W", self.W_out, self.grads["W_out"]
        yield "out.b", self.b_out, self.grads["b_out"]

    def param_count(self):
        """参数总数（标量个数）"""
        return sum(len(p) if not isinstance(p[0], list) else
                   sum(len(r) for r in p)
                   for _, p, _ in self._named_params())

    def forward(self, ids, pad_mask=None):
        """ids: List[int] → logits (seq_len, vocab_size)（未经 softmax）

        pad_mask: 与 ids 等长的 0/1 列表，0 表示填充位置（不当 key）。
        填充必须靠右，真实 token 的位置编号才不会被挪动。None 时行为与旧版一致。
        """
        if not ids:
            raise ValueError("CausalLM.forward: ids 不能为空")
        if len(ids) > self.max_len:
            raise ValueError(
                f"CausalLM.forward: 序列长度 {len(ids)} 超过 max_len={self.max_len}"
            )
        key_mask = None
        if pad_mask is not None:
            valid = _check_pad_mask(pad_mask, len(ids), "CausalLM.forward")
            key_mask = valid
        emb = self.embedding.forward(ids)
        x = self.pos.forward(emb) if self.use_abs_pos else emb
        h = self.encoder.forward(x, key_mask)
        self.h = h
        logits = _matmul(h, _transpose(self.W_out))
        for t in range(len(h)):
            for v in range(self.vocab_size):
                logits[t][v] += self.b_out[v]
        return logits

    def _forward_step(self, token_id: int, pos: int, caches):
        """单 token 增量前向：返回该位置 logits（长度 vocab_size）

        caches: 每层一个 dict；pos 是该 token 在**当前窗口**里的位置。
        与 `forward(窗口)[-1]` 逐位一致（前提：窗口没有滑动，位置未变）。
        """
        d = self.d_model
        if not isinstance(token_id, int) or not 0 <= token_id < self.vocab_size:
            raise ValueError(
                f"CausalLM._forward_step: token {token_id!r} 越界（词表 {self.vocab_size}）")
        row = self.embedding.weight[token_id]
        if self.use_abs_pos:
            self.pos._ensure_len(pos + 1)
            pe = self.pos.pe[pos]
            x = [row[j] + pe[j] for j in range(d)]
        else:
            x = list(row)
        x = self.encoder.forward_step(x, caches)
        return [sum(map(_mul, self.W_out[v], x)) + self.b_out[v]
                for v in range(self.vocab_size)]

    def _loss_and_backward(self, ids, pad_mask=None, grad_scale: float = 1.0):
        """单条序列的交叉熵（对**有效**预测位置取平均），并累积全部梯度；返回 loss

        pad_mask: 与 ids 等长的 0/1 列表，0 表示填充。
          - 填充位置不当 key（注意力里屏蔽）
          - loss 只在目标位置有效处计入，除以有效目标数
          - 梯度额外乘 grad_scale（批量训练时按 token 数加权）
        pad_mask=None 时与旧版逐位相同。
        """
        if len(ids) < 2:
            raise ValueError("CausalLM: 序列至少要有 2 个 token 才能训练（输入 + 目标）")
        ids = _check_token_ids(ids, self.vocab_size, "CausalLM._loss_and_backward")
        if pad_mask is None:
            valid = None
            key_mask = None
            n = len(ids) - 1
        else:
            valid = _check_pad_mask(pad_mask, len(ids), "CausalLM._loss_and_backward")
            key_mask = valid[:-1]
            n = sum(valid[1:])
            if n < 1:
                raise ValueError(
                    "CausalLM._loss_and_backward: pad_mask 里没有有效的目标位置（序列至少 2 个真实 token）")
        logits = self.forward(ids[:-1], key_mask)
        _check_finite_logits(logits, "CausalLM._loss_and_backward")
        d_logits = []
        loss = 0.0
        for t in range(len(ids) - 1):
            if valid is not None and not valid[t + 1]:
                d_logits.append([0.0] * self.vocab_size)
                continue
            probs = _softmax_row(logits[t])
            target = ids[t + 1]
            p = probs[target]
            loss += -math.log(p if p > 1e-12 else 1e-12)
            row = list(probs)
            row[target] -= 1.0
            d_logits.append(row)
        loss /= n
        inv = grad_scale / n
        d_logits = [[v * inv for v in row] for row in d_logits]

        h = self.h
        seq_len = len(h)
        # 输出头反向: grad_W_out[v][j] = sum_t d_logits[t][v] * h[t][j]
        for v in range(self.vocab_size):
            g_row = self.grads["W_out"][v]
            gb = 0.0
            for t in range(seq_len):
                d = d_logits[t][v]
                gb += d
                ht = h[t]
                for j in range(self.d_model):
                    g_row[j] += d * ht[j]
            self.grads["b_out"][v] += gb
        # d_h[t][j] = sum_v d_logits[t][v] * W_out[v][j]
        dh = [[0.0] * self.d_model for _ in range(seq_len)]
        for t in range(seq_len):
            dl = d_logits[t]
            for j in range(self.d_model):
                s = 0.0
                for v in range(self.vocab_size):
                    s += dl[v] * self.W_out[v][j]
                dh[t][j] = s
        d_seq = self.encoder.backward(dh)
        self.embedding.backward(d_seq)   # 位置编码无参数，直通
        return loss

    def adam_step(self, lr: float):
        """对全部参数做一次 Adam 更新（梯度已由 `_loss_and_backward` 累积）"""
        self.step_count += 1
        t = self.step_count
        for name, param, grad in self._named_params():
            m = self._adam_m.get(name)
            v = self._adam_v.get(name)
            if m is None:
                m = _zeros_like(param)
                v = _zeros_like(param)
                self._adam_m[name] = m
                self._adam_v[name] = v
            _adam_apply(param, grad, m, v, t, lr)

    def fit(self, sequences, epochs: int = 1, lr: float = 3e-3,
            shuffle: bool = True, seed=None, verbose: bool = False,
            batch_size: int = 1):
        """在若干条 token 序列上做 teacher-forcing 训练，返回每个 epoch 的平均 loss

        sequences: List[List[int]]，每条至少 2 个 token，长度可以不同。
        batch_size=1（默认）时每个序列单独一步，与旧版逐位相同；
        batch_size>1 时把一批序列右侧补齐成同一长度、用 pad_mask 屏蔽填充，
        梯度按 token 数加权、每批一次 Adam。
        """
        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise ValueError(f"CausalLM.fit: batch_size 必须是正整数，收到 {batch_size!r}")
        if batch_size < 1:
            raise ValueError(f"CausalLM.fit: batch_size 必须 >= 1，收到 {batch_size!r}")
        seqs = [list(s) for s in sequences]
        if not seqs:
            raise ValueError("CausalLM.fit: sequences 不能为空")
        for s in seqs:
            if len(s) < 2:
                raise ValueError("CausalLM.fit: 每条序列至少 2 个 token")
        if epochs < 0:
            raise ValueError(f"CausalLM.fit: epochs 不能为负，收到 {epochs!r}")
        rng = random.Random(seed) if seed is not None else random
        history = []
        self.train()
        for ep in range(int(epochs)):
            order = list(range(len(seqs)))
            if shuffle:
                rng.shuffle(order)
            total = 0.0
            if batch_size == 1:
                for i in order:
                    self.zero_grad()
                    total += self._loss_and_backward(seqs[i])
                    self.adam_step(lr)
            else:
                for start in range(0, len(order), batch_size):
                    batch = order[start:start + batch_size]
                    width = max(len(seqs[i]) for i in batch)
                    n_valid = sum(len(seqs[i]) - 1 for i in batch)
                    self.zero_grad()
                    for i in batch:
                        s = seqs[i]
                        pad = width - len(s)
                        total += self._loss_and_backward(
                            s + [0] * pad,
                            [1] * len(s) + [0] * pad,
                            grad_scale=(len(s) - 1) / n_valid)
                    self.adam_step(lr)
            avg = total / len(seqs)
            history.append(avg)
            if verbose:
                print(f"epoch {ep + 1}/{epochs} loss={avg:.4f}")
        return history

    def evaluate(self, sequences):
        """不更新参数的交叉熵（eval 模式，按预测位置总数平均）"""
        seqs = [list(s) for s in sequences]
        if not seqs:
            raise ValueError("CausalLM.evaluate: sequences 不能为空")
        was_training = self.encoder.layers[0].self_attn.training
        self.eval()
        try:
            total = 0.0
            count = 0
            for s in seqs:
                if len(s) < 2:
                    raise ValueError("CausalLM.evaluate: 每条序列至少 2 个 token")
                s = _check_token_ids(s, self.vocab_size, "CausalLM.evaluate")
                logits = self.forward(s[:-1])
                _check_finite_logits(logits, "CausalLM.evaluate")
                for t in range(len(s) - 1):
                    probs = _softmax_row(logits[t])
                    p = probs[s[t + 1]]
                    total += -math.log(p if p > 1e-12 else 1e-12)
                    count += 1
            return total / count
        finally:
            if was_training:
                self.train()

    def perplexity(self, sequences):
        """困惑度 = exp(交叉熵)。语言模型常用指标，越小越好"""
        loss = self.evaluate(sequences)
        if loss > 700.0:          # exp(709.78) 就溢出，直接给 inf
            return float("inf")
        return math.exp(loss)


    def _config(self):
        """构造参数（存档用）；load 时原样喂回构造器"""
        return {"vocab_size": self.vocab_size, "d_model": self.d_model,
                "num_heads": self.num_heads, "d_ff": self.d_ff,
                "num_layers": self.num_layers, "max_len": self.max_len,
                "dropout": self.dropout, "rope": self.rope,
                "use_abs_pos": self.use_abs_pos, "window": self.window,
                "num_kv_heads": self.num_kv_heads}

    def save(self, filepath: str, include_optimizer: bool = False,
             quiet: bool = False):
        """把结构与参数写成 JSON（纯标准库）

        include_optimizer=True 时连 Adam 的一阶/二阶矩和步数一起存，
        `load(..., restore_optimizer=True)` 之后可以逐位接着训练。
        """
        return _lm_save(self, "causal", filepath, include_optimizer, quiet)

    @staticmethod
    def load(filepath: str, restore_optimizer: bool = False):
        """从 `save` 写出的 JSON 恢复 CausalLM（参数逐位还原）"""
        return _lm_load(filepath, restore_optimizer)

    def _generate_beam(self, ids, max_new_tokens, num_beams, top_k, min_p: float = 0.0,
                       repetition_penalty: float = 1.0, presence_penalty: float = 0.0,
                       frequency_penalty: float = 0.0, no_repeat_ngram_size: int = 0):
        """束搜索：每步保留累计对数概率最高的 num_beams 条前缀（不用随机数）

        num_beams=1 时与贪心解码（`generate(top_k=1)`）逐位相同。
        top_k>0 时每步只展开概率最高的 top_k 个候选，否则展开 num_beams 个。
        min_p 也在这里生效（对每条前缀各自的分布过滤）。
        """
        beams = [(0.0, list(ids))]
        for _ in range(max_new_tokens):
            cands = []
            for score, toks in beams:
                window = toks[-self.max_len:]
                logits = _apply_generation_penalties(
                    self.forward(window)[-1], toks, repetition_penalty,
                    presence_penalty, frequency_penalty, no_repeat_ngram_size)
                logits = _mask_min_p(logits, min_p)
                logp = _log_softmax_row(logits)
                order = sorted(range(self.vocab_size),
                               key=lambda v: logp[v], reverse=True)
                take = order[:int(top_k)] if top_k else order[:num_beams]
                for v in take:
                    cands.append((score + logp[v], toks + [v]))
            if not cands:
                break
            cands.sort(key=lambda c: c[0], reverse=True)
            beams = cands[:num_beams]
        return beams[0][1]

    def generate(self, prompt_ids, max_new_tokens: int = 16, temperature: float = 1.0,
                 top_k: int = 0, rng=None, use_cache: bool = True, top_p: float = 1.0,
                 num_beams: int = 1, min_p: float = 0.0,
                 repetition_penalty: float = 1.0, presence_penalty: float = 0.0,
                 frequency_penalty: float = 0.0, no_repeat_ngram_size: int = 0):
        """自回归生成：返回 prompt + 新 token 的索引列表（调用后保持 eval 模式）

        temperature: >1 更随机，<1 更确定；top_k>0 时只在最可能的 k 个里采样。
        top_p: nucleus 采样阈值（(0,1]，默认 1.0 = 不过滤，与旧行为逐位相同）。

        num_beams: >1 时改用束搜索（确定性，不抽随机数），返回最优的一条完整序列。

        use_cache: True（默认）走 KV cache —— 每个新 token 只前向一次，复用之前
        各层 K/V，生成 N 个 token 的代价从 O(N·T²) 降到 O(N·T)。与 use_cache=False
        的**结果逐位相同**（cache 路径前向的行与整段前向的最后一行一致）。

        抗重复 / 截尾：min_p（(0,1)，相对最大概率截尾）、repetition_penalty（>1 抑制
        重复 token）、presence_penalty / frequency_penalty（按出现与否/次数扣分）、
        no_repeat_ngram_size（禁止 n-gram 立刻重复）。默认值全部等于「不干预」，
        与旧行为逐位相同。
        """
        if max_new_tokens < 0:
            raise ValueError(f"CausalLM.generate: max_new_tokens 不能为负，收到 {max_new_tokens!r}")
        if temperature <= 0:
            raise ValueError(f"CausalLM.generate: temperature 必须 > 0，收到 {temperature!r}")
        if not (0.0 < top_p <= 1.0):
            raise ValueError(f"CausalLM.generate: top_p 必须在 (0,1] 内，收到 {top_p!r}")
        top_k = _check_top_k(top_k, "CausalLM.generate")
        if isinstance(num_beams, bool) or not isinstance(num_beams, int) or num_beams < 1:
            raise ValueError(
                f"CausalLM.generate: num_beams 必须是 >= 1 的整数，收到 {num_beams!r}")
        min_p = _check_min_p(min_p, "CausalLM.generate")
        _check_penalty_args(repetition_penalty, presence_penalty, frequency_penalty,
                            no_repeat_ngram_size, "CausalLM.generate")
        ids = list(prompt_ids)
        if not ids:
            raise ValueError("CausalLM.generate: prompt_ids 不能为空")
        rng = rng or random
        self.eval()
        n_new = int(max_new_tokens)
        if num_beams > 1:
            return self._generate_beam(ids, n_new, num_beams, top_k, min_p,
                                       repetition_penalty, presence_penalty,
                                       frequency_penalty, no_repeat_ngram_size)
        if use_cache and n_new > 0:
            return self._generate_cached(ids, n_new, temperature, top_k, rng, top_p, min_p,
                                         repetition_penalty, presence_penalty,
                                         frequency_penalty, no_repeat_ngram_size)
        for _ in range(n_new):
            window = ids[-self.max_len:]
            logits = _apply_generation_penalties(
                self.forward(window)[-1], ids, repetition_penalty,
                presence_penalty, frequency_penalty, no_repeat_ngram_size)
            ids.append(self._sample(logits, temperature, top_k, rng, top_p, min_p))
        return ids

    def _sample(self, logits, temperature, top_k, rng, top_p: float = 1.0,
                min_p: float = 0.0):
        """从 logits 采一个 token（top_k / top_p / min_p 过滤 + 温度 + 累积分布采样）

        top_p（nucleus 采样）：按概率从高到低累加，只保留累计概率刚好覆盖 top_p
        的头部集合。top_p=1.0（默认）不做过滤，与旧行为逐位相同。
        min_p：在 top_k/top_p 之后再按「概率 >= min_p * 最大概率」截尾（0 = 关闭）。
        """
        if not (0.0 < top_p <= 1.0):
            raise ValueError(f"CausalLM._sample: top_p 必须在 (0,1] 内，收到 {top_p!r}")
        min_p = _check_min_p(min_p, "CausalLM._sample")
        top_k = _check_top_k(top_k, "CausalLM._sample")
        if top_k and top_k < self.vocab_size:
            order = sorted(range(self.vocab_size), key=lambda v: logits[v], reverse=True)
            keep = set(order[:int(top_k)])
            logits = [v if i in keep else float("-inf") for i, v in enumerate(logits)]
        if top_p < 1.0:
            # nucleus：按温度缩放后的概率排序，保留累计概率刚好覆盖 top_p 的头部
            probs_raw = _softmax_row([v / temperature for v in logits])
            order = sorted(range(self.vocab_size), key=lambda v: probs_raw[v], reverse=True)
            keep = set()
            acc_p = 0.0
            for v in order:
                keep.add(v)
                acc_p += probs_raw[v]
                if acc_p >= top_p:
                    break
            logits = [v if i in keep else float("-inf") for i, v in enumerate(logits)]
        if min_p > 0.0:
            peak = max(_softmax_row([v / temperature for v in logits]))
            thresh = float(min_p) * peak
            probs_now = _softmax_row([v / temperature for v in logits])
            logits = [v if probs_now[i] >= thresh else float("-inf")
                      for i, v in enumerate(logits)]
        probs = _softmax_row([v / temperature for v in logits])
        # 累积分布采样（不依赖 random.choices，可注入 rng 复现）
        u = rng.random()
        acc = 0.0
        pick = self.vocab_size - 1
        for v, p in enumerate(probs):
            acc += p
            if u < acc:
                pick = v
                break
        return pick

    def _generate_cached(self, ids, max_new_tokens, temperature, top_k, rng, top_p=1.0,
                         min_p: float = 0.0, repetition_penalty: float = 1.0,
                         presence_penalty: float = 0.0, frequency_penalty: float = 0.0,
                         no_repeat_ngram_size: int = 0):
        """KV cache 版生成：先整段预填充，然后每个新 token 只做单步前向

        窗口满（len(ids) > max_len）时会整体左移，位置编码/因果掩码都依赖绝对
        位置，此时缓存失效 → 用新窗口重建一次（代价 O(max_len)，摊还后仍远小于
        每步重算整段）。
        """
        window = ids[-self.max_len:]
        caches = [{} for _ in self.encoder.layers]
        pos = 0
        logits = None
        for tok in window:
            logits = self._forward_step(tok, pos, caches)
            pos += 1
        for _ in range(max_new_tokens):
            cur = _apply_generation_penalties(logits, ids, repetition_penalty,
                                              presence_penalty, frequency_penalty,
                                              no_repeat_ngram_size)
            pick = self._sample(cur, temperature, top_k, rng, top_p, min_p)
            ids.append(pick)
            new_window = ids[-self.max_len:]
            if len(new_window) == len(window) + 1:
                logits = self._forward_step(pick, pos, caches)
                pos += 1
            else:
                # 窗口滑动：位置整体左移，旧 K/V 全部作废，按新窗口重建
                caches = [{} for _ in self.encoder.layers]
                pos = 0
                for tok in new_window:
                    logits = self._forward_step(tok, pos, caches)
                    pos += 1
            window = new_window
        return ids


class Seq2SeqLM:
    """编码器-解码器序列到序列模型（翻译 / 改写 / 复制类任务）

    数据流::

        src_ids → Embedding → PositionalEncoding → TransformerEncoder ─┐
                                                                        │ memory
        tgt_ids(右移一位) → Embedding → PosEnc → TransformerDecoder ────┘
                                              ↓
                                        词表头 → logits

    解码器每一层先用因果自注意力看自己已生成的前缀，再用交叉注意力读整段
    编码器输出；训练用 teacher forcing（把目标右移一位当输入），推理时逐 token
    生成并把结果接回输入。

    BOS 用「词表外的一个 id」（即 `tgt_vocab` 本身），所以目标端 embedding 有
    `tgt_vocab + 1` 行：这样既能区分句首和真实 token，又不用改目标词表大小。
    """

    def __init__(self, src_vocab: int, tgt_vocab: int, d_model: int = 32,
                 num_heads: int = 4, d_ff: int = 64, num_layers: int = 2,
                 max_len: int = 64, dropout: float = 0.0):
        for name, val in (("src_vocab", src_vocab), ("tgt_vocab", tgt_vocab),
                          ("max_len", max_len), ("num_layers", num_layers)):
            if isinstance(val, bool) or not isinstance(val, int) or val < 1:
                raise ValueError(f"Seq2SeqLM: {name} 必须是正整数，收到 {val!r}")
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.d_model = d_model
        self.max_len = max_len
        self.bos_id = tgt_vocab          # 词表外的 BOS
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.dropout = dropout
        self.step_count = 0
        self._adam_m = {}
        self._adam_v = {}

        self.src_embedding = Embedding(src_vocab, d_model)
        self.tgt_embedding = Embedding(tgt_vocab + 1, d_model)
        self.pos = PositionalEncoding(d_model, max_len)
        self.encoder = TransformerEncoder(d_model, num_heads, d_ff, num_layers, dropout)
        self.decoder = TransformerDecoder(d_model, num_heads, d_ff, num_layers,
                                          dropout, causal=True)

        std = math.sqrt(2.0 / d_model)
        self.W_out = [[random.gauss(0, std) for _ in range(d_model)]
                      for _ in range(tgt_vocab)]
        self.b_out = [0.0] * tgt_vocab

        self.zero_grad()
        self.h = None       # 最近一次前向的解码器输出（反向要用）
        self.enc = None     # 最近一次前向的编码器输出

    def train(self):
        """训练模式：编码器、解码器的 Dropout 全部生效"""
        self.encoder.train()
        self.decoder.train()
        return self

    def eval(self):
        """推理模式：关闭全部 Dropout"""
        self.encoder.eval()
        self.decoder.eval()
        return self

    def zero_grad(self):
        """清零全部梯度（两个词表、编码器、解码器、输出头）"""
        self.src_embedding.zero_grad()
        self.tgt_embedding.zero_grad()
        self.encoder.zero_grad()
        self.decoder.zero_grad()
        self.grads = {
            "W_out": [[0.0] * self.d_model for _ in range(self.tgt_vocab)],
            "b_out": [0.0] * self.tgt_vocab,
        }

    def _named_params(self):
        """产出 (名字, 参数, 梯度)；名字保证 Adam 的矩跨步稳定

        必须在 `zero_grad()` 之后调用（各模块的 grads 是重建的新列表）。
        """
        yield "src_emb.weight", self.src_embedding.weight, self.src_embedding.grad_weight
        yield "tgt_emb.weight", self.tgt_embedding.weight, self.tgt_embedding.grad_weight
        for li, layer in enumerate(self.encoder.layers):
            attn = layer.self_attn
            for k in ("W_q", "W_k", "W_v", "W_o", "b_q", "b_k", "b_v", "b_o"):
                yield f"enc.L{li}.attn.{k}", getattr(attn, k), attn.grads[k]
            ffn = layer.ffn
            for k in ("W1", "b1", "W2", "b2"):
                yield f"enc.L{li}.ffn.{k}", getattr(ffn, k), ffn.grads[k]
            for ln in ("ln1", "ln2"):
                mod = getattr(layer, ln)
                yield f"enc.L{li}.{ln}.gamma", mod.gamma, layer.grads[ln + "_gamma"]
                yield f"enc.L{li}.{ln}.beta", mod.beta, layer.grads[ln + "_beta"]
        for li, layer in enumerate(self.decoder.layers):
            attn = layer.self_attn
            for k in ("W_q", "W_k", "W_v", "W_o", "b_q", "b_k", "b_v", "b_o"):
                yield f"dec.L{li}.attn.{k}", getattr(attn, k), attn.grads[k]
            cross = layer.cross_attn
            for k in ("W_q", "W_k", "W_v", "W_o", "b_q", "b_k", "b_v", "b_o"):
                yield f"dec.L{li}.cross.{k}", getattr(cross, k), cross.grads[k]
            ffn = layer.ffn
            for k in ("W1", "b1", "W2", "b2"):
                yield f"dec.L{li}.ffn.{k}", getattr(ffn, k), ffn.grads[k]
            for ln in ("ln1", "ln2", "ln3"):
                mod = getattr(layer, ln)
                yield f"dec.L{li}.{ln}.gamma", mod.gamma, layer.grads[ln + "_gamma"]
                yield f"dec.L{li}.{ln}.beta", mod.beta, layer.grads[ln + "_beta"]
        yield "out.W", self.W_out, self.grads["W_out"]
        yield "out.b", self.b_out, self.grads["b_out"]

    def param_count(self):
        """参数总数（标量个数）"""
        return sum(len(p) if not isinstance(p[0], list) else
                   sum(len(r) for r in p)
                   for _, p, _ in self._named_params())

    def _encode(self, src_ids, src_mask=None):
        """编码器前向，返回 memory (Ts, d_model)

        src_mask: 长度 Ts 的 0/1 列表，0 表示右填充的位置（批量训练用）
        """
        emb = self.src_embedding.forward(src_ids)
        x = self.pos.forward(emb)
        return self.encoder.forward(x, key_mask=src_mask)

    def _decode(self, tgt_ids, memory, memory_mask=None):
        """解码器前向，返回 logits (Tt, tgt_vocab)（未经 softmax）"""
        emb = self.tgt_embedding.forward(tgt_ids)
        x = self.pos.forward(emb)
        h = self.decoder.forward(x, memory, memory_mask)
        self.h = h
        logits = _matmul(h, _transpose(self.W_out))
        for t in range(len(h)):
            for v in range(self.tgt_vocab):
                logits[t][v] += self.b_out[v]
        return logits

    def forward(self, src_ids, tgt_in_ids, src_mask=None):
        """完整前向：返回 (Tt, tgt_vocab) 的 logits

        src_mask: 长度 len(src_ids) 的 0/1 列表，0 表示右填充位置；
                  同时用于「编码器自注意力的 key」与「解码器的 memory」。
        """
        if not src_ids:
            raise ValueError("Seq2SeqLM.forward: src_ids 不能为空")
        if not tgt_in_ids:
            raise ValueError("Seq2SeqLM.forward: tgt_in_ids 不能为空")
        if len(src_ids) > self.max_len:
            raise ValueError(
                f"Seq2SeqLM.forward: 源序列长度 {len(src_ids)} 超过 max_len={self.max_len}")
        if len(tgt_in_ids) > self.max_len:
            raise ValueError(
                f"Seq2SeqLM.forward: 目标序列长度 {len(tgt_in_ids)} 超过 max_len={self.max_len}")
        self.enc = self._encode(src_ids, src_mask)
        return self._decode(tgt_in_ids, self.enc, src_mask)

    def _loss_and_backward(self, src_ids, tgt_ids, src_mask=None, tgt_mask=None,
                           grad_scale: float = 1.0):
        """单对序列的 teacher-forcing 交叉熵（对目标 token 数取平均）并累积梯度

        src_mask / tgt_mask: 批量训练时的右填充掩码（None 时行为与之前完全一致）
        grad_scale:          梯度整体缩放（批量训练里按 token 数加权用）
        """
        if len(tgt_ids) < 1:
            raise ValueError("Seq2SeqLM: 目标序列不能为空")
        src_ids = _check_token_ids(src_ids, self.src_vocab, "Seq2SeqLM._loss_and_backward")
        tgt_ids = _check_token_ids(tgt_ids, self.tgt_vocab, "Seq2SeqLM._loss_and_backward")
        if tgt_mask is not None:
            tgt_mask = _check_pad_mask(tgt_mask, len(tgt_ids), "Seq2SeqLM._loss_and_backward")
        tgt_in = [self.bos_id] + list(tgt_ids[:-1])
        logits = self.forward(src_ids, tgt_in, src_mask=src_mask)
        _check_finite_logits(logits, "Seq2SeqLM._loss_and_backward")
        n = len(tgt_ids) if tgt_mask is None else sum(1 for v in tgt_mask if v)
        if n <= 0:
            raise ValueError("Seq2SeqLM: 目标序列至少要有一个非填充位置")
        d_logits = []
        loss = 0.0
        for t in range(len(tgt_ids)):
            if tgt_mask is not None and not tgt_mask[t]:
                d_logits.append([0.0] * self.tgt_vocab)
                continue
            probs = _softmax_row(logits[t])
            target = tgt_ids[t]
            p = probs[target]
            loss += -math.log(p if p > 1e-12 else 1e-12)
            row = list(probs)
            row[target] -= 1.0
            d_logits.append(row)
        loss /= n
        inv = grad_scale / n
        d_logits = [[v * inv for v in row] for row in d_logits]

        h = self.h
        seq_len = len(h)
        for v in range(self.tgt_vocab):
            g_row = self.grads["W_out"][v]
            gb = 0.0
            for t in range(seq_len):
                d = d_logits[t][v]
                gb += d
                ht = h[t]
                for j in range(self.d_model):
                    g_row[j] += d * ht[j]
            self.grads["b_out"][v] += gb
        dh = [[0.0] * self.d_model for _ in range(seq_len)]
        for t in range(seq_len):
            dl = d_logits[t]
            for j in range(self.d_model):
                s = 0.0
                for v in range(self.tgt_vocab):
                    s += dl[v] * self.W_out[v][j]
                dh[t][j] = s

        d_dec_in, d_enc = self.decoder.backward(dh)
        self.tgt_embedding.backward(d_dec_in)      # 位置编码无参数，直通
        d_src = self.encoder.backward(d_enc)
        self.src_embedding.backward(d_src)
        return loss

    def adam_step(self, lr: float):
        """对全部参数做一次 Adam 更新（梯度由 `_loss_and_backward` 累积）"""
        self.step_count += 1
        t = self.step_count
        for name, param, grad in self._named_params():
            m = self._adam_m.get(name)
            v = self._adam_v.get(name)
            if m is None:
                m = _zeros_like(param)
                v = _zeros_like(param)
                self._adam_m[name] = m
                self._adam_v[name] = v
            _adam_apply(param, grad, m, v, t, lr)

    def fit(self, pairs, epochs: int = 1, lr: float = 3e-3,
            shuffle: bool = True, seed=None, verbose: bool = False,
            batch_size: int = 1):
        """在若干 (src_ids, tgt_ids) 上训练，返回每个 epoch 的平均 loss

        batch_size > 1 时把长度不同的样本右填充成一批，用 pad 掩码屏蔽填充位，
        梯度按 token 数加权（等价于把整批的目标 token 拉平后取平均），
        每批做一次 Adam 更新。默认 batch_size=1 时行为与之前完全一致。
        """
        items = [(list(s), list(t)) for s, t in pairs]
        if not items:
            raise ValueError("Seq2SeqLM.fit: pairs 不能为空")
        for s, t in items:
            if not s or not t:
                raise ValueError("Seq2SeqLM.fit: 每对的源/目标都不能为空")
        if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 0:
            raise ValueError(f"Seq2SeqLM.fit: epochs 不能为负，收到 {epochs!r}")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError(f"Seq2SeqLM.fit: batch_size 必须是 >=1 的整数，收到 {batch_size!r}")
        rng = random.Random(seed) if seed is not None else random
        history = []
        self.train()
        for ep in range(epochs):
            order = list(range(len(items)))
            if shuffle:
                rng.shuffle(order)
            total = 0.0
            if batch_size == 1:
                for i in order:
                    self.zero_grad()
                    total += self._loss_and_backward(items[i][0], items[i][1])
                    self.adam_step(lr)
            else:
                for start in range(0, len(order), batch_size):
                    group = [items[i] for i in order[start:start + batch_size]]
                    width_src = max(len(s) for s, _ in group)
                    width_tgt = max(len(t) for _, t in group)
                    if width_src > self.max_len or width_tgt > self.max_len:
                        raise ValueError(
                            "Seq2SeqLM.fit: 批量补齐后超过 max_len="
                            f"{self.max_len}（源 {width_src}、目标 {width_tgt}）"
                        )
                    n_total = sum(len(t) for _, t in group)
                    self.zero_grad()
                    for s, t in group:
                        pad_s = list(s) + [0] * (width_src - len(s))
                        pad_t = list(t) + [0] * (width_tgt - len(t))
                        src_mask = [1] * len(s) + [0] * (width_src - len(s))
                        tgt_mask = [1] * len(t) + [0] * (width_tgt - len(t))
                        total += self._loss_and_backward(
                            pad_s, pad_t, src_mask=src_mask, tgt_mask=tgt_mask,
                            grad_scale=len(t) / n_total)
                    self.adam_step(lr)
            avg = total / len(items)
            history.append(avg)
            if verbose:
                print(f"epoch {ep + 1}/{epochs} loss={avg:.4f}")
        return history

    def evaluate(self, pairs):
        """不更新参数的 teacher-forcing 交叉熵（按目标 token 总数平均）"""
        items = [(list(s), list(t)) for s, t in pairs]
        if not items:
            raise ValueError("Seq2SeqLM.evaluate: pairs 不能为空")
        was_training = self.decoder.layers[0].self_attn.training
        self.eval()
        try:
            total = 0.0
            count = 0
            for s, t in items:
                if not s or not t:
                    raise ValueError("Seq2SeqLM.evaluate: 每对的源/目标都不能为空")
                s = _check_token_ids(s, self.src_vocab, "Seq2SeqLM.evaluate")
                t = _check_token_ids(t, self.tgt_vocab, "Seq2SeqLM.evaluate")
                tgt_in = [self.bos_id] + list(t[:-1])
                logits = self.forward(s, tgt_in)
                _check_finite_logits(logits, "Seq2SeqLM.evaluate")
                for i in range(len(t)):
                    probs = _softmax_row(logits[i])
                    p = probs[t[i]]
                    total += -math.log(p if p > 1e-12 else 1e-12)
                    count += 1
            return total / count
        finally:
            if was_training:
                self.train()

    def _sample(self, logits, temperature, top_k, rng, top_p, vocab_size,
                min_p: float = 0.0):
        """从 logits 采一个 token（top_k / top_p / min_p + 温度 + 累积分布）"""
        if not (0.0 < top_p <= 1.0):
            raise ValueError(f"Seq2SeqLM._sample: top_p 必须在 (0,1] 内，收到 {top_p!r}")
        min_p = _check_min_p(min_p, "Seq2SeqLM._sample")
        top_k = _check_top_k(top_k, "Seq2SeqLM._sample")
        if top_k and top_k < vocab_size:
            order = sorted(range(vocab_size), key=lambda v: logits[v], reverse=True)
            keep = set(order[:int(top_k)])
            logits = [v if i in keep else float("-inf") for i, v in enumerate(logits)]
        if top_p < 1.0:
            probs_raw = _softmax_row([v / temperature for v in logits])
            order = sorted(range(vocab_size), key=lambda v: probs_raw[v], reverse=True)
            keep = set()
            acc_p = 0.0
            for v in order:
                keep.add(v)
                acc_p += probs_raw[v]
                if acc_p >= top_p:
                    break
            logits = [v if i in keep else float("-inf") for i, v in enumerate(logits)]
        if min_p > 0.0:
            probs_now = _softmax_row([v / temperature for v in logits])
            thresh = float(min_p) * max(probs_now)
            logits = [v if probs_now[i] >= thresh else float("-inf")
                      for i, v in enumerate(logits)]
        probs = _softmax_row([v / temperature for v in logits])
        u = rng.random()
        acc = 0.0
        pick = vocab_size - 1
        for v, p in enumerate(probs):
            acc += p
            if u < acc:
                pick = v
                break
        return pick

    def _config(self):
        """构造参数（存档用）"""
        return {"src_vocab": self.src_vocab, "tgt_vocab": self.tgt_vocab,
                "d_model": self.d_model, "num_heads": self.num_heads,
                "d_ff": self.d_ff, "num_layers": self.num_layers,
                "max_len": self.max_len, "dropout": self.dropout}

    def save(self, filepath: str, include_optimizer: bool = False,
             quiet: bool = False):
        """把结构与参数写成 JSON（include_optimizer=True 连 Adam 状态一起存）"""
        return _lm_save(self, "seq2seq", filepath, include_optimizer, quiet)

    @staticmethod
    def load(filepath: str, restore_optimizer: bool = False):
        """从 `save` 写出的 JSON 恢复 Seq2SeqLM（参数逐位还原）"""
        return _lm_load(filepath, restore_optimizer)

    def _generate_beam(self, src_ids, max_new_tokens, num_beams, top_k, eos_id=None,
                       min_p: float = 0.0, repetition_penalty: float = 1.0,
                       presence_penalty: float = 0.0, frequency_penalty: float = 0.0,
                       no_repeat_ngram_size: int = 0):
        """束搜索解码：编码一次，每步对每条前缀重算解码器并保留前 num_beams

        eos_id 非 None 时把以 eos 结尾的候选放进「已完成」，最后按累计对数概率
        在所有已完成/未完成候选里挑最好的。
        """
        memory = self._encode(list(src_ids))
        beams = [(0.0, [self.bos_id], [])]
        finished = []
        for _ in range(max_new_tokens):
            cands = []
            for score, tgt, out in beams:
                if len(tgt) >= self.max_len:
                    finished.append((score, out))
                    continue
                logits = _apply_generation_penalties(
                    self._decode(tgt, memory)[-1], tgt, repetition_penalty,
                    presence_penalty, frequency_penalty, no_repeat_ngram_size)
                logits = _mask_min_p(logits, min_p)
                logp = _log_softmax_row(logits)
                order = sorted(range(self.tgt_vocab),
                               key=lambda v: logp[v], reverse=True)
                take = order[:int(top_k)] if top_k else order[:num_beams]
                for v in take:
                    if eos_id is not None and v == eos_id:
                        finished.append((score + logp[v], out))
                    else:
                        cands.append((score + logp[v], tgt + [v], out + [v]))
            if not cands:
                break
            cands.sort(key=lambda c: c[0], reverse=True)
            beams = cands[:num_beams]
        best = None
        if finished:
            finished.sort(key=lambda c: c[0], reverse=True)
            best = finished[0]
        if best is not None and (not beams or best[0] >= beams[0][0]):
            return list(best[1])
        return list(beams[0][2])

    def generate(self, src_ids, max_new_tokens: int = 5, temperature: float = 1.0,
                 top_k: int = 0, top_p: float = 1.0, rng=None, eos_id=None,
                 num_beams: int = 1, min_p: float = 0.0,
                 repetition_penalty: float = 1.0, presence_penalty: float = 0.0,
                 frequency_penalty: float = 0.0, no_repeat_ngram_size: int = 0):
        """编码一次、解码逐步生成；返回生成的 token 列表（不含 BOS）

        eos_id 非 None 时遇到它立即停止（也不把它放进结果）。
        每步都对整个已生成前缀重算一次解码器（T 通常很短，实现保持简单）。

        min_p / repetition_penalty / presence_penalty / frequency_penalty /
        no_repeat_ngram_size 与 CausalLM 同义，默认值等于「不干预」。
        """
        if not src_ids:
            raise ValueError("Seq2SeqLM.generate: src_ids 不能为空")
        if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) \
                or max_new_tokens < 0:
            raise ValueError(
                f"Seq2SeqLM.generate: max_new_tokens 不能为负，收到 {max_new_tokens!r}")
        if temperature <= 0:
            raise ValueError(
                f"Seq2SeqLM.generate: temperature 必须 > 0，收到 {temperature!r}")
        if isinstance(num_beams, bool) or not isinstance(num_beams, int) or num_beams < 1:
            raise ValueError(
                f"Seq2SeqLM.generate: num_beams 必须是 >= 1 的整数，收到 {num_beams!r}")
        top_k = _check_top_k(top_k, "Seq2SeqLM.generate")
        min_p = _check_min_p(min_p, "Seq2SeqLM.generate")
        _check_penalty_args(repetition_penalty, presence_penalty, frequency_penalty,
                            no_repeat_ngram_size, "Seq2SeqLM.generate")
        src_ids = _check_token_ids(src_ids, self.src_vocab, "Seq2SeqLM.generate")
        if len(src_ids) > self.max_len:
            raise ValueError(
                f"Seq2SeqLM.generate: 源序列长度 {len(src_ids)} 超过 max_len={self.max_len}")
        rng = rng or random
        self.eval()
        if num_beams > 1:
            return self._generate_beam(src_ids, int(max_new_tokens), num_beams,
                                       top_k, eos_id, min_p, repetition_penalty,
                                       presence_penalty, frequency_penalty,
                                       no_repeat_ngram_size)
        memory = self._encode(list(src_ids))
        out = []
        tgt = [self.bos_id]
        for _ in range(max_new_tokens):
            if len(tgt) >= self.max_len:
                break
            logits = _apply_generation_penalties(
                self._decode(tgt, memory)[-1], tgt, repetition_penalty,
                presence_penalty, frequency_penalty, no_repeat_ngram_size)
            pick = self._sample(logits, temperature, top_k, rng, top_p,
                                self.tgt_vocab, min_p)
            tgt.append(pick)
            if eos_id is not None and pick == eos_id:
                break
            out.append(pick)
        return out


class ResidualBlock:
    """残差连接：output = x + F(x)"""

    def __init__(self, sublayer):
        self.sublayer = sublayer
        self.x = None

    def forward(self, x):
        self.x = x
        out = self.sublayer.forward(x)
        # 形状不符时下面的逐元素相加会静默丢掉多余分量（或抛 IndexError），必须早失败
        if isinstance(x, list) and x and isinstance(x[0], list):
            if (len(out) != len(x)
                    or any(len(out[t]) != len(x[t]) for t in range(len(x)))):
                raise ValueError(
                    f"ResidualBlock.forward: 子层输出形状与输入不一致"
                    f"（输入 {len(x)}×{len(x[0])}，输出 {len(out)}"
                    f"×{len(out[0]) if out else 0}），残差相加要求形状相同"
                )
        elif len(out) != len(x):
            raise ValueError(
                f"ResidualBlock.forward: 子层输出长度 {len(out)} 与输入长度 {len(x)} 不一致，"
                f"残差相加要求形状相同（是否是子层改了维度？）"
            )
        if isinstance(x, list) and isinstance(x[0], list):
            return [[x[t][j] + out[t][j] for j in range(len(x[0]))] for t in range(len(x))]
        return [x[i] + out[i] for i in range(len(x))]

    def backward(self, dout):
        # sublayer 的 backward 返回值有两种约定（同 Sequential）：
        #   元组 (..., delta) 取其末位；list 则整体就是 delta。
        # Layer / LayerNorm 返回元组，Dropout / Embedding / 注意力等返回 list。
        result = self.sublayer.backward(dout)
        # 各模块的 delta 位置不一定相同（BatchNorm 放在第 0 位），用 delta_index 约定
        d_sublayer = (result[getattr(self.sublayer, "delta_index", -1)]
                      if isinstance(result, tuple) else result)
        if isinstance(dout, list) and isinstance(dout[0], list):
            return [[dout[t][j] + d_sublayer[t][j] for j in range(len(dout[0]))] for t in range(len(dout))]
        return [dout[i] + d_sublayer[i] for i in range(len(dout))]


class Sequential:
    """通用顺序容器，混合 Layer, Dropout, LayerNorm, Embedding 等。

    返回协议约定：
      每个 module.backward(delta) 的返回值必须是以下两种之一：
        - 非元组 (list): 直接作为上一层 delta（如 Dropout、Embedding）
        - 元组 (..., delta): 末尾元素为上一层 delta（如 Layer、LayerNorm）
      即：若返回元组则取其最后一维为 dx；否则整体作为 dx。
    """

    def __init__(self, *modules):
        # 两种写法都接受：Sequential(a, b) 与 Sequential([a, b])。
        # 后者原来会把整个列表当成一个模块，直到 forward 才炸成
        # AttributeError: 'list' object has no attribute 'forward'，现在直接摊平。
        if len(modules) == 1 and isinstance(modules[0], (list, tuple)):
            modules = tuple(modules[0])
        for _i, _m in enumerate(modules):
            if not hasattr(_m, "forward"):
                raise ValueError(
                    f"Sequential: 第 {_i} 个模块 {type(_m).__name__} 没有 forward()，"
                    f"不能放进容器"
                )
        self.modules = list(modules)

    def train(self):
        """切到训练模式（转发给实现了 train() 的子模块）"""
        for m in self.modules:
            if hasattr(m, "train"):
                m.train()
        return self

    def eval(self):
        """切到推理模式（关闭内部的 Dropout，否则推理结果随机）"""
        for m in self.modules:
            if hasattr(m, "eval"):
                m.eval()
        return self

    def forward(self, x):
        for m in self.modules:
            x = m.forward(x)
        return x

    def backward(self, delta):
        for m in reversed(self.modules):
            result = m.backward(delta)
            if result is not None:
                # delta_index 默认 -1（末位）：Layer 的 (gw, gb, dx)、LayerNorm 的
                # (dγ, dβ, dx) 都以 dx 结尾；BatchNorm2d/CrossAttention 这类
                # 「多返回值但主梯度在前」的模块自己声明 delta_index = 0
                delta = (result[getattr(m, "delta_index", -1)]
                         if isinstance(result, tuple) else result)
        return delta



class LoRA:
    """低秩适配（LoRA）：冻结原层，只训练两个小矩阵

    ``y = Layer(x) + scaling * B (A x)``，``A`` 形状 (rank, fan_in)、
    ``B`` 形状 (fan_out, rank)，``scaling = alpha / rank``（alpha 默认 = rank）。

    训练时原层参数被冻结（梯度算出来就丢弃），只更新 A/B；``merge()`` 把
    ``scaling * B A`` 折进原层权重，之后推理零额外开销。1000×1000 的层用
    rank=8 只需 16000 个新参数（0.16%）。
    """

    def __init__(self, layer: Layer, rank: int = 4, alpha: float = None):
        if not isinstance(layer, Layer):
            raise TypeError(f"LoRA: layer 必须是 Layer 实例，收到 {type(layer).__name__}")
        if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
            raise ValueError(f"LoRA: rank 必须是正整数，收到 {rank!r}")
        self.layer = layer
        self.rank = rank
        self.alpha = float(rank if alpha is None else alpha)
        if self.alpha <= 0.0:
            raise ValueError(f"LoRA: alpha 必须 > 0，收到 {alpha!r}")
        self.scaling = self.alpha / rank
        self.fan_in = layer.fan_in
        self.fan_out = layer.fan_out
        std = 1.0 / math.sqrt(self.fan_in)
        self.A = [[random.gauss(0, std) for _ in range(self.fan_in)]
                  for _ in range(self.rank)]
        # B 初始化为 0：训练开始时适配分支输出为 0，原模型行为不变
        self.B = [[0.0] * self.rank for _ in range(self.fan_out)]
        self.merged = False
        self.zero_grad()
        self.h = self.x = None

    def zero_grad(self):
        """清零 A/B 的梯度"""
        self.grad_A = [[0.0] * self.fan_in for _ in range(self.rank)]
        self.grad_B = [[0.0] * self.rank for _ in range(self.fan_out)]

    def params(self):
        """返回 [(参数, 梯度)]，可直接喂给上面的优化器"""
        return [(self.A, self.grad_A), (self.B, self.grad_B)]

    def forward(self, x):
        base = self.layer.forward(x)
        if self.merged:
            return base
        self.x = x
        self.h = [sum(map(_mul, row, x)) for row in self.A]        # (rank,)
        delta = [sum(map(_mul, row, self.h)) for row in self.B]    # (fan_out,)
        return [base[j] + self.scaling * delta[j] for j in range(self.fan_out)]

    def backward(self, delta_out):
        """返回 dx（穿过原层与低秩分支）；原层梯度被丢弃（冻结）"""
        if self.merged:
            return self.layer.backward(delta_out)[-1]
        if self.h is None:
            raise RuntimeError("LoRA.backward: 必须先调用 forward()")
        dx_base = self.layer.backward(delta_out)[-1]
        # grad_B += scaling * outer(delta_out, h)
        for j in range(self.fan_out):
            g = self.scaling * delta_out[j]
            row = self.grad_B[j]
            for r in range(self.rank):
                row[r] += g * self.h[r]
        # dh = scaling * B^T delta_out
        dh = [0.0] * self.rank
        for r in range(self.rank):
            acc = 0.0
            for j in range(self.fan_out):
                acc += self.B[j][r] * delta_out[j]
            dh[r] = self.scaling * acc
        # grad_A += outer(dh, x)
        for r in range(self.rank):
            g = dh[r]
            row = self.grad_A[r]
            for i in range(self.fan_in):
                row[i] += g * self.x[i]
        # dx = dx_base + A^T dh
        dx = list(dx_base)
        for i in range(self.fan_in):
            acc = 0.0
            for r in range(self.rank):
                acc += self.A[r][i] * dh[r]
            dx[i] += acc
        return dx

    def merge(self):
        """把 scaling*B@A 折进原层权重，之后 forward 不再走低秩分支"""
        if self.merged:
            return self
        W = self.layer.weights
        for j in range(self.fan_out):
            for i in range(self.fan_in):
                acc = 0.0
                for r in range(self.rank):
                    acc += self.B[j][r] * self.A[r][i]
                W[j][i] += self.scaling * acc
        for j in range(self.fan_out):
            for r in range(self.rank):
                self.B[j][r] = 0.0
        self.merged = True
        return self

    def trainable_param_count(self):
        """可训练参数个数：只算 A/B（原层被冻结）"""
        return self.rank * self.fan_in + self.fan_out * self.rank

    def total_param_count(self):
        """包装后的参数总数：原层参数 + 可训练的 A/B"""
        return self.fan_out * (self.fan_in + 1) + self.trainable_param_count()



_ATTN_RAMP = " .:-=+*#%@"


def _copy_attn_maps(heads, who):
    """把 MHA 缓存的每头权重整理成纯 float 的嵌套列表，并做形状自检。"""
    if heads is None:
        raise RuntimeError(f"{who}: 没有注意力缓存（先 forward 一次）")
    if not isinstance(heads, list) or not heads:
        raise RuntimeError(f"{who}: 注意力缓存为空")
    out = []
    seq = None
    for h, mat in enumerate(heads):
        if not isinstance(mat, list) or not mat:
            raise RuntimeError(f"{who}: 第 {h} 头的注意力矩阵为空")
        cols = len(mat[0])
        if seq is None:
            seq = len(mat)
        if len(mat) != seq or any(len(row) != cols for row in mat):
            raise RuntimeError(f"{who}: 第 {h} 头的注意力矩阵不是方阵")
        out.append([[float(v) for v in row] for row in mat])
    return out, seq


def attention_maps(module, x, layer_index=None):
    """跑一次前向，取回每头注意力权重（只读，不改参数）。

    支持：
      - `MultiHeadAttention`：返回 `{"num_heads", "seq_len", "maps"}`，
        其中 `maps[h][t][j]` 是第 h 头里位置 t 对位置 j 的权重（行和 1）。
      - `TransformerEncoderLayer`：取它的 `self_attn`。
      - `TransformerEncoder`：返回 `{"layers": [{"layer": i, ...}, ...]}`；
        给 `layer_index` 时只取那一层。
      - `CausalLM`：`x` 是 token id 列表，返回编码器各层的图。

    为了不受 dropout 随机遮罩影响，采样期间会把模型切到 eval，结束后恢复原模式。
    """
    if layer_index is not None and (isinstance(layer_index, bool)
                                    or not isinstance(layer_index, int)):
        raise TypeError("attention_maps: layer_index 必须是 int 或 None")

    def _grab(mha, who):
        maps, seq = _copy_attn_maps(getattr(mha, "attn_weights", None), who)
        return {"num_heads": len(maps), "seq_len": seq, "maps": maps}

    if isinstance(module, MultiHeadAttention):
        was = module.training
        module.eval()
        try:
            module.forward(x)
        finally:
            module.train() if was else module.eval()
        return _grab(module, "attention_maps(MultiHeadAttention)")

    if isinstance(module, TransformerEncoderLayer):
        was = module.self_attn.training
        module.eval()
        try:
            module.forward(x)
        finally:
            module.train() if was else module.eval()
        return _grab(module.self_attn, "attention_maps(TransformerEncoderLayer)")

    if isinstance(module, TransformerEncoder):
        was = module.layers[0].self_attn.training
        module.eval()
        try:
            module.forward(x)
        finally:
            module.train() if was else module.eval()
        layers = []
        for i, layer in enumerate(module.layers):
            if layer_index is not None and i != layer_index:
                continue
            item = _grab(layer.self_attn, f"attention_maps(encoder 第 {i} 层)")
            item["layer"] = i
            layers.append(item)
        if not layers:
            raise ValueError(
                f"attention_maps: layer_index={layer_index} 超出范围 [0, {len(module.layers)})")
        return {"layers": layers}

    if isinstance(module, CausalLM):
        was = _lm_training_state(module)
        module.eval()
        try:
            module.forward(x)
        finally:
            _lm_restore_mode(module, was)
        layers = []
        for i, layer in enumerate(module.encoder.layers):
            if layer_index is not None and i != layer_index:
                continue
            item = _grab(layer.self_attn, f"attention_maps(CausalLM 第 {i} 层)")
            item["layer"] = i
            layers.append(item)
        if not layers:
            raise ValueError(
                f"attention_maps: layer_index={layer_index} 超出范围 [0, {len(module.encoder.layers)})")
        return {"layers": layers}

    raise TypeError(
        "attention_maps: 支持 MultiHeadAttention / TransformerEncoderLayer / "
        f"TransformerEncoder / CausalLM，收到 {type(module).__name__}")


def attention_ascii(maps, head=0, max_width=48):
    """把一张注意力矩阵画成字符热图（每行归一化到自身最大值）。

    `maps` 可以是 `attention_maps` 的返回值，也可以直接是 `heads` 列表；
    返回多行字符串，含 0-9 的强度刻度说明。
    """
    if isinstance(maps, dict):
        if "maps" in maps:
            maps = maps["maps"]
        elif "layers" in maps:
            if not maps["layers"]:
                raise ValueError("attention_ascii: layers 为空")
            maps = maps["layers"][0]["maps"]
        else:
            raise ValueError("attention_ascii: 不认识的结构")
    if not isinstance(maps, (list, tuple)) or not maps:
        raise ValueError("attention_ascii: maps 必须是非空列表")
    if isinstance(head, bool) or not isinstance(head, int):
        raise TypeError("attention_ascii: head 必须是 int")
    if head < 0 or head >= len(maps):
        raise ValueError(f"attention_ascii: head={head} 越界 [0, {len(maps)})")
    if isinstance(max_width, bool) or not isinstance(max_width, int):
        raise TypeError("attention_ascii: max_width 必须是 int")
    if max_width < 1:
        raise ValueError("attention_ascii: max_width 必须 >= 1")

    mat = maps[head]
    n = len(mat)
    step = max(1, (n + max_width - 1) // max_width)
    rows = [f"head {head}  ({n}x{n}，每 {step} 个位置取一格)"]
    for t in range(0, n, step):
        row = mat[t]
        peak = max(row) if row else 0.0
        chars = []
        for j in range(0, n, step):
            v = row[j] / peak if peak > 0 else 0.0
            chars.append(_ATTN_RAMP[min(len(_ATTN_RAMP) - 1, int(v * (len(_ATTN_RAMP) - 1) + 0.5))])
        rows.append(f"{t:>4} |" + "".join(chars) + "|")
    rows.append("     " + "-" * len(range(0, n, step)))
    rows.append("     强度：'" + _ATTN_RAMP.strip() + "' = 0 -> 1（每行按自身最大值缩放）")
    return "\n".join(rows)


def profile_model(model, x, backward: bool = False, repeat: int = 3):
    """逐层统计前向（可选反向）耗时，返回 dict

    参数：
        model: `MLP` / `Model` / `Sequential`，或任何带 `.layers` 与 `.forward` 的对象
        x: 一个样本（调用方保证维度正确）
        backward: 为 True 时一并统计反向（用全 1 的 delta 起反传）
        repeat: 每种测量重复次数，取**最小值**（受调度抖动影响最小）

    返回：`{"layers": [{"index","type","forward_ms","backward_ms"}],
            "total_ms", "backward_total_ms", "params"}`

    说明：这是纯墙钟测量，不含 CUDA/内存统计；单层耗时很小（微秒级）时
    数字抖动明显，用 `repeat` 取最小值和相对占比来读更可靠。
    """
    if isinstance(repeat, bool) or not isinstance(repeat, int) or repeat < 1:
        raise ValueError(f"profile_model: repeat 必须是正整数，收到 {repeat!r}")
    net = model.model if isinstance(model, Model) else model
    if not hasattr(net, "layers") or not hasattr(net, "forward"):
        raise TypeError(
            f"profile_model: 需要带 layers/forward 的网络，收到 {type(model).__name__}")
    if not net.layers:
        raise ValueError("profile_model: 网络没有任何层")

    params = sum(getattr(l, "fan_in", 0) * getattr(l, "fan_out", 0)
                 + getattr(l, "fan_out", 0) for l in net.layers)

    # 前向：逐层测量，同时把输出串给下一层
    fwd = []
    cur = list(x)
    for layer in net.layers:
        best = None
        for _ in range(repeat):
            t0 = time.perf_counter()
            out = layer.forward(list(cur))
            dt = time.perf_counter() - t0
            best = dt if best is None else min(best, dt)
        fwd.append(best)
        cur = out

    # 反向：先整网前向一次建立每层的缓存，再从最后一层往前测
    bwd = [0.0] * len(net.layers)
    if backward:
        net.forward(list(x))
        delta = [1.0] * net.layers[-1].fan_out
        for i in range(len(net.layers) - 1, -1, -1):
            layer = net.layers[i]
            base_delta = delta          # 每次重复都用同一个输入，才能比较耗时
            best = None
            nxt = None
            for _ in range(repeat):
                t0 = time.perf_counter()
                _gw, _gb, nxt = layer.backward(base_delta)
                dt = time.perf_counter() - t0
                best = dt if best is None else min(best, dt)
            bwd[i] = best
            delta = nxt

    layers = [{
        "index": i,
        "type": type(layer).__name__,
        "forward_ms": fwd[i] * 1000.0,
        "backward_ms": bwd[i] * 1000.0,
    } for i, layer in enumerate(net.layers)]
    return {
        "layers": layers,
        "total_ms": sum(fwd) * 1000.0,
        "backward_total_ms": (sum(bwd) * 1000.0) if backward else None,
        "params": params,
    }


def lr_finder(model, data=None, x=None, y=None, start_lr: float = 1e-5,
              end_lr: float = 1.0, steps: int = 60, smooth: float = 0.05,
              seed=None):
    """学习率扫描（LR range test 简化版）

    从 `start_lr` 指数增长到 `end_lr`，每步取一个样本做一次真实更新并记录 loss。
    返回 `{"history": [(lr, loss, smoothed)], "best_lr", "suggested_lr", "final_loss"}`。

    `suggested_lr` 取「平滑 loss 最低点 / 10」并夹在 `[start_lr, end_lr]` 内：
    最低点通常已在发散边缘，低一个量级更安全。

    注意：扫描会**真实修改模型权重**（与 PyTorch 的 lr_finder 一样），
    需要干净权重请先复制；结束后 `optimizer.lr` 会恢复原值。
    """
    if not isinstance(model, Model):
        raise TypeError(f"lr_finder: 需要 Model，收到 {type(model).__name__}")
    for name, val in (("start_lr", start_lr), ("end_lr", end_lr)):
        if isinstance(val, bool) or not isinstance(val, (int, float)) or not val > 0:
            raise ValueError(f"lr_finder: {name} 必须是正数，收到 {val!r}")
    if not end_lr > start_lr:
        raise ValueError(
            f"lr_finder: 要求 end_lr > start_lr，收到 {start_lr!r} -> {end_lr!r}")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 2:
        raise ValueError(f"lr_finder: steps 必须是 >=2 的整数，收到 {steps!r}")
    if isinstance(smooth, bool) or not isinstance(smooth, (int, float)) \
            or not 0.0 < float(smooth) <= 1.0:
        raise ValueError(f"lr_finder: smooth 必须在 (0,1] 内，收到 {smooth!r}")

    if data is not None:
        dataset = [(list(a), b) for a, b in data]
    elif x is not None and y is not None:
        if isinstance(y, (int, float)) and not isinstance(y, bool):
            y = [y] * len(x)
        if len(x) != len(y):
            raise ValueError(
                f"lr_finder: x 有 {len(x)} 个样本，y 有 {len(y)} 个，数量必须一致")
        dataset = [(list(a), b) for a, b in zip(x, y)]
    else:
        raise ValueError("lr_finder: 需要 data 或 (x, y)")
    if not dataset:
        raise ValueError("lr_finder: 数据集不能为空")

    rng = random.Random(seed) if seed is not None else random
    order = list(range(len(dataset)))
    rng.shuffle(order)

    old_lr = model.optimizer.lr
    lr = float(start_lr)
    mult = (float(end_lr) / float(start_lr)) ** (1.0 / (steps - 1))
    history = []
    ema = None
    diverged_at = None
    # 单个样本的 loss 噪声太大，指数平滑也救不回来；这里每步用小批的平均值
    batch = min(8, len(dataset))
    layers = model.model.layers
    try:
        for step in range(steps):
            idxs = [order[(step * batch + t) % len(order)] for t in range(batch)]
            model.optimizer.lr = lr
            acc_w = [[[0.0] * layer.fan_in for _ in range(layer.fan_out)]
                     for layer in layers]
            acc_b = [[0.0] * layer.fan_out for layer in layers]
            loss_sum = 0.0
            for i in idxs:
                xi, yi = dataset[i]
                pred = model.forward(xi)
                loss_i, grad_i = model._loss(pred, yi)
                loss_sum += loss_i
                delta = grad_i
                li = len(layers) - 1
                for layer in reversed(layers):
                    gw, gb, delta = layer.backward(delta)
                    aw, ab = acc_w[li], acc_b[li]
                    for j in range(layer.fan_out):
                        awj, gwj = aw[j], gw[j]
                        for k in range(layer.fan_in):
                            awj[k] += gwj[k]
                        ab[j] += gb[j]
                    li -= 1
            loss = loss_sum / batch
            n = float(batch)
            grads = [([[v / n for v in row] for row in acc_w[i]],
                      [v / n for v in acc_b[i]]) for i in range(len(layers))]
            model.optimizer.step(grads)
            # 一旦 loss 变成 inf/NaN 就停：继续扫描只会污染权重，也没有信息量
            if not math.isfinite(loss):
                diverged_at = lr
                break
            ema = loss if ema is None else (1.0 - smooth) * ema + smooth * loss
            if not math.isfinite(ema):
                diverged_at = lr
                break
            history.append((lr, loss, ema))
            lr *= mult
    finally:
        model.optimizer.lr = old_lr

    # 经典 fastai 规则：取平滑 loss 的最低点，推荐它的 1/10
    # （最低点往往已在发散边缘，降一个量级更稳；这与我们的 lr_finder 探针实测一致）
    note = None
    if len(history) > 1:
        inner = history[1:] if len(history) > 4 else history   # 跳过第一步（EMA 尚未稳定）
        low_item = min(inner, key=lambda r: r[2])
    else:
        low_item = history[0]
    best_lr = low_item[0]
    if low_item is history[-1] or best_lr == history[-1][0]:
        note = "loss 到扫描末尾仍在下降，建议把 end_lr 调大后再扫一次"
    lowest = best_lr
    suggested = max(float(start_lr), min(float(end_lr), float(best_lr) / 10.0))
    return {
        "history": history,
        "best_lr": best_lr,
        "lowest_loss_lr": lowest,
        "suggested_lr": suggested,
        "final_loss": history[-1][1],
        "diverged_at": diverged_at,
        "note": note,
    }


# ============================================================================
#  Loss Functions
# ============================================================================

def _numeric_pair(pred, target, who):
    """把 (pred, target) 规范成等长 float 列表；target 允许标量（只能配单输出）"""
    if isinstance(target, (list, tuple)):
        if len(target) != len(pred):
            raise ValueError(f"{who}: target 长度 {len(target)} != pred 长度 {len(pred)}")
        return [float(t) for t in target], [float(p) for p in pred]
    ps = [float(p) for p in pred]
    if len(ps) != 1:
        raise ValueError(f"{who}: 标量 target 只能配单个输出，pred 长度是 {len(ps)}")
    return [float(target)], ps


def l1_loss(pred, target):
    """L1（MAE）损失：逐元素 |p-y| 的平均值；梯度是 sign(p-y)/n

    对离群点不敏感，但在 0 附近不可导（本实现取 sign，两侧极限一致）。
    返回 (loss, grad)，grad 相对 pred。
    """
    ts, ps = _numeric_pair(pred, target, "l1_loss")
    n = len(ts)
    loss = 0.0
    grad = []
    for y, p in zip(ts, ps):
        e = p - y
        loss += abs(e)
        grad.append((1.0 if e > 0 else (-1.0 if e < 0 else 0.0)) / n)
    return loss / n, grad


def log_cosh_loss(pred, target):
    """Log-Cosh 损失：log(cosh(p-y)) 的平均值；梯度 tanh(p-y)/n

    兼具 MSE 的平滑与 L1 的抗离群（对大误差近似 |e|-log2），数值实现用
    `|e| + log1p(exp(-2|e|)) - log2` 避免 cosh 溢出。返回 (loss, grad)。
    """
    ts, ps = _numeric_pair(pred, target, "log_cosh_loss")
    n = len(ts)
    loss = 0.0
    grad = []
    log2 = math.log(2.0)
    for y, p in zip(ts, ps):
        e = p - y
        ae = abs(e)
        loss += ae + math.log1p(math.exp(-2.0 * ae)) - log2
        grad.append(math.tanh(e) / n)
    return loss / n, grad


def quantile_loss(pred, target, quantile: float = 0.5):
    """分位数（pinball）损失：q=0.5 就是 MAE 的一半，q>0.5 偏保守（高估代价小）

    逐元素 `max(q*e, (q-1)*e)` 的平均，其中 `e = p - y`；梯度
    `-(1-q)/n`（e<0）或 `q/n`（e>=0）。返回 (loss, grad)。
    """
    q = float(quantile)
    if not 0.0 < q < 1.0:
        raise ValueError(f"quantile_loss: quantile 必须在 (0,1) 内，收到 {quantile!r}")
    ts, ps = _numeric_pair(pred, target, "quantile_loss")
    n = len(ts)
    loss = 0.0
    grad = []
    for y, p in zip(ts, ps):
        e = p - y
        loss += q * e if e >= 0 else (q - 1.0) * e
        grad.append((q if e >= 0 else (q - 1.0)) / n)
    return loss / n, grad


def smooth_l1_loss(pred, target, delta: float = 1.0):
    """Smooth L1 = Huber 的别名（PyTorch 叫法），实现直接复用 `huber_loss`"""
    return huber_loss(pred, target, delta=delta)


def _class_counts_vector(class_counts, num_classes, who):
    """把 class_counts 规范成与类别数等长的正数列表（每类样本数）"""
    try:
        counts = [float(c) for c in class_counts]
    except TypeError:
        raise TypeError(f"{who}: class_counts 必须是每类样本数的序列，收到 {class_counts!r}")
    if len(counts) != num_classes:
        raise ValueError(
            f"{who}: class_counts 长度 {len(counts)} != 类别数 {num_classes}")
    if any(c <= 0.0 for c in counts):
        raise ValueError(f"{who}: class_counts 每项都必须 > 0（样本数），收到 {list(counts)}")
    return counts


def _class_target_weights(target_idx, num_classes, weights, who):
    """按目标算样本权重：索引/one-hot 取对应类，真软标签取 Σt·w 的加权平均

    返回 (weight, target_is_index, index_or_vector)
    """
    if isinstance(target_idx, (list, tuple)):
        if len(target_idx) != num_classes:
            raise ValueError(
                f"{who}: target 长度 {len(target_idx)} != pred 长度 {num_classes}")
        ts = [float(t) for t in target_idx]
        if any(t < 0.0 for t in ts):
            raise ValueError(f"{who}: 目标向量不能含负值")
        ssum = sum(ts)
        if ssum <= 0.0:
            raise ValueError(f"{who}: 目标向量不能全 0")
        w = sum(t * c for t, c in zip(ts, weights)) / ssum
        return w, False, ts
    idx = _ce_scalar_index(target_idx, num_classes, who)
    return weights[idx], True, idx


def class_balanced_loss(pred, target_idx, class_counts, beta: float = 0.999,
                        normalize: bool = True):
    """类别均衡损失（Cui et al. 2019）：每类权重 ∝ (1-β)/(1-β^{n_c})

    `n_c` 是类别 c 的样本数，样本越多权重越小（β→1 时退化成 1/n_c）。
    `normalize=True` 再把权重归一化到「按出现频率加权后的均值为 1」，避免整体学习率被放大。
    pred 是 softmax 概率；返回 (loss, grad)，grad 是 softmax+CE 的联合梯度（配 softmax 输出层）。
    支持类别索引与 one-hot/软标签（软标签按 Σt·w/Σt 取样本权重）。
    """
    if not isinstance(normalize, bool):
        raise TypeError(f"class_balanced_loss: normalize 必须是 bool，收到 {normalize!r}")
    beta = float(beta)
    if not 0.0 <= beta < 1.0:
        raise ValueError(f"class_balanced_loss: beta 必须在 [0,1) 内，收到 {beta!r}")
    V = len(pred)
    counts = _class_counts_vector(class_counts, V, "class_balanced_loss")
    if beta == 0.0:
        raw = [1.0] * V
    else:
        raw = [(1.0 - beta) / (1.0 - beta ** n) for n in counts]
    weights = raw
    if normalize:
        total = sum(c * w for c, w in zip(counts, raw))
        mean_w = total / sum(counts)
        if mean_w <= 0.0:
            raise ValueError("class_balanced_loss: 权重归一化失败（均值为 0）")
        weights = [w / mean_w for w in raw]
    w, is_index, tgt = _class_target_weights(target_idx, V, weights, "class_balanced_loss")
    eps = 1e-15
    if is_index:
        loss = -w * math.log(max(float(pred[tgt]), eps))
        grad = [w * float(p) for p in pred]
        grad[tgt] -= w
        return loss, grad
    ssum = sum(tgt)
    loss = -w * sum(t * math.log(max(float(p), eps)) for t, p in zip(tgt, pred))
    grad = [w * (ssum * float(p) - t) for p, t in zip(pred, tgt)]
    return loss, grad


def ldam_loss(logits, target_idx, class_counts, max_m: float = 0.5, s: float = 30.0):
    """LDAM（Cao et al. 2019）：给目标类 logit 减一个与类频率相关的间隔

        m_c = max_m / n_c^{1/4},  z' = s·(z - m_y·e_y),  loss = -log softmax(z')[y]

    `s` 是 logit 缩放（论文取 30），`max_m` 是最大间隔。**输入必须是 logits**，
    因此输出层要用 `linear`（配 softmax 会在构造时被拒绝）。返回 (loss, grad)，
    grad 相对 logits，可直接经 linear 层反传。
    """
    V = len(logits)
    counts = _class_counts_vector(class_counts, V, "ldam_loss")
    max_m = float(max_m)
    if max_m < 0.0:
        raise ValueError(f"ldam_loss: max_m 必须 >= 0，收到 {max_m!r}")
    s = float(s)
    if s <= 0.0:
        raise ValueError(f"ldam_loss: s 必须 > 0，收到 {s!r}")
    y = _ce_scalar_index(target_idx, V, "ldam_loss")
    z = [s * float(v) for v in logits]
    z[y] -= s * (max_m / counts[y] ** 0.25)
    mx = max(z)
    exps = [math.exp(v - mx) for v in z]
    total = sum(exps)
    probs = [e / total for e in exps]
    loss = -(z[y] - mx - math.log(total))
    grad = [s * (p - (1.0 if k == y else 0.0)) for k, p in enumerate(probs)]
    return loss, grad


def ohem_loss(preds, targets, keep_ratio: float = 0.7):
    """OHEM（在线难例挖掘）：只保留损失最大的那一批样本

    `preds` 是**一批**概率向量（list of list），`targets` 是等长类别索引列表；
    `keep_ratio` 是保留的「容易样本」比例 (0,1)，即丢掉最容易的 `keep_ratio` 部分、
    只对最难的前 `1-keep_ratio` 训练。返回 `(loss, grads)`：`loss` 是被选中样本的
    平均损失，`grads` 与输入同形（未选中的样本梯度全 0）。

    注意：这是**批级**接口，不能塞进 `Model(loss_fn=...)`（那是逐样本接口）；
    自定义训练循环里用它替换逐样本 CE 即可。
    """
    if not isinstance(preds, (list, tuple)) or not preds:
        raise ValueError("ohem_loss: preds 必须是非空的概率向量列表")
    if not isinstance(targets, (list, tuple)) or len(targets) != len(preds):
        raise ValueError(
            f"ohem_loss: targets 长度必须与 preds 一致，收到 {len(targets) if isinstance(targets, (list, tuple)) else targets!r}")
    kr = float(keep_ratio)
    if not 0.0 <= kr < 1.0:
        raise ValueError(f"ohem_loss: keep_ratio 必须在 [0,1) 内，收到 {keep_ratio!r}")
    n = len(preds)
    losses = []
    grads = []
    for row, tgt in zip(preds, targets):
        loss, grad = cross_entropy_loss(row, tgt)
        losses.append(loss)
        grads.append(grad)
    keep = max(1, int(round((1.0 - kr) * n)))
    order = sorted(range(n), key=lambda i: losses[i], reverse=True)[:keep]
    chosen = set(order)
    out = [(grads[i] if i in chosen else [0.0] * len(grads[i])) for i in range(n)]
    mean_loss = sum(losses[i] for i in order) / keep
    return mean_loss, out


def distill_loss(student_logits, teacher_logits, T: float = 2.0):
    """知识蒸馏（soft-target KL）：`T^2 * KL(teacher_T || student_T)`

    对学生 logits 求导得到 `T * (p_student - p_teacher)`（p 是温度为 T 时的 softmax）。
    乘以 T^2 是为了让不同温度下的梯度量级可比（Hinton 等人的做法）。

    `teacher_logits` 直接作为 `Model(loss_fn='distill')` 的 target 传入即可；
    学生的输出层必须是 `linear`（本函数吃 logits，不吃概率）。
    """
    if not isinstance(T, (int, float)) or isinstance(T, bool) or not (T > 0.0):
        raise ValueError(f"distill_loss: T 必须是 > 0 的数字，收到 {T!r}")
    if not (T == T and T != float("inf")):
        raise ValueError("distill_loss: T 必须是有限数字")
    tgt, stud = _numeric_pair(student_logits, teacher_logits, "distill_loss")
    ps = _softmax_with_temp(stud, float(T))
    pt = _softmax_with_temp(tgt, float(T))
    loss = 0.0
    for a, b in zip(ps, pt):
        if b > 0.0:
            loss += b * math.log(b / max(a, 1e-300))
    loss *= T * T
    grad = [T * (a - b) for a, b in zip(ps, pt)]
    return loss, grad


def hint_loss(student_feat, teacher_feat, weight: float = 1.0):
    """隐藏层对齐（hint）：学生特征与教师特征的均方误差

    用于 FitNet 式的中间层蒸馏；两层维度必须一致（需要投影时自己套一层 `Layer`）。
    返回 `(loss, grad)`，`grad` 相对 `student_feat`；`weight` 用来调节与主损失的配比。
    """
    if not isinstance(weight, (int, float)) or isinstance(weight, bool):
        raise TypeError(f"hint_loss: weight 必须是数字，收到 {weight!r}")
    if not (weight >= 0.0 and weight == weight and weight != float("inf")):
        raise ValueError(f"hint_loss: weight 必须是非负有限数字，收到 {weight!r}")
    tgt, stud = _numeric_pair(student_feat, teacher_feat, "hint_loss")
    n = len(stud)
    if n == 0:
        raise ValueError("hint_loss: 特征不能为空")
    total = 0.0
    for a, b in zip(stud, tgt):
        d = a - b
        total += d * d
    loss = weight * total / n
    grad = [weight * 2.0 * (a - b) / n for a, b in zip(stud, tgt)]
    return loss, grad


def _pref_number(v, name, who):
    """偏好对齐损失的标量参数校验（拒绝 bool / 非有限 / NaN）。"""
    if not isinstance(v, (int, float)) or isinstance(v, bool):
        raise TypeError(f"{who}: {name} 必须是数字，收到 {v!r}")
    fv = float(v)
    if not (fv == fv) or fv in (float("inf"), float("-inf")):
        raise ValueError(f"{who}: {name} 必须是有限数字，收到 {v!r}")
    return fv


def _log_sigmoid(x):
    """数值稳定的 `log σ(x)`。"""
    if x >= 0.0:
        return -math.log1p(math.exp(-x))
    return x - math.log1p(math.exp(x))


def _sigmoid(x):
    """数值稳定的 sigmoid（避免 `exp` 溢出）。"""
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def _logit_odds(logp):
    """把 log 概率换成「logit」：`log(p/(1-p))`，用于 ORPO。"""
    if not (logp < 0.0):
        raise ValueError(f"preference_loss: ORPO 需要 log 概率 < 0，收到 {logp!r}")
    p = math.exp(logp)
    return logp - math.log1p(-p)


def preference_loss(chosen_logp, rejected_logp, ref_chosen_logp=None,
                    ref_rejected_logp=None, method: str = "dpo",
                    beta: float = 0.1, margin: float = 0.0,
                    sft_weight: float = 0.1):
    """偏好对齐损失：给定「好回答/坏回答」的对数概率，返回 `(loss, d_chosen, d_rejected)`

    支持五种方法（`method`）：

    - `'dpo'`：`-log σ(β(Δ − Δ_ref))`，`Δ = log π(c) − log π(r)`（需要参考模型的 logp）
    - `'ipo'`：`(Δ − Δ_ref − 1/(2β))²`，把偏好差直接回归到目标间隔
    - `'simpo'`：`-log σ(β·Δ − margin)`，不需要参考模型（SimPO 的卖点）
    - `'cpo'`：DPO 项 + `sft_weight · (−log π(c))`，一个损失里同时做对齐与 SFT
    - `'orpo'`：`-log σ(β·(logit π(c) − logit π(r)))`，在 logit 空间比较优势

    梯度是标量，分别相对 `chosen_logp` 与 `rejected_logp`（可以直接乘到序列的
    log-prob 梯度上）。`β>0`、`margin>=0`；只有 `'simpo'` / `'orpo'` 不需要 `ref_*`。
    """
    who = "preference_loss"
    if not isinstance(method, str):
        raise TypeError(f"{who}: method 必须是字符串，收到 {method!r}")
    m = method.strip().lower()
    if m not in ("dpo", "ipo", "simpo", "cpo", "orpo"):
        raise ValueError(f"{who}: method 必须是 dpo/ipo/simpo/cpo/orpo 之一，收到 {method!r}")
    beta = _pref_number(beta, "beta", who)
    margin = _pref_number(margin, "margin", who)
    sft_weight = _pref_number(sft_weight, "sft_weight", who)
    if not (beta > 0.0):
        raise ValueError(f"{who}: beta 必须 > 0，收到 {beta!r}")
    if margin < 0.0:
        raise ValueError(f"{who}: margin 必须 >= 0，收到 {margin!r}")
    if sft_weight < 0.0:
        raise ValueError(f"{who}: sft_weight 必须 >= 0，收到 {sft_weight!r}")
    c = _pref_number(chosen_logp, "chosen_logp", who)
    r = _pref_number(rejected_logp, "rejected_logp", who)

    needs_ref = m in ("dpo", "ipo", "cpo")
    if needs_ref:
        if ref_chosen_logp is None or ref_rejected_logp is None:
            raise ValueError(f"{who}: method={m!r} 需要 ref_chosen_logp 与 ref_rejected_logp")
        rc = _pref_number(ref_chosen_logp, "ref_chosen_logp", who)
        rr = _pref_number(ref_rejected_logp, "ref_rejected_logp", who)
    else:
        rc = rr = 0.0

    delta = c - r
    if m == "dpo":
        x = beta * (delta - (rc - rr))
        loss = -_log_sigmoid(x)
        d_delta = beta * (_sigmoid(x) - 1.0)
        dc, dr = d_delta, -d_delta
    elif m == "ipo":
        target = 1.0 / (2.0 * beta)
        gap = delta - (rc - rr) - target
        loss = gap * gap
        d_delta = 2.0 * gap
        dc, dr = d_delta, -d_delta
    elif m == "simpo":
        x = beta * delta - margin
        loss = -_log_sigmoid(x)
        d_delta = beta * (_sigmoid(x) - 1.0)
        dc, dr = d_delta, -d_delta
    elif m == "orpo":
        gap = _logit_odds(c) - _logit_odds(r)
        x = beta * gap
        loss = -_log_sigmoid(x)
        d_x = _sigmoid(x) - 1.0
        # d logit_odds / d logp = 1 + p/(1-p)
        dc = beta * d_x * (1.0 + math.exp(c) / (1.0 - math.exp(c)))
        dr = -beta * d_x * (1.0 + math.exp(r) / (1.0 - math.exp(r)))
    else:  # cpo
        x = beta * (delta - (rc - rr))
        loss = -_log_sigmoid(x) + sft_weight * (-c)
        d_delta = beta * (_sigmoid(x) - 1.0)
        dc, dr = d_delta - sft_weight, -d_delta
    return loss, dc, dr


def kto_loss(logp, ref_logp, desirable: bool = True, beta: float = 0.1,
             baseline: float = 0.0):
    """KTO（Kahneman-Tversky 优化）：单条回答的二分类式对齐

    `desirable=True` 时 `loss = 1 − σ(β(z − baseline))`，`False` 时
    `loss = 1 − σ(β(baseline − z))`，其中 `z = log π − log π_ref`。
    返回 `(loss, d_logp)`；`baseline` 通常取「参考模型下的平均 KL」的估计值，
    教学实现里默认为 0。
    """
    who = "kto_loss"
    beta = _pref_number(beta, "beta", who)
    baseline = _pref_number(baseline, "baseline", who)
    if not (beta > 0.0):
        raise ValueError(f"{who}: beta 必须 > 0，收到 {beta!r}")
    if not isinstance(desirable, bool):
        raise TypeError(f"{who}: desirable 必须是 bool，收到 {desirable!r}")
    z = _pref_number(logp, "logp", who) - _pref_number(ref_logp, "ref_logp", who)

    if desirable:
        u = beta * (z - baseline)
    else:
        u = beta * (baseline - z)
    sig = _sigmoid(u)
    loss = 1.0 - sig
    # d(1 - σ(u))/du = -σ(u)(1-σ(u))
    du = -sig * (1.0 - sig)
    d_logp = du * beta if desirable else -du * beta
    return loss, d_logp


def _mt_vector(values, name, who):
    """多任务权重工具的公共入参校验：非空、每个元素是有限数字（bool 不算）。"""
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError(f"{who}: {name} 必须是非空序列")
    out = []
    for i, v in enumerate(values):
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise TypeError(f"{who}: {name}[{i}] 必须是数字，收到 {type(v).__name__}")
        f = float(v)
        if not (f == f and -float("inf") < f < float("inf")):
            raise ValueError(f"{who}: {name}[{i}] 必须是有限数，收到 {f!r}")
        out.append(f)
    return out


def uncertainty_weighting(losses, log_vars):
    """Kendall 等人的多任务不确定性加权。

    ``L = Σ_i exp(-s_i)·L_i + s_i``，其中 ``s_i = log_vars[i]`` 是任务 i 的
    对数方差（可学习参数）。返回 ``(total, weights, d_losses, d_log_vars)``：
    ``weights[i] = exp(-s_i)``，``d_losses[i] = exp(-s_i)``，
    ``d_log_vars[i] = 1 - exp(-s_i)·L_i``。
    """
    ls = _mt_vector(losses, "losses", "uncertainty_weighting")
    ss = _mt_vector(log_vars, "log_vars", "uncertainty_weighting")
    if len(ls) != len(ss):
        raise ValueError(
            f"uncertainty_weighting: losses 长度 {len(ls)} 与 log_vars 长度 {len(ss)} 不一致"
        )
    weights = []
    total = 0.0
    for li, si in zip(ls, ss):
        w = math.exp(-si)
        weights.append(w)
        total += w * li + si
    d_losses = list(weights)
    d_log_vars = [1.0 - w * li for w, li in zip(weights, ls)]
    return total, weights, d_losses, d_log_vars


def gradnorm_weights(losses, grad_norms, alpha=1.5, lr=0.025,
                     weights=None, G0=None):
    """GradNorm 的一步权重更新（纯函数、无状态）。

    ``r_i = (L_i / mean(L))^alpha``、``Ḡ = mean(G_i)``，
    目标 ``target_i = Ḡ·r_i``；损失 ``L1 = Σ|G_i − target_i|``。
    ``G0`` 是参考尺度（缺省用当前 ``Ḡ``），
    权重更新 ``w_i −= lr·sign(G_i − target_i)·G0/r_i``，并归一化到均值为 1。

    返回 ``{"weights", "targets", "loss", "gradients"}``（都是 list）。
    """
    ls = _mt_vector(losses, "losses", "gradnorm_weights")
    gs = _mt_vector(grad_norms, "grad_norms", "gradnorm_weights")
    if len(ls) != len(gs):
        raise ValueError(
            f"gradnorm_weights: losses 长度 {len(ls)} 与 grad_norms 长度 {len(gs)} 不一致"
        )
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise TypeError("gradnorm_weights: alpha 必须是数字")
    if not (alpha > 0):
        raise ValueError(f"gradnorm_weights: alpha 必须 > 0，收到 {alpha!r}")
    if isinstance(lr, bool) or not isinstance(lr, (int, float)):
        raise TypeError("gradnorm_weights: lr 必须是数字")
    if not (lr > 0):
        raise ValueError(f"gradnorm_weights: lr 必须 > 0，收到 {lr!r}")
    for x in ls:
        if x <= 0.0:
            raise ValueError(f"gradnorm_weights: losses 必须全为正，收到 {x!r}")

    if weights is None:
        ws = [1.0] * len(ls)
    else:
        ws = _mt_vector(weights, "weights", "gradnorm_weights")
        if len(ws) != len(ls):
            raise ValueError(
                f"gradnorm_weights: weights 长度 {len(ws)} 与 losses 长度 {len(ls)} 不一致"
            )
        for x in ws:
            if x <= 0.0:
                raise ValueError(f"gradnorm_weights: weights 必须全为正，收到 {x!r}")

    mean_l = sum(ls) / len(ls)
    mean_g = sum(gs) / len(gs)
    ref = mean_g if G0 is None else float(G0)
    if isinstance(G0, bool) or (G0 is not None and not isinstance(G0, (int, float))):
        raise TypeError("gradnorm_weights: G0 必须是数字或 None")
    if ref <= 0.0:
        raise ValueError(f"gradnorm_weights: G0 必须 > 0，收到 {ref!r}")

    ratios = [(li / mean_l) ** float(alpha) for li in ls]
    targets = [mean_g * r for r in ratios]
    loss = 0.0
    for gi, ti in zip(gs, targets):
        loss += abs(gi - ti)
    gradients = []
    new_w = []
    for wi, gi, ti, r in zip(ws, gs, targets, ratios):
        diff = gi - ti
        sign = 0.0 if diff == 0.0 else (1.0 if diff > 0.0 else -1.0)
        grad = sign * ref / r
        gradients.append(grad)
        new_w.append(wi - float(lr) * grad)
    # 归一化到均值为 1（保持整体尺度，免得学习率被间接改变）
    lo = min(new_w)
    if lo <= 0.0:
        shift = 1.0 - lo
        new_w = [w + shift for w in new_w]
    m = sum(new_w) / len(new_w)
    new_w = [w / m for w in new_w]
    return {"weights": new_w, "targets": targets, "loss": loss, "gradients": gradients}


def _flat_dot(a, b):
    if isinstance(a, (list, tuple)):
        if not isinstance(b, (list, tuple)) or len(a) != len(b):
            raise ValueError("pcgrad: 两个梯度的结构不一致")
        return sum(_flat_dot(x, y) for x, y in zip(a, b))
    if isinstance(a, bool) or not isinstance(a, (int, float)):
        raise TypeError("pcgrad: 梯度里只能是数字或数字列表")
    return float(a) * float(b)


def _flat_sub(a, b):
    if isinstance(a, (list, tuple)):
        return [_flat_sub(x, y) for x, y in zip(a, b)]
    return float(a) - float(b)


def _flat_scale(a, k):
    if isinstance(a, (list, tuple)):
        return [_flat_scale(x, k) for x in a]
    return float(a) * k


def pcgrad(grad_a, grad_b, eps=1e-12):
    """PCGrad：两个任务梯度冲突（内积 < 0）时互相投影掉冲突分量。

    返回 ``(grad_a_projected, grad_b_projected)``；不冲突时原样返回副本。
    梯度用「数字的嵌套 list」表示（与 ``Model.backward`` 的返回同构）。
    """
    if isinstance(eps, bool) or not isinstance(eps, (int, float)):
        raise TypeError("pcgrad: eps 必须是数字")
    if not (eps > 0):
        raise ValueError(f"pcgrad: eps 必须 > 0，收到 {eps!r}")
    dot_ab = _flat_dot(grad_a, grad_b)
    ga = [_flat_scale(x, 1.0) for x in grad_a] if isinstance(grad_a, list) else grad_a
    gb = [_flat_scale(x, 1.0) for x in grad_b] if isinstance(grad_b, list) else grad_b
    if dot_ab < 0.0:
        n_b = _flat_dot(grad_b, grad_b)
        n_a = _flat_dot(grad_a, grad_a)
        if n_b > eps:
            coef_a = dot_ab / n_b
            ga = _flat_sub(grad_a, _flat_scale(grad_b, coef_a))
        if n_a > eps:
            coef_b = dot_ab / n_a
            gb = _flat_sub(grad_b, _flat_scale(grad_a, coef_b))
    return ga, gb


def mse_loss(pred: List[float], target):
    """MSE 损失 + 对 pred 的梯度

    target 可以是序列，也可以是单个标量（单输出回归传 y=0.5 也行，
    与 Model._wrap_loss 的行为保持一致）。
    """
    if not isinstance(target, (list, tuple)):
        target = [target]
    if len(pred) == 0:
        raise ValueError("mse_loss: pred 不能为空")
    if len(pred) != len(target):
        # 不能靠 zip 静默截断：分母用的是 len(pred)，截断后会得到一个"看起来正常"的错值
        raise ValueError(
            f"mse_loss: pred 长度 {len(pred)} != target 长度 {len(target)}，"
            f"请对齐维度（单输出回归请传 y=[value]，多输出请传 y=[v1, v2, ...]）"
        )
    n = len(pred)
    # 用乘法而不是 `** 2`：CPython 的 float pow 在 |差值| >~1.34e154 时会抛
    # OverflowError（而 mse_loss([1e308],[-1e308]) 又返回 inf，行为不一致）；
    # 乘法按 IEEE 溢出成 inf，语义统一
    diffs = [p - t for p, t in zip(pred, target)]
    loss = sum(d * d for d in diffs) / n
    grad = [2 * d / n for d in diffs]
    return loss, grad


def cross_entropy_loss(pred: List[float], target_idx, label_smoothing: float = 0.0):
    """
    pred: softmax 输出 (概率分布)
    target_idx: 正确类别的索引 (int)，或 one-hot / 概率向量 (list)

    两种目标都支持：
        索引     -> loss = -log(pred[idx])，梯度 = pred - one_hot(idx)
        向量     -> loss = -Σ t·log(pred)，梯度 = pred - t
    后者用于 TextProcessor.prepare_data 产出的 one-hot 标签。

    label_smoothing > 0 时把目标分布软化成
        q = (1-ε)·target + ε/K        （K = 类别数）
    loss = -Σ q·log(pred)，梯度 = pred·Σq - q。
    ε 取值 (0,1)；默认 0 时走原来的精确路径（逐位不变）。
    """
    eps = 1e-15
    if label_smoothing:
        return _cross_entropy_smoothed(pred, target_idx, label_smoothing, eps)
    if isinstance(target_idx, (list, tuple)):
        if len(target_idx) != len(pred):
            raise ValueError(
                f"cross_entropy_loss: target 长度 {len(target_idx)} != pred 长度 {len(pred)}，"
                f"请对齐维度（原先 zip 会静默截断）"
            )
        if any(float(t) < 0.0 for t in target_idx):
            raise ValueError("cross_entropy_loss: 目标向量不能含负值")
        # softmax+CE 对任意非负软标签的联合梯度是 s*p - t（s = Σt）。
        # 只在 Σt == 1 时才退化成 p - t；全 0 目标应得到全 0 梯度，
        # 原先写成 p - t 会让 loss=0 却有非零梯度。
        ssum = sum(float(t) for t in target_idx)
        loss = -sum(float(t) * math.log(max(float(p), eps))
                    for t, p in zip(target_idx, pred))
        grad = [ssum * float(p) - float(t) for p, t in zip(pred, target_idx)]
        return loss, grad
    # bool 是 int 的子类，会静默变成索引 0/1
    if isinstance(target_idx, bool):
        raise TypeError(f"cross_entropy_loss: 类别索引不能是 bool，收到 {target_idx!r}")
    if isinstance(target_idx, float):
        if not target_idx.is_integer():
            raise ValueError(f"cross_entropy_loss: 类别索引必须是整数，收到 {target_idx}")
        target_idx = int(target_idx)
    if not isinstance(target_idx, int):
        try:
            target_idx = operator.index(target_idx)
        except TypeError:
            raise TypeError(
                f"cross_entropy_loss: 目标必须是类别索引或向量，收到 {target_idx!r}"
            )
    # 负数索引在 Python 里会回绕成最后一个类，必须拦掉
    if not 0 <= target_idx < len(pred):
        raise ValueError(
            f"cross_entropy_loss: 类别索引 {target_idx} 越界（pred 长度 {len(pred)}）"
        )
    p = max(pred[target_idx], eps)
    loss = -math.log(p)
    # softmax + CE 的梯度 = pred - one_hot(target)
    grad = list(pred)
    grad[target_idx] -= 1.0
    return loss, grad


def _ce_scalar_index(target, n, who="cross_entropy_loss"):
    """把标量目标规范成合法类别索引（bool / 非整数 float / 越界都拒绝）"""
    if isinstance(target, bool):
        raise TypeError(f"{who}: 类别索引不能是 bool，收到 {target!r}")
    if isinstance(target, float):
        if not target.is_integer():
            raise ValueError(f"{who}: 类别索引必须是整数，收到 {target}")
        target = int(target)
    if not isinstance(target, int):
        try:
            target = operator.index(target)
        except TypeError:
            raise TypeError(f"{who}: 目标必须是类别索引，收到 {target!r}")
    if not 0 <= target < n:
        raise ValueError(f"{who}: 类别索引 {target} 越界（pred 长度 {n}）")
    return target


def _cross_entropy_smoothed(pred, target, smoothing, eps):
    """cross_entropy_loss 的 label_smoothing 分支（目标分布 q 见调用处文档）"""
    if not 0.0 < float(smoothing) < 1.0:
        raise ValueError(
            f"cross_entropy_loss: label_smoothing 必须在 (0,1)，收到 {smoothing!r}")
    k = len(pred)
    if isinstance(target, (list, tuple)):
        if len(target) != k:
            raise ValueError(
                f"cross_entropy_loss: target 长度 {len(target)} != pred 长度 {k}，"
                f"请对齐维度（原先 zip 会静默截断）"
            )
        if any(float(t) < 0.0 for t in target):
            raise ValueError("cross_entropy_loss: 目标向量不能含负值")
        ts = [float(t) for t in target]
    else:
        idx = _ce_scalar_index(target, k)
        ts = [0.0] * k
        ts[idx] = 1.0
    uniform = float(smoothing) / k
    q = [(1.0 - float(smoothing)) * t + uniform for t in ts]
    loss = -sum(qi * math.log(max(float(p), eps)) for qi, p in zip(q, pred))
    ssum = sum(q)
    grad = [ssum * float(p) - qi for p, qi in zip(pred, q)]
    return loss, grad


def focal_loss(pred: List[float], target_idx, gamma: float = 2.0, alpha=None):
    """Focal Loss：FL = -α_t·(1-p_t)^γ·log(p_t)

    针对类别不平衡：易分样本（p_t 接近 1）的 (1-p_t)^γ 会把它的梯度压小，
    训练集中到难样本上。pred 是 softmax 概率，target 只支持类别索引
    （软标签下 focal 的定义不唯一）。alpha 可为标量或逐类列表（None=1）。

    返回 (loss, grad)，grad 已含 softmax 与 focal 的联合求导，可直接喂给
    `_softmax_deriv`（返回 1.0）那一层的 backward。
    """
    if gamma is None or gamma < 0:
        raise ValueError(f"focal_loss: gamma 必须 >= 0，收到 {gamma!r}")
    k = len(pred)
    idx = _ce_scalar_index(target_idx, k, "focal_loss")
    if isinstance(alpha, (list, tuple)):
        if len(alpha) != k:
            raise ValueError(
                f"focal_loss: alpha 长度 {len(alpha)} != pred 长度 {k}")
        a_t = float(alpha[idx])
    else:
        a_t = 1.0 if alpha is None else float(alpha)
    p = min(max(float(pred[idx]), 1e-15), 1.0)
    rest = 1.0 - p
    log_p = math.log(p)
    if rest <= 0.0:
        # p_t 恰为 1：loss 与梯度都是 0（γ>0 时）
        return 0.0, [0.0] * k
    p_gamma = rest ** gamma
    loss = -a_t * p_gamma * log_p
    # dz_j = A·(δ_tj - p_j)，A = α·[γ·p_t·(1-p_t)^(γ-1)·log(p_t) - (1-p_t)^γ]
    term = gamma * p * (rest ** (gamma - 1.0)) * log_p if gamma > 0 else 0.0
    a = a_t * (term - p_gamma)
    grad = [-a * float(pj) for pj in pred]
    grad[idx] += a
    return loss, grad


def bce_loss(pred, target, eps: float = 1e-12):
    """二元交叉熵（pred 是概率，通常来自 output 层的 sigmoid 激活）

    target 可为 0/1 标量，或与 pred 等长的 0/1 列表。
    loss 取逐元素平均；返回 (loss, grad)，grad 相对 pred
    （Layer 的 sigmoid 导数会再乘上去，所以对 logits 的梯度是对的）。
    """
    if isinstance(target, (list, tuple)):
        if len(target) != len(pred):
            raise ValueError(
                f"bce_loss: target 长度 {len(target)} != pred 长度 {len(pred)}")
        ts, ps = [float(t) for t in target], [float(p) for p in pred]
    else:
        ts, ps = [float(target)], [float(p) for p in pred]
        if len(ps) != 1:
            raise ValueError(
                f"bce_loss: 标量 target 只能配单个输出，pred 长度是 {len(ps)}；"
                f"多输出请传等长列表")
    n = len(ts)
    loss = 0.0
    grad = []
    for y, p in zip(ts, ps):
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"bce_loss: pred 必须是概率 [0,1]，收到 {p}")
        q = min(max(p, eps), 1.0 - eps)
        loss += -(y * math.log(q) + (1.0 - y) * math.log(1.0 - q))
        grad.append((q - y) / (q * (1.0 - q)) / n)
    return loss / n, grad


def bce_with_logits_loss(logits, target):
    """带 logits 的二元交叉熵（数值稳定，内部做 sigmoid）

    loss = max(z,0) - z·y + log(1+exp(-|z|))，返回 (loss, grad)，grad 相对 logits
    （= sigmoid(z) - y）。target 规则同 bce_loss。
    """
    if isinstance(target, (list, tuple)):
        if len(target) != len(logits):
            raise ValueError(
                f"bce_with_logits_loss: target 长度 {len(target)} != logits 长度 {len(logits)}")
        ts, zs = [float(t) for t in target], [float(z) for z in logits]
    else:
        ts, zs = [float(target)], [float(z) for z in logits]
        if len(zs) != 1:
            raise ValueError(
                f"bce_with_logits_loss: 标量 target 只能配单个输出，logits 长度是 {len(zs)}")
    n = len(ts)
    loss = 0.0
    grad = []
    for y, z in zip(ts, zs):
        loss += max(z, 0.0) - z * y + math.log1p(math.exp(-abs(z)))
        if z >= 0:
            s = 1.0 / (1.0 + math.exp(-z))
        else:
            e = math.exp(z)
            s = e / (1.0 + e)
        grad.append((s - y) / n)
    return loss / n, grad


def huber_loss(pred, target, delta: float = 1.0):
    """Huber 损失（Smooth L1）：|e|<=delta 用 0.5e²，否则 delta(|e|-0.5delta)

    对离群点比 MSE 稳、对小误差比 MAE 光滑。target 可为标量或等长列表，
    loss 取逐元素平均；返回 (loss, grad)，grad 相对 pred。
    """
    if delta is None or delta <= 0:
        raise ValueError(f"huber_loss: delta 必须 > 0，收到 {delta!r}")
    if isinstance(target, (list, tuple)):
        if len(target) != len(pred):
            raise ValueError(
                f"huber_loss: target 长度 {len(target)} != pred 长度 {len(pred)}")
        ts, ps = [float(t) for t in target], [float(p) for p in pred]
    else:
        ts, ps = [float(target)], [float(p) for p in pred]
        if len(ps) != 1:
            raise ValueError(
                f"huber_loss: 标量 target 只能配单个输出，pred 长度是 {len(ps)}")
    n = len(ts)
    loss = 0.0
    grad = []
    for y, p in zip(ts, ps):
        e = p - y
        if abs(e) <= delta:
            loss += 0.5 * e * e
            grad.append(e / n)
        else:
            loss += delta * (abs(e) - 0.5 * delta)
            grad.append((delta if e > 0 else -delta) / n)
    return loss / n, grad


def kl_divergence(pred, target, eps: float = 1e-12):
    """KL(target ‖ pred) = Σ t·log(t/p)

    pred 是模型输出的概率分布，target 是目标分布（不要求归一化）。
    返回 (loss, grad)，grad 相对 pred（= -t/p）。
    """
    if len(target) != len(pred):
        raise ValueError(
            f"kl_divergence: target 长度 {len(target)} != pred 长度 {len(pred)}")
    loss = 0.0
    grad = []
    for t, p in zip(target, pred):
        t, p = float(t), float(p)
        if t < 0.0:
            raise ValueError("kl_divergence: target 不能含负值")
        q = max(p, eps)
        if t > 0.0:
            loss += t * (math.log(t) - math.log(q))
        grad.append(-t / q)
    return loss, grad


def _loss_vector(v, name, who):
    """校验一维数字序列（拒绝 bool / NaN / Inf / 空）"""
    if not isinstance(v, (list, tuple)):
        raise TypeError(f"{who}: {name} 必须是序列，收到 {type(v).__name__}")
    out = []
    for x in v:
        if isinstance(x, bool) or not isinstance(x, (int, float)):
            raise TypeError(f"{who}: {name} 只能含数字，收到 {x!r}")
        x = float(x)
        if x != x or x in (float("inf"), float("-inf")):
            raise ValueError(f"{who}: {name} 不能含 NaN/Inf")
        out.append(x)
    if not out:
        raise ValueError(f"{who}: {name} 不能为空")
    return out


def _loss_norm(x):
    return math.sqrt(sum(v * v for v in x))


def triplet_loss(anchor, positive, negative, margin: float = 1.0):
    """Triplet 损失：max(0, d(a,p) - d(a,n) + margin)，d 为欧氏距离

    返回 (loss, d_anchor, d_positive, d_negative)，三个梯度都相对输入向量。
    margin 必须 > 0：等于 0 时正负样本距离相等就没有梯度，等于静默失效。
    已满足间隔时 loss 为 0、三个梯度全 0。
    """
    who = "triplet_loss"
    a = _loss_vector(anchor, "anchor", who)
    p = _loss_vector(positive, "positive", who)
    n = _loss_vector(negative, "negative", who)
    if not (len(a) == len(p) == len(n)):
        raise ValueError(f"{who}: 三个向量长度必须相同，收到 {len(a)}/{len(p)}/{len(n)}")
    margin = float(margin)
    if margin <= 0.0:
        raise ValueError(f"{who}: margin 必须 > 0，收到 {margin!r}")

    dap = _loss_norm([a[i] - p[i] for i in range(len(a))])
    dan = _loss_norm([a[i] - n[i] for i in range(len(a))])
    loss = dap - dan + margin
    if loss <= 0.0:
        zeros = [0.0] * len(a)
        return 0.0, list(zeros), list(zeros), list(zeros)

    d_anchor = [0.0] * len(a)
    d_positive = [0.0] * len(a)
    d_negative = [0.0] * len(a)
    for i in range(len(a)):
        d_anchor[i] = (a[i] - p[i]) / dap - (a[i] - n[i]) / dan
        d_positive[i] = -(a[i] - p[i]) / dap
        d_negative[i] = (a[i] - n[i]) / dan
    return loss, d_anchor, d_positive, d_negative


def info_nce_loss(anchor, positive, negatives, temperature: float = 0.07):
    """InfoNCE 对比损失：正样本 1 个 + 负样本若干

    相似度用归一化点积（余弦），loss = -log( exp(s_pos/τ) / Σ_j exp(s_j/τ) )，
    j 遍历 [positive] + negatives。
    返回 (loss, d_anchor, d_positive, d_negatives)。
    """
    who = "info_nce_loss"
    a = _loss_vector(anchor, "anchor", who)
    if not isinstance(negatives, (list, tuple)) or not negatives:
        raise ValueError(f"{who}: negatives 必须是非空序列")
    cands = [_loss_vector(positive, "positive", who)]
    for k, neg in enumerate(negatives):
        cands.append(_loss_vector(neg, f"negatives[{k}]", who))
    for k, c in enumerate(cands):
        if len(c) != len(a):
            raise ValueError(f"{who}: 第 {k} 个候选长度 {len(c)} != anchor 长度 {len(a)}")
    temperature = float(temperature)
    if temperature <= 0.0:
        raise ValueError(f"{who}: temperature 必须 > 0，收到 {temperature!r}")

    a_norm = _loss_norm(a)
    if a_norm == 0.0:
        raise ValueError(f"{who}: anchor 不能是零向量")
    c_norms = [_loss_norm(c) for c in cands]
    for k, cw in enumerate(c_norms):
        if cw == 0.0:
            raise ValueError(f"{who}: 第 {k} 个候选不能是零向量")

    sims = [sum(map(_mul, a, c)) / (a_norm * c_norms[k]) / temperature
            for k, c in enumerate(cands)]
    mx = max(sims)
    exps = [math.exp(v - mx) for v in sims]
    total = sum(exps)
    loss = -(sims[0] - mx - math.log(total))

    probs = [e / total for e in exps]
    d_anchor = [0.0] * len(a)
    d_cands = [[0.0] * len(c) for c in cands]
    for j, c in enumerate(cands):
        coeff = probs[j] - (1.0 if j == 0 else 0.0)
        if coeff == 0.0:
            continue
        inv_t = 1.0 / temperature
        for i in range(len(a)):
            d_anchor[i] += coeff * (inv_t * c[i] / (a_norm * c_norms[j])
                                    - sims[j] * a[i] / (a_norm * a_norm))
        for i in range(len(c)):
            d_cands[j][i] += coeff * (inv_t * a[i] / (a_norm * c_norms[j])
                                      - sims[j] * c[i] / (c_norms[j] ** 2))
    d_pos = d_cands[0]
    d_negs = d_cands[1:]
    return loss, d_anchor, d_pos, d_negs


def _loss_flatten(x, who):
    """把嵌套 list 摊平，返回 (flat, shape)；顺带校验形状一致与数值有限"""
    shape = []
    cur = x
    while isinstance(cur, (list, tuple)):
        shape.append(len(cur))
        if not cur:
            break
        cur = cur[0]
    flat = []

    def walk(v, depth):
        if depth == len(shape):
            if isinstance(v, (list, tuple)):
                raise ValueError(f"{who}: 嵌套层数不一致")
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise TypeError(f"{who}: 只能含数字，收到 {v!r}")
            v = float(v)
            if v != v or v in (float("inf"), float("-inf")):
                raise ValueError(f"{who}: 不能含 NaN/Inf")
            flat.append(v)
            return
        if not isinstance(v, (list, tuple)):
            raise ValueError(f"{who}: 嵌套层数不一致")
        if len(v) != shape[depth]:
            raise ValueError(f"{who}: 第 {depth} 维长度不一致（{len(v)} != {shape[depth]}）")
        for item in v:
            walk(item, depth + 1)

    walk(x, 0)
    return flat, tuple(shape)


def _loss_unflatten(flat, shape):
    if not shape:
        return flat[0]
    if len(shape) == 1:
        return list(flat)
    step = len(flat) // shape[0]
    return [_loss_unflatten(flat[i * step:(i + 1) * step], shape[1:]) for i in range(shape[0])]


def dice_loss(pred, target, eps: float = 1e-6):
    """Dice 损失（分割）：1 - (2|p∩t| + eps') / (|p| + |t| + eps')

    pred 是概率（0~1），target 是 0/1 掩码，形状需一致（(C,H,W) 或扁平列表均可）。
    返回 (loss, grad)，grad 相对 pred。全局 Dice（不是逐类平均）。
    """
    who = "dice_loss"
    flat_p, shape_p = _loss_flatten(pred, who)
    flat_t, shape_t = _loss_flatten(target, who)
    if shape_p != shape_t:
        raise ValueError(f"{who}: pred 形状 {shape_p} 与 target 形状 {shape_t} 不一致")
    if not flat_p:
        raise ValueError(f"{who}: 输入不能为空")
    eps = float(eps)
    if eps <= 0.0:
        raise ValueError(f"{who}: eps 必须 > 0，收到 {eps!r}")

    num = 2.0 * sum(p * t for p, t in zip(flat_p, flat_t))
    den = sum(flat_p) + sum(flat_t) + eps
    loss = 1.0 - num / den
    grad_flat = [-2.0 * t / den + num / (den * den) for t in flat_t]
    return loss, _loss_unflatten(grad_flat, shape_p)


def ctc_loss(log_probs, targets, blank: int = 0):
    """CTC 损失（单样本，log 空间前向后向）

    log_probs: (T, K) 的 log 概率（softmax 之后取 log）
    targets: 目标标签序列（不含 blank）
    返回 (loss, grad)，grad 相对 log_probs，形状 (T, K)。

    用 log-sum-exp 累加，长序列不会下溢；不可达的标签序列直接报错。
    """
    who = "ctc_loss"
    flat_lp, shape = _loss_flatten(log_probs, who)
    if len(shape) != 2:
        raise ValueError(f"{who}: log_probs 必须是 (T, K) 二维结构，收到形状 {shape}")
    T, K = shape
    if T <= 0 or K <= 0:
        raise ValueError(f"{who}: log_probs 不能有空维度（{T}×{K}）")
    if isinstance(blank, bool) or not isinstance(blank, int) or not 0 <= blank < K:
        raise ValueError(f"{who}: blank 必须是 [0, {K}) 的整数，收到 {blank!r}")
    if isinstance(targets, (int, float, str)) or not isinstance(targets, (list, tuple)):
        raise TypeError(f"{who}: targets 必须是序列，收到 {type(targets).__name__}")
    labels = []
    for t in targets:
        if isinstance(t, bool) or not isinstance(t, int):
            raise TypeError(f"{who}: targets 只能含整数，收到 {t!r}")
        if not 0 <= t < K:
            raise ValueError(f"{who}: targets 含越界标签 {t}（词表大小 {K}）")
        if t == blank:
            raise ValueError(f"{who}: targets 不能包含 blank={blank}（与空符号冲突）")
        labels.append(t)
    if not labels:
        raise ValueError(f"{who}: targets 不能为空")

    lp = [flat_lp[t * K:(t + 1) * K] for t in range(T)]
    L = len(labels)
    S = 2 * L + 1
    ext = [blank if s % 2 == 0 else labels[s // 2] for s in range(S)]
    NEG = float("-inf")

    def lse(vals):
        m = max(vals)
        if m == NEG:
            return NEG
        return m + math.log(sum(math.exp(v - m) for v in vals))

    alpha = [[NEG] * S for _ in range(T)]
    alpha[0][0] = lp[0][ext[0]]
    if S > 1:
        alpha[0][1] = lp[0][ext[1]]
    for t in range(1, T):
        for s in range(S):
            prev = [alpha[t - 1][s]]
            if s >= 1:
                prev.append(alpha[t - 1][s - 1])
            if s >= 2 and ext[s] != blank and ext[s] != ext[s - 2]:
                prev.append(alpha[t - 1][s - 2])
            base = lse(prev)
            alpha[t][s] = base + lp[t][ext[s]] if base != NEG else NEG

    beta = [[NEG] * S for _ in range(T)]
    beta[T - 1][S - 1] = lp[T - 1][ext[S - 1]]
    if S > 1:
        beta[T - 1][S - 2] = lp[T - 1][ext[S - 2]]
    for t in range(T - 2, -1, -1):
        for s in range(S):
            nxt = [beta[t + 1][s]]
            if s + 1 < S:
                nxt.append(beta[t + 1][s + 1])
            if s + 2 < S and ext[s + 2] != blank and ext[s + 2] != ext[s]:
                nxt.append(beta[t + 1][s + 2])
            base = lse(nxt)
            beta[t][s] = base + lp[t][ext[s]] if base != NEG else NEG

    ends = [alpha[T - 1][S - 1]]
    if S > 1:
        ends.append(alpha[T - 1][S - 2])
    log_p = lse(ends)
    if log_p == NEG:
        raise ValueError(f"{who}: 该标签序列在给定长度下不可达（T={T}, 标签 {L} 个）")

    grad = [[0.0] * K for _ in range(T)]
    for t in range(T):
        for s in range(S):
            a_t, b_t = alpha[t][s], beta[t][s]
            if a_t == NEG or b_t == NEG:
                continue
            grad[t][ext[s]] -= math.exp(a_t + b_t - log_p - lp[t][ext[s]])
    return -log_p, grad


# ============================================================================
#  Optimizer: Adam
# ============================================================================

class Adam:
    """Adam 优化器（一阶矩 + 二阶矩 + 偏差修正）

    每个参数：m = b1*m + (1-b1)*g，v = b2*v + (1-b2)*g*g，
              p -= lr * (m/(1-b1^t)) / (sqrt(v/(1-b2^t)) + eps)
    动量存在每层上（权重 m_w/v_w，偏置 m_b/v_b），`step_count` 记已更新步数。
    想精确续训要用 `_optimizer_state` / `load_optimizer_state` 连状态一起存取；
    只存权重再 load 会从 step_count=0 重新开始，偏差修正对不上。
    """

    def __init__(self, model: MLP, lr: float = 1e-3,
                 beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.model = model
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.step_count = 0

    def step(self, all_grads):
        """
        all_grads: List[(grad_w, grad_b)] 每层一个元组
        """
        self.step_count += 1
        # 偏差修正系数只依赖 step_count，必须提到循环外：
        # 原来写在内层，每个权重都要重新算一次 beta ** step_count（pow 很贵）
        b1, b2, lr, eps = self.beta1, self.beta2, self.lr, self.eps
        c1 = 1 - b1 ** self.step_count
        c2 = 1 - b2 ** self.step_count
        o1, o2 = 1 - b1, 1 - b2
        sqrt = math.sqrt

        for (layer, (grad_w, grad_b)) in zip(self.model.layers, all_grads):
            m_w, v_w, W = layer.m_w, layer.v_w, layer.weights
            m_b, v_b, B = layer.m_b, layer.v_b, layer.biases

            for j in range(layer.fan_out):
                # bias
                g = grad_b[j]
                m = b1 * m_b[j] + o1 * g
                v = b2 * v_b[j] + o2 * g * g
                m_b[j] = m
                v_b[j] = v
                B[j] -= lr * (m / c1) / (sqrt(v / c2) + eps)

                # weight
                gwj, mwj, vwj, Wj = grad_w[j], m_w[j], v_w[j], W[j]
                for i in range(layer.fan_in):
                    g = gwj[i]
                    m = b1 * mwj[i] + o1 * g
                    v = b2 * vwj[i] + o2 * g * g
                    mwj[i] = m
                    vwj[i] = v
                    Wj[i] -= lr * (m / c1) / (sqrt(v / c2) + eps)


class AdamW:
    """AdamW：Adam + 解耦权重衰减（现代 LLM 训练的默认优化器）

    与 Adam 的区别：weight_decay 不进入梯度（不进一阶/二阶矩的自适应缩放），
    而是每步直接把参数乘上 (1 - lr*wd)，再套标准 Adam 更新。
    Adam 里的 L2 会被 sqrt(v) 归一化，导致大梯度参数实际衰减更小；AdamW 没有这个问题。
    """

    def __init__(self, model: MLP, lr: float = 1e-3,
                 beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8,
                 weight_decay: float = 0.01):
        if weight_decay is None or weight_decay < 0:
            raise ValueError(f"AdamW: weight_decay 必须 >= 0，收到 {weight_decay!r}")
        self.model = model
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = float(weight_decay)
        self.step_count = 0

    def step(self, all_grads):
        self.step_count += 1
        b1, b2, lr, eps = self.beta1, self.beta2, self.lr, self.eps
        decay = 1.0 - lr * self.weight_decay
        c1 = 1 - b1 ** self.step_count
        c2 = 1 - b2 ** self.step_count
        o1, o2 = 1 - b1, 1 - b2
        sqrt = math.sqrt

        for (layer, (grad_w, grad_b)) in zip(self.model.layers, all_grads):
            m_w, v_w, W = layer.m_w, layer.v_w, layer.weights
            m_b, v_b, B = layer.m_b, layer.v_b, layer.biases

            for j in range(layer.fan_out):
                g = grad_b[j]
                m = b1 * m_b[j] + o1 * g
                v = b2 * v_b[j] + o2 * g * g
                m_b[j] = m
                v_b[j] = v
                B[j] *= decay
                B[j] -= lr * (m / c1) / (sqrt(v / c2) + eps)

                gwj, mwj, vwj, Wj = grad_w[j], m_w[j], v_w[j], W[j]
                for i in range(layer.fan_in):
                    g = gwj[i]
                    m = b1 * mwj[i] + o1 * g
                    v = b2 * vwj[i] + o2 * g * g
                    mwj[i] = m
                    vwj[i] = v
                    Wj[i] *= decay
                    Wj[i] -= lr * (m / c1) / (sqrt(v / c2) + eps)


# ============================================================================
#  More Optimizers: SGD / SGD+Momentum / RMSprop
# ============================================================================

class SGD:
    """随机梯度下降（可带 Weight Decay）"""

    def __init__(self, model: MLP, lr: float = 0.01, weight_decay: float = 0.0):
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay

    def step(self, all_grads):
        lr, wd = self.lr, self.weight_decay
        for (layer, (grad_w, grad_b)) in zip(self.model.layers, all_grads):
            W, B = layer.weights, layer.biases
            for j in range(layer.fan_out):
                gb = grad_b[j]
                if wd > 0:
                    gb += wd * B[j]
                B[j] -= lr * gb
                gwj, Wj = grad_w[j], W[j]
                for i in range(layer.fan_in):
                    gw = gwj[i]
                    if wd > 0:
                        gw += wd * Wj[i]
                    Wj[i] -= lr * gw

    def zero_grad(self):
        pass


class SGDMomentum:
    """带动量的 SGD（SGD + Momentum）"""

    def __init__(self, model: MLP, lr: float = 0.01, momentum: float = 0.9):
        self.model = model
        self.lr = lr
        self.momentum = momentum
        self.v_w = [[[0.0] * layer.fan_in for _ in range(layer.fan_out)] for layer in model.layers]
        self.v_b = [[0.0] * layer.fan_out for layer in model.layers]

    def step(self, all_grads):
        lr, mu = self.lr, self.momentum
        for idx, (layer, (grad_w, grad_b)) in enumerate(zip(self.model.layers, all_grads)):
            vw, vb = self.v_w[idx], self.v_b[idx]
            W, B = layer.weights, layer.biases
            for j in range(layer.fan_out):
                v = mu * vb[j] + lr * grad_b[j]
                vb[j] = v
                B[j] -= v
                gwj, vwj, Wj = grad_w[j], vw[j], W[j]
                for i in range(layer.fan_in):
                    v = mu * vwj[i] + lr * gwj[i]
                    vwj[i] = v
                    Wj[i] -= v


class RMSprop:
    """RMSprop 优化器"""

    def __init__(self, model: MLP, lr: float = 0.01, alpha: float = 0.99, eps: float = 1e-8):
        self.model = model
        self.lr = lr
        self.alpha = alpha
        self.eps = eps
        self.s_w = [[[0.0] * layer.fan_in for _ in range(layer.fan_out)] for layer in model.layers]
        self.s_b = [[0.0] * layer.fan_out for layer in model.layers]

    def step(self, all_grads):
        lr, al, eps = self.lr, self.alpha, self.eps
        om = 1 - al
        sqrt = math.sqrt
        for idx, (layer, (grad_w, grad_b)) in enumerate(zip(self.model.layers, all_grads)):
            sw, sb = self.s_w[idx], self.s_b[idx]
            W, B = layer.weights, layer.biases
            for j in range(layer.fan_out):
                g = grad_b[j]
                s = al * sb[j] + om * g * g
                sb[j] = s
                B[j] -= lr * g / (sqrt(s) + eps)
                gwj, swj, Wj = grad_w[j], sw[j], W[j]
                for i in range(layer.fan_in):
                    g = gwj[i]
                    s = al * swj[i] + om * g * g
                    swj[i] = s
                    Wj[i] -= lr * g / (sqrt(s) + eps)




class Lion:
    """Lion 优化器（EvoLved Sign Momentum，Google 2023）

    只对「动量符号」做更新：``p -= lr * (sign(b1*m + (1-b1)*g) + wd*p)``，
    动量用 b2 更新。每步只存一份动量（Adam 存两份），显存减半；
    论文里在同等算力下常能追平 AdamW。

    注意：Lion 对 lr 比 Adam 敏感，通常取 AdamW 的 1/3 ~ 1/10。
    """

    def __init__(self, model: MLP, lr: float = 1e-4, beta1: float = 0.9,
                 beta2: float = 0.99, weight_decay: float = 0.0):
        if not 0.0 <= float(beta1) <= 1.0:
            raise ValueError(f"Lion: beta1 必须在 [0,1]，收到 {beta1!r}")
        if not 0.0 <= float(beta2) <= 1.0:
            raise ValueError(f"Lion: beta2 必须在 [0,1]，收到 {beta2!r}")
        if float(weight_decay) < 0.0:
            raise ValueError(f"Lion: weight_decay 必须 >= 0，收到 {weight_decay!r}")
        self.model = model
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.weight_decay = weight_decay
        self.step_count = 0
        self.m_w = [[[0.0] * layer.fan_in for _ in range(layer.fan_out)]
                    for layer in model.layers]
        self.m_b = [[0.0] * layer.fan_out for layer in model.layers]

    def step(self, all_grads):
        """all_grads: List[(grad_w, grad_b)]，与 Adam.step 同签名"""
        self.step_count += 1
        lr, b1, b2, wd = self.lr, self.beta1, self.beta2, self.weight_decay
        for idx, (layer, (grad_w, grad_b)) in enumerate(zip(self.model.layers, all_grads)):
            mw, mb = self.m_w[idx], self.m_b[idx]
            W, B = layer.weights, layer.biases
            for j in range(layer.fan_out):
                g = grad_b[j]
                c = b1 * mb[j] + (1.0 - b1) * g
                upd = (1.0 if c > 0.0 else (-1.0 if c < 0.0 else 0.0)) + wd * B[j]
                B[j] -= lr * upd
                mb[j] = b2 * mb[j] + (1.0 - b2) * g
                gwj, mwj, Wj = grad_w[j], mw[j], W[j]
                for i in range(layer.fan_in):
                    g = gwj[i]
                    c = b1 * mwj[i] + (1.0 - b1) * g
                    upd = (1.0 if c > 0.0 else (-1.0 if c < 0.0 else 0.0)) + wd * Wj[i]
                    Wj[i] -= lr * upd
                    mwj[i] = b2 * mwj[i] + (1.0 - b2) * g


class Adafactor:
    """Adafactor 优化器（Google 2018）：二阶矩「行列因子化」，显存 O(m+n)

    Adam 给每个元素存一份 v；对 (m,n) 权重矩阵，Adafactor 只存
    行均值 R(m) 与列均值 C(n)，近似 ``v_ij ≈ R_i * C_j / mean(R)``。
    更新量乘以相对步长 ``max(eps2, rms(g))``（论文的 relative step size），
    否则梯度很小时几乎不前进。

    没有一阶动量（省一份缓冲）——这是论文的默认配置；
    ``weight_decay`` 是解耦的（不乘进梯度）。
    """

    def __init__(self, model: MLP, lr: float = 1e-3, eps: float = 1e-30,
                 eps2: float = 1e-3, beta2: float = 0.999, weight_decay: float = 0.0):
        if float(eps) <= 0.0 or float(eps2) <= 0.0:
            raise ValueError("Adafactor: eps / eps2 必须 > 0")
        if not 0.0 <= float(beta2) < 1.0:
            raise ValueError(f"Adafactor: beta2 必须在 [0,1)，收到 {beta2!r}")
        if float(weight_decay) < 0.0:
            raise ValueError(f"Adafactor: weight_decay 必须 >= 0，收到 {weight_decay!r}")
        self.model = model
        self.lr = lr
        self.eps = eps
        self.eps2 = eps2
        self.beta2 = beta2
        self.weight_decay = weight_decay
        self.step_count = 0
        # R: 每个输出通道一行（对 fan_in 求均值）；C: 每个输入通道一列
        self.R = [[0.0] * layer.fan_out for layer in model.layers]
        self.C = [[0.0] * layer.fan_in for layer in model.layers]
        self.V_b = [[0.0] * layer.fan_out for layer in model.layers]

    def step(self, all_grads):
        self.step_count += 1
        t = self.step_count
        decay = 1.0 - (t ** -0.8)
        if decay < 0.0:
            decay = 0.0
        lr, eps, eps2, wd = self.lr, self.eps, self.eps2, self.weight_decay
        for idx, (layer, (grad_w, grad_b)) in enumerate(zip(self.model.layers, all_grads)):
            R, C, Vb = self.R[idx], self.C[idx], self.V_b[idx]
            W, B = layer.weights, layer.biases
            fo, fi = layer.fan_out, layer.fan_in
            om = 1.0 - decay
            # 1) 行列二阶矩（行均值 / 列均值）
            sq = 0.0
            for j in range(fo):
                row, gwj = R, grad_w[j]
                acc = 0.0
                for i in range(fi):
                    g = gwj[i]
                    acc += g * g
                    sq += g * g
                row[j] = decay * row[j] + om * (acc / fi)
            for i in range(fi):
                acc = 0.0
                for j in range(fo):
                    g = grad_w[j][i]
                    acc += g * g
                C[i] = decay * C[i] + om * (acc / fo)
            mean_R = sum(R) / fo if fo else 1.0
            if mean_R <= 0.0:
                mean_R = 1.0
            # 相对步长（论文的 relative step size）
            lr_t = lr * max(eps2, math.sqrt(sq / (fo * fi) if fo * fi else 0.0))
            # 2) 更新
            for j in range(fo):
                gwj, Wj = grad_w[j], W[j]
                for i in range(fi):
                    v = R[j] * C[i] / mean_R
                    # 解耦权重衰减必须用「更新前」的权重：写成单个表达式求值，
                    # 否则 wd 会作用在已经减过一次的值上（顺序错就多衰减一次）
                    Wj[i] = (Wj[i] - lr_t * gwj[i] / math.sqrt(v + eps)
                             - lr * wd * Wj[i])
                g = grad_b[j]
                Vb[j] = decay * Vb[j] + om * g * g
                B[j] = B[j] - lr_t * g / math.sqrt(Vb[j] + eps) - lr * wd * B[j]


# ============================================================================
#  Gradient Clipping / LR Scheduler
# ============================================================================

def clip_grad_by_norm(all_grads, max_norm: float):
    """按范数裁剪梯度，防止梯度爆炸（适用于 RNN/LSTM/GRU）"""
    if max_norm is None or max_norm <= 0:
        # 负阈值会把所有梯度乘上负数，等于整体翻转符号
        raise ValueError(f"clip_grad_by_norm: max_norm 必须 > 0，收到 {max_norm!r}")
    total_sq = 0.0
    for gw, gb in all_grads:
        for row in gw:
            for g in row:
                total_sq += g * g
        for g in gb:
            total_sq += g * g
    total_norm = math.sqrt(total_sq)
    if total_norm > max_norm:
        scale = max_norm / total_norm
        for gw, gb in all_grads:
            for j in range(len(gw)):
                for i in range(len(gw[j])):
                    gw[j][i] *= scale
                gb[j] *= scale
    return total_norm


def clip_grad_by_value(all_grads, clip_val: float):
    """按值裁剪梯度"""
    if clip_val is None or clip_val < 0:
        raise ValueError(f"clip_grad_by_value: clip_val 必须 >= 0，收到 {clip_val!r}")
    for gw, gb in all_grads:
        for j in range(len(gw)):
            for i in range(len(gw[j])):
                gw[j][i] = max(-clip_val, min(clip_val, gw[j][i]))
            gb[j] = max(-clip_val, min(clip_val, gb[j]))


class StepLR:
    """每 step_size 个 epoch 将 lr 乘以 gamma"""

    def __init__(self, optimizer, step_size: int = 30, gamma: float = 0.1):
        if step_size is None or step_size <= 0:
            raise ValueError(f"StepLR: step_size 必须 >= 1，收到 {step_size!r}")
        self.optimizer = optimizer
        self.step_size = step_size
        self.gamma = gamma
        self.initial_lr = optimizer.lr

    def step(self, epoch: int):
        if epoch > 0 and epoch % self.step_size == 0:
            self.optimizer.lr = self.initial_lr * (self.gamma ** (epoch // self.step_size))


class ExponentialLR:
    """每 epoch 将 lr 乘以 gamma^epoch"""

    def __init__(self, optimizer, gamma: float = 0.99):
        self.optimizer = optimizer
        self.gamma = gamma
        self.initial_lr = optimizer.lr

    def step(self, epoch: int):
        self.optimizer.lr = self.initial_lr * (self.gamma ** epoch)


class CosineAnnealingLR:
    """余弦退火学习率调度"""

    def __init__(self, optimizer, T_max: int, eta_min: float = 0.0):
        if T_max is None or T_max <= 0:
            raise ValueError(f"CosineAnnealingLR: T_max 必须 >= 1，收到 {T_max!r}")
        self.optimizer = optimizer
        self.T_max = T_max
        self.eta_min = eta_min
        self.initial_lr = optimizer.lr

    def step(self, epoch: int):
        # 支持周期重启：epoch 每 T_max 后重新开始余弦退火周期
        # 分母用 T_max-1，使每个周期**最后一轮**正好到达 eta_min；
        # 原先除以 T_max，最低只到 eta_min + 0.0488*(initial-eta_min)，永远够不着
        # epoch 从 1 开始计数（Model.fit 就是这么调的）；单独传 0/负数时按 1 处理，
        # 否则 ((-1) % T_max) 会绕到周期末尾，第一步就落到 eta_min
        epoch = max(int(epoch), 1)
        # T_max=1 时每一轮都是一个完整周期的末尾：直接落在 eta_min
        if self.T_max == 1:
            progress = 1.0
        else:
            progress = ((epoch - 1) % self.T_max) / (self.T_max - 1)
        self.optimizer.lr = self.eta_min + 0.5 * (self.initial_lr - self.eta_min) * (
            1 + math.cos(math.pi * progress))


class WarmupLR:
    """线性 warmup + 可选后续调度器（LLM 训练标配）

    前 warmup_epochs 轮把 lr 从 initial_lr*start_factor 线性升到 initial_lr，
    之后交给 base_scheduler（StepLR / ExponentialLR / CosineAnnealingLR），
    不传则保持 initial_lr。epoch 从 1 开始计数，与 Model.fit 一致。

    不 warmup 直接上大学习率时，注意力与 LayerNorm 的早期梯度很容易把模型打散；
    LLM 训练普遍用总步数的 1%~5% 做线性 warmup。
    """

    def __init__(self, optimizer, warmup_epochs: int = 5,
                 start_factor: float = 0.0, base_scheduler=None):
        if warmup_epochs is None or int(warmup_epochs) < 1:
            raise ValueError(f"WarmupLR: warmup_epochs 必须 >= 1，收到 {warmup_epochs!r}")
        if not 0.0 <= float(start_factor) <= 1.0:
            raise ValueError(f"WarmupLR: start_factor 必须在 [0,1]，收到 {start_factor!r}")
        self.optimizer = optimizer
        self.warmup_epochs = int(warmup_epochs)
        self.start_factor = float(start_factor)
        self.base_scheduler = base_scheduler
        self.initial_lr = optimizer.lr

    def step(self, epoch: int):
        epoch = max(int(epoch), 1)
        if epoch <= self.warmup_epochs:
            frac = epoch / self.warmup_epochs
            self.optimizer.lr = self.initial_lr * (
                self.start_factor + (1.0 - self.start_factor) * frac)
            return
        # warmup 结束：先回到基准 lr，再让后续调度器按绝对 epoch 接管
        self.optimizer.lr = self.initial_lr
        if self.base_scheduler is not None:
            self.base_scheduler.step(epoch)




class ReduceLROnPlateau:
    """按验证指标停滞降学习率：``lr *= factor``（PyTorch 同名调度器）

    step(metric)：指标在 ``patience`` 轮内没有改善（相对 ``threshold``）时把
    lr 乘 ``factor``，降 ``max_reductions`` 次后停在 ``min_lr``。
    mode='min' 表示指标越小越好（loss），'max' 表示越大越好（accuracy）。
    """

    def __init__(self, optimizer, factor: float = 0.5, patience: int = 10,
                 min_lr: float = 0.0, mode: str = "min", threshold: float = 1e-4,
                 max_reductions: int = None):
        if not 0.0 < float(factor) < 1.0:
            raise ValueError(f"ReduceLROnPlateau: factor 必须在 (0,1)，收到 {factor!r}")
        if mode not in ("min", "max"):
            raise ValueError(f"ReduceLROnPlateau: mode 只能是 'min' 或 'max'，收到 {mode!r}")
        if isinstance(patience, bool) or not isinstance(patience, int) or patience < 0:
            raise ValueError(f"ReduceLROnPlateau: patience 必须是 >=0 的整数，收到 {patience!r}")
        if float(min_lr) < 0.0:
            raise ValueError(f"ReduceLROnPlateau: min_lr 必须 >= 0，收到 {min_lr!r}")
        if max_reductions is not None and (isinstance(max_reductions, bool)
                                           or not isinstance(max_reductions, int)
                                           or max_reductions < 0):
            raise ValueError("ReduceLROnPlateau: max_reductions 必须是 None 或 >=0 的整数")
        self.optimizer = optimizer
        self.factor = float(factor)
        self.patience = int(patience)
        self.min_lr = float(min_lr)
        self.mode = mode
        self.threshold = float(threshold)
        self.max_reductions = max_reductions
        self.initial_lr = optimizer.lr
        self.best = None
        self.num_bad = 0
        self.reductions = 0

    def step(self, metric: float):
        """用这一轮的验证指标调用一次；返回当前 lr"""
        m = float(metric)
        if m != m:                      # NaN 指标不参与比较
            return self.optimizer.lr
        if self.best is None:
            self.best = m
            return self.optimizer.lr
        if self.mode == "min":
            better = m < self.best - self.threshold
        else:
            better = m > self.best + self.threshold
        if better:
            self.best = m
            self.num_bad = 0
            return self.optimizer.lr
        self.num_bad += 1
        if self.num_bad > self.patience:
            if self.max_reductions is not None and self.reductions >= self.max_reductions:
                return self.optimizer.lr
            self.optimizer.lr = max(self.min_lr, self.optimizer.lr * self.factor)
            self.num_bad = 0
            self.reductions += 1
        return self.optimizer.lr



# ============================================================================
#  Data Utilities
# ============================================================================

def _grads_norm(grads):
    """把 [(grad_w, grad_b), ...] 里所有元素的平方和开根号。

    求和顺序沿用 Model.fit 里原来内联的写法（层 → 行 → 列 → 偏置），
    这样把内联改成函数调用不会改变任何一位。
    """
    sq = 0.0
    for gw_l, gb_l in grads:
        for row in gw_l:
            for v in row:
                sq += v * v
        for v in gb_l:
            sq += v * v
    return math.sqrt(sq)


def estimate_flops(model, input_shape):
    """估算前向的计算量（纯标准库，不计显存/访存）。

    model:       MLP / Model / Sequential（支持 Layer、Conv1d、Conv2d、
                 ConvTranspose2d、MaxPool2d、AvgPool2d、GlobalAvgPool2d、
                 AdaptiveAvgPool2d、Flatten、BatchNorm1d、LayerNorm、GroupNorm、
                 InstanceNorm、Dropout、DropPath；遇到不认识的模块抛 TypeError，
                 而不是静默给个错数）
    input_shape: MLP 传 [fan_in]；二维卷积/池化/归一化传 (C, H, W)；
                 Conv1d 传 (C, L)；BatchNorm1d 也接受 (C, H, W)

    返回 dict：
        {'macs': 乘加次数, 'flops': 2*macs, 'params': 参数量,
         'output_shape': 输出形状, 'by_layer': [(名字, macs), ...]}

    注意：这是**解析估算**，不是实测。卷积按 im2col 后的稠密矩阵乘法计数，
    不做 winograd/FFT 之类的算法级优化假设。
    """
    if isinstance(model, Model):
        seq = model.model
    else:
        seq = model
    if isinstance(seq, MLP):
        layers = list(seq.layers)
    elif isinstance(seq, Sequential):
        layers = list(seq.modules)
        # 允许 Sequential([a, b]) 这种把列表当唯一 module 传进来的写法
        if len(layers) == 1 and isinstance(layers[0], (list, tuple)):
            layers = list(layers[0])
    elif hasattr(seq, 'layers'):
        layers = list(seq.layers)
    else:
        raise TypeError(f"estimate_flops: 不支持的模型类型 {type(seq).__name__}")

    shape = list(input_shape)
    if not shape:
        raise ValueError("estimate_flops: input_shape 不能为空")
    by_layer = []
    macs = 0
    params = 0

    def _dense(fan_in, fan_out):
        return fan_in * fan_out

    for m in layers:
        name = type(m).__name__
        if isinstance(m, Layer):
            fan_in, fan_out = m.fan_in, m.fan_out
            if len(shape) != 1 or shape[0] != fan_in:
                raise ValueError(
                    f"estimate_flops: {fan_in} 输入的 Layer 收到了形状 {shape}"
                )
            step = _dense(fan_in, fan_out)
            shape = [fan_out]
            params += fan_in * fan_out + fan_out
        elif isinstance(m, Conv2d):
            if len(shape) != 3:
                raise ValueError(f"estimate_flops: Conv2d 需要 (C, H, W)，收到 {shape}")
            c_in, h, w = shape
            if c_in != m.in_channels:
                raise ValueError(
                    f"estimate_flops: Conv2d 期望 {m.in_channels} 通道，输入是 {c_in}"
                )
            pad_h, pad_w = m.padding
            out_h = (h + 2 * pad_h - m.kh_eff) // m.sh + 1
            out_w = (w + 2 * pad_w - m.kw_eff) // m.sw + 1
            if out_h <= 0 or out_w <= 0:
                raise ValueError(
                    f"estimate_flops: 输入 {h}x{w} 经 kernel={m.kernel_size} "
                    f"stride={m.stride} padding={m.padding} 后输出尺寸非正"
                )
            # 分组卷积每个输出通道只连 cg = in_channels/groups 个输入通道，
            # 按 in_channels 计数会把 groups>1 的 MACs/参数量成倍高估
            step = m.out_channels * m.cg * m.kh * m.kw * out_h * out_w
            shape = [m.out_channels, out_h, out_w]
            params += m.out_channels * m.cg * m.kh * m.kw + m.out_channels
        elif isinstance(m, MaxPool2d):
            if len(shape) != 3:
                raise ValueError(f"estimate_flops: MaxPool2d 需要 (C, H, W)，收到 {shape}")
            c, h, w = shape
            out_h = (h - m.kh) // m.sh + 1
            out_w = (w - m.kw) // m.sw + 1
            step = c * m.kh * m.kw * out_h * out_w  # 比较次数
            shape = [c, out_h, out_w]
        elif isinstance(m, AvgPool2d):
            if len(shape) != 3:
                raise ValueError(f"estimate_flops: AvgPool2d 需要 (C, H, W)，收到 {shape}")
            c, h, w = shape
            out_h = (h - m.kh) // m.sh + 1
            out_w = (w - m.kw) // m.sw + 1
            step = c * out_h * out_w * (m.kh * m.kw)  # 求和 + 除
            shape = [c, out_h, out_w]
        elif isinstance(m, BatchNorm2d):
            if len(shape) != 3:
                raise ValueError(f"estimate_flops: BatchNorm2d 需要 (C, H, W)，收到 {shape}")
            c, h, w = shape
            if c != m.c:
                raise ValueError(
                    f"estimate_flops: BatchNorm2d 期望 {m.c} 通道，输入是 {c}")
            # 每元素约 4 次运算（减均值、乘 inv_std、乘 gamma、加 beta）+ 统计量
            step = 4 * c * h * w
            params += 2 * m.c
            shape = [c, h, w]
        elif isinstance(m, GlobalAvgPool2d):
            if len(shape) != 3:
                raise ValueError(
                    f"estimate_flops: GlobalAvgPool2d 需要 (C, H, W)，收到 {shape}"
                )
            c, h, w = shape
            step = c * h * w
            shape = [c]
        elif isinstance(m, Conv1d):
            if len(shape) != 2:
                raise ValueError(f"estimate_flops: Conv1d 需要 (C, L)，收到 {shape}")
            c_in, length = shape
            if c_in != m.in_channels:
                raise ValueError(
                    f"estimate_flops: Conv1d 期望 {m.in_channels} 通道，输入是 {c_in}"
                )
            out_l = (length + 2 * m.p - m.k) // m.s + 1
            if out_l <= 0:
                raise ValueError(
                    f"estimate_flops: 输入长度 {length} 经 kernel={m.k} "
                    f"stride={m.s} padding={m.p} 后输出长度非正"
                )
            step = m.out_channels * m.cg * m.k * out_l
            shape = [m.out_channels, out_l]
            params += m.out_channels * m.cg * m.k + m.out_channels
        elif isinstance(m, ConvTranspose2d):
            if len(shape) != 3:
                raise ValueError(
                    f"estimate_flops: ConvTranspose2d 需要 (C, H, W)，收到 {shape}"
                )
            c_in, h, w = shape
            if c_in != m.in_channels:
                raise ValueError(
                    f"estimate_flops: ConvTranspose2d 期望 {m.in_channels} 通道，"
                    f"输入是 {c_in}"
                )
            out_h = (h - 1) * m.sh - 2 * m.ph + m.kh + m.oph
            out_w = (w - 1) * m.sw - 2 * m.pw + m.kw + m.opw
            if out_h <= 0 or out_w <= 0:
                raise ValueError(
                    f"estimate_flops: 输入 {h}x{w} 经转置卷积后输出尺寸非正"
                )
            # 每个输入元素对每个 (本组输出通道, 核位置) 贡献一次乘加
            step = m.in_channels * m.og * m.kh * m.kw * h * w
            shape = [m.out_channels, out_h, out_w]
            params += m.in_channels * m.og * m.kh * m.kw + m.out_channels
        elif isinstance(m, AdaptiveAvgPool2d):
            if len(shape) != 3:
                raise ValueError(
                    f"estimate_flops: AdaptiveAvgPool2d 需要 (C, H, W)，收到 {shape}"
                )
            c, h, w = shape
            oh, ow = m.out_size
            step = c * h * w          # 每个输入元素恰好参与一次求均
            shape = [c, oh, ow]
        elif isinstance(m, (GroupNorm, InstanceNorm)):
            if len(shape) != 3:
                raise ValueError(
                    f"estimate_flops: {name} 需要 (C, H, W)，收到 {shape}"
                )
            c, h, w = shape
            if c != m.channels:
                raise ValueError(
                    f"estimate_flops: {name} 期望 {m.channels} 通道，输入是 {c}"
                )
            step = 4 * c * h * w      # 均值 + 方差 + 归一化 + 仿射
            params += 2 * m.channels
        elif isinstance(m, DropPath):
            step = 0
        elif isinstance(m, (BatchNorm1d, LayerNorm, Dropout, Flatten)):
            if isinstance(m, Flatten):
                # Flatten.forward 要求 (C,H,W)；1 维/2 维在这里静默通过、
                # 真跑却抛 TypeError，属于「估算说能跑、实际跑不了」，直接报错
                if len(shape) != 3:
                    raise ValueError(
                        f"estimate_flops: Flatten 需要三维输入 (C,H,W)，"
                        f"当前输入形状 {shape}（一维/二维请先补维度）"
                    )
                step = 0
                shape = [shape[0] * shape[1] * shape[2]]
            elif isinstance(m, Dropout):
                step = 0
            elif isinstance(m, LayerNorm):
                step = 2 * shape[0]
            else:  # BatchNorm1d：均值 + 方差 + 归一化 + 仿射，按 4 次元素操作计
                n = 1
                for d in shape:
                    n *= d
                step = 4 * max(n, shape[0])
        else:
            raise TypeError(
                f"estimate_flops: 不认识的模块 {name}（不做静默估算，请手动计数）"
            )
        macs += step
        by_layer.append((name, step))

    return {
        'macs': macs,
        'flops': 2 * macs,
        'params': params,
        'output_shape': list(shape),
        'by_layer': by_layer,
    }


def augment_noise(x, sigma: float = 0.01, rng=None):
    """加性高斯噪声（对样本内的每个数独立采样）。

    x:    单个样本（list）或一组样本（list[list]），原样维度返回，不修改入参
    sigma: 噪声标准差，必须 >= 0（0 表示拷贝一份，逐位相同）
    """
    if not isinstance(sigma, (int, float)) or isinstance(sigma, bool) or sigma < 0:
        raise ValueError(f"augment_noise: sigma 必须是非负数，收到 {sigma!r}")
    r = random if rng is None else rng

    def _one(vec):
        if sigma == 0:
            return [float(v) for v in vec]
        return [float(v) + r.gauss(0.0, sigma) for v in vec]

    if x and isinstance(x[0], (list, tuple)):
        return [_one(list(xi)) for xi in x]
    return _one(list(x))


def augment_image(x, flip: bool = True, max_shift: int = 1, rng=None):
    """图像样本的翻转/平移增强，输入 (C, H, W)，返回同形状的新样本。

    flip:      True 时以 50% 概率左右翻转（逐通道同步）
    max_shift: 水平/垂直各在 [-max_shift, max_shift] 内随机平移，越界处补 0
    """
    if not (isinstance(x, (list, tuple)) and len(x) >= 1 and x[0]
            and isinstance(x[0][0], (list, tuple))):
        raise ValueError("augment_image: 需要 (C, H, W) 的图像样本")
    if not isinstance(max_shift, int) or isinstance(max_shift, bool) or max_shift < 0:
        raise ValueError(f"augment_image: max_shift 必须是非负整数，收到 {max_shift!r}")
    r = random if rng is None else rng
    c, h, w = len(x), len(x[0]), len(x[0][0])
    for ch in x:
        if len(ch) != h or any(len(row) != w for row in ch):
            raise ValueError("augment_image: 各通道的 H/W 必须一致")

    do_flip = flip and r.random() < 0.5
    dx = r.randint(-max_shift, max_shift) if max_shift else 0
    dy = r.randint(-max_shift, max_shift) if max_shift else 0

    out = []
    for ch in x:
        new_ch = [[0.0] * w for _ in range(h)]
        for y in range(h):
            sy = y + dy
            if not 0 <= sy < h:
                continue
            src_row = ch[sy]
            dst_row = new_ch[y]
            for xx in range(w):
                sx = xx + dx
                if not 0 <= sx < w:
                    continue
                v = float(src_row[w - 1 - sx] if do_flip else src_row[sx])
                dst_row[xx] = v
        out.append(new_ch)
    return out


def _aug_image(x, who):
    """校验并返回 (C, H, W) 的副本（不修改入参）"""
    if not isinstance(x, (list, tuple)) or len(x) < 1:
        raise ValueError(f"{who}: 输入必须是 (C,H,W) 图像，收到 {type(x).__name__}")
    out = []
    h = None
    for c in x:
        if not isinstance(c, (list, tuple)) or not c:
            raise ValueError(f"{who}: 每个通道必须是非空二维数组")
        w = None
        chan = []
        for row in c:
            if not isinstance(row, (list, tuple)) or not row:
                raise ValueError(f"{who}: 每个通道必须是非空二维数组")
            if w is None:
                w = len(row)
            elif len(row) != w:
                raise ValueError(f"{who}: 同一通道内各行长度必须相同")
            chan.append([float(v) for v in row])
        if h is None:
            h = len(c)
        elif len(c) != h:
            raise ValueError(f"{who}: 各通道高度必须相同")
        out.append(chan)
    return out


def _beta_lam(alpha, rng, who):
    """用两次 Gamma 抽样合成 Beta(alpha, alpha) 的 lam（与 mixup 一致）"""
    alpha = float(alpha)
    if alpha <= 0.0:
        raise ValueError(f"{who}: alpha 必须 > 0，收到 {alpha!r}")

    def gamma(a):
        # Marsaglia-Tsang：a >= 1 用 squeeze；a < 1 用 x^(1/a) 提升
        if a < 1.0:
            u = rng.random()
            while u == 0.0:
                u = rng.random()
            return gamma(a + 1.0) * (u ** (1.0 / a))
        d = a - 1.0 / 3.0
        c = 1.0 / math.sqrt(9.0 * d)
        while True:
            x = rng.gauss(0.0, 1.0)
            v = (1.0 + c * x) ** 3
            if v <= 0.0:
                continue
            u = rng.random()
            if u < 1.0 - 0.0331 * x ** 4:
                return d * v
            if math.log(u) < 0.5 * x * x + d * (1.0 - v + math.log(v)):
                return d * v

    g1 = gamma(alpha)
    g2 = gamma(alpha)
    return g1 / (g1 + g2)


def augment_cutmix(x1, y1, x2, y2, alpha: float = 1.0, rng=None):
    """CutMix：从 x2 剪一块矩形贴到 x1 上，标签按面积比例混合

    x1/x2: (C,H,W)；y1/y2: 标签（one-hot 或任意等长向量）。
    返回 (image, label)，label 是「按面积加权的混合标签」，和仍为 1（输入是 one-hot 时）。
    """
    who = "augment_cutmix"
    a = _aug_image(x1, who)
    b = _aug_image(x2, who)
    if len(a) != len(b) or len(a[0]) != len(b[0]) or len(a[0][0]) != len(b[0][0]):
        raise ValueError(f"{who}: 两张图的形状必须相同")
    if len(y1) != len(y2) or not y1:
        raise ValueError(f"{who}: 两个标签长度必须相同且非空")
    rng = rng or random
    lam = _beta_lam(alpha, rng, who)

    C, H, W = len(a), len(a[0]), len(a[0][0])
    cut_w = int(round(W * math.sqrt(1.0 - lam)))
    cut_h = int(round(H * math.sqrt(1.0 - lam)))
    if cut_w > 0 and cut_h > 0:
        cx = int(rng.randrange(W))
        cy = int(rng.randrange(H))
        x0, x1_ = max(0, cx - cut_w // 2), min(W, cx + cut_w // 2)
        y0, y1_ = max(0, cy - cut_h // 2), min(H, cy + cut_h // 2)
        for c in range(C):
            for yy in range(y0, y1_):
                for xx in range(x0, x1_):
                    a[c][yy][xx] = b[c][yy][xx]
        lam = 1.0 - ((x1_ - x0) * (y1_ - y0)) / float(H * W)
    else:
        lam = 1.0

    label = [lam * float(u) + (1.0 - lam) * float(v) for u, v in zip(y1, y2)]
    return a, label


def augment_cutout(x, num_holes: int = 1, hole_size: float = 0.25, fill: float = 0.0,
                   rng=None):
    """Cutout：随机抹掉 num_holes 个正方形区域（填 fill）

    hole_size 是相对短边的比例，(0,1)。返回新图像，不修改入参。
    """
    who = "augment_cutout"
    img = _aug_image(x, who)
    if isinstance(num_holes, bool) or not isinstance(num_holes, int) or num_holes < 0:
        raise ValueError(f"{who}: num_holes 必须是 >= 0 的整数，收到 {num_holes!r}")
    hole_size = float(hole_size)
    if not 0.0 < hole_size < 1.0:
        raise ValueError(f"{who}: hole_size 必须在 (0,1)，收到 {hole_size!r}")
    rng = rng or random
    C, H, W = len(img), len(img[0]), len(img[0][0])
    side = max(1, int(round(min(H, W) * hole_size)))
    for _ in range(num_holes):
        x0 = int(rng.randrange(max(1, W - side + 1)))
        y0 = int(rng.randrange(max(1, H - side + 1)))
        for c in range(C):
            for yy in range(y0, min(H, y0 + side)):
                for xx in range(x0, min(W, x0 + side)):
                    img[c][yy][xx] = float(fill)
    return img


def _aug_brightness(img, factor):
    return [[[min(1.0, max(0.0, v * factor)) for v in row] for row in ch] for ch in img]


def _aug_contrast(img, factor):
    n = sum(sum(v for v in row) for ch in img for row in ch)
    cnt = sum(len(row) for ch in img for row in ch)
    mean = n / cnt
    return [[[min(1.0, max(0.0, (v - mean) * factor + mean)) for v in row] for row in ch]
            for ch in img]


def _aug_shift(img, dy, dx):
    C, H, W = len(img), len(img[0]), len(img[0][0])
    out = []
    for c in range(C):
        chan = [[0.0] * W for _ in range(H)]
        for y in range(H):
            sy = y - dy
            if not 0 <= sy < H:
                continue
            for x in range(W):
                sx = x - dx
                if 0 <= sx < W:
                    chan[y][x] = img[c][sy][sx]
        out.append(chan)
    return out


def _aug_rotate90(img, turns):
    out = img
    for _ in range(turns % 4):
        C, H, W = len(out), len(out[0]), len(out[0][0])
        out = [[[out[c][H - 1 - x][y] for x in range(H)] for y in range(W)]
               for c in range(C)]
    return out


def augment_randaugment(x, n: int = 2, magnitude: float = 9.0, ops=None, rng=None):
    """RandAugment：随机挑 n 个操作（可重复）依次应用

    默认操作集：brightness / contrast / translate / rotate90 / identity。
    magnitude ∈ [0, 10]，线性映射到各操作强度（1 = 弱，10 = 强）。
    返回 (image, ops_applied) —— 应用了哪些操作，便于复现和记录。
    """
    who = "augment_randaugment"
    img = _aug_image(x, who)
    if isinstance(n, bool) or not isinstance(n, int) or n < 0:
        raise ValueError(f"{who}: n 必须是 >= 0 的整数，收到 {n!r}")
    magnitude = float(magnitude)
    if not 0.0 <= magnitude <= 10.0:
        raise ValueError(f"{who}: magnitude 必须在 [0,10]，收到 {magnitude!r}")
    rng = rng or random
    pool = list(ops) if ops else ["brightness", "contrast", "translate", "rotate90", "identity"]
    for op in pool:
        if op not in ("brightness", "contrast", "translate", "rotate90", "identity"):
            raise ValueError(f"{who}: 不支持的操作 {op!r}")

    scale = magnitude / 10.0
    applied = []
    for _ in range(n):
        op = pool[int(rng.randrange(len(pool)))]
        applied.append(op)
        if op == "brightness":
            img = _aug_brightness(img, 1.0 - 0.5 * scale + scale * rng.random())
        elif op == "contrast":
            img = _aug_contrast(img, 1.0 - 0.5 * scale + scale * rng.random())
        elif op == "translate":
            lim = max(1, int(round(scale * 0.25 * min(len(img[0]), len(img[0][0])))))
            img = _aug_shift(img, int(rng.randint(-lim, lim)), int(rng.randint(-lim, lim)))
        elif op == "rotate90":
            img = _aug_rotate90(img, int(rng.randrange(4)))
    return img, applied


def mixup(x1, y1, x2, y2, alpha: float = 0.2, rng=None):
    """Mixup：返回 (混合后的 x, 混合后的软标签向量)。

    lam ~ Beta(alpha, alpha)（alpha<=0 时取 lam=0.5，即固定中点）；
    y1/y2 必须是**类别索引**，返回的标签是 lam*y1 + (1-lam)*y2 的概率向量，
    可直接喂给 cross_entropy_loss 的软标签分支。
    """
    if not isinstance(alpha, (int, float)) or isinstance(alpha, bool):
        raise ValueError(f"mixup: alpha 必须是数字，收到 {alpha!r}")
    if not isinstance(y1, int) or isinstance(y1, bool) or not isinstance(y2, int) or isinstance(y2, bool):
        raise ValueError("mixup: y1/y2 必须是类别索引")
    if len(x1) != len(x2):
        raise ValueError(f"mixup: 两个样本维度不同（{len(x1)} vs {len(x2)}）")
    n_cls = max(y1, y2) + 1
    r = random if rng is None else rng
    if alpha > 0:
        # Beta(a, a) 用两个 Gamma(a) 采样相除；标准库没有 gamma，用 Marsaglia-Tsang
        def _gamma(a):
            if a < 1.0:
                return _gamma(a + 1.0) * (r.random() ** (1.0 / a))
            d = a - 1.0 / 3.0
            c = 1.0 / math.sqrt(9.0 * d)
            while True:
                while True:
                    v = r.gauss(0.0, 1.0)
                    if v > -1.0 / c:
                        break
                u = r.random()
                x = (1.0 + c * v) ** 3
                if math.log(u) < 0.5 * v * v + d - d * x + d * math.log(x):
                    return d * x
        # Beta(a, a)：用两个独立 Gamma(a) 采样相除；分母必须包含分子那一项，
        # 原来写成 _gamma(a)/(_gamma(a)+_gamma(a)) 漏了分子，实测 lam 会超过 1
        g1 = _gamma(alpha)
        g2 = _gamma(alpha)
        lam = g1 / (g1 + g2)
    else:
        lam = 0.5
    mixed_x = [lam * float(a) + (1.0 - lam) * float(b) for a, b in zip(x1, x2)]
    soft = [0.0] * n_cls
    soft[y1] += lam
    soft[y2] += 1.0 - lam
    return mixed_x, soft


def split_data(data, val_ratio: float = 0.2, test_ratio: float = 0.1, seed: int = 42,
               stratify=None):
    """将数据拆分为 train / val / test

    stratify: 与 data 等长的标签序列（如类别 id）。给出时按标签分层抽样，
        使各类别在 train/val/test 中的比例尽量与整体一致（类别极不均衡时很有用）。
    """
    if not (0.0 <= val_ratio < 1.0) or not (0.0 <= test_ratio < 1.0) \
            or val_ratio + test_ratio >= 1.0:
        raise ValueError(
            f"split_data: 比例非法（val_ratio={val_ratio}, test_ratio={test_ratio}）；"
            f"要求各自在 [0,1) 内且之和 < 1"
        )
    # 用局部 Random 实例，避免 random.seed() 污染全局随机状态
    rng = random.Random(seed)
    items = list(data)
    if stratify is not None:
        if len(stratify) != len(items):
            raise ValueError(
                f"split_data: stratify 长度 {len(stratify)} 与数据量 {len(items)} 不一致"
            )
        groups = {}
        for idx, label in enumerate(stratify):
            groups.setdefault(label, []).append(idx)
        test_idx, val_idx, train_idx = [], [], []
        for label, idxs in groups.items():
            rng.shuffle(idxs)
            k = len(idxs)
            n_t = int(k * test_ratio)
            n_v = int(k * val_ratio)
            test_idx.extend(idxs[:n_t])
            val_idx.extend(idxs[n_t:n_t + n_v])
            train_idx.extend(idxs[n_t + n_v:])
        return ([items[i] for i in train_idx], [items[i] for i in val_idx],
                [items[i] for i in test_idx])
    rng.shuffle(items)

    n = len(items)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)

    test_set = items[:n_test]
    val_set = items[n_test:n_test + n_val]
    train_set = items[n_test + n_val:]

    return train_set, val_set, test_set


class DataLoader:
    """简易数据加载器：按 batch 迭代，支持 shuffle"""

    def __init__(self, dataset: list, batch_size: int = 32, shuffle: bool = True,
                 seed=None, collate_fn=None):
        if batch_size is None or batch_size <= 0:
            raise ValueError(f"DataLoader: batch_size 必须 >= 1，收到 {batch_size!r}")
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        # 用局部 RNG：全局 random.shuffle 会推进调用方的随机状态，破坏可复现性
        self._rng = random.Random(seed)
        # collate_fn(batch) -> batch：自定义批次拼装（padding、转张量等）
        self.collate_fn = collate_fn

    def __iter__(self):
        items = list(self.dataset)
        if self.shuffle:
            self._rng.shuffle(items)
        for b in range(0, len(items), self.batch_size):
            batch = items[b:b + self.batch_size]
            if self.collate_fn is not None:
                batch = self.collate_fn(batch)
            yield batch

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


def k_fold_indices(n_samples: int, n_splits: int = 5, shuffle: bool = True,
                   seed: int = 42, stratify=None):
    """生成 K 折交叉验证的 (train_idx, val_idx) 列表（纯索引，不复制数据）。

    n_samples: 样本总数
    n_splits:  折数（>=2）
    stratify:  与样本等长的标签序列；给出时每折按类别分层
    """
    n_samples = int(n_samples)
    n_splits = int(n_splits)
    if n_samples <= 0:
        raise ValueError(f"k_fold_indices: n_samples 必须 > 0，收到 {n_samples}")
    if n_splits < 2:
        raise ValueError(f"k_fold_indices: n_splits 必须 >= 2，收到 {n_splits}")
    if n_splits > n_samples:
        raise ValueError(
            f"k_fold_indices: n_splits={n_splits} 超过样本数 {n_samples}"
        )
    rng = random.Random(seed)
    if stratify is not None:
        if len(stratify) != n_samples:
            raise ValueError(
                f"k_fold_indices: stratify 长度 {len(stratify)} 与样本数 {n_samples} 不一致"
            )
        groups = {}
        for idx, label in enumerate(stratify):
            groups.setdefault(label, []).append(idx)
        folds = [[] for _ in range(n_splits)]
        for label, idxs in groups.items():
            rng.shuffle(idxs)
            for i, idx in enumerate(idxs):
                folds[i % n_splits].append(idx)
    else:
        order = list(range(n_samples))
        if shuffle:
            rng.shuffle(order)
        folds = [[] for _ in range(n_splits)]
        for i, idx in enumerate(order):
            folds[i % n_splits].append(idx)
    out = []
    for k in range(n_splits):
        val_idx = sorted(folds[k])
        val_set = set(val_idx)
        train_idx = [i for i in range(n_samples) if i not in val_set]
        out.append((train_idx, val_idx))
    return out


class StandardScaler:
    """逐特征标准化：(x - mean) / std（std 为 0 时按 1 处理，不做缩放）"""

    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def _check(self, x):
        if not isinstance(x, (list, tuple)) or not x:
            raise ValueError("StandardScaler: 输入必须是非空的二维列表")
        width = len(x[0])
        if width == 0:
            raise ValueError("StandardScaler: 样本特征维度不能为 0")
        for row in x:
            if len(row) != width:
                raise ValueError(
                    f"StandardScaler: 各样本维度不一致（{len(row)} != {width}）"
                )
        return width

    def fit(self, x):
        width = self._check(x)
        n = len(x)
        self.mean_ = [sum(row[j] for row in x) / n for j in range(width)]
        self.std_ = []
        for j in range(width):
            var = sum((row[j] - self.mean_[j]) ** 2 for row in x) / n
            s = math.sqrt(var)
            self.std_.append(s if s > 0.0 else 1.0)
        return self

    def transform(self, x):
        if self.mean_ is None:
            raise ValueError("StandardScaler: 请先 fit")
        self._check(x)
        return [[(row[j] - self.mean_[j]) / self.std_[j]
                 for j in range(len(self.mean_))] for row in x]

    def fit_transform(self, x):
        return self.fit(x).transform(x)

    def inverse_transform(self, x):
        if self.mean_ is None:
            raise ValueError("StandardScaler: 请先 fit")
        self._check(x)
        return [[row[j] * self.std_[j] + self.mean_[j]
                 for j in range(len(self.mean_))] for row in x]


class MinMaxScaler:
    """逐特征缩放到 [feature_range]（默认 [0, 1]；极差为 0 时映射到下界）"""

    def __init__(self, feature_range=(0.0, 1.0)):
        lo, hi = feature_range
        if not (lo < hi):
            raise ValueError(
                f"MinMaxScaler: feature_range 要求 lo < hi，收到 {feature_range!r}"
            )
        self.range_ = (float(lo), float(hi))
        self.data_min_ = None
        self.data_max_ = None

    def _check(self, x):
        if not isinstance(x, (list, tuple)) or not x:
            raise ValueError("MinMaxScaler: 输入必须是非空的二维列表")
        width = len(x[0])
        if width == 0:
            raise ValueError("MinMaxScaler: 样本特征维度不能为 0")
        for row in x:
            if len(row) != width:
                raise ValueError(
                    f"MinMaxScaler: 各样本维度不一致（{len(row)} != {width}）"
                )
        return width

    def fit(self, x):
        width = self._check(x)
        self.data_min_ = [min(row[j] for row in x) for j in range(width)]
        self.data_max_ = [max(row[j] for row in x) for j in range(width)]
        return self

    def transform(self, x):
        if self.data_min_ is None:
            raise ValueError("MinMaxScaler: 请先 fit")
        self._check(x)
        lo, hi = self.range_
        out = []
        for row in x:
            new_row = []
            for j in range(len(self.data_min_)):
                span = self.data_max_[j] - self.data_min_[j]
                if span > 0.0:
                    new_row.append(lo + (row[j] - self.data_min_[j]) / span * (hi - lo))
                else:
                    new_row.append(lo)
            out.append(new_row)
        return out

    def fit_transform(self, x):
        return self.fit(x).transform(x)

    def inverse_transform(self, x):
        if self.data_min_ is None:
            raise ValueError("MinMaxScaler: 请先 fit")
        self._check(x)
        lo, hi = self.range_
        out = []
        for row in x:
            new_row = []
            for j in range(len(self.data_min_)):
                span = self.data_max_[j] - self.data_min_[j]
                if span > 0.0 and hi > lo:
                    new_row.append(
                        self.data_min_[j] + (row[j] - lo) / (hi - lo) * span
                    )
                else:
                    new_row.append(self.data_min_[j])
            out.append(new_row)
        return out


def ascii_plot(history, keys=("train_loss", "val_loss"), width: int = 60,
               height: int = 12, title: str = None):
    """把 train_history 画成纯 ASCII 折线图（终端里直接看训练曲线）。

    history: Model.fit 填的 {'train_loss': [(epoch, loss)...], ...}
    keys:    要画的曲线名
    """
    if not isinstance(history, dict):
        raise ValueError("ascii_plot: history 必须是 dict")
    width = max(10, int(width))
    height = max(3, int(height))
    series = []
    for k in keys:
        pts = history.get(k)
        if not pts:
            continue
        clean = [(float(e), float(v)) for e, v in pts if v is not None]
        if clean:
            series.append((k, clean))
    if not series:
        return "(ascii_plot: 没有可画的数据)"
    all_vals = [v for _, pts in series for _, v in pts]
    lo, hi = min(all_vals), max(all_vals)
    if hi <= lo:
        hi = lo + 1.0
    all_epochs = [e for _, pts in series for e, _ in pts]
    e_lo, e_hi = min(all_epochs), max(all_epochs)
    span_e = (e_hi - e_lo) or 1.0
    marks = [".", "o", "*", "+", "x"]
    canvas = [[" "] * width for _ in range(height)]
    for si, (name, pts) in enumerate(series):
        ch = marks[si % len(marks)]
        for e, v in pts:
            cx = int((e - e_lo) / span_e * (width - 1))
            cy = height - 1 - int((v - lo) / (hi - lo) * (height - 1))
            canvas[cy][cx] = ch
    lines = []
    if title:
        lines.append(str(title))
    for r in range(height):
        v = hi - (hi - lo) * r / (height - 1)
        lines.append(f"{v:10.4f} |" + "".join(canvas[r]))
    lines.append(" " * 10 + " +" + "-" * width)
    lines.append(" " * 10 + f"  epoch {e_lo:g} .. {e_hi:g}")
    legend = "  ".join(f"{marks[i % len(marks)]}={name}"
                       for i, (name, _) in enumerate(series))
    lines.append("  " + legend)
    return "\n".join(lines)


# ============================================================================
#  Text Processing（分词与文本工具）
# ============================================================================

def _bpe_pretokenize(text):
    """把文本切成词块：字母数字连续段 / 空白连续段 / 其它符号连续段

    只在词块内部学习合并，避免跨单词边界生成 「the  cat」这类没用的 token
    （GPT-2 的做法，规则简化到纯标准库能表达的程度）。
    """
    out, cur, kind = [], [], None
    for ch in text:
        if ch.isalnum():
            k = "w"
        elif ch.isspace():
            k = "s"
        else:
            k = "p"
        if k != kind:
            if cur:
                out.append("".join(cur))
            cur, kind = [ch], k
        else:
            cur.append(ch)
    if cur:
        out.append("".join(cur))
    return out


def _bpe_apply_merge(tokens, pair, new_id):
    """把 tokens 里所有不重叠的 pair 换成 new_id（从左往右）"""
    out = []
    i = 0
    n = len(tokens)
    while i < n:
        if i + 1 < n and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(tokens[i])
            i += 1
    return out


class BytePairTokenizer:
    """字节级 BPE 分词器：任意文本都能无损往返，词表从 256 个字节值开始长

    为什么用**字节**而不是字符：任何 UTF-8 文本（中文、emoji、二进制脏数据）
    都能被 256 个基础 id 覆盖，不会出现「词表外字符」。合并规则在词块内部学习，
    所以 `encode` / `decode` 都可复现、可存档。

    用法::

        tok = BytePairTokenizer().train(text, vocab_size=512)
        ids = tok.encode("hello 世界")
        assert tok.decode(ids) == "hello 世界"
        tok.save("bpe.json")
    """

    def __init__(self, merges=None):
        if merges is not None and not isinstance(merges, (list, tuple)):
            raise ValueError(
                f"BytePairTokenizer: merges 必须是列表，收到 {type(merges).__name__}")
        try:
            self.merges = [tuple(int(x) for x in m) for m in (merges or [])]
        except (TypeError, ValueError) as e:
            raise ValueError(f"BytePairTokenizer: 合并规则必须是 (a, b) 整数对（{e}）")
        for k, m in enumerate(self.merges):
            limit = 256 + k          # 只能引用「之前已经定义」的 id
            if len(m) != 2 or not all(0 <= v < limit for v in m):
                raise ValueError(
                    f"BytePairTokenizer: 第 {k} 条合并规则 {m!r} 非法"
                    f"（只能引用已定义的 id，范围 [0,{limit})）")
        self._rebuild()

    def _rebuild(self):
        """按 merges 重建 id → 字节串 / 合并优先级表"""
        self.id_to_bytes = {i: bytes([i]) for i in range(256)}
        self.merge_rank = {}
        for k, (a, b) in enumerate(self.merges):
            self.id_to_bytes[256 + k] = self.id_to_bytes[a] + self.id_to_bytes[b]
            self.merge_rank[(a, b)] = k
        self.vocab_size = 256 + len(self.merges)

    # ------------------------------------------------------------------ 训练
    def train(self, text, vocab_size: int = 512, min_frequency: int = 2,
              verbose: bool = False):
        """在 text 上学习合并规则，直到词表达 vocab_size（或没有可合并的片段）

        min_frequency: 出现次数少于它的片段不再合并，避免为一次性片段浪费词表。
        同样的输入永远得到同样的 merges（按「出现次数降序、id 升序」挑下一个）。
        """
        if not isinstance(text, str):
            raise ValueError(f"BytePairTokenizer.train: text 必须是字符串，收到 {type(text).__name__}")
        if isinstance(vocab_size, bool) or not isinstance(vocab_size, int) or vocab_size < 256:
            raise ValueError(
                f"BytePairTokenizer.train: vocab_size 必须 >= 256（字节基数），收到 {vocab_size!r}")
        if isinstance(min_frequency, bool) or not isinstance(min_frequency, int) or min_frequency < 1:
            raise ValueError(
                f"BytePairTokenizer.train: min_frequency 必须是 >= 1 的整数，收到 {min_frequency!r}")
        words = [list(w.encode("utf-8")) for w in _bpe_pretokenize(text)]
        self.merges = []
        self._rebuild()
        target = vocab_size - 256
        for step in range(target):
            counts = {}
            for w in words:
                for i in range(len(w) - 1):
                    pair = (w[i], w[i + 1])
                    counts[pair] = counts.get(pair, 0) + 1
            if not counts:
                break
            pair, cnt = max(counts.items(), key=lambda kv: (kv[1], -kv[0][0], -kv[0][1]))
            if cnt < min_frequency:
                break
            self.merges.append(pair)
            self._rebuild()
            new_id = 255 + len(self.merges)
            words = [_bpe_apply_merge(w, pair, new_id) for w in words]
            if verbose:
                piece = self.id_to_bytes[new_id].decode("utf-8", "replace")
                print(f"merge {step + 1}/{target}: {pair} -> {new_id} ({piece!r}) x{cnt}")
        return self

    # ------------------------------------------------------------------ 编解码
    def encode(self, text: str) -> list:
        """文本 → id 列表（按学习到的优先级从高到低贪心合并）"""
        if not isinstance(text, str):
            raise ValueError(f"BytePairTokenizer.encode: text 必须是字符串，收到 {type(text).__name__}")
        ids = []
        for word in _bpe_pretokenize(text):
            toks = list(word.encode("utf-8"))
            while True:
                best_rank, best_i = None, -1
                for i in range(len(toks) - 1):
                    r = self.merge_rank.get((toks[i], toks[i + 1]))
                    if r is not None and (best_rank is None or r < best_rank):
                        best_rank, best_i = r, i
                if best_rank is None:
                    break
                toks = (toks[:best_i] + [256 + best_rank] + toks[best_i + 2:])
            ids.extend(toks)
        return ids

    def decode(self, ids) -> str:
        """id 列表 → 文本；拼出来的字节不是合法 UTF-8 时报错（而不是静默替换）"""
        chunks = []
        for i in ids:
            if isinstance(i, bool) or not isinstance(i, int) or i not in self.id_to_bytes:
                raise ValueError(
                    f"BytePairTokenizer.decode: 非法 id {i!r}（词表 {self.vocab_size}）")
            chunks.append(self.id_to_bytes[i])
        try:
            return b"".join(chunks).decode("utf-8")
        except UnicodeDecodeError:
            raise ValueError(
                "BytePairTokenizer.decode: 这些 id 拼起来不是合法 UTF-8"
                "（可能被截断，或来自另一个分词器）")

    def encode_corpus(self, text: str, seq_len: int = None) -> list:
        """把整段文本编码成 CausalLM.fit 可直接用的序列

        seq_len=None 时返回一条长序列；给定时切成不重叠的块（最后一块可能更短，
        长度 < 2 的块会被丢掉，因为语言模型至少要 2 个 token 才有预测目标）。
        """
        ids = self.encode(text)
        if seq_len is None:
            return [ids]
        if isinstance(seq_len, bool) or not isinstance(seq_len, int) or seq_len < 2:
            raise ValueError(
                f"BytePairTokenizer.encode_corpus: seq_len 必须 >= 2，收到 {seq_len!r}")
        return [ids[i:i + seq_len] for i in range(0, len(ids), seq_len)
                if len(ids[i:i + seq_len]) >= 2]

    # ------------------------------------------------------------------ 存档
    def save(self, filepath: str, quiet: bool = False):
        """把合并规则写成 JSON（纯标准库）"""
        if not isinstance(filepath, str) or not filepath:
            raise ValueError(f"BytePairTokenizer.save: filepath 必须是非空字符串，收到 {filepath!r}")
        data = {"format": "ai_bpe_v1", "merges": [list(m) for m in self.merges]}
        with io.open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        if not quiet:
            print(f"已保存分词器: {filepath}（词表 {self.vocab_size}）")
        return filepath

    @staticmethod
    def load(filepath: str):
        """从 `save` 写出的 JSON 恢复分词器"""
        with io.open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict) or data.get("format") != "ai_bpe_v1":
            raise ValueError(f"BytePairTokenizer.load: {filepath} 不是 ai_bpe_v1 存档")
        merges = data.get("merges")
        if not isinstance(merges, list):
            raise ValueError("BytePairTokenizer.load: 存档缺少 merges 数组")
        return BytePairTokenizer(merges)

    def __len__(self):
        return self.vocab_size

    def __repr__(self):
        return f"BytePairTokenizer(vocab_size={self.vocab_size})"


_WP_SPECIALS = ("[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]")
_WP_CONT = "##"


class WordPieceTokenizer:
    """WordPiece 分词器（BERT 风格）：词表 = 特殊标记 + 字符 + 子词。

    与 `BytePairTokenizer` 的区别：
      - 词表里存**字符串**（可读、可手写、可存档对比），不是字节序列；
      - 词内非首字符一律带 `##` 前缀，所以「词中间的子词」和「词首的子词」
        是两个不同 token —— 这也是 BERT 能把 `playing` 拆成 `play + ##ing`
        却不把 `ing` 当成独立词的原因；
      - 训练按 `频次 / (左频次 × 右频次)` 打分（BERT 原始做法），高频组合优先，
        避免 `the` 这类高频词把词表占满；
      - 编码用「最长前缀匹配」，词表外字符落到 `[UNK]`。

    词表顺序固定：`[PAD] [UNK] [CLS] [SEP] [MASK]` 在最前，`id` 即下标，
    存档/加载后 id 与训练时完全一致。
    """

    def __init__(self, vocab=None):
        if vocab is None:
            vocab = list(_WP_SPECIALS)
        self._install(vocab)

    def _install(self, vocab):
        if not isinstance(vocab, (list, tuple)):
            raise TypeError("WordPieceTokenizer: vocab 必须是 list 或 tuple")
        tokens = [self._check_token(t) for t in vocab]
        if len(set(tokens)) != len(tokens):
            raise ValueError("WordPieceTokenizer: vocab 里有重复 token")
        for name in _WP_SPECIALS:
            if name not in tokens:
                raise ValueError(f"WordPieceTokenizer: vocab 缺少特殊标记 {name}")
        self.tokens = list(tokens)
        self.token_to_id = {t: k for k, t in enumerate(self.tokens)}
        self.unk_id = self.token_to_id["[UNK]"]
        self.pad_id = self.token_to_id["[PAD]"]
        self.cls_id = self.token_to_id["[CLS]"]
        self.sep_id = self.token_to_id["[SEP]"]
        self.mask_id = self.token_to_id["[MASK]"]
        return self

    @staticmethod
    def _check_token(token):
        if not isinstance(token, str):
            raise TypeError(f"WordPieceTokenizer: token 必须是 str，收到 {type(token).__name__}")
        if not token:
            raise ValueError("WordPieceTokenizer: token 不能是空字符串")
        if token.startswith(_WP_CONT) and token == _WP_CONT:
            raise ValueError(f"WordPieceTokenizer: token {token!r} 只有前缀、没有内容")
        return token

    @property
    def vocab_size(self):
        return len(self.tokens)

    def __len__(self):
        return len(self.tokens)

    def __repr__(self):
        return f"WordPieceTokenizer(vocab_size={self.vocab_size})"

    # ---------- 切词 ----------
    @staticmethod
    def _split_words(text):
        """切成词块：字母数字连写 / 空白连写 / 其它符号连写。"""
        words = []
        cur = ""
        kind = None
        for ch in text:
            k = "w" if ch.isalnum() else ("s" if ch.isspace() else "o")
            if k == kind:
                cur += ch
            else:
                if cur:
                    words.append(cur)
                cur = ch
                kind = k
        if cur:
            words.append(cur)
        return words

    @staticmethod
    def _to_pieces(word):
        """词 -> WordPiece 初始切片：首字符原样，其余加 ##。"""
        pieces = []
        for k, ch in enumerate(word):
            pieces.append(ch if k == 0 else _WP_CONT + ch)
        return pieces

    # ---------- 训练 ----------
    def train(self, text, vocab_size=512, min_frequency=2, verbose=False):
        if not isinstance(text, str):
            raise TypeError("WordPieceTokenizer.train: text 必须是 str")
        if isinstance(vocab_size, bool) or not isinstance(vocab_size, int):
            raise TypeError("WordPieceTokenizer.train: vocab_size 必须是 int")
        if vocab_size < len(_WP_SPECIALS) + 1:
            raise ValueError(
                f"WordPieceTokenizer.train: vocab_size 至少 {len(_WP_SPECIALS) + 1}")
        if isinstance(min_frequency, bool) or not isinstance(min_frequency, int):
            raise TypeError("WordPieceTokenizer.train: min_frequency 必须是 int")
        if min_frequency < 1:
            raise ValueError("WordPieceTokenizer.train: min_frequency 必须 >= 1")

        tokens = list(_WP_SPECIALS)
        seen = set(tokens)
        words = []
        for word in self._split_words(text):
            pieces = self._to_pieces(word)
            words.append(pieces)
            for p in pieces:
                if p not in seen:
                    tokens.append(p)
                    seen.add(p)

        while len(tokens) < vocab_size:
            pair_freq = {}
            token_freq = {}
            for pieces in words:
                for t in pieces:
                    token_freq[t] = token_freq.get(t, 0) + 1
                for k in range(len(pieces) - 1):
                    pair = (pieces[k], pieces[k + 1])
                    pair_freq[pair] = pair_freq.get(pair, 0) + 1
            if not pair_freq:
                break
            best = None
            best_score = -1.0
            for (left, right), freq in pair_freq.items():
                if freq < min_frequency:
                    continue
                score = freq / float(token_freq[left] * token_freq[right])
                if score > best_score or (score == best_score and (left, right) < best):
                    best = (left, right)
                    best_score = score
            if best is None:
                break
            left, right = best
            body = right[len(_WP_CONT):] if right.startswith(_WP_CONT) else right
            merged = (left if left.startswith(_WP_CONT) else left) + body
            if merged in seen:
                break
            tokens.append(merged)
            seen.add(merged)
            new_words = []
            for pieces in words:
                out = []
                k = 0
                while k < len(pieces):
                    if k + 1 < len(pieces) and pieces[k] == left and pieces[k + 1] == right:
                        out.append(merged)
                        k += 2
                    else:
                        out.append(pieces[k])
                        k += 1
                new_words.append(out)
            words = new_words
            if verbose:
                print(f"  merge {left!r} + {right!r} -> {merged!r}  (freq={pair_freq[best]})")

        return self._install(tokens)

    # ---------- 编码 ----------
    def _encode_word(self, word, max_chars):
        if len(word) > max_chars:
            return [self.unk_id]
        ids = []
        start = 0
        n = len(word)
        while start < n:
            end = n
            found = None
            while start < end:
                piece = word[start:end]
                if start > 0:
                    piece = _WP_CONT + piece
                if piece in self.token_to_id:
                    found = piece
                    break
                end -= 1
            if found is None:
                return [self.unk_id]
            ids.append(self.token_to_id[found])
            start = end
        return ids

    def encode(self, text, max_chars_per_word=100):
        if not isinstance(text, str):
            raise TypeError("WordPieceTokenizer.encode: text 必须是 str")
        if isinstance(max_chars_per_word, bool) or not isinstance(max_chars_per_word, int):
            raise TypeError("WordPieceTokenizer.encode: max_chars_per_word 必须是 int")
        if max_chars_per_word < 1:
            raise ValueError("WordPieceTokenizer.encode: max_chars_per_word 必须 >= 1")
        ids = []
        for word in self._split_words(text):
            ids.extend(self._encode_word(word, max_chars_per_word))
        return ids

    def encode_pair(self, text_a, text_b):
        """BERT 句式输入：[CLS] a [SEP] b [SEP]。"""
        return ([self.cls_id] + self.encode(text_a) + [self.sep_id]
                + self.encode(text_b) + [self.sep_id])

    def decode(self, ids):
        if not isinstance(ids, (list, tuple)):
            raise TypeError("WordPieceTokenizer.decode: ids 必须是 list 或 tuple")
        out = []
        # 逐 token 拼接：`##xx` 直接接上，空白 token 原样输出，词首 token 视情况补空格
        for i, tid in enumerate(ids):
            if isinstance(tid, bool) or not isinstance(tid, int):
                raise TypeError(f"WordPieceTokenizer.decode: 第 {i} 个 id 不是 int")
            if tid < 0 or tid >= self.vocab_size:
                raise ValueError(
                    f"WordPieceTokenizer.decode: 第 {i} 个 id {tid} 越界 [0, {self.vocab_size})")
            token = self.tokens[tid]
            if token in _WP_SPECIALS:
                continue
            if token.startswith(_WP_CONT):
                out.append(token[len(_WP_CONT):])
                continue
            if token.strip() == "":
                out.append(token)
                continue
            # 词首 token：只有当前不在空白处才补一个空格
            if out and not out[-1].endswith((" ", "\t", "\n")):
                out.append(" ")
            out.append(token)
        return "".join(out)

    # ---------- 存档 ----------
    def save(self, filepath, quiet=False):
        payload = {"format": "ai_wordpiece_v1", "tokens": list(self.tokens)}
        with io.open(filepath, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        if not quiet:
            print(f"WordPieceTokenizer 已保存: {filepath}（{self.vocab_size} 个 token）")

    @staticmethod
    def load(filepath):
        with io.open(filepath, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, dict) or payload.get("format") != "ai_wordpiece_v1":
            raise ValueError(f"WordPieceTokenizer.load: {filepath} 不是 wordpiece 存档")
        tokens = payload.get("tokens")
        if not isinstance(tokens, list):
            raise ValueError("WordPieceTokenizer.load: tokens 字段必须是 list")
        return WordPieceTokenizer(tokens)


class TextProcessor:
    """字符级文本处理器：词表构建、one-hot 编码、滑动窗口数据生成"""

    def __init__(self, text: str):
        self.chars = sorted(list(set(text)))
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
        self.vocab_size = len(self.chars)

    def encode_one_hot(self, char: str, strict: bool = False) -> list:
        """将单个字符编码为 one-hot 向量

        strict=True 时遇到词表外字符会报错，避免静默产出全 0 向量污染训练数据。
        """
        vec = [0.0] * self.vocab_size
        if char in self.char_to_idx:
            vec[self.char_to_idx[char]] = 1.0
        elif strict:
            raise ValueError(f"encode_one_hot: 字符 {char!r} 不在词表中")
        return vec

    def encode_indices(self, text: str, strict: bool = False) -> list:
        """将字符串编码为索引列表（strict=True 时未知字符报错，而非静默映射成 0）

        strict=False（默认）时未知字符会被**静默映射到索引 0**：训练数据里若有
        词表外字符会被当成第 0 个字符来学。训练/评估路径应显式传 strict=True
        （TextProcessor.prepare_index_data 内部就是这么做的）。
        """
        if strict:
            unknown = [c for c in text if c not in self.char_to_idx]
            if unknown:
                raise ValueError(f"encode_indices: 字符 {unknown[:3]!r} 不在词表中")
        return [self.char_to_idx.get(c, 0) for c in text]

    def decode_indices(self, indices: list) -> str:
        """将索引列表解码回字符串"""
        return ''.join(self.idx_to_char.get(i, '?') for i in indices)

    def prepare_data(self, text: str, window_size: int) -> list:
        """滑动窗口生成 (input_vec, target_vec) 数据集"""
        if not isinstance(window_size, int) or window_size < 1:
            raise ValueError(f"prepare_data: window_size 必须 >= 1，收到 {window_size!r}")
        data = []
        for i in range(len(text) - window_size):
            inp_chars = text[i:i + window_size]
            target_char = text[i + window_size]
            # 拼接窗口内所有字符的 one-hot 编码
            inp_vec = []
            for c in inp_chars:
                inp_vec.extend(self.encode_one_hot(c, strict=True))
            target_vec = self.encode_one_hot(target_char, strict=True)
            data.append((inp_vec, target_vec))
        return data

    def prepare_index_data(self, text: str, window_size: int) -> list:
        """滑动窗口生成 (indices_list, target_idx) 数据集（配合 Embedding 使用）

        与 prepare_data 一致：词表外字符直接报错，不再静默映射成索引 0。
        """
        data = []
        for i in range(len(text) - window_size):
            inp = self.encode_indices(text[i:i + window_size], strict=True)
            target_char = text[i + window_size]
            if target_char not in self.char_to_idx:
                raise ValueError(
                    f"prepare_index_data: 目标字符 {target_char!r} 不在词表中"
                )
            data.append((inp, self.char_to_idx[target_char]))
        return data


def _softmax_with_temp(logits: list, temperature: float = 1.0) -> list:
    """带温度参数的 Softmax：T 越小越保守，T 越大越随机"""
    if not isinstance(temperature, (int, float)) or temperature <= 0:
        raise ValueError(f"_softmax_with_temp: temperature 必须 > 0，收到 {temperature!r}")
    if temperature != 1.0:
        logits = [x / temperature for x in logits]
    mx = max(logits)
    exps = [math.exp(x - mx) for x in logits]
    s = sum(exps)
    return [e / s for e in exps]


def sample_index(probs: list) -> int:
    """根据概率分布进行随机采样（不依赖 numpy）"""
    if not probs:
        raise ValueError("sample_index: 概率分布不能为空")
    total = sum(probs)
    if not (total > 0):
        raise ValueError("sample_index: 概率之和必须 > 0")
    if abs(total - 1.0) > 1e-9:
        # 未归一化的输入会让采样区间对不上（例如 [0.2,0.3] 有 50% 概率返回末尾索引）
        probs = [p / total for p in probs]
    r = random.random()
    cumulative = 0.0
    for i, p in enumerate(probs):
        cumulative += p
        if r < cumulative:
            return i
    return len(probs) - 1


def generate_text(network, tp: TextProcessor, start_str: str,
                  length: int = 15, window_size: int = 3, temp: float = 1.0) -> str:
    """自回归文本生成（network 只需提供 forward）

    与 generate_text_with_mlp 是同一实现，保留这个名字只为兼容旧调用。
    """
    if not isinstance(length, int) or length < 0:
        raise ValueError(f"generate_text: length 必须 >= 0，收到 {length!r}")
    return generate_text_with_mlp(network, tp, start_str, length, window_size, temp)


def generate_text_with_mlp(model, tp: TextProcessor, start_str: str,
                           length: int = 15, window_size: int = 3,
                           temp: float = 1.0) -> str:
    """使用 nn_all_in_one MLP 模型进行自回归文本生成

    start_str 里的字符必须都在词表内，否则抛 ValueError（不再静默编码成全 0）。
    上下文不足 window_size 时用**零向量**占位，不引入词表里的假字符。
    """
    generated = start_str
    for _ in range(length):
        context = generated[-window_size:]
        test_vec = []
        pad = window_size - len(context)
        if pad > 0:
            # 用零向量占位：原先用 tp.chars[0] 填充，可能是换行/生僻字，
            # 且空格在中文语料里常不在词表，都会引入不存在的字符
            test_vec.extend([0.0] * (tp.vocab_size * pad))
        for c in context:
            test_vec.extend(tp.encode_one_hot(c, strict=True))
        out = model.forward(test_vec)
        if len(out) != tp.vocab_size:
            raise ValueError(
                f"generate_text_with_mlp: 模型输出维度 {len(out)} 与词表大小 "
                f"{tp.vocab_size} 不一致（采样会越界或永远采样不到部分字符）"
            )
        # 文本模型的输出层常常已经是 softmax 概率（make_classifier 就是），
        # 再套一次 softmax 会把分布压平：temp=1 时 [0.7,0.2,0.1] -> [0.46,0.28,0.25]。
        # 这里按「是否已是合法概率分布」自动区分，概率输出先取 log 还原成 logits。
        is_prob = (all(0.0 <= v <= 1.0 for v in out)
                   and abs(sum(out) - 1.0) < 1e-6)
        logits = [math.log(max(v, 1e-12)) for v in out] if is_prob else out
        probs = _softmax_with_temp(logits, temp)
        next_idx = sample_index(probs)
        generated += tp.idx_to_char[next_idx]
    return generated


# ============================================================================
#  Trainer
# ============================================================================

def train(model: MLP, loss_fn: Callable, optimizer: Adam, data: list,
          epochs: int = 2000, batch_size: int = 2, patience: int = 50,
          val_data: Optional[list] = None, verbose: bool = True,
          shuffle: bool = True, seed=None):
    """
    loss_fn: 接受 (pred, target) -> (loss, grad)
    shuffle: 每轮是否打乱训练数据（默认 True）
    seed:    打乱用的随机种子（None = 全局 random，与旧版一致；传整数则用局部
             Random，同样的调用再跑一次结果完全相同，也不推进全局随机状态）

    注意：与 Model 一样，softmax 输出层不能搭配 mse 损失——_softmax_deriv 返回全 1
    依赖 softmax+cross_entropy 的联合梯度，配 mse 会静默产生错误梯度。
    """
    # S3: 把 Model 里的 softmax/mse 校验下沉到这里，避免绕过 Model 直接训练时训歪
    last = model.layers[-1]
    last_act = getattr(last, "activation_name", None)
    if last_act is None and last.act_fn == last._softmax:
        last_act = "softmax"
    # 与 Model 的守卫保持一致：传函数对象或 functools.partial 包装都要拦住
    _is_mse = (loss_fn is mse_loss) or (getattr(loss_fn, 'func', None) is mse_loss)
    if last_act == "softmax" and _is_mse:
        raise ValueError(
            "输出层使用 softmax 时不能搭配 mse_loss：softmax 的反向用的是 "
            "softmax+cross_entropy 的联合梯度。请改用 cross_entropy_loss，"
            "或把输出层激活改成 'linear'。"
        )
    best_val_loss = float("inf")
    no_improve = 0
    history = {"train_loss": [], "val_loss": []}

    # seed 只影响本轮打乱用的随机流：不传时仍用全局 random（行为与旧版一致），
    # 传了就换成局部 Random，既不推进全局状态、也能完整复现训练过程
    rng = random.Random(seed) if seed is not None else random

    # 复制一份再打乱，避免修改调用者传入的列表顺序
    data = list(data)
    if not data:
        raise ValueError("train: 数据集为空（train_loss /= len(data) 会除零）")
    if batch_size is None or batch_size <= 0:
        # 与 Model.fit 一致：-1 / 0 表示整批（原先 range(0,n,-1) 是空循环，
        # 会静默「训练 0 步、loss 0」）
        batch_size = len(data)

    for epoch in range(1, epochs + 1):
# ============================================================================
#  训练
# ============================================================================
        if shuffle:
            rng.shuffle(data)
        train_loss = 0.0

        for b in range(0, len(data), batch_size):
            batch = data[b:b + batch_size]

            # 与 Model.fit 相同：反向传播时直接累加，不再保存每个样本的完整梯度
            acc_w = [[[0.0] * layer.fan_in for _ in range(layer.fan_out)]
                     for layer in model.layers]
            acc_b = [[0.0] * layer.fan_out for layer in model.layers]

            for x, y in batch:
                pred = model.forward(x)
                loss, grad = loss_fn(pred, y)
                train_loss += loss

                # 反向传播
                delta = grad
                idx = len(model.layers) - 1
                for layer in reversed(model.layers):
                    gw, gb, delta = layer.backward(delta)
                    gw_acc, gb_acc = acc_w[idx], acc_b[idx]
                    for j in range(layer.fan_out):
                        gwj, gwaj = gw[j], gw_acc[j]
                        for k in range(layer.fan_in):
                            gwaj[k] += gwj[k]
                        gb_acc[j] += gb[j]
                    idx -= 1

            # 累加完成后再统一平均
            B = len(batch)
            avg_grads = [
                ([[v / B for v in row] for row in acc_w[i]],
                 [v / B for v in acc_b[i]])
                for i in range(len(model.layers))
            ]

            optimizer.step(avg_grads)

        train_loss /= len(data)
        history["train_loss"].append((epoch, train_loss))

# ============================================================================
#  验证
# ============================================================================
        val_loss = None
        if val_data:
            val_loss = evaluate(model, loss_fn, val_data)
            history["val_loss"].append((epoch, val_loss))

            if patience > 0:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    no_improve = 0
                else:
                    no_improve += 1

                # patience=0 表示**关闭**早停（与 Model.fit 一致）；
                # 原先 0 >= 0 会立刻 break，patience=0 反而变成「只训 1 轮」
                if no_improve >= patience:
                    if verbose:
                        print(f"\nEarly stopping at epoch {epoch}")
                    break

        if verbose and epoch % 100 == 0:
            if val_loss is not None:
                print(f"Epoch {epoch} | Train: {train_loss:.6f} | Val: {val_loss:.6f}")
            else:
                print(f"Epoch {epoch} | Train: {train_loss:.6f}")

    return history


def evaluate(model: MLP, loss_fn: Callable, data: list) -> float:
    """在 `data`（样本对列表）上算平均损失，不训练、不改权重。

    `loss_fn(pred, y)` 返回 `(loss, grad)`，这里只用第一项；空数据集报 ValueError。
    只看损失、要完整指标时用 `Model.evaluate_all` / `evaluate_all`。
    """
    if not data:
        raise ValueError("evaluate: 数据集为空")
    total = 0.0
    for x, y in data:
        pred = model.forward(x)
        loss, _ = loss_fn(pred, y)
        total += loss
    return total / len(data)


# ============================================================================
#  评估指标（分类 / 回归 / 序列 / 生成）
# ============================================================================

def _metric_pairs(y_true, y_pred, who):
    """校验两个等长的标签/预测序列，返回 list 副本。"""
    if not isinstance(y_true, (list, tuple)) or not isinstance(y_pred, (list, tuple)):
        raise ValueError(f"{who}: y_true / y_pred 必须是列表")
    if len(y_true) != len(y_pred):
        raise ValueError(f"{who}: 长度不一致（{len(y_true)} vs {len(y_pred)}）")
    if not y_true:
        raise ValueError(f"{who}: 不能是空列表")
    return list(y_true), list(y_pred)


def _metric_label(v, who, i):
    """类别索引必须是非负整数（bool 不算）。"""
    if isinstance(v, bool) or not isinstance(v, int):
        raise ValueError(f"{who}: 第 {i} 个标签必须是整数类别索引，收到 {v!r}")
    if v < 0:
        raise ValueError(f"{who}: 第 {i} 个标签不能是负数，收到 {v}")
    return v


def _metric_number(v, who, i):
    """数值型指标的值必须是有限实数（bool 不算）。"""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ValueError(f"{who}: 第 {i} 个值必须是数字，收到 {v!r}")
    v = float(v)
    if v != v or v in (float("inf"), float("-inf")):
        raise ValueError(f"{who}: 第 {i} 个值必须是有限数，收到 {v!r}")
    return v


def confusion_matrix(y_true, y_pred, num_classes: int = None):
    """混淆矩阵：`matrix[t][p]` = 真实类别 t 被预测成 p 的样本数。

    num_classes 省略时取 `max(标签)+1`（不够则自动给出更小的方阵）。
    返回 list[list[int]]，`len(matrix) == num_classes`。
    """
    yt, yp = _metric_pairs(y_true, y_pred, "confusion_matrix")
    labels = []
    for i, (t, p) in enumerate(zip(yt, yp)):
        labels.append(_metric_label(t, "confusion_matrix.y_true", i))
        labels.append(_metric_label(p, "confusion_matrix.y_pred", i))
    n = max(labels) + 1
    if num_classes is not None:
        if isinstance(num_classes, bool) or not isinstance(num_classes, int):
            raise ValueError(f"confusion_matrix: num_classes 必须是整数，收到 {num_classes!r}")
        if num_classes < n:
            raise ValueError(
                f"confusion_matrix: num_classes={num_classes} 小于实际类别数 {n}"
            )
        n = num_classes
    mat = [[0] * n for _ in range(n)]
    for t, p in zip(yt, yp):
        mat[t][p] += 1
    return mat


def accuracy_score(y_true, y_pred) -> float:
    """分类准确率（逐样本比较）。"""
    yt, yp = _metric_pairs(y_true, y_pred, "accuracy_score")
    hit = 0
    for i, (t, p) in enumerate(zip(yt, yp)):
        hit += 1 if _metric_label(t, "accuracy_score.y_true", i) == \
            _metric_label(p, "accuracy_score.y_pred", i) else 0
    return hit / len(yt)


def precision_recall_f1(y_true, y_pred, average: str = "macro",
                        num_classes: int = None) -> dict:
    """精确率 / 召回率 / F1。

    average: 'macro'（各类等权，默认）、'micro'（全局计数）、'weighted'（按支持数加权）
             或 'none'（只返回 per_class）。
    返回 {'precision','recall','f1','per_class':[{'precision','recall','f1','support'},...]}
    """
    if average not in ("macro", "micro", "weighted", "none"):
        raise ValueError(
            f"precision_recall_f1: average 必须是 macro/micro/weighted/none，收到 {average!r}"
        )
    mat = confusion_matrix(y_true, y_pred, num_classes)
    n = len(mat)
    per_class = []
    total_tp = total_fp = total_fn = 0
    for c in range(n):
        tp = mat[c][c]
        support = sum(mat[c])
        predicted = sum(mat[r][c] for r in range(n))
        fp = predicted - tp
        fn = support - tp
        total_tp += tp
        total_fp += fp
        total_fn += fn
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        per_class.append({
            "precision": precision, "recall": recall, "f1": f1, "support": support,
        })

    if average == "micro":
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    elif average == "weighted":
        total_support = sum(row["support"] for row in per_class)
        if total_support:
            precision = sum(row["precision"] * row["support"] for row in per_class) / total_support
            recall = sum(row["recall"] * row["support"] for row in per_class) / total_support
            f1 = sum(row["f1"] * row["support"] for row in per_class) / total_support
        else:
            precision = recall = f1 = 0.0
    else:
        # macro / none：只在「出现过」的类别上平均（支持数或预测数为 0 的类不计入）
        used = [row for row in per_class if row["support"] or row["f1"]]
        if used:
            precision = sum(row["precision"] for row in used) / len(used)
            recall = sum(row["recall"] for row in used) / len(used)
            f1 = sum(row["f1"] for row in used) / len(used)
        else:
            precision = recall = f1 = 0.0

    return {"precision": precision, "recall": recall, "f1": f1, "per_class": per_class}


def matthews_corrcoef(y_true, y_pred, num_classes: int = None) -> float:
    """多分类 MCC（二分类时即标准 MCC，取值 [-1,1]，0 表示随机）。"""
    mat = confusion_matrix(y_true, y_pred, num_classes)
    n = len(mat)
    total = sum(sum(row) for row in mat)
    if total == 0:
        raise ValueError("matthews_corrcoef: 没有样本")
    correct = sum(mat[c][c] for c in range(n))
    row_sums = [sum(mat[c]) for c in range(n)]
    col_sums = [sum(mat[r][c] for r in range(n)) for c in range(n)]
    cov_ytyp = correct * total - sum(row_sums[c] * col_sums[c] for c in range(n))
    cov_ypyp = total * total - sum(v * v for v in col_sums)
    cov_ytyt = total * total - sum(v * v for v in row_sums)
    denom = math.sqrt(cov_ypyp * cov_ytyt)
    if denom == 0:
        return 0.0
    return cov_ytyp / denom


def top_k_accuracy(y_true, y_score, k: int = 1) -> float:
    """Top-k 准确率：正确类别出现在前 k 个预测里就算对。

    y_score: 每个样本一个得分列表（长度 = 类别数，越大越可能）。
    """
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError(f"top_k_accuracy: k 必须是 ≥1 的整数，收到 {k!r}")
    yt, ys = _metric_pairs(y_true, y_score, "top_k_accuracy")
    hit = 0
    for i, (t, row) in enumerate(zip(yt, ys)):
        t = _metric_label(t, "top_k_accuracy.y_true", i)
        if not isinstance(row, (list, tuple)) or not row:
            raise ValueError(f"top_k_accuracy: 第 {i} 个得分必须是列表")
        scores = [_metric_number(v, "top_k_accuracy.y_score", i) for v in row]
        if t >= len(scores):
            raise ValueError(
                f"top_k_accuracy: 第 {i} 个样本的真实类别 {t} 超出得分长度 {len(scores)}"
            )
        order = sorted(range(len(scores)), key=lambda c: (-scores[c], c))
        if t in order[:k]:
            hit += 1
    return hit / len(yt)


def roc_auc_score(y_true, y_score) -> float:
    """二分类 ROC-AUC（用秩和/Mann-Whitney U 统计量，并列取平均秩）。"""
    yt, ys = _metric_pairs(y_true, y_score, "roc_auc_score")
    labels = [_metric_label(t, "roc_auc_score.y_true", i) for i, t in enumerate(yt)]
    scores = [_metric_number(s, "roc_auc_score.y_score", i) for i, s in enumerate(ys)]
    if any(t not in (0, 1) for t in labels):
        raise ValueError("roc_auc_score: y_true 只能是 0/1（二分类）")
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("roc_auc_score: 需要同时包含正例和负例")

    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        # 并列值取平均秩（秩从 1 开始）
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    rank_sum = sum(r for r, t in zip(ranks, labels) if t == 1)
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _regression_pairs(y_true, y_pred, who):
    """把回归目标转成一维浮点列表（支持标量列表或多输出逐维平均）。"""
    yt, yp = _metric_pairs(y_true, y_pred, who)
    flat_t, flat_p = [], []
    for i, (t, p) in enumerate(zip(yt, yp)):
        # 单输出模型常返回 [v]，而标签是标量：这种「长度 1 列表 vs 标量」先摊平
        if isinstance(t, (list, tuple)) and len(t) == 1 and not isinstance(p, (list, tuple)):
            t = t[0]
        if isinstance(p, (list, tuple)) and len(p) == 1 and not isinstance(t, (list, tuple)):
            p = p[0]
        if isinstance(t, (list, tuple)) or isinstance(p, (list, tuple)):
            if not isinstance(t, (list, tuple)) or not isinstance(p, (list, tuple)):
                raise ValueError(f"{who}: 第 {i} 个样本的形式不一致（一个标量、一个向量）")
            if len(t) != len(p):
                raise ValueError(f"{who}: 第 {i} 个样本维度不同（{len(t)} vs {len(p)}）")
            for j, (tv, pv) in enumerate(zip(t, p)):
                flat_t.append(_metric_number(tv, f"{who}.y_true[{i}][{j}]", j))
                flat_p.append(_metric_number(pv, f"{who}.y_pred[{i}][{j}]", j))
        else:
            flat_t.append(_metric_number(t, f"{who}.y_true", i))
            flat_p.append(_metric_number(p, f"{who}.y_pred", i))
    if not flat_t:
        raise ValueError(f"{who}: 没有可比较的数值")
    return flat_t, flat_p


def mean_absolute_error(y_true, y_pred) -> float:
    """MAE（多输出时按全部元素平均）。"""
    yt, yp = _regression_pairs(y_true, y_pred, "mean_absolute_error")
    return sum(abs(t - p) for t, p in zip(yt, yp)) / len(yt)


def root_mean_squared_error(y_true, y_pred) -> float:
    """RMSE（多输出时按全部元素平均）。"""
    yt, yp = _regression_pairs(y_true, y_pred, "root_mean_squared_error")
    return math.sqrt(sum((t - p) ** 2 for t, p in zip(yt, yp)) / len(yt))


def mean_absolute_percentage_error(y_true, y_pred, eps: float = 1e-12) -> float:
    """MAPE（百分比，0.1 表示 10%）；真实值为 0 时用 eps 兜底，避免除零。"""
    if not isinstance(eps, (int, float)) or isinstance(eps, bool) or eps <= 0:
        raise ValueError(f"mean_absolute_percentage_error: eps 必须是正数，收到 {eps!r}")
    yt, yp = _regression_pairs(y_true, y_pred, "mean_absolute_percentage_error")
    total = 0.0
    for t, p in zip(yt, yp):
        denom = abs(t) if abs(t) > eps else eps
        total += abs(t - p) / denom
    return 100.0 * total / len(yt)


def r2_score(y_true, y_pred) -> float:
    """决定系数 R²（多输出时按全部元素统一计算）。

    真实值恒定时返回值与 `Model.score` 一致：完全预测对返回 1.0，否则 0.0
    （原来的 `-1e15` 兜底会让结果无法解读）。
    """
    yt, yp = _regression_pairs(y_true, y_pred, "r2_score")
    mean = sum(yt) / len(yt)
    ss_tot = sum((t - mean) ** 2 for t in yt)
    ss_res = sum((t - p) ** 2 for t, p in zip(yt, yp))
    if ss_tot == 0.0:
        return 1.0 if ss_res == 0.0 else 0.0
    return 1.0 - ss_res / ss_tot


def sequence_accuracy(preds, targets) -> float:
    """整条序列全对才算对（Seq2Seq 常用）。"""
    yt, yp = _metric_pairs(targets, preds, "sequence_accuracy")
    hit = 0
    for i, (t, p) in enumerate(zip(yt, yp)):
        if not isinstance(t, (list, tuple)) or not isinstance(p, (list, tuple)):
            raise ValueError(f"sequence_accuracy: 第 {i} 个样本必须是序列")
        if list(t) == list(p):
            hit += 1
    return hit / len(yt)


def _as_tokens(x, who):
    """字符串按空白切分；list/tuple 原样转成 str 列表。"""
    if isinstance(x, str):
        return x.split()
    if isinstance(x, (list, tuple)):
        return [str(t) for t in x]
    raise ValueError(f"{who}: 需要字符串或 token 列表，收到 {type(x).__name__}")


def _ngram_counts(tokens, n):
    counts = {}
    for i in range(len(tokens) - n + 1):
        key = tuple(tokens[i:i + n])
        counts[key] = counts.get(key, 0) + 1
    return counts


def bleu_score(pred, ref, max_n: int = 4) -> float:
    """BLEU（1-4 gram 几何平均 + 简短惩罚，缺失的 n-gram 用 add-1 平滑）。

    pred / ref 可以是字符串（按空白切分）或 token 列表。
    语料级 BLEU 请对多条分别调用后取平均，或自行拼接。
    """
    if isinstance(max_n, bool) or not isinstance(max_n, int) or max_n < 1:
        raise ValueError(f"bleu_score: max_n 必须是 ≥1 的整数，收到 {max_n!r}")
    p_tokens = _as_tokens(pred, "bleu_score.pred")
    r_tokens = _as_tokens(ref, "bleu_score.ref")
    if not p_tokens or not r_tokens:
        return 0.0
    log_sum = 0.0
    for n in range(1, max_n + 1):
        p_counts = _ngram_counts(p_tokens, n)
        r_counts = _ngram_counts(r_tokens, n)
        overlap = 0
        total = max(0, len(p_tokens) - n + 1)
        for key, cnt in p_counts.items():
            overlap += min(cnt, r_counts.get(key, 0))
        if n == 1 and overlap == 0:
            # 连一个词都不重叠：直接判 0，否则 add-1 平滑会让无关句子拿到 0.3+
            return 0.0
        # add-1 平滑只用在 n>=2：短句子的 3/4-gram 常常一个都不命中，
        # 不平滑会让 BLEU 恒为 0（这也是 BLEU 在小样本上的已知缺陷）
        precision = (overlap + 1.0) / (total + 1.0) if n >= 2 else overlap / total
        log_sum += math.log(precision)
    bp = 1.0
    if len(p_tokens) < len(r_tokens):
        bp = math.exp(1.0 - len(r_tokens) / len(p_tokens))
    return bp * math.exp(log_sum / max_n)


def rouge_l(pred, ref) -> float:
    """ROUGE-L 的 F1（基于最长公共子序列）。"""
    p_tokens = _as_tokens(pred, "rouge_l.pred")
    r_tokens = _as_tokens(ref, "rouge_l.ref")
    if not p_tokens or not r_tokens:
        return 0.0
    m, n = len(p_tokens), len(r_tokens)
    # 只存两行，避免长序列吃内存
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        cur = [0] * (n + 1)
        pi = p_tokens[i - 1]
        for j in range(1, n + 1):
            if pi == r_tokens[j - 1]:
                cur[j] = prev[j - 1] + 1
            else:
                cur[j] = prev[j] if prev[j] >= cur[j - 1] else cur[j - 1]
        prev = cur
    lcs = prev[n]
    if lcs == 0:
        return 0.0
    precision = lcs / m
    recall = lcs / n
    return 2 * precision * recall / (precision + recall)


def evaluate_all(model, x, y) -> dict:
    """按模型任务一次性算出常用指标，返回 dict。

    - 回归：mae / rmse / mape / r2
    - 分类：accuracy / precision / recall / f1（macro）/ confusion_matrix，
           二分类另给 roc_auc；输出是概率时给 top_k_accuracy
    - 其它：只给 loss
    模型类型支持 Model（用它的 task）以及 MLP（按回归处理）。
    """
    if not hasattr(model, "forward"):
        raise TypeError(f"evaluate_all: model 需要 forward()，收到 {type(model).__name__}")
    if not isinstance(x, (list, tuple)) or not isinstance(y, (list, tuple)):
        raise ValueError("evaluate_all: x / y 必须是列表")
    if len(x) != len(y):
        raise ValueError(f"evaluate_all: x 与 y 长度不一致（{len(x)} vs {len(y)}）")
    if not x:
        raise ValueError("evaluate_all: x / y 不能为空")

    task = getattr(model, "task", None)
    result = {"task": task or "unknown", "n_samples": len(x)}

    if task == "regression" or (task is None and not _looks_like_probs(model, x[0])):
        preds = [model.predict(xi) for xi in x]
        result.update({
            "mae": mean_absolute_error(y, preds),
            "rmse": root_mean_squared_error(y, preds),
            "mape": mean_absolute_percentage_error(y, preds),
            "r2": r2_score(y, preds),
        })
        return result

    # 分类：一次前向拿到概率行
    probs = [model.forward(xi) for xi in x]
    n_cls = max(len(row) for row in probs)
    for i, row in enumerate(probs):
        if len(row) != n_cls:
            raise ValueError(
                f"evaluate_all: 第 {i} 个样本输出 {len(row)} 维，其它样本是 {n_cls} 维"
            )
    preds = [max(range(len(row)), key=lambda c: (row[c], -c)) for row in probs]
    labels = [_metric_label(v, "evaluate_all.y", i) for i, v in enumerate(y)]
    for i, t in enumerate(labels):
        if t >= n_cls:
            raise ValueError(
                f"evaluate_all: 第 {i} 个真实类别 {t} 超出输出维度 {n_cls}"
            )

    prf = precision_recall_f1(labels, preds, average="macro")
    result.update({
        "accuracy": accuracy_score(labels, preds),
        "precision": prf["precision"],
        "recall": prf["recall"],
        "f1": prf["f1"],
        "confusion_matrix": confusion_matrix(labels, preds, n_cls),
        "mcc": matthews_corrcoef(labels, preds, n_cls),
    })
    if n_cls > 1:
        result["top_k_accuracy"] = top_k_accuracy(labels, probs, k=min(2, n_cls))
    if n_cls == 2:
        result["roc_auc"] = roc_auc_score(labels, [row[1] for row in probs])
    return result


def _looks_like_probs(model, sample) -> bool:
    """模型没有 task 时，靠输出和是否为 1、是否非负来判断「像不像概率」。"""
    out = model.forward(sample)
    if not isinstance(out, (list, tuple)) or len(out) < 2:
        return False
    total = sum(float(v) for v in out)
    return all(float(v) >= 0.0 for v in out) and abs(total - 1.0) < 1e-9



# ============================================================================
#  模型保存 / 加载 (JSON)
# ============================================================================



def _optimizer_state(optimizer):
    """把 Adam / AdamW 的动量与步数导出成可 JSON 化的 dict。

    其它优化器（SGD / SGDMomentum / RMSprop）的缓冲结构不同，返回 None，
    对应文件里就不写 optimizer 字段（保持向后兼容）。
    """
    name = type(optimizer).__name__
    if name not in ("Adam", "AdamW"):
        return None
    model = optimizer.model
    return {
        "type": name.lower(),
        "lr": optimizer.lr,
        "beta1": optimizer.beta1,
        "beta2": optimizer.beta2,
        "eps": optimizer.eps,
        "step_count": getattr(optimizer, "step_count", 0),
        "m_w": [layer.m_w for layer in model.layers],
        "v_w": [layer.v_w for layer in model.layers],
        "m_b": [layer.m_b for layer in model.layers],
        "v_b": [layer.v_b for layer in model.layers],
    }


def load_optimizer_state(optimizer, state):
    """把 save_model 写下的 optimizer 状态灌回优化器（同类型、同形状才行）。

    返回 True 表示恢复成功；state 为 None 时返回 False。类型或形状不匹配会
    raise ValueError——静默忽略会让人以为「续训」了，其实动量是空的。
    """
    if state is None:
        return False
    want = str(state.get("type", "")).lower()
    got = type(optimizer).__name__.lower()
    if want != got:
        raise ValueError(
            f"load_optimizer_state: 文件里是 {want!r} 的状态，当前优化器是 {got!r}；"
            f"请用 optimizer={want!r} 加载模型"
        )
    model = optimizer.model
    for key in ("m_w", "v_w", "m_b", "v_b"):
        saved = state.get(key)
        if saved is None or len(saved) != len(model.layers):
            raise ValueError(
                f"load_optimizer_state: {key} 的层数 {0 if saved is None else len(saved)}"
                f" 与模型层数 {len(model.layers)} 不一致"
            )
    for i, layer in enumerate(model.layers):
        for attr, key in (("m_w", "m_w"), ("v_w", "v_w"),
                          ("m_b", "m_b"), ("v_b", "v_b")):
            saved_layer = state[key][i]
            cur = getattr(layer, attr)
            # 权重缓冲是二维、偏置缓冲是一维：分别校验形状
            if isinstance(cur[0], list):
                ok = len(saved_layer) == len(cur) and all(
                    isinstance(a, list) and len(a) == len(b)
                    for a, b in zip(saved_layer, cur)
                )
            else:
                ok = len(saved_layer) == len(cur) and all(
                    not isinstance(a, list) for a in saved_layer
                )
            if not ok:
                raise ValueError(
                    f"load_optimizer_state: 第 {i} 层 {key} 形状不匹配"
                )
            setattr(layer, attr, [[float(v) for v in row] for row in saved_layer]
                    if isinstance(cur[0], list)
                    else [float(v) for v in saved_layer])
    optimizer.step_count = int(state.get("step_count", 0))
    return True


def save_model(model, filepath: str, task: str = None, optimizer=None,
               quiet: bool = False, loss: str = None, loss_kwargs: dict = None):
    """将 MLP 模型保存为 JSON 文件

    task: 可选，"regression" / "classification"。会写进文件，供旧格式（没有逐层
          activation 记录）加载时判断输出层该用 linear 还是 softmax。
    每层还会写入 activation 字段，因此新格式文件不依赖 task 也能正确还原。
    optimizer: 可选，传入 Adam / AdamW 时同时保存动量与步数，配合
          load_optimizer_state 可以精确续训（不传则文件与旧版完全一致）。
    quiet: True 时不打印保存提示（checkpoint 每轮保存会刷屏）。
    """
    data = {
        "type": "MLP",
        "layer_sizes": model.layer_sizes,
        # task 用于加载时判断是分类器还是回归器（旧文件没有该字段）
        "task": task,
        "layers": []
    }
    # 损失函数名与参数（如 focal 的 gamma、CE 的 label_smoothing）：
    # 旧文件没有这两个字段，读的时候缺省即可
    if loss is not None:
        data["loss"] = loss
    if loss_kwargs:
        data["loss_kwargs"] = dict(loss_kwargs)
    for layer in model.layers:
        # 推断激活函数类型名称：优先用 Layer 自己记录的名字
        act_name = getattr(layer, "activation_name", None)
        if act_name is None:
            # 兼容没有 activation_name 属性的旧对象
            act_name = "linear"
            if layer.act_fn == layer._leaky_relu:
                act_name = "leaky_relu"
            elif layer.act_fn == layer._relu:
                act_name = "relu"
            elif layer.act_fn == layer._softmax:
                act_name = "softmax"

        layer_data = {
            "fan_in": layer.fan_in,
            "fan_out": layer.fan_out,
            "is_output": layer.is_output,
            "activation": act_name,
            "weights": layer.weights,
            "biases": layer.biases,
        }
        data["layers"].append(layer_data)

    if optimizer is not None:
        state = _optimizer_state(optimizer)
        if state is not None:
            data["optimizer"] = state

    try:
        with open(filepath, "w", encoding="utf-8") as f:
            # allow_nan=False：否则会写出 RFC-4627 不合法的 NaN / Infinity 字面量，
            # 严格 JSON 解析器（以及大多数其它语言）读不了
            json.dump(data, f, indent=2, ensure_ascii=False, allow_nan=False)
    except ValueError as e:
        raise ValueError(
            f"save_model: 权重里含 NaN / Inf，无法写出合法 JSON（{e}）；"
            "训练可能发散了"
        ) from None
    if not quiet:
        print(f"模型已保存到: {filepath}")


def load_model(filepath: str, default_output_activation: Optional[str] = None) -> MLP:
    """从 JSON 文件加载 MLP 模型

    default_output_activation:
        只对**旧格式**（没有逐层 activation 记录）生效；带 activation 的文件会忽略它。
        旧文件无法判断输出层原本是 softmax 还是 linear，回退优先级为：
            显式传入的参数  >  文件里记录的 task（== "classification" 时用 softmax）
            >  "linear"
        也就是说：不传参数时，一个没有 activation、也没有 task 的旧分类器
        会被当成 linear 输出，从而被上层判定为回归模型——这种情况请显式传 "softmax"。

    加载后会校验层数与每层形状（含 weights 实际行列数）是否与 layer_sizes 自洽，
    不一致直接抛 ValueError，不会带着坏权重返回。
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    if data.get("type") != "MLP":
        raise ValueError(f"不支持的类型: {data.get('type')}")
    for _k in ("layer_sizes", "layers"):
        if _k not in data:
            raise ValueError(f"load_model: 文件缺少必需字段 {_k!r}")
    for _i, _ld in enumerate(data["layers"]):
        for _k in ("fan_in", "fan_out", "weights", "biases"):
            if _k not in _ld:
                raise ValueError(f"load_model: 第 {_i} 层缺少字段 {_k!r}")

    # 从保存的数据中读取各层激活函数，兼容旧版（无 activation 字段）
    activations = []
    n_layers = len(data["layers"])
    for i, ld in enumerate(data["layers"]):
        if "activation" in ld:
            activations.append(ld["activation"])
        else:
            # 旧版兼容：最后一层按调用方指定回退，其余按隐层处理
            if i == n_layers - 1:
                # 旧文件没有逐层 activation：优先用调用方指定，其次用 task 推断
                fallback = default_output_activation
                if fallback is None and data.get("task") == "classification":
                    fallback = "softmax"
                activations.append(fallback or "linear")
            else:
                activations.append("leaky_relu")

    # 先校验、后构造：层数偏少时若先造 MLP，会在内部抛裸 IndexError 而非可读信息
    layer_sizes = data["layer_sizes"]
    saved = data["layers"]
    if len(layer_sizes) < 2:
        raise ValueError(
            f"load_model: layer_sizes={layer_sizes} 至少需要 2 个维度（输入+输出）；"
            "0 层的模型没有任何投影，等于恒等映射"
        )
    expect = len(layer_sizes) - 1
    if len(saved) != expect:
        raise ValueError(
            f"load_model: 文件里有 {len(saved)} 层，但 layer_sizes={layer_sizes} 需要 {expect} 层"
        )
    for i, ld in enumerate(saved):
        exp_in, exp_out = layer_sizes[i], layer_sizes[i + 1]
        if ld["fan_in"] != exp_in or ld["fan_out"] != exp_out:
            raise ValueError(
                f"load_model: 第 {i} 层声明形状 {ld['fan_in']}x{ld['fan_out']}，"
                f"与 layer_sizes 推出的 {exp_in}x{exp_out} 不一致"
            )
        w = ld["weights"]
        if len(w) != exp_out or any(len(row) != exp_in for row in w):
            raise ValueError(
                f"load_model: 第 {i} 层 weights 实际形状与声明不符"
                f"（应为 {exp_out} 行 × {exp_in} 列）"
            )
        # 值类型也要校验：JSON 里可能是字符串，读进来不会报错，等到 forward 才
        # TypeError: can't multiply sequence by non-int of type 'float'
        for row in w:
            for v in row:
                if not isinstance(v, (int, float)) or isinstance(v, bool):
                    raise ValueError(
                        f"load_model: 第 {i} 层 weights 含非数值 {v!r}（应为数字）"
                    )
        b = ld["biases"]
        if len(b) != exp_out:
            raise ValueError(
                f"load_model: 第 {i} 层 biases 长度 {len(b)} != fan_out={exp_out}"
            )
        for v in b:
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                raise ValueError(f"load_model: 第 {i} 层 biases 含非数值 {v!r}")

    model = MLP(layer_sizes, activations=activations)
    for i, layer in enumerate(model.layers):
        layer.weights = saved[i]["weights"]
        layer.biases = saved[i]["biases"]

    # 供 Model.load 判断 task 用（MLP 本身没有 task 概念）
    model.saved_task = data.get("task")
    # 损失配置同理（旧文件没有这两个键，get 到 None）
    model.saved_loss = data.get("loss")
    model.saved_loss_kwargs = data.get("loss_kwargs")

    print(f"模型已从 {filepath} 加载")
    return model


def save_conv_model(conv_layers, fc_model, filepath: str):
    """保存 CNN + MLP 组合模型

    限制（加载方必须知道）：
      - 只接受 Conv2d 作为卷积层，其它模块（MaxPool2d / Flatten 等）会直接抛
        ValueError —— 它们是**无参数**的，存档里不保留其超参数；
      - 因此 load_conv_model 只能恢复 Conv2d 与 MLP，加载方要按自己的结构
        重新构造 pool / flatten（顺序与参数须与保存时一致）。
    """
    data = {
        "type": "CNN_MLP",
        "conv": [],
        "mlp": {
            "layer_sizes": fc_model.layer_sizes,
            "layers": []
        }
    }
    for k, conv in enumerate(conv_layers):
        if not isinstance(conv, Conv2d):
            # 原先静默跳过：MaxPool2d 之类的模块会悄悄从存档里消失，拓扑被改变
            raise ValueError(
                f"save_conv_model: 第 {k} 个模块是 {type(conv).__name__}，"
                f"目前只支持 Conv2d（其余模块无法序列化，静默跳过会丢失拓扑）"
            )
        if isinstance(conv, Conv2d):
            data["conv"].append({
                "cls": "Conv2d",
                "in_channels": conv.in_channels,
                "out_channels": conv.out_channels,
                "kernel_size": conv.kernel_size,
                "stride": conv.stride,
                "padding": conv.padding,
                "groups": conv.groups,
                "dilation": list(conv.dilation),
                "weights": conv.weights,
                "biases": conv.biases,
            })

    for layer in fc_model.layers:
        # 与 save_model 一致：必须记录激活函数，否则 softmax 分类头
        # 重新加载后会退回 MLP 默认的 linear，静默变成回归器
        act_name = getattr(layer, "activation_name", None)
        if act_name is None:
            act_name = "linear"
            if layer.act_fn == layer._leaky_relu:
                act_name = "leaky_relu"
            elif layer.act_fn == layer._relu:
                act_name = "relu"
            elif layer.act_fn == layer._softmax:
                act_name = "softmax"
        data["mlp"]["layers"].append({
            "fan_in": layer.fan_in,
            "fan_out": layer.fan_out,
            "is_output": layer.is_output,
            "activation": act_name,
            "weights": layer.weights,
            "biases": layer.biases,
        })

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"CNN+MLP 模型已保存到: {filepath}")


def load_conv_model(filepath: str, default_output_activation: Optional[str] = None):
    """加载 CNN + MLP 组合模型

    default_output_activation:
        只对旧格式（mlp.layers 里没有 activation）生效；旧文件无法判断输出层
        原本是否是 softmax，不传时按 linear 处理，分类头会退化成回归输出。
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    for _k in ("conv", "mlp"):
        if _k not in data:
            raise ValueError(f"load_conv_model: 文件缺少必需字段 {_k!r}（不是 CNN_MLP 存档？）")
    if "layer_sizes" not in data["mlp"] or "layers" not in data["mlp"]:
        raise ValueError("load_conv_model: mlp 缺少 layer_sizes 或 layers 字段")
    for _i, _ld in enumerate(data["mlp"]["layers"]):
        for _k in ("fan_in", "fan_out", "weights", "biases"):
            if _k not in _ld:
                raise ValueError(f"load_conv_model: 第 {_i} 层缺少字段 {_k!r}")

    # 重建 CNN 层（先校验形状，坏文件在加载期就报错）
    conv_layers = []
    for k, cd in enumerate(data["conv"]):
        # groups 是后加的字段：旧存档没有它，按 1 处理
        conv = Conv2d(cd["in_channels"], cd["out_channels"],
                       cd["kernel_size"], cd["stride"], cd["padding"],
                       groups=int(cd.get("groups", 1)),
                       dilation=tuple(cd.get("dilation", (1, 1))))
        oc, ic, kh, kw = conv.out_channels, conv.cg, conv.kh, conv.kw
        cw = cd["weights"]
        shape_ok = (len(cw) == oc
                    and all(len(ch) == ic for ch in cw)
                    and all(len(row) == kw for ch in cw for m in ch for row in m)
                    and all(len(m) == kh for ch in cw for m in ch))
        if not shape_ok:
            raise ValueError(
                f"load_conv_model: 第 {k} 个 Conv2d 的 weights 形状与超参数不符"
                f"（应为 {oc}×{ic}×{kh}×{kw}，注意分组卷积每个输出通道只连 {ic} 个输入通道）"
            )
        if len(cd["biases"]) != oc:
            raise ValueError(
                f"load_conv_model: 第 {k} 个 Conv2d 的 biases 长度 "
                f"{len(cd['biases'])} != out_channels={oc}"
            )
        conv.weights = cw
        conv.biases = cd["biases"]
        conv_layers.append(conv)

    # 重建 MLP（带激活函数）：先校验结构，再构造
    layer_sizes = data["mlp"]["layer_sizes"]
    saved_layers = data["mlp"]["layers"]
    if len(layer_sizes) < 2:
        raise ValueError(
            f"load_conv_model: mlp.layer_sizes={layer_sizes} 至少需要 2 个维度"
        )
    expect = len(layer_sizes) - 1
    if len(saved_layers) != expect:
        raise ValueError(
            f"load_conv_model: 文件里有 {len(saved_layers)} 层，"
            f"但 mlp.layer_sizes={layer_sizes} 需要 {expect} 层"
        )
    for i, ld in enumerate(saved_layers):
        exp_in, exp_out = layer_sizes[i], layer_sizes[i + 1]
        if ld["fan_in"] != exp_in or ld["fan_out"] != exp_out:
            raise ValueError(
                f"load_conv_model: 第 {i} 层声明形状 {ld['fan_in']}x{ld['fan_out']}，"
                f"与 mlp.layer_sizes 推出的 {exp_in}x{exp_out} 不一致"
            )
        w = ld["weights"]
        if len(w) != exp_out or any(len(row) != exp_in for row in w):
            raise ValueError(f"load_conv_model: 第 {i} 层 weights 实际形状与声明不符")
        if len(ld["biases"]) != exp_out:
            raise ValueError(
                f"load_conv_model: 第 {i} 层 biases 长度 {len(ld['biases'])} != fan_out={exp_out}"
            )

    activations = []
    legacy = any("activation" not in ld for ld in saved_layers)
    for i, ld in enumerate(saved_layers):
        if "activation" in ld:
            activations.append(ld["activation"])
        elif i == len(saved_layers) - 1:
            activations.append(default_output_activation or "linear")
        else:
            activations.append("leaky_relu")
    if legacy:
        print("提示: 该文件是旧格式（没有激活记录），隐层按 leaky_relu、"
              f"输出层按 {activations[-1]} 处理；分类头请用 "
              "default_output_activation='softmax' 重新加载")

    model = MLP(layer_sizes, activations=activations)
    for i, layer in enumerate(model.layers):
        layer.weights = saved_layers[i]["weights"]
        layer.biases = saved_layers[i]["biases"]

    print(f"CNN+MLP 模型已从 {filepath} 加载")
    return conv_layers, model


# ============================================================================
#  序列标注 / 多标签分类头
# ============================================================================

def _sigmoid_scalar(z: float) -> float:
    """数值稳定的 sigmoid（与 Layer 的 sigmoid 分支同写法）。"""
    if z >= 0.0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


class TokenClassificationHead:
    """Token 级分类头：把 (T, d_model) 序列映射成 (T, num_labels) logits。

    典型用法是接在 `TransformerEncoder` / `CausalLM` 的隐状态之后做 NER、词性标注：

        head = TokenClassificationHead(d_model=32, num_labels=5)
        logits = head.forward(hidden)                 # (T, 5)
        loss, d_logits = head.loss_grad(logits, tags, mask=valid)
        d_hidden = head.backward(d_logits)            # (T, d_model)
        head.adam_step(lr)

    `mask` 为 False 的位置不参与 loss，梯度也恰为 0（右填充时直接用 `mask`）。
    """

    def __init__(self, d_model: int, num_labels: int):
        for name, value in (("d_model", d_model), ("num_labels", num_labels)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"TokenClassificationHead: {name} 必须是 >= 1 的整数")
        self.d_model = d_model
        self.num_labels = num_labels
        std = math.sqrt(2.0 / d_model)
        self.W = [[random.gauss(0.0, std) for _ in range(d_model)] for _ in range(num_labels)]
        self.b = [0.0] * num_labels
        self.x = None
        self._init_adam()
        self.zero_grad()

    # ---- 参数与状态
    def _named_params(self):
        yield "W", self.W, self.grad_W
        yield "b", self.b, self.grad_b

    def zero_grad(self):
        self.grad_W = _zeros_like(self.W)
        self.grad_b = _zeros_like(self.b)
        self._sync_adam()

    def _init_adam(self):
        self.m = {}
        self.v = {}
        self.step_count = 0
        self._sync_adam()

    def _sync_adam(self):
        for name, param in (("W", self.W), ("b", self.b)):
            if self.m.get(name) is None or len(self.m[name]) != len(param):
                self.m[name] = _zeros_like(param)
                self.v[name] = _zeros_like(param)

    def _init_adam_buffers(self):  # 兼容旧调用名
        self._init_adam()

    def param_count(self):
        return sum(len(row) for row in self.W) + len(self.b)

    # ---- 前向 / 反向
    def forward(self, x_seq):
        if not isinstance(x_seq, (list, tuple)) or not x_seq:
            raise ValueError("TokenClassificationHead.forward: 输入必须是非空序列")
        T = len(x_seq)
        for t, row in enumerate(x_seq):
            if not isinstance(row, (list, tuple)) or len(row) != self.d_model:
                raise ValueError(
                    f"TokenClassificationHead.forward: 第 {t} 个位置维度应为 {self.d_model}")
        self.x = [list(row) for row in x_seq]
        out = []
        for t in range(T):
            xt = self.x[t]
            out.append([sum(map(_mul, self.W[l], xt)) + self.b[l] for l in range(self.num_labels)])
        return out

    def loss_grad(self, logits, targets, mask=None):
        """softmax + 交叉熵，按有效 token 取平均；返回 (loss, d_logits)。"""
        if not isinstance(logits, (list, tuple)) or not logits:
            raise ValueError("TokenClassificationHead.loss_grad: logits 必须是非空序列")
        T = len(logits)
        if not isinstance(targets, (list, tuple)) or len(targets) != T:
            raise ValueError(
                f"TokenClassificationHead.loss_grad: targets 长度应为 {T}，收到 "
                f"{len(targets) if isinstance(targets, (list, tuple)) else type(targets).__name__}")
        if mask is None:
            mask = [True] * T
        if not isinstance(mask, (list, tuple)) or len(mask) != T:
            raise ValueError(
                f"TokenClassificationHead.loss_grad: mask 长度应为 {T}")
        valid = [1 if m else 0 for m in mask]
        n = sum(valid)
        if n == 0:
            raise ValueError("TokenClassificationHead.loss_grad: mask 全为 0，没有可训练的 token")
        loss = 0.0
        d_logits = []
        for t in range(T):
            target = targets[t]
            if isinstance(target, bool) or not isinstance(target, int):
                raise ValueError(
                    f"TokenClassificationHead.loss_grad: 第 {t} 个标签必须是整数，"
                    f"收到 {type(target).__name__}")
            if not 0 <= target < self.num_labels:
                raise ValueError(
                    f"TokenClassificationHead.loss_grad: 第 {t} 个标签 {target} 超出 "
                    f"[0, {self.num_labels})")
            probs = _softmax_row(logits[t])
            if not valid[t]:
                d_logits.append([0.0] * self.num_labels)
                continue
            p = probs[target]
            loss += -math.log(p if p > 1e-12 else 1e-12)
            row = list(probs)
            row[target] -= 1.0
            d_logits.append([v / n for v in row])
        loss /= n
        return loss, d_logits

    def backward(self, d_logits):
        if self.x is None:
            raise RuntimeError("TokenClassificationHead.backward: 必须先 forward")
        if not isinstance(d_logits, (list, tuple)) or len(d_logits) != len(self.x):
            raise ValueError(
                f"TokenClassificationHead.backward: d_logits 长度应为 {len(self.x)}")
        T = len(self.x)
        dx = [[0.0] * self.d_model for _ in range(T)]
        for t in range(T):
            drow = d_logits[t]
            if len(drow) != self.num_labels:
                raise ValueError(
                    f"TokenClassificationHead.backward: 第 {t} 行梯度维度应为 {self.num_labels}")
            xt = self.x[t]
            for l in range(self.num_labels):
                d = drow[l]
                if d == 0.0:
                    continue
                self.grad_b[l] += d
                gw = self.grad_W[l]
                for j in range(self.d_model):
                    gw[j] += d * xt[j]
                for j in range(self.d_model):
                    dx[t][j] += d * self.W[l][j]
        return dx

    def predict(self, logits, mask=None):
        T = len(logits)
        mask = [True] * T if mask is None else mask
        return [max(range(self.num_labels), key=lambda l: logits[t][l]) if mask[t] else -1
                for t in range(T)]

    # ---- 优化
    def sgd_step(self, lr: float = 0.01):
        for name, param, grad in self._named_params():
            if isinstance(param[0], list):
                for i in range(len(param)):
                    for j in range(len(param[i])):
                        param[i][j] -= lr * grad[i][j]
            else:
                for i in range(len(param)):
                    param[i] -= lr * grad[i]

    def adam_step(self, lr: float = 1e-3, b1: float = 0.9, b2: float = 0.999,
                  eps: float = 1e-8):
        self.step_count += 1
        t = self.step_count
        for name, param, grad in self._named_params():
            _adam_apply(param, grad, self.m[name], self.v[name], t, lr, b1, b2, eps)

    def train(self):
        return self

    def eval(self):
        return self


class MultiLabelHead:
    """多标签分类头：每个标签独立 sigmoid，用 BCE-with-logits 计算损失。

    与 `Layer(activation='softmax')` 的区别：多标签允许一个样本同时命中多个类别，
    标签之间不竞争，因此逐标签做二分类。
    """

    def __init__(self, fan_in: int, num_labels: int):
        for name, value in (("fan_in", fan_in), ("num_labels", num_labels)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"MultiLabelHead: {name} 必须是 >= 1 的整数")
        self.fan_in = fan_in
        self.num_labels = num_labels
        std = math.sqrt(2.0 / fan_in)
        self.W = [[random.gauss(0.0, std) for _ in range(fan_in)] for _ in range(num_labels)]
        self.b = [0.0] * num_labels
        self.x = None
        self._init_adam()
        self.zero_grad()

    def _named_params(self):
        yield "W", self.W, self.grad_W
        yield "b", self.b, self.grad_b

    def zero_grad(self):
        self.grad_W = _zeros_like(self.W)
        self.grad_b = _zeros_like(self.b)
        self._sync_adam()

    def _init_adam(self):
        self.m = {}
        self.v = {}
        self.step_count = 0
        self._sync_adam()

    def _sync_adam(self):
        for name, param in (("W", self.W), ("b", self.b)):
            if self.m.get(name) is None or len(self.m[name]) != len(param):
                self.m[name] = _zeros_like(param)
                self.v[name] = _zeros_like(param)

    def param_count(self):
        return sum(len(row) for row in self.W) + len(self.b)

    def forward(self, x):
        if not isinstance(x, (list, tuple)) or len(x) != self.fan_in:
            raise ValueError(f"MultiLabelHead.forward: 输入维度应为 {self.fan_in}")
        self.x = [float(v) for v in x]
        self.logits = [sum(map(_mul, self.W[l], self.x)) + self.b[l]
                       for l in range(self.num_labels)]
        self.probs = [_sigmoid_scalar(z) for z in self.logits]
        return list(self.probs)

    def loss_grad(self, probs, targets):
        """BCE-with-logits（数值稳定），返回 (loss, d_logits)。"""
        if not isinstance(probs, (list, tuple)) or len(probs) != self.num_labels:
            raise ValueError(f"MultiLabelHead.loss_grad: probs 长度应为 {self.num_labels}")
        if not isinstance(targets, (list, tuple)) or len(targets) != self.num_labels:
            raise ValueError(f"MultiLabelHead.loss_grad: targets 长度应为 {self.num_labels}")
        loss = 0.0
        d_logits = []
        for l in range(self.num_labels):
            y = targets[l]
            if y not in (0, 1, True, False):
                raise ValueError(
                    f"MultiLabelHead.loss_grad: 第 {l} 个标签必须是 0/1 或布尔，收到 {y!r}")
            y = 1.0 if y else 0.0
            z = self.logits[l]
            loss += max(z, 0.0) - z * y + math.log1p(math.exp(-abs(z)))
            d_logits.append((self.probs[l] - y) / self.num_labels)
        return loss / self.num_labels, d_logits

    def backward(self, d_logits):
        if self.x is None:
            raise RuntimeError("MultiLabelHead.backward: 必须先 forward")
        if not isinstance(d_logits, (list, tuple)) or len(d_logits) != self.num_labels:
            raise ValueError(f"MultiLabelHead.backward: d_logits 长度应为 {self.num_labels}")
        dx = [0.0] * self.fan_in
        for l in range(self.num_labels):
            d = d_logits[l]
            if d == 0.0:
                continue
            self.grad_b[l] += d
            gw = self.grad_W[l]
            for j in range(self.fan_in):
                gw[j] += d * self.x[j]
                dx[j] += d * self.W[l][j]
        return dx

    def predict(self, probs=None, threshold: float = 0.5):
        probs = self.probs if probs is None else probs
        if not 0.0 < threshold < 1.0:
            raise ValueError("MultiLabelHead.predict: threshold 必须在 (0, 1) 内")
        return [1 if p >= threshold else 0 for p in probs]

    def sgd_step(self, lr: float = 0.01):
        for _name, param, grad in self._named_params():
            if isinstance(param[0], list):
                for i in range(len(param)):
                    for j in range(len(param[i])):
                        param[i][j] -= lr * grad[i][j]
            else:
                for i in range(len(param)):
                    param[i] -= lr * grad[i]

    def adam_step(self, lr: float = 1e-3, b1: float = 0.9, b2: float = 0.999,
                  eps: float = 1e-8):
        self.step_count += 1
        t = self.step_count
        for name, param, grad in self._named_params():
            _adam_apply(param, grad, self.m[name], self.v[name], t, lr, b1, b2, eps)

    def train(self):
        return self

    def eval(self):
        return self


# ============================================================================
#  PyTorch 数值对齐对比
# ============================================================================

def compare_with_pytorch():
    """
    可选：安装 torch 后运行此函数，逐层对比前向/反向传播数值。
    如果未安装 torch，会自动跳过。
    """
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("[跳过] 未安装 PyTorch，无法进行数值对比。pip install torch 后可重试。")
        return

    print("\n" + "=" * 60)
    print("PyTorch 数值对齐验证")
    print("=" * 60)

# ============================================================================
#  设定相同随机种子 & 初始化
# ============================================================================
    torch.manual_seed(42)
    random.seed(42)

    input_dim = 4
    hidden_dim = 6
    output_dim = 2

# ============================================================================
#  PyTorch 模型
# ============================================================================
    pt_model = nn.Sequential(
        nn.Linear(input_dim, hidden_dim, bias=True),
        nn.LeakyReLU(0.01),
        nn.Linear(hidden_dim, hidden_dim, bias=True),
        nn.LeakyReLU(0.01),
        nn.Linear(hidden_dim, output_dim, bias=True),
    )

# ============================================================================
#  我们的模型
# ============================================================================
    our_model = MLP([input_dim, hidden_dim, hidden_dim, output_dim],
                    activations=["leaky_relu", "leaky_relu", "linear"])

    # 对齐权重
    pt_layers = [m for m in pt_model if isinstance(m, nn.Linear)]
    with torch.no_grad():
        for i, (pt_linear, our_layer) in enumerate(zip(pt_layers, our_model.layers)):
            w = pt_linear.weight.data.numpy()
            b = pt_linear.bias.data.numpy()
            # PyTorch Linear: (fan_out, fan_in)
            for j in range(our_layer.fan_out):
                our_layer.weights[j] = w[j].tolist()
                our_layer.biases[j] = float(b[j])

# ============================================================================
#  前向对齐
# ============================================================================
    x_np = [random.uniform(-1, 1) for _ in range(input_dim)]
    x_pt = torch.tensor(x_np, dtype=torch.float32).unsqueeze(0)  # (1, input_dim)

    our_out = our_model.forward(x_np)
    with torch.no_grad():
        pt_out = pt_model(x_pt).squeeze(0).numpy()

    print(f"\n输入: {[round(v, 4) for v in x_np]}")
    print(f"PyTorch 前向: {[round(v, 6) for v in pt_out.tolist()]}")
    print(f"Ours   前向: {[round(v, 6) for v in our_out]}")
    fwd_diff = sum(abs(a - b) for a, b in zip(our_out, pt_out))
    print(f"前向 |Δ| 总和: {fwd_diff:.2e}  {'✓ 通过' if fwd_diff < 1e-5 else '✗ 偏差较大'}")

# ============================================================================
#  反向对齐
# ============================================================================
    target_np = [random.uniform(-1, 1) for _ in range(output_dim)]
    target_pt = torch.tensor(target_np, dtype=torch.float32).unsqueeze(0)

    # PyTorch
    pt_model.zero_grad()
    pt_out2 = pt_model(x_pt)
    pt_loss = nn.functional.mse_loss(pt_out2, target_pt)
    pt_loss.backward()

    # Ours
    our_out2 = our_model.forward(x_np)
    our_loss, our_grad = mse_loss(our_out2, target_np)
    delta = our_grad
    our_grads = []
    for layer in reversed(our_model.layers):
        gw, gb, delta = layer.backward(delta)
        our_grads.insert(0, (gw, gb))

    print(f"\n损失对齐:")
    print(f"  PyTorch loss: {pt_loss.item():.10f}")
    print(f"  Ours   loss:  {our_loss:.10f}")

    all_grad_ok = True
    for i, ((gw, gb), pt_linear) in enumerate(zip(our_grads, pt_layers)):
        pt_gw = pt_linear.weight.grad.numpy()
        pt_gb = pt_linear.bias.grad.numpy()

        gw_flat = [w for row in gw for w in row]
        pt_gw_flat = pt_gw.flatten().tolist()
        diff_w = sum(abs(a - b) for a, b in zip(gw_flat, pt_gw_flat))
        diff_b = sum(abs(a - b) for a, b in zip(gb, pt_gb.tolist()))

        status = "✓" if (diff_w < 1e-5 and diff_b < 1e-5) else "✗"
        if status == "✗":
            all_grad_ok = False
        print(f"  第 {i + 1} 层 grad_w |Δ|={diff_w:.2e}  grad_b |Δ|={diff_b:.2e}  {status}")

    print(f"\n{'全部通过 ✓' if all_grad_ok else '存在偏差 ✗'}")
    return fwd_diff < 1e-5 and all_grad_ok


# ============================================================================
#  训练基础设施：seed / Callback / 形状推断 / 批量推理 / 诊断
# ============================================================================


def set_seed(seed: int):
    """固定全局随机种子（对应 PyTorch 的 torch.manual_seed）。

    只影响 `random` 模块。想让「训练完全可复现」还要给 fit / quick_train /
    train_from_config 传 seed —— 它们用局部 Random，不受这里影响。
    """
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError(f"set_seed: seed 必须是整数，收到 {type(seed).__name__}")
    random.seed(seed)


class Callback:
    """训练回调基类：继承后按需覆盖钩子，传给 fit(callbacks=[...])。

    钩子：
        on_train_begin(model, info)   info: epochs/batch_size/lr/task/n_train/n_val/resumed_step
        on_epoch_begin(epoch, info)   info: epoch/epochs/lr
        on_epoch_end(epoch, logs)     logs: epoch/train_loss/val_loss/grad_norm/layer_grad_norm/lr
        on_train_end(history)
    回调里抛异常会中断训练（不吞异常）。
    """

    def on_train_begin(self, model, info):
        pass

    def on_epoch_begin(self, epoch, info):
        pass

    def on_epoch_end(self, epoch, logs):
        pass

    def on_train_end(self, history):
        pass


class EarlyStopping(Callback):
    """监控指标，连续 `patience` 轮没有改善就请求 `fit` 停止。

    `mode="min"` 用于 loss（越小越好），`"max"` 用于 accuracy（越大越好）。
    只发出停止请求，不负责回滚权重——要回滚最优权重请用 `fit(restore_best_weights=True)`。
    """

    def __init__(self, monitor='val_loss', patience=5, min_delta=0.0, mode='min'):
        if monitor not in ('train_loss', 'val_loss', 'grad_norm'):
            raise ValueError(
                f"EarlyStopping: monitor 只能是 train_loss/val_loss/grad_norm，收到 {monitor!r}")
        if not isinstance(patience, int) or isinstance(patience, bool) or patience < 1:
            raise ValueError(f"EarlyStopping: patience 必须是 >=1 的整数，收到 {patience!r}")
        if float(min_delta) < 0.0:
            raise ValueError("EarlyStopping: min_delta 不能为负")
        if mode not in ('min', 'max'):
            raise ValueError(f"EarlyStopping: mode 只能是 'min' 或 'max'，收到 {mode!r}")
        self.monitor = monitor
        self.patience = patience
        self.min_delta = float(min_delta)
        self.mode = mode
        self.best = None
        self.bad_epochs = 0
        self.should_stop = False
        self.stopped_epoch = None

    def on_train_begin(self, model, info):
        self.best = None
        self.bad_epochs = 0
        self.should_stop = False
        self.stopped_epoch = None

    def on_epoch_end(self, epoch, logs):
        cur = logs.get(self.monitor)
        if cur is None:
            return
        if self.best is None:
            self.best = cur
            return
        improved = (cur < self.best - self.min_delta) if self.mode == 'min' \
            else (cur > self.best + self.min_delta)
        if improved:
            self.best = cur
            self.bad_epochs = 0
            return
        self.bad_epochs += 1
        if self.bad_epochs >= self.patience:
            self.should_stop = True
            self.stopped_epoch = epoch


class ModelCheckpoint(Callback):
    """每隔 `every` 轮把模型（含优化器状态）存一份；`save_best_only=True` 时只存最优。

    存出来的文件可以直接 `Model.load(path, restore_optimizer=True)`，
    也可以作为 `fit(resume_from=...)` 的输入。
    """

    def __init__(self, path, every=1, monitor='val_loss', save_best_only=False, mode='min'):
        if not isinstance(path, str) or not path:
            raise ValueError("ModelCheckpoint: path 必须是非空字符串")
        if not isinstance(every, int) or isinstance(every, bool) or every < 1:
            raise ValueError(f"ModelCheckpoint: every 必须是 >=1 的整数，收到 {every!r}")
        if mode not in ('min', 'max'):
            raise ValueError(f"ModelCheckpoint: mode 只能是 'min' 或 'max'，收到 {mode!r}")
        self.path = path
        self.every = every
        self.monitor = monitor
        self.save_best_only = bool(save_best_only)
        self.mode = mode
        self.best = None
        self.saved_epochs = []

    def on_train_begin(self, model, info):
        self.best = None
        self.saved_epochs = []
        self._model = model

    def _save(self, model):
        save_model(model.model, self.path, task=model.task,
                   optimizer=model.optimizer, quiet=True)

    def on_epoch_end(self, epoch, logs):
        if not hasattr(self, '_model'):
            return
        model = self._model
        if self.save_best_only:
            cur = logs.get(self.monitor)
            if cur is None:
                return
            better = self.best is None or (
                cur < self.best if self.mode == 'min' else cur > self.best)
            if better:
                self.best = cur
                self._save(model)
                self.saved_epochs.append(epoch)
            return
        if epoch % self.every == 0:
            self._save(model)
            self.saved_epochs.append(epoch)


class LoggingCallback(Callback):
    """把每轮指标写进标准库 `logging`（默认 logger 名 `ai.train`）。"""

    def __init__(self, logger=None, level=logging.INFO):
        self.logger = logger or logging.getLogger('ai.train')
        self.level = level

    def on_train_begin(self, model, info):
        self.logger.log(self.level, "训练开始: epochs=%s batch=%s lr=%s n_train=%s",
                        info.get('epochs'), info.get('batch_size'),
                        info.get('lr'), info.get('n_train'))

    def on_epoch_end(self, epoch, logs):
        self.logger.log(self.level, "epoch %s: train=%.6f val=%s lr=%.6g",
                        epoch, logs['train_loss'],
                        'None' if logs['val_loss'] is None else "%.6f" % logs['val_loss'],
                        logs['lr'])

    def on_train_end(self, history):
        self.logger.log(self.level, "训练结束: %s 轮", len(history.get('train_loss', [])))


class HistoryCallback(Callback):
    """把每轮的 logs 原样收进 `self.records`（画曲线、导出 CSV 都够用）。"""

    def __init__(self):
        self.records = []

    def on_train_begin(self, model, info):
        self.records = []

    def on_epoch_end(self, epoch, logs):
        self.records.append(dict(logs))


_CALLBACK_HOOKS = ('on_train_begin', 'on_epoch_begin', 'on_epoch_end', 'on_train_end')


def _normalize_callbacks(callbacks):
    """None / 单个回调 / 回调列表 → 列表，并校验四个钩子齐备"""
    if callbacks is None:
        return []
    items = list(callbacks) if isinstance(callbacks, (list, tuple)) else [callbacks]
    for cb in items:
        for hook in _CALLBACK_HOOKS:
            if not callable(getattr(cb, hook, None)):
                raise TypeError(
                    f"fit: 回调 {type(cb).__name__} 缺少可调用的 {hook}()"
                    "（继承 AI.Callback 即可获得默认空实现）"
                )
    return items


def _as_mlp(model, who):
    """Model 或 MLP → MLP"""
    mlp = model.model if isinstance(model, Model) else model
    if not isinstance(mlp, MLP):
        raise TypeError(f"{who}: 需要 MLP 或 Model，收到 {type(model).__name__}")
    if not mlp.layers:
        raise ValueError(f"{who}: 模型没有层")
    return mlp


def _check_samples(x, mlp, who):
    """校验样本列表并转成 list[list[float]]"""
    if isinstance(x, (str, bytes)) or not isinstance(x, (list, tuple)) or not x:
        raise ValueError(f"{who}: x 必须是非空样本列表（单个样本请写成 [x]）")
    n_in = mlp.layers[0].fan_in
    samples = []
    for i, xi in enumerate(x):
        if isinstance(xi, (int, float, bool)) or not isinstance(xi, (list, tuple)):
            raise ValueError(f"{who}: 第 {i} 个样本不是序列（应为 list[float]）")
        if len(xi) != n_in:
            raise ValueError(
                f"{who}: 第 {i} 个样本维度 {len(xi)} 与模型输入维度 {n_in} 不一致"
            )
        samples.append(list(xi))
    return samples


def infer_shapes(model, x):
    """跑一次前向，报告每层的输入/输出维度、激活名、参数量与真实输出。

    不训练、不改权重、不建图。x 是样本列表（单个样本写成 [x]）。
    """
    mlp = _as_mlp(model, "infer_shapes")
    samples = _check_samples(x, mlp, "infer_shapes")
    out = mlp.forward(samples[0])
    layers = []
    total = 0
    for i, layer in enumerate(mlp.layers):
        n = layer.fan_in * layer.fan_out + layer.fan_out
        total += n
        layers.append({
            "layer": i,
            "in": layer.fan_in,
            "out": layer.fan_out,
            "activation": getattr(layer, "activation_name", "unknown"),
            "params": n,
            "output": list(layer.a),
        })
    return {
        "input_shape": [mlp.layers[0].fan_in],
        "output_shape": [mlp.layers[-1].fan_out],
        "params": total,
        "layers": layers,
        "output": list(out),
    }


def predict_batch(model, x, batch_size: int = 64):
    """分块推理：结果与 predict 相同（单样本→list，多样本→list[list]）。

    只是把样本切块逐个前向，数值与逐个 predict 完全一致（同一条计算路径）。
    """
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError(f"predict_batch: batch_size 必须是 >=1 的整数，收到 {batch_size!r}")
    is_model = isinstance(model, Model)
    if not is_model and not isinstance(model, MLP):
        raise TypeError(f"predict_batch: 需要 MLP 或 Model，收到 {type(model).__name__}")
    if not isinstance(x, (list, tuple)) or not x:
        raise ValueError("predict_batch: x 必须非空（单个样本请写成 [x]）")
    if not isinstance(x[0], (list, tuple)):        # 单个样本
        return model.predict(list(x)) if is_model else model.forward(list(x))
    out = []
    for i in range(0, len(x), batch_size):
        chunk = [list(xi) for xi in x[i:i + batch_size]]
        if is_model:
            out.extend(model.predict(chunk))
        else:
            out.extend(model.forward(xi) for xi in chunk)
    return out


def _collect_activations(model, x, who):
    """跑前向并收集每层的激活输出 → (samples, per_layer[list[float]])"""
    mlp = _as_mlp(model, who)
    samples = _check_samples(x, mlp, who)
    per_layer = [[] for _ in mlp.layers]
    for xi in samples:
        mlp.forward(xi)
        for i, layer in enumerate(mlp.layers):
            per_layer[i].extend(layer.a)
    return samples, mlp, per_layer


def activation_stats(model, x):
    """统计各层激活输出的分布：均值 / 标准差 / 最小 / 最大 / 零值比例。

    用来判断「某层是不是整体死掉了/饱和了」：std 接近 0 或 zero_ratio 接近 1
    通常说明这一层没在学东西。
    """
    samples, mlp, per_layer = _collect_activations(model, x, "activation_stats")
    layers = []
    for i, layer in enumerate(mlp.layers):
        vals = per_layer[i]
        n = len(vals)
        mean = sum(vals) / n
        var = sum((v - mean) ** 2 for v in vals) / n
        zeros = sum(1 for v in vals if v == 0.0)
        layers.append({
            "layer": i,
            "activation": getattr(layer, "activation_name", "unknown"),
            "mean": mean,
            "std": math.sqrt(var),
            "min": min(vals),
            "max": max(vals),
            "zero_ratio": zeros / n,
        })
    return {"samples": len(samples), "layers": layers}


def dead_units(model, x, atol: float = 1e-12):
    """找出「对这批输入从不激活」的单元（按层返回索引与比例）。

    判据：该单元在所有样本上的 |激活| 都 <= atol。对 ReLU 家族这意味着
    「永远不点火」（典型是学习率过大或初始化不当）；linear / softmax 层照算，
    但零激活不代表死亡，结果里用 note 标注。
    """
    if not isinstance(atol, (int, float)) or isinstance(atol, bool) or atol < 0:
        raise ValueError(f"dead_units: atol 必须是 >=0 的数字，收到 {atol!r}")
    samples, mlp, per_layer = _collect_activations(model, x, "dead_units")
    atol = float(atol)
    layers = []
    for i, layer in enumerate(mlp.layers):
        vals = per_layer[i]
        n_samples = len(samples)
        dead = []
        for j in range(layer.fan_out):
            worst = 0.0
            for k in range(n_samples):
                a = abs(vals[k * layer.fan_out + j])
                if a > worst:
                    worst = a
            if worst <= atol:
                dead.append(j)
        act = getattr(layer, "activation_name", "unknown")
        note = None
        if act in ("linear", "softmax"):
            note = f"{act} 层的零激活不代表死亡，仅作参考"
        layers.append({
            "layer": i,
            "activation": act,
            "dead": dead,
            "dead_ratio": len(dead) / layer.fan_out,
            "note": note,
        })
    return {"samples": len(samples), "atol": atol, "layers": layers}


def _layer_norms(grads):
    """每层的梯度范数（层内先平方和再开根号，求和顺序与 _grads_norm 一致）"""
    out = []
    for gw_l, gb_l in grads:
        sq = 0.0
        for row in gw_l:
            for v in row:
                sq += v * v
        for v in gb_l:
            sq += v * v
        out.append(math.sqrt(sq))
    return out


def _accum_layer_norms(dst, grads):
    """把一层的梯度范数累加进 dst"""
    for i, v in enumerate(_layer_norms(grads)):
        dst[i] += v


# ============================================================================
#  High-Level API: Model 封装（sklearn 风格） + 工厂函数
# ============================================================================

def _param_leaves(param):
    """枚举嵌套 list 的叶子路径：[(path, value), ...]"""
    if isinstance(param, list):
        out = []
        for i, item in enumerate(param):
            for path, value in _param_leaves(item):
                out.append(((i,) + path, value))
        return out
    return [((), param)]


def _pget(param, path):
    for i in path:
        param = param[i]
    return param


def _pset(param, path, value):
    for i in path[:-1]:
        param = param[i]
    param[path[-1]] = value


def grad_check(module, x, y=None, loss_fn=None, eps: float = 1e-6,
               tol: float = 1e-5, max_params=None):
    """用中心差分校验解析梯度（纯标准库实现，方便在任意模型上自检）。

    支持三类对象：
        grad_check(mlp, x, y)         # MLP，默认 mse_loss
        grad_check(model, x, y)       # Model（MLP 包装），默认用 model 自己的损失
        grad_check(lm, ids)           # CausalLM，用下一个 token 的交叉熵
        grad_check(s2s, (src, tgt))   # Seq2SeqLM，用 teacher forcing 的交叉熵

    eps:        差分步长
    tol:        相对误差阈值
    max_params: 最多抽查多少个标量参数（None = MLP/Model 全查、CausalLM 抽查 24 个）

    返回 dict：{'ok': bool, 'worst_name': str, 'worst_rel': float,
                'checks': [{'name','rel','analytic','numeric'}, ...],
                'zeros': [名字, ...]}

    解析与数值都 < 1e-8 的参数视为「本应为 0」（如 softmax 前的 b_k：
    给整行加常数不改变输出），只记进 `zeros`、不参与相对误差判断；
    只有一边接近 0 时仍按相对误差严格比较。
    """
    if isinstance(module, CausalLM):
        return _grad_check_lm(module, x, eps, tol, max_params)
    if isinstance(module, Seq2SeqLM):
        return _grad_check_seq2seq(module, x, eps, tol, max_params)

    if isinstance(module, MLP):
        forward = module.forward
        params = module.get_parameters()

        def backward(dloss):
            return _mlp_backward(module, dloss)
        default_loss = mse_loss
    elif isinstance(module, Model):
        forward = module.forward
        params = module.model.get_parameters()

        def backward(dloss):
            return module.backward(dloss)
        default_loss = None
    else:
        raise TypeError(
            f"grad_check: 支持 MLP / Model / CausalLM / Seq2SeqLM，"
                f"收到 {type(module).__name__}"
        )

    if default_loss is None and loss_fn is None:
        loss_fn = module._loss

        def loss_of(xx, yy):
            return loss_fn(forward(xx), yy)
    else:
        loss_fn = loss_fn or default_loss

        def loss_of(xx, yy):
            return loss_fn(forward(xx), yy)

    _, dloss = loss_of(x, y)
    grads = backward(dloss)

    leaves = []
    for li, (weights, biases) in enumerate(params):
        for path, _ in _param_leaves(weights):
            leaves.append((f"layers{li}.weight" + "".join(f"[{p}]" for p in path),
                           weights, path, grads[li][0], path))
        for path, _ in _param_leaves(biases):
            leaves.append((f"layers{li}.bias" + "".join(f"[{p}]" for p in path),
                           biases, path, grads[li][1], path))

    if max_params is not None:
        leaves = leaves[:int(max_params)]

    checks = []
    zeros = []
    worst_name, worst_rel = "", 0.0
    for name, holder, path, grad_holder, gpath in leaves:
        orig = _pget(holder, path)
        ana = _pget(grad_holder, gpath)
        _pset(holder, path, orig + eps)
        lp, _ = loss_of(x, y)
        _pset(holder, path, orig - eps)
        lm_, _ = loss_of(x, y)
        _pset(holder, path, orig)
        num = (lp - lm_) / (2.0 * eps)
        if abs(ana) < _GRAD_CHECK_ZERO and abs(num) < _GRAD_CHECK_ZERO:
            zeros.append(name)
            continue
        rel = abs(ana - num) / max(1e-12, abs(ana) + abs(num))
        checks.append({"name": name, "rel": rel, "analytic": ana, "numeric": num})
        if rel > worst_rel:
            worst_name, worst_rel = name, rel

    return {
        "ok": worst_rel <= tol,
        "worst_name": worst_name,
        "worst_rel": worst_rel,
        "checks": checks,
        "zeros": zeros,
    }


# 解析与数值梯度都小于该阈值时视为「本应为 0」，跳过相对误差判断：
# 例如偏置 b_k 对 softmax 等价于「给整行加常数」，真实梯度恰好为 0，
# 数值差分只剩 1e-10 级浮点噪声，此时相对误差必然算成 1 而误报。
_GRAD_CHECK_ZERO = 1e-8


def _lm_training_state(lm):
    """读当前是不是训练模式（取编码器第一层的注意力标志）"""
    try:
        return bool(lm.encoder.layers[0].self_attn.training)
    except (AttributeError, IndexError, TypeError):
        return None


def _lm_restore_mode(lm, was_training):
    """把模型恢复成调用 grad_check 之前的状态（None 表示无法判断，保持不动）"""
    if was_training is None:
        return
    if was_training:
        lm.train()
    else:
        lm.eval()


def _grad_check_named(lm, loss_fn, eps: float, tol: float, max_params):
    """按参数块名抽查数值梯度（CausalLM / Seq2SeqLM 共用）

    loss_fn() 负责「清零梯度 → 前向+反向 → 返回标量 loss」，
    且必须可重复调用（中心差分要正负各一次）。
    """
    lm.zero_grad()
    loss_fn()
    named = lm._named_params()  # (name, param, grad)

    leaves = []
    for name, param, grad in named:
        for path, _ in _param_leaves(param):
            leaves.append((name + "".join(f"[{p}]" for p in path),
                           param, path, grad, path))
    if max_params is None:
        max_params = 24
    max_params = int(max_params)
    if max_params < len(leaves):
        rng = random.Random(0)
        leaves = rng.sample(leaves, max_params)

    def loss_now():
        return loss_fn()

    checks = []
    zeros = []
    worst_name, worst_rel = "", 0.0
    for name, holder, path, grad_holder, gpath in leaves:
        orig = _pget(holder, path)
        ana = _pget(grad_holder, gpath)
        _pset(holder, path, orig + eps)
        lp = loss_now()
        _pset(holder, path, orig - eps)
        lm_ = loss_now()
        _pset(holder, path, orig)
        num = (lp - lm_) / (2.0 * eps)
        if abs(ana) < _GRAD_CHECK_ZERO and abs(num) < _GRAD_CHECK_ZERO:
            zeros.append(name)
            continue
        rel = abs(ana - num) / max(1e-12, abs(ana) + abs(num))
        checks.append({"name": name, "rel": rel, "analytic": ana, "numeric": num})
        if rel > worst_rel:
            worst_name, worst_rel = name, rel
    loss_now()  # 复原缓存/grads 状态
    return {
        "ok": worst_rel <= tol,
        "worst_name": worst_name,
        "worst_rel": worst_rel,
        "checks": checks,
        "zeros": zeros,
    }


def _grad_check_lm(lm, ids, eps: float, tol: float, max_params):
    """CausalLM 的梯度自检：按参数块名抽查（默认 24 个标量）"""
    was_training = _lm_training_state(lm)
    lm.eval()  # 关掉 dropout，否则固定 seed 也未必复现
    try:
        def loss_fn():
            lm.zero_grad()
            return lm._loss_and_backward(list(ids))

        return _grad_check_named(lm, loss_fn, eps, tol, max_params)
    finally:
        _lm_restore_mode(lm, was_training)


def _grad_check_seq2seq(lm, pair, eps: float, tol: float, max_params):
    """Seq2SeqLM 的梯度自检：pair 是 (src_ids, tgt_ids)"""
    if not isinstance(pair, (list, tuple)) or len(pair) != 2:
        raise TypeError(
            "grad_check: Seq2SeqLM 的 x 必须是 (src_ids, tgt_ids) 二元组，"
            f"收到 {pair!r}"
        )
    src_ids, tgt_ids = pair
    was_training = _lm_training_state(lm)
    lm.eval()
    try:
        def loss_fn():
            lm.zero_grad()
            return lm._loss_and_backward(list(src_ids), list(tgt_ids))

        return _grad_check_named(lm, loss_fn, eps, tol, max_params)
    finally:
        _lm_restore_mode(lm, was_training)


def _mlp_backward(model: 'MLP', loss_grad: list) -> list:
    """内部工具：对 MLP 执行反向传播，返回每层的 (grad_w, grad_b) 列表"""
    delta = loss_grad
    grads = []
    fhooks = model._bhooks
    for i in range(len(model.layers) - 1, -1, -1):
        layer = model.layers[i]
        gw, gb, delta = layer.backward(delta)
        if fhooks:
            fn = fhooks.get(i)
            if fn is not None:
                fn(layer, delta, gw, gb)
        grads.append((gw, gb))
    grads.reverse()
    return grads


class Model:
    """
    高层封装：将模型、损失函数、优化器打包为 sklearn 风格 API。

    ── 快速开始 ──────────────────────────────────────────────
        # 一行创建并训练
        model = quick_train(x, y, task='classification', epochs=200)
        print(model.score(x_test, y_test))

        # 工厂函数
        model = make_classifier(input_dim=10, num_classes=5, hidden_layers=(64, 32))
        model = make_regressor(input_dim=5, output_dim=2, hidden_layers=(64, 32))

        # 手动创建
        model = Model(MLP([5, 16, 8, 2]), loss_fn='mse', optimizer='adam', lr=0.01)
        history = model.fit(x, y, epochs=300, batch_size=8, patience=30)
        pred = model.predict(x_test)

        # 保存 / 加载
        model.save("model.json")
        model2 = Model.load("model.json")
    ──────────────────────────────────────────────────────────
    """

    _OPTIMIZER_MAP = {
        'adam': Adam,
        'adamw': AdamW,
        'sgd': SGD,
        'sgd_momentum': SGDMomentum,
        'rmsprop': RMSprop,
    }

    _LOSS_MAP = {
        'mse': mse_loss,
        'cross_entropy': cross_entropy_loss,
        'ce': cross_entropy_loss,
        'focal': focal_loss,
        'bce': bce_loss,
        'bce_with_logits': bce_with_logits_loss,
        'huber': huber_loss,
        'smooth_l1': smooth_l1_loss,
        'l1': l1_loss,
        'log_cosh': log_cosh_loss,
        'quantile': quantile_loss,
        'class_balanced': class_balanced_loss,
        'distill': distill_loss,
        'ldam': ldam_loss,
    }

    # 目标按「类别索引」解释的损失（其余按数值向量解释）
    _INDEX_LOSSES = ('cross_entropy', 'ce', 'focal', 'class_balanced', 'ldam')
    # 目标按数值向量解释的损失（标量会被包成 [target]）
    _NUMERIC_LOSSES = ('mse', 'bce', 'bce_with_logits', 'huber', 'distill', 'smooth_l1',
                       'l1', 'log_cosh', 'quantile')

    def __init__(self, model, loss_fn=None, optimizer=None, lr=0.01,
                 opt_kwargs=None, task='auto', loss_kwargs=None):
        """
        model:       MLP 实例（必传）
        loss_fn:     'mse' / 'cross_entropy' / 'ce' / 'focal' / 'bce' /
                     'bce_with_logits' / 'huber' / callable(pred, target) -> (loss, grad)
                     （kl_divergence 不在此列：它要求 pred 是概率分布，而 softmax 层的
                     反向是 softmax+CE 的联合梯度，非 CE 损失叠在 softmax 上梯度不自洽；
                     需要 KL 时请手动调用 AI.kl_divergence 并自己反传）
        loss_kwargs: dict，传给损失函数的额外参数（如交叉熵的 label_smoothing、
                     focal 的 gamma/alpha、huber 的 delta）
        optimizer:   'adam' / 'sgd' / 'sgd_momentum' / 'rmsprop' / 优化器实例
        lr:          学习率（optimizer 为字符串时生效）
        opt_kwargs:  dict，传给优化器构造函数的额外参数（如 weight_decay）
        task:        'regression' / 'classification' / 'auto'（自动推断）
        """
        if not isinstance(model, MLP):
            raise TypeError(f"model 必须是 MLP 实例，当前类型: {type(model).__name__}")
        if len(model.layers) == 0:
            raise ValueError("Model: model 至少要有 1 层（MLP 的层数 = 维度数-1）")

        self.model = model
        self.lr = lr
        self.opt_kwargs = opt_kwargs or {}
        if loss_kwargs is not None and not isinstance(loss_kwargs, dict):
            raise TypeError(f"loss_kwargs 必须是 dict 或 None，收到 {type(loss_kwargs).__name__}")
        self.loss_kwargs = dict(loss_kwargs or {})

        # 自动推断任务类型
        if task == 'auto':
            last_layer = model.layers[-1]
            self.task = 'classification' if last_layer.act_fn == last_layer._softmax else 'regression'
        else:
            self.task = task

        # 单类分类：softmax 只有 1 个单元时输出恒为 1，CE 的 loss 与梯度恒为 0，
        # 模型永远学不到东西，但 score 会报 100% —— 必须拒绝（sklearn 同样报错）
        if self.task == 'classification' and model.layers[-1].fan_out < 2:
            raise ValueError(
                f"分类模型的输出维度必须 >= 2（当前 {model.layers[-1].fan_out}）；"
                "单类 softmax 的交叉熵恒为 0，模型无法学习"
            )

        # 损失函数
        if loss_fn is None:
            loss_fn = 'cross_entropy' if self.task == 'classification' else 'mse'
        if isinstance(loss_fn, str):
            key = loss_fn.lower()
            if key not in self._LOSS_MAP:
                raise ValueError(f"未知损失函数: {loss_fn}，可选 {list(self._LOSS_MAP.keys())}")
            self._loss_name = key
            self.loss_fn = self._LOSS_MAP[key]
        else:
            self._loss_name = 'custom'
            self.loss_fn = loss_fn

        # 构建带 target 规范化的损失函数
        kw = self.loss_kwargs

        def _as_numeric_target(target):
            """标量目标包成 [target]；list/tuple 原样复制（向量/概率向量目标）"""
            if isinstance(target, (list, tuple)):
                return list(target)
            return [target]

        if self._loss_name in self._INDEX_LOSSES:
            # 标量 / float / 长度 1 的列表归一化成类别索引；
            # 长度等于类数的 one-hot / 概率向量直接交给 CE 的软标签分支，
            # 不再 argmax 硬化（否则软目标、label smoothing、蒸馏信息全被丢弃）
            num_classes = model.layers[-1].fan_out

            def _wrap_index_loss(pred, target):
                idx = target
                if isinstance(target, (list, tuple)):
                    if len(target) == num_classes:
                        # focal 是按类别索引定义的（没有软标签形式），one-hot 先硬化成索引；
                        # 真正的软目标（概率向量）只交给支持软标签的 CE 家族
                        if self._loss_name == 'focal':
                            ones = [j for j, v in enumerate(target) if float(v) == 1.0]
                            zeros_ok = all(
                                float(v) == 0.0 for j, v in enumerate(target) if j not in ones
                            )
                            if len(ones) != 1 or not zeros_ok:
                                raise ValueError(
                                    "focal_loss 不支持软标签：target 必须是类别索引或严格 one-hot，"
                                    f"收到 {list(target)}"
                                )
                            return self.loss_fn(pred, ones[0], **kw)
                        # 1 类分类器：[1.0] 是唯一的 one-hot，也必须按软标签走，
                        # 否则会被当成索引 1 而越界
                        if num_classes == 1:
                            return self.loss_fn(pred, [float(target[0])], **kw)
                        return self.loss_fn(pred, target, **kw)
                    if len(target) != 1:
                        raise ValueError(
                            f"{self._loss_name} 的 target 长度 {len(target)} 与模型输出维度 "
                            f"{num_classes} 不一致（应传类别索引、one-hot 或长度 1 的列表）"
                        )
                    idx = target[0]
                if isinstance(idx, float):
                    if not idx.is_integer():
                        raise ValueError(
                            f"{self._loss_name} 的 target 必须是整数类别索引，收到 {idx}"
                        )
                    idx = int(idx)
                if not isinstance(idx, int):
                    try:
                        idx = operator.index(idx)  # 兼容 numpy 等实现了 __index__ 的整数类型
                    except TypeError:
                        raise TypeError(
                            f"{self._loss_name} 的 target 必须是类别索引或 one-hot 列表，收到 {target!r}"
                        )
                if not 0 <= idx < num_classes:
                    raise ValueError(
                        f"类别索引 {idx} 越界（模型输出维度 {num_classes}）"
                    )
                return self.loss_fn(pred, idx, **kw)

            self._loss = _wrap_index_loss
        elif self._loss_name in self._NUMERIC_LOSSES:
            def _wrap_numeric_loss(pred, target):
                return self.loss_fn(pred, _as_numeric_target(target), **kw)
            self._loss = _wrap_numeric_loss
        else:
            if kw:
                def _wrap_custom_loss(pred, target):
                    return self.loss_fn(pred, target, **kw)
                self._loss = _wrap_custom_loss
            else:
                self._loss = self.loss_fn  # custom 原样

        # 输出层激活函数与损失函数的搭配校验：
        # Layer._softmax_deriv 返回全 1，隐含假设是「softmax + cross_entropy 联合梯度」，
        # 与 MSE 组合会得到静默错误的梯度，因此这里直接报错而不是让它训歪。
        # 只拦截明确已知的 mse；自定义 callable 无法判断其内部实现，交给用户自己负责。
        last_layer = model.layers[-1]
        last_act = getattr(last_layer, "activation_name", None)
        if last_act is None and last_layer.act_fn == last_layer._softmax:
            last_act = "softmax"  # 兼容没有 activation_name 属性的旧对象
        # 传函数对象（loss_fn=mse_loss）时 _loss_name 是 'custom'，
        # 也要落到同一个守卫里，否则两条路径行为不一致
        _is_mse = (self._loss_name == 'mse') or (self.loss_fn is mse_loss) \
            or (getattr(self.loss_fn, 'func', None) is mse_loss)
        if self._loss_name in self._INDEX_LOSSES and last_act != "softmax":
            raise ValueError(
                f"输出层激活是 {last_act!r}，但 loss_fn='{self._loss_name}'："
                "内置交叉熵假设 pred 是 softmax 概率，并直接使用 softmax+CE（或 focal）的联合梯度，"
                "与非 softmax 输出层不匹配（算出的 loss 与梯度不自洽）。"
                "请把输出层激活设为 'softmax'，或改用自定义 loss_fn。"
            )
        # 其余损失对「输出层激活」也有硬性要求，配错会在第一批就抛错或静默训歪，
        # 因此在构造时就拒绝（与上面 CE/mse 的守卫同一套风格）
        _LOSS_ACT_C = {
            'bce': ('sigmoid', "bce_loss 期望 pred 是概率（[0,1]），输出层激活请用 'sigmoid'"),
            'bce_with_logits': ('linear', "bce_with_logits_loss 期望 logits，输出层激活请用 'linear'"),
            'l1': ('linear', "l1_loss 是回归损失，输出层激活请用 'linear'"),
            'log_cosh': ('linear', "log_cosh_loss 是回归损失，输出层激活请用 'linear'"),
            'quantile': ('linear', "quantile_loss 是回归损失，输出层激活请用 'linear'"),
            'smooth_l1': ('linear', "smooth_l1_loss 是回归损失，输出层激活请用 'linear'"),
            'class_balanced': ('softmax', "class_balanced_loss 期望 softmax 概率，输出层激活请用 'softmax'"),
            'ldam': ('linear', "ldam_loss 直接吃 logits，输出层激活请用 'linear'"),
            'distill': ('linear', "distill_loss 吃学生 logits，输出层激活请用 'linear'"),
        }
        want_act = _LOSS_ACT_C.get(self._loss_name)
        if want_act is not None and last_act != want_act[0]:
            raise ValueError(
                f"输出层激活是 {last_act!r}，但 loss_fn='{self._loss_name}'："
                f"{want_act[1]}。"
            )

        if last_act == "softmax" and _is_mse:
            raise ValueError(
                "输出层使用 softmax 时不能搭配 mse 损失："
                "softmax 的反向用的是 softmax+cross_entropy 的联合梯度，"
                "与 mse 组合会静默产生错误梯度。"
                "请改用 loss_fn='cross_entropy'，或把输出层激活改成 'linear'。"
            )

        # 优化器
        if optimizer is None:
            optimizer = 'adam'
        if isinstance(optimizer, str):
            key = optimizer.lower()
            if key not in self._OPTIMIZER_MAP:
                raise ValueError(f"未知优化器: {optimizer}，可选 {list(self._OPTIMIZER_MAP.keys())}")
            self.optimizer = self._OPTIMIZER_MAP[key](model, lr=lr, **self.opt_kwargs)
        else:
            # 优化器实例必须绑定到同一个模型：否则 fit 会更新**别人的**参数，
            # 本模型的 loss 从头到尾不变（静默训练错对象，最隐蔽的一类错误）
            owner = getattr(optimizer, 'model', None)
            if owner is not None and owner is not model:
                raise ValueError(
                    "Model: 传入的优化器绑定的不是这个 model（例如 Adam(A) 传给了 Model(B)）。"
                    "优化器只更新自己持有的 model 参数，这样训练会静默地改错对象；"
                    "请传字符串（如 'adam'）或为本 model 新建优化器。"
                )
            self.optimizer = optimizer

        self.history = {'train_loss': [], 'val_loss': []}
        self._is_trained = False

# ── 核心操作

    def forward(self, x):
        """前向传播：接受单样本 list，返回输出 list"""
        return self.model.forward(x if isinstance(x, list) else list(x))

    def backward(self, loss_grad):
        """反向传播：返回 [(gw, gb), ...] 梯度列表（逐层）"""
        return _mlp_backward(self.model, loss_grad)

    def register_forward_hook(self, index: int, fn):
        """转发到 MLP：在第 index 层前向后调用 fn(layer, output)"""
        return self.model.register_forward_hook(index, fn)

    def register_backward_hook(self, index: int, fn):
        """转发到 MLP：在第 index 层反向后调用 fn(layer, delta, grad_w, grad_b)"""
        return self.model.register_backward_hook(index, fn)

    def clear_hooks(self):
        """转发到 MLP：注销全部 hook"""
        return self.model.clear_hooks()

# ── 训练

    def _load_checkpoint(self, path):
        """从 checkpoint 恢复权重与优化器动量，返回恢复的 step_count。"""
        if not isinstance(path, str) or not path:
            raise ValueError(f"fit: resume_from 必须是文件路径字符串，收到 {path!r}")
        try:
            with io.open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"fit: resume_from 文件不存在: {path}")
        except (ValueError, UnicodeDecodeError) as exc:
            raise ValueError(f"fit: resume_from 解析失败（{exc}）: {path}")
        if not isinstance(data, dict) or data.get("type") != "MLP":
            raise ValueError(f"fit: resume_from 的文件不是 MLP 存档: {path}")
        saved = data.get("layers")
        if not isinstance(saved, list) or len(saved) != len(self.model.layers):
            raise ValueError(
                f"fit: resume_from 的层数与当前模型不一致"
                f"（存档 {len(saved) if isinstance(saved, list) else '?'} 层，"
                f"当前 {len(self.model.layers)} 层）"
            )
        for i, (layer, ld) in enumerate(zip(self.model.layers, saved)):
            if not isinstance(ld, dict):
                raise ValueError(f"fit: resume_from 第 {i} 层不是对象")
            w, b = ld.get("weights"), ld.get("biases")
            if (not isinstance(w, list) or len(w) != layer.fan_out
                    or any(not isinstance(row, list) or len(row) != layer.fan_in for row in w)):
                raise ValueError(
                    f"fit: resume_from 第 {i} 层权重形状与当前模型不符"
                    f"（应为 {layer.fan_out}×{layer.fan_in}）"
                )
            if not isinstance(b, list) or len(b) != layer.fan_out:
                raise ValueError(
                    f"fit: resume_from 第 {i} 层偏置形状与当前模型不符"
                    f"（应为 {layer.fan_out} 个）"
                )
            for j in range(layer.fan_out):
                layer.weights[j] = [float(v) for v in w[j]]
            layer.biases[:] = [float(v) for v in b]
        state = data.get("optimizer")
        if isinstance(state, dict):
            load_optimizer_state(self.optimizer, state)
            return int(state.get("step_count", 0))
        return 0

    def _mixup_batch(self, batch, alpha, rng):
        """把一个 batch 变成 mixup 后的 (x_mix, y_soft) 列表（长度不变）。"""
        num_classes = self.model.layers[-1].fan_out

        def _label_vec(y, k):
            if isinstance(y, (list, tuple)):
                if len(y) != num_classes:
                    raise ValueError(
                        f"fit(mixup_alpha): 第 {k} 个样本的软标签长度 {len(y)}"
                        f" 与输出维度 {num_classes} 不一致"
                    )
                vec = [float(v) for v in y]
                total = sum(vec)
                if abs(total - 1.0) > 1e-6:
                    raise ValueError(
                        f"fit(mixup_alpha): 第 {k} 个样本的软标签和为 {total:.6f}，应为 1"
                    )
                return vec
            if isinstance(y, bool) or not isinstance(y, int):
                raise ValueError(
                    f"fit(mixup_alpha): 目标必须是类别索引或 one-hot 向量，"
                    f"第 {k} 个样本收到 {type(y).__name__}"
                )
            if not (0 <= y < num_classes):
                raise ValueError(
                    f"fit(mixup_alpha): 第 {k} 个样本的类别索引 {y} 越界"
                    f"（应为 0..{num_classes - 1}）"
                )
            vec = [0.0] * num_classes
            vec[y] = 1.0
            return vec

        xs = [list(x) for x, _ in batch]
        ys = [_label_vec(y, k) for k, (_, y) in enumerate(batch)]
        n = len(xs)
        mixed = []
        for k in range(n):
            j = rng.randrange(n)
            lam = _beta_lam(alpha, rng, "fit(mixup_alpha)")
            x1, x2 = xs[k], xs[j]
            y1, y2 = ys[k], ys[j]
            xm = [lam * a + (1.0 - lam) * b for a, b in zip(x1, x2)]
            ym = [lam * a + (1.0 - lam) * b for a, b in zip(y1, y2)]
            mixed.append((xm, ym))
        return mixed

    def _normalize_val_data(self, val_data):
        """把 val_data 规范化为 [(x, y), ...]，并做维度校验。

        支持的写法：
            {'x': x_arr, 'y': y_arr}   显式并行数组（推荐，无歧义）
            (x_arr, y_arr)             并行数组
            [(x, y), ...]              样本对列表
        歧义处理：恰好 2 个元素且两种解释都成立时（无法从结构上区分），
        按容器类型决定——list 视为样本对，tuple 视为并行数组，与上面两条写法一致。
        拿不准时请用 dict 写法。
        """
        def as_pairs(x_arr, y_arr):
            x_arr, y_arr = list(x_arr), list(y_arr)
            if len(x_arr) != len(y_arr):
                raise ValueError(
                    f"val_data: x 有 {len(x_arr)} 个样本，y 有 {len(y_arr)} 个，数量必须一致"
                )
            return [(list(xi), yi) for xi, yi in zip(x_arr, y_arr)]

        def looks_like_parallel_arrays(vd):
            return (isinstance(vd, (list, tuple)) and len(vd) == 2
                    and isinstance(vd[0], (list, tuple)) and isinstance(vd[1], (list, tuple))
                    and len(vd[0]) == len(vd[1]) and len(vd[0]) > 0
                    and all(isinstance(e, (list, tuple)) for e in vd[0]))

        def looks_like_pairs(vd):
            return (isinstance(vd, (list, tuple)) and len(vd) > 0
                    and all(isinstance(it, (list, tuple)) and len(it) == 2
                            and isinstance(it[0], (list, tuple)) for it in vd))

        if isinstance(val_data, (list, tuple, dict)) and len(val_data) == 0:
            return []  # 空验证集 → 等同不做验证（保持旧行为）

        if isinstance(val_data, dict):
            x_arr = val_data.get('x', val_data.get('X'))
            y_arr = val_data.get('y', val_data.get('Y'))
            if x_arr is None or y_arr is None:
                raise ValueError("val_data 为 dict 时必须提供 'x' 和 'y' 两个键")
            pairs = as_pairs(x_arr, y_arr)
        elif looks_like_parallel_arrays(val_data) and (
                not looks_like_pairs(val_data) or isinstance(val_data, tuple)):
            pairs = as_pairs(val_data[0], val_data[1])
        elif looks_like_pairs(val_data):
            pairs = [(list(xi), yi) for xi, yi in val_data]
        else:
            raise ValueError(
                "val_data 格式无法识别，支持三种写法："
                "{'x': x_arr, 'y': y_arr}、(x_arr, y_arr)、[(x, y), ...]"
            )

        # 维度校验：尽早暴露形状错误，而不是让后面的 zip 静默截断
        n_in = self.model.layers[0].fan_in
        n_out = self.model.layers[-1].fan_out
        for k, (xi, yi) in enumerate(pairs):
            if len(xi) != n_in:
                raise ValueError(
                    f"val_data 第 {k} 个样本的特征维度是 {len(xi)}，模型输入维度是 {n_in}"
                )
            if isinstance(yi, (list, tuple)) and len(yi) != n_out:
                raise ValueError(
                    f"val_data 第 {k} 个样本的 target 维度是 {len(yi)}，模型输出维度是 {n_out}"
                )
        return pairs

    def fit(self, x=None, y=None, data=None, epochs=100, batch_size=32,
            val_split=0.1, val_data=None, patience=50, verbose=True,
            shuffle=True, lr_scheduler=None, grad_clip=None,
            early_stop_metric='val', restore_best_weights=False,
            checkpoint_path=None, checkpoint_every=1, grad_accum: int = 1,
            seed=None, callbacks=None, resume_from=None, mixup_alpha: float = 0.0,
            record_layer_norms: bool = False):
        """
        训练模型。

        数据传入方式（二选一）：
            model.fit(x=[[1,2],[3,4]], y=[0, 1])        ← 分开传
            model.fit(data=[([1,2],0), ([3,4],1)])       ← 元组列表
            (data 优先级高于 x/y)

        epochs:       训练轮数
        batch_size:   批次大小（设为 -1 = 全批量）
        val_split:    验证集比例（仅 val_data 未提供时从训练集切分）
        val_data:     自定义验证集 {'x':…, 'y':…} / (x_arr, y_arr) / [(x, y), ...]
        patience:     Early stopping 耐心值（0 = 不使用）
        verbose:      打印进度
        shuffle:      每轮打乱数据
        lr_scheduler: StepLR/ExponentialLR/CosineAnnealingLR 实例
        grad_clip:    float，梯度范数裁剪阈值（None = 不裁剪）
            early_stop_metric: 'val'（默认，盯验证损失）或 'train'（盯训练损失，且不依赖验证集）
            restore_best_weights: True 时在训练结束后回滚到「被监控指标最好」的那一轮权重
                （早停只在指标变差时停止，**不会自动回滚**；需要最优权重时打开这个开关）
            checkpoint_path: 每 checkpoint_every 轮把权重 + 优化器状态写到该路径
                （中断后可用 Model.load(path, restore_optimizer=True) 续训）
            seed:         打乱数据的随机种子（None = 用全局 random，与旧版行为一致）。
                传了整数就改用局部 Random：再跑一次同样的 fit 结果完全相同，
                也不会推进调用方的全局随机状态。
            grad_accum:   梯度累积步数（1 = 每个 batch 更新一次）。大于 1 时先累加
                grad_accum 个 batch 的**平均梯度**，再更新一次；代价是更新次数变少，
                收益是等效 batch 更大而显存不变。每个 epoch 末尾会把剩余不足
                grad_accum 的 batch 也更新一次（不会白算）。
            callbacks:    训练回调列表（继承 AI.Callback，钩子见该类 docstring）
            resume_from:  checkpoint 路径：训练前先恢复权重与优化器动量（含 step_count），
                再用给定的 epochs 继续训练（epoch 编号从 1 重新开始）
            mixup_alpha:  >0 时对每个 batch 做 mixup（仅分类任务、仅索引型损失），
                lam ~ Beta(alpha, alpha)；0（默认）表示关闭，关闭时数值完全不变
            record_layer_norms: True 时把每层梯度范数记进 history['layer_grad_norm']
        """
        # === 规范化数据 ===
        if data is not None:
            dataset = [(list(x_i), y_i) for x_i, y_i in data]
            if not dataset:
                raise ValueError("fit: 训练集为空")
        elif x is not None and y is not None:
            # 训练集同样要校验：zip 会静默截断到较短的一方
            # 标量 y（如 y=0.5）广播到每个样本，与 train(data=[(x, 0.5)]) 的语义一致
            if isinstance(y, (int, float)) and not isinstance(y, bool):
                y = [y] * len(x)
            if len(x) != len(y):
                raise ValueError(
                    f"fit: x 有 {len(x)} 个样本，y 有 {len(y)} 个，数量必须一致"
                )
            if len(x) == 0:
                raise ValueError("fit: 训练集为空")
            n_in = self.model.layers[0].fan_in
            if len(x[0]) != n_in:
                raise ValueError(
                    f"fit: 训练样本特征维度 {len(x[0])} 与模型输入维度 {n_in} 不一致"
                )
            dataset = [(list(x_i), y_i) for x_i, y_i in zip(x, y)]
        else:
            raise ValueError("请提供 data 或 (x, y)")

        # 复校验：optimizer.model 可能在构造之后被替换，step 前再确认一次
        _owner = getattr(self.optimizer, 'model', None)
        if _owner is not None and _owner is not self.model:
            raise ValueError(
                "fit: 当前优化器绑定的不是 self.model（被替换过？）；"
                "请重建优化器，否则会更新错误对象的参数"
            )

        if early_stop_metric not in ('val', 'train'):
            raise ValueError(
                f"fit: early_stop_metric 只能是 'val' 或 'train'，收到 {early_stop_metric!r}"
            )
        if not isinstance(val_split, (int, float)) or not (0.0 <= val_split < 1.0):
            raise ValueError(
                f"fit: val_split 必须在 [0, 1) 内，收到 {val_split!r}"
                "（与 split_data 的校验保持一致；原实现会把 >=1 静默钳成 n-1）"
            )

        rng = random.Random(seed) if seed is not None else random

        # === 验证集 ===
        if val_data is not None:
            val_set = self._normalize_val_data(val_data)
        elif val_split > 0:
            n_val = max(1, int(len(dataset) * val_split))
            # 至少要留 1 个训练样本：样本数很少时 max(1,...) 会把训练集切空，
            # 随后 train_loss /= len(dataset) 抛 ZeroDivisionError
            if n_val >= len(dataset):
                n_val = len(dataset) - 1
            if n_val <= 0:
                val_set = None
            elif shuffle:
                rng.shuffle(dataset)
                val_set = dataset[:n_val]
                dataset = dataset[n_val:]
            else:
                # shuffle=False 时不能为了切验证集而打乱训练顺序：取尾部当验证集
                val_set = dataset[-n_val:]
                dataset = dataset[:-n_val]
        else:
            val_set = None

        if batch_size is None or batch_size <= 0:
            batch_size = len(dataset)

        if not isinstance(grad_accum, int) or isinstance(grad_accum, bool) or grad_accum < 1:
            raise ValueError(f"fit: grad_accum 必须是 >=1 的整数，收到 {grad_accum!r}")

        callbacks = _normalize_callbacks(callbacks)

        if not isinstance(mixup_alpha, (int, float)) or isinstance(mixup_alpha, bool) \
                or mixup_alpha != mixup_alpha or mixup_alpha < 0:
            raise ValueError(f"fit: mixup_alpha 必须是 >=0 的数字，收到 {mixup_alpha!r}")
        if mixup_alpha > 0.0:
            if self.task != 'classification':
                raise ValueError(
                    f"fit: mixup_alpha 只支持分类任务（需要 one-hot 软标签），当前 task={self.task!r}"
                )
            if self._loss_name not in self._INDEX_LOSSES:
                raise ValueError(
                    f"fit: mixup_alpha 需要索引型损失（{list(self._INDEX_LOSSES)}），"
                    f"当前 {self._loss_name!r} 不支持软标签"
                )

        self.history = {'train_loss': [], 'val_loss': [], 'grad_norm': []}
        if record_layer_norms:
            self.history['layer_grad_norm'] = []
        best_val_loss = float('inf')
        no_improve = 0
        best_snapshot = None      # restore_best_weights 用的最优权重快照

        _resumed_step = 0
        if resume_from is not None:
            _resumed_step = self._load_checkpoint(resume_from)
            if verbose:
                print(f"已恢复权重与优化器状态：{resume_from}（优化器步数 {_resumed_step}）")

        for _cb in callbacks:
            _cb.on_train_begin(self, {
                'epochs': epochs, 'batch_size': batch_size, 'lr': self.lr,
                'task': self.task, 'n_train': len(dataset),
                'n_val': len(val_set) if val_set else 0,
                'resumed_step': _resumed_step,
            })

        for epoch in range(1, epochs + 1):
            for _cb in callbacks:
                _cb.on_epoch_begin(epoch, {'epoch': epoch, 'epochs': epochs, 'lr': self.lr})
            ln_sum = [0.0] * len(self.model.layers) if record_layer_norms else None
            ln_n = 0
            if shuffle:
                rng.shuffle(dataset)

            train_loss = 0.0
            gn_sum = 0.0   # 本轮各次更新的梯度范数之和（裁剪前）
            gn_n = 0
            pending = None      # grad_accum > 1 时的累加缓冲
            pending_n = 0
            for b in range(0, len(dataset), batch_size):
                batch = dataset[b:b + batch_size]
                if mixup_alpha > 0.0:
                    batch = self._mixup_batch(batch, mixup_alpha, rng)

                # 直接在反向传播时把各样本梯度累加到 buffer：
                #   1) 不再保存 batch 内每个样本的完整梯度（省下 B×参数量 的中间结构）
                #   2) 不再对每个 (层, j, k) 都遍历一遍 batch 求和（那是 O(B) 倍重复）
                acc_w = [[[0.0] * layer.fan_in for _ in range(layer.fan_out)]
                         for layer in self.model.layers]
                acc_b = [[0.0] * layer.fan_out for layer in self.model.layers]

                # 逐样本前向 + 反向
                for xi, yi in batch:
                    pred = self.forward(xi)
                    loss, grad = self._loss(pred, yi)
                    train_loss += loss

                    delta = grad
                    idx = len(self.model.layers) - 1
                    for layer in reversed(self.model.layers):
                        gw, gb, delta = layer.backward(delta)
                        gw_acc, gb_acc = acc_w[idx], acc_b[idx]
                        for j in range(layer.fan_out):
                            gwj, gwaj = gw[j], gw_acc[j]
                            for k in range(layer.fan_in):
                                gwaj[k] += gwj[k]
                            gb_acc[j] += gb[j]
                        idx -= 1

                # 平均梯度（累加顺序与逐样本求和一致，数值与原来完全等价）
                B = len(batch)
                avg_grads = [
                    ([[v / B for v in row] for row in acc_w[i]],
                     [v / B for v in acc_b[i]])
                    for i in range(len(self.model.layers))
                ]

                if grad_accum == 1:
                    # 记录梯度范数（裁剪前，用来判断训练是否发散/爆炸）
                    gn_sum += _grads_norm(avg_grads)
                    gn_n += 1
                    if ln_sum is not None:
                        _accum_layer_norms(ln_sum, avg_grads)
                        ln_n += 1

                    # 梯度裁剪
                    if grad_clip is not None:
                        clip_grad_by_norm(avg_grads, max_norm=grad_clip)

                    self.optimizer.step(avg_grads)
                else:
                    # 先累加各 batch 的**平均梯度**（等价于把 batch 拼大，
                    # 但不需要同时把它们的前向缓存都留在内存里）
                    if pending is None:
                        pending = [([row[:] for row in gw], gb[:])
                                   for gw, gb in avg_grads]
                    else:
                        for (pw, pb), (gw, gb) in zip(pending, avg_grads):
                            for j in range(len(gw)):
                                pwj, gwj = pw[j], gw[j]
                                for k in range(len(gwj)):
                                    pwj[k] += gwj[k]
                                pb[j] += gb[j]
                    pending_n += 1
                    if pending_n >= grad_accum:
                        merged = [
                            ([[v / pending_n for v in row] for row in pw],
                             [v / pending_n for v in pb])
                            for pw, pb in pending
                        ]
                        gn_sum += _grads_norm(merged)
                        gn_n += 1
                        if ln_sum is not None:
                            _accum_layer_norms(ln_sum, merged)
                            ln_n += 1
                        if grad_clip is not None:
                            clip_grad_by_norm(merged, max_norm=grad_clip)
                        self.optimizer.step(merged)
                        pending = None
                        pending_n = 0

            # 收尾：把本 epoch 剩下的不足 grad_accum 个 batch 也更新掉
            if grad_accum > 1 and pending is not None and pending_n > 0:
                merged = [
                    ([[v / pending_n for v in row] for row in pw],
                     [v / pending_n for v in pb])
                    for pw, pb in pending
                ]
                gn_sum += _grads_norm(merged)
                gn_n += 1
                if ln_sum is not None:
                    _accum_layer_norms(ln_sum, merged)
                    ln_n += 1
                if grad_clip is not None:
                    clip_grad_by_norm(merged, max_norm=grad_clip)
                self.optimizer.step(merged)
                pending = None
                pending_n = 0

            train_loss /= len(dataset)
            self.history['train_loss'].append((epoch, train_loss))
            self.history['grad_norm'].append((epoch, gn_sum / max(gn_n, 1)))
            if record_layer_norms:
                self.history['layer_grad_norm'].append(
                    (epoch, [v / max(ln_n, 1) for v in ln_sum])
                )

            # 验证
            val_loss = None
            if not val_set and verbose and patience > 0 and \
                    early_stop_metric != 'train' and epoch == 1:
                print("提示: 没有验证集，early_stop_metric='val' 不会触发早停"
                      "（可用 val_split 或 early_stop_metric='train'）")
            if val_set:
                val_loss = self.evaluate(data=val_set)
                self.history['val_loss'].append((epoch, val_loss))

            for _cb in callbacks:
                _cb.on_epoch_end(epoch, {
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'grad_norm': gn_sum / max(gn_n, 1),
                    'layer_grad_norm': None if ln_sum is None
                    else [v / max(ln_n, 1) for v in ln_sum],
                    'lr': self.lr,
                })
            if any(getattr(_cb, 'should_stop', False) for _cb in callbacks):
                if verbose:
                    print("回调请求停止训练（EarlyStopping）")
                break

            # 早停判据：默认盯验证损失（需要验证集）；
            # early_stop_metric='train' 时改盯训练损失，此时不依赖验证集
            # （与 C 引擎 FitConfig.early_stop_on_train 同一语义）
            # 被监控指标：'train' 用训练损失，否则用验证损失（没有验证集则不监控）。
            # 注意：快照与「是否早停」解耦——patience=0 只关闭早停，不关闭最优权重回滚。
            if early_stop_metric == 'train':
                monitored = train_loss
            elif val_set:
                monitored = val_loss
            else:
                monitored = None
            if monitored is not None:
                if monitored < best_val_loss - 1e-12:
                    best_val_loss = monitored
                    no_improve = 0
                    if restore_best_weights:
                        best_snapshot = [
                            ([row[:] for row in l.weights], l.biases[:])
                            for l in self.model.layers
                        ]
                else:
                    no_improve += 1
                if patience > 0 and no_improve >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch}")
                    break

            # 学习率调度
            if lr_scheduler is not None:
                lr_scheduler.step(epoch)

            # 落盘 checkpoint（含优化器动量，可精确续训）
            if checkpoint_path is not None and epoch % max(int(checkpoint_every), 1) == 0:
                save_model(self.model, checkpoint_path, task=self.task,
                           optimizer=self.optimizer, quiet=True)

            # 打印
            if verbose and epoch % max(1, epochs // 10) == 0:
                parts = [f"Epoch {epoch}/{epochs}", f"loss={train_loss:.6f}"]
                if val_loss is not None:
                    parts.append(f"val_loss={val_loss:.6f}")
                print(" | ".join(parts))

        if restore_best_weights and best_snapshot is not None:
            # 回滚到被监控指标最好的那一轮（早停本身不会回滚权重）
            for layer, (w, b) in zip(self.model.layers, best_snapshot):
                for j in range(layer.fan_out):
                    layer.weights[j][:] = w[j]
                layer.biases[:] = b
            if verbose:
                print(f"已回滚到最优权重（monitored={best_val_loss:.6f}）")

        for _cb in callbacks:
            _cb.on_train_end(self.history)

        # epochs=0 时什么都没训练，不能标记为已训练
        self._is_trained = epochs > 0
        return self.history

# ── 推理

    def predict(self, x):
        """预测：单样本返回 list，多样本返回 list[list]"""
        if len(x) > 0 and isinstance(x[0], (list, tuple)):
            return [self.forward(list(xi)) for xi in x]
        return self.forward(list(x))

    def predict_classes(self, x):
        """分类任务专用：返回预测类别索引（单样本 int / 多样本 list[int]）"""
        probs = self.predict(x)
        if isinstance(probs[0], list):
            return [max(range(len(p)), key=lambda i: p[i]) for p in probs]
        return max(range(len(probs)), key=lambda i: probs[i])

# ── 评估

    def evaluate(self, x=None, y=None, data=None):
        """计算平均损失"""
        if x and not isinstance(x[0], (list, tuple)):
            raise ValueError(
                "evaluate: x 必须是样本列表（每个样本是一个序列）；单个样本请写成 [x]"
            )
        if data is not None:
            dataset = [(list(xi), yi) for xi, yi in data]
        elif x is not None and y is not None:
            if len(x) != len(y):
                raise ValueError(
                    f"evaluate: x 有 {len(x)} 个样本，y 有 {len(y)} 个，数量必须一致"
                )
            dataset = [(list(xi), yi) for xi, yi in zip(x, y)]
        else:
            raise ValueError("请提供 data 或 (x, y)")
        if not dataset:
            raise ValueError("evaluate: 数据集为空")

        total = 0.0
        for xi, yi in dataset:
            pred = self.forward(xi)
            loss, _ = self._loss(pred, yi)
            total += loss
        return total / len(dataset)

    def score(self, x, y):
        """分类：准确率 | 回归：R² 决定系数"""
        if x and not isinstance(x[0], (list, tuple)):
            raise ValueError(
                "score: x 必须是样本列表（每个样本是一个序列）；单个样本请写成 [x]"
            )
        if len(x) != len(y):
            raise ValueError(
                f"score: x 有 {len(x)} 个样本，y 有 {len(y)} 个，数量必须一致"
            )
        if not x:
            raise ValueError("score: 数据集为空")
        if self.task == 'classification':
            preds = self.predict_classes(x)
            # y 可能是类别索引，也可能是 one-hot / 概率分布，统一取 argmax 后再比
            targets = [
                max(range(len(yi)), key=lambda i: yi[i]) if isinstance(yi, (list, tuple)) else yi
                for yi in y
            ]
            correct = sum(1 for p, t in zip(preds, targets) if p == t)
            return correct / len(y)
        else:
            preds = self.predict(x)
            # 多输出回归：对每个输出维度分别算 R² 再取平均（单输出时与原来完全一致，
            # 原来是只取第 0 维、静默忽略其余维度）
            n_out = len(preds[0]) if preds and isinstance(preds[0], (list, tuple)) else 1
            if n_out > 1 and not isinstance(y[0], (list, tuple)):
                raise ValueError(
                    f"score: 模型是 {n_out} 维输出，但 y 是标量；"
                    f"请传同维度的目标（多输出不能只对一个维度打分）"
                )
            scores = []
            for d in range(n_out):
                y_d = [(yi if not isinstance(yi, (list, tuple)) else yi[d]) for yi in y]
                p_d = [(pi if not isinstance(pi, (list, tuple)) else pi[d]) for pi in preds]
                ss_res = sum((a - b) ** 2 for a, b in zip(y_d, p_d))
                y_mean = sum(y_d) / len(y_d)
                ss_tot = sum((a - y_mean) ** 2 for a in y_d)
                if ss_tot == 0.0:
                    # 目标恒定：完全预测对是 1，否则 0（原先 max(ss_tot,1e-15)
                    # 会给出 -1e15 量级的荒谬值）
                    scores.append(1.0 if ss_res == 0.0 else 0.0)
                else:
                    scores.append(1 - ss_res / ss_tot)
            return sum(scores) / len(scores)

# ── 评估指标

    def evaluate_all(self, x, y):
        """一次算出常用指标：回归给 mae/rmse/mape/r2，分类给 accuracy/precision/recall/f1、
        confusion_matrix、mcc，二分类另给 roc_auc。详见模块级 `evaluate_all`。"""
        return evaluate_all(self, x, y)

# ── 持久化

    def save(self, filepath):
        """保存模型到 JSON 文件（Adam/AdamW 会连动量与步数一起存）

        除结构与权重外，还会记下损失函数名与 `loss_kwargs`（用 focal / Focal 的
        gamma、CE 的 label_smoothing 训练时，重新加载后不会悄悄退回默认损失）。
        """
        save_model(self.model, filepath, task=self.task, optimizer=self.optimizer,
                   loss=None if self._loss_name == 'custom' else self._loss_name,
                   loss_kwargs=self.loss_kwargs or None)

    @staticmethod
    def load(filepath, loss_fn=None, optimizer='adam', lr=0.01,
             task='auto', default_output_activation=None,
             restore_optimizer=False, loss_kwargs=None):
        """从 JSON 加载，返回 Model 实例

        default_output_activation:
            仅对旧格式（无逐层 activation）文件生效，用来还原旧的 softmax 分类器；
            不传时其输出层会退回 linear，从而被当成回归模型。
        task:
            'auto'（默认）时优先采用文件中记录的 task，其次按输出层激活推断。
        restore_optimizer:
            True 时把文件里的 Adam/AdamW 动量与 step_count 恢复进新优化器，
            这样接着 fit 就是「精确续训」而不是重新热身；文件里没有该字段时忽略。
        loss_fn / loss_kwargs:
            不传时优先采用文件里记录的值（save 会写入），这样用 focal(gamma=…)
            或 CE(label_smoothing=…) 训练的模型重新加载后损失配置不变；
            显式传入则覆盖文件里的值。旧文件没有这些字段时退回默认。
        """
        mlp = load_model(filepath, default_output_activation=default_output_activation)
        if task == 'auto':
            saved = getattr(mlp, "saved_task", None)
            if saved in ('regression', 'classification'):
                task = saved
        # 输出层是 softmax 时只能是分类（softmax + mse 会触发守卫）。
        # 文件里记录的 task 若与激活矛盾，以激活为准，避免自己写出的文件反而加载失败。
        last_act = getattr(mlp.layers[-1], "activation_name", None)
        if last_act == "softmax" and task != 'classification':
            print("提示: 输出层是 softmax，按分类加载（文件里的 task 与输出层激活矛盾）")
            task = 'classification'
        if loss_fn is None:
            saved_loss = getattr(mlp, "saved_loss", None)
            if isinstance(saved_loss, str) and saved_loss in Model._LOSS_MAP:
                loss_fn = saved_loss
        if loss_kwargs is None:
            saved_kw = getattr(mlp, "saved_loss_kwargs", None)
            if isinstance(saved_kw, dict):
                loss_kwargs = saved_kw
        obj = Model(mlp, loss_fn=loss_fn, optimizer=optimizer, lr=lr, task=task,
                    loss_kwargs=loss_kwargs)
        if restore_optimizer:
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    raw = json.load(f)
            except (OSError, ValueError):
                raw = {}
            state = raw.get("optimizer")
            if state is not None:
                load_optimizer_state(obj.optimizer, state)
                if getattr(obj.optimizer, "step_count", 1) > 1:
                    print(f"已恢复优化器状态（step_count={obj.optimizer.step_count}）")
        return obj

# ── 信息

    def summary(self):
        """打印模型结构摘要"""
        print(f"{'='*50}")
        print(f"  Model ({self.task})")
        print(f"  Loss: {self._loss_name}  |  Optimizer: {type(self.optimizer).__name__}(lr={self.lr})")
        print(f"  {'='*50}")
        total = 0
        _ACT_DISPLAY = {
            'leaky_relu': 'LeakyReLU', 'relu': 'ReLU', 'softmax': 'Softmax',
            'linear': 'Linear', 'gelu': 'GELU', 'sigmoid': 'Sigmoid',
        }
        for i, layer in enumerate(self.model.layers):
            # 优先用 Layer 自己记下的名字：gelu/sigmoid 原来会被显示成 Linear
            act = _ACT_DISPLAY.get(getattr(layer, 'activation_name', None))
            if act is None:
                if layer.act_fn == layer._leaky_relu:
                    act = 'LeakyReLU'
                elif layer.act_fn == layer._relu:
                    act = 'ReLU'
                elif layer.act_fn == layer._softmax:
                    act = 'Softmax'
                elif layer.act_fn == getattr(layer, '_gelu', None):
                    act = 'GELU'
                elif layer.act_fn == getattr(layer, '_sigmoid', None):
                    act = 'Sigmoid'
                else:
                    act = 'Linear'
            params = layer.fan_in * layer.fan_out + layer.fan_out
            total += params
            print(f"  [{i}] Linear({layer.fan_in} → {layer.fan_out}) + {act}   [{params:,}]")
        print(f"  {'='*50}")
        print(f"  Total params: {total:,}")

    def __repr__(self):
        return f"Model(task={self.task}, loss={self._loss_name}, " \
               f"optimizer={type(self.optimizer).__name__}, lr={self.lr})"


# ============================================================================
#  生成模型：VAE / GAN / DDPM
# ============================================================================

def _gauss_vec(n, rng):
    r = rng or random
    return [r.gauss(0.0, 1.0) for _ in range(n)]


def _time_features(t, steps, dim=8):
    """把扩散步 t 编成 sinusoidal 特征（与位置编码同思路）。"""
    out = []
    for i in range(dim // 2):
        freq = 1.0 / (10000.0 ** (2.0 * i / max(1, dim)))
        out.append(math.sin(t * freq))
        out.append(math.cos(t * freq))
    return out


class VAE:
    """变分自编码器（MLP encoder/decoder，2 维潜空间常用来做可视化）

    损失 = 重构 MSE（对维度取平均） + beta * KL(q(z|x) || N(0,1))，
    KL 按维求和、按样本取平均。重参数化 z = mu + eps*exp(logvar/2)。
    """

    def __init__(self, input_dim: int, hidden_dim: int = 32, latent_dim: int = 2,
                 beta: float = 1.0):
        for name, value in (("input_dim", input_dim), ("hidden_dim", hidden_dim),
                            ("latent_dim", latent_dim)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"VAE: {name} 必须是 >= 1 的整数")
        if not isinstance(beta, (int, float)) or isinstance(beta, bool) or beta < 0:
            raise ValueError("VAE: beta 必须是 >= 0 的数")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.beta = float(beta)
        self.enc1 = Layer(input_dim, hidden_dim, activation="relu")
        self.enc2 = Layer(hidden_dim, 2 * latent_dim, activation="linear")
        self.dec1 = Layer(latent_dim, hidden_dim, activation="relu")
        self.dec2 = Layer(hidden_dim, input_dim, activation="linear")
        self.layers = [self.enc1, self.enc2, self.dec1, self.dec2]
        self.step_count = 0

    def _zeros(self):
        return [None] * len(self.layers)

    def zero_grad(self):
        self._g = self._zeros()

    def _acc(self, i, gw, gb):
        if self._g[i] is None:
            self._g[i] = [gw, gb]
        else:
            cur_w, cur_b = self._g[i]
            for r in range(len(cur_w)):
                row_c, row_n = cur_w[r], gw[r]
                for c in range(len(row_c)):
                    row_c[c] += row_n[c]
            for k in range(len(cur_b)):
                cur_b[k] += gb[k]

    def encode(self, x):
        h = self.enc1.forward(x)
        out = self.enc2.forward(h)
        mu = out[:self.latent_dim]
        logvar = out[self.latent_dim:]
        return mu, logvar

    def decode(self, z):
        return self.dec2.forward(self.dec1.forward(z))

    def forward(self, x, eps=None, rng=None):
        mu, logvar = self.encode(x)
        if eps is None:
            eps = _gauss_vec(self.latent_dim, rng)
        if len(eps) != self.latent_dim:
            raise ValueError(f"VAE.forward: eps 维度应为 {self.latent_dim}")
        std = [math.exp(0.5 * v) for v in logvar]
        z = [mu[i] + eps[i] * std[i] for i in range(self.latent_dim)]
        x_hat = self.decode(z)
        self._x, self._mu, self._logvar, self._eps, self._std, self._z = \
            list(x), mu, logvar, list(eps), std, z
        return x_hat

    def loss_and_backward(self, x, eps=None, rng=None):
        if not isinstance(x, (list, tuple)) or len(x) != self.input_dim:
            raise ValueError(f"VAE.loss_and_backward: x 维度应为 {self.input_dim}")
        x_hat = self.forward(x, eps=eps, rng=rng)
        n = self.input_dim
        recon = sum((x_hat[j] - x[j]) ** 2 for j in range(n)) / n
        kl = 0.5 * sum(self._mu[i] ** 2 + math.exp(self._logvar[i])
                       - 1.0 - self._logvar[i] for i in range(self.latent_dim))
        loss = recon + self.beta * kl

        # 重构项 → decoder
        d_x_hat = [2.0 * (x_hat[j] - x[j]) / n for j in range(n)]
        gw2, gb2, d_h = self.dec2.backward(d_x_hat)
        gw1, gb1, d_z = self.dec1.backward(d_h)
        # 重参数化 + KL → encoder 输出（mu 在前、logvar 在后）
        d_mu = [d_z[i] + self.beta * self._mu[i] for i in range(self.latent_dim)]
        d_logvar = [d_z[i] * self._eps[i] * 0.5 * self._std[i]
                    + self.beta * 0.5 * (math.exp(self._logvar[i]) - 1.0)
                    for i in range(self.latent_dim)]
        gwe, gbe, d_hidden = self.enc2.backward(d_mu + d_logvar)
        gw0, gb0, _ = self.enc1.backward(d_hidden)

        self._acc(0, gw0, gb0)
        self._acc(1, gwe, gbe)
        self._acc(2, gw1, gb1)
        self._acc(3, gw2, gb2)
        return loss

    def fit(self, data, epochs=200, lr=1e-2, batch_size=None, seed=None,
            verbose=False):
        if not data:
            raise ValueError("VAE.fit: data 不能为空")
        if isinstance(lr, bool) or not isinstance(lr, (int, float)) or lr <= 0:
            raise ValueError("VAE.fit: lr 必须为正数")
        if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 0:
            raise ValueError("VAE.fit: epochs 必须是非负整数")
        rng = random.Random(seed) if seed is not None else random
        history = []
        for _epoch in range(epochs):
            total = 0.0
            self.zero_grad()
            for x in data:
                total += self.loss_and_backward(x, eps=_gauss_vec(self.latent_dim, rng))
            self._scale_grads(1.0 / len(data))
            self.adam_step(lr)
            history.append(total / len(data))
        return history

    def _scale_grads(self, factor):
        if factor == 1.0:
            return
        for item in self._g:
            if item is None:
                continue
            gw, gb = item
            for row in gw:
                for c in range(len(row)):
                    row[c] *= factor
            for k in range(len(gb)):
                gb[k] *= factor

    def adam_step(self, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.step_count += 1
        t = self.step_count
        for i, layer in enumerate(self.layers):
            if self._g[i] is None:
                continue
            gw, gb = self._g[i]
            _adam_apply(layer.weights, gw, layer.m_w, layer.v_w, t, lr, b1, b2, eps)
            _adam_apply(layer.biases, gb, layer.m_b, layer.v_b, t, lr, b1, b2, eps)

    def sgd_step(self, lr=1e-2):
        for i, layer in enumerate(self.layers):
            if self._g[i] is None:
                continue
            gw, gb = self._g[i]
            _sgd_apply(layer.weights, gw, lr)
            _sgd_apply(layer.biases, gb, lr)

    def sample(self, n=1, rng=None):
        if isinstance(n, bool) or not isinstance(n, int) or n < 1:
            raise ValueError("VAE.sample: n 必须是 >= 1 的整数")
        return [self.decode(_gauss_vec(self.latent_dim, rng)) for _ in range(n)]


class GAN:
    """最小可训练 GAN（MLP 生成器 / 判别器，BCE 损失）

    判别器：D(x) = sigmoid(MLP(x))；生成器：G(z) 直接输出样本。
    训练时 D 用「真实样本 + 生成样本」各一次 BCE，G 只走判别器的生成分支
    （梯度穿过 D 但不更新 D 的参数）。
    """

    def __init__(self, input_dim: int, hidden_dim: int = 32, latent_dim: int = 2):
        for name, value in (("input_dim", input_dim), ("hidden_dim", hidden_dim),
                            ("latent_dim", latent_dim)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"GAN: {name} 必须是 >= 1 的整数")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.d1 = Layer(input_dim, hidden_dim, activation="leaky_relu")
        self.d2 = Layer(hidden_dim, 1, activation="sigmoid")
        self.g1 = Layer(latent_dim, hidden_dim, activation="relu")
        self.g2 = Layer(hidden_dim, input_dim, activation="linear")

    # ---- 判别器
    def discriminate(self, x):
        if not isinstance(x, (list, tuple)) or len(x) != self.input_dim:
            raise ValueError(f"GAN.discriminate: x 维度应为 {self.input_dim}")
        h = self.d1.forward(x)
        out = self.d2.forward(h)
        return out[0]

    def _d_grads(self, x, target):
        p = self.discriminate(x)
        p = min(max(p, 1e-12), 1.0 - 1e-12)
        loss = -(target * math.log(p) + (1.0 - target) * math.log(1.0 - p))
        d_out = [-(target / p) + (1.0 - target) / (1.0 - p)]
        gw2, gb2, d_h = self.d2.backward(d_out)
        gw1, gb1, d_x = self.d1.backward(d_h)
        return loss, (gw1, gb1), (gw2, gb2), d_x

    # ---- 生成器
    def generate(self, z):
        if not isinstance(z, (list, tuple)) or len(z) != self.latent_dim:
            raise ValueError(f"GAN.generate: z 维度应为 {self.latent_dim}")
        return self.g2.forward(self.g1.forward(z))

    def _g_grads(self, z):
        """生成样本的 D 分数对 G 的梯度（希望 D 判为 1）。"""
        x_fake = self.generate(z)
        p = self.discriminate(x_fake)
        p = min(max(p, 1e-12), 1.0 - 1e-12)
        d_out = [-1.0 / p]
        gw2, gb2, d_h = self.d2.backward(d_out)
        _gw1, _gb1, d_x = self.d1.backward(d_h)
        gw_g2, gb_g2, d_h1 = self.g2.backward(d_x)
        gw_g1, gb_g1, _ = self.g1.backward(d_h1)
        return (gw_g1, gb_g1), (gw_g2, gb_g2)

    def step(self, real_batch, rng=None, lr_d=1e-3, lr_g=1e-3, adam=True):
        if not real_batch:
            raise ValueError("GAN.step: real_batch 不能为空")
        r = rng or random
        # 1) 判别器：真实 + 生成各一次
        d_loss = 0.0
        accs = {"d1": None, "d2": None}
        for x in real_batch:
            loss, (w1, b1), (w2, b2), _ = self._d_grads(x, 1.0)
            d_loss += loss
            self._acc_pair(accs, "d1", w1, b1)
            self._acc_pair(accs, "d2", w2, b2)
        for _ in real_batch:
            z = _gauss_vec(self.latent_dim, r)
            loss, (w1, b1), (w2, b2), _ = self._d_grads(self.generate(z), 0.0)
            d_loss += loss
            self._acc_pair(accs, "d1", w1, b1)
            self._acc_pair(accs, "d2", w2, b2)
        scale = 1.0 / (2 * len(real_batch))
        self._apply(self.d1, accs["d1"], lr_d, scale, adam, "d1")
        self._apply(self.d2, accs["d2"], lr_d, scale, adam, "d2")

        # 2) 生成器：只更新 g1/g2
        g_loss = 0.0
        gaccs = {"g1": None, "g2": None}
        for _ in real_batch:
            z = _gauss_vec(self.latent_dim, r)
            (w1, b1), (w2, b2) = self._g_grads(z)
            g_loss += -math.log(max(1e-12, self.discriminate(self.generate(z))))
            self._acc_pair(gaccs, "g1", w1, b1)
            self._acc_pair(gaccs, "g2", w2, b2)
        scale_g = 1.0 / len(real_batch)
        self._apply(self.g1, gaccs["g1"], lr_g, scale_g, adam, "g1")
        self._apply(self.g2, gaccs["g2"], lr_g, scale_g, adam, "g2")
        return d_loss / (2 * len(real_batch)), g_loss / len(real_batch)

    @staticmethod
    def _acc_pair(accs, key, gw, gb):
        if accs[key] is None:
            accs[key] = [[[v for v in row] for row in gw], list(gb)]
        else:
            cur = accs[key]
            for r_ in range(len(cur[0])):
                for c in range(len(cur[0][r_])):
                    cur[0][r_][c] += gw[r_][c]
            for k in range(len(cur[1])):
                cur[1][k] += gb[k]

    def _apply(self, layer, acc, lr, scale, adam, tag):
        if acc is None:
            return
        gw, gb = acc
        for row in gw:
            for c in range(len(row)):
                row[c] *= scale
        for k in range(len(gb)):
            gb[k] *= scale
        self._adam_t = getattr(self, "_adam_t", 0)
        if adam:
            self._adam_t += 1
            _adam_apply(layer.weights, gw, layer.m_w, layer.v_w, self._adam_t, lr)
            _adam_apply(layer.biases, gb, layer.m_b, layer.v_b, self._adam_t, lr)
        else:
            _sgd_apply(layer.weights, gw, lr)
            _sgd_apply(layer.biases, gb, lr)

    def fit(self, data, epochs=200, lr_d=1e-3, lr_g=1e-3, seed=None,
            verbose=False):
        if not data:
            raise ValueError("GAN.fit: data 不能为空")
        rng = random.Random(seed) if seed is not None else random
        history = []
        for _epoch in range(epochs):
            d_loss, g_loss = self.step(data, rng=rng, lr_d=lr_d, lr_g=lr_g)
            history.append((d_loss, g_loss))
        return history


class DDPM:
    """最小 DDPM（去噪扩散概率模型）

    前向：x_t = sqrt(abar_t)*x0 + sqrt(1-abar_t)*noise
    训练：给网络 (x_t, t) 让它预测 noise，损失是逐维 MSE
    采样：从 x_T ~ N(0,1) 逐步反向去噪
    """

    def __init__(self, data_dim: int, hidden_dim: int = 32, steps: int = 50,
                 beta_start: float = 1e-4, beta_end: float = 0.02, t_dim: int = 8):
        for name, value in (("data_dim", data_dim), ("hidden_dim", hidden_dim),
                            ("steps", steps), ("t_dim", t_dim)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"DDPM: {name} 必须是 >= 1 的整数")
        if not 0.0 < beta_start < beta_end < 1.0:
            raise ValueError("DDPM: 需要 0 < beta_start < beta_end < 1")
        self.data_dim = data_dim
        self.hidden_dim = hidden_dim
        self.steps = steps
        self.t_dim = t_dim if t_dim % 2 == 0 else t_dim + 1

        self.betas = [beta_start + (beta_end - beta_start) * i / max(1, steps - 1)
                      for i in range(steps)]
        self.alphas = [1.0 - b for b in self.betas]
        self.abar = []
        prod = 1.0
        for a in self.alphas:
            prod *= a
            self.abar.append(prod)

        self.l1 = Layer(data_dim + self.t_dim, hidden_dim, activation="relu")
        self.l2 = Layer(hidden_dim, hidden_dim, activation="relu")
        self.l3 = Layer(hidden_dim, data_dim, activation="linear")
        self.layers = [self.l1, self.l2, self.l3]
        self.step_count = 0
        self.zero_grad()

    def zero_grad(self):
        self._g = [None] * len(self.layers)

    def _acc(self, i, gw, gb):
        if self._g[i] is None:
            self._g[i] = [gw, gb]
            return
        cur_w, cur_b = self._g[i]
        for r in range(len(cur_w)):
            for c in range(len(cur_w[r])):
                cur_w[r][c] += gw[r][c]
        for k in range(len(cur_b)):
            cur_b[k] += gb[k]

    def _grid(self, x, t):
        if t < 0 or t >= self.steps:
            raise ValueError(f"DDPM: 步数 t 必须在 [0, {self.steps})")
        if not isinstance(x, (list, tuple)) or len(x) != self.data_dim:
            raise ValueError(f"DDPM: x 维度应为 {self.data_dim}")
        return list(x) + _time_features(t, self.steps, self.t_dim)

    def q_sample(self, x0, t, noise):
        """前向加噪：x_t = sqrt(abar_t) * x0 + sqrt(1 - abar_t) * noise。"""
        if isinstance(t, bool) or not isinstance(t, int):
            raise TypeError("DDPM.q_sample: t 必须是整数")
        if t < 0 or t >= self.steps:
            raise ValueError(f"DDPM.q_sample: 步数 t 必须在 [0, {self.steps})")
        for name, v in (("x0", x0), ("noise", noise)):
            if not isinstance(v, (list, tuple)) or len(v) != self.data_dim:
                raise ValueError(f"DDPM.q_sample: {name} 维度应为 {self.data_dim}")
        a = math.sqrt(self.abar[t])
        b = math.sqrt(1.0 - self.abar[t])
        return [a * x0[i] + b * noise[i] for i in range(self.data_dim)]

    def predict_noise(self, x_t, t):
        inp = self._grid(x_t, t)
        h1 = self.l1.forward(inp)
        h2 = self.l2.forward(h1)
        return self.l3.forward(h2)

    def train_step(self, x0, rng=None, t=None, noise=None):
        """一步训练：随机取 t 与噪声，预测噪声并累积梯度。

        `t` / `noise` 显式给出时用于测试（固定随机性做中心差分）。
        """
        if not isinstance(x0, (list, tuple)) or len(x0) != self.data_dim:
            raise ValueError(f"DDPM.train_step: x0 维度应为 {self.data_dim}")
        r = rng or random
        if t is None:
            t = r.randrange(self.steps)
        if noise is None:
            noise = _gauss_vec(self.data_dim, r)
        x_t = self.q_sample(x0, t, noise)
        pred = self.predict_noise(x_t, t)
        n = self.data_dim
        loss = sum((pred[i] - noise[i]) ** 2 for i in range(n)) / n
        d_pred = [2.0 * (pred[i] - noise[i]) / n for i in range(n)]
        gw3, gb3, d2 = self.l3.backward(d_pred)
        gw2, gb2, d1 = self.l2.backward(d2)
        gw1, gb1, _ = self.l1.backward(d1)
        self._acc(0, gw1, gb1)
        self._acc(1, gw2, gb2)
        self._acc(2, gw3, gb3)
        return loss

    def fit(self, data, epochs=200, lr=1e-3, seed=None, verbose=False):
        if not data:
            raise ValueError("DDPM.fit: data 不能为空")
        rng = random.Random(seed) if seed is not None else random
        history = []
        for _epoch in range(epochs):
            total = 0.0
            self.zero_grad()
            for x in data:
                total += self.train_step(x, rng=rng)
            for item in self._g:
                if item is None:
                    continue
                gw, gb = item
                for row in gw:
                    for c in range(len(row)):
                        row[c] /= len(data)
                for k in range(len(gb)):
                    gb[k] /= len(data)
            self.adam_step(lr)
            history.append(total / len(data))
        return history

    def adam_step(self, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.step_count += 1
        t = self.step_count
        for i, layer in enumerate(self.layers):
            if self._g[i] is None:
                continue
            gw, gb = self._g[i]
            _adam_apply(layer.weights, gw, layer.m_w, layer.v_w, t, lr, b1, b2, eps)
            _adam_apply(layer.biases, gb, layer.m_b, layer.v_b, t, lr, b1, b2, eps)

    def sgd_step(self, lr=1e-2):
        for i, layer in enumerate(self.layers):
            if self._g[i] is None:
                continue
            gw, gb = self._g[i]
            _sgd_apply(layer.weights, gw, lr)
            _sgd_apply(layer.biases, gb, lr)

    def sample(self, n=1, rng=None):
        if isinstance(n, bool) or not isinstance(n, int) or n < 1:
            raise ValueError("DDPM.sample: n 必须是 >= 1 的整数")
        r = rng or random
        out = []
        for _ in range(n):
            x = _gauss_vec(self.data_dim, r)
            for t in range(self.steps - 1, -1, -1):
                eps_pred = self.predict_noise(x, t)
                a = self.alphas[t]
                ab = self.abar[t]
                mean = [(x[i] - (1.0 - a) / math.sqrt(max(1e-12, 1.0 - ab)) * eps_pred[i])
                        / math.sqrt(a) for i in range(self.data_dim)]
                if t > 0:
                    sigma = math.sqrt(self.betas[t])
                    z = _gauss_vec(self.data_dim, r)
                    x = [mean[i] + sigma * z[i] for i in range(self.data_dim)]
                else:
                    x = mean
            out.append(x)
        return out


# ── 工厂函数

def distill(teacher, student, x, y, epochs=400, batch_size=16,
            val_split=0.1, patience=50, verbose=True):
    """蒸馏：让 student 去拟合 teacher 的输出（软目标）

    与 C 引擎 ai_cli 的 distill 子命令同一套做法：先用教师生成软目标，再训练学生。
    返回 (student, 与教师的一致率)。分类按类别一致率算，回归按误差 <= 1e-2 判定。

    用法:
        teacher = make_regressor(1, 1, hidden_layers=(16, 16)); teacher.fit(x, y)
        student = make_regressor(1, 1, hidden_layers=(4,))
        student, agree = distill(teacher, student, x, y)
    """
    cls = getattr(teacher, 'task', 'regression') == 'classification'
    if cls:
        # 用教师的**软输出**（概率分布）当目标：argmax 硬标签会丢掉教师的不确定性，
        # 与 README 里「用教师输出当软目标」的描述也不一致
        soft_y = teacher.predict(x)                   # list[list] 概率分布
        student.fit(x, soft_y, epochs=epochs, batch_size=batch_size,
                    val_split=val_split, patience=patience, verbose=False)
        agree = sum(1 for xi, yi in zip(x, soft_y)
                    if student.predict_classes([xi])[0]
                    == max(range(len(yi)), key=lambda i: yi[i]))
    else:
        soft_y = teacher.predict(x)                  # 多样本 -> list[list]
        student.fit(x, soft_y, epochs=epochs, batch_size=batch_size,
                    val_split=val_split, patience=patience, verbose=False)
        agree = sum(1 for xi, yi in zip(x, soft_y)
                    if max(abs(a - b) for a, b in
                           zip(student.predict([xi])[0], yi)) <= 1e-2)

    rate = agree / len(x) if x else 0.0
    if verbose:
        print(f"蒸馏完成: 教师 {teacher.model.param_count()} 参数 -> "
              f"学生 {student.model.param_count()} 参数，与教师一致率 {rate * 100:.1f}%")
    return student, rate


def make_classifier(input_dim, num_classes, hidden_layers=(32,),
                    act='leaky_relu', lr=0.01, optimizer='adam',
                    opt_kwargs=None):
    """
    快速创建分类模型（Softmax + CrossEntropy）。

    用法:
        model = make_classifier(input_dim=10, num_classes=5, hidden_layers=(64, 32))
        model.fit(x, y, epochs=200)
        print(model.score(x_test, y_test))  # 准确率
    """
    sizes = [input_dim] + list(hidden_layers) + [num_classes]
    activations = [act] * (len(sizes) - 2) + ['softmax']
    mlp = MLP(sizes, activations=activations)
    return Model(mlp, loss_fn='cross_entropy', optimizer=optimizer,
                 lr=lr, opt_kwargs=opt_kwargs, task='classification')


def make_regressor(input_dim, output_dim=1, hidden_layers=(32,),
                   act='leaky_relu', lr=0.01, optimizer='adam',
                   opt_kwargs=None):
    """
    快速创建回归模型（MSE）。

    用法:
        model = make_regressor(input_dim=5, output_dim=2, hidden_layers=(64, 32))
        model.fit(x, y, epochs=300)
        print(model.score(x_test, y_test))  # R²
    """
    sizes = [input_dim] + list(hidden_layers) + [output_dim]
    activations = [act] * (len(sizes) - 2) + ['linear']
    mlp = MLP(sizes, activations=activations)
    return Model(mlp, loss_fn='mse', optimizer=optimizer,
                 lr=lr, opt_kwargs=opt_kwargs, task='regression')


_CONFIG_KEYS = frozenset({
    "input_dim", "output_dim", "hidden", "activations", "output_activation", "task",
    "loss", "loss_kwargs", "optimizer", "lr", "opt_kwargs",
    "epochs", "batch_size", "val_split", "patience", "seed", "shuffle",
    "grad_clip", "grad_accum", "restore_best_weights", "verbose",
    "lr_scheduler", "checkpoint_path", "checkpoint_every",
})

_SCHEDULER_KEYS = frozenset({
    "type", "step_size", "gamma", "t_max", "eta_min",
    "warmup_epochs", "start_factor", "base_scheduler",
    "factor", "patience", "min_lr", "mode", "threshold", "max_reductions",
})


def _config_mapping(config, who):
    if isinstance(config, str):
        with io.open(config, "r", encoding="utf-8") as f:
            config = json.load(f)
    if not isinstance(config, dict):
        raise TypeError(f"{who}: 配置必须是 dict 或 JSON 文件路径")
    unknown = sorted(set(config) - _CONFIG_KEYS)
    if unknown:
        raise ValueError(
            f"{who}: 不认识的配置项 {unknown}；可用项见 AI._CONFIG_KEYS")
    return dict(config)


def _config_int(config, key, default, minimum=None, who="config"):
    value = config.get(key, default)
    if value is None:
        value = default
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{who}: {key} 必须是 int，收到 {value!r}")
    if minimum is not None and value < minimum:
        raise ValueError(f"{who}: {key} 必须 >= {minimum}，收到 {value}")
    return value


def _config_number(config, key, default, who="config"):
    value = config.get(key, default)
    if value is None:
        value = default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{who}: {key} 必须是数字，收到 {value!r}")
    if not math.isfinite(value):
        raise ValueError(f"{who}: {key} 必须是有限数，收到 {value!r}")
    return float(value)


def _config_bool(config, key, default, who="config"):
    value = config.get(key, default)
    if isinstance(value, bool):
        return value
    raise TypeError(f"{who}: {key} 必须是 bool，收到 {value!r}")


def _build_scheduler(spec, optimizer, who="config"):
    """把 {"type": "step", ...} 这样的描述变成调度器对象。"""
    if spec is None:
        return None
    if not isinstance(spec, dict):
        raise TypeError(f"{who}: lr_scheduler 必须是 dict 或 None")
    unknown = sorted(set(spec) - _SCHEDULER_KEYS)
    if unknown:
        raise ValueError(f"{who}: lr_scheduler 里有不认识的键 {unknown}")
    kind = spec.get("type")
    if not isinstance(kind, str):
        raise TypeError(f"{who}: lr_scheduler.type 必须是字符串")
    kind = kind.lower()
    if kind == "step":
        return StepLR(optimizer,
                      step_size=_config_int(spec, "step_size", 30, 1, who),
                      gamma=_config_number(spec, "gamma", 0.1, who))
    if kind == "exponential":
        return ExponentialLR(optimizer, gamma=_config_number(spec, "gamma", 0.99, who))
    if kind == "cosine":
        return CosineAnnealingLR(optimizer,
                                 T_max=_config_int(spec, "t_max", 10, 1, who),
                                 eta_min=_config_number(spec, "eta_min", 0.0, who))
    if kind == "warmup":
        base_spec = spec.get("base_scheduler")
        base = None
        if base_spec is not None:
            if not isinstance(base_spec, dict):
                raise TypeError(f"{who}: warmup.base_scheduler 必须是 dict 或 None")
            base = _build_scheduler(base_spec, optimizer, who)
        return WarmupLR(optimizer,
                        warmup_epochs=_config_int(spec, "warmup_epochs", 5, 1, who),
                        start_factor=_config_number(spec, "start_factor", 0.0, who),
                        base_scheduler=base)
    if kind == "plateau":
        return ReduceLROnPlateau(optimizer,
                                 factor=_config_number(spec, "factor", 0.5, who),
                                 patience=_config_int(spec, "patience", 10, 0, who),
                                 min_lr=_config_number(spec, "min_lr", 0.0, who),
                                 mode=spec.get("mode", "min"),
                                 threshold=_config_number(spec, "threshold", 1e-4, who),
                                 max_reductions=spec.get("max_reductions"))
    raise ValueError(
        f"{who}: 未知的 lr_scheduler.type {kind!r}（可选 step / exponential / cosine / warmup / plateau）")


def _infer_dims(data):
    """从 [(x, y), ...] 推断输入/输出维度。"""
    if not isinstance(data, (list, tuple)) or not data:
        raise ValueError("build_model: 没有给 input_dim/output_dim 时必须提供非空 data")
    first = data[0]
    if not isinstance(first, (list, tuple)) or len(first) != 2:
        raise ValueError("build_model: data 的每个样本必须是 (x, y)")
    x, y = first
    x = x if isinstance(x, (list, tuple)) else [x]
    y = y if isinstance(y, (list, tuple)) else [y]
    if not x or not y:
        raise ValueError("build_model: 样本的 x / y 不能为空")
    return len(x), len(y)


def build_model(config, data=None):
    """按配置字典（或 JSON 文件路径）建一个 `Model`。

    必需：`input_dim` / `output_dim`（也可以给 `data` 让函数推断）。
    `hidden` 缺省 `[16]`；`activations` 缺省全部 `relu`；
    `output_activation` 缺省按 `task` 取（分类 softmax、回归 linear）。
    """
    cfg = _config_mapping(config, "build_model")
    in_dim, out_dim = cfg.get("input_dim"), cfg.get("output_dim")
    if in_dim is None or out_dim is None:
        guessed_in, guessed_out = _infer_dims(data)
        in_dim = guessed_in if in_dim is None else in_dim
        out_dim = guessed_out if out_dim is None else out_dim
    in_dim = _config_int({"v": in_dim}, "v", 1, 1, "build_model")
    out_dim = _config_int({"v": out_dim}, "v", 1, 1, "build_model")

    hidden = cfg.get("hidden", [16])
    if not isinstance(hidden, (list, tuple)):
        raise TypeError("build_model: hidden 必须是 list")
    if not hidden:
        raise ValueError("build_model: hidden 不能为空（至少一层隐藏层）")
    hidden = [_config_int({"h": h}, "h", 1, 1, "build_model") for h in hidden]

    task = cfg.get("task", "auto")
    if task not in ("auto", "regression", "classification"):
        raise ValueError(f"build_model: task 只能是 auto/regression/classification，收到 {task!r}")
    if task == "auto":
        task = "classification" if out_dim > 1 else "regression"

    default_out = "softmax" if task == "classification" else "linear"
    out_act = cfg.get("output_activation", default_out)

    n_hidden = len(hidden)
    activations = cfg.get("activations")
    if activations is None:
        activations = ["relu"] * n_hidden
    if not isinstance(activations, (list, tuple)):
        raise TypeError("build_model: activations 必须是 list")
    if len(activations) != n_hidden:
        raise ValueError(
            f"build_model: activations 有 {len(activations)} 项，但 hidden 有 {n_hidden} 层")
    activations = list(activations) + [out_act]

    mlp = MLP([in_dim] + hidden + [out_dim], activations=activations)
    return Model(
        mlp,
        loss_fn=cfg.get("loss"),
        optimizer=cfg.get("optimizer", "adam"),
        lr=_config_number(cfg, "lr", 0.01, "build_model"),
        opt_kwargs=cfg.get("opt_kwargs"),
        task=task,
        loss_kwargs=cfg.get("loss_kwargs"),
    )


def train_from_config(config, data=None, x=None, y=None, verbose=None):
    """按配置建模型并训练，返回训练好的 `Model`（`model.history` 里有曲线）。

    `data` 与 `x`/`y` 二选一；给了 `data` 时若配置里没有维度就自动推断。
    `verbose` 不传时用配置里的 `verbose`（缺省 True）。
    """
    cfg = _config_mapping(config, "train_from_config")
    if data is not None and (x is not None or y is not None):
        raise ValueError("train_from_config: data 与 x/y 不能同时给")

    seed = cfg.get("seed")
    if seed is not None:
        # 与 quick_train 一致：seed 同时固定初始化与打乱，
        # 且只在初始化期间临时改全局 random，之后原样恢复。
        _saved_state = random.getstate()
        try:
            random.seed(seed)
            model = build_model(cfg, data=data)
        finally:
            random.setstate(_saved_state)
    else:
        model = build_model(cfg, data=data)

    if verbose is None:
        verbose = cfg.get("verbose", True)
    if not isinstance(verbose, bool):
        raise TypeError("train_from_config: verbose 必须是 bool")

    scheduler = _build_scheduler(cfg.get("lr_scheduler"), model.optimizer,
                                 "train_from_config")

    fit_kwargs = dict(
        epochs=_config_int(cfg, "epochs", 100, 0, "train_from_config"),
        batch_size=_config_int(cfg, "batch_size", 32, 1, "train_from_config"),
        patience=_config_int(cfg, "patience", 50, 0, "train_from_config"),
        grad_accum=_config_int(cfg, "grad_accum", 1, 1, "train_from_config"),
        shuffle=_config_bool(cfg, "shuffle", True, "train_from_config"),
        restore_best_weights=_config_bool(cfg, "restore_best_weights", False,
                                          "train_from_config"),
        verbose=verbose,
        lr_scheduler=scheduler,
        checkpoint_path=cfg.get("checkpoint_path"),
        checkpoint_every=_config_int(cfg, "checkpoint_every", 1, 1, "train_from_config"),
        seed=seed,
    )
    if cfg.get("val_split") is not None:
        fit_kwargs["val_split"] = _config_number(cfg, "val_split", 0.1, "train_from_config")
    if cfg.get("grad_clip") is not None:
        fit_kwargs["grad_clip"] = _config_number(cfg, "grad_clip", 0.0, "train_from_config")

    if data is not None:
        model.fit(data=data, **fit_kwargs)
    else:
        if x is None or y is None:
            raise ValueError("train_from_config: 必须给 data 或同时给 x 和 y")
        model.fit(x=x, y=y, **fit_kwargs)
    return model


_SEARCH_METRICS = {
    "val_loss": "min", "train_loss": "min", "grad_norm": "min",
    "accuracy": "max",
}


def _search_score(model, metric, x, y):
    """按指标名从训练结果里取分；accuracy 需要 x/y。"""
    hist = model.history
    if metric in ("val_loss", "train_loss", "grad_norm"):
        if metric == "val_loss" and not hist.get("val_loss"):
            raise ValueError(
                "hyperparameter_search: 配置里没有验证集（val_split=0），无法用 val_loss 评分")
        series = hist.get(metric) or []
        if not series:
            raise ValueError(f"hyperparameter_search: history 里没有 {metric}")
        return float(series[-1][1] if isinstance(series[-1], tuple) else series[-1])
    if metric == "accuracy":
        if x is None or y is None:
            raise ValueError("hyperparameter_search: 用 accuracy 评分时必须提供 x 和 y")
        preds = []
        for xi in x:
            out = model.predict(xi)
            out = out if isinstance(out, (list, tuple)) else [out]
            preds.append(max(range(len(out)), key=lambda k: out[k]) if len(out) > 1
                         else (1 if out[0] > 0.5 else 0))
        truth = []
        for yi in y:
            if isinstance(yi, (list, tuple)):
                truth.append(max(range(len(yi)), key=lambda k: yi[k]))
            elif isinstance(yi, bool):
                truth.append(1 if yi else 0)
            else:
                truth.append(int(yi))
        hit = sum(1 for p, t in zip(preds, truth) if p == t)
        return hit / max(1, len(truth))
    raise ValueError(f"hyperparameter_search: 未知指标 {metric!r}")


def hyperparameter_search(base_config, grid, data=None, x=None, y=None,
                          method="grid", n_trials=16, metric="val_loss",
                          seed=None, verbose=False):
    """在配置空间里搜超参，返回最优配置与全部试验记录。

    base_config: 基础配置（同 `train_from_config`），必须含维度或给 `data`。
    grid:        {参数名: [候选值, ...]}，参数名必须是合法配置项。
    method:      "grid" 全组合；"random" 随机抽 `n_trials` 组（去重后不足则按顺序补）。
    metric:      "val_loss" / "train_loss" / "grad_norm"（越小越好）或 "accuracy"（越大越好）。
    seed:        随机搜索与训练共用的种子；给了就整轮可复现。

    返回 {"best": {...}, "trials": [...], "metric", "mode", "params_names"}。
    每次试验都用 `train_from_config` 训练，`verbose` 默认关掉以免刷屏。
    """
    if not isinstance(grid, dict) or not grid:
        raise ValueError("hyperparameter_search: grid 必须是非空 dict")
    base = _config_mapping(base_config, "hyperparameter_search")
    unknown = sorted(set(grid) - _CONFIG_KEYS)
    if unknown:
        raise ValueError(f"hyperparameter_search: grid 里有不认识的配置项 {unknown}")
    values = {}
    for key, options in grid.items():
        if not isinstance(options, (list, tuple)) or not options:
            raise ValueError(f"hyperparameter_search: grid[{key!r}] 必须是非空 list")
        values[key] = list(options)
    if method not in ("grid", "random"):
        raise ValueError(f"hyperparameter_search: method 只能是 grid/random，收到 {method!r}")
    if isinstance(n_trials, bool) or not isinstance(n_trials, int) or n_trials < 1:
        raise ValueError(f"hyperparameter_search: n_trials 必须是 >= 1 的 int")
    if metric not in _SEARCH_METRICS:
        raise ValueError(
            f"hyperparameter_search: 未知指标 {metric!r}（可选 {sorted(_SEARCH_METRICS)}）")
    mode = _SEARCH_METRICS[metric]

    keys = list(values)
    if method == "grid":
        combos = [dict(zip(keys, picks)) for picks in itertools.product(*(values[k] for k in keys))]
    else:
        rng = random.Random(seed) if seed is not None else random
        seen = set()
        combos = []
        while len(combos) < n_trials:
            picks = tuple(rng.choice(values[k]) for k in keys)
            if picks in seen:
                if len(seen) >= len(list(itertools.product(*(values[k] for k in keys)))):
                    break
                continue
            seen.add(picks)
            combos.append(dict(zip(keys, picks)))

    trials = []
    best = None
    for trial_no, combo in enumerate(combos):
        cfg = dict(base)
        cfg.update(combo)
        model = train_from_config(cfg, data=data, x=x, y=y, verbose=verbose)
        score = _search_score(model, metric, x, y)
        item = {"trial": trial_no, "params": dict(combo), "score": score,
                "history": model.history}
        trials.append(item)
        if best is None or (score < best["score"] if mode == "min" else score > best["score"]):
            best = item
    return {"best": best, "trials": trials, "metric": metric, "mode": mode,
            "params_names": keys}


def quick_train(x, y, task='regression', hidden_layers=(32,), epochs=200,
                batch_size=32, lr=0.01, val_split=0.1, patience=30,
                optimizer='adam', verbose=True, seed=None):
    """
    一行代码训练并返回模型。

    task='classification' 时 y 可以是整数类别索引（允许非连续，如 [0, 2]），
    也可以是 one-hot / 概率向量——类数按向量长度或 max+1 推断，不会因为
    某个类在本批数据里没出现而少建输出单元。小数标签与负索引会直接报错。

    seed: 传整数时，「权重初始化」与「每轮打乱」都用这个种子，同样的调用
          再跑一次得到的模型权重与 loss 曲线完全相同；初始化期间会临时设置
          全局 random 并在结束后恢复，不会推进调用方的随机状态。

    用法:
        model = quick_train(x, y, task='classification', epochs=300)
        model = quick_train(x, y, task='regression', hidden_layers=(64, 32, 16))
    """
    if not x:
        raise ValueError("quick_train: x 不能为空")
    if not y:
        raise ValueError("quick_train: y 不能为空")
    input_dim = len(x[0])

    if task == 'classification':
        # 标签既可以是类别索引 int，也可以是 one-hot / 概率向量；
        # 后者直接 set(y) 会抛 unhashable，这里统一转成索引
        if not y:
            raise ValueError("quick_train: 分类任务的 y 不能为空")
        if isinstance(y[0], (list, tuple)):
            # 类数 = 向量长度（用 len(set(标签)) 会漏掉本批没出现的类）
            num_classes = len(y[0])
            if len({isinstance(v, (list, tuple)) for v in y}) > 1:
                raise ValueError("quick_train: y 中不能混用向量标签与整数标签")
            # 概率 / one-hot 向量原样保留：Model 的 CE 路径会把等长向量直通给
            # cross_entropy_loss 的软标签分支，argmax 硬化会丢掉软标签信息
            y = [list(row) for row in y]
        else:
            if any(isinstance(v, (list, tuple)) for v in y):
                raise ValueError("quick_train: y 中不能混用向量标签与整数标签")
            # 整数标签：类数 = max+1（不能用 len(set(y))：标签 [0,2] 会越界）。
            # 注意不能直接 int(v)：那会把 1.5 静默截成 1、把 -0.5 截成 0，
            # 让非法标签混进训练集。
            labels = []
            for v in y:
                fv = float(v)
                if not fv.is_integer():
                    raise ValueError(
                        f"quick_train: 分类标签必须是整数类别索引，收到 {v!r}"
                    )
                if fv < 0:
                    raise ValueError(f"quick_train: 类别索引不能为负（发现 {fv}）")
                labels.append(int(fv))
            y = labels
            num_classes = max(y) + 1
        build = lambda: make_classifier(input_dim, num_classes,
                                        hidden_layers=hidden_layers, lr=lr,
                                        optimizer=optimizer)
    else:
        output_dim = len(y[0]) if isinstance(y[0], (list, tuple)) else 1
        build = lambda: make_regressor(input_dim, output_dim,
                                       hidden_layers=hidden_layers, lr=lr,
                                       optimizer=optimizer)

    # 权重初始化也吃同一个 seed：临时设置全局 random，构造完立刻恢复原状态
    if seed is not None:
        _saved_state = random.getstate()
        try:
            random.seed(seed)
            model = build()
        finally:
            random.setstate(_saved_state)
    else:
        model = build()

    model.fit(x, y, epochs=epochs, batch_size=batch_size,
              val_split=val_split, patience=patience, verbose=verbose,
              seed=seed)
    return model


# ============================================================================
#  Main: 回归 + 分类 + CNN 完整示例
# ============================================================================

if __name__ == "__main__":

    # 修复 Windows 控制台中文乱码问题（仅直接运行时生效，避免干扰 pytest 等工具）
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    # ==================== 回归示例（高层 API） ====================
    print("=" * 60)
    print("回归任务: y = x^2 / 5 的近似（make_regressor + fit）")
    print("=" * 60)

    random.seed(42)

    x_reg = [
        [0.2, 0.4, 0.6, 0.8, 1.0],
        [0.4, 0.6, 0.8, 1.0, 1.2],
        [0.6, 0.8, 1.0, 1.2, 1.4],
        [0.1, 0.3, 0.5, 0.7, 0.9],
        [0.3, 0.5, 0.7, 0.9, 1.1],
    ]
    y_reg = [[0.04, 0.08], [0.16, 0.24], [0.36, 0.48], [0.01, 0.02], [0.09, 0.18]]

    model_reg = make_regressor(input_dim=5, output_dim=2,
                               hidden_layers=(6, 6, 6, 6, 5, 4, 3), lr=0.01)
    model_reg.summary()

    model_reg.fit(x_reg, y_reg, epochs=2000, batch_size=2, patience=50, val_split=0.2)

    print("\n--- 回归测试结果 ---")
    for i in range(len(x_reg)):
        pred = model_reg.predict(x_reg[i])
        print(f"  输入: {tuple(x_reg[i])}  目标: {tuple(y_reg[i])}"
              f"  预测: {[round(p, 6) for p in pred]}")

    # 保留原始 reg_data 供后续底层示例使用
    reg_data = list(zip(x_reg, y_reg))
    train_data, _, test_data = split_data(reg_data, val_ratio=0.2, test_ratio=0.2)

    # ==================== 分类示例（高层 API） ====================
    print("\n" + "=" * 60)
    print("分类任务: Softmax + Cross Entropy（make_classifier + fit）")
    print("=" * 60)

    x_cls = [
        [0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9],
        [0.15, 0.25, 0.35], [0.45, 0.55, 0.65], [0.75, 0.85, 0.95],
    ]
    y_cls = [0, 1, 2, 0, 1, 2]

    model_cls = make_classifier(input_dim=3, num_classes=3,
                                hidden_layers=(8, 8), lr=0.01)
    model_cls.summary()

    model_cls.fit(x_cls, y_cls, epochs=2000, batch_size=2, patience=50, val_split=0.2)

    print("\n--- 分类测试结果 ---")
    acc = model_cls.score(x_cls, y_cls)
    for i in range(len(x_cls)):
        pred_idx = model_cls.predict_classes(x_cls[i])
        status = "✓" if pred_idx == y_cls[i] else "✗"
        print(f"  输入: {tuple(x_cls[i])}  预测: {pred_idx}  实际: {y_cls[i]}  {status}")
    print(f"  准确率: {acc:.1%}")

    # ==================== CNN 示例 ====================
    print("\n" + "=" * 60)
    print("CNN 示例: 2D 卷积 → 池化 → 展平 → MLP 分类")
    print("=" * 60)

    random.seed(42)

    # 构建简单图像数据: (C, H, W) = (1, 4, 4)
    imgs = [
        # 右上角亮的 → class 0
        ([[[0.1, 0.1, 0.8, 0.9],
           [0.1, 0.2, 0.7, 0.8],
           [0.1, 0.1, 0.1, 0.2],
           [0.1, 0.1, 0.1, 0.1]]], 0),
        ([[[0.2, 0.1, 0.9, 0.8],
           [0.1, 0.1, 0.6, 0.7],
           [0.2, 0.1, 0.2, 0.1],
           [0.1, 0.2, 0.1, 0.1]]], 0),
        # 左下角亮的 → class 1
        ([[[0.1, 0.1, 0.1, 0.2],
           [0.1, 0.2, 0.1, 0.1],
           [0.7, 0.8, 0.1, 0.1],
           [0.9, 0.7, 0.2, 0.1]]], 1),
        ([[[0.2, 0.1, 0.1, 0.1],
           [0.1, 0.1, 0.2, 0.1],
           [0.8, 0.6, 0.1, 0.2],
           [0.7, 0.9, 0.1, 0.1]]], 1),
    ]

    # 构建 CNN: Conv2d(1→2, kernel=2, stride=1) → MaxPool2d(2) → Flatten → MLP
    conv = Conv2d(in_channels=1, out_channels=2, kernel_size=2, stride=1)
    pool = MaxPool2d(kernel_size=2)
    flatten = Flatten()

    # 计算展平后维度: (2, 3, 3) → pool → (2, 1, 1) → flatten → 2
    fc = MLP([2, 4, 2], activations=["leaky_relu", "softmax"])

    # 手动训练循环（简化版）
    def cnn_forward(img):
        x = conv.forward(img)
        x = pool.forward(x)
        x = flatten.forward(x)
        x = fc.forward(x)
        return x

    def cnn_backward(loss_grad):
        delta = loss_grad
        # 收集 MLP 各层梯度（反向传播）
        fc_grads = []
        for layer in reversed(fc.layers):
            gw, gb, delta = layer.backward(delta)
            fc_grads.insert(0, (gw, gb))
        delta = flatten.backward(delta)
        delta = pool.backward(delta)
        _ = conv.backward(delta)

# ============================================================================
#  应用梯度更新权重 (SGD)
# ============================================================================
        cnn_lr = 0.01
        # 更新 Conv2d 权重
        for oc in range(conv.out_channels):
            conv.biases[oc] -= cnn_lr * conv.grad_b[oc]
            for ic in range(conv.in_channels):
                for kh in range(conv.kh):
                    for kw in range(conv.kw):
                        conv.weights[oc][ic][kh][kw] -= cnn_lr * conv.grad_w[oc][ic][kh][kw]
        # 更新 MLP 权重
        for lidx, (gw, gb) in enumerate(fc_grads):
            layer = fc.layers[lidx]
            for j in range(layer.fan_out):
                layer.biases[j] -= cnn_lr * gb[j]
                for i in range(layer.fan_in):
                    layer.weights[j][i] -= cnn_lr * gw[j][i]

    print(f"Conv: in=1 out=2 kernel=2  →  Pool: 2×2  →  Flatten  →  {fc}")
    print(f"训练集: {len(imgs)} 样本")

    for epoch in range(1, 301):
        random.shuffle(imgs)
        total_loss = 0.0
        for img, label in imgs:
            conv.zero_grad()
            pred = cnn_forward(img)
            loss, grad = cross_entropy_loss(pred, label)
            total_loss += loss
            cnn_backward(grad)

        if epoch % 100 == 0:
            print(f"  Epoch {epoch} | Loss: {total_loss / len(imgs):.6f}")

    print("\n--- CNN 测试结果 ---")
    for img, label in imgs:
        pred = cnn_forward(img)
        pred_cls = pred.index(max(pred))
        status = "✓" if pred_cls == label else "✗"
        print(f"  预测: {pred_cls}  实际: {label}  {status}  probs={[round(p, 4) for p in pred]}")

    # ==================== RNN / LSTM 示例 ====================
    print("\n" + "=" * 60)
    print("RNN & LSTM 示例: 序列预测")
    print("=" * 60)

    random.seed(42)

    # 序列数据: 递增序列 → 下一个数
    seq_data = [
        ([0.1, 0.2, 0.3], 0.4),
        ([0.5, 0.6, 0.7], 0.8),
        ([0.2, 0.4, 0.6], 0.8),
        ([0.3, 0.5, 0.7], 0.9),
        ([0.1, 0.3, 0.5], 0.7),
    ]

    # 将每个输入转为序列: 每个时间步是 (1,) 向量
    def to_seq(nums):
        return [[n] for n in nums]

    # RNN
    print("\n--- RNN ---")
    rnn = RNN(input_size=1, hidden_size=4, num_layers=1)
    # 输出层: hidden_size → 1
    out_w = [[random.gauss(0, math.sqrt(1.0 / 4)) for _ in range(4)]]
    out_b = [0.0]

    for epoch in range(1, 201):
        random.shuffle(seq_data)
        total_loss = 0.0
        for seq, target in seq_data:
            x_seq = to_seq(seq)
            outputs, _ = rnn.forward(x_seq)
            # 取最后一个时间步的输出
            last_h = outputs[-1]

            # 线性输出
            pred_val = out_b[0] + sum(out_w[0][i] * last_h[i] for i in range(4))
            target_val = target

            # MSE
            loss = (pred_val - target_val) ** 2
            total_loss += loss

            # 梯度: dL/dpred = 2*(pred - target)
            d_pred = 2 * (pred_val - target)

            # 输出层梯度
            grad_out_w = [[d_pred * last_h[i] for i in range(4)]]
            grad_out_b = [d_pred]

            # 传回 RNN 的梯度
            dh = [d_pred * out_w[0][i] for i in range(4)]

            # BPTT: 逐时间步反向传播，反向过程会原地累加到 cell 的梯度缓冲区
            cell = rnn.cells[0]
            cell.zero_grad()          # 整个 BPTT 只清一次

            dh_cur = dh
            for t in reversed(range(len(x_seq))):
                # RNN.forward 已把每一步的缓存存进 step_cache，这里直接恢复，
                # 不需要再 forward 一次来"重建"缓存
                cell.restore_cache(rnn.step_cache[t][0])
                _, dh_cur = cell.backward(dh_cur)

            # cell 内部已经把所有时间步的梯度累加好了，直接取用。
            # 注意：get_grads() 返回的就是内部缓冲区本身，不能再去"手工求和"——
            # 那等于把缓冲区加到它自己身上，梯度会按 2 的幂被放大。
            grad_W_ih, grad_W_hh, grad_b_ih, grad_b_hh = cell.get_grads()

            # 更新 RNN 权重 (SGD)
            # lr 从 0.01 提到 0.03：修掉上面的重复累加后梯度不再被放大，
            # 用 0.03 才能和修复前的收敛速度持平（且实测略好）
            lr = 0.03
            for j in range(4):
                cell.b_ih[j] -= lr * grad_b_ih[j]
                cell.b_hh[j] -= lr * grad_b_hh[j]
                for i in range(1):  # input_size = 1
                    cell.W_ih[j][i] -= lr * grad_W_ih[j][i]
                for i in range(4):
                    cell.W_hh[j][i] -= lr * grad_W_hh[j][i]

            # 更新输出层权重 (SGD)
            for i in range(4):
                out_w[0][i] -= lr * grad_out_w[0][i]
            out_b[0] -= lr * grad_out_b[0]

        if epoch % 50 == 0:
            print(f"  Epoch {epoch} | Loss: {total_loss / len(seq_data):.6f}")

    print("RNN 测试:")
    for seq, target in seq_data:
        outputs, _ = rnn.forward(to_seq(seq))
        pred = out_b[0] + sum(out_w[0][i] * outputs[-1][i] for i in range(4))
        print(f"  序列: {seq}  预测: {pred:.4f}  目标: {target}")

    # LSTM
    print("\n--- LSTM ---")
    lstm = LSTM(input_size=1, hidden_size=4, num_layers=1)
    lstm_out_w = [[random.gauss(0, math.sqrt(1.0 / 4)) for _ in range(4)]]
    lstm_out_b = [0.0]

    for epoch in range(1, 201):
        random.shuffle(seq_data)
        total_loss = 0.0
        for seq, target in seq_data:
            x_seq = to_seq(seq)
            outputs, (h_n, c_n) = lstm.forward(x_seq)
            last_h = outputs[-1]

            pred_val = lstm_out_b[0] + sum(lstm_out_w[0][i] * last_h[i] for i in range(4))
            loss = (pred_val - target) ** 2
            total_loss += loss

            d_pred = 2 * (pred_val - target)

            grad_out_w = [[d_pred * last_h[i] for i in range(4)]]
            grad_out_b = [d_pred]

            # 传回 LSTM 的梯度
            dh = [d_pred * lstm_out_w[0][i] for i in range(4)]

            # BPTT: 逐时间步反向传播，反向过程会原地累加到 cell 的梯度缓冲区
            cell = lstm.cells[0]
            cell.zero_grad()          # 整个 BPTT 只清一次

            dh_cur = dh
            dc_cur = [0.0] * 4        # 最后一步没有来自后续的 cell gradient
            for t in reversed(range(len(x_seq))):
                # LSTM.forward 已把每步的 gates / c / x / h_prev 存进 step_cache，
                # 直接恢复即可：既不用额外跑一轮前向收集 cs_list，也不用逐步重算
                cell.restore_cache(lstm.step_cache[t][0])
                _, dh_cur, dc_cur = cell.backward(dh_cur, dc_cur)

            # cell 内部已累加好所有时间步的梯度，直接取用（get_grads() 返回的是
            # 内部缓冲区本身，不能再手工求和，否则会自己加自己、按 2 的幂放大）
            grad_W_ih, grad_W_hh, grad_b_ih, grad_b_hh = cell.get_grads()

            # 更新 LSTM 权重 (SGD)；lr 见 RNN 段的说明
            lr = 0.03
            gates = 16  # 4 * hidden_size
            for j in range(gates):
                cell.b_ih[j] -= lr * grad_b_ih[j]
                cell.b_hh[j] -= lr * grad_b_hh[j]
                for i in range(1):  # input_size = 1
                    cell.W_ih[j][i] -= lr * grad_W_ih[j][i]
                for i in range(4):
                    cell.W_hh[j][i] -= lr * grad_W_hh[j][i]

            # 更新输出层权重 (SGD)
            for i in range(4):
                lstm_out_w[0][i] -= lr * grad_out_w[0][i]
            lstm_out_b[0] -= lr * grad_out_b[0]

        if epoch % 50 == 0:
            print(f"  Epoch {epoch} | Loss: {total_loss / len(seq_data):.6f}")

    print("LSTM 测试:")
    for seq, target in seq_data:
        outputs, (h_n, c_n) = lstm.forward(to_seq(seq))
        pred = lstm_out_b[0] + sum(lstm_out_w[0][i] * outputs[-1][i] for i in range(4))
        print(f"  序列: {seq}  预测: {pred:.4f}  目标: {target}")

    # ==================== 保存 / 加载（高层 API） ====================
    print("\n" + "=" * 60)
    print("模型保存 & 加载（Model.save / Model.load）")
    print("=" * 60)

    model_cls.save("classification_model.json")
    loaded_model = Model.load("classification_model.json")

    print("加载后的模型测试:")
    for i in range(len(x_cls)):
        pred_idx = loaded_model.predict_classes(x_cls[i])
        print(f"  预测: {pred_idx}  实际: {y_cls[i]}  {'✓' if pred_idx == y_cls[i] else '✗'}")

    # ==================== GRU 示例 ====================
    print("\n" + "=" * 60)
    print("GRU 示例: 序列预测（与 RNN/LSTM 对比）")
    print("=" * 60)

    random.seed(42)
    gru = GRU(input_size=1, hidden_size=4, num_layers=1)
    gru_out_w = [[random.gauss(0, math.sqrt(1.0 / 4)) for _ in range(4)]]
    gru_out_b = [0.0]

    for epoch in range(1, 201):
        random.shuffle(seq_data)
        total_loss = 0.0
        for seq, target in seq_data:
            x_seq = to_seq(seq)
            outputs, _ = gru.forward(x_seq)
            last_h = outputs[-1]

            pred_val = gru_out_b[0] + sum(gru_out_w[0][i] * last_h[i] for i in range(4))
            loss = (pred_val - target) ** 2
            total_loss += loss

            d_pred = 2 * (pred_val - target)
            grad_out_w = [[d_pred * last_h[i] for i in range(4)]]
            grad_out_b = [d_pred]

            # BPTT for GRU：反向过程会原地累加到 cell 的梯度缓冲区
            cell = gru.cells[0]
            cell.zero_grad()          # 整个 BPTT 只清一次

            dh = [d_pred * gru_out_w[0][i] for i in range(4)]
            dh_cur = dh
            for t in reversed(range(len(x_seq))):
                # GRU.forward 已存好每步缓存，直接恢复，避免二次前向
                cell.restore_cache(gru.step_cache[t][0])
                _, dh_cur = cell.backward(dh_cur)

            # cell 内部已累加好所有时间步的梯度，直接取用（get_grads() 返回的是
            # 内部缓冲区本身，不能再手工求和，否则会自己加自己、按 2 的幂放大）
            grad_W_ih, grad_W_hh, grad_b_ih, grad_b_hh = cell.get_grads()

            # 更新 GRU 权重 (SGD)；lr 见 RNN 段的说明
            lr = 0.03
            for j in range(12):  # 3 * hidden
                cell.b_ih[j] -= lr * grad_b_ih[j]
                cell.b_hh[j] -= lr * grad_b_hh[j]
                for i in range(1):  # input_size = 1
                    cell.W_ih[j][i] -= lr * grad_W_ih[j][i]
                for i in range(4):
                    cell.W_hh[j][i] -= lr * grad_W_hh[j][i]
            for i in range(4):
                gru_out_w[0][i] -= lr * grad_out_w[0][i]
            gru_out_b[0] -= lr * grad_out_b[0]

        if epoch % 50 == 0:
            print(f"  Epoch {epoch} | Loss: {total_loss / len(seq_data):.6f}")

    print("GRU 测试:")
    for seq, target in seq_data:
        outputs, _ = gru.forward(to_seq(seq))
        pred = gru_out_b[0] + sum(gru_out_w[0][i] * outputs[-1][i] for i in range(4))
        print(f"  序列: {seq}  预测: {pred:.4f}  目标: {target}")

    # ==================== Dropout + LayerNorm 示例 ====================
    print("\n" + "=" * 60)
    print("Dropout + LayerNorm 示例")
    print("=" * 60)

    random.seed(42)

    # 多层感知机中插入 Dropout（纯手写循环）
    model_drop = MLP([5, 10, 10, 5, 2])
    dropout1 = Dropout(p=0.2)
    dropout2 = Dropout(p=0.2)
    dropout3 = Dropout(p=0.2)
    ln = LayerNorm(10)
    optimizer_drop = Adam(model_drop, lr=0.005)

    print(f"模型: {model_drop} + Dropout(0.2)×3 + LayerNorm(10)")
    print(f"训练集: {len(train_data)} 样本")

    for epoch in range(1, 301):
        random.shuffle(train_data)
        total_loss = 0.0
        for x, y in train_data:
            # 前向: Layer0 → Dropout → Layer1 → LN → Dropout → Layer2 → Dropout → Layer3
            x = model_drop.layers[0].forward(list(x))
            x = dropout1.forward(x)
            x = model_drop.layers[1].forward(x)
            x = ln.forward(x)
            x = dropout2.forward(x)
            x = model_drop.layers[2].forward(x)
            x = dropout3.forward(x)
            pred = model_drop.layers[3].forward(x)

            loss, grad = mse_loss(pred, list(y))
            total_loss += loss
            delta = grad

            # 反向
            gw3, gb3, delta = model_drop.layers[3].backward(delta)
            delta = dropout3.backward(delta)
            gw2, gb2, delta = model_drop.layers[2].backward(delta)
            delta = dropout2.backward(delta)
            gg, gb_ln, delta = ln.backward(delta)
            gw1, gb1, delta = model_drop.layers[1].backward(delta)
            delta = dropout1.backward(delta)
            gw0, gb0, delta = model_drop.layers[0].backward(delta)

            # 更新 LN 参数 (简化 SGD)
            ln_lr = 1e-3
            for j in range(10):
                ln.gamma[j] -= ln_lr * gg[j]
                ln.beta[j] -= ln_lr * gb_ln[j]

            grads = [(gw0, gb0), (gw1, gb1), (gw2, gb2), (gw3, gb3)]
            optimizer_drop.step(grads)

        if epoch % 100 == 0:
            print(f"  Epoch {epoch} | Loss: {total_loss / len(train_data):.6f}")

    # 推理时关 Dropout
    dropout1.eval(); dropout2.eval(); dropout3.eval()   # 推理模式：关闭 dropout
    for x, y in test_data:
        x = model_drop.layers[0].forward(list(x))
        x = dropout1.forward(x)
        x = model_drop.layers[1].forward(x)
        x = ln.forward(x)
        x = dropout2.forward(x)
        x = model_drop.layers[2].forward(x)
        x = dropout3.forward(x)
        pred = model_drop.layers[3].forward(x)
        print(f"  输入: {x}  预测: {[round(p, 4) for p in pred]}  目标: {y}")

    # ==================== Gradient Clipping + LR Scheduler（高层 API） ====================
    print("\n" + "=" * 60)
    print("Gradient Clipping + StepLR（高层 API）")
    print("=" * 60)

    random.seed(42)

    model_gc = make_regressor(input_dim=5, output_dim=2,
                              hidden_layers=(8, 8, 4), lr=0.01)
    scheduler = StepLR(model_gc.optimizer, step_size=100, gamma=0.5)

    model_gc.fit(x_reg, y_reg, epochs=300, batch_size=2, patience=0,
                 lr_scheduler=scheduler, grad_clip=1.0)

    print(f"  最终 Loss: {model_gc.history['train_loss'][-1][1]:.6f}  |  LR: {model_gc.optimizer.lr:.6f}")
    r2 = model_gc.score(x_reg, y_reg)
    print(f"  回归 R²: {r2:.4f}")

    # ==================== quick_train 一行训练 ====================
    print("\n" + "=" * 60)
    print("quick_train: 一行代码训练")
    print("=" * 60)

    random.seed(42)

    # 一行回归
    qm_reg = quick_train(x_reg, y_reg, task='regression',
                        hidden_layers=(16, 8), epochs=300, patience=20)
    r2 = qm_reg.score(x_reg, y_reg)
    print(f"  回归 R² = {r2:.4f}")

    # 一行分类
    qm_cls = quick_train(x_cls, y_cls, task='classification',
                         hidden_layers=(16,), epochs=300, patience=20)
    acc2 = qm_cls.score(x_cls, y_cls)
    print(f"  分类准确率 = {acc2:.1%}")

    # ==================== Transformer 示例 ====================
    print("\n" + "=" * 60)
    print("Transformer 示例: 简单序列分类")
    print("=" * 60)

    random.seed(42)
    d_model = 4
    num_heads = 2
    d_ff = 8

    # 构建小型 Transformer + 全局平均池化 + 分类头
    pos_enc = PositionalEncoding(d_model)
    encoder = TransformerEncoder(d_model, num_heads, d_ff, num_layers=1)
    cls_w = [[random.gauss(0, math.sqrt(2.0 / d_model)) for _ in range(d_model)] for _ in range(2)]
    cls_b = [0.0] * 2

    # 玩具序列数据: 3 个 token, 每个 d_model=4 维, 二分类
    trans_data = [
        ([[0.1, 0.2, 0.3, 0.4], [0.5, 0.1, 0.2, 0.8], [0.3, 0.7, 0.4, 0.2]], 0),
        ([[0.2, 0.1, 0.4, 0.3], [0.6, 0.2, 0.1, 0.7], [0.2, 0.8, 0.3, 0.1]], 0),
        ([[0.9, 0.8, 0.1, 0.2], [0.3, 0.9, 0.7, 0.1], [0.8, 0.2, 0.6, 0.3]], 1),
        ([[0.8, 0.9, 0.2, 0.1], [0.2, 0.8, 0.9, 0.1], [0.7, 0.1, 0.5, 0.4]], 1),
    ]

    encoder.train()   # 显式声明训练模式（dropout 生效）

    for epoch in range(1, 201):
        random.shuffle(trans_data)
        total_loss = 0.0
        for x_seq, label in trans_data:
            # 前向
            x = pos_enc.forward(x_seq)        # (3, 4) + PE
            x = encoder.forward(x)            # (3, 4) 经过 Transformer

            # 池化: 取各 token 均值作为序列表示
            pooled = [sum(x[t][j] for t in range(len(x))) / len(x) for j in range(d_model)]

            # 分类头
            logits = [cls_b[j] + sum(cls_w[j][i] * pooled[i] for i in range(d_model)) for j in range(2)]
            probs = _softmax_row(logits)

            loss = -math.log(max(probs[label], 1e-15))
            total_loss += loss

            # 反向: d_probs = softmax - one_hot
            d_logits = list(probs)
            d_logits[label] -= 1.0

            # 分类头梯度
            grad_cls_w = [[d_logits[j] * pooled[i] for i in range(d_model)] for j in range(2)]
            grad_cls_b = list(d_logits)

            # 传回池化层
            d_pooled = [sum(cls_w[j][i] * d_logits[j] for j in range(2)) / len(x)
                        for i in range(d_model)]
            d_encoder = [d_pooled for _ in range(len(x))]
            encoder.zero_grad()
            d_x = encoder.backward(d_encoder)

            # 更新 (SGD 简化): 分类头 + Transformer encoder 参数
            lr = 0.01
            for j in range(2):
                cls_b[j] -= lr * grad_cls_b[j]
                for i in range(d_model):
                    cls_w[j][i] -= lr * grad_cls_w[j][i]
            encoder.sgd_step(lr)

        if epoch % 50 == 0:
            print(f"  Epoch {epoch} | Loss: {total_loss / len(trans_data):.6f}")

    encoder.eval()    # 推理前必须关闭 dropout，否则同一输入两次结果不同
    print("Transformer 测试:")
    for x_seq, label in trans_data:
        x = pos_enc.forward(x_seq)
        x = encoder.forward(x)
        pooled = [sum(x[t][j] for t in range(len(x))) / len(x) for j in range(d_model)]
        logits = [cls_b[j] + sum(cls_w[j][i] * pooled[i] for i in range(d_model)) for j in range(2)]
        probs = _softmax_row(logits)
        pred = 1 if probs[1] > probs[0] else 0
        print(f"  预测: {pred}  实际: {label}  {'✓' if pred == label else '✗'}  probs={[round(p, 3) for p in probs]}")

    # ==================== 因果语言模型示例（causal mask → LLM） ====================
    print("\n" + "=" * 60)
    print("因果语言模型: causal mask 让每个位置只能看到左侧（GPT 式预训练）")
    print("=" * 60)

    random.seed(42)
    lm = CausalLM(vocab_size=3, d_model=16, num_heads=2, d_ff=32,
                  num_layers=1, max_len=32)
    lm_ids = [0, 1, 2] * 5
    hist = lm.fit([lm_ids], epochs=120, lr=0.01, shuffle=False)
    print(f"  语料: 012 循环 ×5 | loss {hist[0]:.4f} → {hist[-1]:.4f}")
    lm_prompt = [0]
    lm_out = lm.generate(lm_prompt, max_new_tokens=8, temperature=1e-6)
    print(f"  前缀 {''.join(map(str, lm_prompt))} → 续写 "
          f"{''.join(map(str, lm_out))}  （期望 012012012…）")

    # ==================== 迷你 GPT：BPE 分词 + 存档续训 + 束搜索 ====================
    print("\n" + "=" * 60)
    print("迷你 GPT: BPE 分词 → 训练 → 存档 → 续训 → 束搜索")
    print("=" * 60)

    random.seed(7)
    mini_corpus = "the quick brown fox jumps over the lazy dog. " * 6
    mini_bpe = BytePairTokenizer().train(mini_corpus, vocab_size=320)
    mini_seqs = mini_bpe.encode_corpus(mini_corpus, seq_len=24)
    raw_bytes = len(mini_corpus.encode("utf-8"))
    print(f"  语料 {raw_bytes} 字节 → 词表 {mini_bpe.vocab_size}"
          f"（256 基础 + {len(mini_bpe.merges)} 合并）| {len(mini_seqs)} 条序列")

    random.seed(11)
    mini_lm = CausalLM(vocab_size=mini_bpe.vocab_size, d_model=24, num_heads=2,
                       d_ff=48, num_layers=1, max_len=24)
    mini_hist = mini_lm.fit(mini_seqs, epochs=30, lr=0.01, shuffle=True, seed=3)
    print(f"  训练 30 轮: loss {mini_hist[0]:.4f} → {mini_hist[-1]:.4f}"
          f" | 参数 {mini_lm.param_count()}")

    mini_path = "demo_mini_gpt.json"
    mini_lm.save(mini_path, include_optimizer=True, quiet=True)
    mini_reload = CausalLM.load(mini_path, restore_optimizer=True)
    mini_hist2 = mini_reload.fit(mini_seqs, epochs=5, lr=0.01, shuffle=True, seed=3)
    print(f"  存档 → 重载 → 续训 5 轮: loss {mini_hist2[0]:.4f} → {mini_hist2[-1]:.4f}")

    mini_prompt = mini_bpe.encode("the quick")
    greedy = mini_lm.generate(mini_prompt, max_new_tokens=10, top_k=1)
    beam = mini_lm.generate(mini_prompt, max_new_tokens=10, num_beams=3)
    print(f"  贪心续写: {mini_bpe.decode(greedy)!r}")
    print(f"  束搜索  : {mini_bpe.decode(beam)!r}")
    try:
        os.remove(mini_path)
    except OSError:
        pass

    # ==================== 文本生成示例（字符级语言模型） ====================
    print("\n" + "=" * 60)
    print("文本生成示例: 字符级语言模型（唐诗风格）")
    print("=" * 60)

    random.seed(42)

    # 语料库
    text = "君不见黄河之水天上来奔流到海不复回君不见高堂明镜悲白发朝如青丝暮成雪人生得意须尽欢莫使金樽空对月"
    window_size = 3
    tp = TextProcessor(text)
    dataset = tp.prepare_data(text, window_size)

    # 分割训练/测试
    random.shuffle(dataset)
    split_idx = int(len(dataset) * 0.85)
    text_train, text_test = dataset[:split_idx], dataset[split_idx:]

    input_dim = window_size * tp.vocab_size
    text_model = MLP([input_dim, 64, tp.vocab_size], activations=["leaky_relu", "softmax"])
    text_opt = Adam(text_model, lr=0.005)

    print(f"词表大小: {tp.vocab_size} ({''.join(tp.chars)})")
    print(f"窗口大小: {window_size} | 输入维度: {input_dim}")
    print(f"训练集: {len(text_train)} 样本 | 测试集: {len(text_test)} 样本")

    for epoch in range(1, 201):
        random.shuffle(text_train)
        total_loss = 0.0
        for x, y in text_train:
            probs = text_model.forward(x)
            loss = -sum(y[j] * math.log(max(probs[j], 1e-15)) for j in range(len(y)))
            total_loss += loss

            # Cross Entropy 梯度: probs - y (softmax + CE 联合梯度)
            delta = [probs[j] - y[j] for j in range(len(y))]
            grads = []
            for layer in reversed(text_model.layers):
                gw, gb, delta = layer.backward(delta)
                grads.append((gw, gb))
            grads.reverse()
            text_opt.step(grads)

        if epoch % 50 == 0:
            # 测试集准确率
            correct = 0
            for x, y in text_test:
                probs = text_model.forward(x)
                pred_idx = probs.index(max(probs))
                true_idx = y.index(max(y))
                if pred_idx == true_idx:
                    correct += 1
            acc = correct / len(text_test)
            print(f"  Epoch {epoch} | Loss: {total_loss / len(text_train):.4f} | Test Acc: {acc:.2%}")

    # 生成文本
    print("\n文本生成测试:")
    start_key = "君不见"
    for t in [0.3, 0.7, 1.2]:
        result = generate_text_with_mlp(text_model, tp, start_key, length=20,
                                        window_size=window_size, temp=t)
        print(f"  温度 {t:.1f}: {result}")

    # ==================== PyTorch 对齐 ====================
    compare_with_pytorch()

    print("\n" + "=" * 60)
    print("全部示例运行完成!")
    print("=" * 60)
