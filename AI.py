"""
================================================================================
  全栈神经网络工具箱（纯 Python 标准库实现，零第三方依赖）
  - Layer / MLP（全连接层 + 激活函数）
  - Conv2d / MaxPool2d（卷积与池化）
  - RNN / LSTM / GRU（循环神经网络家族，含完整 BPTT）
  - Transformer（MultiHeadAttention / FeedForward / Encoder / PositionalEncoding）
  - Dropout / BatchNorm1d / LayerNorm（正则化与归一化）
  - Embedding / ResidualBlock / Sequential（基础组件）
  - Adam / SGD / SGDMomentum / RMSprop（优化器）
  - MSE / CrossEntropy 损失函数
  - 梯度裁剪 / 学习率调度器（StepLR / ExponentialLR / CosineAnnealingLR）
  - 训练 / 验证 / 测试集拆分 + Early Stopping + DataLoader
  - 模型保存 / 加载（JSON 格式）
================================================================================
"""

import json
import sys
import io

import math
import operator
import random
from typing import List, Tuple, Optional, Callable, Any

# 热路径别名：Layer / _matmul 每次调用都要用到 operator.mul
_mul = operator.mul


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
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # 记录激活函数名字符串（save_model 持久化 / 激活与损失的搭配校验用）
        self.activation_name = activation

    # ---------- 激活函数 ----------
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

    # ---------- 前向 / 反向 ----------
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
        """delta: 本层输出的梯度，返回上一层需要的 delta"""
        d_act = self.act_deriv(self.z)

        # delta * d_act 同时是 grad_b 与回传给上一层的权重向量，只算一次
        w = [delta[j] * d_act[j] for j in range(self.fan_out)]
        grad_w = [[s * xi for xi in self.input] for s in w]
        grad_b = list(w)

        # 传给上一层的 delta = W^T · (delta * d_act)
        # zip(*W) 在 C 层完成转置，再用 map(mul) 做点积，比双重索引的 genexpr 快约 1.9 倍
        next_delta = [sum(map(_mul, col, w)) for col in zip(*self.weights)]

        return grad_w, grad_b, next_delta


class MLP:
    """多层感知机（PyTorch 风格）"""

    def __init__(self, layer_sizes: List[int],
                 activations: Optional[List[str]] = None):
        self.layer_sizes = layer_sizes

        if activations is None:
            activations = ["leaky_relu"] * (len(layer_sizes) - 2)
            activations.append("linear")

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

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def get_parameters(self):
        params = []
        for layer in self.layers:
            params.append((layer.weights, layer.biases))
        return params

    # ── 参数量与模型瘦身（与 C 引擎 c_model.cpp 的实现语义一致） ──

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
                 kernel_size, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.padding = (padding, padding) if isinstance(padding, int) else padding
        self.kh, self.kw = self.kernel_size
        self.sh, self.sw = self.stride
        self.ph, self.pw = self.padding

        # He 初始化
        fan_in = in_channels * self.kh * self.kw
        std = math.sqrt(2.0 / fan_in)
        self.weights = [
            [[[random.gauss(0, std) for _ in range(self.kw)]
              for _ in range(self.kh)]
             for _ in range(in_channels)]
            for _ in range(out_channels)
        ]
        self.biases = [0.0] * out_channels

        # Adam 状态
        self.m_w = [[[[0.0] * self.kw for _ in range(self.kh)]
                      for _ in range(in_channels)] for _ in range(out_channels)]
        self.v_w = [[[[0.0] * self.kw for _ in range(self.kh)]
                      for _ in range(in_channels)] for _ in range(out_channels)]
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
             for _ in range(self.in_channels)]
            for _ in range(self.out_channels)
        ]
        self.grad_b = [0.0] * self.out_channels

    # ---------- 辅助 ----------
    def _img_shape(self, H, W):
        ho = (H + 2 * self.ph - self.kh) // self.sh + 1
        wo = (W + 2 * self.pw - self.kw) // self.sw + 1
        # 输入比卷积核还小时 ho/wo 会是 0 或负数，原实现返回空输出而不报错
        if ho <= 0 or wo <= 0:
            raise ValueError(
                f"Conv2d: 输入 {H}x{W} 配 kernel {self.kh}x{self.kw}、"
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
                            dst[oh * wo:(oh + 1) * wo] = ch[oh + kh][kw:kw + wo]
            return col

        for c in range(C):
            ch = x_padded[c]                      # 局部化，省掉每元素的二次索引
            for kh in range(self.kh):
                for kw in range(self.kw):
                    row = c * self.kh * self.kw + kh * self.kw + kw
                    dst = col[row]
                    for oh in range(ho):
                        src = ch[oh * self.sh + kh]
                        off = oh * wo
                        for ow in range(wo):
                            dst[off + ow] = src[ow * self.sw + kw]
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
                            ih = oh * self.sh + kh
                            iw = ow * self.sw + kw
                            dx[c][ih][iw] += dcol[row][oh * wo + ow]
        return dx

    def _flatten_weights(self, oc):
        """把第 oc 个输出通道的卷积核展平成 (C*kh*kw,)，顺序与 im2col 的行一致"""
        out = []
        for ic in range(self.in_channels):
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

    # ---------- 前向 / 反向 ----------
    def forward(self, x):
        """x: (C, H, W) -> 输出 (out_channels, Ho, Wo)"""
        C, H, W = len(x), len(x[0]), len(x[0][0])
        ho, wo = self._img_shape(H, W)
        self.h_out, self.w_out = ho, wo
        self.x_shape = (C, H, W)

        x_pad = self._pad(x) if (self.ph > 0 or self.pw > 0) else x
        col = self._im2col(x_pad, ho, wo)       # (C*kh*kw, ho*wo)
        self.x_col = col

        # 一次 matmul 算出所有输出通道: (out_channels, C*kh*kw) × (C*kh*kw, ho*wo)
        W_row = [self._flatten_weights(oc) for oc in range(self.out_channels)]
        flat = _matmul(W_row, col)
        return [
            [[flat[oc][oh * wo + ow] + self.biases[oc] for ow in range(wo)]
             for oh in range(ho)]
            for oc in range(self.out_channels)
        ]

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

        # grad_w: dW = dout_flat × col^T（累积到内部梯度缓冲区）
        # 即: grad_w[oc][ic][kh][kw] += sum over ho,wo: dout[oc][oh][ow] * col_row[oh*wo]
        # 点积交给 C 层的 map(mul)/sum，求和顺序与逐元素累加一致
        for oc in range(self.out_channels):
            d = dout_flat[oc]
            for ic in range(self.in_channels):
                for kh in range(self.kh):
                    for kw in range(self.kw):
                        row = ic * self.kh * self.kw + kh * self.kw + kw
                        self.grad_w[oc][ic][kh][kw] += sum(map(_mul, d, self.x_col[row]))

        for oc in range(self.out_channels):
            self.grad_b[oc] += sum(dout_flat[oc])

        # dx: col2im  dcol = W_flat^T × dout_flat  (C*kh*kw, ho*wo)
        dcol = [[0.0] * (ho * wo) for _ in range(len(self.x_col))]
        for oc in range(self.out_channels):
            d = dout_flat[oc]
            w_flat = self._flatten_weights(oc)
            for row in range(len(w_flat)):
                w = w_flat[row]
                # 累加顺序仍是「外层 oc、内层逐元素」，与原写法逐位一致
                dcol[row] = [a + w * b for a, b in zip(dcol[row], d)]

        dx_padded = self._col2im(dcol, (C, H + 2 * self.ph, W + 2 * self.pw), ho, wo)

        if self.ph > 0 or self.pw > 0:
            return self._unpad(dx_padded, H, W)
        return dx_padded


class MaxPool2d:
    """2D 最大池化层"""

    def __init__(self, kernel_size, stride=None):
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride = self.kernel_size if stride is None else (
            (stride, stride) if isinstance(stride, int) else stride
        )
        self.kh, self.kw = self.kernel_size
        self.sh, self.sw = self.stride
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
        self.x_batch = x_batch
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
            return [[self.gamma[j] * ((row[j] - self.running_mean[j]) /
                    math.sqrt(self.running_var[j] + self.eps)) + self.beta[j]
                     for j in range(self.d)] for row in x]
        out = [0.0] * self.d
        for j in range(self.d):
            xh = (x[j] - self.running_mean[j]) / math.sqrt(self.running_var[j] + self.eps)
            out[j] = self.gamma[j] * xh + self.beta[j]
        return out

    def backward(self, dout_batch):
        """返回 (dx_batch, grad_gamma, grad_beta)"""
        n, d = self.n, self.d

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
        self.x = x
        self.mean = sum(x) / self.d
        self.var = sum((xi - self.mean) ** 2 for xi in x) / self.d
        self.inv_std = 1.0 / math.sqrt(self.var + self.eps)
        self.x_hat = [(xi - self.mean) * self.inv_std for xi in x]
        return [self.gamma[j] * self.x_hat[j] + self.beta[j] for j in range(self.d)]

    def backward(self, dout):
        d = self.d
        grad_gamma = [dout[j] * self.x_hat[j] for j in range(d)]
        grad_beta = list(dout)

        dx_hat = [dout[j] * self.gamma[j] for j in range(d)]
        s1 = sum(dx_hat)
        s2 = sum(dx_hat[j] * self.x_hat[j] for j in range(d))

        dx = [self.inv_std * (d * dx_hat[j] - s1 - self.x_hat[j] * s2) / d for j in range(d)]
        return grad_gamma, grad_beta, dx  # 约定: 最后一个元素为传播的 delta


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
        return [1.0 / (1.0 + math.exp(-xi)) for xi in x]

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
    assert n == n2, f"维度不匹配: ({m},{n}) × ({n2},{p})"
    # 先把 B 转置一次，让每一列成为元组，再用 C 层的 map(mul) / sum 做点积：
    # 与逐个元素 += 的三重循环结果完全一致，但快约 1.6 倍
    Bt = list(zip(*B))
    return [[sum(map(_mul, row, col)) for col in Bt] for row in A]


def _transpose(M):
    """转置 (m×n) → (n×m)"""
    m, n = len(M), len(M[0])
    return [[M[i][j] for i in range(m)] for j in range(n)]


def _softmax_row(v):
    """对一维列表做 softmax"""
    mx = max(v)
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
        return [1.0 / (1.0 + math.exp(-xi)) for xi in x]

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

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        """dropout: **目前未实现**（注意力权重上不做 dropout，参数保留仅为接口兼容）。

        TransformerEncoderLayer 的 dropout1/dropout2 仍然工作，所以 eval() 依然有意义；
        但这里传大于 0 的值不会带来任何正则化效果。
        """
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.dropout_p = dropout

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
        self.grads = {
            "W_q": zeros_2d(d, d), "b_q": [0.0] * d,
            "W_k": zeros_2d(d, d), "b_k": [0.0] * d,
            "W_v": zeros_2d(d, d), "b_v": [0.0] * d,
            "W_o": zeros_2d(d, d), "b_o": [0.0] * d,
        }

    def _init_adam(self):
        d = self.d_model
        def zeros_2d(r, c):
            return [[0.0] * c for _ in range(r)]
        self.m_W_q = zeros_2d(d, d); self.v_W_q = zeros_2d(d, d)
        self.m_W_k = zeros_2d(d, d); self.v_W_k = zeros_2d(d, d)
        self.m_W_v = zeros_2d(d, d); self.v_W_v = zeros_2d(d, d)
        self.m_W_o = zeros_2d(d, d); self.v_W_o = zeros_2d(d, d)
        self.m_b_q = [0.0] * d; self.v_b_q = [0.0] * d
        self.m_b_k = [0.0] * d; self.v_b_k = [0.0] * d
        self.m_b_v = [0.0] * d; self.v_b_v = [0.0] * d
        self.m_b_o = [0.0] * d; self.v_b_o = [0.0] * d

    def forward(self, x_seq):
        """x_seq: List[List[float]]  (seq_len, d_model)"""
        # 复制一份，避免调用者之后原地修改 x_seq 污染反向传播缓存
        self.x = [list(row) for row in x_seq]
        self.seq_len = len(x_seq)
        d_k = self.d_k
        h = self.num_heads

        # 线性投影 Q, K, V
        q_full = _matmul(x_seq, _transpose(self.W_q))  # (seq_len, d_model)
        k_full = _matmul(x_seq, _transpose(self.W_k))
        v_full = _matmul(x_seq, _transpose(self.W_v))
        # 加 bias
        for t in range(self.seq_len):
            for j in range(self.d_model):
                q_full[t][j] += self.b_q[j]
                k_full[t][j] += self.b_k[j]
                v_full[t][j] += self.b_v[j]

        # 拆分为多头: (seq_len, d_model) → h × (seq_len, d_k)
        self.Q = []
        self.K = []
        self.V = []
        for head in range(h):
            offset = head * d_k
            q_h = [[q_full[t][offset + j] for j in range(d_k)] for t in range(self.seq_len)]
            k_h = [[k_full[t][offset + j] for j in range(d_k)] for t in range(self.seq_len)]
            v_h = [[v_full[t][offset + j] for j in range(d_k)] for t in range(self.seq_len)]
            self.Q.append(q_h)
            self.K.append(k_h)
            self.V.append(v_h)

        # 各头分别计算注意力
        scale = 1.0 / math.sqrt(d_k)
        self.attn_weights = []
        self.head_out = []
        self.concat = [[0.0] * self.d_model for _ in range(self.seq_len)]

        for head in range(h):
            q_h, k_h, v_h = self.Q[head], self.K[head], self.V[head]

            # scores = Q @ K^T / sqrt(d_k)  (seq_len, seq_len)
            k_t = _transpose(k_h)  # (d_k, seq_len)
            scores = [[v * scale for v in row] for row in _matmul(q_h, k_t)]

            # softmax per row
            attn = [_softmax_row(row) for row in scores]

            # output = attn @ V  (seq_len, d_k)
            out = _matmul(attn, v_h)

            self.attn_weights.append(attn)
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
            attn = self.attn_weights[head]
            V_h = self.V[head]
            Q_h = self.Q[head]
            K_h = self.K[head]
            d_out = d_head_out[head]

            # dV = attn^T @ d_out  (seq_len, d_k)
            dV = _matmul(_transpose(attn), d_out)

            # d_attn = d_out @ V^T  (seq_len, seq_len)
            d_attn = _matmul(d_out, _transpose(V_h))

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

            dQ_list.append(dQ)
            dK_list.append(dK)
            dV_list.append(dV)

        # 4. 拼接各头的 dQ, dK, dV → (seq_len, d_model)
        dQ_full = [[0.0] * d_model for _ in range(seq_len)]
        dK_full = [[0.0] * d_model for _ in range(seq_len)]
        dV_full = [[0.0] * d_model for _ in range(seq_len)]
        for head in range(h):
            offset = head * d_k
            for t in range(seq_len):
                for j in range(d_k):
                    dQ_full[t][offset + j] = dQ_list[head][t][j]
                    dK_full[t][offset + j] = dK_list[head][t][j]
                    dV_full[t][offset + j] = dV_list[head][t][j]

        # 5. 通过投影矩阵反传梯度
        # 前向是 q_full = x @ W_q^T，即 W_q[b][a] 表示「输入 a → 输出 b」，
        # 所以 grad_W_q[b][a] = sum_t dQ_full[t][b] * x[t][a]，即 dQ_full^T @ x。
        # （不能写成 x^T @ dQ_full：那是正确梯度的转置，方阵下不会报错但会静默训歪）
        grad_W_q = _matmul(_transpose(dQ_full), self.x)  # (d_model, d_model)
        grad_W_k = _matmul(_transpose(dK_full), self.x)
        grad_W_v = _matmul(_transpose(dV_full), self.x)
        grad_b_q = [sum(dQ_full[t][j] for t in range(seq_len)) for j in range(d_model)]
        grad_b_k = [sum(dK_full[t][j] for t in range(seq_len)) for j in range(d_model)]
        grad_b_v = [sum(dV_full[t][j] for t in range(seq_len)) for j in range(d_model)]

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
                self.grads["W_k"][i][j] += grad_W_k[i][j]
                self.grads["W_v"][i][j] += grad_W_v[i][j]
                self.grads["W_o"][i][j] += grad_W_o[i][j]
            self.grads["b_q"][i] += grad_b_q[i]
            self.grads["b_k"][i] += grad_b_k[i]
            self.grads["b_v"][i] += grad_b_v[i]
            self.grads["b_o"][i] += grad_b_o[i]
        return dx_seq


class FeedForward:
    """Transformer 前馈网络: Linear → ReLU → Linear"""

    def __init__(self, d_model: int, d_ff: int):
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
        # W1 @ x + b1 → ReLU → W2 @ x + b2
        h = _matmul(x_seq, _transpose(self.W1))
        for t in range(self.seq_len):
            for j in range(self.d_ff):
                h[t][j] = max(0, h[t][j] + self.b1[j])  # ReLU
        self.h = h
        out = _matmul(h, _transpose(self.W2))
        for t in range(self.seq_len):
            for j in range(self.d_model):
                out[t][j] += self.b2[j]
        return out

    def backward(self, dout_seq):
        """dout_seq: (seq_len, d_model) → dx_seq: (seq_len, d_model)"""
        seq_len = self.seq_len

        # dout → W2:  dW2 = dout^T @ h  (d_model, seq_len) @ (seq_len, d_ff) = (d_model, d_ff)
        grad_W2 = _matmul(_transpose(dout_seq), self.h)
        grad_b2 = [sum(dout_seq[t][j] for t in range(seq_len)) for j in range(self.d_model)]

        # d_h = dout @ W2
        dh = _matmul(dout_seq, self.W2)  # (seq_len, d_ff)

        # ReLU backward
        for t in range(seq_len):
            for j in range(self.d_ff):
                if self.h[t][j] <= 0:
                    dh[t][j] = 0.0

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

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
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
        """训练模式：两个 Dropout 都生效"""
        self.dropout1.train()
        self.dropout2.train()
        return self

    def eval(self):
        """推理模式：关闭 Dropout，保证同一输入输出确定"""
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

    def forward(self, x_seq):
        self.x = x_seq
        d_model = self.self_attn.d_model
        # Self-Attention + Dropout1（逐 token 存储 mask）
        attn_out = self.self_attn.forward(x_seq)
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
                 num_layers: int = 6, dropout: float = 0.1):
        self.layers = [
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
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

    def forward(self, x_seq):
        for layer in self.layers:
            x_seq = layer.forward(x_seq)
        return x_seq

    def backward(self, dout_seq):
        for layer in reversed(self.layers):
            dout_seq = layer.backward(dout_seq)
        return dout_seq

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


class ResidualBlock:
    """残差连接：output = x + F(x)"""

    def __init__(self, sublayer):
        self.sublayer = sublayer
        self.x = None

    def forward(self, x):
        self.x = x
        out = self.sublayer.forward(x)
        if isinstance(x, list) and isinstance(x[0], list):
            return [[x[t][j] + out[t][j] for j in range(len(x[0]))] for t in range(len(x))]
        return [x[i] + out[i] for i in range(len(x))]

    def backward(self, dout):
        # sublayer 的 backward 返回值有两种约定（同 Sequential）：
        #   元组 (..., delta) 取其末位；list 则整体就是 delta。
        # Layer / LayerNorm 返回元组，Dropout / Embedding / 注意力等返回 list。
        result = self.sublayer.backward(dout)
        d_sublayer = result[-1] if isinstance(result, tuple) else result
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
                delta = result[-1] if isinstance(result, tuple) else result
        return delta


# ============================================================================
#  Loss Functions
# ============================================================================

def mse_loss(pred: List[float], target):
    """MSE 损失 + 对 pred 的梯度

    target 可以是序列，也可以是单个标量（单输出回归传 y=0.5 也行，
    与 Model._wrap_loss 的行为保持一致）。
    """
    if not isinstance(target, (list, tuple)):
        target = [target]
    if len(pred) != len(target):
        # 不能靠 zip 静默截断：分母用的是 len(pred)，截断后会得到一个"看起来正常"的错值
        raise ValueError(
            f"mse_loss: pred 长度 {len(pred)} != target 长度 {len(target)}，"
            f"请对齐维度（单输出回归请传 y=[value]，多输出请传 y=[v1, v2, ...]）"
        )
    n = len(pred)
    loss = sum((p - t) ** 2 for p, t in zip(pred, target)) / n
    grad = [2 * (p - t) / n for p, t in zip(pred, target)]
    return loss, grad


def cross_entropy_loss(pred: List[float], target_idx):
    """
    pred: softmax 输出 (概率分布)
    target_idx: 正确类别的索引 (int)，或 one-hot / 概率向量 (list)

    两种目标都支持：
        索引     -> loss = -log(pred[idx])，梯度 = pred - one_hot(idx)
        向量     -> loss = -Σ t·log(pred)，梯度 = pred - t
    后者用于 TextProcessor.prepare_data 产出的 one-hot 标签。
    """
    eps = 1e-15
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
    p = max(pred[target_idx], eps)
    loss = -math.log(p)
    # softmax + CE 的梯度 = pred - one_hot(target)
    grad = list(pred)
    grad[target_idx] -= 1.0
    return loss, grad


# ============================================================================
#  Optimizer: Adam
# ============================================================================

class Adam:
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


# ============================================================================
#  Gradient Clipping / LR Scheduler
# ============================================================================

def clip_grad_by_norm(all_grads, max_norm: float):
    """按范数裁剪梯度，防止梯度爆炸（适用于 RNN/LSTM/GRU）"""
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
    for gw, gb in all_grads:
        for j in range(len(gw)):
            for i in range(len(gw[j])):
                gw[j][i] = max(-clip_val, min(clip_val, gw[j][i]))
            gb[j] = max(-clip_val, min(clip_val, gb[j]))


class StepLR:
    """每 step_size 个 epoch 将 lr 乘以 gamma"""

    def __init__(self, optimizer, step_size: int = 30, gamma: float = 0.1):
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
        self.optimizer = optimizer
        self.T_max = T_max
        self.eta_min = eta_min
        self.initial_lr = optimizer.lr

    def step(self, epoch: int):
        # 支持周期重启：epoch 每 T_max 后重新开始余弦退火周期
        progress = ((epoch - 1) % self.T_max) / self.T_max
        self.optimizer.lr = self.eta_min + 0.5 * (self.initial_lr - self.eta_min) * (
            1 + math.cos(math.pi * progress))


# ============================================================================
#  Data Utilities
# ============================================================================

def split_data(data, val_ratio: float = 0.2, test_ratio: float = 0.1, seed: int = 42):
    """将数据拆分为 train / val / test"""
    # 用局部 Random 实例，避免 random.seed() 污染全局随机状态
    rng = random.Random(seed)
    items = list(data)
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

    def __init__(self, dataset: list, batch_size: int = 32, shuffle: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        items = list(self.dataset)
        if self.shuffle:
            random.shuffle(items)
        for b in range(0, len(items), self.batch_size):
            yield items[b:b + self.batch_size]

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


# ============================================================================
#  Text Processing（字符级语言模型工具）
# ============================================================================

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
        """将字符串编码为索引列表（strict=True 时未知字符报错，而非静默映射成 0）"""
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
    if temperature != 1.0:
        logits = [x / temperature for x in logits]
    mx = max(logits)
    exps = [math.exp(x - mx) for x in logits]
    s = sum(exps)
    return [e / s for e in exps]


def sample_index(probs: list) -> int:
    """根据概率分布进行随机采样（不依赖 numpy）"""
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
        logits = model.forward(test_vec)
        probs = _softmax_with_temp(logits, temp)
        next_idx = sample_index(probs)
        generated += tp.idx_to_char[next_idx]
    return generated


# ============================================================================
#  Trainer
# ============================================================================

def train(model: MLP, loss_fn: Callable, optimizer: Adam, data: list,
          epochs: int = 2000, batch_size: int = 2, patience: int = 50,
          val_data: Optional[list] = None, verbose: bool = True):
    """
    loss_fn: 接受 (pred, target) -> (loss, grad)

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

    # 复制一份再打乱，避免修改调用者传入的列表顺序
    data = list(data)

    for epoch in range(1, epochs + 1):
        # ---- 训练 ----
        random.shuffle(data)
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

        # ---- 验证 ----
        val_loss = None
        if val_data:
            val_loss = evaluate(model, loss_fn, val_data)
            history["val_loss"].append((epoch, val_loss))

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improve = 0
            else:
                no_improve += 1

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
    total = 0.0
    for x, y in data:
        pred = model.forward(x)
        loss, _ = loss_fn(pred, y)
        total += loss
    return total / len(data)


# ============================================================================
#  模型保存 / 加载 (JSON)
# ============================================================================



def save_model(model, filepath: str, task: str = None):
    """将 MLP 模型保存为 JSON 文件

    task: 可选，"regression" / "classification"。会写进文件，供旧格式（没有逐层
          activation 记录）加载时判断输出层该用 linear 还是 softmax。
    每层还会写入 activation 字段，因此新格式文件不依赖 task 也能正确还原。
    """
    data = {
        "type": "MLP",
        "layer_sizes": model.layer_sizes,
        # task 用于加载时判断是分类器还是回归器（旧文件没有该字段）
        "task": task,
        "layers": []
    }
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

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
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
        if len(ld["biases"]) != exp_out:
            raise ValueError(
                f"load_model: 第 {i} 层 biases 长度 {len(ld['biases'])} != fan_out={exp_out}"
            )

    model = MLP(layer_sizes, activations=activations)
    for i, layer in enumerate(model.layers):
        layer.weights = saved[i]["weights"]
        layer.biases = saved[i]["biases"]

    # 供 Model.load 判断 task 用（MLP 本身没有 task 概念）
    model.saved_task = data.get("task")

    print(f"模型已从 {filepath} 加载")
    return model


def save_conv_model(conv_layers, fc_model, filepath: str):
    """保存 CNN + MLP 组合模型"""
    data = {
        "type": "CNN_MLP",
        "conv": [],
        "mlp": {
            "layer_sizes": fc_model.layer_sizes,
            "layers": []
        }
    }
    for conv in conv_layers:
        if isinstance(conv, Conv2d):
            data["conv"].append({
                "cls": "Conv2d",
                "in_channels": conv.in_channels,
                "out_channels": conv.out_channels,
                "kernel_size": conv.kernel_size,
                "stride": conv.stride,
                "padding": conv.padding,
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

    # 重建 CNN 层（先校验形状，坏文件在加载期就报错）
    conv_layers = []
    for k, cd in enumerate(data["conv"]):
        conv = Conv2d(cd["in_channels"], cd["out_channels"],
                       cd["kernel_size"], cd["stride"], cd["padding"])
        oc, ic, kh, kw = conv.out_channels, conv.in_channels, conv.kh, conv.kw
        cw = cd["weights"]
        shape_ok = (len(cw) == oc
                    and all(len(ch) == ic for ch in cw)
                    and all(len(row) == kw for ch in cw for m in ch for row in m)
                    and all(len(m) == kh for ch in cw for m in ch))
        if not shape_ok:
            raise ValueError(
                f"load_conv_model: 第 {k} 个 Conv2d 的 weights 形状与超参数不符"
                f"（应为 {oc}×{ic}×{kh}×{kw}）"
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

    # ---------- 设定相同随机种子 & 初始化 ----------
    torch.manual_seed(42)
    random.seed(42)

    input_dim = 4
    hidden_dim = 6
    output_dim = 2

    # --- PyTorch 模型 ---
    pt_model = nn.Sequential(
        nn.Linear(input_dim, hidden_dim, bias=True),
        nn.LeakyReLU(0.01),
        nn.Linear(hidden_dim, hidden_dim, bias=True),
        nn.LeakyReLU(0.01),
        nn.Linear(hidden_dim, output_dim, bias=True),
    )

    # --- 我们的模型 ---
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

    # ---------- 前向对齐 ----------
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

    # ---------- 反向对齐 ----------
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
#  High-Level API: Model 封装（sklearn 风格） + 工厂函数
# ============================================================================

def _mlp_backward(model: 'MLP', loss_grad: list) -> list:
    """内部工具：对 MLP 执行反向传播，返回每层的 (grad_w, grad_b) 列表"""
    delta = loss_grad
    grads = []
    for layer in reversed(model.layers):
        gw, gb, delta = layer.backward(delta)
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
        'sgd': SGD,
        'sgd_momentum': SGDMomentum,
        'rmsprop': RMSprop,
    }

    _LOSS_MAP = {
        'mse': mse_loss,
        'cross_entropy': cross_entropy_loss,
        'ce': cross_entropy_loss,
    }

    def __init__(self, model, loss_fn=None, optimizer=None, lr=0.01,
                 opt_kwargs=None, task='auto'):
        """
        model:       MLP 实例（必传）
        loss_fn:     'mse' / 'cross_entropy' / 'ce' / callable(pred, target) -> (loss, grad)
        optimizer:   'adam' / 'sgd' / 'sgd_momentum' / 'rmsprop' / 优化器实例
        lr:          学习率（optimizer 为字符串时生效）
        opt_kwargs:  dict，传给优化器构造函数的额外参数（如 weight_decay）
        task:        'regression' / 'classification' / 'auto'（自动推断）
        """
        if not isinstance(model, MLP):
            raise TypeError(f"model 必须是 MLP 实例，当前类型: {type(model).__name__}")

        self.model = model
        self.lr = lr
        self.opt_kwargs = opt_kwargs or {}

        # 自动推断任务类型
        if task == 'auto':
            last_layer = model.layers[-1]
            self.task = 'classification' if last_layer.act_fn == last_layer._softmax else 'regression'
        else:
            self.task = task

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
        if self._loss_name == 'mse':
            def _wrap_loss(pred, target):
                if not isinstance(target, (list, tuple)):
                    target = [target]
                return self.loss_fn(pred, target)
            self._loss = _wrap_loss
        elif self._loss_name in ('cross_entropy', 'ce'):
            # 标量 / float / 长度 1 的列表归一化成类别索引；
            # 长度等于类数的 one-hot / 概率向量直接交给 CE 的软标签分支，
            # 不再 argmax 硬化（否则软目标、label smoothing、蒸馏信息全被丢弃）
            num_classes = model.layers[-1].fan_out

            def _wrap_ce_loss(pred, target):
                idx = target
                if isinstance(target, (list, tuple)):
                    if len(target) == num_classes:
                        # 1 类分类器：[1.0] 是唯一的 one-hot，也必须按软标签走，
                        # 否则会被当成索引 1 而越界
                        if num_classes == 1:
                            return self.loss_fn(pred, [float(target[0])])
                        return self.loss_fn(pred, target)
                    if len(target) != 1:
                        raise ValueError(
                            f"cross_entropy 的 target 长度 {len(target)} 与模型输出维度 "
                            f"{num_classes} 不一致（应传类别索引、one-hot 或长度 1 的列表）"
                        )
                    idx = target[0]
                if isinstance(idx, float):
                    if not idx.is_integer():
                        raise ValueError(
                            f"cross_entropy 的 target 必须是整数类别索引，收到 {idx}"
                        )
                    idx = int(idx)
                if not isinstance(idx, int):
                    try:
                        idx = operator.index(idx)  # 兼容 numpy 等实现了 __index__ 的整数类型
                    except TypeError:
                        raise TypeError(
                            f"cross_entropy 的 target 必须是类别索引或 one-hot 列表，收到 {target!r}"
                        )
                if not 0 <= idx < num_classes:
                    raise ValueError(
                        f"类别索引 {idx} 越界（模型输出维度 {num_classes}）"
                    )
                return self.loss_fn(pred, idx)

            self._loss = _wrap_ce_loss
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
            self.optimizer = optimizer

        self.history = {'train_loss': [], 'val_loss': []}
        self._is_trained = False

    # ── 核心操作 ─────────────────────────────────────────────

    def forward(self, x):
        """前向传播：接受单样本 list，返回输出 list"""
        return self.model.forward(x if isinstance(x, list) else list(x))

    def backward(self, loss_grad):
        """反向传播：返回 [(gw, gb), ...] 梯度列表（逐层）"""
        return _mlp_backward(self.model, loss_grad)

    # ── 训练 ─────────────────────────────────────────────────

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
            early_stop_metric='val'):
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
        """
        # === 规范化数据 ===
        if data is not None:
            dataset = [(list(x_i), y_i) for x_i, y_i in data]
        elif x is not None and y is not None:
            # 训练集同样要校验：zip 会静默截断到较短的一方
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
            else:
                random.shuffle(dataset)
                val_set = dataset[:n_val]
                dataset = dataset[n_val:]
        else:
            val_set = None

        if batch_size <= 0:
            batch_size = len(dataset)

        self.history = {'train_loss': [], 'val_loss': []}
        best_val_loss = float('inf')
        no_improve = 0

        for epoch in range(1, epochs + 1):
            if shuffle:
                random.shuffle(dataset)

            train_loss = 0.0
            for b in range(0, len(dataset), batch_size):
                batch = dataset[b:b + batch_size]

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

                # 梯度裁剪
                if grad_clip is not None:
                    clip_grad_by_norm(avg_grads, max_norm=grad_clip)

                self.optimizer.step(avg_grads)

            train_loss /= len(dataset)
            self.history['train_loss'].append((epoch, train_loss))

            # 验证
            val_loss = None
            if not val_set and verbose and patience > 0 and \
                    early_stop_metric != 'train' and epoch == 1:
                print("提示: 没有验证集，early_stop_metric='val' 不会触发早停"
                      "（可用 val_split 或 early_stop_metric='train'）")
            if val_set:
                val_loss = self.evaluate(data=val_set)
                self.history['val_loss'].append((epoch, val_loss))

            # 早停判据：默认盯验证损失（需要验证集）；
            # early_stop_metric='train' 时改盯训练损失，此时不依赖验证集
            # （与 C 引擎 FitConfig.early_stop_on_train 同一语义）
            if patience > 0:
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
                    else:
                        no_improve += 1
                    if no_improve >= patience:
                        if verbose:
                            print(f"Early stopping at epoch {epoch}")
                        break

            # 学习率调度
            if lr_scheduler is not None:
                lr_scheduler.step(epoch)

            # 打印
            if verbose and epoch % max(1, epochs // 10) == 0:
                parts = [f"Epoch {epoch}/{epochs}", f"loss={train_loss:.6f}"]
                if val_loss is not None:
                    parts.append(f"val_loss={val_loss:.6f}")
                print(" | ".join(parts))

        self._is_trained = True
        return self.history

    # ── 推理 ─────────────────────────────────────────────────

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

    # ── 评估 ─────────────────────────────────────────────────

    def evaluate(self, x=None, y=None, data=None):
        """计算平均损失"""
        if data is not None:
            dataset = [(list(xi), yi) for xi, yi in data]
        elif x is not None and y is not None:
            dataset = [(list(xi), yi) for xi, yi in zip(x, y)]
        else:
            raise ValueError("请提供 data 或 (x, y)")

        total = 0.0
        for xi, yi in dataset:
            pred = self.forward(xi)
            loss, _ = self._loss(pred, yi)
            total += loss
        return total / len(dataset)

    def score(self, x, y):
        """分类：准确率 | 回归：R² 决定系数"""
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
            scores = []
            for d in range(n_out):
                y_d = [(yi if not isinstance(yi, (list, tuple)) else yi[d]) for yi in y]
                p_d = [(pi if not isinstance(pi, (list, tuple)) else pi[d]) for pi in preds]
                ss_res = sum((a - b) ** 2 for a, b in zip(y_d, p_d))
                y_mean = sum(y_d) / len(y_d)
                ss_tot = sum((a - y_mean) ** 2 for a in y_d)
                scores.append(1 - ss_res / max(ss_tot, 1e-15))
            return sum(scores) / len(scores)

    # ── 持久化 ───────────────────────────────────────────────

    def save(self, filepath):
        """保存模型到 JSON 文件"""
        save_model(self.model, filepath, task=self.task)

    @staticmethod
    def load(filepath, loss_fn=None, optimizer='adam', lr=0.01,
             task='auto', default_output_activation=None):
        """从 JSON 加载，返回 Model 实例

        default_output_activation:
            仅对旧格式（无逐层 activation）文件生效，用来还原旧的 softmax 分类器；
            不传时其输出层会退回 linear，从而被当成回归模型。
        task:
            'auto'（默认）时优先采用文件中记录的 task，其次按输出层激活推断。
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
        return Model(mlp, loss_fn=loss_fn, optimizer=optimizer, lr=lr, task=task)

    # ── 信息 ─────────────────────────────────────────────────

    def summary(self):
        """打印模型结构摘要"""
        print(f"{'='*50}")
        print(f"  Model ({self.task})")
        print(f"  Loss: {self._loss_name}  |  Optimizer: {type(self.optimizer).__name__}(lr={self.lr})")
        print(f"  {'='*50}")
        total = 0
        for i, layer in enumerate(self.model.layers):
            if layer.act_fn == layer._leaky_relu:
                act = 'LeakyReLU'
            elif layer.act_fn == layer._relu:
                act = 'ReLU'
            elif layer.act_fn == layer._softmax:
                act = 'Softmax'
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


# ── 工厂函数 ─────────────────────────────────────────────────

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
        soft_y = teacher.predict_classes(x)          # list[int]
        student.fit(x, soft_y, epochs=epochs, batch_size=batch_size,
                    val_split=val_split, patience=patience, verbose=False)
        agree = sum(1 for xi, yi in zip(x, soft_y)
                    if student.predict_classes([xi])[0] == yi)
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


def quick_train(x, y, task='regression', hidden_layers=(32,), epochs=200,
                batch_size=32, lr=0.01, val_split=0.1, patience=30,
                optimizer='adam', verbose=True):
    """
    一行代码训练并返回模型。

    task='classification' 时 y 可以是整数类别索引（允许非连续，如 [0, 2]），
    也可以是 one-hot / 概率向量——类数按向量长度或 max+1 推断，不会因为
    某个类在本批数据里没出现而少建输出单元。小数标签与负索引会直接报错。

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
        model = make_classifier(input_dim, num_classes,
                                hidden_layers=hidden_layers, lr=lr,
                                optimizer=optimizer)
    else:
        output_dim = len(y[0]) if isinstance(y[0], (list, tuple)) else 1
        model = make_regressor(input_dim, output_dim,
                               hidden_layers=hidden_layers, lr=lr,
                               optimizer=optimizer)

    model.fit(x, y, epochs=epochs, batch_size=batch_size,
              val_split=val_split, patience=patience, verbose=verbose)
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

        # --- 应用梯度更新权重 (SGD) ---
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
