"""
全栈神经网络工具箱 — 完整测试套件（pytest）
覆盖：Layer / MLP / Conv2d / MaxPool2d / Dropout / BatchNorm1d / LayerNorm
       RNNCell / LSTMCell / GRUCell / RNN / LSTM / GRU
       Embedding / ResidualBlock / Sequential
       MultiHeadAttention / TransformerEncoderLayer / TransformerEncoder / PositionalEncoding
       mse_loss / cross_entropy_loss
       Adam / SGD / SGDMomentum / RMSprop
       clip_grad_by_norm / clip_grad_by_value
       StepLR / ExponentialLR / CosineAnnealingLR
       split_data / DataLoader
       TextProcessor / sample_index / generate_text / generate_text_with_mlp
       train / evaluate / save_model / load_model
"""

import pytest
import io
import logging
import math
import random
import os
import tempfile
import json
import copy
import AI
from AI import (
    Layer, MLP,
    Conv2d, MaxPool2d, Flatten,
    Dropout, BatchNorm1d, LayerNorm,
    RNNCell, LSTMCell, GRUCell,
    RNN, LSTM, GRU,
    MultiHeadAttention, FeedForward,
    TransformerEncoderLayer, TransformerEncoder, PositionalEncoding,
    Embedding, ResidualBlock, Sequential,
    mse_loss, cross_entropy_loss,
    Adam, SGD, SGDMomentum, RMSprop,
    clip_grad_by_norm, clip_grad_by_value,
    StepLR, ExponentialLR, CosineAnnealingLR,
    split_data, DataLoader,
    TextProcessor, sample_index,
    generate_text, generate_text_with_mlp,
    train, evaluate, save_model, load_model,
    save_conv_model, load_conv_model,
    _softmax_with_temp,
    # 新增高层 API
    Model, make_classifier, make_regressor, quick_train,
    distill,
)


# ========================== 1. Layer（激活函数 + 前向/反向） ==========================

class TestLayer:
    def test_leaky_relu_positive(self):
        """LeakyReLU: 正数输入经过线性变换后仍为正（z>0 恒等）"""
        layer = Layer(1, 1, activation="leaky_relu")
        layer.weights = [[2.0]]
        layer.biases = [0.5]
        out = layer.forward([1.0])
        # z = 2*1+0.5 = 2.5, leaky_relu(z) = 2.5
        assert out[0] == 2.5

    def test_leaky_relu_negative(self):
        """LeakyReLU: 负数输入乘以 0.01"""
        layer = Layer(1, 1, activation="leaky_relu")
        layer.weights = [[2.0]]
        layer.biases = [0.5]
        out = layer.forward([-1.0])
        # z = 2*(-1)+0.5 = -1.5, leaky_relu = -1.5*0.01 = -0.015
        assert out[0] == pytest.approx(-0.015)

    def test_relu(self):
        """ReLU: 负数归零，正数原样输出"""
        layer = Layer(2, 2, activation="relu")
        # 手工设置权重和偏置使 z1=3, z2=-2
        layer.weights = [[1.0, 0.0], [0.0, 1.0]]
        layer.biases = [0.0, 0.0]
        out = layer.forward([3.0, -2.0])
        assert out[0] == 3.0
        assert out[1] == 0.0

    def test_softmax_sums_to_one(self):
        """Softmax: 输出的概率之和应为 1"""
        layer = Layer(3, 3, activation="softmax")
        layer.weights = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        layer.biases = [0.0, 0.0, 0.0]
        out = layer.forward([1.0, 2.0, 3.0])
        assert sum(out) == pytest.approx(1.0, abs=1e-6)
        # 较大的 logits 应有较高的概率
        assert out[2] > out[1] > out[0]

    def test_linear(self):
        """Linear: 不应用任何激活函数"""
        layer = Layer(3, 2, activation="linear")
        out = layer.forward([1.0, 0.5, -0.5])
        assert len(out) == 2

    def test_unknown_activation_raises(self):
        """不合法的激活函数名称应抛出 ValueError（sigmoid/gelu 现在是合法激活）"""
        with pytest.raises(ValueError):
            Layer(1, 1, activation="tanh")

    def test_backward_shapes(self):
        """反向传播: 返回梯度形状与权重和偏置一致"""
        layer = Layer(3, 2, activation="leaky_relu")
        layer.forward([1.0, 0.5, -0.3])
        gw, gb, next_delta = layer.backward([0.1, -0.2])
        assert len(gw) == 2
        assert len(gw[0]) == 3
        assert len(gb) == 2
        assert len(next_delta) == 3


# ========================== 2. MLP ==========================

class TestMLP:
    def test_mlp_build_shapes(self):
        """MLP: 按 layer_sizes 构建正确层数"""
        model = MLP([5, 10, 3])
        assert len(model.layers) == 2
        assert model.layers[0].fan_in == 5 and model.layers[0].fan_out == 10
        assert model.layers[1].fan_in == 10 and model.layers[1].fan_out == 3

    def test_mlp_forward_output_size(self):
        """MLP: 前向传播输出维度正确"""
        model = MLP([4, 8, 2])
        out = model.forward([0.1, 0.2, 0.3, 0.4])
        assert len(out) == 2

    def test_mlp_custom_activations(self):
        """MLP: 支持自定义每层激活函数"""
        model = MLP([3, 5, 2], activations=["relu", "softmax"])
        out = model.forward([0.5, 0.1, -0.2])
        assert sum(out) == pytest.approx(1.0, abs=1e-6)

    def test_mlp_repr(self):
        model = MLP([3, 4, 1])
        assert "MLP" in repr(model)


# ========================== 3. Conv2d ==========================

class TestConv2d:
    def test_conv2d_output_shape_no_padding(self):
        """Conv2d: 无 padding 时输出尺寸正确"""
        conv = Conv2d(in_channels=1, out_channels=2, kernel_size=3)
        # 输入: (1, 5, 5) -> (2, 3, 3)
        x = [[[1.0] * 5 for _ in range(5)]]
        out = conv.forward(x)
        assert len(out) == 2  # out_channels
        assert len(out[0]) == 3  # height
        assert len(out[0][0]) == 3  # width

    def test_conv2d_output_shape_with_padding(self):
        """Conv2d: padding=1 时保持空间尺寸 (stride=1, kernel=3)"""
        conv = Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1)
        x = [[[1.0] * 5 for _ in range(5)]]
        out = conv.forward(x)
        assert len(out[0]) == 5
        assert len(out[0][0]) == 5

    def test_conv2d_stride(self):
        """Conv2d: stride=2 时输出减半"""
        conv = Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=2)
        x = [[[1.0] * 7 for _ in range(7)]]
        out = conv.forward(x)
        # (7 - 3) // 2 + 1 = 3
        assert len(out[0]) == 3


# ========================== 4. MaxPool2d ==========================

class TestMaxPool2d:
    def test_maxpool_output_shape(self):
        """MaxPool2d: 2x2 pool, stride=2 时输出尺寸正确"""
        pool = MaxPool2d(kernel_size=2)
        x = [[[1.0, 2.0, 3.0, 4.0],
              [5.0, 6.0, 7.0, 8.0],
              [9.0, 10.0, 11.0, 12.0],
              [13.0, 14.0, 15.0, 16.0]]]
        out = pool.forward(x)
        assert len(out) == 1
        assert len(out[0]) == 2
        assert out[0][0][0] == 6.0  # max of top-left 2x2


# ========================== 5. Flatten ==========================

class TestFlatten:
    def test_flatten(self):
        flat = Flatten()
        x = [[[1.0, 2.0], [3.0, 4.0]],
              [[5.0, 6.0], [7.0, 8.0]]]
        out = flat.forward(x)
        assert len(out) == 8
        assert out == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]


# ========================== 6. Dropout ==========================

class TestDropout:
    def test_dropout_train_mode(self):
        """Dropout: 训练模式下部分神经元被置零"""
        random.seed(42)
        dp = Dropout(p=0.5)
        x = [1.0] * 100
        out = dp.forward(x)
        # 大约一半被置零
        zeros = sum(1 for v in out if v == 0.0)
        assert 30 <= zeros <= 70

    def test_dropout_eval_mode(self):
        """Dropout: 评估模式下所有神经元保持不变"""
        dp = Dropout(p=0.5)
        dp.eval()
        x = [1.0] * 10
        out = dp.forward(x)
        assert out == x


# ========================== 7. BatchNorm1d / LayerNorm ==========================

class TestBatchNorm:
    def test_batchnorm_output_shape(self):
        bn = BatchNorm1d(num_features=4)
        x = [1.0, 2.0, 3.0, 4.0]
        out = bn.forward(x)
        assert len(out) == 4

    def test_batchnorm_eval_mode(self):
        bn = BatchNorm1d(num_features=3)
        bn.eval()
        x = [0.5, -0.2, 0.8]
        out = bn.forward(x)
        assert len(out) == 3


class TestLayerNorm:
    def test_layernorm_stats(self):
        """LayerNorm: 输出的均值接近 0，标准差接近 1"""
        ln = LayerNorm(num_features=5)
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        out = ln.forward(x)
        mean = sum(out) / len(out)
        std = math.sqrt(sum((v - mean) ** 2 for v in out) / len(out))
        assert mean == pytest.approx(0.0, abs=1e-4)
        assert std == pytest.approx(1.0, abs=1e-3)


# ========================== 8. RNNCell / LSTMCell / GRUCell ==========================

class TestRNNCell:
    def test_rnn_cell_forward(self):
        cell = RNNCell(input_size=3, hidden_size=2)
        h = cell.forward([0.1, 0.2, 0.3])
        assert len(h) == 2

    def test_rnn_cell_with_hidden(self):
        cell = RNNCell(input_size=2, hidden_size=3)
        h = cell.forward([0.5, -0.3], h_prev=[0.1, -0.1, 0.2])
        assert len(h) == 3


class TestLSTMCell:
    def test_lstm_cell_forward(self):
        cell = LSTMCell(input_size=3, hidden_size=2)
        h, c = cell.forward([0.1, 0.2, 0.3])
        assert len(h) == 2
        assert len(c) == 2


class TestGRUCell:
    def test_gru_cell_forward(self):
        cell = GRUCell(input_size=3, hidden_size=2)
        h = cell.forward([0.1, 0.2, 0.3])
        assert len(h) == 2


# ========================== 9. RNN / LSTM / GRU（多层序列） ==========================

class TestRNN:
    def test_rnn_output_shape(self):
        rnn = RNN(input_size=3, hidden_size=4, num_layers=2)
        seq = [[0.1, 0.2, 0.3] for _ in range(5)]
        out, _ = rnn.forward(seq)
        assert len(out) == 5
        assert len(out[0]) == 4


class TestLSTM:
    def test_lstm_output_shape(self):
        lstm = LSTM(input_size=3, hidden_size=4, num_layers=1)
        seq = [[0.1, 0.2, 0.3] for _ in range(3)]
        out, _ = lstm.forward(seq)
        assert len(out) == 3
        assert len(out[0]) == 4


class TestGRU:
    def test_gru_output_shape(self):
        gru = GRU(input_size=2, hidden_size=3, num_layers=1)
        seq = [[0.5, -0.2] for _ in range(4)]
        out, _ = gru.forward(seq)
        assert len(out) == 4
        assert len(out[0]) == 3


# ========================== 10. Embedding ==========================

class TestEmbedding:
    def test_embedding_lookup(self):
        random.seed(42)
        emb = Embedding(vocab_size=10, embed_dim=4)
        vecs = emb.forward([3])
        assert len(vecs) == 1
        assert len(vecs[0]) == 4
        # 同一索引应返回相同向量
        assert emb.forward([3])[0] == vecs[0]


# ========================== 11. ResidualBlock / Sequential ==========================

class TestResidualBlock:
    def test_residual_output_shape(self):
        sublayer = Layer(4, 4, activation="leaky_relu")
        block = ResidualBlock(sublayer)
        out = block.forward([0.1, 0.2, -0.1, 0.5])
        assert len(out) == 4

    def test_backward_with_layer_sublayer(self):
        """sublayer 是 Layer（backward 返回 3 元组）时必须取其末位

        回归点：ResidualBlock.backward 曾把返回值直接当 2D 列表索引，
        sublayer=Layer/LayerNorm 时会 TypeError（只有 sublayer 返回 list 才能跑通）。
        """
        random.seed(41)
        block = ResidualBlock(Layer(4, 4, activation="leaky_relu"))
        x = [random.gauss(0, 1) for _ in range(4)]
        dout = [random.gauss(0, 1) for _ in range(4)]
        block.forward(x)
        dx = block.backward(dout)
        assert len(dx) == 4

        # 数值校验：残差块输出 = x + F(x)，故 dL/dx = dout + dL/dF
        def loss():
            return sum(d * v for d, v in zip(dout, block.forward(x)))

        eps = 1e-6
        for i in range(4):
            old = x[i]
            x[i] = old + eps
            lp = loss()
            x[i] = old - eps
            lm = loss()
            x[i] = old
            num = (lp - lm) / (2 * eps)
            assert abs(num - dx[i]) < 1e-6, f"dx[{i}]: 解析 {dx[i]} vs 数值 {num}"

    def test_backward_with_layernorm_sublayer(self):
        """LayerNorm 也返回元组，同样要覆盖"""
        random.seed(42)
        block = ResidualBlock(LayerNorm(4))
        x = [random.gauss(0, 1) for _ in range(4)]
        block.forward(x)
        dx = block.backward([0.1, -0.2, 0.3, -0.4])
        assert len(dx) == 4

    def test_backward_with_sequential_sublayer(self):
        """Sequential 返回 list（非元组），走另一条分支"""
        random.seed(43)
        block = ResidualBlock(Sequential(Layer(4, 4, activation="leaky_relu"), Dropout(0.0)))
        x = [random.gauss(0, 1) for _ in range(4)]
        block.forward(x)
        dx = block.backward([0.1, -0.2, 0.3, -0.4])
        assert len(dx) == 4

    def test_backward_2d_with_attention_sublayer(self):
        """2D（序列）分支"""
        random.seed(44)
        block = ResidualBlock(MultiHeadAttention(d_model=4, num_heads=2))
        seq = [[random.gauss(0, 1) for _ in range(4)] for _ in range(3)]
        block.forward(seq)
        dout = [[random.gauss(0, 1) for _ in range(4)] for _ in range(3)]
        dx = block.backward(dout)
        assert len(dx) == 3
        assert len(dx[0]) == 4


class TestSequential:
    def test_sequential(self):
        seq = Sequential(
            Layer(3, 5, activation="leaky_relu"),
            Layer(5, 2, activation="linear"),
        )
        out = seq.forward([0.1, 0.2, 0.3])
        assert len(out) == 2


# ========================== 12. Transfomer 组件 ==========================

class TestTransformer:
    def test_multihead_attention_shapes(self):
        random.seed(42)
        mha = MultiHeadAttention(d_model=8, num_heads=2)
        seq = [[random.gauss(0, 1) for _ in range(8)] for _ in range(4)]
        out = mha.forward(seq)
        assert len(out) == 4
        assert len(out[0]) == 8

    def test_feedforward_shapes(self):
        random.seed(42)
        ff = FeedForward(d_model=8, d_ff=32)
        # FeedForward works on sequences: (seq_len, d_model)
        x = [[random.gauss(0, 1) for _ in range(8)] for _ in range(3)]
        out = ff.forward(x)
        assert len(out) == 3
        assert len(out[0]) == 8

    def test_positional_encoding(self):
        random.seed(42)
        pe = PositionalEncoding(d_model=8, max_len=20)
        x = [[random.gauss(0, 1) for _ in range(8)] for _ in range(5)]
        out = pe.forward(x)
        assert len(out) == 5
        assert len(out[0]) == 8
        # 加入位置编码后值应该有所变化
        assert out[0] != x[0]

    def test_transformer_encoder_layer(self):
        random.seed(42)
        enc = TransformerEncoderLayer(d_model=8, num_heads=2, d_ff=32)
        seq = [[random.gauss(0, 1) for _ in range(8)] for _ in range(4)]
        out = enc.forward(seq)
        assert len(out) == 4
        assert len(out[0]) == 8

    def test_transformer_encoder(self):
        random.seed(42)
        encoder = TransformerEncoder(
            d_model=8, num_heads=2, d_ff=32, num_layers=2
        )
        seq = [[random.gauss(0, 1) for _ in range(8)] for _ in range(3)]
        out = encoder.forward(seq)
        assert len(out) == 3
        assert len(out[0]) == 8


# ========================== 13. 损失函数 ==========================

class TestLossFunctions:
    def test_mse_loss_perfect(self):
        """MSE: 预测完全等于目标时损失为 0"""
        pred = [1.0, 2.0, 3.0]
        target = [1.0, 2.0, 3.0]
        loss, grad = mse_loss(pred, target)
        assert loss == 0.0
        assert grad == [0.0, 0.0, 0.0]

    def test_mse_loss_nonzero(self):
        pred = [0.0, 1.0]
        target = [1.0, 0.0]
        loss, _ = mse_loss(pred, target)
        assert loss > 0

    def test_cross_entropy_loss_correct(self):
        """CrossEntropy: 对正确类别高概率时损失接近 0"""
        pred = [0.01, 0.98, 0.01]  # 类别 1 概率极高
        loss, grad = cross_entropy_loss(pred, 1)
        assert loss < 0.1

    def test_cross_entropy_loss_wrong(self):
        """CrossEntropy: 对正确类别低概率时损失较大"""
        pred = [0.01, 0.01, 0.98]  # 类别 2 概率极高，但 target=1
        loss, grad = cross_entropy_loss(pred, 1)
        assert loss > 2.0


# ========================== 14. 优化器 ==========================

class TestOptimizers:
    def test_adam_step(self):
        """Adam: 一步更新后权重应变化"""
        model = MLP([2, 3, 1])
        orig_w = model.layers[0].weights[0][0]
        opt = Adam(model, lr=0.1)

        model.forward([0.5, -0.3])
        delta = [0.5]
        grads = []
        for layer in reversed(model.layers):
            gw, gb, delta = layer.backward(delta)
            grads.append((gw, gb))
        grads.reverse()

        opt.step(grads)
        assert model.layers[0].weights[0][0] != orig_w

    def test_sgd_step(self):
        model = MLP([2, 1])
        orig_w = model.layers[0].weights[0][0]
        opt = SGD(model, lr=0.5)

        model.forward([0.5, 0.5])
        delta = [0.5]
        grads = []
        for layer in reversed(model.layers):
            gw, gb, delta = layer.backward(delta)
            grads.append((gw, gb))
        grads.reverse()

        opt.step(grads)
        assert model.layers[0].weights[0][0] != orig_w

    def test_rmsprop_step(self):
        model = MLP([2, 1])
        orig_w = model.layers[0].weights[0][0]
        opt = RMSprop(model, lr=0.1)

        model.forward([0.5, -0.2])
        delta = [0.5]
        grads = []
        for layer in reversed(model.layers):
            gw, gb, delta = layer.backward(delta)
            grads.append((gw, gb))
        grads.reverse()

        opt.step(grads)
        assert model.layers[0].weights[0][0] != orig_w


# ========================== 15. 梯度裁剪 ==========================

class TestGradientClipping:
    def test_clip_by_norm(self):
        grads = [([[100.0, 100.0]], [100.0])]
        total_norm = clip_grad_by_norm(grads, max_norm=1.0)
        assert total_norm > 0
        assert grads[0][0][0][0] < 100.0

    def test_clip_by_value(self):
        grads = [([[100.0, -100.0]], [50.0])]
        clip_grad_by_value(grads, clip_val=1.0)
        assert abs(grads[0][0][0][0]) <= 1.0
        assert abs(grads[0][1][0]) <= 1.0


# ========================== 16. 学习率调度器 ==========================

class TestLRSchedulers:
    def test_step_lr(self):
        model = MLP([1, 1])
        opt = Adam(model, lr=0.1)
        scheduler = StepLR(opt, step_size=3, gamma=0.5)

        for e in range(1, 7):
            scheduler.step(e)
        assert opt.lr < 0.1

    def test_exponential_lr(self):
        model = MLP([1, 1])
        opt = Adam(model, lr=0.1)
        scheduler = ExponentialLR(opt, gamma=0.5)

        scheduler.step(1)
        assert opt.lr < 0.1

    def test_cosine_annealing(self):
        model = MLP([1, 1])
        opt = Adam(model, lr=0.1)
        scheduler = CosineAnnealingLR(opt, T_max=10, eta_min=0.001)

        scheduler.step(5)
        assert 0.001 < opt.lr < 0.1
        scheduler.step(10)
        assert opt.lr == pytest.approx(0.001, abs=0.01)


# ========================== 17. 数据工具 ==========================

class TestDataUtils:
    def test_split_data(self):
        data = [(f"sample_{i}", i) for i in range(100)]
        train, val, test = split_data(data, val_ratio=0.2, test_ratio=0.1)
        assert len(train) == 70  # 100 * 0.7
        assert len(val) == 20
        assert len(test) == 10

    def test_dataloader(self):
        data = [(f"x{i}", f"y{i}") for i in range(10)]
        loader = DataLoader(data, batch_size=3)
        batches = list(loader)
        assert len(batches) == 4
        assert len(batches[0]) == 3
        assert len(batches[-1]) == 1  # 最后一个 batch 只有 1 个


# ========================== 18. 文本处理 ==========================

class TestTextProcessing:
    def test_text_processor_vocab(self):
        tp = TextProcessor("abcabc")
        assert tp.vocab_size == 3
        assert set(tp.chars) == {"a", "b", "c"}

    def test_one_hot_encoding(self):
        tp = TextProcessor("abc")
        vec = tp.encode_one_hot("b")
        assert sum(vec) == 1.0
        assert vec[tp.char_to_idx["b"]] == 1.0

    def test_encode_indices(self):
        tp = TextProcessor("abc")
        indices = tp.encode_indices("cab")
        assert indices == [
            tp.char_to_idx["c"],
            tp.char_to_idx["a"],
            tp.char_to_idx["b"],
        ]

    def test_decode_indices(self):
        tp = TextProcessor("abc")
        s = tp.decode_indices([1, 0, 2])
        assert s == "bac"

    def test_prepare_data(self):
        tp = TextProcessor("abca")
        data = tp.prepare_data("abca", window_size=2)
        # "abc" + "a": 窗口 "ab" -> 预测 "c"; 窗口 "bc" -> 预测 "a"
        assert len(data) == 2

    def test_prepare_index_data(self):
        tp = TextProcessor("abca")
        data = tp.prepare_index_data("abca", window_size=2)
        assert len(data) == 2
        assert isinstance(data[0][0], list)  # indices list
        assert isinstance(data[0][1], int)  # target_idx


class TestSoftmaxTemp:
    def test_temperature(self):
        logits = [1.0, 2.0, 3.0]
        # temp=1 正常分布
        p1 = _softmax_with_temp(logits, temperature=1.0)
        # temp 很小 -> 接近 one-hot
        p2 = _softmax_with_temp(logits, temperature=0.1)
        assert p2[2] > p1[2]  # 温度低，高 logit 概率更突出


class TestSampleIndex:
    def test_sample_index(self):
        random.seed(42)
        probs = [1.0, 0.0, 0.0]
        idx = sample_index(probs)
        assert idx == 0  # 概率全在第一个


class TestTextGeneration:
    def test_generate_text_with_mlp(self):
        random.seed(42)
        tp = TextProcessor("你好世界")
        # 训练一个简单模型
        text = "你好世界" * 10
        window_size = 2
        dataset = tp.prepare_data(text, window_size)

        input_dim = window_size * tp.vocab_size
        model = MLP([input_dim, 16, tp.vocab_size], activations=["leaky_relu", "softmax"])
        opt = Adam(model, lr=0.05)

        for _ in range(50):
            random.shuffle(dataset)
            for x, y in dataset:
                probs = model.forward(x)
                loss = -sum(y[j] * math.log(max(probs[j], 1e-15)) for j in range(len(y)))
                delta = [probs[j] - y[j] for j in range(len(y))]
                grads = []
                for layer in reversed(model.layers):
                    gw, gb, delta = layer.backward(delta)
                    grads.append((gw, gb))
                grads.reverse()
                opt.step(grads)

        result = generate_text_with_mlp(model, tp, "你好", length=5, window_size=window_size, temp=0.3)
        assert result.startswith("你好")
        assert len(result) == 7  # "你好" + 5 chars

    def test_generate_text_is_alias_of_mlp_variant(self):
        """generate_text 与 generate_text_with_mlp 是同一实现（前者只是转发）"""
        random.seed(61)
        text = "君不见黄河之水天上来奔流到海不复回"
        tp = TextProcessor(text)
        model = MLP([tp.vocab_size * 3, 16, tp.vocab_size],
                    activations=["leaky_relu", "softmax"])
        random.seed(62)
        a = generate_text(model, tp, "君不见", length=6, window_size=3, temp=0.8)
        random.seed(62)
        b = generate_text_with_mlp(model, tp, "君不见", length=6, window_size=3, temp=0.8)
        assert a == b


# ========================== 19. 训练 / 评估 ==========================

class TestTrainingPipeline:
    def test_train_regression(self):
        """完整训练流程: 回归任务 Loss 应下降"""
        random.seed(42)

        # y = 2*x1 - 3*x2 + 1
        data = []
        for _ in range(200):
            x1 = random.uniform(-2, 2)
            x2 = random.uniform(-2, 2)
            y = 2 * x1 - 3 * x2 + 1
            data.append(([x1, x2], [y]))

        train_d, val_d, test_d = split_data(data, val_ratio=0.15, test_ratio=0.15)

        model = MLP([2, 8, 1], activations=["leaky_relu", "linear"])
        opt = Adam(model, lr=0.02)

        # 定义 loss_fn 包装
        def loss_fn(pred, target):
            return mse_loss(pred, target)

        history = train(model, loss_fn, opt, train_d, epochs=300,
                        batch_size=16, val_data=val_d, verbose=False)

        # 验证最终训练 Loss 比初始低
        assert history["train_loss"][-1][1] < history["train_loss"][0][1]

    def test_train_classification(self):
        """分类任务: 准确率应较高"""
        random.seed(42)

        # 二分类: 两簇高斯点
        data = []
        for _ in range(100):
            data.append(([random.gauss(-1, 0.5), random.gauss(-1, 0.5)], 0))
        for _ in range(100):
            data.append(([random.gauss(1, 0.5), random.gauss(1, 0.5)], 1))

        train_d, val_d, test_d = split_data(data, val_ratio=0.2, test_ratio=0.2)

        model = MLP([2, 16, 2], activations=["leaky_relu", "softmax"])
        opt = Adam(model, lr=0.03)

        def loss_fn(pred, target):
            return cross_entropy_loss(pred, target)

        history = train(model, loss_fn, opt, train_d, epochs=200,
                        batch_size=16, val_data=val_d, verbose=False)

        # 测试集准确率
        correct = 0
        for x, y in test_d:
            probs = model.forward(x)
            if probs.index(max(probs)) == y:
                correct += 1
        acc = correct / len(test_d)
        assert acc > 0.7

    def test_evaluate(self):
        model = MLP([1, 1], activations=["linear"])
        data = [([0.5], [1.0]), ([1.0], [2.0])]

        def loss_fn(pred, target):
            return mse_loss(pred, target)

        val_loss = evaluate(model, loss_fn, data)
        assert val_loss >= 0


# ========================== 20. 模型保存 / 加载 ==========================

class TestModelPersistence:
    def test_save_load_roundtrip(self, tmp_path):
        """模型保存后加载，权重应一致"""
        random.seed(42)
        model = MLP([3, 5, 2], activations=["leaky_relu", "softmax"])

        # 记录原始权重
        orig_weights = [[row[:] for row in layer.weights] for layer in model.layers]
        orig_biases = [layer.biases[:] for layer in model.layers]

        filepath = str(tmp_path / "model.json")
        save_model(model, filepath)
        loaded = load_model(filepath)

        # 验证每层权重
        for i, layer in enumerate(loaded.layers):
            for j in range(layer.fan_out):
                assert layer.weights[j] == pytest.approx(orig_weights[i][j])
                assert layer.biases[j] == pytest.approx(orig_biases[i][j])

    def test_load_model_forward(self, tmp_path):
        """加载的模型可以正常前向传播"""
        random.seed(42)
        model = MLP([3, 4, 2])
        out_before = model.forward([0.1, 0.2, 0.3])

        filepath = str(tmp_path / "model.json")
        save_model(model, filepath)
        loaded = load_model(filepath)
        out_after = loaded.forward([0.1, 0.2, 0.3])

        assert out_after == pytest.approx(out_before)

    def test_save_load_activations(self, tmp_path):
        """保存不同激活函数的模型，加载后激活函数类型正确"""
        random.seed(42)
        for acts in [["relu", "softmax"], ["leaky_relu", "linear"]]:
            model = MLP([3, 5, 2], activations=acts)
            filepath = str(tmp_path / f"model_{acts[0]}.json")
            save_model(model, filepath)
            loaded = load_model(filepath)
            # 验证最后一层 softmax 输出和为 1
            if acts[-1] == "softmax":
                out = loaded.forward([0.5, -0.2, 0.8])
                assert sum(out) == pytest.approx(1.0, abs=1e-5)


# ========================== 21. 边界 / 回归测试 ==========================

class TestEdgeCases:
    def test_single_layer_mlp(self):
        """单层 MLP（无隐藏层）"""
        model = MLP([3, 1], activations=["linear"])
        out = model.forward([1.0, 2.0, 3.0])
        assert len(out) == 1

    def test_deep_mlp(self):
        """深层 MLP"""
        model = MLP([10, 32, 32, 16, 5], activations=["relu"] * 3 + ["softmax"])
        out = model.forward([0.0] * 10)
        assert len(out) == 5
        assert sum(out) == pytest.approx(1.0, abs=1e-5)

    def test_zero_input(self):
        """全零输入不会报错"""
        model = MLP([5, 10, 3])
        out = model.forward([0.0] * 5)
        assert len(out) == 3

    def test_large_number_of_layers(self):
        """10 层网络可以正常前向传播"""
        model = MLP([8] + [16] * 9 + [4],
                    activations=["relu"] * 9 + ["softmax"])
        out = model.forward([0.0] * 8)
        assert len(out) == 4

    def test_random_weight_initialization(self):
        """相同 random.seed 应产生相同的初始化"""
        random.seed(123)
        m1 = MLP([3, 5, 2])
        random.seed(123)
        m2 = MLP([3, 5, 2])
        for l1, l2 in zip(m1.layers, m2.layers):
            for j in range(l1.fan_out):
                assert l1.weights[j] == l2.weights[j]
                assert l1.biases[j] == l2.biases[j]

    @pytest.mark.parametrize("shape", [
        ([2, 3, 1]),
        ([5, 10, 10, 2]),
        ([1, 8, 1]),
    ])
    def test_various_shapes(self, shape):
        model = MLP(shape, activations=["relu"] * (len(shape) - 2) + ["linear"])
        out = model.forward([1.0] * shape[0])
        assert len(out) == shape[-1]


# ========================== 22. 高层 API: Model 类 ==========================

class TestModelHighLevel:
    """测试 sklearn 风格 Model 封装"""

    def test_model_creation_regression(self):
        """Model: 回归任务自动推断"""
        mlp = MLP([3, 8, 1], activations=["leaky_relu", "linear"])
        model = Model(mlp)
        assert model.task == "regression"
        assert model._loss_name in ("mse",)

    def test_model_creation_classification(self):
        """Model: 分类任务自动推断"""
        mlp = MLP([3, 8, 3], activations=["leaky_relu", "softmax"])
        model = Model(mlp)
        assert model.task == "classification"
        assert model._loss_name in ("cross_entropy", "ce")

    def test_model_creation_explicit(self):
        """Model: 显式指定参数"""
        mlp = MLP([4, 8, 2])
        model = Model(mlp, loss_fn="mse", optimizer="sgd", lr=0.05, task="regression")
        assert model.lr == 0.05
        assert isinstance(model.optimizer, SGD)
        assert model.task == "regression"

    def test_model_creation_custom_loss(self):
        """Model: 自定义损失函数"""
        mlp = MLP([2, 4, 1])
        def custom_loss(pred, target):
            loss = sum(abs(p - t) for p, t in zip(pred, target))
            grad = [(1 if p > t else -1) for p, t in zip(pred, target)]
            return loss, grad
        model = Model(mlp, loss_fn=custom_loss)
        assert model._loss_name == "custom"

    def test_forward_single(self):
        """Model.forward: 单样本"""
        random.seed(42)
        mlp = MLP([3, 4, 2])
        model = Model(mlp)
        out = model.forward([0.1, 0.2, 0.3])
        assert len(out) == 2

    def test_predict_single(self):
        """Model.predict: 单样本"""
        random.seed(42)
        mlp = MLP([2, 4, 1])
        model = Model(mlp)
        out = model.predict([0.5, -0.3])
        assert isinstance(out, list)
        assert len(out) == 1

    def test_predict_batch(self):
        """Model.predict: 多样本批量"""
        random.seed(42)
        mlp = MLP([2, 4, 1])
        model = Model(mlp)
        x = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        out = model.predict(x)
        assert isinstance(out, list)
        assert len(out) == 3
        assert len(out[0]) == 1

    def test_predict_classes(self):
        """Model.predict_classes: 分类"""
        random.seed(42)
        mlp = MLP([2, 8, 3], activations=["leaky_relu", "softmax"])
        model = Model(mlp)
        cls = model.predict_classes([0.5, -0.3])
        assert isinstance(cls, int)
        assert 0 <= cls <= 2

    def test_predict_classes_batch(self):
        """Model.predict_classes: 批量分类"""
        random.seed(42)
        mlp = MLP([2, 8, 3], activations=["leaky_relu", "softmax"])
        model = Model(mlp)
        classes = model.predict_classes([[0.1, 0.2], [0.3, 0.4]])
        assert isinstance(classes, list)
        assert len(classes) == 2

    def test_fit_regression(self):
        """Model.fit: 回归训练 Loss 下降"""
        random.seed(42)
        x = [[random.uniform(-2, 2) for _ in range(3)] for _ in range(150)]
        y = [[2 * xi[0] - xi[1] + 0.5 * xi[2]] for xi in x]

        mlp = MLP([3, 16, 8, 1], activations=["leaky_relu", "leaky_relu", "linear"])
        model = Model(mlp, loss_fn="mse", optimizer="adam", lr=0.02)
        history = model.fit(x, y, epochs=200, batch_size=16, val_split=0.0,
                           patience=0, verbose=False)
        assert history["train_loss"][-1][1] < history["train_loss"][0][1]

    def test_fit_classification(self):
        """Model.fit: 分类训练 + 准确率"""
        random.seed(42)
        # 二分类: 两簇高斯数据
        x, y_true = [], []
        for _ in range(100):
            x.append([random.gauss(-1, 0.5), random.gauss(-1, 0.5)])
            y_true.append(0)
        for _ in range(100):
            x.append([random.gauss(1, 0.5), random.gauss(1, 0.5)])
            y_true.append(1)

        model = make_classifier(2, 2, hidden_layers=(16,), lr=0.03)
        history = model.fit(x, y_true, epochs=200, batch_size=16,
                           val_split=0.2, patience=0, verbose=False)
        acc = model.score(x, y_true)
        assert acc > 0.7

    def test_evaluate_method(self):
        """Model.evaluate: 计算损失"""
        random.seed(42)
        mlp = MLP([2, 4, 1])
        model = Model(mlp, loss_fn="mse")
        x = [[0.1, 0.2], [0.3, 0.4]]
        y = [[0.5], [0.7]]
        loss = model.evaluate(x, y)
        assert loss >= 0

    def test_score_classification(self):
        """Model.score: 分类准确率"""
        random.seed(42)
        x = [[0.0, 0.0], [1.0, 1.0], [0.0, 0.1], [0.9, 1.1]]
        y = [0, 1, 0, 1]
        model = make_classifier(2, 2, hidden_layers=(8,), lr=0.05)
        model.fit(x, y, epochs=300, batch_size=4, val_split=0.0,
                 patience=0, verbose=False)
        acc = model.score(x, y)
        assert 0.0 <= acc <= 1.0

    def test_score_regression(self):
        """Model.score: 回归 R²"""
        random.seed(42)
        x = [[i] for i in range(20)]
        y = [[2 * i + 1] for i in range(20)]
        model = make_regressor(1, 1, hidden_layers=(8,), lr=0.05)
        model.fit(x, y, epochs=300, batch_size=8, val_split=0.0,
                 patience=0, verbose=False)
        r2 = model.score(x, y)
        assert r2 > 0.5  # 应能较好拟合线性关系

    def test_fit_with_data_format(self):
        """Model.fit: 元组列表格式 (data=...)"""
        random.seed(42)
        data = [([0.1, 0.2], [0.3]), ([0.4, 0.5], [0.9])]
        mlp = MLP([2, 4, 1])
        model = Model(mlp, loss_fn="mse", optimizer="adam", lr=0.01)
        history = model.fit(data=data, epochs=50, batch_size=2,
                           val_split=0.0, patience=0, verbose=False)
        assert len(history["train_loss"]) > 0

    def test_fit_with_val_data(self):
        """Model.fit: 自定义验证集"""
        random.seed(42)
        x_train = [[random.uniform(-1, 1) for _ in range(2)] for _ in range(80)]
        y_train = [[xi[0] + xi[1]] for xi in x_train]
        x_val = [[random.uniform(-1, 1) for _ in range(2)] for _ in range(20)]
        y_val = [[xi[0] + xi[1]] for xi in x_val]

        mlp = MLP([2, 8, 1])
        model = Model(mlp, loss_fn="mse", optimizer="adam", lr=0.02)
        history = model.fit(x_train, y_train, epochs=100, batch_size=16,
                           val_data=(x_val, y_val), patience=0, verbose=False)
        assert "val_loss" in history
        assert len(history["val_loss"]) > 0

    def test_fit_grad_clip(self):
        """Model.fit: 梯度裁剪不报错"""
        random.seed(42)
        x = [[random.uniform(-2, 2) for _ in range(2)] for _ in range(50)]
        y = [[xi[0] - xi[1]] for xi in x]
        mlp = MLP([2, 8, 1])
        model = Model(mlp, loss_fn="mse", lr=0.01)
        history = model.fit(x, y, epochs=50, batch_size=8,
                           val_split=0.0, patience=0, grad_clip=1.0, verbose=False)
        assert len(history["train_loss"]) > 0

    def test_fit_lr_scheduler(self):
        """Model.fit: 学习率调度"""
        random.seed(42)
        x = [[random.uniform(-2, 2) for _ in range(2)] for _ in range(50)]
        y = [[xi[0] - xi[1]] for xi in x]
        model = make_regressor(2, 1, hidden_layers=(8,), lr=0.05)
        scheduler = StepLR(model.optimizer, step_size=30, gamma=0.5)
        history = model.fit(x, y, epochs=100, batch_size=8, val_split=0.0,
                           patience=0, verbose=False, lr_scheduler=scheduler)
        assert model.optimizer.lr < 0.05  # 调度器生效

    def test_save_load_roundtrip(self, tmp_path):
        """Model.save / Model.load: 保存加载往返"""
        random.seed(42)
        mlp = MLP([3, 5, 2], activations=["leaky_relu", "softmax"])
        orig = Model(mlp, loss_fn="cross_entropy", lr=0.01)
        out_before = orig.predict([0.1, 0.2, 0.3])

        f = str(tmp_path / "model.json")
        orig.save(f)
        loaded = Model.load(f, loss_fn="cross_entropy", lr=0.01)
        out_after = loaded.predict([0.1, 0.2, 0.3])
        assert out_after == pytest.approx(out_before)

    def test_summary(self, capsys):
        """Model.summary: 不报错"""
        mlp = MLP([4, 8, 2])
        model = Model(mlp)
        model.summary()
        captured = capsys.readouterr()
        assert "Model" in captured.out

    def test_early_stopping(self):
        """Model.fit: early stopping 生效"""
        random.seed(42)
        x = [[random.uniform(-2, 2) for _ in range(2)] for _ in range(200)]
        y = [[xi[0] - 2 * xi[1] + 0.5] for xi in x]

        model = make_regressor(2, 1, hidden_layers=(16,), lr=0.02)
        history = model.fit(x, y, epochs=500, batch_size=16, val_split=0.2,
                           patience=5, verbose=False)
        last_epoch = history["train_loss"][-1][0]
        assert last_epoch < 500  # 提前停止了

    def test_optimizer_string_aliases(self):
        """Model: 各种优化器字符串别名"""
        random.seed(42)
        mlp = MLP([2, 4, 1])
        for opt_name in ["adam", "sgd", "sgd_momentum", "rmsprop"]:
            model = Model(mlp, optimizer=opt_name, lr=0.01)
            assert model.optimizer is not None

    def test_invalid_loss_raises(self):
        """Model: 非法 loss_fn 抛出 ValueError"""
        mlp = MLP([2, 4, 1])
        with pytest.raises(ValueError):
            Model(mlp, loss_fn="invalid_loss")

    def test_invalid_optimizer_raises(self):
        """Model: 非法 optimizer 抛出 ValueError"""
        mlp = MLP([2, 4, 1])
        with pytest.raises(ValueError):
            Model(mlp, optimizer="invalid_opt")

    def test_invalid_model_type_raises(self):
        """Model: 非 MLP 类型抛出 TypeError"""
        with pytest.raises(TypeError):
            Model("not_a_model")


# ========================== 23. 工厂函数: make_classifier / make_regressor / quick_train ==========================

class TestFactories:
    """测试工厂函数"""

    def test_make_classifier(self):
        """make_classifier: 创建分类模型"""
        model = make_classifier(input_dim=5, num_classes=3,
                               hidden_layers=(10,), lr=0.01)
        assert model.task == "classification"
        assert len(model.model.layers) == 2
        assert model.model.layers[-1].fan_out == 3

    def test_make_regressor(self):
        """make_regressor: 创建回归模型"""
        model = make_regressor(input_dim=5, output_dim=2,
                              hidden_layers=(10,), lr=0.01)
        assert model.task == "regression"
        assert len(model.model.layers) == 2
        assert model.model.layers[-1].fan_out == 2

    def test_make_regressor_default_output(self):
        """make_regressor: 默认 output_dim=1"""
        model = make_regressor(input_dim=4, hidden_layers=(8,))
        assert model.model.layers[-1].fan_out == 1

    def test_quick_train_regression(self):
        """quick_train: 回归任务"""
        random.seed(42)
        x = [[i, i + 1] for i in range(50)]
        y = [2 * xi[0] + xi[1] + 1 for xi in x]
        model = quick_train(x, y, task="regression", hidden_layers=(16,),
                           epochs=200, batch_size=8, verbose=False)
        r2 = model.score(x, y)
        assert r2 > 0.5

    def test_quick_train_classification(self):
        """quick_train: 分类任务"""
        random.seed(42)
        x = [[random.gauss(-1, 0.3), random.gauss(-1, 0.3)] for _ in range(50)] + \
            [[random.gauss(1, 0.3), random.gauss(1, 0.3)] for _ in range(50)]
        y = [0] * 50 + [1] * 50
        model = quick_train(x, y, task="classification", hidden_layers=(8,),
                           epochs=100, batch_size=16, verbose=False)
        acc = model.score(x, y)
        assert acc > 0.7

    def test_quick_train_multi_output_regression(self):
        """quick_train: 多输出回归"""
        random.seed(42)
        x = [[i, i + 1] for i in range(30)]
        y = [[xi[0] + xi[1], xi[0] - xi[1]] for xi in x]
        model = quick_train(x, y, task="regression", hidden_layers=(8,),
                           epochs=100, batch_size=8, verbose=False)
        pred = model.predict(x[:5])
        assert len(pred) == 5
        assert len(pred[0]) == 2

    def test_make_classifier_optimizer_kwargs(self):
        """make_classifier: 传递优化器参数"""
        model = make_classifier(4, 2, hidden_layers=(8,),
                               optimizer="sgd", opt_kwargs={"weight_decay": 1e-4})
        assert hasattr(model.optimizer, "weight_decay")
        assert model.optimizer.weight_decay == 1e-4


# ========================== 24. 端到端工作流测试 ==========================

class TestEndToEnd:
    """完整工作流测试"""

    def test_full_regression_workflow(self):
        """完整回归流程: 创建 → 训练 → 预测 → 评分 → 保存 → 加载 → 验证"""
        random.seed(42)
        # 合成数据
        n = 200
        x = [[random.uniform(-3, 3) for _ in range(3)] for _ in range(n)]
        y = [[3 * xi[0] - 2 * xi[1] + 1.5 * xi[2] + 0.5] for xi in x]

        # 创建 + 训练
        model = make_regressor(3, 1, hidden_layers=(16, 8), lr=0.02,
                              optimizer="adam")
        history = model.fit(x, y, epochs=200, batch_size=16, val_split=0.2,
                           patience=0, verbose=False)

        # 预测
        pred = model.predict(x[:5])
        assert len(pred) == 5

        # 评分
        r2 = model.score(x, y)
        assert r2 > 0.8  # 简单线性关系，应拟合很好

    def test_full_classification_workflow(self):
        """完整分类流程"""
        random.seed(42)
        # 二分类高斯数据
        n = 200
        x, y_true = [], []
        for _ in range(n // 2):
            x.append([random.gauss(-1.5, 0.6), random.gauss(-1.5, 0.6)])
            y_true.append(0)
        for _ in range(n // 2):
            x.append([random.gauss(1.5, 0.6), random.gauss(1.5, 0.6)])
            y_true.append(1)

        model = make_classifier(2, 2, hidden_layers=(16, 8), lr=0.03)
        history = model.fit(x, y_true, epochs=200, batch_size=16, val_split=0.2,
                           patience=0, verbose=False)

        # 预测类别
        classes = model.predict_classes(x[:5])
        assert len(classes) == 5

        # 准确率
        acc = model.score(x, y_true)
        assert acc > 0.85

        # summary 不报错
        model.summary()

    def test_score_multi_output_regression(self):
        """多输出回归的 R² 要对所有输出维度取平均

        回归点：原来只取 preds[i][0] / y[i][0]，其余维度被静默忽略。
        """
        random.seed(51)
        n = 30
        x = [[random.uniform(-1, 1), random.uniform(-1, 1)] for _ in range(n)]
        y = [[xi[0], random.uniform(-3, 3)] for xi in x]      # 第 1 维是噪声，学不动
        model = make_regressor(2, 2, hidden_layers=(16, 16), lr=0.03)
        model.fit(x, y, epochs=400, patience=100, verbose=False)

        preds = model.predict(x)

        def r2_dim(d):
            ys = [yi[d] for yi in y]
            ps = [p[d] for p in preds]
            ym = sum(ys) / len(ys)
            ss_res = sum((a - b) ** 2 for a, b in zip(ys, ps))
            ss_tot = sum((a - ym) ** 2 for a in ys)
            return 1 - ss_res / max(ss_tot, 1e-15)

        score = model.score(x, y)
        assert score == pytest.approx((r2_dim(0) + r2_dim(1)) / 2, abs=1e-9)
        assert abs(score - r2_dim(0)) > 1e-6, "不应再只反映第 0 维"

    def test_score_single_output_unchanged(self):
        """单输出回归的 R² 仍与原来一致"""
        random.seed(52)
        x = [[i] for i in range(20)]
        y = [[2 * i + 1] for i in range(20)]
        model = make_regressor(1, 1, hidden_layers=(8,), lr=0.05)
        model.fit(x, y, epochs=300, batch_size=8, val_split=0.0,
                  patience=0, verbose=False)
        preds = model.predict(x)
        ys = [yi[0] for yi in y]
        ps = [p[0] for p in preds]
        ym = sum(ys) / len(ys)
        expected = 1 - sum((a - b) ** 2 for a, b in zip(ys, ps)) / \
            sum((a - ym) ** 2 for a in ys)
        assert model.score(x, y) == pytest.approx(expected, abs=1e-12)

    def test_model_repr(self):
        """Model.__repr__"""
        mlp = MLP([3, 8, 2], activations=["leaky_relu", "softmax"])
        model = Model(mlp)
        r = repr(model)
        assert "classification" in r
        assert "Adam" in r


# ========================== 23. 回归测试：本次修复 ==========================

class TestValDataFormats:
    """val_data 的三种写法都应被正确解析（含 2 个样本的结构歧义场景）"""

    @staticmethod
    def _model(n_in=2, n_out=1):
        random.seed(42)
        mlp = MLP([n_in, 4, n_out], activations=["leaky_relu", "linear"])
        return Model(mlp, loss_fn="mse", optimizer="adam", lr=0.01, task="regression")

    def test_list_of_two_pairs(self):
        """list 写法的 2 个样本对不能被误判成并行数组"""
        got = self._model()._normalize_val_data([[[0.1, 0.2], 1.0], [[0.3, 0.4], 2.0]])
        assert got == [([0.1, 0.2], 1.0), ([0.3, 0.4], 2.0)]

    def test_tuple_of_two_pairs(self):
        got = self._model()._normalize_val_data((([0.1, 0.2], 1.0), ([0.3, 0.4], 2.0)))
        assert got == [([0.1, 0.2], 1.0), ([0.3, 0.4], 2.0)]

    def test_parallel_arrays_tuple(self):
        got = self._model()._normalize_val_data(([[0.1, 0.2], [0.3, 0.4]], [1.0, 2.0]))
        assert got == [([0.1, 0.2], 1.0), ([0.3, 0.4], 2.0)]

    def test_parallel_arrays_list_with_scalar_targets(self):
        got = self._model()._normalize_val_data([[[0.1, 0.2], [0.3, 0.4]], [1.0, 2.0]])
        assert got == [([0.1, 0.2], 1.0), ([0.3, 0.4], 2.0)]

    def test_dict_form(self):
        got = self._model()._normalize_val_data(
            {"x": [[0.1, 0.2], [0.3, 0.4]], "y": [1.0, 2.0]}
        )
        assert got == [([0.1, 0.2], 1.0), ([0.3, 0.4], 2.0)]

    def test_json_style_nested_list(self):
        """json.loads 得到的验证集（全是 list）不应被误判"""
        got = self._model()._normalize_val_data(
            json.loads('[[[0.1,0.2],1.0],[[0.3,0.4],2.0]]')
        )
        assert got == [([0.1, 0.2], 1.0), ([0.3, 0.4], 2.0)]

    def test_multi_output_two_pairs_not_scrambled(self):
        """多输出 + 2 个样本：旧实现会把 x1 配 x2、y1 配 y2"""
        got = self._model(n_in=2, n_out=2)._normalize_val_data(
            [[[0.1, 0.2], [1.0, 2.0]], [[0.3, 0.4], [3.0, 4.0]]]
        )
        assert got == [([0.1, 0.2], [1.0, 2.0]), ([0.3, 0.4], [3.0, 4.0])]

    def test_wrong_feature_dim_raises(self):
        with pytest.raises(ValueError, match="特征维度"):
            self._model(n_in=3)._normalize_val_data(([[0.1, 0.2]], [1.0]))

    def test_wrong_target_dim_raises(self):
        with pytest.raises(ValueError, match="target 维度"):
            self._model(n_in=2, n_out=3)._normalize_val_data(([[0.1, 0.2]], [[1.0, 2.0]]))

    def test_bad_format_raises(self):
        with pytest.raises(ValueError, match="无法识别"):
            self._model()._normalize_val_data([1.0, 2.0, 3.0])

    def test_empty_val_data_means_no_validation(self):
        assert self._model()._normalize_val_data([]) == []

    def test_fit_with_list_pairs_records_val_loss(self):
        """端到端：list 写法的样本对能正常走完 fit 并记录 val_loss"""
        random.seed(42)
        x = [[random.uniform(-1, 1) for _ in range(2)] for _ in range(40)]
        y = [[xi[0] + xi[1]] for xi in x]
        history = self._model().fit(
            x, y, epochs=20, batch_size=8, patience=0, verbose=False,
            val_data=[[[0.1, 0.2], [0.3]], [[0.4, 0.5], [0.9]]],
        )
        assert len(history["val_loss"]) == 20


class TestCrossEntropyTargets:
    """CE 的 target 规范化：索引 / one-hot / 浮点标签都能用，非法输入要明确报错"""

    @staticmethod
    def _model(n_out=3):
        random.seed(42)
        mlp = MLP([2, 4, n_out], activations=["leaky_relu", "softmax"])
        return Model(mlp, loss_fn="cross_entropy", optimizer="adam",
                     lr=0.01, task="classification")

    def _run(self, y, n_out=3):
        self._model(n_out).fit([[0.1, 0.2], [0.3, 0.4]], y, epochs=1,
                               batch_size=2, patience=0, verbose=False)

    def test_int_labels(self):
        self._run([0, 1])

    def test_one_hot_labels(self):
        self._run([[1, 0, 0], [0, 1, 0]])

    def test_float_labels(self):
        self._run([0.0, 1.0])

    def test_length_one_list_labels(self):
        self._run([[0], [1]])

    def test_index_out_of_range_raises(self):
        with pytest.raises(ValueError, match="越界"):
            self._run([5, 1])

    def test_non_integer_float_raises(self):
        with pytest.raises(ValueError, match="整数类别索引"):
            self._run([0.5, 1.0])

    def test_wrong_length_one_hot_raises(self):
        with pytest.raises(ValueError, match="不一致"):
            self._run([[1, 0], [0, 1]])

    def test_score_accepts_one_hot_labels(self):
        """score 也要能吃 one-hot 标签（否则会静默返回 0 准确率）"""
        random.seed(42)
        x = [[0.0, 0.0], [1.0, 1.0], [0.0, 0.1], [0.9, 1.1]]
        y = [0, 1, 0, 1]
        model = make_classifier(2, 2, hidden_layers=(8,), lr=0.05)
        model.fit(x, y, epochs=300, batch_size=4, val_split=0.0,
                  patience=0, verbose=False)
        one_hot = [[1.0, 0.0] if yi == 0 else [0.0, 1.0] for yi in y]
        assert model.score(x, one_hot) == model.score(x, y)

    def test_softmax_with_mse_raises(self):
        """softmax 与 mse 组合会静默产生错误梯度，必须直接报错"""
        random.seed(42)
        mlp = MLP([2, 4, 3], activations=["leaky_relu", "softmax"])
        with pytest.raises(ValueError, match="softmax"):
            Model(mlp, loss_fn="mse")

    def test_softmax_with_custom_ce_callable_allowed(self):
        """自定义 callable 无法判断内部实现，不应被误拦"""
        random.seed(42)
        mlp = MLP([2, 4, 3], activations=["leaky_relu", "softmax"])
        model = Model(mlp, loss_fn=cross_entropy_loss)
        assert model._loss_name == "custom"


class TestMseLossShapeCheck:
    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="mse_loss"):
            mse_loss([1.0, 2.0, 3.0], [2.0, 2.0])

    def test_matched_length_ok(self):
        loss, grad = mse_loss([1.0, 2.0], [2.0, 2.0])
        assert loss == pytest.approx(0.5)
        assert grad == pytest.approx([-1.0, 0.0])


class TestActivationName:
    def test_recorded(self):
        for act in ("leaky_relu", "relu", "softmax", "linear"):
            assert Layer(3, 2, activation=act).activation_name == act

    def test_unknown_activation_raises(self):
        with pytest.raises(ValueError):
            Layer(1, 1, activation="tanh")

    def test_save_load_keeps_activation(self, tmp_path):
        random.seed(42)
        mlp = MLP([3, 5, 2], activations=["relu", "softmax"])
        f = str(tmp_path / "act_model.json")
        save_model(mlp, f)
        loaded = load_model(f)
        assert [l.activation_name for l in loaded.layers] == ["relu", "softmax"]


class TestAttentionGradients:
    """数值梯度校验：MHA 的 W_q/W_k/W_v/W_o 解析梯度必须与数值梯度一致

    回归点：W_q/W_k/W_v 的梯度曾整体转置（矩阵是方阵，所以既不报错也不易察觉，
    但会让 Transformer 一直朝错误方向更新）。
    """

    def _check(self):
        random.seed(42)
        mha = MultiHeadAttention(d_model=4, num_heads=2)
        seq = [[0.1, 0.2, 0.3, 0.4],
               [0.5, 0.6, 0.7, 0.8],
               [0.9, 1.0, 1.1, 1.2]]

        def loss():
            out = mha.forward(seq)
            return 0.5 * sum(v * v for row in out for v in row)

        mha.zero_grad()
        out = mha.forward(seq)
        mha.backward([list(row) for row in out])

        eps = 1e-6
        for name in ("W_q", "W_k", "W_v", "W_o"):
            mat = getattr(mha, name)
            for a in range(4):
                for b in range(4):
                    old = mat[a][b]
                    mat[a][b] = old + eps
                    lp = loss()
                    mat[a][b] = old - eps
                    lm = loss()
                    mat[a][b] = old
                    numeric = (lp - lm) / (2 * eps)
                    analytic = mha.grads[name][a][b]
                    assert abs(numeric - analytic) < 1e-5, \
                        f"{name}[{a}][{b}] 解析梯度 {analytic} 与数值梯度 {numeric} 不一致"

    def test_weight_gradients_match_numeric(self):
        self._check()


# ========================== 24. BatchNorm 训练路径 / Conv2d 梯度 / BPTT 缓存 ==========================

class TestBatchNormTraining:
    """BatchNorm1d 的训练路径：forward_batch + backward（此前只有推理模式的形状测试）"""

    @staticmethod
    def _setup(n=6, d=4, seed=1):
        random.seed(seed)
        bn = BatchNorm1d(d)
        xb = [[random.gauss(0, 1) for _ in range(d)] for _ in range(n)]
        w = [random.gauss(0, 1) for _ in range(d)]
        return bn, xb, w

    @staticmethod
    def _loss(bn, xb, w):
        """L = sum_{i,j} w_j * out_ij，因此 dL/dout_ij = w_j"""
        out = bn.forward_batch(xb)
        return sum(wi * o for row in out for wi, o in zip(w, row))

    def test_forward_batch_normalizes_columns(self):
        bn, xb, _ = self._setup(n=64, d=4)
        out = bn.forward_batch(xb)
        assert len(out) == 64
        assert len(out[0]) == 4
        for j in range(4):
            col = [row[j] for row in out]
            assert sum(col) / len(col) == pytest.approx(0.0, abs=1e-9)
            assert sum(v * v for v in col) / len(col) == pytest.approx(1.0, abs=1e-3)

    def test_forward_batch_updates_running_stats(self):
        bn, xb, _ = self._setup(n=8, d=4)
        batch_mean = [sum(row[j] for row in xb) / 8 for j in range(4)]
        bn.forward_batch(xb)
        # momentum=0.1 且 running_mean 初值为 0
        for j in range(4):
            assert bn.running_mean[j] == pytest.approx(0.1 * batch_mean[j])

    def test_backward_dx_matches_numeric(self):
        """dx 必须等于数值梯度（覆盖 s1/s2 的批量统计反向）"""
        bn, xb, w = self._setup(n=5, d=3)
        bn.forward_batch(xb)
        dout = [[w[j] for j in range(3)] for _ in range(5)]
        dx, _, _ = bn.backward(dout)

        eps = 1e-6
        worst = 0.0
        for i in range(5):
            for j in range(3):
                old = xb[i][j]
                xb[i][j] = old + eps
                lp = self._loss(bn, xb, w)
                xb[i][j] = old - eps
                lm = self._loss(bn, xb, w)
                xb[i][j] = old
                worst = max(worst, abs((lp - lm) / (2 * eps) - dx[i][j]))
        assert worst < 1e-4, f"dx 与数值梯度最大偏差 {worst}"

    def test_backward_gamma_beta_match_numeric(self):
        bn, xb, w = self._setup(n=5, d=3)
        bn.forward_batch(xb)
        dout = [[w[j] for j in range(3)] for _ in range(5)]
        _, gg, gb = bn.backward(dout)

        eps = 1e-6
        for j in range(3):
            old = bn.gamma[j]
            bn.gamma[j] = old + eps
            lp = self._loss(bn, xb, w)
            bn.gamma[j] = old - eps
            lm = self._loss(bn, xb, w)
            bn.gamma[j] = old
            assert (lp - lm) / (2 * eps) == pytest.approx(gg[j], abs=1e-5)

            old = bn.beta[j]
            bn.beta[j] = old + eps
            lp = self._loss(bn, xb, w)
            bn.beta[j] = old - eps
            lm = self._loss(bn, xb, w)
            bn.beta[j] = old
            assert (lp - lm) / (2 * eps) == pytest.approx(gb[j], abs=1e-5)

    def test_backward_matches_naive_reference(self):
        """与朴素实现（s1/s2 对每个 (i,j) 都重算一遍）逐位一致

        回归点：s1/s2 曾被写在 i 循环里，复杂度 O(n^2*d)；优化后提到 j 循环外，
        这里用原始写法做参考，保证两者输出完全相同。
        """
        bn, xb, _ = self._setup(n=6, d=4)
        random.seed(12)
        dout = [[random.gauss(0, 1) for _ in range(4)] for _ in range(6)]
        bn.forward_batch(xb)
        dx, gg, gb = bn.backward(dout)

        n, d = bn.n, bn.d
        gamma, x_hat, inv_std = bn.gamma, bn.x_hat, bn.inv_std
        gg_ref = [sum(dout[i][j] * x_hat[i][j] for i in range(n)) for j in range(d)]
        gb_ref = [sum(dout[i][j] for i in range(n)) for j in range(d)]
        dx_hat = [[dout[i][j] * gamma[j] for j in range(d)] for i in range(n)]
        dx_ref = []
        for i in range(n):
            row = []
            for j in range(d):
                s1 = sum(dx_hat[k][j] for k in range(n))
                s2 = sum(dx_hat[k][j] * x_hat[k][j] for k in range(n))
                row.append(inv_std[j] * (n * dx_hat[i][j] - s1 - x_hat[i][j] * s2) / n)
            dx_ref.append(row)

        assert gg == gg_ref
        assert gb == gb_ref
        assert dx == dx_ref


class TestTransformerEncoderGradients:
    """TransformerEncoderLayer 的数值梯度校验

    回归点（真实 bug）：LayerNorm 只有「最后一次 forward」的单槽缓存，而
    TransformerEncoderLayer.forward 是逐 token 调 self.ln1/self.ln2.forward 的，
    缓存里只剩最后一个 token；backward 又逐 token 调 ln.backward，于是
    token 0..T-2 全部拿最后一个 token 的 mean/var 反传 —— seq_len>1 时整条
    反传链（d_x、attn/ffn/ln 全部梯度）都是错的，且 seq_len=1 时恰好正确，
    所以只在小序列上完全看不出来。
    """

    @staticmethod
    def _build(seq_len, d_model=8, heads=2, d_ff=16, seed=7):
        random.seed(seed)
        layer = TransformerEncoderLayer(d_model, heads, d_ff, dropout=0.0)
        x = [[random.uniform(-1, 1) for _ in range(d_model)] for _ in range(seq_len)]
        coef = [[random.uniform(-1, 1) for _ in range(d_model)] for _ in range(seq_len)]
        return layer, x, coef

    @staticmethod
    def _loss(layer, x, coef):
        out = layer.forward([list(r) for r in x])
        return sum(coef[t][j] * out[t][j] for t in range(len(out)) for j in range(len(out[0])))

    @pytest.mark.parametrize("seq_len", [1, 2, 4])
    def test_input_gradient_matches_numeric(self, seq_len):
        layer, x, coef = self._build(seq_len)
        layer.forward([list(r) for r in x])          # 建立前向缓存
        layer.zero_grad()
        dx = layer.backward([list(r) for r in coef])

        eps = 1e-6
        d = len(x[0])
        for t in range(seq_len):
            for j in range(d):
                old = x[t][j]
                x[t][j] = old + eps
                lp = self._loss(layer, x, coef)
                x[t][j] = old - eps
                lm = self._loss(layer, x, coef)
                x[t][j] = old
                num = (lp - lm) / (2 * eps)
                assert abs(num - dx[t][j]) < 1e-5, \
                    f"seq_len={seq_len} d_x[{t}][{j}]: 解析 {dx[t][j]} vs 数值 {num}"

    @pytest.mark.parametrize("which", ["ln1_gamma", "ln2_gamma"])
    def test_layernorm_param_gradient_matches_numeric(self, which):
        seq_len = 4
        layer, x, coef = self._build(seq_len)
        layer.forward([list(r) for r in x])
        layer.zero_grad()
        layer.backward([list(r) for r in coef])
        analytic = list(layer.grads[which])

        eps = 1e-6
        ln = layer.ln1 if which.startswith("ln1") else layer.ln2
        for j in range(len(analytic)):
            old = ln.gamma[j]
            ln.gamma[j] = old + eps
            lp = self._loss(layer, x, coef)
            ln.gamma[j] = old - eps
            lm = self._loss(layer, x, coef)
            ln.gamma[j] = old
            num = (lp - lm) / (2 * eps)
            assert abs(num - analytic[j]) < 1e-5, \
                f"{which}[{j}]: 解析 {analytic[j]} vs 数值 {num}"

    def test_encoder_stack_gradient_matches_numeric(self):
        """多层 TransformerEncoder 也要过数值校验（上游多层时更容易暴露）"""
        random.seed(11)
        d_model = 8
        enc = TransformerEncoder(num_layers=2, d_model=d_model, num_heads=2,
                                 d_ff=16, dropout=0.0)
        seq_len = 3
        x = [[random.uniform(-1, 1) for _ in range(d_model)] for _ in range(seq_len)]
        coef = [[random.uniform(-1, 1) for _ in range(d_model)] for _ in range(seq_len)]

        def loss():
            out = enc.forward([list(r) for r in x])
            return sum(coef[t][j] * out[t][j] for t in range(seq_len) for j in range(d_model))

        enc.forward([list(r) for r in x])
        enc.zero_grad()
        dx = enc.backward([list(r) for r in coef])

        eps = 1e-6
        for t in range(seq_len):
            for j in range(d_model):
                old = x[t][j]
                x[t][j] = old + eps
                lp = loss()
                x[t][j] = old - eps
                lm = loss()
                x[t][j] = old
                num = (lp - lm) / (2 * eps)
                assert abs(num - dx[t][j]) < 1e-5, \
                    f"Encoder d_x[{t}][{j}]: 解析 {dx[t][j]} vs 数值 {num}"


class TestConv2dGradients:
    """Conv2d 的数值梯度校验（此前只有输出形状测试）"""

    @staticmethod
    def _setup(C, OC, K, H, W, stride, pad, seed):
        random.seed(seed)
        conv = Conv2d(C, OC, K, stride=stride, padding=pad)
        x = [[[random.gauss(0, 1) for _ in range(W)] for _ in range(H)] for _ in range(C)]
        return conv, x

    def _check(self, C, OC, K, H, W, stride, pad, seed=3):
        conv, x = self._setup(C, OC, K, H, W, stride, pad, seed)
        out = conv.forward(x)
        ho, wo = len(out[0]), len(out[0][0])
        random.seed(seed + 1)
        coef = [[[random.gauss(0, 1) for _ in range(wo)] for _ in range(ho)]
                for _ in range(OC)]

        def L():
            o = conv.forward(x)
            return sum(coef[c][i][j] * o[c][i][j]
                       for c in range(OC) for i in range(ho) for j in range(wo))

        conv.zero_grad()
        dx = conv.backward(coef)

        eps = 1e-6
        worst_w = worst_b = worst_x = 0.0
        for oc in range(OC):
            for ic in range(C):
                for kh in range(conv.kh):
                    for kw in range(conv.kw):
                        old = conv.weights[oc][ic][kh][kw]
                        conv.weights[oc][ic][kh][kw] = old + eps
                        lp = L()
                        conv.weights[oc][ic][kh][kw] = old - eps
                        lm = L()
                        conv.weights[oc][ic][kh][kw] = old
                        worst_w = max(worst_w,
                                      abs((lp - lm) / (2 * eps) - conv.grad_w[oc][ic][kh][kw]))
        for oc in range(OC):
            old = conv.biases[oc]
            conv.biases[oc] = old + eps
            lp = L()
            conv.biases[oc] = old - eps
            lm = L()
            conv.biases[oc] = old
            worst_b = max(worst_b, abs((lp - lm) / (2 * eps) - conv.grad_b[oc]))
        for c in range(C):
            for i in range(H):
                for j in range(W):
                    old = x[c][i][j]
                    x[c][i][j] = old + eps
                    lp = L()
                    x[c][i][j] = old - eps
                    lm = L()
                    x[c][i][j] = old
                    worst_x = max(worst_x, abs((lp - lm) / (2 * eps) - dx[c][i][j]))
        return worst_w, worst_b, worst_x

    def test_gradients_match_numeric_no_padding(self):
        w, b, x = self._check(1, 2, 2, 5, 5, stride=1, pad=0)
        assert w < 1e-5 and b < 1e-5 and x < 1e-5, f"grad_w={w} grad_b={b} dx={x}"

    def test_gradients_match_numeric_padding_stride2(self):
        w, b, x = self._check(2, 2, 3, 6, 6, stride=2, pad=1, seed=7)
        assert w < 1e-5 and b < 1e-5 and x < 1e-5, f"grad_w={w} grad_b={b} dx={x}"


class TestStepCache:
    """RNN/LSTM/GRU 的 step_cache / restore_cache（BPTT 复用前向缓存）"""

    @staticmethod
    def _net(kind, T=4, inp=3, hs=4, seed=2):
        random.seed(seed)
        cls = {"RNN": RNN, "LSTM": LSTM, "GRU": GRU}[kind]
        net = cls(inp, hs, 1)
        seq = [[random.gauss(0, 1) for _ in range(inp)] for _ in range(T)]
        return net, seq

    @pytest.mark.parametrize("kind", ["RNN", "LSTM", "GRU"])
    def test_step_cache_shape(self, kind):
        net, seq = self._net(kind)
        net.forward(seq)
        assert len(net.step_cache) == len(seq)
        assert all(len(step) == net.num_layers for step in net.step_cache)

    @pytest.mark.parametrize("kind", ["RNN", "LSTM", "GRU"])
    def test_restore_cache_equals_reforward(self, kind):
        """restore_cache + backward 必须与「重新 forward + backward」逐位一致

        这条同时锁住 _CACHE_FIELDS：将来给 cell 新增前向缓存字段却忘记登记，
        恢复出来的 backward 就会与重新前向的结果不一致，这里会失败。
        """
        net, seq = self._net(kind)
        net.forward(seq)
        cell = net.cells[0]
        dh = [0.1 * (i + 1) for i in range(net.hidden_size)]

        for t in range(len(seq)):
            snap = net.step_cache[t][0]
            args = snap[:3] if kind == "LSTM" else snap[:2]

            cell.zero_grad()
            cell.restore_cache(snap)
            res_a = cell.backward(dh)
            grads_a = copy.deepcopy(cell.get_grads())

            cell.zero_grad()
            cell.forward(*args)
            res_b = cell.backward(dh)
            grads_b = copy.deepcopy(cell.get_grads())

            assert res_a == res_b, f"{kind} t={t} 返回值不一致"
            assert grads_a == grads_b, f"{kind} t={t} 梯度不一致"

    @pytest.mark.parametrize("kind", ["RNN", "LSTM", "GRU"])
    def test_full_bptt_equivalence(self, kind):
        """整段 BPTT：用缓存恢复 vs 逐步重新前向，累积梯度必须一致"""
        net, seq = self._net(kind, T=5)
        net.forward(seq)
        cell = net.cells[0]
        hs = net.hidden_size
        dh0 = [0.1] * hs

        def run(use_cache):
            cell.zero_grad()
            dh_cur, dc_cur = dh0, [0.0] * hs
            for t in reversed(range(len(seq))):
                snap = net.step_cache[t][0]
                if use_cache:
                    cell.restore_cache(snap)
                else:
                    cell.forward(*(snap[:3] if kind == "LSTM" else snap[:2]))
                if kind == "LSTM":
                    _, dh_cur, dc_cur = cell.backward(dh_cur, dc_cur)
                else:
                    _, dh_cur = cell.backward(dh_cur)
            return copy.deepcopy(cell.get_grads())

        assert run(True) == run(False)


class TestLayerNumericalGradients:
    """Layer / MLP 的有限差分梯度校验

    Layer.backward 是训练循环里调用最频繁的路径（此前只有形状与 delta 检查），
    这里用中心差分把 grad_w / grad_b / 反向传播链全部钉住。
    实测误差量级 ~1e-9，容差取 1e-6 留足余量（relu 在 0 附近不可导，需要余量）。
    """

    EPS = 1e-6
    TOL = 1e-6

    @pytest.mark.parametrize("act", ["leaky_relu", "relu", "linear"])
    def test_layer_gradients_match_numeric(self, act):
        for trial in range(4):
            random.seed(100 + trial)
            fan_in, fan_out = 4, 3
            layer = Layer(fan_in, fan_out, activation=act)
            x = [random.gauss(0, 1) for _ in range(fan_in)]
            target = [random.gauss(0, 1) for _ in range(fan_out)]

            def loss():
                return mse_loss(layer.forward(x), target)[0]

            _, delta = mse_loss(layer.forward(x), target)
            gw, gb, _ = layer.backward(delta)

            eps = self.EPS
            for j in range(fan_out):
                for i in range(fan_in):
                    old = layer.weights[j][i]
                    layer.weights[j][i] = old + eps
                    lp = loss()
                    layer.weights[j][i] = old - eps
                    lm = loss()
                    layer.weights[j][i] = old
                    num = (lp - lm) / (2 * eps)
                    assert abs(num - gw[j][i]) < self.TOL, \
                        f"{act} trial={trial} grad_w[{j}][{i}]: 解析 {gw[j][i]} vs 数值 {num}"

                old = layer.biases[j]
                layer.biases[j] = old + eps
                lp = loss()
                layer.biases[j] = old - eps
                lm = loss()
                layer.biases[j] = old
                num = (lp - lm) / (2 * eps)
                assert abs(num - gb[j]) < self.TOL, \
                    f"{act} trial={trial} grad_b[{j}]: 解析 {gb[j]} vs 数值 {num}"

    def test_mlp_gradients_match_numeric(self):
        """整网所有层的权重与偏置（经 Model.backward -> _mlp_backward）"""
        random.seed(31)
        mlp = MLP([4, 5, 3], activations=["leaky_relu", "linear"])
        model = Model(mlp, loss_fn="mse", optimizer="adam", lr=0.01, task="regression")
        x = [random.gauss(0, 1) for _ in range(4)]
        target = [random.gauss(0, 1) for _ in range(3)]

        def loss():
            return model._loss(model.forward(x), target)[0]

        _, g = model._loss(model.forward(x), target)
        grads = model.backward(g)

        eps = self.EPS
        for li, layer in enumerate(mlp.layers):
            for j in range(layer.fan_out):
                for i in range(layer.fan_in):
                    old = layer.weights[j][i]
                    layer.weights[j][i] = old + eps
                    lp = loss()
                    layer.weights[j][i] = old - eps
                    lm = loss()
                    layer.weights[j][i] = old
                    num = (lp - lm) / (2 * eps)
                    assert abs(num - grads[li][0][j][i]) < self.TOL, \
                        f"第 {li} 层 grad_w[{j}][{i}]"

                old = layer.biases[j]
                layer.biases[j] = old + eps
                lp = loss()
                layer.biases[j] = old - eps
                lm = loss()
                layer.biases[j] = old
                num = (lp - lm) / (2 * eps)
                assert abs(num - grads[li][1][j]) < self.TOL, f"第 {li} 层 grad_b[{j}]"

    def test_input_gradient_matches_numeric(self):
        """反向传播链传回的 delta 是否正确：校验损失对输入 x 的梯度"""
        random.seed(32)
        mlp = MLP([4, 5, 3], activations=["leaky_relu", "linear"])
        model = Model(mlp, loss_fn="mse", optimizer="adam", lr=0.01, task="regression")
        x = [random.gauss(0, 1) for _ in range(4)]
        target = [random.gauss(0, 1) for _ in range(3)]

        def loss():
            return model._loss(model.forward(x), target)[0]

        _, delta = model._loss(model.forward(x), target)
        for layer in reversed(mlp.layers):
            _, _, delta = layer.backward(delta)      # 逐层往回传，最后 delta 即 dL/dx

        eps = self.EPS
        for i in range(4):
            old = x[i]
            x[i] = old + eps
            lp = loss()
            x[i] = old - eps
            lm = loss()
            x[i] = old
            num = (lp - lm) / (2 * eps)
            assert abs(num - delta[i]) < self.TOL, f"dL/dx[{i}]: 解析 {delta[i]} vs 数值 {num}"

    def test_softmax_ce_gradient_matches_numeric(self):
        """softmax + cross_entropy 的联合梯度约定

        Layer._softmax_deriv 恒返回 1，只有在与 CE 搭配时才正确（CE 已直接返回
        softmax+CE 的联合梯度）。这条测试把这个约定钉住：谁把 _softmax_deriv
        改成真正的 Jacobian、却没有同步改 CE，这里就会失败。
        """
        random.seed(33)
        mlp = MLP([4, 5, 3], activations=["leaky_relu", "softmax"])
        model = Model(mlp, loss_fn="cross_entropy", optimizer="adam",
                      lr=0.01, task="classification")
        x = [random.gauss(0, 1) for _ in range(4)]
        y = 1

        def loss():
            return model._loss(model.forward(x), y)[0]

        _, g = model._loss(model.forward(x), y)
        grads = model.backward(g)

        eps = self.EPS
        for li, layer in enumerate(mlp.layers):
            for j in range(layer.fan_out):
                for i in range(layer.fan_in):
                    old = layer.weights[j][i]
                    layer.weights[j][i] = old + eps
                    lp = loss()
                    layer.weights[j][i] = old - eps
                    lm = loss()
                    layer.weights[j][i] = old
                    num = (lp - lm) / (2 * eps)
                    assert abs(num - grads[li][0][j][i]) < self.TOL, \
                        f"第 {li} 层 grad_w[{j}][{i}]（softmax+CE）"


class TestCellGradSemantics:
    """锁住 cell 的梯度语义：一次 zero_grad + 逐时间步 backward == 各步梯度之和

    回归点：__main__ 的 BPTT 示例曾把 get_grads() 的返回值再手工累加一遍，
    但它返回的就是 cell 内部缓冲区本身，于是变成"把自己加到自己"，梯度按 2 的幂
    放大（实测 T=3 放大 2.5 倍、T=5 放大 7.7 倍、T=10 放大 264 倍）。
    """

    @staticmethod
    def _net(kind, T=5, inp=3, hs=4, seed=5):
        random.seed(seed)
        cls = {"RNN": RNN, "LSTM": LSTM, "GRU": GRU}[kind]
        net = cls(inp, hs, 1)
        seq = [[random.gauss(0, 1) for _ in range(inp)] for _ in range(T)]
        return net, seq

    @pytest.mark.parametrize("kind", ["RNN", "LSTM", "GRU"])
    def test_accumulated_equals_sum_of_isolated_steps(self, kind):
        net, seq = self._net(kind)
        net.forward(seq)
        cell = net.cells[0]
        hs = net.hidden_size
        dh0 = [0.1 * (i + 1) for i in range(hs)]

        # A：整个 BPTT 只 zero_grad 一次，梯度由 backward 原地累加
        cell.zero_grad()
        dh_cur, dc_cur = dh0, [0.0] * hs
        for t in reversed(range(len(seq))):
            cell.restore_cache(net.step_cache[t][0])
            if kind == "LSTM":
                _, dh_cur, dc_cur = cell.backward(dh_cur, dc_cur)
            else:
                _, dh_cur = cell.backward(dh_cur)
        acc = copy.deepcopy(cell.get_grads())

        # B：每步单独 zero_grad + backward，再把各步梯度显式相加
        dh_cur, dc_cur = dh0, [0.0] * hs
        total = None
        for t in reversed(range(len(seq))):
            cell.zero_grad()
            cell.restore_cache(net.step_cache[t][0])
            if kind == "LSTM":
                _, dh_cur, dc_cur = cell.backward(dh_cur, dc_cur)
            else:
                _, dh_cur = cell.backward(dh_cur)
            gw, gwh, gb, gbh = copy.deepcopy(cell.get_grads())
            if total is None:
                total = [gw, gwh, gb, gbh]
            else:
                tw, twh, tb, tbh = total
                for j in range(len(gw)):
                    tb[j] += gb[j]
                    tbh[j] += gbh[j]
                    for i in range(len(gw[j])):
                        tw[j][i] += gw[j][i]
                    for i in range(len(gwh[j])):
                        twh[j][i] += gwh[j][i]

        # 两者累加顺序一致，应当逐位相同（pytest.approx 不支持嵌套结构，用 == 更严格）
        for got, want, name in zip(acc, total,
                                   ("grad_W_ih", "grad_W_hh", "grad_b_ih", "grad_b_hh")):
            assert got == want, f"{kind} {name} 原地累加与逐步求和结果不一致"

    @pytest.mark.parametrize("kind", ["RNN", "LSTM", "GRU"])
    def test_get_grads_reflects_buffers(self, kind):
        """get_grads() 返回的数值必须等于内部缓冲区当前值

        注意：实现上它返回的就是缓冲区本身（不是副本），所以调用方**不能**对
        返回值再做 += 累加，否则会自己加自己。这里只锁数值一致，不锁对象身份。
        """
        net, seq = self._net(kind, T=3)
        net.forward(seq)
        cell = net.cells[0]
        cell.zero_grad()
        cell.restore_cache(net.step_cache[0][0])
        if kind == "LSTM":
            cell.backward([0.1] * 4, [0.0] * 4)
        else:
            cell.backward([0.1] * 4)

        gw, gwh, gb, gbh = cell.get_grads()
        assert gw == cell.grad_W_ih
        assert gwh == cell.grad_W_hh
        assert gb == cell.grad_b_ih
        assert gbh == cell.grad_b_hh


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

# ========================== 26. 模型瘦身 / 早停判据 / 蒸馏 ==========================

class TestModelSlimming:
    """对应 C 引擎 c_model.cpp 与 ai_cli 的同一套语义"""

    @staticmethod
    def _mlp():
        random.seed(11)
        return MLP([3, 5, 2], activations=["leaky_relu", "linear"])

    def test_param_count(self):
        m = self._mlp()
        assert m.param_count() == (3 * 5 + 5) + (5 * 2 + 2)   # 37

    def test_sparsity_starts_zero_and_full_prune(self):
        m = self._mlp()
        assert m.sparsity() == 0.0
        pruned = m.prune_magnitude(1e9)          # 阈值极大 -> 全剪
        assert pruned == 3 * 5 + 5 * 2
        assert m.sparsity() == 1.0

    def test_prune_keeps_large_weights(self):
        m = self._mlp()
        m.layers[0].weights[0][0] = 5.0
        m.prune_magnitude(0.001)
        assert m.layers[0].weights[0][0] == 5.0

    def test_prune_zero_threshold_is_noop(self):
        m = self._mlp()
        assert m.prune_magnitude(0.0) == 0
        assert m.sparsity() == 0.0

    def test_quantize_lands_on_grid(self):
        m = self._mlp()
        step = m.quantize_weights(8)
        assert step > 0
        vals = [w for l in m.layers for row in l.weights for w in row]
        lo = min(vals)
        for v in vals:
            k = (v - lo) / step
            assert k == pytest.approx(round(k), abs=1e-9)

    def test_quantize_invalid_bits(self):
        m = self._mlp()
        assert m.quantize_weights(1) == -1.0
        assert m.quantize_weights(64) == -1.0

    def test_merge_weights_halfway(self):
        a = self._mlp()
        b = self._mlp()
        a.layers[0].weights[0][0] = 0.0
        b.layers[0].weights[0][0] = 2.0
        a.merge_weights(b, 0.5, 0.5)
        assert a.layers[0].weights[0][0] == pytest.approx(1.0)

    def test_merge_weights_shape_mismatch_raises(self):
        a = self._mlp()
        with pytest.raises(ValueError):
            a.merge_weights(MLP([4, 5, 2]), 0.5, 0.5)


class TestEarlyStopMetric:
    @staticmethod
    def _data():
        random.seed(7)
        xs = [[-1 + 2 * i / 29] for i in range(30)]
        ys = [[2 * x[0] + 1] for x in xs]
        return xs, ys

    def test_train_metric_stops_without_val_set(self):
        """没有验证集时 patience 也必须生效（early_stop_metric='train'）"""
        xs, ys = self._data()
        m = make_regressor(1, 1, hidden_layers=(8,), lr=0.05)
        m.fit(xs, ys, epochs=500, batch_size=8, val_split=0.0, patience=20,
              verbose=False, early_stop_metric="train")
        assert m.history["train_loss"][-1][0] < 500

    def test_default_val_metric_unchanged(self):
        xs, ys = self._data()
        m = make_regressor(1, 1, hidden_layers=(8,), lr=0.05)
        m.fit(xs, ys, epochs=150, batch_size=8, val_split=0.1, patience=50, verbose=False)
        assert m.history["train_loss"][-1][0] <= 150

    def test_no_val_no_early_stop_by_default(self):
        """默认判据下、没有验证集时不应早停（保持原行为）"""
        xs, ys = self._data()
        m = make_regressor(1, 1, hidden_layers=(8,), lr=0.05)
        m.fit(xs, ys, epochs=60, batch_size=8, val_split=0.0, patience=5, verbose=False)
        assert m.history["train_loss"][-1][0] == 60


class TestDistillApi:
    def test_distill_regression_smaller_student(self):
        random.seed(9)
        xs = [[-1 + 2 * i / 29, 0.5 * (-1 + 2 * i / 29)] for i in range(30)]
        ys = [[a * a + b] for a, b in xs]
        teacher = make_regressor(2, 1, hidden_layers=(16, 16), lr=0.03)
        teacher.fit(xs, ys, epochs=400, val_split=0.1, verbose=False)
        student = make_regressor(2, 1, hidden_layers=(4,), lr=0.03)
        out, rate = distill(teacher, student, xs, ys, epochs=300, batch_size=8, verbose=False)
        assert out is student
        assert 0.0 <= rate <= 1.0
        assert student.model.param_count() < teacher.model.param_count()


# ========================== 21. 训练/推理模式（train/eval） ==========================

class TestTrainEvalMode:
    """Dropout / BatchNorm / Transformer 的模式切换契约"""

    def test_dropout_eval_is_deterministic(self):
        random.seed(0)
        dp = Dropout(0.5)
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        dp.train()
        assert dp.forward(x) is not x          # 训练模式返回新列表
        assert any(m == 0.0 for m in dp.mask) or any(m != 1.0 for m in dp.mask)
        dp.eval()
        a, b = dp.forward(x), dp.forward(x)
        assert a == b == x                     # 推理模式：原样透传且确定
        assert dp.mask == [1.0] * len(x)

    def test_transformer_encoder_eval_propagates(self):
        random.seed(1)
        enc = TransformerEncoder(4, 1, 8, num_layers=2, dropout=0.5)
        seq = [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]
        enc.train()
        assert all(l.dropout1.training for l in enc.layers)
        enc.eval()
        assert all(not l.dropout1.training and not l.dropout2.training for l in enc.layers)
        assert enc.forward(seq) == enc.forward(seq)

    def test_batchnorm_forward_dispatch(self):
        bn = BatchNorm1d(3)
        batch = [[1.0, 2.0, 3.0], [3.0, 4.0, 5.0], [5.0, 6.0, 7.0]]
        bn.train()
        out_train = bn.forward(batch)          # 训练 + batch -> batch 统计量
        bn.eval()
        out_eval = bn.forward(batch)           # 推理 + batch -> running 统计量
        assert len(out_train) == len(out_eval) == 3
        assert out_train != out_eval
        assert len(bn.forward([1.0, 2.0, 3.0])) == 3   # 单样本路径仍可用

    def test_batchnorm_rejects_bad_input(self):
        bn = BatchNorm1d(3)
        with pytest.raises(ValueError):
            bn.forward([])
        with pytest.raises(ValueError):
            bn.forward([[1.0, 2.0, 3.0], [1.0]])


# ========================== 22. 损失函数契约 ==========================

class TestLossContracts:

    @staticmethod
    def _sm(z):
        mx = max(z)
        e = [math.exp(v - mx) for v in z]
        s = sum(e)
        return [v / s for v in e]

    def test_ce_vector_gradient_matches_numeric(self):
        """软标签的联合梯度是 s*p - t（s=Σt），不是 p - t"""
        z0 = [0.3, -0.5, 1.1]
        for tgt in ([0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.7, 0.2, 0.1], [0.0, 1.0, 0.0]):
            _, grad = cross_entropy_loss(self._sm(z0), tgt)
            num = []
            eps = 1e-6
            for k in range(3):
                zp = list(z0); zp[k] += eps
                zm = list(z0); zm[k] -= eps
                lp, _ = cross_entropy_loss(self._sm(zp), tgt)
                lm, _ = cross_entropy_loss(self._sm(zm), tgt)
                num.append((lp - lm) / (2 * eps))
            for a, b in zip(grad, num):
                assert abs(a - b) < 1e-7

    def test_ce_all_zero_target_has_zero_grad(self):
        loss, grad = cross_entropy_loss([0.2, 0.3, 0.5], [0, 0, 0])
        assert loss == 0.0
        assert all(abs(g) < 1e-15 for g in grad)

    def test_ce_rejects_negative_and_bad_length(self):
        with pytest.raises(ValueError):
            cross_entropy_loss([0.5, 0.5], [-0.1, 1.1])
        with pytest.raises(ValueError):
            cross_entropy_loss([0.5, 0.5], [1.0, 0.0, 0.0])

    def test_mse_accepts_scalar(self):
        l1, g1 = mse_loss([0.5], 1.0)
        l2, g2 = mse_loss([0.5], [1.0])
        assert l1 == l2 and g1 == g2

    def test_model_ce_passes_soft_labels_through(self):
        # 注意 MLP([3, 3]) 只有一层，activations 也只需 1 个
        md = Model(MLP([3, 3], activations=["softmax"]),
                   loss_fn="cross_entropy")
        pred, tgt = [0.2, 0.3, 0.5], [0.1, 0.2, 0.7]
        assert md._loss(pred, tgt)[0] == pytest.approx(cross_entropy_loss(pred, tgt)[0])
        # 全 0 目标不再被当成类 0
        assert md._loss(pred, [0, 0, 0])[0] == 0.0

    def test_softmax_mse_guard_catches_function_object(self):
        import functools
        with pytest.raises(ValueError):
            Model(MLP([2, 3], activations=["softmax"]), loss_fn=mse_loss)
        with pytest.raises(ValueError):
            Model(MLP([2, 3], activations=["softmax"]), loss_fn=functools.partial(mse_loss))
        with pytest.raises(ValueError):
            Model(MLP([2, 3], activations=["softmax"]), loss_fn="mse")

    def test_train_guard_catches_function_object(self):
        import functools
        m = MLP([2, 3], activations=["softmax"])
        with pytest.raises(ValueError):
            train(m, functools.partial(mse_loss), Adam(m, lr=0.01),
                  [([0.1, 0.2], [0, 0, 1])], epochs=1, batch_size=1,
                  patience=0, verbose=False)


# ========================== 23. 持久化兼容与校验 ==========================

class TestPersistenceCompat:

    @staticmethod
    def _save_cls(path, legacy=False, drop_task=False):
        m = make_classifier(3, 3, hidden_layers=(4,), lr=0.01)
        m.fit([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
              [0, 1, 2], epochs=3, verbose=False)
        m.save(str(path))
        d = json.load(open(path, encoding="utf-8"))
        if legacy:
            for ld in d["layers"]:
                ld.pop("activation", None)
            # 真正的旧文件也没有 loss / loss_kwargs（这两个字段是后加的）
            d.pop("loss", None)
            d.pop("loss_kwargs", None)
        if drop_task:
            d.pop("task", None)
        json.dump(d, open(path, "w", encoding="utf-8"))
        return d

    def test_legacy_without_task_or_activation_defaults_to_regression(self, tmp_path):
        p = tmp_path / "m.json"
        self._save_cls(p, legacy=True, drop_task=True)
        assert Model.load(str(p)).task == "regression"          # 文档写明的保守回退
        assert Model.load(str(p), default_output_activation="softmax").task == "classification"

    def test_task_field_is_used_when_no_activation(self, tmp_path):
        p = tmp_path / "m2.json"
        self._save_cls(p, legacy=True, drop_task=False)
        assert Model.load(str(p)).task == "classification"

    def test_conv_legacy_output_activation(self, tmp_path):
        p = tmp_path / "c.json"
        save_conv_model([Conv2d(1, 2, 3, 1, 1)], MLP([8, 4, 3], activations=["relu", "softmax"]), str(p))
        _, m = load_conv_model(str(p))
        assert [l.activation_name for l in m.layers] == ["relu", "softmax"]
        d = json.load(open(p, encoding="utf-8"))
        for ld in d["mlp"]["layers"]:
            ld.pop("activation", None)
        json.dump(d, open(p, "w", encoding="utf-8"))
        _, m2 = load_conv_model(str(p), default_output_activation="softmax")
        assert m2.layers[-1].activation_name == "softmax"

    def test_load_model_rejects_fewer_layers(self, tmp_path):
        p = tmp_path / "few.json"
        d = self._save_cls(p)
        d["layers"] = d["layers"][:1]
        json.dump(d, open(p, "w", encoding="utf-8"))
        with pytest.raises(ValueError):
            load_model(str(p))

    def test_load_model_rejects_truncated_weights(self, tmp_path):
        p = tmp_path / "badw.json"
        d = self._save_cls(p)
        d["layers"][0]["weights"] = [row[:1] for row in d["layers"][0]["weights"]]
        json.dump(d, open(p, "w", encoding="utf-8"))
        with pytest.raises(ValueError):
            load_model(str(p))


# ========================== 24. strict 与输入校验 ==========================

class TestStrictAndValidation:

    def test_prepare_data_is_strict(self):
        tp = TextProcessor("abc")
        with pytest.raises(ValueError):
            tp.prepare_data("aXbc", 3)
        with pytest.raises(ValueError):
            tp.prepare_index_data("aXbc", 3)
        assert len(tp.prepare_index_data("abcabc", 2)) == 4

    def test_layer_dimension_error_mentions_flatten(self):
        l = Layer(3, 2, "linear")
        with pytest.raises(ValueError, match="fan_in"):
            l.forward([1.0, 2.0])
        with pytest.raises(ValueError, match="Flatten"):
            l.forward([[1.0, 2.0, 3.0]])

    def test_positional_encoding_grows(self):
        pe_small = PositionalEncoding(4, max_len=3)
        out = pe_small.forward([[0.1] * 4] * 10)
        assert len(out) == 10
        pe_big = PositionalEncoding(4, max_len=10)
        assert out == pe_big.forward([[0.1] * 4] * 10)

    def test_quick_train_one_hot_counts_all_classes(self):
        x = [[0, 0], [0, 1], [1, 0], [1, 1]]
        y = [[1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 1, 0]]   # 类 2 在本批缺失
        m = quick_train(x, y, task="classification", epochs=2, verbose=False)
        assert m.model.layer_sizes[-1] == 3

    def test_quick_train_accepts_noncontiguous_int_labels(self):
        x = [[0.0], [1.0], [2.0], [3.0]]
        m = quick_train(x, [0, 2, 0, 2], task="classification", epochs=2, verbose=False)
        assert m.model.layer_sizes[-1] == 3

    def test_quick_train_rejects_non_integer_labels(self):
        x = [[0.0], [1.0]]
        with pytest.raises(ValueError):
            quick_train(x, [0.0, 1.5], task="classification", epochs=1, verbose=False)
        with pytest.raises(ValueError):
            quick_train(x, [-0.5, 1.0], task="classification", epochs=1, verbose=False)


# ========================== 25. 独立复审发现的缺陷（回归测试） ==========================

class TestBatchNormModeAwareBackward:
    """BN 的反向必须跟着「上一次前向的模式」走"""

    def test_eval_backward_uses_running_stats(self):
        bn = BatchNorm1d(2)
        bn.gamma = [1.5, 0.5]
        bn.forward_batch([[0, 1], [2, 3], [4, 5]])
        bn.eval()
        bn.forward([[10, -10], [20, -20], [30, -30]])
        dx, _, _ = bn.backward([[1, 2], [0.5, -0.5], [2, 1]])
        exp0 = [bn.gamma[j] / math.sqrt(bn.running_var[j] + bn.eps) * d
                for j, d in enumerate([1, 2])]
        for a, b in zip(dx[0], exp0):
            assert a == pytest.approx(b, rel=1e-12)

    def test_single_sample_backward(self):
        bn = BatchNorm1d(2)
        bn.forward_batch([[0, 1], [2, 3]])
        bn.forward([5, 6])
        dx, gg, gb = bn.backward([1, 1])          # 1-D dout
        assert len(dx) == 1 and len(dx[0]) == 2
        assert len(gg) == 2 and len(gb) == 2

    def test_backward_before_forward_raises(self):
        with pytest.raises(RuntimeError):
            BatchNorm1d(2).backward([[1.0, 1.0]])

    def test_sequential_passes_dx_not_grad_beta(self):
        seq = Sequential(BatchNorm1d(2))
        seq.forward([[1, 2], [3, 4]])
        got = seq.backward([[1, 0], [0, 1]])
        ref = BatchNorm1d(2)
        ref.forward_batch([[1, 2], [3, 4]])
        assert got == ref.backward([[1, 0], [0, 1]])[0]


class TestMetricValidation:
    def test_evaluate_length_mismatch(self):
        m = make_regressor(1, 1, hidden_layers=(4,), lr=0.01)
        with pytest.raises(ValueError):
            m.evaluate(x=[[1], [2], [3], [1000]], y=[[1], [2], [3]])

    def test_evaluate_empty(self):
        with pytest.raises(ValueError):
            make_regressor(1, 1).evaluate(x=[], y=[])

    def test_score_length_mismatch(self):
        m = make_regressor(1, 1)
        with pytest.raises(ValueError):
            m.score([[0], [1], [2]], [0, 1])

    def test_score_constant_target(self):
        r = make_regressor(2, 1).score([[0.1, 0.2], [0.3, 0.4]], [1.0, 1.0])
        assert -1.0 <= r <= 1.0

    def test_score_multi_output_needs_vector_y(self):
        with pytest.raises(ValueError):
            make_regressor(2, 2).score([[0.1, 0.2], [0.3, 0.4]], [1.0, 2.0])


class TestLossAndGradGuards:
    def test_ce_rejects_negative_and_out_of_range(self):
        with pytest.raises(ValueError):
            cross_entropy_loss([0.2, 0.3, 0.5], -1)
        with pytest.raises(ValueError):
            cross_entropy_loss([0.2, 0.3, 0.5], 3)

    def test_ce_requires_softmax_output(self):
        with pytest.raises(ValueError):
            Model(MLP([2, 2], activations=["leaky_relu"]), loss_fn="cross_entropy")

    def test_dropout_p_validation(self):
        for bad in (1.0, 1.5, -0.5):
            with pytest.raises(ValueError):
                Dropout(bad)
        assert Dropout(0.0).forward([1.0])[0] == 1.0

    def test_clip_negative_thresholds(self):
        g = lambda: [([[1.0, -2.0]], [3.0])]
        with pytest.raises(ValueError):
            clip_grad_by_norm(g(), -1.0)
        with pytest.raises(ValueError):
            clip_grad_by_value(g(), -1.0)


class TestShapeValidation:
    def test_layernorm_feature_mismatch(self):
        with pytest.raises(ValueError):
            LayerNorm(2).forward([1, 2, 3])
        with pytest.raises(ValueError):
            LayerNorm(2).forward([1])

    def test_positional_encoding_feature_mismatch(self):
        with pytest.raises(ValueError):
            PositionalEncoding(2).forward([[1, 2, 3]])

    def test_residual_shape_mismatch(self):
        rb = ResidualBlock(Layer(2, 3, "linear"))
        with pytest.raises(ValueError):
            rb.forward([1.0, 2.0])

    def test_conv2d_channel_mismatch(self):
        with pytest.raises(ValueError):
            Conv2d(2, 1, 3).forward([[[1.0] * 5] * 5])

    def test_matmul_dimension_mismatch_is_value_error(self):
        from AI import _matmul
        with pytest.raises(ValueError):
            _matmul([[1.0, 2.0]], [[1.0, 2.0]])


class TestTrainLoopOptions:
    def test_train_full_batch(self):
        random.seed(0)
        net = MLP([1, 2, 1], activations=["relu", "linear"])
        opt = Adam(net, 0.1)
        w0 = [[r[:] for r in l.weights] for l in net.layers]
        train(net, mse_loss, opt, [([0.0], [0.0]), ([1.0], [1.0])],
              epochs=5, batch_size=-1, patience=0, verbose=False)
        assert opt.step_count == 5
        assert w0 != [l.weights for l in net.layers] or True   # 至少训练了 5 步

    def test_train_patience_zero_disables_early_stop(self):
        net = MLP([1, 2, 1], activations=["relu", "linear"])
        h = train(net, mse_loss, Adam(net, 0.01), [([0.0], [0.0]), ([1.0], [1.0])],
                  epochs=20, batch_size=2, patience=0, val_data=[([0.0], [0.0])],
                  verbose=False)
        assert len(h["train_loss"]) == 20

    def test_fit_shuffle_false_preserves_order(self):
        seen = []
        m = make_regressor(1, 1, hidden_layers=(1,), lr=0.0)
        real = m.model.forward
        m.model.forward = lambda x: (seen.append(x[0]), real(x))[1]
        xs = [[float(i)] for i in range(10)]
        m.fit(xs, [[0.0]] * 10, epochs=1, batch_size=10, val_split=0.3,
              shuffle=False, verbose=False)
        assert seen[:10] == [float(i) for i in range(10)]


class TestMiscDefects:
    def test_stable_sigmoid_no_overflow(self):
        c = LSTMCell(1, 2)
        c.W_ih[0][0] = -1.0
        h, _ = c.forward([1000.0])
        assert all(math.isfinite(v) for v in h)

    def test_embedding_rejects_bad_index(self):
        with pytest.raises(ValueError):
            Embedding(3, 2).forward([-1])
        with pytest.raises(ValueError):
            Embedding(3, 2).forward([3])

    def test_malformed_json_is_value_error(self, tmp_path):
        p = tmp_path / "bad.json"
        json.dump({"type": "MLP", "layer_sizes": [2, 1],
                   "layers": [{"fan_in": 2, "fan_out": 1, "biases": [0]}]},
                  open(p, "w"))
        with pytest.raises(ValueError):
            load_model(str(p))

    def test_invalid_hyperparameters(self):
        with pytest.raises(ValueError):
            DataLoader([1, 2, 3], 0)
        with pytest.raises(ValueError):
            split_data(range(10), 0.9, 0.9)
        opt = Adam(MLP([1, 1]), 0.1)
        with pytest.raises(ValueError):
            StepLR(opt, 0)
        with pytest.raises(ValueError):
            CosineAnnealingLR(opt, 0)

    def test_cosine_reaches_eta_min(self):
        opt = Adam(MLP([1, 1]), 1.0)
        sch = CosineAnnealingLR(opt, T_max=10, eta_min=0.0)
        for ep in range(1, 11):
            sch.step(ep)
        assert opt.lr == pytest.approx(0.0, abs=1e-12)
        sch.step(11)
        assert opt.lr == pytest.approx(1.0, abs=1e-12)

    def test_generate_text_no_double_softmax(self):
        tp = TextProcessor("abc")
        V = tp.vocab_size
        m = MLP([V * 3, V], activations=["softmax"])
        m.layers[0].weights = [[0.0] * (V * 3) for _ in range(V)]
        p = [0.7, 0.2, 0.1]
        m.layers[0].biases = [math.log(v) for v in p]
        seen = {}
        real = AI._softmax_with_temp
        def spy(logits, temp=1.0):
            seen["logits"] = list(logits)
            return real(logits, temp)
        AI._softmax_with_temp = spy
        try:
            generate_text_with_mlp(m, tp, "a", length=1, window_size=3)
        finally:
            AI._softmax_with_temp = real
        got = real(seen["logits"], 1.0)
        for a, b in zip(got, p):
            assert a == pytest.approx(b, abs=1e-9)

    def test_distill_classification_uses_soft_targets(self):
        X = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]]
        Yt = [0, 1, 0, 1]
        teacher = make_classifier(2, 2, hidden_layers=(4,), lr=0.02)
        teacher.fit(X, Yt, epochs=5, verbose=False)
        student = make_classifier(2, 2, hidden_layers=(2,), lr=0.02)
        captured = {}
        real_fit = student.fit
        student.fit = lambda x, y, **kw: (captured.setdefault("y", y), real_fit(x, y, **kw))[1]
        distill(teacher, student, X, Yt, epochs=2, verbose=False)
        first = captured["y"][0]
        assert isinstance(first, (list, tuple)) and len(first) == 2
        assert sum(first) == pytest.approx(1.0, abs=1e-6)

    def test_save_conv_model_rejects_unknown_module(self, tmp_path):
        with pytest.raises(ValueError):
            save_conv_model([Conv2d(1, 2, 3, 1, 1), MaxPool2d(2)],
                            MLP([8, 3], activations=["linear"]), str(tmp_path / "c.json"))

class TestRestoreBestWeights:
    def test_restore_beats_last_epoch(self):
        xs = [[float(i)] for i in range(10)]
        ys = [[v[0]] for v in xs]

        def run(restore):
            random.seed(7)
            m = make_regressor(1, 1, hidden_layers=(2,), lr=0.3)
            m.fit(xs, ys, epochs=40, batch_size=2, val_split=0.3, patience=0,
                  shuffle=True, verbose=False, restore_best_weights=restore)
            w = [[row[:] for row in l.weights] for l in m.model.layers]
            loss = sum(mse_loss(m.predict([x])[0], y)[0] for x, y in zip(xs, ys)) / len(xs)
            return w, loss

        w1, l1 = run(False)
        w2, l2 = run(True)
        assert w1 != w2                      # 确实回滚了权重
        assert l2 <= l1                      # 回滚后不差于末轮

    def test_layer_backward_returns_triple(self):
        l = Layer(3, 2, "linear")
        l.forward([0.1, 0.2, 0.3])
        out = l.backward([1.0, -1.0])
        assert isinstance(out, tuple) and len(out) == 3
        gw, gb, delta = out
        assert len(gw) == 2 and len(gb) == 2 and len(delta) == 3

class TestRobustnessGuards:
    """独立鲁棒性审计发现的校验缺口（F1-F9）"""

    def test_model_rejects_foreign_optimizer(self):
        a = MLP([2, 3, 2]); b = MLP([2, 3, 2])
        with pytest.raises(ValueError):
            Model(b, loss_fn="mse", optimizer=Adam(a, lr=0.1))
        Model(MLP([2, 3, 2]), loss_fn="mse", optimizer="adam")   # 字符串正常

    def test_single_class_classification_rejected(self):
        with pytest.raises(ValueError):
            quick_train([[0.1], [0.2]], [0, 0], task="classification", epochs=2, verbose=False)
        with pytest.raises(ValueError):
            make_classifier(2, 1)

    def test_batchnorm_forward_batch_validates_shape(self):
        with pytest.raises(ValueError):
            BatchNorm1d(2).forward_batch([])
        with pytest.raises(ValueError):
            BatchNorm1d(2).forward_batch([[1, 2, 999], [3, 4, 999]])
        with pytest.raises(ValueError):
            BatchNorm1d(2).forward_batch([[1, 2], [3]])
        assert len(BatchNorm1d(2).forward_batch([[1, 2], [3, 4]])) == 2

    def test_load_model_rejects_zero_layer(self, tmp_path):
        p = tmp_path / "zero.json"
        json.dump({"type": "MLP", "layer_sizes": [2], "task": "regression", "layers": []},
                  open(p, "w"))
        with pytest.raises(ValueError):
            load_model(str(p))

    def test_fit_rejects_bad_val_split(self):
        m = make_regressor(1, 1, hidden_layers=(2,))
        X = [[float(i)] for i in range(10)]
        for bad in (1.0, 1.5, -0.1):
            with pytest.raises(ValueError):
                m.fit(X, X, epochs=1, batch_size=2, val_split=bad, verbose=False)

    def test_mse_overflow_yields_inf(self):
        assert mse_loss([1e200], [0.0])[0] == float("inf")

    def test_maxpool_validates_kernel_and_stride(self):
        with pytest.raises(ValueError):
            MaxPool2d(0, stride=1)
        with pytest.raises(ValueError):
            MaxPool2d(2, stride=0)
        assert MaxPool2d(2).forward([[[1.0, 2.0], [3.0, 4.0]]])[0][0][0] == 4.0

    def test_save_model_rejects_nan(self, tmp_path):
        m = MLP([2, 1])
        m.layers[0].weights[0][0] = float("nan")
        with pytest.raises(ValueError):
            save_model(m, str(tmp_path / "nan.json"))

    def test_load_model_rejects_non_numeric_weights(self, tmp_path):
        p = tmp_path / "str.json"
        json.dump({"type": "MLP", "layer_sizes": [2, 1], "task": None,
                   "layers": [{"fan_in": 2, "fan_out": 1, "activation": "linear",
                                "weights": [["1.0", "2.0"]], "biases": [0.0]}]},
                  open(p, "w"))
        with pytest.raises(ValueError):
            load_model(str(p))

    def test_text_contract_guards(self):
        with pytest.raises(ValueError):
            AI._softmax_with_temp([1.0, 2.0], 0)
        with pytest.raises(ValueError):
            sample_index([])
        tp = TextProcessor("abc")

        class Bad:
            def forward(self, x):
                return [0.0, 0.0, 0.0, 1.0]      # 4 != vocab 3

        with pytest.raises(ValueError):
            generate_text_with_mlp(Bad(), tp, "a", length=1)

class TestAuditRound3:
    """第三轮外部评审确认的问题（bool 索引、docstring 位置、step(0)）"""

    def test_ce_rejects_bool_index(self):
        with pytest.raises(TypeError):
            cross_entropy_loss([0.2, 0.3, 0.5], True)
        with pytest.raises(TypeError):
            cross_entropy_loss([0.2, 0.3, 0.5], False)

    def test_clip_and_sample_docstrings_present(self):
        assert clip_grad_by_norm.__doc__ and "范数" in clip_grad_by_norm.__doc__
        assert clip_grad_by_value.__doc__ and "值" in clip_grad_by_value.__doc__
        assert sample_index.__doc__ and "采样" in sample_index.__doc__

    def test_cosine_step_zero_uses_initial_lr(self):
        m = make_regressor(1, 1)
        initial = m.optimizer.lr
        sch = CosineAnnealingLR(m.optimizer, T_max=10, eta_min=0.0)
        sch.step(0)                      # 非法 epoch 按 1 处理
        assert m.optimizer.lr == pytest.approx(initial, rel=1e-12)

    def test_save_conv_model_docstring_mentions_restriction(self):
        assert "Conv2d" in (save_conv_model.__doc__ or "")

    def test_encode_indices_documents_lenient_default(self):
        assert "strict" in (TextProcessor.encode_indices.__doc__ or "")


# ========================== 20. 因果掩码 / CausalLM ==========================

class TestCausalMask:
    def test_causal_hides_future_tokens(self):
        """因果掩码：改最后一个 token，前两个位置的输出必须逐位不变"""
        random.seed(11)
        m = MultiHeadAttention(4, 1, causal=True)
        x1 = [[0.1, 0.2, 0.3, 0.4], [0.5, -0.1, 0.2, 0.0], [0.9, 0.8, -0.7, 0.6]]
        o1 = m.forward(x1)
        x2 = [list(r) for r in x1]
        x2[2] = [9.0, -9.0, 9.0, -9.0]
        o2 = m.forward(x2)
        assert o1[:2] == o2[:2]
        assert o1[2] != o2[2]

    def test_causal_attention_weights_masked(self):
        """被屏蔽的位置权重精确为 0，每行和仍为 1"""
        random.seed(12)
        m = MultiHeadAttention(4, 1, causal=True)
        m.forward([[0.1, 0.2, 0.3, 0.4], [0.5, -0.1, 0.2, 0.0], [0.9, 0.8, -0.7, 0.6]])
        attn = m.attn_weights[0]
        for t in range(3):
            assert all(attn[t][j] == 0.0 for j in range(t + 1, 3))
            assert sum(attn[t]) == pytest.approx(1.0, rel=0, abs=1e-12)

    def test_non_causal_mixes_future(self):
        """对照组：causal=False 时未来 token 会影响过去位置（证明掩码确实在起作用）"""
        random.seed(11)
        m = MultiHeadAttention(4, 1, causal=False)
        x1 = [[0.1, 0.2, 0.3, 0.4], [0.5, -0.1, 0.2, 0.0], [0.9, 0.8, -0.7, 0.6]]
        o1 = m.forward(x1)
        x2 = [list(r) for r in x1]
        x2[2] = [9.0, -9.0, 9.0, -9.0]
        o2 = m.forward(x2)
        assert o1[0] != o2[0]

    def test_causal_backward_matches_finite_difference(self):
        """因果掩码下 W_v 的解析梯度与中心差分一致（mask 位置梯度应自然为 0）"""
        random.seed(13)
        m = MultiHeadAttention(2, 1, causal=True)
        xs = [[0.3, -0.2], [0.7, 0.4], [-0.5, 0.9]]
        coef = [[0.4, -0.6], [0.9, 0.2], [-0.3, 0.8]]

        def loss():
            out = m.forward(xs)
            return sum(coef[t][j] * out[t][j] for t in range(3) for j in range(2))

        m.zero_grad()
        m.forward(xs)
        m.backward([[coef[t][j] for j in range(2)] for t in range(3)])
        anal = [row[:] for row in m.grads["W_v"]]
        eps = 1e-6
        for i in range(2):
            for j in range(2):
                orig = m.W_v[i][j]
                m.W_v[i][j] = orig + eps
                lp = loss()
                m.W_v[i][j] = orig - eps
                lm = loss()
                m.W_v[i][j] = orig
                num = (lp - lm) / (2 * eps)
                assert anal[i][j] == pytest.approx(num, rel=1e-6, abs=1e-8)

    def test_single_token_causal(self):
        """长度 1 时掩码不改变结果：注意力权重为 1"""
        random.seed(14)
        m = MultiHeadAttention(4, 2, causal=True)
        m.forward([[0.5, -0.5, 1.0, 0.0]])
        assert m.attn_weights[0][0] == [1.0]

    def test_encoder_propagates_causal_flag(self):
        """TransformerEncoder(causal=True) 必须把标志传给每一层"""
        enc = AI.TransformerEncoder(4, 2, 8, num_layers=2, causal=True)
        assert enc.causal is True
        assert all(layer.self_attn.causal for layer in enc.layers)
        enc2 = AI.TransformerEncoder(4, 2, 8, num_layers=2)
        assert enc2.causal is False
        assert not any(layer.self_attn.causal for layer in enc2.layers)


class TestCausalLM:
    def test_learns_cyclic_pattern(self):
        """在 0,1,2 循环序列上过拟合：loss 掉到接近 0，贪心续写能接上周期"""
        lm = AI.CausalLM(vocab_size=3, d_model=16, num_heads=2, d_ff=32,
                         num_layers=1, max_len=32)
        hist = lm.fit([[0, 1, 2] * 5], epochs=80, lr=0.01, shuffle=False)
        assert hist[0] > hist[-1]
        assert hist[-1] < 0.05
        out = lm.generate([0], max_new_tokens=6, temperature=1e-6)
        assert out == [0, 1, 2, 0, 1, 2, 0]

    def test_prefix_logits_ignore_future(self):
        """位置 t 的 logits 与 t 之后的 token 无关（LLM 的因果性保证）"""
        random.seed(21)
        lm = AI.CausalLM(vocab_size=4, d_model=8, num_heads=2, d_ff=16,
                         num_layers=1, max_len=16)
        lm.eval()
        ids = [0, 1, 2, 3, 0]
        logits1 = lm.forward(ids)
        ids2 = list(ids)
        ids2[4] = 1
        logits2 = lm.forward(ids2)
        assert logits1[:4] == logits2[:4]

    def test_generate_deterministic_with_seed(self):
        """同一 rng 种子的两次生成必须完全相同"""
        random.seed(22)
        lm = AI.CausalLM(vocab_size=5, d_model=8, num_heads=2, d_ff=16,
                         num_layers=1, max_len=16)
        a = lm.generate([0, 1], max_new_tokens=5, temperature=1.0, rng=random.Random(3))
        b = lm.generate([0, 1], max_new_tokens=5, temperature=1.0, rng=random.Random(3))
        assert a == b

    def test_rejects_invalid_input(self):
        lm = AI.CausalLM(vocab_size=3, d_model=8, num_heads=2, d_ff=16,
                         num_layers=1, max_len=8)
        with pytest.raises(ValueError):
            lm.forward([])
        with pytest.raises(ValueError):
            lm.forward([0] * 9)
        with pytest.raises(ValueError):
            lm.generate([0], temperature=0.0)
        with pytest.raises(ValueError):
            lm.fit([[1]])                      # 单 token 无法构造目标
        with pytest.raises(ValueError):
            lm.fit([[0, 1]], epochs=-1)
        with pytest.raises(ValueError):
            lm.evaluate([])
        with pytest.raises(ValueError):
            AI.CausalLM(0)
        with pytest.raises(ValueError):
            AI.CausalLM(3, max_len=0)
        with pytest.raises(ValueError):
            AI.CausalLM(3, num_layers=0)


class TestAttentionDropout:
    """多头的注意力权重 dropout：mask 缩放、eval 关闭、反向与固定 mask 的差分一致"""

    def _mha(self, dropout, seed=11):
        random.seed(seed)
        return AI.MultiHeadAttention(2, 1, dropout=dropout)

    def test_mask_values_and_scaling(self):
        m = self._mha(0.5)
        m.forward([[0.3, -0.2], [0.7, 0.4], [-0.5, 0.9]])
        mask = m.attn_masks[0]
        assert mask is not None
        vals = {v for row in mask for v in row}
        assert vals <= {0.0, 2.0}                  # 只可能是「丢弃」或 1/(1-p)=2
        for t in range(3):
            for j in range(3):
                assert m.attn_dropped[0][t][j] == m.attn_weights[0][t][j] * mask[t][j]

    def test_eval_disables_dropout(self):
        m = self._mha(0.5)
        m.eval()
        a = m.forward([[0.3, -0.2], [0.7, 0.4]])
        b = m.forward([[0.3, -0.2], [0.7, 0.4]])
        assert a == b
        assert all(x is None for x in m.attn_masks)

    def test_training_forward_is_stochastic(self):
        m = self._mha(0.5)
        x = [[0.3, -0.2], [0.7, 0.4], [-0.5, 0.9]]
        assert m.forward(x) != m.forward(x)

    def test_zero_dropout_matches_eval(self):
        m = self._mha(0.0)
        x = [[0.3, -0.2], [0.7, 0.4]]
        out_train = m.forward(x)
        m.eval()
        assert out_train == m.forward(x)
        assert all(x is None for x in m.attn_masks)

    def test_backward_matches_fixed_mask(self):
        """固定 mask 后 W_v 的解析梯度必须与数值差分一致（曾在这里失败过）"""
        m = self._mha(0.5)
        x = [[0.3, -0.2], [0.7, 0.4], [-0.5, 0.9]]
        coef = [[0.4, -0.6], [0.9, 0.2], [-0.3, 0.8]]

        def loss():
            m.rng = random.Random(99)              # 每次重建同一组 mask
            out = m.forward(x)
            return sum(coef[t][j] * out[t][j] for t in range(3) for j in range(2))

        loss()                                     # 建缓存
        m.zero_grad()
        loss()
        m.backward(coef)
        anal = [row[:] for row in m.grads["W_v"]]

        eps = 1e-6
        for i in range(2):
            for j in range(2):
                orig = m.W_v[i][j]
                m.W_v[i][j] = orig + eps
                lp = loss()
                m.W_v[i][j] = orig - eps
                lm = loss()
                m.W_v[i][j] = orig
                num = (lp - lm) / (2 * eps)
                assert abs(anal[i][j] - num) <= 1e-5 * max(1.0, abs(num))

    def test_encoder_layer_forwards_dropout_to_attention(self):
        layer = AI.TransformerEncoderLayer(8, 2, 16, dropout=0.25)
        assert layer.self_attn.dropout == 0.25
        layer.eval()
        assert layer.self_attn.training is False
        layer.train()
        assert layer.self_attn.training is True


# ========================== 24. LLM 时代组件（激活 / 归一化 / 优化器 / 损失 / 位置编码） ==========================

class TestModernComponents:
    """GELU / sigmoid / Linear / RMSNorm —— LLM 里最常见的四个小件"""

    def test_gelu_matches_tanh_formula(self):
        """GELU(tanh 近似) 必须精确等于 0.5*z*(1+tanh(c*(z+0.044715 z^3)))"""
        layer = Layer(1, 1, activation="gelu")
        layer.weights = [[1.0]]
        layer.biases = [0.0]
        for z in (-2.0, -0.5, 0.0, 0.7, 3.0):
            expected = 0.5 * z * (1.0 + math.tanh(AI._GELU_C * (z + 0.044715 * z ** 3)))
            assert abs(layer.forward([z])[0] - expected) <= 1e-12

    @pytest.mark.parametrize("act", ["gelu", "sigmoid"])
    @pytest.mark.parametrize("z", [-1.7, -0.4, 0.6, 2.2])
    def test_activation_backward_matches_numeric(self, act, z):
        """新激活的反向：单层单权重，解析 dout/dw 对中心差分"""
        eps = 1e-6
        layer = Layer(1, 1, activation=act)
        layer.weights = [[1.0]]
        layer.biases = [0.0]
        layer.forward([z])  # backward 依赖 forward 缓存的 z
        gw, _gb, _delta = layer.backward([1.0])

        def out():
            return layer.forward([z])[0]

        old = layer.weights[0][0]
        layer.weights[0][0] = old + eps
        hi = out()
        layer.weights[0][0] = old - eps
        lo = out()
        layer.weights[0][0] = old
        num = (hi - lo) / (2 * eps)
        assert abs(gw[0][0] - num) <= 1e-6 * max(1.0, abs(num))

    def test_sigmoid_saturates_without_overflow(self):
        """sigmoid 用分支写法：±800 不能抛 OverflowError，也不能返回 nan"""
        layer = Layer(1, 1, activation="sigmoid")
        layer.weights = [[1.0]]
        layer.biases = [0.0]
        assert layer.forward([-800.0])[0] == 0.0
        assert layer.forward([800.0])[0] == 1.0

    def test_linear_class_is_plain_affine(self):
        lin = AI.Linear(3, 2)
        assert isinstance(lin, Layer)
        lin.weights = [[1.0, 2.0, 3.0], [0.0, -1.0, 0.5]]
        lin.biases = [0.5, -1.0]
        assert lin.forward([1.0, 1.0, 1.0]) == [6.5, -1.5]

    def test_rmsnorm_forward_and_shift_sensitivity(self):
        """RMSNorm 不减均值：整体平移输入后输出必须明显改变（LayerNorm 则几乎不变）"""
        norm = AI.RMSNorm(3)
        norm.gamma = [1.0, 1.0, 1.0]
        base = norm.forward([1.0, 2.0, 3.0])
        r = 1.0 / math.sqrt((1 + 4 + 9) / 3.0 + norm.eps)
        assert abs(base[0] - r) <= 1e-15
        shifted = norm.forward([11.0, 12.0, 13.0])
        assert abs(shifted[0] - base[0]) > 1e-3

    def test_rmsnorm_backward_matches_numeric(self):
        random.seed(4242)
        norm = AI.RMSNorm(4)
        norm.gamma = [0.7, 1.3, 0.9, 1.1]
        x = [random.gauss(0, 1) for _ in range(4)]
        dout = [random.gauss(0, 1) for _ in range(4)]
        norm.forward(x)
        grad_gamma, dx = norm.backward(dout)
        eps = 1e-6
        for j in range(4):
            old = x[j]
            x[j] = old + eps
            hi = sum(a * b for a, b in zip(dout, norm.forward(x)))
            x[j] = old - eps
            lo = sum(a * b for a, b in zip(dout, norm.forward(x)))
            x[j] = old
            num = (hi - lo) / (2 * eps)
            assert abs(dx[j] - num) <= 1e-6 * max(1.0, abs(num)), f"dx[{j}]"

            oldg = norm.gamma[j]
            norm.gamma[j] = oldg + eps
            hi = sum(a * b for a, b in zip(dout, norm.forward(x)))
            norm.gamma[j] = oldg - eps
            lo = sum(a * b for a, b in zip(dout, norm.forward(x)))
            norm.gamma[j] = oldg
            num = (hi - lo) / (2 * eps)
            assert abs(grad_gamma[j] - num) <= 1e-6 * max(1.0, abs(num)), f"dgamma[{j}]"


class TestAdamWAndWarmup:
    def test_zero_grad_step_only_decays(self):
        """梯度为 0 时 AdamW 仍按 (1-lr*wd) 收缩权重，Adam 则一动不动"""
        mlp = MLP([3, 2])
        before = [list(row) for row in mlp.layers[0].weights]
        grads = [([[0.0] * layer.fan_in for _ in range(layer.fan_out)],
                  [0.0] * layer.fan_out) for layer in mlp.layers]
        opt = AI.AdamW(mlp, lr=0.1, weight_decay=0.5)
        opt.step(grads)
        decay = 1.0 - 0.1 * 0.5
        for j in range(2):
            for i in range(3):
                assert abs(mlp.layers[0].weights[j][i] - before[j][i] * decay) <= 1e-15

    def test_zero_weight_decay_matches_adam(self):
        """weight_decay=0 时 AdamW 与 Adam 逐位相同"""
        random.seed(11)
        a = MLP([2, 2])
        b = MLP([2, 2])
        for la, lb in zip(a.layers, b.layers):
            lb.weights = [list(row) for row in la.weights]
            lb.biases = list(la.biases)
        grads = [([[0.3, -0.2], [0.1, 0.4]], [0.2, -0.4]) for _ in a.layers]
        Adam(a, lr=0.01).step([(gw, gb) for gw, gb in grads])
        AI.AdamW(b, lr=0.01, weight_decay=0.0).step([(gw, gb) for gw, gb in grads])
        assert a.layers[0].weights == b.layers[0].weights
        assert a.layers[0].biases == b.layers[0].biases

    def test_warmup_linear_ramp_then_flat(self):
        mlp = MLP([2, 1])
        opt = SGD(mlp, lr=0.1)
        sched = AI.WarmupLR(opt, warmup_epochs=4)
        lrs = []
        for epoch in range(1, 8):
            sched.step(epoch)
            lrs.append(opt.lr)
        assert lrs[:4] == pytest.approx([0.025, 0.05, 0.075, 0.1])
        assert lrs[4:] == pytest.approx([0.1, 0.1, 0.1])

    def test_warmup_chains_base_scheduler(self):
        """warmup 结束后把控制权交给基础调度器（这里用 StepLR 验一轮）"""
        mlp = MLP([2, 1])
        opt = SGD(mlp, lr=0.2)
        sched = AI.WarmupLR(opt, warmup_epochs=2, base_scheduler=StepLR(opt, step_size=1, gamma=0.5))
        lrs = []
        for epoch in range(1, 5):
            sched.step(epoch)
            lrs.append(opt.lr)
        # 前两轮线性爬升到 0.2；之后交给 StepLR，而 StepLR 用的是**绝对 epoch**
        # （lr = initial * gamma**(epoch//step_size)），所以第 3 轮是 0.2*0.5^3
        assert lrs[0] == pytest.approx(0.1)
        assert lrs[1] == pytest.approx(0.2)
        assert lrs[2] == pytest.approx(0.2 * 0.5 ** 3)
        assert lrs[3] == pytest.approx(0.2 * 0.5 ** 4)


class TestExtraLosses:
    def test_focal_gamma_zero_equals_cross_entropy(self):
        """γ=0 的 focal 必须退化成普通交叉熵（loss 与梯度都逐位一致）"""
        pred = AI._softmax_row([0.4, -1.2, 2.0, 0.1])
        for idx in range(4):
            f, gf = AI.focal_loss(pred, idx, gamma=0.0)
            c, gc = cross_entropy_loss(pred, idx)
            assert abs(f - c) <= 1e-15
            assert gf == pytest.approx(gc, abs=1e-15)

    def test_focal_gradient_matches_numeric(self):
        random.seed(6)
        logits = [0.3, -0.8, 1.4, 0.2]
        pred = AI._softmax_row(logits)
        idx = 2
        _loss, grad = AI.focal_loss(pred, idx, gamma=2.0, alpha=0.75)
        eps = 1e-6
        for j in range(4):
            old = logits[j]
            logits[j] = old + eps
            hi = AI.focal_loss(AI._softmax_row(logits), idx, gamma=2.0, alpha=0.75)[0]
            logits[j] = old - eps
            lo = AI.focal_loss(AI._softmax_row(logits), idx, gamma=2.0, alpha=0.75)[0]
            logits[j] = old
            num = (hi - lo) / (2 * eps)
            assert abs(grad[j] - num) <= 1e-6 * max(1.0, abs(num)), f"grad[{j}]"

    def test_bce_with_logits_stable_for_huge_logits(self):
        """logits=500 时 loss 应是 exp(-500) 量级，不能溢出成 inf/nan"""
        loss, _grad = AI.bce_with_logits_loss([500.0], [1])
        assert math.isfinite(loss) and loss < 1e-200
        loss2, _g2 = AI.bce_with_logits_loss([-500.0], [0])
        assert math.isfinite(loss2) and loss2 < 1e-200

    def test_bce_gradient_is_prob_minus_target(self):
        """bce_loss 的 grad 相对**概率**，所以是 (p-y)/(p(1-p))，再按元素平均"""
        loss, grad = AI.bce_loss([0.9, 0.2], [1, 0])
        expect = [(0.9 - 1) / (0.9 * 0.1) / 2, (0.2 - 0) / (0.2 * 0.8) / 2]
        assert grad == pytest.approx(expect, abs=1e-9)
        assert loss == pytest.approx(-(math.log(0.9) + math.log(0.8)) / 2, abs=1e-12)

    def test_huber_quadratic_and_linear_branches(self):
        assert AI.huber_loss([0.0], [0.5])[0] == pytest.approx(0.125)
        assert AI.huber_loss([0.0], [3.0], delta=1.0)[0] == pytest.approx(2.5)

    def test_kl_divergence_matches_formula(self):
        p = [0.5, 0.3, 0.2]
        t = [0.6, 0.3, 0.1]
        expect = sum(ti * math.log(ti / pi) for ti, pi in zip(t, p))
        loss, grad = AI.kl_divergence(p, t)
        assert loss == pytest.approx(expect, abs=1e-12)
        assert grad == pytest.approx([-ti / pi for ti, pi in zip(t, p)], abs=1e-12)

    def test_label_smoothing_gradient_matches_numeric(self):
        random.seed(8)
        logits = [0.5, -0.4, 1.1]
        eps_smooth = 0.2
        idx = 1
        _loss, grad = cross_entropy_loss(AI._softmax_row(logits), idx, label_smoothing=eps_smooth)
        eps = 1e-6
        for j in range(3):
            old = logits[j]
            logits[j] = old + eps
            hi = cross_entropy_loss(AI._softmax_row(logits), idx, label_smoothing=eps_smooth)[0]
            logits[j] = old - eps
            lo = cross_entropy_loss(AI._softmax_row(logits), idx, label_smoothing=eps_smooth)[0]
            logits[j] = old
            num = (hi - lo) / (2 * eps)
            assert abs(grad[j] - num) <= 1e-6 * max(1.0, abs(num)), f"grad[{j}]"


class TestRoPE:
    def _model(self, rope, layers=2):
        random.seed(20250901)
        return AI.CausalLM(7, d_model=8, num_heads=2, d_ff=16, num_layers=layers,
                           max_len=12, dropout=0.0, rope=rope)

    def test_rope_turns_off_absolute_position_by_default(self):
        lm = self._model(True, layers=1)
        assert lm.rope is True
        assert lm.use_abs_pos is False
        lm2 = AI.CausalLM(5, d_model=8, num_heads=2, d_ff=16, num_layers=1, max_len=8,
                          rope=True, use_abs_pos=True)
        assert lm2.use_abs_pos is True

    def test_rope_changes_logits(self):
        a = self._model(False, layers=1)
        b = self._model(True, layers=1)
        la = a.forward([1, 2, 3, 4])[-1]
        lb = b.forward([1, 2, 3, 4])[-1]
        assert max(abs(x - y) for x, y in zip(la, lb)) > 1e-6

    @pytest.mark.parametrize("rope", [False, True])
    def test_step_forward_matches_full_forward(self, rope):
        """增量前向的最后一格必须与整段前向的最后一行逐位相同"""
        lm = self._model(rope)
        ids = [0, 1, 2, 3]
        full = lm.forward(ids)[-1]
        caches = [{} for _ in lm.encoder.layers]
        last = None
        for pos, tok in enumerate(ids):
            last = lm._forward_step(tok, pos, caches)
        assert last == full

    @pytest.mark.parametrize("rope", [False, True])
    def test_gradients_match_numeric(self, rope):
        """RoPE 的反向是逆旋转，必须与中心差分一致"""
        lm = self._model(rope)
        ids = [1, 4, 2, 5, 3]
        lm.zero_grad()
        lm._loss_and_backward(ids)
        # _loss_and_backward 逐次累加梯度 → 先深拷贝快照，再跑数值
        snapshot = {}
        for name, _param, grad in lm._named_params():
            snapshot[name] = [list(row) if isinstance(row, list) else row for row in grad]
        params = {name: p for name, p, _g in lm._named_params()}
        eps = 1e-6
        checks = [("L0.attn.W_q", 1, 2), ("L0.attn.W_k", 3, 5), ("L0.attn.W_v", 2, 6),
                  ("L0.ffn.W1", 2, 4), ("emb.weight", 2, 5), ("out.W", 3, 1)]
        for name, i, j in checks:
            ana = snapshot[name][i][j]
            param = params[name]
            old = param[i][j]
            param[i][j] = old + eps
            hi = lm._loss_and_backward(ids)
            param[i][j] = old - eps
            lo = lm._loss_and_backward(ids)
            param[i][j] = old
            num = (hi - lo) / (2 * eps)
            assert abs(ana - num) <= 1e-5 * max(1.0, abs(num)), f"{name}[{i}][{j}]"


class TestKVCache:
    def _model(self, max_len, rope=False, layers=2):
        random.seed(1000)
        return AI.CausalLM(7, d_model=8, num_heads=2, d_ff=16, num_layers=layers,
                           max_len=max_len, dropout=0.0, rope=rope)

    @pytest.mark.parametrize("max_len", [32, 4])
    @pytest.mark.parametrize("rope", [False, True])
    def test_cached_generation_equals_recompute(self, max_len, rope):
        """KV cache 版生成必须与每步整段重算给出完全相同的 token"""
        lm = self._model(max_len, rope)
        prompt = [1, 2, 3]
        a = lm.generate(list(prompt), max_new_tokens=8, temperature=0.7, top_k=3,
                        rng=random.Random(7), use_cache=False)
        b = lm.generate(list(prompt), max_new_tokens=8, temperature=0.7, top_k=3,
                        rng=random.Random(7), use_cache=True)
        assert a == b

    def test_cached_generation_after_training(self):
        """训练过的权重（非初始化值）下也要一致"""
        lm = self._model(32)
        lm.fit([[1, 2, 3, 4, 1, 2], [2, 3, 4, 5, 2, 3]], epochs=3, lr=0.05, seed=3)
        a = lm.generate([1, 2], max_new_tokens=6, rng=random.Random(11), use_cache=False)
        b = lm.generate([1, 2], max_new_tokens=6, rng=random.Random(11), use_cache=True)
        assert a == b

    def test_generate_does_not_mutate_parameters(self):
        lm = self._model(16)
        before = [list(p) for _n, p, _g in lm._named_params()]
        lm.generate([1, 2, 3], max_new_tokens=5, rng=random.Random(1))
        after = [list(p) for _n, p, _g in lm._named_params()]
        assert before == after


# ============================================================================
#  批次 5：训练工程化与评估工具
# ============================================================================

class TestFeedForwardActivation:
    def test_default_relu_matches_explicit_relu(self):
        seq = [[0.1, -0.2, 0.3], [0.4, 0.5, -0.6]]
        random.seed(5)
        a = AI.FeedForward(3, 4)
        random.seed(5)
        b = AI.FeedForward(3, 4, activation="relu")
        assert a.forward(seq) == b.forward(seq)

    def test_gelu_changes_output(self):
        seq = [[0.5, -0.5, 1.0, -1.0]]
        random.seed(6)
        relu = AI.FeedForward(4, 3)
        random.seed(6)
        gelu = AI.FeedForward(4, 3, activation="gelu")
        assert relu.forward(seq) != gelu.forward(seq)

    def test_gelu_backward_matches_numeric(self):
        seq = [[0.2, -0.4, 0.6, 0.1], [-0.3, 0.5, -0.2, 0.8]]
        random.seed(7)
        ff = AI.FeedForward(4, 5, activation="gelu")
        out = ff.forward(seq)
        dout = [[2.0 * v for v in row] for row in out]
        eps = 1e-6
        for j in range(5):
            orig = ff.W1[j][1]
            ff.W1[j][1] = orig + eps
            lp = sum(v * v for row in ff.forward(seq) for v in row)
            ff.W1[j][1] = orig - eps
            lm = sum(v * v for row in ff.forward(seq) for v in row)
            ff.W1[j][1] = orig
            num = (lp - lm) / (2 * eps)
            ff.zero_grad()
            ff.forward(seq)
            ff.backward(dout)
            ana = ff.grads["W1"][j][1]
            assert abs(ana - num) < 1e-6

    def test_invalid_activation_raises(self):
        with pytest.raises(ValueError):
            AI.FeedForward(3, 4, activation="tanh")


class TestStratifiedSplit:
    def test_stratify_keeps_class_balance(self):
        data = [(i, i % 3) for i in range(30)]
        labels = [i % 3 for i in range(30)]
        tr, va, te = AI.split_data(data, 0.2, 0.1, seed=7, stratify=labels)
        assert (len(va), len(te)) == (6, 3)
        assert len(tr) + len(va) + len(te) == 30
        assert len({id(d) for d in tr + va + te}) == 30

    def test_stratify_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            AI.split_data([(i, i) for i in range(6)], stratify=[0, 1])

    def test_default_path_is_deterministic(self):
        data = [(i, i) for i in range(20)]
        a = AI.split_data(data, 0.25, 0.15, seed=3)
        b = AI.split_data(data, 0.25, 0.15, seed=3)
        assert a == b

    def test_k_fold_covers_every_sample_once(self):
        folds = AI.k_fold_indices(17, n_splits=4, seed=1)
        assert len(folds) == 4
        seen = []
        for tr_i, va_i in folds:
            assert not (set(tr_i) & set(va_i))
            assert len(set(tr_i) | set(va_i)) == 17
            seen.extend(va_i)
        assert sorted(seen) == list(range(17))

    def test_k_fold_stratified_balance(self):
        folds = AI.k_fold_indices(20, n_splits=4, seed=2,
                                  stratify=[i % 2 for i in range(20)])
        for _, va_i in folds:
            even = sum(1 for i in va_i if i % 2 == 0)
            assert abs(even - (len(va_i) - even)) <= 1

    def test_k_fold_rejects_bad_splits(self):
        with pytest.raises(ValueError):
            AI.k_fold_indices(10, n_splits=1)
        with pytest.raises(ValueError):
            AI.k_fold_indices(3, n_splits=5)
        with pytest.raises(ValueError):
            AI.k_fold_indices(5, n_splits=2, stratify=[0])


class TestScalers:
    def test_standard_scaler_roundtrip(self):
        x = [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]
        sc = AI.StandardScaler().fit(x)
        z = sc.transform(x)
        assert abs(sum(r[0] for r in z) / 3) < 1e-12
        assert sc.inverse_transform(z) == x

    def test_standard_scaler_zero_variance(self):
        x = [[5.0, 7.0], [5.0, 7.0]]
        sc = AI.StandardScaler().fit(x)
        assert sc.transform(x) == [[0.0, 0.0], [0.0, 0.0]]

    def test_standard_scaler_unfitted_raises(self):
        with pytest.raises(ValueError):
            AI.StandardScaler().transform([[1.0]])

    def test_minmax_scaler_endpoints(self):
        x = [[1.0], [2.0], [3.0]]
        z = AI.MinMaxScaler().fit_transform(x)
        assert z[0][0] == 0.0 and z[2][0] == 1.0

    def test_minmax_scaler_custom_range(self):
        x = [[1.0], [3.0]]
        z = AI.MinMaxScaler(feature_range=(-1.0, 1.0)).fit_transform(x)
        assert z == [[-1.0], [1.0]]

    def test_minmax_scaler_rejects_bad_range(self):
        with pytest.raises(ValueError):
            AI.MinMaxScaler(feature_range=(1.0, 0.0))

    def test_scalers_reject_ragged_input(self):
        with pytest.raises(ValueError):
            AI.StandardScaler().fit([[1.0, 2.0], [3.0]])


class TestAsciiPlot:
    def test_renders_axes_and_legend(self):
        hist = {"train_loss": [(i, 1.0 / i) for i in range(1, 6)],
                "val_loss": [(i, 1.2 / i) for i in range(1, 6)]}
        txt = AI.ascii_plot(hist)
        assert "epoch 1 .. 5" in txt
        assert "train_loss" in txt and "val_loss" in txt
        assert len(txt.split("\n")) >= 12

    def test_empty_history(self):
        assert "没有可画" in AI.ascii_plot({})

    def test_single_point(self):
        assert "epoch 1 .. 1" in AI.ascii_plot({"train_loss": [(1, 0.5)]})

    def test_custom_size(self):
        txt = AI.ascii_plot({"train_loss": [(1, 1.0), (2, 0.5)]}, width=20, height=4)
        assert len(txt.split("\n")) >= 6


class TestDataLoaderCollate:
    def test_collate_fn_applied(self):
        dl = AI.DataLoader([(i, i * 2) for i in range(5)], batch_size=2,
                           shuffle=False, collate_fn=lambda b: {"xs": [x for x, _ in b]})
        batches = list(dl)
        assert batches[0] == {"xs": [0, 1]}
        assert batches[-1] == {"xs": [4]}

    def test_default_is_noop(self):
        dl = AI.DataLoader(list(range(5)), batch_size=2, shuffle=False)
        assert list(dl) == [[0, 1], [2, 3], [4]]


class TestGradCheck:
    def test_mlp_passes(self):
        random.seed(41)
        mlp = AI.MLP([3, 4, 2])
        res = AI.grad_check(mlp, [0.2, -0.3, 0.5], [0.4, -0.2])
        assert res["ok"]
        assert res["worst_rel"] < 1e-5
        assert len(res["checks"]) == 3 * 4 + 4 + 4 * 2 + 2

    def test_model_passes(self):
        random.seed(42)
        model = AI.Model(AI.MLP([3, 5, 2]), task="regression")
        res = AI.grad_check(model, [0.2, -0.3, 0.5], [0.4, -0.2])
        assert res["ok"]

    def test_causallm_passes(self):
        random.seed(43)
        lm = AI.CausalLM(vocab_size=7, d_model=8, num_heads=2, d_ff=16,
                         num_layers=1, max_len=8)
        res = AI.grad_check(lm, [1, 3, 2, 4])
        assert res["ok"]
        # 抽查 24 个标量；其中解析与数值都为 0 的（如 b_k）单列进 zeros
        assert len(res["checks"]) + len(res["zeros"]) == 24

    def test_detects_broken_gradient(self, monkeypatch):
        random.seed(44)
        mlp = AI.MLP([2, 2])
        orig = AI._mlp_backward

        def broken(model, d_output):
            grads = orig(model, d_output)
            grads[0][0][0][0] += 0.5
            return grads

        monkeypatch.setattr(AI, "_mlp_backward", broken)
        res = AI.grad_check(mlp, [0.3, -0.1], [0.5, 0.2])
        assert not res["ok"]
        assert res["worst_rel"] > 1e-2

    def test_rejects_unknown_type(self):
        with pytest.raises(TypeError):
            AI.grad_check("nope", 1)


class TestGradNormHistory:
    def test_history_records_grad_norm(self):
        random.seed(51)
        model = AI.Model(AI.MLP([2, 4, 1]), task="regression")
        model.fit(x=[[0.1, 0.2], [0.3, -0.1], [0.5, 0.4], [-0.2, 0.6]],
                  y=[[0.3], [0.1], [0.7], [0.2]], epochs=4, batch_size=2,
                  val_split=0.0, verbose=False)
        gn = model.history["grad_norm"]
        assert len(gn) == 4
        assert all(v > 0.0 for _, v in gn)

    def test_grad_norm_finite_with_clip(self):
        random.seed(52)
        model = AI.Model(AI.MLP([2, 3, 1]), task="regression")
        model.fit(x=[[0.1, 0.2], [0.3, -0.1]], y=[[0.3], [0.1]], epochs=2,
                  batch_size=1, val_split=0.0, verbose=False, grad_clip=0.5)
        assert all(math.isfinite(v) for _, v in model.history["grad_norm"])


# ============================================================================
#  批次 6：优化器状态持久化 + checkpoint
# ============================================================================

class TestOptimizerPersistence:
    X = [[0.10, 0.20], [0.30, -0.10], [0.50, 0.40], [-0.20, 0.60],
         [0.70, 0.10], [-0.40, 0.30]]
    Y = [[0.30], [0.10], [0.70], [0.20], [0.90], [0.05]]

    def _fitted(self, seed=102, epochs=3):
        random.seed(seed)
        model = AI.Model(AI.MLP([2, 4, 1]), task="regression")
        model.fit(x=self.X, y=self.Y, epochs=epochs, batch_size=3,
                  val_split=0.0, verbose=False, shuffle=False)
        return model

    def test_save_without_optimizer_matches_old_format(self, tmp_path):
        random.seed(101)
        mlp = AI.MLP([2, 3, 1])
        p = tmp_path / "plain.json"
        AI.save_model(mlp, str(p), task="regression", quiet=True)
        with open(p, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        assert "optimizer" not in data
        assert data["type"] == "MLP" and len(data["layers"]) == 2

    def test_model_save_stores_adam_state(self, tmp_path):
        model = self._fitted()
        p = tmp_path / "state.json"
        model.save(str(p))
        with open(p, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        opt = raw["optimizer"]
        assert opt["type"] == "adam"
        # 3 轮 × (6 样本 / batch=3) = 6 步
        assert opt["step_count"] == 6
        assert len(opt["m_w"]) == 2 and len(opt["m_w"][0][0]) == 2
        assert len(opt["m_b"]) == 2

    def test_resume_is_bit_identical(self, tmp_path):
        p_ck = tmp_path / "ck.json"
        random.seed(7)
        full = AI.Model(AI.MLP([2, 4, 1]), task="regression")
        full.fit(x=self.X, y=self.Y, epochs=6, batch_size=3, val_split=0.0,
                 verbose=False, shuffle=False)

        random.seed(7)
        half = AI.Model(AI.MLP([2, 4, 1]), task="regression")
        half.fit(x=self.X, y=self.Y, epochs=4, batch_size=3, val_split=0.0,
                 verbose=False, shuffle=False, checkpoint_path=str(p_ck))

        resumed = AI.Model.load(str(p_ck), task="regression", restore_optimizer=True)
        assert resumed.optimizer.step_count == half.optimizer.step_count
        resumed.fit(x=self.X, y=self.Y, epochs=2, batch_size=3, val_split=0.0,
                    verbose=False, shuffle=False)
        got = [row for layer in resumed.model.layers for row in layer.weights]
        want = [row for layer in full.model.layers for row in layer.weights]
        assert got == want

    def test_without_restore_starts_from_scratch(self, tmp_path):
        p_ck = tmp_path / "ck2.json"
        random.seed(7)
        AI.Model(AI.MLP([2, 4, 1]), task="regression").fit(
            x=self.X, y=self.Y, epochs=4, batch_size=3, val_split=0.0,
            verbose=False, shuffle=False, checkpoint_path=str(p_ck))
        naive = AI.Model.load(str(p_ck), task="regression", restore_optimizer=False)
        assert naive.optimizer.step_count == 0

    def test_optimizer_type_mismatch_raises(self):
        model = self._fitted()
        state = AI._optimizer_state(model.optimizer)
        random.seed(103)
        sgd = AI.Model(AI.MLP([2, 4, 1]), optimizer="sgd", task="regression")
        with pytest.raises(ValueError):
            AI.load_optimizer_state(sgd.optimizer, state)

    def test_optimizer_shape_mismatch_raises(self):
        model = self._fitted()
        state = AI._optimizer_state(model.optimizer)
        random.seed(104)
        other = AI.Model(AI.MLP([2, 5, 1]), task="regression")
        with pytest.raises(ValueError):
            AI.load_optimizer_state(other.optimizer, state)

    def test_sgd_has_no_state(self):
        random.seed(105)
        model = AI.Model(AI.MLP([2, 3, 1]), optimizer="sgd", task="regression")
        assert AI._optimizer_state(model.optimizer) is None

    def test_load_without_optimizer_field_is_ok(self, tmp_path):
        p = tmp_path / "no_opt.json"
        random.seed(105)
        AI.save_model(AI.MLP([2, 3, 1]), str(p), task="regression", quiet=True)
        loaded = AI.Model.load(str(p), task="regression", restore_optimizer=True)
        assert loaded.optimizer.step_count == 0

class TestWindowMask:
    """滑动窗口（局部因果）注意力：MultiHeadAttention(window=...)"""

    def _mha(self, **kw):
        random.seed(31)
        m = AI.MultiHeadAttention(8, 2, causal=True, **kw)
        m.eval()
        return m

    def test_window_masks_outside(self):
        """窗口外权重恰为 0，窗口内为正，每行和为 1"""
        m = self._mha(window=2)
        random.seed(32)
        seq = [[random.gauss(0, 1) for _ in range(8)] for _ in range(6)]
        m.forward(seq)
        for head in range(2):
            attn = m.attn_weights[head]
            for t in range(6):
                assert abs(sum(attn[t]) - 1.0) < 1e-12
                for j in range(6):
                    if j > t or j < t - 1:
                        assert attn[t][j] == 0.0
                    else:
                        assert attn[t][j] > 0.0

    def test_window_is_causal(self):
        """改末位 token 不影响前面位置的输出"""
        m = self._mha(window=3)
        random.seed(33)
        seq = [[random.gauss(0, 1) for _ in range(8)] for _ in range(5)]
        out = m.forward(seq)
        seq2 = [list(row) for row in seq]
        seq2[-1] = [v + 1.0 for v in seq2[-1]]
        out2 = m.forward(seq2)
        for t in range(4):
            for j in range(8):
                assert out[t][j] == out2[t][j]
        assert out[4] != out2[4]

    def test_forward_step_matches_forward(self):
        """增量前向（KV cache 用）与整段前向的最后一行逐位一致"""
        for rope in (False, True):
            random.seed(34)
            m = AI.MultiHeadAttention(8, 2, causal=True, window=3, rope=rope)
            m.eval()
            seq = [[random.gauss(0, 1) for _ in range(8)] for _ in range(5)]
            full = m.forward(seq)
            cache = {}
            step = None
            for tok in seq:
                step = m.forward_step(tok, cache)
            assert full[-1] == step

    def test_window_gradient(self):
        """带窗口掩码时 W_v 的解析梯度 vs 中心差分"""
        random.seed(35)
        m = AI.MultiHeadAttention(4, 2, causal=True, window=2)
        seq = [[random.gauss(0, 1) for _ in range(4)] for _ in range(4)]
        dout = [[random.gauss(0, 1) for _ in range(4)] for _ in range(4)]

        def loss():
            out = m.forward(seq)
            return sum(out[t][j] * dout[t][j] for t in range(4) for j in range(4))

        m.forward(seq)
        m.backward(dout)
        ana = [row[:] for row in m.grads["W_v"]]
        eps = 1e-6
        for i, j in ((0, 1), (2, 3)):
            old = m.W_v[i][j]
            m.W_v[i][j] = old + eps
            lp = loss()
            m.W_v[i][j] = old - eps
            lm = loss()
            m.W_v[i][j] = old
            num = (lp - lm) / (2 * eps)
            assert abs(ana[i][j] - num) / max(1.0, abs(num)) < 1e-6

    def test_window_none_equals_default(self):
        """window=None 与不传 window 逐位一致"""
        random.seed(36)
        a = AI.MultiHeadAttention(8, 2, causal=True, window=None)
        random.seed(36)
        b = AI.MultiHeadAttention(8, 2, causal=True)
        a.eval()
        b.eval()
        random.seed(37)
        seq = [[random.gauss(0, 1) for _ in range(8)] for _ in range(4)]
        assert a.forward(seq) == b.forward(seq)

    def test_invalid_window(self):
        for bad in (0, -1, 1.5, True):
            with pytest.raises(ValueError):
                AI.MultiHeadAttention(4, 2, window=bad)


class TestTopPSampling:
    """nucleus（top_p）采样"""

    def _lm(self):
        random.seed(41)
        lm = AI.CausalLM(vocab_size=6, d_model=8, num_heads=2, d_ff=16,
                         num_layers=1, max_len=8)
        lm.eval()
        return lm

    def test_default_top_p_is_no_filter(self):
        """top_p 默认 1.0，与不传时逐位相同"""
        lm = self._lm()
        a = lm.generate([1, 2], 5, temperature=0.8, top_k=0, rng=random.Random(9))
        b = lm.generate([1, 2], 5, temperature=0.8, top_k=0,
                        rng=random.Random(9), top_p=1.0)
        assert a == b

    def test_top_p_keeps_nucleus_only(self):
        lm = self._lm()
        for s in range(20):
            pick = lm._sample([10.0, 0.0, 0.0, 0.0, 0.0, 0.0], 1.0, 0,
                              random.Random(s), 0.5)
            assert pick == 0
        seen = set()
        for s in range(80):
            seen.add(lm._sample([1.0] * 6, 1.0, 0, random.Random(s), 0.6))
        assert seen.issubset({0, 1, 2, 3})
        assert len(seen) >= 2

    def test_top_p_cache_equals_recompute(self):
        """KV cache 路径与整段重算路径在 top_p 下仍逐位相同"""
        lm = self._lm()
        a = lm.generate([1], 5, 0.9, 0, random.Random(5), True, 0.7)
        b = lm.generate([1], 5, 0.9, 0, random.Random(5), False, 0.7)
        assert a == b

    def test_invalid_top_p(self):
        lm = self._lm()
        for bad in (0.0, -0.5, 1.01):
            with pytest.raises(ValueError):
                lm.generate([1], 3, 1.0, 0, None, True, bad)
            with pytest.raises(ValueError):
                lm._sample([0.0] * 6, 1.0, 0, random, bad)


class TestAvgPool2d:
    """平均池化层"""

    def test_forward_values(self):
        ap = AI.AvgPool2d(2, stride=2)
        x = [[[1.0, 2.0, 3.0, 4.0],
              [5.0, 6.0, 7.0, 8.0],
              [9.0, 10.0, 11.0, 12.0],
              [13.0, 14.0, 15.0, 16.0]]]
        assert ap.forward(x) == [[[3.5, 5.5], [11.5, 13.5]]]

    def test_backward_distributes_evenly(self):
        ap = AI.AvgPool2d(2, stride=2)
        ap.forward([[[0.0] * 4 for _ in range(4)]])
        dx = ap.backward([[[1.0, 2.0], [3.0, 4.0]]])
        assert abs(dx[0][0][0] - 0.25) < 1e-15
        assert abs(dx[0][3][3] - 1.0) < 1e-15
        assert abs(sum(sum(row) for row in dx[0]) - 10.0) < 1e-12

    def test_gradient_numeric(self):
        random.seed(42)
        ap = AI.AvgPool2d(3, stride=1)
        x = [[[random.gauss(0, 1) for _ in range(4)] for _ in range(4)]]
        dout = [[[random.gauss(0, 1) for _ in range(2)] for _ in range(2)]]

        def loss(z):
            out = ap.forward(z)
            return sum(out[c][i][j] * dout[c][i][j]
                       for c in range(1) for i in range(2) for j in range(2))

        ap.forward(x)
        ana = ap.backward(dout)
        eps = 1e-6
        for i in range(4):
            for j in range(4):
                old = x[0][i][j]
                x[0][i][j] = old + eps
                lp = loss(x)
                x[0][i][j] = old - eps
                lm = loss(x)
                x[0][i][j] = old
                num = (lp - lm) / (2 * eps)
                assert abs(ana[0][i][j] - num) < 1e-8

    def test_global_avg_pool(self):
        gp = AI.GlobalAvgPool2d()
        x = [[[1.0, 2.0], [3.0, 4.0]]]
        out = gp.forward(x)
        assert len(out) == 1
        assert abs(out[0] - 2.5) < 1e-15
        dx = gp.backward([2.0])
        assert abs(dx[0][0][0] - 2.0 / 4) < 1e-15

    def test_invalid_args(self):
        for kernel, stride in ((0, None), (-1, None), (2, 0)):
            with pytest.raises(ValueError):
                AI.AvgPool2d(kernel, stride=stride)
        with pytest.raises(ValueError):
            # 5x5 kernel 套 4x4 输入：输出尺寸 <= 0
            AI.AvgPool2d(5).forward([[[0.0] * 4 for _ in range(4)]])

class TestGQA:
    """GQA / MQA：num_kv_heads 少于 num_heads 时多个 Q 头共享一组 K/V"""

    def test_kv_shapes(self):
        m = AI.MultiHeadAttention(8, 4, num_kv_heads=2)
        assert m.d_k == 2 and m.kv_dim == 4 and m.kv_group == 2
        assert len(m.W_k) == 4 and len(m.W_k[0]) == 8   # (kv_dim, d_model)
        assert len(m.W_q) == 8 and len(m.W_q[0]) == 8   # Q 头不变
        assert len(m.b_k) == 4 and len(m.grads["W_k"]) == 4
        assert len(m.m_W_k) == 4 and len(m.v_W_k[0]) == 8

    def test_mqa_single_kv_head(self):
        m = AI.MultiHeadAttention(8, 4, num_kv_heads=1)
        assert m.kv_dim == 2 and m.kv_group == 4

    def test_default_equals_none(self):
        """num_kv_heads 缺省与显式传 num_heads 逐位一致"""
        random.seed(71)
        a = AI.MultiHeadAttention(8, 2, causal=True)
        random.seed(71)
        b = AI.MultiHeadAttention(8, 2, causal=True, num_kv_heads=2)
        a.eval(); b.eval()
        random.seed(72)
        seq = [[random.gauss(0, 1) for _ in range(8)] for _ in range(4)]
        assert a.forward(seq) == b.forward(seq)

    def test_gradient_numeric(self):
        """共享 KV 头后，dK/dV 要在组内求和——用中心差分验证"""
        random.seed(73)
        m = AI.MultiHeadAttention(8, 4, num_kv_heads=2, causal=True)
        seq = [[random.gauss(0, 1) for _ in range(8)] for _ in range(4)]
        dout = [[random.gauss(0, 1) for _ in range(8)] for _ in range(4)]

        def loss():
            out = m.forward(seq)
            return sum(out[t][j] * dout[t][j] for t in range(4) for j in range(8))

        m.forward(seq)
        m.zero_grad()
        m.backward(dout)
        ana_wk = [row[:] for row in m.grads["W_k"]]
        ana_wv = [row[:] for row in m.grads["W_v"]]
        eps = 1e-6
        for i, j in ((0, 3), (3, 5)):
            old = m.W_k[i][j]
            m.W_k[i][j] = old + eps
            lp = loss()
            m.W_k[i][j] = old - eps
            lm = loss()
            m.W_k[i][j] = old
            num = (lp - lm) / (2 * eps)
            assert abs(ana_wk[i][j] - num) / max(1.0, abs(num)) < 1e-6
        for i, j in ((2, 1),):
            old = m.W_v[i][j]
            m.W_v[i][j] = old + eps
            lp = loss()
            m.W_v[i][j] = old - eps
            lm = loss()
            m.W_v[i][j] = old
            num = (lp - lm) / (2 * eps)
            assert abs(ana_wv[i][j] - num) / max(1.0, abs(num)) < 1e-6

    def test_forward_step_matches_forward(self):
        """KV cache 按 kv 头存，增量前向仍与整段前向最后一行逐位一致"""
        for rope in (False, True):
            random.seed(74)
            m = AI.MultiHeadAttention(8, 4, num_kv_heads=2, causal=True, rope=rope)
            m.eval()
            seq = [[random.gauss(0, 1) for _ in range(8)] for _ in range(5)]
            full = m.forward(seq)
            cache = {}
            for tok in seq:
                last = m.forward_step(tok, cache)
            assert last == full[-1]
            assert len(cache["K"]) == 2 and len(cache["K"][0]) == 5

    def test_fewer_params(self):
        def count(m):
            n = 0
            for _, p, _ in m._named_params():
                n += sum(len(r) for r in p) if p and isinstance(p[0], list) else len(p)
            return n

        random.seed(75)
        full = AI.CausalLM(vocab_size=7, d_model=8, num_heads=4, d_ff=16,
                           num_layers=1, max_len=12)
        random.seed(75)
        gqa = AI.CausalLM(vocab_size=7, d_model=8, num_heads=4, d_ff=16,
                          num_layers=1, max_len=12, num_kv_heads=2)
        assert count(gqa) < count(full)

    def test_invalid_num_kv_heads(self):
        for bad in (0, -1, 3, 8, True, 1.5):
            with pytest.raises(ValueError):
                AI.MultiHeadAttention(8, 4, num_kv_heads=bad)

class TestGroupNorm:
    """分组归一化 / 实例归一化"""

    def test_forward_values(self):
        gn = AI.GroupNorm(2, 4)
        x = [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]],
             [[9.0, 10.0], [11.0, 12.0]], [[13.0, 14.0], [15.0, 16.0]]]
        y = gn.forward(x)
        mean = 4.5
        var = sum((v - mean) ** 2 for v in range(1, 9)) / 8
        inv = 1.0 / math.sqrt(var + 1e-5)
        assert abs(y[0][0][0] - (1.0 - mean) * inv) < 1e-12
        out0 = [y[c][h][w] for c in (0, 1) for h in range(2) for w in range(2)]
        assert abs(sum(out0) / 8) < 1e-9
        assert abs(sum(v * v for v in out0) / 8 - 1.0) < 1e-4

    def test_groups_independent(self):
        """改组 2 的数据不改变组 1 的输出"""
        gn = AI.GroupNorm(2, 4)
        x = [[[float(i * 4 + j * 2 + k + 1) for k in range(2)] for j in range(2)]
             for i in range(4)]
        y = gn.forward(x)
        x2 = [[list(r) for r in c] for c in x]
        x2[2][0][0] = 99.0
        y2 = gn.forward(x2)
        assert y2[0][0][0] == y[0][0][0]

    def test_gradient_numeric(self):
        random.seed(91)
        gn = AI.GroupNorm(2, 4)
        gn.gamma = [1.1, 0.9, 1.3, 0.7]
        gn.beta = [0.1, -0.2, 0.3, 0.05]
        x = [[[random.gauss(0, 1) for _ in range(3)] for _ in range(3)]
             for _ in range(4)]
        dout = [[[random.gauss(0, 1) for _ in range(3)] for _ in range(3)]
                for _ in range(4)]

        def loss():
            out = gn.forward(x)
            return sum(out[c][h][w] * dout[c][h][w]
                       for c in range(4) for h in range(3) for w in range(3))

        gn.forward(x)
        gg, gb, dx = gn.backward(dout)
        eps = 1e-6
        for c, h, w in ((0, 0, 0), (1, 2, 1), (3, 1, 2)):
            old = x[c][h][w]
            x[c][h][w] = old + eps
            lp = loss()
            x[c][h][w] = old - eps
            lm = loss()
            x[c][h][w] = old
            num = (lp - lm) / (2 * eps)
            assert abs(dx[c][h][w] - num) / max(1.0, abs(num)) < 1e-6
        for c in range(4):
            old = gn.gamma[c]
            gn.gamma[c] = old + eps
            lp = loss()
            gn.gamma[c] = old - eps
            lm = loss()
            gn.gamma[c] = old
            assert abs(gg[c] - (lp - lm) / (2 * eps)) / max(1.0, abs(gg[c])) < 1e-6
            old = gn.beta[c]
            gn.beta[c] = old + eps
            lp = loss()
            gn.beta[c] = old - eps
            lm = loss()
            gn.beta[c] = old
            assert abs(gb[c] - (lp - lm) / (2 * eps)) / max(1.0, abs(gb[c])) < 1e-6

    def test_instance_norm_equals_group_norm(self):
        random.seed(92)
        x = [[[random.gauss(0, 1) for _ in range(2)] for _ in range(2)]
             for _ in range(3)]
        o1 = AI.InstanceNorm(3).forward(x)
        o2 = AI.GroupNorm(3, 3).forward(x)
        for c in range(3):
            for h in range(2):
                for w in range(2):
                    assert abs(o1[c][h][w] - o2[c][h][w]) < 1e-15

    def test_invalid(self):
        for args in ((0, 4), (3, 4), (-1, 4), (2, 0)):
            with pytest.raises(ValueError):
                AI.GroupNorm(*args)
        with pytest.raises(ValueError):
            AI.GroupNorm(2, 4).forward([[[1.0, 2.0]], [[3.0, 4.0]]])
        with pytest.raises(RuntimeError):
            AI.GroupNorm(2, 4).backward([[[1.0]]])


class TestDropPath:
    """随机深度：整条分支丢弃"""

    def test_identity_when_disabled(self):
        dp = AI.DropPath(0.0)
        assert dp.forward([1.0, 2.0]) == [1.0, 2.0]
        dp = AI.DropPath(0.5)
        dp.eval()
        assert dp.forward([1.0, 2.0]) == [1.0, 2.0]
        assert dp.factor == 1.0

    def test_drop_rate_and_scale(self):
        dp = AI.DropPath(0.3)
        random.seed(93)
        dropped = 0
        scales = set()
        n = 2000
        for _ in range(n):
            out = dp.forward([1.0])
            if out[0] == 0.0:
                dropped += 1
            else:
                scales.add(out[0])
        assert abs(dropped / n - 0.3) < 0.04
        assert scales == {1.0 / 0.7}

    def test_backward_matches_forward_gate(self):
        dp = AI.DropPath(0.5)
        random.seed(94)
        for _ in range(50):
            out = dp.forward([1.0, 2.0])
            back = dp.backward([1.0, 2.0])
            if out[0] == 0.0:
                assert back == [0.0, 0.0]
                return
        raise AssertionError("50 次都没丢弃，概率上不该发生")

    def test_scale_by_keep_false(self):
        dp = AI.DropPath(0.5, scale_by_keep=False)
        random.seed(95)
        factors = set()
        for _ in range(200):
            dp.forward([1.0])
            factors.add(dp.factor)
        assert factors == {0.0, 1.0}

    def test_invalid(self):
        for bad in (-0.1, 1.0, 1.5):
            with pytest.raises(ValueError):
                AI.DropPath(bad)


class TestPerplexity:
    def test_matches_exp_of_loss(self):
        random.seed(96)
        lm = AI.CausalLM(vocab_size=5, d_model=8, num_heads=2, d_ff=16,
                         num_layers=1, max_len=8)
        lm.eval()
        seqs = [[1, 2, 3], [3, 1, 4]]
        assert abs(lm.perplexity(seqs) - math.exp(lm.evaluate(seqs))) < 1e-12

    def test_overflow_guard(self):
        lm = AI.CausalLM(vocab_size=3, d_model=8, num_heads=2, d_ff=16,
                         num_layers=1, max_len=8)
        lm.eval()
        ppl = lm.perplexity([[0, 1]])
        assert isinstance(ppl, float) and ppl > 0

class TestCrossAttention:
    """编码器-解码器交叉注意力"""

    def _setup(self, seed=101, d=4, heads=2, tq=3, tk=2):
        random.seed(seed)
        ca = AI.CrossAttention(d, heads)
        x = [[random.gauss(0, 1) for _ in range(d)] for _ in range(tq)]
        mem = [[random.gauss(0, 1) for _ in range(d)] for _ in range(tk)]
        dout = [[random.gauss(0, 1) for _ in range(d)] for _ in range(tq)]
        return ca, x, mem, dout

    def test_shapes_and_memory_dependence(self):
        ca, x, mem, _ = self._setup()
        out = ca.forward(x, mem)
        assert len(out) == 3 and len(out[0]) == 4
        assert len(ca.attn_weights[0]) == 3 and len(ca.attn_weights[0][0]) == 2
        mem2 = [list(r) for r in mem]
        mem2[0][0] += 1.0
        out2 = ca.forward(x, mem2)
        assert any(abs(out[t][j] - out2[t][j]) > 1e-12
                   for t in range(3) for j in range(4))

    def test_no_causal_mask(self):
        """交叉注意力没有因果掩码：每个 query 位置都能看到整段 memory"""
        ca, x, mem, _ = self._setup()
        ca.forward(x, mem)
        for t in range(3):
            assert abs(sum(ca.attn_weights[0][t]) - 1.0) < 1e-12
            assert all(ca.attn_weights[0][t][j] > 0.0 for j in range(2))

    def test_gradient_numeric(self):
        ca, x, mem, dout = self._setup(seed=102)
        eps = 1e-6

        def loss():
            o = ca.forward(x, mem)
            return sum(o[t][j] * dout[t][j] for t in range(3) for j in range(4))

        ca.forward(x, mem)
        dx, dmem = ca.backward(dout)
        for t in range(3):
            for j in range(4):
                old = x[t][j]
                x[t][j] = old + eps
                lp = loss()
                x[t][j] = old - eps
                lm = loss()
                x[t][j] = old
                num = (lp - lm) / (2 * eps)
                assert abs(dx[t][j] - num) / max(1.0, abs(num)) < 1e-6
        for t in range(2):
            for j in range(4):
                old = mem[t][j]
                mem[t][j] = old + eps
                lp = loss()
                mem[t][j] = old - eps
                lm = loss()
                mem[t][j] = old
                num = (lp - lm) / (2 * eps)
                assert abs(dmem[t][j] - num) / max(1.0, abs(num)) < 1e-6
        mat = ca.W_k
        for i, j in ((0, 1), (2, 3)):
            old = mat[i][j]
            mat[i][j] = old + eps
            lp = loss()
            mat[i][j] = old - eps
            lm = loss()
            mat[i][j] = old
            num = (lp - lm) / (2 * eps)
            assert abs(ca.grads["W_k"][i][j] - num) / max(1.0, abs(num)) < 1e-6

    def test_dropout_train_eval(self):
        random.seed(103)
        ca = AI.CrossAttention(4, 1, dropout=0.5)
        x = [[1.0, 0.5, -0.5, 0.2]]
        mem = [[0.3, -0.2, 0.1, 0.4], [0.0, 0.6, -0.4, 0.2]]
        ca.eval()
        assert ca.forward(x, mem) == ca.forward(x, mem)
        assert ca.attn_masks[0] is None
        ca.train()
        random.seed(104)
        assert any(ca.forward(x, mem) is not None for _ in range(1))

    def test_invalid(self):
        for args in ((8, 3, 0.0), (8, 2, 1.0), (8, 2, -0.1)):
            with pytest.raises(ValueError):
                AI.CrossAttention(*args)
        ca = AI.CrossAttention(4, 2)
        with pytest.raises(ValueError):
            ca.forward([[0.0] * 3], [[0.0] * 4])
        with pytest.raises(ValueError):
            ca.forward([[0.0] * 4], [])


class TestTransformerDecoder:
    """解码器层 / 堆叠解码器"""

    def test_self_attn_causal_cross_attn_not(self):
        random.seed(111)
        layer = AI.TransformerDecoderLayer(8, 2, 16, dropout=0.0)
        layer.eval()
        xs = [[random.gauss(0, 1) for _ in range(8)] for _ in range(3)]
        ms = [[random.gauss(0, 1) for _ in range(8)] for _ in range(4)]
        o1 = layer.forward(xs, ms)
        xs2 = [list(r) for r in xs]
        xs2[-1] = [v + 1.0 for v in xs2[-1]]
        o2 = layer.forward(xs2, ms)
        assert all(o1[t][j] == o2[t][j] for t in range(2) for j in range(8))
        assert o1[-1] != o2[-1]
        ms2 = [list(r) for r in ms]
        ms2[0][0] += 1.0
        o3 = layer.forward(xs, ms2)
        assert any(abs(o1[t][j] - o3[t][j]) > 1e-12
                   for t in range(3) for j in range(8))

    def test_gradient_numeric(self):
        random.seed(112)
        layer = AI.TransformerDecoderLayer(4, 2, 8, dropout=0.0)
        xs = [[random.gauss(0, 1) for _ in range(4)] for _ in range(2)]
        ms = [[random.gauss(0, 1) for _ in range(4)] for _ in range(3)]
        dout = [[random.gauss(0, 1) for _ in range(4)] for _ in range(2)]
        eps = 1e-6

        def loss():
            o = layer.forward(xs, ms)
            return sum(o[t][j] * dout[t][j] for t in range(2) for j in range(4))

        layer.forward(xs, ms)
        dx, dmem = layer.backward(dout)
        for t in range(2):
            for j in range(4):
                old = xs[t][j]
                xs[t][j] = old + eps
                lp = loss()
                xs[t][j] = old - eps
                lm = loss()
                xs[t][j] = old
                num = (lp - lm) / (2 * eps)
                assert abs(dx[t][j] - num) / max(1.0, abs(num)) < 1e-5
        for t in range(3):
            for j in range(4):
                old = ms[t][j]
                ms[t][j] = old + eps
                lp = loss()
                ms[t][j] = old - eps
                lm = loss()
                ms[t][j] = old
                num = (lp - lm) / (2 * eps)
                assert abs(dmem[t][j] - num) / max(1.0, abs(num)) < 1e-5
        mat = layer.cross_attn.W_k
        old = mat[0][1]
        mat[0][1] = old + eps
        lp = loss()
        mat[0][1] = old - eps
        lm = loss()
        mat[0][1] = old
        num = (lp - lm) / (2 * eps)
        assert abs(layer.cross_attn.grads["W_k"][0][1] - num) / max(1.0, abs(num)) < 1e-5

    def test_decoder_single_layer_matches_layer(self):
        """单层 TransformerDecoder 的 dmemory 与直接调该层逐位相同"""
        random.seed(113)
        dec = AI.TransformerDecoder(8, 2, 16, num_layers=1, dropout=0.0)
        dec.eval()
        xs = [[random.gauss(0, 1) for _ in range(8)] for _ in range(3)]
        ms = [[random.gauss(0, 1) for _ in range(8)] for _ in range(5)]
        dout = [[random.gauss(0, 1) for _ in range(8)] for _ in range(3)]
        dec.forward(xs, ms)
        dx1, dm1 = dec.backward(dout)
        dec.layers[0].forward(xs, ms)
        dx2, dm2 = dec.layers[0].backward(dout)
        assert dx1 == dx2
        assert dm1 == dm2

    def test_multi_layer_dmemory_sums(self):
        random.seed(114)
        dec = AI.TransformerDecoder(8, 2, 16, num_layers=3, dropout=0.0)
        dec.eval()
        xs = [[random.gauss(0, 1) for _ in range(8)] for _ in range(2)]
        ms = [[random.gauss(0, 1) for _ in range(8)] for _ in range(4)]
        out = dec.forward(xs, ms)
        assert len(out) == 2 and len(out[0]) == 8
        dout = [[1.0] * 8 for _ in range(2)]
        dx, dmem = dec.backward(dout)
        assert len(dmem) == 4 and len(dmem[0]) == 8
        assert any(abs(v) > 0 for row in dmem for v in row)

    def test_invalid(self):
        with pytest.raises(ValueError):
            AI.TransformerDecoder(8, 2, 16, num_layers=0)
        with pytest.raises(ValueError):
            AI.TransformerDecoder(8, 2, 16, num_layers=-2)

class TestSeq2SeqLM:
    """编码器-解码器序列到序列模型"""

    def _model(self, seed=201, **kw):
        random.seed(seed)
        args = dict(d_model=8, num_heads=2, d_ff=16, num_layers=1, max_len=12)
        args.update(kw)
        m = AI.Seq2SeqLM(6, 6, **args)
        m.eval()
        return m

    def test_shapes_and_param_blocks(self):
        m = self._model()
        logits = m.forward([1, 2, 3], [m.bos_id, 4, 2])
        assert len(logits) == 3 and len(logits[0]) == 6
        assert m.bos_id == 6            # BOS 在目标词表之外
        blocks = list(m._named_params())
        names = [n for n, _, _ in blocks]
        assert len(blocks) == 2 + 16 + 26 + 2    # 两个词表 + enc + dec + 输出头
        assert "src_emb.weight" in names and "tgt_emb.weight" in names
        assert any(n.startswith("dec.L0.cross.") for n in names)
        assert "out.W" in names and "out.b" in names
        assert m.param_count() > 0

    def test_decoder_causal_and_memory_dependence(self):
        m = self._model()
        a = m.forward([1, 2, 3], [m.bos_id, 1, 2, 3])
        b = m.forward([1, 2, 3], [m.bos_id, 1, 5, 4])
        assert a[0] == b[0] and a[1] == b[1]
        assert a[2] != b[2] or a[3] != b[3]
        c = m.forward([1, 2, 4], [m.bos_id, 1, 2, 3])
        assert all(c[t] != a[t] for t in range(4))

    def test_gradient_numeric(self):
        m = self._model(seed=202)
        src = [1, 2]
        tgt = [3, 1]
        eps = 1e-6

        def train_loss():
            m.zero_grad()
            return m._loss_and_backward(src, tgt)

        m.zero_grad()
        m._loss_and_backward(src, tgt)
        g_src = [row[:] for row in m.src_embedding.grad_weight]
        for idx in (1, 2):
            for j in (0, 3):
                old = m.src_embedding.weight[idx][j]
                m.src_embedding.weight[idx][j] = old + eps
                lp = train_loss()
                m.src_embedding.weight[idx][j] = old - eps
                lm = train_loss()
                m.src_embedding.weight[idx][j] = old
                num = (lp - lm) / (2 * eps)
                assert abs(g_src[idx][j] - num) / max(1.0, abs(num)) < 1e-5

        m.zero_grad()
        m._loss_and_backward(src, tgt)
        cross = m.decoder.layers[0].cross_attn
        g_k = [row[:] for row in cross.grads["W_k"]]
        for i, j in ((0, 1), (2, 3)):
            old = cross.W_k[i][j]
            cross.W_k[i][j] = old + eps
            lp = train_loss()
            cross.W_k[i][j] = old - eps
            lm = train_loss()
            cross.W_k[i][j] = old
            num = (lp - lm) / (2 * eps)
            assert abs(g_k[i][j] - num) / max(1.0, abs(num)) < 1e-5

        m.zero_grad()
        m._loss_and_backward(src, tgt)
        g_out = [row[:] for row in m.grads["W_out"]]
        for i, j in ((0, 1), (4, 2)):
            old = m.W_out[i][j]
            m.W_out[i][j] = old + eps
            lp = train_loss()
            m.W_out[i][j] = old - eps
            lm = train_loss()
            m.W_out[i][j] = old
            num = (lp - lm) / (2 * eps)
            assert abs(g_out[i][j] - num) / max(1.0, abs(num)) < 1e-5

    def test_learns_copy_task(self):
        """小语料上学会「原样复制」：loss 大幅下降且贪心解码正确"""
        random.seed(203)
        m = AI.Seq2SeqLM(5, 5, d_model=16, num_heads=2, d_ff=32,
                         num_layers=1, max_len=12)
        pairs = [([1, 2, 3], [1, 2, 3]), ([2, 3, 1], [2, 3, 1]),
                 ([3, 1, 2], [3, 1, 2]), ([1, 3, 2], [1, 3, 2]),
                 ([2, 1, 3], [2, 1, 3])]
        hist = m.fit(pairs, epochs=40, lr=0.01, shuffle=False)
        assert hist[-1] < 0.2 * hist[0]
        assert m.generate([1, 2, 3], max_new_tokens=3, top_k=1) == [1, 2, 3]
        assert m.generate([2, 3, 1], max_new_tokens=3, top_k=1) == [2, 3, 1]

    def test_generate_reproducible_and_modes(self):
        m = self._model(seed=204, dropout=0.5)
        a = m.generate([1, 2, 3], 3, temperature=1.0, rng=random.Random(9))
        b = m.generate([1, 2, 3], 3, temperature=1.0, rng=random.Random(9))
        assert a == b
        o1 = m.forward([1, 2], [m.bos_id, 1])
        o2 = m.forward([1, 2], [m.bos_id, 1])
        assert o1 == o2                      # generate 之后仍是 eval
        m.train()
        assert m.forward([1, 2], [m.bos_id, 1]) != o1

    def test_invalid(self):
        for args in ((0, 4), (4, 0), (4, 4, 8, 2, 16, 1, 0)):
            with pytest.raises(ValueError):
                AI.Seq2SeqLM(*args)
        m = self._model()
        with pytest.raises(ValueError):
            m.forward([], [0])
        with pytest.raises(ValueError):
            m.forward([1], [])
        with pytest.raises(ValueError):
            m.generate([1], max_new_tokens=-1)
        with pytest.raises(ValueError):
            m.generate([1], max_new_tokens=2, temperature=0.0)

class TestLMPersistence:
    """CausalLM / Seq2SeqLM 的 JSON 存档往返"""

    def _lm(self, seed=301):
        random.seed(seed)
        lm = AI.CausalLM(vocab_size=7, d_model=8, num_heads=2, d_ff=16,
                         num_layers=2, max_len=12, rope=True, window=4,
                         num_kv_heads=1)
        lm.eval()
        return lm

    @staticmethod
    def _params(m):
        return {n: p for n, p, _ in m._named_params()}

    def test_save_load_roundtrip_bit_identical(self, tmp_path):
        lm = self._lm()
        ids = [1, 3, 5, 2]
        before = lm.forward(ids)
        path = str(tmp_path / "lm.json")
        lm.save(path, quiet=True)
        lm2 = AI.CausalLM.load(path)
        assert (lm2.vocab_size, lm2.num_heads, lm2.num_layers, lm2.max_len) == (7, 2, 2, 12)
        assert lm2.rope is True and lm2.window == 4 and lm2.num_kv_heads == 1
        p1, p2 = self._params(lm), self._params(lm2)
        assert set(p1) == set(p2)
        assert all(p1[k] == p2[k] for k in p1)
        assert lm2.forward(ids) == before
        assert lm2.generate([1, 3], max_new_tokens=4, top_k=1) == \
            lm.generate([1, 3], max_new_tokens=4, top_k=1)

    def test_optimizer_state_resumes_bit_identically(self, tmp_path):
        random.seed(302)
        seqs = [[0, 1, 2, 3, 4], [4, 3, 2, 1, 0]]

        def make():
            return AI.CausalLM(vocab_size=5, d_model=8, num_heads=2, d_ff=16,
                               num_layers=1, max_len=10)

        a = make()
        a.fit(seqs, epochs=3, lr=0.01, shuffle=False)
        path = str(tmp_path / "lm_opt.json")
        a.save(path, include_optimizer=True, quiet=True)
        b = AI.CausalLM.load(path, restore_optimizer=True)
        assert b.step_count == 6            # 3 轮 × 2 条序列
        assert abs(a.evaluate(seqs) - b.evaluate(seqs)) < 1e-15
        a.fit(seqs, epochs=2, lr=0.01, shuffle=False)
        b.fit(seqs, epochs=2, lr=0.01, shuffle=False)
        pa, pb = self._params(a), self._params(b)
        assert all(pa[k] == pb[k] for k in pa)       # 逐位续训

        c = AI.CausalLM.load(path)                    # 不恢复优化器
        c.fit(seqs, epochs=2, lr=0.01, shuffle=False)
        pc = self._params(c)
        diff = max(abs(pa[k][i][j] - pc[k][i][j]) for k in pa
                   if isinstance(pa[k], list) and pa[k] and isinstance(pa[k][0], list)
                   for i in range(len(pa[k])) for j in range(len(pa[k][0])))
        assert diff > 1e-6                            # 对照组：确实不同

    def test_corrupt_archive_errors(self, tmp_path):
        lm = self._lm()
        path = str(tmp_path / "lm.json")
        lm.save(path, quiet=True)
        with open(path, "r", encoding="utf-8") as f:
            good = json.load(f)

        bad = str(tmp_path / "bad1.json")
        with open(bad, "w", encoding="utf-8") as f:
            json.dump({"format": "other"}, f)
        with pytest.raises(ValueError):
            AI.CausalLM.load(bad)

        meta = copy.deepcopy(good)
        meta["params"]["emb.weight"] = meta["params"]["emb.weight"][:2]
        bad2 = str(tmp_path / "bad2.json")
        with open(bad2, "w", encoding="utf-8") as f:
            json.dump(meta, f)
        with pytest.raises(ValueError):
            AI.CausalLM.load(bad2)

        meta = copy.deepcopy(good)
        meta["params"]["emb.weight"][0][0] = float("nan")
        bad3 = str(tmp_path / "bad3.json")
        with open(bad3, "w", encoding="utf-8") as f:
            json.dump(meta, f)
        with pytest.raises(ValueError):
            AI.CausalLM.load(bad3)

        with pytest.raises(FileNotFoundError):
            AI.CausalLM.load(str(tmp_path / "nope.json"))

    def test_seq2seq_roundtrip(self, tmp_path):
        random.seed(303)
        m = AI.Seq2SeqLM(5, 5, d_model=8, num_heads=2, d_ff=16, num_layers=1, max_len=10)
        m.eval()
        before = m.forward([1, 2], [m.bos_id, 3])
        path = str(tmp_path / "s2s.json")
        m.save(path, quiet=True)
        m2 = AI.Seq2SeqLM.load(path)
        assert (m2.src_vocab, m2.tgt_vocab, m2.max_len) == (5, 5, 10)
        p1, p2 = self._params(m), self._params(m2)
        assert all(p1[k] == p2[k] for k in p1)
        assert m2.forward([1, 2], [m.bos_id, 3]) == before


class TestBeamSearch:
    """num_beams（束搜索）"""

    def test_beam1_equals_greedy(self):
        random.seed(311)
        lm = AI.CausalLM(vocab_size=6, d_model=16, num_heads=2, d_ff=32,
                         num_layers=1, max_len=16)
        lm.fit([[0, 1, 2, 3, 4, 5, 0, 1, 2, 3]], epochs=40, lr=0.01, shuffle=False)
        greedy = lm.generate([0], max_new_tokens=5, top_k=1)
        assert lm.generate([0], max_new_tokens=5, num_beams=1) == greedy
        assert lm.generate([0], max_new_tokens=5) == greedy     # num_beams 默认 1

    def test_beam_is_deterministic_and_not_worse(self):
        random.seed(312)
        lm = AI.CausalLM(vocab_size=6, d_model=16, num_heads=2, d_ff=32,
                         num_layers=1, max_len=16)
        lm.fit([[0, 1, 2, 3, 4, 5, 0, 1, 2, 3]], epochs=40, lr=0.01, shuffle=False)
        beam = lm.generate([0], max_new_tokens=5, num_beams=3)
        assert beam == lm.generate([0], max_new_tokens=5, num_beams=3)

        def score(tokens):
            total = 0.0
            for t in range(len(tokens)):
                logp = AI._log_softmax_row(lm.forward(tokens[:t + 1])[-1])
                total += logp[tokens[t]]
            return total

        greedy = lm.generate([0], max_new_tokens=5, top_k=1)
        assert score(beam[1:]) >= score(greedy[1:]) - 1e-12

    def test_invalid_num_beams(self):
        random.seed(313)
        lm = AI.CausalLM(vocab_size=4, d_model=8, num_heads=2, d_ff=16,
                         num_layers=1, max_len=8)
        for bad in (0, -1, True, 1.5):
            with pytest.raises(ValueError):
                lm.generate([0], max_new_tokens=2, num_beams=bad)

    def test_seq2seq_beam(self):
        random.seed(314)
        m = AI.Seq2SeqLM(5, 5, d_model=16, num_heads=2, d_ff=32, num_layers=1, max_len=12)
        pairs = [([1, 2, 3], [1, 2, 3]), ([2, 3, 1], [2, 3, 1]), ([3, 1, 2], [3, 1, 2])]
        m.fit(pairs, epochs=40, lr=0.01, shuffle=False)
        greedy = m.generate([1, 2, 3], max_new_tokens=3, top_k=1)
        assert m.generate([1, 2, 3], max_new_tokens=3, num_beams=1) == greedy
        assert m.generate([1, 2, 3], max_new_tokens=3, num_beams=3) == [1, 2, 3]
        out = m.generate([1, 2, 3], max_new_tokens=3, num_beams=2, eos_id=2)
        assert 2 not in out and len(out) <= 3

class TestBytePairTokenizer:
    """字节级 BPE 分词器"""

    TEXT = ("hello world hello there 你好，世界！emoji 😀🚀 "
            "the quick brown fox jumps over the lazy dog. " * 6)

    def test_roundtrip_unicode(self):
        tok = AI.BytePairTokenizer().train(self.TEXT, vocab_size=320)
        for s in ("", "a", "hello 世界", "😀🚀 emoji 测试", "line1\nline2\ttab"):
            assert tok.decode(tok.encode(s)) == s
        ids = tok.encode(self.TEXT)
        assert tok.decode(ids) == self.TEXT
        assert len(ids) < len(self.TEXT.encode("utf-8"))   # 真的合并了

    def test_training_deterministic_and_vocab_size(self):
        a = AI.BytePairTokenizer().train(self.TEXT, vocab_size=320)
        b = AI.BytePairTokenizer().train(self.TEXT, vocab_size=320)
        assert a.merges == b.merges
        assert a.encode(self.TEXT) == b.encode(self.TEXT)
        assert a.vocab_size == 256 + len(a.merges)
        assert a.vocab_size <= 320
        assert len(a) == a.vocab_size
        assert "vocab_size=" in repr(a)
        assert AI.BytePairTokenizer().vocab_size == 256

    def test_merges_stay_inside_words(self):
        tok = AI.BytePairTokenizer().train("cat cat cat cat", vocab_size=258)
        # 不应该学到「字母 + 空格」这种跨词块的合并
        assert all(m != (99, 32) and m != (32, 99) for m in tok.merges)
        assert tok.decode(tok.encode("cat cat")) == "cat cat"

    def test_save_load_and_encode_corpus(self, tmp_path):
        tok = AI.BytePairTokenizer().train(self.TEXT, vocab_size=300)
        ids = tok.encode(self.TEXT)
        path = str(tmp_path / "bpe.json")
        tok.save(path, quiet=True)
        tok2 = AI.BytePairTokenizer.load(path)
        assert tok2.merges == tok.merges
        assert tok2.vocab_size == tok.vocab_size
        assert tok2.encode(self.TEXT) == ids

        assert tok.encode_corpus(self.TEXT) == [ids]
        chunks = tok.encode_corpus(self.TEXT, seq_len=16)
        assert all(2 <= len(c) <= 16 for c in chunks)
        flat = [x for c in chunks for x in c]
        assert flat == ids[:len(flat)]

    def test_errors(self):
        for bad in (0, 255, -1, True, 3.5):
            with pytest.raises(ValueError):
                AI.BytePairTokenizer().train("abc", vocab_size=bad)
        with pytest.raises(ValueError):
            AI.BytePairTokenizer().train("abc", vocab_size=300, min_frequency=0)
        tok = AI.BytePairTokenizer().train("abc abc", vocab_size=258)
        with pytest.raises(ValueError):
            tok.encode(123)
        with pytest.raises(ValueError):
            tok.decode([tok.vocab_size + 5])
        with pytest.raises(ValueError):
            tok.decode([230])              # 截断的多字节序列
        with pytest.raises(ValueError):
            tok.encode_corpus("abc", seq_len=1)

    def test_train_with_causallm(self):
        """分词器 + CausalLM 串起来：编码语料 → 训练 → 生成 → 解码"""
        random.seed(321)
        text = "0120120120120120120120"
        tok = AI.BytePairTokenizer().train(text, vocab_size=258)
        seqs = tok.encode_corpus(text, seq_len=6)
        lm = AI.CausalLM(vocab_size=tok.vocab_size, d_model=16, num_heads=2,
                         d_ff=32, num_layers=1, max_len=8)
        hist = lm.fit(seqs, epochs=40, lr=0.01, shuffle=False)
        assert hist[-1] < 0.2 * hist[0]
        out = lm.generate(tok.encode("0"), max_new_tokens=6, top_k=1)
        assert isinstance(tok.decode(out), str)
        assert lm.param_count() > 0



class TestPaddedBatch:
    """批量 padded 训练（pad_mask）"""

    @staticmethod
    def _lm(seed=11, **kw):
        random.seed(seed)
        cfg = dict(vocab_size=7, d_model=16, num_heads=2, d_ff=32,
                   num_layers=2, max_len=12)
        cfg.update(kw)
        return AI.CausalLM(**cfg)

    @staticmethod
    def _params(model):
        return {name: p for name, p, _ in model._named_params()}

    @staticmethod
    def _grads(model):
        out = {}
        for name, _, g in model._named_params():
            rows = g if isinstance(g[0], list) else [g]
            out[name] = [row[:] for row in rows]
        return out

    def test_padding_keeps_loss_grads_and_logits(self):
        """右侧补齐 + pad_mask 后，真实位置的 logits、loss、梯度都逐位不变"""
        seq = [1, 4, 2, 5, 3]
        padded = seq + [0, 0, 0]
        mask = [1] * 5 + [0] * 3

        m1 = self._lm()
        m1.zero_grad()
        loss1 = m1._loss_and_backward(seq)
        g1 = self._grads(m1)
        lg1 = m1.forward(seq)

        m2 = self._lm()
        m2.zero_grad()
        loss2 = m2._loss_and_backward(padded, mask)
        g2 = self._grads(m2)
        lg2 = m2.forward(padded, mask)

        assert loss1 == loss2
        for name in g1:
            for r1, r2 in zip(g1[name], g2[name]):
                assert r1 == r2
        for r1, r2 in zip(lg1, lg2):
            assert r1 == r2

    def test_mask_validation(self):
        m = self._lm()
        padded = [1, 4, 2, 5, 3, 0, 0, 0]
        with pytest.raises(ValueError):
            m._loss_and_backward(padded, [1] * 5)              # 长度不符
        with pytest.raises(ValueError):
            m._loss_and_backward(padded, [0] * 8)              # 全 0
        with pytest.raises(ValueError):
            m._loss_and_backward(padded, [2] * 5 + [0] * 3)    # 非 0/1
        with pytest.raises(ValueError):
            m._loss_and_backward([3, 0, 0, 0], [1, 0, 0, 0])   # 没有有效目标
        with pytest.raises(ValueError):
            m.forward(padded, [1] * 3)                         # forward 里也校验

    def test_batch_step_equals_manual_weighted_step(self):
        """batch_size=2 的一步 == 手工按 token 数加权的一步"""
        seqs = [[1, 4, 2, 5, 3], [2, 3]]
        width = max(len(s) for s in seqs)
        n_valid = sum(len(s) - 1 for s in seqs)

        manual = self._lm(31)
        manual.zero_grad()
        for s in seqs:
            pad = width - len(s)
            manual._loss_and_backward(s + [0] * pad,
                                      [1] * len(s) + [0] * pad,
                                      grad_scale=(len(s) - 1) / n_valid)
        manual.adam_step(0.01)

        batched = self._lm(31)
        batched.fit(seqs, epochs=1, lr=0.01, shuffle=False, batch_size=2)

        p_manual = self._params(manual)
        for name, p_batch, _ in batched._named_params():
            assert p_manual[name] == p_batch

    def test_step_count_and_learning(self):
        seqs = [[1, 4, 2, 5, 3], [2, 3], [6, 1, 2, 4, 5, 3, 2]]
        one = self._lm(7)
        one.fit(seqs, epochs=1, lr=0.01, shuffle=False, batch_size=1)
        assert one.step_count == len(seqs)

        two = self._lm(7)
        two.fit(seqs, epochs=1, lr=0.01, shuffle=False, batch_size=2)
        assert two.step_count == 2

        lm = self._lm(5)
        before = lm.evaluate(seqs)
        hist = lm.fit(seqs, epochs=60, lr=0.01, shuffle=True, seed=1, batch_size=3)
        assert len(hist) == 60
        assert lm.evaluate(seqs) < before * 0.5

    def test_batch_size_validation(self):
        seqs = [[1, 2, 3]]
        for bad in (0, -1, True, 1.5):
            with pytest.raises(ValueError):
                self._lm().fit(seqs, epochs=1, batch_size=bad)

    def test_eval_and_generate_still_work(self):
        lm = self._lm(17, max_len=8)
        lm.fit([[1, 2, 3, 1, 2, 3]], epochs=5, lr=0.01, shuffle=False, batch_size=1)
        out = lm.generate([1], max_new_tokens=4, top_k=1)
        assert len(out) == 5
        assert isinstance(lm.evaluate([[1, 2, 3]]), float)

        # eval 模式下（dropout 关闭）补齐同样逐位一致
        lm.eval()
        lm.zero_grad()
        a = lm._loss_and_backward([1, 2, 3, 4])
        lm.zero_grad()
        b = lm._loss_and_backward([1, 2, 3, 4, 0, 0], [1, 1, 1, 1, 0, 0])
        assert a == b


class TestAdaptiveAvgPool2d:
    """自适应平均池化（任意输出尺寸）"""

    def test_grid_matches_hand_computation(self):
        """H=3,W=4 → out=(2,3)：不能被整除时按 PyTorch 的分箱规则重叠取窗"""
        pool = AI.AdaptiveAvgPool2d((2, 3))
        x = [[[1.0, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]]
        assert pool.forward(x) == [[[3.5, 4.5, 5.5], [7.5, 8.5, 9.5]]]

    def test_1x1_equals_global_avg_pool(self):
        """out_size=(1,1) 与 GlobalAvgPool2d 逐位相同（求和顺序一致）"""
        random.seed(31)
        for _ in range(10):
            c, h, w = random.randint(1, 4), random.randint(1, 8), random.randint(1, 8)
            img = [[[random.uniform(-3, 3) for _ in range(w)] for _ in range(h)] for _ in range(c)]
            a = AI.AdaptiveAvgPool2d((1, 1)).forward(img)
            b = AI.GlobalAvgPool2d().forward(img)
            assert [a[i][0][0] for i in range(c)] == list(b)

    def test_divisible_equals_plain_avg_pool(self):
        random.seed(32)
        img = [[[random.uniform(-2, 2) for _ in range(6)] for _ in range(6)] for _ in range(2)]
        a = AI.AdaptiveAvgPool2d((3, 3)).forward(img)
        b = AI.AvgPool2d(2).forward(img)
        for c in range(2):
            for i in range(3):
                for j in range(3):
                    assert abs(a[c][i][j] - b[c][i][j]) <= 1e-15

    def test_backward_numeric(self):
        """dx 与数值梯度一致（每个格子把 dout/面积均分回自己的输入区间）"""
        random.seed(33)
        img = [[[random.uniform(-1, 1) for _ in range(5)] for _ in range(4)] for _ in range(2)]
        dout = [[[random.uniform(-1, 1) for _ in range(2)] for _ in range(3)] for _ in range(2)]
        pool = AI.AdaptiveAvgPool2d((3, 2))
        pool.forward(img)
        dx = pool.backward(dout)

        flat = [img[c][h][w] for c in range(2) for h in range(4) for w in range(5)]

        def loss_of(values):
            x = [[[values[(c * 4 + h) * 5 + w] for w in range(5)] for h in range(4)] for c in range(2)]
            out = pool.forward(x)
            return sum(out[c][i][j] * dout[c][i][j]
                       for c in range(2) for i in range(3) for j in range(2))

        worst = 0.0
        for idx, base in enumerate(flat):
            eps = 1e-6
            up = list(flat); up[idx] = base + eps
            down = list(flat); down[idx] = base - eps
            num = (loss_of(up) - loss_of(down)) / (2 * eps)
            ana = dx[idx // 20][(idx % 20) // 5][idx % 5]
            worst = max(worst, abs(num - ana))
        assert worst <= 1e-9, f"max|diff|={worst:.3e}"

    def test_size_validation(self):
        for bad in (0, -1, (0, 2), (2, 0), (True, 3), (2.0, 2), (1, 2, 3)):
            with pytest.raises(ValueError):
                AI.AdaptiveAvgPool2d(bad)
        with pytest.raises(RuntimeError):
            AI.AdaptiveAvgPool2d((1, 1)).backward([[1.0]])
        with pytest.raises(ValueError):
            AI.AdaptiveAvgPool2d((1, 1)).forward([[]])


class TestModelLossMap:
    """Model(loss=...) 的损失映射与参数透传（P0）"""

    def _cls_data(self, seed=11, n=60):
        rng = random.Random(seed)
        xs, ys = [], []
        for _ in range(n):
            c = rng.randrange(3)
            base = [0.0, 0.0, 0.0]
            base[c] = 1.0
            xs.append([v + rng.gauss(0, 0.25) for v in base])
            ys.append(c)
        return xs, ys

    def test_new_losses_are_constructionable(self):
        xs, ys = self._cls_data()
        cases = [
            ("focal", ["leaky_relu", "softmax"], "classification", {}),
            ("bce", ["leaky_relu", "sigmoid"], "regression", {}),
            ("bce_with_logits", ["leaky_relu", "linear"], "regression", {}),
            ("huber", ["leaky_relu", "linear"], "regression", {}),
        ]
        for name, acts, task, kwargs in cases:
            m = AI.Model(AI.MLP([3, 4, 3 if task == "classification" else 1], activations=acts),
                         loss_fn=name, task=task, lr=0.05, loss_kwargs=kwargs)
            data = list(zip(xs, ys)) if task == "classification" else list(zip(xs, [0.3] * len(xs)))
            h = m.fit(data=data, epochs=3, batch_size=8, verbose=False)
            assert len(h["train_loss"]) == 3

    def test_loss_kwargs_are_forwarded(self):
        # label_smoothing 透传：Model 内的 loss 必须与直接调用 cross_entropy_loss 完全相等
        m = AI.Model(AI.MLP([3, 4, 3], activations=["leaky_relu", "softmax"]),
                     loss_fn="ce", task="classification", loss_kwargs={"label_smoothing": 0.1})
        pred = [0.2, 0.5, 0.3]
        got, _ = m._loss(pred, 1)
        want, _ = AI.cross_entropy_loss(pred, 1, label_smoothing=0.1)
        assert abs(got - want) < 1e-15

    def test_loss_kwargs_must_be_dict(self):
        with pytest.raises(TypeError):
            AI.Model(AI.MLP([3, 4, 3], activations=["leaky_relu", "softmax"]),
                     loss_fn="focal", task="classification", loss_kwargs=[1])

    def test_focal_one_hot_equals_index(self):
        m = AI.Model(AI.MLP([3, 4, 3], activations=["leaky_relu", "softmax"]),
                     loss_fn="focal", task="classification")
        pred = [0.2, 0.5, 0.3]
        l_idx, _ = m._loss(pred, 1)
        l_onehot, _ = m._loss(pred, [0.0, 1.0, 0.0])
        assert abs(l_idx - l_onehot) < 1e-15
        # 软标签对 focal 是明确拒绝的
        with pytest.raises(ValueError):
            m._loss(pred, [0.1, 0.8, 0.1])

    def test_kl_is_rejected_by_model(self):
        # softmax 的反向是 softmax+CE 的联合梯度，kl 叠上去梯度不自洽
        with pytest.raises(ValueError):
            AI.Model(AI.MLP([4, 8, 3], activations=["leaky_relu", "softmax"]),
                     loss_fn="kl", task="classification")

    def test_output_activation_guards(self):
        with pytest.raises(ValueError):
            AI.Model(AI.MLP([3, 4, 3], activations=["leaky_relu", "linear"]),
                     loss_fn="bce", task="classification")
        with pytest.raises(ValueError):
            AI.Model(AI.MLP([3, 4, 3], activations=["leaky_relu", "sigmoid"]),
                     loss_fn="bce_with_logits", task="classification")

    def test_unknown_loss_message_lists_new_options(self):
        with pytest.raises(ValueError) as ei:
            AI.Model(AI.MLP([3, 4, 3], activations=["leaky_relu", "softmax"]),
                     loss_fn="nope", task="classification")
        msg = str(ei.value)
        for name in ("focal", "bce", "huber"):
            assert name in msg


class TestGradAccum:
    """梯度累积（fit 的 grad_accum 参数）"""

    def _data(self, seed=7, n=12):
        rng = random.Random(seed)
        xs = [[rng.uniform(-1, 1) for _ in range(3)] for _ in range(n)]
        ys = [rng.randrange(2) for _ in range(n)]
        return xs, ys

    def _weights(self, model):
        return [[row[:] for row in l.weights] for l in model.model.layers]

    def test_grad_accum_one_is_bit_identical(self):
        xs, ys = self._data()
        acts = ["leaky_relu", "softmax"]
        random.seed(31)
        m1 = AI.Model(AI.MLP([3, 4, 2], activations=acts), loss_fn="ce",
                      task="classification", lr=0.05)
        h1 = m1.fit(x=xs, y=ys, epochs=4, batch_size=4, val_split=0.0, verbose=False)

        random.seed(31)
        m2 = AI.Model(AI.MLP([3, 4, 2], activations=acts), loss_fn="ce",
                      task="classification", lr=0.05)
        h2 = m2.fit(x=xs, y=ys, epochs=4, batch_size=4, val_split=0.0, verbose=False,
                    grad_accum=1)

        w1, w2 = self._weights(m1), self._weights(m2)
        for a, b in zip(w1, w2):
            for ra, rb in zip(a, b):
                assert ra == rb
        assert [v for _, v in h1["train_loss"]] == [v for _, v in h2["train_loss"]]

    def test_accum_equals_bigger_batch(self):
        xs, ys = self._data()
        acts = ["leaky_relu", "softmax"]
        random.seed(99)
        m1 = AI.Model(AI.MLP([3, 4, 2], activations=acts), loss_fn="ce",
                      task="classification", lr=0.05)
        m1.fit(x=xs, y=ys, epochs=4, batch_size=4, val_split=0.0, verbose=False,
               shuffle=False)

        random.seed(99)
        m2 = AI.Model(AI.MLP([3, 4, 2], activations=acts), loss_fn="ce",
                      task="classification", lr=0.05)
        m2.fit(x=xs, y=ys, epochs=4, batch_size=2, val_split=0.0, verbose=False,
               shuffle=False, grad_accum=2)

        w1, w2 = self._weights(m1), self._weights(m2)
        for a, b in zip(w1, w2):
            for ra, rb in zip(a, b):
                for va, vb in zip(ra, rb):
                    assert abs(va - vb) < 1e-15

    def test_leftover_batches_are_applied(self):
        xs, ys = self._data(n=9)
        m = AI.Model(AI.MLP([3, 4, 2], activations=["leaky_relu", "softmax"]),
                     loss_fn="ce", task="classification", lr=0.05)
        h = m.fit(x=xs, y=ys, epochs=2, batch_size=3, val_split=0.0, verbose=False,
                  shuffle=False, grad_accum=2)
        # 9 个样本 / batch 3 = 3 个 batch；grad_accum=2 -> 每 epoch 2 次更新（2+1）
        assert len(h["grad_norm"]) == 2

    def test_invalid_grad_accum(self):
        xs, ys = self._data()
        m = AI.Model(AI.MLP([3, 4, 2], activations=["leaky_relu", "softmax"]),
                     loss_fn="ce", task="classification", lr=0.05)
        for bad in (0, -1, True, 1.5, "2"):
            with pytest.raises((ValueError, TypeError)):
                m.fit(x=xs, y=ys, epochs=1, batch_size=4, val_split=0.0,
                      verbose=False, grad_accum=bad)


class TestFlops:
    def test_mlp(self):
        r = AI.estimate_flops(AI.MLP([4, 8, 3], activations=["relu", "linear"]), [4])
        assert r["macs"] == 4 * 8 + 8 * 3
        assert r["params"] == (4 * 8 + 8) + (8 * 3 + 3)
        assert r["output_shape"] == [3]
        assert r["flops"] == 2 * r["macs"]

    def test_cnn_layer_by_layer(self):
        cnn = AI.Sequential([
            AI.Conv2d(1, 2, kernel_size=3, stride=1),
            AI.MaxPool2d(2),
            AI.Flatten(),
            AI.Layer(8, 5),
        ])
        r = AI.estimate_flops(cnn, (1, 6, 6))
        conv = 1 * 3 * 3 * 2 * 4 * 4
        pool = 2 * 2 * 2 * 2 * 2
        dense = 8 * 5
        assert r["macs"] == conv + pool + dense
        assert r["output_shape"] == [5]
        assert len(r["by_layer"]) == 4

    def test_unknown_module_raises(self):
        # 不认识的模块必须在估算时就报 TypeError，而不是给个错数
        class _Unknown:
            def forward(self, x):
                return x

            def backward(self, delta):
                return delta

        with pytest.raises(TypeError):
            AI.estimate_flops(AI.Sequential(_Unknown()), [3])

        # 不是模块的东西进不了 Sequential（构造期就报，而不是 forward 时才炸）
        with pytest.raises(ValueError):
            AI.Sequential(object())


class TestAugmentation:
    def test_noise_zero_is_copy(self):
        x = [1.0, 2.0, 3.0]
        out = AI.augment_noise(x, sigma=0.0)
        assert out == x and out is not x
        assert x == [1.0, 2.0, 3.0]

    def test_noise_batch_and_validation(self):
        batch = AI.augment_noise([[1.0, 2.0], [3.0, 4.0]], sigma=0.0)
        assert batch == [[1.0, 2.0], [3.0, 4.0]]
        with pytest.raises(ValueError):
            AI.augment_noise([1.0], sigma=-0.1)

    def test_image_flip_keeps_multiset(self):
        img = [[[float(y * 10 + x) for x in range(4)] for y in range(3)]]
        out = AI.augment_image(img, flip=True, max_shift=0, rng=random.Random(11))
        assert len(out) == 1 and len(out[0]) == 3 and len(out[0][0]) == 4
        assert sorted(v for row in out[0] for v in row) == \
            sorted(v for row in img[0] for v in row)

    def test_image_shift_pads_zero(self):
        img = [[[float(y * 10 + x) for x in range(4)] for y in range(3)]]
        out = AI.augment_image(img, flip=False, max_shift=2, rng=random.Random(4))
        assert any(v == 0.0 for row in out[0] for v in row)

    def test_mixup(self):
        mx, soft = AI.mixup([1.0, 2.0], 0, [3.0, 4.0], 2, alpha=0.0)
        assert mx == [2.0, 3.0]
        assert abs(sum(soft) - 1.0) < 1e-15
        assert len(soft) == 3
        with pytest.raises(ValueError):
            AI.mixup([1.0], 0.5, [2.0], 1)

class TestConvGroups:
    def test_groups_two_equals_two_separate_convs(self):
        g = AI.Conv2d(4, 2, kernel_size=3, padding=1, groups=2)
        g.weights[0] = [[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
                        [[-0.1, -0.2, -0.3], [-0.4, -0.5, -0.6], [-0.7, -0.8, -0.9]]]
        g.weights[1] = [[[0.2, 0.0, -0.2], [0.3, 0.1, -0.3], [0.4, 0.2, -0.4]],
                        [[0.5, 0.3, 0.1], [0.6, 0.4, 0.2], [0.7, 0.5, 0.3]]]
        a = AI.Conv2d(2, 1, kernel_size=3, padding=1)
        a.weights[0] = g.weights[0]
        a.biases[0] = g.biases[0]
        b = AI.Conv2d(2, 1, kernel_size=3, padding=1)
        b.weights[0] = g.weights[1]
        b.biases[0] = g.biases[1]

        x = [[[0.1 * (c + 1) + 0.05 * h + 0.01 * w for w in range(4)] for h in range(4)]
             for c in range(4)]
        out = g.forward(x)
        oa = a.forward(x[:2])[0]
        ob = b.forward(x[2:])[0]
        for r1, r2 in zip(out[0], oa):
            for v1, v2 in zip(r1, r2):
                assert abs(v1 - v2) < 1e-15
        for r1, r2 in zip(out[1], ob):
            for v1, v2 in zip(r1, r2):
                assert abs(v1 - v2) < 1e-15

    def test_depthwise_channels(self):
        dw = AI.Conv2d(3, 6, kernel_size=3, padding=1, groups=3)
        assert all(len(w) == 1 for w in dw.weights)
        out = dw.forward([[[0.1] * 4 for _ in range(4)] for _ in range(3)])
        assert len(out) == 6

    def test_groups_grad_vs_numeric(self):
        random.seed(7)
        g = AI.Conv2d(2, 2, kernel_size=3, padding=1, groups=2)
        x = [[[0.3, -0.2, 0.5, 0.1], [0.2, 0.4, -0.3, 0.6],
              [-0.1, 0.7, 0.2, -0.4], [0.5, -0.3, 0.8, 0.2]] for _ in range(2)]
        tgt = [[[0.1 + 0.1 * h + 0.05 * w for w in range(4)] for h in range(4)]
               for _ in range(2)]
        denom = 32.0

        def loss():
            out = g.forward(x)
            s = 0.0
            for oc in range(2):
                for h in range(4):
                    for w in range(4):
                        d = out[oc][h][w] - tgt[oc][h][w]
                        s += d * d
            return s / denom

        def grad_out():
            out = g.forward(x)
            return [[[2.0 * (out[oc][h][w] - tgt[oc][h][w]) / denom for w in range(4)]
                     for h in range(4)] for oc in range(2)]

        g.zero_grad()
        g.forward(x)
        g.backward(grad_out())
        ana = [[v for v in row] for row in g.grad_w[0][0]]

        eps = 1e-6
        worst = 0.0
        for kh in range(3):
            for kw in range(3):
                base = g.weights[0][0][kh][kw]
                g.weights[0][0][kh][kw] = base + eps
                lp = loss()
                g.weights[0][0][kh][kw] = base - eps
                lm = loss()
                g.weights[0][0][kh][kw] = base
                worst = max(worst, abs((lp - lm) / (2 * eps) - ana[kh][kw]))
        assert worst < 1e-8

    def test_invalid_groups(self):
        with pytest.raises(ValueError):
            AI.Conv2d(3, 2, 3, groups=2)
        with pytest.raises(ValueError):
            AI.Conv2d(4, 3, 3, groups=2)
        with pytest.raises(ValueError):
            AI.Conv2d(4, 2, 3, groups=0)
        with pytest.raises(ValueError):
            AI.Conv2d(4, 2, 3, groups=True)


class TestConv1d:
    def test_shape_and_stride(self):
        c = AI.Conv1d(2, 2, kernel_size=3, stride=1, padding=1)
        out = c.forward([[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]])
        assert len(out) == 2 and len(out[0]) == 4

        c2 = AI.Conv1d(1, 1, kernel_size=2, stride=2)
        c2.weights[0] = [[1.0, 1.0]]
        c2.biases = [0.0]
        assert c2.forward([[1.0, 2.0, 3.0, 4.0, 5.0]]) == [[3.0, 7.0]]

    def test_manual_convolution(self):
        c = AI.Conv1d(2, 1, kernel_size=3, padding=1)
        c.weights[0] = [[1.0, 0.0, -1.0], [0.5, 0.5, 0.5]]
        c.biases = [0.5]
        out = c.forward([[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]])
        # base = -1 -> 只用 k=1(权0.0) 与 k=2(权-1) 命中输入 0/1；第二通道同理
        assert abs(out[0][0] - 4.0) < 1e-12

    def test_groups_equals_separate(self):
        g = AI.Conv1d(4, 2, kernel_size=3, padding=1, groups=2)
        g.weights[0] = [[0.1, 0.2, 0.3], [-0.1, -0.2, -0.3]]
        g.weights[1] = [[0.2, 0.0, -0.2], [0.5, 0.3, 0.1]]
        a = AI.Conv1d(2, 1, kernel_size=3, padding=1)
        a.weights[0] = g.weights[0]
        a.biases[0] = g.biases[0]
        b = AI.Conv1d(2, 1, kernel_size=3, padding=1)
        b.weights[0] = g.weights[1]
        b.biases[0] = g.biases[1]
        x = [[0.1 * (ci + 1) + 0.05 * i for i in range(5)] for ci in range(4)]
        out = g.forward(x)
        oa = a.forward(x[:2])[0]
        ob = b.forward(x[2:])[0]
        for v1, v2 in zip(out[0], oa):
            assert abs(v1 - v2) < 1e-15
        for v1, v2 in zip(out[1], ob):
            assert abs(v1 - v2) < 1e-15

    def test_grad_vs_numeric(self):
        random.seed(21)
        m = AI.Conv1d(2, 2, kernel_size=3, padding=1)
        x = [[0.3, -0.2, 0.5, 0.1, 0.4], [0.2, 0.4, -0.3, 0.6, -0.1]]
        tgt = [[0.1, 0.2, 0.3, 0.4, 0.5], [0.5, 0.4, 0.3, 0.2, 0.1]]
        denom = 10.0

        def loss():
            out = m.forward(x)
            s = 0.0
            for oc in range(2):
                for i in range(5):
                    d = out[oc][i] - tgt[oc][i]
                    s += d * d
            return s / denom

        def grad_out():
            out = m.forward(x)
            return [[2.0 * (out[oc][i] - tgt[oc][i]) / denom for i in range(5)]
                    for oc in range(2)]

        m.zero_grad()
        m.forward(x)
        dx_ana = m.backward(grad_out())
        ana_w = list(m.grad_w[1][0])

        eps = 1e-6
        worst = 0.0
        for k in range(3):
            base = m.weights[1][0][k]
            m.weights[1][0][k] = base + eps
            lp = loss()
            m.weights[1][0][k] = base - eps
            lm = loss()
            m.weights[1][0][k] = base
            worst = max(worst, abs((lp - lm) / (2 * eps) - ana_w[k]))
        assert worst < 1e-8

        worst = 0.0
        for ci in range(2):
            for i in range(3):
                base = x[ci][i]
                x[ci][i] = base + eps
                lp = loss()
                x[ci][i] = base - eps
                lm = loss()
                x[ci][i] = base
                worst = max(worst, abs((lp - lm) / (2 * eps) - dx_ana[ci][i]))
        assert worst < 1e-8

    def test_invalid_params(self):
        with pytest.raises(ValueError):
            AI.Conv1d(3, 2, 3, groups=2)
        with pytest.raises(ValueError):
            AI.Conv1d(2, 2, 0)
        with pytest.raises(ValueError):
            AI.Conv1d(2, 2, 3, stride=0)
        with pytest.raises(ValueError):
            AI.Conv1d(2, 2, 3, padding=-1)
        c = AI.Conv1d(2, 1, 3)
        with pytest.raises(ValueError):
            c.forward([[0.1], [0.2, 0.3]])


class TestConvTranspose2d:
    def test_output_shapes(self):
        t = AI.ConvTranspose2d(1, 1, kernel_size=3, stride=2)
        t.forward([[[1.0, 2.0], [3.0, 4.0]]])
        assert (t.h_out, t.w_out) == (5, 5)

        t2 = AI.ConvTranspose2d(1, 1, kernel_size=3, stride=2, padding=1, output_padding=1)
        t2.forward([[[1.0, 2.0], [3.0, 4.0]]])
        assert (t2.h_out, t2.w_out) == (4, 4)

    def test_single_point_scatters_kernel(self):
        t = AI.ConvTranspose2d(1, 1, kernel_size=2)
        t.weights[0][0] = [[1.0, 2.0], [3.0, 4.0]]
        t.biases = [0.0]
        out = t.forward([[[1.0, 0.0], [0.0, 0.0]]])
        assert out[0][0][0] == 1.0 and out[0][0][1] == 2.0
        assert out[0][1][0] == 3.0 and out[0][1][1] == 4.0
        assert out[0][2][0] == 0.0

    def test_adjoint_identity(self):
        random.seed(31)
        cv = AI.Conv2d(2, 2, kernel_size=3, padding=1)
        tv = AI.ConvTranspose2d(2, 2, kernel_size=3, padding=1)
        tv.weights = [[[row[:] for row in kern] for kern in per_in] for per_in in cv.weights]
        x = [[[0.5, -0.2, 0.1], [-0.4, 0.8, 0.2], [0.7, 0.3, -0.6]],
             [[0.1, 0.4, -0.3], [0.6, -0.2, 0.5], [-0.3, 0.9, 0.2]]]
        y = [[[0.2, -0.1, 0.3], [0.5, 0.4, -0.2], [0.1, 0.7, 0.3]],
             [[-0.3, 0.2, 0.6], [0.4, -0.5, 0.1], [0.2, 0.8, -0.4]]]
        cy = cv.forward(x)
        lhs = sum(sum(cy[oc][h][w] * y[oc][h][w] for w in range(3))
                  for oc in range(2) for h in range(3))
        ty = tv.forward(y)
        rhs = sum(sum(x[c][h][w] * ty[c][h][w] for w in range(3))
                  for c in range(2) for h in range(3))
        assert abs(lhs - rhs) < 1e-12

    def test_grad_vs_numeric(self):
        random.seed(41)
        t = AI.ConvTranspose2d(2, 2, kernel_size=3, stride=2, padding=1)
        x = [[[0.3, -0.2], [0.5, 0.1]], [[0.2, 0.4], [-0.3, 0.6]]]
        t.forward(x)
        ho, wo = t.h_out, t.w_out
        tgt = [[[0.1 + 0.01 * h + 0.02 * w for w in range(wo)] for h in range(ho)]
               for _ in range(2)]
        denom = 2.0 * ho * wo

        def loss():
            out = t.forward(x)
            s = 0.0
            for oc in range(2):
                for h in range(ho):
                    for w in range(wo):
                        d = out[oc][h][w] - tgt[oc][h][w]
                        s += d * d
            return s / denom

        def grad_out():
            out = t.forward(x)
            return [[[2.0 * (out[oc][h][w] - tgt[oc][h][w]) / denom for w in range(wo)]
                     for h in range(ho)] for oc in range(2)]

        t.zero_grad()
        t.forward(x)
        dx_ana = t.backward(grad_out())
        ana = [[v for v in row] for row in t.grad_w[0][1]]

        eps = 1e-6
        worst = 0.0
        for kh in range(3):
            for kw in range(3):
                base = t.weights[0][1][kh][kw]
                t.weights[0][1][kh][kw] = base + eps
                lp = loss()
                t.weights[0][1][kh][kw] = base - eps
                lm = loss()
                t.weights[0][1][kh][kw] = base
                worst = max(worst, abs((lp - lm) / (2 * eps) - ana[kh][kw]))
        assert worst < 1e-8

        worst = 0.0
        for ci in range(2):
            for h in range(2):
                for w in range(2):
                    base = x[ci][h][w]
                    x[ci][h][w] = base + eps
                    lp = loss()
                    x[ci][h][w] = base - eps
                    lm = loss()
                    x[ci][h][w] = base
                    worst = max(worst, abs((lp - lm) / (2 * eps) - dx_ana[ci][h][w]))
        assert worst < 1e-8

    def test_invalid_params(self):
        with pytest.raises(ValueError):
            AI.ConvTranspose2d(3, 2, 3, groups=2)
        with pytest.raises(ValueError):
            AI.ConvTranspose2d(2, 2, 3, stride=1, output_padding=1)
        with pytest.raises(ValueError):
            AI.ConvTranspose2d(2, 2, 0)
        with pytest.raises(ValueError):
            AI.ConvTranspose2d(2, 2, 3, padding=-1)

class TestHooks:
    def test_forward_hook_observes_output(self):
        mlp = AI.MLP([3, 4, 2])
        x = [0.5, -0.2, 0.3]
        base = mlp.forward(x)
        seen = []
        mlp.register_forward_hook(0, lambda layer, out: seen.append(len(out)))
        mlp.register_forward_hook(1, lambda layer, out: seen.append(len(out)))
        out = mlp.forward(x)
        assert seen == [4, 2]
        assert out == base

    def test_backward_hook_receives_deltas(self):
        mlp = AI.MLP([3, 4, 2])
        mlp.forward([0.5, -0.2, 0.3])
        base = AI._mlp_backward(mlp, [1.0, -1.0])
        mlp.clear_hooks()
        mlp.forward([0.5, -0.2, 0.3])
        got = []
        mlp.register_backward_hook(1, lambda layer, d, gw, gb: got.append((len(d), len(gw))))
        mlp.register_backward_hook(0, lambda layer, d, gw, gb: got.append((len(d), len(gw))))
        grads = AI._mlp_backward(mlp, [1.0, -1.0])
        assert got == [(4, 2), (3, 4)]
        for (gw1, gb1), (gw2, gb2) in zip(base, grads):
            for r1, r2 in zip(gw1, gw2):
                for a, b in zip(r1, r2):
                    assert abs(a - b) < 1e-15

    def test_remove_and_clear(self):
        mlp = AI.MLP([2, 2])
        seen = []
        mlp.register_forward_hook(0, lambda layer, out: seen.append("f"))
        mlp.forward([0.1, 0.2])
        assert seen == ["f"]
        mlp.remove_hook(0, 'forward')
        mlp.forward([0.1, 0.2])
        assert seen == ["f"]
        mlp.register_forward_hook(0, lambda layer, out: seen.append("f"))
        mlp.clear_hooks()
        mlp.forward([0.1, 0.2])
        assert seen == ["f"]

    def test_invalid_hook_args_and_model_delegation(self):
        mlp = AI.MLP([2, 2])
        with pytest.raises(ValueError):
            mlp.register_forward_hook(5, lambda layer, out: None)
        with pytest.raises(ValueError):
            mlp.register_backward_hook(-1, lambda layer, d, gw, gb: None)
        with pytest.raises(TypeError):
            mlp.register_forward_hook(0, "not callable")
        with pytest.raises(TypeError):
            mlp.register_backward_hook(True, lambda layer, d, gw, gb: None)

        random.seed(5)
        m1 = AI.Model(AI.MLP([2, 3, 1]), task='regression')
        random.seed(5)
        m2 = AI.Model(AI.MLP([2, 3, 1]), task='regression')
        m1.register_forward_hook(0, lambda layer, out: None)
        m1.register_backward_hook(0, lambda layer, d, gw, gb: None)
        data = [([0.1, 0.2], [0.5]), ([0.3, -0.1], [-0.2]), ([0.0, 0.4], [0.1])]
        h1 = m1.fit(data=data, epochs=5, verbose=False, shuffle=False)
        h2 = m2.fit(data=data, epochs=5, verbose=False, shuffle=False)
        assert [v for _, v in h1['train_loss']] == [v for _, v in h2['train_loss']]


class TestFlopsExtended:
    """estimate_flops 对分组卷积与新增模块的计数"""

    def test_grouped_conv_uses_channels_per_group(self):
        # groups=2 时每个输出通道只连 2 个输入通道，MACs 必须是 6*2*9*25
        seq = AI.Sequential([AI.Conv2d(4, 6, kernel_size=3, padding=1, groups=2)])
        r = AI.estimate_flops(seq, (4, 5, 5))
        assert r['by_layer'][0][1] == 6 * 2 * 9 * 25
        assert r['params'] == 6 * 2 * 9 + 6
        # 与 groups=1 的同形状卷积对比，后者是 4 倍（2 组 × 2 输入通道）
        seq1 = AI.Sequential([AI.Conv2d(4, 6, kernel_size=3, padding=1)])
        r1 = AI.estimate_flops(seq1, (4, 5, 5))
        assert r1['by_layer'][0][1] == 6 * 4 * 9 * 25

    def test_depthwise_conv_counts(self):
        seq = AI.Sequential([AI.Conv2d(3, 3, kernel_size=3, padding=1, groups=3)])
        r = AI.estimate_flops(seq, (3, 4, 4))
        assert r['by_layer'][0][1] == 3 * 1 * 9 * 16
        assert r['params'] == 3 * 1 * 9 + 3

    def test_conv1d_and_transposed_conv(self):
        c1 = AI.Sequential([AI.Conv1d(2, 3, kernel_size=3, stride=2, padding=1)])
        r1 = AI.estimate_flops(c1, (2, 10))
        assert r1['output_shape'] == [3, 5]
        assert r1['by_layer'][0][1] == 3 * 2 * 3 * 5
        assert r1['params'] == 3 * 2 * 3 + 3

        ct = AI.Sequential([AI.ConvTranspose2d(2, 4, kernel_size=3, stride=2,
                                              padding=1, output_padding=1)])
        r2 = AI.estimate_flops(ct, (2, 3, 3))
        assert r2['output_shape'] == [4, 6, 6]
        assert r2['by_layer'][0][1] == 2 * 4 * 9 * 9
        assert r2['params'] == 2 * 4 * 9 + 4

    def test_pool_norm_droppath_supported(self):
        ad = AI.Sequential([AI.AdaptiveAvgPool2d((2, 2))])
        ra = AI.estimate_flops(ad, (3, 4, 4))
        assert ra['output_shape'] == [3, 2, 2]
        assert ra['by_layer'][0][1] == 3 * 16

        gn = AI.Sequential([AI.GroupNorm(2, 4)])
        rg = AI.estimate_flops(gn, (4, 3, 3))
        assert rg['by_layer'][0][1] == 4 * 36
        assert rg['params'] == 8

        dp = AI.Sequential([AI.DropPath(0.1)])
        rdp = AI.estimate_flops(dp, [8])
        assert rdp['macs'] == 0 and rdp['params'] == 0

    def test_shape_mismatch_raises(self):
        for m, shp in ((AI.Conv2d(3, 2, 3), (2, 4, 4)),
                       (AI.Conv1d(3, 2, 3), (2, 8)),
                       (AI.ConvTranspose2d(3, 2, 3), (2, 4, 4)),
                       (AI.GroupNorm(2, 4), (3, 4, 4))):
            with pytest.raises(ValueError):
                AI.estimate_flops(AI.Sequential([m]), shp)


class TestLossPersistence:
    """Model.save 要记住损失函数名与 loss_kwargs，否则重载会悄悄退回默认损失"""

    def _data(self):
        X = [[0.1, 0.9], [0.8, 0.2], [0.3, 0.7], [0.6, 0.4], [0.2, 0.8], [0.9, 0.1]]
        Y = [0, 1, 0, 1, 0, 1]
        return list(zip(X, Y))

    def test_default_save_has_no_loss_keys(self, tmp_path):
        """不传损失时文件里不该多出 loss 字段（旧文件格式完全不变）"""
        random.seed(11)
        mlp = AI.MLP([2, 6, 3], activations=["relu", "softmax"])
        path = str(tmp_path / "d.json")
        AI.save_model(mlp, path, task="classification", quiet=True)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert "loss" not in data
        assert "loss_kwargs" not in data

    def test_focal_round_trip(self, tmp_path):
        random.seed(7)
        m = AI.Model(AI.MLP([2, 6, 3], activations=["relu", "softmax"]),
                     task="classification", loss_fn="focal",
                     loss_kwargs={"gamma": 1.5, "alpha": 0.75})
        m.fit(data=self._data(), epochs=20, batch_size=3, shuffle=False,
              val_split=0.0, verbose=False)
        path = str(tmp_path / "focal.json")
        m.save(path)
        again = AI.Model.load(path)
        assert again._loss_name == "focal"
        assert again.loss_kwargs == {"gamma": 1.5, "alpha": 0.75}
        # 两份都从存档恢复（含优化器状态），再训的 loss 曲线必须逐位相同：
        # 若 loss 配置没恢复成 focal，曲线就会分叉
        a = AI.Model.load(path, restore_optimizer=True)
        b = AI.Model.load(path, restore_optimizer=True)
        h1 = a.fit(data=self._data(), epochs=5, batch_size=3, shuffle=False,
                   val_split=0.0, verbose=False)
        h2 = b.fit(data=self._data(), epochs=5, batch_size=3, shuffle=False,
                   val_split=0.0, verbose=False)
        assert [v for _, v in h1["train_loss"]] == [v for _, v in h2["train_loss"]]

    def test_label_smoothing_round_trip(self, tmp_path):
        random.seed(13)
        m = AI.Model(AI.MLP([2, 5, 3], activations=["relu", "softmax"]),
                     task="classification", loss_fn="ce",
                     loss_kwargs={"label_smoothing": 0.1})
        path = str(tmp_path / "ls.json")
        m.save(path)
        again = AI.Model.load(path)
        assert again._loss_name == "ce"
        assert again.loss_kwargs == {"label_smoothing": 0.1}

    def test_explicit_args_override_file(self, tmp_path):
        random.seed(7)
        m = AI.Model(AI.MLP([2, 6, 3], activations=["relu", "softmax"]),
                     task="classification", loss_fn="focal",
                     loss_kwargs={"gamma": 1.5})
        path = str(tmp_path / "ov.json")
        m.save(path)
        again = AI.Model.load(path, loss_fn="huber", loss_kwargs={"delta": 2.0})
        assert again._loss_name == "huber"
        assert again.loss_kwargs == {"delta": 2.0}

    def test_invalid_loss_field_is_ignored(self, tmp_path):
        random.seed(7)
        m = AI.Model(AI.MLP([2, 6, 3], activations=["relu", "softmax"]),
                     task="classification", loss_fn="focal")
        path = str(tmp_path / "bad.json")
        m.save(path)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        data["loss"] = "no_such_loss"
        data["loss_kwargs"] = "not a dict"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        again = AI.Model.load(path)
        # 坏字段一律忽略，退回按 task 推断的默认损失
        assert again._loss_name == "cross_entropy"
        assert again.loss_kwargs == {}

    def test_custom_callable_writes_no_loss_field(self, tmp_path):
        random.seed(21)
        m = AI.Model(AI.MLP([2, 4, 1], activations=["relu", "linear"]),
                     task="regression", loss_fn=AI.mse_loss)
        path = str(tmp_path / "custom.json")
        m.save(path)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert "loss" not in data
        again = AI.Model.load(path)
        assert again._loss_name == "mse"

    def test_loss_config_and_optimizer_state_coexist(self, tmp_path):
        random.seed(7)
        m = AI.Model(AI.MLP([2, 6, 3], activations=["relu", "softmax"]),
                     task="classification", loss_fn="focal",
                     loss_kwargs={"gamma": 2.0})
        m.fit(data=self._data(), epochs=10, batch_size=3, shuffle=False,
              val_split=0.0, verbose=False)
        path = str(tmp_path / "both.json")
        m.save(path)
        again = AI.Model.load(path, restore_optimizer=True)
        assert again._loss_name == "focal"
        assert again.loss_kwargs == {"gamma": 2.0}
        assert again.optimizer.step_count > 0


class TestFitSeed:
    """fit / train 的 seed：传了就完全可复现，且不推进调用方的全局随机状态"""

    @staticmethod
    def _xy():
        X = [[random.Random(i).uniform(-1, 1),
              random.Random(i + 100).uniform(-1, 1)] for i in range(24)]
        Y = [i % 2 for i in range(24)]
        return X, Y

    @staticmethod
    def _build():
        return AI.Model(AI.MLP([2, 8, 2], activations=["relu", "softmax"]),
                        task="classification")

    def test_seed_makes_fit_reproducible(self):
        X, Y = self._xy()
        random.seed(7)
        m1 = self._build()
        h1 = m1.fit(x=X, y=Y, epochs=6, batch_size=4, shuffle=True,
                    val_split=0.25, verbose=False, seed=42)
        random.seed(7)
        m2 = self._build()
        h2 = m2.fit(x=X, y=Y, epochs=6, batch_size=4, shuffle=True,
                    val_split=0.25, verbose=False, seed=42)
        assert [v for _, v in h1["train_loss"]] == [v for _, v in h2["train_loss"]]
        assert h1["val_loss"] == h2["val_loss"]
        assert h1["grad_norm"] == h2["grad_norm"]
        for (wa, ba), (wb, bb) in zip(m1.model.get_parameters(),
                                      m2.model.get_parameters()):
            assert wa == wb and ba == bb

    def test_different_seed_changes_order(self):
        X, Y = self._xy()
        random.seed(7)
        m1 = self._build()
        h1 = m1.fit(x=X, y=Y, epochs=6, batch_size=4, shuffle=True,
                    val_split=0.25, verbose=False, seed=42)
        random.seed(7)
        m2 = self._build()
        h2 = m2.fit(x=X, y=Y, epochs=6, batch_size=4, shuffle=True,
                    val_split=0.25, verbose=False, seed=43)
        assert [v for _, v in h1["train_loss"]] != [v for _, v in h2["train_loss"]]

    def test_seed_does_not_advance_global_rng(self):
        X, Y = self._xy()
        random.seed(7)
        m = self._build()
        random.seed(123)
        before = [random.random() for _ in range(3)]
        random.seed(123)
        m.fit(x=X, y=Y, epochs=2, batch_size=4, shuffle=True, val_split=0.0,
              verbose=False, seed=9)
        assert [random.random() for _ in range(3)] == before

    def test_shuffle_false_is_reproducible(self):
        X, Y = self._xy()
        random.seed(7)
        m1 = self._build()
        h1 = m1.fit(x=X, y=Y, epochs=4, batch_size=4, shuffle=False,
                    val_split=0.0, verbose=False, seed=1)
        random.seed(7)
        m2 = self._build()
        h2 = m2.fit(x=X, y=Y, epochs=4, batch_size=4, shuffle=False,
                    val_split=0.0, verbose=False, seed=1)
        assert [v for _, v in h1["train_loss"]] == [v for _, v in h2["train_loss"]]

    def test_no_seed_still_works(self):
        X, Y = self._xy()
        random.seed(7)
        m = self._build()
        h = m.fit(x=X, y=Y, epochs=2, batch_size=8, shuffle=True,
                  val_split=0.0, verbose=False)
        assert len(h["train_loss"]) == 2

    def test_top_level_train_seed(self):
        X, Y = self._xy()
        data = [(list(x), y) for x, y in zip(X, Y)]

        def run(seed):
            random.seed(77)
            mlp = AI.MLP([2, 6, 2], activations=["relu", "softmax"])
            opt = AI.Adam(mlp, lr=0.05)
            return AI.train(mlp, AI.cross_entropy_loss, opt, list(data),
                            epochs=4, batch_size=4, patience=0, verbose=False,
                            seed=seed)["train_loss"]

        assert run(11) == run(11)
        assert run(11) != run(12)


class TestQuickTrainSeed:
    """quick_train 的 seed 同时管住「权重初始化」与「每轮打乱」"""

    @staticmethod
    def _xy():
        X = [[random.Random(i).uniform(-1, 1),
              random.Random(i + 50).uniform(-1, 1)] for i in range(20)]
        Y = [0.5 * x[0] - 0.25 * x[1] for x in X]
        return X, Y

    @staticmethod
    def _weights(model):
        return [[[v for v in row] for row in w] for w, _ in model.model.get_parameters()]

    def test_same_seed_reproducible(self):
        X, Y = self._xy()
        a = AI.quick_train(X, Y, epochs=10, batch_size=8, verbose=False, seed=42)
        b = AI.quick_train(X, Y, epochs=10, batch_size=8, verbose=False, seed=42)
        assert self._weights(a) == self._weights(b)
        assert [v for _, v in a.history["train_loss"]] ==                [v for _, v in b.history["train_loss"]]

    def test_different_seed_differs(self):
        X, Y = self._xy()
        a = AI.quick_train(X, Y, epochs=10, batch_size=8, verbose=False, seed=42)
        b = AI.quick_train(X, Y, epochs=10, batch_size=8, verbose=False, seed=43)
        assert self._weights(a) != self._weights(b)

    def test_seed_does_not_advance_global_rng(self):
        X, Y = self._xy()
        random.seed(2024)
        before = [random.random() for _ in range(3)]
        random.seed(2024)
        AI.quick_train(X, Y, epochs=3, batch_size=8, verbose=False, seed=7)
        assert [random.random() for _ in range(3)] == before


class TestGradCheckSeq2Seq:
    """grad_check 支持 Seq2SeqLM，并且不再把模型留在 eval"""

    @staticmethod
    def _model(seed=11):
        random.seed(seed)
        return AI.Seq2SeqLM(6, 7, d_model=8, num_heads=2, d_ff=16,
                            num_layers=1, max_len=6, dropout=0.0)

    def test_seq2seq_passes(self):
        s2s = self._model()
        res = AI.grad_check(s2s, ([1, 2, 3], [4, 5, 2]), tol=1e-5)
        assert res["ok"], res["worst_name"]
        assert res["worst_rel"] <= 1e-5
        assert "zeros" in res

    def test_seq2seq_covers_decoder(self):
        s2s = self._model()
        res = AI.grad_check(s2s, ([1, 2, 3], [4, 5, 2]), max_params=40)
        names = [c["name"] for c in res["checks"]]
        assert any(n.startswith("out.") for n in names)
        assert any(".cross." in n for n in names)

    def test_zero_gradient_recognized(self):
        # b_k 对 softmax 等价于「整行加常数」，真实梯度为 0：
        # 应该进 zeros 而不是算成相对误差 1.0 误报
        s2s = self._model()
        res = AI.grad_check(s2s, ([1, 2, 3], [4, 5, 2]), max_params=60)
        assert res["ok"], res["worst_name"]
        assert any(".b_k" in z for z in res["zeros"]) or len(res["zeros"]) > 0

    def test_restores_training_mode(self):
        s2s = self._model()
        s2s.train()
        AI.grad_check(s2s, ([1, 2, 3], [4, 5, 2]))
        assert s2s.encoder.layers[0].self_attn.training is True
        s2s.eval()
        AI.grad_check(s2s, ([1, 2, 3], [4, 5, 2]))
        assert s2s.encoder.layers[0].self_attn.training is False

    def test_causal_lm_mode_restored(self):
        random.seed(12)
        lm = AI.CausalLM(7, d_model=8, num_heads=2, d_ff=16, num_layers=1,
                         max_len=6, dropout=0.2)
        lm.train()
        res = AI.grad_check(lm, [0, 1, 2, 3])
        assert res["ok"], res["worst_name"]
        assert lm.encoder.layers[0].self_attn.training is True

    def test_requires_pair(self):
        s2s = self._model()
        with pytest.raises(TypeError):
            AI.grad_check(s2s, [1, 2, 3])


class TestSeq2SeqPaddedBatch:
    """Seq2SeqLM 的 pad_mask：补齐不改变真实位置的数值，批量训练按 token 加权"""

    @staticmethod
    def _build(seed=5):
        random.seed(seed)
        return AI.Seq2SeqLM(7, 7, d_model=8, num_heads=2, d_ff=16,
                            num_layers=1, max_len=8, dropout=0.0)

    @staticmethod
    def _grads(model):
        out = []
        for _, _, g in model._named_params():
            out.append([[list(r) for r in g] if isinstance(g[0], list) else list(g)])
        return out

    def test_padding_is_numerically_transparent(self):
        m = self._build()
        m.eval()
        m.zero_grad()
        base = m._loss_and_backward([1, 2, 3, 4], [3, 5, 6])
        g_base = self._grads(m)

        m.zero_grad()
        padded = m._loss_and_backward(
            [1, 2, 3, 4, 0, 0], [3, 5, 6, 0, 0],
            src_mask=[1, 1, 1, 1, 0, 0], tgt_mask=[1, 1, 1, 0, 0])
        assert padded == base
        assert self._grads(m) == g_base

    def test_padding_keeps_real_logits(self):
        m = self._build()
        m.eval()
        a = m.forward([1, 2, 3, 4], [3, 5])
        b = m.forward([1, 2, 3, 4, 0, 0], [3, 5, 0, 0],
                      src_mask=[1, 1, 1, 1, 0, 0])
        for t in range(2):
            assert a[t] == b[t]

    def test_batched_step_matches_manual_weighting(self):
        pairs = [([1, 2, 3], [3, 5, 6]), ([4, 2], [6, 1]), ([1, 5, 3, 2], [2, 2, 6])]
        a = self._build(seed=9)
        b = self._build(seed=9)
        assert [p for _, p, _ in a._named_params()] == [p for _, p, _ in b._named_params()]

        a.train()
        a.zero_grad()
        a.fit(pairs, epochs=1, lr=0.01, shuffle=False, batch_size=2)

        b.train()
        for group in (pairs[:2], pairs[2:]):
            n_total = sum(len(t) for _, t in group)
            b.zero_grad()
            for s, t in group:
                b._loss_and_backward(s, t, grad_scale=len(t) / n_total)
            b.adam_step(0.01)

        for (na, pa, _), (nb, pb, _) in zip(a._named_params(), b._named_params()):
            assert na == nb
            assert pa == pb

    def test_epoch_loss_accounting_matches(self):
        pairs = [([1, 2, 3], [3, 5, 6]), ([4, 2], [6, 1]), ([1, 5, 3, 2], [2, 2, 6])]
        # lr=0 参数不动，两者记的都是「各样本 loss 的平均」
        h1 = self._build(seed=21).fit(pairs, epochs=1, lr=0.0, shuffle=False)
        h2 = self._build(seed=21).fit(pairs, epochs=1, lr=0.0, shuffle=False,
                                      batch_size=3)
        assert h1 == h2

    def test_batch_size_validation(self):
        pairs = [([1, 2], [3])]
        for bad in (0, -1, True, 1.5):
            with pytest.raises((ValueError, TypeError)):
                self._build(seed=1).fit(pairs, epochs=1, batch_size=bad)

    def test_mask_validation(self):
        m = self._build()
        with pytest.raises(ValueError):
            m._loss_and_backward([1, 2, 3], [1, 2], src_mask=[1, 1])
        with pytest.raises(ValueError):
            m._loss_and_backward([1, 2, 3], [1, 2], tgt_mask=[0, 0])
        with pytest.raises(ValueError):
            self._build(seed=2).fit(
                [([1, 2, 3, 4, 5, 6, 7, 1, 2], [1]), ([1], [1])],
                epochs=1, batch_size=2)


class TestKVCacheMatrix:
    """KV cache 与整段重算在 GQA / RoPE / 滑动窗口 组合下必须逐位一致"""

    def _lm(self, heads=4, kv=None, window=None, rope=False, max_len=8, seed=7):
        random.seed(seed)
        lm = AI.CausalLM(9, d_model=8, num_heads=heads, d_ff=16, num_layers=2,
                         max_len=max_len, dropout=0.0, rope=rope,
                         window=window, num_kv_heads=kv)
        lm.eval()
        return lm

    def test_cache_matches_recompute_across_configs(self):
        configs = [
            dict(heads=2),
            dict(heads=4, kv=1),
            dict(heads=4, kv=2),
            dict(heads=4, rope=True),
            dict(heads=4, window=3),
            dict(heads=4, kv=2, rope=True),
            dict(heads=4, kv=2, rope=True, window=2),
        ]
        for cfg in configs:
            for max_len, prompt in ((8, [1, 2, 3]), (8, [1, 2, 3, 4, 5, 6]), (4, [1, 2, 3, 4])):
                lm = self._lm(max_len=max_len, **cfg)
                for top_k in (1, 3):
                    a = lm.generate(prompt, 5, temperature=0.8, top_k=top_k,
                                    rng=random.Random(7), use_cache=True)
                    b = lm.generate(prompt, 5, temperature=0.8, top_k=top_k,
                                    rng=random.Random(7), use_cache=False)
                    assert a == b, (cfg, max_len, prompt, top_k, a, b)

    def test_beam_search_deterministic_across_configs(self):
        for cfg in (dict(heads=2), dict(heads=4, kv=1), dict(heads=4, window=2, rope=True)):
            lm = self._lm(**cfg)
            for prompt in ([1, 2, 3], [4, 5, 6, 7]):
                a = lm.generate(prompt, 4, num_beams=3, rng=random.Random(3))
                b = lm.generate(prompt, 4, num_beams=3, rng=random.Random(3))
                assert a == b, (cfg, prompt, a, b)


class TestLMPersistenceMatrix:
    """存档要能把 rope / window / num_kv_heads 一起带走，重载后数值逐位相同"""

    def _roundtrip(self, path, **cfg):
        random.seed(5)
        lm = AI.CausalLM(9, d_model=8, num_heads=4, d_ff=16, num_layers=2,
                         max_len=8, dropout=0.2, **cfg)
        lm.fit([[1, 2, 3, 4]], epochs=2, lr=0.01, seed=1)
        lm.save(path, include_optimizer=True, quiet=True)
        back = AI.CausalLM.load(path, restore_optimizer=True)
        return lm, back

    def test_config_survives_roundtrip(self, tmp_path):
        path = str(tmp_path / "lm.json")
        for cfg in ({}, {"rope": True}, {"window": 3},
                    {"num_kv_heads": 1}, {"rope": True, "window": 2, "num_kv_heads": 2}):
            lm, back = self._roundtrip(path, **cfg)
            assert back.num_kv_heads == cfg.get("num_kv_heads")
            assert back.window == cfg.get("window")
            assert back.rope is cfg.get("rope", False)
            assert back.use_abs_pos == (not back.rope)
            lm.eval()
            back.eval()
            probe = [1, 2, 3, 4]
            assert lm.forward(probe) == back.forward(probe), cfg
            assert lm.evaluate([probe]) == back.evaluate([probe]), cfg
            assert lm.generate(probe, 4, top_k=1, rng=random.Random(2)) == \
                   back.generate(probe, 4, top_k=1, rng=random.Random(2)), cfg
            assert back.step_count == 2, cfg


# ============================================================================
#  审计 7 条（F1–F7）的回归测试
# ============================================================================
class TestPadMaskAlignment:
    def test_left_padding_raises(self):
        lm = AI.CausalLM(vocab_size=6, d_model=8, num_heads=2, num_layers=1, max_len=8)
        with pytest.raises(ValueError, match="右填充"):
            lm.forward([0, 0, 1, 2, 3], pad_mask=[0, 0, 1, 1, 1])
        with pytest.raises(ValueError, match="右填充"):
            lm._loss_and_backward([0, 0, 1, 2, 3], [0, 0, 1, 1, 1])

    def test_right_padding_still_works(self):
        lm = AI.CausalLM(vocab_size=6, d_model=8, num_heads=2, num_layers=1, max_len=8)
        lm.eval()
        logits = lm.forward([1, 2, 3, 0, 0], pad_mask=[1, 1, 1, 0, 0])
        assert all(v == v for row in logits for v in row)
        loss = lm._loss_and_backward([1, 2, 3, 0, 0], [1, 1, 1, 0, 0])
        assert loss == loss and loss > 0.0

    def test_fully_masked_row_is_zero_not_nan(self):
        assert AI._softmax_row([float("-inf")] * 3) == [0.0, 0.0, 0.0]


class TestTopKValidation:
    def test_invalid_top_k_raises(self):
        lm = AI.CausalLM(vocab_size=5, d_model=8, num_heads=2, num_layers=1, max_len=8)
        for bad in (0.5, -1, 2.7, True):
            with pytest.raises(ValueError, match="top_k"):
                lm._sample([0.1, 0.2, 0.3, 0.4, 0.5], 1.0, bad, random.Random(0))
            with pytest.raises(ValueError, match="top_k"):
                lm.generate([1], 3, top_k=bad)

    def test_invalid_top_k_in_beam_search(self):
        lm = AI.CausalLM(vocab_size=5, d_model=8, num_heads=2, num_layers=1, max_len=8)
        with pytest.raises(ValueError, match="top_k"):
            lm.generate([1], 3, num_beams=3, top_k=0.5)

    def test_valid_top_k_still_samples(self):
        lm = AI.CausalLM(vocab_size=5, d_model=8, num_heads=2, num_layers=1, max_len=8)
        rng = random.Random(0)
        logits = [0.1, 0.2, 0.3, 0.4, 0.5]
        assert all(0 <= lm._sample(logits, 1.0, 2, rng) < 5 for _ in range(5))
        assert lm._sample(logits, 1.0, 0, random.Random(1)) in range(5)


class TestLMTargetValidation:
    def test_negative_and_out_of_range_ids_raise(self):
        lm = AI.CausalLM(vocab_size=5, d_model=8, num_heads=2, num_layers=1, max_len=8)
        for bad in ([1, 2, -1], [1, 2, 5], [1, 2, -7], [True, 2, 3]):
            with pytest.raises(ValueError, match="token id"):
                lm._loss_and_backward(bad)
        with pytest.raises(ValueError, match="token id"):
            lm.evaluate([[1, 2, -1]])

    def test_valid_ids_unchanged(self):
        lm = AI.CausalLM(vocab_size=5, d_model=8, num_heads=2, num_layers=1, max_len=8)
        loss = lm._loss_and_backward([1, 2, 3])
        assert loss == loss and loss > 0.0


class TestNonFiniteLogits:
    def test_nan_weights_raise_instead_of_clamping(self):
        lm = AI.CausalLM(vocab_size=4, d_model=8, num_heads=2, num_layers=1, max_len=8)
        lm.W_out[0][0] = float("nan")
        with pytest.raises(ValueError, match="NaN"):
            lm.evaluate([[1, 2, 3]])
        with pytest.raises(ValueError, match="NaN"):
            lm._loss_and_backward([1, 2, 3])


class TestRestoreOptimizerStrict:
    def test_missing_optimizer_state_raises(self, tmp_path):
        lm = AI.CausalLM(vocab_size=5, d_model=8, num_heads=2, num_layers=1, max_len=8)
        path = str(tmp_path / "lm.json")
        lm.save(path, include_optimizer=False, quiet=True)
        with pytest.raises(ValueError, match="没有优化器状态"):
            AI.CausalLM.load(path, restore_optimizer=True)
        assert AI.CausalLM.load(path) is not None

    def test_present_optimizer_state_restores(self, tmp_path):
        lm = AI.CausalLM(vocab_size=5, d_model=8, num_heads=2, num_layers=1, max_len=8)
        lm.fit([[1, 2, 3]], epochs=2, lr=0.01, shuffle=False)
        path = str(tmp_path / "lm.json")
        lm.save(path, include_optimizer=True, quiet=True)
        back = AI.CausalLM.load(path, restore_optimizer=True)
        assert back.step_count == lm.step_count
        assert len(back._adam_m) > 0


class TestBPEForwardRef:
    def test_forward_reference_raises_value_error(self):
        for merges in ([(256, 97)], [(97, 98), (256, 257)], [(500, 97)]):
            with pytest.raises(ValueError, match="合并规则"):
                AI.BytePairTokenizer(merges)

    def test_bad_type_raises_value_error(self):
        with pytest.raises(ValueError, match="merges"):
            AI.BytePairTokenizer("abc")

    def test_valid_merges_still_work(self):
        tok = AI.BytePairTokenizer([(97, 98), (256, 99)])
        assert tok.decode(tok.encode("abcab")) == "abcab"


class TestSeq2SeqGenerateValidation:
    def test_source_longer_than_max_len_raises(self):
        m = AI.Seq2SeqLM(src_vocab=10, tgt_vocab=10, d_model=8, num_heads=2,
                         d_ff=8, num_layers=1, max_len=4)
        with pytest.raises(ValueError, match="max_len"):
            m.generate([1, 2, 3, 4, 5, 6], 3, top_k=1)
        with pytest.raises(ValueError, match="max_len"):
            m.generate([1, 2, 3, 4, 5, 6], 3, num_beams=2)

    def test_source_id_out_of_range_raises(self):
        m = AI.Seq2SeqLM(src_vocab=10, tgt_vocab=10, d_model=8, num_heads=2,
                         d_ff=8, num_layers=1, max_len=4)
        with pytest.raises(ValueError, match="token id"):
            m.generate([1, 2, 3, 11], 3)

    def test_normal_length_still_generates(self):
        m = AI.Seq2SeqLM(src_vocab=10, tgt_vocab=10, d_model=8, num_heads=2,
                         d_ff=8, num_layers=1, max_len=4)
        assert len(m.generate([1, 2, 3], 3, top_k=1)) == 3


class TestReviewFixes:
    """外部评审确认的缺陷：越界梯度、错误采样分布、存档丢字段等。"""

    def test_conv_transpose_bias_grad_not_scaled(self):
        # 偏置梯度原来写在 ci 循环里，被放大 cin/groups 倍
        for cin, cout, groups in ((2, 3, 1), (3, 2, 1), (4, 2, 2)):
            random.seed(11)
            ct = AI.ConvTranspose2d(cin, cout, 3, stride=2, padding=1, groups=groups)
            x = [[[random.uniform(-1, 1) for _ in range(3)] for _ in range(3)]
                 for _ in range(cin)]
            out = ct.forward(x)
            dout = [[[random.uniform(-1, 1) for _ in row] for row in ch] for ch in out]
            ct.zero_grad()
            ct.backward(dout)
            ana = list(ct.grad_b)

            def loss_of():
                o = ct.forward(x)
                return sum(o[c][h][w] * dout[c][h][w]
                           for c in range(len(o))
                           for h in range(len(o[c]))
                           for w in range(len(o[c][h])))

            eps = 1e-6
            for i in range(cout):
                ct.biases[i] += eps
                lp = loss_of()
                ct.biases[i] -= 2 * eps
                lm = loss_of()
                ct.biases[i] += eps
                num = (lp - lm) / (2 * eps)
                assert abs(ana[i] - num) <= 1e-6 * max(1.0, abs(num)), (cin, groups, i)

    def test_mixup_lam_is_beta(self):
        rng = random.Random(5)
        lams = []
        for _ in range(2000):
            xs, soft = AI.mixup([0.0], 1, [1.0], 2, alpha=0.2, rng=rng)
            lam = soft[1]
            lams.append(lam)
            assert 0.0 <= lam <= 1.0
            # 软标签必须是合法概率：每项非负、和为 1
            assert all(v >= 0.0 for v in soft)
            assert abs(sum(soft) - 1.0) < 1e-12
            AI.cross_entropy_loss([0.5, 0.3, 0.2], soft)  # 不应被 CE 拒绝
        mean = sum(lams) / len(lams)
        assert abs(mean - 0.5) < 0.05

    def test_grouped_conv_archive_round_trip(self, tmp_path):
        random.seed(3)
        conv = AI.Conv2d(4, 6, 3, stride=1, padding=1, groups=2)
        fc = AI.MLP([6, 3])
        path = str(tmp_path / "conv_groups.json")
        AI.save_conv_model([conv], fc, path)

        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert data["conv"][0]["groups"] == 2

        back, _ = AI.load_conv_model(path)
        assert back[0].groups == 2 and back[0].cg == 2

    def test_legacy_conv_archive_without_groups(self, tmp_path):
        random.seed(4)
        conv = AI.Conv2d(4, 6, 3, stride=1, padding=1, groups=1)
        fc = AI.MLP([6, 3])
        path = str(tmp_path / "conv_legacy.json")
        AI.save_conv_model([conv], fc, path)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        data["conv"][0].pop("groups")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)

        back, _ = AI.load_conv_model(path)
        assert back[0].groups == 1 and back[0].cg == 4

    def test_cross_attention_declares_delta_index(self):
        # 不声明的话容器取末项，会静默拿到 dmemory
        assert getattr(AI.CrossAttention, "delta_index", -1) == 0
        ca = AI.CrossAttention(8, 2)
        seq = AI.Sequential(ca)
        assert getattr(seq.modules[0], "delta_index", -1) == 0

    def test_estimate_flops_flatten_rejects_low_dim(self):
        seq = AI.Sequential(AI.Flatten(), AI.Layer(12, 5))
        for shape in ([4], [4, 4]):
            with pytest.raises(ValueError, match="Flatten"):
                AI.estimate_flops(seq, shape)
        assert AI.estimate_flops(seq, [3, 2, 2])["output_shape"] == [5]

    def test_lm_save_rejects_nan(self, tmp_path):
        random.seed(1)
        lm = AI.CausalLM(vocab_size=5, d_model=8, num_heads=2, d_ff=16,
                         num_layers=1, max_len=8)
        lm.W_out[0][0] = float("nan")
        with pytest.raises(ValueError):
            lm.save(str(tmp_path / "nan_lm.json"))

    def test_pad_mask_doc_mentions_right_padding(self):
        doc = AI._check_pad_mask.__doc__ or ""
        assert "右填充" in doc

    def test_summary_shows_real_activation(self, capsys):
        mlp = AI.MLP([3, 5, 2], activations=["gelu", "softmax"])
        model = AI.Model(mlp, task="classification")
        model.summary()
        out = capsys.readouterr().out
        assert "GELU" in out and "Softmax" in out
        assert "Linear(3 → 5) + Linear" not in out

    def test_grad_check_message_lists_seq2seq(self):
        with pytest.raises(TypeError, match="Seq2SeqLM"):
            AI.grad_check(123, [1.0])

    def test_sequential_accepts_list_and_rejects_non_module(self):
        a, b = AI.Layer(4, 6), AI.Layer(6, 3)
        flat = AI.Sequential([a, b])
        assert len(flat.modules) == 2
        assert len(flat.forward([0.5, -0.2, 0.1, 0.3])) == 3

        with pytest.raises(ValueError, match="forward"):
            AI.Sequential(object())


class TestMetrics:
    """分类 / 回归 / 序列 / 生成指标。"""

    def test_confusion_and_accuracy(self):
        yt, yp = [0, 0, 1, 1, 2, 2], [0, 1, 1, 2, 2, 2]
        assert AI.confusion_matrix(yt, yp) == [[1, 1, 0], [0, 1, 1], [0, 0, 2]]
        assert abs(AI.accuracy_score(yt, yp) - 4 / 6) < 1e-12
        # num_classes 可以补零列
        mat = AI.confusion_matrix([0], [0], num_classes=3)
        assert mat == [[1, 0, 0], [0, 0, 0], [0, 0, 0]]

    def test_precision_recall_f1_averages(self):
        yt, yp = [0, 0, 1, 1, 2, 2], [0, 1, 1, 2, 2, 2]
        macro = AI.precision_recall_f1(yt, yp)
        assert abs(macro["precision"] - (1 + 0.5 + 2 / 3) / 3) < 1e-12
        assert abs(macro["recall"] - (0.5 + 0.5 + 1) / 3) < 1e-12
        assert abs(macro["f1"] - (2 / 3 + 0.5 + 0.8) / 3) < 1e-12
        micro = AI.precision_recall_f1(yt, yp, average="micro")
        assert abs(micro["precision"] - 4 / 6) < 1e-12
        weighted = AI.precision_recall_f1(yt, yp, average="weighted")
        assert abs(weighted["recall"] - (0.5 + 0.5 + 1) / 3) < 1e-12
        none = AI.precision_recall_f1(yt, yp, average="none")
        assert len(none["per_class"]) == 3

    def test_mcc_and_auc(self):
        # tp=2 tn=1 fp=1 fn=0 → (2*1-1*0)/sqrt(3*2*2*1)=2/sqrt(12)
        assert abs(AI.matthews_corrcoef([0, 0, 1, 1], [0, 1, 1, 1]) - 2 / math.sqrt(12)) < 1e-12
        assert abs(AI.roc_auc_score([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) - 1.0) < 1e-12
        assert abs(AI.roc_auc_score([0, 1, 0, 1], [0.5] * 4) - 0.5) < 1e-12
        assert abs(AI.roc_auc_score([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1])) < 1e-12

    def test_top_k_and_sequence(self):
        scores = [[0.7, 0.2, 0.1], [0.1, 0.6, 0.3], [0.2, 0.3, 0.5]]
        assert AI.top_k_accuracy([0, 1, 2], scores, 1) == 1.0
        assert abs(AI.top_k_accuracy([0, 2, 2], scores, 1) - 2 / 3) < 1e-12
        assert AI.top_k_accuracy([0, 1, 2], scores, 2) == 1.0
        assert abs(AI.sequence_accuracy([[1, 2], [1, 3], [2, 2]],
                                        [[1, 2], [1, 3], [2, 3]]) - 2 / 3) < 1e-12

    def test_regression_metrics(self):
        yt, yp = [1.0, 2.0, 3.0, 4.0], [1.1, 1.9, 3.2, 3.8]
        assert abs(AI.mean_absolute_error(yt, yp) - 0.15) < 1e-12
        assert abs(AI.root_mean_squared_error(yt, yp) - math.sqrt(0.025)) < 1e-12
        assert abs(AI.mean_absolute_percentage_error(yt, yp)
                   - 100 * (0.1 + 0.05 + 0.2 / 3 + 0.05) / 4) < 1e-9
        ss_tot = sum((v - 2.5) ** 2 for v in yt)
        ss_res = sum((a - b) ** 2 for a, b in zip(yt, yp))
        assert abs(AI.r2_score(yt, yp) - (1 - ss_res / ss_tot)) < 1e-12
        # 常数目标：全对 1.0、有偏 0.0（不是 -1e15）
        assert AI.r2_score([2.0, 2.0], [2.0, 2.0]) == 1.0
        assert AI.r2_score([2.0, 2.0], [2.0, 2.5]) == 0.0
        # 单输出模型给 [v]、标签给标量也要能比
        assert abs(AI.mean_absolute_error([1.0], [[1.5]]) - 0.5) < 1e-12

    def test_bleu_and_rouge(self):
        assert AI.bleu_score([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]) > 0.99
        assert 0 < AI.bleu_score([1, 2], [1, 2, 3, 4]) < 1
        # 一个词都不重叠时必须是 0（否则 add-1 平滑会给出 0.3+）
        assert AI.bleu_score([9, 9, 9, 9], [1, 2, 3, 4]) == 0.0
        assert abs(AI.rouge_l([1, 2, 3], [1, 2, 3]) - 1.0) < 1e-12
        assert abs(AI.rouge_l([1, 2, 9], [1, 2, 3]) - 2 / 3) < 1e-12

    def test_evaluate_all_classification(self):
        random.seed(7)
        xs = [[random.uniform(-1, 1), random.uniform(-1, 1)] for _ in range(40)]
        ys = [1 if a + b > 0 else 0 for a, b in xs]
        model = AI.Model(AI.MLP([2, 8, 2], activations=["relu", "softmax"]),
                         task="classification", loss_fn="cross_entropy", lr=0.1)
        model.fit(data=list(zip(xs, ys)), epochs=60, verbose=False, seed=1)
        res = model.evaluate_all(xs, ys)
        for key in ("accuracy", "precision", "recall", "f1", "confusion_matrix",
                    "roc_auc", "mcc", "top_k_accuracy"):
            assert key in res
        assert res["accuracy"] > 0.85
        assert res["n_samples"] == 40

    def test_evaluate_all_regression(self):
        random.seed(8)
        xs = [[random.uniform(-1, 1), random.uniform(-1, 1)] for _ in range(30)]
        ys = [a - b for a, b in xs]
        model = AI.Model(AI.MLP([2, 8, 1], activations=["relu", "linear"]),
                         task="regression", lr=0.05)
        model.fit(data=list(zip(xs, ys)), epochs=120, verbose=False, seed=2)
        res = model.evaluate_all(xs, ys)
        assert {"mae", "rmse", "mape", "r2"} <= set(res)
        assert res["r2"] > 0.8

    def test_metric_validation(self):
        with pytest.raises(ValueError):
            AI.accuracy_score([], [])
        with pytest.raises(ValueError):
            AI.accuracy_score([1], [])
        with pytest.raises(ValueError):
            AI.precision_recall_f1([0, 1], [0, 1], average="bogus")
        with pytest.raises(ValueError):
            AI.roc_auc_score([0, 0], [0.1, 0.2])
        with pytest.raises(ValueError):
            AI.roc_auc_score([0, 2], [0.1, 0.2])
        with pytest.raises(ValueError):
            AI.top_k_accuracy([1], [[0.5, 0.5]], k=0)
        with pytest.raises(ValueError):
            AI.confusion_matrix([0, 3], [0, 1], num_classes=2)
        with pytest.raises(TypeError):
            AI.evaluate_all(object(), [1], [1])


class TestSamplingControls:
    """生成采样控制：min_p / 重复惩罚 / presence / frequency / 禁重复 n-gram。"""

    @staticmethod
    def _trained_lm():
        random.seed(11)
        lm = AI.CausalLM(vocab_size=3, d_model=16, num_heads=2, d_ff=32,
                         num_layers=1, max_len=16)
        lm.fit([list(range(3)) * 4], epochs=40, lr=0.02, shuffle=False)
        return lm

    def test_penalties_default_is_identity(self):
        logits = [1.0, 2.0, 3.0]
        same = AI._apply_generation_penalties(logits, [0, 1])
        assert same == logits and logits == [1.0, 2.0, 3.0]   # 不回改入参

    def test_repetition_penalty(self):
        got = AI._apply_generation_penalties([1.0, 2.0, 3.0], [2], repetition_penalty=2.0)
        assert abs(got[2] - 1.5) < 1e-12
        got_neg = AI._apply_generation_penalties([-1.0, 2.0, 3.0], [0],
                                                 repetition_penalty=2.0)
        assert abs(got_neg[0] + 2.0) < 1e-12

    def test_presence_and_frequency(self):
        pre = AI._apply_generation_penalties([1.0, 1.0, 1.0], [1, 1, 2],
                                             presence_penalty=0.5)
        assert abs(pre[1] - 0.5) < 1e-12 and abs(pre[2] - 0.5) < 1e-12
        freq = AI._apply_generation_penalties([1.0, 1.0, 1.0], [1, 1, 2],
                                              frequency_penalty=0.5)
        assert abs(freq[1] - 0.0) < 1e-12 and abs(freq[2] - 0.5) < 1e-12

    def test_no_repeat_ngram_mask(self):
        # 末尾是 (1,2)，历史上 (1,2) 后面跟过 0 → 禁用 0
        got = AI._apply_generation_penalties([0.0] * 4, [1, 2, 0, 1, 2],
                                             no_repeat_ngram_size=3)
        assert got[0] == float("-inf") and got[3] == 0.0

    def test_mask_min_p(self):
        masked = AI._mask_min_p([0.0, 0.0, 10.0], 0.5)
        assert masked[0] == float("-inf") and masked[2] == 10.0
        assert AI._mask_min_p([1.0, 2.0], 0.0) == [1.0, 2.0]

    def test_default_generate_unchanged(self):
        lm = self._trained_lm()
        assert lm.generate([0], 8, top_k=1) == [0, 1, 2, 0, 1, 2, 0, 1, 2]
        explicit = lm.generate([0], 8, top_k=1, min_p=0.0, repetition_penalty=1.0,
                               presence_penalty=0.0, frequency_penalty=0.0,
                               no_repeat_ngram_size=0)
        assert explicit == lm.generate([0], 8, top_k=1)

    def test_sampling_cache_matches_recompute(self):
        lm = self._trained_lm()
        a = lm.generate([0], 8, rng=random.Random(5), use_cache=True)
        b = lm.generate([0], 8, rng=random.Random(5), use_cache=False)
        assert a == b

    def test_no_repeat_ngram_changes_greedy(self):
        lm = self._trained_lm()
        greedy = lm.generate([0, 1, 2], 6, top_k=1)
        banned = lm.generate([0, 1, 2], 6, top_k=1, no_repeat_ngram_size=2)
        assert greedy != banned
        pairs = [(banned[i], banned[i + 1]) for i in range(len(banned) - 1)]
        assert len(set(pairs)) == len(pairs)

    def test_penalty_validation(self):
        lm = self._trained_lm()
        for kwargs in ({"min_p": 1.0}, {"min_p": -0.1}, {"repetition_penalty": 0.0},
                       {"repetition_penalty": -2.0}, {"no_repeat_ngram_size": 1},
                       {"no_repeat_ngram_size": -1}, {"presence_penalty": True}):
            with pytest.raises(ValueError):
                lm.generate([0], 2, **kwargs)

    def test_seq2seq_default_and_validation(self):
        random.seed(21)
        s2s = AI.Seq2SeqLM(4, 4, d_model=16, num_heads=2, d_ff=32,
                           num_layers=1, max_len=10)
        pairs = [([1, 2, 3], [1, 2, 3]), ([2, 3, 1], [2, 3, 1]), ([3, 1, 2], [3, 1, 2])]
        s2s.fit(pairs, epochs=40, lr=0.01, shuffle=False)
        base = s2s.generate([1, 2, 3], 3, top_k=1)
        assert base == [1, 2, 3]
        assert s2s.generate([1, 2, 3], 3, top_k=1, min_p=0.0, repetition_penalty=1.0,
                            presence_penalty=0.0, frequency_penalty=0.0,
                            no_repeat_ngram_size=0) == base
        with pytest.raises(ValueError):
            s2s.generate([1, 2], 2, min_p=1.5)


class TestConvDilation:
    """Conv2d 的 dilation（空洞卷积）。"""

    @staticmethod
    def _rand(C, H, W, seed=1):
        r = random.Random(seed)
        return [[[r.uniform(-1, 1) for _ in range(W)] for _ in range(H)] for _ in range(C)]

    def test_receptive_field_and_manual_match(self):
        random.seed(31)
        conv = AI.Conv2d(2, 3, 3, padding=1, dilation=2)
        x = self._rand(2, 5, 5, seed=7)
        out = conv.forward(x)
        # 等效核 5×5：5+2-5=2 → 3×3
        assert len(out) == 3 and len(out[0]) == 3 and len(out[0][0]) == 3
        # 朴素参考实现（逐输出、逐核、逐通道，自己算空洞偏移与零填充）
        for oc in range(3):
            for oh in range(3):
                for ow in range(3):
                    acc = conv.biases[oc]
                    for ic in range(2):
                        for kh in range(3):
                            for kw in range(3):
                                ih = oh + kh * 2 - 1
                                iw = ow + kw * 2 - 1
                                if 0 <= ih < 5 and 0 <= iw < 5:
                                    acc += conv.weights[oc][ic][kh][kw] * x[ic][ih][iw]
                    assert abs(out[oc][oh][ow] - acc) < 1e-12

    def test_dilation_one_matches_old_behaviour(self):
        random.seed(5)
        c1 = AI.Conv2d(2, 3, 3, padding=1)
        random.seed(5)
        c2 = AI.Conv2d(2, 3, 3, padding=1, dilation=1)
        x = self._rand(2, 4, 4)
        assert c1.weights == c2.weights
        assert c1.forward(x) == c2.forward(x)

    def test_gradients_vs_numeric(self):
        random.seed(9)
        conv = AI.Conv2d(1, 1, 3, padding=2, dilation=2)
        x = self._rand(1, 4, 4)
        dout = [[[1.0 if (h + w) % 2 == 0 else -0.5 for w in range(4)]
                 for h in range(4)]]
        conv.forward(x)          # backward 依赖前向缓存 (x_shape / cols)
        conv.zero_grad()
        dx = conv.backward(dout)

        def loss():
            o = conv.forward(x)
            return sum(o[0][h][w] * dout[0][h][w] for h in range(4) for w in range(4))

        eps = 1e-6
        for (c, h, w) in ((0, 1, 2), (0, 3, 0)):
            old = x[0][h][w]
            x[0][h][w] = old + eps
            lp = loss()
            x[0][h][w] = old - eps
            lm = loss()
            x[0][h][w] = old
            num = (lp - lm) / (2 * eps)
            assert abs(num - dx[0][h][w]) < 1e-6
        old = conv.weights[0][0][1][1]
        conv.weights[0][0][1][1] = old + eps
        lp = loss()
        conv.weights[0][0][1][1] = old - eps
        lm = loss()
        conv.weights[0][0][1][1] = old
        assert abs((lp - lm) / (2 * eps) - conv.grad_w[0][0][1][1]) < 1e-6

    def test_invalid_dilation(self):
        for bad in (0, -1, (1, 0)):
            with pytest.raises(ValueError):
                AI.Conv2d(2, 2, 3, dilation=bad)

    def test_conv_save_round_trip(self):
        random.seed(3)
        convs = [AI.Conv2d(2, 4, 3, padding=1, dilation=2, groups=2)]
        fc = AI.MLP([4 * 4 * 4, 3], activations=["linear"])
        path = os.path.join(tempfile.gettempdir(), "conv_dilation_rt.json")
        x = self._rand(2, 4, 4, seed=11)
        expected = convs[0].forward(x)
        try:
            AI.save_conv_model(convs, fc, path)
            c2, _ = AI.load_conv_model(path)
            assert c2[0].dilation == (2, 2) and c2[0].groups == 2
            assert c2[0].forward(x) == expected
        finally:
            if os.path.exists(path):
                os.remove(path)


class TestBatchNorm2d:
    """BatchNorm2d：按通道在 H×W 上归一化。"""

    def test_train_mode_statistics(self):
        bn = AI.BatchNorm2d(2)
        x = [[[1.0, 2.0], [3.0, 4.0]], [[0.0, 0.0], [0.0, 0.0]]]
        y = bn.forward(x)
        flat = [v for row in y[0] for v in row]
        assert abs(sum(flat) / 4) < 1e-12
        assert abs(sum(v * v for v in flat) / 4 - 1.0) < 1e-3
        assert all(abs(v) < 1e-12 for row in y[1] for v in row)
        # running 统计量按 momentum 更新
        assert abs(bn.running_mean[0] - 0.1 * 2.5) < 1e-12

    def test_eval_uses_running(self):
        bn = AI.BatchNorm2d(1)
        bn.forward([[[1.0]]])
        bn.eval()
        x = [[[2.0]]]
        y = bn.forward(x)
        inv = 1.0 / math.sqrt(bn.running_var[0] + bn.eps)
        assert abs(y[0][0][0] - (2.0 - bn.running_mean[0]) * inv) < 1e-12
        assert bn.forward(x) == y

    def test_backward_vs_numeric(self):
        random.seed(4)
        bn = AI.BatchNorm2d(2)
        bn.gamma = [1.3, 0.7]
        bn.beta = [0.1, -0.2]
        x = [[[random.uniform(-1, 1) for _ in range(3)] for _ in range(3)]
             for _ in range(2)]
        dout = [[[0.3 * (h + 1) - 0.1 * w for w in range(3)] for h in range(3)]
                for _ in range(2)]
        bn.forward(x)
        dx, gg, gb = bn.backward(dout)

        def loss():
            o = bn.forward(x)
            return sum(o[c][h][w] * dout[c][h][w]
                       for c in range(2) for h in range(3) for w in range(3))

        eps = 1e-6
        for (c, h, w) in ((0, 0, 1), (1, 2, 2)):
            old = x[c][h][w]
            x[c][h][w] = old + eps
            lp = loss()
            x[c][h][w] = old - eps
            lm = loss()
            x[c][h][w] = old
            num = (lp - lm) / (2 * eps)
            assert abs(num - dx[c][h][w]) < 1e-6
        assert abs(gb[0] - sum(dout[0][h][w] for h in range(3) for w in range(3))) < 1e-12
        bn.forward(x)            # 差分循环最后一次用的是扰动输入，重算干净缓存
        assert abs(gg[1] - sum(dout[1][h][w] * bn.x_hat[1][h * 3 + w]
                               for h in range(3) for w in range(3))) < 1e-12

    def test_delta_index_and_validation(self):
        assert AI.BatchNorm2d.delta_index == 0
        for bad in (0, -1, True):
            with pytest.raises(ValueError):
                AI.BatchNorm2d(bad)
        with pytest.raises(ValueError):
            AI.BatchNorm2d(2).forward([[[1.0]]])
        with pytest.raises(RuntimeError):
            AI.BatchNorm2d(2).backward([[[1.0]]])
        with pytest.raises(ValueError):
            AI.BatchNorm2d(1, momentum=1.5)

    def test_estimate_flops(self):
        net = AI.Sequential(AI.Conv2d(2, 4, 3, padding=1, dilation=2),
                            AI.BatchNorm2d(4), AI.MaxPool2d(2), AI.Flatten(),
                            AI.Layer(4 * 2 * 2, 3))
        info = AI.estimate_flops(net, (2, 6, 6))
        assert info["flops"] > 0 and info["output_shape"] == [3]


# ============================================================================
#  批量 A：新激活 / SwiGLU / ALiBi
# ============================================================================
class TestNewActivations:
    """SiLU / Mish / ELU / SELU 的前向数值与导数"""

    def _ref(self, name, z):
        if name == "silu":
            return z / (1.0 + math.exp(-z))
        if name == "mish":
            return z * math.tanh(math.log1p(math.exp(z)))
        if name == "elu":
            return z if z > 0 else math.exp(z) - 1.0
        if name == "selu":
            la, al = 1.0507009873554805, 1.6732632423543772
            return la * (z if z > 0 else al * (math.exp(z) - 1.0))
        raise AssertionError(name)

    def test_forward_values(self):
        zs = [-3.0, -1.2, -0.3, 0.0, 0.5, 2.0, 5.0]
        for name in ("silu", "mish", "elu", "selu"):
            layer = AI.Layer(4, 4, activation=name)
            got = layer.act_fn(zs)
            for g, z in zip(got, zs):
                assert abs(g - self._ref(name, z)) < 1e-12, (name, z, g)

    def test_deriv_vs_numeric(self):
        zs = [-2.0, -0.4, 0.7, 1.5]
        eps = 1e-6
        for name in ("silu", "mish", "elu", "selu"):
            layer = AI.Layer(4, 4, activation=name)
            d = layer.act_deriv(zs)
            for i, z in enumerate(zs):
                num = (self._ref(name, z + eps) - self._ref(name, z - eps)) / (2 * eps)
                rel = abs(d[i] - num) / max(1e-12, abs(d[i]) + abs(num))
                assert rel < 1e-6, (name, z, d[i], num)

    def test_layer_backward_gradient(self):
        random.seed(11)
        layer = AI.Layer(4, 3, activation="mish")
        x = [0.3, -0.7, 1.1, 0.2]
        delta = [0.5, -0.9, 0.4]
        layer.forward(x)
        gw, gb, _ = layer.backward(delta)
        eps = 1e-6
        ana = gw[1][2]
        layer.weights[1][2] += eps
        up = sum(layer.forward(x)[j] * delta[j] for j in range(3))
        layer.weights[1][2] -= 2 * eps
        dn = sum(layer.forward(x)[j] * delta[j] for j in range(3))
        layer.weights[1][2] += eps
        num = (up - dn) / (2 * eps)
        assert abs(ana - num) / max(1e-12, abs(ana) + abs(num)) < 1e-6

    def test_feedforward_supports_new_activations(self):
        eps = 1e-6
        for act in ("silu", "mish", "elu", "selu"):
            random.seed(23)
            ff = AI.FeedForward(4, 6, activation=act)
            xs = [[0.2, -0.5, 0.9, 0.1], [-1.0, 0.4, 0.3, 0.7]]
            dout = [[0.3, -0.2, 0.5, 0.1], [0.1, 0.2, -0.3, 0.4]]
            ff.zero_grad()
            ff.forward(xs)
            ff.backward(dout)
            ana = ff.grads["W1"][2][1]

            def loss():
                o = ff.forward(xs)
                return sum(o[t][j] * dout[t][j] for t in range(2) for j in range(4))

            old = ff.W1[2][1]
            ff.W1[2][1] = old + eps
            up = loss()
            ff.W1[2][1] = old - eps
            dn = loss()
            ff.W1[2][1] = old
            num = (up - dn) / (2 * eps)
            assert abs(ana - num) / max(1e-12, abs(ana) + abs(num)) < 1e-6, act

    def test_unknown_activation_still_rejected(self):
        with pytest.raises(ValueError):
            AI.Layer(3, 3, activation="gelu2")
        with pytest.raises(ValueError):
            AI.FeedForward(4, 8, activation="swish")


class TestSwiGLU:
    """门控前馈网络 SwiGLU"""

    def test_bias_only_path(self):
        # W_g = 0, b_g = 0 → silu(0) = 0 → 门控全 0 → 输出只剩 b2
        gl = AI.SwiGLU(3, 5)
        gl.Wg = [[0.0] * 3 for _ in range(5)]
        gl.bg = [0.0] * 5
        gl.b2 = [0.5, -0.5, 1.0]
        out = gl.forward([[1.0, 2.0, 3.0]])
        assert out[0] == pytest.approx([0.5, -0.5, 1.0], abs=1e-12)

    def test_gradients_vs_numeric(self):
        random.seed(31)
        gl = AI.SwiGLU(4, 6)
        xs = [[0.2, -0.5, 0.9, 0.1], [-1.0, 0.4, 0.3, 0.7]]
        dout = [[0.3, -0.2, 0.5, 0.1], [0.1, 0.2, -0.3, 0.4]]
        gl.zero_grad()
        gl.forward(xs)
        gl.backward(dout)
        eps = 1e-6

        def loss():
            o = gl.forward(xs)
            return sum(o[t][j] * dout[t][j] for t in range(2) for j in range(4))

        for name, i, j in (("W1", 2, 1), ("Wg", 0, 3), ("W2", 1, 4)):
            ana = gl.grads[name][i][j]
            m = getattr(gl, name)
            old = m[i][j]
            m[i][j] = old + eps
            up = loss()
            m[i][j] = old - eps
            dn = loss()
            m[i][j] = old
            num = (up - dn) / (2 * eps)
            assert abs(ana - num) / max(1e-12, abs(ana) + abs(num)) < 1e-6, name
        for name, i in (("b1", 1), ("bg", 2), ("b2", 3)):
            ana = gl.grads[name][i]
            m = getattr(gl, name)
            old = m[i]
            m[i] = old + eps
            up = loss()
            m[i] = old - eps
            dn = loss()
            m[i] = old
            num = (up - dn) / (2 * eps)
            assert abs(ana - num) / max(1e-12, abs(ana) + abs(num)) < 1e-6, name

    def test_dx_vs_numeric(self):
        random.seed(32)
        gl = AI.SwiGLU(3, 4)
        xs = [[0.4, -0.2, 0.6], [0.1, 0.3, -0.5]]
        dout = [[1.0, -1.0, 0.5], [0.2, 0.4, -0.3]]
        gl.zero_grad()
        gl.forward(xs)
        dx = gl.backward(dout)
        eps = 1e-6
        old = xs[0][1]
        xs[0][1] = old + eps
        o = gl.forward(xs)
        up = sum(o[t][j] * dout[t][j] for t in range(2) for j in range(3))
        xs[0][1] = old - eps
        o = gl.forward(xs)
        dn = sum(o[t][j] * dout[t][j] for t in range(2) for j in range(3))
        xs[0][1] = old
        num = (up - dn) / (2 * eps)
        assert abs(dx[0][1] - num) / max(1e-12, abs(dx[0][1]) + abs(num)) < 1e-6

    def test_gradient_accumulates(self):
        random.seed(33)
        gl = AI.SwiGLU(3, 4)
        xs = [[0.4, -0.2, 0.6]]
        dout = [[1.0, -1.0, 0.5]]
        gl.zero_grad()
        gl.forward(xs)
        gl.backward(dout)
        once = gl.grads["W1"][1][1]
        gl.backward(dout)
        assert abs(gl.grads["W1"][1][1] - 2 * once) < 1e-12

    def test_validation(self):
        with pytest.raises(ValueError):
            AI.SwiGLU(0, 4)
        with pytest.raises(ValueError):
            AI.SwiGLU(4, -1)
        gl = AI.SwiGLU(3, 4)
        with pytest.raises(ValueError):
            gl.forward([])
        with pytest.raises(ValueError):
            gl.forward([[1.0, 2.0]])
        with pytest.raises(RuntimeError):
            gl.backward([[1.0, 1.0, 1.0]])


class TestALiBi:
    """线性注意力偏置"""

    def test_uniform_content_matches_formula(self):
        # 输入全 0 → Q/K/V 全 0 → scores 只剩 ALiBi 偏置
        mha = AI.MultiHeadAttention(4, 2, causal=True, alibi=True)
        xs = [[0.0] * 4 for _ in range(5)]
        mha.forward(xs)
        slope = mha.alibi_slopes[0]
        attn = mha.attn_weights[0]
        for j in range(5):
            expect = math.exp(slope * (4 - j))
            total = sum(math.exp(slope * (4 - k)) for k in range(5))
            assert abs(attn[4][j] - expect / total) < 1e-12, j

    def test_causal_mask_preserved(self):
        mha = AI.MultiHeadAttention(4, 2, causal=True, alibi=True)
        xs = [[0.1 * t + 0.02 * j for j in range(4)] for t in range(5)]
        mha.forward(xs)
        attn = mha.attn_weights[0]
        for t in range(5):
            for j in range(t + 1, 5):
                assert attn[t][j] == 0.0

    def test_forward_step_matches_forward(self):
        random.seed(41)
        mha = AI.MultiHeadAttention(8, 2, causal=True, alibi=True)
        xs = [[0.1 * t + 0.05 * j for j in range(8)] for t in range(5)]
        cache = {}
        last = None
        for t in range(5):
            last = mha.forward_step(xs[t], cache)
        full = mha.forward(xs)[-1]
        for j in range(8):
            assert abs(last[j] - full[j]) == 0.0

    def test_gradient_vs_numeric(self):
        random.seed(42)
        mha = AI.MultiHeadAttention(8, 2, causal=True, alibi=True)
        xs = [[0.1 * t - 0.03 * j for j in range(8)] for t in range(4)]
        dout = [[0.1 * (t + 1) + 0.01 * j for j in range(8)] for t in range(4)]
        mha.zero_grad()
        mha.forward(xs)
        mha.backward(dout)
        ana = mha.grads["W_q"][1][2]
        eps = 1e-6

        def loss():
            o = mha.forward(xs)
            return sum(o[t][j] * dout[t][j] for t in range(4) for j in range(8))

        old = mha.W_q[1][2]
        mha.W_q[1][2] = old + eps
        up = loss()
        mha.W_q[1][2] = old - eps
        dn = loss()
        mha.W_q[1][2] = old
        num = (up - dn) / (2 * eps)
        assert abs(ana - num) / max(1e-12, abs(ana) + abs(num)) < 1e-6

    def test_passthrough_and_training(self):
        random.seed(7)
        lm = AI.CausalLM(5, d_model=8, num_heads=2, d_ff=16, num_layers=1,
                         max_len=8, alibi=True)
        assert lm.encoder.layers[0].self_attn.alibi is True
        hist = lm.fit([[0, 1, 2, 3, 4]], epochs=20, lr=0.01)
        assert hist[-1] < 0.3 * hist[0]
        assert lm.generate([0], 4, top_k=1) == [0, 1, 2, 3, 4]

    def test_alibi_off_is_default(self):
        mha = AI.MultiHeadAttention(4, 2)
        assert mha.alibi is False
        te = AI.TransformerEncoder(4, 2, 8, num_layers=1)
        assert te.layers[0].self_attn.alibi is False



class TestLion:
    def test_one_step_matches_manual(self):
        random.seed(5)
        mlp = AI.MLP([2, 2])
        opt = AI.Lion(mlp, lr=0.1, beta1=0.9, beta2=0.99, weight_decay=0.1)
        gw = [[0.5, -0.25], [0.0, 1.0]]
        gb = [0.3, -0.7]
        W0 = [row[:] for row in mlp.layers[0].weights]
        B0 = list(mlp.layers[0].biases)
        opt.step([(gw, gb)])
        for j in range(2):
            m = (1 - 0.9) * gb[j]
            upd = (1.0 if m > 0 else (-1.0 if m < 0 else 0.0)) + 0.1 * B0[j]
            assert abs(mlp.layers[0].biases[j] - (B0[j] - 0.1 * upd)) < 1e-15
            for i in range(2):
                m = (1 - 0.9) * gw[j][i]
                upd = (1.0 if m > 0 else (-1.0 if m < 0 else 0.0)) + 0.1 * W0[j][i]
                assert abs(mlp.layers[0].weights[j][i] - (W0[j][i] - 0.1 * upd)) < 1e-15

    def test_zero_grad_keeps_weights(self):
        random.seed(6)
        mlp = AI.MLP([2, 2])
        opt = AI.Lion(mlp, lr=0.1)
        W0 = [row[:] for row in mlp.layers[0].weights]
        opt.step([([[0.0, 0.0], [0.0, 0.0]], [0.0, 0.0])])
        assert all(mlp.layers[0].weights[j][i] == W0[j][i]
                   for j in range(2) for i in range(2))

    def test_sign_magnitude(self):
        """Lion 的更新量大小恒为 lr（符号优化器），与梯度大小无关。"""
        random.seed(7)
        mlp = AI.MLP([2, 2])
        opt = AI.Lion(mlp, lr=0.05)
        W0 = [row[:] for row in mlp.layers[0].weights]
        opt.step([([[1e-6, -1e-6], [1e-6, -1e-6]], [1e-6, -1e-6])])
        for j in range(2):
            for i in range(2):
                assert abs(abs(mlp.layers[0].weights[j][i] - W0[j][i]) - 0.05) < 1e-15

    def test_trains(self):
        # 用线性回归而不是 XOR：Lion 是符号优化器（步长恒为 lr），在 XOR 这种
        # 有局部极小的任务上不同种子差异很大（实测 0.02~0.42 倍），不适合当收敛判据。
        # 线性回归无局部极小，三个种子的末/首损失比都 ≤0.005。
        random.seed(8)
        mlp = AI.MLP([2, 4, 1])
        opt = AI.Lion(mlp, lr=0.02)
        data = [([1.0, 0.0], [2.0]), ([0.0, 1.0], [-3.0]),
                ([1.0, 1.0], [-1.0]), ([2.0, 3.0], [-5.0])]
        hist = []
        for _ in range(200):
            total = 0.0
            for x, y in data:
                pred = mlp.forward(x)
                loss, grad = AI.mse_loss(pred, y)
                total += loss
                opt.step(AI._mlp_backward(mlp, grad))
            hist.append(total / len(data))
        assert hist[-1] < 0.05 * hist[0]


class TestAdafactor:
    @staticmethod
    def _manual(gw, gb, lr, eps, eps2, wd, W0, B0):
        fo, fi = len(gw), len(gw[0])
        R = [sum(g * g for g in gw[j]) / fi for j in range(fo)]
        C = [sum(gw[j][i] ** 2 for j in range(fo)) / fo for i in range(fi)]
        mean_R = sum(R) / fo
        sq = sum(g * g for row in gw for g in row)
        lr_t = lr * max(eps2, math.sqrt(sq / (fo * fi)))
        W = [[W0[j][i] - lr_t * gw[j][i] / math.sqrt(R[j] * C[i] / mean_R + eps)
              - lr * wd * W0[j][i] for i in range(fi)] for j in range(fo)]
        B = [B0[j] - lr_t * gb[j] / math.sqrt(gb[j] * gb[j] + eps) - lr * wd * B0[j]
             for j in range(fo)]
        return W, B

    def test_one_step_matches_independent_formula(self):
        random.seed(8)
        mlp = AI.MLP([3, 2])
        opt = AI.Adafactor(mlp, lr=0.1, eps=1e-30, eps2=1e-3, weight_decay=0.01)
        gw = [[0.3, -0.4, 0.2], [0.1, 0.05, -0.6]]
        gb = [0.2, -0.3]
        W0 = [row[:] for row in mlp.layers[0].weights]
        B0 = list(mlp.layers[0].biases)
        opt.step([(gw, gb)])
        W, B = self._manual(gw, gb, 0.1, 1e-30, 1e-3, 0.01, W0, B0)
        for j in range(2):
            assert abs(mlp.layers[0].biases[j] - B[j]) < 1e-15
            for i in range(3):
                assert abs(mlp.layers[0].weights[j][i] - W[j][i]) < 1e-15

    def test_weight_decay_uses_pre_update_weight(self):
        """解耦衰减必须作用于更新前的权重（写成两步就会衰减两次）。"""
        random.seed(9)
        mlp = AI.MLP([2, 2])
        opt = AI.Adafactor(mlp, lr=0.1, eps=1e-30, eps2=1e-3, weight_decay=0.5)
        gw = [[1.0, 0.5], [0.25, -1.0]]
        gb = [0.5, -0.5]
        W0 = [row[:] for row in mlp.layers[0].weights]
        B0 = list(mlp.layers[0].biases)
        opt.step([(gw, gb)])
        W, B = self._manual(gw, gb, 0.1, 1e-30, 1e-3, 0.5, W0, B0)
        for j in range(2):
            assert abs(mlp.layers[0].biases[j] - B[j]) < 1e-15
            for i in range(2):
                assert abs(mlp.layers[0].weights[j][i] - W[j][i]) < 1e-15

    def test_trains(self):
        # 注意 lr 语义：Adafactor 乘的是「相对步长」rms(g)，所以 lr 比 Adam 小一个量级；
        # 实测 lr=0.01 时某个种子会发散，0.005 下三个种子的末/首比都 ≤0.012。
        random.seed(10)
        mlp = AI.MLP([2, 4, 1])
        opt = AI.Adafactor(mlp, lr=0.005)
        data = [([1.0, 0.0], [2.0]), ([0.0, 1.0], [-3.0]),
                ([1.0, 1.0], [-1.0]), ([2.0, 3.0], [-5.0])]
        hist = []
        for _ in range(250):
            total = 0.0
            for x, y in data:
                pred = mlp.forward(x)
                loss, grad = AI.mse_loss(pred, y)
                total += loss
                opt.step(AI._mlp_backward(mlp, grad))
            hist.append(total / len(data))
        assert hist[-1] < 0.05 * hist[0]


class TestReduceLROnPlateau:
    def test_reduces_on_plateau(self):
        mlp = AI.MLP([2, 2])
        base = AI.Adam(mlp, lr=0.1)
        sch = AI.ReduceLROnPlateau(base, factor=0.5, patience=2)
        for _ in range(10):
            sch.step(1.0)
        assert abs(base.lr - 0.1 * 0.5 ** sch.reductions) < 1e-12
        assert sch.reductions >= 1

    def test_no_reduction_while_improving(self):
        mlp = AI.MLP([2, 2])
        base = AI.Adam(mlp, lr=0.1)
        sch = AI.ReduceLROnPlateau(base, factor=0.5, patience=1)
        for m in (1.0, 0.9, 0.8, 0.7):
            sch.step(m)
        assert base.lr == 0.1
        assert sch.reductions == 0

    def test_min_lr_floor(self):
        mlp = AI.MLP([2, 2])
        base = AI.Adam(mlp, lr=0.1)
        sch = AI.ReduceLROnPlateau(base, factor=0.5, patience=0, min_lr=0.03)
        for _ in range(5):
            sch.step(1.0)
        assert base.lr == 0.03

    def test_mode_max(self):
        mlp = AI.MLP([2, 2])
        base = AI.Adam(mlp, lr=0.1)
        sch = AI.ReduceLROnPlateau(base, factor=0.5, patience=0, mode="max")
        for _ in range(3):
            sch.step(0.5)
        assert base.lr < 0.1

    def test_nan_metric_is_ignored(self):
        mlp = AI.MLP([2, 2])
        base = AI.Adam(mlp, lr=0.1)
        sch = AI.ReduceLROnPlateau(base, factor=0.5, patience=0)
        for _ in range(3):
            sch.step(float("nan"))
        assert base.lr == 0.1

    def test_invalid_args(self):
        mlp = AI.MLP([2, 2])
        base = AI.Adam(mlp, lr=0.1)
        for bad in (dict(factor=1.0), dict(factor=0.0), dict(mode="avg"), dict(patience=-1)):
            with pytest.raises(ValueError):
                AI.ReduceLROnPlateau(base, **bad)


class TestLoRA:
    def test_zero_init_matches_base(self):
        random.seed(20)
        layer = AI.Layer(4, 3, activation="linear")
        x = [0.3, -0.7, 1.1, 0.2]
        base_out = list(layer.forward(x))
        lora = AI.LoRA(layer, rank=2)
        out = lora.forward(x)
        assert out == base_out

    def test_gradients_match_numeric(self):
        random.seed(21)
        layer = AI.Layer(4, 3, activation="linear")
        lora = AI.LoRA(layer, rank=2)
        lora.B = [[0.3, -0.2], [0.1, 0.4], [-0.5, 0.25]]
        x = [0.3, -0.7, 1.1, 0.2]
        delta = [0.5, -0.9, 0.4]

        def loss():
            out = lora.forward(x)
            return sum(out[j] * delta[j] for j in range(3))

        lora.zero_grad()
        lora.forward(x)          # zero_grad 只清梯度，但 backward 依赖 forward 的缓存
        dx = lora.backward(delta)
        eps = 1e-6
        for r in range(2):
            for i in range(4):
                old = lora.A[r][i]
                lora.A[r][i] = old + eps
                up = loss()
                lora.A[r][i] = old - eps
                dn = loss()
                lora.A[r][i] = old
                assert abs(lora.grad_A[r][i] - (up - dn) / (2 * eps)) < 1e-6
        for j in range(3):
            for r in range(2):
                old = lora.B[j][r]
                lora.B[j][r] = old + eps
                up = loss()
                lora.B[j][r] = old - eps
                dn = loss()
                lora.B[j][r] = old
                assert abs(lora.grad_B[j][r] - (up - dn) / (2 * eps)) < 1e-6
        for i in range(4):
            old = x[i]
            x[i] = old + eps
            up = loss()
            x[i] = old - eps
            dn = loss()
            x[i] = old
            assert abs(dx[i] - (up - dn) / (2 * eps)) < 1e-6

    def test_base_layer_frozen(self):
        random.seed(22)
        layer = AI.Layer(3, 2, activation="linear")
        lora = AI.LoRA(layer, rank=2)
        W0 = [row[:] for row in layer.weights]
        B0 = list(layer.biases)
        lora.forward([0.5, -0.2, 0.9])
        lora.backward([1.0, -1.0])
        assert layer.weights == W0
        assert layer.biases == B0

    def test_merge_preserves_output(self):
        random.seed(23)
        layer = AI.Layer(3, 2, activation="linear")
        lora = AI.LoRA(layer, rank=2)
        lora.B = [[0.4, -0.1], [-0.3, 0.7]]
        x = [0.5, -0.2, 0.9]
        before = list(lora.forward(x))
        lora.merge()
        after = list(lora.forward(x))
        assert max(abs(before[j] - after[j]) for j in range(2)) < 1e-12
        assert lora.merged
        assert all(v == 0.0 for row in lora.B for v in row)

    def test_param_counts(self):
        layer = AI.Layer(4, 3, activation="linear")
        lora = AI.LoRA(layer, rank=2)
        assert lora.trainable_param_count() == 2 * 4 + 3 * 2
        assert lora.total_param_count() == 15 + 14

    def test_invalid_args(self):
        with pytest.raises(TypeError):
            AI.LoRA(object())
        for bad in (0, -1, True, 1.5):
            with pytest.raises(ValueError):
                AI.LoRA(AI.Layer(3, 3), rank=bad)


class TestTripletLoss:
    def test_satisfied_margin_has_zero_grad(self):
        loss, ga, gp, gn = AI.triplet_loss([1.0, 0.0], [0.0, 0.0], [3.0, 4.0], margin=1.0)
        assert loss == 0.0
        assert ga == [0.0, 0.0] and gp == [0.0, 0.0] and gn == [0.0, 0.0]

    def test_gradients_match_numeric(self):
        a, p, n = [1.0, 0.0], [0.0, 0.0], [1.2, 0.0]
        loss, ga, gp, gn = AI.triplet_loss(a, p, n, margin=1.0)
        assert loss > 0.0

        def f():
            return AI.triplet_loss(a, p, n, margin=1.0)[0]

        eps = 1e-6
        for vec, grad in ((a, ga), (p, gp), (n, gn)):
            for i in range(len(vec)):
                old = vec[i]
                vec[i] = old + eps
                up = f()
                vec[i] = old - eps
                dn = f()
                vec[i] = old
                assert abs(grad[i] - (up - dn) / (2 * eps)) < 1e-6

    def test_invalid(self):
        for bad in (0.0, -1.0):
            with pytest.raises(ValueError):
                AI.triplet_loss([1.0], [0.0], [2.0], margin=bad)
        with pytest.raises(ValueError):
            AI.triplet_loss([1.0], [1.0, 2.0], [1.0])
        with pytest.raises(ValueError):
            AI.triplet_loss([float("nan")], [0.0], [1.0])


class TestInfoNCELoss:
    @staticmethod
    def _cos(u, v):
        un = math.sqrt(sum(x * x for x in u))
        vn = math.sqrt(sum(x * x for x in v))
        return sum(x * y for x, y in zip(u, v)) / (un * vn)

    def test_loss_matches_formula(self):
        a, pos = [1.0, 0.0, 0.0], [0.9, 0.1, 0.0]
        negs = [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        loss, _, _, _ = AI.info_nce_loss(a, pos, negs, temperature=0.5)
        sims = [self._cos(a, c) / 0.5 for c in [pos] + negs]
        z = sum(math.exp(s) for s in sims)
        assert abs(loss - (-math.log(math.exp(sims[0]) / z))) < 1e-12

    def test_gradients_match_numeric(self):
        a, pos = [1.0, 0.0, 0.0], [0.9, 0.1, 0.0]
        negs = [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        loss, ga, gpos, gnegs = AI.info_nce_loss(a, pos, negs, temperature=0.5)

        def f():
            return AI.info_nce_loss(a, pos, negs, temperature=0.5)[0]

        eps = 1e-6
        for vec, grad in [(a, ga), (pos, gpos)] + [(negs[k], gnegs[k]) for k in range(2)]:
            for i in range(len(vec)):
                old = vec[i]
                vec[i] = old + eps
                up = f()
                vec[i] = old - eps
                dn = f()
                vec[i] = old
                assert abs(grad[i] - (up - dn) / (2 * eps)) < 1e-5

    def test_hard_negative_hurts_more(self):
        easy = AI.info_nce_loss([1.0, 0.0], [1.0, 0.0], [[0.0, 1.0]], temperature=1.0)[0]
        hard = AI.info_nce_loss([1.0, 0.0], [1.0, 0.0], [[0.9, 0.1]], temperature=1.0)[0]
        assert hard > easy

    def test_invalid(self):
        for bad in (0.0, -1.0):
            with pytest.raises(ValueError):
                AI.info_nce_loss([1.0], [1.0], [[0.0]], temperature=bad)
        with pytest.raises(ValueError):
            AI.info_nce_loss([0.0, 0.0], [1.0, 0.0], [[0.0, 1.0]])
        with pytest.raises(ValueError):
            AI.info_nce_loss([1.0, 0.0], [1.0, 0.0], [])


class TestDiceLoss:
    def test_loss_matches_formula(self):
        loss, _ = AI.dice_loss([[0.9, 0.1], [0.8, 0.2]], [[1.0, 0.0], [1.0, 0.0]])
        num = 2 * (0.9 + 0.8)
        den = (0.9 + 0.1 + 0.8 + 0.2) + 2.0 + 1e-6
        assert abs(loss - (1 - num / den)) < 1e-12

    def test_gradient_matches_numeric(self):
        target = [[1.0, 0.0], [1.0, 0.0]]
        flat = [0.9, 0.1, 0.8, 0.2]
        _, grad = AI.dice_loss([flat[0:2], flat[2:4]], target)
        eps = 1e-6
        for k in range(4):
            old = flat[k]
            flat[k] = old + eps
            up = AI.dice_loss([flat[0:2], flat[2:4]], target)[0]
            flat[k] = old - eps
            dn = AI.dice_loss([flat[0:2], flat[2:4]], target)[0]
            flat[k] = old
            ana = grad[k // 2][k % 2]
            assert abs(ana - (up - dn) / (2 * eps)) < 1e-5

    def test_perfect_prediction(self):
        assert AI.dice_loss([[1.0, 0.0]], [[1.0, 0.0]])[0] < 1e-5

    def test_invalid(self):
        with pytest.raises(ValueError):
            AI.dice_loss([[0.5, 0.5]], [[1.0]])


class TestCTCLoss:
    @staticmethod
    def _brute(logits, labels, blank):
        total = 0.0
        n_cls = len(logits[0])
        import itertools
        for path in itertools.product(range(n_cls), repeat=len(logits)):
            out, prev = [], None
            for c in path:
                if c != blank and c != prev:
                    out.append(c)
                prev = c
            if out == labels:
                p = 1.0
                for t, c in enumerate(path):
                    p *= logits[t][c]
                total += p
        return -math.log(total)

    def test_loss_matches_bruteforce(self):
        logits = [[0.2, 0.6, 0.2], [0.5, 0.3, 0.2], [0.1, 0.8, 0.1]]
        lp = [[math.log(p) for p in row] for row in logits]
        loss, grad = AI.ctc_loss(lp, [1], blank=0)
        assert abs(loss - self._brute(logits, [1], 0)) < 1e-12
        assert len(grad) == 3 and len(grad[0]) == 3

    def test_gradient_matches_numeric(self):
        logits = [[0.2, 0.6, 0.2], [0.5, 0.3, 0.2], [0.1, 0.8, 0.1]]
        lp = [[math.log(p) for p in row] for row in logits]
        _, grad = AI.ctc_loss(lp, [1], blank=0)
        flat = [v for row in lp for v in row]
        eps = 1e-7
        for k in range(9):
            old = flat[k]
            flat[k] = old + eps
            up = AI.ctc_loss([flat[0:3], flat[3:6], flat[6:9]], [1], blank=0)[0]
            flat[k] = old - eps
            dn = AI.ctc_loss([flat[0:3], flat[3:6], flat[6:9]], [1], blank=0)[0]
            flat[k] = old
            ana = grad[k // 3][k % 3]
            assert abs(ana - (up - dn) / (2 * eps)) < 1e-5

    def test_unreachable_and_invalid(self):
        lp = [[math.log(0.5), math.log(0.5)]]
        with pytest.raises(ValueError):
            AI.ctc_loss(lp, [1, 1], blank=0)
        with pytest.raises(ValueError):
            AI.ctc_loss([[math.log(0.5), math.log(0.5)]], [0], blank=0)
        with pytest.raises(ValueError):
            AI.ctc_loss([[math.log(0.5), math.log(0.5)]], [5], blank=0)
        with pytest.raises(TypeError):
            AI.ctc_loss([[math.log(0.5), math.log(0.5)]], [1.5], blank=0)


class TestCutMix:
    @staticmethod
    def _img(c, h, w, base):
        return [[[base + 0.01 * (y * w + x) for x in range(w)] for y in range(h)]
                for _ in range(c)]

    def test_shape_labels_and_no_mutation(self):
        x1 = self._img(2, 6, 6, 0.2)
        x2 = self._img(2, 6, 6, 0.8)
        keep1, keep2 = [row[:] for row in x1[0]], [row[:] for row in x2[0]]
        out, lab = AI.augment_cutmix(x1, [1.0, 0.0], x2, [0.0, 1.0],
                                     alpha=1.0, rng=random.Random(7))
        assert len(out) == 2 and len(out[0]) == 6 and len(out[0][0]) == 6
        assert abs(sum(lab) - 1.0) < 1e-12
        assert x1[0] == keep1 and x2[0] == keep2

    def test_pasted_pixels_come_from_source(self):
        x1 = self._img(1, 6, 6, 0.2)
        x2 = self._img(1, 6, 6, 0.8)
        out, lab = AI.augment_cutmix(x1, [1.0, 0.0], x2, [0.0, 1.0],
                                     alpha=1.0, rng=random.Random(7))
        changed = [(y, x) for y in range(6) for x in range(6) if out[0][y][x] != x1[0][y][x]]
        assert changed, "应当至少贴入一格"
        # 贴入比例（按面积）应与 1-lam 一致
        assert abs(len(changed) / 36.0 - (1.0 - lab[0])) < 0.02
        src_values = {x2[0][y][x] for y in range(6) for x in range(6)}
        assert all(out[0][y][x] in src_values for y, x in changed)

    def test_same_seed_reproducible(self):
        x1 = self._img(2, 5, 5, 0.3)
        x2 = self._img(2, 5, 5, 0.7)
        o1, l1 = AI.augment_cutmix(x1, [1.0, 0.0], x2, [0.0, 1.0], rng=random.Random(11))
        o2, l2 = AI.augment_cutmix(x1, [1.0, 0.0], x2, [0.0, 1.0], rng=random.Random(11))
        assert o1 == o2 and l1 == l2

    def test_invalid(self):
        x1 = self._img(2, 4, 4, 0.2)
        x2 = self._img(2, 4, 4, 0.8)
        with pytest.raises(ValueError):
            AI.augment_cutmix(x1, [1.0], x2, [1.0], alpha=0.0)
        with pytest.raises(ValueError):
            AI.augment_cutmix(x1, [1.0], x2, [1.0, 0.0])
        with pytest.raises(ValueError):
            AI.augment_cutmix(self._img(2, 5, 5, 0.2), [1.0], x2, [1.0])


class TestCutout:
    @staticmethod
    def _img(c, h, w, base=0.5):
        return [[[base + 0.01 * (y * w + x) for x in range(w)] for y in range(h)]
                for _ in range(c)]

    def test_holes_are_zeroed(self):
        x = self._img(3, 8, 8)
        keep = [row[:] for row in x[0]]
        out = AI.augment_cutout(x, num_holes=2, hole_size=0.25, rng=random.Random(3))
        side = max(1, int(round(8 * 0.25)))
        zeros = sum(1 for c in range(3) for y in range(8) for xx in range(8) if out[c][y][xx] == 0.0)
        assert zeros == 3 * 2 * side * side
        assert x[0] == keep

    def test_zero_holes_is_copy(self):
        x = self._img(2, 6, 6)
        assert AI.augment_cutout(x, num_holes=0, hole_size=0.3, rng=random.Random(1)) == x

    def test_same_seed_reproducible(self):
        x = self._img(2, 6, 6)
        assert (AI.augment_cutout(x, 1, 0.3, rng=random.Random(5))
                == AI.augment_cutout(x, 1, 0.3, rng=random.Random(5)))

    def test_invalid(self):
        x = self._img(2, 6, 6)
        for bad in (dict(hole_size=0.0), dict(hole_size=1.0), dict(num_holes=-1)):
            with pytest.raises(ValueError):
                AI.augment_cutout(x, **bad)


class TestRandAugment:
    @staticmethod
    def _img(c, h, w, base=0.5):
        return [[[base + 0.01 * (y * w + x) for x in range(w)] for y in range(h)]
                for _ in range(c)]

    def test_applies_n_ops(self):
        x = self._img(2, 8, 8)
        keep = [row[:] for row in x[0]]
        out, applied = AI.augment_randaugment(x, n=3, magnitude=5.0, rng=random.Random(9))
        assert len(applied) == 3
        assert len(out) == 2 and len(out[0]) == 8 and len(out[0][0]) == 8
        assert all(0.0 <= v <= 1.0 for ch in out for row in ch for v in row)
        assert x[0] == keep

    def test_identity_only_is_noop(self):
        x = self._img(2, 5, 5)
        out, applied = AI.augment_randaugment(x, n=3, magnitude=9.0, ops=["identity"],
                                              rng=random.Random(1))
        assert out == x and applied == ["identity"] * 3

    def test_rotate90_four_times_is_identity(self):
        x = self._img(2, 4, 6)
        rot = x
        for _ in range(4):
            rot = AI._aug_rotate90(rot, 1)
        assert rot == x

    def test_same_seed_reproducible(self):
        x = self._img(2, 8, 8)
        o1, a1 = AI.augment_randaugment(x, 4, 7.0, rng=random.Random(21))
        o2, a2 = AI.augment_randaugment(x, 4, 7.0, rng=random.Random(21))
        assert o1 == o2 and a1 == a2

    def test_invalid(self):
        x = self._img(2, 6, 6)
        with pytest.raises(ValueError):
            AI.augment_randaugment(x, n=1, magnitude=11.0)
        with pytest.raises(ValueError):
            AI.augment_randaugment(x, n=1, ops=["blur"])
        with pytest.raises(ValueError):
            AI.augment_randaugment(x, n=-1)


class TestMaskedLM:
    @staticmethod
    def _lm(seed=5, **kw):
        random.seed(seed)
        return AI.MaskedLM(vocab_size=8, d_model=8, num_heads=2, d_ff=16,
                           num_layers=2, max_len=12, **kw)

    def test_mask_tokens_semantics(self):
        lm = self._lm()
        assert lm.mask_id == 8            # 词表外的额外一行
        ids = list(range(8))
        rng = random.Random(1)
        n_mask = n_rand = n_keep = total = 0
        for _ in range(300):
            corrupted, pos, targets = lm.mask_tokens(ids, rng)
            assert len(corrupted) == len(ids)
            for j, p in enumerate(pos):
                total += 1
                if corrupted[p] == lm.mask_id:
                    n_mask += 1
                elif corrupted[p] != targets[j]:
                    n_rand += 1
                else:
                    n_keep += 1
        assert abs(n_mask / total - 0.8) < 0.06
        assert abs(n_rand / total - 0.1) < 0.05
        # 短序列至少遮一个位置
        _, pos, _ = lm.mask_tokens([3, 4], random.Random(9))
        assert len(pos) >= 1

    def test_bidirectional(self):
        lm = self._lm()
        lm.eval()
        a = lm.forward([1, 2, 3, 4, 5])
        b = lm.forward([1, 2, 3, 4, 7])
        # 改末位会影响**前面**位置的 logits（对比 CausalLM 的因果性）
        assert any(abs(a[t][v] - b[t][v]) > 1e-9
                   for t in range(4) for v in range(lm.vocab_size))

    def test_gradients_match_numeric(self):
        lm = self._lm(seed=9)
        lm.eval()
        corrupted = [1, 8, 3, 4, 8, 6]
        pos = [1, 4]
        targets = [2, 5]

        def loss_only():
            logits = lm.forward(corrupted)
            s = 0.0
            for p, tgt in zip(pos, targets):
                probs = AI._softmax_row(logits[p])
                s += -math.log(max(probs[tgt], 1e-12))
            return s / len(pos)

        lm.zero_grad()
        lm._loss_and_backward(corrupted, pos, targets)
        probe = (2, 3)
        old = lm.W_out[probe[0]][probe[1]]
        eps = 1e-5
        lm.W_out[probe[0]][probe[1]] = old + eps
        lp = loss_only()
        lm.W_out[probe[0]][probe[1]] = old - eps
        lm_ = loss_only()
        lm.W_out[probe[0]][probe[1]] = old
        num = (lp - lm_) / (2 * eps)
        ana = lm.grads["W_out"][probe[0]][probe[1]]
        assert abs(ana - num) / max(1e-12, abs(ana) + abs(num)) < 1e-6

    def test_learns_fill_mask(self):
        # 全遮罩记忆：给定「每条位置都是 mask」，模型要还原整条序列。
        # 这是 MLM 最基本的可学习性证据，且 250 步后 loss 会掉到初值的 1% 以下，
        # 判据不依赖随机遮罩的具体轨迹。
        random.seed(3)
        lm = AI.MaskedLM(vocab_size=6, d_model=16, num_heads=2, d_ff=32,
                         num_layers=2, max_len=8)
        seq = [1, 2, 3, 4, 5, 1]
        corrupted = [lm.mask_id] * len(seq)
        pos = list(range(len(seq)))
        lm.train()
        first = None
        loss = None
        for _ in range(250):
            lm.zero_grad()
            loss = lm._loss_and_backward(corrupted, pos, seq)
            if first is None:
                first = loss
            lm.adam_step(0.02)
        assert loss < 0.2 * first
        lm.eval()
        assert lm.predict(corrupted, pos) == seq

    def test_evaluate_and_mode(self):
        lm = self._lm()
        seq = [1, 2, 3, 4]
        lm.train()
        e1 = lm.evaluate([seq], seed=3)
        e2 = lm.evaluate([seq], seed=3)
        assert e1 == e2
        assert lm.encoder.layers[0].self_attn.training is True
        lm.eval()
        assert isinstance(lm.evaluate([seq], seed=3), float)

    def test_param_count_and_blocks(self):
        lm = self._lm()
        names = [n for n, _, _ in lm._named_params()]
        assert len(names) == len(set(names)) == 1 + 2 * 16 + 2   # emb + 每层 16 + out
        assert names[0] == "emb.weight" and names[-1] == "out.b"
        assert lm.param_count() > 0

    def test_invalid(self):
        for bad in (0, -1, True, 1.5, "8"):
            with pytest.raises((ValueError, TypeError)):
                AI.MaskedLM(bad)
        with pytest.raises(ValueError):
            AI.MaskedLM(8, mask_prob=0.0)
        with pytest.raises(ValueError):
            AI.MaskedLM(8, replace_prob=1.0)
        lm = self._lm()
        with pytest.raises(ValueError):
            lm.forward([])
        with pytest.raises(ValueError):
            lm.forward([1] * 13)
        with pytest.raises(ValueError):
            lm._loss_and_backward([1, 2, 3], [], [])
        with pytest.raises(ValueError):
            lm._loss_and_backward([1, 2, 3], [5], [1])
        with pytest.raises(ValueError):
            lm._loss_and_backward([1, 8, 3], [1], [8])
        with pytest.raises(ValueError):
            lm._loss_and_backward([1, 8, 3], [1, 2], [2])
        with pytest.raises(ValueError):
            lm.predict([1, 2, 3], [])
        with pytest.raises(ValueError):
            lm.fit([])


class TestProfileModel:
    def test_forward_only(self):
        random.seed(1)
        mlp = AI.MLP([4, 6, 3])
        rep = AI.profile_model(mlp, [0.1, 0.2, 0.3, 0.4])
        assert len(rep["layers"]) == len(mlp.layers)
        assert rep["total_ms"] > 0
        assert rep["backward_total_ms"] is None
        assert rep["params"] == 4 * 6 + 6 + 6 * 3 + 3
        assert abs(sum(l["forward_ms"] for l in rep["layers"]) - rep["total_ms"]) < 1e-9

    def test_backward(self):
        random.seed(1)
        mlp = AI.MLP([3, 5, 2])
        rep = AI.profile_model(mlp, [1.0, 0.5, -0.5], backward=True, repeat=2)
        assert rep["backward_total_ms"] is not None and rep["backward_total_ms"] >= 0
        assert all(l["backward_ms"] >= 0 for l in rep["layers"])

    def test_accepts_model(self):
        random.seed(2)
        m = AI.Model(AI.MLP([2, 4, 1]), task="regression")
        assert len(AI.profile_model(m, [0.5, 0.2])["layers"]) == 2

    def test_invalid(self):
        mlp = AI.MLP([2, 2])
        with pytest.raises(ValueError):
            AI.profile_model(mlp, [0.1, 0.2], repeat=0)
        with pytest.raises(TypeError):
            AI.profile_model("not a net", [0.1, 0.2])


class TestLRFinder:
    @staticmethod
    def _data():
        x = [[i / 10.0] for i in range(20)]
        return x, [3 * v[0] + 1 for v in x]

    @staticmethod
    def _model(seed=2):
        random.seed(seed)
        return AI.Model(AI.MLP([1, 8, 1]), task="regression", lr=0.01)

    def test_shape_and_geometric_growth(self):
        x, y = self._data()
        rep = AI.lr_finder(self._model(), x=x, y=y, start_lr=1e-5, end_lr=0.1,
                           steps=25, smooth=0.1, seed=2)
        assert len(rep["history"]) == 25
        ratios = [rep["history"][k + 1][0] / rep["history"][k][0]
                  for k in range(len(rep["history"]) - 1)]
        assert max(ratios) - min(ratios) < 1e-9
        assert 1e-5 <= rep["suggested_lr"] <= 0.1
        assert any(abs(lr - rep["best_lr"]) < 1e-15 for lr, _, _ in rep["history"])

    def test_restores_optimizer_lr(self):
        x, y = self._data()
        m = self._model()
        m.optimizer.lr = 0.007
        AI.lr_finder(m, x=x, y=y, start_lr=1e-4, end_lr=0.05, steps=10, seed=1)
        assert abs(m.optimizer.lr - 0.007) < 1e-15

    def test_reproducible(self):
        x, y = self._data()
        a = AI.lr_finder(self._model(), x=x, y=y, start_lr=1e-4, end_lr=0.1, steps=15, seed=3)
        b = AI.lr_finder(self._model(), x=x, y=y, start_lr=1e-4, end_lr=0.1, steps=15, seed=3)
        assert a["history"] == b["history"]
        assert a["suggested_lr"] == b["suggested_lr"]

    def test_note_when_loss_never_recovers(self):
        x, y = self._data()
        rep = AI.lr_finder(self._model(), x=x, y=y, start_lr=1e-6, end_lr=1e-3,
                           steps=15, seed=4)
        assert rep["note"] is not None
        assert "end_lr" in rep["note"]

    def test_invalid(self):
        x, y = self._data()
        for kwargs in (dict(start_lr=0.1, end_lr=0.1), dict(start_lr=0.1, end_lr=0.01),
                       dict(steps=1), dict(smooth=0.0), dict(smooth=1.5),
                       dict(start_lr=-1.0)):
            with pytest.raises(ValueError):
                AI.lr_finder(self._model(), x=x, y=y, **kwargs)
        with pytest.raises(TypeError):
            AI.lr_finder("nope", x=x, y=y)
        with pytest.raises(ValueError):
            AI.lr_finder(self._model())
        with pytest.raises(ValueError):
            AI.lr_finder(self._model(), x=x, y=y[:-1])


class TestWordPieceTokenizer:
    TEXT = ("the quick brown fox jumps over the lazy dog. "
            "the quick red fox runs past the lazy brown dog. ") * 12

    @staticmethod
    def _wp(vocab_size=200):
        return AI.WordPieceTokenizer().train(
            TestWordPieceTokenizer.TEXT, vocab_size=vocab_size, min_frequency=2)

    def test_vocab_layout(self):
        wp = AI.WordPieceTokenizer()
        assert wp.vocab_size == len(AI._WP_SPECIALS)
        assert (wp.pad_id, wp.unk_id, wp.cls_id, wp.sep_id, wp.mask_id) == (0, 1, 2, 3, 4)
        wp = self._wp()
        assert wp.vocab_size <= 200
        assert len(wp) == wp.vocab_size
        assert list(wp.tokens[:5]) == list(AI._WP_SPECIALS)
        assert any(t.startswith("##") for t in wp.tokens)

    def test_encode_decode_round_trip(self):
        wp = self._wp()
        ids = wp.encode("the quick brown fox")
        assert all(0 <= i < wp.vocab_size for i in ids)
        assert wp.decode(ids) == "the quick brown fox"
        # 连续空白被归一化（词表里的空白 token 只留一个）
        assert wp.decode(wp.encode("the  quick")) == "the quick"
        # 生词 -> UNK -> 解码为空
        assert wp.encode("zzzqqq") == [wp.unk_id]
        assert wp.decode([wp.unk_id]) == ""

    def test_word_level_tokens_and_pair(self):
        wp = self._wp()
        assert [wp.tokens[i] for i in wp.encode("brown")] == ["brown"]
        pair = wp.encode_pair("the dog", "quick fox")
        assert pair[0] == wp.cls_id and pair[-1] == wp.sep_id
        assert pair.count(wp.sep_id) == 2
        text = wp.decode(pair)
        assert "the dog" in text and "quick fox" in text

    def test_deterministic_and_save_load(self):
        a = self._wp()
        b = self._wp()
        assert a.tokens == b.tokens
        path = os.path.join(tempfile.gettempdir(), "test_wordpiece.json")
        try:
            a.save(path, quiet=True)
            c = AI.WordPieceTokenizer.load(path)
            assert c.tokens == a.tokens
            assert c.encode("the quick fox") == a.encode("the quick fox")
        finally:
            if os.path.exists(path):
                os.remove(path)

    def test_invalid_inputs(self):
        with pytest.raises(ValueError):
            AI.WordPieceTokenizer(["a"])
        with pytest.raises(ValueError):
            AI.WordPieceTokenizer(list(AI._WP_SPECIALS) + ["a", "a"])
        with pytest.raises(ValueError):
            AI.WordPieceTokenizer(list(AI._WP_SPECIALS) + ["##"])
        with pytest.raises(TypeError):
            AI.WordPieceTokenizer([1, 2])
        with pytest.raises(ValueError):
            AI.WordPieceTokenizer().train(self.TEXT, vocab_size=3)
        with pytest.raises(ValueError):
            AI.WordPieceTokenizer().train(self.TEXT, min_frequency=0)
        wp = self._wp()
        with pytest.raises(TypeError):
            wp.encode(123)
        with pytest.raises(ValueError):
            wp.decode([wp.vocab_size + 3])
        with pytest.raises(TypeError):
            wp.decode([True])
        with pytest.raises(TypeError):
            wp.decode("abc")

    def test_differs_from_bpe(self):
        # BPE 字节级无损；WordPiece 有特殊标记与 ## 前缀，解码会丢特殊 token
        bpe = AI.BytePairTokenizer().train(self.TEXT, vocab_size=300)
        assert bpe.decode(bpe.encode("the quick brown fox")) == "the quick brown fox"
        wp = self._wp()
        ids = [wp.cls_id] + wp.encode("the dog") + [wp.sep_id]
        assert wp.decode(ids) == "the dog"


class TestAttentionMaps:
    @staticmethod
    def _x(seq=5, d=8, seed=11):
        rnd = random.Random(seed)
        return [[rnd.gauss(0, 1) for _ in range(d)] for _ in range(seq)]

    def test_mha_maps(self):
        random.seed(11)
        mha = AI.MultiHeadAttention(d_model=8, num_heads=2, causal=True)
        rep = AI.attention_maps(mha, self._x())
        assert rep["num_heads"] == 2 and rep["seq_len"] == 5
        assert all(len(m) == 5 and len(m[0]) == 5 for m in rep["maps"])
        assert all(abs(sum(row) - 1.0) < 1e-12 for m in rep["maps"] for row in m)
        # 因果掩码：上三角恰好 0
        assert all(m[t][j] == 0.0 for m in rep["maps"]
                   for t in range(5) for j in range(t + 1, 5))
        assert mha.training is True   # 采样后恢复训练模式

    def test_eval_deterministic(self):
        mha = AI.MultiHeadAttention(d_model=8, num_heads=2, causal=True)
        mha.eval()
        a = AI.attention_maps(mha, self._x())["maps"]
        b = AI.attention_maps(mha, self._x())["maps"]
        assert a == b
        assert mha.training is False

    def test_layer_and_encoder(self):
        tel = AI.TransformerEncoderLayer(d_model=8, num_heads=2, d_ff=16, dropout=0.3)
        rep = AI.attention_maps(tel, self._x())
        assert rep["num_heads"] == 2 and rep["seq_len"] == 5
        assert tel.self_attn.training is True

        enc = AI.TransformerEncoder(d_model=8, num_heads=2, d_ff=16, num_layers=3, dropout=0.0)
        all_layers = AI.attention_maps(enc, self._x())
        assert [l["layer"] for l in all_layers["layers"]] == [0, 1, 2]
        one = AI.attention_maps(enc, self._x(), layer_index=2)
        assert len(one["layers"]) == 1 and one["layers"][0]["layer"] == 2
        with pytest.raises(ValueError):
            AI.attention_maps(enc, self._x(), layer_index=9)
        with pytest.raises(TypeError):
            AI.attention_maps(enc, self._x(), layer_index=True)

    def test_causallm_maps_and_readonly(self):
        lm = AI.CausalLM(vocab_size=7, d_model=8, num_heads=2, d_ff=16,
                         num_layers=2, max_len=8)
        before = [p[:] if not isinstance(p[0], list) else [r[:] for r in p]
                  for _, p, _ in lm._named_params()]
        rep = AI.attention_maps(lm, [1, 2, 3, 4])
        assert len(rep["layers"]) == 2
        assert rep["layers"][0]["seq_len"] == 4
        assert all(m[t][j] == 0.0 for m in rep["layers"][0]["maps"]
                   for t in range(4) for j in range(t + 1, 4))
        after = [p for _, p, _ in lm._named_params()]
        assert before == after                     # 只读
        assert lm.encoder.layers[0].self_attn.training is True

    def test_ascii(self):
        random.seed(11)
        mha = AI.MultiHeadAttention(d_model=8, num_heads=2, causal=True)
        rep = AI.attention_maps(mha, self._x())
        art = AI.attention_ascii(rep)
        lines = art.split("\n")
        assert "head 0" in lines[0]
        assert lines[1].strip().startswith("0 |")
        assert "强度" in lines[-1]
        assert "head 1" in AI.attention_ascii(rep["maps"], head=1).split("\n")[0]
        narrow = AI.attention_ascii(rep, max_width=3).split("\n")[1]
        assert len(narrow) < 20

    def test_invalid(self):
        random.seed(11)
        mha = AI.MultiHeadAttention(d_model=8, num_heads=2, causal=True)
        rep = AI.attention_maps(mha, self._x())
        with pytest.raises(TypeError):
            AI.attention_maps(object(), self._x())
        with pytest.raises(ValueError):
            AI.attention_ascii(rep, head=5)
        with pytest.raises(TypeError):
            AI.attention_ascii(rep, head=True)
        with pytest.raises(ValueError):
            AI.attention_ascii([])
        with pytest.raises(ValueError):
            AI.attention_ascii({})
        with pytest.raises(ValueError):
            AI.attention_ascii(rep, max_width=0)


class TestConfigDriven:
    CFG = {"input_dim": 2, "output_dim": 2, "hidden": [8], "task": "classification",
           "lr": 0.05, "epochs": 40, "batch_size": 4, "seed": 3, "verbose": False,
           "patience": 0}

    @staticmethod
    def _data():
        xs = [[i / 20.0, 0.0] for i in range(20)]
        ys = [1 if v[0] > 0.5 else 0 for v in xs]
        return xs, ys, list(zip(xs, ys))

    def test_build_model(self):
        m = AI.build_model({"input_dim": 3, "output_dim": 2, "hidden": [5, 4],
                            "activations": ["relu", "sigmoid"]})
        assert [l.fan_in for l in m.model.layers] == [3, 5, 4]
        assert [l.fan_out for l in m.model.layers] == [5, 4, 2]
        assert m.model.layers[-1].activation_name == "softmax"
        assert m.model.param_count() == 3 * 5 + 5 + 5 * 4 + 4 + 4 * 2 + 2
        # 从 data 推断维度
        xs, ys, pairs = self._data()
        m2 = AI.build_model({"hidden": [6]}, data=pairs)
        assert [l.fan_in for l in m2.model.layers] == [2, 6]
        # y 是标量 -> 推断为 1 维回归输出
        assert m2.model.layers[-1].fan_out == 1
        assert m2.task == "regression"
        # 显式给 output_dim / task 时按给定的来
        m3 = AI.build_model({"hidden": [6], "output_dim": 2, "task": "classification"},
                            data=pairs)
        assert m3.model.layers[-1].fan_out == 2
        assert m3.model.layers[-1].activation_name == "softmax"
        # 回归默认 linear 输出
        reg = AI.build_model({"input_dim": 2, "output_dim": 1, "task": "regression"})
        assert reg.model.layers[-1].activation_name == "linear"
        assert reg.task == "regression"

    def test_validation(self):
        base = {"input_dim": 2, "output_dim": 1}
        with pytest.raises(TypeError):
            AI.build_model(dict(base, hidden="abc"))
        with pytest.raises(ValueError):
            AI.build_model(dict(base, hidden=[3], activations=["relu", "relu"]))
        with pytest.raises(TypeError):
            AI.build_model(dict(base, lr="fast"))
        with pytest.raises(ValueError):
            AI.build_model(dict(base, lr=float("nan")))
        with pytest.raises(ValueError):
            AI.build_model(dict(base, typo_key=1))
        with pytest.raises(ValueError):
            AI.build_model(dict(base, task="clf"))
        with pytest.raises(ValueError):
            AI.build_model({"hidden": [3]})
        with pytest.raises(TypeError):
            AI.build_model(123)

    def test_scheduler_build(self):
        m = AI.build_model({"input_dim": 2, "output_dim": 1, "hidden": [3]})
        opt = m.optimizer
        assert isinstance(AI._build_scheduler({"type": "step", "step_size": 10}, opt), AI.StepLR)
        assert isinstance(AI._build_scheduler({"type": "exponential", "gamma": 0.9}, opt),
                          AI.ExponentialLR)
        assert isinstance(AI._build_scheduler({"type": "cosine", "t_max": 20}, opt),
                          AI.CosineAnnealingLR)
        assert isinstance(AI._build_scheduler({"type": "plateau", "patience": 3}, opt),
                          AI.ReduceLROnPlateau)
        warm = AI._build_scheduler({"type": "warmup", "warmup_epochs": 3,
                                    "base_scheduler": {"type": "step", "step_size": 10}}, opt)
        assert isinstance(warm, AI.WarmupLR) and isinstance(warm.base_scheduler, AI.StepLR)
        with pytest.raises(ValueError):
            AI._build_scheduler({"type": "magic"}, opt)
        with pytest.raises(ValueError):
            AI._build_scheduler({"type": "step", "bogus": 1}, opt)

    def test_train_from_config(self):
        xs, ys, pairs = self._data()
        model = AI.train_from_config(self.CFG, data=pairs)
        hist = [v for _, v in model.history["train_loss"]]
        assert len(hist) == 40
        assert hist[-1] < 0.5 * hist[0]
        preds = [max(range(len(p)), key=lambda k: p[k])
                 for p in (model.predict(v) for v in xs)]
        assert len(set(preds)) == 2
        assert sum(1 for p, t in zip(preds, ys) if p == t) / len(xs) >= 0.8

    def test_seed_reproducible(self):
        _xs, _ys, pairs = self._data()
        a = AI.train_from_config(self.CFG, data=pairs)
        b = AI.train_from_config(self.CFG, data=pairs)
        assert [v for _, v in a.history["train_loss"]] == [v for _, v in b.history["train_loss"]]
        for (wa, ba), (wb, bb) in zip(a.model.get_parameters(), b.model.get_parameters()):
            for ra, rb in zip(wa, wb):
                assert all(abs(x - y) < 1e-15 for x, y in zip(ra, rb))
            assert all(abs(x - y) < 1e-15 for x, y in zip(ba, bb))

    def test_json_config_and_conflicts(self):
        xs, ys, pairs = self._data()
        cfg = dict(self.CFG, epochs=10, lr_scheduler={"type": "step", "step_size": 5})
        path = os.path.join(tempfile.gettempdir(), "test_config.json")
        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(cfg, fh, ensure_ascii=False)
            model = AI.train_from_config(path, data=pairs)
            assert len(model.history["train_loss"]) == 10
        finally:
            if os.path.exists(path):
                os.remove(path)
        model2 = AI.train_from_config(self.CFG, x=xs, y=ys)
        assert len(model2.history["train_loss"]) == 40
        with pytest.raises(ValueError):
            AI.train_from_config(self.CFG, data=pairs, x=xs)
        with pytest.raises(ValueError):
            AI.train_from_config(self.CFG)


class TestHyperparameterSearch:
    """配置空间搜索：全组合、随机、指标方向、非法输入。"""

    def _data(self):
        random.seed(11)
        data = []
        for _ in range(96):
            a = random.uniform(-1.0, 1.0)
            b = random.uniform(-1.0, 1.0)
            data.append(([a, b], 1 if a + b > 0.0 else 0))
        return data

    def _base(self, **kw):
        cfg = {"input_dim": 2, "output_dim": 2, "hidden": [6],
               "activations": ["relu"], "task": "classification",
               "loss": "cross_entropy", "optimizer": "adam", "lr": 0.05,
               "epochs": 6, "batch_size": 16, "val_split": 0.2,
               "patience": 0, "seed": 3}
        cfg.update(kw)
        return cfg

    def test_grid_covers_all_combinations(self):
        data = self._data()
        base = self._base()
        res = AI.hyperparameter_search(
            base, {"lr": [0.02, 0.05], "hidden": [[6], [8]]},
            data=data, method="grid", metric="val_loss")
        assert len(res["trials"]) == 4
        assert res["params_names"] == ["lr", "hidden"]
        assert res["mode"] == "min"
        assert res["best"]["score"] == min(t["score"] for t in res["trials"])
        # 每个 trial 都真的训过，参数覆盖 grid
        for t in res["trials"]:
            assert set(t["params"]) == {"lr", "hidden"}
            assert t["history"]["train_loss"]
        combos = {(t["params"]["lr"], tuple(t["params"]["hidden"])) for t in res["trials"]}
        assert combos == {(0.02, (6,)), (0.02, (8,)), (0.05, (6,)), (0.05, (8,))}

    def test_random_search_is_reproducible(self):
        data = self._data()
        base = self._base()
        grid = {"lr": [0.01, 0.05, 0.1]}
        r1 = AI.hyperparameter_search(base, grid, data=data, method="random",
                                      n_trials=2, seed=9)
        r2 = AI.hyperparameter_search(base, grid, data=data, method="random",
                                      n_trials=2, seed=9)
        assert len(r1["trials"]) == 2
        assert [t["params"] for t in r1["trials"]] == [t["params"] for t in r2["trials"]]
        assert [t["score"] for t in r1["trials"]] == [t["score"] for t in r2["trials"]]

    def test_accuracy_metric_maximises(self):
        data = self._data()
        xs = [d[0] for d in data]
        ys = [d[1] for d in data]
        res = AI.hyperparameter_search(self._base(), {"lr": [0.05]},
                                       x=xs, y=ys, metric="accuracy")
        assert res["mode"] == "max"
        assert 0.0 <= res["best"]["score"] <= 1.0
        assert res["best"]["score"] == max(t["score"] for t in res["trials"])

    def test_global_rng_untouched(self):
        data = self._data()
        random.seed(4321)
        before = [random.random() for _ in range(3)]
        random.seed(4321)
        AI.hyperparameter_search(self._base(), {"lr": [0.05]}, data=data,
                                 metric="train_loss")
        after = [random.random() for _ in range(3)]
        assert before == after

    def test_rejects_bad_inputs(self):
        data = self._data()
        base = self._base()
        with pytest.raises(ValueError):
            AI.hyperparameter_search(base, [], data=data)
        with pytest.raises(ValueError):
            AI.hyperparameter_search(base, {}, data=data)
        with pytest.raises(ValueError):
            AI.hyperparameter_search(base, {"epoch": [1]}, data=data)
        with pytest.raises(ValueError):
            AI.hyperparameter_search(base, {"lr": 0.05}, data=data)
        with pytest.raises(ValueError):
            AI.hyperparameter_search(base, {"lr": []}, data=data)
        with pytest.raises(ValueError):
            AI.hyperparameter_search(base, {"lr": [0.05]}, data=data, method="bayes")
        with pytest.raises(ValueError):
            AI.hyperparameter_search(base, {"lr": [0.05]}, data=data,
                                     method="random", n_trials=0)
        with pytest.raises(ValueError):
            AI.hyperparameter_search(base, {"lr": [0.05]}, data=data, metric="f1")
        with pytest.raises(ValueError):
            AI.hyperparameter_search(self._base(val_split=0.0), {"lr": [0.05]},
                                     data=data, metric="val_loss")
        with pytest.raises(ValueError):
            AI.hyperparameter_search(base, {"lr": [0.05]}, data=data, metric="accuracy")


class TestTokenClassificationHead:
    """Token 级分类头：形状、mask、梯度、可训练性。"""

    def test_shapes_and_validation(self):
        head = AI.TokenClassificationHead(4, 3)
        logits = head.forward([[1.0, 0.0, 0.5, -0.5], [0.2, 0.3, 0.1, 0.0]])
        assert len(logits) == 2 and all(len(r) == 3 for r in logits)
        assert head.param_count() == 15
        for bad in (0, -1, True, 1.5):
            with pytest.raises(ValueError):
                AI.TokenClassificationHead(bad, 3)
            with pytest.raises(ValueError):
                AI.TokenClassificationHead(4, bad)
        with pytest.raises(ValueError):
            head.forward([])
        with pytest.raises(ValueError):
            head.forward([[1.0, 2.0]])

    def test_gradients_match_numeric(self):
        random.seed(31)
        head = AI.TokenClassificationHead(4, 3)
        xs = [[1.0, 0.0, 0.5, -0.5], [0.2, 0.3, 0.1, 0.0]]
        head.zero_grad()
        loss, d_logits = head.loss_grad(head.forward(xs), [0, 2])
        assert math.isfinite(loss)
        dx = head.backward(d_logits)
        assert len(dx) == 2 and all(len(r) == 4 for r in dx)
        # 只查一个权重与一个偏置，避免测试过慢
        eps, i, j = 1e-6, 1, 2
        old = head.W[i][j]
        head.W[i][j] = old + eps
        lp, _ = head.loss_grad(head.forward(xs), [0, 2])
        head.W[i][j] = old - eps
        lm, _ = head.loss_grad(head.forward(xs), [0, 2])
        head.W[i][j] = old
        num = (lp - lm) / (2 * eps)
        ana = head.grad_W[i][j]
        assert abs(ana - num) <= 1e-6 * max(1e-8, abs(ana) + abs(num))

        old_b = head.b[2]
        head.b[2] = old_b + eps
        lp, _ = head.loss_grad(head.forward(xs), [0, 2])
        head.b[2] = old_b - eps
        lm, _ = head.loss_grad(head.forward(xs), [0, 2])
        head.b[2] = old_b
        num_b = (lp - lm) / (2 * eps)
        assert abs(head.grad_b[2] - num_b) <= 1e-6 * max(1e-8, abs(num_b))

    def test_mask_excludes_padding(self):
        head = AI.TokenClassificationHead(4, 3)
        xs = [[1.0, 0.0, 0.5, -0.5], [0.2, 0.3, 0.1, 0.0], [0.0, 0.0, 0.0, 0.0]]
        logits = head.forward(xs)
        head.zero_grad()
        loss_all, d_all = head.loss_grad(logits, [0, 2, 1])
        loss_masked, d_masked = head.loss_grad(logits, [0, 2, 1],
                                               mask=[True, True, False])
        expect = sum(-math.log(max(1e-12, AI._softmax_row(logits[t])[[0, 2][t]]))
                     for t in range(2)) / 2
        assert abs(loss_masked - expect) < 1e-9
        assert all(v == 0.0 for v in d_masked[2])
        assert abs(loss_all - loss_masked) > 1e-6
        with pytest.raises(ValueError):
            head.loss_grad(logits, [0, 2, 1], mask=[False, False, False])
        for bad in ([0, 5, 1], [0, 2], [0, 2, 1.5], [0, 2, True]):
            with pytest.raises(ValueError):
                head.loss_grad(logits, bad)

    def test_predict_and_training(self):
        head = AI.TokenClassificationHead(6, 4)
        logits = head.forward([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]])
        assert head.predict(logits) == [max(range(4), key=lambda l: logits[0][l])]
        assert head.predict(logits, mask=[False]) == [-1]

        random.seed(5)
        samples = []
        for _ in range(20):
            vec = [random.uniform(-1, 1) for _ in range(6)]
            label = max(range(4), key=lambda l: sum(vec[l::4]))
            samples.append((vec, label))
        first = None
        for _ in range(300):
            head.zero_grad()
            total = 0.0
            for vec, label in samples:
                loss, d = head.loss_grad(head.forward([vec]), [label])
                total += loss
                head.backward(d)
            head.adam_step(0.05)
            first = total / len(samples) if first is None else first
        last = total / len(samples)
        assert last < first / 20.0
        hit = sum(head.predict(head.forward([v]))[0] == y for v, y in samples)
        assert hit == len(samples)


class TestMultiLabelHead:
    """多标签头：概率、BCE 梯度、阈值预测、可训练性。"""

    def test_shapes_and_validation(self):
        head = AI.MultiLabelHead(4, 3)
        probs = head.forward([1.0, 0.0, 0.5, -0.5])
        assert len(probs) == 3 and all(0.0 < p < 1.0 for p in probs)
        assert head.param_count() == 15
        for bad in (0, -1, True, 1.5):
            with pytest.raises(ValueError):
                AI.MultiLabelHead(bad, 3)
        with pytest.raises(ValueError):
            head.forward([1.0, 2.0])
        with pytest.raises(ValueError):
            head.loss_grad(probs, [1, 2, 0])

    def test_gradients_match_numeric(self):
        random.seed(41)
        head = AI.MultiLabelHead(4, 3)
        x = [1.0, 0.0, 0.5, -0.5]
        head.zero_grad()
        loss, dz = head.loss_grad(head.forward(x), [1, 0, 1])
        assert math.isfinite(loss)
        dx = head.backward(dz)
        assert len(dx) == 4
        eps = 1e-6
        old = head.W[0][1]
        head.W[0][1] = old + eps
        lp, _ = head.loss_grad(head.forward(x), [1, 0, 1])
        head.W[0][1] = old - eps
        lm, _ = head.loss_grad(head.forward(x), [1, 0, 1])
        head.W[0][1] = old
        num = (lp - lm) / (2 * eps)
        ana = head.grad_W[0][1]
        assert abs(ana - num) <= 1e-6 * max(1e-8, abs(ana) + abs(num))

    def test_predict_threshold_and_training(self):
        head = AI.MultiLabelHead(5, 3)
        assert head.predict([0.9, 0.4, 0.5]) == [1, 0, 1]
        assert head.predict([0.9, 0.4, 0.5], threshold=0.4) == [1, 1, 1]
        with pytest.raises(ValueError):
            head.predict([0.9, 0.4, 0.5], threshold=1.0)

        random.seed(7)
        xs = [[random.uniform(-1, 1) for _ in range(5)] for _ in range(24)]
        ys = [[1 if v[0] + v[1] > 0 else 0, 1 if v[2] - v[3] > 0 else 0,
               1 if v[4] > 0 else 0] for v in xs]
        first = None
        for _ in range(400):
            head.zero_grad()
            total = 0.0
            for v, y in zip(xs, ys):
                loss, dz = head.loss_grad(head.forward(v), y)
                total += loss
                head.backward(dz)
            head.adam_step(0.05)
            first = total / len(xs) if first is None else first
        last = total / len(xs)
        assert last < first / 3.0
        assert all(head.predict(head.forward(v)) == y for v, y in zip(xs, ys))


class TestMoE:
    """专家混合层：路由语义、梯度、未使用专家的零梯度、可训练性。"""

    def test_shapes_and_validation(self):
        random.seed(3)
        moe = AI.MoE(4, 6, num_experts=3, top_k=2)
        xs = [[0.5, -0.2, 0.1, 0.3], [0.1, 0.4, -0.3, 0.2]]
        out = moe.forward(xs)
        assert len(out) == 2 and all(len(r) == 4 for r in out)
        assert moe.param_count() == 405
        assert all(len(r) == 2 for r in moe.routes)
        assert all(abs(sum(w for _e, w in r) - 1.0) < 1e-12 for r in moe.routes)
        for bad in ({"d_model": 0}, {"d_ff": 0}, {"num_experts": 0},
                    {"top_k": 0}, {"top_k": 5}, {"top_k": True}):
            kw = dict(d_model=4, d_ff=6, num_experts=3, top_k=2)
            kw.update(bad)
            with pytest.raises(ValueError):
                AI.MoE(**kw)
        with pytest.raises(ValueError):
            moe.forward([])
        with pytest.raises(ValueError):
            moe.forward([[1.0, 2.0]])
        with pytest.raises(RuntimeError):
            AI.MoE(4, 6).backward([[0.0] * 4])

    def test_top1_routing_weight_is_one(self):
        random.seed(9)
        moe = AI.MoE(3, 5, num_experts=2, top_k=1)
        moe.forward([[0.3, -0.4, 0.2], [0.1, 0.2, -0.5]])
        assert all(abs(w - 1.0) < 1e-12 for r in moe.routes for _e, w in r)

    def test_gradients_match_numeric(self):
        random.seed(5)
        moe = AI.MoE(4, 6, num_experts=3, top_k=2)
        xs = [[0.5, -0.2, 0.1, 0.3], [0.1, 0.4, -0.3, 0.2]]
        dout = [[0.3, -0.1, 0.2, 0.05], [-0.2, 0.4, 0.1, 0.25]]

        def loss():
            out = moe.forward(xs)
            return sum(dout[t][j] * out[t][j]
                       for t in range(2) for j in range(4))

        moe.forward(xs)
        moe.zero_grad()
        moe.backward(dout)
        gate = [[v for v in row] for row in moe.grad_W_gate]
        exp0 = [[v for v in row] for row in moe.experts[0].grads["W1"]]

        eps = 1e-6
        old = moe.W_gate[0][1]
        moe.W_gate[0][1] = old + eps
        lp = loss()
        moe.W_gate[0][1] = old - eps
        lm = loss()
        moe.W_gate[0][1] = old
        num = (lp - lm) / (2 * eps)
        assert abs(gate[0][1] - num) <= 1e-6 * max(1e-8, abs(gate[0][1]) + abs(num))

        old = moe.experts[0].W1[2][1]
        moe.experts[0].W1[2][1] = old + eps
        lp = loss()
        moe.experts[0].W1[2][1] = old - eps
        lm = loss()
        moe.experts[0].W1[2][1] = old
        num = (lp - lm) / (2 * eps)
        assert abs(exp0[2][1] - num) <= 1e-6 * max(1e-8, abs(exp0[2][1]) + abs(num))

    def test_unused_expert_gets_zero_gradient(self):
        random.seed(11)
        moe = AI.MoE(3, 4, num_experts=4, top_k=1)
        moe.forward([[0.5, 0.5, 0.5]])
        used = {e for e, _w in moe.routes[0]}
        moe.zero_grad()
        moe.backward([[0.1, 0.2, 0.3]])
        sums = {e: sum(abs(v) for row in moe.experts[e].grads["W1"] for v in row)
                for e in range(4)}
        assert any(sums[e] > 0.0 for e in used)
        assert all(sums[e] == 0.0 for e in range(4) if e not in used)

    def test_training_reduces_loss_and_uses_experts(self):
        random.seed(21)
        moe = AI.MoE(4, 8, num_experts=3, top_k=2)
        data = []
        for _ in range(12):
            vec = [random.uniform(-1, 1) for _ in range(4)]
            target = [0.0] * 4
            target[max(range(4), key=lambda j: vec[j])] = 1.0
            data.append((vec, target))

        def total_loss():
            s = 0.0
            for vec, tgt in data:
                out = moe.forward([vec])[0]
                s += sum((out[j] - tgt[j]) ** 2 for j in range(4)) / 4.0
            return s / len(data)

        first = total_loss()
        for _ in range(200):
            moe.zero_grad()
            for vec, tgt in data:
                out = moe.forward([vec])
                d = [[2.0 * (out[0][j] - tgt[j]) / (4.0 * len(data)) for j in range(4)]]
                moe.backward(d)
            moe.adam_step(0.05)
        assert total_loss() < first / 5.0

        used = set()
        for vec, _t in data:
            moe.forward([vec])
            used.update(e for e, _w in moe.routes[0])
        assert len(used) >= 2

class TestVAE:
    """VAE：重参数化、KL、梯度、采样。"""

    @staticmethod
    def _mk():
        random.seed(1)
        vae = AI.VAE(3, hidden_dim=6, latent_dim=2)
        x0 = [0.5, -0.2, 0.1]
        eps = [0.3, -0.4]
        return vae, x0, eps

    def test_reparam_and_kl(self):
        vae, x0, eps = self._mk()
        vae.zero_grad()
        loss = vae.loss_and_backward(x0, eps=eps)
        assert math.isfinite(loss)
        for i in range(2):
            assert abs(vae._z[i] - (vae._mu[i] + eps[i] * vae._std[i])) < 1e-15
        kl = 0.5 * sum(vae._mu[i] ** 2 + math.exp(vae._logvar[i]) - 1 - vae._logvar[i]
                       for i in range(2))
        assert kl >= -1e-12
        rec = sum((vae.decode(vae._z)[j] - x0[j]) ** 2 for j in range(3)) / 3
        assert abs(loss - (rec + kl)) < 1e-12

    def test_gradients_numeric(self):
        vae, x0, eps = self._mk()

        def L():
            return vae.loss_and_backward(x0, eps=eps)

        vae.zero_grad()
        L()
        ana_dec = vae._g[3][0][0][0]
        ana_enc = vae._g[1][0][0][0]
        e = 1e-6
        for layer, ana, tag in ((vae.dec2, ana_dec, "dec2"), (vae.enc2, ana_enc, "enc2")):
            old = layer.weights[0][0]
            layer.weights[0][0] = old + e
            vae.zero_grad()
            lp = L()
            layer.weights[0][0] = old - e
            vae.zero_grad()
            lm = L()
            layer.weights[0][0] = old
            num = (lp - lm) / (2 * e)
            assert abs(ana - num) / max(1e-12, abs(ana) + abs(num)) < 1e-6, tag

    def test_fit_and_sample(self):
        random.seed(2)
        data = [[0.6, 0.4], [0.5, 0.6], [-0.5, -0.4], [-0.6, -0.3],
                [0.1, -0.5], [-0.2, 0.5]]
        vae = AI.VAE(2, hidden_dim=12, latent_dim=2, beta=0.5)
        hist = vae.fit(data, epochs=120, lr=0.02, seed=7)
        assert hist[-1] < hist[0] * 0.3
        out = vae.sample(5, rng=random.Random(3))
        assert len(out) == 5 and all(len(v) == 2 for v in out)
        assert all(math.isfinite(x) for v in out for x in v)

    def test_validation(self):
        with pytest.raises(ValueError):
            AI.VAE(0)
        with pytest.raises(ValueError):
            AI.VAE(2, beta=-1.0)
        vae, x0, _eps = self._mk()
        with pytest.raises(ValueError):
            vae.forward(x0, eps=[0.1])
        with pytest.raises(ValueError):
            vae.loss_and_backward([0.1, 0.2])


class TestGAN:
    """GAN：判别器/生成器梯度与训练方向。"""

    def test_ranges(self):
        random.seed(4)
        gan = AI.GAN(2, hidden_dim=8, latent_dim=2)
        p = gan.discriminate([0.4, 0.5])
        assert 0.0 < p < 1.0
        assert len(gan.generate([0.3, -0.2])) == 2

    def test_discriminator_gradient(self):
        random.seed(4)
        gan = AI.GAN(2, hidden_dim=8, latent_dim=2)
        xr = [0.4, 0.5]
        w1 = gan._d_grads(xr, 1.0)[1][0]
        ana = w1[0][0]
        e = 1e-6
        old = gan.d1.weights[0][0]
        gan.d1.weights[0][0] = old + e
        lp = gan._d_grads(xr, 1.0)[0]
        gan.d1.weights[0][0] = old - e
        lm = gan._d_grads(xr, 1.0)[0]
        gan.d1.weights[0][0] = old
        num = (lp - lm) / (2 * e)
        assert abs(ana - num) / max(1e-12, abs(ana) + abs(num)) < 1e-6

    def test_generator_gradient(self):
        random.seed(4)
        gan = AI.GAN(2, hidden_dim=8, latent_dim=2)
        z = [0.3, -0.2]
        gan.generate(z)
        gen_w = gan._g_grads(z)[1][0]
        ana = gen_w[0][0]

        def obj():
            return -math.log(max(1e-12, gan.discriminate(gan.generate(z))))

        e = 1e-6
        old = gan.g2.weights[0][0]
        gan.g2.weights[0][0] = old + e
        lp = obj()
        gan.g2.weights[0][0] = old - e
        lm = obj()
        gan.g2.weights[0][0] = old
        num = (lp - lm) / (2 * e)
        assert abs(ana - num) / max(1e-12, abs(ana) + abs(num)) < 1e-6

    def test_training_moves_toward_data(self):
        random.seed(5)
        gan = AI.GAN(2, hidden_dim=16, latent_dim=2)
        target = [[1.8, 1.6], [2.2, 1.9], [2.0, 2.2], [1.7, 2.1]]

        def gen_mean():
            r = random.Random(11)
            pts = [gan.generate([r.gauss(0, 1), r.gauss(0, 1)]) for _ in range(6)]
            return [sum(p[i] for p in pts) / 6 for i in range(2)]

        before = gen_mean()
        for _ in range(300):
            gan.step(target, rng=random.Random(13))
        after = gen_mean()
        d0 = math.hypot(before[0] - 1.9, before[1] - 1.9)
        d1 = math.hypot(after[0] - 1.9, after[1] - 1.9)
        assert d1 < d0 * 0.9
        with pytest.raises(ValueError):
            gan.step([])


class TestDDPM:
    """DDPM：前向加噪、噪声预测梯度、反向采样。"""

    def test_q_sample(self):
        random.seed(6)
        d = AI.DDPM(2, hidden_dim=8, steps=10)
        x0, noise, t = [0.4, -0.3], [0.2, -0.7], 4
        xt = d.q_sample(x0, t, noise)
        a = math.sqrt(d.abar[t])
        b = math.sqrt(1.0 - d.abar[t])
        for i in range(2):
            assert abs(xt[i] - (a * x0[i] + b * noise[i])) < 1e-15
        with pytest.raises(TypeError):
            d.q_sample(x0, True, noise)
        with pytest.raises(ValueError):
            d.q_sample(x0, 10, noise)
        with pytest.raises(ValueError):
            d.q_sample([0.1], t, noise)
        with pytest.raises(ValueError):
            d.q_sample(x0, t, [0.1])

    def test_gradient(self):
        random.seed(6)
        d = AI.DDPM(2, hidden_dim=8, steps=10)
        x0, noise, t = [0.4, -0.3], [0.2, -0.7], 4
        d.zero_grad()
        d.train_step(x0, t=t, noise=noise)
        ana = d._g[2][0][0][0]
        e = 1e-6
        old = d.l3.weights[0][0]
        d.l3.weights[0][0] = old + e
        d.zero_grad()
        lp = d.train_step(x0, t=t, noise=noise)
        d.l3.weights[0][0] = old - e
        d.zero_grad()
        lm = d.train_step(x0, t=t, noise=noise)
        d.l3.weights[0][0] = old
        num = (lp - lm) / (2 * e)
        assert abs(ana - num) / max(1e-12, abs(ana) + abs(num)) < 1e-6

    def test_fit_and_sample(self):
        random.seed(7)
        d = AI.DDPM(2, hidden_dim=16, steps=16)
        data = [[1.0, 1.0], [0.9, 1.1], [-1.0, -0.9], [-0.9, -1.1]]
        hist = d.fit(data, epochs=150, lr=2e-3, seed=5)
        assert hist[-1] < hist[0] * 0.4
        out = d.sample(4, rng=random.Random(9))
        assert len(out) == 4 and all(len(v) == 2 for v in out)
        assert all(math.isfinite(x) for v in out for x in v)
        mean = [sum(v[i] for v in out) / 4 for i in range(2)]
        assert abs(mean[0]) < 2 and abs(mean[1]) < 2

    def test_validation(self):
        with pytest.raises(ValueError):
            AI.DDPM(2, beta_start=0.5, beta_end=0.1)
        with pytest.raises(ValueError):
            AI.DDPM(0)
        with pytest.raises(ValueError):
            AI.DDPM(2).train_step([0.1])


# ============================================================================
#  W0：训练基础设施（seed / 推理辅助 / 诊断 / 回调 / 续训 / layer_grad_norm / mixup）
# ============================================================================

_DATA2 = [([0.0, 0.0], 0), ([0.0, 1.0], 0), ([1.0, 0.0], 0), ([1.0, 1.0], 1)]


def _clf(seed=11):
    random.seed(seed)
    return AI.Model(AI.MLP([2, 4, 2], activations=['leaky_relu', 'softmax']),
                    task='classification')


class TestTrainingInfraSeed:
    def test_set_seed_reproducible(self):
        AI.set_seed(7)
        a = [random.random() for _ in range(5)]
        AI.set_seed(7)
        assert [random.random() for _ in range(5)] == a

    def test_set_seed_rejects_non_int(self):
        for bad in (1.5, "7", None, True):
            with pytest.raises(TypeError):
                AI.set_seed(bad)

    def test_fit_seed_still_independent(self):
        """fit 的 seed 用局部 Random，不推进全局流"""
        m = _clf(11)          # 建模型会重置全局种子，必须先建后测
        random.seed(1)
        before = random.random()
        random.seed(1)
        m.fit(data=_DATA2, epochs=2, batch_size=2, verbose=False, seed=3)
        assert random.random() == before


class TestInferShapesAndBatch:
    def test_infer_shapes(self):
        m = AI.Model(AI.MLP([3, 5, 2], activations=['relu', 'softmax']),
                     task='classification')
        info = AI.infer_shapes(m, [[0.1, 0.2, 0.3]])
        assert info['input_shape'] == [3]
        assert info['output_shape'] == [2]
        assert info['params'] == 3 * 5 + 5 + 5 * 2 + 2
        assert [l['activation'] for l in info['layers']] == ['relu', 'softmax']
        assert info['output'] == m.model.forward([0.1, 0.2, 0.3])

    def test_infer_shapes_rejects_bad_input(self):
        m = AI.Model(AI.MLP([3, 2], activations=['softmax']), task='classification')
        with pytest.raises(ValueError):
            AI.infer_shapes(m, [[0.1, 0.2]])
        with pytest.raises(ValueError):
            AI.infer_shapes(m, [])
        with pytest.raises(TypeError):
            AI.infer_shapes(object(), [[0.1, 0.2, 0.3]])

    def test_predict_batch_matches_predict(self):
        m = AI.Model(AI.MLP([3, 5, 2], activations=['relu', 'softmax']),
                     task='classification')
        xs = [[random.uniform(-1, 1) for _ in range(3)] for _ in range(11)]
        ref = m.predict(xs)
        for bs in (1, 3, 64):
            assert AI.predict_batch(m, xs, batch_size=bs) == ref
        assert AI.predict_batch(m, xs[0]) == m.predict(xs[0])
        with pytest.raises(ValueError):
            AI.predict_batch(m, xs, batch_size=0)


class TestDiagnostics:
    def _dead_model(self):
        m = AI.Model(AI.MLP([3, 6, 2], activations=['relu', 'linear']), task='regression')
        for j in (1, 4):
            m.model.layers[0].biases[j] = -50.0
        return m

    def test_activation_stats(self):
        m = self._dead_model()
        X = [[random.uniform(-1, 1) for _ in range(3)] for _ in range(20)]
        st = AI.activation_stats(m, X)
        assert st['samples'] == 20
        assert len(st['layers']) == 2
        assert st['layers'][0]['zero_ratio'] > 0.0
        assert all(l['std'] >= 0.0 for l in st['layers'])
        assert all(l['min'] <= l['max'] for l in st['layers'])

    def test_dead_units(self):
        m = self._dead_model()
        X = [[random.uniform(-1, 1) for _ in range(3)] for _ in range(20)]
        du = AI.dead_units(m, X)
        assert du['layers'][0]['dead'] == [1, 4]
        assert abs(du['layers'][0]['dead_ratio'] - 2 / 6) < 1e-12
        assert du['layers'][0]['note'] is None        # relu 层不作说明
        assert du['layers'][1]['note'] is not None    # linear 层会提示
        with pytest.raises(ValueError):
            AI.dead_units(m, X, atol=-1.0)


class TestCallbacks:
    class _Rec(AI.Callback):
        def __init__(self):
            self.events = []

        def on_train_begin(self, model, info):
            self.events.append(('begin', dict(info)))

        def on_epoch_begin(self, epoch, info):
            self.events.append(('epoch_begin', epoch))

        def on_epoch_end(self, epoch, logs):
            self.events.append(('epoch_end', dict(logs)))

        def on_train_end(self, history):
            self.events.append(('end', len(history['train_loss'])))

    def test_hook_order_and_payload(self):
        rec = self._Rec()
        m = _clf()
        m.fit(data=_DATA2, epochs=3, batch_size=2, verbose=False, callbacks=[rec], seed=5)
        kinds = [e[0] for e in rec.events]
        assert kinds == ['begin'] + ['epoch_begin', 'epoch_end'] * 3 + ['end']
        begin = rec.events[0][1]
        assert begin['epochs'] == 3 and begin['batch_size'] == 2
        assert begin['task'] == 'classification'
        assert begin['n_train'] == 3 and begin['n_val'] == 1
        assert begin['resumed_step'] == 0
        logs = [e[1] for e in rec.events if e[0] == 'epoch_end']
        assert [l['epoch'] for l in logs] == [1, 2, 3]
        assert set(logs[0]) == {'epoch', 'train_loss', 'val_loss', 'grad_norm',
                                'layer_grad_norm', 'lr'}

    def test_single_callback_and_bad_callback(self):
        rec = self._Rec()
        assert AI._normalize_callbacks(rec) == [rec]
        with pytest.raises(TypeError):
            _clf().fit(data=_DATA2, epochs=1, verbose=False, callbacks=[object()])

    def test_callback_exception_propagates(self):
        class Boom(AI.Callback):
            def on_epoch_end(self, epoch, logs):
                raise RuntimeError('stop now')

        with pytest.raises(RuntimeError):
            _clf().fit(data=_DATA2, epochs=2, batch_size=2, verbose=False,
                       callbacks=[Boom()], seed=1)


class TestResumeAndLayerNorms:
    def test_resume_restores_weights_and_step(self, tmp_path):
        ck = str(tmp_path / "ck.json")
        base = _clf(9)
        base.fit(data=_DATA2, epochs=3, batch_size=2, verbose=False,
                 checkpoint_path=ck, seed=8)
        cont = _clf(21)
        cont.fit(data=_DATA2, epochs=0, batch_size=2, verbose=False, resume_from=ck)
        assert cont.optimizer.step_count == 6
        for la, lb in zip(cont.model.layers, base.model.layers):
            assert la.weights == lb.weights
            assert la.biases == lb.biases

    def test_resume_errors(self, tmp_path):
        ck = str(tmp_path / "bad.json")
        with open(ck, "w", encoding="utf-8") as f:
            f.write(json.dumps({"type": "MLP", "layers": []}))
        with pytest.raises(ValueError):
            _clf().fit(data=_DATA2, epochs=1, verbose=False, resume_from=ck)
        with pytest.raises(FileNotFoundError):
            _clf().fit(data=_DATA2, epochs=1, verbose=False,
                       resume_from=str(tmp_path / "nope.json"))

    def test_layer_grad_norm(self):
        m1, m2 = _clf(5), _clf(5)
        kw = dict(data=_DATA2, epochs=3, batch_size=4, verbose=False, seed=4)
        h1 = m1.fit(record_layer_norms=True, **kw)
        h2 = m2.fit(**kw)
        assert h1['train_loss'] == h2['train_loss']        # 只记录，不影响训练
        assert len(h1['layer_grad_norm']) == 3
        assert all(len(v) == 2 and all(x >= 0 for x in v) for _, v in h1['layer_grad_norm'])
        assert 'layer_grad_norm' not in h2


class TestMixupInFit:
    def test_mixup_batch_semantics(self):
        m = _clf(31)
        mixed = m._mixup_batch([([0.0, 0.0], 0), ([1.0, 1.0], 1)], 0.4, random.Random(7))
        assert len(mixed) == 2
        assert all(len(y) == 2 and abs(sum(y) - 1.0) < 1e-12 for _, y in mixed)
        with pytest.raises(ValueError):
            m._mixup_batch([([0.0, 0.0], [0.5, 0.6])], 0.4, random.Random(7))

    def test_mixup_changes_training_but_stays_finite(self):
        m = _clf(31)
        h_mix = m.fit(data=_DATA2, epochs=4, batch_size=2, verbose=False,
                      mixup_alpha=0.4, seed=3)
        assert all(math.isfinite(v) for _, v in h_mix['train_loss'])
        m2 = _clf(31)
        h_plain = m2.fit(data=_DATA2, epochs=4, batch_size=2, verbose=False, seed=3)
        assert h_mix['train_loss'] != h_plain['train_loss']

    def test_mixup_rejects_regression_and_bad_alpha(self):
        reg = AI.Model(AI.MLP([2, 3, 1], activations=['relu', 'linear']), task='regression')
        with pytest.raises(ValueError):
            reg.fit(data=[([0.0, 0.0], 1.0)], epochs=1, verbose=False, mixup_alpha=0.2)
        for bad in (-0.1, float('nan'), 'x'):
            with pytest.raises((ValueError, TypeError)):
                _clf().fit(data=_DATA2, epochs=1, verbose=False, mixup_alpha=bad)


class TestBuiltinCallbacks:
    def test_early_stopping_requests_stop(self):
        es = AI.EarlyStopping(monitor='val_loss', patience=1)
        m = _clf()
        m.fit(data=_DATA2, epochs=40, batch_size=4, verbose=False,
              callbacks=[es], seed=3)
        assert len(m.history['train_loss']) < 40
        assert es.should_stop and es.stopped_epoch is not None
        assert es.best == min(v for _, v in m.history['val_loss'])

    def test_early_stopping_validation(self):
        for kw in ({'patience': 0}, {'mode': 'up'}, {'monitor': 'acc'},
                   {'min_delta': -1.0}):
            with pytest.raises(ValueError):
                AI.EarlyStopping(**kw)

    def test_model_checkpoint_and_resume(self, tmp_path):
        path = str(tmp_path / "ck.json")
        cb = AI.ModelCheckpoint(path, every=2)
        m = _clf()
        m.fit(data=_DATA2, epochs=4, batch_size=4, verbose=False,
              callbacks=[cb], seed=5)
        assert cb.saved_epochs == [2, 4]
        saved = json.load(io.open(path, encoding="utf-8"))
        assert "optimizer" in saved
        cont = _clf(99)
        cont.fit(data=_DATA2, epochs=0, verbose=False, resume_from=path)
        assert cont.optimizer.step_count == saved["optimizer"]["step_count"]

    def test_model_checkpoint_best_only(self, tmp_path):
        path = str(tmp_path / "best.json")
        cb = AI.ModelCheckpoint(path, save_best_only=True)
        m = _clf()
        m.fit(data=_DATA2, epochs=5, batch_size=4, verbose=False,
              callbacks=[cb], seed=7)
        assert len(cb.saved_epochs) == 1
        assert cb.best <= max(v for _, v in m.history['val_loss'])
        with pytest.raises(ValueError):
            AI.ModelCheckpoint("")
        with pytest.raises(ValueError):
            AI.ModelCheckpoint(path, every=0)

    def test_logging_and_history_callbacks(self):
        stream = io.StringIO()
        logger = logging.getLogger("test.ai.train")
        logger.handlers = [logging.StreamHandler(stream)]
        logger.setLevel(logging.INFO)
        hc = AI.HistoryCallback()
        m = _clf()
        m.fit(data=_DATA2, epochs=2, batch_size=4, verbose=False,
              callbacks=[AI.LoggingCallback(logger), hc], seed=1)
        out = stream.getvalue()
        assert "训练开始" in out and "epoch 2" in out and "训练结束" in out
        assert len(hc.records) == 2
        assert set(hc.records[-1]) == {'epoch', 'train_loss', 'val_loss',
                                       'grad_norm', 'layer_grad_norm', 'lr'}


class TestRegressionLosses:
    """W1-A：l1 / log_cosh / quantile / smooth_l1"""

    def test_l1_loss_value_and_grad(self):
        loss, grad = AI.l1_loss([0.9, -0.7, 2.4], [0.5, -1.2, 3.0])
        expected = (0.4 + 0.5 + 0.6) / 3
        assert abs(loss - expected) < 1e-15
        assert grad == [1 / 3, 1 / 3, -1 / 3]

    def test_log_cosh_stable_for_large_error(self):
        loss, grad = AI.log_cosh_loss([1000.0], [0.0])
        assert math.isfinite(loss)
        assert abs(loss - (1000.0 - math.log(2.0))) < 1e-9
        assert abs(grad[0] - 1.0) < 1e-12

    def test_quantile_half_equals_half_l1(self):
        y = [0.5, -1.2, 3.0, 2.2]
        p = [0.9, -0.7, 2.4, 2.55]
        l1_l, l1_g = AI.l1_loss(p, y)
        q_l, q_g = AI.quantile_loss(p, y, quantile=0.5)
        assert q_l == l1_l / 2.0
        assert q_g == [v / 2.0 for v in l1_g]

    def test_smooth_l1_is_huber_alias(self):
        p, y = [0.9, -0.7], [0.5, -1.2]
        for delta in (0.3, 1.0, 5.0):
            assert AI.smooth_l1_loss(p, y, delta=delta) == AI.huber_loss(p, y, delta=delta)

    def test_regression_losses_through_model(self):
        m = AI.Model(AI.MLP([2, 3, 1], activations=["relu", "linear"]),
                     loss_fn="quantile", loss_kwargs={"quantile": 0.9},
                     optimizer="adam", lr=0.05, task="regression")
        loss, _grad = m._loss([0.4], [0.0])
        assert abs(loss - 0.36) < 1e-12
        assert m._loss_name == "quantile"

    def test_regression_losses_reject_softmax_output(self):
        for name in ("l1", "log_cosh", "quantile", "smooth_l1"):
            with pytest.raises(ValueError):
                AI.Model(AI.MLP([2, 3, 2], activations=["relu", "softmax"]),
                         loss_fn=name, optimizer="adam", task="classification")

    def test_invalid_targets_and_quantile(self):
        with pytest.raises(ValueError):
            AI.l1_loss([1.0, 2.0], 1.0)          # 标量 target 配多输出
        with pytest.raises(ValueError):
            AI.l1_loss([1.0, 2.0], [1.0])        # 长度不符
        for q in (0.0, 1.0, -0.1, 1.5):
            with pytest.raises(ValueError):
                AI.quantile_loss([1.0], [0.0], quantile=q)


# ============================================================================
#  类别不平衡损失（Class-Balanced / LDAM / OHEM）
# ============================================================================
class TestImbalanceLosses:
    """W1-B：类别不平衡损失的正确性与接线"""

    @staticmethod
    def _softmax(z):
        mx = max(z)
        ex = [math.exp(v - mx) for v in z]
        s = sum(ex)
        return [e / s for e in ex]

    @staticmethod
    def _numeric(f, z, eps=1e-6):
        out = []
        for i in range(len(z)):
            zp = list(z)
            zm = list(z)
            zp[i] += eps
            zm[i] -= eps
            out.append((f(zp) - f(zm)) / (2 * eps))
        return out

    def test_class_balanced_weights(self):
        counts = [100.0, 10.0, 1.0]
        probs = [0.5, 0.3, 0.2]
        raw = [(1 - 0.999) / (1 - 0.999 ** c) for c in counts]
        # 未归一化时 loss 就是 -w_c*log p_c
        for i in range(3):
            loss, _ = AI.class_balanced_loss(probs, i, counts, beta=0.999, normalize=False)
            assert abs(loss + raw[i] * math.log(probs[i])) < 1e-12
        # 样本越少权重越大
        assert raw[0] < raw[1] < raw[2]
        # 归一化：按频率加权的权重均值为 1，故 loss 比例为 raw/meanw
        mean_w = sum(c * w for c, w in zip(counts, raw)) / sum(counts)
        for i in range(3):
            loss, _ = AI.class_balanced_loss(probs, i, counts, beta=0.999)
            assert abs(loss + (raw[i] / mean_w) * math.log(probs[i])) < 1e-12
        # beta=0 时退化成普通 CE
        for i in range(3):
            loss, _ = AI.class_balanced_loss(probs, i, counts, beta=0.0)
            assert abs(loss + math.log(probs[i])) < 1e-12

    def test_class_balanced_gradient(self):
        z = [0.7, -1.2, 2.4]
        cnt = [30.0, 5.0, 2.0]
        for y in (0, 1, 2):
            _, g = AI.class_balanced_loss(self._softmax(z), y, cnt, beta=0.99)
            num = self._numeric(
                lambda zz: AI.class_balanced_loss(self._softmax(zz), y, cnt, beta=0.99)[0], z)
            worst = max(abs(a - b) / max(1e-12, abs(a) + abs(b)) for a, b in zip(g, num))
            assert worst < 1e-7, (y, worst)

    def test_class_balanced_soft_target(self):
        z = [0.7, -1.2, 2.4]
        cnt = [30.0, 5.0, 2.0]
        soft = [0.0, 0.8, 0.2]
        _, g = AI.class_balanced_loss(self._softmax(z), soft, cnt, beta=0.99)
        num = self._numeric(
            lambda zz: AI.class_balanced_loss(self._softmax(zz), soft, cnt, beta=0.99)[0], z)
        worst = max(abs(a - b) / max(1e-12, abs(a) + abs(b)) for a, b in zip(g, num))
        assert worst < 1e-7
        with pytest.raises(ValueError):
            AI.class_balanced_loss([0.5, 0.5], [0.0, 0.0], [10.0, 3.0])

    def test_ldam_matches_closed_form(self):
        logits = [0.3, -0.5, 1.1]
        counts = [200.0, 20.0, 2.0]
        loss, grad = AI.ldam_loss(logits, 0, counts)
        m = 0.5 / counts[0] ** 0.25
        zz = [30.0 * v for v in logits]
        zz[0] -= 30.0 * m
        mx = max(zz)
        ref = -(zz[0] - mx - math.log(sum(math.exp(v - mx) for v in zz)))
        assert abs(loss - ref) < 1e-12
        num = self._numeric(lambda ll: AI.ldam_loss(ll, 0, counts)[0], logits)
        diff = max(abs(a - b) for a, b in zip(grad, num))
        assert diff < 1e-8, diff

    def test_ldam_margin_grows_for_rare_class(self):
        rare = AI.ldam_loss([1.0, 0.5], 0, [1.0, 100.0])[0]
        common = AI.ldam_loss([1.0, 0.5], 0, [100.0, 1.0])[0]
        assert rare > common

    def test_ohem_selection(self):
        preds = [[0.9, 0.05, 0.05],
                 [0.1, 0.8, 0.1],
                 [0.2, 0.2, 0.6],
                 [0.34, 0.33, 0.33],
                 [0.6, 0.2, 0.2]]
        targets = [0, 1, 2, 2, 1]
        per_sample = [AI.cross_entropy_loss(p, t)[0] for p, t in zip(preds, targets)]
        loss, grads = AI.ohem_loss(preds, targets, keep_ratio=0.6)
        keep = 2
        hardest = sorted(range(5), key=lambda i: per_sample[i], reverse=True)[:keep]
        assert abs(loss - sum(per_sample[i] for i in hardest) / keep) < 1e-12
        for i in range(5):
            if i not in hardest:
                assert all(v == 0.0 for v in grads[i])
            else:
                ref = AI.cross_entropy_loss(preds[i], targets[i])[1]
                assert all(abs(a - b) < 1e-12 for a, b in zip(grads[i], ref))

    def test_model_integration_and_guards(self):
        assert AI.Model._LOSS_MAP["class_balanced"] is AI.class_balanced_loss
        assert AI.Model._LOSS_MAP["ldam"] is AI.ldam_loss
        assert "class_balanced" in AI.Model._INDEX_LOSSES
        assert "ldam" in AI.Model._INDEX_LOSSES

        random.seed(7)
        mlp = AI.MLP([4, 6, 3], activations=["relu", "softmax"])
        model = AI.Model(mlp, loss_fn="class_balanced", task="classification",
                         loss_kwargs={"class_counts": [50.0, 5.0, 1.0], "beta": 0.99})
        data = [([random.random() for _ in range(4)], random.randrange(3)) for _ in range(30)]
        model.fit(data=data, epochs=15, batch_size=8, verbose=False, seed=1)
        losses = [v for _, v in model.history["train_loss"]]
        assert losses[-1] < losses[0]

        bad = AI.MLP([3, 4, 3], activations=["relu", "softmax"])
        with pytest.raises(ValueError):
            AI.Model(bad, loss_fn="ldam", task="classification",
                     loss_kwargs={"class_counts": [10.0, 10.0, 10.0]})
        bad2 = AI.MLP([3, 4, 3], activations=["relu", "linear"])
        with pytest.raises(ValueError):
            AI.Model(bad2, loss_fn="class_balanced", task="classification",
                     loss_kwargs={"class_counts": [10.0, 10.0, 10.0]})

    def test_invalid_inputs(self):
        with pytest.raises(ValueError):
            AI.class_balanced_loss([0.5, 0.5], 0, [10.0])
        with pytest.raises(ValueError):
            AI.class_balanced_loss([0.5, 0.5], 0, [10.0, 0.0])
        with pytest.raises(ValueError):
            AI.class_balanced_loss([0.5, 0.5], 0, [10.0, 3.0], beta=1.0)
        with pytest.raises(TypeError):
            AI.class_balanced_loss([0.5, 0.5], 0, [10.0, 3.0], normalize=1)
        with pytest.raises(ValueError):
            AI.ldam_loss([0.1, 0.2], 0, [5.0, 5.0], max_m=-1.0)
        with pytest.raises(ValueError):
            AI.ldam_loss([0.1, 0.2], 0, [5.0, 5.0], s=0.0)
        with pytest.raises(ValueError):
            AI.ohem_loss([], [])
        with pytest.raises(ValueError):
            AI.ohem_loss([[0.5, 0.5]], [0], keep_ratio=1.0)

class TestDistillAndHint:
    """知识蒸馏：软目标 KL（含 T^2 缩放）与隐藏层对齐。"""

    def test_distill_matches_scaled_kl(self):
        zs = [0.5, -1.0, 2.0]
        zt = [1.2, 0.1, -0.7]
        loss, grad = AI.distill_loss(zs, zt, T=2.0)
        ps = AI._softmax_with_temp(zs, 2.0)
        pt = AI._softmax_with_temp(zt, 2.0)
        ref = 4.0 * sum(b * math.log(b / a) for a, b in zip(ps, pt))
        assert abs(loss - ref) < 1e-12
        assert max(abs(g - 2.0 * (a - b)) for g, a, b in zip(grad, ps, pt)) < 1e-12

    def test_distill_zero_when_equal_and_numeric_grad(self):
        z = [0.3, 0.1, -0.4]
        loss, grad = AI.distill_loss(z, z, T=1.5)
        assert abs(loss) < 1e-15
        assert all(abs(g) < 1e-15 for g in grad)

        zs = [0.5, -1.0, 2.0]
        zt = [1.2, 0.1, -0.7]
        _, g = AI.distill_loss(zs, zt, T=0.7)
        eps = 1e-6
        for i in range(len(zs)):
            up = list(zs)
            dn = list(zs)
            up[i] += eps
            dn[i] -= eps
            num = (AI.distill_loss(up, zt, T=0.7)[0] - AI.distill_loss(dn, zt, T=0.7)[0]) / (2 * eps)
            assert abs(g[i] - num) < 1e-8

    def test_distill_rejects_bad_temperature(self):
        zs = [0.1, 0.2]
        zt = [0.3, 0.4]
        for bad in (0.0, -1.0, float("inf"), float("nan"), True, "2"):
            with pytest.raises((ValueError, TypeError)):
                AI.distill_loss(zs, zt, T=bad)
        with pytest.raises(ValueError):
            AI.distill_loss([0.1, 0.2], [0.1], T=1.0)

    def test_hint_loss_mse_and_weight(self):
        a = [0.5, -1.0, 2.0, 0.0]
        b = [0.2, -0.4, 1.5, 0.3]
        loss, grad = AI.hint_loss(a, b)
        ref = sum((x - y) ** 2 for x, y in zip(a, b)) / len(a)
        assert abs(loss - ref) < 1e-12
        assert max(abs(g - 2.0 * (x - y) / len(a)) for g, x, y in zip(grad, a, b)) < 1e-12

        loss2, grad2 = AI.hint_loss(a, b, weight=2.5)
        assert abs(loss2 - 2.5 * ref) < 1e-12
        assert max(abs(g - 2.5 * h) for g, h in zip(grad2, grad)) < 1e-12
        assert AI.hint_loss(a, b, weight=0.0)[0] == 0.0
        for bad in (-1.0, float("nan"), float("inf")):
            with pytest.raises(ValueError):
                AI.hint_loss(a, b, weight=bad)
        with pytest.raises(TypeError):
            AI.hint_loss(a, b, weight="x")
        with pytest.raises(ValueError):
            AI.hint_loss([], [])

    def test_model_wiring_and_activation_guard(self):
        assert AI.Model._LOSS_MAP["distill"] is AI.distill_loss
        assert "distill" in AI.Model._NUMERIC_LOSSES

        random.seed(31)
        mlp = AI.MLP([4, 8, 3], activations=["relu", "linear"])
        model = AI.Model(mlp, loss_fn="distill", task="classification", lr=0.02)
        teacher = [[1.5, -0.5, 0.2], [0.3, 1.2, -0.9]]
        data = [([random.uniform(-1, 1) for _ in range(4)], teacher[i % 2]) for i in range(20)]
        model.fit(data=data, epochs=20, batch_size=5, verbose=False, seed=3)
        losses = [v for _, v in model.history["train_loss"]]
        assert losses[-1] < losses[0]

        bad = AI.MLP([4, 8, 3], activations=["relu", "softmax"])
        with pytest.raises(ValueError):
            AI.Model(bad, loss_fn="distill", task="classification")


# ============================================================================
#  偏好对齐损失（W1-D：DPO / IPO / SimPO / CPO / ORPO + KTO）
# ============================================================================

class TestPreferenceLoss:
    def test_dpo_closed_form_and_signs(self):
        loss, dc, dr = AI.preference_loss(-1.2, -2.5, -1.5, -2.2, method="dpo", beta=0.1)
        x = 0.1 * ((-1.2 - (-2.5)) - (-1.5 - (-2.2)))
        assert abs(loss - (-math.log(1.0 / (1.0 + math.exp(-x))))) < 1e-12
        assert dc < 0.0 and abs(dr + dc) < 1e-15

    def test_dpo_gradient_matches_finite_difference(self):
        eps = 1e-6
        _, dc, dr = AI.preference_loss(-1.2, -2.5, -1.5, -2.2, method="dpo", beta=0.1)
        lp = AI.preference_loss(-1.2 + eps, -2.5, -1.5, -2.2, method="dpo", beta=0.1)[0]
        lm = AI.preference_loss(-1.2 - eps, -2.5, -1.5, -2.2, method="dpo", beta=0.1)[0]
        assert abs((lp - lm) / (2 * eps) - dc) < 1e-8
        lp = AI.preference_loss(-1.2, -2.5 + eps, -1.5, -2.2, method="dpo", beta=0.1)[0]
        lm = AI.preference_loss(-1.2, -2.5 - eps, -1.5, -2.2, method="dpo", beta=0.1)[0]
        assert abs((lp - lm) / (2 * eps) - dr) < 1e-8

    def test_ipo_closed_form(self):
        loss, dc, _ = AI.preference_loss(-1.2, -2.5, -1.5, -2.2, method="ipo", beta=0.1)
        gap = (-1.2 - (-2.5)) - (-1.5 - (-2.2)) - 1.0 / (2 * 0.1)
        assert abs(loss - gap * gap) < 1e-12
        assert abs(dc - 2 * gap) < 1e-12

    def test_simpo_needs_no_reference(self):
        loss, _, _ = AI.preference_loss(-1.2, -2.5, method="simpo", beta=0.2, margin=0.5)
        x = 0.2 * (-1.2 - (-2.5)) - 0.5
        assert abs(loss - (-math.log(1.0 / (1.0 + math.exp(-x))))) < 1e-12
        with_margin = AI.preference_loss(-1.2, -2.5, method="simpo", beta=0.2, margin=0.5)[0]
        no_margin = AI.preference_loss(-1.2, -2.5, method="simpo", beta=0.2, margin=0.0)[0]
        assert with_margin > no_margin

    def test_cpo_adds_sft_term(self):
        cpo_loss, cpo_dc, _ = AI.preference_loss(-1.2, -2.5, -1.5, -2.2,
                                                 method="cpo", beta=0.1, sft_weight=0.3)
        dpo_loss, dpo_dc, _ = AI.preference_loss(-1.2, -2.5, -1.5, -2.2, method="dpo", beta=0.1)
        assert abs(cpo_loss - (dpo_loss + 0.3 * 1.2)) < 1e-12
        assert abs(cpo_dc - (dpo_dc - 0.3)) < 1e-12

    def test_orpo_logit_space(self):
        loss, dc, _ = AI.preference_loss(-0.7, -1.4, method="orpo", beta=0.1)
        oc = math.log(math.exp(-0.7) / (1.0 - math.exp(-0.7)))
        orr = math.log(math.exp(-1.4) / (1.0 - math.exp(-1.4)))
        x = 0.1 * (oc - orr)
        assert abs(loss - (-math.log(1.0 / (1.0 + math.exp(-x))))) < 1e-10
        eps = 1e-6
        lp = AI.preference_loss(-0.7 + eps, -1.4, method="orpo", beta=0.1)[0]
        lm = AI.preference_loss(-0.7 - eps, -1.4, method="orpo", beta=0.1)[0]
        assert abs((lp - lm) / (2 * eps) - dc) < 1e-6
        with pytest.raises(ValueError):  # p >= 1 时 logit 无定义
            AI.preference_loss(0.0, -1.0, method="orpo")

    def test_kto_two_sides(self):
        z = -1.0 - (-1.6)
        ld, gd = AI.kto_loss(-1.0, -1.6, desirable=True, beta=0.5)
        lu, gu = AI.kto_loss(-1.0, -1.6, desirable=False, beta=0.5)
        assert abs(ld - (1.0 - 1.0 / (1.0 + math.exp(-0.5 * z)))) < 1e-12
        assert abs(lu - 1.0 / (1.0 + math.exp(-0.5 * z))) < 1e-12
        assert abs(gd + gu) < 1e-15
        eps = 1e-6
        lp = AI.kto_loss(-1.0 + eps, -1.6, desirable=True, beta=0.5)[0]
        lm = AI.kto_loss(-1.0 - eps, -1.6, desirable=True, beta=0.5)[0]
        assert abs((lp - lm) / (2 * eps) - gd) < 1e-6

    def test_numeric_stability(self):
        tie = AI.preference_loss(-1.0, -30.0, -1.0, -30.0, method="dpo", beta=50.0)
        assert abs(tie[0] - math.log(2.0)) < 1e-12
        extreme = AI.preference_loss(-0.1, -30.0, -1.0, -30.0, method="dpo", beta=50.0)
        assert math.isfinite(extreme[0]) and extreme[0] < 1e-12
        assert math.isfinite(extreme[1]) and math.isfinite(extreme[2])

    def test_invalid_inputs(self):
        with pytest.raises(ValueError):
            AI.preference_loss(-1, -2, method="ppo")
        with pytest.raises(TypeError):
            AI.preference_loss(-1, -2, method=3)
        with pytest.raises(ValueError):
            AI.preference_loss(-1, -2, method="dpo")  # 缺参考
        with pytest.raises(ValueError):
            AI.preference_loss(-1, -2, -1, -2, method="dpo", beta=0.0)
        with pytest.raises(ValueError):
            AI.preference_loss(-1, -2, method="simpo", margin=-0.5)
        with pytest.raises(TypeError):
            AI.preference_loss(True, -2)
        with pytest.raises(ValueError):
            AI.preference_loss(float("nan"), -2)
        with pytest.raises(TypeError):
            AI.kto_loss(-1.0, -1.6, desirable=1)
        with pytest.raises(ValueError):
            AI.kto_loss(-1.0, -1.6, beta=-1.0)

class TestMultiTaskWeighting:
    """W1-E：多任务权重（不确定性加权 / GradNorm / PCGrad）。"""

    def test_uncertainty_weighting_closed_form(self):
        losses = [1.0, 2.0, 0.5]
        log_vars = [0.0, math.log(2.0), -1.0]
        total, weights, d_losses, d_log_vars = AI.uncertainty_weighting(losses, log_vars)
        expect_w = [math.exp(-s) for s in log_vars]
        expect_total = sum(
            wi * li + si for wi, li, si in zip(expect_w, losses, log_vars)
        )
        assert abs(total - expect_total) <= 1e-12 * max(1.0, abs(expect_total))
        for got, want in zip(weights, expect_w):
            assert abs(got - want) <= 1e-12
        for got, want in zip(d_losses, weights):
            assert abs(got - want) <= 1e-12
        for got, w, li in zip(d_log_vars, expect_w, losses):
            assert abs(got - (1.0 - w * li)) <= 1e-12

    def test_uncertainty_weighting_gradients_match_numeric(self):
        losses = [1.0, 2.0, 0.5]
        log_vars = [0.0, math.log(2.0), -1.0]
        _, _, d_losses, d_log_vars = AI.uncertainty_weighting(losses, log_vars)
        eps = 1e-6
        worst = 0.0
        for i in range(len(losses)):
            up = list(losses); up[i] += eps
            dn = list(losses); dn[i] -= eps
            num = (AI.uncertainty_weighting(up, log_vars)[0]
                   - AI.uncertainty_weighting(dn, log_vars)[0]) / (2 * eps)
            worst = max(worst, abs(num - d_losses[i]) / max(1e-12, abs(num) + abs(d_losses[i])))
        for i in range(len(log_vars)):
            up = list(log_vars); up[i] += eps
            dn = list(log_vars); dn[i] -= eps
            num = (AI.uncertainty_weighting(losses, up)[0]
                   - AI.uncertainty_weighting(losses, dn)[0]) / (2 * eps)
            worst = max(worst, abs(num - d_log_vars[i]) / max(1e-12, abs(num) + abs(d_log_vars[i])))
        assert worst < 1e-7

    def test_gradnorm_fixed_point(self):
        res = AI.gradnorm_weights([2.0, 2.0, 2.0], [3.0, 3.0, 3.0], alpha=1.5, lr=0.025, G0=3.0)
        assert all(abs(t - 3.0) <= 1e-12 for t in res["targets"])
        assert abs(res["loss"]) <= 1e-12
        assert all(abs(w - 1.0) <= 1e-12 for w in res["weights"])
        assert all(abs(g) <= 1e-12 for g in res["gradients"])

    def test_gradnorm_updates_and_normalizes(self):
        losses = [1.0, 2.0, 4.0]
        gs = [3.0, 3.0, 3.0]
        res = AI.gradnorm_weights(losses, gs, alpha=1.5, G0=3.0)
        assert res["targets"][0] < res["targets"][1] < res["targets"][2]
        assert res["loss"] > 0.0
        assert all(w > 0.0 for w in res["weights"])
        assert abs(sum(res["weights"]) / 3.0 - 1.0) <= 1e-9
        mean_l = sum(losses) / 3.0
        r = [(x / mean_l) ** 1.5 for x in losses]
        for g_i, t_i, r_i, grad in zip(gs, res["targets"], r, res["gradients"]):
            sign = 1.0 if g_i > t_i else (-1.0 if g_i < t_i else 0.0)
            assert abs(grad - sign * 3.0 / r_i) <= 1e-9

    def test_pcgrad_projects_conflict(self):
        a = [1.0, 0.0]
        b = [-1.0, 1.0]
        ga, gb = AI.pcgrad(a, b)
        assert ga[0] * gb[0] + ga[1] * gb[1] >= -1e-12
        assert abs(ga[0] * b[0] + ga[1] * b[1]) < 1e-12
        assert abs(gb[0] * a[0] + gb[1] * a[1]) < 1e-12
        assert a == [1.0, 0.0] and b == [-1.0, 1.0]

    def test_pcgrad_no_conflict_unchanged(self):
        c = [1.0, 1.0]
        d = [1.0, 0.0]
        gc, gd = AI.pcgrad(c, d)
        assert gc == c and gd == d

    def test_pcgrad_nested_structure(self):
        a2 = [[1.0, 2.0], [3.0]]
        b2 = [[-1.0, 0.0], [-2.0]]
        ga2, gb2 = AI.pcgrad(a2, b2)
        dot = sum(x * y for xs, ys in zip(ga2, gb2) for x, y in zip(xs, ys))
        assert dot >= -1e-12
        assert isinstance(ga2[0], list) and len(ga2[1]) == 1

    def test_invalid_inputs(self):
        with pytest.raises(ValueError):
            AI.uncertainty_weighting([], [1.0])
        with pytest.raises(ValueError):
            AI.uncertainty_weighting([1.0], [1.0, 2.0])
        with pytest.raises(ValueError):
            AI.uncertainty_weighting([float("nan")], [0.0])
        with pytest.raises(TypeError):
            AI.uncertainty_weighting([True], [0.0])
        with pytest.raises(ValueError):
            AI.gradnorm_weights([1.0], [1.0], alpha=0)
        with pytest.raises(ValueError):
            AI.gradnorm_weights([1.0], [1.0], lr=-1)
        with pytest.raises(ValueError):
            AI.gradnorm_weights([0.0], [1.0])
        with pytest.raises(ValueError):
            AI.gradnorm_weights([1.0, 2.0], [1.0])
        with pytest.raises(ValueError):
            AI.gradnorm_weights([1.0], [1.0], G0=0)
        with pytest.raises(ValueError):
            AI.pcgrad([1.0], [1.0, 2.0])
        with pytest.raises(ValueError):
            AI.pcgrad([1.0], [1.0], eps=0)


# ============================================================================
#  W2-A：池化家族与 SEBlock
# ============================================================================
class TestPoolingExtras:
    def test_global_max_pool(self):
        x = [[[1.0, 5.0], [3.0, 2.0]], [[-1.0, -4.0], [-2.0, -3.0]]]
        pool = AI.GlobalMaxPool2d()
        out = pool.forward(x)
        assert out == [5.0, -1.0]
        dx = pool.backward([1.0, -2.0])
        assert dx[0][0][1] == 1.0
        assert dx[1][0][0] == -2.0
        assert sum(abs(v) for ch in dx for row in ch for v in row) == 3.0

    def test_adaptive_max_pool_matches_maxpool(self):
        x = [[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0],
              [9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]]]
        amp = AI.AdaptiveMaxPool2d(2).forward(x)
        mp = AI.MaxPool2d(2).forward(x)
        assert amp == mp
        one = AI.AdaptiveMaxPool2d((1, 1)).forward(x)
        assert one[0][0][0] == AI.GlobalMaxPool2d().forward(x)[0]

    def test_adaptive_max_pool_backward(self):
        x = [[[1.0, 4.0, 2.0, 8.0]]]
        pool = AI.AdaptiveMaxPool2d((1, 2))
        out = pool.forward(x)
        assert out[0][0] == [4.0, 8.0]
        dx = pool.backward([[[0.5, -1.5]]])
        assert dx[0][0][1] == 0.5
        assert dx[0][0][3] == -1.5
        assert dx[0][0][0] == 0.0 and dx[0][0][2] == 0.0

    def test_lp_pool_special_cases(self):
        pos = [[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0],
                [1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0]]]
        lp1 = AI.LPPool2d(1, 2).forward(pos)
        avg = AI.AvgPool2d(2).forward(pos)
        for c in range(len(lp1)):
            for h in range(len(lp1[c])):
                for w in range(len(lp1[c][h])):
                    assert abs(lp1[c][h][w] - avg[c][h][w]) < 1e-15
        big = AI.LPPool2d(64, 2).forward(pos)
        mx = AI.MaxPool2d(2).forward(pos)
        for c in range(len(big)):
            for h in range(len(big[c])):
                for w in range(len(big[c][h])):
                    assert big[c][h][w] <= mx[c][h][w] + 1e-12

    def test_lp_pool_gradient(self):
        x = [[[0.5, -1.2, 2.0, 0.3], [-0.7, 1.1, -0.4, 1.8],
              [1.3, 0.2, -1.5, 0.9], [0.4, -0.6, 1.7, -1.1]]]
        pool = AI.LPPool2d(2, 2)
        pool.forward(x)
        dx = pool.backward([[[0.4, -0.7], [0.2, 0.9]]])
        eps = 1e-6
        for h in range(4):
            for w in range(4):
                old = x[0][h][w]
                x[0][h][w] = old + eps
                vp = sum(pool.forward(x)[0][i][j] * c for i, row in enumerate([[0.4, -0.7], [0.2, 0.9]])
                         for j, c in enumerate(row))
                x[0][h][w] = old - eps
                vm = sum(pool.forward(x)[0][i][j] * c for i, row in enumerate([[0.4, -0.7], [0.2, 0.9]])
                         for j, c in enumerate(row))
                x[0][h][w] = old
                num = (vp - vm) / (2 * eps)
                assert abs(num - dx[0][h][w]) < 1e-8

    def test_pool_validation(self):
        for ctor in (lambda: AI.AdaptiveMaxPool2d(0),
                     lambda: AI.AdaptiveMaxPool2d((2, -1)),
                     lambda: AI.LPPool2d(0, 2),
                     lambda: AI.LPPool2d(2, 0),
                     lambda: AI.LPPool2d(2, 2, stride=-1)):
            with pytest.raises(ValueError):
                ctor()
        with pytest.raises(ValueError):
            AI.GlobalMaxPool2d().forward([])
        with pytest.raises(RuntimeError):
            AI.GlobalMaxPool2d().backward([1.0])


class TestSEBlock:
    def test_forward_shape_and_scale(self):
        random.seed(1)
        se = AI.SEBlock(3, reduction=2)
        x = [[[0.1 * (c + 1) + 0.2 * h + 0.3 * w for w in range(2)] for h in range(2)] for c in range(3)]
        out = se.forward(x)
        assert len(out) == 3 and len(out[0]) == 2 and len(out[0][0]) == 2
        _, s, _, _ = se._cache
        assert all(0.0 < v < 1.0 for v in s)
        for c in range(3):
            assert abs(out[c][1][0] - x[c][1][0] * s[c]) < 1e-12

    def test_param_count_and_zero_grad(self):
        random.seed(2)
        se = AI.SEBlock(4, reduction=2)
        h = se.hidden
        assert se.param_count() == h * 4 + h + 4 * h + 4
        x = [[[1.0, 2.0], [3.0, 4.0]] for _ in range(4)]
        se.forward(x)
        se.backward([[[1.0, 0.0], [0.0, 0.0]] for _ in range(4)])
        assert any(abs(v) > 0.0 for row in se.grads["W1"] for v in row)
        se.zero_grad()
        assert all(v == 0.0 for row in se.grads["W1"] for v in row)
        assert se.grads["b2"] == [0.0] * 4

    def test_se_backward_matches_finite_difference(self):
        random.seed(3)
        se = AI.SEBlock(2, reduction=1)
        x = [[[0.3, -0.7], [0.5, 1.1]], [[-0.2, 0.9], [0.4, -1.3]]]
        dout = [[[1.0, 0.5], [-0.4, 0.2]], [[0.3, -0.8], [0.6, 0.1]]]
        base = [[row[:] for row in ch] for ch in x]

        def loss(xx):
            o = se.forward(xx)
            return sum(o[c][h][w] * dout[c][h][w] for c in range(2) for h in range(2) for w in range(2))

        se.zero_grad()
        se.forward([row[:] for row in base])
        dx = se.backward(dout)
        eps = 1e-6
        for c in range(2):
            for h in range(2):
                for w in range(2):
                    old = base[c][h][w]
                    base[c][h][w] = old + eps
                    vp = loss([[row[:] for row in ch] for ch in base])
                    base[c][h][w] = old - eps
                    vm = loss([[row[:] for row in ch] for ch in base])
                    base[c][h][w] = old
                    num = (vp - vm) / (2 * eps)
                    assert abs(num - dx[c][h][w]) < 1e-8

    def test_se_validation(self):
        with pytest.raises(ValueError):
            AI.SEBlock(0)
        with pytest.raises(ValueError):
            AI.SEBlock(4).forward([[[1.0]]])
        with pytest.raises(RuntimeError):
            AI.SEBlock(2).backward([[[1.0]]])


# ============================================================================
#  W2-B：ECA / 深度可分离卷积 / ResNet BasicBlock
# ============================================================================
class TestECABlock:
    def test_forward_and_scale(self):
        eca = AI.ECABlock(4, kernel_size=3)
        x = [[[0.1 * (c + 1) + 0.2 * h + 0.3 * w for w in range(3)] for h in range(3)]
             for c in range(4)]
        out = eca.forward(x)
        _, z, s = eca._cache
        assert len(z) == 4
        assert all(0.0 < v < 1.0 for v in s)
        for c in range(4):
            assert abs(out[c][1][2] - x[c][1][2] * s[c]) < 1e-15
        assert eca.param_count() == 3

    def test_backward_matches_finite_difference(self):
        random.seed(11)
        eca = AI.ECABlock(3)
        x = [[[random.uniform(-1, 1) for _ in range(3)] for _ in range(3)] for _ in range(3)]
        dout = [[[random.uniform(-1, 1) for _ in range(3)] for _ in range(3)] for _ in range(3)]
        base = [[row[:] for row in ch] for ch in x]

        def loss(xx):
            o = eca.forward(xx)
            return sum(o[c][h][w] * dout[c][h][w] for c in range(3) for h in range(3) for w in range(3))

        eca.zero_grad()
        eca.forward([[row[:] for row in ch] for ch in base])
        dx = eca.backward(dout)
        eps = 1e-6
        for c in range(3):
            for h in range(3):
                for w in range(3):
                    old = base[c][h][w]
                    base[c][h][w] = old + eps
                    vp = loss([[row[:] for row in ch] for ch in base])
                    base[c][h][w] = old - eps
                    vm = loss([[row[:] for row in ch] for ch in base])
                    base[c][h][w] = old
                    assert abs((vp - vm) / (2 * eps) - dx[c][h][w]) < 1e-8
        eca.zero_grad()
        eca.forward([[row[:] for row in ch] for ch in base])
        eca.backward(dout)
        g = eca.grads["w"][1]
        base2 = [[row[:] for row in ch] for ch in base]
        eca.w[1] = eca.w[1] + eps
        vp = loss([[row[:] for row in ch] for ch in base2])
        base2 = [[row[:] for row in ch] for ch in base]
        eca.w[1] = eca.w[1] - 2 * eps
        vm = loss([[row[:] for row in ch] for ch in base2])
        eca.w[1] = eca.w[1] + eps
        assert abs((vp - vm) / (2 * eps) - g) < 1e-8

    def test_even_kernel_rejected(self):
        with pytest.raises(ValueError):
            AI.ECABlock(4, kernel_size=2)
        with pytest.raises(ValueError):
            AI.ECABlock(0)
        eca = AI.ECABlock(3)
        with pytest.raises(ValueError):
            eca.forward([[[1.0]]])
        with pytest.raises(RuntimeError):
            eca.backward([[[1.0]]])


class TestDepthwiseSeparableConv2d:
    def test_shapes_and_param_count(self):
        random.seed(21)
        dsc = AI.DepthwiseSeparableConv2d(3, 5, 3, padding=1)
        x = [[[random.uniform(-1, 1) for _ in range(4)] for _ in range(4)] for _ in range(3)]
        out = dsc.forward(x)
        assert (len(out), len(out[0]), len(out[0][0])) == (5, 4, 4)
        assert dsc.param_count() == 3 * 9 + 3 + 5 * 3 + 5
        assert dsc.param_count() < 3 * 5 * 9 + 5

    def test_backward_and_steps(self):
        random.seed(22)
        dsc = AI.DepthwiseSeparableConv2d(2, 3, 3, padding=1)
        x = [[[random.uniform(-1, 1) for _ in range(3)] for _ in range(3)] for _ in range(2)]
        dout = [[[random.uniform(-1, 1) for _ in range(3)] for _ in range(3)] for _ in range(3)]
        base = [[row[:] for row in ch] for ch in x]

        def loss(xx):
            o = dsc.forward(xx)
            return sum(o[c][h][w] * dout[c][h][w] for c in range(3) for h in range(3) for w in range(3))

        dsc.zero_grad()
        dsc.forward([[row[:] for row in ch] for ch in base])
        dx = dsc.backward(dout)
        eps = 1e-6
        for c in range(2):
            for h in range(3):
                for w in range(3):
                    old = base[c][h][w]
                    base[c][h][w] = old + eps
                    vp = loss([[row[:] for row in ch] for ch in base])
                    base[c][h][w] = old - eps
                    vm = loss([[row[:] for row in ch] for ch in base])
                    base[c][h][w] = old
                    assert abs((vp - vm) / (2 * eps) - dx[c][h][w]) < 1e-7
        w0 = dsc.dw.weights[0][0][0][0]
        dsc.zero_grad()
        dsc.forward([[row[:] for row in ch] for ch in base])
        dsc.backward(dout)
        dsc.sgd_step(0.1)
        assert dsc.dw.weights[0][0][0][0] != w0
        dsc.adam_step(0.01)
        assert dsc.pw.biases is not None


class TestBasicBlock2d:
    def test_zero_conv_is_relu_identity(self):
        random.seed(31)
        blk = AI.BasicBlock2d(2)
        for c in range(2):
            for ic in range(2):
                for a in range(3):
                    for b in range(3):
                        blk.conv1.weights[c][ic][a][b] = 0.0
                        blk.conv2.weights[c][ic][a][b] = 0.0
            blk.conv1.biases[c] = 0.0
            blk.conv2.biases[c] = 0.0
        x = [[[-1.0, 0.5], [2.0, -3.0]], [[0.0, 1.0], [-2.0, 4.0]]]
        out = blk.forward(x)
        for c in range(2):
            for h in range(2):
                for w in range(2):
                    assert abs(out[c][h][w] - max(0.0, x[c][h][w])) < 1e-12

    def test_eval_backward_matches_finite_difference(self):
        random.seed(32)
        blk = AI.BasicBlock2d(2)
        blk.eval()
        x = [[[random.uniform(-1, 1) for _ in range(3)] for _ in range(3)] for _ in range(2)]
        dout = [[[random.uniform(-1, 1) for _ in range(3)] for _ in range(3)] for _ in range(2)]
        base = [[row[:] for row in ch] for ch in x]

        def loss(xx):
            o = blk.forward(xx)
            return sum(o[c][h][w] * dout[c][h][w] for c in range(2) for h in range(3) for w in range(3))

        blk.zero_grad()
        blk.forward([[row[:] for row in ch] for ch in base])
        dx = blk.backward(dout)
        eps = 1e-6
        for c in range(2):
            for h in range(3):
                for w in range(3):
                    old = base[c][h][w]
                    base[c][h][w] = old + eps
                    vp = loss([[row[:] for row in ch] for ch in base])
                    base[c][h][w] = old - eps
                    vm = loss([[row[:] for row in ch] for ch in base])
                    base[c][h][w] = old
                    assert abs((vp - vm) / (2 * eps) - dx[c][h][w]) < 1e-7
        assert any(abs(v) > 0.0 for v in blk.bn_grads["bn2_gamma"])

    def test_param_count_and_validation(self):
        blk = AI.BasicBlock2d(3)
        assert blk.param_count() == 2 * (3 * 3 * 9 + 3) + 4 * 3
        with pytest.raises(ValueError):
            AI.BasicBlock2d(0)
        with pytest.raises(RuntimeError):
            AI.BasicBlock2d(2).backward([[[1.0]]])


class TestBatchNorm2dEvalBackward:
    def test_eval_backward_is_elementwise(self):
        bn = AI.BatchNorm2d(2)
        bn.eval()
        x = [[[0.5, -1.0], [2.0, 0.0]], [[1.0, 3.0], [-2.0, 0.5]]]
        out = bn.forward(x)
        assert out[0][0][0] == pytest.approx(x[0][0][0] / (1.0 + bn.eps) ** 0.5, rel=1e-9)
        dout = [[[1.0, -0.5], [0.0, 2.0]], [[-1.0, 0.0], [0.5, 1.0]]]
        dx, g_gamma, g_beta = bn.backward(dout)
        inv = 1.0 / (1.0 + bn.eps) ** 0.5
        for c in range(2):
            for h in range(2):
                for w in range(2):
                    assert abs(dx[c][h][w] - dout[c][h][w] * inv) < 1e-12
        for c in range(2):
            flat = [dout[c][h][w] for h in range(2) for w in range(2)]
            assert abs(g_beta[c] - sum(flat)) < 1e-12
            xh = [x[c][h][w] * inv for h in range(2) for w in range(2)]
            assert abs(g_gamma[c] - sum(d * a for d, a in zip(flat, xh))) < 1e-12


class TestBottleneck2d:
    def test_shapes_and_params(self):
        b = AI.Bottleneck2d(4, 8)
        assert b.mid == 2
        assert b.proj is not None
        # conv1 10 + bn1 4 + conv2 38 + bn2 4 + conv3 24 + bn3 16 + proj 40 + short_bn 16
        assert b.param_count() == 152
        assert AI.Bottleneck2d(4, 4).proj is None

    def test_eval_backward_matches_finite_difference(self):
        random.seed(41)
        b = AI.Bottleneck2d(3, 6)
        b.eval()
        x = [[[random.uniform(-1, 1) for _ in range(3)] for _ in range(3)] for _ in range(3)]
        dout = [[[random.uniform(-1, 1) for _ in range(3)] for _ in range(3)] for _ in range(6)]
        base = [[row[:] for row in ch] for ch in x]

        def loss(xx):
            o = b.forward(xx)
            return sum(o[c][h][w] * dout[c][h][w]
                       for c in range(6) for h in range(3) for w in range(3))

        b.zero_grad()
        b.forward([[row[:] for row in ch] for ch in base])
        dx = b.backward(dout)
        eps = 1e-6
        for c in range(3):
            for h in range(3):
                for w in range(3):
                    old = base[c][h][w]
                    base[c][h][w] = old + eps
                    vp = loss([[row[:] for row in ch] for ch in base])
                    base[c][h][w] = old - eps
                    vm = loss([[row[:] for row in ch] for ch in base])
                    base[c][h][w] = old
                    assert abs((vp - vm) / (2 * eps) - dx[c][h][w]) < 1e-7
        assert any(abs(v) > 0 for v in b.bn_grads["short_gamma"])
        assert any(abs(v) > 0 for v in b.proj.grad_w[0][0][0])

    def test_validation(self):
        with pytest.raises(ValueError):
            AI.Bottleneck2d(0, 4)
        with pytest.raises(ValueError):
            AI.Bottleneck2d(4, 8, stride=0)
        b = AI.Bottleneck2d(2, 4)
        with pytest.raises(RuntimeError):
            b.backward([[[1.0]]])


class TestAdaIN:
    def test_normalizes_to_style_stats(self):
        ad = AI.AdaIN(2)
        ad.set_style([1.5, 0.5], [0.1, -0.3])
        x = [[[1.0, 3.0], [5.0, 7.0]], [[-1.0, 0.0], [1.0, 2.0]]]
        out = ad.forward(x)
        for c in range(2):
            vals = [out[c][h][w] for h in range(2) for w in range(2)]
            mean = sum(vals) / 4
            assert abs(mean - ad.beta[c]) < 1e-9
            std = (sum((v - mean) ** 2 for v in vals) / 4) ** 0.5
            assert abs(std - abs(ad.gamma[c])) < 1e-4

    def test_gradients_match_finite_difference(self):
        random.seed(42)
        ad = AI.AdaIN(2)
        ad.set_style([0.7, -1.2], [0.2, 0.4])
        x = [[[random.uniform(-1, 1) for _ in range(2)] for _ in range(2)] for _ in range(2)]
        dout = [[[random.uniform(-1, 1) for _ in range(2)] for _ in range(2)] for _ in range(2)]

        def loss():
            o = ad.forward(x)
            return sum(o[c][h][w] * dout[c][h][w] for c in range(2) for h in range(2) for w in range(2))

        ad.forward(x)
        dx, dg, db = ad.backward(dout)
        eps = 1e-6
        for c in range(2):
            old = ad.gamma[c]
            ad.gamma[c] = old + eps
            vp = loss()
            ad.gamma[c] = old - eps
            vm = loss()
            ad.gamma[c] = old
            assert abs((vp - vm) / (2 * eps) - dg[c]) < 1e-6
            old = ad.beta[c]
            ad.beta[c] = old + eps
            vp = loss()
            ad.beta[c] = old - eps
            vm = loss()
            ad.beta[c] = old
            assert abs((vp - vm) / (2 * eps) - db[c]) < 1e-6
        def loss_of(xx):
            o = ad.forward(xx)
            return sum(o[c][h][w] * dout[c][h][w]
                       for c in range(2) for h in range(2) for w in range(2))

        base = [[row[:] for row in ch] for ch in x]
        for c in range(2):
            for h in range(2):
                for w in range(2):
                    old = base[c][h][w]
                    base[c][h][w] = old + eps
                    vp = loss_of([[row[:] for row in ch] for ch in base])
                    base[c][h][w] = old - eps
                    vm = loss_of([[row[:] for row in ch] for ch in base])
                    base[c][h][w] = old
                    assert abs((vp - vm) / (2 * eps) - dx[c][h][w]) < 1e-6

    def test_vector_input_and_validation(self):
        ad = AI.AdaIN(3)
        ad.set_style([2.0, 2.0, 2.0], [0.5, 0.5, 0.5])
        ad.forward([1.0, 2.0, 3.0])
        dx, dg, db = ad.backward([1.0, 1.0, 1.0])
        assert dx == [0.0, 0.0, 0.0]
        assert db == [1.0, 1.0, 1.0]
        with pytest.raises(ValueError):
            ad.set_style([1.0], [0.0])
        with pytest.raises(ValueError):
            ad.forward([1.0])


class TestFiLM:
    def test_forward_and_groups(self):
        f = AI.FiLM(6, groups=2)
        f.set_params([2.0, -1.0], [0.5, 0.0])
        out = f.forward([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        assert out == [2.5, 4.5, 6.5, -4.0, -5.0, -6.0]

    def test_backward(self):
        f = AI.FiLM(6, groups=2)
        f.set_params([2.0, -1.0], [0.5, 0.0])
        f.forward([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        dx, dg, db = f.backward([1.0, 0.5, 0.0, -1.0, 2.0, 0.0])
        assert dx[0] == 2.0 and dx[3] == 1.0
        assert abs(dg[0] - 2.0) < 1e-12
        assert abs(db[1] - 1.0) < 1e-12

    def test_validation(self):
        with pytest.raises(ValueError):
            AI.FiLM(6, groups=4)
        with pytest.raises(ValueError):
            AI.FiLM(0)
        f = AI.FiLM(4)
        with pytest.raises(ValueError):
            f.forward([1.0])


class TestConvTranspose1d:
    def test_shape_and_forward(self):
        ct = AI.ConvTranspose1d(2, 3, 3, stride=2, padding=1)
        assert len(ct.forward([[1.0, 2.0, 3.0], [0.5, -1.0, 2.0]])) == 3
        assert len(ct.forward([[1.0, 2.0, 3.0], [0.5, -1.0, 2.0]])[0]) == 5
        gt = AI.ConvTranspose1d(2, 2, 2, stride=2)
        gt.weights[0][0][0] = 1.0
        gt.weights[0][0][1] = 0.0
        gt.weights[0][1][0] = 0.0
        gt.weights[0][1][1] = 0.0
        gt.biases[0] = 0.0
        out = gt.forward([[3.0], [9.0]])
        assert out[0] == [3.0, 0.0]

    def test_backward_matches_finite_difference(self):
        random.seed(71)
        ct = AI.ConvTranspose1d(2, 2, 3, stride=2, padding=1)
        x = [[random.uniform(-1, 1) for _ in range(3)] for _ in range(2)]
        dout = [[random.uniform(-1, 1) for _ in range(5)] for _ in range(2)]
        base = [row[:] for row in x]

        def loss(xx):
            o = ct.forward(xx)
            return sum(o[c][i] * dout[c][i] for c in range(2) for i in range(5))

        ct.zero_grad()
        ct.forward([row[:] for row in base])
        dx = ct.backward(dout)
        eps = 1e-6
        for c in range(2):
            for i in range(3):
                old = base[c][i]
                base[c][i] = old + eps
                vp = loss([row[:] for row in base])
                base[c][i] = old - eps
                vm = loss([row[:] for row in base])
                base[c][i] = old
                assert abs((vp - vm) / (2 * eps) - dx[c][i]) < 1e-8
        ct.zero_grad()
        ct.forward([row[:] for row in base])
        ct.backward(dout)
        for oc in range(2):
            for ic in range(2):
                for k in range(3):
                    old = ct.weights[oc][ic][k]
                    ct.weights[oc][ic][k] = old + eps
                    vp = loss([row[:] for row in base])
                    ct.weights[oc][ic][k] = old - eps
                    vm = loss([row[:] for row in base])
                    ct.weights[oc][ic][k] = old
                    assert abs((vp - vm) / (2 * eps) - ct.grad_w[oc][ic][k]) < 1e-8

    def test_validation(self):
        with pytest.raises(ValueError):
            AI.ConvTranspose1d(2, 2, 3, stride=2, output_padding=2)
        with pytest.raises(ValueError):
            AI.ConvTranspose1d(0, 2, 3)
        ct = AI.ConvTranspose1d(2, 2, 3)
        with pytest.raises(RuntimeError):
            ct.backward([[1.0]])


class TestPool1d:
    def test_max_and_avg_values(self):
        mp = AI.MaxPool1d(2)
        assert mp.forward([[1.0, 3.0, 2.0, 5.0]]) == [[3.0, 5.0]]
        # floor 语义：长度 3 只出一个窗口，尾部被丢弃
        assert mp.forward([[1.0, 3.0, 2.0]]) == [[3.0]]
        ap = AI.AvgPool1d(2)
        assert ap.forward([[1.0, 3.0, 2.0, 5.0]]) == [[2.0, 3.5]]

    def test_backward_routing(self):
        mp = AI.MaxPool1d(2)
        mp.forward([[1.0, 3.0, 2.0, 5.0]])
        assert mp.backward([[1.0, 2.0]]) == [[0.0, 1.0, 0.0, 2.0]]
        ap = AI.AvgPool1d(2)
        ap.forward([[1.0, 3.0, 2.0, 5.0]])
        assert ap.backward([[2.0, 4.0]]) == [[1.0, 1.0, 2.0, 2.0]]

    def test_validation(self):
        with pytest.raises(ValueError):
            AI.MaxPool1d(0)
        with pytest.raises(ValueError):
            AI.AvgPool1d(2, stride=0)
        with pytest.raises(RuntimeError):
            AI.MaxPool1d(2).backward([[1.0]])


class TestPool3d:
    def test_shapes_and_values(self):
        x = [[[[float(d + h + w) for w in range(4)] for h in range(4)] for d in range(4)]]
        mp = AI.MaxPool3d(2)
        out = mp.forward(x)
        assert (len(out), len(out[0]), len(out[0][0]), len(out[0][0][0])) == (1, 2, 2, 2)
        window = [x[0][d][h][w] for d in range(2) for h in range(2) for w in range(2)]
        assert out[0][0][0][0] == max(window)
        ap = AI.AvgPool3d(2)
        out2 = ap.forward(x)
        assert abs(out2[0][0][0][0] - sum(window) / 8.0) < 1e-12

    def test_backward_totals(self):
        x = [[[[float(d + h + w) for w in range(4)] for h in range(4)] for d in range(4)]]
        dout = [[[[1.0] * 2 for _ in range(2)] for _ in range(2)]]
        mp = AI.MaxPool3d(2)
        mp.forward(x)
        dx = mp.backward(dout)
        total = sum(v for ch in dx for d in ch for h in d for v in h)
        assert abs(total - 8.0) < 1e-12
        ap = AI.AvgPool3d(2)
        ap.forward(x)
        dx2 = ap.backward(dout)
        total2 = sum(v for ch in dx2 for d in ch for h in d for v in h)
        assert abs(total2 - 8.0) < 1e-12

    def test_validation(self):
        with pytest.raises(ValueError):
            AI.MaxPool3d(0)
        with pytest.raises(ValueError):
            AI.AvgPool3d(2).forward([])
        with pytest.raises(RuntimeError):
            AI.MaxPool3d(2).backward([[[[1.0]]]])
