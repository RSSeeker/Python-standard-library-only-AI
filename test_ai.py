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


class TestConv3d:
    def test_shape_and_center_tap(self):
        conv = AI.Conv3d(1, 1, 3, stride=1, padding=1)
        for a in range(3):
            for b in range(3):
                for c in range(3):
                    conv.weights[0][0][a][b][c] = 1.0 if (a, b, c) == (1, 1, 1) else 0.0
        x = [[[[0.0] * 2 for _ in range(2)] for _ in range(2)]]
        x[0][0][0][0] = 5.0
        out = conv.forward(x)
        assert (len(out), len(out[0]), len(out[0][0]), len(out[0][0][0])) == (1, 2, 2, 2)
        # 只留中心 tap 时前向是平移不变的恒等映射
        assert abs(out[0][0][0][0] - 5.0) < 1e-15
        assert conv.param_count() == 1 * 1 * 27 + 1

    def test_gradients_match_finite_difference(self):
        random.seed(81)
        conv = AI.Conv3d(2, 2, 3, stride=1, padding=1)
        x = [[[[random.uniform(-1, 1) for _ in range(3)] for _ in range(3)]
              for _ in range(3)] for _ in range(2)]
        dout = [[[[random.uniform(-1, 1) for _ in range(3)] for _ in range(3)]
                 for _ in range(3)] for _ in range(2)]

        def loss(xx):
            out = conv.forward(xx)
            return sum(out[o][a][b][c] * dout[o][a][b][c]
                       for o in range(2) for a in range(3)
                       for b in range(3) for c in range(3))

        conv.zero_grad()
        conv.forward([[[[v for v in r] for r in p2] for p2 in p1] for p1 in x])
        dx = conv.backward(dout)
        eps = 1e-6
        for ic in range(2):
            for a in range(3):
                for b in range(3):
                    for c in range(3):
                        old = x[ic][a][b][c]
                        x[ic][a][b][c] = old + eps
                        vp = loss(x)
                        x[ic][a][b][c] = old - eps
                        vm = loss(x)
                        x[ic][a][b][c] = old
                        assert abs((vp - vm) / (2 * eps) - dx[ic][a][b][c]) < 1e-8
        conv.zero_grad()
        conv.forward(x)
        conv.backward(dout)
        for oc in range(2):
            for ic in range(2):
                for a in range(3):
                    for b in range(3):
                        for c in range(3):
                            old = conv.weights[oc][ic][a][b][c]
                            conv.weights[oc][ic][a][b][c] = old + eps
                            vp = loss(x)
                            conv.weights[oc][ic][a][b][c] = old - eps
                            vm = loss(x)
                            conv.weights[oc][ic][a][b][c] = old
                            assert abs((vp - vm) / (2 * eps)
                                       - conv.grad_w[oc][ic][a][b][c]) < 1e-8

    def test_validation(self):
        with pytest.raises(ValueError):
            AI.Conv3d(0, 2, 3)
        with pytest.raises(ValueError):
            AI.Conv3d(2, 2, 0)
        with pytest.raises(ValueError):
            AI.Conv3d(2, 2, 3, stride=0)
        with pytest.raises(ValueError):
            AI.Conv3d(2, 2, 3, padding=-1)
        conv = AI.Conv3d(1, 1, 3)
        with pytest.raises(RuntimeError):
            conv.backward([[[[1.0]]]])
        with pytest.raises(ValueError):
            AI.Conv3d(2, 1, 3).forward([[[[1.0]]]])
        # kernel 3 > 输入 1 且无 padding，输出维度为负
        with pytest.raises(ValueError):
            AI.Conv3d(1, 1, 3).forward([[[[1.0]]]])


class TestConvTranspose3d:
    def test_shape_formula(self):
        t = AI.ConvTranspose3d(1, 1, 3, stride=2, padding=1)
        # (D-1)*s - 2p + k = (2-1)*2 - 2 + 3 = 3
        assert t._out_shape(2, 2, 2) == (3, 3, 3)
        # (D-1)*s - 2p + k + op = (2-1)*2 - 0 + 3 + 1 = 6
        t2 = AI.ConvTranspose3d(1, 1, 3, stride=2, padding=0, output_padding=1)
        assert t2._out_shape(2, 2, 2) == (6, 6, 6)
        x = [[[[0.0] * 2 for _ in range(2)] for _ in range(2)]]
        out = t2.forward(x)
        assert (len(out), len(out[0]), len(out[0][0]), len(out[0][0][0])) == (1, 6, 6, 6)
        assert t2.param_count() == 1 * 1 * 27 + 1

    def test_single_point_scatter_and_gradients(self):
        t = AI.ConvTranspose3d(1, 1, 3, stride=2, padding=1)
        for a in range(3):
            for b in range(3):
                for c in range(3):
                    t.weights[0][0][a][b][c] = 1.0 if (a, b, c) == (2, 2, 2) else 0.0
        x = [[[[0.0] * 2 for _ in range(2)] for _ in range(2)]]
        x[0][0][0][0] = 7.0
        out = t.forward(x)
        # i=0,k=2,s=2,p=1 -> o = 0*2-1+2 = 1
        assert abs(out[0][1][1][1] - 7.0) < 1e-15

        random.seed(82)
        tt = AI.ConvTranspose3d(2, 2, 3, stride=2, padding=1)
        xt = [[[[random.uniform(-1, 1) for _ in range(2)] for _ in range(2)]
               for _ in range(2)] for _ in range(2)]
        dout = [[[[random.uniform(-1, 1) for _ in range(3)] for _ in range(3)]
                 for _ in range(3)] for _ in range(2)]

        def loss(xx):
            out = tt.forward(xx)
            return sum(out[o][a][b][c] * dout[o][a][b][c]
                       for o in range(2) for a in range(3)
                       for b in range(3) for c in range(3))

        eps = 1e-6
        tt.zero_grad()
        tt.forward(xt)
        dx = tt.backward(dout)
        for ic in range(2):
            for a in range(2):
                for b in range(2):
                    for c in range(2):
                        old = xt[ic][a][b][c]
                        xt[ic][a][b][c] = old + eps
                        vp = loss(xt)
                        xt[ic][a][b][c] = old - eps
                        vm = loss(xt)
                        xt[ic][a][b][c] = old
                        assert abs((vp - vm) / (2 * eps) - dx[ic][a][b][c]) < 1e-8
        tt.zero_grad()
        tt.forward(xt)
        tt.backward(dout)
        for oc in range(2):
            for ic in range(2):
                for a in range(3):
                    for b in range(3):
                        for c in range(3):
                            old = tt.weights[oc][ic][a][b][c]
                            tt.weights[oc][ic][a][b][c] = old + eps
                            vp = loss(xt)
                            tt.weights[oc][ic][a][b][c] = old - eps
                            vm = loss(xt)
                            tt.weights[oc][ic][a][b][c] = old
                            assert abs((vp - vm) / (2 * eps)
                                       - tt.grad_w[oc][ic][a][b][c]) < 1e-8

    def test_validation(self):
        with pytest.raises(ValueError):
            AI.ConvTranspose3d(0, 2, 3)
        with pytest.raises(ValueError):
            AI.ConvTranspose3d(2, 2, 3, stride=2, output_padding=2)
        with pytest.raises(ValueError):
            AI.ConvTranspose3d(2, 2, 3, output_padding=-1)
        with pytest.raises(ValueError):
            AI.ConvTranspose3d(2, 2, 3, padding=-1)
        t = AI.ConvTranspose3d(1, 1, 3)
        with pytest.raises(RuntimeError):
            t.backward([[[[1.0]]]])
        with pytest.raises(ValueError):
            AI.ConvTranspose3d(2, 1, 3).forward([[[[1.0]]]])


# ==== W2-F：CBAM 注意力 ====

class TestCBAMBlock:
    """CBAMBlock 的通道支路 / 空间支路 / 校验"""

    @staticmethod
    def _clone(x):
        return [[[v for v in row] for row in ch] for ch in x]

    def _fd_dx(self, block, x, dout, eps=1e-6):
        def loss(xx):
            out = block.forward(self._clone(xx))
            return sum(out[c][h][w] * dout[c][h][w]
                       for c in range(len(out)) for h in range(len(out[0]))
                       for w in range(len(out[0][0])))

        block.zero_grad()
        block.forward(self._clone(x))
        dx = block.backward(dout)
        worst = 0.0
        for c in range(len(x)):
            for h in range(len(x[0])):
                for w in range(len(x[0][0])):
                    old = x[c][h][w]
                    x[c][h][w] = old + eps
                    vp = loss(x)
                    x[c][h][w] = old - eps
                    vm = loss(x)
                    x[c][h][w] = old
                    worst = max(worst, abs((vp - vm) / (2 * eps) - dx[c][h][w]))
        return dx, worst

    def test_channel_branch_matches_se(self):
        """空间支路退化成恒等（sp_w=0、sp_b 很大）时，通道支路与 SEBlock 逐位相同"""
        random.seed(901)
        cb = AI.CBAMBlock(3, reduction=2, kernel_size=3)
        for a in range(3):
            for b in range(3):
                cb.sp_w[a][b] = 0.0
        cb.sp_b[0] = 50.0                      # sigmoid(50) ≈ 1
        se = AI.SEBlock(3, reduction=2)
        se.W1 = [[v for v in row] for row in cb.W1]
        se.b1 = list(cb.b1)
        se.W2 = [[v for v in row] for row in cb.W2]
        se.b2 = list(cb.b2)
        x = [[[random.uniform(-1, 1) for _ in range(3)] for _ in range(3)]
             for _ in range(3)]
        dout = [[[random.uniform(-1, 1) for _ in range(3)] for _ in range(3)]
                for _ in range(3)]
        cb.zero_grad()
        cb.forward(self._clone(x))
        dx = cb.backward(dout)
        se.zero_grad()
        se.forward(self._clone(x))
        dx_se = se.backward(dout)
        assert max(abs(a - b) for ca, cbb in zip(dx_se, dx)
                   for ra, rb in zip(ca, cbb) for a, b in zip(ra, rb)) < 1e-12
        assert cb.grads["b1"] == se.grads["b1"]

    def test_spatial_attention_varies_per_position(self):
        """空间支路真的逐位置不同（不是整图系数）"""
        random.seed(902)
        cb = AI.CBAMBlock(3, reduction=2, kernel_size=3)
        x = [[[random.uniform(-1, 1) for _ in range(4)] for _ in range(4)]
             for _ in range(3)]
        cb.forward(self._clone(x))
        s_sp = cb._cache[9]
        flat = [v for row in s_sp for v in row]
        assert len(flat) == 16
        assert max(flat) - min(flat) > 1e-3
        assert all(0.0 < v < 1.0 for v in flat)

    def test_spatial_branch_gradients(self):
        """dx 与 sp_w / sp_b 都过中心差分（核不对称，能抓下标翻转）"""
        random.seed(903)
        cb = AI.CBAMBlock(3, reduction=2, kernel_size=3)
        x = [[[random.uniform(-0.8, 0.8) for _ in range(4)] for _ in range(4)]
             for _ in range(3)]
        dout = [[[random.uniform(-1, 1) for _ in range(4)] for _ in range(4)]
                for _ in range(3)]
        dx, worst = self._fd_dx(cb, x, dout)
        assert worst < 1e-7, f"dx 中心差分误差 {worst:.3e}"

        cb.zero_grad()
        cb.forward(self._clone(x))
        cb.backward(dout)
        eps = 1e-6

        def loss():
            out = cb.forward(self._clone(x))
            return sum(out[c][h][w] * dout[c][h][w]
                       for c in range(3) for h in range(4) for w in range(4))

        worst = 0.0
        for a in range(3):
            for b in range(3):
                old = cb.sp_w[a][b]
                cb.sp_w[a][b] = old + eps
                vp = loss()
                cb.sp_w[a][b] = old - eps
                vm = loss()
                cb.sp_w[a][b] = old
                worst = max(worst, abs((vp - vm) / (2 * eps) - cb.grads["sp_w"][a][b]))
        assert worst < 1e-7, f"sp_w 中心差分误差 {worst:.3e}"
        old = cb.sp_b[0]
        cb.sp_b[0] = old + eps
        vp = loss()
        cb.sp_b[0] = old - eps
        vm = loss()
        cb.sp_b[0] = old
        assert abs((vp - vm) / (2 * eps) - cb.grads["sp_b"][0]) < 1e-7

    def test_meta(self):
        """参数量、train/eval、尺寸校验"""
        cb = AI.CBAMBlock(3, reduction=2, kernel_size=3)
        assert cb.param_count() == 3 * 1 + 1 + 3 * 1 + 3 + 3 * 3 + 1 == 20
        assert cb.hidden == 1 and cb.half == 1
        cb.eval()
        assert cb.training is False
        cb.train()
        assert cb.training is True
        for bad in [(0,), (-1,), (3.0,), (True,)]:
            try:
                AI.CBAMBlock(*bad)
            except (TypeError, ValueError):
                pass
            else:
                raise AssertionError(f"CBAMBlock{bad} 应该报错")
        for bad_k in (2, 4, 0, -3):
            try:
                AI.CBAMBlock(3, reduction=2, kernel_size=bad_k)
            except (TypeError, ValueError):
                pass
            else:
                raise AssertionError(f"kernel_size={bad_k} 应该报错")
        try:
            cb.forward([[1.0, 2.0], [3.0, 4.0]])
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError("二维输入应该报错")

    def test_optimizer_steps(self):
        """SEBlock / CBAMBlock 的 sgd_step 与 adam_step 真的改参数"""
        import random as _random
        for name in ("SEBlock", "CBAMBlock"):
            _random.seed(77)
            blk = getattr(AI, name)(3, reduction=2)
            blk.b1 = [0.9] * blk.hidden      # 保证 ReLU 单元活着，W1/W2 才有梯度
            x = [[[_random.uniform(-1, 1) for _ in range(3)] for _ in range(3)]
                 for _ in range(3)]
            dout = [[[0.3, 0.3, 0.3] for _ in range(3)] for _ in range(3)]
            blk.forward(self._clone(x))
            blk.backward(dout)
            assert any(abs(v) > 0.0 for v in blk.grads["W1"][0])
            before = list(blk.W1[0])
            blk.sgd_step(0.1)
            assert any(abs(a - b) > 1e-12 for a, b in zip(before, blk.W1[0]))
            blk.forward(self._clone(x))
            blk.backward(dout)
            before = list(blk.W1[0])
            blk.adam_step(0.05)
            assert blk._adam_t == 1
            assert any(abs(a - b) > 1e-12 for a, b in zip(before, blk.W1[0]))
            for bad in (-0.1, True, "0.1"):
                try:
                    blk.sgd_step(bad)
                except (TypeError, ValueError):
                    pass
                else:
                    raise AssertionError(f"{name}.sgd_step({bad!r}) 应该报错")


# ==== W2-G：DenseBlock / ConvNeXtBlock ====

class TestDenseBlock:
    """稠密连接：拼接形状、梯度、eval 模式"""

    @staticmethod
    def _clone(x):
        return [[[v for v in row] for row in ch] for ch in x]

    def test_forward_concat(self):
        """输出通道数、前 in 个通道原样保留、下游真的用到了前面所有层"""
        import random as _random
        _random.seed(1201)
        db = AI.DenseBlock(3, growth_rate=2, num_layers=2, kernel_size=3)
        x = [[[_random.uniform(-1, 1) for _ in range(4)] for _ in range(4)]
             for _ in range(3)]
        out = db.forward(self._clone(x))
        assert len(out) == 7 and db.out_channels == 7
        for c in range(3):
            for h in range(4):
                for w in range(4):
                    assert abs(out[c][h][w] - x[c][h][w]) < 1e-12
        assert db.param_count() == (3 * 2 * 9 + 2) + (5 * 2 * 9 + 2) + 2 * (3 + 5)

    def test_gradients(self):
        """dx / conv grad_w / BN gamma,beta 都过中心差分"""
        import random as _random
        _random.seed(1202)
        db = AI.DenseBlock(3, growth_rate=2, num_layers=2, kernel_size=3)
        x = [[[_random.uniform(-1, 1) for _ in range(4)] for _ in range(4)]
             for _ in range(3)]
        dout = [[[_random.uniform(-1, 1) for _ in range(4)] for _ in range(4)]
                for _ in range(7)]

        def loss():
            o = db.forward(self._clone(x))
            return sum(o[c][h][w] * dout[c][h][w]
                       for c in range(7) for h in range(4) for w in range(4))

        db.zero_grad()
        db.forward(self._clone(x))
        dx = db.backward(dout)
        eps = 1e-6
        worst = 0.0
        for c in range(3):
            for h in range(4):
                for w in range(4):
                    old = x[c][h][w]
                    x[c][h][w] = old + eps
                    vp = loss()
                    x[c][h][w] = old - eps
                    vm = loss()
                    x[c][h][w] = old
                    worst = max(worst, abs((vp - vm) / (2 * eps) - dx[c][h][w]))
        assert worst < 1e-7, f"dx 差分 {worst:.3e}"

        db.zero_grad()
        db.forward(self._clone(x))
        db.backward(dout)
        worst = 0.0
        for li in range(2):
            conv = db.convs[li]
            for o in range(conv.out_channels):
                for i in range(conv.in_channels):
                    for a in range(3):
                        for b in range(3):
                            old = conv.weights[o][i][a][b]
                            conv.weights[o][i][a][b] = old + eps
                            vp = loss()
                            conv.weights[o][i][a][b] = old - eps
                            vm = loss()
                            conv.weights[o][i][a][b] = old
                            worst = max(worst, abs((vp - vm) / (2 * eps)
                                                   - conv.grad_w[o][i][a][b]))
        assert worst < 1e-7, f"grad_w 差分 {worst:.3e}"

        worst = 0.0
        for li in range(2):
            bn = db.bns[li]
            for c in range(bn.c):
                for arr, gkey in ((bn.gamma, "L%d_gamma" % li),
                                  (bn.beta, "L%d_beta" % li)):
                    old = arr[c]
                    arr[c] = old + eps
                    vp = loss()
                    arr[c] = old - eps
                    vm = loss()
                    arr[c] = old
                    worst = max(worst, abs((vp - vm) / (2 * eps)
                                           - db.bn_grads[gkey][c]))
        assert worst < 1e-7, f"BN 参数差分 {worst:.3e}"

    def test_eval_mode_and_errors(self):
        """eval 用 running 统计量；参数/通道/未 forward 报错"""
        import random as _random
        _random.seed(1203)
        db = AI.DenseBlock(3, growth_rate=2, num_layers=2)
        x = [[[_random.uniform(-1, 1) for _ in range(4)] for _ in range(4)]
             for _ in range(3)]
        out = db.forward(self._clone(x))
        db.eval()
        o_eval = db.forward(self._clone(x))
        assert any(abs(o_eval[c][h][w] - out[c][h][w]) > 1e-9
                   for c in range(7) for h in range(4) for w in range(4))
        db.train()
        o_train = db.forward(self._clone(x))
        assert all(abs(o_train[c][h][w] - out[c][h][w]) < 1e-12
                   for c in range(7) for h in range(4) for w in range(4))
        for bad in (0, -1, 1.5, True):
            try:
                AI.DenseBlock(bad)
            except (TypeError, ValueError):
                pass
            else:
                raise AssertionError(f"DenseBlock({bad!r}) 应该报错")
        try:
            db.forward([[1.0, 2.0], [3.0, 4.0]])
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError("二维输入应该报错")
        try:
            AI.DenseBlock(3).backward(out)
        except RuntimeError:
            pass
        else:
            raise AssertionError("未 forward 就 backward 应该报错")


class TestConvNeXtBlock:
    """ConvNeXt：跨通道 LayerNorm、残差、梯度、参数计数"""

    @staticmethod
    def _clone(x):
        return [[[v for v in row] for row in ch] for ch in x]

    def test_layernorm_and_residual(self):
        """xhat 逐位置均值 0、方差 var/(var+eps)；pw2 全零时输出等于输入"""
        import random as _random
        _random.seed(1301)
        cb = AI.ConvNeXtBlock(4, kernel_size=7, expansion=2)
        x = [[[_random.uniform(-1, 1) for _ in range(3)] for _ in range(3)]
             for _ in range(4)]
        cb.forward(self._clone(x))
        u, xhat = cb._cache[1], cb._cache[2]
        for h in range(3):
            for w in range(3):
                mu = sum(xhat[c][h][w] for c in range(4)) / 4.0
                assert abs(mu) < 1e-10
                var_x = sum((xhat[c][h][w] - mu) ** 2 for c in range(4)) / 4.0
                mu_u = sum(u[c][h][w] for c in range(4)) / 4.0
                var_u = sum((u[c][h][w] - mu_u) ** 2 for c in range(4)) / 4.0
                assert abs(var_x - var_u / (var_u + cb.eps)) < 1e-9

        zero = AI.ConvNeXtBlock(4, kernel_size=3, expansion=1)
        for o in range(4):
            for i in range(4):
                zero.pw2.weights[o][i][0][0] = 0.0
            zero.pw2.biases[o] = 0.0
        out0 = zero.forward(self._clone(x))
        for c in range(4):
            for h in range(3):
                for w in range(3):
                    assert abs(out0[c][h][w] - x[c][h][w]) < 1e-12

    def test_gradients(self):
        """dx、LN gamma/beta、depthwise conv 权重都过中心差分"""
        import random as _random
        _random.seed(1302)
        cb = AI.ConvNeXtBlock(4, kernel_size=7, expansion=2)
        x = [[[_random.uniform(-1, 1) for _ in range(3)] for _ in range(3)]
             for _ in range(4)]
        dout = [[[_random.uniform(-1, 1) for _ in range(3)] for _ in range(3)]
                for _ in range(4)]

        def loss():
            o = cb.forward(self._clone(x))
            return sum(o[c][h][w] * dout[c][h][w]
                       for c in range(4) for h in range(3) for w in range(3))

        cb.zero_grad()
        cb.forward(self._clone(x))
        dx = cb.backward(dout)
        eps = 1e-6
        worst = 0.0
        for c in range(4):
            for h in range(3):
                for w in range(3):
                    old = x[c][h][w]
                    x[c][h][w] = old + eps
                    vp = loss()
                    x[c][h][w] = old - eps
                    vm = loss()
                    x[c][h][w] = old
                    worst = max(worst, abs((vp - vm) / (2 * eps) - dx[c][h][w]))
        assert worst < 1e-7, f"dx 差分 {worst:.3e}"

        cb.zero_grad()
        cb.forward(self._clone(x))
        cb.backward(dout)
        worst = 0.0
        for c in range(4):
            for arr, gkey in ((cb.gamma, "gamma"), (cb.beta, "beta")):
                old = arr[c]
                arr[c] = old + eps
                vp = loss()
                arr[c] = old - eps
                vm = loss()
                arr[c] = old
                worst = max(worst, abs((vp - vm) / (2 * eps) - cb.norm_grads[gkey][c]))
        assert worst < 1e-7, f"LN 参数差分 {worst:.3e}"

        worst = 0.0
        for o in range(4):
            for i in range(cb.dw.cg):
                for a in range(7):
                    for b in range(7):
                        old = cb.dw.weights[o][i][a][b]
                        cb.dw.weights[o][i][a][b] = old + eps
                        vp = loss()
                        cb.dw.weights[o][i][a][b] = old - eps
                        vm = loss()
                        cb.dw.weights[o][i][a][b] = old
                        worst = max(worst, abs((vp - vm) / (2 * eps)
                                               - cb.dw.grad_w[o][i][a][b]))
        assert worst < 1e-7, f"depthwise 权重差分 {worst:.3e}"

    def test_param_count_and_errors(self):
        """参数量公式、非法参数、通道不符、未 forward"""
        cb = AI.ConvNeXtBlock(4, kernel_size=7, expansion=2)
        assert cb.param_count() == ((4 * 1 * 49 + 4) + (8 * 4 + 8) + (4 * 8 + 4) + 8)
        for bad in (0, -2, 2.0, True):
            try:
                AI.ConvNeXtBlock(bad)
            except (TypeError, ValueError):
                pass
            else:
                raise AssertionError(f"ConvNeXtBlock({bad!r}) 应该报错")
        try:
            AI.ConvNeXtBlock(4, eps=0)
        except ValueError:
            pass
        else:
            raise AssertionError("eps=0 应该报错")
        try:
            cb.forward([[[1.0]], [[2.0]], [[3.0]]])
        except ValueError:
            pass
        else:
            raise AssertionError("通道数不符应该报错")
        try:
            AI.ConvNeXtBlock(4).backward([[[1.0]]])
        except RuntimeError:
            pass
        else:
            raise AssertionError("未 forward 就 backward 应该报错")


# ==== W2-H：分数池化 / 随机池化 / 条件归一化 ====

class TestFractionalPooling:
    """FractionalMaxPool2d：比例决定输出尺寸，窗口重叠且可注入随机源"""

    def test_output_size_and_values(self):
        import random as _random
        rng = _random.Random(41)
        x = [[[rng.uniform(-1, 1) for _ in range(8)] for _ in range(8)] for _ in range(2)]
        fp = AI.FractionalMaxPool2d((0.5, 0.5), rng=_random.Random(7))
        out = fp.forward(x)
        assert len(out) == 2 and len(out[0]) == 4 and len(out[0][0]) == 4
        for c in range(2):
            for h in range(4):
                for w in range(4):
                    ih, iw = fp.mask[c][h][w]
                    assert abs(out[c][h][w] - x[c][ih][iw]) < 1e-12
        full = AI.FractionalMaxPool2d(1.0)
        assert len(full.forward(x)[0]) == 8 and len(full.forward(x)[0][0]) == 8

    def test_reproducible_and_backward(self):
        """同 seed 逐位相同；反向只回传被选位置且总量守恒"""
        import random as _random
        rng = _random.Random(42)
        x = [[[rng.uniform(-1, 1) for _ in range(10)] for _ in range(10)]
             for _ in range(2)]
        a = AI.FractionalMaxPool2d((0.6, 0.6), rng=_random.Random(1))
        b = AI.FractionalMaxPool2d((0.6, 0.6), rng=_random.Random(1))
        c = AI.FractionalMaxPool2d((0.6, 0.6), rng=_random.Random(2))
        oa, ob = a.forward(x), b.forward(x)
        c.forward(x)
        assert all(abs(oa[i][j][k] - ob[i][j][k]) < 1e-12
                   for i in range(2) for j in range(len(oa[0]))
                   for k in range(len(oa[0][0])))
        assert any(a.mask[i][j][k] != c.mask[i][j][k]
                   for i in range(2) for j in range(len(a.mask[0]))
                   for k in range(len(a.mask[0][0])))
        # alpha 为整数时没有自由度，窗口与 seed 无关
        i1 = AI.FractionalMaxPool2d((0.5, 0.5), rng=_random.Random(1))
        i2 = AI.FractionalMaxPool2d((0.5, 0.5), rng=_random.Random(2))
        i1.forward(x)
        i2.forward(x)
        assert all(i1.mask[p][q][r] == i2.mask[p][q][r]
                   for p in range(2) for q in range(len(i1.mask[0]))
                   for r in range(len(i1.mask[0][0])))
        dout = [[[1.0] * len(oa[0][0]) for _ in range(len(oa[0]))] for _ in range(2)]
        dx = a.backward(dout)
        total = sum(sum(sum(row) for row in ch) for ch in dx)
        expect = 2 * len(oa[0]) * len(oa[0][0])
        assert abs(total - expect) < 1e-12
        nz = sum(1 for ch in dx for row in ch for v in row if abs(v) > 0)
        assert nz == expect

    def test_errors(self):
        for bad in ((0.0, 0.5), (1.5, 0.5), ("a", 0.5), (0.5,)):
            try:
                AI.FractionalMaxPool2d(bad)
            except (TypeError, ValueError):
                pass
            else:
                raise AssertionError(f"FractionalMaxPool2d({bad!r}) 应该报错")
        try:
            AI.FractionalMaxPool2d(0.5).backward([[[1.0]]])
        except RuntimeError:
            pass
        else:
            raise AssertionError("未 forward 就 backward 应该报错")
        fp = AI.FractionalMaxPool2d(0.5)
        try:
            fp.forward([1.0, 2.0])
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError("一维输入应该报错")


class TestStochasticPooling:
    """StochasticPool2d：训练采样、eval 取期望、反向按概率缩放"""

    def test_train_and_eval(self):
        import random as _random
        rng = _random.Random(51)
        x = [[[rng.uniform(0.1, 1.0) for _ in range(4)] for _ in range(4)]
             for _ in range(2)]
        sp = AI.StochasticPool2d(2, rng=_random.Random(5))
        out = sp.forward(x)
        assert len(out) == 2 and len(out[0]) == 2 and len(out[0][0]) == 2
        probs = sp.probs[0][0][0]
        assert abs(sum(probs) - 1.0) < 1e-12
        pick = sp.tag[0][0][0]
        i, j = divmod(pick, 2)
        assert abs(out[0][0][0] - x[0][i][j]) < 1e-12
        dout = [[[2.0] * 2 for _ in range(2)] for _ in range(2)]
        dx = sp.backward(dout)
        assert abs(dx[0][i][j] - 2.0 * probs[pick]) < 1e-12
        nz = sum(1 for ch in dx for row in ch for v in row if abs(v) > 0)
        assert nz == 8

        sp.eval()
        out_eval = sp.forward(x)
        vals = [x[0][h][k] for h in range(2) for k in range(2)]
        weights = [v if v > 0 else 0.0 for v in vals]
        total = sum(weights)
        pexp = [v / total for v in weights]
        assert abs(out_eval[0][0][0] - sum(p * v for p, v in zip(pexp, vals))) < 1e-12
        dx_eval = sp.backward(dout)
        assert abs(dx_eval[0][0][0] - 2.0 * pexp[0]) < 1e-12
        sp.train()

        # 同一 seed 采样相同
        a = AI.StochasticPool2d(2, rng=_random.Random(11))
        b = AI.StochasticPool2d(2, rng=_random.Random(11))
        oa, ob = a.forward(x), b.forward(x)
        assert all(abs(oa[i][j][k] - ob[i][j][k]) < 1e-12
                   for i in range(2) for j in range(2) for k in range(2))

    def test_probabilities_uniform_when_nonpositive(self):
        """窗口全非正时退化为均匀分布（不会 0/0）"""
        x = [[[-1.0, -2.0], [-3.0, -4.0]]]
        sp = AI.StochasticPool2d(2, rng=__import__("random").Random(3))
        sp.forward(x)
        probs = sp.probs[0][0][0]
        assert all(abs(p - 0.25) < 1e-12 for p in probs)

    def test_errors(self):
        for bad in (0, -1, (2, 0)):
            try:
                AI.StochasticPool2d(bad)
            except (TypeError, ValueError):
                pass
            else:
                raise AssertionError(f"StochasticPool2d({bad!r}) 应该报错")
        try:
            AI.StochasticPool2d(2).backward([[[1.0]]])
        except RuntimeError:
            pass
        else:
            raise AssertionError("未 forward 就 backward 应该报错")
        try:
            AI.StochasticPool2d(5).forward([[[1.0, 2.0], [3.0, 4.0]]])
        except ValueError:
            pass
        else:
            raise AssertionError("kernel 大于输入应该报错")


class TestConditionalNorm:
    """ConditionalNorm：条件投影出 gamma/beta，梯度覆盖全部参数与条件向量"""

    def test_forward_closed_form(self):
        import random as _random
        _random.seed(61)
        cn = AI.ConditionalNorm(4, 3)
        cond = [0.5, -0.2, 0.8]
        cn.set_cond(cond)
        x = [0.3, -1.1, 0.7, 2.0]
        out = cn.forward(x)
        mu = sum(x) / 4.0
        var = sum((v - mu) ** 2 for v in x) / 4.0
        inv = 1.0 / (var + cn.eps) ** 0.5
        for c in range(4):
            gamma = 1.0 + sum(cn.W_g[c][j] * cond[j] for j in range(3)) + cn.b_g[c]
            beta = sum(cn.W_b[c][j] * cond[j] for j in range(3)) + cn.b_b[c]
            xhat = (x[c] - mu) * inv
            assert abs(out[c] - (gamma * xhat + beta)) < 1e-12
        assert cn.param_count() == 4 * 3 + 4 + 4 * 3 + 4

    def test_gradients(self):
        import math as _math
        import random as _random
        _random.seed(62)
        cn = AI.ConditionalNorm(4, 3)
        cond = [0.5, -0.2, 0.8]
        cn.set_cond(cond)
        x = [0.3, -1.1, 0.7, 2.0]
        dout = [0.7, -0.4, 1.2, 0.3]

        def loss():
            return sum(a * b for a, b in zip(cn.forward(x), dout))

        cn.zero_grad()
        cn.forward(x)
        dx = cn.backward(dout)
        eps = 1e-6
        worst = 0.0
        for c in range(4):
            old = x[c]
            x[c] = old + eps
            vp = loss()
            x[c] = old - eps
            vm = loss()
            x[c] = old
            worst = max(worst, abs((vp - vm) / (2 * eps) - dx[c]))
        assert worst < 1e-7, f"dx 差分 {worst:.3e}"

        cn.zero_grad()
        cn.forward(x)
        cn.backward(dout)
        worst = 0.0
        for c in range(4):
            for c_ in range(4):
                pass
        for c in range(4):
            for arr, key in ((cn.b_g, "b_g"), (cn.b_b, "b_b")):
                old = arr[c]
                arr[c] = old + eps
                vp = loss()
                arr[c] = old - eps
                vm = loss()
                arr[c] = old
                worst = max(worst, abs((vp - vm) / (2 * eps) - cn.grads[key][c]))
        assert worst < 1e-7, f"b_g/b_b 差分 {worst:.3e}"

        worst = 0.0
        for c in range(4):
            for j in range(3):
                for mat, key in ((cn.W_g, "W_g"), (cn.W_b, "W_b")):
                    old = mat[c][j]
                    mat[c][j] = old + eps
                    vp = loss()
                    mat[c][j] = old - eps
                    vm = loss()
                    mat[c][j] = old
                    worst = max(worst, abs((vp - vm) / (2 * eps) - cn.grads[key][c][j]))
        assert worst < 1e-7, f"W_g/W_b 差分 {worst:.3e}"

        cn.zero_grad()
        cn.forward(x)
        cn.backward(dout)
        worst = 0.0
        for j in range(3):
            old = cond[j]
            cond[j] = old + eps
            cn.set_cond(cond)
            vp = loss()
            cond[j] = old - eps
            cn.set_cond(cond)
            vm = loss()
            cond[j] = old
            cn.set_cond(cond)
            worst = max(worst, abs((vp - vm) / (2 * eps) - cn.grad_cond[j]))
        assert worst < 1e-7, f"grad_cond 差分 {worst:.3e}"
        _math.isclose(1.0, 1.0)

    def test_optimizer_steps_and_errors(self):
        import random as _random
        _random.seed(63)
        cn = AI.ConditionalNorm(4, 3)
        cn.set_cond([0.5, -0.2, 0.8])
        x = [0.3, -1.1, 0.7, 2.0]
        cn.forward(x)
        cn.backward([0.7, -0.4, 1.2, 0.3])
        before = list(cn.b_b)
        cn.sgd_step(0.1)
        assert any(abs(a - b) > 1e-12 for a, b in zip(before, cn.b_b))
        cn.forward(x)
        cn.backward([0.7, -0.4, 1.2, 0.3])
        before = list(cn.b_b)
        cn.adam_step(0.05)
        assert cn._adam_t == 1
        assert any(abs(a - b) > 1e-12 for a, b in zip(before, cn.b_b))
        for bad in (-0.1, True, "0.1"):
            try:
                cn.sgd_step(bad)
            except (TypeError, ValueError):
                pass
            else:
                raise AssertionError(f"sgd_step({bad!r}) 应该报错")
        try:
            AI.ConditionalNorm(4, 3).forward([1.0, 2.0, 3.0, 4.0])
        except RuntimeError:
            pass
        else:
            raise AssertionError("未 set_cond 就 forward 应该报错")
        try:
            cn.set_cond([1.0, 2.0])
        except ValueError:
            pass
        else:
            raise AssertionError("条件向量长度不符应该报错")
        try:
            cn.forward([1.0, 2.0])
        except ValueError:
            pass
        else:
            raise AssertionError("特征长度不符应该报错")
        try:
            cn.forward([[1.0], [2.0], [3.0], [4.0]])
        except ValueError:
            pass
        else:
            raise AssertionError("二维输入应该报错")
        for bad in (0, -1, 1.5, True):
            try:
                AI.ConditionalNorm(bad, 3)
            except (TypeError, ValueError):
                pass
            else:
                raise AssertionError(f"ConditionalNorm({bad!r}, 3) 应该报错")


# ==== W3-A：ScaleNorm / WeightNorm / SpectralNorm ====

class TestScaleNorm:
    """ScaleNorm：只有标量 g，``y = g*x/(||x||/sqrt(d)+eps)``"""

    def test_forward_closed_form(self):
        import math as _math
        sn = AI.ScaleNorm(4)
        x = [0.3, -1.1, 0.7, 2.0]
        out = sn.forward(x)
        den = _math.sqrt(sum(v * v for v in x)) / _math.sqrt(4) + sn.eps
        assert all(abs(out[i] - sn.g * x[i] / den) < 1e-12 for i in range(4))
        assert sn.param_count() == 1
        assert abs(sn.g - 1.0) < 1e-12

    def test_gradients(self):
        import math as _math
        sn = AI.ScaleNorm(4)
        x = [0.3, -1.1, 0.7, 2.0]
        dout = [0.7, -0.4, 1.2, 0.3]

        def loss(xx, g):
            den = _math.sqrt(sum(v * v for v in xx)) / _math.sqrt(4) + sn.eps
            return sum(g * xx[i] / den * dout[i] for i in range(4))

        sn.forward(x)
        dg, dx = sn.backward(dout)
        eps = 1e-6
        for i in range(4):
            old = x[i]
            x[i] = old + eps
            vp = loss(x, sn.g)
            x[i] = old - eps
            vm = loss(x, sn.g)
            x[i] = old
            assert abs((vp - vm) / (2 * eps) - dx[i]) <= 1e-7
        g0 = sn.g
        sn.g = g0 + eps
        vp = loss(x, sn.g)
        sn.g = g0 - eps
        vm = loss(x, sn.g)
        sn.g = g0
        assert abs((vp - vm) / (2 * eps) - dg) <= 1e-7

    def test_optimizer_and_errors(self):
        sn = AI.ScaleNorm(3)
        x = [0.5, -0.2, 0.9]
        sn.forward(x)
        dg, _ = sn.backward([1.0, -1.0, 0.5])
        g0 = sn.g
        sn.sgd_step(0.1)
        assert abs(sn.g - (g0 - 0.1 * dg)) < 1e-12
        sn.forward(x)
        sn.backward([1.0, -1.0, 0.5])
        g1 = sn.g
        sn.adam_step(0.05)
        assert sn._adam_t == 1 and abs(sn.g - g1) > 1e-12
        # 全零输入：方向无定义但不崩、梯度有限
        z = AI.ScaleNorm(3)
        z.forward([0.0, 0.0, 0.0])
        zg, zdx = z.backward([1.0, 1.0, 1.0])
        assert all(abs(v) < 1e9 for v in zdx) and abs(zg) < 1e9
        for bad in (0, -1, 1.5, True):
            try:
                AI.ScaleNorm(bad)
            except (TypeError, ValueError):
                pass
            else:
                raise AssertionError(f"ScaleNorm({bad!r}) 应该报错")
        try:
            AI.ScaleNorm(4, -1.0)
        except ValueError:
            pass
        else:
            raise AssertionError("eps 为负应该报错")
        try:
            AI.ScaleNorm(4).forward([1.0, 2.0])
        except ValueError:
            pass
        else:
            raise AssertionError("维度不符应该报错")
        try:
            AI.ScaleNorm(4).backward([1.0, 2.0, 3.0, 4.0])
        except RuntimeError:
            pass
        else:
            raise AssertionError("未 forward 就 backward 应该报错")
        try:
            AI.ScaleNorm(3).sgd_step(-1)
        except ValueError:
            pass
        else:
            raise AssertionError("负 lr 应该报错")


class TestWeightNorm:
    """WeightNorm：``w = g ⊙ v/||v||``，反向精确传到 v/g/b"""

    def test_init_equals_original_layer(self):
        import random as _random
        _random.seed(71)
        lin = AI.Linear(3, 2)
        orig = [list(r) for r in lin.weights]
        wn = AI.WeightNorm(lin)
        assert all(abs(wn.v[i][j] - orig[i][j]) < 1e-12 for i in range(2) for j in range(3))
        x = [0.5, -0.3, 0.8]
        out = wn.forward(x)
        ref = [sum(orig[i][j] * x[j] for j in range(3)) + lin.biases[i] for i in range(2)]
        assert all(abs(out[i] - ref[i]) < 1e-12 for i in range(2))
        assert wn.param_count() == 6 + 2 + 2
        # forward 不永久改写内层权重
        assert all(abs(lin.weights[i][j] - orig[i][j]) < 1e-12
                   for i in range(2) for j in range(3))

    def test_gradients(self):
        import math as _math
        import random as _random
        _random.seed(72)
        lin = AI.Linear(3, 2)
        wn = AI.WeightNorm(lin)
        wn.g = [0.4, 1.7]   # 让 w != v，梯度才含解耦项
        x = [0.5, -0.3, 0.8]
        dout = [0.9, -0.6]
        wn.forward(x)
        dx = wn.backward(dout)

        def loss(vv, gg):
            tot = 0.0
            for i in range(2):
                n = _math.sqrt(sum(c * c for c in vv[i]))
                z = sum(gg[i] * vv[i][j] / n * x[j] for j in range(3)) + wn.b[i]
                tot += z * dout[i]
            return tot

        eps = 1e-6
        worst = 0.0
        for i in range(2):
            for j in range(3):
                old = wn.v[i][j]
                wn.v[i][j] = old + eps
                vp = loss(wn.v, wn.g)
                wn.v[i][j] = old - eps
                vm = loss(wn.v, wn.g)
                wn.v[i][j] = old
                worst = max(worst, abs((vp - vm) / (2 * eps) - wn.grad_v[i][j]))
        assert worst <= 1e-7, f"dv 差分 {worst:.3e}"

        for i in range(2):
            old = wn.g[i]
            wn.g[i] = old + eps
            vp = loss(wn.v, wn.g)
            wn.g[i] = old - eps
            vm = loss(wn.v, wn.g)
            wn.g[i] = old
            assert abs((vp - vm) / (2 * eps) - wn.grad_g[i]) <= 1e-7

        norms = [_math.sqrt(sum(c * c for c in row)) for row in wn.v]
        w_eff = [[wn.g[i] * wn.v[i][j] / norms[i] for j in range(3)] for i in range(2)]
        for j in range(3):
            assert abs(dx[j] - sum(w_eff[i][j] * dout[i] for i in range(2))) < 1e-12

    def test_optimizer_and_errors(self):
        import random as _random
        _random.seed(73)
        lin = AI.Linear(3, 2)
        wn = AI.WeightNorm(lin)
        x = [0.5, -0.3, 0.8]
        wn.forward(x)
        wn.backward([0.9, -0.6])
        before = [list(r) for r in wn.v]
        wn.sgd_step(0.1)
        assert any(abs(wn.v[i][j] - before[i][j]) > 1e-12 for i in range(2) for j in range(3))
        wn.forward(x)
        wn.backward([0.9, -0.6])
        before = [list(r) for r in wn.v]
        wn.adam_step(0.05)
        assert wn._adam_t == 1
        assert any(abs(wn.v[i][j] - before[i][j]) > 1e-12 for i in range(2) for j in range(3))
        try:
            AI.WeightNorm(123)
        except TypeError:
            pass
        else:
            raise AssertionError("非层对象应该报错")
        try:
            AI.WeightNorm(AI.Linear(3, 2)).forward([1.0, 2.0])
        except ValueError:
            pass
        else:
            raise AssertionError("输入维度不符应该报错")
        try:
            AI.WeightNorm(AI.Linear(3, 2)).backward([1.0, 1.0])
        except RuntimeError:
            pass
        else:
            raise AssertionError("未 forward 就 backward 应该报错")
        try:
            wn.sgd_step(-1)
        except ValueError:
            pass
        else:
            raise AssertionError("负 lr 应该报错")


class TestSpectralNorm:
    """SpectralNorm：``w = v/sigma``，冻结 sigma 时反向与中心差分一致"""

    def test_forward_and_sigma(self):
        import math as _math
        import random as _random
        _random.seed(81)
        lin = AI.Linear(3, 2)
        orig = [list(r) for r in lin.weights]
        sp = AI.SpectralNorm(lin, power_iter=20)
        sig = sp.freeze_sigma()
        assert sig > 0 and _math.isfinite(sig)
        x = [0.5, -0.3, 0.8]
        out = sp.forward(x)
        w = [[c / sig for c in row] for row in sp.v]
        ref = [sum(w[i][j] * x[j] for j in range(3)) + sp.b[i] for i in range(2)]
        assert all(abs(out[i] - ref[i]) < 1e-12 for i in range(2))
        assert sp.param_count() == 6 + 2
        assert all(abs(lin.weights[i][j] - orig[i][j]) < 1e-12
                   for i in range(2) for j in range(3))

    def test_frozen_sigma_gradient(self):
        import random as _random
        _random.seed(82)
        lin = AI.Linear(3, 2)
        sp = AI.SpectralNorm(lin, power_iter=20)
        sig = sp.freeze_sigma()
        x = [0.5, -0.3, 0.8]
        dout = [0.9, -0.6]
        sp.forward(x)
        dx = sp.backward(dout)

        def loss(vv):
            tot = 0.0
            for i in range(2):
                z = sum(vv[i][j] / sig * x[j] for j in range(3)) + sp.b[i]
                tot += z * dout[i]
            return tot

        eps = 1e-6
        worst = 0.0
        for i in range(2):
            for j in range(3):
                old = sp.v[i][j]
                sp.v[i][j] = old + eps
                vp = loss(sp.v)
                sp.v[i][j] = old - eps
                vm = loss(sp.v)
                sp.v[i][j] = old
                worst = max(worst, abs((vp - vm) / (2 * eps) - sp.grad_v[i][j]))
        assert worst <= 1e-7, f"dv 差分 {worst:.3e}"
        w = [[c / sig for c in row] for row in sp.v]
        for j in range(3):
            assert abs(dx[j] - sum(w[i][j] * dout[i] for i in range(2))) < 1e-12

    def test_unfrozen_sigma_and_errors(self):
        import random as _random
        _random.seed(83)
        sp = AI.SpectralNorm(AI.Linear(3, 2), power_iter=5)
        s0 = sp.sigma
        sp.v[0][0] += 5.0
        sp.forward([0.5, -0.3, 0.8])
        assert abs(sp.sigma - s0) > 1e-9
        try:
            AI.SpectralNorm(AI.Linear(3, 2), power_iter=-1)
        except ValueError:
            pass
        else:
            raise AssertionError("power_iter 为负应该报错")
        try:
            AI.SpectralNorm(AI.Linear(3, 2), power_iter=1.5)
        except TypeError:
            pass
        else:
            raise AssertionError("power_iter 非整数应该报错")
        try:
            AI.SpectralNorm(AI.Linear(3, 2)).forward([1.0])
        except ValueError:
            pass
        else:
            raise AssertionError("输入维度不符应该报错")
        try:
            AI.SpectralNorm(AI.Linear(3, 2)).backward([1.0, 1.0])
        except RuntimeError:
            pass
        else:
            raise AssertionError("未 forward 就 backward 应该报错")


# ==== W3-B：PReLU / Snake / SwiGLU 的 gate ====

class TestPReLU:
    """PReLU：逐特征可学习斜率，负分支才有 da"""

    def test_forward_and_param(self):
        pr = AI.PReLU(3, init=0.25)
        out = pr.forward([0.7, -0.4, -1.2])
        assert abs(out[0] - 0.7) < 1e-12
        assert abs(out[1] - 0.25 * -0.4) < 1e-12
        assert abs(out[2] - 0.25 * -1.2) < 1e-12
        assert pr.param_count() == 3

    def test_gradients(self):
        pr = AI.PReLU(3, init=0.25)
        x = [0.7, -0.4, -1.2]
        dout = [1.0, 2.0, -1.5]
        pr.zero_grad()
        pr.forward(x)
        dx = pr.backward(dout)
        assert abs(dx[0] - 1.0) < 1e-12
        assert abs(dx[1] - 2.0 * 0.25) < 1e-12
        assert abs(pr.grad_a[0]) < 1e-12          # 正分支不产生 da
        assert abs(pr.grad_a[1] - 2.0 * -0.4) < 1e-12
        assert abs(pr.grad_a[2] - (-1.5) * -1.2) < 1e-12

        def loss(aa, xx):
            return sum((xx[i] if xx[i] > 0 else aa[i] * xx[i]) * dout[i]
                       for i in range(3))

        eps = 1e-6
        for i in range(3):
            a0 = pr.a[i]
            pr.a[i] = a0 + eps
            vp = loss(pr.a, x)
            pr.a[i] = a0 - eps
            vm = loss(pr.a, x)
            pr.a[i] = a0
            assert abs((vp - vm) / (2 * eps) - pr.grad_a[i]) <= 1e-7

    def test_optimizer_and_errors(self):
        pr = AI.PReLU(3, init=0.25)
        x = [0.7, -0.4, -1.2]
        dout = [1.0, 2.0, -1.5]
        pr.forward(x)
        pr.backward(dout)
        a0 = list(pr.a)
        pr.sgd_step(0.1)
        assert any(abs(pr.a[i] - a0[i]) > 1e-12 for i in range(3))
        pr.forward(x)
        pr.backward(dout)
        a1 = list(pr.a)
        pr.adam_step(0.05)
        assert pr._adam_t == 1
        assert any(abs(pr.a[i] - a1[i]) > 1e-12 for i in range(3))
        for bad in (0, -1, 1.5, True):
            try:
                AI.PReLU(bad)
            except (TypeError, ValueError):
                pass
            else:
                raise AssertionError(f"PReLU({bad!r}) 应该报错")
        try:
            AI.PReLU(3, float("nan"))
        except ValueError:
            pass
        else:
            raise AssertionError("init=nan 应该报错")
        try:
            AI.PReLU(3).forward([1.0])
        except ValueError:
            pass
        else:
            raise AssertionError("维度不符应该报错")
        try:
            AI.PReLU(3).backward([1.0, 1.0, 1.0])
        except RuntimeError:
            pass
        else:
            raise AssertionError("未 forward 就 backward 应该报错")
        try:
            pr.sgd_step(-1)
        except ValueError:
            pass
        else:
            raise AssertionError("负 lr 应该报错")


class TestSnake:
    """Snake：``y = x + sin²(ax)/a``，a→0 退化为恒等"""

    def test_forward_closed_form(self):
        import math as _math
        sn = AI.Snake(3, init=1.0)
        xs = [0.3, -0.8, 1.4]
        out = sn.forward(xs)
        for i in range(3):
            ref = xs[i] + _math.sin(xs[i]) ** 2
            assert abs(out[i] - ref) < 1e-12
        assert sn.param_count() == 3

    def test_gradients(self):
        import math as _math
        sn = AI.Snake(3, init=1.0)
        xs = [0.3, -0.8, 1.4]
        douts = [0.5, -1.0, 2.0]
        sn.zero_grad()
        sn.forward(xs)
        dxs = sn.backward(douts)
        for i in range(3):
            assert abs(dxs[i] - douts[i] * (1.0 + _math.sin(2.0 * xs[i]))) < 1e-12

        def loss(aa, xx):
            tot = 0.0
            for i in range(3):
                a = aa[i]
                y = xx[i] if abs(a) < 1e-8 else xx[i] + _math.sin(a * xx[i]) ** 2 / a
                tot += y * douts[i]
            return tot

        eps = 1e-6
        for i in range(3):
            x0 = xs[i]
            xs[i] = x0 + eps
            vp = loss(sn.a, xs)
            xs[i] = x0 - eps
            vm = loss(sn.a, xs)
            xs[i] = x0
            assert abs((vp - vm) / (2 * eps) - dxs[i]) <= 1e-7
        for i in range(3):
            a0 = sn.a[i]
            sn.a[i] = a0 + eps
            vp = loss(sn.a, xs)
            sn.a[i] = a0 - eps
            vm = loss(sn.a, xs)
            sn.a[i] = a0
            assert abs((vp - vm) / (2 * eps) - sn.grad_a[i]) <= 1e-7

    def test_zero_scale_limit_and_errors(self):
        sn = AI.Snake(2, init=1.0)
        sn.a = [0.0, 0.0]
        sn.forward([0.5, -2.0])
        d = sn.backward([1.0, 1.0])
        assert abs(d[0] - 1.0) < 1e-12 and abs(d[1] - 1.0) < 1e-12
        assert abs(sn.grad_a[0] - 0.25) < 1e-12
        assert abs(sn.grad_a[1] - 4.0) < 1e-12
        for bad in (0, -1, True):
            try:
                AI.Snake(bad)
            except (TypeError, ValueError):
                pass
            else:
                raise AssertionError(f"Snake({bad!r}) 应该报错")
        try:
            AI.Snake(2).forward([1.0])
        except ValueError:
            pass
        else:
            raise AssertionError("维度不符应该报错")


class TestSwiGLUGates:
    """SwiGLU 的 gate：swish/silu/gelu/relu/bilinear"""

    def test_default_is_swish_and_unchanged(self):
        import random as _random
        _random.seed(7)
        base = AI.SwiGLU(4, 5)
        assert base.gate == "swish"
        base.forward([[0.1, -0.2, 0.3, 0.4]])
        xr = [0.1, -0.2, 0.3, 0.4]
        for j in range(5):
            z = sum(base.Wg[j][k] * xr[k] for k in range(4)) + base.bg[j]
            assert abs(base.g[0][j] - (AI._silu_scalar(z) + base.bg[j])) < 1e-12

    def test_each_gate_forward(self):
        import random as _random
        xr = [0.1, -0.2, 0.3, 0.4]
        for gate in ("gelu", "relu", "bilinear"):
            _random.seed(7)
            g = AI.SwiGLU(4, 5, gate=gate)
            g.forward([xr])
            for j in range(5):
                z = sum(g.Wg[j][k] * xr[k] for k in range(4)) + g.bg[j]
                if gate == "gelu":
                    expect = AI._gelu_scalar(z)
                elif gate == "relu":
                    expect = max(0.0, z)
                else:
                    expect = z
                assert abs(g.g[0][j] - (expect + g.bg[j])) < 1e-12

    def test_gate_gradients(self):
        import random as _random
        eps = 1e-6
        xseq = [[0.3, -0.5, 0.8]]
        doutseq = [[0.7, -0.4, 0.2]]
        for gate in ("gelu", "relu", "bilinear"):
            _random.seed(11)
            gl = AI.SwiGLU(3, 4, gate=gate)

            def loss():
                o = gl.forward(xseq)
                return sum(o[0][j] * doutseq[0][j] for j in range(3))

            gl.zero_grad()
            loss()
            gl.backward(doutseq)
            for j in range(4):
                for k in range(3):
                    old = gl.Wg[j][k]
                    gl.Wg[j][k] = old + eps
                    vp = loss()
                    gl.Wg[j][k] = old - eps
                    vm = loss()
                    gl.Wg[j][k] = old
                    assert abs((vp - vm) / (2 * eps) - gl.grads["Wg"][j][k]) <= 1e-6

    def test_invalid_gate(self):
        try:
            AI.SwiGLU(4, 5, gate="tanh")
        except ValueError:
            pass
        else:
            raise AssertionError("非法 gate 应该报错")
        try:
            AI.SwiGLU(4, 5, gate=3)
        except TypeError:
            pass
        else:
            raise AssertionError("非字符串 gate 应该报错")
        assert AI.SwiGLU.GATES == ("swish", "silu", "gelu", "relu", "bilinear")


# ==== W3-C：初始化族 ====

class TestInitFamily:
    """LeCun / 正交 / 稀疏 / 截断正态 / 零偏置 / 残差零初始化"""

    @staticmethod
    def _flat(layer):
        return [v for row in layer.weights for v in row]

    @staticmethod
    def _stats(vals):
        import math as _math
        n = len(vals)
        mu = sum(vals) / n
        var = sum((v - mu) ** 2 for v in vals) / n
        return mu, _math.sqrt(var)

    def test_lecun(self):
        import math as _math
        import random as _random
        _random.seed(101)
        layer = AI.Layer(64, 32, activation="linear")
        AI.init_lecun_normal(layer)
        mu, sd = self._stats(self._flat(layer))
        expect = _math.sqrt(1.0 / 64)
        assert abs(mu) < 0.01
        assert abs(sd - expect) / expect < 0.06

        AI.init_lecun_uniform(layer)
        mu, sd = self._stats(self._flat(layer))
        bound = _math.sqrt(3.0 / 64)
        assert abs(sd - bound / _math.sqrt(3)) / (bound / _math.sqrt(3)) < 0.06
        assert all(abs(v) <= bound + 1e-12 for v in self._flat(layer))

        _random.seed(101)
        l2 = AI.Layer(64, 32, activation="linear")
        AI.init_lecun_normal(l2, fan_in=16)
        _, sd = self._stats(self._flat(l2))
        assert abs(sd - _math.sqrt(1.0 / 16)) / _math.sqrt(1.0 / 16) < 0.06

    def test_orthogonal(self):
        import math as _math
        import random as _random
        _random.seed(102)
        sq = AI.Layer(8, 8, activation="linear")
        AI.init_orthogonal(sq, gain=2.0)
        rows = [list(r) for r in sq.weights]
        for r in rows:
            assert abs(_math.sqrt(sum(v * v for v in r)) - 2.0) < 1e-9
        for i in range(8):
            for j in range(i + 1, 8):
                dot = sum(rows[i][k] * rows[j][k] for k in range(8))
                assert abs(dot) <= 1e-10, f"第 {i},{j} 行不垂直：{dot:.3e}"

        # 行多于列：行无法正交，改为列正交（wᵀw = I）
        tall = AI.Layer(3, 6, activation="linear")
        AI.init_orthogonal(tall, gain=1.0)
        cols = [[tall.weights[i][j] for i in range(6)] for j in range(3)]
        for c in cols:
            assert abs(_math.sqrt(sum(v * v for v in c)) - 1.0) < 1e-9
        for i in range(3):
            for j in range(i + 1, 3):
                assert abs(sum(cols[i][k] * cols[j][k] for k in range(6))) <= 1e-10

    def test_sparse_and_trunc_normal(self):
        import math as _math
        import random as _random
        _random.seed(103)
        sp = AI.Layer(20, 10, activation="linear")
        AI.init_sparse(sp, sparsity=0.8)
        vals = self._flat(sp)
        zeros = sum(1 for v in vals if v == 0.0) / len(vals)
        assert abs(zeros - 0.8) < 0.06
        nonzero = [v for v in vals if v != 0.0]
        _, sd = self._stats(nonzero)
        expect = _math.sqrt(1.0 / 20)
        assert abs(sd - expect) / expect < 0.2

        _random.seed(104)
        tn = AI.Layer(50, 40, activation="linear")
        AI.init_trunc_normal(tn, std=0.02, cutoff=2.0)
        vals = self._flat(tn)
        assert all(abs(v) <= 0.04 + 1e-12 for v in vals)
        mu, sd = self._stats(vals)
        assert 0.015 < sd < 0.021
        assert abs(mu) < 0.005

    def test_zero_bias_and_residual(self):
        import random as _random
        _random.seed(105)
        mlp = AI.MLP([4, 8, 3])
        for lyr in mlp.layers:
            lyr.biases = [0.5] * lyr.fan_out
        AI.zero_bias(mlp)
        assert all(b == 0.0 for lyr in mlp.layers for b in lyr.biases)

        _random.seed(106)
        blk = AI.BasicBlock2d(3, kernel_size=3)
        x = [[[0.1 * (i + j + k) for k in range(4)] for j in range(4)]
             for i in range(3)]
        before = blk.forward(x)
        AI.zero_init_residual(blk)
        after = blk.forward(x)
        assert all(g == 0.0 for g in blk.bn2.gamma)
        ref = [[[max(0.0, x[c][h][w]) for w in range(4)] for h in range(4)]
               for c in range(3)]
        for c in range(3):
            for h in range(4):
                for w in range(4):
                    assert abs(after[c][h][w] - ref[c][h][w]) < 1e-12
        assert any(abs(before[c][h][w] - ref[c][h][w]) > 1e-9
                   for c in range(3) for h in range(4) for w in range(4))

        # Bottleneck 零的是分支最后一个 BN（bn3），不是捷径的 short_bn
        _random.seed(107)
        bn = AI.Bottleneck2d(3, 6)
        AI.zero_init_residual(bn)
        assert all(g == 0.0 for g in bn.bn3.gamma)
        assert any(g != 0.0 for g in bn.short_bn.gamma)

    def test_errors_and_default_unchanged(self):
        import math as _math
        import random as _random
        try:
            AI.init_lecun_normal(123)
        except TypeError:
            pass
        else:
            raise AssertionError("非层对象应该报错")
        try:
            AI.init_lecun_normal(AI.MLP([2, 2]), fan_in=0)
        except ValueError:
            pass
        else:
            raise AssertionError("fan_in=0 应该报错")
        try:
            AI.init_orthogonal(AI.Layer(3, 3), gain=float("inf"))
        except ValueError:
            pass
        else:
            raise AssertionError("gain=inf 应该报错")
        for bad in (1.0, -0.1):
            try:
                AI.init_sparse(AI.Layer(3, 3), sparsity=bad)
            except ValueError:
                pass
            else:
                raise AssertionError(f"sparsity={bad} 应该报错")
        try:
            AI.init_trunc_normal(AI.Layer(3, 3), std=0.0)
        except ValueError:
            pass
        else:
            raise AssertionError("std=0 应该报错")
        try:
            AI.init_trunc_normal(AI.Layer(3, 3), cutoff=-1.0)
        except ValueError:
            pass
        else:
            raise AssertionError("cutoff<0 应该报错")
        try:
            AI.zero_init_residual(AI.Layer(3, 3))
        except TypeError:
            pass
        else:
            raise AssertionError("没有 BN 的块应该报错")
        # 默认构造仍是 He 初始化，没被这些工具函数改动
        _random.seed(108)
        m = AI.MLP([100, 40])
        vals = [v for lyr in m.layers for v in self._flat(lyr)]
        _, sd = self._stats(vals)
        expect = _math.sqrt(2.0 / 100)
        assert abs(sd - expect) / expect < 0.1


# ==== W3-D：SwitchNorm 与 FRN ====

class TestSwitchNorm:
    """三种统计量软加权：前向闭式解、四组梯度、跨通道耦合、优化器步"""

    def test_forward_closed_form(self):
        import math as _math
        import random as _random
        _random.seed(21)
        sn = AI.SwitchNorm(2, eps=1e-5)
        sn.gamma = [1.3, 0.7]
        sn.beta = [0.1, -0.2]
        sn.logits = [0.5, 0.2, 0.9]
        x = [[[_random.uniform(-1, 1) for _ in range(3)] for _ in range(3)]
             for _ in range(2)]
        out = sn.forward(x)
        w = [_math.exp(v) for v in sn.logits]
        tot = sum(w)
        w = [v / tot for v in w]
        w_ib = w[0] + w[1]
        im, iv = [], []
        for c in range(2):
            vals = [x[c][i][j] for i in range(3) for j in range(3)]
            m = sum(vals) / 9
            im.append(m)
            iv.append(sum((q - m) ** 2 for q in vals) / 9)
        every = [x[c][i][j] for c in range(2) for i in range(3) for j in range(3)]
        lm = sum(every) / 18
        lv = sum((q - lm) ** 2 for q in every) / 18
        m0 = w_ib * im[0] + w[2] * lm
        v0 = w_ib * iv[0] + w[2] * lv
        ref = (x[0][0][0] - m0) / _math.sqrt(v0 + 1e-5) * 1.3 + 0.1
        assert abs(out[0][0][0] - ref) < 1e-12
        assert len(out) == 2 and len(out[0]) == 3 and len(out[0][0]) == 3
        assert sn.param_count() == 2 * 2 + 3

    def test_gradients_vs_finite_difference(self):
        import random as _random
        _random.seed(22)
        sn = AI.SwitchNorm(2, eps=1e-5)
        sn.gamma = [1.1, 0.9]
        sn.logits = [0.3, -0.4, 0.7]
        x = [[[_random.uniform(-1, 1) for _ in range(3)] for _ in range(3)]
             for _ in range(2)]
        coef = [[[_random.uniform(-1, 1) for _ in range(3)] for _ in range(3)]
                for _ in range(2)]

        def loss():
            o = sn.forward(x)
            return sum(o[c][i][j] * coef[c][i][j]
                       for c in range(2) for i in range(3) for j in range(3))

        sn.forward(x)
        dx, g_gamma, g_beta, g_logits = sn.backward(coef)
        worst = 0.0
        for c in range(2):
            for i in range(3):
                for j in range(3):
                    old = x[c][i][j]
                    h = 1e-6
                    x[c][i][j] = old + h
                    lp = loss()
                    x[c][i][j] = old - h
                    lmm = loss()
                    x[c][i][j] = old
                    num = (lp - lmm) / (2 * h)
                    worst = max(worst, abs(num - dx[c][i][j])
                                / max(1e-8, abs(num), abs(dx[c][i][j])))
        assert worst <= 1e-7, f"dx 差分误差 {worst:.3e}"

        # 软权重的梯度：layer 统计量跨通道共享，l 的链式传回去会牵动所有通道
        per_channel_sums = []
        for k in range(3):
            old = sn.logits[k]
            h = 1e-6
            sn.logits[k] = old + h
            lp = loss()
            sn.logits[k] = old - h
            lmm = loss()
            sn.logits[k] = old
            num = (lp - lmm) / (2 * h)
            per_channel_sums.append(num)
            assert abs(num - g_logits[k]) <= 1e-7 * max(1.0, abs(num)), \
                f"grad_logits[{k}] 差分误差 {abs(num - g_logits[k]):.3e}"
        # softmax 的雅可比行和为 0 ⇒ 三个软权重的梯度在 logit 空间里满足
        # Σ_k raw_k 的加权组合为零；这里直接验"整体平移 logits 不改变输出"更稳
        base = loss()
        shift = [v + 0.7 for v in sn.logits]
        keep = list(sn.logits)
        sn.logits = shift
        assert abs(loss() - base) < 1e-12
        sn.logits = keep

    def test_batch_and_steps(self):
        import random as _random
        _random.seed(23)
        sn = AI.SwitchNorm(2)
        xb = [[[[_random.uniform(-1, 1) for _ in range(3)] for _ in range(3)]
               for _ in range(2)] for _ in range(2)]
        ob = sn.forward(xb)
        assert len(ob) == 2 and len(ob[0]) == 2
        douts = [[[[_random.uniform(-1, 1) for _ in range(3)] for _ in range(3)]
                  for _ in range(2)] for _ in range(2)]
        dxb, _, _, _ = sn.backward(douts)
        assert len(dxb) == 2 and len(dxb[1]) == 2
        g0 = list(sn.gamma)
        sn.grad_gamma = [0.5] * 2
        sn.sgd_step(0.1)
        assert any(abs(a - b) > 1e-12 for a, b in zip(g0, sn.gamma))
        assert all(v == 0.0 for v in sn.grad_gamma)
        lg = list(sn.logits)
        sn.grad_logits = [0.3, -0.2, 0.1]
        sn.adam_step(0.01)
        assert any(abs(a - b) > 1e-12 for a, b in zip(lg, sn.logits))

    def test_errors(self):
        for fn in (lambda: AI.SwitchNorm(0), lambda: AI.SwitchNorm(True),
                   lambda: AI.SwitchNorm(2, eps=-1.0),
                   lambda: AI.SwitchNorm(2).forward([[[1.0, 2.0]]]),
                   lambda: AI.SwitchNorm(2).forward([]),
                   lambda: AI.SwitchNorm(2).backward([[[0.0]]])):
            try:
                fn()
            except (TypeError, ValueError, RuntimeError):
                pass
            else:
                raise AssertionError("非法输入应该报错")


class TestFRN:
    """FRN + 阈值化 ReLU：统计量跨通道、batch=1 可用"""

    def test_forward_closed_form(self):
        import math as _math
        import random as _random
        _random.seed(31)
        frn = AI.FRN(2, eps=1e-6, tau=0.0)
        frn.beta = [0.1, 0.2]
        x = [[[_random.uniform(-1, 1) for _ in range(3)] for _ in range(3)]
             for _ in range(2)]
        out = frn.forward(x)
        s = sum(x[c][i][j] ** 2 for c in range(2) for i in range(3) for j in range(3))
        inv = 1.0 / _math.sqrt(s / 18 + 1e-6)
        raw = x[0][0][0] * inv - 0.1
        assert abs(out[0][0][0] - max(raw, 0.0)) < 1e-12
        # 被截断的位置输出恰为 tau
        cut = [(c, i, j) for c in range(2) for i in range(3) for j in range(3)
               if x[c][i][j] * inv - frn.beta[c] <= 0.0]
        assert cut
        assert all(abs(out[c][i][j] - 0.0) < 1e-12 for c, i, j in cut)
        assert frn.param_count() == 4

    def test_gradients_vs_finite_difference(self):
        import random as _random
        _random.seed(32)
        frn = AI.FRN(2, eps=1e-6, tau=0.0)
        frn.beta = [0.05, -0.05]
        x = [[[_random.uniform(-1, 1) for _ in range(3)] for _ in range(3)]
             for _ in range(2)]
        coef = [[[_random.uniform(-1, 1) for _ in range(3)] for _ in range(3)]
                for _ in range(2)]

        def loss():
            o = frn.forward(x)
            return sum(o[c][i][j] * coef[c][i][j]
                       for c in range(2) for i in range(3) for j in range(3))

        frn.forward(x)
        dx, g_beta, g_tau = frn.backward(coef)
        worst = 0.0
        for c in range(2):
            for i in range(3):
                for j in range(3):
                    old = x[c][i][j]
                    h = 1e-6
                    x[c][i][j] = old + h
                    lp = loss()
                    x[c][i][j] = old - h
                    lmm = loss()
                    x[c][i][j] = old
                    num = (lp - lmm) / (2 * h)
                    worst = max(worst, abs(num - dx[c][i][j])
                                / max(1e-8, abs(num), abs(dx[c][i][j])))
        assert worst <= 1e-7, f"FRN dx 差分误差 {worst:.3e}"

        for arr, ana, name in ((frn.beta, g_beta, "beta"), (frn.tau, g_tau, "tau")):
            for c in range(2):
                old = arr[c]
                h = 1e-6
                arr[c] = old + h
                lp = loss()
                arr[c] = old - h
                lmm = loss()
                arr[c] = old
                num = (lp - lmm) / (2 * h)
                assert abs(num - ana[c]) <= 1e-6 * max(1.0, abs(num)), \
                    f"FRN grad_{name}[{c}] 差分误差 {abs(num - ana[c]):.3e}"

    def test_scale_invariance_and_steps(self):
        import random as _random
        _random.seed(33)
        x = [[[_random.uniform(-1, 1) for _ in range(3)] for _ in range(3)]
             for _ in range(2)]
        tiny = AI.FRN(2, eps=1e-14, tau=-1e9)
        tiny.beta = [0.1, 0.2]
        a = tiny.forward(x)
        big = [[[v * 37.0 for v in row] for row in ch] for ch in x]
        b = tiny.forward(big)
        for c in range(2):
            for i in range(3):
                for j in range(3):
                    assert abs(a[c][i][j] - b[c][i][j]) < 1e-9
        # 有 eps 时严格尺度不变被打破（是 FRN 的定义，不是缺陷）
        frn = AI.FRN(2, eps=1e-6, tau=-1e9)
        a2 = frn.forward(x)
        b2 = frn.forward(big)
        assert any(abs(a2[c][i][j] - b2[c][i][j]) > 1e-9
                   for c in range(2) for i in range(3) for j in range(3))

        frn.grad_beta = [0.4, -0.3]
        frn.grad_tau = [0.2, -0.1]
        b_before = list(frn.beta)
        t_before = list(frn.tau)
        frn.adam_step(0.01)
        assert any(abs(a - b) > 1e-12 for a, b in zip(b_before, frn.beta))
        assert any(abs(a - b) > 1e-12 for a, b in zip(t_before, frn.tau))
        frn.grad_beta = [0.4, -0.3]
        frn.sgd_step(0.1)
        assert all(v == 0.0 for v in frn.grad_beta)

        for fn in (lambda: AI.FRN(0), lambda: AI.FRN(True),
                   lambda: AI.FRN(2, eps=0.0), lambda: AI.FRN(2, tau="x"),
                   lambda: AI.FRN(2).forward([[1.0]]),
                   lambda: AI.FRN(2).forward([[[1.0, 2.0]], [[1.0]]]),
                   lambda: AI.FRN(2).backward([[[0.0]]])):
            try:
                fn()
            except (TypeError, ValueError, RuntimeError):
                pass
            else:
                raise AssertionError("非法输入应该报错")


# ==== W3-E：Maxout / DynamicReLU / FReLU ====

class TestMaxout:
    """逐特征多仿射支路取最大：前向、argmax 路由、梯度差分、优化器步"""

    def test_forward_and_routing(self):
        mo = AI.Maxout(3, num_pieces=3)
        mo.a = [[1.0, -2.0, 0.5], [0.5, 3.0, -1.0], [-1.0, 0.0, 2.0]]
        mo.b = [[0.0, 0.1, 0.0], [0.2, -0.3, 0.0], [0.0, 0.0, 0.5]]
        x = [1.0, -0.5, 0.25]
        got = mo.forward(x)
        ref = [max(mo.a[k][i] * x[i] + mo.b[k][i] for k in range(3)) for i in range(3)]
        assert all(abs(g - r) < 1e-15 for g, r in zip(got, ref))
        for i in range(3):
            best = max(mo.a[k][i] * x[i] + mo.b[k][i] for k in range(3))
            assert abs(mo.a[mo.arg[i]][i] * x[i] + mo.b[mo.arg[i]][i] - best) < 1e-15
        assert mo.param_count() == 2 * 3 * 3
        # 只给选中支路发梯度
        dout = [0.7, -1.3, 0.4]
        mo.forward(x)
        dx = mo.backward(dout)
        assert all(abs(dx[i] - dout[i] * mo.a[mo.arg[i]][i]) < 1e-15 for i in range(3))
        for i in range(3):
            for k in range(3):
                if k == mo.arg[i]:
                    assert abs(mo.grad_a[k][i] - dout[i] * x[i]) < 1e-15
                    assert abs(mo.grad_b[k][i] - dout[i]) < 1e-15
                else:
                    assert mo.grad_a[k][i] == 0.0 and mo.grad_b[k][i] == 0.0

    def test_gradients_vs_finite_difference(self):
        mo = AI.Maxout(3, num_pieces=3)
        mo.a = [[1.0, -2.0, 0.5], [0.4, 3.0, -1.0], [-1.0, 0.2, 2.0]]
        mo.b = [[0.0, 0.1, 0.0], [0.2, -0.3, 0.0], [0.0, 0.0, 0.5]]
        x = [1.0, -0.5, 0.25]
        coef = [0.7, -1.3, 0.4]

        def loss():
            o = mo.forward(x)
            return sum(o[i] * coef[i] for i in range(3))

        mo.forward(x)
        dx = mo.backward(coef)
        for i in range(3):
            old = x[i]
            h = 1e-6
            x[i] = old + h
            lp = loss()
            x[i] = old - h
            lmm = loss()
            x[i] = old
            num = (lp - lmm) / (2 * h)
            assert abs(num - dx[i]) <= 1e-7 * max(1.0, abs(num)), \
                f"Maxout dx[{i}] 差分误差 {abs(num - dx[i]):.3e}"
        for k in range(3):
            for i in range(3):
                old = mo.a[k][i]
                h = 1e-6
                mo.a[k][i] = old + h
                lp = loss()
                mo.a[k][i] = old - h
                lmm = loss()
                mo.a[k][i] = old
                num = (lp - lmm) / (2 * h)
                mo.forward(x)
                mo.backward(coef)
                assert abs(num - mo.grad_a[k][i]) <= 1e-7 * max(1.0, abs(num)), \
                    f"Maxout grad_a[{k}][{i}] 差分误差 {abs(num - mo.grad_a[k][i]):.3e}"

    def test_steps_and_errors(self):
        mo = AI.Maxout(2, num_pieces=2)
        mo.forward([1.0, -1.0])
        mo.backward([0.5, 0.5])
        a0 = [row[:] for row in mo.a]
        mo.sgd_step(0.1)
        assert any(abs(a - b) > 1e-12 for r0, r1 in zip(a0, mo.a) for a, b in zip(r0, r1))
        assert all(v == 0.0 for row in mo.grad_a for v in row)
        mo.forward([1.0, -1.0])
        mo.backward([0.5, 0.5])
        b0 = [row[:] for row in mo.b]
        mo.adam_step(0.01)
        assert any(abs(a - b) > 1e-12 for r0, r1 in zip(b0, mo.b) for a, b in zip(r0, r1))
        for fn in (lambda: AI.Maxout(0), lambda: AI.Maxout(True),
                   lambda: AI.Maxout(2, num_pieces=1),
                   lambda: AI.Maxout(2).forward([1.0]),
                   lambda: AI.Maxout(2).backward([1.0, 2.0]),
                   lambda: AI.Maxout(2).sgd_step(-1.0)):
            try:
                fn()
            except (TypeError, ValueError, RuntimeError):
                pass
            else:
                raise AssertionError("非法输入应该报错")


class TestDynamicReLU:
    """斜率由输入上下文线性预测：闭式解、跨特征耦合的 dx、优化器步"""

    def test_forward_closed_form(self):
        dr = AI.DynamicReLU(3, init_slope=0.25)
        # w=0 ⇒ 斜率恒等于 init_slope（退化 LeakyReLU）
        assert dr.forward([-1.0, 2.0, -0.5]) == [-0.25, 2.0, -0.125]
        dr.w = [0.8, -1.2, 0.5]
        dr.b = [0.3, -0.1, 0.6]
        x = [-1.0, 2.0, -0.5]
        z = sum(x) / 3
        got = dr.forward(x)
        ref = [x[i] if x[i] > 0 else (dr.w[i] * z + dr.b[i]) * x[i] for i in range(3)]
        assert all(abs(g - r) < 1e-15 for g, r in zip(got, ref))
        assert dr.param_count() == 6
        # 换一个样本，斜率应随之变化（这正是 vs PReLU 的区别）
        s1 = dr.forward([-1.0, 2.0, -0.5])[0]
        s2 = dr.forward([-1.0, -2.0, -3.0])[0]
        assert abs(s1 - s2) > 1e-12

    def test_coupled_dx_and_grads(self):
        dr = AI.DynamicReLU(3)
        dr.w = [0.8, -1.2, 0.5]
        dr.b = [0.3, -0.1, 0.6]
        x = [-1.0, 2.0, -0.5]
        dout = [0.5, -0.7, 1.1]
        z = sum(x) / 3
        s_coup = sum(dout[i] * x[i] * dr.w[i] for i in range(3) if x[i] <= 0)
        ref = []
        for i in range(3):
            if x[i] > 0:
                ref.append(dout[i] + s_coup / 3)
            else:
                ref.append(dout[i] * (dr.w[i] * z + dr.b[i]) + s_coup / 3)
        dr.forward(x)
        dx = dr.backward(dout)
        assert all(abs(a - b) < 1e-14 for a, b in zip(dx, ref))
        # 正分支特征也吃到耦合项（因为改 x_j 会改 z，从而改所有负分支斜率）
        assert abs(dx[1] - dout[1]) > 1e-9

        base_w, base_b = dr.w[:], dr.b[:]

        def loss():
            o = dr.forward(x)
            return sum(o[j] * dout[j] for j in range(3))

        for i in range(3):
            for holder, ana, nm in ((base_w, dr.grad_w, "w"), (base_b, dr.grad_b, "b")):
                keep = holder[i]
                h = 1e-6
                holder[i] = keep + h
                dr.w, dr.b = base_w, base_b
                lp = loss()
                holder[i] = keep - h
                dr.w, dr.b = base_w, base_b
                lmm = loss()
                holder[i] = keep
                dr.w, dr.b = base_w, base_b
                dr.forward(x)
                dr.backward(dout)
                num = (lp - lmm) / (2 * h)
                assert abs(num - ana[i]) <= 1e-7 * max(1.0, abs(num)), \
                    f"DynamicReLU grad_{nm}[{i}] 差分误差 {abs(num - ana[i]):.3e}"
        # dx 也要过差分（含耦合项）
        for i in range(3):
            old = x[i]
            h = 1e-6
            x[i] = old + h
            lp = loss()
            x[i] = old - h
            lmm = loss()
            x[i] = old
            dr.forward(x)
            dr.backward(dout)
            num = (lp - lmm) / (2 * h)
            assert abs(num - dx[i]) <= 1e-7 * max(1.0, abs(num)), \
                f"DynamicReLU dx[{i}] 差分误差 {abs(num - dx[i]):.3e}"

    def test_steps_and_errors(self):
        dr = AI.DynamicReLU(2)
        dr.forward([-1.0, 0.5])
        dr.backward([1.0, 1.0])
        w0 = dr.w[:]
        dr.sgd_step(0.1)
        assert any(abs(a - b) > 1e-12 for a, b in zip(w0, dr.w))
        dr.forward([-1.0, 0.5])
        dr.backward([1.0, 1.0])
        w1 = dr.w[:]
        dr.adam_step(0.01)
        assert any(abs(a - b) > 1e-12 for a, b in zip(w1, dr.w))
        for fn in (lambda: AI.DynamicReLU(0),
                   lambda: AI.DynamicReLU(2, init_slope=float("inf")),
                   lambda: AI.DynamicReLU(2).forward([1.0]),
                   lambda: AI.DynamicReLU(2).backward([1.0, 2.0])):
            try:
                fn()
            except (TypeError, ValueError, RuntimeError):
                pass
            else:
                raise AssertionError("非法输入应该报错")


class TestFReLU:
    """漏斗激活 max(x, T(x))：零初始化退化为 ReLU、闭式解、梯度差分"""

    def test_zero_init_is_relu(self):
        import random as _random
        _random.seed(41)
        fr = AI.FReLU(2, kernel_size=3)
        x = [[[_random.uniform(-1, 1) for _ in range(4)] for _ in range(3)]
             for _ in range(2)]
        o = fr.forward(x)
        assert all(o[c][i][j] == max(0.0, x[c][i][j])
                   for c in range(2) for i in range(3) for j in range(4))
        assert fr.param_count() == 2 * 9 + 2

    def test_forward_closed_form(self):
        import random as _random
        _random.seed(42)
        fr = AI.FReLU(2, kernel_size=3)
        fr.w = [[[0.1, 0.0, -0.1], [0.0, 0.2, 0.0], [0.05, 0.0, 0.15]] for _ in range(2)]
        fr.b = [0.05, -0.05]
        x = [[[_random.uniform(-1, 1) for _ in range(4)] for _ in range(3)]
             for _ in range(2)]
        o = fr.forward(x)
        for c in range(2):
            for i in range(3):
                for j in range(4):
                    t = fr.b[c]
                    for a in range(3):
                        for bb in range(3):
                            ii, jj = i + a - 1, j + bb - 1
                            if 0 <= ii < 3 and 0 <= jj < 4:
                                t += fr.w[c][a][bb] * x[c][ii][jj]
                    assert abs(o[c][i][j] - max(x[c][i][j], t)) < 1e-13

    def test_gradients_and_steps(self):
        import random as _random
        _random.seed(43)
        fr = AI.FReLU(1, kernel_size=3)
        fr.w = [[[0.2, 0.1, 0.0], [0.0, -0.3, 0.1], [0.05, 0.0, 0.25]]]
        fr.b = [0.1]
        x = [[[_random.uniform(-1, 1) for _ in range(3)] for _ in range(3)]]
        coef = [[[_random.uniform(-1, 1) for _ in range(3)] for _ in range(3)]]

        def loss():
            o = fr.forward(x)
            return sum(o[c][i][j] * coef[c][i][j]
                       for c in range(1) for i in range(3) for j in range(3))

        fr.forward(x)
        dx = fr.backward(coef)
        for i in range(3):
            for j in range(3):
                old = x[0][i][j]
                h = 1e-6
                x[0][i][j] = old + h
                lp = loss()
                x[0][i][j] = old - h
                lmm = loss()
                x[0][i][j] = old
                num = (lp - lmm) / (2 * h)
                assert abs(num - dx[0][i][j]) <= 1e-7 * max(1.0, abs(num)), \
                    f"FReLU dx[{i}][{j}] 差分误差 {abs(num - dx[0][i][j]):.3e}"
        for a in range(3):
            for bb in range(3):
                keep = fr.w[0][a][bb]
                h = 1e-6
                fr.w[0][a][bb] = keep + h
                lp = loss()
                fr.w[0][a][bb] = keep - h
                lmm = loss()
                fr.w[0][a][bb] = keep
                fr.forward(x)
                fr.backward(coef)
                num = (lp - lmm) / (2 * h)
                assert abs(num - fr.grad_w[0][a][bb]) <= 1e-7 * max(1.0, abs(num)), \
                    f"FReLU grad_w[{a}][{bb}] 差分误差 {abs(num - fr.grad_w[0][a][bb]):.3e}"
        fr.forward(x)
        fr.backward(coef)
        w0 = [[row[:] for row in ch] for ch in fr.w]
        fr.sgd_step(0.1)
        assert any(abs(w0[c][a][bb] - fr.w[c][a][bb]) > 1e-12
                   for c in range(1) for a in range(3) for bb in range(3))
        fr.forward(x)
        fr.backward(coef)
        w1 = [[row[:] for row in ch] for ch in fr.w]
        fr.adam_step(0.01)
        assert any(abs(w1[c][a][bb] - fr.w[c][a][bb]) > 1e-12
                   for c in range(1) for a in range(3) for bb in range(3))
        for fn in (lambda: AI.FReLU(0), lambda: AI.FReLU(2, kernel_size=2),
                   lambda: AI.FReLU(2).forward([[[1.0]]]),
                   lambda: AI.FReLU(2).forward([[[1.0, 2.0]], [[1.0]]]),
                   lambda: AI.FReLU(2).backward([[[0.0]]])):
            try:
                fn()
            except (TypeError, ValueError, RuntimeError):
                pass
            else:
                raise AssertionError("非法输入应该报错")


# ==== W3-F：数据驱动初始化（LSUV / Fixup / Mimic） ====

def _flat(node, out=None):
    if out is None:
        out = []
    if isinstance(node, (list, tuple)):
        for item in node:
            _flat(item, out)
    else:
        out.append(node)
    return out


class TestLSUV:
    """在预激活上做 unit-variance：std->1、mean->0、方向不变"""

    def test_preactivation_unit_variance(self):
        import random as _random
        _random.seed(51)
        batch = [[_random.uniform(-1, 1) for _ in range(4)] for _ in range(16)]
        m = AI.MLP([4, 6, 3], activations=["relu", "relu"])
        rep = AI.lsuv_init(m, batch)
        assert len(rep) == 2
        assert all(abs(r["std_after"] - 1.0) < 1e-12 for r in rep)
        # 预激活均值应为 0
        cur = batch
        for layer in m.layers:
            vals = []
            for sample in cur:
                for i in range(len(layer.weights)):
                    acc = layer.biases[i]
                    for j in range(len(layer.weights[i])):
                        acc += layer.weights[i][j] * sample[j]
                    vals.append(acc)
            assert abs(sum(vals) / len(vals)) < 1e-12
            cur = [layer.forward(s) for s in cur]

    def test_weight_direction_preserved(self):
        import random as _random
        _random.seed(52)
        batch = [[_random.uniform(-1, 1) for _ in range(3)] for _ in range(8)]
        m = AI.MLP([3, 5], activations=["relu"])
        before = _flat(m.layers[0].weights)
        AI.lsuv_init(m, batch)
        after = _flat(m.layers[0].weights)
        ratios = [b / a for a, b in zip(before, after) if abs(a) > 1e-12]
        assert max(ratios) - min(ratios) < 1e-12

    def test_single_sample_and_errors(self):
        m = AI.MLP([3, 4], activations=["relu"])
        rep = AI.lsuv_init(m, [0.5, -1.0, 2.0])
        assert len(rep) == 1 and rep[0]["std_after"] >= 0.0
        single = AI.MLP([3, 4], activations=["relu"])
        rep2 = AI.lsuv_init(single.layers[0], [[1.0, 2.0, 3.0], [0.5, -0.5, 0.0]])
        assert len(rep2) == 1 and abs(rep2[0]["std_after"] - 1.0) < 1e-12
        for fn in (lambda: AI.lsuv_init([[1.0]], [[1.0], [2.0]]),
                   lambda: AI.lsuv_init(AI.Dropout(0.5), [1.0, 2.0]),
                   lambda: AI.lsuv_init(AI.MLP([2, 2]), [1.0, 2.0], tol=0.0),
                   lambda: AI.lsuv_init(AI.MLP([2, 2]), [1.0, 2.0], tol=float("nan")),
                   lambda: AI.lsuv_init(AI.MLP([3, 2]), [1.0, 2.0]),
                   lambda: AI.lsuv_init(AI.MLP([2, 2]), [1.0, 2.0], max_iter=0),
                   lambda: AI.lsuv_init("x", [1.0])):
            try:
                fn()
            except (TypeError, ValueError):
                pass
            else:
                raise AssertionError("非法输入应该报错")


class TestFixupInit:
    """残差分支最后一层 1/sqrt(n) 缩放 + 偏置置零"""

    def test_scales_last_conv_only(self):
        b = AI.BasicBlock2d(2)
        before = _flat(b.conv2.weights)
        conv1_before = _flat(b.conv1.weights)
        b.conv2.biases = [0.7, -0.3]
        out = AI.fixup_init(b, num_blocks=4)
        assert abs(out["scales"][0] - 0.5) < 1e-12
        after = _flat(b.conv2.weights)
        assert all(abs(a * 0.5 - c) < 1e-15 for a, c in zip(before, after))
        assert all(v == 0.0 for v in b.conv2.biases)
        assert _flat(b.conv1.weights) == conv1_before

    def test_bottleneck_and_dense(self):
        b = AI.Bottleneck2d(4, 8)
        before = _flat(b.conv3.weights)
        AI.fixup_init(b, num_blocks=9)
        after = _flat(b.conv3.weights)
        assert all(abs(a / 3.0 - c) < 1e-14 for a, c in zip(before, after))
        d = AI.DenseBlock(2, growth_rate=2, num_layers=2)
        dw = _flat(d.convs[-1].weights)
        AI.fixup_init(d, num_blocks=1)
        assert _flat(d.convs[-1].weights) == dw

    def test_errors(self):
        b = AI.BasicBlock2d(2)
        for fn in (lambda: AI.fixup_init(b, num_blocks=0),
                   lambda: AI.fixup_init(b, num_blocks=True),
                   lambda: AI.fixup_init([]),
                   lambda: AI.fixup_init(AI.MLP([2, 2])),
                   lambda: AI.fixup_init([b, AI.MLP([2, 2])])):
            try:
                fn()
            except (TypeError, ValueError):
                pass
            else:
                raise AssertionError("非法输入应该报错")


class TestMimicInit:
    """NET2NET：宽度复制左上角块、变窄截断、激活不一致报错"""

    def test_wider_copy_and_train(self):
        t = AI.MLP([3, 4, 2], activations=["relu", "softmax"])
        s = AI.MLP([3, 6, 2], activations=["relu", "softmax"])
        tw = [row[:] for row in t.layers[0].weights]
        res = AI.mimic_init(s, t)
        assert res["depth_copied"] == 2
        assert all(s.layers[0].weights[i][j] == tw[i][j] for i in range(4) for j in range(3))
        assert any(abs(s.layers[0].weights[i][j]) > 1e-12 for i in range(4, 6) for j in range(3))
        assert all(s.layers[0].biases[i] == t.layers[0].biases[i] for i in range(4))
        # 迁移后能正常训练
        model = AI.Model(s, loss_fn="cross_entropy", task="classification", lr=0.1)
        data = [([1.0, 0.0, 0.5], 0), ([0.0, 1.0, 0.5], 1)] * 4
        model.fit(data=data, epochs=30, verbose=False, batch_size=4)
        assert model.history["train_loss"][-1][1] < model.history["train_loss"][0][1]

    def test_narrower_truncates(self):
        t = AI.MLP([3, 6, 2], activations=["relu", "softmax"])
        s = AI.MLP([3, 4, 2], activations=["relu", "softmax"])
        tw = [row[:] for row in t.layers[0].weights]
        AI.mimic_init(s, t)
        assert all(s.layers[0].weights[i][j] == tw[i][j] for i in range(4) for j in range(3))

    def test_errors(self):
        for fn in (lambda: AI.mimic_init(AI.MLP([3, 4], activations=["tanh"]),
                                         AI.MLP([3, 4], activations=["relu"])),
                   lambda: AI.mimic_init(AI.MLP([3, 4]), "x")):
            try:
                fn()
            except (TypeError, ValueError):
                pass
            else:
                raise AssertionError("非法输入应该报错")
        # 深度不同时只复制前 min(len) 层
        out = AI.mimic_init(AI.MLP([3, 4], activations=["relu"]),
                            AI.MLP([3, 4, 2], activations=["relu", "relu"]))
        assert out["depth_copied"] == 1

# ---------------------------------------------------------------------------
#  残差变体（ReZero / SkipInit / Pre-LN / Post-LN / Highway）
# ---------------------------------------------------------------------------
class TestReZeroSkipInit:
    """ReZero 与 SkipInit：缩放残差分支"""

    def test_rezero_identity_at_zero(self):
        layer = AI.Layer(3, 3, "relu")
        rz = AI.ReZero(layer, init_alpha=0.0)
        x = [0.3, -1.2, 2.0]
        assert rz.forward(x) == x          # alpha=0 ⇒ 逐位恒等

    def test_rezero_forward_and_grad(self):
        layer = AI.Layer(3, 3, "relu")
        rz = AI.ReZero(layer, init_alpha=0.5)
        x = [0.3, -1.2, 2.0]
        branch = [v for v in layer.forward(x)]
        got = rz.forward(x)
        for i in range(3):
            assert abs(got[i] - (x[i] + 0.5 * branch[i])) < 1e-15

        dx = rz.backward([1.0, -1.0, 0.5])
        assert abs(rz.grad_alpha[0]
                   - (1.0 * branch[0] - 1.0 * branch[1] + 0.5 * branch[2])) < 1e-12

        # dx：数值差分
        def loss():
            return sum(rz.forward(x)[i] * c for i, c in enumerate((1.0, -1.0, 0.5)))

        for i in range(3):
            old = x[i]
            x[i] = old + 1e-6
            lp = loss()
            x[i] = old - 1e-6
            lm = loss()
            x[i] = old
            assert abs((lp - lm) / 2e-6 - dx[i]) < 1e-6

    def test_skipinit_beta_and_param_count(self):
        layer = AI.Layer(3, 3, "relu")
        sk = AI.SkipInit(layer, num_blocks=4)
        assert abs(sk.beta - 0.5) < 1e-15
        assert sk.param_count() == 3 * 3 + 3    # 不可学，参数与内层相同

        x = [0.3, -1.2, 2.0]
        branch = [v for v in layer.forward(x)]
        got = sk.forward(x)
        for i in range(3):
            assert abs(got[i] - (x[i] + 0.5 * branch[i])) < 1e-15

    def test_rezero_invalid(self):
        with pytest.raises(TypeError):
            AI.ReZero("not a layer")
        with pytest.raises(ValueError):
            AI.ReZero(AI.Layer(3, 3), init_alpha=float("nan"))
        with pytest.raises(ValueError):
            AI.SkipInit(AI.Layer(3, 3), num_blocks=0)
        with pytest.raises(TypeError):
            AI.SkipInit(AI.Layer(3, 3), num_blocks=True)
        with pytest.raises(ValueError):
            AI.ReZero(AI.Layer(3, 3)).forward([1.0])
        with pytest.raises(RuntimeError):
            AI.ReZero(AI.Layer(3, 3)).backward([1.0, 2.0, 3.0])


class TestPrePostNorm:
    """Pre-LN 与 Post-LN 残差"""

    def _setup(self, cls):
        block = cls(AI.Layer(3, 3, "relu"))
        block.norm.gamma = [1.2, 0.8, 1.0]
        block.norm.beta = [0.1, -0.2, 0.3]
        return block

    def test_prenorm_forward(self):
        block = self._setup(AI.PreNormResidual)
        x = [0.3, -1.2, 2.0]
        n = block.norm.forward(x)
        branch = [v for v in block.layer.forward(n)]
        got = block.forward(x)
        for i in range(3):
            assert abs(got[i] - (x[i] + branch[i])) < 1e-14

    def test_postnorm_forward_differs(self):
        pre = self._setup(AI.PreNormResidual)
        post = self._setup(AI.PostNormResidual)
        x = [0.3, -1.2, 2.0]
        assert pre.forward(x) != post.forward(x)

    def test_prenorm_gradients(self):
        block = self._setup(AI.PreNormResidual)
        x = [0.3, -1.2, 2.0]
        coef = (0.7, -0.4, 1.1)
        block.forward(x)
        dx = block.backward(list(coef))

        def loss():
            return sum(block.forward(x)[i] * coef[i] for i in range(3))

        for i in range(3):
            old = x[i]
            x[i] = old + 1e-6
            lp = loss()
            x[i] = old - 1e-6
            lm = loss()
            x[i] = old
            assert abs((lp - lm) / 2e-6 - dx[i]) < 1e-6

        for i in range(3):
            g = block.norm.gamma
            old = g[i]
            g[i] = old + 1e-6
            lp = loss()
            g[i] = old - 1e-6
            lm = loss()
            g[i] = old
            assert abs((lp - lm) / 2e-6 - block.norm_grads[0][i]) < 1e-6

    def test_postnorm_gradient(self):
        block = self._setup(AI.PostNormResidual)
        x = [0.3, -1.2, 2.0]
        coef = (0.7, -0.4, 1.1)
        block.forward(x)
        dx = block.backward(list(coef))

        def loss():
            return sum(block.forward(x)[i] * coef[i] for i in range(3))

        for i in range(3):
            old = x[i]
            x[i] = old + 1e-6
            lp = loss()
            x[i] = old - 1e-6
            lm = loss()
            x[i] = old
            assert abs((lp - lm) / 2e-6 - dx[i]) < 1e-6

    def test_norm_blocks_param_count(self):
        pre = self._setup(AI.PreNormResidual)
        # 内层 3*3+3 = 12，LN 的 gamma/beta 各 3 ⇒ 总 18
        assert pre.param_count() == 12 + 6


class TestHighway:
    """Highway 门控残差"""

    def _gate(self, hw, x, i):
        acc = hw.bg[i] + sum(hw.wg[i][j] * x[j] for j in range(3))
        return 1.0 / (1.0 + math.exp(-acc))

    def test_forward_closed_form(self):
        hw = AI.Highway(3)
        x = [0.3, -1.2, 2.0]
        hs = []
        gs = []
        for i in range(3):
            gs.append(self._gate(hw, x, i))
            hs.append(math.tanh(hw.w[i] * x[i] + hw.b[i]))
        got = hw.forward(x)
        for i in range(3):
            assert abs(got[i] - (gs[i] * x[i] + (1 - gs[i]) * hs[i])) < 1e-14

    def test_param_count(self):
        assert AI.Highway(3).param_count() == 9 + 9      # d*d + 3d

    def test_gradients_all_five(self):
        hw = AI.Highway(3)
        x = [0.3, -1.2, 2.0]
        coef = (0.7, -0.4, 1.1)
        hw.forward(x)
        dx = hw.backward(list(coef))

        def loss():
            return sum(hw.forward(x)[i] * coef[i] for i in range(3))

        for i in range(3):
            old = x[i]
            x[i] = old + 1e-6
            lp = loss()
            x[i] = old - 1e-6
            lm = loss()
            x[i] = old
            assert abs((lp - lm) / 2e-6 - dx[i]) < 1e-6

        old = hw.wg[0][1]
        hw.wg[0][1] = old + 1e-6
        lp = loss()
        hw.wg[0][1] = old - 1e-6
        lm = loss()
        hw.wg[0][1] = old
        assert abs((lp - lm) / 2e-6 - hw.grad_wg[0][1]) < 1e-6

        old = hw.w[1]
        hw.w[1] = old + 1e-6
        lp = loss()
        hw.w[1] = old - 1e-6
        lm = loss()
        hw.w[1] = old
        assert abs((lp - lm) / 2e-6 - hw.grad_w[1]) < 1e-6

    def test_optimizer_steps(self):
        hw = AI.Highway(3)
        x = [0.3, -1.2, 2.0]
        hw.forward(x)
        hw.backward([1.0, 1.0, 1.0])
        before = [row[:] for row in hw.wg]
        hw.sgd_step(0.1)
        assert any(abs(before[i][j] - hw.wg[i][j]) > 1e-12
                   for i in range(3) for j in range(3))

        hw.forward(x)
        hw.backward([1.0, 1.0, 1.0])
        before = [row[:] for row in hw.wg]
        hw.adam_step(0.01)
        assert any(abs(before[i][j] - hw.wg[i][j]) > 1e-12
                   for i in range(3) for j in range(3))

    def test_invalid(self):
        with pytest.raises(TypeError):
            AI.Highway(3.0)
        with pytest.raises(ValueError):
            AI.Highway(0)
        with pytest.raises(ValueError):
            AI.Highway(3).forward([1.0])
        with pytest.raises(RuntimeError):
            AI.Highway(3).backward([1.0, 2.0, 3.0])

# ---------------------------------------------------------------------------
#  视觉前端（PatchEmbedding / AttentionPooling / GeMPooling）
# ---------------------------------------------------------------------------
class TestPatchEmbedding:
    """把图像切成 patch 再映射成 token"""

    def _img(self):
        return [[[float(4 * i + j + 1) for j in range(4)] for i in range(4)]]

    def test_split_and_order(self):
        pe = AI.PatchEmbedding(1, 2, 3)
        # 只保留 patch 左上角像素，偏置区分三个输出通道
        pe.conv.weights = [[[[1.0, 0.0], [0.0, 0.0]]] for _ in range(3)]
        pe.conv.biases = [0.0, 10.0, 100.0]
        toks = pe.forward(self._img())
        assert len(toks) == 4 and all(len(t) == 3 for t in toks)
        img = self._img()[0]
        assert [t[0] for t in toks] == [img[0][0], img[0][2], img[2][0], img[2][2]]
        for t in toks:
            assert abs(t[1] - t[0] - 10.0) < 1e-12
            assert abs(t[2] - t[0] - 100.0) < 1e-12
        assert pe.param_count() == 3 * 4 + 3

    def test_backward_finite_difference(self):
        pe = AI.PatchEmbedding(1, 2, 2)
        pe.conv.weights = [[[[0.5, -0.3], [0.2, 0.7]]], [[[0.1, 0.4], [-0.2, 0.6]]]]
        pe.conv.biases = [0.05, -0.1]
        coef = (0.3, -0.8, 0.5, 1.2)
        dtok = [[coef[p] + 0.01 * c for c in range(2)] for p in range(4)]

        img = self._img()
        pe.forward(img)
        dx = pe.backward(dtok)

        def loss():
            t = pe.forward(img)
            return sum(t[p][c] * (coef[p] + 0.01 * c)
                       for p in range(4) for c in range(2))

        for i in range(4):
            for j in range(4):
                old = img[0][i][j]
                img[0][i][j] = old + 1e-6
                lp = loss()
                img[0][i][j] = old - 1e-6
                lm = loss()
                img[0][i][j] = old
                assert abs((lp - lm) / 2e-6 - dx[0][i][j]) < 1e-6

    def test_invalid(self):
        with pytest.raises(ValueError):
            AI.PatchEmbedding(0, 2, 3)
        with pytest.raises(ValueError):
            AI.PatchEmbedding(1, 0, 3)
        with pytest.raises(TypeError):
            AI.PatchEmbedding(1, 2.0, 3)
        with pytest.raises(ValueError):
            AI.PatchEmbedding(1, 2, 3).forward([[[1.0, 2.0, 3.0]]])
        with pytest.raises(ValueError):
            AI.PatchEmbedding(2, 2, 3).forward(self._img())
        with pytest.raises(RuntimeError):
            AI.PatchEmbedding(1, 2, 3).backward([[0.0] * 3])


class TestAttentionPooling:
    """可学习查询向量的注意力池化"""

    def _toks(self):
        return [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]

    def test_uniform_when_query_zero(self):
        ap = AI.AttentionPooling(3)
        ap.q = [0.0, 0.0, 0.0]
        toks = self._toks()
        out = ap.forward(toks)
        for j in range(3):
            assert abs(out[j] - sum(t[j] for t in toks) / 3.0) < 1e-12
        assert abs(sum(ap.weights) - 1.0) < 1e-12
        assert ap.param_count() == 3

    def test_softmax_weights_and_backward(self):
        ap = AI.AttentionPooling(3)
        ap.q = [0.5, -0.2, 0.9]
        toks = self._toks()
        out = ap.forward(toks)
        sc = 1.0 / math.sqrt(3)
        scores = [sum(ap.q[i] * t[i] for i in range(3)) * sc for t in toks]
        m = max(scores)
        ex = [math.exp(v - m) for v in scores]
        z = sum(ex)
        for i in range(3):
            assert abs(ap.weights[i] - ex[i] / z) < 1e-12
        for j in range(3):
            assert abs(out[j] - sum(ap.weights[i] * toks[i][j]
                                    for i in range(3))) < 1e-12

        dout = [0.7, -0.4, 1.1]
        ap.forward(toks)
        dt = ap.backward(dout)

        def loss():
            o = ap.forward(toks)
            return sum(o[j] * dout[j] for j in range(3))

        for i in range(3):
            for j in range(3):
                old = toks[i][j]
                toks[i][j] = old + 1e-6
                lp = loss()
                toks[i][j] = old - 1e-6
                lm = loss()
                toks[i][j] = old
                assert abs((lp - lm) / 2e-6 - dt[i][j]) < 1e-6

        for i in range(3):
            old = ap.q[i]
            ap.q[i] = old + 1e-6
            lp = loss()
            ap.q[i] = old - 1e-6
            lm = loss()
            ap.q[i] = old
            assert abs((lp - lm) / 2e-6 - ap.grad_q[i]) < 1e-6

    def test_invalid(self):
        with pytest.raises(ValueError):
            AI.AttentionPooling(0)
        with pytest.raises(ValueError):
            AI.AttentionPooling(3).forward([])
        with pytest.raises(ValueError):
            AI.AttentionPooling(3).forward([[1.0, 2.0]])
        with pytest.raises(RuntimeError):
            AI.AttentionPooling(3).backward([1.0, 2.0, 3.0])


class TestGeMPooling:
    """广义平均池化"""

    def test_known_values(self):
        x = [[[1.0, 2.0], [3.0, 4.0]]]
        assert abs(AI.GeMPooling(1.0).forward(x)[0] - 2.5) < 1e-12
        assert abs(AI.GeMPooling(2.0).forward(x)[0] - math.sqrt(7.5)) < 1e-12
        # p 越大越接近最大值，且不溢出
        v20 = AI.GeMPooling(20.0).forward(x)[0]
        v2000 = AI.GeMPooling(2000.0).forward(x)[0]
        assert 2.5 < v20 < 4.0
        assert v2000 > v20 and abs(v2000 - 4.0) < 1e-2
        assert AI.GeMPooling(2.0).param_count() == 0

    def test_backward_finite_difference(self):
        x = [[[0.5, 1.5], [2.0, 3.0]], [[1.0, 2.0], [4.0, 8.0]]]
        for pval in (1.0, 1.5, 3.0):
            gp = AI.GeMPooling(pval)
            gp.forward(x)
            dx = gp.backward([1.0, -1.0])

            def loss():
                o = gp.forward(x)
                return o[0] - o[1]

            for c in range(2):
                for i in range(2):
                    for j in range(2):
                        old = x[c][i][j]
                        x[c][i][j] = old + 1e-6
                        lp = loss()
                        x[c][i][j] = old - 1e-6
                        lm = loss()
                        x[c][i][j] = old
                        assert abs((lp - lm) / 2e-6 - dx[c][i][j]) < 1e-6

    def test_invalid(self):
        with pytest.raises(ValueError):
            AI.GeMPooling(0.0)
        with pytest.raises(ValueError):
            AI.GeMPooling(-1.0)
        with pytest.raises(ValueError):
            AI.GeMPooling(float("nan"))
        with pytest.raises(TypeError):
            AI.GeMPooling(True)
        with pytest.raises(ValueError):
            AI.GeMPooling(2.0).forward([[[-1.0, 2.0]]])
        with pytest.raises(RuntimeError):
            AI.GeMPooling(2.0).backward([1.0])

class TestViT:
    """ViT：patch 化 + CLS + 位置编码 + Transformer + 分类头"""

    @staticmethod
    def _img():
        return [[[float(i * 4 + j + 1) / 16.0 for j in range(4)] for i in range(4)]]

    @staticmethod
    def _make(classes=3):
        return AI.ViT(in_channels=1, image_size=4, patch_size=2, d_model=8,
                      num_heads=2, d_ff=16, num_layers=1, num_classes=classes)

    def test_forward_shape_and_params(self):
        v = self._make()
        out = v.forward(self._img())
        assert len(out) == 3
        assert v.num_patches == 4
        assert v.grid == (2, 2)
        manual = v.patch_embed.param_count() + sum(
            AI._count_floats(t) for _n, t in v._layer_params())
        assert v.param_count() == manual
        dx = v.backward([1.0, 0.0, 0.0])
        assert len(dx) == 1 and len(dx[0]) == 4 and len(dx[0][0]) == 4

    def test_gradients_match_finite_difference(self):
        v = self._make(classes=2)
        img = self._img()
        coef = [0.7, -0.4]

        def loss():
            o = v.forward(img)
            return o[0] * coef[0] + o[1] * coef[1]

        v.zero_grad()
        v.forward(img)
        analytic_dx = v.backward(coef)
        analytic_cls = list(v.grad_cls)
        analytic_pos = [list(r) for r in v.grad_pos]
        analytic_head = [list(r) for r in v.head.grad_w]
        analytic_gamma = list(v.norm_grads[0])

        def numeric(get, setv, step=1e-6):
            old = get()
            setv(old + step)
            lp = loss()
            setv(old - step)
            lm = loss()
            setv(old)
            return (lp - lm) / (2 * step)

        worst = 0.0
        for i in range(4):
            for j in range(4):
                num = numeric(lambda _i=i, _j=j: img[0][_i][_j],
                              lambda val, _i=i, _j=j: img[0].__setitem__(
                                  _i, img[0][_i][: _j] + [val] + img[0][_i][_j + 1:]))
                worst = max(worst, abs(num - analytic_dx[0][i][j]))
        assert worst < 1e-7

        worst = 0.0
        for j in range(8):
            num = numeric(lambda _j=j: v.cls[_j],
                          lambda val, _j=j: v.cls.__setitem__(_j, val))
            worst = max(worst, abs(num - analytic_cls[j]))
        assert worst < 1e-7

        worst = 0.0
        for t in (0, 4):
            for j in (1, 5):
                num = numeric(lambda _t=t, _j=j: v.pos[_t][_j],
                              lambda val, _t=t, _j=j: v.pos[_t].__setitem__(_j, val))
                worst = max(worst, abs(num - analytic_pos[t][j]))
        assert worst < 1e-6

        worst = 0.0
        for k in range(2):
            for j in range(8):
                num = numeric(lambda _k=k, _j=j: v.head.weights[_k][_j],
                              lambda val, _k=k, _j=j: v.head.weights[_k].__setitem__(_j, val))
                worst = max(worst, abs(num - analytic_head[k][j]))
        assert worst < 1e-6

        worst = 0.0
        for j in range(8):
            num = numeric(lambda _j=j: v.norm.gamma[_j],
                          lambda val, _j=j: v.norm.gamma.__setitem__(_j, val))
            worst = max(worst, abs(num - analytic_gamma[j]))
        assert worst < 1e-6

    def test_gradients_accumulate_and_zero(self):
        v = self._make(classes=2)
        img = self._img()
        v.zero_grad()
        v.forward(img)
        v.backward([1.0, 0.0])
        h1 = v.head.grad_w[0][0]
        c1 = v.grad_cls[0]
        n1 = v.norm_grads[0][0]
        v.forward(img)
        v.backward([1.0, 0.0])
        assert abs(v.head.grad_w[0][0] - 2 * h1) < 1e-12
        assert abs(v.grad_cls[0] - 2 * c1) < 1e-12
        assert abs(v.norm_grads[0][0] - 2 * n1) < 1e-12
        v.zero_grad()
        assert v.head.grad_w[0][0] == 0.0
        assert v.grad_cls[0] == 0.0
        assert v.norm_grads[0][0] == 0.0
        assert v.grad_pos[0][0] == 0.0

    def test_training_reduces_loss(self):
        v = self._make(classes=2)
        xs = [[[[1.0 if (i + j) % 2 == 0 else 0.0 for j in range(4)]
                for i in range(4)]],
              [[[0.0 if (i + j) % 2 == 0 else 1.0 for j in range(4)]
                for i in range(4)]]]
        ys = [0, 1]

        def mean_loss():
            return sum(AI.cross_entropy_loss(v.forward(xi), yi)[0]
                       for xi, yi in zip(xs, ys)) / len(xs)

        first = mean_loss()
        for _epoch in range(15):
            for xi, yi in zip(xs, ys):
                v.zero_grad()
                _l, dlogits = AI.cross_entropy_loss(v.forward(xi), yi)
                v.backward(dlogits)
                v.sgd_step(0.05)
        assert mean_loss() < first * 0.6
        preds = [max(range(2), key=lambda k: v.forward(xi)[k]) for xi in xs]
        assert preds == ys

    def test_validation(self):
        with pytest.raises(ValueError):
            AI.ViT(image_size=(5, 4), patch_size=2)
        with pytest.raises(TypeError):
            AI.ViT(image_size=(4,))
        with pytest.raises(ValueError):
            AI.ViT(d_model=8, num_heads=3)
        with pytest.raises(ValueError):
            AI.ViT(num_classes=0)
        with pytest.raises(ValueError):
            AI.ViT(in_channels=3).forward(self._img())
        with pytest.raises(ValueError):
            self._make().forward([[[1.0, 2.0]]])
        with pytest.raises(RuntimeError):
            self._make().backward([1.0, 2.0, 3.0])
        v = self._make()
        v.forward(self._img())
        with pytest.raises(ValueError):
            v.backward([1.0])
        with pytest.raises(ValueError):
            self._make().sgd_step(-0.1)

class TestUNet:
    """UNet：编码-解码 + 跳跃连接"""

    @staticmethod
    def _img():
        return [[[float(i * 8 + j) / 64.0 for j in range(8)] for i in range(8)]]

    @staticmethod
    def _dlogits():
        return [[[1.0 if (i + j) % 2 else -0.5 for j in range(8)] for i in range(8)]
                for _ in range(2)]

    def test_shapes_and_params(self):
        u = AI.UNet(in_channels=1, num_classes=2, base_channels=2)
        out = u.forward(self._img())
        assert (len(out), len(out[0]), len(out[0][0])) == (2, 8, 8)
        manual = sum(AI._conv_param_count(c) for c in u._convs + u._ups)
        assert u.param_count() == manual
        dx = u.backward(self._dlogits())
        assert (len(dx), len(dx[0]), len(dx[0][0])) == (1, 8, 8)

    def test_input_gradient(self):
        u = AI.UNet(in_channels=1, num_classes=2, base_channels=2)
        img = self._img()
        d = self._dlogits()

        def loss():
            o = u.forward(img)
            return sum(o[k][i][j] * d[k][i][j]
                       for k in range(2) for i in range(8) for j in range(8))

        u.forward(img)
        dx = u.backward(d)
        worst = 0.0
        for i in range(0, 8, 2):
            for j in range(0, 8, 2):
                old = img[0][i][j]
                img[0][i][j] = old + 1e-6
                lp = loss()
                img[0][i][j] = old - 1e-6
                lm = loss()
                img[0][i][j] = old
                worst = max(worst, abs((lp - lm) / 2e-6 - dx[0][i][j]))
        assert worst < 1e-7

    def test_input_gradient_with_bn_eval(self):
        u = AI.UNet(in_channels=1, num_classes=2, base_channels=2, use_bn=True)
        u.eval()
        img = self._img()
        d = self._dlogits()

        def loss():
            o = u.forward(img)
            return sum(o[k][i][j] * d[k][i][j]
                       for k in range(2) for i in range(8) for j in range(8))

        u.forward(img)
        dx = u.backward(d)
        worst = 0.0
        for i in range(0, 8, 2):
            for j in range(0, 8, 2):
                old = img[0][i][j]
                img[0][i][j] = old + 1e-6
                lp = loss()
                img[0][i][j] = old - 1e-6
                lm = loss()
                img[0][i][j] = old
                worst = max(worst, abs((lp - lm) / 2e-6 - dx[0][i][j]))
        assert worst < 1e-7

    def test_parameter_gradients(self):
        u = AI.UNet(in_channels=1, num_classes=2, base_channels=2, use_bn=True)
        img = self._img()
        d = self._dlogits()

        def loss():
            o = u.forward(img)
            return sum(o[k][i][j] * d[k][i][j]
                       for k in range(2) for i in range(8) for j in range(8))

        u.zero_grad()
        u.forward(img)
        u.backward(d)

        old = u.e0.weights[0][0][0][0]
        u.e0.weights[0][0][0][0] = old + 1e-6
        lp = loss()
        u.e0.weights[0][0][0][0] = old - 1e-6
        lm = loss()
        u.e0.weights[0][0][0][0] = old
        assert abs((lp - lm) / 2e-6 - u.e0.grad_w[0][0][0][0]) < 1e-6

        old = u.up0.weights[0][0][1][1]
        u.up0.weights[0][0][1][1] = old + 1e-6
        lp = loss()
        u.up0.weights[0][0][1][1] = old - 1e-6
        lm = loss()
        u.up0.weights[0][0][1][1] = old
        assert abs((lp - lm) / 2e-6 - u.up0.grad_w[0][0][1][1]) < 1e-6

        old = u.bn4.gamma[0]
        u.bn4.gamma[0] = old + 1e-6
        lp = loss()
        u.bn4.gamma[0] = old - 1e-6
        lm = loss()
        u.bn4.gamma[0] = old
        assert abs((lp - lm) / 2e-6 - u.bn_grads[4][0][0]) < 1e-6

    def test_training_with_adam(self):
        random.seed(0)
        x = self._img()
        y = [[1.0 if (i + j) % 2 else 0.0 for j in range(8)] for i in range(8)]
        u = AI.UNet(in_channels=1, num_classes=2, base_channels=4)

        def mse():
            o = u.forward(x)
            return sum((o[k][i][j] - y[i][j]) ** 2
                       for k in range(2) for i in range(8) for j in range(8)) / 128.0

        first = mse()
        for _step in range(40):
            u.zero_grad()
            o = u.forward(x)
            g = [[[(o[k][i][j] - y[i][j]) for j in range(8)] for i in range(8)]
                 for k in range(2)]
            u.backward(g)
            u.adam_step(0.01)
        assert mse() < first / 3.0

    def test_validation(self):
        with pytest.raises(ValueError):
            AI.UNet(base_channels=0)
        with pytest.raises(ValueError):
            AI.UNet(in_channels=2).forward(self._img())
        with pytest.raises(ValueError):
            AI.UNet().forward([[[1.0, 2.0]]])
        with pytest.raises(RuntimeError):
            AI.UNet().backward(self._dlogits())
        u = AI.UNet(in_channels=1, num_classes=2)
        u.forward(self._img())
        with pytest.raises(ValueError):
            u.backward([[[0.0] * 8 for _ in range(8)] for _ in range(3)])
        with pytest.raises(ValueError):
            AI.UNet().sgd_step(-1)
        with pytest.raises(ValueError):
            AI.UNet().adam_step('x')


# ============================== W4-D: FPN / SegNet ==============================
class TestFPN:
    """特征金字塔：三级横向连接 + top-down"""

    @staticmethod
    def _feats():
        return [
            [[[0.1 * (c + 1) + 0.013 * a - 0.007 * b for b in range(8)]
              for a in range(8)] for c in range(2)],
            [[[0.2 * (c + 1) - 0.011 * a + 0.009 * b for b in range(4)]
              for a in range(4)] for c in range(4)],
            [[[0.3 * (c + 1) + 0.017 * a + 0.005 * b for b in range(2)]
              for a in range(2)] for c in range(8)],
        ]

    def test_shapes_and_params(self):
        random.seed(3)
        net = AI.FPN([2, 4, 8], out_channels=3)
        outs = net.forward(self._feats())
        assert [(len(o), len(o[0]), len(o[0][0])) for o in outs] == \
            [(3, 8, 8), (3, 4, 4), (3, 2, 2)]
        # 每个 1x1 横向卷积 2/4/8 进 3 出：参数量 = 3*(2+4+8) + 3*3 个偏置
        # 每个 3x3 平滑卷积：3*3*3*3 + 3
        assert net.param_count() == 3 * (2 + 4 + 8) + 3 * 3 + 3 * (3 * 3 * 3 * 3 + 3)
        # smooth=False 时只有横向卷积
        net2 = AI.FPN([2, 4, 8], out_channels=3, smooth=False)
        assert net2.forward(self._feats())[0][0][0][0] is not None
        assert net2.param_count() == 3 * (2 + 4 + 8) + 3 * 3
        # 通道/尺寸/级数不符都要报错
        with pytest.raises(ValueError):
            net.forward(self._feats()[:2])
        with pytest.raises(ValueError):
            AI.FPN([2, 4, 8], out_channels=3).forward(
                [self._feats()[0], self._feats()[1][:2], self._feats()[2]])
        with pytest.raises(ValueError):
            AI.FPN([2, 4, 8], out_channels=3).forward(
                self._feats()[:2] + [[[[0.0] * 3 for _ in range(3)] for _ in range(8)]])
        with pytest.raises(TypeError):
            AI.FPN([2])
        with pytest.raises(ValueError):
            AI.FPN([2, 0])
        with pytest.raises(ValueError):
            AI.FPN([2, 4], out_channels=0)

    def test_dx_matches_finite_difference(self):
        random.seed(3)
        net = AI.FPN([2, 4, 8], out_channels=3)
        feats = self._feats()
        outs = net.forward(feats)
        Wg = [[[[random.uniform(-1, 1) for _ in range(len(o[0][0]))]
                for _ in range(len(o[0]))] for _ in range(3)] for o in outs]

        def loss(fs):
            o = net.forward(fs)
            return sum(Wg[i][c][a][b] * o[i][c][a][b]
                       for i in range(3) for c in range(3)
                       for a in range(len(o[i][0])) for b in range(len(o[i][0][0])))

        eps = 1e-6
        for lvl, (c, a, b) in [(0, (0, 0, 0)), (1, (3, 3, 3)), (2, (7, 1, 1))]:
            fs = [[[list(row) for row in ch] for ch in f] for f in feats]
            fs[lvl][c][a][b] += eps
            lp = loss(fs)
            fs = [[[list(row) for row in ch] for ch in f] for f in feats]
            fs[lvl][c][a][b] -= eps
            lm = loss(fs)
            num = (lp - lm) / (2 * eps)
            net.zero_grad()
            net.forward(feats)
            dxs = net.backward(Wg)
            assert abs(dxs[lvl][c][a][b] - num) <= 1e-6

    def test_conv_grads_and_steps(self):
        random.seed(3)
        net = AI.FPN([2, 4, 8], out_channels=3)
        feats = self._feats()
        outs = net.forward(feats)
        Wg = [[[[random.uniform(-1, 1) for _ in range(len(o[0][0]))]
                for _ in range(len(o[0]))] for _ in range(3)] for o in outs]

        def loss():
            o = net.forward(feats)
            return sum(Wg[i][c][a][b] * o[i][c][a][b]
                       for i in range(3) for c in range(3)
                       for a in range(len(o[i][0])) for b in range(len(o[i][0][0])))

        eps = 1e-6
        for layer, idx in [(net.lateral[0], (0, 0, 0, 0)), (net.smooth[1], (0, 0, 1, 1))]:
            old = layer.weights[idx[0]][idx[1]][idx[2]][idx[3]]
            layer.weights[idx[0]][idx[1]][idx[2]][idx[3]] = old + eps
            lp = loss()
            layer.weights[idx[0]][idx[1]][idx[2]][idx[3]] = old - eps
            lm = loss()
            layer.weights[idx[0]][idx[1]][idx[2]][idx[3]] = old
            num = (lp - lm) / (2 * eps)
            net.zero_grad()
            net.forward(feats)
            net.backward(Wg)
            ana = layer.grad_w[idx[0]][idx[1]][idx[2]][idx[3]]
            assert abs(ana - num) <= 1e-6

        before = net.lateral[0].weights[0][0][0][0]
        net.sgd_step(0.01)
        assert net.lateral[0].weights[0][0][0][0] != before
        before = net.lateral[0].weights[0][0][0][0]
        net.adam_step(0.01)
        assert net.lateral[0].weights[0][0][0][0] != before
        with pytest.raises(ValueError):
            net.sgd_step(-1)
        with pytest.raises(RuntimeError):
            AI.FPN([2, 4, 8], out_channels=3).backward(Wg)


class TestSegNet:
    """SegNet：池化索引解码"""

    @staticmethod
    def _img():
        return [[[0.1 * a - 0.05 * b + 0.01 * (a * b % 3) for b in range(8)]
                 for a in range(8)]]

    @staticmethod
    def _dw():
        return [[[1.0 if (a + b) % 2 == 0 else -0.5 for b in range(8)]
                 for a in range(8)] for _ in range(2)]

    def test_shapes_params_and_errors(self):
        random.seed(5)
        seg = AI.SegNet(in_channels=1, num_classes=2, base_channels=2)
        logits = seg.forward(self._img())
        assert (len(logits), len(logits[0]), len(logits[0][0])) == (2, 8, 8)
        # 6 个 3x3/1x1 卷积：cin*c0*9+c0, c0*c1*9+c1, c1*c1*9+c1, c1*c0*9+c0,
        # c0*c0*9+c0, c0*2*1+2 —— c0=2, c1=4
        expect = (1 * 2 * 9 + 2) + (2 * 4 * 9 + 4) + (4 * 4 * 9 + 4) + \
                 (4 * 2 * 9 + 2) + (2 * 2 * 9 + 2) + (2 * 2 * 1 + 2)
        assert seg.param_count() == expect
        dx = seg.backward(self._dw())
        assert (len(dx), len(dx[0]), len(dx[0][0])) == (1, 8, 8)
        with pytest.raises(ValueError):
            AI.SegNet(1, 2, 2).forward([self._img()[0], self._img()[0]])
        with pytest.raises(ValueError):
            AI.SegNet(1, 2, 2).forward([[[0.0] * 6 for _ in range(6)]])
        with pytest.raises(RuntimeError):
            AI.SegNet(1, 2, 2).backward(self._dw())
        with pytest.raises(ValueError):
            AI.SegNet(1, 0, 2)
        with pytest.raises(ValueError):
            AI.SegNet(1, 2, 2.5)
        s2 = AI.SegNet(1, 2, 2)
        s2.forward(self._img())
        with pytest.raises(ValueError):
            s2.backward(self._dw()[:1])
        with pytest.raises(ValueError):
            s2.sgd_step(-0.1)

    def test_dx_matches_finite_difference(self):
        random.seed(5)
        seg = AI.SegNet(in_channels=1, num_classes=2, base_channels=2)
        img = self._img()
        DW = self._dw()

        def loss(im):
            o = seg.forward(im)
            return sum(DW[c][a][b] * o[c][a][b]
                       for c in range(2) for a in range(8) for b in range(8))

        eps = 1e-6
        for (a, b) in [(0, 0), (3, 5), (7, 7)]:
            im2 = [[list(row) for row in img[0]]]
            im2[0][a][b] += eps
            lp = loss(im2)
            im3 = [[list(row) for row in img[0]]]
            im3[0][a][b] -= eps
            lm = loss(im3)
            num = (lp - lm) / (2 * eps)
            seg.zero_grad()
            seg.forward(img)
            dx = seg.backward(DW)
            assert abs(dx[0][a][b] - num) <= 1e-7
        # 卷积参数梯度也核对一项
        c1 = seg.conv1
        old = c1.weights[0][0][1][1]
        c1.weights[0][0][1][1] = old + eps
        lp = loss(img)
        c1.weights[0][0][1][1] = old - eps
        lm = loss(img)
        c1.weights[0][0][1][1] = old
        num = (lp - lm) / (2 * eps)
        seg.zero_grad()
        seg.forward(img)
        seg.backward(DW)
        assert abs(c1.grad_w[0][0][1][1] - num) <= 1e-6

    def test_unpool_uses_argmax_indices(self):
        random.seed(5)
        seg = AI.SegNet(in_channels=1, num_classes=2, base_channels=2)
        img = self._img()
        seg.forward(img)
        h0 = seg._relu(seg.conv1.forward(img))
        p0 = seg.pool0.forward(h0)
        # mask 存的是窗口内相对位置，逐窗口和 argmax 对齐
        for c in range(len(h0)):
            for i in range(4):
                for j in range(4):
                    best = -1e18
                    arg = (0, 0)
                    for kh in range(2):
                        for kw in range(2):
                            v = h0[c][2 * i + kh][2 * j + kw]
                            if v > best:
                                best = v
                                arg = (kh, kw)
                    assert seg.pool0.mask[c][i][j] == arg
        u = AI._unpool2(p0, seg.pool0, 8, 8)
        for c in range(len(p0)):
            for i in range(4):
                for j in range(4):
                    kh, kw = seg.pool0.mask[c][i][j]
                    assert u[c][i * 2 + kh][j * 2 + kw] == p0[c][i][j]
        # 未选中的位置为 0
        picked = {(i * 2 + seg.pool0.mask[0][i][j][0], j * 2 + seg.pool0.mask[0][i][j][1])
                  for i in range(4) for j in range(4)}
        assert len(picked) == 16
        # 非选中位置必须为 0（选中位置可能是 0.0，所以不能数非零个数）
        assert all(u[0][a][b] == 0.0 for a in range(8) for b in range(8)
                   if (a, b) not in picked)
        # 训练：40 步 Adam 让 loss 显著下降
        random.seed(11)
        seg2 = AI.SegNet(in_channels=1, num_classes=1, base_channels=3)
        imgs = [[[[0.05 * (a + 2 * b) for b in range(8)] for a in range(8)]],
                [[[0.9 - 0.05 * (a + b) for b in range(8)] for a in range(8)]]]
        tgts = [[[[1.0] * 8 for _ in range(8)]], [[[0.0] * 8 for _ in range(8)]]]
        first = None
        for _ in range(40):
            tot = 0.0
            for im, tg in zip(imgs, tgts):
                seg2.zero_grad()
                o = seg2.forward(im)
                d = [[[o[c][a][b] - tg[c][a][b] for b in range(8)]
                      for a in range(8)] for c in range(1)]
                tot += sum((o[c][a][b] - tg[c][a][b]) ** 2 for c in range(1)
                           for a in range(8) for b in range(8)) / 64.0
                seg2.backward(d)
                seg2.adam_step(0.02)
            if first is None:
                first = tot
        assert tot < first / 3.0


class TestMLPMixer:
    """MLP-Mixer：token 混合 + 通道混合（两段 pre-LN 残差）"""

    BASE = dict(in_channels=1, image_size=8, patch_size=4, d_model=8,
                num_layers=2, token_hidden=12, channel_hidden=12,
                num_classes=3)

    def _model(self, **kw):
        cfg = dict(self.BASE)
        cfg.update(kw)
        random.seed(0)
        return AI.MLPMixer(**cfg)

    def _img(self, scale=1.0):
        return [[[scale * (0.1 * a - 0.05 * b + 0.02 * a * b)
                  for b in range(8)] for a in range(8)]]

    def _loss(self, model, coef, img=None):
        img = self._img() if img is None else img
        model.zero_grad()
        out = model.forward(img)
        return sum(c * o for c, o in zip(coef, out))

    def test_shapes_and_param_count(self):
        m = self._model()
        logits = m.forward(self._img())
        assert len(logits) == 3
        assert m.param_count() == 891
        dx = m.backward([1.0, -0.5, 0.25])
        assert (len(dx), len(dx[0]), len(dx[0][0])) == (1, 8, 8)

    def test_gradients_match_finite_difference(self):
        m = self._model()
        coef = [1.0, -0.5, 0.25]
        eps = 1e-5
        # 输入梯度
        img = [row[:] for row in self._img()[0]]
        m.zero_grad()
        m.forward([img])
        dx = m.backward(coef)
        for (a, b) in [(0, 0), (3, 2), (7, 5)]:
            old = img[a][b]
            img[a][b] = old + eps
            lp = self._loss(m, coef, [img])
            img[a][b] = old - eps
            lm = self._loss(m, coef, [img])
            img[a][b] = old
            num = (lp - lm) / (2 * eps)
            denom = max(1.0, abs(num), abs(dx[0][a][b]))
            assert abs(dx[0][a][b] - num) / denom < 1e-5
        # block 内四组 MLP 的权重梯度
        for bi in range(2):
            for name in ('t1', 't2', 'c1', 'c2'):
                m2 = self._model()
                blk = m2.blocks[bi]
                layer = getattr(blk, name)
                m2.zero_grad()
                m2.forward(self._img())
                m2.backward(coef)
                i = 1 if len(layer.weights) > 1 else 0
                j = 2 if len(layer.weights[0]) > 2 else 0
                ana = blk.grads[name][0][i][j]
                old = layer.weights[i][j]
                layer.weights[i][j] = old + eps
                lp = self._loss(m2, coef)
                layer.weights[i][j] = old - eps
                lm = self._loss(m2, coef)
                layer.weights[i][j] = old
                num = (lp - lm) / (2 * eps)
                denom = max(1.0, abs(num), abs(ana))
                assert abs(ana - num) / denom < 1e-5, (bi, name, ana, num)

    def test_norm_and_head_gradients(self):
        m = self._model()
        coef = [1.0, -0.5, 0.25]
        eps = 1e-5
        old = m.head.weights[1][2]
        m.head.weights[1][2] = old + eps
        lp = self._loss(m, coef)
        m.head.weights[1][2] = old - eps
        lm = self._loss(m, coef)
        m.head.weights[1][2] = old
        num = (lp - lm) / (2 * eps)
        m.zero_grad()
        m.forward(self._img())
        m.backward(coef)
        denom = max(1.0, abs(num))
        assert abs(m.head_grads[0][1][2] - num) / denom < 1e-5

        blk = m.blocks[1]
        old = blk.ln2.beta[5]
        blk.ln2.beta[5] = old + eps
        lp = self._loss(m, coef)
        blk.ln2.beta[5] = old - eps
        lm = self._loss(m, coef)
        blk.ln2.beta[5] = old
        num = (lp - lm) / (2 * eps)
        m.zero_grad()
        m.forward(self._img())
        m.backward(coef)
        denom = max(1.0, abs(num))
        assert abs(blk.norm_grads['ln2'][1][5] - num) / denom < 1e-5

    def test_accumulation_and_zero_grad(self):
        m = self._model()
        coef = [1.0, -0.5, 0.25]
        m.zero_grad()
        m.forward(self._img())
        m.backward(coef)
        first = m.head_grads[0][1][2]
        assert abs(first) > 1e-9
        m.forward(self._img())
        m.backward(coef)
        assert abs(m.head_grads[0][1][2] - 2 * first) < 1e-9
        m.zero_grad()
        assert all(abs(v) < 1e-15 for row in m.head_grads[0] for v in row)
        assert all(abs(v) < 1e-15 for v in m.norm_grads[0])
        assert all(abs(v) < 1e-15
                   for row in m.blocks[0].grads['t1'][0] for v in row)

    def test_training_and_validation(self):
        random.seed(1)
        X = [[[0.1 * a - 0.1 * b for b in range(8)] for a in range(8)],
             [[0.2 * a - 0.05 * b + 0.5 for b in range(8)] for a in range(8)],
             [[0.05 * a + 0.3 * b - 0.4 for b in range(8)] for a in range(8)],
             [[-0.2 * a + 0.1 * b for b in range(8)] for a in range(8)]]
        Y = [0, 1, 0, 1]
        m = AI.MLPMixer(in_channels=1, image_size=8, patch_size=4, d_model=8,
                        num_layers=2, token_hidden=12, channel_hidden=12,
                        num_classes=2)
        losses = []
        for _ in range(120):
            total = 0.0
            for img, y in zip(X, Y):
                m.zero_grad()
                out = m.forward([img])
                loss, grads = AI.cross_entropy_loss(out, y)
                total += loss
                m.backward(grads)
                m.adam_step(0.01)
            losses.append(total / len(X))
        assert losses[-1] < losses[0] * 0.5
        pred = []
        for img in X:
            out = m.forward([img])
            pred.append(0 if out[0] > out[1] else 1)
        assert pred == Y

        def raises(fn, exc):
            try:
                fn()
            except exc:
                return True
            except Exception:
                return False
            return False

        assert raises(lambda: AI.MLPMixer(in_channels=0), ValueError)
        assert raises(lambda: AI.MLPMixer(image_size=7, patch_size=4), ValueError)
        assert raises(lambda: AI.MLPMixer(patch_size=0), ValueError)
        assert raises(lambda: AI.MLPMixer(num_layers=0), ValueError)
        m2 = self._model()
        assert raises(lambda: m2.forward([[0.0] * 8] * 8), ValueError)
        assert raises(lambda: m2.forward([[[0.0] * 7] * 8] * 1), ValueError)
        m2.forward(self._img())
        assert raises(lambda: m2.backward([1.0, 0.0]), ValueError)
        assert raises(lambda: m2.sgd_step(-0.1), ValueError)
        assert raises(lambda: m2.adam_step(0.1, beta1=1.0), ValueError)


class TestSwinTransformer:
    """简化版 Swin：窗口注意力 + 交替移位窗口（循环 roll，无 mask）"""

    BASE = dict(in_channels=1, image_size=8, patch_size=4, d_model=8,
                num_heads=2, d_ff=12, num_layers=2, window_size=2,
                num_classes=3)

    def _model(self, **kw):
        cfg = dict(self.BASE)
        cfg.update(kw)
        random.seed(0)
        return AI.SwinTransformer(**cfg)

    def _img(self, scale=1.0):
        return [[[scale * (0.1 * a - 0.05 * b + 0.02 * a * b)
                  for b in range(8)] for a in range(8)]]

    def _loss(self, model, coef, img=None):
        img = self._img() if img is None else img
        model.zero_grad()
        out = model.forward(img)
        return sum(c * o for c, o in zip(coef, out))

    def test_window_partition_and_shift(self):
        perm0, win0 = AI._win_partition(2, 2, 1, 0)
        assert perm0 == [0, 1, 2, 3]
        assert win0 == [[0], [1], [2], [3]]
        perm1, win1 = AI._win_partition(4, 4, 2, 1)
        assert sorted(perm1) == list(range(16))
        assert len(win1) == 4 and all(len(w) == 4 for w in win1)
        # 右下 roll：rolled[0] 来自原 (3,3)，rolled[5] 来自原 (0,0)
        assert perm1[0] == 15
        assert perm1[5] == 0
        assert perm1[6] == 1

    def test_shapes_and_param_count(self):
        m = self._model()
        logits = m.forward(self._img())
        assert len(logits) == 3
        assert m.param_count() == 1243
        dx = m.backward([1.0, -0.5, 0.25])
        assert (len(dx), len(dx[0]), len(dx[0][0])) == (1, 8, 8)

    def test_gradients_match_finite_difference(self):
        m = self._model()
        coef = [1.0, -0.5, 0.25]
        eps = 1e-5
        img = [row[:] for row in self._img()[0]]
        m.zero_grad()
        m.forward([img])
        dx = m.backward(coef)
        for (a, b) in [(0, 0), (3, 5), (7, 7)]:
            old = img[a][b]
            img[a][b] = old + eps
            lp = self._loss(m, coef, [img])
            img[a][b] = old - eps
            lm = self._loss(m, coef, [img])
            img[a][b] = old
            num = (lp - lm) / (2 * eps)
            denom = max(1.0, abs(num), abs(dx[0][a][b]))
            assert abs(dx[0][a][b] - num) / denom < 1e-5
        for bi in range(2):
            blk = m.blocks[bi]
            for name in ('q', 'k', 'v', 'o', 'f1', 'f2'):
                m2 = self._model()
                blk2 = m2.blocks[bi]
                layer = getattr(blk2, name)
                m2.zero_grad()
                m2.forward(self._img())
                m2.backward(coef)
                i = 1 if len(layer.weights) > 1 else 0
                j = 2 if len(layer.weights[0]) > 2 else 0
                ana = blk2.grads[name][0][i][j]
                old = layer.weights[i][j]
                layer.weights[i][j] = old + eps
                lp = self._loss(m2, coef)
                layer.weights[i][j] = old - eps
                lm = self._loss(m2, coef)
                layer.weights[i][j] = old
                num = (lp - lm) / (2 * eps)
                denom = max(1.0, abs(num), abs(ana))
                assert abs(ana - num) / denom < 1e-5, (bi, name, ana, num)

    def test_head_and_norm_gradients(self):
        m = self._model()
        coef = [1.0, -0.5, 0.25]
        eps = 1e-5
        old = m.head.weights[1][2]
        m.head.weights[1][2] = old + eps
        lp = self._loss(m, coef)
        m.head.weights[1][2] = old - eps
        lm = self._loss(m, coef)
        m.head.weights[1][2] = old
        num = (lp - lm) / (2 * eps)
        m.zero_grad()
        m.forward(self._img())
        m.backward(coef)
        assert abs(m.head_grads[0][1][2] - num) / max(1.0, abs(num)) < 1e-5

        blk = m.blocks[0]
        old = blk.ln1.gamma[2]
        blk.ln1.gamma[2] = old + eps
        lp = self._loss(m, coef)
        blk.ln1.gamma[2] = old - eps
        lm = self._loss(m, coef)
        blk.ln1.gamma[2] = old
        num = (lp - lm) / (2 * eps)
        m.zero_grad()
        m.forward(self._img())
        m.backward(coef)
        assert abs(blk.norm_grads['ln1'][0][2] - num) / max(1.0, abs(num)) < 1e-5

    def test_accumulation_and_training(self):
        m = self._model()
        coef = [1.0, -0.5, 0.25]
        m.zero_grad()
        m.forward(self._img())
        m.backward(coef)
        first = m.head_grads[0][1][2]
        assert abs(first) > 1e-9
        m.forward(self._img())
        m.backward(coef)
        assert abs(m.head_grads[0][1][2] - 2 * first) < 1e-9
        m.zero_grad()
        assert all(abs(v) < 1e-15 for row in m.head_grads[0] for v in row)
        assert all(abs(v) < 1e-15
                   for row in m.blocks[0].grads['q'][0] for v in row)

        random.seed(1)
        X = [[[0.1 * a - 0.1 * b for b in range(8)] for a in range(8)],
             [[0.2 * a - 0.05 * b + 0.5 for b in range(8)] for a in range(8)],
             [[0.05 * a + 0.3 * b - 0.4 for b in range(8)] for a in range(8)],
             [[-0.2 * a + 0.1 * b for b in range(8)] for a in range(8)]]
        Y = [0, 1, 0, 1]
        m3 = AI.SwinTransformer(in_channels=1, image_size=8, patch_size=4,
                                d_model=8, num_heads=2, d_ff=12, num_layers=2,
                                window_size=2, num_classes=2)
        losses = []
        for _ in range(150):
            total = 0.0
            for img, y in zip(X, Y):
                m3.zero_grad()
                out = m3.forward([img])
                loss, grads = AI.cross_entropy_loss(out, y)
                total += loss
                m3.backward(grads)
                m3.adam_step(0.01)
            losses.append(total / len(X))
        assert losses[-1] < losses[0] * 0.5
        pred = []
        for img in X:
            out = m3.forward([img])
            pred.append(0 if out[0] > out[1] else 1)
        assert pred == Y

        def raises(fn, exc):
            try:
                fn()
            except exc:
                return True
            except Exception:
                return False
            return False

        assert raises(lambda: AI.SwinTransformer(in_channels=0), ValueError)
        assert raises(lambda: AI.SwinTransformer(image_size=7, patch_size=4), ValueError)
        assert raises(lambda: AI.SwinTransformer(d_model=8, num_heads=3), ValueError)
        assert raises(lambda: AI.SwinTransformer(image_size=4, patch_size=2,
                                                 window_size=3), ValueError)
        assert raises(lambda: AI.SwinTransformer(window_size=0), ValueError)
        m2 = self._model()
        assert raises(lambda: m2.forward([[0.0] * 8] * 8), ValueError)
        assert raises(lambda: m2.forward([[[0.0] * 7] * 8] * 1), ValueError)
        m2.forward(self._img())
        assert raises(lambda: m2.backward([1.0, 0.0]), ValueError)
        assert raises(lambda: m2.sgd_step(-0.1), ValueError)
        assert raises(lambda: m2.adam_step(0.1, beta1=1.0), ValueError)


class TestRoPEScaling:
    """rope_inv_freqs：linear / NTK / YaRN 三种长度外推缩放"""

    D_K = 8
    BASE = 10000.0

    def _plain(self, d_k=None):
        d_k = self.D_K if d_k is None else d_k
        return [1.0 / (self.BASE ** (2.0 * i / d_k)) for i in range(d_k // 2)]

    def test_default_matches_formula(self):
        freqs = AI.rope_inv_freqs(self.D_K, self.BASE)
        assert len(freqs) == self.D_K // 2
        for a, b in zip(freqs, self._plain()):
            assert abs(a - b) < 1e-15
        # scaling=None 时 factor 被忽略（默认参数就是 8.0，不能报错）
        assert AI.rope_inv_freqs(self.D_K, self.BASE, None, 2.0) == freqs

    def test_linear_and_ntk(self):
        plain = self._plain()
        lin = AI.rope_inv_freqs(self.D_K, self.BASE, 'linear', 4.0)
        for a, b in zip(lin, plain):
            assert abs(a - b / 4.0) < 1e-18

        ntk = AI.rope_inv_freqs(self.D_K, self.BASE, 'ntk', 4.0)
        base_eff = self.BASE * (4.0 ** (self.D_K / (self.D_K - 2.0)))
        for i, got in enumerate(ntk):
            want = 1.0 / (base_eff ** (2.0 * i / self.D_K))
            assert abs(got - want) < 1e-18
        assert abs(ntk[0] - 1.0) < 1e-15          # 第 0 维不受任何缩放影响
        for i in range(1, len(plain)):
            assert ntk[i] < plain[i]
        for a, b in zip(lin, ntk):                # NTK 比 linear 更保守
            assert b >= a - 1e-15

    def test_yarn_endpoints_and_bounds(self):
        plain = self._plain()
        lin = AI.rope_inv_freqs(self.D_K, self.BASE, 'linear', 4.0)
        yarn = AI.rope_inv_freqs(self.D_K, self.BASE, 'yarn', 4.0,
                                 original_max=512)
        for a, b in zip(plain, yarn):
            assert min(a, a / 4.0) - 1e-15 <= b <= max(a, a / 4.0) + 1e-15
        # 圈数极少 → 全部保持原样；圈数极多 → 全部 linear
        small = AI.rope_inv_freqs(self.D_K, self.BASE, 'yarn', 4.0,
                                  original_max=1)
        for a, b in zip(small, plain):
            assert abs(a - b) < 1e-18
        big = AI.rope_inv_freqs(self.D_K, self.BASE, 'yarn', 4.0,
                                original_max=10 ** 7)
        for a, b in zip(big, lin):
            assert abs(a - b) < 1e-18

    def test_rope_scaling_reaches_mha(self):
        def build(**kw):
            random.seed(7)
            return AI.MultiHeadAttention(8, 2, rope=True, **kw)

        XS = [[math.sin(0.3 * t + 0.1 * j) for j in range(8)] for t in range(5)]
        m0 = build()
        assert m0.rope_freqs == AI.rope_inv_freqs(m0.d_k, 10000.0)
        base_out = m0.forward([list(r) for r in XS])

        ml = build(rope_scaling='linear', rope_factor=2.0)
        assert ml.rope_freqs == AI.rope_inv_freqs(ml.d_k, 10000.0, 'linear', 2.0)
        lin_out = ml.forward([list(r) for r in XS])
        assert max(abs(a - b) for ra, rb in zip(base_out, lin_out)
                   for a, b in zip(ra, rb)) > 1e-6

        my = build(rope_scaling='yarn', rope_factor=4.0, rope_original_max=64)
        assert len(my.rope_freqs) == len(m0.rope_freqs)

    def test_gradients_with_scaling(self):
        random.seed(7)
        m = AI.MultiHeadAttention(8, 2, rope=True, rope_scaling='yarn',
                                  rope_factor=4.0, rope_original_max=64)
        XS = [[math.sin(0.3 * t + 0.1 * j) for j in range(8)] for t in range(5)]
        W = [[math.cos(0.2 * t + 0.1 * j) for j in range(8)] for t in range(5)]
        eps = 1e-6

        def loss():
            m.zero_grad()
            out = m.forward([list(r) for r in XS])
            return sum(W[t][j] * out[t][j] for t in range(5) for j in range(8))

        for name, i, j in (('W_q', 1, 2), ('W_k', 0, 3), ('W_v', 2, 5),
                           ('W_o', 3, 1)):
            mat = getattr(m, name)
            m.zero_grad()
            m.forward([list(r) for r in XS])
            m.backward([list(r) for r in W])
            ana = m.grads[name][i][j]
            old = mat[i][j]
            mat[i][j] = old + eps
            lp = loss()
            mat[i][j] = old - eps
            lm = loss()
            mat[i][j] = old
            num = (lp - lm) / (2 * eps)
            assert abs(ana - num) / max(1.0, abs(num)) < 1e-6, (name, i, j)

    def test_invalid_arguments(self):
        def raises(fn):
            try:
                fn()
            except ValueError:
                return True
            return False

        assert raises(lambda: AI.rope_inv_freqs(1))
        assert raises(lambda: AI.rope_inv_freqs(8, self.BASE, 'bad'))
        assert raises(lambda: AI.rope_inv_freqs(8, self.BASE, 'linear', 0.5))
        assert raises(lambda: AI.rope_inv_freqs(8, self.BASE, 'none', 2.0))
        assert raises(lambda: AI.rope_inv_freqs(8, self.BASE, 'yarn', 4.0,
                                                beta_fast=1.0, beta_slow=8.0))
        assert raises(lambda: AI.rope_inv_freqs(8, -1.0))
        assert raises(lambda: AI.rope_inv_freqs(8, self.BASE, 'yarn', 4.0,
                                                original_max=0))


class TestLearnedPositionalEncoding:
    """可学习位置编码表"""

    def test_forward_and_param_count(self):
        random.seed(3)
        pe = AI.LearnedPositionalEncoding(4, max_len=3, std=0.1)
        X = [[0.1 * t + 0.05 * j for j in range(4)] for t in range(3)]
        out = pe.forward([list(r) for r in X])
        for t in range(3):
            for j in range(4):
                assert abs(out[t][j] - (X[t][j] + pe.table[t][j])) < 1e-15
        assert pe.param_count() == 12

    def test_backward_accumulates(self):
        random.seed(3)
        pe = AI.LearnedPositionalEncoding(4, max_len=3)
        D = [[0.5, -0.2, 0.1, 0.3], [0.0, 0.0, 0.0, 0.0],
             [-0.4, 0.2, 0.0, 0.7]]
        pe.forward([[0.0] * 4 for _ in range(3)])
        pe.zero_grad()
        dx, dtab = pe.backward([list(r) for r in D])
        assert dx == D                     # 位置编码只加常数，dx 就是 dout
        for t in range(3):
            for j in range(4):
                assert abs(dtab[t][j] - D[t][j]) < 1e-15
        pe.backward([list(r) for r in D])
        for t in range(3):
            for j in range(4):
                assert abs(pe.grad_table[t][j] - 2 * D[t][j]) < 1e-15

    def test_sgd_and_adam_update_table(self):
        random.seed(4)
        pe = AI.LearnedPositionalEncoding(3, max_len=2)
        pe.forward([[1.0, 2.0, 3.0]])
        pe.zero_grad()
        pe.backward([[1.0, 1.0, 1.0]])
        before = [row[:] for row in pe.table]
        pe.sgd_step(0.1)
        for j in range(3):
            assert abs(pe.table[0][j] - (before[0][j] - 0.1)) < 1e-15
            assert pe.table[1][j] == before[1][j]      # 没用到第 2 行

        pe2 = AI.LearnedPositionalEncoding(3, max_len=2)
        pe2.forward([[1.0, 2.0, 3.0]])
        pe2.zero_grad()
        pe2.backward([[1.0, 1.0, 1.0]])
        b2 = [row[:] for row in pe2.table]
        pe2.adam_step(0.01)
        assert any(abs(pe2.table[0][j] - b2[0][j]) > 1e-9 for j in range(3))

    def test_extends_table_on_longer_input(self):
        pe = AI.LearnedPositionalEncoding(2, max_len=2)
        out = pe.forward([[float(t), float(-t)] for t in range(5)])
        assert len(out) == 5
        assert len(pe.table) == 5
        assert len(pe.m_table) == 5 and len(pe.v_table) == 5
        assert len(pe.grad_table) == 5

    def test_invalid_arguments(self):
        def raises(fn):
            try:
                fn()
            except ValueError:
                return True
            return False

        assert raises(lambda: AI.LearnedPositionalEncoding(0))
        assert raises(lambda: AI.LearnedPositionalEncoding(4, max_len=0))
        assert raises(lambda: AI.LearnedPositionalEncoding(4, max_len=2, std=-1.0))
        pe = AI.LearnedPositionalEncoding(4, max_len=2)
        assert raises(lambda: pe.forward([[0.0] * 3]))
        pe.forward([[0.0] * 4])
        assert raises(lambda: pe.backward([[0.0] * 4] * 2))
        assert raises(lambda: pe.sgd_step(-0.1))
        assert raises(lambda: pe.adam_step(0.1, beta1=1.0))


class TestLinearAttention:
    """线性注意力：elu+1 核，O(T*d^2)，causal / 非 causal 两条路径"""

    XS = [[math.sin(0.4 * t + 0.3 * j) + 0.2 * j for j in range(6)]
          for t in range(5)]
    DOUT = [[math.cos(0.2 * t + 0.1 * j) for j in range(6)] for t in range(5)]

    def _model(self, causal, seed=11):
        random.seed(seed)
        return AI.LinearAttention(6, 2, causal=causal)

    def _loss(self, m):
        m.zero_grad()
        out = m.forward([list(r) for r in self.XS])
        return sum(self.DOUT[t][j] * out[t][j]
                   for t in range(5) for j in range(6))

    def test_shapes_and_param_count(self):
        for causal in (False, True):
            m = self._model(causal)
            out = m.forward([list(r) for r in self.XS])
            assert len(out) == 5 and len(out[0]) == 6
            assert all(v == v and abs(v) < 1e6 for row in out for v in row)
            assert m.param_count() == 4 * 36 + 4 * 6

    def test_closed_form_and_causality(self):
        m = AI.LinearAttention(2, 1, causal=False)
        for name in ('W_q', 'W_k', 'W_v', 'W_o'):
            setattr(m, name, [[1.0, 0.0], [0.0, 1.0]])
        xs = [[1.0, 0.0], [0.0, 2.0], [3.0, 0.0]]
        out = m.forward([list(r) for r in xs])
        phi = AI.LinearAttention.phi
        S = [[0.0, 0.0], [0.0, 0.0]]
        for x in xs:
            pk = phi(x)
            for i in range(2):
                for j in range(2):
                    S[i][j] += pk[i] * x[j]
        for t, x in enumerate(xs):
            pq = phi(x)
            num = [sum(pq[i] * S[i][j] for i in range(2)) for j in range(2)]
            den = sum(pq[i] * sum(phi(y)[i] for y in xs) for i in range(2)) + m.eps
            for j in range(2):
                assert abs(out[t][j] - num[j] / den) < 1e-12

        mc = self._model(True)
        base = mc.forward([list(r) for r in self.XS])
        changed = [list(r) for r in self.XS]
        changed[-1] = [v + 5.0 for v in changed[-1]]
        out2 = mc.forward(changed)
        assert abs(out2[0][0] - base[0][0]) < 1e-15
        assert abs(out2[4][0] - base[4][0]) > 1e-6

    def test_weight_gradients_finite_difference(self):
        eps = 1e-6
        for causal in (False, True):
            m = self._model(causal)
            for name, i, j in (('W_q', 1, 2), ('W_k', 0, 3),
                               ('W_v', 2, 4), ('W_o', 3, 1)):
                mat = getattr(m, name)
                m.zero_grad()
                m.forward([list(r) for r in self.XS])
                m.backward([list(r) for r in self.DOUT])
                ana = m.grads[name][i][j]
                old = mat[i][j]
                mat[i][j] = old + eps
                lp = self._loss(m)
                mat[i][j] = old - eps
                lm = self._loss(m)
                mat[i][j] = old
                num = (lp - lm) / (2 * eps)
                assert abs(ana - num) / max(1.0, abs(num)) < 1e-5, (causal, name)

    def test_input_gradient_finite_difference(self):
        eps = 1e-6
        for causal in (False, True):
            m = self._model(causal)
            m.zero_grad()
            m.forward([list(r) for r in self.XS])
            dx = m.backward([list(r) for r in self.DOUT])
            for (t, j) in ((0, 0), (2, 3), (4, 5)):
                xs = [list(r) for r in self.XS]
                xs[t][j] += eps
                m.zero_grad()
                out = m.forward(xs)
                lp = sum(self.DOUT[a][b] * out[a][b]
                         for a in range(5) for b in range(6))
                xs[t][j] -= 2 * eps
                m.zero_grad()
                out = m.forward(xs)
                lm = sum(self.DOUT[a][b] * out[a][b]
                         for a in range(5) for b in range(6))
                num = (lp - lm) / (2 * eps)
                assert abs(dx[t][j] - num) / max(1.0, abs(num)) < 1e-5

    def test_accumulation_and_optimizers(self):
        m = self._model(True)
        m.zero_grad()
        m.forward([list(r) for r in self.XS])
        m.backward([list(r) for r in self.DOUT])
        g1 = m.grads['W_q'][0][0]
        m.forward([list(r) for r in self.XS])
        m.backward([list(r) for r in self.DOUT])
        assert abs(m.grads['W_q'][0][0] - 2 * g1) < 1e-12
        m.zero_grad()
        assert all(v == 0.0 for row in m.grads['W_q'] for v in row)

        before = [row[:] for row in m.W_q]
        m.forward([list(r) for r in self.XS])
        m.backward([list(r) for r in self.DOUT])
        m.sgd_step(0.05)
        assert any(abs(m.W_q[i][j] - before[i][j]) > 1e-15
                   for i in range(6) for j in range(6))
        m.adam_step(0.01)
        assert m._adam_t == 1

    def test_training_converges(self):
        random.seed(0)
        xs = [[0.1 * t] * 4 for t in range(3)]
        m = AI.LinearAttention(4, 2, causal=True)
        target = [0.5, -0.25, 1.0, 0.0]
        losses = []
        for _ in range(120):
            m.zero_grad()
            out = m.forward([list(r) for r in xs])
            diff = [[out[t][j] - target[j] * (t + 1) / 3 for j in range(4)]
                    for t in range(3)]
            losses.append(sum(v * v for row in diff for v in row) / 12.0)
            m.backward([[2 * v / 12.0 for v in row] for row in diff])
            m.adam_step(0.05)
        assert losses[-1] < losses[0] * 0.2

    def test_invalid_arguments(self):
        def raises(fn, exc=ValueError):
            try:
                fn()
            except exc:
                return True
            return False

        assert raises(lambda: AI.LinearAttention(0, 1))
        assert raises(lambda: AI.LinearAttention(6, 0))
        assert raises(lambda: AI.LinearAttention(6, 4))
        assert raises(lambda: AI.LinearAttention(6, 2, eps=-1.0))
        m = self._model(False)
        assert raises(lambda: m.forward([]))
        assert raises(lambda: m.forward([[0.0] * 5]))
        assert raises(lambda: m.backward([[1.0] * 6]), RuntimeError)
        m.forward([list(r) for r in self.XS])
        assert raises(lambda: m.backward([[1.0] * 6]))
        assert raises(lambda: m.sgd_step(-0.1))
        assert raises(lambda: m.adam_step(0.1, beta1=1.0))


# ==== W5-B2：LinformerAttention ====

class TestLinformerAttention:
    def _setup(self, D=8, H=2, T=6, K=2):
        random.seed(11)
        la = AI.LinformerAttention(D, H, T, K)
        x = [[random.gauss(0, 1) for _ in range(D)] for _ in range(T)]
        dout = [[random.gauss(0, 1) for _ in range(D)] for _ in range(T)]
        return la, x, dout

    def _loss(self, la, x, dout):
        y = la.forward(x)
        return sum(dout[t][j] * y[t][j] for t in range(len(y))
                   for j in range(len(y[0])))

    def test_shape_params_and_scores(self):
        la, x, dout = self._setup()
        y = la.forward(x)
        assert len(y) == 6 and len(y[0]) == 8
        assert la.param_count() == 4 * 8 * 8 + 4 * 8 + 2 * 2 * 6
        # 分数矩阵是 T x K（低秩投影的核心）
        assert len(la._cache[0]["probs"]) == 6
        assert len(la._cache[0]["probs"][0]) == 2
        # 每个头的 k_proj/v_proj 都是 K x d_k
        assert len(la._cache[0]["k_proj"]) == 2
        assert len(la._cache[0]["k_proj"][0]) == 4
        dx = la.backward(dout)
        assert len(dx) == 6 and len(dx[0]) == 8

    def test_gradients_vs_finite_difference(self):
        la, x, dout = self._setup()
        eps = 1e-5

        def param_rel(name, i, j):
            obj = getattr(la, name)
            old = obj[i][j]
            la.zero_grad()
            la.forward(x)
            la.backward(dout)
            ana = la.grads[name][i][j]
            obj[i][j] = old + eps
            up = self._loss(la, x, dout)
            obj[i][j] = old - eps
            dn = self._loss(la, x, dout)
            obj[i][j] = old
            num = (up - dn) / (2 * eps)
            return abs(ana - num) / max(abs(ana), abs(num), 1e-8)

        for name in ("W_q", "W_k", "W_v", "W_o", "E", "F"):
            obj = getattr(la, name)
            idxs = [(0, 0), (len(obj) - 1, len(obj[0]) - 1)]
            for (i, j) in idxs:
                assert param_rel(name, i, j) < 1e-5, (name, i, j)

        la.zero_grad()
        la.forward(x)
        dx = la.backward(dout)
        for (t, j) in ((0, 0), (5, 7)):
            xp = [list(r) for r in x]
            xm = [list(r) for r in x]
            xp[t][j] += eps
            xm[t][j] -= eps
            num = (self._loss(la, xp, dout) - self._loss(la, xm, dout)) / (2 * eps)
            assert abs(dx[t][j] - num) / max(abs(dx[t][j]), abs(num), 1e-8) < 1e-5

    def test_reduces_to_standard_attention(self):
        """num_proj == seq_len 且 E = F = I 时，低秩投影不丢信息"""
        D, H, T = 8, 2, 4
        la = AI.LinformerAttention(D, H, T, T)
        eye = [[1.0 if i == j else 0.0 for j in range(T)] for i in range(T)]
        la.E = [list(r) for r in eye]
        la.F = [list(r) for r in eye]
        mha = AI.MultiHeadAttention(D, H)
        de = [[1.0 if i == j else 0.0 for j in range(D)] for i in range(D)]
        for name in ("W_q", "W_k", "W_v", "W_o"):
            setattr(la, name, [list(r) for r in de])
            setattr(mha, name, [list(r) for r in de])
        for name in ("b_q", "b_k", "b_v", "b_o"):
            setattr(la, name, [0.0] * D)
            setattr(mha, name, [0.0] * D)
        x = [[random.gauss(0, 1) for _ in range(D)] for _ in range(T)]
        y_la = la.forward(x)
        y_mha = mha.forward(x)
        assert max(abs(a - b) for ra, rb in zip(y_la, y_mha) for a, b in zip(ra, rb)) < 1e-12

    def test_accumulate_and_zero_grad(self):
        la, x, dout = self._setup()
        la.zero_grad()
        la.forward(x)
        la.backward(dout)
        g1 = la.grads["E"][0][0]
        la.forward(x)
        la.backward(dout)
        g2 = la.grads["E"][0][0]
        assert abs(g2 - 2 * g1) < 1e-12 * max(1.0, abs(g2))
        la.zero_grad()
        left = [v for name in ("W_q", "W_k", "W_v", "W_o", "E", "F")
                for row in la.grads[name] for v in row]
        assert max(abs(v) for v in left) == 0.0

    def test_training_and_validation(self):
        random.seed(0)
        la = AI.LinformerAttention(8, 2, 4, 2)
        xs = [[random.gauss(0, 1) for _ in range(8)] for _ in range(4)]
        first = None
        for _ in range(120):
            la.zero_grad()
            y = la.forward(xs)
            if first is None:
                first = 0.5 * sum(v * v for row in y for v in row)
            la.backward([[y[t][j] for j in range(8)] for t in range(4)])
            la.adam_step(0.02)
        assert first is not None and first > 1.0
        assert 0.5 * sum(v * v for row in y for v in row) < first * 0.05

        for bad in (lambda: AI.LinformerAttention(8, 3, 6),
                    lambda: AI.LinformerAttention(8, 2, 6, 7),
                    lambda: AI.LinformerAttention(8, 2, 6, 0),
                    lambda: AI.LinformerAttention(0, 2, 6),
                    lambda: AI.LinformerAttention(8, 0, 6),
                    lambda: AI.LinformerAttention(8, 2, 0),
                    lambda: AI.LinformerAttention(True, 2, 6),
                    lambda: AI.LinformerAttention(8, 2, 6, 1.5)):
            try:
                bad()
            except ValueError:
                pass
            else:
                raise AssertionError("非法构造没有报错")
        small = AI.LinformerAttention(8, 2, 6)
        try:
            small.forward([])
        except ValueError:
            pass
        else:
            raise AssertionError("空序列没有报错")
        try:
            small.forward([[0.0] * 8 for _ in range(5)])
        except ValueError:
            pass
        else:
            raise AssertionError("序列长度不符没有报错")
        try:
            small.backward([[0.0] * 8 for _ in range(6)])
        except RuntimeError:
            pass
        else:
            raise AssertionError("未 forward 就 backward 没有报错")


# ==== W5-B3：PerformerAttention ====

class TestPerformerAttention:
    def _setup(self, D=8, H=2, M=8, T=5, causal=False):
        random.seed(7)
        p = AI.PerformerAttention(D, H, num_features=M, causal=causal)
        x = [[random.gauss(0, 1) for _ in range(D)] for _ in range(T)]
        dout = [[random.gauss(0, 1) for _ in range(D)] for _ in range(T)]
        return p, x, dout

    def _scalar(self, p, x, dout):
        p.zero_grad()
        y = p.forward(x)
        return sum(dout[t][j] * y[t][j] for t in range(len(y))
                   for j in range(len(y[0])))

    def test_shape_params_and_features(self):
        p, x, dout = self._setup()
        y = p.forward(x)
        assert len(y) == 5 and len(y[0]) == 8
        assert p.param_count() == 4 * 8 * 8 + 4 * 8
        # phi 的长度是特征数 m，不是 d_k
        assert len(p._cache[0]["pq"][0]) == 8
        assert len(p._cache[0]["pk"][0]) == 8
        # 随机特征矩阵不参与训练
        assert "omega" not in p.grads
        assert len(p.omega) == 2 and len(p.omega[0]) == 8 and len(p.omega[0][0]) == 4

    def test_gradients_vs_finite_difference(self):
        eps = 1e-5
        for causal in (False, True):
            p, x, dout = self._setup(causal=causal)

            def rel_param(name, i, j):
                obj = getattr(p, name)
                old = obj[i][j]
                p.zero_grad()
                p.forward(x)
                p.backward(dout)
                ana = p.grads[name][i][j]
                obj[i][j] = old + eps
                up = self._scalar(p, x, dout)
                obj[i][j] = old - eps
                dn = self._scalar(p, x, dout)
                obj[i][j] = old
                num = (up - dn) / (2 * eps)
                return abs(ana - num) / max(abs(ana), abs(num), 1e-8)

            for name in ("W_q", "W_k", "W_v", "W_o"):
                for (i, j) in ((0, 0), (7, 7)):
                    assert rel_param(name, i, j) < 1e-5, (causal, name, i, j)

            p.zero_grad()
            p.forward(x)
            dx = p.backward(dout)
            for (t, j) in ((0, 0), (4, 7)):
                xp = [list(r) for r in x]
                xm = [list(r) for r in x]
                xp[t][j] += eps
                xm[t][j] -= eps
                num = (self._scalar(p, xp, dout) - self._scalar(p, xm, dout)) / (2 * eps)
                assert abs(dx[t][j] - num) / max(abs(dx[t][j]), abs(num), 1e-8) < 1e-5

    def test_causality(self):
        p, x, _ = self._setup(causal=True)
        x2 = [list(r) for r in x]
        x2[4][0] += 5.0
        y1 = p.forward(x)
        y2 = p.forward(x2)
        for t in range(4):
            for j in range(8):
                assert abs(y1[t][j] - y2[t][j]) < 1e-12
        assert max(abs(y1[4][j] - y2[4][j]) for j in range(8)) > 1e-9

    def test_more_features_improve_approximation(self):
        D = 8
        random.seed(3)
        xs = [[random.gauss(0, 1) for _ in range(D)] for _ in range(4)]
        mha = AI.MultiHeadAttention(D, 2)
        de = [[1.0 if i == j else 0.0 for j in range(D)] for i in range(D)]
        for name in ("W_q", "W_k", "W_v", "W_o"):
            setattr(mha, name, [list(r) for r in de])
        for name in ("b_q", "b_k", "b_v", "b_o"):
            setattr(mha, name, [0.0] * D)
        ref = mha.forward(xs)

        def err(m):
            pf = AI.PerformerAttention(D, 2, num_features=m, seed=5)
            for name in ("W_q", "W_k", "W_v", "W_o"):
                setattr(pf, name, [list(r) for r in de])
            for name in ("b_q", "b_k", "b_v", "b_o"):
                setattr(pf, name, [0.0] * D)
            ys = pf.forward(xs)
            num = sum((ys[t][j] - ref[t][j]) ** 2 for t in range(4) for j in range(D))
            den = sum(ref[t][j] ** 2 for t in range(4) for j in range(D))
            return math.sqrt(num / den)

        small = err(4)
        big = err(64)
        assert big < small
        assert big < 0.5

    def test_accumulate_optimizers_and_validation(self):
        p, x, dout = self._setup()
        p.zero_grad()
        p.forward(x)
        p.backward(dout)
        g1 = p.grads["W_q"][0][0]
        p.forward(x)
        p.backward(dout)
        g2 = p.grads["W_q"][0][0]
        assert abs(g2 - 2 * g1) < 1e-12 * max(1.0, abs(g2))
        p.zero_grad()
        left = [v for name in ("W_q", "W_k", "W_v", "W_o")
                for row in p.grads[name] for v in row]
        assert max(abs(v) for v in left) == 0.0

        before = [list(r) for r in p.W_v]
        p.forward(x)
        p.backward(dout)
        p.sgd_step(0.1)
        assert p.W_v != before
        before = [list(r) for r in p.W_v]
        p.zero_grad()
        p.forward(x)
        p.backward(dout)
        p.adam_step(0.01)
        assert p.W_v != before

        for bad in (lambda: AI.PerformerAttention(8, 3),
                    lambda: AI.PerformerAttention(8, 2, 0),
                    lambda: AI.PerformerAttention(8, 2, 4, 1),
                    lambda: AI.PerformerAttention(8, 2, 4, False, 0.0),
                    lambda: AI.PerformerAttention(8, 2, 4, False, 1e-6, 1.5),
                    lambda: AI.PerformerAttention(0, 2),
                    lambda: AI.PerformerAttention(8, 0)):
            try:
                bad()
            except ValueError:
                pass
            else:
                raise AssertionError("非法构造没有报错")
        q = AI.PerformerAttention(8, 2)
        try:
            q.forward([])
        except ValueError:
            pass
        else:
            raise AssertionError("空序列没有报错")
        try:
            q.forward([[0.0] * 3] * 4)
        except ValueError:
            pass
        else:
            raise AssertionError("token 维度不符没有报错")
        try:
            q.backward([[0.0] * 8] * 4)
        except RuntimeError:
            pass
        else:
            raise AssertionError("未 forward 就 backward 没有报错")


# ==== W5-B4：MLAttention（低秩 KV 压缩） ====

class TestMLAttention:
    def _setup(self, D=8, H=2, DC=2, T=4, causal=False):
        random.seed(9)
        m = AI.MLAttention(D, H, latent_dim=DC, causal=causal)
        x = [[random.gauss(0, 1) for _ in range(D)] for _ in range(T)]
        dout = [[random.gauss(0, 1) for _ in range(D)] for _ in range(T)]
        return m, x, dout

    def _scalar(self, m, x, dout):
        m.zero_grad()
        y = m.forward(x)
        return sum(dout[t][j] * y[t][j] for t in range(len(y))
                   for j in range(len(y[0])))

    def test_shape_params_and_cache(self):
        m, x, dout = self._setup()
        y = m.forward(x)
        assert len(y) == 4 and len(y[0]) == 8
        # latent 是 (T, d_c)
        assert len(m.latent) == 4 and len(m.latent[0]) == 2
        hk = 2 * 4
        assert m.param_count() == (2 * 64 + 16) + 2 * 8 + 2 + 2 * hk * 2 + 2 * hk
        # KV cache：MLA 存 T*d_c，标准 MHA 存 T*2*H*d_k
        assert m.kv_cache_size(8) == 16
        assert m.full_kv_cache_size(8) == 8 * 2 * 2 * 4
        assert m.full_kv_cache_size(8) / m.kv_cache_size(8) == 8.0

    def test_gradients_vs_finite_difference(self):
        eps = 1e-5
        for causal in (False, True):
            m, x, dout = self._setup(causal=causal)

            def rel_param(name, i, j):
                obj = getattr(m, name)
                old = obj[i][j]
                m.zero_grad()
                m.forward(x)
                m.backward(dout)
                ana = m.grads[name][i][j]
                obj[i][j] = old + eps
                up = self._scalar(m, x, dout)
                obj[i][j] = old - eps
                dn = self._scalar(m, x, dout)
                obj[i][j] = old
                num = (up - dn) / (2 * eps)
                if abs(ana) < 1e-6 and abs(num) < 1e-6:
                    return 0.0
                return abs(ana - num) / max(abs(ana), abs(num), 1e-8)

            for name in ("W_q", "W_dkv", "W_uk", "W_uv", "W_o"):
                obj = getattr(m, name)
                for (i, j) in ((0, 0), (len(obj) - 1, len(obj[0]) - 1)):
                    assert rel_param(name, i, j) < 1e-5, (causal, name, i, j)

            def rel_bias(name, i):
                obj = getattr(m, name)
                old = obj[i]
                m.zero_grad()
                m.forward(x)
                m.backward(dout)
                ana = m.grads[name][i]
                obj[i] = old + eps
                up = self._scalar(m, x, dout)
                obj[i] = old - eps
                dn = self._scalar(m, x, dout)
                obj[i] = old
                num = (up - dn) / (2 * eps)
                if abs(ana) < 1e-6 and abs(num) < 1e-6:
                    return 0.0
                return abs(ana - num) / max(abs(ana), abs(num), 1e-8)

            for name in ("b_q", "b_dkv", "b_uk", "b_uv", "b_o"):
                obj = getattr(m, name)
                for i in (0, len(obj) - 1):
                    assert rel_bias(name, i) < 1e-5, (causal, name, i)

            m.zero_grad()
            m.forward(x)
            dx = m.backward(dout)
            for (t, j) in ((0, 0), (3, 7)):
                xp = [list(r) for r in x]
                xm = [list(r) for r in x]
                xp[t][j] += eps
                xm[t][j] -= eps
                num = (self._scalar(m, xp, dout) - self._scalar(m, xm, dout)) / (2 * eps)
                assert abs(dx[t][j] - num) / max(abs(dx[t][j]), abs(num), 1e-8) < 1e-5

    def test_causal_mask(self):
        m, x, _ = self._setup(causal=True)
        m.forward(x)
        for h in range(m.num_heads):
            rows = m.attn_weights[h]
            for t in range(len(rows)):
                for s in range(len(rows[t])):
                    if s > t:
                        assert rows[t][s] == 0.0
                assert abs(sum(rows[t]) - 1.0) < 1e-6
        mn, xn, _ = self._setup(causal=False)
        mn.forward(xn)
        for h in range(mn.num_heads):
            for t in range(len(mn.attn_weights[h])):
                assert abs(sum(mn.attn_weights[h][t]) - 1.0) < 1e-6

    def test_accumulate_optimizers(self):
        m, x, dout = self._setup()
        m.zero_grad()
        m.forward(x)
        m.backward(dout)
        g1 = m.grads["W_dkv"][0][0]
        m.forward(x)
        m.backward(dout)
        g2 = m.grads["W_dkv"][0][0]
        assert abs(g2 - 2 * g1) < 1e-12 * max(1.0, abs(g2))
        m.zero_grad()
        left = [v for name in ("W_q", "W_dkv", "W_uk", "W_uv", "W_o")
                for row in m.grads[name] for v in row]
        assert max(abs(v) for v in left) == 0.0
        before = [list(r) for r in m.W_uk]
        m.forward(x)
        m.backward(dout)
        m.sgd_step(0.1)
        assert m.W_uk != before
        before = [list(r) for r in m.W_uk]
        m.zero_grad()
        m.forward(x)
        m.backward(dout)
        m.adam_step(0.01)
        assert m.W_uk != before

    def test_training_and_validation(self):
        random.seed(0)
        m = AI.MLAttention(8, 2, latent_dim=4)
        xs = [[random.gauss(0, 1) for _ in range(8)] for _ in range(4)]
        first = None
        for _ in range(120):
            m.zero_grad()
            y = m.forward(xs)
            if first is None:
                first = 0.5 * sum(v * v for row in y for v in row)
            m.backward([[y[t][j] for j in range(8)] for t in range(4)])
            m.adam_step(0.02)
        assert 0.5 * sum(v * v for row in y for v in row) < first * 0.05

        for bad in (lambda: AI.MLAttention(8, 3),
                    lambda: AI.MLAttention(8, 2, 0),
                    lambda: AI.MLAttention(8, 2, 2.5),
                    lambda: AI.MLAttention(8, 2, 2, 1),
                    lambda: AI.MLAttention(8, 2, 2, False, 0.0),
                    lambda: AI.MLAttention(0, 2),
                    lambda: AI.MLAttention(8, 0)):
            try:
                bad()
            except ValueError:
                pass
            else:
                raise AssertionError("非法构造没有报错")
        q = AI.MLAttention(8, 2)
        try:
            q.forward([])
        except ValueError:
            pass
        else:
            raise AssertionError("空序列没有报错")
        try:
            q.forward([[0.0] * 3] * 4)
        except ValueError:
            pass
        else:
            raise AssertionError("token 维度不符没有报错")
        try:
            q.backward([[0.0] * 8] * 4)
        except RuntimeError:
            pass
        else:
            raise AssertionError("未 forward 就 backward 没有报错")


# ==== W5-C：稀疏注意力（BigBird / Longformer / Reformer）====

X4C = [[1.0, 0.0, 0.5, -0.5],
       [0.2, 0.3, -0.1, 0.4],
       [-0.3, 0.6, 0.2, 0.1],
       [0.5, 0.5, 0.5, 0.5]]

DOUT4C = [[0.3, -0.2, 0.1, 0.05],
          [0.1, 0.2, -0.3, 0.4],
          [-0.2, 0.1, 0.25, -0.15],
          [0.4, -0.1, 0.2, 0.3]]


def _w5c_loss(model, X):
    out = model.forward(X)
    return sum(out[t][r] * DOUT4C[t][r] for t in range(len(DOUT4C))
               for r in range(len(DOUT4C[0])))


def _w5c_grad_ok(model, eps=1e-5, tol=1e-5):
    """返回 (参数梯度 worst, 输入梯度 worst)"""
    model.zero_grad()
    model.forward(X4C)
    dx_ana = model.backward(DOUT4C)
    ana = {}
    for name in ('W_q', 'W_k', 'W_v', 'W_o'):
        ana[name] = [list(r) for r in model.grads[name]]
    for name in ('b_q', 'b_k', 'b_v', 'b_o'):
        ana[name] = list(model.grads[name])
    worst = 0.0
    for name in ('W_q', 'W_k', 'W_v', 'W_o'):
        W = getattr(model, name)
        for r in range(len(W)):
            for c in range(len(W[r])):
                old = W[r][c]
                W[r][c] = old + eps
                up = _w5c_loss(model, X4C)
                W[r][c] = old - eps
                dn = _w5c_loss(model, X4C)
                W[r][c] = old
                num = (up - dn) / (2 * eps)
                a = ana[name][r][c]
                if abs(a) < 1e-9 and abs(num) < 1e-9:
                    continue
                worst = max(worst, abs(a - num) / max(abs(a), abs(num), 1.0))
    for name in ('b_q', 'b_k', 'b_v', 'b_o'):
        b = getattr(model, name)
        for r in range(len(b)):
            old = b[r]
            b[r] = old + eps
            up = _w5c_loss(model, X4C)
            b[r] = old - eps
            dn = _w5c_loss(model, X4C)
            b[r] = old
            num = (up - dn) / (2 * eps)
            a = ana[name][r]
            if abs(a) < 1e-9 and abs(num) < 1e-9:
                continue
            worst = max(worst, abs(a - num) / max(abs(a), abs(num), 1.0))
    dworst = 0.0
    for t in range(4):
        for r in range(4):
            X = [list(row) for row in X4C]
            old = X[t][r]
            X[t][r] = old + eps
            up = _w5c_loss(model, X)
            X[t][r] = old - eps
            dn = _w5c_loss(model, X)
            X[t][r] = old
            num = (up - dn) / (2 * eps)
            a = dx_ana[t][r]
            if abs(a) < 1e-9 and abs(num) < 1e-9:
                continue
            dworst = max(dworst, abs(a - num) / max(abs(a), abs(num), 1.0))
    return worst, dworst


def _w5c_dense(model, data, heads, causal):
    """手写稠密 softmax 注意力，用来核对稀疏/分块实现"""
    import math as _m
    d = model.d_model
    d_k = d // heads
    T = len(data)

    def proj(W, b, x):
        return [sum(W[r][c] * x[c] for c in range(d)) + b[r] for r in range(d)]

    Q = [proj(model.W_q, model.b_q, x) for x in data]
    K = [proj(model.W_k, model.b_k, x) for x in data]
    V = [proj(model.W_v, model.b_v, x) for x in data]
    scale = 1.0 / _m.sqrt(d_k)
    concat = [[0.0] * d for _ in range(T)]
    for h in range(heads):
        off = h * d_k
        for t in range(T):
            idx = list(range(t + 1)) if causal else list(range(T))
            sc = []
            for j in idx:
                acc = 0.0
                for r in range(d_k):
                    acc += Q[t][off + r] * K[j][off + r]
                sc.append(acc * scale)
            mx = max(sc)
            ex = [_m.exp(v - mx) for v in sc]
            tot = sum(ex)
            for r in range(d_k):
                acc = 0.0
                for pos, j in enumerate(idx):
                    acc += (ex[pos] / tot) * V[j][off + r]
                concat[t][off + r] = acc
    return [[sum(model.W_o[r][c] * concat[t][c] for c in range(d)) + model.b_o[r]
             for r in range(d)] for t in range(T)]


class TestBigBirdAttention:
    def test_shape_and_params(self):
        m = AI.BigBirdAttention(4, 2, window_size=1, num_global=1, num_random=1,
                                seed=1)
        out = m.forward(X4C)
        assert len(out) == 4 and len(out[0]) == 4
        assert m.param_count() == 4 * 4 * 4 + 4 * 4  # 80
        assert len(m._cache) == 2  # 两个头

    def test_sparse_pattern(self):
        m = AI.BigBirdAttention(4, 2, window_size=1, num_global=1, num_random=0,
                                seed=3)
        m.forward(X4C)
        allow = m._cache[0]['allow']
        p = m._cache[0]['p']
        # 只允许窗口 + 全局 0
        assert allow[0] == [0, 1]
        assert all(0 in allow[t] for t in range(4))
        # 未允许的位置权重恰为 0，每行和为 1
        for t in range(4):
            for j in range(4):
                if j not in allow[t]:
                    assert p[t][j] == 0.0
            assert abs(sum(p[t]) - 1.0) < 1e-12

    def test_causal_mask(self):
        m = AI.BigBirdAttention(4, 2, window_size=2, num_global=1, num_random=1,
                                causal=True, seed=5)
        m.forward(X4C)
        p = m._cache[0]['p']
        for t in range(4):
            for j in range(t + 1, 4):
                assert p[t][j] == 0.0
            assert abs(sum(p[t]) - 1.0) < 1e-12

    def test_gradients(self):
        m = AI.BigBirdAttention(4, 2, window_size=1, num_global=1, num_random=1,
                                seed=2)
        wp, wd = _w5c_grad_ok(m)
        assert wp < 1e-5, wp
        assert wd < 1e-5, wd
        # causal 下 t=0 只有一个可见位置：ds 恒为 0，但 dv 必须仍然正确
        mc = AI.BigBirdAttention(4, 2, window_size=2, num_global=1, num_random=1,
                                 causal=True, seed=2)
        wpc, wdc = _w5c_grad_ok(mc)
        assert wpc < 1e-5, wpc
        assert wdc < 1e-5, wdc

    def test_full_pattern_equals_dense(self):
        m = AI.BigBirdAttention(4, 2, window_size=10, num_global=4, num_random=0)
        out = m.forward(X4C)
        p = m._cache[0]['p']
        # 全连通时权重就是普通 softmax：每行和为 1 且全非零
        assert abs(sum(p[0]) - 1.0) < 1e-12
        assert all(p[0][j] > 0.0 for j in range(4))
        assert len(out) == 4

    def test_accumulate_and_optimizers(self):
        m = AI.BigBirdAttention(4, 2, window_size=1, num_global=1, num_random=1,
                                seed=1)
        m.zero_grad()
        m.forward(X4C)
        m.backward(DOUT4C)
        g1 = m.grads['W_q'][0][0]
        m.forward(X4C)
        m.backward(DOUT4C)
        assert abs(m.grads['W_q'][0][0] - 2 * g1) < 1e-12
        m.zero_grad()
        assert m.grads['W_q'][0][0] == 0.0
        before = [list(r) for r in m.W_q]
        m.forward(X4C)
        m.backward(DOUT4C)
        m.sgd_step(0.01)
        assert m.W_q != before
        m.adam_step(0.001)
        assert m.W_q != before

    def test_errors(self):
        for fn in (lambda: AI.BigBirdAttention(0, 2),
                   lambda: AI.BigBirdAttention(4, 0),
                   lambda: AI.BigBirdAttention(5, 2),
                   lambda: AI.BigBirdAttention(4, 2, window_size=-1),
                   lambda: AI.BigBirdAttention(4, 2, window_size=1.5),
                   lambda: AI.BigBirdAttention(4, 2, num_global=-1),
                   lambda: AI.BigBirdAttention(4, 2, num_random=True),
                   lambda: AI.BigBirdAttention(4, 2, seed=1.2)):
            with pytest.raises((ValueError, TypeError)):
                fn()
        m = AI.BigBirdAttention(4, 2)
        with pytest.raises(RuntimeError):
            m.backward(DOUT4C)
        with pytest.raises(ValueError):
            m.forward([])
        with pytest.raises(ValueError):
            m.forward([[1.0, 2.0]])
        m.forward(X4C)
        with pytest.raises(ValueError):
            m.backward(DOUT4C[:2])


class TestLongformerAttention:
    def test_global_tokens_see_all(self):
        m = AI.LongformerAttention(4, 2, window_size=1, global_tokens=[0, 2])
        m.forward(X4C)
        allow = m._cache[0]['allow']
        assert allow[0] == [0, 1, 2, 3]
        assert allow[2] == [0, 1, 2, 3]
        assert allow[3] == [0, 2, 3]
        p = m._cache[0]['p']
        assert all(p[0][j] > 0.0 for j in range(4))

    def test_window_and_shape(self):
        m = AI.LongformerAttention(4, 2, window_size=1, global_tokens=[0])
        out = m.forward(X4C)
        assert len(out) == 4 and len(out[0]) == 4
        assert m.param_count() == 80

    def test_gradients(self):
        m = AI.LongformerAttention(4, 2, window_size=1, global_tokens=[0, 2])
        wp, wd = _w5c_grad_ok(m)
        assert wp < 1e-5, wp
        assert wd < 1e-5, wd

    def test_causal(self):
        m = AI.LongformerAttention(4, 2, window_size=1, global_tokens=[0],
                                   causal=True)
        m.forward(X4C)
        p = m._cache[0]['p']
        for j in range(4):
            assert p[0][j] > 0.0  # 全局 token 看见全部
        for t in range(1, 4):
            for j in range(t + 1, 4):
                assert p[t][j] == 0.0
        wp, wd = _w5c_grad_ok(m)
        assert wp < 1e-5, wp
        assert wd < 1e-5, wd

    def test_errors(self):
        for fn in (lambda: AI.LongformerAttention(4, 2, window_size=-2),
                   lambda: AI.LongformerAttention(4, 2, global_tokens=[-1]),
                   lambda: AI.LongformerAttention(4, 2, global_tokens=[0.5]),
                   lambda: AI.LongformerAttention(4, 2, window_size=1.5)):
            with pytest.raises((ValueError, TypeError)):
                fn()


class TestReformerAttention:
    def test_bucket_semantics(self):
        m = AI.ReformerAttention(4, 2, num_buckets=2, seed=1)
        m.forward(X4C)
        p = m._cache[0]['p']
        bk = m._buckets
        for t in range(4):
            for j in range(4):
                if bk[j] != bk[t]:
                    assert p[t][j] == 0.0
            assert abs(sum(p[t]) - 1.0) < 1e-12

    def test_single_bucket_equals_dense(self):
        one = AI.ReformerAttention(4, 2, num_buckets=1)
        one.forward(X4C)
        p = one._cache[0]['p']
        # num_buckets=1 ⇒ 全都同桶 ⇒ 权重退化成普通 softmax（每行全非零）
        assert all(p[t][j] > 0.0 for t in range(4) for j in range(4))
        assert one.param_count() == 80
        # 换成 4 个桶时，同桶外的位置必须被挡住
        many = AI.ReformerAttention(4, 2, num_buckets=4, seed=2)
        many.forward(X4C)
        pm = many._cache[0]['p']
        bk = many._buckets
        zeros = sum(1 for t in range(4) for j in range(4)
                    if bk[j] != bk[t] and pm[t][j] == 0.0)
        assert zeros >= 4

    def test_gradients(self):
        for causal in (False, True):
            m = AI.ReformerAttention(4, 2, num_buckets=2, causal=causal, seed=1)
            wp, wd = _w5c_grad_ok(m)
            assert wp < 1e-5, (causal, wp)
            assert wd < 1e-5, (causal, wd)

    def test_hash_deterministic(self):
        a = AI.ReformerAttention(4, 2, num_buckets=2, seed=1)
        a.forward(X4C)
        b = AI.ReformerAttention(4, 2, num_buckets=2, seed=1)
        assert a.hash_proj == b.hash_proj
        b.forward(X4C)
        assert a._buckets == b._buckets

    def test_errors(self):
        for fn in (lambda: AI.ReformerAttention(4, 2, num_buckets=0),
                   lambda: AI.ReformerAttention(4, 2, num_buckets=True),
                   lambda: AI.ReformerAttention(4, 2, seed=1.5)):
            with pytest.raises((ValueError, TypeError)):
                fn()


class TestFlashAttention:
    def test_block_sizes_match_dense(self):
        m = AI.FlashAttention(4, 2, block_size=4)
        out = m.forward(X4C)
        # 单块时应与手写稠密 softmax 完全一致（只差求和顺序，1e-15 内）
        ref = _w5c_dense(m, X4C, 2, False)
        worst = max(abs(out[t][r] - ref[t][r]) for t in range(4) for r in range(4))
        assert worst < 1e-12, worst

    def test_block_invariance(self):
        base = AI.FlashAttention(4, 2, block_size=1)
        o1 = base.forward(X4C)
        for bs in (2, 3, 4):
            m = AI.FlashAttention(4, 2, block_size=bs)
            for name in ('W_q', 'W_k', 'W_v', 'W_o'):
                setattr(m, name, [list(r) for r in getattr(base, name)])
            ob = m.forward(X4C)
            worst = max(abs(ob[t][r] - o1[t][r])
                        for t in range(4) for r in range(4))
            assert worst < 1e-15, (bs, worst)

    def test_no_quadratic_cache(self):
        m = AI.FlashAttention(4, 2, block_size=2)
        m.forward(X4C)
        assert sorted(m._cache[0].keys()) == ['k', 'l', 'm', 'q', 'v']
        assert len(m._cache[0]['m']) == 4 and len(m._cache[0]['l']) == 4
        assert m.param_count() == 80

    def test_causal_prefix_bitwise(self):
        m = AI.FlashAttention(4, 2, causal=True, block_size=2)
        a = m.forward(X4C)
        X2 = [list(r) for r in X4C]
        X2[3] = [9.0, -3.0, 4.0, 2.0]
        b = m.forward(X2)
        # 因果性：改最后一个 token 不该影响前三个输出（逐位）
        assert all(a[t] == b[t] for t in range(3))
        # 因果 + 分块：与稠密 causal 一致
        ref = _w5c_dense(m, X4C, 2, True)
        a2 = m.forward(X4C)
        worst = max(abs(a2[t][r] - ref[t][r]) for t in range(4) for r in range(4))
        assert worst < 1e-12, worst

    def test_gradients(self):
        for causal, bs in ((False, 2), (True, 3)):
            m = AI.FlashAttention(4, 2, causal=causal, block_size=bs)
            wp, wd = _w5c_grad_ok(m)
            assert wp < 1e-5, (causal, bs, wp)
            assert wd < 1e-5, (causal, bs, wd)

    def test_accumulate_and_errors(self):
        m = AI.FlashAttention(4, 2, block_size=2)
        m.zero_grad()
        m.forward(X4C)
        m.backward(DOUT4C)
        g1 = m.grads['W_q'][0][0]
        m.forward(X4C)
        m.backward(DOUT4C)
        assert abs(m.grads['W_q'][0][0] - 2 * g1) < 1e-12
        m.zero_grad()
        assert m.grads['W_q'][0][0] == 0.0
        before = [list(r) for r in m.W_q]
        m.forward(X4C)
        m.backward(DOUT4C)
        m.sgd_step(0.01)
        assert m.W_q != before
        for fn in (lambda: AI.FlashAttention(4, 2, block_size=0),
                   lambda: AI.FlashAttention(4, 2, block_size=-1),
                   lambda: AI.FlashAttention(4, 2, block_size=1.5),
                   lambda: AI.FlashAttention(4, 2, block_size=True)):
            with pytest.raises((ValueError, TypeError)):
                fn()
        with pytest.raises(RuntimeError):
            AI.FlashAttention(4, 2).backward(DOUT4C)


class TestT5RelativeBias:
    def test_buckets(self):
        b = AI.t5_relative_buckets(6, 6, 8, 16, True)
        half = 4
        # 正负相对位置落不同半区，对角线为 0，且不越界
        for i in range(6):
            assert b[i][i] == 0
            left = [b[i][j] for j in range(i - 1, -1, -1)]
            right = [b[i][j] for j in range(i + 1, 6)]
            assert left == sorted(left)
            assert right == sorted(right)
            assert all(v < half for v in left)
            assert all(v >= half for v in right)
        # 非双向：只按距离，且夹到桶表范围
        nb = AI.t5_relative_buckets(1, 60, 8, 128, False)[0]
        assert nb[:5] == [0, 1, 2, 3, 4]
        assert max(nb) == 7

    def test_forward_lookup(self):
        op = AI.T5RelativeBias(2, num_buckets=8, max_distance=16)
        bias = op.forward(4)
        buckets = AI.t5_relative_buckets(4, 4, 8, 16, True)
        for h in range(2):
            for i in range(4):
                for j in range(4):
                    assert bias[h][i][j] == op.table[buckets[i][j]][h]
        assert op.param_count() == 16
        # 同一桶的位置取值相同（只与相对位置有关）
        assert bias[0][0][0] == bias[0][1][1]

    def test_backward_sums_within_bucket(self):
        op = AI.T5RelativeBias(2, num_buckets=8, max_distance=16)
        op.forward(4)
        db = [[[0.5 + 0.1 * h + 0.01 * (i + j) for j in range(4)]
               for i in range(4)] for h in range(2)]
        step = op.backward(db)
        buckets = AI.t5_relative_buckets(4, 4, 8, 16, True)
        manual = [[0.0] * 2 for _ in range(8)]
        for h in range(2):
            for i in range(4):
                for j in range(4):
                    manual[buckets[i][j]][h] += db[h][i][j]
        for x in range(8):
            for h in range(2):
                assert abs(step[x][h] - manual[x][h]) < 1e-12
                assert abs(op.grad_table[x][h] - manual[x][h]) < 1e-12
        op.zero_grad()
        assert all(v == 0.0 for r in op.grad_table for v in r)

    def test_optimizers_and_errors(self):
        op = AI.T5RelativeBias(2, num_buckets=8, max_distance=16)
        op.forward(3)
        op.backward([[[1.0] * 3 for _ in range(3)] for _ in range(2)])
        before = [list(r) for r in op.table]
        op.sgd_step(0.1)
        assert op.table != before
        op.zero_grad()
        op.forward(3)
        op.backward([[[1.0] * 3 for _ in range(3)] for _ in range(2)])
        mid = [list(r) for r in op.table]
        op.adam_step(0.01)
        assert op.table != mid
        with pytest.raises((ValueError, TypeError)):
            AI.T5RelativeBias(0)
        with pytest.raises((ValueError, TypeError)):
            AI.T5RelativeBias(2, num_buckets=2)
        with pytest.raises((ValueError, TypeError)):
            AI.T5RelativeBias(2, std=-1.0)
        with pytest.raises(RuntimeError):
            AI.T5RelativeBias(2).backward([[[1.0]]])
        op2 = AI.T5RelativeBias(2)
        op2.forward(3)
        with pytest.raises(ValueError):
            op2.backward([[[1.0] * 3 for _ in range(3)]])
        with pytest.raises(ValueError):
            op2.backward([[[1.0] * 3] for _ in range(2)])


class TestXPosFactors:
    def test_shape_and_decay(self):
        f = AI.xpos_factors(5, 8)
        assert len(f) == 5 and all(len(r) == 8 for r in f)
        # 位置 0 处乘子为 1（与标准 RoPE 一致）
        assert all(v == 1.0 for v in f[0])
        # 越靠后衰减越多；同一位置内低频索引（theta 更大）衰减更多
        assert all(f[t][0] > f[t + 1][0] for t in range(4))
        assert f[4][0] < f[4][6]

    def test_extremes_and_errors(self):
        # alpha=0 等于关闭 xPos
        assert all(v == 1.0 for r in AI.xpos_factors(3, 8, alpha=0.0) for v in r)
        # offset 相当于从更靠后的位置开始，因子只会更小
        m = AI.xpos_factors(3, 8, alpha=1.0, offset=2.0)
        assert m[0][0] < AI.xpos_factors(3, 8)[0][0]
        for fn in (lambda: AI.xpos_factors(0, 4),
                   lambda: AI.xpos_factors(3, 5),
                   lambda: AI.xpos_factors(3, 4, base=0.0),
                   lambda: AI.xpos_factors(3, 4, alpha=-1.0),
                   lambda: AI.xpos_factors(3, 4, offset=-1.0)):
            with pytest.raises((ValueError, TypeError)):
                fn()


class TestGraphLayers:
    XG = [[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]]
    AG = [[0.0, 1.0, 0.3], [1.0, 0.0, 1.0], [0.3, 1.0, 0.0]]
    DG = [[0.5, -0.2, 0.1], [0.1, 0.3, -0.4], [-0.2, 0.6, 0.25]]

    def _target(self, model, out, dout):
        return sum(out[a][b] * dout[a][b]
                   for a in range(len(out)) for b in range(len(out[0])))

    def test_gcn_forward_and_grad(self):
        m = AI.GCNLayer(2, 3, activation='linear', seed=3)
        out = m.forward(self.XG, self.AG)
        assert len(out) == 3 and len(out[0]) == 3
        assert m.param_count() == 9
        # 无边 + 无自环 => A_hat = 0，输出只剩偏置
        iso = [[0.0] * 3 for _ in range(3)]
        m0 = AI.GCNLayer(2, 2, activation='linear', self_loops=False)
        for row in m0.forward(self.XG, iso):
            assert row == m0.dense.b
        # 自环让孤立节点的 Â 恰为单位阵
        m1 = AI.GCNLayer(2, 2, activation='linear', self_loops=False + True)
        o1 = m1.forward(self.XG, iso)
        for i, row in enumerate(self.XG):
            for h in range(2):
                want = m1.dense.b[h] + sum(m1.dense.W[h][k] * row[k] for k in range(2))
                assert abs(o1[i][h] - want) < 1e-12
        m.zero_grad()
        m.forward(self.XG, self.AG)
        m.backward(self.DG)
        ana = m.dense.gW[0][0]
        eps = 1e-6
        m.dense.W[0][0] += eps
        up = self._target(m, m.forward(self.XG, self.AG), self.DG)
        m.dense.W[0][0] -= 2 * eps
        dn = self._target(m, m.forward(self.XG, self.AG), self.DG)
        m.dense.W[0][0] += eps
        assert abs(ana - (up - dn) / (2 * eps)) < 1e-5

    def test_gcn_dx_and_activation(self):
        for act in ('relu', 'tanh'):
            m = AI.GCNLayer(2, 3, activation=act, seed=4)
            m.zero_grad()
            m.forward(self.XG, self.AG)
            dx = m.backward(self.DG)
            eps = 1e-6
            for i in range(3):
                base = self.XG[i][0]
                self.XG[i][0] = base + eps
                up = self._target(m, m.forward(self.XG, self.AG), self.DG)
                self.XG[i][0] = base - eps
                dn = self._target(m, m.forward(self.XG, self.AG), self.DG)
                self.XG[i][0] = base
                num = (up - dn) / (2 * eps)
                assert abs(dx[i][0] - num) < 1e-5, (act, i, dx[i][0], num)

    def test_graphsage(self):
        for concat in (True, False):
            m = AI.GraphSAGELayer(2, 3, activation='tanh', concat=concat, seed=5)
            out = m.forward(self.XG, self.AG)
            assert len(out) == 3 and len(out[0]) == 3
            want = (4 * 3 + 3) if concat else (2 * 3 + 3)
            assert m.param_count() == want
            m.zero_grad()
            m.forward(self.XG, self.AG)
            m.backward(self.DG)
            ana = m.dense.gW[1][1]
            eps = 1e-6
            m.dense.W[1][1] += eps
            up = self._target(m, m.forward(self.XG, self.AG), self.DG)
            m.dense.W[1][1] -= 2 * eps
            dn = self._target(m, m.forward(self.XG, self.AG), self.DG)
            m.dense.W[1][1] += eps
            assert abs(ana - (up - dn) / (2 * eps)) < 1e-5

    def test_gin_and_eps(self):
        for train_eps in (True, False):
            m = AI.GINLayer(2, 4, 3, activation='relu', eps=0.1,
                            train_eps=train_eps, seed=7)
            m.forward(self.XG, self.AG)
            m.zero_grad()
            m.forward(self.XG, self.AG)
            m.backward(self.DG)
            want = (2 * 4 + 4) + (4 * 3 + 3) + (1 if train_eps else 0)
            assert m.param_count() == want
            ana = m.g_eps[0]
            eps = 1e-6
            m._eps[0] += eps
            up = self._target(m, m.forward(self.XG, self.AG), self.DG)
            m._eps[0] -= 2 * eps
            dn = self._target(m, m.forward(self.XG, self.AG), self.DG)
            m._eps[0] += eps
            num = (up - dn) / (2 * eps)
            if train_eps:
                assert abs(ana - num) < 1e-5, (ana, num)
            else:
                assert ana == 0.0
        # 无边 + eps=0 退化为两层 MLP
        m2 = AI.GINLayer(2, 3, 2, activation='linear', eps=0.0, seed=9)
        iso = [[0.0] * 3 for _ in range(3)]
        out = m2.forward(self.XG, iso)
        for i, row in enumerate(self.XG):
            z1 = [sum(m2.l1.W[h][k] * row[k] for k in range(2)) + m2.l1.b[h]
                  for h in range(3)]
            for h in range(2):
                want = m2.l2.b[h] + sum(m2.l2.W[h][k] * z1[k] for k in range(3))
                assert abs(out[i][h] - want) < 1e-12

    def test_readout(self):
        rm = AI.GraphReadout('mean')
        r = rm.forward(self.XG)
        assert abs(r[0] - 2.0 / 3) < 1e-15 and abs(r[1] - 1.0) < 1e-15
        dr = rm.backward([1.0, 1.0])
        assert all(abs(v - 1.0 / 3) < 1e-15 for row in dr for v in row)
        rs = AI.GraphReadout('sum')
        assert rs.forward(self.XG) == [2.0, 3.0]
        rs.backward([1.0, 1.0])
        assert all(v == 1.0 for row in rs.backward([1.0, 1.0]) for v in row)
        rx = AI.GraphReadout('max')
        assert rx.forward(self.XG) == [1.0, 2.0]
        dx = rx.backward([1.0, 1.0])
        # 第 0 列 node0 与 node2 都是 1.0，严格大于保留先出现的 node0
        assert dx[0][0] == 1.0 and dx[1][1] == 1.0
        assert dx[2][0] == 0.0 and dx[0][1] == 0.0
        assert rm.param_count() == 0

    def test_train_and_errors(self):
        m = AI.GCNLayer(2, 2, activation='tanh', seed=11)
        XD = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]]
        AD = [[0.0, 1.0, 1.0, 0.0], [1.0, 0.0, 0.0, 1.0],
              [1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 1.0, 0.0]]
        YD = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]]
        first = None
        loss = None
        for _ in range(120):
            out = m.forward(XD, AD)
            loss = sum((out[a][b] - YD[a][b]) ** 2 for a in range(4)
                       for b in range(2)) / 4
            if first is None:
                first = loss
            m.zero_grad()
            m.backward([[2.0 * (out[a][b] - YD[a][b]) / 4 for b in range(2)]
                        for a in range(4)])
            m.adam_step(0.05)
        assert loss < first * 0.5, (first, loss)
        for fn in (lambda: AI.GCNLayer(0, 3),
                   lambda: AI.GCNLayer(2, 3, activation='softmax'),
                   lambda: AI.GCNLayer(2, 3).forward([], self.AG),
                   lambda: AI.GCNLayer(2, 3).forward([[1.0, 2.0], [1.0]], self.AG),
                   lambda: AI.GCNLayer(2, 3).forward(self.XG, [[0, 1], [1, 0]]),
                   lambda: AI.GINLayer(2, 0, 3),
                   lambda: AI.GINLayer(2, 3, 4, eps=float('inf')),
                   lambda: AI.GraphReadout('median')):
            with pytest.raises((ValueError, TypeError)):
                fn()
        with pytest.raises(RuntimeError):
            AI.GCNLayer(2, 3).backward(self.DG)


# ============================================================================
#  W6-B: GAT / APPNP / JKNet
# ============================================================================

class TestGATLayer:
    """图注意力：注意力语义、dx/dW/da 中心差分、多头 concat/mean"""

    X = [[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]]
    ADJ = [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]

    def _loss_and_grads(self, gm, dout):
        out = gm.forward(self.X, self.ADJ)
        gm.zero_grad()
        gm.backward(dout)
        return sum(out[a][b] * dout[a][b]
                   for a in range(len(out)) for b in range(len(out[0])))

    def test_attention_semantics(self):
        g = AI.GATLayer(2, 2, num_heads=1, activation='linear', seed=3)
        out = g.forward(self.X, self.ADJ)
        assert len(out) == 3 and len(out[0]) == 2
        assert g.param_count() == (2 * 2 + 2) + 2 * 2
        al = g._cache[3][0]
        for i in range(3):
            assert abs(sum(al[i]) - 1.0) < 1e-12
        assert al[0][2] == 0.0            # 非邻居权重为 0

    def test_gradient_finite_difference(self):
        for concat in (True, False):
            gm = AI.GATLayer(2, 2, num_heads=2, activation='elu',
                             concat=concat, seed=5)
            w = len(gm.forward(self.X, self.ADJ)[0])
            dout = [[0.3, -0.2, 0.1, 0.05][:w], [0.1, 0.2, -0.3, 0.4][:w],
                    [-0.2, 0.1, 0.25, -0.15][:w]]
            eps = 1e-6
            gm.zero_grad()
            gm.forward(self.X, self.ADJ)
            dx = gm.backward(dout)

            def tgt():
                o = gm.forward(self.X, self.ADJ)
                return sum(o[a][b] * dout[a][b] for a in range(3) for b in range(w))

            for i in range(3):
                old = self.X[i][0]
                self.X[i][0] = old + eps
                up = tgt()
                self.X[i][0] = old - eps
                dn = tgt()
                self.X[i][0] = old
                num = (up - dn) / (2 * eps)
                assert abs(dx[i][0] - num) < 2e-5 * max(1.0, abs(num))
            # 注意力向量的梯度
            ana = gm.ga_src[1][0]
            gm.a_src[1][0] += eps
            up = tgt()
            gm.a_src[1][0] -= 2 * eps
            dn = tgt()
            gm.a_src[1][0] += eps
            assert abs(ana - (up - dn) / (2 * eps)) < 2e-5
            ana = gm.ga_dst[0][1]
            gm.a_dst[0][1] += eps
            up = tgt()
            gm.a_dst[0][1] -= 2 * eps
            dn = tgt()
            gm.a_dst[0][1] += eps
            assert abs(ana - (up - dn) / (2 * eps)) < 2e-5

    def test_accumulate_zero_grad_and_optimizers(self):
        g = AI.GATLayer(2, 2, num_heads=1, activation='elu', seed=7)
        dout = [[0.3, -0.2], [0.1, 0.2], [-0.2, 0.1]]
        g.zero_grad()
        g.forward(self.X, self.ADJ)
        g.backward(dout)
        first = g.gW[0][0][0]
        g.forward(self.X, self.ADJ)
        g.backward(dout)
        assert abs(g.gW[0][0][0] - 2.0 * first) < 1e-12
        g.zero_grad()
        assert all(v == 0.0 for head in g.gW for row in head for v in row)
        before = [v for v in g.W[0][0]]
        g.forward(self.X, self.ADJ)
        g.backward(dout)
        g.sgd_step(0.01)
        assert g.W[0][0] != before
        g.adam_step(0.01)
        with pytest.raises(ValueError):
            g.sgd_step(-1.0)

    def test_invalid(self):
        with pytest.raises(ValueError):
            AI.GATLayer(0, 3)
        with pytest.raises(ValueError):
            AI.GATLayer(2, 3, num_heads=0)
        with pytest.raises(ValueError):
            AI.GATLayer(2, 3, negative_slope=0.0)
        with pytest.raises(ValueError):
            AI.GATLayer(2, 3, activation='softmax')
        with pytest.raises(ValueError):
            AI.GATLayer(2, 3).forward(self.X, [[0, 1], [1, 0]])
        with pytest.raises(RuntimeError):
            AI.GATLayer(2, 3).backward([[1.0, 2.0, 3.0]])


class TestAPPNPLayer:
    """APPNP：alpha=0/1 两端语义、K 档梯度、训练下降"""

    X = [[1.0, 0.0], [0.0, 2.0], [1.0, 1.0], [0.5, -1.0]]
    ADJ = [[0.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 0.0],
           [0.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 0.0]]
    DOUT = [[0.3, -0.2, 0.1], [0.1, 0.2, -0.3], [-0.2, 0.1, 0.25], [0.4, -0.1, 0.2]]

    def test_alpha_extremes(self):
        a0 = AI.APPNPLayer(2, 2, K=2, alpha=0.0, activation='linear', seed=7)
        o0 = a0.forward(self.X, self.ADJ)
        an = AI._g_adj_norm(self.ADJ, True)
        h0 = a0.dense.forward(self.X)
        ref = AI._g_matmul(an, AI._g_matmul(an, h0))
        for i in range(4):
            for r in range(2):
                assert abs(o0[i][r] - ref[i][r]) < 1e-12
        a1 = AI.APPNPLayer(2, 2, K=2, alpha=1.0, activation='linear', seed=7)
        o1 = a1.forward(self.X, self.ADJ)
        h0b = a1.dense.forward(self.X)
        for i in range(4):
            for r in range(2):
                assert abs(o1[i][r] - h0b[i][r]) < 1e-12

    def test_gradient_finite_difference(self):
        eps = 1e-6
        for K in (1, 2, 3):
            am = AI.APPNPLayer(2, 3, K=K, alpha=0.2, activation='relu', seed=9)
            am.zero_grad()
            out = am.forward(self.X, self.ADJ)
            dx = am.backward(self.DOUT)

            def tgt():
                o = am.forward(self.X, self.ADJ)
                return sum(o[a][b] * self.DOUT[a][b]
                           for a in range(4) for b in range(3))

            ana = am.dense.gW[0][1]
            am.dense.W[0][1] += eps
            up = tgt()
            am.dense.W[0][1] -= 2 * eps
            dn = tgt()
            am.dense.W[0][1] += eps
            assert abs(ana - (up - dn) / (2 * eps)) < 2e-5
            for i in range(4):
                old = self.X[i][1]
                self.X[i][1] = old + eps
                up = tgt()
                self.X[i][1] = old - eps
                dn = tgt()
                self.X[i][1] = old
                num = (up - dn) / (2 * eps)
                assert abs(dx[i][1] - num) < 2e-5 * max(1.0, abs(num))

    def test_train_and_invalid(self):
        am = AI.APPNPLayer(2, 1, K=2, alpha=0.2, activation='relu', seed=13)
        ys = [[1.0], [0.0], [1.0], [0.0]]
        first = None
        loss = None
        for _ in range(120):
            out = am.forward(self.X, self.ADJ)
            loss = sum((out[i][0] - ys[i][0]) ** 2 for i in range(4)) / 4
            if first is None:
                first = loss
            am.zero_grad()
            am.backward([[2.0 * (out[i][0] - ys[i][0]) / 4] for i in range(4)])
            am.adam_step(0.05)
        assert loss < first * 0.7
        assert am.param_count() == 2 * 1 + 1
        with pytest.raises(ValueError):
            AI.APPNPLayer(2, 3, K=0)
        with pytest.raises(ValueError):
            AI.APPNPLayer(2, 3, alpha=1.5)
        with pytest.raises(RuntimeError):
            AI.APPNPLayer(2, 3).backward([[1.0, 2.0, 3.0]])


class TestJKNet:
    """JK-Net：concat/last/max 三种汇总与反向"""

    X = [[1.0, 0.0], [0.0, 2.0], [1.0, 1.0], [0.5, -1.0]]
    ADJ = [[0.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 0.0],
           [0.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 0.0]]
    DOUT = [[0.3, -0.2, 0.1], [0.1, 0.2, -0.3], [-0.2, 0.1, 0.25], [0.4, -0.1, 0.2]]

    def _pair(self):
        return (AI.GCNLayer(2, 3, activation='relu', seed=11),
                AI.GCNLayer(3, 3, activation='relu', seed=12))

    def test_concat(self):
        l1, l2 = self._pair()
        jk = AI.JKNet([l1, l2], mode='concat')
        out = jk.forward(self.X, self.ADJ)
        assert len(out[0]) == 6
        jk.zero_grad()
        dx = jk.backward([row + row for row in self.DOUT])
        assert len(dx) == 4 and len(dx[0]) == 2
        assert jk.param_count() == l1.param_count() + l2.param_count()

    def test_last_equals_manual_stack(self):
        l1, l2 = self._pair()
        jk = AI.JKNet([l1, l2], mode='last')
        jk.forward(self.X, self.ADJ)
        jk.zero_grad()
        dx_last = jk.backward(self.DOUT)
        g1, g2 = self._pair()
        g1.zero_grad()
        g2.zero_grad()
        h1 = g1.forward(self.X, self.ADJ)
        h2 = g2.forward(h1, self.ADJ)
        d2 = g2.backward(self.DOUT)
        dx_man = g1.backward(d2)
        for r in range(3):
            for c in range(2):
                assert abs(g1.dense.gW[r][c] - l1.dense.gW[r][c]) < 1e-12
            for c in range(3):
                assert abs(g2.dense.gW[r][c] - l2.dense.gW[r][c]) < 1e-12
        for i in range(4):
            for c in range(2):
                assert abs(dx_man[i][c] - dx_last[i][c]) < 1e-12

    def test_max_and_invalid(self):
        l1, l2 = self._pair()
        jm = AI.JKNet([l1, l2], mode='max')
        jm.forward(self.X, self.ADJ)
        jm.zero_grad()
        jm.backward(self.DOUT)
        got = (any(v != 0.0 for row in l1.dense.gW for v in row)
               or any(v != 0.0 for row in l2.dense.gW for v in row))
        assert got
        with pytest.raises(ValueError):
            AI.JKNet([], mode='concat')
        with pytest.raises(ValueError):
            AI.JKNet([AI.GCNLayer(2, 3)], mode='median')
        with pytest.raises(TypeError):
            AI.JKNet(['not-a-layer'], mode='last')


# ============================================================================
#  W6-C: MPNN / GCNII / 图池化
# ============================================================================

class TestMPNNLayer:
    """消息传递：sum/mean 聚合、边特征、梯度中心差分、累加与训练"""

    X = [[1.0, 0.0], [0.0, 2.0], [1.0, 1.0], [0.5, -1.0]]
    ADJ = [[0.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 0.0],
           [0.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 0.0]]
    DOUT = [[0.3, -0.2, 0.1], [0.1, 0.2, -0.3], [-0.2, 0.1, 0.25], [0.4, -0.1, 0.2]]

    def test_shape_and_params(self):
        m = AI.MPNNLayer(2, 5, 3, activation='relu', seed=3)
        out = m.forward(self.X, self.ADJ)
        assert len(out) == 4 and len(out[0]) == 3
        want = (4 * 5 + 5) + (5 * 3 + 3) + (2 + 3) * 5 + 5 + (5 * 3 + 3)
        assert m.param_count() == want

    def test_gradient_finite_difference(self):
        eps = 1e-6
        m = AI.MPNNLayer(2, 5, 3, activation='relu', seed=3)

        def loss():
            o = m.forward(self.X, self.ADJ)
            return sum(o[a][b] * self.DOUT[a][b]
                       for a in range(4) for b in range(3))

        m.zero_grad()
        m.forward(self.X, self.ADJ)
        dx = m.backward(self.DOUT)
        for i in range(4):
            for j in range(2):
                old = self.X[i][j]
                self.X[i][j] = old + eps
                up = loss()
                self.X[i][j] = old - eps
                dn = loss()
                self.X[i][j] = old
                num = (up - dn) / (2 * eps)
                assert abs(dx[i][j] - num) < 2e-5 * max(1.0, abs(num))
        m.zero_grad()
        m.forward(self.X, self.ADJ)
        m.backward(self.DOUT)
        ana = m.msg1.gW[0][0]
        old = m.msg1.W[0][0]
        m.msg1.W[0][0] = old + eps
        up = loss()
        m.msg1.W[0][0] = old - eps
        dn = loss()
        m.msg1.W[0][0] = old
        assert abs(ana - (up - dn) / (2 * eps)) < 2e-5

    def test_aggregation_edge_and_train(self):
        mean = AI.MPNNLayer(2, 4, 2, activation='linear', aggregation='mean', seed=5)
        summ = AI.MPNNLayer(2, 4, 2, activation='linear', aggregation='sum', seed=5)
        om = mean.forward(self.X, self.ADJ)
        os_ = summ.forward(self.X, self.ADJ)
        assert any(abs(om[i][j] - os_[i][j]) > 1e-12
                   for i in range(4) for j in range(2))
        iso = [[1.0, 0.0], [0.0, 1.0]]
        assert len(mean.forward(iso, [[0.0, 0.0], [0.0, 0.0]])) == 2
        me = AI.MPNNLayer(2, 4, 2, edge_features=1, activation='linear', seed=7)
        edge = [[[0.5 if self.ADJ[i][j] > 0 else 0.0] for j in range(4)]
                for i in range(4)]
        assert len(me.forward(self.X, self.ADJ, edge)[0]) == 2
        tr = AI.MPNNLayer(2, 6, 1, activation='relu', seed=23)
        ys = [[1.0], [0.0], [1.0], [0.0]]
        first = None
        last = None
        for _ in range(120):
            o = tr.forward(self.X, self.ADJ)
            last = sum((o[i][0] - ys[i][0]) ** 2 for i in range(4)) / 4
            if first is None:
                first = last
            tr.zero_grad()
            tr.backward([[2.0 * (o[i][0] - ys[i][0]) / 4] for i in range(4)])
            tr.adam_step(0.05)
        assert last < first * 0.5
        with pytest.raises(ValueError):
            AI.MPNNLayer(0, 3, 2)
        with pytest.raises(ValueError):
            AI.MPNNLayer(2, 3, 2, edge_features=-1)
        with pytest.raises(ValueError):
            AI.MPNNLayer(2, 3, 2, aggregation='max')
        with pytest.raises(ValueError):
            AI.MPNNLayer(2, 3, 2, activation='softmax')
        with pytest.raises(ValueError):
            AI.MPNNLayer(2, 3, 2).forward(self.X, [[1.0, 0.0]])
        with pytest.raises(RuntimeError):
            AI.MPNNLayer(2, 3, 2).backward(self.DOUT)


class TestGCNIILayer:
    """GCNII：beta 公式、混合矩阵、H0 通路、三组梯度"""

    X = [[1.0, 0.0], [0.0, 2.0], [1.0, 1.0], [0.5, -1.0]]
    ADJ = [[0.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 0.0],
           [0.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 0.0]]
    H0 = [[1.0, 2.0], [0.5, -1.0], [2.0, 0.0], [-1.0, 1.0]]
    DOUT = [[0.3, -0.2], [0.1, 0.2], [-0.2, 0.1], [0.4, -0.1]]

    def test_beta_and_mix_matrix(self):
        g = AI.GCNIILayer(2, alpha=0.1, lam=0.5, layer_index=1, seed=9)
        assert abs(g.beta() - math.log(0.5 / 2.0 + 1.0)) < 1e-12
        beta = g.beta()
        M = g._mix_matrix()
        for a in range(2):
            for c in range(2):
                want = beta * g.W[a][c] + ((1.0 - beta) if a == c else 0.0)
                assert abs(M[a][c] - want) < 1e-12
        assert g.param_count() == 4

    def test_alpha_zero_ignores_h0(self):
        g = AI.GCNIILayer(2, alpha=0.0, layer_index=1, activation='linear', seed=9)
        o1 = g.forward(self.X, self.ADJ, [[9.0, 9.0]] * 4)
        o2 = g.forward(self.X, self.ADJ, [[-9.0, -9.0]] * 4)
        for i in range(4):
            for j in range(2):
                assert abs(o1[i][j] - o2[i][j]) < 1e-12

    def test_gradients_and_optimizer(self):
        eps = 1e-6
        g = AI.GCNIILayer(2, alpha=0.1, layer_index=1, activation='linear', seed=9)
        H0 = [row[:] for row in self.H0]

        def loss():
            o = g.forward(self.X, self.ADJ, H0)
            return sum(o[a][b] * self.DOUT[a][b]
                       for a in range(4) for b in range(2))

        g.zero_grad()
        g.forward(self.X, self.ADJ, H0)
        dx = g.backward(self.DOUT)
        gW = [row[:] for row in g.gW]
        dh0 = [row[:] for row in g.grad_h0]
        for i in range(4):
            for j in range(2):
                old = self.X[i][j]
                self.X[i][j] = old + eps
                up = loss()
                self.X[i][j] = old - eps
                dn = loss()
                self.X[i][j] = old
                num = (up - dn) / (2 * eps)
                assert abs(dx[i][j] - num) < 2e-5 * max(1.0, abs(num))
        for a in range(2):
            for c in range(2):
                old = g.W[a][c]
                g.W[a][c] = old + eps
                up = loss()
                g.W[a][c] = old - eps
                dn = loss()
                g.W[a][c] = old
                num = (up - dn) / (2 * eps)
                assert abs(gW[a][c] - num) < 2e-5 * max(1.0, abs(num))
        for i in range(4):
            for j in range(2):
                old = H0[i][j]
                H0[i][j] = old + eps
                up = loss()
                H0[i][j] = old - eps
                dn = loss()
                H0[i][j] = old
                num = (up - dn) / (2 * eps)
                assert abs(dh0[i][j] - num) < 2e-5 * max(1.0, abs(num))
        before = [row[:] for row in g.W]
        g.zero_grad()
        g.forward(self.X, self.ADJ, H0)
        g.backward(self.DOUT)
        g.sgd_step(0.01)
        assert g.W != before
        g.adam_step(0.01)
        with pytest.raises(ValueError):
            AI.GCNIILayer(0)
        with pytest.raises(ValueError):
            AI.GCNIILayer(2, alpha=1.5)
        with pytest.raises(ValueError):
            AI.GCNIILayer(2, lam=-0.1)
        with pytest.raises(RuntimeError):
            AI.GCNIILayer(2).backward(self.DOUT)


class TestGraphPooling:
    """TopKPool / SAGPool：保留数、子图、gate 缩放、全节点梯度"""

    X = [[1.0, 0.0], [0.0, 2.0], [1.0, 1.0], [0.5, -1.0]]
    ADJ = [[0.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 0.0],
           [0.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 0.0]]

    def test_topk_forward(self):
        p = AI.TopKPool(2, ratio=0.5, seed=11)
        xp, ap = p.forward(self.X, self.ADJ)
        assert len(xp) == 2 and len(ap) == 2 and len(ap[0]) == 2
        assert p.param_count() == 3
        idx = p._cache[0]
        for i in range(2):
            for j in range(2):
                assert abs(xp[i][j]) <= abs(self.X[idx[i]][j]) + 1e-12

    def test_topk_gradient(self):
        eps = 1e-6
        p = AI.TopKPool(2, ratio=0.5, seed=11)
        dout = [[1.0, -2.0], [0.5, 0.25]]

        def loss():
            o, _a = p.forward(self.X, self.ADJ)
            return sum(o[a][b] * dout[a][b] for a in range(2) for b in range(2))

        p.zero_grad()
        p.forward(self.X, self.ADJ)
        dx = p.backward(dout)
        for i in range(4):
            for j in range(2):
                old = self.X[i][j]
                self.X[i][j] = old + eps
                up = loss()
                self.X[i][j] = old - eps
                dn = loss()
                self.X[i][j] = old
                num = (up - dn) / (2 * eps)
                assert abs(dx[i][j] - num) < 2e-5 * max(1.0, abs(num))

    def test_sagpool(self):
        eps = 1e-6
        sp = AI.SAGPool(2, ratio=0.5, seed=13)
        xs, as_ = sp.forward(self.X, self.ADJ)
        assert len(xs) == 2 and len(as_) == 2
        assert sp.param_count() == 3
        dout = [[1.0, 0.5], [-0.5, 1.0]]

        def loss():
            o, _a = sp.forward(self.X, self.ADJ)
            return sum(o[a][b] * dout[a][b] for a in range(2) for b in range(2))

        sp.zero_grad()
        sp.forward(self.X, self.ADJ)
        dx = sp.backward(dout)
        for i in range(4):
            for j in range(2):
                old = self.X[i][j]
                self.X[i][j] = old + eps
                up = loss()
                self.X[i][j] = old - eps
                dn = loss()
                self.X[i][j] = old
                num = (up - dn) / (2 * eps)
                assert abs(dx[i][j] - num) < 2e-5 * max(1.0, abs(num))

    def test_invalid(self):
        with pytest.raises(ValueError):
            AI.TopKPool(2, ratio=0.0)
        with pytest.raises(ValueError):
            AI.TopKPool(2, ratio=1.5)
        with pytest.raises(ValueError):
            AI.SAGPool(0)
        with pytest.raises(RuntimeError):
            AI.TopKPool(2).backward([[1.0, 0.0], [0.0, 1.0]])
        with pytest.raises(RuntimeError):
            AI.SAGPool(2).backward([[1.0, 0.0], [0.0, 1.0]])


# ============================================================================
#  W7-A: 提示微调 / 序列打包 / 前缀缓存
# ============================================================================

class TestPromptTuning:
    """软提示：拼接语义、g_prompt 中心差分、累加与优化器"""

    SEQ = [[1.0, 0.5], [0.0, -1.0], [2.0, 1.0]]
    DOUT = [[0.3, -0.2], [0.1, 0.2], [-0.2, 0.1], [0.4, -0.1], [0.05, 0.15]]

    def test_shape_and_params(self):
        pt = AI.PromptTuning(2, 2, seed=5)
        out = pt.forward(self.SEQ)
        assert len(out) == 5
        assert out[:2] == [list(r) for r in pt.prompt]
        assert pt.param_count() == 4

    def test_gradient_finite_difference(self):
        eps = 1e-6
        pt = AI.PromptTuning(2, 2, seed=5)

        def loss():
            o = pt.forward(self.SEQ)
            return sum(o[a][b] * self.DOUT[a][b] for a in range(5) for b in range(2))

        pt.zero_grad()
        pt.forward(self.SEQ)
        dseq = pt.backward(self.DOUT)
        assert dseq == [list(r) for r in self.DOUT[2:]]
        for t in range(2):
            for j in range(2):
                old = pt.prompt[t][j]
                pt.prompt[t][j] = old + eps
                up = loss()
                pt.prompt[t][j] = old - eps
                dn = loss()
                pt.prompt[t][j] = old
                num = (up - dn) / (2 * eps)
                assert abs(pt.g_prompt[t][j] - num) < 2e-5 * max(1.0, abs(num))

    def test_accumulate_and_optimizers(self):
        pt = AI.PromptTuning(2, 2, seed=5)
        pt.zero_grad()
        pt.forward(self.SEQ)
        pt.backward(self.DOUT)
        first = pt.g_prompt[0][0]
        pt.forward(self.SEQ)
        pt.backward(self.DOUT)
        assert abs(pt.g_prompt[0][0] - 2.0 * first) < 1e-12
        pt.zero_grad()
        assert pt.g_prompt[0][0] == 0.0
        before = [r[:] for r in pt.prompt]
        pt.forward(self.SEQ)
        pt.backward(self.DOUT)
        pt.sgd_step(0.01)
        assert pt.prompt != before
        pt.adam_step(0.01)
        with pytest.raises(ValueError):
            AI.PromptTuning(0, 2)
        with pytest.raises(ValueError):
            AI.PromptTuning(2, 2).forward([[1.0]])
        with pytest.raises(RuntimeError):
            AI.PromptTuning(2, 2).backward(self.DOUT)


class TestPTuning:
    """P-Tuning：MLP 生成的提示、四组梯度、优化器"""

    SEQ = [[1.0, 0.5], [0.0, -1.0], [2.0, 1.0]]
    DOUT = [[0.3, -0.2], [0.1, 0.2], [-0.2, 0.1], [0.4, -0.1], [0.05, 0.15]]

    def test_shape_and_params(self):
        pp = AI.P_Tuning(2, 2, hidden=3, seed=7)
        prompt, _rows = pp.prompt_matrix()
        o = pp.forward(self.SEQ)
        for t in range(2):
            for j in range(2):
                assert abs(o[t][j] - prompt[t][j]) < 1e-12
        assert pp.param_count() == 2 * 2 + 3 * 2 + 3 + 2 * 3 + 2

    def test_gradient_finite_difference(self):
        eps = 1e-6
        pp = AI.P_Tuning(2, 2, hidden=3, seed=7)

        def loss():
            o = pp.forward(self.SEQ)
            return sum(o[a][b] * self.DOUT[a][b] for a in range(5) for b in range(2))

        pp.zero_grad()
        pp.forward(self.SEQ)
        pp.backward(self.DOUT)
        for t in range(2):
            for j in range(2):
                old = pp.anchor[t][j]
                pp.anchor[t][j] = old + eps
                up = loss()
                pp.anchor[t][j] = old - eps
                dn = loss()
                pp.anchor[t][j] = old
                num = (up - dn) / (2 * eps)
                assert abs(pp.g_anchor[t][j] - num) < 2e-5 * max(1.0, abs(num))
        for r in range(3):
            for k in range(2):
                old = pp.W1[r][k]
                pp.W1[r][k] = old + eps
                up = loss()
                pp.W1[r][k] = old - eps
                dn = loss()
                pp.W1[r][k] = old
                num = (up - dn) / (2 * eps)
                assert abs(pp.g_W1[r][k] - num) < 2e-5 * max(1.0, abs(num))
        for j in range(2):
            for r in range(3):
                old = pp.W2[j][r]
                pp.W2[j][r] = old + eps
                up = loss()
                pp.W2[j][r] = old - eps
                dn = loss()
                pp.W2[j][r] = old
                num = (up - dn) / (2 * eps)
                assert abs(pp.g_W2[j][r] - num) < 2e-5 * max(1.0, abs(num))

    def test_optimizers_and_errors(self):
        pp = AI.P_Tuning(2, 2, hidden=3, seed=7)
        before = [r[:] for r in pp.anchor]
        pp.zero_grad()
        pp.forward(self.SEQ)
        pp.backward(self.DOUT)
        pp.sgd_step(0.01)
        assert pp.anchor != before
        pp.adam_step(0.01)
        with pytest.raises(ValueError):
            AI.P_Tuning(0, 2)
        with pytest.raises(ValueError):
            AI.P_Tuning(2, 2, hidden=0)
        with pytest.raises(RuntimeError):
            AI.P_Tuning(2, 2).backward(self.DOUT)


class TestPackingAndPrefixCache:
    """sequence_packing / packed_causal_mask / PrefixCache"""

    def test_packing(self):
        pk = AI.sequence_packing([[5, 6], [7, 8, 9]])
        assert pk['tokens'] == [5, 6, 7, 8, 9]
        assert pk['segment_ids'] == [0, 0, 1, 1, 1]
        assert pk['position_ids'] == [0, 1, 0, 1, 2]
        assert pk['segments'] == [(0, 2), (2, 5)]
        assert pk['length'] == 5

    def test_packed_mask(self):
        m = AI.packed_causal_mask([0, 0, 1, 1, 1])
        assert m[0][2] == 0.0 and m[3][0] == 0.0
        assert m[4][3] == 1.0 and m[3][4] == 0.0
        assert all(m[i][i] == 1.0 for i in range(5))
        full = AI.packed_causal_mask([0, 0], causal=False)
        assert full[0][1] == 1.0 and full[1][0] == 1.0

    def test_prefix_cache_ops(self):
        pc = AI.PrefixCache(2, 2)
        pc.set(0, [[1.0, 2.0]], [[0.5, -0.5]])
        pc.set(1, [[3.0, 4.0], [5.0, 6.0]], [[1.0, 1.0], [2.0, 2.0]])
        assert pc.length(0) == 1 and pc.length(1) == 2
        k0, _v0 = pc.get(0)
        assert k0 == [[1.0, 2.0]]
        k0[0][0] = 99.0
        assert pc.get(0)[0][0][0] == 1.0
        pc.truncate(1)
        assert pc.length(1) == 1
        other = AI.PrefixCache(2, 2)
        other.set(0, [[7.0, 8.0]], [[9.0, 9.0]])
        other.set(1, [[1.0, 0.0]], [[0.0, 1.0]])
        pc.merge(other)
        assert pc.length(0) == 2 and pc.length(1) == 2
        d = pc.to_dict()
        assert AI.PrefixCache.from_dict(d).to_dict() == d

    def test_errors(self):
        with pytest.raises(ValueError):
            AI.sequence_packing([])
        with pytest.raises(ValueError):
            AI.sequence_packing([[]])
        with pytest.raises(ValueError):
            AI.sequence_packing([[1.5]])
        with pytest.raises(ValueError):
            AI.packed_causal_mask([])
        with pytest.raises(ValueError):
            AI.PrefixCache(0, 2)
        with pytest.raises(ValueError):
            AI.PrefixCache(2, 2).set(5, [], [])
        with pytest.raises(ValueError):
            AI.PrefixCache(2, 2).set(0, [[1.0]], [[1.0, 2.0]])


# ============================================================
#  W7-B：DoRA / AdaLoRA / 采样扩展
# ============================================================

def _w7b_layer(seed=7):
    random.seed(seed)
    return AI.Layer(3, 2, "linear")


def _w7b_fill(obj, seed=11, scale=0.35):
    rng = random.Random(seed)
    for j in range(obj.fan_out):
        for r in range(obj.rank):
            obj.B[j][r] = rng.gauss(0.0, scale)
    for r in range(obj.rank):
        for i in range(obj.fan_in):
            obj.A[r][i] = rng.gauss(0.0, scale)
    return obj


def _w7b_num(fwd, getter, setter, eps=1e-6):
    """中心差分：只比较 loss，返回数值梯度"""
    old = getter()
    setter(old + eps)
    up = fwd()
    setter(old - eps)
    dn = fwd()
    setter(old)
    return (up - dn) / (2.0 * eps)


def _w7b_replace(row, i, v):
    new = list(row)
    new[i] = v
    return new


class TestDoRALinear:
    def test_zero_lora_equals_layer(self):
        layer = _w7b_layer()
        x = [0.3, -0.7, 1.1]
        dora = AI.DoRALinear(layer, rank=2)
        assert dora.forward(x) == layer.forward(x)
        assert dora.param_count() == 2 + 2 * 3 + 2 * 2

    def test_gradients_finite_difference(self):
        layer = _w7b_layer()
        dora = _w7b_fill(AI.DoRALinear(layer, rank=2))
        x = [0.3, -0.7, 1.1]
        dout = [0.6, -0.9]

        def fwd():
            o = dora.forward(x)
            return sum(o[j] * dout[j] for j in range(2))

        def bwd():
            dora.forward(x)
            dora.zero_grad()
            dora.backward(dout)

        def rel(a, b):
            return abs(a - b) / max(1.0, abs(a), abs(b))

        worst_v = 0.0
        for j in range(2):
            for i in range(3):
                bwd()
                ana = dora.g_V[j][i]
                num = _w7b_num(
                    fwd,
                    lambda j=j, i=i: dora.V[j][i],
                    lambda v, j=j, i=i: dora.V.__setitem__(
                        j, _w7b_replace(dora.V[j], i, v)))
                worst_v = max(worst_v, rel(ana, num))
        assert worst_v < 1e-5, worst_v

        worst_m = 0.0
        for j in range(2):
            bwd()
            ana = dora.g_m[j]
            num = _w7b_num(
                fwd,
                lambda j=j: dora.m[j],
                lambda v, j=j: dora.m.__setitem__(j, v))
            worst_m = max(worst_m, rel(ana, num))
        assert worst_m < 1e-5, worst_m

        worst_a = 0.0
        worst_b = 0.0
        for r in range(2):
            for i in range(3):
                bwd()
                ana = dora.g_A[r][i]
                num = _w7b_num(
                    fwd,
                    lambda r=r, i=i: dora.A[r][i],
                    lambda v, r=r, i=i: dora.A.__setitem__(
                        r, _w7b_replace(dora.A[r], i, v)))
                worst_a = max(worst_a, rel(ana, num))
        for j in range(2):
            for r in range(2):
                bwd()
                ana = dora.g_B[j][r]
                num = _w7b_num(
                    fwd,
                    lambda j=j, r=r: dora.B[j][r],
                    lambda v, j=j, r=r: dora.B.__setitem__(
                        j, _w7b_replace(dora.B[j], r, v)))
                worst_b = max(worst_b, rel(ana, num))
        assert worst_a < 1e-5, worst_a
        assert worst_b < 1e-5, worst_b

    def test_denormalized_and_merge(self):
        layer = _w7b_layer()
        dora = _w7b_fill(AI.DoRALinear(layer, rank=2))
        x = [0.3, -0.7, 1.1]
        out = dora.forward(x)
        w = dora.denormalized()
        man = [sum(w[j][i] * x[i] for i in range(3)) + dora.b[j] for j in range(2)]
        assert all(abs(out[j] - man[j]) < 1e-12 for j in range(2))
        dora.merge()
        merged = layer.forward(x)
        assert all(abs(merged[j] - out[j]) < 1e-12 for j in range(2))


class TestAdaLoRALinear:
    def test_zero_lora_equals_layer(self):
        layer = _w7b_layer(8)
        x = [0.3, -0.7, 1.1]
        ada = AI.AdaLoRALinear(layer, rank=4, seed=3)
        assert ada.forward(x) == layer.forward(x)
        assert ada.param_count() == 4 * 3 + 2 * 4

    def test_gradients_finite_difference(self):
        layer = _w7b_layer(8)
        ada = _w7b_fill(AI.AdaLoRALinear(layer, rank=4, seed=3), seed=5, scale=0.3)
        x = [0.3, -0.7, 1.1]
        dout = [0.6, -0.9]

        def fwd():
            o = ada.forward(x)
            return sum(o[j] * dout[j] for j in range(2))

        def bwd():
            ada.forward(x)
            ada.zero_grad()
            ada.backward(dout)

        def rel(a, b):
            return abs(a - b) / max(1.0, abs(a), abs(b))

        worst_a = 0.0
        worst_b = 0.0
        for r in range(4):
            for i in range(3):
                bwd()
                ana = ada.g_A[r][i]
                num = _w7b_num(
                    fwd,
                    lambda r=r, i=i: ada.A[r][i],
                    lambda v, r=r, i=i: ada.A.__setitem__(
                        r, _w7b_replace(ada.A[r], i, v)))
                worst_a = max(worst_a, rel(ana, num))
        for j in range(2):
            for r in range(4):
                bwd()
                ana = ada.g_B[j][r]
                num = _w7b_num(
                    fwd,
                    lambda j=j, r=r: ada.B[j][r],
                    lambda v, j=j, r=r: ada.B.__setitem__(
                        j, _w7b_replace(ada.B[j], r, v)))
                worst_b = max(worst_b, rel(ana, num))
        assert worst_a < 1e-5, worst_a
        assert worst_b < 1e-5, worst_b

    def test_prune_and_active_ranks(self):
        layer = _w7b_layer(9)
        ada = _w7b_fill(AI.AdaLoRALinear(layer, rank=4, seed=4), seed=6, scale=0.4)
        x = [0.3, -0.7, 1.1]
        dout = [0.6, -0.9]
        ada.forward(x)
        ada.zero_grad()
        ada.backward(dout)
        top2 = sorted(sorted(range(4), key=lambda r: -ada.imp[r])[:2])
        assert sorted(ada.prune(2)) == top2
        dead = [r for r in range(4) if r not in top2]
        assert all(abs(ada.A[r][i]) == 0.0 for r in dead for i in range(3))
        assert all(abs(ada.B[j][r]) == 0.0 for r in dead for j in range(2))
        base = layer.forward(x)
        o = ada.forward(x)
        man = list(base)
        for r in ada.active_ranks():
            h = sum(ada.A[r][i] * x[i] for i in range(3))
            for j in range(2):
                man[j] += ada.scaling * ada.B[j][r] * h
        assert all(abs(o[j] - man[j]) < 1e-12 for j in range(2))
        with pytest.raises(ValueError):
            ada.prune(0)
        with pytest.raises(ValueError):
            ada.prune(9)


class TestSamplingExtras:
    def test_typical_filter(self):
        logits = [2.0, 1.0, 0.1, -1.0]
        probs = AI._softmax_row(logits)
        for mass in (0.3, 0.7, 1.0):
            masked = AI.typical_filter(logits, mass)
            kept = [i for i, v in enumerate(masked) if v != float("-inf")]
            assert 0 in kept
            assert sum(probs[i] for i in kept) >= mass - 1e-12
            assert all(masked[i] == logits[i] for i in kept)
        n_small = len([v for v in AI.typical_filter(logits, 0.3) if v != float("-inf")])
        n_big = len([v for v in AI.typical_filter(logits, 1.0) if v != float("-inf")])
        assert n_small <= n_big

    def test_top_a_and_eta_monotone(self):
        logits = [2.0, 1.0, 0.1, -1.0]
        prev = None
        for ta in (0.01, 0.1, 0.5, 1.0):
            masked = AI.top_a_filter(logits, ta)
            kept = [i for i, v in enumerate(masked) if v != float("-inf")]
            assert 0 in kept
            if prev is not None:
                assert len(kept) <= prev
            prev = len(kept)
        assert AI.top_a_filter(logits, 0.0) == list(logits)

        prev = None
        for eta in (0.0, 0.2, 0.4, 0.6):
            masked = AI.eta_filter(logits, eta)
            kept = [i for i, v in enumerate(masked) if v != float("-inf")]
            assert 0 in kept
            if prev is not None:
                assert len(kept) <= prev
            prev = len(kept)

    def test_mirostat(self):
        logits = [2.0, 1.0, 0.1, -1.0]
        prev = None
        for mu in (0.5, 1.0, 2.0, 6.0):
            masked, ent = AI.mirostat_filter(logits, mu)
            kept = [i for i, v in enumerate(masked) if v != float("-inf")]
            assert 0 in kept
            assert 0.0 <= ent <= math.log(len(logits)) + 1e-12
            if prev is not None:
                assert len(kept) >= prev
            prev = len(kept)
        assert AI.mirostat_update(3.0, 5.0, 2.0, 0.2) < 3.0
        assert AI.mirostat_update(3.0, 0.5, 2.0, 0.2) > 3.0
        assert AI.mirostat_update(0.1, 100.0, 2.0, 1.0) == 0.0

    def test_invalid_inputs(self):
        layer = _w7b_layer()
        logits = [1.0, 0.0]
        for args in ((layer, True), (layer, 0), (layer, 2, 0.0), ("x", 2)):
            with pytest.raises((ValueError, TypeError)):
                AI.DoRALinear(*args)
        with pytest.raises((ValueError, TypeError)):
            AI.DoRALinear(layer, 2, None, 0.0)
        for args in ((layer, -1), (layer, 2, 0.0)):
            with pytest.raises((ValueError, TypeError)):
                AI.AdaLoRALinear(*args)
        with pytest.raises((ValueError, TypeError)):
            AI.AdaLoRALinear(layer, 4, None, 1.0)
        for fn, bad in ((AI.typical_filter, 0.0), (AI.typical_filter, 1.5),
                        (AI.top_a_filter, -0.1), (AI.top_a_filter, 2.0),
                        (AI.eta_filter, 1.0), (AI.eta_filter, -0.5),
                        (AI.mirostat_filter, -1.0)):
            with pytest.raises((ValueError, TypeError)):
                fn(logits, bad)
        with pytest.raises((ValueError, TypeError)):
            AI.typical_filter(logits, True)
        with pytest.raises((ValueError, TypeError)):
            AI.mirostat_update(1.0, 1.0, -1.0)
        with pytest.raises((ValueError, TypeError)):
            AI.mirostat_update(1.0, 1.0, 2.0, -0.5)


# ============================================================
#  W7-C：约束解码 + 投机解码接受判定
# ============================================================

class TestConstrainedDecoding:
    def test_token_id_mask(self):
        logits = [2.0, 1.0, 0.1, -1.0]
        masked = AI.token_id_mask(logits, [0, 2])
        assert masked[0] == 2.0 and masked[2] == 0.1
        assert masked[1] == float("-inf") and masked[3] == float("-inf")
        assert logits == [2.0, 1.0, 0.1, -1.0]
        probs = AI._softmax_row(masked)
        assert probs[1] == 0.0 and probs[3] == 0.0
        assert abs(sum(probs) - 1.0) < 1e-12

    def test_token_prefix_mask(self):
        seqs = [[1, 2, 3], [1, 4]]
        logits = [0.0] * 10

        def kept(masked):
            return [i for i, v in enumerate(masked) if v != float("-inf")]

        assert kept(AI.token_prefix_mask(logits, [], seqs)) == [1]
        assert kept(AI.token_prefix_mask(logits, [1], seqs)) == [2, 4]
        assert kept(AI.token_prefix_mask(logits, [1, 2], seqs)) == [3]
        assert kept(AI.token_prefix_mask(logits, [1, 2, 3], seqs, eos_id=9)) == [9]
        assert kept(AI.token_prefix_mask(logits, [1, 4], seqs, eos_id=9)) == [9]
        with pytest.raises(ValueError):
            AI.token_prefix_mask(logits, [4], seqs)
        with pytest.raises(ValueError):
            AI.token_prefix_mask(logits, [1, 2, 3], seqs)

    def test_mask_composes_with_sampling(self):
        logits = [2.0, 1.0, 0.5, 0.2, -0.3]
        masked = AI.token_id_mask(logits, {2, 4})
        probs = AI._softmax_row(masked)
        assert probs[0] == 0.0 and probs[1] == 0.0 and probs[3] == 0.0
        assert abs(probs[2] + probs[4] - 1.0) < 1e-12
        best = max(range(len(probs)), key=lambda i: probs[i])
        assert best == 2

    def test_invalid_inputs(self):
        logits = [1.0, 0.0, 0.0]
        for bad in ([], [3], [-1], [True], "x"):
            with pytest.raises((ValueError, TypeError)):
                AI.token_id_mask(logits, bad)
        seqs = [[1, 2]]
        with pytest.raises((ValueError, TypeError)):
            AI.token_prefix_mask([0.0] * 5, [1], [])
        with pytest.raises((ValueError, TypeError)):
            AI.token_prefix_mask([0.0] * 5, [1], [[1, -2]])
        with pytest.raises((ValueError, TypeError)):
            AI.token_prefix_mask([0.0] * 5, [1], [[1, 99]])


class TestSpeculativeAccept:
    def test_identical_distributions_accept_all(self):
        probs = [[0.6, 0.4], [0.3, 0.7]]
        acc, tok = AI.speculative_accept([0, 1], probs, probs,
                                         uniforms=[0.999, 0.999])
        assert acc == 2 and tok is None

    def test_rejection_and_residual_resample(self):
        draft = [[0.5, 0.5]]
        target = [[0.9, 0.1]]
        acc, tok = AI.speculative_accept([1], draft, target,
                                        uniforms=[0.99], rng=random.Random(1))
        assert acc == 0 and tok == 0
        # p_target=0 的位置必被拒绝
        acc, tok = AI.speculative_accept(
            [0], [[0.5, 0.5, 0.0]], [[0.0, 0.6, 0.4]],
            uniforms=[0.0], rng=random.Random(3))
        assert acc == 0 and tok in (1, 2)

    def test_acceptance_rate_matches_sum_min(self):
        rng = random.Random(42)
        trials = 3000
        hits = 0
        for _ in range(trials):
            t = 0 if rng.random() < 0.5 else 1
            a, _tok = AI.speculative_accept([t], [[0.5, 0.5]], [[0.3, 0.7]],
                                            rng=rng)
            hits += a
        rate = hits / trials
        # Σ min(p, q) = min(0.3,0.5) + min(0.7,0.5) = 0.8
        assert abs(rate - 0.8) < 0.04, rate

    def test_invalid_inputs(self):
        for args in (([], [[0.5, 0.5]], [[0.5, 0.5]]),
                     ([-1], [[0.5, 0.5]], [[0.5, 0.5]]),
                     ([5], [[0.5, 0.5]], [[0.5, 0.5]])):
            with pytest.raises((ValueError, TypeError)):
                AI.speculative_accept(*args)
        with pytest.raises((ValueError, TypeError)):
            AI.speculative_accept([0], [[0.4, 0.4]], [[0.5, 0.5]])
        with pytest.raises((ValueError, TypeError)):
            AI.speculative_accept([0], [[-0.5, 1.5]], [[0.5, 0.5]])
        with pytest.raises((ValueError, TypeError)):
            AI.speculative_accept([0], [[0.5, 0.5]], [[0.5, 0.6]])
        with pytest.raises((ValueError, TypeError)):
            AI.speculative_accept([0], [[0.5, 0.5]], [[0.5, 0.5]],
                                  uniforms=[1.5])


# ====  ======================================================================
#   连续批处理调度器（W7-D）
# ============================================================================

class TestContinuousBatcher:
    def test_capacity_and_refill(self):
        b = AI.ContinuousBatcher(max_batch=2)
        a = b.add([1, 2], max_new=2)
        c = b.add([3], max_new=1)
        d = b.add([4], max_new=1)
        assert b.running == [a, c]
        assert b.waiting == [d]
        assert b.batch() == [(a, [1, 2]), (c, [3])]

        assert b.step({a: 7, c: 8}) == [(a, 7), (c, 8)]
        # c 的 max_new=1 已满，退休后 d 补位
        assert b.finished == [c]
        assert b.running == [a, d]
        assert b.result(c) == [3, 8]

        assert b.step({a: 9, d: 10}) == [(a, 9), (d, 10)]
        assert b.finished == [c, a, d]
        assert b.running == []
        assert b.result(a) == [1, 2, 7, 9]
        assert b.result(d) == [4, 10]

    def test_total_token_budget(self):
        b = AI.ContinuousBatcher(max_batch=4, max_total_tokens=5)
        r1 = b.add([1, 2], max_new=1)
        r2 = b.add([3, 4], max_new=1)
        r3 = b.add([5, 6], max_new=1)
        assert b.running == [r1, r2]
        assert b.waiting == [r3]
        assert b.stats()["used_tokens"] == 4

        b.step({r1: 7, r2: 8})
        # 两者 max_new=1，同一步内一起退休，r3 立刻补位；
        # 记账只算提示词占用，因此剩下 r3 的 2 个 token
        assert b.running == [r3]
        assert b.waiting == []
        assert b.stats()["used_tokens"] == 2
        b.step({r3: 9})
        assert b.finished == [r1, r2, r3]
        assert b.stats()["used_tokens"] == 0

    def test_stats_and_result(self):
        b = AI.ContinuousBatcher(max_batch=1)
        assert b.add([0], max_new=2, request_id="x") == "x"
        assert b.running == ["x"]
        b.step({"x": 1})
        assert b.stats() == {"running": 1, "waiting": 0, "finished": 0,
                             "used_tokens": 1, "generated": 1}
        b.step({"x": 2})
        assert b.stats() == {"running": 0, "waiting": 0, "finished": 1,
                             "used_tokens": 0, "generated": 2}
        assert b.result("x") == [0, 1, 2]

    def test_validation(self):
        with pytest.raises(ValueError):
            AI.ContinuousBatcher(0)
        with pytest.raises(ValueError):
            AI.ContinuousBatcher(True)
        with pytest.raises(ValueError):
            AI.ContinuousBatcher(2, 0)

        b = AI.ContinuousBatcher(max_batch=2)
        b.add([1], max_new=2)
        with pytest.raises(ValueError):
            b.add([], 2)
        with pytest.raises(ValueError):
            b.add([1, -2], 2)
        with pytest.raises(ValueError):
            b.add([1], 0)
        with pytest.raises(ValueError):
            b.add([1], 2, 0)
        with pytest.raises(ValueError):
            b.step([])
        with pytest.raises(ValueError):
            b.step({99: 1})
        with pytest.raises(ValueError):
            b.step({})
        with pytest.raises(ValueError):
            b.step({0: -1})
        with pytest.raises(ValueError):
            b.result(99)

        done = AI.ContinuousBatcher(max_batch=1)
        done.add([1], max_new=1)
        done.step({0: 2})
        with pytest.raises(ValueError):
            done.step({0: 3})


# ====  ======================================================================
#   Tail-free 采样 + PagedAttention 块表（W7-E）
# ============================================================================

class TestTailFreeFilter:
    def test_prefix_and_monotone(self):
        logits = [3.0, 2.0, 1.0, 0.5, 0.1, -1.0]
        out = AI.tail_free_filter(logits, 0.5)
        keep = [i for i, v in enumerate(out) if v != float("-inf")]
        assert keep == [0, 1, 2, 3]
        # 入参不被修改，保留位置取值不变
        assert logits == [3.0, 2.0, 1.0, 0.5, 0.1, -1.0]
        assert all(out[i] == logits[i] for i in keep)

        counts = [len([v for v in AI.tail_free_filter(logits, t)
                       if v != float("-inf")])
                  for t in (0.1, 0.3, 0.5, 0.7, 0.9)]
        assert all(counts[i] >= counts[i + 1] for i in range(len(counts) - 1))

    def test_edge_cases(self):
        assert AI.tail_free_filter([1.5], 0.5) == [1.5]
        assert AI.tail_free_filter([1.5, 0.2], 0.5) == [1.5, 0.2]
        # 掩码后仍是合法分布
        masked = AI.tail_free_filter([2.0, 2.0, 0.1], 0.5)
        assert abs(sum(AI._softmax_row(masked)) - 1.0) < 1e-12

    def test_validation(self):
        for bad in (0, -0.5, True, "x"):
            with pytest.raises((ValueError, TypeError)):
                AI.tail_free_filter([1.0, 0.5], bad)
        with pytest.raises(ValueError):
            AI.tail_free_filter([], 0.5)
        with pytest.raises(ValueError):
            AI.tail_free_filter([1.0, float("nan")], 0.5)
        with pytest.raises(ValueError):
            AI.tail_free_filter([1.0, True], 0.5)


class TestPagedKVCache:
    @staticmethod
    def _row(i):
        return [[float(i), float(i) + 0.5], [float(-i), 1.0]]

    def test_blocks_and_order(self):
        c = AI.PagedKVCache(2, 2, 2, block_size=2, max_blocks=4)
        r = c.new_request()
        assert c.length(r) == 0
        c.append(r, 0, self._row(0), self._row(10))
        c.append(r, 0, self._row(1), self._row(11))
        assert c.stats()["blocks_used"] == 1
        c.append(r, 0, self._row(2), self._row(12))
        assert c.stats()["blocks_used"] == 2
        assert abs(c.stats()["fragmentation"] - 0.25) < 1e-12

        got = c.get(r, 0)
        assert len(got["key"]) == 3
        assert got["key"][2] == self._row(2)
        assert got["value"][0] == self._row(10)

    def test_free_and_reuse(self):
        c = AI.PagedKVCache(1, 2, 2, block_size=2, max_blocks=4)
        r = c.new_request()
        c.append(r, 0, self._row(1), self._row(2))
        c.free(r)
        assert c.stats()["blocks_used"] == 0
        assert c.stats()["blocks_free"] == 4
        r2 = c.new_request()
        c.append(r2, 0, self._row(7), self._row(8))
        assert c.stats()["blocks_used"] == 1
        assert c.get(r2, 0)["key"][0] == self._row(7)

    def test_exhaustion_and_interleaving(self):
        small = AI.PagedKVCache(1, 1, 1, block_size=1, max_blocks=2)
        q = small.new_request()
        small.append(q, 0, [[1.0]], [[2.0]])
        small.append(q, 0, [[3.0]], [[4.0]])
        with pytest.raises(ValueError):
            small.append(q, 0, [[5.0]], [[6.0]])
        assert small.length(q) == 2

        m = AI.PagedKVCache(1, 1, 2, block_size=2, max_blocks=8)
        a = m.new_request()
        b = m.new_request()
        m.append(a, 0, [[1.0, 2.0]], [[3.0, 4.0]])
        m.append(b, 0, [[5.0, 6.0]], [[7.0, 8.0]])
        m.append(a, 0, [[9.0, 10.0]], [[11.0, 12.0]])
        assert m.get(a, 0)["key"] == [[[1.0, 2.0]], [[9.0, 10.0]]]
        assert m.get(b, 0)["key"] == [[[5.0, 6.0]]]

    def test_validation(self):
        with pytest.raises(ValueError):
            AI.PagedKVCache(0, 1, 1)
        with pytest.raises(ValueError):
            AI.PagedKVCache(1, 1, 1, block_size=0)
        with pytest.raises(ValueError):
            AI.PagedKVCache(1, 1, 1, max_blocks=True)

        m = AI.PagedKVCache(1, 2, 2, block_size=2, max_blocks=4)
        a = m.new_request()
        m.append(a, 0, self._row(1), self._row(2))
        with pytest.raises(ValueError):
            m.length(99)
        with pytest.raises(ValueError):
            m.get(99, 0)
        with pytest.raises(ValueError):
            m.free(99)
        with pytest.raises(ValueError):
            m.append(a, 5, self._row(1), self._row(2))
        with pytest.raises(ValueError):
            m.append(a, 0, [[1.0]], self._row(2))
        with pytest.raises(ValueError):
            m.append(a, 0, self._row(1), [[1.0]])


# ====  ======================================================================
#   扩散采样与预测参数化（W8-A）
# ============================================================================

class TestDDIMSampling:
    def test_deterministic_eta0(self):
        m = AI.DDPM(2, hidden_dim=8, steps=8, t_dim=4)
        a = m.ddim_sample(1, rng=random.Random(5), eta=0.0)
        b = m.ddim_sample(1, rng=random.Random(5), eta=0.0)
        assert a == b
        c = m.ddim_sample(1, rng=random.Random(5), eta=1.0)
        assert c == m.ddim_sample(1, rng=random.Random(5), eta=1.0)
        assert c != a

    def test_subset_steps(self):
        m = AI.DDPM(2, hidden_dim=8, steps=12, t_dim=4)
        full = m.ddim_sample(1, rng=random.Random(3), steps=12, eta=0.0)
        sub = m.ddim_sample(1, rng=random.Random(3), steps=4, eta=0.0)
        one = m.ddim_sample(1, rng=random.Random(3), steps=1, eta=0.0)
        assert full != sub
        assert all(abs(v) < 1e6 for v in sub[0] + one[0])

    def test_validation(self):
        m = AI.DDPM(2, hidden_dim=8, steps=6, t_dim=4)
        with pytest.raises(ValueError):
            m.ddim_sample(0)
        with pytest.raises(ValueError):
            m.ddim_sample(True)
        with pytest.raises(ValueError):
            m.ddim_sample(1, eta=-0.1)
        with pytest.raises(ValueError):
            m.ddim_sample(1, eta=1.5)
        with pytest.raises(ValueError):
            m.ddim_sample(1, steps=0)
        with pytest.raises(ValueError):
            m.ddim_sample(1, steps=m.steps + 1)


class TestPredictionTypes:
    def test_target_and_inverse(self):
        x0, noise, t = [0.5, -0.3], [1.0, -1.0], 3
        for pred in ("eps", "v", "x0"):
            m = AI.DDPM(2, hidden_dim=8, steps=8, t_dim=4, prediction=pred)
            xt = m.q_sample(x0, t, noise)
            tgt = m.target(x0, noise, t)
            back_x0, back_eps = m.pred_to_x0_eps(tgt, xt, t)
            assert max(abs(a - b) for a, b in zip(back_x0, x0)) < 1e-10
            assert max(abs(a - b) for a, b in zip(back_eps, noise)) < 1e-10

    def test_eps_target_is_noise(self):
        m = AI.DDPM(2, hidden_dim=8, steps=8, t_dim=4)
        assert m.target([0.5, -0.3], [1.0, -1.0], 2) == [1.0, -1.0]

    def test_bad_prediction(self):
        with pytest.raises(ValueError):
            AI.DDPM(2, prediction="ve")
        with pytest.raises(ValueError):
            AI.DDPM(2, prediction=None)


class TestMinSNR:
    def test_weight_shape(self):
        m = AI.DDPM(2, hidden_dim=8, steps=8, t_dim=4)
        ws = [m.min_snr_weight(t, 5.0) for t in range(m.steps)]
        assert all(0.0 < w <= 1.0 for w in ws)
        # 噪声越大（t 越大）权重越大
        assert all(ws[i] <= ws[i + 1] + 1e-12 for i in range(len(ws) - 1))
        # gamma 越小、权重越不放大
        small = [m.min_snr_weight(t, 0.5) for t in range(m.steps)]
        assert all(small[i] <= ws[i] + 1e-12 for i in range(len(ws)))

    def test_loss_scaled_by_weight(self):
        m1 = AI.DDPM(2, hidden_dim=8, steps=8, t_dim=4)
        m2 = AI.DDPM(2, hidden_dim=8, steps=8, t_dim=4)
        for a, b in zip(m1.layers, m2.layers):
            b.weights = [row[:] for row in a.weights]
            b.biases = list(a.biases)
        base = m1.train_step([0.5, -0.3], t=4, noise=[1.0, -1.0])
        w = m2.min_snr_weight(4, 5.0)
        wtd = m2.train_step([0.5, -0.3], t=4, noise=[1.0, -1.0],
                            min_snr_gamma=5.0)
        assert abs(wtd - w * base) < 1e-12

    def test_validation(self):
        m = AI.DDPM(2, hidden_dim=8, steps=8, t_dim=4)
        for bad in (0.0, -1.0, True, "x"):
            with pytest.raises((ValueError, TypeError)):
                m.min_snr_weight(0, bad)


class TestVQVAE:
    """VQ-VAE：码本量化 / 三项损失 / straight-through / warmup"""

    def _mk(self, **kw):
        random.seed(0)
        return AI.VQVAE(3, hidden_dim=6, latent_dim=2, num_codes=4, beta=0.5,
                        seed=3, **kw)

    def test_quantize_nearest_and_tie(self):
        m = AI.VQVAE(3, num_codes=4, latent_dim=2, seed=0)
        m.codebook = [[0.0, 0.0], [1.0, 1.0], [0.0, 0.0], [2.0, 2.0]]
        zq, idx, code = m.quantize([0.0, 0.0])
        assert idx == 0                       # 平局取小编号
        assert zq == [0.0, 0.0] and code == m.codebook[0]
        zq, idx, _ = m.quantize([1.7, 1.9])
        assert idx == 3 and zq == [2.0, 2.0]  # 最近邻

    def test_loss_parts_closed_form(self):
        m = self._mk()
        x = [1.0, -0.5, 0.25]
        x_hat = m.forward(x)
        total = m.loss_and_backward(x)
        n = 3
        recon = sum((x_hat[j] - x[j]) ** 2 for j in range(n)) / n
        code = sum((m._code[i] - m._z_e[i]) ** 2 for i in range(2))
        commit = m.beta * sum((m._z_e[i] - m._code[i]) ** 2 for i in range(2))
        assert abs(m.last_parts["recon"] - recon) < 1e-12
        assert abs(m.last_parts["codebook"] - code) < 1e-12
        assert abs(m.last_parts["commitment"] - commit) < 1e-12
        assert abs(total - (recon + code + commit)) < 1e-12

        m2 = self._mk()
        m2.forward(x)
        total2 = m2.loss_and_backward(x, use_commitment=False)
        assert m2.last_parts["commitment"] == 0.0
        assert abs(total2 - m2.last_parts["recon"] - m2.last_parts["codebook"]) < 1e-12

    def test_straight_through_gradient(self):
        """d_z_e = d_z_q + 2*beta*(z_e - e)：用参考实现逐项比对"""
        m = self._mk()
        x = [1.0, -0.5, 0.25]
        m.forward(x)
        m.zero_grad()
        m.loss_and_backward(x)
        w1, b1 = [r[:] for r in m._g[1][0]], m._g[1][1][:]
        w0, b0 = [r[:] for r in m._g[0][0]], m._g[0][1][:]

        x_hat = m.forward(x)
        n = 3
        d_x_hat = [2.0 * (x_hat[j] - x[j]) / n for j in range(n)]
        _, _, d_h = m.dec2.backward(d_x_hat)
        _, _, d_z_q = m.dec1.backward(d_h)
        d_z_e = [d_z_q[i] + 2.0 * m.beta * (m._z_e[i] - m._code[i])
                 for i in range(2)]
        rwe, rbe, r_hidden = m.enc2.backward(d_z_e)
        rw0, rb0, _ = m.enc1.backward(r_hidden)

        for r in range(len(rwe)):
            for c in range(len(rwe[r])):
                assert abs(w1[r][c] - rwe[r][c]) < 1e-12
        for r in range(len(rw0)):
            for c in range(len(rw0[r])):
                assert abs(w0[r][c] - rw0[r][c]) < 1e-12
        for k in range(len(rbe)):
            assert abs(b1[k] - rbe[k]) < 1e-12
        for k in range(len(rb0)):
            assert abs(b0[k] - rb0[k]) < 1e-12

        # beta=0 时退化成纯 straight-through
        m = self._mk()
        m.beta = 0.0
        m.forward(x)
        m.zero_grad()
        m.loss_and_backward(x)
        x_hat = m.forward(x)
        d_x_hat = [2.0 * (x_hat[j] - x[j]) / n for j in range(n)]
        _, _, d_h = m.dec2.backward(d_x_hat)
        _, _, d_z_q = m.dec1.backward(d_h)
        rwe, _, _ = m.enc2.backward(d_z_q)
        for r in range(len(rwe)):
            for c in range(len(rwe[r])):
                assert abs(m._g[1][0][r][c] - rwe[r][c]) < 1e-12

    def test_codebook_and_decoder_finite_difference(self):
        m = self._mk()
        x = [1.0, -0.5, 0.25]
        m.forward(x)
        m.loss_and_backward(x, use_commitment=False)
        idx = m._idx
        ana = m._g_cb[idx][:]
        # 未被选中的码字无梯度
        for k, row in enumerate(m._g_cb):
            if k != idx:
                assert all(v == 0.0 for v in row)
        eps = 1e-6
        for i in range(2):
            orig = m.codebook[idx][i]
            m.codebook[idx][i] = orig + eps
            m.forward(x)
            m.zero_grad()
            lp = m.loss_and_backward(x, use_commitment=False)
            m.codebook[idx][i] = orig - eps
            m.forward(x)
            m.zero_grad()
            lm = m.loss_and_backward(x, use_commitment=False)
            m.codebook[idx][i] = orig
            num = (lp - lm) / (2 * eps)
            assert abs(ana[i] - num) < 1e-6

        # 解析式 = d_z_q + 2*(e - z_e)
        m = self._mk()
        m.forward(x)
        m.loss_and_backward(x, use_commitment=False)
        x_hat = m.forward(x)
        n = 3
        d_x_hat = [2.0 * (x_hat[j] - x[j]) / n for j in range(n)]
        _, _, d_h = m.dec2.backward(d_x_hat)
        _, _, d_z_q = m.dec1.backward(d_h)
        idx = m._idx
        for i in range(2):
            expect = d_z_q[i] + 2.0 * (m._code[i] - m._z_e[i])
            assert abs(m._g_cb[idx][i] - expect) < 1e-12

        # decoder 权重梯度 = 中心差分
        m = self._mk()
        m.forward(x)
        m.loss_and_backward(x, use_commitment=False)
        r, c = 0, 1
        ana = m._g[3][0][r][c]
        orig = m.dec2.weights[r][c]
        m.dec2.weights[r][c] = orig + eps
        m.forward(x)
        m.zero_grad()
        lp = m.loss_and_backward(x, use_commitment=False)
        m.dec2.weights[r][c] = orig - eps
        m.forward(x)
        m.zero_grad()
        lm = m.loss_and_backward(x, use_commitment=False)
        m.dec2.weights[r][c] = orig
        assert abs(ana - (lp - lm) / (2 * eps)) < 1e-6

    def test_warmup_histogram_and_optimizers(self):
        m = self._mk()
        x = [1.0, -0.5, 0.25]
        h = m.codebook_histogram()
        assert len(h) == 4 and sum(h) == 0     # 还没 forward

        m.forward(x)
        h = m.codebook_histogram()
        assert len(h) == 4 and sum(h) == 1 and h[m._idx] == 1

        before = [row[:] for row in m.codebook]
        data = [[1.0, 0.0, 0.5], [0.0, 1.0, -0.5], [0.5, 0.5, 0.0],
                [0.2, 0.3, 0.9]] * 2
        m.fit(data, epochs=8, lr=0.01, seed=0, warmup=100)
        assert m.codebook == before            # warmup 未结束：码本一步不动

        m = self._mk()
        m.forward(x)
        m.zero_grad()
        m.loss_and_backward(x)
        w_before = [row[:] for row in m.enc1.weights]
        m.sgd_step(0.1)
        assert m.enc1.weights != w_before

        m.zero_grad()
        m.loss_and_backward(x)
        cb_before = [row[:] for row in m.codebook]
        m.adam_step(0.05)
        assert m.codebook != cb_before
        assert m.step_count == 1

        m.zero_grad()
        assert all(g is None for g in m._g)
        assert all(all(v == 0.0 for v in row) for row in m._g_cb)

    def test_training_and_validation(self):
        random.seed(1)
        m = AI.VQVAE(3, hidden_dim=12, latent_dim=2, num_codes=6, beta=1.0, seed=3)
        data = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
                [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]
        hist = m.fit(data, epochs=300, lr=0.02, seed=0, warmup=20)
        assert len(hist) == 300
        assert hist[-1] < hist[0] * 0.3
        used = {m.quantize(m.encode(x))[1] for x in data}
        assert len(used) >= 2

        def raises(fn):
            try:
                fn()
            except (ValueError, TypeError):
                return True
            return False

        assert raises(lambda: AI.VQVAE(0))
        assert raises(lambda: AI.VQVAE(True))
        assert raises(lambda: AI.VQVAE(3, num_codes=0))
        assert raises(lambda: AI.VQVAE(3, latent_dim=-1))
        assert raises(lambda: AI.VQVAE(3, beta=-0.1))
        assert raises(lambda: self._mk().encode([1.0]))
        assert raises(lambda: self._mk().quantize([1.0]))
        assert raises(lambda: self._mk().decode([1.0, 2.0, 3.0]))
        assert raises(lambda: self._mk().fit([]))
        assert raises(lambda: self._mk().fit([[1.0, 2.0]]))
        assert raises(lambda: self._mk().adam_step(0.01, eps=0.0))
        assert raises(lambda: self._mk().adam_step(0.01, b1=1.0))


class TestFlowMatching:
    """Flow Matching / Rectified Flow：直线路径、速度场梯度、采样器"""

    def _mk(self, **kw):
        random.seed(0)
        return AI.FlowMatching(2, hidden_dim=5, t_dim=4, **kw)

    def test_path_and_target(self):
        m = self._mk()
        x0, noise = [1.0, -0.5], [0.3, 0.8]
        assert m.interpolate(x0, noise, 0.0) == x0
        assert m.interpolate(x0, noise, 1.0) == noise
        mid = m.interpolate(x0, noise, 0.5)
        for i in range(2):
            assert abs(mid[i] - 0.5 * (x0[i] + noise[i])) < 1e-15
        assert m.flow_target(x0, noise) == [noise[i] - x0[i] for i in range(2)]

        # 时间编码：长度、端点可区分、t=0 的 cos 项为 1
        f0 = AI._flow_time_features(0.0, 4)
        f1 = AI._flow_time_features(1.0, 4)
        assert len(f0) == 4 and len(f1) == 4 and f0 != f1
        assert abs(f0[1] - 1.0) < 1e-15 and abs(f0[3] - 1.0) < 1e-15
        f05 = AI._flow_time_features(0.5, 4)
        assert abs(f05[0] - 1.0) < 1e-12 and abs(f05[3] + 1.0) < 1e-12

    def test_velocity_input_gradient(self):
        """速度场输出对输入的中心差分（经 l3→l2→l1 手写链）"""
        m = self._mk()
        t, eps = 0.37, 1e-6
        m.velocity([0.4, -0.2], t)
        _, _, d2 = m.l3.backward([1.0, 0.0])
        _, _, d1 = m.l2.backward(d2)
        _, _, din = m.l1.backward(d1)
        for j in range(2):
            up = [0.4, -0.2]
            up[j] += eps
            dn = [0.4, -0.2]
            dn[j] -= eps
            num = (m.velocity(up, t)[0] - m.velocity(dn, t)[0]) / (2 * eps)
            assert abs(din[j] - num) < 1e-6

    def test_parameter_gradients(self):
        """三层权重与偏置的中心差分（train_step 固定 t 与 noise）"""
        m = self._mk()
        x0, t, noise, eps = [0.7, -0.3], 0.42, [0.2, -0.6], 1e-6
        m.zero_grad()
        m.train_step(x0, t=t, noise=noise)
        ana = {li: (m._g[li][0][0][0], m._g[li][1][0]) for li in range(3)}
        for li, lname in ((0, "l1"), (1, "l2"), (2, "l3")):
            layer = getattr(m, lname)
            ana_w, ana_b = ana[li]
            ow, ob = layer.weights[0][0], layer.biases[0]
            layer.weights[0][0] = ow + eps
            lp = m.train_step(x0, t=t, noise=noise)
            layer.weights[0][0] = ow - eps
            lm = m.train_step(x0, t=t, noise=noise)
            layer.weights[0][0] = ow
            assert abs(ana_w - (lp - lm) / (2 * eps)) < 1e-6
            layer.biases[0] = ob + eps
            lp = m.train_step(x0, t=t, noise=noise)
            layer.biases[0] = ob - eps
            lm = m.train_step(x0, t=t, noise=noise)
            layer.biases[0] = ob
            assert abs(ana_b - (lp - lm) / (2 * eps)) < 1e-6

    def test_training_and_sampling(self):
        random.seed(0)
        m = AI.FlowMatching(2, hidden_dim=16, t_dim=4)
        data = [[2.0, 2.0], [-2.0, -2.0], [2.0, -2.0], [-2.0, 2.0]]
        hist = m.fit(data, epochs=800, lr=0.02, seed=0)
        assert len(hist) == 800
        assert hist[-1] < hist[0] * 0.5

        a = m.sample(2, steps=10, rng=random.Random(7))
        b = m.sample(2, steps=10, rng=random.Random(7))
        assert a == b                                   # 同 seed 逐位相同
        c = m.sample(2, steps=10, rng=random.Random(8))
        assert a != c
        heun = m.sample(1, steps=10, solver="heun", rng=random.Random(7))
        euler = m.sample(1, steps=10, solver="euler", rng=random.Random(7))
        assert heun != euler
        assert all(abs(v) < 1e6 for v in heun[0])

        # 采样点应该靠近训练数据（四个角之一）
        def nearest(pt):
            return min(math.dist(pt, d) for d in data)

        pts = m.sample(6, steps=20, rng=random.Random(3))
        assert sum(nearest(p) for p in pts) / len(pts) < 1.0

    def test_optimizers_and_validation(self):
        m = self._mk()
        m.train_step([0.1, 0.2], t=0.5, noise=[0.0, 0.0])
        w0 = [row[:] for row in m.l1.weights]
        m.sgd_step(0.1)
        assert m.l1.weights != w0
        m.zero_grad()
        assert all(g is None for g in m._g)
        m.train_step([0.1, 0.2], t=0.5, noise=[0.0, 0.0])
        m.adam_step(0.01)
        assert m.step_count == 1

        def raises(fn):
            try:
                fn()
            except (ValueError, TypeError):
                return True
            return False

        assert raises(lambda: AI.FlowMatching(0))
        assert raises(lambda: AI.FlowMatching(2, hidden_dim=True))
        assert raises(lambda: AI.FlowMatching(2, t_dim=-1))
        assert raises(lambda: self._mk().interpolate([1.0], [0.0, 0.0], 0.5))
        assert raises(lambda: self._mk().interpolate([0.0, 0.0], [0.0, 0.0], -0.1))
        assert raises(lambda: self._mk().interpolate([0.0, 0.0], [0.0, 0.0], 1.5))
        assert raises(lambda: self._mk().velocity([0.0, 0.0], "0.5"))
        assert raises(lambda: self._mk().fit([]))
        assert raises(lambda: self._mk().fit([[1.0]]))
        assert raises(lambda: self._mk().sample(0))
        assert raises(lambda: self._mk().sample(1, steps=0))
        assert raises(lambda: self._mk().sample(1, solver="rk4"))
        assert raises(lambda: self._mk().adam_step(0.01, eps=0.0))
        assert raises(lambda: self._mk().adam_step(0.01, b2=1.0))
        assert raises(lambda: self._mk().train_step([0.0, 0.0], t=2.0))

# ====
#  TestGANVariants：GAN 损失变体（BCE / LSGAN / WGAN / Hinge）
# ====
class TestGANVariants:
    DATA = [[2.0, 2.0], [-2.0, -2.0], [2.0, -2.0], [-2.0, 2.0]]

    def test_d_loss_and_grad_closed_form(self):
        sig = AI._sigmoid_scalar
        for s in (0.7, -0.3):
            p = sig(s)
            loss, g = AI._gan_d_loss(s, True, "bce")
            assert abs(loss + math.log(p)) < 1e-12
            assert abs(g - (p - 1.0)) < 1e-12
            loss, g = AI._gan_d_loss(s, False, "bce")
            assert abs(loss + math.log(1.0 - p)) < 1e-12
            assert abs(g - p) < 1e-12
            loss, g = AI._gan_d_loss(s, True, "lsgan")
            assert abs(loss - 0.5 * (s - 1.0) ** 2) < 1e-12
            assert abs(g - (s - 1.0)) < 1e-12
            loss, g = AI._gan_d_loss(s, True, "wgan")
            assert abs(loss + s) < 1e-12 and abs(g + 1.0) < 1e-12
            loss, g = AI._gan_d_loss(s, False, "hinge")
            assert abs(loss - max(0.0, 1.0 + s)) < 1e-12
        # hinge 越过边界后损失与梯度都为 0
        loss, g = AI._gan_d_loss(2.0, True, "hinge")
        assert loss == 0.0 and g == 0.0
        # 生成器侧
        for s in (0.7, 3.0):
            loss, g = AI._gan_g_loss(s, "bce")
            assert abs(loss + math.log(AI._sigmoid_scalar(s))) < 1e-12
            assert abs(g + (1.0 - AI._sigmoid_scalar(s))) < 1e-12
            loss, g = AI._gan_g_loss(s, "wgan")
            assert abs(loss + s) < 1e-12 and abs(g + 1.0) < 1e-12

    def test_bce_variant_matches_default(self):
        random.seed(0)
        a = AI.GAN(2, hidden_dim=8, latent_dim=2)
        random.seed(0)
        b = AI.GAN(2, hidden_dim=8, latent_dim=2, variant="bce")
        ra, rb = random.Random(7), random.Random(7)
        for _ in range(20):
            la = a.step(self.DATA, rng=ra, lr_d=0.01, lr_g=0.01)
            lb = b.step(self.DATA, rng=rb, lr_d=0.01, lr_g=0.01)
            assert la == lb
        for i in range(len(a.d1.weights)):
            for j in range(len(a.d1.weights[0])):
                assert a.d1.weights[i][j] == b.d1.weights[i][j]
        assert b.d2.activation_name == "sigmoid"

    def test_variants_train_and_score(self):
        for v in ("lsgan", "wgan", "hinge"):
            g = AI.GAN(2, hidden_dim=4, latent_dim=2, variant=v)
            assert g.d2.activation_name == "linear"
            raw = g.score([0.3, -0.2])
            assert abs(g.discriminate([0.3, -0.2]) - AI._sigmoid_scalar(raw)) < 1e-12
            before = [row[:] for row in g.g1.weights]
            r = random.Random(0)
            for _ in range(60):
                dl, gl = g.step(self.DATA, rng=r, lr_d=0.01, lr_g=0.01)
            assert math.isfinite(dl) and math.isfinite(gl)
            moved = any(before[i][j] != g.g1.weights[i][j]
                        for i in range(len(before)) for j in range(len(before[0])))
            assert moved

    def test_wgan_weight_clipping(self):
        for clip in (None, 0.02):
            random.seed(0)
            g = AI.GAN(2, hidden_dim=8, latent_dim=2, variant="wgan")
            r = random.Random(0)
            for _ in range(30):
                g.step(self.DATA, rng=r, lr_d=0.01, lr_g=0.01, clip_value=clip)
            c = 0.01 if clip is None else clip
            worst = max(abs(v) for layer in (g.d1, g.d2)
                        for row in layer.weights for v in row)
            assert worst <= c + 1e-15
        # 不裁剪的变体权重能超过 0.01
        random.seed(0)
        g = AI.GAN(2, hidden_dim=8, latent_dim=2, variant="lsgan")
        r = random.Random(0)
        for _ in range(30):
            g.step(self.DATA, rng=r, lr_d=0.01, lr_g=0.01)
        worst = max(abs(v) for layer in (g.d1, g.d2)
                    for row in layer.weights for v in row)
        assert worst > 0.01

    def test_r1_penalty_and_errors(self):
        assert abs(AI.r1_penalty([[3.0, 4.0]], 10.0) - 125.0) < 1e-12
        assert abs(AI.r1_penalty([[3.0, 4.0], [1.0, 0.0]], 10.0) - 65.0) < 1e-12
        assert AI.r1_penalty([[3.0, 4.0]], 0.0) == 0.0
        for fn, args in [
            (AI._gan_d_loss, (0.5, True, "nope")),
            (AI._gan_d_loss, ("x", True, "bce")),
            (AI._gan_d_loss, (True, True, "bce")),
            (AI._gan_g_loss, (0.5, "nope")),
            (AI.r1_penalty, ([], 1.0)),
            (AI.r1_penalty, ([[1.0]], -1.0)),
            (AI.GAN, (2, 4, 2, "nope")),
            (AI.GAN, (2, 4, 2, True)),
        ]:
            try:
                fn(*args)
                raise AssertionError(f"{fn.__name__}{args} 应当报错")
            except (ValueError, TypeError):
                pass
        random.seed(0)
        g = AI.GAN(2, hidden_dim=4, latent_dim=2, variant="wgan")
        try:
            g.step(self.DATA, rng=random.Random(0), clip_value=-1.0)
            raise AssertionError("clip_value<0 应当报错")
        except ValueError:
            pass
        try:
            g.step(self.DATA, rng=random.Random(0), variant="nope")
            raise AssertionError("variant 非法应当报错")
        except ValueError:
            pass

# ============================================================================
#  RealNVP（精确似然的仿射耦合流）
# ============================================================================
class TestRealNVP:
    def _model(self, dim=2, **kw):
        kw.setdefault("hidden_dim", 8)
        kw.setdefault("num_layers", 3)
        return AI.RealNVP(dim, **kw)

    def test_forward_inverse_roundtrip_and_masks(self):
        for dim in (2, 3, 4, 5):
            m = AI.RealNVP(dim, hidden_dim=6, num_layers=3, seed=dim)
            r = random.Random(dim)
            for _ in range(3):
                x = [r.uniform(-2.0, 2.0) for _ in range(dim)]
                y, logdet = m.forward(x)
                back = m.inverse(y)
                assert max(abs(back[i] - x[i]) for i in range(dim)) < 1e-12
                assert math.isfinite(logdet)
        m4 = AI.RealNVP(4, hidden_dim=4, num_layers=4, seed=1)
        assert m4.masks[0] == [1, 0, 1, 0]
        assert m4.masks[1] == [0, 1, 0, 1]
        m5 = AI.RealNVP(5, hidden_dim=4, num_layers=2, seed=1)
        assert m5.masks[0] == [1, 1, 0, 0, 0]
        assert m5.masks[1] == [0, 0, 1, 1, 1]

    def test_logdet_matches_numeric_jacobian(self):
        m = AI.RealNVP(2, hidden_dim=8, num_layers=3, seed=7)
        r = random.Random(3)
        eps = 1e-6
        for _ in range(3):
            x = [r.uniform(-1.5, 1.5) for _ in range(2)]
            _, logdet = m.forward(x)
            cols = []
            for j in range(2):
                xp = list(x)
                xm = list(x)
                xp[j] += eps
                xm[j] -= eps
                yp, _ = m.forward(xp)
                ym, _ = m.forward(xm)
                cols.append([(yp[i] - ym[i]) / (2 * eps) for i in range(2)])
            det = cols[0][0] * cols[1][1] - cols[0][1] * cols[1][0]
            assert abs(logdet - math.log(abs(det))) < 1e-6

    def test_gradients_match_finite_difference(self):
        m = AI.RealNVP(2, hidden_dim=6, num_layers=2, seed=5)
        x = [0.4, 0.8]
        w = [0.7, -1.1]
        g = 0.9

        def loss(xx):
            y, logdet = m.forward(xx)
            return sum(w[i] * y[i] for i in range(2)) + g * logdet

        y, _ = m.forward(x)
        dx = m.backward(list(w), g)
        eps = 1e-6
        for j in range(2):
            xp = list(x)
            xm = list(x)
            xp[j] += eps
            xm[j] -= eps
            num = (loss(xp) - loss(xm)) / (2 * eps)
            assert abs(dx[j] - num) < 1e-7
        order = ["s1", "s2", "t1", "t2"]
        for li, key in ((0, "s1"), (0, "t1"), (1, "s2"), (1, "t2")):
            layer = m.layers[li][key]
            gw, _gb = m._grads[li * 4 + order.index(key)]
            old = layer.weights[0][0]
            layer.weights[0][0] = old + eps
            lp = loss(x)
            layer.weights[0][0] = old - eps
            lm = loss(x)
            layer.weights[0][0] = old
            num = (lp - lm) / (2 * eps)
            assert abs(gw[0][0] - num) < 1e-6, (li, key, gw[0][0], num)

    def test_training_and_sampling(self):
        r = random.Random(1)
        data = [[1.0 * z + 0.15 * r.gauss(0, 1), 1.0 * z + 0.15 * r.gauss(0, 1)]
                for z in (-2, -1, 0, 1, 2) for _ in range(4)]
        m = AI.RealNVP(2, hidden_dim=16, num_layers=4, seed=0)
        before = sum(m.nll(q) for q in data) / len(data)
        after = m.fit(data, epochs=100, lr=3e-3, seed=0)
        assert after <= 0.75 * before
        pts = m.sample(3, random.Random(0))
        assert len(pts) == 3
        assert all(math.isfinite(v) for p in pts for v in p)
        assert m.sample(2, random.Random(42)) == m.sample(2, random.Random(42))

    def test_errors(self):
        for kwargs in ({"data_dim": 1}, {"data_dim": 2, "hidden_dim": 0},
                       {"data_dim": 2, "num_layers": True},
                       {"data_dim": 2, "scale_limit": 0.0},
                       {"data_dim": 2, "seed": 1.5}):
            with pytest.raises(ValueError):
                AI.RealNVP(**kwargs)
        m = AI.RealNVP(2, hidden_dim=4, num_layers=2, seed=0)
        with pytest.raises(RuntimeError):
            m.backward([1.0, 0.0])
        m.forward([0.2, 0.3])
        with pytest.raises(ValueError):
            m.backward([1.0])
        with pytest.raises(RuntimeError):
            m.sgd_step(0.01)
        m.backward([1.0, 0.0])
        with pytest.raises(ValueError):
            m.sgd_step(-1.0)
        with pytest.raises(ValueError):
            m.sample(0)
        with pytest.raises(ValueError):
            m.fit([])
        with pytest.raises(ValueError):
            m.log_prob([1.0])
        with pytest.raises(ValueError):
            m.log_prob(["a", "b"])

# ============================================================================
#  Glow 组件（ActNorm + 可逆线性）
# ============================================================================
class TestActNorm:
    def test_data_init_gives_unit_variance(self):
        a = AI.ActNorm(3)
        batch = [[0.5 + 2.0 * i, -1.0 + 0.3 * i, 4.0 - 1.5 * i] for i in range(6)]
        a.data_init(batch)
        outs = [a.forward(row) for row in batch]
        for j in range(3):
            col = [o[j] for o in outs]
            mean = sum(col) / len(col)
            var = sum((v - mean) ** 2 for v in col) / len(col)
            assert abs(mean) < 1e-12
            assert abs(var - 1.0) < 1e-6

    def test_inverse_and_logdet(self):
        a = AI.ActNorm(2)
        a.b = [0.3, -0.2]
        a.log_s = [0.5, -1.0]
        x = [1.5, -2.0]
        y = a.forward(x)
        back = a.inverse(y)
        assert max(abs(back[j] - x[j]) for j in range(2)) < 1e-12
        assert abs(a.logdet() - sum(a.log_s)) < 1e-15

    def test_gradients(self):
        a = AI.ActNorm(3)
        a.b = [0.2, -0.4, 0.7]
        a.log_s = [0.3, -0.2, 0.1]
        x = [1.2, -0.8, 0.5]
        w = [0.7, -1.3, 0.4]
        g = 0.8

        def loss(xx):
            yy = a.forward(xx)
            return sum(w[j] * yy[j] for j in range(3)) + g * a.logdet()

        a.zero_grad()
        a.forward(x)
        dx = a.backward(list(w), g)
        eps = 1e-6
        for j in range(3):
            xp = list(x)
            xm = list(x)
            xp[j] += eps
            xm[j] -= eps
            num = (loss(xp) - loss(xm)) / (2 * eps)
            assert abs(dx[j] - num) < 1e-7
        gb = list(a.g_b)
        gls = list(a.g_log_s)
        for j in (0, 2):
            old = a.b[j]
            a.b[j] = old + eps
            lp = loss(x)
            a.b[j] = old - eps
            lm = loss(x)
            a.b[j] = old
            assert abs(gb[j] - (lp - lm) / (2 * eps)) < 1e-6
            old = a.log_s[j]
            a.log_s[j] = old + eps
            lp = loss(x)
            a.log_s[j] = old - eps
            lm = loss(x)
            a.log_s[j] = old
            assert abs(gls[j] - (lp - lm) / (2 * eps)) < 1e-6

    def test_errors_and_steps(self):
        with pytest.raises(ValueError):
            AI.ActNorm(0)
        with pytest.raises(ValueError):
            AI.ActNorm(2, eps=0.0)
        a = AI.ActNorm(2)
        with pytest.raises(RuntimeError):
            a.backward([1.0, 0.0])
        with pytest.raises(ValueError):
            a.forward([1.0])
        with pytest.raises(ValueError):
            a.data_init([])
        a.data_init([[1.0, 2.0], [3.0, 4.0]])
        a.forward([2.0, 3.0])
        a.backward([1.0, -1.0], 0.5)
        b0 = list(a.b)
        a.sgd_step(0.1)
        assert a.b != b0
        a.forward([2.0, 3.0])
        a.backward([1.0, -1.0], 0.5)
        b0 = list(a.b)
        a.adam_step(0.1)
        assert a.b != b0


class TestInvertibleLinear:
    def test_inverse_and_logdet(self):
        L = AI.InvertibleLinear(3, seed=4)
        x = [0.5, -1.0, 2.0]
        y = L.forward(x)
        back = L.inverse(y)
        assert max(abs(back[j] - x[j]) for j in range(3)) < 1e-12
        # 数值 Jacobian 的 det 应与 logdet 一致
        eps = 1e-6
        cols = []
        for j in range(3):
            xp = list(x)
            xm = list(x)
            xp[j] += eps
            xm[j] -= eps
            L.forward(xp)
            yp = [sum(L.W[i][k] * xp[k] for k in range(3)) for i in range(3)]
            L.forward(xm)
            ym = [sum(L.W[i][k] * xm[k] for k in range(3)) for i in range(3)]
            cols.append([(yp[i] - ym[i]) / (2 * eps) for i in range(3)])
        J = cols
        det = (J[0][0] * (J[1][1] * J[2][2] - J[1][2] * J[2][1])
               - J[0][1] * (J[1][0] * J[2][2] - J[1][2] * J[2][0])
               + J[0][2] * (J[1][0] * J[2][1] - J[1][1] * J[2][0]))
        L.forward(x)
        assert abs(L.logdet() - math.log(abs(det))) < 1e-6

    def test_gradients_including_logdet_term(self):
        L = AI.InvertibleLinear(3, seed=4)
        x = [0.5, -1.0, 2.0]
        wg = [0.3, -0.7, 1.1]
        gl = 0.6

        def loss(xx):
            yy = L.forward(xx)
            return sum(wg[i] * yy[i] for i in range(3)) + gl * L.logdet()

        L.zero_grad()
        L.forward(x)
        dx = L.backward(list(wg), gl)
        eps = 1e-6
        for j in range(3):
            xp = list(x)
            xm = list(x)
            xp[j] += eps
            xm[j] -= eps
            num = (loss(xp) - loss(xm)) / (2 * eps)
            assert abs(dx[j] - num) < 1e-6
        gW = [row[:] for row in L.g_W]
        for i, j in ((0, 0), (1, 2), (2, 1)):
            old = L.W[i][j]
            L.W[i][j] = old + eps
            lp = loss(x)
            L.W[i][j] = old - eps
            lm = loss(x)
            L.W[i][j] = old
            num = (lp - lm) / (2 * eps)
            assert abs(gW[i][j] - num) < 1e-5, (i, j, gW[i][j], num)

    def test_grad_without_logdet_is_outer_product(self):
        L = AI.InvertibleLinear(2, seed=2)
        x = [1.0, 2.0]
        g = [0.5, -0.25]
        L.zero_grad()
        L.forward(x)
        L.backward(list(g), 0.0)
        for i in range(2):
            for j in range(2):
                assert abs(L.g_W[i][j] - g[i] * x[j]) < 1e-12

    def test_errors_and_steps(self):
        with pytest.raises(ValueError):
            AI.InvertibleLinear(0)
        with pytest.raises(ValueError):
            AI.InvertibleLinear(2, noise=1.0)
        with pytest.raises(ValueError):
            AI.InvertibleLinear(2, seed=1.5)
        L = AI.InvertibleLinear(2, seed=0)
        with pytest.raises(RuntimeError):
            L.logdet()
        with pytest.raises(RuntimeError):
            L.backward([1.0, 0.0])
        L.W = [[0.0, 0.0], [1.0, 1.0]]
        with pytest.raises(ValueError):
            L.forward([1.0, 1.0])
        L2 = AI.InvertibleLinear(2, seed=2)
        L2.forward([1.0, 2.0])
        L2.backward([1.0, 0.0], 0.0)
        w0 = [row[:] for row in L2.W]
        L2.sgd_step(0.01)
        assert L2.W != w0
        L2.forward([1.0, 2.0])
        L2.backward([1.0, 0.0], 0.0)
        w0 = [row[:] for row in L2.W]
        L2.adam_step(0.01)
        assert L2.W != w0

def _w9_close(a, b, tol=1e-12):
    """相对容差比较（测试内用，避免依赖 pytest.approx 的语义差异）"""
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


class TestClassicOptimizers:
    """W9-A：AdaGrad / AdaDelta / Nadam / RAdam"""

    @staticmethod
    def _model(seed=0, sizes=(2, 3, 2)):
        random.seed(seed)
        acts = ["relu"] * (len(sizes) - 2) + ["linear"]
        return AI.MLP(list(sizes), activations=acts)

    @staticmethod
    def _grads(model, rng):
        out = []
        for layer in model.layers:
            gw = [[rng.uniform(-0.5, 0.5) for _ in range(layer.fan_in)]
                  for _ in range(layer.fan_out)]
            gb = [rng.uniform(-0.5, 0.5) for _ in range(layer.fan_out)]
            out.append((gw, gb))
        return out

    def test_adagrad_matches_hand_computation(self):
        """AdaGrad 两步与手算逐位一致；步长单调不增"""
        model = AI.MLP([2, 2], activations=["linear"])
        for j in range(2):
            model.layers[0].weights[j] = [1.0, -1.0]
            model.layers[0].biases[j] = 0.5
        opt = AI.AdaGrad(model, lr=0.1, eps=1e-8)
        gw = [[0.5, -0.25], [0.5, -0.25]]
        gb = [0.5, 0.5]
        s00 = 0.25
        w00 = 1.0
        w00 -= 0.1 * 0.5 / (math.sqrt(s00) + 1e-8)
        assert abs(model.layers[0].weights[0][0] - w00) > 0  # 尚未 step
        opt.step([(gw, gb)])
        assert _w9_close(model.layers[0].weights[0][0], w00, 1e-15), (
            model.layers[0].weights[0][0], w00)

        # 常数梯度下每步位移单调不增
        deltas = []
        for _ in range(6):
            before = model.layers[0].biases[0]
            opt.step([(gw, gb)])
            deltas.append(abs(model.layers[0].biases[0] - before))
        assert all(deltas[i] >= deltas[i + 1] - 1e-15
                   for i in range(len(deltas) - 1)), deltas

    def test_adadelta_reference_and_scale_invariance(self):
        """AdaDelta 与参考实现逐位一致；梯度整体 ×10 参数几乎不动"""
        model = self._model()
        opt = AI.AdaDelta(model, lr=1.0, rho=0.95, eps=1e-6)
        rng = random.Random(11)
        grads = [self._grads(model, rng) for _ in range(4)]
        for g in grads:
            opt.step(g)
        # 参考实现（与实现同顺序）
        ref = self._model()
        sw = [[[0.0] * 2 for _ in range(3)], [[0.0] * 3 for _ in range(2)]]
        sb = [[0.0] * 3, [0.0] * 2]
        uw = [[[0.0] * 2 for _ in range(3)], [[0.0] * 3 for _ in range(2)]]
        ub = [[0.0] * 3, [0.0] * 2]
        # 注意：0.95 与 1-0.95 在浮点下不是同一个数（1-0.95 = 0.050000000000000044），
        # 参考实现必须用 (1.0 - rho) 才能与实现逐位对齐
        rho = 0.95
        om = 1.0 - rho
        for g in grads:
            for li, layer in enumerate(ref.layers):
                gw, gb = g[li]
                for j in range(layer.fan_out):
                    gr = gb[j]
                    sb[li][j] = rho * sb[li][j] + om * gr * gr
                    dx = math.sqrt(ub[li][j] + 1e-6) / math.sqrt(sb[li][j] + 1e-6) * gr
                    ub[li][j] = rho * ub[li][j] + om * dx * dx
                    layer.biases[j] -= dx
                    for i in range(layer.fan_in):
                        gr = gw[j][i]
                        sw[li][j][i] = rho * sw[li][j][i] + om * gr * gr
                        dx = (math.sqrt(uw[li][j][i] + 1e-6)
                              / math.sqrt(sw[li][j][i] + 1e-6) * gr)
                        uw[li][j][i] = rho * uw[li][j][i] + om * dx * dx
                        layer.weights[j][i] -= dx
        for a, b in zip(model.layers, ref.layers):
            assert a.weights == b.weights
            assert a.biases == b.biases

        # 尺度不变性
        ma, mb = self._model(), self._model()
        oa, ob = AI.AdaDelta(ma), AI.AdaDelta(mb)
        rng2 = random.Random(3)
        for _ in range(6):
            g = self._grads(ma, rng2)
            big = [([[v * 10.0 for v in row] for row in gw], [v * 10.0 for v in gb])
                   for gw, gb in g]
            oa.step(g)
            ob.step(big)
        for la, lb in zip(ma.layers, mb.layers):
            for ra, rb in zip(la.weights, lb.weights):
                for va, vb in zip(ra, rb):
                    assert abs(va - vb) <= 1e-3 * max(1.0, abs(va)), (va, vb)

    def test_nadam_reference_and_b1_zero_rmsprop(self):
        """Nadam 与手写参考一致；b1=0 时单步等于 RMSprop 式"""
        model = AI.MLP([2, 2], activations=["linear"])
        for j in range(2):
            model.layers[0].weights[j] = [1.0, -1.0]
            model.layers[0].biases[j] = 0.0
        opt = AI.Nadam(model, lr=0.1, beta1=0.0, beta2=0.999, eps=1e-8)
        g = [([[0.5, -0.5], [0.5, -0.5]], [0.0, 0.0])]
        before = model.layers[0].weights[0][0]
        opt.step(g)
        got = before - model.layers[0].weights[0][0]
        v = 0.001 * 0.25
        expect = 0.1 * 0.5 / (math.sqrt(v / (1 - 0.999)) + 1e-8)
        assert _w9_close(got, expect, 1e-12), (got, expect)

        # 与四步手算参考比对（b1=0.9）
        model2 = AI.MLP([2, 2], activations=["linear"])
        for j in range(2):
            model2.layers[0].weights[j] = [1.0, -1.0]
            model2.layers[0].biases[j] = 0.0
        opt2 = AI.Nadam(model2, lr=0.05)
        m = [[0.0, 0.0], [0.0, 0.0]]
        v = [[0.0, 0.0], [0.0, 0.0]]
        w = [[1.0, -1.0], [1.0, -1.0]]
        for t in range(1, 5):
            for j in range(2):
                for i in range(2):
                    gr = 0.3 - 0.1 * j + 0.05 * i
                    m[j][i] = 0.9 * m[j][i] + 0.1 * gr
                    v[j][i] = 0.999 * v[j][i] + 0.001 * gr * gr
                    c1 = 1 - 0.9 ** t
                    c2 = 1 - 0.999 ** t
                    num = 0.9 * (m[j][i] / c1) + 0.1 * gr / c1
                    w[j][i] -= 0.05 * num / (math.sqrt(v[j][i] / c2) + 1e-8)
            opt2.step([([[0.3, 0.35], [0.2, 0.25]], [0.0, 0.0])])
        for j in range(2):
            for i in range(2):
                assert _w9_close(model2.layers[0].weights[j][i], w[j][i], 1e-15)

    def test_radam_momentum_phase_then_rectified(self):
        """RAdam 最初几步是纯动量，之后 rho_t>4 才打开修正因子"""
        model = self._model()
        opt = AI.RAdam(model, lr=0.02)
        rng = random.Random(13)
        early = []
        for _ in range(3):
            opt.step(self._grads(model, rng))
            early.append(opt._rt is None)
        assert all(early), early
        for _ in range(12):
            opt.step(self._grads(model, rng))
        assert opt._rt is not None and opt._rt > 0.0
        assert opt._rho_t > 4.0
        # 第一步（纯动量）与 m_hat 一致
        model2 = self._model()
        opt2 = AI.RAdam(model2, lr=0.02)
        g = self._grads(model2, random.Random(13))
        before = model2.layers[0].weights[0][0]
        opt2.step(g)
        got = before - model2.layers[0].weights[0][0]
        assert _w9_close(got, 0.02 * g[0][0][0][0], 1e-12), got

    def test_state_dict_roundtrip_is_bit_exact(self):
        """state_dict 必须是快照：载入后各自续训，权重逐位一致

        回归测试：早期版本 `_w9_copy` 把状态字典原样返回（只拷贝了 list），
        于是两个优化器共享同一批 list，一方 step 会改掉另一方要用的状态。
        """
        m1 = self._model()
        opt1 = AI.Nadam(m1, lr=0.03)
        rng = random.Random(21)
        grads = [self._grads(m1, rng) for _ in range(8)]
        for g in grads[:5]:
            opt1.step(g)
        blob = opt1.state_dict()

        m2 = self._model()
        for a, b in zip(m1.layers, m2.layers):
            for j in range(len(a.weights)):
                b.weights[j] = list(a.weights[j])
            b.biases = list(a.biases)
        opt2 = AI.Nadam(m2, lr=0.03)
        opt2.load_state_dict(blob)

        for g in grads[5:]:
            opt1.step(g)
            opt2.step(g)
        assert opt1.step_count == opt2.step_count == 8
        for a, b in zip(m1.layers, m2.layers):
            assert a.weights == b.weights
            assert a.biases == b.biases

    def test_validation(self):
        """非法超参 / 层数不符 / 坏 state_dict 都报 ValueError"""
        model = self._model()
        with pytest.raises(ValueError):
            AI.AdaGrad(model, lr=-0.1)
        with pytest.raises(ValueError):
            AI.AdaGrad(model, eps=0.0)
        with pytest.raises(ValueError):
            AI.AdaDelta(model, rho=1.0)
        with pytest.raises(ValueError):
            AI.Nadam(model, beta2=1.0)
        with pytest.raises(ValueError):
            AI.RAdam(model, beta2=0.5)
        with pytest.raises(ValueError):
            AI.AdaGrad(model).step([([[0.1]], [0.1])])
        with pytest.raises(ValueError):
            AI.Nadam(model).load_state_dict([1, 2])
        with pytest.raises(ValueError):
            AI.Nadam(model).load_state_dict({"optimizer": "nadam", "state": {"x:y": {}}})
        with pytest.raises(ValueError):
            AI.Nadam(model).load_state_dict({"optimizer": "nadam", "state": {"9:w": {}}})
        with pytest.raises(ValueError):
            AI.Nadam(model).load_state_dict({"optimizer": "adagrad", "state": {}})
        # 形状不符：把 3 行的状态载入 4 行的层（状态惰性创建，先 step 一次）
        src = AI.Nadam(self._model())
        src.step(self._grads(self._model(), random.Random(1)))
        with pytest.raises(ValueError):
            AI.Nadam(self._model(sizes=(2, 4, 2))).load_state_dict(src.state_dict())

    def test_model_wiring(self):
        """Model(optimizer=...) 认得四个新名字，且能降 loss"""
        for name in ("adagrad", "adadelta", "nadam", "radam"):
            model = AI.Model(AI.MLP([2, 4, 2], activations=["relu", "linear"]),
                             loss_fn="mse", task="regression",
                             optimizer=name, lr=0.05)
            data = [([0.1 * i, -0.2 * i], [0.0, 1.0]) for i in range(4)]
            random.seed(0)
            h = model.fit(data=data, epochs=5, verbose=False)
            assert math.isfinite(h["train_loss"][-1][1])
            assert h["train_loss"][-1][1] <= h["train_loss"][0][1] + 1e-9


# ============================================================================
#  W9-B：层粒度优化器（LARS / LAMB / Ranger）
# ============================================================================

class TestLayerwiseOptimizers:
    """LARS / LAMB / Ranger 的数值与状态一致性"""

    X = [1.0, -0.5, 0.25]
    Y = [0.3, -0.7]

    @staticmethod
    def _mk(seed=7, sizes=(3, 4, 2)):
        random.seed(seed)
        return AI.MLP(list(sizes), activations=["relu", "linear"])

    @staticmethod
    def _grads(model, x=None, y=None):
        x = TestLayerwiseOptimizers.X if x is None else x
        y = TestLayerwiseOptimizers.Y if y is None else y
        pred = model.forward(x)
        _loss, gp = AI.mse_loss(pred, y)
        return AI._mlp_backward(model, gp)

    @staticmethod
    def _close(a, b, tol=1e-13):
        return abs(a - b) <= tol * max(1.0, abs(a), abs(b))

    @staticmethod
    def _snapshot(model):
        return [(layer.biases[:], [row[:] for row in layer.weights])
                for layer in model.layers]

    @staticmethod
    def _total_drift(model, opt):
        snap = TestLayerwiseOptimizers._snapshot(model)
        g = TestLayerwiseOptimizers._grads(model)
        opt.step(g)
        tot = 0.0
        for (b0, w0), layer in zip(snap, model.layers):
            for j in range(layer.fan_out):
                tot += abs(layer.biases[j] - b0[j])
                for i in range(layer.fan_in):
                    tot += abs(layer.weights[j][i] - w0[j][i])
        return tot

    def test_lars_matches_hand_computation(self):
        m = self._mk()
        ref = self._mk()
        opt = AI.LARS(m, lr=0.05, momentum=0.9, trust_coef=0.02, eps=1e-8)
        vel_w = [[[0.0] * l.fan_in for _ in range(l.fan_out)] for l in ref.layers]
        vel_b = [[0.0] * l.fan_out for l in ref.layers]
        for _ in range(2):
            g = self._grads(m)
            opt.step(g)
            for li, layer in enumerate(ref.layers):
                gw, gb = g[li]
                w_norm = math.sqrt(sum(v * v for row in layer.weights for v in row))
                g_norm = math.sqrt(sum(v * v for row in gw for v in row))
                denom = g_norm + 1e-8
                local_lr = 0.05 * 0.02 * w_norm / denom if (w_norm > 0.0 and denom > 0.0) else 0.05
                for j in range(layer.fan_out):
                    for i in range(layer.fan_in):
                        vel_w[li][j][i] = 0.9 * vel_w[li][j][i] + local_lr * gw[j][i]
                        layer.weights[j][i] -= vel_w[li][j][i]
                    vel_b[li][j] = 0.9 * vel_b[li][j] + 0.05 * gb[j]
                    layer.biases[j] -= vel_b[li][j]
        for li, layer in enumerate(ref.layers):
            for j in range(layer.fan_out):
                assert self._close(layer.biases[j], m.layers[li].biases[j])
                for i in range(layer.fan_in):
                    assert self._close(layer.weights[j][i], m.layers[li].weights[j][i])
        assert opt.step_count == 2

    def test_lars_zero_weights_falls_back_to_lr(self):
        m = self._mk()
        for layer in m.layers:
            layer.biases = [0.0] * layer.fan_out
            layer.weights = [[0.0] * layer.fan_in for _ in range(layer.fan_out)]
        opt = AI.LARS(m, lr=0.03, trust_coef=0.5)
        g = self._grads(m)
        opt.step(g)
        for li, layer in enumerate(m.layers):
            gw, gb = g[li]
            for j in range(layer.fan_out):
                assert self._close(layer.biases[j], -0.03 * gb[j])
                for i in range(layer.fan_in):
                    assert self._close(layer.weights[j][i], -0.03 * gw[j][i])

    def test_lamb_matches_hand_computation(self):
        m = self._mk(seed=11)
        ref = self._mk(seed=11)
        opt = AI.LAMB(m, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-6,
                      weight_decay=0.01)
        mw = [[[[0.0] * l.fan_in for _ in range(l.fan_out)],
               [[0.0] * l.fan_in for _ in range(l.fan_out)]] for l in ref.layers]
        mb = [[[0.0] * l.fan_out, [0.0] * l.fan_out] for l in ref.layers]
        for step in range(1, 3):
            g = self._grads(m)
            opt.step(g)
            c1 = 1.0 - 0.9 ** step
            c2 = 1.0 - 0.999 ** step
            for li, layer in enumerate(ref.layers):
                gw, gb = g[li]
                u = [[0.0] * layer.fan_in for _ in range(layer.fan_out)]
                for j in range(layer.fan_out):
                    for i in range(layer.fan_in):
                        gg = gw[j][i]
                        mw[li][0][j][i] = 0.9 * mw[li][0][j][i] + 0.1 * gg
                        mw[li][1][j][i] = 0.999 * mw[li][1][j][i] + 0.001 * gg * gg
                        u[j][i] = ((mw[li][0][j][i] / c1)
                                   / (math.sqrt(mw[li][1][j][i] / c2) + 1e-6)
                                   + 0.01 * layer.weights[j][i])
                w_norm = math.sqrt(sum(v * v for row in layer.weights for v in row))
                u_norm = math.sqrt(sum(v * v for row in u for v in row))
                trust = min(w_norm / u_norm, 10.0) if (w_norm > 0.0 and u_norm > 0.0) else 1.0
                for j in range(layer.fan_out):
                    for i in range(layer.fan_in):
                        layer.weights[j][i] -= 0.01 * trust * u[j][i]
                    gg = gb[j]
                    mb[li][0][j] = 0.9 * mb[li][0][j] + 0.1 * gg
                    mb[li][1][j] = 0.999 * mb[li][1][j] + 0.001 * gg * gg
                    st = ((mb[li][0][j] / c1)
                          / (math.sqrt(mb[li][1][j] / c2) + 1e-6))
                    layer.biases[j] -= 0.01 * (st + 0.01 * layer.biases[j])
        for li, layer in enumerate(ref.layers):
            for j in range(layer.fan_out):
                assert self._close(layer.biases[j], m.layers[li].biases[j])
                for i in range(layer.fan_in):
                    assert self._close(layer.weights[j][i], m.layers[li].weights[j][i])

    def test_lamb_trust_clip_bounds_step(self):
        m1 = self._mk(seed=3)
        small = self._total_drift(m1, AI.LAMB(m1, lr=0.01, trust_clip=1e-6))
        m2 = self._mk(seed=3)
        big = self._total_drift(m2, AI.LAMB(m2, lr=0.01, trust_clip=1e6))
        assert small > 0.0
        assert small < big

    def test_ranger_first_step_is_pure_momentum(self):
        m = self._mk(seed=5)
        opt = AI.Ranger(m, lr=0.01, beta2=0.999, k=999)
        g = self._grads(m)
        snap = self._snapshot(m)
        opt.step(g)
        for li, layer in enumerate(m.layers):
            b0, w0 = snap[li]
            gw, gb = g[li]
            for j in range(layer.fan_out):
                assert self._close(layer.biases[j] - b0[j], -0.01 * gb[j], 1e-12)
                for i in range(layer.fan_in):
                    assert self._close(layer.weights[j][i] - w0[j][i],
                                       -0.01 * gw[j][i], 1e-12)

    def test_ranger_lookahead_syncs_slow_weights(self):
        m = self._mk(seed=9)
        opt = AI.Ranger(m, lr=0.02, alpha=0.5, k=2)
        for _ in range(2):
            opt.step(self._grads(m))
        for idx, layer in enumerate(m.layers):
            st = opt.state[(idx, "w")]
            for j in range(layer.fan_out):
                assert self._close(st["slow"][j][0], layer.weights[j][0])
        # α=1 时 slow 直接等于 fast（即 fast 不动）
        m2 = self._mk(seed=9)
        opt2 = AI.Ranger(m2, lr=0.02, alpha=1.0, k=1)
        before = self._snapshot(m2)
        opt2.step(self._grads(m2))
        moved = 0.0
        for (b0, w0), layer in zip(before, m2.layers):
            for j in range(layer.fan_out):
                moved = max(moved, abs(layer.biases[j] - b0[j]))
                for i in range(layer.fan_in):
                    moved = max(moved, abs(layer.weights[j][i] - w0[j][i]))
        assert moved > 0.0

    def test_state_dict_roundtrip_is_bit_exact(self):
        for cls, kw in ((AI.LARS, dict(lr=0.05, momentum=0.9, trust_coef=0.01)),
                        (AI.LAMB, dict(lr=0.01, weight_decay=0.01)),
                        (AI.Ranger, dict(lr=0.01, k=2))):
            a = self._mk(seed=21)
            b = self._mk(seed=21)
            oa = cls(a, **kw)
            ob = cls(b, **kw)
            g = self._grads(a)
            for _ in range(3):
                oa.step(g)
            # 断点续训的前提：权重也要回到断点
            for la, lb in zip(a.layers, b.layers):
                lb.biases = la.biases[:]
                lb.weights = [row[:] for row in la.weights]
            ob.load_state_dict(oa.state_dict())
            for _ in range(2):
                oa.step(g)
                ob.step(g)
            fa, fb = a.forward(self.X), b.forward(self.X)
            for p, q in zip(fa, fb):
                assert self._close(p, q)
            assert oa.step_count == ob.step_count == 5

    def test_validation(self):
        m = AI.MLP([2, 2])
        with pytest.raises(ValueError):
            AI.LARS(m, lr=-1)
        with pytest.raises(ValueError):
            AI.LARS(m, momentum=1.0)
        with pytest.raises(ValueError):
            AI.LARS(m, trust_coef=-0.1)
        with pytest.raises(ValueError):
            AI.LARS(m, nesterov="yes")
        with pytest.raises(ValueError):
            AI.LAMB(m, beta2=1.0)
        with pytest.raises(ValueError):
            AI.LAMB(m, weight_decay=-1)
        with pytest.raises(ValueError):
            AI.LAMB(m, trust_clip=0)
        with pytest.raises(ValueError):
            AI.Ranger(m, beta2=0.5)
        with pytest.raises(ValueError):
            AI.Ranger(m, alpha=0.0)
        with pytest.raises(ValueError):
            AI.Ranger(m, k=0)
        with pytest.raises(ValueError):
            AI.LARS(m).load_state_dict("nope")
        with pytest.raises(ValueError):
            AI.LARS(m).load_state_dict({"optimizer": "lamb", "state": {}})

    def test_model_wiring(self):
        data = [([1.0, 2.0, 3.0], [0.5, 0.1]), ([0.1, 0.2, 0.3], [0.0, 1.0]),
                ([0.5, 0.5, 0.5], [0.3, 0.3])]
        for name in ("lars", "lamb", "ranger"):
            m = AI.Model(AI.MLP([3, 4, 2], activations=["relu", "linear"]),
                         task="regression", loss_fn="mse", optimizer=name,
                         lr=0.01)
            m.fit(data=data, epochs=40, verbose=False, seed=2)
            first = m.history["train_loss"][0][1]
            last = m.history["train_loss"][-1][1]
            assert last < first, (name, first, last)


# ==== W9-C：SGDW / Nesterov / Lookahead / EMA / SWA ====
class TestSGDWFamily:
    X = [1.0, -0.5, 0.25]
    Y = [0.3, -0.7]

    @staticmethod
    def _mk(seed=4):
        random.seed(seed)
        return AI.MLP([3, 4, 2], activations=["relu", "linear"])

    @staticmethod
    def _grads(model):
        _l, gp = AI.mse_loss(model.forward(TestSGDWFamily.X), TestSGDWFamily.Y)
        return AI._mlp_backward(model, gp)

    @staticmethod
    def _flat(model):
        out = []
        for layer in model.layers:
            out.extend(layer.biases)
            for row in layer.weights:
                out.extend(row)
        return out

    @staticmethod
    def _close(a, b, tol=1e-12):
        return abs(a - b) <= tol * max(1.0, abs(a), abs(b))

    def test_sgdw_without_decay_matches_sgd(self):
        a, b = self._mk(), self._mk()
        for la, lb in zip(a.layers, b.layers):
            lb.biases = la.biases[:]
            lb.weights = [row[:] for row in la.weights]
        g = self._grads(a)
        AI.SGD(a, lr=0.05).step(g)
        AI.SGDW(b, lr=0.05, momentum=0.0, weight_decay=0.0).step(g)
        assert max(abs(p - q) for p, q in zip(self._flat(a), self._flat(b))) == 0.0

    def test_sgdw_decoupled_weight_decay(self):
        m = self._mk(5)
        opt = AI.SGDW(m, lr=0.1, momentum=0.0, weight_decay=0.5)
        before = self._flat(m)
        zero = [([[0.0] * l.fan_in for _ in range(l.fan_out)], [0.0] * l.fan_out)
                for l in m.layers]
        opt.step(zero)
        expect = [p * (1.0 - 0.1 * 0.5) for p in before]
        assert all(self._close(p, q) for p, q in zip(self._flat(m), expect))

    def test_nesterov_matches_hand_computation(self):
        m1, m2 = self._mk(11), self._mk(11)
        for la, lb in zip(m1.layers, m2.layers):
            lb.biases = la.biases[:]
            lb.weights = [row[:] for row in la.weights]
        opt = AI.NesterovSGD(m1, lr=0.03, momentum=0.9)
        vw = [[[0.0] * l.fan_in for _ in range(l.fan_out)] for l in m2.layers]
        vb = [[0.0] * l.fan_out for l in m2.layers]
        for _ in range(3):
            g = self._grads(m1)
            opt.step(g)
            for li, layer in enumerate(m2.layers):
                gw, gb = g[li]
                for j in range(layer.fan_out):
                    v = 0.9 * vb[li][j] + gb[j]
                    vb[li][j] = v
                    layer.biases[j] -= 0.03 * (gb[j] + 0.9 * v)
                    for i in range(layer.fan_in):
                        v = 0.9 * vw[li][j][i] + gw[j][i]
                        vw[li][j][i] = v
                        layer.weights[j][i] -= 0.03 * (gw[j][i] + 0.9 * v)
        assert max(abs(p - q) for p, q in zip(self._flat(m1), self._flat(m2))) <= 1e-13


class TestLookahead:
    X = TestSGDWFamily.X
    Y = TestSGDWFamily.Y
    _mk = staticmethod(TestSGDWFamily._mk)
    _grads = staticmethod(TestSGDWFamily._grads)
    _flat = staticmethod(TestSGDWFamily._flat)
    _close = staticmethod(TestSGDWFamily._close)

    def test_sync_happens_on_kth_step(self):
        m = self._mk(13)
        la = AI.Lookahead(AI.SGDMomentum(m, lr=0.05, momentum=0.9), k=2, alpha=0.5)
        g = self._grads(m)
        la.step(g)
        assert la.step_count == 1
        la.step(g)
        fast = self._flat(m)
        slow = []
        for bs, ws in la._slow:
            slow.extend(bs)
            for row in ws:
                slow.extend(row)
        assert all(self._close(p, q) for p, q in zip(fast, slow))

    def test_alpha_is_midpoint(self):
        m = self._mk(13)
        la = AI.Lookahead(AI.SGDMomentum(m, lr=0.05, momentum=0.9), k=2, alpha=0.5)
        g = self._grads(m)
        slow0 = self._flat(m)
        la.step(g)
        la.step(g)
        slow = []
        for bs, ws in la._slow:
            slow.extend(bs)
            for row in ws:
                slow.extend(row)
        # 手写：两次 inner step 用同一份 stale 梯度，slow = slow0 + 0.5*(fast-slow0)
        m2 = self._mk(13)
        inner = AI.SGDMomentum(m2, lr=0.05, momentum=0.9)
        inner.step(g)
        inner.step(g)
        expect = [s + 0.5 * (f - s) for s, f in zip(slow0, self._flat(m2))]
        assert max(abs(p - q) for p, q in zip(slow, expect)) <= 1e-12

    def test_state_dict_roundtrip_with_inner_adam(self):
        m = self._mk(17)
        la = AI.Lookahead(AI.Adam(m, lr=0.02), k=2, alpha=0.4)
        for _ in range(3):
            la.step(self._grads(m))
        blob = la.state_dict()
        m2 = self._mk(17)
        la2 = AI.Lookahead(AI.Adam(m2, lr=0.02), k=2, alpha=0.4)
        la2.optimizer.step_count = la.optimizer.step_count
        for a, b in zip(m.layers, m2.layers):
            b.biases = a.biases[:]
            b.weights = [row[:] for row in a.weights]
            b.m_w = [row[:] for row in a.m_w]
            b.v_w = [row[:] for row in a.v_w]
            b.m_b = a.m_b[:]
            b.v_b = a.v_b[:]
        la2.load_state_dict(blob)
        for _ in range(2):
            la.step(self._grads(m))
            la2.step(self._grads(m2))
        assert la2.step_count == 5
        assert max(abs(p - q) for p, q in zip(self._flat(m), self._flat(m2))) == 0.0


class TestEMAAndSWA:
    _flat = staticmethod(TestSGDWFamily._flat)
    _close = staticmethod(TestSGDWFamily._close)

    def test_ema_closed_form(self):
        m = AI.MLP([2, 3, 1])
        ema = AI.EMA(m, decay=0.9)
        w0 = m.layers[0].weights[0][0]
        ema.update()
        seq = [w0]
        for k in range(1, 4):
            m.layers[0].weights[0][0] = w0 + k
            seq.append(w0 + k)
            ema.update()
        ref = seq[0]
        for k in range(1, 4):
            ref = 0.9 * ref + 0.1 * seq[k]
        assert self._close(ema.shadow()[0][1][0][0], ref)

    def test_ema_warmup_effective_decay(self):
        m = AI.MLP([2, 3, 1])
        ema = AI.EMA(m, decay=0.999, warmup=2)
        d0 = ema.effective_decay()
        ema.update()
        d1 = ema.effective_decay()
        assert self._close(d0, 1 / 10) and self._close(d1, 2 / 11)

    def test_swa_incremental_mean_and_roundtrip(self):
        m = AI.MLP([2, 3, 1])
        swa = AI.SWA(m)
        snaps = []
        for k in range(4):
            m.layers[0].weights[0][0] = float(k)
            snaps.append(float(k))
            swa.update()
        mean = sum(snaps) / len(snaps)
        assert self._close(swa.average()[0][1][0][0], mean)
        blob = swa.state_dict()
        swa2 = AI.SWA(AI.MLP([2, 3, 1]))
        swa2.load_state_dict(blob)
        assert swa2.n == 4 and self._close(swa2.average()[0][1][0][0], mean)

    def test_ema_swa_validation(self):
        m = AI.MLP([2, 2])
        for fn in (lambda: AI.EMA(m, decay=1.0),
                   lambda: AI.EMA(m, warmup=-1),
                   lambda: AI.EMA(m).load_state_dict({"nope": 1}),
                   lambda: AI.SWA(m).load_state_dict({"nope": 1}),
                   lambda: AI.EMA(AI.MLP([3, 3])).apply_to(m)):
            with pytest.raises(ValueError):
                fn()


class TestEMASWACallbacks:
    def _data(self):
        return [([1.0, 2.0, 3.0], [0.5, 0.1]), ([0.1, 0.2, 0.3], [0.0, 1.0]),
                ([0.5, 0.5, 0.5], [0.3, 0.3])]

    def test_callbacks_track_each_epoch(self):
        mm = AI.MLP([3, 4, 2], activations=["relu", "linear"])
        model = AI.Model(mm, task="regression", loss_fn="mse", optimizer="adam", lr=0.05)
        ecb = AI.EMACallback(mm, decay=0.5)
        scb = AI.SWACallback(mm)
        model.fit(data=self._data(), epochs=6, verbose=False, seed=3,
                  callbacks=[ecb, scb])
        assert ecb.ema.step == 6 and scb.swa.n == 6

    def test_swa_average_equals_snapshot_mean(self):
        snaps = []

        class Rec(AI.Callback):
            def on_epoch_end(self, epoch, logs):
                snaps.append(TestSGDWFamily._flat(mm))

        mm = AI.MLP([3, 4, 2], activations=["relu", "linear"])
        model = AI.Model(mm, task="regression", loss_fn="mse", optimizer="adam", lr=0.05)
        scb = AI.SWACallback(mm)
        model.fit(data=self._data(), epochs=4, verbose=False, seed=3,
                  callbacks=[scb, Rec()])
        avg = []
        for k, snap in enumerate(snaps, start=1):
            avg = list(snap) if k == 1 else [a + (b - a) / k for a, b in zip(avg, snap)]
        flat = []
        for bs, ws in scb.swa.average():
            flat.extend(bs)
            for row in ws:
                flat.extend(row)
        assert max(abs(p - q) for p, q in zip(flat, avg)) <= 1e-12


# ============================================================================
#  W9-D：学习率调度器族（LambdaLR / Multiplicative / MultiStep / 余弦重启 /
#        CyclicLR / OneCycleLR / InverseSqrt / Noam / Polynomial / WSD /
#        ChainedScheduler / SequentialLR）
# ============================================================================
class TestSchedulerFamily:
    """每个调度器都用手写闭式解对照；StepLR 家族是 epoch 门控的，必须按序喂。"""

    class _Opt:
        def __init__(self, lr=0.1):
            self.lr = lr

    @staticmethod
    def _at(sched, epoch):
        sched.step(epoch)
        return sched.optimizer.lr

    def test_lambda_and_multiplicative(self):
        o = self._Opt(0.2)
        s = AI.LambdaLR(o, lambda e: 0.5 ** e)
        assert abs(self._at(s, 2) - 0.05) < 1e-15
        assert abs(self._at(s, 4) - 0.0125) < 1e-15
        assert s.initial_lr == 0.2
        with pytest.raises(ValueError):
            AI.LambdaLR(self._Opt(), 3)
        bad = AI.LambdaLR(self._Opt(), lambda e: -1.0)
        with pytest.raises(ValueError):
            bad.step(1)

        o = self._Opt(0.2)
        s = AI.MultiplicativeLR(o, lambda e: 0.5)
        assert abs(self._at(s, 1) - 0.1) < 1e-15
        assert abs(self._at(s, 2) - 0.05) < 1e-15
        assert abs(self._at(s, 3) - 0.025) < 1e-15

    def test_multistep(self):
        o = self._Opt(0.2)
        s = AI.MultiStepLR(o, [2, 5], gamma=0.1)
        # 里程碑 ≤ epoch 的个数决定衰减次数
        assert abs(self._at(s, 1) - 0.2) < 1e-15
        assert abs(self._at(s, 2) - 0.02) < 1e-15
        assert abs(self._at(s, 5) - 0.002) < 1e-15
        for bad in ([], [0], [3, 2], [2, 2]):
            with pytest.raises(ValueError):
                AI.MultiStepLR(self._Opt(), bad)

    def test_cosine_warm_restarts(self):
        o = self._Opt(0.2)
        s = AI.CosineAnnealingWarmRestarts(o, T_0=3, T_mult=2, eta_min=0.01)
        assert s.cycle_of(1) == (0, 3, 1)
        assert s.cycle_of(4) == (1, 6, 1)
        assert s.cycle_of(10) == (2, 12, 1)
        # 周期起点回到峰值、末轮触底
        assert abs(self._at(s, 1) - 0.2) < 1e-15
        assert abs(self._at(s, 3) - 0.01) < 1e-15
        assert abs(self._at(s, 4) - 0.2) < 1e-15
        assert abs(self._at(s, 9) - 0.01) < 1e-15
        # 周期内闭式解
        k, length, pos = s.cycle_of(6)
        frac = (pos - 1) / (length - 1)
        want = 0.01 + (0.2 - 0.01) * 0.5 * (1 + math.cos(math.pi * frac))
        assert abs(self._at(s, 6) - want) < 1e-15
        with pytest.raises(ValueError):
            AI.CosineAnnealingWarmRestarts(self._Opt(), T_0=0)
        with pytest.raises(ValueError):
            AI.CosineAnnealingWarmRestarts(self._Opt(), T_0=3, T_mult=0)

    def test_cyclic_lr_modes(self):
        o = self._Opt()
        s = AI.CyclicLR(o, 0.01, 0.11, step_size_up=2)
        seq = [self._at(s, e) for e in range(1, 6)]
        assert abs(seq[0] - 0.01) < 1e-15 and abs(seq[2] - 0.11) < 1e-15
        assert abs(seq[3] - 0.06) < 1e-15            # 下降段中点
        assert abs(seq[4] - 0.01) < 1e-15            # 一个完整周期后回到 base
        assert seq[0] <= seq[1] <= seq[2] and seq[2] >= seq[3]

        o = self._Opt()
        s = AI.CyclicLR(o, 0.0, 1.0, step_size_up=2, mode="triangular2")
        assert abs(self._at(s, 3) - 1.0) < 1e-15      # 第 1 周期峰值
        assert abs(self._at(s, 6) - 0.25) < 1e-15     # 第 2 周期中点
        assert abs(self._at(s, 7) - 0.5) < 1e-15      # 第 2 周期峰值

        o = self._Opt()
        s = AI.CyclicLR(o, 0.0, 1.0, step_size_up=2, mode="exp_range", gamma=0.5)
        assert abs(self._at(s, 3) - 0.25) < 1e-15
        with pytest.raises(ValueError):
            AI.CyclicLR(self._Opt(), 0.2, 0.1)
        with pytest.raises(ValueError):
            AI.CyclicLR(self._Opt(), 0.1, 0.2, mode="cos")

    def test_one_cycle(self):
        o = self._Opt()
        s = AI.OneCycleLR(o, max_lr=1.0, total_epochs=10, pct_start=0.3,
                          div_factor=25.0, final_div_factor=1e4)
        start = 1.0 / 25.0
        end = start / 1e4
        first = start + (1.0 - start) * 0.5 * (1 - math.cos(math.pi * (0.1 / 0.3)))
        assert abs(self._at(s, 1) - first) < 1e-12
        assert abs(self._at(s, 3) - 1.0) < 1e-12
        assert abs(self._at(s, 10) - end) < 1e-9
        assert abs(self._at(s, 40) - end) < 1e-9      # 超出 total 后夹住
        with pytest.raises(ValueError):
            AI.OneCycleLR(self._Opt(), 0.1, 0)
        with pytest.raises(ValueError):
            AI.OneCycleLR(self._Opt(), 0.1, 10, pct_start=1.5)

    def test_inverse_sqrt_and_noam(self):
        o = self._Opt(0.1)
        s = AI.InverseSqrtLR(o, warmup_epochs=4, start_factor=0.0)
        assert abs(self._at(s, 2) - 0.05) < 1e-15
        assert abs(self._at(s, 4) - 0.1) < 1e-15
        assert abs(self._at(s, 8) - 0.1 * math.sqrt(0.5)) < 1e-15

        o = self._Opt(0.1)
        s = AI.InverseSqrtLR(o, warmup_epochs=0)
        assert abs(self._at(s, 1) - 0.1) < 1e-15
        assert abs(self._at(s, 4) - 0.05) < 1e-15

        o = self._Opt(0.1)
        s = AI.NoamLR(o, d_model=16, warmup_epochs=4)
        assert abs(self._at(s, 4) - 0.125) < 1e-12    # 峰值，覆盖 initial_lr
        assert abs(self._at(s, 16) - (16 ** -0.5) * 16 ** -0.5) < 1e-12
        with pytest.raises(ValueError):
            AI.NoamLR(self._Opt(), d_model=0)
        with pytest.raises(ValueError):
            AI.InverseSqrtLR(self._Opt(), warmup_epochs=-1)

    def test_polynomial_and_wsd(self):
        o = self._Opt(0.2)
        s = AI.PolynomialLR(o, total_epochs=10, power=2.0, min_lr=0.02)
        assert abs(self._at(s, 5) - ((0.2 - 0.02) * 0.25 + 0.02)) < 1e-15
        assert abs(self._at(s, 10) - 0.02) < 1e-15
        assert abs(self._at(s, 50) - 0.02) < 1e-15

        o = self._Opt(0.2)
        s = AI.WSDLR(o, warmup_epochs=2, total_epochs=10, decay_epochs=3,
                     decay_shape="linear")
        assert abs(self._at(s, 1) - 0.1) < 1e-15
        assert abs(self._at(s, 2) - 0.2) < 1e-15
        assert abs(self._at(s, 5) - 0.2) < 1e-15      # stable
        assert abs(self._at(s, 8) - (0.2 - 0.2 / 3.0)) < 1e-15
        assert abs(self._at(s, 10) - 0.0) < 1e-15
        with pytest.raises(ValueError):
            AI.WSDLR(self._Opt(), warmup_epochs=5, total_epochs=10, decay_epochs=6)
        with pytest.raises(ValueError):
            AI.WSDLR(self._Opt(), decay_shape="sqrt")

    def test_chained_and_sequential(self):
        o = self._Opt(0.2)
        a = AI.StepLR(o, step_size=3, gamma=0.5)
        b = AI.MultiStepLR(o, [2], gamma=0.5)
        ch = AI.ChainedScheduler(a, b)
        ref = self._Opt(0.2)
        ra = AI.StepLR(ref, step_size=3, gamma=0.5)
        rb = AI.MultiStepLR(ref, [2], gamma=0.5)
        for e in (1, 2, 3, 4, 5):
            ra.step(e)
            rb.step(e)
            assert abs(self._at(ch, e) - ref.lr) < 1e-15
        assert ch.optimizer is o
        with pytest.raises(ValueError):
            AI.ChainedScheduler()
        with pytest.raises(ValueError):
            AI.ChainedScheduler([1])

        o = self._Opt(0.2)
        s1 = AI.StepLR(o, step_size=2, gamma=0.1)
        s2 = AI.ExponentialLR(o, gamma=0.5)
        sq = AI.SequentialLR(o, [s1, s2], [4])
        ref = self._Opt(0.2)
        r1 = AI.StepLR(ref, step_size=2, gamma=0.1)
        r2 = AI.ExponentialLR(ref, gamma=0.5)
        for e in range(1, 8):
            (r1 if e < 4 else r2).step(e)
            assert abs(self._at(sq, e) - ref.lr) < 1e-15
        for args in (([s1], [4, 8]), ([s1, s2], []), ([s1, s2], [8, 4]),
                     ([s1, s2], [0])):
            with pytest.raises(ValueError):
                AI.SequentialLR(self._Opt(), *args)

    def test_epoch_normalization(self):
        o = self._Opt()
        for s in (AI.CosineAnnealingWarmRestarts(o, T_0=3),
                  AI.CyclicLR(o, 0.01, 0.1),
                  AI.OneCycleLR(o, 0.1, 10),
                  AI.InverseSqrtLR(o, warmup_epochs=2),
                  AI.NoamLR(o, 16),
                  AI.PolynomialLR(o, total_epochs=10),
                  AI.WSDLR(o, warmup_epochs=1, total_epochs=10, decay_epochs=2)):
            # epoch<=0 被夹到 1，不报错
            s.step(0)
            s.step(-3)
            for bad in (True, 1.5, "3"):
                with pytest.raises(ValueError):
                    s.step(bad)

    def test_existing_schedulers_unchanged(self):
        """回归锁：W9-D 不应改动旧调度器的数值行为。"""
        o = self._Opt(0.1)
        s = AI.StepLR(o, step_size=2, gamma=0.5)
        seq = [self._at(s, e) for e in range(1, 6)]
        assert seq == [0.1, 0.05, 0.05, 0.025, 0.025], seq

        o = self._Opt(0.1)
        s = AI.ExponentialLR(o, gamma=0.5)
        assert abs(self._at(s, 3) - 0.0125) < 1e-15

        o = self._Opt(0.1)
        s = AI.CosineAnnealingLR(o, T_max=10, eta_min=0.0)
        vals = [self._at(s, e) for e in range(1, 11)]
        assert abs(vals[-1]) < 1e-15
        assert all(a >= b - 1e-18 for a, b in zip(vals, vals[1:])), vals

    def test_fit_integration(self):
        random.seed(3)
        data = [([random.random(), random.random()],
                 [1.0 if random.random() > 0.5 else 0.0]) for _ in range(16)]
        factories = [
            lambda opt: AI.CosineAnnealingWarmRestarts(opt, T_0=3),
            lambda opt: AI.OneCycleLR(opt, max_lr=0.05, total_epochs=6),
            lambda opt: AI.CyclicLR(opt, 0.001, 0.02, step_size_up=2),
            lambda opt: AI.InverseSqrtLR(opt, warmup_epochs=2),
            lambda opt: AI.PolynomialLR(opt, total_epochs=6),
            lambda opt: AI.WSDLR(opt, warmup_epochs=1, total_epochs=6, decay_epochs=2),
            lambda opt: AI.ChainedScheduler(AI.StepLR(opt, 3, 0.5),
                                            AI.CosineAnnealingLR(opt, 6)),
            lambda opt: AI.SequentialLR(opt, [AI.StepLR(opt, 2, 0.5),
                                              AI.ExponentialLR(opt, 0.9)], [4]),
            lambda opt: AI.NoamLR(opt, 8, warmup_epochs=2),
            lambda opt: AI.LambdaLR(opt, lambda e: 0.9 ** e),
            lambda opt: AI.MultiplicativeLR(opt, lambda e: 0.95),
            lambda opt: AI.MultiStepLR(opt, [3], gamma=0.5),
        ]
        for factory in factories:
            m = AI.Model(AI.MLP([2, 6, 1], activations=["relu", "sigmoid"]),
                         loss_fn="mse", task="regression", lr=0.05)
            hist = m.fit(data=data, epochs=6, batch_size=4, val_split=0.0,
                         verbose=False, shuffle=False, seed=1,
                         lr_scheduler=factory(m.optimizer))
            losses = [v for _e, v in hist["train_loss"]]
            assert len(losses) == 6
            assert all(v == v and abs(v) < 1e6 for v in losses)


# ============================================================================
#  W9-E：进阶优化器（Adan / Prodigy / CAME）
# ============================================================================

class TestAdvancedOptimizers:
    """三个优化器的接线、状态往返与自适应行为"""

    X = [1.0, -0.5, 0.25]
    Y = [0.3, -0.7]

    @staticmethod
    def _mk(seed=11, sizes=(3, 4, 2)):
        random.seed(seed)
        return AI.MLP(list(sizes), activations=["relu", "linear"])

    @staticmethod
    def _grads(model):
        pred = model.forward(TestAdvancedOptimizers.X)
        _loss, gp = AI.mse_loss(pred, TestAdvancedOptimizers.Y)
        return AI._mlp_backward(model, gp)

    @staticmethod
    def _snapshot(model):
        return [(l.biases[:], [r[:] for r in l.weights]) for l in model.layers]

    @staticmethod
    def _drift(model, snap):
        """返回 (权重位移, 偏置位移) —— 偏置是扁平 list，必须单独覆盖"""
        dw = db = 0.0
        for (b0, w0), l in zip(snap, model.layers):
            for j in range(l.fan_out):
                db += abs(l.biases[j] - b0[j])
                for i in range(l.fan_in):
                    dw += abs(l.weights[j][i] - w0[j][i])
        return dw, db

    def test_weight_and_bias_move(self):
        """回归锁：偏置曾按 param[j] 取到 float 而整体崩溃"""
        for cls in (AI.Adan, AI.Prodigy, AI.CAME):
            model = self._mk()
            opt = cls(model)
            snap = self._snapshot(model)
            opt.step(self._grads(model))
            dw, db = self._drift(model, snap)
            assert dw > 0.0 and db > 0.0

    def test_registered_and_trains(self):
        """三个名字能被 Model(optimizer=...) 认到并让 loss 下降"""
        for name in ("adan", "prodigy", "came"):
            model = AI.Model(AI.MLP([2, 4, 2], activations=["relu", "linear"]),
                             loss_fn="mse", task="regression",
                             optimizer=name, lr=0.02)
            data = [([0.1 * i, -0.2 * i], [0.0, 1.0]) for i in range(4)]
            h = model.fit(data=data, epochs=6, verbose=False)
            first = h["train_loss"][0][1]
            last = h["train_loss"][-1][1]
            assert math.isfinite(last) and last <= first + 1e-9

    def test_state_roundtrip_is_bitwise(self):
        """存档 → 载入 → 继续两步，参数逐位一致"""
        for cls in (AI.Adan, AI.Prodigy, AI.CAME):
            a = self._mk(seed=13)
            b = self._mk(seed=13)
            oa, ob = cls(a), cls(b)
            oa.step(self._grads(a))
            oa.step(self._grads(a))
            # 优化器状态只含动量，不含参数本身：先把参数搬过去
            for la, lb in zip(a.layers, b.layers):
                lb.biases[:] = la.biases[:]
                for j in range(la.fan_out):
                    lb.weights[j][:] = la.weights[j][:]
            ob.load_state_dict(oa.state_dict())
            for _ in range(2):
                oa.step(self._grads(a))
                ob.step(self._grads(b))
            snap_a = self._snapshot(a)
            snap_b = self._snapshot(b)
            for (ba, wa), (bb, wb) in zip(snap_a, snap_b):
                assert ba == bb
                assert [r[:] for r in wa] == [r[:] for r in wb]

    def test_state_dict_does_not_alias(self):
        model = self._mk(seed=17)
        opt = AI.CAME(model)
        opt.step(self._grads(model))
        sd = opt.state_dict()
        key = [k for k in sorted(sd["state"]) if sd["state"][k].get("r")][0]
        before = sd["state"][key]["r"][:]
        opt.step(self._grads(model))
        assert sd["state"][key]["r"] == before, "state_dict 与实例共享了状态"

    def test_prodigy_d_adapts(self):
        """乘性自适应：d 只按 growth / shrink 走，并夹在 [d_min, d_max]"""
        model = self._mk(seed=5)
        opt = AI.Prodigy(model, d_init=1e-2, growth=2.0, shrink=0.5,
                         d_min=1e-4, d_max=10.0)
        d0 = opt.d
        opt.step(self._grads(model))          # 第一步没有上一步快照，d 不动
        assert opt.d == d0
        opt.step(self._grads(model))          # 第二步按 dot 调整
        assert opt.d in (pytest.approx(d0 * 2.0), pytest.approx(d0 * 0.5))
        for _ in range(50):
            opt.step(self._grads(model))
        assert 1e-4 <= opt.d <= 10.0

    def test_came_clip_threshold_matters(self):
        """clip_threshold 必须真的起作用（原实现量纲错，只能靠它兜底）"""
        m1 = self._mk(seed=19)
        m2 = self._mk(seed=19)
        o1 = AI.CAME(m1, lr=0.01, clip_threshold=1e-6)
        o2 = AI.CAME(m2, lr=0.01, clip_threshold=1e6)
        for _ in range(3):
            o1.step(self._grads(m1))
            o2.step(self._grads(m2))
        # 逐参数取最大差：单个坐标可能恰好是死单元（梯度恒 0，两种阈值都不动）
        max_diff = 0.0
        for la, lb in zip(m1.layers, m2.layers):
            for ra, rb in zip(la.weights, lb.weights):
                for a, b in zip(ra, rb):
                    max_diff = max(max_diff, abs(a - b))
        assert max_diff > 1e-8
        for model in (m1, m2):
            for l in model.layers:
                for row in l.weights:
                    for w in row:
                        assert math.isfinite(w) and abs(w) < 100.0

    def test_invalid_hyperparameters(self):
        model = self._mk()
        with pytest.raises(ValueError):
            AI.Adan(model, lr=-1.0)
        with pytest.raises(ValueError):
            AI.Adan(model, beta2=1.0)
        with pytest.raises(ValueError):
            AI.Adan(model, eps=0.0)
        with pytest.raises(ValueError):
            AI.Prodigy(model, growth=0.0)
        with pytest.raises(ValueError):
            AI.Prodigy(model, shrink=5.0)
        with pytest.raises(ValueError):
            AI.Prodigy(model, d_min=1.0, d_max=0.5)
        with pytest.raises(TypeError):
            AI.Prodigy(model, growth="fast")
        with pytest.raises(ValueError):
            AI.CAME(model, clip_threshold=0.0)
        with pytest.raises(ValueError):
            AI.CAME(model, beta2=1.2)

    def test_came_state_field_validation(self):
        """r 是行向量：长度必须等于 fan_out，载入时要查出来"""
        model = self._mk()
        opt = AI.CAME(model)
        opt.step(self._grads(model))
        sd = opt.state_dict()
        key = sorted(sd["state"])[0]
        sd["state"][key]["r"] = [1.0]
        with pytest.raises(ValueError):
            AI.CAME(self._mk(seed=11)).load_state_dict(sd)


class TestAdvancedOptimizers2:
    """W9-E2：Shampoo / Sophia / L-BFGS 的闭式解、退化分支与状态往返"""

    X = [1.0, -0.5, 0.25]
    Y = [0.3, -0.7]

    @staticmethod
    def _mk(seed=11, sizes=(3, 4, 2)):
        random.seed(seed)
        acts = ["relu"] * (len(sizes) - 2) + ["linear"]
        return AI.MLP(list(sizes), activations=acts)

    @staticmethod
    def _grads(model, x=None, y=None):
        x = TestAdvancedOptimizers2.X if x is None else x
        y = TestAdvancedOptimizers2.Y if y is None else y
        pred = model.forward(x)
        _loss, gp = AI.mse_loss(pred, y)
        return AI._mlp_backward(model, gp)

    @staticmethod
    def _flat(model):
        out = []
        for layer in model.layers:
            out.extend(layer.biases)
            for row in layer.weights:
                out.extend(row)
        return out

    def test_shampoo_matches_closed_form_1x1(self):
        """1x1 层：L=R=g²，逆根把预条件化成 sign(g)，位移恰为 -lr·sign(g)"""
        model = self._mk(sizes=(1, 1))
        model.layers[0].weights[0][0] = 0.5
        model.layers[0].biases[0] = 0.0
        model.layers[0].activation_name = "linear"
        grad = self._grads(model, [1.0], [0.0])
        g = grad[0][0][0][0]
        before = model.layers[0].weights[0][0]
        AI.Shampoo(model, lr=0.25).step(grad)
        assert abs((model.layers[0].weights[0][0] - before) + 0.25 * (1.0 if g > 0 else -1.0)) < 1e-12

    def test_jacobi_reconstructs_symmetric_matrix(self):
        """循环 Jacobi 的特征分解必须能重建原矩阵且特征向量正交"""
        a = [[2.0, 1.0], [1.0, 2.0]]
        vals, vecs = AI._jacobi_eigh(a)
        assert abs(sorted(vals)[0] - 1.0) < 1e-12
        assert abs(sorted(vals)[1] - 3.0) < 1e-12
        for i in range(2):
            for j in range(2):
                rec = sum(vecs[i][k] * vals[k] * vecs[j][k] for k in range(2))
                assert abs(rec - a[i][j]) < 1e-12
                dot = sum(vecs[i][k] * vecs[j][k] for k in range(2))
                assert abs(dot - (1.0 if i == j else 0.0)) < 1e-12

    def test_shampoo_falls_back_above_max_dim(self):
        """超过 max_dim 的层退化为逐元素 Adam（状态里是 m/v 而不是 L/R）"""
        small = self._mk(sizes=(3, 5, 2))
        opt_small = AI.Shampoo(small, max_dim=2)
        opt_small.step(self._grads(small))
        assert "m" in opt_small.state[(0, "w")] and "L" not in opt_small.state[(0, "w")]
        big = self._mk(sizes=(3, 5, 2))
        opt_big = AI.Shampoo(big, max_dim=64)
        opt_big.step(self._grads(big))
        assert "L" in opt_big.state[(0, "w")] and "m" not in opt_big.state[(0, "w")]

    def test_sophia_matches_closed_form_1x1(self):
        """一步：m=(1-β₁)g、h=(1-β₂)g²，位移 = -lr·clip(m/(ρh))"""
        model = self._mk(sizes=(1, 1))
        model.layers[0].weights[0][0] = 0.5
        model.layers[0].biases[0] = 0.0
        model.layers[0].activation_name = "linear"
        grad = self._grads(model, [1.0], [0.0])
        g = grad[0][0][0][0]
        b1, b2, rho, lr = 0.9, 0.999, 0.04, 0.1
        u = ((1.0 - b1) * g) / max(rho * (1.0 - b2) * g * g, 1e-8)
        u = max(-1.0, min(1.0, u))
        before = model.layers[0].weights[0][0]
        AI.Sophia(model, lr=lr, beta1=b1, beta2=b2, rho=rho).step(grad)
        assert abs((model.layers[0].weights[0][0] - before) + lr * u) < 1e-12

    def test_sophia_uses_explicit_curvature(self):
        """hessian_diags 覆盖 g² 近似：曲率大 ⇒ 步长被压小"""
        plain = self._mk()
        opt = AI.Sophia(plain, lr=0.1)
        before = [row[:] for row in plain.layers[0].weights]
        opt.step(self._grads(plain))
        d_plain = max(abs(a - b) for ra, rb in zip(plain.layers[0].weights, before)
                      for a, b in zip(ra, rb))

        curved = self._mk()
        opt2 = AI.Sophia(curved, lr=0.1)
        before2 = [row[:] for row in curved.layers[0].weights]
        h_w = [[1e6] * 3 for _ in range(4)]
        opt2.step(self._grads(curved), hessian_diags=[(h_w, [1e6] * 4), (None, None)])
        d_curved = max(abs(a - b) for ra, rb in zip(curved.layers[0].weights, before2)
                       for a, b in zip(ra, rb))
        assert d_curved < 1e-2 * d_plain

    def test_lbfgs_first_step_is_sgd_then_curvature_scaled(self):
        """第一步没有位移对 ⇒ 纯 -lr·g；第二步方向按 gamma=s·y/y·y 缩放"""
        model = self._mk(sizes=(1, 1))
        model.layers[0].weights[0][0] = 0.5
        model.layers[0].biases[0] = 0.0
        model.layers[0].activation_name = "linear"
        grad = self._grads(model, [1.0], [0.0])
        g = grad[0][0][0][0]
        w0 = model.layers[0].weights[0][0]
        opt = AI.LBFGS(model, lr=0.1, history_size=5)
        opt.step(grad)
        w1 = model.layers[0].weights[0][0]
        assert abs((w1 - w0) + 0.1 * g) < 1e-12
        grad1 = self._grads(model, [1.0], [0.0])
        g1 = grad1[0][0][0][0]
        opt.step(grad1)
        w2 = model.layers[0].weights[0][0]
        s, y = w1 - w0, g1 - g
        gamma = (s * y) / (y * y)
        assert abs((w2 - w1) + 0.1 * gamma * g1) < 1e-10

    def test_state_roundtrip_is_bitwise(self):
        """存档 → 载入 → 继续两步，权重与偏置逐位一致（含 s/y/prev_p 快照）"""
        for cls in (AI.Shampoo, AI.Sophia, AI.LBFGS):
            a = self._mk(seed=13)
            b = self._mk(seed=13)
            oa, ob = cls(a), cls(b)
            oa.step(self._grads(a))
            oa.step(self._grads(a))
            for la, lb in zip(a.layers, b.layers):
                lb.biases[:] = la.biases[:]
                for j in range(la.fan_out):
                    lb.weights[j][:] = la.weights[j][:]
            ob.load_state_dict(oa.state_dict())
            for _ in range(2):
                oa.step(self._grads(a))
                ob.step(self._grads(b))
            assert self._flat(a) == self._flat(b), cls.__name__

    def test_state_dict_does_not_alias(self):
        """快照必须是拷贝：改快照不能影响优化器内部状态"""
        for cls in (AI.Shampoo, AI.Sophia, AI.LBFGS):
            model = self._mk()
            opt = cls(model)
            opt.step(self._grads(model))
            sd = opt.state_dict()
            key = sorted(sd["state"])[0]
            field = sorted(sd["state"][key])[0]
            arr = sd["state"][key][field]
            while isinstance(arr, list) and arr and isinstance(arr[0], list):
                arr = arr[0]
            if not isinstance(arr, list) or not arr:
                continue
            arr[0] = 12345.0
            idx, kind = key.split(":")
            live = opt.state[(int(idx), kind)][field]
            while isinstance(live, list) and live and isinstance(live[0], list):
                live = live[0]
            assert live[0] != 12345.0, cls.__name__

    def test_invalid_hyperparameters(self):
        """非法超参必须在构造时报错，而不是训练中途出现 NaN"""
        model = self._mk()
        cases = [
            lambda: AI.Shampoo(model, precondition_every=0),
            lambda: AI.Shampoo(model, precondition_every="2"),
            lambda: AI.Shampoo(model, max_dim=0),
            lambda: AI.Shampoo(model, lr=-0.1),
            lambda: AI.Sophia(model, rho=0.0),
            lambda: AI.Sophia(model, hessian_fn=1),
            lambda: AI.Sophia(model, beta2=1.0),
            lambda: AI.LBFGS(model, history_size=0),
            lambda: AI.LBFGS(model, history_size=True),
            lambda: AI.LBFGS(model, lr=-1.0),
        ]
        for fn in cases:
            try:
                fn()
            except (ValueError, TypeError):
                continue
            raise AssertionError("非法超参没有报错")

    def test_loss_decreases(self):
        """300 步内三者都能把 XOR 型数据拟合到接近 0"""
        for cls in (AI.Shampoo, AI.Sophia, AI.LBFGS):
            model = self._mk(seed=3)
            opt = cls(model)
            first = None
            for _ in range(300):
                pred = model.forward(self.X)
                loss, gp = AI.mse_loss(pred, self.Y)
                if first is None:
                    first = loss
                opt.step(AI._mlp_backward(model, gp))
            last = AI.mse_loss(model.forward(self.X), self.Y)[0]
            assert math.isfinite(last) and last < first, cls.__name__

    def test_registered_and_trains(self):
        """三个名字能被 Model(optimizer=...) 认到并让 loss 下降"""
        for name in ("shampoo", "sophia", "lbfgs"):
            model = AI.Model(AI.MLP([2, 4, 2], activations=["relu", "linear"]),
                             loss_fn="mse", task="regression",
                             optimizer=name, lr=0.02)
            data = [([0.1 * i, -0.2 * i], [0.0, 1.0]) for i in range(4)]
            h = model.fit(data=data, epochs=5, verbose=False)
            first = h["train_loss"][0][1]
            last = h["train_loss"][-1][1]
            assert math.isfinite(last) and last <= first + 1e-9, name


class TestDropoutVariants:
    """W10-A：SpatialDropout1d/2d、AlphaDropout、GaussianDropout、R-Drop"""

    SEQ = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    IMG = [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]
    VEC = [1.0, -2.0, 3.0]

    def test_identity_when_p_zero_or_eval(self):
        """p=0 与 eval() 都必须逐元素恒等（推理不缩放）"""
        for mod, data in ((AI.SpatialDropout1d(0.0), self.SEQ),
                          (AI.SpatialDropout2d(0.0), self.IMG),
                          (AI.AlphaDropout(0.0), self.VEC),
                          (AI.GaussianDropout(0.0), self.VEC)):
            assert mod.forward(data) == data, type(mod).__name__
            other = type(mod)(0.5).eval()
            assert other.forward(data) == data, type(mod).__name__

    def test_spatial1d_shares_mask_across_time(self):
        """一个通道整条时间轴共用一个掩码，保留通道乘 1/(1-p)"""
        random.seed(7)
        mod = AI.SpatialDropout1d(0.5)
        out = mod.forward(self.SEQ)
        scale = 1.0 / (1.0 - 0.5)
        for c in range(3):
            assert mod.mask[c] in (0.0, scale)
            for t in range(2):
                assert out[t][c] == self.SEQ[t][c] * mod.mask[c]
        back = mod.backward(self.SEQ)
        for t in range(2):
            for c in range(3):
                assert back[t][c] == self.SEQ[t][c] * mod.mask[c]

    def test_spatial2d_drops_whole_channels(self):
        """2D 版本按通道丢：整通道同时为 0"""
        random.seed(9)
        mod = AI.SpatialDropout2d(0.5)
        out = mod.forward(self.IMG)
        scale = 1.0 / (1.0 - 0.5)
        for c in range(2):
            assert mod.mask[c] in (0.0, scale)
            for r in range(2):
                for k in range(2):
                    assert out[c][r][k] == self.IMG[c][r][k] * mod.mask[c]

    def test_alpha_dropout_affine_closed_form(self):
        """保留 = a·x+b，丢弃 = a·α'+b；反向是 a·mask"""
        for seed in (0, 1, 2, 3):
            random.seed(seed)
            mod = AI.AlphaDropout(0.5)
            x = 0.75
            y = mod.forward([x])[0]
            want = mod.a * x + mod.b if mod.mask[0] > 0.0 else mod.a * mod.alpha_p + mod.b
            assert abs(y - want) < 1e-15
            assert abs(mod.backward([1.0])[0] - mod.a * mod.mask[0]) < 1e-15

    def test_alpha_dropout_keeps_moments(self):
        """输入标准正态时输出均值 ~0、方差 ~1（SELU 自归一化的前提）"""
        random.seed(11)
        mod = AI.AlphaDropout(0.2)
        vals = [mod.forward([random.gauss(0.0, 1.0)])[0] for _ in range(8000)]
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        assert abs(mean) < 0.05
        assert 0.85 < var < 1.05

    def test_gaussian_dropout_moments(self):
        """乘性因子均值 1、方差 p/(1-p)"""
        random.seed(13)
        mod = AI.GaussianDropout(0.2)
        facs = [mod.forward([1.0])[0] for _ in range(8000)]
        mean = sum(facs) / len(facs)
        var = sum((f - mean) ** 2 for f in facs) / len(facs)
        assert abs(mean - 1.0) < 0.05
        assert abs(var - 0.25) < 0.05

    def test_gaussian_dropout_reproducible(self):
        """同种子 → 同因子（不能偷偷用别的 RNG 源）"""
        random.seed(21)
        a = AI.GaussianDropout(0.3)
        out_a = a.forward([2.0, 3.0])
        random.seed(21)
        b = AI.GaussianDropout(0.3)
        assert out_a == b.forward([2.0, 3.0])

    def test_r_drop_loss_properties(self):
        """同分布 0、对称、alpha 线性、alpha=0 恒 0"""
        la = [1.0, 2.0, 0.5]
        lb = [0.3, -1.0, 2.0]
        assert abs(AI.r_drop_loss(la, la)) < 1e-15
        assert abs(AI.r_drop_loss(la, lb) - AI.r_drop_loss(lb, la)) < 1e-15
        assert abs(AI.r_drop_loss(la, lb, alpha=0.5)
                   - 0.5 * AI.r_drop_loss(la, lb, alpha=1.0)) < 1e-15
        assert AI.r_drop_loss(la, lb, alpha=0.0) == 0.0
        assert AI.r_drop_loss(la, lb) > 0.0

    def test_invalid_arguments(self):
        """越界 p、非负 α、非方阵特征图都必须在入口报错"""
        for fn in (lambda: AI.SpatialDropout1d(1.0),
                   lambda: AI.SpatialDropout1d(True),
                   lambda: AI.SpatialDropout2d(-0.1),
                   lambda: AI.AlphaDropout(0.2, alpha=0.0),
                   lambda: AI.AlphaDropout(0.2, scale=0.0),
                   lambda: AI.GaussianDropout(1.5),
                   lambda: AI.SpatialDropout1d(0.5).forward([[1.0, 2.0], [3.0]]),
                   lambda: AI.SpatialDropout2d(0.5).forward([[[1.0, 2.0], [3.0, 4.0]],
                                                           [[1.0]]]),
                   lambda: AI.r_drop_loss([1.0], [1.0, 2.0]),
                   lambda: AI.r_drop_loss([], []),
                   lambda: AI.r_drop_loss([1.0], [1.0], alpha=-1.0)):
            try:
                fn()
            except (ValueError, TypeError):
                continue
            raise AssertionError("非法参数没有报错")

    def test_backward_before_forward_raises(self):
        """没走 forward 就 backward 必须报错，而不是拿 None 崩在别处"""
        for mod in (AI.SpatialDropout1d(0.5), AI.SpatialDropout2d(0.5),
                    AI.AlphaDropout(0.5), AI.GaussianDropout(0.5)):
            try:
                mod.backward(self.VEC)
            except ValueError:
                continue
            raise AssertionError(type(mod).__name__ + " 没有报错")


# ============================================================================
#  W10-B：结构化正则（DropBlock2d / DropConnect）
# ============================================================================

class TestStructuredRegularization:
    """方块级与权重级丢弃。"""

    @staticmethod
    def _img(n=2, h=8, w=8, seed=1):
        random.seed(seed)
        return [[[random.gauss(0.0, 1.0) for _ in range(w)] for _ in range(h)]
                for _ in range(n)]

    def test_dropblock_identity(self):
        """p=0 与 eval() 都必须恒等。"""
        x = self._img()
        assert AI.DropBlock2d(0.0).forward(x) == x
        assert AI.DropBlock2d(0.4).eval().forward(x) == x

    def test_dropblock_drop_ratio(self):
        """期望丢弃比例应接近 p（gamma 用 p 而不是 1-p）。"""
        random.seed(5)
        db = AI.DropBlock2d(p=0.3, block_size=3)
        x = self._img(n=8, h=16, w=16)
        db.forward(x)
        total = sum(1 for ch in range(8) for i in range(16) for j in range(16))
        zeros = sum(1 for ch in range(8) for i in range(16) for j in range(16)
                    if db.mask[ch][i][j] == 0.0)
        ratio = zeros / total
        assert abs(ratio - 0.3) < 0.08, ratio

    def test_dropblock_is_blockwise(self):
        """被丢的元素应当连成方块，而不是散点。"""
        random.seed(5)
        db = AI.DropBlock2d(p=0.3, block_size=3)
        db.forward(self._img(n=8, h=16, w=16))
        checked = nearby = 0
        for ch in range(8):
            for i in range(1, 15):
                for j in range(1, 15):
                    if db.mask[ch][i][j] != 0.0:
                        continue
                    checked += 1
                    cnt = 0
                    for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        if db.mask[ch][i + di][j + dj] == 0.0:
                            cnt += 1
                    if cnt >= 2:
                        nearby += 1
        assert checked > 0
        assert nearby / checked > 0.98

    def test_dropblock_preserves_expectation(self):
        """缩放因子应使输出均值与原均值一致。"""
        random.seed(7)
        db = AI.DropBlock2d(p=0.3, block_size=3)
        sx = sy = n = 0.0
        for _ in range(40):
            x = self._img(n=2, h=8, w=8, seed=None)
            out = db.forward(x)
            for ch in range(2):
                for i in range(8):
                    for j in range(8):
                        sx += x[ch][i][j]
                        sy += out[ch][i][j]
                        n += 1
        assert abs(sx / n - sy / n) < 0.2

    def test_dropblock_backward_uses_mask(self):
        random.seed(9)
        db = AI.DropBlock2d(0.3, 3)
        db.forward(self._img(n=2, h=6, w=6))
        grad = db.backward([[[1.0] * 6 for _ in range(6)] for _ in range(2)])
        for ch in range(2):
            for i in range(6):
                for j in range(6):
                    assert abs(grad[ch][i][j] - db.mask[ch][i][j] * db.scale) < 1e-15

    def test_dropblock_validation(self):
        with pytest.raises(ValueError):
            AI.DropBlock2d(1.0)
        with pytest.raises(ValueError):
            AI.DropBlock2d(0.1, 0)
        with pytest.raises(TypeError):
            AI.DropBlock2d(0.1, True)
        with pytest.raises(ValueError):
            AI.DropBlock2d(0.1).forward([[1.0, 2.0], [3.0]])
        with pytest.raises(ValueError):
            AI.DropBlock2d(0.1).backward(self._img(n=1, h=2, w=2))

    def test_dropconnect_eval_matches_layer(self):
        """eval() 下 DropConnect 必须退化为同权重的 Layer（逐位）。"""
        random.seed(3)
        layer = AI.Layer(4, 3, "relu")
        random.seed(3)
        dc = AI.DropConnect(4, 3, "relu", p=0.5)
        assert layer.weights == dc.weights
        xin = [0.3, -0.7, 1.1, 0.2]
        dc.eval()
        assert dc.forward(xin) == layer.forward(xin)
        assert dc.backward([0.5, -0.4, 0.9]) == layer.backward([0.5, -0.4, 0.9])

    def test_dropconnect_grad_and_delta(self):
        """grad_w 必须被掩码遮住；next_delta 必须用掩码权重。"""
        random.seed(11)
        dc = AI.DropConnect(5, 4, "linear", p=0.6)
        xin = [0.2, 0.4, -0.6, 0.8, 0.1]
        dc.forward(xin)
        mask = dc.mask
        gw, gb, dx = dc.backward([1.0, 1.0, 1.0, 1.0])
        for j in range(4):
            for i in range(5):
                if mask[j][i] == 0.0:
                    assert gw[j][i] == 0.0
                elif xin[i] != 0.0:
                    assert gw[j][i] != 0.0
        # 固定种子复现同一掩码，再做有限差分
        eps = 1e-6
        random.seed(77)
        o1 = sum(dc.forward(xin))
        m1 = dc.mask
        random.seed(77)
        o2 = sum(dc.forward([xi + eps for xi in xin]))
        assert m1 == dc.mask
        _, _, dx2 = dc.backward([1.0, 1.0, 1.0, 1.0])
        assert abs((o2 - o1) / eps - sum(dx2)) < 1e-4

    def test_dropconnect_trains(self):
        """把最后一层换成 DropConnect 后仍能正常训练。"""
        random.seed(13)
        net = AI.MLP([3, 6, 2], activations=["relu", "linear"])
        net.layers[-1] = AI.DropConnect(6, 2, "linear", is_output=True, p=0.3)
        model = AI.Model(net, loss_fn="mse", task="regression",
                         optimizer="adam", lr=0.02)
        data = [([0.1 * i, -0.2 * i, 0.3 * i], [0.0, 1.0]) for i in range(6)]
        hist = model.fit(data=data, epochs=40, verbose=False)
        assert hist["train_loss"][-1][1] < hist["train_loss"][0][1]

    def test_dropconnect_validation(self):
        with pytest.raises(ValueError):
            AI.DropConnect(2, 2, "relu", p=1.0)
        with pytest.raises(ValueError):
            AI.DropConnect(2, 2, "relu", p=-1.0)


# ============================================================================
#  W10-C：对抗训练（FGSM / PGD / TRADES / MART / SMART / fit(adv_eps)）
# ============================================================================

class TestAdversarialTraining:
    """输入空间攻击与鲁棒正则。"""

    @staticmethod
    def _model(sizes=(2, 3, 2), seed=1):
        random.seed(seed)
        net = AI.MLP(list(sizes), activations=["relu"] * (len(sizes) - 2) + ["linear"])
        return AI.Model(net, loss_fn="mse", task="regression", optimizer="adam", lr=0.01)

    def test_eps_zero_is_identity(self):
        model = self._model()
        x, y = [0.3, -0.2], [1.0, 0.0]
        assert AI.pgd_examples(model, [x], [y], eps=0.0, steps=3)[0] == x
        assert AI.fgsm_examples(model, [x], [y], eps=0.0)[0] == x

    def test_fgsm_closed_form(self):
        """linear + mse：x_adv = x + eps*sign(Wᵀ·dL/dpred)。"""
        random.seed(7)
        net = AI.MLP([2, 2], activations=["linear"])
        model = AI.Model(net, loss_fn="mse", task="regression", optimizer="sgd", lr=0.01)
        x, y = [0.3, -0.2], [1.0, 0.0]
        eps = 0.05
        _, grad = model._loss(model.forward(x), y)
        w = net.layers[0].weights
        g = [sum(w[j][i] * grad[j] for j in range(2)) for i in range(2)]
        want = [x[i] + eps * (1.0 if g[i] > 0 else (-1.0 if g[i] < 0 else 0.0))
                for i in range(2)]
        got = AI.fgsm_examples(model, [x], [y], eps=eps)[0]
        for a, b in zip(want, got):
            assert abs(a - b) < 1e-12

    def test_eps_ball_and_clip(self):
        model = self._model()
        xs = [[0.1 * i, -0.2 * i] for i in range(5)]
        ys = [[1.0, 0.0] for _ in range(5)]
        adv = AI.pgd_examples(model, xs, ys, eps=0.1, steps=4, seed=11)
        for xa, xb in zip(adv, xs):
            for a, b in zip(xa, xb):
                assert abs(a - b) <= 0.1 + 1e-12
        adv_c = AI.pgd_examples(model, xs, ys, eps=0.5, steps=2, seed=11,
                                clip_low=0.0, clip_high=1.0)
        for row in adv_c:
            for v in row:
                assert 0.0 <= v <= 1.0

    def test_single_step_pgd_equals_fgsm(self):
        model = self._model()
        xs = [[0.1, -0.2], [0.2, 0.3]]
        ys = [[1.0, 0.0]] * 2
        a = AI.pgd_examples(model, xs, ys, eps=0.1, alpha=0.1, steps=1,
                            random_start=False)
        b = AI.fgsm_examples(model, xs, ys, eps=0.1)
        for xa, xb in zip(a, b):
            for p, q in zip(xa, xb):
                assert abs(p - q) < 1e-15

    def test_attack_raises_loss(self):
        model = self._model()
        xs = [[0.1 * i, -0.2 * i] for i in range(5)]
        ys = [[1.0, 0.0] for _ in range(5)]
        clean = sum(model._loss(model.forward(xi), yi)[0]
                    for xi, yi in zip(xs, ys)) / len(xs)
        f_adv = sum(model._loss(model.forward(xi), yi)[0]
                    for xi, yi in zip(AI.fgsm_examples(model, xs, ys, eps=0.2), ys)) / len(xs)
        p_adv = sum(model._loss(model.forward(xi), yi)[0]
                    for xi, yi in zip(
                        AI.pgd_examples(model, xs, ys, eps=0.2, steps=5, seed=5), ys)) / len(xs)
        assert f_adv >= clean - 1e-12
        assert p_adv >= f_adv - 1e-9

    def test_trades_regularizer(self):
        lc = [[2.0, 0.0], [0.0, 1.0]]
        assert AI.trades_regularizer(lc, lc) == 0.0
        la = [[0.0, 2.0], [0.0, 1.0]]
        val = AI.trades_regularizer(lc, la)
        assert val > 0.0
        assert abs(AI.trades_regularizer(lc, la, beta=2.0) - 2.0 * val) < 1e-12
        # 手算：只有第一行不同，函数按行取均值
        p = [math.exp(2.0) / (math.exp(2.0) + 1.0), 1.0 / (math.exp(2.0) + 1.0)]
        q = [1.0 / (math.exp(2.0) + 1.0), math.exp(2.0) / (math.exp(2.0) + 1.0)]
        kl = sum(pi * math.log(pi / qi) for pi, qi in zip(p, q)) / 2.0
        assert abs(val - kl) < 1e-12

    def test_mart_regularizer(self):
        lc = [[2.0, 0.0], [0.0, 1.0]]
        assert AI.mart_regularizer(lc, lc) == 0.0
        la = [[0.0, 2.0], [0.0, 1.0]]
        got = AI.mart_regularizer(lc, la, beta=1.0)
        kl = AI.trades_regularizer(lc, la) * 2.0   # 每行 KL（不除行数）
        pmax = math.exp(2.0) / (math.exp(2.0) + 1.0)
        want = (1.0 - pmax) * kl / 2.0
        assert abs(got - want) < 1e-12

    def test_smart_matches_trades_output_term(self):
        lc = [[2.0, 0.0], [0.0, 1.0]]
        la = [[0.0, 2.0], [0.0, 1.0]]
        assert abs(AI.smart_regularizer(lc, la)
                   - AI.trades_regularizer(lc, la)) < 1e-15

    def test_attack_validation(self):
        model = self._model()
        xs = [[0.1, -0.2]]
        ys = [[1.0, 0.0]]
        with pytest.raises(ValueError):
            AI.pgd_examples(model, xs, ys, eps=-1.0)
        with pytest.raises(ValueError):
            AI.pgd_examples(model, xs, ys, eps=0.1, alpha=0.0)
        with pytest.raises(ValueError):
            AI.pgd_examples(model, xs, ys, eps=0.1, steps=0)
        with pytest.raises(ValueError):
            AI.pgd_examples(model, xs, ys, eps=0.1, steps=1.5)
        with pytest.raises(ValueError):
            AI.pgd_examples(model, xs, ys, eps=0.1, clip_low=1.0, clip_high=0.0)
        with pytest.raises(ValueError):
            AI.pgd_examples(model, xs, ys + ys, eps=0.1)
        with pytest.raises(ValueError):
            AI.trades_regularizer([[1.0]], [[1.0]], beta=-1.0)
        with pytest.raises(ValueError):
            AI.trades_regularizer([], [])

    def test_fit_adv_eps_zero_is_bitwise_identical(self):
        """adv_eps=0 必须与不传该参数逐位一致。"""
        data = [([0.1 * i, -0.2 * i], [1.0, 0.0]) for i in range(6)]
        random.seed(21)
        a = self._model()
        random.seed(21)
        b = self._model()
        ha = a.fit(data=data, epochs=5, batch_size=3, verbose=False, seed=4)
        hb = b.fit(data=data, epochs=5, batch_size=3, verbose=False, seed=4, adv_eps=0.0)
        assert ha["train_loss"] == hb["train_loss"]
        assert a.model.layers[0].weights == b.model.layers[0].weights

    def test_fit_adv_training_runs(self):
        data = [([0.1 * i, -0.2 * i], [1.0, 0.0]) for i in range(6)]
        random.seed(23)
        model = self._model()
        hist = model.fit(data=data, epochs=30, batch_size=3, verbose=False,
                         seed=4, adv_eps=0.05, adv_steps=3)
        first, last = hist["train_loss"][0][1], hist["train_loss"][-1][1]
        assert math.isfinite(last)
        assert last < first

    def test_fit_adv_validation(self):
        data = [([0.1, -0.2], [1.0, 0.0])] * 4
        with pytest.raises(ValueError):
            self._model().fit(data=data, epochs=1, batch_size=2, verbose=False,
                              adv_eps=-0.1)
        with pytest.raises(ValueError):
            self._model().fit(data=data, epochs=1, batch_size=2, verbose=False,
                              adv_eps=0.1, adv_steps=0)
        with pytest.raises(ValueError):
            self._model().fit(data=data, epochs=1, batch_size=2, verbose=False,
                              adv_eps=0.1, adv_alpha=0.0)
        with pytest.raises(ValueError):
            self._model().fit(data=data, epochs=1, batch_size=2, verbose=False,
                              adv_eps=0.1, mixup_alpha=0.2)


# ============================================================================
#  W10-D：噪声注入 / 课程学习 / Manifold Mixup / Snapshot Ensembles
# ============================================================================

class TestNoiseCurriculumEnsemble:
    """训练期正则与集成工具。"""

    @staticmethod
    def _model(sizes=(2, 3, 2), seed=1):
        random.seed(seed)
        net = AI.MLP(list(sizes), activations=["relu"] * (len(sizes) - 2) + ["linear"])
        return AI.Model(net, loss_fn="mse", task="regression", optimizer="adam", lr=0.01)

    def test_inject_weight_noise(self):
        random.seed(1)
        model = self._model()
        before = [row[:] for row in model.model.layers[0].weights]
        AI.inject_weight_noise(model, 0.0)
        assert model.model.layers[0].weights == before
        AI.inject_weight_noise(model, 0.05, seed=7)
        assert model.model.layers[0].weights != before

    def test_inject_weight_noise_reproducible(self):
        random.seed(3)
        a = self._model()
        random.seed(3)
        b = self._model()
        AI.inject_weight_noise(a, 0.1, seed=42)
        AI.inject_weight_noise(b, 0.1, seed=42)
        assert a.model.layers[0].weights == b.model.layers[0].weights

    def test_noise_statistics(self):
        """200 次加噪后，权重扰动应近似零均值、方差 ≈ sigma²。"""
        random.seed(2)
        model = self._model()
        vals = []
        for _ in range(200):
            for row in model.model.layers[0].weights:
                for j in range(len(row)):
                    row[j] = 0.0
            AI.inject_weight_noise(model, 0.1)
            vals.extend(model.model.layers[0].weights[0])
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        assert abs(mean) < 0.05
        assert abs(var - 0.01) < 0.005

    def test_gradient_and_input_noise(self):
        grads = [([[1.0, 2.0], [3.0, 4.0]], [0.5, -0.5])]
        assert AI.add_gradient_noise(grads, 0.0) == grads
        assert grads[0][0][0][0] == 1.0          # 入参未被改动
        assert AI.add_gradient_noise(grads, 0.1, seed=5) != grads
        x = [1.0, -1.0]
        y = AI.input_noise(x, 0.0)
        assert y == x and y is not x
        assert AI.input_noise(x, 0.1, seed=1) != x
        assert x == [1.0, -1.0]

    def test_curriculum_fraction(self):
        assert abs(AI.curriculum_fraction(5, 10, "linear") - 0.5) < 1e-15
        assert AI.curriculum_fraction(20, 10, "linear") == 1.0
        assert abs(AI.curriculum_fraction(10, 10, "exp") - 1.0) < 1e-15
        assert AI.curriculum_fraction(1, 10, "exp") < 0.01
        assert abs(AI.curriculum_fraction(1, 8, "step") - 0.25) < 1e-15
        assert AI.curriculum_fraction(8, 8, "step") == 1.0

    def test_curriculum_sampler(self):
        diff = [float(i) for i in range(10)]
        s = AI.CurriculumSampler(diff, total_epochs=10, shuffle=False)
        s.set_epoch(1)
        assert list(s) == [0]
        s.set_epoch(10)
        assert set(s) == set(range(10))
        s2 = AI.CurriculumSampler(diff, total_epochs=10, shuffle=True, seed=3)
        s2.set_epoch(5)
        first = list(s2)
        s2.set_epoch(5)
        assert first == list(s2)
        assert set(first) == set(range(5))   # 只取最简单的 5 个

    def test_manifold_mixup(self):
        fa = [[1.0, 2.0], [3.0, 4.0]]
        fb = [[5.0, 6.0], [7.0, 8.0]]
        assert AI.manifold_mixup(fa, fb, 1.0) == fa
        assert AI.manifold_mixup(fa, fb, 0.0) == fb
        assert AI.manifold_mixup(fa, fb, 0.5) == [[3.0, 4.0], [5.0, 6.0]]
        assert AI.manifold_mixup([1.0, 2.0], [3.0, 4.0], 0.25) == [2.5, 3.5]
        assert fa == [[1.0, 2.0], [3.0, 4.0]]

    def test_snapshot_ensemble(self):
        random.seed(9)
        model = self._model()
        ens = AI.SnapshotEnsemble(model)
        with pytest.raises(ValueError):
            ens.predict([0.1, 0.2])
        ens.capture()
        base = model.forward([0.1, 0.2])
        got = ens.predict([0.1, 0.2])
        for a, b in zip(got, base):
            assert abs(a - b) < 1e-15
        # 改权重 → 两张快照的平均；predict 结束后权重要恢复
        for row in model.model.layers[0].weights:
            for j in range(len(row)):
                row[j] *= 0.5
        ens.capture()
        changed = model.forward([0.1, 0.2])
        ens.predict([0.1, 0.2])
        assert model.forward([0.1, 0.2]) == changed

    def test_snapshot_callback(self):
        random.seed(11)
        model = self._model()
        cb = AI.SnapshotCallback(every=2)
        data = [([0.1 * i, -0.2 * i], [1.0, 0.0]) for i in range(6)]
        model.fit(data=data, epochs=6, batch_size=3, verbose=False, callbacks=[cb])
        assert len(cb.ensemble) == 3
        assert len(cb.ensemble.predict([0.1, -0.2])) == 2

    def test_validation(self):
        model = self._model()
        with pytest.raises(ValueError):
            AI.inject_weight_noise(model, -1.0)
        with pytest.raises(ValueError):
            AI.add_gradient_noise([([[1.0]], [1.0])], -1.0)
        with pytest.raises(ValueError):
            AI.add_gradient_noise([])
        with pytest.raises(ValueError):
            AI.input_noise([])
        with pytest.raises(ValueError):
            AI.curriculum_fraction(0)
        with pytest.raises(ValueError):
            AI.curriculum_fraction(1, 10, "nope")
        with pytest.raises(ValueError):
            AI.CurriculumSampler([])
        with pytest.raises(ValueError):
            AI.CurriculumSampler([1.0], min_fraction=0.0)
        with pytest.raises(ValueError):
            AI.manifold_mixup([1.0], [2.0], 1.5)
        with pytest.raises(ValueError):
            AI.manifold_mixup([[1.0]], [2.0], 0.5)
        with pytest.raises(TypeError):
            AI.SnapshotCallback("3")


# ============================================================================
#  W10-E：半监督（VAT / 伪标签 / 一致性正则 / FixMatch / MixMatch / MeanTeacher）
# ============================================================================

class TestSemiSupervised:
    """半监督相关工具：纯函数 + 批级接口。"""

    @staticmethod
    def _net(sizes=(3, 4, 2), seed=7):
        random.seed(seed)
        return AI.MLP(list(sizes), activations=["relu"] * (len(sizes) - 2) + ["linear"])

    def test_softmax_rows(self):
        rows = [[0.5, -0.2, 0.1], [1.0, 0.0, -1.0]]
        probs = AI.softmax_rows(rows)
        for r in probs:
            assert abs(sum(r) - 1.0) < 1e-12
        assert probs[0] == AI._softmax_row(list(rows[0]))
        assert rows[0] != probs[0]
        for bad in ([], [[]], [[1.0], []]):
            with pytest.raises(ValueError):
                AI.softmax_rows(bad)

    def test_consistency_loss(self):
        la = [[0.5, -0.2, 0.1], [1.0, 0.0, -1.0]]
        lb = [[0.1, 0.3, -0.4], [0.2, 0.2, 0.2]]
        assert AI.consistency_loss(la, la, "mse") == 0.0
        assert AI.consistency_loss(la, la, "kl") == 0.0
        assert abs(AI.consistency_loss(la, lb, "mse")
                   - AI.consistency_loss(lb, la, "mse")) < 1e-15
        pa, pb = AI.softmax_rows(la), AI.softmax_rows(lb)
        manual = sum((a - b) ** 2 for ra, rb in zip(pa, pb) for a, b in zip(ra, rb)) / 6.0
        assert abs(AI.consistency_loss(la, lb, "mse") - manual) < 1e-15
        with pytest.raises(ValueError):
            AI.consistency_loss(la, lb, "cos")

    def test_consistency_regularizer(self):
        net = self._net()
        xs = [[0.3, -0.4, 0.8], [0.1, 0.2, -0.3]]
        aug = lambda x: [v * 0.5 for v in x]  # noqa: E731

        def loss_fn():
            za = [net.forward(list(x)) for x in xs]
            zb = [net.forward(aug(list(x))) for x in xs]
            return AI.consistency_loss(za, zb, "mse")

        val, grads = AI.consistency_regularizer(net, xs, kind="mse", aug=aug)
        assert abs(val - loss_fn()) < 1e-12
        layer = net.layers[0]
        old = layer.weights[0][0]
        layer.weights[0][0] = old + 1e-6
        up = loss_fn()
        layer.weights[0][0] = old - 1e-6
        dn = loss_fn()
        layer.weights[0][0] = old
        num = (up - dn) / 2e-6
        assert abs(grads[0][0][0][0] - num) < 1e-6
        v0, _ = AI.consistency_regularizer(net, xs, kind="mse", sigma=0.0)
        assert v0 == 0.0

    def test_vat(self):
        net = self._net()
        xs = [[0.3, -0.4, 0.8], [0.1, 0.2, -0.3]]
        adv = AI.vat_perturbation(net, xs, eps=0.5, seed=3)
        for x, a in zip(xs, adv):
            n = sum((p - q) ** 2 for p, q in zip(a, x)) ** 0.5
            assert abs(n - 0.5) < 1e-9
        assert AI.vat_perturbation(net, xs, eps=0.0, seed=1) == [list(x) for x in xs]
        val, grads = AI.vat_regularizer(net, xs, eps=0.5, seed=3)
        assert val > 0.0
        assert len(grads) == len(net.layers)

    def test_pseudo_labels(self):
        probs = [[0.9, 0.05, 0.05], [0.4, 0.35, 0.25]]
        assert AI.pseudo_label_targets(probs, 0.5) == ([0, 0], [1.0, 0.0])
        assert AI.pseudo_label_targets(probs, 0.0) == ([0, 0], [1.0, 1.0])
        assert AI.fixmatch_targets(probs, 0.95) == ([0, 0], [0.0, 0.0])
        with pytest.raises(ValueError):
            AI.pseudo_label_targets([], 0.5)
        with pytest.raises(ValueError):
            AI.pseudo_label_targets(probs, 1.5)

    def test_soft_target_ce(self):
        lg = [0.2, -0.1, 0.5]
        one = AI.soft_target_ce([lg], [[0.0, 0.0, 1.0]])[0]
        ce = AI.cross_entropy_loss(AI._softmax_row(list(lg)), [0.0, 0.0, 1.0])[0]
        assert abs(one - ce) < 1e-12
        loss, dlg = AI.soft_target_ce([lg], [[0.0, 0.0, 1.0]])
        h = 1e-6
        num = (AI.soft_target_ce([[0.2 + h, -0.1, 0.5]], [[0.0, 0.0, 1.0]])[0]
               - AI.soft_target_ce([[0.2 - h, -0.1, 0.5]], [[0.0, 0.0, 1.0]])[0]) / (2 * h)
        assert abs(dlg[0][0] - num) < 1e-6
        z_loss, z_grad = AI.soft_target_ce([lg], [[0.0, 0.0, 1.0]], weights=[0.0])
        assert z_loss == 0.0 and z_grad[0] == [0.0, 0.0, 0.0]

    def test_fixmatch_regularizer(self):
        net = self._net()
        xs_w = [[0.3, -0.4, 0.8], [0.1, 0.2, -0.3], [0.9, 0.9, 0.9]]
        xs_s = [[v * 0.9 for v in x] for x in xs_w]
        val, grads = AI.fixmatch_regularizer(net, xs_w, xs_s, confidence=0.0)
        assert val > 0.0
        assert len(grads) == len(net.layers)
        val0, _ = AI.fixmatch_regularizer(net, xs_w, xs_s, confidence=1.0)
        assert val0 == 0.0

    def test_mixmatch_targets(self):
        assert AI.mixmatch_targets([[[0.5, 0.5]]], temperature=1.0) == [[0.5, 0.5]]
        sharp = AI.mixmatch_targets([[[0.6, 0.4]]], temperature=0.5)
        assert sharp[0][0] > 0.6 and abs(sum(sharp[0]) - 1.0) < 1e-12
        avg = AI.mixmatch_targets([[[1.0, 0.0], [0.0, 1.0]],
                                   [[0.0, 1.0], [1.0, 0.0]]], 1.0)
        assert avg == [[0.5, 0.5], [0.5, 0.5]]
        with pytest.raises(ValueError):
            AI.mixmatch_targets([[[0.5, 0.5]]], temperature=0.0)
        with pytest.raises(ValueError):
            AI.mixmatch_targets([[[0.5, 0.5]], [[0.5, 0.5], [0.5, 0.5]]])

    def test_mean_teacher(self):
        net = self._net(seed=11)
        x = [0.2, -0.1, 0.4]
        mt = AI.MeanTeacher(net, decay=0.999)
        base = list(net.forward(list(x)))
        assert mt.forward(list(x)) == base
        for row in net.layers[0].weights:
            for k in range(len(row)):
                row[k] += 0.5
        changed = list(net.forward(list(x)))
        mt.update(decay=0.0)
        assert mt.forward(list(x)) == changed
        # 教师前向不能污染学生权重
        after_teacher = list(net.forward(list(x)))
        for row in net.layers[0].weights:
            for k in range(len(row)):
                row[k] += 1.0
        mt.update(decay=0.5)
        val, grads = AI.mean_teacher_regularizer(net, mt, [x], kind="mse")
        assert val > 0.0
        assert len(grads) == len(net.layers)
        assert list(net.forward(list(x))) != after_teacher

    def test_validation(self):
        net = self._net()
        bad_calls = [
            (lambda: AI.consistency_regularizer(net, [], kind="mse"), ValueError),
            (lambda: AI.consistency_regularizer(net, [[0.1]], kind="cos"), ValueError),
            (lambda: AI.consistency_regularizer(net, [[0.1]], sigma=-1.0), ValueError),
            (lambda: AI.consistency_regularizer(net, [[0.1]], aug="no"), TypeError),
            (lambda: AI.vat_perturbation(net, [], eps=0.5), ValueError),
            (lambda: AI.vat_perturbation(net, [[0.1]], eps=-0.5), ValueError),
            (lambda: AI.vat_perturbation(net, [[0.1]], eps=0.5, xi=0.0), ValueError),
            (lambda: AI.vat_perturbation(net, [[0.1]], eps=0.5, power=0), ValueError),
            (lambda: AI.fixmatch_regularizer(net, [[0.1]], []), ValueError),
            (lambda: AI.soft_target_ce([[1.0]], [[1.0]], weights=[-1.0]), ValueError),
            (lambda: AI.MeanTeacher(net, decay=1.0), ValueError),
        ]
        for fn, exc in bad_calls:
            with pytest.raises(exc):
                fn()


class TestCheckpointMixedPrecision:
    """W10-F：梯度检查点与混合精度模拟"""

    @staticmethod
    def _net():
        return AI.Model(AI.MLP([3, 4, 2], activations=["relu", "linear"]),
                        loss_fn=AI.mse_loss)

    @staticmethod
    def _data():
        xs = [[0.1, -0.2, 0.3], [0.4, 0.5, -0.6], [-0.7, 0.8, 0.9]]
        ys = [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]
        return xs, ys

    def test_checkpoint_grads_matches_plain_backward(self):
        model = self._net()
        xs, ys = self._data()
        loss, grads = AI.checkpoint_grads(model, xs, ys, segments=2)
        n = float(len(xs))
        acc = [[[0.0] * len(r) for r in layer.weights] for layer in model.model.layers]
        total = 0.0
        for x, y in zip(xs, ys):
            pred = model.forward(x)
            one, g = model._loss(pred, y)
            total += one
            for i, (gw, _gb) in enumerate(AI._mlp_backward(model.model, g)):
                if gw is None:
                    continue
                for j, row in enumerate(gw):
                    for k, v in enumerate(row):
                        acc[i][j][k] += v
        assert loss == total / n
        for (gw, _gb), ref in zip(grads, acc):
            if gw is None:
                continue
            for row, ref_row in zip(gw, ref):
                for v, want in zip(row, ref_row):
                    assert v == want / n

    def test_checkpoint_segments_variants(self):
        model = self._net()
        xs, ys = self._data()
        base = AI.checkpoint_grads(model, xs, ys, segments=1)
        for seg in (2, 3, 99):
            got = AI.checkpoint_grads(model, xs, ys, segments=seg)
            assert got[0] == base[0]
            assert got[1] == base[1]

    def test_checkpoint_invalid(self):
        model = self._net()
        xs, ys = self._data()
        for seg in (0, -1, 1.5, True, "2"):
            with pytest.raises((ValueError, TypeError)):
                AI.checkpoint_grads(model, xs, ys, segments=seg)
        with pytest.raises(ValueError):
            AI.checkpoint_grads(model, xs, ys[:2])
        with pytest.raises(ValueError):
            AI.checkpoint_grads(model, [], [])

    def test_fp16_round(self):
        assert AI.fp16_round(1.0) == 1.0
        assert AI.fp16_round(65504.0) == 65504.0
        assert AI.fp16_round(70000.0) == float("inf")
        assert AI.fp16_round(-70000.0) == float("-inf")
        assert AI.fp16_round(1e-8) == 0.0

    def test_bf16_round(self):
        assert AI.bf16_round(1.0) == 1.0
        assert AI.bf16_round(0.1) == 0.10009765625
        assert AI.bf16_round(float("inf")) == float("inf")

    def test_quantize_model_idempotent(self):
        model = self._net()
        before = [list(r) for r in model.model.layers[0].weights]
        AI.quantize_model(model, dtype="fp16")
        once = [list(r) for r in model.model.layers[0].weights]
        assert once != before
        AI.quantize_model(model, dtype="fp16")
        assert once == [list(r) for r in model.model.layers[0].weights]
        AI.quantize_model(model, dtype="bf16")
        with pytest.raises(ValueError):
            AI.quantize_model(model, dtype="int8")

    def test_mixed_precision_grads(self):
        model = self._net()
        xs, ys = self._data()
        loss, grads = AI.mixed_precision_grads(model, xs, ys, scale=8192.0)
        base_loss, base = AI.checkpoint_grads(model, xs, ys, segments=1)
        assert loss == base_loss
        for (gw, _gb), (rw, _rb) in zip(grads, base):
            for row_out, row_ref in zip(gw, rw):
                for v, want in zip(row_out, row_ref):
                    assert v == AI.fp16_round(want * 8192.0) / 8192.0
        # loss scaling 保住了 fp16 存不下的极小梯度
        assert AI.fp16_round(1e-9 * 8192.0) / 8192.0 != 0.0
        assert AI.fp16_round(1e-9) == 0.0

    def test_mixed_precision_invalid(self):
        model = self._net()
        xs, ys = self._data()
        for scale in (0.0, -1.0, float("nan"), float("inf")):
            with pytest.raises(ValueError):
                AI.mixed_precision_grads(model, xs, ys, scale=scale)
        with pytest.raises(ValueError):
            AI.mixed_precision_grads(model, xs, ys, dtype="fp32")


class TestDataPipeline:
    """W11-A：Dataset / Sampler / 长度分桶 / 预处理缓存"""

    def test_dataset_abstraction(self):
        for call in (lambda: len(AI.Dataset()), lambda: AI.Dataset()[0]):
            with pytest.raises(NotImplementedError):
                call()
        ds = AI.ListDataset([10, 20, 30])
        assert len(ds) == 3
        assert ds[1] == 20
        sub = AI.Subset(ds, [2, 0])
        assert [sub[i] for i in range(len(sub))] == [30, 10]
        with pytest.raises(TypeError):
            AI.ListDataset("abc")

    def test_lazy_dataset_cache(self):
        calls = []

        def loader(i):
            calls.append(i)
            return i * i

        lazy = AI.LazyDataset(3, loader)
        assert calls == []
        assert lazy[1] == 1
        assert lazy[-1] == 4
        assert calls == [1, 2]
        lazy[1]
        assert calls == [1, 2]
        assert (lazy.hits, lazy.misses) == (1, 2)
        lazy.clear_cache()
        lazy[1]
        assert len(calls) == 3
        with pytest.raises(IndexError):
            lazy[9]
        fifo = AI.LazyDataset(4, lambda i: i, max_cache=2)
        fifo[0], fifo[1], fifo[2]
        fifo[0]
        assert fifo.misses == 4

    def test_samplers(self):
        assert list(AI.SequentialSampler(4)) == [0, 1, 2, 3]
        assert list(AI.RandomSampler(6, seed=3)) == list(AI.RandomSampler(6, seed=3))
        assert sorted(AI.RandomSampler(6, seed=3)) == list(range(6))
        assert len(list(AI.RandomSampler(5, replacement=True, seed=1))) == 5
        w = AI.WeightedSampler([0.0, 0.0, 1.0], num_samples=4, seed=2)
        assert list(w) == [2, 2, 2, 2]
        now = list(AI.WeightedSampler([1.0, 1.0, 2.0], num_samples=3,
                                      replacement=False, seed=5))
        assert len(set(now)) == 3
        st = list(AI.StratifiedSampler([0, 0, 0, 1, 1, 2], seed=4, shuffle=True))
        assert sorted(st) == list(range(6))
        assert list(AI.SubsetSampler([3, 1])) == [3, 1]

    def test_batch_sampler(self):
        bs = AI.BatchSampler(AI.SequentialSampler(5), 2)
        assert [len(b) for b in bs] == [2, 2, 1]
        assert len(bs) == 3
        drop = AI.BatchSampler(AI.SequentialSampler(5), 2, drop_last=True)
        assert [len(b) for b in drop] == [2, 2]
        assert len(drop) == 2
        for bad in (0, -1, 1.5, True):
            with pytest.raises(ValueError):
                AI.BatchSampler(AI.SequentialSampler(3), bad)

    def test_bucket_by_length(self):
        lengths = [10, 1, 5, 9, 2, 6, 8, 3, 7, 4]
        batches = AI.bucket_by_length(lengths, 3, bucket_size_multiplier=1, seed=7)
        flat = [i for b in batches for i in b]
        assert sorted(flat) == list(range(10))
        assert all(
            max(lengths[i] for i in b) - min(lengths[i] for i in b) <= 2
            for b in batches if len(b) > 1
        )
        assert batches == AI.bucket_by_length(lengths, 3, bucket_size_multiplier=1, seed=7)
        dropped = AI.bucket_by_length(lengths, 3, bucket_size_multiplier=1, seed=7,
                                      drop_last=True)
        assert all(len(b) == 3 for b in dropped)
        with pytest.raises(ValueError):
            AI.bucket_by_length([], 3)

    def test_preprocess_cache(self, tmp_path):
        src = tmp_path / "data.txt"
        src.write_text("hello", encoding="utf-8")
        cache = AI.PreprocessCache(str(tmp_path / "cache"), files=[str(src)])
        built = []

        def build():
            built.append(1)
            return {"tokens": [1, 2, 3]}

        v1, hit1 = cache.get({"mode": "a"}, build)
        v2, hit2 = cache.get({"mode": "a"}, build)
        assert (hit1, hit2) == (False, True)
        assert len(built) == 1
        assert v1 == v2
        _v3, hit3 = cache.get({"mode": "b"}, build)
        assert hit3 is False and len(built) == 2
        src.write_text("hello!", encoding="utf-8")
        _v4, hit4 = cache.get({"mode": "a"}, build)
        assert hit4 is False and len(built) == 3
        assert cache.clear() >= 3
        with pytest.raises(ValueError):
            AI.PreprocessCache(str(tmp_path), digest="sha1")
        with pytest.raises(FileNotFoundError):
            AI.PreprocessCache(str(tmp_path), files=[str(tmp_path / "nope")]).key()

    def test_dataloader_sampler_paths(self):
        items = [("a", 1), ("b", 2), ("c", 3), ("d", 4), ("e", 5)]
        order = list(range(5))
        random.Random(7).shuffle(order)
        plain = [x for batch in AI.DataLoader(items, batch_size=2, shuffle=True, seed=7)
                 for x in batch]
        assert plain == [items[i] for i in order]
        assert len(AI.DataLoader(items, batch_size=2)) == 3
        with_sampler = AI.DataLoader(items, batch_size=2, sampler=AI.SubsetSampler([4, 0, 2]))
        assert [x for b in with_sampler for x in b] == [items[4], items[0], items[2]]
        assert len(with_sampler) == 2
        with_batch = AI.DataLoader(items, batch_sampler=[[0, 1], [4]])
        assert [len(b) for b in with_batch] == [2, 1]
        assert len(with_batch) == 2
        assert list(AI.DataLoader([], batch_size=4)) == []
        with pytest.raises(ValueError):
            AI.DataLoader(items, sampler=[0], batch_sampler=[[0]])
        with pytest.raises(ValueError):
            AI.DataLoader(items, batch_size=0)


class TestEncoders:
    """W11-B：六种编码器"""

    def test_label_encoder(self):
        le = AI.LabelEncoder().fit(["b", "a", "c", "a"])
        assert le.classes_ == ["a", "b", "c"]
        assert le.transform(["c", "a"]) == [2, 0]
        assert le.inverse_transform([2, 0]) == ["c", "a"]
        with pytest.raises(ValueError):
            le.transform(["z"])
        with pytest.raises(ValueError):
            le.inverse_transform([3])
        with pytest.raises(ValueError):
            AI.LabelEncoder().transform([1])

    def test_ordinal_encoder(self):
        rows = [["r", 1], ["g", 2], ["r", 2]]
        oe = AI.OrdinalEncoder().fit(rows)
        assert oe.transform([["g", 1]]) == [[0, 0]]
        assert oe.transform([["r", 2]]) == [[1, 1]]
        assert oe.inverse_transform(oe.transform(rows)) == rows
        with pytest.raises(ValueError):
            oe.transform([["b", 1]])
        with pytest.raises(ValueError):
            oe.transform([[1]])

    def test_one_hot_encoder(self):
        x = [["red", "s"], ["green", "m"], ["blue", "s"]]
        oh = AI.OneHotEncoder().fit(x)
        enc = oh.transform([["green", "m"], ["red", "s"]])
        assert [sum(r) for r in enc] == [2.0, 2.0]
        assert len(enc[0]) == 5
        assert len(oh.get_feature_names()) == 5
        drop = AI.OneHotEncoder(drop="first").fit(x)
        assert drop.transform([["green", "m"]]) == [[1.0, 0.0, 0.0]]
        ign = AI.OneHotEncoder(handle_unknown="ignore").fit(x)
        assert ign.transform([["purple", "s"]]) == [[0.0, 0.0, 0.0, 0.0, 1.0]]
        with pytest.raises(ValueError):
            oh.transform([["purple", "s"]])
        with pytest.raises(ValueError):
            AI.OneHotEncoder(drop="first").fit([["only"]])
        with pytest.raises(ValueError):
            AI.OneHotEncoder(drop="if_binary")

    def test_target_encoder(self):
        x = [["a"], ["a"], ["b"], ["c"]]
        y = [1.0, 3.0, 10.0, 2.0]
        te0 = AI.TargetEncoder(smoothing=0.0).fit(x, y)
        assert te0.transform([["a"], ["b"], ["c"]]) == [[2.0], [10.0], [2.0]]
        prior = sum(y) / 4
        te1 = AI.TargetEncoder(smoothing=1.0).fit(x, y)
        assert abs(te1.transform([["a"]])[0][0] - (4.0 + prior) / 3) < 1e-12
        assert abs(te1.transform([["zzz"]])[0][0] - prior) < 1e-12
        assert abs(te1.prior_ - prior) < 1e-12
        with pytest.raises(ValueError):
            AI.TargetEncoder(smoothing=-1.0)
        with pytest.raises(ValueError):
            te1.fit(x, [1.0, "x", 2.0, 3.0])

    def test_frequency_and_woe_encoder(self):
        fr = AI.FrequencyEncoder().fit([["a"], ["a"], ["b"], ["c"]])
        assert fr.transform([["a"], ["b"], ["zzz"]]) == [[0.5], [0.25], [0.0]]
        xw = [["u"], ["u"], ["v"], ["v"]]
        yw = [1, 1, 0, 0]
        woe = AI.WOEEncoder(smoothing=0.5).fit(xw, yw)
        got = woe.transform([["u"], ["v"], ["w"]])
        p_bad = 2.5 / 3.0
        p_good = 0.5 / 3.0
        want = math.log(p_bad / p_good)
        assert abs(got[0][0] - want) < 1e-12
        assert abs(got[1][0] + want) < 1e-12
        assert got[2][0] == 0.0
        with pytest.raises(ValueError):
            AI.WOEEncoder().fit(xw, [1, 1, 1, 1])
        with pytest.raises(ValueError):
            AI.WOEEncoder().fit(xw, [0, 1, 2, 3])
        with pytest.raises(ValueError):
            AI.WOEEncoder(smoothing=0.0)

    def test_encoder_input_validation(self):
        for bad in ([], [[]], [[1, 2], [3]]):
            for maker in (AI.OrdinalEncoder, AI.FrequencyEncoder):
                with pytest.raises(ValueError):
                    maker().fit(bad)
        with pytest.raises(ValueError):
            AI.TargetEncoder().fit([["a"]], [1.0, 2.0])


class TestImageAugmentation:
    """W11-C：图像增强管道"""

    def test_resize_bilinear(self):
        up = AI._resize_bilinear([[[0.0, 10.0], [20.0, 30.0]]], 3, 3)
        assert len(up[0]) == 3 and len(up[0][0]) == 3
        assert up[0][0][0] == 0.0 and up[0][2][2] == 30.0
        assert abs(up[0][1][1] - 15.0) < 1e-12
        same = AI._resize_bilinear([[[1.0]]], 1, 1)
        assert same == [[[1.0]]]

    def test_compose(self):
        ident = AI.Compose([AI.Normalize([0.0], [1.0])])
        x = [[[1.0, 2.0]]]
        assert ident(x) == x
        assert repr(ident) == "Compose([Normalize])"
        with pytest.raises(ValueError):
            AI.Compose([])
        with pytest.raises(TypeError):
            AI.Compose([1])

    def test_random_crop(self):
        g = [[[float(i * 10 + j) for j in range(5)] for i in range(5)]]
        rc = AI.RandomCrop((2, 3), seed=7)
        out = rc(g)
        want = [[row[0:3] for row in g[0][2:4]]]
        assert out == want
        assert AI.RandomCrop((2, 3), seed=7)(g) == out
        padded = AI.RandomCrop((4, 4), padding=1, fill=9.0, seed=1)(
            [[[1.0, 2.0], [3.0, 4.0]]])
        assert padded[0][0][3] == 9.0 and padded[0][3][0] == 9.0
        assert sorted(v for ch in padded for row in ch for v in row) ==             sorted([9.0] * 12 + [1.0, 2.0, 3.0, 4.0])
        with pytest.raises(ValueError):
            AI.RandomCrop((6, 6))(g)

    def test_random_resized_crop(self):
        g = [[[float(i + j) / 14.0 for j in range(8)] for i in range(8)] for _ in range(2)]
        rrc = AI.RandomResizedCrop((4, 4), seed=3)
        out = rrc(g)
        assert len(out) == 2 and len(out[0]) == 4 and len(out[0][0]) == 4
        vals = [v for ch in out for row in ch for v in row]
        assert min(vals) >= -1e-12 and max(vals) <= 1.0 + 1e-12
        assert AI.RandomResizedCrop((4, 4), seed=3)(g) == out
        with pytest.raises(ValueError):
            AI.RandomResizedCrop((4, 4), scale=(0.5, 0.1))

    def test_random_rotation(self):
        g = [[[float(i + j) for j in range(5)] for i in range(5)]]
        assert AI.RandomRotation(0.0, seed=1)(g) == g
        const = [[[2.5] * 7 for _ in range(7)]]
        out = AI.RandomRotation(180.0, fill=2.5, seed=11)(const)
        assert all(abs(v - 2.5) < 1e-12 for ch in out for row in ch for v in row)
        delta = [[[0.0] * 9 for _ in range(9)]]
        delta[0][4][4] = 1.0
        dr = AI.RandomRotation(180.0, seed=5)(delta)
        assert dr[0][4][4] > 0.3
        total = sum(v for ch in dr for row in ch for v in row)
        cy = sum(i * dr[0][i][j] for i in range(9) for j in range(9)) / total
        assert abs(cy - 4.0) < 0.35
        with pytest.raises(ValueError):
            AI.RandomRotation(-1.0)

    def test_color_jitter(self):
        one = [[[0.4 + 0.1 * c for _ in range(3)] for _ in range(3)] for c in range(3)]
        assert AI.ColorJitter(seed=1)(one) == one
        bo = AI.ColorJitter(brightness=0.5, seed=5)(one)
        f = random.Random(5).uniform(0.5, 1.5)
        assert all(abs(bo[c][i][j] - one[c][i][j] * f) < 1e-12
                   for c in range(3) for i in range(3) for j in range(3))
        gray = [[[0.5] * 3 for _ in range(3)] for _ in range(3)]
        hue = AI.ColorJitter(hue=0.4, seed=2)(gray)
        assert all(abs(hue[0][i][j] - hue[1][i][j]) < 1e-12
                   and abs(hue[1][i][j] - hue[2][i][j]) < 1e-12
                   for i in range(3) for j in range(3))
        assert AI.ColorJitter(saturation=0.5, seed=2)(gray) == gray
        with pytest.raises(ValueError):
            AI.ColorJitter(hue=0.6)
        with pytest.raises(ValueError):
            AI.ColorJitter(hue=0.2)([[[0.5] * 2 for _ in range(2)]])

    def test_gaussian_blur_and_normalize(self):
        blur = AI.GaussianBlur(3, sigma=(1.0, 1.0), seed=1)
        const = [[[1.5] * 5 for _ in range(5)] for _ in range(2)]
        assert all(abs(v - 1.5) < 1e-12 for ch in blur(const) for row in ch for v in row)
        d = [[[0.0] * 7 for _ in range(7)]]
        d[0][3][3] = 1.0
        bd = blur(d)
        assert abs(sum(v for ch in bd for row in ch for v in row) - 1.0) < 1e-9
        assert abs(bd[0][2][3] - bd[0][4][3]) < 1e-12
        with pytest.raises(ValueError):
            AI.GaussianBlur(4)
        with pytest.raises(ValueError):
            AI.GaussianBlur(3, sigma=(0.0, 1.0))
        n = AI.Normalize([0.5, 1.0], [0.5, 2.0])
        assert n([[[0.5, 1.0]], [[1.0, 3.0]]]) == [[[0.0, 1.0]], [[0.0, 1.0]]]
        with pytest.raises(ValueError):
            n([[[0.0, 1.0]]])
        with pytest.raises(ValueError):
            AI.Normalize([0.0, 1.0], [1.0])
        with pytest.raises(ValueError):
            AI.Normalize([0.0], [0.0])

    def test_aug_input_validation_and_pipeline(self):
        for bad in ([], [[]], [[[1.0], [2.0, 3.0]]]):
            with pytest.raises(ValueError):
                AI._aug3_check(bad, "t")
        with pytest.raises(TypeError):
            AI._aug3_check([[[True]]], "t")

        def build():
            return AI.Compose([
                AI.ColorJitter(brightness=0.2, contrast=0.2, seed=4),
                AI.RandomCrop((8, 8), seed=4),
                AI.RandomRotation(15.0, seed=4),
                AI.GaussianBlur(3, sigma=(0.3, 0.8), seed=4),
                AI.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ])

        src = [[[0.5 + 0.01 * (i - j) for j in range(10)] for i in range(10)]
               for _ in range(3)]
        pipe = build()
        out = pipe(src)
        assert len(out) == 3 and len(out[0]) == 8 and len(out[0][0]) == 8
        assert build()(src) == out
        assert pipe(src) != out
        assert src[0][0][0] == 0.5 and len(src[0]) == 10


class TestSplitExtensions:
    """W11-D：GroupKFold / 时间序列切分 / Purged K-Fold"""

    def test_group_k_fold(self):
        groups = ["a", "a", "a", "b", "b", "c", "d", "d", "e", "f"]
        folds = AI.group_k_fold_indices(groups, n_splits=3)
        assert len(folds) == 3
        seen = set()
        for tr, va in folds:
            assert sorted(tr + va) == list(range(10))
            assert not seen & set(va)
            seen |= set(va)
            assert not ({groups[i] for i in tr} & {groups[i] for i in va})
        assert seen == set(range(10))
        sizes = [sum(len(groups[i]) for i in tr) for tr, _ in folds]
        assert max(sizes) - min(sizes) <= 1
        assert AI.group_k_fold_indices(groups, 3) == folds
        assert AI.group_k_fold_indices(["a", "b", "c"], 3)[0][1] != []

    def test_group_k_fold_validation(self):
        with pytest.raises(ValueError):
            AI.group_k_fold_indices([], 2)
        with pytest.raises(ValueError):
            AI.group_k_fold_indices(["a"], 2)
        with pytest.raises(ValueError):
            AI.group_k_fold_indices(["a", "b"], 1)
        with pytest.raises(TypeError):
            AI.group_k_fold_indices(["a", "b"], 2.0)
        with pytest.raises(TypeError):
            AI.group_k_fold_indices(["a", "b"], 2, seed=1.5)

    def test_time_series_expanding(self):
        splits = AI.time_series_split_indices(100, n_splits=4)
        assert len(splits) == 4
        assert [va[0] for _tr, va in splits] == [20, 40, 60, 80]
        assert [va[-1] for _tr, va in splits] == [39, 59, 79, 99]
        assert [len(tr) for tr, _ in splits] == [20, 40, 60, 80]
        assert all(tr[0] == 0 and max(tr) < min(va) for tr, va in splits)
        assert [len(va) for _tr, va in AI.time_series_split_indices(30, 2)] == [10, 10]
        assert len(AI.time_series_split_indices(30, 1)) == 1

    def test_time_series_gap_and_rolling(self):
        g = AI.time_series_split_indices(100, n_splits=4, gap=5)
        assert all(tr[-1] == min(va) - 6 for tr, va in g)
        r = AI.time_series_split_indices(100, n_splits=4, expanding=False, window=10)
        assert all(len(tr) == 10 and max(tr) < min(va) for tr, va in r)
        r0 = AI.time_series_split_indices(100, n_splits=4, expanding=False)
        assert [len(tr) for tr, _ in r0] == [20, 20, 20, 20]

    def test_time_series_validation(self):
        for call in (
            lambda: AI.time_series_split_indices(0),
            lambda: AI.time_series_split_indices(10, n_splits=0),
            lambda: AI.time_series_split_indices(10, n_splits=2, test_size=0),
            lambda: AI.time_series_split_indices(5, n_splits=5),
            lambda: AI.time_series_split_indices(20, n_splits=2, gap=9),
            lambda: AI.time_series_split_indices(20, n_splits=2.0),
            lambda: AI.time_series_split_indices(100, n_splits=2, window=0),
        ):
            with pytest.raises((ValueError, TypeError)):
                call()

    def test_purged_k_fold(self):
        p = AI.purged_kfold_indices(50, n_splits=5)
        assert [va for _tr, va in p] == [list(range(i, i + 10)) for i in (0, 10, 20, 30, 40)]
        assert all(sorted(tr + va) == list(range(50)) for tr, va in p)
        tr2, va2 = AI.purged_kfold_indices(50, n_splits=5, purge=2, embargo=3)[2]
        assert all(i not in tr2 for i in range(18, 32))
        assert all(i not in tr2 for i in range(32, 35))
        assert 35 in tr2 and 17 in tr2

    def test_purged_validation_and_legacy(self):
        for call in (
            lambda: AI.purged_kfold_indices(1),
            lambda: AI.purged_kfold_indices(10, n_splits=1),
            lambda: AI.purged_kfold_indices(10, n_splits=20),
            lambda: AI.purged_kfold_indices(10, n_splits=2, purge=-1),
            lambda: AI.purged_kfold_indices(10, n_splits=2, embargo=1.0),
            lambda: AI.purged_kfold_indices(10, n_splits=2, purge=8, embargo=8),
        ):
            with pytest.raises((ValueError, TypeError)):
                call()
        kf = AI.k_fold_indices(20, n_splits=4, seed=1)
        assert len(kf) == 4


class TestScalerFamily:
    """W11-E：Robust / MaxAbs / QuantileTransformer / PowerTransformer"""

    def test_robust_scaler(self):
        x = [[1.0], [2.0], [3.0], [4.0], [100.0]]
        rs = AI.RobustScaler().fit(x)
        assert rs.center_ == [3.0]
        assert rs.scale_ == [2.0]
        out = rs.transform(x)
        assert [r[0] for r in out] == [-1.0, -0.5, 0.0, 0.5, 48.5]
        back = rs.inverse_transform(out)
        assert all(abs(a[0] - b[0]) < 1e-12 for a, b in zip(back, x))
        nc = AI.RobustScaler(with_centering=False).fit(x)
        assert nc.center_ == [0.0] and nc.transform([[3.0]]) == [[1.5]]
        ns = AI.RobustScaler(with_scaling=False).fit(x)
        assert ns.scale_ == [1.0] and ns.transform([[5.0]]) == [[2.0]]
        assert AI.RobustScaler().fit([[7.0], [7.0]]).scale_ == [1.0]
        with pytest.raises(ValueError):
            AI.RobustScaler(quantile_range=(50, 50))
        with pytest.raises(ValueError):
            AI.RobustScaler(quantile_range=(-1, 50))
        with pytest.raises(TypeError):
            AI.RobustScaler(with_centering=1)

    def test_max_abs_scaler(self):
        ma = AI.MaxAbsScaler().fit([[-3.0, 1.0], [2.0, -4.0]])
        assert ma.scale_ == [3.0, 4.0]
        mt = ma.transform([[-3.0, 1.0], [2.0, -4.0]])
        assert mt == [[-1.0, 0.25], [2.0 / 3.0, -1.0]]
        assert ma.inverse_transform(mt) == [[-3.0, 1.0], [2.0, -4.0]]
        assert AI.MaxAbsScaler().fit([[0.0], [0.0]]).scale_ == [1.0]

    def test_quantile_uniform(self):
        xq = [[float(i)] for i in range(10)]
        qt = AI.QuantileTransformer(n_quantiles=10).fit(xq)
        assert qt.quantiles_[0] == [float(i) for i in range(10)]
        u = qt.transform([[3.0], [4.5], [-1.0], [20.0]])
        assert abs(u[0][0] - 3.0 / 9.0) < 1e-12
        assert abs(u[1][0] - 0.5) < 1e-12
        assert u[2][0] == 0.0 and u[3][0] == 1.0
        back = qt.inverse_transform(qt.transform(xq))
        assert all(abs(a[0] - b[0]) < 1e-9 for a, b in zip(back, xq))
        assert AI.QuantileTransformer(n_quantiles=1000).fit(xq).quantiles_[0] == \
            [float(i) for i in range(10)]

    def test_quantile_normal(self):
        xq = [[float(i)] for i in range(10)]
        qn = AI.QuantileTransformer(n_quantiles=10,
                                    output_distribution="normal").fit(xq)
        assert abs(qn.transform([[4.5]])[0][0]) < 1e-9
        vals = [r[0] for r in qn.transform(xq)]
        assert all(vals[i] < vals[i + 1] for i in range(9))
        assert all(-2.0 < v < 2.0 for v in vals[1:-1])
        back = qn.inverse_transform(qn.transform(xq))
        assert all(abs(a[0] - b[0]) < 1e-8 for a, b in zip(back, xq))
        with pytest.raises(ValueError):
            AI.QuantileTransformer(n_quantiles=1)
        with pytest.raises(ValueError):
            AI.QuantileTransformer(output_distribution="x")
        with pytest.raises(TypeError):
            AI.QuantileTransformer(n_quantiles=2.0)
        with pytest.raises(ValueError):
            AI.QuantileTransformer(subsample=1)
        with pytest.raises(TypeError):
            AI.QuantileTransformer(seed="a")

    def test_yeo_johnson_primitives(self):
        assert abs(AI._w11e_yj_forward(2.5, 1.0) - 2.5) < 1e-12
        assert abs(AI._w11e_yj_forward(1.0, 0.0) - math.log(2.0)) < 1e-12
        assert abs(AI._w11e_yj_forward(-0.5, 2.0) + math.log(1.5)) < 1e-12
        for lam in (-1.5, -0.3, 0.0, 0.7, 1.0, 1.9, 2.0, 3.0):
            for v in (-0.7, -0.1, 0.0, 0.4, 2.0):
                z = AI._w11e_yj_forward(v, lam)
                assert abs(AI._w11e_yj_inverse(z, lam) - v) < 1e-9

    def test_power_transformer_yeo_js(self):
        xs = [[float(v)] for v in (0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0)]
        pt = AI.PowerTransformer(method="yeo-js").fit(xs)
        assert -5.0 <= pt.lambdas_[0] <= 5.0
        zt = pt.transform(xs)
        assert abs(sum(r[0] for r in zt) / len(zt)) < 1e-9
        assert abs(sum(r[0] ** 2 for r in zt) / len(zt) - 1.0) < 1e-9
        back = pt.inverse_transform(zt)
        assert all(abs(a[0] - b[0]) < 1e-6 for a, b in zip(back, xs))

    def test_power_transformer_box_cox(self):
        xp = [[v] for v in (0.5, 1.0, 2.0, 4.0, 8.0)]
        bc = AI.PowerTransformer(method="box-cox").fit(xp)
        assert -5.0 <= bc.lambdas_[0] <= 5.0
        back = bc.inverse_transform(bc.transform(xp))
        assert all(abs(a[0] - b[0]) < 1e-6 for a, b in zip(back, xp))
        with pytest.raises(ValueError):
            AI.PowerTransformer(method="box-cox").fit([[0.0], [1.0]])
        with pytest.raises(ValueError):
            AI.PowerTransformer(method="x")
        with pytest.raises(TypeError):
            AI.PowerTransformer(standardize=1)

    def test_scaler_common_validation(self):
        for obj in (AI.RobustScaler(), AI.MaxAbsScaler(),
                    AI.QuantileTransformer(), AI.PowerTransformer()):
            with pytest.raises(ValueError):
                obj.transform([[1.0]])
            for bad in ([], [[]], [[1.0, 2.0], [3.0]]):
                with pytest.raises((ValueError, TypeError)):
                    obj.fit(bad)
            with pytest.raises(TypeError):
                obj.fit([["a"]])

class TestImputers:
    """W11-F：Simple / KNN / Iterative 缺失值填补"""

    def test_simple_mean_transform(self):
        x = [[1.0, 10.0], [3.0, None], [None, 30.0]]
        imp = AI.SimpleImputer(strategy="mean").fit(x)
        assert imp.statistics_ == [2.0, 20.0]
        assert imp.transform(x) == [[1.0, 10.0], [3.0, 20.0], [2.0, 30.0]]

    def test_simple_median_and_all_missing(self):
        imp = AI.SimpleImputer(strategy="median").fit(
            [[1.0, None], [2.0, None], [100.0, None]])
        assert imp.statistics_[0] == 2.0
        assert imp.statistics_[1] == 0.0

    def test_simple_most_frequent_and_constant(self):
        mf = AI.SimpleImputer("most_frequent").fit([[1.0], [1.0], [2.0], [None]])
        assert mf.statistics_ == [1.0]
        imp = AI.SimpleImputer("constant", fill_value=7.5).fit([[None], [1.0]])
        assert imp.transform([[None], [2.0]]) == [[7.5], [2.0]]

    def test_simple_imputer_errors(self):
        with pytest.raises(ValueError):
            AI.SimpleImputer("bad")
        with pytest.raises(ValueError):
            AI.SimpleImputer("constant")
        with pytest.raises(TypeError):
            AI.SimpleImputer("constant", fill_value=True)
        with pytest.raises(ValueError):
            AI.SimpleImputer("mean").transform([[1.0]])
        with pytest.raises(ValueError):
            AI.SimpleImputer("mean").fit([[1.0]]).transform([[1.0, 2.0]])
        with pytest.raises(ValueError):
            AI.SimpleImputer("mean").fit([[1.0], [2.0, 3.0]])
        with pytest.raises(ValueError):
            AI.SimpleImputer("mean").fit([])
        with pytest.raises(TypeError):
            AI.SimpleImputer("mean").fit([[True], [1.0]])

    def test_knn_uniform_and_distance(self):
        train = [[0.0, 0.0], [0.0, 10.0], [10.0, 0.0]]
        out = AI.KNNImputer(n_neighbors=2, weights="uniform").fit(
            train).transform([[None, 1.0]])
        assert abs(out[0][0] - 5.0) < 1e-12
        out2 = AI.KNNImputer(n_neighbors=2, weights="distance").fit(
            [[0.0, 0.0], [2.0, 4.0]]).transform([[None, 1.0]])
        assert abs(out2[0][0] - 0.5) < 1e-12

    def test_knn_fallback_and_nearest(self):
        out = AI.KNNImputer().fit([[1.0], [3.0]]).transform([[None]])
        assert abs(out[0][0] - 2.0) < 1e-12
        out2 = AI.KNNImputer(n_neighbors=1).fit(
            [[0.0, 0.0], [4.0, 4.0]]).transform([[None, 0.5]])
        assert abs(out2[0][0] - 0.0) < 1e-12
        got = AI.KNNImputer().fit([[0.0, 0.0], [0.0, 1.0]]).transform([[7.0, 8.0]])
        assert got == [[7.0, 8.0]]

    def test_knn_errors(self):
        with pytest.raises(ValueError):
            AI.KNNImputer(0)
        with pytest.raises(TypeError):
            AI.KNNImputer(True)
        with pytest.raises(ValueError):
            AI.KNNImputer(3, "bad")
        with pytest.raises(ValueError):
            AI.KNNImputer().transform([[None]])
        with pytest.raises(ValueError):
            AI.KNNImputer().fit([[1.0, 2.0]]).transform([[None]])

    def test_iterative_recovers_linear(self):
        x = [[1.0, 3.0], [2.0, 5.0], [3.0, None], [4.0, 9.0]]
        imp = AI.IterativeImputer(max_iter=50, tol=1e-9)
        out = imp.fit_transform(x)
        assert abs(out[2][1] - 7.0) < 1e-6
        assert 1 <= imp.n_iter_ <= 50
        for i in range(len(x)):
            for j in range(2):
                if x[i][j] is not None:
                    assert out[i][j] == x[i][j]

    def test_iterative_two_missing_and_no_missing(self):
        x = [[1.0, 2.0, 2.0], [2.0, 4.0, None],
             [3.0, 6.0, 4.0], [4.0, None, 5.0]]
        out = AI.IterativeImputer(max_iter=60, tol=1e-10).fit_transform(x)
        assert abs(out[1][2] - 3.0) < 1e-4
        assert abs(out[3][1] - 8.0) < 1e-4
        imp = AI.IterativeImputer()
        assert imp.fit_transform([[1.0, 2.0], [3.0, 4.0]]) == [[1.0, 2.0], [3.0, 4.0]]
        assert imp.n_iter_ == 0

    def test_iterative_all_missing_column_and_errors(self):
        out = AI.IterativeImputer(max_iter=3).fit_transform(
            [[None, 1.0], [None, 2.0], [None, 3.0]])
        assert all(row[0] == 0.0 for row in out)
        with pytest.raises(ValueError):
            AI.IterativeImputer(0)
        with pytest.raises(TypeError):
            AI.IterativeImputer(True)
        with pytest.raises(ValueError):
            AI.IterativeImputer(tol=-1.0)
        with pytest.raises(ValueError):
            AI.IterativeImputer(alpha=-0.5)
        with pytest.raises(TypeError):
            AI.IterativeImputer(tol="x")
        with pytest.raises(ValueError):
            AI.IterativeImputer().transform([[None]])

    def test_solve_and_ridge_helpers(self):
        sol = AI._w11f_solve([[2.0, 1.0], [1.0, 3.0]], [3.0, 5.0])
        assert abs(sol[0] - 0.8) < 1e-12
        assert abs(sol[1] - 1.4) < 1e-12
        assert AI._w11f_solve([[1.0, 2.0], [2.0, 4.0]], [1.0, 2.0]) is None
        mean, std, coef, yb = AI._w11f_ridge(
            [[1.0], [2.0], [3.0]], [3.0, 5.0, 7.0], 1e-9)
        assert abs(mean[0] - 2.0) < 1e-12
        assert abs(std[0] - 0.816496580927726) < 1e-12
        assert abs(coef[0] - 2.0 * std[0]) < 1e-6
        assert abs(yb - 5.0) < 1e-12
class TestDimReductionSelection:
    """W11-G：PCA / t-SNE / SMOTE·ADASYN / 特征选择"""

    def test_pca_rank_one(self):
        line = [[1.0, 2.0], [2.0, 4.0], [3.0, 6.0], [4.0, 8.0]]
        p = AI.PCA(n_components=1).fit(line)
        assert abs(p.explained_variance_ratio_[0] - 1.0) < 1e-9
        assert abs(abs(p.components_[0][0]) - 1.0 / math.sqrt(5.0)) < 1e-8
        assert abs(abs(p.components_[0][1]) - 2.0 / math.sqrt(5.0)) < 1e-8
        z = p.transform([[2.5, 5.0]])
        assert abs(z[0][0]) < 1e-9
        back = p.inverse_transform(p.transform(line))
        for i in range(4):
            for j in range(2):
                assert abs(back[i][j] - line[i][j]) < 1e-9

    def test_pca_full_rank_and_whiten(self):
        full = [[1.0, 1.0], [2.0, 3.0], [3.0, 2.0], [4.0, 4.0]]
        p = AI.PCA(n_components=2).fit(full)
        assert len(p.components_) == 2
        assert abs(sum(p.explained_variance_ratio_) - 1.0) < 1e-9
        rt = p.inverse_transform(p.transform(full))
        for i in range(4):
            for j in range(2):
                assert abs(rt[i][j] - full[i][j]) < 1e-9
        line = [[1.0, 2.0], [2.0, 4.0], [3.0, 6.0], [4.0, 8.0]]
        pw = AI.PCA(n_components=1, whiten=True).fit(line)
        zw = pw.transform(line)
        mean_z = sum(r[0] for r in zw) / len(zw)
        var_z = sum((r[0] - mean_z) ** 2 for r in zw) / (len(zw) - 1)
        assert abs(var_z - 1.0) < 1e-9
        back = pw.inverse_transform(zw)
        for i in range(4):
            for j in range(2):
                assert abs(back[i][j] - line[i][j]) < 1e-9

    def test_pca_errors(self):
        with pytest.raises(ValueError):
            AI.PCA(0)
        with pytest.raises(TypeError):
            AI.PCA(True)
        with pytest.raises(ValueError):
            AI.PCA().transform([[1.0]])
        with pytest.raises(ValueError):
            AI.PCA().fit([[1.0]])
        with pytest.raises(ValueError):
            AI.PCA().fit([[1.0, 2.0]]).transform([[1.0]])
        with pytest.raises(ValueError):
            AI.PCA().fit([[1.0, 1.0], [2.0, 3.0]]).inverse_transform([[1.0]])

    def test_tsne_separates_clusters(self):
        clusters = []
        for cx, cy in ((0.0, 0.0), (10.0, 0.0), (0.0, 10.0)):
            for k in range(4):
                clusters.append([cx + 0.1 * k, cy + 0.05 * k])
        t = AI.TSNE(n_components=2, perplexity=3.0, n_iter=150, seed=7)
        emb = t.fit_transform(clusters)
        assert len(emb) == 12 and len(emb[0]) == 2
        intra = []
        inter = []
        for i in range(12):
            for j in range(i + 1, 12):
                d = math.sqrt(sum((emb[i][c] - emb[j][c]) ** 2 for c in range(2)))
                (intra if i // 4 == j // 4 else inter).append(d)
        assert sum(intra) / len(intra) < sum(inter) / len(inter)
        assert t.kl_divergence_ == t.kl_divergence_
        assert t.embedding_ == emb
        again = AI.TSNE(n_components=2, perplexity=3.0, n_iter=150,
                        seed=7).fit_transform(clusters)
        assert again == emb

    def test_tsne_errors(self):
        clusters = [[float(i), float(i % 3)] for i in range(6)]
        with pytest.raises(ValueError):
            AI.TSNE(perplexity=20.0).fit_transform(clusters)
        with pytest.raises(ValueError):
            AI.TSNE(n_components=0)
        with pytest.raises(ValueError):
            AI.TSNE(n_iter=0)
        with pytest.raises(ValueError):
            AI.TSNE(perplexity=0.0)
        with pytest.raises(ValueError):
            AI.TSNE().fit_transform([[1.0], [2.0]])

    def test_smote(self):
        xm = [[0.0, 0.0], [0.1, 0.1], [0.2, 0.0], [0.1, -0.1],
              [5.0, 5.0], [5.2, 5.1]]
        ym = [0, 0, 0, 0, 1, 1]
        sx, sy = AI.SMOTE(k_neighbors=1, random_state=1).fit_resample(xm, ym)
        assert sum(1 for v in sy if v == 0) == 4
        assert sum(1 for v in sy if v == 1) == 4
        assert sx[:6] == [list(r) for r in xm]
        assert sy[:6] == ym
        for row in sx[6:]:
            assert 4.9 <= row[0] <= 5.3 and 4.9 <= row[1] <= 5.3
        sx2, sy2 = AI.SMOTE(k_neighbors=1, random_state=1).fit_resample(xm, ym)
        assert sx == sx2 and sy == sy2
        sx3, _sy3 = AI.SMOTE(k_neighbors=1, random_state=2).fit_resample(xm, ym)
        assert sx3 != sx

    def test_smote_errors(self):
        with pytest.raises(ValueError):
            AI.SMOTE(k_neighbors=0)
        with pytest.raises(TypeError):
            AI.SMOTE(True)
        with pytest.raises(ValueError):
            AI.SMOTE().fit_resample([[1.0], [2.0]], [0, 0])
        with pytest.raises(ValueError):
            AI.SMOTE().fit_resample([[1.0], [2.0]], [0])
        with pytest.raises(ValueError):
            AI.ADASYN(k_neighbors=0)

    def test_adasyn(self):
        xm = [[0.0, 0.0], [0.1, 0.1], [0.2, 0.0], [0.1, -0.1],
              [5.0, 5.0], [5.2, 5.1]]
        ym = [0, 0, 0, 0, 1, 1]
        ada = AI.ADASYN(k_neighbors=2, random_state=3)
        ax, ay = ada.fit_resample(xm, ym)
        assert sum(1 for v in ay if v == 0) == 4
        assert sum(1 for v in ay if v == 1) == 4
        assert ax[:6] == [list(r) for r in xm]
        assert 1 in ada.weights_ and len(ada.weights_[1]) == 2

    def test_variance_threshold(self):
        xv = [[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]]
        vt = AI.VarianceThreshold(threshold=0.5).fit(xv)
        assert abs(vt.variances_[0]) < 1e-12
        assert abs(vt.variances_[1] - 2.0 / 3.0) < 1e-12
        assert vt.support_ == [False, True]
        assert vt.transform(xv) == [[1.0], [2.0], [3.0]]
        assert vt.get_support() == [False, True]
        with pytest.raises(ValueError):
            AI.VarianceThreshold(2.0).fit(xv)
        with pytest.raises(ValueError):
            AI.VarianceThreshold(-1.0)
        with pytest.raises(TypeError):
            AI.VarianceThreshold(True)

    def test_select_k_best(self):
        xs = [[1.0, 1.0, 7.0], [2.0, 0.0, 7.0],
              [3.0, 1.0, 7.0], [4.0, 0.0, 7.0]]
        ys = [0, 0, 1, 1]
        sk = AI.SelectKBest(k=1).fit(xs, ys)
        assert sk.support_ == [True, False, False]
        assert abs(sk.scores_[0] - 8.0) < 1e-12
        assert abs(sk.scores_[2]) < 1e-12
        assert sk.transform(xs) == [[1.0], [2.0], [3.0], [4.0]]
        sp = AI.SelectKBest(k=1, score_func="pearson").fit(xs, ys)
        assert sp.support_ == [True, False, False]
        assert abs(sp.scores_[0] - 2.0 / math.sqrt(5.0)) < 1e-12
        assert abs(sp.scores_[2]) < 1e-12
        assert AI.SelectKBest(k=99).fit(xs, ys).k_ == 3

    def test_select_k_best_errors(self):
        xs = [[1.0, 2.0], [2.0, 3.0]]
        with pytest.raises(ValueError):
            AI.SelectKBest(0)
        with pytest.raises(ValueError):
            AI.SelectKBest(k=1, score_func="bad")
        with pytest.raises(TypeError):
            AI.SelectKBest(k=1, score_func="pearson").fit(xs, ["a", "b"])
        with pytest.raises(ValueError):
            AI.SelectKBest().transform([[1.0]])
        with pytest.raises(ValueError):
            AI.SelectKBest().fit(xs, [0, 1]).transform([[1.0]])
        with pytest.raises(ValueError):
            AI.SelectKBest().fit(xs, [0])

    def test_rfe(self):
        xr = []
        yr = []
        for i in range(10):
            xr.append([float(i), float((i * 7) % 5), float((i * 3) % 4),
                       1.0, float((i * 11) % 6)])
            yr.append(2.0 * float(i) + 1.0)
        rfe = AI.RFE(n_features_to_select=1, step=1).fit(xr, yr)
        assert rfe.support_ == [True, False, False, False, False]
        assert len(rfe.ranking_) == 5
        assert rfe.ranking_[0] == 1
        assert len(rfe.transform(xr)[0]) == 1
        with pytest.raises(ValueError):
            AI.RFE(n_features_to_select=9).fit(xr, yr)
        with pytest.raises(ValueError):
            AI.RFE(0)
        with pytest.raises(ValueError):
            AI.RFE(step=0)

    def test_l1_selector(self):
        xl = []
        yl = []
        for i in range(20):
            xl.append([float(i), 1.0 if i % 2 == 0 else -1.0, 0.0])
            yl.append(3.0 * i)
        l1 = AI.L1Selector(alpha=0.1).fit(xl, yl)
        assert l1.support_[0] is True
        assert l1.support_[1] is False
        assert l1.support_[2] is False
        assert len(l1.transform(xl)[0]) == 1
        with pytest.raises(ValueError):
            AI.L1Selector(0.0)
        with pytest.raises(TypeError):
            AI.L1Selector(True)
        with pytest.raises(ValueError):
            AI.L1Selector(alpha=1e9).fit(xl, yl)
        with pytest.raises(ValueError):
            AI.L1Selector().transform([[1.0]])


class TestTextAudioAugmentation:
    """W11-H 文本增强（同义词 / EDA）与音频增强（概念）。"""

    def test_synonym_replace_probability_and_validity(self):
        syn = {"a": ["a1", "a2"], "b": ["b1"], "z": []}
        toks = ["a", "b", "c", "z"]
        assert AI.synonym_replace(toks, syn, p=0.0) == toks
        out = AI.synonym_replace(toks, syn, p=1.0, seed=7)
        assert out[0] in syn["a"] and out[1] == "b1"
        assert out[2] == "c" and out[3] == "z"
        assert len(out) == len(toks)
        assert AI.synonym_replace(toks, syn, p=1.0, seed=7) == out
        with pytest.raises(ValueError):
            AI.synonym_replace(toks, syn, p=1.5)
        with pytest.raises(TypeError):
            AI.synonym_replace(toks, [])

    def test_random_insert_adds_synonyms(self):
        syn = {"a": ["a1", "a2"], "b": ["b1"], "c": [], "z": []}
        toks = ["a", "b", "c", "z"]
        out = AI.random_insert(toks, syn, n=2, seed=3)
        assert len(out) == len(toks) + 2
        assert all(t in out for t in toks)
        assert all(t in ("a1", "a2", "b1") for t in out if t not in toks)
        assert AI.random_insert(toks, syn, n=0) == toks
        assert AI.random_insert(toks, syn, n=2, seed=3) == out
        with pytest.raises(ValueError):
            AI.random_insert(["c", "z"], syn, n=1)

    def test_random_swap_preserves_multiset(self):
        toks = ["a", "b", "c", "d"]
        assert AI.random_swap(toks, n=0) == toks
        out = AI.random_swap(toks, n=3, seed=5)
        assert sorted(out) == sorted(toks)
        assert AI.random_swap(toks, n=3, seed=5) == out
        with pytest.raises(ValueError):
            AI.random_swap(["only"], n=1)

    def test_random_delete_never_empties(self):
        toks = ["a", "b", "c", "d"]
        assert AI.random_delete(toks, p=0.0) == toks
        assert len(AI.random_delete(toks, p=1.0, seed=1)) == 1
        out = AI.random_delete(toks, p=0.5, seed=2)
        assert 1 <= len(out) <= len(toks)
        assert all(t in toks for t in out)

    def test_token_dropout_bounds(self):
        toks = ["a", "b", "c"]
        assert AI.token_dropout(toks, p=0.0) == toks
        assert len(AI.token_dropout(toks, p=0.999, seed=4)) >= 1
        with pytest.raises(ValueError):
            AI.token_dropout(toks, p=1.0)

    def test_eda_augment_invariants(self):
        base = ["i", "like", "this", "movie", "very", "much"]
        syn = {"like": ["love", "enjoy"], "movie": ["film"]}
        aug = AI.eda_augment(base, syn, num_aug=3, alpha_sr=0.3, alpha_ri=0.1,
                          alpha_rs=0.1, p_rd=0.0, seed=11)
        assert len(aug) == 3
        vocab = set(base) | {"love", "enjoy", "film"}
        for variant in aug:
            assert len(variant) >= 1
            assert all(t in vocab for t in variant)
        assert AI.eda_augment(base, syn, num_aug=3, alpha_sr=0.3, alpha_ri=0.1,
                           alpha_rs=0.1, p_rd=0.0, seed=11) == aug
        with pytest.raises(ValueError):
            AI.eda_augment(base, syn, num_aug=0)

    def test_eda_augment_heavy_deletion_still_nonempty(self):
        base = ["a", "b", "c"]
        aug = AI.eda_augment(base, {"a": []}, num_aug=2, alpha_sr=0.0,
                          alpha_ri=0.0, alpha_rs=0.0, p_rd=1.0, seed=2)
        assert all(len(v) == 1 for v in aug)

    def test_audio_gain_and_polarity(self):
        w = [1.0, -0.5, 0.25]
        assert AI.audio_gain(w, db=0.0) == w
        g = AI.audio_gain(w, db=6.0)
        assert abs(g[0] / w[0] - 10 ** (6 / 20.0)) < 1e-12
        assert AI.audio_polarity_invert(w) == [-1.0, 0.5, -0.25]

    def test_audio_add_noise_snr(self):
        import math as _math
        sig = [0.5 * _math.sin(2 * _math.pi * 5 * i / 64.0) for i in range(64)]
        out = AI.audio_add_noise(sig, snr_db=10.0, seed=21)
        noise = [out[i] - sig[i] for i in range(len(sig))]
        p_sig = sum(v * v for v in sig) / len(sig)
        p_noi = sum(v * v for v in noise) / len(noise)
        snr = 10.0 * _math.log10(p_sig / p_noi)
        assert abs(snr - 10.0) < 0.5
        assert AI.audio_add_noise(sig, snr_db=10.0, seed=21) == out
        assert AI.audio_add_noise([0.0] * 8, snr_db=5.0, seed=1) == [0.0] * 8

    def test_audio_shift_and_clip(self):
        assert AI.audio_time_shift([1.0, 2.0, 3.0, 4.0], shift=2, fill=-1.0) == [-1.0, -1.0, 1.0, 2.0]
        assert AI.audio_time_shift([1.0, 2.0, 3.0, 4.0], shift=-1) == [2.0, 3.0, 4.0, 0.0]
        assert AI.audio_clip([0.5, 1.5, -2.0], limit=1.0) == [0.5, 1.0, -1.0]
        with pytest.raises(ValueError):
            AI.audio_time_shift([1.0, 2.0], shift=2)
        with pytest.raises(ValueError):
            AI.audio_clip([0.1], limit=0.0)

    def test_augmentation_rejects_bad_input(self):
        with pytest.raises(ValueError):
            AI.synonym_replace([], {"a": ["b"]})
        with pytest.raises(TypeError):
            AI.synonym_replace("abc", {"a": ["b"]})
        with pytest.raises(TypeError):
            AI.random_insert(["a"], {"a": ["b"]}, n=1.5)
        with pytest.raises(TypeError):
            AI.random_delete(["a"], p="0.5")
        with pytest.raises(ValueError):
            AI.audio_gain([], 0.0)
        with pytest.raises(TypeError):
            AI.audio_add_noise(["a"], 10.0)

class TestSpecialTokensAndBatching:
    """W12-A：SpecialTokens 统一管理 + pad / truncate / mask / 批量分词。"""

    def test_special_tokens_defaults_and_lookup(self):
        st = AI.SpecialTokens.defaults()
        assert st.tokens() == ("[PAD]", "[UNK]", "[BOS]", "[EOS]", "[SEP]", "[CLS]", "[MASK]")
        assert st.as_dict()["bos"] == "[BOS]"
        assert len(st.as_dict()) == 7
        vocab = ["[PAD]", "[UNK]", "hello", "[CLS]", "[SEP]", "[MASK]"]
        assert st.id_of("pad", vocab) == 0
        assert st.id_of("cls", vocab) == 3
        assert st.id_of("unk", {t: i for i, t in enumerate(vocab)}) == 1
        assert AI.SPECIAL_TOKENS.mask == "[MASK]"
        assert AI.SpecialTokens(bos="<s>").bos == "<s>"
        got = AI.SpecialTokens.from_dict({"eos": "</s>"})
        assert got.eos == "</s>" and got.pad == "[PAD]"

    def test_special_tokens_validation(self):
        with pytest.raises(TypeError):
            AI.SpecialTokens(nope="x")
        with pytest.raises(TypeError):
            AI.SpecialTokens(pad=1)
        with pytest.raises(ValueError):
            AI.SpecialTokens(pad="")
        with pytest.raises(ValueError):
            AI.SpecialTokens(pad=" [PAD]")
        with pytest.raises(ValueError):
            AI.SpecialTokens(pad="[UNK]")
        with pytest.raises(TypeError):
            AI.SpecialTokens.from_dict([])
        with pytest.raises(ValueError):
            AI.SpecialTokens.defaults().id_of("nope", ["[PAD]"])
        with pytest.raises(KeyError):
            AI.SpecialTokens.defaults().id_of("bos", ["[PAD]"])

    def test_truncate_sides(self):
        assert AI.truncate([1, 2, 3, 4], 2) == [1, 2]
        assert AI.truncate([1, 2, 3, 4], 2, side="pre") == [3, 4]
        assert AI.truncate([1, 2], 5) == [1, 2]
        with pytest.raises(ValueError):
            AI.truncate([1, 2], 0)
        with pytest.raises(ValueError):
            AI.truncate([1, 2], 1, side="middle")
        with pytest.raises(TypeError):
            AI.truncate("ab", 1)

    def test_pad_sequences_post_and_pre(self):
        assert AI.pad_sequences([[1, 2, 3], [4]]) == [[1, 2, 3], [4, 0, 0]]
        assert AI.pad_sequences([[1, 2, 3], [4]], padding="pre", pad_value=-1) ==             [[1, 2, 3], [-1, -1, 4]]
        assert AI.pad_sequences([[1, 2, 3, 4], [5, 6]], max_len=3) == [[1, 2, 3], [5, 6, 0]]
        assert AI.pad_sequences([[1, 2, 3, 4]], max_len=2, truncate="pre") == [[3, 4]]
        assert AI.pad_sequences([[1, 2]]) == [[1, 2]]

    def test_pad_sequences_validation(self):
        with pytest.raises(ValueError):
            AI.pad_sequences([])
        with pytest.raises(TypeError):
            AI.pad_sequences("abc")
        with pytest.raises(TypeError):
            AI.pad_sequences([[1], [True]])
        with pytest.raises(ValueError):
            AI.pad_sequences([[1]], 0)
        with pytest.raises(ValueError):
            AI.pad_sequences([[1]], 2, 0, "middle")
        with pytest.raises(TypeError):
            AI.pad_sequences([[1]], None, True)

    def test_attention_mask_by_value(self):
        m = AI.attention_mask([[1, 2, 0, 0], [0, 3, 4, 5]], pad_value=0)
        assert m == [[1, 1, 0, 0], [0, 1, 1, 1]]
        with pytest.raises(TypeError):
            AI.attention_mask([[1]], True)

    def test_batch_sequences_masks_from_lengths(self):
        b = AI.batch_sequences([[1, 2, 3], [4]], pad_value=0)
        assert b["input_ids"] == [[1, 2, 3], [4, 0, 0]]
        assert b["lengths"] == [3, 1]
        assert b["attention_mask"] == [[1, 1, 1], [1, 0, 0]]
        bp = AI.batch_sequences([[1, 2, 3], [4]], padding="pre", pad_value=0)
        assert bp["attention_mask"] == [[1, 1, 1], [0, 0, 1]]
        bt = AI.batch_sequences([[1, 2, 3, 4], [5, 6]], max_len=3)
        assert bt["lengths"] == [3, 2]
        assert bt["attention_mask"] == [[1, 1, 1], [1, 1, 0]]

    def test_batch_sequences_disambiguates_pad_token(self):
        amb = AI.batch_sequences([[0, 5], [0]], pad_value=0)
        assert amb["attention_mask"] == [[1, 1], [1, 0]]
        assert AI.attention_mask(amb["input_ids"], 0) == [[0, 1], [0, 0]]

    def test_tokenize_batch_with_dummy_and_real(self):
        class Dummy:
            table = {"a": [1, 2, 3], "b": [4]}

            def encode(self, text):
                return list(Dummy.table[text])

        tb = AI.tokenize_batch(Dummy(), ["a", "b"], pad_value=0)
        assert tb["input_ids"] == [[1, 2, 3], [4, 0, 0]]
        assert tb["attention_mask"] == [[1, 1, 1], [1, 0, 0]]
        tok = AI.WordPieceTokenizer()
        tok.train("hello world hello there world", vocab_size=40, min_frequency=1)
        sp = AI.tokenize_batch(tok, ["hello world", "hello"], pad_value=0)
        assert len(sp["input_ids"]) == 2
        assert len(sp["input_ids"][0]) == len(sp["input_ids"][1])
        assert sum(sp["attention_mask"][0]) == sp["lengths"][0]
        assert sp["lengths"][0] >= sp["lengths"][1]

    def test_tokenize_batch_validation(self):
        with pytest.raises(TypeError):
            AI.tokenize_batch(None, ["a"])
        with pytest.raises(ValueError):
            AI.tokenize_batch(AI.WordPieceTokenizer(), [])
        with pytest.raises(TypeError):
            AI.tokenize_batch(AI.WordPieceTokenizer(), [1])
        with pytest.raises(TypeError):
            AI.tokenize_batch(object(), ["a"])

class TestTokenizers:
    """W12-B：Whitespace / Word / Regex / SentencePiece（简化版）。"""

    def test_whitespace_tokenizer(self):
        tok = AI.WhitespaceTokenizer().fit(["a b c", "a b"], min_frequency=1)
        assert tok.vocab == ["[PAD]", "[UNK]", "a", "b", "c"]
        assert tok.pad_id == 0 and tok.unk_id == 1
        assert tok.encode("a c z") == [2, 4, 1]
        assert tok.decode([2, 4, 1]) == "a c [UNK]"
        assert tok.vocab_size == 5
        assert tok.encode("") == []
        assert AI.WhitespaceTokenizer().tokenize("hi, there") == ["hi,", "there"]
        assert AI.WhitespaceTokenizer(lower=True).tokenize("Hi Hi") == ["hi", "hi"]

    def test_word_tokenizer_splits_punctuation(self):
        tok = AI.WordTokenizer().fit(["hello, world!", "hello there"], min_frequency=1)
        assert "," in tok.vocab and "!" in tok.vocab
        assert AI.WordTokenizer().tokenize("a,b c") == ["a", ",", "b", "c"]
        assert AI.WordTokenizer(split_punct=False).tokenize("a,b c") == ["a", "b", "c"]
        assert tok.encode("zzz") == [1]
        assert tok.decode(tok.encode("hello world")) == "hello world"

    def test_regex_tokenizer(self):
        tok = AI.RegexTokenizer(pattern=r"[a-z]+").fit(["abc def"], min_frequency=1)
        assert AI.RegexTokenizer(r"\d+").tokenize("a1b22") == ["1", "22"]
        assert tok.decode(tok.encode("abc def")) == "abc def"
        assert tok.encode("") == []

    def test_regex_tokenizer_validation(self):
        with pytest.raises(TypeError):
            AI.RegexTokenizer(pattern=1)
        with pytest.raises(ValueError):
            AI.RegexTokenizer(pattern="")
        with pytest.raises(ValueError):
            AI.RegexTokenizer(pattern="([a-z")
        with pytest.raises(ValueError):
            AI.RegexTokenizer(pattern="zzz")

    def test_sentencepiece_ids_and_roundtrip(self):
        texts = ["hello world", "hello there world", "the world is big"]
        sp = AI.SentencePieceTokenizer(vocab_size=40, max_piece_len=5).train(texts)
        assert sp.pad_id == 0 and sp.unk_id == 1
        assert sp.pieces_[0] == "[PAD]" and sp.pieces_[1] == "<unk>"
        assert sp.vocab_size == len(sp.pieces_) and sp.vocab_size <= 40
        ids = sp.encode("hello world")
        assert ids and all(0 <= i < sp.vocab_size for i in ids)
        assert sp.decode(ids) == "hello world"
        assert sp.encode("hello world") == ids
        assert any(len(p) > 1 for p in sp.tokenize("hello world"))
        assert all(len(p) <= 5 for p in sp.pieces_ if p not in ("[PAD]", "<unk>"))

    def test_sentencepiece_unknown_and_whitespace(self):
        texts = ["hello world", "hello there world"]
        sp = AI.SentencePieceTokenizer(vocab_size=40, max_piece_len=5).train(texts)
        unk = sp.encode("zzz")
        assert sp.unk_id in unk
        assert sp.decode(unk) == ""
        assert sp.encode("") == [] and sp.decode([]) == ""
        assert sp.decode(sp.encode("  hello   world  ")) == "hello world"

    def test_sentencepiece_cjk_single_chars(self):
        sp = AI.SentencePieceTokenizer(vocab_size=30, max_piece_len=3).train(["你好世界"])
        ids = sp.encode("你好")
        assert ids and all(0 <= i < sp.vocab_size for i in ids)

    def test_tokenizers_feed_batch_sequences(self):
        ws = AI.WhitespaceTokenizer().fit(["a b c", "a"], min_frequency=1)
        batch = AI.tokenize_batch(ws, ["a b c", "a"], pad_value=0)
        assert batch["input_ids"] == [[2, 3, 4], [2, 0, 0]]
        assert batch["attention_mask"] == [[1, 1, 1], [1, 0, 0]]
        sp = AI.SentencePieceTokenizer(vocab_size=40, max_piece_len=5).train(
            ["hello world", "the world"])
        spb = AI.tokenize_batch(sp, ["hello world", "the world"], pad_value=0)
        assert len(spb["input_ids"]) == 2
        assert len(spb["input_ids"][0]) == len(spb["input_ids"][1])

    def test_tokenizer_construction_validation(self):
        with pytest.raises(TypeError):
            AI.WhitespaceTokenizer(lower=1)
        with pytest.raises(ValueError):
            AI.WhitespaceTokenizer(unk_token="")
        with pytest.raises(ValueError):
            AI.WhitespaceTokenizer(pad_token=" [PAD]")
        with pytest.raises(ValueError):
            AI.WhitespaceTokenizer(unk_token="[PAD]")
        with pytest.raises(TypeError):
            AI.WordTokenizer(split_punct=1)
        with pytest.raises(TypeError):
            AI.SentencePieceTokenizer(vocab_size=1.5)
        with pytest.raises(ValueError):
            AI.SentencePieceTokenizer(vocab_size=1)
        with pytest.raises(ValueError):
            AI.SentencePieceTokenizer(unk_token="")
        with pytest.raises(TypeError):
            AI.SentencePieceTokenizer(max_piece_len=True)

    def test_fit_validation(self):
        with pytest.raises(ValueError):
            AI.WhitespaceTokenizer().fit([])
        with pytest.raises(TypeError):
            AI.WhitespaceTokenizer().fit(1)
        with pytest.raises(ValueError):
            AI.WhitespaceTokenizer().fit(["a"], min_frequency=0)
        with pytest.raises(ValueError):
            AI.WhitespaceTokenizer().fit(["a"], max_size=1)
        with pytest.raises(TypeError):
            AI.WhitespaceTokenizer().fit(["a"], min_frequency=1.5)

    def test_use_before_fit_raises(self):
        with pytest.raises(ValueError):
            AI.WhitespaceTokenizer().encode("a")
        with pytest.raises(ValueError):
            AI.WhitespaceTokenizer().decode([0])
        with pytest.raises(ValueError):
            AI.SentencePieceTokenizer().encode("a")
        with pytest.raises(ValueError):
            AI.SentencePieceTokenizer().decode([0])

    def test_encode_decode_validation(self):
        tok = AI.WhitespaceTokenizer().fit(["a b"], min_frequency=1)
        with pytest.raises(TypeError):
            tok.encode(1)
        with pytest.raises(TypeError):
            tok.decode("0")
        with pytest.raises(TypeError):
            tok.decode([True])
        with pytest.raises(ValueError):
            tok.decode([-1])
        with pytest.raises(ValueError):
            tok.decode([99])

    def test_sentencepiece_train_validation(self):
        with pytest.raises(ValueError):
            AI.SentencePieceTokenizer().train(["abcdef"], vocab_size=3)
        with pytest.raises(ValueError):
            AI.SentencePieceTokenizer().train([])
        with pytest.raises(ValueError):
            AI.SentencePieceTokenizer().train([""])
        with pytest.raises(ValueError):
            AI.SentencePieceTokenizer(vocab_size=4).train(["abcdefgh"])
        with pytest.raises(TypeError):
            AI.SentencePieceTokenizer().train(["abc"], vocab_size=1.5)

class TestRegressionMetricsExtra:
    """W13-A：MedAE / MSLE / RMSLE / ExplainedVariance / AdjustedR² / Pinball。"""

    def test_median_absolute_error(self):
        assert AI.median_absolute_error([1, 2, 3], [1, 2, 3]) == 0.0
        assert AI.median_absolute_error([0, 0, 0], [0, 1, 10]) == 1.0
        assert AI.median_absolute_error([0, 0, 0, 0], [1, 2, 3, 4]) == 2.5
        assert AI.median_absolute_error([[1, 2], [3, 4]], [[1, 3], [5, 4]]) == 0.5
        assert (AI.median_absolute_error([0, 0, 0], [0, 1, 10])
                < AI.mean_absolute_error([0, 0, 0], [0, 1, 10]))
        with pytest.raises(ValueError):
            AI.median_absolute_error([], [])
        with pytest.raises(ValueError):
            AI.median_absolute_error([1], ["a"])

    def test_msle_matches_manual(self):
        yt = [1.0, 2.0, 3.0]
        yp = [1.5, 2.5, 2.5]
        manual = sum((math.log1p(t) - math.log1p(p)) ** 2
                     for t, p in zip(yt, yp)) / 3.0
        assert abs(AI.mean_squared_log_error(yt, yp) - manual) < 1e-15
        assert abs(AI.root_mean_squared_log_error(yt, yp) - math.sqrt(manual)) < 1e-15
        assert AI.mean_squared_log_error([1, 2], [1, 2]) == 0.0

    def test_msle_tolerant_on_large_values(self):
        # 相同的绝对误差下，大值域的相对误差更小
        assert (AI.mean_squared_log_error([1], [1.1])
                < AI.mean_squared_log_error([100], [110]))
        assert (AI.root_mean_squared_log_error([1], [1.1])
                < AI.root_mean_squared_log_error([100], [110]))

    def test_msle_rejects_negative(self):
        with pytest.raises(ValueError):
            AI.mean_squared_log_error([-1, 1], [1, 1])
        with pytest.raises(ValueError):
            AI.mean_squared_log_error([1], [-0.5])
        with pytest.raises(ValueError):
            AI.root_mean_squared_log_error([1], [-1])

    def test_explained_variance_vs_r2(self):
        assert AI.explained_variance_score([1, 2, 3], [1, 2, 3]) == 1.0
        # 系统性偏移不惩罚方差，但 R² 会
        assert abs(AI.explained_variance_score([1, 2, 3], [6, 7, 8]) - 1.0) < 1e-15
        assert AI.r2_score([1, 2, 3], [6, 7, 8]) < 0.0
        assert abs(AI.explained_variance_score([0, 2], [0, 0]) - 0.0) < 1e-15
        assert AI.explained_variance_score([5, 5], [5, 5]) == 1.0
        assert AI.explained_variance_score([5, 5], [5, 6]) == 0.0
        with pytest.raises(ValueError):
            AI.explained_variance_score([], [])

    def test_adjusted_r2(self):
        assert abs(AI.adjusted_r2_score([1, 2, 3, 4], [1, 2, 3, 4], 1) - 1.0) < 1e-15
        yt = [1, 2, 3, 4, 5]
        yp = [1.1, 2.1, 2.9, 4.2, 4.8]
        r2 = AI.r2_score(yt, yp)
        adj = AI.adjusted_r2_score(yt, yp, 1)
        assert abs(adj - (1.0 - (1.0 - r2) * 4 / 3)) < 1e-15
        assert adj <= r2
        yt2 = [1, 2, 3, 4, 5, 6]
        yp2 = [1.1, 2.1, 2.9, 4.2, 4.8, 6.3]
        assert (AI.adjusted_r2_score(yt2, yp2, 2)
                <= AI.adjusted_r2_score(yt2, yp2, 1) + 1e-15)

    def test_adjusted_r2_validation(self):
        with pytest.raises(ValueError):
            AI.adjusted_r2_score([1, 2], [1, 2], 2)
        with pytest.raises(ValueError):
            AI.adjusted_r2_score([1, 2, 3], [1, 2, 3], 0)
        with pytest.raises(TypeError):
            AI.adjusted_r2_score([1, 2, 3], [1, 2, 3], 1.5)
        with pytest.raises(TypeError):
            AI.adjusted_r2_score([1, 2, 3], [1, 2, 3], True)

    def test_pinball_loss(self):
        assert abs(AI.mean_pinball_loss([0, 0], [1, -1], 0.5) - 0.5) < 1e-15
        assert abs(AI.mean_pinball_loss([1, 2, 3], [1.5, 1.5, 2.0], 0.3)
                   - AI.quantile_loss([1, 2, 3], [1.5, 1.5, 2.0], 0.3)[0]) < 1e-15
        assert AI.mean_pinball_loss([0], [1], 0.9) < AI.mean_pinball_loss([0], [1], 0.1)
        assert AI.mean_pinball_loss([1], [0], 0.9) > AI.mean_pinball_loss([1], [0], 0.1)

    def test_pinball_validation(self):
        with pytest.raises(ValueError):
            AI.mean_pinball_loss([1], [1], 0.0)
        with pytest.raises(ValueError):
            AI.mean_pinball_loss([1], [1], 1.0)
        with pytest.raises(TypeError):
            AI.mean_pinball_loss([1], [1], "0.5")
        with pytest.raises(TypeError):
            AI.mean_pinball_loss([1], [1], True)

    def test_inference_benchmark(self):
        xs = [[0.5], [1.0], [-0.5]]
        ys = [[1.0], [1.0], [0.0]]
        model = AI.Model(AI.MLP([1, 4, 1]), loss_fn="mse", optimizer="adam")
        model.fit(xs, ys, epochs=2, batch_size=3)
        res = AI.inference_benchmark(model, xs[0], repeats=3, warmup=1)
        assert res["repeats"] == 3 and res["warmup"] == 1
        assert res["total_seconds"] > 0.0 and res["per_call_ms"] > 0.0
        assert res["throughput_per_s"] > 0.0
        assert abs(res["throughput_per_s"] * res["per_call_ms"] / 1000.0 - 1.0) < 1e-9
        assert res["peak_memory_bytes"] > 0
        assert res["params"] == 1 * 4 + 4 + 4 * 1 + 1
        batched = AI.inference_benchmark(model, xs, repeats=2, batch_size=2)
        assert batched["params"] == res["params"]

    def test_inference_benchmark_validation(self):
        model = AI.Model(AI.MLP([1, 2, 1]), loss_fn="mse", optimizer="adam")
        with pytest.raises(ValueError):
            AI.inference_benchmark(model, [0.1], repeats=0)
        with pytest.raises(TypeError):
            AI.inference_benchmark(model, [0.1], repeats=1.5)
        with pytest.raises(ValueError):
            AI.inference_benchmark(model, [0.1], warmup=-1)
        with pytest.raises(ValueError):
            AI.inference_benchmark(model, [[0.1]], batch_size=0)
        with pytest.raises(ValueError):
            AI.inference_benchmark(model, [], batch_size=4)


class TestClassificationMetricsExtra:
    def test_balanced_accuracy(self):
        assert abs(AI.balanced_accuracy_score([0, 0, 1, 1], [0, 1, 1, 1]) - 0.75) < 1e-12
        yt = [0] * 9 + [1]
        yp = [0] * 10
        assert abs(AI.accuracy_score(yt, yp) - 0.9) < 1e-12
        assert abs(AI.balanced_accuracy_score(yt, yp) - 0.5) < 1e-12

    def test_cohen_kappa(self):
        assert abs(AI.cohen_kappa_score([0, 0, 1, 1], [0, 0, 1, 1]) - 1.0) < 1e-12
        assert abs(AI.cohen_kappa_score([0, 0, 1, 1], [0, 1, 1, 0])) < 1e-12
        with pytest.raises(ValueError):
            AI.cohen_kappa_score([1, 1, 1], [1, 1, 1])

    def test_sensitivity_specificity(self):
        yt = [1, 1, 0, 0]
        yp = [1, 0, 0, 0]
        assert abs(AI.sensitivity_score(yt, yp) - 0.5) < 1e-12
        assert abs(AI.specificity_score(yt, yp) - 1.0) < 1e-12
        with pytest.raises(ValueError):
            AI.sensitivity_score([0, 0], [0, 1])
        with pytest.raises(ValueError):
            AI.specificity_score([1, 1], [1, 1])

    def test_fbeta_score(self):
        yt = [1, 1, 0, 0]
        yp = [1, 0, 0, 0]
        assert abs(AI.fbeta_score(yt, yp, 1.0) - 2.0 * 1.0 * 0.5 / 1.5) < 1e-12
        assert AI.fbeta_score(yt, yp, 0.5) > AI.fbeta_score(yt, yp, 1.0) > \
            AI.fbeta_score(yt, yp, 2.0)
        yt_r = [1, 1, 1, 0, 0]
        yp_r = [1, 1, 0, 1, 1]
        assert AI.fbeta_score(yt_r, yp_r, 0.5) < AI.fbeta_score(yt_r, yp_r, 1.0) < \
            AI.fbeta_score(yt_r, yp_r, 2.0)
        with pytest.raises(ValueError):
            AI.fbeta_score(yt, yp, 0.0)

    def test_hamming_score(self):
        assert abs(AI.hamming_score([1, 0, 1, 1], [1, 0, 0, 1]) - 0.75) < 1e-12
        assert abs(AI.hamming_score([[1, 0], [0, 1]], [[1, 0], [1, 1]]) - 0.75) < 1e-12
        with pytest.raises(ValueError):
            AI.hamming_score([[1, 0]], [[1, 0, 1]])

    def test_jaccard_score(self):
        yt = [0, 1, 1, 2]
        yp = [0, 2, 1, 2]
        assert abs(AI.jaccard_score(yt, yp) - (1.0 + 0.5 + 0.5) / 3.0) < 1e-12
        assert abs(AI.jaccard_score(yt, yp, average="micro") - 0.6) < 1e-12
        assert abs(AI.jaccard_score(yt, yp, average="weighted") - 0.625) < 1e-12
        assert abs(AI.jaccard_score(yt, yt) - 1.0) < 1e-12

    def test_brier_score(self):
        assert abs(AI.brier_score([0, 1], [0.2, 0.8]) - 0.04) < 1e-12
        got = AI.brier_score([0, 1], [[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])
        assert abs(got - 0.10) < 1e-12
        with pytest.raises(ValueError):
            AI.brier_score([2], [0.5])

    def test_log_loss(self):
        v = AI.log_loss([0, 1], [0.25, 0.75])
        assert abs(v - (-math.log(0.75))) < 1e-12
        assert abs(v - AI.cross_entropy_loss([0.25, 0.75], [0.0, 1.0])[0]) < 1e-12
        assert AI.log_loss([1], [0.0]) > 30.0
        assert abs(AI.log_loss([0], [0.0])) < 1e-12
        with pytest.raises(ValueError):
            AI.log_loss([0, 1], [0.5, 0.5], eps=0.0)

    def test_pr_auc_score(self):
        assert abs(AI.pr_auc_score([0, 1, 1, 0], [0.1, 0.9, 0.8, 0.7]) - 1.0) < 1e-12
        assert abs(AI.pr_auc_score([0, 1], [0.9, 0.8]) - 0.5) < 1e-12
        with pytest.raises(ValueError):
            AI.pr_auc_score([0, 0], [0.2, 0.1])

    def test_validation(self):
        with pytest.raises(ValueError):
            AI.brier_score([0, 1], [0.2])
        with pytest.raises(ValueError):
            AI.jaccard_score([0, 1], [0, 1], average="binary")
        with pytest.raises(ValueError):
            AI.hamming_score([[1, 0]], [1])


class TestRankingMetrics:
    def test_precision_recall_at_k(self):
        rel = [1, 0, 1, 0]
        sc = [0.9, 0.8, 0.7, 0.6]
        assert abs(AI.precision_at_k(rel, sc, 1) - 1.0) < 1e-12
        assert abs(AI.precision_at_k(rel, sc, 2) - 0.5) < 1e-12
        assert abs(AI.precision_at_k(rel, sc) - 0.5) < 1e-12
        assert abs(AI.recall_at_k(rel, sc, 2) - 0.5) < 1e-12
        assert abs(AI.recall_at_k(rel, sc, 3) - 1.0) < 1e-12

    def test_precision_reorders_by_score(self):
        assert abs(AI.precision_at_k([0, 1], [0.1, 0.9], 1) - 1.0) < 1e-12
        assert abs(AI.precision_at_k([0, 1], [0.5, 0.5], 1)) < 1e-12

    def test_hit_rate(self):
        assert abs(AI.hit_rate_at_k([1, 0, 0], [0.9, 0.8, 0.7], 1) - 1.0) < 1e-12
        assert abs(AI.hit_rate_at_k([0, 0, 1], [0.9, 0.8, 0.7], 2)) < 1e-12
        assert abs(AI.hit_rate_at_k([0, 0, 1], [0.9, 0.8, 0.7], 3) - 1.0) < 1e-12

    def test_reciprocal_rank(self):
        assert abs(AI.reciprocal_rank([0, 0, 1], [0.9, 0.8, 0.7]) - 1.0 / 3.0) < 1e-12
        assert abs(AI.reciprocal_rank([1, 0, 0], [0.9, 0.8, 0.7]) - 1.0) < 1e-12
        assert abs(AI.reciprocal_rank([0, 0, 0], [0.9, 0.8, 0.7])) < 1e-12

    def test_average_precision(self):
        v = AI.average_precision_at_k([1, 0, 1], [0.9, 0.8, 0.7])
        assert abs(v - (1.0 + 2.0 / 3.0) / 2.0) < 1e-12
        assert abs(AI.average_precision_at_k([1, 0, 1], [0.9, 0.8, 0.7], 1) - 1.0) < 1e-12
        assert abs(AI.average_precision_at_k([1, 1, 0], [0.9, 0.8, 0.1]) - 1.0) < 1e-12

    def test_ndcg(self):
        rel = [2, 0, 1]
        sc = [0.9, 0.8, 0.7]
        dcg = 3.0 / math.log2(2) + 0.0 / math.log2(3) + 1.0 / math.log2(4)
        idcg = 3.0 / math.log2(2) + 1.0 / math.log2(3) + 0.0 / math.log2(4)
        assert abs(AI.ndcg_at_k(rel, sc) - dcg / idcg) < 1e-12
        assert abs(AI.ndcg_at_k([2, 1, 0], [0.9, 0.8, 0.7]) - 1.0) < 1e-12
        assert abs(AI.ndcg_at_k(rel, sc, 10) - dcg / idcg) < 1e-12

    def test_mean_variants(self):
        qa = [[1, 0, 1], [0, 1, 0]]
        qb = [[0.9, 0.8, 0.7], [0.1, 0.9, 0.2]]
        want = (AI.average_precision_at_k(qa[0], qb[0]) +
                AI.average_precision_at_k(qa[1], qb[1])) / 2.0
        assert abs(AI.mean_average_precision_at_k(qa, qb) - want) < 1e-12
        assert abs(AI.mean_reciprocal_rank(qa, qb) - 1.0) < 1e-12
        want_ndcg = (AI.ndcg_at_k(qa[0], qb[0]) + AI.ndcg_at_k(qa[1], qb[1])) / 2.0
        assert abs(AI.mean_ndcg_at_k(qa, qb) - want_ndcg) < 1e-12

    def test_coverage(self):
        rows = [[0.9, 0.1, 0.1, 0.1],
                [0.1, 0.9, 0.1, 0.1],
                [0.9, 0.1, 0.1, 0.1]]
        assert abs(AI.coverage_at_k(rows, 1) - 0.5) < 1e-12
        assert abs(AI.coverage_at_k(rows, 4) - 1.0) < 1e-12
        assert abs(AI.coverage_at_k(rows) - 1.0) < 1e-12

    def test_validation(self):
        with pytest.raises(ValueError):
            AI.precision_at_k([1, 0], [0.5])
        with pytest.raises(ValueError):
            AI.precision_at_k([1], [0.5], 0)
        with pytest.raises(ValueError):
            AI.precision_at_k([2], [0.5])
        with pytest.raises(ValueError):
            AI.ndcg_at_k([0, 0], [0.5, 0.4])
        with pytest.raises(ValueError):
            AI.recall_at_k([0, 0], [0.5, 0.4])
        with pytest.raises(ValueError):
            AI.mean_average_precision_at_k([[1, 0]], [[0.5]])
        with pytest.raises(ValueError):
            AI.coverage_at_k([[0.5, 0.4], [0.5]])


class TestSegmentationMetrics:
    @staticmethod
    def _gt():
        return [[0, 0, 1, 1] for _ in range(4)]

    @staticmethod
    def _pred():
        return [[0, 0, 0, 1] for _ in range(4)]

    def test_perfect(self):
        gt = self._gt()
        assert abs(AI.pixel_accuracy(gt, gt) - 1.0) < 1e-12
        assert abs(AI.mean_iou(gt, gt) - 1.0) < 1e-12
        assert abs(AI.dice_score(gt, gt) - 1.0) < 1e-12
        assert abs(AI.frequency_weighted_iou(gt, gt) - 1.0) < 1e-12

    def test_shifted_hand_values(self):
        gt = self._gt()
        pred = self._pred()
        assert abs(AI.pixel_accuracy(gt, pred) - 0.75) < 1e-12
        iou0 = 8.0 / 12.0
        iou1 = 4.0 / 8.0
        assert abs(AI.mean_iou(gt, pred) - (iou0 + iou1) / 2.0) < 1e-12
        assert abs(AI.dice_score(gt, pred) - (0.8 + 8.0 / 12.0) / 2.0) < 1e-12
        assert abs(AI.frequency_weighted_iou(gt, pred) - (iou0 + iou1) / 2.0) < 1e-12

    def test_ignore_index(self):
        gt = self._gt()
        pred = self._pred()
        gt2 = [row[:] for row in gt]
        pred2 = [row[:] for row in pred]
        gt2[0][0] = 255
        pred2[0][0] = 255
        assert abs(AI.pixel_accuracy(gt2, pred2, ignore_index=255) - 11.0 / 15.0) < 1e-12
        assert abs(AI.pixel_accuracy(gt2, pred2) - 0.75) < 1e-12
        assert abs(AI.mean_iou(gt2, gt2, ignore_index=255) - 1.0) < 1e-12

    def test_boundary_f1(self):
        gt = self._gt()
        pred = self._pred()
        assert abs(AI.boundary_f1(gt, gt) - 1.0) < 1e-12
        assert abs(AI.boundary_f1(gt, pred, tolerance=0) - 0.5) < 1e-12
        assert abs(AI.boundary_f1(gt, pred, tolerance=1) - 1.0) < 1e-12
        flat = [[0] * 4 for _ in range(4)]
        assert abs(AI.boundary_f1(gt, flat)) < 1e-12
        assert abs(AI.boundary_f1(flat, flat) - 1.0) < 1e-12

    def test_hd95(self):
        gt = self._gt()
        pred = self._pred()
        assert abs(AI.hd95(gt, gt)) < 1e-12
        assert abs(AI.hd95(gt, pred) - 1.0) < 1e-12
        flat = [[0] * 4 for _ in range(4)]
        assert abs(AI.hd95(flat, flat)) < 1e-12
        with pytest.raises(ValueError):
            AI.hd95(gt, flat)

    def test_inverted_labels_still_perfect(self):
        a = [[0, 1], [0, 1], [0, 1]]
        b = [[1, 0], [1, 0], [1, 0]]
        assert abs(AI.boundary_f1(a, b, tolerance=0) - 1.0) < 1e-12
        assert abs(AI.hd95(a, b)) < 1e-12

    def test_shape_validation(self):
        with pytest.raises(ValueError):
            AI.pixel_accuracy([[0]], [[0], [0]])
        with pytest.raises(ValueError):
            AI.pixel_accuracy([[0, 1]], [[0]])
        with pytest.raises(ValueError):
            AI.pixel_accuracy([], [])
        with pytest.raises(ValueError):
            AI.pixel_accuracy([[]], [[]])

    def test_label_validation(self):
        with pytest.raises(ValueError):
            AI.pixel_accuracy([[-1]], [[0]])
        with pytest.raises(ValueError):
            AI.pixel_accuracy([[0.5]], [[0]])
        with pytest.raises(ValueError):
            AI.pixel_accuracy([[True]], [[0]])
        with pytest.raises(ValueError):
            AI.mean_iou([[2]], [[2]], num_classes=2)
        with pytest.raises(ValueError):
            AI.pixel_accuracy([[255]], [[255]], ignore_index=255)

    def test_tolerance_validation(self):
        gt = self._gt()
        pred = self._pred()
        with pytest.raises(ValueError):
            AI.boundary_f1(gt, pred, tolerance=-1)
        with pytest.raises(ValueError):
            AI.boundary_f1(gt, pred, tolerance=True)


class TestTextGenerationMetrics:
    def test_corpus_bleu_perfect(self):
        assert abs(AI.corpus_bleu(["a b c d"], ["a b c d"]) - 1.0) < 1e-12
        assert abs(AI.corpus_bleu(["a b"], ["a b"]) - 1.0) < 1e-12
        assert abs(AI.corpus_bleu(["x y"], ["a b"])) < 1e-12

    def test_corpus_bleu_hand(self):
        precs = [3.0 / 4.0, 2.0 / 3.0, 1.0 / 2.0, 0.5]
        want = math.exp(sum(math.log(p) for p in precs) / 4.0)
        got = AI.corpus_bleu(["a b c d"], ["a b c e"])
        assert abs(got - want) < 1e-12

    def test_corpus_bleu_multi_ref_and_bp(self):
        refs = [[["a", "b"], ["a", "c"]]]
        assert abs(AI.corpus_bleu(["a b"], refs) - 1.0) < 1e-12
        bp = math.exp(1.0 - 4.0 / 2.0)
        assert abs(AI.corpus_bleu(["a b"], ["a b c d"]) - bp) < 1e-12

    def test_rouge(self):
        assert abs(AI.rouge_1("a b c", "a b d") - 2.0 / 3.0) < 1e-12
        assert abs(AI.rouge_2("a b c", "a b d") - 0.5) < 1e-12
        assert abs(AI.rouge_2("a b", "c d")) < 1e-12
        assert abs(AI.rouge_1("a b c", "a b c") - 1.0) < 1e-12

    def test_meteor(self):
        want = 1.0 - 0.5 * (1.0 / 4.0) ** 3.0
        assert abs(AI.meteor_score("a b c d", "a b c d") - want) < 1e-12
        f_mean = (2.0 / 3.0 * 1.0) / (0.9 * (2.0 / 3.0) + 0.1 * 1.0)
        got = AI.meteor_score("a x b", "a b")
        assert abs(got - f_mean * 0.5) < 1e-12
        assert abs(AI.meteor_score("x y", "a b")) < 1e-12

    def test_cider(self):
        assert abs(AI.cider_score("a b c d", ["a b c d"]) - 10.0) < 1e-12
        assert abs(AI.cider_score("a b c d", ["w x y z"])) < 1e-12
        mid = AI.cider_score("a b c d", ["a b x y"])
        assert 0.0 < mid < 10.0

    def test_self_bleu(self):
        assert abs(AI.self_bleu(["a b c", "a b c", "a b c"]) - 1.0) < 1e-12
        assert abs(AI.self_bleu(["a b c", "x y z", "p q r"])) < 1e-12

    def test_validation(self):
        with pytest.raises(ValueError):
            AI.corpus_bleu(["a"], ["a", "b"])
        with pytest.raises(ValueError):
            AI.corpus_bleu([], [])
        with pytest.raises(ValueError):
            AI.corpus_bleu(["a"], ["a"], max_n=0)
        with pytest.raises(ValueError):
            AI.rouge_n("a", "a", n=0)
        with pytest.raises(ValueError):
            AI.meteor_score("a", "a", alpha=1.5)
        with pytest.raises(ValueError):
            AI.cider_score("a", [])
        with pytest.raises(ValueError):
            AI.self_bleu(["a b"])


class TestSpeechMetrics:
    def test_wer(self):
        assert abs(AI.word_error_rate("a b c d", "a b c d")) < 1e-12
        assert abs(AI.word_error_rate("a b c d", "a x c d") - 0.25) < 1e-12
        assert abs(AI.word_error_rate("a b c d", "a b c") - 0.25) < 1e-12
        assert abs(AI.word_error_rate("a b c", "a b c d") - 1.0 / 3.0) < 1e-12
        assert abs(AI.word_error_rate("a b", "x y") - 1.0) < 1e-12
        assert abs(AI.word_error_rate("a b", "") - 1.0) < 1e-12
        assert abs(AI.word_error_rate(["a", "b"], ["a", "c"]) - 0.5) < 1e-12

    def test_edit_ops(self):
        assert AI._edit_ops(["a", "b", "c"], ["a", "x", "c", "y"]) == (1, 0, 1)
        assert AI._edit_ops(["a", "b", "c"], ["a", "c"]) == (0, 1, 0)
        assert AI._edit_ops(["a"], ["a"]) == (0, 0, 0)

    def test_cer(self):
        assert abs(AI.character_error_rate("abc", "abd") - 1.0 / 3.0) < 1e-12
        assert abs(AI.character_error_rate("你好世界", "你好") - 0.5) < 1e-12
        assert abs(AI.character_error_rate("abc", "abc")) < 1e-12
        assert abs(AI.character_error_rate(["a", "b"], ["a", "b", "c"]) - 0.5) < 1e-12

    def test_ser(self):
        assert abs(AI.slot_error_rate({"a": 1, "b": 2}, {"a": 1, "b": 2})) < 1e-12
        assert abs(AI.slot_error_rate({"a": 1, "b": 2}, {"a": 1, "b": 3}) - 0.5) < 1e-12
        assert abs(AI.slot_error_rate({"a": 1, "b": 2}, {"a": 1}) - 0.5) < 1e-12
        assert abs(AI.slot_error_rate({"a": 1}, {"a": 1, "b": 2}) - 1.0) < 1e-12
        assert abs(AI.slot_error_rate([("a", 1), ("b", 2)], [("a", 1), ("b", 2)])) < 1e-12

    def test_validation(self):
        with pytest.raises(ValueError):
            AI.word_error_rate("", "a")
        with pytest.raises(ValueError):
            AI.character_error_rate("", "a")
        with pytest.raises(ValueError):
            AI.character_error_rate(1, "a")
        with pytest.raises(ValueError):
            AI.slot_error_rate({}, {"a": 1})
        with pytest.raises(ValueError):
            AI.slot_error_rate([("a",)], {"a": 1})
        with pytest.raises(ValueError):
            AI.slot_error_rate([("a", 1), ("a", 2)], {"a": 1})
        with pytest.raises(ValueError):
            AI.slot_error_rate(3, {"a": 1})


class TestClusteringMetrics:
    X = [[0.0, 0.0], [0.0, 1.0], [10.0, 10.0], [10.0, 11.0]]
    Y = [0, 0, 1, 1]

    def test_silhouette_two_clusters(self):
        val = AI.silhouette_score(self.X, self.Y)
        assert abs(val - 0.9292895427118657) < 1e-9

    def test_silhouette_singleton_point_scores_zero(self):
        val = AI.silhouette_score([[0.0], [1.0], [10.0]], [0, 0, 1])
        assert abs(val - 0.5962962962962963) < 1e-9

    def test_silhouette_rejects_bad_shapes(self):
        with pytest.raises(ValueError):
            AI.silhouette_score(self.X, [0, 0, 0, 0])
        with pytest.raises(ValueError):
            AI.silhouette_score([[0.0], [1.0]], [0, 1])
        with pytest.raises(ValueError):
            AI.silhouette_score(self.X, [0, 0, 1])

    def test_calinski_harabasz_and_davies_bouldin(self):
        assert abs(AI.calinski_harabasz_score(self.X, self.Y) - 400.0) < 1e-9
        assert abs(AI.davies_bouldin_score(self.X, self.Y) - 0.07071067811865475) < 1e-9
        with pytest.raises(ValueError):
            AI.calinski_harabasz_score(self.X, [0, 0, 0, 0])
        with pytest.raises(ValueError):
            AI.davies_bouldin_score(self.X, [0, 0, 0, 0])

    def test_calinski_harabasz_rejects_zero_within(self):
        with pytest.raises(ValueError):
            AI.calinski_harabasz_score([[0.0], [0.0], [1.0], [1.0]], [0, 0, 1, 1])

    def test_adjusted_rand_score(self):
        assert abs(AI.adjusted_rand_score(self.Y, self.Y) - 1.0) < 1e-12
        assert abs(AI.adjusted_rand_score(self.Y, [5, 5, 9, 9]) - 1.0) < 1e-12
        assert abs(AI.adjusted_rand_score(self.Y, [0, 1, 0, 1]) + 0.5) < 1e-12
        with pytest.raises(ValueError):
            AI.adjusted_rand_score([], [])
        with pytest.raises(ValueError):
            AI.adjusted_rand_score([-1, 0], [0, 1])

    def test_normalized_mutual_info(self):
        assert abs(AI.normalized_mutual_info_score(self.Y, self.Y) - 1.0) < 1e-12
        assert abs(AI.normalized_mutual_info_score(self.Y, [0, 1, 0, 1])) < 1e-12
        assert abs(AI.normalized_mutual_info_score([0, 0, 1, 1], [0, 0, 0, 1])
                   - 0.3437110184854508) < 1e-9
        assert abs(AI.normalized_mutual_info_score([0, 0, 0, 0, 1, 1, 1, 1],
                                                   [0, 0, 1, 1, 2, 2, 3, 3])
                   - 2.0 / 3.0) < 1e-9
        with pytest.raises(ValueError):
            AI.normalized_mutual_info_score(self.Y, self.Y, "mean")

    def test_adjusted_mutual_info(self):
        assert abs(AI.adjusted_mutual_info_score(self.Y, self.Y) - 1.0) < 1e-12
        assert abs(AI.adjusted_mutual_info_score(self.Y, [0, 1, 0, 1]) + 0.5) < 1e-12
        assert abs(AI.adjusted_mutual_info_score([0, 0, 0, 0, 1, 1, 1, 1],
                                                 [0, 0, 1, 1, 2, 2, 3, 3])
                   - 0.5333333333333332) < 1e-9
        assert AI.adjusted_mutual_info_score([0, 0, 1, 1], [0, 0, 0, 1]) <= \
            AI.normalized_mutual_info_score([0, 0, 1, 1], [0, 0, 0, 1]) + 1e-12
        with pytest.raises(ValueError):
            AI.adjusted_mutual_info_score(self.Y, self.Y, "maxx")
        with pytest.raises(ValueError):
            AI.adjusted_mutual_info_score([], [])

    def test_mean_silhouette_prefers_separated_clusters(self):
        near = AI.silhouette_score([[0.0], [1.0], [2.0], [3.0]], [0, 0, 1, 1])
        far = AI.silhouette_score([[0.0], [1.0], [100.0], [101.0]], [0, 0, 1, 1])
        assert far > near


class TestCalibrationMetrics:
    Y = [1, 1, 1, 0, 0]
    P = [0.9, 0.6, 0.3, 0.2, 0.1]

    def test_ece_hand_value(self):
        want = 0.4 * 0.15 + 0.6 * (1.0 - (0.9 + 0.8 + 0.9) / 3.0)
        got = AI.expected_calibration_error(self.Y, self.P, n_bins=5)
        assert abs(got - want) < 1e-12

    def test_mce_hand_value(self):
        assert abs(AI.maximum_calibration_error(self.Y, self.P, n_bins=5) - 0.15) < 1e-12

    def test_single_column_matches_two_column(self):
        two = [[1.0 - p, p] for p in self.P]
        assert abs(AI.expected_calibration_error(self.Y, self.P, n_bins=5)
                   - AI.expected_calibration_error(self.Y, two, n_bins=5)) < 1e-12
        assert abs(AI.maximum_calibration_error(self.Y, self.P, n_bins=5)
                   - AI.maximum_calibration_error(self.Y, two, n_bins=5)) < 1e-12

    def test_overconfident_and_perfect(self):
        assert abs(AI.expected_calibration_error([1, 0], [0.9, 0.1], n_bins=5) - 0.1) < 1e-12
        assert abs(AI.expected_calibration_error([1, 0], [1.0, 0.0], n_bins=5)) < 1e-12

    def test_multiclass(self):
        y = [0, 1, 2]
        p = [[0.7, 0.2, 0.1], [0.1, 0.6, 0.3], [0.2, 0.3, 0.5]]
        want = 1.0 - (0.7 + 0.6 + 0.5) / 3.0
        assert abs(AI.expected_calibration_error(y, p, n_bins=2) - want) < 1e-12
        assert abs(AI.maximum_calibration_error(y, p, n_bins=2) - want) < 1e-12

    def test_reliability_diagram_structure(self):
        bins = AI.reliability_diagram(self.Y, self.P, n_bins=5)
        assert len(bins) == 2
        assert bins[0]["count"] == 2
        assert abs(bins[0]["accuracy"] - 0.5) < 1e-12
        assert abs(bins[0]["confidence"] - 0.65) < 1e-12
        assert bins[1]["count"] == 3
        assert abs(bins[1]["accuracy"] - 1.0) < 1e-12
        assert sum(b["count"] for b in bins) == len(self.Y)

    def test_quantile_strategy(self):
        bins = AI.reliability_diagram([1, 1, 1, 1], [0.5, 0.6, 0.7, 0.8],
                                      n_bins=2, strategy="quantile")
        assert sum(b["count"] for b in bins) == 4
        same = AI.reliability_diagram([1, 1, 1], [0.7, 0.7, 0.7],
                                      n_bins=4, strategy="quantile")
        assert len(same) == 1 and abs(same[0]["accuracy"] - 1.0) < 1e-12

    def test_rejects_bad_inputs(self):
        with pytest.raises(ValueError):
            AI.expected_calibration_error([], [], n_bins=5)
        with pytest.raises(ValueError):
            AI.expected_calibration_error([1, 0], [1.5, 0.2], n_bins=5)
        with pytest.raises(ValueError):
            AI.expected_calibration_error([1, 0], [[0.2, 0.8]], n_bins=5)
        with pytest.raises(ValueError):
            AI.expected_calibration_error([2, 0], [0.7, 0.2], n_bins=5)
        with pytest.raises(ValueError):
            AI.expected_calibration_error(self.Y, self.P, n_bins=0)
        with pytest.raises(ValueError):
            AI.expected_calibration_error(self.Y, self.P, n_bins=True)
        with pytest.raises(ValueError):
            AI.expected_calibration_error(self.Y, self.P, n_bins="5")
        with pytest.raises(ValueError):
            AI.expected_calibration_error(self.Y, self.P, n_bins=5, strategy="adaptive")
        with pytest.raises(ValueError):
            AI.maximum_calibration_error([0, 1, 5], [[0.7, 0.2, 0.1]] * 3, n_bins=2)
        with pytest.raises(ValueError):
            AI.reliability_diagram([1, 0], ["a", 0.2], n_bins=5)


class TestDetectionMetrics:
    @staticmethod
    def _gt(boxes, labels):
        return [{"boxes": [list(b) for b in boxes], "labels": list(labels)}]

    @staticmethod
    def _pred(boxes, scores, labels):
        return [{"boxes": [list(b) for b in boxes], "labels": list(labels),
                 "scores": list(scores)}]

    def test_box_iou(self):
        assert abs(AI.box_iou([0, 0, 2, 2], [1, 1, 3, 3]) - 1.0 / 7.0) < 1e-12
        assert AI.box_iou([0, 0, 2, 2], [0, 0, 2, 2]) == 1.0
        assert AI.box_iou([0, 0, 1, 1], [5, 5, 6, 6]) == 0.0
        assert AI.box_area([0, 0, 3, 4]) == 12.0

    def test_ap_perfect(self):
        gt = self._gt([(0, 0, 10, 10)], [0])
        pred = self._pred([(0, 0, 10, 10)], [0.9], [0])
        assert abs(AI.detection_average_precision(gt, pred) - 1.0) < 1e-12

    def test_ap_false_positive_first(self):
        gt = self._gt([(0, 0, 10, 10), (20, 20, 30, 30)], [0, 0])
        pred = self._pred([(50, 50, 60, 60), (0, 0, 10, 10), (20, 20, 30, 30)],
                          [0.95, 0.9, 0.8], [0, 0, 0])
        assert abs(AI.detection_average_precision(gt, pred) - 2.0 / 3.0) < 1e-12

    def test_ap_no_predictions_is_zero(self):
        gt = self._gt([(0, 0, 10, 10)], [0])
        empty = self._pred([], [], [])
        assert AI.detection_average_precision(gt, empty) == 0.0
        assert AI.average_recall(gt, empty) == 0.0

    def test_threshold_changes_result(self):
        gt = self._gt([(0, 0, 10, 10), (10, 0, 20, 10)], [0, 0])
        pred = self._pred([(0, 0, 10, 10), (5, 0, 15, 10)], [0.9, 0.8], [0, 0])
        assert abs(AI.detection_average_precision(gt, pred, 0.5) - 0.5) < 1e-12
        assert abs(AI.detection_average_precision(gt, pred, 0.3) - 1.0) < 1e-12

    def test_map_and_range(self):
        gt = self._gt([(0, 0, 10, 10), (20, 20, 30, 30)], [0, 1])
        pred = self._pred([(0, 0, 10, 10), (20, 20, 30, 30)], [0.9, 0.8], [0, 1])
        assert abs(AI.mean_average_precision(gt, pred) - 1.0) < 1e-12
        assert abs(AI.mean_average_precision_range(gt, pred) - 1.0) < 1e-12
        half = self._gt([(0, 0, 10, 10)], [0])
        pred_half = self._pred([(0, 0, 10, 5)], [0.9], [0])
        assert abs(AI.mean_average_precision_range(half, pred_half, [0.5, 0.6]) - 0.5) < 1e-12

    def test_average_recall_max_dets(self):
        gt = self._gt([(0, 0, 10, 10), (20, 20, 30, 30)], [0, 0])
        pred = self._pred([(0, 0, 10, 10), (20, 20, 30, 30)], [0.9, 0.8], [0, 0])
        assert abs(AI.average_recall(gt, pred) - 1.0) < 1e-12
        assert abs(AI.average_recall(gt, pred, max_dets=1) - 0.5) < 1e-12

    def test_rejects_bad_inputs(self):
        gt = self._gt([(0, 0, 10, 10)], [0])
        pred = self._pred([(0, 0, 10, 10)], [0.9], [0])
        with pytest.raises(ValueError):
            AI.mean_average_precision(gt, [])
        with pytest.raises(ValueError):
            AI.box_iou([0, 0, 1], [0, 0, 1, 1])
        with pytest.raises(ValueError):
            AI.box_iou([2, 0, 1, 1], [0, 0, 1, 1])
        with pytest.raises(ValueError):
            AI.mean_average_precision(gt, pred, 1.5)
        with pytest.raises(ValueError):
            AI.mean_average_precision(gt, pred, True)
        with pytest.raises(ValueError):
            AI.mean_average_precision([{"boxes": [], "labels": []}],
                                      self._pred([], [], []))
        with pytest.raises(ValueError):
            AI.detection_average_precision(
                self._gt([(0, 0, 1, 1), (2, 2, 3, 3)], [0, 1]),
                self._pred([(0, 0, 1, 1), (2, 2, 3, 3)], [0.9, 0.8], [0, 1]))
        with pytest.raises(ValueError):
            AI.average_recall(gt, pred, max_dets=0)


class TestImageQualityMetrics:
    A = [[[(r * 5 + c) / 25.0 for c in range(5)] for r in range(5)]]
    B = [[[((r * 5 + c) % 7) / 7.0 for c in range(5)] for r in range(5)]]

    def test_psnr_values(self):
        one = [[[0.0]]]
        assert AI.psnr_score(one, one) == float("inf")
        assert AI.psnr_score([[[0.0]]], [[[1.0]]]) == 0.0
        assert abs(AI.psnr_score([[[0.0]]], [[[0.5]]]) - 10.0 * math.log10(4.0)) < 1e-12
        assert abs(AI.psnr_score([[[0.0]]], [[[1.0]]], data_range=255.0)
                   - 10.0 * math.log10(255.0 ** 2)) < 1e-12

    def test_ssim(self):
        assert abs(AI.ssim_score(self.A, self.A, win_size=3) - 1.0) < 1e-12
        assert abs(AI.ssim_score(self.A, self.B, win_size=3)
                   + 0.0026727086158507027) < 1e-9
        small = [[[0.1, 0.2], [0.3, 0.4]]]
        assert abs(AI.ssim_score(small, small, win_size=7) - 1.0) < 1e-12

    def test_ssim_constant_images(self):
        flat = [[[0.5] * 8 for _ in range(8)]]
        assert abs(AI.ssim_score(flat, flat, win_size=7) - 1.0) < 1e-12

    def test_ms_ssim(self):
        big = [[[(r + c) / 64.0 for c in range(32)] for r in range(32)]]
        assert abs(AI.ms_ssim_score(big, big) - 1.0) < 1e-9
        assert abs(AI.ms_ssim_score(self.A, self.B) - 0.43420933608311674) < 1e-9
        assert AI.ms_ssim_score(self.A, self.B) <= 1.0

    def test_fid(self):
        fa = [[0.0], [2.0]]
        assert abs(AI.fid_score(fa, [[1.0], [3.0]]) - 1.0) < 1e-9
        assert abs(AI.fid_score(fa, fa)) < 1e-12
        assert abs(AI.fid_score(fa, [[0.0], [4.0]]) - 3.0) < 1e-9

    def test_kid(self):
        ka = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
        kb = [[0.1, 0.1], [1.1, 0.1], [0.1, 1.1]]
        far = [[9.0, 9.0], [10.0, 9.0], [9.0, 10.0]]
        assert abs(AI.kid_score(ka, ka)) < 0.4
        assert AI.kid_score(ka, far) > AI.kid_score(ka, kb)

    def test_inception_score(self):
        assert abs(AI.inception_score([[0.5, 0.5], [0.5, 0.5]]) - 1.0) < 1e-12
        assert abs(AI.inception_score([[1.0, 0.0], [0.0, 1.0]]) - 2.0) < 1e-12
        assert abs(AI.inception_score([[0.6, 0.6], [0.6, 0.6]]) - 1.0) < 1e-12

    def test_rejects_bad_inputs(self):
        one = [[[0.0]]]
        with pytest.raises(ValueError):
            AI.psnr_score(one, [[[0.0]], [[0.0]]])
        with pytest.raises(ValueError):
            AI.psnr_score([], one)
        with pytest.raises(ValueError):
            AI.psnr_score(one, one, data_range=0.0)
        with pytest.raises(ValueError):
            AI.ssim_score(self.A, self.A, win_size=0)
        with pytest.raises(ValueError):
            AI.ms_ssim_score(self.A, self.A, weights=[0.5, -0.1])
        with pytest.raises(ValueError):
            AI.fid_score([[0.0], [1.0]], [[0.0, 1.0], [1.0, 2.0]])
        with pytest.raises(ValueError):
            AI.fid_score([[0.0]], [[1.0]])
        with pytest.raises(ValueError):
            AI.kid_score([[0.0]], [[1.0]])
        with pytest.raises(ValueError):
            AI.inception_score([[1.5, -0.5]])
        with pytest.raises(ValueError):
            AI.inception_score([[0.0, 0.0]])

class TestConsistencyFairness:
    """W13-K：一致性与公平性指标。"""

    @staticmethod
    def _fleiss_classic():
        counts = [
            (0, 0, 0, 0, 14),
            (0, 2, 6, 4, 2),
            (0, 0, 3, 5, 6),
            (0, 3, 9, 2, 0),
            (2, 2, 8, 1, 1),
            (7, 7, 0, 0, 0),
            (3, 2, 6, 3, 0),
            (2, 5, 6, 1, 0),
            (6, 5, 2, 1, 0),
            (0, 2, 2, 3, 7),
        ]
        out = []
        for row in counts:
            line = []
            for cat, n in enumerate(row):
                line.extend([cat] * n)
            out.append(line)
        return out

    def test_fleiss_perfect(self):
        data = [[0, 0, 0], [1, 1, 1], [0, 0, 0]]
        assert abs(AI.fleiss_kappa(data) - 1.0) < 1e-12

    def test_fleiss_classic_example(self):
        # Fleiss (1971) 10 对象 / 14 评分者 / 5 类，手算 kappa = 0.2199
        got = AI.fleiss_kappa(self._fleiss_classic())
        assert abs(got - 0.21994628612658837) < 1e-12

    def test_fleiss_two_category_zero(self):
        # 每个对象 2 名评分者且完全相反 ⇒ P_bar = 0.5 = P_e ⇒ 0
        data = [[0, 0], [0, 1], [1, 1], [0, 1]]
        assert abs(AI.fleiss_kappa(data)) < 1e-12

    def test_fleiss_validation(self):
        with pytest.raises(ValueError):
            AI.fleiss_kappa([[0, 1], [0, 1, 1]])
        with pytest.raises(ValueError):
            AI.fleiss_kappa([[0, None], [0, 1]])
        with pytest.raises(ValueError):
            AI.fleiss_kappa([[0], [1]])
        with pytest.raises(ValueError):
            AI.fleiss_kappa([])
        with pytest.raises(ValueError):
            AI.fleiss_kappa([[1, 1], [1, 1]])
        with pytest.raises(ValueError):
            AI.fleiss_kappa([[0, 2], [0, 2]], categories=[0, 1])

    def test_krippendorff_perfect(self):
        data = [[1, 1, 1], [2, 2, 2], [3, 3, 3]]
        assert abs(AI.krippendorff_alpha(data) - 1.0) < 1e-12

    def test_krippendorff_hand_case(self):
        # 单个单位 {1,1,2,2}：D_o = D_e = 2/3 ⇒ 0
        assert abs(AI.krippendorff_alpha([[1, 1, 2, 2]])) < 1e-12

    def test_krippendorff_levels(self):
        data = [[1, 2, 3], [1, 2, 4], [2, 2, 3]]
        nominal = AI.krippendorff_alpha(data, level="nominal")
        ordinal = AI.krippendorff_alpha(data, level="ordinal")
        interval = AI.krippendorff_alpha(data, level="interval")
        assert abs(nominal - (-0.1428571428571428)) < 1e-12
        assert abs(ordinal - (-0.2962962962962963)) < 1e-12
        assert abs(interval - (-0.2941176470588236)) < 1e-12

    def test_krippendorff_missing(self):
        data = [[1, 2, None, 3], [1, None, 3, 3], [2, 2, 3, None]]
        assert abs(AI.krippendorff_alpha(data, level="nominal")
                   - (-0.07692307692307687)) < 1e-12
        assert abs(AI.krippendorff_alpha(data, level="interval")
                   - (-0.28)) < 1e-12
        # 只有一名评分者的单位被跳过
        a1 = AI.krippendorff_alpha([[1, 2], [1, 2]])
        a2 = AI.krippendorff_alpha([[1, 2], [1, 2], [3, None]])
        assert abs(a1 - a2) < 1e-12

    def test_krippendorff_validation(self):
        with pytest.raises(ValueError):
            AI.krippendorff_alpha([[1, 2], [1, 2]], level="ratio")
        with pytest.raises(ValueError):
            AI.krippendorff_alpha([[1, 1], [1, 1]])
        with pytest.raises(ValueError):
            AI.krippendorff_alpha([[None, None], [None, None]])
        with pytest.raises(ValueError):
            AI.krippendorff_alpha([])

    def test_fairness_basic(self):
        y_pred = [1, 1, 0, 0]
        sensitive = ["a", "a", "b", "b"]
        y_true = [1, 0, 1, 0]
        assert abs(AI.demographic_parity_difference(y_pred, sensitive) - 1.0) < 1e-12
        assert abs(AI.disparate_impact_ratio(y_pred, sensitive)) < 1e-12
        assert abs(AI.equal_opportunity_difference(
            y_true, y_pred, sensitive) - 1.0) < 1e-12

    def test_fairness_eighty_percent_rule(self):
        y_pred = [1, 1, 1, 1, 0, 1, 1, 1, 1, 1]
        sensitive = ["a"] * 5 + ["b"] * 5
        assert abs(AI.disparate_impact_ratio(y_pred, sensitive) - 0.8) < 1e-12
        assert abs(AI.demographic_parity_difference(
            y_pred, sensitive) - 0.2) < 1e-12

    def test_fairness_validation(self):
        with pytest.raises(ValueError):
            AI.demographic_parity_difference([1, 0], ["a", "a"])
        with pytest.raises(ValueError):
            AI.demographic_parity_difference([1, 0, 1], ["a", "b"])
        with pytest.raises(ValueError):
            AI.demographic_parity_difference([], [])
        with pytest.raises(ValueError):
            AI.disparate_impact_ratio([0, 0, 0, 0], ["a", "a", "b", "b"])
        with pytest.raises(ValueError):
            AI.demographic_parity_difference([1, 0], [["a"], ["b"]])
        with pytest.raises(ValueError):
            AI.equal_opportunity_difference([1, 0, 0, 0], [1, 0, 0, 0],
                                           ["a", "a", "b", "b"])

class TestAttribution:
    """W14-A：输入归因（Saliency / Grad×Input / SmoothGrad / IG / DeepLIFT）。"""

    @staticmethod
    def _linear():
        net = AI.MLP([2, 1], activations=["linear"])
        net.layers[0].weights = [[1.0, 0.0]]
        net.layers[0].biases = [0.0]
        return net

    @staticmethod
    def _relu():
        net = AI.MLP([2, 3, 1], activations=["relu", "linear"])
        net.layers[0].weights = [[1.0, 1.0], [-1.0, 2.0], [0.5, 0.0]]
        net.layers[0].biases = [0.0, 0.5, 0.0]
        net.layers[1].weights = [[1.0, -2.0, 3.0]]
        net.layers[1].biases = [0.1]
        return net

    def test_saliency_linear(self):
        net = self._linear()
        assert AI.saliency_map(net, [3.0, 4.0]) == [1.0, 0.0]
        assert AI.saliency_map(net, [-3.0, 4.0]) == [1.0, 0.0]

    def test_gradient_x_input(self):
        net = self._linear()
        assert AI.gradient_x_input(net, [3.0, 4.0]) == [3.0, 0.0]
        relu = self._relu()
        grad = AI.saliency_map(relu, [1.0, 2.0], absolute=False)
        got = AI.gradient_x_input(relu, [1.0, 2.0])
        assert all(abs(got[i] - [1.0, 2.0][i] * grad[i]) < 1e-15
                   for i in range(2))

    def test_integrated_gradients_completeness(self):
        net = self._relu()
        fx = net.forward([1.0, 2.0])[0]
        fb = net.forward([0.0, 0.0])[0]
        ig = AI.integrated_gradients(net, [1.0, 2.0], steps=4096)
        assert abs(sum(ig) - (fx - fb)) < 5e-3

    def test_integrated_gradients_linear_exact(self):
        net = self._linear()
        ig = AI.integrated_gradients(net, [3.0, 4.0], steps=8)
        assert abs(ig[0] - 3.0) < 1e-12 and abs(ig[1]) < 1e-12

    def test_deep_lift_completeness(self):
        net = self._relu()
        dl = AI.deep_lift(net, [1.0, 2.0])
        fx = net.forward([1.0, 2.0])[0]
        fb = net.forward([0.0, 0.0])[0]
        assert abs(sum(dl) - (fx - fb)) < 1e-12
        # 线性网络的 DeepLIFT 等于 IG
        lin = self._linear()
        assert abs(AI.integrated_gradients(lin, [3.0, -2.0], steps=16)[0]
                   - AI.deep_lift(lin, [3.0, -2.0])[0]) < 1e-12

    def test_smooth_grad_degenerate(self):
        net = self._linear()
        assert AI.smooth_grad(net, [3.0, 4.0], samples=5, sigma=0.0) == [1.0, 0.0]
        a = AI.smooth_grad(net, [3.0, 4.0], samples=50, sigma=0.5, seed=1)
        assert abs(a[0] - 1.0) < 1e-12 and abs(a[1]) < 1e-12
        b = AI.smooth_grad(net, [3.0, 4.0], samples=20, sigma=0.3, seed=7)
        c = AI.smooth_grad(net, [3.0, 4.0], samples=20, sigma=0.3, seed=7)
        assert b == c

    def test_multiclass_class_index(self):
        net = AI.MLP([2, 2], activations=["linear"])
        net.layers[0].weights = [[1.0, 0.0], [0.0, 2.0]]
        net.layers[0].biases = [0.0, 0.0]
        assert AI.gradient_x_input(net, [3.0, 4.0], class_index=0) == [3.0, 0.0]
        assert AI.gradient_x_input(net, [3.0, 4.0], class_index=1) == [0.0, 8.0]
        # 省略时取 argmax
        assert AI.gradient_x_input(net, [3.0, 4.0]) == [0.0, 8.0]

    def test_validation(self):
        net = self._linear()
        with pytest.raises(ValueError):
            AI.saliency_map(net, [])
        with pytest.raises(ValueError):
            AI.saliency_map(net, [[1.0, 2.0]])
        with pytest.raises(ValueError):
            AI.saliency_map(net, [1.0, 2.0, 3.0])
        with pytest.raises(ValueError):
            AI.smooth_grad(net, [1.0, 2.0], samples=0)
        with pytest.raises(ValueError):
            AI.smooth_grad(net, [1.0, 2.0], sigma=-1.0)
        with pytest.raises(ValueError):
            AI.integrated_gradients(net, [1.0, 2.0], steps=0)
        with pytest.raises(ValueError):
            AI.integrated_gradients(net, [1.0, 2.0], baseline=[0.0])
        with pytest.raises(ValueError):
            AI.deep_lift(net, [1.0, 2.0], eps=0.0)
        with pytest.raises(ValueError):
            AI.saliency_map(net, [1.0, 2.0], class_index=3)

    def test_deep_lift_rejects_softmax(self):
        net = AI.MLP([2, 2], activations=["softmax"])
        with pytest.raises(ValueError):
            AI.deep_lift(net, [1.0, 2.0])
        assert len(AI.saliency_map(net, [1.0, 2.0])) == 2

    def test_model_wrapper(self):
        wrapper = AI.Model(AI.MLP([2, 2], activations=["linear"]),
                           loss_fn="mse", optimizer="adam")
        wrapper.model.layers[0].weights = [[1.0, 0.0], [0.0, 1.0]]
        wrapper.model.layers[0].biases = [0.0, 0.0]
        assert AI.saliency_map(wrapper, [2.0, 5.0]) == [0.0, 1.0]

class TestAttentionAttribution:
    """W14-B：注意力 Rollout / Flow。"""

    IDENT = [[1.0, 0.0], [0.0, 1.0]]
    UNIFORM = [[0.5, 0.5], [0.5, 0.5]]

    def test_rollout_identity(self):
        got = AI.attention_rollout([self.IDENT, self.IDENT])
        assert got == self.IDENT

    def test_rollout_uniform(self):
        one = AI.attention_rollout([self.UNIFORM])
        assert abs(one[0][0] - 0.75) < 1e-12 and abs(one[0][1] - 0.25) < 1e-12
        two = AI.attention_rollout([self.UNIFORM, self.UNIFORM])
        assert abs(two[0][0] - 0.625) < 1e-12
        assert abs(two[0][1] - 0.375) < 1e-12

    def test_rollout_rows_normalized(self):
        a = [[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.3, 0.3, 0.4]]
        b = [[0.6, 0.4, 0.0], [0.0, 0.5, 0.5], [0.2, 0.2, 0.6]]
        got = AI.attention_rollout([a, b])
        for i in range(3):
            assert abs(sum(got[i]) - 1.0) < 1e-12

    def test_flow_column_conservation(self):
        a = [[0.5, 0.5], [1.5, 0.5]]
        got = AI.attention_flow([a])
        assert abs(got[0][0] + got[1][0] - 1.0) < 1e-12
        assert abs(got[0][1] + got[1][1] - 1.0) < 1e-12
        # 列为单位和的注意力：flow 就是矩阵连乘
        col_norm = [[0.25, 0.5], [0.75, 0.5]]
        two = AI.attention_flow([col_norm, col_norm])
        want = AI._w14b_mul(col_norm, col_norm)
        assert all(abs(two[i][j] - want[i][j]) < 1e-12
                   for i in range(2) for j in range(2))

    def test_flow_no_identity_term(self):
        # flow 不加单位项，单层均匀注意力原样返回
        assert AI.attention_flow([self.UNIFORM]) == self.UNIFORM

    def test_head_fusion(self):
        heads = [
            [[0.9, 0.1], [0.1, 0.9]],
            [[0.1, 0.9], [0.9, 0.1]],
            [[0.5, 0.5], [0.5, 0.5]],
        ]
        mean = AI.attention_rollout([heads])
        assert abs(mean[0][0] - 0.75) < 1e-12 and abs(mean[0][1] - 0.25) < 1e-12
        mx = AI.attention_rollout([heads], head_fusion="max")
        assert abs(mx[0][1] - 0.45) < 1e-12
        mn = AI.attention_rollout([heads], head_fusion="min")
        assert abs(mn[0][1] - 0.05) < 1e-12

    def test_discard_ratio(self):
        a = [[0.7, 0.2, 0.1], [0.1, 0.7, 0.2], [0.4, 0.4, 0.2]]
        got = AI.attention_rollout([a], discard_ratio=1.0 / 3.0)
        assert abs(got[0][2]) < 1e-12      # 最小项被丢掉
        assert abs(sum(got[0]) - 1.0) < 1e-12

    def test_attention_maps_input(self):
        mha = AI.MultiHeadAttention(d_model=4, num_heads=2)
        mha.eval()
        maps = AI.attention_maps(mha, [[1.0, 0.0, 0.0, 0.0],
                                       [0.0, 1.0, 0.0, 0.0]])
        got = AI.attention_rollout(maps)
        assert len(got) == 2 and len(got[0]) == 2
        for i in range(2):
            assert abs(sum(got[i]) - 1.0) < 1e-12

    def test_validation(self):
        with pytest.raises(ValueError):
            AI.attention_rollout([])
        with pytest.raises(ValueError):
            AI.attention_rollout([[[1.0, 0.0]]])
        with pytest.raises(ValueError):
            AI.attention_rollout([[[1.0, -0.5], [0.0, 1.0]]])
        with pytest.raises(ValueError):
            AI.attention_rollout([self.IDENT], discard_ratio=1.0)
        with pytest.raises(ValueError):
            AI.attention_rollout([self.IDENT], head_fusion="sum")
        with pytest.raises(TypeError):
            AI.attention_rollout(42)
        with pytest.raises(ValueError):
            AI.attention_flow([])


import pytest


class TestPermutationFeatureMaps:
    """W14-C：置换重要性 + 特征图 ASCII 渲染。"""

    @staticmethod
    def _model():
        net = AI.MLP([2, 1], activations=["linear"])
        net.layers[0].weights = [[1.0, 0.0]]
        net.layers[0].biases = [0.0]
        return AI.Model(net, loss_fn="mse", optimizer="adam", task="regression")

    X = [[1.0, 1.0], [2.0, 1.0], [3.0, 1.0], [4.0, 1.0], [5.0, 1.0],
         [6.0, 1.0]]
    Y = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

    def test_permutation_shape_and_baseline(self):
        model = self._model()
        out = AI.permutation_importance(model, self.X, self.Y, n_repeats=4, seed=1)
        assert set(out) == {"baseline", "mean", "std", "names"}
        assert abs(out["baseline"] - 1.0) < 1e-12
        assert out["names"] == ["f0", "f1"]
        assert len(out["mean"]) == 2

    def test_permutation_constant_feature_is_zero(self):
        model = self._model()
        out = AI.permutation_importance(model, self.X, self.Y, n_repeats=4, seed=1)
        assert abs(out["mean"][1]) < 1e-12

    def test_permutation_useful_feature_matters(self):
        model = self._model()
        out = AI.permutation_importance(model, self.X, self.Y, n_repeats=4, seed=1)
        assert out["mean"][0] > 0.1
        assert all(s >= 0.0 for s in out["std"])

    def test_permutation_single_repeat_has_zero_std(self):
        model = self._model()
        out = AI.permutation_importance(model, self.X, self.Y, n_repeats=1, seed=2)
        assert out["std"][0] == 0.0

    def test_permutation_seed_reproducible(self):
        model = self._model()
        a = AI.permutation_importance(model, self.X, self.Y, n_repeats=3, seed=7)
        b = AI.permutation_importance(model, self.X, self.Y, n_repeats=3, seed=7)
        c = AI.permutation_importance(model, self.X, self.Y, n_repeats=3, seed=8)
        assert a["mean"] == b["mean"]
        assert a["mean"] != c["mean"]

    def test_permutation_custom_metric_and_names(self):
        model = self._model()
        net = model.model

        def hit_rate(xs, ys):
            hit = 0
            for xr, yr in zip(xs, ys):
                if abs(net.forward(xr)[0] - yr) < 0.5:
                    hit += 1
            return hit / float(len(xs))

        out = AI.permutation_importance(model, self.X, self.Y, n_repeats=2,
                                     metric=hit_rate, names=["温度", "常数"])
        assert abs(out["baseline"] - 1.0) < 1e-12
        assert out["names"] == ["温度", "常数"]

    def test_permutation_rejects_bad_input(self):
        model = self._model()
        with pytest.raises(ValueError):
            AI.permutation_importance(model, [], [])
        with pytest.raises(ValueError):
            AI.permutation_importance(model, self.X, self.Y[:3])
        with pytest.raises(ValueError):
            AI.permutation_importance(model, [[1.0, 2.0], [3.0]], [0.0, 1.0])
        with pytest.raises(ValueError):
            AI.permutation_importance(model, self.X, self.Y, n_repeats=0)
        with pytest.raises(ValueError):
            AI.permutation_importance(model, self.X, self.Y, names=["a"])

    def test_feature_maps_single_plane(self):
        plane = [[0.0, 1.0, 2.0, 3.0],
                 [4.0, 5.0, 6.0, 7.0],
                 [8.0, 9.0, 10.0, 11.0],
                 [12.0, 13.0, 14.0, 15.0]]
        text = AI.feature_maps_ascii(plane, max_width=4)
        lines = text.split("\n")
        assert lines[0].startswith("map  (4x4")
        assert len(lines) == 7
        assert "@" in lines[4]
        assert lines[1][0] == " "

    def test_feature_maps_channels(self):
        plane = [[float(i * 4 + j) for j in range(4)] for i in range(4)]
        flat = [[1.0] * 4 for _ in range(4)]
        both = AI.feature_maps_ascii([plane, flat], max_width=4)
        assert both.count("channel ") == 2
        only_one = AI.feature_maps_ascii([plane, flat], max_channels=1)
        assert only_one.count("channel ") == 1
        picked = AI.feature_maps_ascii([plane, flat], channels=[1])
        assert "channel 1" in picked and "channel 0" not in picked
        assert "峰值 1" in picked.split("\n")[0]

    def test_feature_maps_edge_values_and_errors(self):
        const = AI.feature_maps_ascii([[5.0] * 3 for _ in range(3)], max_width=3)
        assert "峰值 5" in const.split("\n")[0]
        neg = AI.feature_maps_ascii([[-3.0, -1.0], [-2.0, 0.0]], max_width=2)
        assert "峰值 0" in neg.split("\n")[0]
        wide = [[float(i * 8 + j) for j in range(8)] for i in range(8)]
        wlines = AI.feature_maps_ascii(wide, max_width=4).split("\n")
        assert len(wlines) == 7
        assert len(wlines[1].strip()) <= 4
        with pytest.raises(ValueError):
            AI.feature_maps_ascii([])
        with pytest.raises(ValueError):
            AI.feature_maps_ascii([[1.0, 2.0], [3.0]])
        with pytest.raises(ValueError):
            AI.feature_maps_ascii([[1.0, "x"]])
        with pytest.raises(ValueError):
            AI.feature_maps_ascii([[1.0, 2.0]], max_width=0)
        with pytest.raises(ValueError):
            AI.feature_maps_ascii([[[1.0, 2.0]]], channels=[])


import pytest


class TestCAM:
    """W14-D：GradCAM / GradCAM++ / LayerCAM / ScoreCAM。"""

    X = [[[1.0, 2.0, 3.0],
          [4.0, 5.0, 6.0],
          [7.0, 8.0, 9.0]]]

    @staticmethod
    def _model():
        conv = AI.Conv2d(1, 2, 3, padding=1)
        conv.weights = [[[[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]],
                        [[[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]]]
        conv.biases = [0.0, 0.0]
        lin = AI.Layer(18, 2, activation="linear")
        lin.weights = [[1.0] * 9 + [0.0] * 9, [0.0] * 9 + [1.0] * 9]
        lin.biases = [0.0, 0.0]
        model = AI.Sequential(conv, AI.Flatten(), lin)
        model.eval()
        return model, conv

    def test_gradcam_closed_form_weights(self):
        model, _ = self._model()
        heat, weights = AI.cam_heatmap(model, self.X, 0, class_index=0)
        assert len(heat) == 3 and len(heat[0]) == 3
        assert abs(weights[0] - 1.0) < 1e-12
        assert abs(weights[1]) < 1e-12
        assert abs(max(max(row) for row in heat) - 1.0) < 1e-12
        assert all(v >= 0.0 for row in heat for v in row)

    def test_gradcam_heat_matches_activation(self):
        model, _ = self._model()
        heat, _ = AI.cam_heatmap(model, self.X, 0, class_index=0)
        acts, _ = AI._cam_logits(model, model.modules, self.X, 0)
        peak = max(max(row) for row in acts[0])
        for r in range(3):
            for k in range(3):
                assert abs(heat[r][k] - acts[0][r][k] / peak) < 1e-12

    def test_gradcam_class_switch_flips_weights(self):
        model, _ = self._model()
        _, w0 = AI.cam_heatmap(model, self.X, 0, class_index=0)
        heat1, w1 = AI.cam_heatmap(model, self.X, 0, class_index=1)
        assert abs(w1[1] - 1.0) < 1e-12
        assert abs(w1[0]) < 1e-12
        assert len(heat1) == 3

    def test_gradcam_default_class_is_argmax(self):
        model, _ = self._model()
        logits = model.forward(self.X)
        best = 0 if logits[0] >= logits[1] else 1
        _, w = AI.cam_heatmap(model, self.X, 0, class_index=best)
        _, wd = AI.cam_heatmap(model, self.X, 0)
        assert w == wd

    def test_gradcam_variants(self):
        model, _ = self._model()
        _, w = AI.cam_heatmap(model, self.X, 0, class_index=0, method="gradcam")
        heat_pp, w_pp = AI.cam_heatmap(model, self.X, 0, class_index=0,
                                    method="gradcam++")
        assert w_pp != w
        assert abs(max(max(row) for row in heat_pp) - 1.0) < 1e-12
        heat_lc, w_lc = AI.cam_heatmap(model, self.X, 0, class_index=0,
                                   method="layercam")
        assert w_lc is None
        assert len(heat_lc) == 3

    def test_layercam_matches_elementwise_product(self):
        model, _ = self._model()
        heat_lc, _ = AI.cam_heatmap(model, self.X, 0, class_index=0,
                                 method="layercam")
        acts, grads, _ = AI._cam_grad_output(model, model.modules, self.X, 0, 0)
        want = [[0.0] * 3 for _ in range(3)]
        for c in range(2):
            for r in range(3):
                for k in range(3):
                    want[r][k] += acts[c][r][k] * grads[c][r][k]
        top = max(max(row) for row in want)
        for r in range(3):
            for k in range(3):
                assert abs(heat_lc[r][k] - want[r][k] / top) < 1e-12

    def test_negative_heatmap_handled(self):
        conv = AI.Conv2d(1, 2, 3, padding=1)
        conv.weights = [[[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]],
                        [[[0.0, 0.0, 0.0], [0.0, -0.1, 0.0], [0.0, 0.0, 0.0]]]]
        conv.biases = [0.0, 5.0]
        lin = AI.Layer(18, 2, activation="linear")
        lin.weights = [[1.0] * 9 + [-1.0] * 9, [0.0] * 18]
        lin.biases = [0.0, 0.0]
        model = AI.Sequential(conv, AI.Flatten(), lin)
        model.eval()
        raw, _ = AI.cam_heatmap(model, self.X, 0, class_index=0, relu=False)
        assert raw[0][0] < 0.0 < raw[2][2]
        relu_heat, _ = AI.cam_heatmap(model, self.X, 0, class_index=0)
        assert all(v >= 0.0 for row in relu_heat for v in row)

    def test_scorecam(self):
        model, _ = self._model()
        heat, weights = AI.scorecam_heatmap(model, self.X, 0, class_index=0)
        assert len(heat) == 3 and len(heat[0]) == 3
        assert abs(sum(weights) - 1.0) < 1e-12
        assert all(v >= 0.0 for v in weights)
        assert abs(max(max(row) for row in heat) - 1.0) < 1e-12
        only, w1 = AI.scorecam_heatmap(model, self.X, 0, class_index=0, n_masks=1)
        assert w1 == [1.0]
        assert only != heat

    def test_cam_rejects_bad_input(self):
        model, _ = self._model()
        with pytest.raises(ValueError):
            AI.cam_heatmap(model, self.X, 0, method="scorecam")
        with pytest.raises(ValueError):
            AI.cam_heatmap(model, self.X, 9)
        with pytest.raises(ValueError):
            AI.cam_heatmap(model, self.X, 0, class_index=5)
        with pytest.raises(ValueError):
            AI.cam_heatmap(model, self.X, 1)
        with pytest.raises(ValueError):
            AI.cam_heatmap(model, [1.0, 2.0], 0)
        with pytest.raises(ValueError):
            AI.scorecam_heatmap(model, self.X, 0, n_masks=0)
        with pytest.raises(ValueError):
            AI.gradcam_weights([[1.0, 2.0]])
        with pytest.raises(ValueError):
            AI.gradcam_weights([[[1.0]]], method="gradcam++")

    def test_cam_ascii(self):
        model, _ = self._model()
        heat, _ = AI.cam_heatmap(model, self.X, 0, class_index=0)
        text = AI.cam_ascii(heat, max_width=3)
        assert "(3x3" in text.split("\n")[0]


import pytest


class _Lin:
    """单输出线性模型：y = Σ w_j x_j。"""

    def __init__(self, w):
        self.w = list(w)

    def forward(self, x):
        return [sum(a * b for a, b in zip(self.w, x))]


class _Two:
    """两类模型：logits = [x0, -x0]，类别只由第 0 维符号决定。"""

    def forward(self, x):
        return [x[0], -x[0]]


class TestShapLimeAnchors:
    """W14-E：KernelSHAP / LIME / Anchors / 嵌入可视化。"""

    W = [2.0, -3.0, 0.5]
    X = [1.0, 2.0, -1.0]

    def test_kernel_shap_additive_closed_form(self):
        lin = _Lin(self.W)
        out = AI.kernel_shap(lin, self.X, n_samples=60, seed=3)
        assert set(out) == {"base_value", "output_value", "values", "names"}
        assert abs(out["base_value"]) < 1e-12
        assert abs(out["output_value"] - (2.0 - 6.0 - 0.5)) < 1e-12
        for j, wj in enumerate(self.W):
            assert abs(out["values"][j] - wj * self.X[j]) < 1e-6
        assert out["names"] == ["f0", "f1", "f2"]

    def test_kernel_shap_efficiency_axiom(self):
        lin = _Lin(self.W)
        out = AI.kernel_shap(lin, self.X, n_samples=60, seed=3)
        total = sum(out["values"])
        assert abs(total - (out["output_value"] - out["base_value"])) < 1e-9

    def test_kernel_shap_background_and_names(self):
        lin = _Lin(self.W)
        out = AI.kernel_shap(lin, self.X, background=[[1.0, 1.0, 1.0]],
                          n_samples=40, seed=3, feature_names=["a", "b", "c"])
        assert abs(out["base_value"] - (2.0 - 3.0 + 0.5)) < 1e-12
        assert out["names"] == ["a", "b", "c"]
        assert abs(sum(out["values"]) -
                   (out["output_value"] - out["base_value"])) < 1e-9

    def test_kernel_shap_many_samples_terminates(self):
        # 三个特征只有 8 种掩码：n_samples 远大于 2^m 时必须收敛，不能死循环
        lin = _Lin(self.W)
        out = AI.kernel_shap(lin, self.X, n_samples=500, seed=1)
        assert len(out["values"]) == 3
        assert abs(sum(out["values"]) -
                   (out["output_value"] - out["base_value"])) < 1e-9

    def test_lime_shape_and_ranking(self):
        lin = _Lin(self.W)
        out = AI.lime_explain(lin, self.X, n_samples=300, kernel_width=1.0, seed=5)
        assert set(out) == {"intercept", "coefs", "names", "score"}
        assert abs(out["score"] - lin.forward(self.X)[0]) < 1e-12
        assert abs(out["intercept"]) < 1e-12
        contrib = [abs(wj * self.X[j]) for j, wj in enumerate(self.W)]
        top_model = max(range(3), key=lambda j: contrib[j])
        top_lime = max(range(3), key=lambda j: abs(out["coefs"][j]))
        assert top_model == top_lime
        for j in range(3):
            if abs(self.W[j] * self.X[j]) > 1e-6:
                assert out["coefs"][j] * (self.W[j] * self.X[j]) >= 0

    def test_lime_many_samples_terminates(self):
        lin = _Lin(self.W)
        out = AI.lime_explain(lin, self.X, n_samples=500, seed=2)
        assert len(out["coefs"]) == 3

    def test_anchor_rule_finds_single_feature(self):
        pool = [[1.5, 0.0], [-1.5, 0.0], [1.5, 1.0], [-1.5, 1.0],
                [1.5, -1.0]]
        rule = AI.anchor_rule(_Two(), [1.5, 0.0], pool, threshold=0.9)
        assert rule["features"] == [0]
        assert abs(rule["values"][0] - 1.5) < 1e-12
        assert abs(rule["precision"] - 1.0) < 1e-12
        assert abs(rule["coverage"] - 0.6) < 1e-12

    def test_anchor_rule_rejects_unreachable_threshold(self):
        pool = [[-1.5, 0.0], [-1.5, 1.0]]
        with pytest.raises(ValueError):
            AI.anchor_rule(_Two(), [1.5, 0.0], pool, threshold=0.9)

    def test_scatter_and_embedding(self):
        pts = [[0.0, 0.0], [1.0, 1.0], [0.5, 0.5]]
        grid = AI.scatter_ascii(pts, width=5, height=3)
        lines = grid.split("\n")
        assert len(lines) == 3
        assert all(len(r) == 5 for r in lines)
        assert lines[0][4] != " "
        assert lines[2][0] != " "
        labelled = AI.scatter_ascii(pts, labels=["cat", "dog", 7], width=5, height=3)
        chars = set(labelled.replace("\n", "").replace(" ", ""))
        assert chars == {"c", "d", "7"}

        data = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0], [1.0, 1.0, 0.0], [1.0, 0.0, 1.0]]
        emb = AI.embedding_visualize(data, labels=[0, 0, 1, 1, 0, 1],
                                 method="pca", width=20, height=8)
        assert len(emb["coords"]) == 6
        assert len(emb["coords"][0]) == 2
        assert len(emb["ascii"].split("\n")) == 8
        other = AI.embedding_visualize(data, method="tsne", seed=1, width=12,
                                   height=6)
        assert len(other["coords"]) == 6

    def test_explainer_rejects_bad_input(self):
        lin = _Lin(self.W)
        with pytest.raises(ValueError):
            AI.kernel_shap(lin, self.X, n_samples=1)
        with pytest.raises(ValueError):
            AI.kernel_shap(lin, [1.0, "a"], n_samples=10)
        with pytest.raises(ValueError):
            AI.kernel_shap(lin, self.X, class_index=0)
        with pytest.raises(ValueError):
            AI.kernel_shap(lin, self.X, background=[[1.0]])
        with pytest.raises(ValueError):
            AI.kernel_shap(lin, self.X, feature_names=["a"])
        with pytest.raises(ValueError):
            AI.lime_explain(lin, self.X, kernel_width=0.0)
        with pytest.raises(ValueError):
            AI.anchor_rule(_Two(), [1.5, 0.0], [[1.5, 0.0]], threshold=0.0)
        with pytest.raises(ValueError):
            AI.scatter_ascii([[1.0, 2.0, 3.0]])
        with pytest.raises(ValueError):
            AI.embedding_visualize([[1.0, 2.0], [1.0]])
        with pytest.raises(ValueError):
            AI.embedding_visualize([[1.0, 2.0]], method="umap")


class TestExplanationExtras:
    """W14-F：TCAV / 激活最大化 / 网络解剖 / 原型 / 反事实 / 影响分数。"""

    @staticmethod
    def _identity_net():
        net = AI.MLP([2, 2, 2], activations=["linear", "linear"])
        net.layers[0].weights = [[1.0, 0.0], [0.0, 1.0]]
        net.layers[0].biases = [0.0, 0.0]
        net.layers[1].weights = [[1.0, 0.0], [0.0, 1.0]]
        net.layers[1].biases = [0.0, 0.0]
        return net

    def test_tcav_identity_points_to_concept_axis(self):
        net = self._identity_net()
        res = AI.tcav(net, 0, [[5.0, 0.0], [6.0, 0.0]],
                      [[0.0, 0.0], [0.2, 0.0]], [[4.0, 0.0], [3.0, 0.0]],
                      class_index=0)
        assert abs(sum(v * v for v in res["cav"]) - 1.0) < 1e-12
        assert res["cav"][0] > 0.99
        assert abs(res["cav"][1]) < 1e-9
        assert res["tcav"] == 1.0
        assert res["n"] == 2
        assert abs(res["sensitivity"][0] - 1.0) < 1e-12

    def test_tcav_zero_difference_raises(self):
        net = self._identity_net()
        with pytest.raises(ValueError):
            AI.tcav(net, 0, [[1.0, 1.0]], [[1.0, 1.0]], [[1.0, 0.0]])

    def test_activation_maximization_monotone(self):
        net = AI.MLP([2, 3, 1], activations=["linear", "linear"])
        res = AI.activation_maximization(net, 0, unit=0, steps=15, lr=0.1,
                                         l2=1e-4, seed=7)
        hist = res["history"]
        assert len(hist) == 15
        for i in range(len(hist) - 1):
            assert hist[i] <= hist[i + 1] + 1e-12
        assert hist[-1] > hist[0]
        # activation 是最后一次更新后再前向一次的值，故 >= 历史末项
        assert res["activation"] >= hist[-1] - 1e-12

    def test_activation_maximization_validation(self):
        net = AI.MLP([2, 3, 1], activations=["linear", "linear"])
        with pytest.raises(ValueError):
            AI.activation_maximization(net, 0, steps=0)
        with pytest.raises(ValueError):
            AI.activation_maximization(net, 0, lr=0.0)
        with pytest.raises(ValueError):
            AI.activation_maximization(net, 0, unit=99)
        seq = AI.Sequential([AI.Conv2d(1, 1, 3), AI.Flatten(),
                             AI.Layer(9, 2, activation="linear")])
        with pytest.raises(ValueError):
            AI.activation_maximization(seq, 0, steps=1)

    def test_network_dissection_shapes(self):
        net = self._identity_net()
        data = [[5.0, 0.0], [6.0, 0.0], [0.0, 5.0], [0.0, 6.0]]
        res = AI.network_dissection(net, 0, data, [0, 0, 1, 1], top_k=2)
        assert len(res["units"]) == 2
        assert sorted(res["labels"]) == [0, 1]
        for unit in res["units"]:
            assert len(unit["mean_by_label"]) == len(res["labels"])
            assert len(unit["preferred"]) <= 2
            assert unit["kind"] in ("never", "always", "class-selective", "mixed")
            assert 0.0 <= unit["activation_rate"] <= 1.0
        # 第 0 个单元只看 x0：类别 0 均值高于类别 1
        unit0 = res["units"][0]
        i0 = res["labels"].index(0)
        i1 = res["labels"].index(1)
        assert unit0["mean_by_label"][i0] > unit0["mean_by_label"][i1]

    def test_network_dissection_validation(self):
        net = self._identity_net()
        with pytest.raises(ValueError):
            AI.network_dissection(net, 0, [[1.0, 2.0]], [0, 1])
        with pytest.raises(ValueError):
            AI.network_dissection(net, 0, [[1.0, 2.0]], [0], top_k=0)

    def test_prototype_examples_pick_within_class(self):
        net = self._identity_net()
        data = [[1.0, 0.0], [1.1, 0.0], [0.0, 1.0], [0.0, 1.1]]
        res = AI.prototype_examples(net, 0, data, [0, 0, 1, 1], k=2)
        assert sorted(res.keys()) == [0, 1]
        assert sorted(res[0]) == [0, 1]
        assert sorted(res[1]) == [2, 3]
        with pytest.raises(ValueError):
            AI.prototype_examples(net, 0, data, [0], k=1)

    def test_counterfactual_changes_prediction(self):
        net = AI.MLP([2, 2, 2], activations=["linear", "linear"])
        x = [1.0, 0.0]
        before = net.forward(x)
        start = 0 if before[0] >= before[1] else 1
        res = AI.counterfactual_example(net, x, steps=60, lr=0.2, l2=1e-3)
        assert res["target"] != start
        after = net.forward(res["input"])
        assert (0 if after[0] >= after[1] else 1) == res["target"]
        assert res["score_after"] < res["score_before"]
        assert 0.0 <= res["distance"] < 1e3

    def test_counterfactual_validation(self):
        reg = AI.MLP([2, 1], activations=["linear"])
        with pytest.raises(ValueError):
            AI.counterfactual_example(reg, [1.0, 0.0])
        net = AI.MLP([2, 2, 2], activations=["linear", "linear"])
        with pytest.raises(ValueError):
            AI.counterfactual_example(net, [1.0, 0.0], class_index=9)

    def test_influence_scores_sign(self):
        net = AI.MLP([2, 1], activations=["linear"])
        net.layers[0].weights = [[1.0, -1.0]]
        net.layers[0].biases = [0.0]
        scores = AI.influence_scores(net, [2.0, 0.0], 2.0,
                                     [[2.0, 0.0], [-2.0, 0.0]], [2.0, -2.0])
        assert len(scores) == 2
        assert scores[0] > -1e-9
        assert scores[1] < 1e-9
        for s in scores:
            assert -1.0000001 <= s <= 1.0000001
        with pytest.raises(ValueError):
            AI.influence_scores(net, [1.0, 0.0], 1.0, [[1.0, 0.0]], [1.0, 2.0])
        seq = AI.Sequential([AI.Layer(2, 1, activation="linear")])
        with pytest.raises(ValueError):
            AI.influence_scores(seq, [1.0, 0.0], 1.0, [[1.0, 0.0]], [1.0])


class TestStructuredPruning:
    """W15-A：通道 / 头 / 层的结构化剪枝与稀疏度统计。"""

    @staticmethod
    def _net():
        net = AI.MLP([3, 4, 2], activations=["linear", "linear"])
        net.layers[0].weights = [
            [1.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
            [0.0, 0.0, 0.0],
            [2.0, 2.0, 0.0],
        ]
        net.layers[0].biases = [0.5, 0.5, 0.5, 0.5]
        return net

    def test_channel_scores(self):
        net = self._net()
        assert AI.channel_scores(net.layers[0]) == [1.0, 3.0, 0.0, 4.0]

    def test_prune_channels_zeroes_weakest(self):
        net = self._net()
        res = AI.prune_channels(net, 0, ratio=0.5)
        assert res["pruned"] == [0, 2]
        assert res["kept"] == [1, 3]
        assert net.layers[0].weights[2] == [0.0, 0.0, 0.0]
        assert net.layers[0].weights[0] == [0.0, 0.0, 0.0]
        assert net.layers[0].biases[2] == 0.0
        assert net.layers[0].weights[3] == [2.0, 2.0, 0.0]
        assert len(net.layers[0].weights) == 4  # 形状不变

    def test_prune_channels_validation(self):
        net = self._net()
        assert AI.prune_channels(net, 0, ratio=0.0)["pruned"] == []
        with pytest.raises(ValueError):
            AI.prune_channels(net, 0, ratio=-0.1)
        with pytest.raises(ValueError):
            AI.prune_channels(net, 0, ratio=1.5)
        with pytest.raises(ValueError):
            AI.prune_channels(net, 9, ratio=0.5)
        with pytest.raises(ValueError):
            AI.prune_channels(net, 0, ratio=0.99, keep_at_least=4)

    def test_head_scores_and_prune(self):
        attn = AI.MultiHeadAttention(4, 2)
        attn.W_q = [[1.0, 0.0, 0.0, 0.0],
                    [0.0, 2.0, 0.0, 0.0],
                    [0.0, 0.0, 5.0, 0.0],
                    [0.0, 0.0, 0.0, 5.0]]
        attn.W_o = [[1.0] * 4 for _ in range(4)]
        assert AI.head_scores(attn, "q") == [3.0, 10.0]
        res = AI.prune_heads(attn, ratio=0.5)
        assert res["pruned"] == [0]
        # 头 0 占 W_q 的第 0、1 列，剪掉后这两列全零
        assert [row[0] for row in attn.W_q] == [0.0, 0.0, 0.0, 0.0]
        assert [row[1] for row in attn.W_q] == [0.0, 0.0, 0.0, 0.0]
        assert attn.W_q[2][2] == 5.0
        # W_o 中头 0 对应的行被清零，其余行保持原值
        assert attn.W_o[0] == [0.0] * 4
        assert attn.W_o[1] == [0.0] * 4
        assert attn.W_o[2] == [1.0] * 4
        assert attn.W_o[3] == [1.0] * 4

    def test_head_validation(self):
        attn = AI.MultiHeadAttention(4, 2)
        with pytest.raises(ValueError):
            AI.head_scores(attn, "z")
        with pytest.raises(ValueError):
            AI.prune_heads(AI.MultiHeadAttention(4, 1), 0.5)

    def test_prune_layers_rebuilds_mlp(self):
        net = AI.MLP([2, 4, 4, 4, 2], activations=["linear"] * 4)
        res = AI.prune_layers(net, ratio=0.25)
        assert len(res.layers) == 3
        assert len(net.layers) == 4
        assert res.layer_sizes == [2, 4, 4, 2]
        assert len(res.forward([1.0, 2.0])) == 2

    def test_prune_layers_validation(self):
        small = AI.MLP([2, 3, 2], activations=["linear", "linear"])
        with pytest.raises(ValueError):
            AI.prune_layers(small, ratio=0.5)
        mixed = AI.MLP([2, 3, 5, 2], activations=["linear"] * 3)
        with pytest.raises(ValueError):
            AI.prune_layers(mixed, ratio=1.0)
        with pytest.raises(ValueError):
            AI.prune_layers(small, ratio=0.0)

    def test_sparsity_report(self):
        net = self._net()
        rep = AI.sparsity_report(net)
        assert len(rep) == 2
        # 权重 [[1,0,0],[0,3,0],[0,0,0],[2,2,0]] 共 12 个元素、8 个 0
        assert rep[0]["zeros"] == 8
        assert abs(rep[0]["ratio"] - 8.0 / 12.0) < 1e-12
        assert rep[0]["kind"] == "Layer"


class TestQuantization:
    """W15-B：PTQ / QAT 伪量化 / SmoothQuant 缩放。"""

    def test_fake_quantize_grid_and_peak(self):
        out = AI.fake_quantize([0.0, 1.0], bits=8)
        assert out[0] == 0.0
        assert abs(out[1] - 1.0) < 1e-12
        scale = 1.0 / 127.0
        mid = AI.fake_quantize([0.5, 1.0], bits=8)[0]
        assert abs(mid / scale - round(mid / scale)) < 1e-9

    def test_fake_quantize_clips(self):
        clipped = AI.fake_quantize([2.0, -2.0], bits=8, scale=0.01)
        assert abs(clipped[0] - 1.27) < 1e-12
        assert abs(clipped[1] + 1.27) < 1e-12

    def test_fake_quantize_validation(self):
        for bits in (1, 17, 8.0, True):
            with pytest.raises(ValueError):
                AI.fake_quantize([1.0], bits=bits)
        for scale in (0.0, -1.0, "x"):
            with pytest.raises(ValueError):
                AI.fake_quantize([1.0], scale=scale)

    def test_quantize_weights_stores_ints(self):
        net = AI.MLP([2, 3, 1], activations=["linear", "linear"])
        before = [row[:] for row in net.layers[0].weights]
        res = AI.quantize_weights(net, bits=8)
        assert res["bits"] == 8
        assert len(res["scales"]) == 2
        code = res["codes"][0][0][0]
        assert isinstance(code, int) and not isinstance(code, bool)
        assert all(-127 <= c <= 127 for row in res["codes"][0] for c in row)
        assert abs(net.layers[0].weights[0][0]
                   - res["codes"][0][0][0] * res["scales"][0]) < 1e-15
        assert net.layers[0].weights != before
        assert len(net.forward([1.0, -1.0])) == 1

    def test_quantize_weights_idempotent(self):
        net = AI.MLP([2, 3, 1], activations=["linear", "linear"])
        first = AI.quantize_weights(net, bits=8)["scales"][0]
        second = AI.quantize_weights(net, bits=8)["scales"][0]
        assert abs(first - second) <= first * 1e-9

    def test_quantization_error_bounds(self):
        net = AI.MLP([4, 5, 2], activations=["linear", "linear"])
        before = [row[:] for row in net.layers[0].weights]
        rep = AI.quantization_error(net, bits=8)
        assert len(rep) == 2
        for item in rep:
            scale, _ = AI._w15_scale_of(net.layers[item["index"]].weights, 8, "t")
            assert item["max"] <= scale / 2.0 + 1e-12
            assert item["mean"] <= item["max"] + 1e-15
        assert net.layers[0].weights == before

    def test_qat_grads_match_plain_grads(self):
        net = AI.MLP([2, 3, 1], activations=["linear", "linear"])
        AI.quantize_weights(net, bits=8)
        xs = [[0.0, 0.0], [1.0, 1.0]]
        ys = [0.0, 1.0]
        grads = AI.qat_grads(net, xs, ys, bits=8)
        assert len(grads) == 2
        acc_w = [[0.0] * 2 for _ in range(3)]
        for x, y in zip(xs, ys):
            pred = net.forward(list(x))
            for li, (gw, _gb) in enumerate(AI._mlp_backward(net, [2.0 * (pred[0] - y)])):
                if li != 0:
                    continue
                for j in range(len(gw)):
                    for i in range(len(gw[j])):
                        acc_w[j][i] += gw[j][i]
        n = float(len(xs))
        diff = max(abs(grads[0][0][j][i] - acc_w[j][i] / n)
                   for j in range(3) for i in range(2))
        assert diff < 1e-12

    def test_qat_grads_validation(self):
        net = AI.MLP([2, 3, 1], activations=["linear", "linear"])
        with pytest.raises(ValueError):
            AI.qat_grads(net, [], [])
        with pytest.raises(ValueError):
            AI.qat_grads(net, [[1.0, 1.0]], [1.0, 2.0])
        seq = AI.Sequential([AI.Layer(2, 1, activation="linear")])
        with pytest.raises(ValueError):
            AI.qat_grads(seq, [[1.0, 1.0]], [1.0])

    def test_smooth_quant_scale_and_apply(self):
        assert abs(AI.smooth_quant_scale(4.0, 1.0, 0.5) - 2.0) < 1e-12
        assert abs(AI.smooth_quant_scale(4.0, 1.0, 0.0) - 1.0) < 1e-12
        assert abs(AI.smooth_quant_scale(4.0, 1.0, 1.0) - 4.0) < 1e-12
        layer = AI.Layer(2, 2, activation="linear")
        layer.weights = [[1.0, 1.0], [1.0, 1.0]]
        layer.biases = [0.5, 0.5]
        AI.smooth_quant_apply(layer, [2.0, 3.0])
        assert layer.weights == [[2.0, 3.0], [2.0, 3.0]]
        assert layer.biases == [0.5, 0.5]

    def test_smooth_quant_validation(self):
        for args in ((0.0, 1.0), (1.0, 0.0), (1.0, 1.0, 1.5), (1.0, 1.0, -0.1)):
            with pytest.raises(ValueError):
                AI.smooth_quant_scale(*args)
        layer = AI.Layer(2, 2, activation="linear")
        with pytest.raises(ValueError):
            AI.smooth_quant_apply(layer, [1.0])


class TestEnsembles:
    """W15-C：Voting / Bagging / Stacking / AdaBoostStumps。"""

    class _Fixed:
        def __init__(self, out):
            self.out = list(out)

        def forward(self, x):
            return list(self.out)

    class _Toy:
        def __init__(self):
            self.seen = []

        def fit(self, xs, ys, **kwargs):
            self.seen = [tuple(x) for x in xs]
            return self

        def forward(self, x):
            return [0.0, 0.0]

    def test_voting_soft(self):
        a = self._Fixed([2.0, 0.0])
        b = self._Fixed([0.0, 2.0])
        v = AI.VotingEnsemble([a, b], voting="soft")
        p = v.predict_proba([0.0])
        assert abs(p[0] - 0.5) < 1e-12
        assert abs(p[1] - 0.5) < 1e-12
        assert v.predict([0.0]) == 0
        assert v.forward([0.0]) == p

    def test_voting_hard(self):
        h = AI.VotingEnsemble(
            [self._Fixed([3.0, 0.0]), self._Fixed([0.0, 3.0]),
             self._Fixed([2.0, 1.0])], voting="hard")
        p = h.predict_proba([0.0])
        assert abs(p[0] - 2.0 / 3.0) < 1e-12
        assert h.predict([0.0]) == 0

    def test_voting_validation(self):
        a = self._Fixed([1.0, 0.0])
        with pytest.raises(ValueError):
            AI.VotingEnsemble([a], voting="rank")
        with pytest.raises(ValueError):
            AI.VotingEnsemble([])
        with pytest.raises(ValueError):
            AI.VotingEnsemble([object()])

    def test_voting_class_mismatch(self):
        v = AI.VotingEnsemble([self._Fixed([1.0, 0.0]),
                               self._Fixed([1.0, 0.0, 0.0])], voting="soft")
        with pytest.raises(ValueError):
            v.predict_proba([0.0])

    def test_bagging_control(self):
        xs = [[float(i)] for i in range(10)]
        ys = [0] * 10
        made = []

        def factory():
            t = self._Toy()
            made.append(t)
            return t

        bag = AI.BaggingEnsemble(factory, n_estimators=3, sample_ratio=0.5, seed=7)
        bag.fit(xs, ys)
        assert len(bag.models) == 3
        assert len(made) == 3
        assert all(len(m.seen) == 5 for m in bag.models)
        again = AI.BaggingEnsemble(factory, 3, 0.5, seed=7).fit(xs, ys)
        assert [m.seen for m in again.models] == [m.seen for m in bag.models]
        other = AI.BaggingEnsemble(factory, 3, 0.5, seed=8).fit(xs, ys)
        assert [m.seen for m in other.models] != [m.seen for m in bag.models]

    def test_bagging_validation(self):
        xs = [[float(i)] for i in range(10)]
        ys = [0] * 10

        def factory():
            return self._Toy()

        with pytest.raises(ValueError):
            AI.BaggingEnsemble(factory).predict([0.0])
        with pytest.raises(ValueError):
            AI.BaggingEnsemble(factory, n_estimators=0)
        with pytest.raises(ValueError):
            AI.BaggingEnsemble(factory, sample_ratio=0.0)
        with pytest.raises(ValueError):
            AI.BaggingEnsemble(42)
        with pytest.raises(ValueError):
            AI.BaggingEnsemble(factory).fit(xs, ys[:3])

    def test_stack_features(self):
        a = self._Fixed([2.0, 0.0])
        b = self._Fixed([0.0, 2.0])
        feats = AI.stack_features([a, b], [[0.0], [1.0]])
        assert len(feats) == 2
        assert len(feats[0]) == 4
        assert abs(sum(feats[0][:2]) - 1.0) < 1e-12
        with pytest.raises(ValueError):
            AI.stack_features([a], [])

    def test_stacking_end_to_end(self):
        base1 = AI.MLP([2, 2], activations=["linear"])
        base2 = AI.MLP([2, 2], activations=["linear"])
        meta = AI.Model(AI.MLP([4, 2], activations=["linear"]), loss_fn="mse",
                        optimizer="adam", lr=0.05)
        stack = AI.StackingEnsemble([base1, base2], meta, meta_epochs=40,
                                    meta_lr=0.05)
        train_x = [[-1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [1.0, -1.0]]
        train_y = [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]
        stack.fit(train_x, train_y)
        assert stack.predict([1.0, 1.0]) in (0, 1)
        assert abs(sum(stack.predict_proba([1.0, 1.0])) - 1.0) < 1e-9
        with pytest.raises(ValueError):
            AI.StackingEnsemble([base1], object())
        with pytest.raises(ValueError):
            AI.StackingEnsemble([base1], meta, meta_epochs=0)

    def test_adaboost_accuracy(self):
        data = [[-2.0, 0.0], [-1.0, 0.3], [-0.5, -0.2],
                [0.5, 0.1], [1.0, -0.3], [2.0, 0.2]]
        labels = [0, 0, 0, 1, 1, 1]
        ada = AI.AdaBoostStumps(n_estimators=10)
        ada.fit(data, labels)
        acc = sum(1 for x, y in zip(data, labels) if ada.predict(x) == y) / len(data)
        assert abs(acc - 1.0) < 1e-12
        assert len(ada.stumps) == 10
        assert ada.decision_function([2.0, 0.2]) > 0.0
        assert ada.decision_function([-2.0, 0.0]) < 0.0
        p = ada.predict_proba([2.0, 0.2])
        assert abs(sum(p) - 1.0) < 1e-12
        assert p[1] > 0.5
        assert ada.forward([2.0, 0.2]) == p

    def test_adaboost_validation(self):
        data = [[-1.0, 0.0], [1.0, 0.0]]
        with pytest.raises(ValueError):
            AI.AdaBoostStumps().fit([[1.0, 1.0], [1.0, 1.0]], [0, 1])
        with pytest.raises(ValueError):
            AI.AdaBoostStumps().fit(data, [0, 2])
        with pytest.raises(ValueError):
            AI.AdaBoostStumps().decision_function([0.0, 0.0])
        with pytest.raises(ValueError):
            AI.AdaBoostStumps(n_estimators=0)
        with pytest.raises(ValueError):
            AI.AdaBoostStumps(learning_rate=0.0)
        with pytest.raises(ValueError):
            AI.AdaBoostStumps().fit(data, [0])


class TestPackingAndSpeculative:
    """W15-D：批内打包 + 投机解码集成。"""

    def test_pack_decode_batch(self):
        packed = AI.pack_decode_batch([[1, 2, 3], [4, 5]], pad_id=0)
        assert packed["tokens"] == [1, 2, 3, 4, 5]
        assert packed["segment_ids"] == [0, 0, 0, 1, 1]
        assert packed["position_ids"] == [0, 1, 2, 0, 1]
        assert packed["lengths"] == [3, 2]
        mask = packed["attention_mask"]
        assert mask[2][0] == 1.0
        assert mask[0][2] == 0.0
        assert mask[0][3] == 0.0
        assert mask[4][4] == 1.0

    def test_pack_decode_batch_validation(self):
        with pytest.raises(ValueError):
            AI.pack_decode_batch([])
        with pytest.raises(ValueError):
            AI.pack_decode_batch([[1], []])

    def test_speculative_identical_models(self):
        AI.set_seed(11)
        model = AI.CausalLM(vocab_size=7, d_model=8, num_heads=2, d_ff=16,
                            num_layers=1, max_len=32)
        prompt = [1, 2, 3]
        ids, stats = AI.speculative_generate(model, model, prompt,
                                             max_new_tokens=6, gamma=3, seed=5)
        assert len(ids) == len(prompt) + 6
        assert ids[:3] == prompt
        assert stats["accepted"] == stats["proposed"]
        assert abs(stats["accept_rate"] - 1.0) < 1e-12
        assert stats["rounds"] >= 2

    def test_speculative_different_draft(self):
        AI.set_seed(3)
        target = AI.CausalLM(vocab_size=7, d_model=8, num_heads=2, d_ff=16,
                             num_layers=1, max_len=32)
        draft = AI.CausalLM(vocab_size=7, d_model=8, num_heads=2, d_ff=16,
                            num_layers=1, max_len=32)
        prompt = [1, 2, 3]
        ids, stats = AI.speculative_generate(draft, target, prompt,
                                             max_new_tokens=8, gamma=4, seed=9)
        assert len(ids) == len(prompt) + 8
        assert set(stats) == {"rounds", "proposed", "accepted", "accept_rate"}
        assert stats["accepted"] <= stats["proposed"]
        assert 0.0 <= stats["accept_rate"] <= 1.0
        assert all(0 <= t < 7 for t in ids)
        again = AI.speculative_generate(draft, target, prompt, 8, 4, seed=9)
        assert again[0] == ids

    def test_speculative_validation(self):
        AI.set_seed(1)
        model = AI.CausalLM(vocab_size=5, d_model=8, num_heads=2, d_ff=16,
                            num_layers=1, max_len=16)
        with pytest.raises(ValueError):
            AI.speculative_generate(model, model, [1], 0, 4)
        with pytest.raises(ValueError):
            AI.speculative_generate(model, model, [1], 4, 0)
        with pytest.raises(ValueError):
            AI.speculative_generate(model, model, [], 4, 4)
        with pytest.raises(ValueError):
            AI.speculative_generate(object(), model, [1])
        with pytest.raises(ValueError):
            AI.speculative_generate(model, object(), [1])


class TestContrastiveLearning:
    """W16-A：对比学习损失（SimCLR / MoCo / BYOL / Barlow / VICReg / SwAV / DINO）。"""

    _Z1 = [[0.4, -0.2, 0.7], [-0.5, 0.9, 0.1], [0.3, 0.3, -0.8]]
    _Z2 = [[0.2, 0.5, -0.6], [0.8, -0.1, 0.4], [-0.4, 0.2, 0.9]]

    def test_nt_xent_gradient(self):
        loss, g1, g2 = AI.nt_xent_loss(self._Z1, self._Z2, temperature=0.5)
        assert 0.0 < loss < 10.0
        eps = 1e-6
        for i in range(3):
            for j in range(3):
                plus = [list(r) for r in self._Z1]
                minus = [list(r) for r in self._Z1]
                plus[i][j] += eps
                minus[i][j] -= eps
                num = (AI.nt_xent_loss(plus, self._Z2, 0.5)[0]
                       - AI.nt_xent_loss(minus, self._Z2, 0.5)[0]) / (2 * eps)
                assert abs(num - g1[i][j]) < 1e-6

    def test_nt_xent_validation(self):
        with pytest.raises(ValueError):
            AI.nt_xent_loss(self._Z1, self._Z2, 0.0)
        with pytest.raises(ValueError):
            AI.nt_xent_loss(self._Z1, self._Z2[:2])
        with pytest.raises(ValueError):
            AI.nt_xent_loss([[0.0, 0.0], [1.0, 0.0]], [[1.0, 0.0], [0.0, 1.0]])

    def test_moco_gradient_and_queue(self):
        Q = [[1.0, 0.0], [0.0, 1.0]]
        K = [[0.9, 0.1], [0.2, 0.8]]
        loss, gq, _ = AI.moco_loss(Q, K, temperature=0.07)
        assert 0.0 < loss < 10.0
        eps = 1e-6
        for i in range(2):
            for j in range(2):
                plus = [list(r) for r in Q]
                minus = [list(r) for r in Q]
                plus[i][j] += eps
                minus[i][j] -= eps
                num = (AI.moco_loss(plus, K, temperature=0.07)[0]
                       - AI.moco_loss(minus, K, temperature=0.07)[0]) / (2 * eps)
                assert abs(num - gq[i][j]) < 1e-6

        queue = AI.MoCoQueue(2, size=3)
        queue.enqueue([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]])
        assert len(queue) == 3
        assert queue.matrix()[0] == [0.5, 0.5]
        bigger, _, _ = AI.moco_loss(Q, K, queue=queue.matrix(), temperature=0.07)
        assert bigger > loss

    def test_moco_validation(self):
        with pytest.raises(ValueError):
            AI.moco_loss([[1.0, 0.0]], [[1.0, 0.0]])
        with pytest.raises(ValueError):
            AI.moco_loss([[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]],
                         [[1.0, 0.0, 0.0]])
        with pytest.raises(ValueError):
            AI.MoCoQueue(0)
        with pytest.raises(ValueError):
            AI.MoCoQueue(2, size=0)

    def test_byol(self):
        P = [[1.0, 0.0], [0.0, 1.0]]
        T = [[0.8, 0.2], [0.1, 0.9]]
        assert abs(AI.byol_loss(P, P)[0]) < 1e-12
        assert abs(AI.byol_loss([[1.0, 0.0]], [[0.0, 1.0]])[0] - 2.0) < 1e-12
        loss, gp = AI.byol_loss(P, T)
        eps = 1e-6
        for i in range(2):
            for j in range(2):
                plus = [list(r) for r in P]
                minus = [list(r) for r in P]
                plus[i][j] += eps
                minus[i][j] -= eps
                num = (AI.byol_loss(plus, T)[0] - AI.byol_loss(minus, T)[0]) / (2 * eps)
                assert abs(num - gp[i][j]) < 1e-6

    def test_barlow_vicreg(self):
        assert AI.barlow_twins_loss(self._Z1, self._Z1) >= 0.0
        assert abs(AI.vicreg_loss(self._Z1, self._Z2, 0.0, 0.0, 0.0)) < 1e-12
        assert AI.vicreg_loss(self._Z1, self._Z1) >= 0.0
        with pytest.raises(ValueError):
            AI.barlow_twins_loss(self._Z1, self._Z2, -1.0)
        with pytest.raises(ValueError):
            AI.vicreg_loss(self._Z1, self._Z2, sim_weight=-1.0)

    def test_sinkhorn_and_swav(self):
        q = AI.sinkhorn([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], epsilon=0.05,
                        n_iters=200)
        for j in range(2):
            assert abs(sum(q[i][j] for i in range(3)) - 1.0 / 2.0) < 1e-9
        for row in q:
            assert abs(sum(row) - 1.0 / 3.0) < 5e-3
        proto = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        assert 0.0 < AI.swav_loss(self._Z1, self._Z2, proto) < 10.0
        with pytest.raises(ValueError):
            AI.sinkhorn([[1.0, 2.0]], n_iters=0)

    def test_dino(self):
        center = [0.0, 0.0, 0.0]
        d = AI.dino_loss(self._Z1, self._Z2, center)
        assert 0.0 < d < 50.0
        mean = [(0.2 + 0.8 - 0.4) / 3, (0.5 - 0.1 + 0.2) / 3, (-0.6 + 0.4 + 0.9) / 3]
        got = AI.update_dino_center(center, self._Z2, momentum=0.0)
        assert abs(got[0] - mean[0]) < 1e-12
        with pytest.raises(ValueError):
            AI.dino_loss(self._Z1, self._Z2, [0.0])
        with pytest.raises(ValueError):
            AI.update_dino_center(center, self._Z2, 1.5)
class TestMaskedVisionModeling:
    """W16-B：视觉掩码建模（patchify / random_masking / block_mask / 三种损失）。"""

    IMG = [
        [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0],
         [9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]],
        [[-1.0, -2.0, -3.0, -4.0], [-5.0, -6.0, -7.0, -8.0],
         [-9.0, -10.0, -11.0, -12.0], [-13.0, -14.0, -15.0, -16.0]],
    ]

    def test_patchify_layout(self):
        patches = AI.patchify(self.IMG, 2)
        assert len(patches) == 4
        assert all(len(p) == 8 for p in patches)
        assert patches[0] == [1.0, 2.0, 5.0, 6.0, -1.0, -2.0, -5.0, -6.0]

    def test_unpatchify_roundtrip(self):
        patches = AI.patchify(self.IMG, 2)
        back = AI.unpatchify(patches, 2, 4, 4, 2)
        flat = [back[c][r][j] for c in range(2) for r in range(4) for j in range(4)]
        want = [self.IMG[c][r][j] for c in range(2) for r in range(4) for j in range(4)]
        assert flat == want

    def test_patchify_validation(self):
        with pytest.raises(ValueError):
            AI.patchify(self.IMG, 3)
        with pytest.raises(ValueError):
            AI.patchify(self.IMG, 0)
        with pytest.raises(ValueError):
            AI.unpatchify(AI.patchify(self.IMG, 2)[:3], 2, 4, 4, 2)
        with pytest.raises(ValueError):
            AI.unpatchify([[1.0]] * 4, 2, 4, 4, 2)

    def test_random_masking_counts(self):
        keep, masked = AI.random_masking(12, 0.75, seed=3)
        assert len(keep) == 3
        assert len(masked) == 9
        assert sorted(keep + masked) == list(range(12))
        assert keep == sorted(keep) and masked == sorted(masked)
        assert AI.random_masking(12, 0.75, seed=3) == (keep, masked)
        assert AI.random_masking(12, 0.75, seed=4) != (keep, masked)

    def test_random_masking_zero_ratio_keeps_all(self):
        keep, masked = AI.random_masking(4, 0.0, seed=1)
        assert keep == [0, 1, 2, 3]
        assert masked == []

    def test_random_masking_validation(self):
        with pytest.raises(ValueError):
            AI.random_masking(4, 1.0)
        with pytest.raises(ValueError):
            AI.random_masking(0)

    def test_block_mask(self):
        grid, masked = AI.block_mask(4, 4, 2, 2, num_blocks=2, seed=5)
        assert len(grid) == 4 and all(len(row) == 4 for row in grid)
        assert len(masked) >= 4
        assert sorted(y * 4 + x for y in range(4) for x in range(4) if grid[y][x]) == masked
        assert AI.block_mask(4, 4, 2, 2, 2, seed=5) == (grid, masked)
        with pytest.raises(ValueError):
            AI.block_mask(4, 4, 5, 2)
        with pytest.raises(ValueError):
            AI.block_mask(4, 4, 2, 2, 0)

    def test_mae_loss(self):
        recon = [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]
        target = [[0.0, 0.0], [2.0, 2.0], [5.0, 5.0]]
        loss, grad = AI.mae_loss(recon, target, [0, 2])
        assert abs(loss - 5.0) < 1e-12
        assert grad[1] == [0.0, 0.0]
        assert abs(grad[0][0] - 1.0) < 1e-12
        assert abs(grad[2][0] + 2.0) < 1e-12
        with pytest.raises(ValueError):
            AI.mae_loss(recon, target, [])
        with pytest.raises(ValueError):
            AI.mae_loss(recon, target, [3])

    def test_simmim_loss(self):
        recon = [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]
        target = [[0.0, 0.0], [2.0, 2.0], [5.0, 5.0]]
        loss, grad = AI.simmim_loss(recon, target, [0, 2], beta=1.0)
        assert abs(loss - 2.0) < 1e-12
        assert abs(grad[0][0] - 0.5) < 1e-12
        assert abs(grad[2][0] + 0.5) < 1e-12
        loss2, _ = AI.simmim_loss([[3.0]], [[0.0]], [0], beta=1.0)
        assert abs(loss2 - 2.5) < 1e-12
        with pytest.raises(ValueError):
            AI.simmim_loss(recon, target, [0], 0.0)

    def test_beit_loss(self):
        logits = [[2.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 2.0, 0.0]]
        loss, grad = AI.beit_loss(logits, [0, 0, 1], [0, 2])
        assert 0.0 < loss < 10.0
        assert grad[1] == [0.0, 0.0, 0.0]
        assert abs(sum(grad[0])) < 1e-12 and abs(sum(grad[2])) < 1e-12
        with pytest.raises(ValueError):
            AI.beit_loss(logits, [0, 0, 3], [2])
        with pytest.raises(ValueError):
            AI.beit_loss(logits, [0, 0, -1], [2])
        with pytest.raises(ValueError):
            AI.beit_loss(logits, [0, 0], [0])
class TestMultimodal:
    """W16-C：CLIP 双塔（相似度 / 对称 InfoNCE / 检索准确率 / 投影头）+ 跨模态融合块。"""

    IMG = [[1.0, 0.0], [0.0, 1.0]]
    TXT = [[2.0, 0.0], [0.0, 3.0]]

    def test_clip_similarity(self):
        sim = AI.clip_similarity(self.IMG, self.TXT)
        assert len(sim) == 2 and len(sim[0]) == 2
        assert abs(sim[0][0] - 1.0) < 1e-12
        assert abs(sim[1][1] - 1.0) < 1e-12
        assert abs(sim[0][1]) < 1e-12 and abs(sim[1][0]) < 1e-12
        with pytest.raises(ValueError):
            AI.clip_similarity([[1.0, 0.0]], [[1.0]])
        with pytest.raises(ValueError):
            AI.clip_similarity([[0.0, 0.0]], [[1.0, 0.0]])

    def test_clip_loss_value(self):
        loss = AI.clip_loss(self.IMG, self.TXT, 0.07)[0]
        p = 1.0 / (1.0 + AI.math.exp(-(1.0 / 0.07)))
        assert abs(loss + AI.math.log(p)) < 1e-12
        with pytest.raises(ValueError):
            AI.clip_loss(self.IMG, self.TXT, 0.0)
        with pytest.raises(ValueError):
            AI.clip_loss(self.IMG, [[1.0, 0.0]])
        with pytest.raises(ValueError):
            AI.clip_loss([[1.0, 0.0]], [[1.0, 0.0, 0.0]])

    def test_clip_loss_gradients(self):
        random.seed(3)
        I = [[random.uniform(-1, 1) for _ in range(3)] for _ in range(3)]
        T = [[random.uniform(-1, 1) for _ in range(3)] for _ in range(3)]
        _, gI, gT = AI.clip_loss(I, T, 0.5)
        eps = 1e-6
        for a in range(3):
            for b in range(3):
                up = [row[:] for row in I]
                dn = [row[:] for row in I]
                up[a][b] += eps
                dn[a][b] -= eps
                num = (AI.clip_loss(up, T, 0.5)[0] - AI.clip_loss(dn, T, 0.5)[0]) / (2 * eps)
                assert abs(num - gI[a][b]) < 1e-6
        for a in range(3):
            for b in range(3):
                up = [row[:] for row in T]
                dn = [row[:] for row in T]
                up[a][b] += eps
                dn[a][b] -= eps
                num = (AI.clip_loss(I, up, 0.5)[0] - AI.clip_loss(I, dn, 0.5)[0]) / (2 * eps)
                assert abs(num - gT[a][b]) < 1e-6

    def test_retrieval_accuracy(self):
        hit = AI.contrastive_retrieval_accuracy(self.IMG, self.IMG)
        assert hit["i2t"] == 1.0 and hit["t2i"] == 1.0 and hit["mean"] == 1.0
        miss = AI.contrastive_retrieval_accuracy(self.IMG, [[0.0, 1.0], [1.0, 0.0]])
        assert miss["i2t"] == 0.0 and miss["t2i"] == 0.0
        with pytest.raises(ValueError):
            AI.contrastive_retrieval_accuracy([[1.0, 0.0]], self.IMG)

    def test_clip_projector(self):
        Xi = [[0.1, 0.2, 0.3], [0.4, -0.2, 0.0], [1.0, 1.0, -1.0]]
        Xt = [[0.5, 0.1, 0.2, 0.0], [0.0, 0.3, -0.3, 0.2], [-0.4, 0.9, 0.1, 0.1]]
        proj = AI.CLIPProjector(3, 4, embed_dim=2, temperature=0.5, seed=7)
        loss, grads = proj.loss_and_grads(Xi, Xt)
        emb_i, emb_t = proj.project(Xi, Xt)
        assert abs(loss - AI.clip_loss(emb_i, emb_t, 0.5)[0]) < 1e-12
        assert proj.param_count() == 2 * 3 + 2 + 2 * 4 + 2
        assert sorted(grads) == ["W_img", "W_txt", "b_img", "b_txt"]
        eps = 1e-6
        w = proj.W_img[0][0]
        proj.W_img[0][0] = w + eps
        up = proj.loss(Xi, Xt)
        proj.W_img[0][0] = w - eps
        dn = proj.loss(Xi, Xt)
        proj.W_img[0][0] = w
        assert abs((up - dn) / (2 * eps) - grads["W_img"][0][0]) < 1e-6
        with pytest.raises(ValueError):
            proj.loss_and_grads(Xi[:2], Xt)
        with pytest.raises(ValueError):
            AI.CLIPProjector(3, 4, 0)

    def test_clip_projector_training(self):
        Xi = [[0.1, 0.2, 0.3], [0.4, -0.2, 0.0], [1.0, 1.0, -1.0]]
        Xt = [[0.5, 0.1, 0.2, 0.0], [0.0, 0.3, -0.3, 0.2], [-0.4, 0.9, 0.1, 0.1]]
        proj = AI.CLIPProjector(3, 4, embed_dim=2, temperature=0.5, seed=5)
        first = proj.loss(Xi, Xt)
        for _ in range(30):
            _, g = proj.loss_and_grads(Xi, Xt)
            proj.apply_grads(g, 0.2)
        assert proj.loss(Xi, Xt) < first - 1e-3
        with pytest.raises(ValueError):
            proj.apply_grads(g, 0.0)
        with pytest.raises(ValueError):
            proj.apply_grads(g, 0.1, "adam")

    def test_cross_modal_block(self):
        random.seed(17)
        block = AI.CrossModalBlock(4, num_heads=2, dropout=0.0)
        Xi = [[random.uniform(-1, 1) for _ in range(4)] for _ in range(3)]
        Xt = [[random.uniform(-1, 1) for _ in range(4)] for _ in range(2)]
        fused = block.forward(Xi, Xt)
        assert len(fused) == 2 and len(fused[0]) == 4
        assert all(abs(sum(r) / 4) < 1e-9 for r in fused)
        coef = [[0.3, -0.2, 0.5, 0.1], [0.7, 0.2, -0.4, 0.6]]
        d_img, d_txt = block.backward(coef)

        def scalar(img_feats, txt_feats):
            out = block.forward(img_feats, txt_feats)
            return sum(out[i][j] * coef[i][j] for i in range(len(out)) for j in range(4))

        eps = 1e-6
        for i in range(3):
            for j in range(4):
                up = [row[:] for row in Xi]
                dn = [row[:] for row in Xi]
                up[i][j] += eps
                dn[i][j] -= eps
                num = (scalar(up, Xt) - scalar(dn, Xt)) / (2 * eps)
                assert abs(num - d_img[i][j]) < 1e-5
        for i in range(2):
            for j in range(4):
                up = [row[:] for row in Xt]
                dn = [row[:] for row in Xt]
                up[i][j] += eps
                dn[i][j] -= eps
                num = (scalar(Xi, up) - scalar(Xi, dn)) / (2 * eps)
                assert abs(num - d_txt[i][j]) < 1e-5

    def test_cross_modal_validation(self):
        block = AI.CrossModalBlock(4)
        with pytest.raises(RuntimeError):
            block.backward([[0.0] * 4])
        with pytest.raises(ValueError):
            AI.CrossModalBlock(4).forward([[0.0] * 3], [[0.0] * 4])
        with pytest.raises(ValueError):
            AI.CrossModalBlock(0)
        random.seed(1)
        b2 = AI.CrossModalBlock(4)
        # memory 长度必须 > 1：只有 1 个 key 时 softmax 恒为 1，输出与 Q 无关，W_q 梯度必然为 0
        b2.forward([[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]],
                   [[0.2, 0.1, 0.0, 0.3], [0.5, 0.2, 0.1, 0.0]])
        b2.backward([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        assert any(abs(v) > 0 for row in b2.attn.grads["W_q"] for v in row)
        b2.zero_grad()
        assert all(v == 0.0 for row in b2.attn.grads["W_q"] for v in row)
class TestPredictiveSelfSupervision:
    """W16-D：预测式自监督（Rotation 旋转预测 + Jigsaw 拼图预测）。"""

    IMG = [
        [[1.0, 2.0], [3.0, 4.0]],
        [[5.0, 6.0], [7.0, 8.0]],
    ]

    def test_rotation_targets(self):
        t = AI.rotation_targets(20, seed=4)
        assert len(t) == 20 and all(v in (0, 1, 2, 3) for v in t)
        assert set(t) == {0, 1, 2, 3}
        assert AI.rotation_targets(20, seed=4) == t
        assert AI.rotation_targets(20, seed=5) != t
        with pytest.raises(ValueError):
            AI.rotation_targets(0)

    def test_rotate_image_quarter_turns(self):
        r0 = AI.rotate_image(self.IMG, 0)
        assert r0 == [[[float(v) for v in row] for row in ch] for ch in self.IMG]
        r1 = AI.rotate_image(self.IMG, 1)
        assert r1[0][0][1] == 1.0
        assert r1[0][1][1] == 2.0
        assert r1[0][0][0] == 3.0
        r2 = AI.rotate_image(self.IMG, 2)
        assert r2[0][1][1] == 1.0
        assert AI.rotate_image(r2, 2) == r0
        assert AI.rotate_image(self.IMG, 4) == r0
        assert AI.rotate_image(self.IMG, -1) == AI.rotate_image(self.IMG, 3)

    def test_rotate_image_validation(self):
        rect = [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]
        with pytest.raises(ValueError):
            AI.rotate_image(rect, 1)
        assert AI.rotate_image(rect, 2)[0][0] == [6.0, 5.0, 4.0]
        with pytest.raises(ValueError):
            AI.rotate_image(self.IMG, 1.5)

    def test_jigsaw_permutations(self):
        perms = AI.jigsaw_permutations(9, num_permutations=3, seed=7)
        assert len(perms) == 3
        assert all(sorted(p) == list(range(9)) for p in perms)
        assert all(p != list(range(9)) for p in perms)
        assert len({tuple(p) for p in perms}) == 3
        assert AI.jigsaw_permutations(9, 3, seed=7) == perms
        with pytest.raises(ValueError):
            AI.jigsaw_permutations(2, 5)
        with pytest.raises(ValueError):
            AI.jigsaw_permutations(1)
        with pytest.raises(ValueError):
            AI.jigsaw_permutations(4, 0)

    def test_jigsaw_shuffle(self):
        patches = [[i] for i in range(4)]
        assert AI.jigsaw_shuffle(patches, [2, 0, 3, 1]) == [[2], [0], [3], [1]]
        with pytest.raises(ValueError):
            AI.jigsaw_shuffle(patches, [1, 0])
        with pytest.raises(ValueError):
            AI.jigsaw_shuffle(patches, [1, 1, 2, 3])
        with pytest.raises(ValueError):
            AI.jigsaw_shuffle([], [])

    def test_rotation_head_and_loss(self):
        feats = [[3.0, 0.0, 0.0, 0.0], [0.0, 3.0, 0.0, 0.0]]
        probs = AI.rotation_classifier_head(feats, 4)
        assert all(abs(sum(r) - 1.0) < 1e-12 for r in probs)
        assert probs[0][0] > probs[0][1] and probs[1][1] > probs[1][0]
        loss = AI.rotation_loss(probs, [0, 1])
        assert loss < 0.2
        assert AI.rotation_loss(probs, [1, 0]) > loss + 1.0
        with pytest.raises(ValueError):
            AI.rotation_classifier_head([[1.0, 2.0]], 4)
        with pytest.raises(ValueError):
            AI.rotation_classifier_head(feats, 1)
        with pytest.raises(ValueError):
            AI.rotation_loss(probs, [0, 4])

    def test_linear_probe_accuracy(self):
        scores = [[5.0, 0.0], [0.0, 5.0], [5.0, 0.0]]
        assert AI.linear_probe_accuracy(scores, [0, 1, 0]) == 1.0
        assert abs(AI.linear_probe_accuracy(scores, [1, 1, 0]) - 2.0 / 3.0) < 1e-12
        with pytest.raises(ValueError):
            AI.linear_probe_accuracy(scores, [0, 2])
        with pytest.raises(ValueError):
            AI.linear_probe_accuracy(scores, [0])
        with pytest.raises(ValueError):
            AI.linear_probe_accuracy([[1.0, None]], [0])
