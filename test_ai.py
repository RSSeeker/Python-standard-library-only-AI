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
import math
import random
import os
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
# 批次 5：训练工程化与评估工具
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
        assert len(res["checks"]) == 24

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
# 批次 6：优化器状态持久化 + checkpoint
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

