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
        """不合法的激活函数名称应抛出 ValueError"""
        with pytest.raises(ValueError):
            Layer(1, 1, activation="sigmoid")

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
            Layer(1, 1, activation="sigmoid")

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
