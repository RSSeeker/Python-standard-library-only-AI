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
    generate_text_with_mlp,
    train, evaluate, save_model, load_model,
    _softmax_with_temp,
    # 新增高层 API
    Model, make_classifier, make_regressor, quick_train,
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
        dp.training = False
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
        bn.training = False
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

    def test_model_repr(self):
        """Model.__repr__"""
        mlp = MLP([3, 8, 2], activations=["leaky_relu", "softmax"])
        model = Model(mlp)
        r = repr(model)
        assert "classification" in r
        assert "Adam" in r


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
