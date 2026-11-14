"""tests/ 目录冒烟测试：快速确认核心公开 API 的可用性（约 12 项）。

与仓库根的 test_ai.py 不同，这里只做「能跑通」级联检查，不追求覆盖；
完整数值校验仍在 test_ai.py。CI 的裸 pytest 会同时收集两处。
"""

import math
import random

import AI


def _close(a, b, tol=1e-9):
    return abs(a - b) <= tol


def test_mlp_forward_backward():
    model = AI.MLP([2, 3, 1], ["relu", "linear"])
    out = model.forward([0.5, -0.25])
    assert len(out) == 1
    grads, dx = AI._mlp_backward_with_input(model, [1.0])
    assert len(grads) == 2 and len(dx) == 2


def test_adam_step_reduces_loss():
    model = AI.MLP([1, 1], ["linear"])
    opt = AI.Adam(model, lr=0.1)
    xs, ys = [[1.0], [2.0], [-1.0]], [[2.0], [4.0], [-2.0]]
    loss0 = None
    for _ in range(200):
        total, grads = 0.0, None
        for x, y in zip(xs, ys):
            pred = model.forward(x)[0]
            total += (pred - y[0]) ** 2 / len(xs)
            g, _ = AI._mlp_backward_with_input(model, [2.0 * (pred - y[0]) / len(xs)])
            if grads is None:
                grads = g
            else:
                for (gw_a, gb_a), (gw_b, gb_b) in zip(grads, g):
                    for i in range(len(gw_a)):
                        for j in range(len(gw_a[i])):
                            gw_a[i][j] += gw_b[i][j]
                    for i in range(len(gb_a)):
                        gb_a[i] += gb_b[i]
        if loss0 is None:
            loss0 = total
        opt.step(grads)
    assert total < loss0 * 0.01


def test_gumbel_softmax_straight_through():
    out = AI.gumbel_softmax([1.0, 2.0, 0.5, -1.0], temperature=0.5, hard=True, seed=1)
    assert len(out) == 4 and _close(sum(out), 1.0)
    assert sum(1 for v in out if v == 1.0) == 1


def test_sparsemax_simplex():
    probs = AI.sparsemax([0.5, 2.0, -1.0, 0.2])
    assert _close(sum(probs), 1.0) and all(v >= 0.0 for v in probs)
    assert probs[2] < 1e-12


def test_augmix_jsd_identical_is_zero():
    view = [1.0, 0.2, -0.5]
    assert _close(AI.augmix_jsd_loss([view, view], temperature=1.0), 0.0, 1e-12)


def test_consistency_coeffs_limits():
    coeffs = AI.consistency_model_coeffs(0.0)
    assert _close(coeffs["c_skip"], 1.0) and _close(coeffs["c_out"], 0.0)
    assert _close(coeffs["c_in"], 1.0 / math.sqrt(0.5 ** 2 + 0.0 ** 2))


def test_itransformer_forward_shape():
    model = AI.ITransformer(lookback=4, num_features=3, d_model=8, num_heads=2,
                            d_ff=16, num_layers=1, horizon=2, seed=1)
    series = [[0.1 * (j + 1) + 0.01 * t for t in range(4)] for j in range(3)]
    pred = model.forward(series)
    assert len(pred) == 3 and len(pred[0]) == 2


def test_ft_transformer_forward_shape():
    model = AI.FTTransformer(3, 2, d_model=8, num_heads=2, d_ff=16, num_layers=1, seed=1)
    logits = model.forward([0.5, -0.3, 1.2])
    assert len(logits) == 2


def test_tent_entropy_is_finite():
    model = AI.TENTClassifier(4, 2, seed=1)
    xs = [[1.0, 0.0, 0.5, -0.5], [0.0, 1.0, -0.2, 0.3]]
    ent = model.entropy(xs)
    assert math.isfinite(ent) and ent > 0.0


def test_scaled_cosine_attention_param_count():
    model = AI.ScaledCosineAttention(8, 2, seed=1)
    assert model.param_count() == 4 * (8 * 8 + 8) + 1
    out = model.forward([[0.1 * i for i in range(8)] for _ in range(2)])
    assert len(out) == 2 and len(out[0]) == 8


def test_dlinear_forward_shape():
    model = AI.DLinear(lookback=6, horizon=3, kernel_size=3, seed=1)
    x = [[0.1 * t + 0.2 * c for c in range(2)] for t in range(6)]
    pred = model.forward(x)
    assert len(pred) == 3 and len(pred[0]) == 2


def test_grn_initial_identity():
    model = AI.GRN(4)
    x = [[0.3, -0.2, 1.0, 0.5]]
    out = model.forward(x)
    assert all(_close(a, b) for a, b in zip(out[0], x[0]))
