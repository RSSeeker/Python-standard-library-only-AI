"""数值回归快照：把一组确定性计算的输出固定下来，CI 里检查是否发生漂移。

用法：
    python tools/numeric_regression.py --update   # 重新生成 tools/numeric_snapshot.json
    python tools/numeric_regression.py --check    # 检查与快照一致（CI 使用）

设计：只取**确定性**计算（显式设置权重 / 固定 seed / 无随机初始化），数值
四舍五入到 1e-10 后再比较，避免不同平台 math 库 1 ulp 差异造成假失败。
"""

import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import AI  # noqa: E402

SNAPSHOT = os.path.join(HERE, "numeric_snapshot.json")
DIGITS = 10


def _round(v):
    if isinstance(v, float):
        return round(v, DIGITS)
    if isinstance(v, (list, tuple)):
        return [_round(x) for x in v]
    if isinstance(v, dict):
        return {k: _round(x) for k, x in sorted(v.items())}
    return v


def build_snapshot():
    values = {}

    # 1) MLP 固定权重前向
    mlp = AI.MLP([3, 4, 1], ["relu", "linear"])
    mlp.layers[0].weights = [[0.5, -0.2, 0.1], [0.3, 0.4, -0.5],
                             [-0.1, 0.2, 0.6], [0.7, -0.3, 0.2]]
    mlp.layers[0].biases = [0.1, -0.1, 0.0, 0.2]
    mlp.layers[1].weights = [[0.6, -0.4, 0.2, 0.5]]
    mlp.layers[1].biases = [-0.1]
    values["mlp_forward"] = mlp.forward([0.3, -0.5, 0.8])

    # 2) 交叉熵
    values["cross_entropy"] = AI.cross_entropy_loss([1.0, 2.0, 0.5], 1)

    # 3) Gumbel-Softmax（固定 seed）
    values["gumbel_softmax"] = AI.gumbel_softmax([0.5, 1.5, -0.5], temperature=0.7,
                                                 hard=True, seed=42)

    # 4) Sparsemax / Entmax
    values["sparsemax"] = AI.sparsemax([0.5, 2.0, -1.0, 0.2])
    values["entmax_1.5"] = AI.entmax([0.5, 2.0, -1.0, 0.2], alpha=1.5)

    # 5) 度量学习损失
    values["supcon_loss"] = AI.supcon_loss(
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]],
        [0, 0, 1, 1], temperature=0.2)
    values["cosface_loss"] = AI.cosface_loss(
        [[0.8, 0.2], [0.3, 0.7]], [0, 1], margin=0.3, scale=16.0)

    # 6) AugMix JSD
    view = [0.4, 1.2, -0.3, 0.0]
    values["augmix_jsd"] = AI.augmix_jsd_loss([view, view, view])

    # 7) Consistency 系数
    values["consistency_coeffs"] = AI.consistency_model_coeffs(1.5, sigma_data=0.5)

    # 8) 注意力：固定权重 MHA
    att = AI.MultiHeadAttention(4, 2, dropout=0.0, causal=False)
    att.forward([[0.1, 0.2, 0.3, 0.4], [0.5, -0.1, 0.0, 0.2]])  # 触发参数惰性创建
    for name in ("W_q", "W_k", "W_v", "W_o"):
        setattr(att, name, [[0.1 * (i + 1) - 0.05 * j for j in range(4)]
                            for i in range(4)])
    for name in ("b_q", "b_k", "b_v", "b_o"):
        setattr(att, name, [0.0] * 4)
    values["mha_forward"] = att.forward([[0.1, 0.2, 0.3, 0.4],
                                         [0.5, -0.1, 0.0, 0.2],
                                         [0.0, 0.3, -0.2, 0.1]])

    # 9) 量化 / NF4
    values["nf4_quantize"] = AI.nf4_quantize([0.1234, -1.5, 0.0, 2.75])

    # 10) 缩放余弦注意力（固定 seed）
    sca = AI.ScaledCosineAttention(4, 1, logit_scale=0.5, seed=7)
    values["scaled_cosine_attn"] = sca.forward([[0.2, -0.1, 0.4, 0.0],
                                                [0.0, 0.5, -0.3, 0.1]])

    return _round(values)


def _read_snapshot():
    with open(SNAPSHOT, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_snapshot(values):
    payload = {
        "digits": DIGITS,
        "note": "由 tools/numeric_regression.py --update 生成；CI 用 --check 校验数值漂移",
        "values": values,
    }
    with open(SNAPSHOT, "w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def _diff(exp, got, prefix="", out=None):
    if out is None:
        out = []
    if isinstance(exp, tuple):
        exp = list(exp)
    if isinstance(got, tuple):
        got = list(got)
    if isinstance(exp, dict) and isinstance(got, dict):
        for k in sorted(set(exp) | set(got)):
            if k not in exp:
                out.append(f"{prefix}{k}: 快照缺失")
            elif k not in got:
                out.append(f"{prefix}{k}: 当前缺失")
            else:
                _diff(exp[k], got[k], prefix + k + ".", out)
        return out
    if isinstance(exp, list) and isinstance(got, list):
        if len(exp) != len(got):
            out.append(f"{prefix}长度不同")
            return out
        for i, (a, b) in enumerate(zip(exp, got)):
            _diff(a, b, f"{prefix}[{i}].", out)
        return out
    if isinstance(exp, float) and isinstance(got, float):
        if abs(exp - got) > 1e-6:
            out.append(f"{prefix}{exp!r} != {got!r}")
        return out
    if exp != got:
        out.append(f"{prefix}{exp!r} != {got!r}")
    return out


def main(argv):
    if "--update" in argv:
        values = build_snapshot()
        _write_snapshot(values)
        print(f"wrote {SNAPSHOT}: {len(values)} cases")
        return 0
    if "--check" in argv:
        exp = _read_snapshot()
        if exp.get("digits") != DIGITS:
            print(f"snapshot digits={exp.get('digits')} 与代码 {DIGITS} 不一致")
            return 1
        got = build_snapshot()
        diffs = _diff(exp["values"], got)
        if diffs:
            print(f"数值回归失败：{len(diffs)} 处漂移")
            for d in diffs[:20]:
                print("  -", d)
            return 1
        print(f"数值回归通过：{len(got)} 项与快照一致（1e-6 容差）")
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
