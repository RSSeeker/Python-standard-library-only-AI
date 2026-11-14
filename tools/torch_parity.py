"""与 PyTorch 的对拍（golden test）套件。

用固定权重把本库实现与等价 PyTorch 模块逐项比较：前向数值、反向梯度、
优化器一步更新。没有安装 torch 时打印 SKIP 并以 0 退出（CI 环境通常没有
torch，本地运行时才有意义）。

用法：
    python tools/torch_parity.py
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import torch
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - 取决于环境
    print("SKIP: 未安装 torch，跳过与 PyTorch 的对拍")
    sys.exit(0)

import AI

CHECKS = []
FAILS = []


def close(name, got, want, tol=1e-6):
    ok = abs(float(got) - float(want)) <= tol * max(1.0, abs(float(want)))
    CHECKS.append(name)
    if not ok:
        FAILS.append(f"{name}: got {got!r}, want {want!r}")
    print(("PASS " if ok else "FAIL ") + name)


def close_list(name, got, want, tol=1e-6):
    flat_g = flatten(got)
    flat_w = flatten(want)
    ok = len(flat_g) == len(flat_w)
    if ok:
        for a, b in zip(flat_g, flat_w):
            if abs(float(a) - float(b)) > tol * max(1.0, abs(float(b))):
                ok = False
                break
    CHECKS.append(name)
    if not ok:
        worst = max((abs(float(a) - float(b)) for a, b in zip(flat_g, flat_w)),
                    default=0.0)
        FAILS.append(f"{name}: max_abs_diff={worst:.3e}")
    print(("PASS " if ok else "FAIL ") + name)


def flatten(obj):
    if isinstance(obj, (list, tuple)):
        out = []
        for item in obj:
            out.extend(flatten(item))
        return out
    return [obj]


def mlp_params(mlp):
    return [(torch.tensor(layer.weights, dtype=torch.float64),
             torch.tensor(layer.biases, dtype=torch.float64))
            for layer in mlp.layers]


def load_mlp_into_torch(mlp, torch_model):
    linears = [m for m in torch_model if isinstance(m, torch.nn.Linear)]
    pairs = mlp_params(mlp)
    assert len(linears) == len(pairs)
    with torch.no_grad():
        for lin, (w, b) in zip(linears, pairs):
            lin.weight.copy_(w)
            lin.bias.copy_(b)


def case_mlp_forward_backward():
    model = AI.MLP([4, 5, 3], ["sigmoid", "linear"])
    x = [0.1, -0.2, 0.3, 0.4]
    target = [0.2, -0.1, 0.5]
    out = model.forward(x)
    torch_model = torch.nn.Sequential(
        torch.nn.Linear(4, 5, dtype=torch.float64),
        torch.nn.Sigmoid(),
        torch.nn.Linear(5, 3, dtype=torch.float64),
    )
    load_mlp_into_torch(model, torch_model)
    tx = torch.tensor(x, dtype=torch.float64)
    tout = torch_model(tx)
    close_list("mlp_forward", out, tout.tolist())

    loss_grad = [2.0 * (out[k] - target[k]) / 3.0 for k in range(3)]
    grads = AI._mlp_backward(model, loss_grad)
    tloss = sum((tout[k] - target[k]) ** 2 for k in range(3)) / 3.0
    tloss.backward()
    for i, layer in enumerate(torch_model):
        if isinstance(layer, torch.nn.Linear):
            pass
    t_params = [(p, p.grad) for p in torch_model.parameters()]
    got = flatten([g for gw, gb in grads for g in (gw, gb)])
    want = flatten([p.grad.tolist() for p, g in t_params])
    close_list("mlp_backward", got, want)


def case_layer_norm():
    norm = AI.LayerNorm(4)
    norm.gamma = [1.1, 0.9, 1.0, 1.2]
    norm.beta = [0.1, -0.2, 0.0, 0.3]
    rows = [[0.5, -0.3, 0.8, 0.1], [1.0, 0.2, -0.4, 0.7]]
    out, _ = AI._norm_rows_forward(norm, rows)
    tout = F.layer_norm(torch.tensor(rows, dtype=torch.float64), (4,),
                        torch.tensor(norm.gamma, dtype=torch.float64),
                        torch.tensor(norm.beta, dtype=torch.float64),
                        norm.eps)
    close_list("layer_norm_forward", out, tout.tolist())


def case_conv2d():
    conv = AI.Conv2d(1, 2, 3, padding=1)
    x = [[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]]
    out = conv.forward(x)
    tw = torch.tensor(conv.weights, dtype=torch.float64, requires_grad=True)
    tb = torch.tensor(conv.biases, dtype=torch.float64, requires_grad=True)
    tout = F.conv2d(torch.tensor([x], dtype=torch.float64), tw, tb, padding=1)
    close_list("conv2d_forward", out, tout[0].tolist(), tol=1e-6)

    d_out = [[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]] for _ in range(2)]
    conv.zero_grad()
    conv.backward(d_out)
    t_out = tout.sum()
    t_out.backward()
    close_list("conv2d_grad_w", conv.grad_w, tw.grad.tolist(), tol=1e-5)


def case_cross_entropy():
    logits = [0.5, -0.3, 1.2, 0.1]
    probs = AI._softmax_row(logits)
    res = AI.cross_entropy_loss(probs, 2)
    got = res[0] if isinstance(res, (list, tuple)) else res
    want = F.cross_entropy(torch.tensor([logits], dtype=torch.float64),
                           torch.tensor([2]))
    close("cross_entropy", got, want.item())


def case_adam_step2():
    model = AI.MLP([3, 2], ["linear"])
    tmodel = torch.nn.Sequential(torch.nn.Linear(3, 2, dtype=torch.float64))
    load_mlp_into_torch(model, tmodel)
    grads = [([[0.1, -0.2, 0.3], [0.4, 0.1, -0.1]], [0.2, -0.3])]
    tgrads = [torch.tensor(grads[0][0], dtype=torch.float64),
              torch.tensor(grads[0][1], dtype=torch.float64)]
    for p, g in zip(tmodel.parameters(), tgrads):
        p.grad = g.clone()
    topt = torch.optim.Adam(tmodel.parameters(), lr=0.01, betas=(0.9, 0.999),
                            eps=1e-8)
    topt.step()
    AI.Adam(model, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8).step(grads)
    got = flatten([layer.weights for layer in model.layers]
                  + [layer.biases for layer in model.layers])
    want = flatten([p.detach().tolist() for p in tmodel.parameters()]
                   + [b.detach().tolist() for b in tmodel.parameters()])
    # 参数顺序：weights 然后 biases，torch parameters 是 weight,bias 交替
    want = flatten([[p.detach().tolist() for p in tmodel.parameters()][::2],
                    [p.detach().tolist() for p in tmodel.parameters()][1::2]])
    close_list("adam_step", got, want)


def main():
    case_mlp_forward_backward()
    case_layer_norm()
    case_conv2d()
    case_cross_entropy()
    case_adam_step2()
    total = len(CHECKS)
    print(f"\nPARITY RESULT: {total - len(FAILS)}/{total} passed")
    for msg in FAILS:
        print(" -", msg)
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
