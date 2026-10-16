# -*- coding: utf-8 -*-
"""与 C 版对等的性能基准（工作负载必须与 c/bench.c 一致）

用法：python tools/c_bench.py [epochs] [samples]
"""
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from AI import MLP, Model              # noqa: E402


def main():
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    in_dim, out_dim = 64, 8

    x = [[(i * 7 + k * 13) % 100 / 50.0 - 1.0 for k in range(in_dim)] for i in range(n)]
    y = []
    for i in range(n):
        row = []
        for k in range(out_dim):
            s = sum(x[i][(k * 8 + t) % in_dim] for t in range(8))
            row.append(s / 8.0)
        y.append(row)

    mlp = MLP([64, 256, 128, 8], activations=["leaky_relu", "leaky_relu", "linear"])
    model = Model(mlp, loss_fn="mse", optimizer="adam", lr=0.001, task="regression")
    params = sum(l.fan_in * l.fan_out + l.fan_out for l in mlp.layers)

    print(f"Python 版: MLP[64,256,128,8] 参数 {params}, {n} 样本, {epochs} 轮, batch=32, Adam(0.001)")
    t0 = time.perf_counter()
    model.fit(x, y, epochs=epochs, batch_size=32, shuffle=True, val_split=0.0,
              patience=0, verbose=False)
    dt = time.perf_counter() - t0
    r2 = model.score(x, y)
    print(f"  耗时 {dt:.3f} s，最终 R²(训练集) = {r2:.6f}")
    print(f"  每轮 {dt * 1000.0 / epochs:.1f} ms，每样本每轮 {dt * 1e6 / (epochs * n):.1f} us")


if __name__ == "__main__":
    main()
