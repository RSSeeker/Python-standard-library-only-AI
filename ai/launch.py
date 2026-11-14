"""进程级分布式启动器（torchrun 的极简教学替代）。

用法::

    python -m ai.launch --nproc_per_node 2 --steps 2

在 ``nproc_per_node`` 个独立进程里并行执行同一个「rank 函数」，收集各 rank
的返回值后聚合。内置 demo 用数据并行的线性回归验证：各 rank 只在自己的数据分片
上算梯度，协调端按样本数加权平均后与单进程全量梯度逐位对比。

注意：这是**教学用的进程级启动器**，不做真实 GPU/CUDA 通信；分布式算法本身
（ZeRO/TP/PP/Ring/FSDP）仍是 AI.py 里的数值模拟。
"""

import argparse
import json
import multiprocessing as mp
import os
import sys
import traceback

import AI

_MP = mp


def _worker(queue, rank, world_size, payload):
    try:
        result = _rank_demo(rank, world_size, payload)
        queue.put(("ok", rank, result))
    except BaseException:
        queue.put(("err", rank, traceback.format_exc()))


# ---------- 内置 demo：数据并行线性回归 ----------


def _make_data(n=8, seed=7):
    rng = __import__("random").Random(seed)
    data = []
    for _ in range(n):
        x = [rng.uniform(-1.0, 1.0) for _ in range(3)]
        y = 1.5 * x[0] - 2.0 * x[1] + 0.5 * x[2] + 0.25
        data.append((x, y))
    return data


def _shard(rank, world_size, data):
    return [item for i, item in enumerate(data) if i % world_size == rank]


def _linear_grads(data, params):
    w, b = params["w"], params["b"]
    gw = [0.0] * len(w)
    gb = 0.0
    for x, y in data:
        pred = sum(w[i] * x[i] for i in range(len(w))) + b
        err = pred - y
        scale = 2.0 * err / len(data)
        for i in range(len(w)):
            gw[i] += scale * x[i]
        gb += scale
    return {"w": gw, "b": gb, "n": len(data)}


def _rank_demo(rank, world_size, payload):
    data = payload["data"]
    params = payload["params"]
    shard = _shard(rank, world_size, data)
    if not shard:
        raise ValueError("rank %d 没有分配到任何样本（样本数需 >= nproc_per_node）" % rank)
    grad = _linear_grads(shard, params)
    return {"rank": rank, "n": grad["n"], "grad_w": grad["w"], "grad_b": grad["b"]}


def launch(fn, nproc_per_node=1, payload=None, timeout=120.0):
    """并行执行 ``fn(rank, world_size, payload)``，返回 ``[(rank, result), ...]``。"""
    if not isinstance(nproc_per_node, int) or isinstance(nproc_per_node, bool) or nproc_per_node < 1:
        raise ValueError("launch: nproc_per_node 必须是 >=1 的整数")
    payload = {} if payload is None else payload
    ctx = _MP.get_context("spawn")
    queue = ctx.Queue()
    procs = []
    for rank in range(nproc_per_node):
        p = ctx.Process(target=_worker, args=(queue, rank, nproc_per_node, payload))
        p.start()
        procs.append(p)
    results = []
    try:
        for _ in range(nproc_per_node):
            status, rank, value = queue.get(timeout=timeout)
            if status != "ok":
                raise RuntimeError("rank %d 执行失败:\n%s" % (rank, value))
            results.append((rank, value))
    finally:
        for p in procs:
            p.join(timeout=10.0)
            if p.is_alive():
                p.terminate()
                p.join(timeout=5.0)
    results.sort(key=lambda kv: kv[0])
    return results


def run_data_parallel_demo(nproc_per_node=2, steps=2, seed=7):
    """启动 nproc 个 rank，验证按样本数加权平均的梯度 == 单进程全量梯度。"""
    data = _make_data(8, seed=seed)
    params = {"w": [0.30, -0.10, 0.05], "b": 0.0}
    results = launch(_rank_demo, nproc_per_node, {"data": data, "params": params})
    total = sum(r["n"] for _rank, r in results)
    avg_w = [sum(r["n"] * r["grad_w"][i] for _rank, r in results) / total
             for i in range(3)]
    avg_b = sum(r["n"] * r["grad_b"] for _rank, r in results) / total
    full = _linear_grads(data, params)
    diff = max(max(abs(avg_w[i] - full["w"][i]) for i in range(3)),
               abs(avg_b - full["b"]))
    # 多 step：每步都可用同样的 allreduce 结果做一次 SGD 更新（这里只演示前向+反向）
    for _ in range(max(0, steps - 1)):
        for i in range(3):
            params["w"][i] -= 0.01 * avg_w[i]
        params["b"] -= 0.01 * avg_b
        results = launch(_rank_demo, nproc_per_node, {"data": data, "params": params})
        total = sum(r["n"] for _rank, r in results)
        avg_w = [sum(r["n"] * r["grad_w"][i] for _rank, r in results) / total
                 for i in range(3)]
        avg_b = sum(r["n"] * r["grad_b"] for _rank, r in results) / total
    return {
        "nproc_per_node": nproc_per_node,
        "steps": steps,
        "ranks": [rank for rank, _r in results],
        "samples_per_rank": [r["n"] for _rank, r in results],
        "grad_b": avg_b,
        "max_abs_diff_vs_single_process": diff,
        "equivalent": diff < 1e-12,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m ai.launch",
                                     description="进程级分布式启动器（教学 demo）")
    parser.add_argument("--nproc_per_node", type=int, default=2)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)
    try:
        payload = run_data_parallel_demo(args.nproc_per_node, args.steps, args.seed)
    except Exception as exc:  # noqa: BLE001 - CLI 统一错误出口
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
