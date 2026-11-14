"""生成 API_STABILITY.md：哪些 API 承诺逐位一致、哪些只是教学实现。

用法：``python tools/api_stability.py``（写文件）或 ``--check``（校验表格里的
名字都真实存在于 AI.py）。
"""

import inspect
import io
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import AI  # noqa: E402

ROWS = [
    ("数值逐位一致（序列化/导出）", "bit-exact", [
        ("safetensors_save", "写入的 f32 字节可直接被读回"),
        ("safetensors_load", "读回与写入逐位相同"),
        ("export_onnx_mlp", "MLP 的 ONNX 图与权重确定性导出"),
        ("cross_entropy_loss", "固定输入下结果确定"),
        ("mse_loss", "固定输入下结果确定"),
        ("ScaledCosineAttention", "固定 seed 初始化后前向确定"),
    ]),
    ("数值稳定但仍为教学实现", "teaching-numeric", [
        ("zero_adam_step", "算法数值模拟，未接真实通信"),
        ("tensor_parallel_linear_backward", "分片数学正确，未接 NCCL"),
        ("pipeline_1f1b_schedule", "调度顺序模拟"),
        ("Shampoo", "Newton 逆根有限步迭代，非精确矩阵根"),
        ("sophia_hutchinson_diag", "Hutchinson 估计有方差"),
    ]),
    ("仅教学/演示（不承诺 API 稳定）", "demo-only", [
        ("rope_2d", "形状与数学约定可能随文档演进"),
        ("mcts_search", "纯 Python 搜索，性能非目标"),
        ("consistency_model_sample", "简化采样器"),
    ]),
]


def build_markdown():
    lines = ["# API 稳定性约定", "",
             "本表说明哪些 API 是「逐位一致」承诺、哪些只是教学实现。",
             "逐位一致 = 固定输入/固定 seed 下跨平台数值相同（CI 的",
             "`tools/numeric_regression.py --check` 对此有快照校验）。", ""]
    for title, tag, rows in ROWS:
        lines.append("## %s" % title)
        lines.append("")
        lines.append("| API | 约定 | 说明 |")
        lines.append("|---|---|---|")
        for name, note in rows:
            lines.append("| `%s` | `%s` | %s |" % (name, tag, note))
        lines.append("")
    lines.append("## 说明")
    lines.append("")
    lines.append("- 未列出的名字默认按「教学实现」处理，不承诺跨版本签名稳定；")
    lines.append("- 逐位一致的判定依赖 CI 快照（12 组固定计算，容差 1e-6）；")
    lines.append("- 序列化格式（safetensors/ONNX）以本库写出→本库读回为准，")
    lines.append("  不保证与第三方实现的字节级兼容。")
    lines.append("")
    return "\n".join(lines)


def check_names():
    missing = []
    for _title, _tag, rows in ROWS:
        for name, _note in rows:
            if not hasattr(AI, name):
                missing.append(name)
    return missing


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    md = build_markdown()
    path = os.path.join(REPO, "API_STABILITY.md")
    if "--check" in argv:
        missing = check_names()
        if missing:
            print("MISSING: %s" % ", ".join(missing))
            return 1
        print("API_STABILITY OK (%d names)"
              % sum(len(rows) for _t, _g, rows in ROWS))
        return 0
    io.open(path, "w", encoding="utf-8", newline="\n").write(md)
    print("wrote", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
