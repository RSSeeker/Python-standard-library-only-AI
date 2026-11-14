"""报告导出：profiler JSON/HTML、内存报告、模型卡/数据卡。

全部是标准库实现；数据来自 AI.py 已有的 ``profile_model`` / ``track_memory`` /
``model_metadata``。
"""

import io
import json
import os
import time

import AI


def _report_check_path(path, name="path"):
    if not isinstance(path, str) or not path.strip():
        raise ValueError("report: %s 必须是非空字符串" % name)
    return path


def _report_write_text(path, text):
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with io.open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    return path


def export_profile_json(model, x, path, backward=False, repeat=3):
    """把 ``profile_model`` 的结果写成 JSON，返回 payload dict。"""
    _report_check_path(path)
    prof = AI.profile_model(model, x, backward=backward, repeat=repeat)
    payload = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
        "kind": "profile",
        "backward": bool(backward),
        "repeat": int(repeat),
        "profile": prof,
    }
    _report_write_text(path, json.dumps(payload, ensure_ascii=False,
                                        sort_keys=True, indent=2,
                                        allow_nan=False) + "\n")
    return payload


def _profile_rows(prof):
    rows = []
    for layer in prof.get("layers", []):
        rows.append(
            "<tr><td>%d</td><td>%s</td><td>%.4f</td><td>%.4f</td></tr>"
            % (layer.get("index", -1), layer.get("type", ""),
               float(layer.get("forward_ms", 0.0)),
               float(layer.get("backward_ms", 0.0) or 0.0)))
    return "".join(rows)


def export_profile_html(model, x, path, backward=False, repeat=3):
    """生成一个自包含 HTML profiler 报告。"""
    _report_check_path(path)
    payload = export_profile_json(model, x, path + ".json", backward=backward,
                                  repeat=repeat)
    prof = payload["profile"]
    html = """<!doctype html>
<html lang="zh"><head><meta charset="utf-8"><title>profile report</title>
<style>body{font-family:system-ui,Arial,sans-serif;margin:2em}
table{border-collapse:collapse}td,th{border:1px solid #bbb;padding:4px 10px}
th{background:#eee}</style></head><body>
<h1>Profiler 报告</h1>
<p>生成时间：%(created)s　总耗时：%(total).4f ms　反向合计：%(bwd).4f ms　参数：%(params)s</p>
<table><tr><th>#</th><th>层类型</th><th>前向 ms</th><th>反向 ms</th></tr>%(rows)s</table>
</body></html>
""" % {
        "created": payload["created_at"],
        "total": float(prof.get("total_ms", 0.0)),
        "bwd": float(prof.get("backward_total_ms", 0.0) or 0.0),
        "params": prof.get("params", "?"),
        "rows": _profile_rows(prof),
    }
    _report_write_text(path, html)
    return path


def export_memory_report(fn, path, label=None):
    """跑一次 ``fn()`` 并记录 ``track_memory`` 统计。"""
    _report_check_path(path)
    if not callable(fn):
        raise TypeError("report: fn 必须是可调用对象")
    with AI.track_memory(label=label) as tm:
        result = fn()
    stats = dict(tm.stats)
    stats["result_repr"] = repr(result)[:200]
    payload = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
        "kind": "memory",
        "stats": stats,
    }
    _report_write_text(path, json.dumps(payload, ensure_ascii=False,
                                        sort_keys=True, indent=2,
                                        allow_nan=False) + "\n")
    return payload


def generate_model_card(model, path, config=None, metrics=None, extra=None):
    """生成 Markdown 模型卡（架构/参数量/配置/指标）。"""
    _report_check_path(path)
    meta = AI.model_metadata(model, config=config, metrics=metrics, extra=extra)
    net = getattr(model, "model", model)
    lines = ["# 模型卡", ""]
    lines.append("- 生成时间：%s" % meta.get("created_at"))
    lines.append("- 参数量：%s" % meta.get("param_count"))
    if meta.get("git_hash"):
        lines.append("- git：`%s`" % meta["git_hash"])
    lines.append("")
    lines.append("## 架构")
    lines.append("")
    lines.append("| 层 | 输入 | 输出 | 激活 |")
    lines.append("|---|---|---|---|")
    for i, layer in enumerate(getattr(net, "layers", [])):
        lines.append("| %d | %d | %d | %s |"
                     % (i, int(getattr(layer, "fan_in", 0)),
                        int(getattr(layer, "fan_out", 0)),
                        getattr(layer, "activation_name", "?")))
    lines.append("")
    if config:
        lines.append("## 训练配置")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(config, ensure_ascii=False, sort_keys=True,
                                indent=2, allow_nan=False))
        lines.append("```")
        lines.append("")
    if metrics:
        lines.append("## 指标")
        lines.append("")
        for key in sorted(metrics):
            lines.append("- `%s`: %r" % (key, metrics[key]))
        lines.append("")
    lines.append("## 边界")
    lines.append("")
    lines.append("本卡片由 `ai.report.generate_model_card` 从运行时元数据生成；"
                 "指标来自调用方传入，未做独立验证。")
    text = "\n".join(lines) + "\n"
    _report_write_text(path, text)
    return text


def generate_data_card(path, name, num_samples, num_features, target,
                       notes=None):
    """生成 Markdown 数据卡。"""
    _report_check_path(path)
    if not isinstance(name, str) or not name.strip():
        raise ValueError("report: name 必须是非空字符串")
    if not isinstance(num_samples, int) or isinstance(num_samples, bool) or num_samples < 0:
        raise ValueError("report: num_samples 必须是非负整数")
    if not isinstance(num_features, int) or isinstance(num_features, bool) or num_features < 0:
        raise ValueError("report: num_features 必须是非负整数")
    lines = ["# 数据卡：%s" % name, ""]
    lines.append("- 样本数：%d" % num_samples)
    lines.append("- 特征数：%d" % num_features)
    lines.append("- 目标列：`%s`" % target)
    lines.append("- 生成时间：%s" % time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()))
    if notes:
        lines.append("")
        lines.append("## 说明")
        lines.append("")
        lines.append(str(notes))
    _report_write_text(path, "\n".join(lines) + "\n")
    return path
