#!/usr/bin/env python3
"""AI 工具箱的命令行入口（纯标准库：argparse + json）。

    python ai_cli.py train  --data train.json --out model.json --task classification
    python ai_cli.py predict --model model.json --input x.json
    python ai_cli.py eval   --model model.json --data test.json
    python ai_cli.py export --model model.json --out exported.json
    python ai_cli.py finetune --model model.json --data more.json --out model2.json

数据文件（JSON）支持三种形状，任选其一：

    {"x": [[...], ...], "y": [...]}            # 分开给
    {"data": [[[...], y], ...]}                # 元组列表
    [[[...], y], ...]                          # 直接就是列表

预测输入只接受 `{"x": [...]}`、`[[...], ...]` 或 `{"data": [...]}` 三种。
所有子命令成功返回 0，参数/文件错误返回 2，并把原因打到 stderr。
"""

import argparse
import contextlib
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import AI  # noqa: E402  （必须在 sys.path 修好之后）


def _die(message):
    sys.stderr.write("error: %s\n" % message)
    return 2


@contextlib.contextmanager
def _quiet():
    """把库里的中文提示音（"模型已保存到…"）收进缓冲，保证 stdout 只有 JSON。"""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        yield buffer


def _read_json(path, what):
    if not isinstance(path, str) or not path:
        raise ValueError("%s路径必须是非空字符串" % what)
    if not os.path.exists(path):
        raise ValueError("%s不存在: %s" % (what, path))
    try:
        return json.loads(io.open(path, encoding="utf-8").read())
    except ValueError as exc:
        raise ValueError("%s不是合法 JSON（%s）: %s" % (what, exc, path))


def _write_json(path, payload):
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2,
                      allow_nan=False)
    io.open(path, "w", encoding="utf-8", newline="\n").write(text + "\n")
    return path


def _as_rows(value, what):
    if not isinstance(value, list) or not value:
        raise ValueError("%s必须是非空列表" % what)
    return value


def read_dataset(path):
    """读 `(x, y)` 训练/评估数据，返回 `(xs, ys)`。"""
    payload = _read_json(path, "数据文件")
    if isinstance(payload, dict):
        if "x" in payload and "y" in payload:
            xs = _as_rows(payload["x"], "x")
            ys = _as_rows(payload["y"], "y")
        elif "data" in payload:
            pairs = _as_rows(payload["data"], "data")
            xs = [row[0] for row in pairs]
            ys = [row[1] for row in pairs]
        else:
            raise ValueError("数据文件字典必须含 x/y 或 data 字段")
    elif isinstance(payload, list):
        xs = [row[0] for row in payload]
        ys = [row[1] for row in payload]
    else:
        raise ValueError("数据文件必须是 dict 或 list")
    if len(xs) != len(ys):
        raise ValueError("x 与 y 长度不一致（%d != %d）" % (len(xs), len(ys)))
    return xs, ys


def read_inputs(path):
    """读预测输入，返回样本列表。"""
    payload = _read_json(path, "输入文件")
    if isinstance(payload, dict):
        if "x" in payload:
            return _as_rows(payload["x"], "x")
        if "data" in payload:
            return _as_rows(payload["data"], "data")
        raise ValueError("输入文件字典必须含 x 或 data 字段")
    if isinstance(payload, list):
        return payload
    raise ValueError("输入文件必须是 dict 或 list")


def parse_hidden(text):
    """把 "16,8" 解析成 (16, 8)；空串表示没有隐藏层。"""
    if text is None:
        return (32,)
    if not isinstance(text, str):
        raise ValueError("--hidden 必须是字符串")
    text = text.strip()
    if not text:
        return ()
    out = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            raise ValueError("--hidden 里有空项: %r" % text)
        try:
            value = int(part)
        except ValueError:
            raise ValueError("--hidden 的每一项必须是整数: %r" % part)
        if value <= 0:
            raise ValueError("--hidden 的每一项必须 > 0: %r" % part)
        out.append(value)
    return tuple(out)


def _build_parser():
    parser = argparse.ArgumentParser(
        prog="ai_cli.py", description="AI 工具箱命令行（train / predict / eval / export / finetune）")
    sub = parser.add_subparsers(dest="command")

    p_train = sub.add_parser("train", help="训练一个新模型")
    p_train.add_argument("--data", required=True, help="训练数据 JSON")
    p_train.add_argument("--out", required=True, help="模型输出路径")
    p_train.add_argument("--task", default="auto",
                         choices=("auto", "classification", "regression"))
    p_train.add_argument("--hidden", default="32", help="隐藏层宽度，逗号分隔，如 16,8")
    p_train.add_argument("--epochs", type=int, default=200)
    p_train.add_argument("--lr", type=float, default=0.01)
    p_train.add_argument("--val-split", type=float, default=0.1)
    p_train.add_argument("--seed", type=int, default=None)

    p_pred = sub.add_parser("predict", help="用已有模型预测")
    p_pred.add_argument("--model", required=True)
    p_pred.add_argument("--input", required=True, help="输入数据 JSON")
    p_pred.add_argument("--classes", action="store_true", help="输出类别下标而不是概率")

    p_eval = sub.add_parser("eval", help="在带标签数据上评估")
    p_eval.add_argument("--model", required=True)
    p_eval.add_argument("--data", required=True)

    p_export = sub.add_parser("export", help="导出模型 + 元数据")
    p_export.add_argument("--model", required=True)
    p_export.add_argument("--out", required=True)

    p_ft = sub.add_parser("finetune", help="在已有模型上继续训练")
    p_ft.add_argument("--model", required=True)
    p_ft.add_argument("--data", required=True)
    p_ft.add_argument("--out", required=True)
    p_ft.add_argument("--epochs", type=int, default=100)
    p_ft.add_argument("--lr", type=float, default=0.005)
    p_ft.add_argument("--val-split", type=float, default=0.0)

    return parser


def _load_model(path):
    if not os.path.exists(path):
        raise ValueError("模型文件不存在: %s" % path)
    with _quiet():
        return AI.Model.load(path)


def cmd_train(args):
    xs, ys = read_dataset(args.data)
    hidden = parse_hidden(args.hidden)
    task = "auto" if args.task == "auto" else args.task
    if task == "auto":
        first = ys[0]
        task = "regression" if isinstance(first, (int, float)) and not isinstance(first, bool) \
            else "classification"
    with _quiet():
        model = AI.quick_train(xs, ys, task=task, hidden_layers=hidden, epochs=args.epochs,
                               lr=args.lr, val_split=args.val_split, verbose=False,
                               seed=args.seed)
        model.save(args.out)
    print(json.dumps({"command": "train", "out": args.out, "task": task,
                      "samples": len(xs), "param_count": _param_count(model)},
                     ensure_ascii=False, sort_keys=True))
    return 0


def _param_count(model):
    net = getattr(model, "model", model)
    if hasattr(net, "param_count"):
        try:
            return int(net.param_count())
        except TypeError:
            return None
    total = 0
    for layer in getattr(net, "layers", []):
        if hasattr(layer, "fan_in") and hasattr(layer, "fan_out"):
            total += layer.fan_in * layer.fan_out + layer.fan_out
    return total


def cmd_predict(args):
    model = _load_model(args.model)
    rows = read_inputs(args.input)
    with _quiet():
        if args.classes:
            preds = [int(v) for v in model.predict_classes(rows)]
        else:
            preds = [model.predict(row) for row in rows]
    print(json.dumps({"command": "predict", "count": len(preds), "predictions": preds},
                     ensure_ascii=False, sort_keys=True))
    return 0


def cmd_eval(args):
    model = _load_model(args.model)
    xs, ys = read_dataset(args.data)
    with _quiet():
        score = model.score(xs, ys)
    print(json.dumps({"command": "eval", "score": score, "samples": len(xs)},
                     ensure_ascii=False, sort_keys=True))
    return 0


def cmd_export(args):
    model = _load_model(args.model)
    net = getattr(model, "model", model)
    payload = {
        "format": "ai-export-1",
        "model": json.loads(io.open(args.model, encoding="utf-8").read()),
        "metadata": AI.model_metadata(
            net,
            directory=os.path.dirname(os.path.abspath(args.model)),
            extra={"task": getattr(net, "saved_task", None)},
        ),
    }
    _write_json(args.out, payload)
    print(json.dumps({"command": "export", "out": args.out,
                      "param_count": payload["metadata"]["param_count"]},
                     ensure_ascii=False, sort_keys=True))
    return 0


def cmd_finetune(args):
    model = _load_model(args.model)
    xs, ys = read_dataset(args.data)
    if hasattr(model.optimizer, "lr"):
        model.optimizer.lr = args.lr
    with _quiet():
        history = model.fit(xs, ys, epochs=args.epochs, val_split=args.val_split, verbose=False)
        model.save(args.out)
    last = history["train_loss"][-1][1] if history.get("train_loss") else None
    print(json.dumps({"command": "finetune", "out": args.out,
                      "epochs": len(history.get("train_loss", [])),
                      "final_train_loss": last},
                     ensure_ascii=False, sort_keys=True))
    return 0


HANDLERS = {
    "train": cmd_train,
    "predict": cmd_predict,
    "eval": cmd_eval,
    "export": cmd_export,
    "finetune": cmd_finetune,
}


def main(argv=None):
    """CLI 主入口；返回退出码（0 成功，2 参数/文件错误）。"""
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 0
    handler = HANDLERS[args.command]
    try:
        return handler(args)
    except (ValueError, TypeError, OSError) as exc:
        return _die(str(exc))


if __name__ == "__main__":
    sys.exit(main())
