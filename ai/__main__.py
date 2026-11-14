"""`python -m ai` 命令实现：train / eval / export / serve。

设计原则：只是把已有能力（load_config、Model、ExperimentTracker、
benchmark_forward、export_onnx_mlp）串成可执行入口，不引入新依赖。
"""

import argparse
import csv
import json
import os
import sys
import time

import AI


def _config_dir(config_path):
    return os.path.dirname(os.path.abspath(config_path))


def _resolve(base, path):
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(base, path))


def _read_csv(path, target):
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        rows = [row for row in reader if row]
    if len(rows) < 2:
        raise ValueError("CSV 至少需要表头与一行数据: %s" % path)
    header = rows[0]
    if target not in header:
        raise ValueError("CSV 中找不到目标列 %r（表头: %s）" % (target, header))
    tgt = header.index(target)
    features = [h for i, h in enumerate(header) if i != tgt]
    xs, ys = [], []
    for line_no, row in enumerate(rows[1:], start=2):
        if len(row) != len(header):
            raise ValueError("CSV 第 %d 行列数 %d != 表头列数 %d"
                             % (line_no, len(row), len(header)))
        try:
            x = [float(row[i]) for i in range(len(header)) if i != tgt]
        except ValueError:
            raise ValueError("CSV 第 %d 行特征不是数字" % line_no)
        raw = row[tgt].strip()
        try:
            y = float(raw)
        except ValueError:
            y = raw
        xs.append(x)
        ys.append(y)
    return xs, ys, features


def _is_classification(ys):
    return any(isinstance(y, str) for y in ys)


def _encode_labels(ys):
    labels = []
    for y in ys:
        if y not in labels:
            labels.append(y)
    return [labels.index(y) for y in ys], labels


def _build_model(config, n_features, n_outputs):
    model_cfg = config.get("model", {})
    hidden = model_cfg.get("hidden", [16])
    act = model_cfg.get("activation", "relu")
    sizes = [n_features] + list(hidden) + [n_outputs]
    acts = [act] * len(hidden) + ["linear" if n_outputs == 1 else "softmax"]
    return AI.MLP(sizes, acts)


def _flatten(v):
    if isinstance(v, (list, tuple)) and len(v) == 1:
        return v[0]
    return v


def _metrics(y_true, y_pred, labels=None):
    y_pred = [_flatten(v) for v in y_pred]
    if labels is not None:
        correct = sum(1 for a, b in zip(y_true, y_pred) if a == b)
        return {"accuracy": correct / len(y_true), "samples": len(y_true)}
    mse = sum((a - b) ** 2 for a, b in zip(y_true, y_pred)) / len(y_true)
    return {"mse": mse, "samples": len(y_true)}


def cmd_train(args):
    config = AI.load_config(args.config)
    base = _config_dir(args.config)
    data_cfg = config.get("data", {})
    train_cfg = config.get("train", {})
    out_dir = _resolve(base, config.get("output", "run"))
    os.makedirs(out_dir, exist_ok=True)
    train_csv = _resolve(base, data_cfg.get("train", "train.csv"))
    target = data_cfg.get("target", "y")
    xs, ys, features = _read_csv(train_csv, target)
    classification = _is_classification(ys)
    labels = None
    if classification:
        ys, labels = _encode_labels(ys)
    n_outputs = len(labels) if classification else 1
    mlp = _build_model(config, len(features), n_outputs)
    model = AI.Model(mlp, loss_fn="cross_entropy" if classification else "mse",
                     optimizer=train_cfg.get("optimizer", "adam"),
                     lr=train_cfg.get("lr", 0.05), task="classification" if classification else "regression")
    tracker = AI.ExperimentTracker(out_dir, name="train", extra={"config": config})
    tracker.start()
    started = time.time()
    try:
        history = model.fit(xs, ys, epochs=train_cfg.get("epochs", 50),
                            batch_size=train_cfg.get("batch_size", 16))
        ckpt = os.path.join(out_dir, "model.json")
        model.save(ckpt)
        curve = []
        if isinstance(history, dict):
            curve = [float(v) for (_ep, v) in history.get("train_loss", [])]
        n_epochs = len(curve)
        preds = model.predict_classes(xs) if classification else model.predict(xs)
        m = _metrics(ys, preds, labels)
        tracker.log_metrics(m)
        if curve:
            tracker.log_curve("loss", curve)
        elapsed = time.time() - started
        benchmark = AI.benchmark_forward(lambda: model.forward(xs[0]))
        tracker.finish()
        result = {"checkpoint": ckpt, "metrics": m, "epochs": n_epochs,
                  "seconds": round(elapsed, 3), "features": features,
                  "labels": labels, "forward_ms": round(benchmark["mean_ms"], 6)}
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        tracker.finish(status="failed", error=str(exc))
        raise


def cmd_eval(args):
    config = AI.load_config(args.config) if args.config else {}
    base = _config_dir(args.config) if args.config else os.getcwd()
    data_cfg = config.get("data", {})
    eval_csv = _resolve(base, args.data or data_cfg.get("eval", data_cfg.get("train", "train.csv")))
    target = data_cfg.get("target", "y")
    xs, ys, features = _read_csv(eval_csv, target)
    classification = _is_classification(ys)
    labels = None
    if classification:
        ys, labels = _encode_labels(ys)
    ckpt = args.checkpoint or _resolve(base, os.path.join(config.get("output", "run"), "model.json"))
    model = AI.Model.load(ckpt)
    preds = model.predict_classes(xs) if classification else model.predict(xs)
    m = _metrics(ys, preds, labels)
    print(json.dumps({"checkpoint": ckpt, "metrics": m, "labels": labels}, ensure_ascii=False, indent=2))
    return 0


def cmd_export(args):
    path = args.checkpoint
    model = AI.Model.load(path)
    out = args.output
    if args.format == "json":
        model.save(out)
    elif args.format == "onnx":
        AI.export_onnx_mlp(model.model, out)
    else:  # safetensors
        AI.save_safetensors({"weights": model.model}, out) if hasattr(AI, "save_safetensors") else _unsupported()
    print(json.dumps({"exported": out, "format": args.format}, ensure_ascii=False))
    return 0


def _unsupported():
    raise ValueError("当前版本不支持该导出格式")


def cmd_serve(args):
    import http.server

    model = AI.Model.load(args.checkpoint)

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, fmt, *a):
            pass

        def _reply(self, code, payload):
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                self._reply(200, {"status": "ok"})
            else:
                self._reply(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/predict":
                self._reply(404, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                x = payload["x"]
                if not isinstance(x, list) or any(not isinstance(v, (int, float)) for v in x):
                    raise ValueError("x 必须是数字列表")
                pred = model.predict(x)
                self._reply(200, {"prediction": pred})
            except Exception as exc:
                self._reply(400, {"error": str(exc)})

    server = http.server.HTTPServer((args.host, args.port), Handler)
    print(json.dumps({"serving": args.checkpoint, "host": args.host, "port": args.port}))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


def build_parser():
    p = argparse.ArgumentParser(prog="python -m ai", description="标准库 AI 工具箱命令行")
    sub = p.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train", help="按配置训练模型")
    t.add_argument("--config", required=True)
    t.set_defaults(func=cmd_train)

    e = sub.add_parser("eval", help="评估检查点")
    e.add_argument("--checkpoint", default=None)
    e.add_argument("--config", default=None)
    e.add_argument("--data", default=None)
    e.set_defaults(func=cmd_eval)

    ex = sub.add_parser("export", help="导出检查点")
    ex.add_argument("--checkpoint", required=True)
    ex.add_argument("--output", required=True)
    ex.add_argument("--format", choices=["json", "onnx", "safetensors"], default="json")
    ex.set_defaults(func=cmd_export)

    s = sub.add_parser("serve", help="启动 HTTP 推理服务")
    s.add_argument("--checkpoint", required=True)
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(func=cmd_serve)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except Exception as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
