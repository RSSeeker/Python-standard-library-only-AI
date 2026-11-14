import json
import os
import subprocess
import sys
import time
import urllib.request

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _write_dataset(tmp_path, n=40, classification=False):
    rows = ["x1,x2,y"]
    for i in range(n):
        a = round(((i % 7) - 3) / 3.0, 6)
        b = round(((i % 5) - 2) / 2.0, 6)
        if classification:
            rows.append("%s,%s,%s" % (a, b, "pos" if a + b >= 0 else "neg"))
        else:
            rows.append("%s,%s,%.6f" % (a, b, 2 * a - b))
    p = tmp_path / "train.csv"
    p.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return p


def _config(tmp_path, classification=False, epochs=20):
    cfg = {
        "model": {"hidden": [8], "activation": "relu"},
        "data": {"train": "train.csv", "target": "y"},
        "train": {"epochs": epochs, "lr": 0.05, "batch_size": 8},
        "output": "run",
    }
    if classification:
        cfg["task"] = "classification"
    p = tmp_path / "config.json"
    p.write_text(json.dumps(cfg), encoding="utf-8")
    return p


def _env():
    env = dict(os.environ)
    env["PYTHONPATH"] = REPO + os.pathsep + env.get("PYTHONPATH", "")
    return env


def _run(tmp_path, *args):
    return subprocess.run([sys.executable, "-m", "ai", *args], cwd=str(tmp_path),
                          env=_env(), capture_output=True, text=True, timeout=120)


def test_train_creates_checkpoint_and_metrics(tmp_path):
    _write_dataset(tmp_path)
    _config(tmp_path)
    proc = _run(tmp_path, "train", "--config", "config.json")
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout[proc.stdout.index("{"):])
    assert os.path.exists(tmp_path / payload["checkpoint"])
    assert payload["metrics"]["samples"] == 40
    assert payload["epochs"] >= 1
    assert os.path.exists(tmp_path / "run" / "train-000.json")


def test_eval_reports_metrics(tmp_path):
    _write_dataset(tmp_path)
    _config(tmp_path)
    assert _run(tmp_path, "train", "--config", "config.json").returncode == 0
    proc = _run(tmp_path, "eval", "--checkpoint", "run/model.json", "--config", "config.json")
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout[proc.stdout.index("{"):])
    assert payload["metrics"]["samples"] == 40
    assert "mse" in payload["metrics"]


def test_export_json_and_onnx(tmp_path):
    _write_dataset(tmp_path)
    _config(tmp_path)
    assert _run(tmp_path, "train", "--config", "config.json").returncode == 0
    proc = _run(tmp_path, "export", "--checkpoint", "run/model.json",
                "--output", "m2.json", "--format", "json")
    assert proc.returncode == 0, proc.stderr
    assert (tmp_path / "m2.json").exists()
    proc = _run(tmp_path, "export", "--checkpoint", "run/model.json",
                "--output", "m2.onnx", "--format", "onnx")
    assert proc.returncode == 0, proc.stderr
    assert (tmp_path / "m2.onnx").exists()


def test_serve_health_and_predict(tmp_path):
    _write_dataset(tmp_path)
    _config(tmp_path)
    assert _run(tmp_path, "train", "--config", "config.json").returncode == 0
    port = 8127
    proc = subprocess.Popen([sys.executable, "-m", "ai", "serve",
                             "--checkpoint", "run/model.json", "--port", str(port)],
                            cwd=str(tmp_path), env=_env(), stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.time() + 30
        while True:
            try:
                body = urllib.request.urlopen("http://127.0.0.1:%d/health" % port, timeout=1).read()
                assert json.loads(body.decode())["status"] == "ok"
                break
            except Exception:
                if time.time() > deadline:
                    raise
                time.sleep(0.2)
        req = urllib.request.Request(
            "http://127.0.0.1:%d/predict" % port,
            data=json.dumps({"x": [0.5, -0.2]}).encode(),
            headers={"Content-Type": "application/json"})
        payload = json.loads(urllib.request.urlopen(req, timeout=5).read().decode())
        assert isinstance(payload["prediction"], list) and len(payload["prediction"]) == 1
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


def test_bad_config_returns_error_code(tmp_path):
    bad = tmp_path / "config.json"
    bad.write_text('{"data": {"train": "missing.csv", "target": "y"}}', encoding="utf-8")
    proc = _run(tmp_path, "train", "--config", "config.json")
    assert proc.returncode != 0
    assert "error" in proc.stderr


def test_train_resume_continues_epochs(tmp_path):
    _write_dataset(tmp_path)
    _config(tmp_path, epochs=3)
    assert _run(tmp_path, "train", "--config", "config.json").returncode == 0
    _config(tmp_path, epochs=6)
    proc = _run(tmp_path, "train", "--config", "config.json", "--resume")
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout[proc.stdout.index("{"):])
    assert payload["epochs"] == 6
    manifest = json.loads((tmp_path / "run" / "checkpoints" / "manifest.json").read_text())
    assert manifest["latest"]["epoch"] == 6
