import json
import os

import pytest

from ai import hf_compat as HF

import AI


def _model():
    m = AI.MLP([3, 4, 2], ["relu", "linear"])
    for i, layer in enumerate(m.layers):
        for r in range(layer.fan_out):
            for c in range(layer.fan_in):
                layer.weights[r][c] = 0.01 * (r + 1) - 0.02 * (c + 1)
            layer.biases[r] = 0.1 * i - 0.05 * r
    return m


def test_save_load_roundtrip(tmp_path):
    model = _model()
    x = [0.3, -0.2, 0.5]
    expected = model.forward(x)
    files = HF.save_hf_directory(model, str(tmp_path), tokenizer=["a", "b"])
    names = sorted(os.path.basename(f) for f in files)
    assert names == ["config.json", "model.safetensors", "tokenizer.json"]
    got = HF.load_hf_directory(str(tmp_path))
    assert got["model"].forward(x) == pytest.approx(expected)
    assert got["config"]["framework"] == "ai-py"
    assert got["config"]["ai_layers"][0] == {
        "in_features": 3, "out_features": 4, "activation": "relu"}
    assert got["tokenizer"]["words"] == ["a", "b"]
    assert got["tokenizer"]["word2id"]["b"] == 1


def test_config_fields(tmp_path):
    HF.save_hf_directory(_model(), str(tmp_path), num_labels=2)
    cfg = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert cfg["architectures"] == ["MLPForClassification"]
    assert cfg["num_hidden_layers"] == 2
    assert cfg["hidden_size"] == 4
    assert cfg["ai_num_labels"] == 2


def test_missing_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        HF.load_hf_directory(str(tmp_path / "nope"))
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError):
        HF.load_hf_directory(str(tmp_path / "empty"))
    (tmp_path / "bad").mkdir()
    (tmp_path / "bad" / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "bad" / "model.safetensors").write_bytes(b"")
    with pytest.raises(ValueError):
        HF.load_hf_directory(str(tmp_path / "bad"))


def test_invalid_inputs():
    with pytest.raises(TypeError):
        HF.save_hf_directory("not-a-model", "x")
    with pytest.raises(ValueError):
        HF.save_hf_directory(_model(), "")
    with pytest.raises(ValueError):
        HF.load_hf_directory("")
