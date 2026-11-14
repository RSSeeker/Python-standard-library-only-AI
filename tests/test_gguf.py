import os
import struct

import pytest

from ai import gguf

import AI


def _model():
    m = AI.MLP([3, 4, 2], ["relu", "linear"])
    for layer in m.layers:
        for r in range(layer.fan_out):
            for c in range(layer.fan_in):
                layer.weights[r][c] = 0.25 * (r + 1) - 0.1 * (c + 1)
            layer.biases[r] = 0.05 * r
    return m


def test_roundtrip(tmp_path):
    model = _model()
    path = str(tmp_path / "m.gguf")
    gguf.save_gguf(gguf.model_to_gguf_tensors(model), path,
                   metadata={"ai.arch": "mlp", "ai.hidden": 4})
    out = gguf.load_gguf(path)
    assert out["metadata"] == {"ai.arch": "mlp", "ai.hidden": 4}
    assert out["dims"]["model.layers.0.weight"] == [3, 4]
    assert out["dims"]["model.layers.0.bias"] == [4]
    for i, layer in enumerate(model.layers):
        got = out["tensors"]["model.layers.%d.weight" % i]
        for r in range(layer.fan_out):
            for c in range(layer.fan_in):
                assert got[r][c] == pytest.approx(layer.weights[r][c], rel=1e-6)


def test_header_and_alignment(tmp_path):
    path = str(tmp_path / "m.gguf")
    gguf.save_gguf(gguf.model_to_gguf_tensors(_model()), path)
    raw = open(path, "rb").read()
    assert raw[:4] == b"GGUF"
    assert struct.unpack("<I", raw[4:8])[0] == 3


def test_invalid_inputs(tmp_path):
    with pytest.raises(ValueError):
        gguf.save_gguf({}, str(tmp_path / "a.gguf"))
    with pytest.raises(ValueError):
        gguf.save_gguf({"w": 1.0}, str(tmp_path / "a.gguf"))
    with pytest.raises(ValueError):
        gguf.save_gguf({"w": [1.0]}, "")
    with pytest.raises(ValueError):
        gguf.save_gguf({"w": [1.0]}, str(tmp_path / "a.gguf"), metadata={"ok": True})
    with pytest.raises(TypeError):
        gguf.model_to_gguf_tensors([1, 2, 3])
    with pytest.raises(FileNotFoundError):
        gguf.load_gguf(str(tmp_path / "nope.gguf"))


def test_bad_magic_and_truncated(tmp_path):
    bad = tmp_path / "bad.gguf"
    bad.write_bytes(b"NOPE" + b"\x00" * 32)
    with pytest.raises(ValueError):
        gguf.load_gguf(str(bad))
    path = str(tmp_path / "m.gguf")
    gguf.save_gguf(gguf.model_to_gguf_tensors(_model()), path)
    raw = open(path, "rb").read()
    trunc = tmp_path / "trunc.gguf"
    trunc.write_bytes(raw[: len(raw) // 2])
    with pytest.raises((ValueError, struct.error)):
        gguf.load_gguf(str(trunc))
