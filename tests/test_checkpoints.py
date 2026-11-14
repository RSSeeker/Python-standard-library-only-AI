import os

import pytest

import AI


def _manager(tmp_path, **kw):
    kw.setdefault("keep_last", 2)
    kw.setdefault("keep_best", 1)
    return AI.CheckpointManager(str(tmp_path / "ckpts"), **kw)


def _model():
    return AI.MLP([2, 3, 1], ["relu", "linear"])


def test_rolling_and_best_union(tmp_path):
    cm = _manager(tmp_path)
    model = _model()
    for epoch, value in enumerate([0.9, 0.5, 0.7, 0.3, 0.4, 0.6]):
        cm.save(model, epoch, {"val_loss": value})
    files = sorted(f for f in os.listdir(str(tmp_path / "ckpts")) if f.endswith(".json"))
    # 最近 2 个（epoch 5、4） + 最优 1 个（epoch 3） + manifest
    assert "manifest.json" in files
    assert "ckpt-epoch-00005.json" in files
    assert "ckpt-epoch-00004.json" in files
    assert "ckpt-epoch-00003.json" in files
    assert len([f for f in files if f.startswith("ckpt-epoch-")]) == 3
    latest = cm.resume()
    assert latest["epoch"] == 5
    assert os.path.exists(latest["path"])


def test_best_mode_max_and_k(tmp_path):
    cm = _manager(tmp_path, metric="acc", mode="max", keep_best=2)
    model = _model()
    for epoch, value in enumerate([0.1, 0.9, 0.5, 0.8, 0.2]):
        cm.save(model, epoch, {"acc": value})
    best = cm.best(2)
    assert [r["value"] for r in best] == [0.9, 0.8]
    assert [r["epoch"] for r in best] == [1, 3]


def test_missing_metric_keeps_last_only(tmp_path):
    cm = _manager(tmp_path, keep_last=1, keep_best=1)
    model = _model()
    cm.save(model, 0, {"loss": 1.0})
    cm.save(model, 1, None)
    cm.save(model, 2, {"other": 1.0})
    files = [f for f in os.listdir(str(tmp_path / "ckpts")) if f.startswith("ckpt-epoch-")]
    assert files == ["ckpt-epoch-00002.json"]
    assert cm.best(1) == []


def test_load_best_roundtrip(tmp_path):
    cm = _manager(tmp_path)
    model = _model()
    weights = [row[:] for row in model.layers[0].weights]
    cm.save(model, 0, {"val_loss": 1.0})
    model.layers[0].weights[0][0] += 0.5
    cm.save(model, 1, {"val_loss": 0.1})
    loaded, rec = cm.load_best(1)
    assert rec["epoch"] == 1
    assert isinstance(loaded, AI.MLP)
    assert loaded.layers[0].weights[0][0] == pytest.approx(
        model.layers[0].weights[0][0])


def test_resume_empty_directory(tmp_path):
    cm = _manager(tmp_path)
    assert cm.resume() is None
    assert cm.best(1) == []


def test_invalid_inputs(tmp_path):
    with pytest.raises(ValueError):
        AI.CheckpointManager("")
    with pytest.raises(ValueError):
        AI.CheckpointManager(str(tmp_path), mode="avg")
    with pytest.raises(ValueError):
        AI.CheckpointManager(str(tmp_path), keep_last=0)
    with pytest.raises(ValueError):
        AI.CheckpointManager(str(tmp_path), keep_best=-1)
    cm = AI.CheckpointManager(str(tmp_path / "x"))
    with pytest.raises(ValueError):
        cm.save(_model(), 0, "bad")
    with pytest.raises(ValueError):
        cm.save(_model(), -1, {"val_loss": 1.0})
    with pytest.raises(ValueError):
        cm.best(0)


def test_callback_drives_fit(tmp_path):
    cm = _manager(tmp_path)
    model = AI.Model(_model(), loss_fn="mse", optimizer="adam", lr=0.05,
                     task="regression")
    xs = [[0.1, 0.2], [0.3, -0.1], [-0.4, 0.5], [0.6, 0.7]]
    ys = [0.1, 0.2, -0.3, 0.4]
    cb = AI.CheckpointCallback(cm, start_epoch=0)
    model.fit(xs, ys, epochs=3, batch_size=2, verbose=False, callbacks=[cb])
    assert cm.resume()["epoch"] == 3
    assert len(cm.best(1)) == 1


def test_callback_start_epoch_offset(tmp_path):
    cm = _manager(tmp_path)
    model = AI.Model(_model(), loss_fn="mse", optimizer="adam", lr=0.05,
                     task="regression")
    cb = AI.CheckpointCallback(cm, start_epoch=10)
    model.fit([[0.1, 0.2], [0.3, -0.1]], [0.1, 0.2], epochs=2,
              batch_size=2, verbose=False, callbacks=[cb])
    assert cm.resume()["epoch"] == 12
