import json
import os

import pytest

from ai import report

import AI


def _model():
    return AI.MLP([3, 4, 2], ["relu", "linear"])


def test_profile_json(tmp_path):
    path = str(tmp_path / "p.json")
    payload = report.export_profile_json(_model(), [0.1, 0.2, 0.3], path,
                                         backward=True, repeat=2)
    assert payload["kind"] == "profile"
    assert payload["profile"]["params"] == 3 * 4 + 4 + 4 * 2 + 2
    data = json.loads(open(path, encoding="utf-8").read())
    assert data["backward"] is True
    assert len(data["profile"]["layers"]) == 2


def test_profile_html(tmp_path):
    path = str(tmp_path / "p.html")
    report.export_profile_html(_model(), [0.1, 0.2, 0.3], path)
    html = open(path, encoding="utf-8").read()
    assert "Profiler 报告" in html and "<table>" in html
    assert os.path.exists(path + ".json")


def test_memory_report(tmp_path):
    path = str(tmp_path / "m.json")
    payload = report.export_memory_report(lambda: [0] * 10000, path, label="demo")
    assert payload["kind"] == "memory"
    assert payload["stats"]["peak"] > 0
    assert "result_repr" in payload["stats"]


def test_model_card(tmp_path):
    path = str(tmp_path / "card.md")
    text = report.generate_model_card(_model(), path,
                                      config={"lr": 0.05},
                                      metrics={"acc": 0.9})
    assert "模型卡" in text and "参数量" in text
    assert "| 0 | 3 | 4 | relu |" in text
    assert "acc" in text


def test_data_card_and_invalid(tmp_path):
    path = str(tmp_path / "data.md")
    report.generate_data_card(path, "toy", 100, 3, "label", notes="synthetic")
    text = open(path, encoding="utf-8").read()
    assert "数据卡：toy" in text and "样本数：100" in text
    with pytest.raises(ValueError):
        report.generate_data_card(path, "", 1, 1, "y")
    with pytest.raises(ValueError):
        report.generate_data_card(path, "toy", -1, 1, "y")
    with pytest.raises(ValueError):
        report.export_profile_json(_model(), [0.1], "")
    with pytest.raises(TypeError):
        report.export_memory_report("not-callable", path)
