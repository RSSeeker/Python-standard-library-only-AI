import os
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

import AI


def _model():
    return AI.MLP([4, 8, 3], ["relu", "linear"])


def test_parallel_forward_deterministic():
    model = _model()
    x = [0.1, -0.2, 0.3, 0.4]
    expected = model.forward(x)

    def work(_i):
        return model.forward(x)

    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(work, range(200)))
    assert all(r == expected for r in results)


def test_memory_no_leak_in_forward_loop():
    model = _model()
    x = [0.1, -0.2, 0.3, 0.4]
    for _ in range(200):  # 预热，让缓存/分配器稳定
        model.forward(x)
    with AI.track_memory(label="loop") as tm:
        for _ in range(3000):
            model.forward(x)
    # 纯前向不应逐次累积对象；允许 512KB 波动
    assert tm.stats["delta_current"] < 512 * 1024


def test_versioned_store_thread_writes(tmp_path):
    store = AI.VersionedStore(str(tmp_path / "store"))

    def put(i):
        store.put("name-%d" % i, {"i": i})

    with ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(put, range(40)))
    for i in range(40):
        assert store.get("name-%d" % i) == {"i": i}


def test_versioned_store_same_name_versions(tmp_path):
    store = AI.VersionedStore(str(tmp_path / "store"))
    for i in range(5):
        store.put("conf", {"lr": i})
    latest = store.get("conf")
    assert latest == {"lr": 4}
    versions = store.versions("conf")
    assert len(versions) == 5


def test_api_stability_doc_names_exist():
    import tools.api_stability as S
    assert S.check_names() == []
    assert os.path.exists(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "API_STABILITY.md"))
