import json
import subprocess
import sys

import pytest

from ai import launch as L


def test_launch_returns_all_ranks():
    data = L._make_data(6, seed=3)
    params = {"w": [0.1, 0.2, -0.1], "b": 0.0}
    results = L.launch(L._rank_demo, 3, {"data": data, "params": params})
    assert [rank for rank, _r in results] == [0, 1, 2]
    assert sum(r["n"] for _rank, r in results) == 6


def test_launch_rejects_bad_nproc():
    with pytest.raises(ValueError):
        L.launch(L._rank_demo, 0)
    with pytest.raises(ValueError):
        L.launch(L._rank_demo, True)


def test_data_parallel_demo_equivalent():
    out = L.run_data_parallel_demo(nproc_per_node=2, steps=2, seed=5)
    assert out["equivalent"] is True
    assert out["max_abs_diff_vs_single_process"] < 1e-12
    assert out["ranks"] == [0, 1]
    assert out["samples_per_rank"] == [4, 4]


def test_data_parallel_demo_uneven_shards():
    out = L.run_data_parallel_demo(nproc_per_node=3, steps=1, seed=11)
    assert out["equivalent"] is True
    assert sorted(out["samples_per_rank"]) == [2, 3, 3]


def test_launch_cli():
    import os
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    proc = subprocess.run(
        [sys.executable, "-m", "ai.launch", "--nproc_per_node", "2", "--steps", "1"],
        capture_output=True, text=True, cwd=repo,
        env={**os.environ, "PYTHONPATH": repo},
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["equivalent"] is True
    assert payload["ranks"] == [0, 1]
