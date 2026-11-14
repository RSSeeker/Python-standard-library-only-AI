"""与 PyTorch 的数值对拍（golden test）。

没有安装 torch 的环境（例如默认 CI）自动跳过；本地装了 torch 时会实际运行
tools/torch_parity.py 并断言全部用例通过。
"""

import importlib.util
import os
import subprocess
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(REPO_ROOT, "tools", "torch_parity.py")


def test_torch_parity_suite():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("未安装 torch，跳过 PyTorch 对拍")
    env = dict(os.environ)
    env["PYTHONPATH"] = REPO_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, SCRIPT],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    out = proc.stdout + proc.stderr
    assert proc.returncode == 0, out
    assert "PARITY RESULT: 7/7 passed" in out, out
