import ast
import io
import os


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_stub_syntax():
    path = os.path.join(REPO, "AI.pyi")
    text = io.open(path, encoding="utf-8").read()
    tree = ast.parse(text)  # .pyi 也必须是合法 Python 语法
    names = {node.name for node in tree.body
             if isinstance(node, (ast.ClassDef, ast.FunctionDef))}
    for required in ("MLP", "Layer", "Model", "CheckpointManager",
                     "ExperimentTracker", "profile_model", "safetensors_load"):
        assert required in names, required


def test_py_typed_and_mypy_ini():
    assert os.path.exists(os.path.join(REPO, "py.typed"))
    ini = io.open(os.path.join(REPO, "mypy.ini"), encoding="utf-8").read()
    assert "files" in ini
    # 存根覆盖的名字必须真实存在于运行时模块，避免存根与实现漂移
    import AI
    for name in ("MLP", "Layer", "Model", "CheckpointManager",
                 "ExperimentTracker", "profile_model"):
        assert hasattr(AI, name), name
