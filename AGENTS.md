# AGENTS.md

本仓库是主仓库（source of truth）。

- 所有新需求先在 Python 侧实现：`AI.py` + `test_ai.py`，测试通过后再同步到：
  - C 侧 `D:\vscode\AI_C`（`c/` 实现 + `tests/` 对拍）
  - Web 页 `D:\vscode\RSSeeker.github.io`（`index.html` + `tools/`）
- 文档顺序：先回填本仓库 `README.md` / `ROADMAP.md`，再同步 web 的 `README.md`；
  ROADMAP 的同步范围与计数以本仓库为准。
- 三个仓库各自单独提交、单独推送；C / Web 的同步提交建议在 message 中引用 Python 侧提交号。
