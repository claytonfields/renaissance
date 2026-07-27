# Renaissance 2.0

Ground-up redesign of Renaissance as a composable multimodal transformer
platform. Design source of truth: [`docs/v2-design.md`](../docs/v2-design.md).

Developed in-tree alongside the 1.3 line; becomes the mainline at cutover
(Phase 8). Both packages use the import name `renaissance`, so **never install
v2 and 1.3 into the same environment** — use a dedicated venv:

```bash
python3.11 -m venv .venv-v2 && source .venv-v2/bin/activate
pip install -e "v2[dev]"       # from the repo root
```

Run the tests from inside `v2/` so its own pytest config applies:

```bash
cd v2 && pytest
```
