"""
Legacy modeling package — superseded by `renaissance.modeling`.

`RenaissanceTransformer` was removed in the Phase 7 cutover; `run.py`,
`eval.py`, and the tests now use `renaissance.modeling.RenaissanceModel`.

What remains here is still load-bearing until the Phase 8 backbone
modernization:
- `one_tower_encoder` / `two_tower_encoder` / `embeddings` /
  `fusion_encoder` — wrapped by `renaissance.modeling.backbones`.
- `heads.Pooler`, `objectives.init_weights` — imported by those encoders.
- `renaissance_utils.set_schedule` — used by `renaissance.trainer`.
- `dist_utils` — used by `objectives`.

Phase 8 deletes this package entirely once the encoder internals are
rewritten and these last dependencies are gone.
"""
