# core-patches — the one change this node still needs in ComfyUI

Everything else the node does lives in `custom_nodes/ComfyUI-gfx906-Tuning`:
`deltanet.py` (Triton decode kernels), `qwen35_model.py` (the Qwen3.5 compute
classes), `core_llama_patches.py` (two value-identical function replacements in
`llama.py`), `gfx906_tuning.py` (int8 GEMV, convrot, attention, the folded GQA
decode). Those need no file outside the node directory.

What cannot be done that way is the **per-block CUDA graph** capture: it hooks
the execution path (`prefetch_queue_pop`, the `decode_buffers` allocation in
`BaseGenerate.generate`, and the queue cleanup in `execution.py`), which is flow
control rather than a single function, and copying those functions at runtime
would drift silently when ComfyUI changes. So it ships as three known-good files:

```
overlay/comfy/model_prefetch.py          RESIDENT_GRAPH_MODULES, per-block capture/replay
overlay/comfy/text_encoders/llama.py     decode_buffers allocation, enable_graph for resident blocks
overlay/execution.py                     cleanup_prefetch_queues called outside the aimdo gate
```

## Install

```bash
git clone <this node> ComfyUI/custom_nodes/ComfyUI-gfx906-Tuning

python custom_nodes/ComfyUI-gfx906-Tuning/tools/core_patch.py --check   # read-only
python custom_nodes/ComfyUI-gfx906-Tuning/tools/core_patch.py --apply   # backs up, then copies
# restart ComfyUI
```

`--check` prints, per file, `applied`, `not applied <N lines changed>`, and
whether a backup exists. `--apply` stores the current files under
`<ComfyUI>/.core-patch-backup/` before overwriting, so `--revert` restores them.

## If you skip it

Nothing breaks. The node checks for `RESIDENT_GRAPH_MODULES` at startup and logs
`gfx906: resident graph hooks absent ...`. Decode then runs the blocks eagerly
instead of replaying a captured graph: on the Qwen3.5 text encoder that is
roughly 25 ms/token → 40 ms/token. Results are identical either way — this is a
launch-overhead optimization, not a correctness feature.

## Upgrading ComfyUI

```bash
python custom_nodes/ComfyUI-gfx906-Tuning/tools/core_patch.py --revert
# upgrade / replace the ComfyUI tree
python custom_nodes/ComfyUI-gfx906-Tuning/tools/core_patch.py --check
```

The overlay is made for ComfyUI **0.39.0** (the `pyproject.toml` version is
checked). On a different version `--check` says so; compare the three files
before applying, and re-generate the overlay if the core hooks moved.

## Why an overlay instead of a `.patch`

A unified diff needs the pristine originals of these three files. This tree was
edited before any baseline copy existed (no root git repo, no backups), so a diff
cannot be produced retroactively. The overlay gives the same result — a known-good
file set plus backup/restore — without inventing a fake baseline. Once a clean
0.39.0 tree is available, `diff -u` against it produces the real patch and this
directory can carry it instead.
