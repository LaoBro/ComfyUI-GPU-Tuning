"""ComfyUI node pack: runtime tuning for AMD GPUs, measured on gfx906 (MI50 / MI60 / Radeon VII).

Applies the patches on startup so an unmodified ComfyUI gets the tuned int8
linear tiles, the butterfly convrot rotation and the fp16 attention kernel.
Two settings nodes expose the runtime switches (one in the model chain, one in
the CLIP chain); the model node's profile switch additionally reports what was
applied and prints per-prompt timings.
"""

import logging
import os

# Triton 3.9 warns once per process when this knob is unset, and 3.10 flips the
# default to disabled. Every measurement in gfx906fix assumed fusion enabled,
# so pin it before the first kernel compiles; an explicit env still wins.
os.environ.setdefault("TRITON_DEFAULT_FP_FUSION", "1")

from typing_extensions import override

from comfy_api.latest import ComfyExtension, io

from .gpu_tuning import apply, report_lines
from .settings_nodes import GpuTuningClipSettings, GpuTuningModelSettings
from .patches.profiler import install as install_profiler
from .patches.profiler import watch as watch_profiler


class GpuTuningExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [GpuTuningModelSettings, GpuTuningClipSettings]


async def comfy_entrypoint() -> GpuTuningExtension:
    apply()
    watch_profiler(report_lines)
    hooks = install_profiler()
    if hooks:
        logging.info("gpu-tune profiler hooks: %s (report printed when each prompt finishes)", ", ".join(hooks))
    return GpuTuningExtension()
