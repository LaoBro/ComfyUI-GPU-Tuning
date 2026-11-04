"""Print the tuning status in a plain terminal, without opening ComfyUI.

    cd /home/liu/ComfyUI-0.39.0
    .venv/bin/python custom_nodes/ComfyUI-gfx906-Tuning/tools/status.py

Same text the `profile` switch on the GFX906 Model Settings node logs at startup
(see gfx906_tuning.status()), including the manifest line and the measured-hit
counters. Each run is a fresh process, so the counters start at zero: this shows
what would be applied, not what a running ComfyUI has served.

It runs apply() on this process only -- no file is touched, and the process
exits right after printing.
"""

import os
import sys
from pathlib import Path

NODE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(NODE_DIR))
sys.path.insert(0, str(NODE_DIR.parent.parent))          # ComfyUI root, for `import comfy`

# Same pin as the pack's __init__.py, which these top-level imports bypass.
os.environ.setdefault("TRITON_DEFAULT_FP_FUSION", "1")

import gfx906_tuning                                     # noqa: E402


def main():
    gfx906_tuning.apply()
    print(gfx906_tuning.status())
    return 0


if __name__ == "__main__":
    sys.exit(main())
