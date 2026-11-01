"""Install / remove the resident-graph core overlay shipped with this node.

The node owns the model side (deltanet.py, qwen35_model.py, core_llama_patches.py),
so the only changes it still needs in ComfyUI itself are the per-block CUDA-graph
hooks in three files. They touch the execution path rather than a single function
and cannot be monkeypatched safely, so they ship as known-good copies under
core-patches/overlay/.

    python custom_nodes/ComfyUI-gfx906-Tuning/tools/core_patch.py --check
    python custom_nodes/ComfyUI-gfx906-Tuning/tools/core_patch.py --apply
    python custom_nodes/ComfyUI-gfx906-Tuning/tools/core_patch.py --revert

Every target file is backed up once to <.core-patch-backup/> before it is
overwritten; --revert puts those back. Skipping the overlay is safe: the node
notices the hooks are absent and logs it, decode just runs without per-block
graph capture.
"""

import argparse
import difflib
import hashlib
import shutil
import sys
from pathlib import Path

OVERLAY = Path(__file__).resolve().parent.parent / "core-patches" / "overlay"
FILES = (
    "comfy/model_prefetch.py",
    "comfy/text_encoders/llama.py",
    "execution.py",
)
EXPECTED_VERSION = "0.39.0"
MARKER = ("comfy/model_prefetch.py", "RESIDENT_GRAPH_MODULES")


def root():
    # tools/ -> node dir -> custom_nodes -> ComfyUI root
    return Path(__file__).resolve().parents[3]


def backup_dir(r):
    return r / ".core-patch-backup"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def changed_lines(target, source):
    a = target.read_text(encoding="utf-8", errors="replace").splitlines()
    b = source.read_text(encoding="utf-8", errors="replace").splitlines()
    diff = list(difflib.unified_diff(a, b, n=0))
    return sum(1 for line in diff if line[:1] in "+-" and not line.startswith(("+++", "---")))


def check(r):
    ok = True
    pyproject = r / "pyproject.toml"
    if pyproject.exists():
        text = pyproject.read_text(encoding="utf-8", errors="replace")
        if f'version = "{EXPECTED_VERSION}"' not in text:
            print(f"warning: this overlay was made for ComfyUI {EXPECTED_VERSION}; pyproject.toml says otherwise")
            ok = False
    marker_file, marker_text = MARKER
    print(f"{'file':34s} {'state':10s} {'lines':>6s}")
    for rel in FILES:
        src = OVERLAY / rel
        dst = r / rel
        if not src.exists():
            print(f"{rel:34s} missing overlay copy; reinstall the node")
            ok = False
            continue
        if not dst.exists():
            print(f"{rel:34s} missing in this tree")
            ok = False
            continue
        if digest(dst) == digest(src):
            print(f"{rel:34s} applied          -")
        else:
            print(f"{rel:34s} not applied {changed_lines(dst, src):6d}")
        if (backup_dir(r) / rel).exists():
            print(f"{'':34s} backup present")
    if not ok:
        print("\ncheck failed; see core-patches/README.md")
    return ok


def apply(r):
    backup = backup_dir(r)
    for rel in FILES:
        src = OVERLAY / rel
        dst = r / rel
        if not src.exists():
            print(f"{rel}: missing overlay copy, skipped")
            continue
        if not dst.exists():
            print(f"{rel}: missing in this tree, skipped")
            continue
        if digest(dst) == digest(src):
            print(f"{rel}: already applied")
            continue
        keep = backup / rel
        keep.parent.mkdir(parents=True, exist_ok=True)
        if not keep.exists():
            shutil.copy2(dst, keep)
        shutil.copy2(src, dst)
        print(f"{rel}: applied ({changed_lines(keep, src)} lines changed, backup in {backup.relative_to(r)})")
    print("restart ComfyUI; the node logs 'gfx906: resident graph hooks absent' if this did not take")
    return True


def revert(r):
    backup = backup_dir(r)
    if not backup.exists():
        print(f"nothing to revert: no {backup.relative_to(r)}")
        return False
    for rel in FILES:
        keep = backup / rel
        dst = r / rel
        if not keep.exists():
            print(f"{rel}: no backup, skipped")
            continue
        shutil.copy2(keep, dst)
        print(f"{rel}: restored from backup")
    print("restart ComfyUI")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--check", action="store_true", help="report state, change nothing")
    action.add_argument("--apply", action="store_true", help="back up the current files and copy the overlay in")
    action.add_argument("--revert", action="store_true", help="restore the backed-up files")
    args = parser.parse_args()

    r = root()
    if not (r / "comfy").is_dir():
        print(f"cannot find a ComfyUI tree at {r}")
        return 1
    if args.check:
        return 0 if check(r) else 1
    if args.apply:
        return 0 if apply(r) else 1
    return 0 if revert(r) else 1


if __name__ == "__main__":
    sys.exit(main())
