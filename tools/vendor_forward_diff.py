"""Summarize upstream Qwen3.5 changes since the vendored fork's base.

    python custom_nodes/ComfyUI-GPU-Tuning/tools/vendor_forward_diff.py
    python .../vendor_forward_diff.py --repo /path/to/ComfyUI --ref origin/master
    python .../vendor_forward_diff.py --save-diff /tmp/qwen35_upstream.diff

The vendored encoder pins 0.39.0 compute classes but keeps the struct layer
(config/tokenizer/te factories) in sync with core. Upstream keeps changing
qwen35.py, so this tool answers two questions without eyeballing a 500-line
diff:

  1. did the stable boundary (te/tokenizer signatures, weight-key config) move?
     -- if so the fork needs a port or it will stop loading / injecting;
  2. which definitions changed, split by layer: STRUCT (port) vs COMPUTE/VISION
     (we own -- compare only, port a single function only if it is faster).

Only stdlib + git are needed; no torch import, no GPU.
"""

import argparse
import ast
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

NODE_DIR = Path(__file__).resolve().parent.parent
QWEN35_FILE = "comfy/text_encoders/qwen35.py"
DEFAULT_REPO = Path("/home/liu/项目/ComfyUI")

# Layer of each top-level definition in qwen35.py. STRUCT is the contract core
# calls or that decides weight keys -- changes here must be ported. COMPUTE is
# the decode math we pinned; VISION is the encode path (separate port decision).
STRUCT = {
    "_qwen35_layer_types", "Qwen35Config", "QWEN35_VISION_DEFAULTS",
    "QWEN35_MODELS", "_make_config", "Qwen35Tokenizer", "Qwen35ImageTokenizer",
    "Qwen35ClipModel", "Qwen35TEModel", "tokenizer", "te",
}
VISION = {
    "Qwen35VisionPatchEmbed", "Qwen35VisionMLP", "Qwen35VisionRotaryEmbedding",
    "Qwen35VisionAttention", "Qwen35VisionBlock", "Qwen35VisionPatchMerger",
    "Qwen35VisionModel",
}
# everything else in the file is COMPUTE (LinearKV, Gated*, Transformer, Qwen35)
# plus upstream-only additions (VerifyTree, tree_spec, best_widths, ...).
BOUNDARY = ("te", "tokenizer")  # signatures must match core's sd.py call
WEIGHT_KEYS = ("Qwen35Config", "QWEN35_MODELS", "QWEN35_VISION_DEFAULTS",
               "_make_config", "_qwen35_layer_types")


def git(repo, *args):
    out = subprocess.run(["git", "-C", str(repo), *args],
                         capture_output=True, text=True)
    if out.returncode:
        raise RuntimeError("git %s failed: %s" % (" ".join(args), out.stderr.strip()))
    return out.stdout


def defs_by_line(source):
    """Map top-level def/class name -> (start, end) line, plus signature text."""
    tree = ast.parse(source)
    out = {}
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            end = getattr(node, "end_lineno", node.lineno)
            sig = ""
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                all_args = list(node.args.posonlyargs) + list(node.args.args) + list(node.args.kwonlyargs)
                sig = "def %s(%s):" % (node.name, ", ".join(a.arg for a in all_args))
            out[node.name] = (node.lineno, end, sig)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    out[t.id] = (node.lineno, getattr(node, "end_lineno", node.lineno), "")
    return out


def layer_of(name):
    if name in STRUCT:
        return "STRUCT"
    if name in VISION:
        return "VISION"
    return "COMPUTE"


def parse_hunks(diff):
    """Yield (old_start, new_start, added_lines, removed_lines) per hunk (-U0)."""
    hunk = None
    for line in diff.splitlines():
        if line.startswith("@@"):
            m = re.match(r"@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@", line)
            if m:
                if hunk:
                    yield hunk
                hunk = [int(m.group(1)), int(m.group(2)), [], []]
        elif hunk is not None:
            if line.startswith("+") and not line.startswith("+++"):
                hunk[2].append(line[1:])
            elif line.startswith("-") and not line.startswith("---"):
                hunk[3].append(line[1:])
    if hunk:
        yield hunk


def enclosing(name_by_line, line, default):
    best, best_span = default, 0
    for name, (s, e, _) in name_by_line.items():
        if s <= line <= e:
            span = e - s
            if span <= best_span and best_span:
                continue
            best, best_span = name, span
    return best


def signature_params(sig):
    if not sig or "(" not in sig:
        return None
    try:
        node = ast.parse(sig + "\n    pass").body[0]
        args = list(node.args.posonlyargs) + list(node.args.args) + list(node.args.kwonlyargs)
        return [a.arg for a in args]
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", type=Path, default=DEFAULT_REPO,
                    help="ComfyUI git checkout (default: %(default)s)")
    ap.add_argument("--ref", default="origin/master",
                    help="upstream ref to compare against (default: %(default)s)")
    ap.add_argument("--base", default=None,
                    help="fork base tag (default: read vendor/qwen35/FORK_BASE)")
    ap.add_argument("--save-diff", type=Path, default=None,
                    help="also write the full raw diff to this path")
    args = ap.parse_args()

    repo = args.repo.resolve()
    if not (repo / ".git").exists():
        print("ERROR: not a git checkout: %s" % repo, file=sys.stderr)
        return 2

    base = args.base or (NODE_DIR / "vendor" / "qwen35" / "FORK_BASE").read_text().strip()
    ref = args.ref

    try:
        ref_sha = git(repo, "rev-parse", "--short", ref).strip()
        old_src = git(repo, "show", "%s:%s" % (base, QWEN35_FILE))
        new_src = git(repo, "show", "%s:%s" % (ref, QWEN35_FILE))
    except RuntimeError as e:
        print("ERROR: %s" % e, file=sys.stderr)
        return 2

    diff = git(repo, "diff", "-U0", "%s..%s" % (base, ref), "--", QWEN35_FILE)
    if args.save_diff:
        args.save_diff.write_text(diff)

    old_defs = defs_by_line(old_src)
    new_defs = defs_by_line(new_src)

    # aggregate per-definition add/remove counts, keyed by the NEW file's def
    changed = defaultdict(lambda: [0, 0])
    for old_start, new_start, added, removed in parse_hunks(diff):
        new_def = enclosing(new_defs, new_start, "<module>")
        old_def = enclosing(old_defs, old_start, "<module>")
        key = new_def if new_def != "<module>" else old_def
        changed[key][0] += len(added)
        changed[key][1] += len(removed)

    total_add = sum(v[0] for v in changed.values())
    total_del = sum(v[1] for v in changed.values())

    print("Vendored Qwen3.5 forward-diff")
    print("  fork base : %s" % base)
    print("  upstream  : %s @ %s" % (ref, ref_sha))
    print("  diff      : +%d / -%d lines across %d hunks" % (
        total_add, total_del, len(list(parse_hunks(diff)))))
    if args.save_diff:
        print("  raw diff  : %s" % args.save_diff)
    print()

    # 1) critical: does the vendored fork cover upstream's injection boundary?
    # Compare the VENDORED signature (which may already be ported) against
    # upstream's, not base-vs-upstream history. A ported kwarg is "covered".
    vendored_src_local = (NODE_DIR / "vendor" / "qwen35" / "model.py").read_text()
    vendored_defs = defs_by_line(vendored_src_local)
    print("Critical -- injection boundary & weight keys (port or the fork breaks):")
    critical = False
    for name in BOUNDARY:
        vp = signature_params(vendored_defs.get(name, (0, 0, ""))[2])
        np = signature_params(new_defs.get(name, (0, 0, ""))[2])
        if vp != np:
            missing = [a for a in (np or []) if a not in (vp or [])]
            critical = True
            print("  [!!] %s() vendored %s != upstream %s (port missing: %s)" % (
                name, vp, np, missing))
        else:
            print("  [ok] %s() vendored covers upstream: %s" % (name, vp))
    for name in WEIGHT_KEYS:
        if name in changed and changed[name][0] + changed[name][1]:
            critical = True
            print("  [!!] %s changed (+%d/-%d): may shift state_dict keys" % (
                name, changed[name][0], changed[name][1]))
    if not critical:
        print("  [ok] no boundary/weight-key changes detected")
    print()

    # 2) changed definitions grouped by layer
    groups = {"STRUCT": [], "VISION": [], "COMPUTE": [], "<module>": []}
    for name, (add, dele) in sorted(changed.items(), key=lambda kv: -(kv[1][0] + kv[1][1])):
        if add == 0 and dele == 0:
            continue
        groups.setdefault(layer_of(name) if name != "<module>" else "<module>", []).append(
            (name, add, dele))

    for layer in ("STRUCT", "VISION", "COMPUTE", "<module>"):
        items = groups.get(layer, [])
        if not items:
            continue
        hint = {
            "STRUCT": "port these -- core calls them or they set weight keys",
            "VISION": "encode path -- port only if image conditioning regresses",
            "COMPUTE": "we own these -- port a single function only if benchmarked faster",
            "<module>": "module-level (imports/constants) -- review manually",
        }[layer]
        print("%s (%d defs) -- %s:" % (layer, len(items), hint))
        for name, add, dele in items:
            print("  %-34s +%-4d -%-4d" % (name, add, dele))
        print()

    # 3) upstream commit subjects since base, for intent
    print("Upstream commits touching qwen35.py since %s:" % base)
    try:
        log = git(repo, "log", "--oneline", "%s..%s" % (base, ref), "--", QWEN35_FILE)
        for line in log.splitlines():
            print("  %s" % line)
    except RuntimeError:
        print("  (none)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
